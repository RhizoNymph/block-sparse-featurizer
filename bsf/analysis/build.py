"""Drive a full concept analysis: checkpoint + capture root -> ``Analysis``.

One streaming pass over a slice of the capture tree collects, per concept:
  - firing statistics (rate, mean/max activation)
  - top-activating token occurrences with decoded context
  - a subsample of its firing cloud, PCA'd to 3D (the concept manifold)
  - the boolean gate matrix, from which co-activation is derived

Token alignment: captures taken with ``positions='all_prompt'`` store a
request's rows in the same order as its ``prompt_token_ids`` sidecar field, so
row *i* is token *i*. Token strings are decoded from the GGUF vocabulary when
available, so the artifact needs no tokenizer at view time.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch

from .. import capture_format as cf
from .. import normalize
from .types import (
    Analysis, Meta, ConceptExample, ConceptBand, ARTIFACT_VERSION,
    AnalysisError, CheckpointMismatchError,
)
from . import compute as C


def load_gguf_vocab(path):
    """Decoded token strings from a GGUF tokenizer, or ``None`` if unavailable."""
    if not path:
        return None
    try:
        from gguf import GGUFReader
    except ImportError:
        return None
    try:
        f = GGUFReader(path).fields['tokenizer.ggml.tokens']
    except (KeyError, OSError, ValueError) as exc:
        raise AnalysisError(f'could not read tokenizer vocab from {path}: {exc}')
    return [str(bytes(f.parts[i]), 'utf-8', errors='replace') for i in f.data]


def clean_token(tok):
    """GPT-2-style byte markers -> readable text."""
    return tok.replace('Ġ', ' ').replace('Ċ', '\n').replace('ĉ', '\t')


def _model_from_checkpoint(state, kind, d, n_groups, group_size, l0=None):
    """Instantiate the right featurizer and check its geometry against the ckpt.

    ``l0`` is required for the TopK-gated featurizers (`vanilla`,
    `grassmannian`): it is architecture-defining but is NOT stored in the state
    dict, so a wrong value silently analyses the model at the wrong sparsity.
    `group_lasso` ignores it -- its threshold lives in the checkpoint.
    """
    from ..group_lasso import GroupLassoBSF
    from ..grassmannian import GrassmannianBSF
    from ..vanilla import VanillaBSF

    key = 'W_dec' if 'W_dec' in state else 'B_raw'
    if key == 'W_dec':
        rows, ckpt_d = state['W_dec'].shape
        if ckpt_d != d:
            raise CheckpointMismatchError('d', ckpt_d, d)
        if rows != n_groups * group_size:
            raise CheckpointMismatchError('n_groups*group_size', rows,
                                          n_groups * group_size)
    cls = {'group_lasso': GroupLassoBSF, 'grassmannian': GrassmannianBSF,
           'vanilla': VanillaBSF}.get(kind)
    if cls is None:
        raise AnalysisError(f'unknown model kind {kind!r}')
    kw = {}
    if cls in (VanillaBSF, GrassmannianBSF):
        if l0 is None:
            raise AnalysisError(
                f'model kind {kind!r} gates by block TopK, so --l0 is required: '
                f'it is not recoverable from the checkpoint and defaulting it '
                f'would analyse the model at the wrong sparsity')
        kw['l0'] = int(l0)
    if cls is VanillaBSF and 'dead_tracker.tokens_since_fired' in state:
        # Trained with the revival loss, so the checkpoint carries the tracker
        # buffer. Enable it here only so the buffer exists to load into -- the
        # analysis never calls `loss`, so the value is irrelevant to the output.
        kw['revival_alpha'] = 1.0
    model = cls(d, n_groups, group_size, **kw)
    model.load_state_dict(state)
    return model


def _cloud_to_3d(Z, A):
    """(n, k) codes + (k, d) decoder block -> (n, 3) PCA of the d-dim cloud.

    The contributions are ``c_i = z_i A``, so every point lies in the row space of
    ``A``. Writing ``A = U S Vt`` (rows of ``Vt`` orthonormal in R^d), the
    coordinates of ``c_i`` in that orthonormal basis are ``y_i = z_i (U S)`` -- and
    since ``Vt`` is an isometry, PCA of ``y`` in R^k is *identical* to PCA of ``c``
    in R^d. So the d-dim cloud never has to be materialised.
    """
    from ..viz import pca_fit
    U, S, _ = np.linalg.svd(A, full_matrices=False)       # A: (k, d) -> U (k,k)
    Y = Z @ (U * S)                                       # (n, k), isometric to c
    mean, comps = pca_fit(Y, 3)
    return (Y - mean) @ comps.T


def build_analysis(ckpt, root, layer, hook, *, n_groups, group_size,
                   model_kind='group_lasso', l0=None, requests=300, top_k=12,
                   n_neighbors=8, manifold_points=200, context=4,
                   band_examples=4, cloud_per_unit=8,
                   embedding_method='graph', coact_threshold=0.1,
                   gguf=None, device=None, seed=0, drop_first=0, progress=None):
    """Compute a complete ``Analysis``. Heavy: needs torch, a GPU is advisable.

    ``drop_first`` must match what the checkpoint was TRAINED with: it selects
    the norm statistics (a different filter is a different distribution, and the
    stats are cached under a different key) and skips the same leading positions
    here. Recorded positions stay absolute, so token lookups remain aligned with
    the ``prompt_token_ids`` sidecar.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    drop_first = int(drop_first)
    if drop_first < 0:
        raise AnalysisError(f'drop_first must be >= 0, got {drop_first}')
    mean, scale = normalize.load_or_compute(root, layer, hook,
                                            drop_first=drop_first)
    d = int(np.asarray(mean).shape[0])
    state = torch.load(ckpt, map_location='cpu')
    model = _model_from_checkpoint(state, model_kind, d, n_groups, group_size,
                                   l0=l0)
    model.to(device).eval()

    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
    units = cf.discover_units(root, layer, hook)[:requests]
    if not units:
        raise AnalysisError(f'no capture units for (layer={layer}, hook={hook!r}) '
                            f'under {root}')

    G, k = int(n_groups), int(group_size)
    fire_count = torch.zeros(G, dtype=torch.float64, device=device)
    act_sum = torch.zeros(G, dtype=torch.float64, device=device)
    act_max = torch.zeros(G, dtype=torch.float32, device=device)
    gates = []           # per-unit boolean (n, G), kept on CPU
    clouds = [[] for _ in range(G)]   # (contribution, unit, pos) subsamples
    # every firing, flat: parallel arrays of (concept, activation, unit, position)
    fire_con, fire_act, fire_unit, fire_pos = [], [], [], []
    n_tokens = 0

    atoms = model.atoms().to(device)                     # (G, k, d)

    for ui, unit in enumerate(units):
        arr = unit.read(layer, hook)
        if arr.shape[0] <= drop_first:
            continue
        arr = arr[drop_first:]
        x = torch.from_numpy(np.array(arr))
        x = (x.view(torch.bfloat16).float() if arr.dtype == np.uint16
             else x.to(torch.float32))
        x = ((x.to(device) - mean_t) * scale)
        with torch.no_grad():
            z = model.encode(x)                          # (n, G, k)
            act = z.norm(dim=-1)                         # (n, G)
        fired = act > 1e-6
        fire_count += fired.sum(0).to(torch.float64)
        act_sum += (act * fired).sum(0).to(torch.float64)
        act_max = torch.maximum(act_max, act.amax(0))
        gates.append(fired.cpu().numpy())
        n_tokens += act.shape[0]

        # Record EVERY firing, fully vectorised. With L0 ~ 9 this is ~n*9 rows per
        # unit (~1.5M over 300 units, ~24MB), which is cheap and buys exact
        # per-concept activation quantiles -- needed because top-k examples alone
        # are the extreme tail of a concept's distribution and misrepresent it.
        # It also removes the per-concept Python loop that used to dominate.
        nz = torch.nonzero(fired, as_tuple=False)         # (nf, 2) [pos, concept]
        if nz.numel():
            # +drop_first: positions stay ABSOLUTE so token_str/make_example
            # index prompt_token_ids correctly.
            fire_pos.append((nz[:, 0] + drop_first).to(torch.int32).cpu().numpy())
            fire_con.append(nz[:, 1].to(torch.int32).cpu().numpy())
            fire_act.append(act[fired].to(torch.float32).cpu().numpy())
            fire_unit.append(np.full(nz.shape[0], ui, dtype=np.int32))

        # Manifold clouds: store the (k,) CODE, never the (d,) contribution.
        #
        # Every contribution of concept g is z_g @ atoms_g, so the whole cloud lives
        # in that concept's own k-dim subspace. Keeping k=4 floats instead of
        # d=5120 is a ~1280x memory saving (15GB -> 13MB at 300 units) and needs no
        # per-concept matmul here -- the isometry into 3D is applied once at the
        # end via the SVD of atoms_g (see _cloud_to_3d).
        need = np.array([manifold_points - len(c) for c in clouds], dtype=np.int64)
        if (need > 0).any():
            cap = min(int(cloud_per_unit), act.shape[0])
            tv, ti = torch.topk(act, cap, dim=0)          # (cap, G)
            tv_np, ti_np = tv.cpu().numpy(), ti.cpu().numpy()
            z_cpu = None
            for g in np.nonzero((tv_np[0] > 1e-6) & (need > 0))[0]:
                g = int(g)
                take = ti_np[:min(cap, int(need[g])), g]
                take = take[tv_np[:take.size, g] > 1e-6]
                if not take.size:
                    continue
                if z_cpu is None:
                    z_cpu = z.cpu().numpy()               # one transfer per unit
                for t in take:
                    # absolute position, as for fire_pos
                    clouds[g].append((z_cpu[t, g, :].copy(), ui,
                                      int(t) + drop_first))
        if progress is not None:
            progress(ui + 1, len(units))

    # ---- geometry over the decoder subspaces
    #
    # The chordal NEIGHBOUR LIST is kept (its tail finds near-duplicate concepts),
    # but the 2D MAP is NOT built from this metric: 4096 4-planes in R^5120 are
    # forced near-orthogonal, so 2D captures under 1% of its variance. The map is
    # a spectral layout of the strong co-activation graph instead. Both facts are
    # recorded in Meta so the UI can state them.
    W_dec = model.decoder_atoms().detach().cpu()
    D = C.chordal_distances(W_dec, G, k)
    neighbor_idx, neighbor_dist = C.nearest_neighbors(D, n=n_neighbors)
    chordal_2d, chordal_dims = C.embedding_quality(D.numpy(), seed=seed)

    gate_all = torch.from_numpy(np.concatenate(gates, axis=0))
    coact_idx, coact_jac = C.coactivation(gate_all, n=n_neighbors)
    edges, edge_w, edges_found = C.coactivation_graph(
        gate_all, threshold=coact_threshold)
    if embedding_method == 'graph':
        embedding = C.graph_layout(edges, edge_w, G, seed=seed)
    else:
        embedding = C.embed_distances(D.numpy(), method=embedding_method, seed=seed)

    # ---- token strings + per-concept examples / manifolds
    vocab = load_gguf_vocab(gguf)
    ids_cache = {}

    def unit_token_ids(ui):
        if ui not in ids_cache:
            try:
                idx = json.loads(pathlib.Path(units[ui].path).read_text())
                ids_cache[ui] = idx.get('prompt_token_ids', [])
            except (OSError, ValueError):
                ids_cache[ui] = []
        return ids_cache[ui]

    def token_str(ui, pos):
        ids = unit_token_ids(ui)
        if vocab is None or pos >= len(ids):
            return f'<{pos}>'
        return clean_token(vocab[ids[pos]])

    def make_example(act_v, ui, pos):
        ids = unit_token_ids(ui)
        lo, hi = max(0, pos - context), min(len(ids), pos + context + 1)
        before = ''.join(token_str(ui, p) for p in range(lo, pos))
        after = ''.join(token_str(ui, p) for p in range(pos + 1, hi))
        return ConceptExample(round(float(act_v), 4), int(pos),
                              token_str(ui, pos), before, after)

    # ---- group every firing by concept, then cut activation bands
    #
    # Bands are rank windows into each concept's firings sorted strongest-first:
    # 'top' is the usual top-k, while p75/p50/p10 sample progressively closer to
    # threshold. A concept that is only coherent in 'top' is a concept whose
    # typical activity means nothing -- which the bands make visible.
    if fire_con:
        f_con = np.concatenate(fire_con)
        f_act = np.concatenate(fire_act)
        f_unit = np.concatenate(fire_unit)
        f_pos = np.concatenate(fire_pos)
    else:
        f_con = f_act = f_unit = f_pos = np.zeros(0, dtype=np.int32)

    # sort by (concept asc, activation desc) so each concept is a contiguous run
    order = np.lexsort((-f_act, f_con))
    f_con, f_act, f_unit, f_pos = (a[order] for a in (f_con, f_act, f_unit, f_pos))
    starts = np.searchsorted(f_con, np.arange(G), side='left')
    ends = np.searchsorted(f_con, np.arange(G), side='right')

    BANDS = (('top', 0.0), ('p75', 0.25), ('p50', 0.5), ('p10', 0.9))
    per_band = max(1, int(band_examples))
    examples, bands = [], []
    act_q = np.zeros((G, 5), dtype=np.float32)
    near_frac = np.zeros(G, dtype=np.float32)

    for g in range(G):
        lo, hi = int(starts[g]), int(ends[g])
        n = hi - lo
        if n == 0:
            examples.append([])
            bands.append([])
            continue
        a = f_act[lo:hi]                                  # descending
        act_q[g] = [a[-1], a[int(0.75 * (n - 1))], a[int(0.5 * (n - 1))],
                    a[int(0.25 * (n - 1))], a[0]]
        near_frac[g] = float((a < 0.4 * a[0]).mean())
        per_concept = []
        for label, frac in BANDS:
            s = min(int(frac * n), max(n - 1, 0))
            e = min(s + per_band, n)
            exs = [make_example(a[i], int(f_unit[lo + i]), int(f_pos[lo + i]))
                   for i in range(s, e)]
            per_concept.append(ConceptBand(
                label=label, lo_act=round(float(a[e - 1]), 4),
                hi_act=round(float(a[s]), 4), rank_lo=s, rank_hi=e, examples=exs))
        bands.append(per_concept)
        # the token view keeps the usual top-k list
        k_top = min(int(top_k), n)
        examples.append([make_example(a[i], int(f_unit[lo + i]), int(f_pos[lo + i]))
                         for i in range(k_top)])

    # ---- concept manifolds: PCA each firing cloud to 3D
    from ..viz import pca_fit
    P = int(manifold_points)
    man_xyz = np.zeros((G, P, 3), dtype=np.float32)
    man_tok = np.zeros((G, P), dtype=np.int32)
    tok_table, tok_index = [], {}

    def intern(s):
        if s not in tok_index:
            tok_index[s] = len(tok_table)
            tok_table.append(s)
        return tok_index[s]

    intern('')
    atoms_cpu = atoms.detach().cpu().numpy().astype(np.float64)   # (G, k, d)
    for g in range(G):
        pts = clouds[g]
        if len(pts) < 4:
            continue
        Z = np.stack([p[0] for p in pts]).astype(np.float64)      # (n, k) codes
        proj = _cloud_to_3d(Z, atoms_cpu[g])
        n = min(len(pts), P)
        man_xyz[g, :n] = proj[:n].astype(np.float32)
        for i in range(n):
            man_tok[g, i] = intern(token_str(pts[i][1], pts[i][2]))

    meta = Meta(version=ARTIFACT_VERSION, layer=int(layer), hook=str(hook), d=d,
                n_groups=G, group_size=k, n_tokens=int(n_tokens),
                n_units=len(units), checkpoint=str(ckpt), model_kind=model_kind,
                embedding_method=embedding_method, seed=int(seed),
                chordal_2d_variance=float(chordal_2d),
                chordal_dims_for_half=int(chordal_dims),
                coact_threshold=float(coact_threshold),
                coact_edges_found=int(edges_found))
    denom = max(n_tokens, 1)
    return Analysis(
        meta=meta,
        embedding=embedding,
        edges=edges,
        edge_weight=edge_w,
        neighbor_idx=neighbor_idx,
        neighbor_dist=neighbor_dist.astype(np.float32),
        coact_idx=coact_idx,
        coact_jaccard=coact_jac,
        fire_rate=(fire_count / denom).cpu().numpy().astype(np.float32),
        mean_act=(act_sum / fire_count.clamp_min(1)).cpu().numpy().astype(np.float32),
        max_act=act_max.cpu().numpy().astype(np.float32),
        manifold_xyz=man_xyz,
        manifold_token_ids=man_tok,
        act_quantiles=act_q,
        near_threshold_frac=near_frac,
        vocab=tok_table,
        examples=examples,
        bands=bands,
    ).validate()


__all__ = ['build_analysis', 'load_gguf_vocab', 'clean_token']
