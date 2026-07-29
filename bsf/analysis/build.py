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

import heapq
import json
import pathlib

import numpy as np
import torch

from .. import capture_format as cf
from .. import normalize
from .types import (
    Analysis, Meta, ConceptExample, ARTIFACT_VERSION,
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


def _model_from_checkpoint(state, kind, d, n_groups, group_size):
    """Instantiate the right featurizer and check its geometry against the ckpt."""
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
    model = cls(d, n_groups, group_size)
    model.load_state_dict(state)
    return model


class _TopK:
    """Bounded min-heap of the strongest (activation, unit, position) triples."""

    def __init__(self, k):
        self.k = int(k)
        self.h = []

    def add(self, act, unit, pos):
        item = (float(act), int(unit), int(pos))
        if len(self.h) < self.k:
            heapq.heappush(self.h, item)
        elif item[0] > self.h[0][0]:
            heapq.heapreplace(self.h, item)

    def sorted(self):
        return sorted(self.h, key=lambda t: -t[0])


def build_analysis(ckpt, root, layer, hook, *, n_groups, group_size,
                   model_kind='group_lasso', requests=300, top_k=12,
                   n_neighbors=8, manifold_points=200, context=4,
                   embedding_method='graph', coact_threshold=0.1,
                   gguf=None, device=None, seed=0, progress=None):
    """Compute a complete ``Analysis``. Heavy: needs torch, a GPU is advisable."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    mean, scale = normalize.load_or_compute(root, layer, hook)
    d = int(np.asarray(mean).shape[0])
    state = torch.load(ckpt, map_location='cpu')
    model = _model_from_checkpoint(state, model_kind, d, n_groups, group_size)
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
    tops = [_TopK(top_k) for _ in range(G)]
    gates = []           # per-unit boolean (n, G), kept on CPU
    clouds = [[] for _ in range(G)]   # (contribution, unit, pos) subsamples
    n_tokens = 0
    rng = np.random.default_rng(seed)

    atoms = model.atoms().to(device)                     # (G, k, d)

    for ui, unit in enumerate(units):
        arr = unit.read(layer, hook)
        if arr.shape[0] == 0:
            continue
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

        # Top activations per concept. One batched top-k on the GPU replaces a
        # per-concept argsort (which made this loop O(G) numpy sorts per unit and
        # dominated the whole analysis); then a concept is skipped entirely unless
        # this unit could actually improve its heap or it still needs cloud points.
        kk = min(int(top_k), act.shape[0])
        tv, ti = torch.topk(act, kk, dim=0)              # (kk, G)
        tv_np = tv.cpu().numpy()
        ti_np = ti.cpu().numpy()
        unit_max = tv_np[0]                              # (G,) per-concept max here
        for g in np.nonzero(unit_max > 1e-6)[0]:
            g = int(g)
            heap = tops[g]
            improves = (len(heap.h) < heap.k) or (unit_max[g] > heap.h[0][0])
            if improves:
                for r in range(kk):
                    v = tv_np[r, g]
                    if v <= 1e-6:
                        break
                    heap.add(v, ui, int(ti_np[r, g]))
            need = manifold_points - len(clouds[g])
            if need > 0:
                take = ti_np[:min(kk, need), g]
                take = take[tv_np[:take.size, g] > 1e-6]
                if take.size:
                    contrib = (z[take, g, :] @ atoms[g]).cpu().numpy()
                    for t, c in zip(take, contrib):
                        clouds[g].append((c, ui, int(t)))
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

    examples = []
    for g in range(G):
        per = []
        for act, ui, pos in tops[g].sorted():
            ids = unit_token_ids(ui)
            lo, hi = max(0, pos - context), min(len(ids), pos + context + 1)
            before = ''.join(token_str(ui, p) for p in range(lo, pos))
            after = ''.join(token_str(ui, p) for p in range(pos + 1, hi))
            per.append(ConceptExample(round(float(act), 4), int(pos),
                                      token_str(ui, pos), before, after))
        examples.append(per)

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
    for g in range(G):
        pts = clouds[g]
        if len(pts) < 4:
            continue
        c = np.stack([p[0] for p in pts]).astype(np.float64)
        m, comps = pca_fit(c, 3)
        proj = (c - m) @ comps.T
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
        vocab=tok_table,
        examples=examples,
    ).validate()


__all__ = ['build_analysis', 'load_gguf_vocab', 'clean_token']
