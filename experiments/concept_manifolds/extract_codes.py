"""Extract per-concept intrinsic coordinates from a trained BSF, for gamfit.

gam's manifold-SAE adjudicator wants each atom's [n, 2] *intrinsic latent
coordinates*. A BSF concept already has them: every firing token carries a signed
code z_g in the concept's own group_size-dim subspace, so the intrinsic
coordinates are the top-2 PCA of that code cloud -- no manifold fit needed to get
them, unlike a flat SAE where the atom is a single direction.

Unlike the dashboard artifact (which keeps top-activating firings only), this
takes a UNIFORM sample of each concept's firings: the earlier audit showed the
top of the distribution is unrepresentative, so a shape verdict must not be built
on it.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

from bsf import capture_format as cf, normalize
from bsf.group_lasso import GroupLassoBSF

ROOT = '/home/nymph/Code/vllm-uv/captures/pile25m'
GGUF = '/home/nymph/Models/Qwen3.6-27B-UD-Q4_K_XL.gguf'


def load_vocab():
    from gguf import GGUFReader
    f = GGUFReader(GGUF).fields['tokenizer.ggml.tokens']
    return [str(bytes(f.parts[i]), 'utf-8', errors='replace') for i in f.data]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--layer', type=int, default=32)
    ap.add_argument('--hook', default='post_block')
    ap.add_argument('--n-groups', type=int, default=4096)
    ap.add_argument('--group-size', type=int, default=4)
    ap.add_argument('--requests', type=int, default=400)
    ap.add_argument('--per-concept', type=int, default=800,
                    help='max firings kept per concept (uniform over firings)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    mean, scale = normalize.load_or_compute(ROOT, args.layer, args.hook)
    mt = torch.as_tensor(mean, dtype=torch.float32, device=args.device)
    model = GroupLassoBSF(5120, args.n_groups, args.group_size)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    model.to(args.device).eval()

    units = cf.discover_units(ROOT, args.layer, args.hook)[:args.requests]
    G, k = args.n_groups, args.group_size
    codes = [[] for _ in range(G)]     # each: list of (k,) float32
    toks = [[] for _ in range(G)]      # each: list of token id
    l0_sum, n_tok = 0.0, 0
    vocab = load_vocab()
    ids_cache = {}

    def unit_ids(ui, path):
        if ui not in ids_cache:
            ids_cache[ui] = json.loads(pathlib.Path(path).read_text()).get(
                'prompt_token_ids', [])
        return ids_cache[ui]

    for ui, u in enumerate(units):
        arr = u.read(args.layer, args.hook)
        if arr.shape[0] == 0:
            continue
        x = torch.from_numpy(np.array(arr)).view(torch.bfloat16).float().to(args.device)
        with torch.no_grad():
            z = model.encode((x - mt) * scale)          # (n, G, k)
            act = z.norm(dim=-1)
        fired = act > 1e-6
        l0_sum += float(fired.sum())
        n_tok += act.shape[0]
        nz = torch.nonzero(fired, as_tuple=False)        # (nf, 2) [pos, concept]
        if not nz.numel():
            continue
        pos_np = nz[:, 0].cpu().numpy()
        con_np = nz[:, 1].cpu().numpy()
        zsel = z[nz[:, 0], nz[:, 1], :].cpu().numpy().astype(np.float32)
        tid = unit_ids(ui, u.path)
        for p, c, zz in zip(pos_np, con_np, zsel):
            c = int(c)
            if len(codes[c]) >= args.per_concept:
                continue
            codes[c].append(zz)
            toks[c].append(tid[p] if p < len(tid) else -1)
        if (ui + 1) % 50 == 0:
            print(f'  [{ui+1}/{len(units)}] units', flush=True)

    mean_l0 = l0_sum / max(n_tok, 1)
    counts = np.array([len(c) for c in codes], dtype=np.int32)
    print(f'{n_tok:,} tokens | mean L0 {mean_l0:.2f} | '
          f'concepts with >=200 firings: {int((counts >= 200).sum())}/{G}')

    flat = np.concatenate([np.stack(c) for c in codes if c]).astype(np.float32)
    tflat = np.concatenate([np.array(t, dtype=np.int32) for t in toks if t])
    offs = np.zeros(G + 1, dtype=np.int64)
    offs[1:] = np.cumsum(counts)
    np.savez_compressed(args.out, codes=flat, token_ids=tflat, offsets=offs,
                        counts=counts, mean_l0=np.array(mean_l0),
                        vocab=np.array(vocab, dtype=object),
                        meta=np.array(json.dumps(dict(
                            layer=args.layer, hook=args.hook, n_groups=G,
                            group_size=k, n_tokens=n_tok, requests=len(units)))))
    print('->', args.out)


if __name__ == '__main__':
    main()
