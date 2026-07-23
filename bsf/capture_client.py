"""Async client that drives running vLLM servers to capture activations.

The vLLM filesystem capture consumer has no dedicated endpoint -- capture rides
on the normal generation endpoints. So this streams a prompt file to one or more
already-running servers' ``/v1/completions`` (launched with
``--capture-consumers filesystem:root=...``), attaching a top-level ``capture``
field that opts each request into the filesystem consumer. Requests are
round-robined across ``servers`` with bounded concurrency.

The filesystem ``root`` is server-side config, not a per-request field; this
client only sets ``tag`` / ``hooks`` / ``positions`` / ``layout``. The training
loader is later pointed at that shared root.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
from dataclasses import dataclass, field

import httpx


def read_prompts(path):
    """One prompt per line. Lines beginning with ``{`` are parsed as JSON and the
    ``"prompt"`` (or ``"text"``) field is taken; everything else is raw text."""
    out = []
    for line in pathlib.Path(path).read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith('{'):
            obj = json.loads(s)
            out.append(obj.get('prompt', obj.get('text', '')))
        else:
            out.append(s)
    return out


def build_capture_spec(request_id, tag, hooks, positions, layout):
    """The per-request ``capture.filesystem`` payload (a FilesystemCaptureRequest)."""
    return {'filesystem': {
        'request_id': request_id, 'tag': tag, 'hooks': hooks,
        'positions': positions, 'layout': layout,
    }}


@dataclass
class CaptureResult:
    index: int
    request_id: str
    status: str                 # 'ok' | 'error'
    server: str
    paths: list = field(default_factory=list)
    error: str | None = None


async def _one(client, sem, server, model, prompt, *, index, tag, hooks, positions,
               layout, max_tokens, capture_wait, retries):
    request_id = f'{tag}-{index:08d}'
    body = {
        'model': model, 'prompt': prompt, 'max_tokens': max_tokens,
        'temperature': 0.0,
        'capture': build_capture_spec(request_id, tag, hooks, positions, layout),
        'capture_wait': capture_wait,
    }
    url = server.rstrip('/') + '/v1/completions'
    async with sem:
        last = None
        for attempt in range(retries + 1):
            try:
                resp = await client.post(url, json=body)
                if resp.status_code != 200:
                    last = f'HTTP {resp.status_code}: {resp.text[:200]}'
                    continue
                data = resp.json()
                fs = (data.get('capture_results') or {}).get('filesystem') or {}
                status = fs.get('status', 'ok' if not capture_wait else 'pending')
                paths = fs.get('payload') or []
                if status in ('error', 'partial_error'):
                    last = fs.get('error') or status
                    continue
                return CaptureResult(index, request_id, 'ok', server, paths)
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                last = f'{type(exc).__name__}: {exc}'
        return CaptureResult(index, request_id, 'error', server, error=last)


async def capture(servers, model, prompts, *, tag, hooks, positions='all_prompt',
                  layout='packed', max_tokens=1, concurrency=16, capture_wait=True,
                  retries=2, timeout=120.0, progress=None):
    """Drive ``servers`` to capture activations for every prompt. Returns a list
    of ``CaptureResult`` in prompt order. ``progress`` is an optional callback
    ``(done, total, result)`` invoked as each request completes."""
    sem = asyncio.Semaphore(concurrency)
    results = [None] * len(prompts)
    done = 0
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            asyncio.ensure_future(_one(
                client, sem, servers[i % len(servers)], model, p, index=i, tag=tag,
                hooks=hooks, positions=positions, layout=layout, max_tokens=max_tokens,
                capture_wait=capture_wait, retries=retries))
            for i, p in enumerate(prompts)
        ]
        for coro in asyncio.as_completed(tasks):
            r = await coro
            results[r.index] = r
            done += 1
            if progress is not None:
                progress(done, len(prompts), r)
    return results
