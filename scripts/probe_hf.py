"""Prints what Hugging Face actually hosts for the models in the catalog (run in CI;
the dev container can't reach huggingface.co)."""
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foresight.models import CATALOG  # noqa: E402

c = httpx.Client(follow_redirects=True, timeout=30)
for q in ("embeddinggemma-2", "embeddinggemma 2 gguf", "gemma-4-E4B-it-GGUF"):
    r = c.get("https://huggingface.co/api/models", params={"search": q, "limit": 20, "sort": "downloads"})
    print(f"\n# search {q!r}: {r.status_code}")
    for m in r.json() if r.status_code == 200 else []:
        print(f"  {m['id']}  downloads={m.get('downloads')} gated={m.get('gated')}")
for spec in CATALOG.values():
    r = c.get(f"https://huggingface.co/api/models/{spec.repo}/tree/main")
    print(f"\n# {spec.repo}: {r.status_code}")
    if r.status_code == 200:
        for f in r.json():
            size = (f.get("lfs") or {}).get("size") or f.get("size")
            print(f"  {f['path']}  {size}")
    else:
        print("  ", r.text[:200])
