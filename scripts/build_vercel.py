"""Assemble a Vercel project (website + serverless agent API) in deploy/vercel/.

  python scripts/build_vercel.py --repo https://github.com/<user>/<repo>
  cd deploy/vercel && npx vercel deploy --prod --yes --token $VERCEL_TOKEN

Static site -> public/ ; FastAPI app -> api/index.py (Vercel's FastAPI preset routes every non-static path to it).
All state lives in Postgres (DATABASE_URL), so the function is stateless and can scale / restart freely."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "deploy", "vercel")

REQUIREMENTS = """fastapi>=0.110
pydantic>=2.6
anthropic>=0.40
httpx>=0.27
langgraph>=0.2
langgraph-checkpoint-postgres>=2.0
psycopg[binary]>=3.1
psycopg_pool>=3.2
"""

INDEX = '''import os

os.environ.setdefault("DEADCLOSER_SERVERLESS", "1")

from deadcloser.api import create_app  # noqa: E402

app = create_app()
'''

VERCEL_JSON = {
    "functions": {"api/index.py": {"maxDuration": 60, "memory": 1024, "includeFiles": "{deadcloser/web/**,bench/results/*.json}"}},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="https://github.com/")
    a = ap.parse_args()
    keep = os.path.join(OUT, ".vercel")                      # preserve the project link between builds
    saved = os.path.join(ROOT, "deploy", ".vercel_link")
    shutil.rmtree(saved, ignore_errors=True)
    if os.path.isdir(keep):
        shutil.copytree(keep, saved)
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(os.path.join(OUT, "api"))
    if os.path.isdir(saved):
        shutil.copytree(saved, keep)
    shutil.copytree(os.path.join(ROOT, "deadcloser"), os.path.join(OUT, "deadcloser"), ignore=shutil.ignore_patterns("__pycache__"))
    os.makedirs(os.path.join(OUT, "bench", "results"))
    for f in ("simulation.json", "safety.json"):
        src = os.path.join(ROOT, "bench", "results", f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(OUT, "bench", "results", f))
    open(os.path.join(OUT, "bench", "__init__.py"), "w").close()
    open(os.path.join(OUT, "api", "index.py"), "w", encoding="utf-8").write(INDEX)
    open(os.path.join(OUT, "requirements.txt"), "w", encoding="utf-8").write(REQUIREMENTS)
    open(os.path.join(OUT, ".python-version"), "w").write("3.12\n")
    json.dump(VERCEL_JSON, open(os.path.join(OUT, "vercel.json"), "w"), indent=2)
    # static site into public/ (API is same-origin, so config.js leaves the API base empty)
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "build_site.py"), "--repo", a.repo], check=True, stdout=subprocess.DEVNULL)
    shutil.copytree(os.path.join(ROOT, "docs"), os.path.join(OUT, "public"))
    print("assembled", OUT)


if __name__ == "__main__":
    main()
