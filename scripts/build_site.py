"""Build the static GitHub Pages site into docs/ (served from main:/docs).

  python scripts/build_site.py --repo https://github.com/<user>/<repo> [--api https://your-app.onrender.com/]

Copies deadcloser/web, the benchmark results (results.json) and the replay transcripts (replay.json).
With no backend reachable the page plays back real engine transcripts; set --api to use a live deployment."""
from __future__ import annotations

import argparse
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="https://github.com/")
    ap.add_argument("--api", default="")
    a = ap.parse_args()
    out = os.path.join(ROOT, "docs")
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(os.path.join(ROOT, "deadcloser", "web"), out)
    res = os.path.join(ROOT, "bench", "results")
    bench = {}
    for name in ("simulation", "safety"):
        p = os.path.join(res, f"{name}.json")
        if os.path.exists(p):
            d = json.load(open(p, encoding="utf-8"))
            if name == "safety":
                d.pop("cases", None)
            if name == "simulation":
                d.pop("detail_seed1", None)
            bench[name] = d
    json.dump(bench, open(os.path.join(out, "results.json"), "w", encoding="utf-8"))
    shutil.copy(os.path.join(res, "replay.json"), os.path.join(out, "replay.json"))
    if os.path.exists(os.path.join(res, "demo.gif")):
        shutil.copy(os.path.join(res, "demo.gif"), os.path.join(out, "demo.gif"))
    with open(os.path.join(out, "config.js"), "w", encoding="utf-8") as fh:
        fh.write(f'window.DEADCLOSER_API = {json.dumps(a.api)};\nwindow.DEADCLOSER_REPO = {json.dumps(a.repo)};\n')
    open(os.path.join(out, ".nojekyll"), "w").close()
    print("built", out, os.listdir(out))


if __name__ == "__main__":
    main()
