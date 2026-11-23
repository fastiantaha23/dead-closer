"""Deploy the agent API to a Hugging Face Space (Docker). Secrets are read from YOUR environment and set as Space
secrets - they are never written to a file or committed.

  pip install huggingface_hub
  set HF_TOKEN=hf_...            (a token with *write* access)
  set DATABASE_URL=postgresql://...      (Neon / Supabase)
  set ANTHROPIC_API_KEY=sk-ant-...
  set DEADCLOSER_ADMIN_KEY=<long random string>
  python scripts/deploy_hf.py --space your-username/dead-closer
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONT_MATTER = """---
title: Dead Closer Agent API
emoji: 🏠
colorFrom: orange
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="owner/name of the Space to create or update")
    a = ap.parse_args()
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Set HF_TOKEN (write access) first.")
        return 1
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(a.space, repo_type="space", space_sdk="docker", exist_ok=True)
    for key in ("DATABASE_URL", "ANTHROPIC_API_KEY", "DEADCLOSER_ADMIN_KEY"):
        if os.getenv(key):
            api.add_space_secret(a.space, key, os.environ[key])
            print("secret set:", key)
        else:
            print("WARNING: not set:", key)
    for key, val in (("DEADCLOSER_MODEL", "claude-haiku-5-5"), ("CHAT_TENANT", os.getenv("CHAT_TENANT", "demo_realty")),
                     ("CORS_ORIGINS", os.getenv("CORS_ORIGINS", "*")), ("DEADCLOSER_DEMO", "1")):
        api.add_space_variable(a.space, key, val)
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    api.upload_file(path_or_fileobj=(FRONT_MATTER + readme).encode("utf-8"), path_in_repo="README.md", repo_id=a.space, repo_type="space")
    api.upload_folder(folder_path=ROOT, repo_id=a.space, repo_type="space",
                      allow_patterns=["Dockerfile", "requirements.txt", "deadcloser/**", "bench/__init__.py", "bench/results/*.json"],
                      ignore_patterns=["**/__pycache__/**", "*.pyc"])
    user, name = a.space.split("/")
    print(f"\nDeployed. Build logs: https://huggingface.co/spaces/{a.space}")
    print(f"API base URL (after the build finishes): https://{user.lower()}-{name.lower().replace('_', '-')}.hf.space")
    print("Check: <base>/health  ->  ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
