#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def run_cmd(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def check_url(url: str, timeout_sec: float = 12.0) -> dict[str, Any]:
    req = Request(url, method="HEAD")
    try:
        with urlopen(req, timeout=timeout_sec) as resp:
            return {"url": url, "ok": True, "http_status": getattr(resp, "status", 200), "error": None}
    except HTTPError as exc:
        host = (urlparse(url).netloc or "").lower()
        if exc.code in {401, 403} and (
            "doi.org" in host
            or "mdpi.com" in host
            or "nature.com" in host
            or "pubmed.ncbi.nlm.nih.gov" in host
            or "pmc.ncbi.nlm.nih.gov" in host
        ):
            return {
                "url": url,
                "ok": True,
                "http_status": exc.code,
                "error": f"restricted({exc.code})",
            }
        # Some providers block HEAD and require GET.
        if exc.code in {400, 401, 403, 405}:
            try:
                with urlopen(url, timeout=timeout_sec) as resp:
                    return {"url": url, "ok": True, "http_status": getattr(resp, "status", 200), "error": None}
            except Exception as exc2:  # pylint: disable=broad-except
                return {"url": url, "ok": False, "http_status": None, "error": str(exc2)}
        return {"url": url, "ok": False, "http_status": exc.code, "error": str(exc)}
    except URLError as exc:
        return {"url": url, "ok": False, "http_status": None, "error": str(exc)}


def get_git_metadata(path: Path) -> dict[str, Any]:
    if not (path / ".git").exists():
        return {
            "is_git_repo": False,
            "origin": None,
            "head_sha": None,
            "head_sha_short": None,
            "branch": None,
            "default_branch": None,
        }

    _, origin_out, _ = run_cmd(["git", "remote", "get-url", "origin"], cwd=path)
    _, sha_out, _ = run_cmd(["git", "rev-parse", "HEAD"], cwd=path)
    _, sha_short_out, _ = run_cmd(["git", "rev-parse", "--short", "HEAD"], cwd=path)
    _, branch_out, _ = run_cmd(["git", "branch", "--show-current"], cwd=path)
    _, head_ref_out, _ = run_cmd(["git", "symbolic-ref", "refs/remotes/origin/HEAD"], cwd=path)
    default_branch = None
    if head_ref_out and "/" in head_ref_out:
        default_branch = head_ref_out.rsplit("/", 1)[-1]

    return {
        "is_git_repo": True,
        "origin": origin_out or None,
        "head_sha": sha_out or None,
        "head_sha_short": sha_short_out or None,
        "branch": branch_out or None,
        "default_branch": default_branch,
    }


def clone_or_update_repo(repo_url: str, dest: Path, update_existing: bool) -> dict[str, Any]:
    if dest.exists() and (dest / ".git").exists():
        if not update_existing:
            return {
                "status": "already_present",
                "action": "none",
                "stderr": None,
            }
        code_fetch, _, err_fetch = run_cmd(["git", "fetch", "--all", "--tags", "--prune"], cwd=dest)
        code_pull, _, err_pull = run_cmd(["git", "pull", "--ff-only"], cwd=dest)
        return {
            "status": "updated" if code_fetch == 0 and code_pull == 0 else "update_failed",
            "action": "fetch_pull",
            "stderr": "\n".join([x for x in [err_fetch, err_pull] if x]) or None,
        }

    dest.parent.mkdir(parents=True, exist_ok=True)
    code_clone, _, err_clone = run_cmd(["git", "clone", "--depth", "1", repo_url, str(dest)])
    return {
        "status": "cloned" if code_clone == 0 else "clone_failed",
        "action": "clone",
        "stderr": err_clone or None,
    }


def load_registry(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "entries" not in data or not isinstance(data["entries"], list):
        raise ValueError(f"Invalid registry format at {path}: missing entries[]")
    return data


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        "total_entries": len(results),
        "repo_present": 0,
        "repo_cloned": 0,
        "repo_updated": 0,
        "repo_clone_failed": 0,
        "repo_update_failed": 0,
        "repo_missing_url": 0,
        "docs_links_total": 0,
        "docs_links_ok": 0,
        "docs_links_failed": 0,
    }
    for item in results:
        repo = item["repo"]
        if repo["repo_url"] is None:
            counts["repo_missing_url"] += 1
        else:
            counts["repo_present"] += 1
            if repo["acquisition"]["status"] == "cloned":
                counts["repo_cloned"] += 1
            elif repo["acquisition"]["status"] == "updated":
                counts["repo_updated"] += 1
            elif repo["acquisition"]["status"] == "clone_failed":
                counts["repo_clone_failed"] += 1
            elif repo["acquisition"]["status"] == "update_failed":
                counts["repo_update_failed"] += 1

        docs_checks = item["docs"]["checks"]
        counts["docs_links_total"] += len(docs_checks)
        counts["docs_links_ok"] += sum(1 for c in docs_checks if c["ok"])
        counts["docs_links_failed"] += sum(1 for c in docs_checks if not c["ok"])
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire and verify external SOTA model source repositories.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--registry", default="sota_model_source_registry.json")
    parser.add_argument(
        "--output",
        default="data/public_benchmarks/acquisition/sota_source_acquisition_status.json",
    )
    parser.add_argument(
        "--dated-output",
        default="",
        help="Optional dated status artifact path, e.g. SOTA_SOURCE_REPRO_STATUS_2026-03-05.json",
    )
    parser.add_argument("--update-existing", action="store_true")
    parser.add_argument("--skip-url-checks", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    registry_path = (repo_root / args.registry).resolve()
    output_path = (repo_root / args.output).resolve()

    registry = load_registry(registry_path)
    source_root = (repo_root / registry.get("source_root", "data/public_benchmarks/sources")).resolve()

    results: list[dict[str, Any]] = []
    for entry in registry["entries"]:
        entry_id = entry["id"]
        repo_url = entry.get("repo_url")
        local_subdir = entry.get("local_subdir")

        repo_payload: dict[str, Any] = {
            "repo_url": repo_url,
            "local_path": None,
            "acquisition": {
                "status": "not_applicable",
                "action": "none",
                "stderr": None,
            },
            "git": {
                "is_git_repo": False,
                "origin": None,
                "head_sha": None,
                "head_sha_short": None,
                "branch": None,
                "default_branch": None,
            },
        }

        if repo_url and local_subdir:
            local_path = source_root / local_subdir
            repo_payload["local_path"] = str(local_path)
            repo_payload["acquisition"] = clone_or_update_repo(repo_url, local_path, args.update_existing)
            repo_payload["git"] = get_git_metadata(local_path)

        urls = []
        urls.extend(entry.get("docs_urls", []))
        urls.extend(entry.get("model_artifact_urls", []))
        checks = [] if args.skip_url_checks else [check_url(url) for url in urls]

        results.append(
            {
                "id": entry_id,
                "task": entry.get("task"),
                "model_name": entry.get("model_name"),
                "role": entry.get("role"),
                "required_for_primary_claim": bool(entry.get("required_for_primary_claim", False)),
                "paper_url": entry.get("paper_url"),
                "repo": repo_payload,
                "docs": {
                    "urls": urls,
                    "checks": checks,
                },
                "reproduction_entrypoint_hint": entry.get("reproduction_entrypoint_hint"),
            }
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "registry_path": str(registry_path),
        "source_root": str(source_root),
        "update_existing": bool(args.update_existing),
        "skip_url_checks": bool(args.skip_url_checks),
        "summary": summarize(results),
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.dated_output:
        dated_path = (repo_root / args.dated_output).resolve()
        dated_path.parent.mkdir(parents=True, exist_ok=True)
        dated_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote: {output_path}")
    if args.dated_output:
        print(f"wrote: {dated_path}")


if __name__ == "__main__":
    main()
