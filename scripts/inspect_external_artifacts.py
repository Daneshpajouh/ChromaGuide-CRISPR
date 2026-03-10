#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, UTC
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


REPOS = {
    "CRISPR_HNN": "https://github.com/xx0220/CRISPR_HNN.git",
    "CRISPR_FMC": "https://github.com/xx0220/CRISPR-FMC.git",
    "DeepHF": "https://github.com/izhangcd/DeepHF.git",
    "CCLMoff": "https://github.com/duwa2/CCLMoff.git",
    "crispAI": "https://github.com/furkanozdenn/crispr-offtarget-uncertainty.git",
}

FIGSHARE_ARTICLES = {
    "CCLMoff": 27080566,
}

ZENODO_RECORDS = {
    "crispAI_data": 12609337,
    "crispAI_model": 13335960,
}


def run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    return proc.returncode, proc.stdout, proc.stderr


def fetch_json(url: str) -> dict:
    req = Request(url, headers={"User-Agent": "codex-artifact-inspector/1.0"})
    with urlopen(req, timeout=60) as r:
        return json.load(r)


def fetch_bytes(url: str, headers: dict[str, str] | None = None) -> bytes:
    req = Request(
        url,
        headers={"User-Agent": "codex-artifact-inspector/1.0", **(headers or {})},
    )
    with urlopen(req, timeout=120) as r:
        return r.read()


def split_image_ref(image: str) -> tuple[str, str]:
    if ":" in image.rsplit("/", 1)[-1]:
        repo, ref = image.rsplit(":", 1)
    else:
        repo, ref = image, "latest"
    return repo, ref


def docker_registry_token(repo: str, ref: str, action: str = "pull") -> str:
    probe = Request(f"https://registry-1.docker.io/v2/{repo}/manifests/{ref}")
    try:
        urlopen(probe, timeout=60)
        return ""
    except HTTPError as exc:
        auth = exc.headers.get("WWW-Authenticate", "")
        realm = re.search(r'realm="([^"]+)"', auth)
        service = re.search(r'service="([^"]+)"', auth)
        scope = re.search(r'scope="([^"]+)"', auth)
        if not (realm and service and scope):
            raise
        token_payload = fetch_json(
            f"{realm.group(1)}?service={service.group(1)}&scope={scope.group(1)}"
        )
        return token_payload["token"]


def ensure_repo(name: str, url: str, dest: Path) -> dict:
    info: dict[str, object] = {"name": name, "url": url, "path": str(dest)}
    if not dest.exists():
        code, out, err = run(["git", "clone", "--depth", "1", url, str(dest)])
        info["clone_exit_code"] = code
        info["clone_stdout"] = out[-2000:]
        info["clone_stderr"] = err[-2000:]
        if code != 0:
            info["status"] = "clone_failed"
            return info
    else:
        code, out, err = run(["git", "pull", "--ff-only"], cwd=dest)
        info["pull_exit_code"] = code
        info["pull_stdout"] = out[-2000:]
        info["pull_stderr"] = err[-2000:]

    code, out, err = run(["git", "rev-parse", "HEAD"], cwd=dest)
    info["head"] = out.strip() if code == 0 else None
    code, out, err = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=dest)
    info["branch"] = out.strip() if code == 0 else None

    interesting: list[str] = []
    for p in sorted(dest.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(dest).as_posix()
        lower = rel.lower()
        if any(
            token in lower
            for token in [
                "readme",
                "require",
                "environment",
                "env/",
                "conda",
                "data/",
                "dataset",
                "preprocess",
                "train",
                "infer",
                "predict",
                "split",
                "fold",
                ".ckpt",
                ".pth",
                ".pt",
                ".h5",
                ".hd5",
                ".pkl",
                ".csv",
                ".tsv",
                ".json",
                ".yml",
                ".yaml",
            ]
        ):
            interesting.append(rel)
    info["interesting_files"] = interesting[:400]
    info["signals"] = {
        "has_requirements": any("requirements" in p.lower() for p in interesting),
        "has_env": any(any(tok in p.lower() for tok in ["environment", "env/", "conda", ".yml", ".yaml"]) for p in interesting),
        "has_data": any("data/" in p.lower() or "dataset" in p.lower() for p in interesting),
        "has_preprocess": any("preprocess" in p.lower() for p in interesting),
        "has_split_logic": any("split" in p.lower() or "fold" in p.lower() for p in interesting),
        "has_train_code": any("train" in p.lower() for p in interesting),
        "has_checkpoint_like": any(any(ext in p.lower() for ext in [".ckpt", ".pth", ".pt", ".h5", ".hd5"]) for p in interesting),
    }
    info["status"] = "ok"
    return info


def inspect_figshare(article_id: int) -> dict:
    data = fetch_json(f"https://api.figshare.com/v2/articles/{article_id}")
    files = []
    for f in data.get("files", []):
        files.append(
            {
                "id": f.get("id"),
                "name": f.get("name"),
                "size": f.get("size"),
                "supplied_md5": f.get("supplied_md5"),
                "computed_md5": f.get("computed_md5"),
                "download_url": f.get("download_url"),
            }
        )
    return {
        "id": data.get("id"),
        "title": data.get("title"),
        "doi": data.get("doi"),
        "published_date": data.get("published_date"),
        "url_public_html": data.get("url_public_html"),
        "files": files,
    }


def inspect_zenodo(record_id: int) -> dict:
    data = fetch_json(f"https://zenodo.org/api/records/{record_id}")
    files = []
    for f in data.get("files", []):
        files.append(
            {
                "key": f.get("key"),
                "size": f.get("size"),
                "checksum": f.get("checksum"),
                "type": f.get("type"),
                "links": f.get("links", {}),
            }
        )
    metadata = data.get("metadata", {})
    return {
        "id": data.get("id"),
        "doi": metadata.get("doi") or data.get("doi"),
        "title": metadata.get("title"),
        "publication_date": metadata.get("publication_date"),
        "files": files,
    }


def inspect_docker(image: str) -> dict:
    registry_repo, registry_ref = split_image_ref(image)
    image_ref = f"https://registry-1.docker.io/v2/{registry_repo}"
    token = docker_registry_token(registry_repo, registry_ref)
    auth_headers = {"Authorization": f"Bearer {token}"} if token else {}
    registry_payload: dict[str, object] = {
        "registry_repository": registry_repo,
        "registry_reference": registry_ref,
    }
    try:
        manifest_bytes = fetch_bytes(f"{image_ref}/manifests/{registry_ref}", headers=auth_headers)
        manifest = json.loads(manifest_bytes)
        registry_payload["registry_manifest"] = {
            "schemaVersion": manifest.get("schemaVersion"),
            "mediaType": manifest.get("mediaType"),
            "config": manifest.get("config"),
            "layers": manifest.get("layers", []),
        }

        config_digest = manifest.get("config", {}).get("digest")
        if config_digest:
            config = json.loads(fetch_bytes(f"{image_ref}/blobs/{config_digest}", headers=auth_headers))
            registry_payload["registry_config"] = {
                "created": config.get("created"),
                "architecture": config.get("architecture"),
                "os": config.get("os"),
                "working_dir": config.get("config", {}).get("WorkingDir"),
                "entrypoint": config.get("config", {}).get("Entrypoint"),
                "cmd": config.get("config", {}).get("Cmd"),
            }

        workspace_files: list[dict[str, object]] = []
        for index, layer in enumerate(manifest.get("layers", []), start=1):
            layer_digest = layer.get("digest")
            if not layer_digest:
                continue
            blob = fetch_bytes(f"{image_ref}/blobs/{layer_digest}", headers=auth_headers)
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tf:
                for member in tf.getmembers():
                    name = member.name
                    if name == "workspace" or name.startswith("workspace/"):
                        workspace_files.append(
                            {
                                "layer": index,
                                "digest": layer_digest,
                                "name": name,
                                "size": member.size,
                            }
                        )
        registry_payload["workspace_files"] = workspace_files
    except Exception as exc:  # noqa: BLE001
        registry_payload["registry_error"] = f"{type(exc).__name__}: {exc}"

    docker = shutil.which("docker")
    if not docker:
        return {"status": "docker_not_installed", **registry_payload}
    code, out, err = run([docker, "manifest", "inspect", image])
    if code != 0:
        return {
            "status": "manifest_inspect_failed",
            "stderr": err[-4000:],
            **registry_payload,
        }
    try:
        payload = json.loads(out)
    except json.JSONDecodeError:
        return {
            "status": "manifest_inspect_unparseable",
            "stdout": out[-4000:],
            **registry_payload,
        }
    manifests = payload.get("manifests", []) if isinstance(payload, dict) else []
    return {
        "status": "ok",
        "schemaVersion": payload.get("schemaVersion") if isinstance(payload, dict) else None,
        "mediaType": payload.get("mediaType") if isinstance(payload, dict) else None,
        "manifest_count": len(manifests),
        "manifests": manifests[:10],
        **registry_payload,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_json = Path(args.output_json).resolve()
    sources_dir = repo_root / "data" / "external_artifact_inspection" / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "repo_root": str(repo_root),
        "repos": {},
        "figshare": {},
        "zenodo": {},
        "docker": {},
    }

    for name, url in REPOS.items():
        payload["repos"][name] = ensure_repo(name, url, sources_dir / name)

    for name, article_id in FIGSHARE_ARTICLES.items():
        try:
            payload["figshare"][name] = inspect_figshare(article_id)
        except Exception as exc:  # noqa: BLE001
            payload["figshare"][name] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    for name, record_id in ZENODO_RECORDS.items():
        try:
            payload["zenodo"][name] = inspect_zenodo(record_id)
        except Exception as exc:  # noqa: BLE001
            payload["zenodo"][name] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    payload["docker"]["CCLMoff"] = inspect_docker("weiandu/cclmoff:gpu")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output_json)


if __name__ == "__main__":
    sys.exit(main())
