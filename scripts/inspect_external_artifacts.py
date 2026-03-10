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
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np


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


def rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    vals = a[order]
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and vals[j] == vals[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j
    return ranks


def spearman_np(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a).reshape(-1)
    bb = np.asarray(b).reshape(-1)
    m = min(len(aa), len(bb))
    ra = rankdata(aa[:m])
    rb = rankdata(bb[:m])
    return float(np.corrcoef(ra, rb)[0, 1])


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


def inspect_crispai_downloads(download_dir: Path) -> dict:
    data_gz = download_dir / "crispAI_result_reproduction.gz"
    model_zip = download_dir / "crispAI_model.zip"
    result: dict[str, object] = {
        "download_dir": str(download_dir),
        "data_bundle_present": data_gz.exists(),
        "model_bundle_present": model_zip.exists(),
    }

    if model_zip.exists():
        with zipfile.ZipFile(model_zip) as zf:
            names = zf.namelist()
        result["model_bundle"] = {
            "path": str(model_zip),
            "size": model_zip.stat().st_size,
            "key_files": [
                n
                for n in names
                if any(
                    pat in n
                    for pat in [
                        "crispAI_score/crispAI.py",
                        "crispAI_score/model.py",
                        "crispAI_score/model_checkpoint/epoch:19-best_valid_loss:0.270.pt",
                        "env/crispAI_env.yml",
                        "env/R_env.csv",
                    ]
                )
            ],
        }

    if not data_gz.exists():
        result["status"] = "data_bundle_missing"
        return result

    key_members = [
        "crispAI_result_reproduction/source_data/changeseq_offtarget_data_flank73_filtered_nupop_gc_bdm_preprocessed_train.csv",
        "crispAI_result_reproduction/source_data/changeseq_offtarget_data_flank73_filtered_nupop_gc_bdm_preprocessed_test.csv",
        "crispAI_result_reproduction/checkpoint/epoch:19-best_valid_loss:0.270.pt",
        "crispAI_result_reproduction/Supplementary_tables/Supp_tab_7.py",
        "crispAI_result_reproduction/Supplementary_tables/Supp_tabs_6_8_9_10_11_12_13_14.py",
        "crispAI_result_reproduction/Fig4/changeseqtest_preds_median_scores_all.npy",
        "crispAI_result_reproduction/Fig4/y_test.npy",
    ]
    with tarfile.open(data_gz, "r:gz") as tf:
        names = tf.getnames()
        names_set = set(names)
        present = {name: name in names_set for name in key_members}

        supp_tab_7_text = ""
        supp_tabs_text = ""
        if present[key_members[3]]:
            supp_tab_7_text = tf.extractfile(key_members[3]).read().decode("utf-8", "replace")
        if present[key_members[4]]:
            supp_tabs_text = tf.extractfile(key_members[4]).read().decode("utf-8", "replace")

        array_metrics: dict[str, object] = {}
        if present[key_members[5]] and present[key_members[6]]:
            preds = np.load(io.BytesIO(tf.extractfile(key_members[5]).read()), allow_pickle=True)
            y_test = np.load(io.BytesIO(tf.extractfile(key_members[6]).read()), allow_pickle=True)
            array_metrics = {
                "changeseqtest_preds_median_scores_all_shape": list(preds.shape),
                "y_test_shape": list(y_test.shape),
                "saved_array_spearman": spearman_np(preds, y_test),
            }

    split_like = [
        n
        for n in names
        if any(tok in n.lower() for tok in ["split", "validation", "valid", "fold"])
    ]

    result["data_bundle"] = {
        "path": str(data_gz),
        "size": data_gz.stat().st_size,
        "key_members_present": present,
        "split_like_members": split_like[:50],
        "signals": {
            "has_explicit_train_csv": present[key_members[0]],
            "has_explicit_test_csv": present[key_members[1]],
            "has_best_checkpoint": present[key_members[2]],
            "has_test_eval_script": present[key_members[3]] or present[key_members[4]],
            "has_saved_test_arrays": present[key_members[5]] and present[key_members[6]],
            "has_explicit_split_manifest": any("split" in n.lower() for n in split_like),
        },
        "supplementary_script_evidence": {
            "supp_tab_7_uses_test_csv": "preprocessed_test.csv" in supp_tab_7_text,
            "supp_tab_7_uses_adjusted_target": "CHANGEseq_reads_adjusted" in supp_tab_7_text,
            "supp_tabs_use_test_csv": "preprocessed_test.csv" in supp_tabs_text,
            "supp_tabs_use_best_checkpoint": "epoch:19-best_valid_loss:0.270.pt" in supp_tabs_text,
            "supp_tabs_compute_spearman": "spearmanr(" in supp_tabs_text,
        },
        "saved_array_metrics": array_metrics,
        "artifact_level_verdict": "artifacts_present_for_exact_test_frame_execution"
        if present[key_members[1]] and present[key_members[2]] and (present[key_members[3]] or present[key_members[4]])
        else "parity_artifacts_incomplete",
    }
    result["status"] = "ok"
    return result


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
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "repos": {},
        "figshare": {},
        "zenodo": {},
        "docker": {},
        "local_bundles": {},
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
    payload["local_bundles"]["crispAI"] = inspect_crispai_downloads(
        repo_root / "data" / "external_artifact_inspection" / "downloads" / "crispAI"
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output_json)


if __name__ == "__main__":
    sys.exit(main())
