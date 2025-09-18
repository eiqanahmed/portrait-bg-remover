# weights.py (root)
from __future__ import annotations
import os, sys, shutil, tarfile, zipfile, hashlib, tempfile, subprocess
from typing import Optional

# Defaults (override via env)
DEFAULT_URL = os.environ.get(
    "MODEL_URL",
    "https://github.com/eiqanahmed/image_background_remover/releases/download/weights-v1/best.keras.tar.gz",
)
DEFAULT_DIR = os.environ.get("MODEL_DIR", "files")
DEFAULT_FILENAME = os.environ.get("MODEL_FILENAME", "best.keras")
ARCHIVE_SHA256 = os.environ.get("MODEL_ARCHIVE_SHA256")  # optional
FILE_SHA256    = os.environ.get("MODEL_FILE_SHA256")     # optional
GITHUB_TOKEN   = os.environ.get("GITHUB_TOKEN")          # optional (for private releases)

CHUNK = 1024 * 1024  # 1 MiB


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _safe_extract_tar(tar: tarfile.TarFile, dest: str) -> None:
    base = os.path.abspath(dest)
    for m in tar.getmembers():
        p = os.path.abspath(os.path.join(dest, m.name))
        if not p.startswith(base + os.sep) and p != base:
            raise Exception("Blocked path traversal in tar file")
    tar.extractall(dest)


def _download_requests(url: str, out_path: str, headers: dict) -> None:
    import requests, certifi
    with requests.get(url, headers=headers, stream=True, timeout=300, verify=certifi.where()) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(CHUNK):
                if chunk:
                    f.write(chunk)


def _download_shell(url: str, out_path: str, headers: dict) -> None:
    # Prefer curl, then wget
    hargs = []
    for k, v in headers.items():
        hargs += ["-H", f"{k}: {v}"]
    curl = _which("curl")
    if curl:
        subprocess.run([curl, "-L", *hargs, "-o", out_path, url], check=True)
        return
    wget = _which("wget")
    if wget:
        wargs = []
        for k, v in headers.items():
            wargs += ["--header", f"{k}: {v}"]
        subprocess.run([wget, "-O", out_path, *wargs, url], check=True)
        return
    raise RuntimeError("No downloader available (curl/wget missing), and requests failed.")


def _download(url: str, out_path: str, headers: dict) -> None:
    try:
        _download_requests(url, out_path, headers)
    except Exception as e:
        print(f"[weights] requests download failed: {e}\n[weights] Trying shell downloader…", file=sys.stderr)
        _download_shell(url, out_path, headers)


def ensure_model(
    url: str = DEFAULT_URL,
    model_dir: str = DEFAULT_DIR,
    filename: str = DEFAULT_FILENAME,
) -> str:
    """
    Ensure `model_dir/filename` exists. If missing, download from `url` and extract if needed.
    Supports .tar.gz, .tgz, .zip, or raw file. Optional SHA-256 checks via env vars.
    """
    os.makedirs(model_dir, exist_ok=True)
    target = os.path.join(model_dir, filename)
    if os.path.exists(target):
        if FILE_SHA256:
            got = _sha256(target)
            if got.lower() != FILE_SHA256.lower():
                raise RuntimeError(f"Checksum mismatch for {target}. got={got} expected={FILE_SHA256}")
        return target

    lower_url = url.split("?")[0].lower()
    is_tar = lower_url.endswith(".tar.gz") or lower_url.endswith(".tgz")
    is_zip = lower_url.endswith(".zip")

    fd, tmp_path = tempfile.mkstemp(suffix=".download")
    os.close(fd)

    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    print(f"[weights] Downloading: {url}")
    try:
        _download(url, tmp_path, headers)
        if ARCHIVE_SHA256:
            got = _sha256(tmp_path)
            if got.lower() != ARCHIVE_SHA256.lower():
                raise RuntimeError(f"Archive checksum mismatch. got={got} expected={ARCHIVE_SHA256}")

        if is_tar:
            with tarfile.open(tmp_path, "r:gz") as tar:
                _safe_extract_tar(tar, model_dir)
        elif is_zip:
            with zipfile.ZipFile(tmp_path, "r") as zf:
                zf.extractall(model_dir)
        else:
            shutil.move(tmp_path, target)
            tmp_path = None

        if not os.path.exists(target):
            candidate = os.path.join(model_dir, os.path.basename(filename))
            if os.path.exists(candidate):
                target = candidate
            else:
                raise FileNotFoundError(f"After extraction, '{filename}' not found under {model_dir}")

        if FILE_SHA256:
            got = _sha256(target)
            if got.lower() != FILE_SHA256.lower():
                raise RuntimeError(f"File checksum mismatch. got={got} expected={FILE_SHA256}")

        print("[weights] Ready:", target)
        return target
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
