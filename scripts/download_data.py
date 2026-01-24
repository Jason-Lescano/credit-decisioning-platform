import hashlib
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, timeout: int = 60) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0"))

        tmp = dest.with_suffix(dest.suffix + ".part")
        downloaded = 0

        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

        tmp.replace(dest)

        if total > 0:
            pct = (downloaded / total) * 100
            print(f"Downloaded: {downloaded}/{total} bytes ({pct:.2f}%) -> {dest}")
        else:
            print(f"Downloaded: {downloaded} bytes -> {dest}")


def main() -> int:
    load_dotenv()

    url = os.getenv("LENDINGCLUB_URL", "").strip()
    expected_sha = os.getenv("LENDINGCLUB_SHA256", "").strip().lower()
    data_dir = Path(os.getenv("DATA_DIR", "data"))

    if not url:
        print(
            "ERROR: Missing LENDINGCLUB_URL.\n"
            "Set it in your .env file, e.g.\n"
            "LENDINGCLUB_URL=https://.../lendingclub.zip",
            file=sys.stderr,
        )
        return 2

    filename = Path(urlparse(url).path).name or "lendingclub_download"
    dest = data_dir / "raw" / filename

    if dest.exists():
        print(f"File already exists: {dest}")
        if expected_sha:
            actual = sha256_file(dest)
            if actual != expected_sha:
                print(
                    f"ERROR: SHA256 mismatch for existing file.\n"
                    f"Expected: {expected_sha}\n"
                    f"Actual:   {actual}\n"
                    "Delete the file and re-run download.",
                    file=sys.stderr,
                )
                return 3
            print("SHA256 OK (existing file).")
        return 0

    try:
        download_file(url, dest)
    except requests.RequestException as e:
        print(f"ERROR: Download failed: {e}", file=sys.stderr)
        return 4

    if expected_sha:
        actual = sha256_file(dest)
        if actual != expected_sha:
            print(
                f"ERROR: SHA256 mismatch after download.\n"
                f"Expected: {expected_sha}\n"
                f"Actual:   {actual}",
                file=sys.stderr,
            )
            return 5
        print("SHA256 OK (downloaded file).")
    else:
        print("WARNING: LENDINGCLUB_SHA256 not set. Skipping checksum validation.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
