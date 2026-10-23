#!/usr/bin/env python3
"""
High-Speed Multi-Threaded Checkpoint Downloader for Cognitive Radiology Report Generator.
Downloads and decrypts Mega.nz files using parallel chunked HTTP Range requests and AES-128-CTR.
Features:
- Parallel range requests with per-thread keep-alive connections
- Chunk-level resume persistence via progress JSON
- Atomic file writes and ZIP archive header verification
"""

import os
import sys
import json
import time
import base64
import struct
import requests
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
except ImportError:
    print("Error: 'cryptography' library is required. Run: pip install cryptography", file=sys.stderr)
    sys.exit(1)


DEFAULT_MEGA_URL = "https://mega.nz/file/azh1iAxK#9dlcjr4OUYtOixJXeKcaXTEy5hsvUw7CfVlXHrrYjjU"
DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB; AES block aligned
DEFAULT_WORKERS = 24
_thread_local = threading.local()


def _worker_session():
    """Return a reusable HTTP session owned by the current worker thread."""
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _thread_local.session = session
    return session


def parse_mega_url(url: str):
    if "#" not in url:
        raise ValueError(f"Invalid Mega URL (missing key fragment '#'): {url}")
    prefix, key = url.split("#", 1)
    handle = prefix.rstrip("/").split("/")[-1]
    return handle, key


def get_file_info(handle: str):
    api_url = "https://g.api.mega.co.nz/cs?id=0"
    payload = [{"a": "g", "g": 1, "p": handle}]
    resp = requests.post(api_url, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and len(data) > 0:
        item = data[0]
        if isinstance(item, int):
            raise RuntimeError(f"Mega API returned error code: {item}")
        return item
    raise RuntimeError(f"Unexpected Mega API response: {data}")


def decrypt_attributes(at_b64: str, key_bytes: bytes) -> dict:
    at_bytes = base64.urlsafe_b64decode(at_b64 + "=" * (-len(at_b64) % 4))
    cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(b"\x00" * 16))
    decryptor = cipher.decryptor()
    decrypted = decryptor.update(at_bytes) + decryptor.finalize()
    attr_str = decrypted.decode("utf-8", errors="ignore").rstrip("\x00")
    if attr_str.startswith("MEGA"):
        attr_str = attr_str[4:]
    return json.loads(attr_str)


def compute_counter_bytes(iv_pair: tuple, block_offset: int) -> bytes:
    high_64 = (iv_pair[0] << 32) + iv_pair[1]
    counter_int = (high_64 << 64) + block_offset
    return counter_int.to_bytes(16, byteorder="big")


def download_chunk(dl_url, start, end, key_bytes, iv_pair, file_handle, write_lock, max_retries=5):
    for attempt in range(max_retries):
        try:
            headers = {"Range": f"bytes={start}-{end}"}
            session = _worker_session()
            resp = session.get(dl_url, headers=headers, timeout=120)
            if resp.status_code != 206:
                raise IOError(f"HTTP {resp.status_code} on chunk {start}-{end}")
            encrypted_data = resp.content
            expected_len = end - start + 1
            if len(encrypted_data) != expected_len:
                raise IOError(f"Incomplete chunk data: got {len(encrypted_data)}, expected {expected_len}")

            # Decrypt with AES-CTR
            block_offset = start // 16
            initial_counter = compute_counter_bytes(iv_pair, block_offset)
            cipher = Cipher(algorithms.AES(key_bytes), modes.CTR(initial_counter))
            decryptor = cipher.decryptor()
            decrypted = decryptor.update(encrypted_data)

            # Thread-safe write to file at offset
            with write_lock:
                file_handle.seek(start)
                file_handle.write(decrypted)

            return len(decrypted)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(1 + attempt)


def download_checkpoint(
    url: str = DEFAULT_MEGA_URL,
    dest_dir: str = "checkpoints",
    workers: int = DEFAULT_WORKERS,
    force: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
):
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if chunk_size < 16 or chunk_size % 16:
        raise ValueError("chunk_size must be a positive multiple of 16 bytes")
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    handle, public_key = parse_mega_url(url)

    # Decode Mega 256-bit key
    raw_key = base64.urlsafe_b64decode(public_key + "=" * (-len(public_key) % 4))
    a32 = struct.unpack(f">{len(raw_key)//4}I", raw_key)
    k = (a32[0] ^ a32[4], a32[1] ^ a32[5], a32[2] ^ a32[6], a32[3] ^ a32[7])
    iv_pair = (a32[4], a32[5])
    key_bytes = struct.pack(">4I", *k)

    print(f"[*] Querying Mega API for handle: {handle}...")
    file_info = get_file_info(handle)
    total_size = file_info["s"]
    dl_url = file_info["g"]

    try:
        attrs = decrypt_attributes(file_info["at"], key_bytes)
        filename = attrs.get("n", "best_model.pth")
    except Exception:
        filename = "best_model.pth"

    final_file = dest_path / filename
    part_file = dest_path / f"{filename}.part"
    progress_file = dest_path / f"{filename}.progress.json"

    if final_file.exists() and not force:
        if final_file.stat().st_size == total_size:
            print(f"[+] Checkpoint verified: {final_file} ({total_size / (1024**3):.2f} GB). Skipping download.")
            return str(final_file)
        else:
            print(f"[!] Existing file size ({final_file.stat().st_size}) != expected ({total_size}). Re-downloading...")

    print(f"[+] Target File: {filename}")
    print(f"[+] Total Size: {total_size:,} bytes ({total_size / (1024**3):.2f} GB)")
    print(f"[+] Worker Threads: {workers} (Parallel Chunk Pipeline)")

    # Build chunk ranges
    chunks = []
    offset = 0
    while offset < total_size:
        chunk_end = min(offset + chunk_size - 1, total_size - 1)
        chunks.append((offset, chunk_end))
        offset = chunk_end + 1

    total_chunks = len(chunks)
    completed_chunks = set()

    # Load progress state if available
    if progress_file.exists() and part_file.exists() and not force:
        try:
            with open(progress_file, "r") as pf:
                saved = json.load(pf)
                if saved.get("chunk_size") == chunk_size and saved.get("total_size") == total_size:
                    completed_chunks = set(saved.get("completed_chunks", []))
                    print(f"[*] Resuming from previous session: {len(completed_chunks)}/{total_chunks} chunks already downloaded.")
                else:
                    print("[*] Download settings changed; restarting incomplete chunk map safely.")
                    completed_chunks = set()
        except Exception:
            completed_chunks = set()

    # Pre-allocate part file if needed
    if not part_file.exists() or part_file.stat().st_size != total_size or force:
        print("[*] Allocating sparse target file...")
        with open(part_file, "wb") as pf:
            pf.seek(total_size - 1)
            pf.write(b"\0")
        completed_chunks.clear()

    remaining_chunks = [c for c in chunks if c[0] not in completed_chunks]
    completed_bytes = sum((end - start + 1) for start, end in chunks if start in completed_chunks)

    if tqdm:
        pbar = tqdm(total=total_size, initial=completed_bytes, unit="B", unit_scale=True, unit_divisor=1024, desc=filename)
    else:
        pbar = None

    write_lock = threading.Lock()
    state_lock = threading.Lock()
    def save_state():
        with state_lock:
            try:
                with open(progress_file, "w") as pf:
                    json.dump({
                        "completed_chunks": list(completed_chunks),
                        "chunk_size": chunk_size,
                        "total_size": total_size,
                    }, pf)
            except Exception:
                pass

    last_save_time = [time.time()]

    with open(part_file, "r+b") as target_file:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_chunk = {
                executor.submit(
                    download_chunk,
                    dl_url,
                    start,
                    end,
                    key_bytes,
                    iv_pair,
                    target_file,
                    write_lock
                ): (start, end)
                for start, end in remaining_chunks
            }

            for future in as_completed(future_to_chunk):
                start, end = future_to_chunk[future]
                try:
                    bytes_written = future.result()
                    with state_lock:
                        completed_chunks.add(start)
                    if pbar:
                        pbar.update(bytes_written)

                    if time.time() - last_save_time[0] > 5:
                        save_state()
                        last_save_time[0] = time.time()
                except Exception as e:
                    print(f"\n[!] Chunk {start}-{end} failed: {e}")
                    raise

    if pbar:
        pbar.close()

    save_state()

    # Final validation
    if part_file.stat().st_size != total_size:
        raise IOError(f"Downloaded file size ({part_file.stat().st_size}) does not match expected ({total_size})")

    with open(part_file, "rb") as check_f:
        magic = check_f.read(4)
        if magic != b"PK\x03\x04":
            print(f"[!] Warning: Magic bytes {magic} do not match standard PyTorch ZIP archive PK\\x03\\x04.")
        else:
            print("[+] Verified PyTorch ZIP archive header integrity.")

    if final_file.exists():
        final_file.unlink()
    part_file.rename(final_file)

    if progress_file.exists():
        progress_file.unlink()

    print(f"[SUCCESS] Checkpoint successfully downloaded to: {final_file}")
    return str(final_file)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parallel Mega checkpoint downloader")
    parser.add_argument("--url", default=DEFAULT_MEGA_URL, help="Mega file URL")
    parser.add_argument("--dest-dir", default="checkpoints", help="Destination folder")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of concurrent worker threads")
    parser.add_argument("--chunk-mb", type=int, default=DEFAULT_CHUNK_SIZE // (1024 * 1024), help="Range request size in MiB (default: 8)")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    download_checkpoint(url=args.url, dest_dir=args.dest_dir, workers=args.workers, force=args.force, chunk_size=args.chunk_mb * 1024 * 1024)
