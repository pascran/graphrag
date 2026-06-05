"""Unit tests for app.utils.hashing."""
from __future__ import annotations

import hashlib
from pathlib import Path

from app.utils.hashing import sha256_bytes, sha256_file


def test_sha256_bytes_matches_stdlib():
    payload = b"graphrag test payload"
    assert sha256_bytes(payload) == hashlib.sha256(payload).hexdigest()


def test_sha256_bytes_empty_input():
    expected = hashlib.sha256(b"").hexdigest()
    assert sha256_bytes(b"") == expected


def test_sha256_file_roundtrip(tmp_path: Path):
    payload = b"abc" * 4096
    f = tmp_path / "blob.bin"
    f.write_bytes(payload)
    assert sha256_file(f) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_handles_large_multiblock_input(tmp_path: Path):
    payload = b"X" * (1024 * 1024 * 3 + 17)
    f = tmp_path / "big.bin"
    f.write_bytes(payload)
    assert sha256_file(f, chunk_size=1024 * 1024) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_accepts_str_and_path(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_bytes(b"hello")
    by_path = sha256_file(f)
    by_str = sha256_file(str(f))
    assert by_path == by_str == hashlib.sha256(b"hello").hexdigest()
