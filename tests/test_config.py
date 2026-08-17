import pytest

from src.core.config import cache_paths_for


def test_cache_paths_for_deterministic(tmp_path):
    f = tmp_path / "a.pdf"
    f.write_bytes(b"hello world")

    first = cache_paths_for(str(f))
    second = cache_paths_for(str(f))

    assert first["key"] == second["key"]
    assert first["collection_name"] == second["collection_name"]


def test_cache_paths_for_differs_by_content(tmp_path):
    f1 = tmp_path / "a.pdf"
    f1.write_bytes(b"content one")
    f2 = tmp_path / "b.pdf"
    f2.write_bytes(b"content two")

    assert cache_paths_for(str(f1))["key"] != cache_paths_for(str(f2))["key"]


def test_cache_paths_for_same_content_different_filename_shares_cache(tmp_path):
    f1 = tmp_path / "a.pdf"
    f1.write_bytes(b"identical bytes")
    f2 = tmp_path / "b.pdf"
    f2.write_bytes(b"identical bytes")

    assert cache_paths_for(str(f1))["key"] == cache_paths_for(str(f2))["key"]


def test_cache_paths_for_missing_file_raises(tmp_path):
    missing = tmp_path / "nope.pdf"

    with pytest.raises(FileNotFoundError):
        cache_paths_for(str(missing))
