"""Local filesystem storage backend (development / single-host deployments)."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from filestore.base import ObjectNotFoundError, StorageError

logger = logging.getLogger(__name__)


class LocalStorage:
    """Stores objects as files under a root directory."""

    name = "local"

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        if not key or key.startswith(("/", "\\")) or ".." in Path(key).parts:
            raise StorageError(f"Invalid storage key: {key!r}")
        path = (self.root / key).resolve()
        if self.root not in path.parents and path != self.root:
            raise StorageError(f"Storage key escapes root: {key!r}")
        return path

    def put_file(self, src: Path, key: str, move: bool = False) -> None:
        dest = self._resolve(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        try:
            if move:
                shutil.move(str(src), str(tmp))
            else:
                shutil.copyfile(src, tmp)
            os.replace(tmp, dest)  # atomic on the same filesystem
        except OSError as err:
            tmp.unlink(missing_ok=True)
            raise StorageError(f"Failed to store object {key!r}: {err}") from err

    def put_bytes(self, data: bytes, key: str) -> None:
        dest = self._resolve(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=".tmp_")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp_name, dest)
        except OSError as err:
            Path(tmp_name).unlink(missing_ok=True)
            raise StorageError(f"Failed to write object {key!r}: {err}") from err

    def get_bytes(self, key: str) -> bytes:
        path = self._resolve(key)
        try:
            return path.read_bytes()
        except FileNotFoundError as err:
            raise ObjectNotFoundError(f"Object not found: {key!r}") from err

    @contextmanager
    def local_path(self, key: str) -> Iterator[Path]:
        path = self._resolve(key)
        if not path.is_file():
            raise ObjectNotFoundError(f"Object not found: {key!r}")
        yield path

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def delete(self, key: str) -> None:
        self._resolve(key).unlink(missing_ok=True)

    def delete_prefix(self, prefix: str) -> None:
        path = self._resolve(prefix)
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.is_file():
            path.unlink(missing_ok=True)
