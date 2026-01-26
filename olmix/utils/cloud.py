"""Cloud storage utilities."""

from urllib.parse import urlparse

import s3fs
from olmo_core.io import is_url


def expand_cloud_globs(paths: list[str], fs: s3fs.S3FileSystem | None = None) -> list[str]:
    """Expand glob patterns in cloud storage paths.

    Args:
        paths: List of paths, some may contain glob patterns (*)
        fs: Optional S3FileSystem instance (created if not provided)

    Returns:
        List of expanded paths with globs resolved
    """
    if fs is None:
        fs = s3fs.S3FileSystem()

    results = []
    for path in paths:
        if "*" not in path:
            results.append(path)
            continue

        if not is_url(path):
            raise NotImplementedError("Glob expansion only supported for URLs")

        parsed = urlparse(str(path))
        if parsed.scheme in ("s3", "r2", "weka"):
            results.extend([f"s3://{obj}" for obj in fs.glob(path)])
        elif parsed.scheme == "gs":
            raise NotImplementedError("'gs' glob expansion not supported")
        else:
            raise NotImplementedError(f"Glob expansion not supported for '{parsed.scheme}'")

    return results
