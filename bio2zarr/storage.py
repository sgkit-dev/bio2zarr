"""Local paths and S3 object prefixes used by VCF and intermediate files."""

import os
import pathlib
import shutil

from upath import UPath
from upath.implementations.cloud import S3Path


def path(value):
    if isinstance(value, S3Path):
        return value
    if str(value).startswith("s3://"):
        result = UPath(
            value,
            endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3")
            or os.environ.get("AWS_ENDPOINT_URL"),
            use_listings_cache=False,
        )
        if (
            not result.drive
            or not result.name
            or result.path.rstrip("/") == result.drive
        ):
            raise ValueError("S3 paths must include a bucket and an object key/prefix")
        return result
    if "://" in str(value):
        raise ValueError(f"Unsupported storage protocol: {str(value).split('://')[0]}")
    return pathlib.Path(value)


def mkdir(value, *, parents=False, exist_ok=False):
    if isinstance(value, S3Path):
        # Object prefixes need no mkdir; never create a bucket implicitly.
        if not value.fs.exists(value.drive):
            raise FileNotFoundError(f"S3 bucket does not exist: {value.drive}")
    else:
        value.mkdir(parents=parents, exist_ok=exist_ok)


def remove_tree(value):
    value = path(value)
    if isinstance(value, S3Path):
        # Match only descendants: a similarly named neighbouring prefix is not ours.
        objects = value.fs.find(value.path.rstrip("/") + "/")
        if objects:
            value.fs.rm(objects)
    else:
        shutil.rmtree(value)


def file_size(value):
    value = path(value)
    return value.stat().st_size
