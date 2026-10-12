"""Download a pinned S3 object manifest with pagination and atomic replacement."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

BUCKET = "https://s3.amazonaws.com/openneuro.org"


def list_objects(dataset):
    token = None
    objects = []
    while True:
        params = {"list-type": "2", "prefix": dataset + "/"}
        if token: params["continuation-token"] = token
        with urlopen(BUCKET + "?" + urlencode(params), timeout=60) as response:
            root = ET.fromstring(response.read())
        for item in root.findall("{*}Contents"):
            key = item.findtext("{*}Key")
            if key.endswith("/"): continue
            objects.append({"path": key[len(dataset) + 1:],
                "size": int(item.findtext("{*}Size")),
                "etag": item.findtext("{*}ETag").strip('"')})
        if root.findtext("{*}IsTruncated") != "true": break
        token = root.findtext("{*}NextContinuationToken")
        if not token: raise ValueError("Truncated listing without continuation token")
    if not objects: raise ValueError("Dataset listing is empty")
    return objects


def valid_file(path, obj):
    if not path.is_file() or path.stat().st_size != obj["size"]: return False
    if "-" not in obj["etag"]:
        digest = hashlib.md5()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
        return digest.hexdigest() == obj["etag"]
    return False  # multipart ETags are not checksums; fetch with If-Match


def download(root, dataset, obj, attempts=5):
    relative = PurePosixPath(obj["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe manifest path")
    dest = root / relative
    if valid_file(dest, obj): return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(attempts):
        try:
            request = Request(BUCKET + "/" + quote(dataset + "/" + str(relative)),
                              headers={"If-Match": '"' + obj["etag"] + '"'})
            with urlopen(request, timeout=120) as response, part.open("wb") as stream:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(chunk)
            if part.stat().st_size != obj["size"]:
                raise ValueError(f"Wrong size: {relative}")
            if "-" not in obj["etag"] and not valid_file(part, obj):
                raise ValueError(f"Checksum mismatch: {relative}")
            part.replace(dest)
            return
        except Exception:
            if attempt + 1 == attempts: raise
            time.sleep(min(2 ** attempt, 10))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ds001787", "ds008836"])
    parser.add_argument("--bids_root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1: parser.error("workers must be positive")
    args.bids_root.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest or args.bids_root / "download-manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        if payload["dataset"] != args.dataset: raise ValueError("Manifest dataset mismatch")
    else:
        payload = {"dataset": args.dataset, "objects": list_objects(args.dataset)}
        with manifest.open("x") as stream: json.dump(payload, stream, indent=2)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda obj: download(args.bids_root, args.dataset, obj), payload["objects"]))
    print("ALL DONE: every manifest object verified or downloaded with its pinned ETag")


if __name__ == "__main__": main()
