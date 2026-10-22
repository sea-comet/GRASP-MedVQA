#!/usr/bin/env python3
"""Resize every image referenced by the given splits once to 512x512 RGB PNG.

    python prepare_images.py --root ../data --splits train

Output: <root>/_training_images_512/<image_id>.png
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from data import CACHE_DIRNAME


def _convert(job: tuple[str, str]) -> None:
    from PIL import Image, ImageFile

    # Some whole-slide pathology figures exceed Pillow's default pixel guard.
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    source, target = job
    with Image.open(source) as image:
        resized = image.resize((512, 512))
        if resized.mode != "RGB":
            resized = resized.convert("RGB")
        resized.save(target, format="PNG", compress_level=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    cache = root / CACHE_DIRNAME
    cache.mkdir(parents=True, exist_ok=True)
    jobs = {}
    for split in args.splits:
        with (root / split / "qa.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                target = cache / f"{row['image_id']}.png"
                if not target.is_file():
                    jobs[str(target)] = str(root / row["image_file"])
    print(f"{len(jobs)} images to convert into {cache}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(_convert, [(source, target) for target, source in jobs.items()], chunksize=16))


if __name__ == "__main__":
    main()
