#!/usr/bin/env python3
import json
import os
import sys
from collections import defaultdict


def polygon_area(xs, ys):
    # Shoelace formula; returns absolute area in pixel units
    if not xs or not ys or len(xs) != len(ys):
        return 0.0
    n = len(xs)
    area2 = 0
    for i in range(n):
        j = (i + 1) % n
        area2 += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area2) * 0.5


def main():
    # Default path from the user's request; allow override via CLI
    default_path = os.path.join("01_labels", "annotations_49.ordered.json")
    json_path = sys.argv[1] if len(sys.argv) > 1 else default_path

    if not os.path.exists(json_path):
        print(f"Error: file not found: {json_path}")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_meta = data.get("_via_img_metadata", {})
    if not img_meta:
        print("Error: _via_img_metadata not found or empty in the JSON")
        sys.exit(1)

    r_pixels_by_file = defaultdict(float)

    for _id, meta in img_meta.items():
        filename = meta.get("filename") or _id
        for region in meta.get("regions", []):
            attrs = region.get("region_attributes", {})
            # VIA converted set uses { name: <label>, type: <labeltype> }
            # Our pipeline maps 'type' to classes; treat BR/YR/R as rust ('R')
            dtype = (attrs.get("type") or attrs.get("name") or attrs.get("label") or attrs.get("class") or "").upper()
            if dtype not in {"R", "BR", "YR"}:
                continue
            shape = region.get("shape_attributes", {})
            if shape.get("name") != "polygon":
                continue
            xs = shape.get("all_points_x") or []
            ys = shape.get("all_points_y") or []
            area = polygon_area(xs, ys)
            r_pixels_by_file[filename] += area

    if not r_pixels_by_file:
        print("No 'R' regions found.")
        return

    top5 = sorted(r_pixels_by_file.items(), key=lambda kv: kv[1], reverse=True)[:5]

    print("Top 5 leaves by 'R' labeled pixels (approx. polygon area):")
    for i, (fname, pixels) in enumerate(top5, 1):
        print(f"{i}. {fname}\t{int(round(pixels))}")


if __name__ == "__main__":
    main()


