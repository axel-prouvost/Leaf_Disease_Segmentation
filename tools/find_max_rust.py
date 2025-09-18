import json
import sys
from pathlib import Path


def polygon_area(xs, ys):
    if not xs or not ys or len(xs) != len(ys):
        return 0.0
    area2 = 0
    n = len(xs)
    for i in range(n):
        j = (i + 1) % n
        area2 += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area2) / 2.0


def main(json_path: str):
    p = Path(json_path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    img_meta = data.get("_via_img_metadata", {})

    totals_r_exact = {}
    totals_yr = {}
    totals_br = {}
    totals_r_combined = {}

    for key, meta in img_meta.items():
        filename = meta.get("filename") or key
        regions = meta.get("regions", [])
        total_area_r_exact = 0.0
        count_r_exact = 0
        total_area_yr = 0.0
        count_yr = 0
        total_area_br = 0.0
        count_br = 0
        total_area_r_combined = 0.0
        count_r_combined = 0
        for region in regions:
            attrs = region.get("region_attributes", {})
            name = (attrs.get("name") or "").strip()
            typ = (attrs.get("type") or "").strip()
            shape = region.get("shape_attributes", {})
            if shape.get("name") != "polygon":
                continue
            xs = shape.get("all_points_x") or []
            ys = shape.get("all_points_y") or []
            a = polygon_area(xs, ys)
            if a <= 0:
                continue
            is_r = (name == "R" or typ == "R")
            is_yr = (name == "YR" or typ == "YR")
            is_br = (name == "BR" or typ == "BR")
            if is_r:
                total_area_r_exact += a
                count_r_exact += 1
            if name == "YR" or typ == "YR":
                total_area_yr += a
                count_yr += 1
            if is_br:
                total_area_br += a
                count_br += 1
            if is_r or is_yr or is_br:
                total_area_r_combined += a
                count_r_combined += 1
        if count_r_exact > 0:
            entry = totals_r_exact.setdefault(filename, {"area": 0.0, "count": 0})
            entry["area"] += total_area_r_exact
            entry["count"] += count_r_exact
        if count_yr > 0:
            entry = totals_yr.setdefault(filename, {"area": 0.0, "count": 0})
            entry["area"] += total_area_yr
            entry["count"] += count_yr
        if count_br > 0:
            entry = totals_br.setdefault(filename, {"area": 0.0, "count": 0})
            entry["area"] += total_area_br
            entry["count"] += count_br
        if count_r_combined > 0:
            entry = totals_r_combined.setdefault(filename, {"area": 0.0, "count": 0})
            entry["area"] += total_area_r_combined
            entry["count"] += count_r_combined

    if totals_r_combined:
        best_file_rc, best_stats_rc = max(totals_r_combined.items(), key=lambda kv: kv[1]["area"])
        print(f"Leaf with most rust (R=YR+BR+R): {best_file_rc}")
        print(f"  polygons: {best_stats_rc['count']}")
        print(f"  area_px2: {best_stats_rc['area']:.2f}")
        top5_rc = sorted(totals_r_combined.items(), key=lambda kv: kv[1]["area"], reverse=True)[:5]
        print("Top 5 by combined R area:")
        for fname, stats in top5_rc:
            print(f"  {fname}: area={stats['area']:.2f}, count={stats['count']}")
    else:
        print("No rust (R, YR, BR) regions found.")

    if totals_r_exact:
        best_file_r, best_stats_r = max(totals_r_exact.items(), key=lambda kv: kv[1]["area"])
        print(f"Leaf with most explicit R: {best_file_r}")
        print(f"  polygons: {best_stats_r['count']}")
        print(f"  area_px2: {best_stats_r['area']:.2f}")
        top5_r = sorted(totals_r_exact.items(), key=lambda kv: kv[1]["area"], reverse=True)[:5]
        print("Top 5 by explicit R area:")
        for fname, stats in top5_r:
            print(f"  {fname}: area={stats['area']:.2f}, count={stats['count']}")
    else:
        print("No explicit R regions found.")

    if totals_yr:
        best_file_yr, best_stats_yr = max(totals_yr.items(), key=lambda kv: kv[1]["area"])
        print(f"Leaf with most yellow rust (YR): {best_file_yr}")
        print(f"  polygons: {best_stats_yr['count']}")
        print(f"  area_px2: {best_stats_yr['area']:.2f}")
        top5_yr = sorted(totals_yr.items(), key=lambda kv: kv[1]["area"], reverse=True)[:5]
        print("Top 5 by YR area:")
        for fname, stats in top5_yr:
            print(f"  {fname}: area={stats['area']:.2f}, count={stats['count']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/find_max_rust.py <annotations.json>")
        sys.exit(1)
    main(sys.argv[1])


