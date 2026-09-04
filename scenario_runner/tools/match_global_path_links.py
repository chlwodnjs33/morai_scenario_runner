#!/usr/bin/env python3
"""Match an XYZ global path to ordered MGeo link ids.

This offline migration tool does not talk to MORAI or modify MGeo inputs.
"""

import argparse
import json
import math
import statistics
from collections import defaultdict


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_path(path):
    points = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            fields = text.split()
            if len(fields) < 2:
                raise ValueError(f"{path}:{line_no}: expected at least x y")
            z = float(fields[2]) if len(fields) >= 3 else 0.0
            points.append((float(fields[0]), float(fields[1]), z))
    if len(points) < 2:
        raise ValueError(f"global path needs at least 2 points: {path}")
    return points


def coord_key(point, decimals):
    return tuple(round(float(value), decimals) for value in point[:3])


def polyline_length(points):
    return sum(
        math.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1]))
        for p0, p1 in zip(points[:-1], points[1:])
    )


def build_link_graph(links):
    outgoing = defaultdict(list)
    for link in links:
        from_node = link.get("from_node_idx")
        if from_node:
            outgoing[str(from_node)].append(str(link["idx"]))

    adjacency = defaultdict(list)
    for link in links:
        link_id = str(link["idx"])
        to_node = link.get("to_node_idx")
        if to_node:
            adjacency[link_id].extend(outgoing.get(str(to_node), []))
        for field in (
            "left_lane_change_dst_link_idx",
            "right_lane_change_dst_link_idx",
        ):
            target = link.get(field)
            if target:
                adjacency[link_id].append(str(target))
        adjacency[link_id] = list(dict.fromkeys(adjacency[link_id]))
    return adjacency


def choose_path_indices(link_points, path_index, decimals):
    candidates = []
    for point in link_points:
        indices = path_index.get(coord_key(point, decimals))
        if indices:
            candidates.append(indices)
    if not candidates:
        return []

    chosen = []
    previous = -1
    for indices in candidates:
        next_indices = [index for index in indices if index >= previous]
        index = next_indices[0] if next_indices else indices[-1]
        chosen.append(index)
        previous = index
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--link-set", required=True)
    parser.add_argument("--global-path", required=True)
    parser.add_argument("--round-decimals", type=int, default=6)
    parser.add_argument("--min-matched-points", type=int, default=2)
    parser.add_argument("--min-match-ratio", type=float, default=0.2)
    args = parser.parse_args()

    links = load_json(args.link_set)
    path_points = load_path(args.global_path)
    path_index = defaultdict(list)
    for index, point in enumerate(path_points):
        path_index[coord_key(point, args.round_decimals)].append(index)

    matched = []
    for link in links:
        link_id = str(link.get("idx", ""))
        points = link.get("points") or []
        if not link_id or len(points) < 2:
            continue
        indices = choose_path_indices(points, path_index, args.round_decimals)
        unique_indices = sorted(set(indices))
        match_count = len(unique_indices)
        match_ratio = match_count / float(len(points))
        required = min(
            len(points),
            max(
                args.min_matched_points,
                math.ceil(len(points) * args.min_match_ratio),
            ),
        )
        if match_count < required:
            continue
        increasing_pairs = sum(b >= a for a, b in zip(indices[:-1], indices[1:]))
        direction_ratio = increasing_pairs / float(max(1, len(indices) - 1))
        if direction_ratio < 0.8:
            continue
        matched.append(
            {
                "id": link_id,
                "start": min(unique_indices),
                "end": max(unique_indices),
                "median": statistics.median(unique_indices),
                "matches": match_count,
                "points": len(points),
                "ratio": match_ratio,
                "length_m": polyline_length(points),
            }
        )

    matched.sort(key=lambda item: (item["median"], item["start"], item["id"]))
    adjacency = build_link_graph(links)
    route = [item["id"] for item in matched]
    disconnected = []
    for current, following in zip(route[:-1], route[1:]):
        if following not in adjacency.get(current, []):
            disconnected.append((current, following))

    first = path_points[0]
    last = path_points[-1]
    closure = math.sqrt(sum((a - b) ** 2 for a, b in zip(first, last)))
    print(f"path_points: {len(path_points)}")
    print(f"path_length_m: {polyline_length(path_points):.3f}")
    print(f"closure_distance_m: {closure:.6f}")
    print(f"matched_links: {len(route)}")
    print(f"disconnected_pairs: {len(disconnected)}")
    for current, following in disconnected:
        print(f"DISCONNECTED {current} -> {following}")
    print("route_links:")
    for item in matched:
        print(
            f"  - {item['id']}  # path[{item['start']}:{item['end']}], "
            f"matched={item['matches']}/{item['points']}, length={item['length_m']:.1f}m"
        )


if __name__ == "__main__":
    main()
