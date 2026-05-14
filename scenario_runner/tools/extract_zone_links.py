import json
import os
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path


def load_link_set(mgeo_root):
    path = os.path.join(mgeo_root, "link_set.json")
    with open(path, "r") as f:
        data = json.load(f)

    links = {}
    for item in data:
        links[item["idx"]] = item
    return links


def link_centroid(points):
    arr = np.array(points)
    return arr[:, 0].mean(), arr[:, 1].mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mgeo-root", required=True)
    parser.add_argument("--output", default="scenario_runner/config/urban_links.yaml")
    parser.add_argument("--zone-name", default="urban")
    args = parser.parse_args()

    links = load_link_set(args.mgeo_root)

    fig, ax = plt.subplots(figsize=(12, 10))

    for link_id, link in links.items():
        pts = np.array(link["points"])
        ax.plot(pts[:, 0], pts[:, 1], linewidth=0.7)
        cx, cy = link_centroid(link["points"])
        ax.text(cx, cy, link_id, fontsize=5)

    ax.set_aspect("equal")
    ax.set_title(
        "Click polygon vertices around the zone. "
        "Press Enter when done."
    )

    clicked = plt.ginput(n=-1, timeout=0)
    plt.close(fig)

    if len(clicked) < 3:
        raise RuntimeError("Need at least 3 points for polygon")

    polygon = Path(clicked)

    selected_links = []

    for link_id, link in links.items():
        pts = np.array(link["points"])
        cx, cy = link_centroid(link["points"])

        # centroid가 polygon 안에 있으면 선택
        if polygon.contains_point((cx, cy)):
            selected_links.append(link_id)

    selected_links = sorted(selected_links)

    result = {
        args.zone_name: {
            "polygon": [[float(x), float(y)] for x, y in clicked],
            "route_links": selected_links,
            "exclude_links": [],
            "event_links": {
                "sudden_brake": [],
                "obstacle": [],
                "pedestrian": []
            }
        }
    }

    with open(args.output, "w") as f:
        yaml.safe_dump(result, f, sort_keys=False, allow_unicode=True)

    print(f"Saved: {args.output}")
    print(f"Selected links: {len(selected_links)}")
    for link_id in selected_links:
        print(link_id)


if __name__ == "__main__":
    main()
