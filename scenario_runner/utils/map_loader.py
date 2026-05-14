import json
import os


class MGeoMapLoader:
    def __init__(self, mgeo_root: str):
        self.mgeo_root = mgeo_root
        self.link_set = self._load_json("link_set.json")
        self.node_set = self._load_json("node_set.json")

    def _load_json(self, filename: str):
        path = os.path.join(self.mgeo_root, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"MGeo file not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        result = {}
        for item in data:
            idx = item.get("idx")
            if idx is not None:
                result[idx] = item
        return result

    def get_link(self, link_id: str):
        if link_id not in self.link_set:
            raise KeyError(f"Unknown link_id: {link_id}")
        return self.link_set[link_id]

    def get_link_points(self, link_id: str):
        link = self.get_link(link_id)
        return link["points"]
