import csv
import json
import os
import shutil
import subprocess
import sys


class GTBEVExpertController:
    """
    Route/config exporter and process launcher for chlwodnjs33/GT_BEV.

    This intentionally keeps GT_BEV's expert model running through its
    original entrypoint:

        python3 -m expert.src.main

    The scenario runner only updates path.csv/config.json and manages the
    process lifetime so GT_BEV can use its own ROS subscriptions for ego,
    object, and traffic-light information.
    """

    def __init__(
        self,
        repo_path,
        map_name,
        mgeo_root,
        route_points,
        start_link=None,
        end_link=None,
        max_velocity_kmh=60.0,
        traffic_light_control=True,
        python_executable=None,
        ros_remaps=None,
    ):
        self.repo_path = os.path.abspath(repo_path)
        self.map_name = map_name
        self.mgeo_root = mgeo_root
        self.route_points = route_points
        self.start_link = start_link
        self.end_link = end_link
        self.max_velocity_kmh = float(max_velocity_kmh)
        self.traffic_light_control = bool(traffic_light_control)
        self.python_executable = python_executable or sys.executable
        self.ros_remaps = list(ros_remaps or [])
        self.process = None

        self._validate_repo()
        self._prepare_config()

    def _validate_repo(self):
        self.src_parent = os.path.join(self.repo_path, "src")
        self.main_py = os.path.join(self.src_parent, "expert", "src", "main.py")
        if not os.path.exists(self.main_py):
            raise RuntimeError(
                "GT_BEV expert repo not found. Expected: "
                f"{self.main_py}. Clone https://github.com/chlwodnjs33/GT_BEV first."
            )

    def _prepare_config(self):
        config_dir = os.path.join(self.repo_path, "src", "expert", "src", "config")
        config_path = os.path.join(config_dir, "config.json")
        map_dir = os.path.join(config_dir, "map", self.map_name)
        os.makedirs(map_dir, exist_ok=True)

        path_csv = os.path.join(map_dir, "path.csv")
        with open(path_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            for point in self.route_points:
                writer.writerow([float(point[0]), float(point[1])])

        path_txt = os.path.join(
            self.repo_path,
            "src",
            "expert",
            "src",
            "path",
            "path_scenario_runner_current.txt",
        )
        with open(path_txt, "w") as f:
            f.write("# scenario_runner current route\n")
            if self.start_link is not None or self.end_link is not None:
                f.write(f"# {self.start_link} -> {self.end_link}\n")
            f.write("# x y\n")
            for point in self.route_points:
                f.write(f"{float(point[0]):.6f} {float(point[1]):.6f}\n")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        config["map"]["name"] = self.map_name
        config["map"]["use_mgeo_path"] = False
        config["map"]["traffic_light_control"] = self.traffic_light_control
        config["planning"]["velocity_profile"]["max_velocity"] = self.max_velocity_kmh

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        self._prepare_mgeo_signal_file()
        print(f"[GT_BEV] route path exported: {path_csv}, points={len(self.route_points)}")
        print(f"[GT_BEV] visualization path exported: {path_txt}")

    def _prepare_mgeo_signal_file(self):
        src = os.path.join(self.mgeo_root, "traffic_light_set.json")
        if not os.path.exists(src):
            print(f"[GT_BEV] traffic_light_set.json not found, skip: {src}")
            return

        dst_dir = os.path.join(self.repo_path, "src", self.map_name)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, "traffic_light_set.json")
        shutil.copyfile(src, dst)

    def start_process(self):
        if self.process is not None and self.process.poll() is None:
            return

        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            self.src_parent
            if not existing_pythonpath
            else self.src_parent + os.pathsep + existing_pythonpath
        )

        cmd = [
            self.python_executable,
            "-m",
            "expert.src.main",
            *self.ros_remaps,
        ]
        self.process = subprocess.Popen(cmd, cwd=self.src_parent, env=env)
        print(f"[GT_BEV] expert process started: pid={self.process.pid}, cmd={' '.join(cmd)}")

    def stop_process(self):
        if self.process is None:
            return

        if self.process.poll() is None:
            print(f"[GT_BEV] stopping expert process: pid={self.process.pid}")
            self.process.terminate()
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3.0)

        self.process = None

    def is_running(self):
        return self.process is not None and self.process.poll() is None
