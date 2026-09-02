import csv
import json
import os
import signal
import subprocess
import sys


class GTBEVExpertController:
    """
    Route/config exporter and process launcher for chlwodnjs33/GT_BEV.

    This intentionally keeps GT_BEV's expert model running through its
    original entrypoint:

        python3 -m expert.src.main

    The scenario runner writes runtime path/config files and manages the
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
        fixed_path_csv=None,
        traffic_map_dir=None,
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
        self.fixed_path_csv = (
            os.path.abspath(fixed_path_csv) if fixed_path_csv else None
        )
        self.traffic_map_dir = os.path.abspath(traffic_map_dir or mgeo_root)
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
        base_config_path = os.path.join(config_dir, "config.json")
        runtime_dir = os.path.join(self.repo_path, ".runtime", "scenario_runner")
        os.makedirs(runtime_dir, exist_ok=True)

        if self.fixed_path_csv:
            if not os.path.isfile(self.fixed_path_csv):
                raise FileNotFoundError(
                    f"Fixed GT_BEV path CSV not found: {self.fixed_path_csv}"
                )
            self.runtime_path_csv = self.fixed_path_csv
            print(f"[GT_BEV] using fixed path CSV without rewriting: {self.runtime_path_csv}")
        else:
            self.runtime_path_csv = os.path.join(runtime_dir, "path.csv")
            with open(self.runtime_path_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y"])
                for point in self.route_points:
                    writer.writerow([float(point[0]), float(point[1])])

        self.runtime_path_txt = os.path.join(runtime_dir, "path_scenario_runner_current.txt")
        with open(self.runtime_path_txt, "w") as f:
            f.write("# scenario_runner current route\n")
            if self.start_link is not None or self.end_link is not None:
                f.write(f"# {self.start_link} -> {self.end_link}\n")
            f.write("# x y\n")
            for point in self.route_points:
                f.write(f"{float(point[0]):.6f} {float(point[1]):.6f}\n")

        with open(base_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        config["map"]["name"] = self.map_name
        config["map"]["data_dir"] = self.traffic_map_dir
        config["map"]["use_mgeo_path"] = False
        config["map"]["traffic_light_control"] = self.traffic_light_control
        config["planning"]["velocity_profile"]["max_velocity"] = self.max_velocity_kmh

        self.runtime_config_path = os.path.join(runtime_dir, "config.json")
        with open(self.runtime_config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.write("\n")

        self._validate_traffic_map_data()
        print(f"[GT_BEV] runtime config exported: {self.runtime_config_path}")
        if self.fixed_path_csv:
            print(
                f"[GT_BEV] fixed route path selected: "
                f"{self.runtime_path_csv}, points={len(self.route_points)}"
            )
        else:
            print(
                f"[GT_BEV] route path exported: "
                f"{self.runtime_path_csv}, points={len(self.route_points)}"
            )
        print(f"[GT_BEV] visualization path exported: {self.runtime_path_txt}")

    def _validate_traffic_map_data(self):
        required_files = ("global_info.json", "link_set.json", "node_set.json")
        missing = [
            name
            for name in required_files
            if not os.path.isfile(os.path.join(self.traffic_map_dir, name))
        ]
        if missing:
            raise FileNotFoundError(
                "GT_BEV traffic map data is incomplete: dir={}, missing={}".format(
                    self.traffic_map_dir, ", ".join(missing)
                )
            )
        print(f"[GT_BEV] direct traffic map data: {self.traffic_map_dir}")

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
        env["GT_BEV_CONFIG_PATH"] = self.runtime_config_path
        env["GT_BEV_PATH_CSV"] = self.runtime_path_csv
        env["GT_BEV_MAP_DIR"] = self.traffic_map_dir
        env["SCENARIO_ROUTE_PATH_TXT"] = self.runtime_path_txt

        cmd = [
            self.python_executable,
            "-m",
            "expert.src.main",
            *self.ros_remaps,
        ]
        self.process = subprocess.Popen(
            cmd,
            cwd=self.src_parent,
            env=env,
            start_new_session=True,
        )
        print(f"[GT_BEV] expert process started: pid={self.process.pid}, cmd={' '.join(cmd)}")

    def stop_process(self):
        if self.process is None:
            return

        if self.process.poll() is None:
            print(f"[GT_BEV] stopping expert process: pid={self.process.pid}")
            self._terminate_process_group(signal.SIGTERM)
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._terminate_process_group(signal.SIGKILL)
                self.process.wait(timeout=3.0)

        self.process = None

    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def _terminate_process_group(self, sig):
        try:
            os.killpg(os.getpgid(self.process.pid), sig)
        except ProcessLookupError:
            pass
        except Exception as e:
            print(f"[GT_BEV] process-group stop fallback: {e}")
            if sig == signal.SIGTERM:
                self.process.terminate()
            else:
                self.process.kill()
