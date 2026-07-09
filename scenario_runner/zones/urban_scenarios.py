import random
import time
import yaml
import math
import os

from base_scenario import BaseScenario
from utils.route_utils import build_route_between
from utils.transform_utils import (
    interpolate_on_polyline,
    nearest_point_index,
    polyline_length,
    project_distance_on_polyline,
    dist_xy,
)
from utils.gt_bev_expert import GTBEVExpertController


class UrbanBasicDriveScenario(BaseScenario):
    zone_name = "urban"
    scenario_name = "basic_drive"

    def setup(self):
        self.ego_spawn_offset_m = self.cfg.get("ego_spawn_offset_m", 5.0)
        self.decision_range = self.cfg.get("decision_range_m", 100.0)

        self.randomize_links = self.cfg.get("randomize_links", False)
        self.random = random.Random(self.cfg.get("random_seed"))
        self.zone_allowed_links = set()
        self.zone_route_links = self.load_zone_route_links()
        self.route_configured = False
        self.route_setup_mode = None
        self.random_route_pool = None
        self.recent_start_links = []
        self.recent_route_keys = []

        self.select_route_for_next_drive()

        # 첫 world 시작
        self.grpc.start_world(self.start_tf)
        self.configure_drive_with_retries()

    def load_zone_route_links(self):
        if not self.randomize_links:
            return []

        path = self.cfg.get("zone_links_path", "scenario_runner/config/urban_links.yaml")
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        zone_data = data.get(self.cfg.get("zone_links_key", self.zone_name), {})
        route_links = zone_data.get("route_links", [])
        exclude_links = set(zone_data.get("exclude_links", []))

        self.zone_allowed_links = {
            link_id
            for link_id in route_links
            if link_id not in exclude_links and link_id in self.map_loader.link_set
        }

        allow_connector_links = self.cfg.get("allow_connector_links", False)
        candidates = []
        for link_id in route_links:
            if link_id not in self.zone_allowed_links:
                continue
            if not allow_connector_links and "-" in link_id:
                continue
            candidates.append(link_id)

        if len(candidates) < 2:
            raise RuntimeError(f"Need at least 2 urban route links for random mode: {path}")

        print(f"[UrbanBasicDrive] random link candidates={len(candidates)} from {path}")
        return candidates

    def flatten_zone_link_group(self, value):
        if value is None:
            return []
        if isinstance(value, dict):
            links = []
            for item in value.values():
                links.extend(self.flatten_zone_link_group(item))
            return links
        if isinstance(value, (list, tuple)):
            links = []
            for item in value:
                links.extend(self.flatten_zone_link_group(item))
            return links
        return [str(value)]

    def load_zone_link_group(self, group_name):
        if not group_name:
            return set()

        path = self.cfg.get("zone_links_path", "scenario_runner/config/urban_links.yaml")
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        zone_data = data.get(self.cfg.get("zone_links_key", self.zone_name), {})
        return {
            link_id
            for link_id in self.flatten_zone_link_group(zone_data.get(group_name, []))
            if link_id in self.map_loader.link_set
        }

    def route_satisfies_random_filters(self, start_link, end_link, route_links):
        start_exclude_group = self.cfg.get("random_route_start_exclude_links_group")
        if start_exclude_group:
            excluded_starts = self.load_zone_link_group(start_exclude_group)
            if start_link in excluded_starts:
                return False, "start_excluded"

        end_exclude_group = self.cfg.get("random_route_end_exclude_links_group")
        if end_exclude_group:
            excluded_ends = self.load_zone_link_group(end_exclude_group)
            if end_link in excluded_ends:
                return False, "end_excluded"

        required_group = self.cfg.get("random_route_required_links_group")
        if required_group:
            required_links = self.load_zone_link_group(required_group)
            if required_links and not any(link_id in required_links for link_id in route_links):
                return False, "missing_required"

        return True, ""

    def select_route_for_next_drive(self):
        if self.randomize_links:
            self.select_random_route()
        else:
            self.start_link = self.cfg["start_link"]
            self.end_link = self.cfg["end_link"]
            self.prepare_route()

    def select_random_route(self):
        if self.cfg.get("random_route_pool_enabled", False):
            self.select_random_route_from_pool()
            return

        attempts = int(self.cfg.get("random_route_attempts", 200))
        min_length_m = float(self.cfg.get("random_min_route_length_m", 20.0))
        max_length_m = float(self.cfg.get("random_max_route_length_m", 0.0))
        max_route_links = int(self.cfg.get("random_max_route_links", 0))
        max_route_point_gap_m = float(self.cfg.get("random_max_route_point_gap_m", 0.0))
        max_route_turn_deg = float(self.cfg.get("random_max_route_turn_deg", 0.0))
        stay_in_zone = self.cfg.get("random_route_stay_in_zone", True)
        zone_link_set = self.zone_allowed_links

        last_error = None
        last_reject_reason = None
        for attempt in range(1, attempts + 1):
            start_link, end_link = self.random.sample(self.zone_route_links, 2)

            try:
                route_links, route_length_m = build_route_between(
                    self.map_loader,
                    start_link,
                    end_link,
                )
            except Exception as e:
                last_error = e
                continue

            if route_length_m < min_length_m:
                continue
            if max_length_m > 0.0 and route_length_m > max_length_m:
                continue
            if max_route_links > 0 and len(route_links) > max_route_links:
                continue
            if stay_in_zone and any(link_id not in zone_link_set for link_id in route_links):
                continue
            filters_ok, last_reject_reason = self.route_satisfies_random_filters(start_link, end_link, route_links)
            if not filters_ok:
                continue
            if max_route_point_gap_m > 0.0 or max_route_turn_deg > 0.0:
                route_points = self.build_route_points(route_links)
                if max_route_point_gap_m > 0.0:
                    max_gap = self.max_route_point_gap(route_points)
                    if max_gap > max_route_point_gap_m:
                        last_reject_reason = f"max route point gap {max_gap:.1f}m"
                        continue
                if max_route_turn_deg > 0.0:
                    max_turn = self.max_route_point_turn_deg(route_points)
                    if max_turn > max_route_turn_deg:
                        last_reject_reason = f"max route point turn {max_turn:.1f}deg"
                        continue

            self.start_link = start_link
            self.end_link = end_link
            self.prepare_route(route_links=route_links, route_length_m=route_length_m)
            print(f"[UrbanBasicDrive] random route selected on attempt={attempt}")
            return

        raise RuntimeError(
            "Failed to select connected random urban route. "
            f"attempts={attempts}, last_error={last_error}, last_reject={last_reject_reason}"
        )

    def build_random_route_pool(self):
        min_length_m = float(self.cfg.get("random_min_route_length_m", 20.0))
        max_length_m = float(self.cfg.get("random_max_route_length_m", 0.0))
        max_route_links = int(self.cfg.get("random_max_route_links", 0))
        max_route_point_gap_m = float(self.cfg.get("random_max_route_point_gap_m", 0.0))
        max_route_turn_deg = float(self.cfg.get("random_max_route_turn_deg", 0.0))
        stay_in_zone = self.cfg.get("random_route_stay_in_zone", True)
        zone_link_set = self.zone_allowed_links

        pool = []
        reject_counts = {
            "no_path": 0,
            "short": 0,
            "long": 0,
            "too_many_links": 0,
            "out_of_zone": 0,
            "gap": 0,
            "turn": 0,
            "start_excluded": 0,
            "end_excluded": 0,
            "missing_required": 0,
        }

        for start_link in self.zone_route_links:
            for end_link in self.zone_route_links:
                if start_link == end_link:
                    continue

                try:
                    route_links, route_length_m = build_route_between(
                        self.map_loader,
                        start_link,
                        end_link,
                    )
                except Exception:
                    reject_counts["no_path"] += 1
                    continue

                if route_length_m < min_length_m:
                    reject_counts["short"] += 1
                    continue
                if max_length_m > 0.0 and route_length_m > max_length_m:
                    reject_counts["long"] += 1
                    continue
                if max_route_links > 0 and len(route_links) > max_route_links:
                    reject_counts["too_many_links"] += 1
                    continue
                if stay_in_zone and any(link_id not in zone_link_set for link_id in route_links):
                    reject_counts["out_of_zone"] += 1
                    continue
                filters_ok, reject_reason = self.route_satisfies_random_filters(start_link, end_link, route_links)
                if not filters_ok:
                    reject_counts[reject_reason] = reject_counts.get(reject_reason, 0) + 1
                    continue

                route_points = self.build_route_points(route_links)
                max_gap = self.max_route_point_gap(route_points)
                if max_route_point_gap_m > 0.0 and max_gap > max_route_point_gap_m:
                    reject_counts["gap"] += 1
                    continue
                if max_route_turn_deg > 0.0 and self.max_route_point_turn_deg(route_points) > max_route_turn_deg:
                    reject_counts["turn"] += 1
                    continue

                pool.append(
                    {
                        "start_link": start_link,
                        "end_link": end_link,
                        "route_links": route_links,
                        "route_length_m": polyline_length(route_points),
                        "max_gap": max_gap,
                    }
                )

        if not pool:
            raise RuntimeError(f"No valid random route pool. rejects={reject_counts}")

        self.random.shuffle(pool)
        print(
            f"[UrbanBasicDrive] random route pool built: routes={len(pool)}, "
            f"unique_starts={len(set(item['start_link'] for item in pool))}, "
            f"unique_ends={len(set(item['end_link'] for item in pool))}, "
            f"rejects={reject_counts}"
        )
        return pool

    def select_random_route_from_pool(self):
        if self.random_route_pool is None:
            self.random_route_pool = self.build_random_route_pool()

        start_memory = int(self.cfg.get("random_recent_start_memory", 0))
        route_memory = int(self.cfg.get("random_recent_route_memory", 0))

        candidates = [
            item
            for item in self.random_route_pool
            if item["start_link"] not in self.recent_start_links
            and tuple(item["route_links"]) not in self.recent_route_keys
        ]
        if not candidates:
            candidates = [
                item
                for item in self.random_route_pool
                if tuple(item["route_links"]) not in self.recent_route_keys
            ]
        if not candidates:
            candidates = self.random_route_pool

        item = self.random.choice(candidates)
        self.start_link = item["start_link"]
        self.end_link = item["end_link"]

        self.recent_start_links.append(self.start_link)
        if start_memory > 0:
            self.recent_start_links = self.recent_start_links[-start_memory:]
        else:
            self.recent_start_links = []

        route_key = tuple(item["route_links"])
        self.recent_route_keys.append(route_key)
        if route_memory > 0:
            self.recent_route_keys = self.recent_route_keys[-route_memory:]
        else:
            self.recent_route_keys = []

        self.prepare_route(
            route_links=item["route_links"],
            route_length_m=item["route_length_m"],
        )
        print(
            f"[UrbanBasicDrive] route pool selected "
            f"candidates={len(candidates)}/{len(self.random_route_pool)}, "
            f"max_gap={item['max_gap']:.2f}m"
        )

    def prepare_route(self, route_links=None, route_length_m=None):
        # 시작 위치 계산
        start_points = self.map_loader.get_link_points(self.start_link)
        x, y, z, yaw = interpolate_on_polyline(start_points, self.ego_spawn_offset_m)
        self.start_tf = self.grpc.make_transform(x, y, z, yaw)

        # 목표 위치 계산.
        # 링크의 마지막 점은 다음 링크와 공유되는 node인 경우가 많아 MORAI destination
        # 매칭이 다음 링크로 튈 수 있으므로 end_link 내부로 조금 당긴 점을 사용한다.
        end_points = self.map_loader.get_link_points(self.end_link)
        goal_offset_from_end_m = self.cfg.get("goal_offset_from_end_m", 3.0)
        goal_offset_m = max(0.0, polyline_length(end_points) - goal_offset_from_end_m)
        self.goal_x, self.goal_y, self.goal_z, self.goal_yaw = interpolate_on_polyline(
            end_points,
            goal_offset_m,
        )

        print(f"[UrbanBasicDrive] start_link={self.start_link}")
        print(f"[UrbanBasicDrive] end_link={self.end_link}")
        print(f"[UrbanBasicDrive] start=({x:.3f}, {y:.3f}, {z:.3f}, yaw={yaw:.3f})")
        print(
            f"[UrbanBasicDrive] goal=({self.goal_x:.3f}, {self.goal_y:.3f}, {self.goal_z:.3f}, "
            f"yaw={self.goal_yaw:.3f}, offset_from_end={goal_offset_from_end_m:.1f}m)"
        )

        if route_links is None or route_length_m is None:
            route_links, route_length_m = build_route_between(
                self.map_loader,
                self.start_link,
                self.end_link,
            )

        self.route_links = route_links
        self.route_points = self.build_route_points(self.route_links)
        self.route_length_m = polyline_length(self.route_points)
        self.pure_pursuit_last_s = 0.0

        print(f"[UrbanBasicDrive] route_links={len(self.route_links)}, length={self.route_length_m:.1f}m")
        print(f"[UrbanBasicDrive] route_max_point_gap={self.max_route_point_gap(self.route_points):.2f}m")
        print(f"[UrbanBasicDrive] route_end_link={self.route_links[-1]}")
        print("[UrbanBasicDrive] route:")
        for i, link_id in enumerate(self.route_links):
            print(f"  {i:02d}: {link_id}")

        self.route_waypoint_indices = self.build_route_waypoint_indices()
        print("[UrbanBasicDrive] route waypoints:")
        for link_id, waypoint_idx in zip(self.route_links, self.route_waypoint_indices):
            print(f"  {link_id}: waypoint_idx={waypoint_idx}")

        if self.cfg.get("route_debug_enabled", False):
            self.dump_route_debug()

    def build_route_points(self, route_links):
        points = []
        for link_id in route_links:
            link_points = self.map_loader.get_link_points(link_id)
            if points and link_points:
                points.extend(link_points[1:])
            else:
                points.extend(link_points)
        return points

    def max_route_point_gap(self, points):
        if len(points) < 2:
            return 0.0

        return max(
            dist_xy(p0[0], p0[1], p1[0], p1[1])
            for p0, p1 in zip(points[:-1], points[1:])
        )

    def max_route_point_turn_deg(self, points):
        """
        인접한 두 세그먼트 사이의 heading 변화(도) 중 최댓값.
        build_route_points()가 링크 이음매에서 만드는 짧고 방향이 어긋난
        세그먼트(실제 도로 형상과 무관한 이음매 아티팩트)를 걸러내기 위함.
        """
        if len(points) < 3:
            return 0.0

        max_turn = 0.0
        prev_heading = None
        for p0, p1 in zip(points[:-1], points[1:]):
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            if dx * dx + dy * dy < 1e-6:
                continue
            heading = math.degrees(math.atan2(dy, dx))
            if prev_heading is not None:
                turn = abs((heading - prev_heading + 180.0) % 360.0 - 180.0)
                max_turn = max(max_turn, turn)
            prev_heading = heading
        return max_turn

    def dump_route_debug(self):
        debug_dir = self.cfg.get("route_debug_dir", "scenario_runner/debug_routes")
        os.makedirs(debug_dir, exist_ok=True)

        route_id = time.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(debug_dir, f"basic_drive_route_{route_id}.csv")
        png_path = os.path.join(debug_dir, f"basic_drive_route_{route_id}.png")

        with open(csv_path, "w") as f:
            f.write("idx,x,y,z\n")
            for i, point in enumerate(self.route_points):
                z = point[2] if len(point) >= 3 else 0.0
                f.write(f"{i},{point[0]},{point[1]},{z}\n")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 10))
            for link_id in self.zone_allowed_links:
                points = self.map_loader.get_link_points(link_id)
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                ax.plot(xs, ys, color="#bbbbbb", linewidth=0.7, alpha=0.7)

            xs = [p[0] for p in self.route_points]
            ys = [p[1] for p in self.route_points]
            ax.plot(xs, ys, color="#d62728", linewidth=2.5, marker=".", markersize=3)
            ax.scatter([xs[0]], [ys[0]], color="#2ca02c", s=80, label="start")
            ax.scatter([self.goal_x], [self.goal_y], color="#1f77b4", s=80, label="goal")
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(
                f"{self.start_link} -> {self.end_link}, "
                f"{len(self.route_links)} links, {self.route_length_m:.1f}m"
            )
            ax.legend()
            ax.grid(True, linewidth=0.3)
            fig.tight_layout()
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            print(f"[UrbanBasicDrive] route debug png: {png_path}")
        except Exception as e:
            print(f"[UrbanBasicDrive] route debug plot skipped: {e}")

        print(f"[UrbanBasicDrive] route debug csv: {csv_path}")

    def build_route_waypoint_indices(self):
        waypoint_indices = []
        for link_id in self.route_links:
            points = self.map_loader.get_link_points(link_id)
            waypoint_indices.append(len(points) - 1)

        if self.route_links:
            end_points = self.map_loader.get_link_points(self.route_links[-1])
            waypoint_indices[-1] = nearest_point_index(end_points, self.goal_x, self.goal_y)

        return waypoint_indices

    def configure_drive_from_start(self):
        """
        현재 MORAI world에서 Ego를 시작점에 두고 route/cruise 설정.
        """
        print("[UrbanBasicDrive] configure drive from start")
        self.route_configured = False
        self.route_setup_mode = None

        self.grpc.set_ego_transform(self.start_tf)
        if self.cfg.get("reset_ego_motion_on_start", False) and hasattr(self.grpc, "reset_ego_motion"):
            self.grpc.reset_ego_motion()
            time.sleep(float(self.cfg.get("reset_ego_motion_wait_sec", 0.5)))
        else:
            time.sleep(0.2)

        drive_control_mode = str(self.cfg.get("drive_control_mode", "morai_cruise")).lower()
        if drive_control_mode in ("gt_bev_expert", "gt_bev_external", "expert"):
            self.stop_gt_bev_expert_controller()
            if hasattr(self.grpc, "stop_ego_cruise"):
                self.grpc.stop_ego_cruise()
            auto_ok = self.grpc.set_ego_control_mode_auto()
            if hasattr(self.grpc, "set_ego_gear_drive"):
                self.grpc.set_ego_gear_drive()
            self.setup_gt_bev_expert_controller()
            self.gt_bev_expert.start_process()
            self.route_configured = auto_ok
            self.route_setup_mode = "gt_bev_expert"
            print("[UrbanBasicDrive] using external GT_BEV expert process")
            return auto_ok

        if drive_control_mode in ("pure_pursuit", "ros_pure_pursuit"):
            if hasattr(self.grpc, "stop_ego_cruise"):
                self.grpc.stop_ego_cruise()
            auto_ok = self.grpc.set_ego_control_mode_auto()
            if hasattr(self.grpc, "set_ego_gear_drive"):
                self.grpc.set_ego_gear_drive()
            if drive_control_mode == "ros_pure_pursuit":
                self.init_ros_ctrl_cmd_publisher()
                self.publish_ros_route_path()
            self.route_configured = auto_ok
            self.route_setup_mode = drive_control_mode
            print(f"[UrbanBasicDrive] using {drive_control_mode} low-level route following")
            return auto_ok

        use_ego_destination = self.cfg.get("use_ego_destination", False)
        if use_ego_destination and hasattr(self.grpc, "set_ego_destination"):
            self.grpc.set_ego_destination(
                self.goal_x,
                self.goal_y,
                self.goal_z,
                decision_range=self.decision_range,
            )
        else:
            print("[UrbanBasicDrive] set_vehicle_destination skipped; using explicit route links only")

        route_ok = False
        use_route_waypoints = self.cfg.get("use_route_waypoints", False)
        if use_route_waypoints:
            route_ok = self.grpc.set_ego_route(
                self.route_links,
                decision_range=self.decision_range,
                waypoint_indices=self.route_waypoint_indices,
            )
            if route_ok:
                self.route_setup_mode = "waypoints"

        if not route_ok:
            if use_route_waypoints:
                print("[UrbanBasicDrive] waypoint route rejected; retrying plain route")
            route_ok = self.grpc.set_ego_route(
                self.route_links,
                decision_range=self.decision_range,
            )
            if route_ok:
                self.route_setup_mode = "plain"

        if not route_ok:
            print("[UrbanBasicDrive] route setup failed; cruise will not start")
            return False

        self.grpc.set_ego_control_mode_cruise()
        cruise_ok = self.grpc.set_ego_cruise(
            enable=True,
            link_speed_ratio=self.cfg.get("link_speed_ratio", 40),
            constant_velocity=self.cfg.get("constant_velocity", 20),
            cruise_type=self.cfg.get("cruise_type", "link"),
        )
        self.route_configured = cruise_ok
        return cruise_ok

    def configure_drive_with_retries(self):
        attempts = int(self.cfg.get("route_setup_attempts", 5))

        for attempt in range(1, attempts + 1):
            if self.configure_drive_from_start():
                if attempt > 1:
                    print(f"[UrbanBasicDrive] route setup recovered on attempt={attempt}")
                return

            print(f"[UrbanBasicDrive] route setup retry {attempt}/{attempts}")
            if self.randomize_links:
                self.select_route_for_next_drive()
            time.sleep(0.5)

        raise RuntimeError(f"Failed to configure MORAI route after {attempts} attempts")

    def restart_to_start_and_drive(self):
        """
        목표 도착 후 다음 시작/목표를 준비하고 MORAI world 자체를 재시작해 다시 주행 시작.
        teleport만 하면 built-in cruise 내부 route 상태가 남아서 경로가 꼬일 수 있음.
        """
        print("[UrbanBasicDrive] restart to start")

        self.select_route_for_next_drive()
        self.grpc.restart_world(self.start_tf)
        time.sleep(0.5)

        self.configure_drive_with_retries()

    def run_timeline(self):
        if self.route_setup_mode == "gt_bev_expert":
            self.run_gt_bev_expert_timeline()
            return

        if self.route_setup_mode in ("pure_pursuit", "ros_pure_pursuit"):
            self.run_pure_pursuit_timeline()
            return

        timeout_sec = self.cfg.get("timeout_sec", 0.0)
        use_timeout = timeout_sec is not None and float(timeout_sec) > 0.0

        goal_tolerance_m = self.cfg.get("goal_tolerance_m", 10.0)
        check_period_sec = self.cfg.get("check_period_sec", 0.2)
        restart_on_route_complete = self.cfg.get("restart_on_morai_route_complete", True)
        route_complete_goal_tolerance_m = self.cfg.get(
            "route_complete_goal_tolerance_m",
            max(goal_tolerance_m, 15.0),
        )
        complete_on_end_link = self.cfg.get("complete_on_end_link", True)
        end_link_hold_sec = float(self.cfg.get("end_link_hold_sec", 0.5))
        off_route_timeout_sec = float(self.cfg.get("off_route_timeout_sec", 3.0))
        off_route_grace_sec = float(self.cfg.get("off_route_grace_sec", 2.0))

        # 0이면 무한 반복
        max_laps = int(self.cfg.get("max_laps", 0))

        timeout_msg = f"{timeout_sec}s" if use_timeout else "disabled"
        max_laps_msg = "infinite" if max_laps <= 0 else str(max_laps)

        print(
            f"[UrbanBasicDrive] running until goal. "
            f"timeout={timeout_msg}, tolerance={goal_tolerance_m}m, max_laps={max_laps_msg}"
        )

        lap = 0
        lap_start_time = time.time()
        last_print_time = 0.0
        end_link_enter_time = None
        off_route_enter_time = None

        while True:
            elapsed = time.time() - lap_start_time

            ego_x, ego_y = self.grpc.get_ego_xy()
            dist_to_goal = dist_xy(ego_x, ego_y, self.goal_x, self.goal_y)
            debug_state = None
            if hasattr(self.grpc, "get_ego_state_debug"):
                debug_state = self.grpc.get_ego_state_debug()

            current_link = debug_state["current_link"] if debug_state is not None else None
            on_route = current_link in self.route_links if current_link is not None else True
            on_end_link = current_link == self.end_link
            if on_end_link:
                if end_link_enter_time is None:
                    end_link_enter_time = time.time()
            else:
                end_link_enter_time = None

            if self.route_configured and elapsed >= off_route_grace_sec and not on_route:
                if off_route_enter_time is None:
                    off_route_enter_time = time.time()
            else:
                off_route_enter_time = None

            if elapsed - last_print_time >= 1.0:
                if debug_state is not None:
                    print(
                        f"[UrbanBasicDrive] lap={lap + 1} "
                        f"t={elapsed:.1f}s "
                        f"ego=({ego_x:.2f}, {ego_y:.2f}) "
                        f"link={debug_state['current_link']} "
                        f"route_end={self.route_links[-1]} "
                        f"route_mode={self.route_setup_mode} "
                        f"on_route={on_route} "
                        f"on_end={on_end_link} "
                        f"remain_dist={debug_state['remaining_distance']:.1f} "
                        f"remain_links={debug_state['remaining_link_count']} "
                        f"pass_dest={debug_state['is_pass_des_pos']} "
                        f"dist_to_goal={dist_to_goal:.2f}m"
                    )
                else:
                    print(
                        f"[UrbanBasicDrive] lap={lap + 1} "
                        f"t={elapsed:.1f}s "
                        f"ego=({ego_x:.2f}, {ego_y:.2f}) "
                        f"dist_to_goal={dist_to_goal:.2f}m"
                    )

                last_print_time = elapsed

            if (
                off_route_enter_time is not None
                and time.time() - off_route_enter_time >= off_route_timeout_sec
            ):
                print(
                    f"[UrbanBasicDrive] OFF ROUTE - restart "
                    f"current_link={current_link}, route={self.route_links}, "
                    f"elapsed={elapsed:.1f}s"
                )

                if hasattr(self.grpc, "stop_ego_cruise"):
                    self.grpc.stop_ego_cruise()

                self.restart_to_start_and_drive()
                lap_start_time = time.time()
                last_print_time = 0.0
                end_link_enter_time = None
                off_route_enter_time = None
                continue

            if (
                complete_on_end_link
                and end_link_enter_time is not None
                and time.time() - end_link_enter_time >= end_link_hold_sec
            ):
                lap += 1
                print(
                    f"[UrbanBasicDrive] END LINK REACHED "
                    f"lap={lap}, current_link={current_link}, "
                    f"route_end={self.route_links[-1]}, "
                    f"dist={dist_to_goal:.2f}m, elapsed={elapsed:.1f}s"
                )

                if hasattr(self.grpc, "stop_ego_cruise"):
                    self.grpc.stop_ego_cruise()

                if max_laps > 0 and lap >= max_laps:
                    print("[UrbanBasicDrive] max_laps reached. finish scenario.")
                    break

                self.notify_lap_end(lap)
                self.restart_to_start_and_drive()
                lap_start_time = time.time()
                last_print_time = 0.0
                end_link_enter_time = None
                off_route_enter_time = None
                continue

            if (
                restart_on_route_complete
                and self.route_configured
                and debug_state is not None
                and debug_state["is_pass_des_pos"]
            ):
                lap += 1
                reached = dist_to_goal <= route_complete_goal_tolerance_m
                result = "GOAL REACHED" if reached else "MORAI ROUTE COMPLETE BEFORE GOAL"
                print(
                    f"[UrbanBasicDrive] {result} "
                    f"lap={lap}, current_link={debug_state['current_link']}, "
                    f"route_end={self.route_links[-1]}, "
                    f"dist={dist_to_goal:.2f}m, elapsed={elapsed:.1f}s"
                )

                if hasattr(self.grpc, "stop_ego_cruise"):
                    self.grpc.stop_ego_cruise()

                if max_laps > 0 and lap >= max_laps:
                    print("[UrbanBasicDrive] max_laps reached. finish scenario.")
                    break

                self.notify_lap_end(lap)
                self.restart_to_start_and_drive()
                lap_start_time = time.time()
                last_print_time = 0.0
                end_link_enter_time = None
                off_route_enter_time = None
                continue

            # 목표점 도착
            if not complete_on_end_link and dist_to_goal <= goal_tolerance_m:
                lap += 1
                print(
                    f"[UrbanBasicDrive] GOAL REACHED "
                    f"lap={lap}, dist={dist_to_goal:.2f}m, elapsed={elapsed:.1f}s"
                )

                if max_laps > 0 and lap >= max_laps:
                    print("[UrbanBasicDrive] max_laps reached. finish scenario.")
                    break

                # world 자체를 재시작해서 route/cruise 상태 초기화
                self.notify_lap_end(lap)
                self.restart_to_start_and_drive()
                lap_start_time = time.time()
                last_print_time = 0.0
                end_link_enter_time = None
                off_route_enter_time = None
                continue

            # timeout
            if use_timeout and elapsed >= float(timeout_sec):
                print(
                    f"[UrbanBasicDrive] TIMEOUT "
                    f"lap={lap + 1}, dist={dist_to_goal:.2f}m, elapsed={elapsed:.1f}s"
                )
                break

            time.sleep(check_period_sec)

    def cleanup(self):
        self.stop_gt_bev_expert_controller()

    def normalize_angle_rad(self, angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def clamp(self, value, min_value, max_value):
        return max(min_value, min(max_value, value))

    def init_ros_ctrl_cmd_publisher(self):
        if hasattr(self, "ros_ctrl_cmd_pub"):
            return

        import rospy
        from morai_msgs.msg import CtrlCmd

        if not rospy.core.is_initialized():
            rospy.init_node("morai_scenario_runner_basic_drive", anonymous=True, disable_signals=True)

        topic = self.cfg.get("ros_ctrl_cmd_topic", "/ctrl_cmd_0")
        self.ros_ctrl_cmd_msg_type = CtrlCmd
        self.ros_ctrl_cmd_pub = rospy.Publisher(topic, CtrlCmd, queue_size=1)
        self.init_ros_route_path_publisher(rospy)
        time.sleep(0.2)
        print(f"[UrbanBasicDrive] ROS CtrlCmd publisher ready: {topic}")

    def init_ros_route_path_publisher(self, rospy):
        if hasattr(self, "ros_route_path_pub"):
            return

        from nav_msgs.msg import Path
        from geometry_msgs.msg import PoseStamped

        topic = self.cfg.get("ros_route_path_topic", "/scenario_route_path")
        self.ros_path_msg_type = Path
        self.ros_pose_stamped_msg_type = PoseStamped
        self.ros_route_path_pub = rospy.Publisher(topic, Path, queue_size=1, latch=True)
        print(f"[UrbanBasicDrive] ROS route Path publisher ready: {topic}")

    def publish_ros_route_path(self):
        if not hasattr(self, "ros_route_path_pub"):
            return

        import rospy

        path_msg = self.ros_path_msg_type()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = self.cfg.get("ros_route_frame_id", "map")

        for point in self.route_points:
            pose = self.ros_pose_stamped_msg_type()
            pose.header = path_msg.header
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = float(point[2]) if len(point) >= 3 else 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.ros_route_path_pub.publish(path_msg)
        print(
            f"[UrbanBasicDrive] published route Path: "
            f"{self.cfg.get('ros_route_path_topic', '/scenario_route_path')} "
            f"points={len(path_msg.poses)}"
        )

    def publish_ros_ctrl_cmd(self, steer, target_speed, current_speed, brake_override=None):
        cmd = self.ros_ctrl_cmd_msg_type()
        long_cmd_type = int(self.cfg.get("ros_ctrl_cmd_longitudinal_type", 1))
        cmd.longlCmdType = long_cmd_type
        cmd.steering = float(steer)

        if brake_override is not None:
            cmd.accel = 0.0
            cmd.brake = float(brake_override)
            cmd.velocity = 0.0
            self.ros_ctrl_cmd_pub.publish(cmd)
            return

        if long_cmd_type == 2:
            cmd.velocity = float(target_speed)
            cmd.accel = 0.0
            cmd.brake = 0.0
            self.ros_ctrl_cmd_pub.publish(cmd)
            return

        kp = float(self.cfg.get("pure_pursuit_speed_kp", 0.3))
        speed_error = float(target_speed) - float(current_speed)
        accel_cmd = kp * speed_error
        if accel_cmd >= 0.0:
            cmd.accel = self.clamp(accel_cmd, 0.0, 1.0)
            cmd.brake = 0.0
        else:
            cmd.accel = 0.0
            cmd.brake = self.clamp(-accel_cmd, 0.0, 1.0)

        self.ros_ctrl_cmd_pub.publish(cmd)

    def setup_gt_bev_expert_controller(self):
        repo_path = self.cfg.get(
            "gt_bev_repo_path",
            os.path.join("third_party", "GT_BEV"),
        )
        if not os.path.isabs(repo_path):
            repo_path = os.path.abspath(os.path.join(os.getcwd(), repo_path))
        map_name = self.cfg.get(
            "gt_bev_map_name",
            os.path.basename(os.path.normpath(self.global_cfg["paths"]["mgeo_root"])),
        )
        max_velocity_kmh = float(
            self.cfg.get(
                "gt_bev_max_velocity_kmh",
                self.cfg.get("constant_velocity", 60.0),
            )
        )
        traffic_light_control = bool(self.cfg.get("gt_bev_traffic_light_control", True))
        ros_remaps = self.cfg.get("gt_bev_ros_remaps", [])
        python_executable = self.cfg.get("gt_bev_python_executable")

        self.gt_bev_expert = GTBEVExpertController(
            repo_path=repo_path,
            map_name=map_name,
            mgeo_root=self.global_cfg["paths"]["mgeo_root"],
            route_points=self.route_points,
            start_link=self.start_link,
            end_link=self.end_link,
            max_velocity_kmh=max_velocity_kmh,
            traffic_light_control=traffic_light_control,
            python_executable=python_executable,
            ros_remaps=ros_remaps,
        )

    def stop_gt_bev_expert_controller(self):
        if hasattr(self, "gt_bev_expert") and self.gt_bev_expert is not None:
            self.gt_bev_expert.stop_process()
            self.gt_bev_expert = None

    def stop_pure_pursuit_control(self):
        if self.route_setup_mode == "gt_bev_expert":
            self.stop_gt_bev_expert_controller()
            if hasattr(self.grpc, "stop_ego_control"):
                self.grpc.stop_ego_control()
        elif self.route_setup_mode == "ros_pure_pursuit" and hasattr(self, "ros_ctrl_cmd_pub"):
            self.publish_ros_ctrl_cmd(steer=0.0, target_speed=0.0, current_speed=0.0, brake_override=1.0)
        elif hasattr(self.grpc, "stop_ego_control"):
            self.grpc.stop_ego_control()

    def project_on_route_near_progress(self, x, y, prev_s=None):
        if len(self.route_points) < 2:
            raise ValueError("Route must have at least 2 points")

        lookahead_m = float(self.cfg.get("pure_pursuit_lookahead_m", 8.0))
        back_window_m = float(self.cfg.get("pure_pursuit_projection_back_window_m", 8.0))
        front_window_m = float(
            self.cfg.get(
                "pure_pursuit_projection_front_window_m",
                max(40.0, lookahead_m * 5.0),
            )
        )

        min_s = 0.0
        max_s = float("inf")
        if prev_s is not None:
            min_s = max(0.0, prev_s - back_window_m)
            max_s = min(self.route_length_m, prev_s + front_window_m)

        best_s = 0.0
        best_x = self.route_points[0][0]
        best_y = self.route_points[0][1]
        best_dist = float("inf")
        cumulative = 0.0

        for p0, p1 in zip(self.route_points[:-1], self.route_points[1:]):
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-9:
                continue

            seg_len = math.sqrt(seg_len_sq)
            seg_start_s = cumulative
            seg_end_s = cumulative + seg_len
            cumulative = seg_end_s

            if seg_end_s < min_s or seg_start_s > max_s:
                continue

            t = ((x - p0[0]) * dx + (y - p0[1]) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            proj_s = seg_start_s + t * seg_len
            if proj_s < min_s or proj_s > max_s:
                continue

            proj_x = p0[0] + t * dx
            proj_y = p0[1] + t * dy
            dist = dist_xy(x, y, proj_x, proj_y)
            if dist < best_dist:
                best_dist = dist
                best_s = proj_s
                best_x = proj_x
                best_y = proj_y

        if best_dist == float("inf"):
            return self.project_on_route_near_progress(x, y, prev_s=None)

        return best_s, best_x, best_y, best_dist

    def compute_pure_pursuit_command(self, ego_state):
        x = ego_state["x"]
        y = ego_state["y"]
        yaw_rad = math.radians(ego_state["yaw_deg"])

        lookahead_m = float(self.cfg.get("pure_pursuit_lookahead_m", 8.0))
        wheel_base_m = float(self.cfg.get("pure_pursuit_wheel_base_m", 2.9))
        max_steer_rad = math.radians(float(self.cfg.get("pure_pursuit_max_steer_deg", 35.0)))
        target_speed = float(self.cfg.get("pure_pursuit_target_speed", 8.0))
        min_speed = float(self.cfg.get("pure_pursuit_min_speed", min(target_speed, 3.0)))
        slowdown_distance_m = float(self.cfg.get("pure_pursuit_slowdown_distance_m", 15.0))

        prev_s = getattr(self, "pure_pursuit_last_s", None)
        current_s, nearest_x, nearest_y, cross_track_error = self.project_on_route_near_progress(
            x,
            y,
            prev_s=prev_s,
        )
        if prev_s is not None:
            monotonic_s = max(prev_s, current_s)
            if monotonic_s != current_s:
                current_s = monotonic_s
                nearest_x, nearest_y, _, _ = interpolate_on_polyline(self.route_points, current_s)
                cross_track_error = dist_xy(x, y, nearest_x, nearest_y)
        self.pure_pursuit_last_s = current_s

        target_s = min(current_s + lookahead_m, self.route_length_m)
        target_x, target_y, _, _ = interpolate_on_polyline(self.route_points, target_s)

        dx = target_x - x
        dy = target_y - y
        target_dist = max(1e-3, math.hypot(dx, dy))
        alpha = self.normalize_angle_rad(math.atan2(dy, dx) - yaw_rad)
        steer_angle = math.atan2(2.0 * wheel_base_m * math.sin(alpha), target_dist)
        steer_angle = self.clamp(steer_angle, -max_steer_rad, max_steer_rad)

        steer_mode = str(self.cfg.get("pure_pursuit_steer_mode", "angle")).lower()
        if steer_mode == "normalized":
            steer_cmd = steer_angle / max_steer_rad if max_steer_rad > 1e-6 else 0.0
        else:
            steer_cmd = steer_angle
        steer_cmd *= float(self.cfg.get("pure_pursuit_steer_sign", 1.0))
        steer_cmd = self.clamp(steer_cmd, -1.0, 1.0)

        remaining_s = max(0.0, self.route_length_m - current_s)
        if slowdown_distance_m > 0.0 and remaining_s < slowdown_distance_m:
            speed_ratio = max(0.0, remaining_s / slowdown_distance_m)
            target_speed = max(min_speed, target_speed * speed_ratio)

        return {
            "steer": steer_cmd,
            "target_speed": target_speed,
            "target_x": target_x,
            "target_y": target_y,
            "current_s": current_s,
            "target_s": target_s,
            "remaining_s": remaining_s,
            "cross_track_error": cross_track_error,
            "alpha_deg": math.degrees(alpha),
        }

    def run_gt_bev_expert_timeline(self):
        timeout_sec = self.cfg.get("timeout_sec", 0.0)
        use_timeout = timeout_sec is not None and float(timeout_sec) > 0.0
        goal_tolerance_m = float(self.cfg.get("goal_tolerance_m", 4.0))
        check_period_sec = float(self.cfg.get("check_period_sec", 0.05))
        max_cross_track_error_m = float(self.cfg.get("gt_bev_max_cross_track_error_m", 10.0))
        arrival_stop_distance_m = float(
            self.cfg.get(
                "gt_bev_arrival_stop_distance_m",
                max(goal_tolerance_m, 6.0),
            )
        )
        off_route_timeout_sec = float(self.cfg.get("off_route_timeout_sec", 3.0))
        off_route_grace_sec = float(self.cfg.get("off_route_grace_sec", 2.0))
        max_laps = int(self.cfg.get("max_laps", 0))

        print(
            f"[UrbanBasicDrive] external GT_BEV expert running. "
            f"arrival_stop={arrival_stop_distance_m}m, "
            f"timeout={'disabled' if not use_timeout else str(timeout_sec) + 's'}"
        )

        lap = 0
        lap_start_time = time.time()
        last_print_time = 0.0
        off_route_enter_time = None
        self.gt_bev_last_s = 0.0

        while True:
            elapsed = time.time() - lap_start_time
            if not hasattr(self, "gt_bev_expert") or self.gt_bev_expert is None or not self.gt_bev_expert.is_running():
                raise RuntimeError("GT_BEV expert process stopped unexpectedly")

            ego_state = self.grpc.get_ego_motion_state()
            ego_x = ego_state["x"]
            ego_y = ego_state["y"]
            dist_to_goal = dist_xy(ego_x, ego_y, self.goal_x, self.goal_y)

            current_s, _, _, cross_track_error = self.project_on_route_near_progress(
                ego_x,
                ego_y,
                prev_s=getattr(self, "gt_bev_last_s", None),
            )
            current_s = max(getattr(self, "gt_bev_last_s", 0.0), current_s)
            self.gt_bev_last_s = current_s
            remaining_s = max(0.0, self.route_length_m - current_s)

            tick_hook = getattr(self, "on_gt_bev_timeline_tick", None)
            if tick_hook is not None:
                tick_hook(
                    elapsed=elapsed,
                    ego_state=ego_state,
                    current_s=current_s,
                    remaining_s=remaining_s,
                )

            near_route_end = remaining_s <= arrival_stop_distance_m
            route_end_reached = near_route_end and cross_track_error <= max_cross_track_error_m
            goal_reached = route_end_reached or dist_to_goal <= goal_tolerance_m

            if elapsed >= off_route_grace_sec and cross_track_error > max_cross_track_error_m:
                if off_route_enter_time is None:
                    off_route_enter_time = time.time()
            else:
                off_route_enter_time = None

            if elapsed - last_print_time >= 1.0:
                print(
                    f"[UrbanBasicDrive] GT_BEV lap={lap + 1} "
                    f"t={elapsed:.1f}s "
                    f"ego=({ego_x:.2f}, {ego_y:.2f}) "
                    f"yaw={ego_state['yaw_deg']:.1f} "
                    f"speed={ego_state['speed']:.1f} "
                    f"link={ego_state['current_link']} "
                    f"s={current_s:.1f}/{self.route_length_m:.1f} "
                    f"cte={cross_track_error:.2f} "
                    f"remain={remaining_s:.1f} "
                    f"dist_to_goal={dist_to_goal:.2f}m"
                )
                last_print_time = elapsed

            if (
                off_route_enter_time is not None
                and time.time() - off_route_enter_time >= off_route_timeout_sec
            ):
                print(
                    f"[UrbanBasicDrive] GT_BEV OFF ROUTE - restart "
                    f"cte={cross_track_error:.2f}m, link={ego_state['current_link']}, "
                    f"elapsed={elapsed:.1f}s"
                )
                self.stop_pure_pursuit_control()
                self.restart_to_start_and_drive()
                lap_start_time = time.time()
                last_print_time = 0.0
                off_route_enter_time = None
                self.gt_bev_last_s = 0.0
                continue

            if goal_reached:
                lap += 1
                print(
                    f"[UrbanBasicDrive] GT_BEV GOAL REACHED "
                    f"lap={lap}, link={ego_state['current_link']}, dist={dist_to_goal:.2f}m, "
                    f"s={current_s:.1f}/{self.route_length_m:.1f}, "
                    f"elapsed={elapsed:.1f}s"
                )
                self.stop_pure_pursuit_control()

                if max_laps > 0 and lap >= max_laps:
                    print("[UrbanBasicDrive] max_laps reached. finish scenario.")
                    break

                self.notify_lap_end(lap)
                self.restart_to_start_and_drive()
                lap_start_time = time.time()
                last_print_time = 0.0
                off_route_enter_time = None
                self.gt_bev_last_s = 0.0
                continue

            if use_timeout and elapsed >= float(timeout_sec):
                print(
                    f"[UrbanBasicDrive] GT_BEV TIMEOUT "
                    f"lap={lap + 1}, dist={dist_to_goal:.2f}m, elapsed={elapsed:.1f}s"
                )
                self.stop_pure_pursuit_control()
                break

            time.sleep(check_period_sec)

    def run_pure_pursuit_timeline(self):
        timeout_sec = self.cfg.get("timeout_sec", 0.0)
        use_timeout = timeout_sec is not None and float(timeout_sec) > 0.0
        goal_tolerance_m = float(self.cfg.get("goal_tolerance_m", 4.0))
        check_period_sec = float(self.cfg.get("check_period_sec", 0.05))
        max_cross_track_error_m = float(self.cfg.get("pure_pursuit_max_cross_track_error_m", 8.0))
        arrival_stop_distance_m = float(
            self.cfg.get(
                "pure_pursuit_arrival_stop_distance_m",
                max(goal_tolerance_m, 6.0),
            )
        )
        off_route_timeout_sec = float(self.cfg.get("off_route_timeout_sec", 3.0))
        off_route_grace_sec = float(self.cfg.get("off_route_grace_sec", 2.0))
        max_laps = int(self.cfg.get("max_laps", 0))

        print(
            f"[UrbanBasicDrive] pure pursuit running. "
            f"speed={self.cfg.get('pure_pursuit_target_speed', 8.0)}, "
            f"lookahead={self.cfg.get('pure_pursuit_lookahead_m', 8.0)}m, "
            f"arrival_stop={arrival_stop_distance_m}m, "
            f"timeout={'disabled' if not use_timeout else str(timeout_sec) + 's'}"
        )

        lap = 0
        lap_start_time = time.time()
        last_print_time = 0.0
        off_route_enter_time = None

        while True:
            elapsed = time.time() - lap_start_time
            ego_state = self.grpc.get_ego_motion_state()
            ego_x = ego_state["x"]
            ego_y = ego_state["y"]
            dist_to_goal = dist_xy(ego_x, ego_y, self.goal_x, self.goal_y)
            command = self.compute_pure_pursuit_command(ego_state)
            current_link = ego_state["current_link"]

            near_route_end = command["remaining_s"] <= arrival_stop_distance_m
            route_end_reached = near_route_end and command["cross_track_error"] <= max_cross_track_error_m
            goal_reached = route_end_reached or dist_to_goal <= goal_tolerance_m

            if elapsed >= off_route_grace_sec and command["cross_track_error"] > max_cross_track_error_m:
                if off_route_enter_time is None:
                    off_route_enter_time = time.time()
            else:
                off_route_enter_time = None

            if elapsed - last_print_time >= 1.0:
                print(
                    f"[UrbanBasicDrive] lap={lap + 1} "
                    f"t={elapsed:.1f}s "
                    f"ego=({ego_x:.2f}, {ego_y:.2f}) "
                    f"yaw={ego_state['yaw_deg']:.1f} "
                    f"speed={ego_state['speed']:.1f} "
                    f"link={current_link} "
                    f"s={command['current_s']:.1f}/{self.route_length_m:.1f} "
                    f"cte={command['cross_track_error']:.2f} "
                    f"remain={command['remaining_s']:.1f} "
                    f"steer={command['steer']:.3f} "
                    f"cmd_speed={command['target_speed']:.1f} "
                    f"dist_to_goal={dist_to_goal:.2f}m"
                )
                last_print_time = elapsed

            if (
                off_route_enter_time is not None
                and time.time() - off_route_enter_time >= off_route_timeout_sec
            ):
                print(
                    f"[UrbanBasicDrive] PURE PURSUIT OFF ROUTE - restart "
                    f"cte={command['cross_track_error']:.2f}m, link={current_link}, "
                    f"elapsed={elapsed:.1f}s"
                )
                self.stop_pure_pursuit_control()
                self.restart_to_start_and_drive()
                lap_start_time = time.time()
                last_print_time = 0.0
                off_route_enter_time = None
                continue

            if goal_reached:
                lap += 1
                print(
                    f"[UrbanBasicDrive] PURE PURSUIT GOAL REACHED "
                    f"lap={lap}, link={current_link}, dist={dist_to_goal:.2f}m, "
                    f"s={command['current_s']:.1f}/{self.route_length_m:.1f}, "
                    f"elapsed={elapsed:.1f}s"
                )
                self.stop_pure_pursuit_control()

                if max_laps > 0 and lap >= max_laps:
                    print("[UrbanBasicDrive] max_laps reached. finish scenario.")
                    break

                self.notify_lap_end(lap)
                self.restart_to_start_and_drive()
                lap_start_time = time.time()
                last_print_time = 0.0
                off_route_enter_time = None
                continue

            if use_timeout and elapsed >= float(timeout_sec):
                print(
                    f"[UrbanBasicDrive] PURE PURSUIT TIMEOUT "
                    f"lap={lap + 1}, dist={dist_to_goal:.2f}m, elapsed={elapsed:.1f}s"
                )
                self.stop_pure_pursuit_control()
                break

            if self.route_setup_mode == "ros_pure_pursuit":
                self.publish_ros_ctrl_cmd(
                    steer=command["steer"],
                    target_speed=command["target_speed"],
                    current_speed=ego_state["speed"],
                )
            else:
                self.grpc.control_ego(
                    steer=command["steer"],
                    target_speed=command["target_speed"],
                    brake=0.0,
                    throttle=0.0,
                )
            time.sleep(check_period_sec)


class UrbanSuddenBrakeScenario(BaseScenario):
    zone_name = "urban"
    scenario_name = "sudden_brake"

    def setup(self):
        self.random = random.Random(self.cfg.get("random_seed"))
        self.randomize_links = self.cfg.get("randomize_links", False)
        self.ego_spawn_offset_m = self.cfg.get("ego_spawn_offset_m", 5.0)
        self.decision_range = self.cfg.get("decision_range_m", 30.0)

        if self.randomize_links:
            self.zone_allowed_links, self.zone_route_links = self.load_zone_route_links()
            self.select_random_route()
        else:
            self.start_link = self.cfg["start_link"]
            self.end_link = self.cfg["end_link"]
            self.route_links, self.route_length_m = build_route_between(
                self.map_loader,
                self.start_link,
                self.end_link,
            )

        self.prepare_transforms()

        self.grpc.start_world(self.start_tf)
        self.spawn_npc_vehicles()

        lead_warmup_sec = float(self.cfg.get("lead_warmup_sec", 0.5))
        print(f"[UrbanSuddenBrake] NPC warmup {lead_warmup_sec:.1f}s before ego starts")
        time.sleep(lead_warmup_sec)

        self.configure_ego()

    def load_zone_route_links(self):
        path = self.cfg.get("zone_links_path", "scenario_runner/config/urban_links.yaml")
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        zone_data = data.get(self.zone_name, {})
        route_links = zone_data.get("route_links", [])
        exclude_links = set(zone_data.get("exclude_links", []))

        allowed_links = {
            link_id
            for link_id in route_links
            if link_id not in exclude_links and link_id in self.map_loader.link_set
        }

        candidates = []
        for link_id in route_links:
            if link_id not in allowed_links:
                continue
            if not self.cfg.get("allow_connector_links", False) and "-" in link_id:
                continue
            candidates.append(link_id)

        if len(candidates) < 2:
            raise RuntimeError(f"Need at least 2 urban sudden_brake candidate links: {path}")

        print(f"[UrbanSuddenBrake] random link candidates={len(candidates)} from {path}")
        return allowed_links, candidates

    def select_random_route(self):
        attempts = int(self.cfg.get("random_route_attempts", 300))
        min_length_m = float(self.cfg.get("random_min_route_length_m", 80.0))
        max_route_links = int(self.cfg.get("random_max_route_links", 10))
        prefer_stable_links = self.cfg.get("prefer_stable_brake_links", True)
        candidates = self.get_stable_route_candidates() if prefer_stable_links else self.zone_route_links

        route = self.try_select_random_route(candidates, attempts, min_length_m, max_route_links)
        if route is None and candidates is not self.zone_route_links:
            print("[UrbanSuddenBrake] stable route selection failed; fallback to all urban links")
            route = self.try_select_random_route(self.zone_route_links, attempts, min_length_m, max_route_links)

        if route is not None:
            self.start_link, self.end_link, self.route_links, self.route_length_m, attempt, candidate_count = route
            print(
                f"[UrbanSuddenBrake] random route selected on attempt={attempt} "
                f"candidates={candidate_count}"
            )
            return

        raise RuntimeError(f"Failed to select sudden_brake random route after {attempts} attempts")

    def try_select_random_route(self, candidates, attempts, min_length_m, max_route_links):
        if len(candidates) < 2:
            return None

        max_route_point_gap_m = float(self.cfg.get("random_max_route_point_gap_m", 0.0))

        for attempt in range(1, attempts + 1):
            start_link, end_link = self.random.sample(candidates, 2)
            try:
                route_links, route_length_m = build_route_between(
                    self.map_loader,
                    start_link,
                    end_link,
                )
            except Exception:
                continue

            if route_length_m < min_length_m:
                continue
            if max_route_links > 0 and len(route_links) > max_route_links:
                continue
            if self.cfg.get("random_route_stay_in_zone", True):
                if any(link_id not in self.zone_allowed_links for link_id in route_links):
                    continue
            if max_route_point_gap_m > 0.0:
                route_points = self.build_route_points(route_links)
                if self.max_route_point_gap(route_points) > max_route_point_gap_m:
                    continue

            return start_link, end_link, route_links, route_length_m, attempt, len(candidates)

        return None

    def get_stable_route_candidates(self):
        min_len = float(self.cfg.get("stable_link_min_length_m", 35.0))
        max_heading_change = float(self.cfg.get("stable_link_max_heading_change_deg", 25.0))

        candidates = []
        for link_id in self.zone_route_links:
            if "-" in link_id:
                continue
            points = self.map_loader.get_link_points(link_id)
            if len(points) < 3:
                continue
            if polyline_length(points) < min_len:
                continue
            if self.estimate_heading_change_deg(points) > max_heading_change:
                continue
            candidates.append(link_id)

        if len(candidates) >= 2:
            print(f"[UrbanSuddenBrake] stable link candidates={len(candidates)}")
            return candidates

        print("[UrbanSuddenBrake] stable link candidates too few; fallback to all urban candidates")
        return self.zone_route_links

    def estimate_heading_change_deg(self, points):
        headings = []
        for p0, p1 in zip(points[:-1], points[1:]):
            if dist_xy(p0[0], p0[1], p1[0], p1[1]) < 1e-3:
                continue
            headings.append(math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])))

        if len(headings) < 2:
            return 0.0

        first = headings[0]
        last = headings[-1]
        diff = (last - first + 180.0) % 360.0 - 180.0
        return abs(diff)

    def prepare_transforms(self):
        self.route_points = self.build_route_points(self.route_links)

        start_points = self.map_loader.get_link_points(self.start_link)
        min_spawn_offset = float(self.cfg.get("ego_spawn_min_offset_m", self.ego_spawn_offset_m))
        max_spawn_offset = float(self.cfg.get("ego_spawn_max_offset_m", min_spawn_offset))
        start_link_len = polyline_length(start_points)
        max_spawn_offset = min(max_spawn_offset, max(min_spawn_offset, start_link_len - 12.0))
        if max_spawn_offset < min_spawn_offset:
            min_spawn_offset = min(self.ego_spawn_offset_m, max(0.0, start_link_len * 0.5))
            max_spawn_offset = min_spawn_offset
        ego_offset = self.random.uniform(min_spawn_offset, max_spawn_offset)
        self.ego_spawn_offset_m = ego_offset
        ego_x, ego_y, ego_z, ego_yaw = interpolate_on_polyline(start_points, ego_offset)

        end_points = self.map_loader.get_link_points(self.end_link)
        self.goal_x, self.goal_y, self.goal_z, self.goal_yaw = interpolate_on_polyline(
            end_points,
            max(0.0, polyline_length(end_points) - self.cfg.get("goal_offset_from_end_m", 3.0)),
        )

        self.start_tf = self.grpc.make_transform(ego_x, ego_y, ego_z, ego_yaw)

        print(f"[UrbanSuddenBrake] start_link={self.start_link}")
        print(f"[UrbanSuddenBrake] end_link={self.end_link}")
        print(
            f"[UrbanSuddenBrake] ego_start=({ego_x:.3f}, {ego_y:.3f}, {ego_z:.3f}, "
            f"yaw={ego_yaw:.3f}, offset={ego_offset:.1f}m)"
        )
        print(f"[UrbanSuddenBrake] route_links={len(self.route_links)}, length={self.route_length_m:.1f}m")
        print(f"[UrbanSuddenBrake] route_max_point_gap={self.max_route_point_gap(self.route_points):.2f}m")
        print("[UrbanSuddenBrake] route:")
        for i, link_id in enumerate(self.route_links):
            print(f"  {i:02d}: {link_id}")

    def build_route_points(self, route_links):
        points = []
        for link_id in route_links:
            link_points = self.map_loader.get_link_points(link_id)
            if points and link_points:
                points.extend(link_points[1:])
            else:
                points.extend(link_points)
        return points

    def max_route_point_gap(self, points):
        if len(points) < 2:
            return 0.0

        return max(
            dist_xy(p0[0], p0[1], p1[0], p1[1])
            for p0, p1 in zip(points[:-1], points[1:])
        )

    def configure_ego(self):
        self.grpc.set_ego_transform(self.start_tf)
        time.sleep(0.2)

        self.grpc.set_ego_route(self.route_links, decision_range=self.decision_range)
        if self.cfg.get("use_ego_destination", False) and hasattr(self.grpc, "set_ego_destination"):
            self.grpc.set_ego_destination(
                self.goal_x,
                self.goal_y,
                self.goal_z,
                decision_range=self.decision_range,
            )

        ego_control_mode = str(self.cfg.get("ego_control_mode", "external_expert")).lower()
        if ego_control_mode in ("external_expert", "external", "expert"):
            if hasattr(self.grpc, "stop_ego_cruise"):
                self.grpc.stop_ego_cruise()
            auto_ok = self.grpc.set_ego_control_mode_auto()
            if hasattr(self.grpc, "set_ego_gear_drive"):
                self.grpc.set_ego_gear_drive()
            print("[UrbanSuddenBrake] Ego control is external_expert; scenario will not publish Ego controls")
            return auto_ok

        self.grpc.set_ego_control_mode_cruise()
        self.grpc.set_ego_cruise(
            enable=True,
            link_speed_ratio=self.cfg.get("link_speed_ratio", 40),
            constant_velocity=self.cfg.get("constant_velocity", 20),
            cruise_type=self.cfg.get("cruise_type", "link"),
        )
        return True

    def spawn_npc_vehicles(self):
        self.npc_vehicles = []

        min_count = int(self.cfg.get("npc_count_min", self.cfg.get("npc_count", 4)))
        max_count = int(self.cfg.get("npc_count_max", self.cfg.get("npc_count", min_count)))
        if max_count < min_count:
            max_count = min_count
        npc_count = self.random.randint(min_count, max_count)

        self.npc_models = self.get_npc_vehicle_models()
        self.spawned_npc_positions = []
        print(f"[UrbanSuddenBrake] spawning npc_count={npc_count}, models={self.npc_models}")

        self.spawn_brake_target_npc()
        self.spawn_background_npc_vehicles(max(0, npc_count - 1))

    def get_npc_vehicle_models(self):
        configured = self.cfg.get("npc_vehicle_models")
        if configured is None:
            configured = [self.cfg.get("npc_vehicle_model", self.cfg.get("lead_vehicle_model", "2014_Kia_K7"))]

        available = []
        if hasattr(self.grpc, "get_available_surround_vehicle_models"):
            available = self.grpc.get_available_surround_vehicle_models()

        if not available:
            return list(configured)

        configured_available = [model for model in configured if model in available]
        if configured_available:
            return configured_available

        sampled = list(available)
        self.random.shuffle(sampled)
        return sampled[: min(8, len(sampled))]

    def choose_npc_model(self):
        return self.random.choice(self.npc_models)

    def spawn_vehicle_with_model_retry(self, transform, label, velocity):
        tried = []
        models = list(self.npc_models)
        self.random.shuffle(models)

        for model in models:
            tried.append(model)
            npc = self.grpc.spawn_vehicle(
                transform=transform,
                model_name=model,
                label=label,
                velocity=velocity,
                multi_ego=False,
            )
            if npc is not None:
                return npc, model

        print(f"[UrbanSuddenBrake] failed to spawn {label}, tried_models={tried}")
        return None, None

    def remember_spawn_position(self, x, y):
        self.spawned_npc_positions.append((x, y))

    def has_spawn_clearance(self, x, y, min_gap):
        return all(dist_xy(x, y, px, py) >= min_gap for px, py in self.spawned_npc_positions)

    def spawn_brake_target_npc(self):
        base_speed = float(self.cfg.get("npc_speed_mps", self.cfg.get("lead_vehicle_speed_mps", 5.0)))
        speed_jitter = float(self.cfg.get("npc_speed_jitter_mps", 1.0))
        speed_limit = float(self.cfg.get("npc_speed_limit", base_speed))
        min_offset = float(self.cfg.get("npc_min_offset_m", 10.0))
        max_offset = float(self.cfg.get("npc_max_offset_m", 45.0))

        route_available_m = max(0.0, self.route_length_m - self.ego_spawn_offset_m - 5.0)
        max_offset = min(max_offset, route_available_m)
        if max_offset <= min_offset:
            max_offset = min_offset + 2.0

        offset = self.random.uniform(min_offset, max_offset)
        spawn_s = min(
            max(self.ego_spawn_offset_m + offset, self.ego_spawn_offset_m + 6.0),
            max(self.ego_spawn_offset_m + 6.0, self.route_length_m - 8.0),
        )
        x, y, z, yaw = interpolate_on_polyline(
            self.route_points,
            spawn_s,
        )
        speed = max(0.5, base_speed + self.random.uniform(-speed_jitter, speed_jitter))
        label = "npc_brake_target"
        npc, model = self.spawn_vehicle_with_model_retry(
            self.grpc.make_transform(x, y, z, yaw),
            label,
            speed,
        )
        if npc is None:
            raise RuntimeError("Failed to spawn brake target NPC")

        route_ok = self.grpc.set_vehicle_route(
            npc,
            self.route_links,
            decision_range=self.decision_range,
            label=label,
        )
        print(
            f"[UrbanSuddenBrake] {label} route={route_ok}, model={model}, "
            f"offset={offset:.1f}m, spawn_s={spawn_s:.1f}m, speed={speed:.1f}"
        )
        if not route_ok:
            print(
                f"[UrbanSuddenBrake] {label} route rejected by MORAI; "
                f"keep AI vehicle and trigger only by measured gap/speed"
            )

        npc.set_pause(False)
        if self.cfg.get("lead_control_mode", "ai") == "scripted":
            self.grpc.set_vehicle_ai(npc, False)
            if hasattr(self.grpc, "set_vehicle_physics"):
                self.grpc.set_vehicle_physics(npc, False)
        else:
            if hasattr(self.grpc, "set_vehicle_speed_limit"):
                self.grpc.set_vehicle_speed_limit(npc, speed_limit, enabled=True)
            self.grpc.set_vehicle_ai(npc, True)
            self.grpc.set_vehicle_velocity(npc, speed)
        self.remember_spawn_position(x, y)
        self.npc_vehicles.append(
            {
                "label": label,
                "actor": npc,
                "spawn_s": spawn_s,
                "route_ok": route_ok,
                "last_xy": (x, y),
                "last_time": time.time(),
                "scripted_s": spawn_s,
                "speed_mps": 0.0,
                "stopped": False,
                "can_brake_target": True,
            }
        )

    def spawn_background_npc_vehicles(self, count):
        if count <= 0:
            return

        min_gap = float(self.cfg.get("background_npc_min_spawn_gap_m", 20.0))
        min_speed = float(self.cfg.get("background_npc_speed_min_mps", 3.0))
        max_speed = float(self.cfg.get("background_npc_speed_max_mps", 7.0))
        require_stable_links = self.cfg.get("background_npc_stable_links_only", True)

        candidate_links = [
            link_id
            for link_id in self.zone_route_links
            if link_id not in self.route_links and "-" not in link_id
        ]
        if require_stable_links:
            stable_links = set(self.get_stable_route_candidates())
            candidate_links = [link_id for link_id in candidate_links if link_id in stable_links]
        self.random.shuffle(candidate_links)

        spawned = 0
        for link_id in candidate_links:
            if spawned >= count:
                break

            points = self.map_loader.get_link_points(link_id)
            link_len = polyline_length(points)
            if link_len < 12.0:
                continue

            offset = self.random.uniform(3.0, max(3.0, link_len - 3.0))
            x, y, z, yaw = interpolate_on_polyline(points, offset)
            if not self.has_spawn_clearance(x, y, min_gap):
                continue

            speed = self.random.uniform(min_speed, max_speed)
            label = f"npc_bg_{spawned}"
            npc, model = self.spawn_vehicle_with_model_retry(
                self.grpc.make_transform(x, y, z, yaw),
                label,
                speed,
            )
            if npc is None:
                continue

            route_links = self.build_background_route(link_id)
            route_ok = False
            if route_links:
                route_ok = self.grpc.set_vehicle_route(
                    npc,
                    route_links,
                    decision_range=self.decision_range,
                    label=label,
                )

            if not route_ok:
                print(
                    f"[UrbanSuddenBrake] {label} route failed; destroy unstable background npc "
                    f"model={model}, link={link_id}"
                )
                if hasattr(npc, "destroy"):
                    npc.destroy()
                continue

            print(
                f"[UrbanSuddenBrake] {label} route={route_ok}, model={model}, "
                f"link={link_id}, offset={offset:.1f}m, speed={speed:.1f}"
            )

            npc.set_pause(False)
            self.grpc.set_vehicle_ai(npc, True)
            self.grpc.set_vehicle_velocity(npc, speed)
            self.remember_spawn_position(x, y)
            self.npc_vehicles.append(
                {
                    "label": label,
                    "actor": npc,
                    "link_id": link_id,
                    "last_xy": (x, y),
                    "last_time": time.time(),
                    "speed_mps": 0.0,
                    "stopped": False,
                    "can_brake_target": False,
                }
            )
            spawned += 1

        if spawned < count:
            print(f"[UrbanSuddenBrake] background NPC spawn reduced {count}->{spawned}")

    def build_background_route(self, start_link):
        end_candidates = [
            link_id
            for link_id in self.zone_route_links
            if link_id != start_link and link_id not in self.route_links
        ]
        self.random.shuffle(end_candidates)

        for end_link in end_candidates[:20]:
            try:
                route_links, _ = build_route_between(self.map_loader, start_link, end_link)
            except Exception:
                continue

            if len(route_links) < 2:
                continue
            if any(link_id not in self.zone_allowed_links for link_id in route_links):
                continue
            return route_links

        return [start_link]

    def run_timeline(self):
        check_period_sec = self.cfg.get("check_period_sec", 0.2)
        min_follow_before_brake_sec = float(self.cfg.get("min_follow_before_brake_sec", 3.0))
        brake_after_min_sec = float(self.cfg.get("brake_after_min_sec", self.cfg.get("brake_after_sec", 0.0)))
        brake_after_max_sec = float(self.cfg.get("brake_after_max_sec", brake_after_min_sec))
        if brake_after_max_sec < brake_after_min_sec:
            brake_after_max_sec = brake_after_min_sec
        brake_after_sec = self.random.uniform(brake_after_min_sec, brake_after_max_sec)
        brake_trigger_gap_m = float(self.cfg.get("brake_trigger_gap_m", 8.0))
        brake_trigger_timeout_sec = float(self.cfg.get("brake_trigger_timeout_sec", 0.0))
        brake_min_target_speed_mps = float(self.cfg.get("brake_min_target_speed_mps", 1.0))
        brake_min_ego_speed_mps = float(self.cfg.get("brake_min_ego_speed_mps", 1.0))
        min_speed_hold_sec = float(self.cfg.get("brake_min_speed_hold_sec", 1.0))
        brake_stop_sec = float(self.cfg.get("brake_stop_sec", self.cfg.get("brake_duration_sec", 5.0)))
        after_resume_sec = float(self.cfg.get("after_resume_sec", 10.0))
        print_period_sec = float(self.cfg.get("print_period_sec", 1.0))
        lead_control_mode = self.cfg.get("lead_control_mode", "ai")
        lead_follow_gap_m = float(self.cfg.get("lead_follow_gap_m", 9.0))
        lead_script_speed_mps = float(self.cfg.get("lead_script_speed_mps", 5.0))
        lead_script_catchup_ratio = float(self.cfg.get("lead_script_catchup_ratio", 1.05))

        start_time = time.time()
        last_print_time = -print_period_sec
        last_phase = None
        stopped_vehicle = None
        resumed = False
        stop_start_time = None
        resume_time = None
        last_ego_xy = None
        last_ego_time = None
        moving_condition_start = None

        print(
            f"[UrbanSuddenBrake] running. "
            f"min_follow={min_follow_before_brake_sec}s, random_brake_after={brake_after_sec:.1f}s, "
            f"trigger_gap={brake_trigger_gap_m}m, "
            f"min_target_speed={brake_min_target_speed_mps}m/s, min_ego_speed={brake_min_ego_speed_mps}m/s, "
            f"speed_hold={min_speed_hold_sec}s, trigger_timeout={brake_trigger_timeout_sec}s, "
            f"stop={brake_stop_sec}s"
        )

        while True:
            elapsed = time.time() - start_time
            now = time.time()
            ego_x, ego_y = self.grpc.get_ego_xy()
            ego_s = project_distance_on_polyline(self.route_points, ego_x, ego_y)
            ego_speed_mps = 0.0
            if last_ego_xy is not None and last_ego_time is not None:
                ego_speed_mps = dist_xy(last_ego_xy[0], last_ego_xy[1], ego_x, ego_y) / max(
                    1e-6,
                    now - last_ego_time,
                )
            last_ego_xy = (ego_x, ego_y)
            last_ego_time = now

            front_target = None
            for npc_info in self.npc_vehicles:
                if npc_info["stopped"]:
                    continue
                if not npc_info.get("can_brake_target", False):
                    continue

                npc = npc_info["actor"]
                if lead_control_mode == "scripted":
                    target_s = min(
                        ego_s + lead_follow_gap_m,
                        max(0.0, self.route_length_m - 3.0),
                    )
                    prev_s = float(npc_info.get("scripted_s", target_s))
                    last_script_time = float(npc_info.get("last_script_time", now))
                    script_dt = max(1e-3, now - last_script_time)
                    max_speed = max(lead_script_speed_mps, ego_speed_mps * lead_script_catchup_ratio)
                    max_step = max_speed * script_dt
                    if target_s > prev_s:
                        npc_s = min(target_s, prev_s + max_step)
                    else:
                        npc_s = prev_s
                    npc_x, npc_y, npc_z, npc_yaw = interpolate_on_polyline(self.route_points, npc_s)
                    npc.set_transform(self.grpc.make_transform(npc_x, npc_y, npc_z, npc_yaw))
                    npc_info["scripted_s"] = npc_s
                    npc_info["last_script_time"] = now
                else:
                    state = npc.get_actor_state()
                    if state is None:
                        continue

                    npc_x = float(state.transform.location.x)
                    npc_y = float(state.transform.location.y)
                    npc_s = project_distance_on_polyline(self.route_points, npc_x, npc_y)
                last_x, last_y = npc_info["last_xy"]
                dt = max(1e-6, now - npc_info["last_time"])
                speed_mps = dist_xy(last_x, last_y, npc_x, npc_y) / dt
                euclid_gap = dist_xy(ego_x, ego_y, npc_x, npc_y)
                route_gap = npc_s - ego_s

                npc_info["last_xy"] = (npc_x, npc_y)
                npc_info["last_time"] = now
                npc_info["speed_mps"] = speed_mps
                npc_info["gap_m"] = route_gap
                npc_info["euclid_gap_m"] = euclid_gap
                npc_info["route_s"] = npc_s

                if route_gap <= 0.0:
                    continue

                if front_target is None or route_gap < front_target["gap_m"]:
                    front_target = npc_info

            can_trigger_brake = elapsed >= max(brake_after_sec, min_follow_before_brake_sec)
            trigger_timed_out = brake_trigger_timeout_sec > 0.0 and elapsed >= brake_trigger_timeout_sec

            if stopped_vehicle is None:
                phase = "cruise"
            elif not resumed:
                phase = "stopped"
            else:
                phase = "cruise_after"

            if phase != last_phase:
                print(f"[UrbanSuddenBrake] phase={phase} t={elapsed:.1f}s")
                last_phase = phase

            candidate = None
            if front_target is not None:
                if front_target.get("gap_m", float("inf")) <= brake_trigger_gap_m:
                    if (
                        front_target.get("speed_mps", 0.0) >= brake_min_target_speed_mps
                        and ego_speed_mps >= brake_min_ego_speed_mps
                    ):
                        candidate = front_target
                        if moving_condition_start is None:
                            moving_condition_start = time.time()
                    else:
                        moving_condition_start = None
                else:
                    moving_condition_start = None
            else:
                moving_condition_start = None

            moving_condition_held = (
                moving_condition_start is not None
                and time.time() - moving_condition_start >= min_speed_hold_sec
            )

            if stopped_vehicle is None and can_trigger_brake and (
                (candidate is not None and moving_condition_held) or trigger_timed_out
            ):
                target = candidate or front_target
                if target is None:
                    time.sleep(check_period_sec)
                    continue

                reason = "gap_and_speed" if candidate is not None else "timeout"
                print(
                    f"[UrbanSuddenBrake] LEAD SUDDEN STOP command "
                    f"target={target['label']}, t={elapsed:.1f}s, "
                    f"gap={target.get('gap_m', -1.0):.1f}m, "
                    f"target_speed={target.get('speed_mps', 0.0):.1f}m/s, "
                    f"ego_speed={ego_speed_mps:.1f}m/s, reason={reason}"
                )
                self.grpc.stop_vehicle(target["actor"])
                target["stopped"] = True
                stopped_vehicle = target
                stop_start_time = time.time()
                phase = "stopped"

            if stopped_vehicle is not None and not resumed and stop_start_time is not None:
                if time.time() - stop_start_time >= brake_stop_sec:
                    print(
                        f"[UrbanSuddenBrake] LEAD RESUME moving "
                        f"target={stopped_vehicle['label']} t={elapsed:.1f}s"
                    )
                    if lead_control_mode == "scripted":
                        stopped_vehicle["actor"].set_pause(False)
                        self.grpc.set_vehicle_ai(stopped_vehicle["actor"], False)
                        if hasattr(self.grpc, "set_vehicle_physics"):
                            self.grpc.set_vehicle_physics(stopped_vehicle["actor"], False)
                        stopped_vehicle["stopped"] = False
                    else:
                        self.grpc.resume_vehicle_ai(stopped_vehicle["actor"])
                    resumed = True
                    resume_time = time.time()

            if resumed and resume_time is not None and time.time() - resume_time >= after_resume_sec:
                print(f"[UrbanSuddenBrake] finish t={elapsed:.1f}s")
                break

            if elapsed - last_print_time >= print_period_sec:
                front_msg = "none"
                if front_target is not None:
                    front_msg = (
                        f"{front_target['label']} route_gap={front_target.get('gap_m', -1.0):.1f}m "
                        f"euclid_gap={front_target.get('euclid_gap_m', -1.0):.1f}m "
                        f"speed={front_target.get('speed_mps', 0.0):.1f}m/s"
                    )
                print(
                    f"[UrbanSuddenBrake] t={elapsed:.1f}s "
                    f"phase={phase} "
                    f"ego=({ego_x:.2f}, {ego_y:.2f}) "
                    f"ego_speed={ego_speed_mps:.1f}m/s "
                    f"front={front_msg}"
                )
                last_print_time = elapsed

            time.sleep(check_period_sec)

    def cleanup(self):
        if hasattr(self.grpc, "stop_ego_cruise"):
            self.grpc.stop_ego_cruise()


class UrbanSuddenBrakeExpertScenario(UrbanBasicDriveScenario):
    zone_name = "urban"
    scenario_name = "sudden_brake"

    def setup(self):
        self.ego_spawn_offset_m = self.cfg.get("ego_spawn_offset_m", 5.0)
        self.decision_range = self.cfg.get("decision_range_m", 30.0)

        self.randomize_links = self.cfg.get("randomize_links", False)
        self.random = random.Random(self.cfg.get("random_seed"))
        self.zone_allowed_links = set()
        self.zone_route_links = self.load_zone_route_links()
        self.route_configured = False
        self.route_setup_mode = None
        self.random_route_pool = None
        self.recent_start_links = []
        self.recent_route_keys = []
        self.npc_vehicles = []
        self.stopped_npc = None
        self.reset_brake_event_state()

        self.select_route_for_next_drive()
        self.grpc.start_world(self.start_tf)
        self.spawn_npc_vehicles()

        lead_warmup_sec = float(self.cfg.get("lead_warmup_sec", 0.0))
        if lead_warmup_sec > 0.0:
            print(f"[UrbanSuddenBrake] NPC warmup {lead_warmup_sec:.1f}s before ego starts")
            time.sleep(lead_warmup_sec)

        self.configure_drive_with_retries()

    def restart_to_start_and_drive(self):
        print("[UrbanSuddenBrake] restart to start")
        self.stop_gt_bev_expert_controller()
        self.select_route_for_next_drive()
        self.grpc.restart_world(self.start_tf)
        time.sleep(0.5)
        self.reset_brake_event_state()
        self.npc_vehicles = []
        self.stopped_npc = None
        self.spawn_npc_vehicles()
        self.configure_drive_with_retries()

    def prepare_route(self, route_links=None, route_length_m=None):
        start_points = self.map_loader.get_link_points(self.start_link)
        start_link_len = polyline_length(start_points)
        min_offset = float(self.cfg.get("ego_spawn_min_offset_m", self.ego_spawn_offset_m))
        max_offset = float(self.cfg.get("ego_spawn_max_offset_m", min_offset))
        max_offset = min(max_offset, max(min_offset, start_link_len - 12.0))
        if max_offset < min_offset:
            min_offset = min(float(self.ego_spawn_offset_m), max(0.0, start_link_len * 0.5))
            max_offset = min_offset

        self.ego_spawn_offset_m = self.random.uniform(min_offset, max_offset)
        super().prepare_route(route_links=route_links, route_length_m=route_length_m)
        print(f"[UrbanSuddenBrake] randomized ego offset={self.ego_spawn_offset_m:.1f}m")

    def reset_brake_event_state(self):
        brake_after_min_sec = float(self.cfg.get("brake_after_min_sec", 2.0))
        brake_after_max_sec = float(self.cfg.get("brake_after_max_sec", brake_after_min_sec))
        if brake_after_max_sec < brake_after_min_sec:
            brake_after_max_sec = brake_after_min_sec
        brake_after_sec = self.random.uniform(brake_after_min_sec, brake_after_max_sec)

        self.brake_event = {
            "triggered": False,
            "braking": False,
            "brake_start_speed_mps": 0.0,
            "brake_start_time": None,
            "stopped": False,
            "resumed": False,
            "stop_time": None,
            "moving_condition_start": None,
            "brake_after_sec": brake_after_sec,
        }

    def get_npc_vehicle_models(self):
        configured = self.cfg.get("npc_vehicle_models")
        if configured is None:
            configured = [self.cfg.get("npc_vehicle_model", "2014_Kia_K7")]

        available = []
        if hasattr(self.grpc, "get_available_surround_vehicle_models"):
            available = self.grpc.get_available_surround_vehicle_models()

        if not available:
            return list(configured)

        configured_available = [model for model in configured if model in available]
        if configured_available:
            return configured_available

        sampled = list(available)
        self.random.shuffle(sampled)
        return sampled[: min(8, len(sampled))]

    def spawn_vehicle_with_model_retry(self, transform, label, velocity):
        models = self.get_npc_vehicle_models()
        self.random.shuffle(models)

        for model in models:
            npc = self.grpc.spawn_vehicle(
                transform=transform,
                model_name=model,
                label=label,
                velocity=velocity,
                multi_ego=False,
            )
            if npc is not None:
                return npc, model

        return None, None

    def spawn_npc_vehicles(self):
        min_count = int(self.cfg.get("npc_count_min", self.cfg.get("npc_count", 4)))
        max_count = int(self.cfg.get("npc_count_max", self.cfg.get("npc_count", min_count)))
        if max_count < min_count:
            max_count = min_count
        npc_count = self.random.randint(min_count, max_count)

        near_count = min(npc_count, max(1, int(self.cfg.get("npc_near_ego_min_count", 5))))

        print(f"[UrbanSuddenBrake] spawning AI npc_count={npc_count} (near_ego={near_count})")
        existing_xy = []
        for i in range(near_count):
            label = f"npc_brake_candidate_{i}"
            if self.spawn_ai_npc_near_route(label, existing_xy, force_own_lane=(i == 0)):
                existing_xy.append(self.npc_vehicles[-1]["last_xy"])
            else:
                print(f"[UrbanSuddenBrake] {label} spawn skipped")

        for i in range(near_count, npc_count):
            if not self.spawn_random_ai_npc(label=f"npc_brake_candidate_{i}"):
                print(f"[UrbanSuddenBrake] npc_brake_candidate_{i} spawn skipped")

    def register_npc(self, npc, label, model, x, y, speed, route_ok=False):
        self.npc_vehicles.append(
            {
                "label": label,
                "actor": npc,
                "model": model,
                "last_xy": (x, y),
                "last_time": time.time(),
                "speed_mps": 0.0,
                "distance_m": float("inf"),
                "ahead_m": 0.0,
                "route_ok": route_ok,
                "initial_speed_mps": speed,
            }
        )

    def configure_ai_npc(self, npc, speed):
        speed_limit = float(self.cfg.get("npc_speed_limit", speed))
        npc.set_pause(False)
        if hasattr(self.grpc, "set_vehicle_speed_limit"):
            self.grpc.set_vehicle_speed_limit(npc, speed_limit, enabled=True)
        self.grpc.set_vehicle_ai(npc, True)

    def find_route_link_at_s(self, target_s):
        """route_links를 따라가다가 arc-length target_s가 속하는 (link_id, link 내부 offset)을 반환."""
        accumulated = 0.0
        for link_id in self.route_links:
            link_len = polyline_length(self.map_loader.get_link_points(link_id))
            if target_s <= accumulated + link_len:
                return link_id, max(0.0, target_s - accumulated)
            accumulated += link_len
        return self.route_links[-1], 0.0

    def spawn_ai_npc_near_route(self, label, existing_xy, force_own_lane=False):
        """
        ego 진행 경로를 따라 앞쪽 min~max 거리 범위 안에서, ego 링크 또는 좌우 인접 차로 중
        하나를 골라 스폰한다. 여러 차로에 걸쳐 차량이 퍼져 있어야 앞차가 차선을 바꿔도
        다른 차로의 NPC가 급정거 트리거를 이어받을 수 있다.
        force_own_lane=True면 좌우 차로를 고려하지 않고 항상 ego 자신의 차로(base link)에 스폰한다
        (ego와 같은 차선에 리드 차량이 반드시 있도록 보장하기 위함).
        """
        min_dist = float(self.cfg.get("background_npc_min_dist_from_ego_m", 20.0))
        max_dist = float(self.cfg.get("background_npc_max_dist_from_ego_m", 80.0))
        if max_dist < min_dist:
            max_dist = min_dist

        base_speed = float(self.cfg.get("npc_speed_mps", 12.0))
        speed_jitter = float(self.cfg.get("npc_speed_jitter_mps", 0.0))
        min_gap = float(self.cfg.get("npc_min_spacing_m", 8.0))
        attempts = int(self.cfg.get("npc_near_route_spawn_attempts", 15))

        for _ in range(attempts):
            target_s = float(self.ego_spawn_offset_m) + self.random.uniform(min_dist, max_dist)
            base_link_id, local_offset = self.find_route_link_at_s(target_s)

            if force_own_lane:
                link_id = base_link_id
            else:
                lane_links = list(self.map_loader.get_lane_group_link_ids(base_link_id))
                link_id = self.random.choice(lane_links)

            points = self.map_loader.get_link_points(link_id)
            link_len = polyline_length(points)
            if link_len < 4.0:
                continue

            offset = min(max(local_offset, 2.0), max(2.0, link_len - 2.0))
            x, y, z, yaw = interpolate_on_polyline(points, offset)
            if any(dist_xy(x, y, px, py) < min_gap for px, py in existing_xy):
                continue

            speed = max(0.5, base_speed + self.random.uniform(-speed_jitter, speed_jitter))
            npc, model = self.spawn_vehicle_with_model_retry(
                self.grpc.make_transform(x, y, z, yaw),
                label,
                speed,
            )
            if npc is None:
                continue

            self.configure_ai_npc(npc, speed)
            self.register_npc(npc, label, model, x, y, speed, route_ok=False)
            print(
                f"[UrbanSuddenBrake] npc spawned label={label}, model={model}, "
                f"link={link_id} (base={base_link_id}), route_s~={target_s:.1f}m, "
                f"speed={speed:.1f}m/s"
            )
            return True

        return False

    def spawn_ai_npc_on_route(self, label, prefer_ahead=False):
        base_speed = float(self.cfg.get("npc_speed_mps", 12.0))
        speed_jitter = float(self.cfg.get("npc_speed_jitter_mps", 0.0))
        lead_gap = float(
            self.cfg.get(
                "lead_spawn_gap_m",
                self.cfg.get("npc_max_offset_m", 18.0),
            )
        )

        if prefer_ahead:
            min_ahead_s = float(self.ego_spawn_offset_m) + 8.0
            spawn_s = min(
                max(float(self.ego_spawn_offset_m) + lead_gap, min_ahead_s),
                max(min_ahead_s, self.route_length_m - 8.0),
            )
        else:
            spawn_s = self.random.uniform(
                min(float(self.ego_spawn_offset_m) + 8.0, self.route_length_m),
                max(float(self.ego_spawn_offset_m) + 8.0, self.route_length_m - 8.0),
            )
        x, y, z, yaw = interpolate_on_polyline(self.route_points, spawn_s)
        speed = max(0.5, base_speed + self.random.uniform(-speed_jitter, speed_jitter))

        npc, model = self.spawn_vehicle_with_model_retry(
            self.grpc.make_transform(x, y, z, yaw),
            label,
            speed,
        )
        if npc is None:
            return False

        route_ok = False
        if self.cfg.get("npc_use_route", False):
            route_ok = self.grpc.set_vehicle_route(
                npc,
                self.route_links,
                decision_range=self.decision_range,
                label=label,
            )
        self.configure_ai_npc(npc, speed)
        self.register_npc(npc, label, model, x, y, speed, route_ok=route_ok)

        print(
            f"[UrbanSuddenBrake] npc spawned label={label}, model={model}, route={route_ok}, "
            f"spawn_s={spawn_s:.1f}m, "
            f"speed={speed:.1f}m/s"
        )
        return True

    def spawn_random_ai_npc(self, label):
        candidate_links = [
            link_id
            for link_id in self.zone_route_links
            if link_id in self.map_loader.link_set and "-" not in link_id
        ]
        self.random.shuffle(candidate_links)

        min_gap = float(self.cfg.get("npc_min_spacing_m", 8.0))
        existing_xy = [item["last_xy"] for item in self.npc_vehicles]
        base_speed = float(self.cfg.get("npc_speed_mps", 12.0))
        speed_jitter = float(self.cfg.get("npc_speed_jitter_mps", 0.0))

        for link_id in candidate_links[:40]:
            points = self.map_loader.get_link_points(link_id)
            link_len = polyline_length(points)
            if link_len < 12.0:
                continue

            offset = self.random.uniform(4.0, max(4.0, link_len - 4.0))
            x, y, z, yaw = interpolate_on_polyline(points, offset)
            if any(dist_xy(x, y, px, py) < min_gap for px, py in existing_xy):
                continue

            speed = max(0.5, base_speed + self.random.uniform(-speed_jitter, speed_jitter))
            npc, model = self.spawn_vehicle_with_model_retry(
                self.grpc.make_transform(x, y, z, yaw),
                label,
                speed,
            )
            if npc is None:
                continue

            self.configure_ai_npc(npc, speed)
            self.register_npc(npc, label, model, x, y, speed, route_ok=False)
            print(
                f"[UrbanSuddenBrake] npc spawned label={label}, model={model}, "
                f"link={link_id}, offset={offset:.1f}m, speed={speed:.1f}m/s"
            )
            return True

        return False

    def update_npc_states(self, ego_x, ego_y, ego_yaw_deg, ego_current_link=None):
        if not self.npc_vehicles:
            return None

        now = time.time()
        yaw_rad = math.radians(ego_yaw_deg)
        forward_x = math.cos(yaw_rad)
        forward_y = math.sin(yaw_rad)
        front_only = bool(self.cfg.get("npc_brake_front_only", True))
        same_direction_only = bool(self.cfg.get("npc_brake_same_direction_only", True))
        heading_min_cos = float(self.cfg.get("npc_brake_heading_min_cos", 0.5))

        lane_group_filter = bool(self.cfg.get("brake_lane_group_filter", False))
        include_adjacent_lanes = bool(self.cfg.get("brake_lane_group_include_adjacent", False))
        lane_group_links = None
        if lane_group_filter and ego_current_link:
            if include_adjacent_lanes:
                lane_group_links = self.map_loader.get_lane_group_link_ids(str(ego_current_link))
            else:
                lane_group_links = {str(ego_current_link)}

        closest = None
        for npc_info in self.npc_vehicles:
            if now < float(npc_info.get("next_state_check_time", 0.0)):
                continue

            actor = npc_info["actor"]
            try:
                state = actor.get_actor_state()
            except Exception as e:
                state = None
                npc_info["state_error_count"] = int(npc_info.get("state_error_count", 0)) + 1
                npc_info["next_state_check_time"] = now + float(self.cfg.get("npc_state_retry_sec", 1.0))
                if npc_info["state_error_count"] <= 3:
                    print(f"[UrbanSuddenBrake] skip npc state {npc_info['label']}: {e}")
            if state is None:
                npc_info["next_state_check_time"] = now + float(self.cfg.get("npc_state_retry_sec", 1.0))
                continue

            npc_x = float(state.transform.location.x)
            npc_y = float(state.transform.location.y)
            npc_current_link = ""
            try:
                npc_current_link = state.vehicle_state.current_link_info.id.value
            except Exception:
                npc_current_link = ""
            dx = npc_x - ego_x
            dy = npc_y - ego_y
            distance_m = math.hypot(dx, dy)
            ahead_m = dx * forward_x + dy * forward_y
            lateral_m = abs(-dx * forward_y + dy * forward_x)
            npc_yaw_rad = math.radians(float(state.transform.rotation.z))
            heading_cos = math.cos(npc_yaw_rad - yaw_rad)

            last_x, last_y = npc_info["last_xy"]
            dt = max(1e-6, now - npc_info["last_time"])
            speed_mps = dist_xy(last_x, last_y, npc_x, npc_y) / dt

            npc_info.update(
                {
                    "last_xy": (npc_x, npc_y),
                    "last_time": now,
                    "speed_mps": speed_mps,
                    "distance_m": distance_m,
                    "ahead_m": ahead_m,
                    "lateral_m": lateral_m,
                    "heading_cos": heading_cos,
                    "current_link": npc_current_link,
                    "next_state_check_time": 0.0,
                }
            )

            if lane_group_links is not None and str(npc_current_link) not in lane_group_links:
                continue
            if front_only and ahead_m < 0.0:
                continue
            if same_direction_only and heading_cos < heading_min_cos:
                continue
            if closest is None or distance_m < closest["distance_m"]:
                closest = npc_info

        return closest

    def maybe_update_brake_event(self, elapsed, ego_speed_mps, target):
        brake_stop_sec = float(self.cfg.get("brake_stop_sec", 4.0))
        brake_decel_mps2 = float(self.cfg.get("npc_brake_decel_mps2", 0.0))

        if target is None:
            self.brake_event["moving_condition_start"] = None
            if self.brake_event.get("braking") and not self.brake_event["stopped"] and self.stopped_npc is not None:
                self.grpc.stop_vehicle(self.stopped_npc["actor"])
                self.brake_event["braking"] = False
                self.brake_event["stopped"] = True
                self.brake_event["stop_time"] = time.time()
            return

        brake_trigger_gap_m = float(self.cfg.get("brake_trigger_gap_m", 10.0))
        brake_trigger_ahead_min_m = float(self.cfg.get("brake_trigger_ahead_min_m", 0.0))
        brake_trigger_lateral_max_m = float(self.cfg.get("brake_trigger_lateral_max_m", 4.0))
        min_follow_sec = float(self.cfg.get("min_follow_before_brake_sec", 2.0))
        brake_after_sec = float(self.brake_event.get("brake_after_sec", min_follow_sec))
        brake_min_target_speed_mps = float(self.cfg.get("brake_min_target_speed_mps", 0.5))
        brake_min_ego_speed_mps = float(self.cfg.get("brake_min_ego_speed_mps", 1.0))
        min_speed_hold_sec = float(self.cfg.get("brake_min_speed_hold_sec", 0.3))
        trigger_timeout_sec = float(self.cfg.get("brake_trigger_timeout_sec", 0.0))

        if not self.brake_event["triggered"]:
            gap_ok = (
                target.get("distance_m", float("inf")) <= brake_trigger_gap_m
                and target.get("ahead_m", -float("inf")) >= brake_trigger_ahead_min_m
                and target.get("lateral_m", float("inf")) <= brake_trigger_lateral_max_m
            )
            speed_ok = (
                target.get("speed_mps", 0.0) >= brake_min_target_speed_mps
                and ego_speed_mps >= brake_min_ego_speed_mps
            )
            time_ok = elapsed >= max(min_follow_sec, brake_after_sec)
            timeout_ok = trigger_timeout_sec > 0.0 and elapsed >= trigger_timeout_sec

            if (gap_ok and speed_ok and time_ok) or timeout_ok:
                if self.brake_event["moving_condition_start"] is None:
                    self.brake_event["moving_condition_start"] = time.time()
            else:
                self.brake_event["moving_condition_start"] = None

            held = (
                self.brake_event["moving_condition_start"] is not None
                and time.time() - self.brake_event["moving_condition_start"] >= min_speed_hold_sec
            )
            if held:
                print(
                    f"[UrbanSuddenBrake] LEAD SUDDEN BRAKE "
                    f"target={target['label']}, "
                    f"dist={target.get('distance_m', -1.0):.1f}m, "
                    f"ahead={target.get('ahead_m', 0.0):.1f}m, "
                    f"lat={target.get('lateral_m', 0.0):.1f}m, "
                    f"heading_cos={target.get('heading_cos', 0.0):.2f}, "
                    f"npc_speed={target.get('speed_mps', 0.0):.1f}m/s, "
                    f"ego_speed={ego_speed_mps:.1f}m/s, "
                    f"reason={'timeout' if timeout_ok and not gap_ok else 'near'}, "
                    f"t={elapsed:.1f}s"
                )
                self.stopped_npc = target
                self.brake_event["triggered"] = True
                if brake_decel_mps2 > 0.0:
                    self.brake_event["braking"] = True
                    self.brake_event["brake_start_speed_mps"] = target.get("speed_mps", 0.0)
                    self.brake_event["brake_start_time"] = time.time()
                else:
                    self.grpc.stop_vehicle(target["actor"])
                    self.brake_event["stopped"] = True
                    self.brake_event["stop_time"] = time.time()
            return

        if self.brake_event.get("braking") and not self.brake_event["stopped"]:
            brake_elapsed = time.time() - self.brake_event["brake_start_time"]
            start_speed = self.brake_event["brake_start_speed_mps"]
            current_speed = max(0.0, start_speed - brake_decel_mps2 * brake_elapsed)
            brake_npc = self.stopped_npc or target
            if current_speed <= 0.0:
                self.grpc.stop_vehicle(brake_npc["actor"])
                self.brake_event["braking"] = False
                self.brake_event["stopped"] = True
                self.brake_event["stop_time"] = time.time()
            else:
                self.grpc.set_vehicle_speed(brake_npc["actor"], current_speed)
            return

        if self.brake_event["stopped"] and not self.brake_event["resumed"]:
            if time.time() - self.brake_event["stop_time"] >= brake_stop_sec:
                print(f"[UrbanSuddenBrake] LEAD RESUME t={elapsed:.1f}s")
                resume_target = self.stopped_npc or target
                self.grpc.resume_vehicle_ai(resume_target["actor"])
                self.brake_event["stopped"] = False
                self.brake_event["resumed"] = True

    def run_gt_bev_expert_timeline(self):
        timeout_sec = self.cfg.get("timeout_sec", 0.0)
        use_timeout = timeout_sec is not None and float(timeout_sec) > 0.0
        goal_tolerance_m = float(self.cfg.get("goal_tolerance_m", 4.0))
        check_period_sec = float(self.cfg.get("check_period_sec", 0.05))
        max_cross_track_error_m = float(self.cfg.get("gt_bev_max_cross_track_error_m", 10.0))
        arrival_stop_distance_m = float(
            self.cfg.get("gt_bev_arrival_stop_distance_m", max(goal_tolerance_m, 6.0))
        )
        off_route_timeout_sec = float(self.cfg.get("off_route_timeout_sec", 3.0))
        off_route_grace_sec = float(self.cfg.get("off_route_grace_sec", 2.0))
        max_laps = int(self.cfg.get("max_laps", 0))
        print_period_sec = float(self.cfg.get("print_period_sec", 1.0))

        print(
            f"[UrbanSuddenBrake] external GT_BEV expert running. "
            f"arrival_stop={arrival_stop_distance_m}m, trigger_gap={self.cfg.get('brake_trigger_gap_m', 10.0)}m"
        )

        lap = 0
        lap_start_time = time.time()
        last_print_time = 0.0
        off_route_enter_time = None
        self.gt_bev_last_s = 0.0

        while True:
            elapsed = time.time() - lap_start_time
            if not hasattr(self, "gt_bev_expert") or self.gt_bev_expert is None or not self.gt_bev_expert.is_running():
                raise RuntimeError("GT_BEV expert process stopped unexpectedly")

            ego_state = self.grpc.get_ego_motion_state()
            ego_x = ego_state["x"]
            ego_y = ego_state["y"]
            ego_speed_mps = float(ego_state["speed"])
            dist_to_goal = dist_xy(ego_x, ego_y, self.goal_x, self.goal_y)

            current_s, _, _, cross_track_error = self.project_on_route_near_progress(
                ego_x,
                ego_y,
                prev_s=getattr(self, "gt_bev_last_s", None),
            )
            current_s = max(getattr(self, "gt_bev_last_s", 0.0), current_s)
            self.gt_bev_last_s = current_s
            remaining_s = max(0.0, self.route_length_m - current_s)
            target = self.update_npc_states(
                ego_x,
                ego_y,
                ego_state["yaw_deg"],
                ego_current_link=ego_state.get("current_link"),
            )
            self.maybe_update_brake_event(elapsed, ego_speed_mps, target)

            near_route_end = remaining_s <= arrival_stop_distance_m
            route_end_reached = near_route_end and cross_track_error <= max_cross_track_error_m
            goal_reached = route_end_reached or dist_to_goal <= goal_tolerance_m

            if elapsed >= off_route_grace_sec and cross_track_error > max_cross_track_error_m:
                if off_route_enter_time is None:
                    off_route_enter_time = time.time()
            else:
                off_route_enter_time = None

            if elapsed - last_print_time >= print_period_sec:
                front_msg = "none"
                if target is not None:
                    front_msg = (
                        f"{target['label']} dist={target.get('distance_m', -1.0):.1f}m "
                        f"ahead={target.get('ahead_m', 0.0):.1f}m "
                        f"heading_cos={target.get('heading_cos', 0.0):.2f} "
                        f"speed={target.get('speed_mps', 0.0):.1f}m/s"
                    )
                print(
                    f"[UrbanSuddenBrake] lap={lap + 1} t={elapsed:.1f}s "
                    f"ego_speed={ego_speed_mps:.1f}m/s "
                    f"s={current_s:.1f}/{self.route_length_m:.1f} "
                    f"cte={cross_track_error:.2f} remain={remaining_s:.1f} "
                    f"event={self.brake_event} front={front_msg}"
                )
                last_print_time = elapsed

            if (
                off_route_enter_time is not None
                and time.time() - off_route_enter_time >= off_route_timeout_sec
            ):
                print(
                    f"[UrbanSuddenBrake] OFF ROUTE - restart "
                    f"cte={cross_track_error:.2f}m, link={ego_state['current_link']}, elapsed={elapsed:.1f}s"
                )
                self.stop_pure_pursuit_control()
                self.restart_to_start_and_drive()
                lap_start_time = time.time()
                last_print_time = 0.0
                off_route_enter_time = None
                self.gt_bev_last_s = 0.0
                continue

            if goal_reached:
                lap += 1
                print(
                    f"[UrbanSuddenBrake] GOAL REACHED lap={lap}, "
                    f"dist={dist_to_goal:.2f}m, s={current_s:.1f}/{self.route_length_m:.1f}, "
                    f"elapsed={elapsed:.1f}s"
                )
                self.stop_pure_pursuit_control()

                if max_laps > 0 and lap >= max_laps:
                    print("[UrbanSuddenBrake] max_laps reached. finish scenario.")
                    break

                self.notify_lap_end(lap)
                self.restart_to_start_and_drive()
                lap_start_time = time.time()
                last_print_time = 0.0
                off_route_enter_time = None
                self.gt_bev_last_s = 0.0
                continue

            if use_timeout and elapsed >= float(timeout_sec):
                print(
                    f"[UrbanSuddenBrake] TIMEOUT lap={lap + 1}, "
                    f"dist={dist_to_goal:.2f}m, elapsed={elapsed:.1f}s"
                )
                self.stop_pure_pursuit_control()
                break

            time.sleep(check_period_sec)


class HighwayMergeJudgementScenario(UrbanSuddenBrakeExpertScenario):
    zone_name = "highway"
    scenario_name = "merge_judgement"

    def setup(self):
        self.ego_spawn_offset_m = self.cfg.get("ego_spawn_offset_m", 5.0)
        self.decision_range = self.cfg.get("decision_range_m", 30.0)

        self.randomize_links = False
        self.random = random.Random(self.cfg.get("random_seed"))
        self.zone_allowed_links = set()
        self.zone_route_links = []
        self.route_configured = False
        self.route_setup_mode = None
        self.random_route_pool = None
        self.recent_start_links = []
        self.recent_route_keys = []
        self.npc_vehicles = []
        self.stopped_npc = None
        self.merge_vehicle = None
        self.merge_vehicles = []
        self.merge_vehicle_info = None
        self.merge_conflict_s = None

        self.select_route_for_next_drive()
        self.grpc.start_world(self.start_tf)
        self.spawn_merge_vehicle()
        self.configure_drive_with_retries()

    def select_route_for_next_drive(self):
        route_links = list(self.cfg.get("route_links", []))
        if len(route_links) < 2:
            raise RuntimeError("merge_judgement requires at least two route_links")

        self.start_link = self.cfg.get("start_link", route_links[0])
        self.end_link = self.cfg.get("end_link", route_links[-1])
        self.prepare_route(route_links=route_links, route_length_m=0.0)
        self.merge_conflict_s = self.project_link_midpoint_on_ego_route(
            self.cfg.get("merge_conflict_link")
        )
        if self.merge_conflict_s is not None:
            print(f"[HighwayMerge] conflict_link={self.cfg.get('merge_conflict_link')} s={self.merge_conflict_s:.1f}m")

    def restart_to_start_and_drive(self):
        print("[HighwayMerge] restart to start")
        self.stop_gt_bev_expert_controller()
        self.select_route_for_next_drive()
        self.grpc.restart_world(self.start_tf)
        time.sleep(0.5)
        self.npc_vehicles = []
        self.stopped_npc = None
        self.merge_vehicle = None
        self.merge_vehicles = []
        self.merge_vehicle_info = None
        self.spawn_merge_vehicle()
        self.configure_drive_with_retries()

    def project_link_midpoint_on_ego_route(self, link_id):
        if not link_id or link_id not in self.map_loader.link_set:
            return None
        points = self.map_loader.get_link_points(link_id)
        if not points:
            return None
        mid = points[len(points) // 2]
        return project_distance_on_polyline(self.route_points, mid[0], mid[1])

    def spawn_merge_vehicle(self):
        route_links = list(self.cfg.get("merge_vehicle_route_links", []))
        if len(route_links) < 2:
            raise RuntimeError("merge_judgement requires merge_vehicle_route_links")

        start_link = self.cfg.get("merge_vehicle_start_link", route_links[0])
        start_points = self.map_loader.get_link_points(start_link)
        start_link_len = polyline_length(start_points)
        speed = float(self.cfg.get("merge_vehicle_speed_mps", 12.0))

        configured_offsets = self.cfg.get("merge_vehicle_spawn_offsets_m")
        if configured_offsets:
            spawn_offsets = [float(offset) for offset in configured_offsets]
        else:
            count = int(self.cfg.get("merge_vehicle_count", 1))
            base_offset = float(self.cfg.get("merge_vehicle_spawn_offset_m", 5.0))
            spacing = float(self.cfg.get("merge_vehicle_spacing_m", 25.0))
            spawn_offsets = [base_offset - spacing * i for i in range(count)]

        self.merge_vehicles = []
        self.merge_vehicle = None
        self.merge_vehicle_info = None

        for i, raw_offset in enumerate(spawn_offsets):
            spawn_offset = min(max(0.0, raw_offset), max(0.0, start_link_len - 3.0))
            x, y, z, yaw = interpolate_on_polyline(start_points, spawn_offset)
            label = f"npc_merge_{i}"

            npc, model = self.spawn_vehicle_with_model_retry(
                self.grpc.make_transform(x, y, z, yaw),
                label,
                speed,
            )
            if npc is None:
                print(f"[HighwayMerge] failed to spawn {label}")
                continue

            route_ok = self.grpc.set_vehicle_route(
                npc,
                route_links,
                decision_range=self.decision_range,
                label=label,
            )

            npc.set_pause(False)
            if hasattr(self.grpc, "set_vehicle_speed_limit"):
                self.grpc.set_vehicle_speed_limit(
                    npc,
                    float(self.cfg.get("merge_vehicle_speed_limit", speed)),
                    enabled=True,
                )
            self.grpc.set_vehicle_velocity(npc, speed)
            self.grpc.set_vehicle_ai(npc, bool(self.cfg.get("merge_vehicle_ai", True)))

            info = {
                "actor": npc,
                "label": label,
                "model": model,
                "route_links": route_links,
                "start_link": start_link,
                "spawn_offset_m": spawn_offset,
                "last_xy": (x, y),
                "last_time": time.time(),
                "speed_mps": 0.0,
                "route_ok": route_ok,
            }
            self.merge_vehicles.append(info)
            self.register_npc(npc, label, model, x, y, speed, route_ok=route_ok)

            if self.merge_vehicle is None:
                self.merge_vehicle = npc
                self.merge_vehicle_info = info

            print(
                f"[HighwayMerge] npc spawned label={label}, model={model}, "
                f"start_link={start_link}, offset={spawn_offset:.1f}m, "
                f"route={route_ok}, speed={speed:.1f}m/s"
            )

        if not self.merge_vehicles:
            raise RuntimeError("Failed to spawn any merge NPC")

    def get_merge_vehicle_state(self):
        candidates = self.merge_vehicles or ([self.merge_vehicle_info] if self.merge_vehicle_info else [])
        best = None
        for info in candidates:
            if not info:
                continue
            actor = info.get("actor")
            if actor is None:
                continue
            state = actor.get_actor_state()
            if state is None:
                continue
            x = float(state.transform.location.x)
            y = float(state.transform.location.y)
            link_id = ""
            try:
                link_id = state.vehicle_state.current_link_info.id.value
            except Exception:
                link_id = ""
            route_s = project_distance_on_polyline(self.route_points, x, y)
            result = {
                "x": x,
                "y": y,
                "link": link_id,
                "s": route_s,
                "label": info.get("label", "npc_merge"),
            }
            if best is None or result["s"] > best["s"]:
                best = result
        return best

    def run_gt_bev_expert_timeline(self):
        timeout_sec = self.cfg.get("timeout_sec", 0.0)
        use_timeout = timeout_sec is not None and float(timeout_sec) > 0.0
        goal_tolerance_m = float(self.cfg.get("goal_tolerance_m", 4.0))
        check_period_sec = float(self.cfg.get("check_period_sec", 0.05))
        max_cross_track_error_m = float(self.cfg.get("gt_bev_max_cross_track_error_m", 10.0))
        arrival_stop_distance_m = float(
            self.cfg.get("gt_bev_arrival_stop_distance_m", max(goal_tolerance_m, 6.0))
        )
        off_route_timeout_sec = float(self.cfg.get("off_route_timeout_sec", 3.0))
        off_route_grace_sec = float(self.cfg.get("off_route_grace_sec", 2.0))
        max_laps = int(self.cfg.get("max_laps", 1))
        print_period_sec = float(self.cfg.get("print_period_sec", 1.0))
        merge_observation_sec = float(self.cfg.get("merge_observation_sec", 12.0))
        merge_complete_on_goal_only = bool(self.cfg.get("merge_complete_on_goal_only", False))

        print(
            f"[HighwayMerge] external GT_BEV expert running. "
            f"arrival_stop={arrival_stop_distance_m}m, observation={merge_observation_sec}s"
        )

        lap = 0
        lap_start_time = time.time()
        last_print_time = 0.0
        off_route_enter_time = None
        observation_start_time = None
        self.gt_bev_last_s = 0.0

        while True:
            elapsed = time.time() - lap_start_time
            if not hasattr(self, "gt_bev_expert") or self.gt_bev_expert is None or not self.gt_bev_expert.is_running():
                raise RuntimeError("GT_BEV expert process stopped unexpectedly")

            ego_state = self.grpc.get_ego_motion_state()
            ego_x = ego_state["x"]
            ego_y = ego_state["y"]
            ego_speed_mps = float(ego_state["speed"])
            dist_to_goal = dist_xy(ego_x, ego_y, self.goal_x, self.goal_y)

            current_s, _, _, cross_track_error = self.project_on_route_near_progress(
                ego_x,
                ego_y,
                prev_s=getattr(self, "gt_bev_last_s", None),
            )
            current_s = max(getattr(self, "gt_bev_last_s", 0.0), current_s)
            self.gt_bev_last_s = current_s
            remaining_s = max(0.0, self.route_length_m - current_s)

            merge_state = self.get_merge_vehicle_state()
            merge_gap = None
            merge_dist = None
            if merge_state is not None:
                merge_gap = merge_state["s"] - current_s
                merge_dist = dist_xy(ego_x, ego_y, merge_state["x"], merge_state["y"])

            near_route_end = remaining_s <= arrival_stop_distance_m
            route_end_reached = near_route_end and cross_track_error <= max_cross_track_error_m
            goal_reached = route_end_reached or dist_to_goal <= goal_tolerance_m

            if elapsed >= off_route_grace_sec and cross_track_error > max_cross_track_error_m:
                if off_route_enter_time is None:
                    off_route_enter_time = time.time()
            else:
                off_route_enter_time = None

            if observation_start_time is None and self.merge_conflict_s is not None:
                if current_s >= max(0.0, self.merge_conflict_s - 5.0):
                    observation_start_time = time.time()
                    print(f"[HighwayMerge] merge conflict observation started t={elapsed:.1f}s")

            if elapsed - last_print_time >= print_period_sec:
                merge_msg = "none"
                if merge_state is not None:
                    merge_msg = (
                        f"link={merge_state['link']} s={merge_state['s']:.1f} "
                        f"gap={merge_gap:.1f}m dist={merge_dist:.1f}m"
                    )
                print(
                    f"[HighwayMerge] lap={lap + 1} t={elapsed:.1f}s "
                    f"ego_speed={ego_speed_mps:.1f}m/s "
                    f"s={current_s:.1f}/{self.route_length_m:.1f} "
                    f"cte={cross_track_error:.2f} remain={remaining_s:.1f} merge={merge_msg}"
                )
                last_print_time = elapsed

            if (
                not merge_complete_on_goal_only
                and observation_start_time is not None
                and time.time() - observation_start_time >= merge_observation_sec
            ):
                print(f"[HighwayMerge] scenario complete after merge observation t={elapsed:.1f}s")
                self.stop_pure_pursuit_control()
                break

            if (
                off_route_enter_time is not None
                and time.time() - off_route_enter_time >= off_route_timeout_sec
            ):
                print(
                    f"[HighwayMerge] OFF ROUTE - restart "
                    f"cte={cross_track_error:.2f}m, link={ego_state['current_link']}, elapsed={elapsed:.1f}s"
                )
                self.stop_pure_pursuit_control()
                self.restart_to_start_and_drive()
                lap_start_time = time.time()
                last_print_time = 0.0
                off_route_enter_time = None
                observation_start_time = None
                self.gt_bev_last_s = 0.0
                continue

            if goal_reached:
                lap += 1
                print(
                    f"[HighwayMerge] GOAL REACHED lap={lap}, "
                    f"dist={dist_to_goal:.2f}m, s={current_s:.1f}/{self.route_length_m:.1f}, "
                    f"elapsed={elapsed:.1f}s"
                )
                self.stop_pure_pursuit_control()

                if max_laps > 0 and lap >= max_laps:
                    print("[HighwayMerge] max_laps reached. finish scenario.")
                    break

                self.restart_to_start_and_drive()
                lap_start_time = time.time()
                last_print_time = 0.0
                off_route_enter_time = None
                observation_start_time = None
                self.gt_bev_last_s = 0.0
                continue

            if use_timeout and elapsed >= float(timeout_sec):
                print(
                    f"[HighwayMerge] TIMEOUT lap={lap + 1}, "
                    f"dist={dist_to_goal:.2f}m, elapsed={elapsed:.1f}s"
                )
                self.stop_pure_pursuit_control()
                break

            time.sleep(check_period_sec)


class RoundaboutYieldToInsideVehicleScenario(UrbanBasicDriveScenario):
    zone_name = "roundabout"
    scenario_name = "yield_to_inside_vehicle"

    def setup(self):
        self.ego_spawn_offset_m = self.cfg.get("ego_spawn_offset_m", 5.0)
        self.decision_range = self.cfg.get("decision_range_m", 30.0)

        self.randomize_links = self.cfg.get("randomize_links", False)
        self.random = random.Random(self.cfg.get("random_seed"))
        self.zone_allowed_links = set()
        self.zone_route_links = self.load_zone_route_links()
        self.route_configured = False
        self.route_setup_mode = None
        self.random_route_pool = None
        self.recent_start_links = []
        self.recent_route_keys = []
        self.npc_vehicles = []
        self.roundabout_inside_vehicle = None
        self.last_roundabout_entry_route = None

        self.select_route_for_next_drive()
        self.grpc.start_world(self.start_tf)
        self.spawn_inside_roundabout_vehicle()
        self.configure_drive_with_retries()

    def select_route_for_next_drive(self):
        if self.randomize_links:
            self.select_random_route()
            return

        entry_routes = [
            list(route)
            for route in self.cfg.get("ego_entry_routes", [])
            if len(route) >= 2
        ]
        if entry_routes:
            candidates = entry_routes
            if self.last_roundabout_entry_route is not None and len(entry_routes) > 1:
                last_key = tuple(self.last_roundabout_entry_route)
                candidates = [route for route in entry_routes if tuple(route) != last_key]

            route_links = list(self.random.choice(candidates))
            self.last_roundabout_entry_route = route_links
            self.start_link = route_links[0]
            self.end_link = route_links[-1]
            self.prepare_route(route_links=route_links, route_length_m=0.0)
            print(
                f"[RoundaboutYield] selected ego entry route "
                f"start={self.start_link}, end={self.end_link}, links={len(route_links)}"
            )
            return

        route_links = list(self.cfg.get("route_links", []))
        if route_links:
            self.start_link = self.cfg.get("start_link", route_links[0])
            self.end_link = self.cfg.get("end_link", route_links[-1])
            self.prepare_route(route_links=route_links, route_length_m=0.0)
            return

        self.start_link = self.cfg["start_link"]
        self.end_link = self.cfg["end_link"]
        self.prepare_route()

    def restart_to_start_and_drive(self):
        print("[RoundaboutYield] restart ego with new random entry route; keep inside NPCs running")
        self.stop_gt_bev_expert_controller()
        self.select_route_for_next_drive()
        time.sleep(0.2)
        self.configure_drive_with_retries()

    def get_npc_vehicle_models(self):
        configured = self.cfg.get("npc_vehicle_models")
        if configured is None:
            configured = [self.cfg.get("npc_vehicle_model", "2014_Kia_K7")]

        available = []
        if hasattr(self.grpc, "get_available_surround_vehicle_models"):
            available = self.grpc.get_available_surround_vehicle_models()

        if not available:
            return list(configured)

        configured_available = [model for model in configured if model in available]
        if configured_available:
            return configured_available

        sampled = list(available)
        self.random.shuffle(sampled)
        return sampled[: min(8, len(sampled))]

    def spawn_vehicle_with_model_retry(self, transform, label, velocity):
        models = self.get_npc_vehicle_models()
        self.random.shuffle(models)

        for model in models:
            npc = self.grpc.spawn_vehicle(
                transform=transform,
                model_name=model,
                label=label,
                velocity=velocity,
                multi_ego=False,
            )
            if npc is not None:
                return npc, model

        return None, None

    def register_roundabout_npc(self, npc, label, model, x, y, speed, route_ok=False):
        npc_info = {
            "label": label,
            "actor": npc,
            "model": model,
            "last_xy": (x, y),
            "last_time": time.time(),
            "speed_mps": 0.0,
            "route_ok": route_ok,
            "initial_speed_mps": speed,
        }
        self.npc_vehicles.append(npc_info)
        return npc_info

    def route_from_roundabout_start_link(self, base_route_links, start_link, laps=1):
        if start_link not in base_route_links:
            return list(base_route_links)

        start_idx = base_route_links.index(start_link)
        rotated = list(base_route_links[start_idx:]) + list(base_route_links[:start_idx])
        route_links = []
        for _ in range(max(1, int(laps))):
            route_links.extend(rotated)
        route_links.append(rotated[0])
        return route_links

    def spawn_inside_roundabout_vehicle(self):
        base_route_links = list(self.cfg.get("inside_vehicle_route_links", []))
        if len(base_route_links) < 2:
            raise RuntimeError("yield_to_inside_vehicle requires inside_vehicle_route_links")

        speed = float(self.cfg.get("inside_vehicle_speed_mps", self.cfg.get("npc_speed_mps", 5.0)))
        speed_limit = float(self.cfg.get("inside_vehicle_speed_limit", speed))
        route_laps = int(self.cfg.get("inside_vehicle_route_laps", 1))
        count = max(1, int(self.cfg.get("inside_vehicle_count", 1)))
        min_spacing = float(self.cfg.get("inside_vehicle_min_spacing_m", 12.0))
        random_phase = bool(self.cfg.get("inside_vehicle_spawn_phase_random", True))
        inside_ai = bool(self.cfg.get("inside_vehicle_ai", True))
        base_label = self.cfg.get("inside_vehicle_label", "npc_roundabout_inside")

        link_segments = []
        total_loop_length = 0.0
        for link_id in base_route_links:
            points = self.map_loader.get_link_points(link_id)
            link_length = polyline_length(points)
            if link_length <= 0.0:
                continue
            link_segments.append((link_id, points, total_loop_length, total_loop_length + link_length))
            total_loop_length += link_length

        if not link_segments:
            raise RuntimeError("yield_to_inside_vehicle has no valid inside loop links")

        spacing = total_loop_length / float(count)
        if spacing < min_spacing:
            print(
                f"[RoundaboutYield] inside npc spacing warning: "
                f"loop_length={total_loop_length:.1f}m, count={count}, "
                f"spacing={spacing:.1f}m < min_spacing={min_spacing:.1f}m"
            )

        phase = self.random.uniform(0.0, spacing) if random_phase and spacing > 0.0 else 0.0

        def sample_loop(loop_s):
            s = loop_s % total_loop_length
            for link_id, points, start_s, end_s in link_segments:
                if s <= end_s:
                    local_s = max(0.0, min(s - start_s, end_s - start_s))
                    x, y, z, yaw = interpolate_on_polyline(points, local_s)
                    return link_id, local_s, x, y, z, yaw
            link_id, points, start_s, end_s = link_segments[-1]
            x, y, z, yaw = interpolate_on_polyline(points, end_s - start_s)
            return link_id, end_s - start_s, x, y, z, yaw

        spawned = 0
        for idx in range(count):
            loop_s = phase + spacing * idx
            start_link, spawn_offset, x, y, z, yaw = sample_loop(loop_s)
            route_links = self.route_from_roundabout_start_link(base_route_links, start_link, route_laps)
            if len(route_links) < 2:
                print(f"[RoundaboutYield] skip inside npc start_link={start_link}; route too short")
                continue

            label = f"{base_label}_{idx}"
            npc, model = self.spawn_vehicle_with_model_retry(
                self.grpc.make_transform(x, y, z, yaw),
                label,
                speed,
            )
            if npc is None:
                print(f"[RoundaboutYield] failed to spawn inside npc label={label}")
                continue

            route_ok = self.grpc.set_vehicle_route(
                npc,
                route_links,
                decision_range=self.decision_range,
                label=label,
            )
            npc.set_pause(False)
            if hasattr(self.grpc, "set_vehicle_speed_limit"):
                self.grpc.set_vehicle_speed_limit(npc, speed_limit, enabled=True)
            self.grpc.set_vehicle_velocity(npc, speed)
            self.grpc.set_vehicle_ai(npc, inside_ai)

            if self.roundabout_inside_vehicle is None:
                self.roundabout_inside_vehicle = npc
            self.register_roundabout_npc(npc, label, model, x, y, speed, route_ok=route_ok)
            spawned += 1
            print(
                f"[RoundaboutYield] inside npc spawned label={label}, model={model}, "
                f"start_link={start_link}, offset={spawn_offset:.1f}m, loop_s={loop_s % total_loop_length:.1f}m, "
                f"route={route_ok}, speed={speed:.1f}m/s"
            )

        if spawned == 0:
            raise RuntimeError("Failed to spawn roundabout inside NPC")


class RoundaboutMergeScenario(RoundaboutYieldToInsideVehicleScenario):
    zone_name = "roundabout"
    scenario_name = "roundabout_merge"

    def setup(self):
        self.ego_spawn_offset_m = self.cfg.get("ego_spawn_offset_m", 5.0)
        self.decision_range = self.cfg.get("decision_range_m", 30.0)

        self.randomize_links = self.cfg.get("randomize_links", False)
        self.random = random.Random(self.cfg.get("random_seed"))
        self.zone_allowed_links = set()
        self.zone_route_links = self.load_zone_route_links()
        self.route_configured = False
        self.route_setup_mode = None
        self.random_route_pool = None
        self.recent_start_links = []
        self.recent_route_keys = []
        self.npc_vehicles = []
        self.roundabout_inside_vehicle = None
        self.last_roundabout_entry_route = None

        self.select_route_for_next_drive()
        self.prepare_roundabout_merge_npc_trigger()
        self.grpc.start_world(self.start_tf)
        self.configure_drive_with_retries()

    def restart_to_start_and_drive(self):
        print("[RoundaboutMerge] restart ego; keep roundabout NPC traffic running")
        self.stop_gt_bev_expert_controller()
        self.select_route_for_next_drive()
        self.prepare_roundabout_merge_npc_trigger()
        time.sleep(0.2)
        self.configure_drive_with_retries()

    def prepare_roundabout_merge_npc_trigger(self):
        trigger_links = list(
            self.cfg.get(
                "npc_spawn_trigger_links",
                ["A219BS010480", "A219BS010477", "A219BS010478"],
            )
        )
        trigger_distance_m = float(self.cfg.get("npc_spawn_before_roundabout_m", 25.0))
        self.roundabout_merge_npc_trigger_links = {str(link_id) for link_id in trigger_links}

        entry_s = self.route_length_m
        accumulated = 0.0
        found_link = None
        for link_id in self.route_links:
            if link_id in trigger_links:
                entry_s = accumulated
                found_link = link_id
                break
            accumulated += polyline_length(self.map_loader.get_link_points(link_id))

        self.roundabout_merge_npc_spawn_entry_s = entry_s
        self.roundabout_merge_npc_spawn_trigger_s = max(0.0, entry_s - trigger_distance_m)
        if not hasattr(self, "last_npc_topup_time"):
            self.last_npc_topup_time = 0.0
        if not hasattr(self, "last_npc_cleanup_time"):
            self.last_npc_cleanup_time = 0.0
        print(
            f"[RoundaboutMerge] npc spawn armed: trigger_link={found_link}, "
            f"entry_s={entry_s:.1f}m, trigger_s={self.roundabout_merge_npc_spawn_trigger_s:.1f}m, "
            f"before={trigger_distance_m:.1f}m"
        )

    def distance_to_link_polyline(self, x, y, link_id):
        points = self.map_loader.get_link_points(link_id)
        if len(points) < 2:
            return float("inf")

        best_dist = float("inf")
        for p0, p1 in zip(points[:-1], points[1:]):
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-9:
                continue
            t = ((x - p0[0]) * dx + (y - p0[1]) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            proj_x = p0[0] + t * dx
            proj_y = p0[1] + t * dy
            best_dist = min(best_dist, dist_xy(x, y, proj_x, proj_y))
        return best_dist

    def distance_to_roundabout_trigger_links(self, x, y):
        trigger_links = getattr(self, "roundabout_merge_npc_trigger_links", set())
        best_dist = float("inf")
        best_link = None
        for link_id in trigger_links:
            if link_id not in self.map_loader.link_set:
                continue
            distance_m = self.distance_to_link_polyline(x, y, link_id)
            if distance_m < best_dist:
                best_dist = distance_m
                best_link = link_id
        return best_dist, best_link

    def clear_roundabout_npcs(self):
        for npc_info in getattr(self, "npc_vehicles", []):
            actor = npc_info.get("actor")
            if actor is not None and hasattr(actor, "destroy"):
                try:
                    actor.destroy()
                except Exception as e:
                    print(f"[RoundaboutMerge] npc destroy failed label={npc_info.get('label')}: {e}")
        self.npc_vehicles = []
        self.roundabout_inside_vehicle = None

    def cleanup(self):
        super().cleanup()
        self.clear_roundabout_npcs()

    def weighted_choice(self, items):
        total = sum(max(0.0, float(item.get("weight", 1.0))) for item in items)
        if total <= 0.0:
            return self.random.choice(items)

        pick = self.random.uniform(0.0, total)
        acc = 0.0
        for item in items:
            acc += max(0.0, float(item.get("weight", 1.0)))
            if pick <= acc:
                return item
        return items[-1]

    def build_npc_route_from_template(self, template, start_link):
        route_links = list(template.get("route_links", []))
        if len(route_links) < 2:
            return []

        if bool(template.get("circular", False)):
            laps = int(template.get("route_laps", self.cfg.get("npc_route_laps", 1)))
            return self.route_from_roundabout_start_link(route_links, start_link, laps)

        if start_link in route_links:
            return route_links[route_links.index(start_link):]
        return route_links

    def load_roundabout_link_groups(self):
        path = self.cfg.get("zone_links_path", "scenario_runner/config/roundabout_links.yaml")
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        zone_links_key = self.cfg.get("zone_links_key", self.zone_name)
        return data.get(zone_links_key, {})

    def flatten_link_group(self, value):
        if value is None:
            return []
        if isinstance(value, dict):
            links = []
            for item in value.values():
                links.extend(self.flatten_link_group(item))
            return links
        if isinstance(value, (list, tuple)):
            links = []
            for item in value:
                links.extend(self.flatten_link_group(item))
            return links
        return [str(value)]

    def unique_valid_links(self, links):
        valid = []
        for link_id in links:
            link_id = str(link_id)
            if link_id in self.map_loader.link_set and link_id not in valid:
                valid.append(link_id)
        return valid

    def setup_roundabout_dynamic_npc_routes(self):
        groups = self.load_roundabout_link_groups()
        self.roundabout_route_allowed_links = set(
            self.unique_valid_links(groups.get("route_links", []))
        )
        self.roundabout_internal_links = self.unique_valid_links(
            self.cfg.get("npc_internal_links", self.flatten_link_group(groups.get("internal_links", [])))
        )
        self.roundabout_entry_links = self.unique_valid_links(
            self.cfg.get("npc_entry_links", self.flatten_link_group(groups.get("entry_links", [])))
        )
        self.roundabout_exit_links = self.unique_valid_links(
            self.cfg.get("npc_exit_links", self.flatten_link_group(groups.get("exit_links", [])))
        )
        self.roundabout_dynamic_route_cache = {}
        print(
            f"[RoundaboutMerge] dynamic npc route pools: "
            f"entry={len(self.roundabout_entry_links)}, internal={len(self.roundabout_internal_links)}, "
            f"exit={len(self.roundabout_exit_links)}, allowed={len(self.roundabout_route_allowed_links)}"
        )

    def ego_route_avoid_links_for_npc(self):
        avoid = {str(getattr(self, "start_link", ""))}
        ahead_count = int(self.cfg.get("npc_avoid_ego_ahead_link_count", 2))
        for link_id in list(getattr(self, "route_links", []))[: max(1, ahead_count + 1)]:
            avoid.add(str(link_id))
        return {link_id for link_id in avoid if link_id}

    def choose_dynamic_npc_end_link(self, used_end_links):
        exits = [link for link in self.roundabout_exit_links if link in self.map_loader.link_set]
        if not exits:
            return None

        unused = [link for link in exits if link not in used_end_links]
        pool = unused if unused else exits
        return self.random.choice(pool)

    def build_dynamic_npc_route(self, start_link, end_link):
        cache_key = (start_link, end_link)
        if cache_key in getattr(self, "roundabout_dynamic_route_cache", {}):
            return self.roundabout_dynamic_route_cache[cache_key]

        try:
            route_links, route_length_m = build_route_between(self.map_loader, start_link, end_link)
        except Exception:
            result = ([], 0.0)
            self.roundabout_dynamic_route_cache[cache_key] = result
            return result

        if len(route_links) < 2:
            result = ([], 0.0)
            self.roundabout_dynamic_route_cache[cache_key] = result
            return result

        allowed_links = set(getattr(self, "roundabout_route_allowed_links", set()))
        if allowed_links and any(link_id not in allowed_links for link_id in route_links):
            result = ([], 0.0)
            self.roundabout_dynamic_route_cache[cache_key] = result
            return result

        min_length_m = float(self.cfg.get("npc_route_min_length_m", 25.0))
        max_length_m = float(self.cfg.get("npc_route_max_length_m", 180.0))
        if route_length_m < min_length_m or route_length_m > max_length_m:
            result = ([], 0.0)
            self.roundabout_dynamic_route_cache[cache_key] = result
            return result

        max_links = int(self.cfg.get("npc_route_max_links", 8))
        if max_links > 0 and len(route_links) > max_links:
            result = ([], 0.0)
            self.roundabout_dynamic_route_cache[cache_key] = result
            return result

        result = (route_links, route_length_m)
        self.roundabout_dynamic_route_cache[cache_key] = result
        return result

    def build_dynamic_npc_route_candidates(self):
        starts = self.unique_valid_links(self.roundabout_internal_links + self.roundabout_entry_links)
        exits = self.unique_valid_links(self.roundabout_exit_links)
        candidates = []
        for start_link in starts:
            for end_link in exits:
                if start_link == end_link:
                    continue
                route_links, route_length_m = self.build_dynamic_npc_route(start_link, end_link)
                if not route_links:
                    continue
                candidates.append(
                    {
                        "start_link": start_link,
                        "end_link": end_link,
                        "route_links": route_links,
                        "route_length_m": route_length_m,
                    }
                )
        self.random.shuffle(candidates)
        print(f"[RoundaboutMerge] dynamic npc valid route candidates={len(candidates)}")
        return candidates

    def choose_dynamic_npc_speed(self, start_link):
        if start_link in getattr(self, "roundabout_internal_links", []):
            speed_min = float(self.cfg.get("npc_internal_speed_min_mps", 2.0))
            speed_max = float(self.cfg.get("npc_internal_speed_max_mps", 4.0))
        else:
            speed_min = float(self.cfg.get("npc_entry_speed_min_mps", self.cfg.get("npc_speed_min_mps", 3.0)))
            speed_max = float(self.cfg.get("npc_entry_speed_max_mps", self.cfg.get("npc_speed_max_mps", 6.0)))
        if speed_max < speed_min:
            speed_max = speed_min
        return self.random.uniform(speed_min, speed_max)

    def choose_dynamic_npc_candidate(self, spawned_positions, used_start_links, used_end_links, route_candidates):
        min_spacing = float(self.cfg.get("npc_spawn_min_spacing_m", 10.0))
        min_link_spacing = float(self.cfg.get("npc_spawn_min_link_spacing_m", min_spacing))
        attempts = int(self.cfg.get("npc_dynamic_route_attempts", self.cfg.get("npc_spawn_attempts", 40)))
        avoid_links = self.ego_route_avoid_links_for_npc()

        candidates = [
            candidate
            for candidate in route_candidates
            if candidate["start_link"] not in avoid_links
        ]
        if not candidates:
            return None

        internal_probability = float(self.cfg.get("npc_internal_spawn_probability", 0.45))
        internal_links = set(getattr(self, "roundabout_internal_links", []))
        if internal_links:
            if self.random.random() < internal_probability:
                category_candidates = [c for c in candidates if c["start_link"] in internal_links]
            else:
                category_candidates = [c for c in candidates if c["start_link"] not in internal_links]
            if category_candidates:
                candidates = category_candidates

        self.random.shuffle(candidates)
        candidates.sort(key=lambda item: used_end_links.get(item["end_link"], 0))

        for candidate in candidates[:attempts]:
            start_link = candidate["start_link"]
            route_links = candidate["route_links"]
            route_length_m = candidate["route_length_m"]

            pose_attempts = int(self.cfg.get("npc_spawn_pose_attempts_per_route", 5))
            pose = None
            for _ in range(max(1, pose_attempts)):
                candidate_pose = self.choose_spawn_pose_for_template({}, start_link)
                if candidate_pose is None:
                    continue
                _, x, y, _, _ = candidate_pose
                if any(dist_xy(x, y, px, py) < min_spacing for px, py in spawned_positions):
                    continue
                if any(
                    link_id == start_link and dist_xy(x, y, px, py) < min_link_spacing
                    for link_id, px, py in used_start_links.values()
                ):
                    continue
                pose = candidate_pose
                break
            if pose is None:
                continue

            spawn_offset, x, y, z, yaw = pose

            return {
                "name": "dynamic_roundabout_route",
                "start_link": start_link,
                "end_link": candidate["end_link"],
                "spawn_offset": spawn_offset,
                "x": x,
                "y": y,
                "z": z,
                "yaw": yaw,
                "route_links": route_links,
                "route_length_m": route_length_m,
                "speed": self.choose_dynamic_npc_speed(start_link),
            }

        return None

    def choose_spawn_pose_for_template(self, template, start_link):
        points = self.map_loader.get_link_points(start_link)
        link_len = polyline_length(points)
        if link_len <= 0.0:
            return None

        offset_min = float(template.get("spawn_offset_min_m", self.cfg.get("npc_spawn_offset_min_m", 3.0)))
        offset_max = float(template.get("spawn_offset_max_m", self.cfg.get("npc_spawn_offset_max_m", max(3.0, link_len - 3.0))))
        low = min(max(0.0, offset_min), max(0.0, link_len - 3.0))
        high = min(max(low, offset_max), max(0.0, link_len - 3.0))
        offset = self.random.uniform(low, high)
        x, y, z, yaw = interpolate_on_polyline(points, offset)
        return offset, x, y, z, yaw

    def destroy_roundabout_merge_npc(self, npc_info, reason):
        actor = npc_info.get("actor")
        label = npc_info.get("label", "npc_roundabout_merge")
        if actor is not None and hasattr(actor, "destroy"):
            try:
                actor.destroy()
            except Exception as e:
                print(f"[RoundaboutMerge] npc destroy failed label={label}: {e}")
        npc_info["destroyed"] = True
        print(f"[RoundaboutMerge] npc removed label={label}, reason={reason}")

    def cleanup_completed_roundabout_merge_npcs(self):
        if not getattr(self, "npc_vehicles", None):
            return

        arrival_distance_m = float(self.cfg.get("npc_cleanup_arrival_distance_m", 8.0))
        active = []
        for npc_info in self.npc_vehicles:
            if npc_info.get("destroyed"):
                continue

            actor = npc_info.get("actor")
            route_points = npc_info.get("route_points") or []
            route_length_m = float(npc_info.get("route_length_m", 0.0))
            if actor is None or not route_points or route_length_m <= 0.0:
                active.append(npc_info)
                continue

            try:
                state = actor.get_actor_state()
            except Exception as e:
                print(f"[RoundaboutMerge] npc state failed label={npc_info.get('label')}: {e}")
                active.append(npc_info)
                continue

            if state is None:
                active.append(npc_info)
                continue

            x = float(state.transform.location.x)
            y = float(state.transform.location.y)
            route_s = project_distance_on_polyline(route_points, x, y)
            remaining = max(0.0, route_length_m - route_s)
            if remaining <= arrival_distance_m:
                self.destroy_roundabout_merge_npc(
                    npc_info,
                    f"route complete remaining={remaining:.1f}m",
                )
                continue

            active.append(npc_info)

        self.npc_vehicles = active

    def on_gt_bev_timeline_tick(self, **kwargs):
        current_s = float(kwargs.get("current_s", 0.0))
        now = time.time()

        topup_interval = float(
            self.cfg.get("npc_topup_check_sec", self.cfg.get("npc_cleanup_check_sec", 0.2))
        )
        if now - getattr(self, "last_npc_topup_time", 0.0) >= topup_interval:
            self.last_npc_topup_time = now
            min_count = int(self.cfg.get("npc_count_min", self.cfg.get("npc_count", 4)))
            alive_count = len(
                [npc for npc in getattr(self, "npc_vehicles", []) if not npc.get("destroyed")]
            )
            if alive_count < min_count:
                trigger_s = float(getattr(self, "roundabout_merge_npc_spawn_trigger_s", 0.0))
                ego_state = kwargs.get("ego_state", {}) or {}
                ego_link = str(ego_state.get("current_link", ""))
                ego_x = float(ego_state.get("x", 0.0))
                ego_y = float(ego_state.get("y", 0.0))
                proximity_m = float(self.cfg.get("npc_spawn_proximity_m", 12.0))
                trigger_distance_m, trigger_link = self.distance_to_roundabout_trigger_links(ego_x, ego_y)
                reached_by_s = bool(self.cfg.get("npc_spawn_use_route_s_trigger", False)) and current_s >= trigger_s
                reached_by_link = ego_link in getattr(self, "roundabout_merge_npc_trigger_links", set())
                reached_by_distance = trigger_distance_m <= proximity_m

                if reached_by_s or reached_by_link or reached_by_distance:
                    print(
                        f"[RoundaboutMerge] topping up npc traffic (alive={alive_count}, min={min_count}) "
                        f"s={current_s:.1f}m trigger_s={trigger_s:.1f}m "
                        f"ego_link={ego_link} nearest_trigger_link={trigger_link} "
                        f"dist={trigger_distance_m:.1f}m proximity={proximity_m:.1f}m"
                    )
                    self.spawn_roundabout_merge_traffic()

        interval = float(self.cfg.get("npc_cleanup_check_sec", 0.2))
        if now - getattr(self, "last_npc_cleanup_time", 0.0) < interval:
            return
        self.last_npc_cleanup_time = now
        self.cleanup_completed_roundabout_merge_npcs()

    def existing_roundabout_npc_state(self, alive_npcs):
        spawned_positions = []
        used_start_links = {}
        used_end_links = {}
        for npc_info in alive_npcs:
            x, y = npc_info.get("last_xy", (0.0, 0.0))
            actor = npc_info.get("actor")
            if actor is not None:
                try:
                    state = actor.get_actor_state()
                    if state is not None:
                        x = float(state.transform.location.x)
                        y = float(state.transform.location.y)
                except Exception:
                    pass
            spawned_positions.append((x, y))

            start_link = npc_info.get("start_link")
            if start_link:
                used_start_links[npc_info.get("label", str(id(npc_info)))] = (start_link, x, y)

            end_link = npc_info.get("end_link")
            if end_link:
                used_end_links[end_link] = used_end_links.get(end_link, 0) + 1

        return spawned_positions, used_start_links, used_end_links

    def spawn_roundabout_merge_traffic(self):
        alive_npcs = [npc for npc in getattr(self, "npc_vehicles", []) if not npc.get("destroyed")]
        min_count = int(self.cfg.get("npc_count_min", self.cfg.get("npc_count", 4)))
        max_count = int(self.cfg.get("npc_count_max", min_count))
        if max_count < min_count:
            max_count = min_count
        desired_total = self.random.randint(min_count, max_count)
        target_count = max(0, desired_total - len(alive_npcs))
        if target_count == 0:
            print(
                f"[RoundaboutMerge] npc fleet already at target "
                f"(alive={len(alive_npcs)}, desired={desired_total}); skip spawn"
            )
            return

        use_dynamic_routes = bool(self.cfg.get("npc_dynamic_routes", True))
        templates = list(self.cfg.get("npc_route_templates", []))
        if not use_dynamic_routes and not templates:
            raise RuntimeError("roundabout_merge requires npc_route_templates")
        route_candidates = []
        if use_dynamic_routes:
            self.setup_roundabout_dynamic_npc_routes()
            route_candidates = self.build_dynamic_npc_route_candidates()
            if not route_candidates:
                raise RuntimeError("roundabout_merge found no dynamic NPC route candidates")

        min_spacing = float(self.cfg.get("npc_spawn_min_spacing_m", 10.0))
        spawn_attempts = int(self.cfg.get("npc_spawn_attempts", 30))
        global_speed_min = float(self.cfg.get("npc_speed_min_mps", 3.0))
        global_speed_max = float(self.cfg.get("npc_speed_max_mps", 6.0))
        ai_enabled = bool(self.cfg.get("npc_ai", True))
        ego_start_link = str(getattr(self, "start_link", ""))
        unique_templates = bool(self.cfg.get("npc_unique_route_templates", True))
        available_templates = list(templates)
        spawned_positions, used_start_links, used_end_links = self.existing_roundabout_npc_state(alive_npcs)
        failed_route_keys = set()
        failed_spawn_cells = set()

        spawned = 0
        print(
            f"[RoundaboutMerge] topping up npc traffic target_count={target_count} "
            f"(alive={len(alive_npcs)}, desired_total={desired_total}), "
            f"mode={'dynamic' if use_dynamic_routes else 'template'}, exclude_ego_start_link={ego_start_link}"
        )
        for idx in range(target_count):
            npc = None
            model = None
            chosen = None
            start_link = None
            spawn_offset = 0.0
            x = y = z = yaw = 0.0
            route_links = []
            speed = global_speed_min
            route_ok = False
            final_label = f"npc_roundabout_merge_{idx}"

            for attempt_idx in range(spawn_attempts):
                if use_dynamic_routes:
                    candidate = self.choose_dynamic_npc_candidate(
                        spawned_positions,
                        used_start_links,
                        used_end_links,
                        route_candidates,
                    )
                    if candidate is None:
                        continue
                    template = {"name": candidate["name"], "speed_limit_mps": candidate["speed"]}
                    candidate_start = candidate["start_link"]
                    candidate_offset = candidate["spawn_offset"]
                    candidate_x = candidate["x"]
                    candidate_y = candidate["y"]
                    candidate_z = candidate["z"]
                    candidate_yaw = candidate["yaw"]
                    candidate_route = candidate["route_links"]
                    candidate_speed = candidate["speed"]
                    candidate_route_length_m = candidate["route_length_m"]
                else:
                    template_pool = available_templates if unique_templates and available_templates else templates
                    template = self.weighted_choice(template_pool)
                    template_route_links = list(template.get("route_links", []))
                    start_links = list(template.get("start_links", [])) or template_route_links[:1]
                    start_links = [link for link in start_links if str(link) != ego_start_link]
                    if not start_links:
                        continue

                    candidate_start = self.random.choice(start_links)
                    if candidate_start not in self.map_loader.link_set:
                        continue

                    pose = self.choose_spawn_pose_for_template(template, candidate_start)
                    if pose is None:
                        continue

                    candidate_offset, candidate_x, candidate_y, candidate_z, candidate_yaw = pose
                    if any(dist_xy(candidate_x, candidate_y, px, py) < min_spacing for px, py in spawned_positions):
                        continue

                    candidate_route = self.build_npc_route_from_template(template, candidate_start)
                    if len(candidate_route) < 2:
                        continue

                    speed_min = float(template.get("speed_min_mps", global_speed_min))
                    speed_max = float(template.get("speed_max_mps", global_speed_max))
                    if speed_max < speed_min:
                        speed_max = speed_min
                    candidate_speed = self.random.uniform(speed_min, speed_max)
                    candidate_route_length_m = 0.0

                route_key = tuple(candidate_route)
                if route_key in failed_route_keys:
                    continue

                spawn_cell = (
                    candidate_start,
                    int(candidate_offset // max(1.0, min_spacing)),
                )
                if spawn_cell in failed_spawn_cells:
                    continue

                label = f"npc_roundabout_merge_{idx}_try_{attempt_idx}"
                candidate_npc, candidate_model = self.spawn_vehicle_with_model_retry(
                    self.grpc.make_transform(candidate_x, candidate_y, candidate_z, candidate_yaw),
                    label,
                    0.0,
                )
                if candidate_npc is None:
                    failed_spawn_cells.add(spawn_cell)
                    time.sleep(float(self.cfg.get("npc_spawn_retry_wait_sec", 0.05)))
                    continue

                candidate_route_ok = self.grpc.set_vehicle_route(
                    candidate_npc,
                    candidate_route,
                    decision_range=self.decision_range,
                    label=label,
                )
                if not candidate_route_ok:
                    failed_route_keys.add(route_key)
                    print(
                        f"[RoundaboutMerge] route failed; destroy npc label={label}, "
                        f"template={template.get('name', 'unnamed')}, start_link={candidate_start}, "
                        f"route_links={list(candidate_route)}"
                    )
                    if hasattr(candidate_npc, "destroy"):
                        try:
                            if hasattr(candidate_npc, "set_pause"):
                                candidate_npc.set_pause(True)
                            candidate_npc.destroy()
                        except Exception as e:
                            print(f"[RoundaboutMerge] route-failed npc destroy failed label={label}: {e}")
                    time.sleep(float(self.cfg.get("npc_route_failed_destroy_wait_sec", 0.2)))
                    continue

                npc = candidate_npc
                model = candidate_model
                chosen = template
                final_label = label
                start_link = candidate_start
                spawn_offset = candidate_offset
                x, y, z, yaw = candidate_x, candidate_y, candidate_z, candidate_yaw
                route_links = candidate_route
                speed = candidate_speed
                route_ok = candidate_route_ok
                if use_dynamic_routes:
                    used_start_links[f"{start_link}:{idx}"] = (start_link, x, y)
                    used_end_links[route_links[-1]] = used_end_links.get(route_links[-1], 0) + 1
                if not use_dynamic_routes and unique_templates and template in available_templates:
                    available_templates.remove(template)
                break

            if npc is None:
                print(f"[RoundaboutMerge] failed to spawn route-valid npc index={idx}")
                continue

            npc.set_pause(False)
            if hasattr(self.grpc, "set_vehicle_speed_limit"):
                self.grpc.set_vehicle_speed_limit(npc, float(chosen.get("speed_limit_mps", speed)), enabled=True)
            self.grpc.set_vehicle_velocity(npc, speed)
            self.grpc.set_vehicle_ai(npc, ai_enabled)

            npc_info = self.register_roundabout_npc(
                npc,
                final_label,
                model,
                x,
                y,
                speed,
                route_ok=route_ok,
            )
            route_points = self.build_route_points(route_links)
            npc_info["start_link"] = start_link
            npc_info["route_links"] = list(route_links)
            npc_info["route_points"] = route_points
            npc_info["route_length_m"] = candidate_route_length_m or polyline_length(route_points)
            npc_info["end_link"] = route_links[-1] if route_links else ""
            spawned_positions.append((x, y))
            spawned += 1

            print(
                f"[RoundaboutMerge] npc spawned label={final_label}, "
                f"template={chosen.get('name', 'unnamed')}, model={model}, "
                f"start_link={start_link}, offset={spawn_offset:.1f}m, "
                f"end_link={route_links[-1] if route_links else ''}, "
                f"route_links={len(route_links)}, route={route_ok}, speed={speed:.1f}m/s"
            )

        if spawned == 0 and not alive_npcs:
            raise RuntimeError("Failed to spawn roundabout merge NPC traffic")

        print(f"[RoundaboutMerge] npc traffic topped up {spawned}/{target_count} (alive={len(alive_npcs) + spawned})")


class UrbanTrafficJamScenario(UrbanSuddenBrakeExpertScenario):
    zone_name = "urban"
    scenario_name = "traffic_jam"

    def reset_brake_event_state(self):
        self.brake_event = {
            "phase": "stopped",
            "cycles_completed": 0,
            "phase_started_at": time.time(),
            "last_transition": "initial_jam",
        }

    def spawn_npc_vehicles(self):
        self.jam_npcs = []
        self.npc_vehicles = []

        queue_count = self.random.randint(
            int(self.cfg.get("jam_queue_count_min", 3)),
            int(self.cfg.get("jam_queue_count_max", 5)),
        )
        background_count = self.random.randint(
            int(self.cfg.get("jam_background_npc_min", 0)),
            int(self.cfg.get("jam_background_npc_max", 0)),
        )

        print(
            f"[UrbanTrafficJam] spawning jam_queue={queue_count}, "
            f"background={background_count}"
        )
        self.spawn_jam_queue(queue_count)

        for i in range(background_count):
            if not self.spawn_random_ai_npc(label=f"npc_jam_bg_{i}"):
                print(f"[UrbanTrafficJam] npc_jam_bg_{i} spawn skipped")

    def spawn_jam_queue(self, count):
        base_speed = float(self.cfg.get("npc_speed_mps", 7.0))
        speed_jitter = float(self.cfg.get("npc_speed_jitter_mps", 0.0))
        start_gap = float(self.cfg.get("jam_queue_start_gap_m", 20.0))
        spacing = float(self.cfg.get("jam_queue_spacing_m", 7.0))

        for i in range(count):
            spawn_s = min(
                float(self.ego_spawn_offset_m) + start_gap + i * spacing,
                max(float(self.ego_spawn_offset_m) + 8.0, self.route_length_m - 8.0),
            )
            x, y, z, yaw = interpolate_on_polyline(self.route_points, spawn_s)
            speed = max(0.5, base_speed + self.random.uniform(-speed_jitter, speed_jitter))
            label = f"npc_jam_queue_{i}"
            npc, model = self.spawn_vehicle_with_model_retry(
                self.grpc.make_transform(x, y, z, yaw),
                label,
                speed,
            )
            if npc is None:
                print(f"[UrbanTrafficJam] {label} spawn failed")
                continue

            route_ok = False
            if self.cfg.get("npc_use_route", False):
                route_ok = self.grpc.set_vehicle_route(
                    npc,
                    self.route_links,
                    decision_range=self.decision_range,
                    label=label,
                )

            self.configure_ai_npc(npc, speed)
            self.grpc.stop_vehicle(npc)
            self.register_npc(npc, label, model, x, y, speed, route_ok=route_ok)
            npc_info = self.npc_vehicles[-1]
            npc_info["jam_queue"] = True
            self.jam_npcs.append(npc_info)
            print(
                f"[UrbanTrafficJam] jam npc stopped label={label}, model={model}, "
                f"spawn_s={spawn_s:.1f}m, route={route_ok}"
            )

    def pause_jam_npcs(self):
        for npc_info in getattr(self, "jam_npcs", []):
            self.grpc.stop_vehicle(npc_info["actor"])
        print(f"[UrbanTrafficJam] jam STOP vehicles={len(getattr(self, 'jam_npcs', []))}")

    def resume_jam_npcs(self):
        for npc_info in getattr(self, "jam_npcs", []):
            self.grpc.resume_vehicle_ai(npc_info["actor"])
            npc_info["next_state_check_time"] = time.time() + 0.5
        print(f"[UrbanTrafficJam] jam GO vehicles={len(getattr(self, 'jam_npcs', []))}")

    def update_npc_states(self, ego_x, ego_y, ego_yaw_deg, ego_current_link=None):
        if self.brake_event.get("phase") != "stopped":
            return super().update_npc_states(ego_x, ego_y, ego_yaw_deg, ego_current_link=ego_current_link)

        yaw_rad = math.radians(ego_yaw_deg)
        forward_x = math.cos(yaw_rad)
        forward_y = math.sin(yaw_rad)
        closest = None

        for npc_info in getattr(self, "jam_npcs", []):
            npc_x, npc_y = npc_info["last_xy"]
            dx = npc_x - ego_x
            dy = npc_y - ego_y
            distance_m = math.hypot(dx, dy)
            ahead_m = dx * forward_x + dy * forward_y
            lateral_m = abs(-dx * forward_y + dy * forward_x)
            npc_info.update(
                {
                    "distance_m": distance_m,
                    "ahead_m": ahead_m,
                    "lateral_m": lateral_m,
                    "speed_mps": 0.0,
                    "heading_cos": 1.0,
                }
            )

            if ahead_m < 0.0:
                continue
            if closest is None or distance_m < closest["distance_m"]:
                closest = npc_info

        return closest

    def maybe_update_brake_event(self, elapsed, ego_speed_mps, target):
        now = time.time()
        phase = self.brake_event.get("phase", "stopped")
        phase_duration = now - float(self.brake_event.get("phase_started_at", now))
        cycles_completed = int(self.brake_event.get("cycles_completed", 0))
        max_cycles = int(self.cfg.get("jam_cycles_per_episode", 2))

        if phase == "stopped":
            release_gap = float(self.cfg.get("jam_release_gap_m", 18.0))
            force_release_sec = float(self.cfg.get("jam_force_release_sec", 8.0))
            stop_duration_sec = float(self.cfg.get("jam_stop_duration_sec", 3.5))
            ego_close = (
                target is not None
                and target.get("distance_m", float("inf")) <= release_gap
                and target.get("ahead_m", -float("inf")) >= 0.0
            )
            enough_wait = (
                elapsed >= force_release_sec
                if cycles_completed == 0
                else phase_duration >= stop_duration_sec
            )

            if ego_close or enough_wait:
                self.resume_jam_npcs()
                self.brake_event.update(
                    {
                        "phase": "released",
                        "cycles_completed": cycles_completed + 1,
                        "phase_started_at": now,
                        "last_transition": "go_close" if ego_close else "go_wait",
                    }
                )
            return

        if phase == "released":
            go_duration_sec = float(self.cfg.get("jam_go_duration_sec", 6.0))
            if cycles_completed < max_cycles and phase_duration >= go_duration_sec:
                self.pause_jam_npcs()
                self.brake_event.update(
                    {
                        "phase": "stopped",
                        "phase_started_at": now,
                        "last_transition": "stop_again",
                    }
                )


class UrbanPedestrianYieldScenario(UrbanBasicDriveScenario):
    zone_name = "urban"
    scenario_name = "pedestrian_yield"

    def build_random_route_pool(self):
        pool = super().build_random_route_pool()
        filtered = []
        for item in pool:
            route_points = self.build_route_points(item["route_links"])
            crosswalk = self._find_route_crosswalk_spawn(
                route_points=route_points,
                route_length_m=item["route_length_m"],
            )
            if crosswalk is None:
                continue
            item = dict(item)
            item["crosswalk_spawn"] = crosswalk
            filtered.append(item)

        if not filtered:
            raise RuntimeError("No pedestrian_yield routes pass near singlecrosswalk_set.json crosswalks")

        print(
            f"[UrbanPedestrianYield] route pool with crosswalks: "
            f"{len(filtered)}/{len(pool)}"
        )
        return filtered

    def setup(self):
        super().setup()
        self._load_crosswalk_spawn_info()
        self._reset_pedestrian_event()
        if self.cfg.get("pedestrian_yield_stop_enabled", True):
            try:
                self.init_ros_ctrl_cmd_publisher()
            except Exception as e:
                print(f"[UrbanPedestrianYield] ROS brake override unavailable: {e}")

    def restart_to_start_and_drive(self):
        self._despawn_pedestrian()
        super().restart_to_start_and_drive()
        self._load_crosswalk_spawn_info()
        self._reset_pedestrian_event()

    def _reset_pedestrian_event(self):
        self.pedestrian = None
        self.pedestrian_phase = "waiting"
        self.pedestrian_started_at = None
        self._pedestrian_speed = float(self.cfg.get("pedestrian_speed_mps", 1.2))
        self._pedestrian_model = None
        self._pedestrian_stop_active = False

    def _route_projection_for_points(self, route_points, x, y):
        if len(route_points) < 2:
            return 0.0, float("inf")

        best_s = 0.0
        best_dist = float("inf")
        cumulative = 0.0
        for p0, p1 in zip(route_points[:-1], route_points[1:]):
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-9:
                continue

            seg_len = math.sqrt(seg_len_sq)
            t = ((x - p0[0]) * dx + (y - p0[1]) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            proj_x = p0[0] + t * dx
            proj_y = p0[1] + t * dy
            dist = dist_xy(x, y, proj_x, proj_y)
            if dist < best_dist:
                best_dist = dist
                best_s = cumulative + t * seg_len
            cumulative += seg_len
        return best_s, best_dist

    def _find_route_crosswalk_spawn(self, route_points=None, route_length_m=None):
        route_points = route_points or getattr(self, "route_points", [])
        route_length_m = route_length_m if route_length_m is not None else getattr(self, "route_length_m", 0.0)
        if not getattr(self.map_loader, "singlecrosswalk_set", None):
            return None

        max_dist_m = float(self.cfg.get("pedestrian_route_crosswalk_max_dist_m", 8.0))
        min_s_m = float(self.cfg.get("pedestrian_crosswalk_min_s_m", 20.0))
        end_margin_m = float(self.cfg.get("pedestrian_crosswalk_end_margin_m", 20.0))

        candidates = []
        for scw_id in self.map_loader.singlecrosswalk_set:
            try:
                sx, sy, ex, ey, sz, dir_x, dir_y, move_dist = (
                    self.map_loader.get_singlecrosswalk_spawn_info(scw_id)
                )
            except Exception:
                continue

            cx = (sx + ex) * 0.5
            cy = (sy + ey) * 0.5
            route_s, route_dist = self._route_projection_for_points(route_points, cx, cy)
            if route_dist > max_dist_m:
                continue
            if route_s < min_s_m:
                continue
            if route_length_m > 0.0 and route_length_m - route_s < end_margin_m:
                continue

            candidates.append(
                {
                    "singlecrosswalk_id": scw_id,
                    "start_x": sx,
                    "start_y": sy,
                    "end_x": ex,
                    "end_y": ey,
                    "z": sz,
                    "dir_x": dir_x,
                    "dir_y": dir_y,
                    "move_dist": move_dist,
                    "route_s": route_s,
                    "route_dist": route_dist,
                }
            )

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item["route_s"], item["route_dist"]))
        return candidates[0]

    def _load_crosswalk_spawn_info(self):
        crosswalk = self._find_route_crosswalk_spawn()
        if crosswalk is None and self.cfg.get("crosswalk_id"):
            crosswalk_id = self.cfg["crosswalk_id"]
            sx, sy, ex, ey, sz, dir_x, dir_y, move_dist = self.map_loader.get_crosswalk_spawn_info(crosswalk_id)
            crosswalk = {
                "singlecrosswalk_id": crosswalk_id,
                "start_x": sx,
                "start_y": sy,
                "end_x": ex,
                "end_y": ey,
                "z": sz,
                "dir_x": dir_x,
                "dir_y": dir_y,
                "move_dist": move_dist,
                "route_s": project_distance_on_polyline(self.route_points, (sx + ex) * 0.5, (sy + ey) * 0.5),
                "route_dist": 0.0,
            }
        if crosswalk is None:
            raise RuntimeError("No usable singlecrosswalk found near selected route")

        sx = crosswalk["start_x"]
        sy = crosswalk["start_y"]
        ex = crosswalk["end_x"]
        ey = crosswalk["end_y"]
        dir_x = crosswalk["dir_x"]
        dir_y = crosswalk["dir_y"]
        ped_yaw = math.degrees(math.atan2(dir_y, dir_x))
        crosswalk["yaw"] = ped_yaw
        self._crosswalk_spawn = crosswalk
        print(
            f"[UrbanPedestrianYield] singlecrosswalk={crosswalk['singlecrosswalk_id']} "
            f"start=({sx:.2f},{sy:.2f}) end=({ex:.2f},{ey:.2f}) "
            f"route_s={crosswalk['route_s']:.1f}m route_dist={crosswalk['route_dist']:.1f}m"
        )

    def _select_pedestrian_model(self):
        models = list(self.cfg.get("pedestrian_models", []))
        if not models and hasattr(self.grpc, "get_available_pedestrian_models"):
            models = self.grpc.get_available_pedestrian_models()
        if not models:
            raise RuntimeError("No pedestrian model available. Set pedestrian_models in config.")
        return self.random.choice(models)

    def spawn_pedestrian_at_crosswalk(self):
        sp = self._crosswalk_spawn
        # sp: start_x/y = 크로스워크 한쪽 끝 중점, end_x/y = 반대쪽 끝 중점
        #     dir_x/y = start→end 방향 단위벡터, move_dist = 크로스워크 폭, z = 높이
        base_speed = float(self.cfg.get("pedestrian_speed_mps", 1.2))
        label = self.cfg.get("pedestrian_label", "pedestrian_yield_target")
        model = self._select_pedestrian_model()
        sidewalk_offset_m = float(self.cfg.get("pedestrian_sidewalk_offset_m", 1.5))

        # 보행자 스폰: start 쪽 인도 (start에서 dir 반대 방향으로 offset)
        spawn_x = sp["start_x"] - sp["dir_x"] * sidewalk_offset_m
        spawn_y = sp["start_y"] - sp["dir_y"] * sidewalk_offset_m
        # 목적지: end 쪽 인도 (end에서 dir 방향으로 offset)
        dest_x = sp["end_x"] + sp["dir_x"] * sidewalk_offset_m
        dest_y = sp["end_y"] + sp["dir_y"] * sidewalk_offset_m
        # 이동 거리 = 크로스워크 폭 + 양쪽 인도 offset
        move_dist = sp["move_dist"] + sidewalk_offset_m * 2
        speed = base_speed
        if self.cfg.get("pedestrian_auto_speed_by_crosswalk", True):
            target_crossing_sec = float(self.cfg.get("pedestrian_target_crossing_sec", 7.0))
            min_speed = float(self.cfg.get("pedestrian_min_speed_mps", 1.0))
            max_speed = float(self.cfg.get("pedestrian_max_speed_mps", 2.4))
            if target_crossing_sec > 0.0:
                speed = max(min_speed, min(max_speed, move_dist / target_crossing_sec))

        ped_yaw = math.degrees(math.atan2(sp["dir_y"], sp["dir_x"]))
        ped_yaw += float(self.cfg.get("pedestrian_yaw_offset_deg", 0.0))

        self._ped_dest_x = dest_x
        self._ped_dest_y = dest_y
        self._ped_spawn_x = spawn_x
        self._ped_spawn_y = spawn_y
        self._ped_move_dist = move_dist
        self._ped_yaw = ped_yaw
        self._ped_direction_checked = False
        self._ped_direction_respawned = False

        print(
            f"[UrbanPedestrianYield] crosswalk start=({sp['start_x']:.2f},{sp['start_y']:.2f}) "
            f"end=({sp['end_x']:.2f},{sp['end_y']:.2f}) dir=({sp['dir_x']:.3f},{sp['dir_y']:.3f})"
        )
        print(
            f"[UrbanPedestrianYield] ped spawn=({spawn_x:.2f},{spawn_y:.2f}) "
            f"dest=({dest_x:.2f},{dest_y:.2f}) yaw={ped_yaw:.1f}deg "
            f"move_dist={move_dist:.2f}m speed={speed:.2f}m/s"
        )

        transform = self.grpc.make_transform(spawn_x, spawn_y, sp["z"], ped_yaw)
        pedestrian = self.grpc.spawn_pedestrian(
            transform,
            model,
            label,
            velocity=speed,
            active_dist=0.0,
            move_dist=move_dist,
            start_action=True,
        )
        if pedestrian is None:
            raise RuntimeError(f"Failed to spawn pedestrian model={model}")

        self.pedestrian = pedestrian
        self.pedestrian_phase = "crossing"
        self.pedestrian_started_at = time.time()
        self._pedestrian_speed = speed
        self._pedestrian_model = model
        print(
            f"[UrbanPedestrianYield] pedestrian crossing START "
            f"singlecrosswalk={sp['singlecrosswalk_id']}"
        )

    def _respawn_pedestrian_with_yaw(self, yaw):
        if self.pedestrian is not None:
            self._despawn_pedestrian()

        transform = self.grpc.make_transform(
            self._ped_spawn_x,
            self._ped_spawn_y,
            self._crosswalk_spawn["z"],
            yaw,
        )
        pedestrian = self.grpc.spawn_pedestrian(
            transform,
            self._pedestrian_model,
            self.cfg.get("pedestrian_label", "pedestrian_yield_target"),
            velocity=float(self._pedestrian_speed),
            active_dist=0.0,
            move_dist=float(self._ped_move_dist),
            start_action=True,
        )
        if pedestrian is None:
            raise RuntimeError(f"Failed to respawn pedestrian model={self._pedestrian_model}")

        self.pedestrian = pedestrian
        self._ped_yaw = yaw
        self.pedestrian_started_at = time.time()
        print(f"[UrbanPedestrianYield] pedestrian direction corrected yaw={yaw:.1f}deg")

    def _maybe_correct_pedestrian_direction(self, ped_x, ped_y, elapsed):
        if self._ped_direction_checked or self._ped_direction_respawned:
            return False
        if elapsed < float(self.cfg.get("pedestrian_direction_check_sec", 0.8)):
            return False

        desired_x = self._ped_dest_x - self._ped_spawn_x
        desired_y = self._ped_dest_y - self._ped_spawn_y
        desired_len = math.hypot(desired_x, desired_y)
        moved_x = ped_x - self._ped_spawn_x
        moved_y = ped_y - self._ped_spawn_y
        moved_len = math.hypot(moved_x, moved_y)
        if desired_len < 1e-3 or moved_len < float(self.cfg.get("pedestrian_direction_check_min_move_m", 0.4)):
            return False

        dot = (desired_x * moved_x + desired_y * moved_y) / (desired_len * moved_len)
        self._ped_direction_checked = True
        if dot >= 0.0:
            return False

        self._ped_direction_respawned = True
        corrected_yaw = float(self._ped_yaw) + 180.0
        print(
            f"[UrbanPedestrianYield] pedestrian moved opposite direction; "
            f"dot={dot:.2f}, respawn with yaw={corrected_yaw:.1f}"
        )
        self._respawn_pedestrian_with_yaw(corrected_yaw)
        return True

    def _despawn_pedestrian(self):
        if hasattr(self, "pedestrian") and self.pedestrian is not None:
            try:
                self.pedestrian.destroy()
            except Exception:
                pass
            self.pedestrian = None

    def _is_vehicle_signal_green(self, color):
        if color is None:
            return False
        # 차량 신호 GREEN 계열: 비트 4 이상 (SG=16, LG=32, RG=64, ...)
        # RED=1, YELLOW=4
        return bool(color & 0xFFF0)

    def _finish_pedestrian_crossing(self, reason, elapsed):
        dest_x = getattr(self, "_ped_dest_x", None)
        dest_y = getattr(self, "_ped_dest_y", None)
        if dest_x is not None and dest_y is not None and hasattr(self.pedestrian, "set_transform"):
            z = self._crosswalk_spawn["z"]
            yaw = float(getattr(self, "_ped_yaw", self._crosswalk_spawn.get("yaw", 0.0)))
            try:
                self.pedestrian.set_transform(self.grpc.make_transform(dest_x, dest_y, z, yaw))
            except Exception:
                pass

        if self.cfg.get("pedestrian_despawn_after_crossing", True):
            self._despawn_pedestrian()

        self.pedestrian_phase = "cleared"
        print(f"[UrbanPedestrianYield] pedestrian crossing DONE reason={reason} elapsed={elapsed:.1f}s")

    def _update_pedestrian(self, ego_x, ego_y, current_s):
        if getattr(self, "pedestrian_phase", "waiting") == "cleared":
            return

        trigger_dist_m = float(self.cfg.get("pedestrian_trigger_dist_m", 35.0))
        now = time.time()

        if self.pedestrian_phase == "waiting":
            crosswalk_s = float(self._crosswalk_spawn["route_s"])
            remaining_to_crosswalk = crosswalk_s - current_s
            if -5.0 <= remaining_to_crosswalk <= trigger_dist_m:
                self.spawn_pedestrian_at_crosswalk()
            return

        if self.pedestrian_phase == "crossing":
            max_crossing_sec = float(self.cfg.get("pedestrian_max_crossing_sec", 15.0))
            elapsed = now - float(self.pedestrian_started_at or now)
            dest_x = getattr(self, "_ped_dest_x", None)
            dest_y = getattr(self, "_ped_dest_y", None)

            reached = False
            if dest_x is not None and dest_y is not None:
                try:
                    ped_state = self.pedestrian.get_actor_state()
                    ped_x = float(ped_state.transform.location.x)
                    ped_y = float(ped_state.transform.location.y)
                    if self._maybe_correct_pedestrian_direction(ped_x, ped_y, elapsed):
                        return
                    dist_to_dest = dist_xy(ped_x, ped_y, dest_x, dest_y)
                    reached = dist_to_dest <= 2.0
                    if int(elapsed) % 2 == 0 and elapsed - int(elapsed) < 0.1:
                        print(
                            f"[UrbanPedestrianYield] crossing pos=({ped_x:.2f},{ped_y:.2f}) "
                            f"dest=({dest_x:.2f},{dest_y:.2f}) dist={dist_to_dest:.2f}m "
                            f"elapsed={elapsed:.1f}s"
                        )
                except Exception as e:
                    if elapsed - float(getattr(self, "_last_ped_state_error_print", -10.0)) >= 2.0:
                        print(f"[UrbanPedestrianYield] get_actor_state failed: {e}")
                        self._last_ped_state_error_print = elapsed
                    reached = elapsed >= max_crossing_sec

            if reached or elapsed >= max_crossing_sec:
                reason = "reached_dest" if reached else "timeout"
                self._finish_pedestrian_crossing(reason, elapsed)

    def _apply_pedestrian_yield_stop(self, current_s):
        if not self.cfg.get("pedestrian_yield_stop_enabled", True):
            return False

        if getattr(self, "pedestrian_phase", "waiting") != "crossing":
            self._pedestrian_stop_active = False
            return False

        crosswalk_s = float(self._crosswalk_spawn["route_s"])
        remaining_to_crosswalk = crosswalk_s - float(current_s)
        stop_distance = float(self.cfg.get("pedestrian_yield_stop_distance_m", 12.0))
        pass_margin = float(self.cfg.get("pedestrian_yield_pass_margin_m", 2.0))

        should_stop = -pass_margin <= remaining_to_crosswalk <= stop_distance
        if not should_stop:
            self._pedestrian_stop_active = False
            return False

        if not self._pedestrian_stop_active:
            print(
                f"[UrbanPedestrianYield] pedestrian yield STOP "
                f"remaining_to_crosswalk={remaining_to_crosswalk:.1f}m"
            )
            self._pedestrian_stop_active = True

        brake = float(self.cfg.get("pedestrian_yield_stop_brake", 1.0))
        if hasattr(self, "ros_ctrl_cmd_pub"):
            self.publish_ros_ctrl_cmd(
                steer=0.0,
                target_speed=0.0,
                current_speed=0.0,
                brake_override=brake,
            )
        elif hasattr(self.grpc, "stop_ego_control"):
            self.grpc.stop_ego_control()
        return True

    def cleanup(self):
        self._despawn_pedestrian()
        super().cleanup()

    def run_gt_bev_expert_timeline(self):
        timeout_sec = self.cfg.get("timeout_sec", 0.0)
        use_timeout = timeout_sec is not None and float(timeout_sec) > 0.0
        goal_tolerance_m = float(self.cfg.get("goal_tolerance_m", 4.0))
        check_period_sec = float(self.cfg.get("check_period_sec", 0.05))
        print_period_sec = float(self.cfg.get("print_period_sec", 1.0))
        max_cross_track_error_m = float(self.cfg.get("gt_bev_max_cross_track_error_m", 10.0))
        arrival_stop_distance_m = float(
            self.cfg.get("gt_bev_arrival_stop_distance_m", max(goal_tolerance_m, 6.0))
        )
        off_route_timeout_sec = float(self.cfg.get("off_route_timeout_sec", 5.0))
        off_route_grace_sec = float(self.cfg.get("off_route_grace_sec", 2.0))
        max_laps = int(self.cfg.get("max_laps", 0))
        print(
            f"[UrbanPedestrianYield] GT_BEV running. "
            f"crosswalk={self._crosswalk_spawn['singlecrosswalk_id']}, "
            f"crosswalk_s={self._crosswalk_spawn['route_s']:.1f}m, "
            f"arrival_stop={arrival_stop_distance_m}m"
        )

        lap = 0
        lap_start_time = time.time()
        last_print_time = 0.0
        off_route_enter_time = None
        self.gt_bev_last_s = 0.0

        while True:
            elapsed = time.time() - lap_start_time
            if not hasattr(self, "gt_bev_expert") or self.gt_bev_expert is None or not self.gt_bev_expert.is_running():
                raise RuntimeError("GT_BEV expert process stopped unexpectedly")

            ego_state = self.grpc.get_ego_motion_state()
            ego_x = ego_state["x"]
            ego_y = ego_state["y"]
            current_link = ego_state.get("current_link", "")
            dist_to_goal = dist_xy(ego_x, ego_y, self.goal_x, self.goal_y)

            current_s, _, _, cross_track_error = self.project_on_route_near_progress(
                ego_x, ego_y, prev_s=getattr(self, "gt_bev_last_s", None),
            )
            current_s = max(getattr(self, "gt_bev_last_s", 0.0), current_s)
            self.gt_bev_last_s = current_s
            remaining_s = max(0.0, self.route_length_m - current_s)

            self._update_pedestrian(ego_x, ego_y, current_s)
            pedestrian_stop_active = self._apply_pedestrian_yield_stop(current_s)

            goal_reached = (
                remaining_s <= arrival_stop_distance_m and cross_track_error <= max_cross_track_error_m
            ) or dist_to_goal <= goal_tolerance_m

            if elapsed >= off_route_grace_sec and cross_track_error > max_cross_track_error_m:
                if off_route_enter_time is None:
                    off_route_enter_time = time.time()
            else:
                off_route_enter_time = None

            if elapsed - last_print_time >= print_period_sec:
                print(
                    f"[UrbanPedestrianYield] lap={lap + 1} t={elapsed:.1f}s "
                    f"speed={ego_state['speed']:.1f}m/s link={current_link} "
                    f"s={current_s:.1f}/{self.route_length_m:.1f} "
                    f"ped={self.pedestrian_phase} stop={pedestrian_stop_active} "
                    f"cte={cross_track_error:.2f} remain={remaining_s:.1f}"
                )
                last_print_time = elapsed

            if (
                off_route_enter_time is not None
                and time.time() - off_route_enter_time >= off_route_timeout_sec
            ):
                print(
                    f"[UrbanPedestrianYield] OFF ROUTE - restart "
                    f"cte={cross_track_error:.2f}m, link={current_link}, elapsed={elapsed:.1f}s"
                )
                self.stop_pure_pursuit_control()
                self.restart_to_start_and_drive()
                lap_start_time = time.time()
                last_print_time = 0.0
                off_route_enter_time = None
                self.gt_bev_last_s = 0.0
                continue

            if goal_reached:
                lap += 1
                print(
                    f"[UrbanPedestrianYield] GOAL REACHED lap={lap}, "
                    f"dist={dist_to_goal:.2f}m, s={current_s:.1f}/{self.route_length_m:.1f}, "
                    f"elapsed={elapsed:.1f}s"
                )
                self.stop_pure_pursuit_control()

                if max_laps > 0 and lap >= max_laps:
                    print("[UrbanPedestrianYield] max_laps reached. finish scenario.")
                    break

                self.notify_lap_end(lap)
                self.restart_to_start_and_drive()
                lap_start_time = time.time()
                last_print_time = 0.0
                off_route_enter_time = None
                self.gt_bev_last_s = 0.0
                continue

            if use_timeout and elapsed >= float(timeout_sec):
                print(
                    f"[UrbanPedestrianYield] TIMEOUT lap={lap + 1}, "
                    f"dist={dist_to_goal:.2f}m, elapsed={elapsed:.1f}s"
                )
                self.stop_pure_pursuit_control()
                break

            time.sleep(check_period_sec)
