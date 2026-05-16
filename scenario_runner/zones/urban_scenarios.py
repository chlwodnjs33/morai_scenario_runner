import random
import time
import yaml
import math

from base_scenario import BaseScenario
from utils.route_utils import build_route_between
from utils.transform_utils import (
    interpolate_on_polyline,
    nearest_point_index,
    polyline_length,
    project_distance_on_polyline,
    dist_xy,
)


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

        zone_data = data.get(self.zone_name, {})
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

    def select_route_for_next_drive(self):
        if self.randomize_links:
            self.select_random_route()
        else:
            self.start_link = self.cfg["start_link"]
            self.end_link = self.cfg["end_link"]
            self.prepare_route()

    def select_random_route(self):
        attempts = int(self.cfg.get("random_route_attempts", 200))
        min_length_m = float(self.cfg.get("random_min_route_length_m", 20.0))
        max_length_m = float(self.cfg.get("random_max_route_length_m", 0.0))
        max_route_links = int(self.cfg.get("random_max_route_links", 0))
        stay_in_zone = self.cfg.get("random_route_stay_in_zone", True)
        zone_link_set = self.zone_allowed_links

        last_error = None
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

            self.start_link = start_link
            self.end_link = end_link
            self.prepare_route(route_links=route_links, route_length_m=route_length_m)
            print(f"[UrbanBasicDrive] random route selected on attempt={attempt}")
            return

        raise RuntimeError(
            "Failed to select connected random urban route. "
            f"attempts={attempts}, last_error={last_error}"
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
        self.route_length_m = route_length_m

        print(f"[UrbanBasicDrive] route_links={len(self.route_links)}, length={self.route_length_m:.1f}m")
        print(f"[UrbanBasicDrive] route_end_link={self.route_links[-1]}")
        print("[UrbanBasicDrive] route:")
        for i, link_id in enumerate(self.route_links):
            print(f"  {i:02d}: {link_id}")

        self.route_waypoint_indices = self.build_route_waypoint_indices()
        print("[UrbanBasicDrive] route waypoints:")
        for link_id, waypoint_idx in zip(self.route_links, self.route_waypoint_indices):
            print(f"  {link_id}: waypoint_idx={waypoint_idx}")

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
        time.sleep(0.2)

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

        self.grpc.set_ego_control_mode_cruise()
        self.grpc.set_ego_cruise(
            enable=True,
            link_speed_ratio=self.cfg.get("link_speed_ratio", 40),
            constant_velocity=self.cfg.get("constant_velocity", 20),
            cruise_type=self.cfg.get("cruise_type", "link"),
        )

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

        route_ok = npc.set_vehicle_route(self.decision_range, self.route_links)
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
                route_ok = npc.set_vehicle_route(self.decision_range, route_links)

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
