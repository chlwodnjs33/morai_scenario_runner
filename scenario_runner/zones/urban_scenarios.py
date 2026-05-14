import time

from base_scenario import BaseScenario
from utils.route_utils import build_route_between
from utils.transform_utils import (
    interpolate_on_polyline,
    get_polyline_end_point,
    dist_xy,
)


class UrbanBasicDriveScenario(BaseScenario):
    zone_name = "urban"
    scenario_name = "basic_drive"

    def setup(self):
        self.start_link = self.cfg["start_link"]
        self.end_link = self.cfg["end_link"]

        self.ego_spawn_offset_m = self.cfg.get("ego_spawn_offset_m", 5.0)
        self.decision_range = self.cfg.get("decision_range_m", 100.0)

        # 시작 위치 계산
        start_points = self.map_loader.get_link_points(self.start_link)
        x, y, z, yaw = interpolate_on_polyline(start_points, self.ego_spawn_offset_m)
        self.start_tf = self.grpc.make_transform(x, y, z, yaw)

        # 목표 위치 계산
        end_points = self.map_loader.get_link_points(self.end_link)
        self.goal_x, self.goal_y, self.goal_z = get_polyline_end_point(end_points)

        print(f"[UrbanBasicDrive] start_link={self.start_link}")
        print(f"[UrbanBasicDrive] end_link={self.end_link}")
        print(f"[UrbanBasicDrive] start=({x:.3f}, {y:.3f}, {z:.3f}, yaw={yaw:.3f})")
        print(f"[UrbanBasicDrive] goal=({self.goal_x:.3f}, {self.goal_y:.3f}, {self.goal_z:.3f})")

        # route 생성
        self.route_links, self.route_length_m = build_route_between(
            self.map_loader,
            self.start_link,
            self.end_link,
        )

        print(f"[UrbanBasicDrive] route_links={len(self.route_links)}, length={self.route_length_m:.1f}m")
        print("[UrbanBasicDrive] route:")
        for i, link_id in enumerate(self.route_links):
            print(f"  {i:02d}: {link_id}")

        # 첫 world 시작
        self.grpc.start_world(self.start_tf)
        self.configure_drive_from_start()

    def configure_drive_from_start(self):
        """
        현재 MORAI world에서 Ego를 시작점에 두고 route/cruise 설정.
        """
        print("[UrbanBasicDrive] configure drive from start")

        self.grpc.set_ego_transform(self.start_tf)
        time.sleep(0.2)

        self.grpc.set_ego_route(self.route_links, decision_range=self.decision_range)

        if hasattr(self.grpc, "set_ego_destination"):
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
        )

    def restart_to_start_and_drive(self):
        """
        목표 도착 후 MORAI world 자체를 재시작하고 다시 주행 시작.
        teleport만 하면 built-in cruise 내부 route 상태가 남아서 경로가 꼬일 수 있음.
        """
        print("[UrbanBasicDrive] restart to start")

        self.grpc.restart_world(self.start_tf)
        time.sleep(0.5)

        self.configure_drive_from_start()

    def run_timeline(self):
        timeout_sec = self.cfg.get("timeout_sec", 0.0)
        use_timeout = timeout_sec is not None and float(timeout_sec) > 0.0

        goal_tolerance_m = self.cfg.get("goal_tolerance_m", 10.0)
        check_period_sec = self.cfg.get("check_period_sec", 0.2)

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

        while True:
            elapsed = time.time() - lap_start_time

            ego_x, ego_y = self.grpc.get_ego_xy()
            dist_to_goal = dist_xy(ego_x, ego_y, self.goal_x, self.goal_y)

            if elapsed - last_print_time >= 1.0:
                if hasattr(self.grpc, "get_ego_state_debug"):
                    debug_state = self.grpc.get_ego_state_debug()
                    if debug_state is not None:
                        print(
                            f"[UrbanBasicDrive] lap={lap + 1} "
                            f"t={elapsed:.1f}s "
                            f"ego=({ego_x:.2f}, {ego_y:.2f}) "
                            f"link={debug_state['current_link']} "
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
                else:
                    print(
                        f"[UrbanBasicDrive] lap={lap + 1} "
                        f"t={elapsed:.1f}s "
                        f"ego=({ego_x:.2f}, {ego_y:.2f}) "
                        f"dist_to_goal={dist_to_goal:.2f}m"
                    )

                last_print_time = elapsed

            # 목표점 도착
            if dist_to_goal <= goal_tolerance_m:
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
                continue

            # timeout
            if use_timeout and elapsed >= float(timeout_sec):
                print(
                    f"[UrbanBasicDrive] TIMEOUT "
                    f"lap={lap + 1}, dist={dist_to_goal:.2f}m, elapsed={elapsed:.1f}s"
                )
                break

            time.sleep(check_period_sec)
