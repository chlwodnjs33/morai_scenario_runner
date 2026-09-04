"""
full_loop 신호등 교차로(체크포인트1)를 완전히 건너뛰고, 그 다음 구간부터
바로 출발시켜서 조향이 먹는지 확인하는 진단용 스크립트.

체크포인트1(원본 경로 파일 기준 idx=146)의 좌회전이 이미 끝났다고 보고,
그 이후 지점(idx=START_INDEX)에 ego를 바로 스폰해서 GT_BEV로 나머지
구간을 달리게 한다. NPC/장애물/날씨/closed-loop 검증 등은 전부 생략하고
순수하게 "조향이 실제로 걸리는지"만 본다.

실행: python3 scenario_runner/tools/test_steering_probe.py
"""

import math
import os
import sys
import time

import yaml

SCENARIO_RUNNER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCENARIO_RUNNER_DIR)
sys.path.insert(0, SCENARIO_RUNNER_DIR)

from utils.grpc_client import MoraiGrpcClient  # noqa: E402
from utils.gt_bev_expert import GTBEVExpertController  # noqa: E402

# 체크포인트1(idx=146) 좌회전이 다 끝나고 헤딩이 안정된 지점부터 시작.
# 그 지점부터는 다음 신호등 교차로(체크포인트1)를 다시 지나지 않는다.
START_INDEX = 260
RUN_SECONDS = 90.0


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(cfg):
    for key, val in cfg.get("paths", {}).items():
        if isinstance(val, str) and not os.path.isabs(val):
            cfg["paths"][key] = os.path.join(PROJECT_ROOT, val)


def load_route_points(path, start_index):
    points = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            fields = text.split()
            z = float(fields[2]) if len(fields) >= 3 else 0.0
            points.append((float(fields[0]), float(fields[1]), z))
    if start_index >= len(points) - 1:
        raise RuntimeError(f"START_INDEX({start_index}) too large for path of {len(points)} points")
    return points[start_index:]


def main():
    global_cfg = load_yaml(os.path.join(SCENARIO_RUNNER_DIR, "config", "global.yaml"))
    resolve_paths(global_cfg)

    global_path = os.path.join(
        PROJECT_ROOT,
        "third_party",
        "GT_BEV",
        "R_KR_PR_K-city_2025",
        "2026_molit_comp_global_path.txt",
    )
    route_points = load_route_points(global_path, START_INDEX)
    print(f"[Probe] route_points={len(route_points)}, starting at raw path index {START_INDEX}")

    first = route_points[0]
    second = route_points[1]
    yaw = math.degrees(math.atan2(second[1] - first[1], second[0] - first[0]))
    print(f"[Probe] start=({first[0]:.2f},{first[1]:.2f}) yaw={yaw:.2f}deg")

    grpc = MoraiGrpcClient(global_cfg)
    grpc.connect()

    try:
        start_tf = grpc.make_transform(first[0], first[1], first[2], yaw)
        grpc.start_world(start_tf)

        if hasattr(grpc, "stop_ego_cruise"):
            grpc.stop_ego_cruise()
        grpc.set_ego_control_mode_auto()
        if hasattr(grpc, "set_ego_gear_drive"):
            grpc.set_ego_gear_drive()

        gt_bev = GTBEVExpertController(
            repo_path=os.path.join(PROJECT_ROOT, "third_party", "GT_BEV"),
            map_name="R_KR_PR_K-city_2025",
            mgeo_root=global_cfg["paths"]["mgeo_root"],
            route_points=route_points,
            max_velocity_kmh=30.0,
            traffic_light_control=False,
            ros_remaps=["/ctrl_cmd:=/ctrl_cmd"],
            is_closed_path=False,
            velocity_profile_window_size=15,
        )
        gt_bev.start_process()

        print(
            f"[Probe] driving for {RUN_SECONDS:.0f}s from a point AFTER checkpoint1's "
            "intersection/traffic light. MORAI HUD의 Steer Angle / Steering Wheel Angle을 "
            "지켜봐 주세요 -- 여기서도 0.0 고정이면 신호등 문제가 아니라 더 근본적인 원인이고, "
            "여기서는 움직이면 체크포인트1 교차로(IntTL4)가 범인으로 확정됩니다."
        )

        t0 = time.time()
        max_abs_wheel_angle = 0.0
        while time.time() - t0 < RUN_SECONDS:
            if not gt_bev.is_running():
                print("[Probe] GT_BEV process stopped unexpectedly")
                break
            state = grpc.get_ego_motion_state()
            max_abs_wheel_angle = max(max_abs_wheel_angle, abs(state["front_wheel_angle"]))
            print(
                f"[Probe] t={time.time() - t0:5.1f}s "
                f"pos=({state['x']:.2f},{state['y']:.2f}) "
                f"yaw={state['yaw_deg']:.2f} speed={state['speed']:.2f} "
                f"link={state['current_link']} "
                f"front_wheel_angle={state['front_wheel_angle']:.3f} "
                f"(max so far={max_abs_wheel_angle:.3f})"
            )
            time.sleep(1.0)

        print(f"[Probe] done. max |front_wheel_angle| observed = {max_abs_wheel_angle:.3f}")
    except KeyboardInterrupt:
        print("[Probe] interrupted")
    finally:
        try:
            gt_bev.stop_process()
        except Exception as e:
            print(f"[Probe] gt_bev stop_process failed: {e}")
        grpc.stop()


if __name__ == "__main__":
    main()
