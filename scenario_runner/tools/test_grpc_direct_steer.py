"""
GT_BEV/ROS를 완전히 배제하고, 순수하게 gRPC control_ego()로만 조향을 넣어서
MORAI가 gRPC 채널의 조향은 반영하는지 확인하는 최소 테스트.

실행: python3 scenario_runner/tools/test_grpc_direct_steer.py
"""

import os
import sys
import time

import yaml

SCENARIO_RUNNER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCENARIO_RUNNER_DIR)
sys.path.insert(0, SCENARIO_RUNNER_DIR)

from utils.grpc_client import MoraiGrpcClient  # noqa: E402


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(cfg):
    for key, val in cfg.get("paths", {}).items():
        if isinstance(val, str) and not os.path.isabs(val):
            cfg["paths"][key] = os.path.join(PROJECT_ROOT, val)


def main():
    global_cfg = load_yaml(os.path.join(SCENARIO_RUNNER_DIR, "config", "global.yaml"))
    resolve_paths(global_cfg)

    grpc = MoraiGrpcClient(global_cfg)
    grpc.connect()

    try:
        start_tf = grpc.make_transform(-128.70, -326.72, 28.6, 106.49)
        grpc.start_world(start_tf)

        if hasattr(grpc, "stop_ego_cruise"):
            grpc.stop_ego_cruise()
        grpc.set_ego_control_mode_auto()
        if hasattr(grpc, "set_ego_gear_drive"):
            grpc.set_ego_gear_drive()

        print("[GrpcSteerTest] sending steer=+0.6, throttle=0.3 via gRPC control_ego() for 8s")
        t0 = time.time()
        while time.time() - t0 < 8.0:
            ok = grpc.control_ego(steer=0.6, target_speed=15.0, throttle=0.3)
            state = grpc.get_ego_motion_state()
            print(
                f"[GrpcSteerTest] t={time.time() - t0:4.1f}s control_ok={ok} "
                f"speed={state['speed']:.2f} yaw={state['yaw_deg']:.2f} "
                f"front_wheel_angle={state['front_wheel_angle']:.3f}"
            )
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("[GrpcSteerTest] interrupted")
    finally:
        grpc.stop()


if __name__ == "__main__":
    main()
