#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ego 제어 모드 진단/전환 도구 (읽기 + 모드 설정 + 앞바퀴각 모니터).

사용법:
    python3 check_ctrl_mode.py            # 모드를 AUTO_MODE(3)로 설정하고 앞바퀴각 모니터
    python3 check_ctrl_mode.py 4          # AUTO_MODE_LATERAL(4)로 설정 (확정 실험용)
    python3 check_ctrl_mode.py none       # 모드 변경 없이 앞바퀴각만 모니터
"""
import os
import sys
import time
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'scenario_runner'))

from utils.grpc_client import MoraiGrpcClient  # noqa: E402

MODE_NAMES = {
    0: 'UNSPECIFIED', 1: 'KEYBOARD', 2: 'GAME_WHEEL', 3: 'AUTO_MODE',
    4: 'AUTO_MODE_LATERAL', 5: 'AUTO_MODE_LONGITUDINAL',
    6: 'CRUISE_MODE', 7: 'SYNCHRONOUS_MODE',
}


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else '3'

    with open(os.path.join(ROOT, 'scenario_runner/config/global.yaml')) as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg.get('paths', {}).items():
        if isinstance(v, str) and not os.path.isabs(v):
            cfg['paths'][k] = os.path.join(ROOT, v)

    client = MoraiGrpcClient(cfg)
    client.connect()

    from api.simulation_world import SimulationWorld
    from proto.morai.actor.actor_set_pb2 import VehicleControlModeParam
    from proto.morai.common.enum_pb2 import STATUS_CODE_SUCCESS

    world = SimulationWorld(client.client._sim_adapter, cfg['grpc']['client_key'])
    ego = world.get_ego()

    if arg != 'none':
        mode = int(arg)
        param = VehicleControlModeParam()
        param.actor_info.CopyFrom(ego.get_object_info())
        param.mode = mode
        result = ego._sim_adapter.set_vehicle_control_mode(param)
        ok = result is not None and result.status == STATUS_CODE_SUCCESS
        print(f'[mode] set {mode} ({MODE_NAMES.get(mode)}): {ok}')

    print('[monitor] 앞바퀴각/속도 관측 (Ctrl+C 종료)')
    print('          이 상태에서 다른 터미널로 rostopic pub 조향 명령을 쏘세요.')
    while True:
        state = ego.get_actor_state()
        if state is None:
            print('  state=None')
        else:
            vs = state.vehicle_state
            v = state.velocity
            speed = (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5
            print(f'  front_wheel_angle={vs.front_wheel_angle:+8.4f}  '
                  f'speed={speed:6.2f}  yaw={state.transform.rotation.z:+8.3f}')
        time.sleep(0.3)


if __name__ == '__main__':
    main()
