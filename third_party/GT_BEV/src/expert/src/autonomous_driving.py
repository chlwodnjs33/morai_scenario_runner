#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .obstacle.forward_object_detector import ForwardObjectDetector
from .localization.path_manager import PathManager
from .planning.adaptive_cruise_control import AdaptiveCruiseControl
from .control.pure_pursuit import PurePursuit
from .control.pid import Pid
from .control.control_input import ControlInput
from .config.config import Config
from .path.calc_path import mgeo_dijkstra_path

import numpy as np

class AutonomousDriving:
    def __init__(self):
        config = Config()

        if config["map"]["use_mgeo_path"]:
            mgeo_path = mgeo_dijkstra_path(config["map"]["name"])
            self.path = mgeo_path.calc_dijkstra_path(config["map"]["mgeo"]["start_node"], config["map"]["mgeo"]["end_node"])
            self.path_manager = PathManager(
                self.path, config["map"]["is_closed_path"], config["map"]["local_path_size"]
            )
        else:
            self.path = config["map"]["path"]
            self.path_manager = PathManager(
                self.path, config["map"]["is_closed_path"], config["map"]["local_path_size"]
            )
        self.path_manager.set_velocity_profile(**config['planning']['velocity_profile'])

        self.forward_object_detector = ForwardObjectDetector(config["map"]["traffic_light_list"])

        self.adaptive_cruise_control = AdaptiveCruiseControl(
            vehicle_length=config['common']['vehicle_length'], **config['planning']['adaptive_cruise_control']
        )
        self.pid = Pid(sampling_time=1/float(config['common']['sampling_rate']), **config['control']['pid'])
        self.pure_pursuit = PurePursuit(
            wheelbase=config['common']['wheelbase'], **config['control']['pure_pursuit']
        )
        self.max_steering = config['common']['max_steering'] / 180.0 * np.pi

    def execute(self, vehicle_state, dynamic_object_list, current_traffic_light):
        # 현재 위치 기반으로 local path과 planned velocity 추출
        local_path, planned_velocity = self.path_manager.get_local_path(vehicle_state)

        # 전방 장애물 인지
        self.forward_object_detector._dynamic_object_list = dynamic_object_list
        object_info_dic_list = self.forward_object_detector.detect_object(vehicle_state)

        # adaptive cruise control를 활용한 속도 계획
        self.adaptive_cruise_control.check_object(local_path, object_info_dic_list, current_traffic_light)
        target_velocity = self.adaptive_cruise_control.get_target_velocity(vehicle_state.velocity, planned_velocity)
        # 속도 제어를 위한 PID control
        acc_cmd = self.pid.get_output(target_velocity, vehicle_state.velocity)
        # target velocity가 0이고, 일정 속도 이하일 경우 full brake를 하여 차량을 멈추도록 함.
        if round(target_velocity) == 0 and vehicle_state.velocity < 2:
            acc_cmd = -1.
        # 경로 추종을 위한 pure pursuit control
        self.pure_pursuit.path = local_path
        self.pure_pursuit.vehicle_state = vehicle_state
        # CtrlCmd.front_steer in MORAI 26.R1 is a front-wheel angle in radians,
        # not a normalized [-1, 1] command. GT_BEV's pure-pursuit lateral sign
        # is opposite to the sign observed at the simulator interface, so flip
        # it exactly once here and clamp it to the physical steering limit.
        raw_steering_rad = self.pure_pursuit.calculate_steering_angle()
        front_steer_rad = np.clip(
            -raw_steering_rad,
            -self.max_steering,
            self.max_steering,
        )

        self._debug_counter = getattr(self, "_debug_counter", 0) + 1
        if self._debug_counter >= 30:
            self._debug_counter = 0
            print(
                f"[CTRL] raw_steering={np.degrees(raw_steering_rad):+.2f}deg "
                f"front_steer={np.degrees(front_steer_rad):+.2f}deg "
                f"({front_steer_rad:+.3f}rad) acc_cmd={acc_cmd:+.3f} "
                f"target_v={target_velocity:.2f} planned_v={planned_velocity:.2f} "
                f"cur_v={vehicle_state.velocity:.2f}"
            )

        return ControlInput(acc_cmd, front_steer_rad), local_path