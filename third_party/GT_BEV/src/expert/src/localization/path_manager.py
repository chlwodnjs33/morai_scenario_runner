#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .point import Point
import numpy as np
from math import sqrt,pi 


class PathManager:
    def __init__(self, path, is_closed_path, local_path_size):
        self.path = path
        self.is_closed_path = is_closed_path
        self.local_path_size = local_path_size
        self.velocity_profile = []

    def set_velocity_profile(self, max_velocity, road_friction, window_size):
        max_velocity = max_velocity / 3.6
        path_size = len(self.path)
        if path_size == 0:
            self.velocity_profile = []
            return

        window_size = max(1, min(int(window_size), (path_size - 1) // 2))
        x = np.fromiter((point.x for point in self.path), dtype=float, count=path_size)
        y = np.fromiter((point.y for point in self.path), dtype=float, count=path_size)
        center_idx = np.arange(path_size)

        if self.is_closed_path:
            start_idx = (center_idx - window_size) % path_size
            end_idx = (center_idx + window_size) % path_size
        else:
            radius = np.minimum.reduce((np.full(path_size, window_size), center_idx, path_size - 1 - center_idx))
            start_idx = center_idx - radius
            end_idx = center_idx + radius

        d_st_x = x[start_idx] - x
        d_st_y = y[start_idx] - y
        d_ed_x = x[end_idx] - x
        d_ed_y = y[end_idx] - y

        dcom = 2.0 * (d_st_x * d_ed_y - d_st_y * d_ed_x)
        d_st2 = d_st_x * d_st_x + d_st_y * d_st_y
        d_ed2 = d_ed_x * d_ed_x + d_ed_y * d_ed_y

        with np.errstate(divide='ignore', invalid='ignore'):
            u1 = (d_ed_y * d_st2 - d_st_y * d_ed2) / dcom
            u2 = (d_st_x * d_ed2 - d_ed_x * d_st2) / dcom
            radius = np.sqrt(u1 * u1 + u2 * u2)
            velocity_profile = np.sqrt(radius * 9.8 * road_friction)

        velocity_profile = np.where(np.isfinite(velocity_profile), velocity_profile, max_velocity)
        velocity_profile = np.minimum(velocity_profile, max_velocity)

        if not self.is_closed_path:
            velocity_profile[-min(10, path_size):] = 0.0

        self.velocity_profile = velocity_profile.tolist()

    def get_local_path(self, vehicle_state):
        # TODO: 최소값 구하는 로직 개선 필요.
        min_distance=float('inf')
        current_waypoint=0
        for i, point in enumerate(self.path):
            dx = point.x - vehicle_state.position.x
            dy = point.y - vehicle_state.position.y
            distance = dx*dx + dy*dy
            if distance < min_distance:
                min_distance = distance
                current_waypoint = i

        if current_waypoint + self.local_path_size < len(self.path):
            local_path = self.path[current_waypoint:current_waypoint + self.local_path_size]
        else:
            local_path = self.path[current_waypoint:]
            # 연결된 경로 (closed path) 일 경우, 경로 끝과 처음을 이어준다.
            if self.is_closed_path:
                local_path += self.path[:self.local_path_size + len(self.path) - current_waypoint]

        return local_path, self.velocity_profile[current_waypoint]
