#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import io
import pandas as pd
from ..localization.point import Point
from ..obstacle.object_info import ObjectInfo
from ..path.mapping import TrafficLightStoplineMapper


class Config(object):
    _instance = None

    def __new__(cls):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls)

            config_path = os.environ.get(
                'GT_BEV_CONFIG_PATH',
                os.path.join(os.path.dirname(__file__), 'config.json'),
            )
            with io.open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                cls._instance.__dict__ = config

            cls._instance._set_map_data()
        return cls._instance

    def __getitem__(self, key):
        return getattr(self, key)

    def _set_map_data(self):
        path_csv = os.environ.get(
            'GT_BEV_PATH_CSV',
            os.path.join(os.path.dirname(__file__), 'map', self["map"]["name"], 'path.csv')
        )
        path = pd.read_csv(path_csv)
        self["map"]["path"] = path.apply(
            lambda point: Point(point["x"], point["y"]), axis=1
        ).tolist()

        stopline_mapper = TrafficLightStoplineMapper.for_map_name(self["map"]["name"])
        self["map"]["stopline_mapper"] = stopline_mapper

        mgeo_signal_path = os.path.normpath(os.path.join(
            os.path.dirname(__file__), '../../../../', self["map"]["name"], 'traffic_light_set.json'
        ))
        with io.open(mgeo_signal_path, 'r', encoding='utf-8') as f:
            mgeo_signals = json.load(f)

        traffic_light_rows = []
        for s in mgeo_signals:
            if s.get("type") != "car":
                continue
            idx = s["idx"]
            stopline_center = stopline_mapper.get_stopline_center(idx)
            if stopline_center is None:
                continue
            traffic_light_rows.append({
                "name": idx,
                "x": float(stopline_center.x),
                "y": float(stopline_center.y),
                "velocity": 0,
                "object_type": 3,
            })

        if traffic_light_rows:
            traffic_light_list = pd.DataFrame(traffic_light_rows)
            self["map"]["traffic_light_list"] = traffic_light_list.apply(
                lambda traffic_light: ObjectInfo(**traffic_light.to_dict()), axis=1
            ).tolist()
        else:
            self["map"]["traffic_light_list"] = []

    def update_config(self, file_name):
        with io.open(file_name, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.__dict__.update(config)
