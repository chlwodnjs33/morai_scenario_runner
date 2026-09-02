#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys


if __package__:
    from .network.ros_manager import RosManager
    from .autonomous_driving import AutonomousDriving
else:
    # `python3 main.py`로 직접 실행해도 expert 패키지를 찾을 수 있게 한다.
    # Ignore a stale Scenario Runner config when this file is executed
    # directly. Set GT_BEV_DIRECT_USE_ENV_CONFIG=1 to opt back into it.
    if os.environ.get("GT_BEV_DIRECT_USE_ENV_CONFIG", "0") != "1":
        os.environ.pop("GT_BEV_CONFIG_PATH", None)
        os.environ.pop("GT_BEV_MAP_DIR", None)
    gt_bev_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if gt_bev_src not in sys.path:
        sys.path.insert(0, gt_bev_src)
    from expert.src.network.ros_manager import RosManager
    from expert.src.autonomous_driving import AutonomousDriving


def main():
    autonomous_driving = AutonomousDriving()
    ros_manager = RosManager(autonomous_driving)
    ros_manager.execute()


if __name__ == '__main__':
    main()
