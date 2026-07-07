import argparse
import os
import yaml

from utils.grpc_client import MoraiGrpcClient
from utils.map_loader import MGeoMapLoader
from zones.urban_scenarios import (
    UrbanBasicDriveScenario,
    UrbanSuddenBrakeExpertScenario,
    UrbanTrafficJamScenario,
    UrbanPedestrianYieldScenario,
    HighwayMergeJudgementScenario,
    RoundaboutMergeScenario,
    RoundaboutYieldToInsideVehicleScenario,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_yaml(path):
    print(f"[DEBUG] loading yaml: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise RuntimeError(f"YAML is empty: {path}")
    return data


def resolve_paths(cfg):
    paths = cfg.get("paths", {})
    for key, val in paths.items():
        if isinstance(val, str) and not os.path.isabs(val):
            paths[key] = os.path.join(PROJECT_ROOT, val)


def main():
    print("[DEBUG] main() start")

    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", default="urban")
    parser.add_argument("--scenario", default="basic_drive")
    args = parser.parse_args()

    print(f"[DEBUG] args: zone={args.zone}, scenario={args.scenario}")

    global_cfg = load_yaml("scenario_runner/config/global.yaml")

    local_cfg_path = "scenario_runner/config/local.yaml"
    if os.path.exists(local_cfg_path):
        local_cfg = load_yaml(local_cfg_path)
        for section, values in local_cfg.items():
            if isinstance(values, dict) and isinstance(global_cfg.get(section), dict):
                global_cfg[section].update(values)
            else:
                global_cfg[section] = values

    resolve_paths(global_cfg)

    zone_cfg = load_yaml(f"scenario_runner/config/{args.zone}.yaml")

    scenario_cfg = zone_cfg["scenarios"][args.scenario]

    print("[DEBUG] creating MoraiGrpcClient")
    grpc_client = MoraiGrpcClient(global_cfg)

    print("[DEBUG] connecting gRPC")
    grpc_client.connect()

    print("[DEBUG] loading MGeo")
    map_loader = MGeoMapLoader(global_cfg["paths"]["mgeo_root"])

    if args.scenario == "basic_drive":
        scenario = UrbanBasicDriveScenario(
            grpc_client=grpc_client,
            map_loader=map_loader,
            global_cfg=global_cfg,
            scenario_cfg=scenario_cfg,
        )
    elif args.scenario == "sudden_brake":
        scenario = UrbanSuddenBrakeExpertScenario(
            grpc_client=grpc_client,
            map_loader=map_loader,
            global_cfg=global_cfg,
            scenario_cfg=scenario_cfg,
        )
    elif args.scenario == "traffic_jam":
        scenario = UrbanTrafficJamScenario(
            grpc_client=grpc_client,
            map_loader=map_loader,
            global_cfg=global_cfg,
            scenario_cfg=scenario_cfg,
        )
    elif args.scenario == "pedestrian_yield":
        scenario = UrbanPedestrianYieldScenario(
            grpc_client=grpc_client,
            map_loader=map_loader,
            global_cfg=global_cfg,
            scenario_cfg=scenario_cfg,
        )
    elif args.scenario == "merge_judgement":
        scenario = HighwayMergeJudgementScenario(
            grpc_client=grpc_client,
            map_loader=map_loader,
            global_cfg=global_cfg,
            scenario_cfg=scenario_cfg,
        )
    elif args.scenario == "yield_to_inside_vehicle":
        scenario = RoundaboutYieldToInsideVehicleScenario(
            grpc_client=grpc_client,
            map_loader=map_loader,
            global_cfg=global_cfg,
            scenario_cfg=scenario_cfg,
        )
    elif args.scenario == "roundabout_merge":
        scenario = RoundaboutMergeScenario(
            grpc_client=grpc_client,
            map_loader=map_loader,
            global_cfg=global_cfg,
            scenario_cfg=scenario_cfg,
        )
    else:
        raise ValueError(f"Unknown scenario: {args.zone}/{args.scenario}")

    scenario.zone_name = zone_cfg.get("zone", args.zone)

    try:
        print("[DEBUG] running scenario")
        scenario.run()
    finally:
        print("[DEBUG] stopping gRPC client")
        grpc_client.stop()


if __name__ == "__main__":
    main()
