import argparse
import yaml

from utils.grpc_client import MoraiGrpcClient
from utils.map_loader import MGeoMapLoader
from zones.urban_scenarios import UrbanBasicDriveScenario, UrbanSuddenBrakeScenario


def load_yaml(path):
    print(f"[DEBUG] loading yaml: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise RuntimeError(f"YAML is empty: {path}")
    return data


def main():
    print("[DEBUG] main() start")

    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", default="urban")
    parser.add_argument("--scenario", default="basic_drive")
    args = parser.parse_args()

    print(f"[DEBUG] args: zone={args.zone}, scenario={args.scenario}")

    global_cfg = load_yaml("scenario_runner/config/global.yaml")
    zone_cfg = load_yaml(f"scenario_runner/config/{args.zone}.yaml")

    scenario_cfg = zone_cfg["scenarios"][args.scenario]

    print("[DEBUG] creating MoraiGrpcClient")
    grpc_client = MoraiGrpcClient(global_cfg)

    print("[DEBUG] connecting gRPC")
    grpc_client.connect()

    print("[DEBUG] loading MGeo")
    map_loader = MGeoMapLoader(global_cfg["paths"]["mgeo_root"])

    if args.zone == "urban" and args.scenario == "basic_drive":
        scenario = UrbanBasicDriveScenario(
            grpc_client=grpc_client,
            map_loader=map_loader,
            global_cfg=global_cfg,
            scenario_cfg=scenario_cfg,
        )
    elif args.zone == "urban" and args.scenario == "sudden_brake":
        scenario = UrbanSuddenBrakeScenario(
            grpc_client=grpc_client,
            map_loader=map_loader,
            global_cfg=global_cfg,
            scenario_cfg=scenario_cfg,
        )
    else:
        raise ValueError(f"Unknown scenario: {args.zone}/{args.scenario}")

    try:
        print("[DEBUG] running scenario")
        scenario.run()
    finally:
        print("[DEBUG] stopping gRPC client")
        grpc_client.stop()


if __name__ == "__main__":
    main()
