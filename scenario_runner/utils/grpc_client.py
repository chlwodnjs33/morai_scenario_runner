import os
import sys


class MoraiGrpcClient:
    def __init__(self, global_cfg: dict):
        self.global_cfg = global_cfg

        grpc_cfg = global_cfg["grpc"]
        path_cfg = global_cfg["paths"]

        self.host = grpc_cfg.get("host", "127.0.0.1")
        self.port = int(grpc_cfg.get("port", 7789))
        self.client_key = grpc_cfg.get("client_key", "scenario_runner")
        self.grpc_src = path_cfg["grpc_src"]

        self._add_grpc_paths()

        from api.morai_sim_client import MoraiSimClient

        self.client = MoraiSimClient(self.client_key)
        self.world = None

    def _add_grpc_paths(self):
        api_path = os.path.join(self.grpc_src, "api")
        proto_path = os.path.join(self.grpc_src, "proto")

        for p in [self.grpc_src, api_path, proto_path]:
            if p not in sys.path:
                sys.path.append(p)

    def connect(self):
        self.client.connect(self.host, self.port)

        if not self.client.is_connected():
            raise RuntimeError("Failed to connect MORAI gRPC server")

        print(f"[gRPC] connected to {self.host}:{self.port}")

    def make_transform(self, x, y, z, yaw_deg):
        from proto.morai.common.type_pb2 import Transform

        tf = Transform()
        tf.location.x = float(x)
        tf.location.y = float(y)
        tf.location.z = float(z)

        tf.rotation.x = 0.0
        tf.rotation.y = 0.0
        tf.rotation.z = float(yaw_deg)
        return tf

    def start_world(self, ego_transform):
        from proto.morai.simulation.start_param_pb2 import MapAndVehicle
        from proto.morai.actor.actor_set_pb2 import EgoCruiseControl
        from proto.morai.actor.actor_enum_pb2 import EGO_CRUISE_TYPE_LINK
        from proto.morai.simulation.sync_mode_pb2 import SyncMode
        from proto.morai.simulation.simulation_enum_pb2 import SYNC_MODE_TYPE_UNSPECIFIED

        morai_cfg = self.global_cfg["morai"]

        map_and_vehicle = MapAndVehicle()
        map_and_vehicle.map_name = morai_cfg["map_name"]
        map_and_vehicle.ego_vehicle_model = morai_cfg["ego_vehicle_model"]

        cruise = EgoCruiseControl()
        cruise.cruise_on = False
        cruise.cruise_type = EGO_CRUISE_TYPE_LINK
        cruise.link_speed_ratio = 0
        cruise.constant_velocity = 0

        sync_mode = SyncMode()
        sync_mode.type = SYNC_MODE_TYPE_UNSPECIFIED

        self.client.start_simulation(
            map_and_vehicle=map_and_vehicle,
            ego_transform=ego_transform,
            ego_cruise_setting=cruise,
            sync_mode=sync_mode,
        )

        self.world = self.client.get_simulation_world()
        if self.world is None:
            raise RuntimeError("Failed to start MORAI simulation world")

        print("[MORAI] world started")

    def reset_actors(self):
        if self.world is not None:
            self.world.destroy_all_actors()
            print("[MORAI] actors destroyed")

    def get_ego(self):
        if self.world is None:
            raise RuntimeError("world is None. Call start_world() first.")
        return self.world.get_ego()

    def set_ego_transform(self, transform):
        ego = self.get_ego()
        ok = ego.set_transform(transform)
        print(f"[Ego] set_transform: {ok}")
        return ok

    def set_ego_route(self, route_links, decision_range=30.0):
        ego = self.get_ego()
        ok = ego.set_vehicle_route(decision_range, route_links)
        print(f"[Ego] set_vehicle_route: {ok}, links={route_links}")
        return ok

    def set_ego_cruise(self, enable=True, link_speed_ratio=40, constant_velocity=20):
        ego = self.get_ego()
        ok = ego.set_cruise_mode(
            enable=enable,
            link_speed_ratio=link_speed_ratio,
            constant_velocity=constant_velocity,
        )
        print(f"[Ego] set_cruise_mode: {ok}")
        return ok

    def set_ego_control_mode_cruise(self):
        from proto.morai.actor.actor_enum_pb2 import VehicleControlMode

        ego = self.get_ego()
        ok = ego.set_control_mode(VehicleControlMode.VEHICLE_CONTROL_CRUISE_MODE)
        print(f"[Ego] set_control_mode CRUISE: {ok}")
        return ok

    def spawn_vehicle(self, transform, model_name, label, velocity=0.0, multi_ego=True):
        vehicle = self.world.spawn_vehicle(
            transform=transform,
            model_name=model_name,
            label=label,
            velocity=velocity,
            multi_ego=multi_ego,
        )
        print(f"[Spawn] vehicle {label}: {vehicle is not None}")
        return vehicle

    def stop(self):
        if self.client is not None:
            self.client.finalize()
            print("[MORAI] finalized")


def _extract_xy_from_actor_state(state):
    """
    MORAI ActorState에서 x, y 위치 추출.
    """
    return (
        float(state.transform.location.x),
        float(state.transform.location.y),
    )


# class에 나중에 붙이기 위한 monkey patch 방식
def _morai_get_ego_xy(self):
    ego = self.get_ego()
    state = ego.get_actor_state()
    if state is None:
        raise RuntimeError("Failed to get ego actor state")
    return _extract_xy_from_actor_state(state)


MoraiGrpcClient.get_ego_xy = _morai_get_ego_xy


def _morai_set_ego_destination(self, x, y, z, decision_range=50.0):
    from proto.morai.common.type_pb2 import Vector3

    ego = self.get_ego()

    pos = Vector3()
    pos.x = float(x)
    pos.y = float(y)
    pos.z = float(z)

    ok = ego.set_vehicle_destination(decision_range, pos)
    print(f"[Ego] set_vehicle_destination: {ok}, goal=({x:.2f}, {y:.2f}, {z:.2f})")
    return ok


def _morai_get_ego_state_debug(self):
    ego = self.get_ego()
    state = ego.get_actor_state()
    if state is None:
        return None

    x = state.transform.location.x
    y = state.transform.location.y

    vehicle_state = state.vehicle_state
    cur_link = vehicle_state.current_link_info.id.value
    remain_dist = vehicle_state.remaining_distance
    remain_link_count = vehicle_state.remaining_link_count
    is_pass_des_pos = vehicle_state.is_pass_des_pos

    return {
        "x": x,
        "y": y,
        "current_link": cur_link,
        "remaining_distance": remain_dist,
        "remaining_link_count": remain_link_count,
        "is_pass_des_pos": is_pass_des_pos,
    }


MoraiGrpcClient.set_ego_destination = _morai_set_ego_destination
MoraiGrpcClient.get_ego_state_debug = _morai_get_ego_state_debug


def _morai_restart_world(self, ego_transform):
    """
    MORAI world를 완전히 재시작해서 built-in cruise 내부 route 상태까지 초기화.
    """
    print("[MORAI] restart world")

    try:
        if self.world is not None:
            self.client.finalize()
    except Exception as e:
        print(f"[MORAI] finalize warning during restart: {e}")

    # client 객체를 새로 생성해서 gRPC 연결부터 다시 시작
    from api.morai_sim_client import MoraiSimClient

    self.client = MoraiSimClient(self.client_key)
    self.world = None

    self.connect()
    self.start_world(ego_transform)


MoraiGrpcClient.restart_world = _morai_restart_world
