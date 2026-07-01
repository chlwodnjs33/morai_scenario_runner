#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BEV HD Map Generator

채널 구성:
  ch0 (/bev/drivable)   : 주행가능영역 (road_mesh_out_line.json)
  ch1 (/bev/lane)       : 차선 (lane_boundary_set.json)
  ch2 (/bev/vehicle)    : 차량 NPC (MORAI ObjectStatusList, type==1)
  ch3 (/bev/pedestrian) : 보행자 (MORAI ObjectStatusList, type==0)
  /bev/stopline_*       : 신호등 상태가 매핑된 정지선

시각화: /bev/viz (sensor_msgs/Image) + cv2.imshow
"""

import os
import json
import math
import threading
import numpy as np
import cv2
import rospy
import mapbox_earcut as earcut

from morai_msgs.msg import EgoVehicleStatus, GetTrafficLightStatus, ObjectStatusList
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ══════════════════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════════════════

_WS_ROOT  = os.path.expanduser("~/GT_BEV")
MAP_DIR   = os.path.join(_WS_ROOT, "R_KR_PG_KATRI")

SIZE      = 320
RES       = 0.2
FORWARD_M = SIZE * RES
LAT_HALF  = SIZE * RES / 2.0

PUBLISH_HZ     = 10.0
OCCUPIED_VALUE = 100

NPC_TYPE_PEDESTRIAN      = 0
NPC_TYPE_VEHICLE         = 1
NPC_REAR_TO_CENTER_SCALE = 1.0

TRAFFIC_LIGHT_TYPES = {"car", "bus"}
DIRECT_SIGNAL_STOPLINE_IDS = {
    "C119BS010063": ["B219BS010016"],
    "C119BS010064": ["B219BS010016"],
    "SSN000007": ["B219BS010016"],
}

TL_STATUS_RED    = {1}
TL_STATUS_YELLOW = {4, 5, 20, 36}
TL_STATUS_GREEN  = {16, 32, 48}
TL_STATUS_BLUE   = {33}

EGO_CUBE_SIZE_X    = 4.635
EGO_CUBE_SIZE_Y    = 1.890
EGO_CUBE_SIZE_Z    = 1.605
EGO_REAR_TO_CENTER = EGO_CUBE_SIZE_X / 2.0
EGO_MARKER_LIFE    = 0.2

WGS84_A  = 6378137.0
WGS84_F  = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

GLOBAL_INFO_JSON = os.path.join(MAP_DIR, "global_info.json")
FIXED_REF_LLA = (37.238838359501933, 126.772902206454901, 0.0)

# ══════════════════════════════════════════════════════════════════════════════
# 1. 공통 유틸
# ══════════════════════════════════════════════════════════════════════════════

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def lla_to_utm(lat_deg, lon_deg):
    zone     = int((lon_deg + 180.0) // 6.0) + 1
    lon0_deg = (zone - 1) * 6 - 180 + 3
    lat  = math.radians(lat_deg)
    lon  = math.radians(lon_deg)
    lon0 = math.radians(lon0_deg)
    ep2     = WGS84_E2 / (1.0 - WGS84_E2)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    tan_lat = math.tan(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    t = tan_lat * tan_lat
    c = ep2 * cos_lat * cos_lat
    a = cos_lat * (lon - lon0)
    m = WGS84_A * (
        (1.0 - WGS84_E2/4.0 - 3.0*WGS84_E2**2/64.0 - 5.0*WGS84_E2**3/256.0) * lat
        - (3.0*WGS84_E2/8.0 + 3.0*WGS84_E2**2/32.0 + 45.0*WGS84_E2**3/1024.0) * math.sin(2.0*lat)
        + (15.0*WGS84_E2**2/256.0 + 45.0*WGS84_E2**3/1024.0) * math.sin(4.0*lat)
        - (35.0*WGS84_E2**3/3072.0) * math.sin(6.0*lat)
    )
    k0       = 0.9996
    easting  = k0 * n * (
        a + (1.0 - t + c) * a**3 / 6.0
        + (5.0 - 18.0*t + t**2 + 72.0*c - 58.0*ep2) * a**5 / 120.0
    ) + 500000.0
    northing = k0 * (
        m + n * tan_lat * (
            a**2 / 2.0
            + (5.0 - t + 9.0*c + 4.0*c**2) * a**4 / 24.0
            + (61.0 - 58.0*t + t**2 + 600.0*c - 330.0*ep2) * a**6 / 720.0
        )
    )
    if lat_deg < 0.0:
        northing += 10000000.0
    return easting, northing, zone


def utm_grid_convergence_rad(lat_deg, lon_deg, zone):
    lon0_deg = (zone - 1) * 6 - 180 + 3
    lat  = math.radians(lat_deg)
    lon  = math.radians(lon_deg)
    lon0 = math.radians(lon0_deg)
    return math.atan(math.tan(lon - lon0) * math.sin(lat))


def compute_csv_offset_from_origin(ref_lla, global_info_path):
    with open(global_info_path, encoding="utf-8") as f:
        ginfo = json.load(f)
    if "local_origin_in_global" not in ginfo or len(ginfo["local_origin_in_global"]) < 2:
        raise ValueError(f"local_origin_in_global 누락: {global_info_path}")
    utm_e, utm_n, zone = lla_to_utm(ref_lla[0], ref_lla[1])
    conv_rad     = utm_grid_convergence_rad(ref_lla[0], ref_lla[1], zone)
    map_origin_e = float(ginfo["local_origin_in_global"][0])
    map_origin_n = float(ginfo["local_origin_in_global"][1])
    return utm_e - map_origin_e, utm_n - map_origin_n, conv_rad, ref_lla, zone


# ══════════════════════════════════════════════════════════════════════════════
# 2. 좌표 변환
# ══════════════════════════════════════════════════════════════════════════════

def map_to_local(pts, ego_x, ego_y, yaw):
    dx = pts[:, 0] - ego_x
    dy = pts[:, 1] - ego_y
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([c*dx + s*dy, -s*dx + c*dy], axis=1)


def local_to_pixel(local):
    gx = np.rint(local[:, 0] / RES).astype(np.int32)
    gy = np.rint((local[:, 1] + LAT_HALF) / RES).astype(np.int32)
    return np.stack([gx, gy], axis=1)


def in_crop(lx, ly):
    return 0.0 <= lx <= FORWARD_M and -LAT_HALF <= ly <= LAT_HALF


def pts_in_view(local):
    return np.any(
        (local[:, 0] >= 0) & (local[:, 0] <= FORWARD_M) &
        (np.abs(local[:, 1]) <= LAT_HALF)
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. 정적 맵 전처리
# ══════════════════════════════════════════════════════════════════════════════

def build_drivable_triangles(road_mesh_path):
    data = load_json(road_mesh_path)
    all_triangles = []
    for item in data:
        ext_pts   = item.get("points", [])
        interiors = item.get("interiors", [])
        if len(ext_pts) < 3:
            continue
        all_2d    = [[p[0], p[1]] for p in ext_pts]
        ring_ends = [len(ext_pts)]
        for interior in interiors:
            ipts = interior.get("points", [])
            if len(ipts) >= 3:
                all_2d.extend([[p[0], p[1]] for p in ipts])
                ring_ends.append(len(all_2d))
        verts = np.array(all_2d, dtype=np.float64)
        rings = np.array(ring_ends, dtype=np.uint32)
        try:
            tri_idx = earcut.triangulate_float64(verts, rings)
        except Exception as e:
            rospy.logwarn("[Drivable] earcut 실패: %s", e)
            continue
        if len(tri_idx) < 3:
            continue
        for i in range(0, len(tri_idx), 3):
            i0, i1, i2 = tri_idx[i], tri_idx[i+1], tri_idx[i+2]
            all_triangles.append(np.array([all_2d[i0], all_2d[i1], all_2d[i2]], dtype=np.float64))
    rospy.loginfo("[Drivable] 삼각형 %d개 생성 완료", len(all_triangles))
    return all_triangles


# ══════════════════════════════════════════════════════════════════════════════
# 4. 채널 렌더링
# ══════════════════════════════════════════════════════════════════════════════

def render_drivable(drivable_triangles, ego_x, ego_y, yaw):
    img = np.zeros((SIZE, SIZE), dtype=np.uint8)
    for tri in drivable_triangles:
        local = map_to_local(tri, ego_x, ego_y, yaw)
        if local[:, 0].max() < 0 or local[:, 0].min() > FORWARD_M:
            continue
        pix = local_to_pixel(local)
        cv2.fillPoly(img, [pix.reshape(-1, 1, 2)], 1)
    return (img > 0).astype(np.uint8)


def render_lanes(lane_set, ego_x, ego_y, yaw):
    imgs = {k: np.zeros((SIZE, SIZE), dtype=np.uint8)
            for k in ('yellow', 'white_solid', 'white_dashed', 'blue')}
    for item in lane_set:
        pts_raw = item.get("points", [])
        if len(pts_raw) < 2:
            continue
        pts   = np.array([[p[0], p[1]] for p in pts_raw], dtype=np.float64)
        local = map_to_local(pts, ego_x, ego_y, yaw)
        if not pts_in_view(local):
            continue
        pix        = local_to_pixel(local)
        lane_w     = float(item.get("lane_width", 0.15))
        thick      = max(1, int(round(max(0.12, lane_w * 0.8) / RES)))
        color_list = item.get("lane_color", ["white"])
        shape_list = item.get("lane_shape", ["Solid"])
        color = color_list[0] if isinstance(color_list, list) else color_list
        shape = shape_list[0] if isinstance(shape_list, list) else shape_list
        if color == "yellow":
            key = "yellow"
        elif color == "blue":
            key = "blue"
        else:
            key = "white_dashed" if "Broken" in shape else "white_solid"
        cv2.polylines(imgs[key], [pix.reshape(-1, 1, 2)], False, 1, thickness=thick)
    return {k: (v > 0).astype(np.uint8) for k, v in imgs.items()}


def _render_obj_list(objs, ego_x, ego_y, yaw):
    img   = np.zeros((SIZE, SIZE), dtype=np.uint8)
    c_ego = math.cos(yaw)
    s_ego = math.sin(yaw)
    for obj in objs:
        ox, oy = obj.position.x, obj.position.y
        dx, dy = ox - ego_x, oy - ego_y
        lx =  c_ego * dx + s_ego * dy
        ly = -s_ego * dx + c_ego * dy
        hl, hw   = obj.size.x / 2.0, obj.size.y / 2.0
        obj_yaw  = math.radians(obj.heading)
        rel_yaw  = obj_yaw - yaw
        obj_type = int(getattr(obj, "type", -1))
        rear_to_center = hl * NPC_REAR_TO_CENTER_SCALE if obj_type == NPC_TYPE_VEHICLE else 0.0
        cx = lx + rear_to_center * math.cos(rel_yaw)
        cy = ly + rear_to_center * math.sin(rel_yaw)
        if not in_crop(cx, cy):
            continue
        corners_l = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], dtype=np.float64)
        c_r, s_r  = math.cos(rel_yaw), math.sin(rel_yaw)
        rot       = np.array([[c_r, -s_r], [s_r, c_r]])
        corners   = (rot @ corners_l.T).T + np.array([cx, cy])
        pix       = local_to_pixel(corners)
        cv2.fillPoly(img, [pix.reshape(-1, 1, 2)], 1)
    return (img > 0).astype(np.uint8)


def render_vehicles(obj_msg, ego_x, ego_y, yaw):
    if obj_msg is None:
        return np.zeros((SIZE, SIZE), dtype=np.uint8)
    vehicles = [o for o in obj_msg.npc_list if int(getattr(o, "type", -1)) == NPC_TYPE_VEHICLE]
    return _render_obj_list(vehicles, ego_x, ego_y, yaw)


def render_pedestrians(obj_msg, ego_x, ego_y, yaw):
    if obj_msg is None:
        return np.zeros((SIZE, SIZE), dtype=np.uint8)
    pedestrians = list(obj_msg.pedestrian_list)
    pedestrians += [o for o in obj_msg.npc_list if int(getattr(o, "type", -1)) == NPC_TYPE_PEDESTRIAN]
    return _render_obj_list(pedestrians, ego_x, ego_y, yaw)


def render_crosswalks(crosswalk_set, ego_x, ego_y, yaw):
    """
    횡단보도 렌더링 (singlecrosswalk_set)
    각 횡단보도는 4~5개 꼭짓점 폴리곤
    """
    img = np.zeros((SIZE, SIZE), dtype=np.uint8)
    for item in crosswalk_set:
        pts_raw = item.get("points", [])
        if len(pts_raw) < 3:
            continue
        pts   = np.array([[p[0], p[1]] for p in pts_raw], dtype=np.float64)
        local = map_to_local(pts, ego_x, ego_y, yaw)
        if not pts_in_view(local):
            continue
        pix = local_to_pixel(local)
        cv2.fillPoly(img, [pix.reshape(-1, 1, 2)], 1)
    return (img > 0).astype(np.uint8)


def _point_key(point):
    if len(point) < 2:
        return None
    return (round(point[0], 3), round(point[1], 3))


def build_stopline_point_index(stopline_set):
    point_index = {}
    for stopline in stopline_set:
        for point in stopline.get("points", []):
            key = _point_key(point)
            if key is not None:
                point_index.setdefault(key, []).append(stopline)
    return point_index


def _find_stoplines_by_ids(stopline_ids, stopline_by_id):
    stoplines = []
    seen_stoplines = set()
    for stopline_id in stopline_ids:
        stopline = stopline_by_id.get(stopline_id)
        if stopline is None or stopline_id in seen_stoplines:
            continue
        seen_stoplines.add(stopline_id)
        stoplines.append(stopline)
    return stoplines


def _find_stoplines_for_link_ids(link_ids, link_by_id, node_by_id, stopline_point_index):
    stoplines = []
    seen_stoplines = set()
    for link_id in link_ids:
        link = link_by_id.get(link_id)
        if link is None:
            continue
        for node_id in (link.get("from_node_idx"), link.get("to_node_idx")):
            node = node_by_id.get(node_id)
            if node is None or not node.get("on_stop_line"):
                continue
            key = _point_key(node.get("point", []))
            if key is None:
                continue
            for stopline in stopline_point_index.get(key, []):
                stopline_id = stopline.get("idx")
                if stopline_id in seen_stoplines:
                    continue
                seen_stoplines.add(stopline_id)
                stoplines.append(stopline)
    return stoplines


def build_synced_signal_map(synced_traffic_light_set):
    mapping = {}
    for synced in synced_traffic_light_set:
        synced_id = synced.get("idx")
        signal_ids = [str(x) for x in synced.get("signal_id_list", [])]
        if synced_id:
            mapping[str(synced_id)] = signal_ids
    return mapping


def build_traffic_stopline_map(traffic_light_set, synced_traffic_light_set,
                               link_set, node_set, stopline_set):
    link_by_id = {link.get("idx"): link for link in link_set}
    node_by_id = {node.get("idx"): node for node in node_set}
    stopline_by_id = {stopline.get("idx"): stopline for stopline in stopline_set}
    stopline_point_index = build_stopline_point_index(stopline_set)

    mapping = {}
    for light in traffic_light_set:
        if light.get("type") not in TRAFFIC_LIGHT_TYPES:
            continue
        light_id = str(light.get("idx"))
        if light_id in DIRECT_SIGNAL_STOPLINE_IDS:
            stoplines = _find_stoplines_by_ids(
                DIRECT_SIGNAL_STOPLINE_IDS[light_id], stopline_by_id)
        else:
            stoplines = _find_stoplines_for_link_ids(
                light.get("link_id_list", []), link_by_id, node_by_id,
                stopline_point_index)
        if stoplines:
            mapping[light_id] = stoplines

    for synced in synced_traffic_light_set:
        synced_id = str(synced.get("idx"))
        if synced_id in DIRECT_SIGNAL_STOPLINE_IDS:
            stoplines = _find_stoplines_by_ids(
                DIRECT_SIGNAL_STOPLINE_IDS[synced_id], stopline_by_id)
            if stoplines:
                mapping[synced_id] = stoplines
            continue
        signal_ids = set(str(x) for x in synced.get("signal_id_list", []))
        if not signal_ids:
            continue
        if not any(light_id in mapping for light_id in signal_ids):
            continue
        stoplines = _find_stoplines_for_link_ids(
            synced.get("link_id_list", []), link_by_id, node_by_id,
            stopline_point_index)
        if stoplines:
            mapping[synced_id] = stoplines

    rospy.loginfo("[TrafficLight] 신호등-정지선 매핑 %d개 생성", len(mapping))
    return mapping


def traffic_light_status_color(status):
    status = int(status)
    if status in TL_STATUS_BLUE:
        return "blue"
    if status in TL_STATUS_YELLOW:
        return "yellow"
    if status in TL_STATUS_GREEN:
        return "green"
    if status in TL_STATUS_RED:
        return "red"
    return None


def render_signal_stoplines(traffic_stopline_map, traffic_light_status, ego_x, ego_y, yaw):
    imgs = {k: np.zeros((SIZE, SIZE), dtype=np.uint8)
            for k in ("red", "yellow", "green", "blue")}

    for traffic_light_idx, stoplines in traffic_stopline_map.items():
        status = traffic_light_status.get(traffic_light_idx)
        if status is None:
            continue
        color = traffic_light_status_color(status)
        if color is None:
            continue

        for stopline in stoplines:
            pts_raw = stopline.get("points", [])
            if len(pts_raw) < 2:
                continue
            pts = np.array([[p[0], p[1]] for p in pts_raw], dtype=np.float64)
            local = map_to_local(pts, ego_x, ego_y, yaw)
            if not pts_in_view(local):
                continue
            pix = local_to_pixel(local)
            line_w = float(stopline.get("lane_width", 0.6))
            thick = max(1, int(round(max(0.2, line_w) / RES)))
            cv2.polylines(imgs[color], [pix.reshape(-1, 1, 2)], False, 1, thickness=thick)

    return {k: (v > 0).astype(np.uint8) for k, v in imgs.items()}


# ══════════════════════════════════════════════════════════════════════════════
# 5. OccupancyGrid / MarkerArray 생성
# ══════════════════════════════════════════════════════════════════════════════

def make_grid(img, stamp):
    grid = OccupancyGrid()
    grid.header.stamp    = stamp
    grid.header.frame_id = "base_link"
    grid.info.resolution = RES
    grid.info.width      = SIZE
    grid.info.height     = SIZE
    grid.info.origin.position.x    = 0.0
    grid.info.origin.position.y    = -LAT_HALF
    grid.info.origin.position.z    = 0.0
    grid.info.origin.orientation.w = 1.0
    bin_img   = (img > 0).astype(np.int8)
    grid.data = (bin_img.flatten() * OCCUPIED_VALUE).tolist()
    return grid


def make_ego_cube_marker(stamp, rear_to_center):
    ma = MarkerArray()
    m  = Marker()
    m.header.stamp    = stamp
    m.header.frame_id = "base_link"
    m.ns     = "ego_vehicle"
    m.id     = 0
    m.type   = Marker.CUBE
    m.action = Marker.ADD
    m.lifetime = rospy.Duration(EGO_MARKER_LIFE)
    m.pose.position.x    = rear_to_center
    m.pose.position.y    = 0.0
    m.pose.position.z    = EGO_CUBE_SIZE_Z / 2.0
    m.pose.orientation.w = 1.0
    m.scale.x = EGO_CUBE_SIZE_X
    m.scale.y = EGO_CUBE_SIZE_Y
    m.scale.z = EGO_CUBE_SIZE_Z
    m.color.r = 0.2; m.color.g = 0.8; m.color.b = 1.0; m.color.a = 0.9
    ma.markers.append(m)
    return ma


# ══════════════════════════════════════════════════════════════════════════════
# 6. 시각화 이미지
# ══════════════════════════════════════════════════════════════════════════════

_VIZ_COLORS = {
    "background":   ( 30,  30,  30),
    "drivable":     (100, 100, 100),
    "yellow":       (  0, 220, 255),
    "white_solid":  (255, 255, 255),
    "white_dashed": (180, 180, 180),
    "blue":         (255, 100,  50),
    "vehicle":      ( 60,  60, 220),
    "pedestrian":   ( 30, 140, 255),
    "crosswalk":    (200, 200, 200),   # 횡단보도 — 밝은 회색
    "tl_red":       (  0,   0, 255),
    "tl_yellow":    (  0, 220, 255),
    "tl_green":     (  0, 220,   0),
    "tl_blue":      (255,   0,   0),
}

def make_viz_image(drivable, lanes, vehicle, pedestrian, crosswalk_img, stoplines):
    viz = np.full((SIZE, SIZE, 3), _VIZ_COLORS["background"], dtype=np.uint8)
    viz[drivable                == 1] = _VIZ_COLORS["drivable"]
    viz[crosswalk_img           == 1] = _VIZ_COLORS["crosswalk"]
    viz[lanes["white_dashed"]   == 1] = _VIZ_COLORS["white_dashed"]
    viz[lanes["white_solid"]    == 1] = _VIZ_COLORS["white_solid"]
    viz[lanes["blue"]           == 1] = _VIZ_COLORS["blue"]
    viz[lanes["yellow"]         == 1] = _VIZ_COLORS["yellow"]
    viz[stoplines["red"]        == 1] = _VIZ_COLORS["tl_red"]
    viz[stoplines["yellow"]     == 1] = _VIZ_COLORS["tl_yellow"]
    viz[stoplines["green"]      == 1] = _VIZ_COLORS["tl_green"]
    viz[stoplines["blue"]       == 1] = _VIZ_COLORS["tl_blue"]
    viz[pedestrian              == 1] = _VIZ_COLORS["pedestrian"]
    viz[vehicle                 == 1] = _VIZ_COLORS["vehicle"]
    viz = cv2.rotate(viz, cv2.ROTATE_90_CLOCKWISE)
    viz = cv2.flip(viz, 0)
    return viz


# ══════════════════════════════════════════════════════════════════════════════
# 7. ROS 노드
# ══════════════════════════════════════════════════════════════════════════════

class BEVMapGenerator:
    def __init__(self):
        rospy.init_node("bev_map_generator", anonymous=False)

        self.csv_offset_x, self.csv_offset_y, self.conv_rad, self.ref_lla, self.ref_utm_zone = \
            compute_csv_offset_from_origin(FIXED_REF_LLA, GLOBAL_INFO_JSON)
        rospy.loginfo("[Init] 원점 변환 완료 | offset=(%.3f, %.3f), conv=%.4fdeg",
                      self.csv_offset_x, self.csv_offset_y, math.degrees(self.conv_rad))

        self.lane_set           = load_json(os.path.join(MAP_DIR, "lane_boundary_set.json"))
        self.drivable_triangles = build_drivable_triangles(
            os.path.join(MAP_DIR, "road_mesh_out_line.json"))
        self.crosswalk_set      = load_json(os.path.join(MAP_DIR, "singlecrosswalk_set.json"))
        self.traffic_light_set  = load_json(os.path.join(MAP_DIR, "traffic_light_set.json"))
        self.synced_traffic_light_set = load_json(os.path.join(MAP_DIR, "synced_traffic_light_set.json"))
        self.stopline_set       = load_json(os.path.join(MAP_DIR, "stoplane_marking_set.json"))
        self.link_set           = load_json(os.path.join(MAP_DIR, "link_set.json"))
        self.node_set           = load_json(os.path.join(MAP_DIR, "node_set.json"))
        self.synced_signal_map  = build_synced_signal_map(self.synced_traffic_light_set)
        self.traffic_stopline_map = build_traffic_stopline_map(
            self.traffic_light_set, self.synced_traffic_light_set,
            self.link_set, self.node_set, self.stopline_set)
        rospy.loginfo("[Init] lane=%d, triangles=%d, crosswalk=%d, traffic_light=%d, stopline=%d",
                      len(self.lane_set), len(self.drivable_triangles),
                      len(self.crosswalk_set), len(self.traffic_light_set),
                      len(self.stopline_set))

        self.ego_pose   = None
        self.latest_obj = None
        self.traffic_light_status = {}
        self.lock       = threading.Lock()
        self.ego_rear_to_center = float(rospy.get_param("~ego_rear_to_center", EGO_REAR_TO_CENTER))
        self.bridge     = CvBridge()

        self.pub_drivable   = rospy.Publisher("/bev/drivable",   OccupancyGrid, queue_size=1)
        self.pub_lane       = rospy.Publisher("/bev/lane",       OccupancyGrid, queue_size=1)
        self.pub_vehicle    = rospy.Publisher("/bev/vehicle",    OccupancyGrid, queue_size=1)
        self.pub_pedestrian = rospy.Publisher("/bev/pedestrian", OccupancyGrid, queue_size=1)
        self.pub_crosswalk  = rospy.Publisher("/bev/crosswalk",  OccupancyGrid, queue_size=1)
        self.pub_stopline_red    = rospy.Publisher("/bev/stopline_red",    OccupancyGrid, queue_size=1)
        self.pub_stopline_yellow = rospy.Publisher("/bev/stopline_yellow", OccupancyGrid, queue_size=1)
        self.pub_stopline_green  = rospy.Publisher("/bev/stopline_green",  OccupancyGrid, queue_size=1)
        self.pub_stopline_blue   = rospy.Publisher("/bev/stopline_blue",   OccupancyGrid, queue_size=1)
        self.pub_ego_marker = rospy.Publisher("/bev/ego_marker", MarkerArray,   queue_size=1)
        self.pub_viz        = rospy.Publisher("/bev/viz",        Image,         queue_size=1)

        rospy.Subscriber("/Ego_topic",    EgoVehicleStatus, self._ego_cb, queue_size=1)
        rospy.Subscriber("/Object_topic", ObjectStatusList, self._obj_cb, queue_size=1)
        rospy.Subscriber("/GetTrafficLightStatus", GetTrafficLightStatus,
                         self._traffic_light_cb, queue_size=10)

        rospy.Timer(rospy.Duration(1.0 / PUBLISH_HZ), self._timer_cb)
        rospy.loginfo("[BEVMap] 노드 시작 | %.0fHz", PUBLISH_HZ)
        rospy.spin()

    def _ego_cb(self, msg):
        with self.lock:
            self.ego_pose = {
                "x":   msg.position.x,
                "y":   msg.position.y,
                "yaw": math.radians(msg.heading),
            }

    def _obj_cb(self, msg):
        with self.lock:
            self.latest_obj = msg

    def _traffic_light_cb(self, msg):
        traffic_light_idx = str(msg.trafficLightIndex)
        traffic_light_status = int(msg.trafficLightStatus)
        with self.lock:
            self.traffic_light_status[traffic_light_idx] = traffic_light_status
            for signal_id in self.synced_signal_map.get(traffic_light_idx, []):
                self.traffic_light_status[signal_id] = traffic_light_status

    def _timer_cb(self, event):
        with self.lock:
            if self.ego_pose is None:
                return
            ego = self.ego_pose.copy()
            obj = self.latest_obj
            traffic_light_status = self.traffic_light_status.copy()

        ego_x = ego["x"]
        ego_y = ego["y"]
        yaw   = ego["yaw"]
        stamp = rospy.Time.now()

        drivable_img   = render_drivable(self.drivable_triangles, ego_x, ego_y, yaw)
        lanes          = render_lanes(self.lane_set, ego_x, ego_y, yaw)
        lane_img       = np.maximum.reduce(list(lanes.values()))
        vehicle_img    = render_vehicles(obj, ego_x, ego_y, yaw)
        pedestrian_img = render_pedestrians(obj, ego_x, ego_y, yaw)
        crosswalk_img  = render_crosswalks(self.crosswalk_set, ego_x, ego_y, yaw)
        stoplines      = render_signal_stoplines(
            self.traffic_stopline_map, traffic_light_status, ego_x, ego_y, yaw)

        self.pub_drivable.publish(  make_grid(drivable_img,   stamp))
        self.pub_lane.publish(      make_grid(lane_img,       stamp))
        self.pub_vehicle.publish(   make_grid(vehicle_img,    stamp))
        self.pub_pedestrian.publish(make_grid(pedestrian_img, stamp))
        self.pub_crosswalk.publish( make_grid(crosswalk_img,  stamp))
        self.pub_stopline_red.publish(   make_grid(stoplines["red"],    stamp))
        self.pub_stopline_yellow.publish(make_grid(stoplines["yellow"], stamp))
        self.pub_stopline_green.publish( make_grid(stoplines["green"],  stamp))
        self.pub_stopline_blue.publish(  make_grid(stoplines["blue"],   stamp))
        self.pub_ego_marker.publish(make_ego_cube_marker(stamp, self.ego_rear_to_center))

        viz = make_viz_image(drivable_img, lanes, vehicle_img, pedestrian_img, crosswalk_img, stoplines)
        self.pub_viz.publish(self.bridge.cv2_to_imgmsg(viz, encoding="bgr8"))
        cv2.imshow("BEV Segmentation Map", viz)
        cv2.waitKey(1)


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        BEVMapGenerator()
    except rospy.ROSInterruptException:
        pass
