# MORAI Scenario Runner

MORAI Simulator gRPC API와 로컬 MGeo 데이터를 사용해 K-City urban 구간의 주행 시나리오를 실행하는 Python runner입니다.

현재 코드는 `urban` zone만 지원하며, 기본 주행, 급제동 NPC, 정체 대기열, 보행자 양보 시나리오를 제공합니다. 대부분의 시나리오는 GT_BEV expert controller를 외부 프로세스로 띄워 Ego 차량 제어에 사용합니다.

## Requirements

- Python 3.8+
- MORAI Simulator
- MORAI gRPC server
  - 기본 접속 주소: `127.0.0.1:7789`
- ROS 환경
  - GT_BEV external control 사용 시 필요
  - `rospy`, `morai_msgs`, `nav_msgs`, `geometry_msgs`를 import할 수 있어야 합니다.
- MGeo map data
  - 기본 경로: `third_party/GT_BEV/R_KR_PG_KATRI`
- Python packages
  - `pyyaml`
  - `numpy`
  - `matplotlib`은 링크 추출 도구 또는 route debug plot 사용 시 필요
  - `grpcio`, `grpcio-tools` 등 gRPC 관련 패키지는 `third_party/grpc_inha_univ/requirements.txt` 참고

## Project Structure

```text
scenario_runner/
  main_runner.py              # CLI entrypoint
  base_scenario.py            # scenario lifecycle base class
  config/
    global.yaml               # MORAI/gRPC/path 기본 설정
    local.yaml.example        # 로컬 override 예시
    urban.yaml                # urban scenario 설정
    urban_links.yaml          # urban zone route candidate links
  zones/
    urban_scenarios.py        # urban scenario implementations
  utils/
    grpc_client.py            # MORAI gRPC wrapper
    map_loader.py             # local MGeo JSON loader
    link_graph.py             # link connectivity graph
    route_utils.py            # route generation helper
    transform_utils.py        # geometry/transform helper
    gt_bev_expert.py          # GT_BEV runtime config exporter/process manager
  tools/
    extract_zone_links.py     # interactive zone link extraction tool
third_party/
  grpc_inha_univ/             # MORAI gRPC Python client/protobuf API
  GT_BEV/                     # GT_BEV expert stack and MGeo data
```

## Configuration

기본 설정은 `scenario_runner/config/global.yaml`에 있습니다.

```yaml
grpc:
  host: 127.0.0.1
  port: 7789
  client_key: scenario_runner
morai:
  map_name: R_KR_PG_K-City
  ego_vehicle_model: 2023_Hyundai_Ioniq5
paths:
  grpc_src: third_party/grpc_inha_univ/src
  mgeo_root: third_party/GT_BEV/R_KR_PG_KATRI
```

로컬 환경에서 경로 또는 접속 정보를 바꾸려면 `local.yaml`을 만들어 override합니다. 이 파일은 git에 포함되지 않습니다.

```powershell
Copy-Item scenario_runner/config/local.yaml.example scenario_runner/config/local.yaml
```

예시:

```yaml
grpc:
  host: 127.0.0.1
  port: 7789
paths:
  mgeo_root: D:/maps/R_KR_PG_KATRI
```

시나리오별 파라미터는 `scenario_runner/config/urban.yaml`에서 조정합니다.

## Run

프로젝트 루트에서 실행합니다.

```powershell
python scenario_runner/main_runner.py --zone urban --scenario basic_drive
```

사용 가능한 시나리오:

```powershell
# 기본 주행
python scenario_runner/main_runner.py --zone urban --scenario basic_drive

# 선행 NPC 급제동
python scenario_runner/main_runner.py --zone urban --scenario sudden_brake

# 정체 대기열
python scenario_runner/main_runner.py --zone urban --scenario traffic_jam

# 횡단보도 보행자 양보
python scenario_runner/main_runner.py --zone urban --scenario pedestrian_yield
```

## Typical Startup Order

1. MORAI Simulator를 실행하고 gRPC server가 열려 있는지 확인합니다.
2. ROS 및 GT_BEV 실행에 필요한 환경을 source/setup합니다.
3. 프로젝트 루트에서 원하는 scenario command를 실행합니다.

`urban.yaml`의 기본 `drive_control_mode`는 `gt_bev_external`입니다. 이 모드에서는 runner가 route/config를 `third_party/GT_BEV/.runtime/scenario_runner/` 아래에 생성하고 `python -m expert.src.main` 프로세스를 실행합니다.

## Zone Link Extraction Tool

새 urban link 후보 목록을 만들려면 MGeo root를 지정해 interactive tool을 실행합니다. Matplotlib 창에서 zone polygon 꼭짓점을 클릭하고 Enter를 누르면 output YAML이 저장됩니다.

```powershell
python scenario_runner/tools/extract_zone_links.py --mgeo-root third_party/GT_BEV/R_KR_PG_KATRI --output scenario_runner/config/urban_links.yaml --zone-name urban
```

## Notes

- `main_runner.py`는 반드시 프로젝트 루트 기준 상대 경로로 config를 읽습니다.
- `local.yaml`, `.runtime/`, `debug_routes/`는 `.gitignore` 대상입니다.
- `pedestrian_yield`는 MGeo의 `singlecrosswalk_set.json` 또는 `crosswalk_set.json`이 필요합니다.
- `route_debug_enabled: true`인 시나리오는 `scenario_runner/debug_routes/`에 route debug output을 생성할 수 있습니다.
