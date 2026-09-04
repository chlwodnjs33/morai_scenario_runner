# MORAI Scenario Runner

MORAI Simulator 26.R1과 `R_KR_PR_K-city_2025` 맵에서 공식 전역경로를 주행하는 Python 시나리오 러너입니다.

시나리오 actor와 환경은 gRPC로 관리하고, Ego 차량은 GT_BEV가 ROS `/ctrl_cmd`를 통해 제어합니다. 정적·동적 장애물, 회전교차로 교통, 고속도로 상호작용 및 터널 급정거 상황을 한 번의 폐루프 주행에서 시험할 수 있습니다.

## Requirements

- Python 3.8+
- ROS1 Noetic 및 `rosbridge_server`
- MORAI Simulator 26.R1
- MORAI gRPC server: `172.19.0.1:7789`
- MGeo: `third_party/GT_BEV/R_KR_PR_K-city_2025`
- Python packages: `pyyaml`, `numpy`, `matplotlib`, `grpcio`, `grpcio-tools`

## Project Structure

```text
scenario_runner/
  main_runner.py                 # 실행 진입점
  config/
    global.yaml                  # gRPC, 맵, Ego 설정
    full_loop.yaml               # 시나리오 및 랜덤화 설정
    full_loop_links.yaml         # 전역경로와 구간 정보
  zones/
    full_loop_scenario.py        # 새 맵 폐루프 시나리오
  utils/
    grpc_client.py               # MORAI gRPC wrapper
    gt_bev_expert.py             # GT_BEV 실행 및 runtime 설정
third_party/
  grpc_inha_univ/                # MORAI gRPC API
  GT_BEV/                        # Ego 제어기와 MGeo 데이터
```

## Configuration

기본 설정은 `scenario_runner/config/global.yaml`에 있습니다.

```yaml
grpc:
  host: 172.19.0.1
  port: 7789
  client_key: scenario_runner
morai:
  map_name: R_KR_PR_K-city_2025
  ego_vehicle_model: 2023_Hyundai_Ioniq5
paths:
  grpc_src: third_party/grpc_inha_univ/src
  mgeo_root: third_party/GT_BEV/R_KR_PR_K-city_2025
```

MORAI Network Settings는 다음과 같이 사용합니다.

- Simulator Network: `GRPC`, Host PORT `7789`
- Ego Network Cmd Control: `ROS`, `/ctrl_cmd`
- Ego Network Publisher/Subscriber/Service: `ROS`, `127.0.0.1:9090`

## Setup

메시지 패키지를 처음 빌드하거나 변경한 경우:

```bash
cd ~/morai_scenario_runner/third_party/GT_BEV
source /opt/ros/noetic/setup.bash
catkin_make
```

별도 터미널에서 rosbridge를 실행합니다.

```bash
cd ~/morai_scenario_runner
source /opt/ros/noetic/setup.bash
source third_party/GT_BEV/devel/setup.bash
roslaunch rosbridge_server rosbridge_websocket.launch
```

## Run

### 전체 폐루프

원래 START에서 출발하여 정적 장애물부터 터널까지 전체 경로를 주행합니다. END는 START와 같은 지점입니다.

```bash
cd ~/morai_scenario_runner
source /opt/ros/noetic/setup.bash
source third_party/GT_BEV/devel/setup.bash
python3 scenario_runner/main_runner.py --zone full_loop --scenario full_loop
```

### 정적 장애물 이후부터 시작

모든 actor는 동일하게 생성하지만 Ego만 정적 장애물 다음 지점에서 출발합니다. END는 전체 폐루프와 같습니다.

```bash
cd ~/morai_scenario_runner
source /opt/ros/noetic/setup.bash
source third_party/GT_BEV/devel/setup.bash
python3 scenario_runner/main_runner.py --zone full_loop --scenario full_loop_after_static
```

### 고속도로·터널 집중 테스트

고속도로 후반부부터 시작하여 터널 급정거 이벤트와 END 도착을 빠르게 확인합니다.

```bash
python3 scenario_runner/main_runner.py --zone full_loop --scenario full_loop_highway_tunnel_test
```

### MGeo 경로 뷰어

시나리오 실행 후 별도 터미널에서 현재 경로와 START/END 지점을 확인합니다.

```bash
cd ~/morai_scenario_runner
source /opt/ros/noetic/setup.bash
source third_party/GT_BEV/devel/setup.bash
python3 third_party/GT_BEV/src/expert/src/path/mgeo_visualization.py
```

뷰어에서 노란색 선은 주행 경로, 초록색 표시는 START, 빨간색 표시는 END입니다.

`full_loop`과 `full_loop_after_static`은 END 도착 후 다음 시나리오를 자동 실행합니다. 한 번만 실행하려면 `--once`를 추가합니다.

```bash
python3 scenario_runner/main_runner.py --zone full_loop --scenario full_loop --once
```

## Scenario

- 공식 폐루프 경로: 39 links, 4,430 points, 약 2,184.6 m
- 날씨: 실행마다 `SUNNY`와 `FOGGY` 교대
- 시간: 실행마다 11시, 13시, 15시 중 무작위 선택
- 정적 장애물: 회피 가능한 소형 물체 중 무작위 선택
- 동적 장애물: 보행자·자전거 등이 Ego 접근 시 도로를 횡단
- 회전교차로: 승용차 크기 NPC 6대를 약 18.7m 간격, 4.0m/s로 고정하여 반복 주행
- 고속도로: NPC 6~9대와 선행 차량 감속 이벤트
- 터널: 안전거리로 생성된 선행 NPC의 급정거 및 재출발 후 제거

## Notes

- 정상 gRPC 연결 로그: `[gRPC] connected to 172.19.0.1:7789`
- 정상 ROS 연결은 `/Ego_topic`과 `/ctrl_cmd`의 publisher/subscriber로 확인합니다.
- 실행 중 Network Settings 창을 열거나 MORAI를 Pause하면 Ego 상태 갱신이 멈출 수 있습니다.
- runtime 파일은 `third_party/GT_BEV/.runtime/scenario_runner/`에 생성됩니다.
- `Ctrl+C`로 시나리오와 자동 반복을 종료합니다.
