# MORAI Scenario Runner

MORAI 시뮬레이터용 시나리오 자동화 실행기입니다.

## 요구사항

- Python 3.8+
- ROS 2 (GT-BEV 외부 제어 모드 사용 시)
- MORAI Simulator
- MGeo 맵 데이터

## 설치

```bash
git clone <repo_url>
cd morai_scenario_runner
```

### 로컬 경로 설정

MGeo 맵 경로는 머신마다 다르므로 별도 파일로 관리합니다.

```bash
cp scenario_runner/config/local.yaml.example scenario_runner/config/local.yaml
```

`local.yaml`을 열어 MGeo 맵 경로를 수정합니다.

```yaml
paths:
  mgeo_root: /path/to/your/mgeo_ws/R_KR_PG_KATRI
```

`local.yaml`은 `.gitignore`에 포함되어 있어 커밋되지 않습니다.

## 실행

프로젝트 루트에서 실행합니다.

```bash
# 기본 주행 시나리오
python scenario_runner/main_runner.py --zone urban --scenario basic_drive

# 급제동 시나리오
python scenario_runner/main_runner.py --zone urban --scenario sudden_brake
```

## 시나리오 구성

시나리오 파라미터는 `scenario_runner/config/urban.yaml`에서 설정합니다.

| 파라미터 | 설명 |
|---|---|
| `gt_bev_max_velocity_kmh` | 에고 차량 최대 속도 |
| `brake_trigger_gap_m` | NPC 급제동 트리거 거리 |
| `npc_brake_decel_mps2` | NPC 제동 감속도 (0이면 즉각 정지) |
| `lead_spawn_gap_m` | 에고~리드 NPC 초기 이격 거리 |
| `npc_count_min` / `npc_count_max` | 배경 NPC 수 |
