class BaseScenario:
    zone_name = None
    scenario_name = None

    def __init__(self, grpc_client, map_loader, global_cfg, scenario_cfg):
        self.grpc = grpc_client
        self.map_loader = map_loader
        self.global_cfg = global_cfg
        self.cfg = scenario_cfg
        self.on_lap_end = None

    def setup(self):
        raise NotImplementedError

    def run_timeline(self):
        raise NotImplementedError

    def cleanup(self):
        pass

    def notify_lap_end(self, lap_num):
        """랩(구간) 완료 후 재시작(리스폰) 직전에 호출. 데이터 콜렉터가 episode를 전환할 수 있도록 알림."""
        if callable(self.on_lap_end):
            self.on_lap_end(lap_num)

    def run(self):
        try:
            self.setup()
            self.run_timeline()
        finally:
            self.cleanup()
