class BaseScenario:
    zone_name = None
    scenario_name = None

    def __init__(self, grpc_client, map_loader, global_cfg, scenario_cfg):
        self.grpc = grpc_client
        self.map_loader = map_loader
        self.global_cfg = global_cfg
        self.cfg = scenario_cfg

    def setup(self):
        raise NotImplementedError

    def run_timeline(self):
        raise NotImplementedError

    def cleanup(self):
        pass

    def run(self):
        self.setup()
        try:
            self.run_timeline()
        finally:
            self.cleanup()
