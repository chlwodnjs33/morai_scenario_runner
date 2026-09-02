class ControlInput:
    def __init__(self, acc, steering):
        if acc > 0:
            self.accel = acc
            self.brake = 0.
        else:
            self.accel = 0.
            self.brake = -acc
        self.front_steer = steering
        self.rear_steer = 0.
        self.longlCmdType = 1
