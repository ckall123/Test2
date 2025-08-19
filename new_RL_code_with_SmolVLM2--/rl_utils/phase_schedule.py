class Curriculum:
    def __init__(self, start_lambda_v=0.0, end_lambda_v=0.7, warmup_steps=20_000, ramp_steps=100_000, obj_min=2, obj_max=6):
        self.s, self.e = start_lambda_v, end_lambda_v
        self.warm, self.ramp = warmup_steps, ramp_steps
        self.obj_min, self.obj_max = obj_min, obj_max
        self.t = 0
    def update(self, t): self.t = int(t)
    def lambda_v_at(self, step):
        d = max(0, step - self.warm); frac = min(1.0, d/float(self.ramp))
        return (1-frac)*self.s + frac*self.e
    def num_objects(self):
        frac = min(1.0, max(0.0, (self.t - self.warm)/float(self.ramp)))
        return int(round(self.obj_min + frac*(self.obj_max - self.obj_min)))