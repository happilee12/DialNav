from interface.WTA import WTA

class FixedIntervalWtaModule(WTA):
    def __init__(self, interval=4):
        self.interval = interval

    def wta(self, t, prob, nav_outs):
        if t % self.interval == 0:
            return [True] * len(prob)
        else:
            return [False] * len(prob)
