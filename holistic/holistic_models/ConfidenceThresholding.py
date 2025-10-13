from interface.WTA import WTA

class ConfidenceThresholdingWtaModule(WTA):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def wta(self, t, prob, nav_outs):
        probs, a_t = prob.max(1)  
        probs_cpu = probs.cpu()
        return [prob < self.threshold for prob in probs_cpu]
