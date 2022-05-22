from rl.model.value_model.value_advance import ValueAdvance
from rl.common.deep_model import NoisyMLP

class ValueV1Noise(ValueAdvance):
    def __init__(self,config):
        super(ValueV1Noise, self).__init__(config)

        self.encoder = NoisyMLP(self.config["encoder"])
        self.v_layer = NoisyMLP(self.config["v_layer"])
        self.a_layer = NoisyMLP(self.config["a_layer"])

    def v_a(self, x):
        b = x.shape[0]
        x = x.view(b,-1)
        x = self.encoder(x)
        a = self.a_layer(x)
        v = self.v_layer(x)
        return v + (a - a.mean())
