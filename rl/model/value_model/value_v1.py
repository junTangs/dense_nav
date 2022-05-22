from rl.model.value_model.value_advance import ValueAdvance
from rl.common.deep_model import MLP


class ValueV1(ValueAdvance):
    def __init__(self,config):
        super(ValueV1, self).__init__(config)

        self.encoder = MLP(self.config["encoder"])
        self.v_layer = MLP(self.config["v_layer"])
        self.a_layer = MLP(self.config["a_layer"])

    def v_a(self, x):
        b = x.shape[0]
        x = x.view(b,-1)
        x = self.encoder(x)
        a = self.a_layer(x)
        v = self.v_layer(x)
        return v + (a - a.mean())

