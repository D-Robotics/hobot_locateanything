from leap_llm.nn.modules import FakeQuantLinear, FakeQuantMul, FakeQuantSwish
from leap_llm.nn.utils import Module


class MLP(Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.w1 = FakeQuantLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.w3 = FakeQuantLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.w2 = FakeQuantLinear(self.intermediate_size, self.hidden_size, bias=False)
        # self.act_fn = ACT2FN[config.hidden_act]
        self.act_fn = FakeQuantSwish(True, 16)

        self.mul = FakeQuantMul(quantized=False)

    def build(self, hidden_state):

        x = self.w1(hidden_state)
        x = self.act_fn(x)
        up_proj_h = self.w3(hidden_state)
        x = self.mul(x, up_proj_h)
        return self.w2(x)

    def forward(self, hidden_state):
        x = self.w1(hidden_state)
        x = self.act_fn(x)
        up_proj_h = self.w3(hidden_state)
        x = self.mul(x, up_proj_h)
        return self.w2(x)
