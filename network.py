from flax import nnx
from typing import Optional


class MLP(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(768, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, 2, rngs=rngs)

    def __call__(self, x, rngs: Optional[nnx.Rngs] = None):
        x = nnx.relu(self.linear1(x))
        x = self.linear2(x)
        return x