from flax import nnx
from typing import Optional


class MLP(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        output_heads: dict[str, int],
        *,
        rngs: nnx.Rngs,
    ):
        self.linear1 = nnx.Linear(input_dim, 256, rngs=rngs)
        self.heads = nnx.Dict(
            {
                name: nnx.Linear(256, dim, rngs=rngs)
                for name, dim in output_heads.items()
            }
        )

    def __call__(self, x, rngs: Optional[nnx.Rngs] = None):
        x = nnx.relu(self.linear1(x))
        return {name: head(x) for name, head in self.heads.items()}
