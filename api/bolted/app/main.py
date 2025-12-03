from functools import partial
from pathlib import Path
import os
import jax.numpy as jnp
import litserve as ls
import numpy as np
import orbax.checkpoint
from flax import nnx
from jax.nn import softmax

try:
    from .network import MLP
except ImportError:
    from network import MLP


class BoltedAPI(ls.LitAPI):
    def __init__(self, model_checkpoint_path="/app/model", **kwargs):
        super().__init__(**kwargs)
        self.model_checkpoint_path = model_checkpoint_path

    def setup(self, device):
        self.device = device

        # Model configuration
        input_dim = 768
        output_heads = {"bolted": 2}

        # Load the model
        checkpoint_path = Path(self.model_checkpoint_path)

        # Create model with dummy initialization
        temp_model = MLP(
            input_dim=input_dim,
            output_heads=output_heads,
            rngs=nnx.Rngs(0),
        )

        # Split to get graphdef and state
        graphdef, _ = nnx.split(temp_model)

        # Load checkpoint state
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        ckpt = orbax_checkpointer.restore(checkpoint_path.resolve())

        # Merge graphdef with loaded state
        self.model = nnx.merge(graphdef, ckpt["model"])

    def decode_request(self, request):
        embedding = request["embedding"]

        if not isinstance(embedding, list):
            raise ValueError("embedding must be a list")

        if len(embedding) != 768:
            raise ValueError(
                f"embedding must have 768 dimensions, got {len(embedding)}"
            )

        return np.array(embedding, dtype=np.float32)

    def batch(self, inputs):
        # Stack embeddings into a batch
        return jnp.array(inputs)

    @partial(nnx.jit, static_argnames="self")
    def predict(self, batch_input: jnp.ndarray):
        # Run model inference
        preds = self.model(batch_input, rngs=None)

        # Get logits for bolted classification
        logits = preds["bolted"]

        # Apply softmax to get probabilities
        probs = softmax(logits, axis=-1)

        # Extract probability of bolted class (index 1)
        bolted_probs = probs[:, 1]

        return bolted_probs

    def unbatch(self, output):
        # Convert to list for individual responses
        return output.tolist()

    def encode_response(self, bolted_probability):
        return {"bolted_probability": float(bolted_probability)}


if __name__ == "__main__":
    
    model_path = os.environ.get("MODEL_PATH", "/app/model")
    port = int(os.environ.get("PORT", "8804"))

    api = BoltedAPI(
        model_checkpoint_path=model_path, max_batch_size=16, batch_timeout=0.01
    )
    server = ls.LitServer(api)
    server.run(port=port, num_api_servers=1, generate_client_file=False)
