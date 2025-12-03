# Bolted Prediction API

A LitServe-based API for predicting bolting probability from plant image embeddings using the decoder_bolted3 model.

## Features

- Fast batch processing with JAX/Flax
- Accepts 768-dimensional embeddings from DINOv3
- Returns bolting probability [0, 1]
- Lightweight inference API

## Model

Model: `decoder_bolted3` - A JAX/Flax MLP classifier
- Input: 768-dimensional embedding (DINOv3 CLS token)
- Output: Bolting probability [0, 1]

## API Usage

### Request Format

```json
{
  "embedding": [0.123, -0.456, ...]
}
```

**Parameters:**
- `embedding` (required): 768-dimensional array of floats (DINOv3 CLS token embedding)

### Response Format

```json
{
  "bolted_probability": 0.85
}
```

**Response fields:**
- `bolted_probability`: Float between 0 and 1 indicating the probability that the plant is bolted

### Example Usage

#### Python

```python
import requests
import numpy as np

# Assume you have an embedding from the embedding API
embedding = [0.123, -0.456, ...]  # 768-dim array

# Make request
response = requests.post(
    "http://localhost:8803/predict",
    json={"embedding": embedding}
)

result = response.json()
bolted_prob = result["bolted_probability"]
print(f"Bolting probability: {bolted_prob:.2%}")
```

#### cURL

```bash
# First get embedding from image
IMAGE_BASE64=$(base64 -w 0 plant.jpg)
EMBEDDING=$(curl -X POST http://localhost:8803/predict \
  -H "Content-Type: application/json" \
  -d "{\"image_data\": \"$IMAGE_BASE64\", \"embedding_types\": [\"cls_token\"]}" | jq -r '.cls_token')

# Then get bolting prediction
curl -X POST http://localhost:8804/predict \
  -H "Content-Type: application/json" \
  -d "{\"embedding\": $EMBEDDING}"
```

## Development

### Local Setup

1. Install dependencies:
```bash
uv sync --dev
```

2. Run the server:
```bash
uv run python app/main.py
```

### Docker Setup

Build and run with Docker Compose:

```bash
docker compose up --build bolted
```

The API will be available at `http://localhost:8804`.

## Configuration

Edit `app/main.py` to configure:
- `max_batch_size`: Maximum number of embeddings per batch (default: 16)
- `batch_timeout`: Time to wait for batch completion in seconds (default: 0.01)
- `num_api_servers`: Number of parallel API server instances (default: 1)
- `model_checkpoint_path`: Path to model checkpoint (default: "/app/model")

## Performance

The API uses:
- JAX for efficient numerical computation
- Batch inference for improved throughput
- Lightweight model (768 → 256 → 2 MLP)

## Use Cases

- Automated bolting detection in plant phenotyping
- Quality control in plant production
- Research on plant stress responses
- Integration with plant monitoring systems
