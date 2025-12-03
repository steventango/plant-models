import base64
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import litserve as ls
import pytest
import requests

# Add app directory to sys.path
sys.path.append(str(Path(__file__).parent.parent / "app"))
from main import BoltedAPI


TEST_DATA_DIR = Path(__file__).parent / "test_data"
PLANT03_IMAGE_PATH = TEST_DATA_DIR / "2025-10-25T093000_plant03.jpg"
PLANT03_EMBEDDING_PATH = PLANT03_IMAGE_PATH.with_suffix(".json")
PLANT29_IMAGE_PATH = TEST_DATA_DIR / "2025-11-21T093000_plant29.jpg"
PLANT29_EMBEDDING_PATH = PLANT29_IMAGE_PATH.with_suffix(".json")


PORT = 8904

def run_server_proc(port=PORT, model_path="/app/model"):
    """Run the server in a separate process."""
    # Set environment variable for model path
    os.environ["MODEL_PATH"] = str(model_path)
    
    # Initialize API
    api = BoltedAPI(model_checkpoint_path=model_path)
    server = ls.LitServer(api, max_batch_size=16, batch_timeout=0.01)
    server.run(port=port, num_api_servers=1, generate_client_file=False)


def start_server(port=PORT, model_path="../../results/decoder_bolted3/model"):
    """Start the server for testing in a background process."""
    server_process = multiprocessing.Process(
        target=run_server_proc,
        args=(port, model_path),
    )
    server_process.start()
    # Give the server a moment to start
    time.sleep(5)
    return server_process


class TestBolted:
    """Test class for Bolted API integration tests."""

    @pytest.fixture(scope="class")
    def api_url(self):
        port = PORT
        # Adjust model path to be relative to this test file or absolute
        repo_root = Path(__file__).parent.parent.parent.parent
        model_path = repo_root / "results" / "decoder_bolted3" / "model"
        
        if not model_path.exists():
             # Fallback for different running contexts
             model_path = Path("/app/model")
        
        server_process = start_server(port=port, model_path=str(model_path))
        yield f"http://localhost:{port}/predict"
        
        # Cleanup
        server_process.terminate()
        server_process.join(timeout=2)
        if server_process.is_alive():
            server_process.kill()

    def get_embedding(self, image_path, cached_embedding_path):
        """
        Helper to get embedding from the Embedding API.
        """
        if cached_embedding_path.exists():
            with open(cached_embedding_path, "r") as f:
                return json.load(f)
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        embedding_api_url = "http://localhost:8803/predict"
        
        response = requests.post(
            embedding_api_url,
            json={
                "image_data": image_data,
                "embedding_types": ["cls_token"]
            },
            timeout=10
        )
        assert response.status_code == 200, f"Failed to get embedding: {response.text}"
        cls_token = response.json()["cls_token"]
        with open(cached_embedding_path, "w") as f:
            json.dump(cls_token, f)
        return cls_token

    def test_plant03_bolted(self, api_url):
        """Test that plant 03 is predicted as bolted."""
        if not PLANT03_IMAGE_PATH.exists():
            pytest.skip(f"Test image not found at {PLANT03_IMAGE_PATH}")
            
        embedding = self.get_embedding(PLANT03_IMAGE_PATH, PLANT03_EMBEDDING_PATH)
        
        response = requests.post(
            api_url,
            json={"embedding": embedding},
        )
        assert response.status_code == 200, f"Failed: {response.text}"
        result = response.json()
        
        assert "bolted_probability" in result
        prob = result["bolted_probability"]
        print(f"Plant 03 Bolted Probability: {prob}")
        
        # Assert bolted (prob > 0.5)
        assert prob > 0.5, f"Plant 03 should be bolted, got prob {prob}"

    def test_plant29_not_bolted(self, api_url):
        """Test that plant 29 is predicted as NOT bolted."""
        if not PLANT29_IMAGE_PATH.exists():
            pytest.skip(f"Test image not found at {PLANT29_IMAGE_PATH}")
            
        embedding = self.get_embedding(PLANT29_IMAGE_PATH, PLANT29_EMBEDDING_PATH)
        
        response = requests.post(
            api_url,
            json={"embedding": embedding},
        )
        assert response.status_code == 200, f"Failed: {response.text}"
        result = response.json()
        
        assert "bolted_probability" in result
        prob = result["bolted_probability"]
        print(f"Plant 29 Bolted Probability: {prob}")
        
        # Assert not bolted (prob < 0.5)
        assert prob < 0.5, f"Plant 29 should NOT be bolted, got prob {prob}"