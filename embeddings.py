import requests
from config import HF_TOKEN

API_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-base-en-v1.5/pipeline/feature-extraction"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": texts, "normalize": True}
    )
    
    if response.status_code != 200:
        raise Exception(f"HF API error: {response.status_code} - {response.text}")
    
    result = response.json()
    return result