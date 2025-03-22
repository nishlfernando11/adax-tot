
'''

### 5. **Inference Endpoint Tuning**

Implement FastAPI or another lightweight REST API to optimize performance around Ollama’s inference endpoints:

python
'''
import requests
from fastapi import FastAPI

app = FastAPI()

@app.post("/prompt")
async def prompt(query: str):
    response = requests.post(
        'http://127.0.0.1:11434/api/generate',
        json={"model": "mistral:7b-instruct-q4_K_M", "prompt": query}
    ).json()
    return response["response"]
