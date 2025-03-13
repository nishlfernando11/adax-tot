import ollama
from sentence_transformers import SentenceTransformer
import time
import logging
import numpy as np

'''
# Best Local Embedding Model Choices
Model	Dimensions	Speed (Queries/sec)	Best For
BAAI/bge-large-en	1536	🏎 Fast (~50-150/sec)	# Best RAG accuracy
intfloat/e5-large-v2	1024	🔥 Faster (~200-400/sec)	# Balanced accuracy & speed
all-MiniLM-L12-v2	384	⚡ Fastest (~500-1000/sec)	# Ultra-low latency

# Final Recommendation
Your Priority	Best Model
# Highest Quality (Best RAG, Accuracy)	BAAI/bge-large-en (1536D)
# Fastest with High Accuracy	intfloat/e5-large-v2 (1024D)
# Ultra-Fast but Slightly Lower Accuracy	all-MiniLM-L12-v2 (384D)
🚀 Recommendation:

If you need top RAG performance, use bge-large-en (1536D).
If you want fastest + still accurate, use e5-large-v2 (1024D).
If you want ultra-low latency, use MiniLM-L12-v2 (384D).
'''
class EmbeddingGenerator:
    def __init__(self, provider="ollama", model="mistral:latest", target_dim=1536):
        """
        Initialize the embedding provider.
        
        Args:
            provider: "ollama" (default) for local Mistral, "sentence-transformers" for local embedding models.
            model: The model to use for embeddings.
        """
        self.provider = provider
        self.model = model
        self.target_dim = target_dim

        if provider == "sentence-transformers":
            self.transformer_model = SentenceTransformer(model)

    def get_embedding(self, text: str) -> list:
        """
        Generate embedding for the given text using a local model.
        
        Args:
            text: The input text to generate an embedding for.

        Returns:
            A list of floats representing the embedding.
        """
        text = text.replace("\n", " ")  # Ensure clean text input
        start_time = time.time()

        if self.provider == "ollama":
            # Use Ollama's local embedding API (Mistral, LLaMA, etc.)
            embedding = ollama.embeddings(model=self.model, prompt=text)["embedding"]
        
        elif self.provider == "sentence-transformers":
            # Use a Sentence Transformer model (BGE, MiniLM, etc.)
            embedding = self.transformer_model.encode(text).tolist()
        
        else:
            raise ValueError("Unsupported embedding provider. Use 'ollama' or 'sentence-transformers'.")

        # # Fix dimensions (truncate or pad)
        # if len(embedding) > self.target_dim:
        #     embedding = embedding[:self.target_dim]  # Truncate
        # elif len(embedding) < self.target_dim:
        #     embedding = np.pad(embedding, (0, self.target_dim - len(embedding)), 'constant')

        elapsed_time = time.time() - start_time
        logging.info(f"Embedding generated in {elapsed_time:.3f} seconds")
        return embedding
