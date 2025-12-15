"""
Embedding generation utilities using Jina Embeddings v4.
"""

import os
import httpx
from typing import List

# Configuration
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_API_URL = "https://api.jina.ai/v1/embeddings"
EMBEDDING_DIM = 1024  # Jina v4 supports up to 2048, using 1024 for efficiency


async def get_text_embedding(text: str) -> List[float]:
    """Generate text embedding using Jina embeddings v4."""
    print(f"   📊 Generating text embedding for: {text[:50]}...")
    
    if not JINA_API_KEY:
        print(f"   ⚠️ No JINA_API_KEY - using random embedding")
        import random
        return [random.random() for _ in range(EMBEDDING_DIM)]
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                JINA_API_URL,
                headers={
                    "Authorization": f"Bearer {JINA_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "jina-embeddings-v4",
                    "task": "retrieval.passage",
                    "dimensions": EMBEDDING_DIM,
                    "input": [{"text": text}]
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                embedding = data["data"][0]["embedding"]
                print(f"   ✅ Text embedding generated (dim: {len(embedding)})")
                return embedding
            else:
                print(f"   ❌ Jina embedding error: {response.status_code} - {response.text[:200]}")
                import random
                return [random.random() for _ in range(EMBEDDING_DIM)]
    except Exception as e:
        print(f"   ❌ Jina embedding exception: {e}")
        import random
        return [random.random() for _ in range(EMBEDDING_DIM)]


async def get_query_embedding(text: str) -> List[float]:
    """Generate query embedding using Jina embeddings v4 (optimized for queries)."""
    print(f"   🔍 Generating query embedding for: {text[:50]}...")
    
    if not JINA_API_KEY:
        print(f"   ⚠️ No JINA_API_KEY - using random embedding")
        import random
        return [random.random() for _ in range(EMBEDDING_DIM)]
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                JINA_API_URL,
                headers={
                    "Authorization": f"Bearer {JINA_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "jina-embeddings-v4",
                    "task": "retrieval.query",
                    "dimensions": EMBEDDING_DIM,
                    "input": [{"text": text}]
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                embedding = data["data"][0]["embedding"]
                print(f"   ✅ Query embedding generated (dim: {len(embedding)})")
                return embedding
            else:
                print(f"   ❌ Jina query error: {response.status_code} - {response.text[:200]}")
                import random
                return [random.random() for _ in range(EMBEDDING_DIM)]
    except Exception as e:
        print(f"   ❌ Jina query exception: {e}")
        import random
        return [random.random() for _ in range(EMBEDDING_DIM)]


async def get_image_embedding(image_url: str, description: str = "") -> List[float]:
    """Generate image embedding using Jina embeddings v4 (multimodal)."""
    print(f"   🖼️ Generating image embedding for: {image_url[:50]}...")
    
    if not JINA_API_KEY:
        print(f"   ⚠️ No JINA_API_KEY - using random embedding")
        import random
        return [random.random() for _ in range(EMBEDDING_DIM)]
    
    try:
        async with httpx.AsyncClient() as client:
            # Jina v4 supports image URLs directly
            input_data = {"image": image_url}
            if description:
                # Can also include text for better context
                input_data = {"image": image_url, "text": description}
            
            response = await client.post(
                JINA_API_URL,
                headers={
                    "Authorization": f"Bearer {JINA_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "jina-embeddings-v4",
                    "task": "retrieval.passage",
                    "dimensions": EMBEDDING_DIM,
                    "input": [input_data]
                },
                timeout=60.0  # Images may take longer
            )
            
            if response.status_code == 200:
                data = response.json()
                embedding = data["data"][0]["embedding"]
                print(f"   ✅ Image embedding generated (dim: {len(embedding)})")
                return embedding
            else:
                print(f"   ❌ Jina image error: {response.status_code} - {response.text[:200]}")
                # Fallback to text-based embedding
                return await get_text_embedding(f"image: {description} {image_url}")
    except Exception as e:
        print(f"   ❌ Jina image exception: {e}")
        return await get_text_embedding(f"image: {description}")
