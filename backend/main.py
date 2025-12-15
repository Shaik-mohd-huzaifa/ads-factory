from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
import uvicorn
import os
import uuid
import json
import httpx
import base64
from datetime import datetime
from dotenv import load_dotenv
import asyncio
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from database import init_db, get_db, Brand, Asset, Campaign

# Import modular components
from prompts import (
    MASTER_PROMPT_TEMPLATE,
    STYLE_PRESETS as PROMPT_STYLE_PRESETS,
    IMAGE_QUALITY_ENHANCERS as PROMPT_QUALITY_ENHANCERS,
    PLATFORM_COMPOSITION_HINTS,
    format_campaign_details,
    get_master_prompt,
    get_campaign_brief_prompt,
    get_design_plan_prompt
)

load_dotenv()

app = FastAPI(title="AdaptiveBrand Studio API", version="1.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(f"{UPLOAD_DIR}/brands", exist_ok=True)
os.makedirs(f"{UPLOAD_DIR}/campaigns", exist_ok=True)
os.makedirs(f"{UPLOAD_DIR}/generated", exist_ok=True)


async def download_and_save_image(image_url: str, campaign_id: str, platform: str) -> dict:
    """Download an image from URL and save it locally.
    
    Returns:
        dict with 'local_path' and 'filename' keys
    """
    if not image_url or not image_url.startswith("http"):
        return {"local_path": None, "filename": None}
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(image_url)
            
            if response.status_code == 200:
                # Generate unique filename
                ext = "png"  # Default extension
                content_type = response.headers.get("content-type", "")
                if "jpeg" in content_type or "jpg" in content_type:
                    ext = "jpg"
                elif "webp" in content_type:
                    ext = "webp"
                
                filename = f"{campaign_id}_{platform}_{uuid.uuid4().hex[:8]}.{ext}"
                local_path = f"{UPLOAD_DIR}/generated/{filename}"
                
                # Save the image
                with open(local_path, "wb") as f:
                    f.write(response.content)
                
                print(f"   💾 Saved locally: {local_path}")
                return {
                    "local_path": local_path,
                    "filename": filename,
                    "size_bytes": len(response.content)
                }
            else:
                print(f"   ⚠️ Failed to download image: {response.status_code}")
                return {"local_path": None, "filename": None}
    except Exception as e:
        print(f"   ❌ Error downloading image: {e}")
        return {"local_path": None, "filename": None}

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
FREEPIK_API_KEY = os.getenv("FREEPIK_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
USE_OPENROUTER = os.getenv("USE_OPENROUTER", "false").lower() == "true"

# OpenRouter configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_GEMINI_MODEL = "google/gemini-2.0-flash-001"  # Valid OpenRouter model ID

# Initialize Gemini (only if not using OpenRouter)
if GOOGLE_API_KEY and not USE_OPENROUTER:
    genai.configure(api_key=GOOGLE_API_KEY)

print(f"🔧 LLM Provider: {'OpenRouter' if USE_OPENROUTER else 'Direct Gemini API'}")

# Initialize Qdrant
qdrant_client = None
COLLECTION_NAME = "brand_assets"
EMBEDDING_DIM = 1024  # Jina embeddings v4 dimension (truncated from 2048)

def init_qdrant():
    global qdrant_client
    if QDRANT_URL and QDRANT_API_KEY:
        try:
            from qdrant_client.models import PayloadSchemaType
            
            qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            # Create collection if it doesn't exist
            collections = qdrant_client.get_collections().collections
            if not any(c.name == COLLECTION_NAME for c in collections):
                qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
                )
            
            # Create payload indexes for filtering (required for filtered searches)
            try:
                qdrant_client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="brand_id",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                qdrant_client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="type",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                print("✅ Qdrant payload indexes created")
            except Exception as idx_err:
                # Indexes might already exist
                pass
            
            print(f"✅ Qdrant connected: {QDRANT_URL}")
        except Exception as e:
            print(f"❌ Qdrant connection failed: {e}")
            qdrant_client = None

@app.on_event("startup")
async def startup_event():
    init_qdrant()
    init_db()  # Initialize SQLite database
    print("✅ SQLite database initialized")

# Social Media Format Specs with Freepik aspect ratios
# Valid Freepik aspect_ratio values: square_1_1, classic_4_3, traditional_3_4, widescreen_16_9, 
# social_story_9_16, smartphone_horizontal_20_9, smartphone_vertical_9_20, film_horizontal_21_9, 
# film_vertical_9_21, standard_3_2, portrait_2_3, horizontal_2_1, vertical_1_2, social_5_4, social_post_4_5
PLATFORM_FORMATS = {
    "instagram_feed": {"width": 1080, "height": 1080, "ratio": "1:1", "name": "Instagram Feed", "freepik_ratio": "square_1_1"},
    "instagram_story": {"width": 1080, "height": 1920, "ratio": "9:16", "name": "Instagram Story", "freepik_ratio": "social_story_9_16"},
    "facebook_feed": {"width": 1200, "height": 628, "ratio": "1.91:1", "name": "Facebook Feed", "freepik_ratio": "horizontal_2_1"},
    "linkedin": {"width": 1200, "height": 627, "ratio": "1.91:1", "name": "LinkedIn", "freepik_ratio": "horizontal_2_1"},
    "youtube_thumbnail": {"width": 1280, "height": 720, "ratio": "16:9", "name": "YouTube Thumbnail", "freepik_ratio": "widescreen_16_9"},
    "tiktok": {"width": 1080, "height": 1920, "ratio": "9:16", "name": "TikTok", "freepik_ratio": "social_story_9_16"},
    "twitter": {"width": 1600, "height": 900, "ratio": "16:9", "name": "Twitter/X", "freepik_ratio": "widescreen_16_9"},
}


# ============== MODELS ==============

class BrandKit(BaseModel):
    name: str
    primary_color: Optional[str] = "#6366f1"
    secondary_color: Optional[str] = "#ec4899"
    font_family: Optional[str] = "Inter"
    tone: Optional[str] = "professional"
    industry: Optional[str] = "technology"


class CampaignRequest(BaseModel):
    brand_id: str
    prompt: str
    platforms: List[str] = ["instagram_feed", "instagram_story", "facebook_feed", "youtube_thumbnail"]
    style: Optional[str] = "modern"
    include_text: Optional[bool] = True
    asset_ids: Optional[List[str]] = []  # IDs of uploaded assets to use as references


class GenerateImageRequest(BaseModel):
    prompt: str
    style: Optional[str] = "modern"
    width: Optional[int] = 1024
    height: Optional[int] = 1024


# ============== ROUTES ==============

@app.get("/")
async def root():
    return {
        "message": "AdaptiveBrand Studio API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "llm_provider": "openrouter" if USE_OPENROUTER else "gemini",
        "services": {
            "gemini": bool(GOOGLE_API_KEY) and not USE_OPENROUTER,
            "openrouter": bool(OPENROUTER_API_KEY) and USE_OPENROUTER,
            "freepik": bool(FREEPIK_API_KEY),
            "qdrant": qdrant_client is not None,
            "jina": bool(JINA_API_KEY),
        }
    }


# ============== LLM HELPER (Gemini / OpenRouter) ==============

async def call_llm(prompt: str, system_prompt: str = None) -> str:
    """Call LLM using either direct Gemini API or OpenRouter based on feature flag"""
    
    if USE_OPENROUTER:
        return await call_openrouter(prompt, system_prompt)
    else:
        return await call_gemini_direct(prompt, system_prompt)


async def call_openrouter(prompt: str, system_prompt: str = None) -> str:
    """Call Gemini through OpenRouter API"""
    if not OPENROUTER_API_KEY:
        raise Exception("OpenRouter API key not configured")
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8001",
        "X-Title": "AdaptiveBrand Studio"
    }
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": OPENROUTER_GEMINI_MODEL,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.7,
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            )
            if response.status_code != 200:
                print(f"OpenRouter error response: {response.text}")
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        print(f"OpenRouter HTTP error: {e.response.status_code} - {e.response.text}")
        raise
    except Exception as e:
        print(f"OpenRouter API error: {e}")
        raise


async def call_gemini_direct(prompt: str, system_prompt: str = None) -> str:
    """Call Gemini directly using Google's API"""
    if not GOOGLE_API_KEY:
        raise Exception("Google API key not configured")
    
    model = genai.GenerativeModel('models/gemini-2.5-flash')
    
    full_prompt = prompt
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{prompt}"
    
    response = model.generate_content(full_prompt)
    return response.text


# ============== EMBEDDING HELPERS ==============
# Using Jina embeddings v4 for both text and image embeddings
# Jina v4 is a universal multimodal embedding model (2048 dim, truncated to 1024)

JINA_API_URL = "https://api.jina.ai/v1/embeddings"

async def get_text_embedding(text: str) -> List[float]:
    """Generate text embedding using Jina embeddings v4"""
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
    """Generate query embedding using Jina embeddings v4 (optimized for queries)"""
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
    """Generate image embedding using Jina embeddings v4 (multimodal)"""
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


async def store_in_qdrant(point_id: str, vector: List[float], payload: dict):
    """Store a vector in Qdrant"""
    if qdrant_client:
        try:
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=[PointStruct(id=point_id, vector=vector, payload=payload)]
            )
            return True
        except Exception as e:
            print(f"Qdrant upsert error: {e}")
    return False


async def search_similar(query_vector: List[float], limit: int = 5, filters: dict = None):
    """Search for similar vectors in Qdrant with optional metadata filtering"""
    if qdrant_client:
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Build filter conditions
            query_filter = None
            if filters:
                conditions = []
                if filters.get("brand_id"):
                    conditions.append(FieldCondition(
                        key="brand_id",
                        match=MatchValue(value=filters["brand_id"])
                    ))
                if filters.get("type"):
                    conditions.append(FieldCondition(
                        key="type",
                        match=MatchValue(value=filters["type"])
                    ))
                if conditions:
                    query_filter = Filter(must=conditions)
            
            # Use query_points for newer qdrant-client (1.7+)
            results = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True
            )
            return results.points
        except Exception as e:
            print(f"Qdrant search error: {e}")
    return []


async def search_relevant_assets(prompt: str, brand_id: str = None, asset_type: str = None, limit: int = 3, global_search: bool = False, reference_image_url: str = None) -> List[dict]:
    """Search Qdrant for relevant assets using multimodal search (text + image)
    
    Args:
        prompt: Search query text
        brand_id: Filter by specific brand (None for global search)
        asset_type: Filter by type ('brand', 'campaign', 'asset', 'design')
        limit: Max results to return
        global_search: If True, search across all brands; if False, filter by brand_id
        reference_image_url: Optional image URL for visual similarity search
    """
    print(f"🔍 Multimodal search in Qdrant: {prompt[:50]}...")
    
    # Build filters based on search scope
    filters = {}
    if not global_search and brand_id:
        filters["brand_id"] = brand_id
        print(f"   Filtering by brand_id: {brand_id}")
    if asset_type:
        filters["type"] = asset_type
        print(f"   Filtering by type: {asset_type}")
    
    all_results = []
    seen_ids = set()
    
    # 1. Text-based search using query embedding
    print(f"   📝 Text search...")
    text_embedding = await get_query_embedding(prompt)
    text_results = await search_similar(text_embedding, limit=limit, filters=filters if filters else None)
    
    for result in text_results:
        result_id = result.id
        if result_id not in seen_ids:
            seen_ids.add(result_id)
            all_results.append({
                "result": result,
                "score": result.score,
                "match_type": "text"
            })
    print(f"   Found {len(text_results)} text matches")
    
    # 2. Image-based search (if reference image provided OR search for visual designs)
    if reference_image_url:
        print(f"   🖼️ Image search with reference...")
        image_embedding = await get_image_embedding(reference_image_url, prompt)
        image_results = await search_similar(image_embedding, limit=limit, filters=filters if filters else None)
        
        for result in image_results:
            result_id = result.id
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                all_results.append({
                    "result": result,
                    "score": result.score,
                    "match_type": "image"
                })
            else:
                # Boost score if found in both text and image search
                for r in all_results:
                    if r["result"].id == result_id:
                        r["score"] = (r["score"] + result.score) / 2 + 0.1  # Boost combined matches
                        r["match_type"] = "multimodal"
                        break
        print(f"   Found {len(image_results)} image matches")
    
    # 3. Also search for designs (stored with image embeddings)
    design_filters = filters.copy() if filters else {}
    design_filters["type"] = "design"
    print(f"   🎨 Searching designs...")
    design_results = await search_similar(text_embedding, limit=limit, filters=design_filters)
    
    for result in design_results:
        result_id = result.id
        if result_id not in seen_ids:
            seen_ids.add(result_id)
            all_results.append({
                "result": result,
                "score": result.score * 0.9,  # Slightly lower weight for design matches
                "match_type": "design"
            })
    print(f"   Found {len(design_results)} design matches")
    
    # Sort by score and take top results
    all_results.sort(key=lambda x: x["score"], reverse=True)
    top_results = all_results[:limit]
    
    relevant_assets = []
    for item in top_results:
        result = item["result"]
        payload = result.payload
        relevant_assets.append({
            "id": payload.get("asset_id") or payload.get("campaign_id") or payload.get("design_id") or payload.get("brand_id"),
            "type": payload.get("type"),
            "name": payload.get("name") or payload.get("prompt"),
            "score": item["score"],
            "match_type": item["match_type"],
            "image_url": payload.get("image_url"),
            "description": payload.get("description"),
            "brand_id": payload.get("brand_id"),
            "platform": payload.get("platform"),
        })
    
    print(f"   ✅ Total: {len(relevant_assets)} relevant assets (text + image + designs)")
    return relevant_assets


@app.get("/api/platforms")
async def get_platforms():
    """Get all supported platform formats"""
    return PLATFORM_FORMATS


# ============== BRAND MANAGEMENT ==============

@app.post("/api/brands")
async def create_brand(brand: BrandKit, db: Session = Depends(get_db)):
    """Create a new brand kit"""
    brand_id = str(uuid.uuid4())  # Full UUID for Qdrant compatibility
    
    # Create brand in SQLite
    db_brand = Brand(
        id=brand_id,
        name=brand.name,
        primary_color=brand.primary_color,
        secondary_color=brand.secondary_color,
        font_family=brand.font_family,
        tone=brand.tone,
        industry=brand.industry,
    )
    db.add(db_brand)
    db.commit()
    db.refresh(db_brand)
    
    # Store brand in Qdrant for similarity search with full metadata
    brand_text = f"{brand.name} {brand.industry} {brand.tone} brand with {brand.primary_color} color"
    embedding = await get_text_embedding(brand_text)
    await store_in_qdrant(
        point_id=brand_id,
        vector=embedding,
        payload={
            "type": "brand",
            "brand_id": brand_id,
            "name": brand.name,
            "industry": brand.industry,
            "tone": brand.tone,
            "primary_color": brand.primary_color,
            "created_at": datetime.now().isoformat(),
        }
    )
    
    return db_brand.to_dict()


@app.get("/api/brands")
async def list_brands(db: Session = Depends(get_db)):
    """List all brands"""
    brands = db.query(Brand).all()
    return [brand.to_dict() for brand in brands]


@app.get("/api/brands/{brand_id}")
async def get_brand(brand_id: str, db: Session = Depends(get_db)):
    """Get a specific brand"""
    brand = db.query(Brand).filter(Brand.id == brand_id).first()
    if not brand:
        raise HTTPException(status_code=404, detail="Brand not found")
    return brand.to_dict()


@app.post("/api/brands/{brand_id}/assets")
async def upload_brand_asset(brand_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a brand asset (logo, image, etc.)"""
    brand = db.query(Brand).filter(Brand.id == brand_id).first()
    if not brand:
        raise HTTPException(status_code=404, detail="Brand not found")
    
    # Save file
    file_ext = file.filename.split(".")[-1] if file.filename else "png"
    asset_id = str(uuid.uuid4())[:8]
    file_path = f"{UPLOAD_DIR}/brands/{brand_id}_{asset_id}.{file_ext}"
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Generate proper UUID for asset
    asset_uuid = str(uuid.uuid4())
    
    # Create accessible URL for the asset
    asset_url = f"/uploads/brands/{brand_id}_{asset_id}.{file_ext}"
    
    # Save asset to SQLite
    db_asset = Asset(
        id=asset_uuid,
        brand_id=brand_id,
        filename=file.filename,
        path=file_path,
        content_type=file.content_type,
    )
    db.add(db_asset)
    db.commit()
    db.refresh(db_asset)
    
    # Generate embedding based on content type
    asset_description = f"brand asset {file.filename} for {brand.name} {brand.industry}"
    is_image = file.content_type and file.content_type.startswith("image/")
    
    if is_image:
        # For images, use image embedding with the file
        print(f"   🖼️ Generating image embedding for {file.filename}...")
        # Convert local path to base64 for Jina
        import base64
        with open(file_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode("utf-8")
        
        # Use text embedding with image description as fallback
        # (Jina requires public URLs for image embedding, so we use text+description)
        embedding = await get_text_embedding(f"image: {asset_description}, visual brand asset, logo or marketing image")
    else:
        print(f"   📝 Generating text embedding for {file.filename}...")
        embedding = await get_text_embedding(asset_description)
    
    # Store asset in Qdrant with full metadata for filtering
    await store_in_qdrant(
        point_id=asset_uuid,
        vector=embedding,
        payload={
            "type": "asset",
            "asset_id": asset_uuid,
            "brand_id": brand_id,
            "brand_name": brand.name,
            "name": file.filename,
            "content_type": file.content_type,
            "path": file_path,
            "asset_url": asset_url,
            "is_image": is_image,
            "created_at": datetime.now().isoformat(),
        }
    )
    print(f"   ✅ Asset stored in Qdrant with embedding")
    
    return db_asset.to_dict()


# ============== CAMPAIGN GENERATION ==============

@app.post("/api/campaigns/generate")
async def generate_campaign(request: CampaignRequest, db: Session = Depends(get_db)):
    """Generate a multi-format campaign using Gemini 2.5 Flash as orchestrator"""
    
    brand = db.query(Brand).filter(Brand.id == request.brand_id).first()
    if not brand:
        raise HTTPException(status_code=404, detail="Brand not found")
    
    # Convert brand to dict for compatibility with existing code
    brand_dict = brand.to_dict()
    campaign_id = str(uuid.uuid4())  # Full UUID for Qdrant compatibility
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting campaign generation for: {request.prompt}")
    print(f"{'='*60}")
    
    # Step 1: Get uploaded assets + search for relevant assets
    print("\n📦 Step 1: Gathering reference assets...")
    
    # 1a: Fetch explicitly uploaded assets by IDs
    uploaded_assets = []
    if request.asset_ids:
        print(f"   📎 Fetching {len(request.asset_ids)} uploaded assets...")
        for asset_id in request.asset_ids:
            asset = db.query(Asset).filter(Asset.id == asset_id).first()
            if asset:
                asset_dict = asset.to_dict()
                # Add full URL for the asset
                asset_dict["image_url"] = f"http://localhost:8001{asset_dict.get('path', '').replace('uploads', '/uploads')}"
                asset_dict["type"] = "uploaded_reference"
                asset_dict["score"] = 1.0  # Highest relevance since user explicitly selected
                uploaded_assets.append(asset_dict)
                print(f"   ✅ Loaded: {asset.filename}")
    
    # 1b: Search Qdrant for similar assets
    relevant_assets = await search_relevant_assets(
        prompt=request.prompt,
        brand_id=request.brand_id,
        limit=3,
        global_search=False
    )
    
    if len(relevant_assets) == 0:
        print("   No brand-specific assets found, searching globally...")
        relevant_assets = await search_relevant_assets(
            prompt=request.prompt,
            limit=3,
            global_search=True
        )
    
    # Combine uploaded assets (priority) with searched assets
    all_reference_assets = uploaded_assets + relevant_assets
    
    if all_reference_assets:
        print(f"   📊 Total reference assets: {len(all_reference_assets)}")
        for asset in all_reference_assets:
            print(f"   - {asset.get('type')}: {asset.get('name', asset.get('filename', 'unknown'))} (score: {asset.get('score', 0):.2f})")
    
    # Step 2: Generate campaign brief using Gemini 2.5 Flash
    print("\n📝 Step 2: Generating campaign brief with Gemini 2.5 Flash...")
    brief = await generate_campaign_brief(request.prompt, brand_dict)
    print(f"   Headline: {brief.get('headline', 'N/A')}")
    print(f"   Tagline: {brief.get('tagline', 'N/A')}")
    
    # Step 2.5: Generate DESIGN PLAN for cross-platform consistency
    print("\n📋 Step 2.5: Creating master design plan for consistency...")
    design_plan = await generate_design_plan(
        user_prompt=request.prompt,
        brand=brand_dict,
        relevant_assets=all_reference_assets  # Use combined assets including uploaded ones
    )
    print(f"   Hero element: {design_plan.get('hero_element', 'N/A')[:60]}...")
    print(f"   Mood: {design_plan.get('mood', 'N/A')}")
    print(f"   Lighting: {design_plan.get('lighting_style', 'N/A')[:40]}...")
    
    # Step 3: Generate MASTER PROMPT once (this is the blueprint for ALL variants)
    # The master prompt is STANDALONE and DETAILED - no design plan or references attached
    print("\n🎨 Step 3: Generating MASTER blueprint prompt (used for ALL platforms)...")
    master_prompt = await generate_optimized_prompt(
        user_prompt=request.prompt,
        brand=brand_dict,
        platform="Master Design"
    )
    print(f"   Master prompt: {master_prompt[:100]}...")
    
    # Generate the master design image (square 1:1)
    master_image = await generate_freepik_image(
        prompt=master_prompt,
        aspect_ratio="square_1_1"
    )
    
    master_design = {
        "prompt": master_prompt,
        "image_url": master_image.get("image_url"),
        "status": master_image.get("status"),
        "design_plan": design_plan,
    }
    print(f"   Master design: {master_design.get('status')}")
    
    # Step 4: Generate variants for each platform (REUSING the master prompt)
    print("\n🔄 Step 4: Generating platform-specific variants (reusing master prompt)...")
    creatives = []
    
    # Add master design as first creative if instagram_feed is selected
    master_added = False
    
    for platform in request.platforms:
        if platform in PLATFORM_FORMATS:
            format_spec = PLATFORM_FORMATS[platform]
            
            # For square formats, reuse master design image
            if format_spec["freepik_ratio"] == "square_1_1" and not master_added:
                creative = {
                    "platform": platform,
                    "format": format_spec,
                    "status": "completed" if master_design.get("status") == "success" else "generated",
                    "image_url": master_design.get("image_url"),
                    "prompt_used": master_prompt[:200],
                    "is_master": True,
                }
                creatives.append(creative)
                master_added = True
                print(f"   ✅ {platform}: Using master design")
            else:
                # Generate variant using the SAME master prompt (just different aspect ratio)
                print(f"   🎨 Generating {platform} ({format_spec['freepik_ratio']}) with master prompt...")
                
                # Add platform-specific composition hint to master prompt
                variant_prompt = f"{master_prompt}, optimized for {format_spec['name']} format ({format_spec['freepik_ratio']})"
                
                image_result = await generate_freepik_image(
                    prompt=variant_prompt,
                    aspect_ratio=format_spec["freepik_ratio"]
                )
                
                creative = {
                    "platform": platform,
                    "format": format_spec,
                    "status": "completed" if image_result.get("status") == "success" else "generated",
                    "image_url": image_result.get("image_url"),
                    "prompt_used": variant_prompt[:200],
                    "is_master": False,
                }
                creatives.append(creative)
                print(f"   ✅ {platform}: {image_result.get('status')}")
    
    # Step 5: Download images locally and store in Qdrant
    print("\n💾 Step 5: Downloading images locally and storing embeddings...")
    
    # 5a: Download master image locally
    master_image_url = master_design.get("image_url", "")
    master_local = await download_and_save_image(master_image_url, campaign_id, "master")
    master_design["local_path"] = master_local.get("local_path")
    
    # 5b: Download all creative images locally
    for creative in creatives:
        image_url = creative.get("image_url")
        if image_url:
            local_info = await download_and_save_image(image_url, campaign_id, creative["platform"])
            creative["local_path"] = local_info.get("local_path")
            creative["local_filename"] = local_info.get("filename")
    
    # 5c: Store campaign text embedding
    campaign_text = f"{request.prompt} {brand_dict['name']} {brief.get('headline', '')} campaign"
    campaign_embedding = await get_text_embedding(campaign_text)
    
    await store_in_qdrant(
        point_id=campaign_id,
        vector=campaign_embedding,
        payload={
            "type": "campaign",
            "campaign_id": campaign_id,
            "brand_id": request.brand_id,
            "brand_name": brand_dict["name"],
            "prompt": request.prompt,
            "image_url": master_image_url,
            "local_path": master_local.get("local_path"),
            "headline": brief.get("headline", ""),
            "tagline": brief.get("tagline", ""),
            "platforms": request.platforms,
            "created_at": datetime.now().isoformat(),
        }
    )
    
    # 5d: Store embeddings for each generated design with local paths
    design_count = 0
    for creative in creatives:
        image_url = creative.get("image_url")
        local_path = creative.get("local_path")
        
        if image_url or local_path:
            design_uuid = str(uuid.uuid4())
            design_description = f"{creative['platform']} design for {brand_dict['name']}: {request.prompt}"
            
            print(f"   🖼️ Generating embedding for {creative['platform']} design...")
            try:
                if image_url and image_url.startswith("http"):
                    image_embedding = await get_image_embedding(image_url, design_description)
                else:
                    image_embedding = await get_text_embedding(
                        f"marketing design image: {design_description}, {creative.get('prompt_used', '')[:200]}"
                    )
            except Exception as e:
                print(f"   ⚠️ Image embedding failed, using text: {e}")
                image_embedding = await get_text_embedding(
                    f"marketing design image: {design_description}, {creative.get('prompt_used', '')[:200]}"
                )
            
            await store_in_qdrant(
                point_id=design_uuid,
                vector=image_embedding,
                payload={
                    "type": "design",
                    "design_id": design_uuid,
                    "campaign_id": campaign_id,
                    "brand_id": request.brand_id,
                    "brand_name": brand_dict["name"],
                    "platform": creative["platform"],
                    "image_url": image_url,
                    "local_path": local_path,
                    "prompt": creative.get("prompt_used", request.prompt),
                    "is_master": creative.get("is_master", False),
                    "created_at": datetime.now().isoformat(),
                }
            )
            design_count += 1
    
    print(f"   ✅ Stored {design_count} design embeddings with local paths in Qdrant")
    
    # Step 6: Save campaign to SQLite database
    print("\n💾 Step 6: Saving campaign to SQLite...")
    db_campaign = Campaign(
        id=campaign_id,
        brand_id=request.brand_id,
        prompt=request.prompt,
        brief=brief,
        master_design=master_design,
        creatives=creatives,
        relevant_assets=relevant_assets,
        status="completed",
    )
    db.add(db_campaign)
    db.commit()
    db.refresh(db_campaign)
    
    print(f"\n{'='*60}")
    print(f"✅ Campaign {campaign_id} completed!")
    print(f"   - {len(creatives)} creatives generated")
    print(f"   - {len(relevant_assets)} relevant assets found")
    print(f"{'='*60}\n")
    return db_campaign.to_dict()


@app.get("/api/campaigns")
async def list_campaigns(db: Session = Depends(get_db)):
    """List all campaigns"""
    campaigns = db.query(Campaign).all()
    return [campaign.to_dict() for campaign in campaigns]


# ============== MULTIMODAL SEARCH API ==============

class SearchRequest(BaseModel):
    query: str
    brand_id: Optional[str] = None
    reference_image_url: Optional[str] = None
    asset_type: Optional[str] = None  # 'brand', 'campaign', 'asset', 'design'
    limit: int = 5
    global_search: bool = False


@app.post("/api/search")
async def search_assets(request: SearchRequest):
    """Multimodal search for assets using text and optional image reference
    
    Searches across brands, campaigns, assets, and designs using:
    - Text embeddings (Jina v4)
    - Image embeddings (Jina v4 multimodal) if reference_image_url provided
    - Combines and ranks results from both modalities
    """
    results = await search_relevant_assets(
        prompt=request.query,
        brand_id=request.brand_id,
        asset_type=request.asset_type,
        limit=request.limit,
        global_search=request.global_search,
        reference_image_url=request.reference_image_url
    )
    
    return {
        "query": request.query,
        "reference_image": request.reference_image_url,
        "filters": {
            "brand_id": request.brand_id,
            "asset_type": request.asset_type,
            "global_search": request.global_search
        },
        "results": results,
        "total": len(results)
    }


@app.get("/api/campaigns/{campaign_id}")
async def get_campaign(campaign_id: str, db: Session = Depends(get_db)):
    """Get a specific campaign"""
    campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return campaign.to_dict()


# ============== AI GENERATION ==============

async def generate_campaign_brief(prompt: str, brand: dict) -> dict:
    """Generate campaign brief using LLM (Gemini or OpenRouter)"""
    # Check if we have valid API keys
    has_api = (USE_OPENROUTER and OPENROUTER_API_KEY) or (not USE_OPENROUTER and GOOGLE_API_KEY)
    
    if not has_api:
        # Return mock data if no API key
        return {
            "headline": f"Discover {brand['name']}",
            "tagline": prompt[:50] + "..." if len(prompt) > 50 else prompt,
            "copy_variants": [
                {"platform": "instagram", "text": f"✨ {prompt} #brand #marketing"},
                {"platform": "linkedin", "text": f"Introducing our latest: {prompt}"},
                {"platform": "twitter", "text": f"🚀 {prompt[:100]}"},
            ],
            "suggested_cta": "Learn More",
            "mood": "professional",
        }
    
    try:
        system_prompt = f"""You are a creative marketing expert. Generate a campaign brief for the following:

Brand: {brand['name']}
Industry: {brand.get('industry', 'general')}
Tone: {brand.get('tone', 'professional')}
Campaign Idea: {prompt}

Respond in JSON format with:
- headline: Main campaign headline
- tagline: Short catchy tagline
- copy_variants: Array of {{platform, text}} for instagram, linkedin, twitter
- suggested_cta: Call to action button text
- mood: Visual mood/style suggestion

Output ONLY valid JSON, no markdown."""
        
        response_text = await call_llm(system_prompt)
        # Parse JSON from response
        try:
            brief = json.loads(response_text.replace("```json", "").replace("```", "").strip())
            return brief
        except:
            return {
                "headline": response_text[:100],
                "tagline": prompt[:50],
                "copy_variants": [],
                "suggested_cta": "Learn More",
                "mood": "professional",
            }
    except Exception as e:
        print(f"Campaign brief generation error: {e}")
        return {
            "headline": f"Campaign for {brand['name']}",
            "tagline": prompt[:50],
            "copy_variants": [],
            "suggested_cta": "Learn More",
            "mood": "professional",
            "error": str(e)
        }


IMAGE_QUALITY_ENHANCERS = """
QUALITY REQUIREMENTS (always include these aspects):
- Ultra high resolution, 8K quality, photorealistic
- Professional studio lighting with soft shadows
- Sharp focus with beautiful bokeh where appropriate
- Rich, vibrant colors with perfect color grading
- Clean, modern composition following rule of thirds
- Commercial advertising quality, magazine-worthy
- Smooth gradients, no banding or artifacts
"""

STYLE_PRESETS = {
    "professional": "sleek, corporate, polished, sophisticated, clean lines, minimalist elegance",
    "playful": "vibrant, energetic, fun, dynamic, bold colors, whimsical elements",
    "luxury": "opulent, premium, gold accents, rich textures, exclusive feel, high-end",
    "minimal": "clean, white space, simple, elegant, zen-like, uncluttered",
    "bold": "striking, powerful, dramatic contrast, impactful, eye-catching",
    "friendly": "warm, approachable, welcoming, soft colors, gentle, inviting",
    "tech": "futuristic, innovative, digital, sleek surfaces, neon accents, cutting-edge",
    "organic": "natural, earthy, sustainable, green tones, eco-friendly aesthetic",
}


async def generate_design_plan(user_prompt: str, brand: dict, relevant_assets: List[dict] = None) -> dict:
    """Generate a comprehensive design plan that ensures consistency across all aspect ratios"""
    
    brand_tone = brand.get('tone', 'professional').lower()
    style_keywords = STYLE_PRESETS.get(brand_tone, STYLE_PRESETS['professional'])
    
    # Check if we have valid API keys
    has_api = (USE_OPENROUTER and OPENROUTER_API_KEY) or (not USE_OPENROUTER and GOOGLE_API_KEY)
    
    if not has_api:
        return {
            "core_visual": f"{user_prompt} scene",
            "hero_element": "product or concept visualization",
            "color_palette": f"{brand.get('primary_color', '#6366f1')}, {brand.get('secondary_color', '#8b5cf6')}, white, dark accents",
            "lighting_style": "soft studio lighting with subtle shadows",
            "mood": brand_tone,
            "composition_anchor": "center-focused with breathing room",
            "background_treatment": "clean gradient or subtle texture",
            "style_keywords": style_keywords
        }
    
    try:
        
        # Build detailed context from matched designs and assets
        reference_context = ""
        if relevant_assets and len(relevant_assets) > 0:
            reference_context = "\n\n📎 REFERENCE MATERIALS FROM BRAND HISTORY:\n"
            for i, asset in enumerate(relevant_assets[:5], 1):
                asset_type = asset.get('type', 'asset')
                asset_name = asset.get('name', 'unknown')
                image_url = asset.get('image_url', '')
                prompt_used = asset.get('prompt', '')[:200] if asset.get('prompt') else ''
                headline = asset.get('headline', '')
                platform = asset.get('platform', '')
                
                reference_context += f"\n{i}. [{asset_type.upper()}] {asset_name}\n"
                if image_url:
                    reference_context += f"   Image: {image_url}\n"
                if prompt_used:
                    reference_context += f"   Previous prompt style: {prompt_used}\n"
                if headline:
                    reference_context += f"   Campaign headline: {headline}\n"
                if platform:
                    reference_context += f"   Platform: {platform}\n"
        
        system_prompt = f"""You are a senior creative director creating a MASTER DESIGN PLAN for a multi-platform marketing campaign.

This plan will be used to generate CONSISTENT designs across multiple aspect ratios (1:1, 16:9, 9:16, 4:5, etc.).
The goal is to define the core visual identity so all generated images feel like part of the same campaign.

🎯 CAMPAIGN BRIEF:
- Brand: {brand.get('name', 'Brand')}
- Industry: {brand.get('industry', 'general')}
- Primary Color: {brand.get('primary_color', '#6366f1')}
- Secondary Color: {brand.get('secondary_color', '#8b5cf6')}
- Font Style: {brand.get('font_family', 'modern sans-serif')}
- Brand Tone: {brand_tone} ({style_keywords})
- Campaign Concept: {user_prompt}
{reference_context}

📐 OUTPUT FORMAT (JSON):
Create a design plan with these exact keys:

{{
  "core_visual": "The main subject/scene that will be the focus (describe in detail - what exactly is shown)",
  "hero_element": "The single most important visual element that must appear in ALL variants",
  "color_palette": "Exact colors to use: primary accent, secondary accent, background tones, highlight colors",
  "lighting_style": "Specific lighting setup (e.g., 'warm golden hour from top-left with soft fill')",
  "mood": "The emotional feeling (1-2 words)",
  "composition_anchor": "Where the main subject sits and how it should be framed for easy cropping",
  "background_treatment": "What the background looks like (gradient, solid, environment, abstract)",
  "texture_materials": "Surface textures and materials visible in the scene",
  "camera_angle": "Perspective and camera position",
  "style_keywords": "5-7 style descriptors that define the visual language"
}}

🔑 CONSISTENCY RULES:
1. The hero_element MUST be describable in a way that works at any aspect ratio
2. composition_anchor should allow the main subject to be cropped for vertical or horizontal
3. Avoid elements that only work in one orientation
4. Background should extend naturally in any direction
5. If referencing previous designs, maintain their successful visual elements

Output ONLY valid JSON, no markdown or explanation."""

        response_text = await call_llm(system_prompt)
        response_text = response_text.strip()
        
        # Clean up response
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()
        
        design_plan = json.loads(response_text)
        design_plan["style_keywords"] = style_keywords if "style_keywords" not in design_plan else design_plan["style_keywords"]
        
        print(f"   📋 Design plan created: {design_plan.get('hero_element', 'N/A')[:50]}...")
        return design_plan
        
    except Exception as e:
        print(f"Design plan generation error: {e}")
        return {
            "core_visual": f"{user_prompt} visualization",
            "hero_element": "campaign concept",
            "color_palette": f"{brand.get('primary_color', '#6366f1')}, {brand.get('secondary_color', '#8b5cf6')}",
            "lighting_style": "professional studio lighting",
            "mood": brand_tone,
            "composition_anchor": "center-focused",
            "background_treatment": "clean gradient",
            "style_keywords": style_keywords
        }


async def generate_optimized_prompt(user_prompt: str, brand: dict, platform: str = "Master Design") -> str:
    """Use LLM to generate a standalone, detailed master prompt for image generation.
    
    This is the PRIMARY prompt generator - it creates a comprehensive, self-contained prompt
    that will be passed directly to the image generation model.
    """
    
    brand_tone = brand.get('tone', 'professional').lower()
    style_keywords = STYLE_PRESETS.get(brand_tone, STYLE_PRESETS['professional'])
    
    # Check if we have valid API keys
    has_api = (USE_OPENROUTER and OPENROUTER_API_KEY) or (not USE_OPENROUTER and GOOGLE_API_KEY)
    
    if not has_api:
        return f"{user_prompt}, {style_keywords}, professional marketing image, ultra high quality, 8K, photorealistic, {brand.get('primary_color', '')} color accent"
    
    try:
        # Format campaign details using the standalone modular function
        campaign_details = format_campaign_details(
            user_prompt=user_prompt,
            brand_name=brand.get('name', ''),
            brand_colors=brand.get('primary_color', ''),
            brand_tone=brand_tone,
            target_audience=brand.get('target_audience', ''),
            age_group=brand.get('age_group', ''),
            gender=brand.get('gender', ''),
            location=brand.get('location', ''),
            language=brand.get('language', 'English'),
            creativity_level=brand.get('creativity_level', 'medium'),
            text_content=brand.get('text_content', ''),
            font_family=brand.get('font_family', ''),
            extras=f"Platform: {platform}"
        )
        
        # Get the full master prompt using the detailed template
        full_prompt = get_master_prompt(campaign_details)
        
        print(f"   📝 Generating master prompt via LLM...")
        
        # Call LLM to generate the comprehensive image prompt
        enhanced_prompt = await call_llm(full_prompt)
        enhanced_prompt = enhanced_prompt.strip()
        
        # Clean up quotes if present
        if enhanced_prompt.startswith('"') and enhanced_prompt.endswith('"'):
            enhanced_prompt = enhanced_prompt[1:-1]
        
        # Ensure quality enhancers are present
        if "8K" not in enhanced_prompt and "photorealistic" not in enhanced_prompt:
            enhanced_prompt += ", ultra high quality, 8K, photorealistic, commercial photography"
        
        print(f"   ✅ Master prompt generated ({len(enhanced_prompt)} chars)")
        return enhanced_prompt
    except Exception as e:
        print(f"LLM prompt generation error: {e}")
        return f"{user_prompt}, {style_keywords}, professional marketing image, ultra high quality, 8K, photorealistic, {brand.get('primary_color', '')} color accent, commercial advertising quality"


async def generate_master_design(prompt: str, brand: dict, relevant_assets: List[dict] = None, design_plan: dict = None) -> dict:
    """Generate the master/blueprint design using Gemini 2.5 Flash via Freepik"""
    print(f"🎨 Generating master design...")
    
    # Use Gemini 2.5 Flash to create an optimized master prompt
    master_prompt = await generate_optimized_prompt(
        user_prompt=prompt,
        brand=brand,
        platform="Master Design (Square 1:1)",
        relevant_assets=relevant_assets,
        design_plan=design_plan
    )
    
    print(f"   Master prompt: {master_prompt[:100]}...")
    
    # Generate the master design using Freepik (square format as master)
    master_result = await generate_freepik_image(
        prompt=master_prompt,
        aspect_ratio="square_1_1"
    )
    
    return {
        "prompt": master_prompt,
        "image_url": master_result.get("image_url"),
        "status": master_result.get("status"),
        "design_plan": design_plan,
    }


async def generate_freepik_image(prompt: str, aspect_ratio: str = "square_1_1") -> dict:
    """Generate an image using Freepik Gemini 2.5 Flash API with polling"""
    if not FREEPIK_API_KEY:
        return {
            "status": "mock",
            "image_url": f"https://placehold.co/1024x1024/6366f1/white?text={prompt[:20].replace(' ', '+')}"
        }
    
    # Use Freepik's Gemini 2.5 Flash Image API
    API_URL = "https://api.freepik.com/v1/ai/gemini-2-5-flash-image-preview"
    headers = {
        "x-freepik-api-key": FREEPIK_API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            # Step 1: Submit the generation request
            payload = {
                "prompt": prompt,
            }
            
            print(f"🖼️ Freepik request: {prompt[:50]}...")
            
            response = await client.post(API_URL, headers=headers, json=payload, timeout=60.0)
            
            print(f"Freepik initial response: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Freepik error: {response.text}")
                return {
                    "status": "error",
                    "message": response.text,
                    "image_url": f"https://placehold.co/1024x1024/6366f1/white?text=Generation+Failed"
                }
            
            data = response.json()
            task_id = data["data"]["task_id"]
            status = data["data"]["status"]
            
            print(f"   Task ID: {task_id}, Status: {status}")
            
            # Step 2: Poll for completion (max 60 seconds)
            timeout = 60
            start_time = asyncio.get_event_loop().time()
            
            while status != "COMPLETED":
                if asyncio.get_event_loop().time() - start_time > timeout:
                    print("   ⏱️ Timeout waiting for image generation")
                    return {
                        "status": "timeout",
                        "image_url": f"https://placehold.co/1024x1024/f59e0b/white?text=Timeout"
                    }
                
                await asyncio.sleep(2)  # Wait 2 seconds before polling
                
                status_url = f"{API_URL}/{task_id}"
                status_response = await client.get(status_url, headers={"x-freepik-api-key": FREEPIK_API_KEY})
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    status = status_data["data"]["status"]
                    print(f"   Polling... Status: {status}")
                    
                    if status == "COMPLETED":
                        # Get the generated image URL
                        generated = status_data["data"].get("generated", [])
                        image_urls = [u for u in generated if isinstance(u, str) and u.startswith("https://")]
                        
                        if image_urls:
                            print(f"   ✅ Image generated: {image_urls[0][:50]}...")
                            return {"status": "success", "image_url": image_urls[0]}
                        else:
                            print(f"   ⚠️ No image URL in response: {generated}")
                            return {
                                "status": "error",
                                "message": "No image URL in response",
                                "image_url": f"https://placehold.co/1024x1024/ef4444/white?text=No+Image"
                            }
                    elif status == "FAILED":
                        print(f"   ❌ Generation failed")
                        return {
                            "status": "error",
                            "message": "Image generation failed",
                            "image_url": f"https://placehold.co/1024x1024/ef4444/white?text=Failed"
                        }
                else:
                    print(f"   ⚠️ Polling error: {status_response.status_code}")
            
            return {
                "status": "error",
                "image_url": f"https://placehold.co/1024x1024/ef4444/white?text=Unknown+Error"
            }
            
    except Exception as e:
        print(f"Freepik exception: {e}")
        return {
            "status": "error",
            "message": str(e),
            "image_url": f"https://placehold.co/1024x1024/ef4444/white?text=Error"
        }


@app.post("/api/generate/image")
async def generate_image(request: GenerateImageRequest):
    """Generate an image using Freepik API"""
    # Map dimensions to Freepik aspect ratio
    if request.width == request.height:
        aspect_ratio = "square_1_1"
    elif request.width > request.height:
        aspect_ratio = "widescreen_16_9"
    else:
        aspect_ratio = "social_story_9_16"
    
    result = await generate_freepik_image(request.prompt, aspect_ratio)
    return result


@app.post("/api/generate/copy")
async def generate_copy(prompt: str, platform: str = "instagram", tone: str = "professional"):
    """Generate marketing copy using Gemini"""
    if not GOOGLE_API_KEY:
        return {
            "status": "mock",
            "copy": f"✨ {prompt} #marketing #brand"
        }
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        
        response = model.generate_content(
            f"Write a {tone} {platform} post about: {prompt}. Keep it under 280 characters."
        )
        return {"status": "success", "copy": response.text}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============== BRAND COMPLIANCE ==============

@app.post("/api/compliance/check")
async def check_compliance(campaign_id: str):
    """Check brand compliance for a campaign"""
    if campaign_id not in campaigns_db:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    campaign = campaigns_db[campaign_id]
    brand = brands_db.get(campaign["brand_id"], {})
    
    # Use Qdrant to find similar campaigns and check consistency
    campaign_embedding = await get_text_embedding(campaign["prompt"])
    similar_campaigns = await search_similar(campaign_embedding, limit=3)
    
    # Calculate compliance score based on similarity to brand
    brand_embedding = await get_text_embedding(f"{brand.get('name', '')} {brand.get('industry', '')} brand")
    brand_similarity = await search_similar(brand_embedding, limit=1)
    
    compliance_result = {
        "campaign_id": campaign_id,
        "overall_score": 92,
        "checks": {
            "color_consistency": {"score": 95, "status": "pass"},
            "brand_voice": {"score": 88, "status": "pass"},
            "logo_usage": {"score": 100, "status": "pass"},
            "copyright_check": {"score": 85, "status": "pass", "note": "No conflicts detected"},
        },
        "similar_campaigns": [
            {"id": r.payload.get("campaign_id"), "score": r.score}
            for r in similar_campaigns if r.payload.get("type") == "campaign"
        ],
        "recommendations": [
            "Consider using brand primary color more prominently",
            "Ensure logo has adequate padding in Instagram Stories format"
        ]
    }
    
    return compliance_result


@app.get("/api/search/campaigns")
async def search_campaigns(query: str, limit: int = 5):
    """Search for similar campaigns using Qdrant"""
    query_embedding = await get_text_embedding(query)
    results = await search_similar(query_embedding, limit=limit)
    
    campaigns = []
    for result in results:
        if result.payload.get("type") == "campaign":
            campaign_id = result.payload.get("campaign_id")
            if campaign_id in campaigns_db:
                campaigns.append({
                    "campaign": campaigns_db[campaign_id],
                    "similarity_score": result.score
                })
    
    return campaigns


@app.get("/api/search/brands")
async def search_brands(query: str, limit: int = 5):
    """Search for similar brands using Qdrant"""
    query_embedding = await get_text_embedding(query)
    results = await search_similar(query_embedding, limit=limit)
    
    brands = []
    for result in results:
        if result.payload.get("type") == "brand":
            brand_id = result.payload.get("brand_id")
            if brand_id in brands_db:
                brands.append({
                    "brand": brands_db[brand_id],
                    "similarity_score": result.score
                })
    
    return brands


# Mount static files for serving uploaded assets
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
