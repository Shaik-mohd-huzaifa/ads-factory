"""
LLM integration utilities supporting OpenRouter and direct Gemini API.
"""

import os
import httpx
import google.generativeai as genai

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
USE_OPENROUTER = os.getenv("USE_OPENROUTER", "false").lower() == "true"

# OpenRouter configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_GEMINI_MODEL = "google/gemini-2.0-flash-001"

# Initialize Gemini (only if not using OpenRouter)
if GOOGLE_API_KEY and not USE_OPENROUTER:
    genai.configure(api_key=GOOGLE_API_KEY)

print(f"🔧 LLM Provider: {'OpenRouter' if USE_OPENROUTER else 'Direct Gemini API'}")


async def call_openrouter(prompt: str, system_prompt: str = None) -> str:
    """Call OpenRouter API for Gemini model."""
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


async def call_gemini_direct(prompt: str) -> str:
    """Call Gemini API directly."""
    model = genai.GenerativeModel("models/gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text


async def call_llm(prompt: str, system_prompt: str = None) -> str:
    """
    Call the configured LLM provider (OpenRouter or direct Gemini).
    
    Args:
        prompt: The user prompt to send
        system_prompt: Optional system prompt (only used with OpenRouter)
    
    Returns:
        The LLM response text
    """
    if USE_OPENROUTER and OPENROUTER_API_KEY:
        return await call_openrouter(prompt, system_prompt)
    elif GOOGLE_API_KEY:
        # For direct Gemini, combine system prompt with user prompt
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        return await call_gemini_direct(full_prompt)
    else:
        raise ValueError("No LLM API key configured")


def has_llm_configured() -> bool:
    """Check if any LLM provider is configured."""
    return (USE_OPENROUTER and bool(OPENROUTER_API_KEY)) or (not USE_OPENROUTER and bool(GOOGLE_API_KEY))


def get_llm_provider() -> str:
    """Get the current LLM provider name."""
    if USE_OPENROUTER and OPENROUTER_API_KEY:
        return "openrouter"
    elif GOOGLE_API_KEY:
        return "gemini"
    return "none"
