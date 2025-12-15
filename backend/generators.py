"""
Campaign generation utilities - brief, design plan, and master prompt generators.
"""

import json
from typing import List, Optional

from llm import call_llm, has_llm_configured
from prompts import (
    get_master_prompt,
    get_campaign_brief_prompt,
    get_design_plan_prompt,
    format_campaign_details,
    STYLE_PRESETS,
    IMAGE_QUALITY_ENHANCERS,
    PLATFORM_COMPOSITION_HINTS
)


async def generate_campaign_brief(prompt: str, brand: dict) -> dict:
    """Generate a campaign brief using LLM."""
    if not has_llm_configured():
        return {
            "headline": "Campaign Headline",
            "tagline": "Your tagline here",
            "suggested_cta": "Learn More",
            "mood": "professional",
            "key_message": prompt[:100]
        }
    
    try:
        system_prompt = get_campaign_brief_prompt(
            brand_name=brand.get("name", "Brand"),
            industry=brand.get("industry", "general"),
            colors=brand.get("primary_color", "#6366f1"),
            tone=brand.get("tone", "professional"),
            user_prompt=prompt
        )
        
        response_text = await call_llm(system_prompt)
        
        # Clean up response and parse JSON
        response_text = response_text.strip()
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()
        
        brief = json.loads(response_text)
        return brief
    except Exception as e:
        print(f"Campaign brief generation error: {e}")
        return {
            "headline": "Campaign Headline",
            "tagline": prompt[:50],
            "suggested_cta": "Learn More",
            "mood": brand.get("tone", "professional"),
            "key_message": prompt[:100]
        }


async def generate_design_plan(
    user_prompt: str,
    brand: dict,
    relevant_assets: List[dict] = None
) -> dict:
    """Generate a master design plan for multi-platform consistency."""
    if not has_llm_configured():
        tone = brand.get("tone", "professional").lower()
        style = STYLE_PRESETS.get(tone, STYLE_PRESETS["professional"])
        return {
            "core_visual": f"Professional {brand.get('industry', 'business')} scene",
            "hero_element": "Central product/service showcase",
            "color_palette": f"{brand.get('primary_color', '#6366f1')}, {style['colors']}",
            "lighting_style": style["lighting"],
            "mood": style["mood"],
            "composition_anchor": "Center-weighted with breathing room",
            "background_treatment": "Clean gradient or subtle texture",
            "camera_angle": "Eye-level, straight-on professional shot",
            "texture_materials": "Smooth, premium finishes",
            "style_keywords": style["style"]
        }
    
    try:
        system_prompt = get_design_plan_prompt(
            brand_name=brand.get("name", "Brand"),
            industry=brand.get("industry", "general"),
            primary_color=brand.get("primary_color", "#6366f1"),
            tone=brand.get("tone", "professional"),
            user_prompt=user_prompt,
            reference_assets=relevant_assets
        )
        
        response_text = await call_llm(system_prompt)
        
        # Clean up response and parse JSON
        response_text = response_text.strip()
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()
        
        design_plan = json.loads(response_text)
        return design_plan
    except Exception as e:
        print(f"Design plan generation error: {e}")
        tone = brand.get("tone", "professional").lower()
        style = STYLE_PRESETS.get(tone, STYLE_PRESETS["professional"])
        return {
            "core_visual": f"Professional {brand.get('industry', 'business')} visual",
            "hero_element": "Main focal point",
            "color_palette": brand.get("primary_color", "#6366f1"),
            "lighting_style": style["lighting"],
            "mood": style["mood"],
            "composition_anchor": "Centered",
            "background_treatment": "Clean",
            "camera_angle": "Eye-level",
            "texture_materials": "Premium",
            "style_keywords": style["style"]
        }


async def generate_master_prompt(
    user_prompt: str,
    brand: dict,
    design_plan: dict = None,
    relevant_assets: List[dict] = None,
    target_audience: str = "",
    age_group: str = "",
    gender: str = "",
    location: str = "",
    language: str = "English",
    creativity_level: str = "medium",
    text_content: str = "",
    font_family: str = "",
    extras: str = ""
) -> str:
    """
    Generate the master prompt for image generation.
    This prompt is used as the blueprint for ALL platform variants.
    """
    if not has_llm_configured():
        # Fallback prompt construction
        tone = brand.get("tone", "professional").lower()
        style = STYLE_PRESETS.get(tone, STYLE_PRESETS["professional"])
        
        prompt_parts = [
            user_prompt,
            f"Brand: {brand.get('name', 'Brand')}",
            f"Style: {style['style']}",
            f"Mood: {style['mood']}",
            f"Lighting: {style['lighting']}",
            f"Colors: {brand.get('primary_color', '#6366f1')}, {style['colors']}",
        ]
        
        if design_plan:
            if design_plan.get("core_visual"):
                prompt_parts.append(f"Scene: {design_plan['core_visual']}")
            if design_plan.get("hero_element"):
                prompt_parts.append(f"Focus: {design_plan['hero_element']}")
        
        prompt_parts.extend(IMAGE_QUALITY_ENHANCERS[:4])
        
        return ", ".join(prompt_parts)
    
    try:
        # Format campaign details
        campaign_details = format_campaign_details(
            user_prompt=user_prompt,
            brand_name=brand.get("name", ""),
            brand_colors=brand.get("primary_color", ""),
            brand_tone=brand.get("tone", "professional"),
            target_audience=target_audience,
            age_group=age_group,
            gender=gender,
            location=location,
            language=language,
            creativity_level=creativity_level,
            text_content=text_content,
            font_family=font_family,
            extras=extras,
            design_plan=design_plan,
            reference_assets=relevant_assets
        )
        
        # Get the full prompt template
        full_prompt = get_master_prompt(campaign_details)
        
        # Call LLM to generate the image prompt
        enhanced_prompt = await call_llm(full_prompt)
        
        # Clean up the response
        enhanced_prompt = enhanced_prompt.strip()
        if enhanced_prompt.startswith('"') and enhanced_prompt.endswith('"'):
            enhanced_prompt = enhanced_prompt[1:-1]
        
        return enhanced_prompt
    except Exception as e:
        print(f"Master prompt generation error: {e}")
        # Fallback
        tone = brand.get("tone", "professional").lower()
        style = STYLE_PRESETS.get(tone, STYLE_PRESETS["professional"])
        return f"{user_prompt}, {brand.get('name', '')} brand, {style['style']}, {style['mood']}, {', '.join(IMAGE_QUALITY_ENHANCERS[:3])}"


def create_platform_variant_prompt(
    master_prompt: str,
    platform_name: str,
    aspect_ratio: str,
    platform_format: str = ""
) -> str:
    """
    Create a platform-specific variant of the master prompt.
    Adds composition hints based on aspect ratio.
    """
    composition_hint = PLATFORM_COMPOSITION_HINTS.get(
        aspect_ratio,
        "balanced composition"
    )
    
    variant_prompt = f"{master_prompt}, optimized for {platform_name} format ({platform_format}), {composition_hint}"
    
    return variant_prompt
