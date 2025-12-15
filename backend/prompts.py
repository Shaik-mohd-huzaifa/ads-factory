"""
Prompt templates for the AdaptiveBrand campaign generation system.
"""

# Master Prompt Generator Template - STANDALONE, DETAILED, NO ATTACHMENTS
MASTER_PROMPT_TEMPLATE = """You are an Expert Visual Prompt Engineer specializing in creating highly detailed, comprehensive prompts for AI image generation models. Your task is to transform campaign requirements into a single, self-contained, production-ready image generation prompt.

=== YOUR ROLE ===
You generate ONE comprehensive prompt that will be passed DIRECTLY to an image generation model (like Midjourney, DALL-E, Stable Diffusion, Freepik Imagen). The prompt must be complete, detailed, and standalone - requiring no additional context.

=== INPUT PARAMETERS ===
{campaign_details}

=== PROMPT GENERATION RULES ===

- Campaign Intent (Intent of the Campaign or What the Image should Convey)
- User Prompt (Contains the user requirement on how the Ad or Image should look or what exactly he wants)
- Brand Name (If there add it to the Image in sides or as per the user if mention)
- Color Scheme (Color Scheme which should be included in the Add)
- Font Preference (Mention the font type or suggest one depending on the other parameters)
- Target Audience (Explain in a way that the Image Gen Model should understand what to create. Do not include direct Target Audience)
- Age Group (Mention how it should look depending on the age group mention rather that mentioning age directly)
- Gender (Mention the theme or other thing as per the Gender but not direct Gender)
- Location (Understand what location is mention and try to add the pov of that location rather that mentioning it directly)
- Language (Text Language or create a text in the prefered language and then pass mention it in the prompt so that the Image Gen Model don't have to translate it)
- Creativity Level (Modify the prompt and add creativity or mention creative design prompt)
- Text (Text to be mentioned in the Image. Also Say that Do not make any Mistakes in Text On the Image)
- Font Family (If Mentioned Add it also in the Prompt)
- Extras (Anything which user forget to mention above regarding how the image should be)

Here are the Parameters:

=== GENERATE THE PROMPT NOW ===
Create a comprehensive, standalone image generation prompt based on the parameters above. Output ONLY the prompt, nothing else."""

# Campaign Brief Generator Template
CAMPAIGN_BRIEF_TEMPLATE = """You are a creative director generating a campaign brief.

Brand: {brand_name}
Industry: {industry}
Brand Colors: {colors}
Brand Tone: {tone}

User Request: {user_prompt}

Generate a JSON campaign brief with:
{{
    "headline": "A catchy headline for the campaign (max 10 words)",
    "tagline": "A supporting tagline (max 15 words)", 
    "suggested_cta": "Call to action text (2-4 words)",
    "mood": "The overall mood/feeling",
    "key_message": "The core message to convey"
}}

Return ONLY valid JSON, no explanations.
"""

# Design Plan Generator Template
DESIGN_PLAN_TEMPLATE = """You are a senior art director creating a master design blueprint.

This design plan will ensure visual consistency across ALL platform formats (Instagram, Facebook, YouTube, LinkedIn, TikTok, Twitter).

Brand Context:
- Brand: {brand_name}
- Industry: {industry}  
- Primary Color: {primary_color}
- Brand Tone: {tone}

Campaign Request: {user_prompt}

Reference Assets Found:
{reference_assets}

Create a comprehensive design plan that can be applied consistently across square (1:1), portrait (9:16), and landscape (16:9) formats.

Return a JSON object with these exact keys:
{{
    "core_visual": "The central visual element/scene that anchors all designs",
    "hero_element": "The main focal point that draws attention",
    "color_palette": "Primary, secondary, and accent colors to use",
    "lighting_style": "The lighting mood and direction",
    "mood": "The emotional tone of the visuals",
    "composition_anchor": "Where the main elements should be positioned",
    "background_treatment": "How the background should be handled",
    "camera_angle": "The perspective/angle for the main visual",
    "texture_materials": "Surface qualities and material finishes",
    "style_keywords": "5-7 style descriptors for consistency"
}}

Return ONLY valid JSON, no explanations.
"""

# Style presets based on brand tone
STYLE_PRESETS = {
    "professional": {
        "lighting": "clean studio lighting with soft shadows",
        "mood": "confident, trustworthy, polished",
        "colors": "deep blues, grays, white accents",
        "style": "minimalist, corporate, sleek"
    },
    "playful": {
        "lighting": "bright, colorful, vibrant lighting",
        "mood": "fun, energetic, youthful",
        "colors": "bold primaries, pastels, gradients",
        "style": "dynamic, whimsical, bold shapes"
    },
    "luxury": {
        "lighting": "dramatic lighting with rich shadows",
        "mood": "elegant, exclusive, sophisticated",
        "colors": "gold, black, deep jewel tones",
        "style": "refined, premium, high-end textures"
    },
    "modern": {
        "lighting": "natural light with geometric shadows",
        "mood": "innovative, forward-thinking, clean",
        "colors": "monochrome with accent pops",
        "style": "geometric, minimal, tech-forward"
    },
    "warm": {
        "lighting": "golden hour, warm ambient light",
        "mood": "welcoming, friendly, approachable",
        "colors": "earth tones, warm neutrals, sunset hues",
        "style": "organic, natural, comfortable"
    },
    "bold": {
        "lighting": "high contrast, dramatic",
        "mood": "confident, powerful, attention-grabbing",
        "colors": "vibrant contrasts, neon accents",
        "style": "striking, impactful, memorable"
    }
}

# Image quality enhancers
IMAGE_QUALITY_ENHANCERS = [
    "8K resolution",
    "photorealistic",
    "professional photography",
    "studio quality",
    "sharp focus",
    "high detail",
    "perfect composition"
]

# Platform-specific composition hints
PLATFORM_COMPOSITION_HINTS = {
    "square_1_1": "centered composition, balanced elements, suitable for feed posts",
    "portrait_9_16": "vertical flow, top-to-bottom hierarchy, mobile-optimized",
    "landscape_16_9": "horizontal spread, rule of thirds, cinematic feel",
    "widescreen_1_91_1": "panoramic composition, spacious layout, banner-style"
}


def format_campaign_details(
    user_prompt: str,
    brand_name: str = "",
    brand_colors: str = "",
    brand_tone: str = "professional",
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
    """Format campaign details for the master prompt template. 
    This is a STANDALONE formatter - no design plans or references attached.
    """
    
    # Get style preset based on brand tone
    style = STYLE_PRESETS.get(brand_tone.lower(), STYLE_PRESETS["professional"])
    
    # Build comprehensive campaign details
    details = []
    
    # Core campaign info
    details.append(f"Campaign Intent: {user_prompt}")
    details.append(f"User Prompt: {user_prompt}")
    
    # Brand information
    if brand_name:
        details.append(f"Brand Name: {brand_name}")
    
    # Color scheme - combine brand colors with style preset
    color_info = []
    if brand_colors:
        color_info.append(brand_colors)
    color_info.append(style['colors'])
    details.append(f"Color Scheme: {', '.join(color_info)}")
    
    # Typography
    font_desc = font_family if font_family else f"Clean, modern typography matching {brand_tone} aesthetic"
    details.append(f"Font Preference: {font_desc}")
    
    # Target audience (indirect visual cues)
    if target_audience:
        details.append(f"Target Audience: Design visuals that appeal to {target_audience} through color psychology, imagery style, and composition")
    
    # Age group (aesthetic preferences, not direct mention)
    if age_group:
        age_aesthetics = {
            "young": "vibrant, trendy, dynamic with bold graphics and modern aesthetics",
            "millennial": "clean, minimalist, Instagram-worthy with authentic feel",
            "professional": "polished, sophisticated, corporate with refined elegance",
            "senior": "classic, trustworthy, clear with traditional elegance"
        }
        age_style = age_aesthetics.get(age_group.lower(), f"aesthetic preferences suitable for {age_group}")
        details.append(f"Age Group: {age_style}")
    
    # Gender (theme-based, not direct)
    if gender:
        details.append(f"Gender: Incorporate visual themes and color psychology that resonates with {gender} preferences")
    
    # Location (POV and visual elements)
    if location:
        details.append(f"Location: Include visual elements, architectural hints, or environmental cues inspired by {location} without directly showing location text")
    
    # Language for any text
    details.append(f"Language: {language} (any text must be in this language)")
    
    # Creativity level
    creativity_desc = {
        "low": "Conservative approach - clean, proven layouts with minimal risk",
        "medium": "Balanced creativity - modern design with strategic innovation",
        "high": "Maximum creativity - experimental, avant-garde, boundary-pushing visuals"
    }
    details.append(f"Creativity Level: {creativity_desc.get(creativity_level.lower(), creativity_desc['medium'])}")
    
    # Text content (critical - must be exact)
    if text_content:
        details.append(f"Text: '{text_content}'")
        details.append("TEXT RENDERING RULE: Display text EXACTLY as written above. NO spelling mistakes. NO character substitutions. Verify each letter.")
    
    # Extras - combine user extras with style presets and quality enhancers
    all_extras = []
    if extras:
        all_extras.append(extras)
    all_extras.extend([
        f"Visual Style: {style['style']}",
        f"Mood: {style['mood']}",
        f"Lighting: {style['lighting']}",
        f"Quality: {', '.join(IMAGE_QUALITY_ENHANCERS)}"
    ])
    details.append(f"Extras: {'; '.join(all_extras)}")
    
    return "\n".join(details)


def get_master_prompt(campaign_details: str) -> str:
    """Get the complete master prompt with campaign details filled in."""
    return MASTER_PROMPT_TEMPLATE.format(campaign_details=campaign_details)


def get_campaign_brief_prompt(
    brand_name: str,
    industry: str,
    colors: str,
    tone: str,
    user_prompt: str
) -> str:
    """Get the campaign brief generation prompt."""
    return CAMPAIGN_BRIEF_TEMPLATE.format(
        brand_name=brand_name,
        industry=industry,
        colors=colors,
        tone=tone,
        user_prompt=user_prompt
    )


def get_design_plan_prompt(
    brand_name: str,
    industry: str,
    primary_color: str,
    tone: str,
    user_prompt: str,
    reference_assets: list = None
) -> str:
    """Get the design plan generation prompt."""
    assets_text = "None found - create fresh design"
    if reference_assets:
        asset_descriptions = []
        for asset in reference_assets[:5]:
            desc = f"- {asset.get('type', 'asset')}: {asset.get('name', asset.get('prompt', 'unnamed'))[:50]}"
            if asset.get('score'):
                desc += f" (relevance: {asset['score']:.0%})"
            asset_descriptions.append(desc)
        assets_text = "\n".join(asset_descriptions)
    
    return DESIGN_PLAN_TEMPLATE.format(
        brand_name=brand_name,
        industry=industry,
        primary_color=primary_color,
        tone=tone,
        user_prompt=user_prompt,
        reference_assets=assets_text
    )
