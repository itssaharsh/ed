import json
import random
import re
import time
from typing import Any
from pathlib import Path

try:
    import groq
except ImportError:
    groq = None

from google import genai
from google.genai import types

from config import PipelineConfig, ContentBrief, _CONTENT_CATEGORIES, logger
from utils import _limit_words, _ensure_shorts_hashtag, _ensure_shorts_tag, _normalize_search_query, _normalize_text

def _pick_category() -> tuple[str, str, list[str], str]:
    """Weighted-random category selection.
    Returns (category_id, description, video_search_terms, hashtags).
    """
    ids     = [c[0] for c in _CONTENT_CATEGORIES]
    weights = [c[1] for c in _CONTENT_CATEGORIES]
    descs   = [c[2] for c in _CONTENT_CATEGORIES]
    terms   = [c[3] for c in _CONTENT_CATEGORIES]
    tags    = [c[4] for c in _CONTENT_CATEGORIES]
    chosen  = random.choices(range(len(ids)), weights=weights, k=1)[0]
    return ids[chosen], descs[chosen], terms[chosen], tags[chosen]


def _build_prompt(category_id: str, category_desc: str, hashtags: str) -> str:
    """Build the final text prompt for Gemini."""
    return (
        "OUTPUT ONLY A SINGLE RAW JSON OBJECT. "
        "DO NOT wrap in markdown code fences. DO NOT add any text before or after the JSON. "
        "The very first character MUST be '{' and the very last MUST be '}'. "
        "Use exactly these six keys: title, description, script, search_query, video_keywords, punchline_words.\n\n"

        f"CATEGORY: {category_desc}\n\n"

        "FIELD RULES:\n"
        "- title: Under 70 chars. A punchy, relatable title roasting a person or situation. "
        "Like a caption someone would screenshot and text to a friend. Emoji that matches the vibe. '#shorts' at the end.\n"
        f"- description: One witty, sarcastic single-sentence hook that makes someone stop scrolling, then on a new line: {hashtags}\n"
        "- script: EXACTLY 65-85 words. Use this ESCALATION FORMULA:\n"
        "  1. The Hook (~10 words): Start with an everyday, relatable annoyance. "
        "Use 'Ever met someone who...', 'You know that person who...', or 'Nobody talks about how...'. Keep it grounded.\n"
        "  2. The Setup (~12 words): Introduce 'this friend' or 'someone I know'. One specific detail that plants the scene.\n"
        "  3. The Escalation (~30 words): Take it from PLAUSIBLE to COMPLETELY ABSURD using hyperbole. "
        "DO NOT say 'he's bad at cooking' — say 'he tried to microwave a salad and set the fire alarm off'. "
        "Include one dialogue quote. Push past the point of reason.\n"
        "  4. [PAUSE] + Punchline (~12 words): Insert the LITERAL TEXT '[PAUSE]' right before the punchline sentence. "
        "Then deliver a deadpan, devastating final line that reframes the whole story. MUST be the funniest sentence.\n"
        "  TONE: Start grounded, end in chaos. Irony, hyperbole, specific weird details. STRICTLY under 90 words.\n"
        "- search_query: 1-2 English words for the MAIN visual scene. Concrete, filmable noun.\n"
        "- video_keywords: A JSON ARRAY of exactly 8 different 1-3 word English strings. "
        "CRITICAL RULE: Use VISUAL IRONY — do NOT literally match the words being spoken. "
        "The footage should create contrast and comedic juxtaposition with the audio. "
        "If the audio says 'financial genius', show 'man burying jar in dirt'. "
        "If the audio says 'he never leaves the house', show 'person skydiving'. "
        "If audio says 'she's a great cook', show 'fire extinguisher'. "
        "Each clip should feel like a visual punchline, not an illustration. "
        "Format: [\"confident businessman\", \"man burying jar\", \"person tripping\", ...]\n"
        "- punchline_words: A JSON ARRAY of exactly 3-5 strings. "
        "These are the most ridiculous, hyperbolic, or absurd words from the script. "
        "They will be highlighted in bright red in the captions for visual emphasis. "
        "Choose the words that sound the most unhinged out of context. "
        "Must be exact words from the script, uppercase. Example: [\"MICROWAVE\", \"SALAD\", \"HELMET\", \"SUPERVISED\"]\n\n"

        "EXAMPLE OUTPUT (exact JSON structure). "
        "CRITICAL: FORMATTING EXAMPLE ONLY. DO NOT COPY THIS TOPIC. WRITE ABOUT THE SPECIFIED CATEGORY:\n"
        '{"title": "Bro Tried To Meal Prep And Almost Died 🔥 #shorts", '
        '"description": "Some people have a gift. His gift is chaos and a very patient fire department.\\n'
        '#shorts #comedy #relatable #funny #darkhumor", '
        '"script": "Ever met someone who treats the kitchen like a crime scene? Like my buddy Mark. '
        'This man cannot make toast without an incident report. Last Tuesday he goes, I\'m eating healthy now. '
        'So he tried to microwave a salad. Not cook it. Microwave it. Set the fire alarm off, evacuated the building, '
        'and the smoke smelled like betrayal and ranch dressing. [PAUSE] '
        'The salad was fine. Mark was not.", '
        '"search_query": "kitchen", '
        '"video_keywords": ["man burying jar", "person skydiving", "fire extinguisher", "confident businessman", '
        '"dog eating salad", "man pointing proudly", "slow clapping crowd", "person facepalming"], '
        '"punchline_words": ["MICROWAVE", "SALAD", "BETRAYAL", "EVACUATED"]}'
    )



def _call_gemini(api_key: str, model_name: str, prompt: str, temperature: float) -> str:
    client = genai.Client(api_key=api_key)
    safety_settings = [
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
    ]
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            top_p=0.95 if temperature == 1.0 else 0.9,
            max_output_tokens=2048,
            safety_settings=safety_settings
        )
    )
    return response.text or ""
def _parse_brief_json(raw_text: str, category_id: str = "space") -> ContentBrief | None:
    """Parse a ContentBrief from Gemini output.

    Gemini (especially the deprecated SDK) sometimes wraps JSON in markdown
    code fences like ```json\n{...}\n```.  It may also emit a flat object
    without outer braces when the response_mime_type hint is ignored.
    This function handles all three cases robustly.
    """
    if not raw_text:
        return None

    try:
        # Step 1: strip markdown code fences (``` or ```json)
        stripped = re.sub(r"^```[a-zA-Z]*\s*", "", raw_text.strip(), flags=re.MULTILINE)
        stripped = re.sub(r"```\s*$", "", stripped.strip(), flags=re.MULTILINE).strip()

        # Step 2: find JSON object boundaries
        start = stripped.find('{')
        end = stripped.rfind('}')

        if start != -1 and end != -1 and end >= start:
            # Normal case: JSON object found
            cleaned = stripped[start:end + 1]
        elif start == -1 and end == -1:
            # Degenerate case: Gemini emitted flat key-value pairs without {}
            cleaned = '{' + stripped + '}'
            logger.warning("Gemini returned flat JSON (no braces); wrapped automatically. Preview: %s", repr(raw_text[:120]))
        else:
            logger.warning("No JSON object found in raw text. Raw output was: %s", repr(raw_text))
            return None

        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("JSON parsing failed: %s. Raw text: %s", exc, raw_text)
        return None

    if not isinstance(payload, dict):
        return None

    title = _normalize_text(payload.get("title", ""), max_length=120)
    description = _normalize_text(payload.get("description", ""), max_length=500)
    script = _normalize_text(payload.get("script", ""), max_length=1500)
    search_query = _normalize_search_query(payload.get("search_query", ""))

    # Parse video_keywords — ironic/contrasting stock footage search terms
    raw_keywords = payload.get("video_keywords", [])
    if isinstance(raw_keywords, list):
        video_keywords: tuple[str, ...] = tuple(
            str(k).strip() for k in raw_keywords if k and str(k).strip()
        )[:8]
    else:
        video_keywords = ()

    # Parse punchline_words — most ridiculous words, rendered in red in captions
    raw_punchline = payload.get("punchline_words", [])
    if isinstance(raw_punchline, list):
        punchline_words: tuple[str, ...] = tuple(
            str(w).strip().upper() for w in raw_punchline if w and str(w).strip()
        )[:5]
    else:
        punchline_words = ()

    if not title or not description or not script or not search_query:
        logger.warning("Parsed JSON is missing required fields. Payload received: %s", payload)
        return None

    if not video_keywords:
        logger.warning("video_keywords missing or empty — stock footage cuts will be limited.")
    if not punchline_words:
        logger.warning("punchline_words missing — caption emphasis will be disabled.")

    script = _limit_words(script, 90)
    description = _ensure_shorts_hashtag(description)
    return ContentBrief(
        title=title,
        description=description,
        script=script,
        search_query=search_query,
        category=category_id,
        video_keywords=video_keywords,
        punchline_words=punchline_words,
    )


def generate_content_brief(config: PipelineConfig) -> ContentBrief | None:
    if not config.gemini_api_key:
        logger.error("GEMINI_API_KEY is missing. Skipping content generation.")
        # We don't return here in case Groq is configured instead
        # Wait, the logic below expects gemini to run first.
        # But if GEMINI_API_KEY is completely missing, we should still allow Groq to run.

    category_id, category_desc, video_terms, hashtags = _pick_category()
    logger.info("Selected content category: %s", category_id)

    prompt = _build_prompt(category_id, category_desc, hashtags)
    last_exc: Exception | None = None

    if config.gemini_api_key:
        try:
            # De-duplicate candidate models to prevent burning API quota on retries
            base_models = [
                config.gemini_model,
                "gemini-2.5-flash",
                "gemini-2.0-flash",
                "gemini-1.5-flash",
                "gemini-1.5-pro"
            ]
            candidate_models = []
            for m in base_models:
                if m and m not in candidate_models:
                    candidate_models.append(m)

            for candidate in candidate_models:
                attempts = 0
                while attempts < 2:
                    attempts += 1
                    try:
                        logger.info("Attempting Gemini model: %s (attempt %d/2)", candidate, attempts)

                        raw_text = _call_gemini(config.gemini_api_key, candidate, prompt, temperature=1.0)
                        brief = _parse_brief_json(raw_text, category_id)
                        if brief is not None:
                            return brief

                        logger.warning("Gemini response from model %s was not valid JSON.", candidate)
                        time.sleep(2)

                        # Retry once with a stricter instruction
                        strict_prompt = prompt + " Output valid JSON only. No leading or trailing text."
                        raw_text = _call_gemini(config.gemini_api_key, candidate, strict_prompt, temperature=0.8)
                        brief = _parse_brief_json(raw_text, category_id)
                        if brief is not None:
                            return brief

                        time.sleep(2)
                        break # if it failed parsing twice, moving to next model is better
                    except Exception as exc:
                        last_exc = exc
                        err_str = str(exc)
                        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                            if attempts < 2:
                                logger.warning("Gemini model %s hit rate limit (429). Sleeping 60s before retry.", candidate)
                                time.sleep(60)
                                continue
                            else:
                                logger.warning("Gemini model %s rate limit persisted. Moving to next model.", candidate)
                                break
                        elif "404" in err_str or "NOT_FOUND" in err_str:
                            logger.warning("Gemini model %s not found (404). Skipping.", candidate)
                            break
                        else:
                            logger.warning("Gemini model %s failed: %s", candidate, exc)
                            break

            logger.error("All Gemini models failed. Last exception: %s", last_exc)
            
            # Dump diagnostic info
            err_log = Path("gemini_error.log")
            err_log.write_text(f"Last exception: {last_exc}\n", encoding="utf-8")
            logger.info("Wrote Gemini diagnostic to %s", err_log.resolve())
            
        except Exception as exc:
            logger.exception("Fatal error generating content brief with Gemini: %s", exc)

    # --- GROQ FALLBACK ---
    if config.groq_api_key and groq is not None:
        logger.warning("Falling back to Groq API using Llama 3 8B...")
        try:
            client = groq.Groq(api_key=config.groq_api_key)
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": (
                        "You are a stand-up comedy writer specializing in YouTube Shorts. "
                        "Three rules: "
                        "1) ESCALATION: start relatable, escalate to completely absurd hyperbole by the end (don't say 'bad at cooking', say 'tried to microwave a salad and evacuated the building'). "
                        "2) VISUAL IRONY: video_keywords must CONTRAST the audio, not match it literally (audio says 'financial genius' = show 'man burying jar in dirt'). "
                        "3) PAUSE: insert the literal text '[PAUSE]' right before the punchline sentence for comedic timing. "
                        "You ONLY output pure JSON — no markdown fences, no extra text, just raw JSON."
                    )},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=900,
            )
            raw_text = response.choices[0].message.content
            if raw_text:
                brief = _parse_brief_json(raw_text, category_id)
                if brief is not None:
                    logger.info("Groq fallback successful!")
                    return brief
                
                logger.warning("Groq response was not valid JSON.")
                
                # Retry once strictly
                time.sleep(2)
                strict_prompt = prompt + " Output valid JSON only. No leading or trailing text."
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": (
                            "You are a conversational stand-up comedy writer for YouTube Shorts. "
                            "You write like you're roasting a friend to another friend at 2am — sarcastic, observational, grounded in real situations. "
                            "Specific funny details, direct dialogue quotes, and absurd-but-real anecdotes. "
                            "NOT chaos rants or conspiracy theories. Real people doing dumb things, told in a punchy story. "
                            "You ONLY output pure JSON — no markdown fences, no extra text, just raw JSON."
                        )},
                        {"role": "user", "content": strict_prompt}
                    ],
                    model="llama-3.1-8b-instant",
                    temperature=0.5,
                    max_tokens=800,
                )
                raw_text = response.choices[0].message.content
                if raw_text:
                    brief = _parse_brief_json(raw_text, category_id)
                    if brief is not None:
                        logger.info("Groq fallback successful on retry!")
                        return brief
        except Exception as e:
            logger.error("Groq fallback failed: %s", e)
    elif not config.groq_api_key:
        logger.warning("GROQ_API_KEY is missing. Cannot use Groq fallback.")
    elif groq is None:
        logger.warning("groq library is not installed. Cannot use Groq fallback.")
    # --- END GROQ FALLBACK ---
    
    return None
