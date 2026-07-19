import json
import random
import re
import time
from typing import Any

from google import genai
from google.genai import types

from config import PipelineConfig, ContentBrief, _CONTENT_CATEGORIES, logger
from utils import _limit_words, _ensure_shorts_hashtag, _ensure_shorts_tag, _normalize_search_query

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
    """Build a category-specific Gemini prompt using the research-backed viral formula:

    4-Beat structure (proven for 70%+ completion rate):
      1. HOOK   — Contradicts common belief or states shocking fact in <1.5s of audio
      2. CONTEXT — Why you should care (no fluff, 1-2 sentences)
      3. PAYOFF  — The disturbing/surprising depth of the fact
      4. LOOP    — Final sentence loops back to the opening for replay bait

    Target: 55-70 words = ~22-28 seconds at 150 WPM (the proven Shorts sweet spot).
    """
    category_instructions = {
        "finance": (
            "Topic: Pick ONE shocking personal finance or wealth psychology fact. "
            "Examples: how banks profit from your savings account, a cognitive bias that keeps people poor, "
            "a tax trick only the wealthy know, how inflation secretly transfers wealth upward, "
            "why the stock market is rigged against retail investors, or how compound interest works against debt holders. "
            "Hook style: Start with 'The bank is...' or 'Right now...' or 'Every time you...' — immediate, personal, alarming. "
            "Punchline: End with a cynical observation about who benefits from the system. "
            "Loop: Last sentence should echo or contradict the first sentence so it loops naturally."
        ),
        "ai_tech": (
            "Topic: Pick ONE genuinely disturbing AI or tech fact. "
            "Examples: how recommendation algorithms detect depression before you do, "
            "how your phone microphone data is sold, how AI can predict your vote, "
            "how facial recognition is being used without your consent, "
            "how large language models encode bias, or how social media maximizes addiction not connection. "
            "Hook style: Start with 'Right now, AI...' or 'Your phone already knows...' or 'The algorithm...' "
            "Punchline: End with what this means for human autonomy or privacy. "
            "Loop: Last sentence should mirror the opening to encourage replays."
        ),
        "dark_psychology": (
            "Topic: Pick ONE specific, real psychological manipulation technique. "
            "Examples: the foot-in-the-door technique used by subscription services, "
            "how casinos use variable reward schedules to create addiction, "
            "how dark patterns in app design exploit loss aversion, "
            "how social proof is manufactured, or how anchoring is used in pricing. "
            "Hook style: Start with 'This trick is being used on you right now.' or 'You have already fallen for this today.' "
            "Punchline: Name the industry or company exploiting this technique. "
            "Loop: End with a line that makes the viewer want to watch again to catch what they missed."
        ),
        "dark_history": (
            "Topic: Pick ONE genuinely disturbing, lesser-known historical fact. "
            "Examples: a forgotten atrocity, a government experiment on citizens, "
            "a covered-up disaster, a historical figure's hidden crimes, "
            "or a suppressed invention/discovery. "
            "Hook style: Start with 'This actually happened.' or 'In [year]...' or 'A government once...' "
            "Punchline: Connect it to something that still affects us today, or note it was never taught in school. "
            "Loop: End with a question or statement that mirrors the opening."
        ),
        "nature_horror": (
            "Topic: Pick ONE genuinely disturbing fact about a real animal, parasite, or biological phenomenon. "
            "Examples: a parasite that controls its host's brain, a deep sea creature that hunts with bioluminescence, "
            "an animal that digests its prey alive, how cordyceps fungi work, "
            "or the extreme conditions life can survive. "
            "Hook style: Start with the disturbing fact immediately, no preamble. Deadpan tone. "
            "Punchline: End with a twist about how this relates to human biology or everyday life. "
            "Loop: Final sentence creates urgency to rewatch."
        ),
    }

    cat_instruction = category_instructions.get(category_id, category_instructions["dark_psychology"])

    return (
        "OUTPUT ONLY A SINGLE RAW JSON OBJECT. "
        "DO NOT wrap in markdown code fences. DO NOT add any text before or after the JSON. "
        "The very first character MUST be '{' and the very last MUST be '}'. "
        "Use exactly these four string keys: title, description, script, search_query.\n\n"

        f"CATEGORY: {category_desc}\n\n"

        "FIELD RULES:\n"
        "- title: Under 70 chars. Starts with a hook question or alarming statement. "
        "Must create a curiosity gap. Include '#shorts' at the end.\n"
        f"- description: One punchy sarcastic sentence, then on a new line: {hashtags}\n"
        "- script: EXACTLY 55-70 words of spoken narration using the 4-BEAT VIRAL FORMULA:\n"
        "  BEAT 1 (Hook, ~10 words): Contradicts common belief OR states shocking fact. MUST be "
        "completable in under 1.5 seconds of audio. No 'Hey guys' or slow intros.\n"
        "  BEAT 2 (Context, ~15 words): Why this matters to the viewer personally. One sentence.\n"
        "  BEAT 3 (Payoff, ~30 words): The full disturbing depth. Short punchy sentences. Sarcastic, comedic tone.\n"
        "  BEAT 4 (Loop, ~10 words): Final sentence mirrors or contradicts Beat 1 to encourage replays.\n"
        "  NO filler words. NO 'And so...' or 'In conclusion'. Deliver with a very dark or funny angle like a cynical stand-up comedian.\n"
        "- search_query: 1-2 English words describing a VISUAL that exists in stock video libraries. "
        "Use concrete, filmable nouns (e.g. 'money', 'server room', 'ocean', 'crowd', 'ruins'). "
        "NOT abstract concepts like 'freedom' or 'fear'.\n\n"

        f"SPECIFIC INSTRUCTIONS:\n{cat_instruction}\n\n"

        "EXAMPLE OUTPUT (exact JSON structure required):\n"
        '{"title": "Your Savings Account Is a Lie #shorts", '
        '"description": "The bank thanks you for your generous donation.\\n'
        '#shorts #finance #money #wealth #facts #banking #mindblown #moneytips", '
        '"script": "Your savings account is not saving you. The average savings rate is 0.4 percent. '
        'Inflation runs at 3 percent. Every year you leave money in that account, you are paying the bank '
        'to hold it. They lend it out at 20 percent interest and give you 0.4. '
        'Your savings account is not saving you.", '
        '"search_query": "money"}'
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
            response_mime_type="application/json",
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
            logger.warning("No JSON object found in raw text. Raw output was: %s", repr(raw_text[:200]))
            return None

        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("JSON parsing failed: %s. Raw text: %s", exc, raw_text[:300])
        return None

    if not isinstance(payload, dict):
        return None

    title = _normalize_text(payload.get("title", ""), max_length=120)
    description = _normalize_text(payload.get("description", ""), max_length=500)
    script = _normalize_text(payload.get("script", ""), max_length=1500)
    search_query = _normalize_search_query(payload.get("search_query", ""))

    if not title or not description or not script or not search_query:
        logger.warning("Parsed JSON is missing required fields. Payload received: %s", payload)
        return None

    script = _limit_words(script, 90)
    description = _ensure_shorts_hashtag(description)
    return ContentBrief(title=title, description=description, script=script, search_query=search_query, category=category_id)


def generate_content_brief(config: PipelineConfig) -> ContentBrief | None:
    if not config.gemini_api_key:
        logger.error("GEMINI_API_KEY is missing. Skipping content generation.")
        return None

    category_id, category_desc, video_terms, hashtags = _pick_category()
    logger.info("Selected content category: %s", category_id)

    try:
        prompt = _build_prompt(category_id, category_desc, hashtags)

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

        last_exc: Exception | None = None
        for candidate in candidate_models:
            try:
                logger.info("Attempting Gemini model: %s", candidate)

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

            except Exception as exc:  # try the next model
                logger.warning("Gemini model %s failed: %s", candidate, exc)
                last_exc = exc

        # If we reach here, all model attempts failed or returned invalid JSON.
        if last_exc:
            err_path = config.workspace / "gemini_error.log"
            err_text = f"Last Gemini exception for models {candidate_models}: {last_exc!r}\n"
            try:
                err_path.write_text(err_text, encoding="utf-8")
                logger.info("Wrote Gemini diagnostic to %s", err_path)
            except Exception:
                logger.exception("Failed to write Gemini diagnostic file.")

    except Exception as exc:
        logger.error("Gemini generation failed: %s", exc)
        return None
