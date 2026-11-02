
"""
FruitSenseAI: CNN + VLM analysis JSON -> Gemini explanation -> Sarvam AI TTS

Run from:
G:\\project1\\Prototype1\\FruitSenseAI

Text only:
    uv run python src/04_llm_voice_explainer.py --no-tts

Text + English speech:
    uv run python src/04_llm_voice_explainer.py

Tamil speech:
    uv run python src/04_llm_voice_explainer.py --language ta-IN

Optional custom input JSON:
    uv run python src/04_llm_voice_explainer.py --json "path\\to\\analysis_results.json"
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from google import genai
from sarvamai import SarvamAI


# ---------------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "explainability_vlm"
    / "analysis_results.json"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "llm_voice_explainer"

# Change this through --gemini-model if necessary.
GEMINI_MODEL = "gemini-3.1-flash-lite"

SARVAM_TTS_MODEL = "bulbul:v4-flash"

# Short, natural explanations are better for speech.
MAX_SPOKEN_WORDS = 45


# ---------------------------------------------------------
# 2. LOAD EXISTING CNN + VLM JSON
# ---------------------------------------------------------

def load_analysis(json_path: Path) -> dict[str, Any]:
    """Load the existing explainability JSON without rerunning models."""

    if not json_path.exists():
        raise FileNotFoundError(
            f"\nAnalysis JSON not found:\n{json_path}\n\n"
            "Run your explainability script first, or pass --json with "
            "the correct path to analysis_results.json."
        )

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    # Support either one JSON object or a list of result objects.
    if isinstance(data, list):
        if not data:
            raise ValueError("The analysis JSON contains an empty list.")
        data = data[-1]

    if not isinstance(data, dict):
        raise ValueError(
            "Expected analysis_results.json to contain an object or a list."
        )

    return data


def first_value(data: dict[str, Any], *keys: str) -> Any:
    """Return the first non-empty value among possible field names."""

    for key in keys:
        value = data.get(key)
        if value is not None and value != "" and value != [] and value != {}:
            return value

    return None


def normalize_evidence(data: dict[str, Any]) -> dict[str, Any]:
    """
    Extract the useful information from different versions of the
    existing explainability JSON.
    """

    prediction = first_value(
        data,
        "cnn_predicted_class",
        "predicted_class",
        "cnn_prediction",
        "prediction",
        "predicted_label",
    )

    confidence = first_value(
        data,
        "cnn_confidence",
        "confidence",
        "predicted_confidence",
        "top_confidence",
    )

    description = first_value(
        data,
        "vlm_description",
        "vlm_visual_description",
        "visual_description",
        "vlm_response",
        "vlm_caption",
        "description",
    )

    probabilities = first_value(
        data,
        "cnn_class_probabilities",
        "class_probabilities",
        "probabilities",
        "all_probabilities",
    )

    disagreement = first_value(
        data,
        "fruit_label_disagreement",
        "cnn_vlm_disagreement",
        "model_disagreement",
    )

    review_status = first_value(
        data,
        "review_status",
        "freshness_review_status",
    )

    return {
        "predicted_fruit_condition": prediction,
        "model_confidence": confidence,
        "class_probabilities": probabilities,
        "visual_description": description,
        "models_disagree": disagreement,
        "review_status": review_status,
    }


# ---------------------------------------------------------
# 3. GEMINI: CONVERT MODEL RESULTS INTO NORMAL LANGUAGE
# ---------------------------------------------------------

def generate_explanation(
    evidence: dict[str, Any],
    model_name: str,
) -> str:
    """Generate a concise fruit-quality explanation using Gemini."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "\nGEMINI_API_KEY is not set.\n"
            'In PowerShell, run:\n$env:GEMINI_API_KEY="YOUR_API_KEY"'
        )

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are the fruit-quality assistant in a practical app.

The user cares about whether the fruit looks fresh, damaged, or possibly
spoiled. Explain the result as a normal person would speak to another person.

EVIDENCE FROM THE EXISTING ANALYSIS:
{json.dumps(evidence, ensure_ascii=False, default=str)}

STRICT OUTPUT RULES:
1. Output only one or two short sentences, maximum 45 words.
2. Use everyday language, not technical or scientific report language.
3. Start with the likely fruit condition.
4. Mention only the most relevant visible signs or concern.
5. If the visual description suggests mold, say something like:
   "There may be mold on the surface, so check that area carefully."
6. If bruises, soft spots, dark spots, or unusual discoloration are
   described, mention the relevant issue naturally.
7. Never invent mold, bruises, spots, or other defects that are not
   supported by the evidence.
8. If no visible problem is described, say the fruit looks fresh based
   on its visible appearance.
9. If the prediction and visual description conflict, say the condition
   is unclear and briefly explain why.
10. Do not mention CNN, VLM, Gemini, AI, confidence percentages,
    probabilities, JSON, Grad-CAM, or model agreement.
11. Do not say the fruit is guaranteed fresh or safe to eat.
12. Do not add greetings, headings, introductions, generic disclaimers,
    long advice, or unnecessary explanations.
13. Do not describe what you are doing. Return only the final sentences
    that should appear on screen and be spoken aloud.

Examples of the desired style:

"The apple looks fresh, with a smooth red surface. Check for any soft
spots before eating."

"There may be mold on the surface, so check that area carefully. If you
find mold, avoid eating the fruit."

"The fruit's condition is unclear because its appearance does not match
the prediction. Check it carefully for signs of spoilage."
""".strip()

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={
            "temperature": 0.2,
            "max_output_tokens": 120,
        },
    )

    explanation = (response.text or "").strip()

    # Remove accidental markdown or quote wrappers.
    explanation = explanation.strip(" \t\n\"'")

    explanation = re.sub(
        r"^(Explanation|Result|Fruit quality)\s*:\s*",
        "",
        explanation,
        flags=re.IGNORECASE,
    )

    if not explanation:
        raise RuntimeError(
            "Gemini returned an empty response. Check the model name, "
            "API key, and API quota."
        )

    return explanation


def translate_for_speech(
    text: str,
    language_code: str,
    model_name: str,
) -> str:
    """Translate explanation into the target language script for Sarvam AI TTS."""
    if language_code == "en-IN" or not text:
        return text

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return text

    lang_map = {"ta-IN": "Tamil", "hi-IN": "Hindi"}
    lang_name = lang_map.get(language_code)
    if not lang_name:
        return text

    prompt = (
        f"Translate this short fruit quality explanation into natural spoken {lang_name}. "
        f"Output ONLY the translated text in the native {lang_name} script, under 30 words, with no explanations, greetings, quotes, or notes.\n\n"
        f"Text to translate:\n{text}"
    )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "temperature": 0.2,
                "max_output_tokens": 200,
                "thinking_config": {"thinking_budget": 0},
            },
        )
        translated = (response.text or "").strip().strip(" \t\n\"'")
        return translated if translated else text
    except Exception:
        return text


# ---------------------------------------------------------
# 4. SARVAM: CONVERT THE SAME TEXT INTO SPEECH
# ---------------------------------------------------------

def generate_speech(
    explanation: str,
    language_code: str,
    output_path: Path,
) -> None:
    """Generate speech and save the returned WAV audio."""

    api_key = os.getenv("SARVAM_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "\nSARVAM_API_KEY is not set.\n"
            'In PowerShell, run:\n$env:SARVAM_API_KEY="YOUR_API_KEY"\n'
            "Or test Gemini text generation first with --no-tts."
        )

    client = SarvamAI(api_subscription_key=api_key)

    response = client.text_to_speech.convert(
        model=SARVAM_TTS_MODEL,
        text=explanation[:2400],
        language_code=language_code,
    )

    # Sarvam returns audio as base64-encoded WAV data.
    audio_list = getattr(response, "audios", None)

    if not audio_list:
        raise RuntimeError(
            "Sarvam returned no audio. Check the API response and your quota."
        )

    audio_bytes = base64.b64decode(audio_list[0])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(audio_bytes)


# ---------------------------------------------------------
# 5. MAIN PIPELINE
# ---------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Explain existing FruitSenseAI CNN + VLM results "
            "using Gemini and optionally generate speech with Sarvam."
        )
    )

    parser.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_JSON,
        help="Path to the existing analysis_results.json file.",
    )

    parser.add_argument(
        "--language",
        choices=["en-IN", "ta-IN", "hi-IN"],
        default="en-IN",
        help="Speech language: English, Tamil, or Hindi.",
    )

    parser.add_argument(
        "--gemini-model",
        default=GEMINI_MODEL,
        help="Gemini model ID to use.",
    )

    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Generate the explanation without calling Sarvam.",
    )

    args = parser.parse_args()

    try:
        # Step 1: Load existing analysis.
        json_path = args.json.resolve()
        raw_analysis = load_analysis(json_path)
        evidence = normalize_evidence(raw_analysis)

        if (
            not evidence["predicted_fruit_condition"]
            and not evidence["visual_description"]
        ):
            raise ValueError(
                "Could not find a CNN prediction or VLM description in "
                "the JSON. Check the field names in analysis_results.json."
            )

        # Step 2: Ask Gemini for a simple explanation.
        print(f"Input JSON: {json_path}")
        print(f"Gemini model: {args.gemini_model}")
        print("\nGenerating fruit-quality explanation...")

        explanation = generate_explanation(
            evidence=evidence,
            model_name=args.gemini_model,
        )

        # Step 3: Prepare output paths.
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        text_path = OUTPUT_DIR / f"explanation_{timestamp}.txt"
        result_path = OUTPUT_DIR / f"explanation_{timestamp}.json"
        audio_path = OUTPUT_DIR / (
            f"speech_{timestamp}_{args.language}.wav"
        )

        # Step 4: Save and display the explanation.
        text_path.write_text(
            explanation + "\n",
            encoding="utf-8",
        )

        print("\n========== FRUIT QUALITY ==========\n")
        print(explanation)
        print("\n===================================\n")

        # Step 5: Generate speech if requested.
        if args.no_tts:
            saved_audio_path = None
            print("Speech skipped because --no-tts was selected.")
        else:
            print(f"Generating Sarvam speech in {args.language}...")

            spoken_text = translate_for_speech(
                text=explanation,
                language_code=args.language,
                model_name=args.gemini_model,
            )
            generate_speech(
                explanation=spoken_text,
                language_code=args.language,
                output_path=audio_path,
            )

            saved_audio_path = str(audio_path.resolve())
            print(f"Audio saved: {saved_audio_path}")

        # Step 6: Save a record of the result.
        output_record = {
            "source_json": str(json_path),
            "gemini_model": args.gemini_model,
            "sarvam_model": (
                None if args.no_tts else SARVAM_TTS_MODEL
            ),
            "speech_language": (
                None if args.no_tts else args.language
            ),
            "evidence_used": evidence,
            "user_facing_explanation": explanation,
            "audio_path": saved_audio_path,
        }

        result_path.write_text(
            json.dumps(
                output_record,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

        print(f"Text saved: {text_path.resolve()}")
        print(f"Result JSON saved: {result_path.resolve()}")

        return 0

    except Exception as error:
        print("\nERROR:", str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
