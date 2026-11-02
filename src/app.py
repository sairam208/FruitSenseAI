"""FruitSenseAI - AI-Powered Fruit Quality Analysis Web Application

Interactive Streamlit application integrating MobileNetV3Large CNN classification,
Grad-CAM interpretability, SmolVLM visual description, Gemini explanation, and Sarvam AI TTS.
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from PIL import Image
import streamlit as st

from src.config import (
    CLASS_DISPLAY_NAMES,
    CLASS_NAMES,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_GEMINI_MODEL,
    PROJECT_ROOT,
    SUPPORTED_LANGUAGES,
    get_gemini_api_key,
    get_sarvam_api_key,
)
from src.gemini_explainer import (
    generate_fallback_explanation,
    generate_gemini_explanation,
    translate_explanation,
)
from src.inference import (
    get_device,
    load_cnn_model,
    predict_fruit,
    validate_and_load_image,
)
from src.speech_service import generate_speech
from src.visual_analysis import (
    analyze_vlm_cues,
    build_evidence_object,
    compute_gradcam,
    generate_vlm_description,
    load_vlm,
    save_run_artifacts,
)

# Page configuration
st.set_page_config(
    page_title="FruitSenseAI — Visible Fruit Quality Analysis",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for rich agricultural & AI aesthetics
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Background gradient styling */
.stApp {
    background: radial-gradient(circle at 10% 20%, rgba(13, 31, 20, 0.95) 0%, rgba(10, 18, 12, 1) 90%);
    color: #e2e8f0;
}

/* Header badge & title */
.header-container {
    text-align: center;
    padding: 2.2rem 1rem 1.8rem 1rem;
    margin-bottom: 2rem;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.05) 100%);
    border: 1px solid rgba(52, 211, 153, 0.25);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
}

.brand-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #34d399;
    background: rgba(6, 78, 59, 0.6);
    border: 1px solid rgba(52, 211, 153, 0.4);
    border-radius: 9999px;
    margin-bottom: 0.75rem;
}

.main-title {
    font-size: 2.75rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    background: linear-gradient(135deg, #ffffff 40%, #a7f3d0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0.2rem 0;
}

.sub-title {
    font-size: 1.15rem;
    color: #94a3b8;
    margin-top: 0.4rem;
    font-weight: 400;
}

/* Card styling */
.content-card {
    background: rgba(18, 28, 20, 0.75);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(52, 211, 153, 0.18);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.content-card:hover {
    border-color: rgba(52, 211, 153, 0.35);
}

.card-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #f1f5f9;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
}

/* Prediction badge */
.badge-fresh {
    display: inline-flex;
    align-items: center;
    padding: 0.5rem 1.2rem;
    border-radius: 12px;
    font-size: 1.3rem;
    font-weight: 700;
    color: #ecfdf5;
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
}

.badge-rotten {
    display: inline-flex;
    align-items: center;
    padding: 0.5rem 1.2rem;
    border-radius: 12px;
    font-size: 1.3rem;
    font-weight: 700;
    color: #fff1f2;
    background: linear-gradient(135deg, #e11d48 0%, #f43f5e 100%);
    box-shadow: 0 4px 15px rgba(225, 29, 72, 0.4);
}

/* Quality explanation callout */
.explanation-box {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 78, 59, 0.25) 100%);
    border-left: 4px solid #10b981;
    border-radius: 12px;
    padding: 1.3rem;
    margin: 1rem 0;
    font-size: 1.2rem;
    line-height: 1.6;
    color: #f8fafc;
    font-weight: 500;
}

/* VLM description box */
.vlm-box {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 12px;
    padding: 1.1rem;
    font-size: 0.98rem;
    line-height: 1.55;
    color: #cbd5e1;
    font-style: italic;
}

/* Cue chip */
.cue-chip {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
}
.cue-chip-absent {
    background: rgba(16, 185, 129, 0.15);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.cue-chip-present {
    background: rgba(239, 68, 68, 0.15);
    color: #fca5a5;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

/* Custom progress bar wrapper */
.prob-label-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 0.2rem;
}
.prob-bar-container {
    background: rgba(30, 41, 59, 0.7);
    border-radius: 6px;
    height: 10px;
    overflow: hidden;
    margin-bottom: 0.65rem;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.4s ease;
}

/* Privacy & Footer */
.footer-note {
    text-align: center;
    color: #64748b;
    font-size: 0.82rem;
    padding: 2rem 1rem 1rem 1rem;
    border-top: 1px solid rgba(52, 211, 153, 0.12);
    margin-top: 3rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Cached resources for Streamlit lifecycle
@st.cache_resource(show_spinner="Loading trained MobileNetV3Large checkpoint...")
def get_cached_cnn(checkpoint_path: str = str(DEFAULT_CHECKPOINT_PATH)):
    """Load and cache the trained MobileNetV3Large model."""
    device = get_device()
    model, class_names = load_cnn_model(checkpoint_path, device=device)
    return model, class_names, device

@st.cache_resource(show_spinner="Loading SmolVLM-500M model...")
def get_cached_vlm():
    """Load and cache the SmolVLM processor and model."""
    device = get_device()
    processor, model = load_vlm(device=device)
    return processor, model, device

def find_sample_images() -> dict[str, Path]:
    """Find representative images from the local test set for quick testing."""
    test_dir = PROJECT_ROOT / "data" / "processed" / "dataset" / "test"
    samples = {}
    if test_dir.is_dir():
        for category in CLASS_NAMES:
            cat_dir = test_dir / category
            if cat_dir.is_dir():
                images = list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.jpg"))
                if images:
                    display_label = CLASS_DISPLAY_NAMES.get(category, category)
                    samples[display_label] = images[0]
    return samples

# Header
st.markdown(
    """
    <div class="header-container">
        <div class="brand-badge">🌱 Agricultural Vision AI</div>
        <h1 class="main-title">FruitSenseAI</h1>
        <p class="sub-title">Understand your fruit's visible quality.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None
if "current_image_hash" not in st.session_state:
    st.session_state["current_image_hash"] = None
if "audio_bytes" not in st.session_state:
    st.session_state["audio_bytes"] = None
if "audio_language" not in st.session_state:
    st.session_state["audio_language"] = "en-IN"
if "spoken_text" not in st.session_state:
    st.session_state["spoken_text"] = None

# Two Column Layout: Left = Input/Preview, Right = Analysis Results
col_input, col_results = st.columns([1, 1.25], gap="large")

with col_input:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📸 Select or Upload Fruit Image</div>', unsafe_allow_html=True)

    # Sample images quick selector
    samples = find_sample_images()
    selected_sample = None
    if samples:
        sample_options = ["-- Or choose a sample test image --"] + list(samples.keys())
        chosen_sample_name = st.selectbox(
            "Quick Demo Samples",
            sample_options,
            index=0,
            help="Quickly evaluate sample images from the test set without needing to upload.",
        )
        if chosen_sample_name != sample_options[0]:
            selected_sample = samples[chosen_sample_name]

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload image (JPG, JPEG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload an image up to 15 MB.",
    )

    # Determine input source
    active_image = None
    image_source_desc = ""

    if uploaded_file is not None:
        try:
            active_image = validate_and_load_image(uploaded_file)
            image_source_desc = f"Uploaded: {uploaded_file.name}"
        except Exception as e:
            st.error(f"Image validation error: {e}")
    elif selected_sample is not None:
        try:
            active_image = validate_and_load_image(selected_sample)
            image_source_desc = f"Sample: {selected_sample.parent.name}/{selected_sample.name}"
        except Exception as e:
            st.error(f"Failed to load sample image: {e}")

    # Image Preview
    if active_image is not None:
        st.markdown(f"**Selected Image Preview** ({image_source_desc})")
        st.image(active_image, use_container_width=True)

        # Compute image hash to prevent re-running same inference
        img_bytes_buf = io.BytesIO()
        active_image.save(img_bytes_buf, format="PNG")
        img_hash = hashlib.sha256(img_bytes_buf.getvalue()).hexdigest()

        analyze_button = st.button(
            "🔍 Analyze Fruit Quality",
            type="primary",
            use_container_width=True,
        )

        if analyze_button:
            # Run the full pipeline
            with st.status("Analyzing fruit image...", expanded=True) as status:
                try:
                    # Stage 1: Load CNN & Classify
                    status.update(label="1/4 Running MobileNetV3Large classification...")
                    cnn_model, class_names, device = get_cached_cnn()
                    cnn_results = predict_fruit(
                        image=active_image,
                        model=cnn_model,
                        class_names=class_names,
                        device=device,
                    )

                    # Stage 2: Grad-CAM
                    status.update(label="2/4 Generating Grad-CAM attention heatmap...")
                    cam_array, gradcam_overlay = compute_gradcam(
                        model=cnn_model,
                        input_tensor=cnn_results["input_tensor"],
                        predicted_index=cnn_results["predicted_index"],
                        display_image=cnn_results["cropped_image"],
                    )

                    # Stage 3: SmolVLM Visual Description
                    status.update(label="3/4 Inspecting surface features with SmolVLM...")
                    vlm_description = generate_vlm_description(
                        image=active_image,
                        device=device,
                    )
                    cue_analysis = analyze_vlm_cues(
                        vlm_description=vlm_description,
                        cnn_predicted_class=cnn_results["predicted_class"],
                    )

                    # Assemble evidence
                    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
                    evidence = build_evidence_object(
                        cnn_results=cnn_results,
                        vlm_description=vlm_description,
                        cue_analysis=cue_analysis,
                    )

                    # Save artifacts to outputs/web_runs/<run_id>/
                    artifact_paths = save_run_artifacts(
                        run_id=run_id,
                        original_image=active_image,
                        gradcam_overlay=gradcam_overlay,
                        evidence=evidence,
                    )

                    # Stage 4: Gemini Explanation
                    status.update(label="4/4 Generating shopper-friendly explanation with Gemini...")
                    explanation_text = ""
                    explanation_source = "Gemini"
                    try:
                        explanation_text = generate_gemini_explanation(
                            evidence=evidence,
                            model_name=DEFAULT_GEMINI_MODEL,
                        )
                    except Exception as gemini_err:
                        # Graceful fallback to rule-based explanation
                        explanation_text = generate_fallback_explanation(evidence)
                        explanation_source = f"Fallback (Gemini error: {gemini_err})"

                    status.update(label="Analysis complete!", state="complete")

                    # Store results in session state
                    st.session_state["analysis_result"] = {
                        "run_id": run_id,
                        "cnn_results": cnn_results,
                        "gradcam_overlay": gradcam_overlay,
                        "vlm_description": vlm_description,
                        "cue_analysis": cue_analysis,
                        "evidence": evidence,
                        "explanation": explanation_text,
                        "explanation_source": explanation_source,
                        "artifact_paths": artifact_paths,
                    }
                    st.session_state["current_image_hash"] = img_hash
                    # Reset audio for new image
                    st.session_state["audio_bytes"] = None
                    st.session_state["spoken_text"] = None

                except Exception as ex:
                    status.update(label=f"Analysis failed: {ex}", state="error")
                    st.error(f"Error during analysis: {ex}")
    else:
        st.info("👆 Please upload an image or select a sample demo image above to begin.")

    st.markdown("</div>", unsafe_allow_html=True)

# Results Panel
with col_results:
    result = st.session_state.get("analysis_result")

    if result is not None:
        cnn_res = result["cnn_results"]
        cue_res = result["cue_analysis"]
        is_fresh = "fresh" in cnn_res["predicted_class"].lower()
        badge_class = "badge-fresh" if is_fresh else "badge-rotten"
        badge_icon = "✨" if is_fresh else "⚠️"

        # Classification Overview Card
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🏷️ Classification Result</div>', unsafe_allow_html=True)

        res_col1, res_col2 = st.columns([1.2, 1])
        with res_col1:
            st.markdown(
                f'<div class="{badge_class}">{badge_icon} {cnn_res["predicted_label"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='color:#94a3b8; margin-top:0.6rem; font-size:0.95rem;'>"
                f"Predicted condition: <strong>{cnn_res['predicted_condition']}</strong>"
                f"</p>",
                unsafe_allow_html=True,
            )

        with res_col2:
            conf_val = cnn_res["confidence"]
            st.metric(
                label="Model Confidence",
                value=f"{conf_val:.1%}",
                help="Model confidence indicates CNN classification certainty for this image. It is not an absolute guarantee of safety or freshness.",
            )
            st.caption("ℹ️ Confidence reflects visual feature matching, not internal testing.")

        st.markdown("---")

        # Compact Horizontal Probability Distribution
        st.markdown("**Class Probability Distribution**")
        sorted_probs = sorted(
            cnn_res["class_probabilities"].items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for class_name, prob in sorted_probs:
            display_name = CLASS_DISPLAY_NAMES.get(class_name, class_name)
            is_winner = class_name == cnn_res["predicted_class"]
            bar_color = "#10b981" if "fresh" in class_name else "#f43f5e"
            opacity = "1.0" if is_winner else "0.5"

            st.markdown(
                f"""
                <div class="prob-label-row">
                    <span style="color: {'#ffffff' if is_winner else '#94a3b8'};">
                        {'👉 ' if is_winner else ''}{display_name}
                    </span>
                    <span style="color: {'#34d399' if is_winner else '#64748b'};">{prob:.2%}</span>
                </div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill" style="width: {max(prob * 100, 1.5)}%; background-color: {bar_color}; opacity: {opacity};"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Visual Evidence Card (Grad-CAM & VLM)
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔬 Visual Interpretability & Evidence</div>', unsafe_allow_html=True)

        cam_col1, cam_col2 = st.columns(2)
        with cam_col1:
            st.markdown("**CNN Input Crop**")
            st.image(cnn_res["cropped_image"], use_container_width=True)
        with cam_col2:
            st.markdown("**Grad-CAM Attention Overlay**")
            st.image(result["gradcam_overlay"], use_container_width=True)

        st.caption(
            "📌 **Grad-CAM Interpretation:** Highlighted regions indicate features that most heavily influenced "
            "the MobileNetV3Large prediction. It visualizes CNN attention; it does not prove physical spoilage."
        )

        st.markdown("**SmolVLM Visual Appearance Description:**")
        st.markdown(
            f'<div class="vlm-box">"{result["vlm_description"]}"</div>',
            unsafe_allow_html=True,
        )

        # Cue status chips
        cue_chips = []
        for cue, status in cue_res.get("visual_cue_status_heuristic", {}).items():
            if status is True:
                cue_chips.append(f'<span class="cue-chip cue-chip-present">⚠️ {cue.replace("_", " ").title()} Noted</span>')
            elif status is False:
                cue_chips.append(f'<span class="cue-chip cue-chip-absent">✓ No {cue.replace("_", " ").title()}</span>')

        if cue_chips:
            st.markdown(
                '<div style="margin-top:0.8rem;">' + "".join(cue_chips) + "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Quality Explanation & Voice Section
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">💬 Fruit Quality Explanation</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="explanation-box">{result["explanation"]}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown('<div class="card-title" style="font-size:1.1rem;">🔊 Spoken Explanation (Sarvam AI)</div>', unsafe_allow_html=True)

        voice_col1, voice_col2 = st.columns([1, 1])
        with voice_col1:
            selected_lang = st.selectbox(
                "Select Voice Language",
                list(SUPPORTED_LANGUAGES.keys()),
                format_func=lambda code: SUPPORTED_LANGUAGES[code],
                index=list(SUPPORTED_LANGUAGES.keys()).index(st.session_state["audio_language"]),
            )
            st.session_state["audio_language"] = selected_lang

        with voice_col2:
            st.write("") # spacing
            st.write("")
            generate_audio_btn = st.button("🎙️ Generate Spoken Output", use_container_width=True)

        if generate_audio_btn:
            with st.spinner(f"Preparing speech in {SUPPORTED_LANGUAGES[selected_lang]}..."):
                try:
                    # Translate to target language script if needed (Tamil, Hindi, etc.)
                    spoken_text = translate_explanation(
                        text=result["explanation"],
                        target_language_code=selected_lang,
                    )
                    st.session_state["spoken_text"] = spoken_text

                    run_dir = Path(result["artifact_paths"]["original_image_path"]).parent
                    audio_out_path = run_dir / f"speech_{selected_lang}.wav"
                    audio_data = generate_speech(
                        text=spoken_text,
                        language_code=selected_lang,
                        output_path=audio_out_path,
                    )
                    st.session_state["audio_bytes"] = audio_data
                    st.session_state["audio_language"] = selected_lang
                    st.success(f"Generated {SUPPORTED_LANGUAGES[selected_lang]} speech successfully!")
                except Exception as audio_err:
                    st.warning(f"Voice generation unavailable: {audio_err}")

        # Audio Player and Spoken Script if audio has been generated
        if st.session_state.get("audio_bytes"):
            if st.session_state.get("spoken_text") and st.session_state.get("audio_language") != "en-IN":
                lang_title = SUPPORTED_LANGUAGES.get(st.session_state["audio_language"], "Localized")
                st.markdown(f"**Spoken Script ({lang_title}):**")
                st.markdown(
                    f'<div class="vlm-box" style="font-style:normal; font-size:1.05rem; color:#ecfdf5; margin-bottom:0.75rem;">'
                    f'{st.session_state["spoken_text"]}</div>',
                    unsafe_allow_html=True,
                )
            st.audio(st.session_state["audio_bytes"], format="audio/wav")

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Placeholder card before analysis
        st.markdown('<div class="content-card" style="text-align:center; padding: 3rem 1.5rem;">', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size: 3.5rem; margin-bottom: 1rem;">🍎 🍌 🍊</div>
            <h3 style="color: #f1f5f9; font-weight:700;">Ready for Analysis</h3>
            <p style="color: #94a3b8; max-width: 440px; margin: 0 auto; line-height: 1.6;">
                Upload a photo or choose a demo sample fruit to examine visible surface characteristics,
                CNN attention heatmaps, and AI-generated shopper explanations with voice output.
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

# Privacy & Responsible AI Footer
st.markdown(
    """
    <div class="footer-note">
        🔒 <strong>Privacy & Responsible AI:</strong> Uploaded images are processed locally by your on-device CNN and VLM models.
        External AI APIs receive only the necessary minimal textual evidence (Gemini receives structured text evidence;
        Sarvam AI receives the short text explanation for speech synthesis).
        <br><br>
        ⚠️ <strong>Food Safety Notice:</strong> FruitSenseAI evaluates visible surface characteristics from photographic evidence.
        Image-based estimates do not detect internal rot, bacteria, chemical contamination, or certify food safety.
        Always inspect fruits thoroughly before consumption.
    </div>
    """,
    unsafe_allow_html=True,
)
