import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROJECT_ROOT, DEFAULT_GEMINI_MODEL
from src.inference import validate_and_load_image, load_cnn_model, predict_fruit
from src.visual_analysis import (
    compute_gradcam,
    generate_vlm_description,
    analyze_vlm_cues,
    build_evidence_object,
    save_run_artifacts,
)
from src.gemini_explainer import generate_gemini_explanation, generate_fallback_explanation
from src.speech_service import generate_speech

def test_full_pipeline():
    print("--- 1. Testing Image Validation and CNN Loading ---")
    test_img_path = PROJECT_ROOT / "data" / "processed" / "dataset" / "test" / "freshapples" / "rotated_by_15_Screen Shot 2018-06-08 at 4.59.49 PM.png"
    assert test_img_path.is_file(), f"Test image missing: {test_img_path}"

    image = validate_and_load_image(test_img_path)
    model, class_names = load_cnn_model()
    cnn_results = predict_fruit(image, model, class_names)
    print(f"Prediction: {cnn_results['predicted_label']} ({cnn_results['predicted_class']})")
    print(f"Confidence: {cnn_results['confidence']:.2%}")
    assert cnn_results['predicted_class'] == "freshapples"
    assert len(cnn_results['class_probabilities']) == 6

    print("\n--- 2. Testing Grad-CAM Generation ---")
    cam, overlay = compute_gradcam(
        model=model,
        input_tensor=cnn_results['input_tensor'],
        predicted_index=cnn_results['predicted_index'],
        display_image=cnn_results['cropped_image'],
    )
    assert cam.shape == (224, 224)
    assert overlay.size == (224, 224)
    print(f"Grad-CAM generated successfully (overlay size {overlay.size})")

    print("\n--- 3. Testing SmolVLM & Cue Heuristics ---")
    vlm_desc = generate_vlm_description(image)
    print(f"VLM Description: {vlm_desc}")
    assert len(vlm_desc) > 0

    cues = analyze_vlm_cues(vlm_desc, cnn_results['predicted_class'])
    print(f"Review status: {cues['review_status_heuristic']}")
    print(f"Cues analysis: {cues['visual_cue_status_heuristic']}")

    print("\n--- 4. Testing Evidence Construction & Artifact Saving ---")
    evidence = build_evidence_object(
        cnn_results=cnn_results,
        vlm_description=vlm_desc,
        cue_analysis=cues,
    )
    artifacts = save_run_artifacts(
        run_id="test_run",
        original_image=image,
        gradcam_overlay=overlay,
        evidence=evidence,
    )
    assert Path(artifacts['evidence_json_path']).is_file()
    assert Path(artifacts['gradcam_overlay_path']).is_file()
    print(f"Artifacts saved in {Path(artifacts['evidence_json_path']).parent}")

    print("\n--- 5. Testing Gemini Explanation & Fallback ---")
    fallback_exp = generate_fallback_explanation(evidence)
    print(f"Rule-based fallback explanation: {fallback_exp}")
    assert len(fallback_exp) > 0

    gemini_exp = generate_gemini_explanation(evidence, model_name=DEFAULT_GEMINI_MODEL)
    print(f"Gemini explanation: {gemini_exp}")
    assert len(gemini_exp) > 0

    print("\n--- 6. Testing Sarvam AI Speech Synthesis ---")
    wav_en = generate_speech(gemini_exp, language_code="en-IN")
    assert len(wav_en) > 1000
    print(f"English audio bytes: {len(wav_en)}")

    from src.gemini_explainer import translate_explanation
    tamil_script = translate_explanation(gemini_exp, "ta-IN")
    assert len(tamil_script) > 0 and tamil_script != gemini_exp
    print("Tamil translation generated successfully for speech synthesis")

    wav_ta = generate_speech(tamil_script, language_code="ta-IN")
    assert len(wav_ta) > 1000
    print(f"Tamil audio bytes: {len(wav_ta)}")

    print("\n==========================================")
    print("ALL 6 PIPELINE STAGES PASSED SUCCESSFULLY!")
    print("==========================================")

if __name__ == "__main__":
    test_full_pipeline()
