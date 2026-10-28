"""Live demo: say a digit into the microphone and the SVM recognises it.

Optional calibration: record each digit once, so the features can be normalised to your voice
(the same per-speaker normalisation the model was trained with).

Usage: python app.py   (needs models/digit_svm.joblib, made by src/export_model.py)
"""
import os
import sys
import gradio as gr
import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from features import mfcc_from_signal

MODELS = joblib.load(os.path.join(HERE, "models", "digit_svm.joblib"))
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def feats(audio):
    sr, x = audio
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)                 # stereo -> mono
    if np.abs(x).max() > 1:
        x = x / 32768.0                    # int16 -> [-1, 1]
    return mfcc_from_signal(x, sr)


def calibrate(*clips):
    if any(c is None for c in clips):
        return None, "Record all 10 digits first."
    F = np.array([feats(c) for c in clips])
    return {"mu": F.mean(axis=0), "sd": F.std(axis=0) + 1e-9}, "✅ Calibrated to your voice."


def predict(audio, cal):
    if audio is None:
        return None, "Record a digit first."
    f = feats(audio)
    if cal is None:
        m, x, mode = MODELS["plain"], f, "not calibrated (calibrate above for better accuracy)"
    else:
        m, x, mode = MODELS["norm"], (f - cal["mu"]) / cal["sd"], "calibrated to your voice"
    p = m.predict_proba(x[None])[0]
    return {f"{c} ({WORDS[c]})": float(v) for c, v in zip(m.classes_, p)}, "Mode: " + mode


with gr.Blocks(title="Spoken digit recogniser") as demo:
    gr.Markdown("# 🎙️ Spoken digit recogniser\n"
                "Say a single digit (0–9) in English. Features: MFCCs computed from scratch, "
                "model: RBF SVM trained on 4 speakers.")
    cal = gr.State(None)
    with gr.Accordion("Step 1 (optional): calibrate to your voice – say each digit once", open=False):
        with gr.Row():
            mics = [gr.Audio(sources=["microphone"], type="numpy", label=f"{d}: “{w}”")
                    for d, w in enumerate(WORDS[:5])]
        with gr.Row():
            mics += [gr.Audio(sources=["microphone"], type="numpy", label=f"{d}: “{w}”")
                     for d, w in enumerate(WORDS[5:], start=5)]
        cal_btn = gr.Button("Save calibration")
        cal_msg = gr.Markdown()
    gr.Markdown("### Step 2: say a digit")
    with gr.Row():
        inp = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Your digit")
        with gr.Column():
            out = gr.Label(num_top_classes=3, label="Prediction")
            mode = gr.Markdown()
    btn = gr.Button("Recognise", variant="primary")

    cal_btn.click(calibrate, mics, [cal, cal_msg])
    btn.click(predict, [inp, cal], [out, mode])
    inp.stop_recording(predict, [inp, cal], [out, mode])

if __name__ == "__main__":
    demo.launch()
