"""Train the two models used by the live demo and save them to models/digit_svm.joblib.

- "plain": MFCC features without speaker normalisation (works on a single clip).
- "norm":  MFCC features z-scored per speaker; the demo normalises with the user's own
           calibration clips (one recording of each digit).

Usage: python src/export_model.py --data data
"""
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from features import mfcc_feats, spk_norm
from model import SVM_PARAMS, make_svm

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
a = ap.parse_args()

train = pd.read_csv(os.path.join(a.data, "partA_train.csv"))
M = np.array([mfcc_feats(c, a.data) for c in train.clip_id])
y = train.y.to_numpy()


def prob_svm():
    # same SVM, plus sigmoid calibration so the demo can show probabilities
    return CalibratedClassifierCV(make_svm(SVM_PARAMS), ensemble=False)


models = {"plain": prob_svm().fit(M, y), "norm": prob_svm().fit(spk_norm(M, train), y)}
os.makedirs("models", exist_ok=True)
joblib.dump(models, "models/digit_svm.joblib", compress=3)
print("saved models/digit_svm.joblib")
