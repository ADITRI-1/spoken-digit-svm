"""All experiments behind the README: EDA, validation strategy, C/gamma search, error analysis, feature ablation.

Usage: python src/experiments.py --data data
Figures go to results/, key numbers to results/metrics.json.
"""
import argparse
import json
import os
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, GridSearchCV, cross_val_score, cross_val_predict
from features import audio_stats, load_audio, mfcc_feats, spk_norm, make_features
from model import SEED, SVM_PARAMS, make_svm

warnings.filterwarnings("ignore")
OUT = "results"


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=130)
    plt.close()


def main(data):
    os.makedirs(OUT, exist_ok=True)
    train = pd.read_csv(os.path.join(data, "partA_train.csv"))
    val = pd.read_csv(os.path.join(data, "partA_val.csv"))
    F = [c for c in train.columns if c.startswith("f")]
    X, y, g = train[F].to_numpy(), train.y.to_numpy(), train.g.to_numpy()
    Xv, yv = val[F].to_numpy(), val.y.to_numpy()
    logo = LeaveOneGroupOut()
    res = {}

    # 1. EDA: feature scales and recording level per speaker
    d = train[F].describe().T
    print("largest std / smallest std:", round(d["std"].max() / d["std"].min(), 1))
    plt.figure(figsize=(12, 3))
    plt.bar(F, d["std"])
    plt.yscale("log")
    plt.xticks(rotation=90, fontsize=7)
    plt.title("Std of each provided feature (log scale)")
    save("feature_scales.png")

    both = pd.concat([train, val], ignore_index=True)
    A = audio_stats(both, data)
    A["g"] = both.g
    print(A.groupby("g")[["dur", "peak", "rms_db", "dc"]].median().round(3))
    A.boxplot(column="rms_db", by="g", figsize=(6, 3))
    plt.title("Clip level (RMS dB) per speaker")
    plt.suptitle("")
    save("level_per_speaker.png")

    threes = both[both.y == 3].groupby("g").first()
    fig, ax = plt.subplots(1, len(threes), figsize=(15, 2.5), sharey=True)
    for i, (s, row) in enumerate(threes.iterrows()):
        sr, x = load_audio(row.clip_id, data)
        ax[i].plot(np.arange(len(x)) / sr, x)
        ax[i].set_title(s + ": digit 3")
    save("waveforms.png")

    # 2. validation strategy: same SVM, three estimates
    B = dict(kernel="rbf", C=1.0, gamma="scale")
    skf = cross_val_score(make_svm(B), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED)).mean()
    loso = cross_val_score(make_svm(B), X, y, cv=logo, groups=g).mean()
    va = accuracy_score(yv, make_svm(B).fit(X, y).predict(Xv))
    res["validation"] = {"stratified_kfold": round(float(skf), 3), "leave_one_speaker_out": round(float(loso), 3), "unseen_val_speaker": round(float(va), 3)}
    print(res["validation"])

    # 3. scaling and C / gamma grid (provided features)
    for s in [False, True]:
        m = make_svm(B, scale=s)
        a = cross_val_score(m, X, y, cv=logo, groups=g).mean()
        print("scaled =", s, "| LOSO:", round(a, 3), "| val:", round(accuracy_score(yv, m.fit(X, y).predict(Xv)), 3))
    Cs = [0.1, 0.3, 1, 3, 10, 30, 100, 300]
    Gs = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    gs = GridSearchCV(make_svm(dict(kernel="rbf")), {"svc__C": Cs, "svc__gamma": Gs}, cv=logo, n_jobs=-1)
    gs.fit(X, y, groups=g)
    H = gs.cv_results_["mean_test_score"].reshape(len(Cs), len(Gs))
    plt.figure(figsize=(6, 5))
    plt.imshow(H)
    for i in range(len(Cs)):
        for j in range(len(Gs)):
            plt.text(j, i, round(H[i, j], 2), ha="center", va="center", color="w", fontsize=8)
    plt.xticks(range(len(Gs)), Gs, rotation=45)
    plt.yticks(range(len(Cs)), Cs)
    plt.xlabel("gamma")
    plt.ylabel("C")
    plt.title("Leave-one-speaker-out accuracy")
    plt.colorbar()
    save("c_gamma_heatmap.png")
    D = dict(kernel="rbf", C=gs.best_params_["svc__C"], gamma=gs.best_params_["svc__gamma"])
    print("best on provided features:", D, round(gs.best_score_, 3))

    # 4. error analysis on the unseen speaker
    m = make_svm(D)
    vp = m.fit(X, y).predict(Xv)
    ConfusionMatrixDisplay.from_predictions(yv, vp)
    plt.title("Unseen speaker, provided features")
    save("confusion_baseline.png")
    lp = cross_val_predict(m, X, y, cv=logo, groups=g)
    print("LOSO accuracy per speaker:", pd.Series(lp == y).groupby(g).mean().round(3).to_dict())

    # 5. feature ablation
    M = np.array([mfcc_feats(c, data) for c in train.clip_id])
    Mv = np.array([mfcc_feats(c, data) for c in val.clip_id])
    Xf, Xvf = make_features(train, data), make_features(val, data)
    tests = [("provided features", X, Xv, D),
             ("provided + speaker norm", spk_norm(X, train), spk_norm(Xv, val), D),
             ("own MFCC (VAD, level norm)", M, Mv, SVM_PARAMS),
             ("own MFCC + speaker norm", Xf, Xvf, SVM_PARAMS)]
    res["ablation"] = {}
    for name, a, b, p in tests:
        s = cross_val_score(make_svm(p), a, y, cv=logo, groups=g)
        v = accuracy_score(yv, make_svm(p).fit(a, y).predict(b))
        res["ablation"][name] = {"loso": round(float(s.mean()), 3), "worst_speaker": round(float(s.min()), 3), "val": round(float(v), 3)}
        print(name, res["ablation"][name])

    names = list(res["ablation"])
    pos = np.arange(len(names))
    plt.figure(figsize=(8, 3.5))
    plt.bar(pos - 0.2, [res["ablation"][n]["loso"] for n in names], 0.4, label="leave-one-speaker-out")
    plt.bar(pos + 0.2, [res["ablation"][n]["val"] for n in names], 0.4, label="unseen speaker")
    plt.xticks(pos, names, rotation=15, fontsize=8)
    plt.ylabel("accuracy")
    plt.ylim(0, 1)
    plt.legend()
    save("ablation.png")

    vf = make_svm(SVM_PARAMS).fit(Xf, y).predict(Xvf)
    ConfusionMatrixDisplay.from_predictions(yv, vf)
    plt.title("Unseen speaker, final features")
    save("confusion_final.png")

    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    main(ap.parse_args().data)
