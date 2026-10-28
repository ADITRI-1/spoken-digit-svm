"""Train the final model and predict digits for a new file of clips (columns: clip_id, optional g).

Usage: python src/predict.py --data data --test data/partA_val.csv --out predictions.csv
"""
import argparse
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from features import make_features
from model import SVM_PARAMS, make_svm

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--test", default="data/partA_val.csv")
ap.add_argument("--out", default="predictions.csv")
a = ap.parse_args()

train = pd.read_csv(os.path.join(a.data, "partA_train.csv"))
test = pd.read_csv(a.test)
model = make_svm(SVM_PARAMS).fit(make_features(train, a.data), train["y"])
pred = model.predict(make_features(test, a.data))
pd.DataFrame({"clip_id": test["clip_id"], "y_pred": pred}).to_csv(a.out, index=False)
print("saved", len(pred), "predictions to", a.out)
if "y" in test:
    print("accuracy:", round(accuracy_score(test["y"], pred), 3))
