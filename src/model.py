"""Final model: standardised RBF SVM on the speaker-normalised MFCC features."""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 2024027
SVM_PARAMS = dict(kernel="rbf", C=1.0, gamma=0.001)     # chosen by leave-one-speaker-out search


def make_svm(params, scale=True):
    svm = SVC(**{**params, "random_state": SEED})
    return make_pipeline(StandardScaler(), svm) if scale else svm
