from drangler.FeatureExtractor import get_features_from_frame
from time import  time
import numpy as np
from sklearn.externals import joblib

# trained_model = load("trained_model_svm.sav")
trained_model = joblib.load("trained_model_rf.sav")


def predict(data):
    start = time()
    data = get_features_from_frame(data)
    data = np.array(data).reshape(1, -1)
    print(trained_model.predict(data)[0])
    print(f"Time taken: {time() - start}s")