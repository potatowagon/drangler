from drangler import FeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
import numpy
import pickle

''' Example code for extracting features and training model
'''

#load frames and labels
frames_collection = numpy.load("frames_name.npy")
labels = numpy.load("label_name.npy")

# Extract features from frames to create training data
training_data_collection = FeatureExtractor.extract(frames_collection)

# data preprocessing
training_data_collection = normalize(training_data_collection)

# train model
model = RandomForestClassifier() # using default
model.fit(training_data_collection, labels)

# save model
pickle.dump(model, open("model_name.sav", 'wb'))
