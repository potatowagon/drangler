import pickle
import drangler

''' Example code for demo
'''

# load model
model = pickle.load(open("model_name", 'rb'))

# initialise Sampler
num_of_signals = 6
frame_length = 5
sampling_interval = 2

sampler = drangler.Sampler.Sampler(num_of_signals, frame_length, sampling_interval)

# while there are incoming bytestring sets (numpy array)
while(bytestring_set_recieved):
    # Extract frames from bytestring set
    # return_as_numpy=True to return frame_collection as numpy array
    frame_collection = sampler.sample(bytestring_set, return_as_numpy=True)
    
    #Extract features from frames
    data_collection = drangler.FeatureExtractor.extract(frame_collection)
    
    # use trained model to predict
    result = model.predict(data_collection)
