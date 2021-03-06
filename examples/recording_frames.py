from drangler import Sampler
import numpy

''' Example code for recording and saving frames.  
You are now recording a specific dance move.
The pi is recieving Bytesrtring sets for the dance move.
'''

# initialise Sampler
num_of_signals = 6
frame_length = 5
sampling_interval = 2

sampler = Sampler.Sampler(num_of_signals, frame_length, sampling_interval)

frame_collection = []
# while there are incoming bytestring sets (numpy array)
while(bytestring_set_received):
    frame_collection.extend(sampler.sample(bytestring_set))

# When it is time to stop recording, 
# convert frame_collection to numpy
frame_collection = numpy.array(frame_collection)

# save frame collection as .npy
numpy.save("frames_name", frame_collection)

# create labels
move_number = 0 #chicken
labels = numpy.full(frame_collection.shape[0], move_number)

# save labels as .npy
numpy.save("label_name", labels)

