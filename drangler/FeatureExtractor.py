import numpy

def extract(frame_collection):
    data_collection = []
    for i in range(0,frame_collection.shape[0]):
        frame = frame_collection[i]
        data = []
        for j in range(0, frame_collection.shape[1]):
            signal = frame[j]
            # Adjust fetures here
            data.append(mean(signal))
            data.append(median(signal))
            data.append(variance(signal))

        data_collection.append(numpy.array(data))
    return numpy.array(data_collection)


# Time domain
def mean(signal):
    return numpy.mean(signal)

def median(signal):
    return numpy.median(signal)

def variance(signal):
    return numpy.var(signal)





