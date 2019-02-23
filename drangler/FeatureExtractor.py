import numpy

def extract(frame_collection):
    data_collection = []
    for frame in frame_collection:
        data = []
        for signal in frame:
            # Adjust features here
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





