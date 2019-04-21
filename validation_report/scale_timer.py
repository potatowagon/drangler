from timeit import default_timer as timer
import numpy
from sklearn import preprocessing

data = numpy.load('./training_data_all.npy')

labels = data[:,data.shape[1]-1]
labels = numpy.array(labels, dtype=numpy.int32)
print(labels)

unique, counts = numpy.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

data = numpy.delete(data, -1, axis=1)
print(data.shape)

one_sample = data[0]
start = timer()
one_sample = preprocessing.scale(one_sample)
end = timer()
print(end - start)
