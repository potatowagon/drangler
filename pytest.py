import unittest
import Sampler
import numpy

class TestSampler(unittest.TestCase):
    num_of_signals = 6
    frame_length = 5
    sampling_interval = 2

    # create dummy bytestring set
    bytestring_length = 10
    bytestring_set = numpy.zeros((0, bytestring_length))
    for i in range(0, 6):
        bytestring = numpy.array(list(range(0, bytestring_length)))
        bytestring_set = numpy.append(bytestring_set, [bytestring], axis = 0)

    print(bytestring_set)

    # initialise Sampler
    sampler = Sampler.Sampler(num_of_signals, frame_length, sampling_interval)

    def test_sample(self):
        
        frame_collection = self.sampler.sample(self.bytestring_set)
        print(frame_collection)
        self.assertEqual(frame_collection.shape[0], 3)
        self.assertEqual(self.sampler.leftover_bytestring_set.shape[1], 4)

    def test_subsequent_sample(self):
        frame_collection = self.sampler.sample(self.bytestring_set)
        print(frame_collection)
        self.assertEqual(frame_collection.shape[0], 5)
        self.assertEqual(self.sampler.leftover_bytestring_set.shape[1], 4)

if __name__ == '__main__':
    unittest.main()

