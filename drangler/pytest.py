import unittest
import Sampler
import numpy
import FeatureExtractor


class TestSampler(unittest.TestCase):
    num_of_signals = 6
    frame_length = 5
    sampling_interval = 2

    # create dummy bytestring set
    bytestring_length = 10
    dummy_bytestring_set = numpy.zeros((0, bytestring_length))
    for i in range(0, 6):
        bytestring = numpy.array(list(range(0, bytestring_length)))
        dummy_bytestring_set = numpy.append(dummy_bytestring_set, [bytestring], axis = 0)

    print("dummy bytestring set: \n", dummy_bytestring_set)

    # create dummy frame collection
    dummy_frame_collection = numpy.array(
    [[[0, 1, 2, 3, 4],
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, 4]],

    [[2, 3, 4, 5, 6],
    [2, 3, 4, 5, 6],
    [2, 3, 4, 5, 6],
    [2, 3, 4, 5, 6],
    [2, 3, 4, 5, 6],
    [2, 3, 4, 5, 6]],

    [[4, 5, 6, 7, 8],
    [4, 5, 6, 7, 8],
    [4, 5, 6, 7, 8],
    [4, 5, 6, 7, 8],
    [4, 5, 6, 7, 8],
    [4, 5, 6, 7, 8]]])


    # initialise Sampler
    sampler = Sampler.Sampler(num_of_signals, frame_length, sampling_interval)

    def test_sample(self):
        print("\nTest Sample")
        frame_collection = self.sampler.sample(self.dummy_bytestring_set)
        print("frame_collection: \n", frame_collection)
        self.assertEqual(frame_collection.shape[0], 3)
        self.assertEqual(self.sampler.leftover_bytestring_set.shape[1], 4)

    def test_subsequent_sample(self):
        print("\nTest Subsequent Sample")
        frame_collection = self.sampler.sample(self.dummy_bytestring_set)
        print("frame_collection: \n", frame_collection)
        self.assertEqual(frame_collection.shape[0], 5)
        self.assertEqual(self.sampler.leftover_bytestring_set.shape[1], 4)

    def test_feature_extractor(self):
        print("\nTest Feature Extractor")
        data_collection = FeatureExtractor.extract(self.dummy_frame_collection)
        print("data_collection: \n", data_collection)

        num_features_per_signal = 3
        # check number of features in a data is correct
        self.assertEqual(data_collection.shape[1], num_features_per_signal * self.num_of_signals)  
        
        # check number of data in a collection returned is correct
        self.assertEqual(data_collection.shape[0], self.dummy_frame_collection.shape[0]) 

        # check feature placement is correct
        mean_feature_index = 0
        for i in range(0, self.dummy_frame_collection.shape[0]):
            self.assertEqual(data_collection[i][mean_feature_index], numpy.mean(self.dummy_frame_collection[i][0]))
        

if __name__ == '__main__':
    unittest.main()

