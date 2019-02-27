import unittest
import numpy
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from validation import ConfusionMatrix
from validation import LOOCV
from validation import KFold

class TestModel(unittest.TestCase):
    
    #training_data = normalize(pandas.read_csv(".\\test_data\\x_train.csv").values)
    #training_labels = pandas.read_csv(".\\test_data\\y_train.csv").values
    data = normalize(pandas.read_csv(".\\test_data\\x_test.csv").values)
    labels = pandas.read_csv(".\\test_data\\y_test.csv").values
    labels = numpy.reshape(labels, (labels.shape[0],))
    print(data.shape, labels.shape)
    
    training_data_percentage = 0.8
    seperator = int(len(data)*training_data_percentage)
    
    training_data = data[0:seperator]
    test_data = data[seperator:len(data)]
    print(training_data.shape, test_data.shape)

    training_labels = labels[0:seperator]
    test_labels = labels[seperator:len(labels)]
    print(training_labels.shape, test_labels.shape)

    def test_sample(self):
        model = RandomForestClassifier() # using default
        model.fit(self.training_data, self.training_labels)
        predicted_labels = model.predict(self.test_data)
        ConfusionMatrix.print_confusion_matrix(self.test_labels, predicted_labels)
        KFold.get_k_fold_accuracy(model, 5, self.data, self.labels)
        LOOCV.get_loocv_accuracy(model, self.data, self.labels)
        
if __name__ == '__main__':
    unittest.main()

