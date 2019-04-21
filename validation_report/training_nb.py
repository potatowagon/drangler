from sklearn.svm import SVC
import pandas
import numpy
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def get_k_fold_accuracy(model, folds, data, labels):
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import KFold
    print(f"\nCalculating KFold accuracy with {folds} iterations...")
    kf = KFold(folds, shuffle=True)
    total_accuracy = 0
    for training_indices, testing_indices in kf.split(data):
        model.fit(data[training_indices], labels[training_indices])
        y_predicted = model.predict(data[testing_indices])
        total_accuracy += accuracy_score(labels[testing_indices], y_predicted)
    return total_accuracy/folds


data = numpy.load('./training_data_all.npy')
#pandas.DataFrame(data).to_csv("./data.csv")

labels = data[:,data.shape[1]-1]
labels = numpy.array(labels, dtype=numpy.int32)
print(labels)

unique, counts = numpy.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

data = numpy.delete(data, -1, axis=1)
print(data.shape)
data = preprocessing.scale(data)


def naive_bayes(classifier, data, labels):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1)
    print(f"\nTraining {classifier} NB classifier... ")
    if classifier == "gaussian":
        from sklearn.naive_bayes import GaussianNB
        nb = GaussianNB()
    
    elif classifier == "multinomial":
        from sklearn.naive_bayes import MultinomialNB
        nb = MultinomialNB()
    elif classifier == "complement":
        from sklearn.naive_bayes import ComplementNB
        nb = ComplementNB()
    elif classifier == "bernoulli":
        from sklearn.naive_bayes import BernoulliNB
        nb = BernoulliNB()
    else:
        return
    return get_k_fold_accuracy(nb, 10, data, labels)


#print(naive_bayes("multinomial", data, labels))
#print(naive_bayes("complement", data, labels))
print(naive_bayes("bernoulli", data, labels))
print(naive_bayes("gaussian", data, labels))


    