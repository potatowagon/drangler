from sklearn.svm import SVC
import pandas
import numpy
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import GridSearchCV

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

def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_, grid_search

data = numpy.load('./training_data_10.npy')
#pandas.DataFrame(data).to_csv("./data.csv")

labels = data[:,data.shape[1]-1]
labels = numpy.array(labels, dtype=numpy.int32)
print(labels)

unique, counts = numpy.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

data = numpy.delete(data, -1, axis=1)
print(data.shape)
data = preprocessing.scale(data)


svc_classifier = SVC()
acc = get_k_fold_accuracy(svc_classifier, 10, data, labels)
print(acc)

'''
params, model = svc_param_selection(data, labels, 10)
acc = get_k_fold_accuracy(model, 10, data, labels)
print(acc)
'''