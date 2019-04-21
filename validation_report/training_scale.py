from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy
import pandas
import pickle
from sklearn.externals import joblib

#validation stuff
'''
Inputs:
    model: Machine Learning Model
    folds: Number of folds (K)
    data: Training and testing data
    labels: Labels for data
Output:
    Average accuracy from the K Folds
'''
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

data = numpy.load('./training_data_delayed.npy')
pandas.DataFrame(data).to_csv("./data.csv")

labels = data[:,data.shape[1]-1]
labels = numpy.array(labels, dtype=numpy.int32)
print(labels)

unique, counts = numpy.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

data = numpy.delete(data, -1, axis=1)
print(data.shape)
data = preprocessing.scale(data)
pandas.DataFrame(data).to_csv("./data_n_scale.csv")

#optimise
#rf
accuracy = []
num_trees = []
best = 0
for i in range (1,31):
    rf_model = RandomForestClassifier(n_estimators=i, n_jobs=-1)
    acc = get_k_fold_accuracy(rf_model, 10, data, labels)
    if acc >= best:
        best = acc
        best_model = rf_model
    num_trees.append(i)
    accuracy.append(acc)
#pickle.dump(best_model, open('./out/rf_model.sav', 'wb'))
#joblib.dump(best_model, "./out_rf/rf_model.sav")
df = pandas.DataFrame.from_dict({
    'accuracy' : accuracy,
    'num_trees' : num_trees
})
df.to_csv('./out_rf/rf_trees_vs_acc.csv')

#knn
accuracy = []
num_neighbours = []
best = 0
for i in range (1,31):
    knn_model = KNeighborsClassifier(n_neighbors=i)
    acc = get_k_fold_accuracy(knn_model, 10, data, labels)
    if acc > best:
        best = acc
        best_model = knn_model
    num_neighbours.append(i)
    accuracy.append(acc)
#joblib.dump(best_model, "./out_knn/knn_model.sav")
df = pandas.DataFrame.from_dict({
    'accuracy' : accuracy,
    'num_neighbours' : num_neighbours
})
df.to_csv('./out_knn/neighbours_vs_acc.csv')

