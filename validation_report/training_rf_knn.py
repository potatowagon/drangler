from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy
import pandas
import pickle
from sklearn.externals import joblib

#validation stuff
def get_k_fold_accuracy(model, folds, data, labels):
    '''
    Inputs:
        model: Machine Learning Model
        folds: Number of folds (K)
        data: Training and testing data
        labels: Labels for data
    Output:
        Average accuracy from the K Folds
    '''
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

#preprocessing stuff
def normalise_0_1(a):
    return a.astype('float') / a.sum(axis=1)[:, numpy.newaxis]

data = numpy.load('./training_data_10.npy')
pandas.DataFrame(data).to_csv("./data.csv")

def balence_data_set(data):
    '''
    shrinks data of all labels to match the label with the smallest data sample size
    '''
    e = data.shape[1] - 1
    data = data[data[:,e].argsort()]
    print(data)
    dataset_by_class = []

    for i in range(0,11):
        a = data[data[:, e] == i, :]
        if len(a) > 0:
            dataset_by_class.append(a)

    min_class_size = dataset_by_class[0].shape[0]
    for s in dataset_by_class:
        min_class_size = min(min_class_size, s.shape[0])

    resized_data = []
    for s in dataset_by_class:
        if s.shape[0] > min_class_size:
            data, test = train_test_split(s, train_size=min_class_size)
        resized_data.extend(data)
        print(len(resized_data))
    return resized_data

labels = data[:,data.shape[1]-1]
labels = numpy.array(labels, dtype=numpy.int32)
print(labels)

unique, counts = numpy.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

data = numpy.delete(data, -1, axis=1)
print(data.shape)

pandas.DataFrame(data).to_csv("./data_n.csv")
#optimise
#rf
accuracy = []
num_trees = []
best = 0
for i in range (1,31):
    rf_model = RandomForestClassifier(n_estimators=i)
    acc = get_k_fold_accuracy(rf_model, 10, data, labels)
    if acc >= best:
        best = acc
        best_model = rf_model
    num_trees.append(i)
    accuracy.append(acc)
#pickle.dump(best_model, open('./out/rf_model.sav', 'wb'))
joblib.dump(best_model, "./out_rf/rf_model.sav")
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
#pickle.dump(best_model, open('./out_knn/knn_model.sav', 'wb'))
joblib.dump(best_model, "./out_knn/knn_model.sav")
df = pandas.DataFrame.from_dict({
    'accuracy' : accuracy,
    'num_neighbours' : num_neighbours
})
df.to_csv('./out_knn/neighbours_vs_acc.csv')


'''
##NOTE: Normalising did not improve accuracy
# ==================== normalised 0-1 =============
#optimise
#rf
accuracy = []
num_trees = []
best = 0

data_n01 = normalize(data)
#rf
for i in range (1,51):
    rf_model = RandomForestClassifier(n_estimators=i)
    acc = get_k_fold_accuracy(rf_model, 10, data_n01, labels)
    if acc > best:
        best = acc
        best_model = rf_model
    num_trees.append(i)
    accuracy.append(acc)
#pickle.dump(best_model, open('./out/rf_model.sav', 'wb'))
joblib.dump(best_model, "./out_rf_n01/rf_model.sav")
df = pandas.DataFrame.from_dict({
    'accuracy' : accuracy,
    'num_trees' : num_trees
})
df.to_csv('./out_rf_n01/rf_trees_vs_acc.csv')

#knn
accuracy = []
num_neighbours = []
best = 0
for i in range (1,31):
    knn_model = KNeighborsClassifier(n_neighbors=i)
    acc = get_k_fold_accuracy(knn_model, 10, data_n01, labels)
    if acc > best:
        best = acc
        best_model = knn_model
    num_neighbours.append(i)
    accuracy.append(acc)
#pickle.dump(best_model, open('./out_knn/knn_model.sav', 'wb'))
joblib.dump(best_model, "./out_knn_n01/knn_model.sav")
df = pandas.DataFrame.from_dict({
    'accuracy' : accuracy,
    'num_neighbours' : num_neighbours
})
df.to_csv('./out_knn_n01/neighbours_vs_acc.csv')

'''