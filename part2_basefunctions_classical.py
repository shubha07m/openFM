import pickle
import warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')


def preprocess(data):
    X = data.drop(
        columns=['Alert Detail 2', 'Alert ID', 'Available Knowledge Indicators', 'Suggested Action',
                 'Display Date', 'First In Service Date', 'Fwd To 1C Unique ID', 'Host Site Cd', 'Incident ID',
                 'NoLongerImpacted', 'Ring Id Category', 'Ring Id Description', 'Standard Message Created Date',
                 'Standard Message Unique ID', 'TT Id', 'KMS SYS ID', 'KPIInd', 'Standard Message Action Indicator'])

    Y = data.values[:, -1]

    # encode class values as integers
    encoder = LabelEncoder()
    Y = encoder.fit_transform(Y)
    df_col = list(X.columns)
    for i in range(len(df_col)):
        X[df_col[i]] = encoder.fit_transform(X[df_col[i]])
    X = X.values
    # Using PCA for feature reduction #
    # pca = PCA(n_components=10, svd_solver='full')
    # X = pca.fit_transform(X)
    return X, Y


def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


def modelselect(data):
    folds = StratifiedKFold(n_splits=10)
    Naive_Bayes = GaussianNB()
    Logistic_Regression = LogisticRegression()
    K_nearest_Neighbour = KNeighborsClassifier(n_neighbors=10)
    Support_Vector_Machine = SVC()
    Decision_Tree = DecisionTreeClassifier()
    Random_Forest = RandomForestClassifier()

    Feature, Target = preprocess(data)
    Feature_train, Feature_test, Target_train, Target_test = train_test_split(Feature, Target, test_size=0.3,
                                                                              random_state=123)
    X, Y = Feature_train, Target_train
    acc_nvb = []
    acc_logistic = []
    acc_knn = []
    acc_svm = []
    acc_tree = []
    acc_forest = []
    for train_index, test_index in folds.split(X, Y):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]

        # calculating accuracy % #
        acc_nvb.append(get_score(Naive_Bayes, X_train, X_test, y_train, y_test))
        acc_logistic.append(get_score(Logistic_Regression, X_train, X_test, y_train, y_test))
        acc_knn.append(get_score(K_nearest_Neighbour, X_train, X_test, y_train, y_test))
        acc_svm.append(get_score(Support_Vector_Machine, X_train, X_test, y_train, y_test))
        acc_tree.append(get_score(Decision_Tree, X_train, X_test, y_train, y_test))
        acc_forest.append(get_score(Random_Forest, X_train, X_test, y_train, y_test))
    acc_nvb = 100 * np.average(acc_nvb)
    acc_logistic = 100 * np.average(acc_logistic)
    acc_knn = 100 * np.average(acc_knn)
    acc_svm = 100 * np.average(acc_svm)
    acc_tree = 100 * np.average(acc_tree)
    acc_forest = 100 * np.average(acc_forest)
    print('The Naive Bays accuracy is: %0.4f' % acc_nvb)
    print('The Logistic Regression accuracy is: %0.4f' % acc_logistic)
    print('The K - nearest  Neighbour accuracy is: %0.4f' % acc_knn)
    print('The support vector machine accuracy is: %0.4f' % acc_svm)
    print('The decision tree accuracy is: %0.4f' % acc_tree)
    print('The random forest accuracy is: %0.4f' % acc_forest)


def modeltraining(current_data, modelname):
    # Getting training data for fitting #
    X, Y = preprocess(current_data)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=123)
    # Creating and saving the model to disk #
    best_model = RandomForestClassifier(criterion='entropy')
    accuracy = get_score(best_model, X_train, X_test, Y_train, Y_test)
    accuracy = 100 * float(accuracy)
    print('The current test accuracy after training the model: %0.4f' % accuracy)
    print('\n')
    filename = modelname
    pickle.dump(best_model, open(filename, 'wb'))
    print('The trained model has been saved to disk')


def out_ofsample_perf(trained_model, new_data):
    loaded_model = pickle.load(open(trained_model, 'rb'))
    X_ous, Y_ous = preprocess(new_data)
    X_ous_train, X_ous_test, Y_ous_train, Y_ous_test = train_test_split(X_ous, Y_ous, test_size=0.3, random_state=123)
    accuracy = get_score(loaded_model, X_ous_train, X_ous_test, Y_ous_train, Y_ous_test)

    accuracy = 100 * float(accuracy)
    print('The test accuracy on out of sample data is: %0.4f' % accuracy)