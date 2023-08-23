import pickle
import warnings
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')


def preprocess(data):
    X = data.drop(
        columns=['Alert Detail 2', 'Alert ID', 'Available Knowledge Indicators', 'False alarm Indicator',
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
    # Using PCA for feature reduction # If PCA not used, comment out next 2 lines below
    pca = PCA(n_components=10, svd_solver='full')
    X = pca.fit_transform(X)
    return X, Y


def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model.score(X_test, y_test), precision_score(y_test, y_pred), recall_score(y_test, y_pred)


def modelselect(data):
    folds = StratifiedKFold(n_splits=10)
    Naive_Bayes = GaussianNB()
    Logistic_Regression = LogisticRegression(solver='liblinear', C=.5)
    K_nearest_Neighbour = KNeighborsClassifier(n_neighbors=10)
    Support_Vector_Machine = SVC(probability=True)
    Decision_Tree = DecisionTreeClassifier(criterion='entropy')
    Random_Forest = RandomForestClassifier(criterion='entropy')

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
    prec_nvb = []
    prec_logistic = []
    prec_knn = []
    prec_svm = []
    prec_tree = []
    prec_forest = []
    rec_nvb = []
    rec_logistic = []
    rec_knn = []
    rec_svm = []
    rec_tree = []
    rec_forest = []
    for train_index, test_index in folds.split(X, Y):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]

        # calculating accuracy % #
        acc_nvb.append(get_score(Naive_Bayes, X_train, X_test, y_train, y_test)[0])
        acc_logistic.append(get_score(Logistic_Regression, X_train, X_test, y_train, y_test)[0])
        acc_knn.append(get_score(K_nearest_Neighbour, X_train, X_test, y_train, y_test)[0])
        acc_svm.append(get_score(Support_Vector_Machine, X_train, X_test, y_train, y_test)[0])
        acc_tree.append(get_score(Decision_Tree, X_train, X_test, y_train, y_test)[0])
        acc_forest.append(get_score(Random_Forest, X_train, X_test, y_train, y_test)[0])

        # calculating precision % #
        prec_nvb.append(get_score(Naive_Bayes, X_train, X_test, y_train, y_test)[1])
        prec_logistic.append(get_score(Logistic_Regression, X_train, X_test, y_train, y_test)[1])
        prec_knn.append(get_score(K_nearest_Neighbour, X_train, X_test, y_train, y_test)[1])
        prec_svm.append(get_score(Support_Vector_Machine, X_train, X_test, y_train, y_test)[1])
        prec_tree.append(get_score(Decision_Tree, X_train, X_test, y_train, y_test)[1])
        prec_forest.append(get_score(Random_Forest, X_train, X_test, y_train, y_test)[1])

        # calculating recall % #
        rec_nvb.append(get_score(Naive_Bayes, X_train, X_test, y_train, y_test)[2])
        rec_logistic.append(get_score(Logistic_Regression, X_train, X_test, y_train, y_test)[2])
        rec_knn.append(get_score(K_nearest_Neighbour, X_train, X_test, y_train, y_test)[2])
        rec_svm.append(get_score(Support_Vector_Machine, X_train, X_test, y_train, y_test)[2])
        rec_tree.append(get_score(Decision_Tree, X_train, X_test, y_train, y_test)[2])
        rec_forest.append(get_score(Random_Forest, X_train, X_test, y_train, y_test)[2])

    acc_nvb = 100 * np.average(acc_nvb)
    acc_logistic = 100 * np.average(acc_logistic)
    acc_knn = 100 * np.average(acc_knn)
    acc_svm = 100 * np.average(acc_svm)
    acc_tree = 100 * np.average(acc_tree)
    acc_forest = 100 * np.average(acc_forest)

    prec_nvb = 100 * np.average(prec_nvb)
    prec_logistic = 100 * np.average(prec_logistic)
    prec_knn = 100 * np.average(prec_knn)
    prec_svm = 100 * np.average(prec_svm)
    prec_tree = 100 * np.average(prec_tree)
    prec_forest = 100 * np.average(prec_forest)

    rec_nvb = 100 * np.average(rec_nvb)
    rec_logistic = 100 * np.average(rec_logistic)
    rec_knn = 100 * np.average(rec_knn)
    rec_svm = 100 * np.average(rec_svm)
    rec_tree = 100 * np.average(rec_tree)
    rec_forest = 100 * np.average(rec_forest)

    print('The Naive Bays accuracy is: %0.4f' % acc_nvb)
    print('The Naive Bays precision is: %0.4f' % prec_nvb)
    print('The Naive Bays recall is: %0.4f\n' % rec_nvb)

    print('The Logistic Regression accuracy is: %0.4f' % acc_logistic)
    print('The Logistic Regression precision is: %0.4f' % prec_logistic)
    print('The Logistic Regression recall is: %0.4f\n' % rec_logistic)

    print('The K - nearest  Neighbour accuracy is: %0.4f' % acc_knn)
    print('The K - nearest  Neighbour precision is: %0.4f' % prec_knn)
    print('The K - nearest  Neighbour recall is: %0.4f\n' % rec_knn)

    print('The support vector machine accuracy is: %0.4f' % acc_svm)
    print('The support vector machine precision is: %0.4f' % prec_svm)
    print('The support vector machine recall is: %0.4f\n' % rec_svm)

    print('The decision tree accuracy is: %0.4f' % acc_tree)
    print('The decision tree precision is: %0.4f' % prec_tree)
    print('The decision tree recall is: %0.4f\n' % rec_tree)

    print('The random forest accuracy is: %0.4f' % acc_forest)
    print('The random forest precision is: %0.4f' % prec_forest)
    print('The random forest recall is: %0.4f\n' % rec_forest)


def modeltraining(trained_model, current_data):
    # Getting training data for fitting #
    X, Y = preprocess(current_data)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=123)
    # Creating and saving the model to disk #
    best_model = RandomForestClassifier(criterion='entropy')
    accuracy, precision_score, recall_score = get_score(best_model, X_train, X_test, Y_train, Y_test)

    accuracy = 100 * float(accuracy)
    precision_score = 100 * float(precision_score)
    recall_score = 100 * float(recall_score)

    print('The current test accuracy after training the model: %0.4f' % accuracy)
    print('The current precision score after training the model: %0.4f' % precision_score)
    print('The current recall score after training the model: %0.4f' % recall_score)

    print('\n')
    filename = trained_model
    pickle.dump(best_model, open(filename, 'wb'))
    print('The trained model has been saved to disk')


def out_ofsample_perf(trained_model, new_data):
    loaded_model = pickle.load(open(trained_model, 'rb'))
    X_ous, Y_ous = preprocess(new_data)
    X_ous_train, X_ous_test, Y_ous_train, Y_ous_test = train_test_split(X_ous, Y_ous, test_size=0.3, random_state=123)
    accuracy, precision_score, recall_score = get_score(loaded_model, X_ous_train, X_ous_test, Y_ous_train, Y_ous_test)

    accuracy = 100 * float(accuracy)
    precision_score = 100 * float(precision_score)
    recall_score = 100 * float(recall_score)
    print('The test accuracy on out of sample data is: %0.4f' % accuracy)
    print('The precision score on out of sample data is: %0.4f' % precision_score)
    print('The recall score on out of sample data is: %0.4f' % recall_score)


def ROCcurve(current_data):
    Naive_Bayes = GaussianNB()
    Logistic_Regression = LogisticRegression(solver='liblinear', C=.5)
    K_nearest_Neighbour = KNeighborsClassifier(n_neighbors=10)
    Support_Vector_Machine = SVC(probability=True)
    Decision_Tree = DecisionTreeClassifier(criterion='entropy')
    Random_Forest = RandomForestClassifier(criterion='entropy')

    X, Y = preprocess(current_data)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=123)
    classifiers = [Naive_Bayes, Logistic_Regression, K_nearest_Neighbour, Support_Vector_Machine, Decision_Tree,
                   Random_Forest]
    result_table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])

    for cls in classifiers:
        model = cls.fit(X_train, Y_train)
        yproba = model.predict_proba(X_test)[::, 1]

        fpr, tpr, _ = roc_curve(Y_test, yproba)
        auc = roc_auc_score(Y_test, yproba)

        result_table = result_table.append({'classifiers': cls.__class__.__name__,
                                            'fpr': fpr,
                                            'tpr': tpr,
                                            'auc': auc}, ignore_index=True)

    # Set name of the classifiers as index labels
    result_table.set_index('classifiers', inplace=True)

    fig = plt.figure(figsize=(10, 10))

    for i in result_table.index:
        plt.plot(result_table.loc[i]['fpr'],
                 result_table.loc[i]['tpr'],
                 label="{}, AUC={:.4f}".format(i, result_table.loc[i]['auc']))

    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 12}, loc='lower right')
    plt.show()