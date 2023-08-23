import pickle
import warnings
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, KFold
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.optimizers import Adam
from tensorflow.python.keras.models import model_from_json, load_model
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

warnings.filterwarnings('ignore')


def datapreprocess(data):
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


def deep_model():
    model = Sequential()
    model.add(Dense(128, input_dim=39, activation='relu'))
    model.add(Dropout(.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(11, activation='softmax'))

    opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name="Adam")

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # # Fit the model
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=100,
        min_delta=0.0001,
        mode='max')
    return model


def deepmodelselect(data):
    X, Y = datapreprocess(data)
    X = pd.DataFrame(X)
    Y = np_utils.to_categorical(Y)
    X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=123)
    # x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=.25, random_state=123)
    # acc = deep_score(deep_model(), x_train, x_val, y_train, y_val)

    estimator = KerasClassifier(build_fn=deep_model, epochs=500, batch_size=64, verbose=0)
    kfold = KFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, X_train, Y_train, cv=kfold)
    print("Cross validated accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))


def modeltraining(current_data):
    model = deep_model()
    X, Y = datapreprocess(current_data)
    X = pd.DataFrame(X)
    Y = np_utils.to_categorical(Y)
    X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=123)
    history = model.fit(X_train, Y_train, batch_size=64, epochs=500, verbose=1, validation_data=(x_test, y_test))

    # serialize model to JSON
    model_json = model.to_json()
    with open("deeptesting.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("deeptesting.h5")
    print("Saved model to disk")

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # plotting training and test loss from history
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # # evaluate the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("test data %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


def out_ofsample_perf(new_data, model, weight):
    X, Y = datapreprocess(new_data)
    X = pd.DataFrame(X)
    Y = np_utils.to_categorical(Y)

    # load json and create model
    json_file = open(model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weight)
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = loaded_model.evaluate(X, Y, verbose=0)
    print("out of sample %s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))