from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import SGD
import numpy as np

from sklearn.model_selection import train_test_split
import xgboost as xgb


# define CNN model
def CNN_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))

    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# define GBDT model
def GBDT_model(lr, md, ne, mcw):
    # learning_rate=0.5, max_depth=5, n_estimators=40, min_child_weight=3,
    model = xgb.XGBClassifier(learning_rate=lr, max_depth=md, n_estimators=ne, min_child_weight=mcw,
                            objective="multi:softprob", eval_metric="mlogloss")

    return model


def prepare_X_for_CNN(X_data):

    # reshape dataset to have a single channel
    X_data = X_data.reshape((X_data.shape[0], 28, 28, 1))

    # convert from integers to floats
    X_data = X_data.astype('float32')

    # normalize to range 0-1
    X_data = X_data / 255.0

    # return normalized images
    return X_data


def prepare_Y_for_CNN(y_data):

    # one hot encode target values
    y_data = to_categorical(y_data)

    return y_data


def main():

    ### prepare data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train_CNN, X_train_GBDT, y_train_CNN, y_train_GBDT = train_test_split(X_train, y_train, test_size=0.4, random_state=42)

    X_train_CNN = X_train
    y_train_CNN = y_train

    X_train_CNN = prepare_X_for_CNN(X_train_CNN)
    X_train_GBDT = prepare_X_for_CNN(X_train_GBDT)
    X_test = prepare_X_for_CNN(X_test)
    y_train_CNN = prepare_Y_for_CNN(y_train_CNN)
    y_test_CNN = prepare_Y_for_CNN(y_test)

    ### CNN

    # # define CNN model
    # CNN = CNN_model()
    # # fit CNN model
    # CNN.fit(X_train_CNN, y_train_CNN, epochs=10, batch_size=32, verbose=0)
    # # save CNN model
    # # fname = 'CNN_model-full epochs-10.h5'
    fname = 'CNN_model-' + str(0.4) + ' epochs-10.h5'
    # CNN.save(fname)

    # load model
    CNN = load_model(fname)

    _, accuracy = CNN.evaluate(X_test, y_test_CNN)
    print('CNN accuracy: ', accuracy)


    ### GBDT

    # new model to extract features for GBDT
    features_for_GBDT = Model(CNN.inputs, CNN.layers[-2].output)

    X_train_GBDT = features_for_GBDT.predict(X_train_GBDT)
    X_test = features_for_GBDT.predict(X_test)


    # define GBDT model
    GBDT = GBDT_model(0.5, 4, 60, 3)
    # fit GBDT model
    GBDT.fit(X_train_GBDT, y_train_GBDT)

    # X_train_CNN = features_for_GBDT.predict(X_train_CNN)
    # GBDT.fit(X_train_CNN, np.argmax(y_train_CNN, axis=-1))

    # result
    print('CNN-GBDT accuracy: ', GBDT.score(X_test, y_test))


main()
