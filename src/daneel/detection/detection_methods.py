import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from daneel.detection.data_process import *
import tensorflow as tf
from imblearn.over_sampling import SMOTE

def np_X_Y_from_df(df):
    df = shuffle(df)
    df_X = df.drop(['LABEL'], axis=1)
    X = np.array(df_X)
    Y_raw = np.array(df['LABEL']).reshape((len(df['LABEL']),1))
    Y = Y_raw == 2
    return X, Y

def LoadLightFluxData(param):
    df_train = pd.read_csv(param.get("train_dataset_path"), encoding="ISO-8859-1")
    df_test = pd.read_csv(param.get("test_dataset_path"), encoding="ISO-8859-1")
    
    df_train_x = df_train.drop('LABEL', axis=1)
    df_test_x = df_test.drop('LABEL', axis=1)
    df_train_y = df_train.LABEL
    df_test_y = df_test.LABEL
    
    LFP = LightFluxProcessor(
        fourier=True,
        normalize=True,
        gaussian=True,
        standardize=True)
    df_train_x, df_test_x = LFP.process(df_train_x, df_test_x)
    df_train_processed = pd.DataFrame(df_train_x).join(pd.DataFrame(df_train_y))
    df_test_processed = pd.DataFrame(df_test_x).join(pd.DataFrame(df_test_y))
    
    X_train, Y_train = np_X_Y_from_df(df_train_processed)
    Y_train = Y_train.ravel()
    X_test, Y_test = np_X_Y_from_df(df_test_processed)
    Y_test = Y_test.ravel()
    return X_train, Y_train, X_test, Y_test

def DetectionWithSVM(param):
    X_train, Y_train, X_test, Y_test = LoadLightFluxData(param=param)
        
    kernel = param.get("type_of_kernel")
    SVCModel = SVC(kernel=kernel)
    
    SVCModel.fit(X_train,Y_train)
    
    train_outputs = SVCModel.predict(X_train)
    test_outputs = SVCModel.predict(X_test)
    
    train_accuracy = accuracy_score(Y_train,train_outputs)
    test_accuracy = accuracy_score(Y_test,test_outputs)
    
    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")

def build_simple_NN(shape):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    return model

def DetectionWithNN(param):
    # Define Network
    X_train, Y_train, X_test, Y_test = LoadLightFluxData(param=param)
    model = build_simple_NN(X_train.shape[1:])
    sm = SMOTE()
    X_train_sm, Y_train_sm = sm.fit_resample(X_train, Y_train)
    epochs = param.get("epochs")
    batch_size = param.get("batch_size")
    history = model.fit(X_train_sm, Y_train_sm, epochs=epochs, batch_size=batch_size)
    
    train_output = np.rint(model.predict(X_train, batch_size=batch_size))
    test_output = np.rint(model.predict(X_test, batch_size=batch_size))
    
    train_accuracy = accuracy_score(Y_train, train_output)
    test_accuracy = accuracy_score(Y_test, test_output)
    
    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")
    
def build_simple_CNN(shape):
    print("Building model with input shape:", shape)
    model = tf.keras.models.Sequential(
        [
        tf.keras.layers.Input((shape[0],shape[1],1)),
        tf.keras.layers.Conv2D(5, (5,5)),
        tf.keras.layers.Conv2D(3, (2,2)),
        tf.keras.layers.Conv2D(1,(1,1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation = "relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid")]
    )
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    return model

def DetectionWithCNN(param):
    X_train, Y_train, X_test, Y_test = LoadLightFluxData(param=param)
    X_train = np.pad(X_train,((0,0),(0,2)))
    X_test = np.pad(X_test,((0,0),(0,2)))
    nearest_square = np.int64(np.floor(np.sqrt(X_train.shape[1])))
    X_train = X_train[:,:nearest_square**2].reshape(X_train.shape[0],nearest_square,nearest_square,1)
    X_test = X_test[:,:nearest_square**2].reshape(X_test.shape[0],nearest_square,nearest_square,1)
    np.save("/home/ubuntu/comp_astro_24/Assigments/Assigments 2/Data/img_data/img.npy",np.vstack([X_train,X_test]))
    np.save("/home/ubuntu/comp_astro_24/Assigments/Assigments 2/Data/img_data/img_label.npy", np.concatenate([Y_train,Y_test]))
    print("X_train shape:", X_train.shape)
    
    model = build_simple_CNN(X_train.shape[1:])
    epochs = param.get("epochs")
    batch_size = param.get("batch_size")
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)
    
    train_output = np.rint(model.predict(X_train, batch_size=batch_size))
    test_output = np.rint(model.predict(X_test, batch_size=batch_size))
    
    train_accuracy = accuracy_score(Y_train, train_output)
    test_accuracy = accuracy_score(Y_test, test_output)
    
    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")