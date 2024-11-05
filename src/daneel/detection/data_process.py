import pandas as pd
import numpy as np
from scipy import ndimage, fft
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class Detection:

    def __init__(self, dataset):
        self.dataset = dataset


class LightFluxProcessor:

    def __init__(self, fourier=True, normalize=True, gaussian=True, standardize=True):
        self.fourier = fourier
        self.normalize = normalize
        self.gaussian = gaussian
        self.standardize = standardize

    def fourier_transform(self, X):
        return np.abs(fft.fft(X, n=X.size))

    def process(self, df_train_x, df_dev_x):
        # Apply fourier transform
        if self.fourier:
            print("Applying Fourier...")
            shape_train = df_train_x.shape
            shape_dev = df_dev_x.shape
            df_train_x = df_train_x.apply(self.fourier_transform, axis=1)
            df_dev_x = df_dev_x.apply(self.fourier_transform, axis=1)

            df_train_x_build = np.zeros(shape_train)
            df_dev_x_build = np.zeros(shape_dev)

            for ii, x in enumerate(df_train_x):
                df_train_x_build[ii] = x

            for ii, x in enumerate(df_dev_x):
                df_dev_x_build[ii] = x

            df_train_x = pd.DataFrame(df_train_x_build)
            df_dev_x = pd.DataFrame(df_dev_x_build)

            # Keep first half of data as it is symmetrical after previous steps
            df_train_x = df_train_x.iloc[:, : (df_train_x.shape[1] // 2)].values
            df_dev_x = df_dev_x.iloc[:, : (df_dev_x.shape[1] // 2)].values

        # Normalize
        if self.normalize:
            print("Normalizing...")
            df_train_x = pd.DataFrame(normalize(df_train_x))
            df_dev_x = pd.DataFrame(normalize(df_dev_x))

        # Gaussian filter to smooth out data
        if self.gaussian:
            print("Applying Gaussian Filter...")
            df_train_x = ndimage.filters.gaussian_filter(df_train_x, sigma=10)
            df_dev_x = ndimage.filters.gaussian_filter(df_dev_x, sigma=10)

        if self.standardize:
            # Standardize X data
            print("Standardizing...")
            std_scaler = StandardScaler()
            df_train_x = std_scaler.fit_transform(df_train_x)
            df_dev_x = std_scaler.transform(df_dev_x)

        print("Finished Processing!")
        return df_train_x, df_dev_x


def np_X_Y_from_df(df):
    df = shuffle(df)
    df_X = df.drop(['LABEL'], axis=1)
    X = np.array(df_X)
    Y_raw = np.array(df['LABEL']).reshape((len(df['LABEL']),1))
    Y = Y_raw == 2
    return X, Y

def DetectionWithSVM(Parameters):
    df_train = pd.read_csv(Parameters.get("train_dataset_path"), encoding="ISO-8859-1")
    df_test = pd.read_csv(Parameters.get("test_dataset_path"), encoding="ISO-8859-1")
    
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
        
    kernel = Parameters.get("type_of_kernel")
    SVCModel = SVC(kernel=kernel)
    
    SVCModel.fit(X_train,Y_train)
    
    train_outputs = SVCModel.predict(X_train)
    test_outputs = SVCModel.predict(X_test)
    
    train_accuracy = accuracy_score(Y_train,train_outputs)
    test_accuracy = accuracy_score(Y_test,test_outputs)
    
    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")
    