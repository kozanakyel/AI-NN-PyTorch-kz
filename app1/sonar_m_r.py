import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

seed = 7
numpy.random.seed(seed)

dataframe = read_csv('sonar.csv', header=None)  # (208, 61)
dataset = dataframe.values

X = dataset[:, 0:60].astype(float)
Y = dataset[:, 60]  # outpuy string ve biz bunu 0 veya 1 e convert etmeliyiz

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)


def create_baseline():
    model = Sequential()
    model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# for tuning reduce hidden layer neurons numbers
def create_smaller():
    model = Sequential()
    model.add(Dense(30, input_dim=60, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_larger():
    model = Sequential()
    model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_larger, epochs=100, batch_size=5, verbose=0)))
# The pipeline is a wrapper that executes one or more models within a pass of the cross-validation procedure.
pipeline = Pipeline(estimators)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print('Larger: %.2f%% (%.2f%%)' % (results.mean() * 100, results.std() * 100))
