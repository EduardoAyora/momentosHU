import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

dataframe = pandas.read_csv("entrenamiento.csv", header=None, sep=';')
dataset = dataframe.values
inputDimensions = 7
X = dataset[:,0:inputDimensions].astype(float)
Y = dataset[:,inputDimensions]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)


def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(16, input_dim=inputDimensions, activation='relu'))
	model.add(Dense(8, activation='relu'))
	# model.add(Dense(9, activation='softmax'))
	model.add(Dense(9, activation='sigmoid'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model=baseline_model
estimator = KerasClassifier(build_fn=model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

model_json = model().to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model().save_weights("model.h5")
print("Saved model to disk")
