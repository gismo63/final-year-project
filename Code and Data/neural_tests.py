from keras.models import Sequential
from keras.layers import Dense
from sklearn.svm import SVR
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score # Accuracy
# generate regression dataset
X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=1)
scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
scalarX.fit(X)
scalarY.fit(y.reshape(100,1))
X = scalarX.transform(X)
print (X)
y = scalarY.transform(y.reshape(100,1))
print (y)
# define and fit the final model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')


X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.3333333)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.5)

X_mu = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)

X_train_std = (X_train - X_mu) / X_std
X_val_std = (X_val - X_mu) / X_std
X_test_std = (X_test - X_mu) / X_std

y_mu = np.mean(y_train, axis=0)
y_std = np.std(y_train, axis=0)

y_train_std = (y_train - y_mu) / y_std
y_val_std = (y_val - y_mu) / y_std
y_test_std = (y_test - y_mu) / y_std


model_history = model.fit(X_train_std, y_train_std, epochs=1000, verbose=0, validation_data=(X_val_std, y_val_std))

plt.figure()
plt.plot(range(1, len(model_history.history['loss'])+1), model_history.history['loss'], label='Train')
plt.plot(range(1, len(model_history.history['val_loss'])+1), model_history.history['val_loss'], label='Val')
plt.legend()
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')

plt.show()

y_pred = model.predict(X_test_std)
# go from one-hot to classes
y_pred_class = np.argmax(y_pred, axis=1)
y_test_class = np.argmax(y_test, axis=1)

print("Test Accuracy:", accuracy_score(y_true=y_test_class, y_pred=y_pred_class))
