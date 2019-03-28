import numpy as np
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


design = np.load('n4spin_design.npy')
all_B = np.load('n4B_design.npy')
result_B = np.load('n4B_target.npy')
target = np.load('n4spin_target.npy')

spin_tv, spin_test,B_tv, B_test, y_tv, y_test, yB_tv, yB_test = train_test_split(design, all_B, target, result_B, test_size=0.2, random_state = 634)

model2 = load_model('func4large.h5')
model1 = load_model('N4mag-spin.h5')

y1_mu = np.array([ 2.65755092e-05, -9.60312440e-06, -2.44027858e-05,  7.43040106e-06])
y1_std = np.array([0.03884333, 0.03883743, 0.03885922, 0.03886195])
X1_mu = np.array([-1.87163475e-06,  2.03249798e-04,  3.02437797e-04,  1.03804564e-04])
X1_std = np.array([0.0577809,  0.05769463, 0.05773832, 0.0577605 ])


X1design_std = (B_test - X1_mu) / X1_std
y1_test_pred = y1_mu + model1.predict(X1design_std)*y1_std


y2_mu = -1.218065283312295e-05
y2_std = 2.8209918128660617e-05
X2_mu = np.array([ 2.65755092e-05, -9.60312440e-06, -2.44027858e-05,  7.43040106e-06])
X2_std = np.array([0.03884333, 0.03883743, 0.03885922, 0.03886195])

X2design_std = (y1_test_pred - X2_mu) / X2_std
y2_test_pred = y2_mu + model2.predict(X2design_std)*y2_std

yB_test_pred = np.zeros(len(yB_test))
for i in range(len(yB_test)):
    yB_test_pred[i] = np.dot(y1_test_pred[i],B_test[i])/2

"""
yB_test_pred = np.zeros(len(yB_test))
for i in range(len(yB_test)):
    yB_test_pred[i] = np.dot(spin_test[i],B_test[i])/2
"""
print("Generalization MSE: ", (mean_squared_error(y_true=y_test, y_pred=y2_test_pred)))
print("Generalization MAE: ", (mean_absolute_error(y_true=y_test, y_pred=y2_test_pred)))
print("\n")
print("Generalization MSE: ", (mean_squared_error(y_true=yB_test, y_pred=y2_test_pred[0]+yB_test_pred)))
print("Generalization MAE: ", (mean_absolute_error(y_true=yB_test, y_pred=y2_test_pred[0]+yB_test_pred)))
