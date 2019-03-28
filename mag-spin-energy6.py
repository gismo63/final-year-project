import numpy as np
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


design = np.load('n6spin_design.npy')
all_B = np.load('n6B_design.npy')
result_B = np.load('n6B_target.npy')
target = np.load('n6spin_target.npy')

spin_tv, spin_test,B_tv, B_test, y_tv, y_test, yB_tv, yB_test = train_test_split(design, all_B, target, result_B, test_size=0.2, random_state = 634)

model2 = load_model('func6large.h5')
model1 = load_model('N6mag-spin.h5')

y1_mu = np.array([ 0.00026084, -0.00021509,  0.00026743, -0.00018174,  0.00019983, -0.00033128])
y1_std = np.array([0.05153156, 0.05154166, 0.0515238,  0.05154788, 0.05151589, 0.05147954])
X1_mu = np.array([-8.24789646e-05, -1.10607677e-04, -4.12207722e-04, -1.67976533e-04,1.86559274e-04,  6.60258129e-04])
X1_std = np.array([0.05782041, 0.05780924, 0.05765137, 0.05783217, 0.05769232, 0.05756768])


X1design_std = (B_test - X1_mu) / X1_std
y1_test_pred = y1_mu + model1.predict(X1design_std)*y1_std


y2_mu = -4.682268882640634e-05
y2_std = 0.0001131648958961785
X2_mu = np.array([ 0.00026084, -0.00021509,  0.00026743, -0.00018174,  0.00019983, -0.00033128])
X2_std = np.array([0.05153156, 0.05154166, 0.0515238,  0.05154788, 0.05151589, 0.05147954])

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
