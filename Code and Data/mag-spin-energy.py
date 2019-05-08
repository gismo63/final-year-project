import numpy as np
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


design = np.load('n8spin_design.npy')
all_B = np.load('n8B_design.npy')
result_B = np.load('n8B_target.npy')
target = np.load('n8spin_target.npy')

spin_tv, spin_test,B_tv, B_test, y_tv, y_test, yB_tv, yB_test = train_test_split(design, all_B, target, result_B, test_size=0.2, random_state = 634)

model2 = load_model('func8large.h5')
model1 = load_model('N8mag-spin.h5')

y1_mu = np.array([ 0.00026103, -0.00023328,  0.00018578, -0.00013062,  0.00016591, -0.00016217, 0.00013488, -0.00022154])
y1_std = np.array([0.06127522, 0.06122913, 0.06118854, 0.06118521, 0.06112665, 0.06115292, 0.06113876, 0.06115805])
X1_mu = np.array([-3.58910654e-04,  8.48517168e-05, -1.46107879e-04, -2.61895111e-04, -1.90709145e-04,  1.26259411e-05,  1.70400941e-04,  1.32068508e-04])
X1_std = np.array([0.05776641, 0.0576916,  0.05769298, 0.05784509, 0.05784414, 0.05756983, 0.05771698, 0.05767117])


X1design_std = (B_test - X1_mu) / X1_std
y1_test_pred = y1_mu + model1.predict(X1design_std)*y1_std


y2_mu = -0.0001122570807335145
y2_std = 0.00027502511845213116
X2_mu = np.array([ 0.00026103, -0.00023328,  0.00018578, -0.00013062,  0.00016591, -0.00016217, 0.00013488, -0.00022154])
X2_std = np.array([0.06127522, 0.06122913, 0.06118854, 0.06118521, 0.06112665, 0.06115292, 0.06113876, 0.06115805])

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
