import numpy as np
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

strength = 1

#interaction strength where J[i][j] represents the strength of the interaction between particle i and particle j
#J = strength*np.array([[0,1,0,0,1],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[0,0,0,0,0]]) #J for 5 electron chain with periodic boundary conditions
J = strength*np.array([[0,1.,0,1],[0,0,1.,0],[0,0,0,1.],[0,0,0,0]]) #J for 4 electron chain with periodic boundary conditions
#J = strength*np.array([[0,1.,1.],[0,0,1.],[0,0,0]])
#J = np.array([[0,1],[1,0]]) #J for 2 electron system

n = 4 # number of spin sites
n_j = int(n*(n-1)/2.)
design = np.zeros(n_j)

norm = []
p = 0
for j in range(n):
    i=0
    while i<j:
        norm.append(J[i][j]**2)
        i+=1
norm_sum = np.sum(norm)
J /= np.sqrt(norm_sum)
for j in range(n):
    i=0
    while i<j:
        design[p] = J[i][j]
        i+=1
        p+=1

model = load_model('uni4.h5')

print (len(design))
y_test_pred = model.predict(np.array([design,]))
print(y_test_pred)
