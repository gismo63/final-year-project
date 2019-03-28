import numpy as np
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

def fin_dif(model,Sz,h,n,y_mu,y_std, X_mu, X_std):
    grad = np.zeros(n)
    Sz_std = (Sz - X_mu) / X_std
    for i in range(n):
        diff = np.zeros(n)
        diff[i] = h
        #diff[i-1] = -h
        fxplus = y_mu+model.predict(np.array([Sz_std+diff,]))*y_std
        fxminus = y_mu+model.predict(np.array([Sz_std-diff,]))*y_std
        #print (fxplus)
        grad[i] = (fxplus-fxminus)/(2*h)
    return grad


n = 4
h = 100000
#np.random.seed(634)

J = np.zeros((n,n))
for i in range(n-1):
    J[i][i+1] = 1
J[0][n-1] = 1
strength = 1./np.sqrt(n) #overall multiplicative factor of interation strength
#J= J*strength


design = np.load('n4spin_design.npy')
all_B = np.load('n4B_design.npy')
result_B = np.load('n4B_target.npy')
target = np.load('n4spin_target.npy')

ind = np.random.randint(10000)
Sz = np.zeros(n)
B = all_B[ind]/2
gamma = 1e-8
diff = 1e-5
y_mu = -1.218065283312295e-05
y_std = 2.8209918128660617e-05
X_mu = np.array([ 2.65755092e-05, -9.60312440e-06, -2.44027858e-05,  7.43040106e-06])
X_std = np.array([0.03884333, 0.03883743, 0.03885922, 0.03886195])

model = load_model('func4smol.h5')
"""
gradient = B+fin_dif(model,Sz,diff,n,y_mu,y_std)
while np.linalg.norm(gradient)>1e-10:
    Sz -= gamma*gradient
    abs_Sz = np.absolute(Sz)
    if max(abs_Sz)>1:
        break
    gradient = B+fin_dif(model,Sz,diff,n,y_mu,y_std)
    #print (Sz)
"""

gs_spin = design[ind]
gs_spin_std = (gs_spin - X_mu) / X_std
gs_spin_std2 = (2*gs_spin - X_mu) / X_std
print (B)
print (-fin_dif(model,Sz,diff,n,y_mu,y_std,X_mu,X_std))
"""
print (np.linalg.norm(gs_spin)/np.linalg.norm(B))
print (np.dot(B,gs_spin))
print (-np.linalg.norm(B)*np.linalg.norm(gs_spin))
print (np.dot(B,gs_spin)+y_mu+model.predict(np.array([gs_spin,]))[0]*y_std)
"""
delta = np.zeros(n)
delta[0] = diff
delta[2] = -diff

fx = y_mu+model.predict(np.array([gs_spin_std,]))*y_std
fxplus = y_mu+model.predict(np.array([gs_spin_std+diff,]))*y_std
fxminus = y_mu+model.predict(np.array([gs_spin_std-diff,]))*y_std
fx2 = y_mu+model.predict(np.array([gs_spin_std2,]))*y_std

"""
print (fxplus)
print (fxminus)
print (fx2)
"""

print (np.dot(B,gs_spin)+fx[0])
print (np.dot(B,gs_spin+diff)+fxplus[0])
print (np.dot(B,gs_spin-diff)+fxminus[0])
print (np.dot(B,2*gs_spin)+fx2[0])
print(2*gs_spin)
