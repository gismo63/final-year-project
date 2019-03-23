import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense

def plusminus_minusplus(i,j,n): #will calculate S_(i+)S_(j-) + S_(i-)S_(j+)
    order = [] #empty list that will hold the order of the direct product
    for k in range(n):
        if (k)==i:
            order.append(s_plus)
        elif (k)==j:
            order.append(s_minus)
        else:
            order.append(identity)
    plus_minus = order[n-1]
    for l in range(n-1):
        plus_minus = np.kron(order[n-l-2],plus_minus) # calculates the direct product of the matricies in "order" which gives S_(i+)S_(j-)

    order[i], order[j] = order[j], order[i] # swaps the i and j entry in "order" in order to calculate S_(i-)S_(j+)
    minus_plus = order[n-1]
    for l in range(n-1):
        minus_plus = np.kron(order[n-l-2],minus_plus)

    return (plus_minus + minus_plus)

def zz(i,j,n):
    order = []
    for k in range(n):
        if (k)==i:
            order.append(s_z)
        elif (k)==j:
            order.append(s_z)
        else:
            order.append(identity)
    z_mat = order[n-1]
    for l in range(n-1):
        z_mat = np.kron(order[n-l-2],z_mat)

    return z_mat

def mag_field(n): #returns the sum over all S_iz which when multiplied by the field strength gives the contribution of the magnetic field to the hamiltonian
    field = np.zeros((2**n,2**n))
    for i in range(n):
        order = []
        for k in range(n):
            if (k)==i:
                order.append(s_z)
            else:
                order.append(identity)
        z_mat = order[n-1]
        for l in range(n-1):
            z_mat = np.kron(order[n-l-2],z_mat)
        field += z_mat
    return field

def hamiltonian(n,J,B): #calculates the Hamiltonian for the Heisenberg model
    H=0
    for j in range(n):
        i=0
        while i<j: #only want to consider each interaction once and don't want i=j
            if J[i][j] != 0: #check if the spins interact at all
                #print i+1,j+1
                H += (J[i][j])*(plusminus_minusplus(i,j,n)/2. + zz(i,j,n)/4.)
                #print (J[i][j])*(plusminus_minusplus(i,j,n)/2. + zz(i,j,n)/4.)
            i += 1
    if B!=0:
        H = np.add(H, B*mag_field(n)/2) #adds the magnetic field contribution to the hamiltonian, the factor of 1/2 is to account for the lack of this factor in the definition of s_z

    return H

def eigen(matrix):
    return np.linalg.eigh(matrix) #calculates eigenvalues and eigenvectors of a hermitian matrix


def expect_sz(n, g_state): #Calculates the expectation values of S_iz for each i
    Sz = np.zeros(n)
    for i in range(n):
        order = []
        for k in range(n):
            if (k)==i:
                order.append(s_z)
            else:
                order.append(identity)
        z_mat = order[n-1]
        for l in range(n-1):
            z_mat = np.kron(order[n-l-2],z_mat)
        Sz[i] = g_state.dot(z_mat.dot(g_state)) #computes <phi0|S_iz|phi0> where phi0 is the ground state eigenvector
    return Sz

def expect_sy(n, g_state): #Calculates the expectation values of S_iy for each i
    Sy = np.zeros(n)
    for i in range(n):
        order = []
        for k in range(n):
            if (k)==i:
                order.append(s_y)
            else:
                order.append(identity)
        y_mat = order[n-1]
        for l in range(n-1):
            y_mat = np.kron(order[n-l-2],y_mat)
        Sy[i] = g_state.dot(y_mat.dot(g_state)) #computes <phi0|S_iy|phi0> where phi0 is the ground state eigenvector
    return Sy

def expect_sx(n, g_state): #Calculates the expectation values of S_ix for each i
    Sx = np.zeros(n)
    for i in range(n):
        order = []
        for k in range(n):
            if (k)==i:
                order.append(s_x)
            else:
                order.append(identity)
        x_mat = order[n-1]
        for l in range(n-1):
            x_mat = np.kron(order[n-l-2],x_mat)
        Sx[i] = g_state.dot(x_mat.dot(g_state)) #computes <phi0|S_ix|phi0> where phi0 is the ground state eigenvector
    return Sx





s_plus = np.array([[0,1],[0,0]]) #spin raising operator in the |up>=(1,0) and |down>=(0,1) basis
s_minus = np.array([[0,0],[1,0]]) #spin lowering operator in same basis
s_z = np.array([[1,0],[0,-1]]) #z projection spin operator *2 in same basis
s_y = np.array([[0,-1],[1,0]])
s_x = np.array([[0,1],[1,0]])
identity = np.array([[1,0],[0,1]]) #identity matrix in same basis


n = 4 # number of spin sites
n_j = int(n*(n-1)/2)
print (n_j)
h = 10000

design = np.ndarray(shape = (h,n_j))
target = np.zeros(h)

strength = 1 #overall multiplicative factor of interation strength

for k in range(h):
    J = np.zeros((n,n))

    norm = []
    p = 0
    for j in range(n):
        i=0
        while i<j:
            J[i][j] = np.random.normal(0,1) # random number with normal distribution centered on 0, standard deviation 1
            norm.append(J[i][j]**2)
            i+=1
    norm_sum = np.sum(norm)
    J_std = np.std(norm)
    J /= np.sqrt(norm_sum)
    for j in range(n):
        i=0
        while i<j:
            design[k][p] = J[i][j]
            i+=1
            p+=1




    B = np.random.normal(0,1) #magnetic field strength
    B = 0

    H = hamiltonian(n,J,B)

    #print H

    eigenvalues, eigenvectors = eigen(H)

    eigenvalues = eigenvalues.round(10)
    target[k] = eigenvalues[0]


# define the model
model = Sequential()

# add layers
model.add(Dense(2**(n+2),input_dim=n_j, activation='relu'))
model.add(Dense(2**(n+2), activation='relu'))
model.add(Dense(units=1, activation='linear'))

model.compile(optimizer='adam',
              loss='mse')


X_tv, X_test, y_tv, y_test = train_test_split(design, target, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.2)

X_train_std = X_train
X_val_std = X_val
X_test_std = X_test

"""
X_mu = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)

X_train_std = (X_train - X_mu) / X_std
X_val_std = (X_val - X_mu) / X_std
X_test_std = (X_test - X_mu) / X_std
"""
y_train_std = y_train
y_val_std = y_val
y_test_std = y_test
"""
y_mu = np.mean(y_train, axis=0)
y_std = np.std(y_train, axis=0)

y_train_std = (y_train - y_mu) / y_std
y_val_std = (y_val - y_mu) / y_std
y_test_std = (y_test - y_mu) / y_std
"""

print (X_train_std)
print (y_train_std)

model_history = model.fit(X_train_std, y_train_std, epochs=100, verbose=2,validation_data=(X_val_std, y_val_std))
#model.save('low4.h5')


plt.figure()
plt.plot(range(1, len(model_history.history['loss'])+1), model_history.history['loss'], label='Train')
plt.plot(range(1, len(model_history.history['val_loss'])+1), model_history.history['val_loss'], label='Val')
plt.legend()
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')

plt.show()

# Generalization Error

y_test_pred = model.predict(X_test_std)
print("Generalization MSE: %f" % (mean_squared_error(y_true=y_test, y_pred=y_test_pred)))
print("Generalization MAE: %f" % (mean_absolute_error(y_true=y_test, y_pred=y_test_pred)))

plt.figure()
plt.scatter(y_test_pred, y_test)
plt.xlabel('y_pred')
plt.ylabel('y_true')
plt.plot([0,-2], [0,-2], linestyle='dashed', color='k')
plt.grid()
plt.show()
