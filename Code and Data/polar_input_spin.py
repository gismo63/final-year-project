import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler

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

def polar(n, pol, n_j):
    cart = 1
    for i in range(n):
        cart *= np.sin(pol[i])
    if n != n_j:
        cart *= np.cos(pol[n])
    return cart



s_plus = np.array([[0,1],[0,0]]) #spin raising operator in the |up>=(1,0) and |down>=(0,1) basis
s_minus = np.array([[0,0],[1,0]]) #spin lowering operator in same basis
s_z = np.array([[1,0],[0,-1]]) #z projection spin operator *2 in same basis
s_y = np.array([[0,-1],[1,0]])
s_x = np.array([[0,1],[1,0]])
identity = np.array([[1,0],[0,1]]) #identity matrix in same basis


n = 6 # number of spin sites
n_j = int(n*(n-1)/2)-1
print (n_j)
h = 10000
n_spins = int(n/2)+1

design = np.ndarray(shape = (h,n_j))
target = np.zeros(h)

strength = 1 #overall multiplicative factor of interation strength

for k in range(h):
    J = np.zeros((n,n))
    p = 0
    design[k] = np.random.rand(n_j)*np.pi

    for j in range(n):
        i=0
        while i<j:
            J[i][j] = polar(p,design[k]) # random number with normal distribution centered on 0, standard deviation 1
            i+=1
            p+=1




    B = np.random.normal(0,1) #magnetic field strength
    B = 0

    H = hamiltonian(n,J,B)

    #print H

    eigenvalues, eigenvectors = eigen(H)

    eigenvalues = eigenvalues.round(10)

    b,c  = np.unique(eigenvalues,return_counts=True)

    target[k] = int((c[0]+1)/2)-1


# one-hot y
target_one_hot = np.zeros((h, n_spins), dtype=int)
for i in range(h):
    target_one_hot[i, int(target[i])] = 1


# define the model
model = Sequential()

# add layers
model.add(Dense(2**n,input_dim=n_j, activation='relu'))
model.add(Dense(2**n, activation='relu'))
model.add(Dense(units=n_spins, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics = ['acc'])


X_tv, X_test, y_tv, y_test = train_test_split(design, target_one_hot, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.2)

X_mu = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)

X_train_std = (X_train - X_mu) / X_std
X_val_std = (X_val - X_mu) / X_std
X_test_std = (X_test - X_mu) / X_std


model_history = model.fit(X_train_std, y_train, epochs=100, verbose=2,validation_data=(X_val_std, y_val), batch_size=32)

plt.figure()
plt.plot(range(1, len(model_history.history['acc'])+1), model_history.history['acc'], label='Train')
plt.plot(range(1, len(model_history.history['val_acc'])+1), model_history.history['val_acc'], label='Val')
plt.legend()
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')

plt.show()

# Generalization Error

y_test_pred = model.predict(X_test_std)
y_pred_class = np.argmax(y_test_pred, axis=1)
y_test_class = np.argmax(y_test, axis=1)

print("Test Accuracy:", accuracy_score(y_true=y_test_class, y_pred=y_pred_class))
"""
plt.figure()
plt.scatter(y_pred_class, y_test_class)
plt.xlabel('y_pred')
plt.ylabel('y_true')
plt.plot([0,n/2+1], [0,n/2+1], linestyle='dashed', color='k')
plt.grid()
plt.show()
"""
