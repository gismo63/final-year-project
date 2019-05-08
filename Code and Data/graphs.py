import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
s_z = np.array([[1,0],[0,-1]]) #z projection spin operator*2 in same basis
s_y = np.array([[0,-1],[1,0]])
s_x = np.array([[0,1],[1,0]])
identity = np.array([[1,0],[0,1]]) #identity matrix in same basis


strength = 1 #overall multiplicative factor of interation strength
n = 10
sols_even = np.zeros((n-2)/2)
sols_odd = np.zeros((n-2)/2)
for i in range(n-2):
    J = np.zeros((i+3,i+3))
    for j in range(i+2):
        J[j][j+1] = 1
    J[0][i+2] = 1

    J *= strength


    B = 0 #magnetic field strength

    H = hamiltonian(i+3,J,B)

    #print H

    eigenvalues, eigenvectors = eigen(H)

    eigenvalues = eigenvalues.round(10)
    print i
    if (i+3)%2==0:
        sols_odd[i/2] = eigenvalues[0]/(i+3)
    else:
        sols_even[i/2] = eigenvalues[0]/(i+3)
x = np.arange(3,n+1,2)
y = np.arange(4,n+2,2)
x1 = np.linspace(3,n,100)


test, shite = curve_fit(lambda t,a,b: a*np.exp(b*t), x,(sols_even+0.44314718))
test2, shite2 = curve_fit(lambda t,a,b: a*np.exp(b*t), y,(sols_odd+0.44314718))

y1 = test[0]*np.exp(test[1]*x1)-0.44314718
y2 = test2[0]*np.exp(test2[1]*x1)-0.44314718
#y3 = test[0]*np.exp(test[1]*x1)*(np.sin(x1*(np.pi/2)))**2+test2[0]*np.exp(test2[1]*x1)*(np.cos(x1*(np.pi/2)))**2-0.44314718

plt.plot(x,sols_even,'o', label = 'Even data')
plt.plot(y,sols_odd,'o',label = 'Odd data')
plt.plot(x1,y1,label = 'Even fit')
plt.plot(x1,y2,label = 'Odd fit')
#plt.plot(x1,y3)
plt.plot(x1,np.zeros(100)-0.44314718,label = 'Bethe Ansatz')
plt.xlabel('N')
plt.ylabel("$\\frac{E_0}{N}$")
plt.title('Ground state energy per particle (all particles spin 1/2) with exponential fits')
plt.legend()
plt.show()
