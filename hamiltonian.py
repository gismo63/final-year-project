import numpy as np

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
                print i+1,j+1
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







s_plus = np.array([[0,1],[0,0]]) #spin raising operator in the |up>=(1,0) and |down>=(0,1) basis
s_minus = np.array([[0,0],[1,0]]) #spin lowering operator in same basis
s_z = np.array([[1,0],[0,-1]]) #z projection spin operator *2 in same basis
identity = np.array([[1,0],[0,1]]) #identity matrix in same basis


strength = 1 #overall multiplicative factor of interation strength

#interaction strength where J[i][j] represents the strength of the interaction between particle i and particle j
#J = strength*np.array([[0,1,0,0,1],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[0,0,0,0,0]]) #J for 5 electron chain with periodic boundary conditions
J = strength*np.array([[0,1,0,1],[0,0,1,0],[0,0,0,1],[0,0,0,0]]) #J for 4 electron chain with periodic boundary conditions
#J = strength*np.array([[0,1,1],[0,0,1],[0,0,0]])
#J = np.array([[0,1],[1,0]]) #J for 2 electron system

n = len(J)
B = 0.5 #magnetic field strength

H = hamiltonian(n,J,B)

print H

eigenvalues, eigenvectors = eigen(H)

eigenvalues = eigenvalues.round(10)

print np.sort(eigenvalues)


eigenvectors = eigenvectors.round(10)



g_state = eigenvectors[:,0]

print g_state

Sz = expect_sz(n, g_state)/2 #divide by 2 to account for factor of 1/2 missing from s_z definition

print Sz.round(10)
