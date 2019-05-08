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


n = 3 # number of spin sites

strength = 1 #overall multiplicative factor of interation strength

for k in range(3):
    J = np.zeros((n,n))

    norm = []
    for j in range(n):
        i=0
        while i<j:
            J[i][j] = np.random.normal(0,1) # random number with normal distribution centered on 0, standard deviation 1
            norm.append(J[i][j]**2)
            i+=1
    norm_sum = np.sum(norm)
    J_std = np.std(norm)
    J /= np.sqrt(norm_sum)
    #print J_std
    #print J



    """
    strength = -1 #overall multiplicative factor of interation strength
    J = np.zeros((n,n))
    for i in range(n-1):
        J[i][i+1] = 1
    J[0][n-1] = 1

    J= J*strength
    J = J/np.sqrt(n)
    """




    B = np.random.normal(0,1) #magnetic field strength
    B = 0

    H = hamiltonian(n,J,B)

    #print H

    eigenvalues, eigenvectors = eigen(H)

    eigenvalues = eigenvalues.round(10)

    #print (eigenvalues)


    eigenvectors = eigenvectors.round(8)

    b,c  = np.unique(eigenvalues,return_counts=True)
    #print (np.sort(c))

    d,e  = np.unique(c,return_counts=True)
    #print (d)
    #print (e)

    g_state = eigenvectors[:,0]

    print (eigenvectors[:,0])
    print (eigenvectors[:,1])
    print (eigenvectors[:,2])
    print (eigenvectors[:,3])
    print (eigenvectors[:,4])
    print (eigenvectors[:,5])
    print (eigenvectors[:,6])
    print (eigenvectors[:,7])
    print ("\n")
    #print eigenvectors[:,1]
    #print (np.sum(eigenvalues))
    #print (np.sum(eigenvalues**2))
    #print (np.sum(eigenvalues**2)-(0.75*(2**(n-2))))/(B*B)
    #E_std = np.std(eigenvalues**2)
    #print E_std*J_std

    Sz = expect_sz(n, g_state)/2 #divide by 2 to account for factor of 1/2 missing from s_z definition
    Sy = expect_sy(n, g_state)/2 # there is also a factor of i missing from this expectation value
    Sx = expect_sx(n, g_state)/2

    """
    for i in range(2**n):
        print eigenvalues[i]
        for j in range(2**n):
            if eigenvectors[j][i]!=0:
                print str(bin(j))[2:] + ' ' + str(int(np.signbit(eigenvectors[j][i])))
        print ' '
    """
    b,c  = np.unique(eigenvalues,return_counts=True)
    #print (b)
    #print (c)
    #print (Sz.round(10))
    #print Sy.round(10)
    #print Sx.round(10)
