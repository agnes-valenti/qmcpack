import numpy as np
import matplotlib.pyplot as plt

Kmax=0.1
n0_array=np.loadtxt("data.txt")[0,:]
num_densities=np.shape(n0_array)[0]
Nk = 41;
kx = np.linspace(-Kmax,Kmax,Nk)
ky = np.linspace(-Kmax,Kmax,Nk)
Area = (2*np.pi)**2/(kx[1]-kx[0])**2;
#print("Area", Area)
mu=0
#n0=0.1/312.097
#n0_array=np.array([0.1,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0])*n0 #np.array([0.1,0.4,0.7,1.0,1.4,1.8,2.0])*n0
#n0_array=np.array([2.0,1.9,1.8,1.7,1.6,1.5,1.4,1.3,1.2,1.1,1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1])/1.6525e3

#num_densities=np.shape(n0_array)[0]
Ky,Kx=np.meshgrid(kx,ky)
K1x = np.dot(np.reshape(Kx,((Nk*Nk,1))),np.ones((1,Nk*Nk)));
K1y =np.dot(np.reshape(Ky,(Nk*Nk,1)),np.ones((1,Nk*Nk)));
K2x = np.dot(np.ones((Nk*Nk,1)),np.reshape(Kx,(1,Nk*Nk)));
K2y = np.dot(np.ones((Nk*Nk,1)),np.reshape(Ky,(1,Nk*Nk)));
print(n0_array*312.097)
#plt.show()

#num_densities=num_densities//2
diag=np.zeros((1,4,num_densities))
polarized=np.zeros((1,4,num_densities))
full_reducible=np.zeros((10,4,num_densities))
symm=np.zeros((1,4,num_densities))

"""
diag1=diag[0,-1,:]
symm1=symm[0,-1,:]
full1=full[0,-1,:]
plt.figure()
plt.plot(n0_array,symm1-diag1, label='diag')
plt.plot(n0_array,symm1-full1)
plt.legend()
plt.show()
"""

print(diag[0,1,:])


taudiag=np.zeros((num_densities,4))
taufull=np.zeros((num_densities,10))
taufull_reducible=np.zeros((num_densities,6))

n0_array=n0_array*312.097*1e12

kmax_values=np.loadtxt("kmax_values.txt")
#kmax_values=np.loadtxt("polarized_spin/kmax_values.txt")

Eta=5.79

tau=-1

d = 100; # Distance to gate in units of unit cell size
epsilon_d = 10; # AlAs dielectric constant (GaAs? AlGaAs?)

qe = 1.60217662e-19; # electron charge
ke = 8.99e9;         # Coulomb constant
hbar=1.0545718*1e-34;  #hbar (in SI units)
a = 0.56605e-9;        # AlAs lattice spacing in nanometers

e_squared = ke*qe/a*1e3/epsilon_d; # e^2 in units of meV*a
me=9.1093837015*1e-31  #kg - umrechnen?? 1/a? meV?
mstar=0.457*me*qe/(hbar**2*1e3)*a*a
mx=Eta**(-0.5*tau)*mstar
my=Eta**(0.5*tau)*mstar

ky_ = lambda kx, E: np.sqrt(2.0*my*(E-kx**2/(2*mx)))
kym_ = lambda kx, E: -np.sqrt(2.0*my*(E-kx**2/(2*mx)))
kx_ = lambda ky, E: np.sqrt(2.0*mx*(E-ky**2/(2*my)))
kxm_ = lambda ky, E: -np.sqrt(2.0*mx*(E-ky**2/(2*my)))

kmax=0.5
kx=np.linspace(-kmax,kmax,1000)
ky=np.linspace(-kmax,kmax,1000)

km=kmax/2
E=km**2/(2.0*mx)+km**2/(2.0*my)

for i in range(4): #num_densities):
    Nk = 27;
    Kmax=kmax_values[i]

    kx = np.linspace(-Kmax,Kmax,Nk)
    ky = np.linspace(-Kmax,Kmax,Nk)
   
    Ky,Kx=np.meshgrid(kx,ky)

    #diag1[0,i]=symm[0,0,i]
    #full1[0,i]=symm[0,0,i]
    #full_reducible1[0,i]=symm[0,0,i]
    #symm1[0,i]=symm[0,0,i]
    diag0=diag[0,3,i] #/(symm[0,0,i])
    polarized0=polarized[0,3,i]
    #diag1=diag[1,3,i]

    symm0=symm[0,3,i]
    #diag1[1,i]=(diag[6,3,i])/(symm[0,0,i])
    #symm1[1,i]=np.min(symm[:,3,i])/(symm[0,0,i])
   

    j0=i #2*i+np.argmin(polarized[:,3,i])
    Ppolarized0=np.loadtxt("Pmatrix{}.txt".format(j0))

    #kvalues=np.sum(Pdiag0[:,:2],axis=1)
    kvalues=np.sum(Ppolarized0[:,:2],axis=1)
    #print(kvalues, np.shape(kvalues), 61*61, 41*41)
    k2D=np.zeros_like(Kx)
    k_it=0
    for l in range(Nk):
       for r in range(Nk):
           kx=Kx[l,r]
           ky=Ky[l,r]
           k2D[l,r]=kvalues[k_it]
           #print(kx,ky,kvalues[k_it])
           k_it+=1

    print(i,n0_array[i]*1e-12,np.sum(taudiag[i,:]))
    plt.figure()
    #plt.imshow(k2D)
    plt.contourf(Kx,Ky,k2D)
    kmax=Kmax
    kx=np.linspace(-kmax,kmax,1000)
    ky=np.linspace(-kmax,kmax,1000)
    for i in range(8):
      km=Kmax/2.5 #i*(kmax/4.0)
      E=i*(km**2/(2.0*mx)+km**2/(2.0*my))/20
      plt.plot(kx,ky_(kx,E),color='black')
      plt.plot(kx,kym_(kx,E),color='black')
      plt.plot(kx_(ky,E),ky,color='black')
      plt.plot(kxm_(ky,E),ky,color='black')
    plt.show()

    

    
