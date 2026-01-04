import numpy as np
import scipy.special
import matplotlib.pyplot as plt


d = 350; # Distance to gate in units of unit cell size
epsilon_d = 10; # AlAs dielectric constant (GaAs? AlGaAs?)

qe = 1.60217662e-19; # electron charge
ke = 8.99e9;         # Coulomb constant
hbar=1.0545718*1e-34;  #hbar (in SI units)
a = 0.56605e-9;        # AlAs lattice spacing in nanometers

e_squared = ke*qe/a*1e3/epsilon_d;

wf=lambda x,xprime,y,yprime,k1x,k1y: (np.exp(-k1x*x*1j)*np.exp(-k1y*y*1j)-np.exp(-k1x*xprime*1j)*np.exp(-k1y*yprime*1j))*(np.exp(k1x*x*1j)*np.exp(k1y*y*1j)-np.exp(k1x*xprime*1j)*np.exp(k1y*yprime*1j))

U=lambda x,y,M: np.exp(-(x*x+y*y)/M)

Uhat=lambda kx,ky,M: M*np.pi*np.exp(-M*(kx*kx+ky*ky)/4)

#f3_2D= lambda x,xprime,y,yprime,L,M,k1x,k1y: (np.exp(-((x-xprime)*(x-xprime)+ (y-yprime)*(y-yprime) )/M)*(np.exp(-k1x*x*1j)*np.exp(-k1y*y*1j)-np.exp(-k1x*xprime*1j)*np.exp(-k1y*yprime*1j))*(np.exp(k1x*x*1j)*np.exp(k1y*y*1j)-np.exp(k1x*xprime*1j)*np.exp(k1y*yprime*1j)))/2.0 #already divided by norm. Eigentlich: /2L^4, 1/L^4 durch MC sampling

#f3_2D= lambda x,xprime,y,yprime,L,M,k1x,k1y: (( U((x-xprime-L),(y-yprime-L),M)+ U((x-xprime-L),(y-yprime),M) + U((x-xprime-L),(y-yprime+L),M)  + U((x-xprime),(y-yprime-L),M) + U((x-xprime),(y-yprime),M)  + U((x-xprime),(y-yprime+L),M)  + U((x-xprime+L),(y-yprime-L),M)  + U((x-xprime+L),(y-yprime),M)  + U((x-xprime+L),(y-yprime+L),M) )*(np.exp(-k1x*x*1j)*np.exp(-k1y*y*1j)-np.exp(-k1x*xprime*1j)*np.exp(-k1y*yprime*1j))*(np.exp(k1x*x*1j)*np.exp(k1y*y*1j)-np.exp(k1x*xprime*1j)*np.exp(k1y*yprime*1j)))*e_squared/2.0 #already divided by norm. Eigentlich: /2L^4, 1/L^4 durch MC sampling 

f3_2D= lambda x,xprime,y,yprime,L,M,k1x,k1y:(( U((x-xprime),(y-yprime),M) )*(np.exp(-k1x*x*1j)*np.exp(-k1y*y*1j)-np.exp(-k1x*xprime*1j)*np.exp(-k1y*yprime*1j))*(np.exp(k1x*x*1j)*np.exp(k1y*y*1j)-np.exp(k1x*xprime*1j)*np.exp(k1y*yprime*1j)))*e_squared/2.0

#f3_2D= lambda x,xprime,y,yprime,L,M,k1x,k1y:(( U((x-xprime-L),(y-yprime-L),M)+ U((x-xprime-L),(y-yprime),M) + U((x-xprime-L),(y-yprime+L),M)  + U((x-xprime),(y-yprime-L),M) + U((x-xprime),(y-yprime),M)  + U((x-xprime),(y-yprime+L),M)  + U((x-xprime+L),(y-yprime-L),M)  + U((x-xprime+L),(y-yprime),M)  + U((x-xprime+L),(y-yprime+L),M) )*(np.exp(-k1x*x*1j)*np.exp(-k1y*y*1j)-np.exp(-k1x*xprime*1j)*np.exp(-k1y*yprime*1j))*(np.exp(k1x*x*1j)*np.exp(k1y*y*1j)-np.exp(k1x*xprime*1j)*np.exp(k1y*yprime*1j)))*e_squared/2.0 #already divided by norm. Eigentlich: /2L^4, 1/L^4 durch MC sampling 

f3_2D_MCMC= lambda x,xprime,y,yprime,L,M,k1x,k1y: e_squared*U((x-xprime),(y-yprime),M)

#( U((x-xprime-L),(y-yprime-L),M)+ U((x-xprime-L),(y-yprime),M) + U((x-xprime-L),(y-yprime+L),M)  + U((x-xprime),(y-yprime-L),M) + U((x-xprime),(y-yprime),M)  + U((x-xprime),(y-yprime+L),M)  + U((x-xprime+L),(y-yprime-L),M)  + U((x-xprime+L),(y-yprime),M)  + U((x-xprime+L),(y-yprime+L),M) )*e_squared #already divided by norm. Eigentlich: /2L^4, 1/L^4 durch MC sampling 


f4_2D= lambda dx,dy,x,xprime,y,yprime,L,M,k1x,k1y: e_squared*(np.exp(-((dx)*(dx)+ (dy)*(dy) )/M)*(np.exp(-k1x*x*1j)*np.exp(-k1y*y*1j)-np.exp(-k1x*xprime*1j)*np.exp(-k1y*yprime*1j))*(np.exp(k1x*x*1j)*np.exp(k1y*y*1j)-np.exp(k1x*xprime*1j)*np.exp(k1y*yprime*1j)))/2.0 #already divided by norm. Eigentlich: /2L^4, 1/L^4 durch MC sampling

f4_2D_MCMC= lambda dx,dy,L,M,k1x,k1y: e_squared*U((dx),(dy),M)

norm= lambda x,xprime,y,yprime,L,M,k1x,k1y: ((np.exp(-k1x*x*1j)*np.exp(-k1y*y*1j)-np.exp(-k1x*xprime*1j)*np.exp(-k1y*yprime*1j))*(np.exp(k1x*x*1j)*np.exp(k1y*y*1j)-np.exp(k1x*xprime*1j)*np.exp(k1y*yprime*1j)))*L**4 # L^4 for MC sampling (norm=2L^4)

f3_k= lambda k1x,k1y,L,M: (2*Uhat(0,0,M)-Uhat(-k1x,-k1y,M)-Uhat(k1x,k1y,M))*e_squared/(2.0*L**2)  #already divided by norm. E

f=lambda L,M: np.sqrt(M*np.pi)/L*scipy.special.erf(L/np.sqrt(M))+M*np.exp(-L**2/M)/L**2-M/L**2
f2= lambda L,M: 1.0/L*np.sqrt(M*np.pi)
f3= lambda x,xprime,L,M: np.exp(-(x-xprime)*(x-xprime)/M)+np.exp(-(x-xprime-L)*(x-xprime-L)/M)+np.exp(-(x-xprime+L)*(x-xprime+L)/M)


#f=lambda L,M: np.sqrt(M*np.pi)/L*scipy.special.erf(L/np.sqrt(M))+M*np.exp(-L**2/M)/L**2-M/L**2
#f2= lambda L,M: 1.0/L*np.sqrt(M*np.pi)
#g3= lambda x,xprime,k,L,M: np.exp(-(x-xprime)*(x-xprime)/M)*(1-np.cos(k*(x-xprime)))+np.exp(-(x-xprime-L)*(x-xprime-L)/M)*(1-np.cos(k*(x-xprime-L)))+np.exp(-(x-xprime+L)*(x-xprime+L)/M)*(1-np.cos(k*(x-xprime+L)))
g3= lambda x,xprime,k,L,M: np.exp(-(x-xprime)*(x-xprime)/M)*(1-np.cos(k*(x-xprime)))+np.exp(-(x-xprime-L)*(x-xprime-L)/M)*(1-np.cos(k*(x-xprime-L)))+np.exp(-(x-xprime+L)*(x-xprime+L)/M)*(1-np.cos(k*(x-xprime+L)))

psisquared= lambda x,y,xprime,yprime,M,L,k1x,k1y: ((np.exp(-k1x*x*1j)*np.exp(-k1y*y*1j)-np.exp(-k1x*xprime*1j)*np.exp(-k1y*yprime*1j))*(np.exp(k1x*x*1j)*np.exp(k1y*y*1j)-np.exp(k1x*xprime*1j)*np.exp(k1y*yprime*1j)))/(2*L**4)

L1=np.sqrt(15791.3670417)
M1=2
N=10000
lambda_=2/M1
N=20000 #4000000

k1_x=-0.05 #2.0*np.pi*3/L1
k1_y=-0.05 #2.0*np.pi*3/L1




samples_x=np.random.rand(N)*L1 
samples_xp=np.random.rand(N)*L1
samples_y=np.random.rand(N)*L1 
samples_yp=np.random.rand(N)*L1 
#dx=samples_x-samples_xp
#dy=samples_y-samples_yp

k1_x=-0.05 #2.0*np.pi*3/L1
k1_y=-0.05 #2.0*np.pi*3/L1
print(f(L1,M1), f2(L1,M1))
print(1.0/N*np.sum(f3(samples_x,samples_xp,L1,M1)))
print(L1*L1*2.0/N*np.sum(g3(samples_x,samples_xp,k1_x,L1,M1)))


expvalue=1.0/N*np.sum(f3_2D(samples_x,samples_xp, samples_y,samples_yp,L1,M1,k1_x,k1_y))
norm=1.0/N*np.sum(norm(samples_x,samples_xp, samples_y,samples_yp,L1,M1,k1_x,k1_y))
print("expvalue sampled without importance sampling, only one cell: ",expvalue," norm: ",norm) # "expvalue/norm: ",expvalue/norm)
print(f3_k(k1_x,k1_y,L1,M1))


dx=samples_x/L1-samples_xp/L1
dy=samples_y/L1-samples_yp/L1
dx     = L1 * (dx - np.round_(dx));  #finding closest distance like this!!!!
dy     = L1 * (dy - np.round_(dy));
r2=np.sqrt(dx*dx+dy*dy)
print("Wigner-Seitz: ",1.0/np.shape(r2)[0]*np.sum(f4_2D(dx,dy,samples_x, samples_xp, samples_y, samples_yp,L1,M1,k1_x,k1_y)))
"""


samples_x=np.random.rand(N)
samples_xp=np.random.rand(N)
samples_y=np.random.rand(N)
samples_yp=np.random.rand(N)
dx=samples_x-samples_xp
dy=samples_y-samples_yp
dx     = L1 * (dx - np.round_(dx));  #finding closest distance like this!!!!
dy     = L1 * (dy - np.round_(dy));
r=np.sqrt(dx*dx+dy*dy)
print(1.0/N*np.sum(f4_2D(dx,dy,samples_x*L1,samples_xp*L1,samples_y*L1, samples_yp*L1 ,L1,M1,k1_x,k1_y)))
print(1.0/N*np.sum(f3_2D(samples_x*L1,samples_xp*L1,samples_y*L1, samples_yp*L1 ,L1,M1,k1_x,k1_y)))

samples_x=np.loadtxt("DistancesPtcl0.txt")[:,0]
samples_y=np.loadtxt("DistancesPtcl0.txt")[:,1]
samples_xp=np.loadtxt("DistancesPtcl1.txt")[:,0]
samples_yp=np.loadtxt("DistancesPtcl1.txt")[:,1]
dx=samples_x-samples_xp
dy=samples_y-samples_yp
dx     = L1 * (dx - np.round_(dx));  #finding closest distance like this!!!!
dy     = L1 * (dy - np.round_(dy));
r2=np.sqrt(dx*dx+dy*dy)
print(1.0/N*np.sum(f3_2D(samples_x,samples_xp,samples_y, samples_yp ,L1,M1,k1_x,k1_y)))
#print(1.0/np.shape(r2)[0]*np.sum(f4_2D(r2,L1,M1,k1_x,k1_y)))


plt.figure()
n, bins, patches = plt.hist(r, 100, density=True, facecolor='g', alpha=0.75)
plt.savefig("random.png")

#r2=np.loadtxt("distances.txt")
plt.figure()
n, bins, patches = plt.hist(r, 100, density=True, facecolor='g', alpha=0.75)
n, bins, patches = plt.hist(r2, 100, density=True, facecolor='b', alpha=0.25)
plt.savefig("random2.png")
plt.show()
"""
print("L1: ",L1)
samples_x=np.loadtxt("DistancesPtcl0.txt")[:,0]%L1
samples_y=np.loadtxt("DistancesPtcl0.txt")[:,1]%L1
samples_xp=np.loadtxt("DistancesPtcl1.txt")[:,0]%L1
samples_yp=np.loadtxt("DistancesPtcl1.txt")[:,1]%L1

dx=samples_x-samples_xp
dy=samples_y-samples_yp
print(samples_x)
print(samples_xp)
plt.figure()
plt.hist2d(dx,dy, 100, density=True, facecolor='g', alpha=0.75)
plt.show()

dx=samples_x/L1-samples_xp/L1
dy=samples_y/L1-samples_yp/L1
dx     = L1 * (dx - np.round_(dx));  #finding closest distance like this!!!!
dy     = L1 * (dy - np.round_(dy));
r=np.sqrt(dx*dx+dy*dy)
#print(1.0/np.shape(r2)[0]*np.sum(f4_2D(r2,L1,M1,k1_x,k1_y)))
print("qmcpack Wigner-Seitz: ",1.0/np.shape(r)[0]*np.sum(f4_2D_MCMC(dx,dy ,L1,M1,k1_x,k1_y)))

#plt.figure()
#n, bins, patches = plt.hist(r, 100, density=True, facecolor='g', alpha=0.75)
#plt.savefig("random.png")
#plt.show()
print(wf(-6.6972001261e+01,-1.7804086421e+02,2.9090987715e+02,2.9947049250e+02,-0.05,-0.05 )/wf(-6.6961644818e+01,-1.7804086421e+02,2.9119733843e+02,2.9947049250e+02,-0.05,-0.05))

samples_x=np.random.rand(N)*L1 
samples_xp=np.random.rand(N)*L1 
samples_y=np.random.rand(N)*L1 
samples_yp=np.random.rand(N)*L1 
dx=samples_x-samples_xp
dy=samples_y-samples_yp

plt.figure()
plt.hist2d(dx,dy, 100, density=True, facecolor='g', alpha=0.75)
plt.show()

Nx=35
Ny=35
ValuesX=np.linspace(0,L1,Nx)
ValuesY=np.linspace(0,L1,Ny)
xx,yy,xpxp,ypyp=np.meshgrid(ValuesX,ValuesY,ValuesX,ValuesY,indexing='ij')
xx=xx.flatten()
yy=yy.flatten()
xpxp=xpxp.flatten()
ypyp=ypyp.flatten()

#print(np.shape(xx),np.shape(yy))
Psisquared=psisquared(xx,yy,xpxp,ypyp,M1,L1,k1_x,k1_y)
print(np.shape(Psisquared),np.sum(Psisquared)*(L1/Nx)**2*(L1/Ny)**2)

Numsamples=10000
prob=np.real(np.reshape(Psisquared/(np.sum(Psisquared)),(np.shape(Psisquared)[0])))
print(np.shape(Psisquared)[0],np.shape(prob))
a=np.random.choice(np.shape(Psisquared)[0],Numsamples,p=prob)

dx=xx[a]-xpxp[a]
dy=yy[a]-ypyp[a]
print(dx)

plt.figure()
plt.hist2d(dx,dy, 80, density=True, facecolor='g', alpha=0.75)
plt.show()
print(1.0/Numsamples*np.sum(f3_2D_MCMC(xx[a],xpxp[a],yy[a], ypyp[a] ,L1,M1,k1_x,k1_y)))
#print(a,xx[a],yy[a],xpxp[a],ypyp[a],Psisquared[a],prob[a])



def binning_step(data):
    ''' performs a single binning step
    
    Parameters
    ----------
    
    - data=[Q^(l-1)_1 ... Q^(l-1)_N]
      array of length N=2M, containing N measurements Q^(l-1)_i (e.g. magnetization m^(l-1)_i) 
      in the (l-1)'th level of the binning analysis
    
    
    Returns
    --------
    
    - new_data=[Q^(l)_1, ... Q^(l)_N]
      array of length M (lth level array of the binning analysis)
      
    - new_error: double
      error estimate of lth level of the binning analysis (eq. (6) in exercise sheet)
    
    '''
    average = np.mean(data)
    new_data = []
    for i in range(int(len(data)/10)):
        new_data.append((np.mean(data[10*i:10*i+10])))
    new_error = 0.
    for d in new_data:
        new_error += (average - d)**2
    new_error = np.sqrt(new_error/(len(new_data)*(len(new_data)-1)))
    return new_data, new_error

def binning(data, num_levels):
    ''' bins the data up to num_level
     
    Parameters
    ----------
    
    - data=[Q_1 ... Q_N]:
      array of length N, containing N measurements Q_i (e.g. magnetization m_i) 
      
    - num_levels: int
      number of binning levels to be computed
      
      
    Returns
    -------
    
    - errors: array, dtype=double
      array of length num_levels+1, contains error estimates for each level
    
    
    '''
    errors = []
    #first, calculate the level-0 error
    average = np.mean(data)
    error = 0.
    for d in data:
        error += (average - d)**2
    error = np.sqrt(error/(len(data)*(len(data)-1)))
    errors.append(error)
    new_data = data
    for i in range(num_levels):
        new_data, new_error = binning_step(new_data)
        errors.append(new_error)
    return errors


errors=binning(f3_2D_MCMC(xx[a],xpxp[a],yy[a], ypyp[a] ,L1,M1,k1_x,k1_y),3)
print(errors)
"""
Xsamples=1
Ysamples=1
Xpsamples=1
Ypsamples=1

Numsamples=N
for i in range(Nx):
   for j in range(Ny):
      for k in range(Nx):
         for l in range(Ny):
            Xsamples=np.hstack((Xsamples,np.ones(int(Numsamples*Psisquared[i,j,k,l]))*ValuesX[i]))
            Ysamples=np.hstack((Xsamples,np.ones(int(Numsamples*Psisquared[i,j,k,l]))*ValuesY[j]))
            Xpsamples=np.hstack((Xsamples,np.ones(int(Numsamples*Psisquared[i,j,k,l]))*ValuesX[k]))
            Ypsamples=np.hstack((Xsamples,np.ones(int(Numsamples*Psisquared[i,j,k,l]))*ValuesY[l]))

dx=Xsamples-Xpsamples
dy=Ysamples-Ypsamples
plt.figure()
plt.hist2d(dx,dy, 100, density=True, facecolor='g', alpha=0.75)
plt.show()
"""

"""
samples_x=np.random.rand(N)
samples_xp=np.random.rand(N)
samples_y=np.random.rand(N)
samples_yp=np.random.rand(N)
dx=samples_x-samples_xp
dy=samples_y-samples_yp
dx     = L1 * (dx - np.round_(dx));  #finding closest distance like this!!!!
dy     = L1 * (dy - np.round_(dy));
r=np.sqrt(dx*dx+dy*dy)
print(1.0/N*np.sum(f4_2D(r,L1,M1)))

r2=np.loadtxt("distances.txt")
print(1.0/np.shape(r2)[0]*np.sum(f4_2D(r2,L1,M1)))
#print(np.mean(r),np.mean(r2),np.mean(r*r)-np.mean(r)**2,np.mean(r2*r2)-np.mean(r2)**2,np.shape(r2))
##r=np.sqrt((samples_x-samples_xp)*(samples_x-samples_xp)+(samples_y-samples_yp)*(samples_y-samples_yp))
#print(r-r2)


plt.figure()
n, bins, patches = plt.hist(r, 100, density=True, facecolor='g', alpha=0.75)
plt.savefig("random.png")

r2=np.loadtxt("distances.txt")
plt.figure()
n, bins, patches = plt.hist(r2, 100, density=True, facecolor='g', alpha=0.75)
plt.savefig("random2.png")
plt.show()
#samples_xnew=np.random.rand(N)*2*L1-L1
#print(1.0/N*np.sum(func1(samples_xnew,L1)))
#print(1.0/N*np.sum(func2(samples_x,samples_xp,L1)))
"""


