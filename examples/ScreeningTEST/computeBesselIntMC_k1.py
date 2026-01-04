import numpy as np 
import matplotlib.pyplot as plt

d=200
r=np.loadtxt("Uscreenedto1.txt")[0,:]
U=np.loadtxt("Uscreenedto1.txt")[1,:]

r2=np.loadtxt("Uscreened1to5001.txt")[0,:]
U2=np.loadtxt("Uscreened1to5001.txt")[1,:]

r=np.hstack((r,r2))
U=np.hstack((U,U2))*4.0/d


Kmax = 0.05;
Nk = 5;
kx = np.linspace(-Kmax,Kmax,Nk)
ky = np.linspace(-Kmax,Kmax,Nk)
Area = (2*np.pi)**2/(kx[1]-kx[0])**2;
print("Area: ",Area," L1: ",np.sqrt(Area))
epsilon_d = 10; # AlAs dielectric constant (GaAs? AlGaAs?)

qe = 1.60217662e-19; # electron charge
ke = 8.99e9;         # Coulomb constant
hbar=1.0545718*1e-34;  #hbar (in SI units)
a = 0.56605e-9;        # AlAs lattice spacing in nanometers

e_squared = ke*qe/a*1e3/epsilon_d; # e^2 in units of meV*a
#Area=15791.3670417
L1=np.sqrt(Area) #np.sqrt(Area)
print("L1: ",L1)

#Uhat=lambda kx,ky: 2.0*np.pi*np.exp(-(kx*kx+ky*ky)/2)
def Uhat(kx,ky):
  q=np.sqrt((kx)**2+(ky)**2+(1e-8)**2)
  return 2.0*np.pi*np.tanh(q*d/2)/q

def U_k1(dx,dy,x,xprime,y,yprime,k1x,k1y): 
  r_ind=np.abs(r-np.sqrt((dx)**2+(dy)**2)).argmin()
  #print(r_ind,r[r_ind],U[r_ind], np.exp(-(r[r_ind]**2)/2))
  z=(U[r_ind]*(np.exp(-k1x*x*1j)*np.exp(-k1y*y*1j)- np.exp(-k1x*xprime*1j)*np.exp(-k1y*yprime*1j))*(np.exp(k1x*x*1j)*np.exp(k1y*y*1j)-np.exp(k1x*xprime*1j)*np.exp(k1y*yprime*1j)))*e_squared/2.0
  return z

Uhat_k1= lambda k1x,k1y: (2*Uhat(0,0)-Uhat(-k1x,-k1y)-Uhat(k1x,k1y))*e_squared/(2.0*L1**2)

k1_x=np.pi*2/L1*2 #-0.05 #np.pi*2/L1*3 #-0.05 #np.pi*2/L1*3
k1_y=np.pi*2/L1*1 #-0.05 #np.pi*2/L1*3 #-0.05 #np.pi*2/L1*3

print("kx:",k1_x, "ky: ",k1_y)

ind=np.abs(r-L1).argmin()
diff=r[1:ind]-r[:ind-1]
#Int=4*e_squared*np.sum((U[1:ind]+U[:ind-1])/2.0*diff)/d
#Int=np.sum(U[:-1]*diff)
#print(Int/Area)
#d=100
#eps=1e-8
print(Uhat_k1(k1_x,k1_y))

N=20000
samples_x=np.random.rand(N)*L1 
samples_xp=np.random.rand(N)*L1
samples_y=np.random.rand(N)*L1 
samples_yp=np.random.rand(N)*L1 
#dx=samples_x-samples_xp
#dy=samples_y-samples_yp
dx=samples_x/L1-samples_xp/L1
dy=samples_y/L1-samples_yp/L1
dx     = L1 * (dx - np.round_(dx));  #finding closest distance like this!!!!
dy     = L1 * (dy - np.round_(dy));
Uges=np.zeros(N)
for i in range(N):
  Uges[i]=U_k1(dx[i],dy[i],samples_x[i],samples_xp[i],samples_y[i],samples_yp[i],k1_x,k1_y) #\
          #+U_k1(dx[i]-L1,dy[i]-L1,samples_x[i],samples_xp[i],samples_y[i],samples_yp[i],k1_x,k1_y) \
          #+U_k1(dx[i]-L1,dy[i],samples_x[i],samples_xp[i],samples_y[i],samples_yp[i],k1_x,k1_y) \
          #+U_k1(dx[i]-L1,dy[i]+L1,samples_x[i],samples_xp[i],samples_y[i],samples_yp[i],k1_x,k1_y) \
          #+U_k1(dx[i],dy[i]-L1,samples_x[i],samples_xp[i],samples_y[i],samples_yp[i],k1_x,k1_y) \
          #+U_k1(dx[i],dy[i]+L1,samples_x[i],samples_xp[i],samples_y[i],samples_yp[i],k1_x,k1_y) \
          #+U_k1(dx[i]+L1,dy[i]-L1,samples_x[i],samples_xp[i],samples_y[i],samples_yp[i],k1_x,k1_y) \
          #+U_k1(dx[i]+L1,dy[i],samples_x[i],samples_xp[i],samples_y[i],samples_yp[i],k1_x,k1_y) \
          #+U_k1(dx[i]+L1,dy[i]+L1,samples_x[i],samples_xp[i],samples_y[i],samples_yp[i],k1_x,k1_y) 

  
#print(r2_indices)
print(1.0/N*np.sum(Uges))
#print(Ucart[-1])
