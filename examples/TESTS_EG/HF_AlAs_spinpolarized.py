#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import numpy as np
import scipy.optimize
#import matplotlib.pyplot as plt



# # HF trilayer graphene


sym_ind=5
# In[34]:
param_mode =0
if param_mode==-1:
   sym_ind=1

if param_mode==-1:   #% Diagonal elements only
   tind = np.array([[0], [0]]); 
elif param_mode==0:   #% Diagonal elements only
   tind = np.array([[0, 1, 2, 3], [0, 1, 2, 3]]); 
elif param_mode==1: #% Only symmetry distinct elements
   tind = np.array([[0, 1, 2, 3, 2, 3],[0, 1, 2, 3, 0, 1]]);    
elif param_mode==2: # % All elements
   tind = np.array([[0, 1, 2, 3, 1, 2, 3, 2, 3, 3],[0, 0, 0, 0, 1, 1, 1, 2, 2, 3]]);
else:
   sys.exit("param.mode must have a legal value.")
 
tind=np.transpose(tind)

sx=np.array([[0,1],[1,0]])+0j
sy=np.array([[0,-1j],[1j,0]])+0j
sz=np.array([[1,0],[0,-1]])+0j 

Temperature=0.2

Kmax = 0.5;

Nk = 41;
kx = np.linspace(-Kmax,Kmax,Nk)
ky = np.linspace(-Kmax,Kmax,Nk)
Area = (2*np.pi)**2/(kx[1]-kx[0])**2;
#print("Area", Area)
mu=0
n0=0.1/312.097
n0_array=np.array([1.0,1.1,1.2])*n0  #np.array([0.1,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0])*n0
deltaN=n0_array[2]-n0_array[1]
 #np.array([0.1,0.4,0.7,1.0,1.4,1.8,2.0])*n0
#n0_array=np.array([2.0,1.9,1.8,1.7,1.6,1.5,1.4,1.3,1.2,1.1,1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1])/1.6525e3
kmax_values=np.zeros_like(n0_array)
for i in range(np.shape(kmax_values)[0]):
    if n0_array[i]/n0<0.1:
       kmax_values[i]=0.1
    elif n0_array[i]/n0<3.0:
       kmax_values[i]=0.15
    elif n0_array[i]/n0<6.0:
       kmax_values[i]=0.2
    elif n0_array[i]/n0<10.0:
       kmax_values[i]=0.35
    elif n0_array[i]/n0<15.0:
       kmax_values[i]=0.45
    elif n0_array[i]/n0<30.0:
       kmax_values[i]=0.6
    else:
       kmax_values[i]=0.7
      
      


num_densities=np.shape(n0_array)[0]
Ky,Kx=np.meshgrid(kx,ky)
K1x = np.dot(np.reshape(Kx,((Nk*Nk,1))),np.ones((1,Nk*Nk)));
K1y =np.dot(np.reshape(Ky,(Nk*Nk,1)),np.ones((1,Nk*Nk)));
K2x = np.dot(np.ones((Nk*Nk,1)),np.reshape(Kx,(1,Nk*Nk)));
K2y = np.dot(np.ones((Nk*Nk,1)),np.reshape(Ky,(1,Nk*Nk)));
kradius=np.zeros(Nk*Nk)

dKx=K1x - K2x
dKy=K1y - K2y
dKx2=dKx*dKx
dKy2=dKy*dKy
q = np.sqrt(dKx2+dKy2 + (1e-8)**2);
# In[35]:
Eta=5.79

d = 100; # Distance to gate in units of unit cell size
epsilon_d = 10; # AlAs dielectric constant (GaAs? AlGaAs?)

qe = 1.60217662e-19; # electron charge
ke = 8.99e9;         # Coulomb constant
hbar=1.0545718*1e-34;  #hbar (in SI units)
a = 0.56605e-9;        # AlAs lattice spacing in nanometers

e_squared = ke*qe/a*1e3/epsilon_d; # e^2 in units of meV*a
me=9.1093837015*1e-31  #kg - umrechnen?? 1/a? meV?
mstar=0.457*me*qe/(hbar**2*1e3)*a*a  #mstar in units of (1/(meV*a^2)) (from hbar) -> 1/mstar*k^2->meV  (a needed here?)

def make_EA():
   h_uncorr=np.zeros((Nk,Nk,6,6,4,4))+0j
   k_it=0

   
 
   E=np.zeros((Nk*Nk,2))
   A=np.zeros((Nk*Nk,6,4))+0j
   E2=np.zeros((Nk*Nk,2))

   for i in range(Nk):
       for j in range(Nk):
           for Tau in range(2):
               for s in range(2):
                   tau=(2*Tau-1)
                   kx=Kx[i,j]
                   ky=Ky[i,j]
                   E[k_it,Tau]=Eta**(0.5*tau)/(2.0*mstar)*kx**2+Eta**(-0.5*tau)/(2.0*mstar)*ky**2
                   kradius[k_it]=np.sqrt(kx**2+ky**2)
                   #print(E[k_it,Tau])
           k_it+=1
   return E


Earray=make_EA()
print(np.max(Earray),np.min(Earray))

values_2D=np.zeros((Nk,Nk,2))
k_it=0
for i in range(Nk):
    for j in range(Nk):
        values_2D[i,j,:]=Earray[k_it,:]
        k_it+=1

#plt.figure()
#plt.contourf(Kx,Ky,values_2D[:,:,0])
##plt.imshow(values_2D[:,:,2])
#plt.axis('equal')
#plt.colorbar()
#plt.show()

#plt.figure()
#plt.contourf(Kx,Ky,values_2D[:,:,1])
##plt.imshow(values_2D[:,:,2])
#plt.axis('equal')
#plt.colorbar()
#plt.show()

def VC(k1,k2):
    q=np.sqrt((k1[0]-k2[0])**2+(k1[1]-k2[1])**2+(1e-8)**2)
    Vc=2.0*np.pi*e_squared*np.tanh(d*q)/q   #(2*eps*q)
    return Vc
   
def VC_array(q):
    return 2.0*np.pi*e_squared*np.tanh(d*q)/q  

#print(VC(np.array([1,0]),np.array([1,0])))
#print(VC(np.array([1,0]),np.array([1,0]))*n0)


def get_P(Thf, mu_h=0):
    P=np.zeros((Nk*Nk,np.shape(tind)[0]))+0j
    for k_it in range(Nk*Nk):
        if sym_ind==1:
           hk=np.real(E[k_it,0]+Thf[k_it,0])
           P[k_it,0]=np.heaviside(mu_h-hk,1) #1.0/(1.0+np.exp((hk-mu_h)/Temperature))
        elif sym_ind==5:
           hk=np.real(E[k_it,0]+Thf[k_it,0])
           P[k_it,0]=np.heaviside(mu_h-hk,1) #1.0/(1.0+np.exp((hk-mu_h)/Temperature))
           P[k_it,1]=np.heaviside(mu_h-hk,1) #1.0/(1.0+np.exp((hk-mu_h)/Temperature))
           P[k_it,2]=0
           P[k_it,3]=0
        elif sym_ind==6:
           hk=np.real(E[k_it,0]+Thf[k_it,0])
           #print(mu_h, hk)
           P[k_it,0]=np.heaviside(mu_h-hk,1) #1.0/(1.0+np.exp((hk-mu_h)/Temperature))
           P[k_it,1]=0
           P[k_it,2]=0
           P[k_it,3]=0
        else:
           alpha_inds=np.array([0,0,1,1])
           Hk=np.diag(E[k_it,alpha_inds])+0j
           for it in range(np.shape(tind)[0]):
               alpha=tind[it,0]
               beta=tind[it,1]
               Hk[alpha,beta]=Hk[alpha,beta]+Thf[k_it,it]
               if alpha!=beta:
                  Hk[beta,alpha]=np.conj(Hk[alpha,beta])
           vals,vecs=np.linalg.eigh(Hk)
           Ediag=np.diag(vals)
           #print("energy difference", vals-mu_h)
           #nfermi=np.diag(1.0/(1.0+np.exp((vals-mu_h)/Temperature)))
           nfermi=np.diag(np.heaviside(mu_h-vals,1))
           Pmatrix=np.dot(np.dot(vecs,nfermi),np.conjugate(np.transpose(vecs)))
           for it in range(np.shape(tind)[0]):
               alpha=tind[it,0]
               beta=tind[it,1]
               P[k_it,it]=Pmatrix[alpha,beta]
    return P

def densities_test(Thf,mu_h):
    P=np.zeros((Nk*Nk,np.shape(tind)[0]))+0j
    for k_it in range(Nk*Nk):
        if sym_ind==1:
           hk=np.real(E[k_it,0]+Thf[k_it,0])
           P[k_it,0]=np.heaviside(mu_h-hk,1) #1.0/(1.0+np.exp((hk-mu_h)/Temperature))
        elif sym_ind==5:
           hk=np.real(E[k_it,0]+Thf[k_it,0])
           P[k_it,0]=np.heaviside(mu_h-hk,1) #1.0/(1.0+np.exp((hk-mu_h)/Temperature))
           P[k_it,1]=np.heaviside(mu_h-hk,1) #1.0/(1.0+np.exp((hk-mu_h)/Temperature))
           P[k_it,2]=0
           P[k_it,3]=0
        elif sym_ind==6:
           hk=np.real(E[k_it,0]+Thf[k_it,0])
           #print(mu_h, hk)
           P[k_it,0]=np.heaviside(mu_h-hk,1) #1.0/(1.0+np.exp((hk-mu_h)/Temperature))
           P[k_it,1]=0
           P[k_it,2]=0
           P[k_it,3]=0
    dens=np.sum(P)/Area
    if sym_ind==1:
       dens=dens*4
    return dens
# In[40]:



def VRPA(qx,qy):
    q = np.sqrt((qx)**2 + (qy)**2 + (1e-8)**2)
    vcq=VC(q)
    Vrpa=vcq/(1.0-chi0*vcq)
    return Vrpa


def k_to_2D(k_it):
    kx=int(k_it//Nk)
    ky=k_it-Nk*kx
    return np.array([kx,ky])*0.2/(Nk-1)-0.1

def k_to_2D_int(k_it):
    kx=int(k_it//Nk)
    ky=k_it-Nk*kx
    return np.array([kx,ky])


def twoD_to_k(k_array):
    kit=Nk*k_array[0]+k_array[1]
    return kit

def kit_to_negkit(k_it):
    k1=k_to_2D_int(k_it)
    k1xmin=Nk-1-k1[0]
    k1ymin=Nk-1-k1[1]
    negk=twoD_to_k(np.array([k1xmin,k1ymin]))
    return int(negk)


def make_UF_array(Vc_):
    UF=np.zeros((Nk*Nk,Nk*Nk,np.shape(tind)[0]))+0j
    #print(np.shape(UF,u,Vc_))
    
    for it in range(np.shape(tind)[0]):
        alpha=tind[it,0]
        beta=tind[it,1]
        UF[:,:,it]=Vc_[:,:]

    UF=UF/Area
    return UF


def HF_step(P,n0_):
    tnew=np.zeros((Nk*Nk,np.shape(tind)[0]))+0j
    dens=0 #np.trace(np.sum(P,axis=0))/Area
    if sym_ind==1:
        tnew[:,0] = -(np.einsum(UFo[:,:,0],[0,1],(np.conj(P[:,0])),[1]))
        dens=4*np.sum(P[:,0])/Area

        #remove Hartree term
        tnew[:,0]=tnew[:,0] +(dens-n0_)*V0 
    else:
       for it in range(np.shape(tind)[0]):
           alpha=tind[it,0]
           beta=tind[it,1]
           if alpha==beta:
              #tnew[:,it]=tnew[:,it] #+(dens-n0)*V0
              dens+=np.sum(P[:,it])/Area
           tnew[:,it]+=(-np.einsum(UFo[:,:,it],[0,1],(np.conj(P[:,it])),[1])) #UFo[k_it,k_itprime]*P[k_itprime,alpha,beta]
           #print("Fock term: ",tnew[0,it],it)
       for it in range(np.shape(tind)[0]):
           alpha=tind[it,0]
           beta=tind[it,1]
           
           #remove Hartree term
           if alpha==beta:
              tnew[:,it]=tnew[:,it] +(dens-n0_)*V0
    return tnew


            





def make_random_t():
    t_init=np.zeros((Nk*Nk,np.shape(tind)[0]))
    for k_it in range(Nk*Nk):
        if sym_ind==1:
           t_init[k_it,0]=10*(np.random.rand()-0.5)
        else:
           for it in range(np.shape(tind)[0]):
               alpha=tind[it,0]
               beta=tind[it,1]
               if alpha==beta:
                  t_init[k_it,it]=40.0*np.random.rand()
               else:
                  t_init[k_it,it]=10.0*np.random.rand()
           #print(t_init[k_it]) #-np.transpose(np.conjugate(t_init[k_it])))
    return t_init





def get_energy(P_matrix,Tnew,n0_):
    energy=0
    dens=0
    ekin=0
    interactions=0
    for k_it in range(Nk*Nk):
        if sym_ind==1:
           k_it2=kit_to_negkit(k_it)
           #print(k_it2)
           #energy+=2*E[k_it,0]*P_matrix[k_it,0]           #k -> -k: TR symmetry (?)
           #energy+=2*E[k_it,1]*P_matrix[k_it2,0]

           energy+=4*E[k_it,0]*P_matrix[k_it,0]
           ekin+=4*E[k_it,0]*P_matrix[k_it,0]
           #print("E:",E[k_it,0],E[k_it2,1])
           dens+=4*P_matrix[k_it,0]/Area
           energy+=0.5*4*P_matrix[k_it,0]*Tnew[k_it,0]
           #energy+=0.5*2*P_matrix[k_it2,0]*Tnew[k_it2,0]  #P_matrix[k_it,1]=P_matrix[k_it2,0]
           interactions+=0.5*4*P_matrix[k_it,0]*Tnew[k_it,0]
        else:
           for it in range(np.shape(tind)[0]):
               alpha=tind[it,0]
               beta=tind[it,1]
               if alpha==beta:
                  energy+=E[k_it,alpha//2]*P_matrix[k_it,it]
                  dens+=P_matrix[k_it,it]/Area
                  ekin+=E[k_it,alpha//2]*P_matrix[k_it,it] ##

               energy+=0.5*P_matrix[k_it,it]*Tnew[k_it,it]
               interactions+=0.5*P_matrix[k_it,it]*Tnew[k_it,it]
    #energy-=0.5*V0*n0_*dens*Area   #uncomment!!
    return energy/Area, dens, ekin/Area, interactions/Area

def get_free_energy(P_matrix,T,Tnew,n0_):
    energy=0+0j
    dens=0+0j
    En=np.zeros((Nk*Nk,4))+0j
    for k_it in range(Nk*Nk):
        if (sym_ind==1):
           En[k_it,:]= En[k_it,:]+E[k_it,0]+T[k_it,0]
           #print("En: ",En[k_it,:])
        else:
           alpha_inds=np.array([0,0,1,1])
           Hk=np.diag(E[k_it,alpha_inds])+0j #+T[k_it,:,:]
           for it in range(np.shape(tind)[0]):
               alpha=tind[it,0]
               beta=tind[it,1]
               Hk[alpha,beta]=Hk[alpha,beta]+T[k_it,it]
               if alpha!=beta:
                  Hk[beta,alpha]=np.conj(Hk[alpha,beta])
           vals,vecs=np.linalg.eigh(Hk)
           En[k_it,:]=vals
    x=-En/Temperature
    energy=-Temperature/Area*np.sum(np.log(1+np.exp(x)))
    #print(np.shape(np.log(1+np.exp(x))))
    #print("energy: ",energy)
    for k_it in range(Nk*Nk):
        if (sym_ind==1):
            dens+=4*P_matrix[k_it,0]/Area
            energy-=4*P_matrix[k_it,0]*T[k_it,0]/Area
            energy+=0.5*4*P_matrix[k_it,0]*Tnew[k_it,0]/Area   #uncomment!!!
        else:
           for it in range(np.shape(tind)[0]):
               alpha=tind[it,0]
               beta=tind[it,1]
               if alpha==beta:
                  dens+=P_matrix[k_it,it]/Area
               energy-=P_matrix[k_it,it]*T[k_it,it]/Area
               energy+=0.5*P_matrix[k_it,it]*Tnew[k_it,it]/Area  
    #print("dens",dens)
    energy-=0.5*V0*n0_*dens   #uncomment!!
    return energy, dens

def test():
    k_it=0
    for i in range(Nk):
       for j in range(Nk):
           print(Kx[i,j],Ky[i,j],k_it)
           k2=kit_to_negkit(k_it)
           print(i,j,k_to_2D(k_it),k_to_2D(k2))
           print(k_it,k2)
           print(E[k_it,0],E[k2,1],E[-k_it,0])
           k_it+=1


################################################
#E,A=make_EA()
#test()
#print(E[:,0])
#print(E[:,1])

#################################################

V0=VC(np.array([1,0]),np.array([1,0]))
#print(V0)
#print(V0*n0)


#print(q[1,0])
Vc_=VC_array(q)
#print(Vc_[0,0])

E=make_EA()
print(E)
#mu=(np.max(E)-np.min(E))*0.0+np.min(E)+0.03
#mu_array=np.linspace(np.min(E),np.max(E),7)
#num_densities=np.shape(mu_array)[0]
#E=-(E)
#print(E)
#print("energy: ",E)
#print(n0_array,np.min(E),np.max(E), "mu: ",mu)

UFo=make_UF_array(Vc_)

#print("Ufo")
#for it in range(np.shape(tind)[0]):
#    print(UFo[0,:,it], tind[it,0], tind[it,1])

maxit=250


T=np.zeros((Nk*Nk,np.shape(tind)[0])) #make_random_t() #np.ones((Nk*Nk,np.shape(tind)[0])) #make_random_t()

"""
#P=np.zeros((maxit,Nk*Nk,np.shape(tind)[0]))
P=get_P(T) 
Tnew=HF_step(P)  
#print("Tnew: ",Tnew)
Energy=get_energy(P,Tnew)
Energy2=get_free_energy(P,T,Tnew)
print(Energy2)
P=get_P(Tnew) 
Tnew2=HF_step(P)
Energy=get_energy(P,Tnew2)
Energy2=get_free_energy(P,Tnew,Tnew2)
print(Energy,Energy2)
#for it in range(np.shape(tind)[0]):
#        alpha=tind[it,0]
#        beta=tind[it,1]
#        print(alpha,beta)
#        print(P[:,it])
"""
     
#print(Tnew)
tol=0.005
Free_energies = np.zeros(num_densities);
Einteractions = np.zeros(num_densities);
Ekinetic = np.zeros(num_densities);

Densities = np.zeros(num_densities);
Niter = np.zeros(num_densities);
dtmax_final = np.zeros(num_densities);
P_final = np.zeros((num_densities, Nk*Nk, np.shape(tind)[0]));
T_final = np.zeros((num_densities, Nk*Nk, np.shape(tind)[0]));

print("num_densities: ",num_densities)
for n_it in range(num_densities):
    muold=0
    if n_it==0 or abs(kmax_values[n_it]-kmax_values[n_it-1])>1e-5:
       Kmax = kmax_values[n_it]
       kx = np.linspace(-Kmax,Kmax,Nk)
       ky = np.linspace(-Kmax,Kmax,Nk)
       Area = (2*np.pi)**2/(kx[1]-kx[0])**2;
       print("Area", Area)
       print("\n","\n")
       Ky,Kx=np.meshgrid(kx,ky)
       K1x = np.dot(np.reshape(Kx,((Nk*Nk,1))),np.ones((1,Nk*Nk)));
       K1y =np.dot(np.reshape(Ky,(Nk*Nk,1)),np.ones((1,Nk*Nk)));
       K2x = np.dot(np.ones((Nk*Nk,1)),np.reshape(Kx,(1,Nk*Nk)));
       K2y = np.dot(np.ones((Nk*Nk,1)),np.reshape(Ky,(1,Nk*Nk)));

       dKx=K1x - K2x
       dKy=K1y - K2y
       dKx2=dKx*dKx
       dKy2=dKy*dKy
       q = np.sqrt(dKx2+dKy2 + (1e-8)**2);
       V0=VC(np.array([1,0]),np.array([1,0]))

       Vc_=VC_array(q)
       E=make_EA()

       UFo=make_UF_array(Vc_)
    T=make_random_t() #np.ones((Nk*Nk,np.shape(tind)[0])) #make_random_t()
    print(n_it)
    n0_it=n0_array[n_it]
    print(n0_it)
    eta=0.02
    for it in range(maxit):
       #print("shape T: ",it,np.shape(T))

       #mu_values=np.linspace(0.0,deltaN*V0/10*n0_array[n_it]/n0_array[0],10)
       munew=0

       """
       if (it>=0):
          f=lambda x: densities_test(T,x)-n0_it
          limit0=muold
          while f(limit0)>-3e-5:
            limit0=limit0-deltaN*V0/10
          #print("f limit0: ",f(limit0))
          limit1=muold
          while f(limit1)<3e-5:
             limit1=limit1+deltaN*V0/10
          #print("f limit1: ",f(limit1))
          munew=scipy.optimize.bisect(f,limit0,limit1,rtol=2e-5,maxiter=100)
          muold=munew
          #for muv in mu_values:
          #   print("mu: ",muv,"density from density test: ",densities_test(T,muv))
       """

       P=get_P(T,munew) 
       Tnew=HF_step(P,n0_it)
       #Energy, dens=get_free_energy(P,T,Tnew,n0_it)
       Energy, dens, eint, ekin =get_energy(P,Tnew,n0_it)
       print(Energy, eint, ekin) #,Energy2)
       
       #mu_values=np.linspace(0.0,deltaN*V0/10*n0_array[n_it]/n0_array[0],10)
       #munew=scipy.optimize.bisect(densities_test-n0_it,0.0,deltaN*V0/10*n0_array[n_it]/n0_array[0],4*e-5,maxiter=100)
       #for muv in mu_values:
       #   print("mu: ",muv,"density from density test: ",densities_test(T,muv))
       dtmax = 0; 
    
       for jt in range(np.shape(tind)[0]):
           dtmax_cur = np.max(np.abs(Tnew[:,jt] - T[:,jt]));
           dtmax = max(dtmax, dtmax_cur);
           if (dtmax > 20 and eta != 0.02):
               eta = 0.02;
               print('eta changed to ', eta);
           elif (2.0<dtmax and dtmax <= 20 and eta != 0.03):
               eta = 0.03;
               print('eta changed to ', eta);
           elif (0.5<dtmax and dtmax <= 2 and eta != 0.04):
               eta = 0.04;
               print('eta changed to ', eta)
           elif (dtmax <= 0.5 and eta != 0.05):    
               eta = 0.05;
               print('eta changed to ', eta);
    
       T=(1-eta)*T + eta*Tnew; 

       Free_energies[n_it] = Energy;
       Einteractions[n_it]=eint
       Ekinetic[n_it]=ekin
       Densities[n_it]=dens
       print("dens: ",n0_it,dens," kF: ", np.max(P[:,0]*kradius),"\n")
       Niter[n_it] = it;
       dtmax_final[n_it] = dtmax;
       P_final[n_it,:,:] = P;
       T_final[n_it,:,:]=T;
       #T_final[n_it] = T;
       if (dtmax<tol):
            print("stopping after", it, "iterations") 
            break;

data=np.vstack((n0_array,Densities,Niter,Einteractions, Ekinetic,Free_energies))
np.savetxt("data.txt",data)
np.savetxt("kmax_values.txt",kmax_values)
for i in range(num_densities):
    np.savetxt("Pmatrix{}.txt".format(i),P_final[i,:,:])
    np.savetxt("Tmatrix{}.txt".format(i),T_final[i,:,:])

occupancies=np.sum(P_final[:,:,:],axis=1)
np.savetxt("numspins.txt",occupancies)
