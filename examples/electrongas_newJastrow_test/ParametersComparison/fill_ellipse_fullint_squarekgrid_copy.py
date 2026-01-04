#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import numpy as np
import matplotlib.pyplot as plt



# # HF trilayer graphene

factordeg=1
sym_ind=2
if sym_ind==2:
   factordeg=2
if sym_ind==1:
   factordeg=4

sx=np.array([[0,1],[1,0]])+0j
sy=np.array([[0,-1j],[1j,0]])+0j
sz=np.array([[1,0],[0,-1]])+0j 

Temperature=0.3
hbar=1

N=3002  #number of electrons
Etatilde=1.0/5.79 #1.0 #1.0/5.4 #5.79 #4.013381993233915   #5
Nkx=11
Nky=11

n0=0.1/312.097  #10^11 cm^-2
n1=n0*1.0 #200.0
a = 0.56605e-9;        # AlAs lattice spacing in nanometers

kF=np.sqrt(2.0*np.pi*n1)  #SP or VP in Rydberg units?
print("kF: ",kF)
#Area=N*1.0/n1
#deltaK= (2*np.pi)/np.sqrt(Area)
#Area = (2*np.pi)**2/(deltaK)**2;
#N=n1*Area
#print(deltaK,kF)
tau=1

kxF=kF*Etatilde**(-tau/4.0)
kyF=kF*Etatilde**(tau/4.0)

kFmax=max(kxF,kyF)
#kyFmax=max(kxF,kyF)

#k1x=np.linspace(-kxF,kxF,Nkx)
#k1y=np.linspace(-kyF,kyF,Nky)

k1x=np.linspace(-kFmax,kFmax,Nkx)
k1y=np.linspace(-kFmax,kFmax,Nky)

np.savetxt("kFmax.txt",np.array([kFmax]))

#print(k1x,k1y)

#(try beide Reihenfolgen for lowest energy)


#Nkx=np.shape(kxges)[0]
#Nky=np.shape(kyges)[0]
#print(Nkx,Nky)
#print(kxges, kyges)
#Nk=np.max([Nkx,Nky])

Kx,Ky=np.meshgrid(k1x,k1y)
#print(np.shape(Ky),np.shape(Kx))

K1x = np.dot(np.reshape(Kx,((Nkx*Nky,1))),np.ones((1,Nkx*Nky)));
K1y =np.dot(np.reshape(Ky,(Nkx*Nky,1)),np.ones((1,Nkx*Nky)));
K2x = np.dot(np.ones((Nkx*Nky,1)),np.reshape(Kx,(1,Nkx*Nky)));
K2y = np.dot(np.ones((Nkx*Nky,1)),np.reshape(Ky,(1,Nkx*Nky)));

dKx=K1x - K2x
dKy=K1y - K2y
dKx2=dKx*dKx
dKy2=dKy*dKy
q = np.sqrt(dKx2+dKy2 + (1e-8)**2);
Area = (2*np.pi)**2/((k1x[1]-k1x[0])*(k1y[1]-k1y[0]));
#Area = (2*np.pi)**2/(deltaK)**2;
# In[35]:
Eta=5.79  #1.0 #5.79

d = 100*10**(-9)/a; # Distance to gate in units of unit cell size
epsilon_d = 10; # AlAs dielectric constant (GaAs? AlGaAs?)

qe = 1.60217662e-19; # electron charge
ke = 8.99e9;         # Coulomb constant
hbar=1.0545718*1e-34;  #hbar (in SI units)
a = 0.56605e-9;        # AlAs lattice spacing in nanometers

e_squared = 1.0*ke*qe/a*1e3/epsilon_d; # e^2 in units of meV*a
me=9.1093837015*1e-31  #kg - umrechnen?? 1/a? meV?
mstar=0.457*me*qe/(hbar**2*1e3)*a*a  #mstar in units of (1/(meV*a^2)) (from hbar) -> 1/mstar*k^2->meV  (a needed here?)

def make_k():
   Kvalues=np.zeros((Nkx*Nky,2))
   k_it=0
   for i in range(Nkx):
       for j in range(Nky):
             kx_=Kx[j,i]
             ky_=Ky[j,i]
             Kvalues[k_it,0]=kx_
             Kvalues[k_it,1]=ky_
             k_it+=1
   return Kvalues


def make_EA():
   h_uncorr=np.zeros((Nkx,Nky,6,6,4,4))+0j
   k_it=0

   
 
   E=np.zeros((Nkx*Nky,2))
   A=np.zeros((Nkx*Nky,6,4))+0j
   E2=np.zeros((Nkx*Nky,2))

   for i in range(Nkx):
       for j in range(Nky):
           for Tau in range(2):
               for s in range(2):
                   tau=(2*Tau-1)
                   kx_=Kx[j,i]
                   ky_=Ky[j,i]
                   E[k_it,Tau]=Eta**(0.5*tau)/(2.0*mstar)*kx_**2+Eta**(-0.5*tau)/(2.0*mstar)*ky_**2
                   #print(E[k_it,Tau])
           k_it+=1
   return E


Earray=make_EA()
print(np.max(Earray),np.min(Earray))


def k_to_2D_int(k_it):
    kx=int(k_it//Nky)
    ky=k_it-Nky*kx
    return np.array([kx,ky])


def twoD_to_k(k_array):
    kit=Nky*k_array[0]+k_array[1]
    return kit

def kit_mirroredxaxis(k_it):
    k1=k_to_2D_int(k_it)
    k1xmin=k1[0]
    k1ymin=Nky-1-k1[1]
    negk=twoD_to_k(np.array([k1xmin,k1ymin]))
    return int(negk)

def kit_mirroredyaxis(k_it):
    k1=k_to_2D_int(k_it)
    k1xmin=Nkx-1-k1[0]
    k1ymin=k1[1]
    negk=twoD_to_k(np.array([k1xmin,k1ymin]))
    return int(negk)

def kit_mirroredxyaxes(k_it):
    k1=k_to_2D_int(k_it)
    k1xmin=Nkx-1-k1[0]
    k1ymin=Nky-1-k1[1]
    negk=twoD_to_k(np.array([k1xmin,k1ymin]))
    return int(negk)

Kv_=make_k()
#for k_it in range(Nkx*Nky):
#    k=Kv_[k_it]
#    kitmx=kit_mirroredxaxis(k_it)
#    k2=Kv_[kitmx]
#    #print(k,k2)

def sort_filledkit(P):
    Kv_=make_k()
    sortedEn=np.zeros((Nkx*Nky))
    sortedkit=np.zeros((Nkx*Nky),dtype=int)
    markzerosx=np.zeros((Nkx*Nky),dtype=int)
    markzerosy=np.zeros((Nkx*Nky),dtype=int)
    for k_it in range(Nkx*Nky):
        k=Kv_[k_it]
        kx_=k[0]
        ky_=k[1]
        tau=1
        en=Eta**(0.5*tau)/(2.0*mstar)*kx_**2+Eta**(-0.5*tau)/(2.0*mstar)*ky_**2
        #print("kx,ky: ",kx_,ky_)
        if P[k_it]>0.5 and kx_>=-1e-7 and ky_>=-1e-7:
           sortedEn[k_it]=en
           sortedkit[k_it]=int(k_it)
           #print("P")
           if np.abs(kx_)<1e-5:
              markzerosx[k_it]=1
              #print("hi")
           if np.abs(ky_)<1e-5:
              markzerosy[k_it]=1

    sortedkit=(sortedkit[np.argsort(-sortedEn)])
    markzerosx=(markzerosx[np.argsort(-sortedEn)])
    markzerosy=(markzerosy[np.argsort(-sortedEn)])

    sortedEn=sortedEn[np.argsort(-sortedEn)]
    
    #print("sorted",sortedkit,sortedEn[sortedkit],markzerosx,markzerosy)
    return sortedkit,sortedEn, markzerosx,markzerosy

def sort_notfilledkit(P):
    Kv_=make_k()
    sortedEn=np.zeros((Nkx*Nky))
    sortedkit=np.zeros((Nkx*Nky),dtype=int)
    markzerosx=np.zeros((Nkx*Nky),dtype=int)
    markzerosy=np.zeros((Nkx*Nky),dtype=int)
    for k_it in range(Nkx*Nky):
        k=Kv_[k_it]
        kx_=k[0]
        ky_=k[1]
        tau=1
        en=Eta**(0.5*tau)/(2.0*mstar)*kx_**2+Eta**(-0.5*tau)/(2.0*mstar)*ky_**2
        if P[k_it]<0.5 and kx_>=-1e-7 and ky_>=-1e-7:
           sortedEn[k_it]=en
           sortedkit[k_it]=int(k_it)
           if np.abs(kx_)<1e-5:
              markzerosx[k_it]=1
           if np.abs(ky_)<1e-5:
              markzerosy[k_it]=1

    #for k_it in range(Nkx*Nky):
    #    if markzerosx[k_it]>0.5:
    #       print("K!: ",Kv_[k_it])
    sortedEn[sortedEn==0]=2*np.max(sortedEn)

    sortedkit=(sortedkit[np.argsort(sortedEn)])
    markzerosx=(markzerosx[np.argsort(sortedEn)])
    markzerosy=(markzerosy[np.argsort(sortedEn)])

    sortedEn=sortedEn[np.argsort(sortedEn)]
   
    #print("sorted",sortedkit,sortedEn[sortedkit],markzerosx,markzerosy)
    return sortedkit,sortedEn, markzerosx,markzerosy


def VC(k1,k2):
    q=np.sqrt((k1[0]-k2[0])**2+(k1[1]-k2[1])**2+(1e-8)**2)
    Vc=2.0*np.pi*e_squared*np.tanh(d*q)/q   #(2*eps*q)
    return Vc
   
def VC_array(q):
    return 2.0*np.pi*e_squared*np.tanh(d*q)/q  

#print(VC(np.array([1,0]),np.array([1,0])))
#print(VC(np.array([1,0]),np.array([1,0]))*n0)


def get_P(iteration=0):
    P=np.zeros((Nkx*Nky))+0j
    k_it=0
    for i in range(Nkx):
       for j in range(Nky):
           kx_=Kx[j,i]
           ky_=Ky[j,i]
           if P[k_it]<0.5:
              if np.heaviside(kF**2-Etatilde**(tau/2.0)*kx_**2- Etatilde**(-tau/2.0)*ky_**2,1)>0.5:
                P[k_it]=1
                P[kit_mirroredxaxis(k_it)]=1
                P[kit_mirroredxaxis(k_it)]=1
                P[kit_mirroredxyaxes(k_it)]=1
           k_it+=1
    return P




def make_UF_array(Vc_):
    UF=np.zeros((Nkx*Nky,Nkx*Nky))+0j
    #print(np.shape(UF,u,Vc_))
   
    UF[:,:]=Vc_[:,:]

    UF=UF/Area
    return UF


def HF_step(P,n0_):
    tnew=np.zeros((Nkx*Nky))+0j
    dens=0 #np.trace(np.sum(P,axis=0))/Area
    if sym_ind==1:
        tnew[:] = -(np.einsum(UFo[:,:],[0,1],(np.conj(P[:])),[1]))
        dens=4*np.sum(P[:])/Area
        tnew[:]=tnew[:] +(dens-n0_)*V0 
    elif sym_ind==2:
        tnew[:] = -(np.einsum(UFo[:,:],[0,1],(np.conj(P[:])),[1]))
        dens=2*np.sum(P[:])/Area
        #tnew[:]=tnew[:] +(dens-n0_)*V0
    elif sym_ind==3:
        tnew[:] = -(np.einsum(UFo[:,:],[0,1],(np.conj(P[:])),[1]))
        dens=1*np.sum(P[:])/Area
        tnew[:]=tnew[:] +(dens-n0_)*V0
    return tnew








def get_energy(P_matrix,Tnew,n0_):
    energy=0
    dens=0
    interaction=0
    for k_it in range(Nkx*Nky):
        if sym_ind==1:
           energy+=4*E[k_it,0]*P_matrix[k_it]           #k -> -k: TR symmetry (?)
           #print("E:",E[k_it,0],E[k_it2,1])
           dens+=4*P_matrix[k_it]/Area
           energy+=0.5*4*P_matrix[k_it]*Tnew[k_it]
        elif sym_ind==2:
           energy+=2*E[k_it,0]*P_matrix[k_it]           #k -> -k: TR symmetry (?)
           #print("E:",E[k_it,0],E[k_it2,1])
           dens+=2*P_matrix[k_it]/Area
           energy+=0.5*2*P_matrix[k_it]*Tnew[k_it]
           interaction+=0.5*2*P_matrix[k_it]*Tnew[k_it]
        elif sym_ind==3:
           energy+=1*E[k_it,0]*P_matrix[k_it]           #k -> -k: TR symmetry (?)
           #print("E:",E[k_it,0],E[k_it2,1])
           dens+=1*P_matrix[k_it]/Area
           energy+=0.5*1*P_matrix[k_it]*Tnew[k_it]
    energy+=0.5*V0*dens*dens*Area #0.5*V0*n0_*dens*Area   #uncomment!! (- sign: density removed)
    interaction+=0.5*V0*dens*dens*Area #0.5*V0*n0_*dens*Area
    return energy/Area, dens, interaction/Area, 0.5*V0*dens*dens



V0=VC(np.array([1,0]),np.array([1,0]))
Vc_=VC_array(q)

E=make_EA()

UFo=make_UF_array(Vc_)

#print("Ufo")
#for it in range(np.shape(tind)[0]):
#    print(UFo[0,:,it], tind[it,0], tind[it,1])

maxit=1


for it in range(maxit):
    
    print(it)
    P=get_P(0) 
    Tnew=HF_step(P,n1)
    Energy, dens, interaction,hartree=get_energy(P,Tnew,n1)
    Nspins=np.sum(P)*factordeg
    print("\n")
    #print("Energy/N: ",Energy,"Ekin/N SI: ",(Energy-interaction)*qe*1e-3/n1, "Eint/N SI: ",interaction*qe*1e-3/n1,"Hartree: ",hartree*qe*1e-3/n1, dens,n1,dens*Area,n1*Area,"\n")
    print("Energy/N with dens instead of n1: ",Energy,"Ekin/N SI: ",(Energy-interaction)*qe*1e-3/dens, "Eint/N SI: ",interaction*qe*1e-3/dens,"Hartree SI: ",hartree*qe*1e-3/dens,"Exchange SI:", (interaction-hartree)*qe*1e-3/dens, dens,n1,dens*Area,n1*Area,"\n")
    print("Energy, for VMC: ",Energy*Area,"Ekin: ",(Energy-interaction)*Area, "Eint: ",interaction*Area,"Hartree: ",hartree*Area,"Exchange: ", (interaction-hartree)*Area,dens,n1,dens*Area,n1*Area,"\n")
    #sk,mx,my=sort_filledkit(P)
    #print("Energy, for VMC in SI: ",Energy*Area,"Ekin: ",(Energy-interaction)*Area, "Eint: ",interaction*Area,dens,n1,dens*Area,n1*Area,"\n")
    ndiff=np.round(n1*Area-dens*Area)
    ndiff0=ndiff

    print("ndiff0: ",ndiff0, "density diff percentage: ", (n1-dens)/n1)
  
    kvalues=P[:]
    #print(kvalues, np.shape(kvalues), 61*61, 41*41)
    k2D=np.zeros_like(Kx)
    k_it=0
    #plt.figure()
    for l in range(Nkx):
      for r in range(Nky):
        kx=Kx[r,l]
        ky=Ky[r,l]
        k2D[r,l]=kvalues[k_it]
        #k_it+=1
        #if kvalues[k_it]>0.5:
        #   plt.scatter(kx,ky)
        k_it+=1
    #plt.grid(True)
    #plt.show()
  
    plt.figure()
    plt.contourf(Kx,Ky,k2D)
    plt.vlines(k1x,np.min(k1y),np.max(k1y))
    plt.hlines(k1y,np.min(k1x),np.max(k1x))
    plt.show()

Pges=np.zeros((np.shape(P)[0],4))
Pges[:,0]=P
Pges[:,1]=P
numspins=np.sum(Pges,axis=0)
np.savetxt("Pmatrix_ellipse.txt",Pges)
np.savetxt("numspins.txt",numspins)
#    np.savetxt("Tmatrix{}.txt".format(i),T_final[i,:,:])

