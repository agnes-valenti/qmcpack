#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import numpy as np
import matplotlib.pyplot as plt



# # HF trilayer graphene

factordeg=1
sym_ind=2
vp=False #True
sp=False #True #False
if sym_ind==2:
   factordeg=2
if sym_ind==1:
   factordeg=4

sx=np.array([[0,1],[1,0]])+0j
sy=np.array([[0,-1j],[1j,0]])+0j
sz=np.array([[1,0],[0,-1]])+0j 


hbar=1
Nkx=5
Nky=5
Nkx_=Nkx
Nky_=Nkx


if Nkx%2==1:
    Nkx=Nkx*2-1
else:
    Nkx=Nkx*2

if Nky%2==1:
    Nky=Nky*2-1
else:
    Nky=Nky*2



n0=0.1/312.097  #10^11 cm^-2
n1=n0*1.0 #200.0
a = 0.56605e-9;        # AlAs lattice spacing in nanometers

#EnergyInf= -6.9999*1e-22 # -6.465648038279017*1e-22
EkinInf=  4.9616599741533115*1e-23
EnergyInf=-6.465648038279017*1e-22
kF=np.sqrt(2.0*np.pi*n1)  #SP or VP in Rydberg units?

if sym_ind==3:
   kF=np.sqrt(4.0*np.pi*n1)
if sym_ind==1:
   kF=np.sqrt(np.pi*n1)
#Area = (2*np.pi)**2/(deltaK)**2;
# In[35]:
Eta=5.79  #1.0 #5.79  #1.0 #5.79

d = 100*10**(-9)/a; # Distance to gate in units of unit cell size
epsilon_d = 10; # AlAs dielectric constant (GaAs? AlGaAs?)

qe = 1.60217662e-19; # electron charge
ke = 8.99e9;         # Coulomb constant
hbar=1.0545718*1e-34;  #hbar (in SI units)
a = 0.56605e-9;        # AlAs lattice spacing in nanometers

e_squared = 1.0*ke*qe/a*1e3/epsilon_d; # e^2 in units of meV*a
me=9.1093837015*1e-31  #kg - umrechnen?? 1/a? meV?
mstar=0.457*me*qe/(hbar**2*1e3)*a*a  #mstar in units of (1/(meV*a^2)) (from hbar) -> 1/mstar*k^2->meV  (a needed here?)

tau=1

 
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

def kit_switchxyaxes(k_it):
    k1=k_to_2D_int(k_it)
    k1xmin=k1[1]
    k1ymin=k1[0]
    negk=twoD_to_k(np.array([k1xmin,k1ymin]))
    return int(negk)


#Kv_=make_k()
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

"""
def get_P(Etatilde_,deltaK=0):
    nsteps=500
    muvalues=np.linspace(kF-deltaK,kF+deltaK,2*nsteps+1)
    #print("deltaK: ",deltaK)
    area=(2*np.pi)**2/(deltaK**2)
    P_=0
    diff_dens_old=100
    #for mu_it, mu in enumerate(muvalues):
    mu=kF
    it=0
    while(np.sum(P_)*factordeg<Nparticles-1):
       mu=mu+deltaK*it/nsteps
       P=np.zeros((Nkx*Nky))+0j
       k_it=0
       for i in range(Nkx):
           for j in range(Nky):
              kx_=Kx[j,i]
              ky_=Ky[j,i]
              if P[k_it]<0.5 and np.sum(P)*factordeg<Nparticles-1:
                 #if np.heaviside(mu**2-Etatilde_**(tau/2.0)*kx_**2- Etatilde_**(-tau/2.0)*ky_**2,1)>0.5:
                 if np.heaviside(mu**2-Etatilde_**(tau/2.0)*kx_**2- Etatilde_**(-tau/2.0)*ky_**2,1)>0.5:

                   P[k_it]=1
                   #P[kit_mirroredxaxis(k_it)]=1
                   #P[kit_mirroredxaxis(k_it)]=1
                   #P[kit_mirroredxyaxes(k_it)]=1
              k_it+=1
       it+=1
       #diff_dens=np.abs(np.sum(P)*factordeg-n1*area)
       #if diff_dens<diff_dens_old:
       #   P_=np.copy(P)
       #   diff_dens_old=diff_dens
       P_=np.copy(P)
    print("Nparticles: ",Nparticles, "sum P: ", factordeg*np.sum(P_))
    return P_
"""

def get_P(Etatilde_,deltaK=0):
    nsteps=50
    muvalues=np.linspace(kF-deltaK/4.0,kF+deltaK/4.0,2*nsteps+1)
    #print("deltaK: ",deltaK)
    area=(2*np.pi)**2/(deltaK**2)
    P_=0
    diff_dens_old=100
    for mu_it, mu in enumerate(muvalues):
    
    
       P=np.zeros((Nkx*Nky))+0j
       k_it=0
       for i in range(Nkx):
           for j in range(Nky):
              kx_=Kx[j,i]
              ky_=Ky[j,i]
              if P[k_it]<0.5 and np.sum(P)*factordeg<Nparticles-1:
                 #if np.heaviside(mu**2-Etatilde_**(tau/2.0)*kx_**2- Etatilde_**(-tau/2.0)*ky_**2,1)>0.5:
                 if np.heaviside(mu**2-Etatilde_**(tau/2.0)*kx_**2- Etatilde_**(-tau/2.0)*ky_**2,1)>0.5:

                   P[k_it]=1
                   #P[kit_mirroredxaxis(k_it)]=1
                   #P[kit_mirroredxaxis(k_it)]=1
                   #P[kit_mirroredxyaxes(k_it)]=1
              k_it+=1
      
       diff_dens=np.abs(np.sum(P)*factordeg-n1*area)
       if diff_dens<diff_dens_old:
          P_=np.copy(P)
          diff_dens_old=diff_dens
       #P_=np.copy(P)
    print("Nparticles: ",Nparticles, "sum P: ", factordeg*np.sum(P_))
    return P_

def mirror_P(Pold):
    Pnew=np.zeros((Nkx*Nky))+0j
    k_it=0
    for i in range(Nkx):
       for j in range(Nky):
           kx_=Kx[j,i]
           ky_=Ky[j,i]
           #print(kx_,ky_)
           #print("kx,ky: ",k_to_2D_int(k_it), "kx,ky switched: ",k_to_2D_int(kit_switchxyaxes(k_it)))
           if Pold[k_it]>0.5:
              Pnew[kit_switchxyaxes(k_it)]=1      
           k_it+=1
    return Pnew

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
        #tnew[:]=tnew[:] +(dens-n0_)*V0 
    elif sym_ind==2:
        tnew[:] = -(np.einsum(UFo[:,:],[0,1],(np.conj(P[:])),[1]))
        dens=2*np.sum(P[:])/Area
        #tnew[:]=tnew[:] +(dens-n0_)*V0
    elif sym_ind==3:
        tnew[:] = -(np.einsum(UFo[:,:],[0,1],(np.conj(P[:])),[1]))
        dens=1*np.sum(P[:])/Area
        #tnew[:]=tnew[:] +(dens-n0_)*V0
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
           interaction+=0.5*4*P_matrix[k_it]*Tnew[k_it]
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
           interaction+=0.5*1*P_matrix[k_it]*Tnew[k_it]
    energy+=0.5*V0*dens*dens*Area #0.5*V0*n0_*dens*Area   #uncomment!! (- sign: density removed)
    interaction+=0.5*V0*dens*dens*Area #0.5*V0*n0_*dens*Area
    return energy/Area, dens, interaction/Area, 0.5*V0*dens*dens


etatildeValues=np.array([1.7609294853520199,1.7609294853520199])   #np.linspace(5.79, 5.79000000001,nEtas) #1.0,5.79,nEtas)
etatildeValue=1.7609294853520199  #1.0 #1.7609294853520199
if sym_ind==1:
   etatildeValues=np.array([1.6001422848756166, 1.6001422848756166])
   etatildeValue=1.6001422848756166

#etatildeValues=np.linspace(2.0,2.001,nEtas)

kxF_=kF*etatildeValue**(-tau/4.0)
kyF_=kF*etatildeValue**(tau/4.0)

kFmax_=max(kxF_,kyF_)
#N=3002  #number of electrons


k1x_=np.linspace(-2*kFmax_,2*kFmax_,Nkx)
k1y_=np.linspace(-2*kFmax_,2*kFmax_,Nky)
deltaK_=max((k1x_[1]-k1x_[0]),(k1y_[1]-k1y_[0]))
area_=(2.0*np.pi)**2/(deltaK_**2)
Nparticles=np.round(n1*area_)
NumOffsets=600 #1
nEtas=(2*NumOffsets+1)
kxoffsets=np.linspace(0,deltaK_,2*NumOffsets+1) #np.array([0,0,0])
#kyoffsets=np.linspace(-deltaK_,deltaK_,2*NumOffsets+1)
kyoffsets=np.array([0])
Pmatricessp=np.zeros((nEtas,Nkx*Nky,4)) #[eta_it,:,:]=Pges 
Pmatricesvp=np.zeros((nEtas,Nkx*Nky,4)) #[eta_it,:,:]=Pges 
Pmatricesvp2=np.zeros((nEtas,Nkx*Nky,4)) #[eta_it,:,:]=Pges 

Energies=np.zeros(nEtas)
EnergiesSI=np.zeros(nEtas)
Ekins=np.zeros(nEtas)
Exs=np.zeros(nEtas)
EkinsSI=np.zeros(nEtas)
ExsSI=np.zeros(nEtas)
EhartreesSI=np.zeros(nEtas)
Ehartrees=np.zeros(nEtas)
Eintges=np.zeros(nEtas)
area=np.zeros(nEtas)

numberspins=np.zeros(nEtas)
numberspins_species=np.zeros((nEtas,4)) #[eta_it,:]=np.sum(Pges,axis=0)
kFmaxValues=np.zeros(nEtas)



eta_it=0
for offset_x in kxoffsets:
  for offset_y in kyoffsets: 
    print("eta_it: ",eta_it)
    Etatilde=1.0/etatildeValue #5.79 #1.0 #1.0/5.4 #5.79 #4.013381993233915   #5
    #print("Etatilde: ",Etatilde)

    kxF=kF*Etatilde**(-tau/4.0)
    kyF=kF*Etatilde**(tau/4.0)

    kFmax=max(kxF,kyF)
    #N=3002  #number of electrons
 
    print(offset_x)

    k1x=np.linspace(-2*kFmax+offset_x,2*kFmax+offset_x,Nkx)
    k1y=np.linspace(-2*kFmax+offset_x,2*kFmax+offset_x,Nky)
    
    kFmaxValues[eta_it]=kFmax
    #np.savetxt("kFmax.txt",np.array([kFmax]))

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
    deltaK=max((k1x[1]-k1x[0]),(k1y[1]-k1y[0]))
    area[eta_it]=Area
    V0=VC(np.array([1,0]),np.array([1,0]))
    Vc_=VC_array(q)

    E=make_EA()

    UFo=make_UF_array(Vc_)


    En=0

    
    #print(it)
    P=get_P(Etatilde,deltaK) 
    Tnew=HF_step(P,n1)
    Energy, dens, interaction,hartree=get_energy(P,Tnew,n1)
    Nspins=np.sum(P)*factordeg
    #print("\n")
    #print("Energy/N: ",Energy,"Ekin/N SI: ",(Energy-interaction)*qe*1e-3/n1, "Eint/N SI: ",interaction*qe*1e-3/n1,"Hartree: ",hartree*qe*1e-3/n1, dens,n1,dens*Area,n1*Area,"\n")
    #print("Energy/N with dens instead of n1: ",Energy,"Ekin/N SI: ",(Energy-interaction)*qe*1e-3/dens, "Eint/N SI: ",interaction*qe*1e-3/dens,"Hartree SI: ",hartree*qe*1e-3/dens,"Exchange SI:", (interaction-hartree)*qe*1e-3/dens, dens,n1,dens*Area,n1*Area,"\n")
    #print("Energy, for VMC: ",Energy*Area,"Ekin: ",(Energy-interaction)*Area, "Eint: ",interaction*Area,"Hartree: ",hartree*Area,"Exchange: ", (interaction-hartree)*Area,dens,n1,dens*Area,n1*Area,"\n")
    Nspins=factordeg*np.sum(P)
    En=(Energy-interaction)*qe*1e-3/dens+ (interaction-hartree)*qe*1e-3/dens #((Energy-interaction)*Area+(interaction-hartree)*Area)/Nspins
    EkinsSI[eta_it]=(Energy-interaction)*qe*1e-3/dens
    Energies[eta_it]=(Energy-interaction+interaction-hartree)*Area
    ExsSI[eta_it]=(interaction-hartree)*qe*1e-3/dens
    Ekins[eta_it]=(Energy-interaction)*Area
    Exs[eta_it]=(interaction-hartree)*Area
    Ehartrees[eta_it]=hartree*Area
    EhartreesSI[eta_it]=hartree*qe*1e-3/dens
    Eintges[eta_it]=interaction*Area
    #sk,mx,my=sort_filledkit(P)
    #print("Energy, for VMC in SI: ",Energy*Area,"Ekin: ",(Energy-interaction)*Area, "Eint: ",interaction*Area,dens,n1,dens*Area,n1*Area,"\n")
    ndiff=np.round(n1*Area-dens*Area)
    ndiff0=ndiff

    #print("ndiff0: ",ndiff0, "density diff percentage: ", (n1-dens)/n1)
  
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

    kvalues2=mirror_P(P)
    #print(kvalues, np.shape(kvalues), 61*61, 41*41)
    k2D2=np.zeros_like(Kx)
    k_it=0
    #plt.figure()
    for l in range(Nkx):
      for r in range(Nky):
        kx=Kx[r,l]
        ky=Ky[r,l]
        k2D2[r,l]=kvalues2[k_it]
        #k_it+=1
        #if kvalues[k_it]>0.5:
        #   plt.scatter(kx,ky)
        k_it+=1
    #plt.grid(True)
    #plt.show()
  
    #plt.figure()
    #plt.contourf(Kx,Ky,k2D)
    #plt.vlines(k1x,np.min(k1y),np.max(k1y))
    #plt.hlines(k1y,np.min(k1x),np.max(k1x))
    
    #plt.figure()
    #plt.contourf(Kx,Ky,k2D2)
    #plt.vlines(k1x,np.min(k1y),np.max(k1y))
    #plt.hlines(k1y,np.min(k1x),np.max(k1x))
    #plt.show()

    Pgessp=np.zeros((np.shape(P)[0],4))
    Pgesvp2=np.zeros((np.shape(P)[0],4))

    Pgessp[:,0]=P
    if sym_ind==2:
       Pgessp[:,2]=mirror_P(P)
       Pgesvp2[:,2]=mirror_P(P)
       Pgesvp2[:,3]=mirror_P(P)

    elif sym_ind==1:
       Pgessp[:,1]=P
       Pgessp[:,2]=mirror_P(P)
       Pgessp[:,3]=mirror_P(P)

    Pgesvp=np.zeros((np.shape(P)[0],4))
    Pgesvp[:,0]=P
    Pgesvp[:,1]=P
  
    numspins=np.sum(Pgessp,axis=0)
    Pmatricessp[eta_it,:,:]=Pgessp
    Pmatricesvp[eta_it,:,:]=Pgesvp

    Pmatricesvp2[eta_it,:,:]=Pgesvp2

    EnergiesSI[eta_it]=En
    print("En: ",En)
    numberspins[eta_it]=np.sum(Pgessp)
    numberspins_species[eta_it,:]=np.sum(Pgessp,axis=0)
    eta_it+=1


print("Energies: ",Energies)
print("kin Energies: ",EkinsSI)
plt.figure()
plt.plot(kxoffsets,EkinsSI)
plt.hlines(EkinInf,np.min(kxoffsets),np.max(kxoffsets))
plt.show()

plt.figure()
plt.plot(kxoffsets,EnergiesSI)
plt.hlines(EnergyInf,np.min(kxoffsets),np.max(kxoffsets))

plt.show()
print("Exchange Energies: ",Exs)
print(etatildeValues)
print(numberspins)

n=np.argmin(np.abs(EkinsSI-EkinInf))
print("n: ",n,"offset: ",kxoffsets[n],kxoffsets[n])



#print("etatilde min: ",etatildeValues[n])
print("Energy min: ",Energies[n])
print("Ekin min SI: ",EkinsSI[n], "Ekin inf: ", EkinInf)
print("Ex min SI: ",ExsSI[n], "Ex inf: ",EnergyInf-EkinInf)
print("Ehartree: ",Ehartrees[n])
print("Eintges SI: ",EkinsSI[n]+ExsSI[n], EnergiesSI[n])
print("kFmax: ",kFmaxValues[n])
print("dens: ",np.sum(numberspins_species[n,:])/area[n],'dens inf: ',n1,n1*area[n], 'area: ',area[n])
print("numspins: ", numberspins_species[n,:])
thetax=kxoffsets[n]
thetay=kxoffsets[n]
k_=np.zeros(4)
k_[0]=kFmaxValues[n]*2
k_[1]=Nkx
k_[2]=thetax
k_[3]=thetay

ar1=np.array([EnergiesSI[n],EkinsSI[n],ExsSI[n],EhartreesSI[n]])
ar2=np.array([Energies[n],Ekins[n],Exs[n],Ehartrees[n]])
np.savetxt("EgesEkinExsEhartreePerSpinSI.txt",ar1)
np.savetxt("EgesEkinExsEhartree.txt",ar2)

np.savetxt("kFmax_ellipse.txt", k_)
np.savetxt("numspins.txt",numberspins_species[n,:])

if sym_ind==2:
  np.savetxt("Pmatrix_ellipse_sp.txt",Pmatricessp[n,:,:])
  np.savetxt("Pmatrix_ellipse_vp.txt",Pmatricesvp[n,:,:])
  np.savetxt("Pmatrix_ellipse_vp2.txt",Pmatricesvp2[n,:,:])

elif sym_ind==3:
    np.savetxt("Pmatrix_ellipse_svp.txt",Pmatricessp[n,:,:])


elif sym_ind==1:
    np.savetxt("Pmatrix_ellipse_sym.txt",Pmatricessp[n,:,:])
#np.savetxt("Pmatrix_ellipse.txt",Pges)
#np.savetxt("numspins.txt",numspins)
#    np.savetxt("Tmatrix{}.txt".format(i),T_final[i,:,:])

