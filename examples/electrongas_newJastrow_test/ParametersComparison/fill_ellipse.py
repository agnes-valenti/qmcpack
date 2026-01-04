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
Etatilde=1.0 #1.0/5.4 #5.79 #4.013381993233915   #5
Nkx=71
Nky=71

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
kyFmax=max(kxF,kyF)

k1x=np.linspace(-kxF,kxF,Nkx)
k1y=np.linspace(-kyF,kyF,Nky)

#k1x=np.linspace(-kFmax,kFmax,Nkx)
#k1y=np.linspace(-kFmax,kFmax,Nky)



#print(k1x,k1y)
"""
fx=lambda ky: np.sqrt((kF**2-Etatilde**(-tau/2.0)*ky**2)*Etatilde**(-tau/2.0))
fy=lambda kx: np.sqrt((kF**2-Etatilde**(tau/2.0)*kx**2)*Etatilde**(tau/2.0))

#kxmax=(int((2*kxF)/deltaK)*deltaK+6*deltaK)/2
#kymax=(int((2*kyF)/deltaK)*deltaK+6*deltaK)/2
kxmax=(int((kxF)/deltaK)*deltaK+3*deltaK)
kymax=(int((kyF)/deltaK)*deltaK+3*deltaK)
#kxmax=(int((2*kxF)/deltaK)*deltaK+6*deltaK)/2
if sym_ind==2:
 if ((N/2)%2==0):
   print("hi!")
   kxmax=int((kxF)/deltaK)*deltaK+3.5*deltaK

 if ((N/2)%4==0):
   kymax=int((kyF)/deltaK)*deltaK+3.5*deltaK

kxges=np.arange(-kxmax,kxmax+deltaK/3,deltaK)

kyges=np.arange(-kymax,kymax+deltaK/3,deltaK)
print(kyges,kxges)
print(kxges[0], kxges[-1])
print(kyges[0], kyges[-1])


"""
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
Eta=1.0 #5.79

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
    #energy+=0.5*V0*n0_*dens*Area   #uncomment!! (- sign: density removed)
    #interaction+=0.5*V0*n0_*dens*Area
    return energy/Area, dens, interaction/Area



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
    Energy, dens, interaction=get_energy(P,Tnew,n1)
    Nspins=np.sum(P)*factordeg
    print("\n")
    print("Energy/N: ",Energy,"Ekin/N SI: ",(Energy-interaction)*qe*1e-3/n1, "ExchangeEn/N SI: ",interaction*qe*1e-3/n1,dens,n1,dens*Area,n1*Area,"\n")
    print("Energy/N with dens instead of n1: ",Energy,"Ekin/N SI: ",(Energy-interaction)*qe*1e-3/dens, "ExchangeEn/N SI: ",interaction*qe*1e-3/dens,dens,n1,dens*Area,n1*Area,"\n")
    print("Energy, for VMC: ",Energy*Area,"Ekin: ",(Energy-interaction)*Area, "ExchangeEn: ",interaction*Area,dens,n1,dens*Area,n1*Area,"\n")
    #sk,mx,my=sort_filledkit(P)
    print("Energy, for VMC in SI: ",Energy*Area,"Ekin: ",(Energy-interaction)*Area, "ExchangeEn: ",interaction*Area,dens,n1,dens*Area,n1*Area,"\n")
    


 #sk,mx,my=sort_filledkit(P)

    ndiff=np.round(n1*Area-dens*Area)
    ndiff0=ndiff

    print("ndiff0: ",ndiff0, "density diff percentage: ", (n1-dens)/n1)
    """
    if ndiff0<0:
       ndiff=ndiff
       sk,ens,mx,my=sort_filledkit(P)
       i=0
       while ndiff<-3*factordeg:
          if mx[i]<0.5 and my[i]<0.5:
             P[sk[i]]=0
             P[kit_mirroredxaxis(sk[i])]=0
             P[kit_mirroredyaxis(sk[i])]=0
             P[kit_mirroredxyaxes(sk[i])]=0
             ndiff=ndiff+4*factordeg
          if mx[i]>0.5 and my[i]<0.5:
             P[sk[i]]=0
             P[kit_mirroredxaxis(sk[i])]=0
             ndiff=ndiff+2*factordeg
          if mx[i]<0.5 and my[i]>0.5:
             P[sk[i]]=0
             P[kit_mirroredyaxis(sk[i])]=0
             ndiff=ndiff+2*factordeg
          i+=1

    if ndiff0>0:
       sk,ens,mx,my=sort_notfilledkit(P)
       i=0
       while ndiff>3*factordeg:
          if mx[i]<0.5 and my[i]<0.5:
             P[sk[i]]=1
             P[kit_mirroredxaxis(sk[i])]=1
             P[kit_mirroredyaxis(sk[i])]=1
             P[kit_mirroredxyaxes(sk[i])]=1
             ndiff=ndiff-4*factordeg
          if mx[i]>0.5 and my[i]<0.5:
             P[sk[i]]=1
             P[kit_mirroredxaxis(sk[i])]=1
             ndiff=ndiff-2*factordeg
          if mx[i]<0.5 and my[i]>0.5:
             P[sk[i]]=1
             P[kit_mirroredyaxis(sk[i])]=1
             ndiff=ndiff-2*factordeg
          i+=1

    print("ndiff: ",ndiff, np.sum(P)*2)


    
    if ndiff!=0:
         sk1,ens1,mx1,my1=sort_filledkit(P)
         sk2,ens2,mx2,my2=sort_notfilledkit(P)
         kitremovexzero=sk1[np.argmax(ens1*mx1)]

         #print("kitremovexzero: ",Kv_[kitremovexzero], ens1)
         #print(mx1)
         enremovexzero=np.max(ens1*mx1)
         #print(enremovexzero)
         kitremoveyzero=sk1[np.argmax(ens1*my1)]
         enremoveyzero=np.max(ens1*my1)
         if enremovexzero==0:
            enremovexzero=-np.max(ens2)*100
         if enremoveyzero==0:
            enremoveyzero=-np.max(ens2)*100

         #print("enaddxzero: ",enaddxzero, "enaddyzero:",enaddyzero)
         kitaddxzero=0
         kitaddyzero=0
         enaddxzero=np.max(ens2)*100
         enaddyzero=np.max(ens2)*100

         for kit in range(Nkx*Nky):
             #print(mx2[kit],my2[kit])
             if mx2[kit]>0.5 and my2[kit]<0.5:
                kitaddxzero=sk2[kit]
                #print("kitaddxzero: ",kitaddxzero,Kv_[kitaddxzero])
                enaddxzero=ens2[kit]
                print("kit: ",kit)
                break
         #print("enaddxzero: ",enaddxzero, "enaddyzero:",enaddyzero)
         for kit in range(Nkx*Nky):
             if mx2[kit]<0.5 and my2[kit]>0.5:
                kitaddyzero=sk2[kit]
                enaddyzero=ens2[kit]
                print("kit: ",kit)
                break
         print("enaddxzero: ",enaddxzero, "enaddyzero:",enaddyzero, kitaddxzero, kitaddyzero, Kv_[kitaddxzero], Kv_[kitaddyzero],ens2[0])

         
         kitadd4=0
         enadd4=0
         for kit in range(Nkx*Nky):
             if mx2[kit]<0.5 and my2[kit]<0.5:
                kitadd4=sk2[kit]
                enadd4=ens2[kit]
                break

         kitremove4=sk1[np.argmax(ens1*(1-mx1)*(1-my1))]
         enremove4=np.max(ens1*(1-mx1)*(1-my1))


         #######################################
         if ndiff==2*factordeg:

            print("ndiff 2xfactordeg, add 2")
            Energy_yzero=0
            P_yzero=0
            Energy_xzero=0
            P_xzero=0
            xzero=False
            yzero=False
            print("kyges, kxges",kyges, np.shape(kyges[np.abs(kyges)<0.5])[0], kxges, np.shape(kxges[np.abs(kxges)<0.5])[0])
            print("\n")
            if np.shape(kyges[np.abs(kyges)<1e-7])[0]>0.5: #0 in ky
               yzero=True
               #add 4, remove 2 y -> P11
               #add 2 y -> P12
               P11=np.copy(P)
               P11[kitadd4]=1
               P11[kit_mirroredxaxis(kitadd4)]=1
               P11[kit_mirroredyaxis(kitadd4)]=1
               P11[kit_mirroredxyaxes(kitadd4)]=1
               P11[kitremoveyzero]=0
               P11[kit_mirroredyaxis(kitremoveyzero)]=0
               
               T11=HF_step(P11,n1)
               Energy_yzero1, dens_yzero1, interaction=get_energy(P11,T11,n1)

               P12=np.copy(P)
               P12[kitaddyzero]=1
               P12[kit_mirroredyaxis(kitaddyzero)]=1
               T12=HF_step(P12,n1)
               Energy_yzero2, dens_yzero2, interaction=get_energy(P12,T12,n1)
               if Energy_yzero2<Energy_yzero1:
                  Energy_yzero=Energy_yzero2
                  P_yzero=np.copy(P12)
               else:
                  Energy_yzero=Energy_yzero1
                  P_yzero=np.copy(P11)

            if np.shape(kxges[np.abs(kxges)<1e-7])[0]>0: #0 in kx
               xzero=True
               #add 4, remove 2 x -> P11
               #add 2 x -> P12       
               P11=np.copy(P)
               P11[kitadd4]=1
               P11[kit_mirroredxaxis(kitadd4)]=1
               P11[kit_mirroredyaxis(kitadd4)]=1
               P11[kit_mirroredxyaxes(kitadd4)]=1
               P11[kitremovexzero]=0
               P11[kit_mirroredxaxis(kitremovexzero)]=0
               
               T11=HF_step(P11,n1)
               Energy_xzero1, dens_xzero1, interaction=get_energy(P11,T11,n1)

               P12=np.copy(P)
               P12[kitaddxzero]=1
               P12[kit_mirroredxaxis(kitaddxzero)]=1
               T12=HF_step(P12,n1)
               Energy_xzero2, dens_xzero2, interaction=get_energy(P12,T12,n1) 
               if Energy_xzero2<Energy_xzero1:
                  Energy_xzero=Energy_xzero2
                  P_xzero=np.copy(P12)
               else:
                  Energy_xzero=Energy_xzero1
                  P_xzero=np.copy(P11)
            
            if xzero and yzero:
               print("xyzero")
               if Energy_xzero<Energy_yzero:
                  P=np.copy(P_xzero)
               else:
                  P=np.copy(P_yzero)
 
            elif xzero and not yzero:
              print("xzero")
              P=np.copy(P_xzero)

            elif yzero and not xzero:
              print("yzero")
              P=np.copy(P_yzero)

         
         if ndiff==-2*factordeg:
            print("ndiff -2xfactordeg, remove 2")
            Energy_yzero=0
            P_yzero=0
            Energy_xzero=0
            P_xzero=0
            xzero=False
            yzero=False
            if np.shape(kyges[np.abs(kyges)<1e-7])[0]>0.5: #0 in ky
               yzero=True
               #remove 4, add 2 y -> P11
               #remove 2 y -> P12
               P11=np.copy(P)
               P11[kitremove4]=0
               P11[kit_mirroredxaxis(kitremove4)]=0
               P11[kit_mirroredyaxis(kitremove4)]=0
               P11[kit_mirroredxyaxes(kitremove4)]=0
               P11[kitaddyzero]=1
               P11[kit_mirroredyaxis(kitaddyzero)]=1
               
               T11=HF_step(P11,n1)
               Energy_yzero1, dens_yzero1, interaction=get_energy(P11,T11,n1)

               P12=np.copy(P)
               P12[kitremoveyzero]=0
               print("Kv: ",Kv_[kitremoveyzero],kit_mirroredxaxis(kitremoveyzero),Kv_[kit_mirroredyaxis(kitremoveyzero)])
               P12[kit_mirroredyaxis(kitremoveyzero)]=0
               T12=HF_step(P12,n1)
               Energy_yzero2, dens_yzero2, interaction=get_energy(P12,T12,n1)
               if Energy_yzero2<Energy_yzero1:
                  print("yzero1")
                  Energy_yzero=Energy_yzero2
                  P_yzero=np.copy(P12)
               else:
                  print("yzero2")
                  Energy_yzero=Energy_yzero1
                  P_yzero=np.copy(P11)

            if np.shape(kxges[np.abs(kxges)<1e-7])[0]>0.5: #0 in kx
               xzero=True
               #remove 4, add 2 y -> P11
               #remove 2 y -> P12
               P11=np.copy(P)
               P11[kitremove4]=0
               P11[kit_mirroredxaxis(kitremove4)]=0
               P11[kit_mirroredyaxis(kitremove4)]=0
               P11[kit_mirroredxyaxes(kitremove4)]=0
               P11[kitaddxzero]=1
               P11[kit_mirroredxaxis(kitaddxzero)]=1
               
               T11=HF_step(P11,n1)
               Energy_xzero1, dens_xzero1, interaction=get_energy(P11,T11,n1)

               P12=np.copy(P)
               P12[kitremovexzero]=0
               P12[kit_mirroredxaxis(kitremovexzero)]=0
               T12=HF_step(P12,n1)
               Energy_xzero2, dens_xzero2, interaction=get_energy(P12,T12,n1)
               if Energy_xzero2<Energy_xzero1:
                  Energy_xzero=Energy_xzero2
                  P_xzero=np.copy(P12)
               else:
                  Energy_xzero=Energy_xzero1
                  P_xzero=np.copy(P11)


            if xzero and yzero:
               if Energy_xzero<Energy_yzero:
                  print("xzero")
                  P=np.copy(P_xzero)
               else:
                  print("yzero")
                  P=np.copy(P_yzero)
 
            elif xzero and not yzero:
              P=np.copy(P_xzero)

            elif yzero and not xzero:
              P=np.copy(P_yzero)
         ####################################

         

    Tnew=HF_step(P,n1)
    Energy, dens, interaction=get_energy(P,Tnew,n1)
    print("Energy new",Energy,  (Energy-interaction)*qe*1e-3/n1, interaction*qe*1e-3/n1,  dens,n1,dens*Area,n1*Area)
    """
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

