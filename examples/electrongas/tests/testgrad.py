import numpy as np
import logging

A0 = -1.0 / 6.0
A1 = 3.0 / 6.0
A2 = -3.0 / 6.0
A3 = 1.0 / 6.0;
A4 = 3.0 / 6.0
A5 = -6.0 / 6.0
A6 = 0.0 / 6.0
A7 = 4.0 / 6.0;
A8 = -3.0 / 6.0
A9 = 3.0 / 6.0
A10 = 3.0 / 6.0
A11 = 1.0 / 6.0;
A12 = 1.0 / 6.0
A13 = 0.0 / 6.0
A14 = 0.0 / 6.0
A15 = 0.0 / 6.0;

dA0 = 0.0
dA1 = -0.5
dA2 = 1.0
dA3 = -0.5;
dA4 = 0.0
dA5 = 1.5
dA6 = -2.0
dA7 = 0.0;
dA8 = 0.0
dA9 = -1.5
dA10 = 1.0
dA11 = 0.5;
dA12 = 0.0
dA13 = 0.5
dA14 = 0.0
dA15 = 0.0;

d2A0 = 0.0
d2A1 = 0.0
d2A2 = -1.0
d2A3 = 1.0;
d2A4 = 0.0
d2A5 = 0.0
d2A6 = 3.0
d2A7 = -2.0;
d2A8 = 0.0
d2A9 = 0.0
d2A10 = -3.0
d2A11 = 1.0;
d2A12 = 0.0
d2A13 = 0.0
d2A14 = 1.0
d2A15 = 0.0;
kFmax=7.309633904859655362e-02
Nk=5.000000000000000000e+00
deltak_linspace=2.0*kFmax/(Nk-1)
L=2.0*np.pi/deltak_linspace
print(L,deltak_linspace*2)
k0=np.array([-deltak_linspace*2,0])
k1=np.array([deltak_linspace*2,0])


def dMdp(r0,r1,param):
    x0,x1=get_x_backflow(r0,r1,param)
    r,dr=get_dist(r0,r1)
    z=np.zeros((2,2))+0j
    db=bsplinederiv(r)
    z[0,0]=db*np.dot(dr,k0)*np.exp(np.dot(k0,x0)*1j)*1j
    z[0,1]=db*np.dot(-dr,k0)*np.exp(np.dot(k0,x1)*1j)*1j
    z[1,0]=db*np.dot(dr,k1)*np.exp(np.dot(k1,x0)*1j)*1j
    z[1,1]=db*np.dot(-dr,k1)*np.exp(np.dot(k1,x1)*1j)*1j
    return z

def M(x0,x1):
    z=np.zeros((2,2))+0j
    z[0,0]=np.exp(np.dot(k0,x0)*1j)
    z[0,1]=np.exp(np.dot(k0,x1)*1j)
    z[1,0]=np.exp(np.dot(k1,x0)*1j)
    z[1,1]=np.exp(np.dot(k1,x1)*1j)
    return z

def invertM(x0,x1):
    M_=M(x0,x1)
    return np.linalg.inv(M_)

def dM(x0,x1):
    z=np.zeros((2,2,2))+0j
    z[0,0,:]=k0*np.exp(np.dot(k0,x0)*1j)*1j
    z[0,1,:]=k0*np.exp(np.dot(k0,x1)*1j)*1j
    z[1,0,:]=k1*np.exp(np.dot(k1,x0)*1j)*1j
    z[1,1,:]=k1*np.exp(np.dot(k1,x1)*1j)*1j
    return z

def Fij(r0,r1,param):
    x0,x1=get_x_backflow(r0,r1,param)
    Minv=invertM(x0,x1)
    dM_=dM(x0,x1)
    Mcompare=M(x0,x1)
    #print("test invert: " , np.dot(Minv,Mcompare))
   
    #print(" test, M einsum: " ,   np.einsum(Minv, [0,1], dM_, [1,2,3]))
    return np.einsum(Minv, [0,1], dM_, [1,2,3])

def Cj(r0,r1):
    r,dr=get_dist(r0,r1)
    C=np.zeros((2,2))+0j
    C[0,:]=bsplinederiv(r)*dr
    C[1,:]=bsplinederiv(r)*(-dr)
    print("displ: ",dr, "bsplinederiv: ",bsplinederiv(r))
    return C

def test_mderiv(r0,r1,param):
    dmdp=dMdp(r0,r1,param)    
    x0,x1=get_x_backflow(r0,r1,param)
    dmdx=dM(x0,x1)
    dxdp=Cj(r0,r1)
    z=np.zeros((2,2))+0j
    for k in range(2):
        for j in range(2):
            for alpha in range(2):
                z[k,j]+=dmdx[k,j,alpha]*dxdp[j,alpha]

    print("diff dmdp, dmdx*dxdp: ", dmdp-z)
def derivtrace(r0,r1,param):
    F=Fij(r0,r1,param)
    C=Cj(r0,r1)
    for j in range(2):
        print("j", j, "F[j]: ",F[j,j], "C[j]:",C[j])
    return np.einsum(F,[0,0,1], C, [0,1])

def derivtrace2(r0,r1,param):
    dmdp=dMdp(r0,r1,param)    
    x0,x1=get_x_backflow(r0,r1,param)
    Minv=invertM(x0,x1)
    return np.trace(np.dot(Minv,dmdp))

def f2(r0,r1):
    z=(np.exp(np.dot(k0,r0)*1j)*np.exp(np.dot(k1,r1)*1j)- np.exp(np.dot(k0,r1)*1j)*np.exp(np.dot(k1,r0)*1j))
    return z


def f2_normalized(r0,r1):
    z=1.0/np.sqrt(2)*(np.exp(np.dot(k0,r0)*1j)*np.exp(np.dot(k1,r1)*1j)- np.exp(np.dot(k0,r1)*1j)*np.exp(np.dot(k1,r0)*1j))
    return z

def a1(r0,r1):
    return np.exp(np.dot(k0,r0)*1j)*np.exp(np.dot(k1,r1)*1j)

def a2(r0,r1):
    return np.exp(np.dot(k0,r1)*1j)*np.exp(np.dot(k1,r0)*1j)

def get_dist(r0,r1):
    x   = (r0[0] - r1[0]) / L;
    y   = (r0[1] - r1[1]) / L;
    #print(x,y)
    dx    = L * (x - np.round(x)); 
    dy     = L * (y - np.round(y));
    #print(np.round(x),np.round(y))
    #print(x-np.round(x), y-np.round(y))
    return np.sqrt((dx)**2+(dy)**2), np.array([dx,dy])

def bspline(r, param=0.2):

    DeltaRInv=0.058168218407526368
    r *= DeltaRInv;
    i = int(r)
    t   = r-i


    sCoef0 = param
    sCoef1 = 0.3
    sCoef2 = 0.4
    sCoef3 = 0

    d2udr2 = DeltaRInv * DeltaRInv * (sCoef0 * (d2A2 * t + d2A3) + sCoef1 * (d2A6 * t + d2A7) + sCoef2 * (d2A10 * t + d2A11) + sCoef3 * (d2A14 * t + d2A15));

    dudr = DeltaRInv * (sCoef0 * ((dA1 * t + dA2) * t + dA3) + sCoef1 * ((dA5 * t + dA6) * t + dA7) + sCoef2 * ((dA9 * t + dA10) * t + dA11) + sCoef3 * ((dA13 * t + dA14) * t + dA15));

    u = (sCoef0 * (((A0 * t + A1) * t + A2) * t + A3) + sCoef1 * (((A4 * t + A5) * t + A6) * t + A7) + sCoef2 * (((A8 * t + A9) * t + A10) * t + A11) + sCoef3 * (((A12 * t + A13) * t + A14) * t + A15));
    return u;

def bsplinedudr(r, param=0.2):

    DeltaRInv=0.058168218407526368
    r *= DeltaRInv;
    i = int(r)
    t   = r-i


    sCoef0 = param
    sCoef1 = 0.3
    sCoef2 = 0.4
    sCoef3 = 0

    d2udr2 = DeltaRInv * DeltaRInv * (sCoef0 * (d2A2 * t + d2A3) + sCoef1 * (d2A6 * t + d2A7) + sCoef2 * (d2A10 * t + d2A11) + sCoef3 * (d2A14 * t + d2A15));

    dudr = DeltaRInv * (sCoef0 * ((dA1 * t + dA2) * t + dA3) + sCoef1 * ((dA5 * t + dA6) * t + dA7) + sCoef2 * ((dA9 * t + dA10) * t + dA11) + sCoef3 * ((dA13 * t + dA14) * t + dA15));

    u = (sCoef0 * (((A0 * t + A1) * t + A2) * t + A3) + sCoef1 * (((A4 * t + A5) * t + A6) * t + A7) + sCoef2 * (((A8 * t + A9) * t + A10) * t + A11) + sCoef3 * (((A12 * t + A13) * t + A14) * t + A15));
    return dudr;

def bsplinederiv(r):
    DeltaRInv=0.058168218407526368
    r *= DeltaRInv;
    i = int(r)
    t   = r-i
    return ((A0 * t + A1) * t + A2) * t + A3



def get_x_backflow(r0,r1,param=0.2):
    r,dr=get_dist(r0,r1)
    #print("dist:",r,"displ:",dr)
    x0=r0+bspline(r,param)*(dr)
    x1=r1+bspline(r,param)*(-dr)
    return x0,x1

def get_Aij(r0,r1,param=0.2):
    r,dr=get_dist(r0,r1)
    A=np.zeros((2,2,2))+0j
    A[0,0,0]=1+dr[0]/r*bsplinedudr(r,param)*dr[0]+bspline(r,param)
    A[0,1,1]=1+dr[1]/r*bsplinedudr(r,param)*dr[1]+bspline(r,param)
    A[0,1,0]=dr[0]/r*bsplinedudr(r,param)*dr[1]
    A[0,0,1]=dr[1]/r*bsplinedudr(r,param)*dr[0]

    A[1,0,0]=1+dr[0]/r*bsplinedudr(r,param)*dr[0]+bspline(r,param)
    A[1,1,1]=1+dr[1]/r*bsplinedudr(r,param)*dr[1]+bspline(r,param)
    A[1,1,0]=dr[0]/r*bsplinedudr(r,param)*dr[1]
    A[1,0,1]=dr[1]/r*bsplinedudr(r,param)*dr[0]
    return A

def compare_Aij(r0,r1,param=0.2):
    r_a,dr_a=get_dist(r0,r1)
    x0_a=r0+bspline(r_a,param)*(dr_a)
    x1=r1+bspline(r_a,param)*(-dr_a)
    r_b,dr_b=get_dist(r0+np.array([1e-4,0]),r1)
    x0_b=r0+np.array([1e-4,0])+bspline(r_b,param)*(dr_b)

    num_A00=(x0_b-x0_a)/1e-4
    Aij=get_Aij(r0,r1,param)
    print("Aij: ",num_A00, Aij[0,0,0], Aij[0,0,1], Aij[0,1,0], Aij[0,1,1])

def psi(r0,r1,param=0.2):
    x0,x1=get_x_backflow(r0,r1,param)
    #print("r0,r1:", r0,r1)
    #print("x0,x1 (QP): ",x0,x1)
    z=f2(x0,x1)
    return z

def psi2(r0,r1,param=0.2):
    x0,x1=get_x_backflow(r0,r1,param)

    z=np.linalg.det(M(x0,x1))
    return z

def logpsi(r0,r1,param=0.2):
    return np.log(psi(r0,r1,param))

def expA1(r0,r1,param):
    x0,x1=get_x_backflow(r0,r1,param)
    return a1(x0,x1)

def expA2(r0,r1,param):
    x0,x1=get_x_backflow(r0,r1,param)
    return a2(x0,x1)

def derivPsi(r0,r1,param):
    r,dr=get_dist(r0,r1)
    db=bsplinederiv(r)
    z=db*np.dot(dr,k0-k1)*1j*expA1(r0,r1,param)-db*np.dot(dr,k1-k0)*1j*expA2(r0,r1,param)
    return z

def logderiv(r0,r1,param):
    dP=derivPsi(r0,r1,param)
    P=psi(r0,r1,param)
    return dP/P

def grad_num_backward_x(r0,r1,param):
    eps=1e-5
    logvalue1= logpsi(r0,r1,param)
    logvalue2= logpsi(r0-np.array([eps,0]),r1,param)
    return (logvalue1-logvalue2)/eps

def grad_num_forward_x(r0,r1,param):
    eps=1e-5
    logvalue1= logpsi(r0,r1,param)
    logvalue2= logpsi(r0+np.array([eps,0]),r1,param)
    return (logvalue2-logvalue1)/eps

def grad_num_forward_y(r0,r1,param):
    eps=1e-5
    logvalue1= logpsi(r0,r1,param)
    logvalue2= logpsi(r0+np.array([0,eps]),r1,param)
    return (logvalue2-logvalue1)/eps

def grad_num(r0,r1,param):
    return np.array([grad_num_forward_x(r0,r1,param),grad_num_forward_y(r0,r1,param)])

def laplacian_x(r0,r1,param):
    eps=1e-5
    logvalue0= logpsi(r0-np.array([eps,0]),r1,param)
    logvalue1= logpsi(r0,r1,param)
    logvalue2= logpsi(r0+np.array([eps,0]),r1,param)
    return (logvalue0-2*logvalue1+logvalue2)/(eps**2)

def laplacian_y(r0,r1,param):
    eps=1e-5
    logvalue0= logpsi(r0-np.array([0,eps]),r1,param)
    logvalue1= logpsi(r0,r1,param)
    logvalue2= logpsi(r0+np.array([0,eps]),r1,param)
    return (logvalue0-2*logvalue1+logvalue2)/(eps**2)

def laplacian(r0,r1,param):
    return laplacian_x(r0,r1,param)+laplacian_y(r0,r1,param)

#pos0=np.array([1.1739335426e+02,  1.9613684608e+01])
#pos1=np.array([1.2035444075e+02,  1.5424482965e+02])


pos0=np.array([1.1742135309e+02,  2.5838719418e+01]) 
pos1=np.array([1.2035444075e+02,  1.5424482965e+02])
#print(test_mderiv(pos0,pos1,0.2))
#print("psi, psi2",psi(pos0,pos1,0.2), psi2(pos0,pos1,0.2),"\n")
logvalue=logpsi(pos0,pos1,0.200)
print("logvalue: ",(logvalue))

compare_Aij(pos0,pos1,0.2)

#logvalue1=-3.6705247450e-01-4.7123889804e+00*1j
logvalue1= logpsi(pos0,pos1,0.200)
#logvalue2=-3.6704152586e-01-4.7123889804e+00*1j
logvalue2=logpsi(pos0+np.array([0,1e-6]),pos1,0.200)

print("python numerical grad: ", (logvalue2-logvalue1)/1e-6, "grad num: ",grad_num(pos0,pos1,0.2))
print("python numerical grad deriv: ", (grad_num(pos0,pos1,0.2+1e-4)-grad_num(pos0,pos1,0.2))/1e-4)
print("python numerical laplacian: ", laplacian(pos0,pos1,0.2))
print("python numerical laplacianderiv: ", (laplacian(pos0,pos1,0.2+1e-5)-laplacian(pos0,pos1,0.2))/1e-5)
print("logderiv: ",logderiv(pos0,pos1,0.2))
#print("derivtrace: ",derivtrace(pos0,pos1,0.2))
v1=np.log(np.exp(logvalue1)/np.sqrt(2))
v2=np.log(np.exp(logvalue2)/np.sqrt(2))

#print(np.exp(valuecompare1)/f2(x0,x1))
eps=1e-5
eloc1=3.030309486351475e+01 
eloc2= 3.030308122160950e+01
print("numerical derivative eloc: ", (eloc2-eloc1)/eps)
