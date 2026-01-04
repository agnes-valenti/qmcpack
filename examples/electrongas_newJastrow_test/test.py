import numpy as np


Ehf=np.loadtxt('EgesEkinExsEhartree.txt')
EhfSI=np.loadtxt('EgesEkinExsEhartreePerSpinSI.txt')
factorSI=EhfSI[0]/Ehf[0]

data = np.genfromtxt('EG.s000.scalar.dat')
en=data[:,1]
ekin=data[:,4]
eint=np.mean(data[:,5])
esqr=data[:,2]

print(eint-Ehf[3],Ehf[2])
print((eint-Ehf[3])*factorSI,EhfSI[2])
