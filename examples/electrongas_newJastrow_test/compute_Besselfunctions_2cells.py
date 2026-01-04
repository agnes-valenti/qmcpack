import numpy as np 
import scipy as sc
import scipy.special

a_ = 0.56605e-9; 

d=200*10**(-9)/a_
Nvalues=10000
rvalues=np.linspace(-6,0,Nvalues+1)
rvalues=np.exp(rvalues)

print(rvalues[:30])
a=np.ones(10**8)
print(np.sum(a))
cutoffs=(d/rvalues*100).astype(int)
for i in range(Nvalues+1):
	cutoffs[i]=max(cutoffs[i],20)
print(cutoffs[-30:])

Uscreened=np.zeros(Nvalues+1)
for i in range(Nvalues+1):
	n=np.arange(0,cutoffs[i],1)
	Uscreened[i]=np.sum(scipy.special.kn(0,(2*n + 1)*np.pi*rvalues[i]/d))
	print(Uscreened[i])

print(np.shape(np.vstack((rvalues,Uscreened))))
np.savetxt("Uscreenedto1_testnewd.txt",np.vstack((rvalues,Uscreened)))
