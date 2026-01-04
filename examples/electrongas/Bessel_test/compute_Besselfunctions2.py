import numpy as np 
import scipy as sc
import scipy.special

d=200
Nvalues=100000
rvalues=np.linspace(1+1.0/Nvalues,1000,Nvalues)
#rvalues=np.exp(rvalues)

print(rvalues[:30])
a=np.ones(10**8)
print(np.sum(a))
cutoffs=(d/rvalues*100).astype(int)
for i in range(Nvalues):
	cutoffs[i]=max(cutoffs[i],20)
print(cutoffs[-30:])

Uscreened=np.zeros(Nvalues)
for i in range(Nvalues):
	#n=np.arange(0,cutoffs[i],1)
	Uscreened[i]=np.exp(-(rvalues[i]**2)/2)
	print(Uscreened[i])

print(np.shape(np.vstack((rvalues,Uscreened))))
np.savetxt("Uscreened1to1000.txt",np.vstack((rvalues,Uscreened)))