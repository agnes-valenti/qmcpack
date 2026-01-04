import numpy as np 


refU=np.loadtxt("Uscreened1to5001_reference.txt")
testU=np.loadtxt("Uscreened1to5001_test.txt")
print(np.shape(refU), np.shape(testU), (refU-testU)[:,-150:])

refU=np.loadtxt("Uscreenedto1_reference.txt")
testU=np.loadtxt("Uscreenedto1_test.txt")

print(np.shape(refU), np.shape(testU), (refU[:,-2]),testU[:,-2])
