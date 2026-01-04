import numpy as np
import matplotlib.pyplot as plt

U_=np.loadtxt("Uscreenedto1.txt")
distances=U_[0,:]
U=U_[1,:]
c=U[0]*distances[0]
plt.figure()
plt.plot(distances,U)
plt.plot(distances,c/distances)
plt.show()

plt.figure()
plt.plot(distances,U-c/distances)
plt.show()

plt.figure()
plt.plot(distances,U-50/distances)
plt.show()
print("cvalue: ",c)

np.savetxt("CvalueForCusp.txt",np.array([c]))
