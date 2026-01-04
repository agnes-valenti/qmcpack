import numpy as np
import matplotlib.pyplot as plt

eta=1.0/4.0
f=lambda phi: np.sqrt(eta*np.cos(phi)**2+1.0/eta*np.sin(phi)**2)

phivalues=np.linspace(0,2*np.pi,100)
plt.figure()
plt.plot(phivalues,f(phivalues))
plt.hlines(np.sqrt(1.0/eta),0,2.0*np.pi)
plt.show()
