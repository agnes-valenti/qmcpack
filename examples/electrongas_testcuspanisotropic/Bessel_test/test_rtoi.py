import numpy as np 
import scipy as sc
import scipy.special

d=200
Nvalues1=10000
Nvalues2=100000
#rvalues=np.linspace(-6,0,Nvalues+1)
#rvalues=np.exp(rvalues)

rvalues=np.loadtxt("Uscreenedto1.txt")[0,:]
U=np.loadtxt("Uscreenedto1.txt")[1,:]
r2=np.loadtxt("Uscreened1to5001.txt")[0,:]
U2=np.loadtxt("Uscreened1to5001.txt")[1,:]

rvalues=np.hstack((rvalues,r2))
U=np.hstack((U,U2))*4.0/d

def rnew(ind):
	return np.exp(-6+ind*6.0/Nvalues1)

def get_index(rvalue):
	if rvalue<=1:
		return int((np.log(rvalue)+6.0)*Nvalues1/6.0)
	else:
		return int(min((rvalue-1)*Nvalues2/5000-1+(Nvalues1+1),Nvalues1+Nvalues2))

def get_linear_interpolated_U(rvalue):
	ind1=get_index(rvalue)
	diff1=rvalue-rvalues[ind1]
	return (U[ind1+1]-U[ind1])/(rvalues[ind1+1]-rvalues[ind1])*diff1+U[ind1]


for i in range(100):
   #print(rvalues[i],rnew(i))
   #print(get_index(rvalues[i]),"\n")
   a=np.random.rand()*5001
   print(get_index(a),a,rvalues[get_index(a)],rvalues[get_index(a)-1],rvalues[get_index(a)+1])
   print(get_linear_interpolated_U(a),U[get_index(a)],U[get_index(a)+1], "\n")


"""
Nvalues2=100000
rvalues=np.linspace(1+5000.0/Nvalues2,5001,Nvalues2+1)
#rvalues=np.hstack((np.loadtxt("Uscreenedto1.txt")[0,:],rvalues))

def get_r(ind):
	return 1+5000/Nvalues2*(1+ind)

for i in range(0,20):
	print(rvalues[i],get_r(i))
"""