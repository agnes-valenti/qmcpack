import numpy as np
import matplotlib.pyplot as plt


class wf:
    def __init__(self,k0=0,k1=0):
        self.k0=k0
        self.k1=k1
        self.A0 = -1.0 / 6.0
        self.A1 = 3.0 / 6.0
        self.A2 = -3.0 / 6.0
        self.A3 = 1.0 / 6.0;
        self.A4 = 3.0 / 6.0
        self.A5 = -6.0 / 6.0
        self.A6 = 0.0 / 6.0
        self.A7 = 4.0 / 6.0;
        self.A8 = -3.0 / 6.0
        self.A9 = 3.0 / 6.0
        self.A10 = 3.0 / 6.0
        self.A11 = 1.0 / 6.0;
        self.A12 = 1.0 / 6.0
        self.A13 = 0.0 / 6.0
        self.A14 = 0.0 / 6.0
        self.A15 = 0.0 / 6.0;

        self.NumParams=22
        self.NumParamsa2=7
        self.NumParamsb2=7
        self.loadcoefs()

    def loadcoefs(self,component=0):
        self.NumParamsu=self.NumParams-self.NumParamsa2-self.NumParamsb2-1
        self.numCoefs=self.NumParamsu+3+self.NumParamsa2+3+self.NumParamsb2+3
        Parameters=np.loadtxt("coefs.txt")[component*self.NumParams:(component+1)*self.NumParams]
        #coefs=np.array([np.loadtxt("coefs.txt")])
        self.cutoff_radius=158.012 #352.1597518740  #125.66370614359172 #0 #set!!
        self.coefs=np.zeros(self.numCoefs)
        for i in range(self.NumParamsu):
            self.coefs[i] = Parameters[i];

        for i in range(self.NumParamsa2):
           self.coefs[self.NumParamsu+3+i] = Parameters[self.NumParamsu+i];

        for i in range(self.NumParamsb2):
           self.coefs[self.NumParamsu+3+self.NumParamsa2+3+i] = Parameters[self.NumParamsu+self.NumParamsa2+i];

        self.etaVar=Parameters[self.NumParamsu+self.NumParamsa2+self.NumParamsb2];
        self.numKnotsu=self.NumParamsu+3-2
        self.numKnotsa2=self.NumParamsa2+3-2
        self.numKnotsb2=self.NumParamsb2+3-2

        cutoff_factor=1.0;
    
        cutoff_factor=np.sqrt(self.etaVar);
    
   
        cutoff_new=self.cutoff_radius*cutoff_factor;

        self.DeltaRu       = self.cutoff_radius*1.0 / (self.numKnotsu - 1);
        self.DeltaRInvu    = 1.0 / self.DeltaRu;

        self.DeltaRa2       = cutoff_new*1.0 / (self.numKnotsa2 - 1);
        self.DeltaRInva2    = 1.0 / self.DeltaRa2;

        self.DeltaRb2       = cutoff_new *1.0/ (self.numKnotsb2 - 1);
        self.DeltaRInvb2    = 1.0 / self.DeltaRb2;

        #self.numCoefs=np.shape(self.coefs)[0]
    
        #self.DeltaR=self.cutoff_radius/(self.numKnots-1)
        #self.DeltaRInv=1.0/self.DeltaR

    def slaterPart(self,x0,x1):
        print(np.dot(self.k0,x0),np.dot(self.k1,x1),np.dot(self.k0,x1),np.dot(self.k1,x0))
        z=1.0/np.sqrt(2)*(np.exp((np.dot(self.k0,x0)+np.dot(self.k1,x1))*1j)- np.exp((np.dot(self.k0,x1)+np.dot(self.k1,x0))*1j)  )
        return z #np.conjugate(z)*zeros


    def Bspline(self,r):
      if r>self.cutoff_radius:
         return 0;
      else:
        r *= self.DeltaRInvu
        i  = int(r);
        #print(i)
        t = r - i;
        d1      = self.coefs[i + 0] * (((self.A0 * t + self.A1) * t + self.A2) * t + self.A3);
        d2      = self.coefs[i + 1] * (((self.A4 * t + self.A5) * t + self.A6) * t + self.A7);
        d3      = self.coefs[i + 2] * (((self.A8 * t + self.A9) * t + self.A10) * t + self.A11);
        d4      = self.coefs[i + 3] * (((self.A12 * t + self.A13) * t + self.A14) * t + self.A15);
        d = (d1 + d2 + d3 + d4);
        return d

    def Bspline2(self, r, x, y,numpar,component=0):
      if r>self.cutoff_radius:
         return 0;
      else:
        shift_index=0;
        if (numpar==0):
           shift_index=self.NumParamsu+3;
        elif (numpar==1):
           shift_index=self.NumParamsu+3+self.NumParamsa2+3;

        xsquared=x*x
        ysquared=y*y
        DeltaRinvGen=0

        if (numpar==0):
           r=np.sqrt(self.etaVar*xsquared+1.0/self.etaVar*ysquared);
           DeltaRinvGen= self.DeltaRInva2;
    
        elif (numpar==1):
           r=np.sqrt(1.0/self.etaVar*xsquared+self.etaVar*ysquared);
           DeltaRinvGen= self.DeltaRInvb2;
    

        r *= DeltaRinvGen;
        i       = int(r);
        t = r - i
        d1      = self.coefs[shift_index + i + 0] * (((self.A0 * t + self.A1) * t + self.A2) * t + self.A3);
        d2      = self.coefs[shift_index + i + 1] * (((self.A4 * t + self.A5) * t + self.A6) * t + self.A7);
        d3      = self.coefs[shift_index + i + 2] * (((self.A8 * t + self.A9) * t + self.A10) * t + self.A11);
        d4      = self.coefs[shift_index + i + 3] * (((self.A12 * t + self.A13) * t + self.A14) * t + self.A15);
        d = (d1 + d2 + d3 + d4);
        return d


    def evaluateU(self,r,x,y):
        return self.Bspline(r)+self.Bspline2(r,x,y,0)+self.Bspline2(r,x,y,1)

    def jastrowPart(self,x0,x1):
        r=np.sqrt((np.dot(x0-x1,x0-x1)))
        x=(x0-x1)[0]
        y=(y0-y1)[0]
        d=self.evaluateU(r,x,y)
        return np.exp(-d)


    def plotJastrow(self,x=0):
        rvalues = np.linspace(1.0/100.0,self.cutoff_radius-self.cutoff_radius/100.0,100)
        x=np.linspace(-self.cutoff_radius+self.cutoff_radius/100.0,self.cutoff_radius-self.cutoff_radius/100.0,200)
        y=np.linspace(-self.cutoff_radius+self.cutoff_radius/100.0,self.cutoff_radius-self.cutoff_radius/100.0,200)
        X,Y=np.meshgrid(x,y)
        d=np.zeros_like(Y)
     
        for i in range(200):
          for j in range(200):
            xvalue=X[i,j]
            yvalue=Y[i,j]
            #print("x,y: ",xvalue,yvalue)
            d[i,j] = np.exp(-self.evaluateU(np.sqrt(xvalue**2+yvalue**2),xvalue,yvalue))
        #print(np.shape(X))
        print(np.exp(-self.evaluateU(np.sqrt(200**2+0**2),200,0)), np.exp(-self.evaluateU(np.sqrt(200**2+0**2),0,200)))
        plt.figure()
        #plt.title("uu")
        plt.imshow(d,cmap='Oranges_r')
        #plt.contourf(X,Y,d)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("off")
        plt.colorbar()
        plt.clim(-0.2, 0.9)
        plt.savefig("orbital_vp_ud_orange.png")
        #plt.show()
        """
        d=self.evaluateU(np.sqrt(X*X+Y*Y),X,Y)

        plt.figure()
        plt.plot_surface(X,Y,d)
        plt.show()

        
        for it,r in enumerate(rvalues):
 
            d[it] = self.evaluateU(r,0,r)

        print(d)
        print(np.exp(-d))
        plt.figure()
        plt.plot(rvalues,np.exp(-d))
        ymin=0
        ymax=np.max(np.exp(-d))
        #plt.vlines(x, ymin, ymax)
        plt.show()
        """

    def plotSlater(self,x=0):
        rvalues = np.linspace(1.0/100.0,self.cutoff_radius-self.cutoff_radius/100.0,100)
        particle0=np.zeros((100,2))
        particle1=np.zeros((100,2))
        particle1[:,0]=1.0/np.sqrt(2)*rvalues
        particle1[:,1]=1.0/np.sqrt(2)*rvalues


        d=np.zeros_like(rvalues)

        for it,r in enumerate(rvalues):
            print(particle0[it,:],particle1[it,:])
            a=self.slaterPart(particle0[it,:],particle1[it,:])
            print(a)
            d[it] = np.real(np.conjugate(a)*a)*self.jastrowPart(particle0[it,:],particle1[it,:])**2


        print(d)
        plt.figure()
        plt.plot(rvalues,d)
        ymin=0
        ymax=np.max(d)
        #plt.vlines(x, ymin, ymax)
        plt.show()


k0=np.array([0,0]) #-0.025,0])
k1=np.array([0,0]) #-0.025,-0.025])

x0=np.array([3.5947582439e+01,  4.3229612916e+01]) 
x1=np.array([2.3094513107e+01,  1.4955506824e+02])

x0old=np.array([  3.5947582439e+01,  4.3229612916e+01])
x1old=np.array([ 2.2388514583e+01,  1.5065709794e+02 ])

Wf_=wf(k0,k1)
#print(Wf_.slaterPart(x0,x1)/Wf_.slaterPart(x0old,x1old))
#print(Wf_.jastrowPart(x0,x1)/Wf_.jastrowPart(x0old,x1old))

#Wf_.plotJastrow()
Wf_.loadcoefs(0)
#Wf_.plotJastrow()
#Wf_.loadcoefs(2)
Wf_.plotJastrow()

Wf_.loadcoefs(1)
#Wf_.plotJastrow()
#Wf_.loadcoefs(2)
Wf_.plotJastrow()


plt.show()
#Wf_.plotSlater()
