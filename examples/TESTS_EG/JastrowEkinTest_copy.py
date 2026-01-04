import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)

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
        
        hbar=1


        n0=0.1/312.097  #10^11 cm^-2
        n1=n0*1.0 #200.0
        a = 0.56605e-9;        # AlAs lattice spacing in nanometers
        d = 100*10**(-9)/a; # Distance to gate in units of unit cell size
        epsilon_d = 10; # AlAs dielectric constant (GaAs? AlGaAs?)

        qe = 1.60217662e-19; # electron charge
        ke = 8.99e9;         # Coulomb constant
        hbar=1.0545718*1e-34;  #hbar (in SI units)
        a = 0.56605e-9;        # AlAs lattice spacing in nanometers

        e_squared = 1.0*ke*qe/a*1e3/epsilon_d; # e^2 in units of meV*a
        me=9.1093837015*1e-31  #kg - umrechnen?? 1/a? meV?
        self.mstar=0.457*me*qe/(hbar**2*1e3)*a*a  #mstar in units of (1/(meV*a^2)) (from hbar) -> 1/mstar*k^2->meV  (a needed here?)
        
        #self.loadcoefs()

        self.create_slaterPart()


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

    def load_kgrid(self):
        self.Eta=np.loadtxt("EtaEtaVarmax.txt")[0]
        
        #mx=Eta**(-0.5*Tau)
        #my=Eta**(0.5*Tau);

        ks=np.loadtxt("kFmax.txt");
  
        Kstart=ks[0]

        deltak_linspace=ks[1]

        helperkgrid=ks[2];
  
        self.kgridmax=int(np.round(helperkgrid));

        self.theta_x=ks[3];
        self.theta_y=ks[4];

        self.Kx=np.zeros((int(self.kgridmax),int(self.kgridmax)));
        self.Ky=np.zeros((int(self.kgridmax),int(self.kgridmax)));

        k_it=0;
        for i in range(self.kgridmax):
            for j in range(self.kgridmax):
                self.Kx[i,j]=Kstart+i*deltak_linspace+self.theta_x;
                self.Ky[i,j]=Kstart+j*deltak_linspace+self.theta_y;
                k_it+=1
        #print(self.Kx)
        #print(self.Ky)


    def create_slaterPart(self):
        self.load_kgrid()
        numspins_=np.loadtxt("numspins.txt")
        Pmatrix=np.loadtxt("Pmatrix.txt")
        Tau=np.array([-1,-1,1,1])
        self.Kvalues=[]
        self.Tauvalues=[]
        self.ngroups=0
        self.nspins=[]
        #print(Pmatrix)
        for flavour in range(4):
            #print(nspins)
            if numspins_[flavour]>0.5:
                #print("hi 1")
                k_=np.zeros((int(numspins_[flavour]),2))
                k_it=0
                it=0
                for i0 in range(self.kgridmax):
                    for i1 in range(self.kgridmax):
                        #print(k_it,Pmatrix[k_it,flavour])
                        if Pmatrix[k_it,flavour]>0.5:
                           #print("hi")
                           k_[it,:]=np.array([self.Kx[i0,i1], self.Ky[i0,i1]]); 
                           it+=1
          
                        k_it+=1;
                self.ngroups+=1
                self.nspins.append(int(numspins_[flavour]))

                self.Kvalues.append(k_)
                self.Tauvalues.append(Tau[flavour])

        #print(self.Kvalues, self.Tauvalues)
        #print(2*np.dot(self.Kvalues[0],np.transpose(self.Kvalues[0])))
        Ekin_slater=0
        for ig in range(self.ngroups):
            k_=self.Kvalues[ig]
            numspins=np.shape(self.Kvalues[ig])[0]
            for i in range(numspins):
                #print(self.Tauvalues[ig])
                #print(numspins,k_)
                #print(k_[i,0])
                Ekin_slater+=1.0/(2*self.mstar)*(self.Eta**(0.5*self.Tauvalues[ig])*k_[i,0]**2+ self.Eta**(-0.5*self.Tauvalues[ig])*k_[i,1]**2)
        print("Kvalues: ",self.Kvalues)
        print("Ekin slater: ",Ekin_slater)

    def slaterPart(self,xvalues):
        #xvalues: [ngroups] (nspins[flavor],2)
        psi=1
        for ig in range(self.ngroups):
            numspins=self.nspins[ig]
            k_=self.Kvalues[ig]
            x_=xvalues[ig]
            D_=np.zeros((numspins,numspins))+0j
            for i in range(numspins):
                for j in range(numspins):
                    #print(i,j)
                    #print("k_[i,:]", k_[i,:], "x: ",x_[j,:], np.dot(k_[i,:],x_[j,:]))
                    D_[i,j]=np.exp(np.dot(k_[i,:],x_[j,:])*1j)
            D_=1.0/np.sqrt(np.math.factorial(numspins))*D_ 
            #print("D_: ",D_)
            psi=psi*np.linalg.det(D_)
        #print("psi: ",psi)
        return psi


    def Psi(self,xvalues):
        return self.slaterPart(xvalues)
    #def slaterPart(self,x0,x1):
    #    print(np.dot(self.k0,x0),np.dot(self.k1,x1),np.dot(self.k0,x1),np.dot(self.k1,x0))
    #    z=1.0/np.sqrt(2)*(np.exp((np.dot(self.k0,x0)+np.dot(self.k1,x1))*1j)- np.exp((np.dot(self.k0,x1)+np.dot(self.k1,x0))*1j)  )
    #    return z #np.conjugate(z)*zeros

    def Psideriv2(self,dim,ig,iat,xvalues,h=1e-3):
        xvaluesDeltaP=xvalues.copy()
        xvaluesDeltaM=xvalues.copy()
        xP_=np.copy(xvalues[ig])
        xM_=np.copy(xvalues[ig])

        xP_[iat,dim]=xP_[iat][dim]+h
        xM_[iat,dim]=xM_[iat][dim]-h

        xvaluesDeltaP[ig]=xP_
        xvaluesDeltaM[ig]=xM_
        #print("hi")
        #print(xvalues[ig]-xvaluesDeltaP[ig])
        #print(xvaluesDeltaP)
        #print(xvaluesDeltaM)

        deriv=(self.Psi(xvaluesDeltaP)-2*self.Psi(xvalues)+self.Psi(xvaluesDeltaM))/(h**2)
        #print(self.Psi(xvalues), self.Psi(xvaluesDeltaP), deriv)
        return deriv 

    def Psiderivtest(self,dim,ig,iat,xvalues,h=1e-4):
        xvaluesDeltaM=xvalues.copy()
        xM_=np.copy(xvalues[ig])
        xM_[iat,dim]=xM_[iat][dim]-h
        xvaluesDeltaM[ig]=xM_
        

        deriv=(self.Psi(xvalues)-self.Psi(xvaluesDeltaM))/(h*self.Psi(xvalues))
        print(deriv,self.Kvalues[ig][iat,dim],self.Kvalues)
        #print(self.Psi(xvalues), self.Psi(xvaluesDeltaP), deriv)
        
    def Ekinloc_finitediff(self,xvalues,h=1e-5):
        Ekin=0
        psi=self.Psi(xvalues)
        for ig in range(self.ngroups):
            for iat in range(self.nspins[ig]):
                for dim in range(2):
                    signtau=1
                    if dim==0:
                        signtau=1
                    else:
                        signtau=-1
                    Ekin+=(-1.0/(2.0*self.mstar)*self.Eta**(0.5*self.Tauvalues[ig]*signtau)*self.Psideriv2(dim,ig,iat,xvalues,h))
        print(psi)
        return Ekin/psi 

    

    def evaluateV(self,r):
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

    def evaluateV2(self, r, x, y,numpar,component=0, tauvalue=0):
      if r>self.cutoff_radius:
         return 0;
      else:
        shift_index=0;
        if (numpar==0):
           shift_index=self.NumParamsu+3;
        elif (numpar==1):
           shift_index=self.NumParamsu+3+self.NumParamsa2+3;

        if (tauvalue>1.5):
           if (numpar==0):
             shift_index=self.NumParamsu+3+self.NumParamsa2+3;
           elif (numpar==1):
             shift_index=self.NumParamsu+3;
  
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
        return self.evaluateV(r)+self.evaluateV2(r,x,y,0)+self.evaluateV2(r,x,y,1)

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

class ParticlePos:
    def __init__(self):
        numspins=np.loadtxt("numspins.txt")
        self.ngroups=0
        self.nspins=[]
        for flavour in range(4):
            #print(nspins)
            if numspins[flavour]>0.5: 
                self.ngroups+=1
                self.nspins.append(int(numspins[flavour]))
        deltak=np.loadtxt("kFmax.txt")[1];
        self.L=(2.0*np.pi)/deltak

        self.init_random()

    def init_random(self):
        self.xvalues=[]
        for ig in range(self.ngroups):
            x_=np.random.rand(self.nspins[ig],2)*self.L
            #x_[0,0]=0
            #x_[1,0]=1
            #x_[0,1]=0
            #x_[1,1]=1

            self.xvalues.append(x_)

    def get_xvalues(self):
        return self.xvalues

class TestMain:
    def __init__(self):
        self.initialized=1

    def run(self):
        Wf_=wf()
        Pos_=ParticlePos()
        xvalues=Pos_.get_xvalues()
        #Wf_.Psiderivtest(0,0,0,xvalues)
        Ekin=Wf_.Ekinloc_finitediff(xvalues,1e-3)
        print("Ekin finite diff: ",Ekin)

  
        
h0=1e-5
#print(np.exp(-1*0.02384583*1j))
#print(np.exp(-1*0.02384583*1j)*np.exp(-1*0.02384583*1j))
#print((np.exp(-1*0.02384583*1j)-np.exp((1-h0)*0.02384583*1j))/(h0*np.exp(1*0.02384583*1j)))
test=TestMain()
test.run()