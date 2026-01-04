import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)

class wf:
    def __init__(self,deriv=False):
        #self.k0=k0
        #self.k1=k1
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

        self.NumParams=36
        nparams=np.loadtxt("NumParamsa2b2.txt")
        self.NumParamsa2=int(nparams[0])
        self.NumParamsb2=int(nparams[1])
        self.NumParamsx2=int(nparams[2])
        self.NumParamsy2=int(nparams[3])

        
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
        
        self.create_slaterPart()

        if not deriv:
          self.loadcoefs()

    def loadcoefs(self,components=2):
        self.NumParamsu=self.NumParams-self.NumParamsa2-self.NumParamsb2-self.NumParamsx2-self.NumParamsy2-1
        self.numCoefs=self.NumParamsu+3+self.NumParamsa2+3+self.NumParamsb2+3+self.NumParamsx2+3+self.NumParamsy2+3;
        Parameters=[]
        for ig in range(components):
            Parameters.append(np.loadtxt("coefs0.txt")[ig, :])
        #coefs=np.array([np.loadtxt("coefs.txt")])
        self.cutoff_radius=0.5*2.0*np.pi/self.deltaK #158.012 #352.1597518740  #125.66370614359172 #0 #set!!
        #print(self.cutoff_radius)
        self.coefs=[]
        for ig in range(components):
            coefs_=np.zeros(self.numCoefs)
            for i in range(self.NumParamsu):
                coefs_[i] = Parameters[ig][i];

            for i in range(self.NumParamsa2):
                coefs_[self.NumParamsu+3+i] = Parameters[ig][self.NumParamsu+i];

            for i in range(self.NumParamsb2):
                coefs_[self.NumParamsu+3+self.NumParamsa2+3+i] = Parameters[ig][self.NumParamsu+self.NumParamsa2+i];

            for i in range(self.NumParamsx2):
                coefs_[self.NumParamsu+3+self.NumParamsa2+3+self.NumParamsb2+3+ i] = Parameters[ig][self.NumParamsu+self.NumParamsa2+self.NumParamsb2+i];

            for i in range(self.NumParamsy2):
                coefs_[self.NumParamsu+3+self.NumParamsa2+3+self.NumParamsb2+3+ self.NumParamsx2+3+i] = Parameters[ig][self.NumParamsu+self.NumParamsa2+self.NumParamsb2+ self.NumParamsx2+i];      
            self.coefs.append(coefs_)

        self.etaVar=np.zeros(components)
        for ig in range(components):
            self.etaVar[ig]=Parameters[ig][self.NumParamsu+self.NumParamsa2+self.NumParamsb2+self.NumParamsx2+self.NumParamsy2];
        #print("etavar:",self.etaVar)
        self.numKnotsu=self.NumParamsu+3-2
        self.numKnotsa2=self.NumParamsa2+3-2
        self.numKnotsb2=self.NumParamsb2+3-2
        self.numKnotsx2=self.NumParamsx2+3-2
        self.numKnotsy2=self.NumParamsy2+3-2


        cutoff_factor=np.zeros(components)
        for ig in range(components):
            cutoff_factor[ig]=np.sqrt(self.etaVar[ig]);
    
        cutoff_new=np.zeros(components)
        for ig in range(components):
            cutoff_new[ig]=self.cutoff_radius*cutoff_factor[ig];

        self.DeltaRu       = self.cutoff_radius*1.0 / (self.numKnotsu - 1);
        self.DeltaRInvu    = 1.0 / self.DeltaRu;

        self.DeltaRa2=np.zeros(components)
        self.DeltaRb2=np.zeros(components)
        self.DeltaRInva2=np.zeros(components)
        self.DeltaRInvb2=np.zeros(components)

        self.DeltaRx2=np.zeros(components)
        self.DeltaRy2=np.zeros(components)
        self.DeltaRInvx2=np.zeros(components)
        self.DeltaRInvy2=np.zeros(components)
        for ig in range(components):
            self.DeltaRa2[ig]       = cutoff_new[ig]*1.0 / (self.numKnotsa2 - 1);
            self.DeltaRInva2[ig]    = 1.0 / self.DeltaRa2[ig];

            self.DeltaRb2[ig]       = cutoff_new[ig] *1.0/ (self.numKnotsb2 - 1);
            self.DeltaRInvb2[ig]    = 1.0 / self.DeltaRb2[ig];

            self.DeltaRx2[ig]       = self.cutoff_radius *1.0/ (self.numKnotsx2 - 1);
            self.DeltaRInvx2[ig]    = 1.0 / self.DeltaRx2[ig];

            self.DeltaRy2[ig]       = self.cutoff_radius *1.0/ (self.numKnotsy2 - 1);
            self.DeltaRInvy2[ig]    = 1.0 / self.DeltaRy2[ig];
        #print("Deltas: ",self.DeltaRu,self.DeltaRa2,self.DeltaRb2)
        #print("Deltas inv: ",self.DeltaRInvu,self.DeltaRInva2,self.DeltaRInvb2)

        self.F=np.zeros((self.ngroups,self.ngroups),dtype=int)
        self.F[0,0]=0
        self.F[0,1]=1
        self.F[1,0]=1
        self.F[1,1]=0



    def loadcoefs_forderiv(self,igderiv,ideriv,delta=1e-4,components=2):
        self.NumParamsu=self.NumParams-self.NumParamsa2-self.NumParamsb2-self.NumParamsx2-self.NumParamsy2-1
        self.numCoefs=self.NumParamsu+3+self.NumParamsa2+3+self.NumParamsb2+3+self.NumParamsx2+3+self.NumParamsy2+3;
        Parameters=[]
        for ig in range(components):
            c_=np.loadtxt("coefs0.txt")[ig, :]
            if ig==igderiv:
                c_[ideriv]=c_[ideriv]+delta
            Parameters.append(c_)

        #coefs=np.array([np.loadtxt("coefs.txt")])
        self.cutoff_radius=0.5*2.0*np.pi/self.deltaK #158.012 #352.1597518740  #125.66370614359172 #0 #set!!
        #print(self.cutoff_radius)
        self.coefs=[]
        for ig in range(components):
            coefs_=np.zeros(self.numCoefs)
            for i in range(self.NumParamsu):
                coefs_[i] = Parameters[ig][i];

            for i in range(self.NumParamsa2):
                coefs_[self.NumParamsu+3+i] = Parameters[ig][self.NumParamsu+i];

            for i in range(self.NumParamsb2):
                coefs_[self.NumParamsu+3+self.NumParamsa2+3+i] = Parameters[ig][self.NumParamsu+self.NumParamsa2+i];

            for i in range(self.NumParamsx2):
                coefs_[self.NumParamsu+3+self.NumParamsa2+3+self.NumParamsb2+3+ i] = Parameters[ig][self.NumParamsu+self.NumParamsa2+self.NumParamsb2+i];

            for i in range(self.NumParamsy2):
                coefs_[self.NumParamsu+3+self.NumParamsa2+3+self.NumParamsb2+3+ self.NumParamsx2+3+i] = Parameters[ig][self.NumParamsu+self.NumParamsa2+self.NumParamsb2+ self.NumParamsx2+i];      
            self.coefs.append(coefs_)

        self.etaVar=np.zeros(components)
        for ig in range(components):
            self.etaVar[ig]=Parameters[ig][self.NumParamsu+self.NumParamsa2+self.NumParamsb2+self.NumParamsx2+self.NumParamsy2];
        
        #print("etavar:",self.etaVar)
        self.numKnotsu=self.NumParamsu+3-2
        self.numKnotsa2=self.NumParamsa2+3-2
        self.numKnotsb2=self.NumParamsb2+3-2
        self.numKnotsx2=self.NumParamsx2+3-2
        self.numKnotsy2=self.NumParamsy2+3-2


        cutoff_factor=np.zeros(components)
        for ig in range(components):
            cutoff_factor[ig]=np.sqrt(self.etaVar[ig]);
    
        cutoff_new=np.zeros(components)
        for ig in range(components):
            cutoff_new[ig]=self.cutoff_radius*cutoff_factor[ig];

        self.DeltaRu       = self.cutoff_radius*1.0 / (self.numKnotsu - 1);
        self.DeltaRInvu    = 1.0 / self.DeltaRu;

        self.DeltaRa2=np.zeros(components)
        self.DeltaRb2=np.zeros(components)
        self.DeltaRInva2=np.zeros(components)
        self.DeltaRInvb2=np.zeros(components)

        self.DeltaRx2=np.zeros(components)
        self.DeltaRy2=np.zeros(components)
        self.DeltaRInvx2=np.zeros(components)
        self.DeltaRInvy2=np.zeros(components)
        for ig in range(components):
            self.DeltaRa2[ig]       = cutoff_new[ig]*1.0 / (self.numKnotsa2 - 1);
            self.DeltaRInva2[ig]    = 1.0 / self.DeltaRa2[ig];

            self.DeltaRb2[ig]       = cutoff_new[ig] *1.0/ (self.numKnotsb2 - 1);
            self.DeltaRInvb2[ig]    = 1.0 / self.DeltaRb2[ig];

            self.DeltaRx2[ig]       = self.cutoff_radius *1.0/ (self.numKnotsx2 - 1);
            self.DeltaRInvx2[ig]    = 1.0 / self.DeltaRx2[ig];

            self.DeltaRy2[ig]       = self.cutoff_radius *1.0/ (self.numKnotsy2 - 1);
            self.DeltaRInvy2[ig]    = 1.0 / self.DeltaRy2[ig];
        #print("Deltas: ",self.DeltaRu,self.DeltaRa2,self.DeltaRb2)
        #print("Deltas inv: ",self.DeltaRInvu,self.DeltaRInva2,self.DeltaRInvb2)

        self.F=np.zeros((self.ngroups,self.ngroups),dtype=int)
        self.F[0,0]=0
        self.F[0,1]=1
        self.F[1,0]=1
        self.F[1,1]=0
   

    def load_kgrid(self):
        self.Eta=np.loadtxt("EtaEtaVarmax.txt")[0]
        
        #mx=Eta**(-0.5*Tau)
        #my=Eta**(0.5*Tau);

        ks=np.loadtxt("kFmax.txt");
  
        Kstart=ks[0]

        deltak_linspace=ks[1]
        self.deltaK=deltak_linspace
        self.L=2.0*np.pi/self.deltaK
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
        #print("Kvalues: ",self.Kvalues)
        #print("Kvalues[0]: ",self.Kvalues[0])
        #print( "Kvalues[1]: ",self.Kvalues[1])

        #print("Ekin slater: ",Ekin_slater)

    def slaterPart(self,xvalues):
        #xvalues: [ngroups] (nspins[flavor],2)
        psi=1
        #print("Slater component")
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

            #print("ig: ",ig,"D:",D_,"\n")
            D_=1.0/np.sqrt(np.math.factorial(numspins))*D_ 
            #print("D_: ",D_)
            psi=psi*np.linalg.det(D_)
        #print("psi: ",psi)
        return psi

    def evaluateV(self,r,component):
      if r>self.cutoff_radius:
         return 0;
      else:
        r *= self.DeltaRInvu
        i  = int(r);
        #print(i)
        t = r - i;
        #print("component: ",component)        
        #print("i: ",i, self.NumParamsu, self.NumParamsu+3)

        d1      = self.coefs[component][i + 0] * (((self.A0 * t + self.A1) * t + self.A2) * t + self.A3);
        d2      = self.coefs[component][i + 1] * (((self.A4 * t + self.A5) * t + self.A6) * t + self.A7);
        d3      = self.coefs[component][i + 2] * (((self.A8 * t + self.A9) * t + self.A10) * t + self.A11);
        d4      = self.coefs[component][i + 3] * (((self.A12 * t + self.A13) * t + self.A14) * t + self.A15);
        d = (d1 + d2 + d3 + d4);
        return d

    def evaluateV2(self, r, x, y,numpar, tauvalue, component):
      #if(numpar>1.5):
      #  print("numpar: ",numpar,"evaluateV2, r: ",r, " x: ",x, " y: ",y," cutoff_radius: ",self.cutoff_radius)
      if r>self.cutoff_radius:
         return 0;
      else:
        shift_index=0;
        if (numpar==0):
           shift_index=self.NumParamsu+3;
        elif (numpar==1):
           shift_index=self.NumParamsu+3+self.NumParamsa2+3;
        elif (numpar==2):
           shift_index=self.NumParamsu+3+self.NumParamsa2+3+self.NumParamsb2+3;
        elif (numpar==3):
           shift_index=self.NumParamsu+3+self.NumParamsa2+3+self.NumParamsb2+3+self.NumParamsx2+3;

        if (tauvalue>1.5):
           if (numpar==0):
             shift_index=self.NumParamsu+3+self.NumParamsa2+3;
           elif (numpar==1):
             shift_index=self.NumParamsu+3;
           elif (numpar==2):
             shift_index=self.NumParamsu+3+self.NumParamsa2+3+self.NumParamsb2+3+self.NumParamsx2+3; 
           elif (numpar==3):
             shift_index=self.NumParamsu+3+self.NumParamsa2+3+self.NumParamsb2+3; 
  
        xsquared=x*x
        ysquared=y*y
        DeltaRinvGen=0

        if (numpar==0):
           r=np.sqrt(self.etaVar[component]*xsquared+1.0/self.etaVar[component]*ysquared);
           DeltaRinvGen= self.DeltaRInva2[component];
    
        elif (numpar==1):
           r=np.sqrt(1.0/self.etaVar[component]*xsquared+self.etaVar[component]*ysquared);
           DeltaRinvGen= self.DeltaRInvb2[component];

        elif (numpar==2):
           r=np.sqrt(xsquared);
           DeltaRinvGen= self.DeltaRInvx2[component]; 

        elif (numpar==3):
           r=np.sqrt(ysquared);
           DeltaRinvGen= self.DeltaRInvy2[component]; 
    
        #print(r)
        r *= DeltaRinvGen;
        #print("DeltaRinvGen",DeltaRinvGen,r)
        i       = int(r);

        #print("shift_index: ",shift_index, "i: ",i)

        #print("numpar: ",numpar)
        #print(self.coefs[component][shift_index + i + 0],self.coefs[component][shift_index + i + 1],  self.coefs[component][shift_index + i + 2], self.coefs[component][shift_index + i + 3])
        t = r - i
        d1      = self.coefs[component][shift_index + i + 0] * (((self.A0 * t + self.A1) * t + self.A2) * t + self.A3);
        d2      = self.coefs[component][shift_index + i + 1] * (((self.A4 * t + self.A5) * t + self.A6) * t + self.A7);
        d3      = self.coefs[component][shift_index + i + 2] * (((self.A8 * t + self.A9) * t + self.A10) * t + self.A11);
        d4      = self.coefs[component][shift_index + i + 3] * (((self.A12 * t + self.A13) * t + self.A14) * t + self.A15);
        d = (d1 + d2 + d3 + d4);
        return d

    def computeU(self,xvalues,onlyiat=False,igchange=0,iatchange=0):
        Uat=0
        #print("compute U:")
        for ig in range(self.ngroups):
            for iat in range(self.nspins[ig]):
                #print("ig: ",ig," iat: ",iat)
                for jat in range(iat):
                    #print("jat: ",jat)
                    x=(xvalues[ig][jat,0]-xvalues[ig][iat,0])*1.0/self.L
                    y=(xvalues[ig][jat,1]-xvalues[ig][iat,1])*1.0/self.L
                    x=self.L*(x-np.round(x))
                    y=self.L*(y-np.round(y))
                    r=np.sqrt((x)**2+(y)**2)
                    if (not onlyiat) or (ig==igchange and (iat==iatchange or jat==iatchange)):
                      #print(ig,iat,jat)
                      a2=self.evaluateV2(r,x,y,0,self.Tauvalues[ig]+self.Tauvalues[ig], self.F[ig,ig])

                      b2=self.evaluateV2(r,x,y,1, self.Tauvalues[ig]+self.Tauvalues[ig], self.F[ig,ig])

                      x2=self.evaluateV2(r,x,y,2, self.Tauvalues[ig]+self.Tauvalues[ig], self.F[ig,ig])

                      y2=self.evaluateV2(r,x,y,3, self.Tauvalues[ig]+self.Tauvalues[ig], self.F[ig,ig])
                      #print("b2: ",b2)
                      Uat+=self.evaluateV(r,self.F[ig,ig])+a2+b2+x2+y2


        for ig in range(self.ngroups):
            #print("ig: ",ig)
            for jg in range(ig):
                #print("jg: ",jg)
                for iat in range(self.nspins[ig]):
                    #print("iat: ",iat)
                    for jat in range(self.nspins[jg]):
                        #print("jat: ",jat)
                        #r=np.sqrt((np.dot(xvalues[jg][jat]-xvalues[ig][iat],xvalues[jg][jat]-xvalues[ig][iat])))

                        x=(xvalues[jg][jat,0]-xvalues[ig][iat,0])*1.0/self.L
                        #x=x-int(x)
                        y=(xvalues[jg][jat,1]-xvalues[ig][iat,1])*1.0/self.L
                        #y=y-int(y)
                        x=self.L*(x-np.round(x))
                        y=self.L*(y-np.round(y))
                        r=np.sqrt((x)**2+(y)**2)                    
                        if (not onlyiat) or ((ig==igchange and iat==iatchange) or (jg==igchange and jat==iatchange)):
                           #print(ig,jg,iat,jat)

                           a2=self.evaluateV2(r,x,y,0,self.Tauvalues[ig]+self.Tauvalues[jg], self.F[ig,jg])
                           b2=self.evaluateV2(r,x,y,1, self.Tauvalues[ig]+self.Tauvalues[jg], self.F[ig,jg])
                           x2=self.evaluateV2(r,x,y,2, self.Tauvalues[ig]+self.Tauvalues[jg], self.F[ig,jg])
                           y2=self.evaluateV2(r,x,y,3, self.Tauvalues[ig]+self.Tauvalues[jg], self.F[ig,jg])

                           #print("b2: ",b2)
                           Uat+=self.evaluateV(r,self.F[ig,jg])+a2+b2+x2+y2

        #print("Uat: ",Uat)
        return Uat


    def evaluateU(self,r,x,y,ig,jg):
        u=self.evaluateV(r,self.F[ig,jg])
        a2=self.evaluateV2(r,x,y,0, self.Tauvalues[ig]+self.Tauvalues[jg], self.F[ig,jg])
        b2=self.evaluateV2(r,x,y,1, self.Tauvalues[ig]+self.Tauvalues[jg], self.F[ig,jg])
        x2=self.evaluateV2(r,x,y,2, self.Tauvalues[ig]+self.Tauvalues[jg], self.F[ig,jg])
        y2=self.evaluateV2(r,x,y,3, self.Tauvalues[ig]+self.Tauvalues[jg], self.F[ig,jg])
        return u+a2+b2+x2+y2

    def JastrowPart(self,xvalues,onlyiat=False,igchange=0,iatchange=0):
        return np.exp(-self.computeU(xvalues,onlyiat,igchange,iatchange))


    def Psi(self,xvalues,onlyiat=False,igchange=0,iatchange=0):
        return self.JastrowPart(xvalues,onlyiat,igchange,iatchange)*self.slaterPart(xvalues)
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
        print("Psi: ",psi)
        for ig in range(self.ngroups):
            for iat in range(self.nspins[ig]):
                for dim in range(2):
                    signtau=1
                    if dim==0:
                        signtau=1
                    else:
                        signtau=-1
                    Ekin+=(-1.0/(2.0*self.mstar)*self.Eta**(0.5*self.Tauvalues[ig]*signtau)*self.Psideriv2(dim,ig,iat,xvalues,h))
        #print("psi: ",psi)
        return Ekin/psi 

    

    #def evaluateU(self,r,x,y):
    #    return self.evaluateV(r)+self.evaluateV2(r,x,y,0)+self.evaluateV2(r,x,y,1)

    def plotJastrow2(self,ig,jg):
        #rvalues = np.linspace(1.0/100.0,self.cutoff_radius-self.cutoff_radius/100.0,100)
        x=np.linspace(-self.L,self.L,200)
        y=np.zeros(200)+0 #40
        #y=np.linspace(-self.cutoff_radius+self.cutoff_radius/100.0,self.cutoff_radius-self.cutoff_radius/100.0,200)
        d=np.zeros_like(x)
        for i in range(200):
            xvalue=x[i]*1.0/self.L
            xvalue=self.L*(xvalue-np.round(xvalue))
            yvalue=y[i]*1.0/self.L
            yvalue=self.L*(yvalue-np.round(yvalue))
            print("xvalue: ",xvalue,"yvalue: ",yvalue)
            r=np.sqrt((xvalue)**2+(yvalue)**2)  
            #print("x,y: ",xvalue,yvalue)
            d[i] = np.exp(-self.evaluateU(r,xvalue,yvalue,ig,jg))
        
        plt.figure()
        #plt.title("uu")
        plt.plot(x,d)
        plt.vlines(-self.L/2,0,1)
        plt.vlines(self.L/2,0,1)
        plt.hlines(1.0,-self.L,self.L)
        #plt.contourf(X,Y,d)
        plt.xlabel("x")
        plt.show()


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


    def get_xvalues0(self):
        x_up=np.zeros((2,2))
        x_up[0,:]=np.array([ 9.6653953744e+01,  5.7396298431e+01 ])
        x_up[1,:]=np.array([1.5934273130e+01,  1.1098846504e+02 ])
        x_dn=np.zeros((2,2))
        x_dn[0,:]=np.array([ 5.6996449362e+01,  2.2977002131e+01])
        x_dn[1,:]=np.array([ 6.0188181508e+01,  6.4314578463e+01])
        xvalues1=[x_up,x_dn]
        return xvalues1

    def get_xvalues0step(self):
        x_up=np.zeros((2,2))
        x_up[0,:]=np.array([ 9.1929470428e+01,  5.2584637804e+01 ])
        x_up[1,:]=np.array([1.5934273130e+01,  1.1098846504e+02 ])
        x_dn=np.zeros((2,2))
        x_dn[0,:]=np.array([ 5.6996449362e+01,  2.2977002131e+01])
        x_dn[1,:]=np.array([ 6.0188181508e+01,  6.4314578463e+01])
        xvalues1=[x_up,x_dn]
        return xvalues1  

    def get_xvalues1(self):
        x_up=np.zeros((2,2))
        x_up[0,:]=np.array([9.1929470428e+01,  5.2584637804e+01 ])
        x_up[1,:]=np.array([1.5934273130e+01,  1.1098846504e+02 ])
        x_dn=np.zeros((2,2))
        x_dn[0,:]=np.array([ 5.6996449362e+01,  2.2977002131e+01])
        x_dn[1,:]=np.array([ 6.0188181508e+01,  6.4314578463e+01])
        xvalues1=[x_up,x_dn]
        return xvalues1

    def get_xvalues1step(self):
        x_up=np.zeros((2,2))
        x_up[0,:]=np.array([ 9.1929470428e+01,  5.2584637804e+01 ])
        x_up[1,:]=np.array([1.2102203592e+01,  1.1565514383e+02])
        x_dn=np.zeros((2,2))
        x_dn[0,:]=np.array([ 5.6996449362e+01,  2.2977002131e+01])
        x_dn[1,:]=np.array([6.0188181508e+01,  6.4314578463e+01 ])
        xvalues1=[x_up,x_dn]
        return xvalues1

    def get_xvalues2(self):
        x_up=np.zeros((2,2))
        x_up[0,:]=np.array([9.1929470428e+01,  5.2584637804e+01  ])
        x_up[1,:]=np.array([1.2102203592e+01,  1.1565514383e+02])
        x_dn=np.zeros((2,2))
        x_dn[0,:]=np.array([ 5.6996449362e+01,  2.2977002131e+01])
        x_dn[1,:]=np.array([6.0188181508e+01,  6.4314578463e+01 ])
        xvalues1=[x_up,x_dn]
        return xvalues1
    
    def get_xvalues2step(self):
        x_up=np.zeros((2,2))
        x_up[0,:]=np.array([ 9.1929470428e+01,  5.2584637804e+01  ])
        x_up[1,:]=np.array([1.2102203592e+01,  1.1565514383e+02])
        x_dn=np.zeros((2,2))
        x_dn[0,:]=np.array([6.0653254941e+01,  2.7191660190e+01])
        x_dn[1,:]=np.array([6.0188181508e+01,  6.4314578463e+01 ])
        xvalues1=[x_up,x_dn]
        return xvalues1  

    def get_xvalues3(self):
        x_up=np.zeros((2,2))
        x_up[0,:]=np.array([9.1929470428e+01,  5.2584637804e+01  ])
        x_up[1,:]=np.array([1.2102203592e+01,  1.1565514383e+02])
        x_dn=np.zeros((2,2))
        x_dn[0,:]=np.array([5.6996449362e+01,  2.2977002131e+01])
        x_dn[1,:]=np.array([6.0188181508e+01,  6.4314578463e+01 ])
        xvalues1=[x_up,x_dn]
        return xvalues1 

    def get_xvalues3step(self):
        x_up=np.zeros((2,2))
        x_up[0,:]=np.array([9.1929470428e+01,  5.2584637804e+01 ])
        x_up[1,:]=np.array([1.2102203592e+01,  1.1565514383e+02])
        x_dn=np.zeros((2,2))
        x_dn[0,:]=np.array([5.6996449362e+01,  2.2977002131e+01])
        x_dn[1,:]=np.array([ 5.6857588680e+01,  6.0802748459e+01])
        xvalues1=[x_up,x_dn]
        return xvalues1  

    def get_xvaluesekin1(self):
        x_up=np.zeros((2,2))
        x_up[0,:]=np.array([8.4111500783e+01,  4.2907982462e+01])
        x_up[1,:]=np.array([1.2102203592e+01,  1.1565514383e+02])
        x_dn=np.zeros((2,2))
        x_dn[0,:]=np.array([ 5.1923646515e+01,  1.9556433679e+01 ])
        x_dn[1,:]=np.array([6.0188181508e+01,  6.4314578463e+01 ])
        xvalues1=[x_up,x_dn]
        return xvalues1  

    def get_xvaluesekin2(self):
        x_up=np.zeros((2,2))
        x_up[0,:]=np.array([8.3719405123e+01,  4.8917700620e+01 ])
        x_up[1,:]=np.array([1.2102203592e+01,  1.1565514383e+02])
        x_dn=np.zeros((2,2))
        x_dn[0,:]=np.array([ 5.1923646515e+01,  1.9556433679e+01])
        x_dn[1,:]=np.array([ 6.2569337500e+01,  6.6172466467e+01])
        xvalues1=[x_up,x_dn]
        return xvalues1  


class TestMain:
    def __init__(self):
        self.initialized=1

    def get_Ekinfinitediff(self):
        ks=np.loadtxt("kFmax.txt");
  
        Kstart=ks[0]

        deltak_linspace=ks[1]
        self.deltaK=deltak_linspace
        self.L=2.0*np.pi/self.deltaK
        print(self.L)

        Wf_=wf()
        Pos_=ParticlePos()
        xvaluesekin1=Pos_.get_xvaluesekin1()
        psi=Wf_.Psi(xvaluesekin1)
        print(psi)
        Wf_.plotJastrow2(0,0)

        c=np.zeros((2,2))
        c[0,0]=self.L
        xvaluesekin1[0]=xvaluesekin1[0]+c
        print(xvaluesekin1)
        psi=Wf_.Psi(xvaluesekin1)
        print(psi)
        ##
        """
        xvalues0=Pos_.get_xvalues0()
        psi0=Wf_.Psi(xvalues0,False,0,0)

        xvalues0step=Pos_.get_xvalues0step()
        psi0step=Wf_.Psi(xvalues0step,False,0,0)
        print("xvalues diff (0,0): ","\n",xvalues0step[0]-xvalues0[0],"\n", xvalues0step[1]-xvalues0[1])
        print("ratio 0: ",psi0step/psi0  ,np.conjugate(psi0step)*psi0step/(np.conjugate(psi0)*psi0))
        print("\n","\n")


        ##
        xvalues1=Pos_.get_xvalues1()
        psi1=Wf_.Psi(xvalues1,False,0,1)

        xvalues1step=Pos_.get_xvalues1step()
        psi1step=Wf_.Psi(xvalues1step,False,0,1)
        print("xvalues diff (0,1): ","\n",xvalues1step[0]-xvalues1[0], "\n",xvalues1step[1]-xvalues1[1])
        print("ratio 1: ",psi1step/psi1  ,np.conjugate(psi1step)*psi1step/(np.conjugate(psi1)*psi1))
        print("\n","\n")

        ##
        xvalues2=Pos_.get_xvalues2()
        psi2=Wf_.Psi(xvalues2,False,1,0)

        xvalues2step=Pos_.get_xvalues2step()
        psi2step=Wf_.Psi(xvalues2step,False,1,0)
        print("xvalues diff (1,0): ","\n",xvalues2step[0]-xvalues2[0],"\n", xvalues2step[1]-xvalues2[1])

        print("ratio 2: ",psi2step/psi2  ,np.conjugate(psi2step)*psi2step/(np.conjugate(psi2)*psi2))
        print("\n","\n")

        ##
        xvalues3=Pos_.get_xvalues3()
        psi3=Wf_.Psi(xvalues3,True,1,1)

        xvalues3step=Pos_.get_xvalues3step()
        psi3step=Wf_.Psi(xvalues3step,True,1,1)
        print("xvalues diff (1,1): ","\n",xvalues3step[0]-xvalues3[0],"\n", xvalues3step[1]-xvalues3[1])

        print("ratio 3: ",psi3step/psi3  ,np.conjugate(psi3step)*psi3step/(np.conjugate(psi3)*psi3))
        print("\n","\n")


        
        #print(np.exp(1j*np.dot(a,b)))


        Ekin=Wf_.Ekinloc_finitediff(xvalues3,1e-4)
        print("Ekin finite diff: ",Ekin)
        #print(np.exp(1j*np.dot(a,b)))
        xvaluesekin1=Pos_.get_xvaluesekin1()
        Ekin=Wf_.Ekinloc_finitediff(xvaluesekin1,1e-4)
        print("Ekin finite diff: ",Ekin)

        xvaluesekin2=Pos_.get_xvaluesekin2()
        Ekin=Wf_.Ekinloc_finitediff(xvaluesekin2,1e-4)
        print("Ekin finite diff: ",Ekin)
        """

    def get_Psideriv1(self,igderiv,ideriv,delta=1e-4):
        Wf1_=wf()
        Wf2_=wf(True)
        Wf2_.loadcoefs_forderiv(igderiv,ideriv,delta)
        Pos_=ParticlePos()
        xvalues=Pos_.get_xvaluesekin1()
        psi1=Wf1_.Psi(xvalues)

        psi2=Wf2_.Psi(xvalues)

        return ((psi2-psi1)/delta)/psi1

    def get_Psideriv2(self,igderiv,ideriv,delta=1e-4):
        Wf1_=wf()
        Wf2_=wf(True)
        Wf2_.loadcoefs_forderiv(igderiv,ideriv,delta)
        Pos_=ParticlePos()
        xvalues=Pos_.get_xvaluesekin2()
        psi1=Wf1_.Psi(xvalues)

        psi2=Wf2_.Psi(xvalues)

        return ((psi2-psi1)/delta)/psi1

    
    def run(self):
        self.get_Ekinfinitediff()
        it=0
        """
        for ig in range(2):
            for i in range(36):
                print(it, self.get_Psideriv1(ig,i,1e-4))
                it+=1

        it=0
        for ig in range(2):
            for i in range(36):
                print(it, self.get_Psideriv2(ig,i,1e-4))
                it+=1     
        """ 
    

    """
    def run(self):
        Wf_=wf()
        Pos_=ParticlePos()
        xvalues1=Pos_.get_xvalues1()
        psi1=Wf_.Psi(xvalues1)

        xvalues2=Pos_.get_xvalues2()
        psi2=Wf_.Psi(xvalues2)

        print("ratio: ",psi2/psi1  ,np.conjugate(psi2)*psi2/(np.conjugate(psi1)*psi1))
        Ekin=Wf_.Ekinloc_finitediff(xvalues2,1e-4)
        print("Ekin finite diff: ",Ekin)

    """
        
h0=1e-5
#print(np.exp(-1*0.02384583*1j))
#print(np.exp(-1*0.02384583*1j)*np.exp(-1*0.02384583*1j))
#print((np.exp(-1*0.02384583*1j)-np.exp((1-h0)*0.02384583*1j))/(h0*np.exp(1*0.02384583*1j)))
test=TestMain()
test.run()