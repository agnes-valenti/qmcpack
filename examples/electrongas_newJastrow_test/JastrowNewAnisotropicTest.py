import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)

class wf:
    def __init__(self,calcderiv=False):
        #self.k0=k0
        #self.k1=k1
        self.calcderiv=calcderiv
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

        #self.NumParams=36
        nparams=np.loadtxt("NumParamsa2b2.txt")
        self.NumParamsu=int(nparams[0])
        self.Nv=int(nparams[1])
        self.NumParams=self.NumParamsu+self.Nv

        
        hbar=1


        n0=0.1/312.097  #10^11 cm^-2
        n1=n0*1.0 #200.0
        a = 0.56605e-9;        # AlAs lattice spacing in nanometers
        d = 100*10**(-9)/a; # Distance to gate in units of unit cell size
        self.d2_=2*d
        epsilon_d = 10; # AlAs dielectric constant (GaAs? AlGaAs?)

        qe = 1.60217662e-19; # electron charge
        ke = 8.99e9;         # Coulomb constant
        hbar=1.0545718*1e-34;  #hbar (in SI units)
        a = 0.56605e-9;        # AlAs lattice spacing in nanometers

        self.e_squared = 1.0*ke*qe/a*1e3/epsilon_d; # e^2 in units of meV*a
        me=9.1093837015*1e-31  #kg - umrechnen?? 1/a? meV?
        self.mstar=0.457*me*qe/(hbar**2*1e3)*a*a  #mstar in units of (1/(meV*a^2)) (from hbar) -> 1/mstar*k^2->meV  (a needed here?)
        
        self.Eta=np.loadtxt("EtaEtaVarmax.txt")[0]
        m1=self.mstar*(self.Eta**(0.5))
        m2=self.mstar*(self.Eta**(-0.5))
        self.m_effective=(m1+m2)/2.0 #2.0/(1.0/m1+1.0/m2)
        self.NLx=18
        self.NLy=18

        self.load_U()
        self.create_slaterPart()

        #if not deriv:
        self.loadcoefs()
        #else:
        #  self.loadcoefs_forderiv(ig,igderiv,delta)

        self.set_cusp_parallel()
       

    def set_cusp_parallel(self):
        coefs_=self.coefs[0]
        coefs_[0]=coefs_[2]+2.0*self.DeltaRu*self.e_squared*self.cuspvalue*self.m_effective/3.0
        #print("cusp parallel: ",2.0*self.DeltaRu*self.e_squared*self.cuspvalue*self.m_effective/3.0)
        #print("cusp antiparallel: ",2.0*self.DeltaRu*self.e_squared*self.cuspvalue*self.m_effective/1.0 )
        self.coefs[0]=coefs_

    def set_cusp_antiparallel(self):
        coefs_=self.coefs[0]
        coefs_[0]=coefs_[2]+2.0*self.DeltaRu*self.e_squared*self.cuspvalue*self.m_effective/1.0
        #print("cusp antiparallel: ",2.0*self.DeltaRu*self.e_squared*self.cuspvalue*self.m_effective/1.0 )
        self.coefs[0]=coefs_


    def loadcoefs(self,components=2):
        #self.NumParamsu=self.NumParams-self.NumParamsa2-self.NumParamsb2-self.NumParamsx2-self.NumParamsy2-1
        self.numCoefs=self.NumParamsu+4+self.Nv*3;
        Parameters=[]
        for ig in range(components):
            c_=np.loadtxt("coefs0.txt")[ig, :]
            #if ig==igderiv:
            #    c_[ideriv]=c_[ideriv]+delta
            Parameters.append(c_)
            #Parameters.append(np.loadtxt("coefs0.txt")[ig, :])
        
        self.cutoff_radius=0.5*2.0*np.pi/self.deltaK #158.012 #352.1597518740  #125.66370614359172 #0 #set!!
        #print(self.cutoff_radius)
        self.coefs=[]
        for ig in range(components):
            coefs_=np.zeros(self.numCoefs)
            for i in range(self.NumParamsu):
                coefs_[i+1] = Parameters[ig][i];

            for i in range(self.Nv):
                coefs_[self.NumParamsu+4+i] = Parameters[ig][self.NumParamsu+i];

            for i in range(self.Nv):
                coefs_[self.NumParamsu+4+self.Nv+i] = Parameters[ig][self.NumParamsu+self.Nv+i];

            for i in range(self.Nv):
                coefs_[self.NumParamsu+4+self.Nv+self.Nv+ i] = Parameters[ig][self.NumParamsu+self.Nv+self.Nv+i];
            self.coefs.append(coefs_)

            

        
        #print("etavar:",self.etaVar)
        self.numKnotsu=self.NumParamsu+4-2
     


       
        self.DeltaRu       = self.cutoff_radius*1.0 / (self.numKnotsu - 1);
        self.DeltaRInvu    = 1.0 / self.DeltaRu;

        
        #print("Deltas: ",self.DeltaRu,self.DeltaRa2,self.DeltaRb2)
        #print("Deltas inv: ",self.DeltaRInvu,self.DeltaRInva2,self.DeltaRInvb2)

        self.F=np.zeros((self.ngroups,self.ngroups),dtype=int)
        self.F[0,0]=0
        self.F[0,1]=1
        self.F[1,0]=1
        self.F[1,1]=0


    def changecoefs_forderiv(self,igderiv,ideriv,delta=1e-4,components=2):
        self.numCoefs=self.NumParamsu+4+self.Nv*3;
        Parameters=[]
        for ig in range(components):
            c_=np.loadtxt("coefs0.txt")[ig, :]
            if ig==igderiv:
                c_[ideriv]=c_[ideriv]+delta
            Parameters.append(c_)
            #Parameters.append(np.loadtxt("coefs0.txt")[ig, :])
        
        self.cutoff_radius=0.5*2.0*np.pi/self.deltaK #158.012 #352.1597518740  #125.66370614359172 #0 #set!!
        #print(self.cutoff_radius)
        self.coefs=[]
        for ig in range(components):
            coefs_=np.zeros(self.numCoefs)
            for i in range(self.NumParamsu):
                coefs_[i+1] = Parameters[ig][i];

            for i in range(self.Nv):
                coefs_[self.NumParamsu+4+i] = Parameters[ig][self.NumParamsu+i];

            for i in range(self.Nv):
                coefs_[self.NumParamsu+4+self.Nv+i] = Parameters[ig][self.NumParamsu+self.Nv+i];

            for i in range(self.Nv):
                coefs_[self.NumParamsu+4+self.Nv+self.Nv+ i] = Parameters[ig][self.NumParamsu+self.Nv+self.Nv+i];
            self.coefs.append(coefs_)



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


    def load_U(self):
       uscr1=np.loadtxt("Uscreenedto1.txt");
       uscr2=np.loadtxt("Uscreened1to5001.txt");


       self.Nvalues1=10000;
       self.Nvalues2=100000;

       self.rvalues=np.zeros(self.Nvalues1+1+self.Nvalues2+1);
       self.Uvalues=np.zeros(self.Nvalues1+1+self.Nvalues2+1);

  
       self.rvalues[:self.Nvalues1+1]=uscr1[0,:];
       self.Uvalues[:self.Nvalues1+1]=uscr1[1,:]
  
       self.rvalues[self.Nvalues1+1:]=uscr2[0,:];
       self.Uvalues[self.Nvalues1+1:]=uscr2[1,:]


 
       self.Uvalues=self.Uvalues*4.0/self.d2_;

       self.cuspvalue=self.Uvalues[0]*self.rvalues[0]

  
    def testU(self):
        print("rvalues[0]: ",self.rvalues[0]," rvalues[10]: ",self.rvalues[10]," rvalues[2000]: ",self.rvalues[20000])
        print("Uvalues[0]: ",self.Uvalues[0]," Uvalues[10]: ",self.Uvalues[10]," Uvalues[2000]: ",self.Uvalues[20000])
        print(self.get_linear_interpolated_U(1e-10),self.get_linear_interpolated_U(1e-4))
        print(self.get_linear_interpolated_U(1e-2),self.get_linear_interpolated_U(0.5))
        print(self.get_linear_interpolated_U(10),self.get_linear_interpolated_U(100),self.get_linear_interpolated_U(1000))
        print("cusp: ",self.cuspvalue,self.e_squared)


    def get_index(self,rvalue):
       if (rvalue<=1):
          return int((np.log(rvalue)+6.0)*self.Nvalues1/6.0);
       else:
          return (min(int((rvalue-1)*self.Nvalues2/5000-1+(self.Nvalues1+1)),self.Nvalues1+self.Nvalues2));
  
    def get_linear_interpolated_U(self,rvalue):
       return_U=0
       if (rvalue<self.rvalues[1]):
          if (rvalue<1e-30):
             rvalue+=1e-30;
     
          return_U=self.Uvalues[0]*self.rvalues[0]/rvalue;
  
       else:
           ind1=self.get_index(rvalue);
           diff1=rvalue-self.rvalues[ind1];
           return_U=(self.Uvalues[ind1+1]-self.Uvalues[ind1])/(self.rvalues[ind1+1]-self.rvalues[ind1])*diff1+self.Uvalues[ind1];
  
       return return_U;



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
            psi1=np.linalg.det(D_)
            #print("psi1:",psi1,numspins, 1.0/np.sqrt(np.math.factorial(numspins))*psi1)
            
            #print("psi before: ",psi)
            psi=psi*1.0/np.sqrt(np.math.factorial(numspins))*np.linalg.det(D_)
        #print("psi slater: ",psi)
        return psi

    def slaterPart_2particles_relativecoords(self,r12,rp12):
        k0_=self.Kvalues[0][0]
        k1_=self.Kvalues[0][1]
        x0_=0.5*r12+rp12
        x1_=-0.5*r12+rp12
        #print("k0_: ",k0_, "k1_: ",k1_,"x0_: ",x0_,"x1_:",x1_)
        #print(x0_,x1_,k0_,k1_,np.exp(np.dot(k0_,x0_)*1j))
        #print(np.exp(np.dot(k0_,x0_)*1j),np.exp(np.dot(k1_,x1_)*1j),np.exp(np.dot(k1_,x0_)*1j),np.exp(np.dot(k0_,x1_)*1j))
        #print(np.exp(np.dot(k0_,x0_)*1j)*np.exp(np.dot(k1_,x1_)*1j))
        #print(np.exp(np.dot(k1_,x0_)*1j)*np.exp(np.dot(k0_,x1_)*1j))
        #print(np.exp(np.dot(k0_,x0_)*1j)*np.exp(np.dot(k1_,x1_)*1j)- np.exp(np.dot(k1_,x0_)*1j)*np.exp(np.dot(k0_,x1_)*1j))
        psi=1.0/np.sqrt(2)*(np.exp(np.dot(k0_,x0_)*1j)*np.exp(np.dot(k1_,x1_)*1j)-np.exp(np.dot(k1_,x0_)*1j)*np.exp(np.dot(k0_,x1_)*1j))
        #print("psi: ",psi)
        return psi

    def slaterPart_2particles_relativecoords_antiparallel(self,r12,rp12):
        k0_=self.Kvalues[0][0]
        k1_=self.Kvalues[0][1]
        x0_=0.5*r12+rp12
        x1_=-0.5*r12+rp12
        #print(x0_,x1_,k0_,k1_,np.exp(np.dot(k0_,x0_)*1j))
        return np.exp(np.dot(k0_,x0_)*1j)*np.exp(np.dot(k1_,x1_)*1j) #1.0/np.sqrt(2)*(np.exp(np.dot(k0_,x0_)*1j)*np.exp(np.dot(k1_,x1_)*1j)-np.exp(np.dot(k1_,x0_)*1j)*np.exp(np.dot(k0_,x1_)*1j))


    def evaluateV(self,r,component):
      #print("r: ",r,"cutoff_radius: ",self.cutoff_radius)
      if r>self.cutoff_radius:
         return 0;
      else:
        #print("r: ",r," component: ",component)
        r *= self.DeltaRInvu
        i  = int(r);
        #print(i)
        t = r - i;
        #print("component: ",component)        
        #print("i: ",i, self.NumParamsu, self.NumParamsu+3)
        
        #print("coefs[0]:",self.coefs[component][i + 0],"coefs[1]:",self.coefs[component][i + 1],"coefs[2]:",self.coefs[component][i + 2],"coefs[3]:",self.coefs[component][i + 3])
        d1      = self.coefs[component][i + 0] * (((self.A0 * t + self.A1) * t + self.A2) * t + self.A3);
        d2      = self.coefs[component][i + 1] * (((self.A4 * t + self.A5) * t + self.A6) * t + self.A7);
        d3      = self.coefs[component][i + 2] * (((self.A8 * t + self.A9) * t + self.A10) * t + self.A11);
        d4      = self.coefs[component][i + 3] * (((self.A12 * t + self.A13) * t + self.A14) * t + self.A15);
        d = (d1 + d2 + d3 + d4);
        return d

    def evaluateV2(self, r, x, y,numpar, tauvalue, component):
      #if(numpar>1.5):
      #  print("numpar: ",numpar,"evaluateV2, r: ",r, " x: ",x, " y: ",y," cutoff_radius: ",self.cutoff_radius)
      if 1==0:
         return 0;
      else:
        shift_index_c=self.NumParamsu+4;
        shift_index_alpha=self.NumParamsu+4+self.Nv;
        shift_index_beta=self.NumParamsu+4+self.Nv+self.Nv;

        if (tauvalue>1.5):
            shift_index_alpha=self.NumParamsu+4+self.Nv+self.Nv;
            shift_index_beta=self.NumParamsu+4+self.Nv;
  
        xsquared=x*x
        ysquared=y*y
        fx=np.sqrt(xsquared)*(1-(np.sqrt(xsquared)/(self.L/2.0))**3/4.0)
        fy=np.sqrt(ysquared)*(1-(np.sqrt(ysquared)/(self.L/2.0))**3/4.0)

        #print("fx, fy: ",fx,fy,self.L)
        vsum=0
        for i in range(self.Nv):
            vsum+=(self.L/2.0)**(-(i+1))*self.coefs[component][shift_index_c+i]*(self.coefs[component][shift_index_alpha+i]*fx*fx+self.coefs[component][shift_index_beta+i]*fy*fy)**((i+1)/2.0)
            #if self.calcderiv:
            #   print(i, self.coefs[component][shift_index_c+i], self.coefs[component][shift_index_alpha+i], self.coefs[component][shift_index_beta+i])
            #print(self.coefs[component][shift_index_c+i]*(self.coefs[component][shift_index_alpha+i]*fx*fx+self.coefs[component][shift_index_beta+i]*fy*fy)**((i+1)/2.0))
        #print("shift_index: ",shift_index, "i: ",i)

        #print("x: ",x," y: ",y," vsum: ",vsum)
        return vsum

    def computeU(self,xvalues,onlyiat=False,igchange=0,iatchange=0):
        Uat=0
        #print("compute U:")
        """
        for i in range(4):
            for j in range(i):
                ig=int(i/2)
                iat=i-ig*2
                jg=int(j/2)
                jat=j-jg*2
                x=(xvalues[jg][jat,0]-xvalues[ig][iat,0])*1.0/self.L
                        
                y=(xvalues[jg][jat,1]-xvalues[ig][iat,1])*1.0/self.L
                x=self.L*(x-np.round(x))
                y=self.L*(y-np.round(y))
                r=np.sqrt((x)**2+(y)**2)  
                print("i,j: ",i,j)
                print(ig,iat)     
                print(jg,jat)             
                u=self.evaluateV(r,self.F[ig,jg])
                v2=self.evaluateV2(r,x,y,0,self.Tauvalues[ig]+self.Tauvalues[jg], self.F[ig,jg])
                           
                Uat+=u+v2 #+a2+b2+x2+y2
        """

        
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
                      #print("r: ",r)
                      u=self.evaluateV(r,self.F[ig,ig])
                      v2=self.evaluateV2(r,x,y,0,self.Tauvalues[ig]+self.Tauvalues[ig], self.F[ig,ig])

                      
                      #print("hi ")
                      Uat+=u+v2 #+a2+b2+x2+y2


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
                           #print("r: ",r)
                           u=self.evaluateV(r,self.F[ig,jg])
                           v2=self.evaluateV2(r,x,y,0,self.Tauvalues[ig]+self.Tauvalues[jg], self.F[ig,jg])
                           
                           #print("b2: ",b2)
                           Uat+=u+v2 #+a2+b2+x2+y2
        
        #print("Uat: ",Uat)
        return Uat


    def evaluateU(self,r,x,y,ig,jg):
        u=self.evaluateV(r,self.F[ig,jg])
        v2=self.evaluateV2(r,x,y,0, self.Tauvalues[ig]+self.Tauvalues[jg], self.F[ig,jg])
        return u +v2

    def JastrowPart(self,xvalues,onlyiat=False,igchange=0,iatchange=0):
        d=self.computeU(xvalues,onlyiat,igchange,iatchange)
        #print("d: ",d)
        return np.exp(-d)

 

    def Psi(self,xvalues,onlyiat=False,igchange=0,iatchange=0):
        return self.slaterPart(xvalues)*self.JastrowPart(xvalues,onlyiat,igchange,iatchange) #*self.slaterPart(xvalues)
  
    def PsiJastrow(self,xvalues,onlyiat=False,igchange=0,iatchange=0):
        return self.JastrowPart(xvalues,onlyiat,igchange,iatchange)
   
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
        #print("Psi: ",psi)
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

        U=0
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
                     

                    r_periodic=0;
                    esum=0
                    for itx in range(-self.NLx, self.NLx+1):
                       for ity in range(-self.NLy, self.NLy+1):
                           #print("itx: ",itx," ity: ",ity)
                           r_periodic=np.sqrt((x+itx*self.L) * (x+itx*self.L) + (y+ity*self.L) * (y+ity*self.L));
                           esum+= (self.get_linear_interpolated_U(r_periodic));
      
                    U+=esum*self.e_squared  # self.get_linear_interpolated_U(r)*self.e_squared  #self.e_squared/r


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
                        r_periodic=0;
                        esum=0
                        for itx in range(-self.NLx, self.NLx+1):
                           for ity in range(-self.NLy, self.NLy+1):
                              #print("itx: ",itx," ity: ",ity)
                              r_periodic=np.sqrt((x+itx*self.L) * (x+itx*self.L) + (y+ity*self.L) * (y+ity*self.L));
                              esum+= (self.get_linear_interpolated_U(r_periodic));
                        U+=esum*self.e_squared  #self.get_linear_interpolated_U(r)*self.e_squared  #  self.e_squared/r
        print("Ekin: ",Ekin/psi, "U: ",U)
        return Ekin/psi +U

    def Eloc_finitediff_2particles_relativecoords(self,r12,rp12,h=1e-5):
        Ekin=0
        psi=self.Psi_2particles_relativecoords(r12,rp12)
        #print("Psi: ",psi)
        Ekin=-1.0/(2.0*self.mstar)*self.Psideriv2_2particles_relativecoords(0,r12,rp12,h)-1.0/(2.0*self.mstar)*self.Psideriv2_2particles_relativecoords(1,r12,rp12,h)
        #print("psi: ",psi)
        return 0.5*self.get_linear_interpolated_U(np.sqrt(np.dot(r12,r12)))*self.e_squared +Ekin/psi  #  0.5*self.e_squared/np.sqrt(np.dot(r12,r12)) +Ekin/psi

    

    #def evaluateU(self,r,x,y):
    #    return self.evaluateV(r)+self.evaluateV2(r,x,y,0)+self.evaluateV2(r,x,y,1)

    def plotJastrow2(self,ig,jg):
        #rvalues = np.linspace(1.0/100.0,self.cutoff_radius-self.cutoff_radius/100.0,100)
        N=800
        x=np.linspace(-self.L*0.9,self.L*0.9,N)
        y=np.zeros(N)+0
        #y=np.linspace(-self.cutoff_radius+self.cutoff_radius/100.0,self.cutoff_radius-self.cutoff_radius/100.0,200)
        d=np.zeros_like(x)
        for i in range(N):
            xvalue=x[i]*1.0/self.L
            xvalue=self.L*(xvalue-np.round(xvalue))
            yvalue=y[i]*1.0/self.L
            yvalue=self.L*(yvalue-np.round(yvalue))
            #print("xvalue: ",xvalue,"yvalue: ",yvalue)
            r=np.sqrt((xvalue)**2+(yvalue)**2)  
            #print("x,y: ",xvalue,yvalue)
            #print("evaluate U: ",self.evaluateU(r,xvalue,yvalue,ig,jg))
            d[i] = np.exp(-self.evaluateU(r,xvalue,yvalue,ig,jg))
            #print("d: ",d[i])
        
        plt.figure()
        #plt.title("uu")
        plt.plot(x,d)
        plt.vlines(-self.L/2,0,np.max(d),color='grey')
        plt.vlines(self.L/2,0,np.max(d),color='grey')
        plt.hlines(1.0,-self.L,self.L,color='grey')
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
        #print(np.exp(-self.evaluateU(np.sqrt(200**2+0**2),200,0)), np.exp(-self.evaluateU(np.sqrt(200**2+0**2),0,200)))
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
        x_up[0,:]=np.array([ 3.1238295408e+01,  4.2012193946e+01 ])
        x_up[1,:]=np.array([ 9.5666709302e+01,  8.4170479936e+01])
        x_dn=np.zeros((2,2))
        x_dn[0,:]=np.array([ 3.6520194891e+01,  3.5141300905e+01])
        x_dn[1,:]=np.array([ 2.8841957280e+01,  1.7289187838e+01])
        xvalues1=[x_up,x_dn]
        return xvalues1

    def get_xvalues0step(self):
        x_up=np.zeros((2,2))
        x_up[0,:]=np.array([  3.7496158783e+01 ,4.7465922021e+01])
        x_up[1,:]=np.array([ 9.5666709302e+01,  8.4170479936e+01])
        x_dn=np.zeros((2,2))
        x_dn[0,:]=np.array([ 3.6520194891e+01,  3.5141300905e+01])
        x_dn[1,:]=np.array([ 2.8841957280e+01,  1.7289187838e+01])
        xvalues1=[x_up,x_dn]
        return xvalues1  

    def get_xvalues1(self):
        x_up=np.zeros((2,2))
        x_up[0,:]=np.array([ 3.1238295408e+01,  4.2012193946e+01 ])
        x_up[1,:]=np.array([ 9.5666709302e+01,  8.4170479936e+01])
        x_dn=np.zeros((2,2))
        x_dn[0,:]=np.array([ 3.6520194891e+01,  3.5141300905e+01])
        x_dn[1,:]=np.array([ 2.8841957280e+01,  1.7289187838e+01])
        xvalues1=[x_up,x_dn]
        return xvalues1

    def get_xvalues1step(self):
        x_up=np.zeros((2,2))
        x_up[0,:]=np.array([ 3.1238295408e+01,  4.2012193946e+01])
        x_up[1,:]=np.array([  1.0245507766e+02,  7.9054574015e+01])
        x_dn=np.zeros((2,2))
        x_dn[0,:]=np.array([ 3.6520194891e+01,  3.5141300905e+01])
        x_dn[1,:]=np.array([ 2.8841957280e+01,  1.7289187838e+01])
        xvalues1=[x_up,x_dn]
        return xvalues1

    def get_xvalues2(self):
        x_up=np.zeros((2,2))
        x_up[0,:]=np.array([ 3.1238295408e+01,  4.2012193946e+01 ])
        x_up[1,:]=np.array([1.0245507766e+02,  7.9054574015e+01 ])
        x_dn=np.zeros((2,2))
        x_dn[0,:]=np.array([ 3.6520194891e+01,  3.5141300905e+01])
        x_dn[1,:]=np.array([ 2.8841957280e+01,  1.7289187838e+01])
        xvalues1=[x_up,x_dn]
        return xvalues1
    
    def get_xvalues2step(self):
        x_up=np.zeros((2,2))
        x_up[0,:]=np.array([ 3.1238295408e+01,  4.2012193946e+01 ])
        x_up[1,:]=np.array([ 1.0245507766e+02,  7.9054574015e+01 ])
        x_dn=np.zeros((2,2))
        x_dn[0,:]=np.array([ 2.1861897051e+01,  4.3474814145e+01])
        x_dn[1,:]=np.array([ 2.8841957280e+01,  1.7289187838e+01])
        xvalues1=[x_up,x_dn]
        return xvalues1  

    def get_xvalues3(self):
        x_up=np.zeros((2,2))
        x_up[0,:]=np.array([ 3.1238295408e+01,  4.2012193946e+01 ])
        x_up[1,:]=np.array([ 1.0245507766e+02,  7.9054574015e+01  ])
        x_dn=np.zeros((2,2))
        x_dn[0,:]=np.array([  2.1861897051e+01,  4.3474814145e+01])
        x_dn[1,:]=np.array([ 2.8841957280e+01,  1.7289187838e+01])
        xvalues1=[x_up,x_dn]
        return xvalues1 

    def get_xvalues3step(self):
        x_up=np.zeros((2,2))
        x_up[0,:]=np.array([ 3.1238295408e+01,  4.2012193946e+01 ])
        x_up[1,:]=np.array([ 1.0245507766e+02,  7.9054574015e+01 ])
        x_dn=np.zeros((2,2))
        x_dn[0,:]=np.array([ 2.1861897051e+01,  4.3474814145e+01])
        x_dn[1,:]=np.array([3.3801028143e+01,  1.5265475615e+01 ])
        xvalues1=[x_up,x_dn]
        return xvalues1  

    def get_xvaluesekin1(self):
        x_up=np.zeros((2,2))
        x_up[0,:]=np.array([3.1898587304e+01,  4.5489575910e+01])
        x_up[1,:]=np.array([9.7924250465e+01,  8.4196218662e+01 ])
        x_dn=np.zeros((2,2))
        x_dn[0,:]=np.array([ 2.4166590610e+01,  3.7050606437e+01 ])
        x_dn[1,:]=np.array([3.5638298546e+01,  4.8627172893e+00 ])
        xvalues1=[x_up,x_dn]
        return xvalues1  

    def get_xvaluesekin2(self):
        x_up=np.zeros((2,2))
        x_up[0,:]=np.array([3.8023093044e+01,  3.6021501122e+01])
        x_up[1,:]=np.array([ 9.9178621639e+01,  8.8730735449e+01])
        x_dn=np.zeros((2,2))
        x_dn[0,:]=np.array([ 2.3538940661e+01,  3.6271387996e+01])
        x_dn[1,:]=np.array([3.0583036727e+01,  8.1186364174e-01])
        xvalues1=[x_up,x_dn]
        return xvalues1  


class TestMain:
    def __init__(self):
        self.initialized=1

    def get_Ekinfinitediff(self):

        Wf_=wf()
        Pos_=ParticlePos()

        ##

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

        psiJastrow2=Wf_.PsiJastrow(xvalues2)
        psiJastrow2step=Wf_.PsiJastrow(xvalues2step)
        print("Jastrow ratio 2: ",psiJastrow2step/psiJastrow2) 

        ##
        xvalues3=Pos_.get_xvalues3()
        psi3=Wf_.Psi(xvalues3,True,1,1)

        xvalues3step=Pos_.get_xvalues3step()
        psi3step=Wf_.Psi(xvalues3step,True,1,1)
        print("xvalues diff (1,1): ","\n",xvalues3step[0]-xvalues3[0],"\n", xvalues3step[1]-xvalues3[1])

        print("ratio 3: ",psi3step/psi3  ,np.conjugate(psi3step)*psi3step/(np.conjugate(psi3)*psi3))
        print("\n","\n")


        
        #print(np.exp(1j*np.dot(a,b)))


        Ekin=Wf_.Ekinloc_finitediff(xvalues3step,1e-4)
        print("Ekin finite diff: ",Ekin)
        #print(np.exp(1j*np.dot(a,b)))
        xvaluesekin1=Pos_.get_xvaluesekin1()
        Ekin=Wf_.Ekinloc_finitediff(xvaluesekin1,1e-4)
        print("Ekin finite diff: ",Ekin)

        xvaluesekin2=Pos_.get_xvaluesekin2()
        Ekin=Wf_.Ekinloc_finitediff(xvaluesekin2,1e-4)
        print("Ekin finite diff: ",Ekin)

        
        

       

    def get_Psideriv1(self,igderiv,ideriv,delta=1e-4):
        Wf1_=wf()
        Wf2_=wf(True)
        Wf2_.changecoefs_forderiv(igderiv,ideriv,delta)
        Pos_=ParticlePos()
        xvalues=Pos_.get_xvaluesekin1()



        #print((Wf2_.computeU(xvalues)-Wf1_.computeU(xvalues))/delta)
        psi1=Wf1_.Psi(xvalues)

        psi2=Wf2_.Psi(xvalues)

        return ((psi2-psi1)/delta)/psi1

    def get_Psideriv2(self,igderiv,ideriv,delta=1e-4):
        Wf1_=wf()
        Wf2_=wf()
        Wf2_.changecoefs_forderiv(igderiv,ideriv,delta)
        Pos_=ParticlePos()
        xvalues=Pos_.get_xvaluesekin2()
        psi1=Wf1_.Psi(xvalues)

        psi2=Wf2_.Psi(xvalues)

        return ((psi2-psi1)/delta)/psi1

    
    def run(self):
        #self.get_Ekinfinitediff()
        #Wf_=wf()
        #Wf_.plotJastrow2(0,0)
        #Wf_.testU()
        
        #print(7,self.get_Psideriv1(0,7,1e-4))

        
        it=0
        
        for ig in range(2):
            for i in range(16):
                print(it, self.get_Psideriv1(ig,i,1e-6))
                it+=1

        it=0
        for ig in range(2):
            for i in range(16):
                print(it, self.get_Psideriv2(ig,i,1e-6))
                it+=1     
        
        

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