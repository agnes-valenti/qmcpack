//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jaron T. Krogel, krogeljt@ornl.gov, Oak Ridge National Laboratory
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_HEGGRID_H
#define QMCPLUSPLUS_HEGGRID_H

#include "Lattice/CrystalLattice.h"
#include <map>
#include <optional>

namespace qmcplusplus
{
template<typename T>
struct kpdata
{
  TinyVector<T, OHMMS_DIM> k;
  T k2;
  int g;
};


template<typename T>
bool kpdata_comp(const kpdata<T>& left, const kpdata<T>& right)
{
  return left.k2 < right.k2;
}

//three-d specialization
template<class T>
struct HEGGrid
{
  typedef CrystalLattice<T, OHMMS_DIM> PL_t;
  typedef typename PL_t::SingleParticlePos_t PosType;
  typedef typename PL_t::Scalar_t RealType;

  ///number of kpoints of a half sphere excluding gamma
  int NumKptsHalf;
  ///maxmim ksq
  T MaxKsq;
  PL_t& Lattice;
  std::map<int, std::vector<PosType>> rs;
  
  std::vector<PosType> kpt;
  std::vector<PosType> kptxy;  //AV added
  std::vector<T> mk2;
  std::vector<T> mk2xy;  //AV added

  std::vector<int> deg;
  double deltak;

  double mx;
  double my;
  //double AVkmax;
  int kgridmax; //max number of kpoints in 1D (before occupation)
  double theta_x;
  double theta_y;
  double Kstart;

  static constexpr std::array<int, 31> n_within_shell{{1,   7,   19,  27,  33,  57,  81,  93,  123, 147, 171,
                                                       179, 203, 251, 257, 305, 341, 365, 389, 437, 461, 485,
                                                       515, 587, 619, 691, 739, 751, 799, 847, 895}};
  PosType twist{0.0};

  std::vector<std::vector<double> > Kx_;
  std::vector<std::vector<double> > Ky_;
  std::vector<std::vector<double> > Pmatrix_;

  typedef kpdata<T> kpdata_t;
  typedef std::vector<kpdata_t> kpoints_t;

  std::optional<kpoints_t> kpoints_grid;
  int nctmp{-1};
  int Tau;
  double Eta;

  HEGGrid(PL_t& lat, int tau=0) : Lattice(lat), Tau(tau) {
  //int halfnumberpointsm1=1; //AV change this!!
  //AVkmax=0.1;
  std::string filename_eta="EtaEtaVarmax.txt";
  std::ifstream fin_eta(filename_eta.c_str());
  //std::cout<<"(AV) test"<<std::endl;
  if(!fin_eta.good()){
    std::cerr<<"# Error : Cannot load from file "<<filename_eta<<" : file not found."<<std::endl;
    std::abort();
  }
  
  fin_eta>>Eta;
 
  //Eta=5.79; //1.0; //5.76;
  mx=std::pow(Eta,-0.5*Tau);
  my=std::pow(Eta,0.5*Tau);

  std::string filename="kFmax.txt";
  std::ifstream fin(filename.c_str());
  if(!fin.good()){
    std::cerr<<"# Error : Cannot load from file "<<filename<<" : file not found."<<std::endl;
    std::abort();
  }
  //double Kstart=0;
  fin>>Kstart;

  double deltak_linspace=0;
  fin>>deltak_linspace;

  double helperkgrid=0;
  fin>>helperkgrid;
  kgridmax=round(helperkgrid);

  //double theta_x=0;
  //double theta_y=0;
  fin>>theta_x;
  fin>>theta_y;

  //kgridmax=11; //7; //3; //7; //27; //5; //19; //3;
  //std
  //double Kmax=0.06960088759930398161;   //0.044868870512407795; //0.06960088759930398161;   //0.044868870512407795; //0.15; //0.1;
  //double deltak_linspace=2.0*Kmax/(kgridmax-1);

  std::vector<std::vector<double> > Kx(kgridmax,std::vector<double>(kgridmax));
  std::vector<std::vector<double> > Ky(kgridmax,std::vector<double>(kgridmax));

  int k_it=0;
  for (int i=0; i<kgridmax; i++){
    for (int j=0; j<kgridmax; j++){
      Kx[i][j]=Kstart+i*deltak_linspace+theta_x;
      Ky[i][j]=Kstart+j*deltak_linspace+theta_y;
      //std::cout<<"k_it: "<<k_it<<std::endl;
      //std::cout<<"Kx["<<i<<"]["<<j<<"]: "<<Kx[i][j]<<std::endl;
      //std::cout<<"Ky["<<i<<"]["<<j<<"]: "<<Ky[i][j]<<std::endl<<std::endl;
      k_it++;
    }
  }
  
  std::vector<std::vector<double> > Pmatrix(kgridmax*kgridmax,std::vector<double>(4));  //AV: change to 4 when inclding tau!

  std::string filename_Pmat="Pmatrix.txt";
  std::ifstream finPmat(filename_Pmat.c_str());
  if(!finPmat.good()){
    std::cerr<<"# Error : Cannot load from file "<<filename_Pmat<<" : file not found."<<std::endl;
    std::abort();
  }

//std::cout<<"Pmatrix: "<<std::endl;
  for(int i=0; i<kgridmax*kgridmax; i++){  //change!!
    for (int j=0; j<4; j++){
      finPmat>>Pmatrix[i][j];
      //std::cout<<Pmatrix[i][j]<< " ";
    }
    //std::cout<<std::endl;
  }

  deltak=Kx[1][0]-Kx[0][0];

  Kx_=Kx;
  Ky_=Ky;
  Pmatrix_=Pmatrix;

  }


  ~HEGGrid() = default;

  inline void changeType(int particletype){
    if (particletype<1.5){
      Tau=-1;
      int k_it=0;
      for (int i=0; i<kgridmax; i++){
        for (int j=0; j<kgridmax; j++){
          Kx_[i][j]= Kstart+i*deltak+theta_x;
          Ky_[i][j]= Kstart+j*deltak+theta_y;
          //std::cout<<"Kx: "<<Kx_[i][j]<<std::endl;
          //std::cout<<"Ky: "<<Ky_[i][j]<<std::endl;
          k_it++;
        }
      }
    }
    else if (particletype>1.5){
      Tau=1;
      int k_it=0;
      for (int i=0; i<kgridmax; i++){
        for (int j=0; j<kgridmax; j++){
          Kx_[i][j]= Kstart+i*deltak+theta_y;
          Ky_[i][j]= Kstart+j*deltak+theta_x;
          //std::cout<<"Kx: "<<Kx_[i][j]<<std::endl;
          //std::cout<<"Ky: "<<Ky_[i][j]<<std::endl;
          k_it++;
        }
      }
    }
    else{
      std::cout<<"AV in HEGGrid::changeType, Invalid particletype value, abort"<<std::endl;
      std::flush(std::cout);
      abort();
    } 
    //Eta=5.79; //1.0; //5.76;
    mx=std::pow(Eta,-0.5*Tau);
    my=std::pow(Eta,0.5*Tau);
  }

  /** return the cell size  for the number of particles and rs
   * @param nptcl number of particles
   * @param rs_in rs
   */
  inline T getCellLength(T rs_in) const {
    //calculate volume/are here!! What is rs??
    //AV: function not modified, abort if called -----------------------------
    //std::cout<<"AV HEGGrid::getCellLength (presumably unused) called, abort"<<std::endl;
    //std::flush(std::cout);
    //abort();
    //----------------------------------------------------------------------
    //std::cout<<" AV in getCellLength, deltak: "<<deltak<<std::endl;
    //deltak=0.1;
 return 2.0 * M_PI  / deltak; }//std::pow(4.0 * M_PI * nptcl / 3.0, 1.0 / 3.0) * rs_in; }

  void sortGrid(int nc)
  {
    //AV: function not modified, abort if called -----------------------------
    std::cout<<"AV HEGGrid::sort(nc) (presumably unused) called, abort"<<std::endl;
    std::flush(std::cout);
    abort();
    //----------------------------------------------------------------------

    int first_ix2, first_ix3;
    for (int ix1 = 0; ix1 <= nc; ix1++)
    {
      if (ix1 == 0)
        first_ix2 = 0;
      else
        first_ix2 = -nc;
      for (int ix2 = first_ix2; ix2 <= nc; ix2++)
      {
        if (ix1 == 0 && ix2 == 0)
          first_ix3 = 1;
        else
          first_ix3 = -nc;
        for (int ix3 = first_ix3; ix3 <= nc; ix3++)
        {
          int ih = ix1 * ix1 + ix2 * ix2 + ix3 * ix3;
          if (auto it = rs.find(ih); it == rs.end())
            rs[ih] = {PosType(ix1, ix2, ix3)};
          else
            it->second.push_back(PosType(ix1, ix2, ix3));
        }
      }
    }
  }

  

  void clear_kpoints() { kpoints_grid.reset(); }


  void create_kpoints(int nc, int num_kpts, const PosType& tw, T tol = 1e-6)
  {
    //std::cout<<"AV entering HEGGrid::create_kpoints (modify this function to load kpoint grid!!)"<<std::endl;
    if (!kpoints_grid.has_value())
      kpoints_grid = kpoints_t();
  
    //nctmp              = nc;
    kpoints_t& kpoints = *kpoints_grid;

    app_log() << "  resizing kpoint grid" << std::endl;
    app_log() << "  current size = " << kpoints.size() << std::endl;
    // make space for the kpoint grid
    int nkpoints = num_kpts; //*(2*nc+1);
    kpoints.resize(nkpoints);

    app_log() << "AV in HEGGrid::create_kpoint, kpoints size = " << kpoints.size() << std::endl;
    typename kpoints_t::iterator kptmp, kp = kpoints.begin(), kp_end = kpoints.end();
    // make the kpoint grid
    T k2max = std::numeric_limits<RealType>::max();
    int k_it=0;
    kp      = kpoints.begin();
    //kpxy      = kpoints.begin();
    //std::cout<<"Create k grid, component "<<nc<<std::endl;
    for (int i0 = 0 ; i0 < kgridmax ; ++i0){
      for (int i1 = 0 ; i1 < kgridmax ; ++i1)
        {
          if (Pmatrix_[k_it][nc]>0.5){
            //PosType k(i0 + tw[0], i1 + tw[1]);
            //std::cout<<"AV in HEGGrid::create_kpoint, i0+tw[0]: "<<i0 + tw[0]<< " i1+tw[1]: "<<i1+tw[1]<<std::endl;
            //std::cout<<"AV in HEGGrid::create_kpoint, Lattice k cart: "<<Lattice.k_cart(k)<<" k: "<<k<<std::endl;
            PosType k(Kx_[i0][i1], Ky_[i0][i1]);
            kp->k  = k; //Lattice.k_cart(k);
            std::cout<<"AV in HEGGrid::create_kpoint, k: "<<k<<std::endl;
            kp->k2 = dot(k,k);
            //std::cout<<"AV in HEGGrid::create_kpoint, k cart squared: "<<kp->k2<<std::endl;
            //if (std::abs(i0) == (nc + 1) || std::abs(i1) == (nc + 1) || std::abs(i2) == (nc + 1))
            //  k2max = std::min(k2max, kp->k2);
            ++kp;
          }
          k_it+=1;
        }
      }
    // sort kpoints by magnitude
    sort(kpoints.begin(), kpoints.end(), kpdata_comp<T>);
    

    // eliminate kpoints outside of inscribing sphere
    //int nkp = 0;
    //kp      = kpoints.begin();
    //while (kp != kp_end && kp->k2 < k2max + 1e-3)
    //{
    //  nkp++;
    //  ++kp;
    //}
    //kpoints.resize(nkp);
    //app_log() << "  new spherical size = " << kpoints.size() << std::endl;
    //kp_end = kpoints.end();
    //for(kp=kpoints.begin();kp!=kp_end;++kp)
    //   std::cout<<" AV vor count deg   "<<kp->k2<<" "<<kp->g<<" "<<kp->k<< std::endl;
    // count degeneracies
    kp = kpoints.begin();
    while (kp != kp_end)
    {
      T k2  = kp->k2;
      kptmp = kp;
      int g = 1;
      ++kptmp;
      // look ahead to count
      while (kptmp != kp_end && std::abs(kptmp->k2 - k2) < tol)
      {
        g++;
        ++kptmp;
      }
      kp->g = g;
      // run over degenerate states to assign
      for (int n = 0; n < g - 1; ++n)
        (++kp)->g = g;
      ++kp;
    }
    //app_log()<<"create_kpoints"<< std::endl;
    //app_log()<<"  nkpoints = "<<nkpoints<< std::endl;
    //app_log()<<"  kpoints"<< std::endl;
    //for(kp=kpoints.begin();kp!=kp_end;++kp)
    //   std::cout<<" AV nach count deg   "<<kp->k2<<" "<<kp->g<<" "<<kp->k<< std::endl;
    //APP_ABORT("end create_kpoints");
    //std::cout<<"AV exiting HEGGrid::create_kpoints"<<std::endl<<std::endl;
  }


  void createGrid(int nc, int nkpts, const PosType& twistAngle)
  { 
    //AV: function not modified, abort if called -----------------------------
    //std::cout<<"AV HEGGrid::createGrid(nc,nkpts, twistAngle tol=1e-6) presumably unused called, abort"<<std::endl;
    //std::flush(std::cout);
    //abort();
    //----------------------------------------------------------------------

    //std::cout<<"AV entering HEGGrid::createGrid"<<std::endl;
    twist = twistAngle;

    int sumP=0;
    for (int i=0; i<kgridmax*kgridmax; i++){
        sumP=sumP+Pmatrix_[i][nc];}

    std::cout<<" sumP: "<<sumP<<" nkpts: "<<nkpts<<std::endl;
    if (std::abs(sumP-nkpts)>1e-3){
       std::cout<<"Number of occupied states does not match number of isospin-"<<nc<<" particles"<<std::endl;
       std::flush(std::cout);
       abort();}

    create_kpoints(nc, nkpts, twistAngle);
    kpoints_t& kpoints = *kpoints_grid;
    if (nkpts > kpoints.size())
      APP_ABORT("HEGGrid::createGrid  requested more kpoints than created");
    kpt.resize(nkpts);
    kptxy.resize(nkpts);
    mk2.resize(nkpts);
    mk2xy.resize(nkpts);
    deg.resize(nkpts);
    for (int i = 0; i < nkpts; ++i)
    {
      const kpdata_t& kp = kpoints[i];
      kpt[i]             = kp.k;
      kptxy[i][0]           = 1.0/std::sqrt(mx)*kpt[i][0];
      kptxy[i][1]           = 1.0/std::sqrt(my)*kpt[i][1];
      mk2[i]             = -kp.k2;
      mk2xy[i]           = -1.0/mx*kpt[i][0]*kpt[i][0]-1.0/my*kpt[i][1]*kpt[i][1];
      deg[i]             = kp.g;
    }
    //app_log() << "List of kpoints with twist = " << twistAngle << std::endl;
    //for (int ik = 0; ik < kpt.size(); ik++){
    //  app_log() << ik << " " << kpt[ik] << " " << -mk2[ik] << " deg[ik]"<<deg[ik]<<std::endl;
    //}

    //app_log() << "List of anisotropic kpoints with twist = " << twistAngle << std::endl;
    //for (int ik = 0; ik < kpt.size(); ik++){
    //  app_log() << ik << " " << kptxy[ik] << " " << -mk2xy[ik] << " deg[ik]"<<deg[ik]<<std::endl;
    //}

    //std::cout<<"AV exiting HEGGrid::createGrid"<<std::endl;
  }


};

} // namespace qmcplusplus
#endif
