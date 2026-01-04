//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Ken Esler, kpesler@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Jaron T. Krogel, krogeljt@ornl.gov, Oak Ridge National Laboratory
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#include "EwaldRef.h"
#include "CoulombPBCAA.h"
#include "Particle/DistanceTable.h"
#include "Utilities/ProgressReportEngine.h"
#include "Numerics/Struve.h"
#include <cmath>
#include <numeric>
#include <string>


//std::vector<double> AVdistances;

namespace qmcplusplus
{
CoulombPBCAA::CoulombPBCAA(ParticleSet& ref, bool active, bool computeForces)
    : ForceBase(ref, ref),
      is_active(active),
      FirstTime(true),
      myConst(0.0),
      ComputeForces(computeForces),
      Ps(ref),
      d_aa_ID(ref.addTable(ref))
{
  ReportEngine PRE("CoulombPBCAA", "CoulombPBCAA");
  setEnergyDomain(POTENTIAL);
  twoBodyQuantumDomain(ref);
  PtclRefName = ref.getDistTable(d_aa_ID).getName();
  initBreakup(ref);

  if (ComputeForces)
  {
    ref.turnOnPerParticleSK();
    updateSource(ref);
  }
  if (!is_active)
  {
    ref.update();
    updateSource(ref);

    ewaldref::RealMat A;
    ewaldref::PosArray R;
    ewaldref::ChargeArray Q;

    A = Ps.Lattice.R;

    //L0=ref.Lattice.Length[0];  //AV added
    //L1=ref.Lattice.Length[1];  //AV added

    R.resize(NumCenters);
    Q.resize(NumCenters);
    for (int i = 0; i < NumCenters; ++i)
    {
      R[i] = Ps.R[i];
      Q[i] = Zat[i];
    }

    RealType Vii_ref        = ewaldref::ewaldEnergy(A, R, Q);
    RealType Vdiff_per_atom = std::abs(value_ - Vii_ref) / NumCenters;
    app_log() << "Checking ion-ion Ewald energy against reference..." << std::endl;
    if (Vdiff_per_atom > Ps.Lattice.LR_tol)
    {
      std::ostringstream msg;
      msg << std::setprecision(14);
      msg << "in ion-ion Ewald energy exceeds " << Ps.Lattice.LR_tol << " Ha/atom tolerance." << std::endl;
      msg << std::endl;
      msg << "  Reference ion-ion energy: " << Vii_ref << std::endl;
      msg << "  QMCPACK   ion-ion energy: " << value_ << std::endl;
      msg << "            ion-ion diff  : " << value_ - Vii_ref << std::endl;
      msg << "            diff/atom     : " << (value_ - Vii_ref) / NumCenters << std::endl;
      msg << "            tolerance     : " << Ps.Lattice.LR_tol << std::endl;
      msg << std::endl;
      msg << "Please try increasing the LR_dim_cutoff parameter in the <simulationcell/>" << std::endl;
      msg << "input.  Alternatively, the tolerance can be increased by setting the" << std::endl;
      msg << "LR_tol parameter in <simulationcell/> to a value greater than " << Ps.Lattice.LR_tol << ". " << std::endl;
      msg << "If you increase the tolerance, please perform careful checks of energy" << std::endl;
      msg << "differences to ensure this error is controlled for your application." << std::endl;
      msg << std::endl;

      throw std::runtime_error(msg.str());
    }
    else
    {
      app_log() << "  Check passed." << std::endl;
    }
  }
  prefix = "F_AA";
  app_log() << "  Maximum K shell " << AA->MaxKshell << std::endl;
  app_log() << "  Number of k vectors " << AA->Fk.size() << std::endl;
  app_log() << "  Fixed Coulomb potential for " << ref.getName();
  app_log() << "\n    e-e Madelung Const. =" << MC0 << "\n    Vtot     =" << value_ << std::endl;

  double d_ = 200; // Distance to gate in units of unit cell size
  int epsilon_d_ = 10; // AlAs dielectric constant (GaAs? AlGaAs?)

  double qe_ = 1.60217662e-19; // electron charge
  double ke_ = 8.99e9;         // Coulomb constant
  double hbar_=1.0545718*1e-34;  //hbar (in SI units)
  double a_ = 0.56605e-9;        // AlAs lattice spacing in nanometers

  double me_=9.1093837015*1e-31;  //#kg - umrechnen?? 1/a? meV?
  double mstar_=0.457*me_*qe_/(hbar_*hbar_*1e3)*a_*a_; 
  e_squared_ = ke_*qe_/a_*1e3/epsilon_d_; // e^2 in units of meV*a
  std::cout<<"AV in CoulombPBCAA.cpp, e_squared_: "<<e_squared_<<std::endl;
  e_squared_=e_squared_; //*mstar_;  AV change for rescaling..
  std::cout<<"                        e_squarednew: "<<e_squared_<<std::endl<<std::endl;
  q_tf_=0.01;

  //For screened Coulomb: load U,r
  //Teste r-> i Umkehrformel hier in Konstruktor!! (Assert, falls mal aus Versehen anders konstruiertes r/U geladen wird)
  std::string uscr1="Uscreenedto1.txt";
  std::string uscr2="Uscreened1to5001.txt";

  std::ifstream fin1(uscr1.c_str());

    if(!fin1.good()){
      std::cerr<<"# Error : Cannot load from file "<<uscr1<<" : file not found."<<std::endl;
      std::abort();
    }

   std::ifstream fin2(uscr2.c_str());

    if(!fin2.good()){
      std::cerr<<"# Error : Cannot load from file "<<uscr2<<" : file not found."<<std::endl;
      std::abort();
    } 

  Nvalues1=10000;
  Nvalues2=100000;

  rvalues.resize(Nvalues1+1+Nvalues2+1);
  Uvalues.resize(Nvalues1+1+Nvalues2+1);

  for(int i=0; i<Nvalues1+1; i++){
    fin1>>rvalues[i];
  }

  for(int i=0; i<Nvalues1+1; i++){
    fin1>>Uvalues[i];
  }

  for(int i=Nvalues1+1; i<Nvalues1+1+Nvalues2+1; i++){
    fin2>>rvalues[i];
  }

  for(int i=Nvalues1+1; i<Nvalues1+1+Nvalues2+1; i++){
    fin2>>Uvalues[i];
  }

  for(int i=0; i<Nvalues1+1+Nvalues2+1; i++){
    Uvalues[i]=Uvalues[i]*4.0/d_;
  }

  //test r, U:
  if (rvalues[get_index(0.5)]>0.5 || rvalues[get_index(0.5)+1]<0.5){
    std::cout<<"AV in CoulombPBCAA, something not right with loaded rvalues (Uscreenedto1.txt)"<<std::endl;
    std::flush(std::cout);
    abort();
  }

  if (rvalues[get_index(10.5)]>10.5 || rvalues[get_index(10.5)+1]<10.5){
    std::cout<<"AV in CoulombPBCAA, something not right with loaded rvalues (Uscreened1to5001.txt)"<<std::endl;
    std::flush(std::cout);
    abort();
  }

  std::cout<<"AV in CoulombPBCAA, r: "<<rvalues[get_index(0.5)]<<" U(0.5) interpolated"<<get_linear_interpolated_U(0.5)<<" get_index: "<<Uvalues[get_index(0.5)]<<" get index+1: "<<Uvalues[get_index(0.5)+1]<<std::endl;
  if (get_linear_interpolated_U(0.5)>Uvalues[get_index(0.5)] || get_linear_interpolated_U(0.5)<Uvalues[get_index(0.5)+1]){
    std::cout<<"AV in CoulombPBCAA, something not right with loaded Uvalues/ interpolation (Uscreenedto1.txt)"<<std::endl;
    std::flush(std::cout);
    abort();
  }
  
std::cout<<"AV in CoulombPBCAA, r: "<<rvalues[get_index(10.5)]<<" U(10.5) interpolated"<<get_linear_interpolated_U(10.5)<<" get_index: "<<Uvalues[get_index(10.5)]<<" get index+1: "<<Uvalues[get_index(10.5)+1]<<std::endl;
if (get_linear_interpolated_U(10.5)>Uvalues[get_index(10.5)] || get_linear_interpolated_U(10.5)<Uvalues[get_index(10.5)+1]){
    std::cout<<"AV in CoulombPBCAA, something not right with loaded Uvalues/ interpolation (Uscreened1to5001.txt)"<<std::endl;
    std::flush(std::cout);
    abort();
  }

}


CoulombPBCAA::~CoulombPBCAA() = default;


int  CoulombPBCAA::get_index(double rvalue){
  if (rvalue<=1){
    return (int)((std::log(rvalue)+6.0)*Nvalues1/6.0);
  }
  else{
    return (std::min((int)((rvalue-1)*Nvalues2/5000-1+(Nvalues1+1)),Nvalues1+Nvalues2));
  }
}

double CoulombPBCAA::get_linear_interpolated_U(double rvalue){
  int ind1=get_index(rvalue);
  double diff1=rvalue-rvalues[ind1];
  return (Uvalues[ind1+1]-Uvalues[ind1])/(rvalues[ind1+1]-rvalues[ind1])*diff1+Uvalues[ind1];
}





void CoulombPBCAA::addObservables(PropertySetType& plist, BufferType& collectables)
{
  addValue(plist);
  if (ComputeForces)
    addObservablesF(plist);
}

void CoulombPBCAA::updateSource(ParticleSet& s)
{
  mRealType eL(0.0), eS(0.0);
  if (ComputeForces)
  {
    forces = 0.0;
    eS     = evalSRwithForces(s);
    eL     = evalLRwithForces(s);
  }
  else
  {
    eL = evalLR(s);
    eS = evalSR(s);
  }
  new_value_ = value_ = eL + eS + myConst;
}

void CoulombPBCAA::resetTargetParticleSet(ParticleSet& P)
{
  if (is_active)
  {
    PtclRefName = P.getDistTable(d_aa_ID).getName();
    AA->resetTargetParticleSet(P);
  }
}


#if !defined(REMOVE_TRACEMANAGER)
void CoulombPBCAA::contributeParticleQuantities() { request_.contribute_array(name_); }

void CoulombPBCAA::checkoutParticleQuantities(TraceManager& tm)
{
  streaming_particles_ = request_.streaming_array(name_);
  if (streaming_particles_)
  {
    Ps.turnOnPerParticleSK();
    V_sample = tm.checkout_real<1>(name_, Ps);
    if (!is_active)
      evaluate_sp(Ps);
  }
}

void CoulombPBCAA::deleteParticleQuantities()
{
  if (streaming_particles_)
    delete V_sample;
}
#endif


CoulombPBCAA::Return_t CoulombPBCAA::evaluate(ParticleSet& P)  //AV modify (?) here
{
  if (is_active)
  {
#if !defined(REMOVE_TRACEMANAGER)
    if (streaming_particles_)
      value_ = evaluate_sp(P);
    else{
#endif
      //std::cout<<"AV in CoulombPBCAA::evaluate, evalLR: "<<evalLR(P)<<" evalSR: "<<evalSR(P)<<" myconst: "<<myConst<<std::endl;
      value_ = evalSRTF(P); //evalLR(P) + evalSR(P) + myConst; //evalSRTF(P); //Thomas-Fermi Screening; evalLR(P) + evalSR(P) + myConst;
      }
  }
  value_=(value_*e_squared_); //AV changed: e_squared_. Important to change value_, not only return statement: 
                               //member variable of OperatorBase
                               //no additional minus sign: negativ because of constant shift, myConst
  return value_;  
}

CoulombPBCAA::Return_t CoulombPBCAA::evaluateWithIonDerivs(ParticleSet& P,
                                                           ParticleSet& ions,
                                                           TrialWaveFunction& psi,
                                                           ParticleSet::ParticlePos_t& hf_terms,
                                                           ParticleSet::ParticlePos_t& pulay_terms)
{
  if (ComputeForces and !is_active)
    hf_terms -= forces;
  //No pulay term.
  return value_;
}

#if !defined(REMOVE_TRACEMANAGER)
CoulombPBCAA::Return_t CoulombPBCAA::evaluate_sp(ParticleSet& P)
{
  mRealType Vsr              = 0.0;
  mRealType Vlr              = 0.0;
  mRealType& Vc              = myConst;
  Array<RealType, 1>& V_samp = *V_sample;
  V_samp                     = 0.0;
  {
    //SR
    const auto& d_aa(P.getDistTableAA(d_aa_ID));
    RealType z;
    for (int ipart = 1; ipart < NumCenters; ipart++)
    {
      z                = .5 * Zat[ipart];
      const auto& dist = d_aa.getDistRow(ipart);
      for (int jpart = 0; jpart < ipart; ++jpart)
      {
        RealType pairpot = z * Zat[jpart] * rVs->splint(dist[jpart]) / dist[jpart];
        V_samp(ipart) += pairpot;
        V_samp(jpart) += pairpot;
        Vsr += pairpot;
      }
    }
    Vsr *= 2.0;
  }
  {
    //LR
    const StructFact& PtclRhoK(*(P.SK));
    if (PtclRhoK.SuperCellEnum == SUPERCELL_SLAB)
    {
      APP_ABORT("CoulombPBCAA::evaluate_sp single particle traces have not been implemented for slab geometry");
    }
    else
    {
      assert(PtclRhoK.isStorePerParticle()); // ensure this so we know eikr_r has been allocated
      //jtk mark: needs optimizations for USE_REAL_STRUCT_FACTOR
      RealType v1; //single particle energy
      RealType z;
      for (int i = 0; i < NumCenters; i++)
      {
        z  = .5 * Zat[i];
        v1 = 0.0;
        for (int s = 0; s < NumSpecies; ++s)
        {
#if defined(USE_REAL_STRUCT_FACTOR)
          v1 += z * Zspec[s] *
              AA->evaluate(PtclRhoK.getKLists().kshell, PtclRhoK.rhok_r[s], PtclRhoK.rhok_i[s], PtclRhoK.eikr_r[i],
                           PtclRhoK.eikr_i[i]);
#else
          v1 += z * Zspec[s] * AA->evaluate(PtclRhoK.getKLists().kshell, PtclRhoK.rhok[s], PtclRhoK.eikr[i]);
#endif
        }
        V_samp(i) += v1;
        Vlr += v1;
      }
    }
  }
  for (int i = 0; i < V_samp.size(); ++i)
    V_samp(i) += V_const(i);
  value_ = Vsr + Vlr + Vc;
#if defined(TRACE_CHECK)
  RealType Vlrnow = evalLR(P);
  RealType Vsrnow = evalSR(P);
  RealType Vcnow  = myConst;
  RealType Vnow   = Vlrnow + Vsrnow + Vcnow;
  RealType Vsum   = V_samp.sum();
  RealType Vcsum  = V_const.sum();
  if (std::abs(Vsum - Vnow) > TraceManager::trace_tol)
  {
    app_log() << "accumtest: CoulombPBCAA::evaluate()" << std::endl;
    app_log() << "accumtest:   tot:" << Vnow << std::endl;
    app_log() << "accumtest:   sum:" << Vsum << std::endl;
    APP_ABORT("Trace check failed");
  }
  if (std::abs(Vcsum - Vcnow) > TraceManager::trace_tol)
  {
    app_log() << "accumtest: CoulombPBCAA::evalConsts()" << std::endl;
    app_log() << "accumtest:   tot:" << Vcnow << std::endl;
    app_log() << "accumtest:   sum:" << Vcsum << std::endl;
    APP_ABORT("Trace check failed");
  }
#endif
  return value_;
}
#endif

void CoulombPBCAA::initBreakup(ParticleSet& P)
{
  //SpeciesSet& tspecies(PtclRef->getSpeciesSet());
  SpeciesSet& tspecies(P.getSpeciesSet());
  //Things that don't change with lattice are done here instead of InitBreakup()
  ChargeAttribIndx = tspecies.addAttribute("charge");
  MemberAttribIndx = tspecies.addAttribute("membersize");
  NumCenters       = P.getTotalNum();
  NumSpecies       = tspecies.TotalNum;

#if !defined(REMOVE_TRACEMANAGER)
  V_const.resize(NumCenters);
#endif

  Zat.resize(NumCenters);
  Zspec.resize(NumSpecies);
  NofSpecies.resize(NumSpecies);
  for (int spec = 0; spec < NumSpecies; spec++)
  {
    Zspec[spec]      = tspecies(ChargeAttribIndx, spec);
    NofSpecies[spec] = static_cast<int>(tspecies(MemberAttribIndx, spec));
  }
  SpeciesID.resize(NumCenters);
  for (int iat = 0; iat < NumCenters; iat++)
  {
    SpeciesID[iat] = P.GroupID[iat];
    Zat[iat]       = Zspec[P.GroupID[iat]];
  }
  AA = LRCoulombSingleton::getHandler(P);
  //AA->initBreakup(*PtclRef);
  myConst = evalConsts();
  myRcut  = AA->get_rc(); //Basis.get_rc();

  if (rVs == nullptr)
    rVs = LRCoulombSingleton::createSpline4RbyVs(AA.get(), myRcut);

  if (ComputeForces)
  {
    dAA = LRCoulombSingleton::getDerivHandler(P);
    if (rVsforce == nullptr)
    {
      rVsforce = LRCoulombSingleton::createSpline4RbyVs(dAA.get(), myRcut);
    }
  }
}


CoulombPBCAA::Return_t CoulombPBCAA::evalLRwithForces(ParticleSet& P)
{
  //  const StructFact& PtclRhoK(*(P.SK));
  std::vector<TinyVector<RealType, DIM>> grad(P.getTotalNum());
  for (int spec2 = 0; spec2 < NumSpecies; spec2++)
  {
    RealType Z2 = Zspec[spec2];
    for (int iat = 0; iat < grad.size(); iat++)
      grad[iat] = TinyVector<RealType, DIM>(0.0);
    //AA->evaluateGrad(P, P, spec2, Zat, grad);
    dAA->evaluateGrad(P, P, spec2, Zat, grad);
    for (int iat = 0; iat < grad.size(); iat++)
      forces[iat] += Z2 * grad[iat];
  } //spec2
  return evalLR(P);
}


CoulombPBCAA::Return_t CoulombPBCAA::evalSRwithForces(ParticleSet& P)
{
  const auto& d_aa(P.getDistTableAA(d_aa_ID));
  mRealType SR = 0.0;
  for (size_t ipart = 1; ipart < (NumCenters / 2 + 1); ipart++)
  {
    mRealType esum   = 0.0;
    const auto& dist = d_aa.getDistRow(ipart);
    const auto& dr   = d_aa.getDisplRow(ipart);
    for (size_t j = 0; j < ipart; ++j)
    {
      RealType V, rV, d_rV_dr, d2_rV_dr2;
      RealType rinv = 1.0 / dist[j];
      rV            = rVsforce->splint(dist[j], d_rV_dr, d2_rV_dr2);
      V             = rV * rinv;
      esum += Zat[j] * rVs->splint(dist[j]) * rinv;

      PosType grad = Zat[j] * Zat[ipart] * (d_rV_dr - V) * rinv * rinv * dr[j];
      forces[ipart] += grad;
      forces[j] -= grad;
    }
    SR += Zat[ipart] * esum;

    const size_t ipart_reverse = NumCenters - ipart;
    if (ipart == ipart_reverse)
      continue;

    esum              = 0.0;
    const auto& dist2 = d_aa.getDistRow(ipart_reverse);
    const auto& dr2   = d_aa.getDisplRow(ipart_reverse);
    for (size_t j = 0; j < ipart_reverse; ++j)
    {
      RealType V, rV, d_rV_dr, d2_rV_dr2;
      RealType rinv = 1.0 / dist2[j];
      rV            = rVsforce->splint(dist2[j], d_rV_dr, d2_rV_dr2);
      V             = rV * rinv;
      esum += Zat[j] * rVs->splint(dist2[j]) * rinv;

      PosType grad = Zat[j] * Zat[ipart_reverse] * (d_rV_dr - V) * rinv * rinv * dr2[j];
      forces[ipart_reverse] += grad;
      forces[j] -= grad;
    }
    SR += Zat[ipart_reverse] * esum;
  }
  return SR;
}


/** evaluate the constant term that does not depend on the position
 *
 * \htmlonly
 * <ul>
 * <li> self-energy: \f$ -\frac{1}{2}\sum_{i} v_l(r=0) q_i^2 = -\frac{1}{2}v_l(r=0) \sum_{alpha} N^{\alpha} q^{\alpha}^2\f$
 * <li> background term \f$ V_{bg} = -\frac{1}{2}\sum_{\alpha}\sum_{\beta} N^{\alpha}q^{\alpha}N^{\beta}q^{\beta} v_s(k=0)\f$
 * </ul>
 * \endhtmlonly
 * CoulombPBCABTemp contributes additional background term which completes the background term
 */
CoulombPBCAA::Return_t CoulombPBCAA::evalConsts(bool report)
{
  mRealType Consts = 0.0; // constant term
  mRealType v1;           //single particle energy
#if !defined(REMOVE_TRACEMANAGER)
  V_const = 0.0;
#endif
  //v_l(r=0) including correction due to the non-periodic direction
  mRealType vl_r0 = AA->evaluateLR_r0();
  for (int ipart = 0; ipart < NumCenters; ipart++)
  {
    v1 = -.5 * Zat[ipart] * Zat[ipart] * vl_r0;
#if !defined(REMOVE_TRACEMANAGER)
    V_const(ipart) += v1;
#endif
    Consts += v1;
  }
  if (report)
    app_log() << "   PBCAA self-interaction term " << Consts << std::endl;
  //Compute Madelung constant: this is not correct for general cases
  MC0 = 0.0;
  for (int i = 0; i < AA->Fk.size(); i++)
    MC0 += AA->Fk[i];
  MC0 = 0.5 * (MC0 - vl_r0);
  //Neutraling background term
  mRealType vs_k0 = AA->evaluateSR_k0(); //v_s(k=0)
  for (int ipart = 0; ipart < NumCenters; ipart++)
  {
    v1 = 0.0;
    for (int spec = 0; spec < NumSpecies; spec++)
      v1 += NofSpecies[spec] * Zspec[spec];
    v1 *= -.5 * Zat[ipart] * vs_k0;
#if !defined(REMOVE_TRACEMANAGER)
    V_const(ipart) += v1;
#endif
    Consts += v1;
  }
  if (report)
    app_log() << "   PBCAA total constant " << Consts << std::endl;
  //app_log() << "   MC0 of PBCAA " << MC0 << std::endl;
  return Consts;
}


/*
CoulombPBCAA::Return_t CoulombPBCAA::evalSRTF(ParticleSet& P)
{
  const auto& d_aa(P.getDistTableAA(d_aa_ID));
  mRealType SR = 0.0;
//#pragma omp parallel for reduction(+ : SR)  //AV uncomment!!!!!
  for (size_t ipart = 1; ipart < (NumCenters / 2 + 1); ipart++)
  {
    mRealType esum   = 0.0;
    const auto& dist = d_aa.getDistRow(ipart);
    for (size_t j = 0; j < ipart; ++j){
      double h0;
      STVH0(dist[j]*q_tf_, &h0);
      double n0=std::cyl_neumann(0,dist[j]*q_tf_);
      esum += Zat[j] * (1.0 / dist[j]-M_PI/2.0*q_tf_*(h0-n0));  // rVs->splint(dist[j]) / dist[j];  //AV: Coulomb here -> put here some prefactor?? Screened?
      //std::cout<<"dist[j]: "<<dist[j]<<" j: "<<j<<" ipart: "<<ipart<<" h0 n0:"<<h0<<" "<<n0<<" esum: "<<esum<<std::endl;
      //std::cout<<"AV in CoulombPBCAA::evalSR, rVs->splint: "<<rVs->splint(dist[j])<<"ipart: "<<ipart<<" j: "<<j<<"dist[j]: "<<dist[j]<<std::endl;
    }
    SR += Zat[ipart] * esum;

    const size_t ipart_reverse = NumCenters - ipart;
    if (ipart == ipart_reverse)
      continue;

    esum              = 0.0;
    const auto& dist2 = d_aa.getDistRow(ipart_reverse);
    for (size_t j = 0; j < ipart_reverse; ++j){
      double h0;
      STVH0(dist2[j]*q_tf_, &h0);
      double n0=std::cyl_neumann(0,dist2[j]*q_tf_);
      //splint: short-range: not complete Coulomb potential, but short-range part (cutoff). 
      //Splint artificial short-range potential that is zero at rs. Coulomb potential recovered in sum with long-range part
      esum += Zat[j] * (1.0 / dist2[j]-M_PI/2.0*q_tf_*(h0-n0));  //Zat[j] * rVs->splint(dist2[j]) / dist2[j];  //AV: Coulomb here, splint: short-range potential that goes to zero for rs
      //std::cout<<"dist[j]: "<<dist2[j]<<" j: "<<j<<" ipart_reverse: "<<ipart_reverse<<" h0 n0:"<<h0<<" "<<n0<<" esum: "<<esum<<std::endl; //std::cout<<"AV in CoulombPBCAA::evalSR, rVs->splint: "<<rVs->splint(dist2[j])<<"ipart_reverse: "<<ipart_reverse<<" j: "<<j<<"dist2[j]: "<<dist2[j]<<std::endl;
    }
    SR += Zat[ipart_reverse] * esum;
  }
  return SR;
}
*/


CoulombPBCAA::Return_t CoulombPBCAA::evalSRTF(ParticleSet& P)
{ //put: if different spins, factor 2

  double L0=P.Lattice.Length[0];
  double L1=P.Lattice.Length[1];

  const auto& d_aa(P.getDistTableAA(d_aa_ID));
  mRealType SR = 0.0;
//#pragma omp parallel for reduction(+ : SR)  //AV uncomment!!!!!
  for (size_t ipart = 1; ipart < (NumCenters / 2 + 1); ipart++)
  {
    mRealType esum   = 0.0;
    const auto& dist = d_aa.getDistRow(ipart);

    //--------AV adding displacements for potential outside Wigner-Seitz ----
    const auto& displ = d_aa.getDisplRow(ipart);
    //-----------------------------------------------

    //AVdistances.push_back(dist[0]); //AV added for testing

    for (size_t j = 0; j < ipart; ++j){
      //double h0;
      //STVH0(dist[j]*q_tf_, &h0);
      //double n0=std::cyl_neumann(0,dist[j]*q_tf_);
      esum +=  Zat[j] * get_linear_interpolated_U(dist[j]);  // std::exp(-dist[j]*dist[j]/2.0); //(1.0 / dist[j]-M_PI/2.0*q_tf_*(h0-n0));  // rVs->splint(dist[j]) / dist[j];  //AV: Coulomb here -> put here some prefactor?? Screened?
      //std::cout<<"P.Lattice length: "<<P.Lattice.Length<<" dist[j]: "<<dist[j]<<" displ[j]: "<<displ[j]<<" displ dist: "<<std::sqrt(displ[j][0]*displ[j][0]+displ[j][1]*displ[j][1])<<" j: "<<j<<" ipart: "<<ipart<<" esum:"<<esum<<std::endl;
      //std::cout<<"AV in CoulombPBCAA::evalSR, rVs->splint: "<<rVs->splint(dist[j])<<"ipart: "<<ipart<<" j: "<<j<<"dist[j]: "<<dist[j]<<std::endl;
      //-------

      double dx_     = L0 * ((displ[j][0]/L0) - round(displ[j][0]/L0));  //finding closest distance like this!!!!
      double dy_     = L1 * ((displ[j][1]/L1) - round(displ[j][1]/L1));
      //dx[iat]     = L0 * x ;  //AV, change
      //dy[iat]     = L1 * y ;  //AV, change 
      //dz[iat]     = L2 * (z - round(z));
      double r_compare=dist[j];
      if ( std::abs(std::sqrt(dx_ * dx_ + dy_ * dy_)-r_compare)>1e-6){
        std::cout<<"AV in CoulombPBCAA, dist - displ not right"<<std::endl;
        std::flush(std::cout);
        abort;
      }
      double rmm=std::sqrt((dx_-L0) * (dx_-L0) + (dy_-L1) * (dy_-L1));
      double rm0=std::sqrt((dx_-L0) * (dx_-L0) + (dy_) * (dy_));
      double rmp=std::sqrt((dx_-L0) * (dx_-L0) + (dy_+L1) * (dy_+L1));

      double r0m=std::sqrt((dx_) * (dx_) + (dy_-L1) * (dy_-L1));
      double r0p=std::sqrt((dx_) * (dx_) + (dy_+L1) * (dy_+L1));

      double rpm=std::sqrt((dx_+L0) * (dx_+L0) + (dy_-L1) * (dy_-L1));
      double rp0=std::sqrt((dx_+L0) * (dx_+L0) + (dy_) * (dy_));
      double rpp=std::sqrt((dx_+L0) * (dx_+L0) + (dy_+L1) * (dy_+L1));

      esum +=  Zat[j] * (get_linear_interpolated_U(rmm)+  get_linear_interpolated_U(rm0)+ get_linear_interpolated_U(rmp)+get_linear_interpolated_U(r0m)
                         + get_linear_interpolated_U(r0p) + get_linear_interpolated_U(rpm) + get_linear_interpolated_U(rp0) +  get_linear_interpolated_U(rpp));
      //-----------

    }
    SR += Zat[ipart] * esum;

    const size_t ipart_reverse = NumCenters - ipart;
    if (ipart == ipart_reverse)
      continue;

    esum              = 0.0;
    const auto& dist2 = d_aa.getDistRow(ipart_reverse);
    const auto& displ2 = d_aa.getDisplRow(ipart_reverse);  //AV added

    for (size_t j = 0; j < ipart_reverse; ++j){
      //double h0;
      //STVH0(dist2[j]*q_tf_, &h0);
      //double n0=std::cyl_neumann(0,dist2[j]*q_tf_);
      //splint: short-range: not complete Coulomb potential, but short-range part (cutoff). 
      //Splint artificial short-range potential that is zero at rs. Coulomb potential recovered in sum with long-range part
      esum += Zat[j] * get_linear_interpolated_U(dist2[j]);  //std::exp(-dist2[j]*dist2[j]/2.0); //(1.0 / dist2[j]-M_PI/2.0*q_tf_*(h0-n0));  //Zat[j] * rVs->splint(dist2[j]) / dist2[j];  //AV: Coulomb here, splint: short-range potential that goes to zero for rs
      //std::cout<<"dist[j]: "<<dist2[j]<<" j: "<<j<<" ipart_reverse: "<<ipart_reverse<<" esum: "<<esum<<std::endl; //std::cout<<"AV in CoulombPBCAA::evalSR, rVs->splint: "<<rVs->splint(dist2[j])<<"ipart_reverse: "<<ipart_reverse<<" j: "<<j<<"dist2[j]: "<<dist2[j]<<std::endl;
      //-------

      double dx_     = L0 * ((displ2[j][0]/L0) - round(displ2[j][0]/L0));  //finding closest distance like this!!!!
      double dy_     = L1 * ((displ2[j][1]/L1) - round(displ2[j][1]/L1));
      //dx[iat]     = L0 * x ;  //AV, change
      //dy[iat]     = L1 * y ;  //AV, change 
      //dz[iat]     = L2 * (z - round(z));
      double r_compare=dist2[j];
      if ( std::abs(std::sqrt(dx_ * dx_ + dy_ * dy_)-r_compare)>1e-6){
        std::cout<<"AV in CoulombPBCAA, dist - displ not right"<<std::endl;
        std::flush(std::cout);
        abort;
      }

      double rmm=std::sqrt((dx_-L0) * (dx_-L0) + (dy_-L1) * (dy_-L1));
      double rm0=std::sqrt((dx_-L0) * (dx_-L0) + (dy_) * (dy_));
      double rmp=std::sqrt((dx_-L0) * (dx_-L0) + (dy_+L1) * (dy_+L1));

      double r0m=std::sqrt((dx_) * (dx_) + (dy_-L1) * (dy_-L1));
      double r0p=std::sqrt((dx_) * (dx_) + (dy_+L1) * (dy_+L1));

      double rpm=std::sqrt((dx_+L0) * (dx_+L0) + (dy_-L1) * (dy_-L1));
      double rp0=std::sqrt((dx_+L0) * (dx_+L0) + (dy_) * (dy_));
      double rpp=std::sqrt((dx_+L0) * (dx_+L0) + (dy_+L1) * (dy_+L1));

      esum +=  Zat[j] * (get_linear_interpolated_U(rmm)+  get_linear_interpolated_U(rm0)+ get_linear_interpolated_U(rmp)+get_linear_interpolated_U(r0m)
                         + get_linear_interpolated_U(r0p) + get_linear_interpolated_U(rpm) + get_linear_interpolated_U(rp0) +  get_linear_interpolated_U(rpp));
      //-----------

    }
    SR += Zat[ipart_reverse] * esum;
  }
  return SR;
}



CoulombPBCAA::Return_t CoulombPBCAA::evalSR(ParticleSet& P)
{
  const auto& d_aa(P.getDistTableAA(d_aa_ID));
  mRealType SR = 0.0;
#pragma omp parallel for reduction(+ : SR)
  for (size_t ipart = 1; ipart < (NumCenters / 2 + 1); ipart++)
  {
    mRealType esum   = 0.0;
    const auto& dist = d_aa.getDistRow(ipart);
    for (size_t j = 0; j < ipart; ++j){
      esum += Zat[j] * rVs->splint(dist[j]) / dist[j];  //AV: Coulomb here -> put here some prefactor?? Screened?
      //std::cout<<"AV in CoulombPBCAA::evalSR, rVs->splint: "<<rVs->splint(dist[j])<<"ipart: "<<ipart<<" j: "<<j<<"dist[j]: "<<dist[j]<<std::endl;
    }
    SR += Zat[ipart] * esum;

    const size_t ipart_reverse = NumCenters - ipart;
    if (ipart == ipart_reverse)
      continue;

    esum              = 0.0;
    const auto& dist2 = d_aa.getDistRow(ipart_reverse);
    for (size_t j = 0; j < ipart_reverse; ++j){

      //splint: short-range: not complete Coulomb potential, but short-range part (cutoff). 
      //Splint artificial short-range potential that is zero at rs. Coulomb potential recovered in sum with long-range part
      esum += Zat[j] * rVs->splint(dist2[j]) / dist2[j];  //AV: Coulomb here, splint: short-range potential that goes to zero for rs
      //std::cout<<"AV in CoulombPBCAA::evalSR, rVs->splint: "<<rVs->splint(dist2[j])<<"ipart_reverse: "<<ipart_reverse<<" j: "<<j<<"dist2[j]: "<<dist2[j]<<std::endl;
    }
    SR += Zat[ipart_reverse] * esum;
  }
  return SR;
}

CoulombPBCAA::Return_t CoulombPBCAA::evalLR(ParticleSet& P)
{
  mRealType res = 0.0;
  const StructFact& PtclRhoK(*(P.SK));
  if (PtclRhoK.SuperCellEnum == SUPERCELL_SLAB)
  {
    const auto& d_aa(P.getDistTableAA(d_aa_ID));
    //distance table handles jat<iat
    for (int iat = 1; iat < NumCenters; ++iat)
    {
      mRealType u = 0;
#if !defined(USE_REAL_STRUCT_FACTOR)
      const int slab_dir              = OHMMS_DIM - 1;
      const RealType* restrict d_slab = d_aa.Displacements[iat].data(slab_dir);
      for (int jat = 0; jat < iat; ++jat)
        u += Zat[jat] *
            AA->evaluate_slab(-d_slab[jat], //JK: Could be wrong. Check the SIGN
                              PtclRhoK.getKLists().kshell, PtclRhoK.eikr[iat], PtclRhoK.eikr[jat]);
#endif
      res += Zat[iat] * u;
    }
  }
  else
  {
    for (int spec1 = 0; spec1 < NumSpecies; spec1++)
    {
      mRealType Z1 = Zspec[spec1];
      for (int spec2 = spec1; spec2 < NumSpecies; spec2++)
      {
#if defined(USE_REAL_STRUCT_FACTOR)
        //goes to LRHandlerBase::evaluate (rhok^2*Fk_symm, where Fk_symm is the fourier transform of the lr potential (erfc)
        mRealType temp = AA->evaluate(PtclRhoK.getKLists().kshell, PtclRhoK.rhok_r[spec1], PtclRhoK.rhok_i[spec1],
                                      PtclRhoK.rhok_r[spec2], PtclRhoK.rhok_i[spec2]);
#else
        mRealType temp = AA->evaluate(PtclRhoK.getKLists().kshell, PtclRhoK.rhok[spec1], PtclRhoK.rhok[spec2]);
#endif
        if (spec2 == spec1)
          temp *= 0.5;
        res += Z1 * Zspec[spec2] * temp;
      } //spec2
    }   //spec1
  }
  return res;
}

std::unique_ptr<OperatorBase> CoulombPBCAA::makeClone(ParticleSet& qp, TrialWaveFunction& psi)
{
  return std::make_unique<CoulombPBCAA>(*this);
}
} // namespace qmcplusplus
