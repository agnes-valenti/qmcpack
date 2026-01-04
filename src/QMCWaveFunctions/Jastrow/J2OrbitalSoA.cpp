//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021 QMCPACK developers.
//
// File developed by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//                    Amrita Mathuriya, amrita.mathuriya@intel.com, Intel Corp.
//                    Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//////////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-


#include "J2OrbitalSoA.h"
#include "CPU/SIMD/algorithm.hpp"
#include "BsplineFunctor.h"
#include "PadeFunctors.h"
#include "UserFunctor.h"

namespace qmcplusplus
{
template<typename FT>
void J2OrbitalSoA<FT>::checkInVariables(opt_variables_type& active)
{
  myVars.clear();
  auto it(J2Unique.begin()), it_end(J2Unique.end());
  while (it != it_end)
  {
    (*it).second->checkInVariables(active);
    (*it).second->checkInVariables(myVars);
    ++it;
  }
}

template<typename FT>
void J2OrbitalSoA<FT>::checkOutVariables(const opt_variables_type& active)
{
  myVars.getIndex(active);
  Optimizable = myVars.is_optimizable();
  auto it(J2Unique.begin()), it_end(J2Unique.end());
  while (it != it_end)
  {
    (*it).second->checkOutVariables(active);
    ++it;
  }
  if (dPsi)
    dPsi->checkOutVariables(active);
}

template<typename FT>
void J2OrbitalSoA<FT>::resetParameters(const opt_variables_type& active)
{
  if (!Optimizable)
    return;
  auto it(J2Unique.begin()), it_end(J2Unique.end());
  while (it != it_end)
  {
    (*it).second->resetParameters(active);
    ++it;
  }
  if (dPsi)
    dPsi->resetParameters(active);
  for (int i = 0; i < myVars.size(); ++i)
  {
    int ii = myVars.Index[i];
    if (ii >= 0)
      myVars[i] = active[ii];
  }
}

template<typename FT>
void J2OrbitalSoA<FT>::reportStatus(std::ostream& os)
{
  auto it(J2Unique.begin()), it_end(J2Unique.end());
  while (it != it_end)
  {
    (*it).second->myVars.print(os);
    ++it;
  }
}

template<typename FT>
void J2OrbitalSoA<FT>::evaluateRatios(const VirtualParticleSet& VP, std::vector<ValueType>& ratios)
{
  for (int k = 0; k < ratios.size(); ++k){
    valT uvalue=computeU(VP.refPS, VP.refPtcl, VP.getDistTableAB(my_table_ID_).getDistRow(k));
    valT v2value=computeU(VP.refPS, VP.refPtcl, 0, VP.getDistTableAB(my_table_ID_).getDistRow(k), VP.getDistTableAB(my_table_ID_).getDisplRow(k));
    
    ratios[k] = std::exp(Uat[VP.refPtcl] - uvalue - v2value);
  }
}

template<typename FT>
void J2OrbitalSoA<FT>::registerData(ParticleSet& P, WFBufferType& buf)
{
  if (Bytes_in_WFBuffer == 0)
  {
    Bytes_in_WFBuffer = buf.current();
    buf.add(Uat.begin(), Uat.end());
    buf.add(dUat.data(), dUat.end());
    buf.add(d2Uat.begin(), d2Uat.end());
    Bytes_in_WFBuffer = buf.current() - Bytes_in_WFBuffer;
    // free local space
    Uat.free();
    dUat.free();
    d2Uat.free();
  }
  else
  {
    buf.forward(Bytes_in_WFBuffer);
  }
}

template<typename FT>
void J2OrbitalSoA<FT>::copyFromBuffer(ParticleSet& P, WFBufferType& buf)
{
  Uat.attachReference(buf.lendReference<valT>(N), N);
  dUat.attachReference(N, N_padded, buf.lendReference<valT>(N_padded * OHMMS_DIM));
  d2Uat.attachReference(buf.lendReference<valT>(N), N);
}

template<typename FT>
typename J2OrbitalSoA<FT>::LogValueType J2OrbitalSoA<FT>::updateBuffer(ParticleSet& P,
                                                                       WFBufferType& buf,
                                                                       bool fromscratch)
{
  //std::cout<<"AV entering J2OrbitalSoA<FT>::updateBuffer"<<std::endl;
  evaluateGL(P, P.G, P.L, false);
  buf.forward(Bytes_in_WFBuffer);
  //std::cout<<"AV exiting J2OrbitalSoA<FT>::updateBuffer"<<std::endl<<std::endl;
  return log_value_;
}

template<typename FT>
typename J2OrbitalSoA<FT>::valT J2OrbitalSoA<FT>::computeU(const ParticleSet& P, int iat, const DistRow& dist)
{
  valT curUat(0);
  const int igt = P.GroupID[iat] * NumGroups;
  for (int jg = 0; jg < NumGroups; ++jg)
  {
    const FuncType& f2(*F[igt + jg]);
    int iStart = P.first(jg);
    int iEnd   = P.last(jg);
    curUat += f2.evaluateV(iat, iStart, iEnd, dist.data(), DistCompressed.data());
    //std::cout<<"computeU, iat: "<<iat<<" iStart: "<<iStart<<" iEnd: "<<iEnd<<" curUat: "<<curUat<<std::endl;
  }
  return curUat;
}

//AV added
template<typename FT>
typename J2OrbitalSoA<FT>::valT J2OrbitalSoA<FT>::computeU(const ParticleSet& P, int iat, int numpar, const DistRow& dist, const DisplRow& displ)
{
  valT curUat(0);
  const int igt = P.GroupID[iat] * NumGroups;
  //std::cout<<"igt: "<<igt<<std::endl;
  //std::cout<<"NumGroups: "<<NumGroups<<std::endl;
  for (int jg = 0; jg < NumGroups; ++jg)
  {
    const FuncType& f2(*F[igt + jg]);
    int iStart = P.first(jg);
    int iEnd   = P.last(jg);
    //if (numpar==2)
    //   std::cout<<"Group jg: "<<jg<<" iStart: "<<iStart<<" iEnd: "<<iEnd<<std::endl;
    valT f2value= f2.evaluateV2(iat, iStart, iEnd, numpar, dist.data(), displ.data(0), displ.data(1), DistCompressed.data(), DisplCompressedX.data(), DisplCompressedY.data(), Tauvalues[P.GroupID[iat]]+Tauvalues[jg]);
    //std::cout<<"jg: "<<jg<<" iat: "<<iat<<" after evaluateV2: "<<f2value<<" iStart: "<<iStart<<" iEnd: "<<iEnd<<" numpar: "<<numpar<<std::endl;
    //std::cout<<"computeU2, iat: "<<iat<<" iStart: "<<iStart<<" iEnd: "<<iEnd<<" f2value: "<<f2value<<std::endl;

    curUat += f2value; //f2.evaluateV2(iat, iStart, iEnd, numpar, dist.data(), displ.data(0), displ.data(1), DistCompressed.data(), DisplCompressedXsquared.data(), DisplCompressedYsquared.data());
  }
  //if (numpar==2 or numpar==3)
  //  std::cout<<"leave compute U, curUat: "<<curUat<<" Uat: "<<Uat[iat]<<std::endl<<std::endl;
  return curUat;
}

template<typename FT>
typename J2OrbitalSoA<FT>::posT J2OrbitalSoA<FT>::accumulateG(const valT* restrict du, const DisplRow& displ) const
{
  posT grad;
  for (int idim = 0; idim < OHMMS_DIM; ++idim)
  {
    const valT* restrict dX = displ.data(idim);
    valT s                  = valT();

#pragma omp simd reduction(+ : s) aligned(du, dX : QMC_SIMD_ALIGNMENT)
    for (int jat = 0; jat < N; ++jat)
      s += du[jat] * dX[jat];
    grad[idim] = s;
  }
  return grad;
}

template<typename FT>
J2OrbitalSoA<FT>::J2OrbitalSoA(const std::string& obj_name, ParticleSet& p)
    : WaveFunctionComponent("J2OrbitalSoA", obj_name),
      my_table_ID_(p.addTable(p, DTModes::NEED_TEMP_DATA_ON_HOST)),
      j2_ke_corr_helper(p, F)
{
  if (myName.empty())
    throw std::runtime_error("J2OrbitalSoA object name cannot be empty!");
  init(p);
  KEcorr = 0.0;
}

template<typename FT>
J2OrbitalSoA<FT>::~J2OrbitalSoA() = default;

template<typename FT>
void J2OrbitalSoA<FT>::init(ParticleSet& p)
{
  double Eta=0; //5.79; //1.0; //5.79; //1.0 ;//5.76;
  std::string filename="EtaEtaVarmax.txt";
  std::ifstream fin(filename.c_str());
  if(!fin.good()){
    std::cerr<<"# Error : Cannot load from file "<<filename<<" : file not found."<<std::endl;
    std::abort();
  }
  
  fin>>Eta;
  //fin>>etaVar;
  //etaVar=1.0/2.0; //2.0; //1.0; //std::pow(Eta,0.5); //1.0;
  
  int sephelper=0;
  separate=true;
  fin>>sephelper;
  if (sephelper<0.5){
    separate=false;
  }


  N         = p.getTotalNum();
  N_padded  = getAlignedSize<valT>(N);
  NumGroups = p.groups();

  Uat.resize(N);
  dUat.resize(N);
  d2Uat.resize(N);
  cur_u.resize(N);
  cur_du.resize(N);
  cur_d2u.resize(N);

  cur_v.resize(N);
  cur_dvdx.resize(N);
  cur_dvdy.resize(N);
  cur_d2vdx2.resize(N);
  cur_d2vdy2.resize(N);
  
  old_u.resize(N);
  old_du.resize(N);
  old_d2u.resize(N);

  old_v.resize(N);
  old_dvdx.resize(N);
  old_dvdy.resize(N);

  old_d2vdx2.resize(N);
  old_d2vdy2.resize(N);
  
  F.resize(NumGroups * NumGroups, nullptr);
  etavars.resize(NumGroups*NumGroups);


  DistCompressed.resize(N);
  DisplCompressedX.resize(N);
  DisplCompressedY.resize(N);

  DistIndice.resize(N);

  //AV added
  SpeciesSet tspecies(p.getSpeciesSet());
  //std::vector<std::string> name_test=tspecies.speciesName;
  
  int species_set_size=tspecies.size();
  Tauvalues.resize(species_set_size);
  Masses.resize(species_set_size,std::vector<double>(2));
  OneOverSqrtM.resize(species_set_size,std::vector<double>(2));

  int species_index_u=tspecies.findSpecies("u");
  int species_index_d=tspecies.findSpecies("d");
  int species_index_ut=tspecies.findSpecies("ut");
  int species_index_dt=tspecies.findSpecies("dt");
  if (species_index_u<species_set_size){
    int Tau=-1;
    Tauvalues[species_index_u]=Tau;
    Masses[species_index_u][0]=std::pow(Eta,-0.5*(Tau));
    Masses[species_index_u][1]=std::pow(Eta,0.5*Tau);
    OneOverSqrtM[species_index_u][0]=1.0/std::sqrt(Masses[species_index_u][0]);
    OneOverSqrtM[species_index_u][1]=1.0/std::sqrt(Masses[species_index_u][1]);
  }
  if (species_index_d<species_set_size){
    int Tau=-1;
    Tauvalues[species_index_d]=Tau;
    Masses[species_index_d][0]=std::pow(Eta,-0.5*(Tau));
    Masses[species_index_d][1]=std::pow(Eta,0.5*Tau);
    OneOverSqrtM[species_index_d][0]=1.0/std::sqrt(Masses[species_index_d][0]);
    OneOverSqrtM[species_index_d][1]=1.0/std::sqrt(Masses[species_index_d][1]);
  }
  if (species_index_ut<species_set_size){
    int Tau=1;
    Tauvalues[species_index_ut]=Tau;
    Masses[species_index_ut][0]=std::pow(Eta,-0.5*(Tau));
    Masses[species_index_ut][1]=std::pow(Eta,0.5*Tau);
    OneOverSqrtM[species_index_ut][0]=1.0/std::sqrt(Masses[species_index_ut][0]);
    OneOverSqrtM[species_index_ut][1]=1.0/std::sqrt(Masses[species_index_ut][1]);
  }
  if (species_index_dt<species_set_size){
    int Tau=1;
    Tauvalues[species_index_dt]=Tau;
    Masses[species_index_dt][0]=std::pow(Eta,-0.5*(Tau));
    Masses[species_index_dt][1]=std::pow(Eta,0.5*Tau);
    OneOverSqrtM[species_index_dt][0]=1.0/std::sqrt(Masses[species_index_dt][0]);
    OneOverSqrtM[species_index_dt][1]=1.0/std::sqrt(Masses[species_index_dt][1]);
  }
}

template<typename FT>
void J2OrbitalSoA<FT>::addFunc(int ia, int ib, std::unique_ptr<FT> j)
{
  // AVFLAG
  assert(ia < NumGroups);
  assert(ib < NumGroups);
  if (ia == ib)
  {
    if (ia == 0) //first time, assign everything
    {
      int ij = 0;
      for (int ig = 0; ig < NumGroups; ++ig)
        for (int jg = 0; jg < NumGroups; ++jg, ++ij)
          if (F[ij] == nullptr)
            F[ij] = j.get();   //uu default set for everything?
    }
    else{
      if (separate){
         F[ia * NumGroups + ib] = j.get();
      }
      else{
        int ij = 0;
        for (int ig = 0; ig < NumGroups; ++ig){
          for (int jg = 0; jg < NumGroups; ++jg, ++ij){
              if ((ig==jg))  // && (Tauvalues[ia]+Tauvalues[ib]==Tauvalues[ig]+Tauvalues[jg]))
                 F[ij] = j.get();   
          }
        }
      }
    }
  }
  else
  {
    // a very special case, 1 particle of each type (e.g. 1 up + 1 down)
    // uu/dd/etc. was prevented by the builder
    if (N == NumGroups)
      for (int ig = 0; ig < NumGroups; ++ig)
        F[ig * NumGroups + ig] = j.get();
    // generic case
    if (separate){
       F[ia * NumGroups + ib] = j.get();  //ud  (uut etc...)
       F[ib * NumGroups + ia] = j.get();  //du  (utu etc...)
    }
    else{
      int ij=0;
      for (int ig = 0; ig < NumGroups; ++ig){
        for (int jg = 0; jg < NumGroups; ++jg, ++ij){
          if ((ig!=jg) &&  (Tauvalues[ia]*Tauvalues[ib]==Tauvalues[ig]*Tauvalues[jg]))  //(Tauvalues[ia]+Tauvalues[ib]==Tauvalues[ig]+Tauvalues[jg]))
            F[ij] = j.get();   
        }
      }
    }
  }
  std::stringstream aname;
  aname << ia << ib;
  J2Unique[aname.str()] = std::move(j);
}

template<typename FT>
std::unique_ptr<WaveFunctionComponent> J2OrbitalSoA<FT>::makeClone(ParticleSet& tqp) const
{
  auto j2copy = std::make_unique<J2OrbitalSoA<FT>>(myName, tqp);
  if (dPsi)
    j2copy->dPsi = dPsi->makeClone(tqp);
  std::map<const FT*, FT*> fcmap;
  for (int ig = 0; ig < NumGroups; ++ig)
    for (int jg = ig; jg < NumGroups; ++jg)
    {
      int ij = ig * NumGroups + jg;
      if (F[ij] == 0)
        continue;
      typename std::map<const FT*, FT*>::iterator fit = fcmap.find(F[ij]);
      if (fit == fcmap.end())
      {
        auto fc      = std::make_unique<FT>(*F[ij]);
        fcmap[F[ij]] = fc.get();
        j2copy->addFunc(ig, jg, std::move(fc));
      }
    }
  j2copy->KEcorr      = KEcorr;
  j2copy->Optimizable = Optimizable;
  return j2copy;
}

/** intenal function to compute \f$\sum_j u(r_j), du/dr, d2u/dr2\f$
 * @param P particleset
 * @param iat particle index
 * @param dist starting distance
 * @param u starting value
 * @param du starting first deriv
 * @param d2u starting second deriv
 */
template<typename FT>
void J2OrbitalSoA<FT>::computeU3(const ParticleSet& P,
                                 int iat,
                                 const DistRow& dist,
                                 RealType* restrict u,
                                 RealType* restrict du,
                                 RealType* restrict d2u,
                                 bool triangle)
{
  const int jelmax = triangle ? iat : N;  //AV: if triangle, jelmax=iat. Else, jelmax =N (number of particles) 
  constexpr valT czero(0);
  std::fill_n(u, jelmax, czero);
  std::fill_n(du, jelmax, czero);
  std::fill_n(d2u, jelmax, czero);

  const int igt = P.GroupID[iat] * NumGroups;
  for (int jg = 0; jg < NumGroups; ++jg)
  {
    const FuncType& f2(*F[igt + jg]);
    int iStart = P.first(jg);
    int iEnd   = std::min(jelmax, P.last(jg));
    //AV evaluate separately for different groups (->BsplineFunctor.h)
    f2.evaluateVGL(iat, iStart, iEnd, dist.data(), u, du, d2u, DistCompressed.data(), DistIndice.data());
    
    //---test
    //double a,b;
    //f2.evaluate(0.5, a, b);
    //std::cout<<F[igt + jg]<<std::endl;
    //---

  }
  //u[iat]=czero;
  //du[iat]=czero;
  //d2u[iat]=czero;
}


/** intenal function to compute \f$\sum_j u(r_j), du/dr, d2u/dr2\f$
 * @param P particleset
 * @param iat particle index
 * @param dist starting distance
 * @param u starting value
 * @param du starting first deriv
 * @param d2u starting second deriv
 */
template<typename FT>
void J2OrbitalSoA<FT>::computeU3(const ParticleSet& P,
                                 int iat,
                                 int numpar,
                                 const DistRow& dist,
                                 const DisplRow& displ,
                                 RealType* restrict u,
                                 RealType* restrict dudx,
                                 RealType* restrict dudy,
                                 RealType* restrict d2udx2,
                                 RealType* restrict d2udy2,
                                 bool triangle)
{
  const int jelmax = triangle ? iat : N;  //AV: if triangle, jelmax=iat. Else, jelmax =N (number of particles) 
  constexpr valT czero(0);
  std::fill_n(u, jelmax, czero);
  std::fill_n(dudx, jelmax, czero);
  std::fill_n(dudy, jelmax, czero);

  std::fill_n(d2udx2, jelmax, czero);
  std::fill_n(d2udy2, jelmax, czero);

  const int igt = P.GroupID[iat] * NumGroups;
  const int ig  = P.GroupID[iat];
  for (int jg = 0; jg < NumGroups; ++jg)
  {
    const FuncType& f2(*F[igt + jg]);
    int iStart = P.first(jg);
    int iEnd   = std::min(jelmax, P.last(jg));
    //AV evaluate separately for different groups (->BsplineFunctor.h)
    double etavar=0;
    f2.evaluateVGL2(iat, iStart, iEnd, numpar, dist.data(), displ.data(0), displ.data(1), u, dudx, dudy, d2udx2, d2udy2, DistCompressed.data(), DisplCompressedX.data(), DisplCompressedY.data(), DistIndice.data(), Tauvalues[ig]+Tauvalues[jg]);
    etavars[igt+jg]=etavar;
  }
  //if (numpar==2)
  //  std::cout<<"leave compute U3, curUat: "<<u<<" Uat: "<<Uat[iat]<<std::endl<<std::endl;
  //u[iat]=czero;
  //du[iat]=czero;
  //d2u[iat]=czero;
}


template<typename FT>
typename J2OrbitalSoA<FT>::PsiValueType J2OrbitalSoA<FT>::ratio(ParticleSet& P, int iat)
{
  //std::cout<<"AV J2OrbitalSoA<FT>::ratio"<<std::endl;
  //only ratio, ready to compute it again
  UpdateMode = ORB_PBYP_RATIO;
  //std::cout<<"AV in J2OrbitalSoA::ratio, Uat[iat]: "<<Uat[iat]<<" iat: "<<iat<<std::endl;
  cur_Uat    = computeU(P, iat, P.getDistTableAA(my_table_ID_).getTempDists())
               + computeU(P, iat, 0, P.getDistTableAA(my_table_ID_).getTempDists(), P.getDistTableAA(my_table_ID_).getTempDispls()); //current_Uat, iat: particle index that is moved
                                                                                //new distribution-> P.getDistTableAA?
                                                                                //const PosType& r = P.activeR(iat);
  //std::cout<<"cur_Uat: "<<cur_Uat<<" Uat[iat]: "<<Uat[iat]<<" Jastrow ratio:"<<std::exp(static_cast<PsiValueType>(Uat[iat] - cur_Uat)) <<std::endl;
  return std::exp(static_cast<PsiValueType>(Uat[iat] - cur_Uat));
}

template<typename FT>
void J2OrbitalSoA<FT>::evaluateRatiosAlltoOne(ParticleSet& P, std::vector<ValueType>& ratios)
{
  //std::cout<<"AV in J2OrbitalSoA::evaluateRatiosAlltoOne, need to check function!"<<std::endl;
  //std::flush(std::cout);
  //abort();

  const auto& d_table = P.getDistTableAA(my_table_ID_);
  const auto& dist    = d_table.getTempDists();
  const auto& displ   = d_table.getTempDispls();

  for (int ig = 0; ig < NumGroups; ++ig)
  {
    const int igt = ig * NumGroups;
    valT sumU(0);
    for (int jg = 0; jg < NumGroups; ++jg)
    {
      const FuncType& f2(*F[igt + jg]);
      int iStart = P.first(jg);
      int iEnd   = P.last(jg);
      sumU += f2.evaluateV(-1, iStart, iEnd, dist.data(), DistCompressed.data())
            + f2.evaluateV2(-1, iStart, iEnd, 0, dist.data(), displ.data(0), displ.data(1), DistCompressed.data(), DisplCompressedX.data(), DisplCompressedY.data(), Tauvalues[ig]+Tauvalues[jg]);
    //       + f2.evaluateV2(-1, iStart, iEnd, 0, dist.data(), displ.data(0), displ.data(1), DistCompressed.data(), DisplCompressedX.data(), DisplCompressedY.data(), Tauvalues[ig]+Tauvalues[jg])
    //        + f2.evaluateV2(-1, iStart, iEnd, 1, dist.data(), displ.data(0), displ.data(1), DistCompressed.data(), DisplCompressedX.data(), DisplCompressedY.data(), Tauvalues[ig]+Tauvalues[jg]);
    }

    for (int i = P.first(ig); i < P.last(ig); ++i)
    {
      //r=dist[i];
      // remove self-interaction
      const valT Uself = F[igt + ig]->evaluate(dist[i])
              + F[igt + ig]->evaluate(dist[i],displ[i][0],displ[i][1],0,Tauvalues[ig]+Tauvalues[ig]); //displ[i][0]*displ[i][0], displ[i][1]*displ[i][1],0)
              //+ F[igt + ig]->evaluate(dist[i],0.0,0.0,0); //displ[i][0]*displ[i][0], displ[i][1]*displ[i][1],1);  //maybe reverse [i][0]->[0][1]
      ratios[i]        = std::exp(Uat[i] + Uself - sumU);
    }
  }
}

template<typename FT>
typename J2OrbitalSoA<FT>::GradType J2OrbitalSoA<FT>::evalGrad(ParticleSet& P, int iat)
{
  return GradType(dUat[iat]);
}

template<typename FT>
typename J2OrbitalSoA<FT>::PsiValueType J2OrbitalSoA<FT>::ratioGrad(ParticleSet& P, int iat, GradType& grad_iat)
{ //modify for grad!!
  std::cout<<"Ratio not completely implemented for anisotropy (x,y), as well as Grad for anisotropy"<<std::endl;
  std::flush(std::cout);
  abort();
  
  UpdateMode = ORB_PBYP_PARTIAL;

  computeU3(P, iat, P.getDistTableAA(my_table_ID_).getTempDists(), cur_u.data(), cur_du.data(), cur_d2u.data());
  computeU3(P, iat, 0, P.getDistTableAA(my_table_ID_).getTempDists(), P.getDistTableAA(my_table_ID_).getTempDispls(), cur_v.data(), cur_dvdx.data(), cur_dvdy.data(),cur_d2vdx2.data(), cur_d2vdy2.data());
  cur_Uat = simd::accumulate_n(cur_u.data(), N, valT()) + simd::accumulate_n(cur_v.data(), N, valT());
  DiffVal = Uat[iat] - cur_Uat;
  grad_iat += accumulateG(cur_du.data(), P.getDistTableAA(my_table_ID_).getTempDispls());
  return std::exp(static_cast<PsiValueType>(DiffVal));
}

template<typename FT>
void J2OrbitalSoA<FT>::acceptMove(ParticleSet& P, int iat, bool safe_to_delay)
{
  const int ig_iat=P.GroupID[iat];

  //std::cout<<"AV entering J2OrbitalSoA<FT>::acceptMove"<<std::endl;
  // get the old u, du, d2u
  const auto& d_table = P.getDistTableAA(my_table_ID_);

  //du: 1/r*df/dr (term that appears in laplace with spherical coordinates)
  //std::cout<<"Compute old U3"<<std::endl;
  computeU3(P, iat, d_table.getOldDists(), old_u.data(), old_du.data(), old_d2u.data());
  computeU3(P, iat, 0, d_table.getOldDists(), d_table.getOldDispls(), old_v.data(), old_dvdx.data(), old_dvdy.data(), old_d2vdx2.data(), old_d2vdy2.data());
  
  if (UpdateMode == ORB_PBYP_RATIO)
  { //ratio-only during the move; need to compute derivatives
    const auto& dist = d_table.getTempDists();
    const auto& displ = d_table.getTempDispls();
    //std::cout<<"compute new U3"<<std::endl;
    computeU3(P, iat, dist, cur_u.data(), cur_du.data(), cur_d2u.data());
    computeU3(P, iat, 0, dist, displ, cur_v.data(), cur_dvdx.data(), cur_dvdy.data(),cur_d2vdx2.data(), cur_d2vdy2.data());
    
    
  }

  valT cur_d2Uat(0);
  const auto& new_dr    = d_table.getTempDispls();
  const auto& old_dr    = d_table.getOldDispls();
  constexpr valT lapfac = OHMMS_DIM - RealType(1); //set OHMMS_DIM???
  //std::cout<<"AV in J2OrbitalSoA<FT>::acceptMove, lapfac: "<<lapfac<<" d2Uat: "<< d2Uat[0]<< " "<<old_d2u[0] + lapfac * old_du[0] <<" new: "<< cur_d2u[0] + lapfac * cur_du[0]<<std::endl;
  //for (int eta_it=0; eta_it<etavars.size(); eta_it++){
  //  std::cout<<" etavars["<<eta_it<<"]: "<<etavars[eta_it]<<std::endl;
  //}

#pragma omp simd reduction(+ : cur_d2Uat)
  for (int jat = 0; jat < N; jat++)
  { 
    //std::cout<<"cur_ux2["<<jat<<"]: "<<cur_ux2[jat]<<std::endl<<std::endl;
    const int ig_jat=P.GroupID[jat]; //AV
    //valT newl_anisotropic(0);
    //std::cout<<"jat: "<<jat<<" etavars: " <<etavars[ig_iat*NumGroups+ig_jat]<<std::endl;
    double etaVar=etavars[ig_iat*NumGroups+ig_jat];
    valT  dXjat_new = new_dr.data(0)[jat];
    valT  dYjat_new = new_dr.data(1)[jat];
    valT rsquared_new=dXjat_new*dXjat_new+dYjat_new*dYjat_new;
    
    if (rsquared_new<1e-15){
      rsquared_new+=1e-15;
    }
   

    const valT newl_anisotropic_jat=(dXjat_new*dXjat_new/(Masses[ig_jat][0]*rsquared_new)+dYjat_new*dYjat_new/(Masses[ig_jat][1]*rsquared_new))*(-cur_du[jat]+cur_d2u[jat])+cur_du[jat]*(1.0/Masses[ig_jat][0]+1.0/Masses[ig_jat][1])
    +(1.0/(Masses[ig_jat][0])*cur_d2vdx2[jat]+1.0/(Masses[ig_jat][1])*cur_d2vdy2[jat]);

    const valT newl_anisotropic_iat=(dXjat_new*dXjat_new/(Masses[ig_iat][0]*rsquared_new)+dYjat_new*dYjat_new/(Masses[ig_iat][1]*rsquared_new))*(-cur_du[jat]+cur_d2u[jat])+cur_du[jat]*(1.0/Masses[ig_iat][0]+1.0/Masses[ig_iat][1])
    +(1.0/(Masses[ig_iat][0])*cur_d2vdx2[jat]+1.0/(Masses[ig_iat][1])*cur_d2vdy2[jat]);

    valT  dXjat_old = old_dr.data(0)[jat];
    valT  dYjat_old = old_dr.data(1)[jat];
    valT rsquared_old=dXjat_old*dXjat_old+dYjat_old*dYjat_old;
   
    if (rsquared_old<1e-15){
      rsquared_old+=1e-15;
    }
  

    const valT oldl_anisotropic_jat=(dXjat_old*dXjat_old/(Masses[ig_jat][0]*rsquared_old)+dYjat_old*dYjat_old/(Masses[ig_jat][1]*rsquared_old))*(-old_du[jat]+old_d2u[jat])+old_du[jat]*(1.0/Masses[ig_jat][0]+1.0/Masses[ig_jat][1])
    +(1.0/(Masses[ig_jat][0])*old_d2vdx2[jat]+1.0/(Masses[ig_jat][1])*old_d2vdy2[jat]);

    const valT oldl_anisotropic_iat=(dXjat_old*dXjat_old/(Masses[ig_iat][0]*rsquared_old)+dYjat_old*dYjat_old/(Masses[ig_iat][1]*rsquared_old))*(-old_du[jat]+old_d2u[jat])+old_du[jat]*(1.0/Masses[ig_iat][0]+1.0/Masses[ig_iat][1])
    +(1.0/(Masses[ig_iat][0])*old_d2vdx2[jat]+1.0/(Masses[ig_iat][1])*old_d2vdy2[jat]);

    const valT dl_anisotropic_jat=oldl_anisotropic_jat- newl_anisotropic_jat;

    const valT du   = cur_u[jat] +cur_v[jat]- (old_u[jat]+old_v[jat]);

    //const valT newl_isotropic_onlyToCompare = cur_d2u[jat] + lapfac * cur_du[jat];
    //const valT oldl_isotropic_onlyToCompare = old_d2u[jat] + lapfac * old_du[jat];
    //const valT dl_isotropic_onlyToCompare   = oldl_isotropic_onlyToCompare - newl_isotropic_onlyToCompare;  //old-new because sum over different particles!!! Old cancels only old contribution of that particle, not the others. Minus sign because e^(-J) (probably) (?)
    Uat[jat] += du;  //AV: what's the purpose? ->different particles!
    d2Uat[jat] += dl_anisotropic_jat; // dl; //AV: what's the purpose?
    cur_d2Uat -= newl_anisotropic_iat; //newl;

    //std::cout<<"AV in J2OrbitalSoA::acceptMove, newl_isotropic_onlyToCompare: "<<newl_isotropic_onlyToCompare<<" newl_anisotropic_iat: "<<newl_anisotropic_iat<< " newl_anisotropic_jat: "<<newl_anisotropic_jat<<std::endl;
    //std::cout<<"AV in J2OrbitalSoA::acceptMove, dl_isotropic_onlyToCompare: "<<dl_isotropic_onlyToCompare<<" dl_anisotropic_jat: "<<dl_anisotropic_jat<<std::endl;
    //if (std::abs(dl_anisotropic_jat- dl_isotropic_onlyToCompare )>1e-7){
    //      std::cout<<"lap test new and old differ!"<<std::endl;
    //      std::flush(std::cout);
    //      abort();
    //    }
  }
  //std::cout<<"AV in J2OrbitalSoA<FT>::acceptMove, lapfac: "<<lapfac<<" d2Uat: "<< d2Uat[0]<<std::endl;
  posT cur_dUat;
  for (int idim = 0; idim < OHMMS_DIM; ++idim)
  {
    const valT* restrict new_dX    = new_dr.data(idim);
    const valT* restrict old_dX    = old_dr.data(idim);
    const valT* restrict cur_du_pt = cur_du.data();
    const valT* restrict cur_dvdx_pt = cur_dvdx.data();
    const valT* restrict cur_dvdy_pt = cur_dvdy.data();

    const valT* restrict old_du_pt = old_du.data();
    const valT* restrict old_dvdx_pt = old_dvdx.data();
    const valT* restrict old_dvdy_pt = old_dvdy.data();

    valT* restrict save_g          = dUat.data(idim);
    valT cur_g                     = cur_dUat[idim];
    
      //AV maybe add factora2/b2 to simd reduction?
#pragma omp simd reduction(+ : cur_g) aligned(old_dX, new_dX, save_g, cur_du_pt, cur_dvdx_pt, cur_dvdy_pt, old_du_pt, old_dvdx_pt, old_dvdy_pt : QMC_SIMD_ALIGNMENT)
    for (int jat = 0; jat < N; jat++)
    {
     
      const int ig_jat=P.GroupID[jat]; //AV
      valT cur_dv=0.0;
      valT old_dv=0.0;

      if (idim==0){
        cur_dv= cur_dvdx_pt[jat];
        old_dv= old_dvdx_pt[jat];

      }
      else if (idim==1){
        cur_dv= cur_dvdy_pt[jat];
        old_dv= old_dvdy_pt[jat];

      }
      const valT newg = cur_du_pt[jat] * new_dX[jat]+cur_dv;  //(d-1)/r*(df/dr)*x=df/dx
      const valT dg   = newg - (old_du_pt[jat] * old_dX[jat]  +old_dv);
      save_g[jat] -= dg*OneOverSqrtM[ig_jat][idim];  //set dUat
      cur_g += newg*OneOverSqrtM[ig_iat][idim];
    }
    cur_dUat[idim] = cur_g;
    //std::cout<<"AV in J2OrbitalSoA<FT>::acceptMove, OHMMS_DIM: "<< OHMMS_DIM<<" idim: "<<idim<<std::endl;
  }
  log_value_ += Uat[iat] - cur_Uat;
  Uat[iat]   = cur_Uat;
  dUat(iat)  = cur_dUat;
  //std::cout<<"AV in J2OrbitalSoA<FT>::acceptMove, cur_dUat.size(): "<<cur_dUat.size()<<std::endl;
  d2Uat[iat] = cur_d2Uat;
}

template<typename FT>
void J2OrbitalSoA<FT>::recompute(const ParticleSet& P)
{
 
  //for(int i=0; i<NumGroups; i++){
  //  std::cout<<"AV in J2OrbitalSoA::recompute, tau: "<<Tauvalues[i]<<std::endl;
  //  std::cout<<Masses[i][0]<<" "<<Masses[i][1]<<std::endl;
  //  std::cout<<OneOverSqrtM[i][0]<<" "<<OneOverSqrtM[i][0]<<std::endl; 
  //}
  //std::cout<<std::endl;

  const auto& d_table = P.getDistTableAA(my_table_ID_);
  for (int ig = 0; ig < NumGroups; ++ig)
  {
    for (int iat = P.first(ig), last = P.last(ig); iat < last; ++iat)
    {
      computeU3(P, iat, d_table.getDistRow(iat), cur_u.data(), cur_du.data(), cur_d2u.data(), true);  //AV compute Jastrow factor for the first time here (?)
      computeU3(P, iat, 0, d_table.getDistRow(iat), d_table.getDisplRow(iat), cur_v.data(), cur_dvdx.data(), cur_dvdy.data(),cur_d2vdx2.data(), cur_d2vdy2.data(),true);  //AV compute Jastrow factor for the first time here (?)
      
      //AV vary

      //std::cout<<"AV in J2OrbitalSoA::recompute, accumulate_n cur_u: "<< simd::accumulate_n(cur_u.data(), iat, valT())<< "cur_ua2: "<<simd::accumulate_n(cur_ua2.data(), iat, valT())<< "cur_ub2: "<<simd::accumulate_n(cur_ub2.data(), iat, valT());
      Uat[iat] = simd::accumulate_n(cur_u.data(), iat, valT())+simd::accumulate_n(cur_v.data(), iat, valT());


      posT grad;
      valT lap(0);
      const valT* restrict u   = cur_u.data();
      const valT* restrict v   = cur_v.data();

      const valT* restrict du  = cur_du.data();
      const valT* restrict dvdx  = cur_dvdx.data();
      const valT* restrict dvdy  = cur_dvdy.data();

      const valT* restrict d2u = cur_d2u.data();
      const valT* restrict d2vdx2 = cur_d2vdx2.data();
      const valT* restrict d2vdy2 = cur_d2vdy2.data();


      const auto& displ        = d_table.getDisplRow(iat);
      constexpr valT lapfac    = OHMMS_DIM - RealType(1);
#pragma omp simd reduction(+ : lap) aligned(du, dvdx, dvdy, d2u, d2vdx2, d2vdy2: QMC_SIMD_ALIGNMENT)
      for (int jat = 0; jat < iat; ++jat){
        //AV test--------
        const int ig_jat=P.GroupID[jat];

        valT lap_test_old(0);
        lap_test_old+=d2u[jat] + lapfac * du[jat];
        valT lap_test_new(0);
        valT  dXjat = displ.data(0)[jat];
        valT  dYjat = displ.data(1)[jat];
        //std::cout<<"AV in J2OrbitalSoA::recompute, dXjat: "<<dXjat<< " dYjat: "<<dYjat<<std::endl;
        valT rsquared=dXjat*dXjat+dYjat*dYjat;

        if (rsquared<1e-15){
          rsquared+=1e-15;
        }
       
       
        //double mx=1.0;
        //double my=1.0;
        //std::cout<<"lap_test_new: "<<lap_test_new;
        lap_test_new+=(dXjat*dXjat/(Masses[ig][0]*rsquared)+dYjat*dYjat/(Masses[ig][1]*rsquared))*(-du[jat]+d2u[jat])+du[jat]*(1.0/Masses[ig][0]+1.0/Masses[ig][1]);  //what tau is iat?? Look up table or something, masses depend on it
        //std::cout<<" etaVar/Masses[ig][0]: "<<etaVar/Masses[ig][0]<<" " <<(1/etaVar)*1.0/Masses[ig][1]<<std::endl;
        
        lap_test_new+=(1.0/(Masses[ig][0])*d2vdx2[jat]+1.0/(Masses[ig][1])*d2vdy2[jat]);
        //valT lap_testtest(0);
        //lap_testtest+= (etaVar*dXjat*dXjat/(Masses[ig][0]*rsquared)+(1.0/(etaVar))*dYjat*dYjat/(Masses[ig][1]*rsquared))*(-du[jat]+d2u[jat])+du[jat]*(etaVar/Masses[ig][0]+(1.0/etaVar)*1.0/Masses[ig][1]);  //what tau is iat?? Look up table or something, masses depend on it

        //std::cout<<"AV in J2OrbitalSoA::recompute, lap_test_new: "<<lap_test_new<<std::endl;
        //if (std::abs(lap_test_new-lap_test_old)>1e-7){
        //  std::cout<<"lap test new and old differ!"<<std::endl;
        //  std::flush(std::cout);
        //  abort();
        //}
        //---------------
        lap += lap_test_new; //d2u[jat] + lapfac * du[jat];
      }
      for (int idim = 0; idim < OHMMS_DIM; ++idim)
      {
        const valT* restrict dX = displ.data(idim);
        valT s                  = valT();
#pragma omp simd reduction(+ : s) aligned(du, dX : QMC_SIMD_ALIGNMENT)
        for (int jat = 0; jat < iat; ++jat)
          s += du[jat] * dX[jat]*OneOverSqrtM[ig][idim]; //AV needs to be changed: what tau is iat?? Look up table or something, masses depend on it!!
        
        if (idim==0){
#pragma omp simd reduction(+ : s) aligned (dvdx : QMC_SIMD_ALIGNMENT)
          for (int jat = 0; jat < iat; jat++){
            const int ig_jat=P.GroupID[jat];
         
            s += dvdx[jat] * OneOverSqrtM[ig][idim];
          }
        }
        else if (idim==1){
#pragma omp simd reduction(+ : s) aligned (dvdy : QMC_SIMD_ALIGNMENT)
          for (int jat = 0; jat < iat; jat++){
            const int ig_jat=P.GroupID[jat];
           
            s += dvdy[jat] *OneOverSqrtM[ig][idim];

          }
        }

        grad[idim] = s;
      }
      dUat(iat)  = grad;
      d2Uat[iat] = -lap;
// add the contribution from the upper triangle
#pragma omp simd aligned(u, v, du, dvdx, dvdy, d2u, d2vdx2, d2vdy2 : QMC_SIMD_ALIGNMENT)
      for (int jat = 0; jat < iat; jat++)
      {
        const int ig_jat=P.GroupID[jat];

        Uat[jat] += (u[jat]+v[jat]);
        //AV test--------
        valT lap_test_old(0);
        lap_test_old=d2u[jat] + lapfac * du[jat];
        valT lap_test_new(0);
        valT  dXjat = displ.data(0)[jat];
        valT  dYjat = displ.data(1)[jat];
        //std::cout<<"AV in J2OrbitalSoA::recompute, upper triangle, dXjat: "<<dXjat<< " dYjat: "<<dYjat<<std::endl;
        valT rsquared=dXjat*dXjat+dYjat*dYjat;
 

        if (rsquared<1e-15){
          rsquared+=1e-15;
        }
        
        //double mx=1.0;
        //double my=1.0;
        lap_test_new+=(dXjat*dXjat/(Masses[ig_jat][0]*rsquared)+dYjat*dYjat/(Masses[ig_jat][1]*rsquared))*(-du[jat]+d2u[jat])+du[jat]*(1.0/Masses[ig_jat][0]+1.0/Masses[ig_jat][1]);
       
        lap_test_new+=(1.0/(Masses[ig_jat][0])*d2vdx2[jat]+1.0/(Masses[ig_jat][1])*d2vdy2[jat]);


        //std::cout<<"AV in J2OrbitalSoA::recompute, upper triangle, lap_test_old: "<<lap_test_old<<" lap_test_new: "<<lap_test_new<<std::endl;
        //if (std::abs(lap_test_new-lap_test_old)>1e-7){
        //  std::cout<<"In upper triangle, lap test new and old differ!"<<std::endl;
        //  std::flush(std::cout);
        //  abort();
        //}
        d2Uat[jat] -= lap_test_new; //d2u[jat] + lapfac * du[jat];
        //-------------
        //d2Uat[jat] -= d2u[jat] + lapfac * du[jat];
      }
      for (int idim = 0; idim < OHMMS_DIM; ++idim)
      {
        valT* restrict save_g   = dUat.data(idim);
        const valT* restrict dX = displ.data(idim);
#pragma omp simd aligned(save_g, du, dX : QMC_SIMD_ALIGNMENT)
        for (int jat = 0; jat < iat; jat++){
          const int ig_jat=P.GroupID[jat];
          save_g[jat] -= du[jat] * dX[jat] * OneOverSqrtM[ig_jat][idim]; //AV needs to be changed: what tau is iat?? Look up table or something, masses depend on it!!
        
        }
        if (idim==0){
#pragma omp simd aligned(save_g, dvdx : QMC_SIMD_ALIGNMENT)
          for (int jat = 0; jat < iat; jat++){
            const int ig_jat=P.GroupID[jat];
            save_g[jat] -= dvdx[jat] * OneOverSqrtM[ig_jat][idim];
          }
        }
        else if (idim==1){
#pragma omp simd aligned(save_g, dvdy : QMC_SIMD_ALIGNMENT)
          for (int jat = 0; jat < iat; jat++){
            const int ig_jat=P.GroupID[jat];
            save_g[jat] -= dvdy[jat] * OneOverSqrtM[ig_jat][idim];
          }
        }
      }
    }
  }
}

template<typename FT>
typename J2OrbitalSoA<FT>::LogValueType J2OrbitalSoA<FT>::evaluateLog(const ParticleSet& P,
                                                                      ParticleSet::ParticleGradient_t& G,
                                                                      ParticleSet::ParticleLaplacian_t& L)
{
  return evaluateGL(P, G, L, true);
}

template<typename FT>
WaveFunctionComponent::LogValueType J2OrbitalSoA<FT>::evaluateGL(const ParticleSet& P,
                                                                 ParticleSet::ParticleGradient_t& G,
                                                                 ParticleSet::ParticleLaplacian_t& L,
                                                                 bool fromscratch)
{
  if (fromscratch)
    recompute(P);
  log_value_ = valT(0);
  for (int iat = 0; iat < N; ++iat)
  {
    log_value_ += Uat[iat];
    G[iat] += dUat[iat];
    L[iat] += d2Uat[iat];
    //std::cout<<std::endl;
    //std::cout<<"AV in J2OrbitalSoA::evaluateGL, iat: "<<iat<<" G: "<<G[iat]<<" L: "<<L[iat]<<std::endl;
    //std::cout<<std::endl;
  }

  return log_value_ = -log_value_ * 0.5;
}

template<typename FT>
void J2OrbitalSoA<FT>::evaluateHessian(ParticleSet& P, HessVector_t& grad_grad_psi)
{
  log_value_ = 0.0;
  const auto& d_ee(P.getDistTableAA(my_table_ID_));
  valT dudr, d2udr2;

  Tensor<valT, DIM> ident;
  grad_grad_psi = 0.0;
  ident.diagonal(1.0);

  for (int i = 1; i < N; ++i)
  {
    const auto& dist  = d_ee.getDistRow(i);
    const auto& displ = d_ee.getDisplRow(i);
    auto ig           = P.GroupID[i];
    const int igt     = ig * NumGroups;
    for (int j = 0; j < i; ++j)
    {
      auto r    = dist[j];
      auto rinv = 1.0 / r;
      auto dr   = displ[j];
      auto jg   = P.GroupID[j];
      auto uij  = F[igt + jg]->evaluate(r, dudr, d2udr2);
      log_value_ -= uij;
      auto hess = rinv * rinv * outerProduct(dr, dr) * (d2udr2 - dudr * rinv) + ident * dudr * rinv;
      grad_grad_psi[i] -= hess;
      grad_grad_psi[j] -= hess;
    }
  }
}

template class J2OrbitalSoA<BsplineFunctor<QMCTraits::RealType>>;
template class J2OrbitalSoA<PadeFunctor<QMCTraits::RealType>>;
template class J2OrbitalSoA<UserFunctor<QMCTraits::RealType>>;

} // namespace qmcplusplus
