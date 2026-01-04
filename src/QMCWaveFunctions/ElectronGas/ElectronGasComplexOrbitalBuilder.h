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


#ifndef QMCPLUSPLUS_ELECTRONGAS_COMPLEXORBITALS_H
#define QMCPLUSPLUS_ELECTRONGAS_COMPLEXORBITALS_H

#include "QMCWaveFunctions/WaveFunctionComponentBuilder.h"
#include "QMCWaveFunctions/SPOSet.h"
#include "QMCWaveFunctions/SPOSetBuilder.h"
#include "QMCWaveFunctions/ElectronGas/HEGGrid.h"
#include "CPU/math.hpp"


namespace qmcplusplus
{
struct EGOSet : public SPOSet
{
  int KptMax;
  std::vector<PosType> K;
  std::vector<PosType> Kxy; //AV (1/sqrt(mx)*kx, 1/sqrt(my)*ky)
  std::vector<RealType> mK2;
  std::vector<RealType> mK2xy; //AV (1/mx * kx^2 + 1/my * ky^2)

  //EGOSet(const std::vector<PosType>& k, const std::vector<RealType>& k2);
  EGOSet(const std::vector<PosType>& k, const std::vector<PosType>& kxy, const std::vector<RealType>& k2, const std::vector<RealType>& k2xy);
  EGOSet(const std::vector<PosType>& k, const std::vector<RealType>& k2, const std::vector<int>& d);

  std::unique_ptr<SPOSet> makeClone() const override { return std::make_unique<EGOSet>(*this); }

  void resetParameters(const opt_variables_type& optVariables) override {}
  void setOrbitalSetSize(int norbs) override {}

  inline void evaluateValue(const ParticleSet& P, int iat, ValueVector_t& psi) override
  {
    const PosType& r = P.activeR(iat);
    RealType sinkr, coskr;
    for (int ik = 0; ik < KptMax; ik++)
    {
      qmcplusplus::sincos(dot(K[ik], r), &sinkr, &coskr);
      psi[ik] = ValueType(coskr, sinkr);
      //std::cout<<"psi["<<ik<<"]="<<psi[ik]<<std::endl;

    }
  }

  /** generic inline function to handle a row
   * @param P current ParticleSet
   * @param iat active particle
   * @param psi value row
   * @param dpsi gradient row
   * @param d2psi laplacian row
   */
  inline void evaluateVGL(const ParticleSet& P,
                          int iat,
                          ValueVector_t& psi,
                          GradVector_t& dpsi,
                          ValueVector_t& d2psi) override
  {
    const PosType& r = P.activeR(iat);
    RealType sinkr, coskr;
    for (int ik = 0; ik < KptMax; ik++)  //change KptMax!!!!!
    {
      qmcplusplus::sincos(dot(K[ik], r), &sinkr, &coskr);
      psi[ik]   = ValueType(coskr, sinkr);
      //std::cout<<"AV in evaluateVGL, psi[ik]: "<<psi[ik]<<std::endl;
      dpsi[ik]  = ValueType(-sinkr, coskr) * Kxy[ik];  //AV: vector: put here different masses for directions (function is called separately for different spin species, can put it here already! Give valley index to function with default value. 2D: K 2D and kr 2D dot product
      //d2psi[ik] = ValueType(mK2[ik] * coskr, mK2[ik] * sinkr); //AV: also here: mK2 (=-k^2) 2D + different masses in different directions!!
      d2psi[ik] = ValueType(mK2xy[ik] * coskr, mK2xy[ik] * sinkr); //AV changed
    }
  }

  inline void evaluatevgh(const ParticleSet& P,
                          int iat,
                          ValueVector_t& psi,
                          GradVector_t& dpsi,
                          HessVector_t& dabpsi) 
  {
    const PosType& r = P.activeR(iat);
    RealType sinkr, coskr;
    for (int ik = 0; ik < KptMax; ik++)  //change KptMax!!!!!
    {
      qmcplusplus::sincos(dot(K[ik], r), &sinkr, &coskr);
      psi[ik]   = ValueType(coskr, sinkr);
      //std::cout<<"AV in evaluateVGL, psi[ik]: "<<psi[ik]<<std::endl;
      dpsi[ik]  = ValueType(-sinkr, coskr) * K[ik];  //AV: vector: put here different masses for directions (function is called separately for different spin species, can put it here already! Give valley index to function with default value. 2D: K 2D and kr 2D dot product
      //d2psi[ik] = ValueType(mK2[ik] * coskr, mK2[ik] * sinkr); //AV: also here: mK2 (=-k^2) 2D + different masses in different directions!!
      //std::cout<<"ik: "<<ik<<" K[ik]: "<<K[ik][0]<<" "<<K[ik][1]<<" e^ikr: "<<ValueType(coskr,sinkr)<<std::endl;
      for (int alpha=0; alpha<OHMMS_DIM; alpha++){
        for (int beta=0; beta<OHMMS_DIM; beta++){
          dabpsi[ik](alpha,beta)=-K[ik][alpha]*K[ik][beta]*ValueType(coskr,sinkr);
          //std::cout<<"alpha: "<<alpha<<" beta: "<<beta<<" dabpsi: "<<dabpsi[ik](alpha,beta)<<std::endl;
        }
      }
      //std::cout<<std::endl;

      //d2psi[ik] = ValueType(mK2xy[ik] * coskr, mK2xy[ik] * sinkr); //AV changed
    }
  }

  inline void evaluatevghgh(const ParticleSet& P,
                          int iat,
                          ValueVector_t& psi,
                          GradVector_t& dpsi,
                          HessVector_t& dabpsi,
                          GGGVector_t& dabcpsi) 
  {
    const PosType& r = P.activeR(iat);
    RealType sinkr, coskr;
    for (int ik = 0; ik < KptMax; ik++)  //change KptMax!!!!!
    {
      qmcplusplus::sincos(dot(K[ik], r), &sinkr, &coskr);
      psi[ik]   = ValueType(coskr, sinkr);
      //std::cout<<"AV in evaluateVGL, psi[ik]: "<<psi[ik]<<std::endl;
      dpsi[ik]  = ValueType(-sinkr, coskr) * K[ik];  //AV: vector: put here different masses for directions (function is called separately for different spin species, can put it here already! Give valley index to function with default value. 2D: K 2D and kr 2D dot product
      //d2psi[ik] = ValueType(mK2[ik] * coskr, mK2[ik] * sinkr); //AV: also here: mK2 (=-k^2) 2D + different masses in different directions!!
      //std::cout<<"ik: "<<ik<<" K[ik]: "<<K[ik][0]<<" "<<K[ik][1]<<" e^ikr: "<<ValueType(coskr,sinkr)<<std::endl;
      for (int alpha=0; alpha<OHMMS_DIM; alpha++){
        for (int beta=0; beta<OHMMS_DIM; beta++){
          dabpsi[ik](alpha,beta)=-K[ik][alpha]*K[ik][beta]*ValueType(coskr,sinkr);
          //std::cout<<"alpha: "<<alpha<<" beta: "<<beta<<" dabpsi: "<<dabpsi[ik](alpha,beta)<<std::endl;
          //std::cout<<"alpha: "<<alpha<<" beta: "<<beta<<" dabpsi[ik]: "<<dabpsi[ik]<<std::endl;

        }
      }
      for (int alpha=0; alpha<OHMMS_DIM; alpha++){
        for (int beta=0; beta<OHMMS_DIM; beta++){
          for (int gamma=0; gamma<OHMMS_DIM; gamma++){
             dabcpsi[ik][alpha](beta,gamma)=K[ik][alpha]*K[ik][beta]*K[ik][gamma]*ValueType(sinkr,-coskr);
             //std::cout<<"alpha: "<<alpha<<" beta: "<<beta<<" gamma: "<<gamma<<" dabcpsi: "<<dabcpsi[ik][alpha](beta,gamma)<<std::endl;
             //std::cout<<"alpha: "<<alpha<<" beta: "<<beta<<" gamma: "<<gamma<<" dabcpsi[alpha]: "<<dabcpsi[ik][alpha]<<std::endl;
             //std::cout<<"alpha: "<<alpha<<" beta: "<<beta<<" gamma: "<<gamma<<" dabcpsi: "<<dabcpsi[ik]<<std::endl;

          }
        }
      }

      //std::cout<<std::endl;

      //d2psi[ik] = ValueType(mK2xy[ik] * coskr, mK2xy[ik] * sinkr); //AV changed
    }
  }

  void evaluate_notranspose(const ParticleSet& P,
                            int first,
                            int last,
                            ValueMatrix_t& logdet,
                            GradMatrix_t& dlogdet,
                            ValueMatrix_t& d2logdet) override
  {
    for (int iat = first, i = 0; iat < last; ++iat, ++i)
    {
      //std::cout<<"AV in ElectronGasComplexOrbitalBuilder::evaluate_notranspose, iat: "<<iat<<" i: "<<i<<std::endl;
      ValueVector_t v(logdet[i], OrbitalSetSize);
      GradVector_t g(dlogdet[i], OrbitalSetSize);
      ValueVector_t l(d2logdet[i], OrbitalSetSize);
      evaluateVGL(P, iat, v, g, l);
    }
  }
void evaluate_notranspose(const ParticleSet& P,
                            int first,
                            int last,
                            ValueMatrix_t& logdet,
                            GradMatrix_t& dlogdet,
                            HessMatrix_t& grad_grad_logdet) override
  {
    for (int iat = first, i = 0; iat < last; ++iat, ++i)
    {
      //std::cout<<"AV in ElectronGasComplexOrbitalBuilder::evaluate_notranspose, iat: "<<iat<<" i: "<<i<<std::endl;
      ValueVector_t v(logdet[i], OrbitalSetSize);
      GradVector_t g(dlogdet[i], OrbitalSetSize);
      HessVector_t h(grad_grad_logdet[i], OrbitalSetSize);
      evaluatevgh(P, iat, v, g, h);  
      //APP_ABORT("AV: test (!!) specialization of EGOSet::evaluate_notranspose() for grad_grad_logdet. And then implement mass anisotropy in DiracDeterminantWithBackflow \n");

    }
  }

  void evaluate_notranspose(const ParticleSet& P,
                            int first,
                            int last,
                            ValueMatrix_t& logdet,
                            GradMatrix_t& dlogdet,
                            HessMatrix_t& grad_grad_logdet,
                            GGGMatrix_t& grad_grad_grad_logdet) override
  {
    for (int iat = first, i = 0; iat < last; ++iat, ++i)
    {
       ValueVector_t v(logdet[i], OrbitalSetSize);
       GradVector_t g(dlogdet[i], OrbitalSetSize);
       HessVector_t h(grad_grad_logdet[i], OrbitalSetSize);
       GGGVector_t gh(grad_grad_grad_logdet[i], OrbitalSetSize);

       evaluatevghgh(P, iat, v, g, h, gh);  

       //APP_ABORT(
       // "Incomplete implementation EGOSet::evaluate(P,first,last,lodget,dlodet,grad_grad_logdet,grad_grad_grad_logdet");
    }
  }
};


/** OrbitalBuilder for Slater determinants of electron-gas
*/
class ElectronGasComplexOrbitalBuilder : public WaveFunctionComponentBuilder
{
public:
  int nup; //5;   //AV change by hand, number up particles
  int ndn; //0; //2; //0;   //AV change by hand, number down particles
  int nuptau1; //18; //2;
  int ndntau1;
  ///constructor
  ElectronGasComplexOrbitalBuilder(Communicate* comm, ParticleSet& els);

  ///implement vritual function
  std::unique_ptr<WaveFunctionComponent> buildComponent(xmlNodePtr cur) override;
};

/** OrbitalBuilder for Slater determinants of electron-gas
*/
class ElectronGasSPOBuilder : public SPOSetBuilder
{
protected:
  bool has_twist;
  PosType unique_twist;
  HEGGrid<RealType> egGrid;
  //HEGGrid<RealType> egGrid1;
  //HEGGrid<RealType> egGrid2;
  //HEGGrid<RealType> egGrid3;

  xmlNodePtr spo_node;

public:
  std::vector<int> sizeSpecies;
  //int nup; //5;   //AV change by hand, number up particles
  //int ndn; //0; //2; //0;   //AV change by hand, number down particles
  //int nuptau1; //18; //2;
  //int ndntau1;
  ///constructor
  ElectronGasSPOBuilder(ParticleSet& p, Communicate* comm, xmlNodePtr cur);

  /** initialize the Antisymmetric wave function for electrons
  *@param cur the current xml node
  */
  std::unique_ptr<SPOSet> createSPOSetFromXML(xmlNodePtr cur, int particletype=0) override;
  std::unique_ptr<SPOSet> createSPOSetFromIndices(indices_t& indices);
};
} // namespace qmcplusplus
#endif
