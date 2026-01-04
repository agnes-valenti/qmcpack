//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Ken Esler, kpesler@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//                    Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_DIFFERENTIAL_TWOBODYJASTROW_H
#define QMCPLUSPLUS_DIFFERENTIAL_TWOBODYJASTROW_H
#include "Configuration.h"
#include "QMCWaveFunctions/DiffWaveFunctionComponent.h"
#include "Particle/DistanceTable.h"
#include "ParticleBase/ParticleAttribOps.h"
#include "Utilities/IteratorUtility.h"

namespace qmcplusplus
{
/** @ingroup WaveFunctionComponent
 *  @brief Specialization for two-body Jastrow function using multiple functors
 */
template<class FT>
class DiffTwoBodyJastrowOrbital : public DiffWaveFunctionComponent
{
  ///number of variables this object handles
  int NumVars;
  ///number of target particles
  int NumPtcls;
  ///number of groups, e.g., for the up/down electrons
  int NumGroups;
  ///variables handled by this orbital
  opt_variables_type myVars;
  ///container for the Jastrow functions  for all the pairs
  std::vector<FT*> F;
  /// e-e table ID
  const int my_table_ID_;
  /// Map indices from subcomponent variables to component variables
  std::vector<std::pair<int, int>> OffSet;
  Vector<RealType> dLogPsi;
  std::vector<GradVectorType*> gradLogPsi;
  std::vector<ValueVectorType*> lapLogPsi;
  std::map<std::string, std::unique_ptr<FT>> J2Unique;

  std::vector<int> Tauvalues;
  std::vector<std::vector<double> > Masses;
  std::vector<std::vector<double> > OneOverSqrtM;

  int Nparamsu;
  int Nv;
  //RealType etaVar;

  bool separate;

  bool AVSR; //if stochastic reconfiguration: true, and hderivs are not computed

public:
  // return for testing
  const std::vector<FT*>& getPairFunctions() const { return F; }

  ///constructor
  DiffTwoBodyJastrowOrbital(ParticleSet& p) : NumVars(0), my_table_ID_(p.addTable(p))
  {
    AVSR=true; //stochastic reconfiguration, hderivs are not computed

    Nparamsu=0;
    Nv=0;
    NumPtcls  = p.getTotalNum();
    NumGroups = p.groups();
    F.resize(NumGroups * NumGroups, 0);

    //AV added
    SpeciesSet tspecies(p.getSpeciesSet());
    //std::vector<std::string> name_test=tspecies.speciesName;
    double Eta=0; //5.79; //1.0; //5.76;
    std::string filename="EtaEtaVarmax.txt";
    std::ifstream fin(filename.c_str());
    if(!fin.good()){
      std::cerr<<"# Error : Cannot load from file "<<filename<<" : file not found."<<std::endl;
      std::abort();
    }
  
    fin>>Eta;

    int sephelper=1;
    fin>>sephelper;
    separate=true;
    if (sephelper<0.5){
      separate=false;
    }
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

  ~DiffTwoBodyJastrowOrbital() override
  {
    delete_iter(gradLogPsi.begin(), gradLogPsi.end());
    delete_iter(lapLogPsi.begin(), lapLogPsi.end());
  }

  // Accessors for unit testing
  std::pair<int, int> getComponentOffset(int index) { return OffSet.at(index); }

  opt_variables_type& getComponentVars() { return myVars; }


  void addFunc(int ia, int ib, std::unique_ptr<FT> j)
  {
    // make all pair terms equal to uu initially
    //   in case some terms are not provided explicitly
    if (ia == ib)
    {
      if (ia == 0) //first time, assign everything
      {
        int ij = 0;
        for (int ig = 0; ig < NumGroups; ++ig)
          for (int jg = 0; jg < NumGroups; ++jg, ++ij)
            if (F[ij] == nullptr)
              F[ij] = j.get();
      }
      else{
        if (separate){
           F[ia * NumGroups + ib] = j.get();
        }
        else{
          int ij=0;
          for (int ig = 0; ig < NumGroups; ++ig){
            for (int jg = 0; jg < NumGroups; ++jg, ++ij){
              if ((ig==jg) && (Tauvalues[ia]+Tauvalues[ib]==Tauvalues[ig]+Tauvalues[jg]))
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
      if (NumPtcls == NumGroups)
        for (int ig = 0; ig < NumGroups; ++ig)
          F[ig * NumGroups + ig] = j.get();
      // generic case
      if (separate){
        F[ia * NumGroups + ib] = j.get();
        F[ib * NumGroups + ia] = j.get();
      }
      else{
        int ij=0;
        for (int ig = 0; ig < NumGroups; ++ig){
          for (int jg = 0; jg < NumGroups; ++jg, ++ij){
            if ((ig!=jg) && (Tauvalues[ia]+Tauvalues[ib]==Tauvalues[ig]+Tauvalues[jg]))
              F[ij] = j.get();   
          }
        }
      }
    }
    std::stringstream aname;
    aname << ia << ib;
    J2Unique[aname.str()] = std::move(j);
  }

  ///reset the value of all the unique Two-Body Jastrow functions
  void resetParameters(const opt_variables_type& active) override
  {
    auto it     = J2Unique.begin();
    auto it_end = J2Unique.end();
    while (it != it_end)
    {
      (*it++).second->resetParameters(active);
    }
  }

  void checkOutVariables(const opt_variables_type& active) override
  {
    myVars.clear();
    auto it     = J2Unique.begin();
    auto it_end = J2Unique.end();
    while (it != it_end)
    {
      (*it).second->myVars.getIndex(active);
      myVars.insertFrom((*it).second->myVars);
      ++it;
    }
    // Remove inactive variables so the mappings are correct
    myVars.removeInactive();

    myVars.getIndex(active);
    NumVars = myVars.size();

    //myVars.print(std::cout);

    if (NumVars && dLogPsi.size() == 0)
    {
      dLogPsi.resize(NumVars);
      gradLogPsi.resize(NumVars, 0);
      lapLogPsi.resize(NumVars, 0);
      for (int i = 0; i < NumVars; ++i)
      {
        gradLogPsi[i] = new GradVectorType(NumPtcls);
        lapLogPsi[i]  = new ValueVectorType(NumPtcls);
      }
      OffSet.resize(F.size());

      // Find first active variable for the starting offset
      int varoffset = -1;
      for (int i = 0; i < myVars.size(); i++)
      {
        varoffset = myVars.Index[i];
        if (varoffset != -1)
          break;
      }

      for (int i = 0; i < F.size(); ++i)
      {
        if (F[i] && F[i]->myVars.Index.size())
        {
          OffSet[i].first  = F[i]->myVars.Index.front() - varoffset;
          OffSet[i].second = F[i]->myVars.Index.size() + OffSet[i].first;
        }
        else
        {
          OffSet[i].first = OffSet[i].second = -1;
        }
      }
    }
  }

  void evaluateDerivatives(ParticleSet& P,
                           const opt_variables_type& active,
                           std::vector<ValueType>& dlogpsi,
                           std::vector<ValueType>& dhpsioverpsi) override //std::vector<ValueType>& Hdpsioverpsi
  {
    if (myVars.size() == 0)
      return;
    evaluateDerivativesWF(P, active, dlogpsi);
    bool recalculate(false);
    std::vector<bool> rcsingles(myVars.size(), false);
    for (int k = 0; k < myVars.size(); ++k)
    {
      int kk = myVars.where(k);
      if (kk < 0)
        continue;
      if (active.recompute(kk))
        recalculate = true;
      rcsingles[k] = true;
    }
    if (recalculate && !AVSR)
    {
      for (int k = 0; k < myVars.size(); ++k)
      {
        int kk = myVars.where(k);
        if (kk < 0)
          continue;
        if (rcsingles[k])
        {
          //d/dalpha(lapl(logpsi)+(grad(logpsi)^2)))=lapl(d/dalpha logpsi)+2*(grad(logpsi) grad(d/dalpha logpsi)), minus sign from -1/2m. Minus sign from -J already included
          //Hdpsioverpsi[kk] = -RealType(0.5) * ValueType(Sum(*lapLogPsi[k]));
          dhpsioverpsi[kk] = -RealType(0.5) * ValueType(Sum(*lapLogPsi[k])) - ValueType(Dot(P.G, *gradLogPsi[k]));
        }
      }
    }
  }

  void evaluateDerivativesWF(ParticleSet& P, const opt_variables_type& active, std::vector<ValueType>& dlogpsi) override
  {
    if (myVars.size() == 0)
      return;
    bool recalculate(false);
    std::vector<bool> rcsingles(myVars.size(), false);
    for (int k = 0; k < myVars.size(); ++k)
    {
      int kk = myVars.where(k);
      if (kk < 0)
        continue;
      if (active.recompute(kk))
        recalculate = true;
      rcsingles[k] = true;
    }
    if (recalculate)
    {
      ///precomputed recalculation switch
      std::vector<bool> RecalcSwitch(F.size(), false);
      for (int i = 0; i < F.size(); ++i)  //F.size(): numgroups*numgroups
      {
        if (OffSet[i].first < 0)
        {
          // nothing to optimize
          RecalcSwitch[i] = false;
        }
        else
        {
          bool recalcFunc(false);
          for (int rcs = OffSet[i].first; rcs < OffSet[i].second; rcs++)  //loop over number of coefficients (within Particle group)
            if (rcsingles[rcs] == true)
              recalcFunc = true;
          RecalcSwitch[i] = recalcFunc;
        }
      }
      dLogPsi = 0.0;
      for (int p = 0; p < NumVars; ++p)
        (*gradLogPsi[p]) = 0.0;
      for (int p = 0; p < NumVars; ++p)
        (*lapLogPsi[p]) = 0.0;
      std::vector<TinyVector<RealType, 3>> derivs(NumVars);  //array dim (NumVars, 3), 3: derivatives up to second order, not dim
      

      const auto& d_table = P.getDistTableAA(my_table_ID_);
      constexpr RealType cone(1);
      constexpr RealType lapfac(OHMMS_DIM - cone);
      const size_t n  = d_table.sources();  //#particles (?)
      const size_t ng = P.groups();  //#groups (isospin flavours)
      for (size_t i = 1; i < n; ++i)
      {
        const size_t ig   = P.GroupID[i] * ng;
        const auto& dist  = d_table.getDistRow(i);  //distance r (value, norm |x_j-x_i|)
        const auto& displ = d_table.getDisplRow(i);  //vector x_j-x_i
        for (size_t j = 0; j < i; ++j)
        {
          const size_t ptype = ig + P.GroupID[j];
          if (RecalcSwitch[ptype])
          {
            std::fill(derivs.begin(), derivs.end(), 0.0);
            RealType rinv(cone / dist[j]);
            PosType dr(displ[j]);  //vector (dx,dy)
            
            //std::cout<<"i: "<<i<<" tau[i]: "<<Tauvalues[P.GroupID[i]]<<" j: "<<j<<" tau[j]: "<<Tauvalues[P.GroupID[j]]<<std::endl;
            if (!F[ptype]->evaluateDerivatives(dist[j], dr[0], dr[1], Nparamsu, Nv, derivs, Tauvalues[P.GroupID[i]]+Tauvalues[P.GroupID[j]]))
              continue;
            
            //######################################
            //AV TEST DERIV begin--------------------
            
            //maybe fill in later

            //AV TEST DERIV end-----------------------------------
            //########################################
            for (int p = OffSet[ptype].first, ip = 0; p < OffSet[ptype].second; ++p, ++ip)  //number of variables within specific group
            {

              if (!AVSR){
                std::cout<<"Need to implement Hderiv for v"<<std::endl;
                std::flush(std::cout);
                abort();
                
                int groupid_i=P.GroupID[i];
                int groupid_j=P.GroupID[j];

                //RealType dudr(rinv * derivs[ip][1]);  //1/r*du/dr

                RealType  dX = dr[0];
                RealType  dY = dr[1];
                //std::cout<<"AV in J2OrbitalSoA::recompute, dXjat: "<<dXjat<< " dYjat: "<<dYjat<<std::endl;
                RealType rsquared=dX*dX+dY*dY;

             
                if (rsquared<1e-15){
                   rsquared+=1e-15;
                }
                

                RealType dudr(rinv * derivs[ip][1]);  //1/r*du/dr

                //if (ip>=Nparamsu && ip < Nparamsu+Nparamsa2){
                //  dudr=rinva2*derivs[ip][1];
                //}
                //else if (ip>=Nparamsu + Nparamsa2 && ip < Nparamsu+Nparamsa2+Nparamsb2){
                //  dudr=rinvb2*derivs[ip][1];
                //}
              
                //std::cout<<"dudr: "<<dudr<<" derivs1: "<<derivs[ip][1]<<" derivs[2] "<<derivs[ip][2]<<std::endl;

                RealType lap_i;
                RealType lap_j;
                if (ip<Nparamsu){
                  lap_i=(dX*dX/(Masses[groupid_i][0]*rsquared)+dY*dY/(Masses[groupid_i][1]*rsquared))*(-dudr+derivs[ip][2])+dudr*(1.0/Masses[groupid_i][0]+1.0/Masses[groupid_i][1]);  //what tau is iat?? Look up table or something, masses depend on it
                  lap_j=(dX*dX/(Masses[groupid_j][0]*rsquared)+dY*dY/(Masses[groupid_j][1]*rsquared))*(-dudr+derivs[ip][2])+dudr*(1.0/Masses[groupid_j][0]+1.0/Masses[groupid_j][1]);  //what tau is iat?? Look up table or something, masses depend on it
                }
                //else if (ip>=Nparamsu && ip < Nparamsu+Nparamsa2){
                //  lap_i=(etaVar*etaVar*dX*dX/(Masses[groupid_i][0]*rtildesquared_a2)+(1.0/(etaVar*etaVar))*dY*dY/(Masses[groupid_i][1]*rtildesquared_a2))*(-dudr+derivs[ip][2])+dudr*(etaVar/Masses[groupid_i][0]+(1.0/etaVar)*1.0/Masses[groupid_i][1]);  //what tau is iat?? Look up table or something, masses depend on it
                //  lap_j=(etaVar*etaVar*dX*dX/(Masses[groupid_j][0]*rtildesquared_a2)+(1.0/(etaVar*etaVar))*dY*dY/(Masses[groupid_j][1]*rtildesquared_a2))*(-dudr+derivs[ip][2])+dudr*(etaVar/Masses[groupid_j][0]+(1.0/etaVar)*1.0/Masses[groupid_j][1]);  //what tau is iat?? Look up table or something, masses depend on it
          
                //}
                
                PosType gr_i(dudr * dr); //dX/r*delta u/delta r, welches Vorzeichen hat u? J (bzw hier d_alpha J) oder -J?
                PosType gr_j(dudr * dr);
                if (ip<Nparamsu){
                  gr_i[0]=gr_i[0]*OneOverSqrtM[P.GroupID[i]][0];
                  gr_i[1]=gr_i[1]*OneOverSqrtM[P.GroupID[i]][1];

                  gr_j[0]=gr_j[0]*OneOverSqrtM[P.GroupID[j]][0];
                  gr_j[1]=gr_j[1]*OneOverSqrtM[P.GroupID[j]][1];
                }
                //else if (ip>=Nparamsu && ip < Nparamsu+Nparamsa2){
                //  gr_i[0]=gr_i[0]*etaVar*OneOverSqrtM[P.GroupID[i]][0];
                //  gr_i[1]=gr_i[1]*1.0/etaVar*OneOverSqrtM[P.GroupID[i]][1];

                //  gr_j[0]=gr_j[0]*etaVar*OneOverSqrtM[P.GroupID[j]][0];
                //  gr_j[1]=gr_j[1]*1.0/etaVar*OneOverSqrtM[P.GroupID[j]][1];
                //}
                

                //std::cout<<"ip: "<<ip<<"lap i: "<<lap_i<< "lap j: "<<lap_j<<" gr i: "<<gr_i<<" gr j: "<<gr_j<<std::endl;
                (*gradLogPsi[p])[i] += gr_i;  //r_j-r_i (j<i) -> einmal minus von e^-J, nochmal von -r_i. Zusaetzlicher Faktor -1,1 in Bspline fuer abs. value abh von r_i>r_j bzw r_i<r_j -> dr!! dX, dY
                (*gradLogPsi[p])[j] -= gr_j;   //dX=x_j-x_i for j<i
                (*lapLogPsi[p])[i] -= lap_i;
                (*lapLogPsi[p])[j] -= lap_j;  //minus signs: from e^-J
              }
              dLogPsi[p] -= derivs[ip][0];
                 
            }
          }
        }
      }
      for (int k = 0; k < myVars.size(); ++k)
      {
        int kk = myVars.where(k);
        if (kk < 0)
          continue;
        if (rcsingles[k])
        {
          dlogpsi[kk] = dLogPsi[k];
        }
        //optVars.setDeriv(p,dLogPsi[ip],-0.5*Sum(*lapLogPsi[ip])-Dot(P.G,*gradLogPsi[ip]));
      }
    }
  }

  std::unique_ptr<DiffWaveFunctionComponent> makeClone(ParticleSet& tqp) const override
  {
    auto j2copy = std::make_unique<DiffTwoBodyJastrowOrbital<FT>>(tqp);
    std::map<const FT*, FT*> fcmap;
    for (int ig = 0; ig < NumGroups; ++ig)
      for (int jg = ig; jg < NumGroups; ++jg)
      {
        int ij = ig * NumGroups + jg;
        if (F[ij] == nullptr)
          continue;
        auto fit = fcmap.find(F[ij]);
        if (fit == fcmap.end())
        {
          auto fc      = std::make_unique<FT>(*F[ij]);
          fcmap[F[ij]] = fc.get();
          j2copy->addFunc(ig, jg, std::move(fc));
        }
      }
    j2copy->myVars.clear();
    j2copy->myVars.insertFrom(myVars);
    j2copy->NumVars   = NumVars;
    j2copy->NumPtcls  = NumPtcls;
    j2copy->NumGroups = NumGroups;
    j2copy->dLogPsi.resize(NumVars);
    j2copy->gradLogPsi.resize(NumVars, 0);
    j2copy->lapLogPsi.resize(NumVars, 0);
    for (int i = 0; i < NumVars; ++i)
    {
      j2copy->gradLogPsi[i] = new GradVectorType(NumPtcls);
      j2copy->lapLogPsi[i]  = new ValueVectorType(NumPtcls);
    }
    j2copy->OffSet = OffSet;
    return j2copy;
  }
};
} // namespace qmcplusplus
#endif
