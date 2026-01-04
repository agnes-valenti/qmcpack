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


#include "ElectronGasComplexOrbitalBuilder.h"
#include "QMCWaveFunctions/Fermion/SlaterDet.h"
#include "QMCWaveFunctions/Fermion/DiracDeterminant.h"
#include "OhmmsData/AttributeSet.h"

namespace qmcplusplus
{
/** constructor for EGOSet
 * @param norb number of orbitals for the EGOSet
 * @param k list of unique k points in Cartesian coordinate excluding gamma
 * @param k2 k2[i]=dot(k[i],k[i])
 */
//EGOSet::EGOSet(const std::vector<PosType>& k, const std::vector<RealType>& k2) : K(k), mK2(k2)
//{
//  KptMax         = k.size();
//  OrbitalSetSize = k.size();
//  className      = "EGOSet";
//  //assign_energies();
//}

EGOSet::EGOSet(const std::vector<PosType>& k, const std::vector<PosType>& kxy, const std::vector<RealType>& k2, const std::vector<RealType>& k2xy) : K(k), Kxy(kxy), mK2(k2), mK2xy(k2xy)
{
  KptMax         = k.size();
  OrbitalSetSize = k.size();
  className      = "EGOSet";
  //assign_energies();
}

EGOSet::EGOSet(const std::vector<PosType>& k, const std::vector<RealType>& k2, const std::vector<int>& d)
    : K(k), mK2(k2)
{
  KptMax         = k.size();
  OrbitalSetSize = k.size();
  className      = "EGOSet";
  //assign_energies();
  //assign_degeneracies(d);
}

ElectronGasComplexOrbitalBuilder::ElectronGasComplexOrbitalBuilder(Communicate* comm, ParticleSet& els)
    : WaveFunctionComponentBuilder(comm, els)
{
  nup=0;
  ndn=0;
  nuptau1=0;
  ndntau1=0;
  //AV added
  SpeciesSet tspecies(els.getSpeciesSet());
  //std::vector<std::string> name_test=tspecies.speciesName;
  int species_set_size=tspecies.size();

  int species_index_u=tspecies.findSpecies("u");
  int species_index_d=tspecies.findSpecies("d");
  int species_index_ut=tspecies.findSpecies("ut");
  int species_index_dt=tspecies.findSpecies("dt");

  std::vector<int> ng(species_set_size, 0);
  for (int iat = 0; iat < els.GroupID.size(); iat++)
  {
    if (els.GroupID[iat] < species_set_size)
      ng[els.GroupID[iat]]++;
    else
      APP_ABORT("ParticleSet::resetGroups() Failed. GroupID is out of bound.");
  }

  if (species_index_u<species_set_size){
    nup=ng[species_index_u];
  }
  if (species_index_d<species_set_size){
    ndn=ng[species_index_d];
  }
  if (species_index_ut<species_set_size){
    nuptau1=ng[species_index_ut];
  }
  if (species_index_dt<species_set_size){
    
    ndntau1=ng[species_index_dt];
  }

}


std::unique_ptr<WaveFunctionComponent> ElectronGasComplexOrbitalBuilder::buildComponent(xmlNodePtr cur)
{
  int nc = 0;
  PosType twist(0.0);
  OhmmsAttributeSet aAttrib;
  aAttrib.add(nc, "shell");
  aAttrib.add(twist, "twist");
  aAttrib.put(cur);
  //typedef DiracDeterminant<EGOSet>  Det_t;
  //typedef SlaterDeterminant<EGOSet> SlaterDeterminant_t;
  typedef DiracDeterminant<> Det_t;
  typedef SlaterDet SlaterDeterminant_t;
  int nat = targetPtcl.getTotalNum();
  
  //int firstIndexup = targetPtcl.first(0);
  //int lastIndexup  = targetPtcl.last(0);
  //int firstIndexdown = targetPtcl.first(1);
  //int lastIndexdown  = targetPtcl.last(1);
  
  //std::cout<<"AV in ElectronGasComplexOrbitalBuilder::buildComponent, first index up: "<< firstIndexup <<" last index up: "<<lastIndexup<<" firstIndexdown: "<<firstIndexdown<<" lastIndexdown: "<<lastIndexdown<<std::endl;
  //int nup =0; //5;   //AV change by hand, number up particles
  //int ndn =19; //0; //2; //0;   //AV change by hand, number down particles
  //int nuptau1=0; //18; //2;
  //int ndntau1=0;  //2;
  if (nat != (nup + ndn+nuptau1+ndntau1))
  {
    app_error() << "  The number of particles " << nup << "/" << ndn << " does not match the total number of particles " <<nat << std::endl;
    app_error() << "  Change number of particles in ElectronGasComplexOrbitalBuilder.cpp (nup, ndn) and electrongas_test.xml " << std::endl;
    APP_ABORT("ElectronGasOrbitalBuilder::put");
    return nullptr;
  }
  HEGGrid<RealType> egGrid(targetPtcl.Lattice,-1);  //second argument: tau value
  HEGGrid<RealType> egGrid2(targetPtcl.Lattice,-1);
  HEGGrid<RealType> egGrid3(targetPtcl.Lattice,1);
  HEGGrid<RealType> egGrid4(targetPtcl.Lattice,1);
  //if (nc == 0)
  //  nc = egGrid.getShellIndex(nup);

  nc=0;  //AV, spin up
  //new nc parameter: length!!
  int nc2=1;  //AV, spin down

  int nc3=2;  //AV, spin down

  int nc4=3;  //AV, spin down


  if (nup>0.5)
    egGrid.createGrid(nc, nup, twist);

  //AV separate grid for down electrons
  if (ndn>0.5)
    egGrid2.createGrid(nc2, ndn, twist);

  //AV separate grid for down electrons
  if (nuptau1>0.5)
    egGrid3.createGrid(nc3, nuptau1, twist);

  //AV separate grid for down electrons
  if (ndntau1>0.5)
    egGrid4.createGrid(nc4, ndntau1, twist);


  targetPtcl.setTwist(twist);
  std::vector<std::unique_ptr<DiracDeterminantBase>> dets;

  //create up determinant -> set here which kpoints (per isospin, can also do different grid (?))! And how many particles per isospin
  if (nup>0.5)
    dets.push_back(std::make_unique<Det_t>(std::make_unique<EGOSet>(egGrid.kpt, egGrid.kptxy, egGrid.mk2, egGrid.mk2xy), 0, nup));
  
  //create down determinant
  if (ndn>0.5)
    dets.push_back(std::make_unique<Det_t>(std::make_unique<EGOSet>(egGrid2.kpt, egGrid2.kptxy, egGrid2.mk2, egGrid2.mk2xy), nup, nup + ndn));
 
  if (nuptau1>0.5)
    dets.push_back(std::make_unique<Det_t>(std::make_unique<EGOSet>(egGrid3.kpt, egGrid3.kptxy, egGrid3.mk2, egGrid3.mk2xy), nup+ndn, nup + ndn+nuptau1));
 
  if (ndntau1>0.5)
    dets.push_back(std::make_unique<Det_t>(std::make_unique<EGOSet>(egGrid4.kpt, egGrid4.kptxy, egGrid4.mk2, egGrid4.mk2xy), nup+ndn+nuptau1, nup + ndn+nuptau1+ndntau1));
 
  //create a Slater determinant
  return std::make_unique<SlaterDet>(targetPtcl, std::move(dets));
}

ElectronGasSPOBuilder::ElectronGasSPOBuilder(ParticleSet& p, Communicate* comm, xmlNodePtr cur)
    : SPOSetBuilder("ElectronGas", comm), has_twist(false), unique_twist(-1.0), egGrid(p.Lattice,-1), spo_node(NULL)
{
  ClassName = "ElectronGasSPOBuilder";
  sizeSpecies.resize(4);

  
  //AV added
  SpeciesSet tspecies(p.getSpeciesSet());
  //std::vector<std::string> name_test=tspecies.speciesName;
  int species_set_size=tspecies.size();

  int species_index_u=tspecies.findSpecies("u");
  int species_index_d=tspecies.findSpecies("d");
  int species_index_ut=tspecies.findSpecies("ut");
  int species_index_dt=tspecies.findSpecies("dt");

  std::vector<int> ng(species_set_size, 0);
  for (int iat = 0; iat < p.GroupID.size(); iat++)
  {
    if (p.GroupID[iat] < species_set_size)
      ng[p.GroupID[iat]]++;
    else
      APP_ABORT("ParticleSet::resetGroups() Failed. GroupID is out of bound.");
  }

  if (species_index_u<species_set_size){
    sizeSpecies[0]=ng[species_index_u];
  }
  if (species_index_d<species_set_size){
    sizeSpecies[1]=ng[species_index_d];
  }
  if (species_index_ut<species_set_size){
    sizeSpecies[2]=ng[species_index_ut];
  }
  if (species_index_dt<species_set_size){
    
    sizeSpecies[3]=ng[species_index_dt];
  }
}

std::unique_ptr<SPOSet> ElectronGasSPOBuilder::createSPOSetFromXML(xmlNodePtr cur, int particletype)
{
  //std::cout<<"particletype: "<<particletype<<std::endl;
  egGrid.changeType(particletype);
  app_log() << "ElectronGasSPOBuilder::createSPOSet " << std::endl;
  int nc = 0;
  int ns = 0;
  PosType twist(0.0);
  std::string spo_name("heg");
  OhmmsAttributeSet aAttrib;
  aAttrib.add(ns, "size");
  aAttrib.add(twist, "twist");
  aAttrib.add(spo_name, "name");
  aAttrib.add(spo_name, "id");
  aAttrib.put(cur);
  if (has_twist)
    twist = unique_twist;
  else
  {
    unique_twist = twist;
    has_twist    = true;
    for (int d = 0; d < OHMMS_DIM; ++d)
      has_twist &= (unique_twist[d] + 1.0) > 1e-6;
  }






  //std::cout<<"AV in ElectronGasSPOBuilder::createSPOSetFromXML, not defined"<<std::endl;
  //abort();

  //AV changed
  egGrid.createGrid(particletype, sizeSpecies[particletype], twist);
  return std::make_unique<EGOSet>(egGrid.kpt, egGrid.kptxy, egGrid.mk2, egGrid.mk2xy);
}


std::unique_ptr<SPOSet> ElectronGasSPOBuilder::createSPOSetFromIndices(indices_t& indices)
{
  std::cout<<"AV in ElectronGasSPOBuilder::createSPOSetFromIndices, not defined"<<std::endl;
  abort();
  
  //AV, change
  //egGrid.createGrid(indices);  
  return 0; //std::make_unique<EGOSet>(egGrid.kpt, egGrid.mk2, egGrid.deg);
}


} // namespace qmcplusplus
