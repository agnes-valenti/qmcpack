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


#include "VMCUpdatePbyP.h"
#include "QMCDrivers/DriftOperators.h"
#include "Message/OpenMP.h"
#if !defined(REMOVE_TRACEMANAGER)
#include "Estimators/TraceManager.h"
#else
typedef int TraceManager;
#endif

int testvalue;
//std::vector<double> AVenergies;
//std::vector<double> AVdistances;

namespace qmcplusplus
{
/// Constructor
VMCUpdatePbyP::VMCUpdatePbyP(MCWalkerConfiguration& w, TrialWaveFunction& psi, QMCHamiltonian& h, RandomGenerator_t& rg)
    : QMCUpdateBase(w, psi, h, rg),
      buffer_timer_(*timer_manager.createTimer("VMCUpdatePbyP::Buffer", timer_level_medium)),
      movepbyp_timer_(*timer_manager.createTimer("VMCUpdatePbyP::MovePbyP", timer_level_medium)),
      hamiltonian_timer_(*timer_manager.createTimer("VMCUpdatePbyP::Hamiltonian", timer_level_medium)),
      collectables_timer_(*timer_manager.createTimer("VMCUpdatePbyP::Collectables", timer_level_medium))
{}

VMCUpdatePbyP::~VMCUpdatePbyP() {}

void VMCUpdatePbyP::advanceWalker(Walker_t& thisWalker, bool recompute)  //without energies!
{
  //std::cout<<"AV VMCUpdatePbyP advanceWalker"<<std::endl;
  std::flush(std::cout);

  //NewTimer.cpp, measure time
  buffer_timer_.start();

  //loadWalker goes to ParticleSet.cpp, set coordinates, set G and L from thisWalker (thisWalker.G, thisWalker.L)
  //values of G and L? 
  W.loadWalker(thisWalker, true);  //set G (gradient) and L (laplacian) from thisWalker
  

  Walker_t::WFBuffer_t& w_buffer(thisWalker.DataSet);
  Psi.copyFromBuffer(W, w_buffer);
  buffer_timer_.stop();

  // start PbyP moves
  movepbyp_timer_.start();
  bool moved = false;
  constexpr RealType mhalf(-0.5);
  for (int iter = 0; iter < nSubSteps; ++iter)
  {
    //create a 3N-Dimensional Gaussian with variance=1
    makeGaussRandomWithEngine(deltaR, RandomGen);

    //AVTEST
    //AV remove!!
    std::cout<<"deltaR: "<<deltaR[0]<<std::endl; //!!
    //deltaR[0][0]=1.0/600.0*64.428414; //!!
    deltaR[2][0]=1.0/600.0*(-7.678237611); //!!
    std::cout<<"deltaR: "<<deltaR[0]<<std::endl; //!!
    //deltaR[0][1]=1.0/600.0*42.15828599; //!!
    deltaR[2][1]=1.0/600.0*(-17.852113067);  //-7.678237611); //!!


    moved = false;

    //AVTEST
    //AV replace!! with commented
    for (int ig = 0; ig < W.groups(); ++ig) //loop over species
    {
      //mass set to 1
      RealType tauovermass = Tau * MassInvS[ig];
      RealType oneover2tau = 0.5 / (tauovermass);
      RealType sqrttau     = std::sqrt(tauovermass);
      Psi.prepareGroup(W, ig);
      //std::cout<<"AV mass: "<<1.0/tauovermass*Tau<<" mhalf: "<<mhalf<<std::endl;
      
      //AVTEST
      //AV replace!! with commented
      for (int iat = W.first(ig); iat < W.last(ig); ++iat)
      {
        PosType dr;
        if (UseDrift)
        {
          GradType grad_now = Psi.evalGrad(W, iat);
          DriftModifier->getDrift(tauovermass, grad_now, dr);
          dr += sqrttau * deltaR[iat];
        }
        else{
          std::cout<<"deltaR: "<<deltaR[iat]<<std::endl; //!!
          //AVTEST
          //AV replace!! with commented
          //dr = sqrttau * deltaR[iat];  //!!
          dr = deltaR[iat];
        }

        //AV remove -------------
        //std::cout<<"dr: "<<dr<<std::endl;
        //----------------------------
        if (!W.makeMoveAndCheck(iat, dr))
        {
          ++nReject;
          W.accept_rejectMove(iat, false);
          continue;
        }

        RealType prob(0);
        if (UseDrift)
        {
          GradType grad_new;
          prob = std::norm(Psi.calcRatioGrad(W, iat, grad_new));
          DriftModifier->getDrift(tauovermass, grad_new, dr);
          dr             = W.R[iat] - W.activePos - dr;
          RealType logGb = -oneover2tau * dot(dr, dr);
          RealType logGf = mhalf * dot(deltaR[iat], deltaR[iat]);
          prob *= std::exp(logGb - logGf);
        }
        else{
          //std::cout<<"AV no drift"<<std::endl;
          //calcRatio: goes to TrialWaveFunction::calcRatio. From there, loop through the wave function components (Z=0: slater det, Z=1: Jastrow). 
          //Z=0 without Backflow: calls function ratio (in DiracDeterminant). There Phi(r+deltar) (Phi orbital/here plane wave) is built by calling ElectronGasOrbitalBuilder.h (function evaluateValue). In evaluateValue, cos(kr), sin(kr) are evaluated for the Kpoints in the defined grid, for r=r_i+delta r_i. Back in ratio, det/det formula is used to calculate the ratio, using the current inverse of the matrix and the updated row of the determinant corresponding to Phi(r+deltar)
          //Z=0 with Backflow: calls ratio in SlaterDetWithBackflow
          //Z=1: TrialWaveFunction::calcRatio->J2OrbitalSoA<FT>::ratio->J2OrbitalSoA<FT>::computeU->BsplineFunctor<T>::evaluateV
          //From J2OrbitalSoA<FT>::computeU, f.evaluateV is called (where f is the chosen function for the Jastrow potential). Doc of evaluateV in BsplineFunctor.h: 'evaluate sum of the pair potentials for [iStart,iEnd)'. Also implementation in BsplineFunctor.h (ln 955). There, optimizable parameters are (probably, check!) stored in array coefs
          std::complex<double> AVtest=Psi.calcRatio(W, iat); //AV remove
          std::cout<<"AVtest ratio: "<<AVtest<<std::endl;
          prob = std::norm(Psi.calcRatio(W, iat));
          std::cout<<"AV in VMCUpdatePbyP, prob: "<<prob<<std::endl;
          }

        bool is_accepted = false;

        //AVTEST
        //AV replace!! with commented
        //if (prob >= std::numeric_limits<RealType>::epsilon() && RandomGen() < prob)
        if (ig==1 && iat==2) 
        {
          is_accepted = true;
          moved       = true;
          ++nAccept;

          //TrialWaveFunction::acceptMove. Loop through wave function components (Z=0: slater det, Z=1: Jastrow)
          //Z=0: TrialWaveFunction::acceptMove->SlaterDet.h acceptMove (loop through determinants, e.g. 2 for spin up and down) -> DiracDeterminant<DU_TYPE>::acceptMove->DelayedUpdate::acceptRow (and then, updateInvMat). Derivatives calculated in updateBuffer (not updated here)
          //Z=1: TrialWaveFunction::acceptMove->J2OrbitalSoA<FT>::acceptMove
          //update log value TrialWaveFunction::acceptMove->computeU3 (update member variables Uat, dUat and d2Uat that are used to compute G and L after updateBuffer->evaluateGL). 
          //d2Uat is computed using the Laplace equation in spherical coordinates. Re-write/ introduce additional variables storing d2xU and d2yU for anisotropic masses. 
          //For G part, anisotropic masses could be introduced directly at the kinetic energy (vector). Check whether in kinetic energy G*G indeed corresponds to L for VMC (see CASINO manual) - then no need to re-write L, just solely calculating the kinetic energy using G
          //Variable OHMMS_DIM for dimension!! Set for ElectronGas
          Psi.acceptMove(W, iat, true);
          //std::cout<<"accept move"<<std::endl;
        }
        else
        {
          ++nReject;

          //TrialWaveFunction::rejectMove. Loop through wave function components (Z=0: slater det, Z=1: Jastrow)
          //Z=0: TrialWaveFunction::rejectMove -> SlaterDet.h restore (loop through determinants, e.g. 2 for spin up and down) -> DiracDeterminant<DU_TYPE>::restore (set curRatio to 1).
          //Z=1: TrialWaveFunction::rejectMove -> J2OrbitalSoA::restore (empty function, override?)
          Psi.rejectMove(iat);
          //std::cout<<"reject move"<<std::endl;
        }

        //ParticleSet::accept_rejectMove (`ParticleSet::accept_rejectMove(iel)` ensures the distance tables (jel < iel) part is fully up-to-date regardless a move is accepted or rejected. For this reason, the rejecting operation inside `ParticleSet::accept_rejectMove` involves writing the distances with respect to the old particle position.
        //if is_accepted: ParticleSet::accept_rejectMove -> ParticleSet::acceptMoveForwardMode. There: DistTables[i]->updatePartial (update particle positions, fill partially the distance table by the pair relations from the temporary or old particle position)
        W.accept_rejectMove(iat, is_accepted);
      }
    }
    //TrialWaveFunction::completeUpdates. Loop through wave function components (Z=0: slater det, Z=1: Jastrow)
    //Z=0: TrialWaveFunction::completeUpdates -> SlaterDet.h completeUpdates (loop through determinants, e.g. 2 for spin up and down) -> DiracDeterminant<DU_TYPE>::completeUpdates -> DelayedUpdate.h updateInvMat
    //Z=1: TrialWaveFunction::completeUpdates -> ? Maybe WaveFunctionComponent completeUpdates, and empty function? Does not appear in Jastrow files
    Psi.completeUpdates(); //(substeps for all particle groups completed -> probably saving config etc. of current step in a way s.t. observables can be calculated)
  }

  //going to ParticleSet::donePbyP. Coordinates->donePbyP (other class). Going to StructFact:UpdateAllPart. In StructFact::UpdateAllPart: computeRhok(P) (also in StructFact). Rhok=e^ikr_j for each k, each particle position r_j (used then to  compute structure factor).
  W.donePbyP();

  //NewTimer.cpp, measure time
  movepbyp_timer_.stop();

  //NewTimer.cpp, measure time
  buffer_timer_.start();

  //update G, L here!!
  //going to TrialWaveFunction::updateBuffer. In TrialWaveFunction::updateBuffer, loop over wave-function components (Z) (Z[0]: slater, Z[1]: jastrow). 
  //For each component, call function updateBuffer [Returns logPsi (how is it stored) and via call-by-reference P-> P.G, P.L (how is it stored??)]. 
  //For the slater component, SlaterDet::updateBuffer is called. In SlaterDet::updateBuffer for each determinant (2 for spin up, spin down), 
  //DiracDeterminant::updateBuffer is called. There, P.G and P.L are calculated using the function evaluateGL (also in DiracDeterminant). 
  //evaluateGL has the options fromscratch and updateaftersweep. Here, updateaftersweep is chosen. In DiracDeterminant::UpdateAfterSweep, 
  //Phi->evaluate_notranspose evaluates Phi,G,L via call-by reference. The function is ElectronGasOrbitalBuilder::evaluate_notranspose 
  //(given first and last index of given particle set. There, function evaluateVGL is called (also in ElectronGasOrbitalBuilder. 
  //In evaluateVGL, psi,G,L are explicitely set: psi[j1]   = coskr; psi[j2]   = sinkr;dpsi[j1]  = -sinkr * K[ik]; dpsi[j2]  = coskr * K[ik]; d2psi[j1] = mK2[ik] * coskr; d2psi[j2] = mK2[ik] * sinkr; 

// Modify ElectronGasOrbitalBuilder::evaluateVGL for 2D, anisotropic masses (evtl put default valley index)!! K, mK2 (K^2) are set in CrystalLattice.h (There, mK2 corresponds to variable ksq. Modify there also (ln 238)  for anisotropic masses.
  //Slater component: TrialWaveFunction::updateBuffer -> SlaterDet::updateBuffer -> DiracDeterminant::updateBuffer -> DiracDeterminant::evaluateGL -> DiracDeterminant::UpdateAfterSweep -> ElectronGasOrbitalBuilder::evaluate_notranspose -> ElectronGasOrbitalBuilder::evaluateVGL (modify this function!!)
//in DiracDeterminant<DU_TYPE>::updateAfterSweep, after evaluating psiM, dpsiM, d2psiM (call ElectronGasOrbitalBuilder::evaluate_notranspose), G and L are computed: loop over all particles. G and L are stored as vectors, particle indices G[iat],L[iat]. iat current particle. Sum Slater and Jastrow components?
//Jastrow component: TrialWaveFunction::updateBuffer -> J2OrbitalSoA::updateBuffer -> J2OrbitalSoA::evaluateGL 
//(here, G and L are evaluated as derivs of the Jastrow potential U, using log_value_ += Uat[iat]; G[iat] += dUat[iat]; L[iat] += d2Uat[iat];. 
//Uat, dUat, d2Uat are not computed here but in  J2OrbitalSoA::acceptMove, calling J2OrbitalSoA::computeU3)
//comment later: don't modify directly in the calculation of grad and laplacian, because needed for optimization? Or for optimization directly deriv wrt parameters only? 
//(Write different function, or store components seperately in different variables?) -> see in general where G and L are used
//Comment, modify calculation of GL for Jastrow component (anisotropic masses): Look in BsplineFunctor::mw_evaluateVGL, mw_updateVGL for implementation
  RealType logpsi = Psi.updateBuffer(W, w_buffer, recompute);
  

  if (debug_checks_ & DriverDebugChecks::CHECKGL_AFTER_MOVES)
    checkLogAndGL(W, Psi, "checkGL_after_moves");
  W.saveWalker(thisWalker);
  buffer_timer_.stop();
  // end PbyP moves

  //calculate local energy here
  hamiltonian_timer_.start();

  //std::vector<FullPrecRealType> energies; //
  //QMCHamiltonian::evaluate, loop through components
  FullPrecRealType eloc = H.evaluate(W); 
  std::cout<<"eloc: "<<eloc<<std::endl;
  //AVenergies.push_back(eloc);

  //--------
  //const auto& d_aa(P.getDistTableAA(d_aa_ID)); //AV add
  //const auto& dist = d_aa.getDistRow(1);
  //AVdistances.push_back(dist[0]);
  //----------
  //std::cout<<"AV in VMCUpdatePbyP::AdvanceWalker, Eloc: "<<eloc<<std::endl;
  
  thisWalker.resetProperty(logpsi, Psi.getPhase(), eloc);
  hamiltonian_timer_.stop();
  collectables_timer_.start();

  //QMCHamiltonian::auxHevaluate, auxH.size()=0 
  //(for no additional observables not contributing to E to be computed) -> nothing happens
  H.auxHevaluate(W, thisWalker);
  H.saveProperty(thisWalker.getPropertyBase());
  collectables_timer_.stop();
#if !defined(REMOVE_TRACEMANAGER)
  Traces->buffer_sample(W.current_step);
#endif
  if (!moved)
    ++nAllRejected;
}

} // namespace qmcplusplus
