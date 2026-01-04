//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


#ifndef OHMMS_LRBREAKUP_PARAMETERS_H
#define OHMMS_LRBREAKUP_PARAMETERS_H

#include <iostream>
#include "config.h"
#include "OhmmsPETE/TinyVector.h"

namespace qmcplusplus
{
template<typename T, unsigned D>
class LRBreakupParameters;

template<typename T>
class LRBreakupParameters<T, 2>  //AV class was made for <T,3>-> change class to 2D!!!!!!!!!
{
public:
  ///Dimensionless cutoff radius for G/R breakups
  T LR_dim_cutoff;
  T LR_rc;
  T LR_kc;
  T LR_tol;

  ///default constructor
  LRBreakupParameters() : LR_dim_cutoff(15.0), LR_rc(1e6), LR_kc(0.0), LR_tol(3e-4) {}

  ///Set LR_rc = radius of smallest sphere inside box and kc=dim/rc
  void SetLRCutoffs(const TinyVector<TinyVector<T, 2>, 2>& a)    //AV changed 3 to 2
  {
    //#############################test stuff here#####################################
    Tensor<int, 3> test(0, 1, 2, 1, 2, 0, 2, 0, 1);
    for (int i = 0; i < 3; i++)
       std::cout<<" test tensor: "<<test(i,0)<<" "<<test(i,1)<<" "<<test(i,2)<<std::endl;
    //#################################################################################

    //Compute rc as the real-space cutoff of 1/2 the unit-cell.
    //Radius of maximum shere that fits in a...
    TinyVector<T, 2> b, c, d, x; //Unit vector of each surface will be in here  //AV changed 3 to 2
    //Compute the coordinate of the box center
    //c     = 0.5 * (a[0] + a[1] + a[2]);
    c = 0.5 * (a[0] + a[1]);  //AV changed

    std::cout<<" a0: "<<a[0][0]<<" "<<a[0][1]<<" a1: "<<a[1][0]<<" "<<a[1][1]<<std::endl;
    LR_rc = 1.e+6;
    //Tensor<int, 3> Cyclic(0, 1, 2, 1, 2, 0, 2, 0, 1);
    //AV: write this part for 2D! 
    //for (int i = 0; i < 1; i++)
    //{
    //  TinyVector<T, 2> v1 = a[0];
    //  TinyVector<T, 2> v2 = a[1];
    //  T beta1             = (dot(v2, v2) * dot(c, v1) - dot(v1, v2) * dot(c, v2)) /
    //      (dot(v1, v1) * dot(v2, v2) - dot(v1, v2) * dot(v1, v2));
    //  T beta2 = (dot(v1, v1) * dot(c, v2) - dot(v1, v2) * dot(c, v1)) /
    //      (dot(v1, v1) * dot(v2, v2) - dot(v1, v2) * dot(v1, v2));
    //  TinyVector<T, 2> p = beta1 * v1 + beta2 * v2;
    //  T dist             = sqrt(dot(p - c, p - c));
    //  std::cout<<"AV dist: "<<dist<<" LR_rc: "<<LR_rc<<std::endl;
    //  LR_rc              = std::min(LR_rc, dist);
    //}
    //square lattice:
    LR_rc=0.5*sqrt(dot(a[0],a[0]));
    std::cout<<"AV LR_rc: "<<LR_rc<<std::endl;
    //Set KC for structure-factor and LRbreakups.

    // AV until function properly modified -----------------------------------------------------------------------
    //std::cout<<"AV in LRBreakupParameters.h::SetLRCutoffs, Achtung Funktion muss modifiziert werden" <<std::endl;
    //std::cout<<std::endl;
    //abort();
    //---------------------------------------------------------------------------------
    LR_kc = LR_dim_cutoff / LR_rc;
  }

  void printCutoffs(std::ostream& out)
  {
    out << "  Long-range breakup parameters:" << std::endl;
    out << "    rc*kc = " << LR_dim_cutoff << "; rc = " << LR_rc << "; kc = " << LR_kc << "; tol = " << LR_tol
        << std::endl;
  }
};
} // namespace qmcplusplus
#endif
