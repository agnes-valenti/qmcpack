//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Luke Shulenburger, lshulen@sandia.gov, Sandia National Laboratories
//
// File created by: Luke Shulenburger, lshulen@sandia.gov, Sandia National Laboratories
//////////////////////////////////////////////////////////////////////////////////////

#include "OptimizableFunctorBase.h"

void print(OptimizableFunctorBase& func, std::ostream& os, double extent)
{
  typedef OptimizableFunctorBase::real_type real_type;
  int n       = 1000;
  real_type d = extent == -1.0 ? func.cutoff_radius / n : extent / n;
  real_type r = 0;
  real_type u, du, u2;
  for (int i = 0; i < n; ++i)
  {
    real_type x=r; //d;
    real_type y=0; //2*d;

    //u  = func.f(r);
    u=0;
    //u2  = func.f(std::sqrt(x*x+y*y),x*x,y*y,0,-2);
    //du = func.df(std::sqrt(x*x+y*y),x*x,y*y,0,-2);
    //os << std::setw(22) << r << std::setw(22) << u2 << std::setw(22) << du << std::endl;
    //r += d;
  }
}
