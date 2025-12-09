//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file strat.cpp
//! \brief Problem generator for stratified 3D shearing sheet.
//!
//! PURPOSE:  Problem generator for stratified 3D shearing sheet.
//!
//!
//! Code must be configured using -shear
//!
//! REFERENCE:
//! - Stone, J., Hawley, J., Gammie, C.F. & Balbus, S. A., ApJ 463, 656-673 (1996)
//! - Hawley, J. F. & Balbus, S. A., ApJ 400, 595-609 (1992)
//============================================================================

// C headers

// C++ headers
#include <algorithm>
#include <cmath>      // sqrt()
#include <iostream>
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <random>     // mt19937, normal_distribution, uniform_real_distribution

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../parameter_input.hpp"
#include "../utils/utils.hpp"     // ran2()

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

// TODO(felker): many unused arguments in these functions: time, iout, ...
int RefinementCondition(MeshBlock *pmb);
void VertGrav(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);

void StirringThePot(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);
void StirringThePot2(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);

void StirringTheLoop(Mesh *pm);
void StirringTheLoop2(Mesh *pm);
void StirringTheLoop3(Mesh *pm);
void StirringTheLoop4(Mesh *pm);

void WaveDamping(MeshBlock *pmb, const Real time, const Real dt,
                   const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
                   const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                   AthenaArray<Real> &cons_scalar);

void MySourceTerms(MeshBlock *pmb, const Real time, const Real dt,
                   const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
                   const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                   AthenaArray<Real> &cons_scalar);

void StratOutflowInnerX3(MeshBlock *pmb, Coordinates *pco,
                         AthenaArray<Real> &a,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void StratOutflowOuterX3(MeshBlock *pmb, Coordinates *pco,
                         AthenaArray<Real> &a,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void InnerX1(MeshBlock *pmb, Coordinates *pco,
                         AthenaArray<Real> &a,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void OuterX1(MeshBlock *pmb, Coordinates *pco,
                          AthenaArray<Real> &a,
                          FaceField &b, Real time, Real dt,
                          int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void InnerX2(MeshBlock *pmb, Coordinates *pco,
                         AthenaArray<Real> &a,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void OuterX2(MeshBlock *pmb, Coordinates *pco,
                          AthenaArray<Real> &a,
                          FaceField &b, Real time, Real dt,
                          int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void InnerX3(MeshBlock *pmb, Coordinates *pco,
                         AthenaArray<Real> &a,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void OuterX3(MeshBlock *pmb, Coordinates *pco,
                          AthenaArray<Real> &a,
                          FaceField &b, Real time, Real dt,
                          int il, int iu, int jl, int ju, int kl, int ku, int ngh);                


namespace {

Real HistorydVxVy(MeshBlock *pmb, int iout);

// globals for compatibility with oxthena version
int n_bh; 

// apply a density floor - useful for large |z| regions
Real dfloor, pfloor;
Real Omega_0, qshear, H0;
int strat;

// turbulence parameters
int inc_turb;
Real Lx, Ly, Lz,Lmin;
int mxmin,mxmax,mymin,mymax,mzmin,mzmax,Nmodes;
Real turbamp,expo, tcor,solenoidal;
int sign;
TimeIntegratorTaskList *ptlist;

// wave damping parameters
Real dampwidthx, dampwidthy, dampwidthz;
Real damptime;

// global switches
int ALIVE = 1;
int DEAD   = 0;

// global indexing variables for the user mesh data
int TRB_RM;
int TRB_IM = 0, IDEAD= 1, IMB = 2;
int JMX = 0, JMY = 1, JMZ = 2, JALIVE = 3;
int JPHX = 0, JPHY = 1, JPHZ = 2, JT0 = 3, JAMP = 4;

// *** testing Lim method
Real tdrive;

// random number generator
std::mt19937_64 rng_generator;
std::int64_t rseed;
std::uniform_real_distribution<Real> udist(0.0,1.0); // uniform in [0,1)
std::uniform_int_distribution<> idist(0,1); // uniform integer distribution in [0,1]
std::normal_distribution<Real> ndist(0.0,1.0); // normal distribution

int umeshsize;


} // namespace

//====================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {

  // compatibility with oxthena
  n_bh = 0;

  // shearing sheet parameter
  qshear = pin->GetReal("orbital_advection","qshear");
  Omega_0 = pin->GetReal("orbital_advection","Omega0");
  H0 = pin->GetReal("hydro","iso_sound_speed");

  // stratification parameters
  strat = pin->GetOrAddInteger("problem","strat", 1);

  // forced turbulence parameters
  inc_turb = pin->GetOrAddInteger("problem","inc_turb", 0);

  // wave damping parameters
  damptime = pin->GetOrAddReal("problem","damptime", 0.0);
  if (damptime > 0.0) {
    dampwidthx = pin->GetOrAddReal("problem","dampwidthx", 0.1*(mesh_size.x1max - mesh_size.x1min));
    dampwidthy = pin->GetOrAddReal("problem","dampwidthy", 0.1*(mesh_size.x2max - mesh_size.x2min));
    dampwidthz = pin->GetOrAddReal("problem","dampwidthz", 0.1*(mesh_size.x3max - mesh_size.x3min));;
  }

  if (inc_turb){
    umeshsize = n_bh + 2;
    // set any indexing shifts here to adjust default
    TRB_RM = n_bh + 1;
  }
  else{
    umeshsize = n_bh + 1;
  }

  // allocate real user mesh data
  AllocateRealUserMeshDataField(umeshsize);
  
  if (inc_turb) {
  
    Lx = pin->GetReal("mesh","x1max") - pin->GetReal("mesh","x1min");
    Ly = pin->GetReal("mesh","x2max") - pin->GetReal("mesh","x2min");
    Lz = pin->GetReal("mesh","x3max") - pin->GetReal("mesh","x3min"); 
    Real L_min = std::min(Lx, std::min(Ly,Lz));

    // Random number generation global variables
    rseed = 1;
    rng_generator.seed(rseed);
    ptlist = new TimeIntegratorTaskList(pin, this);

    mxmin = pin->GetOrAddInteger("problem","mxmin",1);
    mymin = pin->GetOrAddInteger("problem","mymin",0);
    mzmin = pin->GetOrAddInteger("problem","mzmin",0);
    mxmax = pin->GetOrAddInteger("problem","mxmax",6);
    mymax = pin->GetOrAddInteger("problem","mymax",3);
    mzmax = pin->GetOrAddInteger("problem","mzmax",2);
    Nmodes = pin->GetOrAddInteger("problem","Nmodes",10);
    turbamp = pin->GetOrAddReal("problem","turbamp",1e-3);
    expo = pin->GetOrAddReal("problem","expo",2.0);
    tcor = pin->GetOrAddReal("problem","tcor",0.1);
    solenoidal = pin->GetOrAddReal("problem","solenoidal",1.0);

    // initialize the int mesh data arrays
    AllocateIntUserMeshDataField(3);
    iuser_mesh_data[TRB_IM].NewAthenaArray(Nmodes,4);   // Stores the [Nmodes][mx,my,mz,active/dead]
    iuser_mesh_data[IDEAD].NewAthenaArray(1);          // Stores whether there are any dead modes
    iuser_mesh_data[IMB].NewAthenaArray(1);            // Stores the meshblock counter

    for (int n=0; n<Nmodes; n++) {
      for (int m=0; m<4; m++)
        iuser_mesh_data[TRB_IM](n,m) = 0;
    }
    iuser_mesh_data[IDEAD](0) = DEAD; 
    iuser_mesh_data[IMB](0) = 0; 

    // initialize the real mesh data arrays
    ruser_mesh_data[TRB_RM].NewAthenaArray(Nmodes,5);
    
    for (int n=0; n<Nmodes; n++) {
      ruser_mesh_data[TRB_RM](n,JPHX) = 0.0;
      ruser_mesh_data[TRB_RM](n,JPHY) = 0.0;
      ruser_mesh_data[TRB_RM](n,JPHZ) = 0.0;  
      ruser_mesh_data[TRB_RM](n,JT0)  = 0.0;
      ruser_mesh_data[TRB_RM](n,JAMP) = 0.0;
      ruser_mesh_data[TRB_RM](n,5)    = 0.0;
    }

    // *** test
    tdrive = tcor;

  } //end of inc_turb

  // Enroll user-defined history output
  AllocateUserHistoryOutput(1);
  EnrollUserHistoryOutput(0, HistorydVxVy, "dVxVy");

  // Enroll user-defined physical source terms
  EnrollUserExplicitSourceFunction(MySourceTerms);

  // Enroll user-defined boundary conditions
  // if (mesh_bcs[BoundaryFace::inner_x3] == GetBoundaryFlag("user")) {
  //   EnrollUserBoundaryFunction(BoundaryFace::inner_x3, StratOutflowInnerX3);
  // }
  // if (mesh_bcs[BoundaryFace::outer_x3] == GetBoundaryFlag("user")) {
  //   EnrollUserBoundaryFunction(BoundaryFace::outer_x3, StratOutflowOuterX3);
  // }

  // enroll user-defined boundary condition
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, InnerX1);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, OuterX1);
  }
  if (mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, InnerX2);
  }
  if (mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, OuterX2);
  }
  if (mesh_bcs[BoundaryFace::inner_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, InnerX3);
  }
  if (mesh_bcs[BoundaryFace::outer_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, OuterX3);
  }

  // Enroll user-defined refinement condition
  if(adaptive==true){
    std::cout << "### Refinement is on in problem generator" << std::endl;
    EnrollUserRefinementCondition(RefinementCondition);
  }

  return;
}

//======================================`================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief stratified disk problem generator for 3D problems.
//======================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real pres;
  Real iso_cs;

  Real SumRd=0.0, SumRvx=0.0, SumRvy=0.0, SumRvz=0.0;
  Real x1, x3;
  Real rd(0.0), rp(0.0);
  Real rvx, rvy, rvz;
  Real rval;

  // initialize density
  const Real den=1.0;

  // Initialize boxsize
  Real Lx = pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min;

  // adiabatic gamma
  Real gam = peos->GetGamma();

  if (pmy_mesh->mesh_size.nx3 == 1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in strat.cpp ProblemGenerator" << std::endl
        << "Stratified shearing sheet only works on a 3D grid" << std::endl;
    ATHENA_ERROR(msg);
  }

  // Read problem parameters for initial conditions
  
  Real float_min = std::numeric_limits<float>::min();
  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(float_min)));
  pfloor=pin->GetOrAddReal("hydro","pfloor",(1024*(float_min)));

  // Compute pressure based on the EOS.
  if (NON_BAROTROPIC_EOS) {
    pres  = pin->GetOrAddReal("problem","pres",1.0);
  } else {
    iso_cs = peos->GetIsoSoundSpeed();
    std:: cout << "### Using isothermal sound speed cs = " << iso_cs << std::endl;
    pres = den*SQR(iso_cs);
  }

  // Initialize fluid quantities 
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        x1 = pcoord->x1v(i);
        x3 = pcoord->x3v(k);

        rd = den;
        if (strat){
          rd *= std::exp(-x3*x3/(2.0*H0*H0));
        }
        rvx = 0;
        rvy = 0;
        rvz = 0;

        // Initialize d, M, and P.
        // for_the_future: if FARGO do not initialize the bg shear
        phydro->u(IDN,k,j,i) = rd;
        phydro->u(IM1,k,j,i) = rd*rvx;
        phydro->u(IM2,k,j,i) = rd*rvy;
        if(!porb->orbital_advection_defined)
          phydro->u(IM2,k,j,i) -= rd*qshear*Omega_0*x1;
        phydro->u(IM3,k,j,i) = rd*rvz;
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = rp/(gam-1.0)
                                 + 0.5*(SQR(phydro->u(IM1,k,j,i))
                                        + SQR(phydro->u(IM2,k,j,i))
                                        + SQR(phydro->u(IM3,k,j,i)))/rd;
        } // Hydro

      }
    }
  }

  return;
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  return;
}

void MeshBlock::UserWorkInLoop() {

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real& u_d  = phydro->u(IDN,k,j,i);
        u_d = (u_d > dfloor) ?  u_d : dfloor;
        if (NON_BAROTROPIC_EOS) {
          Real gam = peos->GetGamma();
          Real& w_p  = phydro->w(IPR,k,j,i);
          Real& u_e  = phydro->u(IEN,k,j,i);
          const Real& u_m1 = phydro->u(IM1,k,j,i);
          const Real& u_m2 = phydro->u(IM2,k,j,i);
          const Real& u_m3 = phydro->u(IM3,k,j,i);
          w_p = (w_p > pfloor) ?  w_p : pfloor;
          Real di = 1.0/u_d;
          Real ke = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
          u_e = w_p/(gam-1.0)+ke;
        }
      }
    }
  }
  return;
}

void Mesh::UserWorkInLoop() {

  // Now perform the turbulence update at the end of each tiemstep if needed
  if (inc_turb == 2) StirringTheLoop(this);

  if (inc_turb == 22) StirringTheLoop2(this);

  if (inc_turb == 23) {
    if (time >= tdrive) {
      StirringTheLoop3(this);
      tdrive += tcor;
    }
  }

  if (inc_turb == 24) {
    if (time >= tdrive) {
      StirringTheLoop4(this);
      tdrive += tcor;
    }
  }

  return;
}

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {




  return;
}

int RefinementCondition(MeshBlock *pmb){
  
  for(int i=pmb->is; i<=pmb->ie; i++) {
    // Extract the x location
    Real x1 = pmb->pcoord->x1v(i);
    // Refine after the first cycle only
    if ((pmb->pmy_mesh->ncycle == 1) && (std::abs(x1) < 0.2*Lx)) {
      std::cout << "Refining meshblock " << pmb->gid << std::endl;
      return 1;
    }
  }
  
  return 0;

}

void VertGrav(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar) {
  Real fsmooth, xi, sign;
  Real Lz = pmb->pmy_mesh->mesh_size.x3max - pmb->pmy_mesh->mesh_size.x3min;
  Real z0 = Lz/2.0;
  Real lambda = 0.1*H0/z0;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real den = prim(IDN,k,j,i);
        Real x3 = pmb->pcoord->x3v(k);
        // smoothing function
        if (x3 >= 0) {
          sign = -1.0;
        } else {
          sign = 1.0;
        }
        xi = z0/x3;
        fsmooth = SQR( std::sqrt( SQR(xi+sign) + SQR(xi*lambda) ) + xi*sign );
        // multiply gravitational potential by smoothing function
        cons(IM3,k,j,i) -= dt*den*SQR(Omega_0)*x3*fsmooth;
        if (NON_BAROTROPIC_EOS) {
          cons(IEN,k,j,i) -= dt*den*SQR(Omega_0)*prim(IVZ,k,j,i)*x3*fsmooth;
        }
      }
    }
  }
  return;
}

void StirringThePot(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar){

  Real x1, x2, x3,dvol;
  Real mx,my,mz,mmag,phasex,phasey,phasez,amp;
  Real fx,fy,fz,kx,ky,kz,kamp,Ax,Ay,Az,tlife,t0,qomt;
  Real den;

  //..................................//
  // Remove modes which have exceeded their lifetime
  //.................................//

  // if on the first meshblock on the mesh rank (no need to repeat work shared on mesh)
  if (pmb->pmy_mesh->iuser_mesh_data[IMB](0) == 0){
    // loop across the modes and remove dead ones
    for (int mode = 0; mode<Nmodes; mode++) {
            
      // compute characteristic mode lifetime
      Real mmag = std::sqrt(SQR(pmb->pmy_mesh->iuser_mesh_data[TRB_IM](mode,JMX))+
                           SQR(pmb->pmy_mesh->iuser_mesh_data[TRB_IM](mode,JMY))+
                           SQR(pmb->pmy_mesh->iuser_mesh_data[TRB_IM](mode,JMZ)));
      Real tlife = tcor/mmag;

      // check if mode has exceed lifetime
      Real t0 = pmb->pmy_mesh->ruser_mesh_data[TRB_RM](mode,JT0);
      if ((time-t0) > tlife){
        // flag mode as dead
        pmb->pmy_mesh->iuser_mesh_data[TRB_IM](mode,JALIVE) = DEAD;
        // flag that there are dead modes
        pmb->pmy_mesh->iuser_mesh_data[IDEAD](0) = DEAD;
        // std::cout <<"GID "<< pmb->gid << " Removing mode at " << my << " after t-t0 = " << time-t0 << " at time " << time << std::endl;
      }
    }
  }

  //..................................//
  // Populate full list of active modes
  //.................................//

  // if there are dead modes, repopulate the mode list
  if (pmb->pmy_mesh->iuser_mesh_data[IDEAD](0) == DEAD){

    // loop through the full mode list to find the dead modes
    for (int mode=0; mode<Nmodes; mode++) {

      // if mode is active then continue
      if (pmb->pmy_mesh->iuser_mesh_data[TRB_IM](mode,JALIVE) == ALIVE) continue;

      // otherwise find a new mode
      std::uniform_int_distribution<> distrib_mx(mxmin, mxmax);
      std::uniform_int_distribution<> distrib_my(mymin, mymax);
      std::uniform_int_distribution<> distrib_mz(mzmin, mzmax);

      mx = distrib_mx(rng_generator);
      sign = idist(rng_generator)*2-1;
      my = sign*distrib_my(rng_generator);
      sign = idist(rng_generator)*2-1;
      mz = sign*distrib_mz(rng_generator);
      
      pmb->pmy_mesh->iuser_mesh_data[TRB_IM](mode,JMX) = mx;
      pmb->pmy_mesh->iuser_mesh_data[TRB_IM](mode,JMY) = my;
      pmb->pmy_mesh->iuser_mesh_data[TRB_IM](mode,JMZ) = mz;

      // update the phase, spawn time and amplitude
      pmb->pmy_mesh->ruser_mesh_data[TRB_RM](mode,JPHX) = udist(rng_generator)*TWO_PI;
      pmb->pmy_mesh->ruser_mesh_data[TRB_RM](mode,JPHY) = udist(rng_generator)*TWO_PI;
      pmb->pmy_mesh->ruser_mesh_data[TRB_RM](mode,JPHZ) = udist(rng_generator)*TWO_PI;
      pmb->pmy_mesh->ruser_mesh_data[TRB_RM](mode,JT0) = time;
      pmb->pmy_mesh->ruser_mesh_data[TRB_RM](mode,JAMP) = ndist(rng_generator);

      // mark mode as active
      pmb->pmy_mesh->iuser_mesh_data[TRB_IM](mode,JALIVE) = ALIVE; 
    }

    // mark that the mode list has been fully populated
    pmb->pmy_mesh->iuser_mesh_data[IDEAD](0) = ALIVE; 
  } //end of mode population
  
  //..................................//
  // Now implement forcing 
  //.................................//

  for (int mode=0; mode<Nmodes; mode++){
    
    mx = pmb->pmy_mesh->iuser_mesh_data[TRB_IM](mode,JMX);
    my = pmb->pmy_mesh->iuser_mesh_data[TRB_IM](mode,JMY);
    mz = pmb->pmy_mesh->iuser_mesh_data[TRB_IM](mode,JMZ);

    mmag = std::sqrt(SQR(mx)+SQR(my)+SQR(mz));
    tlife = tcor/mmag; 

    phasex = pmb->pmy_mesh->ruser_mesh_data[TRB_RM](mode,JPHX);
    phasey = pmb->pmy_mesh->ruser_mesh_data[TRB_RM](mode,JPHY);
    phasez = pmb->pmy_mesh->ruser_mesh_data[TRB_RM](mode,JPHZ);

    t0 = pmb->pmy_mesh->ruser_mesh_data[TRB_RM](mode,JT0);
    amp = pmb->pmy_mesh->ruser_mesh_data[TRB_RM](mode,JAMP)*std::sin(M_PI*(time - t0)/tlife);;

    qomt = qshear*Omega_0*(time-t0);

    kx =2*M_PI*mx/Lx+qomt*ky;
    ky = 2*M_PI*my/Ly;
    kz = 2*M_PI*mz/Lz;
    kamp = std::sqrt(SQR(kx)+SQR(ky)+SQR(kz));

    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
        for (int i=pmb->is; i<=pmb->ie; ++i) {

          Real den = prim(IDN,k,j,i);
          Real x1 = pmb->pcoord->x1v(i);
          Real x2 = pmb->pcoord->x2v(j);
          Real x3 = pmb->pcoord->x3v(k);

          Ax = (turbamp*amp)/(std::pow(kamp,(expo+2.0)/2.0))*cos(kx*x1+ky*x2+kz*x3+phasex);
          Ay = (turbamp*amp)/(std::pow(kamp,(expo+2.0)/2.0))*cos(kx*x1+ky*x2+kz*x3+phasey);
          Az = (turbamp*amp)/(std::pow(kamp,(expo+2.0)/2.0))*cos(kx*x1+ky*x2+kz*x3+phasez);

          fx = (ky*Az - kz*Ay);
          fy = (kz*Ax - kx*Az);
          fz = (kx*Ay - ky*Ax);

          // add to the conserved variables
          cons(IM1,k,j,i) += dt*den*fx;
          cons(IM2,k,j,i) += dt*den*fy;
          cons(IM3,k,j,i) += dt*den*fz;

        }
      }
    }

  } //end of forcing loop

  //..................................//
  // Meshblock bookeeping
  //.................................//

  pmb->pmy_mesh->iuser_mesh_data[IMB](0) += 1; // increment the meshblock counter on the mesh
  if (pmb->pmy_mesh->iuser_mesh_data[IMB](0) == pmb->pmy_mesh->nblocal){
    pmb->pmy_mesh->iuser_mesh_data[IMB](0) = 0; // reset the meshblock counter on the mesh
  }
  
  return;
}

void StirringThePot2(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar){

  Real x1, x2, x3,dvol;
  Real mx,my,mz,mmag,phasex,phasey,phasez,amp;
  Real fx,fy,fz,kx,ky,kz,kamp,Ax,Ay,Az,tlife,t0,qomt;
  Real den;
  
  //..................................//
  // Now implement forcing 
  //.................................//


    kx = M_PI;
    ky = M_PI;
    kz = M_PI;
    kamp = std::sqrt(SQR(kx)+SQR(ky)+SQR(kz));

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {

        Real den = prim(IDN,k,j,i);
        Real x1 = pmb->pcoord->x1v(i);
        Real x2 = pmb->pcoord->x2v(j);
        Real x3 = pmb->pcoord->x3v(k);

        fx = +turbamp*sin(kx*x1)*cos(ky*x2)*cos(kz*x3);
        fy = -turbamp*cos(kx*x1)*sin(ky*x2)*cos(kz*x3);
        fz = +turbamp*sin(kz*x3);

        // add to the conserved variables
        cons(IM1,k,j,i) += dt*den*fx;
        cons(IM2,k,j,i) += dt*den*fy;
        cons(IM3,k,j,i) += dt*den*fz;

      }
    }
  }
  
  return;
}

void StirringTheLoop(Mesh *pm){

  Real m[4] = {0};   // cumulative mass and momentum counter

  Real x1, x2, x3,dvol;
  Real mx,my,mz,mmag,phasex,phasey,phasez,amp;
  Real fx,fy,fz,kx,ky,kz,kamp,Ax,Ay,Az,tlife,t0,qomt;
  Real Phi;
  Real den;
  MeshBlock *pmb;
  int is, ie, js, je, ks, ke;
  int il, iu, jl, ju, kl, ku;

  // extract the active cell bounds - same on all AMR refinement levels
  is = pm->my_blocks(0)->is, ie = pm->my_blocks(0)->ie;
  js = pm->my_blocks(0)->js, je = pm->my_blocks(0)->je;
  ks = pm->my_blocks(0)->ks, ke = pm->my_blocks(0)->ke;

  // set the bounds including ghost zones (I think that this structure is preserved with AMR)
  il= is-NGHOST;
  iu= ie+NGHOST;
  jl= js-NGHOST;
  ju= je+NGHOST;
  kl= ks-NGHOST;
  ku= ke+NGHOST;

  //..................................//
  // Remove modes which have exceeded their lifetime
  //.................................//

  // loop across the modes and remove dead ones
  for (int mode = 0; mode<Nmodes; mode++) {
    
    // compute characteristic mode lifetime
    mmag = std::sqrt(SQR(pm->iuser_mesh_data[TRB_IM](mode,JMX))+
                          SQR(pm->iuser_mesh_data[TRB_IM](mode,JMY))+
                          SQR(pm->iuser_mesh_data[TRB_IM](mode,JMZ)));
    tlife = tcor/mmag; 

    // check if mode has exceed lifetime
    t0 = pm->ruser_mesh_data[TRB_RM](mode,JT0);
    if ((pm->time -t0) > tlife){
      // flag mode as dead
      pm->iuser_mesh_data[TRB_IM](mode,JALIVE) = DEAD;
      // flag that there are dead modes
      pm->iuser_mesh_data[IDEAD](0) = DEAD;
      // std::cout << " Removing mode at " << mode << " after t-t0 = " << pm->time-t0 << " at time " << pm->time << std::endl;
    }
  }

  //..................................//
  // Populate full list of active modes
  //.................................//

  // if there are dead modes, repopulate the mode list
  if (pm->iuser_mesh_data[IDEAD](0) == DEAD){

    // loop through the full mode list to find the dead modes
    for (int mode=0; mode<Nmodes; mode++) {

      // if mode is active then continue
      if (pm->iuser_mesh_data[TRB_IM](mode,JALIVE) == ALIVE) continue;

      // otherwise find a new mode
      std::uniform_int_distribution<> distrib_mx(mxmin, mxmax);
      std::uniform_int_distribution<> distrib_my(mymin, mymax);
      std::uniform_int_distribution<> distrib_mz(mzmin, mzmax);

      mx = distrib_mx(rng_generator);
      sign = idist(rng_generator)*2-1;
      my = sign*distrib_my(rng_generator);
      sign = idist(rng_generator)*2-1;
      mz = sign*distrib_mz(rng_generator);

      pm->iuser_mesh_data[TRB_IM](mode,JMX) = mx;
      pm->iuser_mesh_data[TRB_IM](mode,JMY) = my;
      pm->iuser_mesh_data[TRB_IM](mode,JMZ) = mz;

      // update the phase, spawn time and amplitude
      pm->ruser_mesh_data[TRB_RM](mode,JPHX) = udist(rng_generator)*TWO_PI;
      pm->ruser_mesh_data[TRB_RM](mode,JPHY) = udist(rng_generator)*TWO_PI;
      pm->ruser_mesh_data[TRB_RM](mode,JPHZ) = udist(rng_generator)*TWO_PI;
      pm->ruser_mesh_data[TRB_RM](mode,JT0) = pm->time;
      pm->ruser_mesh_data[TRB_RM](mode,JAMP) = ndist(rng_generator);

      // mark mode as active
      pm->iuser_mesh_data[TRB_IM](mode,JALIVE) = ALIVE; 
    }

    // mark that the mode list has been fully populated
    pm->iuser_mesh_data[IDEAD](0) = ALIVE; 

  }//end of mode population

  // std::cout << Globals::my_rank << " Mode list " << pm->iuser_mesh_data[TRB_IM](0,JMX) << std::endl;
  // std::cout << Globals::my_rank << " Mode list " << pm->iuser_mesh_data[TRB_IM](1,JMX) << std::endl;
  // std::cout << Globals::my_rank << " Mode list " << pm->iuser_mesh_data[TRB_IM](2,JMX) << std::endl;

  //.................................//
  // Now perform the forcing....
  //.................................//

  for (int mode=0; mode<Nmodes; mode++){

    int mx = pm->iuser_mesh_data[TRB_IM](mode,JMX);
    int my = pm->iuser_mesh_data[TRB_IM](mode,JMY);
    int mz = pm->iuser_mesh_data[TRB_IM](mode,JMZ);

    // extract amplitude and phases
    mmag = std::sqrt(SQR(mx)+SQR(my)+SQR(mz));
    tlife = tcor/mmag; 

    phasex = pm->ruser_mesh_data[TRB_RM](mode,JPHX);
    phasey = pm->ruser_mesh_data[TRB_RM](mode,JPHY);
    phasez = pm->ruser_mesh_data[TRB_RM](mode,JPHZ);

    t0 = pm->ruser_mesh_data[TRB_RM](mode,JT0);
    amp = pm->ruser_mesh_data[TRB_RM](mode,JAMP)*std::sin(M_PI*(pm->time - t0)/tlife);

    qomt = qshear*Omega_0*(pm->time-t0);

    // create velocity kick
    kx = 2*M_PI*mx/Lx+qomt*ky;
    ky = 2*M_PI*my/Ly;
    kz = 2*M_PI*mz/Lz;
    kamp = std::sqrt(SQR(kx)+SQR(ky)+SQR(kz));

    // loop over the meshblocks and apply the forcing
    for (int bn=0; bn<pm->nblocal; ++bn) {
      pmb = pm->my_blocks(bn);

      // extract the cell volume - possibly different with refinement
      dvol = pmb->pcoord->dx1f(is)*pmb->pcoord->dx2f(js)*pmb->pcoord->dx3f(ks); 

      // loop over all cells (inc. ghost zones)
      for (int k=kl; k<=ku; k++) {
        for (int j=jl; j<=ju; j++) {
          for (int i=il; i<=iu; i++) {

            // extract the cell centered positions
            x1 = pmb->pcoord->x1v(i);
            x2 = pmb->pcoord->x2v(j);
            x3 = pmb->pcoord->x3v(k);

            Ax = (turbamp*amp)/(std::pow(kamp,(expo+2.0)/2.0))*cos(kx*x1+ky*x2+kz*x3+phasex);
            Ay = (turbamp*amp)/(std::pow(kamp,(expo+2.0)/2.0))*cos(kx*x1+ky*x2+kz*x3+phasey);
            Az = (turbamp*amp)/(std::pow(kamp,(expo+2.0)/2.0))*cos(kx*x1+ky*x2+kz*x3+phasez);

            Phi = (turbamp*amp)/(std::pow(kamp,expo))*cos(kx*x1+ky*x2+kz*x3+phasex);

            fx = (solenoidal)*(ky*Az - kz*Ay);
            fy = (solenoidal)*(kz*Ax - kx*Az);
            fz = (solenoidal)*(kx*Ay - ky*Ax);

            fx += (1.0-solenoidal)*kx*Phi;
            fy += (1.0-solenoidal)*ky*Phi;
            fz += (1.0-solenoidal)*kz*Phi; 

            // add the perturbations to the primitive variables (velocity)
            den = pmb->phydro->w(IDN,k,j,i);
            pmb->phydro->w(IVX,k,j,i) += pm->dt*fx;
            pmb->phydro->w(IVY,k,j,i) += pm->dt*fy;
            pmb->phydro->w(IVZ,k,j,i) += pm->dt*fz;

            // if in the active domain, count up the total mass and momentum
            if ( (i >= is) && (i <= ie) && (j >= js) && (j <= je) && (k >= ks) && (k <= ke) ) {
              if (mode == 0) m[0] += den*dvol;
              m[1] += pm->dt*den*fx*dvol;
              m[2] += pm->dt*den*fy*dvol;
              m[3] += pm->dt*den*fz*dvol;

            }
          }
        }
      }
    
    } // end of meshblock loop
  } // end of mode loop

  // sum the perturbations over all processors
  #ifdef MPI_PARALLEL
  int mpierr;
  mpierr = MPI_Allreduce(MPI_IN_PLACE, m, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (mpierr) {
    std::stringstream msg;
    msg << "[normalize]: MPI_Allreduce error = " << mpierr << std::endl;
    ATHENA_ERROR(msg);
  }
  #endif // MPI_PARALLEL

  // correct by removing net momentum injection
  for (int bn=0; bn<pm->nblocal; ++bn) {
        pmb = pm->my_blocks(bn);

          // loop over all cells (inc. ghost zones)
          // reduces momentum injection to zero over active zones
          for (int k=kl; k<=ku; k++) {
            for (int j=jl; j<=ju; j++) {
              for (int i=il; i<=iu; i++) {

              // extract the cell centered positions
              // correct so there is no net momentum injection
              pmb->phydro->w(IVX,k,j,i) -= m[1]/m[0];
              pmb->phydro->w(IVY,k,j,i) -= m[2]/m[0];
              pmb->phydro->w(IVZ,k,j,i) -= m[3]/m[0];

              }
            }
          }

    // update the conserved variables
    AthenaArray<Real> zeros;
    zeros.NewAthenaArray(3, pm->my_blocks(bn)->ncells3, pm->my_blocks(bn)->ncells2, pm->my_blocks(bn)->ncells1);
    pmb->peos->PrimitiveToConserved(pmb->phydro->w, zeros, pmb->phydro->u, pmb->pcoord, il, iu, jl, ju, kl, ku);
  
  }

  return;
}

void StirringTheLoop2(Mesh *pm){

  Real m[4] = {0};   // cumulative mass and momentum counter

  Real x1, x2, x3,dvol;
  Real fx,fy,fz,kx,ky,kz;
  Real den;
  MeshBlock *pmb;
  int is, ie, js, je, ks, ke;
  int il, iu, jl, ju, kl, ku;

  // extract the active cell bounds - same on all AMR refinement levels
  is = pm->my_blocks(0)->is, ie = pm->my_blocks(0)->ie;
  js = pm->my_blocks(0)->js, je = pm->my_blocks(0)->je;
  ks = pm->my_blocks(0)->ks, ke = pm->my_blocks(0)->ke;

  // set the bounds including ghost zones (I think that this structure is preserved with AMR)
  il= is-NGHOST;
  iu= ie+NGHOST;
  jl= js-NGHOST;
  ju= je+NGHOST;
  kl= ks-NGHOST;
  ku= ke+NGHOST;

  //.................................//
  // Now perform the forcing....
  //.................................//

  // loop over the meshblocks and apply the forcing
  for (int bn=0; bn<pm->nblocal; ++bn) {
    pmb = pm->my_blocks(bn);

    // extract the cell volume - possibly different with refinement
    dvol = pmb->pcoord->dx1f(is)*pmb->pcoord->dx2f(js)*pmb->pcoord->dx3f(ks); 

    // loop over all cells (inc. ghost zones)
    for (int k=kl; k<=ku; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {

          // extract the cell centered positions
          x1 = pmb->pcoord->x1v(i);
          x2 = pmb->pcoord->x2v(j);
          x3 = pmb->pcoord->x3v(k);

          kx = M_PI;
          ky = M_PI;
          kz = M_PI;

          fx = +turbamp*(std::sin(kx*x1)*std::cos(ky*x2)*std::cos(kz*x3));
          fy = -turbamp*(std::cos(kx*x1)*std::sin(ky*x2)*std::cos(kz*x3));
          fz = +turbamp*std::sin(kz*x3);

          // add the perturbations to the primitive variables
          den = pmb->phydro->w(IDN,k,j,i);
          pmb->phydro->w(IVX,k,j,i) += pm->dt*fx;
          pmb->phydro->w(IVY,k,j,i) += pm->dt*fy;
          pmb->phydro->w(IVZ,k,j,i) += pm->dt*fz;

        }
      }
    }

    // update the conserved variables
    AthenaArray<Real> zeros;
    zeros.NewAthenaArray(3, pm->my_blocks(bn)->ncells3, pm->my_blocks(bn)->ncells2, pm->my_blocks(bn)->ncells1);
    pmb->peos->PrimitiveToConserved(pmb->phydro->w, zeros, pmb->phydro->u, pmb->pcoord, il, iu, jl, ju, kl, ku);
  
  }

  return;
}

void StirringTheLoop3(Mesh *pm){

  Real den;
  MeshBlock *pmb;
  int is, ie, js, je, ks, ke;
  int il, iu, jl, ju, kl, ku;

  Real x1, x2, x3;
  Real kx, kx0, ky, kz,qomt;
  Real phx1, phy1, phz1, phx2, phy2, phz2;
  int nxt;
  Real dAxdy, dAxdz, dAydx, dAydz, dAzdx, dAzdy;
  Real dv1, dv2, dv3;

  // extract the active cell bounds - same on all AMR refinement levels
  is = pm->my_blocks(0)->is, ie = pm->my_blocks(0)->ie;
  js = pm->my_blocks(0)->js, je = pm->my_blocks(0)->je;
  ks = pm->my_blocks(0)->ks, ke = pm->my_blocks(0)->ke;

  // set the bounds including ghost zones (I think that this structure is preserved with AMR)
  il= is-NGHOST;
  iu= ie+NGHOST;
  jl= js-NGHOST;
  ju= je+NGHOST;
  kl= ks-NGHOST;
  ku= ke+NGHOST;

  //.................................//
  // Now perform the forcing....
  //.................................//

  phx1 = udist(rng_generator)*TWO_PI;
  phy1 = udist(rng_generator)*TWO_PI;
  phz1 = udist(rng_generator)*TWO_PI;
  phx2 = udist(rng_generator)*TWO_PI;
  phy2 = udist(rng_generator)*TWO_PI;
  phz2 = udist(rng_generator)*TWO_PI;

  kx0 = (2.0*M_PI/Ly);
  ky = (2.0*M_PI/Ly);
  kz = (2.0*M_PI/Ly);

  qomt = qshear*Omega_0*pm->time;

  if (pm->time == 0.0) {
    nxt = 1.0;
  } else {
    nxt = floor(1.0-qomt*ky/kx0)+1.0;
  }

  kx = kx0*nxt + qomt*ky;

  // std::cout << " nxt = " << nxt << " kx = " << kx << " ky = " << ky << " kz = " << kz << std::endl;

  // loop over the meshblocks and apply the forcing
  for (int bn=0; bn<pm->nblocal; ++bn) {
    pmb = pm->my_blocks(bn);

    // loop over all cells (inc. ghost zones)
    for (int k=kl; k<=ku; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {

          // extract the cell centered positions
          x1 = pmb->pcoord->x1v(i);
          x2 = pmb->pcoord->x2v(j);
          x3 = pmb->pcoord->x3v(k);

          // Now set compute the curl of the vector potential
          dAxdy = -H0*ky*sin(kx*x1+ky*x2+phx1)*cos(kz*x3+phx2);
          dAxdz = -H0*kz*cos(kx*x1+kx*x2+phx1)*sin(kz*x3+phx2);
          dAydx = -H0*kx*sin(kx*x1+ky*x2+phy1)*cos(kz*x3+phy2);
          dAydz = -H0*kz*cos(kx*x1+ky*x2+phy1)*sin(kz*x3+phy2);
          dAzdx = -H0*kx*sin(kx*x1+ky*x2+phz1)*cos(kz*x3+phz2);
          dAzdy = -H0*ky*sin(kx*x1+ky*x2+phz1)*cos(kz*x3+phz2);

          dv1 = turbamp*H0*(dAzdy - dAydz);
          dv2 = turbamp*H0*(dAxdz - dAzdx);
          dv3 = turbamp*H0*(dAydx - dAxdy);

          // Add the perturbations to the primitive variables
          Real den = pmb->phydro->w(IDN,k,j,i);
          pmb->phydro->w(IVX,k,j,i) += dv1;
          pmb->phydro->w(IVY,k,j,i) += dv2;
          pmb->phydro->w(IVZ,k,j,i) += dv3;

        }
      }
    }

    // update the conserved variables
    AthenaArray<Real> zeros;
    zeros.NewAthenaArray(3, pm->my_blocks(bn)->ncells3, pm->my_blocks(bn)->ncells2, pm->my_blocks(bn)->ncells1);
    pmb->peos->PrimitiveToConserved(pmb->phydro->w, zeros, pmb->phydro->u, pmb->pcoord, il, iu, jl, ju, kl, ku);
  
  }// end of meshblock loop
  
  return;
}

void StirringTheLoop4(Mesh *pm){

  Real den;
  MeshBlock *pmb;
  int is, ie, js, je, ks, ke;
  int il, iu, jl, ju, kl, ku;

  Real x1, x2, x3;
  Real kx, kx0, ky, kz,qomt;
  Real phx1, phy1, phz1, phx2, phy2, phz2;
  int nxt;
  Real dAxdy, dAxdz, dAydx, dAydz, dAzdx, dAzdy;
  Real dv1, dv2, dv3;

  // extract the active cell bounds - same on all AMR refinement levels
  is = pm->my_blocks(0)->is, ie = pm->my_blocks(0)->ie;
  js = pm->my_blocks(0)->js, je = pm->my_blocks(0)->je;
  ks = pm->my_blocks(0)->ks, ke = pm->my_blocks(0)->ke;

  // set the bounds including ghost zones (I think that this structure is preserved with AMR)
  il= is-NGHOST;
  iu= ie+NGHOST;
  jl= js-NGHOST;
  ju= je+NGHOST;
  kl= ks-NGHOST;
  ku= ke+NGHOST;

  //.................................//
  // Now perform the forcing....
  //.................................//

  phx1 = udist(rng_generator)*TWO_PI;
  phy1 = udist(rng_generator)*TWO_PI;
  phz1 = udist(rng_generator)*TWO_PI;
  phx2 = udist(rng_generator)*TWO_PI;
  phy2 = udist(rng_generator)*TWO_PI;
  phz2 = udist(rng_generator)*TWO_PI;

  int nk = floor(Ly/H0);

  kx0 = (2.0*M_PI*nk/Ly);
  ky = (2.0*M_PI*nk/Ly);
  kz = (2.0*M_PI*nk/Ly);

  qomt = qshear*Omega_0*pm->time;

  if (pm->time == 0.0) {
    nxt = 1.0;
  } else {
    nxt = floor(1.0-qomt*ky/kx0)+1.0;
  }

  kx = kx0*nxt + qomt*ky;

  // std::cout << " nxt = " << nxt << " kx = " << kx << " ky = " << ky << " kz = " << kz << std::endl;

  // loop over the meshblocks and apply the forcing
  for (int bn=0; bn<pm->nblocal; ++bn) {
    pmb = pm->my_blocks(bn);

    // loop over all cells (inc. ghost zones)
    for (int k=kl; k<=ku; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {

          // extract the cell centered positions
          x1 = pmb->pcoord->x1v(i);
          x2 = pmb->pcoord->x2v(j);
          x3 = pmb->pcoord->x3v(k);

          // Now set compute the curl of the vector potential
          dAxdy = -H0*ky*sin(kx*x1+ky*x2+phx1)*cos(kz*x3+phx2);
          dAxdz = -H0*kz*cos(kx*x1+kx*x2+phx1)*sin(kz*x3+phx2);
          dAydx = -H0*kx*sin(kx*x1+ky*x2+phy1)*cos(kz*x3+phy2);
          dAydz = -H0*kz*cos(kx*x1+ky*x2+phy1)*sin(kz*x3+phy2);
          dAzdx = -H0*kx*sin(kx*x1+ky*x2+phz1)*cos(kz*x3+phz2);
          dAzdy = -H0*ky*sin(kx*x1+ky*x2+phz1)*cos(kz*x3+phz2);

          dv1 = turbamp*H0*(dAzdy - dAydz);
          dv2 = turbamp*H0*(dAxdz - dAzdx);
          dv3 = turbamp*H0*(dAydx - dAxdy);

          // Add the perturbations to the primitive variables
          Real den = pmb->phydro->w(IDN,k,j,i);
          pmb->phydro->w(IVX,k,j,i) += dv1;
          pmb->phydro->w(IVY,k,j,i) += dv2;
          pmb->phydro->w(IVZ,k,j,i) += dv3;

        }
      }
    }

    // update the conserved variables
    AthenaArray<Real> zeros;
    zeros.NewAthenaArray(3, pm->my_blocks(bn)->ncells3, pm->my_blocks(bn)->ncells2, pm->my_blocks(bn)->ncells1);
    pmb->peos->PrimitiveToConserved(pmb->phydro->w, zeros, pmb->phydro->u, pmb->pcoord, il, iu, jl, ju, kl, ku);
  
  }// end of meshblock loop
  
  return;
}

void WaveDamping(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar){

  // Apply wave damping in the boundary zones
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
        for (int i=pmb->is; i<=pmb->ie; ++i) {

          Real x1 = pmb->pcoord->x1v(i);
          Real x2 = pmb->pcoord->x2v(j);
          Real x3 = pmb->pcoord->x3v(k);

          // lower x1 boundary
          if ((std::abs(x1-pmb->pmy_mesh->mesh_size.x1min) < dampwidthx) || 
              (std::abs(x1-pmb->pmy_mesh->mesh_size.x1max) < dampwidthx) ||
              (std::abs(x2-pmb->pmy_mesh->mesh_size.x2min) < dampwidthy) ||
              (std::abs(x2-pmb->pmy_mesh->mesh_size.x2max) < dampwidthy) ||
              (std::abs(x3-pmb->pmy_mesh->mesh_size.x3min) < dampwidthz) ||
              (std::abs(x3-pmb->pmy_mesh->mesh_size.x3max) < dampwidthz)){

            Real den_target = 1.0;
            if (strat){   
              den_target *= std::exp(-x3*x3/(2.0*H0*H0));
            }

            Real vx_target = 0;
            Real vy_target = -qshear*Omega_0*x1;
            Real vz_target = 0;

            cons(IDN,k,j,i) -= dt*(prim(IDN,k,j,i)-den_target)/damptime;
            cons(IM1,k,j,i) -= dt*(prim(IVX,k,j,i)-vx_target)*prim(IDN,k,j,i)/damptime;
            cons(IM2,k,j,i) -= dt*(prim(IVY,k,j,i)-vy_target)*prim(IDN,k,j,i)/damptime;
            cons(IM3,k,j,i) -= dt*(prim(IVZ,k,j,i)-vz_target)*prim(IDN,k,j,i)/damptime;

          }

        }
      }
  }


  return;
}

void MySourceTerms(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar) {

  // Apply vertical gravity forcing
  if (strat) VertGrav(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);

  if (damptime >0.0) WaveDamping(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);

  //Apply continuous turbulent forcing
  if (inc_turb == 1) StirringThePot(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);

  if (inc_turb == 12) StirringThePot2(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);

  return;
}

//  Here is the lower z outflow boundary.
//  The basic idea is that the pressure and density
//  are exponentially extrapolated in the ghost zones
//  assuming a constant temperature there (i.e., an
//  isothermal atmosphere). The z velocity (NOT the
//  momentum) are set to zero in the ghost zones in the
//  case of the last lower physical zone having an inward
//  flow.  All other variables are extrapolated into the
//  ghost zones with zero slope.

void StratOutflowInnerX3(MeshBlock *pmb, Coordinates *pco,
                         AthenaArray<Real> &prim, FaceField &b,
                         Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh) {

  for (int k=1; k<=ngh; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {
        Real x3 = pco->x3v(kl-k);
        Real x3b = pco->x3v(kl);
        Real den = prim(IDN,kl,j,i);
        // First calculate the effective gas temperature (Tkl=cs^2)
        // in the last physical zone. If isothermal, use H=1
        Real Tkl = 0.5*SQR(Omega_0);
        if (NON_BAROTROPIC_EOS) {
          Real presskl = prim(IPR,kl,j,i);
          presskl = std::max(presskl,pfloor);
          Tkl = presskl/den;
        }
        // Now extrapolate the density to balance gravity
        // assuming a constant temperature in the ghost zones
        prim(IDN,kl-k,j,i) = den*std::exp(-(SQR(x3)-SQR(x3b))/
                                          (2.0*Tkl/SQR(Omega_0)));
        // Copy the velocities, but not the momenta ---
        // important because of the density extrapolation above
        prim(IVX,kl-k,j,i) = prim(IVX,kl,j,i);
        prim(IVY,kl-k,j,i) = prim(IVY,kl,j,i);
        // If there's inflow into the grid, set the normal velocity to zero
        if (prim(IVZ,kl,j,i) >= 0.0) {
          prim(IVZ,kl-k,j,i) = 0.0;
        } else {
          prim(IVZ,kl-k,j,i) = prim(IVZ,kl,j,i);
        }
        if (NON_BAROTROPIC_EOS)
          prim(IPR,kl-k,j,i) = prim(IDN,kl-k,j,i)*Tkl;
      }
    }
  }
  return;
}

// Here is the upper z outflow boundary.
// The basic idea is that the pressure and density
// are exponentially extrapolated in the ghost zones
// assuming a constant temperature there (i.e., an
// isothermal atmosphere). The z velocity (NOT the
// momentum) are set to zero in the ghost zones in the
// case of the last upper physical zone having an inward
// flow.  All other variables are extrapolated into the
// ghost zones with zero slope.
void StratOutflowOuterX3(MeshBlock *pmb, Coordinates *pco,
                         AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt,
                         int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  
  for (int k=1; k<=ngh; k++) {
    for (int j=jl; j<=ju; j++) {
      for (int i=il; i<=iu; i++) {
        Real x3 = pco->x3v(ku+k);
        Real x3b = pco->x3v(ku);
        Real den = prim(IDN,ku,j,i);
        // First calculate the effective gas temperature (Tku=cs^2)
        // in the last physical zone. If isothermal, use H=1
        Real Tku = 0.5*SQR(Omega_0);
        if (NON_BAROTROPIC_EOS) {
          Real pressku = prim(IPR,ku,j,i);
          pressku = std::max(pressku,pfloor);
          Tku = pressku/den;
        }
        // Now extrapolate the density to balance gravity
        // assuming a constant temperature in the ghost zones
        prim(IDN,ku+k,j,i) = den*std::exp(-(SQR(x3)-SQR(x3b))/
                                          (2.0*Tku/SQR(Omega_0)));
        // Copy the velocities, but not the momenta ---
        // important because of the density extrapolation above
        prim(IVX,ku+k,j,i) = prim(IVX,ku,j,i);
        prim(IVY,ku+k,j,i) = prim(IVY,ku,j,i);
        // If there's inflow into the grid, set the normal velocity to zero
        if (prim(IVZ,ku,j,i) <= 0.0) {
          prim(IVZ,ku+k,j,i) = 0.0;
        } else {
          prim(IVZ,ku+k,j,i) = prim(IVZ,ku,j,i);
        }
        if (NON_BAROTROPIC_EOS)
          prim(IPR,ku+k,j,i) = prim(IDN,ku+k,j,i)*Tku;
      }
    }
  }
  return;
}

// Now test the boundary conditions with the oxthena version instead.
// Vertically extrapolated boundary condition
// Radially impose fixed shear at the boundaries

void InnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b, 
			 Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, 
			 int ngh) {

  for (int k = kl; k <= ku; ++k) {    
    const Real rd = std::exp(-pco->x3v(k)*pco->x3v(k)/(2.0*H0*H0));
    for (int j = jl; j <= ju; ++j) {
      for (int i = 1; i <= ngh; ++i) {
        // set all ghosts to ambient shear
        prim(IDN,k,j,il-i) = rd;
        prim(IVX,k,j,il-i) = 0.0;
        prim(IVY,k,j,il-i) = -qshear*Omega_0*pco->x1v(il-i);
        prim(IVZ,k,j,il-i) = 0.0;
      }
    }
  }
  return;
}

void OuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b, 
			        Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, 
			        int ngh) {

  for (int k = kl; k <= ku; ++k) {
    const Real rd = std::exp(-pco->x3v(k)*pco->x3v(k)/(2.0*H0*H0));
    for (int j = jl; j <= ju; ++j) {
      for (int i = 1; i <= ngh; ++i) {
        // set all ghosts to ambient shear
        prim(IDN,k,j,iu+i) = rd;
        prim(IVX,k,j,iu+i) = 0.0;
        prim(IVY,k,j,iu+i) = -qshear*Omega_0*pco->x1v(iu+i);
        prim(IVZ,k,j,iu+i) = 0.0;
      }
    }
  }
  return;
}

void InnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b, 
			        Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, 
			        int ngh) {

  for (int k = kl; k <= ku; ++k) {
    const Real rd = std::exp(-pco->x3v(k)*pco->x3v(k)/(2.0*H0*H0));
    for (int j = 1; j <= ngh; ++j) {
      for (int i = il; i <= iu; ++i) {
        // is cell upstream or downstream
        if (pco->x1v(i) < 0.0) {
	        // upstream, refill
	        prim(IDN,k,jl-j,i) = rd;
	        prim(IVX,k,jl-j,i) = 0.0;
	        prim(IVY,k,jl-j,i) = -qshear*Omega_0*pco->x1v(i);
          prim(IVZ,k,jl-j,i) = 0.0;

	      } else {
	        // downstream, match to domain
	        prim(IDN,k,jl-j,i) = prim(IDN,k,jl,i);
	        prim(IVX,k,jl-j,i) = prim(IVX,k,jl,i);
	        prim(IVY,k,jl-j,i) = prim(IVY,k,jl,i);
          prim(IVZ,k,jl-j,i) = prim(IVZ,k,jl,i);

	      }
      }
    }
  }
  return;
}

void OuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b, 
			        Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, 
			        int ngh) {
  // set primitive variables for upper side of box

  for (int k = kl; k <= ku; ++k) {
    const Real rd = std::exp(-pco->x3v(k)*pco->x3v(k)/(2.0*H0*H0));
    for (int j = 1; j <= ngh; ++j) {
      for (int i  = il; i <= iu; ++i) {
        // is cell upstream or downstream?
        if (pco->x1v(i) > 0.0) {
	  	    // upstream, refill
	        prim(IDN,k,ju+j,i) = rd;
	        prim(IVX,k,ju+j,i) = 0.0;
	        prim(IVY,k,ju+j,i) = -qshear*Omega_0*pco->x1v(i);
          prim(IVZ,k,ju+j,i) = 0.0;
	        
        } else {
	        // downstream, match to domain
	        prim(IDN,k,ju+j,i) = prim(IDN,k,ju,i);
	        prim(IVX,k,ju+j,i) = prim(IVX,k,ju,i);
	        prim(IVY,k,ju+j,i) = prim(IVY,k,ju,i);
          prim(IVZ,k,ju+j,i) = prim(IVZ,k,ju,i);
        }
      }
    }
  }
  return;
}

void InnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b, 
              Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, 
              int ngh) {

  for (int k = 1; k <= ngh; ++k) {
    const Real rd = std::exp(-pco->x3v(kl-k)*pco->x3v(kl-k)/(2.0*H0*H0));
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        // set all ghosts to ambient shear
        prim(IDN,kl-k,j,i) = rd;
        prim(IVX,kl-k,j,i) = 0.0;
        prim(IVY,kl-k,j,i) = - qshear*Omega_0*pco->x1v(i);
        prim(IVZ,kl-k,j,i) = 0.0;
        
      }
    }
  }
  return;
}

void OuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b, 
  Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, 
  int ngh) {

  for (int k = 1; k <= ngh; ++k) {
    const Real rd = std::exp(-pco->x3v(ku+k)*pco->x3v(ku+k)/(2.0*H0*H0));
    for (int j = jl; j <= ju; ++j) {
      for (int i = il; i <= iu; ++i) {
        // set all ghosts to ambient shear
        prim(IDN,ku+k,j,i) = rd;
        prim(IVX,ku+k,j,i) = 0.0;
        prim(IVY,ku+k,j,i) = - qshear*Omega_0*pco->x1v(i);
        prim(IVZ,ku+k,j,i) = 0.0;
        
      }
    }
  }
  return;
}

namespace {

Real HistorydVxVy(MeshBlock *pmb, int iout) {
  Real dvxvy = 0.0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &w = pmb->phydro->w;
  Real vshear = 0.0;
  AthenaArray<Real> volume; // 1D array of volumes
  // allocate 1D array for cell volume used in usr def history
  volume.NewAthenaArray(pmb->ncells1);

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
      for (int i=is; i<=ie; i++) {
        if(!pmb->porb->orbital_advection_defined) {
          vshear = -qshear*Omega_0*pmb->pcoord->x1v(i);
        } else {
          vshear = 0.0;
        }
        dvxvy += volume(i)*w(IDN,k,j,i)*w(IVX,k,j,i)*(w(IVY,k,j,i) + vshear);
      }
    }
  }
  return dvxvy;
}

} // namespace
