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
void TurbForce(MeshBlock *pmb, AthenaArray<Real> &cons, Real dt);
void KickTurbulence(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);
void KickTurbulenceLoop(Mesh *pm);
void SynchronizeArrays();
void StirringThePot(MeshBlock *pmb, const Real time, const Real dt,
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

namespace {

Real HistorydVxVy(MeshBlock *pmb, int iout);

// Apply a density floor - useful for large |z| regions
Real dfloor, pfloor;
Real Omega_0, qshear;
int strat;

// Turbulence parameters
int turb;
Real Lx, Ly, Lz,Lmin;
Real kx0, ky, kz;
int mxmax,mymax,mzmax,Nmodes;
Real turbamp;
int sign;

// Set up some global indexing variables for the user mesh data
int ALIVE = 1;
int DEAD   = 0;
int IMODE = 0, IDEAD= 1, IMB = 2;
int JMX = 0, JMY = 1, JMZ = 2, JALIVE = 3;
int JPH = 0, JT0 = 1;

// Random number generator global variables
std::mt19937_64 rng_generator;
std::int64_t rseed;
std::uniform_real_distribution<Real> udist(0.0,1.0); // uniform in [0,1)
std::uniform_int_distribution<> idist(0,1); // uniform integer distribution in [0,1]

} // namespace

//====================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  // shearing sheet parameter
  qshear = pin->GetReal("orbital_advection","qshear");
  Omega_0 = pin->GetReal("orbital_advection","Omega0");

  // read in the stratification parameters
  strat = pin->GetOrAddInteger("problem","strat", 1);

  // read in the forced turbulence parameters
  turb = pin->GetOrAddInteger("problem","turb", 0);
  if (turb) {
  
      Lx = pin->GetReal("mesh","x1max") - pin->GetReal("mesh","x1min");
      Ly = pin->GetReal("mesh","x2max") - pin->GetReal("mesh","x2min");
      Lz = pin->GetReal("mesh","x3max") - pin->GetReal("mesh","x3min"); 
      Real L_min = std::min(Lx, std::min(Ly,Lz));

      // Random number generation global variables
      rseed = 1;
      rng_generator.seed(rseed);

      mxmax = pin->GetOrAddInteger("problem","mxmax",6);
      mymax = pin->GetOrAddInteger("problem","mymax",3);
      mzmax = pin->GetOrAddInteger("problem","mzmax",2);
      Nmodes = pin->GetOrAddInteger("problem","Nmodes",10);
      turbamp = pin->GetOrAddReal("problem","turbamp",1e-3);

      // These live on the mesh and are accessible by the individual meshblocks
      AllocateIntUserMeshDataField(3);
      iuser_mesh_data[IMODE].NewAthenaArray(Nmodes,4);  // Stores the [Nmodes][mx,my,mz,active/dead]
      iuser_mesh_data[IDEAD].NewAthenaArray(1);         // Stores whether there are any dead modes
      iuser_mesh_data[IMB].NewAthenaArray(1);         // Stores the meshblock counter

      // Initialize the iuser_mesh_data
      iuser_mesh_data[IDEAD](0) = DEAD; // Dead modes at the start
      for (int n=0; n<Nmodes; n++) {
        for (int m=0; m<4; m++)
          // No modes are active at the start
          iuser_mesh_data[IMODE](n,m) = 0;
      }
      iuser_mesh_data[IMB](0) = 0; // meshblock counter

      // Stores the [Nmodes][phase, t0]
      AllocateRealUserMeshDataField(1);
      ruser_mesh_data[IMODE].NewAthenaArray(Nmodes,2);
  } 

  AllocateUserHistoryOutput(1);
  EnrollUserHistoryOutput(0, HistorydVxVy, "dVxVy");

  // Enroll user-defined physical source terms
  EnrollUserExplicitSourceFunction(MySourceTerms);

  // enroll user-defined boundary conditions
  if (mesh_bcs[BoundaryFace::inner_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, StratOutflowInnerX3);
  }
  if (mesh_bcs[BoundaryFace::outer_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, StratOutflowOuterX3);
  }

  if(adaptive==true){
    std::cout << "### Refinement is on in problem generator" << std::endl;
    EnrollUserRefinementCondition(RefinementCondition);
  }

  if (!shear_periodic) {
    std::stringstream msg;
    msg << "### FATAL ERROR in inc_sbox.cpp ProblemGenerator" << std::endl
        << "This problem generator requires shearing box." << std::endl;
    ATHENA_ERROR(msg);
  }

  return;
}

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief stratified disk problem generator for 3D problems.
//======================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real pres;
  Real iso_cs=1.0;

  Real SumRd=0.0, SumRvx=0.0, SumRvy=0.0, SumRvz=0.0;
  Real x1, x3;
  Real rd(0.0), rp(0.0);
  Real rvx, rvy, rvz;
  Real rval;

  // initialize density
  const Real den=1.0;

  // Initialize boxsize
  Real Lx = pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min;

  // Ensure a different initial random seed for each meshblock.
  std::int64_t iseed = -1 - gid;

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
          rd *= std::exp(-x3*x3/2.0);
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

  // // Testing - set all modes to zero
  // iuser_mesh_data[0](0) = 0;
  // for (int n=0; n<Nmodes; n++) {
  //   for (int m=0; m<4; m++){
  //     // No modes are active at the start
  //     iuser_mesh_data[1](n,m) = 0;
  //   }
  // }

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
  Real lambda = 0.1 / z0;
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

  if (pmb->gid == 0){
    std::cout <<"========" << " Time = " << time << std::endl;
  }
    
  //..................................//
  // Remove modes which have exceeded their lifetime
  //.................................//

  // If on the first meshblock on the mesh rank (no need to repeat work shared on mesh)
  if (pmb->pmy_mesh->iuser_mesh_data[IMB](0) == 0){
    // Loop across the modes and remove dead ones
    for (int mode = 0; mode<Nmodes; mode++) {
      
      // Extract info relevant to mode lifetime
      Real my = pmb->pmy_mesh->iuser_mesh_data[IMODE](mode,JMY);
      // Sompute characteristic mode lifetime
      Real tlife = 10*Ly/(std::abs(my)); 

      // Check if mode has exceed lifetime
      Real t0 = pmb->pmy_mesh->ruser_mesh_data[IMODE](mode,JT0);
      if ((time-t0) > tlife){
        // Flag mode as dead
        pmb->pmy_mesh->iuser_mesh_data[IMODE](mode,JALIVE) = DEAD;
        // Flag that there are dead modes
        pmb->pmy_mesh->iuser_mesh_data[IDEAD](0) = DEAD;
        std::cout <<"GID "<< pmb->gid << " Removing mode at " << mode << std::endl;
      }
    }
  }

  //..................................//
  // Populate full list of active modes
  //.................................//

  // If there are dead modes, repopulate the mode list
  if (pmb->pmy_mesh->iuser_mesh_data[IDEAD](0) == DEAD){

    // Loop through the full mode list to find the dead modes
    for (int mode=0; mode<Nmodes; mode++) {

      // If mode is active then continue
      if (pmb->pmy_mesh->iuser_mesh_data[IMODE](mode,JALIVE) == ALIVE) continue;

      // Otherwise find a new mode
      
      // Update the modal numbers
      std::uniform_int_distribution<> distrib_mx(-mxmax, mxmax);
      std::uniform_int_distribution<> distrib_my(1, mymax);
      std::uniform_int_distribution<> distrib_mz(-mzmax, mzmax);
      pmb->pmy_mesh->iuser_mesh_data[IMODE](mode,JMX) = distrib_mx(rng_generator);
      sign = idist(rng_generator)*2-1;
      pmb->pmy_mesh->iuser_mesh_data[IMODE](mode,JMY) = sign*distrib_my(rng_generator);
      pmb->pmy_mesh->iuser_mesh_data[IMODE](mode,JMZ) = distrib_mz(rng_generator);

      // Update the phase and spawn time
      pmb->pmy_mesh->ruser_mesh_data[IMODE](mode,JPH) = udist(rng_generator)*TWO_PI;
      pmb->pmy_mesh->ruser_mesh_data[IMODE](mode,JT0) = time;

      // Mark mode as active
      pmb->pmy_mesh->iuser_mesh_data[IMODE](mode,JALIVE) = ALIVE; 

    }
    // Mark that the mode list has been fully populated
    pmb->pmy_mesh->iuser_mesh_data[IDEAD](0) = ALIVE; 
  }//end of mode population

  for (int n=0; n<Nmodes; n++) {
    std::cout << "gid " << pmb->gid << " " << pmb->pmy_mesh->iuser_mesh_data[IMODE](n,JMX) << std::endl;
  }
  
  //..................................//
  // Now implement forcing 
  //.................................//
  Real dPhidx,dPhidy,dPhidz;
  dPhidx = 0.0;
  dPhidy = 0.0;
  dPhidz = 0.0;

  for (int mode=0; mode<Nmodes; mode++){
    int mx = pmb->pmy_mesh->iuser_mesh_data[IMODE](mode,JMX);
    int my = pmb->pmy_mesh->iuser_mesh_data[IMODE](mode,JMY);
    int mz = pmb->pmy_mesh->iuser_mesh_data[IMODE](mode,JMZ);
    Real phase = pmb->pmy_mesh->ruser_mesh_data[IMODE](mode,JPH);
    Real t0 = pmb->pmy_mesh->ruser_mesh_data[IMODE](mode,JT0);

    kx0 = 2*M_PI*mx/Lx;
    ky = 2*M_PI*my/Ly;
    kz = 2*M_PI*mz/Lz;

    Real qomt = qshear*Omega_0*(time-t0);
    Real kxt = kx0 + qomt*ky;
    Real tlife = 10*Ly/(std::abs(my)); // Change the characteristic mode lifetime here

    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
        for (int i=pmb->is; i<=pmb->ie; ++i) {

          Real den = prim(IDN,k,j,i);
          Real x1 = pmb->pcoord->x1v(i);
          Real x2 = pmb->pcoord->x2v(j);
          Real x3 = pmb->pcoord->x3v(k);

          // Real mask=1;
          // if (x1 < -0.4*Lx || x1 > 0.4*Lx){mask=0;}
          // if (x2 < -0.4*Ly || x2 > 0.4*Ly){mask=0;}
          // if (x3 < -0.4*Lz || x3 > 0.4*Lz){mask=0;}

          Real Phi_fact = (turbamp/Nmodes)*cos(kxt*x1+ky*x2+kz*x3+phase)*sin(M_PI*(time-t0)/tlife);

          dPhidx += kxt*Phi_fact;
          dPhidy += ky*Phi_fact;
          dPhidz += kz*Phi_fact;

          // // Now compute the net momentum injection
          // if (mode==0 && mask==1){
          //   m[0] += den;
          // }
          // m[1] += den*dPhidx;
          // m[2] += den*dPhidy;
          // m[3] += den*dPhidz;

          // if (mask == 1){
            // Now add to the conserved variables
            cons(IM1,k,j,i) += dt*den*dPhidx;
            cons(IM2,k,j,i) += dt*den*dPhidy;
            cons(IM3,k,j,i) += dt*den*dPhidz;
          // }
        }
      }
    }

  } //end of forcing loop

  // // MPI_Allreduce(MPI_IN_PLACE, m, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  // // std::cout << " Total density = " << m[0] << std::endl;

  // // Correct for net momentum injection
  // for (int k=pmb->ks; k<=pmb->ke; ++k) {
  //     for (int j=pmb->js; j<=pmb->je; ++j) {
  //       for (int i=pmb->is; i<=pmb->ie; ++i) {
  //         Real den = prim(IDN,k,j,i);
  //         Real x1 = pmb->pcoord->x1v(i);
  //         Real x2 = pmb->pcoord->x2v(j);
  //         Real x3 = pmb->pcoord->x3v(k);

  //         Real mask=1;
  //         // if (x1 < -0.4*Lx || x1 > 0.4*Lx){mask=0;}
  //         // if (x2 < -0.4*Ly || x2 > 0.4*Ly){mask=0;}
  //         // if (x3 < -0.4*Lz || x3 > 0.4*Lz){mask=0;}

  //         if (mask == 1){
  //           cons(IM1,k,j,i) -= dt*den*m[1]/m[0];
  //           cons(IM2,k,j,i) -= dt*den*m[2]/m[0];
  //           cons(IM3,k,j,i) -= dt*den*m[3]/m[0];
  //         // }
  //         } else {
  //           cons(IM1,k,j,i) = 0.0;
  //           cons(IM2,k,j,i) = 0.0;
  //           }

  //         }
  //       }
  //     }

  pmb->pmy_mesh->iuser_mesh_data[IMB](0) += 1; // Increment the meshblock counter on the mesh
  if (pmb->pmy_mesh->iuser_mesh_data[IMB](0) == pmb->pmy_mesh->nblocal){
    pmb->pmy_mesh->iuser_mesh_data[IMB](0) = 0; // Reset the meshblock counter on the mesh
  }
  
  return;
}

void MySourceTerms(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar) {

  // Apply vertical gravity forcing
  if (strat){
    VertGrav(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);
  }

  //Apply continuous turbulent forcing
  if (turb) {
    StirringThePot(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);
  }

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
