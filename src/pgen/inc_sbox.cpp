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
void VertGrav(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);
void MyRandom(Real *randphase, int nrand, MeshBlock *pmb);
void TurbForce(MeshBlock *pmb, AthenaArray<Real> &cons, Real dt);
void KickTurbulence(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);

void KickTurbulenceLoop(Mesh *pm);

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
int turb;
Real dtdrive, tdrive, alpha_in;
Real Lx, Ly, Lz,Lmin;
Real kx0, ky, kz;

// Random number generator global variables
std::mt19937_64 rng_generator;
std::int64_t rseed;
std::uniform_real_distribution<Real> udist(0.0,1.0); // uniform in [0,1)
int stage;
int mbcount;
TimeIntegratorTaskList *ptlist;
int nrand = 6; // number of random numbers to generate
Real *randphase = new Real[nrand]();

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
      dtdrive = pin->GetOrAddReal("problem","dtdrive", 0.001);
      tdrive = dtdrive;
      alpha_in = pin->GetOrAddReal("problem","alpha_in", 0.1);
  
      Lx = pin->GetReal("mesh","x1max") - pin->GetReal("mesh","x1min");
      Ly = pin->GetReal("mesh","x2max") - pin->GetReal("mesh","x2min");
      Lz = pin->GetReal("mesh","x3max") - pin->GetReal("mesh","x3min"); 
      Real L_min = std::min(Lx, std::min(Ly,Lz));

      kx0 = (2.0*M_PI/L_min);
      ky = (2.0*M_PI/L_min);
      kz = (2.0*M_PI/L_min);

      // Random number generation global variables
      rseed = 1;
      rng_generator.seed(rseed);
      ptlist = new TimeIntegratorTaskList(pin, this);
      stage = 0;
      mbcount = 0;

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
  // Now perform the turbulence kick if needed
  
    if (turb == 2) {
      if (time >= tdrive) {
        KickTurbulenceLoop(this);
      }
    }
  return;
}

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  return;
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

// Function which takes an array of global variables which we want to randomize.
// Care has been taken to ensure that the random numbers are constant
// over the substeps of the time integrator and over all meshblocks.
void MyRandom(Real *randphase, int nrand, MeshBlock *pmb){
  
  if (Globals::my_rank == 0 && mbcount == 0 && stage == 0){

    for (int n = 0; n < nrand; n++) {
      randphase[n] = udist(rng_generator)*TWO_PI;
    }
  }  

  // Wait until all processes have the random number
  for (int n = 0; n < nrand; n++) {
    MPI_Bcast(&randphase[n], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } 
  MPI_Barrier(MPI_COMM_WORLD);

  if (pmb->gid==0){
    stage += 1;
    if (stage == ptlist->nstages){
      stage = 0;
    }
  }

  mbcount += 1;
  if (Globals::my_rank == 0 && mbcount == pmb->pmy_mesh->nblocal){
    mbcount = 0;
  }  

return;
}

// Here is a prescription for driving turbulence in real space
//  according to the methodology of Lim et al. (2024)
void TurbForce(MeshBlock *pmb, AthenaArray<Real> &cons, Real dt){

  // Start by defining variables we will need
  Real x1, x2, x3;
  Real x1_l, x2_l, x3_l, x1_r, x2_r, x3_r;
  int nx, ny, nz; // number of cells in each direction excluding ghost zones
  Real amp_force;

  nx = (pmb->ie - pmb->is) + 1 + 2*NGHOST;
  ny = (pmb->je - pmb->js) + 1 + 2*NGHOST;
  nz = (pmb->ke - pmb->ks) + 1 + 2*NGHOST;

  // Define the different times within the integration cycle
  Real time = pmb->pmy_mesh->time;
  Real dtfullstep = pmb->pmy_mesh->dt;
  Real dtsubstep = dt;
  Real qomt,nxt,kxt;
  long int iseedxx,iseedxy,iseedxz,iseedyx,iseedyy,iseedyz,iseedzx,iseedzy,iseedzz;
  Real phi1x,phi1y,phi1z,phi2x,phi2y,phi2z;   // Randomly varying phases

  // Set the forcing amplitude -- need to motivate forcing amplitude in our case
  amp_force = sqrt(5.64*alpha_in*(Lx*Ly))*dt;

  // Define the overall shearing over the course of the simulation run time
  qomt = qshear*Omega_0*time;

  if (time == 0.0) {
    nxt = 1.0;
  } else {
    nxt = -floor(qomt*ky/kx0)+1.0;
  }

  kxt=nxt*kx0+qomt*ky;

  // ath_pout(0,"kforce=%d Forcing called at t = %.10f, tdrive = %.10f, mx = %.10f, kxt/ky= %.10f \n",kforce, pGrid->time, tdrive, mx,kxt/ky);

  //Generate seeds resulting in uncorrelated phases in time
  // Use the cycle number to generate the random seeds
  int ncycle = pmb->pmy_mesh->ncycle;

  MyRandom(randphase, nrand, pmb);

  phi1x = randphase[0];
  phi1y = randphase[1];
  phi1z = randphase[2];
  phi2x = randphase[3];
  phi2y = randphase[4];
  phi2z = randphase[5];

  // Testing a better random number generator
  // MyRandom(randphase, nrand, pmb);

  // std::cout << "GID: " << pmb->gid << "  random number 1: " << randphase[0] << std::endl;
  // std::cout << "GID: " << pmb->gid << "  random number 2: " << randphase[1] << std::endl;
  // std::cout << "GID: " << pmb->gid << "  random number 3: " << randphase[2] << std::endl;
  // std::cout << "GID: " << pmb->gid << "  random number 4: " << randphase[3] << std::endl;
  // std::cout << "GID: " << pmb->gid << "  random number 5: " << randphase[4] << std::endl;
  // std::cout << "GID: " << pmb->gid << "  random number 6: " << randphase[5] << std::endl;

  // ========================== //


  // Define the cell-faced vector potential
  // Define the vector potential arrays and allocate memory
  // Need to include extra cell for face aligned dimension
  AthenaArray<Real> Axy, Axz, Ayx, Ayz, Azx, Azy;
  Axy.NewAthenaArray(nz,ny+1,nx);
  Axz.NewAthenaArray(nz+1,ny,nx);
  Ayx.NewAthenaArray(nz,ny,nx+1);
  Ayz.NewAthenaArray(nz+1,ny,nx);
  Azx.NewAthenaArray(nz,ny,nx+1);
  Azy.NewAthenaArray(nz,ny+1,nx);

  // Define the velocity forcing
  Real dv1, dv2, dv3;

  // Loop over the active cells
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {

        // Now extract the cell centered positions
        x1 = pmb->pcoord->x1v(i);
        x2 = pmb->pcoord->x2v(j);
        x3 = pmb->pcoord->x3v(k);

        // Extract the left cell faces for the cube
        x1_l = pmb->pcoord->x1f(i);
        x2_l = pmb->pcoord->x2f(j);
        x3_l = pmb->pcoord->x3f(k);

        // Now define the vector potential at the centred faces
        //  Notation: Aij => i: i component of the vector potential, i=x,y,z
        //                   j: shifted along j directon, j=x,y,z */ 

        Axy(k,j,i) = cos(kxt*x1+ky*x2_l+phi1x)*cos(kz*x3+phi2x); // on y face
        Axz(k,j,i) = cos(kxt*x1+ky*x2+phi1x)*cos(kz*x3_l+phi2x); // on z face
        Ayx(k,j,i) = cos(kxt*x1_l+ky*x2+phi1y)*cos(kz*x3+phi2y); // on x face
        Ayz(k,j,i) = cos(kxt*x1+ky*x2+phi1y)*cos(kz*x3_l+phi2y); // on z face
        Azx(k,j,i) = cos(kxt*x1_l+ky*x2+phi1z)*cos(kz*x3+phi2z); // on x face
        Azy(k,j,i) = cos(kxt*x1+ky*x2_l+phi1z)*cos(kz*x3+phi2z); // on y face

        if (i==pmb->ie) {
          x1_r = pmb->pcoord->x1f(i+1);
          Ayx(k,j,i+1) = cos(kxt*x1_r+ky*x2+phi1y)*cos(kz*x3+phi2y); // on x face
          Azx(k,j,i+1) = cos(kxt*x1_r+ky*x2+phi1z)*cos(kz*x3+phi2z); // on x face
        }
        if (j==pmb->je) {
          x2_r = pmb->pcoord->x2f(j+1);
          Axy(k,j+1,i) = cos(kxt*x1+ky*x2_r+phi1x)*cos(kz*x3+phi2x); // on y face
          Azy(k,j+1,i) = cos(kxt*x1+ky*x2_r+phi1z)*cos(kz*x3+phi2z); // on y face
        }
        if (k==pmb->ke) {
          x3_r = pmb->pcoord->x3f(k+1);
          Axz(k+1,j,i) = cos(kxt*x1+ky*x2+phi1x)*cos(kz*x3_r+phi2x); // on z face
          Ayz(k+1,j,i) = cos(kxt*x1+ky*x2+phi1y)*cos(kz*x3_r+phi2y); // on z face
        }

      }  
    }
  } // end i,j,k loops

  // Velocity perturbations from the curl of the vector potential
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
        for (int i=pmb->is; i<=pmb->ie; ++i) {

          dv1 = (amp_force/ky)*( (Azy(k,j+1,i) - Azy(k,j,i))/pmb->pcoord->dx2f(j) -
                                        (Ayz(k+1,j,i) - Ayz(k,j,i))/pmb->pcoord->dx3f(k) );
          dv2 = (amp_force/ky)*( (Axz(k+1,j,i) - Axz(k,j,i))/pmb->pcoord->dx3f(k) -
                                        (Azx(k,j,i+1) - Azx(k,j,i))/pmb->pcoord->dx1f(i) );
          dv3 = (amp_force/ky)*( (Ayx(k,j,i+1) - Ayx(k,j,i))/pmb->pcoord->dx1f(i) -
                                        (Axy(k,j+1,i) - Axy(k,j,i))/pmb->pcoord->dx2f(j) );

          // Add the perturbations to the conserved variables
          cons(IM1,k,j,i) += cons(IDN,k,j,i)*dv1;
          cons(IM2,k,j,i) += cons(IDN,k,j,i)*dv2;
          cons(IM3,k,j,i) += cons(IDN,k,j,i)*dv3;

        }
      }
    }

  return;
}

void KickTurbulence(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar) {

              Real time1=pmb->pmy_mesh->time;
              Real dtfullstep =pmb->pmy_mesh->dt;
              Real time2=time1+dtfullstep;
              
              if((time1 <= tdrive) && (tdrive < time2))
                { 
                  TurbForce(pmb, cons, dt);
                  // if (pmb->gid==0){ std::cout << "Turbulence driving at t= " << time1 << std::endl;}
                }

              if (time1 >= tdrive){
                  TurbForce(pmb, cons, dt);
                  tdrive+=dtdrive;
              }

              // Issue an error if the driving time is less than the full timestep
              // if (dtdrive < dtfullstep) {
              //   std::stringstream msg;
              //   msg << "### FATAL ERROR in inc_sbox.cpp KickTurbulence" << std::endl
              //       << "The turbulence driving time interval must be >= the timestepping" << std::endl;
              //   ATHENA_ERROR(msg);
              // }
                
  return;
}

void KickTurbulenceLoop(Mesh *pm){

  // AthenaArray<Real> vel[3];
  // AthenaArray<Real> dvx, dvy, dvz;

  Real m[4] = {0};
  Real phi1x, phi1y, phi1z, phi2y, phi2z, phi2x;
  Real x1, x2, x3;
  Real dv1, dv2, dv3;
  Real qomt,nxt,kxt,amp_force;
  MeshBlock *pmb;
  int is, ie, js, je, ks, ke;
  int il, iu, jl, ju, kl, ku;

  // Extract the active cell bounds
  is = pm->my_blocks(0)->is, ie = pm->my_blocks(0)->ie;
  js = pm->my_blocks(0)->js, je = pm->my_blocks(0)->je;
  ks = pm->my_blocks(0)->ks, ke = pm->my_blocks(0)->ke;

  // Set the bounds including ghost zones
  il= is-NGHOST;
  iu= ie+NGHOST;
  jl= js-NGHOST;
  ju= je+NGHOST;
  kl= ks-NGHOST;
  ku= ke+NGHOST;

  // dvx.NewAthenaArray(pm->my_blocks(0)->ncells3,
  //                     pm->my_blocks(0)->ncells2,
  //                     pm->my_blocks(0)->ncells1);
  // dvy.NewAthenaArray(pm->my_blocks(0)->ncells3,
  //                     pm->my_blocks(0)->ncells2,
  //                     pm->my_blocks(0)->ncells1);
  // dvz.NewAthenaArray(pm->my_blocks(0)->ncells3,
  //                     pm->my_blocks(0)->ncells2,
  //                     pm->my_blocks(0)->ncells1);

  phi1x = udist(rng_generator)*TWO_PI;
  phi1y = udist(rng_generator)*TWO_PI;
  phi1z = udist(rng_generator)*TWO_PI;
  phi2y = udist(rng_generator)*TWO_PI;
  phi2z = udist(rng_generator)*TWO_PI;
  phi2x = udist(rng_generator)*TWO_PI;

  // Now generate some random numbers
  std::cout << " random number 1 : " << phi1x << std::endl;
  std::cout << " random number 2 : " << phi1y << std::endl;
  std::cout << " random number 3 : " << phi1z << std::endl;
  std::cout << " random number 4 : " << phi2y << std::endl;
  std::cout << " random number 5 : " << phi2z << std::endl;
  std::cout << " random number 6 : " << phi2x << std::endl;

  // Define the overall shearing over the course of the simulation run time
  Real time = pm->time;
  qomt = qshear*Omega_0*time;

  if (time == 0.0) {
    nxt = 1.0;
  } else {
    nxt = -floor(qomt*ky/kx0)+1.0;
  }

  kxt=nxt*kx0+qomt*ky;

  // Set the forcing amplitude -- need to motivate forcing amplitude in our case
  amp_force = sqrt(5.64*alpha_in*(Lx*Ly))*dtdrive;

  // Now loop over the meshblocks and apply the forcing

  for (int bn=0; bn<pm->nblocal; ++bn) {
      pmb = pm->my_blocks(bn);

      std::cout << "Turbulence driving at t= " << pm->time << " on MeshBlock GID= " << pmb->gid << std::endl;

        // Loop over all cells (inc. ghost zones -- my method should be fine for uniform mesh. Not sure about AMR)
        for (int k=kl; k<=ku; k++) {
          for (int j=jl; j<=ju; j++) {
            for (int i=il; i<=iu; i++) {

              // Now extract the cell centered positions
              x1 = pmb->pcoord->x1v(i);
              x2 = pmb->pcoord->x2v(j);
              x3 = pmb->pcoord->x3v(k);

              // Now set compute the curl of the vector potential
              Real dAxdy = -ky*sin(kxt*x1+ky*x2+phi1x)*cos(kz*x3+phi2x);
              Real dAxdz = -kz*cos(kxt*x1+ky*x2+phi1x)*sin(kz*x3+phi2x);
              Real dAydx = -kxt*sin(kxt*x1+ky*x2+phi1y)*cos(kz*x3+phi2y);
              Real dAydz = -kz*cos(kxt*x1+ky*x2+phi1y)*sin(kz*x3+phi2y);
              Real dAzdx = -kxt*sin(kxt*x1+ky*x2+phi1z)*cos(kz*x3+phi2z);
              Real dAzdy = -ky*cos(kxt*x1+ky*x2+phi1z)*sin(kz*x3+phi2z);

              // dvx(k,j,i) = (amp_force/ky)*(dAzdy - dAydz);
              // dvy(k,j,i) = (amp_force/ky)*(dAxdz - dAzdx);
              // dvz(k,j,i) = (amp_force/ky)*(dAydx - dAxdy);

              dv1 = (amp_force/ky)*(dAzdy - dAydz);
              dv2 = (amp_force/ky)*(dAxdz - dAzdx);
              dv3 = (amp_force/ky)*(dAydx - dAxdy);

              // Add the perturbations to the primitive variables
              Real den = pmb->phydro->w(IDN,k,j,i);
              pmb->phydro->w(IVX,k,j,i) += dv1;
              pmb->phydro->w(IVY,k,j,i) += dv2;
              pmb->phydro->w(IVZ,k,j,i) += dv3;

              // If in the active domain, count up the total mass and momentum
              if ( (i >= is) && (i <= ie) && (j >= js) && (j <= je) && (k >= ks) && (k <= ke) ) {
                m[0] += den;
                m[1] += den*dv1;
                m[2] += den*dv2;
                m[3] += den*dv3;
              }
            }
          }
        }
    }

  #ifdef MPI_PARALLEL
  int mpierr;
  // Sum the perturbations over all processors
  mpierr = MPI_Allreduce(MPI_IN_PLACE, m, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (mpierr) {
    std::stringstream msg;
    msg << "[normalize]: MPI_Allreduce error = " << mpierr << std::endl;
    ATHENA_ERROR(msg);
  }
#endif // MPI_PARALLEL

std::cout << " Total density = " << m[0] << std::endl;
std::cout << " Total momentum perturbation in x = " << m[1] << std::endl;
std::cout << " Total momentum perturbation in y = " << m[2] << std::endl;
std::cout << " Total momentum perturbation in z = " << m[3] << std::endl;

// Now correct to remove net momentum injection
for (int bn=0; bn<pm->nblocal; ++bn) {
      pmb = pm->my_blocks(bn);

        // Loop over all cells (inc. ghost zones -- my method should be fine for uniform mesh. Not sure about AMR)
        for (int k=kl; k<=ku; k++) {
          for (int j=jl; j<=ju; j++) {
            for (int i=il; i<=iu; i++) {

              // Now extract the cell centered positions
              // Add the perturbations to the primitive variables
              Real den = pmb->phydro->w(IDN,k,j,i);
              pmb->phydro->w(IVX,k,j,i) -= m[1]/m[0];
              pmb->phydro->w(IVY,k,j,i) -= m[2]/m[0];
              pmb->phydro->w(IVZ,k,j,i) -= m[3]/m[0];

            }
          }
        }

  // Must also update the primitive variables
  AthenaArray<Real> zeros;
  zeros.NewAthenaArray(3, pm->my_blocks(bn)->ncells3, pm->my_blocks(bn)->ncells2, pm->my_blocks(bn)->ncells1);
  pmb->peos->PrimitiveToConserved(pmb->phydro->w, zeros, pmb->phydro->w,pmb->pcoord, il, iu, jl, ju, kl, ku);

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

  //Apply turbulent forcing
  if (turb == 1) {
    KickTurbulence(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);
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
