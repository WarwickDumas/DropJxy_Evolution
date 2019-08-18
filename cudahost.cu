
// Version 1.0 23/04/19:
// Changing to use upwind T for advection. We could do better in future. Interp gives negative T sometimes.
// Corrected ionisation rate.
    

#pragma once 
 
#define BWD_SUBCYCLE_FREQ  1
#define BWD_STEP_RATIO     1    // divide substeps by this for bwd
#define NUM_BWD_ITERATIONS 4
#define FWD_STEP_FACTOR    2    // multiply substeps by this for fwd

// This will be slow but see if it solves it.
   
#define CHOSEN  88519  
#define CHOSEN1 1000110301
#define CHOSEN2 1000110497 
#define VERTCHOSEN 14791
  
#include <math.h>
#include <time.h>
#include <stdio.h>
   
#include "FFxtubes.h"
#include "cuda_struct.h"
#include "flags.h"
#include "kernel.h"
#include "mesh.h"
    
// This is the file for CUDA host code.
#include "simulation.cu"
FILE * fp_trajectory;
FILE * fp_dbg;
 
extern long NumInnerFrills, FirstOuterFrill;
__constant__ long NumInnerFrills_d, FirstOuterFrill_d;
__constant__ long nBlocks, Nverts, uDataLen_d; // Nverts == numVertices 
__constant__ f64_tens2 Anticlockwise_d, Clockwise_d; // use this to do rotation.   
__constant__ f64 kB, c, q, m_e, m_ion, m_i, m_n, 
eoverm, qoverM, moverM, qovermc, qoverMc,
FOURPI_Q_OVER_C, FOURPI_Q, FOURPI_OVER_C,
one_over_kB, one_over_kB_cubed, kB_to_3halves,
NU_EI_FACTOR, nu_eiBarconst, Nu_ii_Factor,
M_i_over_in,// = m_i / (m_i + m_n);
M_e_over_en,// = m_e / (m_e + m_n);
M_n_over_ni,// = m_n / (m_i + m_n);
M_n_over_ne,// = m_n / (m_e + m_n);
M_en, //= m_e * m_n / ((m_e + m_n)*(m_e + m_n));
M_in, // = m_i * m_n / ((m_i + m_n)*(m_i + m_n));
M_ei, // = m_e * m_i / ((m_e + m_i)*(m_e + m_i));
m_en, // = m_e * m_n / (m_e + m_n);
m_ei, // = m_e * m_i / (m_e + m_i);
over_sqrt_m_ion, over_sqrt_m_e, over_sqrt_m_neutral,
over_m_e, over_m_i, over_m_n,
four_pi_over_c_ReverseJz,
FRILL_CENTROID_OUTER_RADIUS_d, FRILL_CENTROID_INNER_RADIUS_d;

__constant__ f64 cross_s_vals_viscosity_ni_d[10], cross_s_vals_viscosity_nn_d[10],
                 cross_T_vals_d[10], cross_s_vals_MT_ni_d[10];
__device__ __constant__ f64 billericay;
__constant__ f64 Ez_strength;
f64 EzStrength_;
__constant__ f64 negative_Iz_per_triangle; // -Iz_prescribed / (real)(numEndZCurrentTriangles - numStartZCurrentTriangles)
__constant__ long numStartZCurrentTriangles, numEndZCurrentTriangles;

#define CallMAC(cudaStatus) Call(cudaStatus, #cudaStatus )   
// { Call(cudaStatus, "cudaStatus") } ?
extern real FRILL_CENTROID_OUTER_RADIUS, FRILL_CENTROID_INNER_RADIUS;

cuSyst cuSyst1, cuSyst2, cuSyst3;
extern cuSyst cuSyst_host;
// Given our restructure, we are going to need to dimension
// a cuSyst type thing that lives on the host??
// Not necessarily and not an easy way to write.
// This time around find another way to populate.
// We do need a temporary such object in the routine where we populate the device one.
// I guess as before we want an InvokeHost routine because of that.
__device__ real * p_summands, *p_Iz0_summands, *p_Iz0_initial, *p_scratch_d;
f64 * p_summands_host, *p_Iz0_summands_host, *p_Iz0_initial_host;
__device__ f64 * p_temp1, *p_temp2, *p_temp3, *p_temp4,*p_temp5, *p_temp6, *p_denom_i, *p_denom_e, *p_coeff_of_vez_upon_viz, *p_beta_ie_z;
__device__ long * p_longtemp;
__device__ bool * p_bool;
bool * p_boolhost;
f64 * p_temphost1, *p_temphost2, *p_temphost3, *p_temphost4, *p_temphost5, *p_temphost6;
long * p_longtemphost;
f64_vec2 * p_GradTe_host, *p_GradAz_host;
f64_vec3 * p_B_host, *p_MAR_ion_host, *p_MAR_elec_host, *p_MAR_ion_compare, *p_MAR_elec_compare,
*p_MAR_neut_host,*p_MAR_neut_compare;
__device__ nn *p_nn_ionrec_minor;
__device__ OhmsCoeffs * p_OhmsCoeffs;
OhmsCoeffs * p_OhmsCoeffs_host; // for display
__device__ f64 * p_Iz0, *p_sigma_Izz;
__device__ f64_vec3 * p_vn0;
__device__ v4 * p_v0;
__device__ nvals * p_one_over_n, *p_one_over_n2;
__device__ f64_vec3 * p_MAR_neut, *p_MAR_ion, *p_MAR_elec;
__device__ f64 * p_Az, *p_LapAz, *p_LapCoeffself, *p_Azdot0, *p_gamma, *p_LapJacobi,
*p_Jacobi_x, *p_epsilon, *p_LapAzNext,
*p_Integrated_div_v_overall,
*p_Div_v_neut, *p_Div_v, *p_Div_v_overall, *p_ROCAzdotduetoAdvection,
*p_ROCAzduetoAdvection, *p_AzNext,
*p_kappa_n,*p_kappa_i,*p_kappa_e,*p_nu_i,*p_nu_e;

__device__ f64 * p_epsilon_heat, *p_Jacobi_heat,
				*p_sum_eps_deps_by_dbeta_heat, *p_sum_depsbydbeta_sq_heat, *p_sum_eps_eps_heat;
f64  *p_sum_eps_deps_by_dbeta_host_heat, *p_sum_depsbydbeta_sq_host_heat, *p_sum_eps_eps_host_heat;

__device__ species3 *p_nu_major;
__device__ f64_vec2 * p_GradAz, *p_GradTe;
__device__ ShardModel *p_n_shards, *p_n_shards_n;
__device__ NTrates *NT_addition_rates_d, *NT_addition_tri_d;
long numReverseJzTriangles;
__device__ f64 *p_sum_eps_deps_by_dbeta, *p_sum_depsbydbeta_sq, *p_sum_eps_eps;
f64  *p_sum_eps_deps_by_dbeta_host, *p_sum_depsbydbeta_sq_host, *p_sum_eps_eps_host;
__device__ char * p_was_vertex_rotated, *p_triPBClistaffected;
__device__ T3 * p_T_upwind_minor;

__device__ f64 * p_d_eps_by_dbeta_n, *p_d_eps_by_dbeta_i, *p_d_eps_by_dbeta_e,
*p_Jacobi_n, *p_Jacobi_i, *p_Jacobi_e, *p_epsilon_n, *p_epsilon_i, *p_epsilon_e,
*p_coeffself_n, *p_coeffself_i, *p_coeffself_e; // Are these fixed or changing with each iteration?

__device__ f64_tens3 * p_InvertedMatrix_i, *p_InvertedMatrix_e;
__device__ f64_vec3 * p_MAR_ion2, *p_MAR_elec2, * p_vJacobi_i, * p_vJacobi_e,
	* p_d_eps_by_d_beta_i, *p_d_eps_by_d_beta_e;
__device__ NTrates * NT_addition_rates_d_temp;
__device__ f64_vec2 * p_epsilon_xy;
__device__ f64 * p_epsilon_iz, *p_epsilon_ez,
*p_sum_eps_deps_by_dbeta_i, *p_sum_eps_deps_by_dbeta_e, *p_sum_depsbydbeta_i_times_i,
*p_sum_depsbydbeta_e_times_e, *p_sum_depsbydbeta_e_times_i;
f64 * p_sum_eps_deps_by_dbeta_i_host, * p_sum_eps_deps_by_dbeta_e_host, * p_sum_depsbydbeta_i_times_i_host,
*p_sum_depsbydbeta_e_times_e_host, *p_sum_depsbydbeta_e_times_i_host;


f64 * temp_array_host;
f64 tempf64;
FILE * fp_traj;
 
//f64 Tri_n_n_lists[NMINOR][6],Tri_n_lists[NMINOR][6];
// Not clear if I ended up using Tri_n_n_lists - but it was a workable way if not.

long * address;
f64 * f64address;
size_t uFree, uTotal;
extern real evaltime;

void RunBackwardJLSForHeat(T3 * p_T_k, T3 * p_T, f64 hsub, cuSyst * pX_use)
{
	// 1. Calculate nu_k on triangles (or 1/nu_k), B etc, really we want n/nu t.y.v.m.
	// We are going to create kappa_par = n T/nu m on the fly.
	// 2. Calculate epsilon: given the est of T, eps = T - (T_k +- h sum kappa dot grad T)
	// 4. To determine deps/dbeta, ie the change in eps for an addition of delta Jacobi_heat to T,
	// 5. Do JLS calcs and update T
	// Report if we are getting closer. Check that L2eps is actually decreasing. 
	// If not do a search for beta that does decrease L2 as well as we can.
	
	// 1. Calculate nu_k on triangles (or n/nu_k), B etc -- this is a given from T_k.
	// In ptic, for ionized version we get T_k^1.5 vs T_k+1^1. Seems fine.
	// But for weakly ionized, we get T_k^-0.5 vs T_k+1^1. Does that seem bad? Actually due to visc cs it is indeterminate effect of T in denominator.

	f64 beta_e, beta_i, beta_n;
	long iTile;

	// seed: just set T to T_k.
	cudaMemcpy(p_T, p_T_k, sizeof(T3)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	
	// JLS:

	f64 sum_eps_deps_by_dbeta, sum_depsbydbeta_sq, sum_eps_eps, depsbydbeta;
	printf("\nJLS [beta L2eps]: ");
	long iMinor;
	f64 beta, L2eps;
	Triangle * pTri;
	int iIteration;
	for (iIteration = 0; iIteration < NUM_BWD_ITERATIONS; iIteration++)
	{		
		// 2. Calculate epsilon: given the est of T, eps = T - (T_k +- h sum kappa dot grad T)
		// 3. Calculate Jacobi: for each point, Jacobi = eps/(deps/dT)
		
		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);

		kernelAccumulateDiffusiveHeatRate_new << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(
				hsub,
				pX_use->p_info,
				pX_use->p_izNeigh_vert,
				pX_use->p_szPBCneigh_vert,
				pX_use->p_izTri_vert,
				pX_use->p_szPBCtri_vert,
				pX_use->p_cc,
				pX_use->p_n_major,
				p_T, // using vert indices
				pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
				p_kappa_n,
				p_kappa_i,
				p_kappa_e,
				p_nu_i,
				p_nu_e,
				NT_addition_rates_d_temp,
				pX_use->p_AreaMajor);
		// used for epsilon (T)
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

		// Better if we would just load in epsilon and create Jacobi? No coeffself array, simpler.
		// Careful: if we find coeffself in d/dt NT then eps <- -(h/N) d/dtNT
		// so we need N as well.
		kernelCalc_SelfCoefficient_for_HeatConduction <<< numTilesMajorClever, threadsPerTileMajorClever >> >
			(
				hsub,
				pX_use->p_info, // minor
				pX_use->p_izNeigh_vert,
				pX_use->p_szPBCneigh_vert,
				pX_use->p_izTri_vert,
				pX_use->p_szPBCtri_vert,
				pX_use->p_cc,
				pX_use->p_n_major,
				p_T, // using vert indices
				pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
				p_kappa_n,
				p_kappa_i,
				p_kappa_e,
				p_nu_i,
				p_nu_e,
				pX_use->p_AreaMajor,

				p_coeffself_n,
				p_coeffself_i,
				p_coeffself_e, // what exactly it calculates?
				1.0); // used for Jacobi
		Call(cudaThreadSynchronize(), "cudaTS CalcSelfCoefficientForHeatConduction");
		
		// Splitting up routines will probably turn out better although it's tempting to combine.

		// Given putative ROC and coeff on self, it is simple to calculate eps and Jacobi. So do so:
		kernelCreateEpsilonAndJacobi_Heat << <numTilesMajor, threadsPerTileMajor >> > (
			// eps = T - (T_k +- h sum kappa dot grad T)
			// x = -eps/coeffself
			hsub,
			pX_use->p_info + BEGINNING_OF_CENTRAL,
			p_T,
			p_T_k,
			NT_addition_rates_d_temp,
			// NEED N = n*AreaMajor
			pX_use->p_n_major, // got this
			pX_use->p_AreaMajor, // got this -> N, Nn

			p_coeffself_n,
			p_coeffself_i,
			p_coeffself_e,

			p_epsilon_n, p_epsilon_i, p_epsilon_e,
			p_Jacobi_n,
			p_Jacobi_i,
			p_Jacobi_e
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");
		
		// 4. To determine deps/dbeta, ie the change in eps for an addition of delta Jacobi_heat to T,
		
		// eps = T - (T_k + h dT/dt)
		// deps/dbeta[index] = Jacobi[index] - h d(dT/dt)/d [increment whole field by Jacobi]
		// ####################################
		// Note this last, it's very important.
		
		cudaMemset(p_d_eps_by_dbeta_n, 0, sizeof(f64)*NUMVERTICES);
		cudaMemset(p_d_eps_by_dbeta_i, 0, sizeof(f64)*NUMVERTICES);
		cudaMemset(p_d_eps_by_dbeta_e, 0, sizeof(f64)*NUMVERTICES);
		
		kernelCalculateROCepsWRTregressorT <<< numTilesMajorClever, threadsPerTileMajorClever >>>(
			hsub, 
			pX_use->p_info,
			pX_use->p_izNeigh_vert,
			pX_use->p_szPBCneigh_vert,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_cc,
			pX_use->p_n_major, // got this
			p_T,
			pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
			p_kappa_n,
			p_kappa_i,
			p_kappa_e,
			p_nu_i,
			p_nu_e,

			pX_use->p_AreaMajor, // got this -> N, Nn
			 
			p_Jacobi_n,
			p_Jacobi_i,
			p_Jacobi_e,
			p_d_eps_by_dbeta_n,
			p_d_eps_by_dbeta_i,
			p_d_eps_by_dbeta_e
		);
		
		Call(cudaThreadSynchronize(), "cudaTS kernelCalculateROCepsWRTregressorT WW");
		 
		// 5. Do JLS calcs and update T
		// But we have to do for 3 species: do separately, so scalar arrays.
		
		kernelAccumulateSummands2 << <numTilesMajor, threadsPerTileMajor >> > (
			pX_use->p_info, 
			p_epsilon_n, 
			p_d_eps_by_dbeta_n,
			p_sum_eps_deps_by_dbeta,
			p_sum_depsbydbeta_sq,
			p_sum_eps_eps);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands OO");
		cudaMemcpy(p_sum_depsbydbeta_sq_host, p_sum_depsbydbeta_sq, sizeof(f64)*numTilesMajor,
			cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_deps_by_dbeta_host, p_sum_eps_deps_by_dbeta, sizeof(f64)*numTilesMajor,
			cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajor,
			cudaMemcpyDeviceToHost);
		
		sum_eps_deps_by_dbeta = 0.0;
		sum_depsbydbeta_sq = 0.0;
		sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMajor; iTile++)
		{
			sum_eps_deps_by_dbeta += p_sum_eps_deps_by_dbeta_host[iTile];
			sum_depsbydbeta_sq += p_sum_depsbydbeta_sq_host[iTile];
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		}
	
		f64 tempf64 = (real)NUMVERTICES;
		printf("sum_depsbydbeta_sq %1.10E (real)NUMVERTICES %1.10E sum_eps_eps %1.10E \n", sum_depsbydbeta_sq, tempf64);
		beta_n = -sum_eps_deps_by_dbeta / sum_depsbydbeta_sq;
		L2eps = sqrt(sum_eps_eps / tempf64);
		
		printf("\n for neutral [ %1.14E %1.14E ] ", beta_n, L2eps);
		
		kernelAccumulateSummands2 << <numTilesMajor, threadsPerTileMajor >> > (
			pX_use->p_info,

			p_epsilon_i,
			p_d_eps_by_dbeta_i,

			p_sum_eps_deps_by_dbeta,
			p_sum_depsbydbeta_sq,
			p_sum_eps_eps);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands 1a");

		cudaMemcpy(p_sum_depsbydbeta_sq_host, p_sum_depsbydbeta_sq, sizeof(f64)*numTilesMajor,
			cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_deps_by_dbeta_host, p_sum_eps_deps_by_dbeta, sizeof(f64)*numTilesMajor,
			cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajor,
			cudaMemcpyDeviceToHost);
		sum_eps_deps_by_dbeta = 0.0;
		sum_depsbydbeta_sq = 0.0;
		sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMajor; iTile++)
		{
			sum_eps_deps_by_dbeta += p_sum_eps_deps_by_dbeta_host[iTile];
			sum_depsbydbeta_sq += p_sum_depsbydbeta_sq_host[iTile];
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		}
		beta_i = -sum_eps_deps_by_dbeta / sum_depsbydbeta_sq;
		L2eps = sqrt(sum_eps_eps / (real)NUMVERTICES);

		printf("\n for ion [ %1.14E %1.14E ] ", beta_i, L2eps);
		
		kernelAccumulateSummands2 << <numTilesMajor, threadsPerTileMajor >> > (
			pX_use->p_info,

			p_epsilon_e,
			p_d_eps_by_dbeta_e,

			p_sum_eps_deps_by_dbeta,
			p_sum_depsbydbeta_sq,
			p_sum_eps_eps);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands 1");

		cudaMemcpy(p_sum_depsbydbeta_sq_host, p_sum_depsbydbeta_sq, sizeof(f64)*numTilesMajor,
			cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_deps_by_dbeta_host, p_sum_eps_deps_by_dbeta, sizeof(f64)*numTilesMajor,
			cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajor,
			cudaMemcpyDeviceToHost);
		sum_eps_deps_by_dbeta = 0.0;
		sum_depsbydbeta_sq = 0.0;
		sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMajor; iTile++)
		{
			sum_eps_deps_by_dbeta += p_sum_eps_deps_by_dbeta_host[iTile];
			sum_depsbydbeta_sq += p_sum_depsbydbeta_sq_host[iTile];
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		}
		beta_e = -sum_eps_deps_by_dbeta / sum_depsbydbeta_sq;
		L2eps = sqrt(sum_eps_eps / (real)NUMVERTICES);

		printf("\n for elec [ %1.14E %1.14E ] ", beta_e, L2eps);
		 
		// maybe do the add after we calc beta_n, beta_i, beta_e.
		kernelAddtoT << <numTilesMajor, threadsPerTileMajor >> > (
			p_T, beta_n, beta_i, beta_e, p_Jacobi_n, p_Jacobi_i, p_Jacobi_e);
		
		// For some reason, beta calculated was the opposite of the beta that was needed.
		// Don't know why... only explanation is that we are missing a - somewhere in deps/dbeta
		// That's probably what it is? Ought to verify.

		Call(cudaThreadSynchronize(), "cudaTS AddtoT ___");		
		
	};
}


void RunBackwardJLSForViscosity(v4 * p_vie_k, v4 * p_vie, f64 const hsub, cuSyst * pX_use)
// BE SURE ABOUT PARAMETER ORDER -- CHECK IT CHECK IT
{
	// 1. Calculate nu_k on triangles (or 1/nu_k), B etc, really we want n/nu t.y.v.m.
	// We are going to create kappa_par = n T/nu m on the fly.
	// 2. Calculate epsilon: given the est of T, eps = T - (T_k +- h sum kappa dot grad T)
	// 4. To determine deps/dbeta, ie the change in eps for an addition of delta Jacobi_heat to T,
	// 5. Do JLS calcs and update T
	// Report if we are getting closer. Check that L2eps is actually decreasing. 
	// If not do a search for beta that does decrease L2 as well as we can.

	// 1. Calculate nu_k on triangles (or n/nu_k), B etc -- this is a given from T_k.
	// In ptic, for ionized version we get T_k^1.5 vs T_k+1^1. Seems fine.
	// But for weakly ionized, we get T_k^-0.5 vs T_k+1^1. Does that seem bad? Actually due to visc cs it is indeterminate effect of T in denominator.

	f64 beta_e, beta_i, beta_n;
	long iTile;

	// seed: just set T to T_k.
	cudaMemcpy(p_vie, p_vie_k, sizeof(v4)*NMINOR, cudaMemcpyDeviceToDevice);

	// JLS:

	f64 sum_eps_deps_by_dbeta, sum_depsbydbeta_sq, sum_eps_eps, depsbydbeta;
	printf("\nJLS [beta L2eps]: ");
	long iMinor;
	f64 beta, L2eps;

	// Do outside:
	//kernelCalculate_ita_visc << <numTilesMinor, threadsPerTileMinor >> >(
	//	pX_use->p_info,
	//	pX_use->p_n_minor,
	//	pX_use->p_T_minor,
	//	p_temp3,	//	p_temp4,	//	p_temp5,	//	p_temp1,	//	p_temp2,	//	p_temp6);
	//Call(cudaThreadSynchronize(), "cudaTS ita 1");

	int iIteration;

	kernelCalc_Matrices_for_Jacobi_Viscosity << < numTriTiles, threadsPerTileMinor >> >//SelfCoefficient
			(
			hsub,
			pX_use->p_info, // minor
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,

			p_temp1, p_temp2, p_temp3, p_temp4,
			pX_use->p_B,
			pX_use->p_n_minor, // eps += -h/N * MAR; thus N features in self coefficient
			pX_use->p_AreaMinor,
			p_InvertedMatrix_i,
			p_InvertedMatrix_e
			); // don't forget +1 in self coefficient
	Call(cudaThreadSynchronize(), "cudaTS kernelCalc_Jacobi_for_Viscosity");
	
	for (iIteration = 0; iIteration < NUM_BWD_ITERATIONS; iIteration++)
	{
		// ***************************************************
		// Requires averaging of n,T to triangles first. & ita
		// ***************************************************

		cudaMemset(p_MAR_ion2, 0, sizeof(f64_vec3)*NMINOR);
		cudaMemset(p_MAR_elec2, 0, sizeof(f64_vec3)*NMINOR);

		cudaMemset(NT_addition_tri_d, 0, sizeof(NTrates)*NUMVERTICES * 2);
		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
		kernelCreate_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> >(

			pX_use->p_info,
			p_vie,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,

			p_temp1, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
			p_temp2, //f64 * __restrict__ p_ita_parallel_elec_minor,   // nT / nu ready to look up
			p_temp3, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
			p_temp4, // f64 * __restrict__ p_nu_elec_minor,   // nT / nu ready to look up

			pX_use->p_B,
			p_MAR_ion2, // just accumulates
			p_MAR_elec2,
			NT_addition_rates_d_temp, 
				// Again need to accumulate on to the existing one, the one here needs to start from zero each time probably
			NT_addition_tri_d);
		Call(cudaThreadSynchronize(), "cudaTS visccontrib ~~");
		
		// Given putative ROC and coeff on self, it is simple to calculate eps and Jacobi. So do so:
		kernelCreateEpsilon_Visc << <numTilesMinor, threadsPerTileMinor >> > (
			// eps = v - (v_k +- h [viscous effect])
			// x = -eps/coeffself
			hsub,
			pX_use->p_info,
			p_vie,
			p_vie_k,
			p_MAR_ion2, p_MAR_elec2,

			pX_use->p_n_minor,
			pX_use->p_AreaMinor,

			p_epsilon_xy, 
			p_epsilon_iz, 
			p_epsilon_ez  
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");

		kernelMultiply_Get_Jacobi_Visc << < numTilesMinor, threadsPerTileMinor >> >//SelfCoefficient
			(
				pX_use->p_info,
				p_epsilon_xy, // input
				p_epsilon_iz, // input
				p_epsilon_ez, // input
				p_InvertedMatrix_i,
				p_InvertedMatrix_e,
					// output:
				p_vJacobi_i, // 3-vec array
				p_vJacobi_e  // 3-vec array	= InvertedMatrix epsilon
				);
		Call(cudaThreadSynchronize(), "cudaTS kernelGetJacobi");

		// 4. To determine deps/dbeta, ie the change in eps for an addition of delta Jacobi_heat to T,

		// eps = v - (v_k + h [viscous effect] + h [known increment rate of v])
		// deps/dbeta[index] = Jacobi[index] - h d(dT/dt)/d [increment whole field by Jacobi]
		// ####################################
		// Note this last, it's very important.

		cudaMemset(p_d_eps_by_d_beta_i, 0, sizeof(f64_vec3)*NMINOR); // unused
		cudaMemset(p_d_eps_by_d_beta_e, 0, sizeof(f64_vec3)*NMINOR);
		
		// We have 4 epsilons and 1 beta
		kernelCalculate_deps_WRT_beta_Visc << < numTriTiles, threadsPerTileMinor >> >(
			hsub,
			pX_use->p_info,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,

			p_temp1, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
			p_temp2, //f64 * __restrict__ p_ita_parallel_elec_minor,   // nT / nu ready to look up
			p_temp3, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
			p_temp4, // f64 * __restrict__ p_nu_elec_minor,   // nT / nu ready to look up
			pX_use->p_B,

			pX_use->p_n_minor, // got this
			pX_use->p_AreaMinor, // got this -> N, Nn

			p_vJacobi_i, // 3-vec
			p_vJacobi_e, // 3-vec

			p_d_eps_by_d_beta_i,
			p_d_eps_by_d_beta_e
			);

		Call(cudaThreadSynchronize(), "cudaTS kernelCalculateROCepsWRTregressorT WW");
		
		// Yeah. I am not 100% if it wouldn't be better to have 3 or 4 beta's. Maybe combine x & y.
		
		// 5. Do JLS calcs and update T
		// But we have to do for 3 species: do separately, so scalar arrays.

		kernelAccumulateSummands3 << <numTilesMinor, threadsPerTileMinor >> > (
			pX_use->p_info,

			p_epsilon_xy,
			p_epsilon_iz,
			p_epsilon_ez,
			
			p_d_eps_by_d_beta_i,
			p_d_eps_by_d_beta_e,
			
			// 6 outputs:
			p_sum_eps_deps_by_dbeta_i,
			p_sum_eps_deps_by_dbeta_e,
			p_sum_depsbydbeta_i_times_i,
			p_sum_depsbydbeta_e_times_e,
			p_sum_depsbydbeta_e_times_i,
			p_sum_eps_eps);

		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands 1aa");

		cudaMemcpy(p_sum_eps_deps_by_dbeta_i_host, p_sum_eps_deps_by_dbeta_i, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_deps_by_dbeta_e_host, p_sum_eps_deps_by_dbeta_e, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_i_times_i_host, p_sum_depsbydbeta_i_times_i, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_e_times_e_host, p_sum_depsbydbeta_e_times_e, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_e_times_i_host, p_sum_depsbydbeta_e_times_i, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		
		f64 sum_eps_deps_by_dbeta_i = 0.0;
		f64 sum_eps_deps_by_dbeta_e = 0.0;
		f64 sum_depsbydbeta_i_times_i = 0.0;
		f64 sum_depsbydbeta_e_times_e = 0.0;
		f64 sum_depsbydbeta_e_times_i = 0.0;
		f64 sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMinor; iTile++)
		{
			sum_eps_deps_by_dbeta_i += p_sum_eps_deps_by_dbeta_i_host[iTile];
			sum_eps_deps_by_dbeta_e += p_sum_eps_deps_by_dbeta_e_host[iTile];
			sum_depsbydbeta_i_times_i += p_sum_depsbydbeta_i_times_i_host[iTile];
			sum_depsbydbeta_e_times_e += p_sum_depsbydbeta_e_times_e_host[iTile];
			sum_depsbydbeta_e_times_i += p_sum_depsbydbeta_e_times_i_host[iTile];
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		};
		
		if ((sum_eps_eps == 0.0) || ((sum_depsbydbeta_i_times_i*sum_depsbydbeta_e_times_e - sum_depsbydbeta_e_times_i*sum_depsbydbeta_e_times_i) == 0.0))
			return;
		
		beta_i = -(sum_eps_deps_by_dbeta_i*sum_depsbydbeta_e_times_e - sum_eps_deps_by_dbeta_e*sum_depsbydbeta_e_times_i)/
				  (sum_depsbydbeta_i_times_i*sum_depsbydbeta_e_times_e - sum_depsbydbeta_e_times_i*sum_depsbydbeta_e_times_i);
		beta_e = -(sum_eps_deps_by_dbeta_e*sum_depsbydbeta_i_times_i - sum_eps_deps_by_dbeta_i*sum_depsbydbeta_e_times_i)/
				  (sum_depsbydbeta_e_times_e*sum_depsbydbeta_i_times_i - sum_depsbydbeta_e_times_i*sum_depsbydbeta_e_times_i);

		L2eps = sqrt(sum_eps_eps / (real)NMINOR);
		printf("\n visc : for ion [ beta_i %1.14E beta_e %1.14E L2eps %1.14E ] ", beta_i, beta_e, L2eps);
		
		// maybe do the add after we calc beta_n, beta_i, beta_e.
		kernelAdd_to_v << <numTilesMinor, threadsPerTileMinor >> > (
			p_vie, beta_i, beta_e, p_vJacobi_i, p_vJacobi_e);
		Call(cudaThreadSynchronize(), "cudaTS Addtov ___");
			
	};

	// Do after calling and recalc'ing MAR:

	//Collect_Nsum_at_tris << <numTriTiles, threadsPerTileMinor >> >(
	//	this->p_info,
	//	this->p_n_minor,
	//	this->p_tri_corner_index,
	//	this->p_AreaMajor, // populated?
	//	p_temp4);
	//Call(cudaThreadSynchronize(), "cudaTS Nsum");

	//kernelTransmitHeatToVerts << <numTilesMajor, threadsPerTileMajor >> > (
	//	this->p_info,
	//	this->p_izTri_vert,
	//	this->p_n_minor,
	//	this->p_AreaMajor,
	//	p_temp4,
	//	NT_addition_rates_d,
	//	NT_addition_tri_d
	//	);
	//Call(cudaThreadSynchronize(), "cudaTS sum up heat 1");

}

int Compare_f64_vec2(f64_vec2 * p1, f64_vec2 * p2, long N);
int Compare_n_shards(ShardModel * p1, ShardModel * p2, const cuSyst * p_cuSyst_host)
{

	f64 maxdiff = 0.0;
	f64 mindiff = 0.0;
	f64 maxreldiff = 0.0;
	long iMin = -1;
	long iMax = -1;
	long iMaxRel = -1; 
	long i;
	for (i = 0; i < NUMVERTICES; i++)
	{
		f64 diff = (p1[i].n_cent - p2[i].n_cent);
		if (diff > maxdiff) { maxdiff = diff; iMax = i; }
		if (diff < mindiff) { mindiff = diff; iMin = i; }
		if (p1[i].n_cent != 0.0) {
			f64 reldiff = fabs(diff / p1[i].n_cent);
			if (reldiff > maxreldiff) {
				maxreldiff = reldiff;
				iMaxRel = i;
			}
		};
	};
	if (iMax != -1) {
		printf(" Max  cent diff: %1.3E at %d : %1.12E %1.12E \n",
			maxdiff, iMax, p1[iMax].n_cent, p2[iMax].n_cent);
	}
	else {
		printf(" Max diff == zero \n");
	};
	if (iMin != -1) {
		printf(" Min diff: %1.3E at %d : %1.12E %1.12E \n",
			mindiff, iMin, p1[iMin].n_cent, p2[iMin].n_cent);
	}
	else {
		printf(" Min diff == zero \n");
	};
	if (iMaxRel != -1) {
		printf(" Max rel diff %1.3E at %d : %1.12E %1.12E \n",
			maxreldiff, iMaxRel, p1[iMaxRel].n_cent, p2[iMaxRel].n_cent);
	}
	else {
		printf(" Max rel diff == zero \n");
	}

	maxdiff = 0.0;
	mindiff = 0.0;
	maxreldiff = 0.0;
	iMin = -1;
	iMax = -1;
	iMaxRel = -1;
	f64 diff;
	int j; f64 diff_;
	for (i = 0; i < NUMVERTICES; i++)
	{
		diff = 0.0;
		short neigh_len = p_cuSyst_host->p_info[i + BEGINNING_OF_CENTRAL].neigh_len;
		for (j = 0; j < neigh_len; j++)
		{
			diff_ = fabs(p1[i].n[j] - p2[i].n[j]);
			if (diff_ > diff) diff = diff_;
		}
		if (diff > maxdiff) { maxdiff = diff; iMax = i; }
		if (diff < mindiff) { mindiff = diff; iMin = i; }
		if (p1[i].n_cent != 0.0) {
			f64 reldiff = fabs(diff / p1[i].n_cent);
			if (reldiff > maxreldiff) {
				maxreldiff = reldiff;
				iMaxRel = i;
			}
		};
	};
	if (iMax != -1) {
		printf(" Max fabs [--] diff: %1.4E at %d ; n_cent = %1.10E \n",
			maxdiff, iMax,p1[iMax].n_cent);
	} else {
		printf(" Max diff == zero \n");
	};

	if (iMaxRel != -1) {
		printf(" Max rel diff %1.4E at %d ; n_cent =  %1.12E %1.12E \n",
			maxreldiff, iMaxRel, p1[iMaxRel].n_cent, p2[iMaxRel].n_cent);
	} else {
		printf(" Max rel diff == zero \n");
	};
	return 0;
}
 
int Compare_f64(f64 * p1, f64 * p2, long N);
int Compare_NTrates(NTrates * p1, NTrates * p2)
{
	f64 temp1[NUMVERTICES], temp2[NUMVERTICES];
	long iVertex;
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		temp1[iVertex] = p1[iVertex].N;
		temp2[iVertex] = p2[iVertex].N;
	}
	printf("N:\n");
	Compare_f64(temp1, temp2, NUMVERTICES);
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		temp1[iVertex] = p1[iVertex].Nn;
		temp2[iVertex] = p2[iVertex].Nn;
	}  
	printf("Nn:\n");
	Compare_f64(temp1, temp2, NUMVERTICES);
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		temp1[iVertex] = p1[iVertex].NiTi;
		temp2[iVertex] = p2[iVertex].NiTi;
	}
	printf("NiTi:\n");
	Compare_f64(temp1, temp2, NUMVERTICES);
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		temp1[iVertex] = p1[iVertex].NeTe;
		temp2[iVertex] = p2[iVertex].NeTe;
	}
	printf("NeTe:\n");
	Compare_f64(temp1, temp2, NUMVERTICES);
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	{
		temp1[iVertex] = p1[iVertex].NnTn;
		temp2[iVertex] = p2[iVertex].NnTn;
	}
	printf("NnTn:\n");
	Compare_f64(temp1, temp2, NUMVERTICES);
	return 0;
}
int Compare_f64(f64 * p1, f64 * p2, long N)
{
	// Arithmetic difference:

	f64 maxdiff = 0.0;
	f64 mindiff = 0.0;
	f64 maxreldiff = 0.0;
	long iMin = -1;
	long iMax = -1;
	long iMaxRel = -1;
	long i;
	for (i = 0; i < N; i++)
	{
		f64 diff = (p1[i] - p2[i]);
		if (diff > maxdiff) { maxdiff = diff; iMax = i; }
		if (diff < mindiff) { mindiff = diff; iMin = i; }

		// Relative difference:
		if (p1[i] != 0.0) {
			f64 reldiff = fabs(diff / p1[i]);
			if (reldiff > maxreldiff) {
				maxreldiff = reldiff;
				iMaxRel = i;
			}
		};
	};
	if (iMax != -1) {
		printf(" Max diff: %1.3E at %d : %1.12E %1.12E \n",
			maxdiff, iMax, p1[iMax], p2[iMax]);
	} else {
		printf(" Max diff == zero \n");
	};
	if (iMin != -1) {
		printf(" Min diff: %1.3E at %d : %1.12E %1.12E \n",
			mindiff, iMin, p1[iMin], p2[iMin]);
	} else {
		printf(" Min diff == zero \n");
	};
	if (iMaxRel != -1) {
		printf(" Max rel diff %1.3E at %d : %1.12E %1.12E \n",
			maxreldiff, iMaxRel, p1[iMaxRel], p2[iMaxRel]);
	} else {
		printf(" Max rel diff == zero \n");
	}
	return 0;
}

int Compare_nvals(nvals * p1, nvals * p2, long N)
{
	// Arithmetic difference:
	f64 *n1 = (f64 *)malloc(N * sizeof(f64));
	f64 *n2 = (f64 *)malloc(N * sizeof(f64));
	if (n2 != 0) {
		long i;
		printf("n:\n");
		for (i = 0; i < N; i++)
		{
			n1[i] = p1[i].n;
			n2[i] = p2[i].n;
		}
		Compare_f64(n1, n2, N);
		printf("n_n:\n");
		for (i = 0; i < N; i++)
		{
			n1[i] = p1[i].n_n;
			n2[i] = p2[i].n_n;
		}
		Compare_f64(n1, n2, N);
		free(n1);
		free(n2);
		return 0;
	}
	else {
		printf("memory error.");
		return 1;
	}
}
int Compare_T3(T3 * p1, T3 * p2, long N)
{
	// Arithmetic difference:
	f64 *n1 = (f64 *)malloc(N * sizeof(f64));
	f64 *n2 = (f64 *)malloc(N * sizeof(f64));
	if (n2 != 0) {
		long i;
		printf("Tn:\n");
		for (i = 0; i < N; i++)
		{
			n1[i] = p1[i].Tn;
			n2[i] = p2[i].Tn;
		}
		Compare_f64(n1, n2, N);
		printf("Ti:\n");
		for (i = 0; i < N; i++)
		{
			n1[i] = p1[i].Ti;
			n2[i] = p2[i].Ti;
		}
		Compare_f64(n1, n2, N);
		printf("Te:\n");
		for (i = 0; i < N; i++)
		{
			n1[i] = p1[i].Te;
			n2[i] = p2[i].Te;
		}
		Compare_f64(n1, n2, N);
		free(n1);
		free(n2);
		return 0;
	}
	else {
		printf("memory error.");
		return 1;
	}
}
int Compare_structural(structural * p1, structural * p2, long N)
{
	f64_vec2 *pos1 = (f64_vec2 *)malloc(N * sizeof(f64_vec2));
	f64_vec2 *pos2 = (f64_vec2 *)malloc(N * sizeof(f64_vec2));
	
	if (pos2 != 0) {
		long i;
		printf("pos:\n");
		for (i = 0; i < N; i++)
		{
			pos1[i] = p1[i].pos;
			pos2[i] = p2[i].pos;
		}
		Compare_f64_vec2(pos1, pos2, N);
		
		free(pos1);
		free(pos2);
	}
	else {
		printf("memory error.");
		return 1;
	}
	bool bFailneighlen = false, bFailflag = false; // has_periodic: never used?
	long iFailflag_start, iFailneighlen_start, iFailflag, iFailneigh_len;
	long i;
	for (i = 0; i < N; i++)
	{
		if (p1[i].neigh_len != p2[i].neigh_len) {
			if (bFailneighlen == false) iFailneighlen_start = i;
			bFailneighlen = true; 
			iFailneigh_len = i;
		}
		if (p1[i].flag != p2[i].flag) {
			if (bFailflag == false) iFailflag_start = i;
			bFailflag = true; iFailflag = i;
		}
	}
	if (bFailneighlen) printf("Start of inconsistent neigh_len: %d end: %d\n",
		iFailneighlen_start, iFailneigh_len);
	if (bFailflag) printf("Start of inconsistent flag: %d end : %d\n",
		iFailflag, iFailflag_start);
}
 
int Compare_f64_vec2(f64_vec2 * p1, f64_vec2 * p2, long N)
{
	f64 maxdiff = 0.0;
	f64 maxreldiff = 0.0;
	long iMin = -1;
	long iMax = -1;
	long iMaxRel = -1;
	long i;
	for (i = 0; i < N; i++)
	{
		f64 diffmod = sqrt((p1[i].x - p2[i].x)*(p1[i].x-p2[i].x)
						+ (p1[i].y - p2[i].y)*(p1[i].y - p2[i].y));
		if (diffmod > maxdiff) { maxdiff = diffmod; iMax = i; }
		
		// Relative difference:
		if ((p1[i].x != 0.0) || (p1[i].y != 0.0)) {
			f64 reldiff = diffmod / p1[i].modulus();
			if (reldiff > maxreldiff) {
				maxreldiff = reldiff;
				iMaxRel = i;
			}
		};
	};
	if (iMax != -1) {
		printf(" Max diff mod: %1.3E at %d : x %1.12E diff %1.3E y %1.12E diff %1.3E\n",
			maxdiff, iMax, p1[iMax].x, p2[iMax].x - p1[iMax].x, p1[iMax].y, p2[iMax].y - p1[iMax].y);
	} else {
		printf(" Max diff == zero. \n");
	}
	if (iMaxRel != -1) {
		printf(" Max rel diff mod: %1.3E at %d : x %1.12E diff %1.3E y %1.12E diff %1.3E\n",
			maxreldiff, iMaxRel, p1[iMaxRel].x, p2[iMaxRel].x - p1[iMaxRel].x, p1[iMaxRel].y, p2[iMaxRel].y - p1[iMaxRel].y);
	} else {
		printf(" Max rel diff zero / not found. \n");
	}
	return 0;
}

int Compare_f64_vec3(f64_vec3 * p1, f64_vec3 * p2, long N)
{
	f64 maxdiff = 0.0;
	f64 maxreldiff = 0.0;
	long iMin = -1;
	long iMax = -1;
	long iMaxRel = -1;
	long i;
	for (i = 0; i < N; i++)
	{
		f64 diffmod = sqrt((p1[i].x - p2[i].x)*(p1[i].x - p2[i].x)
			+ (p1[i].y - p2[i].y)*(p1[i].y - p2[i].y) + (p1[i].z - p2[i].z)*(p1[i].z - p2[i].z));
		if (diffmod > maxdiff) { maxdiff = diffmod; iMax = i; }

		// Relative difference:
		if ((p1[i].x != 0.0) || (p1[i].y != 0.0)) {
			f64 reldiff = diffmod / p1[i].modulus();
			if (reldiff > maxreldiff) {
				maxreldiff = reldiff;
				iMaxRel = i;
			}
		};
	};
	if (iMax != -1) {
		printf(" Max diff mod: %1.3E at %d : x %1.12E diff %1.3E y %1.12E diff %1.3E z %1.12E diff %1.3E\n",
			maxdiff, iMax, p1[iMax].x, p2[iMax].x - p1[iMax].x, p1[iMax].y, p2[iMax].y - p1[iMax].y,
			p1[iMax].z, p2[iMax].z - p1[iMax].z);
	}
	else {
		printf(" Max diff == zero. \n");
	}
	if (iMaxRel != -1) {
		printf(" Max rel diff mod: %1.3E at %d : x %1.12E diff %1.3E y %1.12E diff %1.3E z %1.12E diff %1.3E\n",
			maxreldiff, iMaxRel, p1[iMaxRel].x, p2[iMaxRel].x - p1[iMaxRel].x, p1[iMaxRel].y, p2[iMaxRel].y - p1[iMaxRel].y,
			p1[iMaxRel].z, p2[iMaxRel].z - p1[iMaxRel].z);
	}
	else {
		printf(" Max rel diff zero / not found. \n");
	}
	return 0;
}
 
real GetIzPrescribed(real const t)
{
	real Iz = -PEAKCURRENT_STATCOULOMB * sin((t + ZCURRENTBASETIME) * 0.5* PIOVERPEAKTIME); // half pi / peaktime

	//printf("\nGetIzPrescribed : t + ZCURRENTBASETIME = %1.5E : %1.12E\n", t + ZCURRENTBASETIME, Iz);
	return Iz;
}

long numStartZCurrentTriangles__, numEndZCurrentTriangles__;

void PerformCUDA_Invoke_Populate(
	cuSyst * pX_host, // populate in calling routine...
	long numVerts,
	f64 InnermostFrillCentroidRadius,
	f64 OutermostFrillCentroidRadius,
	long numStartZCurrentTriangles_,
	long numEndZCurrentTriangles_
)
{
	numStartZCurrentTriangles__ = numStartZCurrentTriangles_;
	numEndZCurrentTriangles__ = numEndZCurrentTriangles_;

	long iVertex;
	cuSyst * pX1, *pX2, *pX_half;

	printf("sizeof(CHAR4): %d \n"
		"sizeof(structural): %d \n"
		"sizeof(LONG3): %d \n"
		"sizeof(nn): %d \n",
		sizeof(CHAR4), sizeof(structural), sizeof(LONG3), sizeof(nn));

	if (cuSyst1.bInvoked == false) {

		Call(cudaMemGetInfo(&uFree, &uTotal), "cudaMemGetInfo (&uFree,&uTotal)");
		printf("Before Invokes: uFree %d uTotal %d\n", uFree, uTotal);

		cuSyst1.Invoke();
		cuSyst2.Invoke();
		cuSyst3.Invoke();

		Call(cudaMemGetInfo(&uFree, &uTotal), "cudaMemGetInfo (&uFree,&uTotal)");
		printf("After Invokes: uFree %d uTotal %d\n", uFree, uTotal);
	}

	// Populate video constant memory:
	// ________________________________

	f64_tens2 anticlock2;
	anticlock2.xx = cos(FULLANGLE);
	anticlock2.xy = -sin(FULLANGLE);
	anticlock2.yx = sin(FULLANGLE);
	anticlock2.yy = cos(FULLANGLE);
	Tensor2 * T2address;
	Call(cudaGetSymbolAddress((void **)(&T2address), Anticlockwise_d),
		"cudaGetSymbolAddress((void **)(&T2address),Anticlockwise)");
	Call(cudaMemcpy(T2address, &anticlock2, sizeof(f64_tens2), cudaMemcpyHostToDevice),
		"cudaMemcpy( T2address, &anticlock2, sizeof(f64_tens2),cudaMemcpyHostToDevice) U");
	// Note that objects appearing in constant memory must have empty constructor & destructor.

	f64_tens2 clock2;
	clock2.xx = cos(FULLANGLE);
	clock2.xy = sin(FULLANGLE);
	clock2.yx = -sin(FULLANGLE);
	clock2.yy = cos(FULLANGLE);
	Call(cudaGetSymbolAddress((void **)(&T2address), Clockwise_d),
		"cudaGetSymbolAddress((void **)(&T2address),Clockwise)");
	Call(cudaMemcpy(T2address, &clock2, sizeof(f64_tens2), cudaMemcpyHostToDevice),
		"cudaMemcpy( T2address, &clock2, sizeof(f64_tens2),cudaMemcpyHostToDevice) U");

	Set_f64_constant(kB, kB_);
	Set_f64_constant(c, c_);
	Set_f64_constant(q, q_);
	Set_f64_constant(m_e, m_e_);
	Set_f64_constant(m_ion, m_ion_);
	Set_f64_constant(m_i, m_ion_);
	Set_f64_constant(m_n, m_n_);
	Set_f64_constant(eoverm, eoverm_);
	Set_f64_constant(qoverM, qoverM_);
	Set_f64_constant(moverM, moverM_);
	Set_f64_constant(qovermc, eovermc_);
	Set_f64_constant(qoverMc, qoverMc_);
	Set_f64_constant(FOURPI_Q_OVER_C, FOUR_PI_Q_OVER_C_);
	Set_f64_constant(FOURPI_Q, FOUR_PI_Q_);
	Set_f64_constant(FOURPI_OVER_C, FOURPI_OVER_C_);
	f64 one_over_kB_ = 1.0 / kB_;
	f64 one_over_kB_cubed_ = 1.0 / (kB_*kB_*kB_);
	f64 kB_to_3halves_ = sqrt(kB_)*kB_;
	Set_f64_constant(one_over_kB, one_over_kB_);
	Set_f64_constant(one_over_kB_cubed, one_over_kB_cubed_);
	Set_f64_constant(kB_to_3halves, kB_to_3halves_);
	Set_f64_constant(NU_EI_FACTOR, NU_EI_FACTOR_);
	Set_f64_constant(nu_eiBarconst, nu_eiBarconst_);
	Set_f64_constant(Nu_ii_Factor, Nu_ii_Factor_);

	f64 M_i_over_in_ = m_i_ / (m_i_ + m_n_);
	f64 M_e_over_en_ = m_e_ / (m_e_ + m_n_);
	f64	M_n_over_ni_ = m_n_ / (m_i_ + m_n_);
	f64	M_n_over_ne_ = m_n_ / (m_e_ + m_n_);
	f64	M_en_ = m_e_ * m_n_ / ((m_e_ + m_n_)*(m_e_ + m_n_));
	f64	M_in_ = m_i_ * m_n_ / ((m_i_ + m_n_)*(m_i_ + m_n_));
	f64	M_ei_ = m_e_ * m_i_ / ((m_e_ + m_i_)*(m_e_ + m_i_));
	f64	m_en_ = m_e_ * m_n_ / (m_e_ + m_n_);
	f64	m_ei_ = m_e_ * m_i_ / (m_e_ + m_i_);
	Set_f64_constant(M_i_over_in, M_i_over_in_);
	Set_f64_constant(M_e_over_en, M_e_over_en_);// = m_e / (m_e + m_n);
	Set_f64_constant(M_n_over_ni, M_n_over_ni_);// = m_n / (m_i + m_n);
	Set_f64_constant(M_n_over_ne, M_n_over_ne_);// = m_n / (m_e + m_n);
	Set_f64_constant(M_en, M_en_);
	Set_f64_constant(M_in, M_in_);
	Set_f64_constant(M_ei, M_ei_);
	Set_f64_constant(m_en, m_en_);
	Set_f64_constant(m_ei, m_ei_);

	Set_f64_constant(over_m_e, over_m_e_);
	Set_f64_constant(over_m_i, over_m_i_);
	Set_f64_constant(over_m_n, over_m_n_);

	f64 over_sqrt_m_ion_ = 1.0 / sqrt(m_i_);
	f64 over_sqrt_m_e_ = 1.0 / sqrt(m_e_);
	f64 over_sqrt_m_neutral_ = 1.0 / sqrt(m_n_);
	Set_f64_constant(over_sqrt_m_ion, over_sqrt_m_ion_);
	Set_f64_constant(over_sqrt_m_e, over_sqrt_m_e_);
	Set_f64_constant(over_sqrt_m_neutral, over_sqrt_m_neutral_);

	Set_f64_constant(FRILL_CENTROID_OUTER_RADIUS_d, OutermostFrillCentroidRadius);
	Set_f64_constant(FRILL_CENTROID_INNER_RADIUS_d, InnermostFrillCentroidRadius);


	Call(cudaGetSymbolAddress((void **)(&f64address), m_e ), 
			"cudaGetSymbolAddress((void **)(&f64address), m_e )");
	Call(cudaMemcpy( f64address, &m_e_, sizeof(f64),cudaMemcpyHostToDevice),
			"cudaMemcpy( f64address, &m_e_, sizeof(f64),cudaMemcpyHostToDevice) src dest");
						
	f64 value = 1.25;
	f64 value2 = 1.5;

	Call(cudaMemcpyToSymbol(billericay, &value, sizeof(f64)), "bill the bat.");

	Call(cudaGetSymbolAddress((void **)(&f64address), billericay),"billericay1");

	Call(cudaMemcpy(f64address, &value, sizeof(f64), cudaMemcpyHostToDevice),"can we");
	
	Call(cudaMemcpy(&value2, f64address, sizeof(f64), cudaMemcpyDeviceToHost),"fdfdf");
	printf("value2 = %f\n",value2); // = 1.25

	// So this stuff DOES work
	// But debugger gives incorrect reading of everything as 0

	Call(cudaGetSymbolAddress((void **)(&f64address), m_e), "m_e");
	Call(cudaMemcpy(&value2, f64address, sizeof(f64), cudaMemcpyDeviceToHost), "fdfdf");
	printf("value2 = %1.8E\n", value2); // = m_e.
	// This was a total runaround.

	// four_pi_over_c_ReverseJz, EzStrength_d; // set at the time
	numReverseJzTriangles = numEndZCurrentTriangles_ - numStartZCurrentTriangles_;
	long * longaddress;
	Call(cudaGetSymbolAddress((void **)(&longaddress), numStartZCurrentTriangles),
		"cudaGetSymbolAddress((void **)(&longaddress), numStartZCurrentTriangles)");
	Call(cudaMemcpy(longaddress, &numStartZCurrentTriangles_, sizeof(long), cudaMemcpyHostToDevice),
		"cudaMemcpy(longaddress, &numStartZCurrentTriangles_, sizeof(long), cudaMemcpyHostToDevice)");
	Call(cudaGetSymbolAddress((void **)(&longaddress), numEndZCurrentTriangles),
		"cudaGetSymbolAddress((void **)(&longaddress), numEndZCurrentTriangles)");
	Call(cudaMemcpy(longaddress, &numEndZCurrentTriangles_, sizeof(long), cudaMemcpyHostToDevice),
		"cudaMemcpy(longaddress, &numEndZCurrentTriangles_, sizeof(long), cudaMemcpyHostToDevice)");
	// stored so we can check if it's a triangle that has reverse Jz
	Call(cudaGetSymbolAddress((void **)(&longaddress), NumInnerFrills_d),
		"cudaGetSymbolAddress((void **)(&longaddress), NumInnerFrills_d)");
	Call(cudaMemcpy(longaddress, &NumInnerFrills, sizeof(long), cudaMemcpyHostToDevice),
		"cudaMemcpy(longaddress, &NumInnerFrills, sizeof(long), cudaMemcpyHostToDevice)");
	Call(cudaGetSymbolAddress((void **)(&longaddress), FirstOuterFrill_d),
		"cudaGetSymbolAddress((void **)(&longaddress), FirstOuterFrill_d)");
	Call(cudaMemcpy(longaddress, &FirstOuterFrill, sizeof(long), cudaMemcpyHostToDevice),
		"cudaMemcpy(longaddress, &FirstOuterFrill, sizeof(long), cudaMemcpyHostToDevice)");

	Call(cudaMemcpyToSymbol(cross_T_vals_d, cross_T_vals, 10 * sizeof(f64)),
		"cudaMemcpyToSymbol(cross_T_vals_d,cross_T_vals, 10*sizeof(f64))");
	Call(cudaMemcpyToSymbol(cross_s_vals_viscosity_ni_d, cross_s_vals_viscosity_ni,
		10 * sizeof(f64)),
		"cudaMemcpyToSymbol(cross_s_vals_viscosity_ni_d,cross_s_vals_viscosity_ni, \
		10*sizeof(f64))");
	Call(cudaMemcpyToSymbol(cross_s_vals_viscosity_nn_d, cross_s_vals_viscosity_nn,
		10 * sizeof(f64)),
		"cudaMemcpyToSymbol(cross_s_vals_viscosity_nn_d,cross_s_vals_viscosity_nn, \
		10*sizeof(f64))");
	Call(cudaMemcpyToSymbol(cross_s_vals_MT_ni_d, cross_s_vals_momtrans_ni,
		10 * sizeof(f64)),
		"cudaMemcpyToSymbol(cross_s_vals_MT_ni_d,cross_s_vals_momtrans_ni, \
		10*sizeof(f64))");

	// 1. More cudaMallocs for d/dt arrays and main data:
	// and aggregation arrays...
	// ____________________________________________________

	CallMAC(cudaMalloc((void **)&p_nu_major, NUMVERTICES * sizeof(species3)));
	CallMAC(cudaMalloc((void **)&p_was_vertex_rotated, NUMVERTICES * sizeof(char)));
	CallMAC(cudaMalloc((void **)&p_triPBClistaffected, NUMVERTICES * sizeof(char)));
	CallMAC(cudaMalloc((void **)&p_T_upwind_minor, NMINOR * sizeof(T3)));
	
	CallMAC(cudaMalloc((void **)&p_v0, NMINOR * sizeof(v4)));
	CallMAC(cudaMalloc((void **)&p_vn0, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_sigma_Izz, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Iz0, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_OhmsCoeffs, NMINOR * sizeof(OhmsCoeffs)));

	CallMAC(cudaMalloc((void **)&p_sum_eps_deps_by_dbeta_heat, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_depsbydbeta_sq_heat, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_eps_eps_heat, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Jacobi_heat, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_epsilon_heat, NMINOR * sizeof(f64)));
	 
	CallMAC(cudaMalloc((void **)&p_sum_eps_deps_by_dbeta, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_depsbydbeta_sq, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_eps_eps, numTilesMinor * sizeof(f64)));

	 
	CallMAC(cudaMalloc((void **)&p_Jacobi_n, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Jacobi_i, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Jacobi_e, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_epsilon_n, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_epsilon_i, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_epsilon_e, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_coeffself_n, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_coeffself_i, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_coeffself_e, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_dbeta_n, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_dbeta_i, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_dbeta_e, NMINOR * sizeof(f64)));
	 

	CallMAC(cudaMalloc((void **)&p_MAR_neut, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_MAR_ion, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_MAR_elec, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_Az, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_AzNext, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_LapAz, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_LapAzNext, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_LapCoeffself, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_LapJacobi, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Jacobi_x, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_epsilon, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Azdot0, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_gamma, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Integrated_div_v_overall, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Div_v_neut, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Div_v, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Div_v_overall, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_ROCAzdotduetoAdvection, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_ROCAzduetoAdvection, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_GradAz, NMINOR * sizeof(f64_vec2)));
	CallMAC(cudaMalloc((void **)&p_GradTe, NMINOR * sizeof(f64_vec2)));

	CallMAC(cudaMalloc((void **)&p_one_over_n, NMINOR * sizeof(nvals)));
	CallMAC(cudaMalloc((void **)&p_one_over_n2, NMINOR * sizeof(nvals)));

	CallMAC(cudaMalloc((void **)&p_kappa_n, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_kappa_i, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_kappa_e, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_nu_i, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_nu_e, NMINOR * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&p_n_shards, NUMVERTICES * sizeof(ShardModel)));
	CallMAC(cudaMalloc((void **)&p_n_shards_n, NUMVERTICES * sizeof(ShardModel)));
	CallMAC(cudaMalloc((void **)&NT_addition_rates_d, NMINOR * sizeof(NTrates)));
	CallMAC(cudaMalloc((void **)&NT_addition_tri_d, NMINOR * sizeof(NTrates)));

	CallMAC(cudaMalloc((void **)&p_coeff_of_vez_upon_viz, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_beta_ie_z, NMINOR * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&p_longtemp, NMINOR * sizeof(long)));
	CallMAC(cudaMalloc((void **)&p_temp1, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_temp2, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_temp3, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_temp4, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_temp5, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_temp6, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_bool, NMINOR * sizeof(bool)));
	CallMAC(cudaMalloc((void **)&p_denom_i, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_denom_e, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_summands, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Iz0_summands, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Iz0_initial, numTilesMinor * sizeof(f64)));
	
	CallMAC(cudaMalloc((void **)&p_InvertedMatrix_i, NMINOR * sizeof(f64_tens3)));
	CallMAC(cudaMalloc((void **)&p_InvertedMatrix_e, NMINOR * sizeof(f64_tens3)));
	CallMAC(cudaMalloc((void **)&p_MAR_ion2, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_MAR_elec2, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&NT_addition_rates_d_temp, NMINOR * sizeof(NTrates)));
	CallMAC(cudaMalloc((void **)&p_epsilon_xy, NMINOR * sizeof(f64_vec2)));
	CallMAC(cudaMalloc((void **)&p_epsilon_iz, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_epsilon_ez, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_vJacobi_i, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_vJacobi_e, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_d_beta_i, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_d_beta_e, NMINOR * sizeof(f64_vec3)));

	CallMAC(cudaMalloc((void **)&p_sum_eps_deps_by_dbeta_i, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_eps_deps_by_dbeta_e, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_depsbydbeta_i_times_i, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_depsbydbeta_e_times_e, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_depsbydbeta_e_times_i, numTilesMinor * sizeof(f64)));

	p_GradTe_host = (f64_vec2 *)malloc(NMINOR * sizeof(f64_vec2));
	p_GradAz_host = (f64_vec2 *)malloc(NMINOR * sizeof(f64_vec2));
	p_B_host = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));
	p_MAR_ion_host = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));
	p_MAR_ion_compare = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));
	p_MAR_elec_host = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));
	p_MAR_elec_compare = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));
	p_MAR_neut_host = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));
	p_MAR_neut_compare = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));

	p_longtemphost = (long *)malloc(NMINOR * sizeof(long));
	p_temphost1 = (f64 *)malloc(NMINOR * sizeof(f64)); // changed for debugging
	p_temphost2 = (f64 *)malloc(NMINOR * sizeof(f64)); // changed for debugging
	p_temphost3 = (f64 *)malloc(NMINOR * sizeof(f64));
	p_temphost4 = (f64 *)malloc(NMINOR * sizeof(f64));
	p_temphost5 = (f64 *)malloc(NMINOR * sizeof(f64));
	p_temphost6 = (f64 *)malloc(NMINOR * sizeof(f64));
	p_boolhost = (bool *)malloc(NMINOR * sizeof(bool));

	if (p_temphost6 == 0) { printf("p6 == 0"); }
	else { printf("p6 != 0"); };
	temp_array_host = (f64 *)malloc(NMINOR * sizeof(f64));

	p_OhmsCoeffs_host = (OhmsCoeffs *)malloc(NMINOR * sizeof(OhmsCoeffs));

	p_summands_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_Iz0_summands_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_Iz0_initial_host = (f64 *)malloc(numTilesMinor * sizeof(f64));

	p_sum_eps_deps_by_dbeta_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_sum_depsbydbeta_sq_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_sum_eps_eps_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	// Cannot see that I have ever yet put in anywhere to free this memory.

	p_sum_eps_deps_by_dbeta_host_heat = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_sum_depsbydbeta_sq_host_heat = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_sum_eps_eps_host_heat = (f64 *)malloc(numTilesMinor * sizeof(f64));
	
	p_sum_eps_deps_by_dbeta_i_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_sum_eps_deps_by_dbeta_e_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_sum_depsbydbeta_i_times_i_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_sum_depsbydbeta_e_times_e_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_sum_depsbydbeta_e_times_i_host = (f64 *)malloc(numTilesMinor * sizeof(f64));


	// 2. cudaMemcpy system state from host: this happens always
	// __________________________________________________________

	// Note that we do always need an intermediate system on the host because
	// cudaMemcpy is our MO.
	pX_host->SendToDevice(cuSyst1);
	cuSyst2.CopyStructuralDetailsFrom(cuSyst1);
	cuSyst3.CopyStructuralDetailsFrom(cuSyst1);
	// Any logic to this?
	// Why not make a separate object containing the things that stay the same between typical runs?
	// ie, what is a neighbour of what.
	// info contains both pos and flag so that's not constant under advection; only neigh lists are.

	printf("Done main cudaMemcpy to video memory.\n");

	// Set up kernel L1/shared:
	 
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

	cudaFuncSetCacheConfig(kernelCreateShardModelOfDensities_And_SetMajorArea,
		cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(kernelCalculateNu_eHeartNu_iHeart_nu_nn_visc,
		cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(kernelAdvanceDensityAndTemperature,
		cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(kernelPopulateOhmsLaw,
		cudaFuncCachePreferL1); 
	cudaFuncSetCacheConfig(kernelCalculate_ita_visc, 
		cudaFuncCachePreferL1);

	pX1 = &cuSyst1;
	pX_half = &cuSyst2;
	pX2 = &cuSyst3;

	printf("during Invoke_Populate:\n");
	Call(cudaGetSymbolAddress((void **)(&f64address), m_i), "m_i");
	cudaMemcpy(&value, f64address, sizeof(f64), cudaMemcpyDeviceToHost);
	printf("m_i = %1.10E \n", value);
	Call(cudaGetSymbolAddress((void **)(&f64address), m_e), "m_e");
	cudaMemcpy(&value, f64address, sizeof(f64), cudaMemcpyDeviceToHost);
	printf("m_e = %1.10E \n", value);

	/*
	CallMAC(cudaMemset(p_summands, 0, sizeof(f64)*numTilesMinor));
	Kernel_GetZCurrent << <numTilesMinor, threadsPerTileMinor >> >(
	pX1->p_tri_perinfo,
	pX1->p_nT_ion_minor,
	pX1->p_nT_elec_minor,
	pX1->p_v_ion,
	pX1->p_v_elec, // Not clear if this should be nv or {n,v} yet - think.
	pX1->p_area_minor,
	p_summands
	);
	Call(cudaThreadSynchronize(), "cudaThreadSynchronize GetZCurrent 1.");

	CallMAC(cudaMemcpy(p_summands_host, p_summands, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost));
	Iz0 = 0.0;
	for (int ii = 0; ii < numTilesMinor; ii++)
	{
	Iz0 += p_summands_host[ii];
	};
	printf("Iz X1 before area calc %1.14E \n", Iz0); // == 0.0 since areas = 0
	*/

	//pX1->PerformCUDA_Advance(&pX2, &pX_half);

}

// Sequence:
// _________
// 
// Once: call Invoke routine
// .. and send host system to device

// ... do advance step x 10     [1 advection cycle = 1e-11?]
// ... send back and display & save (1e-11 ... 5s per ns)
// ... do advance step x 10     [1 advection cycle = 1e-11?]
// ... send back and display & save
// ... 
// ... send back and display & save;

// ...               Re-Delaunerize (1e-10)
// ...         send to device

//
// Once: revoke all

void PerformCUDA_RunStepsAndReturnSystem(cuSyst * pX_host)
{
	float elapsedTime;
	static cuSyst * pX1 = &cuSyst1;
	static cuSyst * pX2 = &cuSyst3;    // remember which way round - though with an even number of steps it's given we get back
	static cuSyst * pX_half = &cuSyst2; 
	cuSyst * pXtemp;
	// So let's be careful here
	
	cudaEvent_t start1, stop1;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, 0);
	cudaEventSynchronize(start1);

	long iSubstep;

	// Ultimately this 10 steps .. so 1e-11? .. can be 1 advective step.

	// B is set for pX_half and pX1. So take pX_half value and spit it to host.

	fp_dbg = fopen("dbg1.txt", "a");
	fp_trajectory = fopen("traj.txt", "a");
	for (iSubstep = 0; iSubstep < 10; iSubstep++)
	{
		printf("\nSTEP %d\n-------------\n", iSubstep);
		printf("evaltime = %1.10E \n\n", evaltime);
		pX1->PerformCUDA_Advance(pX2, pX_half);

		pXtemp = pX1;
		pX1 = pX2;
		pX2 = pXtemp;
		 
		// We need to smoosh the izTri etc data on to the pXhalf and new pX dest systems
		// as it doesn't update any other way and this data will be valid until renewed
		// in the dest system at the end of the step -- riiiiiiiight?
		pX_half->CopyStructuralDetailsFrom(*pX1);
		pX2->CopyStructuralDetailsFrom(*pX1);
/*
		cudaMemcpy(pX_half->p_izTri_vert, pX1->p_izTri_vert, 
			NUMVERTICES*MAXNEIGH_d * sizeof(long), cudaMemcpyDeviceToDevice);
		cudaMemcpy(pX2->p_izTri_vert, pX1->p_izTri_vert,
			NUMVERTICES*MAXNEIGH_d * sizeof(long), cudaMemcpyDeviceToDevice);

		cudaMemcpy(pX_half->p_izNeigh_vert, pX1->p_izNeigh_vert,
			NUMVERTICES*MAXNEIGH_d * sizeof(long), cudaMemcpyDeviceToDevice);
		cudaMemcpy(pX2->p_izNeigh_vert, pX1->p_izNeigh_vert,
			NUMVERTICES*MAXNEIGH_d * sizeof(long), cudaMemcpyDeviceToDevice);
		
		cudaMemcpy(pX_half->p_szPBCtri_vert, pX1->p_szPBCtri_vert,
			NUMVERTICES*MAXNEIGH_d * sizeof(char), cudaMemcpyDeviceToDevice);
		cudaMemcpy(pX2->p_szPBCtri_vert, pX1->p_szPBCtri_vert,
			NUMVERTICES*MAXNEIGH_d * sizeof(char), cudaMemcpyDeviceToDevice);
		
		cudaMemcpy(pX_half->p_szPBCneigh_vert, pX1->p_szPBCneigh_vert,
			NUMVERTICES*MAXNEIGH_d * sizeof(char), cudaMemcpyDeviceToDevice);
		cudaMemcpy(pX2->p_szPBCneigh_vert, pX1->p_szPBCneigh_vert,
			NUMVERTICES*MAXNEIGH_d * sizeof(char), cudaMemcpyDeviceToDevice);

		cudaMemcpy(pX_half->p_izNeigh_TriMinor, pX1->p_izNeigh_TriMinor,
			NUMTRIANGLES*6 * sizeof(long), cudaMemcpyDeviceToDevice);
		cudaMemcpy(pX2->p_izNeigh_TriMinor, pX1->p_izNeigh_TriMinor,
			NUMTRIANGLES * 6 * sizeof(long), cudaMemcpyDeviceToDevice);

		cudaMemcpy(pX_half->p_szPBC_triminor, pX1->p_szPBC_triminor,
			NUMTRIANGLES * 6 * sizeof(char), cudaMemcpyDeviceToDevice);
		cudaMemcpy(pX2->p_szPBC_triminor, pX1->p_szPBC_triminor,
			NUMTRIANGLES * 6 * sizeof(char), cudaMemcpyDeviceToDevice);
		
		cudaMemcpy(pX_half->p_szPBC_triminor, pX1->p_szPBC_triminor,
			NUMTRIANGLES * 6 * sizeof(char), cudaMemcpyDeviceToDevice);
		cudaMemcpy(pX_half->p_szPBC_triminor, pX1->p_szPBC_triminor,
			NUMTRIANGLES * 6 * sizeof(char), cudaMemcpyDeviceToDevice);*/
	}; 
	// After an even number of goes, pX1 = &cuSyst1 and this is where we ended up.
		
	fclose(fp_dbg);
	fclose(fp_trajectory);

	cudaEventRecord(stop1, 0);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&elapsedTime, start1, stop1);
	printf("Elapsed time for 10 steps : %f ms\n", elapsedTime);

	// update with the most recent B field since we did not update it properly after subcycle:
	cudaMemcpy(pX1->p_B, pX_half->p_B, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);

	pX1->SendToHost(*pX_host);
	
	// Now store for 1D graphs: temphost3 = n, temphost4 = vr, temphost5 = vez

	f64 Integral_Azdotdot = 0.0;
	f64 Integral_fabsAzdotdot = 0.0;
	f64 Integral_Azdot = 0.0;
	f64 Integral_fabsAzdot = 0.0;
	for (long i = 0; i < NMINOR; i++)
	{
		p_temphost3[i] = pX_host->p_AAdot[i].Azdot;
		p_temphost5[i] = temp_array_host[i]; 
		p_temphost6[i] = -FOURPIOVERC_*q_*pX_host->p_n_minor[i].n*
			(pX_host->p_vie[i].viz - pX_host->p_vie[i].vez);
		p_temphost4[i] = c_*c_*(temp_array_host[i] - p_temphost6[i]);

		Integral_Azdotdot += p_temphost4[i] * pX_host->p_AreaMinor[i];
		Integral_fabsAzdotdot += fabs(p_temphost4[i] * pX_host->p_AreaMinor[i]);
		Integral_Azdot += p_temphost3[i] * pX_host->p_AreaMinor[i];
		Integral_fabsAzdot += fabs(p_temphost3[i] * pX_host->p_AreaMinor[i]);
	}
	printf("Integral Azdotdot %1.10E fabs %1.10E \n", Integral_Azdotdot, Integral_fabsAzdotdot);
	printf("Integral Azdot %1.10E fabs %1.10E \n", Integral_Azdot, Integral_fabsAzdot);
	
	// Here we go .. Azdotdot doesn't say anything sensible because we are missing the +Jz contrib.
	// Azdot however doesn't show near 0 either.
	// That is bad.
	// Next: We should find that the actual change in Azdot sums to zero.
	// We should also therefore be finding that Azdot sums to zero.

	
}

f64_vec3 *vn_compare;
/*
void PerformCUDA_RunStepsAndReturnSystem_Debug(cuSyst * pcuSyst_host, cuSyst * p_cuSyst_compare, TriMesh * pTriMesh, TriMesh * pTriMeshhalf,
	TriMesh * pDestMesh)
{
	float elapsedTime;
	static cuSyst * pX1 = &cuSyst1;
	static cuSyst * pX2 = &cuSyst3;    // remember which way round - though with an even number of steps it's given we get back
	static cuSyst * pX_half = &cuSyst2;
	cuSyst * pXtemp;

	TriMesh * pTriMeshtemp;

	long iSubstep;
	 
	vn_compare = (f64_vec3 *)malloc(sizeof(f64_vec3)*NMINOR);

	for (iSubstep = 0; iSubstep < 10; iSubstep++) // 10 * 2e-12
	{
		pX1->PerformCUDA_Advance_Debug(pX2, pX_half, pcuSyst_host, p_cuSyst_compare, pTriMesh, pTriMeshhalf, pDestMesh);
		// keep sending TriMesh pX to p_cuSyst_compare
		 
		pXtemp = pX1;
		pX1 = pX2;
		pX2 = pXtemp;

		pTriMeshtemp = pTriMesh;
		pTriMesh = pDestMesh;
		pDestMesh = pTriMeshtemp;
	};
	// After an even number of goes, pX1 = &cuSyst1 and this is where we ended up.
	free(vn_compare);

	pX1->SendToHost(*pcuSyst_host);

	// It's then up to the caller to populate TriMesh from pX_host.
}*/
/*
void cuSyst::PerformCUDA_Advance_Debug(const cuSyst * pX_target, const cuSyst * pX_half,
	const cuSyst * p_cuSyst_host, cuSyst * p_cuSyst_compare, TriMesh * pTriMesh, TriMesh * pTriMeshhalf,
	TriMesh * pDestMesh)
{
	long iMinor, iSubstep;
	f64 Iz_prescribed, Iz_k;
	static long runs = 0;
	float elapsedTime;
	f64_vec2 temp_vec2;

	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	SetConsoleTextAttribute(hConsole, 12);
	printf("\n"
		".                   .\n"
		".                   .\n"
		".                   .\n"
		".                   .\n"
		".....................\n");
	SetConsoleTextAttribute(hConsole, 15);
	printf("....  STEP %d   ....\n", runs);
	SetConsoleTextAttribute(hConsole, 12);
	printf(".....................\n\n");
	SetConsoleTextAttribute(hConsole, 15);
	
	f64 value;
	Call(cudaGetSymbolAddress((void **)(&f64address), m_i), "m_i");
	cudaMemcpy(&value, f64address, sizeof(f64), cudaMemcpyDeviceToHost);
	printf("m_i = %1.12E \n", value);
	Call(cudaGetSymbolAddress((void **)(&f64address), m_e), "m_e");
	cudaMemcpy(&value, f64address, sizeof(f64), cudaMemcpyDeviceToHost);
	printf("m_e = %1.12E \n", value);

	p_cuSyst_compare->PopulateFromTriMesh(pTriMesh);
	cudaMemcpy(p_cuSyst_host->p_T_minor, this->p_T_minor, NMINOR * sizeof(T3), cudaMemcpyDeviceToHost);
	Compare_T3(p_cuSyst_host->p_T_minor, p_cuSyst_compare->p_T_minor, NMINOR);
	printf("compared T .\n");

	 



#define USE_N_MAJOR_FOR_VERTEX 
	// and simple average for tris; shards only for upwind density on tris.

	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> >(
		this->p_n_minor,
		this->p_n_major,
		this->p_T_minor,
		this->p_info,
		this->p_cc,
		this->p_tri_corner_index,
		this->p_tri_periodic_corner_flags
		); // call before CreateShardModel 
	Call(cudaThreadSynchronize(), "cudaTS average nTx");
	cudaMemcpy(this->p_n_minor + BEGINNING_OF_CENTRAL,
		this->p_n_major, sizeof(nvals)*NUMVERTICES, cudaMemcpyDeviceToDevice);

	// Will want a better approx here.
	// --------------------------------------------

	pTriMesh->EnsureAnticlockwiseTriangleCornerSequences_SetupTriMinorNeighboursLists();
	pTriMesh->Average_n_T_to_tris_and_calc_centroids_and_minorpos(); // call before CreateShardModel 
	p_cuSyst_compare->PopulateFromTriMesh(pTriMesh);

	cudaMemcpy(p_cuSyst_host->p_n_minor, this->p_n_minor, NMINOR * sizeof(nvals), cudaMemcpyDeviceToHost);
	Compare_nvals(p_cuSyst_host->p_n_minor, p_cuSyst_compare->p_n_minor, NMINOR);
	printf("compared nvals.\n");

	cudaMemcpy(p_cuSyst_host->p_T_minor, this->p_T_minor, NMINOR * sizeof(T3), cudaMemcpyDeviceToHost);
	Compare_T3(p_cuSyst_host->p_T_minor, p_cuSyst_compare->p_T_minor, NMINOR);
	printf("compared T 000.\n");

	printf("Compare v_n: \n");
	cudaMemcpy(p_cuSyst_host->p_v_n, this->p_v_n, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToHost);
	for (long i = 0; i < NMINOR; i++)
	{
		vn_compare[i] = pTriMesh->pData[i].v_n;
	}
	Compare_f64_vec3(p_cuSyst_host->p_v_n, vn_compare, NMINOR);
	printf("\n");


	cudaMemcpy(p_cuSyst_host->p_info, this->p_info, NMINOR * sizeof(structural), cudaMemcpyDeviceToHost);
	Compare_structural(p_cuSyst_host->p_info, p_cuSyst_compare->p_info, NMINOR);
	printf("compared structural info.\n");


	//=================================================================================================

	pTriMesh->CreateShardModelOfDensities_And_SetMajorArea();// sets n_shards_n, n_shards, Tri_n_n_lists, Tri_n_lists

	for (long i = 0; i < NUMVERTICES; i++)
	{
		p_temphost1[i] = n_shards[i].n_cent;
	}
	cudaMemcpy(p_temp1, p_temphost1, sizeof(f64)*NUMVERTICES, cudaMemcpyHostToDevice);

	printf("got to here.\n");

	kernelCreateShardModelOfDensities_And_SetMajorArea_Debug << <numTilesMajor, threadsPerTileMajor >> >(
		this->p_info,
		this->p_n_major,
		this->p_n_minor,
		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		p_n_shards,
		p_n_shards_n,
		//p_Tri_n_lists,
		//p_Tri_n_n_lists,
		this->p_AreaMajor,
		p_temp1 // compare to the CPU n_cent and if different do a bunch of intermediate output.
		);// sets n_shards_n, n_shards, Tri_n_n_lists, Tri_n_lists
	Call(cudaThreadSynchronize(), "cudaTS CreateShardModels");

	ShardModel * shardtemp = (ShardModel *)malloc(sizeof(ShardModel)*NUMVERTICES);
	//cudaMemcpy(shardtemp, p_n_shards_n, sizeof(ShardModel)*NUMVERTICES, cudaMemcpyDeviceToHost);
	// Compare shard models:
	//printf("Compare n_shards_n:\n");
	//Compare_n_shards(shardtemp, n_shards_n, p_cuSyst_host); // ShardModel*NUMVERTICES

	cudaMemcpy(shardtemp, p_n_shards, sizeof(ShardModel)*NUMVERTICES, cudaMemcpyDeviceToHost);
	printf("Compare n_shards:\n");
	Compare_n_shards(shardtemp, n_shards, p_cuSyst_host); // ShardModel*NUMVERTICES

	// inauspicious start: overall v has to be split into 2 routines
	kernelCalculateOverallVelocitiesVertices << <numTilesMajor, threadsPerTileMajor >> > (
		this->p_info,
		this->p_vie + BEGINNING_OF_CENTRAL,
		this->p_v_n + BEGINNING_OF_CENTRAL,
		this->p_n_major,
		this->p_v_overall_minor + BEGINNING_OF_CENTRAL,
		p_n_shards,
		p_n_shards_n,
		this->p_izTri_vert,
		this->p_szPBCtri_vert
		);
	Call(cudaThreadSynchronize(), "cudaTS calculate overall velocities");
	
	kernelAverageOverallVelocitiesTriangles << <numTriTiles, threadsPerTileMinor >> > (
		this->p_v_overall_minor + BEGINNING_OF_CENTRAL,
		this->p_v_overall_minor,
		this->p_info,
		this->p_tri_corner_index,
		this->p_tri_periodic_corner_flags
		);
	Call(cudaThreadSynchronize(), "cudaTS avg overall v to triangles");
	
	// simulation.cu version:
	pTriMesh->CalculateOverallVelocities(p_v); // vertices first, then average to tris
	
	// --------------------------------------------
	cudaMemcpy(p_cuSyst_host->p_v_overall_minor, this->p_v_overall_minor, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToHost);
	Compare_f64_vec2(p_cuSyst_host->p_v_overall_minor, p_v, NMINOR);
	// Var 1 comes out as NaN at 23018.????
	// Now we think it's below????

	SetConsoleTextAttribute(hConsole, 10);
	kernelAdvectPositions << <numTilesMinor, threadsPerTileMinor >> >(
		0.5*Timestep,
		this->p_info,
		pX_half->p_info,
		this->p_v_overall_minor
		);
	Call(cudaThreadSynchronize(), "cudaTS AdvectPositions_CopyTris");

	SetConsoleTextAttribute(hConsole, 15);

	// ----------------------------------------------------------------
	memset(pTriMeshhalf->pData, 0, sizeof(plasma_data)*NMINOR);
	pTriMesh->AdvectPositions_CopyTris(0.5*Timestep, pTriMeshhalf, p_v);
	
	pTriMeshhalf->EnsureAnticlockwiseTriangleCornerSequences_SetupTriMinorNeighboursLists();
	pTriMeshhalf->Average_n_T_to_tris_and_calc_centroids_and_minorpos();
	p_cuSyst_compare->PopulateFromTriMesh(pTriMeshhalf); // Calls Average_n_T_to tris on pTriMeshhalf; needs vertex pos in pData set up before it tries determining n in CROSSING_INS

	cudaMemcpy(p_cuSyst_host->p_info, pX_half->p_info, NMINOR * sizeof(structural), cudaMemcpyDeviceToHost);
	Compare_structural(p_cuSyst_host->p_info, p_cuSyst_compare->p_info, NMINOR);

	printf("compared structural info.\n");

	//====================================================
	
#ifndef USE_N_MAJOR_FOR_VERTEX

	kernelInferMinorDensitiesFromShardModel << <numTilesMinor, threadsPerTileMinor >> >(
		this->p_info,
		this->p_n_minor,
		p_n_shards,
		p_n_shards_n,
		this->p_tri_corner_index,
		this->p_who_am_I_to_corner);
	Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities");

	pTriMesh->InferMinorDensitiesFromShardModel();
	
	// Something in the following has corrupted the n data, which is correct going into it.
	// pX->Average_n_T_to_tris_and_calc_centroids_and_minorpos();

	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
	// It is certainly curious that the avg is the same as the shard reading for 2 of the corners.
	// 29730
	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

	p_cuSyst_compare->PopulateFromTriMesh(pTriMesh);
	cudaMemcpy(p_cuSyst_host->p_n_minor, this->p_n_minor, sizeof(nvals)*NMINOR, cudaMemcpyDeviceToHost);
	Compare_nvals(p_cuSyst_host->p_n_minor, p_cuSyst_compare->p_n_minor, NUMVERTICES*2);
	
	// ########################################################################################
	// Now remember that we used n_shard.n_cent as n_minor on GPU.
	// Important to keep track of this distinction and the logic with n_major.
	// ########################################################################################
#endif

	kernelCalculateUpwindDensity_tris << <numTriTiles, threadsPerTileMinor >> >(
		this->p_info,
		p_n_shards_n,
		p_n_shards,
		this->p_vie,
		this->p_v_n,
		this->p_v_overall_minor,
		this->p_tri_corner_index,
		this->p_tri_neigh_index,
		this->p_who_am_I_to_corner,
		this->p_tri_periodic_neigh_flags,
		this->p_n_upwind_minor,
		this->p_T_minor,
		p_T_upwind_minor
		);
	Call(cudaThreadSynchronize(), "cudaTS CalculateUpwindDensity_tris");

	// Will need to do likewise on CPU - hope it still works.
	printf("Compare upwind density:\n");
	pTriMesh->CalcUpwindDensity_on_tris(p_temphost1, p_temphost2, p_v);
	for (long i = 0; i < NUMTRIANGLES; i++)
	{
		p_cuSyst_compare->p_n_upwind_minor[i].n = p_temphost1[i];
		p_cuSyst_compare->p_n_upwind_minor[i].n_n = p_temphost2[i];
	};
	cudaMemcpy(p_cuSyst_host->p_n_upwind_minor, this->p_n_upwind_minor, sizeof(nvals)*NMINOR, cudaMemcpyDeviceToHost);
	Compare_nvals(p_cuSyst_host->p_n_upwind_minor, p_cuSyst_compare->p_n_upwind_minor, NUMVERTICES * 2);

	// Note that upwind gives a low density the first step in CROSSING_INS. Probably doesn't matter tho since rel v = 0.

	cudaMemset(p_Integrated_div_v_overall, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Div_v_neut, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Div_v, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(NT_addition_rates_d, 0, sizeof(NTrates)*NUMVERTICES);

	printf("Compare v_n: \n");
	cudaMemcpy(p_cuSyst_host->p_v_n, this->p_v_n, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToHost);
	for (long i = 0; i < NMINOR; i++)
	{
		vn_compare[i] = pTriMesh->pData[i].v_n;
	}
	Compare_f64_vec3(p_cuSyst_host->p_v_n, vn_compare, NMINOR);
	printf("\n 8888888888888888888888888888888888 \n\n");

	kernelAccumulateAdvectiveMassHeatRate << <numTilesMajor, threadsPerTileMajor >> >(
		0.5*Timestep, // why it appears here?
		this->p_info,
		this->p_izTri_vert,
		this->p_szPBCtri_vert,

		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL,

		this->p_n_upwind_minor,
		this->p_vie,
		this->p_v_n,
		this->p_v_overall_minor,
		p_T_upwind_minor, // may or may not overlap source: don't we only use from tris? so not overlap

		NT_addition_rates_d,
		p_Div_v,
		p_Div_v_neut,
		p_Integrated_div_v_overall);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateAdvectiveMassHeatRate");

	NTrates * p_NT_addition_rates = (NTrates *)malloc(NUMVERTICES * sizeof(NTrates));
	cudaMemcpy(p_NT_addition_rates, NT_addition_rates_d, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToHost);

	memset(p_div_v_neut, 0, sizeof(f64)*NUMVERTICES);
	memset(p_div_v, 0, sizeof(f64)*NUMVERTICES);
	memset(NTadditionrates, 0, sizeof(NTrates)*NUMVERTICES);
	
	pTriMesh->AccumulateAdvectiveMassHeatRate(p_v, NTadditionrates, p_temphost1, p_temphost2);

	// This is where on 2nd step we are loading in totally different v_n, let's examine how /why that is
	// tri 48129 48128 48127 48471 48473 48472

	printf("done advective 1\n\n\n\n\n");

	// For now went for "advective change in v" where we divide d/dt Nv by N_derivsyst
	cudaMemset(p_MAR_neut, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_ion, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_elec, 0, sizeof(f64_vec3)*NMINOR);

	kernelCreate_pressure_gradT_and_gradA_LapA_CurlA_minor << <numTriTiles, threadsPerTileMinor >> >(
		this->p_info,
		this->p_T_minor,
		this->p_AAdot,

		this->p_izTri_vert,
		this->p_szPBCtri_vert, // ERROR: For such as 73841 we get a different char sequence here than on CPU.
		this->p_izNeigh_TriMinor,
		this->p_szPBC_triminor,
		this->p_who_am_I_to_corner,
		this->p_tri_corner_index,

		p_MAR_neut, p_MAR_ion, p_MAR_elec,

		p_n_shards,				// this kernel is for i+e only
		this->p_n_minor,
		p_GradTe,
		p_GradAz,
		p_LapAz,
		// Unused by anything else:
		p_ROCAzduetoAdvection, // Would probs be better to split out Az calc, remember
		p_ROCAzdotduetoAdvection, // Would probs be better to split out Az calc, remember
		this->p_v_overall_minor, // it's only this time that we need to collect it ofc.

								 // ######################################
								 // should put in a switch to not collect. But DO ZERO ROCAzdotduetoAdvection in that case.

		this->p_B,
		this->p_AreaMinor
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelCreate_pressure_gradT_and_gradA_LapA_CurlA_minor");

	kernelCreate_momflux_minor << <numTriTiles, threadsPerTileMinor >> >(
		this->p_info,
		this->p_vie,
		this->p_v_overall_minor,

		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_izNeigh_TriMinor,
		this->p_szPBC_triminor,

		this->p_who_am_I_to_corner,
		this->p_tri_corner_index,

		p_MAR_neut, p_MAR_ion, p_MAR_elec,

		p_n_shards   // this is so that we can do momentum advection?
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelCreate_momflux_minor");

	kernelNeutral_pressure_and_momflux << <numTriTiles, threadsPerTileMinor >> >(
		this->p_info,

		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_izNeigh_TriMinor,
		this->p_szPBC_triminor,
		this->p_who_am_I_to_corner,
		this->p_tri_corner_index,

		this->p_T_minor,
		this->p_v_n,
		p_n_shards_n,   // Now hang on, why did this appear? momentum advection?
		this->p_n_minor,
		this->p_v_overall_minor,
		p_MAR_neut
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelNeutral_pressure_and_momflux");

	printf("\nResults of AccumulateAdvectiveMassHeatRate & etc:\n");
	Compare_NTrates(p_NT_addition_rates, NTadditionrates);
	// Compare_f64(p_Div_v, p_div_v, NUMVERTICES);
	// p_Div_v is on the device so that doesn't work. But I think we know it's zero.

	// Suspect on 0th step this all comes out 0, there is no velocity here yet.
	  
	kernelCalculateNu_eHeartNu_iHeart_nu_nn_visc << <numTilesMajorClever, threadsPerTileMajorClever >> >(
		this->p_info + BEGINNING_OF_CENTRAL,
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL,
		p_nu_major);
	Call(cudaThreadSynchronize(), "cudaTS CalculateNu");
	  
	SetConsoleTextAttribute(hConsole, 14);
	kernelAccumulateDiffusiveHeatRateAndCalcIonisation << <numTilesMajorClever, threadsPerTileMajorClever >> >(
		0.5*Timestep,
		this->p_info,
		this->p_cc,
		this->p_izNeigh_vert,
		this->p_szPBCneigh_vert,
		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices
		this->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
		p_nu_major,
		NT_addition_rates_d,
		this->p_AreaMajor);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate");
	// To increase the efficiency we want to make a clever 2nd set of major tiles of size 192. Also try 256, 384.
	SetConsoleTextAttribute(hConsole, 15);
	cudaMemcpy(p_NT_addition_rates, NT_addition_rates_d, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToHost);

	pTriMesh->AccumulateDiffusiveHeatRateAndCalcIonisation(0.5*Timestep, NTadditionrates); // Wants minor n,T and B

	printf("\nResults of AccumulateDiffusiveHeatRateAndCalcIonisation:\n");
	Compare_NTrates(p_NT_addition_rates, NTadditionrates);
	

	// See difference at 9th place for NiTi but it's small figures compared with NeTe which is at 11th place.
	// =============================================================================
	// HMMM

	SetConsoleTextAttribute(hConsole, 14);
	kernelAdvanceDensityAndTemperature << <numTilesMajor, threadsPerTileMajor >> >(
		0.5*Timestep,
		this->p_info + BEGINNING_OF_CENTRAL,
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL,
		NT_addition_rates_d,
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL,
		this->p_vie + BEGINNING_OF_CENTRAL,
		this->p_v_n + BEGINNING_OF_CENTRAL,
		 
		p_Div_v_neut, p_Div_v,
		p_Integrated_div_v_overall,
		this->p_AreaMajor,
		pX_half->p_n_major,
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL
		);
	Call(cudaThreadSynchronize(), "cudaTS Advance_n_and_T [this to half]");
	
	SetConsoleTextAttribute(hConsole, 15);
	pTriMesh->AdvanceDensityAndTemperature(0.5*Timestep, pTriMesh, pTriMeshhalf, NTadditionrates);
	 
	// compare n,T:
//	p_cuSyst_compare->PopulateFromTriMesh(pTriMeshhalf);
//	cudaMemcpy(p_cuSyst_host->p_n_major, pX_half->p_n_major, sizeof(nvals)*NUMVERTICES, cudaMemcpyDeviceToHost);
//	Compare_nvals(p_cuSyst_host->p_n_major, p_cuSyst_compare->p_n_major, NUMVERTICES);
//	cudaMemcpy(p_cuSyst_host->p_T_minor, pX_half->p_T_minor, sizeof(T3)*NMINOR, cudaMemcpyDeviceToHost);
//	Compare_T3(p_cuSyst_host->p_T_minor + BEGINNING_OF_CENTRAL, p_cuSyst_compare->p_T_minor + BEGINNING_OF_CENTRAL, NUMVERTICES);
	 
	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_n_minor,
		pX_half->p_n_major,
		pX_half->p_T_minor,
		pX_half->p_info,
		pX_half->p_cc,
		pX_half->p_tri_corner_index,
		pX_half->p_tri_periodic_corner_flags
		); // call before CreateShardModel 
	Call(cudaThreadSynchronize(), "cudaTS average nTx 2");

	// Fill in n_minor = n_major as we are not using shards to give minor densities this time. :
	cudaMemcpy(pX_half->p_n_minor + BEGINNING_OF_CENTRAL,
		pX_half->p_n_major, sizeof(nvals)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	
	pTriMeshhalf->Average_n_T_to_tris_and_calc_centroids_and_minorpos();
	p_cuSyst_compare->PopulateFromTriMesh(pTriMeshhalf);
	cudaMemcpy(p_cuSyst_host->p_n_minor, pX_half->p_n_minor, sizeof(nvals)*NMINOR, cudaMemcpyDeviceToHost);
	Compare_nvals(p_cuSyst_host->p_n_minor, p_cuSyst_compare->p_n_minor, NMINOR);
	cudaMemcpy(p_cuSyst_host->p_T_minor, pX_half->p_T_minor, sizeof(T3)*NMINOR, cudaMemcpyDeviceToHost);
	printf("pTriMeshhalf comparison:\n");
	Compare_T3(p_cuSyst_host->p_T_minor, p_cuSyst_compare->p_T_minor, NMINOR);

	// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	// We are going to want to introduce 2nd degree approx to get n,T desired on tris.
	// Now let's set up the accel move to half-time which will provide us input of v to the full n,T move.
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	printf("this, pTriMesh comparison:\n");
	p_cuSyst_compare->PopulateFromTriMesh(pTriMesh);
	cudaMemcpy(p_cuSyst_host->p_T_minor, this->p_T_minor, NMINOR * sizeof(T3), cudaMemcpyDeviceToHost);
	Compare_T3(p_cuSyst_host->p_T_minor, p_cuSyst_compare->p_T_minor, NMINOR);
	printf("compared T (this, pTriMesh).\n");
	
	memset(AdditionalMomRates, 0, sizeof(three_vec3)*NMINOR); // what a mess!
	pTriMesh->Create_momflux_integral_grad_nT_and_gradA_LapA_CurlA_on_minors(p_v, AdditionalMomRates);

	p_cuSyst_compare->PopulateFromTriMesh(pTriMesh);
	cudaMemcpy(p_cuSyst_host->p_T_minor, this->p_T_minor, NMINOR * sizeof(T3), cudaMemcpyDeviceToHost);
	Compare_T3(p_cuSyst_host->p_T_minor, p_cuSyst_compare->p_T_minor, NMINOR);
	printf("compared T (this, pTriMesh).\n");

	// Now we are looking to compare p_MAR_ion and p_MAR_elec, gradAz, gradTe, curlAz
	printf("Area Minor:\n");
	cudaMemcpy(p_temphost2, this->p_AreaMinor, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
	Compare_f64(p_temphost2, pTriMesh->AreaMinorArray, NMINOR);

	printf("Grad Te:\n");
	cudaMemcpy(p_GradTe_host, p_GradTe, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToHost);
	Compare_f64_vec2(p_GradTe_host, GradTeArray, NMINOR);

	cudaMemcpy(p_GradAz_host, p_GradAz, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToHost);
	Compare_f64_vec2(p_GradAz_host, GradAz, NMINOR);

	printf("B:\n");
	cudaMemcpy(p_B_host, this->p_B, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToHost);
	Compare_f64_vec3(p_B_host, p_cuSyst_compare->p_B, NMINOR);

	// unparcel AdditionalMomRates:
	for (int qqq = 0; qqq < NMINOR; qqq++)
	{
		p_MAR_ion_compare[qqq] = AdditionalMomRates[qqq].ion;
		p_MAR_elec_compare[qqq] = AdditionalMomRates[qqq].elec;
		p_MAR_neut_compare[qqq] = AdditionalMomRates[qqq].neut;
	};

	cudaMemcpy(p_MAR_ion_host, p_MAR_ion, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_MAR_elec_host, p_MAR_elec, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_MAR_neut_host, p_MAR_neut, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToHost);

	printf("MAR_ion:\n");
	Compare_f64_vec3(p_MAR_ion_host, p_MAR_ion_compare, NMINOR);
	printf("MAR_elec:\n");
	Compare_f64_vec3(p_MAR_elec_host, p_MAR_elec_compare, NMINOR);
	printf("MAR_neut:\n");
	Compare_f64_vec3(p_MAR_neut_host, p_MAR_neut_compare, NMINOR);
	printf("MAR_elec[88000].x %1.12E %1.12E \n", p_MAR_elec_host[88000].x, p_MAR_elec_compare[88000].x);

	// Error at 10th place or so
	 
	//////////////////////////////////////////////////////////////////////////
	// Even more shards!: we will be getting advection rate for main step
	
	pTriMeshhalf->CreateShardModelOfDensities_And_SetMajorArea();// sets n_shards_n, n_shards, Tri_n_n_lists, Tri_n_lists

	for (long i = 0; i < NUMVERTICES; i++)
	{
		p_temphost1[i] = n_shards[i].n_cent;
	}
	cudaMemcpy(p_temp1, p_temphost1, sizeof(f64)*NUMVERTICES, cudaMemcpyHostToDevice);

	kernelCreateShardModelOfDensities_And_SetMajorArea_Debug << <numTilesMajor, threadsPerTileMajor >> > (
		pX_half->p_info,
		pX_half->p_n_major, // So, we sent the vertex portion of this but I think it comes through as 0.
		pX_half->p_n_minor,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert, // could be more intelligent way than storing and copying 3x
		p_n_shards,
		p_n_shards_n,
		//		p_Tri_n_lists,
		//	p_Tri_n_n_lists,
		pX_half->p_AreaMajor,
		p_temp1);
	Call(cudaThreadSynchronize(), "cudaTS CreateShardModel pX_half");
	
	//
	cudaMemcpy(shardtemp, p_n_shards, sizeof(ShardModel)*NUMVERTICES, cudaMemcpyDeviceToHost);
	printf("Compare n_shards:\n");
	Compare_n_shards(shardtemp, n_shards, p_cuSyst_host); // ShardModel*NUMVERTICES

#ifndef USE_N_MAJOR_FOR_VERTEX

	kernelInferMinorDensitiesFromShardModel << <numTilesMinor, threadsPerTileMinor >> >(
		pX_half->p_info,
		pX_half->p_n_minor,
		p_n_shards,
		p_n_shards_n,
		pX_half->p_tri_corner_index,
		pX_half->p_who_am_I_to_corner);// (At the moment just repopulating tri minor n.)
	Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities pX_half");
	
	pTriMeshhalf->InferMinorDensitiesFromShardModel();	
#endif

	p_cuSyst_compare->PopulateFromTriMesh(pTriMeshhalf);
	cudaMemcpy(p_cuSyst_host->p_n_minor, pX_half->p_n_minor, sizeof(nvals)*NMINOR, cudaMemcpyDeviceToHost);
	Compare_nvals(p_cuSyst_host->p_n_minor, p_cuSyst_compare->p_n_minor, NUMVERTICES * 2);

	printf("\n\n\n");

//	f64 Iz_prescribed_starttime = GetIzPrescribed(evaltime);
	f64 Iz_prescribed_endtime = GetIzPrescribed(evaltime + 0.5*Timestep); 
	// because we are setting pX_half->v 

	kernelEstimateCurrent << <numTilesMinor, threadsPerTileMinor >> >(
		this->p_info,
		this->p_n_minor,
		this->p_vie,
		this->p_AreaMinor,
		p_temp1); // Iz_k
	Call(cudaThreadSynchronize(), "cudaTS Estimate Iz_k");

	// Get suitable v to use for resistive heating on main step:
	kernelPopulateOhmsLaw << <numTilesMinor, threadsPerTileMinor >> >(
		0.5*Timestep,

		this->p_info,
		p_MAR_neut, p_MAR_ion, p_MAR_elec,
		this->p_B,
		p_LapAz,
		p_GradAz,
		p_GradTe,

		this->p_n_minor,

		this->p_T_minor, // minor : is it populated?
		this->p_vie,
		this->p_v_n,
		this->p_AAdot,
		this->p_AreaMinor, // popd?
		p_ROCAzdotduetoAdvection,

		p_vn0,
		p_v0,
		p_OhmsCoeffs,
		pX_target->p_AAdot, // intermediate value

		p_Iz0_summands,
		p_sigma_Izz,
		p_denom_i,
		p_denom_e,
		p_coeff_of_vez_upon_viz, 
		p_beta_ie_z,
		false,
		true,
		pX_half->p_n_minor
		);

	Call(cudaThreadSynchronize(), "cudaTS AccelerateOhms 1");

	cudaMemcpy(p_Iz0_summands_host, p_Iz0_summands, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_summands_host, p_sigma_Izz, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	long iTile;

	memset(sigma_tiles, 0, sizeof(f64)*numTilesMinor);
	// Get suitable v to use for resistive heating:
	pTriMesh->Accelerate2018(0.5*Timestep, pTriMesh, pTriMeshhalf,
		evaltime, false, true); // current is attained at start of step [ ie Iz == 0 on 1st step ]
	
	f64 Iz0 = 0.0;
	f64 sigma_Izz = 0.0;
	Iz_k = 0.0;
	for (iTile = 0; iTile < numTilesMinor; iTile++)
	{
		Iz0 += p_Iz0_summands_host[iTile];
		sigma_Izz += p_summands_host[iTile];
		Iz_k += p_temphost1[iTile];
	//	printf("%d: %1.12E %1.12E | %1.12E  diff %1.4E\n ", iTile, p_Iz0_summands_host[iTile],
	//		p_summands_host[iTile], sigma_tiles[iTile], p_summands_host[iTile] - sigma_tiles[iTile]);
	};

	f64 Ez_strength_ = (Iz_prescribed_endtime - Iz0) / sigma_Izz;
	Set_f64_constant(Ez_strength, Ez_strength_);

	f64 neg_Iz_per_triangle = -0.5*(Iz_prescribed_endtime + Iz_k) / (f64)numReverseJzTriangles;
	Set_f64_constant(negative_Iz_per_triangle, neg_Iz_per_triangle);

	printf("GPU: Iz_prescribed %1.14E Iz0 %1.14E sigma_Izz %1.14E \n",
		Iz_prescribed_endtime, Iz0, sigma_Izz);
	printf("Ez_strength (GPU) %1.14E \n", Ez_strength_);
	
	// Update velocities and Azdot:
	kernelCalculateVelocityAndAzdot << <numTilesMinor, threadsPerTileMinor >> >(
		0.5*Timestep,
		pX_half->p_info,
		p_vn0,
		p_v0,
		p_OhmsCoeffs,
		pX_target->p_AAdot,
		pX_half->p_n_minor,
		pX_half->p_AreaMinor,

		pX_half->p_AAdot,
		pX_half->p_vie,
		pX_half->p_v_n
		);
	Call(cudaThreadSynchronize(), "cudaTS AccelerateUpdate 1");

	// Compare v after acceleration
	printf("Compare v_e: \n");
	cudaMemcpy(p_cuSyst_host->p_vie, pX_half->p_vie, sizeof(v4)*NMINOR, cudaMemcpyDeviceToHost);
	f64_vec3 *ve_host = (f64_vec3 *)malloc(sizeof(f64_vec3)*NMINOR);
	f64_vec3 *ve_compare = (f64_vec3 *)malloc(sizeof(f64_vec3)*NMINOR);
	for (long i = 0; i < NMINOR; i++)
	{
		ve_compare[i].x = pTriMeshhalf->pData[i].vxy.x;
		ve_compare[i].y = pTriMeshhalf->pData[i].vxy.y;
		ve_compare[i].z = pTriMeshhalf->pData[i].vez;
		ve_host[i].x = p_cuSyst_host->p_vie[i].vxy.x;
		ve_host[i].y = p_cuSyst_host->p_vie[i].vxy.y;
		ve_host[i].z = p_cuSyst_host->p_vie[i].vez;
	}
	Compare_f64_vec3(ve_host, ve_compare, NMINOR);
	printf("\n");
	
	printf("Compare v_n (pX_half, pTriMeshhalf): \n");
	cudaMemcpy(p_cuSyst_host->p_v_n, pX_half->p_v_n, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToHost);
	for (long i = 0; i < NMINOR; i++)
	{
		vn_compare[i] = pTriMeshhalf->pData[i].v_n;
	}
	Compare_f64_vec3(p_cuSyst_host->p_v_n, vn_compare, NMINOR);

	// =============================================================================

	kernelAdvanceAzEuler << <numTilesMinor, threadsPerTileMinor >> >
		(0.5*h, this->p_AAdot, pX_half->p_AAdot, p_ROCAzduetoAdvection);
	Call(cudaThreadSynchronize(), "cudaTS AdvanceAzEuler");

	kernelResetFrillsAz_II << < numTriTiles, threadsPerTileMinor >> > (
		this->p_info, this->p_tri_neigh_index, pX_half->p_AAdot);
	Call(cudaThreadSynchronize(), "cudaTS ResetFrills I");


	// ))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
	// 
	// Now do the n,T,x advance to pDestmesh:
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	kernelCalculateOverallVelocitiesVertices << <numTilesMajor, threadsPerTileMajor >> >(
		pX_half->p_info,
		pX_half->p_vie + BEGINNING_OF_CENTRAL,
		pX_half->p_v_n + BEGINNING_OF_CENTRAL,
		pX_half->p_n_major,
		pX_half->p_v_overall_minor + BEGINNING_OF_CENTRAL,
		p_n_shards,
		p_n_shards_n,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert
		);
	Call(cudaThreadSynchronize(), "cudaTS calculate overall velocities 22");
	kernelAverageOverallVelocitiesTriangles << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_v_overall_minor + BEGINNING_OF_CENTRAL,
		pX_half->p_v_overall_minor,
		pX_half->p_info,
		pX_half->p_tri_corner_index,
		pX_half->p_tri_periodic_corner_flags // questionable wisdom of repeating info in 3 systems
		);
	Call(cudaThreadSynchronize(), "cudaTS avg overall v to triangles 22");

	cudaMemcpy(&temp_vec2, &(this->p_info[21554 + BEGINNING_OF_CENTRAL].pos), sizeof(f64_vec2), cudaMemcpyDeviceToHost);
	printf("\nPOSITION 21554 %1.12E %1.12E\n\n", temp_vec2.x, temp_vec2.y);

	kernelAdvectPositions << <numTilesMinor, threadsPerTileMinor >> >(
		Timestep,
		this->p_info,
		pX_target->p_info,
		pX_half->p_v_overall_minor // WHY WE WERE NOT USING FROM pX_half?
		);
	Call(cudaThreadSynchronize(), "cudaTS AdvectPositions_CopyTris");
	
	cudaMemcpy(&temp_vec2, &(pX_target->p_info[21554 + BEGINNING_OF_CENTRAL].pos), sizeof(f64_vec2), cudaMemcpyDeviceToHost);
	printf("\nPOSITION 21554 %1.12E %1.12E\n\n", temp_vec2.x, temp_vec2.y);


	pTriMeshhalf->CalculateOverallVelocities(p_v); // vertices first, then average to tris
	memset(pDestMesh->pData, 0, sizeof(plasma_data)*NMINOR);
	pTriMesh->AdvectPositions_CopyTris(Timestep, pDestMesh, p_v);

	printf("\nCompare v overall:\n");
	cudaMemcpy(p_cuSyst_compare->p_v_overall_minor, pX_half->p_v_overall_minor, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToHost);
	Compare_f64_vec2(p_cuSyst_compare->p_v_overall_minor, p_v, NMINOR);

	free(ve_host);
	free(ve_compare);

// ==========
	printf("Upwind density calc: \n");

	kernelCalculateUpwindDensity_tris << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_info,
		p_n_shards_n,
		p_n_shards,
		pX_half->p_vie,
		pX_half->p_v_n,
		pX_half->p_v_overall_minor,
		pX_half->p_tri_corner_index,
		pX_half->p_tri_neigh_index,
		pX_half->p_who_am_I_to_corner,
		pX_half->p_tri_periodic_neigh_flags,
		pX_half->p_n_upwind_minor,
		pX_half->p_T_minor,
		p_T_upwind_minor
		);
	Call(cudaThreadSynchronize(), "cudaTS CalculateUpwindDensity_tris pX_half");

	pTriMeshhalf->CalcUpwindDensity_on_tris(p_temphost1, p_temphost2, p_v);
		//f64 * p_n_upwind, f64 * p_nn_upwind, f64_vec2 * p_v_overall_tris)
	
	for (long i = 0; i < NUMTRIANGLES; i++)
	{
		p_cuSyst_compare->p_n_upwind_minor[i].n = p_temphost1[i];
		p_cuSyst_compare->p_n_upwind_minor[i].n_n = p_temphost2[i];
	};
	cudaMemcpy(p_cuSyst_host->p_n_upwind_minor, pX_half->p_n_upwind_minor, sizeof(nvals)*NMINOR, cudaMemcpyDeviceToHost);
	Compare_nvals(p_cuSyst_host->p_n_upwind_minor, p_cuSyst_compare->p_n_upwind_minor, NUMVERTICES * 2);
	
	cudaMemset(p_Integrated_div_v_overall, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Div_v_neut, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Div_v, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(NT_addition_rates_d, 0, sizeof(NTrates)*NUMVERTICES);

	
	kernelAccumulateAdvectiveMassHeatRate << <numTilesMajor, threadsPerTileMajor >> >(
		Timestep,
		pX_half->p_info,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,

		pX_half->p_n_major,
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL,

		pX_half->p_n_upwind_minor,
		pX_half->p_vie,
		pX_half->p_v_n,
		pX_half->p_v_overall_minor,
		p_T_upwind_minor, // may or may not overlap source: don't we only use from tris? so not overlap

		NT_addition_rates_d,
		p_Div_v,
		p_Div_v_neut,
		p_Integrated_div_v_overall);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateAdvectiveMassHeatRate pX_half");
	//

	memset(p_div_v_neut, 0, sizeof(f64)*NUMVERTICES);
	memset(p_div_v, 0, sizeof(f64)*NUMVERTICES);
	memset(NTadditionrates, 0, sizeof(NTrates)*NUMVERTICES);

	// Make this use upward density instead of just referring to shards: 
	pTriMeshhalf->AccumulateAdvectiveMassHeatRate(p_v, NTadditionrates, p_temphost1, p_temphost2);
	
	cudaMemcpy(p_NT_addition_rates, NT_addition_rates_d, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToHost);

	printf("\nDone Advective calc 2 \n\n\n\n\n");


	CallMAC(cudaMemset(p_MAR_neut, 0, sizeof(f64_vec3)*NMINOR));
	CallMAC(cudaMemset(p_MAR_ion, 0, sizeof(f64_vec3)*NMINOR));
	CallMAC(cudaMemset(p_MAR_elec, 0, sizeof(f64_vec3)*NMINOR));

	// Now notice we take a grad Azdot but Azdot has not been defined except from time t_k!!
	kernelCreate_pressure_gradT_and_gradA_LapA_CurlA_minor << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_info,
		pX_half->p_T_minor,
		pX_half->p_AAdot,

		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_izNeigh_TriMinor,
		pX_half->p_szPBC_triminor,
		pX_half->p_who_am_I_to_corner,
		pX_half->p_tri_corner_index,

		p_MAR_neut, p_MAR_ion, p_MAR_elec,
		p_n_shards,				// this kernel is for i+e only
		pX_half->p_n_minor,
		p_GradTe,
		p_GradAz,
		p_LapAz,

		p_ROCAzduetoAdvection, // Would probs be better to split out Az calc, remember
		p_ROCAzdotduetoAdvection, // Would probs be better to split out Az calc, remember
		pX_half->p_v_overall_minor, // it's only this time that we need to collect it ofc.
									// grad Azdot requires storing Azdot. I do not like it.

									// Should make a constant switch.
									// What about grad Azdot          ::slap::
									// Let ROCAzdot = 0 on the first big go-around and it doesn't matter.
									// Let it feed into CalculateVelocityAndAzdot in the leapfrog and BJLS.			

		pX_half->p_B,
		pX_half->p_AreaMinor
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelCreate_pressure_gradT_and_gradA_LapA_CurlA_minor pX_half");

	kernelCreate_momflux_minor << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_info,
		pX_half->p_vie,
		pX_half->p_v_overall_minor,

		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_izNeigh_TriMinor,
		pX_half->p_szPBC_triminor,

		pX_half->p_who_am_I_to_corner,
		pX_half->p_tri_corner_index,

		p_MAR_neut, p_MAR_ion, p_MAR_elec,
		p_n_shards
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelCreate_momflux_minor pX_half");

	kernelNeutral_pressure_and_momflux << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_info,

		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_izNeigh_TriMinor,
		pX_half->p_szPBC_triminor,
		pX_half->p_who_am_I_to_corner,
		pX_half->p_tri_corner_index,

		pX_half->p_T_minor,
		pX_half->p_v_n,
		p_n_shards_n,
		pX_half->p_n_minor,

		pX_half->p_v_overall_minor,
		p_MAR_neut
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelNeutral_pressure_and_momflux pX_half");

	//pX_half->Add_ViscousMomentumFluxRates(AdditionalMomRates);

	memset(AdditionalMomRates, 0, sizeof(three_vec3)*NMINOR);
	
	
	pTriMesh->AntiAdvectAzAndAdvance(0.5*h, pTriMesh, GradAz, pTriMeshhalf);
	// ARE WE MISSING THAT FROM GPU ADVANCE CODE? NOT SURE
	

	pTriMeshhalf->Create_momflux_integral_grad_nT_and_gradA_LapA_CurlA_on_minors(p_v, AdditionalMomRates);

	SetConsoleTextAttribute(hConsole, 10);
	printf("Whaddayathink!\n");
	printf("Whaddayathink!\n\n\n");
	SetConsoleTextAttribute(hConsole, 15);


	Compare_NTrates(p_NT_addition_rates, NTadditionrates);
	 
	kernelCalculateNu_eHeartNu_iHeart_nu_nn_visc << <numTilesMajorClever, threadsPerTileMajorClever >> >(
		pX_half->p_info + BEGINNING_OF_CENTRAL,
		pX_half->p_n_major,
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL,
		p_nu_major);
	Call(cudaThreadSynchronize(), "cudaTS CalculateNu pX_half");

	//char temp[128],temp2[128];
	//cudaMemcpy(temp, pX_half->p_szPBCneigh_vert + MAXNEIGH*CHOSEN, MAXNEIGH, cudaMemcpyDeviceToHost);
	//cudaMemcpy(temp2, this->p_szPBCneigh_vert + MAXNEIGH*CHOSEN, MAXNEIGH, cudaMemcpyDeviceToHost);
	//for (int j = 0; j < MAXNEIGH; j++) printf("j %d PBC %d %d \n", j, (int)(temp2[j]), (int)(temp[j]));

	//T3 temp;
	//cudaMemcpy(&temp, pX_half->p_T_minor + BEGINNING_OF_CENTRAL + 36852, sizeof(T3), cudaMemcpyDeviceToHost);
	//printf("\n\n\n\nCPU 36852 Te %1.14E GPU %1.14E\n\n\n\n",
	//	pTriMeshhalf->pData[36852+BEGINNING_OF_CENTRAL].Te, temp.Te);


	kernelAccumulateDiffusiveHeatRateAndCalcIonisation << <numTilesMajorClever, threadsPerTileMajorClever >> >(
		Timestep,
		pX_half->p_info,
		pX_half->p_cc,
		pX_half->p_izNeigh_vert,
		pX_half->p_szPBCneigh_vert,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_n_major,
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices
		// line 1296 sets up T
		pX_half->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
		p_nu_major,
		NT_addition_rates_d,
		pX_half->p_AreaMajor);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate pX_half");
	// To increase the efficiency we want to make a clever 2nd set of major tiles of size 192. Also try 256, 384.

	pTriMeshhalf->AccumulateDiffusiveHeatRateAndCalcIonisation(h, NTadditionrates); // Wants minor n,T and B
	printf("\nDone Diffusive calc \n\n");

	cudaMemcpy(p_NT_addition_rates, NT_addition_rates_d, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToHost);

	Compare_NTrates(p_NT_addition_rates, NTadditionrates);

	SetConsoleTextAttribute(hConsole, 10);
	kernelAdvanceDensityAndTemperature << <numTilesMajor, threadsPerTileMajor >> >(
		Timestep,
		this->p_info + BEGINNING_OF_CENTRAL,
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL,
		NT_addition_rates_d,
		pX_half->p_n_major,  // ?
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL,  // ?

		pX_half->p_vie + BEGINNING_OF_CENTRAL,
		pX_half->p_v_n + BEGINNING_OF_CENTRAL,

		p_Div_v_neut, p_Div_v,
		p_Integrated_div_v_overall,

		this->p_AreaMajor, // HERE WE USED pX_half->AreaMajor but is that what we want? No, it's original?
		// still not sure why AreaMajor should change by much

		pX_target->p_n_major,
		pX_target->p_T_minor + BEGINNING_OF_CENTRAL
		);
	Call(cudaThreadSynchronize(), "cudaTS Advance_n_and_T 233\n");
	// Includes resistive heating based on pX_half->p_vie

	SetConsoleTextAttribute(hConsole, 15);
	printf("CPU vers:\n");
	SetConsoleTextAttribute(hConsole, 10);
	pTriMesh->AdvanceDensityAndTemperature(h, pTriMeshhalf, pDestMesh, NTadditionrates);
	SetConsoleTextAttribute(hConsole, 15);

	printf("\n");

	p_cuSyst_compare->PopulateFromTriMesh(pDestMesh);
	 
	printf("Compare n_major pX_target vs pDestMesh n_vertex:\n");
	cudaMemcpy(p_cuSyst_host->p_n_major, pX_target->p_n_major, sizeof(nvals)*NUMVERTICES, cudaMemcpyDeviceToHost);
	Compare_nvals(p_cuSyst_host->p_n_major, p_cuSyst_compare->p_n_major, NUMVERTICES);

	nvals temp_nvals;
	printf("runs %d \n", runs);
	cudaMemcpy(&temp_nvals, pX_target->p_n_major + CHOSEN, sizeof(nvals), cudaMemcpyDeviceToHost);
	printf("\nn_n vertex %d %1.14E %1.14E diff %1.14E \n", CHOSEN, temp_nvals.n_n, pDestMesh->pData[CHOSEN + BEGINNING_OF_CENTRAL].n_n,
		temp_nvals.n_n - pDestMesh->pData[CHOSEN + BEGINNING_OF_CENTRAL].n_n);
	SetConsoleTextAttribute(hConsole, 5);
	printf("n_n vertex %d %1.14E %1.14E \n", CHOSEN, temp_nvals.n_n, pDestMesh->pData[CHOSEN + BEGINNING_OF_CENTRAL].n_n);
	SetConsoleTextAttribute(hConsole, 11);
	printf("n_n vertex %d %1.14E %1.14E \n\n\n", CHOSEN, temp_nvals.n_n, pDestMesh->pData[CHOSEN + BEGINNING_OF_CENTRAL].n_n);
	SetConsoleTextAttribute(hConsole, 15);

	printf("Compare T pX_target vs pDestMesh vertices:\n");
	cudaMemcpy(p_cuSyst_host->p_T_minor + BEGINNING_OF_CENTRAL, pX_target->p_T_minor + BEGINNING_OF_CENTRAL, sizeof(T3)*NUMVERTICES, cudaMemcpyDeviceToHost);
	Compare_T3(p_cuSyst_host->p_T_minor + BEGINNING_OF_CENTRAL, p_cuSyst_compare->p_T_minor + BEGINNING_OF_CENTRAL, NUMVERTICES);
	   

	pDestMesh->Average_n_T_to_tris_and_calc_centroids_and_minorpos(); // UPGRADE TO 2ND DEGREE

	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> >(
		pX_target->p_n_minor,
		pX_target->p_n_major,
		pX_target->p_T_minor,
		pX_target->p_info,
		pX_target->p_cc,
		pX_target->p_tri_corner_index,
		pX_target->p_tri_periodic_corner_flags
		); // call before CreateShardModel 
	Call(cudaThreadSynchronize(), "cudaTS average nTx 233");

	// Copy major n to minor n:
	cudaMemcpy(pX_target->p_n_minor + BEGINNING_OF_CENTRAL,
			pX_target->p_n_major, sizeof(nvals)*NUMVERTICES, cudaMemcpyDeviceToDevice);

	p_cuSyst_compare->PopulateFromTriMesh(pDestMesh);

	printf("Compare n pX_target vs pDestMesh\n");
	cudaMemcpy(p_cuSyst_host->p_n_minor, pX_target->p_n_minor, sizeof(nvals)*NMINOR, cudaMemcpyDeviceToHost);
	Compare_nvals(p_cuSyst_host->p_n_minor, p_cuSyst_compare->p_n_minor, NMINOR);

	printf("Compare T pX_target vs pDestMesh:\n");
	cudaMemcpy(p_cuSyst_host->p_T_minor, pX_target->p_T_minor, sizeof(T3)*NMINOR, cudaMemcpyDeviceToHost);
	Compare_T3(p_cuSyst_host->p_T_minor, p_cuSyst_compare->p_T_minor, NMINOR);

	printf("================================\n\n\n\n\n\npoint 200 \n\n\n\n");


	// Now set up inputs such as AMR with advective momflux and aTP, and B
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		// [ do Az advance above when we advance Azdot. ]
	

	p_cuSyst_compare->PopulateFromTriMesh(pTriMeshhalf);
	cudaMemcpy(p_cuSyst_host->p_T_minor, pX_half->p_T_minor, NMINOR * sizeof(T3), cudaMemcpyDeviceToHost);
	Compare_T3(p_cuSyst_host->p_T_minor, p_cuSyst_compare->p_T_minor, NMINOR);
	printf("compared T (pX_half, pTriMeshhalf).\n");

	// Now we are looking to compare p_MAR_ion and p_MAR_elec, gradAz, gradTe, curlAz
	cudaMemcpy(p_temphost2, pX_half->p_AreaMinor, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
	Compare_f64(p_temphost2, pTriMeshhalf->AreaMinorArray, NMINOR);
//
//	Areas show a difference at 10th place.
//	Shall we accept that?
//
//	MAR shows difference at 8th place. Perhaps a good idea to go back and investigate how that came about.
//	Is it just floating point differences?
//
//
	printf("Grad Te:\n");
	cudaMemcpy(p_GradTe_host, p_GradTe, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToHost);
	Compare_f64_vec2(p_GradTe_host, GradTeArray, NMINOR);

	printf("Grad Az:\n");
	cudaMemcpy(p_GradAz_host, p_GradAz, sizeof(f64_vec2)*NMINOR, cudaMemcpyDeviceToHost);
	Compare_f64_vec2(p_GradAz_host, GradAz, NMINOR);

	printf("B:\n");
	cudaMemcpy(p_B_host, pX_half->p_B, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToHost);
	Compare_f64_vec3(p_B_host, p_cuSyst_compare->p_B, NMINOR);
	printf("Compare to %d\n", NUMVERTICES * 2 - 1000);
	Compare_f64_vec3(p_B_host+1000, p_cuSyst_compare->p_B+1000, NUMVERTICES*2-2000);


	// unparcel AdditionalMomRates:
	for (int qqq = 0; qqq < NMINOR; qqq++)
	{
		p_MAR_ion_compare[qqq] = AdditionalMomRates[qqq].ion;
		p_MAR_elec_compare[qqq] = AdditionalMomRates[qqq].elec;
		p_MAR_neut_compare[qqq] = AdditionalMomRates[qqq].neut;
	};

	cudaMemcpy(p_MAR_ion_host, p_MAR_ion, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_MAR_elec_host, p_MAR_elec, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_MAR_neut_host, p_MAR_neut, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToHost);

	printf("MAR_ion:\n");
	Compare_f64_vec3(p_MAR_ion_host, p_MAR_ion_compare, NMINOR);
	printf("MAR_elec:\n");
	Compare_f64_vec3(p_MAR_elec_host, p_MAR_elec_compare, NMINOR);
	printf("MAR_neut:\n");
	Compare_f64_vec3(p_MAR_neut_host, p_MAR_neut_compare, NMINOR);
	
	printf("up to subcycle. press o\n");
	char o;
//	do {
//		o = getch();
//	} while (o != 'o');

	// The rest should be relatively straightforward, keeping it simple.
	
	//-----------------------
	// Set major areas: do without?


#ifndef USE_N_MAJOR_FOR_VERTEX
	kernelCreateShardModelOfDensities_And_SetMajorArea << <numTilesMajor, threadsPerTileMajor >> >(
		pX_target->p_info,
		pX_target->p_n_major,
		pX_target->p_n_minor,
		pX_target->p_izTri_vert,
		pX_target->p_szPBCtri_vert, // could be more intelligent way than storing and copying 3x
		p_n_shards,
		p_n_shards_n,
		//		p_Tri_n_lists,
		//		p_Tri_n_n_lists,
		pX_target->p_AreaMajor);
	Call(cudaThreadSynchronize(), "cudaTS CreateShardModel pX_target");

	pDestMesh->CreateShardModelOfDensities_And_SetMajorArea();



	kernelInferMinorDensitiesFromShardModel << <numTriTiles, threadsPerTileMinor >> >(
		pX_target->p_info,
		pX_target->p_n_minor,
		p_n_shards,
		p_n_shards_n,
		pX_target->p_tri_corner_index,
		pX_target->p_who_am_I_to_corner);// (At the moment just repopulating tri minor n.)
	Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities pX_target");

	// We'd then need to call calc to set n_minor verts to n_shards.n_cent

#endif

#define NUMCOLS 6

	//f64 * storageAz = (f64 *)malloc(sizeof(f64)*450*NUMCOLS);
	//f64 * storageLap = (f64 *)malloc(sizeof(f64) * 450 * NUMCOLS);
	//f64 * storageAzdot0 = (f64 *)malloc(sizeof(f64) * 450);
	//f64 * storageGamma = (f64 *)malloc(sizeof(f64) * 450);

	// store Az and Lap differences from CPU and then output these.

	int ColIndex;

	f64 starttime = evaltime;

	if (runs % 10 == 0)
	{
		// BACKWARD STEPS:
		kernelPullAzFromSyst << <numTilesMinor, threadsPerTileMinor >> >(
			this->p_AAdot,
			p_Az
			);
		Call(cudaThreadSynchronize(), "cudaTS PullAz");

		kernelGetLapCoeffs << <numTriTiles, threadsPerTileMinor >> >(
			pX_half->p_info,
			pX_half->p_izTri_vert,
			pX_half->p_izNeigh_TriMinor,
			pX_half->p_szPBCtri_vert,
			pX_half->p_szPBC_triminor,
			p_LapCoeffself
			);
		Call(cudaThreadSynchronize(), "cudaTS GetLapCoeffs");
		
	//	FILE * fp = fopen("analyse.txt", "w");
	//	FILE * fp1 = fopen("whatgives.txt", "w");

		pTriMeshhalf->GetLapCoeffs();


		f64 Azdotarrays[22][40], Azarrays[22][40];
		f64 Azarraytemp[40];
		AAdot AAdotarray[40];
		int i;

		// SUBCYCLES
		for (iSubstep = 0; iSubstep < SUBCYCLES; iSubstep++)
		{
			SetConsoleTextAttribute(hConsole, 14);
			printf("\n\n#########################\n");
			printf(    "####### SUBSTEP %d #######\n\n",iSubstep);
			SetConsoleTextAttribute(hConsole, 15);

			//// Record Az, Azdot for 2181-2220
			//if (iSubstep == 0) {
			//	cudaMemcpy(AAdotarray, this->p_AAdot + 2180, sizeof(AAdot) * 40, cudaMemcpyDeviceToHost);
			//}
			//else {
			//	cudaMemcpy(AAdotarray, pX_target->p_AAdot + 2180, sizeof(AAdot) * 40, cudaMemcpyDeviceToHost);
			//}
			//cudaMemcpy(Azarraytemp, p_Az + 2180, sizeof(f64) * 40, cudaMemcpyDeviceToHost);
			//for (i = 0; i < 40; i++)
			//{
			//	Azdotarrays[iSubstep][i] = AAdotarray[i].Azdot;
			//	Azarrays[iSubstep][i] = Azarraytemp[i];
			//}



			evaltime += 0.5*SUBSTEP; // Now halfway through substep

			kernelInterpolateVarsAndPositions << <numTilesMinor, threadsPerTileMinor >> >(
				(evaltime - starttime) / Timestep,
				this->p_info,
				pX_target->p_info,
				this->p_n_minor,
				pX_target->p_n_minor,
				this->p_T_minor,
				pX_target->p_T_minor,
	//			this->p_B,
	//			pX_target->p_B,
				pX_half->p_info,
				pX_half->p_n_minor,
				pX_half->p_T_minor
	//			pX_half->p_B
				);
			Call(cudaThreadSynchronize(), "cudaTS InterpolateVars");
			 
			//pX_half->GetLapFromCoeffs(Az_array, LapAzArray);
			// NOTICE # BLOCKS -- THIS SHOULD ALSO APPLY WHEREVER WE DO SIMILAR THING LIKE WITH MOMFLUX.
			
			SetConsoleTextAttribute(hConsole, 13);
			  
			kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> >(
				pX_half->p_info,
				p_Az,
				pX_half->p_izTri_vert,
				pX_half->p_izNeigh_TriMinor,
				pX_half->p_szPBCtri_vert,
				pX_half->p_szPBC_triminor,
				p_LapAz,
		//		p_temp1, p_temp2, p_temp3,
				pX_half->p_AreaMinor
				);
			Call(cudaThreadSynchronize(), "cudaTS GetLapMinor bbb");
			
			// Looks like Az is never pulled on CPU ! Sort that!!

			pTriMesh->InterpolateVarsAndPositions(pTriMeshhalf, pDestMesh, (evaltime - starttime) / Timestep);
			//pHalfMesh->GetLapFromCoeffs(Az_array, LapAzArray);

			pTriMeshhalf->GetLap(Az_array, LapAzArray);

			SetConsoleTextAttribute(hConsole, 15);
			printf("Lap: substep %d \n", iSubstep);

			cudaMemcpy(p_temphost1, p_LapAz, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			Compare_f64(p_temphost1, LapAzArray, NMINOR);


			printf("Az: substep %d \n", iSubstep);

			cudaMemcpy(p_temphost2, p_Az, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			Compare_f64(p_temphost2, Az_array, NMINOR);


			// The equation is A_k+1 - h~gamma~ Lap A_k+1 - A_k - h~Azdot0~ = 0
			// Calculate regressor x_Jacobi from eps/coeff_on_A_i
			// Given a proposed addition of a regressor x, deps/dbeta = x - h~gamma~ Lap x 
			// Set beta = -sum[eps deps/dbeta] / sum[deps/dbeta ^2]

			// evaltime + 0.5*SUBSTEP used for setting EzStrength://
			Iz_prescribed_endtime = GetIzPrescribed(evaltime + 0.5*SUBSTEP); 
			 // we already added +0.5*SUBSTEP so this is for being APPLIED AT END of substep
		//	Iz_prescribed_starttime = GetIzPrescribed(evaltime - 0.5*SUBSTEP);
			

			kernelEstimateCurrent << <numTilesMinor, threadsPerTileMinor >> >(
				pX_half->p_info,
				pX_half->p_n_minor,
				(iSubstep == 0) ? this->p_vie : pX_target->p_vie,
				pX_half->p_AreaMinor,
				p_temp1); // Iz_k
			Call(cudaThreadSynchronize(), "cudaTS Estimate Iz_k");

			kernelPopulateOhmsLaw << <numTilesMinor, threadsPerTileMinor >> >(
				SUBSTEP,// ROCAzdotduetoAdvection, 
				pX_half->p_info,
				p_MAR_neut, p_MAR_ion, p_MAR_elec,
				pX_half->p_B,
				p_LapAz,
				p_GradAz,
				p_GradTe,

				pX_half->p_n_minor,
				pX_half->p_T_minor,
				(iSubstep == 0) ? this->p_vie : pX_target->p_vie,
				(iSubstep == 0) ? this->p_v_n : pX_target->p_v_n,
				(iSubstep == 0) ? this->p_AAdot : pX_target->p_AAdot, //	inputs
				pX_half->p_AreaMinor, // pop'd? interp?
				p_ROCAzdotduetoAdvection,

				p_vn0,
				p_v0,
				p_OhmsCoeffs,
				pX_half->p_AAdot, // intermediate value ............................
								  // .....................................................
				p_Iz0_summands,
				p_sigma_Izz,
				p_denom_i,
				p_denom_e, 
				p_coeff_of_vez_upon_viz, p_beta_ie_z,
				true,
				false, // (iSubstep == SUBCYCLES-1),
				pX_target->p_n_minor
				);
			Call(cudaThreadSynchronize(), "cudaTS kernelPopulateOhmsLaw ");

			cudaMemcpy(p_Iz0_summands_host, p_Iz0_summands, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_summands_host, p_sigma_Izz, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			Iz0 = 0.0;
			f64 Sigma_Izz = 0.0;
			Iz_k = 0.0;
			long iBlock;
		//	FILE * fp_gpu = fopen("gpu_sigma.txt","w");
			for (iBlock = 0; iBlock < numTilesMinor; iBlock++)
			{
				Iz0 += p_Iz0_summands_host[iBlock];
				Sigma_Izz += p_summands_host[iBlock];
				Iz_k += p_temphost1[iBlock];
		//		fprintf(fp_gpu, "%d %1.14E \n", iBlock, p_summands_host[iBlock]);
			}
		//	fclose(fp_gpu);
					
			EzStrength_ = (Iz_prescribed_endtime - Iz0) / Sigma_Izz;
			Set_f64_constant(Ez_strength, EzStrength_);
			neg_Iz_per_triangle = -0.5*(Iz_k + Iz_prescribed_endtime) / (f64)numReverseJzTriangles;
			Set_f64_constant(negative_Iz_per_triangle, neg_Iz_per_triangle);
			// Electrons travel from cathode to anode so Jz is down in filament,
			// up around anode.
			printf("neg_Iz_per_triangle %1.14E Iz_presc %1.14E numRev %d \n", neg_Iz_per_triangle, Iz_prescribed_endtime,
				numReverseJzTriangles);

			SetConsoleTextAttribute(hConsole, 13);
			printf("\nGPU: Iz0 = %1.14E SigmaIzz %1.14E EzStrength = %1.14E \n\n", Iz0, Sigma_Izz, EzStrength_);
			SetConsoleTextAttribute(hConsole, 15);

			kernelCreateLinearRelationship << <numTilesMinor, threadsPerTileMinor >> >(
				SUBSTEP,
				pX_half->p_info,
				p_OhmsCoeffs,
				p_v0,
				p_LapAz,
				pX_half->p_n_minor, // this is the reported n
				p_denom_e,
				p_denom_i, 
				p_coeff_of_vez_upon_viz, 
				p_beta_ie_z,
				pX_half->p_AAdot, 
				pX_half->p_AreaMinor,
				p_Azdot0,
				p_gamma
				); // MODIFY vez0, viz0 IF THEY WILL EVER BE USED.
			Call(cudaThreadSynchronize(), "cudaTS kernelCreateLinearRelationship ");
			SetConsoleTextAttribute(hConsole, 11);
			
			if (iSubstep == 0) {
				pTriMesh->Accelerate2018(SUBSTEP,// ROCAzdotduetoAdvection, 
					pTriMeshhalf, // reported n comes from here
					pDestMesh, evaltime + 0.5*SUBSTEP,
					true, // bool bFeint
					false // (iSubstep == SUBCYCLES - 1) // bool bUse_n_dest_for_Iz
				);
			} else {
				pDestMesh->Accelerate2018(SUBSTEP,// ROCAzdotduetoAdvection, 
					pTriMeshhalf, 
					pDestMesh, evaltime + 0.5*SUBSTEP,
					true, // bool bFeint
					false // (iSubstep == SUBCYCLES - 1) // bool bUse_n_dest_for_Iz
				);
			}
			SetConsoleTextAttribute(hConsole, 15);
			 
			cudaMemcpy(p_temphost1, p_Azdot0, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost2, p_gamma, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			printf("Azdot0:\n");
			Compare_f64(p_temphost1, Azdot0, NMINOR);
			printf("gamma:\n\n&&&&&&&&&&&&&&&\n");
			Compare_f64(p_temphost2, gamma, NMINOR);
		
			// Error at 10th place for reverseJz is caused apparently by AreaMinor not Iz_cell

		
			kernelCreateSeedPartOne << <numTilesMinor, threadsPerTileMinor >> > (
				SUBSTEP,
				p_Az,
				(iSubstep == 0) ? this->p_AAdot : pX_target->p_AAdot, // use 0.5*(Azdot_k + Azdot_k+1) for seed.
				p_AzNext);
			Call(cudaThreadSynchronize(), "cudaTS Create Seed 1");
			// AzNext = Az + 0.5 h p_AAdot_use[iMinor].Azdot

			// Question whether this is even wanted: think no use for it.
			// Did not save adjustment to viz0 -- correct?
			//
			//			kernelCalculateVelocityAndAzdot << <numTilesMinor, threadsPerTileMinor >> >(
			//				SUBSTEP,
			//				p_vn0,
			//				p_v0,
			//				p_OhmsCoeffs,
			//				pX_half->p_AAdot,
			//
			//				pX_target->p_AAdot,
			//				pX_target->p_vie,
			//				pX_target->p_v_n
			//				);
			//			Call(cudaThreadSynchronize(), "cudaTS kernelUpdate_v ");
			//
			//Az_array_next[iMinor] = Az_array[iMinor] + 0.5*SUBSTEP*this->pData[iMinor].Azdot 
			//   + 0.5*SUBSTEP * Azdot0[iMinor] + 0.5*SUBSTEP * gamma[iMinor] * LapAzArray[iMinor];
			//ie use 0.5*(Azdot_k[done] + Azdot_k+1) for seed.

			kernelCreateSeedPartTwo << <numTilesMinor, threadsPerTileMinor >> > (
				SUBSTEP,
				p_Azdot0, p_gamma, p_LapAz,
				p_AzNext);
			Call(cudaThreadSynchronize(), "cudaTS Create Seed 2"); // Okay -- we can now merge these. "Azdot_k" is preserved.
			// += 0.5h p_Azdot0 + p_gamma*p_LapAz
			
			for (iMinor = 0; iMinor < NMINOR; iMinor++)
				Az_array_next[iMinor] = Az_array[iMinor] + 0.5*SUBSTEP*
				((iSubstep == 0)?pTriMesh->pData[iMinor].Azdot:pDestMesh->pData[iMinor].Azdot)
				    + 0.5*SUBSTEP * Azdot0[iMinor] + 0.5*SUBSTEP * gamma[iMinor] * LapAzArray[iMinor];
			
			// JLS:
			// Going to need to store L2eps from here if we want to output for CPU & GPU, obviously

			pTriMeshhalf->JLS_for_Az_bwdstep(4, SUBSTEP); // populate Az_array with k+1 values
			
			f64 sum_eps_deps_by_dbeta, sum_depsbydbeta_sq, sum_eps_eps, depsbydbeta;
			printf("\nJLS [beta L2eps]: ");
			long iMinor;
			f64 beta, L2eps;
			Triangle * pTri;
			int iIteration;
			for (iIteration = 0; iIteration < 4; iIteration++)
			{
				// 1. Create regressor:
				// Careful with major vs minor + BEGINNING_OF_CENTRAL:

				kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> >(
					pX_half->p_info,
					p_AzNext,
					pX_half->p_izTri_vert,
					pX_half->p_izNeigh_TriMinor,
					pX_half->p_szPBCtri_vert,
					pX_half->p_szPBC_triminor,
					p_LapAzNext,
			//		p_temp1, p_temp2, p_temp3,
					pX_half->p_AreaMinor
					);
				Call(cudaThreadSynchronize(), "cudaTS GetLap Az JLS 1");

				kernelCreateEpsilonAndJacobi << <numTilesMinor, threadsPerTileMinor >> >
					(SUBSTEP, // ?
						pX_half->p_info,
						p_AzNext, p_Az,
						p_Azdot0, p_gamma,
						p_LapCoeffself, p_LapAzNext,
						p_epsilon, p_Jacobi_x);
				Call(cudaThreadSynchronize(), "cudaTS CreateEpsAndJacobi 1");

				kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> >(
					pX_half->p_info,
					p_Jacobi_x,
					pX_half->p_izTri_vert,
					pX_half->p_izNeigh_TriMinor,
					pX_half->p_szPBCtri_vert,
					pX_half->p_szPBC_triminor,
					p_LapJacobi,
			//		p_temp1, p_temp2, p_temp3,
					pX_half->p_AreaMinor
					);
				Call(cudaThreadSynchronize(), "cudaTS Get Lap Jacobi 1");

				kernelAccumulateSummands << <numTilesMinor, threadsPerTileMinor >> > (
					pX_half->p_info,
					SUBSTEP,
					p_epsilon, p_Jacobi_x, p_LapJacobi, p_gamma,
					p_sum_eps_deps_by_dbeta,
					p_sum_depsbydbeta_sq,
					p_sum_eps_eps);
				Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands 1");
				
				cudaMemcpy(p_sum_depsbydbeta_sq_host, p_sum_depsbydbeta_sq, sizeof(f64)*numTilesMinor,
					cudaMemcpyDeviceToHost);
				cudaMemcpy(p_sum_eps_deps_by_dbeta_host, p_sum_eps_deps_by_dbeta, sizeof(f64)*numTilesMinor,
					cudaMemcpyDeviceToHost);
				cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMinor,
					cudaMemcpyDeviceToHost);

				sum_eps_deps_by_dbeta = 0.0;
				sum_depsbydbeta_sq = 0.0;
				sum_eps_eps = 0.0;
				for (iTile = 0; iTile < numTilesMinor; iTile++)
				{
					sum_eps_deps_by_dbeta += p_sum_eps_deps_by_dbeta_host[iTile];
					sum_depsbydbeta_sq += p_sum_depsbydbeta_sq_host[iTile];
					sum_eps_eps += p_sum_eps_eps_host[iTile];
				}
				beta = -sum_eps_deps_by_dbeta / sum_depsbydbeta_sq;
				L2eps = sqrt(sum_eps_eps / (real)NMINOR);

				SetConsoleTextAttribute(hConsole, 13);
				printf("GPU [ beta %1.14E L2eps %1.14E ] ", beta, L2eps);
				printf("sum_eps_deps_bydbeta %1.14E sum_depsbydbeta_sq %1.14E \n",
					sum_eps_deps_by_dbeta , sum_depsbydbeta_sq);
				// Probably will show difference in both.
				// Differences have to be caused by differences in eps or deps/dbeta;
				// We might want to assess LapJacobi difference
				SetConsoleTextAttribute(hConsole, 15);
			
				kernelAdd << <numTilesMinor, threadsPerTileMinor >> > (
					p_AzNext, beta, p_Jacobi_x);
				//for (iMinor = 0; iMinor < NMINOR; iMinor++)
				//	Az_array_next[iMinor] += beta * Jacobi_x[iMinor];
				Call(cudaThreadSynchronize(), "cudaTS Add 1");
				
				// Try resetting frills here and ignoring in calculation:
				kernelResetFrillsAz << <numTriTiles, threadsPerTileMinor >> > (
					this->p_info, this->p_tri_neigh_index, p_AzNext);
				Call(cudaThreadSynchronize(), "cudaTS ResetFrills X");

			};

			printf("\n\n");
			
			cudaMemcpy(p_Az, p_AzNext, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
			memcpy(Az_array, Az_array_next, sizeof(f64)*NMINOR);
			
//			cudaMemcpy(&tempf64, &(p_Az[CHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
	//		printf("POINT 1 CHOSEN %d p_Az %1.15E \n", CHOSEN, tempf64);

			// That was:
			//	JLS_for_Az_bwdstep(4, SUBSTEP); // populate Az_array with k+1 values
			
			kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> >(
				pX_half->p_info,
				p_Az,
				pX_half->p_izTri_vert,
				pX_half->p_izNeigh_TriMinor,
				pX_half->p_szPBCtri_vert,
				pX_half->p_szPBC_triminor,
				p_LapAz,
		//		p_temp1, p_temp2, p_temp3,
				pX_half->p_AreaMinor
				);
			Call(cudaThreadSynchronize(), "cudaTS GetLap Az JLS 2");

			pTriMeshhalf->GetLap(Az_array, LapAzArray);

			// Now let's compare Az and LapAz:

			cudaMemcpy(p_temphost1, p_Az, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost2, p_LapAz, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);

			printf("POINT 2 %d LapAz %1.15E \n", CHOSEN, p_temphost2[CHOSEN]);

		//	printf("Az:\n");
		//	Compare_f64(p_temphost1, Az_array, NMINOR);

		//	printf("LapAz:\n");
		//	Compare_f64(p_temphost2, LapAzArray, NMINOR);

			// Leaving Iz_prescribed and reverse_Jz the same:

			// Think I'm right all that has changed is LapAz so do we really have to go through whole thing again? :

			kernelPopulateOhmsLaw << <numTilesMinor, threadsPerTileMinor >> > (
				SUBSTEP,// ROCAzdotduetoAdvection, 
				pX_half->p_info,
				p_MAR_neut, p_MAR_ion, p_MAR_elec,
				pX_half->p_B,
				p_LapAz,
				p_GradAz,
				p_GradTe,

				pX_half->p_n_minor,
				pX_half->p_T_minor,

				(iSubstep == 0) ? this->p_vie : pX_target->p_vie,
				(iSubstep == 0) ? this->p_v_n : pX_target->p_v_n,
				(iSubstep == 0) ? this->p_AAdot : pX_target->p_AAdot, //	src
				pX_half->p_AreaMinor, // pop'd? interp?
				p_ROCAzdotduetoAdvection,

				p_vn0,
				p_v0,
				p_OhmsCoeffs,
				pX_half->p_AAdot, // intermediate value ............................
								  // .....................................................
				p_Iz0_summands,
				p_sigma_Izz,
				p_denom_i,
				p_denom_e, p_coeff_of_vez_upon_viz, p_beta_ie_z,
				false,
				false, // (iSubstep == SUBCYCLES-1),
				pX_target->p_n_minor
				);
				// Might as well recalculate Ez_strength again :
				// Iz already set for t+SUBSTEP.
			cudaMemcpy(p_Iz0_summands_host, p_Iz0_summands, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_summands_host, p_sigma_Izz, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			Iz0 = 0.0;
			Sigma_Izz = 0.0;
			for (iBlock = 0; iBlock < numTilesMinor; iBlock++)
			{
				Iz0 += p_Iz0_summands_host[iBlock];
				Sigma_Izz += p_summands_host[iBlock];
			}
			EzStrength_ = (Iz_prescribed_endtime - Iz0) / Sigma_Izz;
			Set_f64_constant(Ez_strength, EzStrength_);

			printf("\nGPU EzStrength_ %1.14E Iz0 %1.14E Sigma_Izz %1.14E Iz_prescribed %1.14E\n\n",
				EzStrength_, Iz0, Sigma_Izz, Iz_prescribed_endtime);
			
			// Iz_k has not changed

			kernelCalculateVelocityAndAzdot << <numTilesMinor, threadsPerTileMinor >> >(
				SUBSTEP,
				pX_half->p_info,
				p_vn0,
				p_v0,
				p_OhmsCoeffs,
				pX_half->p_AAdot,
				pX_half->p_n_minor,
				pX_half->p_AreaMinor,
				 
				pX_target->p_AAdot,  // not understanding
				pX_target->p_vie,
				pX_target->p_v_n
				);
			Call(cudaThreadSynchronize(), "cudaTS kernelUpdate_v ");
			
			if (iSubstep == 0) {
				pTriMesh->Accelerate2018(SUBSTEP, pTriMeshhalf, pDestMesh,
					evaltime + 0.5*SUBSTEP, false,
					false//, (iSubstep == SUBCYCLES - 1)
				); // Lap Az now given.
			} else {
				pDestMesh->Accelerate2018(SUBSTEP, pTriMeshhalf, pDestMesh,
					evaltime + 0.5*SUBSTEP, false,
					false//(iSubstep == SUBCYCLES - 1)
				); // Lap Az now given.
			};
			 
			// Compare v after acceleration
			printf("Compare v_e: \n");
			cudaMemcpy(p_cuSyst_host->p_vie, pX_target->p_vie, sizeof(v4)*NMINOR, cudaMemcpyDeviceToHost);
			f64_vec3 * ve_host = (f64_vec3 *)malloc(sizeof(f64_vec3)*NMINOR);
			f64_vec3 * ve_compare = (f64_vec3 *)malloc(sizeof(f64_vec3)*NMINOR);
			for (long i = 0; i < NMINOR; i++)
			{
				ve_compare[i].x = pDestMesh->pData[i].vxy.x;
				ve_compare[i].y = pDestMesh->pData[i].vxy.y;
				ve_compare[i].z = pDestMesh->pData[i].vez;
				ve_host[i].x = p_cuSyst_host->p_vie[i].vxy.x;
				ve_host[i].y = p_cuSyst_host->p_vie[i].vxy.y;
				ve_host[i].z = p_cuSyst_host->p_vie[i].vez;
			}
			Compare_f64_vec3(ve_host, ve_compare, NMINOR);
			
			printf("Compare v_n (pX_target, pDestMesh): \n");
			cudaMemcpy(p_cuSyst_host->p_v_n, pX_target->p_v_n, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToHost);
			for (long i = 0; i < NMINOR; i++)
			{
				vn_compare[i] = pDestMesh->pData[i].v_n;
			}
			Compare_f64_vec3(p_cuSyst_host->p_v_n, vn_compare, NMINOR);
			
			free(ve_host);
			free(ve_compare);

			// Compare Azdot:
			cudaMemcpy(p_cuSyst_host->p_AAdot, pX_target->p_AAdot, sizeof(AAdot)*NMINOR, cudaMemcpyDeviceToHost);
			for (long i = 0; i < NMINOR; i++)
			{
				p_temphost1[i] = p_cuSyst_host->p_AAdot[i].Azdot;
				p_temphost2[i] = pDestMesh->pData[i].Azdot;
			}
			printf("Azdot:\n");
			Compare_f64(p_temphost1, p_temphost2, NMINOR);
			
		//	fprintf(fp, "GPU Azdot %d %1.14E CPU %1.14E \n", CHOSEN, p_temphost1[CHOSEN], p_temphost2[CHOSEN]);

			evaltime += 0.5*SUBSTEP;

		}; // next substep
		
		//fclose(fp);
		//fclose(fp1);

		kernelPushAzInto_dest << <numTilesMinor, threadsPerTileMinor >> >(
			pX_target->p_AAdot,
			p_Az);
		Call(cudaThreadSynchronize(), "cudaTS PushAzIntoDest");

		for (iMinor = 0; iMinor < NMINOR; iMinor++)
			pDestMesh->pData[iMinor].Az = Az_array[iMinor];
		
		cudaMemcpy(AAdotarray, pX_target->p_AAdot + 2180, sizeof(AAdot) * 40, cudaMemcpyDeviceToHost);
		for (i = 0; i < 40; i++)
		{
			Azdotarrays[iSubstep][i] = AAdotarray[i].Azdot;
			Azarrays[iSubstep][i] = AAdotarray[i].Az;
		}

		FILE * gerald = fopen("gerald.txt", "w");
		for (i = 0; i < 40; i++)
		{
			fprintf(gerald, "%d Az ", i + 2180);
			for (int j = 0; j < SUBCYCLES + 1; j++)
				fprintf(gerald, "%1.15E ", Azarrays[j][i]);
			fprintf(gerald, "  Azdot  ");
			for (int j = 0; j < SUBCYCLES + 1; j++)
				fprintf(gerald, "%1.15E ", Azdotarrays[j][i]);
			fprintf(gerald, "\n");
		}
		fclose(gerald);
		///////


		// more advanced implicit could be possible and effective.
		// It is almost certain that splitting up BJLS into a few goes in each set of subcycles would be more effective than being a different set all BJLS.
		// This should be experimented with, once it matches CPU output.
	} else {
	  
		kernelPopulateArrayAz << <numTilesMinor, threadsPerTileMinor >> >(
			0.5*SUBSTEP,
			this->p_AAdot,
			p_ROCAzduetoAdvection,
			p_Az
			);   // This is where we create the f64 array of Az from a short step using Adot_k
				 // We can now see that having AAdot in one object was counterproductive.
		Call(cudaThreadSynchronize(), "cudaTS PopulateArrayAz");

		pTriMesh->Create_A_from_advance(0.5*SUBSTEP, ROCAzduetoAdvection, Az_array); // from *this
		
		kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> >(
			this->p_info,
			this->p_tri_neigh_index,
			p_Az
			);
		Call(cudaThreadSynchronize(), "cudaTS kernelResetFrillsAz 1");
		// Create_A_from_advance(0.5*SUBSTEP, ROCAzduetoAdvection, Az_array); // from *this

		// ===========
		printf("Compare Az: \n");
		cudaMemcpy(p_temphost1, p_Az, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
		Compare_f64(p_temphost1, Az_array, NMINOR);
		printf("\n\n\n");
		// ===========

		for (iSubstep = 0; iSubstep < SUBCYCLES; iSubstep++)
		{
			SetConsoleTextAttribute(hConsole, 14);
			printf("\n\n#########################\n");
			printf("####### SUBSTEP %d #######\n\n", iSubstep);
			SetConsoleTextAttribute(hConsole, 15);

			// Interpolate vars: we could have done it by creating B_k+1 and interpolating back --- we just didn't.

			// If anything to do better we should recalculate Grad A, Curl A every substep when we calculate Lap --
			// that wouldn't cost much more presumably.
			// Not clear it serves a purpose.
			// So let's just drop the interpolation of B for now.

			evaltime += 0.5*SUBSTEP;
			kernelInterpolateVarsAndPositions << <numTilesMinor, threadsPerTileMinor >> >(
				(evaltime - starttime) / Timestep,
				this->p_info,
				pX_target->p_info,
				this->p_n_minor,
				pX_target->p_n_minor,
				this->p_T_minor,
				pX_target->p_T_minor,
			//	this->p_B,
			//	pX_target->p_B,
				pX_half->p_info,
				pX_half->p_n_minor,
				pX_half->p_T_minor
			//	pX_half->p_B
				);
			Call(cudaThreadSynchronize(), "cudaTS InterpolateVars");
			// let n,T,x be interpolated on to pX_half. B remains what we populated there.
			// Recalculate areas, or tween them, would make good sense as well.

			pTriMesh->InterpolateVarsAndPositions(pTriMeshhalf, pDestMesh, (evaltime - starttime) / Timestep);
			
			// It might be nice to tween the thermal pressure grad(nT) as well. And logical:
			// InterpolateArrays(AdditionalMomRates_start, AdditionalMomRates_end, AdditionalMomRates, (evaltime - starttime) / Timestep);
			// Have a look how AMR is created.
			kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> >(
				pX_half->p_info,
				p_Az,
				pX_half->p_izTri_vert,
				pX_half->p_izNeigh_TriMinor,
				pX_half->p_szPBCtri_vert,
				pX_half->p_szPBC_triminor,
				p_LapAz,
		//		p_temp1, p_temp2, p_temp3,
				pX_half->p_AreaMinor
				);
			Call(cudaThreadSynchronize(), "cudaTS GetLap Az Leapfrog 1");

			pTriMeshhalf->GetLap(Az_array, LapAzArray);

			// ===============
			printf("Compare LapAz:\n");
			cudaMemcpy(p_temphost2, p_LapAz, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			Compare_f64(p_temphost2, LapAzArray, NMINOR);
			// ===============

			// evaltime + 0.5*SUBSTEP used for setting EzStrength://
			Iz_prescribed_endtime = GetIzPrescribed(evaltime + 0.5*SUBSTEP);
		//	Iz_prescribed_starttime = GetIzPrescribed(evaltime - 0.5*SUBSTEP);
		
			// On the first step we use "this" as src, otherwise pX_targ to pX_targ
			// Simple plan:
			// Pop Ohms just populate's Ohms and advances Azdot to an intermediate state

			SetConsoleTextAttribute(hConsole, 14);

			kernelEstimateCurrent << <numTilesMinor, threadsPerTileMinor >> >(
				pX_half->p_info,
				pX_half->p_n_minor,
				(iSubstep == 0) ? this->p_vie : pX_target->p_vie,
				pX_half->p_AreaMinor,
				p_temp1); // Iz_k
			Call(cudaThreadSynchronize(), "cudaTS Estimate Iz_k");

			kernelPopulateOhmsLaw << <numTilesMinor, threadsPerTileMinor >> >(
				SUBSTEP,// ROCAzdotduetoAdvection, 
				pX_half->p_info,
				p_MAR_neut, p_MAR_ion, p_MAR_elec,
				pX_half->p_B,
				p_LapAz,
				p_GradAz,
				p_GradTe,
				 
				pX_half->p_n_minor,
				pX_half->p_T_minor,
				(iSubstep == 0) ? this->p_vie : pX_target->p_vie,
				(iSubstep == 0) ? this->p_v_n : pX_target->p_v_n,
				(iSubstep == 0) ? this->p_AAdot : pX_target->p_AAdot,
				pX_half->p_AreaMinor, // pop'd????????? interp?
				p_ROCAzdotduetoAdvection,

				p_vn0,
				p_v0,
				p_OhmsCoeffs,
				pX_half->p_AAdot, // intermediate value ............................

				p_Iz0_summands,
				p_sigma_Izz,
				p_denom_i,
				p_denom_e, p_coeff_of_vez_upon_viz, p_beta_ie_z,
				false,
				false , // (iSubstep == SUBCYCLES-1), // if not, use pX_half->p_n_minor for Iz -- roughly right but wrong
				pX_target->p_n_minor); 
			Call(cudaThreadSynchronize(), "cudaTS kernelPopulateOhmsLaw ");

			cudaMemcpy(p_Iz0_summands_host, p_Iz0_summands, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_summands_host, p_sigma_Izz, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			Iz0 = 0.0; sigma_Izz = 0.0; Iz_k = 0.0;
			for (int iBlock = 0; iBlock < numTilesMinor; iBlock++)
			{
				Iz0 += p_Iz0_summands_host[iBlock];
				sigma_Izz += p_summands_host[iBlock];
				Iz_k += p_temphost1[iBlock];
			}
			EzStrength_ = (Iz_prescribed_endtime - Iz0) / sigma_Izz;
			Set_f64_constant(Ez_strength, EzStrength_);
			f64 neg_Iz_per_triangle = -0.5*(Iz_prescribed_endtime + Iz_k) / (f64)numReverseJzTriangles;
			Set_f64_constant(negative_Iz_per_triangle, neg_Iz_per_triangle);

			SetConsoleTextAttribute(hConsole, 13);
			printf("\nGPU: Iz0 = %1.14E Izpresc %1.14E SigmaIzz %1.14E EzStrength = %1.14E \n\n", Iz0,
				Iz_prescribed_endtime, sigma_Izz, EzStrength_);
			SetConsoleTextAttribute(hConsole, 14);

			kernelCalculateVelocityAndAzdot << <numTilesMinor, threadsPerTileMinor >> >(
				SUBSTEP,

				pX_half->p_info,
				p_vn0,
				p_v0,
				p_OhmsCoeffs,
				pX_half->p_AAdot,
				pX_half->p_n_minor,
				pX_half->p_AreaMinor,

				pX_target->p_AAdot,
				pX_target->p_vie,
				pX_target->p_v_n
				);
			Call(cudaThreadSynchronize(), "cudaTS kernelUpdate_v ");

			if (iSubstep == 0) {
				pTriMesh->Accelerate2018(SUBSTEP,// ROCAzdotduetoAdvection, 
					pTriMeshhalf, pDestMesh, evaltime + 0.5*SUBSTEP, false,
					false//(iSubstep == SUBCYCLES - 1)
				);
			} else {
				// Thereafter just update pDestMesh since we can discard the old values of v, Adot.
				pDestMesh->Accelerate2018(SUBSTEP, //ROCAzdotduetoAdvection, 
					pTriMeshhalf, pDestMesh, evaltime + 0.5*SUBSTEP, false,
					false //(iSubstep == SUBCYCLES - 1)
				);
			};

			SetConsoleTextAttribute(hConsole, 15);
			// ===================
			printf("Compare v_e and Azdot:\n");
			printf("Compare v_e: \n");
			cudaMemcpy(p_cuSyst_host->p_vie, pX_target->p_vie, sizeof(v4)*NMINOR, cudaMemcpyDeviceToHost);
			f64_vec3 * ve_host = (f64_vec3 *)malloc(sizeof(f64_vec3)*NMINOR);
			f64_vec3 * ve_compare = (f64_vec3 *)malloc(sizeof(f64_vec3)*NMINOR);
			for (long i = 0; i < NMINOR; i++)
			{
				ve_compare[i].x = pDestMesh->pData[i].vxy.x;
				ve_compare[i].y = pDestMesh->pData[i].vxy.y;
				ve_compare[i].z = pDestMesh->pData[i].vez;
				ve_host[i].x = p_cuSyst_host->p_vie[i].vxy.x;
				ve_host[i].y = p_cuSyst_host->p_vie[i].vxy.y;
				ve_host[i].z = p_cuSyst_host->p_vie[i].vez;
			}
			Compare_f64_vec3(ve_host, ve_compare, NMINOR);

			printf("Compare v_n (pX_target, pDestMesh): \n");
			cudaMemcpy(p_cuSyst_host->p_v_n, pX_target->p_v_n, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToHost);
			for (long i = 0; i < NMINOR; i++)
			{
				vn_compare[i] = pDestMesh->pData[i].v_n;
			}
			Compare_f64_vec3(p_cuSyst_host->p_v_n, vn_compare, NMINOR);

			free(ve_host);
			free(ve_compare);

			// Compare Azdot:
			cudaMemcpy(p_cuSyst_host->p_AAdot, pX_target->p_AAdot, sizeof(AAdot)*NMINOR, cudaMemcpyDeviceToHost);
			for (long i = 0; i < NMINOR; i++)
			{
				p_temphost1[i] = p_cuSyst_host->p_AAdot[i].Azdot;
				p_temphost2[i] = pDestMesh->pData[i].Azdot;
			}
			printf("Azdot:\n");
			Compare_f64(p_temphost1, p_temphost2, NMINOR);

			printf("\nCHOSEN %d : Azdot %1.14E %1.14E diff %1.14E \n\n", CHOSEN, p_temphost1[CHOSEN], p_temphost2[CHOSEN], p_temphost1[CHOSEN] - p_temphost2[CHOSEN]);

			// ===================

			kernelUpdateAz << <numTilesMinor, threadsPerTileMinor >> >(
				(iSubstep == SUBCYCLES - 1) ? 0.5*SUBSTEP : SUBSTEP,
				pX_target->p_AAdot,
				p_ROCAzduetoAdvection,
				p_Az);
			Call(cudaThreadSynchronize(), "cudaTS kernelUpdateAz ");

			kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> >(
				this->p_info,
				this->p_tri_neigh_index,
				p_Az
				);
			Call(cudaThreadSynchronize(), "cudaTS kernelResetFrillsAz 10");

			pDestMesh->AdvanceAz((iSubstep == SUBCYCLES - 1) ? 0.5*SUBSTEP : SUBSTEP, ROCAzduetoAdvection, Az_array);
			
			// Compare Az:
			// ============
			printf("Az:\n");
			cudaMemcpy(p_temphost1, p_Az, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			Compare_f64(p_temphost1, Az_array, NMINOR);
			printf("\nCHOSEN %d : Az %1.14E %1.14E diff %1.14E \n\n", CHOSEN, p_temphost1[CHOSEN], Az_array[CHOSEN], p_temphost1[CHOSEN] - Az_array[CHOSEN]);

			// ============

			evaltime += 0.5*SUBSTEP;
		};

		kernelPushAzInto_dest << <numTilesMinor, threadsPerTileMinor >> >(
			pX_target->p_AAdot,
			p_Az);
		Call(cudaThreadSynchronize(), "cudaTS PushAzIntoDest");
		for (iMinor = 0; iMinor < NMINOR; iMinor++)
		{
			pDestMesh->pData[iMinor].Az = Az_array[iMinor];
		};

		printf("Azdot:\n");
		cudaMemcpy(p_cuSyst_compare->p_AAdot, pX_target->p_AAdot, sizeof(AAdot)*NMINOR, cudaMemcpyDeviceToHost);
		for (iMinor = 0; iMinor < NMINOR; iMinor++)
		{
			p_temphost1[iMinor] = p_cuSyst_compare->p_AAdot[iMinor].Azdot;
			p_temphost2[iMinor] = pDestMesh->pData[iMinor].Azdot;
		}
		Compare_f64(p_temphost1, p_temphost2, NMINOR);

		printf("\nCHOSEN %d : Azdot %1.14E %1.14E diff %1.14E \n\n", CHOSEN, p_temphost1[CHOSEN], p_temphost2[CHOSEN], p_temphost1[CHOSEN] - p_temphost2[CHOSEN]);
		printf("Az:\n");
		for (iMinor = 0; iMinor < NMINOR; iMinor++)
		{
			p_temphost1[iMinor] = p_cuSyst_compare->p_AAdot[iMinor].Az;
			p_temphost2[iMinor] = pDestMesh->pData[iMinor].Az;
		}
		Compare_f64(p_temphost1, p_temphost2, NMINOR);
		printf("\nCHOSEN %d : Az %1.14E %1.14E diff %1.14E \n\n", CHOSEN, p_temphost1[CHOSEN], p_temphost2[CHOSEN], p_temphost1[CHOSEN] - p_temphost2[CHOSEN]);


	}; // whether Backward or Leapfrog

	printf("evaltime %1.14E \n", evaltime);
	printf("------------------------\n");

	//this->AntiAdvectAzAndAdvance(h, pX_half, IntegratedGradAz, pDestMesh); // Might as well subsume this in Accelerate, really
	//pX_half->AntiAdvectAzAndAdvance(h*0.5, pDestMesh, GradAz, pDestMesh);

	// Compare Az:
	// ============
	cudaMemcpy(p_cuSyst_host->p_AAdot, pX_target->p_AAdot, sizeof(AAdot)*NMINOR, cudaMemcpyDeviceToHost);
	for (long i = 0; i < NMINOR; i++)
	{
		p_temphost1[i] = p_cuSyst_host->p_AAdot[i].Az;
		p_temphost2[i] = pDestMesh->pData[i].Az;
	}
	printf("Az (pX_target, pDestMesh) :\n");
	Compare_f64(p_temphost1, p_temphost2, NMINOR);
	
	pDestMesh->Wrap();
	
	kernelWrapVertices << <numTilesMajor, threadsPerTileMajor >> >(
		pX_target->p_info,
		pX_target->p_vie,
		pX_target->p_v_n,
		p_was_vertex_rotated); // B will be recalculated.
	Call(cudaThreadSynchronize(), "cudaTS kernelWrapvertices ");

	// Here put a test of whether any did have to wrap around.

	cudaMemset(p_triPBClistaffected, 0, sizeof(char)*NUMVERTICES);
	kernelWrapTriangles << <numTriTiles, threadsPerTileMinor >> >(
		pX_target->p_info,
		pX_target->p_tri_corner_index,
		p_was_vertex_rotated,
		pX_target->p_vie,
		pX_target->p_v_n,
		p_triPBClistaffected,
		pX_target->p_tri_periodic_corner_flags
		); // B will be recalculated.							   
	Call(cudaThreadSynchronize(), "cudaTS kernelWrapTriangles ");
	 
	kernelReassessTriNeighbourPeriodicFlags_and_populate_PBCIndexneighminor << <numTriTiles, threadsPerTileMinor >> > (
		pX_target->p_info,
		pX_target->p_tri_neigh_index,
		pX_target->p_tri_corner_index,
		p_was_vertex_rotated,
		pX_target->p_tri_periodic_corner_flags,
		pX_target->p_tri_periodic_neigh_flags,
		pX_target->p_szPBC_triminor,
		p_triPBClistaffected
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelReassessTriNeighbourPeriodicFlags_and_populate_PBCIndexneighminor ");

	kernelReset_szPBCtri_vert << <numTilesMajor, threadsPerTileMajor >> >(
		pX_target->p_info,
		pX_target->p_izTri_vert,
		pX_target->p_izNeigh_vert,
		pX_target->p_szPBCtri_vert,
		pX_target->p_szPBCneigh_vert,
		p_triPBClistaffected);

	Call(cudaThreadSynchronize(), "cudaTS Reset for vert. ");

	printf("Done step %d from time %1.8E length %1.2E\n\n", runs, evaltime, Timestep);

	// For graphing Lap Az:
	cudaMemcpy(temp_array_host, p_LapAz, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Elapsed time : %f ms\n\n\n\n", elapsedTime);

	runs++;

	free(shardtemp);
	

}*/
  
void DebugNaN(cuSyst * p_cuSyst)
{
	p_cuSyst->SendToHost(cuSyst_host);
	bool bSwitch = 0;
	long iMinor;
	for (iMinor = 0; iMinor < NMINOR; iMinor++)
	{
		if (cuSyst_host.p_AAdot[iMinor].Azdot != cuSyst_host.p_AAdot[iMinor].Azdot)
		{
			printf("Nan %d Azdot", iMinor);
			bSwitch = 1;
		}
		if (cuSyst_host.p_AAdot[iMinor].Az != cuSyst_host.p_AAdot[iMinor].Az)
		{
			printf("Nan %d Az", iMinor);
			bSwitch = 1;
		}
		if (cuSyst_host.p_n_minor[iMinor].n != cuSyst_host.p_n_minor[iMinor].n)
		{
			printf("Nan %d n", iMinor);
			bSwitch = 1;
		}
		if (cuSyst_host.p_T_minor[iMinor].Te != cuSyst_host.p_T_minor[iMinor].Te)
		{
			printf("Nan %d Te", iMinor);
			bSwitch = 1;
		}
		if (cuSyst_host.p_vie[iMinor].vez != cuSyst_host.p_vie[iMinor].vez)
		{
			printf("Nan %d vez", iMinor);
			bSwitch = 1;
		}
		if (cuSyst_host.p_vie[iMinor].viz != cuSyst_host.p_vie[iMinor].viz)
		{
			printf("Nan %d viz", iMinor);
			bSwitch = 1;
		}
		if ((cuSyst_host.p_T_minor[iMinor].Te < 0.0)
			|| (cuSyst_host.p_T_minor[iMinor].Te > 5.0e-10)) {
			printf("Te = %1.6E %d | ", cuSyst_host.p_T_minor[iMinor].Te, iMinor);
			bSwitch = 1;
		}
		if ((cuSyst_host.p_T_minor[iMinor].Ti < 0.0)
			|| (cuSyst_host.p_T_minor[iMinor].Ti > 1.0e-9)) {
			printf("Ti = %1.6E %d | ", cuSyst_host.p_T_minor[iMinor].Ti, iMinor);
			bSwitch = 1;
		}
	};
	if (bSwitch) {
		printf("end\n");  while (1) getch();
	};
}

void cuSyst::PerformCUDA_Advance(//const 
	cuSyst * pX_target,
	//const 
	cuSyst * pX_half) // just being annoying when we put const - but put it back
{
	long iSubcycles;
	f64 hsub, Timestep;
	FILE * fp;
	long iMinor, iSubstep;
	f64 Iz_prescribed;
	static long runs = 0;
	float elapsedTime;
	f64 Iz_k, Iz_prescribed_endtime;
	f64_vec2 temp_vec2;
	// DEBUG:
	f64 sum_plus, sum_minus, sum_Lap, abs_Lap, sum_Azdot, abs_Azdot;
	FILE * fp_2;

	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

	cudaEvent_t start, stop, middle;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&middle);
	cudaEventRecord(start, 0);
	cudaEventSynchronize(start);

#define USE_N_MAJOR_FOR_VERTEX 

	// DEBUG:
	printf("\nDebugNaN 1\n\n");
	DebugNaN(this);

	//fp_traj = fopen("traj1176.txt", "a");

	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> >(
		this->p_n_minor,
		this->p_n_major,
		this->p_T_minor,
		this->p_info,
		this->p_cc,  // Calculate circumcenters; we would like n and T there for shards.

		this->p_tri_corner_index,
		this->p_tri_periodic_corner_flags,

		false // calculate n and T on circumcenters instead of centroids
		); // call before CreateShardModel 
	Call(cudaThreadSynchronize(), "cudaTS average nTx");
	cudaMemcpy(this->p_n_minor + BEGINNING_OF_CENTRAL,
		this->p_n_major, sizeof(nvals)*NUMVERTICES, cudaMemcpyDeviceToDevice);

	// To match how we do it below we should really be adding in iterations of ShardModel and InferMinorDensity.

	kernelCreateShardModelOfDensities_And_SetMajorArea << <numTilesMajor, threadsPerTileMajor >> >(
		this->p_info,
		this->p_n_major,
		this->p_n_minor,  // DESIRED VALUES
		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_cc,
		p_n_shards,
		p_n_shards_n,
		this->p_AreaMajor,
		false // USE CENTROIDS
		);
	Call(cudaThreadSynchronize(), "cudaTS CreateShardModels this");

	kernelCalculate_kappa_nu << <numTriTiles, threadsPerTileMinor >> >(
		this->p_info,
		this->p_n_minor,
		this->p_T_minor,

		p_kappa_n,
		p_kappa_i,
		p_kappa_e,
		p_nu_i,
		p_nu_e		// define all these.
		);
	Call(cudaThreadSynchronize(), "cudaTS Calculate_kappa_nu(this)");

	kernelCalc_SelfCoefficient_for_HeatConduction << <numTilesMajorClever, threadsPerTileMajorClever >> >(
		1.0, // h == 1
		this->p_info,
		this->p_izNeigh_vert,
		this->p_szPBCneigh_vert,
		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_cc,
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL,
		this->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
		p_kappa_n,
		p_kappa_i,
		p_kappa_e,
		p_nu_i,
		p_nu_e,
		this->p_AreaMajor, // needs to be set !! We could dispense with it as an input!!
		p_coeffself_n, 
		p_coeffself_i,
		p_coeffself_e,
		0.0
		);
	Call(cudaThreadSynchronize(), "cudaTS Calc_SelfCoefficient_for_HeatConduction");
	// To increase the efficiency we want to make a clever 2nd set of major tiles of size 192. Also try 256, 384.

	kernelTileMaxMajor << <numTilesMajorClever, threadsPerTileMajorClever >> > (
		p_coeffself_n,
		p_temp2
		);
	Call(cudaThreadSynchronize(), "cudaTS TileMaxMajor");
	cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);

	// Great problems here -- we took it now as 1 - h/N d/dt NT

	f64 maxcoeff = 0.0;
	for (long iTile = 0; iTile < numTilesMajorClever; iTile++)
	{
		// We want h < 1/maxcoeffself
		f64 h = 1.0 / p_temphost2[iTile];
		printf("iTile %d max_coeffself %1.9E 1over %1.9E \n", iTile, p_temphost2[iTile], h);
		maxcoeff = max(maxcoeff, p_temphost2[iTile]);
	}
	printf("===============\nNEUTRAL h_use = %1.10E maxcoeff %1.10E\n================\n", 0.25 / maxcoeff, maxcoeff);
	f64 store_maxcoeff = maxcoeff;


	kernelTileMaxMajor << <numTilesMajorClever, threadsPerTileMajorClever >> > (
		p_coeffself_i,
		p_temp2
		);
	Call(cudaThreadSynchronize(), "cudaTS TileMaxMajor");
	cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
	maxcoeff = 0.0;
	for (long iTile = 0; iTile < numTilesMajorClever; iTile++)
	{
		// We want h < 1/maxcoeffself
		f64 h = 1.0 / p_temphost2[iTile];
		printf("iTile %d max_coeffself %1.9E 1over %1.9E \n", iTile, p_temphost2[iTile], h);
		maxcoeff = max(maxcoeff, p_temphost2[iTile]);
	}
	printf("===============\nION h_use = %1.10E maxcoeff %1.10E\n================\n", 0.25 / maxcoeff, maxcoeff);
	f64 store_maxcoeff_i = maxcoeff;

	kernelTileMaxMajor << <numTilesMajorClever, threadsPerTileMajorClever >> > (
		p_coeffself_e,
		p_temp2
		);
	Call(cudaThreadSynchronize(), "cudaTS TileMaxMajor");
	cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
	maxcoeff = 0.0;
	for (long iTile = 0; iTile < numTilesMajorClever; iTile++)
	{
		// We want h < 1/maxcoeffself
		f64 h = 1.0 / p_temphost2[iTile];
		printf("iTile %d max_coeffself %1.9E 1over %1.9E \n", iTile, p_temphost2[iTile], h);
		maxcoeff = max(maxcoeff, p_temphost2[iTile]);
	}
	printf("===============\nELECTRON h_use = %1.10E maxcoeff %1.10E\n================\n", 0.25 / maxcoeff, maxcoeff);
	maxcoeff = max(maxcoeff, max(store_maxcoeff, store_maxcoeff_i));
	 
	// See how far we get with this as timestep.
	
	Timestep = min(0.25 / maxcoeff, TIMESTEP); 
	Timestep = max(Timestep, TIMESTEP*0.5); // min h/2

	// Alternative:
	//#ifndef USE_N_MAJOR_FOR_VERTEX	 
	kernelInferMinorDensitiesFromShardModel << <numTilesMinor, threadsPerTileMinor >> >(
		this->p_info,
		this->p_n_minor,
		p_n_shards,
		p_n_shards_n,
		this->p_tri_corner_index,
		this->p_who_am_I_to_corner,
		p_one_over_n);
	Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities");
	// DO SWITCH INSIDE ROUTINE
	
	kernelCalculateOverallVelocitiesVertices << <numTilesMajor, threadsPerTileMajor >> > (
		this->p_info,
		this->p_vie + BEGINNING_OF_CENTRAL,
		this->p_v_n + BEGINNING_OF_CENTRAL,
		this->p_n_major,
		this->p_v_overall_minor + BEGINNING_OF_CENTRAL,
		p_n_shards,
		p_n_shards_n,
		this->p_izTri_vert,
		this->p_szPBCtri_vert
		);
	Call(cudaThreadSynchronize(), "cudaTS calculate overall velocities");
	   
	// Includes drift towards barycenter.
	//
	//kernelAdvectPositionsVertex << <numTilesMajor, threadsPerTileMajor >> >(
	//	0.5*Timestep,
	//	this->p_info + BEGINNING_OF_CENTRAL,
	//	pX_half->p_info + BEGINNING_OF_CENTRAL,
	//	this->p_v_overall_minor + BEGINNING_OF_CENTRAL,
	//	this->p_n_major,
	//	this->p_izNeigh_vert,
	//	this->p_szPBCneigh_vert
	//	);
	//Call(cudaThreadSynchronize(), "cudaTS AdvectPositions_Vertex");
	//
	// Infer tri velocities based on actual moves of verts:
	kernelAverageOverallVelocitiesTriangles << <numTriTiles, threadsPerTileMinor >> > (
		this->p_v_overall_minor + BEGINNING_OF_CENTRAL,
		this->p_v_overall_minor,
		this->p_info,
		this->p_tri_corner_index,
		this->p_tri_periodic_corner_flags
		);
	Call(cudaThreadSynchronize(), "cudaTS avg overall v to triangles");

	// Move tri positions:
	kernelAdvectPositionsTris << <numTilesMinor, threadsPerTileMinor >> >(
		0.5*Timestep,
		this->p_info,
		pX_half->p_info,
		this->p_v_overall_minor
		);
	Call(cudaThreadSynchronize(), "cudaTS AdvectPositions_CopyTris");


	SetConsoleTextAttribute(hConsole, 11);
//	cudaMemcpy(&temp_vec2, &(this->p_info[42940].pos), sizeof(f64_vec2), cudaMemcpyDeviceToHost);
//	printf("\nposition %1.14E %1.14E\n\n", temp_vec2.x, temp_vec2.y);
//	cudaMemcpy(&temp_vec2, &(this->p_info[95115].pos), sizeof(f64_vec2), cudaMemcpyDeviceToHost);
//	printf("\nposition 23187 %1.14E %1.14E\n\n", temp_vec2.x, temp_vec2.y);
//	cudaMemcpy(&temp_vec2, &(this->p_info[21554 + BEGINNING_OF_CENTRAL].pos), sizeof(f64_vec2), cudaMemcpyDeviceToHost);
//	printf("\nPOSITION 21554 %1.12E %1.12E\n\n", temp_vec2.x, temp_vec2.y);

	SetConsoleTextAttribute(hConsole, 15);

	kernelCalculateUpwindDensity_tris << <numTriTiles, threadsPerTileMinor >> >(
		this->p_info,
		p_n_shards_n,
		p_n_shards,
		this->p_vie,
		this->p_v_n,
		this->p_v_overall_minor,
		this->p_tri_corner_index,
		this->p_tri_neigh_index,
		this->p_who_am_I_to_corner,
		this->p_tri_periodic_neigh_flags,
		this->p_n_upwind_minor,
		this->p_T_minor,
		p_T_upwind_minor);
	Call(cudaThreadSynchronize(), "cudaTS CalculateUpwindDensity_tris");

	cudaMemset(p_Integrated_div_v_overall, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Div_v_neut, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Div_v, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(NT_addition_rates_d, 0, sizeof(NTrates)*NUMVERTICES);
	 
	kernelAccumulateAdvectiveMassHeatRate << <numTilesMajor, threadsPerTileMajor >> >(
		0.5*Timestep, // why it appears here?
		this->p_info,
		this->p_izTri_vert,
		this->p_szPBCtri_vert,

		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL,

		this->p_n_upwind_minor,
		this->p_vie,
		this->p_v_n,
		this->p_v_overall_minor,
		p_T_upwind_minor, // may or may not overlap source: don't we only use from tris? so not overlap

		NT_addition_rates_d,
		p_Div_v,
		p_Div_v_neut,
		p_Integrated_div_v_overall);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateAdvectiveMassHeatRate");

	SetConsoleTextAttribute(hConsole, 14);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NeTe %1.10E\n\n", VERTCHOSEN, tempf64);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NiTi %1.10E\n\n", VERTCHOSEN, tempf64);
	SetConsoleTextAttribute(hConsole, 15);

	//this->SendToHost(cuSyst_host);
	//printf("14790: n %1.10E nn %1.10E \nTn %1.10E Ti %1.10E Te %1.10E \n"
	//	"vxy %1.10E %1.10E vn %1.10E %1.10E \n"
	//	"vez %1.10E B %1.10E %1.10E \n",
	//	cuSyst_host.p_n_major[14790].n, cuSyst_host.p_n_major[14790].n_n,
	//	cuSyst_host.p_T_minor[14790+BEGINNING_OF_CENTRAL].Tn,
	//	cuSyst_host.p_T_minor[14790 + BEGINNING_OF_CENTRAL].Ti,
	//	cuSyst_host.p_T_minor[14790 + BEGINNING_OF_CENTRAL].Te,
	//	cuSyst_host.p_vie[14790 + BEGINNING_OF_CENTRAL].vxy.x, cuSyst_host.p_vie[14790 + BEGINNING_OF_CENTRAL].vxy.y,
	//	cuSyst_host.p_v_n[14790 + BEGINNING_OF_CENTRAL].x, cuSyst_host.p_v_n[14790 + BEGINNING_OF_CENTRAL].y,
	//	cuSyst_host.p_vie[14790 + BEGINNING_OF_CENTRAL].vez,
	//	cuSyst_host.p_B[14790 + BEGINNING_OF_CENTRAL].x, cuSyst_host.p_B[14790 + BEGINNING_OF_CENTRAL].y);
	//printf("14638: n %1.10E nn %1.10E \nTn %1.10E Ti %1.10E Te %1.10E \n"
	//	"vxy %1.10E %1.10E vn %1.10E %1.10E \n"
	//	"vez %1.10E B %1.10E %1.10E \n", cuSyst_host.p_n_major[14638].n, cuSyst_host.p_n_major[14638].n_n,
	//	cuSyst_host.p_T_minor[14638 + BEGINNING_OF_CENTRAL].Tn,
	//	cuSyst_host.p_T_minor[14638 + BEGINNING_OF_CENTRAL].Ti,
	//	cuSyst_host.p_T_minor[14638 + BEGINNING_OF_CENTRAL].Te,
	//	cuSyst_host.p_vie[14638 + BEGINNING_OF_CENTRAL].vxy.x, cuSyst_host.p_vie[14638 + BEGINNING_OF_CENTRAL].vxy.y,
	//	cuSyst_host.p_v_n[14638 + BEGINNING_OF_CENTRAL].x, cuSyst_host.p_v_n[14638 + BEGINNING_OF_CENTRAL].y,
	//	cuSyst_host.p_vie[14638 + BEGINNING_OF_CENTRAL].vez,
	//	cuSyst_host.p_B[14638 + BEGINNING_OF_CENTRAL].x, cuSyst_host.p_B[14638 + BEGINNING_OF_CENTRAL].y);
	
	// For now went for "advective change in v" where we divide d/dt Nv by N_derivsyst
	cudaMemset(p_MAR_neut, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_ion, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_elec, 0, sizeof(f64_vec3)*NMINOR);
	 
	kernelCreate_pressure_gradT_and_gradA_LapA_CurlA_minor << <numTriTiles, threadsPerTileMinor >> >(

		this->p_info,
		this->p_T_minor,
		this->p_AAdot,

		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_izNeigh_TriMinor,
		this->p_szPBC_triminor,
		this->p_who_am_I_to_corner,
		this->p_tri_corner_index,

		p_MAR_neut, p_MAR_ion, p_MAR_elec,

		p_n_shards,				// this kernel is for i+e only
		this->p_n_minor,

		p_GradTe,
		p_GradAz,
		p_LapAz,

		// Unused by anything else:
		p_ROCAzduetoAdvection, // Would probs be better to split out Az calc, remember
		p_ROCAzdotduetoAdvection, // Would probs be better to split out Az calc, remember
		this->p_v_overall_minor, // it's only this time that we need to collect it ofc.

								 // ######################################
								 // should put in a switch to not collect. But DO ZERO ROCAzdotduetoAdvection in that case.

		this->p_B, // HERE THIS IS GETTING PoP'D SO WHY WE HAVE A PROBLEM COLLECTING AND DISPLAYING?
		this->p_AreaMinor
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelCreate_pressure_gradT_and_gradA_LapA_CurlA_minor");

	SetConsoleTextAttribute(hConsole, 14);
	//cudaMemcpy(&tempf64, &(p_MAR_elec[42940].z), sizeof(f64), cudaMemcpyDeviceToHost);
	//printf("\np_MAR_elec[42940].z %1.14E\n\n", tempf64);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NiTi %1.10E\n\n", VERTCHOSEN, tempf64);
	SetConsoleTextAttribute(hConsole, 15);
	
	/*
	kernelPullAzFromSyst << <numTilesMinor, threadsPerTileMinor >> >(
		this->p_AAdot,
		p_Az
		);
	Call(cudaThreadSynchronize(), "cudaTS PullAz");

	// Now call this and make sure it's same:
	kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> >(
		this->p_info,
		p_Az,
		this->p_izTri_vert,
		this->p_izNeigh_TriMinor,
		this->p_szPBCtri_vert,
		this->p_szPBC_triminor,
		p_temp4,
		p_temp1, p_temp2, p_temp3,
		this->p_AreaMinor
		);
	Call(cudaThreadSynchronize(), "cudaTS GetLapMinor aa1a");
	// Should populate an array of coefficients s.t. Azdot_k+1 = ~Azdot0~ + ~gamma~ Lap Az
	// Now we will wanna create each eqn for Az with coeffs on neighbour values.
	// So we need a func called "GetLapCoefficients".
	
	cudaMemcpy(p_temphost1, p_LapAz, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_temphost4, p_temp4, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
	// Test difference of the twain:
	f64 sumsq = 0.0, L2diff = 0.0;
	f64 maxdiff = 0.0;
//	fp = fopen("lapvs.txt", "w");
	for (long iMinor = 0; iMinor < NMINOR; iMinor++) {
		sumsq += p_temphost4[iMinor] * p_temphost4[iMinor];
		if (fabs(p_temphost4[iMinor] - p_temphost1[iMinor]) > maxdiff)
			maxdiff = fabs(p_temphost4[iMinor] - p_temphost1[iMinor]);

//		fprintf(fp, "Lap1 %1.15E Lap2 %1.15E \n", p_temphost4[iMinor], p_temphost1[iMinor]);
	}
	sumsq /= (f64)NMINOR;
//	fclose(fp);
	printf("GLM vs Pressure routine: L2lap %1.15E  Linf_diff %1.12E \n\n", sqrt(sumsq), maxdiff);
	 
	cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	
	f64 sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
	for (long iTile = 0; iTile < numTriTiles; iTile++)
	{
		sum1 += p_temphost1[iTile];
		sum2 += p_temphost2[iTile];
		sum3 += p_temphost3[iTile];
		// printf("  %1.10E  |", p_temphost1[iTile]);
		// if (iTile % 4 == 0) printf("\n");
	}
	printf("sum1 : %1.15E sum2 : %1.15E vertsum %1.15E \n",
		sum1, sum2, sum3);
	
	for (long iMinor = 0; iMinor < NMINOR; iMinor++)
		sumsq += p_temphost4[iMinor] * p_temphost4[iMinor];
	sumsq /= (f64)NMINOR;
	 
	printf("sqrt(avg sq) %1.15E  sumtot %1.15E \n\n", sqrt(sumsq), (sum1 + sum2 + sum3));
	getch();
	*/

	////////////////////////////////

	kernelCreate_momflux_minor << <numTriTiles, threadsPerTileMinor >> >(
		this->p_info,
		this->p_vie,
		this->p_v_overall_minor,

		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_izNeigh_TriMinor,
		this->p_szPBC_triminor,

		this->p_who_am_I_to_corner,
		this->p_tri_corner_index,
		p_MAR_neut, p_MAR_ion, p_MAR_elec,
		p_n_shards
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelCreate_momflux_minor");

	SetConsoleTextAttribute(hConsole, 14);
	//cudaMemcpy(&tempf64, &(this->p_vie[42940].vez), sizeof(f64), cudaMemcpyDeviceToHost);
	//printf("\np_MAR_elec[42940].z %1.14E\n\n", tempf64);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NiTi %1.10E\n\n", VERTCHOSEN, tempf64);
	SetConsoleTextAttribute(hConsole, 15);

	kernelNeutral_pressure_and_momflux << <numTriTiles, threadsPerTileMinor >> >(
		this->p_info,

		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_izNeigh_TriMinor,
		this->p_szPBC_triminor,
		this->p_who_am_I_to_corner,
		this->p_tri_corner_index,

		this->p_T_minor,
		this->p_v_n,
		p_n_shards_n,
		this->p_n_minor,

		this->p_v_overall_minor,
		p_MAR_neut
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelNeutral_pressure_and_momflux");
	  
//	kernelCalculateNu_eHeartNu_iHeart_nu_nn_visc << <numTilesMajorClever, threadsPerTileMajorClever >> >(
//		this->p_info + BEGINNING_OF_CENTRAL,
//		this->p_n_major,
//		this->p_T_minor + BEGINNING_OF_CENTRAL,
//		p_nu_major);
//	Call(cudaThreadSynchronize(), "cudaTS CalculateNu");
	 // 12  =  red

	SetConsoleTextAttribute(hConsole, 14);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NeTe %1.12E\n\n", VERTCHOSEN, tempf64);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NiTi %1.10E\n\n", VERTCHOSEN, tempf64);
	SetConsoleTextAttribute(hConsole, 15);

	long i;
/*	fprintf(fp_traj, " | n ");
	for (i = 11765; i < 11770; i++) {
		cudaMemcpy(&tempf64, &(this->p_n_major[i].n), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}
	for (i = 11621; i <= 11623; i++) {
		cudaMemcpy(&tempf64, &(this->p_n_major[i].n), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}
	fprintf(fp_traj, " | Te ");
	for (i = 11765; i < 11770; i++) {
		cudaMemcpy(&tempf64, &(this->p_T_minor[i+BEGINNING_OF_CENTRAL].Te), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}
	for (i = 11621; i <= 11623; i++) {
		cudaMemcpy(&tempf64, &(this->p_T_minor[i + BEGINNING_OF_CENTRAL].Te), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}
	fprintf(fp_traj, " | vez ");
	for (i = 11765; i < 11770; i++) {
		cudaMemcpy(&tempf64, &(this->p_vie[i + BEGINNING_OF_CENTRAL].vez), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}
	for (i = 11621; i <= 11623; i++) {
		cudaMemcpy(&tempf64, &(this->p_vie[i + BEGINNING_OF_CENTRAL].vez), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}
	fprintf(fp_traj, " | nu_eHeart ");
	for (i = 11765; i < 11770; i++) {
		cudaMemcpy(&tempf64, &(p_nu_major[i].e), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}
	for (i = 11621; i <= 11623; i++) {
		cudaMemcpy(&tempf64, &(p_nu_major[i].e), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E",tempf64);
	}*/
	SetConsoleTextAttribute(hConsole, 11);

	// . Create best estimate of n on cc (and avg T to cc:)

	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> >(
		this->p_n_minor,
		this->p_n_major,
		this->p_T_minor,
		this->p_info,
		this->p_cc,  // Calculate circumcenters; we would like n and T there for shards.

		this->p_tri_corner_index,
		this->p_tri_periodic_corner_flags,

		true // calculate n and T on circumcenters instead of centroids
		); // call before CreateShardModel 
	Call(cudaThreadSynchronize(), "cudaTS Average_nTx cc");

	for (long iIterate = 0; iIterate < 3; iIterate++)
	{
		kernelCreateShardModelOfDensities_And_SetMajorArea << <numTilesMajor, threadsPerTileMajor >> >(
			this->p_info,
			this->p_n_major,
			this->p_n_minor,  // DESIRED VALUES
			this->p_izTri_vert,
			this->p_szPBCtri_vert,
			this->p_cc,
			p_n_shards,
			p_n_shards_n,
			this->p_AreaMajor,
			true // Use circumcenter
			);
		Call(cudaThreadSynchronize(), "cudaTS CreateShardModels ..");

		kernelInferMinorDensitiesFromShardModel << <numTriTiles, threadsPerTileMinor >> >(
			this->p_info,
			this->p_n_minor,
			p_n_shards,
			p_n_shards_n,
			this->p_tri_corner_index,
			this->p_who_am_I_to_corner,
			p_one_over_n); // overwrites but it doesn't matter
		Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities ..");
	};
	
	
	/*
	kernelCalculate_kappa_nu<<<numTriTiles, threadsPerTileMinor>>>(
		this->p_info,
		this->p_n_minor,
		this->p_T_minor,

		p_kappa_n,
		p_kappa_i,
		p_kappa_e,
		p_nu_i,
		p_nu_e		// define all these.
	);
	Call(cudaThreadSynchronize(), "cudaTS Calculate_kappa_nu(this)");
	
	kernelAccumulateDiffusiveHeatRate_new << <numTilesMajorClever, threadsPerTileMajorClever >> >(
		0.5*Timestep,
		this->p_info,
		this->p_izNeigh_vert,
		this->p_szPBCneigh_vert,
		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_cc,
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices
		this->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
		p_kappa_n,
		p_kappa_i,
		p_kappa_e,
		p_nu_i,
		p_nu_e,
		NT_addition_rates_d,
		this->p_AreaMajor
//		p_temp1, // spit out effect on dTe/dt of conduction
//		p_temp4 // spit out effect on dTe/dt of ionisation   // vertex
		);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate");
	// To increase the efficiency we want to make a clever 2nd set of major tiles of size 192. Also try 256, 384.
	*/

	RunBackwardJLSForHeat(
		this->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices,
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices
		0.5*Timestep,
		this);

	kernelAccumulateDiffusiveHeatRate_new << <numTilesMajorClever, threadsPerTileMajorClever >> >(
		0.5*Timestep,
		this->p_info,
		this->p_izNeigh_vert,
		this->p_szPBCneigh_vert,
		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_cc,
		this->p_n_major,
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices
		this->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
		p_kappa_n,
		p_kappa_i,
		p_kappa_e,
		p_nu_i,
		p_nu_e,
		NT_addition_rates_d,
		this->p_AreaMajor
		);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate");

	SetConsoleTextAttribute(hConsole, 14);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NeTe %1.14E\n\n", VERTCHOSEN, tempf64);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NiTi %1.10E\n\n", VERTCHOSEN, tempf64);
	SetConsoleTextAttribute(hConsole, 15);

	kernelIonisationRates<<<numTilesMajor,threadsPerTileMajor>>>(
		0.5*Timestep,
		this->p_info,
		this->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices
		this->p_n_major,
		this->p_AreaMajor,
		NT_addition_rates_d
		);
	Call(cudaThreadSynchronize(), "cudaTS Ionisation");

	SetConsoleTextAttribute(hConsole, 14);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NeTe %1.14E\n\n", VERTCHOSEN, tempf64);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NiTi %1.10E\n\n", VERTCHOSEN, tempf64);
	SetConsoleTextAttribute(hConsole, 15);

	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


	kernelReset_v_in_outer_frill_and_outermost << <numTilesMinor, threadsPerTileMinor >> >
		(
			this->p_info,
			this->p_vie,
			this->p_v_n,
			this->p_tri_neigh_index,
			this->p_izNeigh_vert
			);
	Call(cudaThreadSynchronize(), "cudaTS resetv");

	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> >(
		this->p_n_minor,
		this->p_n_major,
		this->p_T_minor,
		this->p_info,
		this->p_cc,  // Calculate circumcenters; we would like n and T there for shards.

		this->p_tri_corner_index,
		this->p_tri_periodic_corner_flags,

		false // 
		); // call before CreateShardModel 
	Call(cudaThreadSynchronize(), "cudaTS Average_nTx pos this");


	kernelCalculate_ita_visc<<<numTilesMinor, threadsPerTileMinor>>>(
		this->p_info,
		this->p_n_minor,
		this->p_T_minor,
		 
		p_temp3,
		p_temp4,
		p_temp5,
		p_temp1,
		p_temp2,
		p_temp6);
	Call(cudaThreadSynchronize(), "cudaTS ita 1");
	/*
	fprintf(fp_traj, " | ita ");
	for (i = 11765; i < 11770; i++) {
		cudaMemcpy(&tempf64, &(p_temp2[i + BEGINNING_OF_CENTRAL]), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}
	for (i = 11621; i <= 11623; i++) {
		cudaMemcpy(&tempf64, &(p_temp2[i + BEGINNING_OF_CENTRAL]), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}
	fprintf(fp_traj, " | nu_ei ");
	for (i = 11765; i < 11770; i++) {
		cudaMemcpy(&tempf64, &(p_temp4[i + BEGINNING_OF_CENTRAL]), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}
	for (i = 11621; i <= 11623; i++) {
		cudaMemcpy(&tempf64, &(p_temp4[i + BEGINNING_OF_CENTRAL]), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}

	fprintf(fp_traj, " | Base_add_nT ");
	for (i = 11765; i < 11770; i++) {
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[i].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}
	for (i = 11621; i <= 11623; i++) {
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[i].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}*/

	SetConsoleTextAttribute(hConsole, 14);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NeTe %1.10E\n\n", VERTCHOSEN, tempf64);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NiTi %1.10E\n\n", VERTCHOSEN, tempf64);
	SetConsoleTextAttribute(hConsole, 15);

	SetConsoleTextAttribute(hConsole, 14);
	cudaMemcpy(&tempf64, &(p_MAR_elec[CHOSEN].y), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\np_MAR_elec[%d].y %1.14E\n\n", CHOSEN, tempf64);
	SetConsoleTextAttribute(hConsole, 15);

	::RunBackwardJLSForViscosity(this->p_vie, pX_half->p_vie, Timestep, this);

	cudaMemset(NT_addition_tri_d, 0, sizeof(NTrates)*NUMVERTICES * 2);
	kernelCreate_viscous_contrib_to_MAR_and_NT<<<numTriTiles,threadsPerTileMinor>>>(

		this->p_info,
		pX_half->p_vie,
		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_izNeigh_TriMinor,
		this->p_szPBC_triminor,

		p_temp1, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
		p_temp2, //f64 * __restrict__ p_ita_parallel_elec_minor,   // nT / nu ready to look up
		p_temp3, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
		p_temp4, // f64 * __restrict__ p_nu_elec_minor,   // nT / nu ready to look up

		this->p_B,

		p_MAR_ion, // just accumulates
		p_MAR_elec,
		NT_addition_rates_d,
		NT_addition_tri_d);
	Call(cudaThreadSynchronize(), "cudaTS visccontrib 1"); 

	SetConsoleTextAttribute(hConsole, 14);
	cudaMemcpy(&tempf64, &(p_MAR_elec[CHOSEN].y), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\np_MAR_elec[%d].y %1.14E\n\n", CHOSEN, tempf64);
	SetConsoleTextAttribute(hConsole, 15);

	SetConsoleTextAttribute(hConsole, 14);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NeTe %1.10E\n\n", VERTCHOSEN, tempf64);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NiTi %1.10E\n\n", VERTCHOSEN, tempf64);
	SetConsoleTextAttribute(hConsole, 15);

	Collect_Nsum_at_tris << <numTriTiles, threadsPerTileMinor >> >(
		this->p_info,
		this->p_n_minor,
		this->p_tri_corner_index,
		this->p_AreaMajor, // populated?
		p_temp4);
	Call(cudaThreadSynchronize(), "cudaTS Nsum");
	 
	kernelTransmitHeatToVerts << <numTilesMajor, threadsPerTileMajor >> > (
		this->p_info,
		this->p_izTri_vert,
		this->p_n_minor,
		this->p_AreaMajor,
		p_temp4,
		NT_addition_rates_d,
		NT_addition_tri_d
		);
	Call(cudaThreadSynchronize(), "cudaTS sum up heat 1");

	/*fprintf(fp_traj, " | Total_nT ");
	for (i = 11765; i < 11770; i++) {
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[i].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}
	for (i = 11621; i <= 11623; i++) {
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[i].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}
	fprintf(fp_traj, "\n");*/

	SetConsoleTextAttribute(hConsole, 14);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NeTe %1.10E\n\n", VERTCHOSEN, tempf64);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NiTi %1.10E\n\n", VERTCHOSEN, tempf64);
	SetConsoleTextAttribute(hConsole, 15);

	SetConsoleTextAttribute(hConsole, 15);
	 
	kernelAdvanceDensityAndTemperature << <numTilesMajor, threadsPerTileMajor >> >(
		0.5*Timestep,
		this->p_info + BEGINNING_OF_CENTRAL,
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL,
		NT_addition_rates_d,
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL,
		this->p_vie + BEGINNING_OF_CENTRAL,
		this->p_v_n + BEGINNING_OF_CENTRAL,

		p_Div_v_neut, p_Div_v,
		p_Integrated_div_v_overall,
		this->p_AreaMajor,
		pX_half->p_n_major,
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL

		//p_temp2, p_temp3 // ei, en resistive effect on T
		);
	Call(cudaThreadSynchronize(), "cudaTS Advance_n_and_T"); // vertex
	 
	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_n_minor,
		pX_half->p_n_major,
		pX_half->p_T_minor,
		pX_half->p_info,
		pX_half->p_cc,
		pX_half->p_tri_corner_index,
		pX_half->p_tri_periodic_corner_flags,
		false
		); // call before CreateShardModel 
	Call(cudaThreadSynchronize(), "cudaTS average nTx 2");
	cudaMemcpy(pX_half->p_n_minor + BEGINNING_OF_CENTRAL,
		pX_half->p_n_major, sizeof(nvals)*NUMVERTICES, cudaMemcpyDeviceToDevice);

	// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	// We are going to want to introduce 2nd degree approx to get n,T desired on tris.
	// Now let's set up the accel move to half-time which will provide us input of v to the full n,T move.
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		
	//////////////////////////////////////////////////////////////////////////
	// Even more shards!:

	kernelCreateShardModelOfDensities_And_SetMajorArea << <numTilesMajor, threadsPerTileMajor >> > (
		pX_half->p_info,
		pX_half->p_n_major,
		pX_half->p_n_minor,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert, // could be more intelligent way than storing and copying 3x
		pX_half->p_cc,
		p_n_shards,
		p_n_shards_n,
		//		p_Tri_n_lists,
		//	p_Tri_n_n_lists,
		pX_half->p_AreaMajor,
		false);
	Call(cudaThreadSynchronize(), "cudaTS CreateShardModel pX_half");

	kernelInferMinorDensitiesFromShardModel << <numTilesMinor, threadsPerTileMinor >> >(
		pX_half->p_info,
		pX_half->p_n_minor,
		p_n_shards,
		p_n_shards_n,
		pX_half->p_tri_corner_index,
		pX_half->p_who_am_I_to_corner,
		p_one_over_n2);// (At the moment just repopulating tri minor n.)// based on pXhalf !!
	Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities pX_half");

//	Iz_prescribed_starttime = GetIzPrescribed(evaltime); // because we are setting pX_half->v
	Iz_prescribed_endtime = GetIzPrescribed(evaltime + 0.5*Timestep); // because we are setting pX_half->v

	kernelEstimateCurrent << <numTilesMinor, threadsPerTileMinor >> >(
		this->p_info,
		this->p_n_minor,
		this->p_vie,
		this->p_AreaMinor,
		p_temp1); // Iz_k
	Call(cudaThreadSynchronize(), "cudaTS Estimate Iz_k");

	// Get suitable v to use for resistive heating:
	kernelPopulateOhmsLaw<<<numTilesMinor, threadsPerTileMinor>>>(
				
		0.5*Timestep,

		this->p_info,
		p_MAR_neut, p_MAR_ion, p_MAR_elec,
		this->p_B,
		p_LapAz,
		p_GradAz,
		p_GradTe,
		 
		this->p_n_minor,
		
		p_one_over_n,

		this->p_T_minor, // minor : is it populated?
		this->p_vie,
		this->p_v_n,
		this->p_AAdot,
		this->p_AreaMinor, // popd?
		p_ROCAzdotduetoAdvection,
		
		p_vn0,
		p_v0,
		p_OhmsCoeffs,
		pX_target->p_AAdot, // intermediate value

		p_Iz0_summands,
		p_sigma_Izz,
		p_denom_i,
		p_denom_e, p_coeff_of_vez_upon_viz, p_beta_ie_z,
		false,
		true, // use this for Iz:
		pX_half->p_n_minor
		//p_temp5, p_temp6 // vez effect from friction, Ez : iMinor
		); 
	Call(cudaThreadSynchronize(), "cudaTS AccelerateOhms 1");
    
	cudaMemcpy(p_Iz0_summands_host, p_Iz0_summands, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_summands_host, p_sigma_Izz, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	long iTile;
	f64 Iz0 = 0.0;
	f64 sigma_Izz = 0.0;
	Iz_k = 0.0; 
	for (iTile = 0; iTile < numTilesMinor; iTile++)
	{
		Iz0 += p_Iz0_summands_host[iTile];
		sigma_Izz += p_summands_host[iTile];
		Iz_k += p_temphost1[iTile];
		if ((Iz0 != Iz0) || (sigma_Izz != sigma_Izz)) printf("tile %d Iz0 %1.9E sigma_Izz %1.9E summands %1.9E %1.9E \n",
			iTile, Iz0, sigma_Izz, p_Iz0_summands_host[iTile], p_summands_host[iTile]);
		// Track down what cell causing NaN Iz0
	};

	f64 Ez_strength_ = (Iz_prescribed_endtime - Iz0) / sigma_Izz;
	Set_f64_constant(Ez_strength, Ez_strength_);

	f64 neg_Iz_per_triangle = -0.5*(Iz_prescribed_endtime + Iz_k) / (f64)numReverseJzTriangles;
	Set_f64_constant(negative_Iz_per_triangle, neg_Iz_per_triangle);

	printf("GPU: Iz_prescribed %1.14E Iz0 %1.14E sigma_Izz %1.14E \n",
		Iz_prescribed_endtime, Iz0, sigma_Izz);
	printf("Ez_strength (GPU) %1.14E \n", Ez_strength_);
	

	//// DEBUG:
	//pX_half->SendToHost(cuSyst_host);

	//sprintf(filename, "syst%d.txt", runs);
	//cuSyst_host.Output(filename);

	// Update velocities and Azdot:
	kernelCalculateVelocityAndAzdot <<<numTilesMinor, threadsPerTileMinor >>>(
		0.5*Timestep,
		pX_half->p_info,
		p_vn0,
		p_v0,
		p_OhmsCoeffs,
		pX_target->p_AAdot,  // why target? intermediate value
		pX_half->p_n_minor,
		this->p_AreaMinor,

		pX_half->p_AAdot,
		pX_half->p_vie,
		pX_half->p_v_n
		);
	Call(cudaThreadSynchronize(), "cudaTS AccelerateUpdate 1");
	
	/*
	Estimate_Effect_on_Integral_Azdot_from_Jz_and_LapAz << <numTilesMinor, threadsPerTileMinor >> > (
		0.5*Timestep,
		pX_half->p_info,
		this->p_n_minor,
		pX_half->p_n_minor, // used in UpdateVelocity
		this->p_vie,
		pX_half->p_vie,
		this->p_AreaMinor,
		p_LapAz,

		pX_half->p_AAdot,

		p_temp1, // +ve Jz
		p_temp2, // -ve Jz
		p_temp3, // LapAz*AreaMinor
		p_temp4, // integrate Azdot diff
		p_temp5,
		p_temp6
		);
	Call(cudaThreadSynchronize(), "cudaTS Estimate Effect");

	cudaMemcpy(p_temphost1, p_temp1, numTilesMinor * sizeof(f64), cudaMemcpyDeviceToHost);
	cudaMemcpy(p_temphost2, p_temp2, numTilesMinor * sizeof(f64), cudaMemcpyDeviceToHost);
	cudaMemcpy(p_temphost3, p_temp3, numTilesMinor * sizeof(f64), cudaMemcpyDeviceToHost);
	cudaMemcpy(p_temphost4, p_temp4, numTilesMinor * sizeof(f64), cudaMemcpyDeviceToHost);
	cudaMemcpy(p_temphost5, p_temp5, numTilesMinor * sizeof(f64), cudaMemcpyDeviceToHost);
	cudaMemcpy(p_temphost6, p_temp6, numTilesMinor * sizeof(f64), cudaMemcpyDeviceToHost);

//	printf("temphost 4  : \n");
	sum_plus = 0.0; sum_minus = 0.0; sum_Lap = 0.0; sum_Azdot = 0.0;
	abs_Lap = 0.0; abs_Azdot = 0.0;

	for (i = 0; i < numTilesMinor; i++)
	{
		sum_plus += p_temphost1[i];
		sum_minus += p_temphost2[i];
		sum_Lap += p_temphost3[i];
		abs_Lap += p_temphost4[i];
		sum_Azdot += p_temphost5[i];
		abs_Azdot += p_temphost6[i];
	//	printf("   %1.15E  |  ",p_temphost4[i]);
	//	if (i % 4 == 0) printf("\n");
	}
	printf("\n\n");
	fprintf(fp_dbg, "runs %d : sum_plus %1.15E sum_minus %1.15E sum_Lap %1.15E abs_Lap %1.15E sum_Azdot %1.15E abs_Azdot %1.15E \n",
		runs, sum_plus, sum_minus, sum_Lap, abs_Lap, sum_Azdot, abs_Azdot);

	printf("runs %d : sum_plus %1.15E sum_minus %1.15E sum_Lap %1.15E abs_Lap %1.15E sum_Azdot %1.15E abs_Azdot %1.15E \n",
		runs, sum_plus, sum_minus, sum_Lap, abs_Lap, sum_Azdot, abs_Azdot);
		*/
	/*
	fp_2 = fopen("LapAzetc1.txt", "w");
	this->SendToHost(cuSyst_host);	
	for (iMinor = 0; iMinor < NMINOR; iMinor++)
	{
		p_temphost4[iMinor] = q_*cuSyst_host.p_n_minor[iMinor].n*(cuSyst_host.p_vie[iMinor].viz -
			cuSyst_host.p_vie[iMinor].vez);
		p_temphost1[iMinor] = cuSyst_host.p_AAdot[iMinor].Azdot; 
	}
	cudaMemcpy(p_temphost3, p_LapAz, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
	pX_half->SendToHost(cuSyst_host);
	cudaMemcpy(cuSyst_host.p_AreaMinor, this->p_AreaMinor, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
	for (iMinor = 0; iMinor < NMINOR; iMinor++)
	{
		p_temphost5[iMinor] = q_*cuSyst_host.p_n_minor[iMinor].n*(cuSyst_host.p_vie[iMinor].viz -
			cuSyst_host.p_vie[iMinor].vez);
		
		p_temphost2[iMinor] = cuSyst_host.p_AAdot[iMinor].Azdot; // the calc'd one which is ~1% out.

		fprintf(fp_2, "%d LapAz %1.15E Azdot0 %1.15E Azdot1 %1.15E Jz0 %1.15E Jz1 %1.15E AreaMinor %1.15E\n",
			iMinor, p_temphost3[iMinor],
			p_temphost1[iMinor], p_temphost2[iMinor],
			p_temphost4[iMinor], p_temphost5[iMinor],
			cuSyst_host.p_AreaMinor[iMinor]
			);
	}
	fclose(fp_2);
	// The point here being to assess why on 2nd encounter it can't explain Azdot diff total from Lap Az total.
	printf("file saved.\n");
	*/
	
	// Output to file:

//	this->SendToHost(cuSyst_host);
//	long const VERTEX1 = CHOSEN1 - BEGINNING_OF_CENTRAL;
//	long const VERTEX2 = CHOSEN2 - BEGINNING_OF_CENTRAL;
//	cudaMemcpy(p_temphost5, p_temp5, NMINOR * sizeof(f64), cudaMemcpyDeviceToHost);
//	cudaMemcpy(p_temphost6, p_temp6, NMINOR * sizeof(f64), cudaMemcpyDeviceToHost);
	
/*	fprintf(fp_trajectory, "runs %d vez %1.14E %1.14E %1.14E %1.14E "
		"Te  %1.14E %1.14E %1.14E %1.14E "
		"n   %1.14E %1.14E %1.14E %1.14E "
		"nn  %1.14E %1.14E %1.14E %1.14E "
		"dTe_cond  %1.14E %1.14E %1.14E %1.14E "
		"dTe_ei  %1.14E %1.14E %1.14E %1.14E "
		"dTe_en  %1.14E %1.14E %1.14E %1.14E "
		"dTe_ionise  %1.14E %1.14E %1.14E %1.14E "
		"dvez_en+ei  %1.14E %1.14E %1.14E %1.14E "
		"dvez_E  %1.14E %1.14E %1.14E %1.14E \n",
		runs,
		cuSyst_host.p_vie[CHOSEN1].vez, cuSyst_host.p_vie[CHOSEN2].vez, cuSyst_host.p_vie[CHOSEN2 + 1].vez, cuSyst_host.p_vie[CHOSEN2 + 2].vez,
		cuSyst_host.p_T_minor[CHOSEN1].Te, cuSyst_host.p_T_minor[CHOSEN2].Te, cuSyst_host.p_T_minor[CHOSEN2 + 1].Te, cuSyst_host.p_T_minor[CHOSEN2 + 2].Te,
		cuSyst_host.p_n_minor[CHOSEN1].n, cuSyst_host.p_n_minor[CHOSEN2].n, cuSyst_host.p_n_minor[CHOSEN2 + 1].n, cuSyst_host.p_n_minor[CHOSEN2 + 2].n,
		cuSyst_host.p_n_minor[CHOSEN1].n_n, cuSyst_host.p_n_minor[CHOSEN2].n_n, cuSyst_host.p_n_minor[CHOSEN2 + 1].n_n, cuSyst_host.p_n_minor[CHOSEN2 + 2].n_n,
		p_temphost1[VERTEX1], p_temphost1[VERTEX2], p_temphost1[VERTEX2 + 1], p_temphost1[VERTEX2 + 2],
		p_temphost2[VERTEX1], p_temphost2[VERTEX2], p_temphost2[VERTEX2 + 1], p_temphost2[VERTEX2 + 2],
		p_temphost3[VERTEX1], p_temphost3[VERTEX2], p_temphost3[VERTEX2 + 1], p_temphost3[VERTEX2 + 2],
		p_temphost4[VERTEX1], p_temphost4[VERTEX2], p_temphost4[VERTEX2 + 1], p_temphost4[VERTEX2 + 2],
		p_temphost5[CHOSEN1], p_temphost5[CHOSEN2], p_temphost5[CHOSEN2 + 1], p_temphost5[CHOSEN2 + 2],
		p_temphost6[CHOSEN1], p_temphost6[CHOSEN2], p_temphost6[CHOSEN2 + 1], p_temphost6[CHOSEN2 + 2]
	);*/
	
	// =====================

SetConsoleTextAttribute(hConsole, 14);
//cudaMemcpy(&tempf64, &(pX_half->p_vie[42940].vez), sizeof(f64), cudaMemcpyDeviceToHost);
//printf("\npX_half->vez[42940] %1.14E\n\n", tempf64);
//cudaMemcpy(&tempf64, &(p_MAR_elec[42940].z), sizeof(f64), cudaMemcpyDeviceToHost);
//printf("\np_MAR_elec[42940].z %1.14E\n\n", tempf64);
//cudaMemcpy(&temp_vec2, &(pX_half->p_info[42940].pos), sizeof(f64_vec2), cudaMemcpyDeviceToHost);
//printf("\nposition %1.14E %1.14E\n\n", temp_vec2.x, temp_vec2.y);

SetConsoleTextAttribute(hConsole, 15);

	kernelAdvanceAzEuler << <numTilesMinor, threadsPerTileMinor >> >
		(0.5*h, this->p_AAdot, pX_half->p_AAdot, p_ROCAzduetoAdvection);
	Call(cudaThreadSynchronize(), "cudaTS AdvanceAzEuler");

	kernelResetFrillsAz_II << < numTriTiles, threadsPerTileMinor >> > (
		this->p_info, this->p_tri_neigh_index, pX_half->p_AAdot);
	Call(cudaThreadSynchronize(), "cudaTS ResetFrills I");
	
	printf("\nDebugNaN pX_half\n\n");
	DebugNaN(pX_half);

	// ))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
	// 
	// Now do the n,T,x advance to pDestmesh:
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	SetConsoleTextAttribute(hConsole, 10);

	kernelCalculateOverallVelocitiesVertices << <numTilesMajor, threadsPerTileMajor >> >(
		pX_half->p_info,
		pX_half->p_vie + BEGINNING_OF_CENTRAL,
		pX_half->p_v_n + BEGINNING_OF_CENTRAL,
		pX_half->p_n_major,
		pX_half->p_v_overall_minor + BEGINNING_OF_CENTRAL,
		p_n_shards,
		p_n_shards_n,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert
		);
	Call(cudaThreadSynchronize(), "cudaTS calculate overall velocities 22");
	// 
	//kernelAdvectPositionsVertex << <numTilesMajor, threadsPerTileMajor >> >(
	//	Timestep,
	//	this->p_info + BEGINNING_OF_CENTRAL,
	//	pX_target->p_info + BEGINNING_OF_CENTRAL,
	//	pX_half->p_v_overall_minor + BEGINNING_OF_CENTRAL,
	//	pX_half->p_n_major,
	//	this->p_izNeigh_vert,
	//	this->p_szPBCneigh_vert
	//	);
	//Call(cudaThreadSynchronize(), "cudaTS AdvectPositions_Vertex");
	//  
	kernelAverageOverallVelocitiesTriangles << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_v_overall_minor + BEGINNING_OF_CENTRAL,
		pX_half->p_v_overall_minor,
		pX_half->p_info,
		pX_half->p_tri_corner_index,
		pX_half->p_tri_periodic_corner_flags // questionable wisdom of repeating info in 3 systems
		);
	Call(cudaThreadSynchronize(), "cudaTS avg overall v to triangles 22");
	 
	kernelAdvectPositionsTris << <numTilesMinor, threadsPerTileMinor >> >(
		Timestep,
		this->p_info,
		pX_target->p_info,
		pX_half->p_v_overall_minor
		);
	Call(cudaThreadSynchronize(), "cudaTS AdvectPositions_CopyTris 22");
	
	SetConsoleTextAttribute(hConsole, 15);

	//cudaMemcpy(&temp_vec2, &(pX_target->p_info[21554 + BEGINNING_OF_CENTRAL].pos), sizeof(f64_vec2), cudaMemcpyDeviceToHost);
	//printf("\nPOSITION 21554 %1.12E %1.12E\n\n", temp_vec2.x, temp_vec2.y);
	
	kernelCalculateUpwindDensity_tris << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_info,
		p_n_shards_n,
		p_n_shards,
		pX_half->p_vie,
		pX_half->p_v_n,
		pX_half->p_v_overall_minor,
		pX_half->p_tri_corner_index,
		pX_half->p_tri_neigh_index,
		pX_half->p_who_am_I_to_corner,
		pX_half->p_tri_periodic_neigh_flags,
		pX_half->p_n_upwind_minor,
		pX_half->p_T_minor,
		p_T_upwind_minor
		);
	Call(cudaThreadSynchronize(), "cudaTS CalculateUpwindDensity_tris pX_half");

	cudaMemset(p_Integrated_div_v_overall, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Div_v_neut, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Div_v, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(NT_addition_rates_d, 0, sizeof(NTrates)*NUMVERTICES);
	
	SetConsoleTextAttribute(hConsole, 14);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NeTe %1.14E\n\n", VERTCHOSEN, tempf64);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NiTi %1.10E\n\n", VERTCHOSEN, tempf64);
	SetConsoleTextAttribute(hConsole, 15);

	kernelAccumulateAdvectiveMassHeatRate << <numTilesMajor, threadsPerTileMajor >> >(
		Timestep,
		pX_half->p_info,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,

		pX_half->p_n_major,
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL, 

		pX_half->p_n_upwind_minor,
		pX_half->p_vie,
		pX_half->p_v_n,
		pX_half->p_v_overall_minor,
		p_T_upwind_minor, 

		NT_addition_rates_d,
		p_Div_v,
		p_Div_v_neut,
		p_Integrated_div_v_overall);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateAdvectiveMassHeatRate pX_half");

	SetConsoleTextAttribute(hConsole, 14);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NeTe %1.10E\n\n", VERTCHOSEN, tempf64);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NiTi %1.10E\n\n", VERTCHOSEN, tempf64);
	SetConsoleTextAttribute(hConsole, 15);
	// [ do Az advance above when we advance Azdot. ]

	CallMAC(cudaMemset(p_MAR_neut, 0, sizeof(f64_vec3)*NMINOR));
	CallMAC(cudaMemset(p_MAR_ion, 0, sizeof(f64_vec3)*NMINOR));
	CallMAC(cudaMemset(p_MAR_elec, 0, sizeof(f64_vec3)*NMINOR));

	// Now notice we take a grad Azdot but Azdot has not been defined except from time t_k!!
	kernelCreate_pressure_gradT_and_gradA_LapA_CurlA_minor << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_info,
		pX_half->p_T_minor,
		pX_half->p_AAdot,
		  
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_izNeigh_TriMinor,
		pX_half->p_szPBC_triminor,
		pX_half->p_who_am_I_to_corner,
		pX_half->p_tri_corner_index,
		 
		p_MAR_neut, p_MAR_ion, p_MAR_elec,
		p_n_shards,				// this kernel is for i+e only
		pX_half->p_n_minor,

		p_GradTe,
		p_GradAz,
		p_LapAz,

		p_ROCAzduetoAdvection, // Would probs be better to split out Az calc, remember
		p_ROCAzdotduetoAdvection, // Would probs be better to split out Az calc, remember
		pX_half->p_v_overall_minor, // it's only this time that we need to collect it ofc.
									// grad Azdot requires storing Azdot. I do not like it.

									// Should make a constant switch.
									// What about grad Azdot          ::slap::
									// Let ROCAzdot = 0 on the first big go-around and it doesn't matter.
									// Let it feed into CalculateVelocityAndAzdot in the leapfrog and BJLS.			

		pX_half->p_B,
		pX_half->p_AreaMinor
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelCreate_pressure_gradT_and_gradA_LapA_CurlA_minor pX_half");

	SetConsoleTextAttribute(hConsole, 14);
	cudaMemcpy(&tempf64, &(p_MAR_elec[CHOSEN].y), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\np_MAR_elec[%d].y %1.14E\n\n", CHOSEN, tempf64);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NiTi %1.10E\n\n", VERTCHOSEN, tempf64);
	SetConsoleTextAttribute(hConsole, 15);

	kernelCreate_momflux_minor << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_info,
		pX_half->p_vie,
		pX_half->p_v_overall_minor,

		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_izNeigh_TriMinor,
		pX_half->p_szPBC_triminor,

		pX_half->p_who_am_I_to_corner,
		pX_half->p_tri_corner_index,

		p_MAR_neut, p_MAR_ion, p_MAR_elec,
		p_n_shards
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelCreate_momflux_minor pX_half");

	SetConsoleTextAttribute(hConsole, 14);
	cudaMemcpy(&tempf64, &(p_MAR_elec[CHOSEN].y), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\np_MAR_elec[%d].y %1.14E\n\n", CHOSEN, tempf64);
	SetConsoleTextAttribute(hConsole, 15);

	kernelNeutral_pressure_and_momflux << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_info,

		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_izNeigh_TriMinor,
		pX_half->p_szPBC_triminor,
		pX_half->p_who_am_I_to_corner,
		pX_half->p_tri_corner_index,

		pX_half->p_T_minor,
		pX_half->p_v_n,
		p_n_shards_n,
		pX_half->p_n_minor,

		pX_half->p_v_overall_minor,
		p_MAR_neut
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelNeutral_pressure_and_momflux pX_half");
	  
	 
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].N), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("N addition rate %1.14E \n\n", tempf64);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NiTi %1.10E\n\n", VERTCHOSEN, tempf64);

	/*
	fprintf(fp_traj, " | n ");
	for (i = 11765; i < 11770; i++) {
		cudaMemcpy(&tempf64, &(pX_half->p_n_major[i].n), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}
	for (i = 11621; i <= 11623; i++) {
		cudaMemcpy(&tempf64, &(pX_half->p_n_major[i].n), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}
	fprintf(fp_traj, " | Te ");
	for (i = 11765; i < 11770; i++) {
		cudaMemcpy(&tempf64, &(pX_half->p_T_minor[i + BEGINNING_OF_CENTRAL].Te), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}
	for (i = 11621; i <= 11623; i++) {
		cudaMemcpy(&tempf64, &(pX_half->p_T_minor[i + BEGINNING_OF_CENTRAL].Te), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}
	fprintf(fp_traj, " | vez ");
	for (i = 11765; i < 11770; i++) {
		cudaMemcpy(&tempf64, &(pX_half->p_vie[i + BEGINNING_OF_CENTRAL].vez), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}
	for (i = 11621; i <= 11623; i++) {
		cudaMemcpy(&tempf64, &(pX_half->p_vie[i + BEGINNING_OF_CENTRAL].vez), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}
	fprintf(fp_traj, " | nu_eHeart ");
	for (i = 11765; i < 11770; i++) {
		cudaMemcpy(&tempf64, &(p_nu_major[i].e), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}
	for (i = 11621; i <= 11623; i++) {
		cudaMemcpy(&tempf64, &(p_nu_major[i].e), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}*/
	SetConsoleTextAttribute(hConsole, 14);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NeTe %1.10E\n\n", VERTCHOSEN, tempf64);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NiTi %1.10E\n\n", VERTCHOSEN, tempf64);
	SetConsoleTextAttribute(hConsole, 15);


	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_n_minor,
		pX_half->p_n_major,
		pX_half->p_T_minor,
		pX_half->p_info,
		pX_half->p_cc,  // Calculate circumcenters; we would like n and T there for shards.

		pX_half->p_tri_corner_index,
		pX_half->p_tri_periodic_corner_flags,

		true // calculate n and T on circumcenters instead of centroids
		); // call before CreateShardModel 
	Call(cudaThreadSynchronize(), "cudaTS Average_nTx cc");

	for (long iIterate = 0; iIterate < 3; iIterate++)
	{
		kernelCreateShardModelOfDensities_And_SetMajorArea << <numTilesMajor, threadsPerTileMajor >> >(
			pX_half->p_info,
			pX_half->p_n_major,
			pX_half->p_n_minor,  // DESIRED VALUES
			pX_half->p_izTri_vert,
			pX_half->p_szPBCtri_vert,
			pX_half->p_cc,
			p_n_shards,
			p_n_shards_n,
			pX_half->p_AreaMajor,
			true // Use circumcenter
			);
		Call(cudaThreadSynchronize(), "cudaTS CreateShardModels ..");

		kernelInferMinorDensitiesFromShardModel << <numTriTiles, threadsPerTileMinor >> >(
			pX_half->p_info,
			pX_half->p_n_minor,
			p_n_shards,
			p_n_shards_n,
			pX_half->p_tri_corner_index,
			pX_half->p_who_am_I_to_corner,
			p_one_over_n); // not used in accel below
		Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities ..");
	};
	/*
	kernelCalculate_kappa_nu << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_info,
		pX_half->p_n_minor,
		pX_half->p_T_minor,

		p_kappa_n,
		p_kappa_i,
		p_kappa_e,
		p_nu_i,
		p_nu_e		// define all these.
		);
	Call(cudaThreadSynchronize(), "cudaTS Calculate_kappa_nu(pXhalf)");
	*/
	RunBackwardJLSForHeat(this->p_T_minor + BEGINNING_OF_CENTRAL,
		pX_target->p_T_minor + BEGINNING_OF_CENTRAL,
		Timestep,
		pX_half);

	kernelAccumulateDiffusiveHeatRate_new << <numTilesMajorClever, threadsPerTileMajorClever >> >(
		Timestep,
		pX_half->p_info,
		pX_half->p_izNeigh_vert,
		pX_half->p_szPBCneigh_vert,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_cc,

		pX_half->p_n_major,
		pX_target->p_T_minor + BEGINNING_OF_CENTRAL, // Use T_k+1 just calculated...
		pX_half->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
		p_kappa_n,
		p_kappa_i,
		p_kappa_e,
		p_nu_i,
		p_nu_e,
		NT_addition_rates_d,
		pX_half->p_AreaMajor);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate pX_half");
	// To increase the efficiency we want to make a clever 2nd set of major tiles of size 192. Also try 256, 384.

	SetConsoleTextAttribute(hConsole, 14);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NeTe %1.10E\n\n", VERTCHOSEN, tempf64);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NiTi %1.10E\n\n", VERTCHOSEN, tempf64);
	SetConsoleTextAttribute(hConsole, 15);

	kernelIonisationRates << <numTilesMajor, threadsPerTileMajor >> >(
		0.5*Timestep,
		pX_half->p_info,
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices
		pX_half->p_n_major,
		pX_half->p_AreaMajor,
		NT_addition_rates_d
		);
	Call(cudaThreadSynchronize(), "cudaTS Ionisation pXhalf");

	SetConsoleTextAttribute(hConsole, 14);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NeTe %1.10E\n\n", VERTCHOSEN, tempf64);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NiTi %1.10E\n\n", VERTCHOSEN, tempf64);
	SetConsoleTextAttribute(hConsole, 15);


	kernelReset_v_in_outer_frill_and_outermost << <numTilesMinor, threadsPerTileMinor >> >
		(
			pX_half->p_info,
			pX_half->p_vie,
			pX_half->p_v_n,
			pX_half->p_tri_neigh_index,
			pX_half->p_izNeigh_vert
			);
	Call(cudaThreadSynchronize(), "cudaTS resetv");


	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_n_minor,
		pX_half->p_n_major,
		pX_half->p_T_minor,
		pX_half->p_info,
		pX_half->p_cc,  // Calculate circumcenters; we would like n and T there for shards.

		pX_half->p_tri_corner_index,
		pX_half->p_tri_periodic_corner_flags,

		false // calculate n and T on centroids
		); // call before CreateShardModel 
	Call(cudaThreadSynchronize(), "cudaTS Average_nTx pos");
	 
	kernelCalculate_ita_visc << <numTilesMinor, threadsPerTileMinor >> >(
		pX_half->p_info,
		pX_half->p_n_minor, // Now on centroids so need to have put it back
		pX_half->p_T_minor,
		 
		p_temp3,
		p_temp4,
		p_temp5,
		p_temp1,
		p_temp2,
		p_temp6);
	Call(cudaThreadSynchronize(), "cudaTS ita");

	//cudaMemcpy(&tempf64, &(NT_addition_rates_d[11797].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
	//printf("\nNT_addition_rates_d[11797].NeTe %1.10E\n\n", tempf64);
	/*
	fprintf(fp_traj, " | ita ");
	for (i = 11765; i < 11770; i++) {
		cudaMemcpy(&tempf64, &(p_temp2[i + BEGINNING_OF_CENTRAL]), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}
	for (i = 11621; i <= 11623; i++) {
		cudaMemcpy(&tempf64, &(p_temp2[i + BEGINNING_OF_CENTRAL]), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}
	fprintf(fp_traj, " | nu_ei ");
	for (i = 11765; i < 11770; i++) {
		cudaMemcpy(&tempf64, &(p_temp4[i + BEGINNING_OF_CENTRAL]), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}
	for (i = 11621; i <= 11623; i++) {
		cudaMemcpy(&tempf64, &(p_temp4[i + BEGINNING_OF_CENTRAL]), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}

	fprintf(fp_traj, " | Base_add_nT ");
	for (i = 11765; i < 11770; i++) {
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[i].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}
	for (i = 11621; i <= 11623; i++) {
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[i].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}*/
	SetConsoleTextAttribute(hConsole, 14);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NeTe %1.10E\n\n", VERTCHOSEN, tempf64);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NiTi %1.10E\n\n", VERTCHOSEN, tempf64);
	SetConsoleTextAttribute(hConsole, 15);

	SetConsoleTextAttribute(hConsole, 14);
	cudaMemcpy(&tempf64, &(p_MAR_elec[CHOSEN].y), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\np_MAR_elec[%d].y %1.14E\n\n", CHOSEN, tempf64);
	SetConsoleTextAttribute(hConsole, 15);

	::RunBackwardJLSForViscosity(this->p_vie, pX_target->p_vie, Timestep, pX_half);

	cudaMemcpy(&tempf64, &(this->p_vie[14791 + BEGINNING_OF_CENTRAL].vxy.y), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("pX vy[vert 14791] %1.14E \n", tempf64);
	cudaMemcpy(&tempf64, &(pX_target->p_vie[14791 + BEGINNING_OF_CENTRAL].vxy.y), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("pX_target vy[vert 14791] %1.14E \n\n", tempf64);



	cudaMemset(NT_addition_tri_d, 0, sizeof(NTrates)*NUMVERTICES * 2);
	kernelCreate_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_info,
		pX_target->p_vie,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_izNeigh_TriMinor,
		pX_half->p_szPBC_triminor,
		p_temp1, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
		p_temp2, //f64 * __restrict__ p_ita_parallel_elec_minor,   // nT / nu ready to look up
		p_temp3, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
		p_temp4, // f64 * __restrict__ p_nu_elec_minor,   // nT / nu ready to look up
		pX_half->p_B,
		p_MAR_ion, // just accumulates
		p_MAR_elec,
		NT_addition_rates_d,
		NT_addition_tri_d);
	Call(cudaThreadSynchronize(), "cudaTS visccontrib 2");

	SetConsoleTextAttribute(hConsole, 14);
	cudaMemcpy(&tempf64, &(p_MAR_elec[CHOSEN].y), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\np_MAR_elec[%d].y %1.14E\n\n", CHOSEN, tempf64);
	SetConsoleTextAttribute(hConsole, 15);

	getch();

	Collect_Nsum_at_tris << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_info,
		pX_half->p_n_minor,
		pX_half->p_tri_corner_index,
		pX_half->p_AreaMajor, // populated?
		p_temp4);
	Call(cudaThreadSynchronize(), "cudaTS Nsum 2");

	SetConsoleTextAttribute(hConsole, 14);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NeTe %1.10E\n\n", VERTCHOSEN, tempf64);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NiTi %1.10E\n\n", VERTCHOSEN, tempf64);
	SetConsoleTextAttribute(hConsole, 15);

	kernelTransmitHeatToVerts << <numTilesMajor, threadsPerTileMajor >> > (
		pX_half->p_info,
		pX_half->p_izTri_vert,
		pX_half->p_n_minor,
		pX_half->p_AreaMajor,
		p_temp4,
		NT_addition_rates_d,
		NT_addition_tri_d
		);
	Call(cudaThreadSynchronize(), "cudaTS sum up heat 2");

	SetConsoleTextAttribute(hConsole, 14);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NeTe %1.10E\n\n", VERTCHOSEN, tempf64);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NiTi %1.10E\n\n", VERTCHOSEN, tempf64);
	SetConsoleTextAttribute(hConsole, 15);

	/*fprintf(fp_traj, " | Total_addnT ");
	for (i = 11765; i < 11770; i++) {
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[i].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}
	for (i = 11621; i <= 11623; i++) {
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[i].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		fprintf(fp_traj, " %1.12E", tempf64);
	}
	fprintf(fp_traj, "\n");

	fclose(fp_traj);*/

	kernelAdvanceDensityAndTemperature << <numTilesMajor, threadsPerTileMajor >> >(
		Timestep,
		this->p_info + BEGINNING_OF_CENTRAL,
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL,
		NT_addition_rates_d,
		pX_half->p_n_major,  // ?
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL,  // ?

		pX_half->p_vie + BEGINNING_OF_CENTRAL,
		pX_half->p_v_n + BEGINNING_OF_CENTRAL,

		p_Div_v_neut, p_Div_v,
		p_Integrated_div_v_overall,
		this->p_AreaMajor,

		pX_target->p_n_major,
		pX_target->p_T_minor + BEGINNING_OF_CENTRAL
		);
	Call(cudaThreadSynchronize(), "cudaTS Advance_n_and_T 2330");
	
	// QUESTION QUESTION : What should params be there?

	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> >(
		pX_target->p_n_minor,
		pX_target->p_n_major,
		pX_target->p_T_minor,
		pX_target->p_info,
		pX_target->p_cc,
		pX_target->p_tri_corner_index,
		pX_target->p_tri_periodic_corner_flags,
		false
		); // call before CreateShardModel 
	Call(cudaThreadSynchronize(), "cudaTS average nTx 233");
	cudaMemcpy(pX_target->p_n_minor + BEGINNING_OF_CENTRAL,
		pX_target->p_n_major, sizeof(nvals)*NUMVERTICES, cudaMemcpyDeviceToDevice);

	//pX_half->Add_ViscousMomentumFluxRates(AdditionalMomRates);

	//////////////////////////////////////////////////////////////////////////
	// Even more shards!:

#ifndef USE_N_MAJOR_FOR_VERTEX
	kernelCreateShardModelOfDensities_And_SetMajorArea << <numTilesMajor, threadsPerTileMajor >> >(
		pX_target->p_info,
		pX_target->p_n_major,
		pX_target->p_n_minor,
		pX_target->p_izTri_vert,
		pX_target->p_szPBCtri_vert, // could be more intelligent way than storing and copying 3x
		p_n_shards,
		p_n_shards_n,
		//		p_Tri_n_lists,
		//		p_Tri_n_n_lists,
		pX_target->p_AreaMajor);
	Call(cudaThreadSynchronize(), "cudaTS CreateShardModel pX_target");

	kernelInferMinorDensitiesFromShardModel << <numTriTiles, threadsPerTileMinor >> >(
		pX_target->p_info,
		pX_target->p_n_minor,
		p_n_shards,
		p_n_shards_n,
		pX_target->p_tri_corner_index,
		pX_target->p_who_am_I_to_corner,
		p_one_over_n);// (At the moment just repopulating tri minor n.)
	Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities pX_target");
#endif

	// ============================================================

	f64 starttime = evaltime;
	printf("run %d ", runs);
	cudaEventRecord(middle,0);
	cudaEventSynchronize(middle);

	kernelGetLapCoeffs_and_min << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_info,
		pX_half->p_izTri_vert,
		pX_half->p_izNeigh_TriMinor,
		pX_half->p_szPBCtri_vert,
		pX_half->p_szPBC_triminor,
		p_LapCoeffself,
		p_temp1, // collect min
		p_longtemp
		);
	Call(cudaThreadSynchronize(), "cudaTS GetLapCoeffs");

	cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTriTiles, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_longtemphost, p_longtemp, sizeof(long)*numTriTiles, cudaMemcpyDeviceToHost);
	
	// It should be universally true that coeffself is negative. Higher self = more negative Lap.
	f64 mincoeffself = 0.0;
	long iMin = -1;
	for (iTile = 0; iTile < numTriTiles; iTile++)
	{
		if (p_temphost1[iTile] < mincoeffself) {
			mincoeffself = p_temphost1[iTile];
			iMin = p_longtemphost[iTile];
		}
	//	printf("iTile %d iMin %d cs %1.12E \n", iTile, p_longtemphost[iTile], p_temphost1[iTile]);
	}
	
	f64 h_sub_max = 1.0 / (c_*sqrt(fabs(mincoeffself))); // not strictly correct - 
	// e.g. with this at 7.3e-14, using 1e-13 as substep works (10% backward) ; with it at 6.4e-14 it does not work. 
	// So the inflation factor that you can get away with, isn't huge.
	printf("\nMin coeffself %1.12E iMin %d 1.0/(c sqrt(-mcs)) %1.12E\n", mincoeffself, iMin,
		h_sub_max);
	// Comes out with sensible values for max abs coeff ~~ delta squared?

	// Add in factor to inflate Timestep when we want to play around.

	iSubcycles = (long)(Timestep / h_sub_max)+1;


	if (runs % BWD_SUBCYCLE_FREQ == 0) {
		printf("backward!\n");
		iSubcycles /= BWD_STEP_RATIO; // some speedup this way
	} else {
		iSubcycles *= FWD_STEP_FACTOR;
	}
	
	hsub = Timestep / (real)iSubcycles;
		
	printf("hsub = %1.14E subcycles %d \n", hsub, iSubcycles);

	FILE * fptr = fopen("hsub.txt","a");
	fprintf(fptr, "GlobalStepsCounter %d runs %d hsub %1.14E iSubcycles %d backward %d h_sub_max %1.14E mincoeffself %1.14E \n",
		GlobalStepsCounter, runs, hsub, iSubcycles, (runs % 10 == 0)?1:0, h_sub_max, mincoeffself);
	fclose(fptr);

	// CHANGE TO hsub NOT PROPAGATED TO THE CPU COMPARISON DEBUG ROUTINE!!!
	
	cudaMemcpy(&tempf64, &(pX_target->p_T_minor[85495].Te), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("pX_target->p_T_minor[85495].Te %1.10E\n", tempf64);
	
	printf("\nDebugNaN target \n\n");
	DebugNaN(pX_target);

	if (runs % BWD_SUBCYCLE_FREQ == 0) // maybe some easy speed up this way...
	{
		
		kernelPullAzFromSyst<<<numTilesMinor, threadsPerTileMinor>>>(
			this->p_AAdot,
			p_Az
		);
		Call(cudaThreadSynchronize(), "cudaTS PullAz");
		 
	//	f64 Azdotarrays[22][40], Azarrays[22][40];
	//	f64 Azarraytemp[40];
	//	AAdot AAdotarray[40];
		int i;
		 
		// iSubcycles
		for (iSubstep = 0; iSubstep < iSubcycles; iSubstep++) // arguably for backward steps we could allow 2x or 3x bigger steps.
		{
			//printf("\n\n#########################\n");
			printf("####### SUBSTEP %d / %d #######\n\n", iSubstep, iSubcycles);

			evaltime += 0.5*hsub;
			kernelInterpolateVarsAndPositions << <numTilesMinor, threadsPerTileMinor >> > (
				(evaltime - starttime) / Timestep,
				this->p_info,
				pX_target->p_info,
				this->p_n_minor,
				pX_target->p_n_minor,
				this->p_T_minor,
				pX_target->p_T_minor,
				//				this->p_B,
					//			pX_target->p_B,
				pX_half->p_info,
				pX_half->p_n_minor,
				pX_half->p_T_minor
				//		pX_half->p_B
				);
			Call(cudaThreadSynchronize(), "cudaTS InterpolateVars");

			//pX_half->GetLapFromCoeffs(Az_array, LapAzArray);
	// NOTICE # BLOCKS -- THIS SHOULD ALSO APPLY WHEREVER WE DO SIMILAR THING LIKE WITH MOMFLUX.

			kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
				pX_half->p_info,
				p_Az,
				pX_half->p_izTri_vert,
				pX_half->p_izNeigh_TriMinor,
				pX_half->p_szPBCtri_vert,
				pX_half->p_szPBC_triminor,
				p_LapAz,
				//			p_temp1, p_temp2, p_temp3,
				pX_half->p_AreaMinor
				);
			Call(cudaThreadSynchronize(), "cudaTS GetLapMinor aaa2");
			// Should populate an array of coefficients s.t. Azdot_k+1 = ~Azdot0~ + ~gamma~ Lap Az
			// Now we will wanna create each eqn for Az with coeffs on neighbour values.
			// So we need a func called "GetLapCoefficients".

			// store at t_k :
			cudaMemcpy(p_temp6, p_LapAz, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);


			// The equation is A_k+1 - h~gamma~ Lap A_k+1 - A_k - h~Azdot0~ = 0
			// Calculate regressor x_Jacobi from eps/coeff_on_A_i
			// Given a proposed addition of a regressor x, deps/dbeta = x - h~gamma~ Lap x 
			// Set beta = -sum[eps deps/dbeta] / sum[deps/dbeta ^2]

			// evaltime + 0.5*hsub used for setting EzStrength://
			Iz_prescribed_endtime = GetIzPrescribed(evaltime + 0.5*hsub); // APPLIED AT END TIME: we are determining
		//	Iz_prescribed_starttime = GetIzPrescribed(evaltime - 0.5*hsub); // APPLIED AT END TIME: we are determining
																	 // Jz, hence Iz at k+hsub initially.

			kernelEstimateCurrent << <numTilesMinor, threadsPerTileMinor >> > (
				pX_half->p_info,
				pX_half->p_n_minor,
				(iSubstep == 0) ? this->p_vie : pX_target->p_vie,
				pX_half->p_AreaMinor,
				p_temp1); // Iz_k
			Call(cudaThreadSynchronize(), "cudaTS Estimate Iz_k");

	//		cudaMemcpy(&tempf64, &(p_MAR_elec[11653 + BEGINNING_OF_CENTRAL].y), sizeof(f64), cudaMemcpyDeviceToHost);
	//		printf("MAR.y 11653 %1.8E", tempf64);

			kernelPopulateOhmsLaw << <numTilesMinor, threadsPerTileMinor >> > (
				hsub,// ROCAzdotduetoAdvection, 
				pX_half->p_info,
				p_MAR_neut, p_MAR_ion, p_MAR_elec,
				pX_half->p_B,
				p_LapAz,
				p_GradAz,
				p_GradTe,

				pX_half->p_n_minor,

				p_one_over_n2, // was that from time pXhalf?

				pX_half->p_T_minor,
				(iSubstep == 0) ? this->p_vie : pX_target->p_vie,
				(iSubstep == 0) ? this->p_v_n : pX_target->p_v_n,
				(iSubstep == 0) ? this->p_AAdot : pX_target->p_AAdot, //	inputs
				pX_half->p_AreaMinor, // pop'd? interp?
	 			p_ROCAzdotduetoAdvection,

				p_vn0,
				p_v0,
				p_OhmsCoeffs,
				pX_half->p_AAdot, // save intermediate value ............................
				p_Iz0_summands,
				p_sigma_Izz,
				p_denom_i,
				p_denom_e, p_coeff_of_vez_upon_viz, p_beta_ie_z,
				true,
				false, // (iSubstep == iSubcycles - 1) ? true : false,
				pX_target->p_n_minor);
			Call(cudaThreadSynchronize(), "cudaTS kernelPopulateOhmsLaw ");
			 
			cudaMemcpy(p_Iz0_summands_host, p_Iz0_summands, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_summands_host, p_sigma_Izz, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			Iz0 = 0.0;
			f64 Sigma_Izz = 0.0;
			Iz_k = 0.0;
			long iBlock;
			for (iBlock = 0; iBlock < numTilesMinor; iBlock++)
			{
				Iz0 += p_Iz0_summands_host[iBlock];
				Sigma_Izz += p_summands_host[iBlock];
				Iz_k += p_temphost1[iBlock];
			}
			EzStrength_ = (Iz_prescribed_endtime - Iz0) / Sigma_Izz;
			Set_f64_constant(Ez_strength, EzStrength_);

			if (EzStrength_ != EzStrength_) { printf("end\n"); while (1) getch(); }
			
			neg_Iz_per_triangle = -0.5*(Iz_prescribed_endtime + Iz_k) / (f64)numReverseJzTriangles;
			Set_f64_constant(negative_Iz_per_triangle, neg_Iz_per_triangle);
			// Electrons travel from cathode to anode so Jz is down in filament,
			// up around anode.

			printf("\nGPU: Iz0 = %1.14E SigmaIzz %1.14E EzStrength = %1.14E \n\n", Iz0, Sigma_Izz, EzStrength_);

			kernelCreateLinearRelationship << <numTilesMinor, threadsPerTileMinor >> > (
				hsub,
				pX_half->p_info,
				p_OhmsCoeffs,
				p_v0,
				p_LapAz,  // used for cancelling .. we now wish to cancel only half.
				pX_half->p_n_minor,
				p_denom_e,
				p_denom_i, p_coeff_of_vez_upon_viz, p_beta_ie_z,
				pX_half->p_AAdot,
				pX_half->p_AreaMinor,
				p_Azdot0,
				p_gamma
				); // MODIFY vez0, viz0 IF THEY WILL EVER BE USED.
			Call(cudaThreadSynchronize(), "cudaTS kernelCreateLinearRelationship ");

			// _____________________________________________________________

			kernelCreateSeedPartOne << <numTilesMinor, threadsPerTileMinor >> > (
				hsub,
				p_Az,
				(iSubstep == 0) ? this->p_AAdot : pX_target->p_AAdot, // use 0.5*(Azdot_k + Azdot_k+1) for seed.
				p_AzNext);
			Call(cudaThreadSynchronize(), "cudaTS Create Seed 1");
			// p_AzNext[iMinor] = p_Az[iMinor] + h_use*0.5*p_AAdot_use[iMinor].Azdot; // good seed

			// Question whether this is even wanted: think no use for it.
			// Did not save adjustment to viz0 -- correct?
//
//			kernelCalculateVelocityAndAzdot << <numTilesMinor, threadsPerTileMinor >> >(
//				hsub,
//				p_vn0,
//				p_v0,
//				p_OhmsCoeffs,
//				pX_half->p_AAdot,
//
//				pX_target->p_AAdot,
//				pX_target->p_vie,
//				pX_target->p_v_n
//				);
//			Call(cudaThreadSynchronize(), "cudaTS kernelUpdate_v ");
//
			//Az_array_next[iMinor] = Az_array[iMinor] + 0.5*hsub*this->pData[iMinor].Azdot + 0.5*hsub * Azdot0[iMinor] + 0.5*hsub * gamma[iMinor] * LapAzArray[iMinor];
			//ie use 0.5*(Azdot_k[done] + Azdot_k+1) for seed.

			kernelCreateSeedPartTwo << <numTilesMinor, threadsPerTileMinor >> > (
				hsub,
				p_Azdot0, p_gamma, p_LapAz,
				p_AzNext);
			Call(cudaThreadSynchronize(), "cudaTS Create Seed 2"); // Okay -- we can now merge these. "Azdot_k" is preserved.
			// p_AzNext_update[iMinor] += 0.5*h_use* (p_Azdot0[iMinor]+ p_gamma[iMinor] * p_LapAz[iMinor]);

			// JLS:

			f64 sum_eps_deps_by_dbeta, sum_depsbydbeta_sq, sum_eps_eps, depsbydbeta;
			printf("\nJLS [beta L2eps]: ");
			long iMinor;
			f64 beta, L2eps;
			Triangle * pTri;
			int iIteration;
			for (iIteration = 0; iIteration < NUM_BWD_ITERATIONS; iIteration++)
			{
				// 1. Create regressor:
				// Careful with major vs minor + BEGINNING_OF_CENTRAL:

				kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
					pX_half->p_info,
					p_AzNext,
					pX_half->p_izTri_vert,
					pX_half->p_izNeigh_TriMinor,
					pX_half->p_szPBCtri_vert,
					pX_half->p_szPBC_triminor,
					p_LapAzNext,
			//		p_temp1, p_temp2, p_temp3,
					pX_half->p_AreaMinor
					);
				Call(cudaThreadSynchronize(), "cudaTS GetLap Az JLS 1");

				kernelCreateEpsilonAndJacobi << <numTilesMinor, threadsPerTileMinor >> >
					(hsub, // ?
						pX_half->p_info,
						p_AzNext, p_Az,
						p_Azdot0, p_gamma,
						p_LapCoeffself, p_LapAzNext,
						p_epsilon, p_Jacobi_x,
						(iSubstep == 0) ? this->p_AAdot : pX_target->p_AAdot // to use Az_dot_k for equation);
						);
				Call(cudaThreadSynchronize(), "cudaTS CreateEpsAndJacobi 1");

				kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
					pX_half->p_info,
					p_Jacobi_x,
					pX_half->p_izTri_vert,
					pX_half->p_izNeigh_TriMinor,
					pX_half->p_szPBCtri_vert,
					pX_half->p_szPBC_triminor,
					p_LapJacobi,
			//		p_temp1, p_temp2, p_temp3,
					pX_half->p_AreaMinor
					);
				Call(cudaThreadSynchronize(), "cudaTS Get Lap Jacobi 1");

				kernelAccumulateSummands << <numTilesMinor, threadsPerTileMinor >> > (
					pX_half->p_info, // ?
					hsub,
					p_epsilon, p_Jacobi_x, p_LapJacobi, p_gamma,
					p_sum_eps_deps_by_dbeta,
					p_sum_depsbydbeta_sq,
					p_sum_eps_eps);
				Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands 1");

				cudaMemcpy(p_sum_depsbydbeta_sq_host, p_sum_depsbydbeta_sq, sizeof(f64)*numTilesMinor,
					cudaMemcpyDeviceToHost);
				cudaMemcpy(p_sum_eps_deps_by_dbeta_host, p_sum_eps_deps_by_dbeta, sizeof(f64)*numTilesMinor,
					cudaMemcpyDeviceToHost);
				cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMinor,
					cudaMemcpyDeviceToHost);

				sum_eps_deps_by_dbeta = 0.0;
				sum_depsbydbeta_sq = 0.0;
				sum_eps_eps = 0.0;
				for (iTile = 0; iTile < numTilesMinor; iTile++)
				{
					sum_eps_deps_by_dbeta += p_sum_eps_deps_by_dbeta_host[iTile];
					sum_depsbydbeta_sq += p_sum_depsbydbeta_sq_host[iTile];
					sum_eps_eps += p_sum_eps_eps_host[iTile];
				}
				beta = -sum_eps_deps_by_dbeta / sum_depsbydbeta_sq;
				L2eps = sqrt(sum_eps_eps / (real)NMINOR);

				printf("\n [ %1.14E %1.14E ] ", beta, L2eps);
				printf("sum_eps_deps_bydbeta %1.14E sum_depsbydbeta_sq %1.14E \n",
					sum_eps_deps_by_dbeta, sum_depsbydbeta_sq);

				kernelAdd << <numTilesMinor, threadsPerTileMinor >> > (
					p_AzNext, beta, p_Jacobi_x);
				//for (iMinor = 0; iMinor < NMINOR; iMinor++)
				//	Az_array_next[iMinor] += beta * Jacobi_x[iMinor];

				Call(cudaThreadSynchronize(), "cudaTS Add 1");

				// Try resetting frills here and ignoring in calculation:
				kernelResetFrillsAz << <numTriTiles, threadsPerTileMinor >> > (
					this->p_info, this->p_tri_neigh_index, p_AzNext);
				Call(cudaThreadSynchronize(), "cudaTS ResetFrills X");

			};

			printf("\n\n");

			// That was:
			//	JLS_for_Az_bwdstep(4, hsub); // populate Az_array with k+1 values
			
			cudaMemcpy(p_Az, p_AzNext, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);

	//		cudaMemcpy(&tempf64, &(p_Az[CHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
	//		printf("POINT 1 CHOSEN %d p_Az %1.15E \n", CHOSEN, tempf64);


			kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
				pX_half->p_info,
				p_Az,
				pX_half->p_izTri_vert,
				pX_half->p_izNeigh_TriMinor,
				pX_half->p_szPBCtri_vert,
				pX_half->p_szPBC_triminor,
				p_LapAz,
		//		p_temp1, p_temp2, p_temp3,
				pX_half->p_AreaMinor
				);
			Call(cudaThreadSynchronize(), "cudaTS GetLap Az JLS 2");

			// Now LapAz was set to the Lap of Az_k+1 ....
			// Average with stored Lap:

			kernelAverage <<< numTilesMinor, threadsPerTileMinor >> > (
				p_LapAz, p_temp6
				);
			Call(cudaThreadSynchronize(), "cudaTS Average JLS 2");



			/*
			cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTriTiles, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTriTiles, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTriTiles, cudaMemcpyDeviceToHost);
			sum1 = 0.0; sum2 = 0.0; sum3 = 0.0;
			for (iBlock = 0; iBlock < numTriTiles; iBlock++)
			{
				sum1 += p_temphost1[iBlock];
				sum2 += p_temphost2[iBlock];
				sum3 += p_temphost3[iBlock];
			}
			printf("substep %d sum1--verts %1.15E sum2 %1.15E sum3--verts %1.15E\n",
				iSubstep, sum1, sum2, sum3);

			cudaMemcpy(p_temphost1, p_LapAz, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost2, pX_half->p_AreaMinor, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost3, p_Az, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);*/

		//	FILE * fofofo = fopen("lap1.txt", "w");
		//	for (iMinor = 0; iMinor < NMINOR; iMinor++)
		//		fprintf(fofofo, "%d Az %1.15E Lap %1.15E Area %1.15E \n", iMinor, p_temphost3[iMinor], p_temphost1[iMinor], p_temphost2[iMinor]);
		//	fclose(fofofo);
		//	printf("done file\n");
		//	getch(); getch();

			// Leaving Iz_prescribed and reverse_Jz the same:

			// Think I'm right all that has changed is LapAz so do we really have to go through whole thing again? :

			//	this->Accelerate2018(hsub, pX_half, pDestMesh, evaltime + 0.5*hsub, false); // Lap Az now given.
			/*
			kernelPreUpdateAzdot(
				hsub,
				pX_half->p_info,
				pX_half->p_AAdot,
				p_ROCAzdotduetoAdvection,
				p_LapAz,
				pX_half->p_n_minor,
				(iSubstep == 0) ? this->p_vie : pX_target->p_vie,
				p_temp1
				);
			Call(cudaThreadSynchronize(), "cudaTS Pre update --");

		// * So now we have to go and sort out how to apply this TotalEffect alongside 1/2 I_k+1

		*/
		// THAT APPROACH WILL NOT WORK, AS WE NEED AZDOT_K VALUE WITHIN THE POPOHMS ROUTINE
		// OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPSSSSSSSSSSS

			kernelEstimateCurrent << <numTilesMinor, threadsPerTileMinor >> > (
				pX_half->p_info,
				pX_half->p_n_minor,
				(iSubstep == 0) ? this->p_vie : pX_target->p_vie,
				pX_half->p_AreaMinor,
				p_temp1); // Iz_k
			Call(cudaThreadSynchronize(), "cudaTS Estimate Iz_k");

			// Now for efficiency we will try and find ways to move some vars into pre routine
			// thus splitting it up and passing info through updateable Ohm's Law somehow.

			kernelPopulateOhmsLaw << <numTilesMinor, threadsPerTileMinor >> > (
				hsub,// ROCAzdotduetoAdvection, 
				pX_half->p_info,
				p_MAR_neut, p_MAR_ion, p_MAR_elec,
				pX_half->p_B,
				p_LapAz,
				p_GradAz, // THIS WE OUGHT TO TWEEN AT LEAST
				p_GradTe,

				pX_half->p_n_minor,

				p_one_over_n2,

				// problem ... we set it to Iz for a different n but now use this for Azdot's J_k contribution
				// as it stands it's hard to debug because we aren't expecting it to be the same.

				// Might it be wrong in other ways for the same reason?
				// Meanwhile we kind of need the n to be tied up with the Azdot advance.

				// And it's a problem that we used different n to update Az_k+1 I think,
				// because we assumed the n we used when we wrote the v update that
				// should be truly simultaneous with Azdot's update.

				// Workings assumed n was a constant. Which it very nearly is but maybe
				// that's not good enough.
				// We should, if we treated n as constant in the formulas, treat it as
				// constant throughout. 
				// If we are going to use same n throughout for v then we have assumed
				// we use the same for Azdot. That is already the case here in this advance.
				// Whatever n we use for setting Iz matters less --
				// but we do want Iz- to match Iz+. So we have to track what Iz+_k we
				// are using to contribute to Azdot.
				// Even if this doesn't fix anything at least it makes some things more debuggable.

				pX_half->p_T_minor,

				(iSubstep == 0) ? this->p_vie : pX_target->p_vie,
				(iSubstep == 0) ? this->p_v_n : pX_target->p_v_n,
				(iSubstep == 0) ? this->p_AAdot : pX_target->p_AAdot, //	src
				pX_half->p_AreaMinor, // pop'd? interp?
				p_ROCAzdotduetoAdvection,

				p_vn0,
				p_v0,
				p_OhmsCoeffs,
				pX_half->p_AAdot, // intermediate value ............................
									// .....................................................
				p_Iz0_summands,
				p_sigma_Izz,
				 
				p_denom_i,
				p_denom_e, p_coeff_of_vez_upon_viz, p_beta_ie_z,
				false,
				// (iSubstep == iSubcycles - 1) ? true : false, // For Iz only...
				false, // Easier to just stop doing this -
				// If we do it then we have to measure current again to get
				// correct input into reverse current.
				pX_target->p_n_minor);

			// #########################################################################################################
			// DEBUG: pass graphing parameters through these.
			// #########################################################################################################
			cudaMemcpy(p_temphost1, p_denom_i, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost2, p_denom_e, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);

			// But if we use it to set Iz
			// We do get an imbalance of + & -.
			// That would be because the Ez we picked does not deliver Iz_presc
			// according to n_used at the final time.
			// What to do about that?
			// Simple answer is we used n_half to apply Jz_k and so should be using it
			// to apply Jz_k+1 to Azdot, as we are doing. Therefore we just want
			// the negative (+) contrib to Azdot to match what + we actually get.
			// Hard to believe that this would ever have a big impact but you never know.
			
			// Might as well recalculate Ez_strength again :
			// Iz already set for t+hsub.
			cudaMemcpy(p_Iz0_summands_host, p_Iz0_summands, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_summands_host, p_sigma_Izz, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			Iz0 = 0.0;
			Sigma_Izz = 0.0;
			Iz_k = 0.0;
			for (iBlock = 0; iBlock < numTilesMinor; iBlock++)
			{
				Iz0 += p_Iz0_summands_host[iBlock];
				Sigma_Izz += p_summands_host[iBlock];
				Iz_k += p_temphost1[iBlock];
			//	printf("Block %d Iz0 %1.10E Sigma_Izz %1.10E  +  ", iBlock, p_Iz0_summands_host[iBlock], p_summands_host[iBlock]);
			}
			EzStrength_ = (Iz_prescribed_endtime - Iz0) / Sigma_Izz;
			Set_f64_constant(Ez_strength, EzStrength_);

			// evaltime + 0.5*hsub used for setting EzStrength://
			// Iz_prescribed_endtime = GetIzPrescribed(evaltime + 0.5*hsub); // APPLIED AT END TIME: we are determining
	//		Iz_prescribed_starttime = GetIzPrescribed(evaltime - 0.5*hsub); // APPLIED AT END TIME: we are determining
																			   // Jz, hence Iz at k+hsub initially.
			neg_Iz_per_triangle = -0.5*(Iz_prescribed_endtime + Iz_k) / (f64)numReverseJzTriangles;
			Set_f64_constant(negative_Iz_per_triangle, neg_Iz_per_triangle);
			// Electrons travel from cathode to anode so Jz is down in filament,
			// up around anode.

	//		printf("Iz_k %1.14E *h*c*4pi %1.14E num %d %d %d neg_Iz %1.14E \n",
	//			Iz_k, hsub*c_*FOURPI_*Iz_k, numReverseJzTriangles,
	//			numStartZCurrentTriangles__, numEndZCurrentTriangles__, neg_Iz_per_triangle);
			
			// DEBUG:
		//	if (iSubstep != 0)
		//		cudaMemcpy(this->p_vie, pX_target->p_vie, sizeof(v4)*NMINOR, cudaMemcpyDeviceToDevice);
		// DEBUG:
			if (EzStrength_ != EzStrength_) {
				printf("EzStrength_ %1.10E Iz_prescribed %1.10E Iz0 %1.10E sigma_Izz %1.10E \n",
					EzStrength_, Iz_prescribed_endtime, Iz0, sigma_Izz);
				while (1) getch();
			}

			kernelCalculateVelocityAndAzdot << <numTilesMinor, threadsPerTileMinor >> > (
				hsub,
				pX_half->p_info,
				p_vn0,
				p_v0,
				p_OhmsCoeffs,
				pX_half->p_AAdot,
				//(iSubstep == iSubcycles - 1) ? pX_target->p_n_minor:pX_half->p_n_minor,
				pX_half->p_n_minor, // NOT OKAY FOR IT TO NOT BE SAME n AS USED THROUGHOUT BY OHMS LAW
				pX_half->p_AreaMinor,

				pX_target->p_AAdot,
				pX_target->p_vie,
				pX_target->p_v_n	
		//		p_bool
				);
			Call(cudaThreadSynchronize(), "cudaTS kernelUpdate_v ");

			//bool bAlert = false;
			//cudaMemcpy(p_boolhost, p_bool, sizeof(bool)*numTilesMinor, cudaMemcpyDeviceToHost);
			//for (iTile = 0; iTile < numTilesMinor; iTile++)
			//{
			//	bAlert = bAlert || p_boolhost[iTile];
			//}
			//if (bAlert) {
			//	while (1) getch();
			//}



			// DEBUG:
			/*kernelEstimateCurrent << <numTilesMinor, threadsPerTileMinor >> > (
				pX_half->p_info,
				pX_half->p_n_minor,
				pX_target->p_vie,
				pX_half->p_AreaMinor,
				p_temp1); // Iz_k+1
			Call(cudaThreadSynchronize(), "cudaTS Estimate Iz_k+1");
			cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			Iz_k = 0.0;
			for (iBlock = 0; iBlock < numTilesMinor; iBlock++)
			{
				Iz_k += p_temphost1[iBlock];
			}
			printf("Iz_k+1 attained %1.14E prescribed %1.14E \n", Iz_k, Iz_prescribed_endtime);

			// On last step we see a difference: sum Azdot inexplicably nonzero
			// So something is still going wrong.

			
			Estimate_Effect_on_Integral_Azdot_from_Jz_and_LapAz << <numTilesMinor, threadsPerTileMinor >> > (
				hsub,
				pX_half->p_info,
				pX_half->p_n_minor,
				pX_half->p_n_minor, // used in UpdateVelocity
				this->p_vie, // save off the old value to (this) first
				pX_target->p_vie,
				pX_half->p_AreaMinor,
				p_LapAz, 

				pX_target->p_AAdot, // we'll be able to look at row to see what came just before...

				p_temp1, // +ve Jz
				p_temp2, // -ve Jz
				p_temp3, // LapAz*AreaMinor
				p_temp4, 
				p_temp5,
				p_temp6
				);
			Call(cudaThreadSynchronize(), "cudaTS Estimate Effect");

			cudaMemcpy(p_temphost1, p_temp1, numTilesMinor * sizeof(f64), cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost2, p_temp2, numTilesMinor * sizeof(f64), cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost3, p_temp3, numTilesMinor * sizeof(f64), cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost4, p_temp4, numTilesMinor * sizeof(f64), cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost5, p_temp5, numTilesMinor * sizeof(f64), cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost6, p_temp6, numTilesMinor * sizeof(f64), cudaMemcpyDeviceToHost);

			sum_plus = 0.0;   sum_minus = 0.0; 	sum_Lap = 0.0;
			abs_Lap = 0.0;   sum_Azdot = 0.0; abs_Azdot = 0.0;

			for (i = 0; i < numTilesMinor; i++)
			{
				sum_plus += p_temphost1[i];
				sum_minus += p_temphost2[i];
				sum_Lap += p_temphost3[i];
				abs_Lap += p_temphost4[i];
				sum_Azdot += p_temphost5[i];
				abs_Azdot += p_temphost6[i];
			}
			printf("\n\n");
			
			//printf("sum_plus %1.9E net %1.9E sum_Lap %1.9E sum_diff %1.9E \n",
			//	sum_plus, sum_minus + sum_plus, sum_Lap, sum_diff);

			fprintf(fp_dbg, "substep %d half-step: sum_plus %1.15E sum_minus %1.15E sum_Lap %1.15E abs_Lap %1.15E sum_Azdot %1.15E abs_Azdot %1.15E \n",
				iSubstep, sum_plus, sum_minus, sum_Lap, abs_Lap, sum_Azdot, abs_Azdot);

			printf("substep %d half - step: sum_plus %1.15E sum_minus %1.15E sum_Lap %1.15E abs_Lap %1.15E sum_Azdot %1.15E abs_Azdot %1.15E \n",
				iSubstep, sum_plus, sum_minus, sum_Lap, abs_Lap, sum_Azdot, abs_Azdot);
		*/
			/////////////////////////////////////////////////////////////////////////////////////
			
			evaltime += 0.5*hsub;
			// Why we do not pass it back and forth? Can't remember.
		}; // next substep
		kernelPushAzInto_dest << <numTilesMinor, threadsPerTileMinor >> >(
			pX_target->p_AAdot,
			p_Az);
		Call(cudaThreadSynchronize(), "cudaTS PushAzIntoDest");

	//	cudaMemcpy(AAdotarray, pX_target->p_AAdot + 2180, sizeof(AAdot) * 40, cudaMemcpyDeviceToHost);
	//	for (i = 0; i < 40; i++)
	//	{
	//		Azdotarrays[iSubstep][i] = AAdotarray[i].Azdot;
	//		Azarrays[iSubstep][i] = AAdotarray[i].Az;
	//	}

	//	FILE * gerald = fopen("geraldineNODEBUG.txt", "w");
	//	for (i = 0; i < 40; i++)
	//	{
	//		fprintf(gerald, "%d Az ", i + 2180);
	//		for (int j = 0; j < iSubcycles + 1; j++)
	//			fprintf(gerald, "%1.15E ", Azarrays[j][i]);
	//		fprintf(gerald, "  Azdot  ");
	//		for (int j = 0; j < iSubcycles + 1; j++)
	//			fprintf(gerald, "%1.15E ", Azdotarrays[j][i]);
	//		fprintf(gerald, "\n");
	//	}
	//	fclose(gerald);

		// more advanced implicit could be possible and effective.

		// It is almost certain that splitting up BJLS into a few goes in each set of subcycles would be more effective than being a different set all BJLS.
		// This should be experimented with, once it matches CPU output.
	} else {

		// Leapfrog:

		kernelPopulateArrayAz << <numTilesMinor, threadsPerTileMinor >> >(
			0.5*hsub,
			this->p_AAdot,
			p_ROCAzduetoAdvection, 
			p_Az
			);   // This is where we create the f64 array of Az from a short step using Adot_k
		// We can now see that having AAdot in one object was counterproductive.
		Call(cudaThreadSynchronize(), "cudaTS PopulateArrayAz");

		kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> >(
			this->p_info,
			this->p_tri_neigh_index,
			p_Az
			);
		Call(cudaThreadSynchronize(), "cudaTS kernelResetFrillsAz 1");
		// Create_A_from_advance(0.5*hsub, ROCAzduetoAdvection, Az_array); // from *this

		for (iSubstep = 0; iSubstep < iSubcycles; iSubstep++)
		{
			printf("####### SUBSTEP %d / %d #######\n\n", iSubstep, iSubcycles);

			evaltime += 0.5*hsub;
			kernelInterpolateVarsAndPositions << <numTilesMinor, threadsPerTileMinor >> >(
				(evaltime - starttime) / Timestep,
				this->p_info,
				pX_target->p_info,
				this->p_n_minor,
				pX_target->p_n_minor,
				this->p_T_minor,
				pX_target->p_T_minor,
	//			this->p_B,
	//			pX_target->p_B,
				pX_half->p_info,
				pX_half->p_n_minor,
				pX_half->p_T_minor
	//			pX_half->p_B
				);
			Call(cudaThreadSynchronize(), "cudaTS InterpolateVars");
			// let n,T,x be interpolated on to pX_half. B remains what we populated there.
			// Recalculate areas, or tween them, would make good sense as well.

			// It might be nice to tween the thermal pressure grad(nT) as well. And logical:
			// InterpolateArrays(AdditionalMomRates_start, AdditionalMomRates_end, AdditionalMomRates, (evaltime - starttime) / Timestep);
			// Have a look how AMR is created.
			kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> >(
				pX_half->p_info,
				p_Az,
				pX_half->p_izTri_vert,
				pX_half->p_izNeigh_TriMinor,
				pX_half->p_szPBCtri_vert,
				pX_half->p_szPBC_triminor,
				p_LapAz,
			//	p_temp1, p_temp2, p_temp3,
				pX_half->p_AreaMinor
				);
			Call(cudaThreadSynchronize(), "cudaTS GetLap Az Leapfrog 1");
			
			// evaltime + 0.5*hsub used for setting EzStrength://
			Iz_prescribed_endtime = GetIzPrescribed(evaltime + 0.5*hsub);
	//		Iz_prescribed_starttime = GetIzPrescribed(evaltime - 0.5*hsub);
			
			// On the first step we use "this" as src, otherwise pX_targ to pX_targ
			// Simple plan:
			// Pop Ohms just populate's Ohms and advances Azdot to an intermediate state

			kernelEstimateCurrent << <numTilesMinor, threadsPerTileMinor >> >(
				pX_half->p_info,
				pX_half->p_n_minor,
				(iSubstep == 0) ? this->p_vie : pX_target->p_vie,
				pX_half->p_AreaMinor,
				p_temp1); // Iz_k
			Call(cudaThreadSynchronize(), "cudaTS Estimate Iz_k");

			cudaMemcpy(&tempf64, &(p_MAR_elec[11653 + BEGINNING_OF_CENTRAL].y), sizeof(f64), cudaMemcpyDeviceToHost);
			printf("MAR.y 11653 %1.8E", tempf64);
			kernelPopulateOhmsLaw << <numTilesMinor, threadsPerTileMinor >> >(
				hsub,// ROCAzdotduetoAdvection, 
				pX_half->p_info,
				p_MAR_neut, p_MAR_ion, p_MAR_elec,
				pX_half->p_B,
				p_LapAz,
				p_GradAz,
				p_GradTe,

				pX_half->p_n_minor,
				p_one_over_n2,

				pX_half->p_T_minor, // minor : is it populated?
				(iSubstep == 0)?this->p_vie:pX_target->p_vie,
				(iSubstep == 0)?this->p_v_n:pX_target->p_v_n,
				(iSubstep == 0)?this->p_AAdot:pX_target->p_AAdot, 
				pX_half->p_AreaMinor, // pop'd????????? interp?
				p_ROCAzdotduetoAdvection,

				p_vn0,
				p_v0,
				p_OhmsCoeffs,
				pX_half->p_AAdot, // intermediate value ............................
				 
				p_Iz0_summands,
				p_sigma_Izz,
				p_denom_i,
				p_denom_e, p_coeff_of_vez_upon_viz, p_beta_ie_z,
				false,
				false, // (iSubstep == iSubcycles-1),

				pX_target->p_n_minor
				); // bFeint
			Call(cudaThreadSynchronize(), "cudaTS kernelPopulateOhmsLaw ");
				 		
			cudaMemcpy(p_Iz0_summands_host, p_Iz0_summands, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_summands_host, p_sigma_Izz, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			Iz0 = 0.0; sigma_Izz = 0.0; Iz_k = 0.0;
			for (int iBlock = 0; iBlock < numTilesMinor; iBlock++)
			{
				Iz0 += p_Iz0_summands_host[iBlock];
				sigma_Izz += p_summands_host[iBlock];
				Iz_k += p_temphost1[iBlock];
			//	printf("Block %d Iz0 %1.10E Sigma_Izz %1.10E  |  ", iBlock, p_Iz0_summands_host[iBlock], p_summands_host[iBlock]);
			}
			EzStrength_ = (Iz_prescribed_endtime - Iz0) / sigma_Izz;

			// DEBUG:
			if (EzStrength_ != EzStrength_) {
				printf("EzStrength_ %1.10E Iz_prescribed %1.10E Iz0 %1.10E sigma_Izz %1.10E \n",
					EzStrength_, Iz_prescribed_endtime, Iz0, sigma_Izz);
				while (1) getch();
			}

			Set_f64_constant(Ez_strength, EzStrength_);
			f64 neg_Iz_per_triangle = -0.5*(Iz_prescribed_endtime + Iz_k) / (f64)numReverseJzTriangles;
			Set_f64_constant(negative_Iz_per_triangle, neg_Iz_per_triangle);

			kernelCalculateVelocityAndAzdot << <numTilesMinor, threadsPerTileMinor >> >(
				hsub,
				pX_half->p_info,
				p_vn0,
				p_v0,
				p_OhmsCoeffs,
				pX_half->p_AAdot,
				pX_half->p_n_minor,
				pX_half->p_AreaMinor,

				pX_target->p_AAdot,
				pX_target->p_vie,
				pX_target->p_v_n
				//p_bool
				);
			Call(cudaThreadSynchronize(), "cudaTS kernelUpdate_v ");
		/*	
			bool bAlert = false;
			cudaMemcpy(p_boolhost, p_bool, sizeof(bool)*numTilesMinor, cudaMemcpyDeviceToHost);
			for (iTile = 0; iTile < numTilesMinor; iTile++)
			{
				bAlert = bAlert || p_boolhost[iTile];
			}
			if (bAlert) {
				while (1) getch();
			}
			*/

			kernelUpdateAz << <numTilesMinor, threadsPerTileMinor >> >(
				(iSubstep == iSubcycles-1)?0.5*hsub:hsub,
				pX_target->p_AAdot,
				p_ROCAzduetoAdvection,
				p_Az);
			Call(cudaThreadSynchronize(), "cudaTS kernelUpdateAz ");

			kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> >(
				this->p_info,
				this->p_tri_neigh_index,
				p_Az
				);
			Call(cudaThreadSynchronize(), "cudaTS kernelResetFrillsAz 10");
			
			evaltime += 0.5*hsub;
		};

		kernelPushAzInto_dest <<<numTilesMinor, threadsPerTileMinor>>>(
			pX_target->p_AAdot,
			p_Az);
		Call(cudaThreadSynchronize(), "cudaTS PushAzIntoDest");

	}; // whether Backward or Leapfrog





	SetConsoleTextAttribute(hConsole, 14);
	cudaMemcpy(&tempf64, &(pX_target->p_vie[42940].vez), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\npX_target->vez[42940] %1.14E\n\n", tempf64);
	cudaMemcpy(&tempf64, &(p_MAR_elec[42940].z), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\np_MAR_elec[42940].z %1.14E\n\n", tempf64);
	cudaMemcpy(&temp_vec2, &(pX_target->p_info[42940].pos), sizeof(f64_vec2), cudaMemcpyDeviceToHost);
	printf("\nposition %1.14E %1.14E\n\n", temp_vec2.x, temp_vec2.y);
	cudaMemcpy(&temp_vec2, &(pX_target->p_info[95115].pos), sizeof(f64_vec2), cudaMemcpyDeviceToHost);
	printf("\nposition 23187 %1.14E %1.14E\n\n", temp_vec2.x, temp_vec2.y);

	SetConsoleTextAttribute(hConsole, 15);
	printf("evaltime %1.5E \n", evaltime);
	printf("-----------------\n");

	//this->AntiAdvectAzAndAdvance(h, pX_half, IntegratedGradAz, pDestMesh); // Might as well subsume this in Accelerate, really
	//pX_half->AntiAdvectAzAndAdvance(h*0.5, pDestMesh, GradAz, pDestMesh);
 
	kernelWrapVertices<<<numTilesMajor,threadsPerTileMajor>>>(
		pX_target->p_info,
		pX_target->p_vie,
		pX_target->p_v_n,
		p_was_vertex_rotated); // B will be recalculated.
	Call(cudaThreadSynchronize(), "cudaTS kernelWrapvertices ");

	// Here put a test of whether any did have to wrap around.

	cudaMemset(p_triPBClistaffected, 0, sizeof(char)*NUMVERTICES);
	kernelWrapTriangles << <numTriTiles, threadsPerTileMinor >> >(
		pX_target->p_info,
		pX_target->p_tri_corner_index,
		p_was_vertex_rotated,
		pX_target->p_vie,
		pX_target->p_v_n,
		p_triPBClistaffected,
		pX_target->p_tri_periodic_corner_flags
		); // B will be recalculated.							   
	Call(cudaThreadSynchronize(), "cudaTS kernelWrapTriangles ");
	 
	kernelReassessTriNeighbourPeriodicFlags_and_populate_PBCIndexneighminor << <numTriTiles, threadsPerTileMinor >> > (
		pX_target->p_info,
		pX_target->p_tri_neigh_index,
		pX_target->p_tri_corner_index,
		p_was_vertex_rotated,
		pX_target->p_tri_periodic_corner_flags,
		pX_target->p_tri_periodic_neigh_flags,
		pX_target->p_szPBC_triminor,
		p_triPBClistaffected
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelReassessTriNeighbourPeriodicFlags_and_populate_PBCIndexneighminor ");
	 
	kernelReset_szPBCtri_vert<<<numTilesMajor,threadsPerTileMajor>>>(
		pX_target->p_info,
		pX_target->p_izTri_vert,
		pX_target->p_izNeigh_vert,
		pX_target->p_szPBCtri_vert,
		pX_target->p_szPBCneigh_vert,
		p_triPBClistaffected);

	Call(cudaThreadSynchronize(), "cudaTS Reset for vert. ");

	printf("Done step %d from time %1.8E length %1.2E\n\n", runs, evaltime, Timestep);

	// For graphing :
	cudaMemcpy(temp_array_host, p_LapAz, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_OhmsCoeffs_host, p_OhmsCoeffs, sizeof(OhmsCoeffs)*NMINOR, cudaMemcpyDeviceToHost);

	cudaMemcpy(p_temphost3, pX_half->p_AreaMinor, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
	f64 integral_lap = 0.0;
	f64 integral_L1 = 0.0;
	for (iMinor = 0; iMinor < NMINOR; iMinor++)
	{
		integral_lap += temp_array_host[iMinor] * p_temphost3[iMinor];
		integral_L1 += fabs(temp_array_host[iMinor] * p_temphost3[iMinor]);
	}
	printf("Integral Lap Az %1.10E integ |Lap| %1.10E \n", integral_lap, integral_L1);
	integral_lap = 0.0;
	integral_L1 = 0.0;
	for (iMinor = BEGINNING_OF_CENTRAL; iMinor < NMINOR; iMinor++)
	{
		integral_lap += temp_array_host[iMinor] * p_temphost3[iMinor];
		integral_L1 += fabs(temp_array_host[iMinor] * p_temphost3[iMinor]);
	}
	printf("Verts Integral Lap Az %1.10E integ |Lap| %1.10E \n", integral_lap, integral_L1);
	integral_lap = 0.0;
	integral_L1 = 0.0;
	for (iMinor = BEGINNING_OF_CENTRAL+11000; iMinor < NMINOR; iMinor++)
	{
		integral_lap += temp_array_host[iMinor] * p_temphost3[iMinor];
		integral_L1 += fabs(temp_array_host[iMinor] * p_temphost3[iMinor]);
	}
	printf("Verts 11000+ Integral Lap Az %1.10E integ |Lap| %1.10E \n", integral_lap, integral_L1);

	// Reveals that vertices only have the positive (in domain?) contributions.
	// Negative is on triangles.
	// What gives? Need to spit .. 
	// Can we intersplice tris and verts on graph? could be good.

	fp = fopen("elapsed.txt", "a");
	SetConsoleTextAttribute(hConsole, 13);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Elapsed time : %f ms ", elapsedTime);
	fprintf(fp, "runs %d Elapsed time : %f ms ", runs, elapsedTime);
	cudaEventElapsedTime(&elapsedTime, start, middle);
	printf("of which pre subcycle was %f ms \n", elapsedTime);
	fprintf(fp, "of which pre subcycle was %f ms \n", elapsedTime);
	SetConsoleTextAttribute(hConsole, 15);

	fclose(fp);
	runs++;
}

void PerformCUDA_Revoke()
{
	CallMAC(cudaFree(p_T_upwind_minor));
	CallMAC(cudaFree(p_bool));
	CallMAC(cudaFree(p_nu_major));
	CallMAC(cudaFree(p_was_vertex_rotated));
	CallMAC(cudaFree(p_triPBClistaffected));
	CallMAC(cudaFree(p_MAR_neut));
	CallMAC(cudaFree(p_MAR_ion));
	CallMAC(cudaFree(p_MAR_elec));
	CallMAC(cudaFree(p_v0));
	CallMAC(cudaFree(p_vn0));
	CallMAC(cudaFree(p_sigma_Izz));
	CallMAC(cudaFree(p_Iz0));
	CallMAC(cudaFree(p_OhmsCoeffs));
	CallMAC(cudaFree(p_sum_eps_deps_by_dbeta));
	CallMAC(cudaFree(p_sum_depsbydbeta_sq));
	CallMAC(cudaFree(p_sum_eps_eps));
	CallMAC(cudaFree(p_sum_eps_deps_by_dbeta_heat));
	CallMAC(cudaFree(p_sum_depsbydbeta_sq_heat));
	CallMAC(cudaFree(p_sum_eps_eps_heat));

	CallMAC(cudaFree(p_Jacobi_n));
	CallMAC(cudaFree(p_Jacobi_i));
	CallMAC(cudaFree(p_Jacobi_e));
	CallMAC(cudaFree(p_epsilon_n));
	CallMAC(cudaFree(p_epsilon_i));
	CallMAC(cudaFree(p_epsilon_e));
	CallMAC(cudaFree(p_coeffself_n));
	CallMAC(cudaFree(p_coeffself_i));
	CallMAC(cudaFree(p_coeffself_e));
	CallMAC(cudaFree(p_d_eps_by_dbeta_n));
	CallMAC(cudaFree(p_d_eps_by_dbeta_i));
	CallMAC(cudaFree(p_d_eps_by_dbeta_e));

	CallMAC(cudaFree(p_Az));
	CallMAC(cudaFree(p_AzNext));
	CallMAC(cudaFree(p_LapAz));
	CallMAC(cudaFree(p_LapAzNext));
	CallMAC(cudaFree(p_LapCoeffself));
	CallMAC(cudaFree(p_LapJacobi));
	CallMAC(cudaFree(p_Jacobi_x));
	CallMAC(cudaFree(p_epsilon));
	
	CallMAC(cudaFree(p_Jacobi_heat));
	CallMAC(cudaFree(p_epsilon_heat));

	CallMAC(cudaFree(p_Azdot0));
	CallMAC(cudaFree(p_gamma));
	CallMAC(cudaFree(p_Integrated_div_v_overall));
	CallMAC(cudaFree(p_Div_v_neut));
	CallMAC(cudaFree(p_Div_v));
	CallMAC(cudaFree(p_Div_v_overall));
	CallMAC(cudaFree(p_ROCAzdotduetoAdvection));
	CallMAC(cudaFree(p_ROCAzduetoAdvection));
	CallMAC(cudaFree(p_GradAz));
	CallMAC(cudaFree(p_GradTe));

	CallMAC(cudaFree(p_one_over_n));
	CallMAC(cudaFree(p_one_over_n2));

	CallMAC(cudaFree(p_kappa_n));
	CallMAC(cudaFree(p_kappa_i));
	CallMAC(cudaFree(p_kappa_e));
	CallMAC(cudaFree(p_nu_i));
	CallMAC(cudaFree(p_nu_e));
	
	CallMAC(cudaFree(p_n_shards));
	CallMAC(cudaFree(p_n_shards_n));
	CallMAC(cudaFree(NT_addition_rates_d));
	CallMAC(cudaFree(NT_addition_tri_d));
	CallMAC(cudaFree(p_denom_i));
	CallMAC(cudaFree(p_denom_e));
	CallMAC(cudaFree(p_temp1));
	CallMAC(cudaFree(p_temp2));
	CallMAC(cudaFree(p_temp3));
	CallMAC(cudaFree(p_temp4));
	CallMAC(cudaFree(p_coeff_of_vez_upon_viz));
	CallMAC(cudaFree(p_longtemp));

	CallMAC(cudaFree(p_InvertedMatrix_i));
	CallMAC(cudaFree(p_InvertedMatrix_e));
	CallMAC(cudaFree(p_MAR_ion2));
	CallMAC(cudaFree(p_MAR_elec2));
	CallMAC(cudaFree(NT_addition_rates_d_temp));
	CallMAC(cudaFree(p_epsilon_xy));
	CallMAC(cudaFree(p_epsilon_iz));
	CallMAC(cudaFree(p_epsilon_ez));
	CallMAC(cudaFree(p_vJacobi_i));
	CallMAC(cudaFree(p_vJacobi_e));
	CallMAC(cudaFree(p_d_eps_by_d_beta_i));
	CallMAC(cudaFree(p_d_eps_by_d_beta_e));
	CallMAC(cudaFree(p_sum_eps_deps_by_dbeta_i));
	CallMAC(cudaFree(p_sum_eps_deps_by_dbeta_e));
	CallMAC(cudaFree(p_sum_depsbydbeta_i_times_i));
	CallMAC(cudaFree(p_sum_depsbydbeta_e_times_e));
	CallMAC(cudaFree(p_sum_depsbydbeta_e_times_i));

	free(p_boolhost);
	free(p_longtemphost);
	free(temp_array_host);
	free(p_temphost1);
	free(p_temphost2);
	free(p_GradTe_host);
	free(p_GradAz_host);
	free(p_B_host);
	free(p_MAR_ion_host);
	free(p_MAR_elec_host);
	free(p_MAR_neut_host);
	free(p_MAR_ion_compare);
	free(p_MAR_elec_compare);
	free(p_MAR_neut_compare);
	free(p_OhmsCoeffs_host);
}

#include "kernel.cu"