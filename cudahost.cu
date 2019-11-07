
// Version 1.0 23/04/19:
// Changing to use upwind T for advection. We could do better in future. Interp gives negative T sometimes.
// Corrected ionisation rate.
       

#pragma once  
  
#define BWD_SUBCYCLE_FREQ  1
#define BWD_STEP_RATIO     1    // divide substeps by this for bwd
#define NUM_BWD_ITERATIONS 4
#define FWD_STEP_FACTOR    2    // multiply substeps by this for fwd
         
// This will be slow but see if it solves it.
              
#define CHOSEN  90721
#define CHOSEN1 1000110301
#define CHOSEN2 1000110497 
#define VERTCHOSEN 16993

#include <math.h>
#include <time.h>
#include <stdio.h> 
     
#include "FFxtubes.h"
#include "cuda_struct.h"
#include "flags.h"
#include "kernel.h"
#include "mesh.h"
#include "matrix_real.h"
     
// This is the file for CUDA host code.
#include "simulation.cu"

extern surfacegraph Graph[7];
extern D3D Direct3D;

FILE * fp_trajectory;
FILE * fp_dbg;
bool GlobalSuppressSuccessVerbosity = false;

long VERTS[3] = {15559, 15405, 15251};

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
four_pi_over_c_ReverseJz, RELTHRESH_AZ_d,
FRILL_CENTROID_OUTER_RADIUS_d, FRILL_CENTROID_INNER_RADIUS_d;

__constant__ f64 cross_s_vals_viscosity_ni_d[10], cross_s_vals_viscosity_nn_d[10],
                 cross_T_vals_d[10], cross_s_vals_MT_ni_d[10];
__constant__ f64 beta_n_c[8], beta_i_c[8], beta_e_c[8];

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
__device__ f64 * p_regressor_n, *p_regressor_i, *p_regressor_e, *p_Effect_self_n, *p_Effect_self_i, *p_Effect_self_e,
*d_eps_by_dx_neigh_n, *d_eps_by_dx_neigh_i, *d_eps_by_dx_neigh_e;
__device__ T3 * p_store_T_move1, *p_store_T_move2;

#define p_slot1n p_Ap_n
#define p_slot1i p_Ap_i
#define p_slot1e p_Ap_e
#define p_slot2n p_NnTn
#define p_slot2i p_NTi
#define p_slot2e p_NTe

__device__ f64 * p_Tn, *p_Ti, *p_Te, *p_Ap_n, *p_Ap_i, *p_Ap_e, * p_NnTn, * p_NTi, * p_NTe,
			* stored_Az_move;
__device__ T3  *zero_array;

__device__ f64 * p_Ax;

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
__device__ bool * p_bFailed, *p_boolarray;

__device__ f64 * p_epsilon_heat, *p_Jacobi_heat,
				*p_sum_eps_deps_by_dbeta_heat, *p_sum_depsbydbeta_sq_heat, *p_sum_eps_eps_heat;
f64  *p_sum_eps_deps_by_dbeta_host_heat, *p_sum_depsbydbeta_sq_host_heat, *p_sum_eps_eps_host_heat;

__device__ f64_vec4 * p_d_eps_by_dbetaJ_n_x4, *p_d_eps_by_dbetaJ_i_x4, *p_d_eps_by_dbetaJ_e_x4,
*p_d_eps_by_dbetaR_n_x4, *p_d_eps_by_dbetaR_i_x4, *p_d_eps_by_dbetaR_e_x4,
*p_sum_eps_deps_by_dbeta_J_x4, *p_sum_eps_deps_by_dbeta_R_x4;
__device__ f64 * p_sum_depsbydbeta_8x8;
f64 * p_sum_depsbydbeta_8x8_host;
f64_vec4 *p_sum_eps_deps_by_dbeta_J_x4_host, *p_sum_eps_deps_by_dbeta_R_x4_host;

__device__ species3 *p_nu_major;
__device__ f64_vec2 * p_GradAz, *p_GradTe;
__device__ ShardModel *p_n_shards, *p_n_shards_n;
__device__ NTrates *NT_addition_rates_d, *NT_addition_tri_d;
long numReverseJzTriangles;
__device__ f64 *p_sum_eps_deps_by_dbeta, *p_sum_depsbydbeta_sq, *p_sum_eps_eps;
f64  *p_sum_eps_deps_by_dbeta_host, *p_sum_depsbydbeta_sq_host, *p_sum_eps_eps_host;
__device__ char * p_was_vertex_rotated, *p_triPBClistaffected;
__device__ T3 * p_T_upwind_minor_and_putative_T;

__device__ f64 * p_d_eps_by_dbeta_n, *p_d_eps_by_dbeta_i, *p_d_eps_by_dbeta_e,
*p_d_eps_by_dbetaR_n, *p_d_eps_by_dbetaR_i, *p_d_eps_by_dbetaR_e,
*p_Jacobi_n, *p_Jacobi_i, *p_Jacobi_e, *p_epsilon_n, *p_epsilon_i, *p_epsilon_e,
*p_coeffself_n, *p_coeffself_i, *p_coeffself_e; // Are these fixed or changing with each iteration?

__device__ f64_tens3 * p_InvertedMatrix_i, *p_InvertedMatrix_e;
__device__ f64_vec3 * p_MAR_ion2, *p_MAR_elec2, * p_vJacobi_i, * p_vJacobi_e,
	* p_d_eps_by_d_beta_i, *p_d_eps_by_d_beta_e, *p_eps_against_deps;
f64_vec3 * p_sum_vec_host;
__device__ NTrates * NT_addition_rates_d_temp;
__device__ f64_vec2 * p_epsilon_xy;
__device__ f64 * p_epsilon_iz, *p_epsilon_ez,
*p_sum_eps_deps_by_dbeta_i, *p_sum_eps_deps_by_dbeta_e, *p_sum_depsbydbeta_i_times_i,
*p_sum_depsbydbeta_e_times_e, *p_sum_depsbydbeta_e_times_i;
f64 * p_sum_eps_deps_by_dbeta_i_host, * p_sum_eps_deps_by_dbeta_e_host, * p_sum_depsbydbeta_i_times_i_host,
*p_sum_depsbydbeta_e_times_e_host, *p_sum_depsbydbeta_e_times_i_host;

__device__ f64 *p_sum_eps_deps_by_dbeta_J, *p_sum_eps_deps_by_dbeta_R, *p_sum_depsbydbeta_J_times_J,
*p_sum_depsbydbeta_R_times_R, *p_sum_depsbydbeta_J_times_R;
f64 * p_sum_eps_deps_by_dbeta_J_host, *p_sum_eps_deps_by_dbeta_R_host, *p_sum_depsbydbeta_J_times_J_host,
*p_sum_depsbydbeta_R_times_R_host, *p_sum_depsbydbeta_J_times_R_host;

TriMesh * pTriMesh;

f64 * temp_array_host;
f64 tempf64;
FILE * fp_traj;
 
//f64 Tri_n_n_lists[NMINOR][6],Tri_n_lists[NMINOR][6];
// Not clear if I ended up using Tri_n_n_lists - but it was a workable way if not.

long * address;
f64 * f64address;
size_t uFree, uTotal;
extern real evaltime;
extern cuSyst cuSyst_host;
//
//
//kernelAccumulateSumOfSquares << < numTilesMajorClever, threadsPerTileMajorClever >> >
//(p_epsilon_n, p_epsilon_i, p_epsilon_e,
//	p_temp1, p_temp2, p_temp3);
//Call(cudaThreadSynchronize(), "cudaTS Sumof squares");
//SS_n = 0.0; SS_i = 0.0; SS_e = 0.0;
//cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
//cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
//cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
//for (iTile = 0; iTile < numTilesMajorClever; iTile++)
//{
//	SS_n += p_temphost1[iTile];
//	SS_i += p_temphost2[iTile];
//	SS_e += p_temphost3[iTile];
//}

/*
void RunBackwardForHeat_BiCGstab(T3 * p_T_k, T3 * p_T, f64 hsub, cuSyst * pX_use)
{
#define UPLIFT_THRESHOLD 0.33

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

	f64 beta_eJ, beta_iJ, beta_nJ, beta_eR, beta_iR, beta_nR;
	long iTile;

	Matrix_real sum_ROC_products, sum_products_i, sum_products_e;
	f64 sum_eps_deps_by_dbeta_vector[8];
	f64 beta[8];

	f64 sum_eps_deps_by_dbeta_J, sum_eps_deps_by_dbeta_R,
		sum_depsbydbeta_J_times_J, sum_depsbydbeta_R_times_R, sum_depsbydbeta_J_times_R,
		sum_eps_eps;
	f64 dot2_n, dot2_i, dot2_e;

	Vertex * pVertex;
	long iVertex;
	plasma_data * pdata;

	printf("\nBiCGStab for heat: ");
	//long iMinor;
	f64 L2eps;// _elec, L2eps_neut, L2eps_ion;
	bool bFailedTest;
	Triangle * pTri;
	f64_vec4 tempf64vec4;
	f64 tempf64;
	f64 SS_n, SS_i, SS_e, oldSS_n, oldSS_i, oldSS_e, ratio_n, ratio_i, ratio_e,
		alpha_n, alpha_i, alpha_e, dot_n, dot_i, dot_e;

	// seed: just set T to T_k.
	//kernelUnpacktoNT << < numTilesMajorClever, threadsPerTileMajorClever >> >
	//	(p_NnTn, p_NTi, p_NTe, p_T_k, 
	//		pX_use->p_AreaMajor,
	//		pX_use->p_n_major);
	//Call(cudaThreadSynchronize(), "cudaTS UnpacktoNT");

	kernelUnpack << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_Tn, p_Ti, p_Te, p_T_k);
	Call(cudaThreadSynchronize(), "cudaTS UnpacktoNT");

	// 1. Compute epsilon:

	// epsilon = T_k+1 - T_k - (h/N)rates.NT;
	// epsilon = b - A T_k+1
	// Their Matrix A = -identity + (h/N) ROC NT due to heat flux
	// Their b = - T_k

	// Compute heat flux given p_T	
	cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
	//
	//	kernelCreateTfromNTbydividing << < numTilesMajorClever, threadsPerTileMajorClever >> >
	//		(p_Tn, p_Ti, p_Te, p_NnTn, p_NTi, p_NTe,
	//			pX_use->p_AreaMajor,
	//			pX_use->p_n_major);
	//	Call(cudaThreadSynchronize(), "cudaTS CreateTfromNT");

	kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_scalarT << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(
			pX_use->p_info,
			pX_use->p_izNeigh_vert,
			pX_use->p_szPBCneigh_vert,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_cc,
			pX_use->p_n_major,
			p_Tn, p_Ti, p_Te,  // using vert indices

			pX_use->p_T_minor + BEGINNING_OF_CENTRAL, // not used!
			pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
			p_kappa_n,
			p_kappa_i, p_kappa_e,
			p_nu_i, p_nu_e,
			NT_addition_rates_d_temp,
			pX_use->p_AreaMajor);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

	// Let's think about this carefully. If we could do it with the same data loaded each vertex for neigh list it would actually save time.
	// We rely on the fact that loading the vertex list data is contiguous fetch.
	// And we do store it for 3 species - so let's stick with all in 1.

	kernelCreateEpsilonHeat << <numTilesMajorClever, threadsPerTileMajorClever >> >
		(hsub,
			pX_use->p_info + BEGINNING_OF_CENTRAL,
			p_epsilon_n, p_epsilon_i, p_epsilon_e,
			p_Tn, p_Ti, p_Te,
			p_T_k,					// p_T_k was not swapped so that might be why different result. All this is tough.
			pX_use->p_AreaMajor,
			pX_use->p_n_major,
			NT_addition_rates_d_temp // it's especially silly having a whole struct of 5 instead of 3 here.
			);
	Call(cudaThreadSynchronize(), "cudaTS CreateEpsilon");

	// Copy to r0hat
	cudaMemcpy(p_temp4, p_epsilon_n, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_temp5, p_epsilon_i, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_temp6, p_epsilon_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);

	rho_prev_n = 1.0; alpha_n = 1.0; omega_n = 1.0;
	rho_prev_i = 1.0; alpha_i = 1.0; omega_i = 1.0;
	rho_prev_e = 1.0; alpha_e = 1.0; omega_e = 1.0;

	cudaMemset(p_Ap_n_BiCG, 0, sizeof(f64)*NUMVERTICES); // v0
	cudaMemset(p_p_n_BiCG, 0, sizeof(f64)*NUMVERTICES); // p0
	cudaMemset(p_Ap_i_BiCG, 0, sizeof(f64)*NUMVERTICES); // v0
	cudaMemset(p_p_i_BiCG, 0, sizeof(f64)*NUMVERTICES); // p0
	cudaMemset(p_Ap_e_BiCG, 0, sizeof(f64)*NUMVERTICES); // v0
	cudaMemset(p_p_e_BiCG, 0, sizeof(f64)*NUMVERTICES); // p0

	bool bContinue = true;
	iIteration = 1;
	do {
		//rho_n = dotproduct(p_temp4, p_epsilon_n); // r_array is for iIter-1
		//rho_i = dotproduct(p_temp5, p_epsilon_i); // r_array is for iIter-1
		//rho_e = dotproduct(p_temp6, p_epsilon_e); // r_array is for iIter-1
		dotproducts(p_temp4, p_epsilon_n,
			p_temp5, p_epsilon_i,
			p_temp6, p_epsilon_e,
			rho_n, rho_i, rho_e);
		beta_n = rho_n*alpha_n / (rho_prev_n*omega_n);
		beta_i = rho_i*alpha_i / (rho_prev_i*omega_i);
		beta_e = rho_e*alpha_e / (rho_prev_e*omega_e);
		UpdateRegressorBiCG(p_p_n_BiCG, p_epsilon_n, beta_n, omega_n, p_Ap_n_BiCG); // omega_i-1
		UpdateRegressorBiCG(p_p_i_BiCG, p_epsilon_i, beta_i, omega_i, p_Ap_i_BiCG); // omega_i-1
		UpdateRegressorBiCG(p_p_e_BiCG, p_epsilon_e, beta_e, omega_e, p_Ap_e_BiCG); // omega_i-1
				// this is now p_i but still v_i-1

		setequaltoAtimes(p_Ap_n_BiCG, p_Ap_i_BiCG, p_Ap_e_BiCG,
			p_p_n_BiCG, p_p_i_BiCG, p_p_e_BiCG,
			p_T_k, hsub, pX_use); // can we fill it in?
		dotproducts(p_temp4, p_Ap_n_BiCG,
			p_temp5, p_Ap_i_BiCG,
			p_temp6, p_Ap_e_BiCG,
			dot_n, dot_i, dot_e);
		alpha_n = rho_n / dot_n;
		alpha_i = rho_i / dot_i;
		alpha_e = rho_e / dot_e;

		// regressor=h, x is at i-1 
		LinearCombo(p_regressor_n, p_Tn, alpha_n, p_p_n_BiCG);
		LinearCombo(p_regressor_i, p_Ti, alpha_i, p_p_i_BiCG);
		LinearCombo(p_regressor_e, p_Te, alpha_e, p_p_e_BiCG);
		LinearCombo(p_s_n_BiCG, p_epsilon_n, -alpha_n, p_Ap_n_BiCG); // r_i-1 is in epsilon
		LinearCombo(p_s_i_BiCG, p_epsilon_i, -alpha_i, p_Ap_i_BiCG); // r_i-1 is in epsilon
		LinearCombo(p_s_e_BiCG, p_epsilon_e, -alpha_e, p_Ap_e_BiCG); // r_i-1 is in epsilon
		setequaltoAtimes(p_As_n, p_As_i, p_As_e,
			p_s_n_BiCG, p_s_i_BiCG, p_s_e_BiCG,
			p_T_k, hsub, pX_use);
		dotproducts(p_s_n_BiCG, p_As_n,
			p_s_i_BiCG, p_As_i,
			p_s_e_BiCG, p_As_e,
			omega_n, omega_i, omega_e);
		SumsOfSquares(p_As_n, p_As_i, p_As_e, SS_n, SS_i, SS_e);
		omega_n /= SS_n;
		omega_i /= SS_i;
		omega_e /= SS_e;
		LinearCombo(p_Tn, p_regressor_n, omega, p_s_BiCG);
		LinearCombo(p_epsilon_n, p_s_BiCG, -omega, p_As);
		SumsOfSquares(p_epsilon_n, p_epsilon_i, p_epsilon_e, SS_n, SS_i, SS_e);
		L2eps_n = sqrt(SS_n / NUMVERTICES);
		L2eps_i = sqrt(SS_i / NUMVERTICES);
		L2eps_e = sqrt(SS_e / NUMVERTICES);
		printf("L2eps %1.9E %1.9E %1.9E \n", L2eps_n, L2eps_i, L2eps_e);
		++iIteration;
	} while (iIteration < 200);

	while (1) getch();
}*/

void SolveBackwardAzAdvanceCG(f64 hsub,
	f64 * pAz_k,
	f64 * pAzdot0, f64 * pgamma,
	f64 * p_Solution, f64 * p_LapCoeff_self, 
	cuSyst * pX_use)// AreaMinor will be set in pX_use when we call GetLapMinor
{
	//long iMinor;
	f64 L2eps;// _elec, L2eps_neut, L2eps_ion;
	bool bFailedTest;
	Triangle * pTri;
	f64_vec4 tempf64vec4;
	f64 tempf64;
	f64 RSS, oldRSS, ratio_n, ratio_i, ratio_e,
		alpha_n, alpha_i, alpha_e, dot_n, dot_i, dot_e, RSS_n, RSS_i, RSS_e;
	int iTile;

#define p_regressor p_temp2
#define p_sqrthgamma_Az p_temp1

	cudaMemset(p_temp4, 0, sizeof(f64)*NMINOR);
	cudaMemset(p_temp5, 0, sizeof(f64)*NMINOR);

	// Do this just in case what we were sent was deficient:
	kernelResetFrillsAz << < numTilesMinor, threadsPerTileMinor >> > (
		pX_use->p_info,
		pX_use->p_tri_neigh_index,
		p_Solution);
	Call(cudaThreadSynchronize(), "cudaTS kernelResetFrillsAz");

	kernelDividebyroothgamma << < numTilesMinor, threadsPerTileMinor >> >
		(p_sqrthgamma_Az, p_Solution, hsub, pgamma); // p_Solution is the seed for Az
	Call(cudaThreadSynchronize(), "cudaTS Divideroothgamma");
	kernelResetFrillsAz << < numTilesMinor, threadsPerTileMinor >> > (
		pX_use->p_info,
		pX_use->p_tri_neigh_index,
		p_sqrthgamma_Az);
	Call(cudaThreadSynchronize(), "cudaTS kernelResetFrillsAz");
	// reset frills because we will not have been able to divide by gamma in frills. But the value is understood.

	// 1. Compute epsilon:
	
	kernelGetLap_minor << < numTriTiles, threadsPerTileMinor >> >
		(pX_use->p_info,
			p_Solution,
			pX_use->p_izTri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBCtri_vert,
			pX_use->p_szPBC_triminor,
			p_LapAz,
			pX_use->p_AreaMinor);
	Call(cudaThreadSynchronize(), "cudaTS GetLapminor");

	kernelCreateEpsilon_Az_CG << < numTilesMinor, threadsPerTileMinor >> >
		(hsub, pX_use->p_info, 
			p_Solution, // in Az units
			pAz_k, pAzdot0, pgamma,
			p_LapAz,
			p_epsilon,
			p_bFailed, false
			); // must include multiply epsilon by h gamma
	Call(cudaThreadSynchronize(), "cudaTS CreateEpsilonAz");
	kernelResetFrillsAz << < numTilesMinor, threadsPerTileMinor >> > (
		pX_use->p_info,
		pX_use->p_tri_neigh_index,
		p_epsilon);
	Call(cudaThreadSynchronize(), "cudaTS kernelResetFrillsAz");

	// Does ResetFrills performed on epsilon make it symmetric or not? :

	// If you are in from a frill, you will create your epsilon using the value in 
	// the frill, albeit this was created from your own value every time.
	
	// epsilon in frill is now a scaled version of epsilon in domain
	// but in domain, we had dependency on other stuff, that is not affected
	// by the frill value.

	// why exactly are we doing this? can we do differently?

	// Symmetry: we must pretend the frill value is just notional, because if it
	// is a real value then epsilon inside depends on it and there is not the
	// equivalent dependence unless we create it.

	// That creates more problems at OUTERMOST: outermost looks at frill value
	// but frill does not look at it.
	// So if we are saying frill value is just representing the inner value, 
	// we then will get unbalanced coefficients between the OUTERMOST and the inner
	// triangle from the frill.


	// . Prob lem !

	// The only answer is probably to make a symmetric equation for epsilon in frill
	// but that won't give us the correct value there!














	// 2. Regressor = epsilon
	cudaMemcpy(p_regressor, p_epsilon, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
	
	// Take RSS :
	kernelAccumulateSumOfSquares1 << < numTilesMinor, threadsPerTileMinor>> >
		(p_epsilon, p_temp3);
	Call(cudaThreadSynchronize(), "cudaTS Sumof squares");
	RSS = 0.0;
	cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	for (int iTile = 0; iTile < numTilesMinor; iTile++)
	{
		RSS += p_temphost3[iTile];
	}
	oldRSS = RSS; 
	long iIteration = 0;
	printf("iIteration %d : L2eps[sqrt(h gamma)Az] %1.9E \n",
		iIteration, sqrt(RSS / (f64)NMINOR));
	
	bool bContinue = true;
	do {
		// remember -Ax is what appears in epsilon.
		
		kernelCreateAzbymultiplying << < numTilesMinor, threadsPerTileMinor >> >
			(p_Solution, p_regressor, hsub, pgamma);
		Call(cudaThreadSynchronize(), "cudaTS CreateAzbydividing");
		kernelResetFrillsAz << < numTilesMinor, threadsPerTileMinor >> > (
			pX_use->p_info,
			pX_use->p_tri_neigh_index,
			p_Solution);
		Call(cudaThreadSynchronize(), "cudaTS kernelResetFrillsAz");

		// now careful : pass Azdot0 = 0, Az_k = 0 and we can get -sqrt(h gamma) Lap (h gamma (regressor))
		kernelGetLap_minor << < numTriTiles, threadsPerTileMinor >> >
			(pX_use->p_info,
				p_Solution,
				pX_use->p_izTri_vert,
				pX_use->p_izNeigh_TriMinor,
				pX_use->p_szPBCtri_vert,
				pX_use->p_szPBC_triminor,
				p_LapAz,
				pX_use->p_AreaMinor);
		Call(cudaThreadSynchronize(), "cudaTS GetLapminor");
		kernelCreateEpsilon_Az_CG << < numTilesMinor, threadsPerTileMinor >> >
			(hsub, pX_use->p_info,
				p_Solution, // in Az units
				p_temp4, p_temp5, // zero, zero
				pgamma, 
				p_LapAz, // lap of Az(regressor)
				p_Ax,
				p_bFailed, false
				); // must include multiply epsilon by h gamma
		Call(cudaThreadSynchronize(), "cudaTS CreateEpsilon Regressor");
		kernelResetFrillsAz << < numTilesMinor, threadsPerTileMinor >> > (
			pX_use->p_info,
			pX_use->p_tri_neigh_index,
			p_Ax);
		Call(cudaThreadSynchronize(), "cudaTS kernelResetFrillsAz");

		// epsilon = b-Ax so even though our own A includes a minus, we now negate
		NegateVector << <numTilesMinor, threadsPerTileMinor >> > (p_Ax);
		Call(cudaThreadSynchronize(), "cudaTS NegateVector");
		
		kernelAccumulateDotProduct << <numTilesMinor, threadsPerTileMinor >> >
			(p_regressor, p_Ax, p_temp3);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDotProduct");
		f64 xdotAx = 0.0; 
		cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMinor; iTile++)
		{
			xdotAx += p_temphost3[iTile];
		};

		f64 alpha = RSS / xdotAx;
		printf("alpha %1.9E \n", alpha);

		VectorAddMultiple1 << < numTilesMinor, threadsPerTileMinor >> > (
			p_sqrthgamma_Az, alpha, p_regressor);
		Call(cudaThreadSynchronize(), "cudaTS AddMultiple");		
		kernelCreateAzbymultiplying << < numTilesMinor, threadsPerTileMinor >> >
			(p_Solution, p_sqrthgamma_Az, hsub, pgamma);
		Call(cudaThreadSynchronize(), "cudaTS CreateAzbymultiplying");
		kernelResetFrillsAz << < numTilesMinor, threadsPerTileMinor >> > (
			pX_use->p_info,
			pX_use->p_tri_neigh_index,
			p_Solution);
		Call(cudaThreadSynchronize(), "cudaTS kernelResetFrillsAz");
		// should be doing nothing...

		// Update Epsilon: or can simply recalculate eps = b - A newx
		// Sometimes would rather update epsilon completely:
		if (iIteration % 1 == 0) {

			kernelGetLap_minor << < numTriTiles, threadsPerTileMinor >> >
				(pX_use->p_info,
					p_Solution,
					pX_use->p_izTri_vert,
					pX_use->p_izNeigh_TriMinor,
					pX_use->p_szPBCtri_vert,
					pX_use->p_szPBC_triminor,
					p_LapAz,
					pX_use->p_AreaMinor);
			Call(cudaThreadSynchronize(), "cudaTS GetLapminor");

			cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMinor);
			kernelCreateEpsilon_Az_CG << < numTilesMinor, threadsPerTileMinor >> >
				(hsub, pX_use->p_info,
					p_Solution, // in Az units
					pAz_k, p_Azdot0, 
					pgamma,
					p_LapAz,
					p_epsilon, 
					p_bFailed, true
					); // must include multiply epsilon by rt h gamma
			// but do test by dividing epsilon by rt h gamma; compare to Solution
			Call(cudaThreadSynchronize(), "cudaTS CreateEpsilon eps");
			kernelResetFrillsAz << < numTilesMinor, threadsPerTileMinor >> > (
				pX_use->p_info,
				pX_use->p_tri_neigh_index,
				p_epsilon);
			Call(cudaThreadSynchronize(), "cudaTS kernelResetFrillsAz");

			// Now test for convergence:
			cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMinor, cudaMemcpyDeviceToHost);
			bContinue = false;
			for (iTile = 0; iTile < numTilesMinor; iTile++)
				if (p_boolhost[iTile]) bContinue = true;
			if (bContinue == true) printf("failed tests\n");

		} else {
			bContinue = true;
			VectorAddMultiple1 << <numTilesMinor, threadsPerTileMinor >> >
				( p_epsilon, -alpha, p_Ax	);
			Call(cudaThreadSynchronize(), "cudaTS AddMultiple eps");
			// it should be true that we setted p_Ax[outer_frill] = 0
			p_boolhost[0] = true;
		};

		// Take RSS :
		kernelAccumulateSumOfSquares1 << < numTilesMinor, threadsPerTileMinor >> >
			(p_epsilon, p_temp3);
		Call(cudaThreadSynchronize(), "cudaTS Sumof squares");
		RSS = 0.0;
		cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMinor; iTile++)
		{
			RSS += p_temphost3[iTile];
		}
		f64 ratio = RSS / oldRSS;
		oldRSS = RSS;
		VectorAddMultiple1 << <numTilesMinor, threadsPerTileMinor >> >
			(p_regressor, ratio, p_epsilon);
		Call(cudaThreadSynchronize(), "cudaTS kernel_AddMultiple");
		kernelResetFrillsAz << < numTilesMinor, threadsPerTileMinor >> > (
			pX_use->p_info,
			pX_use->p_tri_neigh_index,
			p_regressor); // should be unneeded now since epsilon is frilled
		Call(cudaThreadSynchronize(), "cudaTS kernelResetFrillsAz");
		// we affect the frills as well --- in case eps is not frilled, regressor frilled
		
		printf("iIteration %d L2eps[sqrt[h gamma] units] %1.10E \n", iIteration, sqrt(RSS / (real)NMINOR));
		getch();
		  
		iIteration++;

		if (bContinue == false) printf("all tests ok\n");

	} while ((iIteration < 1000) && (bContinue));

	if (iIteration == 1000) { while (1) getch(); };
	
	// Result:
	// It converges super slowly!
	// MUST be something to do with frills surely?
	
	// Can't figure out what.

	// Maybe return to it at a later date.
	// For now try another tack.


	while (1) getch();

#undef p_regressor
#undef p_sqrthgamma_Az
}

 
void SolveBackwardAzAdvanceJ3LS(f64 hsub,
	f64 * pAz_k,
	f64 * p_Azdot0, f64 * p_gamma,
	f64 * p_AzNext, f64 * p_LapCoeffself,
	cuSyst * pX_use)
{  
	f64 L2eps;
	f64 beta[3];
	Tensor3 mat;
	f64 RSS;
	int iTile;
	char buffer[256];
	int iIteration = 0;
	   
	long iMax = 0;
	bool bContinue = true;

	printf("iIteration = %d\n", iIteration);
	// 1. Create regressor:
	// Careful with major vs minor + BEGINNING_OF_CENTRAL:

	GlobalSuppressSuccessVerbosity = true;

	kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
		pX_use->p_info, pX_use->p_tri_neigh_index,
		p_AzNext);
	Call(cudaThreadSynchronize(), "cudaTS ResetFrills further regressor");

	kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
		pX_use->p_info,
		p_AzNext,
		pX_use->p_izTri_vert,
		pX_use->p_izNeigh_TriMinor,
		pX_use->p_szPBCtri_vert,
		pX_use->p_szPBC_triminor,
		p_LapAzNext,
		pX_use->p_AreaMinor // populates it
		);
	Call(cudaThreadSynchronize(), "cudaTS GetLap Az JLS 1");

	cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMinor);
	kernelCreateEpsilonAndJacobi << <numTilesMinor, threadsPerTileMinor >> >
		(hsub, // ?
			pX_use->p_info,
			p_AzNext, pAz_k,
			p_Azdot0, p_gamma,
			p_LapCoeffself, p_LapAzNext,
			p_epsilon, p_Jacobi_x,
			//this->p_AAdot, // to use Az_dot_k for equation);
			p_bFailed
			);
	Call(cudaThreadSynchronize(), "cudaTS CreateEpsAndJacobi 1");
	kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
		pX_use->p_info, pX_use->p_tri_neigh_index, p_Jacobi_x);
	Call(cudaThreadSynchronize(), "cudaTS ResetFrills Jacobi");
	
//	FILE * fpdbg = fopen("J3LS_2.txt", "w");
	L2eps = -1.0;

	do
	{		
		// Now we want to create another regressor, let it be called p_regressor_n
		// Let p_Jacobi_x act as AzNext

		kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
			pX_use->p_info,
			p_Jacobi_x,
			pX_use->p_izTri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBCtri_vert,
			pX_use->p_szPBC_triminor,
			p_LapJacobi,
			//		p_temp1, p_temp2, p_temp3,
			pX_use->p_AreaMinor
			);
		Call(cudaThreadSynchronize(), "cudaTS Get Lap Jacobi 1");

		kernelCreate_further_regressor << <numTilesMinor, threadsPerTileMinor >> >(
			pX_use->p_info,
			hsub,
			p_Jacobi_x,
			p_LapJacobi,
			p_LapCoeffself,
			p_gamma,
			p_regressor_n);
		Call(cudaThreadSynchronize(), "cudaTS Create further regressor");
		kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
			pX_use->p_info, pX_use->p_tri_neigh_index,
			p_regressor_n);
		Call(cudaThreadSynchronize(), "cudaTS ResetFrills further regressor");

		kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
			pX_use->p_info,
			p_regressor_n,
			pX_use->p_izTri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBCtri_vert,
			pX_use->p_szPBC_triminor,
			p_temp4, // Lap of regressor : result
			//		p_temp1, p_temp2, p_temp3,
			pX_use->p_AreaMinor
			);
		Call(cudaThreadSynchronize(), "cudaTS Get Lap Jacobi 2");

		kernelCreate_further_regressor << <numTilesMinor, threadsPerTileMinor >> >(
			pX_use->p_info,
			hsub,
			p_regressor_n,
			p_temp4,
			p_LapCoeffself,
			p_gamma,
			p_regressor_i);
		Call(cudaThreadSynchronize(), "cudaTS Create further regressor");
		kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
			pX_use->p_info, pX_use->p_tri_neigh_index,
			p_regressor_i);
		Call(cudaThreadSynchronize(), "cudaTS ResetFrills further regressor");

		kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
			pX_use->p_info,
			p_regressor_i,
			pX_use->p_izTri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBCtri_vert,
			pX_use->p_szPBC_triminor,
			p_temp5, 
			pX_use->p_AreaMinor
			);
		Call(cudaThreadSynchronize(), "cudaTS Get Lap Jacobi 1");

		// Okay ... now we need to do the routine that creates the matrix deps/dbeta_i deps/dbeta_j
		// and the vector against epsilon

		kernelAccumulateMatrix << <numTilesMinor, threadsPerTileMinor >> >(
			pX_use->p_info,
			hsub,
			p_epsilon,
			p_Jacobi_x,
			p_regressor_n,
			p_regressor_i,
			p_LapJacobi,
			p_temp4,
			p_temp5,
			p_gamma,

			p_temp1, // sum of matrices, in lots of 6
			p_eps_against_deps
			);
		Call(cudaThreadSynchronize(), "cudaTS kernelAccumulateMatrix");
		
		// Now take 6 sums
		f64 sum_mat[6];
		f64_vec3 sumvec(0.0,0.0,0.0);
		memset(sum_mat, 0, sizeof(f64) * 6);
		cudaMemcpy(p_temphost1, p_temp1, sizeof(f64) * 6 * numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_vec_host, p_eps_against_deps, sizeof(f64_vec3)*numTilesMinor, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMinor; iTile++)
		{
			sum_mat[0] += p_temphost1[iTile * 6 + 0];
			sum_mat[1] += p_temphost1[iTile * 6 + 1];
			sum_mat[2] += p_temphost1[iTile * 6 + 2];
			sum_mat[3] += p_temphost1[iTile * 6 + 3];
			sum_mat[4] += p_temphost1[iTile * 6 + 4];
			sum_mat[5] += p_temphost1[iTile * 6 + 5];
			sumvec += p_sum_vec_host[iTile];
		};
		
		// Now populate symmetric matrix
		f64_tens3 mat, mat2;

		mat.xx = sum_mat[0];
		mat.xy = sum_mat[1];
		mat.xz = sum_mat[2];
		mat.yx = mat.xy;
		mat.yy = sum_mat[3];
		mat.yz = sum_mat[4];
		mat.zx = mat.xz;
		mat.zy = mat.yz;
		mat.zz = sum_mat[5];
		// debug:

//		mat.yx = 0.0; mat.yy = 1.0; mat.yz = 0.0;
//		mat.zx = 0.0; mat.zy = 0.0; mat.zz = 1.0;
//		sumvec.y = 0.0;
//		sumvec.z = 0.0;
//
		mat.Inverse(mat2);
		printf(
			" ( %1.6E %1.6E %1.6E ) ( beta0 )   ( %1.6E )\n"
			" ( %1.6E %1.6E %1.6E ) ( beta1 ) = ( %1.6E )\n"
			" ( %1.6E %1.6E %1.6E ) ( beta2 )   ( %1.6E )\n",
			mat.xx, mat.xy, mat.xz, sumvec.x,
			mat.yx, mat.yy, mat.yz, sumvec.y,
			mat.zx, mat.zy, mat.zz, sumvec.z);
		f64_vec3 product = mat2*sumvec;

		beta[0] = -product.x; beta[1] = -product.y; beta[2] = -product.z;

		printf("L2eps %1.9E beta %1.8E %1.8E %1.8E ", L2eps, beta[0], beta[1], beta[2]);

	//	printf("Verify: \n");
	//	f64 z1 = mat.xx*beta[0] + mat.xy*beta[1] + mat.xz*beta[2];
	//	f64 z2 = mat.yx*beta[0] + mat.yy*beta[1] + mat.yz*beta[2];
	//	f64 z3 = mat.zx*beta[0] + mat.zy*beta[1] + mat.zz*beta[2];
	//	printf("z1 %1.14E sumvec.x %1.14E | z2 %1.14E sumvec.y %1.14E | z3 %1.14E sumvec.z %1.14E \n",
	//		z1, sumvec.x, z2, sumvec.y, z3, sumvec.z);

		// Since iterations can be MORE than under Jacobi, something went wrong.
		// Try running with matrix s.t. beta1 = beta2 = 0. Do we get more improvement ever or a different coefficient than Jacobi?
		// If we always do better than Jacobi we _could_ still end up with worse result due to different trajectory but this is unlikely 
		// to be the explanation so we should track it down.
		
	//	cudaMemcpy(p_temphost2, p_Jacobi_x, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);

		kernelAddRegressors << <numTilesMinor, threadsPerTileMinor >> >(
			p_AzNext,
			beta[0], beta[1], beta[2],
			p_Jacobi_x,
			p_regressor_n,
			p_regressor_i
		);
		Call(cudaThreadSynchronize(), "cudaTS kernelAddRegressors");

		// should have no effect since we applied it to regressor(s).

		// Yet it does have an effect. Is this because initial AzNext wasn't frilled? Or because regressors are not?

		// Yes.

		// ok --- spit out Az
		//char buffer[255];
		//sprintf(buffer, "Az%d.txt", iIteration);
		//cudaMemcpy(p_temphost1, p_AzNext, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
		//
		//FILE * jibble = fopen(buffer, "w");
		//for (int i = 0; i < NMINOR; i++)
		//	fprintf(jibble, "%d Az %1.14E Jac_added %1.14E \n",i, p_temphost1[i], p_temphost2[i]);
		//fclose(jibble);

		printf("iIteration = %d\n", iIteration);
		// 1. Create regressor:
		// Careful with major vs minor + BEGINNING_OF_CENTRAL:

		kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
			pX_use->p_info,
			p_AzNext,
			pX_use->p_izTri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBCtri_vert,
			pX_use->p_szPBC_triminor,
			p_LapAzNext,
			pX_use->p_AreaMinor // populates it
			);
		Call(cudaThreadSynchronize(), "cudaTS GetLap Az JLS 1");

		//cudaMemcpy(p_temphost6, p_epsilon, NMINOR * sizeof(f64), cudaMemcpyDeviceToHost);

		cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMinor);
		kernelCreateEpsilonAndJacobi << <numTilesMinor, threadsPerTileMinor >> >
			(hsub, // ?
				pX_use->p_info,
				p_AzNext, pAz_k,
				p_Azdot0, p_gamma,
				p_LapCoeffself, p_LapAzNext,
				p_epsilon, p_Jacobi_x,
				//this->p_AAdot, // to use Az_dot_k for equation);
				p_bFailed
				);
		Call(cudaThreadSynchronize(), "cudaTS CreateEpsAndJacobi 1");
		kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
			pX_use->p_info, pX_use->p_tri_neigh_index, p_Jacobi_x);
		Call(cudaThreadSynchronize(), "cudaTS ResetFrills Jacobi");

		kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> >
			(p_epsilon, p_temp1);
		cudaMemcpy(p_temphost3, p_temp1, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		RSS = 0.0; 
		for (iTile = 0; iTile < numTilesMinor; iTile++) RSS += p_temphost3[iTile];
		L2eps = sqrt(RSS / (f64)NMINOR);
		printf("L2eps: %1.9E \n", L2eps);

	//	fprintf(fpdbg, "L2eps %1.14E beta %1.14E %1.14E %1.14E \n", L2eps, beta[0], beta[1], beta[2]);

		/*
		cudaMemcpy(p_temphost1, p_epsilon, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost4, p_temp6, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);

		f64 eps_predict;
		sprintf(buffer, "eps_vs_eps%d.txt", iIteration);
		FILE * jibble = fopen(buffer, "w");
		for (int i = 0; i < NMINOR; i++)
		{
			eps_predict = p_temphost6[i] + beta[0] * p_temphost2[i] + beta[1] * p_temphost3[i] + beta[2] * p_temphost4[i];
			fprintf(jibble, "%d eps_predict %1.14E epsilon %1.14E predicted dbyd %1.14E %1.14E %1.14E old_eps %1.14E \n", 
				i, eps_predict, p_temphost1[i],
				p_temphost2[i], p_temphost3[i], p_temphost4[i], p_temphost6[i]);
		}
		fclose(jibble);
		printf("\n\nFile saved\a\n\n");
		*/

		/*sprintf(buffer, "eps%d.txt", iIteration);
		cudaMemcpy(p_temphost1, p_epsilon, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
		jibble = fopen(buffer, "w");
		for (int i = 0; i < NMINOR; i++)
			fprintf(jibble, "%d eps %1.14E\n", i, p_temphost1[i]);
		fclose(jibble);*/

		cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMinor, cudaMemcpyDeviceToHost);
		bContinue = false;
		for (iTile = 0; ((p_boolhost[iTile] == 0) && (iTile < numTilesMinor)); iTile++);
		;
		if (iTile < numTilesMinor) {
			printf("failed test\n");
			bContinue = true;
		};
		iIteration++;

	} while (bContinue);
//	fclose(fpdbg);

	GlobalSuppressSuccessVerbosity = false;

}


void RegressionSeedAz(f64 const hsub,
	f64 * pAz_k,
	f64 * p_AzNext,
	f64 * p_x1, f64 * p_x2, f64 * p_Azdot0, 
	f64 * p_gamma,
	f64 * p_LapCoeffself, 
	cuSyst * pX_use)
{
	f64 L2eps;
	f64 beta[3];
	Tensor3 mat;
	f64 RSS;
	int iTile;
	char buffer[256];
	int iIteration = 0;

	cudaMemcpy(p_AzNext, pAz_k, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
	// was this what we were missing?
	// yet it shouldn't have been far out anyway?

	GlobalSuppressSuccessVerbosity = true;

	kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
		pX_use->p_info, pX_use->p_tri_neigh_index,
		p_x1);
	Call(cudaThreadSynchronize(), "cudaTS ResetFrills 1");


	// Create Epsilon for initial state:

	kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
		pX_use->p_info,
		p_AzNext,
		pX_use->p_izTri_vert,
		pX_use->p_izNeigh_TriMinor,
		pX_use->p_szPBCtri_vert,
		pX_use->p_szPBC_triminor,
		p_LapAzNext,
		pX_use->p_AreaMinor // populates it
		);
	Call(cudaThreadSynchronize(), "cudaTS GetLap Az JLS 1");

	kernelCreateEpsilonAndJacobi << <numTilesMinor, threadsPerTileMinor >> >
		(hsub, // ?
			pX_use->p_info,
			p_AzNext, pAz_k,
			p_Azdot0, p_gamma,
			p_LapCoeffself, p_LapAzNext,
			p_epsilon, p_Jacobi_x,
			//this->p_AAdot, // to use Az_dot_k for equation);
			p_bFailed
			);
	Call(cudaThreadSynchronize(), "cudaTS CreateEpsAndJacobi 1");
	//kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
	//	pX_use->p_info, pX_use->p_tri_neigh_index, p_Jacobi_x);
	//Call(cudaThreadSynchronize(), "cudaTS ResetFrills Jacobi");

	// DEBUG:
	cudaMemcpy(p_x1, p_epsilon, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);

	// We don't really want to use Jacobi for epsilon here.

	kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
		pX_use->p_info,
		p_x2,
		pX_use->p_izTri_vert,
		pX_use->p_izNeigh_TriMinor,
		pX_use->p_szPBCtri_vert,
		pX_use->p_szPBC_triminor,
		p_temp5,
		pX_use->p_AreaMinor
		);
	Call(cudaThreadSynchronize(), "cudaTS Get Lap 1");

	kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
		pX_use->p_info,
		p_x1,
		pX_use->p_izTri_vert,
		pX_use->p_izNeigh_TriMinor,
		pX_use->p_szPBCtri_vert,
		pX_use->p_szPBC_triminor,
		p_temp4,
		pX_use->p_AreaMinor
		);
	Call(cudaThreadSynchronize(), "cudaTS Get Lap 2");

	kernelCreate_further_regressor << <numTilesMinor, threadsPerTileMinor >> > (
		pX_use->p_info,
		hsub,
		p_x1,
		p_temp4,
		p_LapCoeffself,
		p_gamma,
		p_Jacobi_x);
	Call(cudaThreadSynchronize(), "cudaTS Create further regressor");
	kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
		pX_use->p_info, pX_use->p_tri_neigh_index,
		p_Jacobi_x);
	Call(cudaThreadSynchronize(), "cudaTS ResetFrills further regressor");
	
	kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
		pX_use->p_info,
		p_Jacobi_x,
		pX_use->p_izTri_vert,
		pX_use->p_izNeigh_TriMinor,
		pX_use->p_szPBCtri_vert,
		pX_use->p_szPBC_triminor,
		p_LapJacobi,
		//		p_temp1, p_temp2, p_temp3,
		pX_use->p_AreaMinor
		);
	Call(cudaThreadSynchronize(), "cudaTS Get Lap Jacobi 3");

	kernelAccumulateMatrix << <numTilesMinor, threadsPerTileMinor >> >(
		pX_use->p_info,
		hsub,
		p_epsilon,
		p_x1,       // used epsilon
		p_x2,       // difference of previous soln's
		p_Jacobi_x, // Jacobi of x1
		p_temp4, // Lap of x1
		p_temp5, // Lap of x2  // don't use temp6 ! It is x2!
		p_LapJacobi,
		p_gamma,
		p_temp1, // sum of matrices, in lots of 6
		p_eps_against_deps
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelAccumulateMatrix");

	// Now take 6 sums
	f64 sum_mat[6];
	f64_vec3 sumvec(0.0, 0.0, 0.0);
	memset(sum_mat, 0, sizeof(f64) * 6);
	cudaMemcpy(p_temphost1, p_temp1, sizeof(f64) * 6 * numTilesMinor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_vec_host, p_eps_against_deps, sizeof(f64_vec3)*numTilesMinor, cudaMemcpyDeviceToHost);
	for (iTile = 0; iTile < numTilesMinor; iTile++)
	{
		sum_mat[0] += p_temphost1[iTile * 6 + 0];
		sum_mat[1] += p_temphost1[iTile * 6 + 1];
		sum_mat[2] += p_temphost1[iTile * 6 + 2];
		sum_mat[3] += p_temphost1[iTile * 6 + 3];
		sum_mat[4] += p_temphost1[iTile * 6 + 4];
		sum_mat[5] += p_temphost1[iTile * 6 + 5];
		sumvec += p_sum_vec_host[iTile];
	};

	// Now populate symmetric matrix
	f64_tens3 mat2;

	mat.xx = sum_mat[0];
	mat.xy = sum_mat[1];
	mat.xz = sum_mat[2];
	mat.yx = mat.xy;
	mat.yy = sum_mat[3];
	mat.yz = sum_mat[4];
	mat.zx = mat.xz;
	mat.zy = mat.yz;
	mat.zz = sum_mat[5];
	mat.Inverse(mat2);
	printf(
		" ( %1.6E %1.6E %1.6E ) ( beta0 )   ( %1.6E )\n"
		" ( %1.6E %1.6E %1.6E ) ( beta1 ) = ( %1.6E )\n"
		" ( %1.6E %1.6E %1.6E ) ( beta2 )   ( %1.6E )\n",
		mat.xx, mat.xy, mat.xz, sumvec.x,
		mat.yx, mat.yy, mat.yz, sumvec.y,
		mat.zx, mat.zy, mat.zz, sumvec.z);
	f64_vec3 product = mat2*sumvec;

	beta[0] = -product.x; beta[1] = -product.y; beta[2] = -product.z;

	printf("beta %1.8E %1.8E %1.8E ", beta[0], beta[1], beta[2]);

	kernelAddRegressors << <numTilesMinor, threadsPerTileMinor >> >(
		p_AzNext,
		beta[0], beta[1], beta[2],
		p_x1,
		p_x2,
		p_Jacobi_x
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelAddRegressors");
	
	// TESTING:

	//kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> >
	//	(p_epsilon, p_temp1);
	//cudaMemcpy(p_temphost3, p_temp1, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	//RSS = 0.0;
	//for (iTile = 0; iTile < numTilesMinor; iTile++) RSS += p_temphost3[iTile];
	//L2eps = sqrt(RSS / (f64)NMINOR);
	//printf("L2eps: %1.9E \n", L2eps);


	//kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
	//	pX_use->p_info,
	//	p_AzNext,
	//	pX_use->p_izTri_vert,
	//	pX_use->p_izNeigh_TriMinor,
	//	pX_use->p_szPBCtri_vert,
	//	pX_use->p_szPBC_triminor,
	//	p_LapAzNext,
	//	pX_use->p_AreaMinor // populates it
	//	);
	//Call(cudaThreadSynchronize(), "cudaTS GetLap Az JLS 1");

	//kernelCreateEpsilonAndJacobi << <numTilesMinor, threadsPerTileMinor >> >
	//	(hsub, // ?
	//		pX_use->p_info,
	//		p_AzNext, pAz_k,
	//		p_Azdot0, p_gamma,
	//		p_LapCoeffself, p_LapAzNext,
	//		p_epsilon, p_Jacobi_x,
	//		//this->p_AAdot, // to use Az_dot_k for equation);
	//		p_bFailed
	//		);
	//Call(cudaThreadSynchronize(), "cudaTS CreateEpsAndJacobi 1");
	//

	//kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> >
	//	(p_epsilon, p_temp1);
	//cudaMemcpy(p_temphost3, p_temp1, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	//RSS = 0.0;
	//for (iTile = 0; iTile < numTilesMinor; iTile++) RSS += p_temphost3[iTile];
	//L2eps = sqrt(RSS / (f64)NMINOR);
	//printf("L2eps: %1.9E \n", L2eps);
	//

	// Totally ineffective -- choosing small coefficients.
	// Can't believe these aren't useful regressors.


}

int RunBackwardForHeat_ConjugateGradient(T3 * p_T_k, T3 * p_T, f64 hsub, cuSyst * pX_use)
{
#define UPLIFT_THRESHOLD 0.33

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

	f64 beta_eJ, beta_iJ, beta_nJ, beta_eR, beta_iR, beta_nR;
	long iTile;

	Matrix_real sum_ROC_products, sum_products_i, sum_products_e;
	f64 sum_eps_deps_by_dbeta_vector[8];
	f64 beta[8];

	f64 sum_eps_deps_by_dbeta_J, sum_eps_deps_by_dbeta_R,
		sum_depsbydbeta_J_times_J, sum_depsbydbeta_R_times_R, sum_depsbydbeta_J_times_R,
		sum_eps_eps;
	f64 dot2_n, dot2_i, dot2_e;
	bool bProgress;

	Vertex * pVertex;
	long iVertex;
	plasma_data * pdata;

	GlobalSuppressSuccessVerbosity = true;


	// Assume p_T contains seed.


	printf("\nConjugate gradient for heat: ");
	//long iMinor;
	f64 L2eps;// _elec, L2eps_neut, L2eps_ion;
	bool bFailedTest;
	Triangle * pTri;
	f64_vec4 tempf64vec4;
	f64 tempf64;
	f64 SS_n, SS_i, SS_e, oldSS_n, oldSS_i, oldSS_e, ratio_n, ratio_i, ratio_e,
		alpha_n, alpha_i, alpha_e, dot_n, dot_i, dot_e, RSS_n, RSS_i, RSS_e;

	// seed: just set T to T_k.
	kernelUnpacktorootNT << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_NnTn, p_NTi, p_NTe, p_T,
			pX_use->p_AreaMajor,
			pX_use->p_n_major);
	Call(cudaThreadSynchronize(), "cudaTS UnpacktoNT");

	// 1. Compute epsilon:

	// epsilon = T_k+1 - T_k - (h/N)rates.NT;
	// epsilon = b - A T_k+1
	// Their Matrix A = -identity + (h/N) ROC NT due to heat flux
	// Their b = - T_k

	// Compute heat flux given p_T	
	cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
	//
	kernelCreateTfromNTbydividing << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_Tn, p_Ti, p_Te, p_NnTn, p_NTi, p_NTe, // divide by root N
			pX_use->p_AreaMajor,
			pX_use->p_n_major);
	Call(cudaThreadSynchronize(), "cudaTS CreateTfromNT");

	kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_scalarT << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(
			pX_use->p_info,
			pX_use->p_izNeigh_vert,
			pX_use->p_szPBCneigh_vert,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_cc,
			pX_use->p_n_major,
			p_Tn, p_Ti, p_Te,  // using vert indices

			pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
			p_kappa_n,
			p_kappa_i, p_kappa_e,
			p_nu_i, p_nu_e,
			NT_addition_rates_d_temp,
			pX_use->p_AreaMajor);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

	// Let's think about this carefully. If we could do it with the same data loaded each vertex for neigh list it would actually save time.
	// We rely on the fact that loading the vertex list data is contiguous fetch.
	// And we do store it for 3 species - so let's stick with all in 1.

	kernelCreateEpsilonHeat << <numTilesMajorClever, threadsPerTileMajorClever >> >
		(hsub,
			pX_use->p_info + BEGINNING_OF_CENTRAL,
			p_epsilon_n, p_epsilon_i, p_epsilon_e,
			p_NnTn, p_NTi, p_NTe,
			p_T_k,					// p_T_k was not swapped so that might be why different result. All this is tough.
			pX_use->p_AreaMajor,
			pX_use->p_n_major,
			NT_addition_rates_d_temp, // it's especially silly having a whole struct of 5 instead of 3 here.
			0); // Must create root N times the equation
	Call(cudaThreadSynchronize(), "cudaTS CreateEpsilon");

	// --==
	// p = eps
	cudaMemcpy(p_regressor_n, p_epsilon_n, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_regressor_i, p_epsilon_i, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_regressor_e, p_epsilon_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);

	// Take RSS :
	kernelAccumulateSumOfSquares << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_epsilon_n, p_epsilon_i, p_epsilon_e,
			p_temp1, p_temp2, p_temp3);
	Call(cudaThreadSynchronize(), "cudaTS Sumof squares");
	SS_n = 0.0; SS_i = 0.0; SS_e = 0.0;
	cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
	for (iTile = 0; iTile < numTilesMajorClever; iTile++)
	{
		SS_n += p_temphost1[iTile];
		SS_i += p_temphost2[iTile];
		SS_e += p_temphost3[iTile];
	}
	oldSS_n = SS_n; oldSS_i = SS_i; oldSS_e = SS_e;
	long iIteration = 0;
	printf("iIteration %d : L2eps[sqrtN T] %1.9E %1.9E %1.9E \n", iIteration, sqrt(SS_n / (f64)NUMVERTICES),
		sqrt(SS_i / (f64)NUMVERTICES), sqrt(SS_e / (f64)NUMVERTICES));


	bool bContinue = true;
	do {

		// See if we can get rid of additional ROCWRTregressor routine.
		// Ap = -p + (h / N) flux(p);
		// remember -Ax is what appears in epsilon.

		// normally eps = T_k+1 - T_k - (h/N) flux(T_k+1)

		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);

		// regressor represents the addition to NT, we need to pass T = regressor/N
//
		kernelCreateTfromNTbydividing << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_Tn, p_Ti, p_Te, p_regressor_n, p_regressor_i, p_regressor_e,
				pX_use->p_AreaMajor, // divide by root N!
				pX_use->p_n_major);
		Call(cudaThreadSynchronize(), "cudaTS CreateTfromNT reg");

		kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_scalarT << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(
				pX_use->p_info,
				pX_use->p_izNeigh_vert,
				pX_use->p_szPBCneigh_vert,
				pX_use->p_izTri_vert,
				pX_use->p_szPBCtri_vert,
				pX_use->p_cc,
				pX_use->p_n_major,
				p_Tn, p_Ti, p_Te,  // using vert indices
				pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
				p_kappa_n,
				p_kappa_i, p_kappa_e,
				p_nu_i, p_nu_e,
				NT_addition_rates_d_temp,
				pX_use->p_AreaMajor);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate 1");

		cudaMemset(zero_array, 0, sizeof(T3)*NUMVERTICES);
		kernelCreateEpsilonHeat << <numTilesMajorClever, threadsPerTileMajorClever >> >
			(hsub,
				pX_use->p_info + BEGINNING_OF_CENTRAL,
				p_Ap_n, p_Ap_i, p_Ap_e, // the result
				p_regressor_n, p_regressor_i, p_regressor_e,
				zero_array, // 0 which we multiply by N...
				pX_use->p_AreaMajor,
				pX_use->p_n_major, // we load this only in order to multiply with 0
				NT_addition_rates_d_temp,
				0);
		Call(cudaThreadSynchronize(), "cudaTS CreateEpsilonHeat");

		// I believe that we should be taking - [eps dot deps/dbeta] / [deps/dbeta dot deps/dbeta]
		/*
		kernelAccumulateDotProducts << <numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_epsilon_n, p_Ap_n, // it's -A really
				p_epsilon_i, p_Ap_i,
				p_epsilon_e, p_Ap_e,
				p_temp1, p_temp2, p_temp3);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDotProducts");
		dot_n = 0.0; dot_i = 0.0; dot_e = 0.0;
		cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMajorClever; iTile++)
		{
			dot_n += p_temphost1[iTile];
			dot_i += p_temphost2[iTile];
			dot_e += p_temphost3[iTile];
		}

		kernelAccumulateSumOfSquares << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_Ap_n, p_Ap_i, p_Ap_e, p_temp1, p_temp2, p_temp3);
		Call(cudaThreadSynchronize(), "cudaTS Sumof squares");
		dot2_n = 0.0; dot2_i = 0.0; dot2_e = 0.0;
		cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMajorClever; iTile++)
		{
			dot2_n += p_temphost1[iTile];
			dot2_i += p_temphost2[iTile];
			dot2_e += p_temphost3[iTile];
		}

		alpha_n = -dot_n / dot2_n;
		alpha_i = -dot_i / dot2_i;
		alpha_e = -dot_e / dot2_e; // ?
		*/

		// Are  we missing a minus??? Yes .. true? epsilon = b-A NT and we just calculated epsilon from NT, b=0
		// ====================================================================================================


		NegateVectors << <numTilesMajorClever, threadsPerTileMajorClever >> > (p_Ap_n, p_Ap_i, p_Ap_e);
		Call(cudaThreadSynchronize(), "cudaTS NegateVectors");

		kernelAccumulateDotProducts << <numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_regressor_n, p_Ap_n,
				p_regressor_i, p_Ap_i,
				p_regressor_e, p_Ap_e,
				p_temp1, p_temp2, p_temp3);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDotProducts");
		dot_n = 0.0; dot_i = 0.0; dot_e = 0.0;
		cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMajorClever; iTile++)
		{
			dot_n += p_temphost1[iTile];
			dot_i += p_temphost2[iTile];
			dot_e += p_temphost3[iTile];
		}

		alpha_n = SS_n / dot_n;
		alpha_i = SS_i / dot_i;
		alpha_e = SS_e / dot_e; // dot is so large that alpha is very small.
		//
		//printf("alpha %1.8E %1.8E %1.8E SS %1.8E %1.8E %1.8E dot %1.8E %1.8E %1.8E\n", alpha_n, alpha_i, alpha_e,
		//	SS_n, SS_i, SS_e, dot_n, dot_i, dot_e);
		/*
		NegateVectors << <numTilesMajorClever, threadsPerTileMajorClever >> >(p_Ap_n, p_Ap_i, p_Ap_e);
		Call(cudaThreadSynchronize(), "cudaTS NegateVectors");

		kernelAccumulateDotProducts << <numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_regressor_n, p_epsilon_n,
				p_regressor_i, p_epsilon_i,
				p_regressor_e, p_epsilon_e,
				p_temp1, p_temp2, p_temp3);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDotProducts");
		dot_n = 0.0; dot_i = 0.0; dot_e = 0.0;
		cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMajorClever; iTile++)
		{
			dot_n += p_temphost1[iTile];
			dot_i += p_temphost2[iTile];
			dot_e += p_temphost3[iTile];
		}
		kernelAccumulateDotProducts << <numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_regressor_n, p_Ap_n,
				p_regressor_i, p_Ap_i,
				p_regressor_e, p_Ap_e,
				p_temp1, p_temp2, p_temp3);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDotProducts");
		dot2_n = 0.0; dot2_i = 0.0; dot2_e = 0.0;
		cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMajorClever; iTile++)
		{
			dot2_n += p_temphost1[iTile];
			dot2_i += p_temphost2[iTile];
			dot2_e += p_temphost3[iTile];
		}

		alpha_n = dot_n / dot2_n;
		alpha_i = dot_i / dot2_i;
		alpha_e = dot_e / dot2_e; // dot is so large that alpha is very small.

		printf("alpha %1.8E %1.8E %1.8E dot %1.8E %1.8E %1.8E dot %1.8E %1.8E %1.8E\n", alpha_n, alpha_i, alpha_e,
			dot_n, dot_i, dot_e, dot2_n, dot2_i, dot2_e);
		*/

		VectorAddMultiple << <numTilesMajorClever, threadsPerTileMajorClever >> > (
			p_NnTn, alpha_n, p_regressor_n,
			p_NTi, alpha_i, p_regressor_i,
			p_NTe, alpha_e, p_regressor_e);
		// T.n += alpha_n*p_regressor_n;
		Call(cudaThreadSynchronize(), "cudaTS AddMultiples");
		//
		kernelCreateTfromNTbydividing << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_Tn, p_Ti, p_Te, p_NnTn, p_NTi, p_NTe,
				pX_use->p_AreaMajor,
				pX_use->p_n_major);
		Call(cudaThreadSynchronize(), "cudaTS CreateTfromNT");

		// Update Epsilon: or can simply recalculate eps = b - A newx
		// Sometimes would rather update epsilon completely:
		if (iIteration % 4 == 0) {

			cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
			kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_scalarT << < numTilesMajorClever, threadsPerTileMajorClever >> >
				(
					pX_use->p_info,
					pX_use->p_izNeigh_vert,
					pX_use->p_szPBCneigh_vert,
					pX_use->p_izTri_vert,
					pX_use->p_szPBCtri_vert,
					pX_use->p_cc,
					pX_use->p_n_major,
					p_Tn, p_Ti, p_Te,
					pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
					p_kappa_n,
					p_kappa_i, p_kappa_e,
					p_nu_i, p_nu_e,
					NT_addition_rates_d_temp,
					pX_use->p_AreaMajor);
			Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

			cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever);
			kernelCreateEpsilonHeat << <numTilesMajorClever, threadsPerTileMajorClever >> >
				(hsub,
					pX_use->p_info + BEGINNING_OF_CENTRAL,
					p_epsilon_n, p_epsilon_i, p_epsilon_e,
					p_NnTn, p_NTi, p_NTe,
					p_T_k,
					pX_use->p_AreaMajor,
					pX_use->p_n_major,
					NT_addition_rates_d_temp, // it's especially silly having a whole struct of 5 instead of 3 here.
					p_bFailed);
			Call(cudaThreadSynchronize(), "cudaTS CreateEpsilon");

			cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMajor, cudaMemcpyDeviceToHost);

		}
		else {
			bContinue = true;
			VectorAddMultiple << <numTilesMajorClever, threadsPerTileMajorClever >> > (
				p_epsilon_n, -alpha_n, p_Ap_n,
				p_epsilon_i, -alpha_i, p_Ap_i,
				p_epsilon_e, -alpha_e, p_Ap_e
				);
			Call(cudaThreadSynchronize(), "cudaTS AddMultiples 2");
			p_boolhost[0] = true;

		};

		kernelAccumulateSumOfSquares << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_epsilon_n, p_epsilon_i, p_epsilon_e,
				p_temp1, p_temp2, p_temp3);
		Call(cudaThreadSynchronize(), "cudaTS Sumof squares");
		SS_n = 0.0; SS_i = 0.0; SS_e = 0.0;
		cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		bContinue = false;
		for (iTile = 0; iTile < numTilesMajorClever; iTile++)
		{
			SS_n += p_temphost1[iTile];
			SS_i += p_temphost2[iTile];
			SS_e += p_temphost3[iTile];
			if (p_boolhost[iTile] == true) bContinue = true;
		}
		ratio_n = SS_n / oldSS_n;
		ratio_i = SS_i / oldSS_i;
		ratio_e = SS_e / oldSS_e;

		printf("iIteration %d : L2eps[sqrtN T] %1.9E %1.9E %1.9E \n", iIteration, sqrt(SS_n / (f64)NUMVERTICES),
			sqrt(SS_i / (f64)NUMVERTICES), sqrt(SS_e / (f64)NUMVERTICES));

		//printf("ratio %1.10E %1.10E %1.10E SS %1.10E %1.10E %1.10E", ratio_n, ratio_i, ratio_e,
		//	SS_n,SS_i,SS_e);

		kernelRegressorUpdate << <numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_regressor_n, p_regressor_i, p_regressor_e,
				p_epsilon_n, p_epsilon_i, p_epsilon_e,
				ratio_n, ratio_i, ratio_e);
		Call(cudaThreadSynchronize(), "cudaTS RegressorUpdate");
		// regressor = epsilon + ratio*regressor; 

		bProgress = false;
		if ((oldSS_n > 0.0) && (sqrt(ratio_n) < 0.98)) bProgress = true;
		if ((oldSS_i > 0.0) && (sqrt(ratio_i) < 0.98)) bProgress = true;
		if ((oldSS_e > 0.0) && (sqrt(ratio_e) < 0.98)) bProgress = true;

		oldSS_n = SS_n;
		oldSS_i = SS_i;
		oldSS_e = SS_e;

		// Now calculate epsilon in original equations
	/*	cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever);
		if (iIteration % 4 == 0) {
			kernelCreateEpsilonHeatOriginalScaling << <numTilesMajorClever, threadsPerTileMajorClever >> >
				(hsub,
					pX_use->p_info + BEGINNING_OF_CENTRAL,
					p_temp4, p_temp5, p_temp6,
					p_Tn, p_Ti, p_Te,
					p_T_k,
					pX_use->p_AreaMajor,
					pX_use->p_n_major,
					NT_addition_rates_d_temp, // we were assuming this was populated for T but it sometimes wasn't.
					p_bFailed);
			Call(cudaThreadSynchronize(), "cudaTS CreateEpsilonOriginal");
			kernelAccumulateSumOfSquares << < numTilesMajorClever, threadsPerTileMajorClever >> >
				(p_temp4, p_temp5, p_temp6,
					p_temp1, p_temp2, p_temp3);
			Call(cudaThreadSynchronize(), "cudaTS Sumof squares");
			RSS_n = 0.0; RSS_i = 0.0; RSS_e = 0.0;
			cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMajor, cudaMemcpyDeviceToHost);
			bContinue = false;
			for (iTile = 0; iTile < numTilesMajorClever; iTile++)
			{
				RSS_n += p_temphost1[iTile];
				RSS_i += p_temphost2[iTile];
				RSS_e += p_temphost3[iTile];
				if (p_boolhost[iTile] == true) bContinue = true;
			}
		} else {
			// NT addition rates was not populated -- skip this
			bContinue = true;
		}
		printf("original eqns: L2eps %1.12E %1.12E %1.12E \n", sqrt(RSS_n / (f64)NUMVERTICES),
			sqrt(RSS_i / (f64)NUMVERTICES), sqrt(RSS_e / (f64)NUMVERTICES));
			*/

		iIteration++;

		if (bContinue == false) printf("all tests ok\n");

		// Seems to present correct result yet gives incorrect figures for L2eps in between - I have no idea why this is.

		// set bContinue according to all species converging
	} while ((bContinue) &&
		((iIteration < 40) || (bProgress))
		);



	GlobalSuppressSuccessVerbosity = false;

		//((sqrt(RSS_i / (f64)NUMVERTICES) > 1.0e-28) ||
		//(sqrt(RSS_e / (f64)NUMVERTICES) > 1.0e-28) ||
			//(sqrt(RSS_n / (f64)NUMVERTICES) > 1.0e-28)));
	
	// It is looking a lot like we could save a lot if we had
	// actually split it out by species. Clearly that didn't
	// occur to me at the time I did the routines but it is now obvious.
	// There should be 1 routine, called for each species. 
	// We are just reusing position - not worth it I think.

	// OKay how much of it is now proposing to use split out arrays? This is when we start doing this.
	// We can zip them back up into T3 struct afterwards.

	// seed: just set T to T_k.
	kernelPackupT3 << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_T, p_Tn, p_Ti, p_Te);	 // we did division since we updated NT.

	if (bContinue == true) return 1;
	return 0;
}
	
int RunBackwardJLSForHeat(T3 * p_T_k, T3 * p_T, f64 hsub, cuSyst * pX_use)
{
#define UPLIFT_THRESHOLD 0.33
	GlobalSuppressSuccessVerbosity = true;

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

	f64 beta_eJ, beta_iJ, beta_nJ, beta_eR, beta_iR, beta_nR;
	long iTile;
	bool bProgress;

	Matrix_real sum_ROC_products, sum_products_i, sum_products_e;
	f64 sum_eps_deps_by_dbeta_vector[8];
	f64 beta[8];

	f64 sum_eps_deps_by_dbeta_J, sum_eps_deps_by_dbeta_R,
		sum_depsbydbeta_J_times_J, sum_depsbydbeta_R_times_R, sum_depsbydbeta_J_times_R,
		sum_eps_eps;

	Vertex * pVertex;
	long iVertex;
	plasma_data * pdata;

	// seed: just set T to T_k.
//	cudaMemcpy(p_T, p_T_k, sizeof(T3)*NUMVERTICES, cudaMemcpyDeviceToDevice);

	// Assume we were passed the seed.

	kernelUnpack << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_Tn, p_Ti, p_Te, p_T);
	Call(cudaThreadSynchronize(), "cudaTS unpack");
	cudaMemset(p_T, 0, sizeof(T3)*NUMVERTICES);
	// Same should apply for all solver routines: initial seed to be created by regression from previous solutions.

	printf("\nJRLS for heat: ");
	//long iMinor;
	f64 L2eps_n, L2eps_e, L2eps_i, L2eps_n_old, L2eps_i_old, L2eps_e_old;
	bool bFailedTest;
	Triangle * pTri;
	f64_vec4 tempf64vec4;
	f64 tempf64;
	int iIteration = 0;

	// Better if we would just load in epsilon and create Jacobi? No coeffself array, simpler.
	// Careful: if we find coeffself in d/dt NT then eps <- -(h/N) d/dtNT
	// so we need N as well.
	kernelCalc_SelfCoefficient_for_HeatConduction << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(
			hsub,
			pX_use->p_info, // minor
			pX_use->p_izNeigh_vert,
			pX_use->p_szPBCneigh_vert,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_cc,
			pX_use->p_n_major,
			pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
			p_kappa_n,
			p_kappa_i,
			p_kappa_e,
			p_nu_i,
			p_nu_e,
			pX_use->p_AreaMajor,

			p_coeffself_n,
			p_coeffself_i,
			p_coeffself_e // what exactly it calculates?
			); // used for Jacobi
	Call(cudaThreadSynchronize(), "cudaTS CalcSelfCoefficientForHeatConduction");

	
	do {
		printf("\nITERATION %d \n\n", iIteration);
		if ((iIteration >= 2000) && (iIteration % 1 == 0))
		{
			// draw a graph:

			cudaMemcpy(p_temphost1, p_epsilon_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost2, p_d_eps_by_dbeta_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost3, p_Jacobi_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);

			pVertex = pTriMesh->X;
			pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
			for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
			{
				pdata->temp.x = p_temphost1[iVertex];
				pdata->temp.y = p_temphost1[iVertex];

				if (fabs(p_temphost1[iVertex]) > 1.0e-14) {
					printf("iVertex %d epsilon %1.10E deps/dJac %1.9E ", iVertex, p_temphost1[iVertex], p_temphost2[iVertex]);						
					cudaMemcpy(&tempf64, &(p_Jacobi_e[iVertex]), sizeof(f64), cudaMemcpyDeviceToHost);
					printf("Jac %1.8E ", tempf64);
					cudaMemcpy(&tempf64, &(p_Te[iVertex]), sizeof(f64), cudaMemcpyDeviceToHost);
					printf("Te %1.8E\n", tempf64);
				};
				//if ((pVertex->flags == DOMAIN_VERTEX) || (pVertex->flags == OUTERMOST))
				//{

				//	} else {
				//		pdata->temp.x = 0.0;
				//		pdata->temp.y = 0.0;
				//	}
				++pVertex;
				++pdata;
			}

			Graph[0].DrawSurface("uuuu",
				DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
				AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
				false,
				GRAPH_EPSILON, pTriMesh);

			//pVertex = pTriMesh->X;
			//pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
			//for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
			//{
			//pdata->temp.x = p_temphost3[iVertex];
			//pdata->temp.y = p_temphost3[iVertex];
			//++pVertex;
			//++pdata;
			//}

			////overdraw:
			//Graph[0].DrawSurface("Jacobi_e",
			//DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
			//AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
			//false,
			//GRAPH_OPTI, pTriMesh);
			
//			pVertex = pTriMesh->X;
//			pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
//			for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
//			{
//				pdata->temp.x = p_temphost3[iVertex];
//				pdata->temp.y = p_temphost3[iVertex];
//				++pVertex;
//				++pdata;
//			}
//			Graph[2].DrawSurface("Jacobi",
//				DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
//				AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
//				false,
//				GRAPH_NINE, pTriMesh);
//
			Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);

			printf("done graph");
		}
		// 2. Calculate epsilon: given the est of T, eps = T - (T_k +- h sum kappa dot grad T)
		// 3. Calculate Jacobi: for each point, Jacobi = eps/(deps/dT)

		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
		kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_scalarT << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(
				pX_use->p_info,
				pX_use->p_izNeigh_vert,
				pX_use->p_szPBCneigh_vert,
				pX_use->p_izTri_vert,
				pX_use->p_szPBCtri_vert,
				pX_use->p_cc,
				pX_use->p_n_major,
				p_Tn, p_Ti, p_Te,
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

		// Note: most are near 1, a few are like 22 or 150.
		CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajor));
		kernelCreateEpsilonAndJacobi_Heat << <numTilesMajor, threadsPerTileMajor >> > (
			// eps = T - (T_k +- h sum kappa dot grad T)
			// x = -eps/coeffself
			hsub,
			pX_use->p_info + BEGINNING_OF_CENTRAL,
			p_Tn, p_Ti, p_Te,
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
			p_Jacobi_e,
			p_bFailed		
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
		cudaMemset(p_d_eps_by_dbetaR_n, 0, sizeof(f64)*NUMVERTICES);
		cudaMemset(p_d_eps_by_dbetaR_i, 0, sizeof(f64)*NUMVERTICES);
		cudaMemset(p_d_eps_by_dbetaR_e, 0, sizeof(f64)*NUMVERTICES);
		
		CallMAC(cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES));
		kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_scalarT << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(
				pX_use->p_info,
				pX_use->p_izNeigh_vert,
				pX_use->p_szPBCneigh_vert,
				pX_use->p_izTri_vert,
				pX_use->p_szPBCtri_vert,
				pX_use->p_cc,
				pX_use->p_n_major,
				p_Jacobi_n, p_Jacobi_i, p_Jacobi_e,
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
		 
		// Note: most are near 1, a few are like 22 or 150.
		kernelCreateEpsilon_Heat_for_Jacobi << <numTilesMajor, threadsPerTileMajor >> > (
			// eps = T - (T_k +- h sum kappa dot grad T)
			// x = -eps/coeffself
			hsub,
			pX_use->p_info + BEGINNING_OF_CENTRAL,
			p_Jacobi_n, p_Jacobi_i, p_Jacobi_e,
			p_T, // zerovec
			NT_addition_rates_d_temp,
			// NEED N = n*AreaMajor
			pX_use->p_n_major, // got this
			pX_use->p_AreaMajor, // got this -> N, Nn

			p_d_eps_by_dbeta_n, p_d_eps_by_dbeta_i, p_d_eps_by_dbeta_e
			// no check threshold
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");

		// and eps as regressor:

		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
		kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_scalarT << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(
				pX_use->p_info,
				pX_use->p_izNeigh_vert,
				pX_use->p_szPBCneigh_vert,
				pX_use->p_izTri_vert,
				pX_use->p_szPBCtri_vert,
				pX_use->p_cc,
				pX_use->p_n_major,
				p_epsilon_n, p_epsilon_i, p_epsilon_e, // We could easily use 2nd iterate of Jacobi instead. Most probably, profitably.
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

		// Note: most are near 1, a few are like 22 or 150.
		kernelCreateEpsilon_Heat_for_Jacobi << <numTilesMajor, threadsPerTileMajor >> > (
			// eps = T - (T_k +- h sum kappa dot grad T)
			// x = -eps/coeffself
			hsub,
			pX_use->p_info + BEGINNING_OF_CENTRAL,
			p_epsilon_n, p_epsilon_i, p_epsilon_e,
			p_T, // T3 zerovec
			NT_addition_rates_d_temp,
			// NEED N = n*AreaMajor
			pX_use->p_n_major, // got this
			pX_use->p_AreaMajor, // got this -> N, Nn

			p_d_eps_by_dbetaR_n, p_d_eps_by_dbetaR_i, p_d_eps_by_dbetaR_e
			// no check threshold
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");


		
		// 5. Do JLS calcs and update T
		// But we have to do for 3 species: do separately, so scalar arrays.

		//		kernelAccumulateSummands2 << <numTilesMajor, threadsPerTileMajor >> > (
		//			pX_use->p_info,
		//			p_epsilon_n,
		//			p_d_eps_by_dbeta_n,
		//			p_sum_eps_deps_by_dbeta,
		//			p_sum_depsbydbeta_sq,
		//			p_sum_eps_eps);
		//		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands OOA");


		kernelAccumulateSummands4 << <numTilesMajor, threadsPerTileMajor >> > (

			// We don't need to test for domain, we need to make sure the summands are zero otherwise.
			p_epsilon_n,

			p_d_eps_by_dbeta_n,
			p_d_eps_by_dbetaR_n,

			// 6 outputs:
			p_sum_eps_deps_by_dbeta_J,
			p_sum_eps_deps_by_dbeta_R,
			p_sum_depsbydbeta_J_times_J,
			p_sum_depsbydbeta_R_times_R,
			p_sum_depsbydbeta_J_times_R,
			p_sum_eps_eps);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands neut");


		cudaMemcpy(p_sum_eps_deps_by_dbeta_J_host, p_sum_eps_deps_by_dbeta_J, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_deps_by_dbeta_R_host, p_sum_eps_deps_by_dbeta_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_J_times_J_host, p_sum_depsbydbeta_J_times_J, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_R_times_R_host, p_sum_depsbydbeta_R_times_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_J_times_R_host, p_sum_depsbydbeta_J_times_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);

		sum_eps_deps_by_dbeta_J = 0.0;
		sum_eps_deps_by_dbeta_R = 0.0;
		sum_depsbydbeta_J_times_J = 0.0;
		sum_depsbydbeta_R_times_R = 0.0;
		sum_depsbydbeta_J_times_R = 0.0;
		sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMajor; iTile++)
		{
			sum_eps_deps_by_dbeta_J += p_sum_eps_deps_by_dbeta_J_host[iTile];
			sum_eps_deps_by_dbeta_R += p_sum_eps_deps_by_dbeta_R_host[iTile];
			sum_depsbydbeta_J_times_J += p_sum_depsbydbeta_J_times_J_host[iTile];
			sum_depsbydbeta_R_times_R += p_sum_depsbydbeta_R_times_R_host[iTile];
			sum_depsbydbeta_J_times_R += p_sum_depsbydbeta_J_times_R_host[iTile];
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		}

		f64 tempf64 = (real)NUMVERTICES;

		if ((sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0)
			|| (sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0))
		{
			printf("Could not define coefficient.");
			beta_nJ = 0.0;
			beta_nR = 0.0;
		}
		else {
			beta_nJ = -(sum_eps_deps_by_dbeta_J*sum_depsbydbeta_R_times_R - sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_R) /
				(sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
			beta_nR = -(sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_J - sum_eps_deps_by_dbeta_J*sum_depsbydbeta_J_times_R) /
				(sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
		}
		L2eps_n_old = L2eps_n;
		L2eps_n = sqrt(sum_eps_eps / (real)NUMVERTICES);
		printf("\n for neutral [ BetaJacobi %1.10E BetaRichardson %1.10E L2eps %1.10E ] ", beta_nJ, beta_nR, L2eps_n);


		kernelAccumulateSummands4 << <numTilesMajor, threadsPerTileMajor >> > (

			// We don't need to test for domain, we need to make sure the summands are zero otherwise.
			p_epsilon_i,
			p_d_eps_by_dbeta_i,
			p_d_eps_by_dbetaR_i,

			// 6 outputs:
			p_sum_eps_deps_by_dbeta_J,
			p_sum_eps_deps_by_dbeta_R,
			p_sum_depsbydbeta_J_times_J,
			p_sum_depsbydbeta_R_times_R,
			p_sum_depsbydbeta_J_times_R,
			p_sum_eps_eps);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands ion");


		cudaMemcpy(p_sum_eps_deps_by_dbeta_J_host, p_sum_eps_deps_by_dbeta_J, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_deps_by_dbeta_R_host, p_sum_eps_deps_by_dbeta_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_J_times_J_host, p_sum_depsbydbeta_J_times_J, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_R_times_R_host, p_sum_depsbydbeta_R_times_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_J_times_R_host, p_sum_depsbydbeta_J_times_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);

		sum_eps_deps_by_dbeta_J = 0.0;
		sum_eps_deps_by_dbeta_R = 0.0;
		sum_depsbydbeta_J_times_J = 0.0;
		sum_depsbydbeta_R_times_R = 0.0;
		sum_depsbydbeta_J_times_R = 0.0;
		sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMajor; iTile++)
		{
			sum_eps_deps_by_dbeta_J += p_sum_eps_deps_by_dbeta_J_host[iTile];
			sum_eps_deps_by_dbeta_R += p_sum_eps_deps_by_dbeta_R_host[iTile];
			sum_depsbydbeta_J_times_J += p_sum_depsbydbeta_J_times_J_host[iTile];
			sum_depsbydbeta_R_times_R += p_sum_depsbydbeta_R_times_R_host[iTile];
			sum_depsbydbeta_J_times_R += p_sum_depsbydbeta_J_times_R_host[iTile];
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		}
		 
		if ((sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0)
			|| (sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0))
		{
			printf("Could not define coefficient.");
			beta_iJ = 0.0; beta_iR = 0.0;
		}
		else {
			beta_iJ = -(sum_eps_deps_by_dbeta_J*sum_depsbydbeta_R_times_R - sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_R) /
				(sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
			beta_iR = -(sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_J - sum_eps_deps_by_dbeta_J*sum_depsbydbeta_J_times_R) /
				(sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);

		};
		
		L2eps_i_old = L2eps_i;
		L2eps_i = sqrt(sum_eps_eps / (real)NUMVERTICES);

		printf("\n for ION [ BetaJacobi %1.10E BetaRichardson %1.10E L2eps %1.10E ] ", beta_iJ, beta_iR, L2eps_i);


		kernelAccumulateSummands4 << <numTilesMajor, threadsPerTileMajor >> > (

			// We don't need to test for domain, we need to make sure the summands are zero otherwise.

			p_epsilon_e,
			p_d_eps_by_dbeta_e,
			p_d_eps_by_dbetaR_e,

			// 6 outputs:
			p_sum_eps_deps_by_dbeta_J,
			p_sum_eps_deps_by_dbeta_R,
			p_sum_depsbydbeta_J_times_J,
			p_sum_depsbydbeta_R_times_R,
			p_sum_depsbydbeta_J_times_R,
			p_sum_eps_eps);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands ion");
		
		cudaMemcpy(p_sum_eps_deps_by_dbeta_J_host, p_sum_eps_deps_by_dbeta_J, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_deps_by_dbeta_R_host, p_sum_eps_deps_by_dbeta_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_J_times_J_host, p_sum_depsbydbeta_J_times_J, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_R_times_R_host, p_sum_depsbydbeta_R_times_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_J_times_R_host, p_sum_depsbydbeta_J_times_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);

		sum_eps_deps_by_dbeta_J = 0.0;
		sum_eps_deps_by_dbeta_R = 0.0;
		sum_depsbydbeta_J_times_J = 0.0;
		sum_depsbydbeta_R_times_R = 0.0;
		sum_depsbydbeta_J_times_R = 0.0;
		sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMajor; iTile++)
		{
			sum_eps_deps_by_dbeta_J += p_sum_eps_deps_by_dbeta_J_host[iTile];
			sum_eps_deps_by_dbeta_R += p_sum_eps_deps_by_dbeta_R_host[iTile];
			sum_depsbydbeta_J_times_J += p_sum_depsbydbeta_J_times_J_host[iTile];
			sum_depsbydbeta_R_times_R += p_sum_depsbydbeta_R_times_R_host[iTile];
			sum_depsbydbeta_J_times_R += p_sum_depsbydbeta_J_times_R_host[iTile];
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		}
		 
		if ((sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0)
			|| (sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0))
		{
			printf("Could not define coefficient.");
			beta_eJ = 0.0; beta_eR = 0.0;
		}
		else {
			beta_eJ = -(sum_eps_deps_by_dbeta_J*sum_depsbydbeta_R_times_R - sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_R) /
				(sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
			beta_eR = -(sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_J - sum_eps_deps_by_dbeta_J*sum_depsbydbeta_J_times_R) /
				(sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
		
//			if (((fabs(beta_eJ) < 0.1) && (fabs(beta_eR) < 0.05)) 
//				|| ((iIteration > 100) && (iIteration % 4 != 0))) {
//				beta_eJ = 0.25; beta_eR = 0.0;
//			}	
			// Sometimes snarls things up. Switch back to CG instead.

		};
		
		L2eps_e_old = L2eps_e;
		L2eps_e = sqrt(sum_eps_eps / (real)NUMVERTICES);

		printf("\n for Electron [ BetaJacobi %1.10E BetaRichardson %1.10E L2eps %1.10E ] ", beta_eJ, beta_eR, L2eps_e);

		// ======================================================================================================================
		
		// bringing back adding both at once WOULD be more efficient.

		VectorAddMultiple << <numTilesMajor, threadsPerTileMajor >> > (
			p_Tn, beta_nJ, p_Jacobi_n,
			p_Ti, beta_iJ, p_Jacobi_i,
			p_Te, beta_eJ, p_Jacobi_e);
		Call(cudaThreadSynchronize(), "cudaTS AddtoT ___1");

		VectorAddMultiple << <numTilesMajor, threadsPerTileMajor >> > (
			p_Tn, beta_nR, p_epsilon_n,
			p_Ti, beta_iR, p_epsilon_i,
			p_Te, beta_eR, p_epsilon_e);		
		Call(cudaThreadSynchronize(), "cudaTS AddtoT ___2");
//		kernelAddtoT << <numTilesMajor, threadsPerTileMajor >> > (
//			p_Tn, p_Ti, p_Te, beta_nJ, beta_nR, beta_iJ, beta_iR, beta_eJ, beta_eR,
//			p_Jacobi_n, p_Jacobi_i, p_Jacobi_e,
//			p_epsilon_n, p_epsilon_i, p_epsilon_e);		
		//Call(cudaThreadSynchronize(), "cudaTS AddtoT ___");
	
		iIteration++;

		bFailedTest = false;
		cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMajor, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMajor; iTile++)
			if (p_boolhost[iTile]) bFailedTest = true;
		if (bFailedTest) {
			printf("bFailedTest true \n");
		}
		else {
			printf("bFailedTest false \n");
		}
		
		f64 ratio;
		bProgress = false;
		if ((L2eps_e_old < 1.0e-30) && (L2eps_i_old < 1.0e-30) && (L2eps_n_old < 1.0e-30)) 
		{
			// good enough for no test
			bProgress = true;
		} else {
			if (L2eps_e_old >= 1.0e-30) {
				ratio = (L2eps_e / L2eps_e_old);
				if (ratio < 0.985) bProgress = true; // 1.5% progress enough to carry on
			};
			if (L2eps_i_old >= 1.0e-30) {
				ratio = (L2eps_i / L2eps_i_old);
				if (ratio < 0.985) bProgress = true; // 1.5% progress enough to carry on
			};
			if (L2eps_n_old >= 1.0e-30) {
				ratio = (L2eps_n / L2eps_n_old);
				if (ratio < 0.985) bProgress = true; // 1.5% progress enough to carry on
			};
		};

	} while ((iIteration < 2)
		|| ( (bFailedTest)
		&& ((iIteration < 40) || (bProgress))
			 ) );

	kernelPackupT3 << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_T, p_Tn, p_Ti, p_Te);	 // we did division since we updated NT.
	Call(cudaThreadSynchronize(), "cudaTS packup");

	GlobalSuppressSuccessVerbosity = false;

	return (bFailedTest == false) ? 0 : 1;
}


void RegressionSeedTe(f64 hsub, T3 * p_move1, T3 * p_move2, T3 * p_T, T3 * p_T_k, cuSyst * pX_use)
{

	f64 beta_eJ, beta_iJ, beta_nJ, beta_eR, beta_iR, beta_nR;
	long iTile;
	f64 beta[8];

	f64 sum_eps_deps_by_dbeta_J, sum_eps_deps_by_dbeta_R,
		sum_depsbydbeta_J_times_J, sum_depsbydbeta_R_times_R, sum_depsbydbeta_J_times_R,
		sum_eps_eps;

	long iVertex;
	f64 L2eps_n, L2eps_e, L2eps_i;

	kernelUnpack << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_Tn, p_Ti, p_Te, p_T);
	Call(cudaThreadSynchronize(), "cudaTS unpack");
	// unpack moves to scalars :
	kernelUnpack << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_slot1n, p_slot1i, p_slot1e, p_move1);
	Call(cudaThreadSynchronize(), "cudaTS unpack move1");	
	kernelUnpack << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_slot2n, p_slot2i, p_slot2e, p_move2);
	Call(cudaThreadSynchronize(), "cudaTS unpack move2");

	cudaMemset(p_T, 0, sizeof(T3)*NUMVERTICES); // use as zerovec
	
	// . Create epsilon for p_Tn etc
 	cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
	kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_scalarT << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(
			pX_use->p_info,
			pX_use->p_izNeigh_vert,
			pX_use->p_szPBCneigh_vert,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_cc,
			pX_use->p_n_major,
			p_Tn, p_Ti, p_Te,
			pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
			p_kappa_n,
			p_kappa_i,
			p_kappa_e,
			p_nu_i,
			p_nu_e,
			NT_addition_rates_d_temp,
			pX_use->p_AreaMajor);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

	kernelCreateEpsilon_Heat_for_Jacobi << <numTilesMajor, threadsPerTileMajor >> > (
		// eps = T - (T_k +- h sum kappa dot grad T)
		hsub,
		pX_use->p_info + BEGINNING_OF_CENTRAL,
		p_Tn, p_Ti, p_Te,
		p_T_k,
		NT_addition_rates_d_temp,
		pX_use->p_n_major, // got this
		pX_use->p_AreaMajor, // got this -> N, Nn
		p_epsilon_n, p_epsilon_i, p_epsilon_e
		);
	Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");

	// . Now pretend p_move1 is p_T and enter zerovec for T_k
	// That gives us d_eps_by_d_1

	cudaMemset(p_d_eps_by_dbeta_n, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_d_eps_by_dbeta_i, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_d_eps_by_dbeta_e, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_d_eps_by_dbetaR_n, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_d_eps_by_dbetaR_i, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_d_eps_by_dbetaR_e, 0, sizeof(f64)*NUMVERTICES);

	CallMAC(cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES));
	kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_scalarT << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(
			pX_use->p_info,
			pX_use->p_izNeigh_vert,
			pX_use->p_szPBCneigh_vert,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_cc,
			pX_use->p_n_major,
			p_slot1n, p_slot1i, p_slot1e,
			pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
			p_kappa_n,
			p_kappa_i,
			p_kappa_e,
			p_nu_i,
			p_nu_e,
			NT_addition_rates_d_temp,
			pX_use->p_AreaMajor);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

	kernelCreateEpsilon_Heat_for_Jacobi << <numTilesMajor, threadsPerTileMajor >> > (
		// eps = T - (T_k +- h sum kappa dot grad T)
		hsub,
		pX_use->p_info + BEGINNING_OF_CENTRAL,
		p_slot1n, p_slot1i, p_slot1e,
		p_T, // zerovec
		NT_addition_rates_d_temp,
		pX_use->p_n_major, // got this
		pX_use->p_AreaMajor, // got this -> N, Nn
		p_d_eps_by_dbeta_n, p_d_eps_by_dbeta_i, p_d_eps_by_dbeta_e
		);
	Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat 1");
	
	CallMAC(cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES));
	kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_scalarT << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(
			pX_use->p_info,
			pX_use->p_izNeigh_vert,
			pX_use->p_szPBCneigh_vert,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_cc,
			pX_use->p_n_major,
			p_slot2n, p_slot2i, p_slot2e,
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

	// Note: most are near 1, a few are like 22 or 150.
	kernelCreateEpsilon_Heat_for_Jacobi << <numTilesMajor, threadsPerTileMajor >> > (
		// eps = T - (T_k +- h sum kappa dot grad T)
		// x = -eps/coeffself
		hsub,
		pX_use->p_info + BEGINNING_OF_CENTRAL,
		p_slot2n, p_slot2i, p_slot2e,
		p_T, // zerovec
		NT_addition_rates_d_temp,
		// NEED N = n*AreaMajor
		pX_use->p_n_major, // got this
		pX_use->p_AreaMajor, // got this -> N, Nn

		p_d_eps_by_dbetaR_n, p_d_eps_by_dbetaR_i, p_d_eps_by_dbetaR_e
		);
	Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat 2");

	// We have a 2-regression programmed. Should we have a 3-regression programmed also?
	// Do each species in turn.


	kernelAccumulateSummands4 << <numTilesMajor, threadsPerTileMajor >> > (
		// We don't need to test for domain, we need to make sure the summands are zero otherwise.
		p_epsilon_n,
		p_d_eps_by_dbeta_n,
		p_d_eps_by_dbetaR_n,
		// 6 outputs:
		p_sum_eps_deps_by_dbeta_J,
		p_sum_eps_deps_by_dbeta_R,
		p_sum_depsbydbeta_J_times_J,
		p_sum_depsbydbeta_R_times_R,
		p_sum_depsbydbeta_J_times_R,
		p_sum_eps_eps);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands neut");
	
	cudaMemcpy(p_sum_eps_deps_by_dbeta_J_host, p_sum_eps_deps_by_dbeta_J, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_eps_deps_by_dbeta_R_host, p_sum_eps_deps_by_dbeta_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_depsbydbeta_J_times_J_host, p_sum_depsbydbeta_J_times_J, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_depsbydbeta_R_times_R_host, p_sum_depsbydbeta_R_times_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_depsbydbeta_J_times_R_host, p_sum_depsbydbeta_J_times_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);

	sum_eps_deps_by_dbeta_J = 0.0;
	sum_eps_deps_by_dbeta_R = 0.0;
	sum_depsbydbeta_J_times_J = 0.0;
	sum_depsbydbeta_R_times_R = 0.0;
	sum_depsbydbeta_J_times_R = 0.0;
	sum_eps_eps = 0.0;
	for (iTile = 0; iTile < numTilesMajor; iTile++)
	{
		sum_eps_deps_by_dbeta_J += p_sum_eps_deps_by_dbeta_J_host[iTile];
		sum_eps_deps_by_dbeta_R += p_sum_eps_deps_by_dbeta_R_host[iTile];
		sum_depsbydbeta_J_times_J += p_sum_depsbydbeta_J_times_J_host[iTile];
		sum_depsbydbeta_R_times_R += p_sum_depsbydbeta_R_times_R_host[iTile];
		sum_depsbydbeta_J_times_R += p_sum_depsbydbeta_J_times_R_host[iTile];
		sum_eps_eps += p_sum_eps_eps_host[iTile];
	}

	f64 tempf64 = (real)NUMVERTICES;

	if ((sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0)
		|| (sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0))
	{
		printf("Could not define coefficient.");
		beta_nJ = 0.0;
		beta_nR = 0.0;
	}
	else {
		beta_nJ = -(sum_eps_deps_by_dbeta_J*sum_depsbydbeta_R_times_R - sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_R) /
			(sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
		beta_nR = -(sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_J - sum_eps_deps_by_dbeta_J*sum_depsbydbeta_J_times_R) /
			(sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
	}
	L2eps_n = sqrt(sum_eps_eps / (real)NUMVERTICES);
	printf("\n for neutral [ Beta1 %1.10E Beta2 %1.10E L2eps(old) %1.10E ] ", beta_nJ, beta_nR, L2eps_n);


	kernelAccumulateSummands4 << <numTilesMajor, threadsPerTileMajor >> > (

		// We don't need to test for domain, we need to make sure the summands are zero otherwise.
		p_epsilon_i,
		p_d_eps_by_dbeta_i,
		p_d_eps_by_dbetaR_i,
		// 6 outputs:
		p_sum_eps_deps_by_dbeta_J,
		p_sum_eps_deps_by_dbeta_R,
		p_sum_depsbydbeta_J_times_J,
		p_sum_depsbydbeta_R_times_R,
		p_sum_depsbydbeta_J_times_R,
		p_sum_eps_eps);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands ion");
	
	cudaMemcpy(p_sum_eps_deps_by_dbeta_J_host, p_sum_eps_deps_by_dbeta_J, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_eps_deps_by_dbeta_R_host, p_sum_eps_deps_by_dbeta_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_depsbydbeta_J_times_J_host, p_sum_depsbydbeta_J_times_J, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_depsbydbeta_R_times_R_host, p_sum_depsbydbeta_R_times_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_depsbydbeta_J_times_R_host, p_sum_depsbydbeta_J_times_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);

	sum_eps_deps_by_dbeta_J = 0.0;
	sum_eps_deps_by_dbeta_R = 0.0;
	sum_depsbydbeta_J_times_J = 0.0;
	sum_depsbydbeta_R_times_R = 0.0;
	sum_depsbydbeta_J_times_R = 0.0;
	sum_eps_eps = 0.0;
	for (iTile = 0; iTile < numTilesMajor; iTile++)
	{
		sum_eps_deps_by_dbeta_J += p_sum_eps_deps_by_dbeta_J_host[iTile];
		sum_eps_deps_by_dbeta_R += p_sum_eps_deps_by_dbeta_R_host[iTile];
		sum_depsbydbeta_J_times_J += p_sum_depsbydbeta_J_times_J_host[iTile];
		sum_depsbydbeta_R_times_R += p_sum_depsbydbeta_R_times_R_host[iTile];
		sum_depsbydbeta_J_times_R += p_sum_depsbydbeta_J_times_R_host[iTile];
		sum_eps_eps += p_sum_eps_eps_host[iTile];
	}

	if ((sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0)
		|| (sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0))
	{
		printf("Could not define coefficient.");
		beta_iJ = 0.0; beta_iR = 0.0;
	}
	else {
		beta_iJ = -(sum_eps_deps_by_dbeta_J*sum_depsbydbeta_R_times_R - sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_R) /
			(sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
		beta_iR = -(sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_J - sum_eps_deps_by_dbeta_J*sum_depsbydbeta_J_times_R) /
			(sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
	};

	L2eps_i = sqrt(sum_eps_eps / (real)NUMVERTICES);

	printf("\n for ION [ BetaJacobi %1.10E BetaRichardson %1.10E L2eps(old) %1.10E ] ", beta_iJ, beta_iR, L2eps_i);


	kernelAccumulateSummands4 << <numTilesMajor, threadsPerTileMajor >> > (
		p_epsilon_e,
		p_d_eps_by_dbeta_e,
		p_d_eps_by_dbetaR_e,

		// 6 outputs:
		p_sum_eps_deps_by_dbeta_J,
		p_sum_eps_deps_by_dbeta_R,
		p_sum_depsbydbeta_J_times_J,
		p_sum_depsbydbeta_R_times_R,
		p_sum_depsbydbeta_J_times_R,
		p_sum_eps_eps);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands elec");
	
	cudaMemcpy(p_sum_eps_deps_by_dbeta_J_host, p_sum_eps_deps_by_dbeta_J, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_eps_deps_by_dbeta_R_host, p_sum_eps_deps_by_dbeta_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_depsbydbeta_J_times_J_host, p_sum_depsbydbeta_J_times_J, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_depsbydbeta_R_times_R_host, p_sum_depsbydbeta_R_times_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_depsbydbeta_J_times_R_host, p_sum_depsbydbeta_J_times_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);

	sum_eps_deps_by_dbeta_J = 0.0;
	sum_eps_deps_by_dbeta_R = 0.0;
	sum_depsbydbeta_J_times_J = 0.0;
	sum_depsbydbeta_R_times_R = 0.0;
	sum_depsbydbeta_J_times_R = 0.0;
	sum_eps_eps = 0.0;
	for (iTile = 0; iTile < numTilesMajor; iTile++)
	{
		sum_eps_deps_by_dbeta_J += p_sum_eps_deps_by_dbeta_J_host[iTile];
		sum_eps_deps_by_dbeta_R += p_sum_eps_deps_by_dbeta_R_host[iTile];
		sum_depsbydbeta_J_times_J += p_sum_depsbydbeta_J_times_J_host[iTile];
		sum_depsbydbeta_R_times_R += p_sum_depsbydbeta_R_times_R_host[iTile];
		sum_depsbydbeta_J_times_R += p_sum_depsbydbeta_J_times_R_host[iTile];
		sum_eps_eps += p_sum_eps_eps_host[iTile];
	}

	if ((sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0)
		|| (sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0))
	{
		printf("Could not define coefficient.");
		beta_eJ = 0.0; beta_eR = 0.0;
	}
	else {
		beta_eJ = -(sum_eps_deps_by_dbeta_J*sum_depsbydbeta_R_times_R - sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_R) /
			(sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
		beta_eR = -(sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_J - sum_eps_deps_by_dbeta_J*sum_depsbydbeta_J_times_R) /
			(sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
	};

	L2eps_e = sqrt(sum_eps_eps / (real)NUMVERTICES);

	printf("\n for Electron [ BetaJacobi %1.10E BetaRichardson %1.10E L2eps(old) %1.10E ] ", beta_eJ, beta_eR, L2eps_e);

	// ======================================================================================================================

	// bringing back adding both at once WOULD be more efficient.

	VectorAddMultiple << <numTilesMajor, threadsPerTileMajor >> > (
		p_Tn, beta_nJ, p_slot1n,
		p_Ti, beta_iJ, p_slot1i,
		p_Te, beta_eJ, p_slot1e);
	Call(cudaThreadSynchronize(), "cudaTS AddtoT ___1");

	VectorAddMultiple << <numTilesMajor, threadsPerTileMajor >> > (
		p_Tn, beta_nR, p_slot2n,
		p_Ti, beta_iR, p_slot2i,
		p_Te, beta_eR, p_slot2e);
	Call(cudaThreadSynchronize(), "cudaTS AddtoT ___2");
	
	kernelPackupT3 << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_T, p_Tn, p_Ti, p_Te);	
	Call(cudaThreadSynchronize(), "cudaTS packup");

}




/*
void RunBackwardJLSForHeat_volleys(T3 * p_T_k, T3 * p_T, f64 hsub, cuSyst * pX_use)
{
#define UPLIFT_THRESH 0.25

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

	f64 beta_eJ, beta_iJ, beta_nJ, beta_eR, beta_iR, beta_nR;
	long iTile;

	Matrix_real sum_ROC_products, sum_products_i, sum_products_e;
	f64 sum_eps_deps_by_dbeta_vector[8];
	f64 beta[8];

	f64 sum_eps_deps_by_dbeta_J, sum_eps_deps_by_dbeta_R,
		sum_depsbydbeta_J_times_J, sum_depsbydbeta_R_times_R, sum_depsbydbeta_J_times_R,
		sum_eps_eps;
	
	Vertex * pVertex;
	long iVertex;
	plasma_data * pdata;

	// seed: just set T to T_k.
	cudaMemcpy(p_T, p_T_k, sizeof(T3)*NUMVERTICES, cudaMemcpyDeviceToDevice);

	printf("\nJRLS for heat: ");
	//long iMinor;
	f64 L2eps;// _elec, L2eps_neut, L2eps_ion;
	bool bFailedTest;
	Triangle * pTri;
	f64_vec4 tempf64vec4;
	f64 tempf64;
	int iIteration = 0;
	do {
		printf("\nITERATION %d \n\n", iIteration);
		if ((iIteration >= 500) && (iIteration % 100 == 0))
		{
	 		// draw a graph:

			cudaMemcpy(p_temphost1, p_epsilon_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
			cudaMemcpy(cuSyst_host.p_iVolley, pX_use->p_iVolley, sizeof(char)*NUMVERTICES, cudaMemcpyDeviceToHost);
		//	cudaMemcpy(p_temphost2, p_regressor_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);

			pVertex = pTriMesh->X;
			pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
			for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
			{
				pdata->temp.x = p_temphost1[iVertex];
				pdata->temp.y = p_temphost1[iVertex];

				if (p_temphost1[iVertex] > 1.0e-15) {
					printf("iVertex %d epsilon %1.10E iVolley %d ",
						iVertex, p_temphost1[iVertex],
						cuSyst_host.p_iVolley[iVertex]);
//
					cudaMemcpy(&tempf64vec4, &(p_d_eps_by_dbetaJ_e_x4[iVertex]), sizeof(f64_vec4), cudaMemcpyDeviceToHost);
					printf("deps/dJ %1.6E %1.6E %1.6E %1.6E ", tempf64vec4.x[0], tempf64vec4.x[1], tempf64vec4.x[2], tempf64vec4.x[3]);
//
					cudaMemcpy(&tempf64, &(p_Jacobi_e[iVertex]), sizeof(f64), cudaMemcpyDeviceToHost);
					printf("Jac %1.8E\n", tempf64);
				};
				//if ((pVertex->flags == DOMAIN_VERTEX) || (pVertex->flags == OUTERMOST))
				//{
					
			//	} else {
			//		pdata->temp.x = 0.0;
			//		pdata->temp.y = 0.0;
			//	}
				++pVertex;
				++pdata;
			}

			Graph[0].DrawSurface("epsilon",
				DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
				AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
				false,
				GRAPH_EPSILON, pTriMesh);


			Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);

			printf("done graph");
		}
		// 2. Calculate epsilon: given the est of T, eps = T - (T_k +- h sum kappa dot grad T)
		// 3. Calculate Jacobi: for each point, Jacobi = eps/(deps/dT)

		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);

		kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly << < numTilesMajorClever, threadsPerTileMajorClever >> >
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
				pX_use->p_T_minor + BEGINNING_OF_CENTRAL, // T_k+1/2 or T_k
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
		kernelCalc_SelfCoefficient_for_HeatConduction << < numTilesMajorClever, threadsPerTileMajorClever >> >
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

		cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajor);
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
			p_Jacobi_e,
			p_bFailed
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");



		// This bit is not yet modified per changing to use derivatives at sides.
		// Let's roll back to a JRLS -- simple?
		//

		//memset(d_eps_by_dx_neigh_n,0,sizeof(f64)*NUMVERTICES*MAXNEIGH);
		//memset(d_eps_by_dx_neigh_i, 0, sizeof(f64)*NUMVERTICES*MAXNEIGH);
		//memset(d_eps_by_dx_neigh_e, 0, sizeof(f64)*NUMVERTICES*MAXNEIGH);
		//memset(p_Effect_self_n, 0, sizeof(f64)*NUMVERTICES);
		//memset(p_Effect_self_i, 0, sizeof(f64)*NUMVERTICES);
		//memset(p_Effect_self_e, 0, sizeof(f64)*NUMVERTICES);
		//                   
		//kernelCalculateArray_ROCwrt_my_neighbours << <numTilesMajor, threadsPerTileMajor >> >(
		//	hsub,
		//	pX_use->p_info,
		//	pX_use->p_izNeigh_vert,
		//	pX_use->p_szPBCneigh_vert,
		//	pX_use->p_izTri_vert,
		//	pX_use->p_szPBCtri_vert,
		//	pX_use->p_cc,
		//	pX_use->p_n_major,
	 //	 	pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
		//	p_kappa_n,
	 //		p_kappa_i,
		//	p_kappa_e,
	 //		p_nu_i,
		//	p_nu_e,
		//	pX_use->p_AreaMajor,
		//	  
		//	// Output:
		//	d_eps_by_dx_neigh_n, // save an array of MAXNEIGH f64 values at this location
		//	d_eps_by_dx_neigh_i,
	 //		d_eps_by_dx_neigh_e,
		//	p_Effect_self_n,
		//	p_Effect_self_i,
		//	p_Effect_self_e
		//);
		//Call(cudaThreadSynchronize(), "cudaTS kernelCalculateArray_ROCwrt_my_neighbours");

		//kernelCalculateOptimalMove<<<numTilesMajorClever, threadsPerTileMajorClever>>>(
		//	pX_use->p_info + BEGINNING_OF_CENTRAL,
		//	d_eps_by_dx_neigh_n,
		//	d_eps_by_dx_neigh_i,
		//	d_eps_by_dx_neigh_e,
		//	p_Effect_self_n,
		//	p_Effect_self_i,
		//	p_Effect_self_e,
		//	pX_use->p_izNeigh_vert,

		//	p_epsilon_n,
		//	p_epsilon_i,
		//	p_epsilon_e,
		//	// output:
		//	p_regressor_n,
		//	p_regressor_i,
		//	p_regressor_e
		//);
		//Call(cudaThreadSynchronize(), "cudaTS kernelCalculateOptimalMove");
		//
		//


		// 4. To determine deps/dbeta, ie the change in eps for an addition of delta Jacobi_heat to T,

		// eps = T - (T_k + h dT/dt)
		// deps/dbeta[index] = Jacobi[index] - h d(dT/dt)/d [increment whole field by Jacobi]
		// ####################################
		// Note this last, it's very important.
		
		cudaMemset(p_d_eps_by_dbeta_n, 0, sizeof(f64)*NUMVERTICES);
		cudaMemset(p_d_eps_by_dbeta_i, 0, sizeof(f64)*NUMVERTICES);
		cudaMemset(p_d_eps_by_dbeta_e, 0, sizeof(f64)*NUMVERTICES);


		
		cudaMemset(p_d_eps_by_dbetaJ_n_x4, 0, sizeof(f64_vec4)*NUMVERTICES);
		cudaMemset(p_d_eps_by_dbetaJ_i_x4, 0, sizeof(f64_vec4)*NUMVERTICES);
		cudaMemset(p_d_eps_by_dbetaJ_e_x4, 0, sizeof(f64_vec4)*NUMVERTICES);
		cudaMemset(p_d_eps_by_dbetaR_n_x4, 0, sizeof(f64_vec4)*NUMVERTICES);
		cudaMemset(p_d_eps_by_dbetaR_i_x4, 0, sizeof(f64_vec4)*NUMVERTICES);
		cudaMemset(p_d_eps_by_dbetaR_e_x4, 0, sizeof(f64_vec4)*NUMVERTICES);
		
		kernelCalculateROCepsWRTregressorT_volleys << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			hsub,
			pX_use->p_info, // THIS WAS USED OK
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
			pX_use->p_iVolley,

			p_Jacobi_n,
			p_Jacobi_i,
			p_Jacobi_e,

			p_d_eps_by_dbetaJ_n_x4,
			p_d_eps_by_dbetaJ_i_x4,
			p_d_eps_by_dbetaJ_e_x4  // 4 dimensional

			);
		Call(cudaThreadSynchronize(), "cudaTS kernelCalculateROCepsWRTregressorT WW");


		kernelCalculateROCepsWRTregressorT_volleys << < numTilesMajorClever, threadsPerTileMajorClever >> > (
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
			pX_use->p_iVolley,

			p_epsilon_n,
			p_epsilon_i, // p_regressor_i,
			p_epsilon_e,
			p_d_eps_by_dbetaR_n_x4,
			p_d_eps_by_dbetaR_i_x4,
			p_d_eps_by_dbetaR_e_x4
			);
		Call(cudaThreadSynchronize(), "cudaTS kernelCalculateROCepsWRTregressorT Richardson");



		// 5. Do JLS calcs and update T
		// But we have to do for 3 species: do separately, so scalar arrays.

		kernelAccumulateSummands6 << <numTilesMajor, threadsPerTileMajor >> > (

			// We don't need to test for domain, we need to make sure the summands are zero otherwise.
			p_epsilon_n,

			p_d_eps_by_dbetaJ_n_x4,
			p_d_eps_by_dbetaR_n_x4,

			// 6 outputs:
			p_sum_eps_deps_by_dbeta_J_x4,
			p_sum_eps_deps_by_dbeta_R_x4,
			p_sum_depsbydbeta_8x8,  // not sure if we want to store 64 things in memory?
			p_sum_eps_eps);			// L1 48K --> divide by 256 --> 24 doubles/thread in L1.
		// Better off running through multiple times and doing 4 saves. But it's optimization.
		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands neut");

		cudaMemcpy(p_sum_eps_deps_by_dbeta_J_x4_host, p_sum_eps_deps_by_dbeta_J_x4, sizeof(f64) * 4 * numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_deps_by_dbeta_R_x4_host, p_sum_eps_deps_by_dbeta_R_x4, sizeof(f64) * 4 * numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_8x8_host, p_sum_depsbydbeta_8x8, sizeof(f64) * 8 * 8 * numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);

		sum_ROC_products.Invoke(8); // does set to zero
		
		memset(&sum_eps_deps_by_dbeta_vector, 0, 8 * sizeof(f64));
		//memset(&sum_ROC_products, 0, 8 * 8 * sizeof(f64));
		sum_eps_eps = 0.0;
		
		int i, j;

		for (i = 0; i < 8; i++)
			for (j = 0; j < 8; j++)
				for (iTile = 0; iTile < numTilesMajor; iTile++)
					sum_ROC_products.LU[i][j] += p_sum_depsbydbeta_8x8_host[iTile*8*8+i*8+j];

		for (iTile = 0; iTile < numTilesMajor; iTile++)
		{
			sum_eps_deps_by_dbeta_vector[0] -= p_sum_eps_deps_by_dbeta_J_x4_host[iTile].x[0];
			sum_eps_deps_by_dbeta_vector[1] -= p_sum_eps_deps_by_dbeta_J_x4_host[iTile].x[1];
			sum_eps_deps_by_dbeta_vector[2] -= p_sum_eps_deps_by_dbeta_J_x4_host[iTile].x[2];
			sum_eps_deps_by_dbeta_vector[3] -= p_sum_eps_deps_by_dbeta_J_x4_host[iTile].x[3];
			sum_eps_deps_by_dbeta_vector[4] -= p_sum_eps_deps_by_dbeta_R_x4_host[iTile].x[0];
			sum_eps_deps_by_dbeta_vector[5] -= p_sum_eps_deps_by_dbeta_R_x4_host[iTile].x[1];
			sum_eps_deps_by_dbeta_vector[6] -= p_sum_eps_deps_by_dbeta_R_x4_host[iTile].x[2];
			sum_eps_deps_by_dbeta_vector[7] -= p_sum_eps_deps_by_dbeta_R_x4_host[iTile].x[3];
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		};

		L2eps = sqrt(sum_eps_eps / (real)NUMVERTICES);
		printf("\n for neutral [ L2eps %1.10E ] \n", L2eps);

		for (i = 0; i < 8; i++) {
			printf("{ ");
			for (j = 0; j < 8; j++)
				printf("\t%1.6E ", sum_ROC_products.LU[i][j]);
			printf("\t} { beta%d } ", i);
			if (i == 3) { printf(" = "); }
			else { printf("   "); };
			printf("{ %1.8E }\n", sum_eps_deps_by_dbeta_vector[i]);
			// Or is it minus??
		};

		memset(beta, 0, sizeof(f64) * 8);
		if (L2eps > 1.0e-28) { // otherwise just STOP !
			// 1e-30 is reasonable because 1e-15 * typical temperature 1e-14 = 1e-29.

			// Test for a zero row:
			bool zero_present = false;
			for (i = 0; i < 8; i++)
			{
				f64 sum = 0.0;
				for (j = 0; j < 8; j++)
					sum += sum_ROC_products.LU[i][j];
				if (sum == 0.0) zero_present = true;
			}
			if (zero_present == false) {
				// LU solve:
				sum_ROC_products.LUdecomp();
				// Now ask: 
				// IS THAT MATRIX THE SAME EVERY ITERATION?
				// As long as we do not adjust kappa it is same, right? For each species.

				sum_ROC_products.LUSolve(sum_eps_deps_by_dbeta_vector, beta);
			}
		} else {
			printf("beta === 0\n");
		};
		printf("\n beta: ");
		for (i = 0; i < 8; i++) 
			printf(" %1.8E ", beta[i]);
		printf("\n\n");

		CallMAC(cudaMemcpyToSymbol(beta_n_c, beta, 8 * sizeof(f64)));

		// ======================================================================================================================

		kernelAccumulateSummands6 << <numTilesMajor, threadsPerTileMajor >> > (

			// We don't need to test for domain, we need to make sure the summands are zero otherwise.
			p_epsilon_i,

			p_d_eps_by_dbetaJ_i_x4,
			p_d_eps_by_dbetaR_i_x4,

			// 6 outputs:
			p_sum_eps_deps_by_dbeta_J_x4,
			p_sum_eps_deps_by_dbeta_R_x4,
			p_sum_depsbydbeta_8x8,  // not sure if we want to store 64 things in memory?
			p_sum_eps_eps);			// L1 48K --> divide by 256 --> 24 doubles/thread in L1.
									// Better off running through multiple times and doing 4 saves. But it's optimization.
		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands neut");

		cudaMemcpy(p_sum_eps_deps_by_dbeta_J_x4_host, p_sum_eps_deps_by_dbeta_J_x4, sizeof(f64) * 4 * numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_deps_by_dbeta_R_x4_host, p_sum_eps_deps_by_dbeta_R_x4, sizeof(f64) * 4 * numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_8x8_host, p_sum_depsbydbeta_8x8, sizeof(f64) * 8 * 8 * numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);

		sum_products_i.Invoke(8); // does set to zero

		memset(&sum_eps_deps_by_dbeta_vector, 0, 8 * sizeof(f64));
		sum_eps_eps = 0.0;

		for (i = 0; i < 8; i++)
			for (j = 0; j < 8; j++)
				for (iTile = 0; iTile < numTilesMajor; iTile++)
					sum_products_i.LU[i][j] += p_sum_depsbydbeta_8x8_host[iTile * 8 * 8 + i * 8 + j];

		for (iTile = 0; iTile < numTilesMajor; iTile++)
		{
			sum_eps_deps_by_dbeta_vector[0] -= p_sum_eps_deps_by_dbeta_J_x4_host[iTile].x[0];
			sum_eps_deps_by_dbeta_vector[1] -= p_sum_eps_deps_by_dbeta_J_x4_host[iTile].x[1];
			sum_eps_deps_by_dbeta_vector[2] -= p_sum_eps_deps_by_dbeta_J_x4_host[iTile].x[2];
			sum_eps_deps_by_dbeta_vector[3] -= p_sum_eps_deps_by_dbeta_J_x4_host[iTile].x[3];
			sum_eps_deps_by_dbeta_vector[4] -= p_sum_eps_deps_by_dbeta_R_x4_host[iTile].x[0];
			sum_eps_deps_by_dbeta_vector[5] -= p_sum_eps_deps_by_dbeta_R_x4_host[iTile].x[1];
			sum_eps_deps_by_dbeta_vector[6] -= p_sum_eps_deps_by_dbeta_R_x4_host[iTile].x[2];
			sum_eps_deps_by_dbeta_vector[7] -= p_sum_eps_deps_by_dbeta_R_x4_host[iTile].x[3];
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		};

		L2eps = sqrt(sum_eps_eps / (real)NUMVERTICES);
		printf("\n for ion [ L2eps %1.10E  ]\n ", L2eps);
		
		for (i = 0; i < 8; i++) {
			printf("{ ");
			for (j = 0; j < 8; j++)
				printf("\t%1.6E ", sum_products_i.LU[i][j]);
			printf("\t} { beta%d } ", i);
			if (i == 3) { printf(" = "); }
			else { printf("   "); };
			printf("{ %1.8E }\n", sum_eps_deps_by_dbeta_vector[i]);
			// Or is it minus??
		}; 
		memset(beta, 0, sizeof(f64) * 8);
		if (L2eps > 1.0e-28) { // otherwise just STOP !

		   // Test for a zero row:
			bool zero_present = false;
			for (i = 0; i < 8; i++)
			{
				f64 sum = 0.0;
				for (j = 0; j < 8; j++)
					sum += sum_products_i.LU[i][j];
				if (sum == 0.0) zero_present = true;
			}
			if (zero_present == false) {
				// LU solve:
				sum_products_i.LUdecomp();
				// Now ask: 
				// IS THAT MATRIX THE SAME EVERY ITERATION?
				// As long as we do not adjust kappa it is same, right? For each species.
				sum_products_i.LUSolve(sum_eps_deps_by_dbeta_vector, beta);
			}
		} else {
			printf("beta === 0\n");
		}
		printf("\n beta: ");
		for (i = 0; i < 8; i++)
			printf(" %1.8E ", beta[i]);
		printf("\n\n");

		CallMAC(cudaMemcpyToSymbol(beta_i_c, beta, 8 * sizeof(f64)));


		// ======================================================================================================================

		kernelAccumulateSummands6 << <numTilesMajor, threadsPerTileMajor >> > (

			// We don't need to test for domain, we need to make sure the summands are zero otherwise.
			p_epsilon_e,

			p_d_eps_by_dbetaJ_e_x4,
			p_d_eps_by_dbetaR_e_x4,

			// 6 outputs:
			p_sum_eps_deps_by_dbeta_J_x4,
			p_sum_eps_deps_by_dbeta_R_x4,
			p_sum_depsbydbeta_8x8,  // not sure if we want to store 64 things in memory?
			p_sum_eps_eps);			// L1 48K --> divide by 256 --> 24 doubles/thread in L1.
									// Better off running through multiple times and doing 4 saves. But it's optimization.
		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands e");

		cudaMemcpy(p_sum_eps_deps_by_dbeta_J_x4_host, p_sum_eps_deps_by_dbeta_J_x4, sizeof(f64) * 4 * numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_deps_by_dbeta_R_x4_host, p_sum_eps_deps_by_dbeta_R_x4, sizeof(f64) * 4 * numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_8x8_host, p_sum_depsbydbeta_8x8, sizeof(f64) * 8 * 8 * numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);

		if (iIteration >= 500) {

			cudaMemcpy(p_temphost1, p_epsilon_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost2, p_Jacobi_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost3, p_regressor_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);

			cudaMemcpy(cuSyst_host.p_iVolley, pX_use->p_iVolley, sizeof(char)*NUMVERTICES, cudaMemcpyDeviceToHost);
			FILE * filedbg = fopen("debug1.txt", "w");
			for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
			{
				cudaMemcpy(&tempf64vec4, &(p_d_eps_by_dbetaJ_e_x4[iVertex]), sizeof(f64_vec4), cudaMemcpyDeviceToHost);

				fprintf(filedbg, "iVertex %d eps %1.12E iVolley %d Jacobi %1.9E opti %1.9E d_eps_by_dbeta_J0 %1.12E \n", iVertex, p_temphost1[iVertex], cuSyst_host.p_iVolley[iVertex],
					p_temphost2[iVertex], p_temphost3[iVertex], tempf64vec4.x[0]);
			};
			fclose(filedbg);
			printf("outputted to file\n");
		};

		sum_products_e.Invoke(8); // does set to zero

		memset(&sum_eps_deps_by_dbeta_vector, 0, 8 * sizeof(f64));
		sum_eps_eps = 0.0;

		for (i = 0; i < 8; i++)
			for (j = 0; j < 8; j++)
				for (iTile = 0; iTile < numTilesMajor; iTile++)
					sum_products_e.LU[i][j] += p_sum_depsbydbeta_8x8_host[iTile * 8 * 8 + i * 8 + j];
		
		for (iTile = 0; iTile < numTilesMajor; iTile++)
		{
			sum_eps_deps_by_dbeta_vector[0] -= p_sum_eps_deps_by_dbeta_J_x4_host[iTile].x[0];
			sum_eps_deps_by_dbeta_vector[1] -= p_sum_eps_deps_by_dbeta_J_x4_host[iTile].x[1];
			sum_eps_deps_by_dbeta_vector[2] -= p_sum_eps_deps_by_dbeta_J_x4_host[iTile].x[2];
			sum_eps_deps_by_dbeta_vector[3] -= p_sum_eps_deps_by_dbeta_J_x4_host[iTile].x[3];
			sum_eps_deps_by_dbeta_vector[4] -= p_sum_eps_deps_by_dbeta_R_x4_host[iTile].x[0];
			sum_eps_deps_by_dbeta_vector[5] -= p_sum_eps_deps_by_dbeta_R_x4_host[iTile].x[1];
			sum_eps_deps_by_dbeta_vector[6] -= p_sum_eps_deps_by_dbeta_R_x4_host[iTile].x[2];
			sum_eps_deps_by_dbeta_vector[7] -= p_sum_eps_deps_by_dbeta_R_x4_host[iTile].x[3];
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		};

		L2eps = sqrt(sum_eps_eps / (real)NUMVERTICES);
		printf("\n for elec [ L2eps %1.10E  ]\n ", L2eps);

		if (iIteration > 100) getch();

		for (i = 0; i < 8; i++) {
			printf("{ ");
			for (j = 0; j < 8; j++)
				printf("\t%1.6E ", sum_products_e.LU[i][j]);
			printf("\t} { beta%d } ", i);
			if (i == 3) { printf(" = "); }
			else { printf("   "); };
			printf("{ %1.8E }\n", sum_eps_deps_by_dbeta_vector[i]);
			// Or is it minus??
		};
		memset(beta, 0, sizeof(f64) * 8);
		if (L2eps > 1.0e-28) {

			// Test for a zero row:
			bool zero_present = false;
			for (i = 0; i < 8; i++)
			{
				f64 sum = 0.0;
				for (j = 0; j < 8; j++)
					sum += sum_products_e.LU[i][j];
				if (sum == 0.0) zero_present = true;
			}
			if (zero_present == false) {
				// LU solve:

//				f64 storedbg[8][8];
//				memcpy(storedbg, sum_products_e.LU, sizeof(f64) * 8 * 8);
//
//				FILE * fpdebug = fopen("matrix_e_result.txt", "w");
//				fprintf(fpdebug, "\n");
//				for (i = 0; i < 8; i++)
//				{
//					for (j = 0; j < 8; j++)
//						fprintf(fpdebug, "%1.14E ", sum_products_e.LU[i][j]);
//
//					fprintf(fpdebug, "   |  %1.14E  \n", sum_eps_deps_by_dbeta_vector[i]);
//				};
				sum_products_e.LUdecomp();
				// Now ask: 
				// IS THAT MATRIX THE SAME EVERY ITERATION?

				// As long as we do not adjust kappa it is same, right? For each species.

				sum_products_e.LUSolve(sum_eps_deps_by_dbeta_vector, beta);

				// Compute test vector:
//				f64 result[8];
//				for (i = 0; i < 8; i++)
	//			{
	//				result[i] = 0.0;
	//				for (j = 0; j < 8; j++)
	//					result[i] += storedbg[i][j] * beta[j];
	//			}

		//		for (i = 0; i < 8; i++)
		//			fprintf(fpdebug, " beta %1.14E result %1.14E \n", beta[i], result[i]);
		//		fprintf(fpdebug, "\n");
		//		fclose(fpdebug); // Test
			};
		} else {
			printf("beta === 0\n");
		};

		printf("\n beta: ");
		for (i = 0; i < 8; i++)
			printf(" %1.8E ", beta[i]);
		printf("\n\n");

		CallMAC(cudaMemcpyToSymbol(beta_e_c, beta, 8 * sizeof(f64)));

		kernelAddtoT_volleys << <numTilesMajor, threadsPerTileMajor >> > (
			p_T, pX_use->p_iVolley,
			p_Jacobi_n, p_Jacobi_i, p_Jacobi_e,
			p_epsilon_n, p_epsilon_i, p_epsilon_e); // p_regressor_i

		/*
		if ((sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0)
			|| (sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0))
		{
			printf("Could not define coefficient.");
			beta_nJ = 0.0;
			beta_nR = 0.0;
		} else {
			beta_nJ = -(sum_eps_deps_by_dbeta_J*sum_depsbydbeta_R_times_R - sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_R) /
				(sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
			beta_nR = -(sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_J - sum_eps_deps_by_dbeta_J*sum_depsbydbeta_J_times_R) /
				(sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
		}
		printf("\n for neutral [ BetaJacobi %1.10E BetaRichardson %1.10E L2eps %1.10E ] ", beta_nJ, beta_nR, L2eps_neut);


		kernelAccumulateSummands4 << <numTilesMajor, threadsPerTileMajor >> > (

			// We don't need to test for domain, we need to make sure the summands are zero otherwise.
			p_epsilon_i,
			p_d_eps_by_dbeta_i,
			p_d_eps_by_dbetaR_i,

			// 6 outputs:
			p_sum_eps_deps_by_dbeta_J,
			p_sum_eps_deps_by_dbeta_R,
			p_sum_depsbydbeta_J_times_J,
			p_sum_depsbydbeta_R_times_R,
			p_sum_depsbydbeta_J_times_R,
			p_sum_eps_eps);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands ion");


		cudaMemcpy(p_sum_eps_deps_by_dbeta_J_host, p_sum_eps_deps_by_dbeta_J, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_deps_by_dbeta_R_host, p_sum_eps_deps_by_dbeta_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_J_times_J_host, p_sum_depsbydbeta_J_times_J, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_R_times_R_host, p_sum_depsbydbeta_R_times_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_J_times_R_host, p_sum_depsbydbeta_J_times_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);

		sum_eps_deps_by_dbeta_J = 0.0;
		sum_eps_deps_by_dbeta_R = 0.0;
		sum_depsbydbeta_J_times_J = 0.0;
		sum_depsbydbeta_R_times_R = 0.0;
		sum_depsbydbeta_J_times_R = 0.0;
		sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMajor; iTile++)
		{
			sum_eps_deps_by_dbeta_J += p_sum_eps_deps_by_dbeta_J_host[iTile];
			sum_eps_deps_by_dbeta_R += p_sum_eps_deps_by_dbeta_R_host[iTile];
			sum_depsbydbeta_J_times_J += p_sum_depsbydbeta_J_times_J_host[iTile];
			sum_depsbydbeta_R_times_R += p_sum_depsbydbeta_R_times_R_host[iTile];
			sum_depsbydbeta_J_times_R += p_sum_depsbydbeta_J_times_R_host[iTile];
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		}

		if ((sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0)
			|| (sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0))
		{
			printf("Could not define coefficient.");
			beta_iJ = 0.0; beta_iR = 0.0;
		} else {
			beta_iJ = -(sum_eps_deps_by_dbeta_J*sum_depsbydbeta_R_times_R - sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_R) /
				(sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
			beta_iR = -(sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_J - sum_eps_deps_by_dbeta_J*sum_depsbydbeta_J_times_R) /
				(sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
		};
		L2eps_ion = sqrt(sum_eps_eps / (real)NUMVERTICES);

		printf("\n for ION [ BetaJacobi %1.10E BetaRichardson %1.10E L2eps %1.10E ] ", beta_iJ, beta_iR, L2eps_ion);


		kernelAccumulateSummands4 << <numTilesMajor, threadsPerTileMajor >> > (
			
			// We don't need to test for domain, we need to make sure the summands are zero otherwise.

			p_epsilon_e,
			p_d_eps_by_dbeta_e,
			p_d_eps_by_dbetaR_e,

			// 6 outputs:
			p_sum_eps_deps_by_dbeta_J,
			p_sum_eps_deps_by_dbeta_R,
			p_sum_depsbydbeta_J_times_J,
			p_sum_depsbydbeta_R_times_R,
			p_sum_depsbydbeta_J_times_R,
			p_sum_eps_eps);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands ion");


		cudaMemcpy(p_sum_eps_deps_by_dbeta_J_host, p_sum_eps_deps_by_dbeta_J, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_deps_by_dbeta_R_host, p_sum_eps_deps_by_dbeta_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_J_times_J_host, p_sum_depsbydbeta_J_times_J, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_R_times_R_host, p_sum_depsbydbeta_R_times_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_J_times_R_host, p_sum_depsbydbeta_J_times_R, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);

		sum_eps_deps_by_dbeta_J = 0.0;
		sum_eps_deps_by_dbeta_R = 0.0;
		sum_depsbydbeta_J_times_J = 0.0;
		sum_depsbydbeta_R_times_R = 0.0;
		sum_depsbydbeta_J_times_R = 0.0;
		sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMajor; iTile++)
		{
			sum_eps_deps_by_dbeta_J += p_sum_eps_deps_by_dbeta_J_host[iTile];
			sum_eps_deps_by_dbeta_R += p_sum_eps_deps_by_dbeta_R_host[iTile];
			sum_depsbydbeta_J_times_J += p_sum_depsbydbeta_J_times_J_host[iTile];
			sum_depsbydbeta_R_times_R += p_sum_depsbydbeta_R_times_R_host[iTile];
			sum_depsbydbeta_J_times_R += p_sum_depsbydbeta_J_times_R_host[iTile];
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		}

		if ((sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0)
			|| (sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R == 0.0))
		{
			printf("Could not define coefficient.");
			beta_eJ = 0.0; beta_eR = 0.0;
		}
		else {
			beta_eJ = -(sum_eps_deps_by_dbeta_J*sum_depsbydbeta_R_times_R - sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_R) /
				(sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
			beta_eR = -(sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_J - sum_eps_deps_by_dbeta_J*sum_depsbydbeta_J_times_R) /
				(sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
		};
		L2eps_elec = sqrt(sum_eps_eps / (real)NUMVERTICES);

		printf("\n for Electron [ BetaJacobi %1.10E BetaRichardson %1.10E L2eps %1.10E ] ", beta_eJ, beta_eR, L2eps_elec);
		
		*/



		/*

		kernelAccumulateSummands2 << <numTilesMajor, threadsPerTileMajor >> > (
			pX_use->p_info,
			
				// WHOOPS

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
		if (sum_depsbydbeta_sq == 0.0) {
			beta_i = 1.0;
		} else {
			beta_i = -sum_eps_deps_by_dbeta / sum_depsbydbeta_sq;
		};
		if (beta_i < UPLIFT_THRESH) {
			printf("beta_i %1.10E ", beta_i);
			beta_i = (UPLIFT_THRESH + beta_i) / (2.0 - UPLIFT_THRESH + beta_i); // Try to navigate space instead of getting stuck.
			printf("beta_i after uplift %1.10E \n", beta_i);
		}
		L2eps_ion = sqrt(sum_eps_eps / (real)NUMVERTICES);

		printf("\n for ion [ %1.14E %1.14E ] ", beta_i, L2eps_ion);

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
		if (sum_depsbydbeta_sq == 0.0) {
			beta_e = 1.0;
		} else {
			beta_e = -sum_eps_deps_by_dbeta / sum_depsbydbeta_sq;
		};
		if (beta_e < UPLIFT_THRESH) {
			printf("beta_e %1.10E ", beta_e);
			beta_e = (UPLIFT_THRESH + beta_e) / (2.0 - UPLIFT_THRESH + beta_e);
			printf("beta_e after uplift %1.10E\n", beta_e);
		}
		
		L2eps_elec = sqrt(sum_eps_eps / (real)NUMVERTICES);

		printf("\n for elec [ %1.14E %1.14E ] ", beta_e, L2eps_elec);
		*/
		// maybe do the add after we calc beta_n, beta_i, beta_e.
		
/*
kernelAddtoT << <numTilesMajor, threadsPerTileMajor >> > (
			p_T, beta_nJ, beta_nR, beta_iJ, beta_iR, beta_eJ, beta_eR, 
			p_Jacobi_n, p_Jacobi_i, p_Jacobi_e,
			p_epsilon_n, p_epsilon_i, p_epsilon_e);

		// For some reason, beta calculated was the opposite of the beta that was needed.
		// Don't know why... only explanation is that we are missing a - somewhere in deps/dbeta
		// That's probably what it is? Ought to verify.

		Call(cudaThreadSynchronize(), "cudaTS AddtoT ___");
		*/
/*
		iIteration++;


		// Test that all temperatures will be > 0 and within 1% of putative value
		// WOULD IT NOT be better just to set them to their putative values? We don't know what heat flux that corresponds to?

		// cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajor);
		//kernelTestEpsilon << <numTilesMajor, threadsPerTileMajor >> > (
		//	p_epsilon_n,
		//	p_epsilon_i,
		//	p_epsilon_e,
		//	p_T,
		//	p_bFailed // each thread can set to 1
		//	);
		bFailedTest = false;
		cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMajor, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMajor; iTile++)
			if (p_boolhost[iTile]) bFailedTest = true;
		if (bFailedTest) {
			printf("bFailedTest true \n");
		} else {
			printf("bFailedTest false \n");
		}
	} while ((iIteration < NUM_BWD_ITERATIONS)// || (L2eps_elec > 1.0e-14) || (L2eps_ion > 1.0e-14) || (L2eps_neut > 1.0e-14)
			|| (bFailedTest));
		
}
*/

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

	GlobalSuppressSuccessVerbosity = true;

	f64 beta_e, beta_i, beta_n;
	long iTile;

	// seed: just set T to T_k.
	// No --- assume we were sent the seed.
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
	
	iIteration = 0;
	bool bContinue = true;
	do 
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
		CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMinor));
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
			p_epsilon_ez  ,
			p_bFailed
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
		






		/*

		cudaMemset(NT_addition_tri_d, 0, sizeof(NTrates)*NUMVERTICES * 2);
		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
		cudaMemset(p_MAR_ion2, 0, sizeof(f64_vec3)*NMINOR);
		cudaMemset(p_MAR_elec2, 0, sizeof(f64_vec3)*NMINOR);

		kernelCreate_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> >(

			pX_use->p_info,
			
			p_regressor_1, // regressor 1 means add to vxy, viz, right?

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
		Call(cudaThreadSynchronize(), "cudaTS visccontrib ~~ff");

		kernelCreateEpsilon_Visc << <numTilesMinor, threadsPerTileMinor >> > (
			// eps = v - (v_k +- h [viscous effect])
			// x = -eps/coeffself
			hsub,
			pX_use->p_info,
			p_regressor_1, // vez=0
			zero_vie,
			p_MAR_ion2, p_MAR_elec2,

			pX_use->p_n_minor,
			pX_use->p_AreaMinor,

			p_d_epsilon_xy_dbeta1,
			p_d_epsilon_iz_dbeta1,
			p_d_epsilon_ez_dbeta1,
			b_Failed
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon viscff");



		*/

		// I see that getting rid of the one routine will be inefficient because it means
		// calling for each regressor whereas our existing routine does 2 in 1.
		// So let's go ahead and modify it to match the updated Create viscous then.





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
		printf("\nIteration %d visc: [ beta_i %1.14E beta_e %1.14E L2eps %1.14E ] ", iIteration, beta_i, beta_e, L2eps);
		
		// maybe do the add after we calc beta_n, beta_i, beta_e.
		kernelAdd_to_v << <numTilesMinor, threadsPerTileMinor >> > (
			p_vie, beta_i, beta_e, p_vJacobi_i, p_vJacobi_e);
		Call(cudaThreadSynchronize(), "cudaTS Addtov ___");
			
		int i;
		bContinue = false;
		cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMinor, cudaMemcpyDeviceToHost);
		for (i = 0; ((i < numTilesMinor) && (p_boolhost[i] == false)); i++);
		
		if (i < numTilesMinor) bContinue = true;

		iIteration++;

	} while ((bContinue) && (iIteration < 20));

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
	GlobalSuppressSuccessVerbosity = false;
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
	GlobalSuppressSuccessVerbosity = false;

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
	
	// We are seriously saying that the rate of heat transfer e-n and e-i is
	// basically affected by factor m_e/m_n --- model document says so. ...

	Set_f64_constant(over_m_e, over_m_e_);
	Set_f64_constant(over_m_i, over_m_i_);
	Set_f64_constant(over_m_n, over_m_n_);

	f64 over_sqrt_m_ion_ = 1.0 / sqrt(m_i_);
	f64 over_sqrt_m_e_ = 1.0 / sqrt(m_e_);
	f64 over_sqrt_m_neutral_ = 1.0 / sqrt(m_n_);
	Set_f64_constant(over_sqrt_m_ion, over_sqrt_m_ion_);
	Set_f64_constant(over_sqrt_m_e, over_sqrt_m_e_);
	Set_f64_constant(over_sqrt_m_neutral, over_sqrt_m_neutral_);

	Set_f64_constant(RELTHRESH_AZ_d, RELTHRESH_AZ);

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
//	Call(cudaGetSymbolAddress((void **)(&longaddress), FirstOuterFrill_d),
//		"cudaGetSymbolAddress((void **)(&longaddress), FirstOuterFrill_d)");
//	Call(cudaMemcpy(longaddress, &FirstOuterFrill, sizeof(long), cudaMemcpyHostToDevice),
//		"cudaMemcpy(longaddress, &FirstOuterFrill, sizeof(long), cudaMemcpyHostToDevice)");
// Cannot be used: FirstOuterFrill is not reliable with retiling.

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

	CallMAC(cudaMalloc((void **)&p_regressor_n, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_regressor_i, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_regressor_e, NMINOR * sizeof(f64))); // only need NUMVERTICES but we reused.
	
	CallMAC(cudaMalloc((void **)&p_store_T_move1, NUMVERTICES * sizeof(T3)));
	CallMAC(cudaMalloc((void **)&p_store_T_move2, NUMVERTICES * sizeof(T3)));
	
	CallMAC(cudaMalloc((void **)&p_Tn, NUMVERTICES * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Ti, NUMVERTICES * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Te, NUMVERTICES * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_NnTn, NUMVERTICES * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_NTi, NUMVERTICES * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_NTe, NUMVERTICES * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Ap_n, NUMVERTICES * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Ap_i, NUMVERTICES * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Ap_e, NUMVERTICES * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&zero_array, NUMVERTICES * sizeof(T3)));

	CallMAC(cudaMalloc((void **)&p_Effect_self_n, NUMVERTICES * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Effect_self_i, NUMVERTICES * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Effect_self_e, NUMVERTICES * sizeof(f64)));


	CallMAC(cudaMalloc((void **)&d_eps_by_dx_neigh_n, NUMVERTICES * MAXNEIGH * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&d_eps_by_dx_neigh_i, NUMVERTICES * MAXNEIGH * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&d_eps_by_dx_neigh_e, NUMVERTICES * MAXNEIGH * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&p_boolarray, 2*NUMVERTICES * sizeof(bool)));

	CallMAC(cudaMalloc((void **)&p_nu_major, NUMVERTICES * sizeof(species3)));
	CallMAC(cudaMalloc((void **)&p_was_vertex_rotated, NUMVERTICES * sizeof(char)));
	CallMAC(cudaMalloc((void **)&p_triPBClistaffected, NUMVERTICES * sizeof(char)));
	CallMAC(cudaMalloc((void **)&p_T_upwind_minor_and_putative_T, NMINOR * sizeof(T3)));
	
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
	CallMAC(cudaMalloc((void **)&p_Ax, NMINOR * sizeof(f64)));
	 
	CallMAC(cudaMalloc((void **)&p_sum_eps_deps_by_dbeta, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_depsbydbeta_sq, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_eps_eps, numTilesMinor * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&p_bFailed, numTilesMinor * sizeof(bool)));
	CallMAC(cudaMalloc((void **)&p_eps_against_deps, numTilesMinor * sizeof(f64_vec3)));
		
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

	CallMAC(cudaMalloc((void **)&p_d_eps_by_dbetaR_n, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_dbetaR_i, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_dbetaR_e, NMINOR * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&stored_Az_move, NMINOR * sizeof(f64)));

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

	CallMAC(cudaMalloc((void **)&p_sum_eps_deps_by_dbeta_J, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_eps_deps_by_dbeta_R, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_depsbydbeta_J_times_J, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_depsbydbeta_R_times_R, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_depsbydbeta_J_times_R, numTilesMinor * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&p_d_eps_by_dbetaJ_n_x4, NMINOR * sizeof(f64_vec4)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_dbetaJ_i_x4, NMINOR * sizeof(f64_vec4)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_dbetaJ_e_x4, NMINOR * sizeof(f64_vec4)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_dbetaR_n_x4, NMINOR * sizeof(f64_vec4)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_dbetaR_i_x4, NMINOR * sizeof(f64_vec4)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_dbetaR_e_x4, NMINOR * sizeof(f64_vec4)));

	CallMAC(cudaMalloc((void **)&p_sum_eps_deps_by_dbeta_J_x4, numTilesMinor*sizeof(f64_vec4)));
	CallMAC(cudaMalloc((void **)&p_sum_eps_deps_by_dbeta_R_x4, numTilesMinor * sizeof(f64_vec4)));
	CallMAC(cudaMalloc((void **)&p_sum_depsbydbeta_8x8, numTilesMinor * sizeof(f64) * 8 * 8));
	
	p_sum_eps_deps_by_dbeta_J_x4_host = (f64_vec4 *) malloc(numTilesMinor * sizeof(f64_vec4));
	p_sum_eps_deps_by_dbeta_R_x4_host = (f64_vec4 *)malloc(numTilesMinor * sizeof(f64_vec4));
	p_sum_depsbydbeta_8x8_host = (f64 *)malloc(numTilesMinor * 8 * 8 * sizeof(f64));

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
	p_sum_vec_host = (f64_vec3 *)malloc(numTilesMinor * sizeof(f64_vec3));
	
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

	p_sum_eps_deps_by_dbeta_J_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_sum_eps_deps_by_dbeta_R_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_sum_depsbydbeta_J_times_J_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_sum_depsbydbeta_R_times_R_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_sum_depsbydbeta_J_times_R_host = (f64 *)malloc(numTilesMinor * sizeof(f64));


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
	 
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared); // default!
	 
	cudaFuncSetCacheConfig(kernelCreateShardModelOfDensities_And_SetMajorArea,
		cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(kernelCalculateNu_eHeartNu_iHeart_nu_nn_visc,
		cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(kernelAdvanceDensityAndTemperature,
		cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(kernelPopulateBackwardOhmsLaw,
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

	pX_half->CopyStructuralDetailsFrom(*pX1);
	pX2->CopyStructuralDetailsFrom(*pX1);

	fp_dbg = fopen("dbg1.txt", "a");
	fp_trajectory = fopen("traj.txt", "a");
	
	for (int iRepeat = 0; iRepeat < ADVECT_STEPS_PER_GPU_VISIT; iRepeat++) {
		

		for (iSubstep = 0; iSubstep < ADVECT_FREQUENCY; iSubstep++)
		{
			printf("\nSTEP %d\n-------------\n", iSubstep);
			printf("evaltime = %1.10E \n\n", evaltime);
			
			pX1->PerformCUDA_Advance_noadvect(pX2, pX_half);

			pXtemp = pX1;
			pX1 = pX2;
			pX2 = pXtemp;

			// We need to smoosh the izTri etc data on to the pXhalf and new pX dest systems
			// as it doesn't update any other way and this data will be valid until renewed
			// in the dest system at the end of the step -- riiiiiiiight?
			
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

		printf("Advection step:\n");
		pX1->PerformCUDA_AdvectionCompressionInstantaneous(TIMESTEP*(real)ADVECT_FREQUENCY, pX2, pX_half);

		pXtemp = pX1;
		pX1 = pX2;
		pX2 = pXtemp;


		pX_half->CopyStructuralDetailsFrom(*pX1);
		pX2->CopyStructuralDetailsFrom(*pX1);

	}



	// After an even number of goes, pX1 = &cuSyst1 and this is where we ended up.
		
	fclose(fp_dbg);
	fclose(fp_trajectory);

	cudaEventRecord(stop1, 0);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&elapsedTime, start1, stop1);
	printf("Elapsed time for %d steps : %f ms\n", GPU_STEPS, elapsedTime);

	// update with the most recent B field since we did not update it properly after subcycle:
	cudaMemcpy(pX1->p_B, pX_half->p_B, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);
	// For graphing :
	cudaMemcpy(temp_array_host, p_LapAz, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_OhmsCoeffs_host, p_OhmsCoeffs, sizeof(OhmsCoeffs)*NMINOR, cudaMemcpyDeviceToHost);
	
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
			|| (cuSyst_host.p_T_minor[iMinor].Te > 1.0e-8)) { // thermal velocity 3e9 = 0.1c
			printf("Te = %1.6E %d | ", cuSyst_host.p_T_minor[iMinor].Te, iMinor);
			bSwitch = 1;
		}
		if ((cuSyst_host.p_T_minor[iMinor].Ti < 0.0)
			|| (cuSyst_host.p_T_minor[iMinor].Ti > 1.0e-7)) { // thermal velocity 2e8
			printf("Ti = %1.6E %d | ", cuSyst_host.p_T_minor[iMinor].Ti, iMinor);
			bSwitch = 1;
		}
	};
	if (bSwitch) {
		printf("end, press p\n");  
		while (getch() != 'p');
		PerformCUDA_Revoke();
		exit(3);
	}
	else {
		printf("\nDebugNans OK\n");
	}
}

void cuSyst::PerformCUDA_Advance(//const 
	cuSyst * pX_target,
	//const 
	cuSyst * pX_half) // just being annoying when we put const - but put it back
{
	long iSubcycles, iVertex;
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
	printf("\nDebugNaN this\n\n");
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
	/*
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
	//	printf("iTile %d max_coeffself %1.9E 1over %1.9E \n", iTile, p_temphost2[iTile], h);
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
	//	printf("iTile %d max_coeffself %1.9E 1over %1.9E \n", iTile, p_temphost2[iTile], h);
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
	//	printf("iTile %d max_coeffself %1.9E 1over %1.9E \n", iTile, p_temphost2[iTile], h);
		maxcoeff = max(maxcoeff, p_temphost2[iTile]);
	}
	printf("===============\nELECTRON h_use = %1.10E maxcoeff %1.10E\n================\n", 0.25 / maxcoeff, maxcoeff);
	maxcoeff = max(maxcoeff, max(store_maxcoeff, store_maxcoeff_i));
	 
	// See how far we get with this as timestep.
	
	Timestep = min(0.25 / maxcoeff, TIMESTEP); 
	Timestep = max(Timestep, TIMESTEP*0.5); // min h/2 = 1e-13
	*/
	Timestep = TIMESTEP;


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
		p_T_upwind_minor_and_putative_T);
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
		p_T_upwind_minor_and_putative_T, // may or may not overlap source: don't we only use from tris? so not overlap

		NT_addition_rates_d,
		p_Div_v,
		p_Div_v_neut,
		p_Integrated_div_v_overall);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateAdvectiveMassHeatRate");
	
	if (0) {
		this->SendToHost(cuSyst_host);
		printf("14790: n %1.10E nminor %1.10E nn %1.10E \nTn %1.10E Ti %1.10E Te %1.10E \n"
			"vxy %1.10E %1.10E vn %1.10E %1.10E \n"
			"vez %1.10E B %1.10E %1.10E \n\n",
			cuSyst_host.p_n_major[14790].n, cuSyst_host.p_n_minor[14790 + BEGINNING_OF_CENTRAL].n,
			cuSyst_host.p_n_major[14790].n_n,
			cuSyst_host.p_T_minor[14790 + BEGINNING_OF_CENTRAL].Tn,
			cuSyst_host.p_T_minor[14790 + BEGINNING_OF_CENTRAL].Ti,
			cuSyst_host.p_T_minor[14790 + BEGINNING_OF_CENTRAL].Te,
			cuSyst_host.p_vie[14790 + BEGINNING_OF_CENTRAL].vxy.x, cuSyst_host.p_vie[14790 + BEGINNING_OF_CENTRAL].vxy.y,
			cuSyst_host.p_v_n[14790 + BEGINNING_OF_CENTRAL].x, cuSyst_host.p_v_n[14790 + BEGINNING_OF_CENTRAL].y,
			cuSyst_host.p_vie[14790 + BEGINNING_OF_CENTRAL].vez,
			cuSyst_host.p_B[14790 + BEGINNING_OF_CENTRAL].x, cuSyst_host.p_B[14790 + BEGINNING_OF_CENTRAL].y);
		printf("14791: n %1.10E nminor %1.10E nn %1.10E \nTn %1.10E Ti %1.10E Te %1.10E \n"
			"vxy %1.10E %1.10E vn %1.10E %1.10E \n"
			"vez %1.10E B %1.10E %1.10E \n\n",
			cuSyst_host.p_n_major[14791].n, cuSyst_host.p_n_minor[14791 + BEGINNING_OF_CENTRAL].n,
			cuSyst_host.p_n_major[14791].n_n,
			cuSyst_host.p_T_minor[14791 + BEGINNING_OF_CENTRAL].Tn,
			cuSyst_host.p_T_minor[14791 + BEGINNING_OF_CENTRAL].Ti,
			cuSyst_host.p_T_minor[14791 + BEGINNING_OF_CENTRAL].Te,
			cuSyst_host.p_vie[14791 + BEGINNING_OF_CENTRAL].vxy.x, cuSyst_host.p_vie[14791 + BEGINNING_OF_CENTRAL].vxy.y,
			cuSyst_host.p_v_n[14791 + BEGINNING_OF_CENTRAL].x, cuSyst_host.p_v_n[14791 + BEGINNING_OF_CENTRAL].y,
			cuSyst_host.p_vie[14791 + BEGINNING_OF_CENTRAL].vez,
			cuSyst_host.p_B[14791 + BEGINNING_OF_CENTRAL].x, cuSyst_host.p_B[14791 + BEGINNING_OF_CENTRAL].y);
		printf("14944: n %1.10E nminor %1.10E nn %1.10E \nTn %1.10E Ti %1.10E Te %1.10E \n"
			"vxy %1.10E %1.10E vn %1.10E %1.10E \n"
			"vez %1.10E B %1.10E %1.10E \n\n", cuSyst_host.p_n_major[14944].n,
			cuSyst_host.p_n_major[14944 + BEGINNING_OF_CENTRAL].n,
			cuSyst_host.p_n_major[14944].n_n,
			cuSyst_host.p_T_minor[14944 + BEGINNING_OF_CENTRAL].Tn,
			cuSyst_host.p_T_minor[14944 + BEGINNING_OF_CENTRAL].Ti,
			cuSyst_host.p_T_minor[14944 + BEGINNING_OF_CENTRAL].Te,
			cuSyst_host.p_vie[14944 + BEGINNING_OF_CENTRAL].vxy.x, cuSyst_host.p_vie[14944 + BEGINNING_OF_CENTRAL].vxy.y,
			cuSyst_host.p_v_n[14944 + BEGINNING_OF_CENTRAL].x, cuSyst_host.p_v_n[14944 + BEGINNING_OF_CENTRAL].y,
			cuSyst_host.p_vie[14944 + BEGINNING_OF_CENTRAL].vez,
			cuSyst_host.p_B[14944 + BEGINNING_OF_CENTRAL].x, cuSyst_host.p_B[14944 + BEGINNING_OF_CENTRAL].y);
		printf("Tri 29734: n %1.10E nn %1.10E \nTn %1.10E Ti %1.10E Te %1.10E \n\n",
			cuSyst_host.p_n_minor[29734].n, cuSyst_host.p_n_minor[29734].n_n,
			cuSyst_host.p_T_minor[29734].Tn,
			cuSyst_host.p_T_minor[29734].Ti,
			cuSyst_host.p_T_minor[29734].Te);
	};

	// For now went for "advective change in v" where we divide d/dt Nv by N_derivsyst
	cudaMemset(p_MAR_neut, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_ion, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_elec, 0, sizeof(f64_vec3)*NMINOR);
	 
	//cudaMemcpy(&tempf64, p_n_, sizeof(f64), cudaMemcpyDeviceToHost);

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
	
	//SetConsoleTextAttribute(hConsole, 14);

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
	/*
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


	RunBackwardForHeat_ConjugateGradient(
		this->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices,
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL, // dest
		0.5*Timestep,
		this); // this did have area populated... within ins
	cudaMemcpy(NT_addition_rates_d_temp, NT_addition_rates_d, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToDevice);

	cudaMemset(p_boolarray, 0, sizeof(bool)*NUMVERTICES*2); // initially allow all flows good

	int iPass = 0;
	bool bContinue;
	do {
		printf("iPass %d :\n", iPass);

		// reset NTrates:
		cudaMemcpy(NT_addition_rates_d, NT_addition_rates_d_temp, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToDevice);

		kernelAccumulateDiffusiveHeatRate_new_Full << <numTilesMajorClever, threadsPerTileMajorClever >> > (
			0.5*Timestep,
			this->p_info,
			this->p_izNeigh_vert,
			this->p_szPBCneigh_vert,
			this->p_izTri_vert,
			this->p_szPBCtri_vert,
			this->p_cc,
			this->p_n_major,

			pX_half->p_T_minor + BEGINNING_OF_CENTRAL, // Use T_k+1 just calculated...
			p_boolarray, // array of which ones require longi flows
						 // 2 x NMAJOR
			this->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
			p_kappa_n,
			p_kappa_i,
			p_kappa_e,
			p_nu_i,
			p_nu_e,
			NT_addition_rates_d,
			this->p_AreaMajor,
			(iPass == 0) ? false : true);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate pX_half");
		// To increase the efficiency we want to make a clever 2nd set of major tiles of size 192. Also try 256, 384.

		cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever);
		kernelCreatePutativeT << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			0.5*Timestep,
			this->p_info,
			this->p_T_minor + BEGINNING_OF_CENTRAL,
			//	p_T_upwind_minor_and_putative_T + BEGINNING_OF_CENTRAL, // putative T storage
			this->p_n_major,
			this->p_AreaMajor,
			NT_addition_rates_d,

			p_boolarray, // an array of whether this one requires longi flows --- did it come out T < 0
						 // 2x NMAJOR
			p_bFailed // did we hit any new negative T to add
			);
		Call(cudaThreadSynchronize(), "cudaTS kernelCreatePutativeT");

		bContinue = false;
		cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		int i;
		for (i = 0; ((i < numTilesMajorClever) && (p_boolhost[i] == 0)); i++);
		if (i < numTilesMajorClever) bContinue = true;
		iPass++;
	} while (bContinue);

	 

	kernelIonisationRates<<<numTilesMajor,threadsPerTileMajor>>>(
		0.5*Timestep,
		this->p_info,
		this->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices
		this->p_n_major,
		this->p_AreaMajor,
		NT_addition_rates_d
		);
	Call(cudaThreadSynchronize(), "cudaTS Ionisation");


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

//	SetConsoleTextAttribute(hConsole, 14);
//	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
//	printf("\nNT_addition_rates_d[%d].NeTe %1.10E\n\n", VERTCHOSEN, tempf64);
//	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
//	printf("\nNT_addition_rates_d[%d].NiTi %1.10E\n\n", VERTCHOSEN, tempf64);
//	SetConsoleTextAttribute(hConsole, 15);
	

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
	 
	kernelCreate_neutral_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> >(

		this->p_info,
		this->p_v_n,
		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_izNeigh_TriMinor,
		this->p_szPBC_triminor,

		p_temp6, // ita
		p_temp5,

		p_MAR_neut, // just accumulates
		NT_addition_rates_d,
		NT_addition_tri_d);
	Call(cudaThreadSynchronize(), "cudaTS visccontrib 1");

	SetConsoleTextAttribute(hConsole, 14);
	cudaMemcpy(&tempf64, &(p_MAR_ion[CHOSEN].y), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\np_MAR_ion[%d].y %1.14E\n\n", CHOSEN, tempf64);
	cudaMemcpy(&tempf64, &(p_n_minor[CHOSEN].n), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\np_n_minor[%d].n %1.14E\n\n", CHOSEN, tempf64);
	SetConsoleTextAttribute(hConsole, 15);

	Collect_Nsum_at_tris << <numTriTiles, threadsPerTileMinor >> >(
		this->p_info,
		this->p_n_minor,
		this->p_tri_corner_index,
		this->p_AreaMajor, // populated?
		p_temp4, p_temp5);
	Call(cudaThreadSynchronize(), "cudaTS Nsum");
	 
	kernelTransmitHeatToVerts << <numTilesMajor, threadsPerTileMajor >> > (
		this->p_info,
		this->p_izTri_vert,
		this->p_n_minor,
		this->p_AreaMajor,
		p_temp4, p_temp5,
		NT_addition_rates_d,
		NT_addition_tri_d
		);
	Call(cudaThreadSynchronize(), "cudaTS sum up heat 1");

	SetConsoleTextAttribute(hConsole, 14);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NeTe %1.10E\n\n", VERTCHOSEN, tempf64);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NiTi %1.10E\n\n", VERTCHOSEN, tempf64);
	SetConsoleTextAttribute(hConsole, 15);

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
															 // check T > 0

	cudaMemcpy(cuSyst_host.p_T_minor + BEGINNING_OF_CENTRAL, pX_half->p_T_minor + BEGINNING_OF_CENTRAL, sizeof(T3)*NUMVERTICES, cudaMemcpyDeviceToHost);
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		if ((cuSyst_host.p_T_minor[iVertex + BEGINNING_OF_CENTRAL].Ti < 0.0) || (cuSyst_host.p_T_minor[iVertex + BEGINNING_OF_CENTRAL].Te < 0.0)) {
			printf("iVertex %d Ti %1.9E Te %1.9E \n", iVertex, (cuSyst_host.p_T_minor[iVertex + BEGINNING_OF_CENTRAL].Ti), (cuSyst_host.p_T_minor[iVertex + BEGINNING_OF_CENTRAL].Te));
			getch();
		}
	
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

	//kernelEstimateCurrent << <numTilesMinor, threadsPerTileMinor >> >(
	//	this->p_info,
	//	this->p_n_minor,
	//	this->p_vie,
	//	this->p_AreaMinor,
	//	p_temp1); // Iz_k
	//Call(cudaThreadSynchronize(), "cudaTS Estimate Iz_k"); 
	// not used

	// Get suitable v to use for resistive heating:

	kernelPopulateBackwardOhmsLaw << <numTilesMinor, threadsPerTileMinor >> > (
		0.5*Timestep,
		this->p_info,
		p_MAR_neut, p_MAR_ion, p_MAR_elec,
		this->p_B, // for target it wasn't populated, right? Only used for thermal & nu_effective ?
		p_LapAz,
		p_GradAz,
		p_GradTe,
		this->p_n_minor,
		this->p_T_minor,
		this->p_vie,
		this->p_v_n,
		this->p_AAdot,
		this->p_AreaMinor, 

		p_ROCAzdotduetoAdvection,

		p_vn0,
		p_v0,
		p_OhmsCoeffs,
		p_Iz0_summands,
		p_sigma_Izz,
		p_denom_i, p_denom_e, p_coeff_of_vez_upon_viz, p_beta_ie_z,
		true);
	Call(cudaThreadSynchronize(), "cudaTS AccelerateOhms 1");
	//kernelPopulateOhmsLaw<<<numTilesMinor, threadsPerTileMinor>>>(
	//			
	//	0.5*Timestep,

	//	this->p_info,
	//	p_MAR_neut, p_MAR_ion, p_MAR_elec,
	//	this->p_B,
	//	p_LapAz,
	//	p_GradAz,
	//	p_GradTe,
	//	 
	//	this->p_n_minor,
	//	
	//	p_one_over_n,

	//	this->p_T_minor, // minor : is it populated?
	//	this->p_vie,
	//	this->p_v_n,
	//	this->p_AAdot,
	//	this->p_AreaMinor, // popd?
	//	p_ROCAzdotduetoAdvection,
	//	
	//	p_vn0,
	//	p_v0,
	//	p_OhmsCoeffs,
	//	pX_target->p_AAdot, // intermediate value

	//	p_Iz0_summands,
	//	p_sigma_Izz,
	//	p_denom_i,
	//	p_denom_e, p_coeff_of_vez_upon_viz, p_beta_ie_z,
	//	false,
	//	true, // use this for Iz:
	//	pX_half->p_n_minor
	//	//p_temp5, p_temp6 // vez effect from friction, Ez : iMinor
	//	); 
	//Call(cudaThreadSynchronize(), "cudaTS AccelerateOhms 1");
    
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

	f64 neg_Iz_per_triangle = -(Iz_prescribed_endtime) / (f64)numReverseJzTriangles;
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
		p_LapAz,
		p_ROCAzdotduetoAdvection,
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
	
	//printf("\nDebugNaN pX_half\n\n");
	//DebugNaN(pX_half);

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
		p_T_upwind_minor_and_putative_T
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
		p_T_upwind_minor_and_putative_T, 

		NT_addition_rates_d,
		p_Div_v,
		p_Div_v_neut,
		p_Integrated_div_v_overall);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateAdvectiveMassHeatRate pX_half");

	CallMAC(cudaMemset(p_MAR_neut, 0, sizeof(f64_vec3)*NMINOR));
	CallMAC(cudaMemset(p_MAR_ion, 0, sizeof(f64_vec3)*NMINOR));
	CallMAC(cudaMemset(p_MAR_elec, 0, sizeof(f64_vec3)*NMINOR));

	//FILE * fpdat = fopen("data_out.txt", "a");

	//fprintf(fpdat, "n ");
	//for (i = 0; i < 3; i++) {
	//	cudaMemcpy(&tempf64, &(pX_half->p_n_major[VERTS[i]].n), sizeof(f64), cudaMemcpyDeviceToHost);
	//	fprintf(fpdat, " %1.12E ", tempf64);
	//};

	//fprintf(fpdat, "Ti ");
	//for (i = 0; i < 3; i++) {
	//	cudaMemcpy(&tempf64, &(pX_half->p_T_minor[VERTS[i] + BEGINNING_OF_CENTRAL].Ti), sizeof(f64), cudaMemcpyDeviceToHost);
	//	fprintf(fpdat, " %1.12E ", tempf64);
	//};

	//fprintf(fpdat, "vy ");
	//for (i = 0; i < 3; i++) {
	//	cudaMemcpy(&tempf64, &(pX_half->p_vie[VERTS[i] + BEGINNING_OF_CENTRAL].vxy.y), sizeof(f64), cudaMemcpyDeviceToHost);
	//	fprintf(fpdat, " %1.12E ", tempf64);
	//};

	//fprintf(fpdat, "Compression_NiTi ");
	//for (i = 0; i < 3; i++) {
	//	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTS[i]].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
	//	fprintf(fpdat, " %1.12E ", tempf64);
	//};
	
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

	//fprintf(fpdat, "Pressure MARion ");
	//for (i = 0; i < 3; i++) {
	//	cudaMemcpy(&tempf64, &(p_MAR_ion[VERTS[i] + BEGINNING_OF_CENTRAL].y), sizeof(f64), cudaMemcpyDeviceToHost);
	//	fprintf(fpdat, " %1.12E ", tempf64);
	//};
	//fprintf(fpdat, "Pressure MARelec ");
	//for (i = 0; i < 3; i++) {
	//	cudaMemcpy(&tempf64, &(p_MAR_elec[VERTS[i] + BEGINNING_OF_CENTRAL].y), sizeof(f64), cudaMemcpyDeviceToHost);
	//	fprintf(fpdat, " %1.12E ", tempf64);
	//};

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

	//fprintf(fpdat, "Advective MARion ");
	//for (i = 0; i < 3; i++) {
	//	cudaMemcpy(&tempf64, &(p_MAR_ion[VERTS[i] + BEGINNING_OF_CENTRAL].y), sizeof(f64), cudaMemcpyDeviceToHost);
	//	fprintf(fpdat, " %1.12E ", tempf64);
	//};
	//fprintf(fpdat, "Advective MARelec ");
	//for (i = 0; i < 3; i++) {
	//	cudaMemcpy(&tempf64, &(p_MAR_elec[VERTS[i] + BEGINNING_OF_CENTRAL].y), sizeof(f64), cudaMemcpyDeviceToHost);
	//	fprintf(fpdat, " %1.12E ", tempf64);
	//};

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
	
	// Why were we missing this?
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
	
	RunBackwardForHeat_ConjugateGradient(this->p_T_minor + BEGINNING_OF_CENTRAL,
		pX_target->p_T_minor + BEGINNING_OF_CENTRAL,
		Timestep,
		pX_half);
		
	// Something to know : we never zero "NT_addition_rates" in the routine.
	// So we need to do it outside.
	cudaMemcpy(NT_addition_rates_d_temp, NT_addition_rates_d, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	cudaMemset(p_boolarray, 0, sizeof(bool)*NUMVERTICES*2); // initially allow all flows good

	iPass = 0;
	do {
		printf("iPass %d :\n", iPass);

		// reset NTrates:
		cudaMemcpy(NT_addition_rates_d, NT_addition_rates_d_temp, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToDevice);
		
		kernelAccumulateDiffusiveHeatRate_new_Full << <numTilesMajorClever, threadsPerTileMajorClever >> > (
			Timestep,
			pX_half->p_info,
			pX_half->p_izNeigh_vert,
			pX_half->p_szPBCneigh_vert,
			pX_half->p_izTri_vert,
			pX_half->p_szPBCtri_vert,
			pX_half->p_cc,

			pX_half->p_n_major,
			pX_target->p_T_minor + BEGINNING_OF_CENTRAL, // Use T_k+1 just calculated...
			p_boolarray, // array of which ones require longi flows
			// 2 x NMAJOR
			pX_half->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
			p_kappa_n,
			p_kappa_i,
			p_kappa_e,
			p_nu_i,
			p_nu_e,
			NT_addition_rates_d,
			pX_half->p_AreaMajor,
			(iPass == 0)?false:true);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate pX_half");
		// To increase the efficiency we want to make a clever 2nd set of major tiles of size 192. Also try 256, 384.

		cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever);
		kernelCreatePutativeT << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			Timestep,
			pX_half->p_info,
			this->p_T_minor + BEGINNING_OF_CENTRAL,
		//	p_T_upwind_minor_and_putative_T + BEGINNING_OF_CENTRAL, // putative T storage
			pX_half->p_n_major,
			pX_half->p_AreaMajor,
			NT_addition_rates_d,

			p_boolarray, // an array of whether this one requires longi flows --- did it come out T < 0
			// 2x NMAJOR
			p_bFailed // did we hit any new negative T to add
			);
		Call(cudaThreadSynchronize(), "cudaTS kernelCreatePutativeT");

		bContinue = false;
		cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		int i;
		for (i = 0; ((i < numTilesMajorClever) && (p_boolhost[i] == 0)); i++);
		if (i < numTilesMajorClever) bContinue = true;
		iPass++;
	} while (bContinue);

//
//	fprintf(fpdat, "wConduction_NiTi ");
//	for (i = 0; i < 3; i++) {
//		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTS[i]].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
//		fprintf(fpdat, " %1.12E ", tempf64);
//	};

	//kernelIonisationRates << <numTilesMajor, threadsPerTileMajor >> >(
	//	0.5*Timestep,
	//	pX_half->p_info,
	//	pX_half->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices
	//	pX_half->p_n_major,
	//	pX_half->p_AreaMajor,
	//	NT_addition_rates_d
	//	);
	//Call(cudaThreadSynchronize(), "cudaTS Ionisation pXhalf");

	// 28/09/19: This sometimes fails because only T_k + h NeTe / Ne is guaranteed to be positive.
	
	kernelIonisationRates << <numTilesMajor, threadsPerTileMajor >> >(
		Timestep,
		pX_half->p_info,
		this->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices
		this->p_n_major,
		pX_half->p_AreaMajor,
		NT_addition_rates_d
		);
	Call(cudaThreadSynchronize(), "cudaTS Ionisation pXhalf");

	//fprintf(fpdat, "wIonization_NiTi ");
	//for (i = 0; i < 3; i++) {
	//	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTS[i]].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
	//	fprintf(fpdat, " %1.12E ", tempf64);
	//};
	//fprintf(fpdat, "RateN ");
	//for (i = 0; i < 3; i++) {
	//	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTS[i]].N), sizeof(f64), cudaMemcpyDeviceToHost);
	//	fprintf(fpdat, " %1.12E ", tempf64);
	//};


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

	SetConsoleTextAttribute(hConsole, 14);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NeTe %1.10E\n\n", VERTCHOSEN, tempf64);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NiTi %1.10E\n\n", VERTCHOSEN, tempf64);
	SetConsoleTextAttribute(hConsole, 15);


	::RunBackwardJLSForViscosity(this->p_vie, pX_target->p_vie, Timestep, pX_half);

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
	kernelCreate_neutral_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_info,
		pX_half->p_v_n,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_izNeigh_TriMinor,
		pX_half->p_szPBC_triminor,
		p_temp6, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
		p_temp5, //f64 * __restrict__ p_ita_parallel_elec_minor,   // nT / nu ready to look up
		p_MAR_neut, 
		NT_addition_rates_d,
		NT_addition_tri_d);
	Call(cudaThreadSynchronize(), "cudaTS visccontribneut 2");

	Collect_Nsum_at_tris << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_info,
		pX_half->p_n_minor,
		pX_half->p_tri_corner_index,
		pX_half->p_AreaMajor, // populated?
		p_temp4, p_temp5);
	Call(cudaThreadSynchronize(), "cudaTS Nsum 2");

	kernelTransmitHeatToVerts << <numTilesMajor, threadsPerTileMajor >> > (
		pX_half->p_info,
		pX_half->p_izTri_vert,
		pX_half->p_n_minor,
		pX_half->p_AreaMajor,
		p_temp4,p_temp5,
		NT_addition_rates_d,
		NT_addition_tri_d
		);
	Call(cudaThreadSynchronize(), "cudaTS sum up heat 2");
//
//	fprintf(fpdat, "wViscous MARion ");
//	for (i = 0; i < 3; i++) {
//		cudaMemcpy(&tempf64, &(p_MAR_ion[VERTS[i] + BEGINNING_OF_CENTRAL].y), sizeof(f64), cudaMemcpyDeviceToHost);
//		fprintf(fpdat, " %1.12E ", tempf64);
//	};
//	fprintf(fpdat, "wViscous MARelec ");
//	for (i = 0; i < 3; i++) {
//		cudaMemcpy(&tempf64, &(p_MAR_elec[VERTS[i] + BEGINNING_OF_CENTRAL].y), sizeof(f64), cudaMemcpyDeviceToHost);
//		fprintf(fpdat, " %1.12E ", tempf64);
//	};
//	fprintf(fpdat, "wViscous_NiTi ");
//	for (i = 0; i < 3; i++) {
//		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTS[i]].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
//		fprintf(fpdat, " %1.12E ", tempf64);
//	};
//
//
//	fprintf(fpdat, "Bx ");
//	for (i = 0; i < 3; i++) {
//		cudaMemcpy(&tempf64, &(pX_half->p_B[VERTS[i] + BEGINNING_OF_CENTRAL].x), sizeof(f64), cudaMemcpyDeviceToHost);
//		fprintf(fpdat, " %1.12E ", tempf64);
//	};
//	fprintf(fpdat, "By ");
//	for (i = 0; i < 3; i++) {
//		cudaMemcpy(&tempf64, &(pX_half->p_B[VERTS[i] + BEGINNING_OF_CENTRAL].y), sizeof(f64), cudaMemcpyDeviceToHost);
//		fprintf(fpdat, " %1.12E ", tempf64);
//	};
//	fprintf(fpdat, "viz ");
//	for (i = 0; i < 3; i++) {
//		cudaMemcpy(&tempf64, &(pX_half->p_vie[VERTS[i] + BEGINNING_OF_CENTRAL].viz), sizeof(f64), cudaMemcpyDeviceToHost);
//		fprintf(fpdat, " %1.12E ", tempf64);
//	};
//	fprintf(fpdat, "vez ");
//	for (i = 0; i < 3; i++) {
//		cudaMemcpy(&tempf64, &(pX_half->p_vie[VERTS[i] + BEGINNING_OF_CENTRAL].vez), sizeof(f64), cudaMemcpyDeviceToHost);
//		fprintf(fpdat, " %1.12E ", tempf64);
//	};
//	fprintf(fpdat, "\n");
//	fclose(fpdat);


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
	cudaMemcpy(cuSyst_host.p_T_minor + BEGINNING_OF_CENTRAL, pX_target->p_T_minor + BEGINNING_OF_CENTRAL, sizeof(T3)*NUMVERTICES, cudaMemcpyDeviceToHost);
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		if ((cuSyst_host.p_T_minor[iVertex + BEGINNING_OF_CENTRAL].Ti < 0.0) || (cuSyst_host.p_T_minor[iVertex + BEGINNING_OF_CENTRAL].Te < 0.0)) {
			printf("iVertex %d Ti %1.9E Te %1.9E \n", iVertex, (cuSyst_host.p_T_minor[iVertex + BEGINNING_OF_CENTRAL].Ti), (cuSyst_host.p_T_minor[iVertex + BEGINNING_OF_CENTRAL].Te));
			getch();
		}

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

	// BETTER:
	// Just make this the determinant of how long the overall timestep is;
	// Make supercycle: advection is not usually applied.

	kernelGetLapCoeffs_and_min << <numTriTiles, threadsPerTileMinor >> >(
		pX_target->p_info,
		pX_target->p_izTri_vert,
		pX_target->p_izNeigh_TriMinor,
		pX_target->p_szPBCtri_vert,
		pX_target->p_szPBC_triminor,
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
	// e.g. with this at 7.3e-14, using 1e-13 as substep works (10% backward) ;
	// with it at 6.4e-14 it does not work. 
	// So the inflation factor that you can get away with, isn't huge.
	printf("\nMin coeffself %1.12E iMin %d 1.0/(c sqrt(-mcs)) %1.12E\n", mincoeffself, iMin,
		h_sub_max);
	// Comes out with sensible values for max abs coeff ~~ delta squared?
	// Add in factor to inflate Timestep when we want to play around.

	// iSubcycles = (long)(Timestep / h_sub_max)+1;
	if (Timestep > h_sub_max*2.0) // YES IT IS LESS THAN 1x h_sub_max . Now that seems bad. But we are doing bwd so .. ???
	{
		printf("\nAlert! Timestep > 2.0 h_sub_max %1.11E %1.11E \a\n", Timestep, h_sub_max);
	} else {
		printf("Timestep %1.11E h_sub_max %1.11E \n", Timestep, h_sub_max);
	}
	
	//if (runs % BWD_SUBCYCLE_FREQ == 0) {
	//	printf("backward!\n");
	//	iSubcycles /= BWD_STEP_RATIO; // some speedup this way
	//} else {
	//	iSubcycles *= FWD_STEP_FACTOR;
	//}
	
	// Don't do this stuff --- just make whole step shorter.

	iSubcycles = 1;
	hsub = Timestep / (real)iSubcycles;
		
	printf("hsub = %1.14E subcycles %d \n", hsub, iSubcycles);
//
//	FILE * fptr = fopen("hsub.txt","a");
//	fprintf(fptr, "GlobalStepsCounter %d runs %d hsub %1.14E iSubcycles %d backward %d h_sub_max %1.14E mincoeffself %1.14E \n",
//		GlobalStepsCounter, runs, hsub, iSubcycles, 1, h_sub_max, mincoeffself);
//	fclose(fptr);
//
	// CHANGE TO hsub NOT PROPAGATED TO THE CPU COMPARISON DEBUG ROUTINE!!!
	
	//printf("\nDebugNaN target \n\n");
	//DebugNaN(pX_target);
	
	kernelPullAzFromSyst << <numTilesMinor, threadsPerTileMinor >> >(
		this->p_AAdot,
		p_Az
		);
	Call(cudaThreadSynchronize(), "cudaTS PullAz");

	evaltime += hsub; // t_k+1

	// BACKWARD EULER:

	kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
		pX_target->p_info, // populated position... not neigh_len apparently
		p_Az,
		pX_target->p_izTri_vert,
		pX_target->p_izNeigh_TriMinor,
		pX_target->p_szPBCtri_vert,
		pX_target->p_szPBC_triminor,
		p_LapAz,
		//			p_temp1, p_temp2, p_temp3,
		pX_half->p_AreaMinor // ONLY BECAUSE WE DO NOT KNOW THAT FOR pX_target IT HAS BEEN POPULATED!!!
		);
	Call(cudaThreadSynchronize(), "cudaTS GetLapMinor aaa2");

	Iz_prescribed_endtime = GetIzPrescribed(evaltime); // APPLIED AT END TIME: we are determining

	kernelPopulateBackwardOhmsLaw << <numTilesMinor, threadsPerTileMinor >> > (
		hsub,
		pX_target->p_info,
		p_MAR_neut, p_MAR_ion, p_MAR_elec,
		pX_half->p_B, // for target it wasn't populated, right? Only used for thermal & nu_effective ?
		p_LapAz,
		p_GradAz,
		p_GradTe,
		pX_target->p_n_minor,
		pX_target->p_T_minor,
		this->p_vie,
		this->p_v_n,
		this->p_AAdot,
		pX_half->p_AreaMinor, // NOT POPULATED FOR PXTARGET
		
		p_ROCAzdotduetoAdvection, 
		
		p_vn0,
		p_v0,
		p_OhmsCoeffs,
		p_Iz0_summands,
		p_sigma_Izz,
		p_denom_i, p_denom_e, p_coeff_of_vez_upon_viz, p_beta_ie_z,
		true);		
	Call(cudaThreadSynchronize(), "cudaTS kernelPopulateBackwardOhmsLaw ");


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
	if (EzStrength_ != EzStrength_) { printf("end\n"); while (1) getch(); }
	Set_f64_constant(Ez_strength, EzStrength_);
	 
	neg_Iz_per_triangle = -(Iz_prescribed_endtime) / (f64)numReverseJzTriangles;
	Set_f64_constant(negative_Iz_per_triangle, neg_Iz_per_triangle);
	// Electrons travel from cathode to anode so Jz is down in filament,
	// up around anode.
	printf("\nGPU: Iz0 = %1.14E SigmaIzz %1.14E EzStrength = %1.14E \n\n", Iz0, Sigma_Izz, EzStrength_);

	kernelCreateLinearRelationshipBwd << <numTilesMinor, threadsPerTileMinor >> > (
		hsub,
		pX_target->p_info,
		p_OhmsCoeffs,
		p_v0,
		p_LapAz,  // used for cancelling .. 
		pX_target->p_n_minor,
		p_denom_e,
		p_denom_i, p_coeff_of_vez_upon_viz, p_beta_ie_z, 
		
		this->p_AAdot,  

		pX_half->p_AreaMinor, // because not populated in PXTARGET
		p_Azdot0, 
		p_gamma,
		p_ROCAzdotduetoAdvection
		); // MODIFY vez0, viz0 IF THEY WILL EVER BE USED.
	Call(cudaThreadSynchronize(), "cudaTS kernelCreateLinearRelationshipBwd ");

	kernelCreateSeedAz << <numTilesMinor, threadsPerTileMinor >> > (
		hsub,
		p_Az,
		p_Azdot0,
		p_gamma,
		p_LapAz,
		p_AzNext);
	Call(cudaThreadSynchronize(), "cudaTS Create Seed Az");
	 
//	SolveBackwardAzAdvanceCG(hsub, p_Az, p_Azdot0, p_gamma, 
//		p_AzNext, p_LapCoeffself, pX_target);

	SolveBackwardAzAdvanceJ3LS(hsub, p_Az, p_Azdot0, p_gamma,
				p_AzNext, p_LapCoeffself, pX_target);
	/*
	// JLS alternative:
	f64 sum_eps_deps_by_dbeta, sum_depsbydbeta_sq, sum_eps_eps, depsbydbeta;
	printf("\nJLS [beta L2eps]: ");
	f64 beta, L2eps;
	Triangle * pTri;
	int iIteration = 0;
	 
	//FILE * fpdbg = fopen("JLS.txt", "w");

	long iMax = 0;
	bool bContinue = true;
	do
	{
		printf("iIteration = %d\n", iIteration);
		// 1. Create regressor:
		// Careful with major vs minor + BEGINNING_OF_CENTRAL:

		kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
			pX_target->p_info,
			p_AzNext,
			pX_target->p_izTri_vert,
			pX_target->p_izNeigh_TriMinor,
			pX_target->p_szPBCtri_vert,
			pX_target->p_szPBC_triminor,
			p_LapAzNext,
			pX_target->p_AreaMinor
			);
		Call(cudaThreadSynchronize(), "cudaTS GetLap Az JLS 1");

		cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMinor);
		kernelCreateEpsilonAndJacobi << <numTilesMinor, threadsPerTileMinor >> >
			(hsub, // ?
				pX_target->p_info,
				p_AzNext, p_Az,
				p_Azdot0, p_gamma,
				p_LapCoeffself, p_LapAzNext,
				p_epsilon, p_Jacobi_x,
				//this->p_AAdot, // to use Az_dot_k for equation);
				p_bFailed
				);
		Call(cudaThreadSynchronize(), "cudaTS CreateEpsAndJacobi 1");
		cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMinor, cudaMemcpyDeviceToHost);
		bContinue = false;
		for (iTile = 0; ((p_boolhost[iTile] == 0) && (iTile < numTilesMinor)); iTile++);
		;
		if (iTile < numTilesMinor) {
			printf("failed test\n");
			bContinue = true;
		};
		////else {
		////	// debug:
		////	f64 tempf64_2;
		////	for (iTile = 0; iTile < numTilesMinor; iTile++)
		////	{
		////		cudaMemcpy(&tempf64, &(p_epsilon[iTile*threadsPerTileMinor]), sizeof(f64), cudaMemcpyDeviceToHost);
		////		cudaMemcpy(&tempf64_2, &(p_AzNext[iTile*threadsPerTileMinor]) , sizeof(f64), cudaMemcpyDeviceToHost);
		////		printf("iTile %d first element %1.10E eps %1.10E \n", iTile, tempf64_2, tempf64);
		////	}
		////}

		kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
			this->p_info, this->p_tri_neigh_index, p_Jacobi_x);
		Call(cudaThreadSynchronize(), "cudaTS ResetFrills Jacobi");

		// ensure that we know Az is going to move in the outer frill to negate the move of the tri inside.
		// This will affect the epsilon of the tri inside.
		
		kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
			pX_target->p_info,
			p_Jacobi_x,
			pX_target->p_izTri_vert,
			pX_target->p_izNeigh_TriMinor,
			pX_target->p_szPBCtri_vert,
			pX_target->p_szPBC_triminor,
			p_LapJacobi,
			//		p_temp1, p_temp2, p_temp3,
			pX_half->p_AreaMinor
			);
		Call(cudaThreadSynchronize(), "cudaTS Get Lap Jacobi 1");

		kernelAccumulateSummands << <numTilesMinor, threadsPerTileMinor >> > (
			pX_target->p_info, // ?
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

		cudaMemcpy(p_temphost2, p_Jacobi_x, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_temphost1, p_epsilon, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
		char buffer[255];
//		sprintf(buffer, "JLS_eps%d.txt", iIteration);
//		FILE * jibble = fopen(buffer, "w");
//		for (i = 0; i < NMINOR; i++)
//			fprintf(jibble, "%d epsilon %1.14E \n", i, p_temphost1[i]);
//		fclose(jibble);
//
		kernelAdd << <numTilesMinor, threadsPerTileMinor >> > (
			p_AzNext, beta, p_Jacobi_x);
		
		Call(cudaThreadSynchronize(), "cudaTS Add 1");

		// Try resetting frills here and ignoring in calculation:
		kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
			this->p_info, this->p_tri_neigh_index, p_AzNext);
		Call(cudaThreadSynchronize(), "cudaTS ResetFrills X");
		
		cudaMemcpy(p_temphost1, p_AzNext, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
		sprintf(buffer, "JLS_Az%d.txt", iIteration);
//		jibble = fopen(buffer, "w");
//		for (int i = 0; i < NMINOR; i++)
//			fprintf(jibble, "%d x %1.14E Az %1.14E \n", i, p_temphost2[i], p_temphost1[i]);
//		fclose(jibble);

	//	fprintf(fpdbg, "Beta %1.14E L2eps %1.14E \n", beta, L2eps);

		++iIteration;
		//if (iIteration % 50 == 0)
		//{
		//	cudaMemcpy(p_temphost1, p_epsilon, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
		//	
		//	f64 maxeps = 0.0;
		//	for (iMinor = 0; iMinor < NMINOR; iMinor++)
		//		if (fabs(p_temphost1[iMinor]) > maxeps) {
		//			maxeps = fabs(p_temphost1[iMinor]);
		//			iMax = iMinor;
		//		};

		//	printf("iMax %d eps %1.10E \n", iMax, p_temphost1[iMax]);
		//	getch();
		//};
		//if (iIteration > 50) {
		//	f64 regressor, Az;
		//	cudaMemcpy(&tempf64, &(p_epsilon[iMax]), sizeof(f64), cudaMemcpyDeviceToHost);
		//	cudaMemcpy(&regressor, &(p_Jacobi_x[iMax]), sizeof(f64), cudaMemcpyDeviceToHost);
		//	cudaMemcpy(&Az, &(p_AzNext[iMax]), sizeof(f64), cudaMemcpyDeviceToHost);
		//	printf("eps %1.10E regressor %1.10E Az %1.10E \n", tempf64, regressor, Az);
		//	// chart how changing.
		//	getch();
		//}

	} while (bContinue);
	*/
	//fclose(fpdbg);
	
	printf("done solve\n");

	cudaMemcpy(p_Az, p_AzNext, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);

	kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
		pX_target->p_info,
		p_Az,
		pX_target->p_izTri_vert,
		pX_target->p_izNeigh_TriMinor,
		pX_target->p_szPBCtri_vert,
		pX_target->p_szPBC_triminor,
		p_LapAz,
		pX_half->p_AreaMinor
		);
	Call(cudaThreadSynchronize(), "cudaTS GetLap Az JLS 2");

	// Lap Az is now known, let's say.
	// So we are again going to call PopOhms Backward -- but this time we do not wish to save off stuff
	// except for the v(Ez) relationship.

	kernelPopulateBackwardOhmsLaw << <numTilesMinor, threadsPerTileMinor >> > (
		hsub,// ROCAzdotduetoAdvection, 
		pX_target->p_info,
		p_MAR_neut, p_MAR_ion, p_MAR_elec,
		pX_half->p_B,
		p_LapAz,
		p_GradAz, // THIS WE OUGHT TO TWEEN AT LEAST
		p_GradTe,
		pX_target->p_n_minor,
		pX_target->p_T_minor,

		this->p_vie,
		this->p_v_n,
		this->p_AAdot,
		pX_half->p_AreaMinor, // pop'd? interp?
		p_ROCAzdotduetoAdvection,

		p_vn0,
		p_v0,
		p_OhmsCoeffs,
		
		p_Iz0_summands,
		p_sigma_Izz,
		p_denom_i,
		p_denom_e, p_coeff_of_vez_upon_viz, p_beta_ie_z,
		false);
	Call(cudaThreadSynchronize(), "cudaTS PopBwdOhms II ");

	// Might as well recalculate Ez_strength again :
	// Iz already set for t+hsub.
	cudaMemcpy(p_Iz0_summands_host, p_Iz0_summands, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_summands_host, p_sigma_Izz, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	Iz0 = 0.0;
	Sigma_Izz = 0.0;
	Iz_k = 0.0;
	for (iBlock = 0; iBlock < numTilesMinor; iBlock++)
	{
		Iz0 += p_Iz0_summands_host[iBlock];
		Sigma_Izz += p_summands_host[iBlock];
	}
	EzStrength_ = (Iz_prescribed_endtime - Iz0) / Sigma_Izz;
	Set_f64_constant(Ez_strength, EzStrength_);

	neg_Iz_per_triangle = -(Iz_prescribed_endtime) / (f64)numReverseJzTriangles;
	Set_f64_constant(negative_Iz_per_triangle, neg_Iz_per_triangle);
	// Electrons travel from cathode to anode so Jz is down in filament,
	// up around anode.

	if (EzStrength_ != EzStrength_) {
		printf("EzStrength_ %1.10E Iz_prescribed %1.10E Iz0 %1.10E sigma_Izz %1.10E \n",
			EzStrength_, Iz_prescribed_endtime, Iz0, sigma_Izz);
		while (1) getch();
	}

	kernelCalculateVelocityAndAzdot << <numTilesMinor, threadsPerTileMinor >> > (
		hsub,
		pX_target->p_info,
		p_vn0,
		p_v0,
		p_OhmsCoeffs,
		this->p_AAdot,
		//(iSubstep == iSubcycles - 1) ? pX_target->p_n_minor:pX_half->p_n_minor,
		pX_target->p_n_minor, // NOT OKAY FOR IT TO NOT BE SAME n AS USED THROUGHOUT BY OHMS LAW
		pX_half->p_AreaMinor,  // Still because pXTARGET Area still not populated

		// We need to go back through, populate AreaMinor before we do all these things.
		// Are we even going to be advecting points every step?
		// Maybe make advection its own thing.
		p_LapAz,

		p_ROCAzdotduetoAdvection,

		pX_target->p_AAdot,
		pX_target->p_vie,
		pX_target->p_v_n
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelUpdate_v ");


	// ROC due to advection?

	// Can we take advection out of typical step?

	// Can we make advection-compression its own separate business? Max frequency 1e-12, generally 1e-11 will do until things are 1e7 cm/s on cells 1e-3 wide.

	// ????????????????????????????????????????????????????????????????????????????????
	// NOT CLEAR WHETHER IT IS PREFERABLE HERE TO WRITE Az_k+1 = Az_k + h Azdot_k+1
	// It should be pretty close on what we solved for however???

	kernelAdvanceAzBwdEuler << <numTilesMinor, threadsPerTileMinor >> > (
		hsub,
		this->p_AAdot,
		pX_target->p_AAdot,
		p_ROCAzduetoAdvection);
	Call(cudaThreadSynchronize(), "cudaTS kernelAdvanceAzBwdEuler ");


	// Making separate advection step is pretty important,
	// Including for optimization.
	// Also for validity of this bit.
	// It is also pretty daft having advection take place at 1e-13 when v = 1e6 to 1e7 and delta > 5e-3.
	// We could comfortably do separate instantaneous lunges. The cross-terms are surely not important.
	// ???
	// Soak and ionize only happen on relatively long timescale similar to advection, so yeah.
	// In fact those are also things that could go on 1e-12.


	/*

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




		//	FILE * fofofo = fopen("lap1.txt", "w");
		//	for (iMinor = 0; iMinor < NMINOR; iMinor++)
		//		fprintf(fofofo, "%d Az %1.15E Lap %1.15E Area %1.15E \n", iMinor, p_temphost3[iMinor], p_temphost1[iMinor], p_temphost2[iMinor]);
		//	fclose(fofofo);
		//	printf("done file\n");
		//	getch(); getch();

			// Leaving Iz_prescribed and reverse_Jz the same:

			// Think I'm right all that has changed is LapAz so do we really have to go through whole thing again? :

			//	this->Accelerate2018(hsub, pX_half, pDestMesh, evaltime + 0.5*hsub, false); // Lap Az now given.

			////			kernelPreUpdateAzdot(
			////				hsub,
			////				pX_half->p_info,
			////				pX_half->p_AAdot,
			////				p_ROCAzdotduetoAdvection,
			////				p_LapAz,
			////				pX_half->p_n_minor,
			////				(iSubstep == 0) ? this->p_vie : pX_target->p_vie,
			////				p_temp1
			////				);
			////			Call(cudaThreadSynchronize(), "cudaTS Pre update --");
			////
			//  So now we have to go and sort out how to apply this TotalEffect alongside 1/2 I_k+1

		
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



		////	// DEBUG:
		////	kernelEstimateCurrent << <numTilesMinor, threadsPerTileMinor >> > (
		////		pX_half->p_info,
		////		pX_half->p_n_minor,
		////		pX_target->p_vie,
		////		pX_half->p_AreaMinor,
		////		p_temp1); // Iz_k+1
		////	Call(cudaThreadSynchronize(), "cudaTS Estimate Iz_k+1");
		////	cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		////	Iz_k = 0.0;
		////	for (iBlock = 0; iBlock < numTilesMinor; iBlock++)
		////	{
		////		Iz_k += p_temphost1[iBlock];
		////	}
		////	printf("Iz_k+1 attained %1.14E prescribed %1.14E \n", Iz_k, Iz_prescribed_endtime);

		////	// On last step we see a difference: sum Azdot inexplicably nonzero
		////	// So something is still going wrong.

		////	
		////	Estimate_Effect_on_Integral_Azdot_from_Jz_and_LapAz << <numTilesMinor, threadsPerTileMinor >> > (
		////		hsub,
		////		pX_half->p_info,
		////		pX_half->p_n_minor,
		////		pX_half->p_n_minor, // used in UpdateVelocity
		////		this->p_vie, // save off the old value to (this) first
		////		pX_target->p_vie,
		////		pX_half->p_AreaMinor,
		////		p_LapAz, 

		////		pX_target->p_AAdot, // we'll be able to look at row to see what came just before...

		////		p_temp1, // +ve Jz
		////		p_temp2, // -ve Jz
		////		p_temp3, // LapAz*AreaMinor
		////		p_temp4, 
		////		p_temp5,
		////		p_temp6
		////		);
		////	Call(cudaThreadSynchronize(), "cudaTS Estimate Effect");

		////	cudaMemcpy(p_temphost1, p_temp1, numTilesMinor * sizeof(f64), cudaMemcpyDeviceToHost);
		////	cudaMemcpy(p_temphost2, p_temp2, numTilesMinor * sizeof(f64), cudaMemcpyDeviceToHost);
		////	cudaMemcpy(p_temphost3, p_temp3, numTilesMinor * sizeof(f64), cudaMemcpyDeviceToHost);
		////	cudaMemcpy(p_temphost4, p_temp4, numTilesMinor * sizeof(f64), cudaMemcpyDeviceToHost);
		////	cudaMemcpy(p_temphost5, p_temp5, numTilesMinor * sizeof(f64), cudaMemcpyDeviceToHost);
		////	cudaMemcpy(p_temphost6, p_temp6, numTilesMinor * sizeof(f64), cudaMemcpyDeviceToHost);

		////	sum_plus = 0.0;   sum_minus = 0.0; 	sum_Lap = 0.0;
		////	abs_Lap = 0.0;   sum_Azdot = 0.0; abs_Azdot = 0.0;

		////	for (i = 0; i < numTilesMinor; i++)
		////	{
		////		sum_plus += p_temphost1[i];
		////		sum_minus += p_temphost2[i];
		////		sum_Lap += p_temphost3[i];
		////		abs_Lap += p_temphost4[i];
		////		sum_Azdot += p_temphost5[i];
		////		abs_Azdot += p_temphost6[i];
		////	}
		////	printf("\n\n");
		////	
		////	//printf("sum_plus %1.9E net %1.9E sum_Lap %1.9E sum_diff %1.9E \n",
		////	//	sum_plus, sum_minus + sum_plus, sum_Lap, sum_diff);

		////	fprintf(fp_dbg, "substep %d half-step: sum_plus %1.15E sum_minus %1.15E sum_Lap %1.15E abs_Lap %1.15E sum_Azdot %1.15E abs_Azdot %1.15E \n",
		////		iSubstep, sum_plus, sum_minus, sum_Lap, abs_Lap, sum_Azdot, abs_Azdot);

		////	printf("substep %d half - step: sum_plus %1.15E sum_minus %1.15E sum_Lap %1.15E abs_Lap %1.15E sum_Azdot %1.15E abs_Azdot %1.15E \n",
		////		iSubstep, sum_plus, sum_minus, sum_Lap, abs_Lap, sum_Azdot, abs_Azdot);
		////
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


	*/

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
	//cudaMemcpy(temp_array_host, p_LapAz, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
	//cudaMemcpy(p_OhmsCoeffs_host, p_OhmsCoeffs, sizeof(OhmsCoeffs)*NMINOR, cudaMemcpyDeviceToHost);
	//cudaMemcpy(p_temphost3, pX_half->p_AreaMinor, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);

	//f64 integral_lap = 0.0;
	//f64 integral_L1 = 0.0;
	//for (iMinor = 0; iMinor < NMINOR; iMinor++)
	//{
	//	integral_lap += temp_array_host[iMinor] * p_temphost3[iMinor];
	//	integral_L1 += fabs(temp_array_host[iMinor] * p_temphost3[iMinor]);
	//}
	//printf("Integral Lap Az %1.10E integ |Lap| %1.10E \n", integral_lap, integral_L1);
	//integral_lap = 0.0;
	//integral_L1 = 0.0;
	//for (iMinor = BEGINNING_OF_CENTRAL; iMinor < NMINOR; iMinor++)
	//{
	//	integral_lap += temp_array_host[iMinor] * p_temphost3[iMinor];
	//	integral_L1 += fabs(temp_array_host[iMinor] * p_temphost3[iMinor]);
	//}
	//printf("Verts Integral Lap Az %1.10E integ |Lap| %1.10E \n", integral_lap, integral_L1);
	//integral_lap = 0.0;
	//integral_L1 = 0.0;
	//for (iMinor = BEGINNING_OF_CENTRAL+11000; iMinor < NMINOR; iMinor++)
	//{
	//	integral_lap += temp_array_host[iMinor] * p_temphost3[iMinor];
	//	integral_L1 += fabs(temp_array_host[iMinor] * p_temphost3[iMinor]);
	//}
	//printf("Verts 11000+ Integral Lap Az %1.10E integ |Lap| %1.10E \n", integral_lap, integral_L1);

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

void cuSyst::PerformCUDA_AdvectionCompressionInstantaneous(//const 
	f64 const Timestep,
	cuSyst * pX_target,
	cuSyst * pX_half) // just being annoying when we put const - but put it back
{
	long iSubcycles, iVertex;
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
		p_T_upwind_minor_and_putative_T);
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
		p_T_upwind_minor_and_putative_T, // may or may not overlap source: don't we only use from tris? so not overlap

		NT_addition_rates_d,
		p_Div_v,
		p_Div_v_neut,
		p_Integrated_div_v_overall);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateAdvectiveMassHeatRate");

	// For now went for "advective change in v" where we divide d/dt Nv by N_derivsyst
	cudaMemset(p_MAR_neut, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_ion, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_elec, 0, sizeof(f64_vec3)*NMINOR);

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

	kernelNeutral_momflux << <numTriTiles, threadsPerTileMinor >> >(
		this->p_info,

		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_izNeigh_TriMinor,
		this->p_szPBC_triminor,
		this->p_who_am_I_to_corner,
		this->p_tri_corner_index,

		this->p_v_n,
		p_n_shards_n,
		this->p_n_minor,

		this->p_v_overall_minor,
		p_MAR_neut
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelNeutral_pressure_and_momflux");

	long i;
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

	for (long iIterate = 0; iIterate < 1; iIterate++)
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
	
	cudaMemset(NT_addition_tri_d, 0, sizeof(NTrates)*NUMVERTICES * 2);
	
	// We need to change this to not do soak:
	
	kernelAdvanceDensityAndTemperature_nosoak_etc << <numTilesMajor, threadsPerTileMajor >> >(
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
	Call(cudaThreadSynchronize(), "cudaTS Advance_n_and_T"); // vertex
														 // check T > 0

	cudaMemcpy(cuSyst_host.p_T_minor + BEGINNING_OF_CENTRAL, pX_half->p_T_minor + BEGINNING_OF_CENTRAL, sizeof(T3)*NUMVERTICES, cudaMemcpyDeviceToHost);
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		if ((cuSyst_host.p_T_minor[iVertex + BEGINNING_OF_CENTRAL].Ti < 0.0) || (cuSyst_host.p_T_minor[iVertex + BEGINNING_OF_CENTRAL].Te < 0.0)) {
			printf("iVertex %d Ti %1.9E Te %1.9E \n", iVertex, (cuSyst_host.p_T_minor[iVertex + BEGINNING_OF_CENTRAL].Ti), (cuSyst_host.p_T_minor[iVertex + BEGINNING_OF_CENTRAL].Te));
			getch();
		}

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


	kernelAccelerate_v_from_advection << <numTilesMinor, threadsPerTileMinor >> >
		(   
			0.5*Timestep,
			this->p_info,
			this->p_n_minor,    // multiply by old mass ..
			pX_half->p_n_minor, // divide by new mass ..
			this->p_vie,
			this->p_v_n,

			p_MAR_neut, // these contain the mom flux due to advection.
			p_MAR_ion,
			p_MAR_elec,
			pX_half->p_AreaMinor, 
			// WAS THIS POP'D BY A CALL SUCH AS GetLapMinor?

			// outputs:
			pX_half->p_vie,
			pX_half->p_v_n
			);
	Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities pX_half");

	// get grad Az, Azdot and anti-advect:

	// skip it!!

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

	kernelAdvectPositionsTris << <numTilesMinor, threadsPerTileMinor >> >(
		Timestep,
		this->p_info,
		pX_target->p_info,
		pX_half->p_v_overall_minor
		);
	Call(cudaThreadSynchronize(), "cudaTS AdvectPositions_CopyTris 22");

	SetConsoleTextAttribute(hConsole, 15);

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
		p_T_upwind_minor_and_putative_T
		);
	Call(cudaThreadSynchronize(), "cudaTS CalculateUpwindDensity_tris pX_half");

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
		p_T_upwind_minor_and_putative_T,

		NT_addition_rates_d,
		p_Div_v,
		p_Div_v_neut,
		p_Integrated_div_v_overall);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateAdvectiveMassHeatRate pX_half");

	CallMAC(cudaMemset(p_MAR_neut, 0, sizeof(f64_vec3)*NMINOR));
	CallMAC(cudaMemset(p_MAR_ion, 0, sizeof(f64_vec3)*NMINOR));
	CallMAC(cudaMemset(p_MAR_elec, 0, sizeof(f64_vec3)*NMINOR));

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


	kernelNeutral_momflux << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_info,

		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_izNeigh_TriMinor,
		pX_half->p_szPBC_triminor,
		pX_half->p_who_am_I_to_corner,
		pX_half->p_tri_corner_index,

		pX_half->p_v_n,
		p_n_shards_n,
		pX_half->p_n_minor,

		pX_half->p_v_overall_minor,
		p_MAR_neut
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelNeutral_pressure_and_momflux pX_half");
	
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

	for (long iIterate = 0; iIterate < 1; iIterate++)
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


	kernelAdvanceDensityAndTemperature_nosoak_etc << <numTilesMajor, threadsPerTileMajor >> >(
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

	cudaMemcpy(cuSyst_host.p_T_minor + BEGINNING_OF_CENTRAL, pX_target->p_T_minor + BEGINNING_OF_CENTRAL, sizeof(T3)*NUMVERTICES, cudaMemcpyDeviceToHost);
	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		if ((cuSyst_host.p_T_minor[iVertex + BEGINNING_OF_CENTRAL].Ti < 0.0) || (cuSyst_host.p_T_minor[iVertex + BEGINNING_OF_CENTRAL].Te < 0.0)) {
			printf("iVertex %d Ti %1.9E Te %1.9E \n", iVertex, (cuSyst_host.p_T_minor[iVertex + BEGINNING_OF_CENTRAL].Ti), (cuSyst_host.p_T_minor[iVertex + BEGINNING_OF_CENTRAL].Te));
			getch();
		}

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

	cudaEventRecord(middle, 0);
	cudaEventSynchronize(middle);

	// BETTER:
	// Just make this the determinant of how long the overall timestep is;
	// Make supercycle: advection is not usually applied.
	
	SetConsoleTextAttribute(hConsole, 15);
	
	//this->AntiAdvect(h, pX_half, IntegratedGradAz, pDestMesh); // Might as well subsume this in Accelerate, really
	//pX_half->AntiAdvect(h*0.5, pDestMesh, GradAz, pDestMesh);

	kernelAccelerate_v_from_advection << <numTilesMinor, threadsPerTileMinor >> >
		(
			Timestep,
			this->p_info,
			this->p_n_minor,    // multiply by old mass ..
			pX_target->p_n_minor, // divide by new mass ..
			this->p_vie,
			this->p_v_n, // v_k

			p_MAR_neut, // these contain the mom flux due to advection.
			p_MAR_ion,
			p_MAR_elec,
			pX_target->p_AreaMinor,
			// WAS THIS POP'D BY A CALL SUCH AS GetLapMinor?

			// outputs:
			pX_target->p_vie,
			pX_target->p_v_n
			);
	Call(cudaThreadSynchronize(), "cudaTS Accelerate_v_from_advection");

	// get grad Az, Azdot and anti-advect:
	 
	kernelAntiAdvect << <numTriTiles, threadsPerTileMinor >> >(
		Timestep,
		this->p_info, 
		
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_izNeigh_TriMinor,
		pX_half->p_szPBC_triminor,

		this->p_AAdot,
		pX_half->p_v_overall_minor, // speed of move of this point
		pX_target->p_AAdot // for output
	);
	Call(cudaThreadSynchronize(), "cudaTS kernelAntiAdvect ");
	 
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

	SetConsoleTextAttribute(hConsole, 13);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Elapsed time advection : %f ms ", elapsedTime);
	SetConsoleTextAttribute(hConsole, 15);

	runs++;
}

void cuSyst::PerformCUDA_Advance_noadvect(//const 
	cuSyst * pX_target,
	//const 
	cuSyst * pX_half) // just being annoying when we put const - but put it back
{
	long iSubcycles, iVertex;
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
	static int iHistory = 0;

	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

	cudaEvent_t start, stop, middle;

	//cudaEvent_t start_heat1, end_heat1, start_heat2, end_heat2,
	//			start_visc1, end_visc1, start_visc2, end_visc2;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&middle);
	cudaEventRecord(start, 0);
	cudaEventSynchronize(start);

#define USE_N_MAJOR_FOR_VERTEX 

	cudaMemset(p_Integrated_div_v_overall, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Div_v_neut, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Div_v, 0, sizeof(f64)*NUMVERTICES);

	// DEBUG:
	printf("\nDebugNaN this\n\n");
	DebugNaN(this); 

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
	
	Timestep = TIMESTEP;
	
	cudaMemset(NT_addition_rates_d, 0, sizeof(NTrates)*NUMVERTICES);

	// For now went for "advective change in v" where we divide d/dt Nv by N_derivsyst
	cudaMemset(p_MAR_neut, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_ion, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_elec, 0, sizeof(f64_vec3)*NMINOR);

	kernelCreate_pressure_gradT_and_gradA_CurlA_minor_noadvect << <numTriTiles, threadsPerTileMinor >> >(

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

		this->p_B, // HERE THIS IS GETTING PoP'D SO WHY WE HAVE A PROBLEM COLLECTING AND DISPLAYING?
		this->p_AreaMinor
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelCreate_pressure_gradT_and_gradA_CurlA_minor");


	kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info, // populated position... not neigh_len apparently
		p_Az,
		this->p_izTri_vert,
		this->p_izNeigh_TriMinor,
		this->p_szPBCtri_vert,
		this->p_szPBC_triminor,
		p_LapAz,
		this->p_AreaMinor // OUTPUT
		);
	Call(cudaThreadSynchronize(), "cudaTS GetLapMinor zz");


	kernelNeutral_pressure << <numTriTiles, threadsPerTileMinor >> >(
		this->p_info,

		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_izNeigh_TriMinor,
		this->p_szPBC_triminor,
		this->p_who_am_I_to_corner,
		this->p_tri_corner_index,

		this->p_T_minor,
		p_n_shards_n,
		this->p_n_minor,

		p_MAR_neut
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelNeutral_pressure");

	long i;
	SetConsoleTextAttribute(hConsole, 11);

	// . Create best estimate of n on cc (and avg T to cc:)

	for (long iIterate = 0; iIterate < 1; iIterate++)
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
	
	cudaMemcpy(pX_half->p_T_minor + BEGINNING_OF_CENTRAL,
		this->p_T_minor + BEGINNING_OF_CENTRAL, sizeof(T3)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	
	if (iHistory > 1)
	{
		RegressionSeedTe(0.5*Timestep, p_store_T_move1, p_store_T_move2,
			pX_half->p_T_minor + BEGINNING_OF_CENTRAL, this->p_T_minor + BEGINNING_OF_CENTRAL,
			this);
	}
	
	int iSuccess;
	do {
		iSuccess = RunBackwardForHeat_ConjugateGradient(
			this->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices,
			pX_half->p_T_minor + BEGINNING_OF_CENTRAL, // dest
			0.5*Timestep,
			this);
		if (iSuccess != 0) iSuccess = RunBackwardForHeat_ConjugateGradient(
			this->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices,
			pX_half->p_T_minor + BEGINNING_OF_CENTRAL, // dest
			0.5*Timestep,
			this);
	} while (iSuccess != 0);

	SubtractT3 << <numTilesMajor, threadsPerTileMajor >> >
		(	p_store_T_move2, pX_half->p_T_minor + BEGINNING_OF_CENTRAL,
			this->p_T_minor + BEGINNING_OF_CENTRAL	);
	Call(cudaThreadSynchronize(), "cudaTS subtractT3");

	// This isn't ideal ---- for this one we might like the old half move and the old
	// full move stored
	// but for the full move which is the more expensive one -- do we really want
	// half move and old move?

	// this did have area populated... within ins
	cudaMemcpy(NT_addition_rates_d_temp, NT_addition_rates_d, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	cudaMemset(p_boolarray, 0, sizeof(bool)*NUMVERTICES*2); // initially allow all flows good

	int iPass = 0;
	bool bContinue;
	do {
		printf("iPass %d :\n", iPass);

		// reset NTrates:
		cudaMemcpy(NT_addition_rates_d, NT_addition_rates_d_temp, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToDevice);

		kernelAccumulateDiffusiveHeatRate_new_Full << <numTilesMajorClever, threadsPerTileMajorClever >> > (
			0.5*Timestep,
			this->p_info,
			this->p_izNeigh_vert,
			this->p_szPBCneigh_vert,
			this->p_izTri_vert,
			this->p_szPBCtri_vert,
			this->p_cc,
			this->p_n_major,

			pX_half->p_T_minor + BEGINNING_OF_CENTRAL, // Use T_k+1 just calculated...
			p_boolarray, // array of which ones require longi flows
						 // 2 x NMAJOR
			this->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
			p_kappa_n,
			p_kappa_i,
			p_kappa_e,
			p_nu_i,
			p_nu_e,
			NT_addition_rates_d,
			this->p_AreaMajor,
			(iPass == 0) ? false : true);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate pX_half");
		// To increase the efficiency we want to make a clever 2nd set of major tiles of size 192. Also try 256, 384.

		cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever);
		kernelCreatePutativeT << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			0.5*Timestep,
			this->p_info,
			this->p_T_minor + BEGINNING_OF_CENTRAL,
			//	p_T_upwind_minor_and_putative_T + BEGINNING_OF_CENTRAL, // putative T storage
			this->p_n_major,
			this->p_AreaMajor,
			NT_addition_rates_d,
			p_boolarray, // an array of whether this one requires longi flows --- did it come out T < 0
						 // 2x NMAJOR
			p_bFailed // did we hit any new negative T to add
			);
		Call(cudaThreadSynchronize(), "cudaTS kernelCreatePutativeT");

		bContinue = false;
		cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		int i;
		for (i = 0; ((i < numTilesMajorClever) && (p_boolhost[i] == 0)); i++);
		if (i < numTilesMajorClever) bContinue = true;
		iPass++;
	} while (bContinue);
	
	kernelIonisationRates << <numTilesMajor, threadsPerTileMajor >> >(
		0.5*Timestep,
		this->p_info,
		this->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices
		this->p_n_major,
		this->p_AreaMajor,
		NT_addition_rates_d
		);
	Call(cudaThreadSynchronize(), "cudaTS Ionisation");

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

	kernelCalculate_ita_visc << <numTilesMinor, threadsPerTileMinor >> > (
		this->p_info,
		this->p_n_minor,
		this->p_T_minor,

		p_temp3,
		p_temp4,
		p_temp5, // nu_neut
		p_temp1,
		p_temp2,
		p_temp6); // ita_neut);
	Call(cudaThreadSynchronize(), "cudaTS ita 1");
	
	RunBackwardJLSForViscosity(this->p_vie, pX_half->p_vie, Timestep, this);

	cudaMemset(NT_addition_tri_d, 0, sizeof(NTrates)*NUMVERTICES * 2);
	kernelCreate_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> >(

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
	
	kernelCreate_neutral_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> >
	(	this->p_info,
		this->p_v_n,
		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_izNeigh_TriMinor,
		this->p_szPBC_triminor,
		p_temp6, // ita
		p_temp5, // nu		
		p_MAR_neut, // just accumulates
		NT_addition_rates_d,
		NT_addition_tri_d);
	Call(cudaThreadSynchronize(), "cudaTS visccontrib 1");

	Collect_Nsum_at_tris << <numTriTiles, threadsPerTileMinor >> >(
		this->p_info,
		this->p_n_minor,
		this->p_tri_corner_index,
		this->p_AreaMajor, // populated?
		p_temp4, p_temp5);
	Call(cudaThreadSynchronize(), "cudaTS Nsum");

	kernelTransmitHeatToVerts << <numTilesMajor, threadsPerTileMajor >> > (
		this->p_info,
		this->p_izTri_vert,
		this->p_n_minor,
		this->p_AreaMajor,
		p_temp4, p_temp5,
		NT_addition_rates_d,
		NT_addition_tri_d
		);
	Call(cudaThreadSynchronize(), "cudaTS sum up heat 1");

	SetConsoleTextAttribute(hConsole, 14);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NeTe %1.10E\n\n", VERTCHOSEN, tempf64);
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\nNT_addition_rates_d[%d].NiTi %1.10E\n\n", VERTCHOSEN, tempf64);
	SetConsoleTextAttribute(hConsole, 15);
	
	kernelAdvanceDensityAndTemperature_noadvectioncompression << <numTilesMajor, threadsPerTileMajor >> >(
		0.5*Timestep,
		this->p_info + BEGINNING_OF_CENTRAL,
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL,
		NT_addition_rates_d,
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL,		
		this->p_vie, // for resistive htg
		this->p_v_n,
		this->p_AreaMajor,
		pX_half->p_n_major,
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL
		);
	// Add in a test for T<0 !!!
	Call(cudaThreadSynchronize(), "cudaTS Advance_n_and_T _noadvect"); // vertex
		
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

	kernelCreateShardModelOfDensities_And_SetMajorArea << <numTilesMajor, threadsPerTileMajor >> > (
		pX_half->p_info,
		pX_half->p_n_major,
		pX_half->p_n_minor,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert, // could be more intelligent way than storing and copying 3x
		pX_half->p_cc,
		p_n_shards,
		p_n_shards_n,
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

	kernelPopulateBackwardOhmsLaw_noadvect << <numTilesMinor, threadsPerTileMinor >> > (
		0.5*Timestep,
		this->p_info,
		p_MAR_neut, p_MAR_ion, p_MAR_elec,
		this->p_B, // for target it wasn't populated, right? Only used for thermal & nu_effective ?
		p_LapAz,
		p_GradAz,
		p_GradTe,
		this->p_n_minor,
		this->p_T_minor,
		this->p_vie,
		this->p_v_n,
		this->p_AAdot,
		this->p_AreaMinor,

		p_vn0,
		p_v0,
		p_OhmsCoeffs,
		p_Iz0_summands,
		p_sigma_Izz,
		p_denom_i, p_denom_e, p_coeff_of_vez_upon_viz, p_beta_ie_z,
		true);
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

	f64 neg_Iz_per_triangle = -(Iz_prescribed_endtime) / (f64)numReverseJzTriangles;
	Set_f64_constant(negative_Iz_per_triangle, neg_Iz_per_triangle);

	printf("GPU: Iz_prescribed %1.14E Iz0 %1.14E sigma_Izz %1.14E \n",
		Iz_prescribed_endtime, Iz0, sigma_Izz);
	printf("Ez_strength (GPU) %1.14E \n", Ez_strength_);

	// Update velocities and Azdot:
	kernelCalculateVelocityAndAzdot_noadvect << <numTilesMinor, threadsPerTileMinor >> >(
		0.5*Timestep,
		pX_half->p_info,
		p_vn0,
		p_v0,
		p_OhmsCoeffs,
		pX_target->p_AAdot,  // why target? intermediate value
		pX_half->p_n_minor,
		this->p_AreaMinor,
		p_LapAz,
		pX_half->p_AAdot,
		pX_half->p_vie,
		pX_half->p_v_n
		);
	Call(cudaThreadSynchronize(), "cudaTS AccelerateUpdate 1");
	
	// =====================

	SetConsoleTextAttribute(hConsole, 15);

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

	CallMAC(cudaMemset(p_MAR_neut, 0, sizeof(f64_vec3)*NMINOR));
	CallMAC(cudaMemset(p_MAR_ion, 0, sizeof(f64_vec3)*NMINOR));
	CallMAC(cudaMemset(p_MAR_elec, 0, sizeof(f64_vec3)*NMINOR));

	// Now notice we take a grad Azdot but Azdot has not been defined except from time t_k!!
	kernelCreate_pressure_gradT_and_gradA_CurlA_minor_noadvect << <numTriTiles, threadsPerTileMinor >> >(
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

		pX_half->p_B,
		pX_half->p_AreaMinor
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelCreate_pressure_gradT_and_gradA_LapA_CurlA_minor pX_half");

	kernelNeutral_pressure << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_info,

		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_izNeigh_TriMinor,
		pX_half->p_szPBC_triminor,
		pX_half->p_who_am_I_to_corner,
		pX_half->p_tri_corner_index,

		pX_half->p_T_minor,
		p_n_shards_n,
		pX_half->p_n_minor,

		p_MAR_neut
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelNeutral_pressure_and_momflux pX_half");
	
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

	for (long iIterate = 0; iIterate < 1; iIterate++)
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

	// Why were we missing this?
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

	cudaMemcpy(pX_half->p_T_minor + BEGINNING_OF_CENTRAL, // better seed than T_k?
		pX_target->p_T_minor + BEGINNING_OF_CENTRAL, sizeof(T3)*NUMVERTICES, cudaMemcpyDeviceToDevice);

	if (iHistory > 1)
		RegressionSeedTe(Timestep, p_store_T_move1, p_store_T_move2,
			pX_target->p_T_minor + BEGINNING_OF_CENTRAL, this->p_T_minor + BEGINNING_OF_CENTRAL,
			pX_half);

	do {
		iSuccess = RunBackwardForHeat_ConjugateGradient(
			this->p_T_minor + BEGINNING_OF_CENTRAL,
			pX_target->p_T_minor + BEGINNING_OF_CENTRAL,
			Timestep,
			pX_half);
		if (iSuccess != 0) iSuccess = RunBackwardJLSForHeat(
			this->p_T_minor + BEGINNING_OF_CENTRAL,
			pX_target->p_T_minor + BEGINNING_OF_CENTRAL,
			Timestep,
			pX_half);
	} while (iSuccess != 0);

	SubtractT3 << <numTilesMajor, threadsPerTileMajor >> >
		(p_store_T_move1, pX_target->p_T_minor + BEGINNING_OF_CENTRAL,
			this->p_T_minor + BEGINNING_OF_CENTRAL);
	Call(cudaThreadSynchronize(), "cudaTS subtractT3");
	
	// Something to know : we never zero "NT_addition_rates" in the routine.
	// So we need to do it outside.

	// Something to know : we never zero "NT_addition_rates" in the routine.
	// So we need to do it outside.
	cudaMemcpy(NT_addition_rates_d_temp, NT_addition_rates_d, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	cudaMemset(p_boolarray, 0, sizeof(bool)*NUMVERTICES*2); // initially allow all flows good

	iPass = 0;
	do {
		printf("iPass %d :\n", iPass);

		// reset NTrates:
		cudaMemcpy(NT_addition_rates_d, NT_addition_rates_d_temp, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToDevice);

		kernelAccumulateDiffusiveHeatRate_new_Full << <numTilesMajorClever, threadsPerTileMajorClever >> > (
			Timestep,
			pX_half->p_info,
			pX_half->p_izNeigh_vert,
			pX_half->p_szPBCneigh_vert,
			pX_half->p_izTri_vert,
			pX_half->p_szPBCtri_vert,
			pX_half->p_cc,

			pX_half->p_n_major,
			pX_target->p_T_minor + BEGINNING_OF_CENTRAL, // Use T_k+1 just calculated...
			p_boolarray, // array of which ones require longi flows
						 // 2 x NMAJOR
			pX_half->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
			p_kappa_n,
			p_kappa_i,
			p_kappa_e,
			p_nu_i,
			p_nu_e,
			NT_addition_rates_d,
			pX_half->p_AreaMajor,
			(iPass == 0) ? false : true);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate pX_half");
		// To increase the efficiency we want to make a clever 2nd set of major tiles of size 192. Also try 256, 384.

		cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever);
		kernelCreatePutativeT << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			Timestep,
			pX_half->p_info,
			this->p_T_minor + BEGINNING_OF_CENTRAL,
			//	p_T_upwind_minor_and_putative_T + BEGINNING_OF_CENTRAL, // putative T storage
			pX_half->p_n_major,
			pX_half->p_AreaMajor,
			NT_addition_rates_d,

			p_boolarray, // an array of whether this one requires longi flows --- did it come out T < 0
						 // 2x NMAJOR
			p_bFailed // did we hit any new negative T to add
			);
		Call(cudaThreadSynchronize(), "cudaTS kernelCreatePutativeT");

		bContinue = false;
		cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		int i;
		for (i = 0; ((i < numTilesMajorClever) && (p_boolhost[i] == 0)); i++);
		if (i < numTilesMajorClever) bContinue = true;
		iPass++;
	} while (bContinue);

	
	//kernelIonisationRates << <numTilesMajor, threadsPerTileMajor >> >(
	//	0.5*Timestep,
	//	pX_half->p_info,
	//	pX_half->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices
	//	pX_half->p_n_major,
	//	pX_half->p_AreaMajor,
	//	NT_addition_rates_d
	//	);
	//Call(cudaThreadSynchronize(), "cudaTS Ionisation pXhalf");

	// 28/09/19: This sometimes fails because only T_k + h NeTe / Ne is guaranteed to be positive.


	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("%d NeTe rate %1.10E \n", VERTCHOSEN, tempf64);


	kernelIonisationRates << <numTilesMajor, threadsPerTileMajor >> >(
		Timestep,
		pX_half->p_info,
		this->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices
		this->p_n_major,
		pX_half->p_AreaMajor,
		NT_addition_rates_d
		);
	Call(cudaThreadSynchronize(), "cudaTS Ionisation pXhalf");
	
	
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("%d NeTe rate %1.10E \n", VERTCHOSEN, tempf64);


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

	::RunBackwardJLSForViscosity(this->p_vie, pX_target->p_vie, Timestep, pX_half);

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
	kernelCreate_neutral_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_info,
		pX_half->p_v_n,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_izNeigh_TriMinor,
		pX_half->p_szPBC_triminor,
		p_temp6, // ita
		p_temp5, // nu
		p_MAR_neut,
		NT_addition_rates_d,
		NT_addition_tri_d);
	Call(cudaThreadSynchronize(), "cudaTS visccontribneut 2");

	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("%d NeTe rate %1.10E \n", VERTCHOSEN, tempf64);

	Collect_Nsum_at_tris << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_info,
		pX_half->p_n_minor,
		pX_half->p_tri_corner_index,
		pX_half->p_AreaMajor, // populated?
		p_temp4, p_temp5);
	Call(cudaThreadSynchronize(), "cudaTS Nsum 2");

	kernelTransmitHeatToVerts << <numTilesMajor, threadsPerTileMajor >> > (
		pX_half->p_info,
		pX_half->p_izTri_vert,
		pX_half->p_n_minor,
		pX_half->p_AreaMajor,
		p_temp4, p_temp5,
		NT_addition_rates_d,
		NT_addition_tri_d
		);
	Call(cudaThreadSynchronize(), "cudaTS sum up heat 2");
	
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("%d NeTe rate %1.10E \n", VERTCHOSEN, tempf64);


	//
	//	fprintf(fpdat, "wViscous MARion ");
	//	for (i = 0; i < 3; i++) {
	//		cudaMemcpy(&tempf64, &(p_MAR_ion[VERTS[i] + BEGINNING_OF_CENTRAL].y), sizeof(f64), cudaMemcpyDeviceToHost);
	//		fprintf(fpdat, " %1.12E ", tempf64);
	//	};
	//	fprintf(fpdat, "wViscous MARelec ");
	//	for (i = 0; i < 3; i++) {
	//		cudaMemcpy(&tempf64, &(p_MAR_elec[VERTS[i] + BEGINNING_OF_CENTRAL].y), sizeof(f64), cudaMemcpyDeviceToHost);
	//		fprintf(fpdat, " %1.12E ", tempf64);
	//	};
	//	fprintf(fpdat, "wViscous_NiTi ");
	//	for (i = 0; i < 3; i++) {
	//		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTS[i]].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
	//		fprintf(fpdat, " %1.12E ", tempf64);
	//	};
	//
	//
	//	fprintf(fpdat, "Bx ");
	//	for (i = 0; i < 3; i++) {
	//		cudaMemcpy(&tempf64, &(pX_half->p_B[VERTS[i] + BEGINNING_OF_CENTRAL].x), sizeof(f64), cudaMemcpyDeviceToHost);
	//		fprintf(fpdat, " %1.12E ", tempf64);
	//	};
	//	fprintf(fpdat, "By ");
	//	for (i = 0; i < 3; i++) {
	//		cudaMemcpy(&tempf64, &(pX_half->p_B[VERTS[i] + BEGINNING_OF_CENTRAL].y), sizeof(f64), cudaMemcpyDeviceToHost);
	//		fprintf(fpdat, " %1.12E ", tempf64);
	//	};
	//	fprintf(fpdat, "viz ");
	//	for (i = 0; i < 3; i++) {
	//		cudaMemcpy(&tempf64, &(pX_half->p_vie[VERTS[i] + BEGINNING_OF_CENTRAL].viz), sizeof(f64), cudaMemcpyDeviceToHost);
	//		fprintf(fpdat, " %1.12E ", tempf64);
	//	};
	//	fprintf(fpdat, "vez ");
	//	for (i = 0; i < 3; i++) {
	//		cudaMemcpy(&tempf64, &(pX_half->p_vie[VERTS[i] + BEGINNING_OF_CENTRAL].vez), sizeof(f64), cudaMemcpyDeviceToHost);
	//		fprintf(fpdat, " %1.12E ", tempf64);
	//	};
	//	fprintf(fpdat, "\n");
	//	fclose(fpdat);

	 
	kernelAdvanceDensityAndTemperature_noadvectioncompression << <numTilesMajor, threadsPerTileMajor >> >(
		Timestep,
		this->p_info + BEGINNING_OF_CENTRAL,
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL,
		NT_addition_rates_d,
		pX_half->p_n_major,  // ?
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL,  // ?

		pX_half->p_vie + BEGINNING_OF_CENTRAL,
		pX_half->p_v_n + BEGINNING_OF_CENTRAL,

		this->p_AreaMajor,

		pX_target->p_n_major,
		pX_target->p_T_minor + BEGINNING_OF_CENTRAL
		); // do check for T<0
	Call(cudaThreadSynchronize(), "cudaTS Advance_n_and_T 2330");

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

	//////////////////////////////////////////////////////////////////////////
	// Even more shards!:
	// Really needed though? :

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
	cudaEventRecord(middle, 0);
	cudaEventSynchronize(middle);

	// BETTER:
	// Just make this the determinant of how long the overall timestep is;
	// Make supercycle: advection is not usually applied.

	kernelGetLapCoeffs_and_min << <numTriTiles, threadsPerTileMinor >> >(
		pX_target->p_info,
		pX_target->p_izTri_vert,
		pX_target->p_izNeigh_TriMinor,
		pX_target->p_szPBCtri_vert,
		pX_target->p_szPBC_triminor,
		p_LapCoeffself,
		p_temp1, // collect min
		p_longtemp
		);
	Call(cudaThreadSynchronize(), "cudaTS GetLapCoeffs");

	//cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTriTiles, cudaMemcpyDeviceToHost);
	//cudaMemcpy(p_longtemphost, p_longtemp, sizeof(long)*numTriTiles, cudaMemcpyDeviceToHost);

	//// It should be universally true that coeffself is negative. Higher self = more negative Lap.
	//f64 mincoeffself = 0.0;
	//long iMin = -1;
	//for (iTile = 0; iTile < numTriTiles; iTile++)
	//{
	//	if (p_temphost1[iTile] < mincoeffself) {
	//		mincoeffself = p_temphost1[iTile];
	//		iMin = p_longtemphost[iTile];
	//	}
	//	//	printf("iTile %d iMin %d cs %1.12E \n", iTile, p_longtemphost[iTile], p_temphost1[iTile]);
	//}

	//f64 h_sub_max = 1.0 / (c_*sqrt(fabs(mincoeffself))); // not strictly correct - 
	//													 // e.g. with this at 7.3e-14, using 1e-13 as substep works (10% backward) ;
	//													 // with it at 6.4e-14 it does not work. 
	//													 // So the inflation factor that you can get away with, isn't huge.
	//printf("\nMin coeffself %1.12E iMin %d 1.0/(c sqrt(-mcs)) %1.12E\n", mincoeffself, iMin,
	//	h_sub_max);
	//// Comes out with sensible values for max abs coeff ~~ delta squared?
	//// Add in factor to inflate Timestep when we want to play around.

	//// iSubcycles = (long)(Timestep / h_sub_max)+1;
	//if (Timestep > h_sub_max*2.0) // YES IT IS LESS THAN 1x h_sub_max . Now that seems bad. But we are doing bwd so .. ???
	//{
	//	printf("\nAlert! Timestep > 2.0 h_sub_max %1.11E %1.11E \a\n", Timestep, h_sub_max);
	//}
	//else {
	//	printf("Timestep %1.11E h_sub_max %1.11E \n", Timestep, h_sub_max);
	//}
	
	
	//if (runs % BWD_SUBCYCLE_FREQ == 0) {
	//	printf("backward!\n");
	//	iSubcycles /= BWD_STEP_RATIO; // some speedup this way
	//} else {
	//	iSubcycles *= FWD_STEP_FACTOR;
	//}

	// Don't do this stuff --- just make whole step shorter.

	iSubcycles = 1;
	hsub = Timestep; //  / (real)iSubcycles;

	kernelPullAzFromSyst << <numTilesMinor, threadsPerTileMinor >> >(
		this->p_AAdot, // A_k
		p_Az
		);
	Call(cudaThreadSynchronize(), "cudaTS PullAz");

	evaltime += hsub; // t_k+1

					  // BACKWARD EULER:

	kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
		pX_target->p_info, // populated position... not neigh_len apparently
		p_Az,
		pX_target->p_izTri_vert,
		pX_target->p_izNeigh_TriMinor,
		pX_target->p_szPBCtri_vert,
		pX_target->p_szPBC_triminor,
		p_LapAz,
		pX_target->p_AreaMinor // OUTPUT
		);
	Call(cudaThreadSynchronize(), "cudaTS GetLapMinor aaa2");

	Iz_prescribed_endtime = GetIzPrescribed(evaltime); // APPLIED AT END TIME: we are determining

	kernelPopulateBackwardOhmsLaw_noadvect << <numTilesMinor, threadsPerTileMinor >> > (
		hsub,
		pX_target->p_info,
		p_MAR_neut, p_MAR_ion, p_MAR_elec,
		pX_half->p_B, // for target it wasn't populated, right? Only used for thermal & nu_effective ?
		p_LapAz,
		p_GradAz,
		p_GradTe,
		pX_target->p_n_minor,
		pX_target->p_T_minor,
		this->p_vie,
		this->p_v_n,
		this->p_AAdot,
		pX_half->p_AreaMinor, // NOT POPULATED FOR PXTARGET

		p_vn0,
		p_v0,
		p_OhmsCoeffs,
		p_Iz0_summands,
		p_sigma_Izz,
		p_denom_i, p_denom_e, p_coeff_of_vez_upon_viz, p_beta_ie_z,
		true);
	Call(cudaThreadSynchronize(), "cudaTS kernelPopulateBackwardOhmsLaw ");
	
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
	if (EzStrength_ != EzStrength_) { printf("end\n"); while (1) getch(); }
	Set_f64_constant(Ez_strength, EzStrength_);

	neg_Iz_per_triangle = -(Iz_prescribed_endtime) / (f64)numReverseJzTriangles;
	Set_f64_constant(negative_Iz_per_triangle, neg_Iz_per_triangle);
	// Electrons travel from cathode to anode so Jz is down in filament,
	// up around anode.
	printf("\nGPU: Iz0 = %1.14E SigmaIzz %1.14E EzStrength = %1.14E \n\n", Iz0, Sigma_Izz, EzStrength_);
	 
	FILE * jillium = fopen("Ez1a.txt", "a");
	fprintf(jillium, "Runs %d GPU: Izpresc %1.14E Iz0 = %1.14E SigmaIzz %1.14E EzStrength = %1.14E \n",
		runs, Iz_prescribed_endtime ,Iz0, Sigma_Izz, EzStrength_);
	fclose(jillium);
	if (EzStrength_ > 1.0e5) {
		printf("time to stop, press p");
		while (getch() != 'p');
		PerformCUDA_Revoke();
		exit(2323);
	} 

	kernelCreateLinearRelationshipBwd_noadvect << <numTilesMinor, threadsPerTileMinor >> > (
		hsub,
		pX_target->p_info,
		p_OhmsCoeffs,
		p_v0,
		p_LapAz,  // used for cancelling .. 
		pX_target->p_n_minor,
		p_denom_e,
		p_denom_i, p_coeff_of_vez_upon_viz, p_beta_ie_z,

		this->p_AAdot,

		pX_half->p_AreaMinor, // because not populated in PXTARGET
		p_Azdot0,
		p_gamma
		); // MODIFY vez0, viz0 IF THEY WILL EVER BE USED.
	Call(cudaThreadSynchronize(), "cudaTS kernelCreateLinearRelationshipBwd ");

	kernelCreateExplicitStepAz << <numTilesMinor, threadsPerTileMinor >> > (
		hsub, 
		p_Azdot0,
		p_gamma,
		p_LapAz, // we based this off of half-time Az. --not any more, time t_k
		p_temp6); // = h (Azdot0 + gamma*LapAz)
	Call(cudaThreadSynchronize(), "cudaTS Create Seed Az");

	// set p_storeAz to some useful value on the very first step.

	if (iHistory > 0)
	{
		RegressionSeedAz(hsub, p_Az, p_AzNext, p_temp6, stored_Az_move, p_Azdot0, p_gamma, p_LapCoeffself, pX_target);
		// Idea: regress epsilon(Az) on p_temp6, stored_Az_move, Jacobi(stored_Az_move);
		// Update p_AzNext as the result.

		// .
		// Do moves really have a low correlation with each other?
		// Save & analyse correls.
		
		// .
		// Alternative way: regress on states t_k and t_k-1 rather than difference.
		// Result there?
		// Or do 2 historic states, then Richardson+JR+JJR, then etc.

	} else {
		kernelCreateSeedAz << <numTilesMinor, threadsPerTileMinor >> >
			(hsub, p_Az, p_Azdot0, p_gamma, p_LapAz, p_AzNext);
		Call(cudaThreadSynchronize(), "cudaTS createSeed");
	}

	//	SolveBackwardAzAdvanceCG(hsub, p_Az, p_Azdot0, p_gamma, 
	//		p_AzNext, p_LapCoeffself, pX_target);

	SolveBackwardAzAdvanceJ3LS(hsub, p_Az, p_Azdot0, p_gamma,
		p_AzNext, p_LapCoeffself, pX_target);
	
	SubtractVector<<<numTilesMinor, threadsPerTileMinor>>>
		(stored_Az_move, p_Az, p_AzNext);

	Call(cudaThreadSynchronize(), "cudaTS subtract");
	iHistory++; // we have now been through this point.

	cudaMemcpy(p_Az, p_AzNext, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);

	kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
		pX_target->p_info,
		p_Az,
		pX_target->p_izTri_vert,
		pX_target->p_izNeigh_TriMinor,
		pX_target->p_szPBCtri_vert,
		pX_target->p_szPBC_triminor,
		p_LapAz,
		pX_half->p_AreaMinor
		);
	Call(cudaThreadSynchronize(), "cudaTS GetLap Az JLS 2");

	// Lap Az is now known, let's say.
	// So we are again going to call PopOhms Backward -- but this time we do not wish to save off stuff
	// except for the v(Ez) relationship.

	kernelPopulateBackwardOhmsLaw_noadvect << <numTilesMinor, threadsPerTileMinor >> > (
		hsub,
		pX_target->p_info,
		p_MAR_neut, p_MAR_ion, p_MAR_elec,
		pX_half->p_B,
		p_LapAz,
		p_GradAz, // THIS WE OUGHT TO TWEEN AT LEAST
		p_GradTe,
		pX_target->p_n_minor,
		pX_target->p_T_minor,

		this->p_vie,
		this->p_v_n,
		this->p_AAdot,
		pX_half->p_AreaMinor, // pop'd? interp?
	
		p_vn0,
		p_v0,
		p_OhmsCoeffs,

		p_Iz0_summands,
		p_sigma_Izz,
		p_denom_i,
		p_denom_e, p_coeff_of_vez_upon_viz, p_beta_ie_z,
		false);
	Call(cudaThreadSynchronize(), "cudaTS PopBwdOhms II ");

	// Might as well recalculate Ez_strength again :
	// Iz already set for t+hsub.
	cudaMemcpy(p_Iz0_summands_host, p_Iz0_summands, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_summands_host, p_sigma_Izz, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	Iz0 = 0.0;
	Sigma_Izz = 0.0;
	Iz_k = 0.0;
	for (iBlock = 0; iBlock < numTilesMinor; iBlock++)
	{
		Iz0 += p_Iz0_summands_host[iBlock];
		Sigma_Izz += p_summands_host[iBlock];
	}
	EzStrength_ = (Iz_prescribed_endtime - Iz0) / Sigma_Izz;
	Set_f64_constant(Ez_strength, EzStrength_);

	neg_Iz_per_triangle = -(Iz_prescribed_endtime) / (f64)numReverseJzTriangles;
	Set_f64_constant(negative_Iz_per_triangle, neg_Iz_per_triangle);
	// Electrons travel from cathode to anode so Jz is down in filament,
	// up around anode.

	if (EzStrength_ != EzStrength_) {
		printf("EzStrength_ %1.10E Iz_prescribed %1.10E Iz0 %1.10E sigma_Izz %1.10E \n",
			EzStrength_, Iz_prescribed_endtime, Iz0, sigma_Izz);
		while (1) getch();
	}

	kernelCalculateVelocityAndAzdot_noadvect << <numTilesMinor, threadsPerTileMinor >> > (
		hsub,
		pX_target->p_info,
		p_vn0,
		p_v0,
		p_OhmsCoeffs,
		this->p_AAdot,
		//(iSubstep == iSubcycles - 1) ? pX_target->p_n_minor:pX_half->p_n_minor,
		pX_target->p_n_minor, // NOT OKAY FOR IT TO NOT BE SAME n AS USED THROUGHOUT BY OHMS LAW
		pX_half->p_AreaMinor,  // Still because pXTARGET Area still not populated

							   // We need to go back through, populate AreaMinor before we do all these things.
							   // Are we even going to be advecting points every step?
							   // Maybe make advection its own thing.
		p_LapAz,

		pX_target->p_AAdot,
		pX_target->p_vie,
		pX_target->p_v_n
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelUpdate_v ");

	kernelAdvanceAzBwdEuler << <numTilesMinor, threadsPerTileMinor >> > (
		hsub,
		this->p_AAdot,
		pX_target->p_AAdot,
		p_ROCAzduetoAdvection);
	Call(cudaThreadSynchronize(), "cudaTS kernelAdvanceAzBwdEuler ");

	SetConsoleTextAttribute(hConsole, 15);
	printf("evaltime %1.5E \n", evaltime);
	
	printf("Done step %d from time %1.8E length %1.2E\n\n", runs, evaltime, Timestep);

	fp = fopen("elapsed_ii.txt", "a");
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

	GlobalSuppressSuccessVerbosity = true;
	CallMAC(cudaFree(d_eps_by_dx_neigh_n));
	CallMAC(cudaFree(d_eps_by_dx_neigh_i));
	CallMAC(cudaFree(d_eps_by_dx_neigh_e));
	CallMAC(cudaFree(p_regressor_n));
	CallMAC(cudaFree(p_regressor_i));
	CallMAC(cudaFree(p_regressor_e));
	CallMAC(cudaFree(p_Effect_self_n));
	CallMAC(cudaFree(p_Effect_self_i));
	CallMAC(cudaFree(p_Effect_self_e));
	CallMAC(cudaFree(p_boolarray));
	CallMAC(cudaFree(p_store_T_move1));
	CallMAC(cudaFree(p_store_T_move2));

	CallMAC(cudaFree(p_T_upwind_minor_and_putative_T));
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
	CallMAC(cudaFree(p_bFailed));
	CallMAC(cudaFree(p_Ax));

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
	CallMAC(cudaFree(p_d_eps_by_dbetaR_n));
	CallMAC(cudaFree(p_d_eps_by_dbetaR_i));
	CallMAC(cudaFree(p_d_eps_by_dbetaR_e));

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

	CallMAC(cudaFree(p_sum_eps_deps_by_dbeta_J));
	CallMAC(cudaFree(p_sum_eps_deps_by_dbeta_R));
	CallMAC(cudaFree(p_sum_depsbydbeta_J_times_J));
	CallMAC(cudaFree(p_sum_depsbydbeta_R_times_R));
	CallMAC(cudaFree(p_sum_depsbydbeta_J_times_R));

	CallMAC(cudaFree(p_d_eps_by_dbetaJ_n_x4));
	CallMAC(cudaFree(p_d_eps_by_dbetaJ_i_x4));
	CallMAC(cudaFree(p_d_eps_by_dbetaJ_e_x4));
	CallMAC(cudaFree(p_d_eps_by_dbetaR_n_x4));
	CallMAC(cudaFree(p_d_eps_by_dbetaR_i_x4));
	CallMAC(cudaFree(p_d_eps_by_dbetaR_e_x4));

	CallMAC(cudaFree(p_sum_eps_deps_by_dbeta_J_x4));
	CallMAC(cudaFree(p_sum_eps_deps_by_dbeta_R_x4));
	CallMAC(cudaFree(p_sum_depsbydbeta_8x8));
	
	CallMAC(cudaFree(stored_Az_move));
	
	CallMAC(cudaFree(p_Tn));
	CallMAC(cudaFree(p_Ti));
	CallMAC(cudaFree(p_Te));
	CallMAC(cudaFree(p_Ap_n));
	CallMAC(cudaFree(p_Ap_i));
	CallMAC(cudaFree(p_Ap_e));
	
	free(p_sum_eps_deps_by_dbeta_J_x4_host);
	free(p_sum_eps_deps_by_dbeta_R_x4_host);
	free(p_sum_depsbydbeta_8x8_host);

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
	GlobalSuppressSuccessVerbosity = false;
	printf("revoke done\n");

}

// Problem: by default, __constant__ variables have file scope. Need special
// compiler settings to do relocatable device code.

#include "kernel.cu"
#include "little_kernels.cu"

// There must be a better way.
