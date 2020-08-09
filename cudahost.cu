
// Version 1.0 23/04/19:
// Changing to use upwind T for advection. We could do better in future. Interp gives negative T sometimes.
// Corrected ionisation rate.
        
#pragma once    

#define PRECISE_VISCOSITY 
#define DEBUGTE               0
      
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <windows.h>
  
#include "mesh.h"
#include "lapacke.h"
#include "FFxtubes.h"
#include "cuda_struct.h"
#include "flags.h"
#include "kernel.h"
#include "matrix_real.h"
          
/* Auxiliary routines prototypes */
extern void print_matrix(char* desc, lapack_int m, lapack_int n, double* a, lapack_int lda);
extern void print_int_vector(char* desc, lapack_int n, lapack_int* a);
 
extern HWND hwndGraphics;
extern TriMesh X4; 
               
#define BWD_SUBCYCLE_FREQ  1
#define BWD_STEP_RATIO     1    // divide substeps by this for bwd
#define NUM_BWD_ITERATIONS 4
#define FWD_STEP_FACTOR    2    // multiply substeps by this for fwd
             
// This will be slow but see if it solves it.
                       
#define CHOSEN  59805
#define CHOSEN1 59805
#define CHOSEN2 59806
#define VERTCHOSEN 22735
#define VERTCHOSEN2 19180
   
#define ITERATIONS_BEFORE_SWITCH  18
#define REQUIRED_IMPROVEMENT_RATE  0.98
#define REQUIRED_IMPROVEMENT_RATE_J  0.985
 
// This is the file for CUDA host code.
#include "simulation.cu"
  
#define p_sqrtDN_Tn p_NnTn
#define p_sqrtDN_Ti p_NTi
#define p_sqrtDN_Te p_NTe
 
#define DEFAULTSUPPRESSVERBOSITY false
  
extern surfacegraph Graph[8];
extern D3D Direct3D;
extern HWND hWnd;
  
FILE * fp_trajectory;
FILE * fp_dbg;
bool GlobalSuppressSuccessVerbosity = DEFAULTSUPPRESSVERBOSITY;
bool bGlobalSaveTGraphs;
bool bViscousHistory;

long VERTS[3] = {15559, 15405, 15251};
long iEquations[3];
  
extern long NumInnerFrills, FirstOuterFrill;
__constant__ long NumInnerFrills_d, FirstOuterFrill_d;
__constant__ long nBlocks, Nverts, uDataLen_d; // Nverts == numVertices 
f64 over_iEquations_n, over_iEquations_i, over_iEquations_e;
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

__constant__ f64 UNIFORM_n_d;

__constant__ f64 cross_s_vals_viscosity_ni_d[10], cross_s_vals_viscosity_nn_d[10],
                 cross_T_vals_d[10], cross_s_vals_MT_ni_d[10];
__constant__ f64 beta_n_c[32], beta_i_c[8], beta_e_c[8];


__constant__ f64 recomb_coeffs[32][3][5];
f64 recomb_coeffs_host[32][3][5];
__constant__ f64 ionize_coeffs[32][5][5];
f64 ionize_coeffs_host[32][5][5];  
__constant__ f64 ionize_temps[32][10];
f64 ionize_temps_host[32][10];
__constant__ long MyMaxIndex;
__device__ __constant__ f64 billericay;
__constant__ f64 Ez_strength;
f64 EzStrength_;
__constant__ f64 negative_Iz_per_triangle; // -Iz_prescribed / (real)(numEndZCurrentTriangles - numStartZCurrentTriangles)
__constant__ long numStartZCurrentTriangles, numEndZCurrentTriangles;

#define CallMAC(cudaStatus) Call(cudaStatus, #cudaStatus )   
// { Call(cudaStatus, "cudaStatus") } ?
extern real FRILL_CENTROID_OUTER_RADIUS, FRILL_CENTROID_INNER_RADIUS;

extern bool flaglist[NMINOR];

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
__device__ f64_vec3 * p_temp3_1, *p_temp3_2, *p_temp3_3;
__device__ f64 * p_graphdata1, *p_graphdata2, *p_graphdata3, *p_graphdata4, *p_graphdata5, *p_graphdata6;
f64 * p_graphdata1_host, *p_graphdata2_host, *p_graphdata3_host, *p_graphdata4_host, *p_graphdata5_host, *p_graphdata6_host;
__device__ f64_vec3* p_MAR_ion_temp_central, *p_MAR_elec_temp_central;
__device__ f64 * p_Tgraph[9];
f64 * p_Tgraph_host[9];
__device__ f64 * p_accelgraph[12];
f64 * p_accelgraph_host[12];
f64 * p_Ohmsgraph_host[20];
__device__ f64 * p_Ohmsgraph[20];
__device__ f64 * p_eps_against_d_eps;

__device__ v4 * v4temparray, *zero_vec4;
__device__ f64_vec3 * p_MAR_ion3, *p_MAR_elec3, *p_MAR_neut3 , *p_MAR_neut2;
__device__ f64_vec3 * p_MAR_ion_pressure_major_stored, *p_MAR_ion_visc_major_stored, *p_MAR_elec_pressure_major_stored, *p_MAR_elec_visc_major_stored, *p_MAR_elec_ionization_major_stored;
__device__ v4 * p_vie_k_stored;
	
__device__ bool * p_pressureflag;
__device__ f64_vec3 * p_d_epsilon_by_d_beta_x, *p_d_epsilon_by_d_beta_y,
*p_d_epsilon_by_d_beta_z, *p_epsilon3, *v3temp, *zero_vec3;
__device__ f64 * p_place_contribs;
__device__ v4 * p_storeviscmove;
__device__ f64 * p_Residuals;
__device__ long * p_longtemp;
__device__ bool * p_bool;
__device__ f64 * p_regressor_n, *p_regressor_i, *p_regressor_e, *p_Effect_self_n, *p_Effect_self_i, *p_Effect_self_e,
*d_eps_by_dx_neigh_n, *d_eps_by_dx_neigh_i, *d_eps_by_dx_neigh_e;
__device__ T3 * p_store_T_move1, *p_store_T_move2;
NTrates * p_NTrates_host;

__device__ f64 * p_sqrtD_inv_n, *p_sqrtD_inv_i, *p_sqrtD_inv_e;
__device__ f64 * p_regressors;
f64 * p_sum_eps_deps_by_dbeta_x8_host;

__device__ f64_vec3 * p_tempvec3, *p_regressors3, *p_stored_move3;
__device__ f64 * p_SS, * p_epsilon_x, * p_epsilon_y, * p_epsilon_z, *p_d_eps_by_d_beta_x_, *p_d_eps_by_d_beta_y_,
			*p_d_eps_by_d_beta_z_;
f64_vec3 * p_tempvec3host;
f64 * p_SS_host;

__device__ short * sz_who_vert_vert;

__device__ long * p_indicator;
__device__ f64 * p_Jacobian_list;

__device__ NTrates * p_store_NTFlux;

#define SQUASH_POINTS  24
__device__ f64 * p_matrix_blocks;
__device__ f64 * p_vector_blocks;

f64 * p_matrix_blocks_host, *p_vector_blocks_host;
f64 * p_sum_product_matrix_host;
f64_vec3 * p_eps_against_deps_host;
f64 * p_eps_against_d_eps_host;

__device__ f64_vec3 * p_eps_against_deps;
__device__ f64 * p_sum_product_matrix;

#define p_slot1n p_Ap_n
#define p_slot1i p_Ap_i
#define p_slot1e p_Ap_e
#define p_slot2n p_NnTn
#define p_slot2i p_NTi
#define p_slot2e p_NTe

__device__ f64 * p_Tn, *p_Ti, *p_Te, *p_Ap_n, *p_Ap_i, *p_Ap_e, * p_NnTn, * p_NTi, * p_NTe,
			* stored_Az_move;

#define p_Tik p_NTi
#define p_Tek p_NTe
#define p_Tnk p_NnTn

// Don't forget made this union.


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
__device__ bool * p_bFailed, *p_boolarray, * p_boolarray2, *p_boolarray_block;

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
__device__ NTrates *NT_addition_rates_d, *NT_addition_tri_d, *NT_addition_rates_d_temp2;
long numReverseJzTriangles;
__device__ f64 *p_sum_eps_deps_by_dbeta, *p_sum_depsbydbeta_sq, *p_sum_eps_eps;
f64  *p_sum_eps_deps_by_dbeta_host, *p_sum_depsbydbeta_sq_host, *p_sum_eps_eps_host;
__device__ char * p_was_vertex_rotated, *p_triPBClistaffected;
__device__ T3 * p_T_upwind_minor_and_putative_T;

__device__ f64 * p_d_eps_by_dbeta_n, *p_d_eps_by_dbeta_i, *p_d_eps_by_dbeta_e,
*p_d_eps_by_dbetaR_n, *p_d_eps_by_dbetaR_i, *p_d_eps_by_dbetaR_e,
*p_Jacobi_n, *p_Jacobi_i, *p_Jacobi_e, *p_epsilon_n, *p_epsilon_i, *p_epsilon_e,
*p_coeffself_n, *p_coeffself_i, *p_coeffself_e; // Are these fixed or changing with each iteration?

__device__ f64_tens3 * p_InvertedMatrix_i, *p_InvertedMatrix_e, *p_InvertedMatrix_n;
__device__ f64_vec3 * p_MAR_ion2, *p_MAR_elec2, * p_vJacobi_i, * p_vJacobi_e, *p_vJacobi_n,
	* p_d_eps_by_d_beta_i, *p_d_eps_by_d_beta_e;

__device__ f64_vec2 *p_d_epsxy_by_d_beta_i, *p_d_epsxy_by_d_beta_e;
__device__ f64 *p_d_eps_iz_by_d_beta_i, *p_d_eps_ez_by_d_beta_i,
			   *p_d_eps_iz_by_d_beta_e, *p_d_eps_ez_by_d_beta_e;

f64_vec3 * p_sum_vec_host;
__device__ NTrates * NT_addition_rates_d_temp, * store_heatcond_NTrates;
__device__ f64_vec2 * p_epsilon_xy;
__device__ f64 * p_epsilon_iz, *p_epsilon_ez,
*p_sum_eps_deps_by_dbeta_i, *p_sum_eps_deps_by_dbeta_e, *p_sum_depsbydbeta_i_times_i,
*p_sum_depsbydbeta_e_times_e, *p_sum_depsbydbeta_e_times_i;
f64 * p_sum_eps_deps_by_dbeta_i_host, * p_sum_eps_deps_by_dbeta_e_host, * p_sum_depsbydbeta_i_times_i_host,
*p_sum_depsbydbeta_e_times_e_host, *p_sum_depsbydbeta_e_times_i_host;

__device__ f64 *p_sum_eps_deps_by_dbeta_J, *p_sum_eps_deps_by_dbeta_R, *p_sum_depsbydbeta_J_times_J,
	*p_sum_depsbydbeta_R_times_R, *p_sum_depsbydbeta_J_times_R,
	*p_sum_eps_deps_by_dbeta_x8;
f64 * p_sum_eps_deps_by_dbeta_J_host, *p_sum_eps_deps_by_dbeta_R_host, *p_sum_depsbydbeta_J_times_J_host,
	*p_sum_depsbydbeta_R_times_R_host, *p_sum_depsbydbeta_J_times_R_host;

__device__ AAdot *p_AAdot_target, *p_AAdot_start;
__device__ f64_vec3 * p_v_n_target, *p_v_n_start;
__device__ v4 * p_vie_target, *p_vie_start;

TriMesh * pTriMesh;

f64 * temp_array_host;
f64 tempf64;
FILE * fp_traj;

void GosubAccelerate(long iSubcycles, f64 hsub, cuSyst * pX_use, cuSyst * pX_intermediate);

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


	
	// The principal reason this routine stutters and doesn't work is that we would have to multiply by AreaMinor
	// to get equation symmetry.




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
	Call(cudaThreadSynchronize(), "cudaTS kernelResetFrillsAz"); // can leave this here but
	// NEVER LOOK INTO FRILL -- can it be done?
 

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
			p_LapAz);
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

	// No. We need to stop looking into frill from ANY DIRECTION. That is the quick and dirty way.
	// The clean proper way would be to find the lc of neighs for A_frill that makes symmetry work.
	// However, we want to do Dirichlet anyway.
	
	// Real problem: AreaMinor not all the same. Not symmetric Lap.

	////  ... Directions now.  ..  Do CG .. get done with programming, on to just writing.
	// CG for visc may be a lot more powerful than JLS since it may not be diagonally dominant.
	//        i. Put on Github before making changes.




	

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
				p_LapAz);
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
					p_LapAz);
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
	f64 beta[SQUASH_POINTS];
	Tensor3 mat;
	f64 RSS;
	int iTile;
	char buffer[256];
	int iIteration = 0;

	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	f64 matrix[SQUASH_POINTS*SQUASH_POINTS];
	f64 vector[SQUASH_POINTS];

	long iMax = 0;
	bool bContinue = true;
	 
//	printf("iIteration = %d\n", iIteration);
	// 1. Create regressor:
	// Careful with major vs minor + BEGINNING_OF_CENTRAL:

	GlobalSuppressSuccessVerbosity = true;

	kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
			pX_use->p_info, pX_use->p_tri_neigh_index,
			p_AzNext);
	Call(cudaThreadSynchronize(), "cudaTS ResetFrills Az");
	// if DIRICHLET, set outer ones to 0 just for sake of it.
	 
	kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
		pX_use->p_info,
		p_AzNext,
		pX_use->p_izTri_vert,
		pX_use->p_izNeigh_TriMinor,
		pX_use->p_szPBCtri_vert,
		pX_use->p_szPBC_triminor,
		p_LapAzNext
		);
	Call(cudaThreadSynchronize(), "cudaTS GetLap Az JLS 1");
	// Put 0 line through outer frill centroid radius.
	// Squash the outermost vertex cell.
	
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
	// Do not recognize Jz outside 5cm
	// Do include self-effect looking at inner frill which is mirror
	// Do include self-effect looking at outer frill which is zero
	
	kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
		pX_use->p_info, pX_use->p_tri_neigh_index, p_Jacobi_x);
	Call(cudaThreadSynchronize(), "cudaTS ResetFrills Jacobi");
	 
	f64 L4L2ratio = 0.0;

//	FILE * fpdbg = fopen("J3LS_2.txt", "w");
	L2eps = -1.0;
	bool bSpitOutErrorAfter = false;
	long iMax0;
	do
	{
		// Now we want to create another regressor, let it be called p_regressor_n
		// Let p_Jacobi_x act as AzNext

		if (((iIteration > 4) && (L4L2ratio > 10.0) && (iIteration % 2 == 0))
			|| ((iIteration > 1600) && (iIteration % 4 == 0))) // if things are messed up, try it anyway
		{
			
			printf("\nDoing the smash! iteration %d\n", iIteration);

			// Alternative: smoosh 24 points

			// find maximum
			cudaMemset(p_indicator, 0, sizeof(long)*NMINOR);
			int number_set = 0; // now we're going to have to number them as 1 through 24 .. be careful.
			do {
				kernelReturnMaximumInBlock << <numTilesMinor, threadsPerTileMinor >> > (
					p_epsilon,
					p_temp1, // max in block
					p_longtemp,
					p_indicator // long*NMINOR : if this point is already used, do not pick it up.
					);
				cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
				cudaMemcpy(p_longtemphost, p_longtemp, sizeof(long)*numTilesMinor, cudaMemcpyDeviceToHost);
				f64 maximum = 0.0;
				long iMax = 0;
				for (iTile = 0; iTile < numTilesMinor; iTile++)
				{
					if (p_temphost1[iTile] > maximum) {
						maximum = p_temphost1[iTile];
						iMax = p_longtemphost[iTile];
					};
				};
				long ii = number_set + 1;
				cudaMemcpy(&(p_indicator[iMax]), &ii, sizeof(long), cudaMemcpyHostToDevice);

				//if (number_set == 0)
				//{
				//	long * longaddress;
				//	Call(cudaGetSymbolAddress((void **)(&longaddress), MyMaxIndex),
				//		"cudaGetSymbolAddress((void **)(&longaddress), MyMaxIndex)");
				//	Call(cudaMemcpy(longaddress, &iMax, sizeof(long), cudaMemcpyHostToDevice),
				//		"cudaMemcpy(longaddress, &iMax, sizeof(long), cudaMemcpyHostToDevice)");
				//	f64 tempf64;
				//	cudaMemcpy(&tempf64, &(p_epsilon[iMax]), sizeof(f64), cudaMemcpyDeviceToHost);
				//	printf("\nError at iMax %1.14E\n\n", tempf64);
				//	bSpitOutErrorAfter = true;
				//	iMax0 = iMax;
				//};

				//printf("%d: %d  || ", ii, iMax);

				number_set++;
				// Just do this 24 times ... dumbest way possible but nvm.
				
				// Quicker way (develop later) :				
				// Just recruit neighbours until we get to 24?
				//if (iMax >= BEGINNING_OF_CENTRAL) {
				//	cudaMemcpy(p_izTri_host, &(pX_use->p_izTri_vert[MAXNEIGH*iMax]), sizeof(long)*MAXNEIGH, cudaMemcpyDeviceToHost);
				//		// Only pick up neighbours if they are not already used.
				//		// Stop when we reach SMASH_POINTS
				//	{
				//		number_set++;
				//	}
				//} else {
				//	{
				// Only pick up neighbours if they are not already used.
				//		// Stop when we reach SMASH_POINTS
				//		number_set++;
				//	}
				//};
			} while (number_set < SQUASH_POINTS);
			//printf("\n");

			kernelComputeJacobianValues << <numTriTiles, threadsPerTileMinor >> > (
				pX_use->p_info,
				
			//	p_AzNext,pAz_k,p_Azdot0,  // needed?
				p_gamma,   // needed?
				hsub,
				p_indicator,
				pX_use->p_izTri_vert,
				pX_use->p_izNeigh_TriMinor,
				pX_use->p_szPBCtri_vert,
				pX_use->p_szPBC_triminor,
				p_Jacobian_list // needs to be NMINOR * SQUASH_POINTS
				);
			Call(cudaThreadSynchronize(), "cudaTS CollectJacobian");
			// Make sure we do not piss it up at the back.

			// *****************************************************************************************

			// Thing is:
			// The thing is of course, if you are at the back, then you just pissed it up because
			// you did not anticipate the change in frill values when you change an individual next
			// to the frill.

			// *****************************************************************************************
			// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


			AggregateSmashMatrix << <numTilesMinor * 2, threadsPerTileMajor >> > (
				p_Jacobian_list,
				p_epsilon,
				p_matrix_blocks,
				p_vector_blocks
				);
			Call(cudaThreadSynchronize(), "cudaTS CollectMatrix");

			cudaMemcpy(p_matrix_blocks_host, p_matrix_blocks,
				sizeof(f64)*SQUASH_POINTS*SQUASH_POINTS*numTilesMinor*2, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_vector_blocks_host, p_vector_blocks,
				sizeof(f64)*SQUASH_POINTS*numTilesMinor * 2, cudaMemcpyDeviceToHost);
			memset(matrix, 0, sizeof(f64)*SQUASH_POINTS*SQUASH_POINTS);
			memset(vector, 0, sizeof(f64)*SQUASH_POINTS);
			for (iTile = 0; iTile < numTilesMinor * 2; iTile++)
			{
			//	for (int j = 0; j < SQUASH_POINTS; j++)
			//		if (p_matrix_blocks_host[j * SQUASH_POINTS + j + iTile*SQUASH_POINTS*SQUASH_POINTS] != 0.0) 
			//			printf("iTile %d contrib to (%d, %d) %1.9E | \n", iTile,j,j, p_matrix_blocks_host[j*SQUASH_POINTS + j + iTile*SQUASH_POINTS*SQUASH_POINTS]);
				 
				for (int i = 0; i < SQUASH_POINTS; i++)
				{
					for (int j = 0; j < SQUASH_POINTS; j++)
						matrix[i*SQUASH_POINTS+j] += p_matrix_blocks_host[i*SQUASH_POINTS+j+iTile*SQUASH_POINTS*SQUASH_POINTS];
					
					// Note that the matrix is symmetric so i, j order doesn't matter anyway.
					vector[i] -= p_vector_blocks_host[i+iTile*SQUASH_POINTS];

					// INSERTED THE MINUS HERE.

				};
			}; 
		//	printf("\n");

			lapack_int ipiv[SQUASH_POINTS];

			lapack_int Nrows = SQUASH_POINTS,
				Ncols = SQUASH_POINTS,  // lda
				Nrhscols = 1, // ldb
				Nrhsrows = SQUASH_POINTS, info;
			
		//	printf("going to call dgesv:\n");

			//for (int i = 0; i < SQUASH_POINTS; i++)
			//{
			//	for (int j = 0; j < SQUASH_POINTS; j++)
			//	{
			//		printf("%1.2E ", matrix[i*SQUASH_POINTS + j]);
			//	};
			//	printf("\n");
			//};

			info = LAPACKE_dgesv(LAPACK_ROW_MAJOR,
				Nrows, 1, matrix,
				Ncols, ipiv, //sum_eps_deps_by_dbeta_vector
				vector, Nrhscols);
			// Check for the exact singularity :
			if (info > 0) {
				printf("The diagonal element of the triangular factor of A,\n");
				printf("U(%i,%i) is zero, so that A is singular;\n", info, info);
				printf("the solution could not be computed.\n\a");
				getch();
			} 	else {
			//	printf("LAPACKE_dgesv ran successfully.\n");
				memcpy(beta, vector, SQUASH_POINTS * sizeof(f64));

			//	printf("===================\n");
			//	for (int j = 0; j < SQUASH_POINTS; j++)
			//		printf("%d : change %1.12E \n", j, beta[j]);
			//	printf("===================\n");
				
				CallMAC(cudaMemcpyToSymbol(beta_n_c, beta, SQUASH_POINTS * sizeof(f64)));
				// proper name for the result.
				// But beta is the set of coefficients on a set of individual dummies
				kernelAddToAz << <numTilesMinor, threadsPerTileMinor >> > (
					p_indicator,
					p_AzNext
					);
				Call(cudaThreadSynchronize(), "cudaTS AddToAz");
				// Think we probably are missing a minus: did we include it in the RHS vector?

				kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
					pX_use->p_info, pX_use->p_tri_neigh_index,
					p_AzNext);
				Call(cudaThreadSynchronize(), "cudaTS ResetFrills Az");

			}

		} else {

			kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
				pX_use->p_info,
				p_Jacobi_x,
				pX_use->p_izTri_vert,
				pX_use->p_izNeigh_TriMinor,
				pX_use->p_szPBCtri_vert,
				pX_use->p_szPBC_triminor,
				p_LapJacobi
				);
			Call(cudaThreadSynchronize(), "cudaTS Get Lap Jacobi 1");

			kernelCreate_further_regressor << <numTilesMinor, threadsPerTileMinor >> > (
				pX_use->p_info,
				hsub,
				p_Jacobi_x,
				p_LapJacobi,
				p_LapCoeffself,
				p_gamma,
				p_regressor_n);
			Call(cudaThreadSynchronize(), "cudaTS Create further regressor");
			// Look over it just to be sure what applies.
			
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
				p_temp4
				);
			Call(cudaThreadSynchronize(), "cudaTS Get Lap Jacobi 2");
//
//			kernelCreate_further_regressor << <numTilesMinor, threadsPerTileMinor >> > (
//				pX_use->p_info,
//				hsub,
//				p_regressor_n,
//				p_temp4,
//				p_LapCoeffself,
//				p_gamma,
//				p_regressor_i);
//			Call(cudaThreadSynchronize(), "cudaTS Create further regressor");
//

			MultiplyVector << <numTilesMinor, threadsPerTileMinor >> >
				(p_Jacobi_x, p_epsilon, p_regressor_i);
			Call(cudaThreadSynchronize(), "cudaTS Multiply Jacobi*epsilon");

			// Wipe out regressor_i with epsilon: J2RLS:
			// Doesn't help.
			// cudaMemcpy(p_regressor_i, p_epsilon, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);

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
				p_temp5
				);
			Call(cudaThreadSynchronize(), "cudaTS Get Lap Jacobi 1");

			// Okay ... now we need to do the routine that creates the matrix deps/dbeta_i deps/dbeta_j
			// and the vector against epsilon

			kernelAccumulateMatrix << <numTilesMinor, threadsPerTileMinor >> > (
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
			//printf(
			//	" ( %1.6E %1.6E %1.6E ) ( beta0 )   ( %1.6E )\n"
			//	" ( %1.6E %1.6E %1.6E ) ( beta1 ) = ( %1.6E )\n"
			//	" ( %1.6E %1.6E %1.6E ) ( beta2 )   ( %1.6E )\n",
			//	mat.xx, mat.xy, mat.xz, sumvec.x,
			//	mat.yx, mat.yy, mat.yz, sumvec.y,
			//	mat.zx, mat.zy, mat.zz, sumvec.z);
			f64_vec3 product = mat2*sumvec;

			beta[0] = -product.x; beta[1] = -product.y; beta[2] = -product.z;

			if (iIteration % 10 == 0) 
				printf("iIteration = %d L2eps %1.9E beta %1.8E %1.8E %1.8E \n", iIteration, L2eps, beta[0], beta[1], beta[2]);

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

			kernelAddRegressors << <numTilesMinor, threadsPerTileMinor >> > (
				p_AzNext,
				beta[0], beta[1], beta[2],
				p_Jacobi_x,
				p_regressor_n,
				p_regressor_i
				);
			Call(cudaThreadSynchronize(), "cudaTS kernelAddRegressors");
		};
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


	//	missing a ResetFrills. Though it follows linearly?


		// 1. Create regressor:
		// Careful with major vs minor + BEGINNING_OF_CENTRAL:
		 
		kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
			pX_use->p_info,
			p_AzNext,
			pX_use->p_izTri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBCtri_vert,
			pX_use->p_szPBC_triminor,
			p_LapAzNext
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
		// Check again.


		//// Now printf what's going on at the back:
		//SetConsoleTextAttribute(hConsole, 10);
		// 
		//f64 eps1, eps2, A1, A2, LapA1, LapA2, Azdot01, Azdot02;
		//cudaMemcpy(&eps1, &(p_epsilon[VERTCHOSEN + BEGINNING_OF_CENTRAL]), sizeof(f64), cudaMemcpyDeviceToHost);
		//cudaMemcpy(&eps2, &(p_epsilon[VERTCHOSEN2 + BEGINNING_OF_CENTRAL]), sizeof(f64), cudaMemcpyDeviceToHost);
		//cudaMemcpy(&A1, &(p_AzNext[VERTCHOSEN + BEGINNING_OF_CENTRAL]), sizeof(f64), cudaMemcpyDeviceToHost);
		//cudaMemcpy(&A2, &(p_AzNext[VERTCHOSEN2 + BEGINNING_OF_CENTRAL]), sizeof(f64), cudaMemcpyDeviceToHost);
		//cudaMemcpy(&LapA1, &(p_LapAzNext[VERTCHOSEN + BEGINNING_OF_CENTRAL]), sizeof(f64), cudaMemcpyDeviceToHost);
		//cudaMemcpy(&LapA2, &(p_LapAzNext[VERTCHOSEN2 + BEGINNING_OF_CENTRAL]), sizeof(f64), cudaMemcpyDeviceToHost);
		//cudaMemcpy(&Azdot01, &(p_Azdot0[VERTCHOSEN + BEGINNING_OF_CENTRAL]), sizeof(f64), cudaMemcpyDeviceToHost);
		//cudaMemcpy(&Azdot02, &(p_Azdot0[VERTCHOSEN2 + BEGINNING_OF_CENTRAL]), sizeof(f64), cudaMemcpyDeviceToHost);
		//printf("%d: eps %1.9E Az %1.9E Lap %1.9E Azdot0 %1.9E\n",
		//	VERTCHOSEN, eps1, A1, LapA1, Azdot01);
		//printf("%d: eps %1.9E Az %1.9E Lap %1.9E Azdot0 %1.9E\n",
		//	VERTCHOSEN2, eps2, A2, LapA2, Azdot02); 
		//SetConsoleTextAttribute(hConsole, 15);


		kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> > (
			pX_use->p_info, pX_use->p_tri_neigh_index, p_Jacobi_x);
		Call(cudaThreadSynchronize(), "cudaTS ResetFrills Jacobi");

		kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> >
			(p_epsilon, p_temp1);
		cudaMemcpy(p_temphost3, p_temp1, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		RSS = 0.0; 
		for (iTile = 0; iTile < numTilesMinor; iTile++) RSS += p_temphost3[iTile];
		L2eps = sqrt(RSS / (f64)NMINOR);
		if (iIteration %10 == 0) printf("L2eps: %1.9E .. ", L2eps);

		kernelAccumulateSumOfQuarts << <numTilesMinor, threadsPerTileMinor >> >
			(p_epsilon, p_temp1);
		cudaMemcpy(p_temphost3, p_temp1, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		f64 RSQ = 0.0;
		for (iTile = 0; iTile < numTilesMinor; iTile++) RSQ += p_temphost3[iTile];
		f64 L4eps = sqrt(sqrt(RSQ / (f64)NMINOR));
		if (iIteration % 10 == 0) printf("L4eps: %1.9E  ratio L4/L2 %1.9E \n", L4eps, L4eps / L2eps);
		L4L2ratio = L4eps / L2eps;

		//if (bSpitOutErrorAfter)
	//	{
//			f64 tempf64;
		//	cudaMemcpy(&tempf64, &(p_epsilon[iMax0]), sizeof(f64), cudaMemcpyDeviceToHost);
	//		printf("\nError at iMax0 %1.14E\n\n", tempf64);
//		}

		if (iIteration == 4001){ //(iIteration > 600) {
			
			// graphs:
			cudaMemcpy(p_temphost1, p_epsilon, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost2, p_Azdot0, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost3, p_regressor_n, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost4, p_Az, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost5, p_regressor_i, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost6, p_Jacobi_x, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
						
			SetActiveWindow(hWnd);
			RefreshGraphs(X1, AZSOLVERGRAPHS);
			SetActiveWindow(hWnd);
			Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);
			printf("done graph\n");
		};

		if (iIteration == 4001) {
			// let's print some stats

			cudaMemcpy(p_temphost3, p_epsilon, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			// Find max, avg, SD
			long iMinor, iMax = 0;
			f64 max = 0.0;
			f64 sum = 0.0, sumsq = 0.0;
			for (iMinor = 0; iMinor < NMINOR; iMinor++)
			{
				if (fabs(p_temphost3[iMinor]) > max) {
					max = fabs(p_temphost3[iMinor]);
					iMax = iMinor;
				};
				sum += p_temphost3[iMinor];
				sumsq += p_temphost3[iMinor] * p_temphost3[iMinor];
			}
			structural info;
			f64 var = sumsq / (real)NMINOR - sum*sum / (real)(NMINOR*NMINOR);
			cudaMemcpy(&info, &(pX_use->p_info[iMax]), sizeof(structural), cudaMemcpyDeviceToHost);
			printf("Avg %1.9E Max %1.10E iMax %d flag %d pos %1.9E %1.9E SD %1.9E \n",
				sum / (real)NMINOR, p_temphost3[iMax], iMax, info.flag, info.pos.x, info.pos.y, sqrt(var));

			while (1) getch();
		}
	



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
//			printf("failed test\n");
			bContinue = true;
		};
		iIteration++;

	} while (bContinue);
//	fclose(fpdbg);

	GlobalSuppressSuccessVerbosity = DEFAULTSUPPRESSVERBOSITY;
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
		p_LapAzNext
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
		p_temp5
		);
	Call(cudaThreadSynchronize(), "cudaTS Get Lap 1");

	kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
		pX_use->p_info,
		p_x1,
		pX_use->p_izTri_vert,
		pX_use->p_izNeigh_TriMinor,
		pX_use->p_szPBCtri_vert,
		pX_use->p_szPBC_triminor,
		p_temp4
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
		p_LapJacobi
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
	//printf(
	//	" ( %1.6E %1.6E %1.6E ) ( beta0 )   ( %1.6E )\n"
	//	" ( %1.6E %1.6E %1.6E ) ( beta1 ) = ( %1.6E )\n"
	//	" ( %1.6E %1.6E %1.6E ) ( beta2 )   ( %1.6E )\n",
	//	mat.xx, mat.xy, mat.xz, sumvec.x,
	//	mat.yx, mat.yy, mat.yz, sumvec.y,
	//	mat.zx, mat.zy, mat.zz, sumvec.z);
	f64_vec3 product = mat2*sumvec;

	beta[0] = -product.x; beta[1] = -product.y; beta[2] = -product.z;

	//printf("beta %1.8E %1.8E %1.8E ", beta[0], beta[1], beta[2]);

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

int RunBackwardForHeat_ConjugateGradient(
	T3 * p_T_k, T3 * p_T, f64 hsub, cuSyst * pX_use,
	bool bUseMask 
	)
{
#define UPLIFT_THRESHOLD 0.33
#define NO_EQUILIBRATE 
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
	f64 old_heatrate, new_heatrate, change_heatrate;

	Vertex * pVertex;
	long iVertex;
	plasma_data * pdata;

	//GlobalSuppressSuccessVerbosity = true;

	// Be sure to bring equilibrate back.


	bool btemp;
	f64 f64temp;

	//cudaMemcpy(&btemp, &(p_boolarray2[25587 + 2*NUMVERTICES]), sizeof(bool), cudaMemcpyDeviceToHost);
	//printf("25587 : %d \n", btemp ? 1 : 0);

	// Assume p_T contains seed.
#define NO_EQUILIBRATE


	printf("\nConjugate gradient for heat: ");
	//long iMinor;
	f64 L2eps;// _elec, L2eps_neut, L2eps_ion;
	bool bFailedTest;
	Triangle * pTri;
	f64_vec4 tempf64vec4;
	f64 tempf64;
	f64 SS_n, SS_i, SS_e, oldSS_n, oldSS_i, oldSS_e, ratio_n, ratio_i, ratio_e,
		alpha_n, alpha_i, alpha_e, dot_n, dot_i, dot_e, RSS_n, RSS_i, RSS_e;

	// How to determine diagonal coefficient in our equations? In order to equilibrate.

	cudaMemset(p_epsilon_n, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_epsilon_i, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_epsilon_e, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Ap_n, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Ap_i, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Ap_e, 0, sizeof(f64)*NUMVERTICES);

#ifndef NO_EQUILIBRATE
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

	// In all honesty, we should only call this the once, not both here and in JLS repeatedly.

	// D = coeffself
	// We are going to need to make the following changes:
	//  : where sqrt(N)T_k applies, multiply by 1/sqrt(D_i)
	//  : Use sqrt(D_i)sqrt(Ni)T_i as the indt variable
	//     (when we divide by sqrt(D_i)sqrt(N_i) to obtain T,
	//     we can then use that to get the input to epsilon
	//  : But when we calc epsilon, multiply all by 1/sqrt(D_i)

	// So all in all it seems 1/sqrt(D_i) is the factor we would like to save.

	kernelPowerminushalf << <numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_coeffself_n, p_sqrtD_inv_n);
	Call(cudaThreadSynchronize(), "cudaTS Powerminushalf n");

	kernelPowerminushalf << <numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_coeffself_i, p_sqrtD_inv_i);
	Call(cudaThreadSynchronize(), "cudaTS Powerminushalf i");

	kernelPowerminushalf << <numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_coeffself_e, p_sqrtD_inv_e);
	Call(cudaThreadSynchronize(), "cudaTS Powerminushalf e");

	// seed: just set T to T_k.
	kernelUnpacktorootDN_T << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_sqrtDN_Tn, p_sqrtDN_Ti, p_sqrtDN_Te, p_T,
			p_coeffself_n,
			p_coeffself_i,
			p_coeffself_e,
			pX_use->p_AreaMajor,
			pX_use->p_n_major);
	Call(cudaThreadSynchronize(), "cudaTS UnpacktoDNT");

	// Was all very well but now we wanted it to be root NT times sqrt(coeffself)

	cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
	// Unimportant that it fills in 0 for the masked values.

	kernelCreateTfromNTbydividing_bysqrtDN << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_Tn, p_Ti, p_Te, p_sqrtDN_Tn, p_sqrtDN_Ti, p_sqrtDN_Te, 
			pX_use->p_AreaMajor,
			pX_use->p_n_major,
			p_sqrtD_inv_n, p_sqrtD_inv_i, p_sqrtD_inv_e
			);
	Call(cudaThreadSynchronize(), "cudaTS CreateTfromNT");

#else
	
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

	// Unimportant that it fills in 0 for the masked values.

	kernelCreateTfromNTbydividing << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_Tn, p_Ti, p_Te, p_NnTn, p_NTi, p_NTe, // divide by root N
			pX_use->p_AreaMajor,
			pX_use->p_n_major);
	Call(cudaThreadSynchronize(), "cudaTS CreateTfromNT");

#endif
	
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
			pX_use->p_AreaMajor,
			p_boolarray2,
			p_boolarray_block,
			bUseMask
			);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

	cudaMemcpy(&old_heatrate, &(NT_addition_rates_d_temp[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);

	// Let's say we don't know what is in NT_rates_d_temp outside of mask, but it is unset.


	// Let's think about this carefully. If we could do it with the same data loaded each vertex for neigh list it would actually save time.
	// We rely on the fact that loading the vertex list data is contiguous fetch.
	// And we do store it for 3 species - so let's stick with all in 1.

#ifndef NO_EQUILIBRATE
	kernelCreateEpsilonHeat_Equilibrated << <numTilesMajorClever, threadsPerTileMajorClever >> >
		(hsub,
			pX_use->p_info + BEGINNING_OF_CENTRAL,
			p_epsilon_n, p_epsilon_i, p_epsilon_e, // outputs - ensure we have set 0 in mask 
			p_sqrtDN_Tn, p_sqrtDN_Ti, p_sqrtDN_Te,
			p_T_k,					// p_T_k was not swapped so that might be why different result. All this is tough.
			pX_use->p_AreaMajor,
			pX_use->p_n_major,
			p_sqrtD_inv_n, p_sqrtD_inv_i, p_sqrtD_inv_e,
			NT_addition_rates_d_temp, // it's especially silly having a whole struct of 5 instead of 3 here.
			0,
			p_boolarray2,
			p_boolarray_block,
			bUseMask // NOTE THAT MOSTLY EPSILON ARE UNSET.
			);
	Call(cudaThreadSynchronize(), "cudaTS CreateEpsilon");

#else
	cudaMemcpy(&f64temp, &(p_epsilon_e[25587]), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\n\n25587 : p_epsilon_e[25587] %1.13E \n\n", f64temp);

	kernelCreateEpsilonHeat << <numTilesMajorClever, threadsPerTileMajorClever >> >
		(hsub,
			pX_use->p_info + BEGINNING_OF_CENTRAL,
			p_epsilon_n, p_epsilon_i, p_epsilon_e,
			p_NnTn, p_NTi, p_NTe,
			p_T_k,					// p_T_k was not swapped so that might be why different result. All this is tough.
			pX_use->p_AreaMajor,
			pX_use->p_n_major,
			NT_addition_rates_d_temp, // it's especially silly having a whole struct of 5 instead of 3 here.
			0, // p_bFailed
			p_boolarray2,
			p_boolarray_block,
			bUseMask // calc eps = 0 if mask is on and maskbool = 0
			);
	Call(cudaThreadSynchronize(), "cudaTS CreateEpsilon"); // sets 0 outside of mask.
	
#endif

	// --==
	// p = eps
	cudaMemcpy(p_regressor_n, p_epsilon_n, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_regressor_i, p_epsilon_i, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_regressor_e, p_epsilon_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);




	//cudaMemcpy(&f64temp, &(p_epsilon_e[25587]), sizeof(f64), cudaMemcpyDeviceToHost);
	//printf("\n\n25587 : p_epsilon_e[25587] %1.13E \n\n", f64temp);



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

		//printf("iTile: %d p_temphost %1.10E %1.10E %1.10E \n", iTile, p_temphost1[iTile], p_temphost2[iTile], p_temphost3[iTile]);
		// Let's see if there is a specific tile where we find differences from one time to the next...
	}

	oldSS_n = SS_n; oldSS_i = SS_i; oldSS_e = SS_e;
	long iIteration = 0;

	if (bUseMask) {
		printf("iIteration %d : L2eps[sqrtN T] %1.9E %1.9E %1.9E \n", iIteration,
			sqrt(SS_n * over_iEquations_n),
			sqrt(SS_i * over_iEquations_i),
			sqrt(SS_e * over_iEquations_e));
	} else {
		printf("iIteration %d : L2eps[sqrtN T] %1.9E %1.9E %1.9E \n", iIteration,
			sqrt(SS_n / (f64)NUMVERTICES),
			sqrt(SS_i / (f64)NUMVERTICES),
			sqrt(SS_e / (f64)NUMVERTICES));
	};

	f64 Store_SS_e = SS_e;
	
	bool bContinue = true;
	do {

		// See if we can get rid of additional ROCWRTregressor routine.
		// Ap = -p + (h / N) flux(p);
		// remember -Ax is what appears in epsilon.

		// normally eps = T_k+1 - T_k - (h/N) flux(T_k+1)

		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
 
		// regressor represents the addition to NT, we need to pass T = regressor/N

#ifndef NO_EQUILIBRATE

		kernelCreateTfromNTbydividing_bysqrtDN << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_Tn, p_Ti, p_Te, p_regressor_n, p_regressor_i, p_regressor_e,
				pX_use->p_AreaMajor,
				pX_use->p_n_major,
				p_sqrtD_inv_n, p_sqrtD_inv_i, p_sqrtD_inv_e
				);
		Call(cudaThreadSynchronize(), "cudaTS CreateTfromDNT reg");

#else
		//cudaMemcpy(&f64temp, &(p_regressor_e[25587]), sizeof(f64), cudaMemcpyDeviceToHost);
		//printf("\n\n25587 : p_regressor_e[25587] %1.13E \n\n", f64temp);

		kernelCreateTfromNTbydividing << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_Tn, p_Ti, p_Te, p_regressor_n, p_regressor_i, p_regressor_e,
				pX_use->p_AreaMajor, // divide by root N!
				pX_use->p_n_major);
		Call(cudaThreadSynchronize(), "cudaTS CreateTfromNT reg");


		// This is 0 outside of mask... because we propose to add 0 outside of mask.

		// debug:
		// now let's copy off this :
		cudaMemcpy(p_temp5, p_Te, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
		// we'll test what happens to heat rate if we add alpha_e*p_temp5 to T.

		//cudaMemcpy(&f64temp, &(p_Te[25587]), sizeof(f64), cudaMemcpyDeviceToHost);
		//printf("\n\n25587 : p_Te (reg) [25587] %1.13E \n\n", f64temp);
		

#endif
		// Note that we passed 0 in the masked cells so we are expecting this to fail when we look at them?
		// Well, those T are not going to change.

		printf("With values from regressors:\n");


		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES); // redundant
		kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_scalarT << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(
				pX_use->p_info,
				pX_use->p_izNeigh_vert,
				pX_use->p_szPBCneigh_vert,
				pX_use->p_izTri_vert,
				pX_use->p_szPBCtri_vert,
				pX_use->p_cc,
				pX_use->p_n_major,
				p_Tn, p_Ti, p_Te,				 // values from regressors
				pX_use->p_B + BEGINNING_OF_CENTRAL, 
				p_kappa_n,
				p_kappa_i, p_kappa_e,
				p_nu_i, p_nu_e,
				NT_addition_rates_d_temp,          // this is zero in mask
				pX_use->p_AreaMajor,
				p_boolarray2,
				p_boolarray_block,
				bUseMask);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate 1");

#ifndef NO_EQUILIBRATE
		cudaMemset(zero_array, 0, sizeof(T3)*NUMVERTICES);
		kernelCreateEpsilonHeat_Equilibrated << <numTilesMajorClever, threadsPerTileMajorClever >> >
			(hsub,
				pX_use->p_info + BEGINNING_OF_CENTRAL,
				p_Ap_n, p_Ap_i, p_Ap_e, // the result
				p_regressor_n, p_regressor_i, p_regressor_e,
				zero_array, // 0 which we multiply by N...
				pX_use->p_AreaMajor,
				pX_use->p_n_major, // we load this only in order to multiply with 0
				p_sqrtD_inv_n, p_sqrtD_inv_i, p_sqrtD_inv_e,
				NT_addition_rates_d_temp,
				0,
				p_boolarray2,
				p_boolarray_block,
				bUseMask); // for masked cells, everything = 0
		Call(cudaThreadSynchronize(), "cudaTS CreateEpsilonHeat_Equilibrated");
#else
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
				0,
				p_boolarray2,
				p_boolarray_block,
				bUseMask); // for masked cells, everything = 0
		Call(cudaThreadSynchronize(), "cudaTS CreateEpsilonHeat");
#endif

		cudaMemcpy(&change_heatrate, &(NT_addition_rates_d_temp[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		
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
		
		NegateVectors << <numTilesMajorClever, threadsPerTileMajorClever >> > 
			(p_Ap_n, p_Ap_i, p_Ap_e);
		Call(cudaThreadSynchronize(), "cudaTS NegateVectors");

		kernelAccumulateDotProducts << <numTilesMajorClever, threadsPerTileMajorClever >> >
			(	p_regressor_n, p_Ap_n,
				p_regressor_i, p_Ap_i,
				p_regressor_e, p_Ap_e,
				p_temp1, p_temp2, p_temp3	);
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

		alpha_n = (dot_n != 0.0) ? (SS_n / dot_n) : 0.0;
		alpha_i = (dot_i != 0.0) ? (SS_i / dot_i) : 0.0;
		alpha_e = (dot_e != 0.0) ? (SS_e / dot_e) : 0.0;
				
		printf("alpha %1.8E %1.8E %1.8E SS %1.8E %1.8E %1.8E dot %1.8E %1.8E %1.8E\n", alpha_n, alpha_i, alpha_e,
			SS_n, SS_i, SS_e, dot_n, dot_i, dot_e); 
				
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


		// DEBUG:
		kernelCreateTfromNTbydividing << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_Tn, p_Ti, p_Te, p_NnTn, p_NTi, p_NTe,
				pX_use->p_AreaMajor,
				pX_use->p_n_major);
		Call(cudaThreadSynchronize(), "cudaTS CreateTfromNT");
		
		VectorAddMultiple1 << <numTilesMajorClever, threadsPerTileMajorClever >> > (
			p_Te, alpha_e, p_temp5);
		Call(cudaThreadSynchronize(), "cudaTS AddMultiple ");
	
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
				pX_use->p_AreaMajor,
				p_boolarray2,
				p_boolarray_block,
				bUseMask);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

	//	cudaMemcpy(&new_heatrate, &(NT_addition_rates_d_temp[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);

		// Compare:
	//	printf("Test adding to T: heatrate difference %1.14E - %1.14E = %1.14E\n",
	//		new_heatrate, old_heatrate, new_heatrate - old_heatrate);
	//	printf("Multiply: alpha_e %1.14E * changerate %1.14E = %1.14E\n",
	//		alpha_e, change_heatrate, alpha_e*change_heatrate);
		// Will be interesting.

		// Now overwrite T:

		// end of debug section
		 
		VectorAddMultiple << <numTilesMajorClever, threadsPerTileMajorClever >> > (
			p_NnTn, alpha_n, p_regressor_n,
			p_NTi, alpha_i, p_regressor_i,
			p_NTe, alpha_e, p_regressor_e);
		Call(cudaThreadSynchronize(), "cudaTS AddMultiples");
		
		// Are we going to do this without using mask? 
		// What is in p_NTi in mask? Correct values hopefully.

#ifndef NO_EQUILIBRATE

		kernelCreateTfromNTbydividing_bysqrtDN << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_Tn, p_Ti, p_Te, p_sqrtDN_Tn, p_sqrtDN_Ti, p_sqrtDN_Te,
				pX_use->p_AreaMajor,
				pX_use->p_n_major,
				p_sqrtD_inv_n, p_sqrtD_inv_i, p_sqrtD_inv_e
				); 
		Call(cudaThreadSynchronize(), "cudaTS CreateTfromsqrtDN_T");
#else
		kernelCreateTfromNTbydividing << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_Tn, p_Ti, p_Te, p_NnTn, p_NTi, p_NTe,
				pX_use->p_AreaMajor,
				pX_use->p_n_major);
		Call(cudaThreadSynchronize(), "cudaTS CreateTfromNT");
#endif
		// Update Epsilon: or can simply recalculate eps = b - A newx
		// Sometimes would rather update epsilon completely:
		if (iIteration % 1 == 0) {

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
					pX_use->p_AreaMajor,
					p_boolarray2,
					p_boolarray_block,
					bUseMask);
			Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

	//		cudaMemcpy(&new_heatrate, &(NT_addition_rates_d_temp[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
	//		
	//		f64 diff_heatrate = new_heatrate - old_heatrate;
	//		f64 predict_heatrate = alpha_e*change_heatrate;
			
	//		printf("VERTCHOSEN %d \n", VERTCHOSEN);
	//		printf("new %1.13E - old %1.13E = %1.14E \n", new_heatrate, old_heatrate, diff_heatrate);
	//		printf("alpha_e %1.14E * changerate %1.14E = predict %1.14E\n", alpha_e, change_heatrate, predict_heatrate);

	//		old_heatrate = new_heatrate;

#ifndef NO_EQUILIBRATE
			cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever);
			kernelCreateEpsilonHeat_Equilibrated << <numTilesMajorClever, threadsPerTileMajorClever >> >
				(hsub,
					pX_use->p_info + BEGINNING_OF_CENTRAL,
					p_epsilon_n, p_epsilon_i, p_epsilon_e,
					p_sqrtDN_Tn, p_sqrtDN_Ti, p_sqrtDN_Te,
					p_T_k,
					pX_use->p_AreaMajor,
					pX_use->p_n_major,
					p_sqrtD_inv_n, p_sqrtD_inv_i, p_sqrtD_inv_e,
					NT_addition_rates_d_temp, // it's especially silly having a whole struct of 5 instead of 3 here.
					p_bFailed,
					p_boolarray2,
					p_boolarray_block,
					bUseMask);
			Call(cudaThreadSynchronize(), "cudaTS CreateEpsilon");
#else
			
			// DEBUG:
			cudaMemcpy(p_temp1, p_epsilon_n, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
			cudaMemcpy(p_temp2, p_epsilon_i, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
			cudaMemcpy(p_temp3, p_epsilon_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);
			VectorAddMultiple << <numTilesMajorClever, threadsPerTileMajorClever >> > (
				p_temp1, -alpha_n, p_Ap_n,
				p_temp2, -alpha_i, p_Ap_i,
				p_temp3, -alpha_e, p_Ap_e
				);
			Call(cudaThreadSynchronize(), "cudaTS AddMultiples 2");

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
					p_bFailed,
					p_boolarray2,
					p_boolarray_block,
					bUseMask);
			Call(cudaThreadSynchronize(), "cudaTS CreateEpsilon");

			VectorCompareMax << <numTilesMajorClever, threadsPerTileMajorClever >> > (
				p_temp2, 
				p_epsilon_i,
				p_longtemp, p_temp4 
				);
			Call(cudaThreadSynchronize(), "cudaTS CompareMax");
			cudaMemcpy(p_longtemphost, p_longtemp, sizeof(long)*numTilesMajorClever, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost4, p_temp4, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
			f64 maxo = 0.0;
			long iWhich;
			for (iTile = 0; iTile < numTilesMajorClever; iTile++)
			{
				if (p_temphost4[iTile] > maxo) {
					maxo = p_temphost4[iTile];
					iWhich = iTile;
				}
			};
			long iMaxVert = p_longtemphost[iWhich];
			printf(" ion  iMaxVert %d max %1.10E \n", iMaxVert, maxo);
		
			VectorCompareMax << <numTilesMajorClever, threadsPerTileMajorClever >> > (
				p_temp3,
				p_epsilon_e,
				p_longtemp, p_temp4
				);
			Call(cudaThreadSynchronize(), "cudaTS CompareMax");
			cudaMemcpy(p_longtemphost, p_longtemp, sizeof(long)*numTilesMajorClever, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost4, p_temp4, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
			maxo = 0.0;
			for (iTile = 0; iTile < numTilesMajorClever; iTile++)
			{
				if (p_temphost4[iTile] > maxo) {
					maxo = p_temphost4[iTile];
					iWhich = iTile;
				}
			};
			iMaxVert = p_longtemphost[iWhich];
			printf(" elec iMaxVert %d max %1.10E \n", iMaxVert, maxo);
			
#endif
			cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMajor, cudaMemcpyDeviceToHost);

			//// DEBUG:
			//cudaMemcpy(p_temphost4, p_epsilon_n, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
			//cudaMemcpy(p_temphost5, p_epsilon_i, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
			//cudaMemcpy(p_temphost6, p_epsilon_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);

			//FILE * fp = fopen("debug0.txt", "a");
			//for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
			//	fprintf(fp, "iVertex %d epsilon_old %1.13E %1.13E %1.13E epsilon %1.13E %1.13E %1.13E \n",
			//		iVertex, p_temphost1[iVertex], p_temphost2[iVertex], p_temphost3[iVertex], 
			//		p_temphost4[iVertex], p_temphost5[iVertex], p_temphost6[iVertex]);
			//
			//cudaMemcpy(p_temphost1, p_Ap_n, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
			//cudaMemcpy(p_temphost2, p_Ap_i, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
			//cudaMemcpy(p_temphost3, p_Ap_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);

			//cudaMemcpy(p_NTrates_host, NT_addition_rates_d_temp, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToHost);
			//cudaMemcpy(cuSyst_host.p_n_major, pX_use->p_n_major, sizeof(nvals)*NUMVERTICES, cudaMemcpyDeviceToHost);
			//cudaMemcpy(cuSyst_host.p_AreaMajor, pX_use->p_AreaMajor, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
			//

			//for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
			//{
			//	fprintf(fp, "iVertex %d alpha %1.13E %1.13E %1.13E p_Ap %1.13E %1.13E %1.13E Nn %1.13E N %1.13E ddtNnTn %1.13E ddtNiTi %1.14E ddtNeTe %1.14E \n",
			//		iVertex, alpha_n, alpha_i, alpha_e,
			//		p_temphost1[iVertex], p_temphost2[iVertex], p_temphost3[iVertex],
			//		cuSyst_host.p_AreaMajor[iVertex]*cuSyst_host.p_n_major[iVertex].n_n, cuSyst_host.p_AreaMajor[iVertex] * cuSyst_host.p_n_major[iVertex].n,
			//		p_NTrates_host[iVertex].NnTn, p_NTrates_host[iVertex].NiTi, p_NTrates_host[iVertex].NeTe);
			//}
			//fclose(fp);
			//printf("file saved (append)\n");

		} else {
			bContinue = true;
			VectorAddMultiple << <numTilesMajorClever, threadsPerTileMajorClever >> > (
				p_epsilon_n, -alpha_n, p_Ap_n,
				p_epsilon_i, -alpha_i, p_Ap_i,
				p_epsilon_e, -alpha_e, p_Ap_e
				);
			Call(cudaThreadSynchronize(), "cudaTS AddMultiples 2");
			p_boolhost[0] = true;
			// addition should be 0 in mask.

			// am I right that this gives different answer? we seem to drop erratically?


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
		
		ratio_n = (oldSS_n > 0.0) ? (SS_n / oldSS_n) : 0.0;
		ratio_i = (oldSS_i > 0.0) ? (SS_i / oldSS_i) : 0.0;
		ratio_e = (oldSS_e > 0.0) ? (SS_e / oldSS_e) : 0.0;

		// get it going first and then profile to see if we want to do more masking.

		if (bUseMask) {
			printf("iIteration %d : L2eps[sqrtN T] %1.9E %1.9E %1.9E \n", iIteration,
				sqrt(SS_n * over_iEquations_n),
				sqrt(SS_i * over_iEquations_i),
				sqrt(SS_e * over_iEquations_e));
		}
		else {
			printf("iIteration %d : L2eps[sqrtN T] %1.9E %1.9E %1.9E \n", iIteration,
				sqrt(SS_n / (f64)NUMVERTICES),
				sqrt(SS_i / (f64)NUMVERTICES),
				sqrt(SS_e / (f64)NUMVERTICES));
		};

		//printf("ratio %1.10E %1.10E %1.10E SS %1.10E %1.10E %1.10E\n", ratio_n, ratio_i, ratio_e,
		//	SS_n,SS_i,SS_e);

		kernelRegressorUpdate << <numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_regressor_n, p_regressor_i, p_regressor_e,
				p_epsilon_n, p_epsilon_i, p_epsilon_e,
				ratio_n, ratio_i, ratio_e,

				p_boolarray_block,
				bUseMask
				);
		Call(cudaThreadSynchronize(), "cudaTS RegressorUpdate");
		// regressor = epsilon + ratio*regressor; 

		bProgress = false;
		if ((oldSS_n > 0.0) && (sqrt(ratio_n) < REQUIRED_IMPROVEMENT_RATE)) bProgress = true;
		if ((oldSS_i > 0.0) && (sqrt(ratio_i) < REQUIRED_IMPROVEMENT_RATE)) bProgress = true;
		if ((oldSS_e > 0.0) && (sqrt(ratio_e) < REQUIRED_IMPROVEMENT_RATE)) bProgress = true;

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
		((iIteration < ITERATIONS_BEFORE_SWITCH) || (bProgress))
		);
	
	GlobalSuppressSuccessVerbosity = true;

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
	
	if ((bContinue == true) && (Store_SS_e < SS_e)) {
			printf("It got worse!\n");
	} else {
		kernelPackupT3 << < numTilesMajorClever, threadsPerTileMajorClever >> >
				(p_T, p_Tn, p_Ti, p_Te);	 // we did division since we updated sqrt(DN)T.
		Call(cudaThreadSynchronize(), "cudaTS PackupT3");
	};

	//Was working before. But it's SO temperamental. Now it makes things worse, masked or unmasked.
		
	if (bContinue == true) return 1;
	return 0;
}

#define REGRESSORS 8

__device__ f64 * regressors;
__device__ f64 * p_coeffself;
 
int RunBwdJnLSForHeat(f64 * p_T_k, f64 * p_T, f64 hsub, cuSyst * pX_use, bool bUseMask,
	int species, f64 * p_kappa, f64 * p_nu) // not sure if we can pass device pointers or not
{
	// The idea: pick Jacobi for definiteness. 
	// First write without equilibration, then compare two versions.
	// Try n = 6, 12 regressors.
	Matrix_real sum_ROC_products;
	f64 sum_eps_deps_by_dbeta_vector[24];
	f64 beta[24];

	long pointless_dummy[8] = { 19180, 28610, 28607, 32192, 32190, 32183, 28592 };//19163 fails

	printf("\nJLS^ %d for heat: \n", REGRESSORS);
	//long iMinor;
	f64 L2eps, L2reg;
	bool bFailedTest, bContinue;
	Triangle * pTri;
	f64 tempf64; 
	long iTile, i;

	int iIteration = 0;

	f64 tempdebug;
	Vertex * pVertex;
	long iVertex;
	plasma_data * pdata;
#define zerovec1 p_temp1

	CallMAC(cudaMemset(p_regressors, 0, sizeof(f64)*NUMVERTICES*(REGRESSORS+1)));
	CallMAC(cudaMemset(p_Ax, 0, sizeof(f64)*NUMVERTICES*REGRESSORS));
	CallMAC(cudaMemset(p_epsilon, 0, sizeof(f64)*NUMVERTICES));
	cudaMemset(zerovec1, 0, sizeof(f64)*NUMVERTICES);
	
	printf("iEquations[%d] %d\n", species, iEquations[species]);

	if (iEquations[species] <= REGRESSORS) {
		// solve with regressors that are Kronecker delta for each point.
		// Solution should be same as just solving equations directly.

		// CPU search for which elements and put them into a list.
		long equationindex[REGRESSORS];

		cudaMemcpy(p_boolhost, p_boolarray2 + species*NUMVERTICES, sizeof(bool)*NUMVERTICES, cudaMemcpyDeviceToHost);
		
		long iCaret = 0;
		for (i = 0; i < NUMVERTICES; i++)
		{
			if (p_boolhost[i]) {
				equationindex[iCaret] = i;
		//		printf("eqnindex[%d] = %d\n", iCaret, i);
				iCaret++;
			};
		}
		if (iCaret != iEquations[species]) {
			printf("(iCaret != iEquations[species])\n");
			getch(); getch(); getch(); getch(); getch(); return 1000;
		}
		else {
		//	printf("iCaret %d iEquations[%d] %d \n", iCaret, species, iEquations[species]);
		}

		f64 one = 1.0;
		for (i = 0; i < iCaret; i++) {
			cudaMemcpy(p_regressors + i*NUMVERTICES + equationindex[i], &one, sizeof(f64), cudaMemcpyHostToDevice);
		};
		// Then we want to fall out of branch into creating Ax.
		
		// And we want to make sure we construct the matrix with ID & 0 RHS for the unused equations.
		
		// Solution should be exact but then we can let it fall out of loop naturally?

		// Leave this as done and just skip regressor creation.

	} else {
		// Else we probably, should be using volleys especially if there is a mask set.
		
		//WE want to look into this:
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

		p_coeffself = p_coeffself_n;
		if (species == 1) p_coeffself = p_coeffself_i;
		if (species == 2) p_coeffself = p_coeffself_e;

		cudaMemcpy(&tempdebug, p_coeffself + pointless_dummy[0], sizeof(f64), cudaMemcpyDeviceToHost);
		printf("coeffself[%d] = %1.14E\n", pointless_dummy[0], tempdebug);
	};
	 
	::GlobalSuppressSuccessVerbosity = true; 
	char buffer[256];
	 
	iIteration = 0;
	do {
		printf("\nspecies %d ITERATION %d : ", species, iIteration);
		// create epsilon, & Jacobi 0th regressor.

//		And we want to look into this:
	//	and all the similar routines. What did we do near insulator?!

		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
		kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_1species << < numTilesMajorClever, threadsPerTileMajorClever >> >
		( 
			pX_use->p_info,
			pX_use->p_izNeigh_vert,
			pX_use->p_szPBCneigh_vert,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_cc,
			pX_use->p_n_major,
			p_T,
			pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
			p_kappa,
			p_nu,
			NT_addition_rates_d_temp,
			pX_use->p_AreaMajor,
			p_boolarray2 + NUMVERTICES*species, // FOR SPECIES NOW
			p_boolarray_block,
			bUseMask,
			species
			);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

		if (REGRESSORS < iEquations[species]) {
			// Note: most are near 1, a few are like 22 or 150.
			CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever));
			kernelCreateEpsilonAndJacobi_Heat_1species << < numTilesMajorClever, threadsPerTileMajorClever >> > (
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

				p_coeffself,

				p_epsilon,
				p_regressors,
				p_bFailed,
				p_boolarray2 + NUMVERTICES*species,
				p_boolarray_block,
				bUseMask,
				species,
				true // yes to eps in regressor
				);
			Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");
			// Note that for most cells it does NOTHING --- so we need Jacobi defined as 0
		} else {
			CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever));
			kernelCreateEpsilonHeat_1species << < numTilesMajorClever, threadsPerTileMajorClever >> > (
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
				p_epsilon,
				p_bFailed,
				p_boolarray2 + NUMVERTICES*species,
				p_boolarray_block,
				bUseMask,
				species
				);
			Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");
		};
		
		kernelAccumulateSumOfSquares1 << < numTilesMajorClever, threadsPerTileMajorClever >> > 
			(p_epsilon, p_sum_eps_eps);
		Call(cudaThreadSynchronize(), "cudaTS AccSum");
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		f64 sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMajorClever; iTile++)
			sum_eps_eps += p_sum_eps_eps_host[iTile];		
		if (bUseMask == 0) {
			L2eps = sqrt(sum_eps_eps / (real)NUMVERTICES);
		} else {
			f64 over = over_iEquations_n;
			if (species == 1) over = over_iEquations_i;
			if (species == 2) over = over_iEquations_e;
			L2eps = sqrt(sum_eps_eps * over);
		}
		printf(" L2eps %1.11E  : ", L2eps);

		// graph:
		// draw a graph:
		/*
		SetActiveWindow(hwndGraphics);

		cudaMemcpy(p_temphost1, p_epsilon, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);

		pVertex = pTriMesh->X;
		pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_temphost1[iVertex];
			pdata->temp.y = p_temphost1[iVertex];

			++pVertex;
			++pdata;
		}

		sprintf(buffer, "epsilon iteration %d", iIteration);
		Graph[0].DrawSurface(buffer,
			DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
			AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
			false,
			GRAPH_EPSILON, pTriMesh);

		cudaMemcpy(p_temphost1, p_regressors, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);

		pVertex = pTriMesh->X;
		pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_temphost1[iVertex];
			pdata->temp.y = p_temphost1[iVertex];

			++pVertex;
			++pdata;
		}
		sprintf(buffer, "Jac0 iteration %d", iIteration);
		Graph[1].DrawSurface(buffer,
			DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
			AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
			false,
			GRAPH_AZ, pTriMesh);
		cudaMemcpy(p_temphost1, p_regressors + NUMVERTICES, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);

		
		// temp5 is predicted difference
		// temp6 is old epsilon
		SubtractVector << < numTilesMajorClever, threadsPerTileMajorClever >> > (p_temp4, p_temp6, p_epsilon);
		// temp4 = actual difference & it's right-left, so new eps-old eps
		Call(cudaThreadSynchronize(), "subtractvector");
		SubtractVector << < numTilesMajorClever, threadsPerTileMajorClever >> > (p_temp3, p_temp5, p_temp4);
		// temp3 = predicted-actual
		Call(cudaThreadSynchronize(), "subtractvector");


		cudaMemcpy(p_temphost3, p_temp5, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);

		pVertex = pTriMesh->X;
		pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_temphost3[iVertex];
			pdata->temp.y = p_temphost3[iVertex];

			++pVertex;
			++pdata;
		}
		
		Graph[2].DrawSurface("predicted difference",
			DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
			AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
			false,
			GRAPH_AZ, pTriMesh);

		//cudaMemcpy(p_temphost1, p_regressors + 2 * NUMVERTICES, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
		   
		cudaMemcpy(p_temphost1, p_temp3, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
		pVertex = pTriMesh->X;
		pdata = pTriMesh->pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = p_temphost1[iVertex];
			pdata->temp.y = p_temphost1[iVertex];

			++pVertex;
			++pdata;
		}

		Graph[3].DrawSurface("difference from pred",
			DATA_HEIGHT, (real *)(&((pTriMesh->pData[0]).temp.x)),
			AZSEGUE_COLOUR, (real *)(&((pTriMesh->pData[0]).temp.y)),
			false,
			GRAPH_AZ, pTriMesh);

		SetActiveWindow(hwndGraphics);
		ShowWindow(hwndGraphics, SW_HIDE);
		ShowWindow(hwndGraphics, SW_SHOW);
		Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);

		printf("done graphs\n\n");

		getch();

		cudaMemcpy(p_temp6, p_epsilon, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToDevice);

		*/

		// Did epsilon now pass test? If so, skip to the end.

		bFailedTest = false;
		cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMajorClever; iTile++)
			if (p_boolhost[iTile]) bFailedTest = true;
		if (bFailedTest) {
			printf("bFailedTest true \n");
		} else {
			printf("bFailedTest false \n");
		};

		bContinue = bFailedTest; 
		if (bContinue) {

			// DEBUG:
			bool bUseVolleys = true;//(iIteration % 2 == 0);
			//if (bUseMask == 0) bUseVolleys = !bUseVolleys; // start without volleys for unmasked.
			
			// To prepare volley regressors we only need 2 x Jacobi:
			if (iEquations[species] > REGRESSORS) {
				
				for (i = 1; ((i <= REGRESSORS) || ((bUseVolleys) && (i <= 2))); i++)
				{
					
					// create depsilon/dbeta and Jacobi for this regressor
					CallMAC(cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES));
					kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_1species << < numTilesMajorClever, threadsPerTileMajorClever >> >
						(
							pX_use->p_info,
							pX_use->p_izNeigh_vert,
							pX_use->p_szPBCneigh_vert,
							pX_use->p_izTri_vert,
							pX_use->p_szPBCtri_vert,
							pX_use->p_cc,
							pX_use->p_n_major,
							p_regressors + (i - 1)*NUMVERTICES, // input as T
							pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
							p_kappa,
							p_nu,
							NT_addition_rates_d_temp, // output
							pX_use->p_AreaMajor,
							p_boolarray2 + NUMVERTICES*species,
							p_boolarray_block,
							bUseMask,
							species);
					// used for epsilon (T)
					Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

					kernelCreateEpsilonAndJacobi_Heat_1species << < numTilesMajorClever, threadsPerTileMajorClever >> > (
						// eps = T - (T_k +- h sum kappa dot grad T)
						// x = -eps/coeffself
						hsub,
						pX_use->p_info + BEGINNING_OF_CENTRAL,
						p_regressors + (i - 1)*NUMVERTICES, // input
						zerovec1,
						NT_addition_rates_d_temp,
						pX_use->p_n_major,
						pX_use->p_AreaMajor,
						p_coeffself,
						p_Ax + (i - 1)*NUMVERTICES, // the Ax for i-1; they will thus be defined for 0 up to 7 
						p_regressors + i*NUMVERTICES, // we need extra space here to create the last one we never use.
						0,
						p_boolarray2 + NUMVERTICES*species,
						p_boolarray_block,
						bUseMask,
						species,
						true // no to eps in regressor
						);
					Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");
					
#define DO_NOT_NORMALIZE_REGRESSORS
#ifndef DO_NOT_NORMALIZE_REGRESSORS

					kernelAccumulateSumOfSquares1 << < numTilesMajorClever, threadsPerTileMajorClever >> >
						(p_regressors + i*NUMVERTICES, p_sum_eps_eps);
					Call(cudaThreadSynchronize(), "cudaTS AccSum");
					cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
					f64 sum_eps_eps = 0.0;
					for (iTile = 0; iTile < numTilesMajorClever; iTile++)
						sum_eps_eps += p_sum_eps_eps_host[iTile];
					if (bUseMask == 0) {
						L2reg = sqrt(sum_eps_eps / (real)NUMVERTICES);
					}
					else {
						f64 over = over_iEquations_n;
						if (species == 1) over = over_iEquations_i;
						if (species == 2) over = over_iEquations_e;
						L2reg = sqrt(sum_eps_eps * over);
					};
					// Now in order to set L2reg = L2eps say, we want to multiply by
					// alpha = sqrt(L2eps/L2reg)
					f64 alpha = sqrt(L2eps / L2reg);
					kernelMultiplyVector << < numTilesMajorClever, threadsPerTileMajorClever >> > (p_regressors + i*NUMVERTICES, alpha);
					Call(cudaThreadSynchronize(), "cudaTS MultiplyVector");

#endif
				};
			};

			if ((iEquations[species] > REGRESSORS) && (bUseVolleys)) {
				// Now create volleys:
				kernelVolleyRegressors << < numTilesMajorClever, threadsPerTileMajorClever >> > (
					p_regressors,
					NUMVERTICES,
					pX_use->p_iVolley
					);
				Call(cudaThreadSynchronize(), "cudaTS volley regressors");
			};

			if ((iEquations[species] <= REGRESSORS) || (bUseVolleys)) {
				cudaMemset(p_Ax, 0, sizeof(f64)*NUMVERTICES*REGRESSORS); 
					// only first few columns actually needed it
				for (i = 0; ((i < iEquations[species]) && (i < REGRESSORS)); i++)
				{
					// create depsilon/dbeta and Jacobi for this regressor
					CallMAC(cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES));
					kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_1species << < numTilesMajorClever, threadsPerTileMajorClever >> >
						(
							pX_use->p_info,
							pX_use->p_izNeigh_vert,
							pX_use->p_szPBCneigh_vert,
							pX_use->p_izTri_vert,
							pX_use->p_szPBCtri_vert,
							pX_use->p_cc,
							pX_use->p_n_major,
							p_regressors + i*NUMVERTICES, // input as T
							pX_use->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
							p_kappa,
							p_nu,
							NT_addition_rates_d_temp, // output
							pX_use->p_AreaMajor,
							p_boolarray2 + NUMVERTICES*species,
							p_boolarray_block,
							bUseMask,
							species);
					// used for epsilon (T)
					Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

					kernelCreateEpsilonHeat_1species << < numTilesMajorClever, threadsPerTileMajorClever >> > (
						// eps = T - (T_k +- h sum kappa dot grad T)
						// x = -eps/coeffself
						hsub,
						pX_use->p_info + BEGINNING_OF_CENTRAL,
						p_regressors + i*NUMVERTICES, // input
						zerovec1,
						NT_addition_rates_d_temp,
						pX_use->p_n_major,
						pX_use->p_AreaMajor,
						p_Ax + i*NUMVERTICES, // the output
						0,
						p_boolarray2 + NUMVERTICES*species,
						p_boolarray_block,
						bUseMask,
						species
						);
					Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");


				};
				
			};


			lapack_int ipiv[REGRESSORS];
			double mat[REGRESSORS*REGRESSORS];
			lapack_int Nrows = REGRESSORS,
				Ncols = REGRESSORS,  // lda
				Nrhscols = 1, // ldb
				Nrhsrows = REGRESSORS, info;
			
			// We don't need to test for domain, we need to make sure the summands are zero otherwise.

			// if we introduce skipping blocks, must change to MajorClever
			kernelAccumulateSummands7 << <numTilesMajor, threadsPerTileMajor >> > (
				p_epsilon,
				p_Ax, // be careful: what do we take minus?
				p_sum_eps_deps_by_dbeta_x8,
				p_sum_depsbydbeta_8x8  // not sure if we want to store 64 things in memory?
				);			// L1 48K --> divide by 256 --> 24 doubles/thread in L1.
							// Better off running through multiple times and doing 4 saves. But it's optimization.
			Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands");
			// Say we store 24 doubles/thread. So 4x4?. We could multiply 2 sets of 4.
			// We are at 8 for now so let's stick with the 8-way routine.

			cudaMemcpy(p_sum_eps_deps_by_dbeta_x8_host, p_sum_eps_deps_by_dbeta_x8, sizeof(f64) * REGRESSORS * numTilesMajor, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_sum_depsbydbeta_8x8_host, p_sum_depsbydbeta_8x8, sizeof(f64) * REGRESSORS * REGRESSORS * numTilesMajor, cudaMemcpyDeviceToHost);
				
			memset(sum_eps_deps_by_dbeta_vector, 0, REGRESSORS * sizeof(f64));
			memset(mat, 0, REGRESSORS*REGRESSORS * sizeof(f64));
			sum_eps_eps = 0.0;
			int i, j;

			//for (iTile = 0; iTile < numTilesMajor; iTile++) {
			//	printf("iTile %d : %1.9E %1.9E\n",iTile,
			//		p_sum_depsbydbeta_8x8_host[iTile *  REGRESSORS *  REGRESSORS],
			//		p_sum_eps_deps_by_dbeta_x8_host[iTile*REGRESSORS]);
			//};

			for (i = 0; i < REGRESSORS; i++)
				for (j = 0; j < REGRESSORS; j++)
					for (iTile = 0; iTile < numTilesMajor; iTile++)
						mat[i*REGRESSORS + j] += p_sum_depsbydbeta_8x8_host[iTile *  REGRESSORS *  REGRESSORS + i *  REGRESSORS + j];

			for (iTile = 0; iTile < numTilesMajor; iTile++)
				for (i = 0; i < REGRESSORS; i++)
					sum_eps_deps_by_dbeta_vector[i] -= p_sum_eps_deps_by_dbeta_x8_host[iTile*REGRESSORS + i];
			// let's say they are in rows of 8 per tile.
			
	//		print_matrix("Entry Matrix A", Nrows, Ncols, mat, Ncols);
	//		print_matrix("Right Hand Side", Nrows, Nrhscols, sum_eps_deps_by_dbeta_vector, Nrhscols);
	//		printf("\n");

		// Here ensure that unwanted rows are 0. First wipe out any that accumulated 0.
			// or are beyond #equations.
			for (i = 0; i < REGRESSORS; i++)
			{
				if ((mat[i*REGRESSORS + i] == 0.0) || (i >= iEquations[species]))
				{
					memset(mat + i*REGRESSORS, 0, sizeof(f64)*REGRESSORS);
					mat[i*REGRESSORS + i] = 1.0;
					sum_eps_deps_by_dbeta_vector[i] = 0.0;
				}
			}
			// Note that if a colour was not relevant for volleys, that just covered it.
			
			f64 storeRHS[REGRESSORS];
			f64 storemat[REGRESSORS*REGRESSORS];
			memcpy(storeRHS, sum_eps_deps_by_dbeta_vector, sizeof(f64)*REGRESSORS);
			memcpy(storemat, mat, sizeof(f64)*REGRESSORS*REGRESSORS);

			// * Need to test speed against our own LU method.
			
			//	printf("LAPACKE_dgesv Results\n");
			// Solve the equations A*X = B 
			info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, Nrows, 1, mat, Ncols, ipiv, sum_eps_deps_by_dbeta_vector, Nrhscols);
			// Check for the exact singularity :

			if (info > 0) {
			//	printf("The diagonal element of the triangular factor of A,\n");
			//	printf("U(%i,%i) is zero, so that A is singular;\n", info, info);
				printf("the solution could not be computed.\n");
					
				if (bUseVolleys) {
					// Try deleting every other regressor
					memcpy(mat, storemat, sizeof(f64)*REGRESSORS*REGRESSORS);
					memcpy(sum_eps_deps_by_dbeta_vector, storeRHS, sizeof(f64)*REGRESSORS);

					memset(mat + 8, 0, sizeof(f64) * 8);
					memset(mat + 3*8, 0, sizeof(f64) * 8);
					memset(mat + 5*8, 0, sizeof(f64) * 8);
					memset(mat + 7*8, 0, sizeof(f64) * 8);
					mat[1 * 8 + 1] = 1.0;
					mat[3 * 8 + 3] = 1.0;
					mat[5 * 8 + 5] = 1.0;
					mat[7 * 8 + 7] = 1.0;
					sum_eps_deps_by_dbeta_vector[1] = 0.0;
					sum_eps_deps_by_dbeta_vector[3] = 0.0;
					sum_eps_deps_by_dbeta_vector[5] = 0.0;
					sum_eps_deps_by_dbeta_vector[7] = 0.0;

			//		print_matrix("Entry Matrix A", Nrows, Ncols, mat, Ncols);
			//		print_matrix("Right Hand Side", Nrows, Nrhscols, sum_eps_deps_by_dbeta_vector, Nrhscols);
			//		printf("\n");

			//		f64 storeRHS[REGRESSORS];
			//		f64 storemat[REGRESSORS*REGRESSORS];
					memcpy(storeRHS, sum_eps_deps_by_dbeta_vector, sizeof(f64)*REGRESSORS);
					memcpy(storemat, mat, sizeof(f64)*REGRESSORS*REGRESSORS);

					// * making sure that we had zeroed any unwanted colours already.

					printf("LAPACKE_dgesv Results (volleys for 1 regressor only) \n");
					// Solve the equations A*X = B 
					info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, Nrows, 1, mat, Ncols, ipiv, sum_eps_deps_by_dbeta_vector, Nrhscols);
					// Check for the exact singularity :
					if (info > 0) {
						printf("still didn't work..\n");
						print_matrix("Entry Matrix A", Nrows, Ncols, storemat, Ncols);
						print_matrix("Right Hand Side", Nrows, Nrhscols, storeRHS, Nrhscols);

						while (1) getch();
					};

					// Do not know whether my own LU is faster than LAPACKE dgesv.
				}; // (bUseVolleys)
			}; 
			
		//	print_matrix("Solution",Nrows, 1, sum_eps_deps_by_dbeta_vector, Nrhscols);
		//	print_matrix("Details of LU factorization",Nrows,Ncols,mat, Ncols);
		//	print_int_vector("Pivot indices",Nrows, ipiv);
			//
			if (info == 0) {
				memcpy(beta, sum_eps_deps_by_dbeta_vector, REGRESSORS * sizeof(f64));

				//sum_ROC_products.Invoke(REGRESSORS); // does set to zero
				// oooh memset(&sum_eps_deps_by_dbeta_vector, 0, REGRESSORS * sizeof(f64)); // why & ????
				//sum_eps_eps = 0.0;
				//int i, j;

				////for (iTile = 0; iTile < numTilesMajor; iTile++) {
				////	printf("iTile %d : %1.9E %1.9E\n",iTile,
				////		p_sum_depsbydbeta_8x8_host[iTile *  REGRESSORS *  REGRESSORS],
				////		p_sum_eps_deps_by_dbeta_x8_host[iTile*REGRESSORS]);
				////};

				//for (i = 0; i < REGRESSORS; i++)
				//	for (j = 0; j < REGRESSORS; j++)
				//		for (iTile = 0; iTile < numTilesMajor; iTile++) {
				//			sum_ROC_products.LU[i][j] += p_sum_depsbydbeta_8x8_host[iTile *  REGRESSORS *  REGRESSORS + i *  REGRESSORS + j];
				//		}
				//for (iTile = 0; iTile < numTilesMajor; iTile++)
				//	for (i = 0; i < REGRESSORS; i++)
				//		sum_eps_deps_by_dbeta_vector[i] -= p_sum_eps_deps_by_dbeta_x8_host[iTile*REGRESSORS + i];
				//// let's say they are in rows of 8 per tile.

				//for (i = 0; i < REGRESSORS; i++) {
				//	printf("{ ");
				//	for (j = 0; j < REGRESSORS; j++)
				//		printf(" %1.8E ", sum_ROC_products.LU[i][j]);
				//	printf(" } { beta%d } ", i);
				//	if (i == 3) { printf(" = "); }
				//	else { printf("   "); };
				//	printf("{ %1.8E }\n", sum_eps_deps_by_dbeta_vector[i]);
				//	// Or is it minus??
				//};

				//memset(beta, 0, sizeof(f64) * REGRESSORS);
				////if (L2eps > 1.0e-28) { // otherwise just STOP !
				//					   // 1e-30 is reasonable because 1e-15 * typical temperature 1e-14 = 1e-29.
				//					   // Test for a zero row:
				// 
				//bool zero_present = false;
				//for (i = 0; i <  REGRESSORS; i++)
				//{
				//	f64 sum = 0.0;
				//	for (j = 0; j < REGRESSORS; j++)
				//		sum += sum_ROC_products.LU[i][j];
				//	if (sum == 0.0) zero_present = true;
				//};
				//if (zero_present == false) {
				//	
				//	// DEBUG:
				//	printf("sum_ROC_products.LUdecomp() :");
				//	sum_ROC_products.LUdecomp();
				//	printf("done\n");
				//	printf("sum_ROC_products.LUSolve : ");
				//	sum_ROC_products.LUSolve(sum_eps_deps_by_dbeta_vector, beta);
				//	printf("done\n");

				//} else {
				//	printf("zero row present -- gah\n");
				//};

				printf("\nbeta: ");
				for (i = 0; i < REGRESSORS; i++)
					printf(" %1.8E ", beta[i]);
				printf("\n");

				CallMAC(cudaMemcpyToSymbol(beta_n_c, beta, REGRESSORS * sizeof(f64)));

				// add lc to our T

				kernelAddtoT_lc << <numTilesMajor, threadsPerTileMajor >> > (
					p_T, p_regressors);
				Call(cudaThreadSynchronize(), "cudaTS AddtoT");
			};
			
			// store predicted difference:
			ScaleVector << <numTilesMajor, threadsPerTileMajor >> > (p_temp5, beta[0], p_Ax);
			Call(cudaThreadSynchronize(), "ScaleVector");
			
			iIteration++;
		
		}; // if (bContinue)
	} while (bContinue);
	
	// To test whether this is sane, we need to spit out typical element in 0th and 8th iterate.
	
	GlobalSuppressSuccessVerbosity = DEFAULTSUPPRESSVERBOSITY;

	return 0;	
}


int RunBackwardJLSForHeat(T3 * p_T_k, T3 * p_T, f64 hsub, cuSyst * pX_use,
	bool bUseMask)
{
#define UPLIFT_THRESHOLD 0.33
//	GlobalSuppressSuccessVerbosity = true;



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

	CallMAC(cudaMemset(p_Jacobi_n, 0, sizeof(f64)*NUMVERTICES));
	CallMAC(cudaMemset(p_Jacobi_i, 0, sizeof(f64)*NUMVERTICES));
	CallMAC(cudaMemset(p_Jacobi_e, 0, sizeof(f64)*NUMVERTICES));

	CallMAC(cudaMemset(p_epsilon_n, 0, sizeof(f64)*NUMVERTICES));
	CallMAC(cudaMemset(p_epsilon_i, 0, sizeof(f64)*NUMVERTICES));
	CallMAC(cudaMemset(p_epsilon_e, 0, sizeof(f64)*NUMVERTICES)); // NEED TO LOOK AND DO SAME IN CG ROUTINE.


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
				pX_use->p_AreaMajor,
				p_boolarray2,
				p_boolarray_block,
				bUseMask
				);
		// used for epsilon (T) 
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");
		    
		// Note: most are near 1, a few are like 22 or 150.
		CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajor));
		kernelCreateEpsilonAndJacobi_Heat << < numTilesMajorClever, threadsPerTileMajorClever >> > (
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
			p_bFailed,
			p_boolarray2,
			p_boolarray_block,
			bUseMask
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");
		// Note that for most cells it does NOTHING --- so we need Jacobi defined as 0

		
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
		cudaMemset(p_d_eps_by_dbetaR_e, 0, sizeof(f64)*NUMVERTICES); // important to zero in mask!
		
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
				pX_use->p_AreaMajor,
				p_boolarray2,
				p_boolarray_block,
				bUseMask);
		// used for epsilon (T)
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");
		 
		// Note: most are near 1, a few are like 22 or 150.
		kernelCreateEpsilon_Heat_for_Jacobi << < numTilesMajorClever, threadsPerTileMajorClever >> > (
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

			p_d_eps_by_dbeta_n, p_d_eps_by_dbeta_i, p_d_eps_by_dbeta_e,
			p_boolarray2,
			p_boolarray_block,
			bUseMask
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
				pX_use->p_AreaMajor,
				p_boolarray2,
				p_boolarray_block,
				bUseMask);
		// used for epsilon (T)
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

		// Note: most are near 1, a few are like 22 or 150.
		kernelCreateEpsilon_Heat_for_Jacobi << < numTilesMajorClever, threadsPerTileMajorClever >> > (
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

			p_d_eps_by_dbetaR_n, p_d_eps_by_dbetaR_i, p_d_eps_by_dbetaR_e,
			p_boolarray2,
			p_boolarray_block,
			bUseMask
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

		cudaMemcpy(p_sum_eps_deps_by_dbeta_J_host, p_sum_eps_deps_by_dbeta_J, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_deps_by_dbeta_R_host, p_sum_eps_deps_by_dbeta_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_J_times_J_host, p_sum_depsbydbeta_J_times_J, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_R_times_R_host, p_sum_depsbydbeta_R_times_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_J_times_R_host, p_sum_depsbydbeta_J_times_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);

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
		printf("\nfor neutral [ BetaJ %1.6E BetaR %1.6E L2eps %1.10E ] ", beta_nJ, beta_nR, L2eps_n);


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

		cudaMemcpy(p_sum_eps_deps_by_dbeta_J_host, p_sum_eps_deps_by_dbeta_J, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_deps_by_dbeta_R_host, p_sum_eps_deps_by_dbeta_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_J_times_J_host, p_sum_depsbydbeta_J_times_J, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_R_times_R_host, p_sum_depsbydbeta_R_times_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_J_times_R_host, p_sum_depsbydbeta_J_times_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);

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

			printf("sum_eps_deps_by_dbeta_J %1.10E sum_depsbydbeta_J_times_J %1.10E \n", sum_eps_deps_by_dbeta_J, sum_depsbydbeta_J_times_J);

		};
		
		L2eps_i_old = L2eps_i;
		L2eps_i = sqrt(sum_eps_eps / (real)NUMVERTICES);

		printf("\n for ION [ BetaJ %1.6E BetaR %1.6E L2eps %1.10E ] ", beta_iJ, beta_iR, L2eps_i);


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
		
		cudaMemcpy(p_sum_eps_deps_by_dbeta_J_host, p_sum_eps_deps_by_dbeta_J, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_deps_by_dbeta_R_host, p_sum_eps_deps_by_dbeta_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_J_times_J_host, p_sum_depsbydbeta_J_times_J, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_R_times_R_host, p_sum_depsbydbeta_R_times_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_depsbydbeta_J_times_R_host, p_sum_depsbydbeta_J_times_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);

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

		printf("\nfor Electron [ BetaJ %1.6E BetaR %1.6E L2eps %1.10E ] ", beta_eJ, beta_eR, L2eps_e);

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
				if (ratio < REQUIRED_IMPROVEMENT_RATE_J) bProgress = true; // 1.5% progress enough to carry on
			};
			if (L2eps_i_old >= 1.0e-30) {
				ratio = (L2eps_i / L2eps_i_old);
				if (ratio < REQUIRED_IMPROVEMENT_RATE_J) bProgress = true; // 1.5% progress enough to carry on
			};
			if (L2eps_n_old >= 1.0e-30) {
				ratio = (L2eps_n / L2eps_n_old);
				if (ratio < REQUIRED_IMPROVEMENT_RATE_J) bProgress = true; // 1.5% progress enough to carry on
			};
		};

	} while ((iIteration < 2)
		|| ( (bFailedTest)
		&& ((iIteration < ITERATIONS_BEFORE_SWITCH) || (bProgress))
			 ) );

	kernelPackupT3 << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_T, p_Tn, p_Ti, p_Te);	 // we did division since we updated NT.
	Call(cudaThreadSynchronize(), "cudaTS packup");

	GlobalSuppressSuccessVerbosity = DEFAULTSUPPRESSVERBOSITY;

	return (bFailedTest == false) ? 0 : 1;



	// Modify denominator of L2 to reflect number of equations.



}


void RegressionSeedTe(f64 hsub, T3 * p_move1, T3 * p_move2, T3 * p_T, T3 * p_T_k, 
	cuSyst * pX_use, bool bUseMask)
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

	if (bUseMask) {
		cudaMemset(p_slot1n, 0, sizeof(f64)*NUMVERTICES);
		cudaMemset(p_slot1i, 0, sizeof(f64)*NUMVERTICES);
		cudaMemset(p_slot1e, 0, sizeof(f64)*NUMVERTICES);
		kernelUnpackWithMask << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_slot1n, p_slot1i, p_slot1e, p_move1,
				p_boolarray2,
				p_boolarray_block);
		Call(cudaThreadSynchronize(), "cudaTS unpack move1");

		cudaMemset(p_slot2n, 0, sizeof(f64)*NUMVERTICES);
		cudaMemset(p_slot2i, 0, sizeof(f64)*NUMVERTICES);
		cudaMemset(p_slot2e, 0, sizeof(f64)*NUMVERTICES);
		kernelUnpackWithMask << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_slot2n, p_slot2i, p_slot2e, p_move2,
				p_boolarray2,
				p_boolarray_block	);
		Call(cudaThreadSynchronize(), "cudaTS unpack move2");

	} else {
		kernelUnpack<< < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_slot1n, p_slot1i, p_slot1e, p_move1	);
		Call(cudaThreadSynchronize(), "cudaTS unpack move1");

		kernelUnpack << < numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_slot2n, p_slot2i, p_slot2e, p_move2);
		Call(cudaThreadSynchronize(), "cudaTS unpack move2");
	}

	// move2 never gets used.
	
	// Important to ensure regressors were 0 in mask.

	cudaMemset(p_T, 0, sizeof(T3)*NUMVERTICES); // use as zerovec
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
			pX_use->p_AreaMajor,
			p_boolarray2,
			p_boolarray_block,
			bUseMask);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

	cudaMemset(p_epsilon_n, 0, sizeof(f64)*NUMVERTICES); 
	cudaMemset(p_epsilon_i, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_epsilon_e, 0, sizeof(f64)*NUMVERTICES);
	kernelCreateEpsilon_Heat_for_Jacobi << < numTilesMajorClever, threadsPerTileMajorClever >> > (
		// eps = T - (T_k +- h sum kappa dot grad T)
		hsub,
		pX_use->p_info + BEGINNING_OF_CENTRAL,
		p_Tn, p_Ti, p_Te,
		p_T_k,
		NT_addition_rates_d_temp,
		pX_use->p_n_major, // got this
		pX_use->p_AreaMajor, // got this -> N, Nn
		p_epsilon_n, p_epsilon_i, p_epsilon_e,
		p_boolarray2,
		p_boolarray_block,
		bUseMask
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
			pX_use->p_AreaMajor,
			p_boolarray2,
			p_boolarray_block,
			bUseMask);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

	cudaMemset(p_slot2n, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_slot2i, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_slot2e, 0, sizeof(f64)*NUMVERTICES); 
		// because we are about to overwrite it

	kernelCreateEpsilonAndJacobi_Heat << < numTilesMajorClever, threadsPerTileMajorClever >> > (
		// eps = T - (T_k +- h sum kappa dot grad T)
		hsub,
		pX_use->p_info + BEGINNING_OF_CENTRAL,
		p_slot1n, p_slot1i, p_slot1e, // regressor
		p_T, // zerovec
		NT_addition_rates_d_temp,
		pX_use->p_n_major, // got this
		pX_use->p_AreaMajor, // got this -> N, Nn
		p_coeffself_n, p_coeffself_i, p_coeffself_e,
		p_d_eps_by_dbeta_n, p_d_eps_by_dbeta_i, p_d_eps_by_dbeta_e,
		p_slot2n, p_slot2i, p_slot2e,
		0,
		p_boolarray2,
		p_boolarray_block,
		bUseMask
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
			pX_use->p_AreaMajor,
			p_boolarray2,
			p_boolarray_block,
			bUseMask);
	// used for epsilon (T)
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");

	//// Note: most are near 1, a few are like 22 or 150.


	//cudaMemset(p_d_eps_by_dbetaR_n, 0, sizeof(f64)*NUMVERTICES);
	//cudaMemset(p_d_eps_by_dbetaR_i, 0, sizeof(f64)*NUMVERTICES);
	//cudaMemset(p_d_eps_by_dbetaR_e, 0, sizeof(f64)*NUMVERTICES);
	// We have done this.
	kernelCreateEpsilon_Heat_for_Jacobi << < numTilesMajorClever, threadsPerTileMajorClever >> > (
		// eps = T - (T_k +- h sum kappa dot grad T)
		// x = -eps/coeffself
		hsub,
		pX_use->p_info + BEGINNING_OF_CENTRAL,
		p_slot2n, p_slot2i, p_slot2e,
		p_T, // zerovec
		NT_addition_rates_d_temp,
		pX_use->p_n_major, // got this
		pX_use->p_AreaMajor, // got this -> N, Nn

		p_d_eps_by_dbetaR_n, p_d_eps_by_dbetaR_i, p_d_eps_by_dbetaR_e,
		p_boolarray2,
		p_boolarray_block,
		bUseMask
		);
	Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat 2");

	// We have a 2-regression programmed. Should we have a 3-regression programmed also?
	// Do each species in turn:
	
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
	
	cudaMemcpy(p_sum_eps_deps_by_dbeta_J_host, p_sum_eps_deps_by_dbeta_J, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_eps_deps_by_dbeta_R_host, p_sum_eps_deps_by_dbeta_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_depsbydbeta_J_times_J_host, p_sum_depsbydbeta_J_times_J, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_depsbydbeta_R_times_R_host, p_sum_depsbydbeta_R_times_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_depsbydbeta_J_times_R_host, p_sum_depsbydbeta_J_times_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);

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
		beta_nJ = 0.0;
		beta_nR = 0.0;
	} else {
		beta_nJ = -(sum_eps_deps_by_dbeta_J*sum_depsbydbeta_R_times_R - sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_R) /
			(sum_depsbydbeta_J_times_J*sum_depsbydbeta_R_times_R - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
		beta_nR = -(sum_eps_deps_by_dbeta_R*sum_depsbydbeta_J_times_J - sum_eps_deps_by_dbeta_J*sum_depsbydbeta_J_times_R) /
			(sum_depsbydbeta_R_times_R*sum_depsbydbeta_J_times_J - sum_depsbydbeta_J_times_R*sum_depsbydbeta_J_times_R);
	}

	if (bUseMask) {
		L2eps_n = sqrt(sum_eps_eps * over_iEquations_n);
	} else {
		L2eps_n = sqrt(sum_eps_eps / (real)NUMVERTICES);
	};
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
	
	cudaMemcpy(p_sum_eps_deps_by_dbeta_J_host, p_sum_eps_deps_by_dbeta_J, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_eps_deps_by_dbeta_R_host, p_sum_eps_deps_by_dbeta_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_depsbydbeta_J_times_J_host, p_sum_depsbydbeta_J_times_J, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_depsbydbeta_R_times_R_host, p_sum_depsbydbeta_R_times_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_depsbydbeta_J_times_R_host, p_sum_depsbydbeta_J_times_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);

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

	if (bUseMask) {
		L2eps_i = sqrt(sum_eps_eps * over_iEquations_i);
	} else {
		L2eps_i = sqrt(sum_eps_eps / (real)NUMVERTICES);
	};
	printf("\n for ION [ Beta1 %1.10E Beta2 %1.10E L2eps(old) %1.10E ] ", beta_iJ, beta_iR, L2eps_i);
//
//
//	FILE * dbgfile = fopen("debugsolve2.txt", "w");
//	cudaMemcpy(p_temphost1, p_epsilon_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
//	cudaMemcpy(p_temphost2, p_d_eps_by_dbeta_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
//	cudaMemcpy(p_temphost3, p_d_eps_by_dbetaR_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
//	cudaMemcpy(p_boolhost, p_boolarray2 + 2 * NUMVERTICES, sizeof(bool)*NUMVERTICES, cudaMemcpyDeviceToHost);
//	for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
//	{
//		if ((p_temphost1[iVertex] != 0.0) || (p_temphost2[iVertex] != 0.0) || (p_temphost3[iVertex] != 0.0))
//		{
//			fprintf(dbgfile, "iVertex %d eps %1.14E depsbydbeta1 %1.14E depsbydbeta2 %1.14E bool %d\n",
//				iVertex, p_temphost1[iVertex], p_temphost2[iVertex], p_temphost3[iVertex], (p_boolhost[iVertex] ? 1 : 0));
//		}
//	}
//	fclose(dbgfile);
//	printf("dbgfile done\n");

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
	
	cudaMemcpy(p_sum_eps_deps_by_dbeta_J_host, p_sum_eps_deps_by_dbeta_J, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_eps_deps_by_dbeta_R_host, p_sum_eps_deps_by_dbeta_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_depsbydbeta_J_times_J_host, p_sum_depsbydbeta_J_times_J, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_depsbydbeta_R_times_R_host, p_sum_depsbydbeta_R_times_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_depsbydbeta_J_times_R_host, p_sum_depsbydbeta_J_times_R, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMajor, cudaMemcpyDeviceToHost);

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
	
		printf("sum_eps_depsJ %1.11E sum_eps_deps2 %1.11E sum_JJ %1.11E sum_RR %1.11E sum_JR %1.11E\n",
			sum_eps_deps_by_dbeta_J, sum_eps_deps_by_dbeta_R, sum_depsbydbeta_J_times_J, sum_depsbydbeta_R_times_R,
			sum_depsbydbeta_J_times_R);
	
	};

	if (bUseMask) {
		L2eps_e = sqrt(sum_eps_eps * over_iEquations_e);
	} else {
		L2eps_e = sqrt(sum_eps_eps / (f64)NUMVERTICES);
	}
	printf("\n for Electron [ Beta1 %1.14E Beta2 %1.14E L2eps(old) %1.10E ] ", beta_eJ, beta_eR, L2eps_e);

	// ======================================================================================================================

	// bringing back adding both at once WOULD be more efficient.

	cudaMemcpy(&tempf64, &(p_Te[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("Te[%d]:%1.9E\n", VERTCHOSEN, tempf64);
	VectorAddMultiple_masked << < numTilesMajorClever, threadsPerTileMajorClever >> >(
		p_Tn, beta_nJ, p_slot1n,
		p_Ti, beta_iJ, p_slot1i,
		p_Te, beta_eJ, p_slot1e,
		p_boolarray2,
		p_boolarray_block,
		bUseMask		
		);
	Call(cudaThreadSynchronize(), "cudaTS AddtoT ___1");

//	cudaMemcpy(&tempf64, &(p_slot1e[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
//	printf("regressor[%d]:%1.9E\n", VERTCHOSEN, tempf64);
	
	VectorAddMultiple_masked << < numTilesMajorClever, threadsPerTileMajorClever >> >(
		p_Tn, beta_nR, p_slot2n,
		p_Ti, beta_iR, p_slot2i,
		p_Te, beta_eR, p_slot2e,
		p_boolarray2,
		p_boolarray_block,
		bUseMask);
	Call(cudaThreadSynchronize(), "cudaTS AddtoT ___2");
	
	cudaMemcpy(&tempf64, &(p_slot2e[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("regressor[%d]:%1.9E\n", VERTCHOSEN, tempf64);
	 
	cudaMemcpy(&tempf64, &(p_Te[VERTCHOSEN]), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("Te[%d]:%1.9E\n", VERTCHOSEN, tempf64);

	// Test effect of additions:

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
			pX_use->p_AreaMajor,
			p_boolarray2,
			p_boolarray_block,
			bUseMask);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate_new");


	cudaMemset(p_epsilon_n, 0, sizeof(f64)*NUMVERTICES); // should be unnecessary calls.
	cudaMemset(p_epsilon_i, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_epsilon_e, 0, sizeof(f64)*NUMVERTICES);
	kernelCreateEpsilon_Heat_for_Jacobi << < numTilesMajorClever, threadsPerTileMajorClever >> >(
		// eps = T - (T_k +- h sum kappa dot grad T)
		hsub,
		pX_use->p_info + BEGINNING_OF_CENTRAL,
		p_Tn, p_Ti, p_Te,
		p_T_k,
		NT_addition_rates_d_temp,
		pX_use->p_n_major, // got this
		pX_use->p_AreaMajor, // got this -> N, Nn
		p_epsilon_n, p_epsilon_i, p_epsilon_e,
		p_boolarray2,
		p_boolarray_block,
		bUseMask
		);
	Call(cudaThreadSynchronize(), "cudaTS Create epsilon heat");


	kernelAccumulateSumOfSquares << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_epsilon_n, p_epsilon_i, p_epsilon_e,
			p_temp1, p_temp2, p_temp3);
	Call(cudaThreadSynchronize(), "cudaTS Sumof squares");
	f64 SS_n = 0.0, SS_i = 0.0, SS_e = 0.0;
	cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMajorClever, cudaMemcpyDeviceToHost);
	for (iTile = 0; iTile < numTilesMajorClever; iTile++)
	{
		SS_n += p_temphost1[iTile];
		SS_i += p_temphost2[iTile];
		SS_e += p_temphost3[iTile];
	}

	if (bUseMask) {
		L2eps_n = sqrt(SS_n * over_iEquations_n);
		L2eps_i = sqrt(SS_i * over_iEquations_i);
		L2eps_e = sqrt(SS_e * over_iEquations_e);
	}
	else {
		L2eps_n = sqrt(SS_n / (real)NUMVERTICES);
		L2eps_i = sqrt(SS_i / (real)NUMVERTICES);
		L2eps_e = sqrt(SS_e / (real)NUMVERTICES);
	};

	printf("L2eps_n %1.10E  L2eps_i %1.10E  L2eps_e %1.10E \n\n", L2eps_n, L2eps_i, L2eps_e);

	//dbgfile = fopen("debugsolve3.txt", "w");
	//cudaMemcpy(p_temphost1, p_epsilon_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
	//cudaMemcpy(p_temphost2, p_d_eps_by_dbeta_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
	//cudaMemcpy(p_temphost3, p_d_eps_by_dbetaR_e, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
	//cudaMemcpy(p_boolhost, p_boolarray2 + 2 * NUMVERTICES, sizeof(bool)*NUMVERTICES, cudaMemcpyDeviceToHost);
	//for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
	//{
	//	if ((p_temphost1[iVertex] != 0.0) || (p_temphost2[iVertex] != 0.0) || (p_temphost3[iVertex] != 0.0))
	//	{
	//		fprintf(dbgfile, "iVertex %d eps %1.14E depsbydbeta1 %1.14E depsbydbeta2 %1.14E bool %d\n",
	//			iVertex, p_temphost1[iVertex], p_temphost2[iVertex], p_temphost3[iVertex], (p_boolhost[iVertex] ? 1 : 0));
	//	}
	//}
	//fclose(dbgfile);
	printf("dbgfile done\n");

	kernelPackupT3 << < numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_T, p_Tn, p_Ti, p_Te);	
	Call(cudaThreadSynchronize(), "cudaTS packup");
	

	// Next stop: be careful what we are adding to what.
	// TO produce seed regressors.

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

void RunBackwardJLSForViscosity(v4 * p_vie_k, v4 * p_vie, f64 const hsub, cuSyst * pX_use,
	v4 * p_initial_regressor, bool bHistory	)
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
	
	// oooh

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
			p_epsilon_ez,
			p_bFailed
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");

		int i;
		bContinue = false;
		cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMinor, cudaMemcpyDeviceToHost);
		for (i = 0; ((i < numTilesMinor) && (p_boolhost[i] == false)); i++);
		if (i < numTilesMinor) bContinue = true;

		if ((iIteration == 0) && (bHistory)) {
			kernelSplitIntoSeedRegressors << <numTilesMinor, threadsPerTileMinor >> >
				(	p_initial_regressor,
					p_vJacobi_i,
					p_vJacobi_e,
					p_epsilon_xy // use this to create 2nd regressor somehow.
					);
			Call(cudaThreadSynchronize(), "cudaTS kernelSplitIntoSeedRegressors ");
		} else {

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
		};
		// 4. To determine deps/dbeta, ie the change in eps for an addition of delta Jacobi_heat to T,

		// eps = v - (v_k + h [viscous effect] + h [known increment rate of v])
		// deps/dbeta[index] = Jacobi[index] - h d(dT/dt)/d [increment whole field by Jacobi]
		// ####################################
		// Note this last, it's very important.

		cudaMemset(p_d_epsxy_by_d_beta_i, 0, sizeof(f64_vec2)*NMINOR);
		cudaMemset(p_d_eps_iz_by_d_beta_i, 0, sizeof(f64)*NMINOR);
		cudaMemset(p_d_eps_ez_by_d_beta_i, 0, sizeof(f64)*NMINOR);

		cudaMemset(p_d_epsxy_by_d_beta_e, 0, sizeof(f64_vec2)*NMINOR);
		cudaMemset(p_d_eps_iz_by_d_beta_e, 0, sizeof(f64)*NMINOR);
		cudaMemset(p_d_eps_ez_by_d_beta_e, 0, sizeof(f64)*NMINOR);
		
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

		cudaMemset(NT_addition_tri_d, 0, sizeof(NTrates)*NUMVERTICES * 2);
		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);

		// Create suggested change from p_vJacobi_i

		kernelSet << <numTilesMinor, threadsPerTileMinor >> >(v4temparray, p_vJacobi_i, SPECIES_ION);
		Call(cudaThreadSynchronize(), "cudaTS kernelSet");

		// This is really dumb, we should split into more regressors.
		// But that means more calls to flow evaluation.

		// xy | z

		cudaMemset(p_MAR_ion3, 0, sizeof(f64_vec3)*NMINOR);
		cudaMemset(p_MAR_elec3, 0, sizeof(f64_vec3)*NMINOR);
		kernelCreate_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> >(

			pX_use->p_info,
			v4temparray,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,

			p_temp1, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
			p_temp2, //f64 * __restrict__ p_ita_parallel_elec_minor,   // nT / nu ready to look up
			p_temp3, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
			p_temp4, // f64 * __restrict__ p_nu_elec_minor,   // nT / nu ready to look up

			pX_use->p_B,
			p_MAR_ion3, // just accumulates
			p_MAR_elec3,
			NT_addition_rates_d_temp, // not used for anything --- hopefully??!
			NT_addition_tri_d);
		Call(cudaThreadSynchronize(), "cudaTS visccontrib ~~");

		kernelCreateEpsilon_Visc << <numTilesMinor, threadsPerTileMinor >> > (
			// eps = v - (v_k +- h [viscous effect])
			hsub,
			pX_use->p_info,
			v4temparray, // Jacobi regressor
			zero_vec4,
			p_MAR_ion3, 
			p_MAR_elec3,
			pX_use->p_n_minor,
			pX_use->p_AreaMinor,
			// It affects all 4 errors.
			p_d_epsxy_by_d_beta_i,
			p_d_eps_iz_by_d_beta_i,
			p_d_eps_ez_by_d_beta_i,
			p_bFailed // is it ok to junk this?
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");


		kernelSet << <numTilesMinor, threadsPerTileMinor >> >(v4temparray, p_vJacobi_e, SPECIES_ELEC);
		Call(cudaThreadSynchronize(), "cudaTS kernelSet");


		cudaMemset(p_MAR_ion3, 0, sizeof(f64_vec3)*NMINOR);
		cudaMemset(p_MAR_elec3, 0, sizeof(f64_vec3)*NMINOR);
		kernelCreate_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> >(

			pX_use->p_info,
			v4temparray,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,

			p_temp1, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
			p_temp2, //f64 * __restrict__ p_ita_parallel_elec_minor,   // nT / nu ready to look up
			p_temp3, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
			p_temp4, // f64 * __restrict__ p_nu_elec_minor,   // nT / nu ready to look up

			pX_use->p_B,
			p_MAR_ion3,
			p_MAR_elec3,
			NT_addition_rates_d_temp,
			NT_addition_tri_d);
		Call(cudaThreadSynchronize(), "cudaTS visccontrib ~~");

		kernelCreateEpsilon_Visc << <numTilesMinor, threadsPerTileMinor >> > (
			// eps = v - (v_k +- h [viscous effect])
			hsub,
			pX_use->p_info,
			v4temparray, // Jacobi regressor
			zero_vec4,
			p_MAR_ion3,
			p_MAR_elec3,
			pX_use->p_n_minor,
			pX_use->p_AreaMinor,
			// It affects all 4 errors.
			p_d_epsxy_by_d_beta_e,
			p_d_eps_iz_by_d_beta_e,
			p_d_eps_ez_by_d_beta_e,
			p_bFailed // is it ok to junk this?
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");


		//kernelCalculate_deps_WRT_beta_Visc << < numTriTiles, threadsPerTileMinor >> >(
		//	hsub,
		//	pX_use->p_info,
		//	pX_use->p_izTri_vert,
		//	pX_use->p_szPBCtri_vert,
		//	pX_use->p_izNeigh_TriMinor,
		//	pX_use->p_szPBC_triminor,

		//	p_temp1, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
		//	p_temp2, //f64 * __restrict__ p_ita_parallel_elec_minor,   // nT / nu ready to look up
		//	p_temp3, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
		//	p_temp4, // f64 * __restrict__ p_nu_elec_minor,   // nT / nu ready to look up
		//	pX_use->p_B,

		//	pX_use->p_n_minor, // got this
		//	pX_use->p_AreaMinor, // got this -> N, Nn

		//	p_vJacobi_i, // 3-vec
		//	p_vJacobi_e, // 3-vec

		//	p_d_eps_by_d_beta_i,
		//	p_d_eps_by_d_beta_e
		//	);

		//Call(cudaThreadSynchronize(), "cudaTS kernelCalculateROCepsWRTregressorT WW");
		//

		// 5. Do JLS calcs and update T
		// But we have to do for 3 species: do separately, so scalar arrays.


		kernelAccumulateSummands3 << <numTilesMinor, threadsPerTileMinor >> > (
			pX_use->p_info,

			p_epsilon_xy,
			p_epsilon_iz,
			p_epsilon_ez,

			p_d_epsxy_by_d_beta_i,
			p_d_eps_iz_by_d_beta_i,
			p_d_eps_ez_by_d_beta_i,
			p_d_epsxy_by_d_beta_e,
			p_d_eps_iz_by_d_beta_e,
			p_d_eps_ez_by_d_beta_e,
			
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
		if (iIteration % 10 == 0) printf("\nIteration %d visc: [ beta_i %1.14E beta_e %1.14E L2eps %1.14E ] ", iIteration, beta_i, beta_e, L2eps);
		
		// maybe do the add after we calc beta_n, beta_i, beta_e.
		kernelAdd_to_v << <numTilesMinor, threadsPerTileMinor >> > (
			p_vie, beta_i, beta_e, p_vJacobi_i, p_vJacobi_e);
		Call(cudaThreadSynchronize(), "cudaTS Addtov ___");
			
		

		iIteration++;

	} while ((bContinue) && (iIteration < 800));

	if (iIteration == 800) printf("\a");



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
	GlobalSuppressSuccessVerbosity = DEFAULTSUPPRESSVERBOSITY;
}


void RunBackwardR8LSForNeutralViscosity(f64_vec3 * p_v_n_k, f64_vec3 * p_v_n, f64 const hsub,
	cuSyst * pX_use
	//f64_vec3 * p_initial_regressor
	) {

	f64 beta[REGRESSORS];
	long iTile;
	f64_vec3 L2eps, Rsquared;
	static bool bHistory = false;
	// Function manages its historic seed move internally, as p_stored_move3.

	GlobalSuppressSuccessVerbosity = true;

	// (soon add last move as first regressor)

	cudaMemcpy(p_v_n, p_v_n_k, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);

	f64 sum_eps_deps_by_dbeta, sum_depsbydbeta_sq, sum_eps_eps, depsbydbeta;
	long iMinor;
	int iIteration, i;
	f64_vec3 TSS, RSS;
	cudaMemset(zero_vec3, 0, sizeof(f64_vec3)*NMINOR);
	iIteration = 0;
	bool bContinue = true;

	// 1. Create residual epsilon for v_k
	cudaMemset(p_MAR_neut2, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(NT_addition_tri_d, 0, sizeof(NTrates)*NUMVERTICES * 2);
	cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);
	kernelCreate_neutral_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> > (
		pX_use->p_info,
		p_v_n,
		pX_use->p_izTri_vert,
		pX_use->p_szPBCtri_vert,
		pX_use->p_izNeigh_TriMinor,
		pX_use->p_szPBC_triminor,
		p_temp6, // ita
		p_temp5,
		p_MAR_neut2, // just accumulates
		NT_addition_rates_d_temp,
		NT_addition_tri_d);
	Call(cudaThreadSynchronize(), "cudaTS visccontrib neut");
	cudaMemset(p_epsilon_x, 0, sizeof(f64)*NMINOR);
	cudaMemset(p_epsilon_y, 0, sizeof(f64)*NMINOR);
	cudaMemset(p_epsilon_z, 0, sizeof(f64)*NMINOR);
	CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMinor));
	kernelCreateEpsilon_NeutralVisc << <numTilesMinor, threadsPerTileMinor >> > (
		hsub,
		pX_use->p_info,
		p_v_n,
		p_v_n_k,
		p_MAR_neut2,
		pX_use->p_n_minor,
		pX_use->p_AreaMinor,
		p_epsilon_x,
		p_epsilon_y,
		p_epsilon_z,
		0
		);
	Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");

	// Collect L2eps :
	RSS.x = 0;
	RSS.y = 0;
	RSS.z = 0;
	
	kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> >(
		p_epsilon_x, p_SS);
	Call(cudaThreadSynchronize(), "cudaTS Accumulate SS");
	cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	for (iTile = 0; iTile < numTilesMinor; iTile++)
		RSS.x += p_SS_host[iTile];

	kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> >(
		p_epsilon_y, p_SS);
	Call(cudaThreadSynchronize(), "cudaTS Accumulate SS");
	cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	for (iTile = 0; iTile < numTilesMinor; iTile++)
		RSS.y += p_SS_host[iTile];
	
	kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> >(
		p_epsilon_z, p_SS);
	Call(cudaThreadSynchronize(), "cudaTS Accumulate SS");
	cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	for (iTile = 0; iTile < numTilesMinor; iTile++)
		RSS.z += p_SS_host[iTile];
	L2eps.x = sqrt(RSS.x / (real)NMINOR);
	L2eps.y = sqrt(RSS.y / (real)NMINOR);
	L2eps.z = sqrt(RSS.z / (real)NMINOR);

	printf("L2eps %1.8E %1.8E %1.8E\n", L2eps.x, L2eps.y, L2eps.z);

	do {

		// 2. Create set of 7 or 8 regressors, starting with epsilon3 normalized,
		// and deps/dbeta for each one.
		// The 8th is usually either for the initial seed regressor (prev move) or comes from previous iteration

		CallMAC(cudaMemset(p_d_eps_by_d_beta_x_, 0, sizeof(f64)*NMINOR*REGRESSORS));
		CallMAC(cudaMemset(p_d_eps_by_d_beta_y_, 0, sizeof(f64)*NMINOR*REGRESSORS));
		CallMAC(cudaMemset(p_d_eps_by_d_beta_z_, 0, sizeof(f64)*NMINOR*REGRESSORS));
		for (i = 0; i < REGRESSORS; i++)
		{
			// purpose of loop: define regressor & take d_eps_by_d_beta.

			// populate regressor i 
			// regressor 0 = p_epsilon3, normalized, since on iteration 0 this is simply h A vk

			if (i == 0) {
				AssembleVector3 <<<numTilesMinor, threadsPerTileMinor >>>(p_regressors3, p_epsilon_x,
					p_epsilon_y, p_epsilon_z);
			} else {
				if ((i == REGRESSORS - 1) && ((iIteration > 0) || (bHistory))) {
					cudaMemcpy(p_regressors3 + i*NMINOR, p_stored_move3, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);
					// might as well used p_stored_move for both prev move and seed from previous call.
				} else {

					// Care to subtract the previous regressor to reduce colinearity? Yep.
					// d_eps was formed including (1-hA) regressor
					// so let's remove 1x regressor from it.
					SubtractVector3stuff << <numTilesMinor, threadsPerTileMinor >> >
						(p_regressors3 + i*NMINOR,
							p_d_eps_by_d_beta_x_ + (i - 1)*NMINOR,
							p_d_eps_by_d_beta_y_ + (i - 1)*NMINOR,
							p_d_eps_by_d_beta_z_ + (i - 1)*NMINOR,
							p_regressors3 + (i - 1)*NMINOR
							); // out.x = a.x-b.x
					Call(cudaThreadSynchronize(), "SubtractVector3stuff");
				};
			};

			// Normalize regressor: divide by L2 norm.
			// ___________________________________________:
			kernelAccumulateSumOfSquares3 << <numTilesMinor, threadsPerTileMinor >> >
				(	p_regressors3 + i*NMINOR,
					p_tempvec3
					);
			Call(cudaThreadSynchronize(), "SS3");
			cudaMemcpy(p_tempvec3host, p_tempvec3, sizeof(f64_vec3)*numTilesMinor, cudaMemcpyDeviceToHost);
			f64_vec3 SS(0.0, 0.0, 0.0);
			for (iTile = 0; iTile < numTilesMinor; iTile++)
			{
				SS += p_tempvec3host[iTile];
			}
			f64_vec3 L2regress;
			L2regress.x = sqrt(SS.x / (real)NMINOR);
			L2regress.y = sqrt(SS.y / (real)NMINOR);
			L2regress.z = sqrt(SS.z / (real)NMINOR);

		//	printf("got to here  -- L2regress %1.8E %1.8E %1.8E \n", L2regress.x, L2regress.y, L2regress.z);
			
			if (L2regress.x == 0.0) L2regress.x = 1.0;
			if (L2regress.y == 0.0) L2regress.y = 1.0;
			if (L2regress.z == 0.0) L2regress.z = 1.0;

			ScaleVector3 << <numTilesMinor, threadsPerTileMinor >> > (p_regressors3 + i*NMINOR, 
				1.0/L2regress.x, 1.0 / L2regress.y, 1.0 / L2regress.z);
			Call(cudaThreadSynchronize(), "ScaleVector3");

			// ============================================================
			cudaMemset(p_MAR_neut2, 0, sizeof(f64_vec3)*NMINOR);
			kernelCreate_neutral_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> > (
				pX_use->p_info,
				p_regressors3 + i*NMINOR,
				pX_use->p_izTri_vert,
				pX_use->p_szPBCtri_vert,
				pX_use->p_izNeigh_TriMinor,
				pX_use->p_szPBC_triminor,
				p_temp6, // ita
				p_temp5,
				p_MAR_neut2, // just accumulates
				NT_addition_rates_d_temp,
				NT_addition_tri_d);
			Call(cudaThreadSynchronize(), "cudaTS visccontrib neut");
			kernelCreateEpsilon_NeutralVisc << <numTilesMinor, threadsPerTileMinor >> > (
				hsub,
				pX_use->p_info,
				p_regressors3 + i*NMINOR,
				zero_vec3,
				p_MAR_neut2,
				pX_use->p_n_minor,
				pX_use->p_AreaMinor,
				p_d_eps_by_d_beta_x_ + i*NMINOR, 
				p_d_eps_by_d_beta_y_ + i*NMINOR, 
				p_d_eps_by_d_beta_z_ + i*NMINOR,  // This is assigning values as if epsilon can be changed in ins.
				0
				);
			Call(cudaThreadSynchronize(), "cudaTS Create epsilon neut visc");


		};

		// Now on to determine coefficients.
		// We can solve for 3 separate sets of coefficients since each dimension was independent.
		// Then we add different amounts for x,y,z.
		// Then we test the overall move to see if the dot product with the consequent forward direction is positive.
		// Though given that dimensions are independent, there is certainly a case to treat the criterion independently.
		// ==============================================================================================================
		f64 * p_epsilon_, * p_deps_by_dbeta;
		int iDim;

		cudaMemset(p_stored_move3, 0, sizeof(f64_vec3)*NMINOR);
		for (iDim = 0; iDim < 3; iDim++)
		{
			if (iDim == 0)			{
				p_epsilon_ = p_epsilon_x;
				p_deps_by_dbeta = p_d_eps_by_d_beta_x_;
			};
			if (iDim == 1) {
				p_epsilon_ = p_epsilon_y;
				p_deps_by_dbeta = p_d_eps_by_d_beta_y_;
			};
			if (iDim == 2) {
				p_epsilon_ = p_epsilon_z;
				p_deps_by_dbeta = p_d_eps_by_d_beta_z_;
			};

			// Neue plan:
			// when we get here we have deps/dbeta for each dimension as separate arrays.
			// Only regressors take the form f64_vec3.
			
			cudaMemset(p_eps_against_deps, 0, sizeof(f64)*REGRESSORS * numTilesMinor);
			cudaMemset(p_sum_product_matrix, 0, sizeof(f64) * REGRESSORS*REGRESSORS* numTilesMinor);
			// DIMENSION!!

			// We are going to want 8 beta for each dimension.

			// It's too much to store in shared memory so split product matrix sum into 3 calls:
			
			// MUST ENSURE THAT things are 0 away from domain cells!!
			kernelAccumulateSummandsNeutVisc2 << <numTilesMinor, threadsPerTileMinor/4 >> > (
				p_epsilon_,
				p_deps_by_dbeta,      // data for this dimension, 8 regressors
				p_eps_against_d_eps,  // 1x8 for each tile
				p_sum_product_matrix // this is 8x8 for each tile, for each dimension
				);
			Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands neutvisc2");
			
			cudaMemcpy(p_eps_against_d_eps_host, p_eps_against_d_eps, sizeof(f64) * 8 * numTilesMinor, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_sum_product_matrix_host, p_sum_product_matrix, sizeof(f64) * 64 * numTilesMinor, cudaMemcpyDeviceToHost);

			// DIMENSION!!
			f64 eps_deps[REGRESSORS];
			f64 sum_product_matrix[REGRESSORS*REGRESSORS];
			memset(eps_deps, 0, sizeof(f64) * REGRESSORS);
			memset(sum_product_matrix, 0, sizeof(f64) * REGRESSORS*REGRESSORS);
			for (iTile = 0; iTile < numTilesMinor; iTile++)
			{
				for (i = 0; i < REGRESSORS; i++)
					eps_deps[i] -= p_eps_against_d_eps_host[iTile *REGRESSORS + i];

				// Note minus, to get beta already negated.

				for (i = 0; i < REGRESSORS*REGRESSORS; i++)
					sum_product_matrix[i] += p_sum_product_matrix_host[iTile *REGRESSORS*REGRESSORS + i];
			};

			// Note that file 1041-Krylov.pdf claims that simple factorization for LS is an
			// unstable method and that is why the complications of GMRES are needed.

			// now we need the LAPACKE dgesv code to solve the 8x8 linear equation.
			f64 storeRHS[REGRESSORS];
			f64 storemat[REGRESSORS*REGRESSORS];
			memcpy(storeRHS, eps_deps, sizeof(f64)*REGRESSORS);
			memcpy(storemat, sum_product_matrix, sizeof(f64)*REGRESSORS*REGRESSORS);

			lapack_int ipiv[REGRESSORS];
			lapack_int Nrows = REGRESSORS,
				Ncols = REGRESSORS,  // lda
				Nrhscols = 1, // ldb
				Nrhsrows = REGRESSORS, info;

			//	printf("LAPACKE_dgesv Results\n");
			// Solve the equations A*X = B 
			info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, 
				Nrows, 1, sum_product_matrix, Ncols, ipiv, eps_deps, Nrhscols);
			// Check for the exact singularity :

			if (info > 0) {
				//	printf("The diagonal element of the triangular factor of A,\n");
				//	printf("U(%i,%i) is zero, so that A is singular;\n", info, info);
				printf("the solution could not be computed.\n");
				printf("press c\n");
				while (getch() != 'c');
			} else {
				if (info == 0) {
					memcpy(beta, eps_deps, REGRESSORS * sizeof(f64)); // that's where LAPACKE saves the result apparently.
				};
			}
			
			// Now beta[8] is the set of coefficients for x
			// Move to the new value: add lc of regressors to proposal vector.
			
			CallMAC(cudaMemcpyToSymbol(beta_n_c, beta, REGRESSORS * sizeof(f64)));

			printf("Iteration %d nvisc: [ beta ", iIteration);
			for (i = 0; i < REGRESSORS; i++) printf("%1.3E ", beta[i]);
			printf(" ]\n");

			AddLCtoVector3component << <numTilesMinor, threadsPerTileMinor >> > (p_v_n, p_regressors3, iDim,
				p_stored_move3);
			Call(cudaThreadSynchronize(), "cudaTS AddLCtoVector3component");
		}; // next iDim
		
		// Finally, test whether the new values satisfy 'reasonable' criteria:
	
		cudaMemset(p_MAR_neut2, 0, sizeof(f64_vec3)*NMINOR);
		kernelCreate_neutral_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> > (
			pX_use->p_info,
			p_v_n,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,
			p_temp6, // ita
			p_temp5,
			p_MAR_neut2, // just accumulates
			NT_addition_rates_d_temp,
			NT_addition_tri_d);
		Call(cudaThreadSynchronize(), "cudaTS visccontrib neut");
		cudaMemset(p_epsilon3, 0, sizeof(f64_vec3)*NMINOR);
		CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMinor));
		kernelCreateEpsilon_NeutralVisc << <numTilesMinor, threadsPerTileMinor >> > (
			// eps = v - (v_k +- h [viscous effect])
			hsub,
			pX_use->p_info,
			p_v_n,
			p_v_n_k,
			p_MAR_neut2,
			pX_use->p_n_minor,
			pX_use->p_AreaMinor,
			p_epsilon_x, // array of 3-vectors  -- overwrite
			p_epsilon_y, 
			p_epsilon_z,
			p_bFailed 
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");

		bContinue = false;
		cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMinor, cudaMemcpyDeviceToHost);
		for (i = 0; ((i < numTilesMinor) && (p_boolhost[i] == false)); i++);
		if (i < numTilesMinor) bContinue = true;
		// primitive mechanism

		// Collect L2eps and R^2:
		TSS = RSS;
		RSS.x = 0;
		RSS.y = 0;
		RSS.z = 0;
		kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> >(
			p_epsilon_x, p_SS);
		Call(cudaThreadSynchronize(), "cudaTS Accumulate SS");
		cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMinor; iTile++)
			RSS.x += p_SS_host[iTile];		
		kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> >(
			p_epsilon_y, p_SS);
		Call(cudaThreadSynchronize(), "cudaTS Accumulate SS");
		cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMinor; iTile++)
			RSS.y += p_SS_host[iTile];
		kernelAccumulateSumOfSquares1 << <numTilesMinor, threadsPerTileMinor >> >(
			p_epsilon_z, p_SS);
		Call(cudaThreadSynchronize(), "cudaTS Accumulate SS");
		cudaMemcpy(p_SS_host, p_SS, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMinor; iTile++)
			RSS.z += p_SS_host[iTile];
		L2eps.x = sqrt(RSS.x / (real)NMINOR);
		L2eps.y = sqrt(RSS.y / (real)NMINOR);
		L2eps.z = sqrt(RSS.z / (real)NMINOR);

		// What is R^2? We should like to report it.
		Rsquared.x = (TSS.x - RSS.x) / TSS.x;
		Rsquared.y = (TSS.y - RSS.y) / TSS.y;
		Rsquared.z = (TSS.z - RSS.z) / TSS.z;
		
		Rsquared.x *= 100.0; Rsquared.y *= 100.0; Rsquared.z *= 100.0;
		printf("L2eps xyz %1.8E %1.8E %1.8E R^2 %2.3f%% %2.3f%% %2.3f%% bCont: %d\n",
			L2eps.x, L2eps.y, L2eps.z, Rsquared.x, Rsquared.y, Rsquared.z, (bContinue?1:0));
		
		// Just to be clear, what do we mean by 'move direction', we're clear move up to here is proposal-Tk
		// hA is the direction from x_k+1 .. so why are we not just asking for epsilon > 0
		// There is more to this than meets the eye.
		
		// We would also like to do this above. Is there a cleverer way to rearrange code?
						
		// 2. A reasonable criterion for proximity to a sensible value. It only has to be within say 1e-4 relative.

		// Ideally would split into 3 loops for xyz but it's only neutral viscosity.
		// For the others the pattern has to be that we take eps.deps as a dot product over all eps including all 3 dimensions
		// So actually simpler.
		
		iIteration++;
	} while (bContinue);
		
	GlobalSuppressSuccessVerbosity = DEFAULTSUPPRESSVERBOSITY;

	// Save move for next time:
	bHistory = true;
	SubtractVector3 << <numTilesMinor, threadsPerTileMinor >> >(p_stored_move3, p_v_n, p_v_n_k);
	Call(cudaThreadSynchronize(), "cudaTS subtract vector 3");

}

/*
void RunBackwardJLSForNeutralViscosity(f64_vec3 * p_v_n_k, f64_vec3 * p_v_n, f64 const hsub,
	cuSyst * pX_use,
	f64_vec3 * p_initial_regressor, bool bHistory) {

	f64_vec3 beta;
	long iTile;
	
	GlobalSuppressSuccessVerbosity = true;

	// (soon add last move as first regressor)

	cudaMemcpy(p_v_n, p_v_n_k, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);

	f64 sum_eps_deps_by_dbeta, sum_depsbydbeta_sq, sum_eps_eps, depsbydbeta;
	printf("\nJLS [beta L2eps]: ");
	long iMinor;
	f64 L2eps;
	int iIteration;

	kernelCalc_Matrices_for_Jacobi_NeutralViscosity << < numTriTiles, threadsPerTileMinor >> >//SelfCoefficient
		(
			hsub,
			pX_use->p_info, // minor
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,

			p_temp6, // ita
			p_temp5,
			pX_use->p_n_minor, // eps += -h/N * MAR; thus N features in self coefficient
			pX_use->p_AreaMinor,
			p_InvertedMatrix_n // Actually just scalings.
			); // don't forget +1 in self coefficient
	Call(cudaThreadSynchronize(), "cudaTS kernelCalc_Jacobi_for_Viscosity");

	iIteration = 0;
	bool bContinue = true;
	do
	{
		cudaMemset(p_MAR_neut2, 0, sizeof(f64_vec3)*NMINOR);
		cudaMemset(NT_addition_tri_d, 0, sizeof(NTrates)*NUMVERTICES * 2);
		cudaMemset(NT_addition_rates_d_temp, 0, sizeof(NTrates)*NUMVERTICES);

		kernelCreate_neutral_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> > (

			pX_use->p_info,
			p_v_n,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,

			p_temp6, // ita
			p_temp5,

			p_MAR_neut2, // just accumulates
			NT_addition_rates_d_temp,
			NT_addition_tri_d);
		Call(cudaThreadSynchronize(), "cudaTS visccontrib neut");

		cudaMemset(p_epsilon3, 0, sizeof(f64_vec3)*NMINOR);
		// Given putative ROC and coeff on self, it is simple to calculate eps and Jacobi. So do so:
		CallMAC(cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMinor));
		kernelCreateEpsilon_NeutralVisc << <numTilesMinor, threadsPerTileMinor >> > (
			// eps = v - (v_k +- h [viscous effect])
			// x = -eps/coeffself
			hsub,
			pX_use->p_info,
			p_v_n,
			p_v_n_k,
			p_MAR_neut2,
			 
			pX_use->p_n_minor,
			pX_use->p_AreaMinor,

			p_epsilon3, // array of 3-vector
			p_bFailed
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");

		int i;
		bContinue = false;
		cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMinor, cudaMemcpyDeviceToHost);
		for (i = 0; ((i < numTilesMinor) && (p_boolhost[i] == false)); i++);
		if (i < numTilesMinor) bContinue = true;
		// primitive mechanism

		kernelMultiply_Get_Jacobi_NeutralVisc << < numTilesMinor, threadsPerTileMinor >> >//SelfCoefficient
			(
				pX_use->p_info,
				p_epsilon3, // input
				p_InvertedMatrix_n,
				// output:
				p_vJacobi_n // 3-vec array
				);
		Call(cudaThreadSynchronize(), "cudaTS kernelGetJacobi");
		
		cudaMemset(p_d_epsilon_by_d_beta_x, 0, sizeof(f64_vec3)*NMINOR);
		cudaMemset(p_d_epsilon_by_d_beta_y, 0, sizeof(f64_vec3)*NMINOR);
		cudaMemset(p_d_epsilon_by_d_beta_z, 0, sizeof(f64_vec3)*NMINOR);
		kernelSetx << <numTilesMinor, threadsPerTileMinor >> >
			(v3temp, p_vJacobi_n);
		Call(cudaThreadSynchronize(), "cudaTS kernelSetx");

		cudaMemset(p_MAR_neut3, 0, sizeof(f64_vec3)*NMINOR);
		kernelCreate_neutral_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> > (

			pX_use->p_info,
			v3temp,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,

			p_temp6, // ita
			p_temp5,

			p_MAR_neut3, // just accumulates
			NT_addition_rates_d_temp,
			NT_addition_tri_d);
		Call(cudaThreadSynchronize(), "cudaTS visccontrib neut J");

		cudaMemset(zero_vec3, 0, sizeof(f64_vec3)*NMINOR);
		kernelCreateEpsilon_NeutralVisc << <numTilesMinor, threadsPerTileMinor >> > (
			// eps = v - (v_k +- h [viscous effect])
			hsub,
			pX_use->p_info,
			v3temp, // Jacobi regressor
			zero_vec3,
			p_MAR_neut3,
			pX_use->p_n_minor,
			pX_use->p_AreaMinor,
			p_d_epsilon_by_d_beta_x, // 3-vector
			0
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");


		kernelSety << <numTilesMinor, threadsPerTileMinor >> >(v3temp, p_vJacobi_n);
		Call(cudaThreadSynchronize(), "cudaTS kernelSety");

		cudaMemset(p_MAR_neut3, 0, sizeof(f64_vec3)*NMINOR);
		kernelCreate_neutral_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> > (

			pX_use->p_info,
			v3temp,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,

			p_temp6, // ita
			p_temp5,

			p_MAR_neut3, // just accumulates
			NT_addition_rates_d_temp,
			NT_addition_tri_d);
		Call(cudaThreadSynchronize(), "cudaTS visccontrib neut J");

		kernelCreateEpsilon_NeutralVisc << <numTilesMinor, threadsPerTileMinor >> > (
			// eps = v - (v_k +- h [viscous effect])
			hsub,
			pX_use->p_info,
			v3temp, // Jacobi regressor
			zero_vec3,
			p_MAR_neut3,
			pX_use->p_n_minor,
			pX_use->p_AreaMinor,
			p_d_epsilon_by_d_beta_y,
			0
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");


		kernelSetz << <numTilesMinor, threadsPerTileMinor >> >(v3temp, p_vJacobi_n);
		Call(cudaThreadSynchronize(), "cudaTS kernelSetx");

		cudaMemset(p_MAR_neut3, 0, sizeof(f64_vec3)*NMINOR);
		kernelCreate_neutral_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> > (

			pX_use->p_info,
			v3temp,
			pX_use->p_izTri_vert,
			pX_use->p_szPBCtri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBC_triminor,

			p_temp6, // ita
			p_temp5,

			p_MAR_neut3, // just accumulates
			NT_addition_rates_d_temp,
			NT_addition_tri_d);
		Call(cudaThreadSynchronize(), "cudaTS visccontrib neut J");

		kernelCreateEpsilon_NeutralVisc << <numTilesMinor, threadsPerTileMinor >> > (
			// eps = v - (v_k +- h [viscous effect])
			hsub,
			pX_use->p_info,
			v3temp, // Jacobi regressor
			zero_vec3,
			p_MAR_neut3,
			pX_use->p_n_minor,
			pX_use->p_AreaMinor,
			p_d_epsilon_by_d_beta_z,
			0
			);
		Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");


		if ((iIteration == 0) && (bHistory))
		{
			kernelSetx << <numTilesMinor, threadsPerTileMinor >> >
				(v3temp, p_initial_regressor);
			Call(cudaThreadSynchronize(), "cudaTS kernelSetx");			
			
			cudaMemset(p_MAR_neut3, 0, sizeof(f64_vec3)*NMINOR);		
			kernelCreate_neutral_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> > (

				pX_use->p_info,
				v3temp,
				pX_use->p_izTri_vert,
				pX_use->p_szPBCtri_vert,
				pX_use->p_izNeigh_TriMinor,
				pX_use->p_szPBC_triminor,

				p_temp6, // ita
				p_temp5,

				p_MAR_neut3, // just accumulates
				NT_addition_rates_d_temp,
				NT_addition_tri_d);
			Call(cudaThreadSynchronize(), "cudaTS visccontrib neut J");

			cudaMemset(zero_vec3, 0, sizeof(f64_vec3)*NMINOR);
			kernelCreateEpsilon_NeutralVisc << <numTilesMinor, threadsPerTileMinor >> > (
				// eps = v - (v_k +- h [viscous effect])
				hsub,
				pX_use->p_info,
				v3temp, // Jacobi regressor
				zero_vec3,
				p_MAR_neut3,
				pX_use->p_n_minor,
				pX_use->p_AreaMinor,
				p_d_epsilon_by_d_beta_x2, // 3-vector
				0
				);
			Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");


			kernelSety << <numTilesMinor, threadsPerTileMinor >> >(v3temp, p_initial_regressor);
			Call(cudaThreadSynchronize(), "cudaTS kernelSety");

			cudaMemset(p_MAR_neut3, 0, sizeof(f64_vec3)*NMINOR);
			kernelCreate_neutral_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> > (

				pX_use->p_info,
				v3temp,
				pX_use->p_izTri_vert,
				pX_use->p_szPBCtri_vert,
				pX_use->p_izNeigh_TriMinor,
				pX_use->p_szPBC_triminor,

				p_temp6, // ita
				p_temp5,

				p_MAR_neut3, // just accumulates
				NT_addition_rates_d_temp,
				NT_addition_tri_d);
			Call(cudaThreadSynchronize(), "cudaTS visccontrib neut J");

			kernelCreateEpsilon_NeutralVisc << <numTilesMinor, threadsPerTileMinor >> > (
				// eps = v - (v_k +- h [viscous effect])
				hsub,
				pX_use->p_info,
				v3temp, // Jacobi regressor
				zero_vec3,
				p_MAR_neut3,
				pX_use->p_n_minor,
				pX_use->p_AreaMinor,
				p_d_epsilon_by_d_beta_y2,
				0
				);
			Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");


			kernelSetz << <numTilesMinor, threadsPerTileMinor >> >(v3temp, p_initial_regressor);
			Call(cudaThreadSynchronize(), "cudaTS kernelSetx");

			cudaMemset(p_MAR_neut3, 0, sizeof(f64_vec3)*NMINOR);
			kernelCreate_neutral_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> > (

				pX_use->p_info,
				v3temp,
				pX_use->p_izTri_vert,
				pX_use->p_szPBCtri_vert,
				pX_use->p_izNeigh_TriMinor,
				pX_use->p_szPBC_triminor,

				p_temp6, // ita
				p_temp5,

				p_MAR_neut3, // just accumulates
				NT_addition_rates_d_temp,
				NT_addition_tri_d);
			Call(cudaThreadSynchronize(), "cudaTS visccontrib neut J");

			kernelCreateEpsilon_NeutralVisc << <numTilesMinor, threadsPerTileMinor >> > (
				// eps = v - (v_k +- h [viscous effect])
				hsub,
				pX_use->p_info,
				v3temp, // Jacobi regressor
				zero_vec3,
				p_MAR_neut3,
				pX_use->p_n_minor,
				pX_use->p_AreaMinor,
				p_d_epsilon_by_d_beta_z2,
				0
				);
			Call(cudaThreadSynchronize(), "cudaTS Create epsilon visc");


		} else {
			if (iIteration > 0) {

				// take a copy of the lc of the existing Ax --- 
				// do this before we go and create Ax



			} else {
				// go with a simple one before there is any history: just use epsilon.




			}
		};





		cudaMemset(p_eps_against_deps, 0, sizeof(f64_vec3) * numTilesMinor);
		cudaMemset(p_sum_product_matrix, 0, sizeof(Symmetric3) * numTilesMinor);
		// jacobian elements: 0 1 2   deps_x / d xyz
		//                    3 4 5 
		//                    6 7 8
		kernelAccumulateSummandsNeutVisc << <numTilesMinor, threadsPerTileMinor >> > (
			pX_use->p_info,

			p_epsilon3,

			p_d_epsilon_by_d_beta_x,
			p_d_epsilon_by_d_beta_y,
			p_d_epsilon_by_d_beta_z,

			// 3+9+1 outputs:
			p_eps_against_deps,
			p_sum_product_matrix,
			p_sum_eps_eps);

		Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands 1aa");

		cudaMemcpy(p_eps_against_deps_host, p_eps_against_deps, sizeof(f64)*3*numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_product_matrix_host, p_sum_product_matrix, sizeof(Symmetric3) * numTilesMinor, cudaMemcpyDeviceToHost);
		cudaMemcpy(p_sum_eps_eps_host, p_sum_eps_eps, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);

		f64_tens3 Y;
		memset(&Y, 0, sizeof(f64_tens3));
		f64_vec3 vec3;
		memset(&vec3, 0, sizeof(f64_vec3));
		f64 sum_eps_eps = 0.0;
		for (iTile = 0; iTile < numTilesMinor; iTile++)
		{
			vec3.x += p_eps_against_deps_host[iTile].x;
			vec3.y += p_eps_against_deps_host[iTile].y;
			vec3.z += p_eps_against_deps_host[iTile].z;
			Y.xx += p_sum_product_matrix_host[iTile].xx;
			Y.xy += p_sum_product_matrix_host[iTile].xy;
			Y.xz += p_sum_product_matrix_host[iTile].xz;
			Y.yy += p_sum_product_matrix_host[iTile].yy;
			Y.yz += p_sum_product_matrix_host[iTile].yz;
			Y.zz += p_sum_product_matrix_host[iTile].zz;
			sum_eps_eps += p_sum_eps_eps_host[iTile];
		};
		Y.yx = Y.xy;
		Y.zx = Y.xz;
		Y.zy = Y.yz;

	//	printf("got to here II\n");
	//	printf("Y %1.8E %1.8E %1.8E , %1.8E %1.8E %1.8E , %1.8E %1.8E %1.8E\n",
	//		Y.xx, Y.xy, Y.xz, Y.yx, Y.yy, Y.yz, Y.zx, Y.zy, Y.zz);

		// f64_tens3 inv;
		// Y.Inverse(inv);
		// f64_vec3 beta = inv*vec3; // ?! does it?! // check pX_use...
		// This is all unnecessary as all the dimensions are independent.

		if (Y.xx != 0.0) { beta.x = vec3.x / Y.xx; }
		else {
			beta.x = 0.0;
		};
		if (Y.yy != 0.0) { beta.y = vec3.y / Y.yy; }
		else {
			beta.y = 0.0;
		};
		if (Y.zz != 0.0) { beta.z = vec3.z / Y.zz; }
		else {
			beta.z = 0.0;
		};
		
		beta.x = -beta.x; beta.y = -beta.y; beta.z = -beta.z;

		L2eps = sqrt(sum_eps_eps / (real)NMINOR);

		if (iIteration % 10== 0) printf("Iteration %d neutvisc: [ beta %1.9E %1.9E %1.9E L2eps %1.10E ] \n", iIteration, beta.x, beta.y, beta.z, L2eps);

		// maybe do the add after we calc beta_n, beta_i, beta_e.
		kernelAddLC_vec3 << <numTilesMinor, threadsPerTileMinor >> > (
			p_v_n, beta, p_vJacobi_n);
		Call(cudaThreadSynchronize(), "cudaTS Addtovn ___");

		iIteration++;

	} while ((bContinue) && (iIteration < 1000));
	
	if (iIteration == 1000) printf("\n\n\nOH DEAR\n\n\n\a");

	GlobalSuppressSuccessVerbosity = DEFAULTSUPPRESSVERBOSITY;
}*/
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
//	static real const C_over_38e6PI = (PEAKCURRENT_STATCOULOMB*0.5 / (38.0e6*PEAKTIME))*cos(PIOVERPEAKTIME*0.5 / 19.0e6);
	
	real Iz = -PEAKCURRENT_STATCOULOMB * sin((t + ZCURRENTBASETIME) * 0.5* PIOVERPEAKTIME); // half pi / peaktime

	// Changed back to no osc.
																							
//	real factor11 = 1.1*(1.0-exp(-t*1.0e10)); // exp(2) = 13.6% at the end of the first cycle.	
//	Iz -= C_over_38e6PI*factor11*sin(PI*0.993 + 38.0e6*PI*t);

	
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
	int i;
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



	//__constant__ f64 recomb_coeffs[32][3][5];
	//f64 recomb_coeffs_host[32][3][5];
	//__constant__ f64 ionize_coeffs[32][5][5];
	//f64 ionize_coeffs_host[32][5][5];
	//__constant__ f64 ionize_temps[32][10];
	//f64 ionize_temps_host[32][10];

	FILE * fp = fopen("ionize_coeffs.txt", "rt");
	rewind(fp);
	for (int iV = 0; iV < 32; iV++)
	{
		for (int j = 0; j < 10; j++)
			fscanf(fp, " %lf", &(ionize_temps_host[iV][j]));
			// check format specifier
		for (int iWhich = 0; iWhich < 5; iWhich++) 
			fscanf(fp, " %lf %lf %lf %lf %lf", &(ionize_coeffs_host[iV][iWhich][0]),
				&(ionize_coeffs_host[iV][iWhich][1]),
				&(ionize_coeffs_host[iV][iWhich][2]),
				&(ionize_coeffs_host[iV][iWhich][3]),
				&(ionize_coeffs_host[iV][iWhich][4]));		
	};
	fclose(fp);
	fp = fopen("rec_coeffs.txt", "rt");
	rewind(fp);
	for (int iV = 0; iV < 32; iV++)
	{
		for (int iWhich = 0; iWhich < 3; iWhich++)
			fscanf(fp, " %lf %lf %lf %lf %lf", &(recomb_coeffs_host[iV][iWhich][0]),
				&(recomb_coeffs_host[iV][iWhich][1]),
				&(recomb_coeffs_host[iV][iWhich][2]),
				&(recomb_coeffs_host[iV][iWhich][3]),
				&(recomb_coeffs_host[iV][iWhich][4]));
	};
	fclose(fp);

	printf("ionize_temps[8][3] %1.14E /n", ionize_temps_host[8][3]);
	printf("ionize_coeffs[11][4][2] %1.14E /n", ionize_coeffs_host[11][4][2]);
	printf("recomb_coeffs[28][1][3] %1.14E /n", recomb_coeffs_host[28][1][3]);
	//getch(); // test what we loaded
	Call(cudaMemcpyToSymbol(ionize_temps, ionize_temps_host, 32*10 * sizeof(f64)), 
		"cudaMemcpyToSymbol(ionize_temps)");
	Call(cudaMemcpyToSymbol(ionize_coeffs, ionize_coeffs_host, 32 * 5*5 * sizeof(f64)),
		"cudaMemcpyToSymbol(ionize_coeffs)");
	Call(cudaMemcpyToSymbol(recomb_coeffs, recomb_coeffs_host, 32 * 3*5 * sizeof(f64)),
		"cudaMemcpyToSymbol(recomb_coeffs)");


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
	//f64 UNIFORM_n_temp = UNIFORM_n;
	Set_f64_constant(UNIFORM_n_d, UNIFORM_n);

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

	CallMAC(cudaMalloc((void **)&p_storeviscmove, NMINOR * sizeof(v4)));
	
	CallMAC(cudaMalloc((void **)&v4temparray, NMINOR * sizeof(v4)));
	CallMAC(cudaMalloc((void **)&zero_vec4, NMINOR * sizeof(v4)));
	CallMAC(cudaMalloc((void **)&p_MAR_ion3, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_MAR_elec3, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_MAR_neut3, NMINOR * sizeof(f64_vec3)));
	cudaMemset(zero_vec4, 0, sizeof(v4)*NMINOR);

	CallMAC(cudaMalloc((void **)&p_Jacobian_list, NMINOR * SQUASH_POINTS * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_indicator, NMINOR * sizeof(long)));
	CallMAC(cudaMalloc((void **)&p_AAdot_target, NMINOR * sizeof(AAdot)));
	CallMAC(cudaMalloc((void **)&p_AAdot_start, NMINOR * sizeof(AAdot)));
	CallMAC(cudaMalloc((void **)&p_v_n_target, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_v_n_start, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_vie_target, NMINOR * sizeof(v4)));
	CallMAC(cudaMalloc((void **)&p_vie_start, NMINOR * sizeof(v4)));

	CallMAC(cudaMalloc((void **)&p_Residuals, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_pressureflag, NUMVERTICES * sizeof(bool)));
	
	CallMAC(cudaMalloc((void **)&p_eps_against_deps, NMINOR * sizeof(f64_vec3)*REGRESSORS*3));
	CallMAC(cudaMalloc((void **)&p_eps_against_d_eps, NMINOR * sizeof(f64)*REGRESSORS * 3));
	
	CallMAC(cudaMalloc((void **)&p_sum_product_matrix, numTilesMinor * sizeof(f64)*REGRESSORS*REGRESSORS*3));

	CallMAC(cudaMalloc((void **)&p_MAR_ion_pressure_major_stored, NUMVERTICES * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_MAR_elec_pressure_major_stored, NUMVERTICES * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_MAR_ion_visc_major_stored, NUMVERTICES * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_MAR_elec_visc_major_stored, NUMVERTICES * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_MAR_elec_ionization_major_stored, NUMVERTICES * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_vie_k_stored, NUMVERTICES * sizeof(v4)));


	CallMAC(cudaMalloc((void **)&p_d_epsilon_by_d_beta_x, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_d_epsilon_by_d_beta_y, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_d_epsilon_by_d_beta_z, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_epsilon3, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&v3temp, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&zero_vec3, NMINOR * sizeof(f64_vec3)));

	CallMAC(cudaMalloc((void **)&p_place_contribs, NMINOR*6 * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&p_regressor_n, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_regressor_i, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_regressor_e, NMINOR * sizeof(f64))); // only need NUMVERTICES but we reused.



	CallMAC(cudaMalloc((void **)&p_store_NTFlux, NUMVERTICES * MAXNEIGH * sizeof(NTrates)));
	CallMAC(cudaMalloc((void **)&sz_who_vert_vert, NUMVERTICES * MAXNEIGH * sizeof(short)));

	CallMAC(cudaMalloc((void **)&p_store_T_move1, NUMVERTICES * sizeof(T3)));
	CallMAC(cudaMalloc((void **)&p_store_T_move2, NUMVERTICES * sizeof(T3)));

	CallMAC(cudaMalloc((void **)&p_temp3_1, NUMVERTICES * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_temp3_2, NUMVERTICES * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_temp3_3, NUMVERTICES * sizeof(f64_vec3)));
	
	CallMAC(cudaMalloc((void **)&p_matrix_blocks, SQUASH_POINTS*SQUASH_POINTS * numTilesMinor*2 * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_vector_blocks, SQUASH_POINTS* numTilesMinor *2* sizeof(f64)));
	
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

	CallMAC(cudaMalloc((void **)&p_regressors, NUMVERTICES * (REGRESSORS + 1) * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_eps_deps_by_dbeta_x8, numTilesMinor * REGRESSORS * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&p_regressors3, NMINOR*REGRESSORS * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_tempvec3, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_SS, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_epsilon_x, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_epsilon_y, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_epsilon_z, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_stored_move3, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_d_beta_x_, NMINOR*REGRESSORS * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_d_beta_y_, NMINOR*REGRESSORS * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_d_beta_z_, NMINOR*REGRESSORS * sizeof(f64)));



	CallMAC(cudaMalloc((void **)&p_Effect_self_n, NUMVERTICES * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Effect_self_i, NUMVERTICES * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Effect_self_e, NUMVERTICES * sizeof(f64)));
	 
	CallMAC(cudaMalloc((void **)&p_sqrtD_inv_n, NUMVERTICES * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sqrtD_inv_i, NUMVERTICES * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sqrtD_inv_e, NUMVERTICES * sizeof(f64)));
		
	CallMAC(cudaMalloc((void **)&d_eps_by_dx_neigh_n, NUMVERTICES * MAXNEIGH * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&d_eps_by_dx_neigh_i, NUMVERTICES * MAXNEIGH * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&d_eps_by_dx_neigh_e, NUMVERTICES * MAXNEIGH * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&p_boolarray, 2*NUMVERTICES * sizeof(bool)));
	CallMAC(cudaMalloc((void **)&p_boolarray2, 3 * NUMVERTICES * sizeof(bool)));
	CallMAC(cudaMalloc((void **)&p_boolarray_block, numTilesMinor * sizeof(bool)));

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
	CallMAC(cudaMalloc((void **)&p_Ax, NUMVERTICES *REGRESSORS * sizeof(f64))); // sometimes use as NMINOR
	 
	CallMAC(cudaMalloc((void **)&p_sum_eps_deps_by_dbeta, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_depsbydbeta_sq, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_eps_eps, numTilesMinor * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&p_bFailed, numTilesMinor * sizeof(bool)));
		
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

	CallMAC(cudaMalloc((void **)&p_longtemp, NMINOR*2* sizeof(long)));
	CallMAC(cudaMalloc((void **)&p_temp1, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_temp2, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_temp3, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_temp4, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_temp5, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_temp6, NMINOR * sizeof(f64)));
	
	CallMAC(cudaMalloc((void **)&p_graphdata1, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_graphdata2, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_graphdata3, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_graphdata4, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_graphdata5, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_graphdata6, NMINOR * sizeof(f64)));

	for (i = 0; i < 9; i++)
		CallMAC(cudaMalloc((void **)&p_Tgraph[i], NUMVERTICES * sizeof(f64)));
	for (i = 0; i < 12; i++)
		CallMAC(cudaMalloc((void **)&p_accelgraph[i], NUMVERTICES * sizeof(f64)));
	for (i = 0; i < 20; i++)
		CallMAC(cudaMalloc((void **)&p_Ohmsgraph[i], NUMVERTICES * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&p_MAR_ion_temp_central, NUMVERTICES * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_MAR_elec_temp_central, NUMVERTICES * sizeof(f64_vec3)));

	CallMAC(cudaMalloc((void **)&p_bool, NMINOR * sizeof(bool)));
	CallMAC(cudaMalloc((void **)&p_denom_i, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_denom_e, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_summands, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Iz0_summands, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Iz0_initial, numTilesMinor * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&p_InvertedMatrix_n, NMINOR * sizeof(f64_tens3)));
	CallMAC(cudaMalloc((void **)&p_InvertedMatrix_i, NMINOR * sizeof(f64_tens3)));
	CallMAC(cudaMalloc((void **)&p_InvertedMatrix_e, NMINOR * sizeof(f64_tens3)));
	CallMAC(cudaMalloc((void **)&p_MAR_ion2, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_MAR_elec2, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_MAR_neut2, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&NT_addition_rates_d_temp, NMINOR * sizeof(NTrates)));
	CallMAC(cudaMalloc((void **)&NT_addition_rates_d_temp2, NMINOR * sizeof(NTrates)));
	CallMAC(cudaMalloc((void **)&p_epsilon_xy, NMINOR * sizeof(f64_vec2)));
	CallMAC(cudaMalloc((void **)&p_epsilon_iz, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_epsilon_ez, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_vJacobi_n, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_vJacobi_i, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_vJacobi_e, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_d_beta_i, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_d_eps_by_d_beta_e, NMINOR * sizeof(f64_vec3)));

	CallMAC(cudaMalloc((void **)&p_d_epsxy_by_d_beta_i, NMINOR * sizeof(f64_vec2)));
	CallMAC(cudaMalloc((void **)&p_d_eps_iz_by_d_beta_i, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_d_eps_ez_by_d_beta_i, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_d_epsxy_by_d_beta_e, NMINOR * sizeof(f64_vec2)));
	CallMAC(cudaMalloc((void **)&p_d_eps_iz_by_d_beta_e, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_d_eps_ez_by_d_beta_e, NMINOR * sizeof(f64)));

	CallMAC(cudaMalloc((void **)&store_heatcond_NTrates, NUMVERTICES * sizeof(NTrates)));


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
	CallMAC(cudaMalloc((void **)&p_sum_depsbydbeta_8x8, numTilesMinor * sizeof(f64) *  REGRESSORS*REGRESSORS));
	
	p_matrix_blocks_host = (f64 *)malloc(SQUASH_POINTS*SQUASH_POINTS * numTilesMinor * 2 * sizeof(f64));
	p_vector_blocks_host = (f64 *)malloc(SQUASH_POINTS* numTilesMinor * 2 * sizeof(f64));

	p_tempvec3host = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));
	p_SS_host = (f64 *)malloc(NMINOR * sizeof(f64));
	p_sum_product_matrix_host = (f64 *)malloc(numTilesMinor * sizeof(f64)*REGRESSORS*REGRESSORS*3);
	p_eps_against_deps_host = (f64_vec3 *)malloc(numTilesMinor * sizeof(f64_vec3)*REGRESSORS*3); //*3 is gratuitous in case we re-use.
	p_eps_against_d_eps_host = (f64 *)malloc(numTilesMinor*sizeof(f64)*REGRESSORS*3); //*3 is gratuitous

	p_sum_eps_deps_by_dbeta_J_x4_host = (f64_vec4 *) malloc(numTilesMinor * sizeof(f64_vec4));
	p_sum_eps_deps_by_dbeta_R_x4_host = (f64_vec4 *)malloc(numTilesMinor * sizeof(f64_vec4));
	p_sum_depsbydbeta_8x8_host = (f64 *)malloc(numTilesMinor * REGRESSORS*REGRESSORS * sizeof(f64));

	p_sum_eps_deps_by_dbeta_x8_host = (f64 *)malloc(numTilesMinor*REGRESSORS*sizeof(f64));
	p_GradTe_host = (f64_vec2 *)malloc(NMINOR * sizeof(f64_vec2));
	p_GradAz_host = (f64_vec2 *)malloc(NMINOR * sizeof(f64_vec2));
	p_B_host = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));
	p_MAR_ion_host = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));
	p_MAR_ion_compare = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));
	p_MAR_elec_host = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));
	p_MAR_elec_compare = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));
	p_MAR_neut_host = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));
	p_MAR_neut_compare = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));

	p_longtemphost = (long *)malloc(NMINOR*2 * sizeof(long));
	p_temphost1 = (f64 *)malloc(NMINOR * sizeof(f64)); // changed for debugging
	p_temphost2 = (f64 *)malloc(NMINOR * sizeof(f64)); // changed for debugging
	p_temphost3 = (f64 *)malloc(NMINOR * sizeof(f64));
	p_temphost4 = (f64 *)malloc(NMINOR * sizeof(f64));
	p_temphost5 = (f64 *)malloc(NMINOR * sizeof(f64));
	p_temphost6 = (f64 *)malloc(NMINOR * sizeof(f64));

	p_graphdata1_host = (f64 *)malloc(NMINOR * sizeof(f64));
	p_graphdata2_host = (f64 *)malloc(NMINOR * sizeof(f64));
	p_graphdata3_host = (f64 *)malloc(NMINOR * sizeof(f64));
	p_graphdata4_host = (f64 *)malloc(NMINOR * sizeof(f64));
	p_graphdata5_host = (f64 *)malloc(NMINOR * sizeof(f64));
	p_graphdata6_host = (f64 *)malloc(NMINOR * sizeof(f64));

	for (i = 0; i < 9; i++)
		p_Tgraph_host[i] = (f64 *)malloc(NUMVERTICES * sizeof(f64));
	for (i = 0; i < 12; i++)
		p_accelgraph_host[i] = (f64 *)malloc(NUMVERTICES * sizeof(f64)); // 3.6 MB
	for (i = 0; i < 20; i++)
		p_Ohmsgraph_host[i] = (f64 *)malloc(NUMVERTICES * sizeof(f64)); 

	p_boolhost = (bool *)malloc(NMINOR * sizeof(bool));
	p_sum_vec_host = (f64_vec3 *)malloc(numTilesMinor * sizeof(f64_vec3));
	
	if (p_temphost6 == 0) { printf("p6 == 0"); }
	else { printf("p6 != 0"); };
	temp_array_host = (f64 *)malloc(NMINOR * sizeof(f64));

	p_NTrates_host = (NTrates *)malloc(NMINOR * sizeof(NTrates));

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
	cudaFuncSetCacheConfig(kernelComputeJacobianValues,
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

	GlobalSuppressSuccessVerbosity = DEFAULTSUPPRESSVERBOSITY;
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
	
	bViscousHistory = false; // it might sometimes be true but hey

	cudaEvent_t start1, stop1;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, 0);
	cudaEventSynchronize(start1);

	kernelSetPressureFlag << <numTilesMajor, threadsPerTileMajor >> > (
		pX1->p_info,
		pX1->p_izTri_vert,
		p_pressureflag
		); // 0 for those that are next to CROSSING_CATH
	Call(cudaThreadSynchronize(), "cudaTS kernelSetPressureFlag");

	kernelCreateWhoAmI_verts << <numTilesMajor, threadsPerTileMajor >> > (
		pX1->p_info + BEGINNING_OF_CENTRAL,
		pX1->p_izNeigh_vert,
		sz_who_vert_vert // array of MAXNEIGH shorts for each vertex.
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelCreateWhoAmI_verts");
	
	long iSubstep;
	int i;
	// Ultimately this 10 steps .. so 1e-11? .. can be 1 advective step.

	// B is set for pX_half and pX1. So take pX_half value and spit it to host.
	for (i = 0; i < 12; i++)
		cudaMemset(p_accelgraph[i], 0, sizeof(f64)*NUMVERTICES);

	pX_half->CopyStructuralDetailsFrom(*pX1);
	pX2->CopyStructuralDetailsFrom(*pX1);

	fp_dbg = fopen("dbg1.txt", "a");
	fp_trajectory = fopen("traj.txt", "a");
	
	for (int iRepeat = 0; iRepeat < ADVECT_STEPS_PER_GPU_VISIT; iRepeat++) 
	{	
		printf("Advection step:\n"); 
		bGlobalSaveTGraphs == true;
		pX1->PerformCUDA_AdvectionCompressionInstantaneous(TIMESTEP*(real)ADVECT_FREQUENCY, pX2, pX_half);

		// We need to smoosh the izTri etc data on to the pXhalf and new pX dest systems
		// as it doesn't update any other way and this data will be valid until renewed
		// in the dest system at the end of the step -- riiiiiiiight?

		pXtemp = pX1;
		pX1 = pX2;
		pX2 = pXtemp;
		 
		pX_half->CopyStructuralDetailsFrom(*pX1);
		pX2->CopyStructuralDetailsFrom(*pX1);	
		cudaMemcpy(pX2->p_AreaMinor, pX1->p_AreaMinor, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
		cudaMemcpy(pX_half->p_AreaMinor, pX1->p_AreaMinor, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
				
		bGlobalSaveTGraphs = false;
		for (iSubstep = 0; iSubstep < ADVECT_FREQUENCY; iSubstep++)
		{
			printf("\nSTEP %d\n-------------\n", iSubstep);
			printf("evaltime = %1.10E \n\n", evaltime);
			if (iSubstep == ADVECT_FREQUENCY-1) bGlobalSaveTGraphs = true;
			pX1->PerformCUDA_Advance_noadvect(pX2, pX_half);

			pXtemp = pX1;
			pX1 = pX2;
			pX2 = pXtemp;
			
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

	printf("Graphing data passed: %d : Lap %1.9E ; %d : Lap %1.9E \n",
		VERTCHOSEN, temp_array_host[VERTCHOSEN + BEGINNING_OF_CENTRAL], VERTCHOSEN2, temp_array_host[VERTCHOSEN2 + BEGINNING_OF_CENTRAL]);
	

	cudaMemcpy(p_OhmsCoeffs_host, p_OhmsCoeffs, sizeof(OhmsCoeffs)*NMINOR, cudaMemcpyDeviceToHost);
	
	pX1->SendToHost(*pX_host);
	
	// Now store for 1D graphs: temphost3 = n, temphost4 = vr, temphost5 = vez

	// This is where we think to fill in temphost1 = nu_ei_effective + nu_en_MT
	// temphost2 = nu_en_MT / temphost1.

	kernelPrepareNuGraphs << <numTilesMinor, threadsPerTileMinor >> > (
		pX1->p_info,
		pX1->p_n_minor,
		pX1->p_T_minor,
		p_temp1,
		p_temp2
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelPrepareNuGraphs");
	cudaMemcpy(p_temphost1 , p_temp1, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_temphost2 , p_temp2, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);



	// Get some graph data:

	cudaMemset(p_temp3_1, 0, sizeof(f64_vec3)*numTilesMajor*threadsPerTileMajor);
	cudaMemset(p_temp3_2, 0, sizeof(f64_vec3)*numTilesMajor*threadsPerTileMajor);
	cudaMemset(p_temp3_3, 0, sizeof(f64_vec3)*numTilesMajor*threadsPerTileMajor);
	cudaMemset(NT_addition_rates_d, 0, sizeof(NTrates)*NUMVERTICES);
	kernelIonisationRates << <numTilesMajor, threadsPerTileMajor >> >(
		TIMESTEP,
		pX1->p_info,
		pX1->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices
		pX1->p_n_major,		
		pX1->p_AreaMajor,

		NT_addition_rates_d,
		p_temp3_1,//p_MAR_temp_major_n,
		p_temp3_2,//p_MAR_temp_major_i,
		p_temp3_3,//p_MAR_temp_major_e,

		pX1->p_vie + BEGINNING_OF_CENTRAL,
		pX1->p_v_n + BEGINNING_OF_CENTRAL,

		pX2->p_T_minor + BEGINNING_OF_CENTRAL,
		false
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelIonisationRates");

	kernelPrepareIonizationGraphs << <numTilesMajor, threadsPerTileMajor >> >(
		pX1->p_info + BEGINNING_OF_CENTRAL,
		pX1->p_n_major,
		pX1->p_AreaMajor,
		NT_addition_rates_d, // dN/dt, dNeTe/dt
		p_temp3_3, // --> d/dt v_e

		p_graphdata1, p_graphdata2, p_graphdata3, p_graphdata4, p_graphdata5, p_graphdata6
	);
	Call(cudaThreadSynchronize(), "cudaTS kernelPrepareIonizationGraphs");
	cudaMemcpy(p_graphdata1_host, p_graphdata1, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_graphdata2_host, p_graphdata2, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_graphdata3_host, p_graphdata3, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_graphdata4_host, p_graphdata4, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_graphdata5_host, p_graphdata5, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_graphdata6_host, p_graphdata6, sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
	
	kernelCollectOhmsGraphs << <numTilesMajor, threadsPerTileMajor >> > (
		pX1->p_info + BEGINNING_OF_CENTRAL,
		
		p_MAR_ion_pressure_major_stored,
		p_MAR_ion_visc_major_stored,
		p_MAR_elec_pressure_major_stored,  // need to distinguish viscous from pressure part.
		p_MAR_elec_visc_major_stored,
		p_MAR_elec_ionization_major_stored,

		pX1->p_B + BEGINNING_OF_CENTRAL,

		p_vie_k_stored, // ALL MAJOR
		pX1->p_vie + BEGINNING_OF_CENTRAL, // k+1

		p_GradTe + BEGINNING_OF_CENTRAL, // stored?
		pX1->p_n_minor + BEGINNING_OF_CENTRAL,
		pX1->p_T_minor + BEGINNING_OF_CENTRAL,

		pX1->p_AAdot + BEGINNING_OF_CENTRAL,
		pX1->p_AreaMinor, // EXCEPT THIS ONE
		p_Ohmsgraph[0], // elastic effective frictional coefficient zz
		p_Ohmsgraph[1], // ionization effective frictional coefficient zz
		p_Ohmsgraph[2], // 2 is combined y pressure accel rate
		p_Ohmsgraph[3],// 3 is q/(M+m) Ez -- do we have
		p_Ohmsgraph[4], // 4 is thermal force accel
		
		p_Ohmsgraph[5], // T_zy
		p_Ohmsgraph[6], // T_zz

		p_Ohmsgraph[7], // T acting on pressure
		p_Ohmsgraph[8], // T acting on electromotive
		p_Ohmsgraph[9], // T acting on thermal force
		p_Ohmsgraph[10], // prediction vez-viz

		p_Ohmsgraph[11], // difference of prediction from vez_k
		p_Ohmsgraph[12], // progress towards eqm: need vez_k+1
		p_Ohmsgraph[13], // viscous acceleration of electrons and ions (z)
		p_Ohmsgraph[14], // Prediction of Jz
		p_Ohmsgraph[15], // sigma zy
		p_Ohmsgraph[16], // sigma zz
		p_Ohmsgraph[17], // sigma zz times electromotive 
		p_Ohmsgraph[18] // Difference of prediction from Jz predicted.
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelCollectOhmsGraphs");

	for (i = 0; i < 9; i++)
		cudaMemcpy(p_Tgraph_host[i], p_Tgraph[i], sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
	for (i = 0; i < 12; i++)
		cudaMemcpy(p_accelgraph_host[i], p_accelgraph[i], sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);
	for (i = 0; i < 20; i++)
		cudaMemcpy(p_Ohmsgraph_host[i], p_Ohmsgraph[i], sizeof(f64)*NUMVERTICES, cudaMemcpyDeviceToHost);

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
//
//	FILE * fpaccel = fopen("accel.txt", "w");
//	for (i = 0; i < NUMVERTICES; i++)
//		fprintf(fpaccel, "%d %1.10E \n", i, p_accelgraph_host[10][i]);
//	fclose(fpaccel);
//	printf("FP ACCEL PRINTED");
//	getch(); getch(); getch();
	 
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
		SendMessage(hWnd, WM_DESTROY, 0, 0);
		exit(3);
	}
	else {
		printf("\nDebugNans OK\n");
	}
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

	SetConsoleTextAttribute(hConsole, 14);

#define USE_N_MAJOR_FOR_VERTEX 
	if (!GlobalSuppressSuccessVerbosity) {
		f64 tempf64, tempb;
		cudaMemcpy(&tempf64, &(this->p_v_n[VERTCHOSEN + BEGINNING_OF_CENTRAL].y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		cudaMemcpy(&tempb, &(this->p_vie[VERTCHOSEN + BEGINNING_OF_CENTRAL].vxy.y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d vn.y %1.9E vy %1.9E", VERTCHOSEN, tempf64, tempb);

		nvals n;
		cudaMemcpy(&n, &(this->p_n_major[VERTCHOSEN]), sizeof(nvals), cudaMemcpyDeviceToHost);
		printf("%d n %1.12E n_n %1.12E \n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", VERTCHOSEN, n.n, n.n_n);
	}

	//char ch;
	//cudaMemcpy(&ch, &(this->p_szPBC_triminor[92250 * 6 + 1]), sizeof(char), cudaMemcpyDeviceToHost);
	//printf("this holds char %d", ch);
	//cudaMemcpy(&ch, &(pX_half->p_szPBC_triminor[92250 * 6 + 1]), sizeof(char), cudaMemcpyDeviceToHost);
	//printf("pX_half holds char %d", ch);
	//cudaMemcpy(&ch, &(pX_target->p_szPBC_triminor[92250 * 6 + 1]), sizeof(char), cudaMemcpyDeviceToHost);
	//printf("pX_target holds char %d", ch);
	//getch();

	// Comes in with triangle centres on insulator: bug.
	// We need to set triangle positions before calling AreaMinorFluid.



	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> >(
		this->p_n_minor,
		this->p_n_major,
		this->p_T_minor,
		this->p_info,
		this->p_cc,  // Calculate circumcenters; we would like n and T there for shards.

		this->p_tri_corner_index,
		this->p_tri_periodic_corner_flags,

		false 
		); // call before CreateShardModel 
	Call(cudaThreadSynchronize(), "cudaTS average nTx");
	cudaMemcpy(this->p_n_minor + BEGINNING_OF_CENTRAL,
		this->p_n_major, sizeof(nvals)*NUMVERTICES, cudaMemcpyDeviceToDevice);

	kernelGet_AreaMinorFluid << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info,
		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_izNeigh_TriMinor,
		this->p_szPBC_triminor,
		p_pressureflag,
		this->p_AreaMinor // output
		);
	Call(cudaThreadSynchronize(), "cudaTS Get_AreaMinorFluid");

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

	kernelInferMinorDensitiesFromShardModel << <numTilesMinor, threadsPerTileMinor >> >(
		this->p_info,
		this->p_n_minor,
		p_n_shards,
		p_n_shards_n,
		this->p_tri_corner_index,
		this->p_who_am_I_to_corner,
		p_one_over_n);
	Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities");
	
	kernelCalculateOverallVelocitiesVertices << <numTilesMajor, threadsPerTileMajor >> > (
		this->p_info,
		this->p_vie + BEGINNING_OF_CENTRAL,
		this->p_v_n + BEGINNING_OF_CENTRAL,
		this->p_n_major,
		this->p_v_overall_minor + BEGINNING_OF_CENTRAL,
		p_n_shards,
		p_n_shards_n,
		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		Timestep // note this is used for determining distance.
		);
	Call(cudaThreadSynchronize(), "cudaTS calculate overall velocities");

	// Infer tri velocities based on actual moves of verts:
	kernelCentroidVelocitiesTriangles << <numTriTiles, threadsPerTileMinor >> > (
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

	// Set AreaMinor:
	
	// Ideally we would reset triangle centroids first but this won't be a lot different.

	kernelGet_AreaMinorFluid << <numTriTiles, threadsPerTileMinor >> > (
		pX_half->p_info,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_izNeigh_TriMinor,
		pX_half->p_szPBC_triminor,
		p_pressureflag,
		pX_half->p_AreaMinor // output
		);
	Call(cudaThreadSynchronize(), "cudaTS Get_AreaMinorFluid");
	 
	cudaMemset(p_Integrated_div_v_overall, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Div_v_neut, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Div_v, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(NT_addition_rates_d, 0, sizeof(NTrates)*NUMVERTICES);
	cudaMemset(p_store_NTFlux, 0, sizeof(NTrates)*NUMVERTICES*MAXNEIGH);

	if (!DEFAULTSUPPRESSVERBOSITY)
	{		
		cudaMemcpy(&tempf64, &(this->p_vie[VERTCHOSEN + BEGINNING_OF_CENTRAL].vez), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nvez [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);		 
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNiTi rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNeTe rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
	};
	kernelAccumulateAdvectiveMassHeatRateNew << <numTilesMajor, threadsPerTileMajor >> >(
		0.5*Timestep, // why it appears here?
		this->p_info,
		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_izNeigh_vert,   // we never pick up position from it so don't need per flag?
		sz_who_vert_vert,

		this->p_n_major, // unused?
		this->p_T_minor + BEGINNING_OF_CENTRAL,

		this->p_vie,
		this->p_v_overall_minor,
		
		p_n_shards,

		NT_addition_rates_d,
		p_Div_v,
		p_Integrated_div_v_overall,
		p_store_NTFlux
		);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateAdvectiveMassHeatRate");
	kernelAccumulateNeutralAdvectiveMassHeatRateNew << <numTilesMajor, threadsPerTileMajor >> >(
		0.5*Timestep, // why it appears here?
		this->p_info,
		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_izNeigh_vert,
		sz_who_vert_vert,
		 
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL,
		 
		this->p_v_n,
		this->p_v_overall_minor,

		p_n_shards_n,

		NT_addition_rates_d,
		p_Div_v_neut,
		p_store_NTFlux);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateAdvectiveMassHeatRate");
	  
	kernelAddStoredNTFlux << <numTilesMajor, threadsPerTileMajor >> > (
		this->p_info + BEGINNING_OF_CENTRAL,
		p_store_NTFlux,
		NT_addition_rates_d
		);
	Call(cudaThreadSynchronize(), "cudaTS AddStoredNTFlux");

	if (!DEFAULTSUPPRESSVERBOSITY)
	{
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNiTi rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNeTe rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NnTn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNnTn rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
	};

	// For now went for "advective change in v" where we divide d/dt Nv by N_derivsyst
	cudaMemset(p_MAR_neut, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_ion, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_elec, 0, sizeof(f64_vec3)*NMINOR);

	DivideNeTe_by_N << <numTilesMajor, threadsPerTileMajor >> >(
		NT_addition_rates_d,
		this->p_AreaMajor,
		this->p_n_major,
		p_Tgraph[6]);
	Call(cudaThreadSynchronize(), "cudaTS DivideNeTe_by_N");


	////////////////////////////////
	cudaMemset(
		NT_addition_tri_d, 0, sizeof(NTrates)*NUMVERTICES * 2);
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
		p_n_shards,
		this->p_n_minor,
		NT_addition_tri_d
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
		p_MAR_neut,

		NT_addition_tri_d
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelNeutral_momflux");
	 
	Collect_Nsum_at_tris << <numTriTiles, threadsPerTileMinor >> >(
		this->p_info,
		this->p_n_minor,
		this->p_tri_corner_index,
		this->p_AreaMajor, // populated?
		p_temp4, p_temp5);
	Call(cudaThreadSynchronize(), "cudaTS Nsum 2");
	 
	kernelTransmitHeatToVerts << <numTilesMajor, threadsPerTileMajor >> > (
		this->p_info,
		this->p_izTri_vert,
		this->p_n_minor,
		this->p_AreaMajor,
		p_temp4, p_temp5,
		NT_addition_rates_d,
		NT_addition_tri_d
		);
	Call(cudaThreadSynchronize(), "cudaTS sum up heat 2");
	long i;
	

	if (!DEFAULTSUPPRESSVERBOSITY)
	{ 
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNiTi rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNeTe rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NnTn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNnTn rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
	};
	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

	kernelReset_v_in_outer_frill_and_outermost << <numTilesMinor, threadsPerTileMinor >> >
		(
			this->p_info,
			this->p_vie,
			this->p_v_n,
			this->p_T_minor,
			this->p_tri_neigh_index,
			this->p_izNeigh_vert
			);
	Call(cudaThreadSynchronize(), "cudaTS resetv");

	//cudaMemset(NT_addition_tri_d, 0, sizeof(NTrates)*NUMVERTICES * 2); // don't know why this was here but it doesn't do anything.
	
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
	Call(cudaThreadSynchronize(), "cudaTS Advance_n_and_T"); 

	if (!DEFAULTSUPPRESSVERBOSITY)
	{
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNiTi rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNeTe rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
	};

	// DEBUG:
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
		); 
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
	
//	Need to figure that v lives on centroids because 1 we have viscosity and 2 
//	Az lives on centroids to avoid trouble.

	kernelAccelerate_v_from_advection << <numTilesMinor, threadsPerTileMinor >> >
		(   
			0.5*Timestep,
			this->p_info,
			this->p_n_minor,    // multiply by old mass ..
			this->p_AreaMinor,    // multiply by old mass ..
			pX_half->p_n_minor, // divide by new mass ..
			pX_half->p_AreaMinor,
			
			this->p_vie,
			this->p_v_n,

			p_MAR_neut, // these contain the mom flux due to advection.
			p_MAR_ion,
			p_MAR_elec,
			 
			// outputs:
			pX_half->p_vie,
			pX_half->p_v_n
			);
	Call(cudaThreadSynchronize(), "cudaTS kernelAccelerate_v_from_advection pX_half");

	if (!GlobalSuppressSuccessVerbosity) {
		f64 tempf64, tempb;
		cudaMemcpy(&tempf64, &(this->p_v_n[VERTCHOSEN + BEGINNING_OF_CENTRAL].y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		cudaMemcpy(&tempb, &(this->p_vie[VERTCHOSEN + BEGINNING_OF_CENTRAL].vxy.y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d this vn.y %1.9E vy %1.9E\n", VERTCHOSEN, tempf64, tempb);
		cudaMemcpy(&tempf64, &(pX_half->p_v_n[VERTCHOSEN + BEGINNING_OF_CENTRAL].y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		cudaMemcpy(&tempb, &(pX_half->p_vie[VERTCHOSEN + BEGINNING_OF_CENTRAL].vxy.y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d pX_half vn.y %1.9E vy %1.9E\n", VERTCHOSEN, tempf64, tempb);
	}
	 
	kernelReset_v_in_outer_frill_and_outermost << <numTilesMinor, threadsPerTileMinor >> >
		(
			pX_half->p_info,
			pX_half->p_vie,
			pX_half->p_v_n,
			pX_half->p_T_minor,
			pX_half->p_tri_neigh_index,
			pX_half->p_izNeigh_vert
			);
	Call(cudaThreadSynchronize(), "cudaTS resetv");

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	 
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	 
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
		pX_half->p_szPBCtri_vert,
		Timestep
		);
	Call(cudaThreadSynchronize(), "cudaTS calculate overall velocities 22");
	   
	kernelCentroidVelocitiesTriangles << <numTriTiles, threadsPerTileMinor >> >(
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


	kernelGet_AreaMinorFluid  << <numTriTiles, threadsPerTileMinor >> > (
		pX_target->p_info,
		pX_target->p_izTri_vert,
		pX_target->p_szPBCtri_vert,
		pX_target->p_izNeigh_TriMinor,
		pX_target->p_szPBC_triminor,
		p_pressureflag,
		pX_target->p_AreaMinor // output
			);
	Call(cudaThreadSynchronize(), "cudaTS Get_AreaMinorFluid");
		
	cudaMemset(p_Integrated_div_v_overall, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Div_v_neut, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Div_v, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(NT_addition_rates_d, 0, sizeof(NTrates)*NUMVERTICES);
	cudaMemset(p_store_NTFlux, 0, sizeof(NTrates)*NUMVERTICES*MAXNEIGH);

	kernelAccumulateAdvectiveMassHeatRateNew << <numTilesMajor, threadsPerTileMajor >> >(
		Timestep,
		pX_half->p_info,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_izNeigh_vert,
		sz_who_vert_vert,

		pX_half->p_n_major,
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL,
		 
		pX_half->p_vie,
		pX_half->p_v_overall_minor,

		p_n_shards,

		NT_addition_rates_d,
		p_Div_v,
		p_Integrated_div_v_overall,
		p_store_NTFlux);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateAdvectiveMassHeatRateNew pX_half");
	kernelAccumulateNeutralAdvectiveMassHeatRateNew << <numTilesMajor, threadsPerTileMajor >> >(
		Timestep,
		pX_half->p_info,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_izNeigh_vert,
		sz_who_vert_vert,

		pX_half->p_n_major,
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL,

		pX_half->p_v_n,
		pX_half->p_v_overall_minor,

		p_n_shards_n,

		NT_addition_rates_d,
		p_Div_v_neut,
		p_store_NTFlux);
	Call(cudaThreadSynchronize(), "cudaTS Accumulate Neutral NT pX_half");
	kernelAddStoredNTFlux << <numTilesMajor, threadsPerTileMajor >> > (
		pX_half->p_info + BEGINNING_OF_CENTRAL,
		p_store_NTFlux,
		NT_addition_rates_d
		);
	Call(cudaThreadSynchronize(), "cudaTS AddStoredNTFlux");

	if (!DEFAULTSUPPRESSVERBOSITY)
	{
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNiTi rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNeTe rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
	};

	CallMAC(cudaMemset(p_MAR_neut, 0, sizeof(f64_vec3)*NMINOR));
	CallMAC(cudaMemset(p_MAR_ion, 0, sizeof(f64_vec3)*NMINOR));
	CallMAC(cudaMemset(p_MAR_elec, 0, sizeof(f64_vec3)*NMINOR));
	cudaMemset(
		NT_addition_tri_d, 0, sizeof(NTrates)*NUMVERTICES * 2);

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
		p_n_shards,
		pX_half->p_n_minor,
		NT_addition_tri_d
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
		p_MAR_neut,
		NT_addition_tri_d
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelNeutral_momflux pX_half");
	 
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

	if (!DEFAULTSUPPRESSVERBOSITY)
	{
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d NT_addition_rates_d Nn rate %1.10E \n", VERTCHOSEN, tempf64);

		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNiTi rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNeTe rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
	};
	    
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

	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("%d NT_addition_rates_d Nn rate %1.10E \n", VERTCHOSEN, tempf64);

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

	kernelCreateShardModelOfDensities_And_SetMajorArea << <numTilesMajor, threadsPerTileMajor >> >(
		
		pX_target->p_info,
		pX_target->p_n_major,
		pX_target->p_n_minor,
		pX_target->p_izTri_vert,
		pX_target->p_szPBCtri_vert, // could be more intelligent way than storing and copying 3x
		pX_target->p_cc,
		p_n_shards,
		p_n_shards_n,
		pX_target->p_AreaMajor,
		false);
	Call(cudaThreadSynchronize(), "cudaTS CreateShardModel pX_target");
	 
	kernelInferMinorDensitiesFromShardModel << <numTilesMinor, threadsPerTileMinor >> >(
		pX_target->p_info,
		pX_target->p_n_minor,
		p_n_shards,
		p_n_shards_n,
		pX_target->p_tri_corner_index,
		pX_target->p_who_am_I_to_corner,
		p_one_over_n);// (At the moment just repopulating tri minor n.)
	Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities pX_target");

	cudaEventRecord(middle, 0);
	cudaEventSynchronize(middle); 

	kernelAccelerate_v_from_advection << <numTilesMinor, threadsPerTileMinor >> >
		(
			Timestep, 
			this->p_info,
			
			this->p_n_minor,    // multiply by old mass ..
			this->p_AreaMinor,
			pX_target->p_n_minor, // divide by new mass ..
			pX_target->p_AreaMinor,

			this->p_vie,
			this->p_v_n, // v_k

			p_MAR_neut, // these contain the mom flux due to advection.
			p_MAR_ion,
			p_MAR_elec,

			// outputs:
			pX_target->p_vie,
			pX_target->p_v_n
			);
	Call(cudaThreadSynchronize(), "cudaTS Accelerate_v_from_advection");
	 
	 
	if (!GlobalSuppressSuccessVerbosity) {
		f64 tempf64, tempb;
		cudaMemcpy(&tempf64, &(this->p_v_n[VERTCHOSEN + BEGINNING_OF_CENTRAL].y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		cudaMemcpy(&tempb, &(this->p_vie[VERTCHOSEN + BEGINNING_OF_CENTRAL].vxy.y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\n\n\n%d this vn.y %1.9E vy %1.9E\n", VERTCHOSEN, tempf64, tempb);
		cudaMemcpy(&tempf64, &(pX_half->p_v_n[VERTCHOSEN + BEGINNING_OF_CENTRAL].y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		cudaMemcpy(&tempb, &(pX_half->p_vie[VERTCHOSEN + BEGINNING_OF_CENTRAL].vxy.y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d pX_half vn.y %1.9E vy %1.9E\n", VERTCHOSEN, tempf64, tempb);
		cudaMemcpy(&tempf64, &(pX_target->p_v_n[VERTCHOSEN + BEGINNING_OF_CENTRAL].y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		cudaMemcpy(&tempb, &(pX_target->p_vie[VERTCHOSEN + BEGINNING_OF_CENTRAL].vxy.y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d pX_target vn.y %1.9E vy %1.9E\n\n\n", VERTCHOSEN, tempf64, tempb);
	}
	

	if (bGlobalSaveTGraphs)
	{
		Divide_diff_get_accel << <numTilesMajor, threadsPerTileMajor >> >(
			pX_target->p_vie + BEGINNING_OF_CENTRAL,
			this->p_vie + BEGINNING_OF_CENTRAL,
			Timestep,
			p_accelgraph[10]
		); 
		Call(cudaThreadSynchronize(), "cudaTS Divide_diff_get_accel");
	}
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

	nvals n;
	cudaMemcpy(&n, &(pX_half->p_n_major[VERTCHOSEN]), sizeof(nvals), cudaMemcpyDeviceToHost);
	printf("%d pX_half n %1.12E n_n %1.12E \n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", VERTCHOSEN, n.n, n.n_n);
	cudaMemcpy(&n, &(pX_target->p_n_major[VERTCHOSEN]), sizeof(nvals), cudaMemcpyDeviceToHost);
	printf("%d pX_target n %1.12E n_n %1.12E \n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", VERTCHOSEN, n.n, n.n_n);
	
	if (!GlobalSuppressSuccessVerbosity) {
		f64 tempf64;
		cudaMemcpy(&tempf64, &(pX_target->p_v_n[VERTCHOSEN + BEGINNING_OF_CENTRAL].y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d v.y %1.9E \n", VERTCHOSEN, tempf64);
	}
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
	
	GlobalSuppressSuccessVerbosity = DEFAULTSUPPRESSVERBOSITY;

	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

	// Clearly I hadn't moved this up.
	cudaMemset(NT_addition_rates_d, 0, sizeof(NTrates)*NUMVERTICES);

	//Came in with it.
	//Did it ever actually use it before? 
	//This seems not to fit. We don't double the change, we get it for the first time.
	//That's actually more concerning than that we came in with it.
	
	cudaEvent_t start, stop, middle;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&middle);
	cudaEventRecord(start, 0);
	cudaEventSynchronize(start);

#define USE_N_MAJOR_FOR_VERTEX 

	cudaMemset(p_Integrated_div_v_overall, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Div_v_neut, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Div_v, 0, sizeof(f64)*NUMVERTICES);

	Timestep = TIMESTEP;
	// DEBUG:
	printf("\nDebugNaN this\n\n");
	DebugNaN(this);
	
	if (!GlobalSuppressSuccessVerbosity) {
		long izTri[MAXNEIGH];

		cudaMemcpy(izTri, this->p_izTri_vert + MAXNEIGH*VERTCHOSEN, sizeof(long)*MAXNEIGH, cudaMemcpyDeviceToHost);
		int i;
		for (i = 0; i < MAXNEIGH; i++)
			printf("%d tri %d : %d\n", VERTCHOSEN, i, izTri[i]);

		long izNeigh[MAXNEIGH];
		cudaMemcpy(izNeigh, this->p_izNeigh_vert + MAXNEIGH*VERTCHOSEN, sizeof(long)*MAXNEIGH, cudaMemcpyDeviceToHost);
		for (i = 0; i < MAXNEIGH; i++)
			printf("%d neigh %d : %d\n", VERTCHOSEN, i, izNeigh[i]);

		LONG3 cornerindex;
		cudaMemcpy(&cornerindex, this->p_tri_corner_index + izTri[0], sizeof(LONG3), cudaMemcpyDeviceToHost);

		for (i = 0; i < 3; i++)
			printf("%d corner 012 : %d %d %d\n", izTri[0], cornerindex.i1, cornerindex.i2, cornerindex.i3);


		f64 tempf64;
		cudaMemcpy(&tempf64, &(this->p_v_n[VERTCHOSEN + BEGINNING_OF_CENTRAL].y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		printf("v_n.y[%d] %1.10E \n", VERTCHOSEN, tempf64);
	}

	// ```````````````````````````````````````````````````````````````
	//                        Thermal Pressure:
	// ```````````````````````````````````````````````````````````````

	// We are going to want n_shards and it is on centroids not circumcenters.
	// v lives on centroids .. as does Az.
	// Can anticipate problems if we tried to change that.

	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> > (
		this->p_n_minor,
		this->p_n_major,
		this->p_T_minor,
		this->p_info,
		this->p_cc,  // Calculate circumcenters; we would like n and T there for shards.

		this->p_tri_corner_index,
		this->p_tri_periodic_corner_flags,

		false // true == calculate n and T on circumcenters instead of centroids
			  // We are using n_minor for things where we never load cc.
		); // call before CreateShardModel 
	Call(cudaThreadSynchronize(), "cudaTS average nTx");
	cudaMemcpy(this->p_n_minor + BEGINNING_OF_CENTRAL,
		this->p_n_major, sizeof(nvals)*NUMVERTICES, cudaMemcpyDeviceToDevice);

	// To match how we do it below we should really be adding in iterations of ShardModel and InferMinorDensity.

	kernelCreateShardModelOfDensities_And_SetMajorArea << <numTilesMajor, threadsPerTileMajor >> > (
		this->p_info,
		this->p_n_major,
		this->p_n_minor,  // DESIRED VALUES
		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_cc,
		p_n_shards,
		p_n_shards_n,
		this->p_AreaMajor,
		false // USE CENTROIDS --- pressure routine does not load in cc's
		);
	Call(cudaThreadSynchronize(), "cudaTS CreateShardModels this");

	kernelInferMinorDensitiesFromShardModel << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info,
		this->p_n_minor,
		p_n_shards,
		p_n_shards_n,
		this->p_tri_corner_index,
		this->p_who_am_I_to_corner,
		p_one_over_n); // overwrites but it doesn't matter
	Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities ..");

	// Used for ?

	// For now went for "advective change in v" where we divide d/dt Nv by N_derivsyst
	cudaMemset(p_MAR_neut, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_ion, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_elec, 0, sizeof(f64_vec3)*NMINOR);

	kernelCreate_pressure_gradT_and_gradA_CurlA_minor_noadvect << <numTriTiles, threadsPerTileMinor >> > (

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
		p_pressureflag,
		p_GradTe,
		p_GradAz,
		this->p_B // HERE THIS IS GETTING PoP'D 
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelCreate_pressure_gradT_and_gradA_CurlA_minor");

	kernelNeutral_pressure << <numTriTiles, threadsPerTileMinor >> > (
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

		p_pressureflag,

		p_MAR_neut
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelNeutral_pressure");

	if (!GlobalSuppressSuccessVerbosity) {
		f64_vec3 tempvec3;
		cudaMemcpy(&tempvec3, &(p_MAR_ion[CHOSEN]),
			sizeof(f64_vec3), cudaMemcpyDeviceToHost);
		printf("\n%d pressure only : pMAR_ion.xy %1.9E %1.9E \n", CHOSEN, tempvec3.x, tempvec3.y);
		cudaMemcpy(&tempvec3, &(p_MAR_neut[CHOSEN]),
			sizeof(f64_vec3), cudaMemcpyDeviceToHost);
		printf("\n%d pressure only : pMAR_neut.xy %1.9E %1.9E \n", CHOSEN, tempvec3.x, tempvec3.y);
	};


	long i;
	SetConsoleTextAttribute(hConsole, 11);

	if (!DEFAULTSUPPRESSVERBOSITY)
	{
		// Report NnTn:
		SetConsoleTextAttribute(hConsole, 14);

		cudaMemcpy(&tempf64, &(this->p_vie[VERTCHOSEN + BEGINNING_OF_CENTRAL].vez), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nvez [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		//if (tempf64 <= 0.0) getch();

		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NnTn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("NnTn rate [%d] : %1.13E\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("NiTi rate [%d] : %1.13E\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNeTe rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		SetConsoleTextAttribute(hConsole, 15);
	}

	// ``````````````````````````````````````````````````````````````````````
	//              Heat conduction:
	// ``````````````````````````````````````````````````````````````````````

	// Putting heat cond first because we chose to do it on cc.
	// That can be questioned but it goes well with longitudinal flow between vertices - see formula in kernel.
	// Not changing it right now, but worth a rethink if possible.
	// The aim of the following is to put T on cc.

	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> > (
		this->p_n_minor,
		this->p_n_major,
		this->p_T_minor,
		this->p_info,
		this->p_cc,  // Calculate circumcenters; we would like n and T there for shards.

		this->p_tri_corner_index,
		this->p_tri_periodic_corner_flags,

		true // true == calculate n and T on circumcenters instead of centroids
			  // We are using n_minor for things where we never load cc.
		); // call before CreateShardModel 
	Call(cudaThreadSynchronize(), "cudaTS average nTx");
	cudaMemcpy(this->p_n_minor + BEGINNING_OF_CENTRAL,
		this->p_n_major, sizeof(nvals)*NUMVERTICES, cudaMemcpyDeviceToDevice);
		
	// We do not need to construct shard model however. That doesn't affect T_minor.
	// n from average is good enough for kappa & nu I should think.
	
	if (!DEFAULTSUPPRESSVERBOSITY)
	{
		SetConsoleTextAttribute(hConsole, 14);
		// Report NnTn:
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNiTi rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNeTe rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		SetConsoleTextAttribute(hConsole, 15);
	};

	kernelCalculate_kappa_nu << <numTriTiles, threadsPerTileMinor >> > (
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

	if (iHistory > 0)
	{
		SetConsoleTextAttribute(hConsole, 11);
		RegressionSeedTe(0.5*Timestep, p_store_T_move2, p_store_T_move1,
			pX_half->p_T_minor + BEGINNING_OF_CENTRAL,
			this->p_T_minor + BEGINNING_OF_CENTRAL,
			this,
			false);
		SetConsoleTextAttribute(hConsole, 14);
	}
	else {
		printf("\a iHistory %d", iHistory);
	}

	int iSuccess;
#define WOOLLY
#ifndef WOOLLY
	do {
		iSuccess = RunBackwardForHeat_ConjugateGradient(
			this->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices,
			pX_half->p_T_minor + BEGINNING_OF_CENTRAL, // dest
			0.5*Timestep,
			this,
			false);
		if (iSuccess != 0) iSuccess = RunBackwardJLSForHeat(
			this->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices,
			pX_half->p_T_minor + BEGINNING_OF_CENTRAL, // dest
			0.5*Timestep,
			this,
			false);
		
	} while (iSuccess != 0);
	getch();

#else

	kernelUnpack << <numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_Tn, p_Ti, p_Te, pX_half->p_T_minor + BEGINNING_OF_CENTRAL); // create T
	Call(cudaThreadSynchronize(), "cudaTS kernelUnpack k+1 B");

	kernelUnpack << <numTilesMajorClever, threadsPerTileMajorClever >> >
		(p_Tnk, p_Tik, p_Tek, this->p_T_minor + BEGINNING_OF_CENTRAL); // create T
	Call(cudaThreadSynchronize(), "cudaTS kernelUnpack k");

	iEquations[0] = NUMVERTICES;
	iEquations[1] = NUMVERTICES;
	iEquations[2] = NUMVERTICES;

	printf("NEUTRAL SOLVE:\n");
	do {
		iSuccess = RunBwdJnLSForHeat(p_Tnk, p_Tn,
			0.5*Timestep, this, false,
			0,
			p_kappa_n,
			p_nu_i);
		
	} while (iSuccess != 0);
	
	printf("ION SOLVE:\n");
	do {
		iSuccess = RunBwdJnLSForHeat(p_Tik, p_Ti,
			0.5*Timestep, this, false,
			1,
			p_kappa_i,
			p_nu_i);
	} while (iSuccess != 0);

	printf("ELECTRON SOLVE:\n");
	do {
		iSuccess = RunBwdJnLSForHeat(p_Tek, p_Te,
			0.5*Timestep, this, false,
			2,
			p_kappa_e,
			p_nu_e);
	} while (iSuccess != 0);
	
	kernelPackupT3 << <numTilesMajorClever, threadsPerTileMajorClever >> >
		(pX_half->p_T_minor + BEGINNING_OF_CENTRAL, p_Tn, p_Ti, p_Te); // create T
	Call(cudaThreadSynchronize(), "cudaTS kernelPack k+1");

#endif
	
	SubtractT3 << <numTilesMajor, threadsPerTileMajor >> >
		(p_store_T_move2,
			pX_half->p_T_minor + BEGINNING_OF_CENTRAL,
			this->p_T_minor + BEGINNING_OF_CENTRAL);
	Call(cudaThreadSynchronize(), "cudaTS subtractT3");

	SetConsoleTextAttribute(hConsole, 15);

	cudaMemcpy(&tempf64, &((pX_half->p_T_minor + BEGINNING_OF_CENTRAL + VERTCHOSEN)->Te), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("solved Te[%d]: %1.9E\n", VERTCHOSEN, tempf64);
	if (tempf64 < 0.0) {
		printf("4getch");  getch(); getch(); getch(); getch(); 
		PerformCUDA_Revoke(); printf("end");
		while (1) getch();
	}
	
	if ((DEBUGTE) && (tempf64 > 1.0e-11)) {
		printf("press f");
		while (getch() != 'f');
	}

	// This isn't ideal ---- for this one we might like the old half move and the old
	// full move stored
	// but for the full move which is the more expensive one -- do we really want
	// half move and old move?

	// this did have area populated... within ins
	// dNT/dt = 0 here anyway!
	cudaMemcpy(NT_addition_rates_d_temp, NT_addition_rates_d, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	cudaMemset(p_boolarray, 0, sizeof(bool)*NUMVERTICES * 2); // initially allow all flows good

	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("%d NT_addition_rates_d Nn rate %1.10E \n", VERTCHOSEN, tempf64);

	int iPass = 0;
	bool bContinue;
	do {
		printf("iPass %d :\n", iPass);

		// reset NTrates:
		cudaMemcpy(NT_addition_rates_d, NT_addition_rates_d_temp, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToDevice);

		kernelAccumulateDiffusiveHeatRate_new_Full << <numTilesMajorClever, threadsPerTileMajorClever >> > (
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
			(iPass == 0) ? false : true,
			     
			p_boolarray2,
			p_boolarray_block,
			false);
		Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate pX_half");
		// To increase the efficiency we want to make a clever 2nd set of major tiles of size 192. Also try 256, 384.
		
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d NT_addition_rates_d Nn rate %1.10E \n", VERTCHOSEN, tempf64);

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
			p_bFailed, // did we hit any new negative T to add

			p_boolarray2,
			p_boolarray_block,
			false
			);
		Call(cudaThreadSynchronize(), "cudaTS kernelCreatePutativeT");

		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d NT_addition_rates_d Nn rate %1.10E \n", VERTCHOSEN, tempf64);

		bContinue = false;
		cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMajorClever, cudaMemcpyDeviceToHost);
		int i;
		for (i = 0; ((i < numTilesMajorClever) && (p_boolhost[i] == 0)); i++);
		if (i < numTilesMajorClever) bContinue = true;
		iPass++;
	} while (bContinue);
	  
	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("%d NT_addition_rates_d Nn rate %1.10E \n", VERTCHOSEN, tempf64);

	cudaMemcpy(store_heatcond_NTrates, NT_addition_rates_d, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToDevice);

	cudaMemcpy(&tempf64, &(store_heatcond_NTrates[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("%d store_heatcond_NTrates Nn rate %1.10E \n", VERTCHOSEN, tempf64);


	// Overwrite minor densities back again:

	kernelInferMinorDensitiesFromShardModel << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info,
		this->p_n_minor,
		p_n_shards,
		p_n_shards_n,
		this->p_tri_corner_index,
		this->p_who_am_I_to_corner,
		p_one_over_n); // overwrites but it doesn't matter
	Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities ..");

	if (!GlobalSuppressSuccessVerbosity) {
		f64_vec3 tempvec3;
		cudaMemcpy(&tempvec3, &(p_MAR_ion[CHOSEN]),
			sizeof(f64_vec3), cudaMemcpyDeviceToHost);
		printf("\n%d before ionization : pMAR_ion.xy %1.9E %1.9E \n", CHOSEN, tempvec3.x, tempvec3.y);
		cudaMemcpy(&tempvec3, &(p_MAR_neut[CHOSEN]),
			sizeof(f64_vec3), cudaMemcpyDeviceToHost);
		printf("\n%d before ionization : pMAR_neut.xy %1.9E %1.9E \n", CHOSEN, tempvec3.x, tempvec3.y);
	};

	cudaMemset(p_temp3_1, 0, sizeof(f64_vec3)*numTilesMajor*threadsPerTileMajor);
	cudaMemset(p_temp3_2, 0, sizeof(f64_vec3)*numTilesMajor*threadsPerTileMajor);
	cudaMemset(p_temp3_3, 0, sizeof(f64_vec3)*numTilesMajor*threadsPerTileMajor);
	kernelIonisationRates << <numTilesMajor, threadsPerTileMajor >> > (
		0.5*Timestep,
		this->p_info,
		this->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices
		this->p_n_major,
		this->p_AreaMajor,
		NT_addition_rates_d,
		p_temp3_1,//p_MAR_temp_major_n,
		p_temp3_2,//p_MAR_temp_major_i,
		p_temp3_3,//p_MAR_temp_major_e,
		this->p_vie + BEGINNING_OF_CENTRAL,
		this->p_v_n + BEGINNING_OF_CENTRAL,
		0,0
		);  
	Call(cudaThreadSynchronize(), "cudaTS Ionisation");
	
	cudaMemcpy(&tempf64, &(p_temp3_3[VERTCHOSEN].z), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("\np_temp3_3.z [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);

	Collect_Ntotal_major << <numTilesMajor, threadsPerTileMajor >> >(
		this->p_info,
		this->p_izTri_vert,
		this->p_n_minor,
		this->p_AreaMinor,
		p_temp1, // Ntotal major
		p_temp2  // Nntotal major
		);
	Call(cudaThreadSynchronize(), "cudaTS Gather Ntotal");
	  
	Augment_dNv_minor << <numTilesMinor, threadsPerTileMinor >> >(
		this->p_info,
		this->p_n_minor,
		this->p_tri_corner_index,
		p_temp1,
		p_temp2,
		this->p_AreaMinor,
		p_temp3_1, p_temp3_2, p_temp3_3,
		p_MAR_neut,
		p_MAR_ion,
		p_MAR_elec);
	Call(cudaThreadSynchronize(), "cudaTS Augment_dNv_minor");
	

	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

	if (!DEFAULTSUPPRESSVERBOSITY)
	{
		SetConsoleTextAttribute(hConsole, 14);
		// Report NnTn:
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NnTn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNnTn rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNiTi rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNeTe rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(p_MAR_elec[VERTCHOSEN + BEGINNING_OF_CENTRAL].z), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nMAR_elec.z [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		SetConsoleTextAttribute(hConsole, 15);
	};

	kernelReset_v_in_outer_frill_and_outermost << <numTilesMinor, threadsPerTileMinor >> >
	(
		this->p_info,
		this->p_vie,
		this->p_v_n,
		this->p_T_minor,
		this->p_tri_neigh_index,
		this->p_izNeigh_vert
		);
	Call(cudaThreadSynchronize(), "cudaTS resetv");
	 
#ifdef PRECISE_VISCOSITY

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


	// MASKING: remember we'll need to dynamically add to mask, the neighbours can be affected.

	/*
	// 1. Compute the putative forward step.
	
	cudaMemset(p_MAR_neut2)
	cudaMemset(p_MAR_ion2)
	cudaMemset(p_MAR_elec2)
	cudaMemset(NT_addition_tri_d, 0, sizeof(NTrates)*NUMVERTICES * 2);
	kernelCreate_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> > (

		this->p_info,
		this->p_vie,
		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_izNeigh_TriMinor,
		this->p_szPBC_triminor,

		p_temp1, // p_ita_parallel_ion_minor,   // nT / nu ready to look up
		p_temp2, //f64 * __restrict__ p_ita_parallel_elec_minor,   // nT / nu ready to look up
		p_temp3, //f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
		p_temp4, // f64 * __restrict__ p_nu_elec_minor,   // nT / nu ready to look up

		this->p_B,

		p_MAR_ion2, 
		p_MAR_elec2,
		NT_addition_rates_d2,
		NT_addition_tri_d2);
	Call(cudaThreadSynchronize(), "cudaTS visccontrib 1");

	kernelCreate_neutral_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> >
		(this->p_info,
			this->p_v_n,
			this->p_izTri_vert,
			this->p_szPBCtri_vert,
			this->p_izNeigh_TriMinor,
			this->p_szPBC_triminor,
			p_temp6, // ita
			p_temp5, // nu		
			p_MAR_neut2, // just accumulates
			NT_addition_rates_d2, // probably throw away
			NT_addition_tri_d2);
	Call(cudaThreadSynchronize(), "cudaTS visccontrib 1");

	kernelComputePutativeVelocity << <numTriTiles, threadsPerTileMinor >> > (
		this->p_info,
		this->p_vie,
		this->p_v_n,
		p_MAR_neut2,
		p_MAR_ion2,
		p_MAR_elec2,

		pX_half->p_vie,
		pX_half->p_v_n,
		);
	Call(cudaThreadSynchronize(), "cudaTS 	kernelComputePutativeVelocity");

	cudaMemset(p_MAR_neut3)
		cudaMemset(p_MAR_ion3)
		cudaMemset(p_MAR_elec3)
	cudaMemset(NT_addition_tri_d, 0, sizeof(NTrates)*NUMVERTICES * 2);
	kernelCreate_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> > (

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

		p_MAR_ion3, 
		p_MAR_elec3,
		NT_addition_rates_d_throwaway,
		NT_addition_tri_d_throwaway);
	Call(cudaThreadSynchronize(), "cudaTS visccontrib 1");

	kernelCreate_neutral_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> >
		(this->p_info,
			pX_half->p_v_n,
			this->p_izTri_vert,
			this->p_szPBCtri_vert,
			this->p_izNeigh_TriMinor,
			this->p_szPBC_triminor,
			p_temp6, // ita
			p_temp5, // nu		
			p_MAR_neut3, // just accumulates
			NT_addition_rates_d_throwaway,
			NT_addition_tri_d_throwaway);
	Call(cudaThreadSynchronize(), "cudaTS visccontrib 1");

	cudaMemset(p_bz_mask_visc)
	cudaMemset(p_bz_mask_visc_neut)
	kernelCompareVelocity_SetStabilityFlag << <numTilesMinor, threadsPerTileMinor >> >
		(
			this->p_info,
			p_MAR_neut2,
			p_MAR_ion2,
			p_MAR_elec2,
			p_MAR_neut3,
			p_MAR_ion3,
			p_MAR_elec3,
			p_bz_mask_visc,
			p_bz_mask_visc_neut
			);
	Call(cudaThreadSynchronize(), "cudaTS kernelCompareVelocity_SetStabilityFlag");

	// Now if we change these v values can it wreck the nearby stability?
	** That's what we need to understand.

	cudaMemset(p_bz_mask_block)
	cudaMemset(p_bz_mask_block_neut)
	kernelCountEquations_SetBlockFlag << <numTilesMinor, threadsPerTileMinor >> >
		(
		p_bz_mask_visc,
		p_bz_mask_visc_neut
		p_boolarray_block
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelCountEquations_SetBlockFlag ");

	SolveBwd_Viscosity
		(true, // use bool mask array for which equations
			p_temp6, // ita
			p_temp5, // nu		
			pX_half->p_vie
		);

	// Think about this ... if the flag is off, we don't need to look through edges there.
	// But having run it we will need to look through them.
	// Can we say that won't have made the place unstable?


	// finally:

	*/

	RunBackwardJLSForViscosity(this->p_vie, pX_half->p_vie, Timestep, this, p_storeviscmove, bViscousHistory);
	RunBackwardR8LSForNeutralViscosity(this->p_v_n, pX_half->p_v_n, Timestep, this);
	
	cudaMemset(NT_addition_tri_d, 0, sizeof(NTrates)*NUMVERTICES * 2);
	kernelCreate_viscous_contrib_to_MAR_and_NT << <numTriTiles, threadsPerTileMinor >> > (

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
		(this->p_info,
			pX_half->p_v_n,
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

		if (!GlobalSuppressSuccessVerbosity) {
			f64_vec3 tempvec3;
			cudaMemcpy(&tempvec3, &(p_MAR_ion[CHOSEN]),
				sizeof(f64_vec3), cudaMemcpyDeviceToHost);
			printf("\n%d post viscosity : pMAR_ion.xy %1.9E %1.9E \n", CHOSEN, tempvec3.x, tempvec3.y);
			cudaMemcpy(&tempvec3, &(p_MAR_neut[CHOSEN]),
				sizeof(f64_vec3), cudaMemcpyDeviceToHost);
			printf("\n%d post viscosity : pMAR_neut.xy %1.9E %1.9E \n\n", CHOSEN, tempvec3.x, tempvec3.y);
		};

	
	Collect_Nsum_at_tris << <numTriTiles, threadsPerTileMinor >> > (
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
	
#endif

	// Well here's a thought.
	// We ARE expecting v to change when we do a backward viscosity.
	// Yet, we will find v off from its trajectory towards that point.
	// That's when we tune the viscous flow.


	if (!DEFAULTSUPPRESSVERBOSITY)
	{
		SetConsoleTextAttribute(hConsole, 14);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNT_addition_rates_d[%d].NeTe %1.10E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNT_addition_rates_d[%d].NiTi %1.10E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NnTn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNnTn rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		SetConsoleTextAttribute(hConsole, 15);
	}

	kernelAdvanceDensityAndTemperature_noadvectioncompression << <numTilesMajor, threadsPerTileMajor >> > (
		0.5*Timestep,
		this->p_info + BEGINNING_OF_CENTRAL,
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL,
		NT_addition_rates_d,
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL,
		this->p_vie + BEGINNING_OF_CENTRAL, // for resistive htg
		this->p_v_n + BEGINNING_OF_CENTRAL, // fixed bug
		this->p_AreaMajor,
		pX_half->p_n_major,
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL,
		this->p_B + BEGINNING_OF_CENTRAL		  
		);
	// Add in a test for T<0 !!!
	Call(cudaThreadSynchronize(), "cudaTS Advance_n_and_T _noadvect"); // vertex


	if (!DEFAULTSUPPRESSVERBOSITY) {
		printf("\nDebugNaN pX_half\n\n");
		DebugNaN(pX_half);
	};

	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> > (
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

	kernelInferMinorDensitiesFromShardModel << <numTilesMinor, threadsPerTileMinor >> > (
		pX_half->p_info,
		pX_half->p_n_minor,
		p_n_shards,
		p_n_shards_n,
		pX_half->p_tri_corner_index,
		pX_half->p_who_am_I_to_corner,
		p_one_over_n2);// (At the moment just repopulating tri minor n.)// based on pXhalf !!
	Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities pX_half");

	//	Iz_prescribed_starttime = GetIzPrescribed(evaltime); // because we are setting pX_half->v

	f64 store_evaltime = evaltime;
	// Iz_prescribed_endtime = GetIzPrescribed(evaltime + 0.5*Timestep); // because we are setting pX_half->v

	if (!GlobalSuppressSuccessVerbosity) {
		f64_vec3 tempvec3;
		cudaMemcpy(&tempvec3, &(p_MAR_ion[CHOSEN]),
			sizeof(f64_vec3), cudaMemcpyDeviceToHost);
		printf("\n%d ? : pMAR_ion.xy %1.9E %1.9E \n", CHOSEN, tempvec3.x, tempvec3.y);
		cudaMemcpy(&tempvec3, &(p_MAR_neut[CHOSEN]),
			sizeof(f64_vec3), cudaMemcpyDeviceToHost);
		printf("\n%d ? : pMAR_neut.xy %1.9E %1.9E \n\n", CHOSEN, tempvec3.x, tempvec3.y);
	};


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

	cudaMemcpy(p_AAdot_start, this->p_AAdot, sizeof(AAdot)*NMINOR, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_vie_start, this->p_vie, sizeof(v4)*NMINOR, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_v_n_start, this->p_v_n, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);

	GosubAccelerate(SUBCYCLES/2,//iSubcycles, 
		(0.5*Timestep) / (real)(SUBCYCLES), // hsub
		pX_half, // pX_use
		this // pX_intermediate
	);

	cudaMemcpy(pX_half->p_AAdot, p_AAdot_target, sizeof(AAdot)*NMINOR, cudaMemcpyDeviceToDevice);
	cudaMemcpy(pX_half->p_vie, p_vie_target, sizeof(v4)*NMINOR, cudaMemcpyDeviceToDevice);
	cudaMemcpy(pX_half->p_v_n, p_v_n_target, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);

	if (!GlobalSuppressSuccessVerbosity) {
		f64 tempf64;
		cudaMemcpy(&tempf64, &(pX_half->p_v_n[VERTCHOSEN + BEGINNING_OF_CENTRAL].y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d pX_half vn.y %1.9E \n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(pX_half->p_v_n[VERTCHOSEN2 + BEGINNING_OF_CENTRAL].y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d pX_half vn.y %1.9E \n", VERTCHOSEN2, tempf64);
		cudaMemcpy(&tempf64, &(pX_half->p_v_n[CHOSEN1].y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d pX_half vn.y %1.9E \n", CHOSEN1, tempf64);
		cudaMemcpy(&tempf64, &(pX_half->p_v_n[CHOSEN2].y),
			sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d pX_half vn.y %1.9E \n", CHOSEN2, tempf64);
	}
	

	long iTile;
	
	//kernelPopulateBackwardOhmsLaw_noadvect << <numTilesMinor, threadsPerTileMinor >> > (
	//	0.5*Timestep,
	//	this->p_info,
	//	p_MAR_neut, p_MAR_ion, p_MAR_elec,
	//	this->p_B, // for target it wasn't populated, right? Only used for thermal & nu_effective ?
	//	p_LapAz,
	//	p_GradAz,
	//	p_GradTe,
	//	this->p_n_minor,
	//	this->p_T_minor,
	//	this->p_vie,
	//	this->p_v_n,
	//	this->p_AAdot,
	//	this->p_AreaMinor,
 //  
	//	p_vn0,
	//	p_v0,
	//	p_OhmsCoeffs,
	//	p_Iz0_summands,
	//	p_sigma_Izz,
	//	p_denom_i, p_denom_e, p_coeff_of_vez_upon_viz, p_beta_ie_z,
	//	true);
	//Call(cudaThreadSynchronize(), "cudaTS AccelerateOhms 1");

	//cudaMemcpy(p_Iz0_summands_host, p_Iz0_summands, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	//cudaMemcpy(p_summands_host, p_sigma_Izz, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	//cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	//
	//f64 Iz0 = 0.0;
	//f64 sigma_Izz = 0.0;
	//Iz_k = 0.0;
	//for (iTile = 0; iTile < numTilesMinor; iTile++)
	//{
	//	Iz0 += p_Iz0_summands_host[iTile];
	//	sigma_Izz += p_summands_host[iTile];
	//	Iz_k += p_temphost1[iTile];
	//	if ((Iz0 != Iz0) || (sigma_Izz != sigma_Izz)) printf("tile %d Iz0 %1.9E sigma_Izz %1.9E summands %1.9E %1.9E \n",
	//		iTile, Iz0, sigma_Izz, p_Iz0_summands_host[iTile], p_summands_host[iTile]);
	//	// Track down what cell causing NaN Iz0
	//};

	//f64 Ez_strength_ = (Iz_prescribed_endtime - Iz0) / sigma_Izz;
	//Set_f64_constant(Ez_strength, Ez_strength_);

	//f64 neg_Iz_per_triangle = -(Iz_prescribed_endtime) / (f64)numReverseJzTriangles;
	//Set_f64_constant(negative_Iz_per_triangle, neg_Iz_per_triangle);

	//printf("GPU: Iz_prescribed %1.14E Iz0 %1.14E sigma_Izz %1.14E \n",
	//	Iz_prescribed_endtime, Iz0, sigma_Izz);
	//printf("Ez_strength (GPU) %1.14E \n", Ez_strength_);

	//// Update velocities and Azdot:
	//kernelCalculateVelocityAndAzdot_noadvect << <numTilesMinor, threadsPerTileMinor >> > (
	//	0.5*Timestep,
	//	pX_half->p_info,
	//	p_vn0,
	//	p_v0,
	//	p_OhmsCoeffs,
	//	pX_target->p_AAdot,  // why target? intermediate value
	//	pX_half->p_n_minor,
	//	this->p_AreaMinor,
	//	p_LapAz,
	//	pX_half->p_AAdot,
	//	pX_half->p_vie,
	//	pX_half->p_v_n
	//	);
	//Call(cudaThreadSynchronize(), "cudaTS AccelerateUpdate 1");

	// =====================

	SetConsoleTextAttribute(hConsole, 15);
	evaltime = store_evaltime;

//	kernelAdvanceAzEuler << <numTilesMinor, threadsPerTileMinor >> >
//		(0.5*h, this->p_AAdot, pX_half->p_AAdot, p_ROCAzduetoAdvection);
//	Call(cudaThreadSynchronize(), "cudaTS AdvanceAzEuler");
		
	// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	// Why are we doing this way? Consistent would be better.
	// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


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

	if (0) {
		SetConsoleTextAttribute(hConsole, 14);
		cudaMemcpy(&tempf64, &(pX_half->p_vie[VERTCHOSEN + BEGINNING_OF_CENTRAL].vez), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nvez pX_half [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);

		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d Nn rate %1.10E \n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NnTn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNnTn rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNT_addition_rates_d[%d].NeTe %1.14E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNT_addition_rates_d[%d].NiTi %1.10E\n\n", VERTCHOSEN, tempf64);
		SetConsoleTextAttribute(hConsole, 15);
	}
	SetConsoleTextAttribute(hConsole, 15);
	
	CallMAC(cudaMemset(p_MAR_neut, 0, sizeof(f64_vec3)*NMINOR));
	CallMAC(cudaMemset(p_MAR_ion, 0, sizeof(f64_vec3)*NMINOR));
	CallMAC(cudaMemset(p_MAR_elec, 0, sizeof(f64_vec3)*NMINOR));

	// Now notice we take a grad Azdot but Azdot has not been defined except from time t_k!!
	kernelCreate_pressure_gradT_and_gradA_CurlA_minor_noadvect << <numTriTiles, threadsPerTileMinor >> > (
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

		p_pressureflag,
		p_GradTe,
		p_GradAz,

		pX_half->p_B
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelCreate_pressure_gradT_and_gradA_CurlA_minor_noadvect pX_half");
	
	// Copy it to pX_target because we haven't called pressure there yet:
	 
	cudaMemcpy(pX_target->p_AreaMinor, pX_half->p_AreaMinor, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
	
	kernelNeutral_pressure << <numTriTiles, threadsPerTileMinor >> > (
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

		p_pressureflag,

		p_MAR_neut
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelNeutral_pressure pX_half");

	if (bGlobalSaveTGraphs) {
		cudaMemcpy(p_MAR_ion_pressure_major_stored,
			p_MAR_ion + BEGINNING_OF_CENTRAL, sizeof(f64_vec3)*NUMVERTICES, cudaMemcpyDeviceToDevice);
		cudaMemcpy(p_MAR_elec_pressure_major_stored,
			p_MAR_elec + BEGINNING_OF_CENTRAL, sizeof(f64_vec3)*NUMVERTICES, cudaMemcpyDeviceToDevice);

		// Pre-store existing:
		cudaMemcpy(p_MAR_elec_ionization_major_stored,
			p_MAR_elec + BEGINNING_OF_CENTRAL, sizeof(f64_vec3)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	};

	if (bGlobalSaveTGraphs) {
		DivideMAR_get_accel << <numTilesMajor, threadsPerTileMajor >> > (
			p_MAR_ion + BEGINNING_OF_CENTRAL,
			p_MAR_elec + BEGINNING_OF_CENTRAL,
			this->p_n_minor + BEGINNING_OF_CENTRAL,
			this->p_AreaMinor + BEGINNING_OF_CENTRAL, // we'll look in the minor central cell, this is where MAR applies.
			p_accelgraph[4],
			p_accelgraph[5]
			);
		Call(cudaThreadSynchronize(), "cudaTS DivideMAR_get_accel");

		cudaMemcpy(p_MAR_ion_temp_central, p_MAR_ion + BEGINNING_OF_CENTRAL, sizeof(f64_vec3)*NUMVERTICES,
			cudaMemcpyDeviceToDevice);
		cudaMemcpy(p_MAR_elec_temp_central, p_MAR_elec + BEGINNING_OF_CENTRAL, sizeof(f64_vec3)*NUMVERTICES,
			cudaMemcpyDeviceToDevice);
	}

	if (!GlobalSuppressSuccessVerbosity) {
		f64_vec3 tempvec3;
		cudaMemcpy(&tempvec3, &(p_MAR_ion[CHOSEN]),
			sizeof(f64_vec3), cudaMemcpyDeviceToHost);
		printf("\n%d pressure : pMAR_ion.xy %1.9E %1.9E \n", CHOSEN, tempvec3.x, tempvec3.y);
		cudaMemcpy(&tempvec3, &(p_MAR_neut[CHOSEN]),
			sizeof(f64_vec3), cudaMemcpyDeviceToHost);
		printf("\n%d pressure : pMAR_neut.xy %1.9E %1.9E \n\n", CHOSEN, tempvec3.x, tempvec3.y);
	};

	// If we're going to wipe out n temporarily, let's store it to put back afterwards:
	cudaMemcpy(pX_target->p_n_minor, pX_half->p_n_minor, sizeof(nvals)*NMINOR, cudaMemcpyDeviceToDevice);
	cudaMemcpy(pX_target->p_T_minor, pX_half->p_T_minor, sizeof(T3)*NMINOR, cudaMemcpyDeviceToDevice);

	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> > (
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

	// Do not need cc shard model

	kernelCalculate_kappa_nu << <numTriTiles, threadsPerTileMinor >> > (
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
	 
#define HEATBASESYST this
#define HEATSTEP Timestep

	// not ready to try and change .. means changing the advance also ...

	
	// 8th Nov 2019. Experiment to see if we can sometimes get away with midpoint step,
	// ie use forward step here off of the half-step of backward.
	// Try stability test:

	// 1. Try midpoint step to pX_target->T_major
	// -- we can if necessary shield against negatives in the same way as we do for bwd.
	// If some T went negative anyway then we scrap the midpoint attempt.
	// 2. Run NTrates on new position. If something reversed sign and increased in magnitude, we scrap the midpoint attempt.
	// Need to think about that carefully in terms that we should produce NTrates overall in the same way.
	// So that requires the rollaround to get rid of negatives.
	
	// // a. Longitudinal. Calc putative T, Check for negative. If they exist, give up on this way entirely.
	// // b. Full with rollaround. 
	// // c. Advance to target system
	// // d. On target,  Calc putative T, check for negative. If they exist, give up on this way.
	// e. Full with rollaround to get rid of negative. What is timestep?

	// -- or,
	// i. just take the ROC we already calculated and use this. 
	// c. Advance to target system.
	// ii. Check for negative T. If they exist, forget it, do backward.
	// iii. on target system, use existing boolarray to calc new rates, full vs longi. 
	// iv. Now compare with the NTRates we used to get us here. If we reversed sign and 
	// it is greater magnitude now, then give up and do backward.

	// maybe just those places need to be backward? That could be a quick solve, set
	// most values to epsilon == 0 and value is sorted. Good project for a day.
	
	// It failed, 1 negative value.

	// New effort. Let boolarray store where we want to do a solve - expand it to 3*NVERTS.
	// Then we just need to execute carefully. 
	
	// 1a. Find how many negative T and switch on bool
	// 1b. Find how many switched over stability and switch on bool
	// 1c. Set their neighbours to on also.

	// . Main task in CG routine is Calc d/dt(NT) given T -- we need to skip for most
	// we can let it exist or just never use it
	// We then calc epsilon which we should leave at 0 and make sure any additive regressor
	// is 0 outside of the "on" cells.
	// 
	// Calc d/dtNT is easily the most expensive routine involved.
	// We should detect create block-level flags so that whole blocks can be switched off,
	// as well as individual flags, which will leave NTrates unamended and uncalculated.
	// It will have to load shared data for tiles that have points in, but that does not 
	// mean that we don't save anything by only running those points. We do.
	// 
	// It should also be a quicker solve if there are effectively only a smaller number
	// of equations --- which may not even connect.
	
	// This is 100% legit.

	bool btemp;
	bool bBackward = false;
	// 1a. Find how many negative T and switch on bool
	cudaMemset(p_boolarray2, 0, sizeof(bool) * 3 * NUMVERTICES);
	// p_boolarray is still needed because it shows how to create midpoint step: use longi
	// and after we do backward longi solve we will need to decide whether those points are longi or full as in general.


	// This is a mistaken decision, we shouldn't be just looking at whether we get negative this way.
	// We should be doing it more simply: take our half-time system, recalculate heat flows, ask if we are getting negative.

	// I see the logic: our first half-step was actually a backward step.
	// That isn't a necessary way of doing it.


	cudaMemcpy(&tempf64, &(store_heatcond_NTrates[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("%d store_heatcond_NTrates Nn rate %1.10E \n", VERTCHOSEN, tempf64);

	cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever);
	kernelCreatePutativeTandsave << < numTilesMajorClever, threadsPerTileMajorClever >> > (
		HEATSTEP,
		pX_half->p_info,
		HEATBASESYST->p_T_minor + BEGINNING_OF_CENTRAL,
		pX_half->p_n_major,
		pX_half->p_AreaMajor,
		store_heatcond_NTrates,
		pX_target->p_T_minor + BEGINNING_OF_CENTRAL, // THE FORWARD MOVE using stored NTrates
		p_boolarray2  // store here 1 if it is below 0
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelCreatePutativeTandsave");
//	 
//	cudaMemcpy(&tempf64, &((pX_target->p_T_minor + BEGINNING_OF_CENTRAL + VERTCHOSEN)->Te), sizeof(f64),
//		cudaMemcpyDeviceToHost);
//	printf("Contents of Te[%d] : %1.10E\n\n", VERTCHOSEN, tempf64);

	cudaMemcpy(&tempf64, &(store_heatcond_NTrates[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("%d store_heatcond_NTrates Nn rate %1.10E \n", VERTCHOSEN, tempf64);

	// For safety let's check again properly: are there negative T or not?
	cudaMemset(p_longtemp, 0, sizeof(long)*numTilesMajorClever);
	kernelReturnNumberNegativeT << < numTilesMajorClever, threadsPerTileMajorClever >> > (
		pX_half->p_info,
		pX_target->p_T_minor + BEGINNING_OF_CENTRAL,
		p_longtemp
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelReturnNumberNegativeT");
	cudaMemcpy(p_longtemphost, p_longtemp, sizeof(long)*numTilesMajorClever, cudaMemcpyDeviceToHost);
	long iNegativeT = 0;
	for (i = 0; (i < numTilesMajorClever); i++)
		iNegativeT += p_longtemphost[i];
	if (iNegativeT > 0) {
		printf("%d negatives!", iNegativeT);
		bBackward = true;
	}
	cudaMemset(NT_addition_rates_d, 0, sizeof(NTrates)*NUMVERTICES);
	kernelAccumulateDiffusiveHeatRate_new_Full << <numTilesMajorClever, threadsPerTileMajorClever >> > (
		pX_half->p_info,
		pX_half->p_izNeigh_vert,
		pX_half->p_szPBCneigh_vert,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_cc,				// We do heat from cc -- seriously?

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
		true,
		p_boolarray2,
		p_boolarray_block,
		false // recalculate everything using pX_target->p_T_minor
		);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate test");
	// To increase the efficiency we want to make a clever 2nd set of major tiles of size 192. Also try 256, 384.

	cudaMemcpy(&tempf64, &(store_heatcond_NTrates[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("%d store_heatcond_NTrates Nn rate %1.10E \n", VERTCHOSEN, tempf64);

	cudaMemset(p_longtemp, 0, sizeof(long)*numTilesMajor);
	kernelCompareForStability_andSetFlag << <numTilesMajor, threadsPerTileMajor >> >
		(
			pX_half->p_info,
			store_heatcond_NTrates,
			NT_addition_rates_d,
			p_longtemp,
			p_boolarray2 // store 1 if it has reversed and amplified
			);
	Call(cudaThreadSynchronize(), "cudaTS CompareStability");

	cudaMemcpy(&tempf64, &(store_heatcond_NTrates[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("%d store_heatcond_NTrates Nn rate %1.10E \n", VERTCHOSEN, tempf64);


	long iReversals = 0;
	cudaMemcpy(p_longtemphost, p_longtemp, sizeof(long)*numTilesMajor, cudaMemcpyDeviceToHost);
	for (iTile = 0; iTile < numTilesMajor; iTile++)
		iReversals += p_longtemphost[iTile];
	if (iReversals > 0)
	{
		printf("%d reversals! \n", iReversals);
		bBackward = true;
	};
	
	if (bBackward == false) {
		printf("midpoint accepted in entirety\n");
		// Are we done? CHECK:
		// We got NT_addition_rates_d
		
	}
	else {
		//	
		//		kernelSetNeighboursBwd << <numTilesMajorClever, threadsPerTileMajorClever >> > (
		//			pX_half->p_info,
		//			pX_half->p_izNeigh_vert,
		//			p_boolarray2);
		//		Call(cudaThreadSynchronize(), "cudaTS kernelSetNeighboursBwd");
		//
				// doesn't work yet.
//		cudaMemcpy(&btemp, &(p_boolarray2[NUMVERTICES*2 + VERTCHOSEN]), sizeof(bool), cudaMemcpyDeviceToHost);
//		SetConsoleTextAttribute(hConsole, 13);
//		printf("\nbool_e[%d] %d \n\n", VERTCHOSEN, (btemp ? 1 : 0));
//		SetConsoleTextAttribute(hConsole, 15);

		cudaMemset(p_boolarray_block, 0, sizeof(bool)*numTilesMajorClever);
		cudaMemset(p_longtemp, 0, sizeof(long)*numTilesMajorClever * 3);
		kernelSetBlockMaskFlag_CountEquations_reset_Tk << <numTilesMajorClever, threadsPerTileMajorClever >> > (
			p_boolarray2,
			p_boolarray_block,
			p_longtemp,
			HEATBASESYST->p_T_minor + BEGINNING_OF_CENTRAL,
			pX_target->p_T_minor + BEGINNING_OF_CENTRAL
			);
		Call(cudaThreadSynchronize(), "cudaTS kernelSetBlockMaskFlag");
		
		iEquations[0] = 0;  iEquations[1] = 0; iEquations[2] = 0;
		cudaMemcpy(p_longtemphost, p_longtemp, sizeof(long)*numTilesMajorClever * 3, cudaMemcpyDeviceToHost);
		for (iTile = 0; iTile < numTilesMajorClever; iTile++)
		{
			iEquations[0] += p_longtemphost[iTile * 3];
			iEquations[1] += p_longtemphost[iTile * 3 + 1];
			iEquations[2] += p_longtemphost[iTile * 3 + 2];
		};
		cudaMemcpy(&btemp, &(p_boolarray2[NUMVERTICES + 22351]), sizeof(bool), cudaMemcpyDeviceToHost);
		SetConsoleTextAttribute(hConsole, 13);
		printf("\nbool[%d] %d \n\n", NUMVERTICES + 22351, (btemp ? 1 : 0));
		SetConsoleTextAttribute(hConsole, 15);
		 
		// IMPORTANT:

		// Block flag can only be used when << < numTilesMajorClever, threadsPerTileMajorClever >> >

		over_iEquations_n = (iEquations[0] > 0) ? (1.0 / (f64)iEquations[0]) : 1.0;
		over_iEquations_i = (iEquations[1] > 0) ? (1.0 / (f64)iEquations[1]) : 1.0;
		over_iEquations_e = (iEquations[2] > 0) ? (1.0 / (f64)iEquations[2]) : 1.0;
		printf("iEquations %d %d %d \n", iEquations[0], iEquations[1], iEquations[2]);

		// Backward Euler step:
	//	 cudaMemcpy(pX_target->p_T_minor + BEGINNING_OF_CENTRAL, 
	//		HEATBASESYST->p_T_minor + BEGINNING_OF_CENTRAL, sizeof(T3)*NUMVERTICES, cudaMemcpyDeviceToDevice);
		// Except we don't necessarily want to have done that.
		// ++ Keep the proper values in T_minor that we get from forward step. ++


		if (iHistory > 0) {

			// Hang on. We are solving here for the system to use for heat flows, is that correct?
			// Yep.

			// Then we can still use these flows along with the whole-length step
			// hsub is not a part of the flow, it's a flux.
			 
			SetConsoleTextAttribute(hConsole, 11);
			RegressionSeedTe(HEATSTEP,
				p_store_T_move1, p_store_T_move2,
				pX_target->p_T_minor + BEGINNING_OF_CENTRAL,
				HEATBASESYST->p_T_minor + BEGINNING_OF_CENTRAL,
				pX_half,
				true // run assuming p_boolarray2 has been set
			);
			SetConsoleTextAttribute(hConsole, 10);
		};
		// We stored result in pX_target->p_T_minor?
		cudaMemcpy(&tempf64, &((pX_target->p_T_minor + BEGINNING_OF_CENTRAL + VERTCHOSEN)->Te), sizeof(f64),
			cudaMemcpyDeviceToHost);
		printf("Contents of Te[%d] : %1.10E\n\n", VERTCHOSEN, tempf64);


		
		// If we want to change back to JRLS then we need to
		// get rid of re-pack as well, remember!


#define JnLS

#ifndef JnLS
		do {
			iSuccess = RunBackwardJLSForHeat(
				HEATBASESYST->p_T_minor + BEGINNING_OF_CENTRAL,
				pX_target->p_T_minor + BEGINNING_OF_CENTRAL,
				HEATSTEP,
				pX_half,
				true // run assuming p_boolarray2 has been set
			);

			if (iSuccess != 0)  RunBackwardForHeat_ConjugateGradient(
				HEATBASESYST->p_T_minor + BEGINNING_OF_CENTRAL,
				pX_target->p_T_minor + BEGINNING_OF_CENTRAL,
				HEATSTEP,
				pX_half,
				true // run assuming p_boolarray2 has been set
			);

			//iSuccess = RunBwdJnLSForHeat(p_Tnk, p_Tn, 
			//	HEATSTEP, pX_half, true,
			//	0, 
			//	p_kappa_n, 
			//	p_nu_i // not used
			//);

		} while (iSuccess != 0);
#endif

#ifdef JnLS
		kernelUnpack << <numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_Tn, p_Ti, p_Te, pX_target->p_T_minor + BEGINNING_OF_CENTRAL); // create T
		Call(cudaThreadSynchronize(), "cudaTS kernelUnpack k+1 A");

		kernelUnpack << <numTilesMajorClever, threadsPerTileMajorClever >> >
			(p_Tnk, p_Tik, p_Tek, HEATBASESYST->p_T_minor + BEGINNING_OF_CENTRAL); // create T
		Call(cudaThreadSynchronize(), "cudaTS kernelUnpack k");

		if (iEquations[0] > 0) {
			//printf("Neutral solve:\n"); // should have a way to skip if iEquations == 0
			do {
			
				iSuccess = RunBwdJnLSForHeat(p_Tnk, p_Tn, 
					HEATSTEP, pX_half, true,
					0, 
					p_kappa_n, 
					p_nu_i // not used
				);

			} while (iSuccess != 0);
			
		};
		if (iEquations[1] > 0) {
			printf("Ion solve:\n");
			do {
				iSuccess = RunBwdJnLSForHeat(p_Tik, p_Ti,
					HEATSTEP, pX_half, true,
					1,
					p_kappa_i,
					p_nu_i);
			} while (iSuccess != 0);
			
		};

		cudaMemcpy(&tempf64, &((pX_target->p_T_minor + BEGINNING_OF_CENTRAL + VERTCHOSEN)->Te), sizeof(f64),
			cudaMemcpyDeviceToHost);
		printf("Contents of Te[%d] : %1.10E\n\n", VERTCHOSEN, tempf64);

		if (iEquations[2] > 0) {
			printf("Electron solve:\n");
			do {
				iSuccess = RunBwdJnLSForHeat(p_Tek, p_Te,
					HEATSTEP, pX_half, true,
					2,
					p_kappa_e,
					p_nu_e);
			} while (iSuccess != 0);
			
		};

//		cudaMemcpy(&tempf64, &((pX_target->p_T_minor + BEGINNING_OF_CENTRAL + VERTCHOSEN)->Te), sizeof(f64),
//			cudaMemcpyDeviceToHost);
//		printf("Contents of Te[%d] : %1.10E\n\n", VERTCHOSEN, tempf64);

		kernelPackupT3 << <numTilesMajorClever, threadsPerTileMajorClever >> >
			(pX_target->p_T_minor + BEGINNING_OF_CENTRAL, p_Tn, p_Ti, p_Te); // create T
		Call(cudaThreadSynchronize(), "cudaTS kernelPack k+1");

//		cudaMemcpy(&tempf64, &((pX_target->p_T_minor + BEGINNING_OF_CENTRAL + VERTCHOSEN)->Te), sizeof(f64),
	//		cudaMemcpyDeviceToHost);
		//printf("Contents of Te[%d] : %1.10E\n\n", VERTCHOSEN, tempf64);


#endif
		if (!DEFAULTSUPPRESSVERBOSITY)
		{
			SetConsoleTextAttribute(hConsole, 14);
			cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
			printf("%d Nn rate %1.10E \n", VERTCHOSEN, tempf64);
			cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NnTn), sizeof(f64), cudaMemcpyDeviceToHost);
			printf("\nNnTn rate [%d] : %1.13E\n", VERTCHOSEN, tempf64);
			cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
			printf("\nNT_addition_rates_d[%d].NeTe %1.14E\n", VERTCHOSEN, tempf64);
			cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
			printf("\nNiTi rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
			SetConsoleTextAttribute(hConsole, 15);
		};

		SetConsoleTextAttribute(hConsole, 15);

		cudaMemcpy(&tempf64, &((pX_target->p_T_minor + BEGINNING_OF_CENTRAL + VERTCHOSEN)->Te), sizeof(f64),
			cudaMemcpyDeviceToHost);
		printf("Contents of Te[%d] : %1.10E\n\n", VERTCHOSEN, tempf64);

		SubtractT3 << <numTilesMajor, threadsPerTileMajor >> >
				(p_store_T_move1,
				pX_target->p_T_minor + BEGINNING_OF_CENTRAL,
				HEATBASESYST->p_T_minor + BEGINNING_OF_CENTRAL);
		Call(cudaThreadSynchronize(), "cudaTS subtractT3");

		iHistory++; // we have now been through this point.

		//debug:
		cudaMemcpy(&tempf64, &((pX_target->p_T_minor + BEGINNING_OF_CENTRAL + VERTCHOSEN)->Te), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("solved Te[%d]: %1.9E\n", VERTCHOSEN, tempf64);
		if (tempf64 < 0.0) {
			printf("4getch");  getch(); getch(); getch(); getch(); PerformCUDA_Revoke(); printf("end");
			while (1) getch();
		}

		// Something to know : we never zero "NT_addition_rates" in the routine.
		// So we need to do it outside.
		
		cudaMemcpy(NT_addition_rates_d_temp, store_heatcond_NTrates, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToDevice);
		cudaMemset(p_boolarray, 0, sizeof(bool)*NUMVERTICES * 2); // initially allow all flows good
		
		// This is the tricky bit --- we shouldn't allow all flows good if we did masked flow longi?
		// Basically for masked points we do NOT want to do any of this.
		// We want to accept the d/dt(NT) that we already went with.
		
		// pX_target->p_T_minor + BEGINNING_OF_CENTRAL
		// & store_heatcond_NTrates
		// contain the relevant data.

		cudaMemcpy(&tempf64, &(store_heatcond_NTrates[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d store_heatcond_NTrates Nn rate %1.10E \n", VERTCHOSEN, tempf64);

		cudaMemcpy(&tempf64, &(NT_addition_rates_d_temp[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d NT_addition_rates_d_temp Nn rate %1.10E \n", VERTCHOSEN, tempf64);

		// We need to basically wipe over NT_addition_rates_d_temp with 0 wherever it's an active cell.

		kernelSelectivelyZeroNTrates << <numTilesMajorClever, threadsPerTileMajorClever >> >(
			NT_addition_rates_d_temp,
			p_boolarray2
		);
		Call(cudaThreadSynchronize(), "cudaTS SelectivelyZeroRates");
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d Nn rate %1.10E \n", VERTCHOSEN, tempf64);

		cudaMemcpy(&tempf64, &(NT_addition_rates_d_temp[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d NT_addition_rates_d_temp Nn rate %1.10E \n", VERTCHOSEN, tempf64);


		iPass = 0;
		do {
			printf("iPass %d :\n", iPass);
			cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
			printf("%d Nn rate %1.10E \n", VERTCHOSEN, tempf64);

			// reset NTrates:
			cudaMemcpy(NT_addition_rates_d, NT_addition_rates_d_temp, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToDevice);
			cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
			printf("%d Nn rate %1.10E \n", VERTCHOSEN, tempf64);

			kernelAccumulateDiffusiveHeatRate_new_Full << <numTilesMajorClever, threadsPerTileMajorClever >> > (
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
				NT_addition_rates_d, // This will increase it!
				pX_half->p_AreaMajor,
				(iPass == 0) ? false : true,
				p_boolarray2,
				p_boolarray_block,
				true // assume p_boolarray2, p_boolarray_blocks have been set : do nothing for already-set values
				);
			Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate pX_half");
			// To increase the efficiency we want to make a clever 2nd set of major tiles of size 192. Also try 256, 384.
	
			cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
			printf("%d Nn rate %1.10E \n", VERTCHOSEN, tempf64);

			cudaMemset(p_bFailed, 0, sizeof(bool)*numTilesMajorClever);
			kernelCreatePutativeT << < numTilesMajorClever, threadsPerTileMajorClever >> > (
				HEATSTEP,
				pX_half->p_info,
				HEATBASESYST->p_T_minor + BEGINNING_OF_CENTRAL,
				//	p_T_upwind_minor_and_putative_T + BEGINNING_OF_CENTRAL, // putative T storage
				pX_half->p_n_major,
				pX_half->p_AreaMajor,
				NT_addition_rates_d,
				p_boolarray, // an array of whether this one requires longi flows --- did it come out T < 0
							 // 2x NMAJOR
				p_bFailed, // did we hit any new negative T to add
				p_boolarray2,
				p_boolarray_block,
				true // only bother with those we are solving for.
				);
			Call(cudaThreadSynchronize(), "cudaTS kernelCreatePutativeT");
		
			cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
			printf("%d Nn rate %1.10E \n", VERTCHOSEN, tempf64);

			bContinue = false;
			cudaMemcpy(p_boolhost, p_bFailed, sizeof(bool)*numTilesMajorClever, cudaMemcpyDeviceToHost);
			int i;
			for (i = 0; ((i < numTilesMajorClever) && (p_boolhost[i] == 0)); i++);
			if (i < numTilesMajorClever) bContinue = true;
			iPass++;
		} while (bContinue);
	}
	if (!DEFAULTSUPPRESSVERBOSITY)
	{

		SetConsoleTextAttribute(hConsole, 14);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d Nn rate %1.10E \n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NnTn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNnTn rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNT_addition_rates_d[%d].NeTe %1.14E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNiTi rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		SetConsoleTextAttribute(hConsole, 15);
	};

	if (bGlobalSaveTGraphs) {
		// Store in Tgraph1, the conductive dT/dt
		DivideNeTe_by_N << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			NT_addition_rates_d,
			pX_half->p_AreaMajor,
			pX_half->p_n_major,
			p_Tgraph[0]
		);
		Call(cudaThreadSynchronize(), "cudaTS DivideNeTe_by_N");

		// Store into temp array:
		cudaMemcpy(NT_addition_rates_d_temp2, NT_addition_rates_d, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	};

	cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
	printf("%d Nn rate %1.10E \n", VERTCHOSEN, tempf64);
	// PUT BACK THE n,T MINOR FOR CENTROIDS THAT WE GOT FROM CALLING FOR SHARD MODEL:

	cudaMemcpy(pX_half->p_n_minor, pX_target->p_n_minor, sizeof(nvals)*NMINOR, cudaMemcpyDeviceToDevice);
	cudaMemcpy(pX_half->p_T_minor, pX_target->p_T_minor, sizeof(T3)*NMINOR, cudaMemcpyDeviceToDevice);

	cudaMemset(p_temp3_1, 0, sizeof(f64_vec3)*numTilesMajor*threadsPerTileMajor);
	cudaMemset(p_temp3_2, 0, sizeof(f64_vec3)*numTilesMajor*threadsPerTileMajor);
	cudaMemset(p_temp3_3, 0, sizeof(f64_vec3)*numTilesMajor*threadsPerTileMajor);

	kernelIonisationRates << <numTilesMajor, threadsPerTileMajor >> >(
		HEATSTEP,
		pX_half->p_info,
		HEATBASESYST->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices
		HEATBASESYST->p_n_major,
		pX_half->p_AreaMajor,
		NT_addition_rates_d,
		p_temp3_1,//p_MAR_temp_major_n,
		p_temp3_2,//p_MAR_temp_major_i,
		p_temp3_3,//p_MAR_temp_major_e,

		HEATBASESYST->p_vie + BEGINNING_OF_CENTRAL,
		HEATBASESYST->p_v_n + BEGINNING_OF_CENTRAL,

		pX_half->p_T_minor + BEGINNING_OF_CENTRAL,
		true
		);
	Call(cudaThreadSynchronize(), "cudaTS Ionisation pXhalf");

	// PERIODIC NEGLECTED:
	Collect_Ntotal_major << <numTilesMajor, threadsPerTileMajor >> >(
		pX_half->p_info,
		pX_half->p_izTri_vert,
		pX_half->p_n_minor,
		pX_half->p_AreaMinor,
		p_temp1, // Ntotal major
		p_temp2  // Nntotal major
		);
	Call(cudaThreadSynchronize(), "cudaTS Gather Ntotal");
	  
	Augment_dNv_minor << <numTilesMinor, threadsPerTileMinor >> >(
		pX_half->p_info,
		pX_half->p_n_minor,
		pX_half->p_tri_corner_index,
		p_temp1,
		p_temp2,
		pX_half->p_AreaMinor,
		p_temp3_1, p_temp3_2, p_temp3_3,
		p_MAR_neut, 
		p_MAR_ion,
		p_MAR_elec);
	Call(cudaThreadSynchronize(), "cudaTS Augment_dNv_minor");

	if (!GlobalSuppressSuccessVerbosity) {
		f64_vec3 tempvec3;
		cudaMemcpy(&tempvec3, &(p_MAR_ion[CHOSEN]),
			sizeof(f64_vec3), cudaMemcpyDeviceToHost);
		printf("\n%d post ionization : pMAR_ion.xy %1.9E %1.9E \n", CHOSEN, tempvec3.x, tempvec3.y);
		cudaMemcpy(&tempvec3, &(p_MAR_neut[CHOSEN]),
			sizeof(f64_vec3), cudaMemcpyDeviceToHost);
		printf("\n%d post ionization : pMAR_neut.xy %1.9E %1.9E \n\n", CHOSEN, tempvec3.x, tempvec3.y);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d Nn rate %1.10E \n", VERTCHOSEN, tempf64);
	};

	if (bGlobalSaveTGraphs) {
		Reversesubtract_vec3 << <numTilesMajor, threadsPerTileMajor >> >
			(p_MAR_elec_ionization_major_stored, p_MAR_elec + BEGINNING_OF_CENTRAL);
		Call(cudaThreadSynchronize(), "cudaTS sss");

		// Pre-store existing:
		cudaMemcpy(p_MAR_ion_visc_major_stored,
			p_MAR_ion + BEGINNING_OF_CENTRAL, sizeof(f64_vec3)*NUMVERTICES, cudaMemcpyDeviceToDevice);
		cudaMemcpy(p_MAR_elec_visc_major_stored,
			p_MAR_elec + BEGINNING_OF_CENTRAL, sizeof(f64_vec3)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	}
	if (bGlobalSaveTGraphs) {
		// Store in Tgraph2, the ionization dT/dt
		DivideNeTeDifference_by_N << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			NT_addition_rates_d_temp2,
			NT_addition_rates_d,
			pX_half->p_AreaMajor,
			pX_half->p_n_major,
			p_Tgraph[1]
			);
		Call(cudaThreadSynchronize(), "cudaTS DivideNeTe_by_N");

		// Store into temp array:
		cudaMemcpy(NT_addition_rates_d_temp2, NT_addition_rates_d, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToDevice);


		DivideMARDifference_get_accel_y << <numTilesMajor, threadsPerTileMajor >> > (
				p_MAR_ion + BEGINNING_OF_CENTRAL,
				p_MAR_elec + BEGINNING_OF_CENTRAL,
			p_MAR_ion_temp_central,
			p_MAR_elec_temp_central,
				this->p_n_minor + BEGINNING_OF_CENTRAL,
				this->p_AreaMinor + BEGINNING_OF_CENTRAL, // we'll look in the minor central cell, this is where MAR applies.
				p_accelgraph[9]
				);
		Call(cudaThreadSynchronize(), "cudaTS DivideMARdiff_get_accel");

		cudaMemcpy(p_MAR_ion_temp_central, p_MAR_ion + BEGINNING_OF_CENTRAL, sizeof(f64_vec3)*NUMVERTICES,
				cudaMemcpyDeviceToDevice);
		cudaMemcpy(p_MAR_elec_temp_central, p_MAR_elec + BEGINNING_OF_CENTRAL, sizeof(f64_vec3)*NUMVERTICES,
				cudaMemcpyDeviceToDevice);
		
	};
	if (!DEFAULTSUPPRESSVERBOSITY)
	{

		SetConsoleTextAttribute(hConsole, 14);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d Nn rate %1.10E \n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NnTn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNnTn rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNT_addition_rates_d[%d].NeTe %1.14E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNiTi rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		SetConsoleTextAttribute(hConsole, 15);
	};

	//
	//cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
	//printf("%d NiTi rate %1.10E \n", VERTCHOSEN, tempf64);
	//cudaMemcpy(&tempf64, &(p_MAR_elec[VERTCHOSEN + BEGINNING_OF_CENTRAL].z), sizeof(f64), cudaMemcpyDeviceToHost);
	//printf("\nMAR_elec.z [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
	//

	kernelReset_v_in_outer_frill_and_outermost << <numTilesMinor, threadsPerTileMinor >> >
		(
			pX_half->p_info,
			pX_half->p_vie,
			pX_half->p_v_n,
			pX_half->p_T_minor,
			pX_half->p_tri_neigh_index,
			pX_half->p_izNeigh_vert
			);
	Call(cudaThreadSynchronize(), "cudaTS resetv");

#ifdef PRECISE_VISCOSITY
	// it is defined. 200620

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
		
	
	::RunBackwardJLSForViscosity(this->p_vie, pX_target->p_vie, Timestep, pX_half, p_storeviscmove, bViscousHistory);
	::RunBackwardR8LSForNeutralViscosity(this->p_v_n, pX_target->p_v_n, Timestep, pX_half);

	Subtract_V4 << <numTilesMinor, threadsPerTileMinor >> >(p_storeviscmove, this->p_vie, pX_half->p_vie);
	Call(cudaThreadSynchronize(), "cudaTS Subtract_V4");
	bViscousHistory = true; // We have now passed through this point. Use this move as regr for both parts of next move.

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
		pX_target->p_v_n,
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

	if (!GlobalSuppressSuccessVerbosity) {
		f64_vec3 tempvec3;
		cudaMemcpy(&tempvec3, &(p_MAR_ion[CHOSEN]),
			sizeof(f64_vec3), cudaMemcpyDeviceToHost);
		printf("\n%d post viscosity : pMAR_ion.xy %1.9E %1.9E \n", CHOSEN, tempvec3.x, tempvec3.y);
		cudaMemcpy(&tempvec3, &(p_MAR_neut[CHOSEN]),
			sizeof(f64_vec3), cudaMemcpyDeviceToHost);
		printf("\n%d post viscosity : pMAR_neut.xy %1.9E %1.9E \n\n", CHOSEN, tempvec3.x, tempvec3.y);
	};


	if (bGlobalSaveTGraphs) {
		// Store in Tgraph3, the viscous dT/dt
		DivideNeTeDifference_by_N << < numTilesMajorClever, threadsPerTileMajorClever >> > (
			NT_addition_rates_d_temp2,
			NT_addition_rates_d,
			pX_half->p_AreaMajor,
			pX_half->p_n_major,
			p_Tgraph[2]
			);
		Call(cudaThreadSynchronize(), "cudaTS DivideNeTe_by_N");

		// Store into temp array:
		cudaMemcpy(NT_addition_rates_d_temp2, NT_addition_rates_d, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToDevice);

		DivideMARDifference_get_accel_y << <numTilesMajor, threadsPerTileMajor >> > (
			p_MAR_ion + BEGINNING_OF_CENTRAL,
			p_MAR_elec + BEGINNING_OF_CENTRAL,
			p_MAR_ion_temp_central,
			p_MAR_elec_temp_central,
			this->p_n_minor + BEGINNING_OF_CENTRAL,
			this->p_AreaMinor + BEGINNING_OF_CENTRAL, // we'll look in the minor central cell, this is where MAR applies.
			p_accelgraph[8] // viscosity y
			);
		Call(cudaThreadSynchronize(), "cudaTS DivideMARdiff_get_accel");

		cudaMemcpy(p_MAR_ion_temp_central, p_MAR_ion + BEGINNING_OF_CENTRAL, sizeof(f64_vec3)*NUMVERTICES,
			cudaMemcpyDeviceToDevice);
		cudaMemcpy(p_MAR_elec_temp_central, p_MAR_elec + BEGINNING_OF_CENTRAL, sizeof(f64_vec3)*NUMVERTICES,
			cudaMemcpyDeviceToDevice);
	};
	// This must be where most runtime cost lies.
	// 2 ways to reduce: reduce frequency to 1e-12, introduce masking.

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
	if (!DEFAULTSUPPRESSVERBOSITY)
	{
		SetConsoleTextAttribute(hConsole, 14);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].Nn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d Nn rate %1.10E \n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NeTe), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("%d NeTe rate %1.10E \n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NnTn), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("\nNnTn rate [%d] : %1.13E\n\n", VERTCHOSEN, tempf64);
		cudaMemcpy(&tempf64, &(NT_addition_rates_d[VERTCHOSEN].NiTi), sizeof(f64), cudaMemcpyDeviceToHost);
		printf("NiTi rate [%d] : %1.13E\n", VERTCHOSEN, tempf64);
		SetConsoleTextAttribute(hConsole, 15);
	};

#endif 

	if (bGlobalSaveTGraphs) {
		Reversesubtract_vec3 << <numTilesMajor, threadsPerTileMajor >> >
			(p_MAR_ion_visc_major_stored, p_MAR_ion + BEGINNING_OF_CENTRAL);
		Call(cudaThreadSynchronize(), "cudaTS sss1");
		Reversesubtract_vec3 << <numTilesMajor, threadsPerTileMajor >> >
			(p_MAR_elec_visc_major_stored, p_MAR_elec + BEGINNING_OF_CENTRAL);
		Call(cudaThreadSynchronize(), "cudaTS sss2");

	}

	if (bGlobalSaveTGraphs == false) {
		kernelAdvanceDensityAndTemperature_noadvectioncompression << <numTilesMajor, threadsPerTileMajor >> > (
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
			pX_target->p_T_minor + BEGINNING_OF_CENTRAL,
			pX_half->p_B + BEGINNING_OF_CENTRAL
			); // do check for T<0
		Call(cudaThreadSynchronize(), "cudaTS Advance_n_and_T 2330");
	} else {
		kernelAdvanceDensityAndTemperature_noadvectioncompression_Copy << <numTilesMajor, threadsPerTileMajor >> > (
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
			pX_target->p_T_minor + BEGINNING_OF_CENTRAL,
			pX_half->p_B + BEGINNING_OF_CENTRAL,
			p_Tgraph[3], // resistive/fric
			p_Tgraph[4], // soak
			p_Tgraph[5],  // dTe/dt total
			p_Tgraph[7]   // dnTe/dt
			); // do check for T<0
		Call(cudaThreadSynchronize(), "cudaTS Advance_n_and_T 2330");
	};
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

	// Fill in full shard-based n_minor for destination system, to match the others:


	kernelCreateShardModelOfDensities_And_SetMajorArea << <numTilesMajor, threadsPerTileMajor >> > (
		pX_target->p_info,
		pX_target->p_n_major,
		pX_target->p_n_minor,
		pX_target->p_izTri_vert,
		pX_target->p_szPBCtri_vert, // could be more intelligent way than storing and copying 3x
		pX_target->p_cc,
		p_n_shards,
		p_n_shards_n,
		pX_target->p_AreaMajor,
		false);
	Call(cudaThreadSynchronize(), "cudaTS CreateShardModel pX_target");

	kernelInferMinorDensitiesFromShardModel << <numTilesMinor, threadsPerTileMinor >> > (
		pX_target->p_info,
		pX_target->p_n_minor,
		p_n_shards,
		p_n_shards_n,
		pX_target->p_tri_corner_index,
		pX_target->p_who_am_I_to_corner,
		p_one_over_n2);// (At the moment just repopulating tri minor n.)
	Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities pX_target");

	if (!DEFAULTSUPPRESSVERBOSITY) {
		printf("DebugNaN pX_target\n");
		DebugNaN(pX_target);
	}
	
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
	 
//	iSubcycles = SUBCYCLES; // 10
//	hsub = Timestep / (real)iSubcycles;
	
	cudaMemcpy(p_AAdot_start, this->p_AAdot, sizeof(AAdot)*NMINOR, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_vie_start, this->p_vie, sizeof(v4)*NMINOR, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_v_n_start, this->p_v_n, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);

	if (bGlobalSaveTGraphs) {
		cudaMemcpy(p_vie_k_stored, this->p_vie + BEGINNING_OF_CENTRAL, sizeof(v4)*NUMVERTICES, cudaMemcpyDeviceToDevice);
	}

	GosubAccelerate(SUBCYCLES,//iSubcycles, 
		Timestep / (real)SUBCYCLES, // hsub
		pX_target, // pX_use
		pX_half // pX_intermediate
	);

	cudaMemcpy(pX_target->p_AAdot, p_AAdot_target, sizeof(AAdot)*NMINOR, cudaMemcpyDeviceToDevice);
	cudaMemcpy(pX_target->p_vie, p_vie_target, sizeof(v4)*NMINOR, cudaMemcpyDeviceToDevice);
	cudaMemcpy(pX_target->p_v_n, p_v_n_target, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);
		
	if (bGlobalSaveTGraphs) {

		MeasureAccelxy_and_JxB_and_soak << <numTilesMajor, threadsPerTileMajor >> >(
			pX_target->p_vie + BEGINNING_OF_CENTRAL,
			this->p_vie + BEGINNING_OF_CENTRAL,
			Timestep,
			p_GradAz + BEGINNING_OF_CENTRAL,
			pX_half->p_n_minor + BEGINNING_OF_CENTRAL,
			pX_half->p_T_minor + BEGINNING_OF_CENTRAL,
			this->p_v_n + BEGINNING_OF_CENTRAL,
			pX_target->p_v_n + BEGINNING_OF_CENTRAL,
			p_accelgraph[0],
			p_accelgraph[1], // accel xy
			p_accelgraph[2],
			p_accelgraph[3], // vxB accel xy
			p_accelgraph[11], // grad_y Az
			p_accelgraph[6]
		);
		Call(cudaThreadSynchronize(), "cudaTS MeasureAccelxy ");
	}
	 
	 
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

	nvals n;
	cudaMemcpy(&n, &(this->p_n_major[VERTCHOSEN]), sizeof(nvals), cudaMemcpyDeviceToHost);
	printf("%d this n %1.12E n_n %1.12E \n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", VERTCHOSEN, n.n, n.n_n);
	cudaMemcpy(&n, &(pX_half->p_n_major[VERTCHOSEN]), sizeof(nvals), cudaMemcpyDeviceToHost);
	printf("%d pX_half n %1.12E n_n %1.12E \n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", VERTCHOSEN, n.n, n.n_n);
	cudaMemcpy(&n, &(pX_target->p_n_major[VERTCHOSEN]), sizeof(nvals), cudaMemcpyDeviceToHost);
	printf("%d pX_target n %1.12E n_n %1.12E \n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", VERTCHOSEN, n.n, n.n_n);


	runs++; 
}  
 
void GosubAccelerate(long iSubcycles, f64 hsub, cuSyst * pX_use, cuSyst * pX_intermediate)
{
	static int iHistory = 0;
	GlobalSuppressSuccessVerbosity = true;

	//char ch;
	//cudaMemcpy(&ch, &(pX_use->p_szPBC_triminor[92250 * 6 + 1]), sizeof(char), cudaMemcpyDeviceToHost);
	//printf("pX_use holds char %d", ch);
	//cudaMemcpy(&ch, &(pX_intermediate->p_szPBC_triminor[92250 * 6 + 1]), sizeof(char), cudaMemcpyDeviceToHost);
	//printf("pX_intermediate holds char %d", ch);
	//getch();
	

	for (int iSubstep = 0; iSubstep < iSubcycles; iSubstep++)
	{
		// I suggest a better alternative not yet tried:
		// .. Advance J first with putative Azdot, given LapAz est at halftime
		// .. Advance (bwd) Az with subcycle steps
		// .. Go again for J,Adot given integral over time of LapAz.
		// ........ but just doing this for now.

		kernelPullAzFromSyst << <numTilesMinor, threadsPerTileMinor >> > (
			p_AAdot_start, // A_k
			p_Az
			);
		Call(cudaThreadSynchronize(), "cudaTS PullAz");

		evaltime += hsub; // t_k+1
		f64 Iz_prescribed_endtime = GetIzPrescribed(evaltime); // APPLIED AT END TIME: we are determining

		kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
			pX_use->p_info, 
			p_Az,
			pX_use->p_izTri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBCtri_vert,
			pX_use->p_szPBC_triminor,
			p_LapAz
			);
		Call(cudaThreadSynchronize(), "cudaTS GetLapMinor aaa2");

		kernelPopulateBackwardOhmsLaw_noadvect << <numTilesMinor, threadsPerTileMinor >> > (
			hsub,
			pX_use->p_info,
			p_MAR_neut, p_MAR_ion, p_MAR_elec,
			pX_intermediate->p_B, // for target it wasn't populated, right? Only used for thermal & nu_effective ?

			p_LapAz,
			p_GradAz,
			p_GradTe,
			pX_use->p_n_minor,  // questionable
			pX_use->p_T_minor,

			p_vie_start,
			p_v_n_start,
			p_AAdot_start,   // dimension these & fill in above.
			pX_intermediate->p_AreaMinor, // NOT POPULATED FOR PXTARGET -- yes it should be

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
		f64 Iz0 = 0.0;
		f64 Sigma_Izz = 0.0;
		f64 Iz_k = 0.0;
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

		f64 neg_Iz_per_triangle = -(Iz_prescribed_endtime) / (f64)numReverseJzTriangles;
		Set_f64_constant(negative_Iz_per_triangle, neg_Iz_per_triangle);
		// Electrons travel from cathode to anode so Jz is down in filament,
		// up around anode.
		printf("Iz0 = %1.14E SigmaIzz %1.14E EzStrength = %1.14E \n", Iz0, Sigma_Izz, EzStrength_);

		if ((EzStrength_ > 1.0e5) || (EzStrength_ < -1.0e6)){
			for (iBlock = 0; iBlock < numTilesMinor; iBlock++)
			{
				printf("Block %d : Iz0 = %1.10E        ~~      ", iBlock, p_Iz0_summands_host[iBlock]);
				if (iBlock % 3 == 0) printf("\n");
			}
			printf("time to stop, press p");
			while (getch() != 'p');
			PerformCUDA_Revoke();
			SendMessage(hWnd, WM_DESTROY, 0, 0);
			exit(2323);
		}

		kernelCreateLinearRelationshipBwd_noadvect << <numTilesMinor, threadsPerTileMinor >> > (
			hsub,
			pX_use->p_info,
			p_OhmsCoeffs,
			p_v0,
			p_LapAz,  // used for cancelling .. 
			pX_use->p_n_minor,
			p_denom_e,
			p_denom_i, p_coeff_of_vez_upon_viz, p_beta_ie_z,

			p_AAdot_start,

			pX_intermediate->p_AreaMinor, // because not populated in PXTARGET
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
			RegressionSeedAz(hsub, p_Az, p_AzNext, p_temp6, stored_Az_move, p_Azdot0, p_gamma, p_LapCoeffself, pX_use);
			// Idea: regress epsilon(Az) on p_temp6, stored_Az_move, Jacobi(stored_Az_move);
			// Update p_AzNext as the result.
			// .
			// Do moves really have a low correlation with each other?
			// Save & analyse correls.
			// .
			// Alternative way: regress on states t_k and t_k-1 rather than difference.
			// Result there?
			// Or do 2 historic states, then Richardson+JR+JJR, then etc.
		}
		else {
			kernelCreateSeedAz << <numTilesMinor, threadsPerTileMinor >> >
				(hsub, p_Az, p_Azdot0, p_gamma, p_LapAz, p_AzNext);
			Call(cudaThreadSynchronize(), "cudaTS createSeed");
		}

		//SolveBackwardAzAdvanceCG(hsub, p_Az, p_Azdot0, p_gamma, 
		//		p_AzNext, p_LapCoeffself, pX_intermediate);

		// Rehabilitate it tmrw


		SolveBackwardAzAdvanceJ3LS(hsub, p_Az, p_Azdot0, p_gamma,
			p_AzNext, p_LapCoeffself, pX_intermediate); // pX_target);
		
		GlobalSuppressSuccessVerbosity = true;

		SubtractVector << <numTilesMinor, threadsPerTileMinor >> >
			(stored_Az_move, p_Az, p_AzNext);
		Call(cudaThreadSynchronize(), "cudaTS subtract");

		iHistory++; // we have now been through this point.

		cudaMemcpy(p_Az, p_AzNext, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);

//		SetConsoleTextAttribute(hConsole, 31);

	//	cudaMemset(p_temp1, 0, sizeof(f64)*numTriTiles);
	//	cudaMemset(p_temp2, 0, sizeof(f64)*numTriTiles);
	//	cudaMemset(p_temp3, 0, sizeof(f64)*numTriTiles);
	//	cudaMemset(p_temp4, 0, sizeof(f64)*numTriTiles);
		kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
			pX_use->p_info,
			p_Az,
			pX_use->p_izTri_vert,
			pX_use->p_izNeigh_TriMinor,
			pX_use->p_szPBCtri_vert,
			pX_use->p_szPBC_triminor,
			p_LapAz// it doesn't really make any difference which syst -- no vertices moving
		);
		Call(cudaThreadSynchronize(), "cudaTS GetLap Az JLS 2");
			
		
	//	cudaMemcpy(p_temphost4, p_temp4, sizeof(f64)*numTriTiles, cudaMemcpyDeviceToHost);
	//	cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTriTiles, cudaMemcpyDeviceToHost);
	//	cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTriTiles, cudaMemcpyDeviceToHost);
	//	cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTriTiles, cudaMemcpyDeviceToHost);

	//	f64 sum = 0.0;
	//	f64 sumVT = 0.0, sumTV = 0.0, sumTT = 0.0;
	//	for (int iTile = 0; iTile < numTriTiles; iTile++) {
	//		sum += p_temphost4[iTile];
	//		sumVT += p_temphost1[iTile];
	//		sumTV += p_temphost2[iTile];
	//		sumTT += p_temphost3[iTile];
	//	};
	//	printf("sum = %1.14E \n", sum);
	//	printf("sumVT = %1.14E sumTV = %1.14E sum TT = %1.14E\n", sumVT, sumTV, sumTT);
	//	getch();


	//	SetConsoleTextAttribute(hConsole, 15);
		// Lap Az is now known, let's say.
		// So we are again going to call PopOhms Backward -- but this time we do not wish to save off stuff
		// except for the v(Ez) relationship.

		kernelPopulateBackwardOhmsLaw_noadvect << <numTilesMinor, threadsPerTileMinor >> > (
			hsub,
			pX_use->p_info,
			p_MAR_neut, p_MAR_ion, p_MAR_elec,
			pX_intermediate->p_B,
			p_LapAz,
			p_GradAz, // THIS WE OUGHT TO TWEEN AT LEAST
			p_GradTe,
			pX_use->p_n_minor,  // this is what is suspect -- dest n
			pX_use->p_T_minor,

			p_vie_start,
			p_v_n_start,
			p_AAdot_start, // not updated...

			pX_intermediate->p_AreaMinor, // pop'd? interp?

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
				EzStrength_, Iz_prescribed_endtime, Iz0, Sigma_Izz);
			while (1) getch();
		}



	//cudaMemset(p_temp1, 0, sizeof(f64)*numTilesMinor); // sum for Jz integral
	//	cudaMemset(p_temp2, 0, sizeof(f64)*numTilesMinor); // sum for Lap Az integral
	//	cudaMemset(p_temp3, 0, sizeof(f64)*numTilesMinor); // sum for Lap Az integral
	//	cudaMemset(p_temp4, 0, sizeof(f64)*numTilesMinor); // sum for Lap Az integral

		kernelCalculateVelocityAndAzdot_noadvect << <numTilesMinor, threadsPerTileMinor >> > (
			hsub,
			pX_use->p_info,
//			pX_use->p_tri_corner_index,
			p_vn0,
			p_v0,
			p_OhmsCoeffs,

			p_AAdot_start,
			//(iSubstep == iSubcycles - 1) ? pX_use->p_n_minor:pX_intermediate->p_n_minor,
			pX_use->p_n_minor, // NOT OKAY FOR IT TO NOT BE SAME n AS USED THROUGHOUT BY OHMS LAW
			pX_intermediate->p_AreaMinor,  // Still because pXuse Area still not populated
					   // We need to go back through, populate AreaMinor before we do all these things.
					   // Are we even going to be advecting points every step?
					   // Maybe make advection its own thing.
			p_LapAz,
			p_AAdot_target,
			p_vie_target,
			p_v_n_target

	//		p_temp1, // Jz-
	//		p_temp2, // Jz+
	//		p_temp3  // LapAz
			);
		Call(cudaThreadSynchronize(), "cudaTS kernelUpdate_v ");

		//// DEBUG:
		//cudaMemcpy(p_temphost1, p_temp1, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		//cudaMemcpy(p_temphost2, p_temp2, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		//cudaMemcpy(p_temphost3, p_temp3, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		////cudaMemcpy(p_temphost4, p_temp4, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
		//f64 integralJz1 = 0.0, integralLapAz = 0.0,
		//	integralJz2 = 0.0;
		//for (int iBlock = 0; iBlock < numTilesMinor; iBlock++) {
		//	integralJz1 += p_temphost1[iBlock];
		//	integralJz2 += p_temphost2[iBlock];
		//	integralLapAz += p_temphost3[iBlock];
		//};
		//printf("Integ Jz domain %1.14E integ Jz reverse %1.14E LapAz %1.14E\n",
		//	integralJz1, integralJz2, integralLapAz);
		//getch();

		kernelAdvanceAzBwdEuler << <numTilesMinor, threadsPerTileMinor >> > (
			hsub,
			p_AAdot_start,
			p_AAdot_target,
			p_ROCAzduetoAdvection, false);
		Call(cudaThreadSynchronize(), "cudaTS kernelAdvanceAzBwdEuler ");

		//kernelKillNeutral_v_OutsideRadius << <numTilesMinor, threadsPerTileMinor >> > (
		//	pX_use->p_info,
		//	p_v_n_target
		//	);
		//Call(cudaThreadSynchronize(), "cudaTS kernelKillNeutral_v_OutsideRadius ");

		// I am curious why vn is silly at the back,, ... but for now just going to kill it off.
		
		if (!DEFAULTSUPPRESSVERBOSITY) {

			cudaMemcpy(&tempf64, &(p_vie_target[VERTCHOSEN + BEGINNING_OF_CENTRAL].vez), sizeof(f64), cudaMemcpyDeviceToHost);
			printf("\nvez p_vie_target [%d] : %1.13E\n", VERTCHOSEN + BEGINNING_OF_CENTRAL, tempf64);
		}
		// Set up next go:
		cudaMemcpy(p_AAdot_start, p_AAdot_target, sizeof(AAdot)*NMINOR, cudaMemcpyDeviceToDevice);
		cudaMemcpy(p_vie_start, p_vie_target, sizeof(v4)*NMINOR, cudaMemcpyDeviceToDevice);
		cudaMemcpy(p_v_n_start, p_v_n_target, sizeof(f64_vec3)*NMINOR, cudaMemcpyDeviceToDevice);
	}; // substeps
	GlobalSuppressSuccessVerbosity = DEFAULTSUPPRESSVERBOSITY;
}


void PerformCUDA_Revoke()
{

	GlobalSuppressSuccessVerbosity = true;
	CallMAC(cudaFree(p_storeviscmove));
	CallMAC(cudaFree(p_temp3_1));
	CallMAC(cudaFree(p_temp3_2));
	CallMAC(cudaFree(p_temp3_3));
	CallMAC(cudaFree(p_regressors));
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
	CallMAC(cudaFree(store_heatcond_NTrates));
	CallMAC(cudaFree(p_store_NTFlux)); 
	CallMAC(cudaFree(sz_who_vert_vert));
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

	CallMAC(cudaFree(p_graphdata1));
	CallMAC(cudaFree(p_graphdata2));
	CallMAC(cudaFree(p_graphdata3));
	CallMAC(cudaFree(p_graphdata4));
	CallMAC(cudaFree(p_graphdata5));
	CallMAC(cudaFree(p_graphdata6));
	for (int i = 0; i < 9; i++)
		CallMAC(cudaFree(p_Tgraph[i]));
	for (int i = 0; i < 12; i++)
		CallMAC(cudaFree(p_accelgraph[i]));
	for (int i = 0; i < 20; i++)
		CallMAC(cudaFree(p_Ohmsgraph[i]));

	CallMAC(cudaFree(p_MAR_ion_temp_central));
	CallMAC(cudaFree(p_MAR_elec_temp_central));

	CallMAC(cudaFree(p_InvertedMatrix_n));
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
	CallMAC(cudaFree(p_vJacobi_n));
	CallMAC(cudaFree(p_d_eps_by_d_beta_i));
	CallMAC(cudaFree(p_d_epsxy_by_d_beta_i));
	CallMAC(cudaFree(p_d_eps_iz_by_d_beta_i));
	CallMAC(cudaFree(p_d_eps_ez_by_d_beta_i));
	CallMAC(cudaFree(p_d_eps_by_d_beta_e));
	CallMAC(cudaFree(p_d_epsxy_by_d_beta_e));
	CallMAC(cudaFree(p_d_eps_iz_by_d_beta_e));
	CallMAC(cudaFree(p_d_eps_ez_by_d_beta_e));
	CallMAC(cudaFree(p_sum_eps_deps_by_dbeta_i));
	CallMAC(cudaFree(p_sum_eps_deps_by_dbeta_e));
	CallMAC(cudaFree(p_sum_depsbydbeta_i_times_i));
	CallMAC(cudaFree(p_sum_depsbydbeta_e_times_e));
	CallMAC(cudaFree(p_sum_depsbydbeta_e_times_i));

	CallMAC(cudaFree(p_epsilon3));
	CallMAC(cudaFree(zero_vec3));
	CallMAC(cudaFree(v3temp));
	CallMAC(cudaFree(p_d_epsilon_by_d_beta_x));
	CallMAC(cudaFree(p_d_epsilon_by_d_beta_y));
	CallMAC(cudaFree(p_d_epsilon_by_d_beta_z));

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

	CallMAC(cudaFree(p_regressors3));
	CallMAC(cudaFree(p_tempvec3));
	CallMAC(cudaFree(p_SS));
	CallMAC(cudaFree(p_epsilon_x));
	CallMAC(cudaFree(p_epsilon_y));
	CallMAC(cudaFree(p_epsilon_z));
	CallMAC(cudaFree(p_stored_move3));
	CallMAC(cudaFree(p_d_eps_by_d_beta_x_));
	CallMAC(cudaFree(p_d_eps_by_d_beta_y_));
	CallMAC(cudaFree(p_d_eps_by_d_beta_z_));

	CallMAC(cudaFree(p_boolarray2));
	CallMAC(cudaFree(p_boolarray_block));
	CallMAC(cudaFree(p_sqrtD_inv_n));
	CallMAC(cudaFree(p_sqrtD_inv_i));
	CallMAC(cudaFree(p_sqrtD_inv_e));
	CallMAC(cudaFree(p_sum_eps_deps_by_dbeta_x8));
	CallMAC(cudaFree(p_eps_against_deps));
	CallMAC(cudaFree(p_sum_product_matrix));

	CallMAC(cudaFree(p_AAdot_start));
	CallMAC(cudaFree(p_AAdot_target));
	CallMAC(cudaFree(p_v_n_start));
	CallMAC(cudaFree(p_vie_start));
	CallMAC(cudaFree(p_v_n_target));
	CallMAC(cudaFree(p_vie_target));

	free(p_sum_eps_deps_by_dbeta_J_x4_host);
	free(p_sum_eps_deps_by_dbeta_R_x4_host);
	free(p_sum_depsbydbeta_8x8_host);
	
	free(p_eps_against_deps_host);
	free(p_sum_product_matrix_host);

	free(p_sum_eps_deps_by_dbeta_x8_host);
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
	free(p_NTrates_host);
	free(p_graphdata1_host);
	free(p_graphdata2_host);
	free(p_graphdata3_host);
	free(p_graphdata4_host);
	free(p_graphdata5_host);
	free(p_graphdata6_host);

	for (int i = 0; i < 9; i++)
		free(p_Tgraph_host[i]);
	for (int i = 0; i < 12; i++)
		free(p_accelgraph_host[i]);
	for (int i = 0; i < 20; i++)
		free(p_Ohmsgraph_host[i]);

	GlobalSuppressSuccessVerbosity = false;
	printf("revoke done\n");

}

void Setup_residual_array()
{
	cuSyst * pX = &cuSyst1; // lazy

	// Find the pre-existing values of LapAz + 4piq/c n(viz-vez) 
	kernelPullAzFromSyst << <numTilesMinor, threadsPerTileMinor >> > (
		pX->p_AAdot,
		p_Az
		);
	Call(cudaThreadSynchronize(), "cudaTS PullAz");

	kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
		pX->p_info, // populated position... not neigh_len apparently
		p_Az,
		pX->p_izTri_vert,
		pX->p_izNeigh_TriMinor,
		pX->p_szPBCtri_vert,
		pX->p_szPBC_triminor,
		p_LapAz
		);
	Call(cudaThreadSynchronize(), "cudaTS GetLapMinor addaa2");

	kernelPopulateResiduals << <numTilesMinor, threadsPerTileMinor >> > (
		p_LapAz,
		pX->p_n_minor, pX->p_vie, // is this the n for which the relationship holds for verts? *************
		p_Residuals
		);
	Call(cudaThreadSynchronize(), "cudaTS PopulateResiduals");

}

void Zap_the_back()
{
	kernelResetNeutralDensityOutsideRadius << <numTilesMinor, threadsPerTileMinor >> >
		(
			cuSyst1.p_info,
			cuSyst1.p_n_major,
			cuSyst1.p_n_minor
			);
	Call(cudaThreadSynchronize(), "cudaTS kernelResetDensityOutsideRadius");

	kernelResetNeutralDensityOutsideRadius << <numTilesMinor, threadsPerTileMinor >> >
		(
			cuSyst2.p_info,
			cuSyst2.p_n_major,
			cuSyst2.p_n_minor
			);
	Call(cudaThreadSynchronize(), "cudaTS kernelResetDensityOutsideRadius");

	kernelResetNeutralDensityOutsideRadius << <numTilesMinor, threadsPerTileMinor >> >
		(
			cuSyst3.p_info,
			cuSyst3.p_n_major,
			cuSyst3.p_n_minor
			);
	Call(cudaThreadSynchronize(), "cudaTS kernelResetDensityOutsideRadius");

}

void Go_visit_the_other_file()
{

	f64 LapAz, viz, vez, n, coeffself, Az;
	int iRepeat;
	//f64 epsilon[NMINOR], p_regressor[NMINOR];

	memset(p_temphost2, 0, sizeof(f64)*NMINOR); // epsilon
	memset(p_temphost1, 0, sizeof(f64)*NMINOR); // regressor
//
//	kernelSetZero << <numTriTiles, threadsPerTileMinor >> > (
//		p_LapCoeffself
//		);
//	Call(cudaThreadSynchronize(), "cudaTS setzero");

	kernelGetLapCoeffs_and_min << <numTriTiles, threadsPerTileMinor >> > (
		cuSyst1.p_info,
		cuSyst1.p_izTri_vert,
		cuSyst1.p_izNeigh_TriMinor,
		cuSyst1.p_szPBCtri_vert,
		cuSyst1.p_szPBC_triminor,
		p_LapCoeffself,
		p_temp1, // collect min
		p_longtemp
		);
	Call(cudaThreadSynchronize(), "cudaTS GetLapCoeffs x");
	// Illegal memory access encountered?!
	// But this stuff should be dimensioned.
	// A bug in what it does, not my fault, by the looks.
	long iIteration = 0;
	bool bContinue;
	
	// 1. Calculate Lap Az and coeffself Lap Az; including at our few points.

	kernelPullAzFromSyst << <numTilesMinor, threadsPerTileMinor >> > (
		cuSyst1.p_AAdot,
		p_Az
		);
	Call(cudaThreadSynchronize(), "cudaTS PullAz");

	nvals nvals1, nvals2, nvals3;
	v4 v1, v2, v3;
	f64 resid, resid1, resid2, resid3;
	f64 LapAz1, LapAz2, LapAz3;

	// Now define p_temphost4 as the constant part of eps and p_temphost5 as "resid"

	// 2. For each of our points bring Lap Az, Jz and coeffself to CPU
	for (long i = 0; i < BEGINNING_OF_CENTRAL; i++)
	{
		if (flaglist[i]) {
			cudaMemcpy(&viz, &(cuSyst1.p_vie[i].viz), sizeof(f64), cudaMemcpyDeviceToHost);
			cudaMemcpy(&vez, &(cuSyst1.p_vie[i].vez), sizeof(f64), cudaMemcpyDeviceToHost);
			cudaMemcpy(&n, &(cuSyst1.p_n_minor[i].n), sizeof(f64), cudaMemcpyDeviceToHost);

			LONG3 cornerindex;
			cudaMemcpy(&cornerindex, &(cuSyst1.p_tri_corner_index[i]), sizeof(LONG3), cudaMemcpyDeviceToHost);

			cudaMemcpy(&resid1, &(p_Residuals[cornerindex.i1]), sizeof(f64), cudaMemcpyDeviceToHost);
			cudaMemcpy(&resid2, &(p_Residuals[cornerindex.i2]), sizeof(f64), cudaMemcpyDeviceToHost);
			cudaMemcpy(&resid3, &(p_Residuals[cornerindex.i3]), sizeof(f64), cudaMemcpyDeviceToHost);
			// cudaMemcpy(&resid, &(p_Residuals[i]), sizeof(f64), cudaMemcpyDeviceToHost);

			resid = 0.33333333*(resid1 + resid2 + resid3);
			p_temphost5[i] = 0.01*fabs(resid); // thresh

			// What is average of LapAz + 4piq/c n(viz-vez) ?
			// eps = -LapAz - 4pi/cJ
			// so aim for LapAz = - 4pi/cJ - eps

			p_temphost4[i] = -FOUR_PI_Q_OVER_C_*n*(viz - vez) - resid;
			
		};
	};
	
	f64 beta;
	do {
		bContinue = false;
		
		kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
			cuSyst1.p_info, // populated position... not neigh_len apparently
			p_Az,
			cuSyst1.p_izTri_vert,
			cuSyst1.p_izNeigh_TriMinor,
			cuSyst1.p_szPBCtri_vert,
			cuSyst1.p_szPBC_triminor,
			p_LapAz
			);
		Call(cudaThreadSynchronize(), "cudaTS GetLapMinor addaa2");
		
		// 2. For each of our points bring Lap Az, Jz and coeffself to CPU
		for (long i = 0; i < BEGINNING_OF_CENTRAL; i++)
		{
			if (flaglist[i]) {
				cudaMemcpy(&LapAz, &(p_LapAz[i]), sizeof(f64), cudaMemcpyDeviceToHost);
				cudaMemcpy(&coeffself, &(p_LapCoeffself[i]), sizeof(f64), cudaMemcpyDeviceToHost);

				// 3. For each of our points, adjust Az per Jacobi:
				//printf("%d Az %1.11E LapAz %1.11E 4pi/c Jz %1.11E coeffself %1.9E resid %1.9E resid123 %1.9E %1.9E %1.9E ", i, Az, LapAz, FOUR_PI_Q_OVER_C_*n*(viz - vez),
				//	coeffself, resid, resid1, resid2, resid3);

				p_temphost2[i] = p_temphost4[i] - LapAz; // epsilon
				if (fabs(p_temphost2[i]) > p_temphost5[i]) bContinue = true;
				p_temphost1[i] = (p_temphost4[i] - LapAz) / coeffself; // Jacobi move
			};
		};

		cudaMemcpy(p_temp1, p_temphost1, sizeof(f64), cudaMemcpyHostToDevice);
		cudaMemset(p_temp2, 0, sizeof(f64)*NMINOR);
		kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> > (
				cuSyst1.p_info, // populated position... not neigh_len apparently
				p_temp1,
				cuSyst1.p_izTri_vert,
				cuSyst1.p_izNeigh_TriMinor,
				cuSyst1.p_szPBCtri_vert,
				cuSyst1.p_szPBC_triminor,
				p_temp2
				);
		Call(cudaThreadSynchronize(), "cudaTS GetLapMinor addaaa2");

		cudaMemcpy(p_temphost3, p_temp2, sizeof(f64), cudaMemcpyDeviceToHost);

		f64 sum_depsbydbeta_sq = 0.0;
		f64 sum_eps_depsbydbeta = 0.0;
		f64 sum_eps_eps = 0.0;
		for (long i = 0; i < BEGINNING_OF_CENTRAL; i++)
		{ 
			if (flaglist[i]) {
				f64 d_eps_by_d_beta = -p_temphost3[i];

				sum_depsbydbeta_sq += d_eps_by_d_beta*d_eps_by_d_beta;
				sum_eps_depsbydbeta += p_temphost2[i] * d_eps_by_d_beta;
				sum_eps_eps += p_temphost2[i] * p_temphost2[i];
			};
		};

		beta = -sum_eps_depsbydbeta / sum_depsbydbeta_sq;

		for (long i = 0; i < BEGINNING_OF_CENTRAL; i++)
		{
			if (flaglist[i]) {
				cudaMemcpy(&Az, &(p_Az[i]), sizeof(f64), cudaMemcpyDeviceToHost);

				Az += beta*p_temphost1[i];

				cudaMemcpy(&(p_Az[i]), &Az, sizeof(f64), cudaMemcpyHostToDevice);
				// Note that we didn't keep track of which system's which, so we need to set it in all of them.				
			};
		};
		iIteration++;
		printf("iteration %d sum_eps_eps %1.10E beta %1.10E\n", iIteration, sum_eps_eps, beta);
	} while (bContinue);

	for (long i = 0; i < BEGINNING_OF_CENTRAL; i++)
	{
		if (flaglist[i]) {
			cudaMemcpy(&Az, &(p_Az[i]), sizeof(f64), cudaMemcpyDeviceToHost);
			cudaMemcpy(&(cuSyst1.p_AAdot[i].Az), &Az, sizeof(f64), cudaMemcpyHostToDevice);
			cudaMemcpy(&(cuSyst2.p_AAdot[i].Az), &Az, sizeof(f64), cudaMemcpyHostToDevice);
			cudaMemcpy(&(cuSyst3.p_AAdot[i].Az), &Az, sizeof(f64), cudaMemcpyHostToDevice);
			// Note that we didn't keep track of which system's which, so we need to set it in all of them.
		};
	};
	
	printf("\n\nRecalc Ampere: underrelaxation Jacobi iterations: %d\n\n\n", iIteration);
	
	Beep(750, 150);

	// Problem: by default, __constant__ variables have file scope. Need special
	// compiler settings to do relocatable device code.
}

#include "kernel.cu"
#include "little_kernels.cu"

// There must be a better way.
