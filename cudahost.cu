
// Version 0.1:

// First draft, getting it to compile.

#pragma once


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

__constant__ f64 negative_Iz_per_triangle; // -Iz_prescribed / (real)(numEndZCurrentTriangles - numStartZCurrentTriangles)

__constant__ long numStartZCurrentTriangles, numEndZCurrentTriangles;


#define CallMAC(cudaStatus) Call(cudaStatus, #cudaStatus )   
// { Call(cudaStatus, "cudaStatus") } ?
extern real FRILL_CENTROID_OUTER_RADIUS, FRILL_CENTROID_INNER_RADIUS;

cuSyst cuSyst1, cuSyst2, cuSyst3;

// Given our restructure, we are going to need to dimension
// a cuSyst type thing that lives on the host??
// Not necessarily and not an easy way to write.
// This time around find another way to populate.
// We do need a temporary such object in the routine where we populate the device one.
// I guess as before we want an InvokeHost routine because of that.
__device__ real * p_summands, *p_Iz0_summands, *p_Iz0_initial, *p_scratch_d;
f64 * p_summands_host, *p_Iz0_summands_host, *p_Iz0_initial_host;
__device__ f64 * p_temp1, *p_temp2, *p_temp3, *p_temp4, *p_denom_i, *p_denom_e, *p_coeff_of_vez_upon_viz, *p_beta_ie_z;
f64 * p_temphost1, *p_temphost2, *p_temphost3, *p_temphost4, *p_temphost5, *p_temphost6;
f64_vec2 * p_GradTe_host, *p_GradAz_host;
f64_vec3 * p_B_host, *p_MAR_ion_host, *p_MAR_elec_host, *p_MAR_ion_compare, *p_MAR_elec_compare,
*p_MAR_neut_host,*p_MAR_neut_compare;
__device__ nn *p_nn_ionrec_minor;
__device__ OhmsCoeffs * p_OhmsCoeffs;
__device__ f64 * p_Iz0, *p_sigma_Izz;
__device__ f64_vec3 * p_vn0;
__device__ v4 * p_v0;
__device__ f64_vec3 * p_MAR_neut, *p_MAR_ion, *p_MAR_elec;
__device__ f64 * p_Az, *p_LapAz, *p_LapCoeffself, *p_Azdot0, *p_gamma, *p_LapJacobi,
*p_Jacobi_x, *p_epsilon, *p_LapAzNext,
*p_Integrated_div_v_overall,
*p_Div_v_neut, *p_Div_v, *p_Div_v_overall, *p_ROCAzdotduetoAdvection,
*p_ROCAzduetoAdvection, *p_AzNext;
__device__ species3 *p_nu_major;
__device__ f64_vec2 * p_GradAz, *p_GradTe;
__device__ ShardModel *p_n_shards, *p_n_shards_n;
__device__ NTrates *NT_addition_rates_d;
long numReverseJzTriangles;
__device__ f64 *p_sum_eps_deps_by_dbeta, *p_sum_depsbydbeta_sq, *p_sum_eps_eps;
f64  *p_sum_eps_deps_by_dbeta_host, *p_sum_depsbydbeta_sq_host, *p_sum_eps_eps_host;
__device__ char * p_was_vertex_rotated, *p_triPBClistaffected;

f64 * temp_array_host;

//f64 Tri_n_n_lists[NMINOR][6],Tri_n_lists[NMINOR][6];
// Not clear if I ended up using Tri_n_n_lists - but it was a workable way if not.

long * address;
f64 * f64address;
size_t uFree, uTotal;
extern real evaltime;

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
	real Iz = -PEAKCURRENT_STATCOULOMB * sin((t + ZCURRENTBASETIME) * PIOVERPEAKTIME);
	//printf("\nGetIzPrescribed : t + ZCURRENTBASETIME = %1.5E : %1.12E\n", t + ZCURRENTBASETIME, Iz);
	return Iz;
}

void PerformCUDA_Invoke_Populate(
	cuSyst * pX_host, // populate in calling routine...
	long numVerts,
	f64 InnermostFrillCentroidRadius,
	f64 OutermostFrillCentroidRadius,
	long numStartZCurrentTriangles_,
	long numEndZCurrentTriangles_
)
{
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

	CallMAC(cudaMalloc((void **)&p_v0, NMINOR * sizeof(v4)));
	CallMAC(cudaMalloc((void **)&p_vn0, NMINOR * sizeof(f64_vec3)));
	CallMAC(cudaMalloc((void **)&p_sigma_Izz, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Iz0, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_OhmsCoeffs, NMINOR * sizeof(OhmsCoeffs)));

	CallMAC(cudaMalloc((void **)&p_sum_eps_deps_by_dbeta, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_depsbydbeta_sq, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_sum_eps_eps, numTilesMinor * sizeof(f64)));

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

	CallMAC(cudaMalloc((void **)&p_n_shards, NUMVERTICES * sizeof(ShardModel)));
	CallMAC(cudaMalloc((void **)&p_n_shards_n, NUMVERTICES * sizeof(ShardModel)));
	CallMAC(cudaMalloc((void **)&NT_addition_rates_d, NMINOR * sizeof(NTrates)));

	CallMAC(cudaMalloc((void **)&p_coeff_of_vez_upon_viz, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_beta_ie_z, NMINOR * sizeof(f64)));
	
	CallMAC(cudaMalloc((void **)&p_temp1, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_temp2, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_temp3, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_temp4, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_denom_i, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_denom_e, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_summands, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Iz0_summands, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Iz0_initial, numTilesMinor * sizeof(f64)));
	
	p_GradTe_host = (f64_vec2 *)malloc(NMINOR * sizeof(f64_vec2));
	p_GradAz_host = (f64_vec2 *)malloc(NMINOR * sizeof(f64_vec2));
	p_B_host = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));
	p_MAR_ion_host = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));
	p_MAR_ion_compare = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));
	p_MAR_elec_host = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));
	p_MAR_elec_compare = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));
	p_MAR_neut_host = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));
	p_MAR_neut_compare = (f64_vec3 *)malloc(NMINOR * sizeof(f64_vec3));

	p_temphost1 = (f64 *)malloc(NMINOR * sizeof(f64)); // changed for debugging
	p_temphost2 = (f64 *)malloc(NMINOR * sizeof(f64)); // changed for debugging
	p_temphost3 = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_temphost4 = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_temphost5 = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_temphost6 = (f64 *)malloc(numTilesMinor * sizeof(f64));
	if (p_temphost6 == 0) { printf("p6 == 0"); }
	else { printf("p6 != 0"); };
	temp_array_host = (f64 *)malloc(NMINOR * sizeof(f64));

	p_summands_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_Iz0_summands_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_Iz0_initial_host = (f64 *)malloc(numTilesMinor * sizeof(f64));

	p_sum_eps_deps_by_dbeta_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_sum_depsbydbeta_sq_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_sum_eps_eps_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	// Cannot see that I have ever yet put in anywhere to free this memory.

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
	cuSyst * pX1, *pX2, *pX_half, *pXtemp;
	pX1 = &cuSyst1;
	pX2 = &cuSyst3;
	pX_half = &cuSyst2;

	long iSubstep;

	// Ultimately this 10 steps .. so 1e-11? .. can be 1 advective step.

	for (iSubstep = 0; iSubstep < 10; iSubstep++)
	{
		pX1->PerformCUDA_Advance(pX2, pX_half);

		pXtemp = pX1;
		pX1 = pX2;
		pX2 = pXtemp;
	};
	// After an even number of goes, pX1 = &cuSyst1 and this is where we ended up.

	pX1->SendToHost(*pX_host);

	// It's then up to the caller to populate TriMesh from pX_host.
}

void PerformCUDA_RunStepsAndReturnSystem_Debug(cuSyst * pcuSyst_host, cuSyst * p_cuSyst_compare, TriMesh * pTriMesh, TriMesh * pTriMeshhalf,
	TriMesh * pDestMesh)
{
	cuSyst * pX1, *pX2, *pX_half, *pXtemp;
	pX1 = &cuSyst1;
	pX2 = &cuSyst3;
	pX_half = &cuSyst2;

	TriMesh * pTriMeshtemp;

	long iSubstep;

	// Ultimately this 10 steps .. so 1e-11? .. can be 1 advective step.

	for (iSubstep = 0; iSubstep < 10; iSubstep++)
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

	pX1->SendToHost(*pcuSyst_host);

	// It's then up to the caller to populate TriMesh from pX_host.
}

void cuSyst::PerformCUDA_Advance_Debug(const cuSyst * pX_target, const cuSyst * pX_half,
	const cuSyst * p_cuSyst_host, cuSyst * p_cuSyst_compare, TriMesh * pTriMesh, TriMesh * pTriMeshhalf,
	TriMesh * pDestMesh)
{
	long iMinor, iSubstep;
	f64 Iz_prescribed;
	static long runs = 0;
	float elapsedTime;

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

	// inauspicious start: overall v has to be split into 2 routines
	kernelCalculateOverallVelocitiesVertices << <numTilesMajor, threadsPerTileMajor >> > (
		this->p_info + BEGINNING_OF_CENTRAL,
		this->p_vie + BEGINNING_OF_CENTRAL,
		this->p_v_n + BEGINNING_OF_CENTRAL,
		this->p_n_major,
		this->p_v_overall_minor + BEGINNING_OF_CENTRAL
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
	
	kernelAdvectPositions << <numTilesMinor, threadsPerTileMinor >> >(
		0.5*TIMESTEP,
		this->p_info,
		pX_half->p_info,
		this->p_v_overall_minor
		);
	Call(cudaThreadSynchronize(), "cudaTS AdvectPositions_CopyTris");
	
	// ----------------------------------------------------------------
	memset(pTriMeshhalf->pData, 0, sizeof(plasma_data)*NMINOR);
	pTriMesh->AdvectPositions_CopyTris(0.5*TIMESTEP, pTriMeshhalf, p_v);
	
	pTriMeshhalf->EnsureAnticlockwiseTriangleCornerSequences_SetupTriMinorNeighboursLists();
	pTriMeshhalf->Average_n_T_to_tris_and_calc_centroids_and_minorpos();
	p_cuSyst_compare->PopulateFromTriMesh(pTriMeshhalf); // Calls Average_n_T_to tris on pTriMeshhalf; needs vertex pos in pData set up before it tries determining n in CROSSING_INS

	cudaMemcpy(p_cuSyst_host->p_info, pX_half->p_info, NMINOR * sizeof(structural), cudaMemcpyDeviceToHost);
	Compare_structural(p_cuSyst_host->p_info, p_cuSyst_compare->p_info, NMINOR);

	printf("compared structural info.\n");

	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> >(
		this->p_n_minor,
		this->p_n_major,
		this->p_T_minor,
		this->p_info,
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

	cudaMemcpy(p_cuSyst_host->p_info, this->p_info, NMINOR * sizeof(structural), cudaMemcpyDeviceToHost);
	Compare_structural(p_cuSyst_host->p_info, p_cuSyst_compare->p_info, NMINOR);
	printf("compared structural info.\n");
	
	getch();

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
		this->p_n_upwind_minor);
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

	kernelAccumulateAdvectiveMassHeatRate << <numTilesMajor, threadsPerTileMajor >> >(
		0.5*TIMESTEP, // why it appears here?
		this->p_info,
		this->p_izTri_vert,
		this->p_szPBCtri_vert,

		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL,

		this->p_n_upwind_minor,
		this->p_vie,
		this->p_v_n,
		this->p_v_overall_minor,
		this->p_T_minor, // may or may not overlap source: don't we only use from tris? so not overlap

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
	//

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
		this->p_v_overall_minor,
		p_MAR_neut
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelNeutral_pressure_and_momflux");

	printf("\nResults of AccumulateAdvectiveMassHeatRate:\n");
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

	kernelAccumulateDiffusiveHeatRateAndCalcIonisation << <numTilesMajorClever, threadsPerTileMajorClever >> >(
		0.5*TIMESTEP,
		this->p_info + BEGINNING_OF_CENTRAL,
		this->p_izNeigh_vert,
		this->p_szPBCneigh_vert,
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices
		this->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
		p_nu_major,
		NT_addition_rates_d,
		this->p_AreaMajor);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate");
	// To increase the efficiency we want to make a clever 2nd set of major tiles of size 192. Also try 256, 384.

	cudaMemcpy(p_NT_addition_rates, NT_addition_rates_d, sizeof(NTrates)*NUMVERTICES, cudaMemcpyDeviceToHost);

	pTriMesh->AccumulateDiffusiveHeatRateAndCalcIonisation(0.5*TIMESTEP, NTadditionrates); // Wants minor n,T and B

	printf("\nResults of AccumulateDiffusiveHeatRateAndCalcIonisation:\n");
	Compare_NTrates(p_NT_addition_rates, NTadditionrates);
	

	// See difference at 9th place for NiTi but it's small figures compared with NeTe which is at 11th place.
	// =============================================================================
	// HMMM
	
	kernelAdvanceDensityAndTemperature << <numTilesMajor, threadsPerTileMajor >> >(
		0.5*TIMESTEP,
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
		
	pTriMesh->AdvanceDensityAndTemperature(0.5*TIMESTEP, pTriMesh, pTriMeshhalf, NTadditionrates);
	 
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
	printf("p_cuSyst_host->p_T_minor[23187] %1.9E p_cuSyst_compare->p_T_minor[23187] %1.9E \n",
		p_cuSyst_host->p_T_minor[23187].Te, p_cuSyst_compare->p_T_minor[23187].Te);
	
	getch();

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

	Iz_prescribed = GetIzPrescribed(evaltime + 0.5*TIMESTEP); // because we are setting pX_half->v

	f64 neg_Iz_per_triangle = -Iz_prescribed / (f64)numReverseJzTriangles;
	Set_f64_constant(negative_Iz_per_triangle, neg_Iz_per_triangle);
	
	// Get suitable v to use for resistive heating on main step:
	kernelPopulateOhmsLaw << <numTilesMinor, threadsPerTileMinor >> >(
		0.5*TIMESTEP,

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

	// Now calculate Ez_strength to achieve Iz_prescribed:
	Iz_prescribed = GetIzPrescribed(evaltime); 
		// LOOK AT THIS CHANGE RIGHT HERE
		// NOT YET REPLICATED IN PerformCUDA_Advance
	// Don't know which was right.
	long iTile;

	memset(sigma_tiles, 0, sizeof(f64)*numTilesMinor);
	// Get suitable v to use for resistive heating:
	pTriMesh->Accelerate2018(0.5*TIMESTEP, pTriMesh, pTriMeshhalf,
		evaltime, false, true); // current is attained at start of step [ ie Iz == 0 on 1st step ]
	
	f64 Iz0 = 0.0;
	f64 sigma_Izz = 0.0;
	for (iTile = 0; iTile < numTilesMinor; iTile++)
	{
		Iz0 += p_Iz0_summands_host[iTile];
		sigma_Izz += p_summands_host[iTile];

	//	printf("%d: %1.12E %1.12E | %1.12E  diff %1.4E\n ", iTile, p_Iz0_summands_host[iTile],
	//		p_summands_host[iTile], sigma_tiles[iTile], p_summands_host[iTile] - sigma_tiles[iTile]);
	};

	f64 Ez_strength_ = (Iz_prescribed - Iz0) / sigma_Izz;
	Set_f64_constant(Ez_strength, Ez_strength_);

	printf("GPU: Iz_prescribed %1.14E Iz0 %1.14E sigma_Izz %1.14E \n",
		Iz_prescribed, Iz0, sigma_Izz);
	printf("Ez_strength (GPU) %1.14E \n", Ez_strength_);


	// Update velocities and Azdot:
	kernelCalculateVelocityAndAzdot << <numTilesMinor, threadsPerTileMinor >> >(
		0.5*TIMESTEP,
		p_vn0,
		p_v0,
		p_OhmsCoeffs,
		pX_target->p_AAdot,
		pX_half->p_n_minor,

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
	// ))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
	// 
	// Now do the n,T,x advance to pDestmesh:
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	kernelCalculateOverallVelocitiesVertices << <numTilesMajor, threadsPerTileMajor >> >(
		pX_half->p_info + BEGINNING_OF_CENTRAL,
		pX_half->p_vie + BEGINNING_OF_CENTRAL,
		pX_half->p_v_n + BEGINNING_OF_CENTRAL,
		pX_half->p_n_major,
		pX_half->p_v_overall_minor + BEGINNING_OF_CENTRAL
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

	kernelAdvectPositions << <numTilesMinor, threadsPerTileMinor >> >(
		TIMESTEP,
		this->p_info,
		pX_target->p_info,
		pX_half->p_v_overall_minor // WHY WE WERE NOT USING FROM pX_half?
		);
	Call(cudaThreadSynchronize(), "cudaTS AdvectPositions_CopyTris");
	
	pTriMeshhalf->CalculateOverallVelocities(p_v); // vertices first, then average to tris
	memset(pDestMesh->pData, 0, sizeof(plasma_data)*NMINOR);
	pTriMesh->AdvectPositions_CopyTris(TIMESTEP, pDestMesh, p_v);

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
		pX_half->p_n_upwind_minor);
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
		TIMESTEP,
		pX_half->p_info,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,

		pX_half->p_n_major,
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL,

		pX_half->p_n_upwind_minor,
		pX_half->p_vie,
		pX_half->p_v_n,
		pX_half->p_v_overall_minor,
		pX_half->p_T_minor, // may or may not overlap source: don't we only use from tris? so not overlap

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

	printf("\nDone Advective calc \n\n");


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
		pX_half->p_v_overall_minor,
		p_MAR_neut
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelNeutral_pressure_and_momflux pX_half");

	//pX_half->Add_ViscousMomentumFluxRates(AdditionalMomRates);

	memset(AdditionalMomRates, 0, sizeof(three_vec3)*NMINOR);
	pTriMesh->AntiAdvectAzAndAdvance(0.5*h, pTriMesh, GradAz, pTriMeshhalf);
	// ARE WE MISSING THAT FROM GPU ADVANCE CODE? NOT SURE
	pTriMeshhalf->Create_momflux_integral_grad_nT_and_gradA_LapA_CurlA_on_minors(p_v, AdditionalMomRates);

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

	kernelAccumulateDiffusiveHeatRateAndCalcIonisation << <numTilesMajorClever, threadsPerTileMajorClever >> >(
		TIMESTEP,
		pX_half->p_info + BEGINNING_OF_CENTRAL,
		pX_half->p_izNeigh_vert,
		pX_half->p_szPBCneigh_vert,
		pX_half->p_n_major,
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices
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
		TIMESTEP,
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
		pX_half->p_AreaMajor,

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

	printf("Compare T pX_target vs pDestMesh vertices:\n");
	cudaMemcpy(p_cuSyst_host->p_T_minor + BEGINNING_OF_CENTRAL, pX_target->p_T_minor + BEGINNING_OF_CENTRAL, sizeof(T3)*NUMVERTICES, cudaMemcpyDeviceToHost);
	Compare_T3(p_cuSyst_host->p_T_minor + BEGINNING_OF_CENTRAL, p_cuSyst_compare->p_T_minor + BEGINNING_OF_CENTRAL, NUMVERTICES);
	 

	pDestMesh->Average_n_T_to_tris_and_calc_centroids_and_minorpos(); // UPGRADE TO 2ND DEGREE

	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> >(
		pX_target->p_n_minor,
		pX_target->p_n_major,
		pX_target->p_T_minor,
		pX_target->p_info,
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

	printf("================================\n");
	getch();
	getch();


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

	getch();

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
	do {
		o = getch();
	} while (o != 'o');

	// The rest should be relatively straightforward, keeping it simple.
	
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

		for (iSubstep = 0; iSubstep < SUBCYCLES; iSubstep++)
		{
			SetConsoleTextAttribute(hConsole, 14);
			printf("\n\n#########################\n");
			printf(    "####### SUBSTEP %d #######\n\n",iSubstep);
			SetConsoleTextAttribute(hConsole, 15);

			evaltime += 0.5*SUBSTEP;
			kernelInterpolateVarsAndPositions << <numTilesMinor, threadsPerTileMinor >> >(
				(evaltime - starttime) / TIMESTEP,
				this->p_info,
				pX_target->p_info,
				this->p_n_minor,
				pX_target->p_n_minor,
				this->p_T_minor,
				pX_target->p_T_minor,
				this->p_B,
				pX_target->p_B,
				pX_half->p_info,
				pX_half->p_n_minor,
				pX_half->p_T_minor,
				pX_half->p_B
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
				p_LapAz
				);
			Call(cudaThreadSynchronize(), "cudaTS GetLapMinor aaa");
			
			// Looks like Az is never pulled on CPU ! Sort that!!

			pTriMesh->InterpolateVarsAndPositions(pTriMeshhalf, pDestMesh, (evaltime - starttime) / TIMESTEP);
			//pHalfMesh->GetLapFromCoeffs(Az_array, LapAzArray);

			pTriMeshhalf->GetLap(Az_array, LapAzArray);

			SetConsoleTextAttribute(hConsole, 15);
			printf("Lap: substep %d \n", iSubstep);

			cudaMemcpy(p_temphost1, p_LapAz, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			Compare_f64(p_temphost1, LapAzArray, NMINOR);
			printf("Az: substep %d \n", iSubstep);

			cudaMemcpy(p_temphost2, p_Az, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			Compare_f64(p_temphost2, Az_array, NMINOR);

			//fprintf(fp, "LapAz CPU GPU %d %1.14E %1.14E ",
			//	CHOSEN, LapAzArray[CHOSEN], p_temphost1[CHOSEN]);
			//fprintf(fp1, "substep %d Az CPU ", iSubstep);
			//for (int jjj = 0; jjj < 7; jjj++)
			//	fprintf(fp1, "%d %1.14E ", jjj, Az_array[Chosens[jjj]]);
			//cudaMemcpy(p_temphost2, p_Az, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			//fprintf(fp1, "| Az GPU ");
			//for (int jjj = 0; jjj < 7; jjj++)
			//	fprintf(fp1, "%d %1.14E ", jjj, p_temphost2[Chosens[jjj]]);
			//fprintf(fp1, "| LapAz CPU ");
			//for (int jjj = 0; jjj < 7; jjj++)
			//	fprintf(fp1, "%d %1.14E ", jjj, LapAzArray[Chosens[jjj]]);
			//fprintf(fp1, "| LapAz GPU ");
			//for (int jjj = 0; jjj < 7; jjj++)
			//	fprintf(fp1, "%d %1.14E ", jjj, p_temphost1[Chosens[jjj]]);
			//fprintf(fp1, "\n");

			//if ((iSubstep == 9) && (0)) {
			//	ColIndex = 0;
			//	
			//	for (int jjj = 25366; jjj < 25661; jjj++) // the triangles
			//	{
			//		storageAz[450 * ColIndex + jjj - 25366] = p_temphost2[jjj] - Az_array[jjj];
			//		storageLap[450 * ColIndex + jjj - 25366] = p_temphost1[jjj] - LapAzArray[jjj];

			//	}
			//	for (int jjj = 86338; jjj < 86485; jjj++) // the triangles 
			//	{
			//		storageAz[450 * ColIndex + 295 + jjj - 86338] = p_temphost2[jjj] - Az_array[jjj];
			//		storageLap[450 * ColIndex + 295 + jjj - 86338] = p_temphost1[jjj] - LapAzArray[jjj];
			//	}
			//};

			//printf("compare n interp:\n");
			//p_cuSyst_compare->PopulateFromTriMesh(pTriMeshhalf);
			//cudaMemcpy(p_cuSyst_host->p_n_minor, pX_half->p_n_minor, sizeof(nvals)*NMINOR, cudaMemcpyDeviceToHost);
			//Compare_nvals(p_cuSyst_host->p_n_minor, p_cuSyst_compare->p_n_minor, NMINOR);
			//
			//printf("compare T interp:\n");
			//cudaMemcpy(p_cuSyst_host->p_T_minor, pX_half->p_T_minor, sizeof(T3)*NMINOR, cudaMemcpyDeviceToHost);
			//Compare_T3(p_cuSyst_host->p_T_minor, p_cuSyst_compare->p_T_minor, NMINOR);
			//printf("\n");

			//printf("compare T dest:\n");
			//p_cuSyst_compare->PopulateFromTriMesh(pDestMesh);
			//cudaMemcpy(p_cuSyst_host->p_T_minor, pX_target->p_T_minor, sizeof(T3)*NMINOR, cudaMemcpyDeviceToHost);
			//Compare_T3(p_cuSyst_host->p_T_minor, p_cuSyst_compare->p_T_minor, NMINOR);
			//printf("\n");

			// The equation is A_k+1 - h~gamma~ Lap A_k+1 - A_k - h~Azdot0~ = 0
			// Calculate regressor x_Jacobi from eps/coeff_on_A_i
			// Given a proposed addition of a regressor x, deps/dbeta = x - h~gamma~ Lap x 
			// Set beta = -sum[eps deps/dbeta] / sum[deps/dbeta ^2]

			// evaltime + 0.5*SUBSTEP used for setting EzStrength://
			Iz_prescribed = GetIzPrescribed(evaltime + 0.5*SUBSTEP); // APPLIED AT END TIME: we are determining
																	 // Jz, hence Iz at k+SUBSTEP initially.
			neg_Iz_per_triangle = -Iz_prescribed / (f64)numReverseJzTriangles;
			Set_f64_constant(negative_Iz_per_triangle, neg_Iz_per_triangle);
			// Electrons travel from cathode to anode so Jz is down in filament,
			// up around anode.
			printf("neg_Iz_per_triangle %1.14E Iz_presc %1.14E numRev %d \n", neg_Iz_per_triangle, Iz_prescribed,
				numReverseJzTriangles);

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
				(iSubstep == SUBCYCLES-1),
				pX_target->p_n_minor
				);
			Call(cudaThreadSynchronize(), "cudaTS kernelPopulateOhmsLaw ");

			cudaMemcpy(p_Iz0_summands_host, p_Iz0_summands, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_summands_host, p_sigma_Izz, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			Iz0 = 0.0;
			f64 Sigma_Izz = 0.0;
			long iBlock;
		//	FILE * fp_gpu = fopen("gpu_sigma.txt","w");
			for (iBlock = 0; iBlock < numTilesMinor; iBlock++)
			{
				Iz0 += p_Iz0_summands_host[iBlock];
				Sigma_Izz += p_summands_host[iBlock];
		//		fprintf(fp_gpu, "%d %1.14E \n", iBlock, p_summands_host[iBlock]);
			}
		//	fclose(fp_gpu);
					
			f64 EzStrength_ = (Iz_prescribed - Iz0) / Sigma_Izz;
			Set_f64_constant(Ez_strength, EzStrength_);
			SetConsoleTextAttribute(hConsole, 13);
			printf("\nGPU: Iz0 = %1.14E SigmaIzz %1.13E EzStrength = %1.14E \n\n", Iz0, Sigma_Izz, EzStrength_);
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
					(iSubstep == SUBCYCLES - 1) // bool bUse_n_dest_for_Iz
				);
			} else {
				pDestMesh->Accelerate2018(SUBSTEP,// ROCAzdotduetoAdvection, 
					pTriMeshhalf, 
					pDestMesh, evaltime + 0.5*SUBSTEP,
					true, // bool bFeint
					(iSubstep == SUBCYCLES - 1) // bool bUse_n_dest_for_Iz
				);
			}
			SetConsoleTextAttribute(hConsole, 15);
			 
			cudaMemcpy(p_temphost1, p_Azdot0, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost2, p_gamma, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			printf("Azdot0:\n");
			Compare_f64(p_temphost1, Azdot0, NMINOR);
			printf("From 4000:\n");
			Compare_f64(p_temphost1 + 4000, Azdot0 + 4000, NMINOR - 4000);
			printf("gamma:\n\n&&&&&&&&&&&&&&&\n");
			Compare_f64(p_temphost2, gamma, NMINOR);
		//	printf("From 4000:\n");
		//	Compare_f64(p_temphost2 + 4000, gamma + 4000, NMINOR - 4000);


			// Error at 10th place for reverseJz is caused apparently by AreaMinor not Iz_cell

			//if ((iSubstep == 9) && (0)) {
			//	printf("\n\n\n");

			//	ColIndex = 0;

			//	for (int jjj = 25366; jjj < 25661; jjj++) // the triangles
			//	{
			//		storageAzdot0[450 * ColIndex + jjj - 25366] = p_temphost1[jjj] - Azdot0[jjj];
			//		storageGamma[450 * ColIndex + jjj - 25366] = p_temphost2[jjj] - gamma[jjj];
			//	}
			//	for (int jjj = 86338; jjj < 86485; jjj++) // the triangles 
			//	{
			//		storageAzdot0[450 * ColIndex + 295 + jjj - 86338] = p_temphost1[jjj] - Azdot0[jjj];
			//		storageGamma[450 * ColIndex + 295 + jjj - 86338] = p_temphost2[jjj] - gamma[jjj];
			//	}
			//}
			// _____________________________________________________________

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

		//	if (iSubstep != 9)
		//	{
			pTriMeshhalf->JLS_for_Az_bwdstep(4, SUBSTEP); // populate Az_array with k+1 values
				// otherwise we are going to have to split it out.... to take differences!
		//	} // otuput: Az_array_next[]
			
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
					p_LapAzNext
					);
				Call(cudaThreadSynchronize(), "cudaTS GetLap Az JLS 1");

				// First time through, this is seed:
				//if ((iSubstep == 9) && (0)) {

				//	// do JLS on CPU:
				//	pTriMeshhalf->GetLap(Az_array_next, Lap_Aznext);

				//	// output:
				//	cudaMemcpy(p_temphost1, p_AzNext, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
				//	cudaMemcpy(p_temphost2, p_LapAzNext, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);

				//	ColIndex = iIteration + 1;

				//	for (int jjj = 25366; jjj < 25661; jjj++) // the triangles
				//	{
				//		storageAz[450 * ColIndex + jjj - 25366] = p_temphost1[jjj] - Az_array_next[jjj];
				//		storageLap[450 * ColIndex + jjj - 25366] = p_temphost2[jjj] - Lap_Aznext[jjj];
				//	}
				//	for (int jjj = 86338; jjj < 86485; jjj++) // the triangles 
				//	{
				//		storageAz[450 * ColIndex + 295 + jjj - 86338] = p_temphost1[jjj] - Az_array_next[jjj];
				//		storageLap[450 * ColIndex + 295 + jjj - 86338] = p_temphost2[jjj] - Lap_Aznext[jjj];
				//	}
				//}
				// printf("25526: Aznext GPU %1.14E CPU %1.14E \n", p_temphost1[25526], Az_array_next[25526]);
				
				//pTri = T;
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
					p_LapJacobi
					);
				Call(cudaThreadSynchronize(), "cudaTS Get Lap Jacobi 1");

			
			//	Worked out the problem is probably that different values at edges and frills
			//	do cause a different total sum and therefore different beta, and this is why
			//	we then gradually deviate.

			//	We need to reconcile both versions ensuring that it's the best frill it can be.

				// ?


				//if ((iSubstep == 9) && (0)) {

				//	pTri = pTriMeshhalf->T;
				//	for (iMinor = 0; iMinor < NMINOR; iMinor++)
				//	{
				//		// Here it immediately turns out we do need GetLapCoeff.
				//		if ((iMinor < BEGINNING_OF_CENTRAL) &&
				//			((pTri->u8domain_flag == OUTER_FRILL) || (pTri->u8domain_flag == INNER_FRILL)))
				//		{
				//			epsilon[iMinor] = Lap_Aznext[iMinor];
				//			Jacobi_x[iMinor] = -epsilon[iMinor] / LapCoeffself[iMinor];
				//			// Why this is coming out zero?

				//			if (iMinor == 0) printf("\nJACOBI_X[0] : eps %1.10E LapCoeffself %1.10E Jacobi %1.10E\n",
				//				epsilon[iMinor], LapCoeffself[iMinor], Jacobi_x[iMinor]);


				//		}
				//		else {
				//			epsilon[iMinor] = Az_array_next[iMinor] - SUBSTEP* gamma[iMinor] * Lap_Aznext[iMinor] - Az_array[iMinor] - SUBSTEP * Azdot0[iMinor];
				//			Jacobi_x[iMinor] = -epsilon[iMinor] / (1.0 - SUBSTEP * gamma[iMinor] * LapCoeffself[iMinor]);

				//			if (iMinor == 25526) printf("\nJacobi_x 25526 CPU: Jac %1.14E eps %1.14E gamma %1.14E LapCoeffself %1.14E\n",
				//				Jacobi_x[iMinor], epsilon[iMinor], gamma[iMinor], LapCoeffself[iMinor]);
				//			if (iMinor == 86412) printf("Jacobi_x 86412 CPU: Jac %1.14E eps %1.14E gamma %1.14E LapCoeffself %1.14E\n",
				//				Jacobi_x[iMinor], epsilon[iMinor], gamma[iMinor], LapCoeffself[iMinor]);
				//			if (iMinor == 69531) printf("Jacobi_x 69531 CPU: Jac %1.14E eps %1.14E gamma %1.14E LapCoeffself %1.14E\n",
				//				Jacobi_x[iMinor], epsilon[iMinor], gamma[iMinor], LapCoeffself[iMinor]);

				//		};

				//		++pTri;
				//	};
				//	pTriMeshhalf->GetLap(Jacobi_x, Lap_Jacobi);

				//	cudaMemcpy(p_temphost1, p_LapJacobi, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
				//	printf("\n\nLAP JACOBI COMPARE:\n");
				//	Compare_f64(p_temphost1, Lap_Jacobi, NMINOR);
				//	
				//	cudaMemcpy(p_temphost2, p_Jacobi_x, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
				//	printf("\n\nJacobi COMPARE:\n");
				//	Compare_f64(p_temphost2, Jacobi_x, NMINOR);
				//	
				//	sum_eps_deps_by_dbeta = 0.0;
				//	sum_depsbydbeta_sq = 0.0;
				//	sum_eps_eps = 0.0;
				//	pTri = pTriMeshhalf->T;
				//	for (iMinor = 0; iMinor < NMINOR; iMinor++)
				//	{
				//		if ((iMinor < BEGINNING_OF_CENTRAL) &&
				//			((pTri->u8domain_flag == OUTER_FRILL) || (pTri->u8domain_flag == INNER_FRILL)))
				//		{
				//			depsbydbeta = 0.0; //  Lap_Jacobi[iMinor]; // try ignoring
				//		}
				//		else {
				//			depsbydbeta = (Jacobi_x[iMinor] - SUBSTEP * gamma[iMinor] * Lap_Jacobi[iMinor]);
				//		};
				//		sum_eps_deps_by_dbeta += epsilon[iMinor] * depsbydbeta;
				//		sum_depsbydbeta_sq += depsbydbeta * depsbydbeta;
				//		sum_eps_eps += epsilon[iMinor] * epsilon[iMinor];
				//		++pTri;
				//	};
				//	beta = -sum_eps_deps_by_dbeta / sum_depsbydbeta_sq;
				//	L2eps = sqrt(sum_eps_eps / (real)NMINOR);

				//	SetConsoleTextAttribute(hConsole, 13);
				//	printf(" CPU[ beta %1.14E L2eps %1.14E ] ", beta, L2eps);
				//	printf(" CPU sum_eps_deps %1.14E sum_depsbydbeta_sq %1.14E ", 
				//		sum_eps_deps_by_dbeta , sum_depsbydbeta_sq);
				//	SetConsoleTextAttribute(hConsole, 15);

				//	for (iMinor = 0; iMinor < NMINOR; iMinor++)
				//	{
				//		Az_array_next[iMinor] += beta * Jacobi_x[iMinor];
				//	};

				////	printf("\n25526: Aznext CPU %1.14E beta %1.14E Jacobi_x %1.14E \n", Az_array_next[25526], beta, Jacobi_x[25526]);
				////	printf("86412: Aznext CPU %1.14E beta %1.14E Jacobi_x %1.14E step %d iter %d\n", Az_array_next[86412], beta, Jacobi_x[86412],
				////		iSubstep, iIteration);

				//	// Try resetting frills here and ignoring in calculation:
				//	pTri = pTriMeshhalf->T;
				//	for (iMinor = 0; iMinor < NUMTRIANGLES; iMinor++)
				//	{
				//		if ((pTri->u8domain_flag == INNER_FRILL) ||
				//			(pTri->u8domain_flag == OUTER_FRILL))
				//			Az_array_next[iMinor] = Az_array_next[pTri->neighbours[0] - pTriMeshhalf->T];
				//		++pTri;
				//	};
				//}

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

	//			f64 temp1,temp2;
	//			cudaMemcpy(&temp1, &(p_AzNext[25526]), sizeof(f64), cudaMemcpyDeviceToHost);
	//			cudaMemcpy(&temp2, &(p_Jacobi_x[25526]), sizeof(f64), cudaMemcpyDeviceToHost);
	//			printf("\n\n####\n25526: Aznext GPU %1.14E beta %1.14E Jacobi_x %1.14E step %d iter %d\n", temp1, beta, temp2, iSubstep,
	//				iIteration);
	//			cudaMemcpy(&temp1, &(p_AzNext[86412]), sizeof(f64), cudaMemcpyDeviceToHost);
	//			cudaMemcpy(&temp2, &(p_Jacobi_x[86412]), sizeof(f64), cudaMemcpyDeviceToHost);
	//			printf("86412: Aznext GPU %1.14E beta %1.14E Jacobi_x %1.14E \n\n", temp1, beta, temp2);
	//					

				// Try resetting frills here and ignoring in calculation:
				kernelResetFrillsAz << <numTriTiles, threadsPerTileMinor >> > (
					this->p_info, this->p_tri_neigh_index, p_AzNext);
				Call(cudaThreadSynchronize(), "cudaTS ResetFrills X");

			};

			printf("\n\n");
			
			cudaMemcpy(p_Az, p_AzNext, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
			memcpy(Az_array, Az_array_next, sizeof(f64)*NMINOR);
			
			// That was:
			//	JLS_for_Az_bwdstep(4, SUBSTEP); // populate Az_array with k+1 values
			
			kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> >(
				pX_half->p_info,
				p_Az,
				pX_half->p_izTri_vert,
				pX_half->p_izNeigh_TriMinor,
				pX_half->p_szPBCtri_vert,
				pX_half->p_szPBC_triminor,
				p_LapAz
				);
			Call(cudaThreadSynchronize(), "cudaTS GetLap Az JLS 2");

			pTriMeshhalf->GetLap(Az_array, LapAzArray);

			// Now let's compare Az and LapAz:

			cudaMemcpy(p_temphost1, p_Az, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_temphost2, p_LapAz, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);

			printf("Az:\n");
			Compare_f64(p_temphost1, Az_array, NMINOR);
			printf("LapAz:\n");
			Compare_f64(p_temphost2, LapAzArray, NMINOR);

			//if ((iSubstep == 9) && (0)) {
			//	ColIndex = 3;

			//	for (int jjj = 25366; jjj < 25661; jjj++) // the triangles
			//	{
			//		storageAz[450 * ColIndex + jjj - 25366] = p_temphost1[jjj] - Az_array[jjj];
			//		storageLap[450 * ColIndex + jjj - 25366] = p_temphost2[jjj] - LapAzArray[jjj];

			//	}
			//	for (int jjj = 86338; jjj < 86485; jjj++) // the triangles 
			//	{
			//		storageAz[450 * ColIndex + 295 + jjj - 86338] = p_temphost1[jjj] - Az_array[jjj];
			//		storageLap[450 * ColIndex + 295 + jjj - 86338] = p_temphost2[jjj] - LapAzArray[jjj];
			//	}
			//}

			// Difference at 11th place. See above.
		//	printf("For chosen %d : LapAz %1.14E %1.14E \n", CHOSEN,
		//		p_temphost2[CHOSEN], LapAzArray[CHOSEN]);

			// Leaving Iz_prescribed and reverse_Jz the same:


			// Think I'm right all that has changed is LapAz so do we really have to go through whole thing again? :

			//	this->Accelerate2018(SUBSTEP, pX_half, pDestMesh, evaltime + 0.5*SUBSTEP, false); // Lap Az now given.
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
				(iSubstep == SUBCYCLES-1),
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
			EzStrength_ = (Iz_prescribed - Iz0) / Sigma_Izz;
			Set_f64_constant(Ez_strength, EzStrength_);

			printf("\nGPU EzStrength_ %1.14E Iz0 %1.14E Sigma_Izz %1.14E Iz_prescribed %1.14E\n\n",
				EzStrength_, Iz0, Sigma_Izz, Iz_prescribed);

			kernelCalculateVelocityAndAzdot << <numTilesMinor, threadsPerTileMinor >> >(
				SUBSTEP,
				p_vn0,
				p_v0,
				p_OhmsCoeffs,
				pX_half->p_AAdot,
				pX_half->p_n_minor,

				pX_target->p_AAdot,  // not understanding
				pX_target->p_vie,
				pX_target->p_v_n
				);
			Call(cudaThreadSynchronize(), "cudaTS kernelUpdate_v ");
			
			if (iSubstep == 0) {
				pTriMesh->Accelerate2018(SUBSTEP, pTriMeshhalf, pDestMesh,
					evaltime + 0.5*SUBSTEP, false,
					(iSubstep == SUBCYCLES - 1)); // Lap Az now given.
			} else {
				pDestMesh->Accelerate2018(SUBSTEP, pTriMeshhalf, pDestMesh,
					evaltime + 0.5*SUBSTEP, false,
					(iSubstep == SUBCYCLES - 1)); // Lap Az now given.
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
			// Why we do not pass it back and forth? Can't remember.

		}; // next substep
		
		//fclose(fp);
		//fclose(fp1);
		/*
		fp1 = fopen("storedALap4iters.txt", "w");
		for (int jjj = 25366; jjj < 25661; jjj++) // the triangles
			fprintf(fp1, "%d Az %1.14E %1.14E %1.14E %1.14E LapAz %1.14E %1.14E %1.14E %1.14E Azdot0 %1.14E gamma %1.14E \n",
				jjj,
				storageAz[0 + jjj - 25366],
				storageAz[450 + jjj - 25366],
				storageAz[900 + jjj - 25366],
				storageAz[1350 + jjj - 25366],
				storageLap[0 + jjj - 25366],
				storageLap[450 + jjj - 25366],
				storageLap[900 + jjj - 25366],
				storageLap[1350 + jjj - 25366],
				storageAzdot0[jjj - 25366], storageGamma[jjj - 25366]
			);
		for (int jjj = 86338; jjj < 86485; jjj++) // the triangles 
			fprintf(fp1, "%d Az %1.14E %1.14E %1.14E %1.14E LapAz %1.14E %1.14E %1.14E %1.14E Azdot0 %1.14E gamma %1.14E \n",
				jjj,
				storageAz[0 + 295 + jjj - 86338],
				storageAz[450 + 295 + jjj - 86338],
				storageAz[900 + 295 + jjj - 86338],
				storageAz[1350 + 295 + jjj - 86338],
				storageLap[0 + 295 + jjj - 86338],
				storageLap[450 + 295 + jjj - 86338],
				storageLap[900 + 295 + jjj - 86338],
				storageLap[1350 + 295 + jjj - 86338],
				storageAzdot0[jjj + 295 - 86338], storageGamma[jjj + 295 - 86338]
			);

		fclose(fp1);
		*/



		kernelPushAzInto_dest << <numTilesMinor, threadsPerTileMinor >> >(
			pX_target->p_AAdot,
			p_Az);
		Call(cudaThreadSynchronize(), "cudaTS PushAzIntoDest");

		for (iMinor = 0; iMinor < NMINOR; iMinor++)
			pDestMesh->pData[iMinor].Az = Az_array[iMinor];

		// more advanced implicit could be possible and effective.
		// It is almost certain that splitting up BJLS into a few goes in each set of subcycles would be more effective than being a different set all BJLS.
		// This should be experimented with, once it matches CPU output.
	} else {

		printf("end\n");
		while (1) getch();

  
		kernelPopulateArrayAz << <numTilesMinor, threadsPerTileMinor >> >(
			0.5*SUBSTEP,
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
		// Create_A_from_advance(0.5*SUBSTEP, ROCAzduetoAdvection, Az_array); // from *this

		for (iSubstep = 0; iSubstep < SUBCYCLES; iSubstep++)
		{
			evaltime += 0.5*SUBSTEP;
			kernelInterpolateVarsAndPositions << <numTilesMinor, threadsPerTileMinor >> >(
				(evaltime - starttime) / TIMESTEP,
				this->p_info,
				pX_target->p_info,
				this->p_n_minor,
				pX_target->p_n_minor,
				this->p_T_minor,
				pX_target->p_T_minor,
				this->p_B,
				pX_target->p_B,
				pX_half->p_info,
				pX_half->p_n_minor,
				pX_half->p_T_minor,
				pX_half->p_B
				);
			Call(cudaThreadSynchronize(), "cudaTS InterpolateVars");
			// let n,T,x be interpolated on to pX_half. B remains what we populated there.
			// Recalculate areas, or tween them, would make good sense as well.

			// It might be nice to tween the thermal pressure grad(nT) as well. And logical:
			// InterpolateArrays(AdditionalMomRates_start, AdditionalMomRates_end, AdditionalMomRates, (evaltime - starttime) / TIMESTEP);
			// Have a look how AMR is created.
			kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> >(
				pX_half->p_info,
				p_Az,
				pX_half->p_izTri_vert,
				pX_half->p_izNeigh_TriMinor,
				pX_half->p_szPBCtri_vert,
				pX_half->p_szPBC_triminor,
				p_LapAz
				);
			Call(cudaThreadSynchronize(), "cudaTS GetLap Az Leapfrog 1");

			// evaltime + 0.5*SUBSTEP used for setting EzStrength://
			Iz_prescribed = GetIzPrescribed(evaltime + 0.5*SUBSTEP);
			f64 neg_Iz_per_triangle = -Iz_prescribed / (f64)numReverseJzTriangles;
			Set_f64_constant(negative_Iz_per_triangle, neg_Iz_per_triangle);

			// On the first step we use "this" as src, otherwise pX_targ to pX_targ
			// Simple plan:
			// Pop Ohms just populate's Ohms and advances Azdot to an intermediate state

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
				(iSubstep == SUBCYCLES-1), // if not, use pX_half->p_n_minor for Iz -- roughly right but wrong
				pX_target->p_n_minor); 
			Call(cudaThreadSynchronize(), "cudaTS kernelPopulateOhmsLaw ");

			cudaMemcpy(p_Iz0_summands_host, p_Iz0_summands, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_summands_host, p_sigma_Izz, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			Iz0 = 0.0; sigma_Izz = 0.0;
			for (int iBlock = 0; iBlock < numTilesMinor; iBlock++)
			{
				Iz0 += p_Iz0_summands_host[iBlock];
				sigma_Izz += p_summands_host[iBlock];
			}
			f64 EzStrength_ = (Iz_prescribed - Iz0) / sigma_Izz;
			Set_f64_constant(Ez_strength, EzStrength_);

			kernelCalculateVelocityAndAzdot << <numTilesMinor, threadsPerTileMinor >> >(
				SUBSTEP,
				p_vn0,
				p_v0,
				p_OhmsCoeffs,
				pX_half->p_AAdot,
				pX_half->p_n_minor,

				pX_target->p_AAdot,
				pX_target->p_vie,
				pX_target->p_v_n
				);
			Call(cudaThreadSynchronize(), "cudaTS kernelUpdate_v ");

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

			evaltime += 0.5*SUBSTEP;
		};

		kernelPushAzInto_dest << <numTilesMinor, threadsPerTileMinor >> >(
			pX_target->p_AAdot,
			p_Az);
		Call(cudaThreadSynchronize(), "cudaTS PushAzIntoDest");

	}; // whether Backward or Leapfrog

	printf("evaltime %1.5E \n", evaltime);
	printf("-----------------\n");

	//this->AntiAdvectAzAndAdvance(h, pX_half, IntegratedGradAz, pDestMesh); // Might as well subsume this in Accelerate, really
	//pX_half->AntiAdvectAzAndAdvance(h*0.5, pDestMesh, GradAz, pDestMesh);

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
		pX_target->p_szPBC_triminor
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

	printf("Done step %d from time %1.8E length %1.2E\n\n", runs, evaltime, TIMESTEP);

	// For graphing Lap Az:
	cudaMemcpy(temp_array_host, p_LapAz, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Elapsed time : %f ms\n", elapsedTime);

	runs++;

	free(shardtemp);

	printf("end\n");
	getch();
}

void cuSyst::PerformCUDA_Advance(const cuSyst * pX_target, const cuSyst * pX_half)
{
	long iMinor, iSubstep;
	f64 Iz_prescribed;
	static long runs = 0;
	float elapsedTime;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// inauspicious start: overall v has to be split into 2 routines
	kernelCalculateOverallVelocitiesVertices << <numTilesMajor, threadsPerTileMajor >> > (
		this->p_info + BEGINNING_OF_CENTRAL,
		this->p_vie + BEGINNING_OF_CENTRAL,
		this->p_v_n + BEGINNING_OF_CENTRAL,
		this->p_n_major,
		this->p_v_overall_minor + BEGINNING_OF_CENTRAL
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

	//	pX_half->ZeroData(); // Is this serving any purpose?

	kernelAdvectPositions<< <numTilesMinor, threadsPerTileMinor >> >(
		0.5*TIMESTEP,
		this->p_info,
		pX_half->p_info,
		this->p_v_overall_minor
		);
	Call(cudaThreadSynchronize(), "cudaTS AdvectPositions_CopyTris");

	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> >(
		this->p_n_minor,
		this->p_n_major,
		this->p_T_minor,
		this->p_info,
		this->p_tri_corner_index,
		this->p_tri_periodic_corner_flags
		); // call before CreateShardModel 
	Call(cudaThreadSynchronize(), "cudaTS average nTx");
	cudaMemcpy(this->p_n_minor + BEGINNING_OF_CENTRAL,
		this->p_n_major, sizeof(nvals)*NUMVERTICES, cudaMemcpyDeviceToDevice);

	kernelCreateShardModelOfDensities_And_SetMajorArea << <numTilesMajor, threadsPerTileMajor >> >(
		this->p_info,
		this->p_n_major,
		this->p_n_minor,
		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		p_n_shards,
		p_n_shards_n,
		//p_Tri_n_lists,
		//p_Tri_n_n_lists,
		this->p_AreaMajor
		);// sets n_shards_n, n_shards, Tri_n_n_lists, Tri_n_lists
	Call(cudaThreadSynchronize(), "cudaTS CreateShardModels");

	kernelInferMinorDensitiesFromShardModel << <numTilesMinor, threadsPerTileMinor >> >(
		this->p_info,
		this->p_n_minor,
		p_n_shards,
		p_n_shards_n,
		this->p_tri_corner_index,
		this->p_who_am_I_to_corner);
	Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities");

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
		this->p_n_upwind_minor);
	Call(cudaThreadSynchronize(), "cudaTS CalculateUpwindDensity_tris");

	cudaMemset(p_Integrated_div_v_overall, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Div_v_neut, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Div_v, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(NT_addition_rates_d, 0, sizeof(NTrates)*NUMVERTICES);

	kernelAccumulateAdvectiveMassHeatRate << <numTilesMajor, threadsPerTileMajor >> >(
		0.5*TIMESTEP, // why it appears here?
		this->p_info,
		this->p_izTri_vert,
		this->p_szPBCtri_vert,

		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL,

		this->p_n_upwind_minor,
		this->p_vie,
		this->p_v_n,
		this->p_v_overall_minor,
		this->p_T_minor, // may or may not overlap source: don't we only use from tris? so not overlap

		NT_addition_rates_d,
		p_Div_v,
		p_Div_v_neut,
		p_Integrated_div_v_overall);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateAdvectiveMassHeatRate");

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
		this->p_v_overall_minor,
		p_MAR_neut
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelNeutral_pressure_and_momflux");

	kernelCalculateNu_eHeartNu_iHeart_nu_nn_visc << <numTilesMajorClever, threadsPerTileMajorClever >> >(
		this->p_info + BEGINNING_OF_CENTRAL,
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL,
		p_nu_major);
	Call(cudaThreadSynchronize(), "cudaTS CalculateNu");

	kernelAccumulateDiffusiveHeatRateAndCalcIonisation << <numTilesMajorClever, threadsPerTileMajorClever >> >(
		0.5*TIMESTEP,
		this->p_info + BEGINNING_OF_CENTRAL,
		this->p_izNeigh_vert,
		this->p_szPBCneigh_vert,
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices
		this->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
		p_nu_major,
		NT_addition_rates_d,
		this->p_AreaMajor);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate");
	// To increase the efficiency we want to make a clever 2nd set of major tiles of size 192. Also try 256, 384.

	kernelAdvanceDensityAndTemperature << <numTilesMajor, threadsPerTileMajor >> >(
		0.5*TIMESTEP,
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

	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> >(
		pX_half->p_n_minor,
		pX_half->p_n_major,
		pX_half->p_T_minor,
		pX_half->p_info,
		pX_half->p_tri_corner_index,
		pX_half->p_tri_periodic_corner_flags
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
		p_n_shards,
		p_n_shards_n,
		//		p_Tri_n_lists,
		//	p_Tri_n_n_lists,
		pX_half->p_AreaMajor);
	Call(cudaThreadSynchronize(), "cudaTS CreateShardModel pX_half");

	kernelInferMinorDensitiesFromShardModel << <numTilesMinor, threadsPerTileMinor >> >(
		pX_half->p_info,
		pX_half->p_n_minor,
		p_n_shards,
		p_n_shards_n,
		pX_half->p_tri_corner_index,
		pX_half->p_who_am_I_to_corner);// (At the moment just repopulating tri minor n.)
	Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities pX_half");

	Iz_prescribed = GetIzPrescribed(evaltime + 0.5*TIMESTEP); // because we are setting pX_half->v
	
	f64 neg_Iz_per_triangle = -Iz_prescribed / (f64)numReverseJzTriangles;
	Set_f64_constant(negative_Iz_per_triangle, neg_Iz_per_triangle);
	// Get suitable v to use for resistive heating:
	kernelPopulateOhmsLaw<<<numTilesMinor, threadsPerTileMinor>>>(
				
		0.5*TIMESTEP,

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
		p_denom_e, p_coeff_of_vez_upon_viz, p_beta_ie_z,
		false,
		true, // use this for Iz:
		pX_half->p_n_minor
		); 

	Call(cudaThreadSynchronize(), "cudaTS AccelerateOhms 1");

	cudaMemcpy(p_Iz0_summands_host, p_Iz0_summands, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	cudaMemcpy(p_summands_host, p_sigma_Izz, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
	
	// Now calculate Ez_strength to achieve Iz_prescribed:
	Iz_prescribed = GetIzPrescribed(evaltime); // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	long iTile;
	f64 Iz0 = 0.0;
	f64 sigma_Izz = 0.0;
	for (iTile = 0; iTile < numTilesMinor; iTile++)
	{
		Iz0 += p_Iz0_summands_host[iTile];
		sigma_Izz += p_summands_host[iTile];
	};
	f64 Ez_strength_ = (Iz_prescribed - Iz0) / sigma_Izz;
	Set_f64_constant(Ez_strength, Ez_strength_);

	// Update velocities and Azdot:
	kernelCalculateVelocityAndAzdot <<<numTilesMinor, threadsPerTileMinor >>>(
		0.5*TIMESTEP,
		p_vn0,
		p_v0,
		p_OhmsCoeffs,
		pX_target->p_AAdot, 
		pX_half->p_n_minor,

		pX_half->p_AAdot,
		pX_half->p_vie,
		pX_half->p_v_n
		);
	Call(cudaThreadSynchronize(), "cudaTS AccelerateUpdate 1");

	// ))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
	// 
	// Now do the n,T,x advance to pDestmesh:
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	kernelCalculateOverallVelocitiesVertices << <numTilesMajor, threadsPerTileMajor >> >(
		pX_half->p_info + BEGINNING_OF_CENTRAL,
		pX_half->p_vie + BEGINNING_OF_CENTRAL,
		pX_half->p_v_n + BEGINNING_OF_CENTRAL,
		pX_half->p_n_major,
		pX_half->p_v_overall_minor + BEGINNING_OF_CENTRAL
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

	kernelAdvectPositions << <numTilesMinor, threadsPerTileMinor >> >(
		TIMESTEP,
		this->p_info,
		pX_target->p_info,
		this->p_v_overall_minor
		);
	Call(cudaThreadSynchronize(), "cudaTS AdvectPositions_CopyTris");
	
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
		pX_half->p_n_upwind_minor);
	Call(cudaThreadSynchronize(), "cudaTS CalculateUpwindDensity_tris pX_half");

	cudaMemset(p_Integrated_div_v_overall, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Div_v_neut, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_Div_v, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(NT_addition_rates_d, 0, sizeof(NTrates)*NUMVERTICES);

	kernelAccumulateAdvectiveMassHeatRate << <numTilesMajor, threadsPerTileMajor >> >(
		TIMESTEP,
		pX_half->p_info,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,

		pX_half->p_n_major,
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL,

		pX_half->p_n_upwind_minor,
		pX_half->p_vie,
		pX_half->p_v_n,
		pX_half->p_v_overall_minor,
		pX_half->p_T_minor, // may or may not overlap source: don't we only use from tris? so not overlap

		NT_addition_rates_d,
		p_Div_v,
		p_Div_v_neut,
		p_Integrated_div_v_overall);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateAdvectiveMassHeatRate pX_half");


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
		pX_half->p_v_overall_minor,
		p_MAR_neut
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelNeutral_pressure_and_momflux pX_half");


	kernelCalculateNu_eHeartNu_iHeart_nu_nn_visc << <numTilesMajorClever, threadsPerTileMajorClever >> >(
		pX_half->p_info + BEGINNING_OF_CENTRAL,
		pX_half->p_n_major,
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL,
		p_nu_major);
	Call(cudaThreadSynchronize(), "cudaTS CalculateNu pX_half");

	kernelAccumulateDiffusiveHeatRateAndCalcIonisation << <numTilesMajorClever, threadsPerTileMajorClever >> >(
		TIMESTEP,
		pX_half->p_info + BEGINNING_OF_CENTRAL,
		pX_half->p_izNeigh_vert,
		pX_half->p_szPBCneigh_vert,
		pX_half->p_n_major,
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices
		pX_half->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
		p_nu_major,
		NT_addition_rates_d,
		pX_half->p_AreaMajor);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate pX_half");
	// To increase the efficiency we want to make a clever 2nd set of major tiles of size 192. Also try 256, 384.

	kernelAdvanceDensityAndTemperature << <numTilesMajor, threadsPerTileMajor >> >(
		TIMESTEP,
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
		pX_half->p_AreaMajor,

		pX_target->p_n_major,
		pX_target->p_T_minor + BEGINNING_OF_CENTRAL
		);
	Call(cudaThreadSynchronize(), "cudaTS Advance_n_and_T 233");

	// QUESTION QUESTION : What should params be there?

	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> >(
		pX_target->p_n_minor,
		pX_target->p_n_major,
		pX_target->p_T_minor,
		pX_target->p_info,
		pX_target->p_tri_corner_index,
		pX_target->p_tri_periodic_corner_flags
		); // call before CreateShardModel 
	Call(cudaThreadSynchronize(), "cudaTS average nTx 233");
	cudaMemcpy(pX_target->p_n_minor + BEGINNING_OF_CENTRAL,
		pX_target->p_n_major, sizeof(nvals)*NUMVERTICES, cudaMemcpyDeviceToDevice);

	//pX_half->Add_ViscousMomentumFluxRates(AdditionalMomRates);

	//////////////////////////////////////////////////////////////////////////
	// Even more shards!:

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
		pX_target->p_who_am_I_to_corner);// (At the moment just repopulating tri minor n.)
	Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities pX_target");

	f64 starttime = evaltime;

	if (runs % 10 == 0)
	{
		// BACKWARD STEPS:
		kernelPullAzFromSyst<<<numTilesMinor, threadsPerTileMinor>>>(
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

		for (iSubstep = 0; iSubstep < SUBCYCLES; iSubstep++)
		{
			evaltime += 0.5*SUBSTEP;
			kernelInterpolateVarsAndPositions<<<numTilesMinor, threadsPerTileMinor>>>(
				(evaltime - starttime) / TIMESTEP,
				this->p_info,
				pX_target->p_info,
				this->p_n_minor,
				pX_target->p_n_minor,
				this->p_T_minor,
				pX_target->p_T_minor,
				this->p_B,
				pX_target->p_B,
				pX_half->p_info,
				pX_half->p_n_minor,
				pX_half->p_T_minor,
				pX_half->p_B
				);
			Call(cudaThreadSynchronize(), "cudaTS InterpolateVars");

			//pX_half->GetLapFromCoeffs(Az_array, LapAzArray);
	// NOTICE # BLOCKS -- THIS SHOULD ALSO APPLY WHEREVER WE DO SIMILAR THING LIKE WITH MOMFLUX.

			kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> >(
				pX_half->p_info,
				p_Az,
				pX_half->p_izTri_vert,
				pX_half->p_izNeigh_TriMinor,
				pX_half->p_szPBCtri_vert,
				pX_half->p_szPBC_triminor,
				p_LapAz
				);
			Call(cudaThreadSynchronize(), "cudaTS GetLapMinor aaa");
			// Should populate an array of coefficients s.t. Azdot_k+1 = ~Azdot0~ + ~gamma~ Lap Az
			// Now we will wanna create each eqn for Az with coeffs on neighbour values.
			// So we need a func called "GetLapCoefficients".

			// The equation is A_k+1 - h~gamma~ Lap A_k+1 - A_k - h~Azdot0~ = 0
			// Calculate regressor x_Jacobi from eps/coeff_on_A_i
			// Given a proposed addition of a regressor x, deps/dbeta = x - h~gamma~ Lap x 
			// Set beta = -sum[eps deps/dbeta] / sum[deps/dbeta ^2]

			// evaltime + 0.5*SUBSTEP used for setting EzStrength://
			Iz_prescribed = GetIzPrescribed(evaltime + 0.5*SUBSTEP); // APPLIED AT END TIME: we are determining
			// Jz, hence Iz at k+SUBSTEP initially.
			neg_Iz_per_triangle = -Iz_prescribed / (f64)numReverseJzTriangles;
			Set_f64_constant(negative_Iz_per_triangle, neg_Iz_per_triangle);
			// Electrons travel from cathode to anode so Jz is down in filament,
			// up around anode.
						
			kernelPopulateOhmsLaw << <numTilesMinor, threadsPerTileMinor >> >(
					SUBSTEP,// ROCAzdotduetoAdvection, 
					pX_half->p_info,
					p_MAR_neut,p_MAR_ion,p_MAR_elec,
					pX_half->p_B,
					p_LapAz,
					p_GradAz, 
					p_GradTe,
					
					pX_half->p_n_minor,
					pX_half->p_T_minor,
				(iSubstep == 0) ? this->p_vie:pX_target->p_vie,
				(iSubstep == 0) ? this->p_v_n:pX_target->p_v_n,
				(iSubstep == 0) ? this->p_AAdot:pX_target->p_AAdot, //	inputs
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
					true,
					(iSubstep == SUBCYCLES-1)?true:false,
					pX_target->p_n_minor);
				
				Call(cudaThreadSynchronize(), "cudaTS kernelPopulateOhmsLaw ");

				cudaMemcpy(p_Iz0_summands_host, p_Iz0_summands, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
				cudaMemcpy(p_summands_host, p_sigma_Izz, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
				Iz0 = 0.0;
				f64 Sigma_Izz = 0.0;
				long iBlock;
				for (iBlock = 0; iBlock < numTilesMinor; iBlock++)
				{
					Iz0 += p_Iz0_summands_host[iBlock];
					Sigma_Izz += p_summands_host[iBlock];
				}
				f64 EzStrength_ = (Iz_prescribed - Iz0) / Sigma_Izz;
				Set_f64_constant(Ez_strength, EzStrength_);

				kernelCreateLinearRelationship << <numTilesMinor, threadsPerTileMinor >> >(
					SUBSTEP,
					pX_half->p_info,
					p_OhmsCoeffs,
					p_v0,
					p_LapAz,
					pX_half->p_n_minor,
					p_denom_e, 
					p_denom_i, p_coeff_of_vez_upon_viz, p_beta_ie_z,
					pX_half->p_AAdot,
					p_Azdot0,
					p_gamma
				); // MODIFY vez0, viz0 IF THEY WILL EVER BE USED.
				Call(cudaThreadSynchronize(), "cudaTS kernelCreateLinearRelationship ");
				
				// _____________________________________________________________
				
				kernelCreateSeedPartOne << <numTilesMinor, threadsPerTileMinor >> > (
					SUBSTEP,
					p_Az,
					(iSubstep == 0) ? this->p_AAdot : pX_target->p_AAdot, // use 0.5*(Azdot_k + Azdot_k+1) for seed.
					p_AzNext);
				Call(cudaThreadSynchronize(), "cudaTS Create Seed 1");
							
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
			//Az_array_next[iMinor] = Az_array[iMinor] + 0.5*SUBSTEP*this->pData[iMinor].Azdot + 0.5*SUBSTEP * Azdot0[iMinor] + 0.5*SUBSTEP * gamma[iMinor] * LapAzArray[iMinor];
			//ie use 0.5*(Azdot_k[done] + Azdot_k+1) for seed.
			
			kernelCreateSeedPartTwo << <numTilesMinor, threadsPerTileMinor >> > (
				SUBSTEP,
				p_Azdot0, p_gamma, p_LapAz,
				p_AzNext);
			Call(cudaThreadSynchronize(), "cudaTS Create Seed 2"); // Okay -- we can now merge these. "Azdot_k" is preserved.
			
			// JLS:

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
					p_LapAzNext
					);
				Call(cudaThreadSynchronize(), "cudaTS GetLap Az JLS 1");

				//pTri = T;
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
					p_LapJacobi
					);
				Call(cudaThreadSynchronize(), "cudaTS Get Lap Jacobi 1");

				//sum_eps_deps_by_dbeta = 0.0;
				//sum_depsbydbeta_sq = 0.0;
				//sum_eps_eps = 0.0;
				//pTri = T;
				//for (iMinor = 0; iMinor < NMINOR; iMinor++)
				//{
				//	if ((iMinor < BEGINNING_OF_CENTRAL) &&
				//		((pTri->u8domain_flag == OUTER_FRILL) || (pTri->u8domain_flag == INNER_FRILL)))
				//	{
				//		depsbydbeta = 0.0; //  p_LapJacobi[iMinor]; // try ignoring
				//	}
				//	else {
				//		depsbydbeta = (Jacobi_x[iMinor] - h_use * gamma[iMinor] * p_LapJacobi[iMinor]);
				//	};
				//	sum_eps_deps_by_dbeta += epsilon[iMinor] * depsbydbeta;
				//	sum_depsbydbeta_sq += depsbydbeta * depsbydbeta;
				//	sum_eps_eps += epsilon[iMinor] * epsilon[iMinor];
				//	++pTri;
				//};
				kernelAccumulateSummands <<<numTilesMinor, threadsPerTileMinor>>> (
					this->p_info,
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
				printf(" [ %1.4f %1.2E ] ", beta, L2eps);

				kernelAdd <<<numTilesMinor, threadsPerTileMinor>>> (
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
			//	JLS_for_Az_bwdstep(4, SUBSTEP); // populate Az_array with k+1 values
			cudaMemcpy(p_Az, p_AzNext, sizeof(f64)*NMINOR, cudaMemcpyDeviceToDevice);
			kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> >(
				pX_half->p_info,
				p_Az,
				pX_half->p_izTri_vert,
				pX_half->p_izNeigh_TriMinor,
				pX_half->p_szPBCtri_vert,
				pX_half->p_szPBC_triminor,
				p_LapAz
				);
			Call(cudaThreadSynchronize(), "cudaTS GetLap Az JLS 2");
			
			// Leaving Iz_prescribed and reverse_Jz the same:
			
			// Think I'm right all that has changed is LapAz so do we really have to go through whole thing again? :

			//	this->Accelerate2018(SUBSTEP, pX_half, pDestMesh, evaltime + 0.5*SUBSTEP, false); // Lap Az now given.
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
				(iSubstep == SUBCYCLES-1)?true:false,
				pX_target->p_n_minor); 
			
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
			EzStrength_ = (Iz_prescribed - Iz0) / Sigma_Izz;
			Set_f64_constant(Ez_strength, EzStrength_);
			
			kernelCalculateVelocityAndAzdot << <numTilesMinor, threadsPerTileMinor >> >(
				0.5*SUBSTEP,
				p_vn0,
				p_v0,
				p_OhmsCoeffs,
				pX_half->p_AAdot,
				pX_half->p_n_minor,

				pX_target->p_AAdot,
				pX_target->p_vie,
				pX_target->p_v_n
				);
			Call(cudaThreadSynchronize(), "cudaTS kernelUpdate_v ");

			evaltime += 0.5*SUBSTEP;
			// Why we do not pass it back and forth? Can't remember.
		}; // next substep
		kernelPushAzInto_dest << <numTilesMinor, threadsPerTileMinor >> >(
			pX_target->p_AAdot,
			p_Az);
		Call(cudaThreadSynchronize(), "cudaTS PushAzIntoDest");

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

		kernelResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> >(
			this->p_info,
			this->p_tri_neigh_index,
			p_Az
			);
		Call(cudaThreadSynchronize(), "cudaTS kernelResetFrillsAz 1");
		// Create_A_from_advance(0.5*SUBSTEP, ROCAzduetoAdvection, Az_array); // from *this

		for (iSubstep = 0; iSubstep < SUBCYCLES; iSubstep++)
		{
			evaltime += 0.5*SUBSTEP;
			kernelInterpolateVarsAndPositions << <numTilesMinor, threadsPerTileMinor >> >(
				(evaltime - starttime) / TIMESTEP,
				this->p_info,
				pX_target->p_info,
				this->p_n_minor,
				pX_target->p_n_minor,
				this->p_T_minor,
				pX_target->p_T_minor,
				this->p_B,
				pX_target->p_B,
				pX_half->p_info,
				pX_half->p_n_minor,
				pX_half->p_T_minor,
				pX_half->p_B
				);
			Call(cudaThreadSynchronize(), "cudaTS InterpolateVars");
			// let n,T,x be interpolated on to pX_half. B remains what we populated there.
			// Recalculate areas, or tween them, would make good sense as well.

			// It might be nice to tween the thermal pressure grad(nT) as well. And logical:
			// InterpolateArrays(AdditionalMomRates_start, AdditionalMomRates_end, AdditionalMomRates, (evaltime - starttime) / TIMESTEP);
			// Have a look how AMR is created.
			kernelGetLap_minor << <numTriTiles, threadsPerTileMinor >> >(
				pX_half->p_info,
				p_Az,
				pX_half->p_izTri_vert,
				pX_half->p_izNeigh_TriMinor,
				pX_half->p_szPBCtri_vert,
				pX_half->p_szPBC_triminor,
				p_LapAz
				);
			Call(cudaThreadSynchronize(), "cudaTS GetLap Az Leapfrog 1");

			// evaltime + 0.5*SUBSTEP used for setting EzStrength://
			Iz_prescribed = GetIzPrescribed(evaltime + 0.5*SUBSTEP);
			f64 neg_Iz_per_triangle = -Iz_prescribed / (f64)numReverseJzTriangles;
			Set_f64_constant(negative_Iz_per_triangle, neg_Iz_per_triangle);

			// On the first step we use "this" as src, otherwise pX_targ to pX_targ
			// Simple plan:
			// Pop Ohms just populate's Ohms and advances Azdot to an intermediate state

			kernelPopulateOhmsLaw << <numTilesMinor, threadsPerTileMinor >> >(
				SUBSTEP,// ROCAzdotduetoAdvection, 
				pX_half->p_info,
				p_MAR_neut, p_MAR_ion, p_MAR_elec,
				pX_half->p_B,
				p_LapAz,
				p_GradAz,
				p_GradTe,

				pX_half->p_n_minor,
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
				(iSubstep < SUBCYCLES-1)?false:true,
				pX_target->p_n_minor
				); // bFeint
			Call(cudaThreadSynchronize(), "cudaTS kernelPopulateOhmsLaw ");
						
			cudaMemcpy(p_Iz0_summands_host, p_Iz0_summands, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			cudaMemcpy(p_summands_host, p_sigma_Izz, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
			Iz0 = 0.0; sigma_Izz = 0.0;
			for (int iBlock = 0; iBlock < numTilesMinor; iBlock++)
			{
				Iz0 += p_Iz0_summands_host[iBlock];
				sigma_Izz += p_summands_host[iBlock];
			}
			f64 EzStrength_ = (Iz_prescribed - Iz0) / sigma_Izz;
			Set_f64_constant(Ez_strength, EzStrength_);
			
			kernelCalculateVelocityAndAzdot << <numTilesMinor, threadsPerTileMinor >> >(
				SUBSTEP,
				p_vn0,
				p_v0,
				p_OhmsCoeffs,
				pX_half->p_AAdot,
				pX_half->p_n_minor,

				pX_target->p_AAdot,
				pX_target->p_vie,
				pX_target->p_v_n
				);
			Call(cudaThreadSynchronize(), "cudaTS kernelUpdate_v ");
			
			kernelUpdateAz << <numTilesMinor, threadsPerTileMinor >> >(
				(iSubstep == SUBCYCLES-1)?0.5*SUBSTEP:SUBSTEP,
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
			
			evaltime += 0.5*SUBSTEP;
		};

		kernelPushAzInto_dest <<<numTilesMinor, threadsPerTileMinor>>>(
			pX_target->p_AAdot,
			p_Az);
		Call(cudaThreadSynchronize(), "cudaTS PushAzIntoDest");

	}; // whether Backward or Leapfrog

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
		pX_target->p_szPBC_triminor
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

	printf("Done step %d from time %1.8E length %1.2E\n\n", runs, evaltime, TIMESTEP);

	// For graphing Lap Az:
	cudaMemcpy(temp_array_host, p_LapAz, sizeof(f64)*NMINOR, cudaMemcpyDeviceToHost);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Elapsed time : %f ms\n", elapsedTime);

	runs++;
}

void PerformCUDA_Revoke()
{

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

	CallMAC(cudaFree(p_Az));
	CallMAC(cudaFree(p_AzNext));
	CallMAC(cudaFree(p_LapAz));
	CallMAC(cudaFree(p_LapAzNext));
	CallMAC(cudaFree(p_LapCoeffself));
	CallMAC(cudaFree(p_LapJacobi));
	CallMAC(cudaFree(p_Jacobi_x));
	CallMAC(cudaFree(p_epsilon));
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

	CallMAC(cudaFree(p_n_shards));
	CallMAC(cudaFree(p_n_shards_n));
	CallMAC(cudaFree(NT_addition_rates_d));
	CallMAC(cudaFree(p_denom_i));
	CallMAC(cudaFree(p_denom_e));
	CallMAC(cudaFree(p_temp1));
	CallMAC(cudaFree(p_temp2));
	CallMAC(cudaFree(p_temp3));
	CallMAC(cudaFree(p_temp4));
	CallMAC(cudaFree(p_coeff_of_vez_upon_viz));
	
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
}

#include "kernel.cu"