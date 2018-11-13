

// Version 0.1:

// First draft, getting it to compile.


#pragma once

#include <math.h>
#include <time.h>
#include <stdio.h>

#include "FFxtubes.h"
#include "cuda_struct.h"
#include "cusyst.cu"
#include "flags.h"
#include "kernel.h"
#include "mesh.h"

// This is the file for CUDA host code.


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
__device__ real * p_summands, *p_Iz0_summands, *p_Iz0_initial,*p_scratch_d;
f64 * p_summands_host, *p_Iz0_summands_host, *p_Iz0_initial_host;
__device__ f64 * p_temp1, *p_temp2, *p_temp3,*p_temp4, *p_temp5, *p_temp6;
f64 * p_temphost1, *p_temphost2, *p_temphost3, *p_temphost4, *p_temphost5, *p_temphost6;
__device__ nn *p_nn_ionrec_minor;

__device__ f64_vec3 * p_MAR_neut, *p_MAR_ion, *p_MAR_elec;
__device__ f64 * p_Az, *p_LapAz, *p_LapCoeffself, *p_Azdot0, *p_gamma, *p_LapJacobi,
*p_Jacobi_x, *p_epsilon, *p_LapAzNext,
*p_Integrated_div_v_overall,
*p_div_v_neut, *p_div_v, *p_div_v_overall, *p_ROCAzdotduetoAdvection,
*p_ROCAzduetoAdvection, *p_AzNext;
__device__ species3 *p_nu_major;
__device__ f64_vec2 * p_GradAz, *p_GradTe;
__device__ ShardModel *p_n_shards, *p_n_shards_n;
__device__ NTrates *NTadditionrates;

long numReverseJzTriangles =

//f64 Tri_n_n_lists[NMINOR][6],Tri_n_lists[NMINOR][6];
// Not clear if I ended up using Tri_n_n_lists - but it was a workable way if not.

long * address;
f64 * f64address;
size_t uFree, uTotal;
extern real evaltime;

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
	Call(cudaGetSymbolAddress((void **)(&T2address), Anticlockwise),
		"cudaGetSymbolAddress((void **)(&T2address),Anticlockwise)");
	Call(cudaMemcpy(T2address, &anticlock2, sizeof(f64_tens2), cudaMemcpyHostToDevice),
		"cudaMemcpy( T2address, &anticlock2, sizeof(f64_tens2),cudaMemcpyHostToDevice) U");
	// Note that objects appearing in constant memory must have empty constructor & destructor.

	f64_tens2 clock2;
	clock2.xx = cos(FULLANGLE);
	clock2.xy = sin(FULLANGLE);
	clock2.yx = -sin(FULLANGLE);
	clock2.yy = cos(FULLANGLE);
	Call(cudaGetSymbolAddress((void **)(&T2address), Clockwise),
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

	f64 over_sqrt_m_ion_ = 1.0 / sqrt(m_i_);
	f64 over_sqrt_m_e_ = 1.0 / sqrt(m_e_);
	f64 over_sqrt_m_neutral_ = 1.0 / sqrt(m_n_);
	Set_f64_constant(over_sqrt_m_ion, over_sqrt_m_ion_);
	Set_f64_constant(over_sqrt_m_e, over_sqrt_m_e_);
	Set_f64_constant(over_sqrt_m_neutral, over_sqrt_m_neutral_);

	Set_f64_constant(FRILL_CENTROID_OUTER_RADIUS_d, OutermostFrillCentroidRadius);
	Set_f64_constant(FRILL_CENTROID_INNER_RADIUS_d, InnermostFrillCentroidRadius);

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
	CallMAC(cudaMalloc((void **)&p_div_v_neut, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_div_v, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_div_v_overall, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_ROCAzdotduetoAdvection, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_ROCAzduetoAdvection, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_GradAz, NMINOR * sizeof(f64_vec2)));
	CallMAC(cudaMalloc((void **)&p_GradTe, NMINOR * sizeof(f64_vec2)));
	
	CallMAC(cudaMalloc((void **)&p_n_shards, NUMVERTICES * sizeof(ShardModel)));
	CallMAC(cudaMalloc((void **)&p_n_shards_n, NUMVERTICES * sizeof(ShardModel)));
	CallMAC(cudaMalloc((void **)&NTadditionrates, NMINOR * sizeof(NTrates)));


	CallMAC(cudaMalloc((void **)&p_temp1, NMINOR* sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_temp2, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_temp3, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_temp4, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_temp5, NMINOR * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_temp6, NMINOR * sizeof(f64)));
	p_temphost1 = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_temphost2 = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_temphost3 = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_temphost4 = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_temphost5 = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_temphost6 = (f64 *)malloc(numTilesMinor * sizeof(f64));
	printf("temphost allocated.\n");
	if (p_temphost6 == 0) { printf("p6 == 0"); }
	else { printf("p6 != 0"); };
	getch();
	
	CallMAC(cudaMalloc((void **)&p_summands, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Iz0_summands, numTilesMinor * sizeof(f64)));
	CallMAC(cudaMalloc((void **)&p_Iz0_initial, numTilesMinor * sizeof(f64)));

	p_summands_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_Iz0_summands_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	p_Iz0_initial_host = (f64 *)malloc(numTilesMinor * sizeof(f64));
	
 	
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

void cuSyst::PerformCUDA_Advance(const cuSyst * pX_target, const cuSyst * pX_half)
{
	long iMinor, iSubstep;
	f64 Iz_prescribed;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// inauspicious start: overall v has to be split into 2 routines
	kernelCalculateOverallVelocitiesVertices<<<numTilesMajor, threadsPerTileMajor>>> (
		this->p_vie + BEGINNING_OF_CENTRAL,
		this->p_v_n+ BEGINNING_OF_CENTRAL,
		this->p_n_major,
		this->p_v_overall_minor + BEGINNING_OF_CENTRAL
		);
	Call(cudaThreadSynchronize(), "cudaTS calculate overall velocities");

	kernelAverageOverallVelocitiesTriangles <<<numTriTiles, threadsPerTileMinor>>> (
		this->p_v_overall_minor + BEGINNING_OF_CENTRAL,
		this->p_v_overall_minor,
		this->p_info,
		this->p_tri_corner_index,
		this->p_tri_periodic_corner_flags 
		);
	Call(cudaThreadSynchronize(), "cudaTS avg overall v to triangles");

//	pX_half->ZeroData(); // Is this serving any purpose?

	kernelAdvectPositions_CopyTris<<<numTilesMinor,threadsPerTileMinor>>>(
		0.5*TIMESTEP, 
		this->p_info,
		pX_half->p_info, 
		this->p_v_overall_minor
	);
	Call(cudaThreadSynchronize(), "cudaTS AdvectPositions_CopyTris");

	kernelAverage_n_T_x_to_tris <<<numTriTiles, threadsPerTileMinor>>>(
		this->p_n_minor,
		this->p_n_major,
		this->p_T_minor,
		this->p_info,
		this->p_tri_corner_index,
		this->p_tri_periodic_corner_flags
		); // call before CreateShardModel 
	Call(cudaThreadSynchronize(), "cudaTS average nTx");
		
	kernelCreateShardModelOfDensities_And_SetMajorArea<<<numTilesMajor,threadsPerTileMajor>>>(
		this->p_info,
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

	kernelInferMinorDensitiesFromShardModel<<<numTilesMinor,threadsPerTileMinor>>>(
		this->p_info,
		this->p_n_minor,
		p_n_shards,
		p_n_shards_n,
		this->p_tri_corner_index,
		this->p_who_am_I_to_corner);
	Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities");
	
	kernelCalculateUpwindDensity_tris<<<numTriTiles,threadsPerTileMinor>>>(
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
	cudaMemset(p_div_v_neut, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_div_v, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(NTadditionrates, 0, sizeof(NTrates)*NUMVERTICES);

	kernelAccumulateAdvectiveMassHeatRate<<<numTilesMajor,threadsPerTileMajor>>>(
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
		
		NTadditionrates,
		p_div_v,
		p_div_v_neut,
		p_Integrated_div_v_overall);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateAdvectiveMassHeatRate");

	kernelCalculateNu_eHeartNu_iHeart_nu_nn_visc<<<numTilesMajorClever, threadsPerTileMajorClever>>>(
		this->p_info,
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL,
		p_nu_major);
	Call(cudaThreadSynchronize(), "cudaTS CalculateNu");

	kernelAccumulateDiffusiveHeatRateAndCalcIonisation<<<numTilesMajorClever,threadsPerTileMajorClever>>>(
		0.5*TIMESTEP, 
		this->p_info,
		this->p_izNeigh_vert,
		this->p_szPBCneigh_vert,
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices
		this->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
		p_nu_major,
		NTadditionrates);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate");
	// To increase the efficiency we want to make a clever 2nd set of major tiles of size 192. Also try 256, 384.

	kernelAdvanceDensityAndTemperature<<<numTilesMajor,threadsPerTileMajor>>>(
		0.5*TIMESTEP, 
		this->p_info+BEGINNING_OF_CENTRAL,
		this->p_n_major, 
		this->p_T_minor + BEGINNING_OF_CENTRAL,
		NTadditionrates,
		this->p_n_minor,
		this->p_T_minor,
		this->p_vie,
		this->p_v_n,

		p_div_v_neut, p_div_v,
		p_Integrated_div_v_overall,
		this->p_AreaMajor,
		pX_half->p_n_major, 
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL
		);
	Call(cudaThreadSynchronize(), "cudaTS Advance_n_and_T");
	
	kernelAverage_n_T_x_to_tris<<<numTriTiles,threadsPerTileMinor>>>(
		pX_half->p_n_minor,
		pX_half->p_n_major,
		pX_half->p_T_minor,
		pX_half->p_info,
		pX_half->p_tri_corner_index,
		pX_half->p_tri_periodic_corner_flags
		); // call before CreateShardModel 
	Call(cudaThreadSynchronize(), "cudaTS average nTx 2");
	
	// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	// We are going to want to introduce 2nd degree approx to get n,T desired on tris.
	// Now let's set up the accel move to half-time which will provide us input of v to the full n,T move.
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	// For now went for "advective change in v" where we divide d/dt Nv by N_derivsyst
	cudaMemset(p_MAR_neut, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_ion, 0, sizeof(f64_vec3)*NMINOR);
	cudaMemset(p_MAR_elec, 0, sizeof(f64_vec3)*NMINOR);

	kernelCreate_pressure_gradT_and_gradA_LapA_CurlA_minor<<<numTriTiles,threadsPerTileMinor>>>(

		this->p_info,
		this->p_T_minor,
		this->p_AAdot,

		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_izNeigh_TriMinor,
		this->p_szPBC_triminor,

		p_MAR_neut, p_MAR_ion, p_MAR_elec,

		p_n_shards,				// this kernel is for i+e only

		p_GradTe,
		p_GradAz,
		p_LapAz,
		this->p_B,
		this->p_AreaMinor
	);
	Call(cudaThreadSynchronize(), "cudaTS kernelCreate_pressure_gradT_and_gradA_LapA_CurlA_minor");
	
	kernelCreate_momflux_minor<<<numTriTiles, threadsPerTileMinor>>>(

		this->p_info,
		this->p_vie,
		this->p_v_overall_minor,

		this->p_izTri_vert,
		this->p_szPBCtri_vert,
		this->p_izNeigh_TriMinor,
		this->p_szPBC_triminor,

		p_MAR_neut, p_MAR_ion, p_MAR_elec,
		p_n_shards
	);
	Call(cudaThreadSynchronize(), "cudaTS kernelCreate_momflux_minor");
	
	kernelNeutral_pressure_and_momflux<<<numTriTiles, threadsPerTileMinor>>>(
		this->p_info,
		this->p_v_n,
		p_n_shards_n,
		this->p_v_overall_minor,
		p_MAR_neut
	);
	Call(cudaThreadSynchronize(), "cudaTS kernelNeutral_pressure_and_momflux");

	//////////////////////////////////////////////////////////////////////////
	// Even more shards!:

	kernelCreateShardModelOfDensities_And_SetMajorArea << <numTilesMajor, threadsPerTileMajor >> > (
		pX_half->p_info,
		pX_half->p_n_minor,
		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert, // could be more intelligent way than storing and copying 3x
		p_n_shards,
		p_n_shards_n,
//		p_Tri_n_lists,
	//	p_Tri_n_n_lists,
		pX_half->p_AreaMajor);
	Call(cudaThreadSynchronize(), "cudaTS CreateShardModel pX_half");
	
	kernelInferMinorDensitiesFromShardModel<<<numTilesMinor,threadsPerTileMinor>>>(
		pX_half->p_info,
		pX_half->p_n_minor,
		p_n_shards,
		p_n_shards_n,
		pX_half->p_tri_corner_index,
		pX_half->p_who_am_I_to_corner);// (At the moment just repopulating tri minor n.)
	Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities pX_half");
	
	f64 neg_Iz_per_triangle = Iz_prescribed / (f64) numReverseJzTriangles;
	Set_f64_constant(negative_Iz_per_triangle, &neg_Iz_per_triangle);
	// Get suitable v to use for resistive heating:
	kernelPopulateOhmsLaw <<<numTilesMinor, threadsPerTileMinor>>>(
		0.5*TIMESTEP,

		this->p_info,
		p_AdditionalMomRates,
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

		p_vn0,
		p_v0,
		p_OhmsCoeffs,
		pX_half->p_AAdot, // intermediate value

		p_Iz0,
		p_sigma_Izz,
		false); // bFeint

	Call(cudaThreadSynchronize(), "cudaTS AccelerateOhms 1");

	cudaMemcpy(p_Iz0_summands_host, p_Iz0, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
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
	Ez_strength_ = (Iz_prescribed - Iz0) / sigma_Izz;
	Set_f64_constant(Ez_strength, Ez_strength_);

	// Update velocities and Azdot:
	kernelUpdateVelocityAndAzdotAndAz<<<numTilesMinor, threadsPerTileMinor>>>(
		0.5*TIMESTEP,
		p_vn0,
		p_v0,
		p_OhmsCoeffs,
		pX_half->p_AAdot,
		pX_half->p_vie,
		pX_half->p_v_n,
		p_GradAz,
		this->p_v_overall_minor // to anti-advect
	);
	Call(cudaThreadSynchronize(), "cudaTS AccelerateUpdate 1");

	// ))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
	// 
	// Now do the n,T,x advance to pDestmesh:
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	kernelCalculateOverallVelocitiesVertices<<<numTilesMajor, threadsPerTileMajor>>>(
		pX_half->p_vie + BEGINNING_OF_CENTRAL,
		pX_half->p_v_n + BEGINNING_OF_CENTRAL,
		pX_half->p_n_major,
		pX_half->p_v_overall_minor + BEGINNING_OF_CENTRAL
		);
	Call(cudaThreadSynchronize(), "cudaTS calculate overall velocities 22");
	kernelAverageOverallVelocitiesTriangles<<<numTriTiles, threadsPerTileMinor>>>(
		pX_half->p_v_overall_minor + BEGINNING_OF_CENTRAL,
		pX_half->p_v_overall_minor,
		pX_half->p_info,
		pX_half->p_tri_corner_index,
		pX_half->p_tri_periodic_corner_flags // questionable wisdom of repeating info in 3 systems
		);
	Call(cudaThreadSynchronize(), "cudaTS avg overall v to triangles 22");

	kernelAdvectPositions_CopyTris<<<numTilesMinor, threadsPerTileMinor>>>(
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
		pX_half->p_trineighindex,
		pX_half->p_who_am_I_to_corner,
		pX_half->p_PBCtriminor, // carries 6, we only need 3
		pX_half->p_n_upwind_minor);
	Call(cudaThreadSynchronize(), "cudaTS CalculateUpwindDensity_tris pX_half");

	cudaMemset(p_Integrated_div_v_overall, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_div_v_neut, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_div_v, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(NTadditionrates, 0, sizeof(NTrates)*NUMVERTICES);

	kernelAccumulateAdvectiveMassHeatRate<<<numTilesMajor, threadsPerMajorTile>>>(
		pX_half->p_info,
		pX_half->p_n_major,
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL,

		pX_half->p_n_upwind_minor,
		pX_half->p_vie,
		pX_half->p_v_n,
		pX_half->p_v_overall_minor,
		pX_half->p_T_minor, // may or may not overlap source: don't we only use from tris? so not overlap

		p_NTadditionrates,
		p_div_v,
		p_div_v_n,
		p_Integrated_div_v_overall);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateAdvectiveMassHeatRate pX_half");

	kernelCalculateNu_eHeartNu_iHeart_nu_nn_visc<<<numTilesMajorClever, threadsPerTileMajorClever>>>(
		pX_half->p_info,
		pX_half->p_n_major,
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL,
		p_nu_major);
	Call(cudaThreadSynchronize(), "cudaTS CalculateNu pX_half");

	kernelAccumulateDiffusiveHeatRateAndCalcIonisation<<<numTilesMajorClever, threadsPerTileMajorClever>>>(
		0.5*TIMESTEP,
		pX_half->p_info,
		pX_half->p_izNeigh_vert,
		pX_half->p_szPBCneigh_vert,
		pX_half->p_n_major,
		pX_half->p_T_minor + BEGINNING_OF_CENTRAL, // using vert indices
		pX_half->p_B + BEGINNING_OF_CENTRAL, // NEED POPULATED
		p_nu_major,
		NTadditionrates);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate pX_half");
	// To increase the efficiency we want to make a clever 2nd set of major tiles of size 192. Also try 256, 384.

	kernelAdvanceDensityAndTemperature<<<numTilesMajor, threadsPerTileMajor>>>(
		0.5*TIMESTEP,
		this->p_info + BEGINNING_OF_CENTRAL,
		this->p_n_major,
		this->p_T_minor + BEGINNING_OF_CENTRAL,
		NTadditionrates,
		pX_half->p_n_minor,  // ?
		pX_half->p_T_minor,  // ?
		p_div_v_neut, p_div_v,
		p_Integrated_div_v_overall,
		pX_half->p_AreaMajor,
		pX_target->p_n_major, pX_target->p_T_minor + BEGINNING_OF_CENTRAL
		);
	Call(cudaThreadSynchronize(), "cudaTS Advance_n_and_T 233");
	
	// QUESTION QUESTION : What should params be there?

	kernelAverage_n_T_x_to_tris<<<numTriTiles, threadsPerTileMinor>>>(
		pX_target->p_n_minor,
		pX_target->p_n_major,
		pX_target->p_T_minor,
		pX_target->p_info
		); // call before CreateShardModel 
	Call(cudaThreadSynchronize(), "cudaTS average nTx 233");

	 // Now set up inputs such as AMR with advective momflux and aTP, and B
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	// [ do Az advance above when we advance Azdot. ]

	memset(AdditionalMomRates, 0, sizeof(three_vec3)*NMINOR);
	
	// Now notice we take a grad Azdot but Azdot has not been defined except from time t_k!!
	kernelCreate_pressure_gradT_and_gradA_LapA_CurlA_minor<<<numTriTiles, threadsPerTileMinor>>>(
		pX_half->p_info,
		pX_half->p_T_minor,
		pX_half->p_AAdot,

		pX_half->p_izTri_verts,
		pX_half->p_szPBC_verts,
		pX_half->p_izNeighTriMinor,
		pX_half->p_szPBCtriminor,

		p_AdditionalMomrates,
		p_n_shards,				// this kernel is for i+e only

		p_GradTe,
		p_GradAz,
		p_LapAz,
		pX_half->p_B
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelCreate_pressure_gradT_and_gradA_LapA_CurlA_minor pX_half");

	kernelCreate_momflux_minor<<<numTriTiles, threadsPerTileMinor>>>(
		pX_half->p_info,
		pX_half->p_vie,
		pX_half->p_v_overall_minor,

		pX_half->p_izTri_vert,
		pX_half->p_szPBCtri_vert,
		pX_half->p_izNeigh_TriMinor,
		pX_half->p_szPBC_triminor,

		p_AdditionalMomrates,
		p_n_shards
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelCreate_momflux_minor pX_half" );

	kernelNeutral_pressure_and_momflux <<<numTriTiles, threadsPerTileMinor>>>(
		pX_half->p_info,
		pX_half->p_v_n,
		p_n_shards_n,
		pX_half->p_v_overall_minor,
		p_AdditionalMomRates
		);
	Call(cudaThreadSynchronize(), "cudaTS kernelNeutral_pressure_and_momflux pX_half");

	//pHalfMesh->Add_ViscousMomentumFluxRates(AdditionalMomRates);

	//////////////////////////////////////////////////////////////////////////
	// Even more shards!:

	kernelCreateShardModelOfDensities_And_SetMajorArea<<<numTilesMajor, threadsPerTileMajor>>>(
		pX_target->p_info,
		pX_target->p_n_minor,
		pX_target->p_izTri_vert,
		pX_target->p_szPBCtri_vert, // could be more intelligent way than storing and copying 3x
		p_n_shards,
		p_n_shards_n,
//		p_Tri_n_lists,
//		p_Tri_n_n_lists,
		pX_target->p_AreaMajor);
	Call(cudaThreadSynchronize(), "cudaTS CreateShardModel pX_target");

	kernelInferMinorDensitiesFromShardModel<<<numTriTiles, threadsPerTileMinor>>>(
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

		kernelGetLapCoeffs<<<numTriTiles, threadsPerTileMinor>>>(
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
				this->info,
				pX_target->info,
				this->p_n_minor,
				pX_target->p_n_minor,
				this->p_T_minor,
				pX_target->p_T_minor,
				this->p_B,
				pX_target->p_B,
				pX_half->info,
				pX_half->p_n_minor,
				pX_half->p_T_minor,
				pX_half->p_B
				);
			Call(cudaThreadSynchronize(), "cudaTS InterpolateVars");

			//pHalfMesh->GetLapFromCoeffs(Az_array, LapAzArray);

			// NOTICE # BLOCKS -- THIS SHOULD ALSO APPLY WHEREVER WE DO SIMILAR THING LIKE WITH MOMFLUX.

			kernelGetLap_minor<<<numTriTiles,threadsPerTileMinor>>>(
				pX_half->p_info, 
				Az_array, 
				pX_half->p_izTri_vert,
				pX_half->p_izNeigh_TriMinor,
				pX_half->p_szPBC_vert,
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
			Iz_prescribed = GetIzPrescribed(evaltime + 0.5*SUBSTEP);
			neg_Iz_per_triangle = Iz_prescribed_ / (f64)numReverseJzTriangles;
			Set_f64_constant(negative_Iz_per_triangle, &neg_Iz_per_triangle);
			// Electrons travel from cathode to anode so Jz is down in filament,
			// up around anode.

			if (iSubstep == 0) {

				kernelPopulateOhmsLaw <<<numTilesMinor, threadsPerTileMinor>>>(
					SUBSTEP,// ROCAzdotduetoAdvection, 
					pX_half->p_info,
					p_AdditionalMomRates,
					pX_half->p_B,
					p_LapAz,
					p_GradAz,
					p_GradTe,

					pX_half->p_n_minor,
					pX_half->p_T_minor, // minor : is it populated?
					this->p_vie,
					this->p_v_n,
					this->p_AAdot, //	inputs
					pX_half->p_AreaMinor, // pop'd????????? interp?

					p_vn0,
					p_v0,
					p_OhmsCoeffs,
					pX_half->p_AAdot, // intermediate value ............................
								  // .....................................................
					p_Iz0_blocks,
					p_sigma_Izz_blocks,
					true); // bFeint ---- SHOULD THIS BE TRUE?
				
				Call(cudaThreadSynchronize(), "cudaTS kernelPopulateOhmsLaw ");
				
				cudaMemcpy(p_Iz0_summands_host, p_Iz0_summands, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
				cudaMemcpy(p_summands_host, p_Sigma_Izz_summands, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
				Iz0 = 0.0; Sigma_Izz = 0.0;
				for (iBlock = 0; iBlock < numTilesMinor; iBlock++)
				{
					Iz0 += p_Iz0_summands_host[iBlock];
					Sigma_Izz += p_summands_host[iBlock];
				}
				EzStrength_ = (Iz_prescribed - Iz0) / Sigma_Izz;
				Set_f64_constant(EzStrength, EzStrength_);

				// _____________________________________________________________
				
				kernelUpdateVelocityAndAzdotAndAz<<<numTilesMinor,threadsPerTileMinor>>>(
					f64 h_use,
					f64_vec3 * __restrict__ p_vn0,
					v4 * __restrict__ p_v0,
					OhmsCoeffs * __restrict__ p_OhmsCoeffs,
					AAdot * __restrict__ p_AAzdot_update,

					v4 * __restrict__ p_vie_out,
					f64_vec3 * __restrict__ p_vn_out,
					f64 * __restrict__ p_GradAz,
					f64_vec2 * __restrict__ p_v_overall_minor
				);
				Call(cudaThreadSynchronize(), "cudaTS kernelUpdate_v ");
				
				kernelCreateSeed <<<numTilesMinor, threadsPerTileMinor>>> (
						Az_array, this->p_AAdot,
						0.5*SUBSTEP,
						p_Azdot0, p_gamma, p_LapAz,
						Az_array_next);
				Call(cudaThreadSynchronize(), "cudaTS Create Seed 1");
					 
			} else {	// half-substep: // and it's a FEINT
				pDestMesh->Accelerate2018(
					SUBSTEP,// ROCAzdotduetoAdvection, 
					pHalfMesh, 
					pDestMesh, 
					evaltime + 0.5*SUBSTEP, 
					true);
				// ?????????????????????????????

				kernelCreateSeed<<<numTilesMinor, threadsPerTileMinor>>>(
						Az_array, pX_targ->p_AAdot,
						0.5*SUBSTEP,
						p_Azdot0, p_gamma, p_LapAz,
						Az_array_next);

				Call(cudaThreadSynchronize(), "cudaTS Get Lap Jacobi 1");
			};
				
				// JLS:
				//Az_array_next[iMinor] = Az_array[iMinor] + 0.5*SUBSTEP*pDestMesh->pData[iMinor].Azdot + 0.5*SUBSTEP * Azdot0[iMinor] + 0.5*SUBSTEP * gamma[iMinor] * LapAzArray[iMinor];
				
			f64 sum_eps_deps_by_dbeta, sum_depsbydbeta_sq, sum_eps_eps, depsbydbeta;
			printf("\nJLS [beta L2eps]: ");
			long iMinor;
			f64 beta, L2eps;
			Triangle * pTri;
			for (iIteration = 0; iIteration < iterations; iIteration++)
			{
					// 1. Create regressor:
					// Careful with major vs minor + BEGINNING_OF_CENTRAL:

				kernelGetLap_minor<<<numTriTiles, threadsPerTileMinor>>>(
					pX_half->p_info,
					Az_array_next,
					pX_half->p_izTri_verts,
					pX_half->p_izNeigh_TriMinor,
					pX_half->p_szPBC_verts,
					pX_half->p_szPBCtriminor,
					p_LapAz_next
					);
				Call(cudaThreadSynchronize(), "cudaTS GetLap Az JLS 1");

					//
					//pTri = T;
				kernelCreateEpsilonAndJacobi <<<numTilesMinor, threadsPerTileMinor>>> 
					(Az_array_next,
					Az_array,
						Azdot0, gamma, LapCoeffself, Lap_Aznext, epsilon, Jacobi_x);

				Call(cudaThreadSynchronize(), "cudaTS CreateEpsAndJacobi 1");

				kernelGetLap_minor<<<numTriTiles, threadsPerTileMinor>>>(
					pX_half->p_info,
					Jacobi_x,
					pX_half->p_izTri_verts,
					pX_half->p_izNeigh_TriMinor,
					pX_half->p_szPBC_verts,
					pX_half->p_szPBCtriminor,
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
					//		depsbydbeta = 0.0; //  Lap_Jacobi[iMinor]; // try ignoring
					//	}
					//	else {
					//		depsbydbeta = (Jacobi_x[iMinor] - h_use * gamma[iMinor] * Lap_Jacobi[iMinor]);
					//	};
					//	sum_eps_deps_by_dbeta += epsilon[iMinor] * depsbydbeta;
					//	sum_depsbydbeta_sq += depsbydbeta * depsbydbeta;
					//	sum_eps_eps += epsilon[iMinor] * epsilon[iMinor];
					//	++pTri;
					//};
				kernelAccumulateSummands << <numTilesMinor, threadsPerTileMinor >> > (
						this->p_info,
						epsilon, Jacobi_x, Lap_Jacobi, gamma,
						p_sum_eps_deps_by_dbeta,
						p_sum_depsbydbeta_sq,
						p_sum_eps_eps);

				Call(cudaThreadSynchronize(), "cudaTS AccumulateSummands 1");
				sum_eps_deps_by_dbeta = 0.0;
				sum_depsbydbeta_sq = 0.0;
				sum_eps_eps = 0.0;
				for (iTile = 0; iTile < numTilesMinor; iTile++)
				{
					sum_eps_deps_by_dbeta += p_sum_eps_deps_by_dbeta[iTile];
					sum_depsbydbeta_sq += p_sum_depsbydbeta_sq[iTile];
					sum_eps_eps += p_sum_eps_eps[iTile];
				}
				beta = -sum_eps_deps_by_dbeta / sum_depsbydbeta_sq;
				L2eps = sqrt(sum_eps_eps / (real)NMINOR);
				printf(" [ %1.4f %1.2E ] ", beta, L2eps);

				kernelAdd <<<numTilesMinor, threadsPerTileMinor>>> (Az_array_next, beta, Jacobi_x);
					//for (iMinor = 0; iMinor < NMINOR; iMinor++)
					//	Az_array_next[iMinor] += beta * Jacobi_x[iMinor];

				Call(cudaThreadSynchronize(), "cudaTS Add 1");

					// Try resetting frills here and ignoring in calculation:
				kernelResetFrillsAz <<<numTriTiles, threadsPerTileMinor>>> (
						this->p_info, Az_array_next);
					//					pTri = T;
						//				for (iMinor = 0; iMinor < NUMTRIANGLES; iMinor++)
							//			{
					//						if ((pTri->u8domain_flag == INNER_FRILL) ||
						//						(pTri->u8domain_flag == OUTER_FRILL))
							//					Az_array_next[iMinor] = Az_array_next[pTri->neighbours[0] - T];
					//						++pTri;
						//				};
				To match the simulation.cpp this needs to be called before, as it appeared in AntiadvectAz routine before.
			};

			printf("\n\n");

				// That was:
			//	JLS_for_Az_bwdstep(4, SUBSTEP); // populate Az_array with k+1 values
			cudaMemcpy(Az_array, Az_array_next, sizeof(f64)*NMINOR,cudaMemcpyDeviceToDevice);
			kernelGetLap_minor<<<numTriTiles, threadsPerTileMinor>>>(
				pX_half->p_info,
				Az_array,
				pX_half->p_izTri_verts,
				pX_half->p_izNeigh_TriMinor,
				pX_half->p_szPBC_verts,
				pX_half->p_szPBCtriminor,
				p_LapAz
				);
			Call(cudaThreadSynchronize(), "cudaTS GetLap Az JLS 2");
			
			if (iSubstep == 0) {
				this->Accelerate2018(SUBSTEP, pHalfMesh, pDestMesh, evaltime + 0.5*SUBSTEP, false); // Lap Az now given.
			} else {
				// Update v:
				pDestMesh->Accelerate2018(SUBSTEP, pHalfMesh, pDestMesh, evaltime + 0.5*SUBSTEP, false); // Lap Az now given.
			};
				// Why we do not pass it back and forth? Can't remember.
		}; // next substep
			
		for (iMinor = 0; iMinor < NMINOR; iMinor++)
			pDestMesh->pData[iMinor].Az = Az_array[iMinor];

		evaltime += 0.5*SUBSTEP;
			// more advanced implicit could be possible and effective.
		
	} else {
		
		kernel_Populate_Az_Array_advance_Az<<<numTilesMinor, threadsPerTileMinor>>>(
			0.5*SUBSTEP,
			this->p_AAdot,
			p_ROCAzduetoAdvection,
			p_Az,
		);
		kernel_ResetFrillsAz<<<numTilesMinor, threadsPerTileMinor>>>(
			this->p_info,
			AzArray,
			this->p_indexneigh_tri,
			);

		// Create_A_from_advance(0.5*SUBSTEP, ROCAzduetoAdvection, Az_array); // from *this
		
		for (iSubstep = 0; iSubstep < SUBCYCLES; iSubstep++)
		{
			evaltime += 0.5*SUBSTEP;
			kernelInterpolateVarsAndPositions<<<numTilesMinor, threadsPerTileMinor>>>(
				(evaltime - starttime) / TIMESTEP,
				this->info,
				pX_target->info,
				this->p_n_minor,
				pX_target->p_n_minor,
				this->p_T_minor,
				pX_target->p_T_minor,
				this->p_B,
				pX_target->p_B,
				pX_half->info,
				pX_half->p_n_minor,
				pX_half->p_T_minor,
				pX_half->p_B
				);
			Call(cudaThreadSynchronize(), "cudaTS InterpolateVars");
			// let n,T,x be interpolated on to pHalfMesh. B remains what we populated there.

			// ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]][[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
			// Recalculate areas, or tween them, would make good sense as well.

			// It might be nice to tween the thermal pressure grad(nT) as well. And logical:
			// InterpolateArrays(AdditionalMomRates_start, AdditionalMomRates_end, AdditionalMomRates, (evaltime - starttime) / TIMESTEP);
			// Have a look how AMR is created.
			kernelGetLap_minor <<<numTriTiles, threadsPerTileMinor>>>(
				pX_half->p_info,
				Az_array,
				pX_half->p_izTri_verts,
				pX_half->p_izNeigh_TriMinor,
				pX_half->p_szPBC_verts,
				pX_half->p_szPBCtriminor,
				p_LapAz
				);
			Call(cudaThreadSynchronize(), "cudaTS GetLap Az Leapfrog 1");

			// evaltime + 0.5*SUBSTEP used for setting EzStrength://
			Iz_prescribed = GetIzPrescribed(evaltime + 0.5*SUBSTEP);
			f64 neg_Iz_per_triangle = Iz_prescribed / (f64)numReverseJzTriangles;
			Set_f64_constant(negative_Iz_per_triangle, &neg_Iz_per_triangle);

			if (iSubstep % 2 == 0) {
				
				kernelPopulateOhmsLaw<<<numTilesMinor, threadsPerTileMinor>>>(
					SUBSTEP,// ROCAzdotduetoAdvection, 
					pX_half->p_info,
					p_AdditionalMomRates,
					pX_half->p_B,
					p_LapAz,
					p_GradAz,
					p_GradTe,

					pX_half->p_n_minor,
					pX_half->p_T_minor, // minor : is it populated?
					this->p_vie,
					this->p_v_n,
					this->p_AAdot, // Note bene
					pX_half->p_AreaMinor, // pop'd????????? interp?

					p_vn0,
					p_v0,
					p_OhmsCoeffs,
					pX_half->p_AAdot, // intermediate value ............................
									
					p_Iz0_block,
					p_sigma_Izz_block,
					false); // bFeint

				Call(cudaThreadSynchronize(), "cudaTS kernelPopulateOhmsLaw ");

				cudaMemcpy(p_Iz0_summands_host, p_Iz0_block, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
				cudaMemcpy(p_summands_host, p_Sigma_Izz_block, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
				Iz0 = 0.0; Sigma_Izz = 0.0;
				for (int iBlock = 0; iBlock < numTilesMinor; iBlock++)
				{
					Iz0 += p_Iz0_summands_host[iBlock];
					Sigma_Izz += p_summands_host[iBlock];
				}
				EzStrength_ = (Iz_prescribed - Iz0) / Sigma_Izz;
				Set_f64_constant(EzStrength, EzStrength_);

				// _____________________________________________________________

				kernelUpdateVelocityAndAzdotAndAz <<<numTilesMinor, threadsPerTileMinor>>>(
					SUBSTEP,
					p_vn0,
					p_v0,
					p_OhmsCoeffs,
					pX_half->p_AAdot,
					
					pX_target->p_AAdot,
					pX_target->p_vie,
					pX_target->p_vn_minor,
					
					p_GradAz, /// ///////////////////////////////////////////////
					p_v_overall_minor
					);
				Call(cudaThreadSynchronize(), "cudaTS kernelUpdate_v ");
				
				// what it does to Az? keep track
			}
			else {
				
				kernelPopulateOhmsLaw << <numTilesMinor, threadsPerTileMinor >> >(
					SUBSTEP,// ROCAzdotduetoAdvection, 
					pX_half->p_info,
					p_AdditionalMomRates,
					pX_half->p_B,
					p_LapAz,
					p_GradAz,
					p_GradTe,

					pX_half->p_n_minor,
					pX_half->p_T_minor, // minor : is it populated?
					pX_target->p_vie,
					pX_target->p_v_n,
					pX_target->p_AAdot, // Note bene
					pX_half->p_AreaMinor, // pop'd????????? interp?

					p_vn0,
					p_v0,
					p_OhmsCoeffs,
					pX_half->p_AAdot, // intermediate value ............................
									  // .....................................................
					p_Iz0_block,
					p_sigma_Izz_block,
					false); // bFeint

				Call(cudaThreadSynchronize(), "cudaTS kernelPopulateOhmsLaw ");

				cudaMemcpy(p_Iz0_summands_host, p_Iz0_block, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
				cudaMemcpy(p_summands_host, p_Sigma_Izz_block, sizeof(f64)*numTilesMinor, cudaMemcpyDeviceToHost);
				Iz0 = 0.0; Sigma_Izz = 0.0;
				for (int iBlock = 0; iBlock < numTilesMinor; iBlock++)
				{
					Iz0 += p_Iz0_summands_host[iBlock];
					Sigma_Izz += p_summands_host[iBlock];
				}
				EzStrength_ = (Iz_prescribed - Iz0) / Sigma_Izz;
				Set_f64_constant(EzStrength, EzStrength_);

				kernelUpdateVelocityAndAzdotAndAz << <numTilesMinor, threadsPerTileMinor >> >(
					SUBSTEP,
					p_vn0,
					p_v0,
					p_OhmsCoeffs,
					pX_half->p_AAzdot_update,
					
					this->p_AAdot,
					this->p_vie_out,
					this->p_vn_out,
					
					p_GradAz, /// ///////////////////////////////////////////////
					p_v_overall_minor
					);
				Call(cudaThreadSynchronize(), "cudaTS kernelUpdate_v ");

				// what it does to Az? keep track
				// but here we deal in AzArray
			};

			if (iSubstep < SUBCYCLES - 1) {
				kernelUpdateAz<<<numTilesMinor,threadsPerTileMinor>>>(
					SUBSTEP,
					ROCAzduetoAdvection,  
					AzArray);

				kernel_ResetFrillsAz << <numTilesMinor, threadsPerTileMinor >> >(
					this->p_info,
					AzArray,
					this->p_indexneigh_tri,
					);
			} else {
				kernelFinalStepAz<<<numTilesMinor, threadsPerTileMinor>>>(
					SUBSTEP*0.5, 
					ROCAzduetoAdvection,
					AzArray,
					pX_target->p_AAdot); 
				kernel_ResetFrillsAz_in_AAdot<<<numTilesMinor, threadsPerTileMinor>>>(
					this->p_info,
					pX_target->p_AAdot,
					this->p_indexneigh_tri
					);
			};
			evaltime += 0.5*SUBSTEP;
		};
	}; // whether Backward or Leapfrog

	printf("evaltime %1.5E \n", evaltime);
	printf("-----------------\n");

	//this->AntiAdvectAzAndAdvance(h, pHalfMesh, IntegratedGradAz, pDestMesh); // Might as well subsume this in Accelerate, really
	//pHalfMesh->AntiAdvectAzAndAdvance(h*0.5, pDestMesh, GradAz, pDestMesh);

	pDestMesh->Wrap();
	printf("Done step %d from time %1.8E length %1.2E\n\n", runs, evaltime, h);

	// For graphing Lap Az:
	for (iMinor = 0; iMinor < NMINOR; iMinor++)
	{
		pDestMesh->pData[iMinor].temp.x = LapAzArray[iMinor];
	};

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Elapsed time : %f ms\n", elapsedTime);

	runs++;
}

void PerformCUDA_Revoke()
{

	CallMAC(cudaFree(p_nu_major));
	CallMAC(cudaFree(p_MAR_neut));
	CallMAC(cudaFree(p_MAR_ion));
	CallMAC(cudaFree(p_MAR_elec));
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
	CallMAC(cudaFree(p_div_v_neut));
	CallMAC(cudaFree(p_div_v));
	CallMAC(cudaFree(p_div_v_overall));
	CallMAC(cudaFree(p_ROCAzdotduetoAdvection));
	CallMAC(cudaFree(p_ROCAzduetoAdvection));
	CallMAC(cudaFree(p_GradAz));
	CallMAC(cudaFree(p_GradTe));

	CallMAC(cudaFree(p_n_shards));
	CallMAC(cudaFree(p_n_shards_n));
	CallMAC(cudaFree(NTadditionrates));
}