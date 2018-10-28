
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

__host__ bool Call(cudaError_t cudaStatus, char str[])
{
	if (cudaStatus == cudaSuccess) return false;
	printf("Error: %s\nReturned %d : %s\n",
		str, cudaStatus, cudaGetErrorString(cudaStatus));
	printf("Anykey.\n");	getch();
	return true;
}
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

// Device-accessible constants not known at compile time:
__constant__ long nBlocks, Nverts, uDataLen_d; // Nverts == numVertices
__constant__ f64_tens2 Anticlockwise, Clockwise; // use this to do rotation.


// Set from host constant definitions:
__constant__ f64 sC, kB, c, Z, e, q, m_e, m_ion, m_n,
	eoverm, qoverM, moverM, eovermc, qoverMc,
	FOURPI_Q_OVER_C, FOURPI_Q, FOURPI_OVER_C,
	NU_EI_FACTOR, // Note: NU_EI_FACTOR goes with T in eV -- !!
	nu_eiBarconst, csq, m_s,
	// New:
	FOUR_PI;

__constant__ f64 cross_s_vals_viscosity_ni_d[10], cross_s_vals_viscosity_nn_d[10],
				cross_T_vals_d[10], cross_s_vals_MT_ni_d[10];

// Set from calculations in host routine:
__constant__ f64 Nu_ii_Factor, kB_to_3halves,one_over_kB, one_over_kB_cubed,
				over_sqrt_m_ion, over_sqrt_m_e, over_sqrt_m_neutral;

__constant__ f64 four_pi_over_c_ReverseJz;
__constant__ f64 FRILL_CENTROID_OUTER_RADIUS_d,FRILL_CENTROID_INNER_RADIUS_d;
__device__ real * p_summands, *p_Iz0_summands, *p_Iz0_initial,*p_scratch_d;
__device__ f64 * p_temp1, *p_temp2, *p_temp3,*p_temp4, *p_temp5, *p_temp6;
f64 * p_temphost1, *p_temphost2, *p_temphost3, *p_temphost4, *p_temphost5, *p_temphost6;
__device__ f64_vec3 * p_MAR_neut, *p_MAR_ion, *p_MAR_elec;
__device__ nn *p_nn_ionrec_minor;
nn * p_nn_host;
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



void cuSyst::PerformCUDA_Advance(const cuSyst * pX_target, const cuSyst * pX_half)
{
	long iMinor, iSubstep;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// inauspicious start: overall v has to be split into 2 routines
	kernelCalculateOverallVelocitiesVertices<<<numTilesMajor, threadsPerTileMajor>>> (
		this->p_vie + BEGINNING_OF_CENTRAL,
		this->p_v_neut + BEGINNING_OF_CENTRAL,
		this->p_n_major,
		this->p_overall_v + BEGINNING_OF_CENTRAL
		);
	Call(cudaThreadSynchronize(), "cudaTS calculate overall velocities");

	kernelAverageOverallVelocitiesTriangles <<<numTriTiles, threadsPerTileMinor>>> (
		this->p_overall_v + BEGINNING_OF_CENTRAL,
		this->p_overall_v,
		this->p_tri_corner_index
		);
	Call(cudaThreadSynchronize(), "cudaTS avg overall v to triangles");

	pX_half->ZeroData(); // Is this serving any purpose?

	kernelAdvectPositions_CopyTris<<<numTilesMinor,threadsPerTileMinor>>>(
		0.5*TIMESTEP, 
		this->p_info,
		pHalfMesh->p_pos, 
		this->p_overall_v
		// what else is copied?
		// something we can easily copy over
		// with cudaMemcpy, even ahead of steps?
		// Is there a reason we cannot put into the above routine
		// with a split for "this is a vertex->just use its overall v"
	);
	Call(cudaThreadSynchronize(), "cudaTS AdvectPositions_CopyTris");

	kernelAverage_n_T_x_to_tris << <numTriTiles, threadsPerTileMinor >> >(
		this->p_n,
		this->p_n_major,
		this->p_T,
		this->p_info
		); // call before CreateShardModel 
	Call(cudaThreadSynchronize(), "cudaTS average nTx");
		
	kernelCreateShardModelOfDensities_And_SetMajorArea<<<numTilesMajor,threadsPerTileMajor>>>(
		this->p_info,
		this->p_n_major
		p_n_shards,
		p_n_n_shards
		);// sets n_shards_n, n_shards, Tri_n_n_lists, Tri_n_lists
	Call(cudaThreadSynchronize(), "cudaTS CreateShardModels");

	kernelInferMinorDensitiesFromShardModel<<<numTilesMinor,threadsPerTileMinor>>>(
		this->p_n,
		p_n_shards,
		p_n_n_shards);
	Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities");

	cudaMemset(p_div_v_neut, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(p_div_v, 0, sizeof(f64)*NUMVERTICES);
	cudaMemset(NTadditionrates, 0, sizeof(NTrates)*NUMVERTICES);

	kernelAccumulateAdvectiveMassHeatRate(
		p_v, NTadditionrates);
	Call(cudaThreadSynchronize(), "cudaTS AccumulateAdvectiveMassHeatRate");

	kernelCalculateNu_eHeartNu_iHeart_nu_nn_visc<<<numTilesMajorClever, threadsPerTileMajorClever>>>(
		this->p_info,
		this->p_n_major,
		this->p_T + BEGINNING_OF_CENTRAL,
		p_nu);
	Call(cudaThreadSynchronize(), "cudaTS CalculateNu");

	kernelAccumulateDiffusiveHeatRateAndCalcIonisation<<<numTilesMajorClever,threadsPerTileMajorClever>>>(
		0.5*h, 
		this->p_n_major,
		this->p_T + BEGINNING_OF_CENTRAL, // using vert indices
		this->p_B + BEGINNING_OF_CENTRAL // NEED POPULATED
		NTadditionrates); 
	Call(cudaThreadSynchronize(), "cudaTS AccumulateDiffusiveHeatRate");
	// To increase the efficiency we want to make a clever 2nd set of major tiles of size 192. Also try 256, 384.

	kernelAdvanceDensityAndTemperature<<<numTilesMajor,threadsPerTileMajor>>>(
		0.5*h, 
		this->p_n_major, this->p_T + BEGINNING_OF_CENTRAL,
		NTadditionrates,
		p_div_v_neut, p_div_v,
		pX_half->p_n_major, pX_half->p_T + BEGINNING_OF_CENTRAL
		);
	Call(cudaThreadSynchronize(), "cudaTS Advance_n_and_T");
	
	kernelAverage_n_T_x_to_tris<<<numTriTiles,threadsPerTileMinor>>>(
		pX_half->p_n,
		pX_half->p_n_major,
		pX_half->p_T
		); // call before CreateShardModel 
	Call(cudaThreadSynchronize(), "cudaTS average nTx 2");
	
	// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	// We are going to want to introduce 2nd degree approx to get n,T desired on tris.
	// Now let's set up the accel move to half-time which will provide us input of v to the full n,T move.
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	// For now went for "advective change in v" where we divide d/dt Nv by N_derivsyst
	cudaMemset(AdditionalMomRates, 0, sizeof(three_vec3)*NMINOR); // what a mess!
	kernelCreate_momflux_grad_nT_and_gradA_LapA_CurlA_verts<<<numTilesMajor, threadsPerTileMajor >> > (
		this->p_iTris_verts,
		this->p_vertPBC,   // MAXNEIGH*numVertices
		this->p_vie,
		this->p_v_n,
		this->p_AAdot,
		this->p_pos,
		AdditionalMomRates,
		p_grad_Az,
		p_Lap_Az,
		this->p_B,
		this->p_AreaMinor);
	// IMPORTANT: We passed the whole minor array not the vertex part
	// That is because the tri part is sometimes needed to go in shared data.
	Call(cudaThreadSynchronize(), "cudaTS average nTx 2");

	// this time needing vert gather from shared as well
	// as tri ...
	// something to be careful about:
	// could we re-do it so that minor tiles are contiguous
	// and don't need vert tiles that come from the end.
	// THINK HARD.

	// For minor we need to gather both vert and tri corresp.
	// So for 256 we need 128+256=384. 384 => 16 doubles in 48K.
	// This can't be the efficient way -- the efficient way
	// is to retile so that we can run say 384 with 384.
	// But we can never run the same instructions for vertex minor so muh.

	kernelCreate_momflux_grad_nT_and_gradA_LapA_CurlA_tris<<<numTriTiles, threadsPerTileMinor>>> (
		this->p_indextriminor,
		this->p_PBCtriminor,
		this->p_vie,
		this->p_v_n,
		this->p_AAdot,
		this->p_pos,
		AdditionalMomRates,
		p_grad_Az,
		p_Lap_Az,
		p_gradTe,
		this->p_B
		this->p_AreaMinor);
	Call(cudaThreadSynchronize(), "cudaTS average nTx 2");

	//Add_ViscousMomentumFluxRates(AdditionalMomRates); // should also add to NTadditionrates.
	
	kernelCreateShardModelOfDensities_And_SetMajorArea << <numTilesMajor, threadsPerTileMajor >> > (
		pX_half->p_pos,
		pX_half->p_n_major
		p_n_shards,
		p_n_n_shards);
	Call(cudaThreadSynchronize(), "cudaTS CreateShardModel pXhalf");
	
	kernelInferMinorDensitiesFromShardModel<<<numTriTiles,threadsPerTileMinor>>>(
		pX_half->p_n,
		p_n_shards,
		p_n_n_shards);// (At the moment just repopulating tri minor n.)
	Call(cudaThreadSynchronize(), "cudaTS InferMinorDensities pXhalf");

	// Get suitable v to use for resistive heating:
	kernelPopulateOhmsLaw_usesrc <<<numTilesMinor, threadsPerTileMinor>>>(
		0.5*h,

		this->p_info,
		AdditionalMomRates,
		this->p_B,
		p_LapAz,
		p_GradAz,
		p_GradTe,

		this->p_n_minor,
		this->p_T, // minor : is it populated?
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

	// Now calculate Ez_strength to get Iz_prescribed:
	long iTile;
	f64 Iz0 = 0.0;
	f64 sigma_Izz = 0.0;
	for (iTile = 0; iTile < numTilesMinor; iTile++)
	{
		Iz0 += p_Iz0[iTile];
		sigma_Izz += p_sigma_Izz[iTile];
	};
	Ez_strength_ = (Iz_prescribed - Iz0) / sigma_Izz;
	Set_f64_constant(Ez_strength, Ez_strength_);

	// Update velocities and Azdot:
	kernelUpdateVelocityAndAzdot<<<numTilesMinor, threadsPerTileMinor>>>(
		0.5*h,
		p_vn0,
		p_v0,
		p_OhmsCoeffs,
		pX_half->p_AAdot,
		pX_half->p_vie,
		pX_half->p_v_n
	);
	Call(cudaThreadSynchronize(), "cudaTS AccelerateUpdate 1");

	// ((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((
	//  ))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

	// Now do the n,T,x advance to pDestmesh:
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	pHalfMesh->CalculateOverallVelocities(p_v); // vertices first, then average to tris
	memset(pDestMesh->pData, 0, sizeof(plasma_data)*NMINOR);
	AdvectPositions_CopyTris(h, pDestMesh, p_v);

	memset(p_div_v_neut, 0, sizeof(f64)*NUMVERTICES);
	memset(p_div_v, 0, sizeof(f64)*NUMVERTICES);
	memset(NTadditionrates, 0, sizeof(NTrates)*NUMVERTICES);
	AccumulateAdvectiveMassHeatRate(p_v, NTadditionrates);
	AccumulateDiffusiveHeatRateAndCalcIonisation(h, NTadditionrates); // Wants minor n,T and B

	AdvanceDensityAndTemperature(h, pDestMesh, NTadditionrates);
	pDestMesh->Average_n_T_to_tris_and_calc_centroids_and_minorpos(); // UPGRADE TO 2ND DEGREE

	 // Now set up inputs such as AMR with advective momflux and aTP, and B
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	memset(AdditionalMomRates, 0, sizeof(three_vec3)*NMINOR);
	this->AntiAdvectAzAndAdvance(0.5*h, this, GradAz, pHalfMesh);
	pHalfMesh->Create_momflux_integral_grad_nT_and_gradA_LapA_CurlA_on_minors(p_v, AdditionalMomRates);
	pHalfMesh->Add_ViscousMomentumFluxRates(AdditionalMomRates);
	// Where is B populated? on pHalfMesh

	pDestMesh->CreateShardModelOfDensities_And_SetMajorArea();
	pDestMesh->InferMinorDensitiesFromShardModel();

	f64 starttime = evaltime;

	if (runs % 10 == 0)
	{
		// BACKWARD STEPS:

		pHalfMesh->GetLapCoeffs();

		for (iSubstep = 0; iSubstep < SUBCYCLES; iSubstep++)
		{
			evaltime += 0.5*SUBSTEP;
			kernelInterpolateVarsAndPositions(pHalfMesh, pDestMesh, (evaltime - starttime) / TIMESTEP);
			//pHalfMesh->GetLapFromCoeffs(Az_array, LapAzArray);

			kernelGetLap_verts(pX_half->p_info, Az_array, LapAzArray);
			kernelGetLap_tris(pX_half->p_info, Az_array, LapAzArray);
			Call(cudaThreadSynchronize(), "vufsTS");
			// Should populate an array of coefficients s.t. Azdot_k+1 = ~Azdot0~ + ~gamma~ Lap Az
			// Now we will wanna create each eqn for Az with coeffs on neighbour values.
			// So we need a func called "GetLapCoefficients".

			// The equation is A_k+1 - h~gamma~ Lap A_k+1 - A_k - h~Azdot0~ = 0
			// Calculate regressor x_Jacobi from eps/coeff_on_A_i
			// Given a proposed addition of a regressor x, deps/dbeta = x - h~gamma~ Lap x 
			// Set beta = -sum[eps deps/dbeta] / sum[deps/dbeta ^2]
			
			if (iSubstep == 0) {

				// feint:
				this->Accelerate2018(SUBSTEP,// ROCAzdotduetoAdvection, 
						pHalfMesh, pDestMesh, evaltime + 0.5*SUBSTEP, true);
				kernelCreateSeed <<<numTilesMinor, threadsPerTileMinor>>> (
						Az_array, this->p_AAdot,
						0.5*SUBSTEP,
						p_Azdot0, p_gamma, p_LapAz,
						Az_array_next);
				Call(cudaThreadSynchronize(), "cudaTS Get Lap Jacobi 1");
					 
			} else {	// half-substep: // and it's a FEINT
				pDestMesh->Accelerate2018(SUBSTEP,// ROCAzdotduetoAdvection, 
					pHalfMesh, pDestMesh, evaltime + 0.5*SUBSTEP, true);
				kernelCreateSeed << <numTilesMinor, threadsPerTileMinor >> > (
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
				kernelGetLap_verts << <numTilesMajor, threadsPerTileMajor >> > (Az_array_next, Lap_Aznext + BEGINNING_OF_CENTRAL);
				kernelGetLap_tris << <numTriTiles, threadsPerTileMinor >> > (Az_array_next, Lap_Aznext);
				Call(cudaThreadSynchronize(), "cudaTS GetLap Az JLS 1");

					//
					//pTri = T;
				kernelCreateEpsilonAndJacobi << <numTilesMinor, threadsPerTileMinor >> > (Az_array_next,
					Az_array,Azdot0, gamma, LapCoeffself, Lap_Aznext, epsilon, Jacobi_x);

				Call(cudaThreadSynchronize(), "cudaTS CreateEpsAndJacobi 1");
				kernelGetLap_verts << <numTilesMajor, threadsPerTileMajor >> > (Jacobi_x, Lap_Jacobi);
				kernelGetLap_tris << <numTriTiles, threadsPerTileMinor >> > (Jacobi_x, Lap_Jacobi);

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

			};

			printf("\n\n");

				// That was:
			//	JLS_for_Az_bwdstep(4, SUBSTEP); // populate Az_array with k+1 values
			memcpy(Az_array, Az_array_next, sizeof(f64)*NMINOR);
			pHalfMesh->GetLap(Az_array, LapAzArray);
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
		Create_A_from_advance(0.5*SUBSTEP, ROCAzduetoAdvection, Az_array); // from *this

		for (iSubstep = 0; iSubstep < SUBCYCLES; iSubstep++)
		{
			evaltime += 0.5*SUBSTEP;
			InterpolateVarsAndPositions(pHalfMesh, pDestMesh, (evaltime - starttime) / TIMESTEP);
			// let n,T,x be interpolated on to pHalfMesh. B remains what we populated there.

			// ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]][[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
			// Recalculate areas, or tween them, would make good sense as well.

			// It might be nice to tween the thermal pressure grad(nT) as well. And logical:
			// InterpolateArrays(AdditionalMomRates_start, AdditionalMomRates_end, AdditionalMomRates, (evaltime - starttime) / TIMESTEP);
			// Have a look how AMR is created.
			pHalfMesh->GetLap(Az_array, LapAzArray); // pHalfMesh has the positions to take Lap.
													 
			if (iSubstep == 0) {
				this->Accelerate2018(SUBSTEP,// ROCAzdotduetoAdvection, 
					pHalfMesh, pDestMesh, evaltime + 0.5*SUBSTEP, false);
			}
			else {
				// Thereafter just update pDestMesh since we can discard the old values of v, Adot.
				pDestMesh->Accelerate2018(SUBSTEP, //ROCAzdotduetoAdvection, 
					pHalfMesh, pDestMesh, evaltime + 0.5*SUBSTEP, false);
				// pHalfMesh is pUseMesh - tick, it contains B
			};

			if (iSubstep < SUBCYCLES - 1) {
				pDestMesh->AdvanceAz(SUBSTEP, ROCAzduetoAdvection, Az_array);
			}
			else {
				pDestMesh->FinalStepAz(SUBSTEP*0.5, ROCAzduetoAdvection, pDestMesh, Az_array); // just catch up to the final time
			};
			evaltime += 0.5*SUBSTEP;

			// FOR GRAPHING ONLY:
			if (iSubstep < SUBCYCLES - 1) {
				for (iMinor = 0; iMinor < NMINOR; iMinor++)
				{
					pDestMesh->pData[iMinor].Az = Az_array[iMinor];
				};
			};
			for (iMinor = 0; iMinor < NMINOR; iMinor++)
			{
				pDestMesh->pData[iMinor].temp.x = LapAzArray[iMinor];
			};

			//RefreshGraphs(*pDestMesh, 10000); // sends data to graphs AND renders them
			//Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);
			//InvalidateRect(hWnd, 0, 0);
			//UpdateWindow(hWnd);
			printf("substep %d evaltime %1.5E \n", iSubstep, evaltime);
			//getch();

			// Think carefully. The ROC for n includes the advection of the mesh rel to the fluid.
			// The ROC for v should, likewise.
			// We have ignored anti-advection for A and Adot : correct? 
			// But they BOTH SHOULD APPLY.
			// The move is VERY small so we could do both in 1 fell swoop and be using Adotz in a different true
			// location ... 
			// I prefer for right now to do the anti-advection throughout.
			// What about floating-point? nvm - just a choice for now
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

// PLAN NOW: 2156 -- give it 1 hour

// 1abc. Write 3 most key routines, going off new code & CUDA code.
// 0. Make sure we keep the CPU version viable.
// 2. Finish tying it together: rest of Advance.
// 3. Write lesser routines.
// 4. Write routines for the Backward JLS part.
// 5 --> debug & compare with CPU vers which we must keep
