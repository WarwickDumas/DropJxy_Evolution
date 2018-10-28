#pragma once
#include "cuda_struct.h"

#define MAXNEIGH_d 12

// Make these global device constants:
static real const one_over_kB = 1.0 / kB; // multiply by this to convert to eV
static real const one_over_kB_cubed = 1.0 / (kB*kB*kB); // multiply by this to convert to eV
static real const kB_to_3halves = sqrt(kB)*kB;

f64 M_i_over_in = m_i / (m_i + m_n);
f64 M_e_over_en = m_e / (m_e + m_n);
f64 M_n_over_ni = m_n / (m_i + m_n);
f64 M_n_over_ne = m_n / (m_e + m_n);

f64 const M_en = m_e * m_n / ((m_e + m_n)*(m_e + m_n));
f64 const M_in = m_i * m_n / ((m_i + m_n)*(m_i + m_n));
f64 const M_ei = m_e * m_i / ((m_e + m_i)*(m_e + m_i));
f64 const m_en = m_e * m_n / (m_e + m_n);
f64 const m_ei = m_e * m_i / (m_e + m_i);



__global__ void kernelCalculateOverallVelocitiesVertices(
	v4 * __restrict__ p_vie_major,
	f64_vec3 * __restrict__ p_v_n_major,
	nvals * __restrict__ p_n_major,
	f64_vec2 * __restrict__ p_v_overall_major);


__global__ void kernelAverageOverallVelocitiesTriangles(
	f64_vec2 * __restrict__ p_overall_v_major,
	f64_vec2 * __restrict__ p_overall_v_minor,
	structural * __restrict__ p_info,
	LONG3 * __restrict__ p_tri_corner_index,
	CHAR3 * __restrict__ p_tri_periodic_corner_flags
);


__global__ void kernelAdvectPositions_CopyTris (
	f64 h_use,
	structural * __restrict__ p_info_src,
	structural * __restrict__ p_info_dest,
	f64_vec2 * __restrict__ p_overall_v);


__global__ void kernelAverage_n_T_x_to_tris  (
	nvals * __restrict__ p_n_minor,
	nvals * __restrict__ p_n_major,
	T3 * __restrict__ p_T_minor,
	structural * __restrict__ p_info
	);


__global__ void kernelCreateShardModelOfDensities_And_SetMajorArea(
	structural * __restrict__ p_info_minor, // note we called for major but passed whole array??
	nvals * __restrict__ p_n_major,
	ShardModel * p_n_shards,
	ShardModel * p_n_n_shards);

__global__ void kernelCalculateNu_eHeartNu_iHeart_nu_nn_visc(
	structural * __restrict__ p_info,
	nvals * __restrict__ p_n,
	T3 * __restrict__ p_T,
	species3 * __restrict__ p_nu);


__global__ void kernelAccumulateDiffusiveHeatRateAndCalcIonisation(
	f64 const h_use,
	structural * __restrict__ p_info_sharing,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,

	nvals * __restrict__ p_n_major,
	T3 * __restrict__ p_T_major,
	f64_vec3 * __restrict__ p_B_major,
	species3 * __restrict__ p_nu_major,

	NTrates * __restrict__ NTadditionrate);

__global__ void kernelAdvanceDensityAndTemperature(
	f64 h_use,
	structural * __restrict__ p_info_major,
	nvals * __restrict__ p_n_major,
	T3 * __restrict__ p_T_major,
	NTrates * __restrict__ NTadditionrates,

	nvals * __restrict__ p_n_use,
	T3 * __restrict__ p_T_use,

	f64 * __restrict__ p_div_v_neut,
	f64 * __restrict__ p_div_v,
	f64 * __restrict__ Integrated_Div_v_overall,
	f64 * __restrict__ p_AreaMajor, // hmm

	nvals * __restrict__ p_n_major_dest,
	T3 * __restrict__ p_T_major_dest
);


__global__ void kernelCalculateUpwindDensity_tris(
	structural * __restrict__ p_info_minor,
	ShardModel * __restrict__ p_n_shard_n_major,
	ShardModel * __restrict__ p_n_shard_major,
	v4 * __restrict__ p_vie_minor,
	f64_vec3 * __restrict__ p_v_n_minor,
	f64_vec2 * __restrict__ p_overall_v_minor,
	LONG3 * __restrict__ p_tricornerindex,
	LONG3 * __restrict__ p_trineighindex,
	LONG3 * __restrict__ p_which_iTri_number_am_I,
	char * __restrict__ p_PBCtriminor, // carries 6, we only need 3
	nvals * __restrict__ p_n_upwind_minor // result
);


__global__ void kernelAccumulateAdvectiveMassHeatRate(
	structural * __restrict__ p_info_minor,
	nvals * __restrict__ p_n_src_major,
	T3 * __restrict__ p_T_src_major,

	nvals * __restrict__ p_n_upwind_minor,
	v4 * __restrict__ p_vie_minor,
	f64_vec3 * __restrict__ p_v_n_minor,
	f64_vec2 * __restrict__ p_v_overall_minor
	T3 * __restrict__ p_T_minor, // may or may not overlap source: don't we only use from tris? so not overlap

	nvals * __restrict__ p_n_dest_major,
	T3 * __restrict__ p_T_dest_major,
	);


__global__ void kernelPopulateOhmsLaw_usesrc(
	f64 h_use,

	structural * __restrict__ p_info_minor,
	three_vec3 * __restrict__ p_AdditionalMomRates,
	f64_vec3 * __restrict__ p_B,
	f64 * __restrict__ p_LapAz,
	f64_vec2 * __restrict__ p_GradAz,
	f64_vec2 * __restrict__ p_GradTe,

	nvals * __restrict__ p_n_minor_src,
	T3 * __restrict__ p_T_minor_src,
	v4 * __restrict__ p_vie_src,
	f64_vec3 * __restrict__ p_v_n_src,
	AAdot * __restrict__ p_AAdot_src,
	f64 * __restrict__ p_AreaMinor,

	f64_vec3 * __restrict__ p_vn0_dest,
	v4 * __restrict__ p_v0_dest,
	OhmsCoeffs * __restrict__ p_OhmsCoeffs_dest,
	AAdot * __restrict__ p_AAdot_intermediate,

	f64 * __restrict__ p_Iz0,
	f64 * __restrict__ p_sigma_zz,
	bool bFeint);


__global__ void kernelUpdateVelocityAndAzdot(
	f64 h_use,
	f64_vec3 * __restrict__ p_vn0,
	v4 * __restrict__ p_v0,
	OhmsCoeffs * __restrict__ p_OhmsCoeffs,
	f64 * __restrict__ p_Azdot_update,

	v4 * __restrict__ p_vie_out,
	f64_vec3 * __restrict__ p_vn_out
);



__global__ void kernelAdd(
	f64 * __restrict__ p_updated,
	f64 beta,
	f64 * __restrict__ p_added
);

__global__ void kernelResetFrillsAz << <numTriTiles, threadsPerTileMinor >> > (
	structural * __restrict__ p_info,
	LONG3 * __restrict__ trineighbourindex,
	f64 * __restrict__ p_Az);


__global__ void kernelCreateEpsilonAndJacobi(
	structural * __restrict__ p_info,
	f64 * __restrict__ p_Az_array_next,
	f64 * __restrict__ p_Az_array,
	f64 * __restrict__ p_Azdot0,
	f64 * __restrict__ p_gamma,
	f64 * __restrict__ p_LapCoeffself,
	f64 * __restrict__ p_Lap_Aznext,
	f64 * __restrict__ p_epsilon,
	f64 * __restrict__ p_Jacobi_x);


__global__ void kernelAccumulateSummands(
	structural * __restrict__ p_info,
	f64 h_use,
	f64 * __restrict__ p_epsilon,
	f64 * __restrict__ p_Jacobi,
	f64 * __restrict__ p_LapJacobi,
	f64 * __restrict__ p_gamma,
	f64 * __restrict__ p_sum_eps_d,
	f64 * __restrict__ p_sum_d_d,
	f64 * __restrict__ p_sum_eps_eps);



__global__ void kernelGetLap_verts(
	structural * __restrict__ p_info,
	f64 * __restrict__ p_Az,
	long * __restrict__ p_izNeighMinor,
	long * __restrict__ p_izTri,
	f64 * __restrict__ p_LapAz);



__global__ void kernelGetLap_minor(
	structural * __restrict__ p_info,
	f64 * __restrict__ p_Az,

	long * __restrict__ p_izTri,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtri_vertex,
	char * __restrict__ p_szPBCtriminor,

	f64 * __restrict__ p_LapAz);



__global__ void kernelCreate_pressure_gradT_and_gradA_LapA_CurlA_minor(

	structural * __restrict__ p_info_minor,
	T3 * __restrict__ p_T_minor,
	AAdot * __restrict__ p_AAdot,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighTriMinor,
	char * __restrict__ p_szPBCtriminor,

	three_vec3 * __restrict__ p_AdditionalMomrates,
	ShardModel * __restrict__ p_n_shards,

	f64_vec2 * __restrict__ p_GradTe,
	f64_vec2 * __restrict__ p_GradAz,
	f64_vec2 * __restrict__ p_LapAz,
	f64_vec3 * __restrict__ p_B
);

__global__ void kernelCreate_momflux_minor(

	structural * __restrict__ p_info_minor,
	v4 * __restrict__ p_vie_minor,
	f64_vec2 * __restrict__ p_v_overall_minor,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighTriMinor,
	char * __restrict__ p_szPBCtriminor,

	three_vec3 * __restrict__ p_AdditionalMomrates,
	ShardModel * __restrict__ p_n_shards
);


__global__ void kernelNeutral_pressure_and_momflux(
	structural * __restrict__ p_info_minor,
	f64_vec3 * __restrict__ p_v_n,
	ShardModel * __restrict__ p_n_shards_n,
	f64_vec2 * __restrict__ p_v_overall,
	three_vec3 * __restrict__ p_AdditionalMomRates
);






