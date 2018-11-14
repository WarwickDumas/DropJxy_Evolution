#pragma once
#include "cuda_struct.h"

__global__ void kernelCalculateOverallVelocitiesVertices(
	v4 * __restrict__ p_vie_major,
	f64_vec3 * __restrict__ p_v_n_major,
	nvals * __restrict__ p_n_major,
	f64_vec2 * __restrict__ p_v_overall_major);


__global__ void kernelAverageOverallVelocitiesTriangles(
	f64_vec2 * __restrict__ p_v_overall_minor_major,
	f64_vec2 * __restrict__ p_v_overall_minor_minor,
	structural * __restrict__ p_info,
	LONG3 * __restrict__ p_tri_corner_index,
	CHAR4 * __restrict__ p_tri_periodic_corner_flags
);


__global__ void kernelAdvectPositions_CopyTris (
	f64 h_use,
	structural * __restrict__ p_info_src,
	structural * __restrict__ p_info_dest,
	f64_vec2 * __restrict__ p_v_overall_minor);


__global__ void kernelAverage_n_T_x_to_tris  (
	nvals * __restrict__ p_n_minor,
	nvals * __restrict__ p_n_major,
	T3 * __restrict__ p_T_minor,
	structural * __restrict__ p_info,
	LONG3 * __restrict__ p_tri_corner_index,
	CHAR4 * __restrict__ p_tri_periodic_corner_flags
	);


__global__ void kernelCreateShardModelOfDensities_And_SetMajorArea(
	structural * __restrict__ p_info_minor,
	nvals * __restrict__ p_n_minor,

	long * __restrict__ p_izTri_vert,
	char * __restrict__ p_szPBCtri_vert,
	ShardModel * __restrict__ p_n_shards,
	ShardModel * __restrict__ p_n_n_shards,
//	long * __restrict__ Tri_n_lists,
//	long * __restrict__ Tri_n_n_lists,
	f64 * __restrict__ p_AreaMajor);

__global__ void kernelInferMinorDensitiesFromShardModel(
	structural * __restrict__ p_info,
	nvals * __restrict__ p_n_minor,
	ShardModel * __restrict__ p_n_shards,
	ShardModel * __restrict__ p_n_shards_n,
	LONG3 * __restrict__ p_tri_corner_index,
	LONG3 * __restrict__ p_who_am_I_to_corner
); 

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
	v4 * __restrict__ p_vie_use,
	f64_vec3 * __restrict__ p_v_n_use,

	f64 * __restrict__ p_div_v_neut,
	f64 * __restrict__ p_div_v,
	f64 * __restrict__ p_Integrated_div_v_overall,
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
	f64_vec2 * __restrict__ p_v_overall_minor_minor,
	LONG3 * __restrict__ p_tricornerindex,
	LONG3 * __restrict__ p_trineighindex,
	LONG3 * __restrict__ p_which_iTri_number_am_I,
	CHAR4 * __restrict__ p_szPBCneigh_tris, 
	nvals * __restrict__ p_n_upwind_minor // result
);


__global__ void kernelAccumulateAdvectiveMassHeatRate(
	f64 h_use,
	structural * __restrict__ p_info_minor,
	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBCtri_verts,

	nvals * __restrict__ p_n_src_major,
	T3 * __restrict__ p_T_src_major,

	nvals * __restrict__ p_n_upwind_minor,
	v4 * __restrict__ p_vie_minor,
	f64_vec3 * __restrict__ p_v_n_minor,
	f64_vec2 * __restrict__ p_v_overall_minor,
	T3 * __restrict__ p_T_minor, // may or may not overlap source: don't we only use from tris? so not overlap

	NTrates * __restrict__ p_NTadditionrates,
	f64 * __restrict__ p_div_v,
	f64 * __restrict__ p_div_v_n,
	f64 * __restrict__ p_Integrated_div_v_overall
	);


__global__ void kernelPopulateOhmsLaw(
	f64 h_use,

	structural * __restrict__ p_info_minor,
	f64_vec3 * __restrict__ p_MAR_neut,
	f64_vec3 * __restrict__ p_MAR_ion,
	f64_vec3 * __restrict__ p_MAR_elec,
	f64_vec3 * __restrict__ p_B,
	f64 * __restrict__ p_LapAz,
	f64_vec2 * __restrict__ p_GradAz,
	f64_vec2 * __restrict__ p_GradTe,

	nvals * __restrict__ p_n_minor_use,
	T3 * __restrict__ p_T_minor_use,
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


__global__ void kernelUpdateVelocityAndAzdotAndAz(
	f64 h_use,
	f64_vec3 * __restrict__ p_vn0,
	v4 * __restrict__ p_v0,
	OhmsCoeffs * __restrict__ p_OhmsCoeffs,
	AAdot * __restrict__ p_Azdot_update,
	v4 * __restrict__ p_vie_out,
	f64_vec3 * __restrict__ p_vn_out,
	f64_vec2 * __restrict__ p_GradAz,
	f64_vec2 * __restrict__ p_v_overall_minor
);



__global__ void kernelAdd(
	f64 * __restrict__ p_updated,
	f64 beta,
	f64 * __restrict__ p_added
);

__global__ void kernelResetFrillsAz(
	structural * __restrict__ p_info,
	LONG3 * __restrict__ trineighbourindex,
	f64 * __restrict__ p_Az);


__global__ void kernelCreateEpsilonAndJacobi(
	f64 const h_use,
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


/*
__global__ void kernelGetLap_verts(
	structural * __restrict__ p_info,
	f64 * __restrict__ p_Az,
	long * __restrict__ p_izNeighMinor,
	long * __restrict__ p_izTri,
	f64 * __restrict__ p_LapAz);
*/


__global__ void kernelGetLapCoeffs(
	structural * __restrict__ p_info,
	long * __restrict__ p_izTri,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtri_vertex,
	char * __restrict__ p_szPBCtriminor,
	f64 * __restrict__ p_LapCoeffSelf);

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

	LONG3 * __restrict__ p_who_am_I_to_corners,
	LONG3 * __restrict__ p_tricornerindex,

	f64_vec3 * __restrict__ p_MAR_neut,
	f64_vec3 * __restrict__ p_MAR_ion,
	f64_vec3 * __restrict__ p_MAR_elec,

	ShardModel * __restrict__ p_n_shards,

	f64_vec2 * __restrict__ p_GradTe,
	f64_vec2 * __restrict__ p_GradAz,
	f64 * __restrict__ p_LapAz,
	f64_vec3 * __restrict__ p_B,
	f64 * __restrict__ p_AreaMinor
);

__global__ void kernelInterpolateVarsAndPositions(
	f64 ppn,
	structural * __restrict__ p_info1,
	structural * __restrict__ p_info2,
	nvals * __restrict__ p_n_minor1,
	nvals * __restrict__ p_n_minor2,
	T3 * __restrict__ p_T_minor1,
	T3 * __restrict__ p_T_minor2,
	f64_vec3 * __restrict__ p_B1,
	f64_vec3 * __restrict__ p_B2,

	structural * __restrict__ p_info_dest,
	nvals * __restrict__ p_n_minor,
	T3 * __restrict__ p_T_minor,
	f64_vec3 * __restrict__ p_B);

__global__ void kernelCreate_momflux_minor(

	structural * __restrict__ p_info_minor,
	v4 * __restrict__ p_vie_minor,
	f64_vec2 * __restrict__ p_v_overall_minor,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighTriMinor,
	char * __restrict__ p_szPBCtriminor,

	f64_vec3 * __restrict__ p_MAR_neut,
	f64_vec3 * __restrict__ p_MAR_ion,
	f64_vec3 * __restrict__ p_MAR_elec,
	ShardModel * __restrict__ p_n_shards
);


__global__ void kernelNeutral_pressure_and_momflux(
	structural * __restrict__ p_info_minor,
	f64_vec3 * __restrict__ p_v_n,
	ShardModel * __restrict__ p_n_shards_n,
	f64_vec2 * __restrict__ p_v_overall_minor,
	f64_vec3 * __restrict__ p_MAR_neut
);






// Device-accessible constants not known at compile time:

__constant__ long nBlocks, Nverts, uDataLen_d; // Nverts == numVertices



__constant__ f64_tens2 Anticlockwise, Clockwise; // use this to do rotation.

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

four_pi_over_c_ReverseJz,

FRILL_CENTROID_OUTER_RADIUS_d, FRILL_CENTROID_INNER_RADIUS_d;



// some of these we can do #define

#define FOUR_PI 12.566370614359



__constant__ f64 cross_s_vals_viscosity_ni_d[10], cross_s_vals_viscosity_nn_d[10],

cross_T_vals_d[10], cross_s_vals_MT_ni_d[10];



__constant__ f64 Ez_strength;

__constant__ f64 negative_Iz_per_triangle; // -Iz_prescribed / (real)(numEndZCurrentTriangles - numStartZCurrentTriangles)

__constant__ long numStartZCurrentTriangles, numEndZCurrentTriangles;

