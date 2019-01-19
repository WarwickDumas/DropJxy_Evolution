#pragma once
#include "cuda_struct.h"

__global__ void kernelCalculateOverallVelocitiesVertices(
	structural * __restrict__ p_info_major,
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


__global__ void kernelAdvectPositions (
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


__global__ void kernelCreateShardModelOfDensities_And_SetMajorArea_Debug(
	structural * __restrict__ p_info_minor,
	nvals * __restrict__ p_n_major,
	nvals * __restrict__ p_n_minor,

	long * __restrict__ p_izTri_vert,
	char * __restrict__ p_szPBCtri_vert,
	ShardModel * __restrict__ p_n_shards,
	ShardModel * __restrict__ p_n_n_shards,
	//	long * __restrict__ Tri_n_lists,
	//	long * __restrict__ Tri_n_n_lists	,
	f64 * __restrict__ p_AreaMajor,
	f64 * __restrict__ p_CPU_n_cent);

__global__ void kernelCreateShardModelOfDensities_And_SetMajorArea(
	structural * __restrict__ p_info_minor,
	nvals * __restrict__ p_n_major,
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

	NTrates * __restrict__ NTadditionrate,
	f64 * __restrict__ p_AreaMajor);

__global__ void kernelAdvanceDensityAndTemperature(
	f64 h_use,
	structural * __restrict__ p_info_major,
	nvals * p_n_major,
	T3 * p_T_major,
	NTrates * __restrict__ NTadditionrates,

	nvals * p_n_use,
	T3 * p_T_use,
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
	f64 * __restrict__ ROCAzdotduetoAdvection,

	f64_vec3 * __restrict__ p_vn0_dest,
	v4 * __restrict__ p_v0_dest,
	OhmsCoeffs * __restrict__ p_OhmsCoeffs_dest,
	AAdot * __restrict__ p_AAdot_intermediate,

	f64 * __restrict__ p_Iz0,
	f64 * __restrict__ p_sigma_zz,

	f64 * __restrict__ p_denom_i,
	f64 * __restrict__ p_denom_e,
	bool const bSwitchSave,
	bool const bUse_dest_n_for_Iz,
	nvals * __restrict__ p_n_dest_minor);


__global__ void kernelCalculateVelocityAndAzdot(
	f64 h_use,
	f64_vec3 * __restrict__ p_vn0,
	v4 * __restrict__ p_v0,
	OhmsCoeffs * __restrict__ p_OhmsCoeffs,
	AAdot * __restrict__ p_AAzdot_intermediate,
	nvals * __restrict__ p_n_minor,

	AAdot * __restrict__ p_AAzdot_out,
	v4 * __restrict__ p_vie_out,
	f64_vec3 * __restrict__ p_vn_out);


__global__ void kernelUpdateAz(
	f64 const h_use,
	AAdot * __restrict__ p_AAdot_use,
	f64 * __restrict__ p_ROCAzduetoAdvection,
	f64 * __restrict__ p_Az
);
__global__ void kernelPopulateArrayAz(
	f64 const h_use,
	AAdot * __restrict__ p_AAdot_use,
	f64 * __restrict__ p_ROCAzduetoAdvection,
	f64 * __restrict__ p_Az
);
__global__ void kernelPushAzInto_dest(
	AAdot * __restrict__ p_AAdot,
	f64 * __restrict__ p_Az
);
__global__ void kernelPullAzFromSyst(
	AAdot * __restrict__ p_AAdot,
	f64 * __restrict__ p_Az
);
__global__ void kernelAdd(
	f64 * __restrict__ p_updated,
	f64 beta,
	f64 * __restrict__ p_added
);
__global__ void kernelCreateSeedPartOne(
	f64 const h_use,
	f64 * __restrict__ p_Az,
	AAdot * __restrict__ p_AAdot_use,
	f64 * __restrict__ p_AzNext
);

__global__ void kernelCreateSeedPartTwo(
	f64 const h_use,
	f64 * __restrict__ p_Azdot0,
	f64 * __restrict__ p_gamma,
	f64 * __restrict__ p_LapAz,
	f64 * __restrict__ p_AzNext_update
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

	f64 * __restrict__ ROCAzduetoAdvection,
	f64 * __restrict__ ROCAzdotduetoAdvection,
	f64_vec2 * __restrict__ p_v_overall_minor,

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

	LONG3 * __restrict__ p_who_am_I_to_corners,
	LONG3 * __restrict__ p_tricornerindex,

	f64_vec3 * __restrict__ p_MAR_neut,
	f64_vec3 * __restrict__ p_MAR_ion,
	f64_vec3 * __restrict__ p_MAR_elec,
	ShardModel * __restrict__ p_n_shards
);

__global__ void kernelCreateLinearRelationship(
	f64 const h_use,
	structural * __restrict__ p_info,
	OhmsCoeffs* __restrict__ p_Ohms,
	v4 * __restrict__ p_v0,
	f64 * __restrict__ p_Lap_Az_use,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_denom_e,
	f64 * __restrict__ p_denom_i,
	AAdot * __restrict__ p_AAdot_intermediate,
	f64 * __restrict__ p_Azdot0,
	f64 * __restrict__ p_gamma
);

__global__ void kernelNeutral_pressure_and_momflux(
	structural * __restrict__ p_info_minor,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighTriMinor,
	char * __restrict__ p_szPBCtriminor,
	LONG3 * __restrict__ p_who_am_I_to_corners,
	LONG3 * __restrict__ p_tricornerindex,

	T3 * __restrict__ p_T_minor,
	f64_vec3 * __restrict__ p_v_n,
	ShardModel * __restrict__ p_n_shards,
	f64_vec2 * __restrict__ p_v_overall_minor,
	f64_vec3 * __restrict__ p_MAR_neut
);


__global__ void kernelWrapTriangles(
	structural * __restrict__ p_info_minor,
	LONG3 * __restrict__ p_tri_corner_index,
	char * __restrict__ p_was_vertex_rotated,

	v4 * __restrict__ p_vie_minor,
	f64_vec3 * __restrict__ p_v_n_minor,
	char * __restrict__ p_triPBClistaffected,

	CHAR4 * __restrict__ p_tri_periodic_corner_flags);

__global__ void kernelWrapVertices(
	structural * __restrict__ p_info_minor,
	v4 * __restrict__ p_vie,
	f64_vec3 * __restrict__ p_v_n,
	char * __restrict__ p_was_vertex_rotated);


__global__ void kernelReassessTriNeighbourPeriodicFlags_and_populate_PBCIndexneighminor(
	structural * __restrict__ p_info_minor,
	LONG3 * __restrict__ p_tri_neigh_index,
	LONG3 * __restrict__ p_tri_corner_index,
	char * __restrict__ p_was_vertex_rotated,
	CHAR4 * __restrict__ p_tri_periodic_corner_flags,
	CHAR4 * __restrict__ p_tri_periodic_neigh_flags,
	char * __restrict__ p_szPBC_triminor
);
__global__ void kernelReset_szPBCtri_vert(
	structural * __restrict__ p_info_minor,
	long * __restrict__ p_izTri_vert,
	long * __restrict__ p_izNeigh_vert,
	char * __restrict__ p_szPBCtri_vert,
	char * __restrict__ p_szPBCneigh_vert,
	char * __restrict__ p_triPBClistaffected
);
