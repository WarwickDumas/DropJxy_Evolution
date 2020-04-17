#include "kernel.h"
#include "cuda_struct.h"
#include "constant.h"
#include "FFxtubes.h"

#define BWDSIDET
#define LONGITUDINAL

// TO DO:
// Line 1420:
// Yes, very much a waste. The edge positions should be calculated from the vertex positions, we can
// load flags to determine if it is an insulator-crossing triangle and that is the proper way to handle that.


#define FOUR_PI 12.5663706143592

__device__ void Augment_Jacobean(
	f64_tens3 * pJ, 
	real Factor, //h_over (N m_i)
	f64_vec2 edge_normal, 
	f64 ita_par, f64 nu, f64_vec3 omega,
	f64 grad_vjdx_coeff_on_vj_self,
	f64 grad_vjdy_coeff_on_vj_self
) {
	if ((VISCMAG == 0) || (omega.dot(omega) < 0.01*0.1*nu*nu))
	{
		// run unmagnetised case

		//Pi_xx = -ita_par*THIRD*(4.0*gradvx.x - 2.0*gradvy.y);
		//Pi_xy = -ita_par*(gradvx.y + gradvy.x);
		//Pi_yx = Pi_xy;
		//Pi_yy = -ita_par*THIRD*(4.0*gradvy.y - 2.0*gradvx.x);
		//Pi_zx = -ita_par*(gradviz.x);
		//Pi_zy = -ita_par*(gradviz.y);		
		//	visc_contrib.x = -over_m_i*(Pi_xx*edge_normal.x + Pi_xy*edge_normal.y);


		// eps_x = vx_k+1 - vx_k - h MAR.x / N
		pJ->xx += Factor*
			((
				// Pi_xx
				-ita_par*THIRD*(4.0*grad_vjdx_coeff_on_vj_self)
				) *edge_normal.x + (
					//Pi_xy
					-ita_par*(grad_vjdy_coeff_on_vj_self)
					)*edge_normal.y);

		pJ->yx += Factor*
			((
				// Pi_yx
				-ita_par*(grad_vjdy_coeff_on_vj_self)
				)*edge_normal.x + (
					// Pi_yy
					-ita_par*THIRD*(-2.0*grad_vjdx_coeff_on_vj_self)
					)*edge_normal.y);

		// The z direction doesn't feature vx --- that is because dvx/dz == 0

		pJ->xy += Factor*
			((
				// Pi_xx
				-ita_par*THIRD*(-2.0*grad_vjdy_coeff_on_vj_self)
				)*edge_normal.x + (
					// Pi_xy
					-ita_par*(grad_vjdx_coeff_on_vj_self)
					)*edge_normal.y);

		pJ->yy += Factor*
			((
				// Pi_yx
				-ita_par*(grad_vjdx_coeff_on_vj_self)
				)*edge_normal.x + (
					// Pi_yy
					-ita_par*THIRD*(4.0*grad_vjdy_coeff_on_vj_self)
					)*edge_normal.y);

		pJ->zz += Factor*
			((
				// Pi_zx
				-ita_par*grad_vjdx_coeff_on_vj_self
				)*edge_normal.x + (
					// Pi_zy
					-ita_par*grad_vjdy_coeff_on_vj_self
					)*edge_normal.y);

		// In this way we develop let's say the J matrix, J for Jacobean
		// Then we could wish to do LU decomp of 4x4 matrix J so that
		// Regressor = J^-1 epsilon[loaded]
		// But we'll do 2 x 3 x 3.
	} else {

		f64 omegamod;
		f64_vec3 unit_b, unit_perp, unit_Hall;
		{
			f64 omegasq = omega.dot(omega);
			omegamod = sqrt(omegasq);
			unit_b = omega / omegamod;
			unit_perp = Make3(edge_normal, 0.0) - unit_b*(unit_b.dotxy(edge_normal));
			unit_perp = unit_perp / unit_perp.modulus();
			unit_Hall = unit_b.cross(unit_perp); // Note sign.
		}

		//f64 momflux_b, momflux_perp, momflux_Hall;
		f64 ita_1 = ita_par*(nu*nu / (nu*nu + omegamod*omegamod));
		f64 ita_2 = ita_par*(nu*nu / (nu*nu + 0.25*omegamod*omegamod));
		f64 ita_3 = ita_par*(nu*omegamod / (nu*nu + omegamod*omegamod));
		f64 ita_4 = 0.5*ita_par*(nu*omegamod / (nu*nu + 0.25*omegamod*omegamod));

		f64_vec3mag mag_edge;
		mag_edge.b = unit_b.x*edge_normal.x + unit_b.y*edge_normal.y;
		mag_edge.P = unit_perp.x*edge_normal.x + unit_perp.y*edge_normal.y;
		mag_edge.H = unit_Hall.x*edge_normal.x + unit_Hall.y*edge_normal.y;

		// ==================================================================

		// Our approach is going to be to populate the "Flux Jacobean".
		// Let F_P = PI_Pb edgenormal_b + PI_PP edgenormal_P + PI_PH edgenormal_H

		// *********************
		//  Accumulate dF_b/dvx 
		// *********************

		f64_tens3mag F;
		memset(&F, 0, sizeof(f64_tens3mag));
		f64 bdotPsi = unit_b.x*grad_vjdx_coeff_on_vj_self + unit_b.y*grad_vjdy_coeff_on_vj_self;
		f64 PdotPsi = unit_perp.x*grad_vjdx_coeff_on_vj_self + unit_perp.y*grad_vjdy_coeff_on_vj_self;
		f64 HdotPsi = unit_Hall.x*grad_vjdx_coeff_on_vj_self + unit_Hall.y*grad_vjdy_coeff_on_vj_self;

		f64 d_Pi_by_dvx;
		// how to use union? Can just put in and out of scope.
		// d_Pi_bb_by_dvx =
		d_Pi_by_dvx = -ita_par*THIRD*(4.0*unit_b.x*bdotPsi - 2.0*unit_perp.x*PdotPsi - 2.0*unit_Hall.x*HdotPsi);

		F.bx += d_Pi_by_dvx * mag_edge.b; // Pi_bb

		d_Pi_by_dvx = -ita_2*(unit_b.x*PdotPsi + unit_perp.x*bdotPsi)
			- ita_4*(unit_b.x*HdotPsi + unit_Hall.x*bdotPsi);
		// Pi_bP

		F.bx += d_Pi_by_dvx * mag_edge.P; // Pi_bP
		F.Px += d_Pi_by_dvx * mag_edge.b; // Pi_Pb

		d_Pi_by_dvx = -(ita_2*(unit_b.x*HdotPsi + unit_Hall.x*bdotPsi) + ita_4*(unit_b.x*PdotPsi + unit_perp.x*bdotPsi));
		// Pi_bH

		F.bx += d_Pi_by_dvx * mag_edge.H; // Pi_bH 
		F.Hx += d_Pi_by_dvx * mag_edge.b; // Pi_Hb	

		d_Pi_by_dvx = -0.5*(ita_par + ita_1)*THIRD*(-2.0*unit_b.x*bdotPsi + 4.0*unit_perp.x*PdotPsi - 2.0*unit_Hall.x*HdotPsi)
			- 0.5*(ita_par - ita_1)*THIRD*(-2.0*unit_b.x*bdotPsi - 2.0*unit_perp.x*PdotPsi + 4.0*unit_Hall.x*HdotPsi)
			- ita_3*(unit_perp.x*HdotPsi + unit_Hall.x*PdotPsi);
		// Pi_PP

		F.Px += d_Pi_by_dvx * mag_edge.P;

		d_Pi_by_dvx = -ita_1*(unit_perp.x*HdotPsi + unit_Hall.x*PdotPsi) + ita_3*(unit_perp.x*PdotPsi - unit_Hall.x*HdotPsi);
		// Pi_PH

		F.Px += d_Pi_by_dvx * mag_edge.H;
		F.Hx += d_Pi_by_dvx * mag_edge.P; // Pi_PH

		d_Pi_by_dvx = -0.5*(ita_par + ita_1)*THIRD*(-2.0*unit_b.x*bdotPsi - 2.0*unit_perp.x*PdotPsi + 4.0*unit_Hall.x*HdotPsi)
			- 0.5*(ita_par - ita_1)*THIRD*(-2.0*unit_b.x*bdotPsi + 4.0*unit_perp.x*PdotPsi - 2.0*unit_Hall.x*HdotPsi)
			+ ita_3*(unit_perp.x*HdotPsi + unit_Hall.x*PdotPsi);
		// Pi_HH

		F.Hx += d_Pi_by_dvx*mag_edge.H;

		// That was the x column.
		// Repeat exact same thing again replacing .x ..
		// first get it sensible.
		f64 d_Pi_by_dvy;

		// d_Pi_bb_by_dvy =
		d_Pi_by_dvy = -ita_par*THIRD*(4.0*unit_b.y*bdotPsi - 2.0*unit_perp.y*PdotPsi - 2.0*unit_Hall.y*HdotPsi);

		F.by += d_Pi_by_dvy * mag_edge.b; // Pi_bb

		d_Pi_by_dvy = -ita_2*(unit_b.y*PdotPsi + unit_perp.y*bdotPsi)
			- ita_4*(unit_b.y*HdotPsi + unit_Hall.y*bdotPsi);
		// Pi_bP

		F.by += d_Pi_by_dvy * mag_edge.P; // Pi_bP
		F.Py += d_Pi_by_dvy * mag_edge.b; // Pi_Pb

		d_Pi_by_dvy = -(ita_2*(unit_b.y*HdotPsi + unit_Hall.y*bdotPsi) + ita_4*(unit_b.y*PdotPsi + unit_perp.y*bdotPsi));
		// Pi_bH

		F.by += d_Pi_by_dvy * mag_edge.H; // Pi_bH 
		F.Hy += d_Pi_by_dvy * mag_edge.b; // Pi_Hb	

		d_Pi_by_dvy = -0.5*(ita_par + ita_1)*THIRD*(-2.0*unit_b.y*bdotPsi + 4.0*unit_perp.y*PdotPsi - 2.0*unit_Hall.y*HdotPsi)
			- 0.5*(ita_par - ita_1)*THIRD*(-2.0*unit_b.y*bdotPsi - 2.0*unit_perp.y*PdotPsi + 4.0*unit_Hall.y*HdotPsi)
			- ita_3*(unit_perp.y*HdotPsi + unit_Hall.y*PdotPsi);
		// Pi_PP

		F.Py += d_Pi_by_dvy * mag_edge.P;

		d_Pi_by_dvy = -ita_1*(unit_perp.y*HdotPsi + unit_Hall.y*PdotPsi) + ita_3*(unit_perp.y*PdotPsi - unit_Hall.y*HdotPsi);
		// Pi_PH

		F.Py += d_Pi_by_dvy * mag_edge.H;
		F.Hy += d_Pi_by_dvy * mag_edge.P; // Pi_PH

		d_Pi_by_dvy = -0.5*(ita_par + ita_1)*THIRD*(-2.0*unit_b.y*bdotPsi - 2.0*unit_perp.y*PdotPsi + 4.0*unit_Hall.y*HdotPsi)
			- 0.5*(ita_par - ita_1)*THIRD*(-2.0*unit_b.y*bdotPsi + 4.0*unit_perp.y*PdotPsi - 2.0*unit_Hall.y*HdotPsi)
			+ ita_3*(unit_perp.y*HdotPsi + unit_Hall.y*PdotPsi);
		// Pi_HH

		F.Hy += d_Pi_by_dvy*mag_edge.H;

		f64 d_Pi_by_dvz;

		// d_Pi_bb_by_dvz =
		d_Pi_by_dvz = -ita_par*THIRD*(4.0*unit_b.z*bdotPsi - 2.0*unit_perp.z*PdotPsi - 2.0*unit_Hall.z*HdotPsi);

		F.bz += d_Pi_by_dvz * mag_edge.b; // Pi_bb

		d_Pi_by_dvz = -ita_2*(unit_b.z*PdotPsi + unit_perp.z*bdotPsi)
			- ita_4*(unit_b.z*HdotPsi + unit_Hall.z*bdotPsi);
		// Pi_bP

		F.bz += d_Pi_by_dvz * mag_edge.P; // Pi_bP
		F.Pz += d_Pi_by_dvz * mag_edge.b; // Pi_Pb

		d_Pi_by_dvz = -(ita_2*(unit_b.z*HdotPsi + unit_Hall.z*bdotPsi) + ita_4*(unit_b.z*PdotPsi + unit_perp.z*bdotPsi));
		// Pi_bH

		F.bz += d_Pi_by_dvz * mag_edge.H; // Pi_bH 
		F.Hz += d_Pi_by_dvz * mag_edge.b; // Pi_Hb	

		d_Pi_by_dvz = -0.5*(ita_par + ita_1)*THIRD*(-2.0*unit_b.z*bdotPsi + 4.0*unit_perp.z*PdotPsi - 2.0*unit_Hall.z*HdotPsi)
			- 0.5*(ita_par - ita_1)*THIRD*(-2.0*unit_b.z*bdotPsi - 2.0*unit_perp.z*PdotPsi + 4.0*unit_Hall.z*HdotPsi)
			- ita_3*(unit_perp.z*HdotPsi + unit_Hall.z*PdotPsi);
		// Pi_PP

		F.Pz += d_Pi_by_dvz * mag_edge.P;

		d_Pi_by_dvz = -ita_1*(unit_perp.z*HdotPsi + unit_Hall.z*PdotPsi) + ita_3*(unit_perp.z*PdotPsi - unit_Hall.z*HdotPsi);
		// Pi_PH

		F.Pz += d_Pi_by_dvz * mag_edge.H;
		F.Hz += d_Pi_by_dvz * mag_edge.P; // Pi_PH

		d_Pi_by_dvz = -0.5*(ita_par + ita_1)*THIRD*(-2.0*unit_b.z*bdotPsi - 2.0*unit_perp.z*PdotPsi + 4.0*unit_Hall.z*HdotPsi)
			- 0.5*(ita_par - ita_1)*THIRD*(-2.0*unit_b.z*bdotPsi + 4.0*unit_perp.z*PdotPsi - 2.0*unit_Hall.z*HdotPsi)
			+ ita_3*(unit_perp.z*HdotPsi + unit_Hall.z*PdotPsi);
		// Pi_HH

		F.Hz += d_Pi_by_dvz*mag_edge.H;

		// *************************
		//  Now use it to create J
		// *************************

		pJ->xx += Factor*(unit_b.x*F.bx + unit_perp.x*F.Px + unit_Hall.x*F.Hx);
		pJ->xy += Factor*(unit_b.x*F.by + unit_perp.x*F.Py + unit_Hall.x*F.Hy); // d eps x / d vy
		pJ->xz += Factor*(unit_b.x*F.bz + unit_perp.x*F.Pz + unit_Hall.x*F.Hz);

		pJ->yx += Factor*(unit_b.y*F.bx + unit_perp.y*F.Px + unit_Hall.y*F.Hx);
		pJ->yy += Factor*(unit_b.y*F.by + unit_perp.y*F.Py + unit_Hall.y*F.Hy);
		pJ->yz += Factor*(unit_b.y*F.bz + unit_perp.y*F.Pz + unit_Hall.y*F.Hz);

		pJ->zx += Factor*(unit_b.z*F.bx + unit_perp.z*F.Px + unit_Hall.z*F.Hx);
		pJ->zy += Factor*(unit_b.z*F.by + unit_perp.z*F.Py + unit_Hall.z*F.Hy);
		pJ->zz += Factor*(unit_b.z*F.bz + unit_perp.z*F.Pz + unit_Hall.z*F.Hz);

	}
}

__global__ void kernelUnpack(f64 * __restrict__ pTn,
	f64 * __restrict__ pTi,
	f64 * __restrict__ pTe,
	T3 * __restrict__ pT)
{
	long const iVertex = blockDim.x * blockIdx.x + threadIdx.x;
	T3 T = pT[iVertex];
	pTn[iVertex] = T.Tn;
	pTi[iVertex] = T.Ti;
	pTe[iVertex] = T.Te;
}

__global__ void kernelUnpackWithMask(f64 * __restrict__ pTn,
	f64 * __restrict__ pTi,
	f64 * __restrict__ pTe,
	T3 * __restrict__ pT,
	bool * __restrict__ p_bMask,
	bool * __restrict__ p_bMaskblock
	)
{	
	if (p_bMaskblock[blockIdx.x] == 0) return;
	
	long const iVertex = blockDim.x * blockIdx.x + threadIdx.x;

	T3 T = pT[iVertex];
	if (p_bMask[iVertex]) pTn[iVertex] = T.Tn;
	if (p_bMask[iVertex + NUMVERTICES]) pTi[iVertex] = T.Ti;
	if (p_bMask[iVertex + 2 * NUMVERTICES]) pTe[iVertex] = T.Te;
}


__global__ void kernelUnpacktorootDN_T(
	f64 * __restrict__ psqrtDNnTn,
	f64 * __restrict__ psqrtDNTi,
	f64 * __restrict__ psqrtDNTe,
	T3 * __restrict__ pT,
	f64 * __restrict__ p_D_n,
	f64 * __restrict__ p_D_i,
	f64 * __restrict__ p_D_e,
	f64 * __restrict__ p_AreaMajor,
	nvals * __restrict__ p_n_major)
{
	long const iVertex = blockDim.x * blockIdx.x + threadIdx.x;
	T3 T = pT[iVertex];
	f64 AreaMajor = p_AreaMajor[iVertex];
	nvals n = p_n_major[iVertex];
	psqrtDNnTn[iVertex] = T.Tn*sqrt(p_D_n[iVertex]*AreaMajor*n.n_n);
	psqrtDNTi[iVertex] = T.Ti*sqrt(p_D_i[iVertex]*AreaMajor*n.n);
	psqrtDNTe[iVertex] = T.Te*sqrt(p_D_e[iVertex]*AreaMajor*n.n);
}


__global__ void kernelUnpacktorootNT(
	f64 * __restrict__ pNnTn,
	f64 * __restrict__ pNTi,
	f64 * __restrict__ pNTe,
	T3 * __restrict__ pT,
	f64 * __restrict__ p_AreaMajor,
	nvals * __restrict__ p_n_major)
{
	long const iVertex = blockDim.x * blockIdx.x + threadIdx.x;
	T3 T = pT[iVertex];
	f64 AreaMajor = p_AreaMajor[iVertex];
	nvals n = p_n_major[iVertex];
	pNnTn[iVertex] = T.Tn*sqrt(AreaMajor*n.n_n);
	pNTi[iVertex] = T.Ti*sqrt(AreaMajor*n.n);
	pNTe[iVertex] = T.Te*sqrt(AreaMajor*n.n);
}
__global__ void kernelUnpacktoNT(
	f64 * __restrict__ pNnTn,
	f64 * __restrict__ pNTi,
	f64 * __restrict__ pNTe,
	T3 * __restrict__ pT,
	f64 * __restrict__ p_AreaMajor,
	nvals * __restrict__ p_n_major)
{
	long const iVertex = blockDim.x * blockIdx.x + threadIdx.x;
	T3 T = pT[iVertex];
	f64 AreaMajor = p_AreaMajor[iVertex];
	nvals n = p_n_major[iVertex];
	pNnTn[iVertex] = T.Tn*AreaMajor*n.n_n;
	pNTi[iVertex] = T.Ti*AreaMajor*n.n;
	pNTe[iVertex] = T.Te*AreaMajor*n.n;
}

__global__ void NegateVectors(
	f64 * __restrict__ p_x1, f64 * __restrict__ p_x2, f64 * __restrict__ p_x3)
{
	long const iVertex = blockDim.x * blockIdx.x + threadIdx.x;
	p_x1[iVertex] = -p_x1[iVertex];
	p_x2[iVertex] = -p_x2[iVertex];
	p_x3[iVertex] = -p_x3[iVertex];
}

__global__ void NegateVector(
	f64 * __restrict__ p_x1)
{
	long const iVertex = blockDim.x * blockIdx.x + threadIdx.x;
	p_x1[iVertex] = -p_x1[iVertex];
}

__global__ void SubtractT3(
	T3 * __restrict__ p_result,
	T3 * __restrict__ p_a, T3 * __restrict__ p_b)
{
	long index = blockIdx.x*blockDim.x + threadIdx.x;
	T3 result;
	T3 T_1 = p_a[index];
	T3 T_2 = p_b[index];
	result.Tn = T_1.Tn - T_2.Tn;
	result.Ti = T_1.Ti - T_2.Ti;
	result.Te = T_1.Te - T_2.Te;
	p_result[index] = result;
}

__global__ void kernelAccumulateSummands3(
	structural * __restrict__ p_info_minor,
	f64_vec2 * __restrict__ p_eps_xy,
	f64 * __restrict__ p_eps_iz,
	f64 * __restrict__ p_eps_ez,
	f64_vec3 * __restrict__ p_d_eps_by_d_beta_i_,
	f64_vec3 * __restrict__ p_d_eps_by_d_beta_e_,
	f64 * __restrict__ p_sum_eps_deps_by_dbeta_i_,
	f64 * __restrict__ p_sum_eps_deps_by_dbeta_e_,
	f64 * __restrict__ p_sum_depsbydbeta_i_times_i_,
	f64 * __restrict__ p_sum_depsbydbeta_e_times_e_,
	f64 * __restrict__ p_sum_depsbydbeta_e_times_i_,
	f64 * __restrict__ p_sum_eps_sq
)
{
	__shared__ f64 sumdata_eps_i[threadsPerTileMinor];
	__shared__ f64 sumdata_eps_e[threadsPerTileMinor];
	__shared__ f64 sumdata_ii[threadsPerTileMinor];
	__shared__ f64 sumdata_ee[threadsPerTileMinor];
	__shared__ f64 sumdata_ei[threadsPerTileMinor];
	__shared__ f64 sumdata_ss[threadsPerTileMinor];

	long const iMinor = threadIdx.x + blockIdx.x * blockDim.x;
	
	sumdata_eps_i[threadIdx.x] = 0.0;
	sumdata_eps_e[threadIdx.x] = 0.0;
	sumdata_ii[threadIdx.x] = 0.0;
	sumdata_ee[threadIdx.x] = 0.0;
	sumdata_ei[threadIdx.x] = 0.0;
	sumdata_ss[threadIdx.x] = 0.0;

	structural info = p_info_minor[iMinor];
	if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS))
	{
		f64_vec2 eps_xy = p_eps_xy[iMinor];
		f64 eps_iz = p_eps_iz[iMinor];
		f64 eps_ez = p_eps_ez[iMinor];
		f64_vec3 depsbydbeta_i = p_d_eps_by_d_beta_i_[iMinor];
		f64_vec3 depsbydbeta_e = p_d_eps_by_d_beta_e_[iMinor];

		sumdata_eps_i[threadIdx.x] = depsbydbeta_i.x * eps_xy.x
			+ depsbydbeta_i.y*eps_xy.y + depsbydbeta_i.z*eps_iz;

		sumdata_eps_e[threadIdx.x] = depsbydbeta_e.x * eps_xy.x
			+ depsbydbeta_e.y*eps_xy.y + depsbydbeta_e.z*eps_ez;

		sumdata_ii[threadIdx.x] = depsbydbeta_i.dot(depsbydbeta_i);
		sumdata_ee[threadIdx.x] = depsbydbeta_e.dot(depsbydbeta_e);
		sumdata_ei[threadIdx.x] = depsbydbeta_e.x*depsbydbeta_i.x
								+ depsbydbeta_e.y*depsbydbeta_i.y; // NO z COMPONENT.
		sumdata_ss[threadIdx.x] = eps_xy.dot(eps_xy) + eps_iz*eps_iz + eps_ez*eps_ez;

	}
	
	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sumdata_eps_i[threadIdx.x] += sumdata_eps_i[threadIdx.x + k];
			sumdata_eps_e[threadIdx.x] += sumdata_eps_e[threadIdx.x + k];
			sumdata_ii[threadIdx.x] += sumdata_ii[threadIdx.x + k];
			sumdata_ee[threadIdx.x] += sumdata_ee[threadIdx.x + k];
			sumdata_ei[threadIdx.x] += sumdata_ei[threadIdx.x + k];
			sumdata_ss[threadIdx.x] += sumdata_ss[threadIdx.x + k];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			sumdata_eps_i[threadIdx.x] += sumdata_eps_i[threadIdx.x + s - 1];
			sumdata_eps_e[threadIdx.x] += sumdata_eps_e[threadIdx.x + s - 1];
			sumdata_ii[threadIdx.x] += sumdata_ii[threadIdx.x + s - 1];
			sumdata_ee[threadIdx.x] += sumdata_ee[threadIdx.x + s - 1];
			sumdata_ei[threadIdx.x] += sumdata_ei[threadIdx.x + s - 1];
			sumdata_ss[threadIdx.x] += sumdata_ss[threadIdx.x + s - 1];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_sum_eps_deps_by_dbeta_i_[blockIdx.x] = sumdata_eps_i[0];
		p_sum_eps_deps_by_dbeta_e_[blockIdx.x] = sumdata_eps_e[0];
		p_sum_depsbydbeta_i_times_i_[blockIdx.x] = sumdata_ii[0];
		p_sum_depsbydbeta_e_times_e_[blockIdx.x] = sumdata_ee[0];
		p_sum_depsbydbeta_e_times_i_[blockIdx.x] = sumdata_ei[0];
		p_sum_eps_sq[blockIdx.x] = sumdata_ss[0];			
	};
}




__global__ void kernelAccumulateSummands4(

	// We don't need to test for domain, we need to make sure the summands are zero otherwise.
	f64 * __restrict__ p_epsilon,
	f64 * __restrict__ p_d_eps_by_d_beta_J_,
	f64 * __restrict__ p_d_eps_by_d_beta_R_,

	f64 * __restrict__ p_sum_eps_deps_by_dbeta_J_,
	f64 * __restrict__ p_sum_eps_deps_by_dbeta_R_,
	f64 * __restrict__ p_sum_depsbydbeta_J_times_J_,
	f64 * __restrict__ p_sum_depsbydbeta_R_times_R_,
	f64 * __restrict__ p_sum_depsbydbeta_J_times_R_,
	f64 * __restrict__ p_sum_eps_sq
)
{
	__shared__ f64 sumdata_eps_J[threadsPerTileMajor];
	__shared__ f64 sumdata_eps_R[threadsPerTileMajor];
	__shared__ f64 sumdata_JJ[threadsPerTileMajor];
	__shared__ f64 sumdata_RR[threadsPerTileMajor];
	__shared__ f64 sumdata_JR[threadsPerTileMajor];
	__shared__ f64 sumdata_ss[threadsPerTileMajor];

	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x;

	sumdata_eps_J[threadIdx.x] = 0.0;
	sumdata_eps_R[threadIdx.x] = 0.0;
	sumdata_JJ[threadIdx.x] = 0.0;
	sumdata_RR[threadIdx.x] = 0.0;
	sumdata_JR[threadIdx.x] = 0.0;
	sumdata_ss[threadIdx.x] = 0.0;


	//structural info = p_info_minor[iMinor];
	//if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS))
	{
		f64 eps = p_epsilon[iVertex];
		f64 depsbydbeta_J = p_d_eps_by_d_beta_J_[iVertex];
		f64 depsbydbeta_R = p_d_eps_by_d_beta_R_[iVertex];

		sumdata_eps_J[threadIdx.x] = depsbydbeta_J * eps;
		sumdata_eps_R[threadIdx.x] = depsbydbeta_R * eps;
		sumdata_JJ[threadIdx.x] = depsbydbeta_J*depsbydbeta_J;
		sumdata_RR[threadIdx.x] = depsbydbeta_R*depsbydbeta_R;
		sumdata_JR[threadIdx.x] = depsbydbeta_J*depsbydbeta_R;
		sumdata_ss[threadIdx.x] = eps*eps;
	}

	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sumdata_eps_J[threadIdx.x] += sumdata_eps_J[threadIdx.x + k];
			sumdata_eps_R[threadIdx.x] += sumdata_eps_R[threadIdx.x + k];
			sumdata_JJ[threadIdx.x] += sumdata_JJ[threadIdx.x + k];
			sumdata_RR[threadIdx.x] += sumdata_RR[threadIdx.x + k];
			sumdata_JR[threadIdx.x] += sumdata_JR[threadIdx.x + k];
			sumdata_ss[threadIdx.x] += sumdata_ss[threadIdx.x + k];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			sumdata_eps_J[threadIdx.x] += sumdata_eps_J[threadIdx.x + s - 1];
			sumdata_eps_R[threadIdx.x] += sumdata_eps_R[threadIdx.x + s - 1];
			sumdata_JJ[threadIdx.x] += sumdata_JJ[threadIdx.x + s - 1];
			sumdata_RR[threadIdx.x] += sumdata_RR[threadIdx.x + s - 1];
			sumdata_JR[threadIdx.x] += sumdata_JR[threadIdx.x + s - 1];
			sumdata_ss[threadIdx.x] += sumdata_ss[threadIdx.x + s - 1];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_sum_eps_deps_by_dbeta_J_[blockIdx.x] = sumdata_eps_J[0];
		p_sum_eps_deps_by_dbeta_R_[blockIdx.x] = sumdata_eps_R[0];
		p_sum_depsbydbeta_J_times_J_[blockIdx.x] = sumdata_JJ[0];
		p_sum_depsbydbeta_R_times_R_[blockIdx.x] = sumdata_RR[0];
		p_sum_depsbydbeta_J_times_R_[blockIdx.x] = sumdata_JR[0];
		p_sum_eps_sq[blockIdx.x] = sumdata_ss[0];
	};
}



__global__ void kernelAccumulateSummands7(
	f64 * __restrict__ p_epsilon,
	f64 * __restrict__ p_d_eps_by_dbeta,
	// outputs:
	f64 * __restrict__ p_sum_eps_depsbydbeta_x8,
	f64 * __restrict__ p_sum_depsbydbeta__8x8
) {
	__shared__ f64 sumdata[threadsPerTileMajor][24]; 
	// Row-major memory layout implies that this is a contiguous array for each thread.
	
	// We can have say 24 doubles in shared. We need to sum 64 + 8 + 1 = 73 things. 24*3 = 72. hah!
	// It would be nicer then if we just called this multiple times. But it has to be for distinct input data..
	// Note that given threadsPerTileMajor = 128 we could comfortably put 48 doubles in shared and still run something.

	// The inputs are only 9 doubles so we can have them.
	// We only need the upper matrix. 1 + 2 + 3 + 4 +5 +6+7+8+9 = 45
	// So we can do it in 2 goes.

	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x;
	f64 eps = p_epsilon[iVertex];
	f64 d_eps_by_d_beta[REGRESSORS]; 
	int i;
	for (i = 0; i < REGRESSORS; i++) {
		d_eps_by_d_beta[i] = p_d_eps_by_dbeta[iVertex + i*NUMVERTICES];
	};	

#pragma unroll 
	for (i = 0; i < REGRESSORS; i++)
		sumdata[threadIdx.x][i] = eps*d_eps_by_d_beta[i];
#pragma unroll 
	for (i = 0; i < REGRESSORS; i++)
		sumdata[threadIdx.x][REGRESSORS+i] = d_eps_by_d_beta[0] *d_eps_by_d_beta[i];
#pragma unroll 
	for (i = 0; i < REGRESSORS; i++)
		sumdata[threadIdx.x][2 * REGRESSORS + i] = d_eps_by_d_beta[1] * d_eps_by_d_beta[i];
	
	// That was 24.

	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			for (int y = 0; y < 24; y++)
				sumdata[threadIdx.x][y] += sumdata[threadIdx.x + k][y];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			for (int y = 0; y < 24; y++)
				sumdata[threadIdx.x][y] += sumdata[threadIdx.x + s - 1][y];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		memcpy(&(p_sum_eps_depsbydbeta_x8[blockIdx.x*REGRESSORS]), &(sumdata[0][0]), sizeof(f64)*8);
		memcpy(&(p_sum_depsbydbeta__8x8[blockIdx.x * REGRESSORS * REGRESSORS]), &(sumdata[0][REGRESSORS]),
			2*REGRESSORS * sizeof(f64));		
	};

	__syncthreads();

#pragma unroll 
	for (i = 0; i < REGRESSORS; i++)
		sumdata[threadIdx.x][i] = d_eps_by_d_beta[2] * d_eps_by_d_beta[i];
#pragma unroll 
	for (i = 0; i < REGRESSORS; i++)
		sumdata[threadIdx.x][i + REGRESSORS] = d_eps_by_d_beta[3] * d_eps_by_d_beta[i];
#pragma unroll 
	for (i = 0; i < REGRESSORS; i++)
		sumdata[threadIdx.x][i + 2*REGRESSORS] = d_eps_by_d_beta[4] * d_eps_by_d_beta[i];

	__syncthreads();

	s = blockDim.x;
	k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			for (int y = 0; y < 24; y++)
				sumdata[threadIdx.x][y] += sumdata[threadIdx.x + k][y];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			for (int y = 0; y < 24; y++)
				sumdata[threadIdx.x][y] += sumdata[threadIdx.x + s - 1][y];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		memcpy(&(p_sum_depsbydbeta__8x8[blockIdx.x * REGRESSORS * REGRESSORS + 2 * REGRESSORS]), &(sumdata[0][0]), 3 * REGRESSORS * sizeof(f64));
	};
	__syncthreads();

#pragma unroll 
	for (i = 0; i < REGRESSORS; i++)
		sumdata[threadIdx.x][i] = d_eps_by_d_beta[5] * d_eps_by_d_beta[i];
#pragma unroll 
	for (i = 0; i < REGRESSORS; i++)
		sumdata[threadIdx.x][i + REGRESSORS] = d_eps_by_d_beta[6] * d_eps_by_d_beta[i];
#pragma unroll 
	for (i = 0; i < REGRESSORS; i++)
		sumdata[threadIdx.x][i + 2*REGRESSORS] = d_eps_by_d_beta[7] * d_eps_by_d_beta[i];

	__syncthreads();

	s = blockDim.x;
	k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			for (int y = 0; y < 24; y++)
				sumdata[threadIdx.x][y] += sumdata[threadIdx.x + k][y];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			for (int y = 0; y < 24; y++)
				sumdata[threadIdx.x][y] += sumdata[threadIdx.x + s - 1][y];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		// Caught ourselves out. We need to do what, quadrants of matrix? It's 8 x 8.
		// We can do rows, if we do 3 sets of rows.

		memcpy(&(p_sum_depsbydbeta__8x8[blockIdx.x * REGRESSORS * REGRESSORS + 5 * REGRESSORS]),
			&(sumdata[0][0]), 3 * REGRESSORS * sizeof(f64));
	};
	
}


__global__ void kernelAccumulateSummands6(
	f64 * __restrict__ p_epsilon,
	f64_vec4 * __restrict__ p_d_eps_by_dbetaJ_x4,
	f64_vec4 * __restrict__ p_d_eps_by_dbetaR_x4,

	// outputs:
	f64_vec4 * __restrict__ p_sum_eps_depsbydbeta_J_x4,
	f64_vec4 * __restrict__ p_sum_eps_depsbydbeta_R_x4,
	f64 * __restrict__ p_sum_depsbydbeta__8x8,  // do we want to store 64 things in memory? .. we don't.
	f64 * __restrict__ p_sum_eps_eps_
) {
	__shared__ f64 sumdata[threadsPerTileMajor][24];
	// Row-major memory layout implies that this is a contiguous array for each thread.

	// We can have say 24 doubles in shared. We need to sum 64 + 8 + 1 = 73 things. 24*3 = 72. hah!
	// It would be nicer then if we just called this multiple times. But it has to be for distinct input data..
	// Note that given threadsPerTileMajor = 128 we could comfortably put 48 doubles in shared and still run something.

	// The inputs are only 9 doubles so we can have them.
	// We only need the upper matrix. 1 + 2 + 3 + 4 +5 +6+7+8+9 = 45
	// So we can do it in 2 goes.

	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x;
	f64 eps = p_epsilon[iVertex];
	f64_vec4 d_eps_by_d_beta_J;
	f64_vec4 d_eps_by_d_beta_R;
	memcpy(&d_eps_by_d_beta_J, &(p_d_eps_by_dbetaJ_x4[iVertex]), sizeof(f64_vec4));
	memcpy(&d_eps_by_d_beta_R, &(p_d_eps_by_dbetaR_x4[iVertex]), sizeof(f64_vec4));

	sumdata[threadIdx.x][0] = eps*d_eps_by_d_beta_J.x[0];
	sumdata[threadIdx.x][1] = eps*d_eps_by_d_beta_J.x[1];
	sumdata[threadIdx.x][2] = eps*d_eps_by_d_beta_J.x[2];
	sumdata[threadIdx.x][3] = eps*d_eps_by_d_beta_J.x[3];
	sumdata[threadIdx.x][4] = eps*d_eps_by_d_beta_R.x[0];
	sumdata[threadIdx.x][5] = eps*d_eps_by_d_beta_R.x[1];
	sumdata[threadIdx.x][6] = eps*d_eps_by_d_beta_R.x[2];
	sumdata[threadIdx.x][7] = eps*d_eps_by_d_beta_R.x[3];

	sumdata[threadIdx.x][8] = d_eps_by_d_beta_J.x[0] * d_eps_by_d_beta_J.x[0];
	sumdata[threadIdx.x][9] = d_eps_by_d_beta_J.x[0] * d_eps_by_d_beta_J.x[1];
	sumdata[threadIdx.x][10] = d_eps_by_d_beta_J.x[0] * d_eps_by_d_beta_J.x[2];
	sumdata[threadIdx.x][11] = d_eps_by_d_beta_J.x[0] * d_eps_by_d_beta_J.x[3];
	sumdata[threadIdx.x][12] = d_eps_by_d_beta_J.x[0] * d_eps_by_d_beta_R.x[0];
	sumdata[threadIdx.x][13] = d_eps_by_d_beta_J.x[0] * d_eps_by_d_beta_R.x[1];
	sumdata[threadIdx.x][14] = d_eps_by_d_beta_J.x[0] * d_eps_by_d_beta_R.x[2];
	sumdata[threadIdx.x][15] = d_eps_by_d_beta_J.x[0] * d_eps_by_d_beta_R.x[3];

	sumdata[threadIdx.x][16] = d_eps_by_d_beta_J.x[1] * d_eps_by_d_beta_J.x[0];
	sumdata[threadIdx.x][17] = d_eps_by_d_beta_J.x[1] * d_eps_by_d_beta_J.x[1];
	sumdata[threadIdx.x][18] = d_eps_by_d_beta_J.x[1] * d_eps_by_d_beta_J.x[2];
	sumdata[threadIdx.x][19] = d_eps_by_d_beta_J.x[1] * d_eps_by_d_beta_J.x[3];
	sumdata[threadIdx.x][20] = d_eps_by_d_beta_J.x[1] * d_eps_by_d_beta_R.x[0];
	sumdata[threadIdx.x][21] = d_eps_by_d_beta_J.x[1] * d_eps_by_d_beta_R.x[1];
	sumdata[threadIdx.x][22] = d_eps_by_d_beta_J.x[1] * d_eps_by_d_beta_R.x[2];
	sumdata[threadIdx.x][23] = d_eps_by_d_beta_J.x[1] * d_eps_by_d_beta_R.x[3];

	// Can we fit the rest into 24? yes

	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			for (int y = 0; y < 24; y++)
				sumdata[threadIdx.x][y] += sumdata[threadIdx.x + k][y];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			for (int y = 0; y < 24; y++)
				sumdata[threadIdx.x][y] += sumdata[threadIdx.x + s - 1][y];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		memcpy(&(p_sum_eps_depsbydbeta_J_x4[blockIdx.x]), &(sumdata[0][0]), sizeof(f64_vec4));
		memcpy(&(p_sum_eps_depsbydbeta_R_x4[blockIdx.x]), &(sumdata[0][4]), sizeof(f64_vec4));

		// Now careful - let's fill in one row at a time. 
		memcpy(&(p_sum_depsbydbeta__8x8[blockIdx.x * 8 * 8]), &(sumdata[0][8]), 16 * sizeof(f64));

		if (sumdata[0][17] < 0.0) printf("blockIdx.x %d sumdata[0][17] %1.5E \n",
			blockIdx.x, sumdata[0][17]);

	};

	__syncthreads();

	sumdata[threadIdx.x][0] = d_eps_by_d_beta_J.x[2] * d_eps_by_d_beta_J.x[0];
	sumdata[threadIdx.x][1] = d_eps_by_d_beta_J.x[2] * d_eps_by_d_beta_J.x[1];
	sumdata[threadIdx.x][2] = d_eps_by_d_beta_J.x[2] * d_eps_by_d_beta_J.x[2];
	sumdata[threadIdx.x][3] = d_eps_by_d_beta_J.x[2] * d_eps_by_d_beta_J.x[3];
	sumdata[threadIdx.x][4] = d_eps_by_d_beta_J.x[2] * d_eps_by_d_beta_R.x[0];
	sumdata[threadIdx.x][5] = d_eps_by_d_beta_J.x[2] * d_eps_by_d_beta_R.x[1];
	sumdata[threadIdx.x][6] = d_eps_by_d_beta_J.x[2] * d_eps_by_d_beta_R.x[2];
	sumdata[threadIdx.x][7] = d_eps_by_d_beta_J.x[2] * d_eps_by_d_beta_R.x[3];

	sumdata[threadIdx.x][8] = d_eps_by_d_beta_J.x[3] * d_eps_by_d_beta_J.x[0];
	sumdata[threadIdx.x][9] = d_eps_by_d_beta_J.x[3] * d_eps_by_d_beta_J.x[1];
	sumdata[threadIdx.x][10] = d_eps_by_d_beta_J.x[3] * d_eps_by_d_beta_J.x[2];
	sumdata[threadIdx.x][11] = d_eps_by_d_beta_J.x[3] * d_eps_by_d_beta_J.x[3];
	sumdata[threadIdx.x][12] = d_eps_by_d_beta_J.x[3] * d_eps_by_d_beta_R.x[0];
	sumdata[threadIdx.x][13] = d_eps_by_d_beta_J.x[3] * d_eps_by_d_beta_R.x[1];
	sumdata[threadIdx.x][14] = d_eps_by_d_beta_J.x[3] * d_eps_by_d_beta_R.x[2];
	sumdata[threadIdx.x][15] = d_eps_by_d_beta_J.x[3] * d_eps_by_d_beta_R.x[3];

	sumdata[threadIdx.x][16] = d_eps_by_d_beta_R.x[0] * d_eps_by_d_beta_J.x[0];
	sumdata[threadIdx.x][17] = d_eps_by_d_beta_R.x[0] * d_eps_by_d_beta_J.x[1];
	sumdata[threadIdx.x][18] = d_eps_by_d_beta_R.x[0] * d_eps_by_d_beta_J.x[2];
	sumdata[threadIdx.x][19] = d_eps_by_d_beta_R.x[0] * d_eps_by_d_beta_J.x[3];
	sumdata[threadIdx.x][20] = d_eps_by_d_beta_R.x[0] * d_eps_by_d_beta_R.x[0];
	sumdata[threadIdx.x][21] = d_eps_by_d_beta_R.x[0] * d_eps_by_d_beta_R.x[1];
	sumdata[threadIdx.x][22] = d_eps_by_d_beta_R.x[0] * d_eps_by_d_beta_R.x[2];
	sumdata[threadIdx.x][23] = d_eps_by_d_beta_R.x[0] * d_eps_by_d_beta_R.x[3];

	__syncthreads();

	s = blockDim.x;
	k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			for (int y = 0; y < 24; y++)
				sumdata[threadIdx.x][y] += sumdata[threadIdx.x + k][y];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			for (int y = 0; y < 24; y++)
				sumdata[threadIdx.x][y] += sumdata[threadIdx.x + s - 1][y];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		// Caught ourselves out. We need to do what, quadrants of matrix? It's 8 x 8.
		// We can do rows, if we do 3 sets of rows.

		memcpy(&(p_sum_depsbydbeta__8x8[blockIdx.x * 8 * 8 + 8 + 8]), &(sumdata[0][0]), 24 * sizeof(f64));
	};
	__syncthreads();


	sumdata[threadIdx.x][0] = d_eps_by_d_beta_R.x[1] * d_eps_by_d_beta_J.x[0];
	sumdata[threadIdx.x][1] = d_eps_by_d_beta_R.x[1] * d_eps_by_d_beta_J.x[1];
	sumdata[threadIdx.x][2] = d_eps_by_d_beta_R.x[1] * d_eps_by_d_beta_J.x[2];
	sumdata[threadIdx.x][3] = d_eps_by_d_beta_R.x[1] * d_eps_by_d_beta_J.x[3];
	sumdata[threadIdx.x][4] = d_eps_by_d_beta_R.x[1] * d_eps_by_d_beta_R.x[0];
	sumdata[threadIdx.x][5] = d_eps_by_d_beta_R.x[1] * d_eps_by_d_beta_R.x[1];
	sumdata[threadIdx.x][6] = d_eps_by_d_beta_R.x[1] * d_eps_by_d_beta_R.x[2];
	sumdata[threadIdx.x][7] = d_eps_by_d_beta_R.x[1] * d_eps_by_d_beta_R.x[3];

	sumdata[threadIdx.x][8] = d_eps_by_d_beta_R.x[2] * d_eps_by_d_beta_J.x[0];
	sumdata[threadIdx.x][9] = d_eps_by_d_beta_R.x[2] * d_eps_by_d_beta_J.x[1];
	sumdata[threadIdx.x][10] = d_eps_by_d_beta_R.x[2] * d_eps_by_d_beta_J.x[2];
	sumdata[threadIdx.x][11] = d_eps_by_d_beta_R.x[2] * d_eps_by_d_beta_J.x[3];
	sumdata[threadIdx.x][12] = d_eps_by_d_beta_R.x[2] * d_eps_by_d_beta_R.x[0];
	sumdata[threadIdx.x][13] = d_eps_by_d_beta_R.x[2] * d_eps_by_d_beta_R.x[1];
	sumdata[threadIdx.x][14] = d_eps_by_d_beta_R.x[2] * d_eps_by_d_beta_R.x[2];
	sumdata[threadIdx.x][15] = d_eps_by_d_beta_R.x[2] * d_eps_by_d_beta_R.x[3];

	sumdata[threadIdx.x][16] = d_eps_by_d_beta_R.x[3] * d_eps_by_d_beta_J.x[0];
	sumdata[threadIdx.x][17] = d_eps_by_d_beta_R.x[3] * d_eps_by_d_beta_J.x[1];
	sumdata[threadIdx.x][18] = d_eps_by_d_beta_R.x[3] * d_eps_by_d_beta_J.x[2];
	sumdata[threadIdx.x][19] = d_eps_by_d_beta_R.x[3] * d_eps_by_d_beta_J.x[3];
	sumdata[threadIdx.x][20] = d_eps_by_d_beta_R.x[3] * d_eps_by_d_beta_R.x[0];
	sumdata[threadIdx.x][21] = d_eps_by_d_beta_R.x[3] * d_eps_by_d_beta_R.x[1];
	sumdata[threadIdx.x][22] = d_eps_by_d_beta_R.x[3] * d_eps_by_d_beta_R.x[2];
	sumdata[threadIdx.x][23] = d_eps_by_d_beta_R.x[3] * d_eps_by_d_beta_R.x[3];

	__syncthreads();

	s = blockDim.x;
	k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			for (int y = 0; y < 24; y++)
				sumdata[threadIdx.x][y] += sumdata[threadIdx.x + k][y];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			for (int y = 0; y < 24; y++)
				sumdata[threadIdx.x][y] += sumdata[threadIdx.x + s - 1][y];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		// Caught ourselves out. We need to do what, quadrants of matrix? It's 8 x 8.
		// We can do rows, if we do 3 sets of rows.

		memcpy(&(p_sum_depsbydbeta__8x8[blockIdx.x * 8 * 8 + 40]), &(sumdata[0][0]), 24 * sizeof(f64));
	};
	__syncthreads();

	sumdata[threadIdx.x][0] = eps*eps;

	__syncthreads();

	s = blockDim.x;
	k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sumdata[threadIdx.x][0] += sumdata[threadIdx.x + k][0];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			sumdata[threadIdx.x][0] += sumdata[threadIdx.x + s - 1][0];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		// Caught ourselves out. We need to do what, quadrants of matrix? It's 8 x 8.
		// We can do rows, if we do 3 sets of rows.

		p_sum_eps_eps_[blockIdx.x] = sumdata[0][0];

	};
}
__global__ void kernelCalculateOverallVelocitiesVertices(
	structural * __restrict__ p_info_minor,
	v4 * __restrict__ p_vie_major,
	f64_vec3 * __restrict__ p_v_n_major,
	nvals * __restrict__ p_n_major,
	f64_vec2 * __restrict__ p_v_overall_major,
	
	ShardModel * __restrict__ p_shards_n,
	ShardModel * __restrict__ p_shards_n_n,
	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBCtri_verts
	)
{
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];
	
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX
	structural const info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
	f64_vec2 v_overall(0.0, 0.0);
	
	long const StartMinor = threadsPerTileMinor*blockIdx.x;
	{
		structural infotemp[2];
		memcpy(infotemp, p_info_minor + threadIdx.x * 2 + threadsPerTileMinor*blockIdx.x, sizeof(structural) * 2);
		shared_pos[threadIdx.x * 2] = infotemp[0].pos;
		shared_pos[threadIdx.x * 2 + 1] = infotemp[1].pos;
	}
	long const EndMinor = threadsPerTileMinor*blockIdx.x + 2 * blockDim.x;

	__syncthreads();

	if (info.flag == DOMAIN_VERTEX)
	{
		v4 const vie = p_vie_major[iVertex];
		f64_vec3 const v_n = p_v_n_major[iVertex];
		nvals const n = p_n_major[iVertex];

		short tri_len = info.neigh_len;
		long izTri[MAXNEIGH];
		char szPBC[MAXNEIGH];
		memcpy(izTri, p_izTri + iVertex * MAXNEIGH, sizeof(long) * MAXNEIGH);
		memcpy(szPBC, p_szPBCtri_verts + iVertex*MAXNEIGH, sizeof(char)*MAXNEIGH);

		// Our own drift:

		v_overall = (vie.vxy*(m_e + m_i)*n.n +
			v_n.xypart()*m_n*n.n_n) /
			((m_e + m_i)*n.n + m_n*n.n_n);
		f64_vec2 v_adv = v_overall;

		if (TEST) printf("%d vie.vxy %1.9E %1.9E v_n %1.9E %1.9E n %1.9E nn %1.9E\n",
			iVertex, vie.vxy.x, vie.vxy.y, v_n.x, v_n.y, n.n, n.n_n);

		// Now add in drift towards barycenter

		// 1. Work out where barycenter is, from n_shards
		// (we have to load for both n and n_n, and need to be careful about combining to find overall barycenter)
		f64_vec2 barycenter;
		ShardModel shards_n, shards_n_n;
		memcpy(&shards_n, p_shards_n + iVertex, sizeof(ShardModel));
		memcpy(&shards_n_n, p_shards_n_n + iVertex, sizeof(ShardModel));
		
		// Sum in triangles the integral of n against (x,y):
		// Sum in triangles the integral of n:
		f64_vec2 numer(0.0,0.0);
		f64 mass = 0.0, Areatot = 0.0;
		short inext, i;
		f64_vec2 pos0, pos1;
		f64 Area_tri;
		f64 wt0, wt1, wtcent;
		for (i = 0; i < tri_len; i++)
		{
			inext = i + 1; if (inext == tri_len) inext = 0;
			// Collect positions:
			if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
			{
				pos0 = shared_pos[izTri[i] - StartMinor];
			} else {
				pos0 = p_info_minor[izTri[i]].pos;
			}
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) pos0 = Clockwise_d*pos0;
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) pos0 = Anticlockwise_d*pos0;
			if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
			{
				pos1 = shared_pos[izTri[inext] - StartMinor];
			} else {
				pos1 = p_info_minor[izTri[inext]].pos;
			}
			if (szPBC[inext] == ROTATE_ME_CLOCKWISE) pos1 = Clockwise_d*pos1;
			if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) pos1 = Anticlockwise_d*pos1;

			// Get Area_tri:

			Area_tri = 0.5*fabs(pos0.x*info.pos.y + info.pos.x*pos1.y + pos1.x*pos0.y
				              - info.pos.x*pos0.y - pos1.x*info.pos.y - pos0.x*pos1.y);
			
			wt0 = (shards_n.n[i] + shards_n_n.n[i]) / 12.0
				+ (shards_n.n[inext] + shards_n_n.n[inext]) / 24.0
				+ (shards_n.n_cent + shards_n_n.n_cent) / 24.0;
			wt1 = (shards_n.n[i] + shards_n_n.n[i]) / 24.0
				+ (shards_n.n[inext] + shards_n_n.n[inext]) / 12.0
				+ (shards_n.n_cent + shards_n_n.n_cent) / 24.0;
			wtcent = (shards_n.n[i] + shards_n_n.n[i]) / 24.0
				+ (shards_n.n[inext] + shards_n_n.n[inext]) / 24.0
				+ (shards_n.n_cent + shards_n_n.n_cent) / 12.0;

			numer.x += 2.0*Area_tri*(pos0.x*wt0 + pos1.x*wt1 + info.pos.x*wtcent);
			numer.y += 2.0*Area_tri*(pos0.y*wt0 + pos1.y*wt1 + info.pos.y*wtcent);

			mass += THIRD*Area_tri*(shards_n.n[i] + shards_n_n.n[i] +
				shards_n.n[inext] + shards_n_n.n[inext] + shards_n.n_cent + shards_n_n.n_cent);
			Areatot += Area_tri; 

		//	if (iVertex == VERTCHOSEN)
		//		printf("%d info.pos %1.9E %1.9E  pos0 %1.9E %1.9E pos1 %1.9E %1.9E\n"
		//			"Area_shard %1.10E mass_shard %1.10E shards_n %1.9E %1.9E %1.9E\n\n",
		//			iVertex,  info.pos.x, info.pos.y, pos0.x, pos0.y,pos1.x,pos1.y,
		//			Area_tri, THIRD*Area_tri*(shards_n.n[i] + shards_n_n.n[i] +
		//				shards_n.n[inext] + shards_n_n.n[inext] + shards_n.n_cent + shards_n_n.n_cent),
		//			shards_n.n_cent + shards_n_n.n_cent,
		//			shards_n.n[i] + shards_n_n.n[i],
		//			shards_n.n[inext] + shards_n_n.n[inext]
		//			);
		}

		// Divide one by the other to give the barycenter:
		barycenter = numer / mass;

				
		// Having great difficulty seeing why we get the result that we do.
		// How close are different points?

		// I think we want to work on optimizing the distance to neighbour relative to average density of this and neighbour.
		// That should control triangle area per triangle density.
		// Alternatively, we could directly look at which triangle centroid propels us away because the triangle is too small ...

		// Also: splint at absolute distance of 4 micron, for now.

		
		// 2. Drift towards it is proportional to normalized distance squared.
		// square root of vertcell area, to normalize dist from barycenter:

		f64_vec2 to_bary = barycenter - info.pos;
		
		f64 factor = min( 4.0 * to_bary.dot(to_bary) / Areatot, 1.0); // r = sqrt(Area)/pi and we take (delta/r)^2
		// uplift by 4/PI

		//f64 distance = to_bary.modulus();
		//f64_vec2 unit_to_bary = to_bary / distance;
		
		if (TEST) {
			printf("iVertex %d pos %1.9E %1.9E barycenter %1.9E %1.9E v_overall %1.9E %1.9E \n"
				"factor %1.9E sqrt(Areatot) %1.9E |to_bary| %1.9E \n",
				iVertex, info.pos.x, info.pos.y, barycenter.x, barycenter.y, v_overall.x, v_overall.y,
				factor, sqrt(Areatot), to_bary.modulus());
		}
		
		// v_overall += unit_to_bary * factor * 1.0e10;
		// We used area of vertcell rather than area with its neighbours at corners.
		// 1e10 > any v
		// but also not too big? If dist = 1e-3 then 1e10*1e-12 = 1e-2. Oh dear.
		// That is too big of a hop.
		// The "speed" should be proportional to actual distance.

		// We can use 1e10 if ||to_bary|| is 1e-2 or more.
		// We need to multiply by distance/1e-2

		v_overall += to_bary * factor * 1.0e12;

		// So then we do want to cancel some of hv as an extra term in case v = ~1e8 and the cell is only 2 micron wide?
		// Hmm
		v_overall += to_bary * factor * max(-v_adv.dot(to_bary)/to_bary.dot(to_bary),0.0);

		// There is almost certainly a more elegant way that puts them together.
		// If v faces to_bary then we do nothing.
		
		if (TEST) 
			printf("%d additional v %1.9E %1.9E cancelv addition %1.9E %1.9E\n\n", 
				iVertex, to_bary.x * factor * 1.0e12,
				to_bary.y * factor * 1.0e12, 
				to_bary.x * factor * max(-v_adv.dot(to_bary) / to_bary.dot(to_bary), 0.0),
				to_bary.y * factor * max(-v_adv.dot(to_bary) / to_bary.dot(to_bary), 0.0)				
				);
		
		// To watch out for: overshooting because hv takes us towards barycenter



		// Hope this stuff works well because the equal masses takes a bit of doing.		
	};
	p_v_overall_major[iVertex] = v_overall;
}

__global__ void kernelAverageOverallVelocitiesTriangles(
	f64_vec2 * __restrict__ p_overall_v_major,
	f64_vec2 * __restrict__ p_overall_v_minor,
	structural * __restrict__ p_info,
	LONG3 * __restrict__ p_tri_corner_index,
	CHAR4 * __restrict__ p_tri_periodic_corner_flags
)
{
	__shared__ f64_vec2 shared_v[threadsPerTileMajor];
	__shared__ f64_vec2 shared_pos[threadsPerTileMajor];


	long const index = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX
	if (threadIdx.x < threadsPerTileMajor)
	{
		long getindex = blockIdx.x * threadsPerTileMajor + threadIdx.x;
		shared_v[threadIdx.x] = p_overall_v_major[getindex];
		shared_pos[threadIdx.x] = p_info[BEGINNING_OF_CENTRAL + getindex].pos;
	};
	long const StartMajor = blockIdx.x*threadsPerTileMajor;
	long const EndMajor = StartMajor + threadsPerTileMajor;
	LONG3 const tri_corner_index = p_tri_corner_index[index];
	CHAR4 const tri_corner_per_flag = p_tri_periodic_corner_flags[index];
	structural info = p_info[index];

	__syncthreads();

	
	// Thoughts:
	// We want it to be the motion of the circumcenter .. but linear interpolation of v is probably good enough?
	// That won't work - consider right-angled.
	// Silver standard approach: empirical estimate of time-derivative of cc position.

	f64_vec2 v(0.0, 0.0);

	if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS)) {

		f64_vec2 poscorner0, poscorner1, poscorner2, vcorner0, vcorner1, vcorner2;

		if ((tri_corner_index.i1 >= StartMajor) && (tri_corner_index.i1 < EndMajor))
		{
			poscorner0 = shared_pos[tri_corner_index.i1 - StartMajor];
			vcorner0 = shared_v[tri_corner_index.i1 - StartMajor];
		}
		else {
			poscorner0 = p_info[tri_corner_index.i1 + BEGINNING_OF_CENTRAL].pos;
			vcorner0 = p_overall_v_major[tri_corner_index.i1];
		};
		if (tri_corner_per_flag.per0 == ROTATE_ME_CLOCKWISE) {
			poscorner0 = Clockwise_d*poscorner0;
			vcorner0 = Clockwise_d*vcorner0;
		}
		if (tri_corner_per_flag.per0 == ROTATE_ME_ANTICLOCKWISE) {
			poscorner0 = Anticlockwise_d*poscorner0;
			vcorner0 = Anticlockwise_d*vcorner0;
		}

		if ((tri_corner_index.i2 >= StartMajor) && (tri_corner_index.i2 < EndMajor))
		{
			poscorner1 = shared_pos[tri_corner_index.i2 - StartMajor];
			vcorner1 = shared_v[tri_corner_index.i2 - StartMajor];
		}
		else {
			poscorner1 = p_info[tri_corner_index.i2 + BEGINNING_OF_CENTRAL].pos;
			vcorner1 = p_overall_v_major[tri_corner_index.i2];
		};
		if (tri_corner_per_flag.per1 == ROTATE_ME_CLOCKWISE) {
			poscorner1 = Clockwise_d*poscorner1;
			vcorner1 = Clockwise_d*vcorner1;
		}
		if (tri_corner_per_flag.per1 == ROTATE_ME_ANTICLOCKWISE) {
			poscorner1 = Anticlockwise_d*poscorner1;
			vcorner1 = Anticlockwise_d*vcorner1;
		}

		if ((tri_corner_index.i3 >= StartMajor) && (tri_corner_index.i3 < EndMajor))
		{
			poscorner2 = shared_pos[tri_corner_index.i3 - StartMajor];
			vcorner2 = shared_v[tri_corner_index.i3 - StartMajor];
		}
		else {
			poscorner2 = p_info[tri_corner_index.i3 + BEGINNING_OF_CENTRAL].pos;
			vcorner2 = p_overall_v_major[tri_corner_index.i3];
		};

		if (tri_corner_per_flag.per2 == ROTATE_ME_CLOCKWISE) {
			poscorner2 = Clockwise_d*poscorner2;
			vcorner2 = Clockwise_d*vcorner2;
		}
		if (tri_corner_per_flag.per2 == ROTATE_ME_ANTICLOCKWISE) {
			poscorner2 = Anticlockwise_d*poscorner2;
			vcorner2 = Anticlockwise_d*vcorner2;
		}
		
		f64_vec2 pos;
		f64_vec2 Bb = poscorner1 - poscorner0;
		f64_vec2 C = poscorner2 - poscorner0;
		f64 D = 2.0*(Bb.x*C.y - Bb.y*C.x);
		f64 modB = Bb.x*Bb.x + Bb.y*Bb.y;
		f64 modC = C.x*C.x + C.y*C.y;
		pos.x = (C.y*modB - Bb.y*modC) / D + poscorner0.x;
		pos.y = (Bb.x*modC - C.x*modB) / D + poscorner0.y;

		// choose step where h*(sqrt(sum of vcorner^2)) is 1e-9 cm
		f64 temp = sqrt(vcorner0.dot(vcorner0) + vcorner1.dot(vcorner1) + vcorner2.dot(vcorner2));
		f64 h_deriv = 1.0e-9 / temp;

		poscorner0 += h_deriv*vcorner0;
		poscorner1 += h_deriv*vcorner1;
		poscorner2 += h_deriv*vcorner2;

		f64_vec2 newpos;
		Bb = poscorner1 - poscorner0;
		C = poscorner2 - poscorner0;
		D = 2.0*(Bb.x*C.y - Bb.y*C.x);
		modB = Bb.x*Bb.x + Bb.y*Bb.y;
		modC = C.x*C.x + C.y*C.y;
		newpos.x = (C.y*modB - Bb.y*modC) / D + poscorner0.x;
		newpos.y = (Bb.x*modC - C.x*modB) / D + poscorner0.y;

		if (info.flag == CROSSING_INS) {
			f64_vec2 pos2 = pos;
			pos2.project_to_radius(pos, DEVICE_RADIUS_INSULATOR_OUTER);
			pos2 = newpos;
			pos2.project_to_radius(newpos, DEVICE_RADIUS_INSULATOR_OUTER);
		};

		v = (newpos - pos) / h_deriv;
		// Empirical estimate of derivative. Saves me messing about with taking derivative of circumcenter position.

		//if (index == 42940)	
		//	printf("\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ \n"
		//		"pos %1.11E %1.11E newpos-pos %1.11E %1.11E hderiv %1.11E\n "
		//		"vcorner0 %1.11E %1.11E vcorner1 %1.11E %1.11E corner2 %1.11E %1.11E v %1.11E %1.11E\n"
		//		"poscorner0 %1.11E %1.11E poscorner1 %1.11E %1.11E poscorner2 %1.11E %1.11E \n"
		//		" @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ \n\n",
		//		pos.x, pos.y, newpos.x - pos.x, newpos.y - pos.y, h_deriv,
		//		vcorner0.x, vcorner0.y, vcorner1.x, vcorner1.y, vcorner2.x, vcorner2.y, v.x, v.y,
		//		poscorner0.x, poscorner0.y, poscorner1.x, poscorner1.y, poscorner2.x, poscorner2.y);			
	

	} else {
		// leave it == 0		
	};
	p_overall_v_minor[index] = v;
}



__global__ void kernelAdvectPositionsVertex(
	f64 h_use,
	structural * __restrict__ p_info_src_major,
	structural * __restrict__ p_info_dest_major,
	f64_vec2 * __restrict__ p_v_overall_major,
	nvals * __restrict__ p_n_major,
	long * __restrict__ p_izNeigh_vert,
	char * __restrict__ p_szPBCneigh_vert
	// what else is copied?
	// something we can easily copy over
	// with cudaMemcpy, even ahead of steps?
	// Is there a reason we cannot put into the above routine
	// with a split for "this is a vertex->just use its overall v"
)
{
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x; 
	structural info = p_info_src_major[iVertex];
	f64_vec2 overall_v = p_v_overall_major[iVertex];
	f64_vec2 oldpos = info.pos;
	info.pos += h_use*overall_v;

	// Now make correction
	long izNeigh[MAXNEIGH_d];
	char PBCneigh[MAXNEIGH_d];
	if (info.flag == DOMAIN_VERTEX) {
		memcpy(izNeigh, p_izNeigh_vert + iVertex*MAXNEIGH_d, sizeof(long)*MAXNEIGH_d);
		memcpy(PBCneigh, p_szPBCneigh_vert + iVertex*MAXNEIGH_d, sizeof(char)*MAXNEIGH_d);
		short i, iMost = -1, iLeast = -1;
		f64 most = 0.0 , least = 1.0e100;
		f64_vec2 leastpos, mostpos, diff;
		nvals n_neigh;
		nvals n_own;
		n_own = p_n_major[iVertex];
		structural infoneigh;
		f64 ratio;
		char buff[255];

		for (i = 0; i < info.neigh_len; i++)
		{ 
			n_neigh = p_n_major[izNeigh[i]];
			infoneigh = p_info_src_major[izNeigh[i]];
			if (infoneigh.flag == DOMAIN_VERTEX) {
				char PBC = PBCneigh[i];
				if (PBC == ROTATE_ME_CLOCKWISE) infoneigh.pos = Clockwise_d*infoneigh.pos;
				if (PBC == ROTATE_ME_ANTICLOCKWISE) infoneigh.pos = Anticlockwise_d*infoneigh.pos;
				f64_vec2 diff = infoneigh.pos - info.pos;
				f64 deltasq = diff.dot(diff);
				ratio = deltasq * (n_own.n + n_own.n_n + n_neigh.n + n_neigh.n_n);
				// Let's think carefully. Area of tri that is 1/2 delta^2 should ~ 1/n
				if (ratio > most) { iMost = i; most = ratio; mostpos = infoneigh.pos; }
				if (ratio < least) { iLeast = i; least = ratio; leastpos = infoneigh.pos; }
			}
		};

		if (most > 2.0*least) {
			// we need to move in the direction away from the 'least' distant point
			// as long as it's improving most/least
		
			for (i = 0; i < info.neigh_len; i++)
			{
				n_neigh = p_n_major[izNeigh[i]];
				infoneigh = p_info_src_major[izNeigh[i]];
				if (infoneigh.flag == DOMAIN_VERTEX) {
					char PBC = PBCneigh[i];
					if (PBC == ROTATE_ME_CLOCKWISE) infoneigh.pos = Clockwise_d*infoneigh.pos;
					if (PBC == ROTATE_ME_ANTICLOCKWISE) infoneigh.pos = Anticlockwise_d*infoneigh.pos;
					f64_vec2 diff = infoneigh.pos - info.pos;
					f64 deltasq = diff.dot(diff);
					ratio = deltasq * (n_own.n + n_own.n_n + n_neigh.n + n_neigh.n_n);
					// Let's think carefully. Area of tri that is 1/2 delta^2 should ~ 1/n
					printf("%d i %d izNeigh %d ratio %1.14E \n",iVertex, i, izNeigh[i], ratio);
				}
			}
						
			diff = info.pos-leastpos;
			// We want squared modulus of dist to equal half of most			
			f64_vec2 oldpos2 = info.pos;
			info.pos += diff*(sqrt(most/(2.0*least))-1.0);
			printf("%d: most %1.10E least %1.10E diff %1.9E %1.9E \noldpos %1.12E %1.12E old2pos %1.12E %1.12E info.pos %1.12E %1.12E\n",
				iVertex, most, least, diff.x, diff.y, oldpos.x, oldpos.y, oldpos2.x, oldpos2.y, info.pos.x, info.pos.y);

			for (i = 0; i < info.neigh_len; i++)
			{
				n_neigh = p_n_major[izNeigh[i]];
				infoneigh = p_info_src_major[izNeigh[i]];
				if (infoneigh.flag == DOMAIN_VERTEX) {
					char PBC = PBCneigh[i];
					if (PBC == ROTATE_ME_CLOCKWISE) infoneigh.pos = Clockwise_d*infoneigh.pos;
					if (PBC == ROTATE_ME_ANTICLOCKWISE) infoneigh.pos = Anticlockwise_d*infoneigh.pos;
					f64_vec2 diff = infoneigh.pos - info.pos;
					f64 deltasq = diff.dot(diff);
					ratio = deltasq * (n_own.n + n_own.n_n + n_neigh.n + n_neigh.n_n);
					// Let's think carefully. Area of tri that is 1/2 delta^2 should ~ 1/n
					printf("%d i %d izNeigh %d ratio %1.14E \n",iVertex, i, izNeigh[i], ratio);
				}
			};
		};

//		least = 1.0e100;
//		iLeast = -1;
//		for (i = 0; i < info.neigh_len; i++)
//		{
//			n_neigh = p_n_major[izNeigh[i]];
//			infoneigh = p_info_src_major[izNeigh[i]];
//			char PBC = PBCneigh[izNeigh[i]];
//			if (PBC == ROTATE_ME_CLOCKWISE) infoneigh.pos = Clockwise2*infoneigh.pos;
//			if (PBC == ROTATE_ME_ANTICLOCKWISE) infoneigh.pos = Anticlockwise2*infoneigh.pos;
//			f64_vec2 diff = infoneigh.pos - info.pos;
//			f64 deltasq = diff.dot(diff);			
//			if (deltasq < least) { iLeast = i; least = deltasq; leastpos = infoneigh.pos; }
//		}
//#define MINDIST 0.0003 // 3 micron
//
//		if (least < MINDIST) {
//
//		}

		overall_v = (info.pos-oldpos) / h_use;
		p_v_overall_major[iVertex] = overall_v;
	}
	p_info_dest_major[iVertex] = info;

}

// Run vertex first, then average v_overall to tris, then run this after.
__global__ void kernelAdvectPositionsTris(
	f64 h_use,
	structural * __restrict__ p_info_src,
	structural * __restrict__ p_info_dest,
	f64_vec2 * __restrict__ p_v_overall_minor
	// what else is copied?
	// something we can easily copy over
	// with cudaMemcpy, even ahead of steps?
	// Is there a reason we cannot put into the above routine
	// with a split for "this is a vertex->just use its overall v"
)
{
	long const index = threadIdx.x + blockIdx.x * blockDim.x;
	structural info = p_info_src[index];
	f64_vec2 overall_v = p_v_overall_minor[index];
	f64_vec2 oldpos = info.pos;
	info.pos += h_use*overall_v;

	p_info_dest[index] = info;
}


__global__ void kernelCalculateNu_eHeartNu_iHeart_nu_nn_visc(
	structural * __restrict__ p_info_major,
	nvals * __restrict__ p_n,
	T3 * __restrict__ p_T,
	species3 * __restrict__ p_nu
) {
	// Save nu_iHeart, nu_eHeart, nu_nn_visc.

	species3 nu;
	f64 TeV, sigma_MT, sigma_visc, sqrt_T, nu_en_visc;
	T3 T;
	f64 nu_in_visc, nu_ni_visc, nu_ii;
	nvals our_n;
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x; // iVertex OF VERTEX
	structural info = p_info_major[iVertex];
	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {

		// We have not ruled out calculating traffic into outermost vertex cell - so this needs nu calculated in it.
		// (Does it actually try and receive traffic?)

		our_n = p_n[iVertex]; // never used again once we have kappa
		T = p_T[iVertex];

		TeV = T.Te * one_over_kB;
		Estimate_Ion_Neutral_Cross_sections_d(TeV, &sigma_MT, &sigma_visc);
		sqrt_T = sqrt(T.Te);
		nu_en_visc = our_n.n_n * sigma_visc * sqrt_T * over_sqrt_m_e;
		f64 nu_eiBar = nu_eiBarconst * kB_to_3halves * our_n.n *
			Get_lnLambda_d(our_n.n, T.Te) / (T.Te*sqrt_T);
		//nu_eHeart:
		nu.e = nu_en_visc + 1.87*nu_eiBar;
		
		if (TEST) printf("%d nu_en %1.9E nu_eiBar %1.9E nu_eHeart %1.9E\n",
			VERTCHOSEN, nu_en_visc, nu_eiBar, nu.e);

		TeV = T.Ti*one_over_kB;
		Estimate_Ion_Neutral_Cross_sections_d(TeV, &sigma_MT, &sigma_visc); // could easily save one call
		sqrt_T = sqrt(T.Ti); // again not that hard to save one call
		nu_in_visc = our_n.n_n * sigma_visc * sqrt(T.Ti / m_ion + T.Tn / m_n);
		nu_ni_visc = nu_in_visc * (our_n.n / our_n.n_n);
		//nu_nn_visc:
		nu.n = our_n.n_n * Estimate_Neutral_Neutral_Viscosity_Cross_section_d(T.Tn * one_over_kB)
			* sqrt(T.Tn / m_n);
		
		nu.n = 0.74*nu_ni_visc + 0.4*nu.n; // Rate to use in thermal conductivity.

		nu_ii = our_n.n*kB_to_3halves*Get_lnLambda_ion_d(our_n.n, T.Ti) *Nu_ii_Factor /
			(sqrt_T*T.Ti);
		// nu_iHeart:
		nu.i = 0.75*nu_in_visc + 0.8*nu_ii - 0.25*(nu_in_visc*nu_ni_visc) / (3.0*nu_ni_visc + nu.n);

		if ((TEST) ) {
			printf("@@@\nGPU %d: nu.i %1.12E |  %1.12E %1.12E %1.12E %1.12E n %1.10E n_n %1.10E Ti %1.10E\n"
				"sigma_visc %1.12E Ti %1.10E Tn %1.10E sqrt %1.10E\n", 
				
				iVertex, nu.i, nu_in_visc, nu_ii, nu_ni_visc, nu.n,
				our_n.n, our_n.n_n, T.Ti,
				sigma_visc, T.Ti, T.Tn, sqrt(T.Ti / m_i + T.Tn / m_n));
			printf("@@@\nGPU %d: nu.e %1.14E | nu_eiBar %1.14E our_n %1.14E lambda %1.14E over T^3/2 %1.14E nu_en_visc %1.14E\n",
				iVertex, nu.e, nu_eiBar, our_n.n, Get_lnLambda_d(our_n.n, T.Te), 1.0 / (T.Te*sqrt_T), nu_en_visc);
		}

		//  shared_n_over_nu[threadIdx.x].e = our_n.n / nu.e;
		//	shared_n_over_nu[threadIdx.x].i = our_n.n / nu.i;
		//	shared_n_over_nu[threadIdx.x].n = our_n.n_n / nu.n;
	}
	else {
		memset(&nu, 0, sizeof(species3));
	}

	p_nu[iVertex] = nu;
}

__global__ void kernelCalculate_kappa_nu(
	structural * __restrict__ p_info_minor,
	nvals * __restrict__ p_n_minor,
	T3 * __restrict__ p_T_minor,

	f64 * __restrict__ p_kappa_n,
	f64 * __restrict__ p_kappa_i,
	f64 * __restrict__ p_kappa_e,
	f64 * __restrict__ p_nu_i,
	f64 * __restrict__ p_nu_e
)
{
	long const iMinor = threadIdx.x + blockIdx.x * blockDim.x;
	f64 TeV, sigma_MT, sigma_visc, sqrt_T, nu_en_visc;
	T3 T;
	f64 nu_in_visc, nu_ni_visc, nu_ii;
	nvals our_n;
	species3 nu;

	structural info = p_info_minor[iMinor];
	if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS))
	{
		our_n = p_n_minor[iMinor];
		T = p_T_minor[iMinor];
		
		TeV = T.Ti*one_over_kB;
		Estimate_Ion_Neutral_Cross_sections_d(TeV, &sigma_MT, &sigma_visc); // could easily save one call
		sqrt_T = sqrt(T.Ti); // again not that hard to save one call
		nu_in_visc = our_n.n_n * sigma_visc * sqrt(T.Ti / m_ion + T.Tn / m_n);
		nu_ni_visc = nu_in_visc * (our_n.n / our_n.n_n);
		//nu_nn_visc:
		nu.n = our_n.n_n * Estimate_Neutral_Neutral_Viscosity_Cross_section_d(T.Tn * one_over_kB)
			* sqrt(T.Tn / m_n);

		nu.n = 0.74*nu_ni_visc + 0.4*nu.n; // Rate to use in thermal conductivity.
				
		nu_ii = our_n.n*kB_to_3halves*Get_lnLambda_ion_d(our_n.n, T.Ti) *Nu_ii_Factor /
			(sqrt_T*T.Ti);
		// nu_iHeart:
		nu.i = 0.75*nu_in_visc + 0.8*nu_ii - 0.25*(nu_in_visc*nu_ni_visc) / (3.0*nu_ni_visc + nu.n);
		//  ita uses nu_ion =  0.3*nu_ii + 0.4*nu_in_visc + 0.000273*nu_eiBar;
		// which again is only about half as much. BUT WE KNOW HE WORKS in DOUBLE Braginskii ??
		// says Vranjes -- so check that the rest of the formula does not compensate.

		// Would like consistent approach.
		// 1. We approached the heat flux directly per Golant. Where did it say what nu to use and how to add up?
		// Can we follow our own logic and then compare with what Zhdanov says?
		// 2. What about the limit of full ionization? Zero ionization? Does Zhdanov attain sensible limit?
		// Our version made sense.

		TeV = T.Te * one_over_kB;
		Estimate_Ion_Neutral_Cross_sections_d(TeV, &sigma_MT, &sigma_visc);
		sqrt_T = sqrt(T.Te);
		nu_en_visc = our_n.n_n * sigma_visc * sqrt_T * over_sqrt_m_e;
		f64 nu_eiBar = nu_eiBarconst * kB_to_3halves * our_n.n *
			Get_lnLambda_d(our_n.n, T.Te) / (T.Te*sqrt_T);
		//nu_eHeart:
		nu.e = nu_en_visc + 1.87*nu_eiBar;

		// Comparison with Zhdanov's denominator for ita looks like this one overestimated
		// by a factor of something like 1.6?
		
		f64 kappa_n = NEUTRAL_KAPPA_FACTOR * our_n.n_n * T.Tn / (m_n * nu.n) ;
		f64 kappa_i = (20.0 / 9.0) * our_n.n*T.Ti / (m_i * nu.i);
		f64 kappa_e = 2.5*our_n.n*T.Te / (m_e * nu.e);
		
		if ((TESTTRI)) printf("kappa_e %1.9E our_n.n %1.9E Te %1.9E nu %1.9E\n",
			kappa_e, our_n.n, T.Te, nu.e);

		if (kappa_i != kappa_i) printf("Tri %d kappa_i = NaN T %1.9E %1.9E %1.9E n %1.9E %1.9E \n", iMinor,
			T.Tn, T.Ti, T.Te, our_n.n_n, our_n.n);
		p_kappa_n[iMinor] = kappa_n;
		p_kappa_i[iMinor] = kappa_i;
		p_kappa_e[iMinor] = kappa_e;
		p_nu_i[iMinor] = nu.i;
		p_nu_e[iMinor] = nu.e;
	}
}


__global__ void kernelCreateTfromNTbydividing_bysqrtDN(
	f64 * __restrict__ p_T_n,
	f64 * __restrict__ p_T_i,
	f64 * __restrict__ p_T_e,
	f64 * __restrict__ p_sqrtDNn_Tn,
	f64 * __restrict__ p_sqrtDN_Ti,
	f64 * __restrict__ p_sqrtDN_Te,
	f64 * __restrict__ p_AreaMajor,
	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p_sqrtDinv_n, f64 * __restrict__ p_sqrtDinv_i,f64 * __restrict__ p_sqrtDinv_e
)
{
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x;

	nvals n = p_n_major[iVertex];
	f64 AreaMajor = p_AreaMajor[iVertex];
	f64 sqrtDNnTn = p_sqrtDNn_Tn[iVertex];
	f64 sqrtDNTi = p_sqrtDN_Ti[iVertex];
	f64 sqrtDNTe = p_sqrtDN_Te[iVertex];
	f64 Tn, Ti, Te;
	if (n.n_n*AreaMajor == 0.0) {
		Tn = 0.0;
	} else {
		Tn = sqrtDNnTn *p_sqrtDinv_n[iVertex] / sqrt(AreaMajor*n.n_n);
	}
	p_T_n[iVertex] = Tn;

	if (Tn != Tn) printf("iVertex %d Tn %1.10E area %1.9E \n",
		iVertex, Tn, AreaMajor);

	if (n.n*AreaMajor == 0.0) {
		Ti = 0.0;
		Te = 0.0;
	} else {
		Ti = sqrtDNTi *p_sqrtDinv_i[iVertex] / sqrt(AreaMajor*n.n);
		Te = sqrtDNTe *p_sqrtDinv_e[iVertex] / sqrt(AreaMajor*n.n);
	}
	p_T_i[iVertex] = Ti;
	p_T_e[iVertex] = Te;
}

__global__ void kernelCreateTfromNTbydividing(
	f64 * __restrict__ p_T_n,
	f64 * __restrict__ p_T_i,
	f64 * __restrict__ p_T_e,
	f64 * __restrict__ p_Nn_Tn,
	f64 * __restrict__ p_N_Ti,
	f64 * __restrict__ p_N_Te,
	f64 * __restrict__ p_AreaMajor,
	nvals * __restrict__ p_n_major)
{
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x; 

	nvals n = p_n_major[iVertex];
	f64 AreaMajor = p_AreaMajor[iVertex];
	f64 NnTn = p_Nn_Tn[iVertex];
	f64 NTi = p_N_Ti[iVertex];
	f64 NTe = p_N_Te[iVertex];
	f64 Tn, Ti, Te;
	if (n.n_n*AreaMajor == 0.0) {
		Tn = 0.0;
	} else {
		Tn = NnTn / sqrt(AreaMajor*n.n_n);
	}
	p_T_n[iVertex] = Tn;
	
	if (Tn != Tn) printf("iVertex %d Tn %1.10E area %1.9E \n",
		iVertex, Tn, AreaMajor);

	if (n.n*AreaMajor == 0.0) {
		Ti = 0.0;
		Te = 0.0;
	} else {
		Ti = NTi / sqrt(AreaMajor*n.n);
		Te = NTe / sqrt(AreaMajor*n.n);
	}
	p_T_i[iVertex] = Ti;
	p_T_e[iVertex] = Te;
}


__global__ void kernelTileMaxMajor(
	f64 * __restrict__ p_z,
	f64 * __restrict__ p_max
) 
{
	__shared__ f64 shared_z[threadsPerTileMajorClever];

	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;
	shared_z[threadIdx.x] = fabs(p_z[iVertex]);	
	
	__syncthreads();

//	if ((blockIdx.x == 0)) printf("iVertex %d threadIdx %d z %1.9E \n",
//		iVertex, threadIdx.x, shared_z[threadIdx.x]);

	int s = blockDim.x;
	int k = s / 2;
	while (s != 1) {
		if (threadIdx.x < k)
		{
			shared_z[threadIdx.x] = max(shared_z[threadIdx.x], shared_z[threadIdx.x + k]);
		//	if (blockIdx.x == 0) printf("s %d thread %d max %1.9E looked at %d : %1.9E\n", s, threadIdx.x, shared_z[threadIdx.x],
		//		threadIdx.x + k, shared_z[threadIdx.x + k]);
		};
		__syncthreads();
		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			shared_z[threadIdx.x] = max(shared_z[threadIdx.x], shared_z[threadIdx.x + s - 1]);
	//		if (blockIdx.x == 0) printf("EXTRA CODE: s %d thread %d max %1.9E \n", s, threadIdx.x, shared_z[threadIdx.x]);
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};
	if (threadIdx.x == 0)
	{
		p_max[blockIdx.x] = shared_z[0];

		if (blockIdx.x == 0) printf("block 0 max %1.10E \n", p_max[blockIdx.x]);
	}
}





__global__ void kernelEstimateCurrent(
	structural * __restrict__ p_info_minor,
	nvals * __restrict__ p_n_minor,
	v4 * __restrict__ p_vie,
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ p_Iz
) {
	__shared__ f64 Izcell[numTilesMinor];

	// This is what we need rather than making PopOhmsLaw even more bloated.
	// Maybe we can look towards gradually moving some more content into this 1st routine.
	long iMinor = blockIdx.x*blockDim.x + threadIdx.x;
	Izcell[threadIdx.x] = 0.0;

 	structural info = p_info_minor[iMinor];
	if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE)
		|| (info.flag == OUTERMOST) || (info.flag == CROSSING_INS))
	{
		nvals n_use = p_n_minor[iMinor];
		v4 vie = p_vie[iMinor];
		f64 AreaMinor = p_AreaMinor[iMinor];

		Izcell[threadIdx.x] = q*n_use.n*(vie.viz - vie.vez)*AreaMinor;
	}
	
	__syncthreads();
	
	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			Izcell[threadIdx.x] += Izcell[threadIdx.x + k];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			Izcell[threadIdx.x] += Izcell[threadIdx.x + s - 1];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_Iz[blockIdx.x] = Izcell[0];
	}
}


__global__ void kernelAverage(
	f64 * __restrict__ p_update,
	f64 * __restrict__ p_input2)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	p_update[index] = 0.5*p_update[index] + 0.5*p_input2[index];
}

__global__ void kernelAdvanceAzEuler(
	f64 const h_use,
	AAdot * __restrict__ p_AAdot_use,
	AAdot * __restrict__ p_AAdot_dest,
	f64 * __restrict__ p_ROCAzduetoAdvection)
{

	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	AAdot AAdot_use = p_AAdot_use[index];
	f64 ROCAz = p_ROCAzduetoAdvection[index];
	AAdot_use.Az += h_use*(AAdot_use.Azdot + ROCAz);
	p_AAdot_dest[index] = AAdot_use;

}

__global__ void kernelAdvanceAzBwdEuler(
	f64 const h_use,
	AAdot * __restrict__ p_AAdot_use,
	AAdot * __restrict__ p_AAdot_dest,
	f64 * __restrict__ p_ROCAzduetoAdvection)
{

	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	AAdot AAdot_use = p_AAdot_use[index];
	AAdot AAdot_dest = p_AAdot_dest[index];
	f64 ROCAz = p_ROCAzduetoAdvection[index];
	AAdot_use.Az += h_use*(AAdot_dest.Azdot + ROCAz);
	AAdot_use.Azdot = AAdot_dest.Azdot;
	p_AAdot_dest[index] = AAdot_use;
	// So we did not predict how Az would change due to ROCAz -- it's neglected when we solve for A(Adot(LapAz))
}

__global__ void kernelUpdateAz(
	f64 const h_use,
	AAdot * __restrict__ p_AAdot_use,
	f64 * __restrict__ p_ROCAzduetoAdvection,
	f64 * __restrict__ p_Az
) {
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	AAdot AAdot_use = p_AAdot_use[index];
	f64 ROCAz = p_ROCAzduetoAdvection[index];
	p_Az[index] += h_use*(AAdot_use.Azdot + ROCAz);
} // perhaps exists a mathematical way to roll up ROC due to advection into our Azdot.

__global__ void kernelPopulateArrayAz(
	f64 const h_use,
	AAdot * __restrict__ p_AAdot_use,
	f64 * __restrict__ p_ROCAzduetoAdvection,
	f64 * __restrict__ p_Az
) {
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	AAdot AAdot_use = p_AAdot_use[index];
	f64 ROCAz = p_ROCAzduetoAdvection[index];
	p_Az[index] = AAdot_use.Az + h_use*(AAdot_use.Azdot + ROCAz);
} // perhaps exists a mathematical way to roll up ROC due to advection into our Azdot.

__global__ void kernelPushAzInto_dest(
	AAdot * __restrict__ p_AAdot,
	f64 * __restrict__ p_Az
) {
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	p_AAdot[index].Az = p_Az[index];
} 
__global__ void kernelPullAzFromSyst(
	AAdot * __restrict__ p_AAdot,
	f64 * __restrict__ p_Az
) {
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	p_Az[index] = p_AAdot[index].Az;
}

__global__ void kernelAddtoT(
	T3 * __restrict__ p_T_dest,
	f64 beta_nJ, f64 beta_nR, 
	f64 beta_iJ, f64 beta_iR, 
	f64 beta_eJ, f64 beta_eR,
	f64 * __restrict__ p_Jacobi_n,
	f64 * __restrict__ p_Jacobi_i,
	f64 * __restrict__ p_Jacobi_e,
	f64 * __restrict__ p_epsilon_n,
	f64 * __restrict__ p_epsilon_i,
	f64 * __restrict__ p_epsilon_e
) {
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	T3 T = p_T_dest[index];
	T.Tn += beta_nJ*p_Jacobi_n[index] + beta_nR*p_epsilon_n[index];
	T.Ti += beta_iJ*p_Jacobi_i[index] + beta_iR*p_epsilon_i[index];
	T.Te += beta_eJ*p_Jacobi_e[index] + beta_eR*p_epsilon_e[index];

	// Testing to see if - makes it get closer instead of further away
	// It does indeed - but we can't explain why.
	
	p_T_dest[index] = T;
}

__global__ void kernelAddtoT_lc(
	f64 * __restrict__ p__T,
	f64 * __restrict__ p_addition
)
{
	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;
	f64 T = p__T[iVertex];
	for (int i = 0; i < REGRESSORS; i++)
		T += beta_n_c[i] * p_addition[i*NUMVERTICES+iVertex];
	p__T[iVertex] = T;
}

__global__ void kernelAddtoT_volleys(
	T3 * __restrict__ p_T_dest,
	//f64 beta_n[8],
	//f64 beta_i[8],
	//f64 beta_e[8], // copy arrays to constant memory ahead of time
	char * __restrict__ p_iVolley,
	f64 * __restrict__ p_Jacobi_n,
	f64 * __restrict__ p_Jacobi_i,
	f64 * __restrict__ p_Jacobi_e,
	f64 * __restrict__ p_epsilon_n,
	f64 * __restrict__ p_epsilon_i,
	f64 * __restrict__ p_epsilon_e
) {
	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;
	T3 T = p_T_dest[iVertex];
	char iVolley = p_iVolley[iVertex];
	switch (iVolley) {
	case 0:
		T.Tn += beta_n_c[0] * p_Jacobi_n[iVertex] + beta_n_c[4] * p_epsilon_n[iVertex];
		T.Ti += beta_i_c[0] * p_Jacobi_i[iVertex] + beta_i_c[4] * p_epsilon_i[iVertex];
		T.Te += beta_e_c[0] * p_Jacobi_e[iVertex] + beta_e_c[4] * p_epsilon_e[iVertex];
		break;
	case 1:
		T.Tn += beta_n_c[1] * p_Jacobi_n[iVertex] + beta_n_c[5] * p_epsilon_n[iVertex];
		T.Ti += beta_i_c[1] * p_Jacobi_i[iVertex] + beta_i_c[5] * p_epsilon_i[iVertex];
		T.Te += beta_e_c[1] * p_Jacobi_e[iVertex] + beta_e_c[5] * p_epsilon_e[iVertex];
		break;
	case 2:
		T.Tn += beta_n_c[2] * p_Jacobi_n[iVertex] + beta_n_c[6] * p_epsilon_n[iVertex];
		T.Ti += beta_i_c[2] * p_Jacobi_i[iVertex] + beta_i_c[6] * p_epsilon_i[iVertex];
		T.Te += beta_e_c[2] * p_Jacobi_e[iVertex] + beta_e_c[6] * p_epsilon_e[iVertex];
		break;
	case 3:
		T.Tn += beta_n_c[3] * p_Jacobi_n[iVertex] + beta_n_c[7] * p_epsilon_n[iVertex];
		T.Ti += beta_i_c[3] * p_Jacobi_i[iVertex] + beta_i_c[7] * p_epsilon_i[iVertex];
		T.Te += beta_e_c[3] * p_Jacobi_e[iVertex] + beta_e_c[7] * p_epsilon_e[iVertex];
		break;
	}
	p_T_dest[iVertex] = T;
}

__global__ void kernelAdd(
	f64 * __restrict__ p_updated,
	f64 beta,
	f64 * __restrict__ p_added
)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	p_updated[index] += beta * p_added[index];
}

__global__ void kernelAdd_to_v(
	v4 * __restrict__ p_vie,
	f64 const beta_i, f64 const beta_e,
	f64_vec3 * __restrict__ p_vJacobi_ion,
	f64_vec3 * __restrict__ p_vJacobi_elec
)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	v4 vie;
	memcpy(&vie, &(p_vie[index]), sizeof(v4));
	f64_vec3 vJ_ion = p_vJacobi_ion[index];
	f64_vec3 vJ_elec = p_vJacobi_elec[index];
	vie.vxy.x += beta_i*vJ_ion.x + beta_e*vJ_elec.x;
	vie.vxy.y += beta_i*vJ_ion.y + beta_e*vJ_elec.y;
	vie.viz += beta_i*vJ_ion.z;
	vie.vez += beta_e*vJ_elec.z;
	memcpy(&(p_vie[index]), &vie, sizeof(v4));
}


// Try resetting frills here and ignoring in calculation:
__global__ void kernelResetFrillsAz(
	structural * __restrict__ p_info,
	LONG3 * __restrict__ trineighbourindex,
	f64 * __restrict__ p_Az)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info[index];
	
	if ((info.flag == INNER_FRILL) || (info.flag == OUTER_FRILL))
	{
		LONG3 izNeigh = trineighbourindex[index];
		if (info.flag == INNER_FRILL)
		{
			p_Az[index] = p_Az[izNeigh.i1]; 
		} else {			
			f64 r = p_info[izNeigh.i1].pos.modulus();
			p_Az[index] = (r/ FRILL_CENTROID_OUTER_RADIUS_d)*p_Az[izNeigh.i1]; // should be something like 0.99*p_Az[izNeigh.i1]
			// Better if we store a constant called Outer_Frill_Factor to save a load and a division.
		};
	};	
}


__global__ void kernelReset_v_in_outer_frill_and_outermost
(
	structural * __restrict__ p_info,
	v4 * __restrict__ p_vie,
	f64_vec3 * __restrict__ p_v_n,
	LONG3 * __restrict__ trineighbourindex,
	long * __restrict__ p_izNeigh_vert
	) {
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info[index];
	if ((info.flag == OUTER_FRILL))
	{
		LONG3 izNeigh = trineighbourindex[index];
		p_vie[index] = p_vie[izNeigh.i1];
		p_v_n[index] = p_v_n[izNeigh.i1];
	}
	if ((info.flag == OUTERMOST))
	{
		long izNeigh[MAXNEIGH_d];
		long iVertex = index - BEGINNING_OF_CENTRAL;
		memcpy(izNeigh, p_izNeigh_vert + MAXNEIGH_d*iVertex, sizeof(long)*MAXNEIGH_d);
		v4 result, temp4;
		f64_vec3 v_n, temp3;
		memset(&result, 0, sizeof(v4));
		memset(&v_n, 0, sizeof(f64_vec3));
		long iDomain = 0;
		for (short i = 0; i < 4; i++)
		{
			structural infoneigh = p_info[izNeigh[i] + BEGINNING_OF_CENTRAL];
			if (infoneigh.flag == DOMAIN_VERTEX)
			{
				temp4 = p_vie[izNeigh[i] + BEGINNING_OF_CENTRAL];
				temp3 = p_v_n[izNeigh[i] + BEGINNING_OF_CENTRAL];
				iDomain++;
				result.vxy += temp4.vxy;
				result.vez += temp4.vez;
				result.viz += temp4.viz;
				v_n += temp3;
			}
		}
		if (iDomain > 0) {
			f64 fac = 1.0 / (f64)iDomain;
			result.vxy *= fac;
			result.vez *= fac;
			result.viz *= fac;
			v_n *= fac;
		}
		p_vie[index] = result;
		p_v_n[index] = v_n;
	}
}

// Try resetting frills here and ignoring in calculation:
__global__ void kernelResetFrillsAz_II(
	structural * __restrict__ p_info,
	LONG3 * __restrict__ trineighbourindex,
	AAdot * __restrict__ p_Az)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info[index];
	if ((info.flag == INNER_FRILL) || (info.flag == OUTER_FRILL))
	{
		LONG3 izNeigh = trineighbourindex[index];
		p_Az[index].Az = p_Az[izNeigh.i1].Az;
	}
}


__global__ void kernelCreateAzbymultiplying(
	f64 * __restrict__ p_Az,
	f64 * __restrict__ p_scaledAz,
	f64 const h_use,
	f64 * __restrict__ p_gamma
)
{
	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	f64 gamma = p_gamma[iMinor];

	p_Az[iMinor] = p_scaledAz[iMinor] * sqrt(h_use*p_gamma[iMinor]);

}

__global__ void kernelAccumulateMatrix(
	structural * __restrict__ p_info,
	f64 const h_use,
	f64 * __restrict__ p_epsilon,
	f64 * __restrict__ p_regressor1,
	f64 * __restrict__ p_regressor2,
	f64 * __restrict__ p_regressor3,
	f64 * __restrict__ p_LapReg1,
	f64 * __restrict__ p_LapReg2,
	f64 * __restrict__ p_LapReg3,
	f64 * __restrict__ p_gamma,
	f64 * __restrict__ p_deps_matrix,
	f64_vec3 * __restrict__ p_eps_against_deps)

{
	__shared__ f64 sum_mat[threadsPerTileMinor][6];
	__shared__ f64 sum_eps_deps[threadsPerTileMinor][3];

	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	f64 d_eps_by_d_beta1, d_eps_by_d_beta2, d_eps_by_d_beta3;
	structural info = p_info[index];
	if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL))
	{
		d_eps_by_d_beta1 = 0.0;
		d_eps_by_d_beta2 = 0.0;
		d_eps_by_d_beta3 = 0.0; // eps here actually is 0.
	}
	else {
		d_eps_by_d_beta1 = (p_regressor1[index] - h_use * p_gamma[index] * p_LapReg1[index]);
		d_eps_by_d_beta2 = (p_regressor2[index] - h_use * p_gamma[index] * p_LapReg2[index]);
		d_eps_by_d_beta3 = (p_regressor3[index] - h_use * p_gamma[index] * p_LapReg3[index]);
	};
	sum_mat[threadIdx.x][0] = d_eps_by_d_beta1*d_eps_by_d_beta1;
	sum_mat[threadIdx.x][1] = d_eps_by_d_beta1*d_eps_by_d_beta2;
	sum_mat[threadIdx.x][2] = d_eps_by_d_beta1*d_eps_by_d_beta3;
	sum_mat[threadIdx.x][3] = d_eps_by_d_beta2*d_eps_by_d_beta2;
	sum_mat[threadIdx.x][4] = d_eps_by_d_beta2*d_eps_by_d_beta3;
	sum_mat[threadIdx.x][5] = d_eps_by_d_beta3*d_eps_by_d_beta3;
	f64 eps = p_epsilon[index];
	sum_eps_deps[threadIdx.x][0] = eps*d_eps_by_d_beta1;
	sum_eps_deps[threadIdx.x][1] = eps*d_eps_by_d_beta2;
	sum_eps_deps[threadIdx.x][2] = eps*d_eps_by_d_beta3;


	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
#pragma unroll
			for (int y = 0; y < 6; y++)
				sum_mat[threadIdx.x][y] += sum_mat[threadIdx.x + k][y];
			sum_eps_deps[threadIdx.x][0] += sum_eps_deps[threadIdx.x + k][0];
			sum_eps_deps[threadIdx.x][1] += sum_eps_deps[threadIdx.x + k][1];
			sum_eps_deps[threadIdx.x][2] += sum_eps_deps[threadIdx.x + k][2];

		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			for (int y = 0; y < 6; y++)
				sum_mat[threadIdx.x][y] += sum_mat[threadIdx.x + s - 1][y];
			sum_eps_deps[threadIdx.x][0] += sum_eps_deps[threadIdx.x + s - 1][0];
			sum_eps_deps[threadIdx.x][1] += sum_eps_deps[threadIdx.x + s - 1][1];
			sum_eps_deps[threadIdx.x][2] += sum_eps_deps[threadIdx.x + s - 1][2];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		memcpy(&(p_deps_matrix[6 * blockIdx.x]), sum_mat[0], sizeof(f64) * 6);
		f64_vec3 tempvec3;
		tempvec3.x = sum_eps_deps[0][0];
		tempvec3.y = sum_eps_deps[0][1];
		tempvec3.z = sum_eps_deps[0][2];

		memcpy(&p_eps_against_deps[blockIdx.x], &tempvec3, sizeof(f64_vec3));
	}
}


__global__ void VectorCompareMax(
	f64 * __restrict__ p_comp1,
	f64 * __restrict__ p_comp2,
	long * __restrict__ p_iWhich,
	f64 * __restrict__ p_max
)
{
	__shared__ f64 diff[threadsPerTileMajorClever];
	__shared__ long longarray[threadsPerTileMajorClever];

	long const iVertex = threadIdx.x + blockDim.x*blockIdx.x;
	diff[threadIdx.x] = fabs(p_comp1[iVertex] - p_comp2[iVertex]);
	longarray[threadIdx.x] = iVertex;
	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
#pragma unroll
			
			if (diff[threadIdx.x] > diff[threadIdx.x + k])
			{
				// do nothing	
			} else {
				diff[threadIdx.x] = diff[threadIdx.x + k];
				longarray[threadIdx.x] = longarray[threadIdx.x + k];
			};
			
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			
			if (diff[threadIdx.x] > diff[threadIdx.x + s-1])
			{
				// do nothing	
			} else {
				diff[threadIdx.x] = diff[threadIdx.x + s-1];
				longarray[threadIdx.x] = longarray[threadIdx.x + s-1];
			};
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_iWhich[blockIdx.x] = longarray[0];
		p_max[blockIdx.x] = diff[0];
	}
}


//
//__global__ void kernelAccumulateMatrix_debug(
//	structural * __restrict__ p_info,
//	f64 const h_use,
//	f64 * __restrict__ p_epsilon,
//	f64 * __restrict__ p_regressor1,
//	f64 * __restrict__ p_regressor2,
//	f64 * __restrict__ p_regressor3,
//	f64 * __restrict__ p_LapReg1,
//	f64 * __restrict__ p_LapReg2,
//	f64 * __restrict__ p_LapReg3,
//	f64 * __restrict__ p_gamma,
//	f64 * __restrict__ p_deps_matrix,
//	f64_vec3 * __restrict__ p_eps_against_deps,
//	
//	f64 * __restrict__ p_deps_1,
//	f64 * __restrict__ p_deps_2,
//	f64 * __restrict__ p_deps_3
//
//	)
//	
//{
//	__shared__ f64 sum_mat[threadsPerTileMinor][6];
//	__shared__ f64 sum_eps_deps[threadsPerTileMinor][3];
//	
//	long const index = blockDim.x*blockIdx.x + threadIdx.x;
//	f64 d_eps_by_d_beta1, d_eps_by_d_beta2, d_eps_by_d_beta3;
//	structural info = p_info[index];
//	if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL))
//	{
//		d_eps_by_d_beta1 = 0.0; 
//		d_eps_by_d_beta2 = 0.0;
//		d_eps_by_d_beta3 = 0.0; // eps here actually is 0.
//	} else {
//		d_eps_by_d_beta1 = (p_regressor1[index] - h_use * p_gamma[index] * p_LapReg1[index]);
//		d_eps_by_d_beta2 = (p_regressor2[index] - h_use * p_gamma[index] * p_LapReg2[index]);
//		d_eps_by_d_beta3 = (p_regressor3[index] - h_use * p_gamma[index] * p_LapReg3[index]);
//	};
//	sum_mat[threadIdx.x][0] = d_eps_by_d_beta1*d_eps_by_d_beta1;
//	sum_mat[threadIdx.x][1] = d_eps_by_d_beta1*d_eps_by_d_beta2;
//	sum_mat[threadIdx.x][2] = d_eps_by_d_beta1*d_eps_by_d_beta3;
//	sum_mat[threadIdx.x][3] = d_eps_by_d_beta2*d_eps_by_d_beta2;
//	sum_mat[threadIdx.x][4] = d_eps_by_d_beta2*d_eps_by_d_beta3;
//	sum_mat[threadIdx.x][5] = d_eps_by_d_beta3*d_eps_by_d_beta3;
//	f64 eps = p_epsilon[index];
//	sum_eps_deps[threadIdx.x][0] = eps*d_eps_by_d_beta1;
//	sum_eps_deps[threadIdx.x][1] = eps*d_eps_by_d_beta2;
//	sum_eps_deps[threadIdx.x][2] = eps*d_eps_by_d_beta3;
//
//	p_deps_1[index] = d_eps_by_d_beta1;
//	p_deps_2[index] = d_eps_by_d_beta2;
//	p_deps_3[index] = d_eps_by_d_beta3;
//
//	__syncthreads();
//
//	int s = blockDim.x;
//	int k = s / 2;
//
//	while (s != 1) {
//		if (threadIdx.x < k)
//		{
//#pragma unroll
//			for (int y = 0; y < 6; y++)
//				sum_mat[threadIdx.x][y] += sum_mat[threadIdx.x + k][y];
//			sum_eps_deps[threadIdx.x][0] += sum_eps_deps[threadIdx.x + k][0];
//			sum_eps_deps[threadIdx.x][1] += sum_eps_deps[threadIdx.x + k][1];
//			sum_eps_deps[threadIdx.x][2] += sum_eps_deps[threadIdx.x + k][2];
//			
//		};
//		__syncthreads();
//
//		// Modify for case blockdim not 2^n:
//		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
//			for (int y = 0; y < 6; y++)
//				sum_mat[threadIdx.x][y] += sum_mat[threadIdx.x + s - 1][y];
//			sum_eps_deps[threadIdx.x][0] += sum_eps_deps[threadIdx.x + s - 1][0];
//			sum_eps_deps[threadIdx.x][1] += sum_eps_deps[threadIdx.x + s - 1][1];
//			sum_eps_deps[threadIdx.x][2] += sum_eps_deps[threadIdx.x + s - 1][2];			
//		};
//		// In case k == 81, add [39] += [80]
//		// Otherwise we only get to 39+40=79.
//		s = k;
//		k = s / 2;
//		__syncthreads();
//	};
//
//	if (threadIdx.x == 0)
//	{
//		memcpy(&(p_deps_matrix[6 * blockIdx.x]), sum_mat[0], sizeof(f64) * 6);
//		f64_vec3 tempvec3;
//		tempvec3.x = sum_eps_deps[0][0];
//		tempvec3.y = sum_eps_deps[0][1];
//		tempvec3.z = sum_eps_deps[0][2];
//		
//		memcpy(&p_eps_against_deps[blockIdx.x], &tempvec3, sizeof(f64_vec3));
//	}
//}

__global__ void kernelAddRegressors(
	f64 * __restrict__ p_AzNext,
	f64 const beta0, f64 const beta1, f64 const beta2,
	f64 * __restrict__ p_reg1,
	f64 * __restrict__ p_reg2,
	f64 * __restrict__ p_reg3)
{
	long const iMinor = blockIdx.x*blockDim.x + threadIdx.x;
	p_AzNext[iMinor] += beta0*p_reg1[iMinor] + beta1*p_reg2[iMinor] + beta2*p_reg3[iMinor];
}

__global__ void kernelAccumulateSummands(
	structural * __restrict__ p_info,
	f64 h_use,
	f64 * __restrict__ p_epsilon,
	f64 * __restrict__ p_Jacobi,
	f64 * __restrict__ p_LapJacobi,
	f64 * __restrict__ p_gamma,
	f64 * __restrict__ p_sum_eps_d,
	f64 * __restrict__ p_sum_d_d,
	f64 * __restrict__ p_sum_eps_eps)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;

	__shared__ f64 sumdata1[threadsPerTileMinor];
	__shared__ f64 sumdata2[threadsPerTileMinor];
	__shared__ f64 sumdata3[threadsPerTileMinor];

	f64 depsbydbeta;
	structural info = p_info[index];
	if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL))
	{
		depsbydbeta = 0.0; //  Lap_Jacobi[iMinor]; // try ignoring
	}
	else {
//#ifdef MIDPT_A_AND_ACTUALLY_MIDPT_A_NOT_JUST_EFFECT_ON_AZDOT
//		depsbydbeta = (p_Jacobi[index] - 0.5*h_use * p_gamma[index] * p_LapJacobi[index]);
//#else
		depsbydbeta = (p_Jacobi[index] - h_use * p_gamma[index] * p_LapJacobi[index]);
//#endif
	};
	f64 eps = p_epsilon[index];
	sumdata1[threadIdx.x] = depsbydbeta * eps;
	sumdata2[threadIdx.x] = depsbydbeta * depsbydbeta;
	sumdata3[threadIdx.x] = eps * eps;

	__syncthreads();
	
	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + k];
			sumdata2[threadIdx.x] += sumdata2[threadIdx.x + k];
			sumdata3[threadIdx.x] += sumdata3[threadIdx.x + k];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + s - 1];
			sumdata2[threadIdx.x] += sumdata2[threadIdx.x + s - 1];
			sumdata3[threadIdx.x] += sumdata3[threadIdx.x + s - 1];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};
	
	if (threadIdx.x == 0)
	{
		p_sum_eps_d[blockIdx.x] = sumdata1[0];
		p_sum_d_d[blockIdx.x] = sumdata2[0];
		p_sum_eps_eps[blockIdx.x] = sumdata3[0];
	}
}



__global__ void kernelAccumulateSumOfSquares(
	f64 * __restrict__ p_eps_n,
	f64 * __restrict__ p_eps_i,
	f64 * __restrict__ p_eps_e,
	f64 * __restrict__ p_SS_n,
	f64 * __restrict__ p_SS_i,
	f64 * __restrict__ p_SS_e)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;

	__shared__ f64 sumdata1[threadsPerTileMajorClever];
	__shared__ f64 sumdata2[threadsPerTileMajorClever];
	__shared__ f64 sumdata3[threadsPerTileMajorClever];

	f64 epsilon_n = p_eps_n[index];
	f64 epsilon_i = p_eps_i[index];
	f64 epsilon_e = p_eps_e[index];

	sumdata1[threadIdx.x] = epsilon_n*epsilon_n;
	sumdata2[threadIdx.x] = epsilon_i*epsilon_i;
	sumdata3[threadIdx.x] = epsilon_e*epsilon_e;
	
	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + k];
			sumdata2[threadIdx.x] += sumdata2[threadIdx.x + k];
			sumdata3[threadIdx.x] += sumdata3[threadIdx.x + k];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + s - 1];
			sumdata2[threadIdx.x] += sumdata2[threadIdx.x + s - 1];
			sumdata3[threadIdx.x] += sumdata3[threadIdx.x + s - 1];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_SS_n[blockIdx.x] = sumdata1[0];
		p_SS_i[blockIdx.x] = sumdata2[0];
		p_SS_e[blockIdx.x] = sumdata3[0];
	}
}


__global__ void kernelAccumulateSumOfSquares1(
	f64 * __restrict__ p_eps,
	f64 * __restrict__ p_SS)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;

	__shared__ f64 sumdata1[threadsPerTileMajorClever];

	f64 epsilon_n = p_eps[index];

	sumdata1[threadIdx.x] = epsilon_n*epsilon_n;

	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + k];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + s - 1];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_SS[blockIdx.x] = sumdata1[0];
	}
}

__global__ void kernelAccumulateDotProducts(
	f64 * __restrict__ p_x1, f64 * __restrict__ p_y1,
	f64 * __restrict__ p_x2, f64 * __restrict__ p_y2,
	f64 * __restrict__ p_x3, f64 * __restrict__ p_y3,
	f64 * __restrict__ p_dot1,
	f64 * __restrict__ p_dot2,
	f64 * __restrict__ p_dot3)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;

	__shared__ f64 sumdata1[threadsPerTileMajorClever];
	__shared__ f64 sumdata2[threadsPerTileMajorClever];
	__shared__ f64 sumdata3[threadsPerTileMajorClever];

	f64 x1 = p_x1[index];
	f64 x2 = p_x2[index];
	f64 x3 = p_x3[index];
	f64 y1 = p_y1[index];
	f64 y2 = p_y2[index];
	f64 y3 = p_y3[index];

	sumdata1[threadIdx.x] = x1*y1;
	sumdata2[threadIdx.x] = x2*y2;
	sumdata3[threadIdx.x] = x3*y3;

	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + k];
			sumdata2[threadIdx.x] += sumdata2[threadIdx.x + k];
			sumdata3[threadIdx.x] += sumdata3[threadIdx.x + k];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + s - 1];
			sumdata2[threadIdx.x] += sumdata2[threadIdx.x + s - 1];
			sumdata3[threadIdx.x] += sumdata3[threadIdx.x + s - 1];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_dot1[blockIdx.x] = sumdata1[0];
		p_dot2[blockIdx.x] = sumdata2[0];
		p_dot3[blockIdx.x] = sumdata3[0];
	}
}

__global__ void kernelAccumulateDotProduct(
	f64 * __restrict__ p_x1, f64 * __restrict__ p_y1,
	f64 * __restrict__ p_dot1)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;

	__shared__ f64 sumdata1[threadsPerTileMajorClever];
	
	f64 x1 = p_x1[index];
	f64 y1 = p_y1[index];
	
	sumdata1[threadIdx.x] = x1*y1;
	
	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + k];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + s - 1];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_dot1[blockIdx.x] = sumdata1[0];
	}
}

__global__ void VectorAddMultiple1(
	f64 * __restrict__ p_T1, f64 const alpha1, f64 * __restrict__ p_x1)
{
	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;
	p_T1[iVertex] += alpha1*p_x1[iVertex];
}

__global__ void VectorAddMultiple(
	f64 * __restrict__ p_T1, f64 const alpha1, f64 * __restrict__ p_x1,
	f64 * __restrict__ p_T2, f64 const alpha2, f64 * __restrict__ p_x2,
	f64 * __restrict__ p_T3, f64 const alpha3, f64 * __restrict__ p_x3)
{
	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;

	//if (iVertex == VERTCHOSEN) printf("%d Ti %1.10E ", iVertex, p_T2[iVertex]);
	
	p_T1[iVertex] += alpha1*p_x1[iVertex];
	p_T2[iVertex] += alpha2*p_x2[iVertex];
	p_T3[iVertex] += alpha3*p_x3[iVertex];

	//if (iVertex == VERTCHOSEN) printf("alpha2 %1.12E x2 %1.12E result %1.12E\n",
	//	alpha2, p_x2[iVertex], p_T2[iVertex]);
	
}

__global__ void VectorAddMultiple_masked(
	f64 * __restrict__ p_T1, f64 const alpha1, f64 * __restrict__ p_x1,
	f64 * __restrict__ p_T2, f64 const alpha2, f64 * __restrict__ p_x2,
	f64 * __restrict__ p_T3, f64 const alpha3, f64 * __restrict__ p_x3,
	bool * __restrict__ p_bMask,
	bool * __restrict__ p_bMaskblock,
	bool const bUseMask)
{
	if ((bUseMask) && (p_bMaskblock[blockIdx.x] == 0)) return;

	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;

	//if (iVertex == VERTCHOSEN) printf("%d Ti %1.10E ", iVertex, p_T2[iVertex]);
	if (bUseMask) {

		bool bMask[3];
		bMask[0] = p_bMask[iVertex];
		bMask[1] = p_bMask[iVertex + NUMVERTICES];
		bMask[2] = p_bMask[iVertex + NUMVERTICES*2];
		if (bMask[0]) p_T1[iVertex] += alpha1*p_x1[iVertex];
		if (bMask[1]) p_T2[iVertex] += alpha2*p_x2[iVertex];

		if (iVertex == VERTCHOSEN) printf("%d old T : %1.12E ", iVertex, p_T3[iVertex]);

		if (bMask[2]) p_T3[iVertex] += alpha3*p_x3[iVertex];

		if (iVertex == VERTCHOSEN) printf("alpha3 %1.12E x3 %1.12E new T %1.12E \n",
			alpha3, p_x3[iVertex], p_T3[iVertex]);

	} else {
		p_T1[iVertex] += alpha1*p_x1[iVertex];
		p_T2[iVertex] += alpha2*p_x2[iVertex];
		p_T3[iVertex] += alpha3*p_x3[iVertex];
	}
	//if (iVertex == VERTCHOSEN) printf("alpha2 %1.12E x2 %1.12E result %1.12E\n",
	//	alpha2, p_x2[iVertex], p_T2[iVertex]);

}
__global__ void kernelRegressorUpdate
(
	f64 * __restrict__ p_x_n,
	f64 * __restrict__ p_x_i,
	f64 * __restrict__ p_x_e,
	f64 * __restrict__ p_a_n, f64 * __restrict__ p_a_i, f64 * __restrict__ p_a_e,
	f64 const ratio1, f64 const ratio2, f64 const ratio3,
	bool * __restrict__ p_bMaskBlock,
	bool bUseMask
	)
{
	if ((bUseMask) && (p_bMaskBlock[blockIdx.x] == 0)) return;

	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;
	
	f64 xn = p_x_n[iVertex];
	p_x_n[iVertex] = p_a_n[iVertex] + ratio1*xn;
	f64 xi = p_x_i[iVertex];
	p_x_i[iVertex] = p_a_i[iVertex] + ratio2*xi;
	f64 xe = p_x_e[iVertex];
	p_x_e[iVertex] = p_a_e[iVertex] + ratio3*xe;
}

__global__ void kernelPackupT3(
	T3 * __restrict__ p_T,
	f64 * __restrict__ p_Tn, f64 * __restrict__ p_Ti, f64 * __restrict__ p_Te)
{
	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;
	T3 T;
	T.Tn = p_Tn[iVertex];
	T.Ti = p_Ti[iVertex];
	T.Te = p_Te[iVertex];
	p_T[iVertex] = T;
}

__global__ void kernelAccumulateSummands2(
	structural * __restrict__ p_info,
	
	f64 * __restrict__ p_epsilon,
	f64 * __restrict__ p_d_eps_by_dbeta,
	
	f64 * __restrict__ p_sum_eps_d,
	f64 * __restrict__ p_sum_d_d,
	f64 * __restrict__ p_sum_eps_eps)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;

	__shared__ f64 sumdata1[threadsPerTileMinor];
	__shared__ f64 sumdata2[threadsPerTileMinor];
	__shared__ f64 sumdata3[threadsPerTileMinor];

	f64 eps = p_epsilon[index];
	f64 depsbydbeta = p_d_eps_by_dbeta[index];

	structural info = p_info[index];
	if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL))
		depsbydbeta = 0.0; //  Lap_Jacobi[iMinor]; // try ignoring
	// could rearrange to not have to do that or load info.

	sumdata1[threadIdx.x] = depsbydbeta * eps;
	sumdata2[threadIdx.x] = depsbydbeta * depsbydbeta;
	sumdata3[threadIdx.x] = eps * eps;

	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + k];
			sumdata2[threadIdx.x] += sumdata2[threadIdx.x + k];
			sumdata3[threadIdx.x] += sumdata3[threadIdx.x + k];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			sumdata1[threadIdx.x] += sumdata1[threadIdx.x + s - 1];
			sumdata2[threadIdx.x] += sumdata2[threadIdx.x + s - 1];
			sumdata3[threadIdx.x] += sumdata3[threadIdx.x + s - 1];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_sum_eps_d[blockIdx.x] = sumdata1[0];
		p_sum_d_d[blockIdx.x] = sumdata2[0];
		p_sum_eps_eps[blockIdx.x] = sumdata3[0];
	}
}


__global__ void kernelDividebyroothgamma
(
	f64 * __restrict__ result,
	f64 * __restrict__ Az,
	f64 const hsub,
	f64 * __restrict__ p_gamma
)
{
	long const index = threadIdx.x + blockIdx.x*blockDim.x;
	f64 gamma = p_gamma[index];
	if (gamma == 0.0) {
		result[index] = 0.0;
	} else {
		result[index] = Az[index] / gamma;
	}
}


__global__ void kernelInterpolateVarsAndPositions(
	f64 ppn,
	structural * __restrict__ p_info1,
	structural * __restrict__ p_info2,
	nvals * __restrict__ p_n_minor1,
	nvals * __restrict__ p_n_minor2,
	T3 * __restrict__ p_T_minor1,
	T3 * __restrict__ p_T_minor2,
//	f64_vec3 * __restrict__ p_B1,
//	f64_vec3 * __restrict__ p_B2,

	structural * __restrict__ p_info_dest,
	nvals * __restrict__ p_n_minor,
	T3 * __restrict__ p_T_minor
	//f64_vec3 * __restrict__ p_B
	)
{
	long const iMinor = blockIdx.x*blockDim.x + threadIdx.x;
	structural info1 = p_info1[iMinor];
	structural info2 = p_info2[iMinor];
	structural info;
	f64 r = 1.0 - ppn;
	info.pos = r*info1.pos + ppn*info2.pos;
	info.flag = info1.flag;
	info.neigh_len = info1.neigh_len;
	p_info_dest[iMinor] = info;

	nvals nvals1 = p_n_minor1[iMinor];
	nvals nvals2 = p_n_minor2[iMinor];
	nvals nvals_dest;
	nvals_dest.n = r*nvals1.n + ppn*nvals2.n;
	nvals_dest.n_n = r*nvals1.n_n + ppn*nvals2.n_n;
	p_n_minor[iMinor] = nvals_dest;

	T3 T1 = p_T_minor1[iMinor];
	T3 T2 = p_T_minor2[iMinor];
	T3 T;
	T.Te = r*T1.Te + ppn*T2.Te;
	T.Ti = r*T1.Ti + ppn*T2.Ti;
	T.Tn = r*T1.Tn + ppn*T2.Tn;
	p_T_minor[iMinor] = T;

//	f64_vec3 B1 = p_B1[iMinor];
//	f64_vec3 B2 = p_B2[iMinor];
//	f64_vec3 B = r*B1 + ppn*B2;
//	p_B[iMinor] = B;
}

// Correct disposition of routines:
// --- union of T and [v + v_overall] -- uses n_shards --> pressure, momflux, grad Te
// --- union of T and [v + v_overall] -- uses n_n shards --> neutral pressure, neutral momflux
// --- Az,Azdot + v_overall -- runs for whole domain ---> Lap A, curl A, grad A, grad Adot, ROCAz, ROCAzdot
//    ^^ base off of GetLap_minor.

// Worst case number of vars:
// (4+2)*1.5+6.5 <-- because we use v_vertex. + 3 for positions. 
// What can we stick in L1? n_cent we could.
// We should be aiming a ratio 3:1 from shared:L1, if registers are small.
// For tris we are using n_shards from shared points.
// And it is for tris that we require vertex data v to be present.
// Idea: vertex code determines array of 12 relevant n and sticks them into shared.
// Only saved us 1 var. 9 + 6 + 3 = 18.
// Still there is premature optimization here -- none of this happens OFTEN.


__global__ void Augment_dNv_minor(
	structural * __restrict__ p_info,
	nvals * __restrict__ p_n_minor,
	LONG3 * __restrict__ p_tricornerindex,
	f64 * __restrict__ p_temp_Ntotalmajor,
	f64 * __restrict__ p_temp_Nntotalmajor,
	f64 * __restrict__ p_AreaMinor,
	f64_vec3 * __restrict__ p_MAR_neut_major,
	f64_vec3 * __restrict__ p_MAR_ion_major,
	f64_vec3 * __restrict__ p_MAR_elec_major,
	f64_vec3 * __restrict__ p_MAR_neut,
	f64_vec3 * __restrict__ p_MAR_ion,
	f64_vec3 * __restrict__ p_MAR_elec)
{
	long iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info[iMinor];

	if (info.flag == DOMAIN_TRIANGLE)
	{
		if (iMinor < BEGINNING_OF_CENTRAL)
		{
			LONG3 tricornerindex = p_tricornerindex[iMinor];
			nvals nminor = p_n_minor[iMinor];
			f64 areaminor = p_AreaMinor[iMinor];
			f64 Nhere = areaminor * nminor.n;
			f64 Nnhere =areaminor * nminor.n_n;
			f64 coeff1 = 0.333333333333333*Nhere / p_temp_Ntotalmajor[tricornerindex.i1];
			f64 coeff2 = 0.333333333333333*Nhere / p_temp_Ntotalmajor[tricornerindex.i2];
			f64 coeff3 = 0.333333333333333*Nhere / p_temp_Ntotalmajor[tricornerindex.i3];
		
			// this may be dividing by 0 if the corner is not a domain vertex -- so for ease we stick to domain minors

			f64_vec3 add_i = p_MAR_ion_major[tricornerindex.i1] * coeff1
				+ p_MAR_ion_major[tricornerindex.i2] * coeff2
				+ p_MAR_ion_major[tricornerindex.i3] * coeff3;
			f64_vec3 add_e = p_MAR_elec_major[tricornerindex.i1] * coeff1
				+ p_MAR_elec_major[tricornerindex.i2] * coeff2
				+ p_MAR_elec_major[tricornerindex.i3] * coeff3;

			coeff1 = 0.333333333333333*Nnhere / p_temp_Nntotalmajor[tricornerindex.i1];
			coeff2 = 0.333333333333333*Nnhere / p_temp_Nntotalmajor[tricornerindex.i2];
			coeff3 = 0.333333333333333*Nnhere / p_temp_Nntotalmajor[tricornerindex.i3];

			f64_vec3 add_n = p_MAR_neut_major[tricornerindex.i1] * coeff1
				+ p_MAR_neut_major[tricornerindex.i2] * coeff2
				+ p_MAR_neut_major[tricornerindex.i3] * coeff3;

			p_MAR_neut[iMinor] += add_n;
			p_MAR_ion[iMinor] += add_i;
			p_MAR_elec[iMinor] += add_e;
		} else {
			nvals nminor = p_n_minor[iMinor];
			f64 Nhere = p_AreaMinor[iMinor] * nminor.n_n;
			f64 coeff = Nhere / p_temp_Ntotalmajor[iMinor - BEGINNING_OF_CENTRAL];
			f64_vec3 add_i = p_MAR_ion_major[iMinor - BEGINNING_OF_CENTRAL] * coeff;
			f64_vec3 add_e = p_MAR_elec_major[iMinor - BEGINNING_OF_CENTRAL] * coeff;
			f64 Nnhere = p_AreaMinor[iMinor] * nminor.n;
			coeff = Nnhere / p_temp_Nntotalmajor[iMinor - BEGINNING_OF_CENTRAL];
			f64_vec3 add_n = p_MAR_neut_major[iMinor - BEGINNING_OF_CENTRAL] * coeff;

			p_MAR_neut[iMinor] += add_n;
			p_MAR_ion[iMinor] += add_i;
			p_MAR_elec[iMinor] += add_e;
		};		
	};
}


__global__ void Collect_Ntotal_major(
	structural * __restrict__ p_info_minor,
	long * __restrict__ p_izTri,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor, 
	f64 * __restrict__ p_temp_Ntotalmajor,
	f64 * __restrict__ p_temp_Nntotalmajor)
{
	long iVertex = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];	
	long izTri[MAXNEIGH_d];
	short i;
	
	if (info.flag == DOMAIN_VERTEX) {
		memcpy(izTri, p_izTri + MAXNEIGH_d*iVertex, sizeof(long)*MAXNEIGH_d);
		nvals ncentral = p_n_minor[iVertex + BEGINNING_OF_CENTRAL];
		f64 areaminorhere = p_AreaMinor[iVertex + BEGINNING_OF_CENTRAL];
		f64 sum_N = ncentral.n*areaminorhere;
		f64 sum_Nn = ncentral.n_n*areaminorhere;
		f64 areaminor;
		nvals nminor;
		for (i = 0; i < info.neigh_len; i++)
		{
			if (p_info_minor[izTri[i]].flag == DOMAIN_TRIANGLE) // see above
			{
				nminor = p_n_minor[izTri[i]];
				areaminor = p_AreaMinor[izTri[i]];
				sum_N += 0.33333333333333*nminor.n*areaminor;
				sum_Nn += 0.33333333333333*nminor.n_n*areaminor;
			}
		};
		p_temp_Ntotalmajor[iVertex] = sum_N;
		p_temp_Nntotalmajor[iVertex] = sum_Nn;
	};
}
// We could probably create a big speedup by having a set of blocks that index only the DOMAIN!


__global__ void Collect_Nsum_at_tris(
	structural * __restrict__ p_info,
	nvals * __restrict__ p_n_minor,
	LONG3 * __restrict__ p_tricornerindex,
	f64 * __restrict__ p_AreaMajor, // populated?
	f64 * __restrict__ p_Nsum,
	f64 * __restrict__ p_Nsum_n)
 {
	long iTri = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info[iTri];
	if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS))
	{
		LONG3 tricornerindex = p_tricornerindex[iTri];
		p_Nsum[iTri] = p_n_minor[BEGINNING_OF_CENTRAL + tricornerindex.i1].n * p_AreaMajor[tricornerindex.i1]
			+ p_n_minor[BEGINNING_OF_CENTRAL + tricornerindex.i2].n * p_AreaMajor[tricornerindex.i2]
			+ p_n_minor[BEGINNING_OF_CENTRAL + tricornerindex.i3].n * p_AreaMajor[tricornerindex.i3];

		p_Nsum_n[iTri] = p_n_minor[BEGINNING_OF_CENTRAL + tricornerindex.i1].n_n * p_AreaMajor[tricornerindex.i1]
			+ p_n_minor[BEGINNING_OF_CENTRAL + tricornerindex.i2].n_n * p_AreaMajor[tricornerindex.i2]
			+ p_n_minor[BEGINNING_OF_CENTRAL + tricornerindex.i3].n_n * p_AreaMajor[tricornerindex.i3];

//		if (tricornerindex.i1 == VERTCHOSEN) printf("%d corner 1 = %d p_n_minor %1.10E AreaMajor %1.10E Nsum %1.10E\n", iTri, VERTCHOSEN,
//			p_n_minor[BEGINNING_OF_CENTRAL + tricornerindex.i1].n, p_AreaMajor[tricornerindex.i1], p_Nsum[iTri]);
//		if (tricornerindex.i2 == VERTCHOSEN) printf("%d corner 2 = %d p_n_minor %1.10E AreaMajor %1.10E Nsum %1.10E\n", iTri, VERTCHOSEN,
//			p_n_minor[BEGINNING_OF_CENTRAL + tricornerindex.i2].n, p_AreaMajor[tricornerindex.i2], p_Nsum[iTri]);
//		if (tricornerindex.i3 == VERTCHOSEN) printf("%d corner 3 = %d p_n_minor %1.10E AreaMajor %1.10E Nsum %1.10E\n", iTri, VERTCHOSEN,
//			p_n_minor[BEGINNING_OF_CENTRAL + tricornerindex.i3].n, p_AreaMajor[tricornerindex.i3], p_Nsum[iTri]);

	} else {
		p_Nsum[iTri] = 1.0;
		p_Nsum_n[iTri] = 1.0;
	}
}

__global__ void kernelTransmitHeatToVerts(
	structural * __restrict__ p_info,
	long * __restrict__ p_izTri,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMajor, // populated?
	f64 * __restrict__ p_Nsum,
	f64 * __restrict__ p_Nsum_n,
	NTrates * __restrict__ NT_addition_rates,
	NTrates * __restrict__ NT_addition_tri
) {
	long iVertex = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info[iVertex + BEGINNING_OF_CENTRAL];	
	nvals n_use = p_n_minor[iVertex + BEGINNING_OF_CENTRAL];
	f64 AreaMajor = p_AreaMajor[iVertex];
	f64 N = n_use.n*AreaMajor;
	f64 Nn = n_use.n_n*AreaMajor;

	long izTri[MAXNEIGH_d];
	short i;
	f64 sum_NeTe = 0.0, sum_NiTi = 0.0, sum_NnTn = 0.0;
	if (info.flag == DOMAIN_VERTEX) {
		memcpy(izTri, p_izTri + MAXNEIGH_d*iVertex, sizeof(long)*MAXNEIGH_d);
		for (i = 0; i < info.neigh_len; i++)
		{
			sum_NiTi += (N / p_Nsum[izTri[i]])*	NT_addition_tri[izTri[i]].NiTi;
			sum_NeTe += (N / p_Nsum[izTri[i]])*	NT_addition_tri[izTri[i]].NeTe;
			sum_NnTn += (Nn / p_Nsum_n[izTri[i]])*	NT_addition_tri[izTri[i]].NnTn;
			// stabilize in the way we apportion heat out of triangle
			
		};
		NT_addition_rates[iVertex].NiTi += sum_NiTi;
		NT_addition_rates[iVertex].NeTe += sum_NeTe;
		NT_addition_rates[iVertex].NnTn += sum_NnTn;
	}

	// Idea: pre-store a value which is the sum of N at corners.
}

// Not optimized: !!
#define FACTOR_HALL (1.0/0.96)
#define FACTOR_PERP (1.2/0.96)
#define DEBUGNANS

__global__ void kernelMultiply_Get_Jacobi_Visc(
	structural * __restrict__ p_info,
	f64_vec2 * __restrict__ p_eps_xy,
	f64 * __restrict__ p_eps_iz,
	f64 * __restrict__ p_eps_ez,
	f64_tens3 * __restrict__ p_Matrix_i,
	f64_tens3 * __restrict__ p_Matrix_e,
	f64_vec3 * __restrict__ p_Jacobi_ion,
	f64_vec3 * __restrict__ p_Jacobi_elec
) {
	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info[iMinor];
	if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS))
	{
		f64_tens3 Matrix;
		memcpy(&Matrix, p_Matrix_i + iMinor, sizeof(f64_tens3));
		f64_vec3 Jacobi;
		f64_vec3 epsilon;
		memcpy(&epsilon, &(p_eps_xy[iMinor]), sizeof(f64_vec2));
		epsilon.z = p_eps_iz[iMinor];
		Jacobi = Matrix*epsilon;
		p_Jacobi_ion[iMinor] = Jacobi;

		memcpy(&Matrix, p_Matrix_e + iMinor, sizeof(f64_tens3));
		epsilon.z = p_eps_ez[iMinor];
		Jacobi = Matrix*epsilon;
		p_Jacobi_elec[iMinor] = Jacobi;

		// That simple.
	}
	else {
		// Jacobi = 0
		memset(&(p_Jacobi_ion[iMinor]), 0, sizeof(f64_vec3));
		memset(&(p_Jacobi_elec[iMinor]), 0, sizeof(f64_vec3));
	}

}

__global__ void kernelCreateSeedPartOne(
	f64 const h_use,
	f64 * __restrict__ p_Az,
	AAdot * __restrict__ p_AAdot_use,
	f64 * __restrict__ p_AzNext
) {
	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	p_AzNext[iMinor] = p_Az[iMinor] + h_use*0.5*p_AAdot_use[iMinor].Azdot;
}

__global__ void kernelCreateSeedPartTwo(
	f64 const h_use,
	f64 * __restrict__ p_Azdot0, 
	f64 * __restrict__ p_gamma, 
	f64 * __restrict__ p_LapAz,
	f64 * __restrict__ p_AzNext_update
) {
	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	p_AzNext_update[iMinor] += 0.5*h_use* (p_Azdot0[iMinor]
		+ p_gamma[iMinor] * p_LapAz[iMinor]);
}

__global__ void SubtractVector(
	f64 * __restrict__ result,
	f64 * __restrict__ b,
	f64 * __restrict__ a) 
{
	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	result[iMinor] = a[iMinor] - b[iMinor];
}

__global__ void kernelCreateSeedAz(
	f64 const h_use,
	f64 * __restrict__ p_Az_k,
	f64 * __restrict__ p_Azdot0,
	f64 * __restrict__ p_gamma,
	f64 * __restrict__ p_LapAz,
	f64 * __restrict__ p_AzNext
) {
	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	p_AzNext[iMinor] = p_Az_k[iMinor] + h_use*
		(p_Azdot0[iMinor] + p_gamma[iMinor] * p_LapAz[iMinor]);
	// This seed is suitable if we have no historic data
	// Given 3 points we can make a cubic extrap that should be better.
	// We could then record the proportion of where the solution lay (least squares) between
	// this seed and the cubic seed, see if that has a pattern, and if so, 
	// be recording it (weighted average 50:30:20 of the last 3), use that to provide a better seed still.
	// Detecting the LS optimal proportion depends on writing another bit of code. We could actually
	// just run a regression and record 2 coefficients. If one of them is negative that's not so funny.
	// We could even come up with a 3rd regressor such as Jz or Azdot_k.
}


__global__ void kernelWrapVertices(
	structural * __restrict__ p_info_minor,
	v4 * __restrict__ p_vie,
	f64_vec3 * __restrict__ p_v_n,
	char * __restrict__ p_was_vertex_rotated
) {
	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
	// I SEE NOW that I am borrowing long const from CPU which is only a backdoor.

	if (info.pos.x*(1.0 - 1.0e-13) > info.pos.y*GRADIENT_X_PER_Y) {
		info.pos = Anticlockwise_d*info.pos;

		v4 vie = p_vie[iVertex + BEGINNING_OF_CENTRAL];
		vie.vxy = Anticlockwise_d*vie.vxy;
		f64_vec3 v_n = p_v_n[iVertex + BEGINNING_OF_CENTRAL];
		v_n = Anticlock_rotate3(v_n);
		p_vie[iVertex + BEGINNING_OF_CENTRAL] = vie;
		p_v_n[iVertex + BEGINNING_OF_CENTRAL] = v_n;
		p_info_minor[iVertex + BEGINNING_OF_CENTRAL] = info;
		
		// Now let's worry about rotating variables in all the triangles that become periodic.
		// Did we do that before in cpp file? Yes.
	
		// We probably need to set a flag on tris modified and launch later.
		// Violating gather-not-scatter. Save a char instead.

		// Also: reassess PBC lists for vertex.
		
		p_was_vertex_rotated[iVertex] = ROTATE_ME_ANTICLOCKWISE;
	};
	if (info.pos.x*(1.0 - 1.0e-13) < -info.pos.y*GRADIENT_X_PER_Y) {

		info.pos = Clockwise_d*info.pos;
		v4 vie = p_vie[iVertex + BEGINNING_OF_CENTRAL];
		vie.vxy = Clockwise_d*vie.vxy;
		f64_vec3 v_n = p_v_n[iVertex + BEGINNING_OF_CENTRAL];
		v_n = Clockwise_rotate3(v_n);

		p_vie[iVertex + BEGINNING_OF_CENTRAL] = vie;
		p_v_n[iVertex + BEGINNING_OF_CENTRAL] = v_n;
		p_info_minor[iVertex + BEGINNING_OF_CENTRAL] = info;
		p_was_vertex_rotated[iVertex] = ROTATE_ME_CLOCKWISE;
	};	

	// Here we could add in some code to add up 1 for each wrapped vertex in the block
	// or just a bool whether any in the block wrapped.

}

__global__ void kernelWrapTriangles(
	structural * __restrict__ p_info_minor,
	LONG3 * __restrict__ p_tri_corner_index, 
	char * __restrict__ p_was_vertex_rotated,

	v4 * __restrict__ p_vie_minor,
	f64_vec3 * __restrict__ p_v_n_minor,
	char * __restrict__ p_triPBClistaffected,
	CHAR4 * __restrict__ p_tri_periodic_corner_flags
) {
	long iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	structural info_tri = p_info_minor[iMinor];

	LONG3 cornerindex = p_tri_corner_index[iMinor];

	// Inefficient, no shared mem used:
	char flag0 = p_was_vertex_rotated[cornerindex.i1];
	char flag1 = p_was_vertex_rotated[cornerindex.i2];
	char flag2 = p_was_vertex_rotated[cornerindex.i3];

	if ((flag0 == 0) && (flag1 == 0) && (flag2 == 0))
	{
		// typical case: do nothing
	} else {

		// okay... it is near the PBC edge, because a vertex wrapped.

		// if all vertices are on left or right, it's not a periodic triangle.
		// We need to distinguish what happened: if on one side all the vertices are newly crossed over,
		// then it didn't used to be periodic but now it is. If that is the left side, we need to rotate tri data.
		// If all are now on right, we can rotate tri data to the right. It used to be periodic, guaranteed.

		structural info[3];
		info[0] = p_info_minor[BEGINNING_OF_CENTRAL + cornerindex.i1];
		info[1] = p_info_minor[BEGINNING_OF_CENTRAL + cornerindex.i2];
		info[2] = p_info_minor[BEGINNING_OF_CENTRAL + cornerindex.i3];

		// We are going to set this for the corners whether this tri rotates or not:
		p_triPBClistaffected[cornerindex.i1] = 1;
		p_triPBClistaffected[cornerindex.i2] = 1;
		p_triPBClistaffected[cornerindex.i3] = 1;

		if ((info[0].pos.x > 0.0) && (info[1].pos.x > 0.0) && (info[2].pos.x > 0.0))
		{
			// All now on right => previously some were on left.

			if (TESTTRI) printf("%d All on right\n",iMinor);
			
			p_vie_minor[iMinor].vxy = Clockwise_d*p_vie_minor[iMinor].vxy;
			info_tri.pos = Clockwise_d*info_tri.pos;
			p_v_n_minor[iMinor] = Clockwise_rotate3(p_v_n_minor[iMinor]);
		} else {
			if (((info[0].pos.x > 0.0) || (flag0 == ROTATE_ME_ANTICLOCKWISE))
				&&
				((info[1].pos.x > 0.0) || (flag1 == ROTATE_ME_ANTICLOCKWISE))
				&&
				((info[2].pos.x > 0.0) || (flag2 == ROTATE_ME_ANTICLOCKWISE)))
			{
				// Logic here?
				// Iff all that are on the left are new, then for the first time we are periodic and need to rotate.
				if (TESTTRI) printf("%d Second condition\n", iMinor);

				p_vie_minor[iMinor].vxy = Anticlockwise_d*p_vie_minor[iMinor].vxy;
				info_tri.pos = Anticlockwise_d*info_tri.pos;
				p_v_n_minor[iMinor] = Anticlock_rotate3(p_v_n_minor[iMinor]);
			}
		}
		p_info_minor[iMinor] = info_tri;
		if (TESTTRI) printf("%d info_tri.pos %1.9E %1.9E \n", iMinor, info_tri.pos.x, info_tri.pos.y);

		// Now reassess periodic for corners:
		CHAR4 tri_per_corner_flags;
		memset(&tri_per_corner_flags, 0, sizeof(CHAR4));
		tri_per_corner_flags.flag = info_tri.flag;
		if (((info[0].pos.x > 0.0) && (info[1].pos.x > 0.0) && (info[2].pos.x > 0.0))
			||
			((info[0].pos.x < 0.0) && (info[1].pos.x < 0.0) && (info[2].pos.x < 0.0)))
		{
			// 0 is correct -- triangles only ever rotate corners anticlockwise
			tri_per_corner_flags.per0 = 0;
			tri_per_corner_flags.per1 = 0;
			tri_per_corner_flags.per2 = 0;
			// this was a bug?
		} else {
			if (info[0].pos.x > 0.0) tri_per_corner_flags.per0 = ROTATE_ME_ANTICLOCKWISE;
			if (info[1].pos.x > 0.0) tri_per_corner_flags.per1 = ROTATE_ME_ANTICLOCKWISE;
			if (info[2].pos.x > 0.0) tri_per_corner_flags.per2 = ROTATE_ME_ANTICLOCKWISE;
		}
		
		p_tri_periodic_corner_flags[iMinor] = tri_per_corner_flags;
		if (TESTTRI) printf("%d flags %d %d %d\n",
			iMinor, tri_per_corner_flags.per0, tri_per_corner_flags.per1, tri_per_corner_flags.per2);
	};
}

__global__ void kernelReassessTriNeighbourPeriodicFlags_and_populate_PBCIndexneighminor(
	structural * __restrict__ p_info_minor,
	LONG3 * __restrict__ p_tri_neigh_index,
	LONG3 * __restrict__ p_tri_corner_index,
	char * __restrict__ p_was_vertex_rotated,
	CHAR4 * __restrict__ p_tri_periodic_corner_flags,
	CHAR4 * __restrict__ p_tri_periodic_neigh_flags,
	char * __restrict__ p_szPBC_triminor,
	char * __restrict__ p_triPBClistaffected
	)
{
	CHAR4 tri_periodic_neigh_flags;

	long const iTri = blockDim.x*blockIdx.x + threadIdx.x;
	LONG3 cornerindex = p_tri_corner_index[iTri];

	// Inefficient, no shared mem used:
	char flag0 = p_triPBClistaffected[cornerindex.i1];
	char flag1 = p_triPBClistaffected[cornerindex.i2];
	char flag2 = p_triPBClistaffected[cornerindex.i3];

	//char flag0 = p_was_vertex_rotated[cornerindex.i1];
	
	if ((flag0 == 0) && (flag1 == 0) && (flag2 == 0))
	{
		// typical case: do nothing
		
	} else {
		// A neighbour tri had a vertex that wrapped.

		structural info = p_info_minor[iTri];

		LONG3 tri_neigh_index = p_tri_neigh_index[iTri];

		memset(&tri_periodic_neigh_flags, 0, sizeof(CHAR4));
		tri_periodic_neigh_flags.flag = info.flag;

		if (info.pos.x > 0.0) {

			CHAR4 test = p_tri_periodic_corner_flags[tri_neigh_index.i1];
			if ((test.per0 != 0) || (test.per1 != 0) || (test.per2 != 0))
				tri_periodic_neigh_flags.per0 = ROTATE_ME_CLOCKWISE;

			test = p_tri_periodic_corner_flags[tri_neigh_index.i2];
			if ((test.per0 != 0) || (test.per1 != 0) || (test.per2 != 0))
				tri_periodic_neigh_flags.per1 = ROTATE_ME_CLOCKWISE;

			test = p_tri_periodic_corner_flags[tri_neigh_index.i3];
			if ((test.per0 != 0) || (test.per1 != 0) || (test.per2 != 0))
				tri_periodic_neigh_flags.per2 = ROTATE_ME_CLOCKWISE;
		}
		else {
			// if we are NOT periodic but on left, neighs are not rotated rel to us.
			// If we ARE periodic but neigh is not and neigh cent > 0.0 then it is rotated.

			CHAR4 ours = p_tri_periodic_corner_flags[iTri];
			if ((ours.per0 != 0) && (ours.per1 != 0) && (ours.per2 != 0)) // ours IS periodic
			{

				structural info0 = p_info_minor[tri_neigh_index.i1];
				structural info1 = p_info_minor[tri_neigh_index.i2];
				structural info2 = p_info_minor[tri_neigh_index.i3];

				if (info0.pos.x > 0.0) tri_periodic_neigh_flags.per0 = ROTATE_ME_ANTICLOCKWISE;
				if (info1.pos.x > 0.0) tri_periodic_neigh_flags.per1 = ROTATE_ME_ANTICLOCKWISE;
				if (info2.pos.x > 0.0) tri_periodic_neigh_flags.per2 = ROTATE_ME_ANTICLOCKWISE;

				//	if ((pTri->neighbours[1]->periodic == 0) && (pTri->neighbours[1]->cent.x > 0.0))
					//	tri_periodic_neigh_flags.per1 = ROTATE_ME_ANTICLOCKWISE;			
			};
		};

		p_tri_periodic_neigh_flags[iTri] = tri_periodic_neigh_flags;

		// Set indexneigh periodic list for this tri:
		CHAR4 tri_periodic_corner_flags = p_tri_periodic_corner_flags[iTri];
		char szPBC_triminor[6];
		szPBC_triminor[0] = tri_periodic_corner_flags.per0;
		szPBC_triminor[1] = tri_periodic_neigh_flags.per2;
		szPBC_triminor[2] = tri_periodic_corner_flags.per1;
		szPBC_triminor[3] = tri_periodic_neigh_flags.per0;
		szPBC_triminor[4] = tri_periodic_corner_flags.per2;
		szPBC_triminor[5] = tri_periodic_neigh_flags.per1;
		memcpy(p_szPBC_triminor + 6 * iTri, szPBC_triminor, sizeof(char) * 6);

	}; // was a corner a corner of a tri that had a corner wrapped
}

__global__ void kernelReset_szPBCtri_vert( // would rather it say Update not Reset
	structural * __restrict__ p_info_minor,
	long * __restrict__ p_izTri_vert,
	long * __restrict__ p_izNeigh_vert,
	char * __restrict__ p_szPBCtri_vert, 
	char * __restrict__ p_szPBCneigh_vert,
	char * __restrict__ p_triPBClistaffected
)
{
	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;

	short i;

	structural info = p_info_minor[BEGINNING_OF_CENTRAL + iVertex];
	if (p_triPBClistaffected[iVertex] != 0) {
		char szPBCtri[MAXNEIGH];
		char szPBCneigh[MAXNEIGH];
		long izTri[MAXNEIGH];
		long izNeigh[MAXNEIGH];

		// Now reassess PBC lists for tris 
		memcpy(izTri, p_izTri_vert + MAXNEIGH*iVertex, sizeof(long)*MAXNEIGH);
		structural infotri;
		if (info.pos.x > 0.0) {
#pragma unroll MAXNEIGH
			for (i = 0; i < info.neigh_len; i++)
			{
				infotri = p_info_minor[izTri[i]];
				szPBCtri[i] = 0;
				if (infotri.pos.x < 0.0) szPBCtri[i] = ROTATE_ME_CLOCKWISE;

				if (TEST) printf("%d info.pos.x %1.9E RIGHT iTri %d : i %d infotri.pos.x %1.9E szPBCtri[i] %d\n",
					iVertex, info.pos.x, izTri[i], i, infotri.pos.x, (int)szPBCtri[i]);
			};
		} else {
#pragma unroll MAXNEIGH
			for (i = 0; i < info.neigh_len; i++)
			{
				infotri = p_info_minor[izTri[i]];
				szPBCtri[i] = 0;
				if (infotri.pos.x > 0.0) szPBCtri[i] = ROTATE_ME_ANTICLOCKWISE;

				if (TEST) printf("%d info.pos.x %1.9E : i %d iTri %d infotri.pos.x %1.9E szPBCtri[i] %d\n",
					iVertex, info.pos.x, i, izTri[i], infotri.pos.x, (int)szPBCtri[i]);
			};
		};
		memcpy(p_szPBCtri_vert + MAXNEIGH*iVertex, szPBCtri, sizeof(char)*MAXNEIGH);

		// If a neighbour wrapped then we share a tri with it that will have given us the
		// PBC tri affected flag. 
		structural infoneigh;

		memcpy(izNeigh, p_izNeigh_vert + MAXNEIGH*iVertex, sizeof(long)*MAXNEIGH);

		if (info.pos.x > 0.0) {
#pragma unroll MAXNEIGH
			for (i = 0; i < info.neigh_len; i++)
			{
				infoneigh = p_info_minor[izNeigh[i] + BEGINNING_OF_CENTRAL];
				szPBCneigh[i] = 0;
				if (infoneigh.pos.x < 0.0) szPBCneigh[i] = ROTATE_ME_CLOCKWISE;
			};
		}
		else {
#pragma unroll MAXNEIGH
			for (i = 0; i < info.neigh_len; i++)
			{
				infoneigh = p_info_minor[izNeigh[i] + BEGINNING_OF_CENTRAL];
				szPBCneigh[i] = 0;
				if (infoneigh.pos.x > 0.0) szPBCneigh[i] = ROTATE_ME_ANTICLOCKWISE;
			};
		};

		memcpy(p_szPBCneigh_vert + MAXNEIGH*iVertex, szPBCneigh, sizeof(char)*MAXNEIGH);

	} else {
		// no update
	}
	
	// Possibly could also argue that if triPBClistaffected == 0 then as it had no wrapping
	// triangle it cannot have a wrapping neighbour. Have to visualise to be sure.
}


