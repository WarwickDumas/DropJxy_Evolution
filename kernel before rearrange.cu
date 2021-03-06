#include "kernel.h"
#include "helpers.cu"
#include "vector_tensor.cu"
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
#define TEST  (0)
#define TESTTRI (0)
#define TESTADVECT (0)// iVertex == VERTCHOSEN) // iVertex == VERTCHOSEN)
#define TESTHEAT (0)
#define TESTHEATFULL (0)
#define TESTHEAT1 (0)// iVertex == VERTCHOSEN)
#define TESTTRI2 (0)
#define TESTTRI3 (0)
#define TESTHEAT2 (0)
#define TESTIONISE (0)
#define VISCMAG 1 
#define MIDPT_A

#define ARTIFICIAL_RELATIVE_THRESH  1.0e9
#define ARTIFICIAL_RELATIVE_THRESH_HEAT  1.0e11

// Change log. 090419: Change upwind density routine to just use n from the lowest cell that is upwind for at least 1 side.
// 230419: Change nu_n used in kappa_neut to be a lc of collision frequencies.

// 250419: Change to use min(ita_ours, ita_theirs). Maybe need to do same for kappa_par. 
// Change to apportion visc heat from tri per N.

//const int Chosens[7] = { 25454, 86529, 25453, 86381, 25455, 86530, 25750 };


__global__ void kernelCreateEpsilon_Visc(
	f64 const hsub,
	structural * __restrict__ p_info_minor,
	v4 * __restrict__ p_vie,
	v4 * __restrict__ p_vie_k,
	f64_vec3 * __restrict__ p_MAR_ion2,
	f64_vec3 * __restrict__ p_MAR_elec2,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor,
	
	f64_vec2 * __restrict__ p_epsilon_xy,
	f64 * __restrict__ p_epsilon_iz,
	f64 * __restrict__ p_epsilon_ez
) {
	long const iMinor = blockDim.x * blockIdx.x + threadIdx.x;

	// eps = v - v_k - h MAR / N
	structural info = p_info_minor[iMinor];
	v4 epsilon;
	memset(&epsilon, 0, sizeof(v4));
	if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE)
		|| (info.flag == CROSSING_INS)) // ?
	{
		v4 vie = p_vie[iMinor];
		v4 vie_k = p_vie_k[iMinor];
		f64_vec3 MAR_ion = p_MAR_ion2[iMinor];
		f64_vec3 MAR_elec = p_MAR_elec2[iMinor];
		f64 N = p_AreaMinor[iMinor] * p_n_minor[iMinor].n;

		epsilon.vxy = vie.vxy - vie_k.vxy
			- hsub*((MAR_ion.xypart()*m_ion + MAR_elec.xypart()*m_e) /
			((m_ion + m_e)*N));
		epsilon.viz = vie.viz - vie_k.viz
			- hsub*(MAR_ion.z / N);
		epsilon.vez = vie.vez - vie_k.vez
			- hsub*(MAR_elec.z / N);

		// ###################################################
		// Remember that when we do it properly we need to add the MAR
		// that comes from other causes. Similar with heat!
		// ###################################################

	} else {
		// epsilon = 0
	};
	p_epsilon_xy[iMinor] = epsilon.vxy;
	p_epsilon_iz[iMinor] = epsilon.viz;
	p_epsilon_ez[iMinor] = epsilon.vez;

}

__global__ void kernelCalcJacobi_Viscosity(

	structural * __restrict__ p_info_minor,
	
	f64_vec2 * __restrict__ p_epsilon_xy,
	f64 * __restrict__ p_epsilon_iz,
	f64 * __restrict__ p_epsilon_ez,

	f64_tens3 * __restrict__ p_matrix_i,
	f64_tens3 * __restrict__ p_matrix_e, // inverted matrix R^-1 so Jacobi = R^-1 epsilon
	
	f64_vec3 * __restrict__ p_Jacobi_i,
	f64_vec3 * __restrict__ p_Jacobi_e)
{
	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	long const iVertex = threadsPerTileMajor*blockIdx.x + threadIdx.x; // only meaningful threadIdx.x < threadsPerTileMajor

	structural info = p_info_minor[iMinor];
	if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS)) {
		f64_vec2 epsilon_xy = p_epsilon_xy[iMinor];
		f64 epsilon_iz = p_epsilon_iz[iMinor];
		f64 epsilon_ez = p_epsilon_ez[iMinor];

		f64_tens3 matrix = p_matrix_i[iMinor];
		p_Jacobi_i[iMinor] = matrix*Make3(epsilon_xy, epsilon_iz);
		matrix = p_matrix_e[iMinor];
		p_Jacobi_e[iMinor] = matrix*Make3(epsilon_xy, epsilon_ez);

	} else {
		memset(&(p_Jacobi_i[iMinor]), 0, sizeof(f64_vec3));
		memset(&(p_Jacobi_e[iMinor]), 0, sizeof(f64_vec3));
	}
	
	if (threadIdx.x < threadsPerTileMajor) {
		
		info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
		if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {
			f64_vec2 epsilon_xy = p_epsilon_xy[iVertex + BEGINNING_OF_CENTRAL];
			f64 epsilon_iz = p_epsilon_iz[iVertex + BEGINNING_OF_CENTRAL];
			f64 epsilon_ez = p_epsilon_ez[iVertex + BEGINNING_OF_CENTRAL];

			f64_tens3 matrix = p_matrix_i[iVertex + BEGINNING_OF_CENTRAL];
			p_Jacobi_i[iVertex + BEGINNING_OF_CENTRAL] = matrix*Make3(epsilon_xy, epsilon_iz);
			matrix = p_matrix_e[iVertex + BEGINNING_OF_CENTRAL];
			p_Jacobi_e[iVertex + BEGINNING_OF_CENTRAL] = matrix*Make3(epsilon_xy, epsilon_ez);

		} else {
			memset(&(p_Jacobi_i[iVertex + BEGINNING_OF_CENTRAL]), 0, sizeof(f64_vec3));
			memset(&(p_Jacobi_e[iVertex + BEGINNING_OF_CENTRAL]), 0, sizeof(f64_vec3));
		};
	}
}

__global__ void kernelCalc_Matrices_for_Jacobi_Viscosity(
	f64 const hsub,
	structural * __restrict__ p_info_minor,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtriminor,

	f64 * __restrict__ p_ita_parallel_ion_minor,   // nT / nu ready to look up
	f64 * __restrict__ p_ita_parallel_elec_minor,   // nT / nu ready to look up
	f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
	f64 * __restrict__ p_nu_elec_minor,   // nT / nu ready to look up
	f64_vec3 * __restrict__ p_B_minor,
		
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor,

	f64_tens3 * __restrict__ p_matrix_i,
	f64_tens3 * __restrict__ p_matrix_e
	)
	// Tomorrow start by doing explanation for Ignas etc.	
{
	//__shared__ v4 shared_vie[threadsPerTileMinor]; // sort of thing we want as input
	// Not used, right? Nothing nonlinear?
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];
	__shared__ f64_vec2 shared_B[threadsPerTileMinor];
	__shared__ f64 shared_ita_par[threadsPerTileMinor]; // reuse for i,e ; or make 2 vars to combine the routines.
	__shared__ f64 shared_nu[threadsPerTileMinor];

	//__shared__ v4 shared_vie_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_B_verts[threadsPerTileMajor];
	__shared__ f64 shared_ita_par_verts[threadsPerTileMajor];
	__shared__ f64 shared_nu_verts[threadsPerTileMajor]; // used for creating ita_perp, ita_cross

														 // 4+2+2+1+1 *1.5 = 15 per thread. That is possibly as slow as having 24 per thread. 
														 // Thus putting some stuff in shared may speed up if there are spills.

	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	long const iVertex = threadsPerTileMajor*blockIdx.x + threadIdx.x; // only meaningful threadIdx.x < threadsPerTileMajor

	long const StartMinor = threadsPerTileMinor*blockIdx.x;
	long const StartMajor = threadsPerTileMajor*blockIdx.x;
	long const EndMinor = StartMinor + threadsPerTileMinor;
	long const EndMajor = StartMajor + threadsPerTileMajor;

	f64_vec2 opppos, prevpos, nextpos;
	f64 nu, ita_par;  // optimization: we always each loop want to get rid of omega, nu once we have calc'd these, if possible!!

	shared_pos[threadIdx.x] = p_info_minor[iMinor].pos;
	shared_B[threadIdx.x] = p_B_minor[iMinor].xypart();
	shared_ita_par[threadIdx.x] = p_ita_parallel_ion_minor[iMinor];
	shared_nu[threadIdx.x] = p_nu_ion_minor[iMinor];

	// Perhaps the real answer is this. Advection and therefore advective momflux
	// do not need to be recalculated very often at all. At 1e6 cm/s, we aim for 1 micron,
	// get 1e-10s to actually do the advection !!
	// So an outer cycle. Still limiting the number of total things in a minor tile. We might like 384 = 192*2.

	structural info;
	if (threadIdx.x < threadsPerTileMajor) {
		info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
		shared_pos_verts[threadIdx.x] = info.pos;
		shared_B_verts[threadIdx.x] = p_B_minor[iVertex + BEGINNING_OF_CENTRAL].xypart();
		if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST))
		{
		//	memcpy(&(shared_vie_verts[threadIdx.x]), &(p_vie_minor[iVertex + BEGINNING_OF_CENTRAL]), sizeof(v4));
			shared_ita_par_verts[threadIdx.x] = p_ita_parallel_ion_minor[iVertex + BEGINNING_OF_CENTRAL];
			shared_nu_verts[threadIdx.x] = p_nu_ion_minor[iVertex + BEGINNING_OF_CENTRAL];
			// But now I am going to set ita == 0 in OUTERMOST and agree never to look there because that's fairer than one-way traffic and I don't wanna handle OUTERMOST?
			// I mean, I could handle it, and do flows only if the flag does not come up OUTER_FRILL.
			// OK just do that.
		} else {
			// it still manages to coalesce, let's hope, because ShardModel is 13 doubles not 12.
		//	memset(&(shared_vie_verts[threadIdx.x]), 0, sizeof(v4));
			shared_ita_par_verts[threadIdx.x] = 0.0;
			shared_nu_verts[threadIdx.x] = 0.0;
		};
	};

	__syncthreads();

	// IONS FIRST:

	if (threadIdx.x < threadsPerTileMajor) {

		long izTri[MAXNEIGH_d];
		char szPBC[MAXNEIGH_d];
		short tri_len = info.neigh_len; // ?!

		if ((info.flag == DOMAIN_VERTEX))
			//|| (info.flag == OUTERMOST)) // !!!!!!!!!!!!!!!!
		{
			// We are losing energy if there is viscosity into OUTERMOST.
			memcpy(izTri, p_izTri + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(long));
			memcpy(szPBC, p_szPBC + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(char));

			f64_tens3 J; // Jacobean
			memset(&J, 0, sizeof(f64_tens3));
			//d_eps_x_by_d_vx = 1.0;
			J.xx = 1.0;
			J.yy = 1.0;
			J.zz = 1.0;
			// d_eps_z_by_d_viz = 1.0;  // Note that eps includes v_k+1
			
			short iprev = tri_len - 1;
			short i = 0;
			short inext;
			if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMinor))
			{
				prevpos = shared_pos[izTri[iprev] - StartMinor];
			}
			else {
				prevpos = p_info_minor[izTri[iprev]].pos;
			}
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
			}

			if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
			{
				opppos = shared_pos[izTri[i] - StartMinor];
			}
			else {
				opppos = p_info_minor[izTri[i]].pos;
			}
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
			}

			f64_vec2 endpt0 = THIRD * (info.pos + opppos + prevpos);
			f64_vec3 omega_ci;

			// ** Be especially vigilant to the changes we need to make to go from ion to electron.
#pragma unroll 
			for (i = 0; i < tri_len; i++)
			{
				// Tri 0 is anticlockwise of neighbour 0, we think
				inext = i + 1; if (inext >= tri_len) inext = 0;

				// Now sort out anticlock vars:
				if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
				{
					nextpos = shared_pos[izTri[inext] - StartMinor];
				}
				else {
					nextpos = p_info_minor[izTri[inext]].pos;
				}
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
				}

				// Order of calculations may help things to go out/into scope at the right times so careful with that.
				
				f64 area_quadrilateral = 0.5*(
					(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
					+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
					+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
					+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
					);

				//gradvy.y = -0.5*(
				//	(our_v.vxy.y + next_v.vxy.y)*(info.pos.x - nextpos.x)
				//	+ (prev_v.vxy.y + our_v.vxy.y)*(prevpos.x - info.pos.x)
				//	+ (opp_v.vxy.y + prev_v.vxy.y)*(opppos.x - prevpos.x)
				//	+ (next_v.vxy.y + opp_v.vxy.y)*(nextpos.x - opppos.x) // nextpos = pos_anti, assumed
				//	) / area_quadrilateral;
				//
				// so we want to know, eps += U v_self for U 4x4
				
				f64 grad_vjdx_coeff_on_vj_self = 0.5*(prevpos.y - nextpos.y) / area_quadrilateral;
				f64 grad_vjdy_coeff_on_vj_self = 0.5*(nextpos.x - prevpos.x) / area_quadrilateral;
				
				f64_vec2 endpt1 = THIRD * (nextpos + info.pos + opppos);
				{
					f64_vec2 opp_B;
					if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
					{
						opp_B = shared_B[izTri[i] - StartMinor];
						if (shared_ita_par_verts[threadIdx.x] < shared_ita_par[izTri[i] - StartMinor])
						{
							ita_par = shared_ita_par_verts[threadIdx.x];
							nu = shared_nu_verts[threadIdx.x];
						} else {
							ita_par = shared_ita_par[izTri[i] - StartMinor];
							nu = shared_nu[izTri[i] - StartMinor];
						};
					} else {
						opp_B = p_B_minor[izTri[i]].xypart();
						f64 ita_theirs = p_ita_parallel_ion_minor[izTri[i]];
						f64 nu_theirs = p_nu_ion_minor[izTri[i]];
						if (shared_ita_par_verts[threadIdx.x] < ita_theirs) {
							ita_par = shared_ita_par_verts[threadIdx.x];
							nu = shared_nu_verts[threadIdx.x];
						} else {
							ita_par = ita_theirs;
							nu = nu_theirs;
						};
					}
					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opp_B = Clockwise_d*opp_B;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opp_B = Anticlockwise_d*opp_B;
					}
					omega_ci = 0.5*qoverMc*(Make3(opp_B + shared_B_verts[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
				} // Guaranteed DOMAIN_VERTEX never needs to skip an edge; we include CROSSING_INS in viscosity.

				f64_vec2 edge_normal;
				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

				Augment_Jacobean(&J, 
					hsub/(p_n_minor[iVertex + BEGINNING_OF_CENTRAL].n*p_AreaMinor[iVertex + BEGINNING_OF_CENTRAL]*m_ion),
					edge_normal, ita_par, nu, omega_ci,
					grad_vjdx_coeff_on_vj_self,
					grad_vjdy_coeff_on_vj_self
				);

				endpt0 = endpt1;
				prevpos = opppos;
				opppos = nextpos;
			}; // next i
			
			f64_tens3 result;
			J.Inverse(result);

			memcpy(&(p_matrix_i[iVertex + BEGINNING_OF_CENTRAL]), &result, sizeof(f64_tens3));
			  // inverted it so that we are ready to put Jacobi = result.eps
			
		} else {
			// NOT domain vertex: Do nothing			

			// NOTE: We did not include OUTERMOST. Justification / effect ??
		};
	};
	// __syncthreads(); // end of first vertex part
	// Do we need syncthreads? Not overwriting any shared data here...

	info = p_info_minor[iMinor];

	//if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL)) {
	{
		long izNeighMinor[6];
		char szPBC[6];

		if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS)) {

			memcpy(izNeighMinor, p_izNeighMinor + iMinor * 6, sizeof(long) * 6);
			memcpy(szPBC, p_szPBCtriminor + iMinor * 6, sizeof(char) * 6);
			
			f64_tens3 J; // Jacobean
			memset(&J, 0, sizeof(f64_tens3));
			//d_eps_x_by_d_vx = 1.0;
			J.xx = 1.0;
			J.yy = 1.0;
			J.zz = 1.0;
			
			short inext, iprev = 5, i = 0;
			if ((izNeighMinor[iprev] >= StartMinor) && (izNeighMinor[iprev] < EndMinor))
			{
				prevpos = shared_pos[izNeighMinor[iprev] - StartMinor];
			}
			else {
				if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
				} else {
					prevpos = p_info_minor[izNeighMinor[iprev]].pos;
				};
			};
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
			}

			i = 0;
			if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
			{
				opppos = shared_pos[izNeighMinor[i] - StartMinor];
			}
			else {
				if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					opppos = shared_pos_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					opppos = p_info_minor[izNeighMinor[i]].pos;
				};
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
			}

			f64_vec2 endpt0 = THIRD * (info.pos + opppos + prevpos);
			f64_vec3 omega_ci;
			// Let's make life easier and load up an array of 6 n's beforehand.
#pragma unroll 
			for (i = 0; i < 6; i++)
			{
				inext = i + 1; if (inext > 5) inext = 0;

				if ((izNeighMinor[inext] >= StartMinor) && (izNeighMinor[inext] < EndMinor))
				{
					nextpos = shared_pos[izNeighMinor[inext] - StartMinor];
				}
				else {
					if ((izNeighMinor[inext] >= StartMajor + BEGINNING_OF_CENTRAL) &&
						(izNeighMinor[inext] < EndMajor + BEGINNING_OF_CENTRAL))
					{
						nextpos = shared_pos_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
					}
					else {
						nextpos = p_info_minor[izNeighMinor[inext]].pos;
					};
				};
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
				}

				//	nu = 1.0e10; // DEBUG
				bool bUsableSide = true;
				{
					f64_vec2 opp_B(0.0, 0.0);
					
					if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
					{
						opp_B = shared_B[izNeighMinor[i] - StartMinor];
						if (shared_ita_par[threadIdx.x] < shared_ita_par[izNeighMinor[i] - StartMinor])
						{
							ita_par = shared_ita_par[threadIdx.x];
							nu = shared_nu[threadIdx.x];
						}
						else {
							ita_par = shared_ita_par[izNeighMinor[i] - StartMinor];
							nu = shared_nu[izNeighMinor[i] - StartMinor];
						};

						if (shared_ita_par[izNeighMinor[i] - StartMinor] == 0.0) bUsableSide = false;
					}
					else {
						if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
							(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
						{
							opp_B = shared_B_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
							if (shared_ita_par[threadIdx.x] < shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL])
							{
								ita_par = shared_ita_par[threadIdx.x];
								nu = shared_nu[threadIdx.x];
							}
							else {
								ita_par = shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
								nu = shared_nu_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
							};

							if (shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL] == 0.0) bUsableSide = false;
						}
						else {
							opp_B = p_B_minor[izNeighMinor[i]].xypart();
							f64 ita_par_opp = p_ita_parallel_ion_minor[izNeighMinor[i]];
							f64 nu_theirs = p_nu_ion_minor[izNeighMinor[i]];
							if (shared_ita_par[threadIdx.x] < ita_par_opp) {
								ita_par = shared_ita_par[threadIdx.x];
								nu = shared_nu[threadIdx.x]; // why do I deliberately use the corresponding nu? nvm
							}
							else {
								ita_par = ita_par_opp;
								nu = nu_theirs; // Did I know we were doing this? We use the MINIMUM ita ?

								// . We should probably stop that.
							}

							if (ita_par_opp == 0.0) bUsableSide = false;
						}
					}
					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opp_B = Clockwise_d*opp_B;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opp_B = Anticlockwise_d*opp_B;
					}
					omega_ci = 0.5*qoverMc*(Make3(opp_B + shared_B[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
				}

				f64_vec2 endpt1 = THIRD * (nextpos + info.pos + opppos);
				
				f64_vec2 edge_normal;
				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;
				
				if (bUsableSide) {

					// New definition of endpoint of minor edge:

					f64 area_quadrilateral = 0.5*(
						(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
						+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
						+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
						+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
						);

					f64 grad_vjdx_coeff_on_vj_self = 0.5*(prevpos.y - nextpos.y) / area_quadrilateral;
					f64 grad_vjdy_coeff_on_vj_self = 0.5*(nextpos.x - prevpos.x) / area_quadrilateral;

					Augment_Jacobean(&J,
						hsub / (p_n_minor[iMinor].n * p_AreaMinor[iMinor] * m_ion),						
						edge_normal, ita_par, nu, omega_ci,
						grad_vjdx_coeff_on_vj_self,
						grad_vjdy_coeff_on_vj_self
					);
				}

				endpt0 = endpt1;
				prevpos = opppos;
				opppos = nextpos;
			};
			
			f64_tens3 result;
			J.Inverse(result);
			memcpy(&(p_matrix_i[iMinor]), &result, sizeof(f64_tens3));
			
		} else {
			// Not domain tri or crossing_ins
			// Did we fairly model the insulator as a reflection of v?
		}
	} // scope

	__syncthreads();

	// Now do electron: overwrite ita and nu, copy-paste the above codes very carefully
	shared_ita_par[threadIdx.x] = p_ita_parallel_elec_minor[iMinor];
	shared_nu[threadIdx.x] = p_nu_elec_minor[iMinor];

	if (threadIdx.x < threadsPerTileMajor) {
		info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];

		if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST))  // keeping consistent with ion above where we did put OUTERMOST here
		{// but we set ita to 0 in the pre routine for outermost.
			shared_ita_par_verts[threadIdx.x] = p_ita_parallel_elec_minor[iVertex + BEGINNING_OF_CENTRAL];
			shared_nu_verts[threadIdx.x] = p_nu_elec_minor[iVertex + BEGINNING_OF_CENTRAL];
		}
		else {
			shared_ita_par_verts[threadIdx.x] = 0.0;
			shared_nu_verts[threadIdx.x] = 0.0;
		};
	};

	__syncthreads();

	if (threadIdx.x < threadsPerTileMajor) {
		
		long izTri[MAXNEIGH];
		char szPBC[MAXNEIGH];
		short tri_len = info.neigh_len; // ?!
		
		if ((info.flag == DOMAIN_VERTEX))
			//|| (info.flag == OUTERMOST)) 
		{
			memcpy(izTri, p_izTri + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(long));
			memcpy(szPBC, p_szPBC + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(char));

			f64_tens3 J; // Jacobean
			memset(&J, 0, sizeof(f64_tens3));
			//d_eps_x_by_d_vx = 1.0;
			J.xx = 1.0;
			J.yy = 1.0;
			J.zz = 1.0;

			short iprev = tri_len - 1;
			short i = 0;
			short inext;
			if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMinor))
			{
				prevpos = shared_pos[izTri[iprev] - StartMinor];
			}
			else {
				prevpos = p_info_minor[izTri[iprev]].pos;
			}
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
			}

			if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
			{
				opppos = shared_pos[izTri[i] - StartMinor];
			}
			else {
				opppos = p_info_minor[izTri[i]].pos;
			}
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
			}

			f64_vec2 endpt0 = THIRD * (info.pos + opppos + prevpos);
			f64_vec3 omega_ce;
#pragma unroll 
			for (i = 0; i < tri_len; i++)
			{
				// Tri 0 is anticlockwise of neighbour 0, we think
				inext = i + 1; if (inext >= tri_len) inext = 0;

				if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
				{
						nextpos = shared_pos[izTri[inext] - StartMinor];
				} else {
					nextpos = p_info_minor[izTri[inext]].pos;
				}
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
				}
				// All same as ion here:
			
				f64 area_quadrilateral = 0.5*(
					(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
					+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
					+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
					+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
					);

				f64 grad_vjdx_coeff_on_vj_self = 0.5*(prevpos.y - nextpos.y) / area_quadrilateral;
				f64 grad_vjdy_coeff_on_vj_self = 0.5*(nextpos.x - prevpos.x) / area_quadrilateral;

				{
					f64_vec2 opp_B;
					f64 opp_ita, opp_nu;
					if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
					{
						opp_B = shared_B[izTri[i] - StartMinor];
						opp_ita = shared_ita_par[izTri[i] - StartMinor];
						opp_nu = shared_nu[izTri[i] - StartMinor];
						//ita_par = 0.5*(shared_ita_par_verts[threadIdx.x] + shared_ita_par[izTri[i] - StartMinor]);
						//nu = 0.5*(shared_nu_verts[threadIdx.x] + shared_nu[izTri[i] - StartMinor]);
					}
					else {
						opp_B = p_B_minor[izTri[i]].xypart();
						opp_ita = p_ita_parallel_elec_minor[izTri[i]];
						opp_nu = p_nu_elec_minor[izTri[i]];
					}
					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opp_B = Clockwise_d*opp_B;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opp_B = Anticlockwise_d*opp_B;
					}
					if (shared_ita_par_verts[threadIdx.x] < opp_ita) {
						ita_par = shared_ita_par_verts[threadIdx.x];
						nu = shared_nu_verts[threadIdx.x];
					}
					else {
						ita_par = opp_ita;
						nu = opp_nu;
					}
					omega_ce = 0.5*qovermc*(Make3(opp_B + shared_B_verts[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
				}

				f64_vec2 endpt1 = THIRD * (nextpos + info.pos + opppos);
				f64_vec2 edge_normal;
				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

				Augment_Jacobean(&J,
					hsub / (p_n_minor[iVertex + BEGINNING_OF_CENTRAL].n * p_AreaMinor[iVertex + BEGINNING_OF_CENTRAL] * m_e),
					edge_normal, ita_par, nu, omega_ce,
					grad_vjdx_coeff_on_vj_self,
					grad_vjdy_coeff_on_vj_self
				);

				endpt0 = endpt1;
				prevpos = opppos;
				opppos = nextpos;
			}; // next i
			
			f64_tens3 result;
			J.Inverse(result);
			memcpy(&(p_matrix_e[iVertex + BEGINNING_OF_CENTRAL]), &result, sizeof(f64_tens3));
		} else {
			// NOT domain vertex: Do nothing			
		};
	};

	// Electrons in tris:
	info = p_info_minor[iMinor];
	long izNeighMinor[6];
	char szPBC[6];
	
	if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL)) {
	}
	else {
		if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS)) {

			memcpy(izNeighMinor, p_izNeighMinor + iMinor * 6, sizeof(long) * 6);
			memcpy(szPBC, p_szPBCtriminor + iMinor * 6, sizeof(char) * 6);
			
			f64_tens3 J; // Jacobean
			memset(&J, 0, sizeof(f64_tens3));
			//d_eps_x_by_d_vx = 1.0;
			J.xx = 1.0;
			J.yy = 1.0;
			J.zz = 1.0;
			
			short inext, iprev = 5, i = 0;
			if ((izNeighMinor[iprev] >= StartMinor) && (izNeighMinor[iprev] < EndMinor))
			{
				prevpos = shared_pos[izNeighMinor[iprev] - StartMinor];
			}
			else {
				if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					prevpos = p_info_minor[izNeighMinor[iprev]].pos;
				};
			};
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
			}

			i = 0;
			if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
			{
				opppos = shared_pos[izNeighMinor[i] - StartMinor];
			}
			else {
				if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					opppos = shared_pos_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					opppos = p_info_minor[izNeighMinor[i]].pos;
				};
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
			}

			f64_vec2 endpt0 = THIRD * (info.pos + opppos + prevpos);
			f64_vec3 omega_ce;
			// Let's make life easier and load up an array of 6 n's beforehand.
#pragma unroll 
			for (i = 0; i < 6; i++)
			{
				inext = i + 1; if (inext > 5) inext = 0;

				if ((izNeighMinor[inext] >= StartMinor) && (izNeighMinor[inext] < EndMinor))
				{
					nextpos = shared_pos[izNeighMinor[inext] - StartMinor];
				}
				else {
					if ((izNeighMinor[inext] >= StartMajor + BEGINNING_OF_CENTRAL) &&
						(izNeighMinor[inext] < EndMajor + BEGINNING_OF_CENTRAL))
					{
						nextpos = shared_pos_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
					}
					else {
						nextpos = p_info_minor[izNeighMinor[inext]].pos;
					};
				};
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
				}

				bool bUsableSide = true;
				{
					f64_vec2 opp_B;
					f64 opp_ita, opp_nu;
					if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
					{
						opp_B = shared_B[izNeighMinor[i] - StartMinor];
						opp_ita = shared_ita_par[izNeighMinor[i] - StartMinor];
						opp_nu = shared_nu[izNeighMinor[i] - StartMinor];
						if (shared_ita_par[izNeighMinor[i] - StartMinor] == 0.0) bUsableSide = false;
					}
					else {
						if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
							(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
						{
							opp_B = shared_B_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
							opp_ita = shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
							opp_nu = shared_nu_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
							if (shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL] == 0.0) bUsableSide = false;
						}
						else {
							opp_B = p_B_minor[izNeighMinor[i]].xypart();
							opp_ita = p_ita_parallel_elec_minor[izNeighMinor[i]];
							opp_nu = p_nu_elec_minor[izNeighMinor[i]];
							if (opp_ita == 0.0) bUsableSide = false;
						}
					}
					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opp_B = Clockwise_d*opp_B;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opp_B = Anticlockwise_d*opp_B;
					}
					if (shared_ita_par[threadIdx.x] < opp_ita) {
						ita_par = shared_ita_par[threadIdx.x];
						nu = shared_nu[threadIdx.x];
					}
					else {
						ita_par = opp_ita;
						nu = opp_nu;
					}
					omega_ce = 0.5*qovermc*(Make3(opp_B + shared_B[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
				}

				f64_vec2 endpt1 = THIRD * (nextpos + info.pos + opppos);
				f64_vec2 edge_normal;
				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

				if (bUsableSide) {
					// New definition of endpoint of minor edge:
					
					f64 area_quadrilateral = 0.5*(
						(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
						+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
						+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
						+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
						);

					f64 grad_vjdx_coeff_on_vj_self = 0.5*(prevpos.y - nextpos.y) / area_quadrilateral;
					f64 grad_vjdy_coeff_on_vj_self = 0.5*(nextpos.x - prevpos.x) / area_quadrilateral;

					Augment_Jacobean(&J,
						hsub / (p_n_minor[iMinor].n * p_AreaMinor[iMinor] * m_e),
						edge_normal, ita_par, nu, omega_ce,
						grad_vjdx_coeff_on_vj_self,
						grad_vjdy_coeff_on_vj_self
					);
				};

				endpt0 = endpt1;
				prevpos = opppos;
				opppos = nextpos;
			};

			f64_tens3 result;
			J.Inverse(result);
			memcpy(&(p_matrix_e[iMinor]), &result, sizeof(f64_tens3));
		}
		else {
			// Not domain, not crossing_ins, not a frill			
		} // non-domain tri
	}; // was it FRILL
}


/*
// CALL with less than 170 threads:
__global__ void kernelCalculateArray_ROCwrt_my_neighbours(
	f64 const h_use,
	structural * __restrict__ p_info_minor,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,
	long * __restrict__ izTri_verts,
	char * __restrict__ szPBCtri_verts,
	f64_vec2 * __restrict__ p_cc,

	nvals * __restrict__ p_n_major,
	f64_vec3 * __restrict__ p_B_major,

	f64 * __restrict__ p_kappa_n,
	f64 * __restrict__ p_kappa_i,
	f64 * __restrict__ p_kappa_e,

	f64 * __restrict__ p_nu_i,
	f64 * __restrict__ p_nu_e,
	f64 * __restrict__ p_AreaMajor,

	// Output:
	f64 * __restrict__ D_eps_by_dx_neigh_n, // save an array of MAXNEIGH f64 values at this location
	f64 * __restrict__ D_eps_by_dx_neigh_i,
	f64 * __restrict__ D_eps_by_dx_neigh_e,
	f64 * __restrict__ Effect_self_n,
	f64 * __restrict__ Effect_self_i,
	f64 * __restrict__ Effect_self_e
) {
	// ******************************************
	//  1. Do this with kappa given from t_k etc
	// ******************************************
	// Then come and change it.

	// Think we might as well take kappa_par and nu from triangles really.
	// If n is modelled well then hopefully a nearby high-n does not have a big impact.

	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajor]; // 2
	__shared__ f64_vec2 shared_pos[2 * threadsPerTileMajor]; 
	// Need why? If using centroid? Sometimes on boundary - is that only reason?
	// Seems like a waste of 2 doubles. Can just average -- and shift if we load the flag that tells us to.

	// ?????????????????

	// Yes, very much a waste. The edge positions should be calculated from the vertex positions, we can
	// load flags to determine if it is an insulator-crossing triangle and that is the proper way to handle that.
	// Ah -- but we are using circumcenters, which are a bugger to calculate, no? Still might be better.
	// I think that is helping avoid dodgy situations.


	// Not needed:
	// __shared__ f64 shared_x[threadsPerTileMajor];
	
	__shared__ f64_vec2 shared_B[threadsPerTileMajor]; // +2
															 // B is smooth. Unfortunately we have not fitted in Bz here.
															 // In order to do that perhaps rewrite so that variables are overwritten in shared.
															 // This way is easier for NOW.
															 // So that's the optimization we want. Scalar arrays for T, nu.

	__shared__ f64 shared_kappa[threadsPerTileMajor * 2];
	__shared__ f64 shared_nu[threadsPerTileMajor * 2];

	__shared__ long Indexneigh[MAXNEIGH_d*threadsPerTileMajor]; // assume 48 bytes = 4*12 = 6 doubles

	__shared__ char PBCneigh[MAXNEIGH_d*threadsPerTileMajor]; // 12 bytes each from L1. Have 42 per thread at 384 threads.
																	// We should at any rate try a major block of size 256. If 1 block will run at a time, so be it.
																	// Leave L1 in case of register overflow into it. <-- don't know how likely - do we only have 31 doubles in registry
																	// regardless # of threads and space? Or can be 63?
	__shared__ char PBCtri[MAXNEIGH_d*threadsPerTileMajor];
	// Balance of shared vs L1: 24*256*8 = 48K. That leaves 8 doublesworth in L1 for variables.

	// 6+2+4 + 6+3 = 12 + 9 = 21
	// Could profitably, in the other routines, swap izTri into shared and PBCtri,PBCneigh into L1.

	__shared__ long izTri[MAXNEIGH_d*threadsPerTileMajor];  // 21+6 = 27 which means we should be moving PBC arrays into L1, to run bigger tile or 2 in parallel.

	// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	// I'm just going to drop izTri entirely and load it from global memory each time. It's a lot of memory to store.
	// That fails for unknown reason. We can verify that the iTri we get is IDENTICAL and it still fails when we access p_cc[iTri].
	// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

	// In the other routines -- maybe limit it to 4 things at a time, and stick in shared too.
	// We REALLY DON'T WANT MUCH IN L1.
	
	// Storing f64[MAXNEIGH_d] is what, 12*8 = 96 bytes/thread. 16K -> 170 threads max
	// 16*1024/(48+96) = 113 so we have no choice: in order to fit more in L1 we MUST put more into shared.

	// OK .. or we could just keep say 6 at a time in memory since that is the width of a busload anyway
	// Therefore break it into 6 and then the rest.
	// Tricky though.
	
	f64 Results[MAXNEIGH_d]; // Make sure we call this with less than 170 threads
	
							 // It would help matters if we get rid of T3. We might as well therefore change to scalar flatpack T.
							 // We are hoping that it works well loading kappa(tri) and that this is not upset by nearby values. Obviously a bit of an experiment.

							 // Does make it somewhat laughable that we go to such efforts to reduce global accesses when we end up overflowing anyway. 
							 // If we can fit 24 doubles/thread in 48K that means we can fit 8 doubles/thread in 16K so that's most of L1 used up.
	
	long const StartMajor = blockIdx.x*blockDim.x;
	long const EndMajor = StartMajor + blockDim.x;
	long const StartMinor = blockIdx.x*blockDim.x * 2;
	long const EndMinor = StartMinor + blockDim.x * 2;
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX	// 2.5 double

	structural info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];     // 3 double
	shared_pos_verts[threadIdx.x] = info.pos;
#ifdef CENTROID_HEATCONDUCTION
	{
		structural infotemp[2];
		memcpy(infotemp, p_info_minor + 2 * iVertex, 2 * sizeof(structural));
		shared_pos[threadIdx.x * 2] = infotemp[0].pos;
		shared_pos[threadIdx.x * 2 + 1] = infotemp[1].pos;
		memcpy(&(shared_kappa[threadIdx.x * 2]), p_kappa_n + 2 * iVertex, 2 * sizeof(f64));
	}
#else
	{
		memcpy(&(shared_pos[threadIdx.x * 2]), p_cc + 2 * iVertex, 2 * sizeof(f64_vec2));
		memcpy(&(shared_kappa[threadIdx.x * 2]), p_kappa_n + 2 * iVertex, 2 * sizeof(f64));
	}
#endif

	// No nu to set for neutrals - not used

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {
		shared_B[threadIdx.x] = p_B_major[iVertex].xypart();
	} else {
		// SHOULD NOT BE LOOKING INTO INS. How do we avoid?
		memset(&(shared_B[threadIdx.x]), 0, sizeof(f64_vec2));
	}

	__syncthreads();
	// works if we cut here

	memset(Results, 0, sizeof(f64)*MAXNEIGH_d); 
	f64 Results_self = 1.0;// d own epsilon by own x includes T_k+1 at the start of formula
	f64_vec2 pos_clock, pos_anti, pos_out;   // +6
	f64_vec2 B_out;       // +2
						  // NTrates ourrates;      // +5
	f64 kappa_parallel;
	f64 nu;                // 20 there  
	f64_vec2 edge_normal;  // 22
	f64_vec2 endpt_anti;    // 24 .. + 6 from above
	long indexneigh;     // into the 2-double buffer in L1
	f64_vec2 endpt_clock;    // As we only use endpt_anti afterwords we could union endpt_clock with edge_normal
							 // Come back and optimize by checking which things we need in scope at the same time?
	long iTri;
	short iNeigh; // only fixed # of addresses so short makes no difference.
	char PBC; // char makes no difference.

	if ((info.flag == INNERMOST) || (info.flag == OUTERMOST))
	{
		// [ Ignore flux into edge of outermost vertex I guess ???]
	}
	else {
		if (info.flag == DOMAIN_VERTEX) {

			// The idea of not sending blocks full of non-domain vertices is another idea. Fiddly with indices.
	
			memcpy(Indexneigh + MAXNEIGH_d * threadIdx.x,
				pIndexNeigh + MAXNEIGH_d * iVertex,
				MAXNEIGH_d * sizeof(long));
			memcpy(PBCneigh + MAXNEIGH_d * threadIdx.x,
				pPBCNeigh + MAXNEIGH_d * iVertex,
				MAXNEIGH_d * sizeof(char));
			memcpy(PBCtri + MAXNEIGH_d * threadIdx.x,
				szPBCtri_verts + MAXNEIGH_d * iVertex,
				MAXNEIGH_d * sizeof(char));

			memcpy(izTri + MAXNEIGH_d*threadIdx.x,
				izTri_verts + MAXNEIGH_d*iVertex,
				MAXNEIGH_d * sizeof(long));

		//	memcpy(izTri, //+ MAXNEIGH_d * threadIdx.x,
		//		izTri_verts + MAXNEIGH_d * iVertex, MAXNEIGH_d * sizeof(long));

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_clock = shared_pos_verts[indexneigh - StartMajor];
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_clock = info2.pos;
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == NEEDS_ANTI) { pos_clock = Anticlock_rotate2(pos_clock); };
			if (PBC == NEEDS_CLOCK) { pos_clock = Clockwise_rotate2(pos_clock); };

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_out = shared_pos_verts[indexneigh - StartMajor];
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_out = info2.pos;
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
			if (PBC == NEEDS_ANTI) { pos_out = Anticlock_rotate2(pos_out); };
			if (PBC == NEEDS_CLOCK) { pos_out = Clockwise_rotate2(pos_out); };
			
			if ((info.neigh_len < 1) || (info.neigh_len >= MAXNEIGH_d)) printf("info.neigh_len %d \n", info.neigh_len); // debug
			//iTri = izTri_verts[MAXNEIGH_d*iVertex + info.neigh_len - 1];
			iTri = izTri[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
	//		long iTri2 = izTri_verts[MAXNEIGH_d*iVertex + info.neigh_len - 1];
	//		if (iTri != iTri2) printf("iTri %d iTri2 %d \n", iTri, iTri2);
	//		printf("iTri %d iTri2 %d || ", iTri, iTri2);
	//		if ((iTri2 >= 73728) || (iTri2 < 0)) printf("iTri = %d info.neigh_len %d iVertex %d \n", iTri, info.neigh_len, iVertex); // debug
	//		endpt_clock = p_cc[iTri]; // DEBUG
		// endpt_clock = p_cc[iTri2]; // this alone will break it? yes.
		// Note that it crashes without *anything* being printed out. It malcompiles the kernel.
		// That's a real problem -- we can't seem to fix it and it's very very basic.

			if ((iTri >= StartMinor) && (iTri < EndMinor))
			{
				endpt_clock = shared_pos[iTri - StartMinor];
			}
			else {
#ifdef CENTROID_HEATCONDUCTION
				endpt_clock = p_info_minor[iTri].pos;
#else
				endpt_clock = p_cc[iTri];
#endif
			}
			PBC = PBCtri[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == ROTATE_ME_CLOCKWISE) endpt_clock = Clockwise_d * endpt_clock;
			if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_clock = Anticlockwise_d * endpt_clock;
			
			f64 Nn = p_AreaMajor[iVertex] * p_n_major[iVertex].n_n;
			
			// somewhere in the above section is something that stops it running
			

#pragma unroll MAXNEIGH_d
			for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
			{
				{
					short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNext];
				}
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					pos_anti = shared_pos_verts[indexneigh - StartMajor];
				}
				else {
					structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
					pos_anti = info2.pos;
				};
				if (PBC == NEEDS_ANTI) {
					pos_anti = Anticlock_rotate2(pos_anti);
				};
				if (PBC == NEEDS_CLOCK) {
					pos_anti = Clockwise_rotate2(pos_anti);
				};

				   // Now let's see
				   // tri 0 has neighs 0 and 1 I'm pretty sure (check....) CHECK
				iTri = izTri[MAXNEIGH_d*threadIdx.x + iNeigh];// izTri_verts[MAXNEIGH_d*iVertex + iNeigh];
				if ((iTri >= StartMinor) && (iTri < EndMinor))
				{
					endpt_anti = shared_pos[iTri - StartMinor];
				}
				else {
#ifdef CENTROID_HEATCONDUCTION
					endpt_anti = p_info_minor[iTri].pos;
#else
					endpt_anti = p_cc[iTri];
#endif
				}
				PBC = PBCtri[MAXNEIGH_d*threadIdx.x + iNeigh];
				if (PBC == ROTATE_ME_CLOCKWISE) endpt_anti = Clockwise_d * endpt_anti;
				if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_anti = Anticlockwise_d * endpt_anti;

				edge_normal.x = (endpt_anti.y - endpt_clock.y);
				edge_normal.y = (endpt_clock.x - endpt_anti.x);

				// SMARTY:
				if (pos_out.x*pos_out.x + pos_out.y*pos_out.y >
					DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					f64 Area_quadrilateral = 0.5*(
						(info.pos.x + pos_anti.x)*(info.pos.y - pos_anti.y)
						+ (pos_clock.x + info.pos.x)*(pos_clock.y - info.pos.y)
						+ (pos_out.x + pos_clock.x)*(pos_out.y - pos_clock.y)
						+ (pos_anti.x + pos_out.x)*(pos_anti.y - pos_out.y)
						);
					short iPrev = iNeigh - 1; if (iPrev < 0) iPrev = info.neigh_len - 1;
					
					kappa_parallel = 0.0;
					if ((iTri >= StartMinor) && (iTri < EndMinor))
					{
						kappa_parallel = 0.5*shared_kappa[iTri - StartMinor];
					}
					else {
						kappa_parallel = 0.5*p_kappa_n[iTri];
					};
					
					{
						iTri = izTri[MAXNEIGH_d*threadIdx.x + iPrev];
						if ((iTri >= StartMinor) && (iTri < EndMinor))
						{
							kappa_parallel += 0.5*shared_kappa[iTri - StartMinor];
						}
						else {
							kappa_parallel += 0.5*p_kappa_n[iTri];
						}
					}
					// kappa_parallel = 1.0;
					// seems like heisenbug

					
					//grad_T.x = 0.5*(
					//	(shared_T[threadIdx.x] + T_anti)*(info.pos.y - pos_anti.y)
					//	+ (T_clock + shared_T[threadIdx.x])*(pos_clock.y - info.pos.y)
					//	+ (T_out + T_clock)*(pos_out.y - pos_clock.y)
					//	+ (T_anti + T_out)*(pos_anti.y - pos_out.y)
					//	) / Area_quadrilateral;
					//grad_T.y = -0.5*( // notice minus
					//	(shared_T[threadIdx.x] + T_anti)*(info.pos.x - pos_anti.x)
					//	+ (T_clock + shared_T[threadIdx.x])*(pos_clock.x - info.pos.x)
					//	+ (T_out + T_clock)*(pos_out.x - pos_clock.x)
					//	+ (T_anti + T_out)*(pos_anti.x - pos_out.x)
					//	) / Area_quadrilateral;
					//ourrates.NnTn += TWOTHIRDS * kappa_parallel * grad_T.dot(edge_normal);

					f64 factor; // see if this fixes it -- did?
					if (Area_quadrilateral*Nn == 0.0) {
						factor = 1.0;
						printf("\niVertex %d getting 0 = Nn !!!!\n\n", iVertex);
						// Yet we do not hit this -- so heisenbug
					}
					else {
						factor = (TWOTHIRDS*kappa_parallel / Area_quadrilateral)*(-h_use / Nn);
					};

					// works with some of this commented out! :

					// really, d_NTrates_by_dT :
					//f64 d_NT_by_dT_clock = (TWOTHIRDS*kappa_parallel / Area_quadrilateral)*
															
					Results[iPrev] += factor*
						(0.5*edge_normal.x*(pos_out.y - info.pos.y)- 0.5* edge_normal.y* (pos_out.x - info.pos.x));
					//f64 d_NT_by_dT_opp = (TWOTHIRDS*kappa_parallel / Area_quadrilateral)*
					Results[iNeigh] += factor*
						(0.5*edge_normal.x*(pos_anti.y - pos_clock.y)- 0.5* edge_normal.y*(pos_anti.x - pos_clock.x));
				//	f64 d_NT_by_dT_anti = (TWOTHIRDS*kappa_parallel / Area_quadrilateral)*
				//		(0.5*edge_normal.x* (info.pos.y - pos_out.y)
				//			- 0.5*edge_normal.y* (info.pos.x - pos_out.x));

					// dies if you put it here

					short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
					Results[iNext] += factor*
						(0.5*edge_normal.x* (info.pos.y - pos_out.y) - 0.5*edge_normal.y* (info.pos.x - pos_out.x));
					
					// f64 d_NT_by_dT_own = (TWOTHIRDS*kappa_parallel / Area_quadrilateral)*
					Results_self += factor*
						(0.5*edge_normal.x*(pos_clock.y - pos_anti.y) - 0.5*edge_normal.y*(pos_clock.x - pos_anti.x));

					// eps involves factor Nn: T_k+1 - [ T_k + h dNnTn/dt / Nn ]
					// Needed to collect deps/d T_clock, deps/d T_opp, deps/d T_anti, deps/d T_own
					
					// 	f64 epsilon_e = p_T_putative[iVertex].Te - p_T_k[iVertex].Te - (h_sub / N)*Rates.NeTe;
					// Note the minus so again it looks like we got the sign right:
					
				} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

				  // Now go round:	
				endpt_clock = endpt_anti;
				pos_clock = pos_out;
				pos_out = pos_anti;
			}; // next iNeigh
			

			memcpy(&(D_eps_by_dx_neigh_n[iVertex*MAXNEIGH_d]), Results, sizeof(f64)*MAXNEIGH_d);
		
			Effect_self_n[iVertex] = Results_self;
			
		}; // was it DOMAIN_VERTEX? Do what otherwise?

	}; // was it OUTERMOST/INNERMOST?

	__syncthreads();
	
	// Did we make sure to include a call to syncthreads every time we carried on to update shared memory data in every other routine?
	// ##################################################################################################################################
	
	{
		memcpy(&(shared_kappa[threadIdx.x * 2]), p_kappa_i + 2 * iVertex, 2 * sizeof(f64));
		memcpy(&(shared_nu[threadIdx.x * 2]), p_nu_i + 2 * iVertex, 2 * sizeof(f64));
	}
	
	__syncthreads();

	memset(Results, 0, sizeof(f64)*MAXNEIGH_d);
	Results_self = 1.0; // d own epsilon by own x includes T_k+1 at the start of formula

	if ((info.flag == INNERMOST) || (info.flag == OUTERMOST))
	{
		// [ Ignore flux into edge of outermost vertex I guess ???]
	}
	else {
		if (info.flag == DOMAIN_VERTEX) {
			
			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_clock = shared_pos_verts[indexneigh - StartMajor];
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_clock = info2.pos;
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == NEEDS_ANTI) {
				pos_clock = Anticlock_rotate2(pos_clock);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_clock = Clockwise_rotate2(pos_clock);
			};

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_out = shared_pos_verts[indexneigh - StartMajor];
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_out = info2.pos;
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
			if (PBC == NEEDS_ANTI) {
				pos_out = Anticlock_rotate2(pos_out);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_out = Clockwise_rotate2(pos_out);
			};

			iTri = izTri[threadIdx.x*MAXNEIGH_d + info.neigh_len - 1];
			if ((iTri >= StartMinor) && (iTri < EndMinor))
			{
				endpt_clock = shared_pos[iTri - StartMinor];
			}
			else {
#ifdef CENTROID_HEATCONDUCTION
				endpt_clock = p_info_minor[iTri].pos;
#else
				endpt_clock = p_cc[iTri];
#endif
			}
			PBC = PBCtri[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == ROTATE_ME_CLOCKWISE) endpt_clock = Clockwise_d * endpt_clock;
			if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_clock = Anticlockwise_d * endpt_clock;

			f64 N = p_AreaMajor[iVertex] * p_n_major[iVertex].n;

#pragma unroll MAXNEIGH_d
			for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
			{
				{
					short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNext];
				}
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					pos_anti = shared_pos_verts[indexneigh - StartMajor];
				}
				else {
					structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
					pos_anti = info2.pos;
				};
				if (PBC == NEEDS_ANTI) {
					pos_anti = Anticlock_rotate2(pos_anti);
				};
				if (PBC == NEEDS_CLOCK) {
					pos_anti = Clockwise_rotate2(pos_anti);
				};

				iTri = izTri[threadIdx.x*MAXNEIGH_d + iNeigh];
				if ((iTri >= StartMinor) && (iTri < EndMinor))
				{
					endpt_anti = shared_pos[iTri - StartMinor];
				}
				else {
#ifdef CENTROID_HEATCONDUCTION
					endpt_anti = p_info_minor[iTri].pos;
#else
					endpt_anti = p_cc[iTri];
#endif					
				}
				PBC = PBCtri[MAXNEIGH_d*threadIdx.x + iNeigh];
				if (PBC == ROTATE_ME_CLOCKWISE) endpt_anti = Clockwise_d * endpt_anti;
				if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_anti = Anticlockwise_d * endpt_anti;

				edge_normal.x = (endpt_anti.y - endpt_clock.y);
				edge_normal.y = (endpt_clock.x - endpt_anti.x);

				// SMARTY:
				if (pos_out.x*pos_out.x + pos_out.y*pos_out.y >
					DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					f64 Area_quadrilateral = 0.5*(
						(info.pos.x + pos_anti.x)*(info.pos.y - pos_anti.y)
						+ (pos_clock.x + info.pos.x)*(pos_clock.y - info.pos.y)
						+ (pos_out.x + pos_clock.x)*(pos_out.y - pos_clock.y)
						+ (pos_anti.x + pos_out.x)*(pos_anti.y - pos_out.y)
						);
					kappa_parallel = 0.0;
					f64 nu;
					if ((iTri >= StartMinor) && (iTri < EndMinor))
					{
						kappa_parallel = 0.5*shared_kappa[iTri - StartMinor];
						nu = 0.5*shared_nu[iTri - StartMinor];
					}
					else {
						kappa_parallel = 0.5*p_kappa_i[iTri];
						nu = 0.5*p_nu_i[iTri];
					};
					short iPrev = iNeigh - 1; if (iPrev < 0) iPrev = info.neigh_len - 1;
					iTri = izTri[threadIdx.x*MAXNEIGH_d + iPrev];
					{
						if ((iTri >= StartMinor) && (iTri < EndMinor))
						{
							kappa_parallel += 0.5*shared_kappa[iTri - StartMinor];
							nu += 0.5*shared_nu[iTri - StartMinor];
						} else {
							kappa_parallel += 0.5*p_kappa_i[iTri];
							nu += 0.5*p_nu_i[iTri];
						}
					}

					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						B_out = shared_B[indexneigh - StartMajor];
					} else {
						f64_vec3 B_out3 = p_B_major[indexneigh];
						B_out = B_out3.xypart();
					}
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if (PBC == NEEDS_ANTI) 	B_out = Anticlock_rotate2(B_out);
					if (PBC == NEEDS_CLOCK)	B_out = Clockwise_rotate2(B_out);

					//kappa.xx = kappa_parallel * (nu_eHeart*nu_eHeart + omega.x*omega.x) / (nu_eHeart * nu_eHeart + omega_sq);
					//kappa.xy = kappa_parallel * (omega.x*omega.y - nu_eHeart *omega.z) / (nu_eHeart * nu_eHeart + omega_sq);
					//kappa.yx = kappa_parallel * (omega.x*omega.y + nu_eHeart * omega.z) / (nu_eHeart * nu_eHeart + omega_sq);
					//kappa.yy = kappa_parallel * (omega.y*omega.y + nu_eHeart * nu_eHeart) / (nu_eHeart * nu_eHeart + omega_sq);

					{ // scoping brace
						f64_vec3 omega = Make3(qoverMc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qoverMc);

						// eps involves factor Nn: T_k+1 - [ T_k + h dNnTn/dt / Nn ]
						// Needed to collect deps/d T_clock, deps/d T_opp, deps/d T_anti, deps/d T_own

						f64 coeff_NT_on_dTbydx = TWOTHIRDS * kappa_parallel *(
							edge_normal.x*(nu*nu + omega.x*omega.x) + edge_normal.y*(omega.x*omega.y - nu * omega.z)
							) / (nu * nu + omega.dot(omega));

						f64 coeff_NT_on_dTbydy = TWOTHIRDS * kappa_parallel *(
							edge_normal.x*(omega.x*omega.y + nu * omega.z) + edge_normal.y*(omega.y*omega.y + nu * nu)
							) / (nu * nu + omega.dot(omega));

						if (N*Area_quadrilateral != 0.0) {
							coeff_NT_on_dTbydx *= (-h_use) / (N*Area_quadrilateral);
							coeff_NT_on_dTbydy *= (-h_use) / (N*Area_quadrilateral);
						} else {
							coeff_NT_on_dTbydx *= 0.0;
							coeff_NT_on_dTbydy *= 0.0;
							printf("alarm!! %d \n\n", iVertex);
						}
						 
						//f64 d_NT_by_dT_clock = 
						Results[iPrev] += (coeff_NT_on_dTbydx*0.5*(pos_out.y - info.pos.y)
							- coeff_NT_on_dTbydy*0.5*(pos_out.x - info.pos.x));

						Results[iNeigh] += (coeff_NT_on_dTbydx*0.5*(pos_anti.y - pos_clock.y)
							- coeff_NT_on_dTbydy*0.5*(pos_anti.x - pos_clock.x));

						short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
						Results[iNext] += (coeff_NT_on_dTbydx*0.5*(info.pos.y - pos_out.y)
							- coeff_NT_on_dTbydy*0.5*(info.pos.x - pos_out.x));

						Results_self += (coeff_NT_on_dTbydx*0.5*(pos_clock.y - pos_anti.y)
							- coeff_NT_on_dTbydy*0.5*(pos_clock.x - pos_anti.x));
//
//						d_eps_by_d_beta += (
//							d_NT_by_dT_clock*regressor_clock
//							+ d_NT_by_dT_opp*regressor_out
//							+ d_NT_by_dT_anti*regressor_anti
//							+ d_NT_by_dT_own*shared_x[threadIdx.x])*(-h_use / N);
//
						//	ourrates.NiTi += TWOTHIRDS * kappa_parallel *(
						//		edge_normal.x*((nu*nu + omega.x*omega.x)*grad_T.x + (omega.x*omega.y + nu * omega.z)*grad_T.y)
						//		+ edge_normal.y*((omega.x*omega.y - nu * omega.z)*grad_T.x + (omega.y*omega.y + nu * nu)*grad_T.y)
						//		) / (nu * nu + omega.dot(omega));

					}
				} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

				  // Now go round:	
				endpt_clock = endpt_anti;
				pos_clock = pos_out;
				pos_out = pos_anti;

			}; // next iNeigh

			memcpy(&(D_eps_by_dx_neigh_i[iVertex*MAXNEIGH_d]), Results, sizeof(f64)*MAXNEIGH_d);
			Effect_self_i[iVertex] = Results_self;

		}; // was it DOMAIN_VERTEX? Do what otherwise?
	}; // was it OUTERMOST/INNERMOST?


	__syncthreads();

	
	// Did we make sure to include a call to syncthreads every time we carried on to update shared memory data in every other routine?
	// ##################################################################################################################################

	{
		memcpy(&(shared_kappa[threadIdx.x * 2]), p_kappa_e + 2 * iVertex, 2 * sizeof(f64));
		memcpy(&(shared_nu[threadIdx.x * 2]), p_nu_e + 2 * iVertex, 2 * sizeof(f64));
	}
	
	__syncthreads();
	
	memset(Results, 0, sizeof(f64)*MAXNEIGH_d);
	Results_self = 1.0;
	// MAKE SURE WE GIVE IT A GOOD OLD BLAST TO ZERO BEFORE WE CALL THESE ROUTINES - TO HANDLE NONDOMAIN VERTICES

	if ((info.flag == INNERMOST) || (info.flag == OUTERMOST))
	{
		// [ Ignore flux into edge of outermost vertex I guess ???]
	}
	else {
		if (info.flag == DOMAIN_VERTEX) {
			
			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_clock = shared_pos_verts[indexneigh - StartMajor];
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_clock = info2.pos;
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == NEEDS_ANTI) {
				pos_clock = Anticlock_rotate2(pos_clock);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_clock = Clockwise_rotate2(pos_clock);
			};

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_out = shared_pos_verts[indexneigh - StartMajor];
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_out = info2.pos;
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
			if (PBC == NEEDS_ANTI) {
				pos_out = Anticlock_rotate2(pos_out);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_out = Clockwise_rotate2(pos_out);
			};

			iTri = izTri[threadIdx.x*MAXNEIGH_d + info.neigh_len - 1];
			if ((iTri >= StartMinor) && (iTri < EndMinor))
			{
				endpt_clock = shared_pos[iTri - StartMinor];
			}
			else {
#ifdef CENTROID_HEATCONDUCTION
				endpt_clock = p_info_minor[iTri].pos;
#else
				endpt_clock = p_cc[iTri];
#endif
			}
			PBC = PBCtri[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == ROTATE_ME_CLOCKWISE) endpt_clock = Clockwise_d * endpt_clock;
			if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_clock = Anticlockwise_d * endpt_clock;

			f64 N = p_AreaMajor[iVertex] * p_n_major[iVertex].n;

#pragma unroll MAXNEIGH_d
			for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
			{
				{
					short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNext];
				}
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					pos_anti = shared_pos_verts[indexneigh - StartMajor];
				}
				else {
					structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
					pos_anti = info2.pos;
				};
				if (PBC == NEEDS_ANTI) {
					pos_anti = Anticlock_rotate2(pos_anti);
				};
				if (PBC == NEEDS_CLOCK) {
					pos_anti = Clockwise_rotate2(pos_anti);
				};

				iTri = izTri[threadIdx.x*MAXNEIGH_d + iNeigh];
				if ((iTri >= StartMinor) && (iTri < EndMinor))
				{
					endpt_anti = shared_pos[iTri - StartMinor];

				}
				else {
#ifdef CENTROID_HEATCONDUCTION
					endpt_anti = p_info_minor[iTri].pos;
#else
					endpt_anti = p_cc[iTri];
#endif
				}
				PBC = PBCtri[MAXNEIGH_d*threadIdx.x + iNeigh];
				if (PBC == ROTATE_ME_CLOCKWISE) endpt_anti = Clockwise_d * endpt_anti;
				if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_anti = Anticlockwise_d * endpt_anti;

				// It decided to rotate something it shouldn't oughta. Rotated tri 23600 = tri 2 for 11582.

				edge_normal.x = (endpt_anti.y - endpt_clock.y);
				edge_normal.y = (endpt_clock.x - endpt_anti.x);

				// SMARTY:
				if (pos_out.x*pos_out.x + pos_out.y*pos_out.y >
					DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					f64 Area_quadrilateral = 0.5*(
						(info.pos.x + pos_anti.x)*(info.pos.y - pos_anti.y)
						+ (pos_clock.x + info.pos.x)*(pos_clock.y - info.pos.y)
						+ (pos_out.x + pos_clock.x)*(pos_out.y - pos_clock.y)
						+ (pos_anti.x + pos_out.x)*(pos_anti.y - pos_out.y)
						);

					kappa_parallel = 0.0;
					f64 nu;
					if ((iTri >= StartMinor) && (iTri < EndMinor))
					{
						kappa_parallel = 0.5*shared_kappa[iTri - StartMinor];
						nu = 0.5*shared_nu[iTri - StartMinor];
					}
					else {
						kappa_parallel = 0.5*p_kappa_e[iTri];
						nu = 0.5*p_nu_e[iTri];
					};
					short iPrev = iNeigh - 1; if (iPrev < 0) iPrev = info.neigh_len - 1;
					iTri = izTri[threadIdx.x*MAXNEIGH_d + iPrev];
					{
						if ((iTri >= StartMinor) && (iTri < EndMinor))
						{
							kappa_parallel += 0.5*shared_kappa[iTri - StartMinor];
							nu += 0.5*shared_nu[iTri - StartMinor];
						}
						else {
							kappa_parallel += 0.5*p_kappa_e[iTri];
							nu += 0.5*p_nu_e[iTri];
						}
					}

					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						B_out = shared_B[indexneigh - StartMajor];
					}
					else {
						f64_vec3 B_out3 = p_B_major[indexneigh];
						B_out = B_out3.xypart();
					}
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if (PBC == NEEDS_ANTI) 	B_out = Anticlock_rotate2(B_out);
					if (PBC == NEEDS_CLOCK)	B_out = Clockwise_rotate2(B_out);

					{ // scoping brace
						f64_vec3 omega = Make3(qovermc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qovermc);
						// PROBABLY ALWAYS SPILLED INTO GLOBAL -- WHAT CAN WE DO?

						f64 coeff_NT_on_dTbydx = TWOTHIRDS * kappa_parallel *(
							edge_normal.x*(nu*nu + omega.x*omega.x) + edge_normal.y*(omega.x*omega.y - nu * omega.z)
							) / (nu * nu + omega.dot(omega));

						f64 coeff_NT_on_dTbydy = TWOTHIRDS * kappa_parallel *(
							edge_normal.x*(omega.x*omega.y + nu * omega.z) + edge_normal.y*(omega.y*omega.y + nu * nu)
							) / (nu * nu + omega.dot(omega));

						if (N*Area_quadrilateral != 0.0) {
							coeff_NT_on_dTbydx *= (-h_use) / (N*Area_quadrilateral);
							coeff_NT_on_dTbydy *= (-h_use) / (N*Area_quadrilateral);
						}
						else {
							coeff_NT_on_dTbydx *= 0.0;
							coeff_NT_on_dTbydy *= 0.0;
							printf("alarm!! %d \n\n", iVertex);
						}
						
						Results[iPrev] += (coeff_NT_on_dTbydx*0.5*(pos_out.y - info.pos.y)
							- coeff_NT_on_dTbydy*0.5*(pos_out.x - info.pos.x));

						Results[iNeigh] += (coeff_NT_on_dTbydx*0.5*(pos_anti.y - pos_clock.y)
							- coeff_NT_on_dTbydy*0.5*(pos_anti.x - pos_clock.x));

						short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
						Results[iNext] += (coeff_NT_on_dTbydx*0.5*(info.pos.y - pos_out.y)
							- coeff_NT_on_dTbydy*0.5*(info.pos.x - pos_out.x));

						Results_self += (coeff_NT_on_dTbydx*0.5*(pos_clock.y - pos_anti.y)
							- coeff_NT_on_dTbydy*0.5*(pos_clock.x - pos_anti.x));
						/*
						// DEBUG:
						if (Indexneigh[MAXNEIGH_d*threadIdx.x + iPrev] == VERTCHOSEN)
							printf("iVertex %d effect of (iPrev) %d : %1.8E iNeigh %d \n",
								iVertex, VERTCHOSEN, (coeff_NT_on_dTbydx*0.5*(pos_out.y - info.pos.y)
									- coeff_NT_on_dTbydy*0.5*(pos_out.x - info.pos.x)),
								indexneigh);
						if (Indexneigh[MAXNEIGH_d*threadIdx.x + iNext] == VERTCHOSEN)
							printf("iVertex %d effect of (iNext) %d : %1.8E iNeigh %d \n",
								iVertex, VERTCHOSEN, (coeff_NT_on_dTbydx*0.5*(info.pos.y - pos_out.y)
									- coeff_NT_on_dTbydy*0.5*(info.pos.x - pos_out.x)),
								indexneigh);
								
						if (indexneigh == VERTCHOSEN)
							printf("iVertex %d effect of (out) %d : %1.8E iNeigh %d \n",
								iVertex, VERTCHOSEN, (coeff_NT_on_dTbydx*0.5*(pos_anti.y - pos_clock.y)
									- coeff_NT_on_dTbydy*0.5*(pos_anti.x - pos_clock.x)),
								indexneigh);
						if (iVertex == VERTCHOSEN) printf("iVertex %d own effect : %1.8E iNeigh %d\n",
							iVertex, (coeff_NT_on_dTbydx*0.5*(pos_clock.y - pos_anti.y)
								- coeff_NT_on_dTbydy*0.5*(pos_clock.x - pos_anti.x)), indexneigh);
								*/
						// With electron commented out it worked
						// Trying with everything but the new addition
						// it worked - now it doesn't.

/*
					}
				} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

				  // Now go round:	
				endpt_clock = endpt_anti;
				pos_clock = pos_out;
				pos_out = pos_anti;

			}; // next iNeigh

			   // Shan't we choose 3 separate beta -- there is a 3 vector of epsilons. Ah.

			memcpy(&(D_eps_by_dx_neigh_e[iVertex*MAXNEIGH_d]), Results, sizeof(f64)*MAXNEIGH_d);
			Effect_self_e[iVertex] = Results_self;

		}; // was it DOMAIN_VERTEX? Do what otherwise?
	}; // was it OUTERMOST/INNERMOST?
	
	
}*/
// BE SURE TO ZERO THE OUTPUT MEMORY FIRST.

//
//__global__ void kernelCalculateOptimalMove(
//	structural * __restrict__ p_info_major,
//	f64 * __restrict__ D_eps_by_dx_neigh_n,
//	f64 * __restrict__ D_eps_by_dx_neigh_i,
//	f64 * __restrict__ D_eps_by_dx_neigh_e,
//	f64 * __restrict__ Effect_self_n,
//	f64 * __restrict__ Effect_self_i,
//	f64 * __restrict__ Effect_self_e,
//	long * __restrict__ izNeigh_verts, 
//	
//	f64 * __restrict__ p_epsilon_n,
//	f64 * __restrict__ p_epsilon_i,
//	f64 * __restrict__ p_epsilon_e,
//	// output:
//	f64 * __restrict__ p_regressor_n,
//	f64 * __restrict__ p_regressor_i,
//	f64 * __restrict__ p_regressor_e
//	) {
//	__shared__ f64 epsilon[threadsPerTileMajorClever];
//	__shared__ f64 effect_arrays[threadsPerTileMajorClever][MAXNEIGH_d];
//	
//	// That is 13 doubles/thread - for 256 on a tile, the max is 24
//	__shared__ long izNeigh[threadsPerTileMajorClever][MAXNEIGH_d];
//	// Note that we have to overwrite for each species.
//
//	long const iVertex = threadIdx.x + blockDim.x*blockIdx.x;
//	structural info = p_info_major[iVertex];
//	long indexneigh;
//	short iNeigh;
//	int i;
//	long temp_izNeigh[MAXNEIGH_d];
//	f64 numer, denom, self_effect;
//
//	memcpy(izNeigh[threadIdx.x], &(izNeigh_verts[iVertex*MAXNEIGH_d]), sizeof(long)*MAXNEIGH_d);
//	memcpy(effect_arrays[threadIdx.x], &(D_eps_by_dx_neigh_n[iVertex*MAXNEIGH_d]), sizeof(f64)*MAXNEIGH_d);
//	epsilon[threadIdx.x] = p_epsilon_n[iVertex];
//
//	long const StartShared = blockDim.x*blockIdx.x;
//	long const EndShared = StartShared + blockDim.x;
//
//	__syncthreads();
//	
//	if (info.flag == DOMAIN_VERTEX) {
//
//		// Start with own self's effect*own eps
//		self_effect = Effect_self_n[iVertex];
//		numer = epsilon[threadIdx.x] * self_effect;
//		denom = self_effect*self_effect;
//		
//		// gather from neighbours, eps*effect and effect*effect
//
//#pragma unroll MAXNEIGH
//		for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
//		{
//			indexneigh = izNeigh[threadIdx.x][iNeigh];
//			if ((indexneigh >= StartShared) && (indexneigh < EndShared)) {
//								
//				for (i = 0; izNeigh[indexneigh - StartShared][i] != iVertex; i++);
//				numer += epsilon[indexneigh - StartShared] * effect_arrays[indexneigh - StartShared][i];
//				denom += effect_arrays[indexneigh - StartShared][i] * effect_arrays[indexneigh - StartShared][i];
//				
//			}
//			else {
//				// Who am I to this one?
//				memcpy(temp_izNeigh, &(izNeigh_verts[indexneigh*MAXNEIGH_d]), sizeof(long)*MAXNEIGH_d);
//				for (i = 0; temp_izNeigh[i] != iVertex; i++);
//
//				// NOTICE 2 n's here:
//				f64 temp = D_eps_by_dx_neigh_n[indexneigh*MAXNEIGH_d + i];
//				numer += p_epsilon_n[indexneigh] * temp;
//				denom += temp*temp;
//			}
//		};
//
//		if (denom == 0.0) {
//			p_regressor_n[iVertex] = 0.0;
//		}
//		else {
//			p_regressor_n[iVertex] = numer / denom;
//		}
//	} else {
//		p_regressor_n[iVertex] = 0.0;
//
//	}
//	__syncthreads();
//
//	// Now ion load:
//	
//	memcpy(effect_arrays[threadIdx.x], &(D_eps_by_dx_neigh_i[iVertex*MAXNEIGH_d]), sizeof(f64)*MAXNEIGH_d);
//	epsilon[threadIdx.x] = p_epsilon_i[iVertex];
//	
//	__syncthreads();
//
//	if (info.flag == DOMAIN_VERTEX) {
//		self_effect = Effect_self_i[iVertex];
//		numer = epsilon[threadIdx.x] * self_effect;
//		denom = self_effect*self_effect;
//
//		// gather from neighbours, eps*effect and effect*effect
//
//#pragma unroll MAXNEIGH
//		for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
//		{
//			indexneigh = izNeigh[threadIdx.x][iNeigh];
//			if ((indexneigh >= StartShared) && (indexneigh < EndShared)) {
//
//				for (i = 0; izNeigh[indexneigh - StartShared][i] != iVertex; i++);
//				numer += epsilon[indexneigh - StartShared] * effect_arrays[indexneigh - StartShared][i];
//				denom += effect_arrays[indexneigh - StartShared][i] * effect_arrays[indexneigh - StartShared][i];
//
//			}
//			else {
//				// Who am I to this one?
//				memcpy(temp_izNeigh, &(izNeigh_verts[indexneigh*MAXNEIGH_d]), sizeof(long)*MAXNEIGH_d);
//				for (i = 0; temp_izNeigh[i] != iVertex; i++);
//				// OPTIMIZATION: Can do at the start and store who_am_I_to_neighs array.
//
//				// NOTICE 2 i's here:
//				f64 temp = D_eps_by_dx_neigh_i[indexneigh*MAXNEIGH_d + i];
//				numer += p_epsilon_i[indexneigh] * temp;
//				denom += temp*temp;
//			}
//		};
//
//		if (denom == 0.0) {
//			p_regressor_i[iVertex] = 0.0;
//		}
//		else {
//			p_regressor_i[iVertex] = numer / denom;
//		}
//	} else {
//		p_regressor_i[iVertex] = 0.0;
//	}
//	__syncthreads();
//
//	// Now electron load
//
//	memcpy(effect_arrays[threadIdx.x], &(D_eps_by_dx_neigh_e[iVertex*MAXNEIGH_d]), sizeof(f64)*MAXNEIGH_d);
//	epsilon[threadIdx.x] = p_epsilon_e[iVertex];
//
//	__syncthreads();
//
//	if (info.flag == DOMAIN_VERTEX) {
//		self_effect = Effect_self_e[iVertex];
//		numer = epsilon[threadIdx.x] * self_effect;
//		denom = self_effect*self_effect;
//
//#pragma unroll MAXNEIGH
//		for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
//		{
//			indexneigh = izNeigh[threadIdx.x][iNeigh];
//			if ((indexneigh >= StartShared) && (indexneigh < EndShared)) {
//
//				for (i = 0; izNeigh[indexneigh - StartShared][i] != iVertex; i++);
//				numer += epsilon[indexneigh - StartShared] * effect_arrays[indexneigh - StartShared][i];
//				denom += effect_arrays[indexneigh - StartShared][i] * effect_arrays[indexneigh - StartShared][i];
//
//			}
//			else {
//				// Who am I to this one?
//				memcpy(temp_izNeigh, &(izNeigh_verts[indexneigh*MAXNEIGH_d]), sizeof(long)*MAXNEIGH_d);
//				for (i = 0; temp_izNeigh[i] != iVertex; i++);
//				// OPTIMIZATION: Can do at the start and store who_am_I_to_neighs array.
//
//				// NOTICE 2 e's here:
//				f64 temp = D_eps_by_dx_neigh_e[indexneigh*MAXNEIGH_d + i];
//				numer += p_epsilon_e[indexneigh] * temp;
//				denom += temp*temp;
//			};
//		};
//
//		if (denom == 0.0) {
//			p_regressor_e[iVertex] = 0.0;
//		} else {
//			p_regressor_e[iVertex] = numer / denom;
//		}
//	} else {
//		p_regressor_e[iVertex] = 0.0;
//	}
//	// thus we saved off the new regressor; next steps, use it in anger, dimension it.
//	
//}




__global__ void kernelAverage_n_T_x_to_tris(
	nvals * __restrict__ p_n_minor,
	nvals * __restrict__ p_n_major,
	T3 * __restrict__ p_T_minor,
	structural * __restrict__ p_info,
	f64_vec2 * __restrict__ p_cc,

	LONG3 * __restrict__ p_tri_corner_index,
	CHAR4 * __restrict__ p_tri_periodic_corner_flags,

	bool bCalculateOnCircumcenters
)
{
	__shared__ nvals shared_n[threadsPerTileMajor];
	__shared__ T3 shared_T[threadsPerTileMajor];
	__shared__ f64_vec2 shared_pos[threadsPerTileMajor];

	long const iMinor = threadIdx.x + blockIdx.x * blockDim.x; // iMinor OF VERTEX
	if (threadIdx.x < threadsPerTileMajor)
	{
		long getindex = blockIdx.x * threadsPerTileMajor + threadIdx.x;
		shared_n[threadIdx.x] = p_n_major[getindex];
		shared_T[threadIdx.x] = p_T_minor[BEGINNING_OF_CENTRAL + getindex];
		shared_pos[threadIdx.x] = p_info[BEGINNING_OF_CENTRAL + getindex].pos;
	};
	long const StartMajor = blockIdx.x*threadsPerTileMajor; // vertex iMinor
	long const EndMajor = StartMajor + threadsPerTileMajor;
	LONG3 const tri_corner_index = p_tri_corner_index[iMinor];
	CHAR4 const tri_corner_per_flag = p_tri_periodic_corner_flags[iMinor];
	structural info = p_info[iMinor];

	__syncthreads();

	T3 T(0.0, 0.0, 0.0);
	nvals n(0.0, 0.0);
	f64_vec2 pos(0.0, 0.0);
	f64_vec2 cc(0.0, 0.0);
	// New plan for this routine: go through position code for all cases except frills.
	// Then compute averaging coefficients for domain and crossing_ins, and use them.

	// 
	n.n = 0.0;
	n.n_n = 0.0;
	T.Te = 0.0; T.Ti = 0.0; T.Tn = 0.0;

	f64_vec2 poscorner0, poscorner1, poscorner2;
	if ((tri_corner_index.i1 >= StartMajor) && (tri_corner_index.i1 < EndMajor))
	{
		poscorner0 = shared_pos[tri_corner_index.i1 - StartMajor];
	} else {
		poscorner0 = p_info[tri_corner_index.i1 + BEGINNING_OF_CENTRAL].pos;
	};
	if (tri_corner_per_flag.per0 == ROTATE_ME_CLOCKWISE) poscorner0 = Clockwise_d*poscorner0;
	if (tri_corner_per_flag.per0 == ROTATE_ME_ANTICLOCKWISE) poscorner0 = Anticlockwise_d*poscorner0;
	
	if ((tri_corner_index.i2 >= StartMajor) && (tri_corner_index.i2 < EndMajor))
	{
		poscorner1 = shared_pos[tri_corner_index.i2 - StartMajor];
	} else {
		poscorner1 = p_info[tri_corner_index.i2 + BEGINNING_OF_CENTRAL].pos;
	};
	if (tri_corner_per_flag.per1 == ROTATE_ME_CLOCKWISE) poscorner1 = Clockwise_d*poscorner1;
	if (tri_corner_per_flag.per1 == ROTATE_ME_ANTICLOCKWISE) poscorner1 = Anticlockwise_d*poscorner1;
	
	if ((info.flag != INNER_FRILL) && (info.flag != OUTER_FRILL))
	{
		if ((tri_corner_index.i3 >= StartMajor) && (tri_corner_index.i3 < EndMajor))
		{
			poscorner2 = shared_pos[tri_corner_index.i3 - StartMajor];
		} else {
			poscorner2 = p_info[tri_corner_index.i3 + BEGINNING_OF_CENTRAL].pos;
		};
		if (tri_corner_per_flag.per2 == ROTATE_ME_CLOCKWISE) poscorner2 = Clockwise_d*poscorner2;
		if (tri_corner_per_flag.per2 == ROTATE_ME_ANTICLOCKWISE) poscorner2 = Anticlockwise_d*poscorner2;
		
		f64_vec2 Bb = poscorner1 - poscorner0;
		f64_vec2 C = poscorner2 - poscorner0;
		f64 D = 2.0*(Bb.x*C.y - Bb.y*C.x);
		f64 modB = Bb.x*Bb.x + Bb.y*Bb.y;
		f64 modC = C.x*C.x + C.y*C.y;
		cc.x = (C.y*modB - Bb.y*modC) / D + poscorner0.x;
		cc.y = (Bb.x*modC - C.x*modB) / D + poscorner0.y;

		pos = THIRD*(poscorner1 + poscorner0 + poscorner2);

		// Hold up:
		// If cc is outside the triangle, move towards pos until it is inside.

		// Take cc-poscorner0 and look at the dimension that is perpendicular to poscorner1-poscorner2
		// Is it greater than we get for poscorner1-poscorner0

		// If so we've got to move towards pos; how do we know how far to move?
		// Presumably component length changes linearly with change in vector so check component length for pos.

		// Then test if we are outside the other edge normals.

		f64_vec2 minus = cc - poscorner0;
		f64_vec2 edgenormal;
		edgenormal.x = poscorner2.y - poscorner1.y;
		edgenormal.y = poscorner1.x - poscorner2.x;
		// Are 0,1,2 anticlockwise? yes
		// so if x = y2-y1 then it points out
		f64 edgemod = edgenormal.modulus();
		edgenormal /= edgemod;
		f64 dist = minus.dot(edgenormal);
		f64 dist2 = (poscorner2 - poscorner0).dot(edgenormal);
		if (dist > dist2) {
			f64 dist3 = (pos - poscorner0).dot(edgenormal);
			// dist2 = lambda*dist3 + (1-lambda) dist
			// lambda = (dist2-dist) / (dist3-dist)
			cc.x += ((dist2 - dist) / (dist3 - dist))*(pos.x - cc.x);
			cc.y += ((dist2 - dist) / (dist3 - dist))*(pos.y - cc.y);
		}

		minus = cc - poscorner2;
		edgenormal.x = poscorner1.y - poscorner0.y;
		edgenormal.y = poscorner0.x - poscorner1.x;
		edgemod = edgenormal.modulus();
		edgenormal /= edgemod;
		dist = minus.dot(edgenormal);
		dist2 = (poscorner0 - poscorner2).dot(edgenormal);
		if (dist > dist2) {
			f64 dist3 = (pos - poscorner2).dot(edgenormal);
			cc.x += ((dist2 - dist) / (dist3 - dist))*(pos.x - cc.x);
			cc.y += ((dist2 - dist) / (dist3 - dist))*(pos.y - cc.y);
		}

		minus = cc - poscorner1;
		edgenormal.x = poscorner0.y - poscorner2.y;
		edgenormal.y = poscorner2.x - poscorner0.x;
		edgemod = edgenormal.modulus();
		edgenormal /= edgemod;
		dist = minus.dot(edgenormal);
		dist2 = (poscorner0 - poscorner1).dot(edgenormal);
		if (dist > dist2) {
			f64 dist3 = (pos - poscorner1).dot(edgenormal);
			cc.x += ((dist2 - dist) / (dist3 - dist))*(pos.x - cc.x);
			cc.y += ((dist2 - dist) / (dist3 - dist))*(pos.y - cc.y);
		}
		
	} else {
		// FRILL
		pos = 0.5*(poscorner1 + poscorner0);
		f64_vec2 pos2 = pos;
		if (info.flag == INNER_FRILL) {
			pos2.project_to_radius(pos, FRILL_CENTROID_INNER_RADIUS_d);
		} else {
			pos2.project_to_radius(pos, FRILL_CENTROID_OUTER_RADIUS_d);
		};
		cc = pos;
	}
	
	// Now set up averaging coefficients and set n,T.
	// Outer frills it is thus set to n=0,T=0.
	// Well, circumcenter is equidistant so 1/3 is still reasonable average.
	
	// I think I prefer linear interpolation, making this a point estimate of n. The masses are saved
	// in the vertcells.
	
	if (info.flag == DOMAIN_TRIANGLE) {

		f64 lambda1, lambda2, lambda3;
		if (bCalculateOnCircumcenters) {
			
			f64_vec2 x0 = poscorner0, x1 = poscorner1, x2 = poscorner2;
			f64_vec2 a1, a2;
			f64 b1, b2;
//			a1.x = (x1.y - x2.y) / ((x0.x - x2.x)*(x1.y - x2.y) - (x1.x - x2.x)*(x0.y - x2.y));
//			a1.y = (x2.x - x1.x) / ((x0.x - x2.x)*(x1.y - x2.y) - (x1.x - x2.x)*(x0.y - x2.y));
//			b1 = -a1.x*x2.x - a1.y*x2.y;
//			a2.x = (x0.y - x2.y) / ((x1.x - x2.x)*(x0.y - x2.y) - (x1.y - x2.y)*(x0.x - x2.x));
//			a2.y = (x2.x - x0.x) / ((x1.x - x2.x)*(x0.y - x2.y) - (x1.y - x2.y)*(x0.x - x2.x));
//			b2 = -a2.x*x2.x - a2.y*x2.y;
//			lambda1 = a1.x*cc.x + a1.y*cc.y + b1;
//			lambda2 = a2.x*cc.x + a2.y*cc.y + b2;
//			lambda3 = 1.0 - lambda1 - lambda2;

			// We are getting lambda3 < 0 when the point is well inside the triangle.
			// What gives?

			// Try this instead:

			lambda1 = ((x1.y - x2.y)*(cc.x - x2.x) + (x2.x - x1.x)*(cc.y - x2.y)) /
				((x1.y - x2.y)*(x0.x - x2.x) + (x2.x - x1.x)*(x0.y - x2.y));
			lambda2 = ((x2.y-x0.y)*(cc.x-x2.x) + (x0.x-x2.x)*(cc.y-x2.y))/
				((x1.y - x2.y)*(x0.x - x2.x) + (x2.x - x1.x)*(x0.y - x2.y));
			lambda3 = 1.0 - lambda1 - lambda2;
			
			
		} else {
			lambda1 = THIRD;
			lambda2 = THIRD;
			lambda3 = THIRD;
		};

		if ((tri_corner_index.i1 >= StartMajor) && (tri_corner_index.i1 < EndMajor))
		{
			n += lambda1*shared_n[tri_corner_index.i1 - StartMajor];
			T += lambda1*shared_T[tri_corner_index.i1 - StartMajor];
			if (TESTTRI) printf("sharedvers n %1.10E contribn %1.10E\n",
				n.n, shared_n[tri_corner_index.i1 - StartMajor].n);
		}
		else {
			n += lambda1*p_n_major[tri_corner_index.i1];
			T += lambda1*p_T_minor[tri_corner_index.i1 + BEGINNING_OF_CENTRAL];
			if (TESTTRI) printf("loadvers n %1.10E contribn %1.10E\n",
				n.n, p_n_major[tri_corner_index.i1].n);
		};

		if ((tri_corner_index.i2 >= StartMajor) && (tri_corner_index.i2 < EndMajor))
		{
			n += lambda2*shared_n[tri_corner_index.i2 - StartMajor];
			T += lambda2*shared_T[tri_corner_index.i2 - StartMajor];
			if (TESTTRI) printf("sharedvers n %1.10E contribn %1.10E\n",
				n.n, shared_n[tri_corner_index.i2 - StartMajor].n);
		}
		else {
			n += lambda2*p_n_major[tri_corner_index.i2];
			T += lambda2*p_T_minor[tri_corner_index.i2 + BEGINNING_OF_CENTRAL];
			if (TESTTRI) printf("loadvers n %1.10E contribn %1.10E\n",
				n.n, p_n_major[tri_corner_index.i2].n);
		};

		if ((tri_corner_index.i3 >= StartMajor) && (tri_corner_index.i3 < EndMajor))
		{
			n += lambda3*shared_n[tri_corner_index.i3 - StartMajor];
			T += lambda3*shared_T[tri_corner_index.i3 - StartMajor];
			if (TESTTRI) printf("sharedvers n %1.10E contribn %1.10E\n",
				n.n, shared_n[tri_corner_index.i3 - StartMajor].n);
		}
		else {
			n += lambda3*p_n_major[tri_corner_index.i3];
			T += lambda3*p_T_minor[tri_corner_index.i3 + BEGINNING_OF_CENTRAL];
			if (TESTTRI) printf("loadvers n %1.10E contribn %1.10E\n",
				n.n, p_n_major[tri_corner_index.i3].n);
		}; 
		if (TESTTRI) 
			printf("%d: lambda %1.10E %1.10E %1.10E\ncorner n %1.10E %1.10E %1.10E\n"
				"cc %1.9E %1.9E | %1.9E %1.9E | %1.9E %1.9E | %1.9E %1.9E \n"
				"indexcorner %d %d %d result n= %1.10E \n\n",
				CHOSEN, lambda1, lambda2, lambda3, 
				p_n_major[tri_corner_index.i1].n,
				p_n_major[tri_corner_index.i2].n,
				p_n_major[tri_corner_index.i3].n,
				cc.x,cc.y, poscorner0.x, poscorner0.y, poscorner1.x, poscorner1.y, poscorner2.x, poscorner2.y,
				tri_corner_index.i1, tri_corner_index.i2, tri_corner_index.i3,
				n.n
			);
		
	} else {
		// What else?
		if (info.flag == CROSSING_INS)
		{
			int iAbove = 0;
			if ((tri_corner_index.i1 >= StartMajor) && (tri_corner_index.i1 < EndMajor))
			{
				if (poscorner0.dot(poscorner0) > DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					n += shared_n[tri_corner_index.i1 - StartMajor];
					T += shared_T[tri_corner_index.i1 - StartMajor];
					iAbove++;
				};

			} else {
				if (poscorner0.dot(poscorner0) > DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					n += p_n_major[tri_corner_index.i1];
					T += p_T_minor[tri_corner_index.i1 + BEGINNING_OF_CENTRAL];
					iAbove++;
				}
			};
			
			if ((tri_corner_index.i2 >= StartMajor) && (tri_corner_index.i2 < EndMajor))
			{
				if (poscorner1.dot(poscorner1) > DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					n += shared_n[tri_corner_index.i2 - StartMajor];
					T += shared_T[tri_corner_index.i2 - StartMajor];
					iAbove++;
				};
			} else {
				if (poscorner1.dot(poscorner1) > DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					n += p_n_major[tri_corner_index.i2];
					T += p_T_minor[tri_corner_index.i2 + BEGINNING_OF_CENTRAL];
					iAbove++;
				};
			};
			
			if ((tri_corner_index.i3 >= StartMajor) && (tri_corner_index.i3 < EndMajor))
			{
				if (poscorner2.dot(poscorner2) > DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					n += shared_n[tri_corner_index.i3 - StartMajor];
					T += shared_T[tri_corner_index.i3 - StartMajor];
					iAbove++;
				};
			} else {
				if (poscorner2.dot(poscorner2) > DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					n += p_n_major[tri_corner_index.i3];
					T += p_T_minor[tri_corner_index.i3 + BEGINNING_OF_CENTRAL];
					iAbove++;
				};
			};
			
			f64_vec2 pos2 = pos;
			pos2.project_to_radius(pos,DEVICE_RADIUS_INSULATOR_OUTER);
			f64 divide = 1.0 / (f64)iAbove;
			n.n *= divide;
			n.n_n *= divide;
			T.Tn *= divide;
			T.Ti *= divide;
			T.Te *= divide;

		} else {
			n.n = 0.0;
			n.n_n = 0.0;
			T.Te = 0.0; T.Ti = 0.0; T.Tn = 0.0;
		};
		// Outer frills it is thus set to n=0,T=0.
	};

	if (TESTTRI) printf("\n%d info.pos.x %1.9E cc.x %1.9E \n", iMinor, pos.x, cc.x);

	p_n_minor[iMinor] = n;
	p_T_minor[iMinor] = T;
	info.pos = pos;
	p_info[iMinor] = info;
	p_cc[iMinor] = cc;
}


__global__ void kernelCreateShardModelOfDensities_And_SetMajorArea(
	structural * __restrict__ p_info_minor,
	nvals * __restrict__ p_n_major, 
	nvals * __restrict__ p_n_minor,
	long * __restrict__ p_izTri_vert,
	char * __restrict__ p_szPBCtri_vert,
	f64_vec2 * __restrict__ p_cc,
	ShardModel * __restrict__ p_n_shards,
	ShardModel * __restrict__ p_n_n_shards,
	//	long * __restrict__ Tri_n_lists,
	//	long * __restrict__ Tri_n_n_lists	,
	f64 * __restrict__ p_AreaMajor,
	bool bUseCircumcenter
	)// sets n_shards_n, n_shards, Tri_n_n_lists, Tri_n_lists
{
	// called for major tile
	// Interpolation to Tri_n_lists, Tri_n_n_lists is not yet implemented. But this would be output.

	// Inputs:
	// n, pTri->cent,  izTri,  pTri->periodic, pVertex->pos

	// Outputs:
	// pVertex->AreaCell
	// n_shards[iVertex]
	// Tri_n_n_lists[izTri[i]][o1 * 2] <--- 0 if not set by domain vertex

	// CALL AVERAGE OF n TO TRIANGLES - WANT QUADRATIC AVERAGE - BEFORE WE BEGIN
	// MUST ALSO POPULATE pVertex->AreaCell with major cell area

	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];
	__shared__ nvals shared_n[threadsPerTileMinor];

	// Here 4 doubles/minor. In 16*1024, 4 double*8 bytes*512 minor. 256 major. 
	// Choosing to store n_n while doing n which is not necessary.

	ShardModel n_; // to be populated
	
	int iNeigh, tri_len;
	f64 N_n, N, interpolated_n, interpolated_n_n;
	long i, inext, o1, o2;

	//memset(Tri_n_n_lists, 0, sizeof(f64)*NUMTRIANGLES * 6);
	//memset(Tri_n_lists, 0, sizeof(f64)*NUMTRIANGLES * 6);

	// We can afford to stick 6-8 doubles in shared. 8 vars*8 bytes*256 threads = 16*1024.
	if (bUseCircumcenter == false)
	{
		structural info2[2];
		memcpy(info2, p_info_minor + blockIdx.x*threadsPerTileMinor + 2 * threadIdx.x, sizeof(structural) * 2);
		shared_pos[2 * threadIdx.x] = info2[0].pos;
		shared_pos[2 * threadIdx.x + 1] = info2[1].pos;
	} else {
		memcpy(&(shared_pos[2 * threadIdx.x]), p_cc + blockIdx.x*threadsPerTileMinor + 2 * threadIdx.x, sizeof(f64_vec2) * 2);
	}
	memcpy(&(shared_n[2 * threadIdx.x]), p_n_minor + blockIdx.x*threadsPerTileMinor + 2 * threadIdx.x, sizeof(nvals) * 2);
		
	__syncthreads();

	long const StartMinor = blockIdx.x*threadsPerTileMinor; // vertex index
	long const EndMinor = StartMinor + threadsPerTileMinor;
	// To fit in Tri_n_n_lists stuff we should first let coeff[] go out of scope.
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX
	structural info = p_info_minor[BEGINNING_OF_CENTRAL + iVertex];

	if (info.flag == DOMAIN_VERTEX) {

		long izTri[MAXNEIGH];
		char szPBC[MAXNEIGH];
		f64 coeff[MAXNEIGH];   // total 21*12 = 252 bytes. 256 max for 192 threads.
		f64 ndesire0, ndesire1;
		f64_vec2 pos0, pos1;

		memcpy(izTri, p_izTri_vert + MAXNEIGH_d*iVertex, sizeof(long)*MAXNEIGH_d);
		memcpy(szPBC, p_szPBCtri_vert + MAXNEIGH_d*iVertex, sizeof(char)*MAXNEIGH_d);

		f64 n_avg = p_n_major[iVertex].n;
		// WHY WAS IT minor NOT major ?????????????????????????

		if ((izTri[0] >= StartMinor) && (izTri[0] < EndMinor)) {
			pos0 = shared_pos[izTri[0] - StartMinor];
			ndesire0 = shared_n[izTri[0] - StartMinor].n;
		}
		else {
			if (bUseCircumcenter) {
				pos0 = p_cc[izTri[0]];
			} else {
				pos0 = p_info_minor[izTri[0]].pos;
			} // there exists a more elegant way than this!!!

			ndesire0 = p_n_minor[izTri[0]].n;
		}
		if (szPBC[0] == ROTATE_ME_CLOCKWISE) pos0 = Clockwise_d*pos0;
		if (szPBC[0] == ROTATE_ME_ANTICLOCKWISE) pos0 = Anticlockwise_d*pos0;

		f64 tri_area;
		f64 N0 = 0.0; f64 coeffcent = 0.0;
		memset(coeff, 0, sizeof(f64)*MAXNEIGH_d);
		short i;
		f64 AreaMajor = 0.0;
		f64 high_n = ndesire0;
		f64 low_n = ndesire0;
#pragma unroll MAXNEIGH
		for (i = 0; i < info.neigh_len; i++)
		{
			// Temporary setting:
			n_.n[i] = ndesire0;

			if TEST printf("GPU VERTCHOSEN %d : ndesire %1.14E \n", VERTCHOSEN, ndesire0);

			inext = i + 1; if (inext == info.neigh_len) inext = 0;
			if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor)) {
				pos1 = shared_pos[izTri[inext] - StartMinor];
				ndesire1 = shared_n[izTri[inext] - StartMinor].n;
			} else {
				if (bUseCircumcenter) {
					pos1 = p_cc[izTri[inext]];
				} else {
					pos1 = p_info_minor[izTri[inext]].pos;
				}
				ndesire1 = p_n_minor[izTri[inext]].n;
			};
			if (szPBC[inext] == ROTATE_ME_CLOCKWISE) pos1 = Clockwise_d*pos1;
			if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) pos1 = Anticlockwise_d*pos1;

			high_n = max(ndesire1, high_n);
			low_n = min(ndesire1, low_n);

			tri_area = fabs(0.5*
				((pos0.x + pos1.x) * (pos1.y - pos0.y)
					+ (info.pos.x + pos1.x) * (info.pos.y - pos1.y)
					+ (info.pos.x + pos0.x) * (pos0.y - info.pos.y)));

			N0 += tri_area*THIRD*(ndesire0 + ndesire1);
			coeff[i] += tri_area*THIRD;
			coeff[inext] += tri_area*THIRD;
			coeffcent += tri_area*THIRD;
			AreaMajor += tri_area;
			pos0 = pos1;
			ndesire0 = ndesire1;
		};
		// . If n_avg > n_max_corners then set all to n_avg.
		// . If n_min < n_needed < n_max then set n_cent = n_needed

		// Otherwise, we now have coeff array populated and will go round
		// repeatedly. We have to reload n lots of times.
		// This is not the typical case.

		p_AreaMajor[iVertex] = AreaMajor;
		
		if ((n_avg > high_n) || (n_avg < low_n)) {
#pragma unroll MAXNEIGH
			for (i = 0; i < info.neigh_len; i++)
				n_.n[i] = n_avg;
			n_.n_cent = n_avg;


			if TEST printf("VERTCHOSEN (n_avg > high_n) || (n_avg < low_n) \n");
	//		if (iVertex == CHOSEN) printf("CHOSEN : Switch1 n_avg %1.12E \n",n_avg);

		} else {
			real n_C_need = (n_avg*AreaMajor - N0) / coeffcent;

			if ((n_C_need > low_n) && (n_C_need < high_n)) {
				n_.n_cent = n_C_need;

				if (TEST) printf("VERTCHOSEN ((n_C_need > low_n) && (n_C_need < high_n)) \n");

	//			if (iVertex == CHOSEN) printf("CHOSEN : Switch2 n_C_need %1.12E low_n %1.12E high_n %1.12E\n", n_C_need,low_n,high_n);

			}
			else {
				// The laborious case.
	//			if (iVertex == CHOSEN) printf("Laborious case...\n");

				if (TEST) printf("VERTCHOSEN  The laborious case. n_avg %1.10E n_C_need %1.10E low_n %1.10E high_n %1.10E\n",
					n_avg, n_C_need, low_n, high_n);
				bool fixed[MAXNEIGH];
				memset(fixed, 0, sizeof(bool) * MAXNEIGH);
				// cannot fit even this alongside the rest we have in L1.
				// Can we make szPBC go out of scope by here?

				f64 n_C, n_acceptable;
				if (n_C_need < low_n) {
					// the mass is low. So for those less than some n_acceptable,
					// let them attain n_desire, and fix n_C = low_n.
					// Then we'll see how high we can go with n_acceptable.

	//				if (iVertex == CHOSEN) printf("(n_C_need < low_n)\n");

					n_C = low_n;
					n_acceptable = (n_avg*AreaMajor - coeffcent*n_C) / (AreaMajor - THIRD*AreaMajor);
					// area-THIRD*area = sum of other coeffs, and of course
					// coeffcent = THIRD*area
					// n_acceptable > N/area since N=area*n_avg > area*low_n.

					// We accept things that are less than this 'max average', and
					// let that increase the threshold; go again until
					// the time we do not find any new lower items ;	
					bool found = 0;
					do {
						found = 0;
						f64 coeffremain = 0.0;
						f64 N_attained = coeffcent*low_n;
						for (i = 0; i < info.neigh_len; i++)
						{
							if (fixed[i] == 0) {
								// Go collect ndesire[i]:

								f64 ndesire;
								if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
								{
									ndesire = shared_n[izTri[i] - StartMinor].n;
								}
								else {
									ndesire = p_n_minor[izTri[i]].n;
								};
								
		//						if (iVertex == CHOSEN) printf("CHOSEN : ndesire %1.14E n_acceptable %1.14E\n", ndesire,n_acceptable);
								
								if (ndesire < n_acceptable) { // yes, use ndesire[i] ...
									fixed[i] = true;
									n_.n[i] = ndesire;
									N_attained += n_.n[i] * coeff[i];
									found = true;
								}
								else {
									coeffremain += coeff[i];
								};
							}
							else {
								N_attained += n_.n[i] * coeff[i];
							};
						};
						// It can happen that eventually ALL are found
						// to be < n_acceptable due to FP error.
						// On next pass found will be false.
						if ((found != 0) && (coeffremain > 0.0)) {
							n_acceptable = (n_avg*AreaMajor - N_attained) / coeffremain;
							// The value to which we have to set the remaining
							// n values.
						};
	//					if (iVertex == CHOSEN) printf("---\n");
					} while (found != 0);

				} else {
					n_C = high_n;
					n_acceptable = (n_avg*AreaMajor - coeffcent*n_C) / (AreaMajor - THIRD*AreaMajor);
					bool found = 0;
					do {
						found = 0;
						f64 coeffremain = 0.0;
						f64 N_attained = coeffcent*high_n;
						for (i = 0; i < info.neigh_len; i++)
						{
							if (fixed[i] == 0) {
								// Go collect ndesire[i]:

								f64 ndesire;
								if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
								{
									ndesire = shared_n[izTri[i] - StartMinor].n;
								} else {
									ndesire = p_n_minor[izTri[i]].n;
								};
								

	//							if (iVertex == CHOSEN) printf("CHOSEN : ndesire %1.14E n_acceptable %1.14E\n", ndesire, n_acceptable);


								if (ndesire > n_acceptable) {
									// yes, use ndesire[i] ...
									fixed[i] = true;
									n_.n[i] = ndesire;
									N_attained += n_.n[i] * coeff[i];
									found = true;
								} else {
									coeffremain += coeff[i];
								};
							} else {
								N_attained += n_.n[i] * coeff[i];
							};
						};
						if ((found != 0) && (coeffremain > 0.0)) {
							n_acceptable = (n_avg*AreaMajor - N_attained) / coeffremain;
						};

	//					if (iVertex == CHOSEN) printf("@@@ \n");
						
					} while (found != 0);
				};
				// Now we should set the remaining values to n_acceptable
				// which is less than ndesire[i] in all those cases.
				for (i = 0; i < info.neigh_len; i++)
				{
					if (fixed[i] == 0) n_.n[i] = n_acceptable;

					if (TEST) printf("n[%d]: %1.10E\n", i, n_.n[i]);
				};
				n_.n_cent = n_C;
				if (TEST) printf("n_.n_cent %1.10E \n", n_.n_cent);

			/*	if (iVertex == CHOSEN) {
					for (i = 0; i < info.neigh_len; i++)
					{
						printf("GPU: n %1.14E\n", n_.n[i]);
					}
					printf("GPU : n_cent %1.14E \n", n_.n_cent);
				};
			*/	
			};
		};

		memcpy(&(p_n_shards[iVertex]), &n_, sizeof(ShardModel));

		// Now start again: neutrals

		n_avg = p_n_major[iVertex].n_n;

		if ((izTri[0] >= StartMinor) && (izTri[0] < EndMinor)) {
			pos0 = shared_pos[izTri[0] - StartMinor];
			ndesire0 = shared_n[izTri[0] - StartMinor].n_n;
		} else {
			if (bUseCircumcenter) {
				pos0 = p_cc[izTri[0]];
			} else {
				pos0 = p_info_minor[izTri[0]].pos;
			};
			ndesire0 = p_n_minor[izTri[0]].n_n;
		}
		if (szPBC[0] == ROTATE_ME_CLOCKWISE) pos0 = Clockwise_d*pos0;
		if (szPBC[0] == ROTATE_ME_ANTICLOCKWISE) pos0 = Anticlockwise_d*pos0;

		N0 = 0.0;
		//coeffcent = 0.0;
		//memset(coeff, 0, sizeof(f64)*MAXNEIGH_d); // keep em
		high_n = ndesire0;
		low_n = ndesire0;

#pragma unroll MAXNEIGH
		for (i = 0; i < info.neigh_len; i++)
		{
			// Temporary setting:
			n_.n[i] = ndesire0;

			inext = i + 1; if (inext == info.neigh_len) inext = 0;
			if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor)) {
				pos1 = shared_pos[izTri[inext] - StartMinor];
				ndesire1 = shared_n[izTri[inext] - StartMinor].n_n;
			} else {
				if (bUseCircumcenter) {
					pos1 = p_cc[izTri[inext]]; 
				} else {
					pos1 = p_info_minor[izTri[inext]].pos;
				}
				ndesire1 = p_n_minor[izTri[inext]].n_n;
			};
			if (szPBC[inext] == ROTATE_ME_CLOCKWISE) pos1 = Clockwise_d*pos1;
			if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) pos1 = Anticlockwise_d*pos1;

			high_n = max(ndesire1, high_n);
			low_n = min(ndesire1, low_n);

			tri_area = fabs(0.5*
				((pos0.x + pos1.x) * (pos1.y - pos0.y)
					+ (info.pos.x + pos1.x) * (info.pos.y - pos1.y)
					+ (info.pos.x + pos0.x) * (pos0.y - info.pos.y)));

			N0 += tri_area*THIRD*(ndesire0 + ndesire1); // Could consider moving it into loop above.

			pos0 = pos1;
			ndesire0 = ndesire1;
		};
		// . If n_avg > n_max_corners then set all to n_avg.
		// . If n_min < n_needed < n_max then set n_cent = n_needed

		// Otherwise, we now have coeff array populated and will go round
		// repeatedly. We have to reload n lots of times.
		// This is not the typical case.

		if ((n_avg > high_n) || (n_avg < low_n)) {
#pragma unroll MAXNEIGH
			for (i = 0; i < info.neigh_len; i++)
				n_.n[i] = n_avg;
			n_.n_cent = n_avg;


		} else {
			real n_C_need = (n_avg*AreaMajor - N0) / coeffcent;

			if ((n_C_need > low_n) && (n_C_need < high_n)) {
				n_.n_cent = n_C_need; // accept desired values

			} else {
				// The laborious case.


				bool fixed[MAXNEIGH];
				memset(fixed, 0, sizeof(bool) * MAXNEIGH);
				// cannot fit even this alongside the rest we have in L1.
				// Can we make szPBC go out of scope by here?

				f64 n_C, n_acceptable;
				if (n_C_need < low_n) {
					// the mass is low. So for those less than some n_acceptable,
					// let them attain n_desire, and fix n_C = low_n.
					// Then we'll see how high we can go with n_acceptable.

					n_C = low_n;
					n_acceptable = (n_avg*AreaMajor - coeffcent*n_C) / (AreaMajor - THIRD*AreaMajor);
					// area-THIRD*area = sum of other coeffs, and of course
					// coeffcent = THIRD*area
					// n_acceptable > N/area since N=area*n_avg > area*low_n.

					// We accept things that are less than this 'max average', and
					// let that increase the threshold; go again until
					// the time we do not find any new lower items ;		
					bool found = 0;
					do {
						found = 0;
						f64 coeffremain = 0.0;
						f64 N_attained = coeffcent*low_n;
						for (i = 0; i < info.neigh_len; i++)
						{
							if (fixed[i] == 0) {
								// Go collect ndesire[i]:

								f64 ndesire;
								if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
								{
									ndesire = shared_n[izTri[i] - StartMinor].n_n;
								}
								else {
									ndesire = p_n_minor[izTri[i]].n_n;
								};
								if (ndesire < n_acceptable) { // yes, use ndesire[i] ...
									fixed[i] = true;
									n_.n[i] = ndesire;
									N_attained += n_.n[i] * coeff[i];
									found = true;
								}
								else {
									coeffremain += coeff[i];
								};
							}
							else {
								N_attained += n_.n[i] * coeff[i];
							};
						};
						// It can happen that eventually ALL are found
						// to be < n_acceptable due to FP error.
						// On next pass found will be false.
						if ((found != 0) && (coeffremain > 0.0)) {
							n_acceptable = (n_avg*AreaMajor - N_attained) / coeffremain;
							// The value to which we have to set the remaining
							// n values.
						};
					} while (found != 0);

				}
				else {
					n_C = high_n;
					n_acceptable = (n_avg*AreaMajor - coeffcent*n_C) / (AreaMajor - THIRD*AreaMajor);
					bool found = 0;
					do {
						found = 0;
						f64 coeffremain = 0.0;
						f64 N_attained = coeffcent*high_n;
						for (i = 0; i < info.neigh_len; i++)
						{
							if (fixed[i] == 0) {
								// Go collect ndesire[i]:

								f64 ndesire;
								if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
								{
									ndesire = shared_n[izTri[i] - StartMinor].n_n;
								}
								else {
									ndesire = p_n_minor[izTri[i]].n_n;
								};
								if (ndesire > n_acceptable) {
									// yes, use ndesire[i] ...
									fixed[i] = true;
									n_.n[i] = ndesire;
									N_attained += n_.n[i] * coeff[i];
									found = true;
								}
								else {
									coeffremain += coeff[i];
								};
							}
							else {
								N_attained += n_.n[i] * coeff[i];
							};

						};
						if ((found != 0) && (coeffremain > 0.0)) {
							n_acceptable = (n_avg*AreaMajor - N_attained) / coeffremain;
						};
					} while (found != 0);

				};
				// Now we should set the remaining values to n_acceptable
				// which is less than ndesire[i] in all those cases.
				for (i = 0; i < info.neigh_len; i++)
				{
					if (fixed[i] == 0) n_.n[i] = n_acceptable;

				};
				n_.n_cent = n_C;
			};
		};

		memcpy(&(p_n_n_shards[iVertex]), &n_, sizeof(ShardModel));

		// Now done both species.

	} else { // NOT DOMAIN_VERTEX
		
		if (info.flag == OUTERMOST) {
			n_.n_cent = p_n_major[iVertex].n;
			for (i = 0; i < MAXNEIGH; i++)
				n_.n[i] = n_.n_cent;
			memcpy(&(p_n_shards[iVertex]), &n_, sizeof(ShardModel));
			n_.n_cent = p_n_major[iVertex].n_n;
			for (i = 0; i < MAXNEIGH; i++)
				n_.n[i] = n_.n_cent;
			memcpy(&(p_n_n_shards[iVertex]), &n_, sizeof(ShardModel));

			f64 AreaTotal = PPN_CIRCLE*M_PI*(DOMAIN_OUTER_RADIUS*DOMAIN_OUTER_RADIUS -
				INNER_A_BOUNDARY*INNER_A_BOUNDARY);
			p_AreaMajor[iVertex] = AreaTotal / (real)(numTilesMajor*threadsPerTileMajor); // ?
			// Setting area of outermost to average vertcell area
			// ...
		}
		else {
			memset(&(p_n_shards[iVertex]), 0, sizeof(ShardModel));
			memset(&(p_n_n_shards[iVertex]), 0, sizeof(ShardModel));


			p_AreaMajor[iVertex] = 0.0; // NOTE BENE
		};
	};

	// NexT:  tri_n_lists.

	// Think I am not using this passing mechanism for n_shards information.

	/*
	for (i = 0; i < cp.numCoords; i++)
	{
	// for 2 triangles each corner:

	// first check which number corner this vertex is
	// make sure we enter them in order that goes anticlockwise for the
	// Then we need to make izMinorNeigh match this somehow

	// Let's say izMinorNeigh goes [across corner 0, across edge 2, corner 1, edge 0, corner 2, edge 1]
	// We want 0,1 to be the values corresp corner 0.

	// shard value 0 is in tri 0. We look at each pair of shard values in turn to interpolate.

	inext = i + 1; if (inext == cp.numCoords) inext = 0;

	interpolated_n = THIRD * (n_shards[iVertex].n[i] + n_shards[iVertex].n[inext] + n_shards[iVertex].n_cent);
	interpolated_n_n = THIRD * (n_shards_n[iVertex].n[i] + n_shards_n[iVertex].n[inext] + n_shards_n[iVertex].n_cent);
	// contribute to tris i and inext:
	o1 = (T + izTri[i])->GetCornerIndex(X + iVertex);
	o2 = (T + izTri[inext])->GetCornerIndex(X + iVertex);

	// Now careful which one's which:

	// inext sees this point as more anticlockwise.

	Tri_n_lists[izTri[inext]][o2 * 2 + 1] = interpolated_n;
	Tri_n_lists[izTri[i]][o1 * 2] = interpolated_n;

	Tri_n_n_lists[izTri[inext]][o2 * 2 + 1] = interpolated_n_n;
	Tri_n_n_lists[izTri[i]][o1 * 2] = interpolated_n_n;
	};*/

}


__global__ void kernelInferMinorDensitiesFromShardModel(
	structural * __restrict__ p_info,
	nvals * __restrict__ p_n_minor,
	ShardModel * __restrict__ p_n_shards,
	ShardModel * __restrict__ p_n_shards_n,
	LONG3 * __restrict__ p_tri_corner_index,
	LONG3 * __restrict__ p_who_am_I_to_corner,
	nvals * __restrict__ p_one_over_n
) {
	// Assume that we do the simplest thing possible.

	long const iMinor = threadIdx.x + blockIdx.x * blockDim.x; // iMinor OF VERTEX
	structural info = p_info[iMinor];
	nvals result;

	if (iMinor >= BEGINNING_OF_CENTRAL)
	{
		if (info.flag == DOMAIN_VERTEX) {
			result.n = p_n_shards[iMinor - BEGINNING_OF_CENTRAL].n_cent;
			result.n_n = p_n_shards_n[iMinor - BEGINNING_OF_CENTRAL].n_cent;
			p_n_minor[iMinor] = result;
			result.n = 1.0 / result.n;
			result.n_n = 1.0 / result.n_n;
			p_one_over_n[iMinor] = result;

			// We are not being consistent.
			// We may wish to use major n here --> minor central n

			// We have not done the shard model for target n, we just average and then tween this back.
			// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

		} else {
			// Outermost vertex?
			result.n = 0.0;
			result.n_n = 0.0;
			if (info.flag == OUTERMOST) {
				result.n_n = 1.0e18;
				result.n = 1.0e12;
			};
			p_n_minor[iMinor] = result;
			result.n_n = 1.0 / result.n_n;
			result.n = 1.0 / result.n;
			p_one_over_n[iMinor] = result;
		}
	} else {
		if (info.flag == DOMAIN_TRIANGLE) {
			LONG3 tri_corner_index = p_tri_corner_index[iMinor];
			LONG3 who_am_I_to_corner = p_who_am_I_to_corner[iMinor];
			result.n = THIRD*
				(p_n_shards[tri_corner_index.i1].n[who_am_I_to_corner.i1]
					+ p_n_shards[tri_corner_index.i2].n[who_am_I_to_corner.i2]
					+ p_n_shards[tri_corner_index.i3].n[who_am_I_to_corner.i3]);
			result.n_n = THIRD*
				(p_n_shards_n[tri_corner_index.i1].n[who_am_I_to_corner.i1]
					+ p_n_shards_n[tri_corner_index.i2].n[who_am_I_to_corner.i2]
					+ p_n_shards_n[tri_corner_index.i3].n[who_am_I_to_corner.i3]);
			p_n_minor[iMinor] = result;
			if (TESTTRI) printf("%d: shards n %1.10E %1.10E %1.10E result %1.10E\n",
				CHOSEN, p_n_shards[tri_corner_index.i1].n[who_am_I_to_corner.i1],
				p_n_shards[tri_corner_index.i2].n[who_am_I_to_corner.i2],
				p_n_shards[tri_corner_index.i3].n[who_am_I_to_corner.i3], result.n);

			result.n = THIRD*(
				1.0/ p_n_shards[tri_corner_index.i1].n[who_am_I_to_corner.i1]
				+ 1.0/ p_n_shards[tri_corner_index.i2].n[who_am_I_to_corner.i2]
					+ 1.0/p_n_shards[tri_corner_index.i3].n[who_am_I_to_corner.i3]);
			result.n_n = THIRD*
				(1.0/p_n_shards_n[tri_corner_index.i1].n[who_am_I_to_corner.i1]
					+ 1.0/p_n_shards_n[tri_corner_index.i2].n[who_am_I_to_corner.i2]
					+ 1.0/p_n_shards_n[tri_corner_index.i3].n[who_am_I_to_corner.i3]);
			p_one_over_n[iMinor] = result;


		} else {
			if (info.flag == CROSSING_INS) {
				LONG3 tri_corner_index = p_tri_corner_index[iMinor];
				LONG3 who_am_I_to_corner = p_who_am_I_to_corner[iMinor];
				result.n = 0.0;
				result.n_n = 0.0;
				
				structural info1, info2, info3;
				info1 = p_info[BEGINNING_OF_CENTRAL + tri_corner_index.i1];
				info2 = p_info[BEGINNING_OF_CENTRAL + tri_corner_index.i2];
				info3 = p_info[BEGINNING_OF_CENTRAL + tri_corner_index.i3];
				int numabove = 0;
				if (info1.flag == DOMAIN_VERTEX) {
					numabove++;
					result.n += p_n_shards[tri_corner_index.i1].n[who_am_I_to_corner.i1];
					result.n_n += p_n_shards_n[tri_corner_index.i1].n[who_am_I_to_corner.i1];
				};
				if (info2.flag == DOMAIN_VERTEX) {
					numabove++;
					result.n += p_n_shards[tri_corner_index.i2].n[who_am_I_to_corner.i2];
					result.n_n += p_n_shards_n[tri_corner_index.i2].n[who_am_I_to_corner.i2];
				};
				if (info3.flag == DOMAIN_VERTEX) {
					numabove++;
					result.n += p_n_shards[tri_corner_index.i3].n[who_am_I_to_corner.i3];
					result.n_n += p_n_shards_n[tri_corner_index.i3].n[who_am_I_to_corner.i3];
				};
				result.n /= (f64)numabove;
				result.n_n /= (f64)numabove;
				p_n_minor[iMinor] = result;
				result.n = 1.0 / result.n;
				result.n_n = 1.0 / result.n_n;
				p_one_over_n[iMinor] = result;
			} else {
				memset(&(p_n_minor[iMinor]), 0, sizeof(nvals));
			}
		}
	}
}



__global__ void kernelCalculate_ita_visc(
	structural * __restrict__ p_info_minor,
	nvals * __restrict__ p_n_minor,
	T3 * __restrict__ p_T_minor,

	f64 * __restrict__ p_nu_ion_minor,
	f64 * __restrict__ p_nu_elec_minor,
	f64 * __restrict__ p_nu_nn_visc,
	f64 * __restrict__ p_ita_par_ion_minor,
	f64 * __restrict__ p_ita_par_elec_minor,
	f64 * __restrict__ p_ita_neutral_minor)
{
	// Save nu_iHeart, nu_eHeart, nu_nn_visc.

	f64 TeV, sigma_MT, sigma_visc, sqrt_T, nu_en_visc;
	T3 T;
	f64 nu_in_visc, nu_ni_visc, nu_ii, nu_nn;
	nvals our_n;
	long const index = threadIdx.x + blockIdx.x * blockDim.x; 
	structural info = p_info_minor[index];
	if ((info.flag == DOMAIN_VERTEX) 
		//|| (info.flag == OUTERMOST)
		|| (info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS)) {

		// We have not ruled out calculating traffic into outermost vertex cell - so this needs nu calculated in it.
		// (Does it actually try and receive traffic?)

		our_n = p_n_minor[index]; // never used again once we have kappa
		T = p_T_minor[index];

		TeV = T.Te * one_over_kB;
		Estimate_Ion_Neutral_Cross_sections_d(TeV, &sigma_MT, &sigma_visc);
		sqrt_T = sqrt(T.Te);
		nu_en_visc = our_n.n_n * sigma_visc * sqrt_T * over_sqrt_m_e;
		f64 nu_eiBar = nu_eiBarconst * kB_to_3halves * our_n.n * Get_lnLambda_d(our_n.n, T.Te) / (T.Te*sqrt_T);

		p_ita_par_elec_minor[index] =
			//0.73*our_n.n*T.Te / nu_eiBar; // ? Check it's not in eV in formulary
			0.5*our_n.n*T.Te / ((0.3*0.87 + 0.6)*nu_eiBar + 0.6*nu_en_visc);
		// This from Zhdanov chapter 7. Compare Braginskii.
		// 0.5/(0.3*0.87+0.6) = 0.58 not 0.73
		
		p_nu_elec_minor[index] = (0.3*0.87 + 0.6)*nu_eiBar + 0.6*nu_en_visc;

//		if ((index == 85822) || (index == 24335))
//			printf("\n###################################\nindex %d nu_e %1.14E ita %1.14E n %1.14E Te %1.14E \nnu_eiBar %1.14E nu_en_visc %1.14E\n\n",
//				index, (0.3*0.87 + 0.6)*nu_eiBar + 0.6*nu_en_visc, p_ita_par_elec_minor[index],
//				our_n.n, T.Te, nu_eiBar, nu_en_visc);
		
		//nu_eHeart:
	//	nu.e = nu_en_visc + 1.87*nu_eiBar;

	// FOR NOW IT DOES NOT MATTER BECAUSE WE IMPLEMENTED UNMAGNETISED
	// HOWEVER, WE SHOULD PROBABLY EXPECT THAT IN THE END OUR NU_eHEART IS RELEVANT TO OUTPUT HERE

		// TeV = T.Ti*one_over_kB;
		// Estimate_Ion_Neutral_Cross_sections_d(TeV, &sigma_MT, &sigma_visc); // could easily save one call
		sqrt_T = sqrt(T.Ti); // again not that hard to save one call
		nu_in_visc = our_n.n_n * sigma_visc * sqrt(T.Ti / m_ion + T.Tn / m_n);
		nu_ni_visc = nu_in_visc * (our_n.n / our_n.n_n);
		nu_ii = our_n.n*kB_to_3halves*Get_lnLambda_ion_d(our_n.n, T.Ti) *Nu_ii_Factor /
			(sqrt_T*T.Ti);
				
		p_ita_par_ion_minor[index] = 0.5*our_n.n*T.Ti / (0.3*nu_ii + 0.4*nu_in_visc + 0.000273*nu_eiBar);
			//0.96*our_n.n*T.Ti / nu_ii; // Formulary
		p_nu_ion_minor[index] = 0.3*nu_ii + 0.4*nu_in_visc + 0.000273*nu_eiBar;

		// What applies to viscosity will surely apply to heat conduction also.
		// We'll have to compare Zhdanov with what we have got.

		//nu_nn_visc:
		nu_nn = our_n.n_n * Estimate_Neutral_Neutral_Viscosity_Cross_section_d(T.Tn * one_over_kB)
			* sqrt(T.Tn / m_n);
				
		// !
		
		p_ita_neutral_minor[index] = our_n.n_n*T.Tn / nu_nn; 
		// Not used yet?
		p_nu_nn_visc[index] = nu_nn; // OBVIOUSLY NOT CORRECT.

	}
	else {
		p_ita_par_elec_minor[index] = 0.0;
		p_nu_elec_minor[index] = 0.0;
		p_ita_par_ion_minor[index] = 0.0;
		p_nu_ion_minor[index] = 0.0;
		p_ita_neutral_minor[index] = 0.0;
		p_nu_nn_visc[index] = 0.0;		
	}
}

// Historical record
// We have to reform heat cond so that we use min n but a good T throughout kappa
// We do try to use same n in numer and denom, including in ln Lambda.
// We have too many spills in the routine so we should make a separate ionisation routine
// and try to do species separately.


/*
__global__ void kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly(
	f64 const h_use,
	structural * __restrict__ p_info_minor,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,
	long * __restrict__ izTri_verts,
	char * __restrict__ szPBCtri_verts,
	f64_vec2 * __restrict__ p_cc,

	nvals * __restrict__ p_n_major,
	T3 * __restrict__ p_T_major,
	T3 * __restrict__ p_T_k,
	f64_vec3 * __restrict__ p_B_major,

	f64 * __restrict__ p_kappa_n,
	f64 * __restrict__ p_kappa_i,
	f64 * __restrict__ p_kappa_e,

	f64 * __restrict__ p_nu_i,
	f64 * __restrict__ p_nu_e,

	NTrates * __restrict__ NTadditionrates,
	f64 * __restrict__ p_AreaMajor)
{
	// Think we might as well take kappa_par and nu from triangles really.
	// If n is modelled well then hopefully a nearby high-n does not have a big impact.

	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajorClever]; // 2
	__shared__ f64_vec2 shared_pos[2 * threadsPerTileMajorClever];

	__shared__ f64 shared_T[threadsPerTileMajorClever];      // +3
															 //__shared__ f64 shared_T[threadsPerTileMajorClever];

	__shared__ f64_vec2 shared_B[threadsPerTileMajorClever]; // +2

															 // B is smooth. Unfortunately we have not fitted in Bz here.
															 // In order to do that perhaps rewrite so that variables are overwritten in shared.
															 // We do not need all T and nu in shared at the same time.
															 // This way is easier for NOW.
	__shared__ f64 shared_kappa[threadsPerTileMajorClever * 2];
	__shared__ f64 shared_nu[threadsPerTileMajorClever * 2];

	__shared__ long Indexneigh[MAXNEIGH_d*threadsPerTileMajorClever]; // assume 48 bytes = 4*12 = 6 doubles

	__shared__ char PBCneigh[MAXNEIGH_d*threadsPerTileMajorClever]; // 12 bytes each from L1. Have 42 per thread at 384 threads.
																	// We should at any rate try a major block of size 256. If 1 block will run at a time, so be it.
																	// Leave L1 in case of register overflow into it. <-- don't know how likely - do we only have 31 doubles in registry
																	// regardless # of threads and space? Or can be 63?
	__shared__ char PBCtri[MAXNEIGH_d*threadsPerTileMajorClever];
	// Balance of shared vs L1: 24*256*8 = 48K. That leaves 8 doublesworth in L1 for variables.

	long izTri[MAXNEIGH_d];  // so only 2 doubles left in L1. 31 in registers??

							 // Set threadsPerTileMajorClever to 256.

							 // It would help matters if we get rid of T3. We might as well therefore change to scalar flatpack T.

							 // We are hoping that it works well loading kappa(tri) and that this is not upset by nearby values. Obviously a bit of an experiment.

							 // Does make it somewhat laughable that we go to such efforts to reduce global accesses when we end up overflowing anyway. 
							 // If we can fit 24 doubles/thread in 48K that means we can fit 8 doubles/thread in 16K so that's most of L1 used up.

	long const StartMajor = blockIdx.x*blockDim.x;
	long const EndMajor = StartMajor + blockDim.x;
	long const StartMinor = blockIdx.x*blockDim.x * 2;
	long const EndMinor = StartMinor + blockDim.x * 2;
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX	// 2.5 double

	structural info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];     // 3 double
	shared_pos_verts[threadIdx.x] = info.pos;
#ifdef CENTROID_HEATCONDUCTION
	{
		structural infotemp[2];
		memcpy(infotemp, p_info_minor + 2 * iVertex, 2 * sizeof(structural));
		shared_pos[threadIdx.x * 2] = infotemp[0].pos;
		shared_pos[threadIdx.x * 2 + 1] = infotemp[1].pos;
		f64 tempf64[2];
		memcpy(&(shared_kappa[threadIdx.x * 2]), p_kappa_n + 2 * iVertex, 2 * sizeof(f64));
		// No nu to set for neutrals - not used
	}
#else
	{
		memcpy(&(shared_pos[threadIdx.x * 2]), p_cc + 2 * iVertex, 2 * sizeof(f64_vec2));
		memcpy(&(shared_kappa[threadIdx.x * 2]), p_kappa_n + 2 * iVertex, 2 * sizeof(f64));
	}
#endif


	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {
		shared_B[threadIdx.x] = p_B_major[iVertex].xypart();
		shared_T[threadIdx.x] = p_T_major[iVertex].Tn;
	}
	else {
		// SHOULD NOT BE LOOKING INTO INS.
		// How do we avoid? IMPORTANT
		memset(&(shared_B[threadIdx.x]), 0, sizeof(f64_vec2));
		shared_T[threadIdx.x] = 0.0;
	}


	__syncthreads();

	f64_vec2 grad_T;
	f64 T_anti, T_clock, T_out, T_outk;		// 5
	f64_vec2 pos_clock, pos_anti, pos_out;   // +6
	f64_vec2 B_out;       // +2
	NTrates ourrates;      // +5
	f64 kappa_parallel; // do we use them all at once or can we save 2 doubles here?
	f64 nu;                // 20 there  
	f64_vec2 edge_normal;  // 22
	f64_vec2 endpt_anti;    // 24 .. + 6 from above
	long indexneigh;     // into the 2-double buffer in L1
	f64_vec2 endpt_clock;    // As we only use endpt_anti afterwords we could union endpt_clock with edge_normal
							 // Come back and optimize by checking which things we need in scope at the same time?

	short iNeigh; // only fixed # of addresses so short makes no difference.
	char PBC; // char makes no difference.


	if ((info.flag == INNERMOST) || (info.flag == OUTERMOST))
	{
		// [ Ignore flux into edge of outermost vertex I guess ???]
	}
	else {
		if (info.flag == DOMAIN_VERTEX) {
			// The idea of not sending blocks full of non-domain vertices is another idea. Fiddly with indices.

			// Need this, we are adding on to existing d/dt N,NT :
			memcpy(&ourrates, NTadditionrates + iVertex, sizeof(NTrates));

			memcpy(Indexneigh + MAXNEIGH_d * threadIdx.x,
				pIndexNeigh + MAXNEIGH_d * iVertex,
				MAXNEIGH_d * sizeof(long));
			memcpy(PBCneigh + MAXNEIGH_d * threadIdx.x,
				pPBCNeigh + MAXNEIGH_d * iVertex,
				MAXNEIGH_d * sizeof(char));
			memcpy(PBCtri + MAXNEIGH_d * threadIdx.x,
				szPBCtri_verts + MAXNEIGH_d * iVertex,
				MAXNEIGH_d * sizeof(char));

			// EXPERIMENT WHETHER IT IS FASTER WITH THESE OUTSIDE OR INSIDE THE BRANCH.


			memcpy(izTri, //+ MAXNEIGH_d * threadIdx.x,
				izTri_verts + MAXNEIGH_d * iVertex, MAXNEIGH_d * sizeof(long));

			// Now do Tn:

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_clock = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
				T_clock = shared_T[indexneigh - StartMajor];
#endif
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_clock = info2.pos;
#ifdef BWDSIDET
				T_clock = p_T_major[indexneigh].Tn;
#endif
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == NEEDS_ANTI) {
				pos_clock = Anticlock_rotate2(pos_clock);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_clock = Clockwise_rotate2(pos_clock);
			};
#ifndef BWDSIDET
			T_clock = p_T_k[indexneigh].Tn;
#endif

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_out = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
				T_out = shared_T[indexneigh - StartMajor];
#endif
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_out = info2.pos;
#ifdef BWDSIDET
				T_out = p_T_major[indexneigh].Tn;
#endif
			};
#ifndef BWDSIDET
			T_outk = p_T_k[indexneigh].Tn; // ready for switch around
#endif

			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
			if (PBC == NEEDS_ANTI) {
				pos_out = Anticlock_rotate2(pos_out);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_out = Clockwise_rotate2(pos_out);
			};

			if ((izTri[info.neigh_len - 1] >= StartMinor) && (izTri[info.neigh_len - 1] < EndMinor))
			{
				endpt_clock = shared_pos[izTri[info.neigh_len - 1] - StartMinor];
			}
			else {
#ifdef CENTROID_HEATCONDUCTION
				endpt_clock = p_info_minor[izTri[info.neigh_len - 1]].pos;
#else
				endpt_clock = p_cc[izTri[info.neigh_len - 1]];
#endif
			}
			PBC = PBCtri[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == ROTATE_ME_CLOCKWISE) endpt_clock = Clockwise_d * endpt_clock;
			if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_clock = Anticlockwise_d * endpt_clock;

			if (T_clock == 0.0) {
#ifdef BWDSIDET
				T_clock = 0.5*(shared_T[threadIdx.x] + T_out);
#else
				T_clock = T_outk;
#endif
			};

#pragma unroll MAXNEIGH_d
			for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
			{
				{
					short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNext];
				}
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					pos_anti = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
					T_anti = shared_T[indexneigh - StartMajor];
#endif
				}
				else {
					structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
					pos_anti = info2.pos;
#ifdef BWDSIDET
					T_anti = p_T_major[indexneigh].Tn;
#endif
				};
				if (PBC == NEEDS_ANTI) {
					pos_anti = Anticlock_rotate2(pos_anti);
				};
				if (PBC == NEEDS_CLOCK) {
					pos_anti = Clockwise_rotate2(pos_anti);
				};
#ifndef BWDSIDET
				T_anti = p_T_k[indexneigh].Tn; // Stupid 3-struct

											   // Also need to update T_opp if it was not done already

				indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					T_out = shared_T[indexneigh - StartMajor];
				}
				else {
					T_out = p_T_major[indexneigh].Tn;
				};
#endif

				if (T_anti == 0.0) {
#ifdef BWDSIDET
					T_anti = 0.5*(shared_T[threadIdx.x] + T_out);
#else
					T_anti = T_outk;
#endif					
				}; // So we are receiving 0 then doing this. But how come?

				   // Now let's see
				   // tri 0 has neighs 0 and 1 I'm pretty sure (check....) CHECK

				if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
				{
					endpt_anti = shared_pos[izTri[iNeigh] - StartMinor];
				}
				else {
#ifdef CENTROID_HEATCONDUCTION
					endpt_anti = p_info_minor[izTri[iNeigh]].pos;
					// we should switch back to centroids!!
#else
					endpt_anti = p_cc[izTri[iNeigh]];
#endif
				}
				PBC = PBCtri[MAXNEIGH_d*threadIdx.x + iNeigh];
				if (PBC == ROTATE_ME_CLOCKWISE) endpt_anti = Clockwise_d * endpt_anti;
				if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_anti = Anticlockwise_d * endpt_anti;

				edge_normal.x = (endpt_anti.y - endpt_clock.y);
				edge_normal.y = (endpt_clock.x - endpt_anti.x);

				if (TEST) {
					printf("%d contrib %1.8E \n"
						"pos_anti %1.9E %1.9E pos_out %1.9E %1.9E pos_clock %1.9E %1.9E\n", iVertex,
						0.5*edge_normal.x*THIRD*(pos_anti.x + pos_clock.x
							+ info.pos.x + info.pos.x + pos_out.x + pos_out.x),
						pos_anti.x, pos_anti.y, pos_out.x, pos_out.y, pos_clock.x, pos_clock.y);
				}

				// SMARTY:
				if (pos_out.x*pos_out.x + pos_out.y*pos_out.y >
					DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{



					// How to detect? Loading a load of flags is a killer! We do need to load ... and this is why we should have not made info struct. Def not.

					//// 
					//if (insulator triangle)
					//{
					//	centroid1 = THIRD*(pos_anti + pos_out + info.pos);
					//	// project to radius of insulator
					//	centroid1.project_to_radius(3.44);
					//	// Now dot with unit vectors:
					//	f64_vec2 tempvec2;
					//	tempvec2.x = unit_vec1.x*centroid1.x + unit_vec1.y*centroid1.y;
					//	tempvec2.y = unit_vec2.x*centroid1.x + unit_vec2.y*centroid1.y;
					//	centroid1.x = tempvec2.x;
					//	centroid1.y = tempvec2.y;
					//} else {
					//	// centroid1 = THIRD*(pos_anti_twist + pos_out_twist);
					//	centroid1.x = THIRD*(
					//		  unit_vec1.x*(pos_anti.x - info.pos.x) + unit_vec1.y*(pos_anti.y - info.pos.y)
					//		+ unit_vec1.x*(pos_out.x - info.pos.x) + unit_vec1.y*(pos_out.y - info.pos.y)
					//		);
					//	centroid1.y = THIRD*(
					//		- unit_vec1.y*(pos_anti.x - info.pos.x) + unit_vec1.x*(pos_anti.y - info.pos.y)
					//		- unit_vec1.y*(pos_out.x - info.pos.x) + unit_vec1.x*(pos_out.y - info.pos.y)
					//		);
					//}

					//if (insulator triangle)
					//{
					//	centroid2 = THIRD*(pos_clock + pos_out + info.pos);

					//	// project to radius of insulator
					//} else {

					//}


					kappa_parallel = 0.0;
					if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
					{
						kappa_parallel = 0.5*shared_kappa[izTri[iNeigh] - StartMinor];
					}
					else {
						kappa_parallel = 0.5*p_kappa_n[izTri[iNeigh]];
					}
					{
						short iPrev = iNeigh - 1; if (iPrev < 0) iPrev = info.neigh_len - 1;
						if ((izTri[iPrev] >= StartMinor) && (izTri[iPrev] < EndMinor))
						{
							kappa_parallel += 0.5*shared_kappa[izTri[iPrev] - StartMinor];
						}
						else {
							kappa_parallel += 0.5*p_kappa_n[izTri[iPrev]];
						}
					}

					if ((pos_clock.x*pos_clock.x + pos_clock.y*pos_clock.y <
						DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
						||
						(pos_anti.x*pos_anti.x + pos_anti.y*pos_anti.y <
							DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER))
					{
						f64 edgelen = edge_normal.modulus();

						ourrates.NnTn += TWOTHIRDS * kappa_parallel * edgelen *
							(T_out - shared_T[threadIdx.x]) / (pos_out - info.pos).modulus();
					}
					else {
						f64 Area_quadrilateral = 0.5*(
							(info.pos.x + pos_anti.x)*(info.pos.y - pos_anti.y)
							+ (pos_clock.x + info.pos.x)*(pos_clock.y - info.pos.y)
							+ (pos_out.x + pos_clock.x)*(pos_out.y - pos_clock.y)
							+ (pos_anti.x + pos_out.x)*(pos_anti.y - pos_out.y)
							);

						// When we come to do the other species, make a subroutine.

						grad_T.x = 0.5*( // notice minus
							(shared_T[threadIdx.x] + T_anti)*(info.pos.y - pos_anti.y)
							+ (T_clock + shared_T[threadIdx.x])*(pos_clock.y - info.pos.y)
							+ (T_out + T_clock)*(pos_out.y - pos_clock.y)
							+ (T_anti + T_out)*(pos_anti.y - pos_out.y)
							) / Area_quadrilateral;

						grad_T.y = -0.5*( // notice minus
							(shared_T[threadIdx.x] + T_anti)*(info.pos.x - pos_anti.x)
							+ (T_clock + shared_T[threadIdx.x])*(pos_clock.x - info.pos.x)
							+ (T_out + T_clock)*(pos_out.x - pos_clock.x)
							+ (T_anti + T_out)*(pos_anti.x - pos_out.x)
							) / Area_quadrilateral;

						ourrates.NnTn += TWOTHIRDS * kappa_parallel * grad_T.dot(edge_normal);
					}
				} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

				  // Now go round:	
				endpt_clock = endpt_anti;
				pos_clock = pos_out;
				pos_out = pos_anti;
#ifdef BWDSIDET
				T_clock = T_out;
				T_out = T_anti;
#else
				T_clock = T_outk;
				T_outk = T_anti;
#endif

			}; // next iNeigh

		}; // was it DOMAIN_VERTEX? Do what otherwise?
	}; // was it OUTERMOST/INNERMOST?

	__syncthreads();

	// Did we make sure to include a call to syncthreads every time we carried on to update shared memory data in every other routine?
	// ##################################################################################################################################

	{
		memcpy(&(shared_kappa[threadIdx.x * 2]), p_kappa_i + 2 * iVertex, 2 * sizeof(f64));
		memcpy(&(shared_nu[threadIdx.x * 2]), p_nu_i + 2 * iVertex, 2 * sizeof(f64));
	}
	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {
		shared_T[threadIdx.x] = p_T_major[iVertex].Ti;
		// Notice major inefficiency caused by not making them scalar T arrays
	}
	else {
		shared_T[threadIdx.x] = 0.0;
	}

	__syncthreads();

	if ((info.flag == INNERMOST) || (info.flag == OUTERMOST))
	{
		// [ Ignore flux into edge of outermost vertex I guess ???]
	}
	else {
		if (info.flag == DOMAIN_VERTEX) {

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_clock = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
				T_clock = shared_T[indexneigh - StartMajor];
#endif
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_clock = info2.pos;
#ifdef BWDSIDET
				T_clock = p_T_major[indexneigh].Ti;
#endif
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == NEEDS_ANTI) {
				pos_clock = Anticlock_rotate2(pos_clock);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_clock = Clockwise_rotate2(pos_clock);
			};
#ifndef BWDSIDET
			T_clock = p_T_k[indexneigh].Ti;
#endif

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_out = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
				T_out = shared_T[indexneigh - StartMajor];
#endif
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_out = info2.pos;
#ifdef BWDSIDET
				T_out = p_T_major[indexneigh].Ti;
#endif
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
			if (PBC == NEEDS_ANTI) {
				pos_out = Anticlock_rotate2(pos_out);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_out = Clockwise_rotate2(pos_out);
			};

#ifndef BWDSIDET
			T_outk = p_T_k[indexneigh].Ti;
#endif

			if ((izTri[info.neigh_len - 1] >= StartMinor) && (izTri[info.neigh_len - 1] < EndMinor))
			{
				endpt_clock = shared_pos[izTri[info.neigh_len - 1] - StartMinor];
			}
			else {
#ifdef CENTROID_HEATCONDUCTION
				endpt_clock = p_info_minor[izTri[info.neigh_len - 1]].pos;
#else
				endpt_clock = p_cc[izTri[info.neigh_len - 1]];
#endif
			}
			PBC = PBCtri[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == ROTATE_ME_CLOCKWISE) endpt_clock = Clockwise_d * endpt_clock;
			if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_clock = Anticlockwise_d * endpt_clock;

			if (T_clock == 0.0) {
#ifdef BWDSIDET
				T_clock = 0.5*(shared_T[threadIdx.x] + T_out);
#else
				T_clock = T_outk;
#endif
			};

#pragma unroll MAXNEIGH_d
			for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
			{
				{
					short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNext];
				}
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					pos_anti = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
					T_anti = shared_T[indexneigh - StartMajor];
#endif
				}
				else {
					structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
					pos_anti = info2.pos;
#ifdef BWDSIDET
					T_anti = p_T_major[indexneigh].Ti;
#endif
				};
#ifndef BWDSIDET
				T_anti = p_T_k[indexneigh].Ti;

				indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					T_out = shared_T[indexneigh - StartMajor];
				}
				else {
					T_out = p_T_major[indexneigh].Ti;
				};
#endif
				if (PBC == NEEDS_ANTI) {
					pos_anti = Anticlock_rotate2(pos_anti);
				};
				if (PBC == NEEDS_CLOCK) {
					pos_anti = Clockwise_rotate2(pos_anti);
				};

				if (T_anti == 0.0) {
#ifdef BWDSIDET
					T_anti = 0.5*(shared_T[threadIdx.x] + T_out);
#else
					T_anti = T_outk;
#endif					
				}; // So we are receiving 0 then doing this. But how come?

				if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
				{
					endpt_anti = shared_pos[izTri[iNeigh] - StartMinor];
				}
				else {
#ifdef CENTROID_HEATCONDUCTION
					endpt_anti = p_info_minor[izTri[iNeigh]].pos;
#else
					endpt_anti = p_cc[izTri[iNeigh]];
#endif					
				}
				PBC = PBCtri[MAXNEIGH_d*threadIdx.x + iNeigh];
				if (PBC == ROTATE_ME_CLOCKWISE) endpt_anti = Clockwise_d * endpt_anti;
				if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_anti = Anticlockwise_d * endpt_anti;

				edge_normal.x = (endpt_anti.y - endpt_clock.y);
				edge_normal.y = (endpt_clock.x - endpt_anti.x);

				// SMARTY:
				if (pos_out.x*pos_out.x + pos_out.y*pos_out.y >
					DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					//f64 Area_quadrilateral = 0.5*(
					//	(info.pos.x + pos_anti.x)*(info.pos.y - pos_anti.y)
					//	+ (pos_clock.x + info.pos.x)*(pos_clock.y - info.pos.y)
					//	+ (pos_out.x + pos_clock.x)*(pos_out.y - pos_clock.y)
					//	+ (pos_anti.x + pos_out.x)*(pos_anti.y - pos_out.y)
					//	);
					//grad_T.x = 0.5*(
					//	(shared_T[threadIdx.x] + T_anti)*(info.pos.y - pos_anti.y)
					//	+ (T_clock + shared_T[threadIdx.x])*(pos_clock.y - info.pos.y)
					//	+ (T_out + T_clock)*(pos_out.y - pos_clock.y)
					//	+ (T_anti + T_out)*(pos_anti.y - pos_out.y)
					//	) / Area_quadrilateral;
					//grad_T.y = -0.5*( // notice minus
					//	(shared_T[threadIdx.x] + T_anti)*(info.pos.x - pos_anti.x)
					//	+ (T_clock + shared_T[threadIdx.x])*(pos_clock.x - info.pos.x)
					//	+ (T_out + T_clock)*(pos_out.x - pos_clock.x)
					//	+ (T_anti + T_out)*(pos_anti.x - pos_out.x)
					//	) / Area_quadrilateral;

					kappa_parallel = 0.0;
					f64 nu;
					if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
					{
						kappa_parallel = 0.5*shared_kappa[izTri[iNeigh] - StartMinor];
						nu = 0.5*shared_nu[izTri[iNeigh] - StartMinor];
					}
					else {
						kappa_parallel = 0.5*p_kappa_i[izTri[iNeigh]];
						nu = 0.5*p_nu_i[izTri[iNeigh]];
					};
					{
						short iPrev = iNeigh - 1; if (iPrev < 0) iPrev = info.neigh_len - 1;
						if ((izTri[iPrev] >= StartMinor) && (izTri[iPrev] < EndMinor))
						{
							kappa_parallel += 0.5*shared_kappa[izTri[iPrev] - StartMinor];
							nu += 0.5*shared_nu[izTri[iPrev] - StartMinor];
						}
						else {
							kappa_parallel += 0.5*p_kappa_i[izTri[iPrev]];
							nu += 0.5*p_nu_i[izTri[iPrev]];
						}
					}

					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						B_out = shared_B[indexneigh - StartMajor];
					}
					else {
						f64_vec3 B_out3 = p_B_major[indexneigh];
						B_out = B_out3.xypart();
					}
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if (PBC == NEEDS_ANTI) 	B_out = Anticlock_rotate2(B_out);
					if (PBC == NEEDS_CLOCK)	B_out = Clockwise_rotate2(B_out);

					//kappa.xx = kappa_parallel * (nu_eHeart*nu_eHeart + omega.x*omega.x) / (nu_eHeart * nu_eHeart + omega_sq);
					//kappa.xy = kappa_parallel * (omega.x*omega.y - nu_eHeart *omega.z) / (nu_eHeart * nu_eHeart + omega_sq);
					//kappa.yx = kappa_parallel * (omega.x*omega.y + nu_eHeart * omega.z) / (nu_eHeart * nu_eHeart + omega_sq);
					//kappa.yy = kappa_parallel * (omega.y*omega.y + nu_eHeart * nu_eHeart) / (nu_eHeart * nu_eHeart + omega_sq);

					{ // scoping brace
						f64_vec3 omega = Make3(qoverMc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qoverMc);
						// PROBABLY ALWAYS SPILLED INTO GLOBAL -- WHAT CAN WE DO?
						//
						//						ourrates.NiTi += TWOTHIRDS * kappa_parallel *(
						//							edge_normal.x*((nu*nu + omega.x*omega.x)*grad_T.x + (omega.x*omega.y + nu * omega.z)*grad_T.y)
						//							+ edge_normal.y*((omega.x*omega.y - nu * omega.z)*grad_T.x + (omega.y*omega.y + nu * nu)*grad_T.y)
						//							) / (nu * nu + omega.dot(omega));
						f64 edgelen = edge_normal.modulus();
						f64 delta_out = sqrt((info.pos.x - pos_out.x)*(info.pos.x - pos_out.x) + (info.pos.y - pos_out.y)*(info.pos.y - pos_out.y));

						ourrates.NiTi += TWOTHIRDS * kappa_parallel *  (T_out - shared_T[threadIdx.x]) *
							(nu*nu*edgelen*edgelen + omega.dotxy(edge_normal)*omega.dotxy(edge_normal))
							/ (delta_out*edgelen *(nu * nu + omega.dot(omega)));


						if (TEST) printf("%d iNeigh %d kappa_ion %1.8E nu %1.8E |o| %1.8E contrib %1.8E \n",
							iVertex, iNeigh, kappa_parallel, nu,
							omega.modulus(),
							TWOTHIRDS * kappa_parallel *(
								edge_normal.x*((nu*nu + omega.x*omega.x)*grad_T.x + (omega.x*omega.y + nu * omega.z)*grad_T.y)
								+ edge_normal.y*((omega.x*omega.y - nu * omega.z)*grad_T.x + (omega.y*omega.y + nu * nu)*grad_T.y)
								) / (nu * nu + omega.dot(omega))
						);

					}
				} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

				  // Now go round:	
				endpt_clock = endpt_anti;
				pos_clock = pos_out;
				pos_out = pos_anti;
#ifndef BWDSIDET
				T_clock = T_outk;
				T_outk = T_anti;
#else
				T_clock = T_out;
				T_out = T_anti;
#endif

			}; // next iNeigh

		}; // was it DOMAIN_VERTEX? Do what otherwise?
	}; // was it OUTERMOST/INNERMOST?


	__syncthreads();

	// Did we make sure to include a call to syncthreads every time we carried on to update shared memory data in every other routine?
	// ##################################################################################################################################

	{
		memcpy(&(shared_kappa[threadIdx.x * 2]), p_kappa_e + 2 * iVertex, 2 * sizeof(f64));
		memcpy(&(shared_nu[threadIdx.x * 2]), p_nu_e + 2 * iVertex, 2 * sizeof(f64));
	}
	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {
		shared_T[threadIdx.x] = p_T_major[iVertex].Te;
	}
	else {
		shared_T[threadIdx.x] = 0.0;
	}

	__syncthreads();


	if ((info.flag == INNERMOST) || (info.flag == OUTERMOST))
	{
		// [ Ignore flux into edge of outermost vertex I guess ???]
	}
	else {
		if (info.flag == DOMAIN_VERTEX) {

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_clock = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
				T_clock = shared_T[indexneigh - StartMajor];
#endif
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_clock = info2.pos;
#ifdef BWDSIDET
				T_clock = p_T_major[indexneigh].Te;
#endif
			};
#ifndef BWDSIDET
			T_clock = p_T_k[indexneigh].Te;
#endif

			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == NEEDS_ANTI) {
				pos_clock = Anticlock_rotate2(pos_clock);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_clock = Clockwise_rotate2(pos_clock);
			};

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_out = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
				T_out = shared_T[indexneigh - StartMajor];
#endif
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_out = info2.pos;
#ifdef BWDSIDET
				T_out = p_T_major[indexneigh].Te;
#endif
			};
#ifndef BWDSIDET
			T_outk = p_T_k[indexneigh].Te;
#endif
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
			if (PBC == NEEDS_ANTI) {
				pos_out = Anticlock_rotate2(pos_out);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_out = Clockwise_rotate2(pos_out);
			};

			if ((izTri[info.neigh_len - 1] >= StartMinor) && (izTri[info.neigh_len - 1] < EndMinor))
			{
				endpt_clock = shared_pos[izTri[info.neigh_len - 1] - StartMinor];
			}
			else {
#ifdef CENTROID_HEATCONDUCTION
				endpt_clock = p_info_minor[izTri[info.neigh_len - 1]].pos;
#else
				endpt_clock = p_cc[izTri[info.neigh_len - 1]];
#endif
			}
			PBC = PBCtri[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == ROTATE_ME_CLOCKWISE) endpt_clock = Clockwise_d * endpt_clock;
			if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_clock = Anticlockwise_d * endpt_clock;

			if (T_clock == 0.0) {
#ifdef BWDSIDET
				T_clock = 0.5*(shared_T[threadIdx.x] + T_out);
#else
				T_clock = T_outk;
#endif
			};

#pragma unroll MAXNEIGH_d
			for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
			{
				{
					short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNext];
				}
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					pos_anti = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
					T_anti = shared_T[indexneigh - StartMajor];
#endif
				}
				else {
					structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
					pos_anti = info2.pos;
#ifdef BWDSIDET
					T_anti = p_T_major[indexneigh].Te;
#endif
				};
#ifndef BWDSIDET
				T_anti = p_T_k[indexneigh].Te;
				indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					T_out = shared_T[indexneigh - StartMajor];
				}
				else {
					T_out = p_T_major[indexneigh].Te;
				}
#endif
				if (PBC == NEEDS_ANTI) {
					pos_anti = Anticlock_rotate2(pos_anti);
				};
				if (PBC == NEEDS_CLOCK) {
					pos_anti = Clockwise_rotate2(pos_anti);
				};

				if (T_anti == 0.0) {
#ifdef BWDSIDET
					T_anti = 0.5*(shared_T[threadIdx.x] + T_out);
#else
					T_anti = T_outk;
#endif					
				}; // So we are receiving 0 then doing this. But how come?

				if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
				{
					endpt_anti = shared_pos[izTri[iNeigh] - StartMinor];

					if (TEST) {
						printf("%d : %d endpt_anti %1.9E %1.9E SHARED endpt_clock %1.9E %1.9E izTri[iNeigh] %d\n",
							iVertex, iNeigh, endpt_anti.x, endpt_anti.y, endpt_clock.x, endpt_clock.y, izTri[iNeigh]);
					}
				}
				else {
#ifdef CENTROID_HEATCONDUCTION
					endpt_anti = p_info_minor[izTri[iNeigh]].pos;
#else
					endpt_anti = p_cc[izTri[iNeigh]];
#endif

					if (TEST) {
						printf("%d : %d endpt_anti %1.9E %1.9E GLOBAL endpt_clock %1.9E %1.9E izTri[iNeigh] %d\n",
							iVertex, iNeigh, endpt_anti.x, endpt_anti.y, endpt_clock.x, endpt_clock.y, izTri[iNeigh]);
					}
				}
				PBC = PBCtri[MAXNEIGH_d*threadIdx.x + iNeigh];
				if (PBC == ROTATE_ME_CLOCKWISE) endpt_anti = Clockwise_d * endpt_anti;
				if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_anti = Anticlockwise_d * endpt_anti;

				// It decided to rotate something it shouldn't oughta. Rotated tri 23600 = tri 2 for 11582.

				edge_normal.x = (endpt_anti.y - endpt_clock.y);
				edge_normal.y = (endpt_clock.x - endpt_anti.x);

				// SMARTY:
				if (pos_out.x*pos_out.x + pos_out.y*pos_out.y >
					DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{

					// f64 grad_out = (T_out - shared_T[threadIdx.x]) / delta_0out;


					//f64 Area_quadrilateral = 0.5*(
					//	(info.pos.x + pos_anti.x)*(info.pos.y - pos_anti.y)
					//	+ (pos_clock.x + info.pos.x)*(pos_clock.y - info.pos.y)
					//	+ (pos_out.x + pos_clock.x)*(pos_out.y - pos_clock.y)
					//	+ (pos_anti.x + pos_out.x)*(pos_anti.y - pos_out.y)
					//	);
					//grad_T.x = 0.5*(
					//	(shared_T[threadIdx.x] + T_anti)*(info.pos.y - pos_anti.y)
					//	+ (T_clock + shared_T[threadIdx.x])*(pos_clock.y - info.pos.y)
					//	+ (T_out + T_clock)*(pos_out.y - pos_clock.y)
					//	+ (T_anti + T_out)*(pos_anti.y - pos_out.y)
					//	) / Area_quadrilateral;
					//grad_T.y = -0.5*( // notice minus
					//	(shared_T[threadIdx.x] + T_anti)*(info.pos.x - pos_anti.x)
					//	+ (T_clock + shared_T[threadIdx.x])*(pos_clock.x - info.pos.x)
					//	+ (T_out + T_clock)*(pos_out.x - pos_clock.x)
					//	+ (T_anti + T_out)*(pos_anti.x - pos_out.x)
					//	) / Area_quadrilateral;

					kappa_parallel = 0.0;
					f64 nu;
					if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
					{
						kappa_parallel = 0.5*shared_kappa[izTri[iNeigh] - StartMinor];
						nu = 0.5*shared_nu[izTri[iNeigh] - StartMinor];
					}
					else {
						kappa_parallel = 0.5*p_kappa_e[izTri[iNeigh]];
						nu = 0.5*p_nu_e[izTri[iNeigh]];
					};

					if (TEST) printf("izTri %d kappa_par %1.9E \n",
						izTri[iNeigh], p_kappa_e[izTri[iNeigh]]);

					{
						short iPrev = iNeigh - 1; if (iPrev < 0) iPrev = info.neigh_len - 1;
						if ((izTri[iPrev] >= StartMinor) && (izTri[iPrev] < EndMinor))
						{
							kappa_parallel += 0.5*shared_kappa[izTri[iPrev] - StartMinor];
							nu += 0.5*shared_nu[izTri[iPrev] - StartMinor];
						}
						else {
							kappa_parallel += 0.5*p_kappa_e[izTri[iPrev]];
							nu += 0.5*p_nu_e[izTri[iPrev]];
						}
					}

					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						B_out = shared_B[indexneigh - StartMajor];
					}
					else {
						f64_vec3 B_out3 = p_B_major[indexneigh];
						B_out = B_out3.xypart();
					}
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if (PBC == NEEDS_ANTI) 	B_out = Anticlock_rotate2(B_out);
					if (PBC == NEEDS_CLOCK)	B_out = Clockwise_rotate2(B_out);

					//kappa.xx = kappa_parallel * (nu_eHeart*nu_eHeart + omega.x*omega.x) / (nu_eHeart * nu_eHeart + omega_sq);
					//kappa.xy = kappa_parallel * (omega.x*omega.y - nu_eHeart *omega.z) / (nu_eHeart * nu_eHeart + omega_sq);
					//kappa.yx = kappa_parallel * (omega.x*omega.y + nu_eHeart * omega.z) / (nu_eHeart * nu_eHeart + omega_sq);
					//kappa.yy = kappa_parallel * (omega.y*omega.y + nu_eHeart * nu_eHeart) / (nu_eHeart * nu_eHeart + omega_sq);

					{ // scoping brace
						f64_vec3 omega = Make3(qovermc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qovermc);
						// PROBABLY ALWAYS SPILLED INTO GLOBAL -- WHAT CAN WE DO?

						//		ourrates.NeTe += TWOTHIRDS * kappa_parallel *(
						//			edge_normal.x*((nu*nu + omega.x*omega.x)*grad_T.x + (omega.x*omega.y - nu * omega.z)*grad_T.y)
						//			+ edge_normal.y*((omega.x*omega.y + nu * omega.z)*grad_T.x + (omega.y*omega.y + nu * nu)*grad_T.y)
						//			) / (nu * nu + omega.dot(omega));
						f64 edgelen = edge_normal.modulus();
						f64 delta_out = sqrt((info.pos.x - pos_out.x)*(info.pos.x - pos_out.x) + (info.pos.y - pos_out.y)*(info.pos.y - pos_out.y));

						ourrates.NeTe += TWOTHIRDS * kappa_parallel *  (T_out - shared_T[threadIdx.x]) *
							(nu*nu*edgelen*edgelen + omega.dotxy(edge_normal)*omega.dotxy(edge_normal))
							/ (delta_out*edgelen*(nu * nu + omega.dot(omega)));

						// Expensive debug: remove!

						if (TESTHEAT2) printf(
							"iVertex %d iNeigh %d %d contribNeTe %1.9E edge_normal %1.8E %1.8E \n"
							"T %1.9E Tout %1.9E T_anti %1.9E T_clock %1.9E\n"
							"   kappa_par %1.9E nu %1.9E |omega| %1.9E Area %1.9E\n"
							"our_n %1.9E our n_n %1.9E nearby n %1.9E %1.9E\n"
							"pos %1.8E %1.8E opp %1.8E %1.8E anti %1.8E %1.8E clock %1.8E %1.8E\n"
							"omega %1.8E %1.8E grad_T %1.9E %1.9E \n"
							"=================================================\n",
							iVertex, iNeigh, indexneigh,
							TWOTHIRDS * kappa_parallel *(
								edge_normal.x*((nu*nu + omega.x*omega.x)*grad_T.x + (omega.x*omega.y - nu * omega.z)*grad_T.y)
								+ edge_normal.y*((omega.x*omega.y + nu * omega.z)*grad_T.x + (omega.y*omega.y + nu * nu)*grad_T.y)
								) / (nu * nu + omega.dot(omega)),
							edge_normal.x, edge_normal.y, shared_T[threadIdx.x], T_out, T_anti, T_clock,
							kappa_parallel, nu, sqrt(omega.dot(omega)),
							p_AreaMajor[iVertex],
							p_n_major[iVertex].n, p_n_major[iVertex].n_n, p_n_major[indexneigh].n, p_n_major[indexneigh].n_n,
							info.pos.x, info.pos.y, pos_out.x, pos_out.y, pos_anti.x, pos_anti.y, pos_clock.x, pos_clock.y,
							omega.x, omega.y, grad_T.x, grad_T.y);


					}
				} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

				  // Now go round:	
				endpt_clock = endpt_anti;
				pos_clock = pos_out;
				pos_out = pos_anti;
#ifdef BWDSIDET
				T_clock = T_out;
				T_out = T_anti;
#else
				T_clock = T_outk;
				T_outk = T_anti;
#endif

			}; // next iNeigh

			memcpy(NTadditionrates + iVertex, &ourrates, sizeof(NTrates));

		}; // was it DOMAIN_VERTEX? Do what otherwise?
	}; // was it OUTERMOST/INNERMOST?

}*/

__global__ void kernelAccumulateDiffusiveHeatRate_new_Longitudinalonly_scalarT(
	structural * __restrict__ p_info_minor,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,
	long * __restrict__ izTri_verts,
	char * __restrict__ szPBCtri_verts,
	f64_vec2 * __restrict__ p_cc,

	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p_T_n, f64 * __restrict__ p_T_i, f64 * __restrict__ p_T_e,
	T3 * __restrict__ p_T_k,
	f64_vec3 * __restrict__ p_B_major,

	f64 * __restrict__ p_kappa_n,
	f64 * __restrict__ p_kappa_i,
	f64 * __restrict__ p_kappa_e,

	f64 * __restrict__ p_nu_i,
	f64 * __restrict__ p_nu_e,

	NTrates * __restrict__ NTadditionrates,
	f64 * __restrict__ p_AreaMajor)
{
	// Think we might as well take kappa_par and nu from triangles really.
	// If n is modelled well then hopefully a nearby high-n does not have a big impact.

	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajorClever]; // 2

	// DO NOT WANT:
	__shared__ f64_vec2 shared_pos[2 * threadsPerTileMajorClever];

	__shared__ f64 shared_T[threadsPerTileMajorClever];      // +3
															 //__shared__ f64 shared_T[threadsPerTileMajorClever];

	__shared__ f64_vec2 shared_B[threadsPerTileMajorClever]; // +2

															 // B is smooth. Unfortunately we have not fitted in Bz here.
															 // In order to do that perhaps rewrite so that variables are overwritten in shared.
															 // We do not need all T and nu in shared at the same time.
															 // This way is easier for NOW.
	__shared__ f64 shared_kappa[threadsPerTileMajorClever * 2];
	__shared__ f64 shared_nu[threadsPerTileMajorClever * 2];

	__shared__ long Indexneigh[MAXNEIGH_d*threadsPerTileMajorClever]; // assume 48 bytes = 4*12 = 6 doubles

	__shared__ char PBCneigh[MAXNEIGH_d*threadsPerTileMajorClever]; // 12 bytes each from L1. Have 42 per thread at 384 threads.
																	// We should at any rate try a major block of size 256. If 1 block will run at a time, so be it.
																	// Leave L1 in case of register overflow into it. <-- don't know how likely - do we only have 31 doubles in registry
																	// regardless # of threads and space? Or can be 63?
	__shared__ char PBCtri[MAXNEIGH_d*threadsPerTileMajorClever];
	// Balance of shared vs L1: 24*256*8 = 48K. That leaves 8 doublesworth in L1 for variables.

	long izTri[MAXNEIGH_d];  // so only 2 doubles left in L1. 31 in registers??

							 // Set threadsPerTileMajorClever to 256.

							 // It would help matters if we get rid of T3. We might as well therefore change to scalar flatpack T.

							 // We are hoping that it works well loading kappa(tri) and that this is not upset by nearby values. Obviously a bit of an experiment.

							 // Does make it somewhat laughable that we go to such efforts to reduce global accesses when we end up overflowing anyway. 
							 // If we can fit 24 doubles/thread in 48K that means we can fit 8 doubles/thread in 16K so that's most of L1 used up.

	long const StartMajor = blockIdx.x*blockDim.x;
	long const EndMajor = StartMajor + blockDim.x;
	long const StartMinor = blockIdx.x*blockDim.x * 2;
	long const EndMinor = StartMinor + blockDim.x * 2;
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX	// 2.5 double

	structural info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];     // 3 double
	shared_pos_verts[threadIdx.x] = info.pos;
#ifdef CENTROID_HEATCONDUCTION
	{
		structural infotemp[2];
		memcpy(infotemp, p_info_minor + 2 * iVertex, 2 * sizeof(structural));
		shared_pos[threadIdx.x * 2] = infotemp[0].pos;
		shared_pos[threadIdx.x * 2 + 1] = infotemp[1].pos;
		f64 tempf64[2];
		memcpy(&(shared_kappa[threadIdx.x * 2]), p_kappa_n + 2 * iVertex, 2 * sizeof(f64));
		// No nu to set for neutrals - not used
	}
#else
	{
		memcpy(&(shared_pos[threadIdx.x * 2]), p_cc + 2 * iVertex, 2 * sizeof(f64_vec2));
		memcpy(&(shared_kappa[threadIdx.x * 2]), p_kappa_n + 2 * iVertex, 2 * sizeof(f64));
	}
#endif


	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {
		shared_B[threadIdx.x] = p_B_major[iVertex].xypart();
		shared_T[threadIdx.x] = p_T_n[iVertex];

	}
	else {
		// SHOULD NOT BE LOOKING INTO INS.
		// How do we avoid?
		memset(&(shared_B[threadIdx.x]), 0, sizeof(f64_vec2));
		shared_T[threadIdx.x] = 0.0;
	}
	
	__syncthreads();

	f64_vec2 grad_T;
	f64 T_anti, T_clock, T_out, T_outk;		// 5
	f64_vec2 pos_clock, pos_anti, pos_out;   // +6
	f64_vec2 B_out;       // +2
	NTrates ourrates;      // +5
	f64 kappa_parallel; // do we use them all at once or can we save 2 doubles here?
	f64 nu;                // 20 there  
	f64_vec2 edge_normal;  // 22
	f64_vec2 endpt_anti;    // 24 .. + 6 from above
	long indexneigh;     // into the 2-double buffer in L1
	f64_vec2 endpt_clock;    // As we only use endpt_anti afterwords we could union endpt_clock with edge_normal
							 // Come back and optimize by checking which things we need in scope at the same time?

	short iNeigh; // only fixed # of addresses so short makes no difference.
	char PBC; // char makes no difference.


	if ((info.flag == INNERMOST) || (info.flag == OUTERMOST))
	{
		// [ Ignore flux into edge of outermost vertex I guess ???]
	}
	else {
		if (info.flag == DOMAIN_VERTEX) {
			// The idea of not sending blocks full of non-domain vertices is another idea. Fiddly with indices.

			// Need this, we are adding on to existing d/dt N,NT :
			memcpy(&ourrates, NTadditionrates + iVertex, sizeof(NTrates));

			memcpy(Indexneigh + MAXNEIGH_d * threadIdx.x,
				pIndexNeigh + MAXNEIGH_d * iVertex,
				MAXNEIGH_d * sizeof(long));
			memcpy(PBCneigh + MAXNEIGH_d * threadIdx.x,
				pPBCNeigh + MAXNEIGH_d * iVertex,
				MAXNEIGH_d * sizeof(char));
			memcpy(PBCtri + MAXNEIGH_d * threadIdx.x,
				szPBCtri_verts + MAXNEIGH_d * iVertex,
				MAXNEIGH_d * sizeof(char));

			// EXPERIMENT WHETHER IT IS FASTER WITH THESE OUTSIDE OR INSIDE THE BRANCH.


			memcpy(izTri, //+ MAXNEIGH_d * threadIdx.x,
				izTri_verts + MAXNEIGH_d * iVertex, MAXNEIGH_d * sizeof(long));

			// Now do Tn:

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_clock = shared_pos_verts[indexneigh - StartMajor];
				T_clock = shared_T[indexneigh - StartMajor];
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_clock = info2.pos;
				T_clock = p_T_n[indexneigh];
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == NEEDS_ANTI) {
				pos_clock = Anticlock_rotate2(pos_clock);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_clock = Clockwise_rotate2(pos_clock);
			};

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_out = shared_pos_verts[indexneigh - StartMajor];
				T_out = shared_T[indexneigh - StartMajor];
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_out = info2.pos;
				T_out = p_T_n[indexneigh]; // saved nothing here, only in loading
			};

			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
			if (PBC == NEEDS_ANTI) {
				pos_out = Anticlock_rotate2(pos_out);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_out = Clockwise_rotate2(pos_out);
			};

			if ((izTri[info.neigh_len - 1] >= StartMinor) && (izTri[info.neigh_len - 1] < EndMinor))
			{
				endpt_clock = shared_pos[izTri[info.neigh_len - 1] - StartMinor];
			}
			else {
#ifdef CENTROID_HEATCONDUCTION
				endpt_clock = p_info_minor[izTri[info.neigh_len - 1]].pos;
#else
				endpt_clock = p_cc[izTri[info.neigh_len - 1]];
#endif
			}
			PBC = PBCtri[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == ROTATE_ME_CLOCKWISE) endpt_clock = Clockwise_d * endpt_clock;
			if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_clock = Anticlockwise_d * endpt_clock;

			if (T_clock == 0.0) {
				T_clock = 0.5*(shared_T[threadIdx.x] + T_out);
			};

#pragma unroll MAXNEIGH_d
			for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
			{
				{
					short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNext];
				}
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					pos_anti = shared_pos_verts[indexneigh - StartMajor];
					T_anti = shared_T[indexneigh - StartMajor];
				}
				else {
					structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
					pos_anti = info2.pos;
					T_anti = p_T_n[indexneigh];
				};
				if (PBC == NEEDS_ANTI) {
					pos_anti = Anticlock_rotate2(pos_anti);
				};
				if (PBC == NEEDS_CLOCK) {
					pos_anti = Clockwise_rotate2(pos_anti);
				};

				if (T_anti == 0.0) {
					T_anti = 0.5*(shared_T[threadIdx.x] + T_out);
				}; // So we are receiving 0 then doing this. But how come?

				   // Now let's see
				   // tri 0 has neighs 0 and 1 I'm pretty sure (check....) CHECK

				if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
				{
					endpt_anti = shared_pos[izTri[iNeigh] - StartMinor];
				}
				else {
#ifdef CENTROID_HEATCONDUCTION
					endpt_anti = p_info_minor[izTri[iNeigh]].pos;
					// we should switch back to centroids!!
#else
					endpt_anti = p_cc[izTri[iNeigh]];
#endif
				}
				PBC = PBCtri[MAXNEIGH_d*threadIdx.x + iNeigh];
				if (PBC == ROTATE_ME_CLOCKWISE) endpt_anti = Clockwise_d * endpt_anti;
				if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_anti = Anticlockwise_d * endpt_anti;

				edge_normal.x = (endpt_anti.y - endpt_clock.y);
				edge_normal.y = (endpt_clock.x - endpt_anti.x);

				if (TEST) {
					printf("%d contrib %1.8E \n"
						"pos_anti %1.9E %1.9E pos_out %1.9E %1.9E pos_clock %1.9E %1.9E\n", iVertex,
						0.5*edge_normal.x*THIRD*(pos_anti.x + pos_clock.x
							+ info.pos.x + info.pos.x + pos_out.x + pos_out.x),
						pos_anti.x, pos_anti.y, pos_out.x, pos_out.y, pos_clock.x, pos_clock.y);
				}

				// SMARTY:
				if (pos_out.x*pos_out.x + pos_out.y*pos_out.y >
					DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{


					kappa_parallel = 0.0;
					if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
					{
						kappa_parallel = 0.5*shared_kappa[izTri[iNeigh] - StartMinor];
					}
					else {
						kappa_parallel = 0.5*p_kappa_n[izTri[iNeigh]];
					}
					{
						short iPrev = iNeigh - 1; if (iPrev < 0) iPrev = info.neigh_len - 1;
						if ((izTri[iPrev] >= StartMinor) && (izTri[iPrev] < EndMinor))
						{
							kappa_parallel += 0.5*shared_kappa[izTri[iPrev] - StartMinor];
						}
						else {
							kappa_parallel += 0.5*p_kappa_n[izTri[iPrev]];
						}
					}

					if ((pos_clock.x*pos_clock.x + pos_clock.y*pos_clock.y <
						DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
						||
						(pos_anti.x*pos_anti.x + pos_anti.y*pos_anti.y <
							DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER))
					{
						f64 edgelen = edge_normal.modulus();

						ourrates.NnTn += TWOTHIRDS * kappa_parallel * edgelen *
							(T_out - shared_T[threadIdx.x]) / (pos_out - info.pos).modulus();
					}
					else {
						f64 Area_quadrilateral = 0.5*(
							(info.pos.x + pos_anti.x)*(info.pos.y - pos_anti.y)
							+ (pos_clock.x + info.pos.x)*(pos_clock.y - info.pos.y)
							+ (pos_out.x + pos_clock.x)*(pos_out.y - pos_clock.y)
							+ (pos_anti.x + pos_out.x)*(pos_anti.y - pos_out.y)
							);

						// When we come to do the other species, make a subroutine.

						grad_T.x = 0.5*( // notice minus
							(shared_T[threadIdx.x] + T_anti)*(info.pos.y - pos_anti.y)
							+ (T_clock + shared_T[threadIdx.x])*(pos_clock.y - info.pos.y)
							+ (T_out + T_clock)*(pos_out.y - pos_clock.y)
							+ (T_anti + T_out)*(pos_anti.y - pos_out.y)
							) / Area_quadrilateral;

						grad_T.y = -0.5*( // notice minus
							(shared_T[threadIdx.x] + T_anti)*(info.pos.x - pos_anti.x)
							+ (T_clock + shared_T[threadIdx.x])*(pos_clock.x - info.pos.x)
							+ (T_out + T_clock)*(pos_out.x - pos_clock.x)
							+ (T_anti + T_out)*(pos_anti.x - pos_out.x)
							) / Area_quadrilateral;

						ourrates.NnTn += TWOTHIRDS * kappa_parallel * grad_T.dot(edge_normal);
					}
					// This is correct, grad T in same coordinates as edge_normal...

				} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

				  // Now go round:	
				endpt_clock = endpt_anti;
				pos_clock = pos_out;
				pos_out = pos_anti;
				T_clock = T_out;
				T_out = T_anti;
			}; // next iNeigh

		}; // was it DOMAIN_VERTEX? Do what otherwise?
	}; // was it OUTERMOST/INNERMOST?

	__syncthreads();

	// Did we make sure to include a call to syncthreads every time we carried on to update shared memory data in every other routine?
	// ##################################################################################################################################
	
	
#pragma unroll
	for (int iSpecies = 1; iSpecies < 3; iSpecies++)
	{
		if (iSpecies == 1)
		{
			memcpy(&(shared_kappa[threadIdx.x * 2]), p_kappa_i + 2 * iVertex, 2 * sizeof(f64));
			memcpy(&(shared_nu[threadIdx.x * 2]), p_nu_i + 2 * iVertex, 2 * sizeof(f64));
			if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {
				shared_T[threadIdx.x] = p_T_i[iVertex];
				// Notice major inefficiency caused by not making them scalar T arrays
			}
			else {
				shared_T[threadIdx.x] = 0.0;
			}
		}
		else {
			memcpy(&(shared_kappa[threadIdx.x * 2]), p_kappa_e + 2 * iVertex, 2 * sizeof(f64));
			memcpy(&(shared_nu[threadIdx.x * 2]), p_nu_e + 2 * iVertex, 2 * sizeof(f64));
			if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {
				shared_T[threadIdx.x] = p_T_e[iVertex];
				// Notice major inefficiency caused by not making them scalar T arrays
			}
			else {
				shared_T[threadIdx.x] = 0.0;
			}
		};
		// Maybe this alone means combining the ion & electron code was stupid. Maybe it can't make contig access.
		
		__syncthreads();

		if ((info.flag == INNERMOST) || (info.flag == OUTERMOST))
		{
			// [ Ignore flux into edge of outermost vertex I guess ???]
		}
		else {
			if (info.flag == DOMAIN_VERTEX) {

				indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					pos_clock = shared_pos_verts[indexneigh - StartMajor];
					T_clock = shared_T[indexneigh - StartMajor];
				}
				else {
					structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
					pos_clock = info2.pos;
					if (iSpecies == 1) {
						T_clock = p_T_i[indexneigh];
					}
					else {
						T_clock = p_T_e[indexneigh];
					};
				};
				PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
				if (PBC == NEEDS_ANTI) {
					pos_clock = Anticlock_rotate2(pos_clock);
				};
				if (PBC == NEEDS_CLOCK) {
					pos_clock = Clockwise_rotate2(pos_clock);
				};

				indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					pos_out = shared_pos_verts[indexneigh - StartMajor];
					T_out = shared_T[indexneigh - StartMajor];
				}
				else {
					structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
					pos_out = info2.pos;
					if (iSpecies == 1) {
						T_out = p_T_i[indexneigh];
					}
					else {
						T_out = p_T_e[indexneigh];
					};
				};
				PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
				if (PBC == NEEDS_ANTI) {
					pos_out = Anticlock_rotate2(pos_out);
				};
				if (PBC == NEEDS_CLOCK) {
					pos_out = Clockwise_rotate2(pos_out);
				};

				if ((izTri[info.neigh_len - 1] >= StartMinor) && (izTri[info.neigh_len - 1] < EndMinor))
				{
					endpt_clock = shared_pos[izTri[info.neigh_len - 1] - StartMinor];
				}
				else {
#ifdef CENTROID_HEATCONDUCTION
					endpt_clock = p_info_minor[izTri[info.neigh_len - 1]].pos;
#else
					endpt_clock = p_cc[izTri[info.neigh_len - 1]];
#endif
				}
				PBC = PBCtri[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
				if (PBC == ROTATE_ME_CLOCKWISE) endpt_clock = Clockwise_d * endpt_clock;
				if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_clock = Anticlockwise_d * endpt_clock;

				if (T_clock == 0.0) {
					T_clock = 0.5*(shared_T[threadIdx.x] + T_out);
				};

#pragma unroll MAXNEIGH_d
				for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
				{
					{
						short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
						indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
						PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNext];
					}
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						pos_anti = shared_pos_verts[indexneigh - StartMajor];
						T_anti = shared_T[indexneigh - StartMajor];
					}
					else {
						structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
						pos_anti = info2.pos;
						if (iSpecies == 1)
						{
							T_anti = p_T_i[indexneigh];
						}
						else {
							T_anti = p_T_e[indexneigh];
						};
					};
					if (PBC == NEEDS_ANTI) {
						pos_anti = Anticlock_rotate2(pos_anti);
					};
					if (PBC == NEEDS_CLOCK) {
						pos_anti = Clockwise_rotate2(pos_anti);
					};

					if (T_anti == 0.0) {
						T_anti = 0.5*(shared_T[threadIdx.x] + T_out);
					}; // So we are receiving 0 then doing this. But how come?

					if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
					{
						endpt_anti = shared_pos[izTri[iNeigh] - StartMinor];
					}
					else {
#ifdef CENTROID_HEATCONDUCTION
						endpt_anti = p_info_minor[izTri[iNeigh]].pos;
#else
						endpt_anti = p_cc[izTri[iNeigh]];
#endif					
					}
					PBC = PBCtri[MAXNEIGH_d*threadIdx.x + iNeigh];
					if (PBC == ROTATE_ME_CLOCKWISE) endpt_anti = Clockwise_d * endpt_anti;
					if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_anti = Anticlockwise_d * endpt_anti;

					edge_normal.x = (endpt_anti.y - endpt_clock.y);
					edge_normal.y = (endpt_clock.x - endpt_anti.x);

					// SMARTY:
					if (pos_out.x*pos_out.x + pos_out.y*pos_out.y >
						DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
					{
						//f64 Area_quadrilateral = 0.5*(
						//	(info.pos.x + pos_anti.x)*(info.pos.y - pos_anti.y)
						//	+ (pos_clock.x + info.pos.x)*(pos_clock.y - info.pos.y)
						//	+ (pos_out.x + pos_clock.x)*(pos_out.y - pos_clock.y)
						//	+ (pos_anti.x + pos_out.x)*(pos_anti.y - pos_out.y)
						//	);
						//grad_T.x = 0.5*(
						//	(shared_T[threadIdx.x] + T_anti)*(info.pos.y - pos_anti.y)
						//	+ (T_clock + shared_T[threadIdx.x])*(pos_clock.y - info.pos.y)
						//	+ (T_out + T_clock)*(pos_out.y - pos_clock.y)
						//	+ (T_anti + T_out)*(pos_anti.y - pos_out.y)
						//	) / Area_quadrilateral;
						//grad_T.y = -0.5*( // notice minus
						//	(shared_T[threadIdx.x] + T_anti)*(info.pos.x - pos_anti.x)
						//	+ (T_clock + shared_T[threadIdx.x])*(pos_clock.x - info.pos.x)
						//	+ (T_out + T_clock)*(pos_out.x - pos_clock.x)
						//	+ (T_anti + T_out)*(pos_anti.x - pos_out.x)
						//	) / Area_quadrilateral;

						kappa_parallel = 0.0;
						f64 nu;
						if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
						{
							kappa_parallel = 0.5*shared_kappa[izTri[iNeigh] - StartMinor];
							nu = 0.5*shared_nu[izTri[iNeigh] - StartMinor];
						}
						else {
							if (iSpecies == 1) {
								kappa_parallel = 0.5*p_kappa_i[izTri[iNeigh]];
								nu = 0.5*p_nu_i[izTri[iNeigh]];
							}
							else {
								kappa_parallel = 0.5*p_kappa_e[izTri[iNeigh]];
								nu = 0.5*p_nu_e[izTri[iNeigh]];
							};
						};
						{
							short iPrev = iNeigh - 1; if (iPrev < 0) iPrev = info.neigh_len - 1;
							if ((izTri[iPrev] >= StartMinor) && (izTri[iPrev] < EndMinor))
							{
								kappa_parallel += 0.5*shared_kappa[izTri[iPrev] - StartMinor];
								nu += 0.5*shared_nu[izTri[iPrev] - StartMinor];
							}
							else {
								if (iSpecies == 1) {
									kappa_parallel += 0.5*p_kappa_i[izTri[iPrev]];
									nu += 0.5*p_nu_i[izTri[iPrev]];
								} else {
									kappa_parallel += 0.5*p_kappa_e[izTri[iPrev]];
									nu += 0.5*p_nu_e[izTri[iPrev]];
								};
							}
							
						}

						indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
						if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
						{
							B_out = shared_B[indexneigh - StartMajor];
						}
						else {
							f64_vec3 B_out3 = p_B_major[indexneigh];
							B_out = B_out3.xypart();
						}
						PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
						if (PBC == NEEDS_ANTI) 	B_out = Anticlock_rotate2(B_out);
						if (PBC == NEEDS_CLOCK)	B_out = Clockwise_rotate2(B_out);

						//kappa.xx = kappa_parallel * (nu_eHeart*nu_eHeart + omega.x*omega.x) / (nu_eHeart * nu_eHeart + omega_sq);
						//kappa.xy = kappa_parallel * (omega.x*omega.y - nu_eHeart *omega.z) / (nu_eHeart * nu_eHeart + omega_sq);
						//kappa.yx = kappa_parallel * (omega.x*omega.y + nu_eHeart * omega.z) / (nu_eHeart * nu_eHeart + omega_sq);
						//kappa.yy = kappa_parallel * (omega.y*omega.y + nu_eHeart * nu_eHeart) / (nu_eHeart * nu_eHeart + omega_sq);

						{ // scoping brace


							f64_vec3 omega;
							if (iSpecies == 1) {
								omega = Make3(qoverMc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qoverMc);
							}
							else {
								omega = Make3(qovermc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qovermc);
							};
							// PROBABLY ALWAYS SPILLED INTO GLOBAL -- WHAT CAN WE DO?
	//
	//						ourrates.NiTi += TWOTHIRDS * kappa_parallel *(
	//							edge_normal.x*((nu*nu + omega.x*omega.x)*grad_T.x + (omega.x*omega.y + nu * omega.z)*grad_T.y)
	//							+ edge_normal.y*((omega.x*omega.y - nu * omega.z)*grad_T.x + (omega.y*omega.y + nu * nu)*grad_T.y)
	//							) / (nu * nu + omega.dot(omega));
							f64 edgelen = edge_normal.modulus();
							f64 delta_out = sqrt((info.pos.x - pos_out.x)*(info.pos.x - pos_out.x) + (info.pos.y - pos_out.y)*(info.pos.y - pos_out.y));

							if (iSpecies == 1) {
								ourrates.NiTi += TWOTHIRDS * kappa_parallel *  (T_out - shared_T[threadIdx.x]) *
									(nu*nu*edgelen*edgelen + omega.dotxy(edge_normal)*omega.dotxy(edge_normal))
									/ (delta_out*edgelen *(nu * nu + omega.dot(omega)));
							}
							else {
								ourrates.NeTe += TWOTHIRDS * kappa_parallel *  (T_out - shared_T[threadIdx.x]) *
									(nu*nu*edgelen*edgelen + omega.dotxy(edge_normal)*omega.dotxy(edge_normal))
									/ (delta_out*edgelen *(nu * nu + omega.dot(omega)));
							};

							if (TESTHEAT1) 
								printf("%d iNeigh %d %d iSpecies %d kappa %1.8E nu %1.8E |o| %1.8E contrib %1.10E T_out %1.14E T_self %1.14E \n",
								iVertex, iNeigh, indexneigh, iSpecies, kappa_parallel, nu,
								omega.modulus(),
									TWOTHIRDS * kappa_parallel *  (T_out - shared_T[threadIdx.x]) *
									(nu*nu*edgelen*edgelen + omega.dotxy(edge_normal)*omega.dotxy(edge_normal))
									/ (delta_out*edgelen *(nu * nu + omega.dot(omega))),
									T_out, shared_T[threadIdx.x]

							);
						}
					} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

					  // Now go round:	
					endpt_clock = endpt_anti;
					pos_clock = pos_out;
					pos_out = pos_anti;
					T_clock = T_out;
					T_out = T_anti;
				}; // next iNeigh

			}; // was it DOMAIN_VERTEX? Do what otherwise?
		}; // was it OUTERMOST/INNERMOST?
		
		__syncthreads();
	};
	

	memcpy(NTadditionrates + iVertex, &ourrates, sizeof(NTrates));

	// It was not necessarily sensible to combine ion and electron
	// However, it is quite daft having a separate routine for vector2 grad T (??)

}

__global__ void kernelCreateEpsilonHeat
(
	f64 const hsub,
	structural * __restrict__ p_info_major,
	f64 * __restrict__ p_eps_n,
	f64 * __restrict__ p_eps_i,
	f64 * __restrict__ p_eps_e,
	f64 * __restrict__ p_NT_n,
	f64 * __restrict__ p_NT_i,
	f64 * __restrict__ p_NT_e,
	T3 * __restrict__ p_T_k,
	f64 * __restrict__ p_AreaMajor,
	nvals * __restrict__ p_n_major,
	NTrates * __restrict__ NTadditionrates, // it's especially silly having a whole struct of 5 instead of 3 here.
	bool * __restrict__ p_b_Failed
	)
{
	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info_major[iVertex];
	if (info.flag == DOMAIN_VERTEX) {
		f64 NnTn = p_NT_n[iVertex];
		f64 NTi = p_NT_i[iVertex];
		f64 NTe = p_NT_e[iVertex];
		T3 T_k = p_T_k[iVertex];
		f64 AreaMajor = p_AreaMajor[iVertex];
		nvals n = p_n_major[iVertex];
		NTrates ourrates;
		memcpy(&ourrates, NTadditionrates + iVertex, sizeof(NTrates));
		f64 sqrtNn = sqrt(AreaMajor*n.n_n);
		f64 epsilon_n = NnTn - T_k.Tn*sqrtNn - (hsub / sqrtNn)*ourrates.NnTn;
		f64 sqrtN = sqrt(AreaMajor*n.n);
		f64 epsilon_i = NTi - T_k.Ti*sqrtN - (hsub / sqrtN)*ourrates.NiTi;
		f64 epsilon_e = NTe - T_k.Te*sqrtN - (hsub / sqrtN)*ourrates.NeTe;

		if (0) printf("epsilon_i %1.14E NTi %1.14E T_k.Ti sqrtN %1.14E flow %1.14E\n",
			epsilon_i, NTi, T_k.Ti*sqrtN, (hsub / sqrtN)*ourrates.NiTi);
		if (epsilon_n != epsilon_n)printf("epsilon_n NaN iVertex %d n_n %1.10E Area %1.10E \n",
			iVertex, n.n_n, AreaMajor);

		p_eps_n[iVertex] = epsilon_n;
		p_eps_i[iVertex] = epsilon_i;
		p_eps_e[iVertex] = epsilon_e;

#define REL_THRESHOLD 1.0e-10

		if (p_b_Failed != 0) {
			if ((epsilon_n*epsilon_n > REL_THRESHOLD*REL_THRESHOLD*(NnTn*NnTn + 1.0e-10*1.0e-10))
				||
				(epsilon_i*epsilon_i > REL_THRESHOLD*REL_THRESHOLD*(NTi*NTi + 1.0e-10*1.0e-10))
				||
				(epsilon_e*epsilon_e > REL_THRESHOLD*REL_THRESHOLD*(NTe*NTe + 1.0e-10*1.0e-10))
				)
				p_b_Failed[blockIdx.x] = true;
			// Why 1.0e-10 in absolute error, for minimum value we care about:
			// N = 2.0e12*7e-5 = 1e8 
			// root N = 1e4
			// root N * 1e-14 erg = 1e-10 for (root N) T
		}
	} else {
		p_eps_n[iVertex] = 0.0;
		p_eps_i[iVertex] = 0.0;
		p_eps_e[iVertex] = 0.0;
	}
}

__global__ void kernelCreateEpsilonHeatOriginalScaling
(
	f64 const hsub,
	structural * __restrict__ p_info_major,
	f64 * __restrict__ p_eps_n,
	f64 * __restrict__ p_eps_i,
	f64 * __restrict__ p_eps_e,
	f64 * __restrict__ p_T_n,
	f64 * __restrict__ p_T_i,
	f64 * __restrict__ p_T_e,
	T3 * __restrict__ p_T_k,
	f64 * __restrict__ p_AreaMajor,
	nvals * __restrict__ p_n_major,
	NTrates * __restrict__ NTadditionrates ,// it's especially silly having a whole struct of 5 instead of 3 here.
	bool * __restrict__ bTest
)
{
	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info_major[iVertex];
	if (info.flag == DOMAIN_VERTEX) {
		f64 Tn = p_T_n[iVertex];
		f64 Ti = p_T_i[iVertex];
		f64 Te = p_T_e[iVertex];
		T3 T_k = p_T_k[iVertex];
		f64 AreaMajor = p_AreaMajor[iVertex];
		nvals n = p_n_major[iVertex];
		NTrates ourrates;
		memcpy(&ourrates, NTadditionrates + iVertex, sizeof(NTrates));
		f64 Nn = (AreaMajor*n.n_n);
		f64 epsilon_n = Tn - T_k.Tn - (hsub / Nn)*ourrates.NnTn;
		f64 N = (AreaMajor*n.n);
		f64 epsilon_i = Ti - T_k.Ti - (hsub / N)*ourrates.NiTi;
		f64 epsilon_e = Te - T_k.Te - (hsub / N)*ourrates.NeTe;
		p_eps_n[iVertex] = epsilon_n;
		p_eps_i[iVertex] = epsilon_i;
		p_eps_e[iVertex] = epsilon_e;

		if ((epsilon_n*epsilon_n > 1.0e-24*(Tn*Tn + 1.0e-14*1.0e-14))
			|| (epsilon_i*epsilon_i > 1.0e-24*(Ti*Ti + 1.0e-14*1.0e-14))
			|| (epsilon_e*epsilon_e > 1.0e-24*(Te*Te + 1.0e-14*1.0e-14))
			)
			bTest[blockIdx.x] = true;
	}
	else {
		p_eps_n[iVertex] = 0.0;
		p_eps_i[iVertex] = 0.0;
		p_eps_e[iVertex] = 0.0;
	}
}

__global__ void kernelAccumulateDiffusiveHeatRate_new_Full(
	f64 const h_use,
	structural * __restrict__ p_info_minor,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,
	long * __restrict__ izTri_verts,
	char * __restrict__ szPBCtri_verts,
	f64_vec2 * __restrict__ p_cc,

	nvals * __restrict__ p_n_major,
	T3 * __restrict__ p_T_major,
	T3 * __restrict__ p_T_putative,
	f64_vec3 * __restrict__ p_B_major,

	f64 * __restrict__ p_kappa_n,
	f64 * __restrict__ p_kappa_i,
	f64 * __restrict__ p_kappa_e,

	f64 * __restrict__ p_nu_i,
	f64 * __restrict__ p_nu_e,

	NTrates * __restrict__ NTadditionrates,
	f64 * __restrict__ p_AreaMajor,

	bool bCheckWhetherToDoctorUp
	//T3 * __restrict__ p_T_putative
	) // test whether we are pushing heat uphill...
{
	// Think we might as well take kappa_par and nu from triangles really.
	// If n is modelled well then hopefully a nearby high-n does not have a big impact.

	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajorClever]; // 2
	__shared__ f64_vec2 shared_pos[2 * threadsPerTileMajorClever];

	__shared__ f64 shared_T[threadsPerTileMajorClever];      // +3
															 //__shared__ f64 shared_T[threadsPerTileMajorClever];

	__shared__ f64_vec2 shared_B[threadsPerTileMajorClever]; // +2

															 // B is smooth. Unfortunately we have not fitted in Bz here.
															 // In order to do that perhaps rewrite so that variables are overwritten in shared.
															 // We do not need all T and nu in shared at the same time.
															 // This way is easier for NOW.
	__shared__ f64 shared_kappa[threadsPerTileMajorClever * 2];
	__shared__ f64 shared_nu[threadsPerTileMajorClever * 2];

	__shared__ long Indexneigh[MAXNEIGH_d*threadsPerTileMajorClever]; // assume 48 bytes = 4*12 = 6 doubles

	__shared__ char PBCneigh[MAXNEIGH_d*threadsPerTileMajorClever]; // 12 bytes each from L1. Have 42 per thread at 384 threads.
																	// We should at any rate try a major block of size 256. If 1 block will run at a time, so be it.
																	// Leave L1 in case of register overflow into it. <-- don't know how likely - do we only have 31 doubles in registry
																	// regardless # of threads and space? Or can be 63?
	__shared__ char PBCtri[MAXNEIGH_d*threadsPerTileMajorClever];
	// Balance of shared vs L1: 24*256*8 = 48K. That leaves 8 doublesworth in L1 for variables.

	long izTri[MAXNEIGH_d];  // so only 2 doubles left in L1. 31 in registers??

							 // Set threadsPerTileMajorClever to 256.

							 // It would help matters if we get rid of T3. We might as well therefore change to scalar flatpack T.

							 // We are hoping that it works well loading kappa(tri) and that this is not upset by nearby values. Obviously a bit of an experiment.

							 // Does make it somewhat laughable that we go to such efforts to reduce global accesses when we end up overflowing anyway. 
							 // If we can fit 24 doubles/thread in 48K that means we can fit 8 doubles/thread in 16K so that's most of L1 used up.

	long const StartMajor = blockIdx.x*blockDim.x;
	long const EndMajor = StartMajor + blockDim.x;
	long const StartMinor = blockIdx.x*blockDim.x * 2;
	long const EndMinor = StartMinor + blockDim.x * 2;
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX	// 2.5 double

	structural info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];     // 3 double
	shared_pos_verts[threadIdx.x] = info.pos;
#ifdef CENTROID_HEATCONDUCTION
	{
		structural infotemp[2];
		memcpy(infotemp, p_info_minor + 2 * iVertex, 2 * sizeof(structural));
		shared_pos[threadIdx.x * 2] = infotemp[0].pos;
		shared_pos[threadIdx.x * 2 + 1] = infotemp[1].pos;
		f64 tempf64[2];
		memcpy(&(shared_kappa[threadIdx.x * 2]), p_kappa_n + 2 * iVertex, 2 * sizeof(f64));
		// No nu to set for neutrals - not used
	}
#else
	{
		memcpy(&(shared_pos[threadIdx.x * 2]), p_cc + 2 * iVertex, 2 * sizeof(f64_vec2));
		memcpy(&(shared_kappa[threadIdx.x * 2]), p_kappa_n + 2 * iVertex, 2 * sizeof(f64));
	}
#endif


	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {
		shared_B[threadIdx.x] = p_B_major[iVertex].xypart();
		shared_T[threadIdx.x] = p_T_major[iVertex].Tn;
	}
	else {
		// SHOULD NOT BE LOOKING INTO INS.
		// How do we avoid?
		memset(&(shared_B[threadIdx.x]), 0, sizeof(f64_vec2));
		shared_T[threadIdx.x] = 0.0;
	}


	__syncthreads();

	f64_vec2 grad_T;
	f64 T_anti, T_clock, T_out, T_outk;		// 5
	f64_vec2 pos_clock, pos_anti, pos_out;   // +6
	f64_vec2 B_out;       // +2
	NTrates ourrates;      // +5
	f64 kappa_parallel; // do we use them all at once or can we save 2 doubles here?
	f64 nu;                // 20 there  
	f64_vec2 edge_normal;  // 22
	f64_vec2 endpt_anti;    // 24 .. + 6 from above
	long indexneigh;     // into the 2-double buffer in L1
	f64_vec2 endpt_clock;    // As we only use endpt_anti afterwords we could union endpt_clock with edge_normal
							 // Come back and optimize by checking which things we need in scope at the same time?

	short iNeigh; // only fixed # of addresses so short makes no difference.
	char PBC; // char makes no difference.


	if ((info.flag == INNERMOST) || (info.flag == OUTERMOST))
	{
		// [ Ignore flux into edge of outermost vertex I guess ???]
	}
	else {
		if (info.flag == DOMAIN_VERTEX) {
			// The idea of not sending blocks full of non-domain vertices is another idea. Fiddly with indices.

			// Need this, we are adding on to existing d/dt N,NT :
			memcpy(&ourrates, NTadditionrates + iVertex, sizeof(NTrates));

			memcpy(Indexneigh + MAXNEIGH_d * threadIdx.x,
				pIndexNeigh + MAXNEIGH_d * iVertex,
				MAXNEIGH_d * sizeof(long));
			memcpy(PBCneigh + MAXNEIGH_d * threadIdx.x,
				pPBCNeigh + MAXNEIGH_d * iVertex,
				MAXNEIGH_d * sizeof(char));
			memcpy(PBCtri + MAXNEIGH_d * threadIdx.x,
				szPBCtri_verts + MAXNEIGH_d * iVertex,
				MAXNEIGH_d * sizeof(char));

			// EXPERIMENT WHETHER IT IS FASTER WITH THESE OUTSIDE OR INSIDE THE BRANCH.


			memcpy(izTri, //+ MAXNEIGH_d * threadIdx.x,
				izTri_verts + MAXNEIGH_d * iVertex, MAXNEIGH_d * sizeof(long));

			// Now do Tn:

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_clock = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
				T_clock = shared_T[indexneigh - StartMajor];
#endif
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_clock = info2.pos;
#ifdef BWDSIDET
				T_clock = p_T_major[indexneigh].Tn;
#endif
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == NEEDS_ANTI) {
				pos_clock = Anticlock_rotate2(pos_clock);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_clock = Clockwise_rotate2(pos_clock);
			};
#ifndef BWDSIDET
			T_clock = p_T_k[indexneigh].Tn; 
#endif

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_out = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
				T_out = shared_T[indexneigh - StartMajor];
#endif
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_out = info2.pos;
#ifdef BWDSIDET
				T_out = p_T_major[indexneigh].Tn;
#endif
			};
#ifndef BWDSIDET
			T_outk = p_T_k[indexneigh].Tn; // ready for switch around
#endif

			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
			if (PBC == NEEDS_ANTI) {
				pos_out = Anticlock_rotate2(pos_out);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_out = Clockwise_rotate2(pos_out);
			};

			if ((izTri[info.neigh_len - 1] >= StartMinor) && (izTri[info.neigh_len - 1] < EndMinor))
			{
				endpt_clock = shared_pos[izTri[info.neigh_len - 1] - StartMinor];
			}
			else {
#ifdef CENTROID_HEATCONDUCTION
				endpt_clock = p_info_minor[izTri[info.neigh_len - 1]].pos;
#else
				endpt_clock = p_cc[izTri[info.neigh_len - 1]];
#endif
			}
			PBC = PBCtri[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == ROTATE_ME_CLOCKWISE) endpt_clock = Clockwise_d * endpt_clock;
			if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_clock = Anticlockwise_d * endpt_clock;

			if (T_clock == 0.0) {
#ifdef BWDSIDET
				T_clock = 0.5*(shared_T[threadIdx.x] + T_out);
#else
				T_clock = T_outk;
#endif
			};

#pragma unroll MAXNEIGH_d
			for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
			{
				{
					short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNext];
				}
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					pos_anti = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
					T_anti = shared_T[indexneigh - StartMajor];
#endif
				}
				else {
					structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
					pos_anti = info2.pos;
#ifdef BWDSIDET
					T_anti = p_T_major[indexneigh].Tn;
#endif
				};
				if (PBC == NEEDS_ANTI) {
					pos_anti = Anticlock_rotate2(pos_anti);
				};
				if (PBC == NEEDS_CLOCK) {
					pos_anti = Clockwise_rotate2(pos_anti);
				};
#ifndef BWDSIDET
				T_anti = p_T_k[indexneigh].Tn; // Stupid 3-struct

				// Also need to update T_opp if it was not done already

				indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					T_out = shared_T[indexneigh - StartMajor];
				} else {
					T_out = p_T_major[indexneigh].Tn;
				};			
#endif

				if (T_anti == 0.0) {
#ifdef BWDSIDET
					T_anti = 0.5*(shared_T[threadIdx.x] + T_out);
#else
					T_anti = T_outk;
#endif					
				}; // So we are receiving 0 then doing this. But how come?

				   // Now let's see
				   // tri 0 has neighs 0 and 1 I'm pretty sure (check....) CHECK

				if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
				{
					endpt_anti = shared_pos[izTri[iNeigh] - StartMinor];
				}
				else {
#ifdef CENTROID_HEATCONDUCTION
					endpt_anti = p_info_minor[izTri[iNeigh]].pos;
					// we should switch back to centroids!!
#else
					endpt_anti = p_cc[izTri[iNeigh]];
#endif
				}
				PBC = PBCtri[MAXNEIGH_d*threadIdx.x + iNeigh];
				if (PBC == ROTATE_ME_CLOCKWISE) endpt_anti = Clockwise_d * endpt_anti;
				if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_anti = Anticlockwise_d * endpt_anti;

				edge_normal.x = (endpt_anti.y - endpt_clock.y);
				edge_normal.y = (endpt_clock.x - endpt_anti.x);

				if (TEST) {
					printf("%d contrib %1.8E \n"
						"pos_anti %1.9E %1.9E pos_out %1.9E %1.9E pos_clock %1.9E %1.9E\n", iVertex,
						0.5*edge_normal.x*THIRD*(pos_anti.x + pos_clock.x
							+ info.pos.x + info.pos.x + pos_out.x + pos_out.x),
						pos_anti.x, pos_anti.y, pos_out.x, pos_out.y, pos_clock.x, pos_clock.y);
				}

				// SMARTY:
				if (pos_out.x*pos_out.x + pos_out.y*pos_out.y >
					DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					
										
					// How to detect? Loading a load of flags is a killer! We do need to load ... and this is why we should have not made info struct. Def not.

					//// 
					//if (insulator triangle)
					//{
					//	centroid1 = THIRD*(pos_anti + pos_out + info.pos);
					//	// project to radius of insulator
					//	centroid1.project_to_radius(3.44);
					//	// Now dot with unit vectors:
					//	f64_vec2 tempvec2;
					//	tempvec2.x = unit_vec1.x*centroid1.x + unit_vec1.y*centroid1.y;
					//	tempvec2.y = unit_vec2.x*centroid1.x + unit_vec2.y*centroid1.y;
					//	centroid1.x = tempvec2.x;
					//	centroid1.y = tempvec2.y;
					//} else {
					//	// centroid1 = THIRD*(pos_anti_twist + pos_out_twist);
					//	centroid1.x = THIRD*(
					//		  unit_vec1.x*(pos_anti.x - info.pos.x) + unit_vec1.y*(pos_anti.y - info.pos.y)
					//		+ unit_vec1.x*(pos_out.x - info.pos.x) + unit_vec1.y*(pos_out.y - info.pos.y)
					//		);
					//	centroid1.y = THIRD*(
					//		- unit_vec1.y*(pos_anti.x - info.pos.x) + unit_vec1.x*(pos_anti.y - info.pos.y)
					//		- unit_vec1.y*(pos_out.x - info.pos.x) + unit_vec1.x*(pos_out.y - info.pos.y)
					//		);
					//}

					//if (insulator triangle)
					//{
					//	centroid2 = THIRD*(pos_clock + pos_out + info.pos);

					//	// project to radius of insulator
					//} else {

					//}


					kappa_parallel = 0.0;
					if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
					{
						kappa_parallel = 0.5*shared_kappa[izTri[iNeigh] - StartMinor];
					}
					else {
						kappa_parallel = 0.5*p_kappa_n[izTri[iNeigh]];
					}
					{
						short iPrev = iNeigh - 1; if (iPrev < 0) iPrev = info.neigh_len - 1;
						if ((izTri[iPrev] >= StartMinor) && (izTri[iPrev] < EndMinor))
						{
							kappa_parallel += 0.5*shared_kappa[izTri[iPrev] - StartMinor];
						}
						else {
							kappa_parallel += 0.5*p_kappa_n[izTri[iPrev]];
						}
					}

					if ((pos_clock.x*pos_clock.x + pos_clock.y*pos_clock.y <
						DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
						||
						(pos_anti.x*pos_anti.x + pos_anti.y*pos_anti.y <
							DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER))
					{
						f64 edgelen = edge_normal.modulus();

						ourrates.NnTn += TWOTHIRDS * kappa_parallel * edgelen *
							(T_out - shared_T[threadIdx.x]) / (pos_out - info.pos).modulus();
					} else {
						f64 Area_quadrilateral = 0.5*(
							(info.pos.x + pos_anti.x)*(info.pos.y - pos_anti.y)
							+ (pos_clock.x + info.pos.x)*(pos_clock.y - info.pos.y)
							+ (pos_out.x + pos_clock.x)*(pos_out.y - pos_clock.y)
							+ (pos_anti.x + pos_out.x)*(pos_anti.y - pos_out.y)
							);

						// When we come to do the other species, make a subroutine.

						grad_T.x = 0.5*( // notice minus
							(shared_T[threadIdx.x] + T_anti)*(info.pos.y - pos_anti.y)
							+ (T_clock + shared_T[threadIdx.x])*(pos_clock.y - info.pos.y)
							+ (T_out + T_clock)*(pos_out.y - pos_clock.y)
							+ (T_anti + T_out)*(pos_anti.y - pos_out.y)
							) / Area_quadrilateral;

						grad_T.y = -0.5*( // notice minus
							(shared_T[threadIdx.x] + T_anti)*(info.pos.x - pos_anti.x)
							+ (T_clock + shared_T[threadIdx.x])*(pos_clock.x - info.pos.x)
							+ (T_out + T_clock)*(pos_out.x - pos_clock.x)
							+ (T_anti + T_out)*(pos_anti.x - pos_out.x)
							) / Area_quadrilateral;

						ourrates.NnTn += TWOTHIRDS * kappa_parallel * grad_T.dot(edge_normal);
						// This is correct, grad T in same coordinates as edge_normal...
					};
				} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

				  // Now go round:	
				endpt_clock = endpt_anti;
				pos_clock = pos_out;
				pos_out = pos_anti;
#ifdef BWDSIDET
				T_clock = T_out;
				T_out = T_anti;
#else
				T_clock = T_outk;
				T_outk = T_anti;
#endif

			}; // next iNeigh

		}; // was it DOMAIN_VERTEX? Do what otherwise?
	}; // was it OUTERMOST/INNERMOST?

	__syncthreads();

	// Did we make sure to include a call to syncthreads every time we carried on to update shared memory data in every other routine?
	// ##################################################################################################################################

	{
		memcpy(&(shared_kappa[threadIdx.x * 2]), p_kappa_i + 2 * iVertex, 2 * sizeof(f64));
		memcpy(&(shared_nu[threadIdx.x * 2]), p_nu_i + 2 * iVertex, 2 * sizeof(f64));
	}
	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {
		shared_T[threadIdx.x] = p_T_major[iVertex].Ti;
		// Notice major inefficiency caused by not making them scalar T arrays
	}
	else {
		shared_T[threadIdx.x] = 0.0;
	}

	__syncthreads();

	if ((info.flag == INNERMOST) || (info.flag == OUTERMOST))
	{
		// [ Ignore flux into edge of outermost vertex I guess ???]
	}
	else {
		if (info.flag == DOMAIN_VERTEX) {

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_clock = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
				T_clock = shared_T[indexneigh - StartMajor];
#endif
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_clock = info2.pos;
#ifdef BWDSIDET
				T_clock = p_T_major[indexneigh].Ti;
#endif
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == NEEDS_ANTI) {
				pos_clock = Anticlock_rotate2(pos_clock);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_clock = Clockwise_rotate2(pos_clock);
			};
#ifndef BWDSIDET
			T_clock = p_T_k[indexneigh].Ti;
#endif

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_out = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
				T_out = shared_T[indexneigh - StartMajor];
#endif
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_out = info2.pos;
#ifdef BWDSIDET
				T_out = p_T_major[indexneigh].Ti;
#endif
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
			if (PBC == NEEDS_ANTI) {
				pos_out = Anticlock_rotate2(pos_out);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_out = Clockwise_rotate2(pos_out);
			};

#ifndef BWDSIDET
			T_outk = p_T_k[indexneigh].Ti;
#endif

			if ((izTri[info.neigh_len - 1] >= StartMinor) && (izTri[info.neigh_len - 1] < EndMinor))
			{
				endpt_clock = shared_pos[izTri[info.neigh_len - 1] - StartMinor];
			}
			else {
#ifdef CENTROID_HEATCONDUCTION
				endpt_clock = p_info_minor[izTri[info.neigh_len - 1]].pos;
#else
				endpt_clock = p_cc[izTri[info.neigh_len - 1]];
#endif
			}
			PBC = PBCtri[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == ROTATE_ME_CLOCKWISE) endpt_clock = Clockwise_d * endpt_clock;
			if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_clock = Anticlockwise_d * endpt_clock;

			if (T_clock == 0.0) {
#ifdef BWDSIDET
				T_clock = 0.5*(shared_T[threadIdx.x] + T_out);
#else
				T_clock = T_outk;
#endif
			};

#pragma unroll MAXNEIGH_d
			for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
			{
				{
					short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNext];
				}
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					pos_anti = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
					T_anti = shared_T[indexneigh - StartMajor];
#endif
				}
				else {
					structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
					pos_anti = info2.pos;
#ifdef BWDSIDET
					T_anti = p_T_major[indexneigh].Ti;
#endif
				};
#ifndef BWDSIDET
				T_anti = p_T_k[indexneigh].Ti;

				indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					T_out = shared_T[indexneigh - StartMajor];
				}
				else {
					T_out = p_T_major[indexneigh].Ti;
				};
#endif
				if (PBC == NEEDS_ANTI) {
					pos_anti = Anticlock_rotate2(pos_anti);
				};
				if (PBC == NEEDS_CLOCK) {
					pos_anti = Clockwise_rotate2(pos_anti);
				};

				if (T_anti == 0.0) {
#ifdef BWDSIDET
					T_anti = 0.5*(shared_T[threadIdx.x] + T_out);
#else
					T_anti = T_outk;
#endif					
				}; // So we are receiving 0 then doing this. But how come?

				if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
				{
					endpt_anti = shared_pos[izTri[iNeigh] - StartMinor];
				}
				else {
#ifdef CENTROID_HEATCONDUCTION
					endpt_anti = p_info_minor[izTri[iNeigh]].pos;
#else
					endpt_anti = p_cc[izTri[iNeigh]];
#endif					
				}
				PBC = PBCtri[MAXNEIGH_d*threadIdx.x + iNeigh];
				if (PBC == ROTATE_ME_CLOCKWISE) endpt_anti = Clockwise_d * endpt_anti;
				if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_anti = Anticlockwise_d * endpt_anti;

				edge_normal.x = (endpt_anti.y - endpt_clock.y);
				edge_normal.y = (endpt_clock.x - endpt_anti.x);

				// SMARTY:
				if (pos_out.x*pos_out.x + pos_out.y*pos_out.y >
					DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{

					kappa_parallel = 0.0;
					f64 nu;
					if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
					{
						kappa_parallel = 0.5*shared_kappa[izTri[iNeigh] - StartMinor];
						nu = 0.5*shared_nu[izTri[iNeigh] - StartMinor];
					}
					else {
						kappa_parallel = 0.5*p_kappa_i[izTri[iNeigh]];
						nu = 0.5*p_nu_i[izTri[iNeigh]];
					};

					{
						short iPrev = iNeigh - 1; if (iPrev < 0) iPrev = info.neigh_len - 1;
						if ((izTri[iPrev] >= StartMinor) && (izTri[iPrev] < EndMinor))
						{
							kappa_parallel += 0.5*shared_kappa[izTri[iPrev] - StartMinor];
							nu += 0.5*shared_nu[izTri[iPrev] - StartMinor];
						}
						else {
							kappa_parallel += 0.5*p_kappa_i[izTri[iPrev]];
							nu += 0.5*p_nu_i[izTri[iPrev]];
						}
					}

					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						B_out = shared_B[indexneigh - StartMajor];
					}
					else {
						f64_vec3 B_out3 = p_B_major[indexneigh];
						B_out = B_out3.xypart();
					}
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if (PBC == NEEDS_ANTI) 	B_out = Anticlock_rotate2(B_out);
					if (PBC == NEEDS_CLOCK)	B_out = Clockwise_rotate2(B_out);

					if ((pos_clock.x*pos_clock.x + pos_clock.y*pos_clock.y <
						DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
						||
						(pos_anti.x*pos_anti.x + pos_anti.y*pos_anti.y <
							DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER))
					{
						
						// Use longitudinal:

						f64 edgelen = edge_normal.modulus();
						f64 delta_out = sqrt((info.pos.x - pos_out.x)*(info.pos.x - pos_out.x) + (info.pos.y - pos_out.y)*(info.pos.y - pos_out.y));
						f64_vec3 omega = Make3(qoverMc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qoverMc);

						f64 long_contrib = TWOTHIRDS * kappa_parallel *  (T_out - shared_T[threadIdx.x]) *
							(nu*nu*edgelen*edgelen + omega.dotxy(edge_normal)*omega.dotxy(edge_normal))
							/ (delta_out*edgelen*(nu * nu + omega.dot(omega)));
						ourrates.NiTi += long_contrib;


					} else {
						f64 Area_quadrilateral = 0.5*(
							(info.pos.x + pos_anti.x)*(info.pos.y - pos_anti.y)
							+ (pos_clock.x + info.pos.x)*(pos_clock.y - info.pos.y)
							+ (pos_out.x + pos_clock.x)*(pos_out.y - pos_clock.y)
							+ (pos_anti.x + pos_out.x)*(pos_anti.y - pos_out.y)
							);
						grad_T.x = 0.5*(
							(shared_T[threadIdx.x] + T_anti)*(info.pos.y - pos_anti.y)
							+ (T_clock + shared_T[threadIdx.x])*(pos_clock.y - info.pos.y)
							+ (T_out + T_clock)*(pos_out.y - pos_clock.y)
							+ (T_anti + T_out)*(pos_anti.y - pos_out.y)
							) / Area_quadrilateral;
						grad_T.y = -0.5*( // notice minus
							(shared_T[threadIdx.x] + T_anti)*(info.pos.x - pos_anti.x)
							+ (T_clock + shared_T[threadIdx.x])*(pos_clock.x - info.pos.x)
							+ (T_out + T_clock)*(pos_out.x - pos_clock.x)
							+ (T_anti + T_out)*(pos_anti.x - pos_out.x)
							) / Area_quadrilateral;

						//kappa.xx = kappa_parallel * (nu_eHeart*nu_eHeart + omega.x*omega.x) / (nu_eHeart * nu_eHeart + omega_sq);
						//kappa.xy = kappa_parallel * (omega.x*omega.y - nu_eHeart *omega.z) / (nu_eHeart * nu_eHeart + omega_sq);
						//kappa.yx = kappa_parallel * (omega.x*omega.y + nu_eHeart * omega.z) / (nu_eHeart * nu_eHeart + omega_sq);
						//kappa.yy = kappa_parallel * (omega.y*omega.y + nu_eHeart * nu_eHeart) / (nu_eHeart * nu_eHeart + omega_sq);

						{ // scoping brace
							f64_vec3 omega = Make3(qoverMc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qoverMc);
							// PROBABLY ALWAYS SPILLED INTO GLOBAL -- WHAT CAN WE DO?

							f64 contrib = TWOTHIRDS * kappa_parallel *(
								edge_normal.x*((nu*nu + omega.x*omega.x)*grad_T.x + (omega.x*omega.y - nu * omega.z)*grad_T.y)
								+ edge_normal.y*((omega.x*omega.y + nu * omega.z)*grad_T.x + (omega.y*omega.y + nu * nu)*grad_T.y)
								) / (nu * nu + omega.dot(omega));

							// Rule 1. Not a greater flow than isotropic
							// Rule 2. Not the opposite direction to isotropic - minimum zero
							f64 iso_contrib = TWOTHIRDS * kappa_parallel *(edge_normal.x*grad_T.x + edge_normal.y*grad_T.y);
							if (contrib > 0.0) {
								if ((iso_contrib > 0.0) && (contrib > iso_contrib)) contrib = iso_contrib;
								if (iso_contrib < 0.0) contrib = 0.0;
							}
							else {
								if ((iso_contrib < 0.0) && (contrib < iso_contrib)) contrib = iso_contrib;
								if (iso_contrib > 0.0) contrib = 0.0;
							}
							//
							//						if (TESTHEATFULL) printf("%d iNeigh %d kappa_ion %1.8E nu %1.8E |o| %1.8E contrib %1.8E \n",
							//							iVertex, iNeigh, kappa_parallel, nu,
							//							omega.modulus(),
							//							TWOTHIRDS * kappa_parallel *(
							//								edge_normal.x*((nu*nu + omega.x*omega.x)*grad_T.x + (omega.x*omega.y + nu * omega.z)*grad_T.y)
							//								+ edge_normal.y*((omega.x*omega.y - nu * omega.z)*grad_T.x + (omega.y*omega.y + nu * nu)*grad_T.y)
							//								) / (nu * nu + omega.dot(omega))
							//						);
							//
							if (bCheckWhetherToDoctorUp) {

								// Now ask if this flow is going uphill:

								f64 Tout2 = p_T_putative[indexneigh].Ti;
								f64 T_here2 = p_T_putative[iVertex].Ti;

								if ((Tout2 < 0.0) || (T_here2 < 0.0)) {
									// use longitudinal flows

									f64 edgelen = edge_normal.modulus();
									f64 delta_out = sqrt((info.pos.x - pos_out.x)*(info.pos.x - pos_out.x) + (info.pos.y - pos_out.y)*(info.pos.y - pos_out.y));

									f64 long_contrib = TWOTHIRDS * kappa_parallel *  (T_out - shared_T[threadIdx.x]) *
										(nu*nu*edgelen*edgelen + omega.dotxy(edge_normal)*omega.dotxy(edge_normal))
										/ (delta_out*edgelen*(nu * nu + omega.dot(omega)));
									printf("ION %d : %d T T_out %1.8E %1.8E T_put T_putout %1.8E %1.8E cont %1.9E long %1.9E\n",
										iVertex, indexneigh, shared_T[threadIdx.x], T_out, T_here2, Tout2, contrib, long_contrib);

									// if (((T_here2 < Tout2) && (contrib < 0.0)) || ((T_here2 > Tout2) && (contrib > 0.0))) {
									// Either we are less but shrinking or more but growing

									contrib = long_contrib;
								};
							};

							ourrates.NiTi += contrib;
						}; // scoping brace
					}
				} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

				  // Now go round:	
				endpt_clock = endpt_anti;
				pos_clock = pos_out;
				pos_out = pos_anti;
#ifndef BWDSIDET
				T_clock = T_outk;
				T_outk = T_anti;
#else
				T_clock = T_out;
				T_out = T_anti;
#endif

			}; // next iNeigh

		}; // was it DOMAIN_VERTEX? Do what otherwise?
	}; // was it OUTERMOST/INNERMOST?


	__syncthreads();

	// Did we make sure to include a call to syncthreads every time we carried on to update shared memory data in every other routine?
	// ##################################################################################################################################

	{
		memcpy(&(shared_kappa[threadIdx.x * 2]), p_kappa_e + 2 * iVertex, 2 * sizeof(f64));
		memcpy(&(shared_nu[threadIdx.x * 2]), p_nu_e + 2 * iVertex, 2 * sizeof(f64));
	}
	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {
		shared_T[threadIdx.x] = p_T_major[iVertex].Te;
	}
	else {
		shared_T[threadIdx.x] = 0.0;
	}

	__syncthreads();


	if ((info.flag == INNERMOST) || (info.flag == OUTERMOST))
	{
		// [ Ignore flux into edge of outermost vertex I guess ???]
	}
	else {
		if (info.flag == DOMAIN_VERTEX) {

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_clock = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
				T_clock = shared_T[indexneigh - StartMajor];
#endif
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_clock = info2.pos;
#ifdef BWDSIDET
				T_clock = p_T_major[indexneigh].Te;
#endif
			};
#ifndef BWDSIDET
			T_clock = p_T_k[indexneigh].Te;
#endif

			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == NEEDS_ANTI) {
				pos_clock = Anticlock_rotate2(pos_clock);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_clock = Clockwise_rotate2(pos_clock);
			};

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_out = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
				T_out = shared_T[indexneigh - StartMajor];
#endif
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_out = info2.pos;
#ifdef BWDSIDET
				T_out = p_T_major[indexneigh].Te;
#endif
			};
#ifndef BWDSIDET
			T_outk = p_T_k[indexneigh].Te;
#endif
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
			if (PBC == NEEDS_ANTI) {
				pos_out = Anticlock_rotate2(pos_out);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_out = Clockwise_rotate2(pos_out);
			};

			if ((izTri[info.neigh_len - 1] >= StartMinor) && (izTri[info.neigh_len - 1] < EndMinor))
			{
				endpt_clock = shared_pos[izTri[info.neigh_len - 1] - StartMinor];
			}
			else {
#ifdef CENTROID_HEATCONDUCTION
				endpt_clock = p_info_minor[izTri[info.neigh_len - 1]].pos;
#else
				endpt_clock = p_cc[izTri[info.neigh_len - 1]];
#endif
			}
			PBC = PBCtri[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == ROTATE_ME_CLOCKWISE) endpt_clock = Clockwise_d * endpt_clock;
			if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_clock = Anticlockwise_d * endpt_clock;

			if (T_clock == 0.0) {
#ifdef BWDSIDET
				T_clock = 0.5*(shared_T[threadIdx.x] + T_out);
#else
				T_clock = T_outk;
#endif
			};

#pragma unroll MAXNEIGH_d
			for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
			{
				{
					short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNext];
				}
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					pos_anti = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
					T_anti = shared_T[indexneigh - StartMajor];
#endif
				}
				else {
					structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
					pos_anti = info2.pos;
#ifdef BWDSIDET
					T_anti = p_T_major[indexneigh].Te;
#endif
				};
#ifndef BWDSIDET
				T_anti = p_T_k[indexneigh].Te;
				indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					T_out = shared_T[indexneigh - StartMajor];
				} else {
					T_out = p_T_major[indexneigh].Te;
				}
#endif
				if (PBC == NEEDS_ANTI) {
					pos_anti = Anticlock_rotate2(pos_anti);
				};
				if (PBC == NEEDS_CLOCK) {
					pos_anti = Clockwise_rotate2(pos_anti);
				};

				if (T_anti == 0.0) {
#ifdef BWDSIDET
					T_anti = 0.5*(shared_T[threadIdx.x] + T_out);
#else
					T_anti = T_outk;
#endif					
				}; // So we are receiving 0 then doing this. But how come?

				if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
				{
					endpt_anti = shared_pos[izTri[iNeigh] - StartMinor];

					if (TEST) {
						printf("%d : %d endpt_anti %1.9E %1.9E SHARED endpt_clock %1.9E %1.9E izTri[iNeigh] %d\n",
							iVertex, iNeigh, endpt_anti.x, endpt_anti.y, endpt_clock.x, endpt_clock.y, izTri[iNeigh]);
					}
				}
				else {
#ifdef CENTROID_HEATCONDUCTION
					endpt_anti = p_info_minor[izTri[iNeigh]].pos;
#else
					endpt_anti = p_cc[izTri[iNeigh]];
#endif

					if (TEST) {
						printf("%d : %d endpt_anti %1.9E %1.9E GLOBAL endpt_clock %1.9E %1.9E izTri[iNeigh] %d\n",
							iVertex, iNeigh, endpt_anti.x, endpt_anti.y, endpt_clock.x, endpt_clock.y, izTri[iNeigh]);
					}
				}
				PBC = PBCtri[MAXNEIGH_d*threadIdx.x + iNeigh];
				if (PBC == ROTATE_ME_CLOCKWISE) endpt_anti = Clockwise_d * endpt_anti;
				if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_anti = Anticlockwise_d * endpt_anti;

				// It decided to rotate something it shouldn't oughta. Rotated tri 23600 = tri 2 for 11582.

				edge_normal.x = (endpt_anti.y - endpt_clock.y);
				edge_normal.y = (endpt_clock.x - endpt_anti.x);

				// SMARTY:
				if (pos_out.x*pos_out.x + pos_out.y*pos_out.y >
					DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					
					kappa_parallel = 0.0;
					f64 nu;
					if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
					{
						kappa_parallel = 0.5*shared_kappa[izTri[iNeigh] - StartMinor];
						nu = 0.5*shared_nu[izTri[iNeigh] - StartMinor];
					}
					else {
						kappa_parallel = 0.5*p_kappa_e[izTri[iNeigh]];
						nu = 0.5*p_nu_e[izTri[iNeigh]];
					};

					{
						short iPrev = iNeigh - 1; if (iPrev < 0) iPrev = info.neigh_len - 1;
						if ((izTri[iPrev] >= StartMinor) && (izTri[iPrev] < EndMinor))
						{
							kappa_parallel += 0.5*shared_kappa[izTri[iPrev] - StartMinor];
							nu += 0.5*shared_nu[izTri[iPrev] - StartMinor];
						}
						else {
							kappa_parallel += 0.5*p_kappa_e[izTri[iPrev]];
							nu += 0.5*p_nu_e[izTri[iPrev]];
						}
					}

					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						B_out = shared_B[indexneigh - StartMajor];
					}
					else {
						f64_vec3 B_out3 = p_B_major[indexneigh];
						B_out = B_out3.xypart();
					}
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if (PBC == NEEDS_ANTI) 	B_out = Anticlock_rotate2(B_out);
					if (PBC == NEEDS_CLOCK)	B_out = Clockwise_rotate2(B_out);

					if ((pos_clock.x*pos_clock.x + pos_clock.y*pos_clock.y <
						DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
						||
						(pos_anti.x*pos_anti.x + pos_anti.y*pos_anti.y <
							DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER))
					{

						// Use longitudinal:

						f64 edgelen = edge_normal.modulus();
						f64 delta_out = sqrt((info.pos.x - pos_out.x)*(info.pos.x - pos_out.x) + (info.pos.y - pos_out.y)*(info.pos.y - pos_out.y));
						f64_vec3 omega = Make3(qoverMc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qoverMc);

						f64 long_contrib = TWOTHIRDS * kappa_parallel *  (T_out - shared_T[threadIdx.x]) *
							(nu*nu*edgelen*edgelen + omega.dotxy(edge_normal)*omega.dotxy(edge_normal))
							/ (delta_out*edgelen*(nu * nu + omega.dot(omega)));
						
						ourrates.NeTe += long_contrib;
					}
					else {
						f64 Area_quadrilateral = 0.5*(
							(info.pos.x + pos_anti.x)*(info.pos.y - pos_anti.y)
							+ (pos_clock.x + info.pos.x)*(pos_clock.y - info.pos.y)
							+ (pos_out.x + pos_clock.x)*(pos_out.y - pos_clock.y)
							+ (pos_anti.x + pos_out.x)*(pos_anti.y - pos_out.y)
							);
						grad_T.x = 0.5*(
							(shared_T[threadIdx.x] + T_anti)*(info.pos.y - pos_anti.y)
							+ (T_clock + shared_T[threadIdx.x])*(pos_clock.y - info.pos.y)
							+ (T_out + T_clock)*(pos_out.y - pos_clock.y)
							+ (T_anti + T_out)*(pos_anti.y - pos_out.y)
							) / Area_quadrilateral;
						grad_T.y = -0.5*( // notice minus
							(shared_T[threadIdx.x] + T_anti)*(info.pos.x - pos_anti.x)
							+ (T_clock + shared_T[threadIdx.x])*(pos_clock.x - info.pos.x)
							+ (T_out + T_clock)*(pos_out.x - pos_clock.x)
							+ (T_anti + T_out)*(pos_anti.x - pos_out.x)
							) / Area_quadrilateral;

						//kappa.xx = kappa_parallel * (nu_eHeart*nu_eHeart + omega.x*omega.x) / (nu_eHeart * nu_eHeart + omega_sq);
						//kappa.xy = kappa_parallel * (omega.x*omega.y - nu_eHeart *omega.z) / (nu_eHeart * nu_eHeart + omega_sq);
						//kappa.yx = kappa_parallel * (omega.x*omega.y + nu_eHeart * omega.z) / (nu_eHeart * nu_eHeart + omega_sq);
						//kappa.yy = kappa_parallel * (omega.y*omega.y + nu_eHeart * nu_eHeart) / (nu_eHeart * nu_eHeart + omega_sq);

						{ // scoping brace
							f64_vec3 omega = Make3(qovermc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qovermc);
							// PROBABLY ALWAYS SPILLED INTO GLOBAL -- WHAT CAN WE DO?

							f64 contrib = TWOTHIRDS * kappa_parallel *(
								edge_normal.x*((nu*nu + omega.x*omega.x)*grad_T.x + (omega.x*omega.y - nu * omega.z)*grad_T.y)
								+ edge_normal.y*((omega.x*omega.y + nu * omega.z)*grad_T.x + (omega.y*omega.y + nu * nu)*grad_T.y)
								) / (nu * nu + omega.dot(omega));

							// Rule 1. Not a greater flow than isotropic
							// Rule 2. Not the opposite direction to isotropic - minimum zero
							f64 iso_contrib = TWOTHIRDS * kappa_parallel *(edge_normal.x*grad_T.x + edge_normal.y*grad_T.y);


							if (TESTHEATFULL) printf(
								"iVertex %d iNeigh %d contrib %1.9E iso_contrib %1.9E \n"
								"edge_normal %1.8E %1.8E \n"
								"T %1.9E Tout %1.9E T_anti %1.9E T_clock %1.9E\n"
								"   kappa_par %1.9E nu %1.9E |omega| %1.9E Area %1.9E\n"
								"our_n %1.9E our n_n %1.9E nearby n %1.9E %1.9E\n"
								"pos %1.8E %1.8E opp %1.8E %1.8E anti %1.8E %1.8E clock %1.8E %1.8E\n"
								"omega %1.8E %1.8E grad_T %1.9E %1.9E \n"
								"=================================================\n"
								, iVertex, iNeigh,
								contrib, iso_contrib,
								edge_normal.x, edge_normal.y, shared_T[threadIdx.x], T_out, T_anti, T_clock,
								kappa_parallel, nu, sqrt(omega.dot(omega)),
								p_AreaMajor[iVertex],
								p_n_major[iVertex].n, p_n_major[iVertex].n_n, p_n_major[indexneigh].n, p_n_major[indexneigh].n_n,
								info.pos.x, info.pos.y, pos_out.x, pos_out.y, pos_anti.x, pos_anti.y, pos_clock.x, pos_clock.y,
								omega.x, omega.y, grad_T.x, grad_T.y);


							if (contrib > 0.0) {
								if ((iso_contrib > 0.0) && (contrib > iso_contrib)) contrib = iso_contrib;
								if (iso_contrib < 0.0) contrib = 0.0;
							}
							else {
								if ((iso_contrib < 0.0) && (contrib < iso_contrib)) contrib = iso_contrib;
								if (iso_contrib > 0.0) contrib = 0.0;
							}

							if (bCheckWhetherToDoctorUp) {

								// Now ask if this flow is going uphill:

								f64 Tout2 = p_T_putative[indexneigh].Te;
								f64 T_here2 = p_T_putative[iVertex].Te;

								if ((Tout2 < 0.0) || (T_here2 < 0.0)) {
									// use longitudinal flows

									f64 edgelen = edge_normal.modulus();
									f64 delta_out = sqrt((info.pos.x - pos_out.x)*(info.pos.x - pos_out.x) + (info.pos.y - pos_out.y)*(info.pos.y - pos_out.y));

									f64 long_contrib = TWOTHIRDS * kappa_parallel *  (T_out - shared_T[threadIdx.x]) *
										(nu*nu*edgelen*edgelen + omega.dotxy(edge_normal)*omega.dotxy(edge_normal))
										/ (delta_out*edgelen*(nu * nu + omega.dot(omega)));
									printf("ELEC %d : %d T T_out %1.8E %1.8E T_put T_putout %1.8E %1.8E cont %1.9E long %1.9E\n",
										iVertex, indexneigh, shared_T[threadIdx.x], T_out, T_here2, Tout2, contrib, long_contrib);

									// if (((T_here2 < Tout2) && (contrib < 0.0)) || ((T_here2 > Tout2) && (contrib > 0.0))) {
									// Either we are less but shrinking or more but growing

									contrib = long_contrib;
								};
							};

							ourrates.NeTe += contrib;

							if (TESTHEATFULL) printf("ourrates.NeTe %1.10E \n", ourrates.NeTe);

						}
					};
				} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

				  // Now go round:	
				endpt_clock = endpt_anti;
				pos_clock = pos_out;
				pos_out = pos_anti;
#ifdef BWDSIDET
				T_clock = T_out;
				T_out = T_anti;
#else
				T_clock = T_outk;
				T_outk = T_anti;
#endif

			}; // next iNeigh

			memcpy(NTadditionrates + iVertex, &ourrates, sizeof(NTrates));

		}; // was it DOMAIN_VERTEX? Do what otherwise?
	}; // was it OUTERMOST/INNERMOST?

}

__global__ void kernelCreatePutativeT (
	f64 hsub,
	structural * __restrict__ p_info_minor,
	T3 * __restrict__ p_T_k,
	T3 * __restrict__ p_T_putative,
	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p_AreaMajor,
	NTrates * __restrict__ NTadditionrates
	)
{
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX	// 2.5 double

	structural info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];     // 3 double
	T3 T_k = p_T_k[iVertex];
	nvals n = p_n_major[iVertex];
	f64 AreaMajor = p_AreaMajor[iVertex];
	NTrates NT = NTadditionrates[iVertex];
	T3 T_put;
	T_put.Tn = T_k.Tn + hsub* NT.NeTe / (n.n_n*AreaMajor);
	T_put.Ti = T_k.Ti + hsub*NT.NiTi / (n.n*AreaMajor);
	T_put.Te = T_k.Te + hsub*NT.NeTe / (n.n*AreaMajor);

	if (iVertex == VERTCHOSEN) printf("%d T_e_k %1.8E NeTe %1.8E N %1.8E T_put %1.8E\n",
		iVertex, T_k.Te, NT.NeTe, (n.n*AreaMajor), T_put.Te);

	p_T_putative[iVertex] = T_put;
}

// It was a completely unnecessary routine. All it does is same as the heat flux calc routine but
// multiplied by -h/N and with +1. :
__global__ void kernelCalculateROCepsWRTregressorT(
	f64 const h_use,
	structural * __restrict__ p_info_minor,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,
	long * __restrict__ izTri_verts,
	char * __restrict__ szPBCtri_verts,
	f64_vec2 * __restrict__ p_cc,

	nvals * __restrict__ p_n_major,
	T3 * __restrict__ p_T_major,
	f64_vec3 * __restrict__ p_B_major,

	f64 * __restrict__ p_kappa_n,
	f64 * __restrict__ p_kappa_i,
	f64 * __restrict__ p_kappa_e,

	f64 * __restrict__ p_nu_i,
	f64 * __restrict__ p_nu_e,
	f64 * __restrict__ p_AreaMajor,

	f64 * __restrict__ p_regressor_n,
	f64 * __restrict__ p_regressor_i,
	f64 * __restrict__ p_regressor_e,
	f64 * __restrict__ dz_d_eps_by_dbeta_n,
	f64 * __restrict__ dz_d_eps_by_dbeta_i,
	f64 * __restrict__ dz_d_eps_by_dbeta_e
)
{
	// ******************************************
	//  1. Do this with kappa given from t_k etc
	// ******************************************
	// Then come and change it.

	// Think we might as well take kappa_par and nu from triangles really.
	// If n is modelled well then hopefully a nearby high-n does not have a big impact.

	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajorClever]; // 2
	__shared__ f64_vec2 shared_pos[2 * threadsPerTileMajorClever]; // Need why? If using centroid? Sometimes on boundary - is that only reason?
																   // Seems like a waste of 2 doubles. Can just average -- and shift if we load the flag that tells us to.

	__shared__ f64 shared_x[threadsPerTileMajorClever];
	// Inefficiency caused by our making a 3-struct. We didn't need to do that.

	__shared__ f64_vec2 shared_B[threadsPerTileMajorClever]; // +2
															 // B is smooth. Unfortunately we have not fitted in Bz here.
															 // In order to do that perhaps rewrite so that variables are overwritten in shared.
															 // This way is easier for NOW.
															 // So that's the optimization we want. Scalar arrays for T, nu.

	__shared__ f64 shared_kappa[threadsPerTileMajorClever * 2];
	__shared__ f64 shared_nu[threadsPerTileMajorClever * 2];

	__shared__ long Indexneigh[MAXNEIGH_d*threadsPerTileMajorClever]; // assume 48 bytes = 4*12 = 6 doubles

	__shared__ char PBCneigh[MAXNEIGH_d*threadsPerTileMajorClever]; // 12 bytes each from L1. Have 42 per thread at 384 threads.
																	// We should at any rate try a major block of size 256. If 1 block will run at a time, so be it.
																	// Leave L1 in case of register overflow into it. <-- don't know how likely - do we only have 31 doubles in registry
																	// regardless # of threads and space? Or can be 63?
	__shared__ char PBCtri[MAXNEIGH_d*threadsPerTileMajorClever];
	// Balance of shared vs L1: 24*256*8 = 48K. That leaves 8 doublesworth in L1 for variables.

	long izTri[MAXNEIGH_d];  // so only 2 doubles left in L1. 31 in registers??
							 // Set threadsPerTileMajorClever to 256.

							 // It would help matters if we get rid of T3. We might as well therefore change to scalar flatpack T.
							 // We are hoping that it works well loading kappa(tri) and that this is not upset by nearby values. Obviously a bit of an experiment.

							 // Does make it somewhat laughable that we go to such efforts to reduce global accesses when we end up overflowing anyway. 
							 // If we can fit 24 doubles/thread in 48K that means we can fit 8 doubles/thread in 16K so that's most of L1 used up.

	long const StartMajor = blockIdx.x*blockDim.x;
	long const EndMajor = StartMajor + blockDim.x;
	long const StartMinor = blockIdx.x*blockDim.x * 2;
	long const EndMinor = StartMinor + blockDim.x * 2;
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX	// 2.5 double

	structural info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];     // 3 double
	shared_pos_verts[threadIdx.x] = info.pos;
#ifdef CENTROID_HEATCONDUCTION
	{
		structural infotemp[2];
		memcpy(infotemp, p_info_minor + 2 * iVertex, 2 * sizeof(structural));
		shared_pos[threadIdx.x * 2] = infotemp[0].pos;
		shared_pos[threadIdx.x * 2 + 1] = infotemp[1].pos;
		memcpy(&(shared_kappa[threadIdx.x * 2]), p_kappa_n + 2 * iVertex, 2 * sizeof(f64));
	}
#else
	{
		memcpy(&(shared_pos[threadIdx.x * 2]), p_cc + 2 * iVertex, 2 * sizeof(f64_vec2));
		memcpy(&(shared_kappa[threadIdx.x * 2]), p_kappa_n + 2 * iVertex, 2 * sizeof(f64));
	}
#endif


	// No nu to set for neutrals - not used

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {

		shared_B[threadIdx.x] = p_B_major[iVertex].xypart();
		shared_x[threadIdx.x] = p_regressor_n[iVertex];
	}
	else {
		// SHOULD NOT BE LOOKING INTO INS. How do we avoid?

		memset(&(shared_B[threadIdx.x]), 0, sizeof(f64_vec2));
		shared_x[threadIdx.x] = 0.0;
	}

	__syncthreads();
	// works if we cut here

	f64 regressor_anti, regressor_clock, regressor_out;		// 5
	f64_vec2 pos_clock, pos_anti, pos_out;   // +6
	f64_vec2 B_out;       // +2
						  // NTrates ourrates;      // +5
	f64 kappa_parallel;
	f64 nu;                // 20 there  
	f64_vec2 edge_normal;  // 22
	f64_vec2 endpt_anti;    // 24 .. + 6 from above
	long indexneigh;     // into the 2-double buffer in L1
	f64_vec2 endpt_clock;    // As we only use endpt_anti afterwords we could union endpt_clock with edge_normal
							 // Come back and optimize by checking which things we need in scope at the same time?
	f64 d_eps_by_d_beta;

	short iNeigh; // only fixed # of addresses so short makes no difference.
	char PBC; // char makes no difference.

	if ((info.flag == INNERMOST) || (info.flag == OUTERMOST))
	{
		// [ Ignore flux into edge of outermost vertex I guess ???]
	}
	else {
		if (info.flag == DOMAIN_VERTEX) {

			// The idea of not sending blocks full of non-domain vertices is another idea. Fiddly with indices.
			d_eps_by_d_beta = shared_x[threadIdx.x]; // T_self affects eps directly.
													 // correct sign as :
													 // 	f64 epsilon_e = p_T_putative[iVertex].Te - p_T_k[iVertex].Te - (h_sub / N)*Rates.NeTe;

			memcpy(Indexneigh + MAXNEIGH_d * threadIdx.x,
				pIndexNeigh + MAXNEIGH_d * iVertex,
				MAXNEIGH_d * sizeof(long));
			memcpy(PBCneigh + MAXNEIGH_d * threadIdx.x,
				pPBCNeigh + MAXNEIGH_d * iVertex,
				MAXNEIGH_d * sizeof(char));
			memcpy(PBCtri + MAXNEIGH_d * threadIdx.x,
				szPBCtri_verts + MAXNEIGH_d * iVertex,
				MAXNEIGH_d * sizeof(char));
			memcpy(izTri, //+ MAXNEIGH_d * threadIdx.x,
				izTri_verts + MAXNEIGH_d * iVertex, MAXNEIGH_d * sizeof(long));


			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_clock = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
				regressor_clock = shared_x[indexneigh - StartMajor];
#endif
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_clock = info2.pos;
#ifdef BWDSIDET
				regressor_clock = p_regressor_n[indexneigh];
#endif
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == NEEDS_ANTI) { pos_clock = Anticlock_rotate2(pos_clock); };
			if (PBC == NEEDS_CLOCK) { pos_clock = Clockwise_rotate2(pos_clock); };

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_out = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
				regressor_out = shared_x[indexneigh - StartMajor];
#endif
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_out = info2.pos;
#ifdef BWDSIDET
				regressor_out = p_regressor_n[indexneigh];
#endif
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
			if (PBC == NEEDS_ANTI) { pos_out = Anticlock_rotate2(pos_out); };
			if (PBC == NEEDS_CLOCK) { pos_out = Clockwise_rotate2(pos_out); };

			if ((izTri[info.neigh_len - 1] >= StartMinor) && (izTri[info.neigh_len - 1] < EndMinor))
			{
				endpt_clock = shared_pos[izTri[info.neigh_len - 1] - StartMinor];
			}
			else {
#ifdef CENTROID_HEATCONDUCTION
				endpt_clock = p_info_minor[izTri[info.neigh_len - 1]].pos;
#else
				endpt_clock = p_cc[izTri[info.neigh_len - 1]];
#endif
			}
			PBC = PBCtri[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == ROTATE_ME_CLOCKWISE) endpt_clock = Clockwise_d * endpt_clock;
			if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_clock = Anticlockwise_d * endpt_clock;
#ifdef BWDSIDET
			if (regressor_clock == 0.0) {
				regressor_clock = 0.5*(shared_x[threadIdx.x] + regressor_out);
			};
#endif

			f64 Nn = p_AreaMajor[iVertex] * p_n_major[iVertex].n_n;

#pragma unroll MAXNEIGH_d
			for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
			{
				{
					short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNext];
				}
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					pos_anti = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
					regressor_anti = shared_x[indexneigh - StartMajor];
#endif
				}
				else {
					structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
					pos_anti = info2.pos;
#ifdef BWDSIDET
					regressor_anti = p_regressor_n[indexneigh];
#endif
				};
				if (PBC == NEEDS_ANTI) {
					pos_anti = Anticlock_rotate2(pos_anti);
				};
				if (PBC == NEEDS_CLOCK) {
					pos_anti = Clockwise_rotate2(pos_anti);
				};
#ifdef BWDSIDET
				if (regressor_anti == 0.0) {
					regressor_anti = 0.5*(shared_x[threadIdx.x] + regressor_out);
				}; // So we are receiving 0 then doing this. But how come?
#else
				indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					regressor_out = shared_x[indexneigh - StartMajor];
				} else {
					regressor_out = p_regressor_n[indexneigh];
				};
#endif
				   // Now let's see
				   // tri 0 has neighs 0 and 1 I'm pretty sure (check....) CHECK

				if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
				{
					endpt_anti = shared_pos[izTri[iNeigh] - StartMinor];
				}
				else {
#ifdef CENTROID_HEATCONDUCTION
					endpt_anti = p_info_minor[izTri[iNeigh]].pos;
#else
					endpt_anti = p_cc[izTri[iNeigh]];
#endif
				}
				PBC = PBCtri[MAXNEIGH_d*threadIdx.x + iNeigh];
				if (PBC == ROTATE_ME_CLOCKWISE) endpt_anti = Clockwise_d * endpt_anti;
				if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_anti = Anticlockwise_d * endpt_anti;

				edge_normal.x = (endpt_anti.y - endpt_clock.y);
				edge_normal.y = (endpt_clock.x - endpt_anti.x);

				// SMARTY:
				if (pos_out.x*pos_out.x + pos_out.y*pos_out.y >
					DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					f64 Area_quadrilateral = 0.5*(
						(info.pos.x + pos_anti.x)*(info.pos.y - pos_anti.y)
						+ (pos_clock.x + info.pos.x)*(pos_clock.y - info.pos.y)
						+ (pos_out.x + pos_clock.x)*(pos_out.y - pos_clock.y)
						+ (pos_anti.x + pos_out.x)*(pos_anti.y - pos_out.y)
						);

					kappa_parallel = 0.0;
					if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
					{
						kappa_parallel = 0.5*shared_kappa[izTri[iNeigh] - StartMinor];
					}
					else {
						kappa_parallel = 0.5*p_kappa_n[izTri[iNeigh]];
					};
					{
						short iPrev = iNeigh - 1; if (iPrev < 0) iPrev = info.neigh_len - 1;
						if ((izTri[iPrev] >= StartMinor) && (izTri[iPrev] < EndMinor))
						{
							kappa_parallel += 0.5*shared_kappa[izTri[iPrev] - StartMinor];
						}
						else {
							kappa_parallel += 0.5*p_kappa_n[izTri[iPrev]];
						}
					}
					//grad_T.x = 0.5*(
					//	(shared_T[threadIdx.x] + T_anti)*(info.pos.y - pos_anti.y)
					//	+ (T_clock + shared_T[threadIdx.x])*(pos_clock.y - info.pos.y)
					//	+ (T_out + T_clock)*(pos_out.y - pos_clock.y)
					//	+ (T_anti + T_out)*(pos_anti.y - pos_out.y)
					//	) / Area_quadrilateral;
					//grad_T.y = -0.5*( // notice minus
					//	(shared_T[threadIdx.x] + T_anti)*(info.pos.x - pos_anti.x)
					//	+ (T_clock + shared_T[threadIdx.x])*(pos_clock.x - info.pos.x)
					//	+ (T_out + T_clock)*(pos_out.x - pos_clock.x)
					//	+ (T_anti + T_out)*(pos_anti.x - pos_out.x)
					//	) / Area_quadrilateral;
					//ourrates.NnTn += TWOTHIRDS * kappa_parallel * grad_T.dot(edge_normal);

					// really, d_NTrates_by_dT :
					f64 d_NT_by_dT_clock = (TWOTHIRDS*kappa_parallel / Area_quadrilateral)*
						(0.5*edge_normal.x*(pos_out.y - info.pos.y)
							- 0.5* edge_normal.y* (pos_out.x - info.pos.x));
					f64 d_NT_by_dT_opp = (TWOTHIRDS*kappa_parallel / Area_quadrilateral)*
						(0.5*edge_normal.x*(pos_anti.y - pos_clock.y)
							- 0.5* edge_normal.y*(pos_anti.x - pos_clock.x));
					f64 d_NT_by_dT_anti = (TWOTHIRDS*kappa_parallel / Area_quadrilateral)*
						(0.5*edge_normal.x* (info.pos.y - pos_out.y)
							- 0.5*edge_normal.y* (info.pos.x - pos_out.x));
					f64 d_NT_by_dT_own = (TWOTHIRDS*kappa_parallel / Area_quadrilateral)*
						(0.5*edge_normal.x*(pos_clock.y - pos_anti.y)
							- 0.5*edge_normal.y*(pos_clock.x - pos_anti.x));



					// eps involves factor Nn: T_k+1 - [ T_k + h dNnTn/dt / Nn ]

					// Needed to collect deps/d T_clock, deps/d T_opp, deps/d T_anti, deps/d T_own

					// 	f64 epsilon_e = p_T_putative[iVertex].Te - p_T_k[iVertex].Te - (h_sub / N)*Rates.NeTe;
					// Note the minus so again it looks like we got the sign right:
#ifdef BWDSIDET

					d_eps_by_d_beta += (
						d_NT_by_dT_clock*regressor_clock
						+ d_NT_by_dT_opp*regressor_out
						+ d_NT_by_dT_anti*regressor_anti
						+ d_NT_by_dT_own*shared_x[threadIdx.x])*(-h_use / Nn);

#else
					d_eps_by_d_beta += (
						d_NT_by_dT_opp*regressor_out
						+ d_NT_by_dT_own*shared_x[threadIdx.x])*(-h_use / Nn);
#endif
				} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

				  // Now go round:	
				endpt_clock = endpt_anti;
				pos_clock = pos_out;
				pos_out = pos_anti;
#ifdef BWDSIDET
				regressor_clock = regressor_out; // We are going to need T_out but we are not going to need T_clock
				regressor_out = regressor_anti;
#endif
			}; // next iNeigh

			dz_d_eps_by_dbeta_n[iVertex] = d_eps_by_d_beta;
		}; // was it DOMAIN_VERTEX? Do what otherwise?

	}; // was it OUTERMOST/INNERMOST?

	__syncthreads();

	// Did we make sure to include a call to syncthreads every time we carried on to update shared memory data in every other routine?
	// ##################################################################################################################################

	{
		memcpy(&(shared_kappa[threadIdx.x * 2]), p_kappa_i + 2 * iVertex, 2 * sizeof(f64));
		memcpy(&(shared_nu[threadIdx.x * 2]), p_nu_i + 2 * iVertex, 2 * sizeof(f64));
	}
	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {
		shared_x[threadIdx.x] = p_regressor_i[iVertex];
	}
	else {
		shared_x[threadIdx.x] = 0.0;
	}

	__syncthreads();

	if ((info.flag == INNERMOST) || (info.flag == OUTERMOST))
	{
		// [ Ignore flux into edge of outermost vertex I guess ???]
	}
	else {
		if (info.flag == DOMAIN_VERTEX) {
			d_eps_by_d_beta = shared_x[threadIdx.x]; // T_self affects eps directly.

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_clock = shared_pos_verts[indexneigh - StartMajor];

#ifdef BWDSIDET
				regressor_clock = shared_x[indexneigh - StartMajor];
#endif
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_clock = info2.pos;

#ifdef BWDSIDET
				regressor_clock = p_regressor_i[indexneigh];
#endif
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == NEEDS_ANTI) {
				pos_clock = Anticlock_rotate2(pos_clock);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_clock = Clockwise_rotate2(pos_clock);
			};

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_out = shared_pos_verts[indexneigh - StartMajor];

#ifdef BWDSIDET
				regressor_out = shared_x[indexneigh - StartMajor];
#endif
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_out = info2.pos;

#ifdef BWDSIDET
				regressor_out = p_regressor_i[indexneigh];
#endif
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
			if (PBC == NEEDS_ANTI) {
				pos_out = Anticlock_rotate2(pos_out);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_out = Clockwise_rotate2(pos_out);
			};

			if ((izTri[info.neigh_len - 1] >= StartMinor) && (izTri[info.neigh_len - 1] < EndMinor))
			{
				endpt_clock = shared_pos[izTri[info.neigh_len - 1] - StartMinor];
			}
			else {
#ifdef CENTROID_HEATCONDUCTION
				endpt_clock = p_info_minor[izTri[info.neigh_len - 1]].pos;
#else
				endpt_clock = p_cc[izTri[info.neigh_len - 1]];
#endif
			}
			PBC = PBCtri[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == ROTATE_ME_CLOCKWISE) endpt_clock = Clockwise_d * endpt_clock;
			if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_clock = Anticlockwise_d * endpt_clock;

#ifdef BWDSIDET
			if (regressor_clock == 0.0) {
				regressor_clock = 0.5*(shared_x[threadIdx.x] + regressor_out);
			};
#endif

			f64 N = p_AreaMajor[iVertex] * p_n_major[iVertex].n;

#pragma unroll MAXNEIGH_d
			for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
			{
				{
					short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNext];
				}
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					pos_anti = shared_pos_verts[indexneigh - StartMajor];

#ifdef BWDSIDET
					regressor_anti = shared_x[indexneigh - StartMajor];
#endif
				}
				else {
					structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
					pos_anti = info2.pos;
#ifdef BWDSIDET
					regressor_anti = p_regressor_i[indexneigh];
#endif
				};
				if (PBC == NEEDS_ANTI) {
					pos_anti = Anticlock_rotate2(pos_anti);
				};
				if (PBC == NEEDS_CLOCK) {
					pos_anti = Clockwise_rotate2(pos_anti);
				};

#ifdef BWDSIDET
				if (regressor_anti == 0.0) {
					regressor_anti = 0.5*(shared_x[threadIdx.x] + regressor_out);
				}; // So we are receiving 0 then doing this. But how come?
#else
				indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					regressor_out = shared_x[indexneigh - StartMajor];
				}
				else {
					regressor_out = p_regressor_i[indexneigh];
				};
#endif

				if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
				{
					endpt_anti = shared_pos[izTri[iNeigh] - StartMinor];
				}
				else {
#ifdef CENTROID_HEATCONDUCTION
					endpt_anti = p_info_minor[izTri[iNeigh]].pos;
#else
					endpt_anti = p_cc[izTri[iNeigh]];
#endif					
				}
				PBC = PBCtri[MAXNEIGH_d*threadIdx.x + iNeigh];
				if (PBC == ROTATE_ME_CLOCKWISE) endpt_anti = Clockwise_d * endpt_anti;
				if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_anti = Anticlockwise_d * endpt_anti;

				edge_normal.x = (endpt_anti.y - endpt_clock.y);
				edge_normal.y = (endpt_clock.x - endpt_anti.x);

				// SMARTY:
				if (pos_out.x*pos_out.x + pos_out.y*pos_out.y >
					DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					f64 Area_quadrilateral = 0.5*(
						(info.pos.x + pos_anti.x)*(info.pos.y - pos_anti.y)
						+ (pos_clock.x + info.pos.x)*(pos_clock.y - info.pos.y)
						+ (pos_out.x + pos_clock.x)*(pos_out.y - pos_clock.y)
						+ (pos_anti.x + pos_out.x)*(pos_anti.y - pos_out.y)
						);
					kappa_parallel = 0.0;
					f64 nu;
					if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
					{
						kappa_parallel = 0.5*shared_kappa[izTri[iNeigh] - StartMinor];
						nu = 0.5*shared_nu[izTri[iNeigh] - StartMinor];
					}
					else {
						kappa_parallel = 0.5*p_kappa_i[izTri[iNeigh]];
						nu = 0.5*p_nu_i[izTri[iNeigh]];
					};
					{
						short iPrev = iNeigh - 1; if (iPrev < 0) iPrev = info.neigh_len - 1;
						if ((izTri[iPrev] >= StartMinor) && (izTri[iPrev] < EndMinor))
						{
							kappa_parallel += 0.5*shared_kappa[izTri[iPrev] - StartMinor];
							nu += 0.5*shared_nu[izTri[iPrev] - StartMinor];
						}
						else {
							kappa_parallel += 0.5*p_kappa_i[izTri[iPrev]];
							nu += 0.5*p_nu_i[izTri[iPrev]];
						}
					}

					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						B_out = shared_B[indexneigh - StartMajor];
					}
					else {
						f64_vec3 B_out3 = p_B_major[indexneigh];
						B_out = B_out3.xypart();
					}
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if (PBC == NEEDS_ANTI) 	B_out = Anticlock_rotate2(B_out);
					if (PBC == NEEDS_CLOCK)	B_out = Clockwise_rotate2(B_out);

					//kappa.xx = kappa_parallel * (nu_eHeart*nu_eHeart + omega.x*omega.x) / (nu_eHeart * nu_eHeart + omega_sq);
					//kappa.xy = kappa_parallel * (omega.x*omega.y - nu_eHeart *omega.z) / (nu_eHeart * nu_eHeart + omega_sq);
					//kappa.yx = kappa_parallel * (omega.x*omega.y + nu_eHeart * omega.z) / (nu_eHeart * nu_eHeart + omega_sq);
					//kappa.yy = kappa_parallel * (omega.y*omega.y + nu_eHeart * nu_eHeart) / (nu_eHeart * nu_eHeart + omega_sq);

					{ // scoping brace
						f64_vec3 omega = Make3(qoverMc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qoverMc);

						// eps involves factor Nn: T_k+1 - [ T_k + h dNnTn/dt / Nn ]
						// Needed to collect deps/d T_clock, deps/d T_opp, deps/d T_anti, deps/d T_own

						f64 coeff_NT_on_dTbydx = TWOTHIRDS * kappa_parallel *(
							edge_normal.x*(nu*nu + omega.x*omega.x) + edge_normal.y*(omega.x*omega.y - nu * omega.z)
							) / (nu * nu + omega.dot(omega));

						f64 coeff_NT_on_dTbydy = TWOTHIRDS * kappa_parallel *(
							edge_normal.x*(omega.x*omega.y + nu * omega.z) + edge_normal.y*(omega.y*omega.y + nu * nu)
							) / (nu * nu + omega.dot(omega));

#ifdef BWDSIDET

#ifndef LONGITUDINAL						

						f64 d_NT_by_dT_opp = (coeff_NT_on_dTbydx*0.5*(pos_anti.y - pos_clock.y)
							- coeff_NT_on_dTbydy*0.5*(pos_anti.x - pos_clock.x)) / Area_quadrilateral;
						f64 d_NT_by_dT_own = (coeff_NT_on_dTbydx*0.5*(pos_clock.y - pos_anti.y)
							- coeff_NT_on_dTbydy*0.5*(pos_clock.x - pos_anti.x)) / Area_quadrilateral;
						f64 d_NT_by_dT_clock = (coeff_NT_on_dTbydx*0.5*(pos_out.y - info.pos.y)
							- coeff_NT_on_dTbydy*0.5*(pos_out.x - info.pos.x)) / Area_quadrilateral;
						f64 d_NT_by_dT_anti = (coeff_NT_on_dTbydx*0.5*(info.pos.y - pos_out.y)
							- coeff_NT_on_dTbydy*0.5*(info.pos.x - pos_out.x)) / Area_quadrilateral;
						
						d_eps_by_d_beta += (
							d_NT_by_dT_clock*regressor_clock
							+ d_NT_by_dT_opp*regressor_out
							+ d_NT_by_dT_anti*regressor_anti
							+ d_NT_by_dT_own*shared_x[threadIdx.x])*(-h_use / N);
#else
						// Longitudinal:

						f64 delta_out = sqrt((info.pos.x - pos_out.x)*(info.pos.x - pos_out.x) + (info.pos.y - pos_out.y)*(info.pos.y - pos_out.y));
						f64 edgelen = edge_normal.modulus();

						f64 d_NT_by_dT_opp = TWOTHIRDS * kappa_parallel *  (1.0) *
							(nu*nu*edgelen*edgelen + omega.dotxy(edge_normal)*omega.dotxy(edge_normal))
							/ (delta_out*edgelen *(nu * nu + omega.dot(omega)));

						f64 d_NT_by_dT_own = -d_NT_by_dT_opp;

						d_eps_by_d_beta += (d_NT_by_dT_opp*regressor_out + d_NT_by_dT_own*shared_x[threadIdx.x])*(-h_use / N);
						 
#endif	

#else

						f64 d_NT_by_dT_opp = (coeff_NT_on_dTbydx*0.5*(pos_anti.y - pos_clock.y)
							- coeff_NT_on_dTbydy*0.5*(pos_anti.x - pos_clock.x)) / Area_quadrilateral;

						f64 d_NT_by_dT_own = (coeff_NT_on_dTbydx*0.5*(pos_clock.y - pos_anti.y)
							- coeff_NT_on_dTbydy*0.5*(pos_clock.x - pos_anti.x)) / Area_quadrilateral;	
						
						d_eps_by_d_beta += (
							+ d_NT_by_dT_opp*regressor_out
							+ d_NT_by_dT_own*shared_x[threadIdx.x])*(-h_use / N);
#endif
						//	ourrates.NiTi += TWOTHIRDS * kappa_parallel *(
						//		edge_normal.x*((nu*nu + omega.x*omega.x)*grad_T.x + (omega.x*omega.y + nu * omega.z)*grad_T.y)
						//		+ edge_normal.y*((omega.x*omega.y - nu * omega.z)*grad_T.x + (omega.y*omega.y + nu * nu)*grad_T.y)
						//		) / (nu * nu + omega.dot(omega));

					}
				} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

				  // Now go round:	
				endpt_clock = endpt_anti;
				pos_clock = pos_out;
				pos_out = pos_anti;
#ifdef BWDSIDET
				regressor_clock = regressor_out;
				regressor_out = regressor_anti;
#endif

			}; // next iNeigh

			dz_d_eps_by_dbeta_i[iVertex] = d_eps_by_d_beta;

		}; // was it DOMAIN_VERTEX? Do what otherwise?
	}; // was it OUTERMOST/INNERMOST?


	__syncthreads();

	// Did we make sure to include a call to syncthreads every time we carried on to update shared memory data in every other routine?
	// ##################################################################################################################################

	{
		memcpy(&(shared_kappa[threadIdx.x * 2]), p_kappa_e + 2 * iVertex, 2 * sizeof(f64));
		memcpy(&(shared_nu[threadIdx.x * 2]), p_nu_e + 2 * iVertex, 2 * sizeof(f64));
	}
	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {
		shared_x[threadIdx.x] = p_regressor_e[iVertex];
	}
	else {
		shared_x[threadIdx.x] = 0.0;
	}

	__syncthreads();


	if ((info.flag == INNERMOST) || (info.flag == OUTERMOST))
	{
		// [ Ignore flux into edge of outermost vertex I guess ???]
	}
	else {
		if (info.flag == DOMAIN_VERTEX) {
			d_eps_by_d_beta = shared_x[threadIdx.x]; // T_self affects eps directly.

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_clock = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
				regressor_clock = shared_x[indexneigh - StartMajor];
#endif
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_clock = info2.pos;
#ifdef BWDSIDET
				regressor_clock = p_regressor_e[indexneigh];
#endif
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == NEEDS_ANTI) {
				pos_clock = Anticlock_rotate2(pos_clock);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_clock = Clockwise_rotate2(pos_clock);
			};

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_out = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
				regressor_out = shared_x[indexneigh - StartMajor];
#endif
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_out = info2.pos;
#ifdef BWDSIDET
				regressor_out = p_regressor_e[indexneigh];
#endif
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
			if (PBC == NEEDS_ANTI) {
				pos_out = Anticlock_rotate2(pos_out);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_out = Clockwise_rotate2(pos_out);
			};

			if ((izTri[info.neigh_len - 1] >= StartMinor) && (izTri[info.neigh_len - 1] < EndMinor))
			{
				endpt_clock = shared_pos[izTri[info.neigh_len - 1] - StartMinor];
			}
			else {
#ifdef CENTROID_HEATCONDUCTION
				endpt_clock = p_info_minor[izTri[info.neigh_len - 1]].pos;
#else
				endpt_clock = p_cc[izTri[info.neigh_len - 1]];
#endif
			}
			PBC = PBCtri[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == ROTATE_ME_CLOCKWISE) endpt_clock = Clockwise_d * endpt_clock;
			if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_clock = Anticlockwise_d * endpt_clock;
#ifdef BWDSIDET
			if (regressor_clock == 0.0) {
				regressor_clock = 0.5*(shared_x[threadIdx.x] + regressor_out);
			};
#endif
			f64 N = p_AreaMajor[iVertex] * p_n_major[iVertex].n;

#pragma unroll MAXNEIGH_d
			for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
			{
				{
					short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNext];
				}
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					pos_anti = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
					regressor_anti = shared_x[indexneigh - StartMajor];
#endif
				}
				else {
					structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
					pos_anti = info2.pos;
#ifdef BWDSIDET
					regressor_anti = p_regressor_e[indexneigh];
#endif
				};
				if (PBC == NEEDS_ANTI) {
					pos_anti = Anticlock_rotate2(pos_anti);
				};
				if (PBC == NEEDS_CLOCK) {
					pos_anti = Clockwise_rotate2(pos_anti);
				};
#ifdef BWDSIDET

				if (regressor_anti == 0.0) {
					regressor_anti = 0.5*(shared_x[threadIdx.x] + regressor_out);
				}; // So we are receiving 0 then doing this. But how come?
#else
				indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					regressor_out = shared_x[indexneigh - StartMajor];
				}
				else {
					regressor_out = p_regressor_e[indexneigh];
				};
#endif
				if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
				{
					endpt_anti = shared_pos[izTri[iNeigh] - StartMinor];

				}
				else {
#ifdef CENTROID_HEATCONDUCTION
					endpt_anti = p_info_minor[izTri[iNeigh]].pos;
#else
					endpt_anti = p_cc[izTri[iNeigh]];
#endif
				}
				PBC = PBCtri[MAXNEIGH_d*threadIdx.x + iNeigh];
				if (PBC == ROTATE_ME_CLOCKWISE) endpt_anti = Clockwise_d * endpt_anti;
				if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_anti = Anticlockwise_d * endpt_anti;

				// It decided to rotate something it shouldn't oughta. Rotated tri 23600 = tri 2 for 11582.

				edge_normal.x = (endpt_anti.y - endpt_clock.y);
				edge_normal.y = (endpt_clock.x - endpt_anti.x);

				// SMARTY:
				if (pos_out.x*pos_out.x + pos_out.y*pos_out.y >
					DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					f64 Area_quadrilateral = 0.5*(
						(info.pos.x + pos_anti.x)*(info.pos.y - pos_anti.y)
						+ (pos_clock.x + info.pos.x)*(pos_clock.y - info.pos.y)
						+ (pos_out.x + pos_clock.x)*(pos_out.y - pos_clock.y)
						+ (pos_anti.x + pos_out.x)*(pos_anti.y - pos_out.y)
						);

					kappa_parallel = 0.0;
					f64 nu;
					if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
					{
						kappa_parallel = 0.5*shared_kappa[izTri[iNeigh] - StartMinor];
						nu = 0.5*shared_nu[izTri[iNeigh] - StartMinor];
					}
					else {
						kappa_parallel = 0.5*p_kappa_e[izTri[iNeigh]];
						nu = 0.5*p_nu_e[izTri[iNeigh]];
					};

					{
						short iPrev = iNeigh - 1; if (iPrev < 0) iPrev = info.neigh_len - 1;
						if ((izTri[iPrev] >= StartMinor) && (izTri[iPrev] < EndMinor))
						{
							kappa_parallel += 0.5*shared_kappa[izTri[iPrev] - StartMinor];
							nu += 0.5*shared_nu[izTri[iPrev] - StartMinor];
						}
						else {
							kappa_parallel += 0.5*p_kappa_e[izTri[iPrev]];
							nu += 0.5*p_nu_e[izTri[iPrev]];
						}
					}

					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						B_out = shared_B[indexneigh - StartMajor];
					}
					else {
						f64_vec3 B_out3 = p_B_major[indexneigh];
						B_out = B_out3.xypart();
					}
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if (PBC == NEEDS_ANTI) 	B_out = Anticlock_rotate2(B_out);
					if (PBC == NEEDS_CLOCK)	B_out = Clockwise_rotate2(B_out);

					{ // scoping brace
						f64_vec3 omega = Make3(qovermc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qovermc);
						// PROBABLY ALWAYS SPILLED INTO GLOBAL -- WHAT CAN WE DO?

						f64 coeff_NT_on_dTbydx = TWOTHIRDS * kappa_parallel *(
							edge_normal.x*(nu*nu + omega.x*omega.x) + edge_normal.y*(omega.x*omega.y - nu * omega.z)
							) / (nu * nu + omega.dot(omega));

						f64 coeff_NT_on_dTbydy = TWOTHIRDS * kappa_parallel *(
							edge_normal.x*(omega.x*omega.y + nu * omega.z) + edge_normal.y*(omega.y*omega.y + nu * nu)
							) / (nu * nu + omega.dot(omega));

#ifdef BWDSIDET

#ifndef LONGITUDINAL						

						f64 d_NT_by_dT_opp = (coeff_NT_on_dTbydx*0.5*(pos_anti.y - pos_clock.y)
							- coeff_NT_on_dTbydy*0.5*(pos_anti.x - pos_clock.x)) / Area_quadrilateral;
						f64 d_NT_by_dT_own = (coeff_NT_on_dTbydx*0.5*(pos_clock.y - pos_anti.y)
							- coeff_NT_on_dTbydy*0.5*(pos_clock.x - pos_anti.x)) / Area_quadrilateral;
						f64 d_NT_by_dT_clock = (coeff_NT_on_dTbydx*0.5*(pos_out.y - info.pos.y)
							- coeff_NT_on_dTbydy*0.5*(pos_out.x - info.pos.x)) / Area_quadrilateral;
						f64 d_NT_by_dT_anti = (coeff_NT_on_dTbydx*0.5*(info.pos.y - pos_out.y)
							- coeff_NT_on_dTbydy*0.5*(info.pos.x - pos_out.x)) / Area_quadrilateral;

						d_eps_by_d_beta += (
							d_NT_by_dT_clock*regressor_clock
							+ d_NT_by_dT_opp*regressor_out
							+ d_NT_by_dT_anti*regressor_anti
							+ d_NT_by_dT_own*shared_x[threadIdx.x])*(-h_use / N);
#else
						// Longitudinal:

						f64 delta_out = sqrt((info.pos.x - pos_out.x)*(info.pos.x - pos_out.x) + (info.pos.y - pos_out.y)*(info.pos.y - pos_out.y));
						f64 edgelen = edge_normal.modulus();

						f64 d_NT_by_dT_opp = TWOTHIRDS * kappa_parallel *  (1.0) *
							(nu*nu*edgelen*edgelen + omega.dotxy(edge_normal)*omega.dotxy(edge_normal))
							/ (delta_out*edgelen *(nu * nu + omega.dot(omega)));

						f64 d_NT_by_dT_own = -d_NT_by_dT_opp;

						d_eps_by_d_beta += (d_NT_by_dT_opp*regressor_out + d_NT_by_dT_own*shared_x[threadIdx.x])*(-h_use / N);

#endif	

#else

						f64 d_NT_by_dT_opp = (coeff_NT_on_dTbydx*0.5*(pos_anti.y - pos_clock.y)
							- coeff_NT_on_dTbydy*0.5*(pos_anti.x - pos_clock.x)) / Area_quadrilateral;

						f64 d_NT_by_dT_own = (coeff_NT_on_dTbydx*0.5*(pos_clock.y - pos_anti.y)
							- coeff_NT_on_dTbydy*0.5*(pos_clock.x - pos_anti.x)) / Area_quadrilateral;

						d_eps_by_d_beta += (
							+ d_NT_by_dT_opp*regressor_out
							+ d_NT_by_dT_own*shared_x[threadIdx.x])*(-h_use / N);
#endif
					}
				} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

				  // Now go round:	
				endpt_clock = endpt_anti;
				pos_clock = pos_out;
				pos_out = pos_anti;
#ifdef BWDSIDET
				regressor_clock = regressor_out;
				regressor_out = regressor_anti;
#endif
			}; // next iNeigh

			   // Shan't we choose 3 separate beta -- there is a 3 vector of epsilons. Ah.

			dz_d_eps_by_dbeta_e[iVertex] = d_eps_by_d_beta; // where zeroed?

		}; // was it DOMAIN_VERTEX? Do what otherwise?
	}; // was it OUTERMOST/INNERMOST?

}

/*
__global__ void kernelCalculateROCepsWRTregressorT_volleys(
	f64 const h_use,
	structural * __restrict__ p_info_minor,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,
	long * __restrict__ izTri_verts,
	char * __restrict__ szPBCtri_verts,
	f64_vec2 * __restrict__ p_cc,

	nvals * __restrict__ p_n_major,
	T3 * __restrict__ p_T_major,
	
	f64_vec3 * __restrict__ p_B_major,

	f64 * __restrict__ p_kappa_n,
	f64 * __restrict__ p_kappa_i,
	f64 * __restrict__ p_kappa_e,

	f64 * __restrict__ p_nu_i,
	f64 * __restrict__ p_nu_e,
	f64 * __restrict__ p_AreaMajor,
	
	char * __restrict__ p_iVolley,
	f64 * __restrict__ p_regressor_n,
	f64 * __restrict__ p_regressor_i,
	f64 * __restrict__ p_regressor_e,

	f64_vec4 * __restrict__ dz_d_eps_by_dbeta_n_x4,
	f64_vec4 * __restrict__ dz_d_eps_by_dbeta_i_x4,
	f64_vec4 * __restrict__ dz_d_eps_by_dbeta_e_x4
	)
{
	// ******************************************
	//  1. Do this with kappa given from t_k etc
	// ******************************************
	// Then come and change it.

	// Think we might as well take kappa_par and nu from triangles really.
	// If n is modelled well then hopefully a nearby high-n does not have a big impact.
	
	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajorClever]; // 2
	__shared__ f64_vec2 shared_pos[2 * threadsPerTileMajorClever]; // Need why? If using centroid? Sometimes on boundary - is that only reason?
	// Seems like a waste of 2 doubles. Can just average -- and shift if we load the flag that tells us to.

	__shared__ f64 shared_x[threadsPerTileMajorClever];      
	// Inefficiency caused by our making a 3-struct. We didn't need to do that.

	__shared__ f64_vec2 shared_B[threadsPerTileMajorClever]; // +2
															 // B is smooth. Unfortunately we have not fitted in Bz here.
															 // In order to do that perhaps rewrite so that variables are overwritten in shared.
															 // This way is easier for NOW.
	// So that's the optimization we want. Scalar arrays for T, nu.

	__shared__ f64 shared_kappa[threadsPerTileMajorClever * 2];
	__shared__ f64 shared_nu[threadsPerTileMajorClever * 2];

	__shared__ long Indexneigh[MAXNEIGH_d*threadsPerTileMajorClever]; // assume 48 bytes = 4*12 = 6 doubles

	__shared__ char PBCneigh[MAXNEIGH_d*threadsPerTileMajorClever]; // 12 bytes each from L1. Have 42 per thread at 384 threads.
																	// We should at any rate try a major block of size 256. If 1 block will run at a time, so be it.
																	// Leave L1 in case of register overflow into it. <-- don't know how likely - do we only have 31 doubles in registry
																	// regardless # of threads and space? Or can be 63?
	__shared__ char PBCtri[MAXNEIGH_d*threadsPerTileMajorClever];
	// Balance of shared vs L1: 24*256*8 = 48K. That leaves 8 doublesworth in L1 for variables.

	// Why do we think that we shouldn't move izTri into shared? Looks like only 18 doublesworth above.

	__shared__ char shared_iVolley[threadsPerTileMajorClever];

	// Given that we added this, shall we also swap izTri into shared and take something else out? More space in L1 would help.a.lot.
	
	long izTri[MAXNEIGH_d];  // so only 2 doubles left in L1. 31 in registers??
							 // Set threadsPerTileMajorClever to 256.

							 // It would help matters if we get rid of T3. We might as well therefore change to scalar flatpack T.
							 // We are hoping that it works well loading kappa(tri) and that this is not upset by nearby values. Obviously a bit of an experiment.

							 // Does make it somewhat laughable that we go to such efforts to reduce global accesses when we end up overflowing anyway. 
							 // If we can fit 24 doubles/thread in 48K that means we can fit 8 doubles/thread in 16K so that's most of L1 used up.
	
	long const StartMajor = blockIdx.x*blockDim.x;
	long const EndMajor = StartMajor + blockDim.x;
	long const StartMinor = blockIdx.x*blockDim.x * 2;
	long const EndMinor = StartMinor + blockDim.x * 2;
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX	// 2.5 double

	structural info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];     // 3 double
	shared_pos_verts[threadIdx.x] = info.pos;
	
	shared_iVolley[threadIdx.x] = p_iVolley[iVertex]; // contiguous load of char

#ifdef CENTROID_HEATCONDUCTION
	{
		structural infotemp[2];
		memcpy(infotemp, p_info_minor + 2 * iVertex, 2 * sizeof(structural));
		shared_pos[threadIdx.x * 2] = infotemp[0].pos;
		shared_pos[threadIdx.x * 2 + 1] = infotemp[1].pos;
		memcpy(&(shared_kappa[threadIdx.x * 2]), p_kappa_n + 2 * iVertex, 2 * sizeof(f64));
	}
#else
	{
		memcpy(&(shared_pos[threadIdx.x * 2]), p_cc + 2 * iVertex, 2 * sizeof(f64_vec2));
		memcpy(&(shared_kappa[threadIdx.x * 2]), p_kappa_n + 2 * iVertex, 2 * sizeof(f64));		
	}
#endif

	
	// No nu to set for neutrals - not used

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {
				
		shared_B[threadIdx.x] = p_B_major[iVertex].xypart();
		shared_x[threadIdx.x] = p_regressor_n[iVertex];
	} else {
		// SHOULD NOT BE LOOKING INTO INS. How do we avoid?
		
		memset(&(shared_B[threadIdx.x]), 0, sizeof(f64_vec2));
		shared_x[threadIdx.x] = 0.0;
	}
	
	__syncthreads();
	// works if we cut here

	f64 regressor_anti, regressor_clock, regressor_out;		// 5
	f64_vec2 pos_clock, pos_anti, pos_out;   // +6
	f64_vec2 B_out;       // +2
	// NTrates ourrates;      // +5
	f64 kappa_parallel; 
	f64 nu;                // 20 there  
	f64_vec2 edge_normal;  // 22
	f64_vec2 endpt_anti;    // 24 .. + 6 from above
	long indexneigh;     // into the 2-double buffer in L1
	f64_vec2 endpt_clock;    // As we only use endpt_anti afterwords we could union endpt_clock with edge_normal
							 // Come back and optimize by checking which things we need in scope at the same time?
	f64_vec4 d_eps_by_d_beta;

	short iNeigh; // only fixed # of addresses so short makes no difference.
	char PBC; // char makes no difference.

	if ((info.flag == INNERMOST) || (info.flag == OUTERMOST))
	{
		// [ Ignore flux into edge of outermost vertex I guess ???]
	} else {
		if (info.flag == DOMAIN_VERTEX) {			
			
			// The idea of not sending blocks full of non-domain vertices is another idea. Fiddly with indices.
			memset(&d_eps_by_d_beta, 0, sizeof(f64_vec4));
			d_eps_by_d_beta.x[shared_iVolley[threadIdx.x]] = shared_x[threadIdx.x]; // T_self affects eps directly.

			// correct sign as :
			// 	f64 epsilon_e = p_T_putative[iVertex].Te - p_T_k[iVertex].Te - (h_sub / N)*Rates.NeTe;
			
			memcpy(Indexneigh + MAXNEIGH_d * threadIdx.x,
				pIndexNeigh + MAXNEIGH_d * iVertex,
				MAXNEIGH_d * sizeof(long));
			memcpy(PBCneigh + MAXNEIGH_d * threadIdx.x,
				pPBCNeigh + MAXNEIGH_d * iVertex,
				MAXNEIGH_d * sizeof(char));
			memcpy(PBCtri + MAXNEIGH_d * threadIdx.x,
				szPBCtri_verts + MAXNEIGH_d * iVertex,
				MAXNEIGH_d * sizeof(char));
			memcpy(izTri, //+ MAXNEIGH_d * threadIdx.x,
				izTri_verts + MAXNEIGH_d * iVertex, MAXNEIGH_d * sizeof(long));
				

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_clock = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
				regressor_clock = shared_x[indexneigh - StartMajor];
#endif 
				// In case that T_clock is an explicit value, the regressor clockwise value is irrelevant!
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_clock = info2.pos;
#ifdef BWDSIDET
				regressor_clock = p_regressor_n[indexneigh];
#endif
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == NEEDS_ANTI) { pos_clock = Anticlock_rotate2(pos_clock); };
			if (PBC == NEEDS_CLOCK) { pos_clock = Clockwise_rotate2(pos_clock); };
			
			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_out = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
				regressor_out = shared_x[indexneigh - StartMajor];
#endif
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_out = info2.pos;
#ifdef BWDSIDET
				regressor_out = p_regressor_n[indexneigh];
#endif
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
			if (PBC == NEEDS_ANTI) { pos_out = Anticlock_rotate2(pos_out);};
			if (PBC == NEEDS_CLOCK) { pos_out = Clockwise_rotate2(pos_out);	};

			if ((izTri[info.neigh_len - 1] >= StartMinor) && (izTri[info.neigh_len - 1] < EndMinor))
			{
				endpt_clock = shared_pos[izTri[info.neigh_len - 1] - StartMinor];
			} else {
#ifdef CENTROID_HEATCONDUCTION
				endpt_clock = p_info_minor[izTri[info.neigh_len - 1]].pos;
#else
				endpt_clock = p_cc[izTri[info.neigh_len - 1]];
#endif
			}
			PBC = PBCtri[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == ROTATE_ME_CLOCKWISE) endpt_clock = Clockwise_d * endpt_clock;
			if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_clock = Anticlockwise_d * endpt_clock;

#ifdef BWDSIDET
			if (regressor_clock == 0.0) {
				regressor_clock = 0.5*(shared_x[threadIdx.x] + regressor_out);
			};
#endif

			f64 Nn = p_AreaMajor[iVertex] * p_n_major[iVertex].n_n;
			
#pragma unroll MAXNEIGH_d
			for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
			{
				{
					short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNext];
				}
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					pos_anti = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
					regressor_anti = shared_x[indexneigh - StartMajor];
#endif
				}
				else {
					structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
					pos_anti = info2.pos;
#ifdef BWDSIDET
					regressor_anti = p_regressor_n[indexneigh];
#endif
				};
				if (PBC == NEEDS_ANTI) {
					pos_anti = Anticlock_rotate2(pos_anti);
				};
				if (PBC == NEEDS_CLOCK) {
					pos_anti = Clockwise_rotate2(pos_anti);
				};
#ifdef BWDSIDET
				if (regressor_anti == 0.0) {
					regressor_anti = 0.5*(shared_x[threadIdx.x] + regressor_out);
				}; // So we are receiving 0 then doing this. But how come?
#else 
				// Compute regressor_out on every go:
				indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					regressor_out = shared_x[indexneigh - StartMajor];
				} else {
					regressor_out = p_regressor_n[indexneigh];
				};
#endif

				// Now let's see
				// tri 0 has neighs 0 and 1 I'm pretty sure (check....) CHECK

				if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
				{
					endpt_anti = shared_pos[izTri[iNeigh] - StartMinor];
				}
				else {
#ifdef CENTROID_HEATCONDUCTION
					endpt_anti = p_info_minor[izTri[iNeigh]].pos;
#else
					endpt_anti = p_cc[izTri[iNeigh]];
#endif
				}
				PBC = PBCtri[MAXNEIGH_d*threadIdx.x + iNeigh];
				if (PBC == ROTATE_ME_CLOCKWISE) endpt_anti = Clockwise_d * endpt_anti;
				if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_anti = Anticlockwise_d * endpt_anti;

				edge_normal.x = (endpt_anti.y - endpt_clock.y);
				edge_normal.y = (endpt_clock.x - endpt_anti.x);
				
				// SMARTY:
				if (pos_out.x*pos_out.x + pos_out.y*pos_out.y >
					DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					f64 Area_quadrilateral = 0.5*(
						(info.pos.x + pos_anti.x)*(info.pos.y - pos_anti.y)
						+ (pos_clock.x + info.pos.x)*(pos_clock.y - info.pos.y)
						+ (pos_out.x + pos_clock.x)*(pos_out.y - pos_clock.y)
						+ (pos_anti.x + pos_out.x)*(pos_anti.y - pos_out.y)
						);

					kappa_parallel = 0.0;
					if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
					{
						kappa_parallel = 0.5*shared_kappa[izTri[iNeigh] - StartMinor];
					}
					else {
						kappa_parallel = 0.5*p_kappa_n[izTri[iNeigh]];
					};
					{
						short iPrev = iNeigh - 1; if (iPrev < 0) iPrev = info.neigh_len - 1;
						if ((izTri[iPrev] >= StartMinor) && (izTri[iPrev] < EndMinor))
						{
							kappa_parallel += 0.5*shared_kappa[izTri[iPrev] - StartMinor];
						}
						else {
							kappa_parallel += 0.5*p_kappa_n[izTri[iPrev]];
						}
					}
					//grad_T.x = 0.5*(
					//	(shared_T[threadIdx.x] + T_anti)*(info.pos.y - pos_anti.y)
					//	+ (T_clock + shared_T[threadIdx.x])*(pos_clock.y - info.pos.y)
					//	+ (T_out + T_clock)*(pos_out.y - pos_clock.y)
					//	+ (T_anti + T_out)*(pos_anti.y - pos_out.y)
					//	) / Area_quadrilateral;
					//grad_T.y = -0.5*( // notice minus
					//	(shared_T[threadIdx.x] + T_anti)*(info.pos.x - pos_anti.x)
					//	+ (T_clock + shared_T[threadIdx.x])*(pos_clock.x - info.pos.x)
					//	+ (T_out + T_clock)*(pos_out.x - pos_clock.x)
					//	+ (T_anti + T_out)*(pos_anti.x - pos_out.x)
					//	) / Area_quadrilateral;
					//ourrates.NnTn += TWOTHIRDS * kappa_parallel * grad_T.dot(edge_normal);

					
					// eps involves factor Nn: T_k+1 - [ T_k + h dNnTn/dt / Nn ]

					// Needed to collect deps/d T_clock, deps/d T_opp, deps/d T_anti, deps/d T_own

					// 	f64 epsilon_e = p_T_putative[iVertex].Te - p_T_k[iVertex].Te - (h_sub / N)*Rates.NeTe;
					// Note the minus so again it looks like we got the sign right:
				//	d_eps_by_d_beta += (
				//		  d_NT_by_dT_clock*regressor_clock
				//		+ d_NT_by_dT_opp*regressor_out
				//		+ d_NT_by_dT_anti*regressor_anti
				//		+ d_NT_by_dT_own*shared_x[threadIdx.x])*(-h_use / Nn);


			//		d_eps_by_d_beta.x[0] += (
			//			d_NT_by_dT_clock*regressor_clock*((iVolley_clock == 0)?1:0)
			//			+ d_NT_by_dT_opp*regressor_out*((iVolley_out == 0) ? 1 : 0)
			//			+ d_NT_by_dT_anti*regressor_anti*((iVolley_anti == 0) ? 1 : 0)
			//			+ d_NT_by_dT_own*shared_x[threadIdx.x]*((shared_iVolley[threadIdx.x] == 0)?1:0))*(-h_use / Nn);

					// Is there a better way? Yes.

#ifdef BWDSIDET
					{
						char iVolley_anti;
						short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
						indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
						if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
						{
							iVolley_anti = shared_iVolley[indexneigh - StartMajor];
						} else {
							iVolley_anti = p_iVolley[indexneigh];
						};
						f64 d_NT_by_dT_anti = (TWOTHIRDS*kappa_parallel / Area_quadrilateral)*
							(0.5*edge_normal.x* (info.pos.y - pos_out.y)
								- 0.5*edge_normal.y* (info.pos.x - pos_out.x));
						d_eps_by_d_beta.x[iVolley_anti] += d_NT_by_dT_anti*regressor_anti*(-h_use / Nn);
					}
#endif

					{
						char iVolley_out;
						indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];					
						if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
						{
							iVolley_out = shared_iVolley[indexneigh - StartMajor];
						} else {
							iVolley_out = p_iVolley[indexneigh];
						};
						// really, d_NTrates_by_dT :
						f64 d_NT_by_dT_opp = (TWOTHIRDS*kappa_parallel / Area_quadrilateral)*
							(0.5*edge_normal.x*(pos_anti.y - pos_clock.y)
								- 0.5* edge_normal.y*(pos_anti.x - pos_clock.x));
						d_eps_by_d_beta.x[iVolley_out] += d_NT_by_dT_opp*regressor_out*(-h_use / Nn);
					}
#ifdef BWDSIDET
					{
						char iVolley_clock;
						short iPrev = iNeigh - 1; if (iPrev<0) iPrev = info.neigh_len-1;
						indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iPrev];
						if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
						{
							iVolley_clock = shared_iVolley[indexneigh - StartMajor];
						} else {
							iVolley_clock = p_iVolley[indexneigh];
						};
						f64 d_NT_by_dT_clock = (TWOTHIRDS*kappa_parallel / Area_quadrilateral)*
							(0.5*edge_normal.x*(pos_out.y - info.pos.y)
								- 0.5* edge_normal.y* (pos_out.x - info.pos.x));
						d_eps_by_d_beta.x[iVolley_clock] += d_NT_by_dT_clock*regressor_clock*(-h_use / Nn);
					}
#endif
					f64 d_NT_by_dT_own = (TWOTHIRDS*kappa_parallel / Area_quadrilateral)*
						(0.5*edge_normal.x*(pos_clock.y - pos_anti.y)
							- 0.5*edge_normal.y*(pos_clock.x - pos_anti.x));
					d_eps_by_d_beta.x[shared_iVolley[threadIdx.x]] += d_NT_by_dT_own*shared_x[threadIdx.x] * (-h_use / Nn);

				} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

				// Now go round:	
				endpt_clock = endpt_anti;
				pos_clock = pos_out;
				pos_out = pos_anti;
#ifdef BWDSIDET
				regressor_clock = regressor_out; // We are going to need T_out but we are not going to need T_clock
				regressor_out = regressor_anti;
#endif

			}; // next iNeigh
			
			memcpy(&(dz_d_eps_by_dbeta_n_x4[iVertex]), &(d_eps_by_d_beta), sizeof(f64_vec4));

		}; // was it DOMAIN_VERTEX? Do what otherwise?
		
	}; // was it OUTERMOST/INNERMOST?


	// Let's say we proceed by repeating the above 4 times since we need to collect for each regressor.
	// We don't want to store multiple regressor_out?

	
	__syncthreads();
	
	// Did we make sure to include a call to syncthreads every time we carried on to update shared memory data in every other routine?
	// ##################################################################################################################################

	{
		memcpy(&(shared_kappa[threadIdx.x * 2]), p_kappa_i + 2 * iVertex, 2 * sizeof(f64));
		memcpy(&(shared_nu[threadIdx.x * 2]), p_nu_i + 2 * iVertex, 2 * sizeof(f64));
	}
	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {
		shared_x[threadIdx.x] = p_regressor_i[iVertex];
	} else {
		shared_x[threadIdx.x] = 0.0;
	}

	__syncthreads();

	if ((info.flag == INNERMOST) || (info.flag == OUTERMOST))
	{
		// [ Ignore flux into edge of outermost vertex I guess ???]
	}
	else {
		if (info.flag == DOMAIN_VERTEX) {
			memset(&d_eps_by_d_beta, 0, sizeof(f64_vec4));
			d_eps_by_d_beta.x[shared_iVolley[threadIdx.x]] = shared_x[threadIdx.x]; // T_self affects eps directly.

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_clock = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
				regressor_clock = shared_x[indexneigh - StartMajor];
#endif
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_clock = info2.pos;
#ifdef BWDSIDET
				regressor_clock = p_regressor_i[indexneigh];
#endif
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == NEEDS_ANTI) {
				pos_clock = Anticlock_rotate2(pos_clock);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_clock = Clockwise_rotate2(pos_clock);
			};

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_out = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
				regressor_out = shared_x[indexneigh - StartMajor];
#endif
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_out = info2.pos;
#ifdef BWDSIDET
				regressor_out = p_regressor_i[indexneigh];
#endif
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
			if (PBC == NEEDS_ANTI) {
				pos_out = Anticlock_rotate2(pos_out);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_out = Clockwise_rotate2(pos_out);
			};

			if ((izTri[info.neigh_len - 1] >= StartMinor) && (izTri[info.neigh_len - 1] < EndMinor))
			{
				endpt_clock = shared_pos[izTri[info.neigh_len - 1] - StartMinor];
			} else {
#ifdef CENTROID_HEATCONDUCTION
				endpt_clock = p_info_minor[izTri[info.neigh_len - 1]].pos;
#else
				endpt_clock = p_cc[izTri[info.neigh_len - 1]];
#endif
			}
			PBC = PBCtri[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == ROTATE_ME_CLOCKWISE) endpt_clock = Clockwise_d * endpt_clock;
			if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_clock = Anticlockwise_d * endpt_clock;
#ifdef BWDSIDET
			if (regressor_clock == 0.0) {
				regressor_clock = 0.5*(shared_x[threadIdx.x] + regressor_out);
			};
#endif

			f64 N = p_AreaMajor[iVertex] * p_n_major[iVertex].n;

#pragma unroll MAXNEIGH_d
			for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
			{
				{
					short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNext];
				}
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					pos_anti = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
					regressor_anti = shared_x[indexneigh - StartMajor];
#endif
				}
				else {
					structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
					pos_anti = info2.pos;
#ifdef BWDSIDET
					regressor_anti = p_regressor_i[indexneigh];
#endif
				};
				if (PBC == NEEDS_ANTI) {
					pos_anti = Anticlock_rotate2(pos_anti);
				};
				if (PBC == NEEDS_CLOCK) {
					pos_anti = Clockwise_rotate2(pos_anti);
				};
#ifdef BWDSIDET
				if (regressor_anti == 0.0) {
					regressor_anti = 0.5*(shared_x[threadIdx.x] + regressor_out);
				}; // So we are receiving 0 then doing this. But how come?
#else
				// Compute regressor_out on every go:
				indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					regressor_out = shared_x[indexneigh - StartMajor];
				}
				else {
					regressor_out = p_regressor_i[indexneigh];
				};
#endif
				if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
				{
					endpt_anti = shared_pos[izTri[iNeigh] - StartMinor];
				}
				else {
#ifdef CENTROID_HEATCONDUCTION
					endpt_anti = p_info_minor[izTri[iNeigh]].pos;
#else
					endpt_anti = p_cc[izTri[iNeigh]];
#endif					
				}
				PBC = PBCtri[MAXNEIGH_d*threadIdx.x + iNeigh];
				if (PBC == ROTATE_ME_CLOCKWISE) endpt_anti = Clockwise_d * endpt_anti;
				if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_anti = Anticlockwise_d * endpt_anti;

				edge_normal.x = (endpt_anti.y - endpt_clock.y);
				edge_normal.y = (endpt_clock.x - endpt_anti.x);

				// SMARTY:
				if (pos_out.x*pos_out.x + pos_out.y*pos_out.y >
					DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					f64 Area_quadrilateral = 0.5*(
						(info.pos.x + pos_anti.x)*(info.pos.y - pos_anti.y)
						+ (pos_clock.x + info.pos.x)*(pos_clock.y - info.pos.y)
						+ (pos_out.x + pos_clock.x)*(pos_out.y - pos_clock.y)
						+ (pos_anti.x + pos_out.x)*(pos_anti.y - pos_out.y)
						);
					kappa_parallel = 0.0;
					f64 nu;
					if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
					{
						kappa_parallel = 0.5*shared_kappa[izTri[iNeigh] - StartMinor];
						nu = 0.5*shared_nu[izTri[iNeigh] - StartMinor];
					}
					else {
						kappa_parallel = 0.5*p_kappa_i[izTri[iNeigh]];
						nu = 0.5*p_nu_i[izTri[iNeigh]];
					};
					{
						short iPrev = iNeigh - 1; if (iPrev < 0) iPrev = info.neigh_len - 1;
						if ((izTri[iPrev] >= StartMinor) && (izTri[iPrev] < EndMinor))
						{
							kappa_parallel += 0.5*shared_kappa[izTri[iPrev] - StartMinor];
							nu += 0.5*shared_nu[izTri[iPrev] - StartMinor];
						}
						else {
							kappa_parallel += 0.5*p_kappa_i[izTri[iPrev]];
							nu += 0.5*p_nu_i[izTri[iPrev]];
						}
					}

					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						B_out = shared_B[indexneigh - StartMajor];
					}
					else {
						f64_vec3 B_out3 = p_B_major[indexneigh];
						B_out = B_out3.xypart();
					}
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if (PBC == NEEDS_ANTI) 	B_out = Anticlock_rotate2(B_out);
					if (PBC == NEEDS_CLOCK)	B_out = Clockwise_rotate2(B_out);

					//kappa.xx = kappa_parallel * (nu_eHeart*nu_eHeart + omega.x*omega.x) / (nu_eHeart * nu_eHeart + omega_sq);
					//kappa.xy = kappa_parallel * (omega.x*omega.y - nu_eHeart *omega.z) / (nu_eHeart * nu_eHeart + omega_sq);
					//kappa.yx = kappa_parallel * (omega.x*omega.y + nu_eHeart * omega.z) / (nu_eHeart * nu_eHeart + omega_sq);
					//kappa.yy = kappa_parallel * (omega.y*omega.y + nu_eHeart * nu_eHeart) / (nu_eHeart * nu_eHeart + omega_sq);

					{ // scoping brace
						f64_vec3 omega = Make3(qoverMc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qoverMc);
						
						// eps involves factor Nn: T_k+1 - [ T_k + h dNnTn/dt / Nn ]
						// Needed to collect deps/d T_clock, deps/d T_opp, deps/d T_anti, deps/d T_own

						f64 coeff_NT_on_dTbydx = TWOTHIRDS * kappa_parallel *(
							edge_normal.x*(nu*nu + omega.x*omega.x) + edge_normal.y*(omega.x*omega.y - nu * omega.z)
							) / (nu * nu + omega.dot(omega));

						f64 coeff_NT_on_dTbydy = TWOTHIRDS * kappa_parallel *(
							edge_normal.x*(omega.x*omega.y + nu * omega.z) + edge_normal.y*(omega.y*omega.y + nu * nu)
							) / (nu * nu + omega.dot(omega));
#ifdef BWDSIDET
						{
							char iVolley_anti;
							short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
							indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
							if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
							{
								iVolley_anti = shared_iVolley[indexneigh - StartMajor];
							}
							else {
								iVolley_anti = p_iVolley[indexneigh];
							};
							f64 d_NT_by_dT_anti = (coeff_NT_on_dTbydx*0.5*(info.pos.y - pos_out.y)
								- coeff_NT_on_dTbydy*0.5*(info.pos.x - pos_out.x)) / Area_quadrilateral;
							d_eps_by_d_beta.x[iVolley_anti] += d_NT_by_dT_anti*regressor_anti*(-h_use / N);
						}
#endif

						{
							char iVolley_out;
							indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
							if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
							{
								iVolley_out = shared_iVolley[indexneigh - StartMajor];
							}
							else {
								iVolley_out = p_iVolley[indexneigh];
							};
							// really, d_NTrates_by_dT :
							f64 d_NT_by_dT_opp = (coeff_NT_on_dTbydx*0.5*(pos_anti.y - pos_clock.y)
								- coeff_NT_on_dTbydy*0.5*(pos_anti.x - pos_clock.x)) / Area_quadrilateral;
							d_eps_by_d_beta.x[iVolley_out] += d_NT_by_dT_opp*regressor_out*(-h_use / N);
						}

#ifdef BWDSIDET
						{
							char iVolley_clock;
							short iPrev = iNeigh - 1; if (iPrev<0) iPrev = info.neigh_len - 1;
							indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iPrev];
							if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
							{
								iVolley_clock = shared_iVolley[indexneigh - StartMajor];
							}
							else {
								iVolley_clock = p_iVolley[indexneigh];
							};
							f64 d_NT_by_dT_clock = (coeff_NT_on_dTbydx*0.5*(pos_out.y - info.pos.y)
								- coeff_NT_on_dTbydy*0.5*(pos_out.x - info.pos.x)) / Area_quadrilateral;
							d_eps_by_d_beta.x[iVolley_clock] += d_NT_by_dT_clock*regressor_clock*(-h_use / N);
						}
#endif

						f64 d_NT_by_dT_own = (coeff_NT_on_dTbydx*0.5*(pos_clock.y - pos_anti.y)
							- coeff_NT_on_dTbydy*0.5*(pos_clock.x - pos_anti.x)) / Area_quadrilateral;
						d_eps_by_d_beta.x[shared_iVolley[threadIdx.x]] += d_NT_by_dT_own*shared_x[threadIdx.x]*(-h_use / N);

					//	ourrates.NiTi += TWOTHIRDS * kappa_parallel *(
					//		edge_normal.x*((nu*nu + omega.x*omega.x)*grad_T.x + (omega.x*omega.y + nu * omega.z)*grad_T.y)
					//		+ edge_normal.y*((omega.x*omega.y - nu * omega.z)*grad_T.x + (omega.y*omega.y + nu * nu)*grad_T.y)
					//		) / (nu * nu + omega.dot(omega));

					}
				} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

				// Now go round:	
				endpt_clock = endpt_anti;
				pos_clock = pos_out;
				pos_out = pos_anti;
#ifdef BWDSIDET
				regressor_clock = regressor_out;
				regressor_out = regressor_anti;
#endif
			}; // next iNeigh

			memcpy(&(dz_d_eps_by_dbeta_i_x4[iVertex]), &(d_eps_by_d_beta), sizeof(f64_vec4));

		}; // was it DOMAIN_VERTEX? Do what otherwise?
	}; // was it OUTERMOST/INNERMOST?


	__syncthreads();

	// Did we make sure to include a call to syncthreads every time we carried on to update shared memory data in every other routine?
	// ##################################################################################################################################

	{
		memcpy(&(shared_kappa[threadIdx.x * 2]), p_kappa_e + 2 * iVertex, 2 * sizeof(f64));
		memcpy(&(shared_nu[threadIdx.x * 2]), p_nu_e + 2 * iVertex, 2 * sizeof(f64));
	}
	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {
		shared_x[threadIdx.x] = p_regressor_e[iVertex];
	}
	else {
		shared_x[threadIdx.x] = 0.0;
	}

	__syncthreads();


	if ((info.flag == INNERMOST) || (info.flag == OUTERMOST))
	{
		// [ Ignore flux into edge of outermost vertex I guess ???]
	}
	else {
		if (info.flag == DOMAIN_VERTEX) {
			
			memset(&d_eps_by_d_beta, 0, sizeof(f64_vec4));
			d_eps_by_d_beta.x[shared_iVolley[threadIdx.x]] = shared_x[threadIdx.x]; // T_self affects eps directly.

			if (iVertex == VERTCHOSEN) printf("iVertex %d d_eps_by_0 %1.9E Jac %1.9E iVolley %d \n",
				iVertex, d_eps_by_d_beta.x[0], shared_x[threadIdx.x], shared_iVolley[threadIdx.x]);

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_clock = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
				regressor_clock = shared_x[indexneigh - StartMajor];
#endif
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_clock = info2.pos;
#ifdef BWDSIDET
				regressor_clock = p_regressor_e[indexneigh];
#endif
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == NEEDS_ANTI) {
				pos_clock = Anticlock_rotate2(pos_clock);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_clock = Clockwise_rotate2(pos_clock);
			};

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_out = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
				regressor_out = shared_x[indexneigh - StartMajor];
#endif
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_out = info2.pos;
#ifdef BWDSIDET
				regressor_out = p_regressor_e[indexneigh];
#endif
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
			if (PBC == NEEDS_ANTI) {
				pos_out = Anticlock_rotate2(pos_out);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_out = Clockwise_rotate2(pos_out);
			};

			if ((izTri[info.neigh_len - 1] >= StartMinor) && (izTri[info.neigh_len - 1] < EndMinor))
			{
				endpt_clock = shared_pos[izTri[info.neigh_len - 1] - StartMinor];
			}
			else {
#ifdef CENTROID_HEATCONDUCTION
				endpt_clock = p_info_minor[izTri[info.neigh_len - 1]].pos;
#else
				endpt_clock = p_cc[izTri[info.neigh_len - 1]];
#endif
			}
			PBC = PBCtri[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == ROTATE_ME_CLOCKWISE) endpt_clock = Clockwise_d * endpt_clock;
			if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_clock = Anticlockwise_d * endpt_clock;
#ifdef BWDSIDET
			if (regressor_clock == 0.0) {
				regressor_clock = 0.5*(shared_x[threadIdx.x] + regressor_out);
			};
#endif
			f64 N = p_AreaMajor[iVertex] * p_n_major[iVertex].n;

#pragma unroll MAXNEIGH_d
			for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
			{
				{
					short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNext];
				}
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					pos_anti = shared_pos_verts[indexneigh - StartMajor];
#ifdef BWDSIDET
					regressor_anti = shared_x[indexneigh - StartMajor];
#endif
				}
				else {
					structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
					pos_anti = info2.pos;
#ifdef BWDSIDET
					regressor_anti = p_regressor_e[indexneigh];
#endif
				};
				if (PBC == NEEDS_ANTI) {
					pos_anti = Anticlock_rotate2(pos_anti);
				};
				if (PBC == NEEDS_CLOCK) {
					pos_anti = Clockwise_rotate2(pos_anti);
				};
#ifdef BWDSIDET
				if (regressor_anti == 0.0) {
					regressor_anti = 0.5*(shared_x[threadIdx.x] + regressor_out);
				}; // So we are receiving 0 then doing this. But how come?
#else
				// Compute regressor_out on every go:
				indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					regressor_out = shared_x[indexneigh - StartMajor];
				}
				else {
					regressor_out = p_regressor_e[indexneigh];
				};
#endif
				if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
				{
					endpt_anti = shared_pos[izTri[iNeigh] - StartMinor];
				}
				else {
#ifdef CENTROID_HEATCONDUCTION
					endpt_anti = p_info_minor[izTri[iNeigh]].pos;
#else
					endpt_anti = p_cc[izTri[iNeigh]];
#endif
				}
				PBC = PBCtri[MAXNEIGH_d*threadIdx.x + iNeigh];
				if (PBC == ROTATE_ME_CLOCKWISE) endpt_anti = Clockwise_d * endpt_anti;
				if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_anti = Anticlockwise_d * endpt_anti;

				// It decided to rotate something it shouldn't oughta. Rotated tri 23600 = tri 2 for 11582.

				edge_normal.x = (endpt_anti.y - endpt_clock.y);
				edge_normal.y = (endpt_clock.x - endpt_anti.x);

				// SMARTY:
				if (pos_out.x*pos_out.x + pos_out.y*pos_out.y >
					DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					f64 Area_quadrilateral = 0.5*(
						(info.pos.x + pos_anti.x)*(info.pos.y - pos_anti.y)
						+ (pos_clock.x + info.pos.x)*(pos_clock.y - info.pos.y)
						+ (pos_out.x + pos_clock.x)*(pos_out.y - pos_clock.y)
						+ (pos_anti.x + pos_out.x)*(pos_anti.y - pos_out.y)
						);

					kappa_parallel = 0.0;
					f64 nu;
					if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
					{
						kappa_parallel = 0.5*shared_kappa[izTri[iNeigh] - StartMinor];
						nu = 0.5*shared_nu[izTri[iNeigh] - StartMinor];
					}
					else {
						kappa_parallel = 0.5*p_kappa_e[izTri[iNeigh]];
						nu = 0.5*p_nu_e[izTri[iNeigh]];
					};

					{
						short iPrev = iNeigh - 1; if (iPrev < 0) iPrev = info.neigh_len - 1;
						if ((izTri[iPrev] >= StartMinor) && (izTri[iPrev] < EndMinor))
						{
							kappa_parallel += 0.5*shared_kappa[izTri[iPrev] - StartMinor];
							nu += 0.5*shared_nu[izTri[iPrev] - StartMinor];
						}
						else {
							kappa_parallel += 0.5*p_kappa_e[izTri[iPrev]];
							nu += 0.5*p_nu_e[izTri[iPrev]];
						}
					}

					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						B_out = shared_B[indexneigh - StartMajor];
					}
					else {
						f64_vec3 B_out3 = p_B_major[indexneigh];
						B_out = B_out3.xypart();
					}
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if (PBC == NEEDS_ANTI) 	B_out = Anticlock_rotate2(B_out);
					if (PBC == NEEDS_CLOCK)	B_out = Clockwise_rotate2(B_out);

					{ // scoping brace
						f64_vec3 omega = Make3(qovermc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qovermc);
						// PROBABLY ALWAYS SPILLED INTO GLOBAL -- WHAT CAN WE DO?

						f64 coeff_NT_on_dTbydx = TWOTHIRDS * kappa_parallel *(
							edge_normal.x*(nu*nu + omega.x*omega.x) + edge_normal.y*(omega.x*omega.y - nu * omega.z)
							) / (nu * nu + omega.dot(omega));

						f64 coeff_NT_on_dTbydy = TWOTHIRDS * kappa_parallel *(
							edge_normal.x*(omega.x*omega.y + nu * omega.z) + edge_normal.y*(omega.y*omega.y + nu * nu)
							) / (nu * nu + omega.dot(omega));
//
//						f64 d_NT_by_dT_clock = (coeff_NT_on_dTbydx*0.5*(pos_out.y - info.pos.y)
//							- coeff_NT_on_dTbydy*0.5*(pos_out.x - info.pos.x)) / Area_quadrilateral;
//
//						d_eps_by_d_beta += (
//							d_NT_by_dT_clock*regressor_clock
//							+ d_NT_by_dT_opp*regressor_out
//							+ d_NT_by_dT_anti*regressor_anti
//							+ d_NT_by_dT_own*shared_x[threadIdx.x])*(-h_use / N);
											
#ifdef BWDSIDET
						{
							char iVolley_anti;
							short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
							indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
							if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
							{
								iVolley_anti = shared_iVolley[indexneigh - StartMajor];
							}
							else {
								iVolley_anti = p_iVolley[indexneigh];
							};
							f64 d_NT_by_dT_anti = (coeff_NT_on_dTbydx*0.5*(info.pos.y - pos_out.y)
								- coeff_NT_on_dTbydy*0.5*(info.pos.x - pos_out.x)) / Area_quadrilateral;
							d_eps_by_d_beta.x[iVolley_anti] += d_NT_by_dT_anti*regressor_anti*(-h_use / N);

					//		if (iVertex == VERTCHOSEN) printf("iVertex %d indexneigh %d iVolley_anti %d contrib %1.9E dbyd %1.9E Jac %1.6E hoverN %1.6E \n",
					//			iVertex, indexneigh, iVolley_anti, d_NT_by_dT_anti*regressor_anti*(-h_use / N),
					//			d_NT_by_dT_anti, regressor_anti, -h_use / N);
						}
#endif

						{
							char iVolley_out;
							indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
							if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
							{
								iVolley_out = shared_iVolley[indexneigh - StartMajor];
							}
							else {
								iVolley_out = p_iVolley[indexneigh];
							};
							// really, d_NTrates_by_dT :
							f64 d_NT_by_dT_opp = (coeff_NT_on_dTbydx*0.5*(pos_anti.y - pos_clock.y)
								- coeff_NT_on_dTbydy*0.5*(pos_anti.x - pos_clock.x)) / Area_quadrilateral;
							d_eps_by_d_beta.x[iVolley_out] += d_NT_by_dT_opp*regressor_out*(-h_use / N);

					//		if (iVertex == VERTCHOSEN) printf("iVertex %d indexneigh %d iVolley_out %d contrib %1.9E dbyd %1.9E Jac %1.6E hoverN %1.6E \n",
					//			iVertex, indexneigh, iVolley_out, d_NT_by_dT_opp*regressor_out*(-h_use / N),
					//			d_NT_by_dT_opp, regressor_out, -h_use / N);
						}

#ifdef BWDSIDET
						{
							char iVolley_clock;
							short iPrev = iNeigh - 1; if (iPrev<0) iPrev = info.neigh_len - 1;
							indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iPrev];
							if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
							{
								iVolley_clock = shared_iVolley[indexneigh - StartMajor];
							}
							else {
								iVolley_clock = p_iVolley[indexneigh];
							};
							f64 d_NT_by_dT_clock = (coeff_NT_on_dTbydx*0.5*(pos_out.y - info.pos.y)
								- coeff_NT_on_dTbydy*0.5*(pos_out.x - info.pos.x)) / Area_quadrilateral;
							d_eps_by_d_beta.x[iVolley_clock] += d_NT_by_dT_clock*regressor_clock*(-h_use / N);

					//		if (iVertex == VERTCHOSEN) printf("iVertex %d indexneigh %d iVolley_clock %d contrib %1.9E dbyd %1.9E Jac %1.6E hoverN %1.6E \n",
					//			iVertex, indexneigh, iVolley_clock, d_NT_by_dT_clock*regressor_clock*(-h_use / N),
					//			d_NT_by_dT_clock, regressor_clock, -h_use / N);
						}
#endif

						f64 d_NT_by_dT_own = (coeff_NT_on_dTbydx*0.5*(pos_clock.y - pos_anti.y)
							- coeff_NT_on_dTbydy*0.5*(pos_clock.x - pos_anti.x)) / Area_quadrilateral;
						d_eps_by_d_beta.x[shared_iVolley[threadIdx.x]] += d_NT_by_dT_own*shared_x[threadIdx.x] * (-h_use / N);

				//		if (iVertex == VERTCHOSEN) printf("iVertex %d indexneigh %d iVolley_own %d contrib %1.9E dbyd %1.9E Jac %1.6E hoverN %1.6E \n",
				//			iVertex, indexneigh, shared_iVolley[threadIdx.x], d_NT_by_dT_own*shared_x[threadIdx.x] * (-h_use / N),
				//			d_NT_by_dT_own, shared_x[threadIdx.x], -h_use / N);

					}
				} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

				  // Now go round:	
				endpt_clock = endpt_anti;
				pos_clock = pos_out;
				pos_out = pos_anti;
#ifdef BWDSIDET
				regressor_clock = regressor_out;
				regressor_out = regressor_anti;
#endif

			}; // next iNeigh

			// Shan't we choose 3 separate beta -- there is a 3 vector of epsilons. Ah.

			memcpy(&(dz_d_eps_by_dbeta_e_x4[iVertex]), &(d_eps_by_d_beta), sizeof(f64_vec4));

		}; // was it DOMAIN_VERTEX? Do what otherwise?
	}; // was it OUTERMOST/INNERMOST?
	
}*/

/*
__global__ void kernelCalc_SelfCoefficient_for_HeatConduction(
	f64 const h_sub,
	structural * __restrict__ p_info_minor,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,
	long * __restrict__ izTri_verts,
	char * __restrict__ szPBCtri_verts,
	f64_vec2 * __restrict__ p_cc,

	nvals * __restrict__ p_n_major,
	T3 * __restrict__ p_T_major, // only using values from vertices, once we change to use T_k+1 in kappa
	// T3 * __restrict__ p_T_use, // the T for side temperatures -- this is T_1/2 on main step, T_k on half-step.
	// We don't even use T_k+1 values , therefore we don't (yet) need T_k+1/2 values for side T either!

	f64_vec3 * __restrict__ p_B_major,

	f64 * __restrict__ p_kappa_n,
	f64 * __restrict__ p_kappa_i,
	f64 * __restrict__ p_kappa_e,

	f64 * __restrict__ p_nu_i,
	f64 * __restrict__ p_nu_e,

	f64 * __restrict__ p_AreaMajor,
	f64 * __restrict__ p_coeffself_n,
	f64 * __restrict__ p_coeffself_i,
	f64 * __restrict__ p_coeffself_e,
	f64 const additional // usually 1.0 but 0 if we are not measuring deps/dT
	)
{
	// Think we might as well take kappa_par and nu from triangles really.
	// If n is modelled well then hopefully a nearby high-n does not have a big impact.

	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajorClever]; // 2
	__shared__ f64_vec2 shared_pos[2 * threadsPerTileMajorClever];
	
	__shared__ f64_vec2 shared_B[threadsPerTileMajorClever]; // +2

															 // B is smooth. Unfortunately we have not fitted in Bz here.
															 // In order to do that perhaps rewrite so that variables are overwritten in shared.
															 // We do not need all T and nu in shared at the same time.
															 // This way is easier for NOW.
	__shared__ f64 shared_kappa[threadsPerTileMajorClever * 2];
	__shared__ f64 shared_nu[threadsPerTileMajorClever * 2];

	__shared__ long Indexneigh[MAXNEIGH_d*threadsPerTileMajorClever]; // assume 48 bytes = 4*12 = 6 doubles

	__shared__ char PBCneigh[MAXNEIGH_d*threadsPerTileMajorClever]; // 12 bytes each from L1. Have 42 per thread at 384 threads.
																	// We should at any rate try a major block of size 256. If 1 block will run at a time, so be it.
																	// Leave L1 in case of register overflow into it. <-- don't know how likely - do we only have 31 doubles in registry
																	// regardless # of threads and space? Or can be 63?
	__shared__ char PBCtri[MAXNEIGH_d*threadsPerTileMajorClever];
	// Balance of shared vs L1: 24*256*8 = 48K. That leaves 8 doublesworth in L1 for variables.

	long izTri[MAXNEIGH_d];  // so only 2 doubles left in L1. 31 in registers??

	// Optimization: probably can move this into shared as we have got rid of T from shared.


	long const StartMajor = blockIdx.x*blockDim.x;
	long const EndMajor = StartMajor + blockDim.x;
	long const StartMinor = blockIdx.x*blockDim.x * 2;
	long const EndMinor = StartMinor + blockDim.x * 2;
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX	// 2.5 double

	structural info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];     // 3 double
	shared_pos_verts[threadIdx.x] = info.pos;
#ifdef CENTROID_HEATCONDUCTION
	{
		structural infotemp[2];
		memcpy(infotemp, p_info_minor + 2 * iVertex, 2 * sizeof(structural));
		shared_pos[threadIdx.x * 2] = infotemp[0].pos;
		shared_pos[threadIdx.x * 2 + 1] = infotemp[1].pos;
		f64 tempf64[2];
		memcpy(&(shared_kappa[threadIdx.x * 2]), p_kappa_n + 2 * iVertex, 2 * sizeof(f64));
		// No nu to set for neutrals - not used
	}
#else
	{
		memcpy(&(shared_pos[threadIdx.x * 2]), p_cc + 2 * iVertex, 2 * sizeof(f64_vec2));
		memcpy(&(shared_kappa[threadIdx.x * 2]), p_kappa_n + 2 * iVertex, 2 * sizeof(f64));		
	}
#endif


	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {
		shared_B[threadIdx.x] = p_B_major[iVertex].xypart();
	} else {
		// SHOULD NOT BE LOOKING INTO INS.
		// How do we avoid?
		memset(&(shared_B[threadIdx.x]), 0, sizeof(f64_vec2));
	}

	
	__syncthreads();

	f64_vec2 pos_clock, pos_anti, pos_out;   // +6
	f64_vec2 B_out;       // +2
	f64 kappa_parallel; // do we use them all at once or can we save 2 doubles here?
	f64 nu;                // 20 there  
	f64_vec2 edge_normal;  // 22
	f64_vec2 endpt_anti;    // 24 .. + 6 from above
	long indexneigh;     // into the 2-double buffer in L1
	f64_vec2 endpt_clock;    // As we only use endpt_anti afterwords we could union endpt_clock with edge_normal
	// Come back and optimize by checking which things we need in scope at the same time?
	f64_vec2 coeff_self_grad_T;
	short iNeigh; // only fixed # of addresses so short makes no difference.
	char PBC; // char makes no difference.
	f64 coeff_self_n = 0.0, coeff_self_i = 0.0, coeff_self_e = 0.0;
	f64 N, Nn;

	if ((info.flag == INNERMOST) || (info.flag == OUTERMOST))
	{
		// [ Ignore flux into edge of outermost vertex I guess ???]
	}
	else {
		if (info.flag == DOMAIN_VERTEX) {
			// The idea of not sending blocks full of non-domain vertices is another idea. Fiddly with indices.

			{
				f64 AreaMajor = p_AreaMajor[iVertex];
				N = AreaMajor*p_n_major[iVertex].n;
				Nn = AreaMajor*p_n_major[iVertex].n_n;
			}

			// Need this, we are adding on to existing d/dt N,NT :
			
			memcpy(Indexneigh + MAXNEIGH_d * threadIdx.x,
				pIndexNeigh + MAXNEIGH_d * iVertex,
				MAXNEIGH_d * sizeof(long));
			memcpy(izTri, //+ MAXNEIGH_d * threadIdx.x,
				izTri_verts + MAXNEIGH_d * iVertex, MAXNEIGH_d * sizeof(long));

			// may not need:
			memcpy(PBCneigh + MAXNEIGH_d * threadIdx.x,
				pPBCNeigh + MAXNEIGH_d * iVertex,
				MAXNEIGH_d * sizeof(char));
			memcpy(PBCtri + MAXNEIGH_d * threadIdx.x,
				szPBCtri_verts + MAXNEIGH_d * iVertex,
				MAXNEIGH_d * sizeof(char));
			
			// Now do Tn:
			 
			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_clock = shared_pos_verts[indexneigh - StartMajor];
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_clock = info2.pos;
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == NEEDS_ANTI) {
				pos_clock = Anticlock_rotate2(pos_clock);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_clock = Clockwise_rotate2(pos_clock);
			};

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_out = shared_pos_verts[indexneigh - StartMajor];
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_out = info2.pos;
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
			if (PBC == NEEDS_ANTI) {
				pos_out = Anticlock_rotate2(pos_out);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_out = Clockwise_rotate2(pos_out);
			};

			if ((izTri[info.neigh_len - 1] >= StartMinor) && (izTri[info.neigh_len - 1] < EndMinor))
			{
				endpt_clock = shared_pos[izTri[info.neigh_len - 1] - StartMinor];
			}
			else {
#ifdef CENTROID_HEATCONDUCTION
				endpt_clock = p_info_minor[izTri[info.neigh_len - 1]].pos;
#else
				endpt_clock = p_cc[izTri[info.neigh_len - 1]];
#endif
			}
			
			PBC = PBCtri[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == ROTATE_ME_CLOCKWISE) endpt_clock = Clockwise_d * endpt_clock;
			if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_clock = Anticlockwise_d * endpt_clock;
						
#pragma unroll MAXNEIGH_d

			// We should do all species at once?

			for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
			{
				{
					short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNext];
				}
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					pos_anti = shared_pos_verts[indexneigh - StartMajor];
				}
				else {
					structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
					pos_anti = info2.pos;
				};
				if (PBC == NEEDS_ANTI) {
					pos_anti = Anticlock_rotate2(pos_anti);
				};
				if (PBC == NEEDS_CLOCK) {
					pos_anti = Clockwise_rotate2(pos_anti);
				};

				// Now let's see
				// tri 0 has neighs 0 and 1 I'm pretty sure (check....) CHECK
				if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
				{
					endpt_anti = shared_pos[izTri[iNeigh] - StartMinor];
				}
				else {
#ifdef CENTROID_HEATCONDUCTION
					endpt_anti = p_info_minor[izTri[iNeigh]].pos;
#else
					endpt_anti = p_cc[izTri[iNeigh]];
#endif
				}
				PBC = PBCtri[MAXNEIGH_d*threadIdx.x + iNeigh];
				if (PBC == ROTATE_ME_CLOCKWISE) endpt_anti = Clockwise_d * endpt_anti;
				if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_anti = Anticlockwise_d * endpt_anti;

				edge_normal.x = (endpt_anti.y - endpt_clock.y);
				edge_normal.y = (endpt_clock.x - endpt_anti.x);

				// SMARTY:
				if (pos_out.x*pos_out.x + pos_out.y*pos_out.y >
					DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					f64 Area_quadrilateral = 0.5*(
						(info.pos.x + pos_anti.x)*(info.pos.y - pos_anti.y)
						+ (pos_clock.x + info.pos.x)*(pos_clock.y - info.pos.y)
						+ (pos_out.x + pos_clock.x)*(pos_out.y - pos_clock.y)
						+ (pos_anti.x + pos_out.x)*(pos_anti.y - pos_out.y)
						);
					
					coeff_self_grad_T.x = 0.5*(pos_clock.y - pos_anti.y) / Area_quadrilateral;
					coeff_self_grad_T.y = -0.5*(pos_clock.x - pos_anti.x) / Area_quadrilateral;

					kappa_parallel = 0.0;
					if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
					{
						kappa_parallel = 0.5*shared_kappa[izTri[iNeigh] - StartMinor];
					}
					else {
						kappa_parallel = 0.5*p_kappa_n[izTri[iNeigh]];
					}
					{
						short iPrev = iNeigh - 1; if (iPrev < 0) iPrev = info.neigh_len - 1;
						if ((izTri[iPrev] >= StartMinor) && (izTri[iPrev] < EndMinor))
						{
							kappa_parallel += 0.5*shared_kappa[izTri[iPrev] - StartMinor];
						}
						else {
							kappa_parallel += 0.5*p_kappa_n[izTri[iPrev]];
						} 
					}

					coeff_self_n += TWOTHIRDS *(-h_sub / Nn)* kappa_parallel * coeff_self_grad_T.dot(edge_normal);
					// dividing by Nn to get coeff on self

					//ourrates.NnTn += TWOTHIRDS * kappa_parallel * grad_T.dot(edge_normal);
				} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

				// Now go round:	
				endpt_clock = endpt_anti;
				pos_clock = pos_out;
				pos_out = pos_anti;
			}; // next iNeigh

		}; // was it DOMAIN_VERTEX? Do what otherwise?
	}; // was it OUTERMOST/INNERMOST?

	__syncthreads();

	// Did we make sure to include a call to syncthreads every time we carried on to update shared memory data in every other routine?
	// ##################################################################################################################################

	{
		memcpy(&(shared_kappa[threadIdx.x * 2]), p_kappa_i + 2 * iVertex, 2 * sizeof(f64));
		memcpy(&(shared_nu[threadIdx.x * 2]), p_nu_i + 2 * iVertex, 2 * sizeof(f64));
	}
	__syncthreads();

	if ((info.flag == INNERMOST) || (info.flag == OUTERMOST))
	{
		// [ Ignore flux into edge of outermost vertex I guess ???]
	}
	else {
		if (info.flag == DOMAIN_VERTEX) {

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_clock = shared_pos_verts[indexneigh - StartMajor];
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_clock = info2.pos;
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == NEEDS_ANTI) {
				pos_clock = Anticlock_rotate2(pos_clock);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_clock = Clockwise_rotate2(pos_clock);
			};

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_out = shared_pos_verts[indexneigh - StartMajor];
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_out = info2.pos;
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
			if (PBC == NEEDS_ANTI) {
				pos_out = Anticlock_rotate2(pos_out);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_out = Clockwise_rotate2(pos_out);
			};

			if ((izTri[info.neigh_len - 1] >= StartMinor) && (izTri[info.neigh_len - 1] < EndMinor))
			{
				endpt_clock = shared_pos[izTri[info.neigh_len - 1] - StartMinor];
			}
			else {
#ifdef CENTROID_HEATCONDUCTION
				endpt_clock = p_info_minor[izTri[info.neigh_len - 1]].pos;
#else
				endpt_clock = p_cc[izTri[info.neigh_len - 1]];
#endif
			}
			PBC = PBCtri[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == ROTATE_ME_CLOCKWISE) endpt_clock = Clockwise_d * endpt_clock;
			if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_clock = Anticlockwise_d * endpt_clock;


#pragma unroll MAXNEIGH_d
			for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
			{
				{
					short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNext];
				}
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					pos_anti = shared_pos_verts[indexneigh - StartMajor];
				}
				else {
					structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
					pos_anti = info2.pos;
				};
				if (PBC == NEEDS_ANTI) {
					pos_anti = Anticlock_rotate2(pos_anti);
				};
				if (PBC == NEEDS_CLOCK) {
					pos_anti = Clockwise_rotate2(pos_anti);
				};

				if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
				{
					endpt_anti = shared_pos[izTri[iNeigh] - StartMinor];
				}
				else {
#ifdef CENTROID_HEATCONDUCTION
					endpt_anti = p_info_minor[izTri[iNeigh]].pos;
#else
					endpt_anti = p_cc[izTri[iNeigh]];
#endif					
				}
				PBC = PBCtri[MAXNEIGH_d*threadIdx.x + iNeigh];
				if (PBC == ROTATE_ME_CLOCKWISE) endpt_anti = Clockwise_d * endpt_anti;
				if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_anti = Anticlockwise_d * endpt_anti;

				edge_normal.x = (endpt_anti.y - endpt_clock.y);
				edge_normal.y = (endpt_clock.x - endpt_anti.x);

				// SMARTY:
				if (pos_out.x*pos_out.x + pos_out.y*pos_out.y >
					DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					f64 Area_quadrilateral = 0.5*(
						(info.pos.x + pos_anti.x)*(info.pos.y - pos_anti.y)
						+ (pos_clock.x + info.pos.x)*(pos_clock.y - info.pos.y)
						+ (pos_out.x + pos_clock.x)*(pos_out.y - pos_clock.y)
						+ (pos_anti.x + pos_out.x)*(pos_anti.y - pos_out.y)
						);
					coeff_self_grad_T.x = 0.5*((pos_clock.y - pos_anti.y)) / Area_quadrilateral;
					coeff_self_grad_T.y = -0.5*((pos_clock.x - pos_anti.x)) / Area_quadrilateral;

					kappa_parallel = 0.0;
					f64 nu;
					if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
					{
						kappa_parallel = 0.5*shared_kappa[izTri[iNeigh] - StartMinor];
						nu = 0.5*shared_nu[izTri[iNeigh] - StartMinor];
					}
					else {
						kappa_parallel = 0.5*p_kappa_i[izTri[iNeigh]];
						nu = 0.5*p_nu_i[izTri[iNeigh]];
					};
					{
						short iPrev = iNeigh - 1; if (iPrev < 0) iPrev = info.neigh_len - 1;
						if ((izTri[iPrev] >= StartMinor) && (izTri[iPrev] < EndMinor))
						{
							kappa_parallel += 0.5*shared_kappa[izTri[iPrev] - StartMinor];
							nu += 0.5*shared_nu[izTri[iPrev] - StartMinor];
						}
						else {
							kappa_parallel += 0.5*p_kappa_i[izTri[iPrev]];
							nu += 0.5*p_nu_i[izTri[iPrev]];
						}
					}
					
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						B_out = shared_B[indexneigh - StartMajor];
					}
					else {
						f64_vec3 B_out3 = p_B_major[indexneigh];
						B_out = B_out3.xypart();
					}
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if (PBC == NEEDS_ANTI) 	B_out = Anticlock_rotate2(B_out);
					if (PBC == NEEDS_CLOCK)	B_out = Clockwise_rotate2(B_out);

					//kappa.xx = kappa_parallel * (nu_eHeart*nu_eHeart + omega.x*omega.x) / (nu_eHeart * nu_eHeart + omega_sq);
					//kappa.xy = kappa_parallel * (omega.x*omega.y - nu_eHeart *omega.z) / (nu_eHeart * nu_eHeart + omega_sq);
					//kappa.yx = kappa_parallel * (omega.x*omega.y + nu_eHeart * omega.z) / (nu_eHeart * nu_eHeart + omega_sq);
					//kappa.yy = kappa_parallel * (omega.y*omega.y + nu_eHeart * nu_eHeart) / (nu_eHeart * nu_eHeart + omega_sq);

					{ // scoping brace
						f64_vec3 omega = Make3(qoverMc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qoverMc);
						// PROBABLY ALWAYS SPILLED INTO GLOBAL -- WHAT CAN WE DO?

						f64 delta_out = sqrt((info.pos.x - pos_out.x)*(info.pos.x - pos_out.x) + (info.pos.y - pos_out.y)*(info.pos.y - pos_out.y));
						f64 edgelen = edge_normal.modulus();

						// LONGITUDINAL

						coeff_self_i += TWOTHIRDS *(-h_sub / N)* kappa_parallel *  (-1.0) *
							(nu*nu*edgelen*edgelen + omega.dotxy(edge_normal)*omega.dotxy(edge_normal))
							/ (delta_out*edgelen *(nu * nu + omega.dot(omega)));

						// divide by Ni to give coeff_self
						// coeff_self_i += (TWOTHIRDS *(-h_sub / N)* kappa_parallel *(
						//	edge_normal.x*((nu*nu + omega.x*omega.x)*coeff_self_grad_T.x + (omega.x*omega.y + nu * omega.z)*coeff_self_grad_T.y)
						//	+ edge_normal.y*((omega.x*omega.y - nu * omega.z)*coeff_self_grad_T.x + (omega.y*omega.y + nu * nu)*coeff_self_grad_T.y)
						//	) / (nu * nu + omega.dot(omega)));

//						ourrates.NiTi += TWOTHIRDS * kappa_parallel *(
//							edge_normal.x*((nu*nu + omega.x*omega.x)*grad_T.x + (omega.x*omega.y + nu * omega.z)*grad_T.y)
//							+ edge_normal.y*((omega.x*omega.y - nu * omega.z)*grad_T.x + (omega.y*omega.y + nu * nu)*grad_T.y)
//							) / (nu * nu + omega.dot(omega));
						
						if (TEST) {
							printf("iVertex %d coeff_self_i %1.10E contrib %1.10E kappa %1.10E nu %1.10E\n"
								"omega %1.10E %1.10E %1.10E coeffselfgradT %1.10E %1.10E N %1.10E \n",
								iVertex, coeff_self_i, (TWOTHIRDS * kappa_parallel *(
									edge_normal.x*((nu*nu + omega.x*omega.x)*coeff_self_grad_T.x + (omega.x*omega.y + nu * omega.z)*coeff_self_grad_T.y)
									+ edge_normal.y*((omega.x*omega.y - nu * omega.z)*coeff_self_grad_T.x + (omega.y*omega.y + nu * nu)*coeff_self_grad_T.y)
									) / (nu * nu + omega.dot(omega))) / N,
								kappa_parallel, nu, omega.x, omega.y, omega.z, coeff_self_grad_T.x, coeff_self_grad_T.y, N);
							
						}
					}
				} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

				  // Now go round:	
				endpt_clock = endpt_anti;
				pos_clock = pos_out;
				pos_out = pos_anti;
			}; // next iNeigh

		}; // was it DOMAIN_VERTEX? Do what otherwise?
	}; // was it OUTERMOST/INNERMOST?

	__syncthreads();

	// Did we make sure to include a call to syncthreads every time we carried on to update shared memory data in every other routine?
	// ##################################################################################################################################

	{
		memcpy(&(shared_kappa[threadIdx.x * 2]), p_kappa_e + 2 * iVertex, 2 * sizeof(f64));
		memcpy(&(shared_nu[threadIdx.x * 2]), p_nu_e + 2 * iVertex, 2 * sizeof(f64));
	}

	__syncthreads();
	
	if ((info.flag == INNERMOST) || (info.flag == OUTERMOST))
	{
		// [ Ignore flux into edge of outermost vertex I guess ???]
	}
	else {
		if (info.flag == DOMAIN_VERTEX) {

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_clock = shared_pos_verts[indexneigh - StartMajor];
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_clock = info2.pos;
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == NEEDS_ANTI) {
				pos_clock = Anticlock_rotate2(pos_clock);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_clock = Clockwise_rotate2(pos_clock);
			};

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_out = shared_pos_verts[indexneigh - StartMajor];
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_out = info2.pos;
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
			if (PBC == NEEDS_ANTI) {
				pos_out = Anticlock_rotate2(pos_out);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_out = Clockwise_rotate2(pos_out);
			};

			if ((izTri[info.neigh_len - 1] >= StartMinor) && (izTri[info.neigh_len - 1] < EndMinor))
			{
				endpt_clock = shared_pos[izTri[info.neigh_len - 1] - StartMinor];
			}
			else {
#ifdef CENTROID_HEATCONDUCTION
				endpt_clock = p_info_minor[izTri[info.neigh_len - 1]].pos;
#else
				endpt_clock = p_cc[izTri[info.neigh_len - 1]];
#endif
			}
			PBC = PBCtri[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == ROTATE_ME_CLOCKWISE) endpt_clock = Clockwise_d * endpt_clock;
			if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_clock = Anticlockwise_d * endpt_clock;
			
#pragma unroll MAXNEIGH_d
			for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
			{
				{
					short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNext];
				}
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					pos_anti = shared_pos_verts[indexneigh - StartMajor];
				}
				else {
					structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
					pos_anti = info2.pos;
				};
				if (PBC == NEEDS_ANTI) {
					pos_anti = Anticlock_rotate2(pos_anti);
				};
				if (PBC == NEEDS_CLOCK) {
					pos_anti = Clockwise_rotate2(pos_anti);
				};

				if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
				{
					endpt_anti = shared_pos[izTri[iNeigh] - StartMinor];
				}
				else {
#ifdef CENTROID_HEATCONDUCTION
					endpt_anti = p_info_minor[izTri[iNeigh]].pos;
#else
					endpt_anti = p_cc[izTri[iNeigh]];
#endif

				}
				PBC = PBCtri[MAXNEIGH_d*threadIdx.x + iNeigh];
				if (PBC == ROTATE_ME_CLOCKWISE) endpt_anti = Clockwise_d * endpt_anti;
				if (PBC == ROTATE_ME_ANTICLOCKWISE) endpt_anti = Anticlockwise_d * endpt_anti;

				// It decided to rotate something it shouldn't oughta. Rotated tri 23600 = tri 2 for 11582.

				edge_normal.x = (endpt_anti.y - endpt_clock.y);
				edge_normal.y = (endpt_clock.x - endpt_anti.x);
				
				// SMARTY:
				if (pos_out.x*pos_out.x + pos_out.y*pos_out.y >
					DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					f64 Area_quadrilateral = 0.5*(
						(info.pos.x + pos_anti.x)*(info.pos.y - pos_anti.y)
						+ (pos_clock.x + info.pos.x)*(pos_clock.y - info.pos.y)
						+ (pos_out.x + pos_clock.x)*(pos_out.y - pos_clock.y)
						+ (pos_anti.x + pos_out.x)*(pos_anti.y - pos_out.y)
						);
					coeff_self_grad_T.x = 0.5*((pos_clock.y - pos_anti.y)) / Area_quadrilateral;
					coeff_self_grad_T.y = -0.5*((pos_clock.x - pos_anti.x)) / Area_quadrilateral;
					
					kappa_parallel = 0.0;
					f64 nu;
					if ((izTri[iNeigh] >= StartMinor) && (izTri[iNeigh] < EndMinor))
					{
						kappa_parallel = 0.5*shared_kappa[izTri[iNeigh] - StartMinor];
						nu = 0.5*shared_nu[izTri[iNeigh] - StartMinor];
					}
					else {
						kappa_parallel = 0.5*p_kappa_e[izTri[iNeigh]];
						nu = 0.5*p_nu_e[izTri[iNeigh]];
					};

					{
						short iPrev = iNeigh - 1; if (iPrev < 0) iPrev = info.neigh_len - 1;
						if ((izTri[iPrev] >= StartMinor) && (izTri[iPrev] < EndMinor))
						{
							kappa_parallel += 0.5*shared_kappa[izTri[iPrev] - StartMinor];
							nu += 0.5*shared_nu[izTri[iPrev] - StartMinor];
						} else {
							kappa_parallel += 0.5*p_kappa_e[izTri[iPrev]];
							nu += 0.5*p_nu_e[izTri[iPrev]];
						}
					}

					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						B_out = shared_B[indexneigh - StartMajor];
					}
					else {
						f64_vec3 B_out3 = p_B_major[indexneigh];
						B_out = B_out3.xypart();
					}
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if (PBC == NEEDS_ANTI) 	B_out = Anticlock_rotate2(B_out);
					if (PBC == NEEDS_CLOCK)	B_out = Clockwise_rotate2(B_out);

					{ // scoping brace
						f64_vec3 omega = Make3(qovermc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qovermc);

						f64 delta_out = sqrt((info.pos.x - pos_out.x)*(info.pos.x - pos_out.x) + (info.pos.y - pos_out.y)*(info.pos.y - pos_out.y));
						f64 edgelen = edge_normal.modulus();

						coeff_self_e += TWOTHIRDS * (-h_sub / N)* kappa_parallel *  (-1.0) *
							(nu*nu*edgelen*edgelen + omega.dotxy(edge_normal)*omega.dotxy(edge_normal))
							/ (delta_out*edgelen *(nu * nu + omega.dot(omega)));
//
//						coeff_self_e += (TWOTHIRDS *(-h_sub / N)* kappa_parallel *(
//							edge_normal.x*((nu*nu + omega.x*omega.x)*coeff_self_grad_T.x + (omega.x*omega.y - nu * omega.z)*coeff_self_grad_T.y)
//							+ edge_normal.y*((omega.x*omega.y + nu * omega.z)*coeff_self_grad_T.x + (omega.y*omega.y + nu * nu)*coeff_self_grad_T.y)
//							) / (nu * nu + omega.dot(omega)));
						if (TEST) {
							printf("iVertex %d coeff_self_e %1.10E contrib %1.10E kappa %1.10E nu %1.10E\n"
								"omega %1.10E %1.10E %1.10E coeffselfgradT %1.10E %1.10E N %1.10E \n",
								iVertex, coeff_self_e, (TWOTHIRDS * kappa_parallel *(
									edge_normal.x*((nu*nu + omega.x*omega.x)*coeff_self_grad_T.x + (omega.x*omega.y + nu * omega.z)*coeff_self_grad_T.y)
									+ edge_normal.y*((omega.x*omega.y - nu * omega.z)*coeff_self_grad_T.x + (omega.y*omega.y + nu * nu)*coeff_self_grad_T.y)
									) / (nu * nu + omega.dot(omega))) / N,
								kappa_parallel, nu, omega.x, omega.y, omega.z, coeff_self_grad_T.x, coeff_self_grad_T.y, N);

						}
					}
				} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

				  // Now go round:	
				endpt_clock = endpt_anti;
				pos_clock = pos_out;
				pos_out = pos_anti;

			}; // next iNeigh

			//memcpy(NTadditionrates + iVertex, &ourrates, sizeof(NTrates));

		}; // was it DOMAIN_VERTEX? Do what otherwise?
	}; // was it OUTERMOST/INNERMOST?	

	p_coeffself_n[iVertex] = coeff_self_n + additional;
	p_coeffself_i[iVertex] = coeff_self_i + additional;
	p_coeffself_e[iVertex] = coeff_self_e + additional; // eps = T_k+1 - T_k - h/N d/dt NT

	//p_coeff_self[iVertex] = max(fabs(coeff_self_e), max(fabs(coeff_self_i), fabs(coeff_self_n)));
	//if ((blockIdx.x == 0) && (p_coeff_self[iVertex] != 0.0)) printf("iVertex %d coeffself %1.9E %1.9E %1.9E \n", iVertex, coeff_self_n, coeff_self_i, coeff_self_e);
	// was not setting 0 because we had it inside the branch!

}*/


__global__ void kernelIonisationRates(
	f64 const h_use,
	structural * __restrict__ p_info_minor,
	T3 * __restrict__ p_T_major,
	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p_AreaMajor,
	NTrates * __restrict__ NTadditionrates
	)
{

	long const iVertex = blockIdx.x*blockDim.x + threadIdx.x;
	structural info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
	NTrates ourrates;

	if (info.flag == DOMAIN_VERTEX)
	{
		// case DOMAIN_VERTEX:
		
		f64 AreaMajor = p_AreaMajor[iVertex];
		T3 T = p_T_major[iVertex];
		nvals our_n = p_n_major[iVertex];
		memcpy(&ourrates, NTadditionrates + iVertex, sizeof(NTrates));

		// So this will be different.
		// now add IONISATION:
	//	f64 TeV = T.Te * one_over_kB; 
		// We loaded in ourrates.NT which indicates the new heat available so we should include some of that.
		// The main impact will be from heat conduction; dN/dt due to advection neglected here.
		f64 TeV = one_over_kB * (T.Te*our_n.n*AreaMajor + h_use*ourrates.NeTe)/
			(our_n.n*AreaMajor + h_use*ourrates.N);
		// Should be very careful here: ourrates.NeTe can soak to neutrals on timescale what? 1e-11?

		if (TeV < 0.0) {
			printf("\n\niVertex %d : ourrates.N %1.14E denominator %1.14E \n"
				" AreaMajor %1.14E TeV %1.14E ourrates.NeTe %1.10E h %1.10E \n"
				"ourrates.Nn %1.10E n %1.10E n_n %1.10E Te %1.10E Tn %1.10E \n\n",
				iVertex, ourrates.N, 
				(our_n.n*AreaMajor + h_use*ourrates.N),
				AreaMajor, TeV, ourrates.NeTe, h_use,
				ourrates.Nn, our_n.n, our_n.n_n, T.Te, T.Tn);
			
		}
		f64 sqrtT = sqrt(TeV);
		f64 temp = 1.0e-5*exp(-13.6 / TeV) / (13.6*(6.0*13.6 + TeV)); // = S / T^1/2
			// Let h n n_n S be the ionising amount,
			// h n S is the proportion of neutrals! Make sure we do not run out!

		//f64 hnS = (h_use*our_n.n*TeV*temp) / (sqrtT + h_use * our_n.n_n*temp*SIXTH*13.6);

			// d/dt (sqrtT) = 1/2 dT/dt T^-1/2. 
			// dT[eV]/dt = -TWOTHIRDS * 13.6* n_n* sqrtT *temp
			// d/dt (sqrtT) = -THIRD*13.6*n_n*temp;

		// kind of midpoint, see SIXTH not THIRD:
		f64 Model_of_T_to_half = TeV / (sqrtT + h_use*SIXTH*13.6*our_n.n_n*temp / (1.0 - h_use*(our_n.n_n - our_n.n)*temp*sqrtT));

		f64 hS = h_use*temp*Model_of_T_to_half;
				
		// NEW:
		f64 ionise_rate = AreaMajor * our_n.n_n * our_n.n*hS / 
							(h_use*(1.0 + hS*(our_n.n-our_n.n_n)));   // dN/dt

		ourrates.N += ionise_rate;
		ourrates.Nn += -ionise_rate;


		// Let nR be the recombining amount, R is the proportion.
		TeV = T.Te * one_over_kB;
		f64 Ttothe5point5 = sqrtT * TeV * TeV*TeV * TeV*TeV;
		f64 hR = h_use * (our_n.n * our_n.n*8.75e-27*TeV) /
			(Ttothe5point5 + h_use * 2.25*TWOTHIRDS*13.6*our_n.n*our_n.n*8.75e-27);

		// T/T^5.5 = T^-4.5
		// T/(T^5.5+eps) < T^-4.5

		// For some reason I picked 2.25 = 4.5/2 instead of 5.5/2.
		// But basically it looks reasonable.

		// Maybe the additional stuff is an estimate of the change in T[eV]^5.5??
		// d/dt T^5.5 = 5.5 T^4.5 dT/dt 
		// dT/dt = TWOTHIRDS * 13.6*( hR / h_use) = TWOTHIRDS * 13.6*( n^2 8.75e-27 T^-4.5) 
		// d/dt T^5.5 = 5.5 TWOTHIRDS * 13.6*( n^2 8.75e-27 )  

		f64 recomb_rate = AreaMajor * our_n.n * hR / h_use; // could reasonably again take hR/(1+hR) for n_k+1
		ourrates.N -= recomb_rate;
		ourrates.Nn += recomb_rate;

		if (TEST) printf("%d recomb rate %1.10E our_n.n %1.10E hR %1.10E h_use %1.8E Ttothe5point5 %1.9E Te %1.9E\n", iVertex,
			recomb_rate, our_n.n, hR, h_use, Ttothe5point5, T.Te);

		ourrates.NeTe += -TWOTHIRDS * 13.6*kB*(ionise_rate - recomb_rate) + 0.5*T.Tn*ionise_rate;
		ourrates.NiTi += 0.5*T.Tn*ionise_rate;
		ourrates.NnTn += (T.Te + T.Ti)*recomb_rate;
		if (TEST) {
			printf("kernelIonisation %d NeTe %1.12E NiTi %1.12E NnTn %1.12E\n"
				"due to I+R : NeTe %1.12E NiTi %1.12E NnTn %1.12E\n"
				"d/dtNeTe/N %1.9E d/dtNiTi/N %1.9E d/dtNnTn/Nn %1.9E \n\n",
				iVertex, ourrates.NeTe, ourrates.NiTi, ourrates.NnTn,
				-TWOTHIRDS * 13.6*kB*(ionise_rate - recomb_rate) + 0.5*T.Tn*ionise_rate,
				0.5*T.Tn*ionise_rate,
				(T.Te + T.Ti)*recomb_rate,
				ourrates.NeTe / (our_n.n*AreaMajor), ourrates.NiTi / (our_n.n*AreaMajor), ourrates.NnTn / (our_n.n_n*AreaMajor));
		};
		memcpy(NTadditionrates + iVertex, &ourrates, sizeof(NTrates));
	};
}

__global__ void kernelAdvanceDensityAndTemperature(
	f64 h_use,
	structural * __restrict__ p_info_major,
	nvals * p_n_major,
	T3 * p_T_major,
	NTrates * __restrict__ NTadditionrates,

	// Think we see the mistake here: are these to be major or minor values?
	// Major, right? Check code:

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
)
{
	// runs for major tile
	// nu would have been a better choice to go in shared as it coexists with the 18 doubles in "LHS","inverted".
	// Important to set 48K L1 for this routine.

	__shared__ nvals n_src_or_use[threadsPerTileMajor];
	__shared__ f64 AreaMajor[threadsPerTileMajor];

	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x; // iVertex OF VERTEX
	structural info = p_info_major[iVertex];
	//	if (iVertex == CHOSEN) printf("GPU iVertex %d info.flag %d \n", CHOSEN, info.flag);

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {

		n_src_or_use[threadIdx.x] = p_n_major[iVertex];  // used throughout so a good candidate to stick in shared mem
		AreaMajor[threadIdx.x] = p_AreaMajor[iVertex]; // ditto

		NTrates newdata;
		{
			NTrates AdditionNT = NTadditionrates[iVertex];
			newdata.N = n_src_or_use[threadIdx.x].n*AreaMajor[threadIdx.x] + h_use * AdditionNT.N;
			newdata.Nn = n_src_or_use[threadIdx.x].n_n*AreaMajor[threadIdx.x] + h_use * AdditionNT.Nn;
			newdata.NnTn = h_use * AdditionNT.NnTn; // start off without knowing 'factor' so we can ditch AdditionNT
			newdata.NiTi = h_use * AdditionNT.NiTi;
			newdata.NeTe = h_use * AdditionNT.NeTe;

			if (TEST)
				printf("Advance_nT  %d : nsrc %1.12E nn %1.12E *AreaMajor %1.12E %1.12E\n"
					"newdata.Nn %1.12E newdata.Ni %1.12E AreaMajor %1.10E \n"
					"h*additionNiTi %1.12E for e %1.12E for n %1.12E \n"
					"AdditionNT.e %1.10E h_use %1.10E\n"
					, VERTCHOSEN,
					n_src_or_use[threadIdx.x].n, n_src_or_use[threadIdx.x].n_n,
					n_src_or_use[threadIdx.x].n*AreaMajor[threadIdx.x],
					n_src_or_use[threadIdx.x].n_n*AreaMajor[threadIdx.x],
					newdata.Nn, newdata.N, AreaMajor[threadIdx.x],
					newdata.NiTi, newdata.NeTe, newdata.NnTn,
					AdditionNT.NeTe, h_use);
		}

		// So at this vertex, near the insulator, NiTi that comes in is NaN. Is that advection or diffusion?
		// Have to go to bed tonight...

		{
			nvals n_dest;
			f64 Div_v_overall_integrated = p_Integrated_div_v_overall[iVertex];
			n_dest.n = newdata.N / (AreaMajor[threadIdx.x] + h_use*Div_v_overall_integrated); // Do have to worry whether advection steps are too frequent.
			n_dest.n_n = newdata.Nn / (AreaMajor[threadIdx.x] + h_use*Div_v_overall_integrated); // What could do differently: know ROC area as well as mass flux through walls
			p_n_major_dest[iVertex] = n_dest;

			//		if (iVertex == CHOSEN) printf("GPU %d n_dest.n_n %1.14E  Area_used %1.14E \n\n", iVertex, n_dest.n_n,
			//			(AreaMajor[threadIdx.x] + h_use*Div_v_overall_integrated));
		}

		// roughly right ; maybe there are improvements.

		// --------------------------------------------------------------------------------------------
		// Simple way of doing area ratio for exponential growth of T: 
		// (1/(1+h div v)) -- v outward grows the area so must be + here. 

		// Compressive heating:
		// USE 1 iteration of Halley's method for cube root:
		// cu_root Q =~~= x0(x0^3+2Q)/(2x0^3+Q) .. for us x0 = 1, Q is (1+eps)^-2
		// Thus (1+2(1+eps)^-2)/(2+(1+eps)^-2)
		// Multiply through by (1+eps)^2:
		// ((1+eps)^2+2)/(1+2*(1+eps)^2) .. well of course it is
		// eps = h div v

		// Way to get reasonable answer without re-doing equations:
		// Take power -1/3 and multiply once before interspecies and once after.

		f64 factor, factor_neut; // used again at end
		{
			f64 Div_v = p_div_v[iVertex];
			f64 Div_v_n = p_div_v_neut[iVertex];
			factor = (3.0 + h_use * Div_v) /
				(3.0 + 2.0* h_use * Div_v);
			factor_neut = (3.0 + h_use * Div_v_n) /
				(3.0 + 2.0*h_use * Div_v_n);
		}
		// gives (1+ h div v)^(-1/3), roughly

		// Alternate version: 
		// factor = pow(pVertex->AreaCell / pVertDest->AreaCell, 2.0 / 3.0);
		// pVertDest->Ion.heat = pVertex->Ion.heat*factor;
		// but the actual law is with 5/3 
		// Comp htg dT/dt = -2/3 T div v_fluid 
		// factor (1/(1+h div v))^(2/3) --> that's same
		{
			T3 T_src = p_T_major[iVertex];
			newdata.NnTn += n_src_or_use[threadIdx.x].n_n*AreaMajor[threadIdx.x] * T_src.Tn*factor_neut;
			newdata.NiTi += n_src_or_use[threadIdx.x].n*AreaMajor[threadIdx.x] * T_src.Ti*factor;
			newdata.NeTe += n_src_or_use[threadIdx.x].n*AreaMajor[threadIdx.x] * T_src.Te*factor;
			//
			if (TEST) {
				printf("\nAdvance_nT %d : n %1.12E Area %1.12E compressfac %1.10E \n"
					"newdate.NiTi %1.12E Ti_k %1.12E newdata.NeTe %1.10E Te_k %1.10E\n",
					VERTCHOSEN, n_src_or_use[threadIdx.x].n, AreaMajor[threadIdx.x], factor,
					newdata.NiTi, T_src.Ti, newdata.NeTe, T_src.Te);
			}
		}

		f64 nu_ne_MT, nu_en_MT, nu_ni_MT, nu_in_MT, nu_ei; // optimize after
		{
			// Dimensioning inside a brace allows the following vars to go out of scope at the end of the brace.
			f64 sqrt_Te, ionneut_thermal, electron_thermal, lnLambda, s_in_MT, s_en_MT, s_en_visc;

			n_src_or_use[threadIdx.x] = p_n_use[iVertex];
			T3 T_use = p_T_use[iVertex];

			sqrt_Te = sqrt(T_use.Te); // should be "usedata"
			ionneut_thermal = sqrt(T_use.Ti / m_ion + T_use.Tn / m_n); // hopefully not sqrt(0)
			electron_thermal = sqrt_Te * over_sqrt_m_e;
			lnLambda = Get_lnLambda_d(n_src_or_use[threadIdx.x].n, T_use.Te);

			{
				f64 s_in_visc_dummy;
				Estimate_Ion_Neutral_Cross_sections_d(T_use.Ti*one_over_kB,
					&s_in_MT, &s_in_visc_dummy);
			}
			Estimate_Ion_Neutral_Cross_sections_d(T_use.Te*one_over_kB, // call with T in electronVolts
				&s_en_MT,
				&s_en_visc);
			//s_en_MT = Estimate_Ion_Neutral_MT_Cross_section(T_use.Te*one_over_kB);
			//s_en_visc = Estimate_Ion_Neutral_Viscosity_Cross_section(T_use.Te*one_over_kB);

			if (n_src_or_use[threadIdx.x].n_n > ARTIFICIAL_RELATIVE_THRESH_HEAT *n_src_or_use[threadIdx.x].n) {
				s_en_MT *= n_src_or_use[threadIdx.x].n_n / (ARTIFICIAL_RELATIVE_THRESH_HEAT *n_src_or_use[threadIdx.x].n);
				s_in_MT *= n_src_or_use[threadIdx.x].n_n / (ARTIFICIAL_RELATIVE_THRESH_HEAT *n_src_or_use[threadIdx.x].n);
				// So at 1e18 vs 1e8 it's 10 times stronger
				// At 1e18 vs 1e6 it's 1000 times stronger
				// nu starts at about 1e11 at the place it failed at 35ns. So 10000 times stronger gives us 1e15.
			}

			// Need nu_ne etc to be defined:

			nu_ne_MT = s_en_MT * n_src_or_use[threadIdx.x].n * electron_thermal; // have to multiply by n_e for nu_ne_MT
			nu_ni_MT = s_in_MT * n_src_or_use[threadIdx.x].n * ionneut_thermal;
			nu_en_MT = s_en_MT * n_src_or_use[threadIdx.x].n_n*electron_thermal;
			nu_in_MT = s_in_MT * n_src_or_use[threadIdx.x].n_n*ionneut_thermal;

			//	
			//	if (iVertex == CHOSEN) {
			//		printf("nu_en_MT components GPU : %1.8E %1.8E %1.8E \n",
			//			s_en_MT, n_src_or_use[threadIdx.x].n_n, electron_thermal);
			//		f64 T = T_use.Te*one_over_kB;
			//		int j;
			//		printf("T = %1.10E\n", T);
			//		for (j = 0; j < 10; j++)
			//			printf("%d : cross_T_vals_d %1.10E cross_s_vals_MT %1.10E \n",
			//				j, cross_T_vals_d[j], cross_s_vals_MT_ni_d[j]);
			//		int i = 1;
			//		if (T > cross_T_vals_d[5]) {
			//			if (T > cross_T_vals_d[7]) {
			//				if (T > cross_T_vals_d[8])
			//				{
			//					i = 9; // top of interval
			//				}
			//				else {
			//					i = 8;
			//				};
			//			}
			//			else {
			//				if (T > cross_T_vals_d[6]) {
			//					i = 7;
			//				}
			//				else {
			//					i = 6;
			//				};
			//			};
			//		}
			//		else {
			//			if (T > cross_T_vals_d[3]) {
			//				if (T > cross_T_vals_d[4]) {
			//					i = 5;
			//				}
			//				else {
			//					i = 4;
			//				};
			//			}
			//			else {
			//				if (T > cross_T_vals_d[2]) {
			//					i = 3;
			//				}
			//				else {
			//					if (T > cross_T_vals_d[1]) {
			//						i = 2;
			//					}
			//					else {
			//						i = 1;
			//					};
			//				};
			//			};
			//		};
			//		// T lies between i-1,i
			//		printf("i = %d\n\n", i);
			//	}

			nu_ei = nu_eiBarconst * kB_to_3halves*n_src_or_use[threadIdx.x].n*lnLambda /
				(T_use.Te*sqrt_Te);

			//		nu_ie = nu_ei;

			//		nu_eHeart = 1.87*nu_eiBar + data_k.n_n*s_en_visc*electron_thermal;
		}


		// For now doing velocity-independent resistive heating.
		// Because although we have a magnetic correction Upsilon_zz involved, we ignored it
		// since we are also squashing the effect of velocity-dependent collisions on vx and vy (which
		// would produce a current in the plane) and this squashing should create heat, which
		// maybe means it adds up to the velocity-independent amount of heating. 
		{
			f64_vec3 v_n = p_v_n_use[iVertex];
			v4 vie = p_vie_use[iVertex];

			newdata.NeTe += h_use*(AreaMajor[threadIdx.x] * TWOTHIRDS*nu_en_MT*m_en*(
				(v_n.x - vie.vxy.x)*(v_n.x - vie.vxy.x)
				+ (v_n.y - vie.vxy.y)*(v_n.y - vie.vxy.y)
				+ (v_n.z - vie.vez)*(v_n.z - vie.vez))

				+ AreaMajor[threadIdx.x] * TWOTHIRDS*nu_ei*m_ei*(vie.vez - vie.viz)*(vie.vez - vie.viz));

			newdata.NiTi += h_use*(AreaMajor[threadIdx.x] * TWOTHIRDS*nu_in_MT*M_in*m_n*(
				(v_n.x - vie.vxy.x)*(v_n.x - vie.vxy.x)
				+ (v_n.y - vie.vxy.y)*(v_n.y - vie.vxy.y)
				+ (v_n.z - vie.viz)*(v_n.z - vie.viz)));

			newdata.NnTn += h_use*(AreaMajor[threadIdx.x] * TWOTHIRDS*nu_ni_MT*M_in*m_i*(
				(v_n.x - vie.vxy.x)*(v_n.x - vie.vxy.x)
				+ (v_n.y - vie.vxy.y)*(v_n.y - vie.vxy.y)
				+ (v_n.z - vie.viz)*(v_n.z - vie.viz)));

			if (TEST)
				printf("%d v_n.z %1.9E vie_use.viz %1.9E vie_use.vez %1.9E \n areamajor %1.8E\n"
					"nu_in %1.10E nu_en %1.8E \n"
					"Frictional htg (NT+=): n i e %1.10E %1.10E %1.10E\n",
					VERTCHOSEN, v_n.z, vie.viz, vie.vez, AreaMajor[threadIdx.x],
					nu_in_MT, nu_en_MT,
					h_use*(AreaMajor[threadIdx.x] * TWOTHIRDS*nu_ni_MT*M_in*m_i*(
					(v_n.x - vie.vxy.x)*(v_n.x - vie.vxy.x)
						+ (v_n.y - vie.vxy.y)*(v_n.y - vie.vxy.y)
						+ (v_n.z - vie.viz)*(v_n.z - vie.viz))),
					h_use*(AreaMajor[threadIdx.x] * TWOTHIRDS*nu_in_MT*M_in*m_n*(
					(v_n.x - vie.vxy.x)*(v_n.x - vie.vxy.x)
						+ (v_n.y - vie.vxy.y)*(v_n.y - vie.vxy.y)
						+ (v_n.z - vie.viz)*(v_n.z - vie.viz))),
					h_use*(AreaMajor[threadIdx.x] * TWOTHIRDS*nu_en_MT*m_en*(
					(v_n.x - vie.vxy.x)*(v_n.x - vie.vxy.x)
						+ (v_n.y - vie.vxy.y)*(v_n.y - vie.vxy.y)
						+ (v_n.z - vie.vez)*(v_n.z - vie.vez))

						+ AreaMajor[threadIdx.x] * TWOTHIRDS*nu_ei*m_ei*(vie.vez - vie.viz)*(vie.vez - vie.viz))
				);
		}
		f64_tens3 inverted;
		{
			f64_tens3 LHS;
			// x = neutral
			// y = ion
			// z = elec
			// This is for NT
			f64 nu_ie = nu_ei;

			LHS.xx = 1.0 - h_use * (-M_en * nu_ne_MT - M_in * nu_ni_MT);
			LHS.xy = -h_use * (M_in * nu_in_MT);
			LHS.xz = -h_use *(M_en * nu_en_MT);

			LHS.yx = -h_use *  M_in * nu_ni_MT;
			LHS.yy = 1.0 - h_use * (-M_in * nu_in_MT - M_ei * nu_ie);
			LHS.yz = -h_use * M_ei * nu_ei;

			LHS.zx = -h_use * M_en * nu_ne_MT;
			LHS.zy = -h_use * M_ei * nu_ie;
			LHS.zz = 1.0 - h_use * (-M_en * nu_en_MT - M_ei * nu_ei);

			// some indices appear reversed because NT not T.

			if (TEST) {
				printf("LHS | \n %1.14E %1.14E %1.14E |\n %1.14E %1.14E %1.14E |  \n %1.14E %1.14E %1.14E | \n",
					LHS.xx, LHS.xy, LHS.xz,
					LHS.yx, LHS.yy, LHS.yz,
					LHS.zx, LHS.zy, LHS.zz);
				printf("GPU %d : NnTn %1.14E NeTe %1.14E \n", CHOSEN, newdata.NnTn, newdata.NeTe);
				printf("GPU nu_en_MT %1.14E\n", nu_en_MT);
			}
			LHS.Inverse(inverted);
		}

		f64_vec3 RHS;
		f64 nu_ie = nu_ei;
		RHS.x = newdata.NnTn - h_use * (nu_ni_MT*M_in + nu_ne_MT * M_en)*newdata.NnTn
			+ h_use * nu_in_MT*M_in*newdata.NiTi + h_use * nu_en_MT*M_en*newdata.NeTe;

		RHS.y = newdata.NiTi - h_use * (nu_in_MT*M_in + nu_ie * M_ei)*newdata.NiTi
			+ h_use * nu_ni_MT*M_in*newdata.NnTn + h_use * nu_ei*M_ei*newdata.NeTe;

		RHS.z = newdata.NeTe - h_use * (nu_en_MT*M_en + nu_ei * M_ei)*newdata.NeTe
			+ h_use * nu_ie*M_ei*newdata.NiTi + h_use * nu_ne_MT*M_en*newdata.NnTn;

		f64_vec3 NT;
		NT = inverted * RHS;
		newdata.NnTn = NT.x;
		newdata.NiTi = NT.y;
		newdata.NeTe = NT.z;

		T3 T_dest;
		T_dest.Tn = newdata.NnTn* factor_neut / newdata.Nn;
		T_dest.Ti = newdata.NiTi* factor / newdata.N;
		T_dest.Te = newdata.NeTe* factor / newdata.N;

		if (TEST) {
			printf("\ninverted | RHS \n %1.14E %1.14E %1.14E | %1.14E \n %1.14E %1.14E %1.14E | %1.14E \n %1.14E %1.14E %1.14E | %1.14E \n",
				inverted.xx, inverted.xy, inverted.xz, RHS.x,
				inverted.yx, inverted.yy, inverted.yz, RHS.y,
				inverted.zx, inverted.zy, inverted.zz, RHS.z);
			printf("GPU %d : NnTn %1.14E NiTi %1.14E NeTe %1.14E \n"
				"Tn Ti Te %1.14E %1.14E %1.14E\n", VERTCHOSEN, newdata.NnTn, newdata.NiTi, newdata.NeTe,
				T_dest.Tn, T_dest.Ti, T_dest.Te);
		} // This came out with a value.

		if (TEST) printf("%d : T_dest %1.8E %1.8E %1.8E \n"
			"newdata .NeTe %1.10E .N %1.10E factor %1.10E\n\n",
			iVertex, T_dest.Tn, T_dest.Ti, T_dest.Te,
			newdata.NeTe, newdata.N, factor
		);

		if (T_dest.Te != T_dest.Te) {
			printf("Advance_n_T %d : Te NaN factor %1.8E newdata.N %1.10E flag %d \n"
				"n %1.10E Area %1.10E hd/dtNT %1.10E\n",
				iVertex, factor, newdata.N, info.flag,
				n_src_or_use[threadIdx.x].n, AreaMajor[threadIdx.x], h_use * NTadditionrates[iVertex].N);
		}

		p_T_major_dest[iVertex] = T_dest;

	}
	else {
		// nothing to do ??
		if (info.flag == OUTERMOST) {
			p_n_major_dest[iVertex] = p_n_major[iVertex];
			p_T_major_dest[iVertex] = p_T_major[iVertex];
		}
		else {
			memset(p_n_major_dest + iVertex, 0, sizeof(nvals));
			memset(p_T_major_dest + iVertex, 0, sizeof(T3));
		};
	};
}

__global__ void kernelAdvanceDensityAndTemperature_nosoak_etc(
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
)
{
	// runs for major tile
	// nu would have been a better choice to go in shared as it coexists with the 18 doubles in "LHS","inverted".
	// Important to set 48K L1 for this routine.

	__shared__ nvals n_src_or_use[threadsPerTileMajor];
	__shared__ f64 AreaMajor[threadsPerTileMajor];

	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x; // iVertex OF VERTEX
	structural info = p_info_major[iVertex];
	//	if (iVertex == CHOSEN) printf("GPU iVertex %d info.flag %d \n", CHOSEN, info.flag);

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {

		n_src_or_use[threadIdx.x] = p_n_major[iVertex];  // used throughout so a good candidate to stick in shared mem
		AreaMajor[threadIdx.x] = p_AreaMajor[iVertex]; // ditto

		NTrates newdata;
		{
			NTrates AdditionNT = NTadditionrates[iVertex];
			newdata.N = n_src_or_use[threadIdx.x].n*AreaMajor[threadIdx.x] + h_use * AdditionNT.N;
			newdata.Nn = n_src_or_use[threadIdx.x].n_n*AreaMajor[threadIdx.x] + h_use * AdditionNT.Nn;
			newdata.NnTn = h_use * AdditionNT.NnTn; // start off without knowing 'factor' so we can ditch AdditionNT
			newdata.NiTi = h_use * AdditionNT.NiTi;
			newdata.NeTe = h_use * AdditionNT.NeTe;

			if (TEST)
				printf("Advance_nT  %d : nsrc %1.12E nn %1.12E *AreaMajor %1.12E %1.12E\n"
					"newdata.Nn %1.12E newdata.Ni %1.12E AreaMajor %1.10E \n"
					"h*additionNiTi %1.12E for e %1.12E for n %1.12E \n"
					"AdditionNT.e %1.10E h_use %1.10E\n"
					, VERTCHOSEN,
					n_src_or_use[threadIdx.x].n, n_src_or_use[threadIdx.x].n_n,
					n_src_or_use[threadIdx.x].n*AreaMajor[threadIdx.x],
					n_src_or_use[threadIdx.x].n_n*AreaMajor[threadIdx.x],
					newdata.Nn, newdata.N, AreaMajor[threadIdx.x],
					newdata.NiTi, newdata.NeTe, newdata.NnTn,
					AdditionNT.NeTe, h_use);
		}

		// So at this vertex, near the insulator, NiTi that comes in is NaN. Is that advection or diffusion?
		// Have to go to bed tonight...

		{
			nvals n_dest;
			f64 Div_v_overall_integrated = p_Integrated_div_v_overall[iVertex];
			n_dest.n = newdata.N / (AreaMajor[threadIdx.x] + h_use*Div_v_overall_integrated); // Do have to worry whether advection steps are too frequent.
			n_dest.n_n = newdata.Nn / (AreaMajor[threadIdx.x] + h_use*Div_v_overall_integrated); // What could do differently: know ROC area as well as mass flux through walls
			p_n_major_dest[iVertex] = n_dest;

			//		if (iVertex == CHOSEN) printf("GPU %d n_dest.n_n %1.14E  Area_used %1.14E \n\n", iVertex, n_dest.n_n,
			//			(AreaMajor[threadIdx.x] + h_use*Div_v_overall_integrated));
		}

		// roughly right ; maybe there are improvements.

		// --------------------------------------------------------------------------------------------
		// Simple way of doing area ratio for exponential growth of T: 
		// (1/(1+h div v)) -- v outward grows the area so must be + here. 

		// Compressive heating:
		// USE 1 iteration of Halley's method for cube root:
		// cu_root Q =~~= x0(x0^3+2Q)/(2x0^3+Q) .. for us x0 = 1, Q is (1+eps)^-2
		// Thus (1+2(1+eps)^-2)/(2+(1+eps)^-2)
		// Multiply through by (1+eps)^2:
		// ((1+eps)^2+2)/(1+2*(1+eps)^2) .. well of course it is
		// eps = h div v

		// Way to get reasonable answer without re-doing equations:
		// Take power -1/3 and multiply once before interspecies and once after.

		f64 factor, factor_neut; // used again at end
		{
			f64 Div_v = p_div_v[iVertex];
			f64 Div_v_n = p_div_v_neut[iVertex];
			factor = (3.0 + h_use * Div_v) /
				(3.0 + 2.0* h_use * Div_v);
			factor_neut = (3.0 + h_use * Div_v_n) /
				(3.0 + 2.0*h_use * Div_v_n);
		}
		// gives (1+ h div v)^(-1/3), roughly

		// Alternate version: 
		// factor = pow(pVertex->AreaCell / pVertDest->AreaCell, 2.0 / 3.0);
		// pVertDest->Ion.heat = pVertex->Ion.heat*factor;
		// but the actual law is with 5/3 
		// Comp htg dT/dt = -2/3 T div v_fluid 
		// factor (1/(1+h div v))^(2/3) --> that's same
		{
			T3 T_src = p_T_major[iVertex];
			newdata.NnTn += n_src_or_use[threadIdx.x].n_n*AreaMajor[threadIdx.x] * T_src.Tn*factor_neut;
			newdata.NiTi += n_src_or_use[threadIdx.x].n*AreaMajor[threadIdx.x] * T_src.Ti*factor;
			newdata.NeTe += n_src_or_use[threadIdx.x].n*AreaMajor[threadIdx.x] * T_src.Te*factor;
			//
			if (TEST) {
				printf("\nAdvance_nT %d : n %1.12E Area %1.12E compressfac %1.10E \n"
					"newdate.NiTi %1.12E Ti_k %1.12E newdata.NeTe %1.10E Te_k %1.10E\n",
					VERTCHOSEN, n_src_or_use[threadIdx.x].n, AreaMajor[threadIdx.x], factor,
					newdata.NiTi, T_src.Ti, newdata.NeTe, T_src.Te);
			}
		}


		T3 T_dest;
		T_dest.Tn = newdata.NnTn* factor_neut / newdata.Nn;
		T_dest.Ti = newdata.NiTi* factor / newdata.N;
		T_dest.Te = newdata.NeTe* factor / newdata.N;

		if (T_dest.Te != T_dest.Te) {
			printf("Advance_n_T %d : Te NaN factor %1.8E newdata.N %1.10E flag %d \n"
				"n %1.10E Area %1.10E hd/dtNT %1.10E\n",
				iVertex, factor, newdata.N, info.flag,
				n_src_or_use[threadIdx.x].n, AreaMajor[threadIdx.x], h_use * NTadditionrates[iVertex].N);
		}

		p_T_major_dest[iVertex] = T_dest;
	}
	else {
		// nothing to do ??
		if (info.flag == OUTERMOST) {
			p_n_major_dest[iVertex] = p_n_major[iVertex];
			p_T_major_dest[iVertex] = p_T_major[iVertex];
		}
		else {
			memset(p_n_major_dest + iVertex, 0, sizeof(nvals));
			memset(p_T_major_dest + iVertex, 0, sizeof(T3));
		};
	};
}


__global__ void kernelAdvanceDensityAndTemperature_noadvectioncompression(
	f64 h_use,
	structural * __restrict__ p_info_major,
	nvals * p_n_major,
	T3 * p_T_major,
	NTrates * __restrict__ NTadditionrates,

	nvals * p_n_use,
	T3 * p_T_use,
	v4 * __restrict__ p_vie_use,
	f64_vec3 * __restrict__ p_v_n_use, 

	f64 * __restrict__ p_AreaMajor, 

	nvals * __restrict__ p_n_major_dest,
	T3 * __restrict__ p_T_major_dest
)
{
	// runs for major tile
	// nu would have been a better choice to go in shared as it coexists with the 18 doubles in "LHS","inverted".
	// Important to set 48K L1 for this routine.

	__shared__ nvals n_src_or_use[threadsPerTileMajor];
	__shared__ f64 AreaMajor[threadsPerTileMajor];

	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x; // iVertex OF VERTEX
	structural info = p_info_major[iVertex];
//	if (iVertex == CHOSEN) printf("GPU iVertex %d info.flag %d \n", CHOSEN, info.flag);

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)){

		n_src_or_use[threadIdx.x] = p_n_major[iVertex];  // used throughout so a good candidate to stick in shared mem
		AreaMajor[threadIdx.x] = p_AreaMajor[iVertex]; // ditto

		NTrates newdata;
		{
			NTrates AdditionNT = NTadditionrates[iVertex];
			newdata.N = n_src_or_use[threadIdx.x].n*AreaMajor[threadIdx.x] + h_use * AdditionNT.N;
			newdata.Nn = n_src_or_use[threadIdx.x].n_n*AreaMajor[threadIdx.x] + h_use * AdditionNT.Nn;
			newdata.NnTn = h_use * AdditionNT.NnTn; // start off without knowing 'factor' so we can ditch AdditionNT
			newdata.NiTi = h_use * AdditionNT.NiTi;
			newdata.NeTe = h_use * AdditionNT.NeTe;

			if (TEST)
				printf("Advance_nT  %d : nsrc %1.12E nn %1.12E *AreaMajor %1.12E %1.12E\n"
					"newdata.Nn %1.12E newdata.Ni %1.12E AreaMajor %1.10E \n"
					"h*additionNiTi %1.12E for e %1.12E for n %1.12E \n"
					"AdditionNT.e %1.10E h_use %1.10E\n"
					, VERTCHOSEN,
					n_src_or_use[threadIdx.x].n, n_src_or_use[threadIdx.x].n_n,
					n_src_or_use[threadIdx.x].n*AreaMajor[threadIdx.x],
					n_src_or_use[threadIdx.x].n_n*AreaMajor[threadIdx.x],
					newdata.Nn, newdata.N, AreaMajor[threadIdx.x],
					newdata.NiTi, newdata.NeTe, newdata.NnTn,
					AdditionNT.NeTe, h_use);
		}

		// So at this vertex, near the insulator, NiTi that comes in is NaN. Is that advection or diffusion?
		// Have to go to bed tonight...

		{
			nvals n_dest;
			n_dest.n = newdata.N / (AreaMajor[threadIdx.x]); 
			n_dest.n_n = newdata.Nn / (AreaMajor[threadIdx.x]);
			p_n_major_dest[iVertex] = n_dest;

		}

		f64 factor, factor_neut; // used again at end
		{
			T3 T_src = p_T_major[iVertex];
			newdata.NnTn += n_src_or_use[threadIdx.x].n_n*AreaMajor[threadIdx.x] * T_src.Tn;
			newdata.NiTi += n_src_or_use[threadIdx.x].n*AreaMajor[threadIdx.x] * T_src.Ti;
			newdata.NeTe += n_src_or_use[threadIdx.x].n*AreaMajor[threadIdx.x] * T_src.Te;
			
			//
			if (TEST) {
				printf("\nAdvance_nT %d : n %1.12E Area %1.12E compressfac %1.10E \n"
					"newdate.NiTi %1.12E Ti_k %1.12E newdata.NeTe %1.10E Te_k %1.10E\n"	, 
					VERTCHOSEN, n_src_or_use[threadIdx.x].n, AreaMajor[threadIdx.x], factor,
					newdata.NiTi,T_src.Ti, newdata.NeTe, T_src.Te);
			}
		}
		
		f64 nu_ne_MT, nu_en_MT, nu_ni_MT, nu_in_MT, nu_ei; // optimize after
		{
			// Dimensioning inside a brace allows the following vars to go out of scope at the end of the brace.
			f64 sqrt_Te, ionneut_thermal, electron_thermal, lnLambda, s_in_MT, s_en_MT, s_en_visc;

			n_src_or_use[threadIdx.x] = p_n_use[iVertex];
			T3 T_use = p_T_use[iVertex];

			sqrt_Te = sqrt(T_use.Te); // should be "usedata"
			ionneut_thermal = sqrt(T_use.Ti / m_ion + T_use.Tn / m_n); // hopefully not sqrt(0)
			electron_thermal = sqrt_Te * over_sqrt_m_e;
			lnLambda = Get_lnLambda_d(n_src_or_use[threadIdx.x].n, T_use.Te);

			{
				f64 s_in_visc_dummy;
				Estimate_Ion_Neutral_Cross_sections_d(T_use.Ti*one_over_kB,
					&s_in_MT, &s_in_visc_dummy);
			}
			Estimate_Ion_Neutral_Cross_sections_d(T_use.Te*one_over_kB, // call with T in electronVolts
				&s_en_MT,
				&s_en_visc);
			//s_en_MT = Estimate_Ion_Neutral_MT_Cross_section(T_use.Te*one_over_kB);
			//s_en_visc = Estimate_Ion_Neutral_Viscosity_Cross_section(T_use.Te*one_over_kB);
			
			if (n_src_or_use[threadIdx.x].n_n > ARTIFICIAL_RELATIVE_THRESH_HEAT *n_src_or_use[threadIdx.x].n) {
				s_en_MT *= n_src_or_use[threadIdx.x].n_n / (ARTIFICIAL_RELATIVE_THRESH_HEAT *n_src_or_use[threadIdx.x].n);
				s_in_MT *= n_src_or_use[threadIdx.x].n_n / (ARTIFICIAL_RELATIVE_THRESH_HEAT *n_src_or_use[threadIdx.x].n);
				// So at 1e18 vs 1e8 it's 10 times stronger
				// At 1e18 vs 1e6 it's 1000 times stronger
				// nu starts at about 1e11 at the place it failed at 35ns. So 10000 times stronger gives us 1e15.
			}

			// Need nu_ne etc to be defined:

			nu_ne_MT = s_en_MT * n_src_or_use[threadIdx.x].n * electron_thermal; // have to multiply by n_e for nu_ne_MT
			nu_ni_MT = s_in_MT * n_src_or_use[threadIdx.x].n * ionneut_thermal;
			nu_en_MT = s_en_MT * n_src_or_use[threadIdx.x].n_n*electron_thermal;
			nu_in_MT = s_in_MT * n_src_or_use[threadIdx.x].n_n*ionneut_thermal;

			nu_ei = nu_eiBarconst * kB_to_3halves*n_src_or_use[threadIdx.x].n*lnLambda /
				(T_use.Te*sqrt_Te);

			//		nu_ie = nu_ei;

			//		nu_eHeart = 1.87*nu_eiBar + data_k.n_n*s_en_visc*electron_thermal;
		}


		// For now doing velocity-independent resistive heating.
		// Because although we have a magnetic correction Upsilon_zz involved, we ignored it
		// since we are also squashing the effect of velocity-dependent collisions on vx and vy (which
		// would produce a current in the plane) and this squashing should create heat, which
		// maybe means it adds up to the velocity-independent amount of heating. 
		{
			f64_vec3 v_n = p_v_n_use[iVertex];
			v4 vie = p_vie_use[iVertex];

			newdata.NeTe += h_use*(AreaMajor[threadIdx.x] * TWOTHIRDS*nu_en_MT*m_en*(
				(v_n.x - vie.vxy.x)*(v_n.x - vie.vxy.x)
				+ (v_n.y - vie.vxy.y)*(v_n.y - vie.vxy.y)
				+ (v_n.z - vie.vez)*(v_n.z - vie.vez))

				+ AreaMajor[threadIdx.x] * TWOTHIRDS*nu_ei*m_ei*(vie.vez - vie.viz)*(vie.vez - vie.viz));

			newdata.NiTi += h_use*(AreaMajor[threadIdx.x] * TWOTHIRDS*nu_in_MT*M_in*m_n*(
				(v_n.x - vie.vxy.x)*(v_n.x - vie.vxy.x)
				+ (v_n.y - vie.vxy.y)*(v_n.y - vie.vxy.y)
				+ (v_n.z - vie.viz)*(v_n.z - vie.viz)));

			newdata.NnTn += h_use*(AreaMajor[threadIdx.x] * TWOTHIRDS*nu_ni_MT*M_in*m_i*(
				(v_n.x - vie.vxy.x)*(v_n.x - vie.vxy.x)
				+ (v_n.y - vie.vxy.y)*(v_n.y - vie.vxy.y)
				+ (v_n.z - vie.viz)*(v_n.z - vie.viz)));

		if (TEST) 
			printf("%d v_n.z %1.9E vie_use.viz %1.9E vie_use.vez %1.9E \n areamajor %1.8E\n"
				"nu_in %1.10E nu_en %1.8E \n"
				"Frictional htg (NT+=): n i e %1.10E %1.10E %1.10E\n",
				VERTCHOSEN, v_n.z, vie.viz, vie.vez, AreaMajor[threadIdx.x],
				nu_in_MT, nu_en_MT,
				h_use*(AreaMajor[threadIdx.x] * TWOTHIRDS*nu_ni_MT*M_in*m_i*(
				(v_n.x - vie.vxy.x)*(v_n.x - vie.vxy.x)
					+ (v_n.y - vie.vxy.y)*(v_n.y - vie.vxy.y)
					+ (v_n.z - vie.viz)*(v_n.z - vie.viz))),
				h_use*(AreaMajor[threadIdx.x] * TWOTHIRDS*nu_in_MT*M_in*m_n*(
				(v_n.x - vie.vxy.x)*(v_n.x - vie.vxy.x)
					+ (v_n.y - vie.vxy.y)*(v_n.y - vie.vxy.y)
					+ (v_n.z - vie.viz)*(v_n.z - vie.viz))),
				h_use*(AreaMajor[threadIdx.x] * TWOTHIRDS*nu_en_MT*m_en*(
				(v_n.x - vie.vxy.x)*(v_n.x - vie.vxy.x)
					+ (v_n.y - vie.vxy.y)*(v_n.y - vie.vxy.y)
					+ (v_n.z - vie.vez)*(v_n.z - vie.vez))

					+ AreaMajor[threadIdx.x] * TWOTHIRDS*nu_ei*m_ei*(vie.vez - vie.viz)*(vie.vez - vie.viz))
			);			
		}
		f64_tens3 inverted;
		{
			f64_tens3 LHS;
			// x = neutral
			// y = ion
			// z = elec
			// This is for NT
			f64 nu_ie = nu_ei;
			
			LHS.xx = 1.0 - h_use * (-M_en * nu_ne_MT - M_in * nu_ni_MT);
			LHS.xy = -h_use * (M_in * nu_in_MT);
			LHS.xz = -h_use *(M_en * nu_en_MT);

			LHS.yx = -h_use *  M_in * nu_ni_MT;
			LHS.yy = 1.0 - h_use * (-M_in * nu_in_MT - M_ei * nu_ie);
			LHS.yz = -h_use * M_ei * nu_ei;

			LHS.zx = -h_use * M_en * nu_ne_MT;
			LHS.zy = -h_use * M_ei * nu_ie; 
			LHS.zz = 1.0 - h_use * (-M_en * nu_en_MT - M_ei * nu_ei);
			
			// some indices appear reversed because NT not T.

			if (TEST) {
				printf("LHS | \n %1.14E %1.14E %1.14E |\n %1.14E %1.14E %1.14E |  \n %1.14E %1.14E %1.14E | \n",
					LHS.xx, LHS.xy, LHS.xz, 
					LHS.yx, LHS.yy, LHS.yz, 
					LHS.zx, LHS.zy, LHS.zz);
				printf("GPU %d : NnTn %1.14E NeTe %1.14E \n", CHOSEN, newdata.NnTn, newdata.NeTe);
				printf("GPU nu_en_MT %1.14E\n", nu_en_MT);
			} 
			LHS.Inverse(inverted);
		}

		f64_vec3 RHS;
		f64 nu_ie = nu_ei;
		RHS.x = newdata.NnTn - h_use * (nu_ni_MT*M_in + nu_ne_MT * M_en)*newdata.NnTn
			+ h_use * nu_in_MT*M_in*newdata.NiTi + h_use * nu_en_MT*M_en*newdata.NeTe;

		RHS.y = newdata.NiTi - h_use * (nu_in_MT*M_in + nu_ie * M_ei)*newdata.NiTi
			+ h_use * nu_ni_MT*M_in*newdata.NnTn + h_use * nu_ei*M_ei*newdata.NeTe;

		RHS.z = newdata.NeTe - h_use * (nu_en_MT*M_en + nu_ei * M_ei)*newdata.NeTe
			+ h_use * nu_ie*M_ei*newdata.NiTi + h_use * nu_ne_MT*M_en*newdata.NnTn;
		
		f64_vec3 NT;
		NT = inverted * RHS;
		newdata.NnTn = NT.x;
		newdata.NiTi = NT.y;
		newdata.NeTe = NT.z;

		T3 T_dest;
		T_dest.Tn = newdata.NnTn / newdata.Nn;
		T_dest.Ti = newdata.NiTi / newdata.N;
		T_dest.Te = newdata.NeTe/ newdata.N;

		if (TEST) {
			printf("\ninverted | RHS \n %1.14E %1.14E %1.14E | %1.14E \n %1.14E %1.14E %1.14E | %1.14E \n %1.14E %1.14E %1.14E | %1.14E \n",
				inverted.xx, inverted.xy, inverted.xz, RHS.x, 
				inverted.yx, inverted.yy, inverted.yz, RHS.y, 
				inverted.zx, inverted.zy, inverted.zz, RHS.z);
			printf("GPU %d : NnTn %1.14E NiTi %1.14E NeTe %1.14E \n"
				"Tn Ti Te %1.14E %1.14E %1.14E\n", VERTCHOSEN, newdata.NnTn, newdata.NiTi, newdata.NeTe,
				T_dest.Tn, T_dest.Ti, T_dest.Te);
		} // This came out with a value.

		if (TEST) printf("%d : T_dest %1.8E %1.8E %1.8E \n"
			"newdata .NeTe %1.10E .N %1.10E factor %1.10E\n\n",
			iVertex, T_dest.Tn, T_dest.Ti, T_dest.Te,
			newdata.NeTe, newdata.N, factor
		);

		if (T_dest.Te != T_dest.Te) {
			printf("Advance_n_T %d : Te NaN factor %1.8E newdata.N %1.10E flag %d \n"
				"n %1.10E Area %1.10E hd/dtNT %1.10E\n",
				iVertex, factor, newdata.N, info.flag,
				n_src_or_use[threadIdx.x].n,AreaMajor[threadIdx.x] , h_use * NTadditionrates[iVertex].N);
		}

		p_T_major_dest[iVertex] = T_dest;

	} else {
		// nothing to do ??
		if (info.flag == OUTERMOST) {
			p_n_major_dest[iVertex] = p_n_major[iVertex];
			p_T_major_dest[iVertex] = p_T_major[iVertex];
		}
		else {
			memset(p_n_major_dest + iVertex, 0, sizeof(nvals));
			memset(p_T_major_dest + iVertex, 0, sizeof(T3));
		};
	};
}

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
	CHAR4 * __restrict__ p_szPBCneigh_tris,
	nvals * __restrict__ p_n_upwind_minor, // result 

	T3 * __restrict__ p_T_minor,
	T3 * __restrict__ p_T_upwind_minor // result
)
{
	// The idea is to take the upwind n on each side of each
	// major edge through this tri, weighted by |v.edge_normal|
	// to produce an average.
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor]; // 4 doubles/vertex
	__shared__ f64_12 shared_shards[threadsPerTileMajor];  // + 12
														   // 15 doubles right there. Max 21 for 288 vertices. 16 is okay.
														   // Might as well stick 1 more double  in there if we get worried about registers.

														   // #############################################%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%###############
														   // We need a reverse index: this triangle carry 3 indices to know who it is to its corners.
	long const iTri = blockDim.x*blockIdx.x + threadIdx.x;
	structural const info = p_info_minor[iTri];
	nvals result;
	T3 upwindT;

	shared_pos[threadIdx.x] = info.pos;
	long const StartMajor = blockIdx.x*threadsPerTileMajor;
	long const EndMajor = StartMajor + threadsPerTileMajor;
	long const StartMinor = blockIdx.x*threadsPerTileMinor;
	long const EndMinor = StartMinor + threadsPerTileMinor;

	if (threadIdx.x < threadsPerTileMajor)
	{
		memcpy(&(shared_shards[threadIdx.x].n), &(p_n_shard_major[threadsPerTileMajor*blockIdx.x + threadIdx.x].n), MAXNEIGH * sizeof(f64));
		// efficiency vs memcpy? We only need 12 here, not the centre.
	}
	__syncthreads();

	f64 n0, n1, n2;
	T3 T0, T1, T2;
	f64_vec2 edge_normal0, edge_normal1, edge_normal2;
	LONG3 tricornerindex, trineighindex;
	LONG3 who_am_I;
	f64_vec2 v_overall;
	char szPBC_triminor[6];
	CHAR4 szPBC_neighs;
	if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS))
	{
		// Several things we need to collect:
		// . v in this triangle and mesh v at this triangle centre.
		// . edge_normal going each way
		// . n that applies from each corner

		// How to get n that applies from each corner:
		tricornerindex = p_tricornerindex[iTri];
		who_am_I = p_which_iTri_number_am_I[iTri];
		szPBC_neighs = p_szPBCneigh_tris[iTri];

		// Wasteful:
		T0 = p_T_minor[tricornerindex.i1 + BEGINNING_OF_CENTRAL];
		T1 = p_T_minor[tricornerindex.i2 + BEGINNING_OF_CENTRAL];
		T2 = p_T_minor[tricornerindex.i3 + BEGINNING_OF_CENTRAL];

		if ((tricornerindex.i1 >= StartMajor) && (tricornerindex.i1 < EndMajor))
		{
			n0 = shared_shards[tricornerindex.i1 - StartMajor].n[who_am_I.i1]; // whoa, be careful with data type / array
		}
		else {
			n0 = p_n_shard_major[tricornerindex.i1].n[who_am_I.i1];
			// at least it's 1 bus journey this way instead of 2 to fetch n_shards.
		}
		if ((tricornerindex.i2 >= StartMajor) && (tricornerindex.i2 < EndMajor))
		{
			n1 = shared_shards[tricornerindex.i2 - StartMajor].n[who_am_I.i2];
		}
		else {
			n1 = p_n_shard_major[tricornerindex.i2].n[who_am_I.i2];
			// at least it's 1 bus journey this way instead of 2 to fetch n_shards.
		}
		if ((tricornerindex.i3 >= StartMajor) && (tricornerindex.i3 < EndMajor))
		{
			n2 = shared_shards[tricornerindex.i3 - StartMajor].n[who_am_I.i3];
		}
		else {
			n2 = p_n_shard_major[tricornerindex.i3].n[who_am_I.i3];
			// at least it's 1 bus journey this way instead of 2 to fetch n_shards.
		}

		v_overall = p_overall_v_minor[iTri];
		f64_vec2 relv = p_vie_minor[iTri].vxy - v_overall;
		
		if (info.flag == CROSSING_INS) {
			int number_within = (n0 > 0.0) ? 1 : 0 + (n1 > 0.0) ? 1 : 0 + (n2 > 0.0) ? 1 : 0;
			if (number_within == 1) {
				result.n = n0 + n1 + n2;
				upwindT.Te = T0.Te + T1.Te + T2.Te;
				upwindT.Tn = T0.Tn + T1.Tn + T2.Tn;
				upwindT.Ti = T0.Ti + T1.Ti + T2.Ti; 
			}
			else {
				// quick way not upwind: 
				result.n = 0.5*(n0 + n1 + n2);
				upwindT.Te = 0.5*(T0.Te + T1.Te + T2.Te);
				upwindT.Tn = 0.5*(T0.Tn + T1.Tn + T2.Tn);
				upwindT.Ti = 0.5*(T0.Ti + T1.Ti + T2.Ti); // watch out for heat evacuating CROSSING_INS tris.
			}
			//if (iTri == 23400) printf("\n23400 was an insulator tri, T012 %1.8E %1.8E %1.8E upwind %1.8E\n"
			//	"indexcorner %d %d %d\n\n",
			//	T0.Te,T1.Te,T2.Te,upwindT.Te,
			//	tricornerindex.i1, tricornerindex.i1, tricornerindex.i3);
		} else {

			trineighindex = p_trineighindex[iTri];

		//	if (iTri == CHOSEN) printf("%d GPU: n0 %1.14E n1 %1.14E n2 %1.14E \n"
		//		"relv GPU %1.14E %1.14E \n",
		//		CHOSEN, n0, n1, n2, relv.x, relv.y);

			f64_vec2 nearby_pos;
			if ((trineighindex.i1 >= StartMinor) && (trineighindex.i1 < EndMinor)) {
				nearby_pos = shared_pos[trineighindex.i1 - StartMinor];
			}
			else {
				nearby_pos = p_info_minor[trineighindex.i1].pos;
			}
			if (szPBC_neighs.per0 == ROTATE_ME_CLOCKWISE) {
				nearby_pos = Clockwise_d*nearby_pos;
			}
			if (szPBC_neighs.per0 == ROTATE_ME_ANTICLOCKWISE) {
				nearby_pos = Anticlockwise_d*nearby_pos;
			}
			// Slightly puzzled why we don't just take difference of 2 corners of our triangle.
			// Why dealing with tri positions instead of vertex positions? Because tri positions
			// are the corners of the major cell.

			edge_normal0.x = nearby_pos.y - info.pos.y;
			edge_normal0.y = info.pos.x - nearby_pos.x;
			// CAREFUL : which side is which???
			// tri centre 2 is on same side of origin as corner 1 -- I think
			// We don't know if the corners have been numbered anticlockwise?
			// Could arrange it though.
			// So 1 is anticlockwise for edge 0.

			f64 numerator = 0.0;
			f64 dot1, dot2;
			f64 dot0 = relv.dot(edge_normal0);

			if ((trineighindex.i2 >= StartMinor) && (trineighindex.i2 < EndMinor)) {
				nearby_pos = shared_pos[trineighindex.i2 - StartMinor];
			}
			else {
				nearby_pos = p_info_minor[trineighindex.i2].pos;
			}
			if (szPBC_neighs.per1 == ROTATE_ME_CLOCKWISE) {
				nearby_pos = Clockwise_d*nearby_pos;
			}
			if (szPBC_neighs.per1 == ROTATE_ME_ANTICLOCKWISE) {
				nearby_pos = Anticlockwise_d*nearby_pos;
			}
			edge_normal1.x = nearby_pos.y - info.pos.y;
			edge_normal1.y = info.pos.x - nearby_pos.x;

			dot1 = relv.dot(edge_normal1);
			if ((trineighindex.i3 >= StartMinor) && (trineighindex.i3 < EndMinor)) {
				nearby_pos = shared_pos[trineighindex.i3 - StartMinor];
			}
			else {
				nearby_pos = p_info_minor[trineighindex.i3].pos;
			}
			if (szPBC_neighs.per2 == ROTATE_ME_CLOCKWISE) {
				nearby_pos = Clockwise_d*nearby_pos;
			}
			if (szPBC_neighs.per2 == ROTATE_ME_ANTICLOCKWISE) {
				nearby_pos = Anticlockwise_d*nearby_pos;
			}

			edge_normal2.x = nearby_pos.y - info.pos.y;
			edge_normal2.y = info.pos.x - nearby_pos.x;

			dot2 = relv.dot(edge_normal2);

			bool b0, b1, b2; // is this n012 legit?
			if (dot0 > 0.0) { b2 = 1; }
			else { b1 = 1; };
			if (dot1 > 0.0) { b0 = 1; }
			else { b2 = 1; };
			if (dot2 > 0.0) { b1 = 1; }
			else { b0 = 1; };
			
			//Usually now only one of b012 is false.

			if (b0 == 0) {
				if (b1 == 0) {
					result.n = n2; // how idk
					memcpy(&upwindT, &T2, sizeof(T3));
				} else {
					if (b2 == 0) { 
						result.n = n1;
						memcpy(&upwindT, &T1, sizeof(T3));
					} else {
						result.n = min(n1, n2);
						upwindT.Te = min(T1.Te, T2.Te);
						upwindT.Ti = min(T1.Ti, T2.Ti);
					}
				}
			} else {
				if ((b1 == 0) && (b2 == 0)) {
					result.n = n0;
					memcpy(&upwindT, &T0, sizeof(T3));
				} else {
					if (b1 == 0) {
						result.n = min(n0, n2);
						memcpy(&upwindT, &T2, sizeof(T3));
					} else {
						if (b2 == 0)
						{
							result.n = min(n0, n1);
							upwindT.Te = min(T1.Te, T0.Te);
							upwindT.Ti = min(T1.Ti, T0.Ti);
						} else {
							result.n = min(min(n0, n1), n2);
							upwindT.Te = min(T0.Te, min(T1.Te, T2.Te));
							upwindT.Ti = min(T0.Ti, min(T1.Ti, T2.Ti));
						}
					}
				}
			}
		//	if (iTri == 23435) printf("CALC UPWIND n\n"
		//		"tricornerindex %d %d %d\n"
		//		"n0 n1 n2 %1.12E %1.12E %1.12E\n"
		//		"relv %1.9E %1.9E \n"
		//		"edge_nml %1.9E %1.9E | %1.9E %1.9E | %1.9E %1.9E \n"
		//		"dot %1.9E %1.9E %1.9E\n"
		//		"b0 b1 b2 %d %d %d \n"
		//		"result.n %1.9E\n\n",
		//		tricornerindex.i1, tricornerindex.i2, tricornerindex.i3,
		//		n0, n1, n2,
		//		relv.x, relv.y,
		//		edge_normal0.x, edge_normal0.y, edge_normal1.x, edge_normal1.y, edge_normal2.x, edge_normal2.y,
		//		dot0, dot1, dot2,
		//		(b0 ? 1 : 0), (b1 ? 1 : 0), (b2 ? 1 : 0),
		//		result.n);
		//	
//
//			if (iTri == 23400) printf("\n23400 was a domain tri, T012 %1.8E %1.8E %1.8E upwind %1.8E\n"
//				"relv %1.8E %1.8E b012 %d %d %d \n\n",
//				T0.Te, T1.Te, T2.Te, upwindT.Te,
//				relv.x, relv.y, (int)b0, (int)b1, (int)b2);*/
						// Alternative way: try using squared weights of upwind n for v.dot(edgenormal).

			// This old, doesn't work when JxB force empties out near ins:
			/*
	//		if (iTri == CHOSEN) printf("GPU %d: edge_normal0 %1.14E %1.14E dot0 %1.14E \n"
	//			"nearby_pos %1.14E %1.14E trineighindex.i1 %d\n",
	//			CHOSEN, edge_normal0.x, edge_normal0.y, dot0,
	//			nearby_pos.x, nearby_pos.y, trineighindex.i1);

			if (dot0 > 0.0) // v faces anticlockwise
			{
				numerator += dot0*n2;
			}
			else {
				dot0 = -dot0;
				numerator += dot0*n1;
			}

			if ((trineighindex.i2 >= StartMinor) && (trineighindex.i2 < EndMinor)) {
				nearby_pos = shared_pos[trineighindex.i2 - StartMinor];
			}
			else {
				nearby_pos = p_info_minor[trineighindex.i2].pos;
			}
			if (szPBC_neighs.per1 == ROTATE_ME_CLOCKWISE) {
				nearby_pos = Clockwise_d*nearby_pos;
			}
			if (szPBC_neighs.per1 == ROTATE_ME_ANTICLOCKWISE) {
				nearby_pos = Anticlockwise_d*nearby_pos;
			}
			edge_normal1.x = nearby_pos.y - info.pos.y;
			edge_normal1.y = info.pos.x - nearby_pos.x;

			dot1 = relv.dot(edge_normal1);

	//		if (iTri == CHOSEN) printf("GPU: edge_normal1 %1.14E %1.14E dot1 %1.14E \n"
	//			"nearby_pos %1.14E %1.14E trineighindex.i2 %d\n",
	//			edge_normal1.x, edge_normal1.y, dot1,
	//			nearby_pos.x, nearby_pos.y, trineighindex.i2);

			if (dot1 > 0.0)
			{
				numerator += dot1*n0;
			}
			else {
				dot1 = -dot1;
				numerator += dot1*n2;
			}

			if ((trineighindex.i3 >= StartMinor) && (trineighindex.i3 < EndMinor)) {
				nearby_pos = shared_pos[trineighindex.i3 - StartMinor];
			}
			else {
				nearby_pos = p_info_minor[trineighindex.i3].pos;
			}
			if (szPBC_neighs.per2 == ROTATE_ME_CLOCKWISE) {
				nearby_pos = Clockwise_d*nearby_pos;
			}
			if (szPBC_neighs.per2 == ROTATE_ME_ANTICLOCKWISE) {
				nearby_pos = Anticlockwise_d*nearby_pos;
			}

			edge_normal2.x = nearby_pos.y - info.pos.y;
			edge_normal2.y = info.pos.x - nearby_pos.x;

			dot2 = relv.dot(edge_normal2);

	//		if (iTri == CHOSEN) printf("GPU: edge_normal2 %1.14E %1.14E dot2 %1.14E \n",
	//			edge_normal2.x, edge_normal2.y, dot2);

			if (dot2 > 0.0)
			{
				numerator += dot2*n1;
			}
			else {
				dot2 = -dot2;
				numerator += dot2*n0;
			}
			
			// Already did fabs so can do just this test without squaring:
			if (dot0 + dot1 + dot2 == 0.0) {
				result.n = THIRD*(n0 + n1 + n2);
	//			if (iTri == CHOSEN) printf("Got to here. GPU. n = %1.14E \n", result.n);
			}
			else {
				result.n = numerator / (dot0 + dot1 + dot2);
				if (iTri == 23435) printf("\n23435 : denom = %1.14E n = %1.10E n012 %1.9E %1.9E %1.9E\n"
					"dot012 %1.9E %1.9E %1.9E relv %1.9E %1.9E\n"
					"edgenormals012 %1.9E %1.9E | %1.9E %1.9E | %1.9E %1.9E\n\n",
					dot0 + dot1 + dot2, result.n, n0, n1, n2,
					dot0, dot1, dot2, relv.x, relv.y,
					edge_normal0.x, edge_normal0.y, edge_normal1.x, edge_normal1.y, edge_normal2.x, edge_normal2.y);

			};*/
			// Argument against fabs in favour of squared weights?
		};
		// Think carefully / debug how it goes for CROSSING_INS.
	} else {
		result.n = 0.0;
		memset(&upwindT, 0, sizeof(T3));
	};

	// Now same for upwind neutral density:
	// In order to use syncthreads we had to come out of the branching.

	if (threadIdx.x < threadsPerTileMajor)
	{
		memcpy(&(shared_shards[threadIdx.x].n), 
			&(p_n_shard_n_major[threadsPerTileMajor*blockIdx.x + threadIdx.x].n),
			sizeof(f64)*MAXNEIGH);
		// efficiency vs memcpy? We only need 12 here, not the centre.
	}
	__syncthreads();

	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

	if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS))
	{
		if ((tricornerindex.i1 >= StartMajor) && (tricornerindex.i1 < EndMajor))
		{
			n0 = shared_shards[tricornerindex.i1 - StartMajor].n[who_am_I.i1];
		}
		else {
			n0 = p_n_shard_n_major[tricornerindex.i1].n[who_am_I.i1];
			// at least it's 1 bus journey this way instead of 2 to fetch n_shards.
		}
		if ((tricornerindex.i2 >= StartMajor) && (tricornerindex.i2 < EndMajor))
		{
			n1 = shared_shards[tricornerindex.i2 - StartMajor].n[who_am_I.i2];
		} else {
			n1 = p_n_shard_n_major[tricornerindex.i2].n[who_am_I.i2];
			// at least it's 1 bus journey this way instead of 2 to fetch n_shards.
		}
		if ((tricornerindex.i3 >= StartMajor) && (tricornerindex.i3 < EndMajor))
		{
			n2 = shared_shards[tricornerindex.i3 - StartMajor].n[who_am_I.i3];
		} else {
			n2 = p_n_shard_n_major[tricornerindex.i3].n[who_am_I.i3];
			// at least it's 1 bus journey this way instead of 2 to fetch n_shards.
		}

		f64_vec2 relv = p_v_n_minor[iTri].xypart() - v_overall;

		if (info.flag == CROSSING_INS) {
			int number_within = (n0 > 0.0) ? 1 : 0 + (n1 > 0.0) ? 1 : 0 + (n2 > 0.0) ? 1 : 0;
			if (number_within == 1) {
				result.n_n = n0 + n1 + n2;
				upwindT.Tn = T0.Tn + T1.Tn + T2.Tn;
			} else {
				// quick way not upwind:
				result.n_n = 0.5*(n0 + n1 + n2);
				upwindT.Tn = 0.5*(T0.Tn + T1.Tn + T2.Tn);
			};
		} else {

			f64 numerator = 0.0;
			f64 dot1, dot2;
			f64 dot0 = relv.dot(edge_normal0);
			dot1 = relv.dot(edge_normal1);
			dot2 = relv.dot(edge_normal2);
			
			bool b0, b1, b2; // is this n012 legit?
			if (dot0 > 0.0) { b2 = 1; }
			else { b1 = 1; };
			if (dot1 > 0.0) { b0 = 1; }
			else { b2 = 1; };
			if (dot2 > 0.0) { b1 = 1; }
			else { b0 = 1; };

			//Usually now only one of b012 is false.

			if (b0 == 0) {
				if (b1 == 0) {
					result.n_n = n2; // how idk

					upwindT.Tn = T2.Tn;
				}
				else {
					if (b2 == 0) { result.n = n1; }
					else {
						result.n_n = min(n1, n2);

						upwindT.Tn = min(T1.Tn, T2.Tn);
					}
				}
			}
			else {
				if ((b1 == 0) && (b2 == 0)) {
					result.n_n = n0;

					upwindT.Tn = T0.Tn;
				}
				else {
					if (b1 == 0) {
						result.n_n = min(n0, n2);

						upwindT.Tn = min(T2.Tn, T0.Tn);
					}
					else {
						if (b2 == 0)
						{
							result.n_n = min(n0, n1);

							upwindT.Tn = min(T1.Tn, T0.Tn);
						} else {
							result.n_n = min(min(n0, n1), n2);

							upwindT.Tn = min(min(T1.Tn, T0.Tn), T2.Tn);
						}
					}
				}
			}




			/*
		//	if (iTri == CHOSEN) {
		//		printf("GPU calc n_n: n %1.10E %1.10E %1.10E \n dot %1.10E %1.10E %1.10E relv %1.10E %1.10E\n",
		//			n0, n1, n2, dot0, dot1, dot2,
		//			relv.x, relv.y);
		//	}
			if (dot0 > 0.0) // v faces anticlockwise
			{
				numerator += dot0*n2;
			}
			else {
				dot0 = -dot0;
				numerator += dot0*n1;
			}

			if (dot1 > 0.0)
			{
				numerator += dot1*n0;
			}
			else {
				dot1 = -dot1;
				numerator += dot1*n2;
			}

			if (dot2 > 0.0)
			{
				numerator += dot2*n1;
			}
			else {
				dot2 = -dot2;
				numerator += dot2*n0;
			}

			if (dot0 + dot1 + dot2 == 0.0) {
				result.n_n = THIRD*(n0 + n1 + n2);
			}
			else {
				result.n_n = numerator / (dot0 + dot1 + dot2);
			};*/
			// Look carefully at what happens for CROSSING_INS.
			// relv should be horizontal, hence it doesn't give a really low density? CHECK IT IN PRACTICE.
		};
	} else {
		result.n_n = 0.0;
		upwindT.Tn = 0.0;
	};

	p_n_upwind_minor[iTri] = result;
	p_T_upwind_minor[iTri] = upwindT;

}
__global__ void kernelAccumulateAdvectiveMassHeatRate(
	f64 const h_use,
	structural * __restrict__ p_info_minor,
	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBCtri_verts,

	nvals * __restrict__ p_n_src_major,
	T3 * __restrict__ p_T_src_major,

	nvals * __restrict__ p_n_upwind_minor,
	v4 * __restrict__ p_vie_minor,
	f64_vec3 * __restrict__ p_v_n_minor,
	f64_vec2 * __restrict__ p_v_overall_minor,
	//T3 * __restrict__ p_T_minor, // may or may not overlap source: don't we only use from tris? so not overlap

	T3 * __restrict__ p_T_upwind_minor,

	NTrates * __restrict__ p_NTadditionrates,
	f64 * __restrict__ p_div_v,
	f64 * __restrict__ p_div_v_n,
	f64 * __restrict__ p_Integrated_div_v_overall
)
{
	// Use the upwind density from tris together with v_tri.
	// Seems to include a factor h

	__shared__ f64_vec2 shared_pos[threadsPerTileMinor]; // only reused what, 3 times?
	__shared__ nvals shared_n_upwind[threadsPerTileMinor];
	__shared__ f64_vec2 shared_vxy[threadsPerTileMinor];
	__shared__ f64_vec2 shared_v_n[threadsPerTileMinor];
	//__shared__ f64_vec2 v_overall[threadsPerTileMinor];
	// choosing just to load it ad hoc
	__shared__ T3 shared_T[threadsPerTileMinor];

	// Do neutral after? Necessitates doing all the random loads again.
	// Is that worse than loading for each point at the time, a 2-vector v_overall?
	// About 6 bus journeys per external point. About 1/4 as many external as internal?
	// ^ only 6 because doing ion&neutral together. Changing to do sep could make sense.

	// 2* (2+2+2+2+3) = 22
	// Max viable threads at 26: 236
	// Max viable threads at 24: 256

	// Can't store rel v: we use div v of each v in what follows.

	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;
	{
		structural info[2];
		memcpy(info, p_info_minor + (threadsPerTileMinor*blockIdx.x + 2 * threadIdx.x), sizeof(structural) * 2);
		shared_pos[2 * threadIdx.x] = info[0].pos;
		shared_pos[2 * threadIdx.x + 1] = info[1].pos;
		
		memcpy(&(shared_n_upwind[2 * threadIdx.x]), 
			p_n_upwind_minor + (threadsPerTileMinor*blockIdx.x + 2 * threadIdx.x), sizeof(nvals) * 2);
		
		v4 vie[2];
		memcpy(&vie, p_vie_minor + (threadsPerTileMinor*blockIdx.x + 2 * threadIdx.x), sizeof(v4) * 2);
		shared_vxy[2 * threadIdx.x] = vie[0].vxy;
		shared_vxy[2 * threadIdx.x + 1] = vie[1].vxy;
		f64_vec3 v_n[2];
		memcpy(v_n, p_v_n_minor + (threadsPerTileMinor*blockIdx.x + 2 * threadIdx.x), sizeof(f64_vec3) * 2);
		shared_v_n[2 * threadIdx.x] = v_n[0].xypart();
		shared_v_n[2 * threadIdx.x + 1] = v_n[1].xypart();
		memcpy(&(shared_T[2 * threadIdx.x]), p_T_upwind_minor + (threadsPerTileMinor*blockIdx.x + 2 * threadIdx.x), sizeof(T3) * 2);
	}
	long const StartMinor = threadsPerTileMinor*blockIdx.x;
	long const EndMinor = threadsPerTileMinor + StartMinor;

	__syncthreads();

	// What happens for abutting ins?
	// T defined reasonably at insulator-crossing tri, A defined, v defined reasonably

	structural info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];

	if (info.flag == DOMAIN_VERTEX) {

		// T3 Tsrc = p_T_src_major[iVertex]; // UNUSED!
		nvals nsrc = p_n_src_major[iVertex];
		long izTri[MAXNEIGH];
		char szPBC[MAXNEIGH];
		memcpy(izTri, p_izTri + iVertex * MAXNEIGH, sizeof(long) * MAXNEIGH);
		short tri_len = info.neigh_len;
		memcpy(szPBC, p_szPBCtri_verts + iVertex*MAXNEIGH, sizeof(char)*MAXNEIGH);
		// Now we are assuming what? Neigh 0 is below tri 0, so 0 1 are on neigh 0
		// Check in debug. Looks true from comments.

		f64_vec2 edge_normal, endpt0, endpt1;
		f64_vec2 vxy_prev, vxy_next;
		f64_vec2 v_n_prev, v_n_next;
		f64 n_next, n_prev, nn_next, nn_prev;
		f64_vec2 v_overall_prev, v_overall_next;
		f64 Te_next, Te_prev, Ti_next, Ti_prev, Tn_next, Tn_prev;

		short inext, i = 0;
		long iTri = izTri[0];
		v_overall_prev = p_v_overall_minor[iTri];
		if ((iTri >= StartMinor) && (iTri < EndMinor)) {
			endpt0 = shared_pos[iTri - StartMinor];
			nvals nvls = shared_n_upwind[iTri - StartMinor];

			n_prev = nvls.n;
			nn_prev = nvls.n_n;
			vxy_prev = shared_vxy[iTri - StartMinor];
			v_n_prev = shared_v_n[iTri - StartMinor];
			Te_prev = shared_T[iTri - StartMinor].Te;
			Ti_prev = shared_T[iTri - StartMinor].Ti;
			Tn_prev = shared_T[iTri - StartMinor].Tn;

		} else {
			// The volume of random bus accesses means that we would have been better off making a separate
			// neutral routine even though it looks efficient with the shared loading. nvm
			endpt0 = p_info_minor[iTri].pos;
			nvals n_upwind = p_n_upwind_minor[iTri];
			n_prev = n_upwind.n;

			nn_prev = n_upwind.n_n;
			vxy_prev = p_vie_minor[iTri].vxy;
			v_n_prev = p_v_n_minor[iTri].xypart();
			T3 Tuse = p_T_upwind_minor[iTri];
			Te_prev = Tuse.Te;
			Ti_prev = Tuse.Ti;
			Tn_prev = Tuse.Tn;
		};
		if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
			endpt0 = Clockwise_d*endpt0;
			vxy_prev = Clockwise_d*vxy_prev;
			v_n_prev = Clockwise_d*v_n_prev;
			v_overall_prev = Clockwise_d*v_overall_prev;
		};
		if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
			endpt0 = Anticlockwise_d*endpt0;
			vxy_prev = Anticlockwise_d*vxy_prev;
			v_n_prev = Anticlockwise_d*v_n_prev;
			v_overall_prev = Anticlockwise_d*v_overall_prev;
		};

		nvals totalmassflux_out;
		memset(&totalmassflux_out, 0, sizeof(nvals));
		T3 totalheatflux_out;
		memset(&totalheatflux_out, 0, sizeof(T3));
		f64 Integrated_div_v = 0.0;
		f64 Integrated_div_v_n = 0.0;
		f64 Integrated_div_v_overall = 0.0;
		f64 AreaMajor = 0.0;

#pragma unroll MAXNEIGH
		for (i = 0; i < tri_len; i++)
		{
			inext = i + 1; if (inext == tri_len) inext = 0;

			long iTri = izTri[inext];
			f64_vec2 v_overall_next = p_v_overall_minor[iTri];
			if ((iTri >= StartMinor) && (iTri < EndMinor)) {
				endpt1 = shared_pos[iTri - StartMinor];
				nvals nvls = shared_n_upwind[iTri - StartMinor];
				n_next = nvls.n;

				nn_next = nvls.n_n;
				vxy_next = shared_vxy[iTri - StartMinor];
				v_n_next = shared_v_n[iTri - StartMinor];
				Te_next = shared_T[iTri - StartMinor].Te;
				Ti_next = shared_T[iTri - StartMinor].Ti;
				Tn_next = shared_T[iTri - StartMinor].Tn;
			} else {
				// The volume of random bus accesses means that we would have been better off making a separate
				// neutral routine even though it looks efficient with the shared loading. nvm
				endpt1 = p_info_minor[iTri].pos;
				nvals n_upwind = p_n_upwind_minor[iTri];
				n_next = n_upwind.n;

				nn_next = n_upwind.n_n;
				vxy_next = p_vie_minor[iTri].vxy;
				v_n_next = p_v_n_minor[iTri].xypart();
				T3 Tuse = p_T_upwind_minor[iTri];
				Te_next = Tuse.Te;
				Ti_next = Tuse.Ti;
				Tn_next = Tuse.Tn;
			};
			if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
				endpt1 = Clockwise_d*endpt1;
				vxy_next = Clockwise_d*vxy_next;
				v_n_next = Clockwise_d*v_n_next;
				v_overall_next = Clockwise_d*v_overall_next;
			};
			if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
				endpt1 = Anticlockwise_d*endpt1;
				vxy_next = Anticlockwise_d*vxy_next;
				v_n_next = Anticlockwise_d*v_n_next;
				v_overall_next = Anticlockwise_d*v_overall_next;
			};
			
			f64_vec2 edge_normal;
			edge_normal.x = endpt1.y - endpt0.y;
			edge_normal.y = endpt0.x - endpt1.x;

			AreaMajor += 0.5*edge_normal.x*(endpt0.x + endpt1.x);

		//	if (iVertex == CHOSEN) printf("GPU %d : AreaMajor %1.9E edge_nml.x %1.6E endpt0.x %1.6E endpt1.x %1.6E \n",
		//		iVertex,
		//		AreaMajor, edge_normal.x, endpt0.x, endpt1.x);

			Integrated_div_v += 0.5*(vxy_prev + vxy_next).dot(edge_normal);
			Integrated_div_v_n += 0.5*(v_n_prev + v_n_next).dot(edge_normal);
			Integrated_div_v_overall += 0.5*(v_overall_prev + v_overall_next).dot(edge_normal); // Average outward velocity of edge...

			totalmassflux_out.n += 0.5*(n_prev*(vxy_prev-v_overall_prev)
				                      + n_next*(vxy_next-v_overall_next)).dot(edge_normal);
			totalheatflux_out.Ti += 0.5*(n_prev*Ti_prev*(vxy_prev-v_overall_prev)
									   + n_next*Ti_next*(vxy_next-v_overall_next)).dot(edge_normal);
			totalheatflux_out.Te += 0.5*(n_prev*Te_prev*(vxy_prev-v_overall_prev)
									   + n_next*Te_next*(vxy_next-v_overall_next)).dot(edge_normal);


			totalmassflux_out.n_n += 0.5*(nn_prev*(v_n_prev-v_overall_prev)
								+ nn_next*(v_n_next-v_overall_next)).dot(edge_normal);
			totalheatflux_out.Tn += 0.5*(nn_prev*Tn_prev*(v_n_prev-v_overall_prev)
								+ nn_next*Tn_next*(v_n_next-v_overall_next)).dot(edge_normal);
//
		//	if (TEST) printf("advect GPU %d : "
		//		"i %d iTri %d heatfluxout_contrib %1.14E \n"
		//		"nprev %1.14E nnext %1.14E\n"
		//		"Ti_prev next %1.14E %1.14E \nrel vxy %1.14E %1.14E ; %1.14E %1.14E\n"
		//		"edge_normal %1.14E %1.14E \n"
		//		"-------------------------\n",
		//		iVertex, i, iTri,
		//		0.5*(n_prev*Ti_prev*(vxy_prev - v_overall_prev)
		//			+ n_next*Ti_next*(vxy_next - v_overall_next)).dot(edge_normal),
		//		n_prev, n_next,
		//		Ti_prev, Ti_next, (vxy_prev-v_overall_prev).x, (vxy_prev - v_overall_prev).y,
		//		(vxy_next - v_overall_next).x, (vxy_next - v_overall_next).y,
		//		edge_normal.x, edge_normal.y);
			if (TEST) printf("advect GPU %d : "
				"i %d iTri %d heatfluxout_contrib e %1.14E \n"
				"nprev %1.14E nnext %1.14E\n"
				"Te_prev next %1.14E %1.14E \nrel vxy %1.14E %1.14E ; %1.14E %1.14E\n"
				"edge_normal %1.14E %1.14E \n"
				"-------------------------\n",
				iVertex, i, iTri,
				0.5*(n_prev*Te_prev*(vxy_prev - v_overall_prev)
					+ n_next*Te_next*(vxy_next - v_overall_next)).dot(edge_normal)	,
				n_prev, n_next,
				Ti_prev, Te_next, (vxy_prev - v_overall_prev).x, (vxy_prev - v_overall_prev).y,
				(vxy_next - v_overall_next).x, (vxy_next - v_overall_next).y,
				edge_normal.x, edge_normal.y);
//
			endpt0 = endpt1;
			n_prev = n_next;
			nn_prev = nn_next;
			vxy_prev = vxy_next;
			v_n_prev = v_n_next;
			v_overall_prev = v_overall_next;
			Ti_prev = Ti_next;
			Te_prev = Te_next;
			Tn_prev = Tn_next;
		};

		NTrates NTplus;

		NTplus.N = -totalmassflux_out.n;
		NTplus.Nn = -totalmassflux_out.n_n;
		NTplus.NeTe = -totalheatflux_out.Te;
		NTplus.NiTi = -totalheatflux_out.Ti;
		NTplus.NnTn = -totalheatflux_out.Tn;
//
//		if (TEST) printf("\n%d : NTplus.NiTi %1.10E NTplus.N %1.10E Tsrc.i %1.10E nsrc.n %1.10E\n"
//			"NTplus.NiTi/NTplus.N (avg temp of those coming/going) %1.10E\n"
//			"NTplus.NiTi/N (ROC Ti) %1.10E\n"
//			"NTplus.NiTi/NiTi (elasticity of T?) %1.10E \n"
//			"NTplus.N/N (elasticity of N) %1.10E \n\n",
//			CHOSEN, NTplus.NiTi, NTplus.N,
//			Tsrc.Ti, nsrc.n,
//			NTplus.NiTi/NTplus.N,
//			NTplus.NiTi/(AreaMajor*nsrc.n),
//			NTplus.NiTi/(AreaMajor*nsrc.n*Tsrc.Ti),
//			NTplus.N/(AreaMajor*nsrc.n)
//			);

		memcpy(p_NTadditionrates + iVertex, &NTplus, sizeof(NTrates));

		// What we need now: 
		//	* Cope with non-domain vertex
		p_div_v[iVertex] = Integrated_div_v / AreaMajor;
		p_div_v_n[iVertex] = Integrated_div_v_n / AreaMajor;
		p_Integrated_div_v_overall[iVertex] = Integrated_div_v_overall;

	//	if (iVertex == CHOSEN) printf(
	//			"Chosen: %d Integrated_div_v_n %1.9E p_div_v_n %1.9E \n",
	//			iVertex, Integrated_div_v_n, p_div_v_n[iVertex]);

		// 3 divisions -- could speed up by creating 1.0/AreaMajor. Except it's bus time anyway.
	} else {
		p_div_v[iVertex] = 0.0;
		p_div_v_n[iVertex] = 0.0;
		p_Integrated_div_v_overall[iVertex] = 0.0;
	};
}


__global__ void kernelCreateLinearRelationship(
	f64 const h_use,
	structural * __restrict__ p_info,
	OhmsCoeffs* __restrict__ p_Ohms,
	v4 * __restrict__ p_v0,
	f64 * __restrict__ p_Lap_Az_use,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_denom_e,
	f64 * __restrict__ p_denom_i,
	f64 * __restrict__ p_coeff_of_vez_upon_viz,
	f64 * __restrict__ p_beta_ie_z,
	AAdot * __restrict__ p_AAdot_intermediate,
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ p_Azdot0,
	f64 * __restrict__ p_gamma
)
{
	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	f64 const Lap_Az_used = p_Lap_Az_use[iMinor];
	structural const info = p_info[iMinor];

	if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == DOMAIN_VERTEX))
	{
		v4 v0 = p_v0[iMinor];
		// Cancel the part that was added in order to get at Ez_strength:

		f64 denom_e = p_denom_e[iMinor];
		f64 denom_i = p_denom_i[iMinor];

		if (((TESTTRI)) && (0)) printf("\nv0.vez before remove Lapcontrib %1.14E \n", v0.vez);

		v0.viz += 0.5*qoverM*h_use*h_use* c* Lap_Az_used / denom_i; // adaptation for this.
		f64 coeff_of_vez_upon_viz = p_coeff_of_vez_upon_viz[iMinor];

		f64 cancel_from_vez = -0.5*eoverm*h_use*h_use* c* Lap_Az_used / denom_e
			+ coeff_of_vez_upon_viz * 0.5*qoverM*h_use*h_use* c* Lap_Az_used / denom_i;

		v0.vez += cancel_from_vez;
		f64 beta_ie_z = p_beta_ie_z[iMinor];
		v0.viz += beta_ie_z * cancel_from_vez;

		if (((TESTTRI)) && (0)) printf("\n##############\nviz before remove LapAzcontrib %1.14E Lapcontrib %1.14E \n\n",
			v0.viz - 0.5*qoverM*h_use*h_use* c* Lap_Az_used / denom_i,
			-0.5*qoverM*h_use*h_use* c* Lap_Az_used / denom_i
		);

		// Inadequate because we need to take account of the effect of Lap Az on vez0 via viz0.

		// We see now that re-jigging things is absolutely not what we should have done.
		// It will make the most complicated overspilling routine, more complicated still.
		if (((TESTTRI)) && (0)) printf("own part of effect (we cancel): %1.14E \n"
			"via viz (we cancel): coeff %1.14E vizeffect %1.14E\n",
			0.5*eoverm*h_use*h_use* c* Lap_Az_used / denom_e,
			coeff_of_vez_upon_viz,
			-0.5*qoverM*h_use*h_use* c* Lap_Az_used / denom_i);

		if (((TESTTRI)) && (0)) printf("v0.vez after remove Lapcontrib %1.14E \n", v0.vez);
		OhmsCoeffs Ohms = p_Ohms[iMinor];

		f64 vez_1 = v0.vez + Ohms.sigma_e_zz * Ez_strength;
		f64 viz_1 = v0.viz + Ohms.sigma_i_zz * Ez_strength;

		if (((TESTTRI)) && (0)) printf("vez_1 with Ezcontrib %1.14E sigma_e_zz %1.14E Ez %1.14E vizeffect %1.14E \n", vez_1,
			Ohms.sigma_e_zz, Ez_strength, Ohms.sigma_i_zz * Ez_strength);

		// Cancelled Lap contrib from vez1 here.
		// Be sure we know that makes sense. Is that what we missed on CPU?

		nvals n_use = p_n_minor[iMinor];

		//	AAzdot_k.Azdot +=
		//	  h_use * ROCAzdot_antiadvect + h_use * c*c*(Lap_Az +
		//		0.5*FOURPI_OVER_C * q*n_use.n*(vie_k.viz - vie_k.vez)); // INTERMEDIATE
		//		p_AAdot_intermediate[iMinor] = AAzdot_k; // not k any more
#ifdef MIDPT_A	
		p_Azdot0[iMinor] = p_AAdot_intermediate[iMinor].Azdot
			- 0.5*h_use*c*c*Lap_Az_used // cancel out half what PopOhms did!
										// + h_use * ROCAzdot_antiadvect[iMinor]   // we did this as part of PopOhms.
										// + h_use *c*2.0*PI* q*n_use.n*(v_src.viz - v_src.vez) // we did this as part of PopOhms
			+ h_use *c*2.0*M_PI* q*n_use.n*(viz_1 - vez_1);

		// HALVED:
		f64 viz0_coeff_on_Lap_Az = -0.25*h_use*h_use*qoverM*c / denom_i;
		f64 vez0_coeff_on_Lap_Az = 0.25* h_use*h_use*eoverm*c / denom_e
			+ coeff_of_vez_upon_viz*viz0_coeff_on_Lap_Az;

#else 
		p_Azdot0[iMinor] = p_AAdot_intermediate[iMinor].Azdot
			- h_use*c*c*Lap_Az_used // cancel out what PopOhms did!
									// + h_use * ROCAzdot_antiadvect[iMinor]   // we did this as part of PopOhms.
									// + h_use *c*2.0*PI* q*n_use.n*(v_src.viz - v_src.vez) // we did this as part of PopOhms
			+ h_use *c*2.0*M_PI* q*n_use.n*(viz_1 - vez_1);

		f64 viz0_coeff_on_Lap_Az = -0.5*h_use*h_use*qoverM*c / denom_i;
		f64 vez0_coeff_on_Lap_Az = 0.5* h_use*h_use*eoverm*c / denom_e
			+ coeff_of_vez_upon_viz*viz0_coeff_on_Lap_Az;
#endif

		viz0_coeff_on_Lap_Az += beta_ie_z*vez0_coeff_on_Lap_Az;

		if (((TESTTRI)) && (0)) printf("vez0_coeff_on_Lap undivided %1.14E coeff_viz_on_vez %1.14E viz0_coeff %1.14E denom_e %1.14E\n",
			0.5* h_use*h_use*eoverm*c,
			coeff_of_vez_upon_viz,
			viz0_coeff_on_Lap_Az,
			denom_e
		);
#ifdef MIDPT_A
		p_gamma[iMinor] = h_use*c*c*(0.5 + 0.5*FOURPI_OVER_C * q*n_use.n*
			(viz0_coeff_on_Lap_Az - vez0_coeff_on_Lap_Az));
#else
		p_gamma[iMinor] = h_use*c*c*(1.0 + 0.5*FOURPI_OVER_C * q*n_use.n*
			(viz0_coeff_on_Lap_Az - vez0_coeff_on_Lap_Az));
#endif

		// This represents the effect on Azdot of LapAz. 
		// Did we get this wrong for CPU also?

		if (((TESTTRI)) && (0)) {
			printf("kernelCLR %d: Azdot_intermed %1.14E Lap_Az_used %1.14E Lapcontrib cancel %1.14E Azdot0 %1.14E\n",
				CHOSEN, p_AAdot_intermediate[iMinor].Azdot, Lap_Az_used,
				-h_use*c*c*Lap_Az_used,
				p_Azdot0[iMinor]);
			printf("Jcontrib1 %1.14E viz1 %1.14E vez1 %1.14E\n",
				h_use *c*2.0*M_PI* q*n_use.n*(viz_1 - vez_1),
				viz_1, vez_1);
			printf("gamma %1.14E components: n %1.14E viz0coeff %1.14E vez0coeff %1.14E",
				p_gamma[iMinor],
				n_use.n, viz0_coeff_on_Lap_Az, vez0_coeff_on_Lap_Az);

		}
	}
	else {
		// In PopOhms:
		// AAdot temp = p_AAdot_src[iMinor];
		// temp.Azdot += h_use * c*(c*p_LapAz[iMinor] 
		// NO: + 4.0*PI*Jz);
		// p_AAdot_intermediate[iMinor] = temp; // 

		// We need to do the same sort of thing here as in CalcVelocityAzdot :

		f64 Jz = 0.0;
		
		//if ((iMinor >= numStartZCurrentTriangles) && (iMinor < numEndZCurrentTriangles))
		if (info.flag == REVERSE_JZ_TRI)
		{
			// We should find a way to set these to exactly what we need for it to work,
			// at TriMesh::Initialise and then propagated through the Invoke function.

			f64 AreaMinor = p_AreaMinor[iMinor];
			Jz = negative_Iz_per_triangle / AreaMinor;
		}

#ifdef MIDPT_A
		p_Azdot0[iMinor] = p_AAdot_intermediate[iMinor].Azdot - h_use*0.5*c*c*Lap_Az_used
			+ h_use*c*FOUR_PI*Jz;
		p_gamma[iMinor] = h_use*0.5 * c*c;
#else
		p_Azdot0[iMinor] = p_AAdot_intermediate[iMinor].Azdot - h_use*c*c*Lap_Az_used
			+ h_use*c*FOUR_PI*Jz;
		p_gamma[iMinor] = h_use * c*c;
#endif	

		if ((info.flag == INNER_FRILL) || (info.flag == OUTER_FRILL))
		{
			p_Azdot0[iMinor] = 0.0; // difference found? But we did set = 0 on CPU.
			p_gamma[iMinor] = 0.0;
		}

		if (((TESTTRI)) && (0)) printf("kernelCLR %d: Azdot_intermed %1.14E Lap_Az_used %1.14E Azdot0 %1.14E\n",
			CHOSEN, p_AAdot_intermediate[iMinor].Azdot, Lap_Az_used, p_Azdot0[iMinor]);
		// Note that for frills these will simply not be used.
	};
}

__global__ void kernelCreateLinearRelationshipBwd(
	f64 const h_use,
	structural * __restrict__ p_info,
	OhmsCoeffs* __restrict__ p_Ohms,
	v4 * __restrict__ p_v0,
	f64 * __restrict__ p_Lap_Az_use,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_denom_e,
	f64 * __restrict__ p_denom_i,
	f64 * __restrict__ p_coeff_of_vez_upon_viz,
	f64 * __restrict__ p_beta_ie_z,
	AAdot * __restrict__ p_AAdot_k,
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ p_Azdot0,
	f64 * __restrict__ p_gamma,
	f64 * __restrict__ ROCAzdotduetoAdvection
)
{
	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	f64 const Lap_Az_used = p_Lap_Az_use[iMinor];
	structural const info = p_info[iMinor];
	
	if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST))
	{
		v4 v0 = p_v0[iMinor];

		// Cancel the part that was added in order to get at Ez_strength:
		
		f64 denom_e = p_denom_e[iMinor];
		f64 denom_i = p_denom_i[iMinor];
		
		v0.viz += qoverM*h_use*h_use* c* Lap_Az_used / denom_i; // adaptation for this.

		f64 coeff_of_vez_upon_viz = p_coeff_of_vez_upon_viz[iMinor];
		
		f64 cancel_from_vez = -eoverm*h_use*h_use* c* Lap_Az_used / denom_e
						+ coeff_of_vez_upon_viz * qoverM*h_use*h_use* c* Lap_Az_used / denom_i;

		v0.vez += cancel_from_vez;
		f64 beta_ie_z = p_beta_ie_z[iMinor];
		v0.viz += beta_ie_z * cancel_from_vez;
		
		// We see now that re-jigging things is absolutely not what we should have done.

		// It will make the most complicated overspilling routine, more complicated still.
		
		OhmsCoeffs Ohms = p_Ohms[iMinor];
		
		f64 vez_1 = v0.vez + Ohms.sigma_e_zz * Ez_strength;
		f64 viz_1 = v0.viz + Ohms.sigma_i_zz * Ez_strength;
		
		nvals n_use = p_n_minor[iMinor];		
		p_Azdot0[iMinor] = p_AAdot_k[iMinor].Azdot
						+ h_use * ROCAzdotduetoAdvection[iMinor] // our prediction contains this
						+ h_use *c*4.0*M_PI* q*n_use.n*(viz_1 - vez_1);
		
		// ROCAzdot_antiadvect --- we need this to be in there only

		// on cycles that we do advection
		
		// So do the addition in here.
		
		f64 viz0_coeff_on_Lap_Az = h_use*h_use*qoverM*c / denom_i;

		f64 vez0_coeff_on_Lap_Az = h_use*h_use*eoverm*c / denom_e
					+ coeff_of_vez_upon_viz*viz0_coeff_on_Lap_Az;
		
		viz0_coeff_on_Lap_Az += beta_ie_z*vez0_coeff_on_Lap_Az;

		p_gamma[iMinor] = h_use*c*c*(1.0 + FOURPI_OVER_C * q*n_use.n*
					(viz0_coeff_on_Lap_Az - vez0_coeff_on_Lap_Az));
		
	} else {

		// We need to do the same sort of thing here as in CalcVelocityAzdot :
		
		f64 Jz = 0.0;
		
		//if ((iMinor >= numStartZCurrentTriangles) && (iMinor < numEndZCurrentTriangles))

		if (info.flag == REVERSE_JZ_TRI)
		{
			f64 AreaMinor = p_AreaMinor[iMinor];
			Jz = negative_Iz_per_triangle / AreaMinor;
		}

		p_Azdot0[iMinor] = p_AAdot_k[iMinor].Azdot + h_use*c*FOUR_PI*Jz;

		p_gamma[iMinor] = h_use * c*c;

		
		if ((info.flag == INNER_FRILL) || (info.flag == OUTER_FRILL))

		{
			p_Azdot0[iMinor] = 0.0; // difference found? But we did set = 0 on CPU.
			p_gamma[iMinor] = 0.0;
		}
	};
}


__global__ void kernelCreateLinearRelationshipBwd_noadvect(
	f64 const h_use,
	structural * __restrict__ p_info,
	OhmsCoeffs* __restrict__ p_Ohms,
	v4 * __restrict__ p_v0,
	f64 * __restrict__ p_Lap_Az_use,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_denom_e, 
	f64 * __restrict__ p_denom_i,
	f64 * __restrict__ p_coeff_of_vez_upon_viz, 
	f64 * __restrict__ p_beta_ie_z,
	AAdot * __restrict__ p_AAdot_k,
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ p_Azdot0,
	f64 * __restrict__ p_gamma
)
{
	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	f64 const Lap_Az_used = p_Lap_Az_use[iMinor];
	structural const info = p_info[iMinor];

	if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST))
	{
		v4 v0 = p_v0[iMinor];
		// Cancel the part that was added in order to get at Ez_strength:
		
		f64 denom_e = p_denom_e[iMinor];
		f64 denom_i = p_denom_i[iMinor];

		v0.viz += qoverM*h_use*h_use* c* Lap_Az_used/denom_i; // adaptation for this.
		f64 coeff_of_vez_upon_viz = p_coeff_of_vez_upon_viz[iMinor];
		
		f64 cancel_from_vez = -eoverm*h_use*h_use* c* Lap_Az_used / denom_e
			+ coeff_of_vez_upon_viz * qoverM*h_use*h_use* c* Lap_Az_used / denom_i;
		
		v0.vez += cancel_from_vez;
		f64 beta_ie_z = p_beta_ie_z[iMinor];
		v0.viz += beta_ie_z * cancel_from_vez;

		// We see now that re-jigging things is absolutely not what we should have done.
		// It will make the most complicated overspilling routine, more complicated still.
		
		OhmsCoeffs Ohms = p_Ohms[iMinor];

		f64 vez_1 = v0.vez + Ohms.sigma_e_zz * Ez_strength;
		f64 viz_1 = v0.viz + Ohms.sigma_i_zz * Ez_strength;

		nvals n_use = p_n_minor[iMinor];

		p_Azdot0[iMinor] = p_AAdot_k[iMinor].Azdot
				+ h_use *c*4.0*M_PI* q*n_use.n*(viz_1 - vez_1);

		// ROCAzdot_antiadvect --- we need this to be in there only
		// on cycles that we do advection

		// So do the addition in here.

		f64 viz0_coeff_on_Lap_Az = h_use*h_use*qoverM*c / denom_i;
		f64 vez0_coeff_on_Lap_Az = h_use*h_use*eoverm*c / denom_e
			+ coeff_of_vez_upon_viz*viz0_coeff_on_Lap_Az;

		viz0_coeff_on_Lap_Az += beta_ie_z*vez0_coeff_on_Lap_Az;

		p_gamma[iMinor] = h_use*c*c*(1.0 + FOURPI_OVER_C * q*n_use.n*
			(viz0_coeff_on_Lap_Az - vez0_coeff_on_Lap_Az));

	} else {
		// We need to do the same sort of thing here as in CalcVelocityAzdot :

		f64 Jz = 0.0;
		
		//if ((iMinor >= numStartZCurrentTriangles) && (iMinor < numEndZCurrentTriangles))
		if (info.flag == REVERSE_JZ_TRI)
		{
			f64 AreaMinor = p_AreaMinor[iMinor];
			Jz = negative_Iz_per_triangle / AreaMinor;			
		}

		p_Azdot0[iMinor] = p_AAdot_k[iMinor].Azdot 
			+ h_use*c*FOUR_PI*Jz;
		p_gamma[iMinor] = h_use * c*c;

		if ((info.flag == INNER_FRILL) || (info.flag == OUTER_FRILL))
		{
			p_Azdot0[iMinor] = 0.0; // difference found? But we did set = 0 on CPU.
			p_gamma[iMinor] = 0.0;			
		}
	};
}

/*
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

	nvals * __restrict__ p_one_over_n,

	T3 * __restrict__ p_T_minor_use,

	v4 * __restrict__ p_vie_src,
	f64_vec3 * __restrict__ p_v_n_src,
	AAdot * __restrict__ p_AAdot_src,
	f64 * __restrict__ p_AreaMinor,

	f64 * __restrict__ ROCAzdotduetoAdvection,
	// Now going to need to go through and see this set 0 or sensible every time.

	f64_vec3 * __restrict__ p_vn0_dest,
	v4 * __restrict__ p_v0_dest,
	OhmsCoeffs * __restrict__ p_OhmsCoeffs_dest,
	AAdot * __restrict__ p_AAdot_intermediate,

	f64 * __restrict__ p_Iz0,
	f64 * __restrict__ p_sigma_zz,
	
	f64 * __restrict__ p_denom_i,
	f64 * __restrict__ p_denom_e, 
	f64 * __restrict__ p_effect_of_viz0_on_vez0,
	f64 * __restrict__ p_beta_ie_z,

	bool const bSwitchSave,
	bool const bUse_dest_n_for_Iz,
	nvals * __restrict__ p_n_dest_minor) // for turning on save of these denom_ quantities
{
	// Don't forget we can use 16KB shared memory to save a bit of overspill:
	// (16*1024)/(512*8) = 4 doubles only for 512 threads. 128K total register space per SM we think.

	__shared__ f64 Iz[threadsPerTileMinor], sigma_zz[threadsPerTileMinor];
//	__shared__ f64 Iz_k[threadsPerTileMinor];

	__shared__ f64_vec2 omega[threadsPerTileMinor], grad_Az[threadsPerTileMinor],
		gradTe[threadsPerTileMinor];
	
	// Putting 8 reduces to 256 simultaneous threads. Experiment with 4 in shared.
	// f64 viz0_coeff_on_Lap_Az, vez0_coeff_on_Lap_Az; // THESE APPLY TO FEINT VERSION. ASSUME NOT FEINT FIRST.

	v4 v0;
	f64 denom, ROCAzdot_antiadvect, AreaMinor;
	f64_vec3 vn0;
	long const iMinor = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX
	structural info = p_info_minor[iMinor];

	// Can see no reason not to put OUTERMOST here. No point creating a big gradient of vz to it.

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE)
		|| (info.flag == CROSSING_INS) || (info.flag == OUTERMOST))
	{
		v4 vie_k = p_vie_src[iMinor];
		f64_vec3 v_n_src = p_v_n_src[iMinor];
		nvals n_use = p_n_minor_use[iMinor];
		AreaMinor = p_AreaMinor[iMinor];
		// Are we better off with operator = or with memcpy?
		vn0 = v_n_src;

//		if ((TESTTRI)) printf("GPU %d vie_k %1.14E %1.14E\n", iMinor, vie_k.vxy.x, vie_k.vxy.y);
		{
			f64_vec3 MAR;
			memcpy(&MAR, p_MAR_neut + iMinor, sizeof(f64_vec3));
			// CHECK IT IS INTENDED TO AFFECT Nv

			// REVERTED THE EDIT TO USE 1/n -- THIS WILL NOT GIVE CORRECT M.A.R. EFFECT ON INTEGRAL nv
			// We need conservation laws around shock fronts.
			vn0.x += h_use * (MAR.x / (AreaMinor*n_use.n_n));								
				// p_one_over_n[iMinor].n_n/ (AreaMinor));
			vn0.y += h_use * (MAR.y/(AreaMinor*n_use.n_n));// MomAddRate is addition rate for Nv. Divide by N.

			memcpy(&MAR, p_MAR_ion + iMinor, sizeof(f64_vec3));
			v0.vxy = vie_k.vxy + h_use * (m_i*MAR.xypart()/ (n_use.n*(m_i + m_e)*AreaMinor));
			v0.viz = vie_k.viz + h_use * MAR.z / (n_use.n*AreaMinor);

			if (((TESTTRI))) printf("\nGPU %d vxyk %1.10E %1.10E aMAR_i.y %1.10E MAR.y %1.10E 1/n %1.10E Area %1.10E\n", iMinor, 
				v0.vxy.x, v0.vxy.y,
				h_use * (m_i*MAR.y / (n_use.n*(m_i + m_e)*AreaMinor)),
				MAR.y,
				p_one_over_n[iMinor].n,
				AreaMinor);
			
			memcpy(&MAR, p_MAR_elec + iMinor, sizeof(f64_vec3));
			v0.vxy += h_use * (m_e*MAR.xypart() / (n_use.n*(m_i + m_e)*AreaMinor));
			v0.vez = vie_k.vez + h_use * MAR.z / (n_use.n*AreaMinor);   // UM WHY WAS THIS NEGATIVE
													 // + !!!!
			if (v0.vez != v0.vez) printf("NANVEZ %d v_k %1.9E MAR.z %1.9E \n", iMinor, vie_k.vez, MAR.z);

			if (((TESTTRI))) printf("\nGPU %d a:MAR_e %1.10E %1.10E MAR.y %1.10E 1/n %1.10E Area %1.10E\n", iMinor,
				h_use * (m_e*MAR.x/ (n_use.n*(m_i + m_e)*AreaMinor)),
				h_use * (m_e*MAR.y/ (n_use.n*(m_i + m_e)*AreaMinor)),
				MAR.y,
				p_one_over_n[iMinor].n, AreaMinor);

	//		if (((TESTTRI))) 
		//		printf("GPU %d WITH MAR v0.vxy %1.14E %1.14E\n", CHOSEN, v0.vxy.x, v0.vxy.y);
				//	printf("GPU %d data_k %1.10E %1.10E MAR %1.10E %1.10E\n", CHOSEN, vie_k.vxy.x, vie_k.vxy.y,
					//	MAR.x, MAR.y);
//				printf("GPU %d n %1.12E AreaMinor %1.12E \n", CHOSEN, n_use.n, AreaMinor);
	//		}
		}

		OhmsCoeffs ohm;
		f64 beta_ie_z, LapAz;
		f64 cross_section_times_thermal_en, cross_section_times_thermal_in,
			nu_eiBar, nu_eHeart;
		T3 T = p_T_minor_use[iMinor];
		{
			// Dimensioning inside a brace allows the following vars to go out of scope at the end of the brace.
			f64 sqrt_Te, ionneut_thermal, electron_thermal,
				lnLambda, s_in_MT, s_en_MT, s_en_visc;
			sqrt_Te = sqrt(T.Te);
			ionneut_thermal = sqrt(T.Ti / m_ion + T.Tn / m_n); // hopefully not sqrt(0)
			electron_thermal = sqrt_Te * over_sqrt_m_e;
			lnLambda = Get_lnLambda_d(n_use.n, T.Te);
			{
				f64 s_in_visc_dummy;
				Estimate_Ion_Neutral_Cross_sections_d(T.Ti*one_over_kB, &s_in_MT, &s_in_visc_dummy);
			}
			Estimate_Ion_Neutral_Cross_sections_d(T.Te*one_over_kB, &s_en_MT, &s_en_visc);

			//nu_ne_MT = s_en_MT * electron_thermal * n_use.n; // have to multiply by n_e for nu_ne_MT
			//nu_ni_MT = s_in_MT * ionneut_thermal * n_use.n;
			//nu_in_MT = s_in_MT * ionneut_thermal * n_use.n_n;
			//nu_en_MT = s_en_MT * electron_thermal * n_use.n_n;

			cross_section_times_thermal_en = s_en_MT * electron_thermal;
			cross_section_times_thermal_in = s_in_MT * ionneut_thermal;
			
			nu_eiBar = nu_eiBarconst * kB_to_3halves*n_use.n*lnLambda / (T.Te*sqrt_Te);
			nu_eHeart = 1.87*nu_eiBar + n_use.n_n*s_en_visc*electron_thermal;
			if (nu_eiBar != nu_eiBar) printf("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n"
				"iMinor %d n_use.n %1.9E lnLambda %1.9E Te %1.9E sqrt %1.9E \n",
				iMinor, n_use.n, lnLambda, T.Te, sqrt_Te);

			// ARTIFICIAL CHANGE TO STOP IONS SMEARING AWAY OFF OF NEUTRAL BACKGROUND:
			if (n_use.n_n > ARTIFICIAL_RELATIVE_THRESH *n_use.n) {
				cross_section_times_thermal_en *= n_use.n_n / (ARTIFICIAL_RELATIVE_THRESH *n_use.n);
				cross_section_times_thermal_in *= n_use.n_n / (ARTIFICIAL_RELATIVE_THRESH *n_use.n);
				// So at 1e18 vs 1e8 it's 10 times stronger
				// At 1e18 vs 1e6 it's 1000 times stronger
				// nu starts at about 1e11 at the place it failed at 35ns. So 10000 times stronger gives us 1e15.
			}

		}

		vn0.x += -0.5*h_use*(M_e_over_en)*(cross_section_times_thermal_en*n_use.n)*(v_n_src.x - vie_k.vxy.x)
			- 0.5*h_use*(M_i_over_in)*(cross_section_times_thermal_in*n_use.n)*(v_n_src.x - vie_k.vxy.x);
		vn0.y += -0.5*h_use*(M_e_over_en)*(cross_section_times_thermal_en*n_use.n)*(v_n_src.y - vie_k.vxy.y)
			- 0.5*h_use*(M_i_over_in)*(cross_section_times_thermal_in*n_use.n)*(v_n_src.y - vie_k.vxy.y);
		vn0.z += -0.5*h_use*(M_e_over_en)*(cross_section_times_thermal_en*n_use.n)*(v_n_src.z - vie_k.vez)
			- 0.5*h_use*(M_i_over_in)*(cross_section_times_thermal_in*n_use.n)*(v_n_src.z - vie_k.viz);
		denom = 1.0 + h_use * 0.5*M_e_over_en* (cross_section_times_thermal_en*n_use.n)
			+ 0.5*h_use*M_i_over_in* (cross_section_times_thermal_in*n_use.n);

		vn0 /= denom; // It is now the REDUCED value

		if (((TESTTRI))) 
			printf("GPU %d vn0 %1.9E %1.9E %1.9E denom %1.14E \n", CHOSEN, vn0.x, vn0.y, vn0.z, denom);
	

		ohm.beta_ne = 0.5*h_use*(M_e_over_en)*(cross_section_times_thermal_en*n_use.n) / denom;
		ohm.beta_ni = 0.5*h_use*(M_i_over_in)*(cross_section_times_thermal_in*n_use.n) / denom;

		// Now we do vexy:

		grad_Az[threadIdx.x] = p_GradAz[iMinor];
		gradTe[threadIdx.x] = p_GradTe[iMinor];
		LapAz = p_LapAz[iMinor];
		f64 ROCAzdot_antiadvect = ROCAzdotduetoAdvection[iMinor];
		
		if (((TESTTRI))) printf("GPU %d: LapAz %1.14E\n", CHOSEN, LapAz);

		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		// Here is where we should be using v_use:
		// We do midpoint instead? Why not? Thus allowing us not to load v_use.
		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		v0.vxy +=
			-h_use * (q / (2.0*c*(m_i + m_e)))*(vie_k.vez - vie_k.viz)*grad_Az[threadIdx.x]

			- (h_use / (2.0*(m_i + m_e)))*(m_n*M_i_over_in*(cross_section_times_thermal_in*n_use.n_n)
				+ m_n * M_e_over_en*(cross_section_times_thermal_en*n_use.n_n))*
				(vie_k.vxy - v_n_src.xypart() - vn0.xypart());


		if (((TESTTRI))) printf("GPU %d vzgradAz contrib_k %1.10E %1.10E vez_k viz_k %1.9E %1.9E gradAz %1.9E %1.9E\n", iMinor, 
			-h_use * (q / (2.0*c*(m_i + m_e)))*(vie_k.vez - vie_k.viz)*grad_Az[threadIdx.x].x,
			-h_use * (q / (2.0*c*(m_i + m_e)))*(vie_k.vez - vie_k.viz)*grad_Az[threadIdx.x].y, vie_k.vez, vie_k.viz,
			grad_Az[threadIdx.x].x, grad_Az[threadIdx.x].y);


		denom = 1.0 + (h_use / (2.0*(m_i + m_e)))*(
			m_n* M_i_over_in* (cross_section_times_thermal_in*n_use.n_n)
			+ m_n * M_e_over_en*(cross_section_times_thermal_en*n_use.n_n))*(1.0 - ohm.beta_ne - ohm.beta_ni);
		v0.vxy /= denom;
//
		if (((TESTTRI))) 
			printf("GPU %d v0.vxy %1.14E %1.14E denom %1.14E \n"
				"nu_in_MT %1.14E nu_en_MT %1.14E beta_ne %1.14E \n", 
				CHOSEN, v0.vxy.x, v0.vxy.y, denom,
				cross_section_times_thermal_in*n_use.n_n, cross_section_times_thermal_en*n_use.n_n, ohm.beta_ne);
			
		ohm.beta_xy_z = (h_use * q / (2.0*c*(m_i + m_e)*denom)) * grad_Az[threadIdx.x];
		/////////////////////////////////////////////////////////////////////////////// midpoint
//		if (((TESTTRI))) printf("ohm.beta_xy_z %1.14E \n", ohm.beta_xy_z);

		omega[threadIdx.x] = qovermc*p_B[iMinor].xypart();

		f64 nu_ei_effective = nu_eiBar * (1.0 - 0.9*nu_eiBar*(nu_eHeart*nu_eHeart + qovermc*BZ_CONSTANT*qovermc*BZ_CONSTANT) /
			(nu_eHeart*(nu_eHeart*nu_eHeart + omega[threadIdx.x].x*omega[threadIdx.x].x + omega[threadIdx.x].y*omega[threadIdx.x].y + qovermc*BZ_CONSTANT*qovermc*BZ_CONSTANT)));

	//	if (nu_ei_effective != nu_ei_effective) printf("nu_ei NaN: omega %1.8E %1.8E nu_eHeart %1.8E nu_eiBar %1.8E\n",
	//		omega[threadIdx.x].x, omega[threadIdx.x].y, nu_eHeart, nu_eiBar);

		AAdot AAzdot_k = p_AAdot_src[iMinor];

		//if ((iPass == 0) || (bFeint == false))
		{
	//		if (((TESTTRI)) && (0)) printf("viz0: %1.14E\n", v0.viz);
			if (((TESTTRI))) printf("GPU %d: LapAz %1.14E\n", CHOSEN, LapAz); // nonzero
			v0.viz +=
				-0.5*h_use*qoverMc*(2.0*AAzdot_k.Azdot
					+ h_use * ROCAzdot_antiadvect + h_use * c*c*(LapAz
						+ FOURPI_OVER_C*0.5 * q*n_use.n*(vie_k.viz - vie_k.vez)))
				- 0.5*h_use*qoverMc*(vie_k.vxy + v0.vxy).dot(grad_Az[threadIdx.x]);

			if (((TESTTRI))) {
				printf("viz0 I: %1.14E contribs:\n", v0.viz);
				printf("   Azdotk %1.14E \n   ROC %1.14E\n   JviaAzdot %1.14E\n   lorenzmag %1.14E\n",
					-0.5*h_use*qoverMc*(2.0*AAzdot_k.Azdot),
					-0.5*h_use*qoverMc*h_use * ROCAzdot_antiadvect,
					-0.5*h_use*qoverMc*h_use * c*c*(FOURPI_OVER_C*0.5 * q*n_use.n*(vie_k.viz - vie_k.vez)),
					-0.5*h_use*qoverMc*(vie_k.vxy + v0.vxy).dot(grad_Az[threadIdx.x])
				);
				printf("due to LapAz: %1.14E = %1.6E %1.6E %1.6E %1.6E\n",
					-0.5*h_use*qoverMc*h_use *c*c*LapAz,
					h_use*h_use*0.5,
					qoverMc,
					c*c,
					LapAz); // == 0
			};

		}
		//else {
		//	viz0 = data_k.viz
		//				- h_use * MomAddRate.ion.z / (data_use.n*AreaMinor)
		//				- 0.5*h_use*qoverMc*(2.0*data_k.Azdot
		//				+ h_use * ROCAzdot_antiadvect + h_use * c*c*(TWOPIoverc * q*data_use.n*(data_k.viz - data_k.vez)))
		//				- 0.5*h_use*qoverMc*(data_k.vxy + vxy0).dot(grad_Az[threadIdx.x]);
		//	};

		//
		// Still omega_ce . Check formulas.
		// 

		v0.viz +=
			1.5*h_use*nu_eiBar*((omega[threadIdx.x].x*qovermc*BZ_CONSTANT - nu_eHeart * omega[threadIdx.x].y)*gradTe[threadIdx.x].x +
			(omega[threadIdx.x].y*qovermc*BZ_CONSTANT + nu_eHeart * omega[threadIdx.x].x)*gradTe[threadIdx.x].y) /
				(m_i*nu_eHeart*(nu_eHeart*nu_eHeart + omega[threadIdx.x].dot(omega[threadIdx.x])));

	//	if (((TESTTRI))) printf("viz0 with thermal force %1.14E \n", v0.viz);

		v0.viz += -h_use * 0.5*M_n_over_ni*(cross_section_times_thermal_in*n_use.n_n) *(vie_k.viz - v_n_src.z - vn0.z) // THIS DOESN'T LOOK RIGHT
			+ h_use * 0.5*(moverM)*nu_ei_effective*(vie_k.vez - vie_k.viz);

		if (((TESTTRI))) printf("viz0 contrib i-n %1.14E contrib e-i %1.14E\nviz0 %1.14E\n",
			-h_use * 0.5*M_n_over_ni*(cross_section_times_thermal_in*n_use.n_n) *(vie_k.viz - v_n_src.z - vn0.z),
			h_use * 0.5*(moverM)*nu_ei_effective*(vie_k.vez - vie_k.viz), v0.viz
			);

		denom = 1.0 + h_use * h_use*M_PI*qoverM*q*n_use.n + h_use * 0.5*qoverMc*(grad_Az[threadIdx.x].dot(ohm.beta_xy_z)) +
			h_use * 0.5*M_n_over_ni*(cross_section_times_thermal_in*n_use.n_n) *(1.0 - ohm.beta_ni) + h_use * 0.5*moverM*nu_ei_effective;

		if (bSwitchSave) p_denom_i[iMinor] = denom;
		//				viz0_coeff_on_Lap_Az = -0.5*h_use*qoverMc*h_use*c*c / denom;

		v0.viz /= denom;

		if (((TESTTRI))) printf("viz0 divided %1.14E denom %1.14E \n", v0.viz, denom);

		ohm.sigma_i_zz = h_use * qoverM / denom;
		beta_ie_z = (h_use*h_use*M_PI*qoverM*q*n_use.n
			+ 0.5*h_use*qoverMc*(grad_Az[threadIdx.x].dot(ohm.beta_xy_z))
			+ h_use * 0.5*M_n_over_ni*(cross_section_times_thermal_in*n_use.n_n) *ohm.beta_ne
			+ h_use * 0.5*moverM*nu_ei_effective) / denom;

		if (((TESTTRI2))) printf("vez0 %1.14E \n", v0.vez);

		v0.vez +=
			h_use * 0.5*qovermc*(2.0*AAzdot_k.Azdot
				+ h_use * ROCAzdot_antiadvect
				+ h_use * c*c*(LapAz
					+ 0.5*FOURPI_Q_OVER_C*n_use.n*(vie_k.viz + v0.viz - vie_k.vez))) // ?????????????????
			+ 0.5*h_use*qovermc*(vie_k.vxy + v0.vxy + v0.viz * ohm.beta_xy_z).dot(grad_Az[threadIdx.x]);


		if (((TESTTRI2))) 
			printf(" %d v0.vez %1.14E Azdotctb %1.14E antiadvect %1.14E LapAzctb %1.14E \n"
				"%d JviaAzdot %1.14E lorenzmag %1.14E \n",
				iMinor, v0.vez, h_use * 0.5*qovermc*2.0*AAzdot_k.Azdot,
				h_use * 0.5*qovermc*h_use * ROCAzdot_antiadvect,
				h_use * 0.5*qovermc*h_use * c*c*LapAz,
				iMinor,
				h_use * 0.5*qovermc*h_use * c*c* 0.5*FOURPI_Q_OVER_C*n_use.n*(vie_k.viz + v0.viz - vie_k.vez),
				0.5*h_use*qovermc*(vie_k.vxy + v0.vxy + v0.viz * ohm.beta_xy_z).dot(grad_Az[threadIdx.x])
				);		

		// implies:
		f64 effect_of_viz0_on_vez0 = 			
			h_use * 0.5*qovermc*h_use * c*c*0.5*FOURPI_Q_OVER_C*n_use.n			
			+ 0.5*h_use*qovermc*( ohm.beta_xy_z.dot(grad_Az[threadIdx.x]));
		
		v0.vez -=
			1.5*h_use*nu_eiBar*((omega[threadIdx.x].x*qovermc*BZ_CONSTANT - nu_eHeart * omega[threadIdx.x].y)*gradTe[threadIdx.x].x +
			(omega[threadIdx.x].y*qovermc*BZ_CONSTANT + nu_eHeart * omega[threadIdx.x].x)*gradTe[threadIdx.x].y) /
				(m_e*nu_eHeart*(nu_eHeart*nu_eHeart + omega[threadIdx.x].dot(omega[threadIdx.x])+ qovermc*BZ_CONSTANT*qovermc*BZ_CONSTANT));

		if (((TESTTRI2)))
			printf("%d v0.vez TF contrib : %1.14E nu_eiBar %1.14E nu_eHeart %1.14E \n"
				"%d omega %1.14E %1.14E %1.14E\n",iMinor,

				-1.5*h_use*nu_eiBar*((omega[threadIdx.x].x*qovermc*BZ_CONSTANT - nu_eHeart * omega[threadIdx.x].y)*gradTe[threadIdx.x].x +
				(omega[threadIdx.x].y*qovermc*BZ_CONSTANT + nu_eHeart * omega[threadIdx.x].x)*gradTe[threadIdx.x].y) /
					(m_e*nu_eHeart*(nu_eHeart*nu_eHeart + omega[threadIdx.x].dot(omega[threadIdx.x]) + qovermc*BZ_CONSTANT*qovermc*BZ_CONSTANT)),
				
				nu_eiBar, nu_eHeart, iMinor,
				omega[threadIdx.x].x, omega[threadIdx.x].y, qovermc*BZ_CONSTANT);
			
		// could store this from above and put opposite -- dividing by m_e instead of m_i

		v0.vez += -0.5*h_use*M_n_over_ne*(cross_section_times_thermal_en*n_use.n_n) *(vie_k.vez - v_n_src.z - vn0.z - ohm.beta_ni * v0.viz)
			- 0.5*h_use*nu_ei_effective*(vie_k.vez - vie_k.viz - v0.viz);
		// implies:
		effect_of_viz0_on_vez0 += 
			0.5*h_use*M_n_over_ne*(cross_section_times_thermal_en*n_use.n_n) *ohm.beta_ni + 0.5*h_use*nu_ei_effective;
		


		if (
			//(iMinor == 11761 + BEGINNING_OF_CENTRAL) ||
			//(iMinor == 11616 + BEGINNING_OF_CENTRAL) ||
			//(iMinor == 11762 + BEGINNING_OF_CENTRAL) ||
			((TESTTRI2)) )
		{
			printf("%d cross_section_times_thermal_en %1.10E n_use.n_n %1.10E vezk %1.10E vez0 %1.10E Mnoverne %1.10E nu_ei_effective %1.10E \n",
				iMinor, cross_section_times_thermal_en, n_use.n_n,
				vie_k.vez, v0.vez,
				M_n_over_ne, nu_ei_effective);
		}
		
		if (((TESTTRI2))) 
			printf("v0.vez contribs e-n e-i: %1.14E %1.14E v0.viz %1.14E\n", 
				-0.5*h_use*M_n_over_ne*(cross_section_times_thermal_en*n_use.n_n) *(vie_k.vez - v_n_src.z - vn0.z - ohm.beta_ni * v0.viz),
				- 0.5*h_use*nu_ei_effective*(vie_k.vez - vie_k.viz - v0.viz),
				v0.viz);

		denom = 1.0 + (h_use*h_use*M_PI*q*eoverm*n_use.n
			+ 0.5*h_use*qovermc*(grad_Az[threadIdx.x].dot(ohm.beta_xy_z)))*(1.0 - beta_ie_z)
			+ 0.5*h_use*M_n_over_ne*(cross_section_times_thermal_en*n_use.n_n) *(1.0 - ohm.beta_ne - ohm.beta_ni * beta_ie_z)
			+ 0.5*h_use*nu_ei_effective*(1.0 - beta_ie_z);

		//		vez0_coeff_on_Lap_Az = h_use * h_use*0.5*qovermc* c*c / denom; 

		ohm.sigma_e_zz = 			
			(-h_use * eoverm
			+ h_use * h_use*M_PI*q*eoverm*n_use.n*ohm.sigma_i_zz
			+ h_use * 0.5*qovermc*(grad_Az[threadIdx.x].dot(ohm.beta_xy_z))*ohm.sigma_i_zz
			+ 0.5*h_use*M_n_over_ne*(cross_section_times_thermal_en*n_use.n_n) *ohm.beta_ni*ohm.sigma_i_zz
			+ 0.5*h_use*nu_ei_effective*ohm.sigma_i_zz)
			/ denom;
		
	//	if (((TESTTRI)1) || ((TESTTRI)2))
//printf("GPU %d vez0 before divide %1.14E \n", iMinor, v0.vez);
//
		v0.vez /= denom;
		effect_of_viz0_on_vez0 /= denom; // of course 

		//if (v0.vez != v0.vez) {
		//	printf("iMinor %d v0.vez %1.10E ohm.sigma_e %1.10E denom %1.10E \n"
		//		"%1.10E %1.10E %1.10E %1.10E n %1.10E Te %1.10E\n"	,
		//		iMinor, v0.vez, ohm.sigma_e_zz, denom,
		//		h_use*h_use*M_PI*q*eoverm*n_use.n,//*(1.0 - beta_ie_z) // this was ok
		//		0.5*h_use*qovermc*(grad_Az[threadIdx.x].dot(ohm.beta_xy_z))*(1.0 - beta_ie_z), // this was not ok
		//		0.5*h_use*M_n_over_ne*(cross_section_times_thermal_en*n_use.n_n) *(1.0 - ohm.beta_ne - ohm.beta_ni * beta_ie_z),
		//		0.5*h_use*nu_ei_effective,//*(1.0 - beta_ie_z) // this was not ok -- even though n,T come out ok
		//		n_use.n, T.Te);			
		//}


		if ( ((TESTTRI2))) 
			printf("GPU %d v0.vez %1.14E denom %1.14E \n"
				"ohm.sigma_e_zz %1.14E n_use %1.10E nn %1.10E Te %1.10E\n"
				"%d %1.12E %1.12E %1.12E %1.12E %1.12E \n"
				"%d denom %1.14E : %1.12E %1.12E %1.12E %1.12E\n",
				iMinor, v0.vez, denom,
				ohm.sigma_e_zz,
				n_use.n,n_use.n_n, T.Te, iMinor, -h_use * eoverm,
				h_use * h_use*M_PI*q*eoverm*n_use.n*ohm.sigma_i_zz,
				h_use * 0.5*qovermc*(grad_Az[threadIdx.x].dot(ohm.beta_xy_z))*ohm.sigma_i_zz,
				0.5*h_use*M_n_over_ne*(cross_section_times_thermal_en*n_use.n_n) *ohm.beta_ni*ohm.sigma_i_zz,
				0.5*h_use*nu_ei_effective*ohm.sigma_i_zz,
				iMinor, denom,
				(h_use*h_use*M_PI*q*eoverm*n_use.n)*(1.0 - beta_ie_z),
				(0.5*h_use*qovermc*(grad_Az[threadIdx.x].dot(ohm.beta_xy_z)))*(1.0 - beta_ie_z),
				0.5*h_use*M_n_over_ne*(cross_section_times_thermal_en*n_use.n_n) *(1.0 - ohm.beta_ne - ohm.beta_ni * beta_ie_z),
				0.5*h_use*nu_ei_effective*(1.0 - beta_ie_z)
			);	

		if (bSwitchSave) {
			p_denom_e[iMinor] = denom;
			p_effect_of_viz0_on_vez0[iMinor] = effect_of_viz0_on_vez0;
			p_beta_ie_z[iMinor] = beta_ie_z; // see that doing it this way was not best.
		} else {
			// #########################################################################################################
			// DEBUG: pass graphing parameters through these.
			// #########################################################################################################
			p_denom_i[iMinor] = M_n_over_ne*cross_section_times_thermal_en*n_use.n_n + nu_ei_effective;
			p_denom_e[iMinor] = M_n_over_ne*cross_section_times_thermal_en*n_use.n_n /
				(M_n_over_ne*cross_section_times_thermal_en*n_use.n_n + nu_ei_effective);
		};
		
		// Now update viz(Ez):
		v0.viz += beta_ie_z * v0.vez;
		ohm.sigma_i_zz += beta_ie_z * ohm.sigma_e_zz;

		// sigma_e_zz and sigma_i_zz are change in vz for a change in Ez
		{
			f64 EzShape = GetEzShape(info.pos.modulus());
			ohm.sigma_i_zz *= EzShape;
			ohm.sigma_e_zz *= EzShape;
		}

		// Think maybe we should get rid of most of this routine out of the subcycle.
		// Rate of acceleration over timestep due to resistance, pressure, thermal force etc could be stored.
		// Saving off some eqn data isn't so bad when we probably overflow registers and L1 here anyway.
		// All we need is to know that we update sigma
		// We can do addition of 
		// ==============================================================================================

		p_v0_dest[iMinor] = v0;
		p_OhmsCoeffs_dest[iMinor] = ohm;
		p_vn0_dest[iMinor] = vn0;

		if (bUse_dest_n_for_Iz) {
			f64 ndest = p_n_dest_minor[iMinor].n;
			Iz[threadIdx.x] = q*AreaMinor*ndest*(v0.viz - v0.vez);
			sigma_zz[threadIdx.x] = q*AreaMinor*ndest*(ohm.sigma_i_zz - ohm.sigma_e_zz);

			if (((TESTTRI2))) {
				printf( "ndest %1.12E sigma_zz/Area %1.12E AreaMinor %1.12E\n\n",
					ndest, q*ndest*(ohm.sigma_i_zz - ohm.sigma_e_zz), AreaMinor);
			}

		} else {
			// On intermediate substeps, the interpolated n that applies halfway through the substep is a reasonable choice...
			Iz[threadIdx.x] = q*AreaMinor*n_use.n*(v0.viz - v0.vez);
			sigma_zz[threadIdx.x] = q*AreaMinor*n_use.n*(ohm.sigma_i_zz - ohm.sigma_e_zz);
			// I'm sure we can do better on this. But we also might prefer to excise a lot of this calc from the subcycle.
			if (((TESTTRI2))) {
				printf("n_use.n %1.12E sigma_zz/Area %1.12E AreaMinor %1.12E\n\n",
					n_use.n, q*n_use.n*(ohm.sigma_i_zz - ohm.sigma_e_zz), AreaMinor);
			}

		}
		
		
		// Totally need to be skipping the load of an extra n.
		// ^^ old remark.
		// But it's too messy never loading it. t_half means changing all the
		// Iz formula to involve v_k. Don't want that.


	//	if (blockIdx.x == 340) printf("%d: %1.14E %1.14E \n",
		//	iMinor, q*n_use.n*(ohm.sigma_i_zz - ohm.sigma_e_zz), sigma_zz[threadIdx.x]);

		// On iPass == 0, we need to do the accumulate.
		//	p_Azdot_intermediate[iMinor] = Azdot_k
		//		+ h_use * ROCAzdot_antiadvect + h_use * c*c*(Lap_Az +
		//			0.5*FOURPI_OVER_C * q*n_use.n*(data_k.viz - data_k.vez)); // INTERMEDIATE

		//if ((0) && ((TESTTRI))) printf("******************* AAzdot_k.Azdot %1.14E \n", AAzdot_k.Azdot);

		AAzdot_k.Azdot +=
			 h_use * ROCAzdot_antiadvect + h_use * c*c*(LapAz +
				0.5*FOURPI_OVER_C * q*n_use.n*(vie_k.viz - vie_k.vez)); // INTERMEDIATE

		p_AAdot_intermediate[iMinor] = AAzdot_k; // not k any more

		//Iz_k[threadIdx.x] = q*n_use.n*(vie_k.viz - vie_k.vez)*AreaMinor;
		
		//if ((0) && ((TESTTRI))) {
		//	printf("\n!!! kernelPopOhms GPU %d: \n******* Azdot_intermediate %1.14E vie_k %1.14E %1.14E\n"
		//		"antiadvect %1.10E Lapcontrib %1.13E Jcontrib_k %1.14E\n\n",
		//		CHOSEN, p_AAdot_intermediate[iMinor].Azdot,
		//		vie_k.viz, vie_k.vez,
		//		h_use * ROCAzdot_antiadvect,
		//		h_use * c*c*LapAz,
		//		h_use * c*c*0.5*FOURPI_OVER_C * q*n_use.n*(vie_k.viz - vie_k.vez)
		//		);
		//}

		//data_1.Azdot = data_k.Azdot
		//	+ h_use * ROCAzdot_antiadvect + h_use * c*c*(LapAz +
		//		0.5*FOURPI_OVER_C * q*data_use.n*(data_k.viz + data_1.viz
		//			- data_k.vez - data_1.vez));

	} else {
		// Non-domain triangle or vertex
		// ==============================
		// Need to decide whether crossing_ins triangle will experience same accel routine as the rest?
		// I think yes so go and add it above??
		// We said v_r = 0 necessarily to avoid sending mass into ins.
		// So how is that achieved there? What about energy loss?
		// Need to determine a good way. Given what v_r in tri represents. We construe it to be AT the ins edge so 
		// ...
		Iz[threadIdx.x] = 0.0;
		sigma_zz[threadIdx.x] = 0.0;

		if ((iMinor < BEGINNING_OF_CENTRAL) && ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL)))
		{
			p_AAdot_intermediate[iMinor].Azdot = 0.0;
			// Set Az equal to neighbour in every case, after Accelerate routine.
		} else {
			// Let's make it go right through the middle of a triangle row for simplicity.

			//f64 Jz = 0.0;
			//if ((iMinor >= numStartZCurrentTriangles) && (iMinor <  numEndZCurrentTriangles))
			//{
			//	// Azdotdot = c^2 (Lap Az + 4pi/c Jz)
			//	// ASSUME we are fed Iz_prescribed.
			//	//Jz = -Iz_prescribed / (real)(numEndZCurrentTriangles - numStartZCurrentTriangles);

			//	AreaMinor = p_AreaMinor[iMinor];
			//	Jz = negative_Iz_per_triangle / AreaMinor; // Iz would come from multiplying back by area and adding.
			//};

			AAdot temp = p_AAdot_src[iMinor];
			temp.Azdot += h_use * c*(c*p_LapAz[iMinor]);// +4.0*M_PI*Jz);
			// + h_use * ROCAzdot_antiadvect // == 0
			p_AAdot_intermediate[iMinor] = temp; // 

		};
	};

	__syncthreads();

	// .Collect Jz = Jz0 + sigma_zz Ez_strength on each minor cell
	// .Estimate Ez
	// sigma_zz should include EzShape for this minor cell

	// The mission if iPass == 0 was passed is to save off Iz0, SigmaIzz.
	// First pass set Ez_strength = 0.0.


	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sigma_zz[threadIdx.x] += sigma_zz[threadIdx.x + k];
			Iz[threadIdx.x] += Iz[threadIdx.x + k];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			sigma_zz[threadIdx.x] += sigma_zz[threadIdx.x + s - 1];
			Iz[threadIdx.x] += Iz[threadIdx.x + s - 1];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_sigma_zz[blockIdx.x] = sigma_zz[0];
		p_Iz0[blockIdx.x] = Iz[0];
	}
	// Wish to make the Jz contribs to Azdot on each side of the ins exactly equal in L1, 
	// meant making this long routine even longer with collecting Iz_k.
}
*/

__global__ void kernelPopulateBackwardOhmsLaw_noadvect(
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
	//AAdot * __restrict__ p_AAdot_intermediate,

	f64 * __restrict__ p_Iz0,
	f64 * __restrict__ p_sigma_zz,

	f64 * __restrict__ p_denom_i,
	f64 * __restrict__ p_denom_e,
	f64 * __restrict__ p_effect_of_viz0_on_vez0,
	f64 * __restrict__ p_beta_ie_z,

	bool const bSwitchSave) 
{
	// Don't forget we can use 16KB shared memory to save a bit of overspill:
	// (16*1024)/(512*8) = 4 doubles only for 512 threads. 128K total register space per SM we think.

	__shared__ f64 Iz[threadsPerTileMinor], sigma_zz[threadsPerTileMinor];
	//	__shared__ f64 Iz_k[threadsPerTileMinor];

	__shared__ f64_vec2 omega[threadsPerTileMinor], grad_Az[threadsPerTileMinor],
		gradTe[threadsPerTileMinor];

	// Putting 8 reduces to 256 simultaneous threads. Experiment with 4 in shared.
	// f64 viz0_coeff_on_Lap_Az, vez0_coeff_on_Lap_Az; // THESE APPLY TO FEINT VERSION. ASSUME NOT FEINT FIRST.

	v4 v0;
	f64 denom, AreaMinor;
	f64_vec3 vn0;
	long const iMinor = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX
	structural info = p_info_minor[iMinor];

	// Can see no reason not to put OUTERMOST here. No point creating a big gradient of vz to it.

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE)
		|| (info.flag == CROSSING_INS) || (info.flag == OUTERMOST))
	{
		v4 vie_k = p_vie_src[iMinor];
		f64_vec3 v_n_src = p_v_n_src[iMinor];
		nvals n_use = p_n_minor_use[iMinor];
		AreaMinor = p_AreaMinor[iMinor];
		// Are we better off with operator = or with memcpy?
		vn0 = v_n_src;

		//		if ((TESTTRI)) printf("GPU %d vie_k %1.14E %1.14E\n", iMinor, vie_k.vxy.x, vie_k.vxy.y);
		{
			f64_vec3 MAR;
			memcpy(&MAR, p_MAR_neut + iMinor, sizeof(f64_vec3));
			// CHECK IT IS INTENDED TO AFFECT Nv

			// REVERTED THE EDIT TO USE 1/n -- THIS WILL NOT GIVE CORRECT M.A.R. EFFECT ON INTEGRAL nv
			// We need conservation laws around shock fronts.
			vn0.x += h_use * (MAR.x / (AreaMinor*n_use.n_n));
			// p_one_over_n[iMinor].n_n/ (AreaMinor));
			vn0.y += h_use * (MAR.y / (AreaMinor*n_use.n_n));// MomAddRate is addition rate for Nv. Divide by N.

			memcpy(&MAR, p_MAR_ion + iMinor, sizeof(f64_vec3));
			v0.vxy = vie_k.vxy + h_use * (m_i*MAR.xypart() / (n_use.n*(m_i + m_e)*AreaMinor));
			v0.viz = vie_k.viz + h_use * MAR.z / (n_use.n*AreaMinor);

			memcpy(&MAR, p_MAR_elec + iMinor, sizeof(f64_vec3));
			v0.vxy += h_use * (m_e*MAR.xypart() / (n_use.n*(m_i + m_e)*AreaMinor));
			v0.vez = vie_k.vez + h_use * MAR.z / (n_use.n*AreaMinor);   

			if (v0.vez != v0.vez) printf("NANVEZ %d v_k %1.9E MAR.z %1.9E \n", iMinor, vie_k.vez, MAR.z);

			if (((TESTTRI))) printf("\nGPU %d a:MAR_e %1.10E %1.10E MAR.y %1.10E 1/n %1.10E Area %1.10E\n", iMinor,
				h_use * (m_e*MAR.x / (n_use.n*(m_i + m_e)*AreaMinor)),
				h_use * (m_e*MAR.y / (n_use.n*(m_i + m_e)*AreaMinor)),
				MAR.y,
				p_one_over_n[iMinor].n, AreaMinor);
		}

		OhmsCoeffs ohm;
		f64 beta_ie_z, LapAz;
		f64 cross_section_times_thermal_en, cross_section_times_thermal_in,
			nu_eiBar, nu_eHeart;
		T3 T = p_T_minor_use[iMinor];
		{
			// Dimensioning inside a brace allows the following vars to go out of scope at the end of the brace.
			f64 sqrt_Te, ionneut_thermal, electron_thermal,
				lnLambda, s_in_MT, s_en_MT, s_en_visc;
			sqrt_Te = sqrt(T.Te);
			ionneut_thermal = sqrt(T.Ti / m_ion + T.Tn / m_n); // hopefully not sqrt(0)
			electron_thermal = sqrt_Te * over_sqrt_m_e;
			lnLambda = Get_lnLambda_d(n_use.n, T.Te);
			{
				f64 s_in_visc_dummy;
				Estimate_Ion_Neutral_Cross_sections_d(T.Ti*one_over_kB, &s_in_MT, &s_in_visc_dummy);
			}
			Estimate_Ion_Neutral_Cross_sections_d(T.Te*one_over_kB, &s_en_MT, &s_en_visc);

			//nu_ne_MT = s_en_MT * electron_thermal * n_use.n; // have to multiply by n_e for nu_ne_MT
			//nu_ni_MT = s_in_MT * ionneut_thermal * n_use.n;
			//nu_in_MT = s_in_MT * ionneut_thermal * n_use.n_n;
			//nu_en_MT = s_en_MT * electron_thermal * n_use.n_n;

			cross_section_times_thermal_en = s_en_MT * electron_thermal;
			cross_section_times_thermal_in = s_in_MT * ionneut_thermal;

			nu_eiBar = nu_eiBarconst * kB_to_3halves*n_use.n*lnLambda / (T.Te*sqrt_Te);
			nu_eHeart = 1.87*nu_eiBar + n_use.n_n*s_en_visc*electron_thermal;
			if (nu_eiBar != nu_eiBar) printf("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n"
				"iMinor %d n_use.n %1.9E lnLambda %1.9E Te %1.9E sqrt %1.9E \n",
				iMinor, n_use.n, lnLambda, T.Te, sqrt_Te);

			// ARTIFICIAL CHANGE TO STOP IONS SMEARING AWAY OFF OF NEUTRAL BACKGROUND:
			if (n_use.n_n > ARTIFICIAL_RELATIVE_THRESH *n_use.n) {
				cross_section_times_thermal_en *= n_use.n_n / (ARTIFICIAL_RELATIVE_THRESH *n_use.n);
				cross_section_times_thermal_in *= n_use.n_n / (ARTIFICIAL_RELATIVE_THRESH *n_use.n);
				// So at 1e18 vs 1e8 it's 10 times stronger
				// At 1e18 vs 1e6 it's 1000 times stronger
				// nu starts at about 1e11 at the place it failed at 35ns. So 10000 times stronger gives us 1e15.
			}

		}

		denom = 1.0 + h_use * M_e_over_en* (cross_section_times_thermal_en*n_use.n)
			+ h_use*M_i_over_in* (cross_section_times_thermal_in*n_use.n);

		vn0 /= denom; // It is now the REDUCED value

		ohm.beta_ne = h_use*(M_e_over_en)*(cross_section_times_thermal_en*n_use.n) / denom;
		ohm.beta_ni = h_use*(M_i_over_in)*(cross_section_times_thermal_in*n_use.n) / denom;

		// Now we do vexy:

		grad_Az[threadIdx.x] = p_GradAz[iMinor];
		gradTe[threadIdx.x] = p_GradTe[iMinor];
		LapAz = p_LapAz[iMinor];
		
		if (((TESTTRI))) printf("GPU %d: LapAz %1.14E\n", CHOSEN, LapAz);

		v0.vxy +=
			- (h_use / ((m_i + m_e)))*(m_n*M_i_over_in*(cross_section_times_thermal_in*n_use.n_n)
				+ m_n * M_e_over_en*(cross_section_times_thermal_en*n_use.n_n))*
				( vn0.xypart());

		denom = 1.0 + (h_use / (m_i + m_e))*(
			m_n* M_i_over_in* (cross_section_times_thermal_in*n_use.n_n)
			+ m_n * M_e_over_en*(cross_section_times_thermal_en*n_use.n_n))*(1.0 - ohm.beta_ne - ohm.beta_ni);
		v0.vxy /= denom;
		
		ohm.beta_xy_z = (h_use * q / (c*(m_i + m_e)*denom)) * grad_Az[threadIdx.x]; // coeff on viz-vez
		
		omega[threadIdx.x] = qovermc*p_B[iMinor].xypart();

		f64 nu_ei_effective = nu_eiBar * (1.0 - 0.9*nu_eiBar*(nu_eHeart*nu_eHeart + qovermc*BZ_CONSTANT*qovermc*BZ_CONSTANT) /
			(nu_eHeart*(nu_eHeart*nu_eHeart + omega[threadIdx.x].x*omega[threadIdx.x].x + omega[threadIdx.x].y*omega[threadIdx.x].y + qovermc*BZ_CONSTANT*qovermc*BZ_CONSTANT)));

		//	if (nu_ei_effective != nu_ei_effective) printf("nu_ei NaN: omega %1.8E %1.8E nu_eHeart %1.8E nu_eiBar %1.8E\n",
		//		omega[threadIdx.x].x, omega[threadIdx.x].y, nu_eHeart, nu_eiBar);

		AAdot AAzdot_k = p_AAdot_src[iMinor];

		v0.viz +=
				-h_use*qoverMc*(AAzdot_k.Azdot + h_use * c*c*LapAz)
				- h_use*qoverMc*(v0.vxy).dot(grad_Az[threadIdx.x]);

		// Still omega_ce . Check formulas.
		
		v0.viz +=
			1.5*h_use*nu_eiBar*((omega[threadIdx.x].x*qovermc*BZ_CONSTANT - nu_eHeart * omega[threadIdx.x].y)*gradTe[threadIdx.x].x +
			(omega[threadIdx.x].y*qovermc*BZ_CONSTANT + nu_eHeart * omega[threadIdx.x].x)*gradTe[threadIdx.x].y) /
				(m_i*nu_eHeart*(nu_eHeart*nu_eHeart + omega[threadIdx.x].dot(omega[threadIdx.x])));

		v0.viz += h_use * M_n_over_ni*(cross_section_times_thermal_in*n_use.n_n) *vn0.z;
			
		denom = 1.0 + h_use * h_use*4.0*M_PI*qoverM*q*n_use.n 
			+ h_use * qoverMc*(grad_Az[threadIdx.x].dot(ohm.beta_xy_z)) +
			h_use * M_n_over_ni*(cross_section_times_thermal_in*n_use.n_n) *(1.0 - ohm.beta_ni)
			+ h_use *moverM*nu_ei_effective;

		if (bSwitchSave) p_denom_i[iMinor] = denom;
		//				viz0_coeff_on_Lap_Az = -0.5*h_use*qoverMc*h_use*c*c / denom;

		v0.viz /= denom;

		if (((TESTTRI))) printf("viz0 divided %1.14E denom %1.14E \n", v0.viz, denom);

		
		ohm.sigma_i_zz = h_use * qoverM / denom;
		beta_ie_z = (h_use*h_use*4.0*M_PI*qoverM*q*n_use.n
			+ h_use*qoverMc*(grad_Az[threadIdx.x].dot(ohm.beta_xy_z))
			+ h_use * M_n_over_ni*(cross_section_times_thermal_in*n_use.n_n) *ohm.beta_ne
			+ h_use * moverM*nu_ei_effective) / denom;
		
		v0.vez +=
			h_use *qovermc*(AAzdot_k.Azdot
				+ h_use * c*c*(LapAz + FOURPI_Q_OVER_C*n_use.n*v0.viz))
			+ h_use*qovermc*(v0.vxy + ohm.beta_xy_z*v0.viz ).dot(grad_Az[threadIdx.x]);
		
		// implies:
		f64 effect_of_viz0_on_vez0 =
			h_use * qovermc*h_use * c*c* FOURPI_Q_OVER_C*n_use.n
			+ h_use*qovermc*(ohm.beta_xy_z.dot(grad_Az[threadIdx.x]));

		v0.vez -=
			1.5*h_use*nu_eiBar*((omega[threadIdx.x].x*qovermc*BZ_CONSTANT - nu_eHeart * omega[threadIdx.x].y)*gradTe[threadIdx.x].x +
			(omega[threadIdx.x].y*qovermc*BZ_CONSTANT + nu_eHeart * omega[threadIdx.x].x)*gradTe[threadIdx.x].y) /
				(m_e*nu_eHeart*(nu_eHeart*nu_eHeart + omega[threadIdx.x].dot(omega[threadIdx.x]) + qovermc*BZ_CONSTANT*qovermc*BZ_CONSTANT));

		// could store this from above and put opposite -- dividing by m_e instead of m_i
		// overdue..?

		v0.vez += h_use*M_n_over_ne*(cross_section_times_thermal_en*n_use.n_n) *(vn0.z + ohm.beta_ni * v0.viz)
				+ h_use*nu_ei_effective*v0.viz;

		// implies:
		effect_of_viz0_on_vez0 +=
				h_use*M_n_over_ne*(cross_section_times_thermal_en*n_use.n_n) *ohm.beta_ni + h_use*nu_ei_effective;
		
		denom = 1.0 + (h_use*h_use*4.0*M_PI*q*eoverm*n_use.n
			+ h_use*qovermc*(grad_Az[threadIdx.x].dot(ohm.beta_xy_z)))*(1.0 - beta_ie_z)
			+ h_use*M_n_over_ne*(cross_section_times_thermal_en*n_use.n_n) *(1.0 - ohm.beta_ne - ohm.beta_ni * beta_ie_z)
			+ h_use*nu_ei_effective*(1.0 - beta_ie_z);

		//		vez0_coeff_on_Lap_Az = h_use * h_use*0.5*qovermc* c*c / denom; 

		ohm.sigma_e_zz =
			(-h_use * eoverm
				+ h_use * h_use*4.0*M_PI*q*eoverm*n_use.n*ohm.sigma_i_zz
				+ h_use *qovermc*(grad_Az[threadIdx.x].dot(ohm.beta_xy_z))*ohm.sigma_i_zz
				+ h_use*M_n_over_ne*(cross_section_times_thermal_en*n_use.n_n) *ohm.beta_ni*ohm.sigma_i_zz
				+ h_use*nu_ei_effective*ohm.sigma_i_zz)
			/ denom;

		v0.vez /= denom;
		effect_of_viz0_on_vez0 /= denom; // of course 

		if (bSwitchSave) {
			p_denom_e[iMinor] = denom;
			p_effect_of_viz0_on_vez0[iMinor] = effect_of_viz0_on_vez0;
			p_beta_ie_z[iMinor] = beta_ie_z; // see that doing it this way was not best.
		} else {
			// #########################################################################################################
			// DEBUG: pass graphing parameters through these.
			// #########################################################################################################
			p_denom_i[iMinor] = M_n_over_ne*cross_section_times_thermal_en*n_use.n_n + nu_ei_effective;
			p_denom_e[iMinor] = M_n_over_ne*cross_section_times_thermal_en*n_use.n_n /
				(M_n_over_ne*cross_section_times_thermal_en*n_use.n_n + nu_ei_effective);
		};

		// Now update viz(Ez):
		v0.viz += beta_ie_z * v0.vez;
		ohm.sigma_i_zz += beta_ie_z * ohm.sigma_e_zz;

		// sigma_e_zz and sigma_i_zz are change in vz for a change in Ez
		{
			f64 EzShape = GetEzShape(info.pos.modulus());
			ohm.sigma_i_zz *= EzShape;
			ohm.sigma_e_zz *= EzShape;
		}

		// Think maybe we should get rid of most of this routine out of the subcycle.
		// Rate of acceleration over timestep due to resistance, pressure, thermal force etc could be stored.
		// Saving off some eqn data isn't so bad when we probably overflow registers and L1 here anyway.
		// All we need is to know that we update sigma
		// We can do addition of 
		// ==============================================================================================

		p_v0_dest[iMinor] = v0;
		p_OhmsCoeffs_dest[iMinor] = ohm;
		p_vn0_dest[iMinor] = vn0;

		Iz[threadIdx.x] = q*AreaMinor*n_use.n*(v0.viz - v0.vez);
		sigma_zz[threadIdx.x] = q*AreaMinor*n_use.n*(ohm.sigma_i_zz - ohm.sigma_e_zz);
			
	}
	else {
		// Non-domain triangle or vertex
		// ==============================
		// Need to decide whether crossing_ins triangle will experience same accel routine as the rest?
		// I think yes so go and add it above??
		// We said v_r = 0 necessarily to avoid sending mass into ins.
		// So how is that achieved there? What about energy loss?
		// Need to determine a good way. Given what v_r in tri represents. We construe it to be AT the ins edge so 
		// ...
		Iz[threadIdx.x] = 0.0;
		sigma_zz[threadIdx.x] = 0.0;

	//	if ((iMinor < BEGINNING_OF_CENTRAL) && ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL)))
//		{
	//		p_AAdot_intermediate[iMinor].Azdot = 0.0;
			// Set Az equal to neighbour in every case, after Accelerate routine.
	//	}
//		else {
			// Let's make it go right through the middle of a triangle row for simplicity.

			//f64 Jz = 0.0;
			//if ((iMinor >= numStartZCurrentTriangles) && (iMinor <  numEndZCurrentTriangles))
			//{
			//	// Azdotdot = c^2 (Lap Az + 4pi/c Jz)
			//	// ASSUME we are fed Iz_prescribed.
			//	//Jz = -Iz_prescribed / (real)(numEndZCurrentTriangles - numStartZCurrentTriangles);

			//	AreaMinor = p_AreaMinor[iMinor];
			//	Jz = negative_Iz_per_triangle / AreaMinor; // Iz would come from multiplying back by area and adding.
			//};

	//		AAdot temp = p_AAdot_src[iMinor];
	//		temp.Azdot += h_use * c*(c*p_LapAz[iMinor]);// +4.0*M_PI*Jz);
														// + h_use * ROCAzdot_antiadvect // == 0
	//		p_AAdot_intermediate[iMinor] = temp; // 

	//	};
	};

	__syncthreads();

	// .Collect Jz = Jz0 + sigma_zz Ez_strength on each minor cell
	// .Estimate Ez
	// sigma_zz should include EzShape for this minor cell

	// The mission if iPass == 0 was passed is to save off Iz0, SigmaIzz.
	// First pass set Ez_strength = 0.0.


	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sigma_zz[threadIdx.x] += sigma_zz[threadIdx.x + k];
			Iz[threadIdx.x] += Iz[threadIdx.x + k];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			sigma_zz[threadIdx.x] += sigma_zz[threadIdx.x + s - 1];
			Iz[threadIdx.x] += Iz[threadIdx.x + s - 1];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_sigma_zz[blockIdx.x] = sigma_zz[0];
		p_Iz0[blockIdx.x] = Iz[0];
	}
	// Wish to make the Jz contribs to Azdot on each side of the ins exactly equal in L1, 
	// meant making this long routine even longer with collecting Iz_k.
}

__global__ void kernelAccelerate_v_from_advection
(
	f64 const h_use,
	structural * __restrict__ p_info_minor,
	nvals * __restrict__ p_n_k,    // multiply by old mass ..
	nvals * __restrict__ p_n_plus, // divide by new mass ..
	v4 * __restrict__ p_vie_k,
	f64_vec3 * __restrict__ p_v_n_k,

	f64_vec3 * __restrict__ p_MAR_neut, // these contain the mom flux due to advection.
	f64_vec3 * __restrict__ p_MAR_ion,
	f64_vec3 * __restrict__ p_MAR_elec,
	f64 * __restrict__ p_AreaMinor,

	// outputs:
	v4 * __restrict__ p_vie_dest,
	f64_vec3 * __restrict__ p_v_n_dest)
{
	long const iMinor = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX
	structural info = p_info_minor[iMinor];

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE)
		|| (info.flag == CROSSING_INS) || (info.flag == OUTERMOST))
	{
		v4 vie_k = p_vie_k[iMinor];
		f64_vec3 v_n_k = p_v_n_k[iMinor];
		nvals n_k = p_n_k[iMinor];
		f64 AreaMinor = p_AreaMinor[iMinor];
		nvals n_dest = p_n_plus[iMinor];
		f64_vec3 MAR;

		memcpy(&MAR, p_MAR_neut + iMinor, sizeof(f64_vec3));
		
		f64_vec3 v_n;

		// v_n = (n_k.n_n*v_n_k + h_use * MAR / (AreaMinor)) / n_dest.n_n;
		
		// . We will really need to overview and see if this formula is correct -- did we already account for change in n?
		// . Need to check if we would be double-counting to include * oldAreaMinor / newAreaMinor --- I think we counted it.
			
		// The problem is that n's change is the change in N inferred in minor
		// so we are preferring to assume that we use the change in N that
		// would follow from looking at v on the boundaries of the minor.

		// That is what we already rounded up in MAR.

		v_n = v_n_k + h_use * MAR / (AreaMinor * n_dest.n_n);
		
		v4 vie;

		memcpy(&MAR, p_MAR_ion + iMinor, sizeof(f64_vec3));
		//vie.vxy = (vie_k.vxy * n_k.n + h_use * (m_i*MAR.xypart() / ((m_i + m_e)*AreaMinor))) / n_dest.n;
		//vie.viz = (vie_k.viz * n_k.n + h_use * MAR.z / (AreaMinor)) / n_dest.n;
		
		// We would PREFER the N_k / N_k+1 version, however 

		vie.vxy = vie_k.vxy + h_use * m_i*MAR.xypart() / ((m_i + m_e)*AreaMinor *n_dest.n);
		vie.viz = vie_k.viz + h_use * MAR.z / (AreaMinor* n_dest.n);

		memcpy(&MAR, p_MAR_elec + iMinor, sizeof(f64_vec3));

		vie.vxy += h_use * (m_e*MAR.xypart() / (n_dest.n*(m_i + m_e)*AreaMinor));
		vie.vez = vie_k.vez + h_use * MAR.z / (AreaMinor * n_dest.n);
		
		memcpy(&(p_vie_dest[iMinor]), &vie, sizeof(v4));
		p_v_n_dest[iMinor] = v_n;
	}
}


__global__ void kernelPopulateBackwardOhmsLaw(
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
	// Now going to need to go through and see this set 0 or sensible every time.

	f64_vec3 * __restrict__ p_vn0_dest,
	v4 * __restrict__ p_v0_dest,
	OhmsCoeffs * __restrict__ p_OhmsCoeffs_dest,
	//AAdot * __restrict__ p_AAdot_intermediate,

	f64 * __restrict__ p_Iz0,
	f64 * __restrict__ p_sigma_zz,

	f64 * __restrict__ p_denom_i,
	f64 * __restrict__ p_denom_e,
	f64 * __restrict__ p_effect_of_viz0_on_vez0,
	f64 * __restrict__ p_beta_ie_z,

	bool const bSwitchSave)
{
	// Don't forget we can use 16KB shared memory to save a bit of overspill:
	// (16*1024)/(512*8) = 4 doubles only for 512 threads. 128K total register space per SM we think.

	__shared__ f64 Iz[threadsPerTileMinor], sigma_zz[threadsPerTileMinor];
	//	__shared__ f64 Iz_k[threadsPerTileMinor];

	__shared__ f64_vec2 omega[threadsPerTileMinor], grad_Az[threadsPerTileMinor],
		gradTe[threadsPerTileMinor];

	// Putting 8 reduces to 256 simultaneous threads. Experiment with 4 in shared.
	// f64 viz0_coeff_on_Lap_Az, vez0_coeff_on_Lap_Az; // THESE APPLY TO FEINT VERSION. ASSUME NOT FEINT FIRST.

	v4 v0;
	f64 denom, ROCAzdot_antiadvect, AreaMinor;
	f64_vec3 vn0;
	long const iMinor = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX
	structural info = p_info_minor[iMinor];

	// Can see no reason not to put OUTERMOST here. No point creating a big gradient of vz to it.

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE)
		|| (info.flag == CROSSING_INS) || (info.flag == OUTERMOST))
	{
		v4 vie_k = p_vie_src[iMinor];
		f64_vec3 v_n_src = p_v_n_src[iMinor];
		nvals n_use = p_n_minor_use[iMinor];
		AreaMinor = p_AreaMinor[iMinor];
		// Are we better off with operator = or with memcpy?
		vn0 = v_n_src;

		//		if ((TESTTRI)) printf("GPU %d vie_k %1.14E %1.14E\n", iMinor, vie_k.vxy.x, vie_k.vxy.y);
		{
			f64_vec3 MAR;
			memcpy(&MAR, p_MAR_neut + iMinor, sizeof(f64_vec3));
			// CHECK IT IS INTENDED TO AFFECT Nv

			// REVERTED THE EDIT TO USE 1/n -- THIS WILL NOT GIVE CORRECT M.A.R. EFFECT ON INTEGRAL nv
			// We need conservation laws around shock fronts.
			vn0.x += h_use * (MAR.x / (AreaMinor*n_use.n_n));
			// p_one_over_n[iMinor].n_n/ (AreaMinor));
			vn0.y += h_use * (MAR.y / (AreaMinor*n_use.n_n));// MomAddRate is addition rate for Nv. Divide by N.

			memcpy(&MAR, p_MAR_ion + iMinor, sizeof(f64_vec3));
			v0.vxy = vie_k.vxy + h_use * (m_i*MAR.xypart() / (n_use.n*(m_i + m_e)*AreaMinor));
			v0.viz = vie_k.viz + h_use * MAR.z / (n_use.n*AreaMinor);

			memcpy(&MAR, p_MAR_elec + iMinor, sizeof(f64_vec3));
			v0.vxy += h_use * (m_e*MAR.xypart() / (n_use.n*(m_i + m_e)*AreaMinor));
			v0.vez = vie_k.vez + h_use * MAR.z / (n_use.n*AreaMinor);

			if (v0.vez != v0.vez) printf("NANVEZ %d v_k %1.9E MAR.z %1.9E \n", iMinor, vie_k.vez, MAR.z);

			if (((TESTTRI))) printf("\nGPU %d a:MAR_e %1.10E %1.10E MAR.y %1.10E 1/n %1.10E Area %1.10E\n", iMinor,
				h_use * (m_e*MAR.x / (n_use.n*(m_i + m_e)*AreaMinor)),
				h_use * (m_e*MAR.y / (n_use.n*(m_i + m_e)*AreaMinor)),
				MAR.y,
				p_one_over_n[iMinor].n, AreaMinor);
		}

		OhmsCoeffs ohm;
		f64 beta_ie_z, LapAz;
		f64 cross_section_times_thermal_en, cross_section_times_thermal_in,
			nu_eiBar, nu_eHeart;
		T3 T = p_T_minor_use[iMinor];
		{
			// Dimensioning inside a brace allows the following vars to go out of scope at the end of the brace.
			f64 sqrt_Te, ionneut_thermal, electron_thermal,
				lnLambda, s_in_MT, s_en_MT, s_en_visc;
			sqrt_Te = sqrt(T.Te);
			ionneut_thermal = sqrt(T.Ti / m_ion + T.Tn / m_n); // hopefully not sqrt(0)
			electron_thermal = sqrt_Te * over_sqrt_m_e;
			lnLambda = Get_lnLambda_d(n_use.n, T.Te);
			{
				f64 s_in_visc_dummy;
				Estimate_Ion_Neutral_Cross_sections_d(T.Ti*one_over_kB, &s_in_MT, &s_in_visc_dummy);
			}
			Estimate_Ion_Neutral_Cross_sections_d(T.Te*one_over_kB, &s_en_MT, &s_en_visc);

			//nu_ne_MT = s_en_MT * electron_thermal * n_use.n; // have to multiply by n_e for nu_ne_MT
			//nu_ni_MT = s_in_MT * ionneut_thermal * n_use.n;
			//nu_in_MT = s_in_MT * ionneut_thermal * n_use.n_n;
			//nu_en_MT = s_en_MT * electron_thermal * n_use.n_n;

			cross_section_times_thermal_en = s_en_MT * electron_thermal;
			cross_section_times_thermal_in = s_in_MT * ionneut_thermal;

			nu_eiBar = nu_eiBarconst * kB_to_3halves*n_use.n*lnLambda / (T.Te*sqrt_Te);
			nu_eHeart = 1.87*nu_eiBar + n_use.n_n*s_en_visc*electron_thermal;
			if (nu_eiBar != nu_eiBar) printf("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n"
				"iMinor %d n_use.n %1.9E lnLambda %1.9E Te %1.9E sqrt %1.9E \n",
				iMinor, n_use.n, lnLambda, T.Te, sqrt_Te);

			// ARTIFICIAL CHANGE TO STOP IONS SMEARING AWAY OFF OF NEUTRAL BACKGROUND:
			if (n_use.n_n > ARTIFICIAL_RELATIVE_THRESH *n_use.n) {
				cross_section_times_thermal_en *= n_use.n_n / (ARTIFICIAL_RELATIVE_THRESH *n_use.n);
				cross_section_times_thermal_in *= n_use.n_n / (ARTIFICIAL_RELATIVE_THRESH *n_use.n);
				// So at 1e18 vs 1e8 it's 10 times stronger
				// At 1e18 vs 1e6 it's 1000 times stronger
				// nu starts at about 1e11 at the place it failed at 35ns. So 10000 times stronger gives us 1e15.
			}

		}

		denom = 1.0 + h_use * M_e_over_en* (cross_section_times_thermal_en*n_use.n)
			+ h_use*M_i_over_in* (cross_section_times_thermal_in*n_use.n);

		vn0 /= denom; // It is now the REDUCED value

		ohm.beta_ne = h_use*(M_e_over_en)*(cross_section_times_thermal_en*n_use.n) / denom;
		ohm.beta_ni = h_use*(M_i_over_in)*(cross_section_times_thermal_in*n_use.n) / denom;

		// Now we do vexy:

		grad_Az[threadIdx.x] = p_GradAz[iMinor];
		gradTe[threadIdx.x] = p_GradTe[iMinor];
		LapAz = p_LapAz[iMinor];

		// SOON GET RID OF THIS CRAP:
		f64 ROCAzdot_antiadvect = ROCAzdotduetoAdvection[iMinor];

		if (((TESTTRI))) printf("GPU %d: LapAz %1.14E\n", CHOSEN, LapAz);

		v0.vxy +=
			-(h_use / ((m_i + m_e)))*(m_n*M_i_over_in*(cross_section_times_thermal_in*n_use.n_n)
				+ m_n * M_e_over_en*(cross_section_times_thermal_en*n_use.n_n))*
				(vn0.xypart());

		denom = 1.0 + (h_use / (m_i + m_e))*(
			m_n* M_i_over_in* (cross_section_times_thermal_in*n_use.n_n)
			+ m_n * M_e_over_en*(cross_section_times_thermal_en*n_use.n_n))*(1.0 - ohm.beta_ne - ohm.beta_ni);
		v0.vxy /= denom;

		ohm.beta_xy_z = (h_use * q / (c*(m_i + m_e)*denom)) * grad_Az[threadIdx.x]; // coeff on viz-vez

		omega[threadIdx.x] = qovermc*p_B[iMinor].xypart();

		f64 nu_ei_effective = nu_eiBar * (1.0 - 0.9*nu_eiBar*(nu_eHeart*nu_eHeart + qovermc*BZ_CONSTANT*qovermc*BZ_CONSTANT) /
			(nu_eHeart*(nu_eHeart*nu_eHeart + omega[threadIdx.x].x*omega[threadIdx.x].x + omega[threadIdx.x].y*omega[threadIdx.x].y + qovermc*BZ_CONSTANT*qovermc*BZ_CONSTANT)));

		//	if (nu_ei_effective != nu_ei_effective) printf("nu_ei NaN: omega %1.8E %1.8E nu_eHeart %1.8E nu_eiBar %1.8E\n",
		//		omega[threadIdx.x].x, omega[threadIdx.x].y, nu_eHeart, nu_eiBar);

		AAdot AAzdot_k = p_AAdot_src[iMinor];

		v0.viz +=
			-h_use*qoverMc*(AAzdot_k.Azdot
				+ h_use * ROCAzdot_antiadvect + h_use * c*c*LapAz)
			- h_use*qoverMc*(v0.vxy).dot(grad_Az[threadIdx.x]);

		// Still omega_ce . Check formulas.

		v0.viz +=
			1.5*h_use*nu_eiBar*((omega[threadIdx.x].x*qovermc*BZ_CONSTANT - nu_eHeart * omega[threadIdx.x].y)*gradTe[threadIdx.x].x +
			(omega[threadIdx.x].y*qovermc*BZ_CONSTANT + nu_eHeart * omega[threadIdx.x].x)*gradTe[threadIdx.x].y) /
				(m_i*nu_eHeart*(nu_eHeart*nu_eHeart + omega[threadIdx.x].dot(omega[threadIdx.x])));

		v0.viz += h_use * M_n_over_ni*(cross_section_times_thermal_in*n_use.n_n) *vn0.z;

		denom = 1.0 + h_use * h_use*4.0*M_PI*qoverM*q*n_use.n
			+ h_use * qoverMc*(grad_Az[threadIdx.x].dot(ohm.beta_xy_z)) +
			h_use * M_n_over_ni*(cross_section_times_thermal_in*n_use.n_n) *(1.0 - ohm.beta_ni)
			+ h_use *moverM*nu_ei_effective;

		if (bSwitchSave) p_denom_i[iMinor] = denom;
		//				viz0_coeff_on_Lap_Az = -0.5*h_use*qoverMc*h_use*c*c / denom;

		v0.viz /= denom;

		if (((TESTTRI))) printf("viz0 divided %1.14E denom %1.14E \n", v0.viz, denom);


		ohm.sigma_i_zz = h_use * qoverM / denom;
		beta_ie_z = (h_use*h_use*4.0*M_PI*qoverM*q*n_use.n
			+ h_use*qoverMc*(grad_Az[threadIdx.x].dot(ohm.beta_xy_z))
			+ h_use * M_n_over_ni*(cross_section_times_thermal_in*n_use.n_n) *ohm.beta_ne
			+ h_use * moverM*nu_ei_effective) / denom;

		v0.vez +=
			h_use *qovermc*(AAzdot_k.Azdot
				+ h_use * ROCAzdot_antiadvect
				+ h_use * c*c*(LapAz + FOURPI_Q_OVER_C*n_use.n*v0.viz))
			+ h_use*qovermc*(v0.vxy + ohm.beta_xy_z*v0.viz).dot(grad_Az[threadIdx.x]);

		// implies:
		f64 effect_of_viz0_on_vez0 =
			h_use * qovermc*h_use * c*c* FOURPI_Q_OVER_C*n_use.n
			+ h_use*qovermc*(ohm.beta_xy_z.dot(grad_Az[threadIdx.x]));

		v0.vez -=
			1.5*h_use*nu_eiBar*((omega[threadIdx.x].x*qovermc*BZ_CONSTANT - nu_eHeart * omega[threadIdx.x].y)*gradTe[threadIdx.x].x +
			(omega[threadIdx.x].y*qovermc*BZ_CONSTANT + nu_eHeart * omega[threadIdx.x].x)*gradTe[threadIdx.x].y) /
				(m_e*nu_eHeart*(nu_eHeart*nu_eHeart + omega[threadIdx.x].dot(omega[threadIdx.x]) + qovermc*BZ_CONSTANT*qovermc*BZ_CONSTANT));

		// could store this from above and put opposite -- dividing by m_e instead of m_i
		// overdue..?

		v0.vez += h_use*M_n_over_ne*(cross_section_times_thermal_en*n_use.n_n) *(vn0.z + ohm.beta_ni * v0.viz)
			+ h_use*nu_ei_effective*v0.viz;

		// implies:
		effect_of_viz0_on_vez0 +=
			h_use*M_n_over_ne*(cross_section_times_thermal_en*n_use.n_n) *ohm.beta_ni + h_use*nu_ei_effective;

		denom = 1.0 + (h_use*h_use*4.0*M_PI*q*eoverm*n_use.n
			+ h_use*qovermc*(grad_Az[threadIdx.x].dot(ohm.beta_xy_z)))*(1.0 - beta_ie_z)
			+ h_use*M_n_over_ne*(cross_section_times_thermal_en*n_use.n_n) *(1.0 - ohm.beta_ne - ohm.beta_ni * beta_ie_z)
			+ h_use*nu_ei_effective*(1.0 - beta_ie_z);

		//		vez0_coeff_on_Lap_Az = h_use * h_use*0.5*qovermc* c*c / denom; 

		ohm.sigma_e_zz =
			(-h_use * eoverm
				+ h_use * h_use*4.0*M_PI*q*eoverm*n_use.n*ohm.sigma_i_zz
				+ h_use *qovermc*(grad_Az[threadIdx.x].dot(ohm.beta_xy_z))*ohm.sigma_i_zz
				+ h_use*M_n_over_ne*(cross_section_times_thermal_en*n_use.n_n) *ohm.beta_ni*ohm.sigma_i_zz
				+ h_use*nu_ei_effective*ohm.sigma_i_zz)
			/ denom;

		v0.vez /= denom;
		effect_of_viz0_on_vez0 /= denom; // of course 

		if (bSwitchSave) {
			p_denom_e[iMinor] = denom;
			p_effect_of_viz0_on_vez0[iMinor] = effect_of_viz0_on_vez0;
			p_beta_ie_z[iMinor] = beta_ie_z; // see that doing it this way was not best.
		}
		else {
			// #########################################################################################################
			// DEBUG: pass graphing parameters through these.
			// #########################################################################################################
			p_denom_i[iMinor] = M_n_over_ne*cross_section_times_thermal_en*n_use.n_n + nu_ei_effective;
			p_denom_e[iMinor] = M_n_over_ne*cross_section_times_thermal_en*n_use.n_n /
				(M_n_over_ne*cross_section_times_thermal_en*n_use.n_n + nu_ei_effective);
		};

		// Now update viz(Ez):
		v0.viz += beta_ie_z * v0.vez;
		ohm.sigma_i_zz += beta_ie_z * ohm.sigma_e_zz;

		// sigma_e_zz and sigma_i_zz are change in vz for a change in Ez
		{
			f64 EzShape = GetEzShape(info.pos.modulus());
			ohm.sigma_i_zz *= EzShape;
			ohm.sigma_e_zz *= EzShape;
		}

		// Think maybe we should get rid of most of this routine out of the subcycle.
		// Rate of acceleration over timestep due to resistance, pressure, thermal force etc could be stored.
		// Saving off some eqn data isn't so bad when we probably overflow registers and L1 here anyway.
		// All we need is to know that we update sigma
		// We can do addition of 
		// ==============================================================================================

		p_v0_dest[iMinor] = v0;
		p_OhmsCoeffs_dest[iMinor] = ohm;
		p_vn0_dest[iMinor] = vn0;

		Iz[threadIdx.x] = q*AreaMinor*n_use.n*(v0.viz - v0.vez);
		sigma_zz[threadIdx.x] = q*AreaMinor*n_use.n*(ohm.sigma_i_zz - ohm.sigma_e_zz);

	}
	else {
		// Non-domain triangle or vertex
		// ==============================
		// Need to decide whether crossing_ins triangle will experience same accel routine as the rest?
		// I think yes so go and add it above??
		// We said v_r = 0 necessarily to avoid sending mass into ins.
		// So how is that achieved there? What about energy loss?
		// Need to determine a good way. Given what v_r in tri represents. We construe it to be AT the ins edge so 
		// ...
		Iz[threadIdx.x] = 0.0;
		sigma_zz[threadIdx.x] = 0.0;

		//	if ((iMinor < BEGINNING_OF_CENTRAL) && ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL)))
		//		{
		//		p_AAdot_intermediate[iMinor].Azdot = 0.0;
		// Set Az equal to neighbour in every case, after Accelerate routine.
		//	}
		//		else {
		// Let's make it go right through the middle of a triangle row for simplicity.

		//f64 Jz = 0.0;
		//if ((iMinor >= numStartZCurrentTriangles) && (iMinor <  numEndZCurrentTriangles))
		//{
		//	// Azdotdot = c^2 (Lap Az + 4pi/c Jz)
		//	// ASSUME we are fed Iz_prescribed.
		//	//Jz = -Iz_prescribed / (real)(numEndZCurrentTriangles - numStartZCurrentTriangles);

		//	AreaMinor = p_AreaMinor[iMinor];
		//	Jz = negative_Iz_per_triangle / AreaMinor; // Iz would come from multiplying back by area and adding.
		//};

		//		AAdot temp = p_AAdot_src[iMinor];
		//		temp.Azdot += h_use * c*(c*p_LapAz[iMinor]);// +4.0*M_PI*Jz);
		// + h_use * ROCAzdot_antiadvect // == 0
		//		p_AAdot_intermediate[iMinor] = temp; // 

		//	};
	};

	__syncthreads();

	// .Collect Jz = Jz0 + sigma_zz Ez_strength on each minor cell
	// .Estimate Ez
	// sigma_zz should include EzShape for this minor cell

	// The mission if iPass == 0 was passed is to save off Iz0, SigmaIzz.
	// First pass set Ez_strength = 0.0.


	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sigma_zz[threadIdx.x] += sigma_zz[threadIdx.x + k];
			Iz[threadIdx.x] += Iz[threadIdx.x + k];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			sigma_zz[threadIdx.x] += sigma_zz[threadIdx.x + s - 1];
			Iz[threadIdx.x] += Iz[threadIdx.x + s - 1];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_sigma_zz[blockIdx.x] = sigma_zz[0];
		p_Iz0[blockIdx.x] = Iz[0];
	}
	// Wish to make the Jz contribs to Azdot on each side of the ins exactly equal in L1, 
	// meant making this long routine even longer with collecting Iz_k.
}

/*__global__ void Estimate_Effect_on_Integral_Azdot_from_Jz_and_LapAz(
	f64 hstep,
	structural * __restrict__ p_info,
	nvals * __restrict__ p_nvals_k,
	nvals * __restrict__ p_nvals_use,
	v4 * __restrict__ p_vie_k,
	v4 * __restrict__ p_vie_kplus1,
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ p_LapAz,

	AAdot * __restrict__ p_Azdot,

	f64 * __restrict__ p_tile1, // +ve Jz
	f64 * __restrict__ p_tile2, // -ve Jz
	f64 * __restrict__ p_tile3, // LapAz
	f64 * __restrict__ p_tile4, // integrate Azdot diff
	f64 * __restrict__ p_tile5,
	f64 * __restrict__ p_tile6
)
{
	__shared__ f64 sum1[threadsPerTileMinor];
	__shared__ f64 sum2[threadsPerTileMinor];
	__shared__ f64 sum3[threadsPerTileMinor];
	__shared__ f64 sum4[threadsPerTileMinor];
	__shared__ f64 sum5[threadsPerTileMinor];
	__shared__ f64 sum6[threadsPerTileMinor];

	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info[iMinor];
	nvals n_k = p_nvals_k[iMinor];
	nvals n_use = p_nvals_use[iMinor];
	v4 v_k = p_vie_k[iMinor];
	v4 v_kplus1 = p_vie_kplus1[iMinor];
	f64 AreaMinor = p_AreaMinor[iMinor];
	f64 LapAz = p_LapAz[iMinor];

	sum1[threadIdx.x] = 0.0; 
	sum2[threadIdx.x] = 0.0;
	sum3[threadIdx.x] = 0.0;
	sum4[threadIdx.x] = 0.0;
	sum5[threadIdx.x] = 0.0;
	sum6[threadIdx.x] = 0.0;
	
	if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST))
	{
		sum1[threadIdx.x] = 
			  hstep*c*c*0.5*FOURPI_OVER_C * q*n_k.n*(v_k.viz - v_k.vez)*AreaMinor
			+ hstep*c*0.5*FOUR_PI*q*n_use.n*(v_kplus1.viz - v_kplus1.vez)*AreaMinor;
		// Was n used consistently?
	} else {
		
										 //if ((iMinor >= numStartZCurrentTriangles) && (iMinor < numEndZCurrentTriangles))
		if (info.flag == REVERSE_JZ_TRI)
			sum2[threadIdx.x] = hstep*c*4.0*M_PI*negative_Iz_per_triangle;
	}

	// make sure we copy from the code:
	sum3[threadIdx.x] = hstep*c*c*LapAz*AreaMinor;
	sum4[threadIdx.x] = fabs(hstep*c*c*LapAz*AreaMinor);

	sum5[threadIdx.x] = p_Azdot[iMinor].Azdot * AreaMinor;
	sum6[threadIdx.x] = fabs(p_Azdot[iMinor].Azdot * AreaMinor);

	// -----------------------------------------------------------------------------

	__syncthreads();

	// .Collect Jz = Jz0 + sigma_zz Ez_strength on each minor cell
	// .Estimate Ez
	// sigma_zz should include EzShape for this minor cell

	// The mission if iPass == 0 was passed is to save off Iz0, SigmaIzz.
	// First pass set Ez_strength = 0.0.
	
	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sum1[threadIdx.x] += sum1[threadIdx.x + k];
			sum2[threadIdx.x] += sum2[threadIdx.x + k];
			sum3[threadIdx.x] += sum3[threadIdx.x + k];
			sum4[threadIdx.x] += sum4[threadIdx.x + k];
			sum5[threadIdx.x] += sum5[threadIdx.x + k];
			sum6[threadIdx.x] += sum6[threadIdx.x + k];
			
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			sum1[threadIdx.x] += sum1[threadIdx.x + s - 1];
			sum2[threadIdx.x] += sum2[threadIdx.x + s - 1];
			sum3[threadIdx.x] += sum3[threadIdx.x + s - 1];
			sum4[threadIdx.x] += sum4[threadIdx.x + s - 1];
			sum5[threadIdx.x] += sum5[threadIdx.x + s - 1];
			sum6[threadIdx.x] += sum6[threadIdx.x + s - 1];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_tile1[blockIdx.x] = sum1[0];
		p_tile2[blockIdx.x] = sum2[0];
		p_tile3[blockIdx.x] = sum3[0];
		p_tile4[blockIdx.x] = sum4[0];
		p_tile5[blockIdx.x] = sum5[0];
		p_tile6[blockIdx.x] = sum6[0];
	}
}*/


__global__ void kernelCalculateVelocityAndAzdot(
	f64 h_use,
	structural * p_info_minor,
	f64_vec3 * __restrict__ p_vn0,
	v4 * __restrict__ p_v0,
	OhmsCoeffs * __restrict__ p_OhmsCoeffs,
	AAdot * __restrict__ p_AAzdot_src,
	nvals * __restrict__ p_n_minor,
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ p_LapAz, // would it be better just to be loading the Azdot0 relation?
	f64 * __restrict__ p_ROCAzdotantiadvect,

	AAdot * __restrict__ p_AAzdot_out,
	v4 * __restrict__ p_vie_out,
	f64_vec3 * __restrict__ p_vn_out)
{
	long iMinor = blockIdx.x*blockDim.x + threadIdx.x;

	structural info = p_info_minor[iMinor];
	AAdot temp = p_AAzdot_src[iMinor];
	temp.Azdot += h_use*(c*c*p_LapAz[iMinor] + p_ROCAzdotantiadvect[iMinor]);
	// We did not add LapAz into Azdot already in PopBackwardOhms.

	if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == DOMAIN_VERTEX)
		|| (info.flag == CROSSING_INS) || (info.flag == OUTERMOST))
	{
		v4 v;
		nvals n_use = p_n_minor[iMinor];
		OhmsCoeffs ohm = p_OhmsCoeffs[iMinor];
		v4 v0 = p_v0[iMinor];
		f64_vec3 v_n = p_vn0[iMinor];							 // 3 sep

		v.vez = v0.vez + ohm.sigma_e_zz * Ez_strength;  // 2
		v.viz = v0.viz + ohm.sigma_i_zz * Ez_strength;  // 2

		v.vxy = v0.vxy + ohm.beta_xy_z * (v.viz - v.vez);   // 4
		v_n.x += (ohm.beta_ne + ohm.beta_ni)*v.vxy.x;    // 2
		v_n.y += (ohm.beta_ne + ohm.beta_ni)*v.vxy.y;
		v_n.z += ohm.beta_ne * v.vez + ohm.beta_ni * v.viz;

		if (info.flag == CROSSING_INS) {
			f64_vec2 rhat = info.pos / info.pos.modulus();
			v_n -= Make3((v_n.dotxy(rhat))*rhat, 0.0);
			v.vxy -= v.vxy.dot(rhat)*rhat;
		}

		memcpy(&(p_vie_out[iMinor]), &v, sizeof(v4)); // operator = vs memcpy
		p_vn_out[iMinor] = v_n;

		if (info.flag == OUTERMOST) {
			temp.Azdot = 0.0;
			temp.Az = 0.0;
		}
		else {
			// BACKWARD:
			temp.Azdot += h_use*c*FOUR_PI*q*n_use.n*(v.viz - v.vez); // logical for C_INS too
		}

		if ((TESTTRI2)) printf(
			"CVAA iMinor %d v0.vez %1.9E sigma_e_zz %1.9E Ez %1.9E v.vez %1.9E\n",
			iMinor, v0.vez, ohm.sigma_e_zz, Ez_strength, v.vez);


	}
	else {

		memset(&(p_vie_out[iMinor]), 0, sizeof(v4));
		memset(&(p_vn_out[iMinor]), 0, sizeof(f64_vec3));

		f64 Jz = 0.0;

		if (info.flag == REVERSE_JZ_TRI)
		{

			// Azdotdot = c^2 (Lap Az + 4pi/c Jz)
			// ASSUME we are fed Iz_prescribed.
			//Jz = -Iz_prescribed / (real)(numEndZCurrentTriangles - numStartZCurrentTriangles);

			f64 AreaMinor = p_AreaMinor[iMinor];
			Jz = negative_Iz_per_triangle / AreaMinor;

			//		printf("temp.Azdot %1.10E ", temp.Azdot);
			temp.Azdot += h_use*c*FOUR_PI*Jz; // Iz would come from multiplying back by area and adding.

											  //		printf("%d Iz %1.14E Area %1.14E Jz %1.14E Azdot %1.14E \n",
											  //		iMinor,
											  //	negative_Iz_per_triangle, AreaMinor, Jz, temp.Azdot);
		};
	}
	// + h_use * ROCAzdot_antiadvect // == 0
	p_AAzdot_out[iMinor] = temp;

	// Would rather make this a separate routine beforehand.


	//data_1.Azdot = data_k.Azdot
	//	+ h_use * ROCAzdot_antiadvect + h_use * c*c*(Lap_Az +
	//		0.5*FOURPI_OVER_C * q*data_use.n*(data_k.viz + data_1.viz
	//			- data_k.vez - data_1.vez));

	//data_1.Azdot = data_k.Azdot
	//		+ h_use * ROCAzdot_antiadvect + h_use * c*c*(Lap_Az +
	//			0.5*FOURPI_OVER_C * q*data_use.n*(data_k.viz - data_k.vez));
	// intermediate
}

__global__ void kernelCalculateVelocityAndAzdot_noadvect(
	f64 h_use,
	structural * p_info_minor,
	f64_vec3 * __restrict__ p_vn0,
	v4 * __restrict__ p_v0,
	OhmsCoeffs * __restrict__ p_OhmsCoeffs,
	AAdot * __restrict__ p_AAzdot_src,
	nvals * __restrict__ p_n_minor, 
	f64 * __restrict__ p_AreaMinor,
	f64 * __restrict__ p_LapAz, // would it be better just to be loading the Azdot0 relation?
	
	AAdot * __restrict__ p_AAzdot_out,
	v4 * __restrict__ p_vie_out,
	f64_vec3 * __restrict__ p_vn_out ) 
{
	long iMinor = blockIdx.x*blockDim.x + threadIdx.x;

	structural info = p_info_minor[iMinor];
	AAdot temp = p_AAzdot_src[iMinor];
	temp.Azdot += h_use*(c*c*p_LapAz[iMinor]);
	// We did not add LapAz into Azdot already in PopBackwardOhms.

	if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == DOMAIN_VERTEX)
		|| (info.flag == CROSSING_INS) || (info.flag == OUTERMOST))
	{
		v4 v;
		nvals n_use = p_n_minor[iMinor];
		OhmsCoeffs ohm = p_OhmsCoeffs[iMinor];
		v4 v0 = p_v0[iMinor];
		f64_vec3 v_n = p_vn0[iMinor];							 // 3 sep

		v.vez = v0.vez + ohm.sigma_e_zz * Ez_strength;  // 2
		v.viz = v0.viz + ohm.sigma_i_zz * Ez_strength;  // 2

		v.vxy = v0.vxy + ohm.beta_xy_z * (v.viz - v.vez);   // 4
		v_n.x += (ohm.beta_ne + ohm.beta_ni)*v.vxy.x;    // 2
		v_n.y += (ohm.beta_ne + ohm.beta_ni)*v.vxy.y;
		v_n.z += ohm.beta_ne * v.vez + ohm.beta_ni * v.viz;
		
		if (info.flag == CROSSING_INS) {
			f64_vec2 rhat = info.pos / info.pos.modulus();
			v_n -= Make3((v_n.dotxy(rhat))*rhat, 0.0);
			v.vxy -= v.vxy.dot(rhat)*rhat;
		}

		memcpy(&(p_vie_out[iMinor]), &v, sizeof(v4)); // operator = vs memcpy
		p_vn_out[iMinor] = v_n;

		if (info.flag == OUTERMOST) {
			temp.Azdot = 0.0;
			temp.Az = 0.0;
		} else {
			// BACKWARD:
			temp.Azdot += h_use*c*FOUR_PI*q*n_use.n*(v.viz - v.vez); // logical for C_INS too
		}
		
		if ((TESTTRI2)) printf(
			"CVAA iMinor %d v0.vez %1.9E sigma_e_zz %1.9E Ez %1.9E v.vez %1.9E\n",
			iMinor, v0.vez, ohm.sigma_e_zz, Ez_strength, v.vez);


	} else {

		memset(&(p_vie_out[iMinor]), 0, sizeof(v4)); 
		memset(&(p_vn_out[iMinor]), 0, sizeof(f64_vec3));

		f64 Jz = 0.0;

		if (info.flag == REVERSE_JZ_TRI)
		{
			
			// Azdotdot = c^2 (Lap Az + 4pi/c Jz)
			// ASSUME we are fed Iz_prescribed.
			//Jz = -Iz_prescribed / (real)(numEndZCurrentTriangles - numStartZCurrentTriangles);

			f64 AreaMinor = p_AreaMinor[iMinor];
			Jz = negative_Iz_per_triangle / AreaMinor;

	//		printf("temp.Azdot %1.10E ", temp.Azdot);
			temp.Azdot += h_use*c*FOUR_PI*Jz; // Iz would come from multiplying back by area and adding.
			
	//		printf("%d Iz %1.14E Area %1.14E Jz %1.14E Azdot %1.14E \n",
		//		iMinor,
			//	negative_Iz_per_triangle, AreaMinor, Jz, temp.Azdot);
		};
	}
	// + h_use * ROCAzdot_antiadvect // == 0
	p_AAzdot_out[iMinor] = temp;

		 // Would rather make this a separate routine beforehand.
	

	//data_1.Azdot = data_k.Azdot
	//	+ h_use * ROCAzdot_antiadvect + h_use * c*c*(Lap_Az +
	//		0.5*FOURPI_OVER_C * q*data_use.n*(data_k.viz + data_1.viz
	//			- data_k.vez - data_1.vez));

	//data_1.Azdot = data_k.Azdot
	//		+ h_use * ROCAzdot_antiadvect + h_use * c*c*(Lap_Az +
	//			0.5*FOURPI_OVER_C * q*data_use.n*(data_k.viz - data_k.vez));
	// intermediate
}

__global__ void kernelCreateEpsilonAndJacobi_Heat
(
	f64 const h_sub,
	structural * __restrict__ p_info_major,
	T3 * __restrict__ p_T_putative,
	T3 * p_T_k, // T_k for substep
	
	// f64 * __restrict__ p_Azdot0,f64 * __restrict__ p_gamma, 
	// corresponded to simple situation where Azdiff = h*(Azdot0+gamma Lap Az)
	
	NTrates * __restrict__ p_NTrates_diffusive,
	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p_AreaMajor,

	f64 * __restrict__ p_coeffself_n, // what about dividing by N?
	f64 * __restrict__ p_coeffself_i,
	f64 * __restrict__ p_coeffself_e,
	f64 * __restrict__ p_epsilon_n,
	f64 * __restrict__ p_epsilon_i,
	f64 * __restrict__ p_epsilon_e,
	f64 * __restrict__ p_Jacobi_n,
	f64 * __restrict__ p_Jacobi_i,
	f64 * __restrict__ p_Jacobi_e	,
	bool * __restrict__ p_bFailedTest
)
{
	// 2. Calculate epsilon: given the est of T, eps = T - (T_k +- h sum kappa dot grad T)

	// So this is a lot like saying, let's call the actual routine...
	// except we also want Jacobi which means we also want coeff on self in epsilon.

	// eps= T_putative - (T_k +- h sum kappa dot grad T_putative)

	// coeff on self we want to be linearized so it incorporates the assumption that it affects kappa.
	// deps/dT = sum [[dkappa/dT = 0.5 kappa/T] dot grad T + kappa dot d/dT grad T]

	// However this means if we know kappa dot grad T then we can * by 0.5/T to get dkappa/dT part
	// But we had to collect a separate value for kappa dot d/dT grad T.

	// We certainly need to somehow modify the existing kappa dot grad T routine here.
	// what about dividing by N?
	
	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;

	structural info = p_info_major[iVertex];
	if (iVertex == VERTCHOSEN) printf("iVertex %d info.flag %d \n", iVertex, info.flag);

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {

		NTrates Rates = p_NTrates_diffusive[iVertex];
		nvals n = p_n_major[iVertex];
		f64 Area = p_AreaMajor[iVertex];
		f64 N = n.n*Area;
		f64 Nn = n.n_n*Area;

		T3 T_putative, T_k;
		memcpy(&T_putative, &(p_T_putative[iVertex]), sizeof(T3));
		memcpy(&T_k, &(p_T_k[iVertex]), sizeof(T3));
		f64 actual_Tn = T_k.Tn + (h_sub / Nn)*Rates.NnTn;
		f64 epsilon_n = T_putative.Tn - actual_Tn;
		p_epsilon_n[iVertex] = epsilon_n;
		f64 actual_Ti = T_k.Ti + (h_sub / N)*Rates.NiTi;
		f64 epsilon_i = T_putative.Ti - actual_Ti;
		p_epsilon_i[iVertex] = epsilon_i;
		f64 actual_Te = T_k.Te + (h_sub / N)*Rates.NeTe;
		f64 epsilon_e = T_putative.Te - actual_Te;
		p_epsilon_e[iVertex] = epsilon_e;

		if (TESTHEAT) printf("iVertex %d epsilon_e %1.10E Te_putative %1.10E Tk %1.10E \n"
			"hsub %1.10E hsub/N %1.10E Rates.NeTe %1.10E hsub/N Rates.NeTe %1.10E\n\n",
			iVertex, epsilon_e, T_putative.Te, T_k.Te, h_sub, h_sub / N, Rates.NeTe,
			(h_sub / N)*Rates.NeTe);
//		
//		if (iVertex == VERTCHOSEN) printf("iVertex %d Te %1.10E Tn %1.10E RatesNnTn %1.10E RatesNeTe %1.10E eps %1.10E %1.10E\n",
//			iVertex, p_T_putative[iVertex].Te, p_T_putative[iVertex].Tn, Rates.NnTn, Rates.NeTe,
//			epsilon_n, epsilon_e);
//
		p_Jacobi_n[iVertex] = -epsilon_n / p_coeffself_n[iVertex]; // should never be 0
		p_Jacobi_i[iVertex] = -epsilon_i / p_coeffself_i[iVertex];
		p_Jacobi_e[iVertex] = -epsilon_e / p_coeffself_e[iVertex];

		if ((epsilon_n*epsilon_n > 1.0e-23*1.0e-23 + 1.0e-16*actual_Tn*actual_Tn) ||
			(epsilon_i*epsilon_i > 1.0e-23*1.0e-23 + 1.0e-16*actual_Ti*actual_Ti) ||
			(epsilon_e*epsilon_e > 1.0e-23*1.0e-23 + 1.0e-16*actual_Te*actual_Te) ||
			(actual_Tn < 0.0) || (actual_Ti < 0.0) || (actual_Te < 0.0))
			p_bFailedTest[blockIdx.x] = true;
//
//		if (epsilon_i*epsilon_i > 1.0e-23*1.0e-23 + 1.0e-16*actual_Ti*actual_Ti) printf("Ti failed %d eps %1.8E Ti %1.8E \n",
//			iVertex, epsilon_i, actual_Ti);
//		if (epsilon_e*epsilon_e > 1.0e-23*1.0e-23 + 1.0e-16*actual_Te*actual_Te) printf("Te failed %d eps %1.8E Te %1.8E \n",
//			iVertex, epsilon_e, actual_Te);
//		if (actual_Tn < 0.0) printf("%d Tn %1.9E \n", iVertex, actual_Tn);
//		if (actual_Ti < 0.0) printf("%d Ti %1.9E \n", iVertex, actual_Ti);
//		if (actual_Te < 0.0) printf("%d Te %1.9E \n", iVertex, actual_Te);
//
		// Next thing is going to have to be to see where & for which species it fails, record the data. 
		// And type of error!
		// Because it looks picky right now.

		// It may be T<0 that is the probs, given that we have arbitrary strength of B-pull on some edge.


		// 1e-28 = 1e-14 1e-14 so that's small. Up to 1e-22 = 1e-9 1e-14.
		// 1e-8 T (so 1e-16 TT) is comparatively quite large -- just past single precision.
		// That seems about right for now.

	//	if (iVertex == 14790) printf("14790 Jacobi %1.9E %1.9E %1.9E eps %1.8E %1.8E %1.8E\n",
	//		p_Jacobi_n[iVertex], p_Jacobi_i[iVertex], p_Jacobi_e[iVertex],
	//		epsilon_n, epsilon_i, epsilon_e);

	} else {
		p_epsilon_n[iVertex] = 0.0;
		p_epsilon_i[iVertex] = 0.0;
		p_epsilon_e[iVertex] = 0.0;
		p_Jacobi_n[iVertex] = 0.0;
		p_Jacobi_i[iVertex] = 0.0;
		p_Jacobi_e[iVertex] = 0.0;
	};

}


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
	f64 * __restrict__ p_Jacobi_x,
	bool * __restrict__ p_bFail)
{
	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	f64 eps;
	structural info = p_info[iMinor];
	if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL))
	{
		eps = 0.0; // p_Lap_Aznext[iMinor];
		p_Jacobi_x[iMinor] = 0.0; // -eps / p_LapCoeffself[iMinor];
//		if (iMinor == 0) printf("\nGPU: eps[0] %1.14E LapCoeffself %1.14E \n", eps, p_LapCoeffself[iMinor]);
		// but we reset it in ResetFrills called for regressor
	}
	else {
#ifdef MIDPT_A_AND_ACTUALLY_MIDPT_A_NOT_JUST_EFFECT_ON_AZDOT

		// WE COULD CHOOSE to leave it so that Az advances with Azdot_k+1 : we don't know a reason why not.

		eps = p_Az_array_next[iMinor] - p_Az_array[iMinor] 
			//- 0.5*h_use* p_AAdot_k[iMinor].Azdot
			- h_use * p_gamma[iMinor] * p_Lap_Aznext[iMinor]
			- h_use * p_Azdot0[iMinor];

		p_Jacobi_x[iMinor] = -eps / (1.0 - h_use * p_gamma[iMinor] * p_LapCoeffself[iMinor]);
#else
		f64 Aznext = p_Az_array_next[iMinor];
		f64 gamma = p_gamma[iMinor];
		eps = Aznext - p_Az_array[iMinor] - h_use * gamma * p_Lap_Aznext[iMinor] - h_use*p_Azdot0[iMinor];

		p_Jacobi_x[iMinor] = -eps / (1.0 - h_use * gamma * p_LapCoeffself[iMinor]);

		
#endif
//		if (iMinor == 25526) printf("\n\n########\nJacobi_x 25526 GPU: %1.14E eps %1.14E gamma %1.14E LapCoeffself %1.14E\n",
//			p_Jacobi_x[iMinor], eps, p_gamma[iMinor], p_LapCoeffself[iMinor]);
//		if (iMinor == 86412) printf("Jacobi_x 86412 GPU: %1.14E eps %1.14E gamma %1.14E LapCoeffself %1.14E\n",
//			p_Jacobi_x[iMinor], eps, p_gamma[iMinor], p_LapCoeffself[iMinor]);
//		if (iMinor == 69531) printf("Jacobi_x 69531 GPU: %1.14E eps %1.14E gamma %1.14E LapCoeffself %1.14E\n",
//			p_Jacobi_x[iMinor], eps, p_gamma[iMinor], p_LapCoeffself[iMinor]);
// Typical value for Az is like 100+ so use 0.1 as minimum that we care about, times relthresh.
		if (eps*eps > 1.0e-10*1.0e-10*(Aznext*Aznext + 1.0*1.0)) p_bFail[blockIdx.x] = true;

		// This does not seem to be triggering.
	};
	p_epsilon[iMinor] = eps;

}

__global__ void kernelCreateEpsilon_Az_CG(
	f64 const h_use,
	structural * __restrict__ p_info,
	f64 * __restrict__ p_Az_plus,
	f64 * __restrict__ p_Az_k,
	f64 * __restrict__ p_Azdot0,
	f64 * __restrict__ p_gamma,
	f64 * __restrict__ p_Lap_Az,
	f64 * __restrict__ p_epsilon,
	bool * __restrict__ p_bFail,
	bool const bSaveFail)
{
	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	f64 eps;
	structural info = p_info[iMinor];
	if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL))
	{
		eps = 0.0; // p_Lap_Aznext[iMinor];		
	}
	else {
		// WE COULD CHOOSE to leave it so that Az advances with Azdot_k+1 : we don't know a reason why not.
		f64 gamma = p_gamma[iMinor];

		f64 sqrthgamma = sqrt(h_use*gamma);
		f64 one_over_sqrt = 1.0 / sqrthgamma;
		f64 Aznext = p_Az_plus[iMinor];
		eps = one_over_sqrt*(Aznext - p_Az_k[iMinor])
			- sqrthgamma * p_Lap_Az[iMinor]
			- one_over_sqrt * h_use * p_Azdot0[iMinor];

		// eps^2 h gamma = original eps squared.
		if (bSaveFail)
			if (eps*eps*h_use*gamma > 1.0e-10*1.0e-10*(Aznext*Aznext + 1.0*1.0))
				p_bFail[blockIdx.x] = true;
		// An optimization is probably to store values in shared then amalgamate, send data to global on 1 thread. ?

		if (eps != eps) printf("iMinor %d eps %1.9E gamma %1.9E sqrthg %1.9E over %1.9E info.flag %d LapAz %1.9E Azdot0 %1.9E\n",
			iMinor, eps, gamma, sqrthgamma, one_over_sqrt, info.flag, p_Lap_Az[iMinor], p_Azdot0[iMinor]);
	};
	p_epsilon[iMinor] = eps;

}


__global__ void kernelCreate_further_regressor(
	structural * __restrict__ p_info,
	f64 h_use,
	f64 * __restrict__ p_regressor,
	f64 * __restrict__ p_Lap_regressor,
	f64 * __restrict__ p_LapCoeffself,
	f64 * __restrict__ p_gamma,
	f64 * __restrict__ p_regressor2)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	f64 d_eps_by_d_beta;
	structural info = p_info[index];
	if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL))
	{
		d_eps_by_d_beta = 0.0; //  Lap_Jacobi[iMinor]; // try ignoring
		// Need to fill this in afterwards by ResetFrills?
	}
	else {
		d_eps_by_d_beta = (p_regressor[index] - h_use * p_gamma[index] * p_Lap_regressor[index]);
	};
	p_regressor2[index] = d_eps_by_d_beta / (1.0 - h_use * p_gamma[index] * p_LapCoeffself[index]);

}




__global__ void kernelGetLap_minor(
	structural * __restrict__ p_info,
	f64 * __restrict__ p_Az,

	long * __restrict__ p_izTri,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtri_vertex,
	char * __restrict__ p_szPBCtriminor,

	f64 * __restrict__ p_LapAz,
	
	f64 * __restrict__ p_AreaMinor)
// debug why it is that we get sum of Lap nonzero when we integrate against AreaMinor, yet sum here to small
{
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];
	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajor];
	__shared__ f64 shared_Az[threadsPerTileMinor];
	__shared__ f64 shared_Az_verts[threadsPerTileMajor];
	
//	__shared__ f64 sum1[threadsPerTileMinor];
//	__shared__ f64 sum2[threadsPerTileMinor];
//	__shared__ f64 sum3[threadsPerTileMinor];
	
	// 4.5 per thread.
	// Not clear if better off with L1 or shared mem in this case?? Probably shared mem.

	// For now, stick with idea that vertices have just major indices that come after tris.
	// Minor indices are not made contiguous - although it might be better ultimately.

	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	long const iVertex = threadsPerTileMajor*blockIdx.x + threadIdx.x; // only meaningful threadIdx.x < threadsPerTileMajor

	long const StartMinor = threadsPerTileMinor*blockIdx.x;
	long const StartMajor = threadsPerTileMajor*blockIdx.x;
	long const EndMinor = StartMinor + threadsPerTileMinor;
	long const EndMajor = StartMajor + threadsPerTileMajor;

//	sum1[threadIdx.x] = 0.0;
//	sum2[threadIdx.x] = 0.0;
//	sum3[threadIdx.x] = 0.0; // save for verts

	shared_pos[threadIdx.x] = p_info[iMinor].pos;
	shared_Az[threadIdx.x] = p_Az[iMinor];
	structural info;
	if (threadIdx.x < threadsPerTileMajor) {
		info = p_info[iVertex + BEGINNING_OF_CENTRAL];
		shared_pos_verts[threadIdx.x] = info.pos;
		shared_Az_verts[threadIdx.x] = p_Az[iVertex + BEGINNING_OF_CENTRAL];
	};

	__syncthreads();

	f64 ourAz, oppAz, prevAz, nextAz;
	f64_vec2 opppos, prevpos, nextpos;
	// Better if we use same share to do both tris and verts
	// Idea: let's make it called for # minor threads, each loads 1 shared value,
	// and only half the threads run first for the vertex part. That is a pretty good idea.

	if (threadIdx.x < threadsPerTileMajor) {

		f64 Our_integral_Lap_Az = 0.0;
		f64 AreaMinor = 0.0;
		long izTri[MAXNEIGH];
		char szPBC[MAXNEIGH];
		short tri_len = info.neigh_len;
		f64_vec2 endpt0, endpt1;

		memcpy(izTri, p_izTri + MAXNEIGH*iVertex, sizeof(long)*MAXNEIGH);
		memcpy(szPBC, p_szPBCtri_vertex + MAXNEIGH*iVertex, sizeof(char)*MAXNEIGH);
		// Is this best way? better than going looking for periodic data on each tri.

		ourAz = shared_Az_verts[threadIdx.x];

		short iprev = tri_len - 1;
		if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMinor))
		{
			prevAz = shared_Az[izTri[iprev] - StartMinor];
			prevpos = shared_pos[izTri[iprev] - StartMinor];
		}
		else {
			prevAz = p_Az[izTri[iprev]];
			prevpos = p_info[izTri[iprev]].pos;
		}
		if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) prevpos = Clockwise_d*prevpos;
		if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) prevpos = Anticlockwise_d*prevpos;

		short i = 0;
		if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
		{
			oppAz = shared_Az[izTri[i] - StartMinor];
			opppos = shared_pos[izTri[i] - StartMinor];
		}
		else {
			oppAz = p_Az[izTri[i]];
			opppos = p_info[izTri[i]].pos;
		}
		if (szPBC[i] == ROTATE_ME_CLOCKWISE) opppos = Clockwise_d*opppos;
		if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) opppos = Anticlockwise_d*opppos;

		endpt0 = THIRD * (info.pos + opppos + prevpos);

		short inext, iend = tri_len;

		f64_vec2 projendpt0, edge_normal;
		if ((info.flag == INNERMOST) || (info.flag == OUTERMOST)) {

			iend = tri_len - 2;
			if (info.flag == OUTERMOST) {
				endpt0.project_to_radius(projendpt0, FRILL_CENTROID_OUTER_RADIUS_d); // back of cell for Lap purposes
			}
			else {
				endpt0.project_to_radius(projendpt0, FRILL_CENTROID_INNER_RADIUS_d); // back of cell for Lap purposes
			};
			edge_normal.x = endpt0.y - projendpt0.y;
			edge_normal.y = projendpt0.x - endpt0.x;
			AreaMinor += (0.5*projendpt0.x + 0.5*endpt0.x)*edge_normal.x;

		};

		if (iend > MAXNEIGH) printf("####################\nvertex %d iend = %d info.neigh_len = %d\n", iVertex, iend, info.neigh_len);

		for (i = 0; i < iend; i++)
		{
			// Tri 0 is anticlockwise of neighbour 0, we think
			inext = i + 1; if (inext >= tri_len) inext = 0;

			if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
			{
				nextAz = shared_Az[izTri[inext] - StartMinor];
				nextpos = shared_pos[izTri[inext] - StartMinor];
			} else {
				nextAz = p_Az[izTri[inext]];
				nextpos = p_info[izTri[inext]].pos;
			}
			if (szPBC[inext] == ROTATE_ME_CLOCKWISE) nextpos = Clockwise_d*nextpos;
			if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) nextpos = Anticlockwise_d*nextpos;

			endpt1 = THIRD * (nextpos + info.pos + opppos); 
			f64_vec2 edge_normal, integ_grad_Az;

			edge_normal.x = endpt1.y - endpt0.y;
			edge_normal.y = endpt0.x - endpt1.x;

			// ______________________________________________________-

			//	Az_edge = SIXTH * (2.0*ourdata.Az + 2.0*oppAz + prevAz + nextAz);
			//	Our_integral_grad_Az += Az_edge * edge_normal;
			//	Our_integral_curl_Az += Az_edge * (endpt1 - endpt0); // looks anticlockwise
			//	pData[iDestTri].B -= Az_edge * (endpt1 - endpt0); // MUST DIVIDE BY AREA
			integ_grad_Az.x = 0.5*(
				(ourAz + nextAz)*(info.pos.y - nextpos.y)
				+ (prevAz + ourAz)*(prevpos.y - info.pos.y)
				+ (oppAz + prevAz)*(opppos.y - prevpos.y)
				+ (nextAz + oppAz)*(nextpos.y - opppos.y)
				);
			integ_grad_Az.y = -0.5*( // notice minus
				(ourAz + nextAz)*(info.pos.x - nextpos.x)
				+ (prevAz + ourAz)*(prevpos.x - info.pos.x)
				+ (oppAz + prevAz)*(opppos.x - prevpos.x)
				+ (nextAz + oppAz)*(nextpos.x - opppos.x)
				);
			f64 area_quadrilateral = 0.5*(
				(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
				+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
				+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
				+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
				);
			//f64_vec2 grad_Az = integ_grad_Az / area_quadrilateral;
			Our_integral_Lap_Az += integ_grad_Az.dot(edge_normal) / area_quadrilateral;

			AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;


	//		if (TEST) printf("VERTCHOSEN %d izTri[i] %d contrib %1.14E \n"
	//			"grad Az %1.14E %1.14E edgenormal %1.14E %1.14E \n",
	//			VERTCHOSEN, izTri[i], integ_grad_Az.dot(edge_normal) / area_quadrilateral,
	//			integ_grad_Az.x / area_quadrilateral, integ_grad_Az.y / area_quadrilateral,
	//			edge_normal.x, edge_normal.y);
			
			endpt0 = endpt1;
			prevpos = opppos;
			opppos = nextpos;
			prevAz = oppAz;
			oppAz = nextAz;

			++iprev;
		}; // next i


//		sum3[threadIdx.x] = Our_integral_Lap_Az; // vertex sum

		if ((info.flag == INNERMOST) || (info.flag == OUTERMOST)) {
			// Now add on the final sides to give area:

			//    3     4
			//     2 1 0
			// endpt0=endpt1 is now the point north of edge facing 2 anyway.
			f64_vec2 projendpt1;

			if (info.flag == OUTERMOST) {
				endpt1.project_to_radius(projendpt1, FRILL_CENTROID_OUTER_RADIUS_d);
			}
			else {
				endpt1.project_to_radius(projendpt1, FRILL_CENTROID_INNER_RADIUS_d);
			};
			edge_normal.x = projendpt1.y - endpt1.y;
			edge_normal.y = endpt1.x - projendpt1.x;
			AreaMinor += (0.5*projendpt1.x + 0.5*endpt1.x)*edge_normal.x;

			edge_normal.x = projendpt0.y - projendpt1.y;
			edge_normal.y = projendpt1.x - projendpt0.x;
			AreaMinor += (0.5*projendpt1.x + 0.5*projendpt0.x)*edge_normal.x;
			// line between out-projected points

			// NEW ADDITION:

			if (info.flag == OUTERMOST)
			{
				// set outer value at Az = (r_here/r_outer)*Az_self
				// contrib to integral of Lap is [delta_edge * dAz/dr] = 
				// delta_edge*Az_self * (r here - r outer)/(r_outer^2 - r_outer*r_here)
				// = delta_edge*Az_self * (-1.0)/(r_outer)

				f64 delta_edge = (projendpt0 - projendpt1).modulus();
				
				Our_integral_Lap_Az -= delta_edge*ourAz / FRILL_CENTROID_OUTER_RADIUS_d;
			}
		};

	//	if (TEST) printf("AreaMinor : %1.14E %1.14E \n", AreaMinor, p_AreaMinor[iVertex + BEGINNING_OF_CENTRAL]);
	//	f64 AreaStored = p_AreaMinor[iVertex + BEGINNING_OF_CENTRAL];
	//	if (fabs(AreaStored - AreaMinor) > 0.0000001*AreaMinor){
	//		printf("ALERT\n"
	//		"AreaStored %1.14E AreaMinor %1.14E iVertex %d \n",
	//			AreaStored, AreaMinor, iVertex);
	//	}

		p_LapAz[iVertex + BEGINNING_OF_CENTRAL] = Our_integral_Lap_Az / AreaMinor;
		p_AreaMinor[iVertex + BEGINNING_OF_CENTRAL] = AreaMinor; // reset just because otherwise we're inconsistent about area/position in a subcycle

	}; // was thread in the first half of the block

	info = p_info[iMinor];
	ourAz = shared_Az[threadIdx.x];

	long izNeighMinor[6];
	char szPBC[6];
	memcpy(izNeighMinor, p_izNeighMinor + iMinor * 6, sizeof(long) * 6);
	memcpy(szPBC, p_szPBCtriminor + iMinor * 6, sizeof(char) * 6);

	if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL)) {

		// izNeighMinor[0] is actually vertex 0 if you are triangle 0.
		// Rethink:  
		// Try izNeighMinor[3] because this is meant to be neighbour 0.

		// Why are we doing all this? Just set = 0 out here.
		p_LapAz[iMinor] = 0.0; 
		/*
		if ((izNeighMinor[3] >= StartMinor) && (izNeighMinor[3] < EndMinor))
		{
			oppAz = shared_Az[izNeighMinor[3] - StartMinor];
		}
		else {
			oppAz = p_Az[izNeighMinor[3]];
		};
		p_LapAz[iMinor] = oppAz - ourAz;
		*/
		// DEBUG:
	//	if ((TESTTRI)1) printf("\nGPU: oppAz %1.14E ourAz %1.14E LapAz %1.14E izNeighMinor[3] start end %d %d %d %1.14E\n",
	//		oppAz, ourAz, p_LapAz[iMinor], izNeighMinor[3], StartMinor, EndMinor, p_Az[izNeighMinor[3]]);

	}
	else {

		f64 Our_integral_Lap_Az = 0.0;
		f64 AreaMinor = 0.0;

		short inext, i = 0, iprev = 5;
		if ((izNeighMinor[iprev] >= StartMinor) && (izNeighMinor[iprev] < EndMinor))
		{
			prevAz = shared_Az[izNeighMinor[iprev] - StartMinor];
			prevpos = shared_pos[izNeighMinor[iprev] - StartMinor];
		}
		else {
			if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
				(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL))
			{
				prevAz = shared_Az_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
				prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
			}
			else {
				prevAz = p_Az[izNeighMinor[iprev]];
				prevpos = p_info[izNeighMinor[iprev]].pos;
			};
		};
		if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) prevpos = Clockwise_d*prevpos;
		if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) prevpos = Anticlockwise_d*prevpos;

		i = 0;
		if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
		{
			oppAz = shared_Az[izNeighMinor[i] - StartMinor];
			opppos = shared_pos[izNeighMinor[i] - StartMinor];
		}
		else {
			if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
				(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
			{
				oppAz = shared_Az_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
				opppos = shared_pos_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
			}
			else {
				oppAz = p_Az[izNeighMinor[i]];
				opppos = p_info[izNeighMinor[i]].pos;
			};
		};
		if (szPBC[i] == ROTATE_ME_CLOCKWISE) opppos = Clockwise_d*opppos;
		if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) opppos = Anticlockwise_d*opppos;

		//		if ((TESTTRI)) printf("\nGPU %d Az %1.14E\n", CHOSEN, ourAz);


#pragma unroll 
		for (i = 0; i < 6; i++)
		{
			inext = i + 1; if (inext > 5) inext = 0;

			//			if ((TESTTRI)) printf("GPU %d Az %1.14E\n", izNeighMinor[i], oppAz);

			if ((izNeighMinor[inext] >= StartMinor) && (izNeighMinor[inext] < EndMinor))
			{
				nextAz = shared_Az[izNeighMinor[inext] - StartMinor];
				nextpos = shared_pos[izNeighMinor[inext] - StartMinor];
			}
			else {
				if ((izNeighMinor[inext] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[inext] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					nextAz = shared_Az_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
					nextpos = shared_pos_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					nextAz = p_Az[izNeighMinor[inext]];
					nextpos = p_info[izNeighMinor[inext]].pos;
				};
			};
			if (szPBC[inext] == ROTATE_ME_CLOCKWISE) nextpos = Clockwise_d*nextpos;
			if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) nextpos = Anticlockwise_d*nextpos;

			// ______________________________________________________-

			//	Az_edge = SIXTH * (2.0*ourdata.Az + 2.0*oppAz + prevAz + nextAz);
			//	Our_integral_grad_Az += Az_edge * edge_normal;
			//	Our_integral_curl_Az += Az_edge * (endpt1 - endpt0); // looks anticlockwise
			f64_vec2 integ_grad_Az;

			integ_grad_Az.x = 0.5*(
				(ourAz + nextAz)*(info.pos.y - nextpos.y)
				+ (prevAz + ourAz)*(prevpos.y - info.pos.y)
				+ (oppAz + prevAz)*(opppos.y - prevpos.y)
				+ (nextAz + oppAz)*(nextpos.y - opppos.y)
				);
			integ_grad_Az.y = -0.5*( // notice minus
				(ourAz + nextAz)*(info.pos.x - nextpos.x)
				+ (prevAz + ourAz)*(prevpos.x - info.pos.x)
				+ (oppAz + prevAz)*(opppos.x - prevpos.x)
				+ (nextAz + oppAz)*(nextpos.x - opppos.x)
				);
			f64 area_quadrilateral = 0.5*(
				(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
				+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
				+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
				+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
				);

			//f64_vec2 grad_Az = integ_grad_Az / area_quadrilateral;
			// DEBUG:

			// New definition of endpoint of minor edge:
			f64_vec2 edge_normal;
			//endpt0 = THIRD * (prevpos + info.pos + opppos);
			//endpt1 = THIRD * (nextpos + info.pos + opppos);

			//edge_normal.x = endpt1.y - endpt0.y;
			//edge_normal.y = endpt0.x - endpt1.x;
			edge_normal.x = THIRD*(nextpos.y - prevpos.y);
			edge_normal.y = THIRD*(prevpos.x - nextpos.x);

			if ((opppos.dot(opppos) < 0.9999*0.9999*FRILL_CENTROID_OUTER_RADIUS_d*FRILL_CENTROID_OUTER_RADIUS_d) &&
				(opppos.dot(opppos) > 1.0001*1.0001*FRILL_CENTROID_INNER_RADIUS_d*FRILL_CENTROID_INNER_RADIUS_d))
			{
				// neighbour's not a frill

				Our_integral_Lap_Az += integ_grad_Az.dot(edge_normal) / area_quadrilateral;

				if (TESTTRI3) {
					printf("iMinor %d izNeighMinor[i] %d ourAz %1.9E theirs %1.9E contrib %1.9E \n",
						iMinor, izNeighMinor[i], ourAz, oppAz,
						integ_grad_Az.dot(edge_normal) / area_quadrilateral);
				}
			}

			AreaMinor += SIXTH*((prevpos.x + info.pos.x + opppos.x) +
				(nextpos.x + info.pos.x + opppos.x))*edge_normal.x;

	//		if ((TESTTRI)2) printf("CHOSEN2 %d izNeighMinor[i] %d our.pos %1.14E %1.14E "
	//			"oppdata.pos %1.14E %1.14E\n"
	//			"gradAz %1.14E %1.14E edge_normal %1.14E %1.14E \n"
	//			"contrib %1.14E Area_quadrilateral %1.14E\n",
	//			CHOSEN2, izNeighMinor[i],
	//			info.pos.x, info.pos.y, opppos.x, opppos.y,
	//			integ_grad_Az.x / area_quadrilateral, integ_grad_Az.y / area_quadrilateral,
	//			edge_normal.x, edge_normal.y,
	//			integ_grad_Az.dot(edge_normal) / area_quadrilateral,
	//			area_quadrilateral);

		//	if (i % 2 == 0) { // vert neighs
		//		sum1[threadIdx.x] += integ_grad_Az.dot(edge_normal) / area_quadrilateral;
				// comes out at order e+7, nothing like what vertex sees from triangles.
				// That makes little sense but it also comes out e+7 when we are at 0?
		//	} else {
		//		if ((izNeighMinor[i] >= NumInnerFrills_d) && (izNeighMinor[i] < FirstOuterFrill_d))
		//			sum2[threadIdx.x] += integ_grad_Az.dot(edge_normal) / area_quadrilateral;
				// comes out very small
		//	}

			//if ((TESTTRI)) printf("GradAz.x comps: %1.14E %1.14E %1.14E %1.14E\n"
			//	"%1.14E %1.14E %1.14E %1.14E\n",
			//	ourAz, prevAz, oppAz, nextAz,
			//	(info.pos.y - nextpos.y) + (prevpos.y - info.pos.y),
			//	(prevpos.y - info.pos.y) + (opppos.y - prevpos.y),
			//	(opppos.y - prevpos.y) + (nextpos.y - opppos.y),
			//	(nextpos.y - opppos.y) + (info.pos.y - nextpos.y)
			//);

			prevAz = oppAz;
			oppAz = nextAz;
			prevpos = opppos;
			opppos = nextpos;
			// There is an even quicker way which is to rotate pointers. No memcpy needed.
		};

		p_LapAz[iMinor] = Our_integral_Lap_Az / AreaMinor;
		p_AreaMinor[iMinor] = AreaMinor; // reset for each substep

	//	f64 AreaStored = p_AreaMinor[iMinor];
	//	if (fabs(AreaStored - AreaMinor) > 0.0000001*AreaMinor){
	//		printf("ALERT\n"
	//		"AreaStored %1.14E AreaMinor %1.14E iMinor %d \n",
	//			AreaStored, AreaMinor, iMinor);
	//	}

	//	if ((TESTTRI)2) printf("GPU CHOSEN2 %d LapAz %1.14E integralLap %1.14E AreaMinor %1.14E\n",
	//		CHOSEN2, p_LapAz[iMinor], Our_integral_Lap_Az, AreaMinor);

	//	sum3[threadIdx.x] += p_LapAz[iMinor] * AreaStored;
	};

	/*
	__syncthreads();

	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			sum1[threadIdx.x] += sum1[threadIdx.x + k];
			sum2[threadIdx.x] += sum2[threadIdx.x + k];
			sum3[threadIdx.x] += sum3[threadIdx.x + k];
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			sum1[threadIdx.x] += sum1[threadIdx.x + s - 1];
			sum2[threadIdx.x] += sum2[threadIdx.x + s - 1];
			sum3[threadIdx.x] += sum3[threadIdx.x + s - 1];
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_Integratedconts_fromtri[blockIdx.x] = sum1[threadIdx.x];
		p_Integratedconts_fromvert[blockIdx.x] = sum2[threadIdx.x];
		p_Integratedconts_vert[blockIdx.x] = sum3[threadIdx.x]; // covers the vertices within this tile
	}	*/

}


__global__ void kernelGetLapCoeffs_and_min(
	structural * __restrict__ p_info,
	long * __restrict__ p_izTri,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtri_vertex,
	char * __restrict__ p_szPBCtriminor,
	f64 * __restrict__ p_LapCoeffSelf,
	f64 * __restrict__ p_min_array,
	long * __restrict__ p_min_index)
{
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];
	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajor];

	__shared__ char shared_flag[threadsPerTileMinor];

	__shared__ f64 mincoeffself[threadsPerTileMinor];
	__shared__ long iMin[threadsPerTileMinor];

	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	long const iVertex = threadsPerTileMajor*blockIdx.x + threadIdx.x; // only meaningful threadIdx.x < threadsPerTileMajor

	long const StartMinor = threadsPerTileMinor*blockIdx.x;
	long const StartMajor = threadsPerTileMajor*blockIdx.x;
	long const EndMinor = StartMinor + threadsPerTileMinor;
	long const EndMajor = StartMajor + threadsPerTileMajor;


	structural info = p_info[iMinor];
	shared_pos[threadIdx.x] = info.pos;
	shared_flag[threadIdx.x] = info.flag;
	if (threadIdx.x < threadsPerTileMajor) {
		info = p_info[iVertex + BEGINNING_OF_CENTRAL];
		shared_pos_verts[threadIdx.x] = info.pos;
	};

	__syncthreads();

	f64_vec2 opppos, prevpos, nextpos;
	// Better if we use same share to do both tris and verts
	// Idea: let's make it called for # minor threads, each loads 1 shared value,
	// and only half the threads run first for the vertex part. That is a pretty good idea.

	mincoeffself[threadIdx.x] = 0.0;
	iMin[threadIdx.x] = -1;

	if (threadIdx.x < threadsPerTileMajor) {

		f64 Our_integral_Lap_Az_contrib_from_own_Az = 0.0;
		f64 AreaMinor = 0.0;
		long izTri[MAXNEIGH];
		char szPBC[MAXNEIGH];
		short tri_len = info.neigh_len;

		memcpy(izTri, p_izTri + MAXNEIGH*iVertex, sizeof(long)*MAXNEIGH);
		memcpy(szPBC, p_szPBCtri_vertex + MAXNEIGH*iVertex, sizeof(char)*MAXNEIGH);
		// Is this best way? better than going looking for periodic data on each tri.

		short iprev = tri_len - 1;
		if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMinor))
		{
			prevpos = shared_pos[izTri[iprev] - StartMinor];
		} else {
			prevpos = p_info[izTri[iprev]].pos;
		}
		if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) prevpos = Clockwise_d*prevpos;
		if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) prevpos = Anticlockwise_d*prevpos;

		short inext, i = 0;
		if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
		{
			opppos = shared_pos[izTri[i] - StartMinor];
		}
		else {
			opppos = p_info[izTri[i]].pos;
		}
		if (szPBC[i] == ROTATE_ME_CLOCKWISE) opppos = Clockwise_d*opppos;
		if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) opppos = Anticlockwise_d*opppos;

		f64_vec2 endpt0 = THIRD * (info.pos + opppos + prevpos);
		f64_vec2 endpt1, edge_normal;



		short iend = tri_len;
		f64_vec2 projendpt0;
		if ((info.flag == INNERMOST) || (info.flag == OUTERMOST)) {

			iend = tri_len - 2;
			if (info.flag == OUTERMOST) {
				endpt0.project_to_radius(projendpt0, FRILL_CENTROID_OUTER_RADIUS_d); // back of cell for Lap purposes
			}
			else {
				endpt0.project_to_radius(projendpt0, FRILL_CENTROID_INNER_RADIUS_d); // back of cell for Lap purposes
			};
			edge_normal.x = endpt0.y - projendpt0.y;
			edge_normal.y = projendpt0.x - endpt0.x;
			AreaMinor += (0.5*projendpt0.x + 0.5*endpt0.x)*edge_normal.x;
		};

		for (i = 0; i < iend; i++)
		{
			// Tri 0 is anticlockwise of neighbour 0, we think
			inext = i + 1; if (inext >= tri_len) inext = 0;

			if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
			{
				nextpos = shared_pos[izTri[inext] - StartMinor];
			} else {
				nextpos = p_info[izTri[inext]].pos;
			}
			if (szPBC[inext] == ROTATE_ME_CLOCKWISE) nextpos = Clockwise_d*nextpos;
			if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) nextpos = Anticlockwise_d*nextpos;

			endpt1 = THIRD * (nextpos + info.pos + opppos);

			edge_normal.x = endpt1.y - endpt0.y;
			edge_normal.y = endpt0.x - endpt1.x;

			// ______________________________________________________-

			//	Az_edge = SIXTH * (2.0*ourdata.Az + 2.0*oppAz + prevAz + nextAz);
			//	Our_integral_grad_Az += Az_edge * edge_normal;
			//	Our_integral_curl_Az += Az_edge * (endpt1 - endpt0); // looks anticlockwise
			//	pData[iDestTri].B -= Az_edge * (endpt1 - endpt0); // MUST DIVIDE BY AREA
			f64_vec2 integ_grad_Az;
			integ_grad_Az.x = 0.5*(
				(1.0)*(info.pos.y - nextpos.y)
				+ (1.0)*(prevpos.y - info.pos.y)
				);
			integ_grad_Az.y = -0.5*( // notice minus
				(1.0)*(info.pos.x - nextpos.x)
				+ (1.0)*(prevpos.x - info.pos.x)
				);
			f64 area_quadrilateral = 0.5*(
				(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
				+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
				+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
				+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
				);

			Our_integral_Lap_Az_contrib_from_own_Az += integ_grad_Az.dot(edge_normal) / area_quadrilateral;


	//		if (iVertex + BEGINNING_OF_CENTRAL == CHOSEN) {
	//			printf("%d contrib %1.14E %d \nourpos %1.14E %1.14E opppos %1.14E %1.14E \n"
	//				"prevpos nextpos %1.14E %1.14E %1.14E %1.14E\n"
	//				"szPBC[i] %d area_quadrilateral %1.14E \n", 
	//				iVertex + BEGINNING_OF_CENTRAL, 
	//				integ_grad_Az.dot(edge_normal) / area_quadrilateral,
	//				izTri[i], 
	//				info.pos.x,info.pos.y,opppos.x,opppos.y,
	//				prevpos.x,prevpos.y,nextpos.x,nextpos.y,
	//				(int)szPBC[i],area_quadrilateral);				
	//		}

			AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;

			++iprev;
			endpt0 = endpt1;
			prevpos = opppos;
			opppos = nextpos;
		}; // next i

		if ((info.flag == INNERMOST) || (info.flag == OUTERMOST)) {
			// Now add on the final sides to give area:

			//    3     4
			//     2 1 0
			// endpt0=endpt1 is now the point north of edge facing 2 anyway.
			f64_vec2 projendpt1;

			if (info.flag == OUTERMOST) {
				endpt1.project_to_radius(projendpt1, FRILL_CENTROID_OUTER_RADIUS_d);
			}
			else {
				endpt1.project_to_radius(projendpt1, FRILL_CENTROID_INNER_RADIUS_d);
			};
			edge_normal.x = projendpt1.y - endpt1.y;
			edge_normal.y = endpt1.x - projendpt1.x;
			AreaMinor += (0.5*projendpt1.x + 0.5*endpt1.x)*edge_normal.x;

			edge_normal.x = projendpt0.y - projendpt1.y;
			edge_normal.y = projendpt1.x - projendpt0.x;
			AreaMinor += (0.5*projendpt1.x + 0.5*projendpt0.x)*edge_normal.x;
			// line between out-projected points


			if (info.flag == OUTERMOST)
			{
				// set outer value at Az = (r_here/r_outer)*Az_self
				// contrib to integral of Lap is [delta_edge * dAz/dr] = 
				// delta_edge*Az_self * (r here - r outer)/(r_outer^2 - r_outer*r_here)
				// = delta_edge*Az_self * (-1.0)/(r_outer)

				f64 delta_edge = (projendpt0 - projendpt1).modulus();

				Our_integral_Lap_Az_contrib_from_own_Az -= delta_edge / FRILL_CENTROID_OUTER_RADIUS_d;
			}
		};

		p_LapCoeffSelf[iVertex + BEGINNING_OF_CENTRAL] = Our_integral_Lap_Az_contrib_from_own_Az / AreaMinor;

		mincoeffself[threadIdx.x] = p_LapCoeffSelf[iVertex + BEGINNING_OF_CENTRAL];
		iMin[threadIdx.x] = iVertex + BEGINNING_OF_CENTRAL;
		// All vertices can count for this.

	}; // was thread in the first half of the block

	info = p_info[iMinor];

	long izNeighMinor[6];
	char szPBC[6];
	memcpy(izNeighMinor, p_izNeighMinor + iMinor * 6, sizeof(long) * 6);
	memcpy(szPBC, p_szPBCtriminor + iMinor * 6, sizeof(char) * 6);
	if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL)) {

		// Look at simulation.cpp
		// Treatment of FRILLS : 

		p_LapCoeffSelf[iMinor] = -1.0;
		// LapCoefftri[iMinor][3] = 1.0; // neighbour 0
	}
	else {

		f64 Our_integral_Lap_Az_contrib_from_own_Az = 0.0;
		f64 AreaMinor = 0.0;

		short iprev = 5; short inext, i = 0;
		if ((izNeighMinor[iprev] >= StartMinor) && (izNeighMinor[iprev] < EndMinor))
		{
			prevpos = shared_pos[izNeighMinor[iprev] - StartMinor];
		}
		else {
			if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
				(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL))
			{
				prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
			}
			else {
				prevpos = p_info[izNeighMinor[iprev]].pos;
			};
		};
		if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) prevpos = Clockwise_d*prevpos;
		if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) prevpos = Anticlockwise_d*prevpos;

		i = 0;
		if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
		{
			opppos = shared_pos[izNeighMinor[i] - StartMinor];
		}
		else {
			if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
				(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
			{
				opppos = shared_pos_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
			}
			else {
				opppos = p_info[izNeighMinor[i]].pos;
			};
		};
		if (szPBC[i] == ROTATE_ME_CLOCKWISE) opppos = Clockwise_d*opppos;
		if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) opppos = Anticlockwise_d*opppos;

#pragma unroll 
		for (i = 0; i < 6; i++)
		{
			inext = i + 1; if (inext > 5) inext = 0;

			if ((izNeighMinor[inext] >= StartMinor) && (izNeighMinor[inext] < EndMinor))
			{
				nextpos = shared_pos[izNeighMinor[inext] - StartMinor];
			}
			else {
				if ((izNeighMinor[inext] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[inext] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					nextpos = shared_pos_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					nextpos = p_info[izNeighMinor[inext]].pos;
				};
			};
			if (szPBC[inext] == ROTATE_ME_CLOCKWISE) nextpos = Clockwise_d*nextpos;
			if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) nextpos = Anticlockwise_d*nextpos;

			// New definition of endpoint of minor edge:
			f64_vec2 endpt0, endpt1, edge_normal, integ_grad_Az;
			endpt0 = THIRD * (prevpos + info.pos + opppos);
			endpt1 = THIRD * (nextpos + info.pos + opppos);

			edge_normal.x = endpt1.y - endpt0.y;
			edge_normal.y = endpt0.x - endpt1.x;

			// ______________________________________________________-

			//	Az_edge = SIXTH * (2.0*ourdata.Az + 2.0*oppAz + prevAz + nextAz);
			//	Our_integral_grad_Az += Az_edge * edge_normal;
			//	Our_integral_curl_Az += Az_edge * (endpt1 - endpt0); // looks anticlockwise
			//integ_grad_Az.x = 0.5*(
			//	(ourAz + nextAz)*(info.pos.y - nextpos.y)
			//	+ (prevAz + ourAz)*(prevpos.y - info.pos.y)
			//	+ (oppAz + prevAz)*(opppos.y - prevpos.y)
			//	+ (nextAz + oppAz)*(nextpos.y - opppos.y)
			//	);
			//integ_grad_Az.y = -0.5*( // notice minus
			//	(ourAz + nextAz)*(info.pos.x - nextpos.x)
			//	+ (prevAz + ourAz)*(prevpos.x - info.pos.x)
			//	+ (oppAz + prevAz)*(opppos.x - prevpos.x)
			//	+ (nextAz + oppAz)*(nextpos.x - opppos.x)
			//	);
			integ_grad_Az.x = 0.5*(prevpos.y - nextpos.y);

			integ_grad_Az.y = -0.5*(prevpos.x - nextpos.x);

			f64 area_quadrilateral = 0.5*(
				(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
				+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
				+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
				+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
				);

			// !
			// Wait a minute: if either prev, info or next is a frill, we need to add
			// to this one.
			// specifically it is worth (pos1.r/pos2.r) times self

			if ((prevpos.dot(prevpos) > 0.9999*0.9999*FRILL_CENTROID_OUTER_RADIUS_d*FRILL_CENTROID_OUTER_RADIUS_d))
			{

//			if (izNeighMinor[iprev] >= FirstOuterFrill_d) {
				f64 ratio = info.pos.modulus() / FRILL_CENTROID_OUTER_RADIUS_d;
				integ_grad_Az.x += 0.5*(opppos.y - info.pos.y)*ratio;
				integ_grad_Az.y -= 0.5*(opppos.x - info.pos.x)*ratio;
			};

			if (opppos.dot(opppos) > 0.9999*0.9999*FRILL_CENTROID_OUTER_RADIUS_d*FRILL_CENTROID_OUTER_RADIUS_d)
			{
			//if (izNeighMinor[i] >= FirstOuterFrill_d) {
				f64 ratio = info.pos.modulus() / FRILL_CENTROID_OUTER_RADIUS_d;
				integ_grad_Az.x += 0.5*(nextpos.y - prevpos.y)*ratio;
				integ_grad_Az.y -= 0.5*(nextpos.x - prevpos.x)*ratio;
			};
			if (nextpos.dot(nextpos) > 0.9999*0.9999*FRILL_CENTROID_OUTER_RADIUS_d*FRILL_CENTROID_OUTER_RADIUS_d)
			{
			//if (izNeighMinor[inext] >= FirstOuterFrill_d) {
				f64 ratio = info.pos.modulus() / FRILL_CENTROID_OUTER_RADIUS_d;
				integ_grad_Az.x += 0.5*(info.pos.y - opppos.y)*ratio;
				integ_grad_Az.y -= 0.5*(info.pos.x - opppos.x)*ratio;
			};
			
			//f64_vec2 grad_Az = integ_grad_Az / area_quadrilateral;
			Our_integral_Lap_Az_contrib_from_own_Az += integ_grad_Az.dot(edge_normal) / area_quadrilateral;

			AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;

			endpt0 = endpt1;
			prevpos = opppos;
			opppos = nextpos;
			iprev = i;
			// There is an even quicker way which is to rotate pointers. No memcpy needed.
		};

		p_LapCoeffSelf[iMinor] = Our_integral_Lap_Az_contrib_from_own_Az / AreaMinor;

		if (p_LapCoeffSelf[iMinor] < mincoeffself[threadIdx.x])
		{
			mincoeffself[threadIdx.x] = p_LapCoeffSelf[iMinor];
			iMin[threadIdx.x] = iMinor;
		};
	};

	__syncthreads();


	int s = blockDim.x;
	int k = s / 2;

	while (s != 1) {
		if (threadIdx.x < k)
		{
			if (mincoeffself[threadIdx.x] > mincoeffself[threadIdx.x + k])
			{
				mincoeffself[threadIdx.x] = mincoeffself[threadIdx.x + k];
				iMin[threadIdx.x] = iMin[threadIdx.x + k];
			}			
		};
		__syncthreads();

		// Modify for case blockdim not 2^n:
		if ((s % 2 == 1) && (threadIdx.x == k - 1)) {
			
			if (mincoeffself[threadIdx.x] > mincoeffself[threadIdx.x + s-1])
			{
				mincoeffself[threadIdx.x] = mincoeffself[threadIdx.x + s-1];
				iMin[threadIdx.x] = iMin[threadIdx.x + s-1];
			}
		};
		// In case k == 81, add [39] += [80]
		// Otherwise we only get to 39+40=79.
		s = k;
		k = s / 2;
		__syncthreads();
	};

	if (threadIdx.x == 0)
	{
		p_min_array[blockIdx.x] = mincoeffself[threadIdx.x];
		p_min_index[blockIdx.x] = iMin[threadIdx.x];
	}	
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

__global__ void kernelCreate_pressure_gradT_and_gradA_LapA_CurlA_minor(

	structural * __restrict__ p_info_minor,
	T3 * __restrict__ p_T_minor,
	AAdot * __restrict__ p_AAdot,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtriminor,

	LONG3 * __restrict__ p_who_am_I_to_corners,
	LONG3 * __restrict__ p_tricornerindex,

	f64_vec3 * __restrict__ p_MAR_neut,
	f64_vec3 * __restrict__ p_MAR_ion,
	f64_vec3 * __restrict__ p_MAR_elec,
	ShardModel * __restrict__ p_n_shards,
	nvals * __restrict__ p_n_minor, // Just so we can handle insulator

	f64_vec2 * __restrict__ p_GradTe,
	f64_vec2 * __restrict__ p_GradAz,
	f64 * __restrict__ p_LapAz,

	f64 * __restrict__ ROCAzduetoAdvection,
	f64 * __restrict__ ROCAzdotduetoAdvection,
	f64_vec2 * __restrict__ p_v_overall_minor,

	f64_vec3 * __restrict__ p_B,
	f64 * __restrict__ p_AreaMinor
)
{
	// Getting this down to 8 vars we could have 512 threads (12 vars/thread total with vertex vars)
	// Down to 6 -> 9 total -> 600+ threads
	// Worry later.

	__shared__ T2 shared_T[threadsPerTileMinor];
	__shared__ f64 shared_Az[threadsPerTileMinor];
	__shared__ f64 shared_Azdot[threadsPerTileMinor];
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];

	__shared__ ShardModel shared_n_shards[threadsPerTileMajor];
	// Problem: we only have room for 1 at a time. Have to run again with n_n. Too bad.
	// Live with it and push through.
	// This applies to both vertices and triangles. And putting in L1 unshared is not better.
	// We can imagine doing it some other way but using shards is true to the design that was created on CPU.
	// Of course this means we'd be better off putting
	// We could also argue that with shards for n_ion in memory we are better off doing an overwrite and doing stuff for nv also.
	// never mind that for now

	__shared__ T2 shared_T_verts[threadsPerTileMajor];
	__shared__ f64 shared_Az_verts[threadsPerTileMajor];
	__shared__ f64 shared_Azdot_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajor];

	// There is a good argument for splitting out A,Adot to a separate routine.
	// That way we could have 10.5 => 585 ie 576 = 288*2 threads.

	// Here we got (2+1+1+2)*1.5 = 9 , + 6.5 = 15.5 -> 384 minor threads max.

	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	long const iVertex = threadsPerTileMajor*blockIdx.x + threadIdx.x; // only meaningful threadIdx.x < threadsPerTileMajor

	long const StartMinor = threadsPerTileMinor*blockIdx.x;
	long const StartMajor = threadsPerTileMajor*blockIdx.x;
	long const EndMinor = StartMinor + threadsPerTileMinor;
	long const EndMajor = StartMajor + threadsPerTileMajor;

	shared_pos[threadIdx.x] = p_info_minor[iMinor].pos;
	{
		AAdot temp = p_AAdot[iMinor];
		shared_Az[threadIdx.x] = temp.Az;
		shared_Azdot[threadIdx.x] = temp.Azdot;
	}
	{
		T3 T_ = p_T_minor[iMinor];
		shared_T[threadIdx.x].Te = T_.Te;
		shared_T[threadIdx.x].Ti = T_.Ti;
	}

	structural info;
	if (threadIdx.x < threadsPerTileMajor) {
		info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
		shared_pos_verts[threadIdx.x] = info.pos;
		AAdot temp = p_AAdot[iVertex + BEGINNING_OF_CENTRAL];
		shared_Az_verts[threadIdx.x] = temp.Az;
		shared_Azdot_verts[threadIdx.x] = temp.Azdot;
		T3 T_ = p_T_minor[iVertex + BEGINNING_OF_CENTRAL];
		shared_T_verts[threadIdx.x].Te = T_.Te;
		shared_T_verts[threadIdx.x].Ti = T_.Ti; // MOVED THIS OUT OF the following branch to see it match CPU
		if (info.flag == DOMAIN_VERTEX) {
			memcpy(&(shared_n_shards[threadIdx.x]), &(p_n_shards[iVertex]), sizeof(ShardModel)); // + 13
		}
		else {
			// save several bus trips;
			// it still manages to coalesce, let's hope, because ShardModel is 13 doubles not 12.
			//shared_T_verts[threadIdx.x].Te = 0.0;
			//shared_T_verts[threadIdx.x].Ti = 0.0;
			memset(&(shared_n_shards[threadIdx.x]), 0, sizeof(ShardModel)); // + 13
		};
	};

	__syncthreads();

	f64 ourAz, oppAz, prevAz, nextAz;
	f64 ourAzdot, oppAzdot, prevAzdot, nextAzdot;
	f64_vec2 opppos, prevpos, nextpos;
	T2 oppT, prevT, nextT;
	//nvals our_n, opp_n, prev_n, next_n;
	f64_vec2 Our_integral_curl_Az, Our_integral_grad_Az, Our_integral_grad_Azdot, Our_integral_grad_Te;
	f64 Our_integral_Lap_Az;

	if (threadIdx.x < threadsPerTileMajor) {
		Our_integral_curl_Az.x = 0.0;
		Our_integral_curl_Az.y = 0.0;
		Our_integral_grad_Az.x = 0.0;
		Our_integral_grad_Az.y = 0.0;
		Our_integral_grad_Azdot.x = 0.0;
		Our_integral_grad_Azdot.y = 0.0;
		Our_integral_grad_Te.x = 0.0;
		Our_integral_grad_Te.y = 0.0;
		Our_integral_Lap_Az = 0.0;
		f64 AreaMinor = 0.0;
		f64_vec3 MAR_ion, MAR_elec;
		memcpy(&(MAR_ion), &(p_MAR_ion[iVertex + BEGINNING_OF_CENTRAL]), sizeof(f64_vec3));
		memcpy(&(MAR_elec), &(p_MAR_elec[iVertex + BEGINNING_OF_CENTRAL]), sizeof(f64_vec3));

		long izTri[MAXNEIGH];
		char szPBC[MAXNEIGH];
		short tri_len = info.neigh_len;

		memcpy(izTri, p_izTri + iVertex*MAXNEIGH, MAXNEIGH * sizeof(long));
		memcpy(szPBC, p_szPBC + iVertex*MAXNEIGH, MAXNEIGH * sizeof(char));

		ourAz = shared_Az_verts[threadIdx.x];
		ourAzdot = shared_Azdot_verts[threadIdx.x];

		if (info.flag == DOMAIN_VERTEX) {

			short iprev = tri_len - 1;
			if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMinor))
			{
				prevT = shared_T[izTri[iprev] - StartMinor];
				prevAz = shared_Az[izTri[iprev] - StartMinor];
				prevAzdot = shared_Azdot[izTri[iprev] - StartMinor];
				prevpos = shared_pos[izTri[iprev] - StartMinor];
			}
			else {
				T3 prev_T = p_T_minor[izTri[iprev]];
				prevT.Te = prev_T.Te; prevT.Ti = prev_T.Ti;
				AAdot temp = p_AAdot[izTri[iprev]];
				prevAz = temp.Az;
				prevAzdot = temp.Azdot;
				prevpos = p_info_minor[izTri[iprev]].pos;
			}
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) prevpos = Clockwise_d*prevpos;
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) prevpos = Anticlockwise_d*prevpos;

			short inext, i = 0;
			if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
			{
				oppT = shared_T[izTri[i] - StartMinor];
				oppAz = shared_Az[izTri[i] - StartMinor];
				oppAzdot = shared_Azdot[izTri[i] - StartMinor];
				opppos = shared_pos[izTri[i] - StartMinor];
			}
			else {
				T3 opp_T = p_T_minor[izTri[i]];
				oppT.Te = opp_T.Te; oppT.Ti = opp_T.Ti;
				AAdot temp = p_AAdot[izTri[i]];
				oppAz = temp.Az;
				oppAzdot = temp.Azdot;
				opppos = p_info_minor[izTri[i]].pos;
			}
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) opppos = Clockwise_d*opppos;
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) opppos = Anticlockwise_d*opppos;

			// Think carefully: DOMAIN vertex cases for n,T ...

			f64 n0 = THIRD*(shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[iprev] + shared_n_shards[threadIdx.x].n_cent);
			f64_vec2 endpt1, endpt0 = THIRD * (info.pos + opppos + prevpos);

			short iend = tri_len;
			f64_vec2 projendpt0, edge_normal;
			if ((info.flag == INNERMOST) || (info.flag == OUTERMOST)) {
				iend = tri_len - 2;
				if (info.flag == OUTERMOST) {
					endpt0.project_to_radius(projendpt0, FRILL_CENTROID_OUTER_RADIUS_d); // back of cell for Lap purposes
				}
				else {
					endpt0.project_to_radius(projendpt0, FRILL_CENTROID_INNER_RADIUS_d); // back of cell for Lap purposes
				}
				edge_normal.x = endpt0.y - projendpt0.y;
				edge_normal.y = projendpt0.x - endpt0.x;
				AreaMinor += (0.5*projendpt0.x + 0.5*endpt0.x)*edge_normal.x;
			};

			for (i = 0; i < iend; i++)
			{
				// Tri 0 is anticlockwise of neighbour 0, we think
				inext = i + 1; if (inext >= tri_len) inext = 0;

				if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
				{
					nextT = shared_T[izTri[inext] - StartMinor];
					nextAz = shared_Az[izTri[inext] - StartMinor];
					nextAzdot = shared_Azdot[izTri[inext] - StartMinor];
					nextpos = shared_pos[izTri[inext] - StartMinor];
				}
				else {
					T3 next_T = p_T_minor[izTri[inext]];
					nextT.Te = next_T.Te; nextT.Ti = next_T.Ti;
					AAdot temp = p_AAdot[izTri[inext]];
					nextAz = temp.Az;
					nextAzdot = temp.Azdot;
					nextpos = p_info_minor[izTri[inext]].pos;
				}
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) nextpos = Clockwise_d*nextpos;
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) nextpos = Anticlockwise_d*nextpos;

				endpt1 = THIRD * (nextpos + info.pos + opppos);
				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

				// ______________________________________________________-

				//	Az_edge = SIXTH * (2.0*ourdata.Az + 2.0*oppAz + prevAz + nextAz);
				//	Our_integral_grad_Az += Az_edge * edge_normal;
				//	Our_integral_curl_Az += Az_edge * (endpt1 - endpt0); // looks anticlockwise
				//	pData[iDestTri].B -= Az_edge * (endpt1 - endpt0); // MUST DIVIDE BY AREA
				f64_vec2 integ_grad_Az;

				integ_grad_Az.x = 0.5*(
					(ourAz + nextAz)*(info.pos.y - nextpos.y)
					+ (prevAz + ourAz)*(prevpos.y - info.pos.y)
					+ (oppAz + prevAz)*(opppos.y - prevpos.y)
					+ (nextAz + oppAz)*(nextpos.y - opppos.y)
					);
				integ_grad_Az.y = -0.5*( // notice minus
					(ourAz + nextAz)*(info.pos.x - nextpos.x)
					+ (prevAz + ourAz)*(prevpos.x - info.pos.x)
					+ (oppAz + prevAz)*(opppos.x - prevpos.x)
					+ (nextAz + oppAz)*(nextpos.x - opppos.x)
					);
				f64 area_quadrilateral = 0.5*(
					(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
					+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
					+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
					+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
					);
				//f64_vec2 grad_Az = integ_grad_Az / area_quadrilateral;
				Our_integral_Lap_Az += integ_grad_Az.dot(edge_normal) / area_quadrilateral;

				T2 T0, T1;
				f64 n1;
				T0.Te = THIRD* (prevT.Te + shared_T_verts[threadIdx.x].Te + oppT.Te);
				T1.Te = THIRD * (nextT.Te + shared_T_verts[threadIdx.x].Te + oppT.Te);
				T0.Ti = THIRD * (prevT.Ti + shared_T_verts[threadIdx.x].Ti + oppT.Ti);
				T1.Ti = THIRD * (nextT.Ti + shared_T_verts[threadIdx.x].Ti + oppT.Ti);
				n1 = THIRD*(shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[inext] + shared_n_shards[threadIdx.x].n_cent);

				// So this is pretty stupid ---
				// If shardmodel went for flat then we have decided that there is no pressure gradient affecting v here.
				// Mind you we didn't expect it to be flat nearly as often as it is flat.
				// Think carefully about what pressure we want to feel.
				// It makes a kind of sense if you have a cliff of density then you feel it in the triangle in between.
				// But that won't push points apart. It just sends stuff through the wall. 

				//		It's a shame we can't just use actual n values to infer gradient over a region. 
				//		It probably creates wobbles in v as well, because if we move fast particles at edge then we leave
				//		Behind a still-lower v in the vertex-centered minor.
				//		The scheme is kind of skewiffifying.

				// Assume neighs 0,1 are relevant to border with tri 0 minor

				// To get integral grad we add the averages along the edges times edge_normals
				MAR_ion -= Make3(0.5*(n0 * T0.Ti + n1 * T1.Ti)*over_m_i*edge_normal, 0.0);
				MAR_elec -= Make3(0.5*(n0 * T0.Te + n1 * T1.Te)*over_m_e*edge_normal, 0.0);

				//				if (iVertex == VERT1) {
				//					printf("GPUpressure %d MAR_ion.x %1.12E contrib.x %1.12E n0 %1.12E Ti0 %1.9E n1 %1.9E Ti1 %1.9E edge_normal.x %1.12E \n",
				//						CHOSEN, MAR_ion.x,
				//						-0.5*(n0*T0.Ti + n1 * T1.Ti)*over_m_i*edge_normal.x,
				//						n0, T0.Ti, n1, T1.Ti, edge_normal.x);
				//				}

				Our_integral_grad_Te += 0.5*(T0.Te + T1.Te) * edge_normal;

				// if (iVertex + BEGINNING_OF_CENTRAL == CHOSEN)
				//	printf("GPU %d : GradTe contrib %1.14E %1.14E Te %1.14E opp %1.14E next %1.14E prev %1.14E edge_normal %1.14E %1.14E\n", iVertex + BEGINNING_OF_CENTRAL,
				//	0.5*(T0.Te + T1.Te) * edge_normal.x,
				//0.5*(T0.Te + T1.Te) * edge_normal.y, 
				//	shared_T_verts[threadIdx.x].Te, oppT.Te, nextT.Te, prevT.Te,
				//edge_normal.x, edge_normal.y);

				f64 Az_edge = SIXTH * (2.0*ourAz + 2.0*oppAz + prevAz + nextAz);
				f64 Azdot_edge = SIXTH * (2.0*ourAzdot + 2.0*oppAzdot + prevAzdot + nextAzdot);
				Our_integral_grad_Azdot += Azdot_edge * edge_normal;
				Our_integral_grad_Az += Az_edge * edge_normal;
				Our_integral_curl_Az += Az_edge * (endpt1 - endpt0);

				// Missing a factor of 3 possibly?
				// ??????????????????????????????????????????????????????????????


				//	if (Az_edge != Az_edge) 
				//		printf("GPU vert %d Az_edge %1.14E oppAz %1.14E endpt1 %1.14E %1.14E Integ_curl %1.14E %1.14E\n",
				//			iVertex, Az_edge, oppAz, endpt1.x,endpt1.y, Our_integral_curl_Az.x, Our_integral_curl_Az.y
				//		);

				AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;
				endpt0 = endpt1;
				n0 = n1;

				prevpos = opppos;
				prevAz = oppAz;
				prevAzdot = oppAzdot;
				prevT = oppT;

				opppos = nextpos;
				oppAz = nextAz;
				oppAzdot = nextAzdot;
				oppT = nextT;
			}; // next i

			   //if ((info.flag == INNERMOST) || (info.flag == OUTERMOST)) {

			   //	// This will never happen because we just asked info.flag == DOMAIN_VERTEX !!

			   //	// Now add on the final sides to give area:

			   //	//    3     4
			   //	//     2 1 0
			   //	// endpt0=endpt1 is now the point north of edge facing 2 anyway.
			   //	f64_vec2 projendpt1;

			   //	if (info.flag == OUTERMOST) {
			   //		endpt1.project_to_radius(projendpt1, FRILL_CENTROID_OUTER_RADIUS_d);
			   //	}
			   //	else {
			   //		endpt1.project_to_radius(projendpt1, FRILL_CENTROID_INNER_RADIUS_d);
			   //	};
			   //	edge_normal.x = projendpt1.y - endpt1.y;
			   //	edge_normal.y = endpt1.x - projendpt1.x;
			   //	AreaMinor += (0.5*projendpt1.x + 0.5*endpt1.x)*edge_normal.x;

			   //	edge_normal.x = projendpt0.y - projendpt1.y;
			   //	edge_normal.y = projendpt1.x - projendpt0.x;
			   //	AreaMinor += (0.5*projendpt1.x + 0.5*projendpt0.x)*edge_normal.x;
			   //	// line between out-projected points
			   //};

			p_GradAz[iVertex + BEGINNING_OF_CENTRAL] = Our_integral_grad_Az / AreaMinor;
			p_LapAz[iVertex + BEGINNING_OF_CENTRAL] = Our_integral_Lap_Az / AreaMinor;
			p_GradTe[iVertex + BEGINNING_OF_CENTRAL] = Our_integral_grad_Te / AreaMinor;
			p_B[iVertex + BEGINNING_OF_CENTRAL] = Make3(Our_integral_curl_Az / AreaMinor, BZ_CONSTANT);
			p_AreaMinor[iVertex + BEGINNING_OF_CENTRAL] = AreaMinor;
			//	if (iVertex + BEGINNING_OF_CENTRAL == CHOSEN) printf("Our_integral_grad_Te.x %1.14E AreaMinor %1.14E\n\n",
			//	Our_integral_grad_Te.x, AreaMinor);

			// wow :
			f64_vec2 overall_v_ours = p_v_overall_minor[iVertex + BEGINNING_OF_CENTRAL];
			ROCAzduetoAdvection[iVertex + BEGINNING_OF_CENTRAL] = overall_v_ours.dot(Our_integral_grad_Az / AreaMinor);
			ROCAzdotduetoAdvection[iVertex + BEGINNING_OF_CENTRAL] = overall_v_ours.dot(Our_integral_grad_Azdot / AreaMinor);

			// No neutral stuff in this kernel, momrates should be set now:
			memcpy(p_MAR_ion + iVertex + BEGINNING_OF_CENTRAL, &MAR_ion, sizeof(f64_vec3));
			memcpy(p_MAR_elec + iVertex + BEGINNING_OF_CENTRAL, &MAR_elec, sizeof(f64_vec3));

		}
		else {
			// NOT domain vertex: Do Az, Azdot only:			
			short iprev = tri_len - 1;
			if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMinor))
			{
				prevAz = shared_Az[izTri[iprev] - StartMinor];
				prevAzdot = shared_Azdot[izTri[iprev] - StartMinor];
				prevpos = shared_pos[izTri[iprev] - StartMinor];
			}
			else {
				AAdot temp = p_AAdot[izTri[iprev]];
				prevAz = temp.Az;
				prevAzdot = temp.Azdot;
				prevpos = p_info_minor[izTri[iprev]].pos;
			};
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) prevpos = Clockwise_d*prevpos;
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) prevpos = Anticlockwise_d*prevpos;
			short inext, i = 0;
			if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
			{
				oppAz = shared_Az[izTri[i] - StartMinor];
				oppAzdot = shared_Azdot[izTri[i] - StartMinor];
				opppos = shared_pos[izTri[i] - StartMinor];
			}
			else {
				AAdot temp = p_AAdot[izTri[i]];
				oppAz = temp.Az;
				oppAzdot = temp.Azdot;
				opppos = p_info_minor[izTri[i]].pos;
			}
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) opppos = Clockwise_d*opppos;
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) opppos = Anticlockwise_d*opppos;

			f64 n0 = THIRD*(shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[iprev] + shared_n_shards[threadIdx.x].n_cent);
			f64_vec2 endpt0 = THIRD * (info.pos + opppos + prevpos);
			f64_vec2 endpt1;
			short iend = tri_len;
			f64_vec2 projendpt0, edge_normal;
			if ((info.flag == INNERMOST) || (info.flag == OUTERMOST)) {
				iend = tri_len - 2;
				if (info.flag == OUTERMOST) {
					endpt0.project_to_radius(projendpt0, FRILL_CENTROID_OUTER_RADIUS_d); // back of cell for Lap purposes
				}
				else {
					endpt0.project_to_radius(projendpt0, FRILL_CENTROID_INNER_RADIUS_d); // back of cell for Lap purposes
				}
				edge_normal.x = endpt0.y - projendpt0.y;
				edge_normal.y = projendpt0.x - endpt0.x;
				AreaMinor += (0.5*projendpt0.x + 0.5*endpt0.x)*edge_normal.x;

			};
			for (i = 0; i < iend; i++)
			{
				// Tri 0 is anticlockwise of neighbour 0, we think
				inext = i + 1; if (inext >= tri_len) inext = 0;

				if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
				{
					nextAz = shared_Az[izTri[inext] - StartMinor];
					nextAzdot = shared_Azdot[izTri[inext] - StartMinor];
					nextpos = shared_pos[izTri[inext] - StartMinor];
				}
				else {
					AAdot temp = p_AAdot[izTri[inext]];
					nextAz = temp.Az;
					nextAzdot = temp.Azdot;
					nextpos = p_info_minor[izTri[inext]].pos;
				}
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) nextpos = Clockwise_d*nextpos;
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) nextpos = Anticlockwise_d*nextpos;

				endpt1 = THIRD * (nextpos + info.pos + opppos);
				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

				// ______________________________________________________-
				f64_vec2 integ_grad_Az;
				//	Az_edge = SIXTH * (2.0*ourdata.Az + 2.0*oppAz + prevAz + nextAz);
				//	Our_integral_grad_Az += Az_edge * edge_normal;
				//	Our_integral_curl_Az += Az_edge * (endpt1 - endpt0); // looks anticlockwise
				//	pData[iDestTri].B -= Az_edge * (endpt1 - endpt0); // MUST DIVIDE BY AREA
				integ_grad_Az.x = 0.5*(
					(ourAz + nextAz)*(info.pos.y - nextpos.y)
					+ (prevAz + ourAz)*(prevpos.y - info.pos.y)
					+ (oppAz + prevAz)*(opppos.y - prevpos.y)
					+ (nextAz + oppAz)*(nextpos.y - opppos.y)
					);
				integ_grad_Az.y = -0.5*( // notice minus
					(ourAz + nextAz)*(info.pos.x - nextpos.x)
					+ (prevAz + ourAz)*(prevpos.x - info.pos.x)
					+ (oppAz + prevAz)*(opppos.x - prevpos.x)
					+ (nextAz + oppAz)*(nextpos.x - opppos.x)
					);
				f64 area_quadrilateral = 0.5*(
					(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
					+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
					+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
					+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
					);
				//f64_vec2 grad_Az = integ_grad_Az / area_quadrilateral;
				Our_integral_Lap_Az += integ_grad_Az.dot(edge_normal) / area_quadrilateral;

				// To get integral grad we add the averages along the edges times edge_normals
				//			f64 Az_edge = SIXTH * (2.0*ourAz + 2.0*oppAz + prevAz + nextAz);
				//			f64 Azdot_edge = SIXTH * (2.0*ourAzdot + 2.0*oppAzdot + prevAzdot + nextAzdot);
				//			Our_integral_grad_Azdot += Azdot_edge * edge_normal;
				//			Our_integral_grad_Az += Az_edge * edge_normal;
				//			Our_integral_curl_Az += Az_edge * (endpt1 - endpt0);

				AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;

				endpt0 = endpt1;
				prevpos = opppos;
				prevAz = oppAz;
				prevAzdot = oppAzdot;
				opppos = nextpos;
				oppAz = nextAz;
				oppAzdot = nextAzdot;
			}; // next i

			if ((info.flag == INNERMOST) || (info.flag == OUTERMOST)) {
				// Now add on the final sides to give area:
				//    3     4
				//     2 1 0
				// endpt0=endpt1 is now the point north of edge facing 2 anyway.
				f64_vec2 projendpt1;
				if (info.flag == OUTERMOST) {
					endpt1.project_to_radius(projendpt1, FRILL_CENTROID_OUTER_RADIUS_d);
				}
				else {
					endpt1.project_to_radius(projendpt1, FRILL_CENTROID_INNER_RADIUS_d);
				};
				edge_normal.x = projendpt1.y - endpt1.y;
				edge_normal.y = endpt1.x - projendpt1.x;
				AreaMinor += (0.5*projendpt1.x + 0.5*endpt1.x)*edge_normal.x;

				edge_normal.x = projendpt0.y - projendpt1.y;
				edge_normal.y = projendpt1.x - projendpt0.x;
				AreaMinor += (0.5*projendpt1.x + 0.5*projendpt0.x)*edge_normal.x;
				// line between out-projected points
			};

			p_GradAz[iVertex + BEGINNING_OF_CENTRAL] = Our_integral_grad_Az / AreaMinor; // 0,0
			p_LapAz[iVertex + BEGINNING_OF_CENTRAL] = Our_integral_Lap_Az / AreaMinor;
			p_B[iVertex + BEGINNING_OF_CENTRAL] = Make3(Our_integral_curl_Az / AreaMinor, BZ_CONSTANT); // 0,0, BZ
			p_AreaMinor[iVertex + BEGINNING_OF_CENTRAL] = AreaMinor;

			ROCAzduetoAdvection[iVertex + BEGINNING_OF_CENTRAL] = 0.0;
			ROCAzdotduetoAdvection[iVertex + BEGINNING_OF_CENTRAL] = 0.0;

			p_GradTe[iVertex + BEGINNING_OF_CENTRAL] = Vector2(0.0, 0.0);

		}; // // was it domain vertex or Az-only

	};//  if (threadIdx.x < threadsPerTileMajor) 
	  // This branching is itself a good argument for doing Az in ITS own separate routine with no need for n_shard.

	  // __syncthreads(); // end of first vertex part
	  // Do we need syncthreads? Not overwriting any shared data here...

	  // now the minor with n_ion part:
	info = p_info_minor[iMinor];
	ourAz = shared_Az[threadIdx.x];

	long izNeighMinor[6];
	char szPBC[6];
	memcpy(izNeighMinor, p_izNeighMinor + iMinor * 6, sizeof(long) * 6);
	memcpy(szPBC, p_szPBCtriminor + iMinor * 6, sizeof(char) * 6);

	//	T2 prevT, nextT, oppT;
	//f64 prevAz, nextAz, oppAz, ourAz;
	//f64 prevAzdot, nextAzdot, oppAzdot, ourAzdot;

	f64_vec3 MAR_ion, MAR_elec;
	// this is not a clever way of doing it. Want more careful.

	if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL)) {
		if ((izNeighMinor[3] >= StartMinor) && (izNeighMinor[3] < EndMinor))
		{
			oppAz = shared_Az[izNeighMinor[3] - StartMinor];
		}
		else {

			AAdot temp = p_AAdot[izNeighMinor[3]];
			oppAz = temp.Az;
		};
		p_LapAz[iMinor] = oppAz - ourAz;

		ROCAzduetoAdvection[iMinor] = 0.0;
		ROCAzdotduetoAdvection[iMinor] = 0.0;
		p_GradAz[iMinor] = Vector2(0.0, 0.0);
		memset(&(p_B[iMinor]), 0, sizeof(f64_vec3));
		p_GradTe[iMinor] = Vector2(0.0, 0.0);
		p_AreaMinor[iMinor] = 1.0e-12;
		memset(&(p_MAR_ion[iMinor]), 0, sizeof(f64_vec3));
		memset(&(p_MAR_elec[iMinor]), 0, sizeof(f64_vec3));
	}
	else {
		Our_integral_curl_Az.x = 0.0;
		Our_integral_curl_Az.y = 0.0;
		Our_integral_grad_Azdot.x = 0.0;
		Our_integral_grad_Azdot.y = 0.0;
		Our_integral_grad_Az.x = 0.0;
		Our_integral_grad_Az.y = 0.0;
		Our_integral_grad_Te.x = 0.0;
		Our_integral_grad_Te.y = 0.0;
		Our_integral_Lap_Az = 0.0;
		f64 AreaMinor = 0.0;
		short iprev, inext, i;
		if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS)) {

			memcpy(&MAR_ion, p_MAR_ion + iMinor, sizeof(f64_vec3));
			memcpy(&MAR_elec, p_MAR_elec + iMinor, sizeof(f64_vec3));

			iprev = 5;
			i = 0;
			if ((izNeighMinor[iprev] >= StartMinor) && (izNeighMinor[iprev] < EndMinor))
			{
				prevAz = shared_Az[izNeighMinor[iprev] - StartMinor];
				prevpos = shared_pos[izNeighMinor[iprev] - StartMinor];
				prevT = shared_T[izNeighMinor[iprev] - StartMinor];
				prevAzdot = shared_Azdot[izNeighMinor[iprev] - StartMinor];
			}
			else {
				if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					prevAz = shared_Az_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
					prevAzdot = shared_Azdot_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
					prevT = shared_T_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
					prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					T3 prev_T = p_T_minor[izNeighMinor[iprev]];
					prevT.Te = prev_T.Te; prevT.Ti = prev_T.Ti;
					AAdot temp = p_AAdot[izNeighMinor[iprev]];
					prevAz = temp.Az;
					prevAzdot = temp.Azdot;
					prevpos = p_info_minor[izNeighMinor[iprev]].pos;
				};
			};
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) prevpos = Clockwise_d*prevpos;
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) prevpos = Anticlockwise_d*prevpos;

			i = 0;
			if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
			{
				oppAz = shared_Az[izNeighMinor[i] - StartMinor];
				oppT = shared_T[izNeighMinor[i] - StartMinor];
				oppAzdot = shared_Azdot[izNeighMinor[i] - StartMinor];
				opppos = shared_pos[izNeighMinor[i] - StartMinor];
			}
			else {
				if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					oppAz = shared_Az_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
					oppAzdot = shared_Azdot_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
					oppT = shared_T_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
					opppos = shared_pos_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					T3 opp_T = p_T_minor[izNeighMinor[i]];
					oppT.Te = opp_T.Te; oppT.Ti = opp_T.Ti;
					AAdot temp = p_AAdot[izNeighMinor[i]];
					oppAz = temp.Az;
					oppAzdot = temp.Azdot;
					opppos = p_info_minor[izNeighMinor[i]].pos;
				};
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) opppos = Clockwise_d*opppos;
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) opppos = Anticlockwise_d*opppos;

			long who_am_I_to_corners[3];
			memcpy(who_am_I_to_corners, &(p_who_am_I_to_corners[iMinor]), sizeof(long) * 3);

			LONG3 cornerindex = p_tricornerindex[iMinor];
			// each corner we want to pick up 3 values off n_shards, as well as n_cent.
			// The three values will not always be contiguous!!!

			// Let's make life easier and load up an array of 6 n's beforehand.
			f64 n_array[6];
			f64 n0, n1;
			// indexminor sequence:
			// 0 = corner 0
			// 1 = neighbour 2
			// 2 = corner 1
			// 3 = neighbour 0
			// 4 = corner 2
			// 5 = neighbour 1

			short who_am_I = who_am_I_to_corners[0];
			short tri_len = p_info_minor[cornerindex.i1 + BEGINNING_OF_CENTRAL].neigh_len;

			if ((cornerindex.i1 >= StartMajor) && (cornerindex.i1 < EndMajor))
			{
				short who_prev = who_am_I - 1;
				if (who_prev < 0) who_prev = tri_len - 1;
				short who_next = who_am_I + 1;
				if (who_next == tri_len) who_next = 0;

				// Worry about pathological cases later.
				// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

				// Pathological case: OUTERMOST vertex where neigh_len is not correct to take as == tri_len

				// !

				// ///////////////////////////////////////////////////////////////////////////////////////////
				// [0] is on our clockwise side rel to [1]. That means it is anticlockwise for the vertex. 
				// That means we interpolate with the value from next tri around.
				n_array[0] = THIRD*(shared_n_shards[cornerindex.i1 - StartMajor].n[who_next]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n_cent);
				n_array[1] = THIRD*(shared_n_shards[cornerindex.i1 - StartMajor].n[who_prev]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n_cent);
			}
			else {
				// comes from elsewhere
				f64 ncent = p_n_shards[cornerindex.i1].n_cent;
				short who_prev = who_am_I - 1;
				if (who_prev < 0) {
					who_prev = tri_len - 1;
					f64_vec2 temp;
					memcpy(&temp, p_n_shards[cornerindex.i1].n, sizeof(f64_vec2));
					n_array[0] = THIRD*(temp.x + temp.y + ncent);
					n_array[1] = THIRD*(p_n_shards[cornerindex.i1].n[who_prev] + temp.x + ncent);
				}
				else {
					short who_next = who_am_I + 1;
					if (who_next == tri_len) {
						f64_vec2 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i1].n[who_prev]), sizeof(f64_vec2));
						n_array[0] = THIRD*(p_n_shards[cornerindex.i1].n[0] + temp.y + ncent);
						n_array[1] = THIRD*(temp.x + temp.y + ncent);
					}
					else {
						// typical case
						f64_vec3 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i1].n[who_prev]), sizeof(f64) * 3);
						n_array[0] = THIRD*(temp.z + temp.y + ncent);
						n_array[1] = THIRD*(temp.x + temp.y + ncent);
					};
				};
			}

			who_am_I = who_am_I_to_corners[1];
			tri_len = p_info_minor[cornerindex.i2 + BEGINNING_OF_CENTRAL].neigh_len;

			if ((cornerindex.i2 >= StartMajor) && (cornerindex.i2 < EndMajor))
			{
				short who_prev = who_am_I - 1;
				if (who_prev < 0) who_prev = tri_len - 1;
				short who_next = who_am_I + 1;
				if (who_next == tri_len) who_next = 0;
				// Worry about pathological cases later.

				n_array[2] = THIRD*(shared_n_shards[cornerindex.i2 - StartMajor].n[who_next]
					+ shared_n_shards[cornerindex.i2 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i2 - StartMajor].n_cent);
				n_array[3] = THIRD*(shared_n_shards[cornerindex.i2 - StartMajor].n[who_prev]
					+ shared_n_shards[cornerindex.i2 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i2 - StartMajor].n_cent);
			}
			else {
				// comes from elsewhere
				f64 ncent = p_n_shards[cornerindex.i2].n_cent;
				short who_prev = who_am_I - 1;
				if (who_prev < 0) {
					who_prev = tri_len - 1;
					f64_vec2 temp;
					memcpy(&temp, p_n_shards[cornerindex.i2].n, sizeof(f64_vec2));
					n_array[2] = THIRD*(temp.x + temp.y + ncent);
					n_array[3] = THIRD*(p_n_shards[cornerindex.i2].n[who_prev] + temp.x + ncent);
				}
				else {
					short who_next = who_am_I + 1;
					if (who_next == tri_len) {
						f64_vec2 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i2].n[who_prev]), sizeof(f64_vec2));
						n_array[2] = THIRD*(p_n_shards[cornerindex.i2].n[0] + temp.y + ncent);
						n_array[3] = THIRD*(temp.x + temp.y + ncent);
					}
					else {
						// typical case
						f64_vec3 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i2].n[who_prev]), sizeof(f64) * 3);
						n_array[2] = THIRD*(temp.z + temp.y + ncent);
						n_array[3] = THIRD*(temp.x + temp.y + ncent);
					};
				};
			}

			who_am_I = who_am_I_to_corners[2];
			tri_len = p_info_minor[cornerindex.i3 + BEGINNING_OF_CENTRAL].neigh_len;

			if ((cornerindex.i3 >= StartMajor) && (cornerindex.i3 < EndMajor))
			{
				short who_prev = who_am_I - 1;
				if (who_prev < 0) who_prev = tri_len - 1;
				// Worry about pathological cases later.
				short who_next = who_am_I + 1;
				if (who_next == tri_len) who_next = 0;

				n_array[4] = THIRD*(shared_n_shards[cornerindex.i3 - StartMajor].n[who_next]
					+ shared_n_shards[cornerindex.i3 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i3 - StartMajor].n_cent);
				n_array[5] = THIRD*(shared_n_shards[cornerindex.i3 - StartMajor].n[who_prev]
					+ shared_n_shards[cornerindex.i3 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i3 - StartMajor].n_cent);

			}
			else {
				// comes from elsewhere
				f64 ncent = p_n_shards[cornerindex.i3].n_cent;
				short who_prev = who_am_I - 1;
				if (who_prev < 0) {
					who_prev = tri_len - 1;
					f64_vec2 temp;
					memcpy(&temp, p_n_shards[cornerindex.i3].n, sizeof(f64_vec2));
					n_array[4] = THIRD*(temp.x + temp.y + ncent);
					n_array[5] = THIRD*(p_n_shards[cornerindex.i3].n[who_prev] + temp.x + ncent);
				}
				else {
					short who_next = who_am_I + 1;
					if (who_next == tri_len) {
						f64_vec2 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i3].n[who_prev]), sizeof(f64_vec2));
						n_array[4] = THIRD*(p_n_shards[cornerindex.i3].n[0] + temp.y + ncent);
						n_array[5] = THIRD*(temp.x + temp.y + ncent);
					}
					else {
						// typical case
						f64_vec3 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i3].n[who_prev]), sizeof(f64) * 3);
						n_array[4] = THIRD*(temp.z + temp.y + ncent);
						n_array[5] = THIRD*(temp.x + temp.y + ncent);
					};
				};
				//This matches a diagram:
				//             
				//     2---(4)----(3)---1 = corner 1 = indexminor 2: (2,3)
				//      \  /       \   /
				//       \/         \ /
				//       (5\       (2/   indexminor 1 = neighbour 2: (1,2)
				//         \        /
				//          \0)--(1/
				//           \   _/
				//             0  = corner 0 = indexminor0
			};

#pragma unroll 
			for (i = 0; i < 6; i++)
			{
				inext = i + 1; if (inext > 5) inext = 0;

				if ((izNeighMinor[inext] >= StartMinor) && (izNeighMinor[inext] < EndMinor))
				{
					nextAz = shared_Az[izNeighMinor[inext] - StartMinor];
					nextT = shared_T[izNeighMinor[inext] - StartMinor];
					nextAzdot = shared_Azdot[izNeighMinor[inext] - StartMinor];
					nextpos = shared_pos[izNeighMinor[inext] - StartMinor];
				}
				else {
					if ((izNeighMinor[inext] >= StartMajor + BEGINNING_OF_CENTRAL) &&
						(izNeighMinor[inext] < EndMajor + BEGINNING_OF_CENTRAL))
					{
						nextAz = shared_Az_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
						nextAzdot = shared_Azdot_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
						nextT = shared_T_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
						nextpos = shared_pos_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
					}
					else {

						AAdot temp = p_AAdot[izNeighMinor[inext]];
						nextAz = temp.Az;
						nextAzdot = temp.Azdot;
						T3 next_T = p_T_minor[izNeighMinor[inext]];
						nextT.Te = next_T.Te; nextT.Ti = next_T.Ti;

						next_T = p_T_minor[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
						nextpos = p_info_minor[izNeighMinor[inext]].pos;
					};
				};
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) nextpos = Clockwise_d*nextpos;
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) nextpos = Anticlockwise_d*nextpos;

				// New definition of endpoint of minor edge:
				f64_vec2 endpt0, endpt1, edge_normal, integ_grad_Az;
				endpt0 = THIRD * (prevpos + info.pos + opppos);
				endpt1 = THIRD * (nextpos + info.pos + opppos);

				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

				// ______________________________________________________-

				//	Az_edge = SIXTH * (2.0*ourdata.Az + 2.0*oppAz + prevAz + nextAz);
				//	Our_integral_grad_Az += Az_edge * edge_normal;
				//	Our_integral_curl_Az += Az_edge * (endpt1 - endpt0); // looks anticlockwise

				integ_grad_Az.x = 0.5*(
					(ourAz + nextAz)*(info.pos.y - nextpos.y)
					+ (prevAz + ourAz)*(prevpos.y - info.pos.y)
					+ (oppAz + prevAz)*(opppos.y - prevpos.y)
					+ (nextAz + oppAz)*(nextpos.y - opppos.y)
					);
				integ_grad_Az.y = -0.5*( // notice minus
					(ourAz + nextAz)*(info.pos.x - nextpos.x)
					+ (prevAz + ourAz)*(prevpos.x - info.pos.x)
					+ (oppAz + prevAz)*(opppos.x - prevpos.x)
					+ (nextAz + oppAz)*(nextpos.x - opppos.x)
					);
				f64 area_quadrilateral = 0.5*(
					(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
					+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
					+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
					+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
					);

				//f64_vec2 grad_Az = integ_grad_Az / area_quadrilateral;
				//if ((i % 2 == 0) || ((izNeighMinor[i] >= NumInnerFrills_d) && (izNeighMinor[i] < FirstOuterFrill_d)))
				if ((opppos.dot(opppos) < 0.9999*0.9999*FRILL_CENTROID_OUTER_RADIUS_d*FRILL_CENTROID_OUTER_RADIUS_d) &&
					(opppos.dot(opppos) > 1.0001*1.0001*FRILL_CENTROID_INNER_RADIUS_d*FRILL_CENTROID_INNER_RADIUS_d))
					Our_integral_Lap_Az += integ_grad_Az.dot(edge_normal) / area_quadrilateral;

				T3 T0, T1; // waste of registers
				f64 n1;
				T0.Te = THIRD* (prevT.Te + shared_T[threadIdx.x].Te + oppT.Te);
				T1.Te = THIRD * (nextT.Te + shared_T[threadIdx.x].Te + oppT.Te);
				T0.Ti = THIRD * (prevT.Ti + shared_T[threadIdx.x].Ti + oppT.Ti);
				T1.Ti = THIRD * (nextT.Ti + shared_T[threadIdx.x].Ti + oppT.Ti);

				n0 = n_array[i];
				n1 = n_array[inext]; // !

									 // To get integral grad we add the averages along the edges times edge_normals

				if (info.flag == CROSSING_INS) {
					char flag = p_info_minor[izNeighMinor[i]].flag;
					if ((flag == DOMAIN_TRIANGLE) || (flag == DOMAIN_VERTEX))
					{
						// do nothing
					}
					else {
						// Looking into the insulator we see a reflection of nT. Here we look into an out-of-domain tri or vert below ins.
						// Or allowed a below-ins value to affect something anyway.
						// Just for sanity for now, let's just set our own n,T for the edge:
						n0 = p_n_minor[iMinor].n;
						n1 = p_n_minor[iMinor].n;
						T0.Ti = shared_T[threadIdx.x].Ti;
						T0.Te = shared_T[threadIdx.x].Te;
						T1.Ti = T0.Ti;
						T1.Te = T0.Te;
					}
				}
				MAR_ion -= Make3(0.5*(n0 * T0.Ti + n1 * T1.Ti)*over_m_i*edge_normal, 0.0);
				MAR_elec -= Make3(0.5*(n0 * T0.Te + n1 * T1.Te)*over_m_e*edge_normal, 0.0);

				if (TESTTRI) {
					printf("GPU : %d : contribs MAR_ion.y %1.11E MAR_elec.y %1.11E \n"
						"n0 %1.10E n1 %1.10E Ti0 %1.10E Ti1 %1.10E edgenormal.y %1.10E\n",
						CHOSEN,
						-0.5*(n0 * T0.Ti + n1 * T1.Ti)*over_m_i*edge_normal.y,
						-0.5*(n0 * T0.Te + n1 * T1.Te)*over_m_e*edge_normal.y,
						n0, n1, T0.Ti, T1.Ti, edge_normal.y);
				}

				f64 Az_edge = SIXTH * (2.0*ourAz + 2.0*oppAz + prevAz + nextAz);
				f64 Azdot_edge = SIXTH * (2.0*ourAzdot + 2.0*oppAzdot + prevAzdot + nextAzdot);
				Our_integral_grad_Azdot += Azdot_edge * edge_normal;
				Our_integral_grad_Az += Az_edge * edge_normal;
				Our_integral_curl_Az += Az_edge * (endpt1 - endpt0);
				Our_integral_grad_Te += 0.5*(T0.Te + T1.Te) * edge_normal;

				//	if (Az_edge != Az_edge) {
				//		printf("GPU : %d : Az_edge %1.9E ourAz %1.9E oppAz %1.9E \n ourintegralgradTe %1.9E %1.9E contrib %1.9E %1.9E T01 %1.9E %1.9E edgenormal %1.9E %1.9E\n"
				//			"prevT.Te %1.9E ourT.Te %1.9E oppT.Te %1.9E nextT.Te %1.9E \n",
				//			iMinor, Az_edge, ourAz, oppAz,
				//	Our_integral_grad_Te.x, Our_integral_grad_Te.y,
				//			0.5*(T0.Te + T1.Te) * edge_normal.x, 0.5*(T0.Te + T1.Te) * edge_normal.y,
				//			T0.Te, T1.Te, edge_normal.x, edge_normal.y,
				//			prevT.Te, shared_T[threadIdx.x].Te,oppT.Te,nextT.Te
				//		);
				//	}

				AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;
				//
				//				if ((TESTTRI)) 
				//					printf("GPU AreaMinor %d : %1.14E from += %1.14E : endpt0.x %1.14E endpt1.x %1.14E edge_normal.x %1.14E\n"
				//						"endpt1.y endpt0.y %1.14E %1.14E \n",
				//					iMinor, AreaMinor, (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x,
				//					endpt0.x, endpt1.x, edge_normal.x,
				//						endpt1.y, endpt0.y);

				// See a way that FP accuracy was eroded: we take a difference of two close things already to get edge_normal.
				// can that be cleverly avoided? For all calcs?


				endpt0 = endpt1;
				n0 = n1;

				prevpos = opppos;
				prevAz = oppAz;
				prevAzdot = oppAzdot;
				prevT = oppT;

				opppos = nextpos;
				oppAz = nextAz;
				oppAzdot = nextAzdot;
				oppT = nextT;
			};

			if (info.flag == CROSSING_INS) {
				// In this case set v_r = 0 and set a_TP_r = 0 and dv/dt _r = 0 in general
				//f64_vec2 rhat = info.pos / info.pos.modulus();
				MAR_ion -= Make3(
					(MAR_ion.dotxy(info.pos) /
					(info.pos.x*info.pos.x + info.pos.y*info.pos.y))*info.pos, 0.0);
				MAR_elec -= Make3(
					(MAR_elec.dotxy(info.pos) /
					(info.pos.x*info.pos.x + info.pos.y*info.pos.y))*info.pos, 0.0);

				// and we looked at insulator values for T so Grad Te was meaningless:
				Our_integral_grad_Te.x = 0.0;
				Our_integral_grad_Te.y = 0.0;

				// I think we do need to make v_r = 0. It's common sense that it IS 0
				// since we site our v_r estimate on the insulator. Since it is sited there,
				// it is used for traffic into the insulator by n,nT unless we pick out
				// insulator-abutting cells on purpose.

				// However, we then should make an energy correction -- at least if
				// momentum is coming into this minor cell and being destroyed.

				// Doesn't quite work like that. We do not destroy, we just do not store a value for the mom in the domain part of cell.
			};

			p_GradAz[iMinor] = Our_integral_grad_Az / AreaMinor;
			p_LapAz[iMinor] = Our_integral_Lap_Az / AreaMinor;
			p_GradTe[iMinor] = Our_integral_grad_Te / AreaMinor;
			p_B[iMinor] = Make3(Our_integral_curl_Az / AreaMinor, BZ_CONSTANT);
			p_AreaMinor[iMinor] = AreaMinor;

			// wow :
			f64_vec2 overall_v_ours = p_v_overall_minor[iMinor];
			ROCAzduetoAdvection[iMinor] = overall_v_ours.dot(Our_integral_grad_Az / AreaMinor);
			ROCAzdotduetoAdvection[iMinor] = overall_v_ours.dot(Our_integral_grad_Azdot / AreaMinor);

			// No neutral stuff in this kernel, momrates should be set now:
			memcpy(p_MAR_ion + iMinor, &(MAR_ion), sizeof(f64_vec3));
			memcpy(p_MAR_elec + iMinor, &(MAR_elec), sizeof(f64_vec3));
		}
		else {
			// Not domain, not crossing_ins, not a frill
			// ==========================================

			iprev = 5; i = 0;
			if ((izNeighMinor[iprev] >= StartMinor) && (izNeighMinor[iprev] < EndMinor))
			{
				prevAz = shared_Az[izNeighMinor[iprev] - StartMinor];
				prevpos = shared_pos[izNeighMinor[iprev] - StartMinor];
				prevAzdot = shared_Azdot[izNeighMinor[iprev] - StartMinor];
			}
			else {
				if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					prevAz = shared_Az_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
					prevAzdot = shared_Azdot_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
					prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					AAdot temp = p_AAdot[izNeighMinor[iprev]];
					prevAz = temp.Az;
					prevAzdot = temp.Azdot;
					prevpos = p_info_minor[izNeighMinor[iprev]].pos;
				};
			};
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) prevpos = Clockwise_d*prevpos;
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) prevpos = Anticlockwise_d*prevpos;

			i = 0;
			if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
			{
				oppAz = shared_Az[izNeighMinor[i] - StartMinor];
				oppAzdot = shared_Azdot[izNeighMinor[i] - StartMinor];
				opppos = shared_pos[izNeighMinor[i] - StartMinor];
			}
			else {
				if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					oppAz = shared_Az_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
					oppAzdot = shared_Azdot_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
					opppos = shared_pos_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					AAdot temp = p_AAdot[izNeighMinor[i]];
					oppAz = temp.Az;
					oppAzdot = temp.Azdot;
					opppos = p_info_minor[izNeighMinor[i]].pos;
				};
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) opppos = Clockwise_d*opppos;
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) opppos = Anticlockwise_d*opppos;


#pragma unroll 
			for (i = 0; i < 6; i++)
			{
				inext = i + 1; if (inext > 5) inext = 0;

				if ((izNeighMinor[inext] >= StartMinor) && (izNeighMinor[inext] < EndMinor))
				{
					nextAz = shared_Az[izNeighMinor[inext] - StartMinor];
					nextAzdot = shared_Azdot[izNeighMinor[inext] - StartMinor];
					nextpos = shared_pos[izNeighMinor[inext] - StartMinor];
				}
				else {
					if ((izNeighMinor[inext] >= StartMajor + BEGINNING_OF_CENTRAL) &&
						(izNeighMinor[inext] < EndMajor + BEGINNING_OF_CENTRAL))
					{
						nextAz = shared_Az_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
						nextAzdot = shared_Azdot_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
						nextpos = shared_pos_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
					}
					else {
						AAdot temp = p_AAdot[izNeighMinor[inext]];
						nextAz = temp.Az;
						nextAzdot = temp.Azdot;
						nextpos = p_info_minor[izNeighMinor[inext]].pos;
					};
				};
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) nextpos = Clockwise_d*nextpos;
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) nextpos = Anticlockwise_d*nextpos;

				// New definition of endpoint of minor edge:

				f64_vec2 endpt0, endpt1, edge_normal, integ_grad_Az;

				endpt0 = THIRD * (prevpos + info.pos + opppos);
				endpt1 = THIRD * (nextpos + info.pos + opppos);

				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

				// ______________________________________________________-

				//	Az_edge = SIXTH * (2.0*ourdata.Az + 2.0*oppAz + prevAz + nextAz);
				//	Our_integral_grad_Az += Az_edge * edge_normal;
				//	Our_integral_curl_Az += Az_edge * (endpt1 - endpt0); // looks anticlockwise

				integ_grad_Az.x = 0.5*(
					(ourAz + nextAz)*(info.pos.y - nextpos.y)
					+ (prevAz + ourAz)*(prevpos.y - info.pos.y)
					+ (oppAz + prevAz)*(opppos.y - prevpos.y)
					+ (nextAz + oppAz)*(nextpos.y - opppos.y)
					);
				integ_grad_Az.y = -0.5*( // notice minus
					(ourAz + nextAz)*(info.pos.x - nextpos.x)
					+ (prevAz + ourAz)*(prevpos.x - info.pos.x)
					+ (oppAz + prevAz)*(opppos.x - prevpos.x)
					+ (nextAz + oppAz)*(nextpos.x - opppos.x)
					);
				f64 area_quadrilateral = 0.5*(
					(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
					+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
					+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
					+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
					);

				//f64_vec2 grad_Az = integ_grad_Az / area_quadrilateral;
				//		if ((i % 2 == 0) || // vertex neigh 
				//				((izNeighMinor[i] >= NumInnerFrills_d) && (izNeighMinor[i] < FirstOuterFrill_d)))
				if ((opppos.dot(opppos) < 0.9999*0.9999*FRILL_CENTROID_OUTER_RADIUS_d*FRILL_CENTROID_OUTER_RADIUS_d) &&
					(opppos.dot(opppos) > 1.0001*1.0001*FRILL_CENTROID_INNER_RADIUS_d*FRILL_CENTROID_INNER_RADIUS_d))
					Our_integral_Lap_Az += integ_grad_Az.dot(edge_normal) / area_quadrilateral;

				f64 Az_edge = SIXTH * (2.0*ourAz + 2.0*oppAz + prevAz + nextAz);
				f64 Azdot_edge = SIXTH * (2.0*ourAzdot + 2.0*oppAzdot + prevAzdot + nextAzdot);
				Our_integral_grad_Azdot += Azdot_edge * edge_normal;
				Our_integral_grad_Az += Az_edge * edge_normal;
				Our_integral_curl_Az += Az_edge * (endpt1 - endpt0);

				AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;

				endpt0 = endpt1;

				prevpos = opppos;
				prevAz = oppAz;
				prevAzdot = oppAzdot;

				opppos = nextpos;
				oppAz = nextAz;
				oppAzdot = nextAzdot;

			};

			p_GradAz[iMinor] = Our_integral_grad_Az / AreaMinor;
			p_LapAz[iMinor] = Our_integral_Lap_Az / AreaMinor;
			p_B[iMinor] = Make3(Our_integral_curl_Az / AreaMinor, BZ_CONSTANT);
			p_AreaMinor[iMinor] = AreaMinor;

			ROCAzduetoAdvection[iMinor] = 0.0;
			ROCAzdotduetoAdvection[iMinor] = 0.0;
		} // non-domain tri
	}; // was it FRILL

	   // Okay. While we have n_shards in memory we could proceed to overwrite with vxy.
	   // But get running first before using union and checking same.
}
__global__ void kernelCreate_pressure_gradT_and_gradA_CurlA_minor_noadvect(

	structural * __restrict__ p_info_minor,
	T3 * __restrict__ p_T_minor,
	AAdot * __restrict__ p_AAdot,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtriminor,

	LONG3 * __restrict__ p_who_am_I_to_corners,
	LONG3 * __restrict__ p_tricornerindex,

	f64_vec3 * __restrict__ p_MAR_neut,
	f64_vec3 * __restrict__ p_MAR_ion,
	f64_vec3 * __restrict__ p_MAR_elec,
	ShardModel * __restrict__ p_n_shards,
	nvals * __restrict__ p_n_minor, // Just so we can handle insulator

	f64_vec2 * __restrict__ p_GradTe,
	f64_vec2 * __restrict__ p_GradAz,

	f64_vec3 * __restrict__ p_B,
	f64 * __restrict__ p_AreaMinor
)
{
	// Getting this down to 8 vars we could have 512 threads (12 vars/thread total with vertex vars)
	
	__shared__ T2 shared_T[threadsPerTileMinor];
	__shared__ f64 shared_Az[threadsPerTileMinor];
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];

	__shared__ ShardModel shared_n_shards[threadsPerTileMajor];  // 5 + 13/2 + below, 2.5 -> 14 doubles.
	
	// Tile size is 256 though so 14 doubles will allow 1x to run. We have extra shared space if we need it.

	// We could also argue that with shards for n_ion in memory we are better off doing an overwrite and doing stuff for nv also.
	// never mind that for now. <-- ?

	// 2019: Hang on. Why did I use shards? It's quite a good idea. If we are flat then the pressure lands more on the triangles
	// at the interface. Makes a consistent set of values of n to pave the space.

	__shared__ T2 shared_T_verts[threadsPerTileMajor];
	__shared__ f64 shared_Az_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajor];

	// There is a good argument for splitting out A,Adot to a separate routine.
	// That way we could have 10.5 => 585 ie 576 = 288*2 threads.
	// Here we got (2+1+1+2)*1.5 = 9 , + 6.5 = 15.5 -> 384 minor threads max.

	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	long const iVertex = threadsPerTileMajor*blockIdx.x + threadIdx.x; // only meaningful threadIdx.x < threadsPerTileMajor

	long const StartMinor = threadsPerTileMinor*blockIdx.x;
	long const StartMajor = threadsPerTileMajor*blockIdx.x;
	long const EndMinor = StartMinor + threadsPerTileMinor;
	long const EndMajor = StartMajor + threadsPerTileMajor;

	shared_pos[threadIdx.x] = p_info_minor[iMinor].pos;
	{
		AAdot temp = p_AAdot[iMinor];
		shared_Az[threadIdx.x] = temp.Az;
	}
	{
		T3 T_ = p_T_minor[iMinor];
		shared_T[threadIdx.x].Te = T_.Te;
		shared_T[threadIdx.x].Ti = T_.Ti;
	}

	structural info;
	if (threadIdx.x < threadsPerTileMajor) {
		info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
		shared_pos_verts[threadIdx.x] = info.pos;
		AAdot temp = p_AAdot[iVertex + BEGINNING_OF_CENTRAL];
		shared_Az_verts[threadIdx.x] = temp.Az;
		T3 T_ = p_T_minor[iVertex + BEGINNING_OF_CENTRAL];
		shared_T_verts[threadIdx.x].Te = T_.Te;
		shared_T_verts[threadIdx.x].Ti = T_.Ti; // MOVED THIS OUT OF the following branch to see it match CPU
		if (info.flag == DOMAIN_VERTEX) {
			memcpy(&(shared_n_shards[threadIdx.x]), &(p_n_shards[iVertex]), sizeof(ShardModel)); // + 13
		}
		else {
			// save several bus trips;
			// it still manages to coalesce, let's hope, because ShardModel is 13 doubles not 12.
			//shared_T_verts[threadIdx.x].Te = 0.0;
			//shared_T_verts[threadIdx.x].Ti = 0.0;
			memset(&(shared_n_shards[threadIdx.x]), 0, sizeof(ShardModel)); // + 13
		};
	};

	__syncthreads();

	f64 ourAz, oppAz, prevAz, nextAz;
	f64 ourAzdot, oppAzdot, prevAzdot, nextAzdot;
	f64_vec2 opppos, prevpos, nextpos;
	T2 oppT, prevT, nextT;
	//nvals our_n, opp_n, prev_n, next_n;
	f64_vec2 Our_integral_curl_Az, Our_integral_grad_Az, Our_integral_grad_Te;
	f64 Our_integral_Lap_Az;

	if (threadIdx.x < threadsPerTileMajor) {
		Our_integral_curl_Az.x = 0.0;
		Our_integral_curl_Az.y = 0.0;
		Our_integral_grad_Az.x = 0.0;
		Our_integral_grad_Az.y = 0.0;
		Our_integral_grad_Te.x = 0.0;
		Our_integral_grad_Te.y = 0.0;
		Our_integral_Lap_Az = 0.0;
		f64 AreaMinor = 0.0;
		f64_vec3 MAR_ion, MAR_elec;
		memcpy(&(MAR_ion), &(p_MAR_ion[iVertex + BEGINNING_OF_CENTRAL]), sizeof(f64_vec3));
		memcpy(&(MAR_elec), &(p_MAR_elec[iVertex + BEGINNING_OF_CENTRAL]), sizeof(f64_vec3));

		long izTri[MAXNEIGH];
		char szPBC[MAXNEIGH];
		short tri_len = info.neigh_len;

		memcpy(izTri, p_izTri + iVertex*MAXNEIGH, MAXNEIGH * sizeof(long));
		memcpy(szPBC, p_szPBC + iVertex*MAXNEIGH, MAXNEIGH * sizeof(char));

		ourAz = shared_Az_verts[threadIdx.x];

		if (info.flag == DOMAIN_VERTEX) {

			short iprev = tri_len - 1;
			if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMinor))
			{
				prevT = shared_T[izTri[iprev] - StartMinor];
				prevAz = shared_Az[izTri[iprev] - StartMinor];
				prevpos = shared_pos[izTri[iprev] - StartMinor];
			}
			else {
				T3 prev_T = p_T_minor[izTri[iprev]];
				prevT.Te = prev_T.Te; prevT.Ti = prev_T.Ti;
				AAdot temp = p_AAdot[izTri[iprev]];
				prevAz = temp.Az;
				prevpos = p_info_minor[izTri[iprev]].pos;
			}
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) prevpos = Clockwise_d*prevpos;
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) prevpos = Anticlockwise_d*prevpos;

			short inext, i = 0;
			if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
			{
				oppT = shared_T[izTri[i] - StartMinor];
				oppAz = shared_Az[izTri[i] - StartMinor];
				opppos = shared_pos[izTri[i] - StartMinor];
			} else {
				T3 opp_T = p_T_minor[izTri[i]];
				oppT.Te = opp_T.Te; oppT.Ti = opp_T.Ti;
				AAdot temp = p_AAdot[izTri[i]];
				oppAz = temp.Az;
				opppos = p_info_minor[izTri[i]].pos;
			}
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) opppos = Clockwise_d*opppos;
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) opppos = Anticlockwise_d*opppos;
			
			// Think carefully: DOMAIN vertex cases for n,T ...

			f64 n0 = THIRD*(shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[iprev] + shared_n_shards[threadIdx.x].n_cent);
			f64_vec2 endpt1, endpt0 = THIRD * (info.pos + opppos + prevpos);

			short iend = tri_len;
			f64_vec2 projendpt0, edge_normal;
			if ((info.flag == INNERMOST) || (info.flag == OUTERMOST)) {
				iend = tri_len - 2;
				if (info.flag == OUTERMOST) {
					endpt0.project_to_radius(projendpt0, FRILL_CENTROID_OUTER_RADIUS_d); // back of cell for Lap purposes
				} else {
					endpt0.project_to_radius(projendpt0, FRILL_CENTROID_INNER_RADIUS_d); // back of cell for Lap purposes
				}
				edge_normal.x = endpt0.y - projendpt0.y;
				edge_normal.y = projendpt0.x - endpt0.x;
				AreaMinor += (0.5*projendpt0.x + 0.5*endpt0.x)*edge_normal.x;
			};

			for (i = 0; i < iend; i++)
			{
				// Tri 0 is anticlockwise of neighbour 0, we think
				inext = i + 1; if (inext >= tri_len) inext = 0;

				if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
				{
					nextT = shared_T[izTri[inext] - StartMinor];
					nextAz = shared_Az[izTri[inext] - StartMinor];
					nextpos = shared_pos[izTri[inext] - StartMinor];
				} else {
					T3 next_T = p_T_minor[izTri[inext]];
					nextT.Te = next_T.Te; nextT.Ti = next_T.Ti;
					AAdot temp = p_AAdot[izTri[inext]];
					nextAz = temp.Az;
					nextpos = p_info_minor[izTri[inext]].pos;
				}
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) nextpos = Clockwise_d*nextpos;
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) nextpos = Anticlockwise_d*nextpos;

				endpt1 = THIRD * (nextpos + info.pos + opppos);
				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

				// ______________________________________________________-

				//	Az_edge = SIXTH * (2.0*ourdata.Az + 2.0*oppAz + prevAz + nextAz);
				//	Our_integral_grad_Az += Az_edge * edge_normal;
				//	Our_integral_curl_Az += Az_edge * (endpt1 - endpt0); // looks anticlockwise
				//	pData[iDestTri].B -= Az_edge * (endpt1 - endpt0); // MUST DIVIDE BY AREA
			
				T2 T0, T1;
				f64 n1;
				T0.Te = THIRD* (prevT.Te + shared_T_verts[threadIdx.x].Te + oppT.Te);
				T1.Te = THIRD * (nextT.Te + shared_T_verts[threadIdx.x].Te + oppT.Te);
				T0.Ti = THIRD * (prevT.Ti + shared_T_verts[threadIdx.x].Ti + oppT.Ti);
				T1.Ti = THIRD * (nextT.Ti + shared_T_verts[threadIdx.x].Ti + oppT.Ti);
				n1 = THIRD*(shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[inext] + shared_n_shards[threadIdx.x].n_cent);

				// So this is pretty stupid ---
				// If shardmodel went for flat then we have decided that there is no pressure gradient affecting v here.
				// Mind you we didn't expect it to be flat nearly as often as it is flat.
				// Think carefully about what pressure we want to feel.
				// It makes a kind of sense if you have a cliff of density then you feel it in the triangle in between.
				// ***************************************************************************************
				//         But that won't push points apart. It just sends stuff through the wall. 
				// ***************************************************************************************
				// Hmm.

		//		It's a shame we can't just use actual n values to infer gradient over a region. 
		//		It probably creates wobbles in v as well, because if we move fast particles at edge then we leave
		//		Behind a still-lower v in the vertex-centered minor. <-- yes, this instability is clear in practice.
		//		The scheme is kind of skewiffifying.
				
				// Assume neighs 0,1 are relevant to border with tri 0 minor

				// To get integral grad we add the averages along the edges times edge_normals
				MAR_ion -= Make3(0.5*(n0 * T0.Ti + n1 * T1.Ti)*over_m_i*edge_normal, 0.0);
				MAR_elec -= Make3(0.5*(n0 * T0.Te + n1 * T1.Te)*over_m_e*edge_normal, 0.0);
				
//				if (iVertex == VERT1) {
//					printf("GPUpressure %d MAR_ion.x %1.12E contrib.x %1.12E n0 %1.12E Ti0 %1.9E n1 %1.9E Ti1 %1.9E edge_normal.x %1.12E \n",
//						CHOSEN, MAR_ion.x,
//						-0.5*(n0*T0.Ti + n1 * T1.Ti)*over_m_i*edge_normal.x,
//						n0, T0.Ti, n1, T1.Ti, edge_normal.x);
//				}

				Our_integral_grad_Te += 0.5*(T0.Te + T1.Te) * edge_normal;

				// if (iVertex + BEGINNING_OF_CENTRAL == CHOSEN)
				//	printf("GPU %d : GradTe contrib %1.14E %1.14E Te %1.14E opp %1.14E next %1.14E prev %1.14E edge_normal %1.14E %1.14E\n", iVertex + BEGINNING_OF_CENTRAL,
					//	0.5*(T0.Te + T1.Te) * edge_normal.x,
						//0.5*(T0.Te + T1.Te) * edge_normal.y, 
					//	shared_T_verts[threadIdx.x].Te, oppT.Te, nextT.Te, prevT.Te,
						//edge_normal.x, edge_normal.y);

				f64 Az_edge = SIXTH * (2.0*ourAz + 2.0*oppAz + prevAz + nextAz);
				Our_integral_grad_Az += Az_edge * edge_normal;
				Our_integral_curl_Az += Az_edge * (endpt1 - endpt0);

				AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;
				endpt0 = endpt1;
				n0 = n1;

				prevpos = opppos;
				prevAz = oppAz;
				prevT = oppT;

				opppos = nextpos;
				oppAz = nextAz;
				oppT = nextT;
			}; // next i

			//if ((info.flag == INNERMOST) || (info.flag == OUTERMOST)) {

			//	// This will never happen because we just asked info.flag == DOMAIN_VERTEX !!

			//	// Now add on the final sides to give area:

			//	//    3     4
			//	//     2 1 0
			//	// endpt0=endpt1 is now the point north of edge facing 2 anyway.
			//	f64_vec2 projendpt1;

			//	if (info.flag == OUTERMOST) {
			//		endpt1.project_to_radius(projendpt1, FRILL_CENTROID_OUTER_RADIUS_d);
			//	}
			//	else {
			//		endpt1.project_to_radius(projendpt1, FRILL_CENTROID_INNER_RADIUS_d);
			//	};
			//	edge_normal.x = projendpt1.y - endpt1.y;
			//	edge_normal.y = endpt1.x - projendpt1.x;
			//	AreaMinor += (0.5*projendpt1.x + 0.5*endpt1.x)*edge_normal.x;

			//	edge_normal.x = projendpt0.y - projendpt1.y;
			//	edge_normal.y = projendpt1.x - projendpt0.x;
			//	AreaMinor += (0.5*projendpt1.x + 0.5*projendpt0.x)*edge_normal.x;
			//	// line between out-projected points
			//};

			p_GradAz[iVertex + BEGINNING_OF_CENTRAL] = Our_integral_grad_Az / AreaMinor;
		//	p_LapAz[iVertex + BEGINNING_OF_CENTRAL] = Our_integral_Lap_Az / AreaMinor;
			p_GradTe[iVertex + BEGINNING_OF_CENTRAL] = Our_integral_grad_Te / AreaMinor;
			p_B[iVertex + BEGINNING_OF_CENTRAL] = Make3(Our_integral_curl_Az / AreaMinor, BZ_CONSTANT);
			p_AreaMinor[iVertex + BEGINNING_OF_CENTRAL] = AreaMinor;
		
			// No neutral stuff in this kernel, momrates should be set now:
			memcpy(p_MAR_ion + iVertex + BEGINNING_OF_CENTRAL, &MAR_ion, sizeof(f64_vec3));
			memcpy(p_MAR_elec + iVertex + BEGINNING_OF_CENTRAL, &MAR_elec, sizeof(f64_vec3));

		} else {
			// NOT domain vertex: Do Az, Azdot only:			
			short iprev = tri_len - 1;
			if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMinor))
			{
				prevpos = shared_pos[izTri[iprev] - StartMinor];
			} else {
				prevpos = p_info_minor[izTri[iprev]].pos;
			};
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) prevpos = Clockwise_d*prevpos;
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) prevpos = Anticlockwise_d*prevpos;
			short inext, i = 0;
			if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
			{
				opppos = shared_pos[izTri[i] - StartMinor];
			} else {
				opppos = p_info_minor[izTri[i]].pos;
			}
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) opppos = Clockwise_d*opppos;
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) opppos = Anticlockwise_d*opppos;
			
			f64 n0 = THIRD*(shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[iprev] + shared_n_shards[threadIdx.x].n_cent);
			f64_vec2 endpt0 = THIRD * (info.pos + opppos + prevpos);
			f64_vec2 endpt1;			
			short iend = tri_len;
			f64_vec2 projendpt0, edge_normal;
			if ((info.flag == INNERMOST) || (info.flag == OUTERMOST)) {
				iend = tri_len - 2;
				if (info.flag == OUTERMOST) {
					endpt0.project_to_radius(projendpt0, FRILL_CENTROID_OUTER_RADIUS_d); // back of cell for Lap purposes
				}
				else {
					endpt0.project_to_radius(projendpt0, FRILL_CENTROID_INNER_RADIUS_d); // back of cell for Lap purposes
				}
				edge_normal.x = endpt0.y - projendpt0.y;
				edge_normal.y = projendpt0.x - endpt0.x;
				AreaMinor += (0.5*projendpt0.x + 0.5*endpt0.x)*edge_normal.x;

			};
			for (i = 0; i < iend; i++)
			{
				// Tri 0 is anticlockwise of neighbour 0, we think
				inext = i + 1; if (inext >= tri_len) inext = 0;

				if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
				{
					nextpos = shared_pos[izTri[inext] - StartMinor];
				}
				else {
					nextpos = p_info_minor[izTri[inext]].pos;
				}
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) nextpos = Clockwise_d*nextpos;
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) nextpos = Anticlockwise_d*nextpos;

				endpt1 = THIRD * (nextpos + info.pos + opppos);
				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

				// ______________________________________________________-
				
				AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;

				endpt0 = endpt1;
				prevpos = opppos;
				opppos = nextpos;
			}; // next i


			// We have not here corrected what happens to set Lap Az properly at outermost edge --- so let's see
			// that we can remove it from the function.
			

			if ((info.flag == INNERMOST) || (info.flag == OUTERMOST)) {
				// Now add on the final sides to give area:
				//    3     4
				//     2 1 0
				// endpt0=endpt1 is now the point north of edge facing 2 anyway.
				f64_vec2 projendpt1;
				if (info.flag == OUTERMOST) {
					endpt1.project_to_radius(projendpt1, FRILL_CENTROID_OUTER_RADIUS_d);
				} else {
					endpt1.project_to_radius(projendpt1, FRILL_CENTROID_INNER_RADIUS_d);
				};
				edge_normal.x = projendpt1.y - endpt1.y;
				edge_normal.y = endpt1.x - projendpt1.x;
				AreaMinor += (0.5*projendpt1.x + 0.5*endpt1.x)*edge_normal.x;

				edge_normal.x = projendpt0.y - projendpt1.y;
				edge_normal.y = projendpt1.x - projendpt0.x;
				AreaMinor += (0.5*projendpt1.x + 0.5*projendpt0.x)*edge_normal.x;
				// line between out-projected points
			};

			p_GradAz[iVertex + BEGINNING_OF_CENTRAL] = f64_vec2(0.0, 0.0);// Our_integral_grad_Az / AreaMinor; // 0,0
		//	p_LapAz[iVertex + BEGINNING_OF_CENTRAL] = Our_integral_Lap_Az / AreaMinor; 
			p_B[iVertex + BEGINNING_OF_CENTRAL] = f64_vec3(0.0, 0.0, BZ_CONSTANT); // !
			
			p_AreaMinor[iVertex + BEGINNING_OF_CENTRAL] = AreaMinor;
			p_GradTe[iVertex + BEGINNING_OF_CENTRAL] = Vector2(0.0, 0.0);

		}; // // was it domain vertex or Az-only

	};//  if (threadIdx.x < threadsPerTileMajor) 
	   // This branching is itself a good argument for doing Az in ITS own separate routine with no need for n_shard.

	// __syncthreads(); // end of first vertex part
	// Do we need syncthreads? Not overwriting any shared data here...

	   // now the minor with n_ion part:
	info = p_info_minor[iMinor];
	ourAz = shared_Az[threadIdx.x];

	long izNeighMinor[6];
	char szPBC[6];
	memcpy(izNeighMinor, p_izNeighMinor + iMinor * 6, sizeof(long) * 6);
	memcpy(szPBC, p_szPBCtriminor + iMinor * 6, sizeof(char) * 6);

	//	T2 prevT, nextT, oppT;
	//f64 prevAz, nextAz, oppAz, ourAz;
	//f64 prevAzdot, nextAzdot, oppAzdot, ourAzdot;

	f64_vec3 MAR_ion,MAR_elec;
	// this is not a clever way of doing it. Want more careful.

	if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL)) {
		if ((izNeighMinor[3] >= StartMinor) && (izNeighMinor[3] < EndMinor))
		{
			oppAz = shared_Az[izNeighMinor[3] - StartMinor];
		} else {
			AAdot temp = p_AAdot[izNeighMinor[3]];
			oppAz = temp.Az;
		};
		// p_LapAz[iMinor] = oppAz - ourAz; // OBSOLETE ---- need to delete this from routine.
		
		p_GradAz[iMinor] = Vector2(0.0, 0.0);
		memset(&(p_B[iMinor]), 0, sizeof(f64_vec3));
		p_GradTe[iMinor] = Vector2(0.0, 0.0);
		p_AreaMinor[iMinor] = 1.0e-12;
		memset(&(p_MAR_ion[iMinor]), 0, sizeof(f64_vec3));
		memset(&(p_MAR_elec[iMinor]), 0, sizeof(f64_vec3));
	} else {
		Our_integral_curl_Az.x = 0.0;
		Our_integral_curl_Az.y = 0.0;
		Our_integral_grad_Az.x = 0.0;
		Our_integral_grad_Az.y = 0.0;
		Our_integral_grad_Te.x = 0.0;
		Our_integral_grad_Te.y = 0.0;
		Our_integral_Lap_Az = 0.0;
		f64 AreaMinor = 0.0;
		short iprev, inext, i;
		if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS)) {

			memcpy(&MAR_ion, p_MAR_ion + iMinor, sizeof(f64_vec3));
			memcpy(&MAR_elec, p_MAR_elec + iMinor, sizeof(f64_vec3));

			iprev = 5;
			i = 0;
			if ((izNeighMinor[iprev] >= StartMinor) && (izNeighMinor[iprev] < EndMinor))
			{
				prevAz = shared_Az[izNeighMinor[iprev] - StartMinor];
				prevpos = shared_pos[izNeighMinor[iprev] - StartMinor];
				prevT = shared_T[izNeighMinor[iprev] - StartMinor];
			}
			else {
				if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					prevAz = shared_Az_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
					prevT = shared_T_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
					prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					T3 prev_T = p_T_minor[izNeighMinor[iprev]];
					prevT.Te = prev_T.Te; prevT.Ti = prev_T.Ti;
					AAdot temp = p_AAdot[izNeighMinor[iprev]];
					prevAz = temp.Az;
					prevpos = p_info_minor[izNeighMinor[iprev]].pos;
				};
			};
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) prevpos = Clockwise_d*prevpos;
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) prevpos = Anticlockwise_d*prevpos;

			i = 0;
			if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
			{
				oppAz = shared_Az[izNeighMinor[i] - StartMinor];
				oppT = shared_T[izNeighMinor[i] - StartMinor];
				opppos = shared_pos[izNeighMinor[i] - StartMinor];
			}
			else {
				if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					oppAz = shared_Az_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
					oppT = shared_T_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
					opppos = shared_pos_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					T3 opp_T = p_T_minor[izNeighMinor[i]];
					oppT.Te = opp_T.Te; oppT.Ti = opp_T.Ti;
					AAdot temp = p_AAdot[izNeighMinor[i]];
					oppAz = temp.Az;
					opppos = p_info_minor[izNeighMinor[i]].pos;
				};
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) opppos = Clockwise_d*opppos;
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) opppos = Anticlockwise_d*opppos;

			long who_am_I_to_corners[3];
			memcpy(who_am_I_to_corners, &(p_who_am_I_to_corners[iMinor]), sizeof(long) * 3);

			LONG3 cornerindex = p_tricornerindex[iMinor];
			// each corner we want to pick up 3 values off n_shards, as well as n_cent.
			// The three values will not always be contiguous!!!

			// Let's make life easier and load up an array of 6 n's beforehand.
			f64 n_array[6];
			f64 n0, n1;
			// indexminor sequence:
			// 0 = corner 0
			// 1 = neighbour 2
			// 2 = corner 1
			// 3 = neighbour 0
			// 4 = corner 2
			// 5 = neighbour 1

			short who_am_I = who_am_I_to_corners[0];
			short tri_len = p_info_minor[cornerindex.i1 + BEGINNING_OF_CENTRAL].neigh_len;

			if ((cornerindex.i1 >= StartMajor) && (cornerindex.i1 < EndMajor))
			{
				short who_prev = who_am_I - 1;
				if (who_prev < 0) who_prev = tri_len - 1;
				short who_next = who_am_I + 1;
				if (who_next == tri_len) who_next = 0;

				// Worry about pathological cases later.
				// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

				// Pathological case: OUTERMOST vertex where neigh_len is not correct to take as == tri_len

				// !

				// ///////////////////////////////////////////////////////////////////////////////////////////
				// [0] is on our clockwise side rel to [1]. That means it is anticlockwise for the vertex. 
				// That means we interpolate with the value from next tri around.
				n_array[0] = THIRD*(shared_n_shards[cornerindex.i1 - StartMajor].n[who_next]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n_cent);
				n_array[1] = THIRD*(shared_n_shards[cornerindex.i1 - StartMajor].n[who_prev]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n_cent);
			} else {
				// comes from elsewhere
				f64 ncent = p_n_shards[cornerindex.i1].n_cent;
				short who_prev = who_am_I - 1;
				if (who_prev < 0) {
					who_prev = tri_len - 1;
					f64_vec2 temp;
					memcpy(&temp, p_n_shards[cornerindex.i1].n, sizeof(f64_vec2));
					n_array[0] = THIRD*(temp.x + temp.y + ncent);
					n_array[1] = THIRD*(p_n_shards[cornerindex.i1].n[who_prev] + temp.x + ncent);					
				} else {
					short who_next = who_am_I + 1;
					if (who_next == tri_len) {
						f64_vec2 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i1].n[who_prev]), sizeof(f64_vec2));
						n_array[0] = THIRD*(p_n_shards[cornerindex.i1].n[0] + temp.y + ncent);
						n_array[1] = THIRD*(temp.x + temp.y + ncent);
					} else {
						// typical case
						f64_vec3 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i1].n[who_prev]), sizeof(f64) * 3);
						n_array[0] = THIRD*(temp.z + temp.y + ncent);
						n_array[1] = THIRD*(temp.x + temp.y + ncent);						
					};
				};
			}

			who_am_I = who_am_I_to_corners[1];
			tri_len = p_info_minor[cornerindex.i2 + BEGINNING_OF_CENTRAL].neigh_len;

			if ((cornerindex.i2 >= StartMajor) && (cornerindex.i2 < EndMajor))
			{
				short who_prev = who_am_I - 1;
				if (who_prev < 0) who_prev = tri_len - 1;
				short who_next = who_am_I + 1;
				if (who_next == tri_len) who_next = 0;
				// Worry about pathological cases later.
				
				n_array[2] = THIRD*(shared_n_shards[cornerindex.i2 - StartMajor].n[who_next]
								+   shared_n_shards[cornerindex.i2 - StartMajor].n[who_am_I]
								+   shared_n_shards[cornerindex.i2 - StartMajor].n_cent);
				n_array[3] = THIRD*(shared_n_shards[cornerindex.i2 - StartMajor].n[who_prev]
								+   shared_n_shards[cornerindex.i2 - StartMajor].n[who_am_I]
								+   shared_n_shards[cornerindex.i2 - StartMajor].n_cent);
			} else {
				// comes from elsewhere
				f64 ncent = p_n_shards[cornerindex.i2].n_cent;
				short who_prev = who_am_I - 1;
				if (who_prev < 0) {
					who_prev = tri_len - 1;
					f64_vec2 temp;
					memcpy(&temp, p_n_shards[cornerindex.i2].n, sizeof(f64_vec2));
					n_array[2] = THIRD*(temp.x + temp.y + ncent);
					n_array[3] = THIRD*(p_n_shards[cornerindex.i2].n[who_prev] + temp.x + ncent);
				}
				else {
					short who_next = who_am_I + 1;
					if (who_next == tri_len) {
						f64_vec2 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i2].n[who_prev]), sizeof(f64_vec2));
						n_array[2] = THIRD*(p_n_shards[cornerindex.i2].n[0] + temp.y + ncent);
						n_array[3] = THIRD*(temp.x + temp.y + ncent);
					}
					else {
						// typical case
						f64_vec3 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i2].n[who_prev]), sizeof(f64) * 3);
						n_array[2] = THIRD*(temp.z + temp.y + ncent);
						n_array[3] = THIRD*(temp.x + temp.y + ncent);
					};
				};
			}

			who_am_I = who_am_I_to_corners[2];
			tri_len = p_info_minor[cornerindex.i3 + BEGINNING_OF_CENTRAL].neigh_len;

			if ((cornerindex.i3 >= StartMajor) && (cornerindex.i3 < EndMajor))
			{
				short who_prev = who_am_I - 1;
				if (who_prev < 0) who_prev = tri_len - 1;
				// Worry about pathological cases later.
				short who_next = who_am_I + 1;
				if (who_next == tri_len) who_next = 0;

				n_array[4] = THIRD*(shared_n_shards[cornerindex.i3 - StartMajor].n[who_next]
					+ shared_n_shards[cornerindex.i3 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i3 - StartMajor].n_cent);
				n_array[5] = THIRD*(shared_n_shards[cornerindex.i3 - StartMajor].n[who_prev]
					+ shared_n_shards[cornerindex.i3 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i3 - StartMajor].n_cent);

			} else {
				// comes from elsewhere
				f64 ncent = p_n_shards[cornerindex.i3].n_cent;
				short who_prev = who_am_I - 1;
				if (who_prev < 0) {
					who_prev = tri_len - 1;
					f64_vec2 temp;
					memcpy(&temp, p_n_shards[cornerindex.i3].n, sizeof(f64_vec2));
					n_array[4] = THIRD*(temp.x + temp.y + ncent);
					n_array[5] = THIRD*(p_n_shards[cornerindex.i3].n[who_prev] + temp.x + ncent);
				} else {
					short who_next = who_am_I + 1;
					if (who_next == tri_len) {
						f64_vec2 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i3].n[who_prev]), sizeof(f64_vec2));
						n_array[4] = THIRD*(p_n_shards[cornerindex.i3].n[0] + temp.y + ncent);
						n_array[5] = THIRD*(temp.x + temp.y + ncent);
					}
					else {
						// typical case
						f64_vec3 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i3].n[who_prev]), sizeof(f64) * 3);
						n_array[4] = THIRD*(temp.z + temp.y + ncent);
						n_array[5] = THIRD*(temp.x + temp.y + ncent);
					};
				};
				//This matches a diagram:
				//             
				//     2---(4)----(3)---1 = corner 1 = indexminor 2: (2,3)
				//      \  /       \   /
				//       \/         \ /
				//       (5\       (2/   indexminor 1 = neighbour 2: (1,2)
				//         \        /
				//          \0)--(1/
				//           \   _/
				//             0  = corner 0 = indexminor0
			};

#pragma unroll 
			for (i = 0; i < 6; i++)
			{
				inext = i + 1; if (inext > 5) inext = 0;

				if ((izNeighMinor[inext] >= StartMinor) && (izNeighMinor[inext] < EndMinor))
				{
					nextAz = shared_Az[izNeighMinor[inext] - StartMinor];
					nextT = shared_T[izNeighMinor[inext] - StartMinor];
					nextpos = shared_pos[izNeighMinor[inext] - StartMinor];
				}
				else {
					if ((izNeighMinor[inext] >= StartMajor + BEGINNING_OF_CENTRAL) &&
						(izNeighMinor[inext] < EndMajor + BEGINNING_OF_CENTRAL))
					{
						nextAz = shared_Az_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
						nextT = shared_T_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
						nextpos = shared_pos_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
					}
					else {

						AAdot temp = p_AAdot[izNeighMinor[inext]];
						nextAz = temp.Az;
						T3 next_T = p_T_minor[izNeighMinor[inext]];
						nextT.Te = next_T.Te; nextT.Ti = next_T.Ti;

						next_T = p_T_minor[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
						nextpos = p_info_minor[izNeighMinor[inext]].pos;
					};
				};
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) nextpos = Clockwise_d*nextpos;
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) nextpos = Anticlockwise_d*nextpos;

				// New definition of endpoint of minor edge:
				f64_vec2 endpt0, endpt1, edge_normal, integ_grad_Az;
				endpt0 = THIRD * (prevpos + info.pos + opppos);
				endpt1 = THIRD * (nextpos + info.pos + opppos);

				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

				// ______________________________________________________-

				//	Az_edge = SIXTH * (2.0*ourdata.Az + 2.0*oppAz + prevAz + nextAz);
				//	Our_integral_grad_Az += Az_edge * edge_normal;
				//	Our_integral_curl_Az += Az_edge * (endpt1 - endpt0); // looks anticlockwise

				//integ_grad_Az.x = 0.5*(
				//	(ourAz + nextAz)*(info.pos.y - nextpos.y)
				//	+ (prevAz + ourAz)*(prevpos.y - info.pos.y)
				//	+ (oppAz + prevAz)*(opppos.y - prevpos.y)
				//	+ (nextAz + oppAz)*(nextpos.y - opppos.y)
				//	);
				//integ_grad_Az.y = -0.5*( // notice minus
				//	(ourAz + nextAz)*(info.pos.x - nextpos.x)
				//	+ (prevAz + ourAz)*(prevpos.x - info.pos.x)
				//	+ (oppAz + prevAz)*(opppos.x - prevpos.x)
				//	+ (nextAz + oppAz)*(nextpos.x - opppos.x)
				//	);
				//f64 area_quadrilateral = 0.5*(
				//	(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
				//	+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
				//	+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
				//	+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
				//	);

				////f64_vec2 grad_Az = integ_grad_Az / area_quadrilateral;
				////if ((i % 2 == 0) || ((izNeighMinor[i] >= NumInnerFrills_d) && (izNeighMinor[i] < FirstOuterFrill_d)))
				//if ( (opppos.dot(opppos) < 0.9999*0.9999*FRILL_CENTROID_OUTER_RADIUS_d*FRILL_CENTROID_OUTER_RADIUS_d) &&
				//	 (opppos.dot(opppos) > 1.0001*1.0001*FRILL_CENTROID_INNER_RADIUS_d*FRILL_CENTROID_INNER_RADIUS_d))
				//	Our_integral_Lap_Az += integ_grad_Az.dot(edge_normal) / area_quadrilateral;

				//// Modified here ..

				T3 T0, T1; // waste of registers
				f64 n1;
				T0.Te = THIRD* (prevT.Te + shared_T[threadIdx.x].Te + oppT.Te);
				T1.Te = THIRD * (nextT.Te + shared_T[threadIdx.x].Te + oppT.Te);
				T0.Ti = THIRD * (prevT.Ti + shared_T[threadIdx.x].Ti + oppT.Ti);
				T1.Ti = THIRD * (nextT.Ti + shared_T[threadIdx.x].Ti + oppT.Ti);

				n0 = n_array[i];
				n1 = n_array[inext]; // !

				// To get integral grad we add the averages along the edges times edge_normals

				if (info.flag == CROSSING_INS) {
					char flag = p_info_minor[izNeighMinor[i]].flag;
					if ((flag == DOMAIN_TRIANGLE) || (flag == DOMAIN_VERTEX))
					{
						// do nothing
					} else {
						// Looking into the insulator we see a reflection of nT. Here we look into an out-of-domain tri or vert below ins.
						// Or allowed a below-ins value to affect something anyway.
						// Just for sanity for now, let's just set our own n,T for the edge:
						n0 = p_n_minor[iMinor].n;
						n1 = p_n_minor[iMinor].n;
						T0.Ti = shared_T[threadIdx.x].Ti;
						T0.Te = shared_T[threadIdx.x].Te;
						T1.Ti = T0.Ti; 
						T1.Te = T0.Te;
					}
				}
				MAR_ion -= Make3(0.5*(n0 * T0.Ti + n1 * T1.Ti)*over_m_i*edge_normal, 0.0);
				MAR_elec -= Make3(0.5*(n0 * T0.Te + n1 * T1.Te)*over_m_e*edge_normal, 0.0);
			
				if (TESTTRI) {
					printf("GPU : %d : contribs MAR_ion.y %1.11E MAR_elec.y %1.11E \n"
						"n0 %1.10E n1 %1.10E Ti0 %1.10E Ti1 %1.10E edgenormal.y %1.10E\n",
						CHOSEN, 
						-0.5*(n0 * T0.Ti + n1 * T1.Ti)*over_m_i*edge_normal.y,
						-0.5*(n0 * T0.Te + n1 * T1.Te)*over_m_e*edge_normal.y,
						n0, n1, T0.Ti, T1.Ti, edge_normal.y);
				}

				f64 Az_edge = SIXTH * (2.0*ourAz + 2.0*oppAz + prevAz + nextAz);
				Our_integral_grad_Az += Az_edge * edge_normal;
				Our_integral_curl_Az += Az_edge * (endpt1 - endpt0);
				Our_integral_grad_Te += 0.5*(T0.Te + T1.Te) * edge_normal;
				
			//	if (Az_edge != Az_edge) {
			//		printf("GPU : %d : Az_edge %1.9E ourAz %1.9E oppAz %1.9E \n ourintegralgradTe %1.9E %1.9E contrib %1.9E %1.9E T01 %1.9E %1.9E edgenormal %1.9E %1.9E\n"
			//			"prevT.Te %1.9E ourT.Te %1.9E oppT.Te %1.9E nextT.Te %1.9E \n",
			//			iMinor, Az_edge, ourAz, oppAz,
			//	Our_integral_grad_Te.x, Our_integral_grad_Te.y,
			//			0.5*(T0.Te + T1.Te) * edge_normal.x, 0.5*(T0.Te + T1.Te) * edge_normal.y,
			//			T0.Te, T1.Te, edge_normal.x, edge_normal.y,
			//			prevT.Te, shared_T[threadIdx.x].Te,oppT.Te,nextT.Te
			//		);
			//	}

				AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;
//
//				if ((TESTTRI)) 
//					printf("GPU AreaMinor %d : %1.14E from += %1.14E : endpt0.x %1.14E endpt1.x %1.14E edge_normal.x %1.14E\n"
//						"endpt1.y endpt0.y %1.14E %1.14E \n",
//					iMinor, AreaMinor, (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x,
//					endpt0.x, endpt1.x, edge_normal.x,
//						endpt1.y, endpt0.y);

				// See a way that FP accuracy was eroded: we take a difference of two close things already to get edge_normal.
				// can that be cleverly avoided? For all calcs?


				endpt0 = endpt1;
				n0 = n1;

				prevpos = opppos;
				prevAz = oppAz;
				prevAzdot = oppAzdot;
				prevT = oppT;

				opppos = nextpos;
				oppAz = nextAz;
				oppAzdot = nextAzdot;
				oppT = nextT; 
			};

			if (info.flag == CROSSING_INS) {
				// In this case set v_r = 0 and set a_TP_r = 0 and dv/dt _r = 0 in general
				//f64_vec2 rhat = info.pos / info.pos.modulus();
				MAR_ion -= Make3(
					(MAR_ion.dotxy(info.pos) / 
					(info.pos.x*info.pos.x + info.pos.y*info.pos.y))*info.pos, 0.0);
				MAR_elec -= Make3(
					(MAR_elec.dotxy(info.pos) / 
					(info.pos.x*info.pos.x + info.pos.y*info.pos.y))*info.pos, 0.0);

				// and we looked at insulator values for T so Grad Te was meaningless:
				Our_integral_grad_Te.x = 0.0;
				Our_integral_grad_Te.y = 0.0;

				// I think we do need to make v_r = 0. It's common sense that it IS 0
				// since we site our v_r estimate on the insulator. Since it is sited there,
				// it is used for traffic into the insulator by n,nT unless we pick out
				// insulator-abutting cells on purpose.

				// However, we then should make an energy correction -- at least if
				// momentum is coming into this minor cell and being destroyed.

				// Doesn't quite work like that. We do not destroy, we just do not store a value for the mom in the domain part of cell.
			};

			p_GradAz[iMinor] = Our_integral_grad_Az / AreaMinor;
			// p_LapAz[iMinor] = Our_integral_Lap_Az / AreaMinor;
			p_GradTe[iMinor] = Our_integral_grad_Te / AreaMinor;
			p_B[iMinor] = Make3(Our_integral_curl_Az / AreaMinor, BZ_CONSTANT);
			p_AreaMinor[iMinor] = AreaMinor;

			// No neutral stuff in this kernel, momrates should be set now:
			memcpy(p_MAR_ion + iMinor, &(MAR_ion), sizeof(f64_vec3));
			memcpy(p_MAR_elec + iMinor, &(MAR_elec), sizeof(f64_vec3));
		}
		else {
			// Not domain, not crossing_ins, not a frill
			// ==========================================

			iprev = 5; i = 0;
			if ((izNeighMinor[iprev] >= StartMinor) && (izNeighMinor[iprev] < EndMinor))
			{
				prevAz = shared_Az[izNeighMinor[iprev] - StartMinor];
				prevpos = shared_pos[izNeighMinor[iprev] - StartMinor];
			}
			else {
				if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					prevAz = shared_Az_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
					prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					AAdot temp = p_AAdot[izNeighMinor[iprev]];
					prevAz = temp.Az;
					prevpos = p_info_minor[izNeighMinor[iprev]].pos;
				};
			};
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) prevpos = Clockwise_d*prevpos;
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) prevpos = Anticlockwise_d*prevpos;

			i = 0;
			if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
			{
				oppAz = shared_Az[izNeighMinor[i] - StartMinor];
				opppos = shared_pos[izNeighMinor[i] - StartMinor];
			}
			else {
				if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					oppAz = shared_Az_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
					opppos = shared_pos_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					AAdot temp = p_AAdot[izNeighMinor[i]];
					oppAz = temp.Az;
					opppos = p_info_minor[izNeighMinor[i]].pos;
				};
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) opppos = Clockwise_d*opppos;
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) opppos = Anticlockwise_d*opppos;


#pragma unroll 
			for (i = 0; i < 6; i++)
			{
				inext = i + 1; if (inext > 5) inext = 0;

				if ((izNeighMinor[inext] >= StartMinor) && (izNeighMinor[inext] < EndMinor))
				{
					nextAz = shared_Az[izNeighMinor[inext] - StartMinor];
					nextpos = shared_pos[izNeighMinor[inext] - StartMinor];
				}
				else {
					if ((izNeighMinor[inext] >= StartMajor + BEGINNING_OF_CENTRAL) &&
						(izNeighMinor[inext] < EndMajor + BEGINNING_OF_CENTRAL))
					{
						nextAz = shared_Az_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
						nextpos = shared_pos_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
					}
					else {
						AAdot temp = p_AAdot[izNeighMinor[inext]];
						nextAz = temp.Az;
						nextpos = p_info_minor[izNeighMinor[inext]].pos;
					};
				};
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) nextpos = Clockwise_d*nextpos;
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) nextpos = Anticlockwise_d*nextpos;

				// New definition of endpoint of minor edge:

				f64_vec2 endpt0, endpt1, edge_normal, integ_grad_Az;

				endpt0 = THIRD * (prevpos + info.pos + opppos);
				endpt1 = THIRD * (nextpos + info.pos + opppos);

				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

				// ______________________________________________________-

				//	Az_edge = SIXTH * (2.0*ourdata.Az + 2.0*oppAz + prevAz + nextAz);
				//	Our_integral_grad_Az += Az_edge * edge_normal;
				//	Our_integral_curl_Az += Az_edge * (endpt1 - endpt0); // looks anticlockwise

		//		integ_grad_Az.x = 0.5*(
		//			(ourAz + nextAz)*(info.pos.y - nextpos.y)
		//			+ (prevAz + ourAz)*(prevpos.y - info.pos.y)
		//			+ (oppAz + prevAz)*(opppos.y - prevpos.y)
		//			+ (nextAz + oppAz)*(nextpos.y - opppos.y)
		//			);
		//		integ_grad_Az.y = -0.5*( // notice minus
		//			(ourAz + nextAz)*(info.pos.x - nextpos.x)
		//			+ (prevAz + ourAz)*(prevpos.x - info.pos.x)
		//			+ (oppAz + prevAz)*(opppos.x - prevpos.x)
		//			+ (nextAz + oppAz)*(nextpos.x - opppos.x)
		//			);
		//		f64 area_quadrilateral = 0.5*(
		//			(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
		//			+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
		//			+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
		//			+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
		//			);

		//		//f64_vec2 grad_Az = integ_grad_Az / area_quadrilateral;
		////		if ((i % 2 == 0) || // vertex neigh 
		////				((izNeighMinor[i] >= NumInnerFrills_d) && (izNeighMinor[i] < FirstOuterFrill_d)))
		//		if ((opppos.dot(opppos) < 0.9999*0.9999*FRILL_CENTROID_OUTER_RADIUS_d*FRILL_CENTROID_OUTER_RADIUS_d) &&
		//			(opppos.dot(opppos) > 1.0001*1.0001*FRILL_CENTROID_INNER_RADIUS_d*FRILL_CENTROID_INNER_RADIUS_d))
		//			Our_integral_Lap_Az += integ_grad_Az.dot(edge_normal) / area_quadrilateral;

				f64 Az_edge = SIXTH * (2.0*ourAz + 2.0*oppAz + prevAz + nextAz);
				Our_integral_grad_Az += Az_edge * edge_normal;
				Our_integral_curl_Az += Az_edge * (endpt1 - endpt0);

				AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;

				endpt0 = endpt1;

				prevpos = opppos;
				prevAz = oppAz;
			
				opppos = nextpos;
				oppAz = nextAz;
			
			};

			p_GradAz[iMinor] = Our_integral_grad_Az / AreaMinor;
		//	p_LapAz[iMinor] = Our_integral_Lap_Az / AreaMinor;
			p_B[iMinor] = Make3(Our_integral_curl_Az / AreaMinor, BZ_CONSTANT);
			p_AreaMinor[iMinor] = AreaMinor;
		} // non-domain tri
	}; // was it FRILL

	   // Okay. While we have n_shards in memory we could proceed to overwrite with vxy.
	   // But get running first before using union and checking same.

}

__global__ void kernelCreate_momflux_minor(

	structural * __restrict__ p_info_minor,
	v4 * __restrict__ p_vie_minor,
	f64_vec2 * __restrict__ p_v_overall_minor,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtriminor,

	LONG3 * __restrict__ p_who_am_I_to_corners,
	LONG3 * __restrict__ p_tricornerindex,

	f64_vec3 * __restrict__ p_MAR_neut,
	f64_vec3 * __restrict__ p_MAR_ion,
	f64_vec3 * __restrict__ p_MAR_elec,
	ShardModel * __restrict__ p_n_shards
)
{
	__shared__ v4 shared_vie[threadsPerTileMinor];
	__shared__ f64_vec2 shared_v_overall[threadsPerTileMinor];
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];
	__shared__ ShardModel shared_n_shards[threadsPerTileMajor];
	__shared__ v4 shared_vie_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_v_overall_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajor];

	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	long const iVertex = threadsPerTileMajor*blockIdx.x + threadIdx.x; // only meaningful threadIdx.x < threadsPerTileMajor

	long const StartMinor = threadsPerTileMinor*blockIdx.x;
	long const StartMajor = threadsPerTileMajor*blockIdx.x;
	long const EndMinor = StartMinor + threadsPerTileMinor;
	long const EndMajor = StartMajor + threadsPerTileMajor;

	shared_pos[threadIdx.x] = p_info_minor[iMinor].pos;
	shared_vie[threadIdx.x] = p_vie_minor[iMinor];
	shared_v_overall[threadIdx.x] = p_v_overall_minor[iMinor];

	// Perhaps the real answer is this. Advection and therefore advective momflux
	// do not need to be recalculated very often at all. At 1e6 cm/s, we aim for 1 micron,
	// get 1e-10s to actually do the advection !!
	// So an outer cycle. Still limiting the number of total things in a minor tile. We might like 384 = 192*2.

	structural info;
	if (threadIdx.x < threadsPerTileMajor) {
		info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
		shared_pos_verts[threadIdx.x] = info.pos;
		if (info.flag == DOMAIN_VERTEX) {
			memcpy(&(shared_n_shards[threadIdx.x]), &(p_n_shards[iVertex]), sizeof(ShardModel)); // + 13
			memcpy(&(shared_vie_verts[threadIdx.x]), &(p_vie_minor[iVertex + BEGINNING_OF_CENTRAL]), sizeof(v4));
			shared_v_overall_verts[threadIdx.x] = p_v_overall_minor[iVertex + BEGINNING_OF_CENTRAL];
		}
		else {
			// it still manages to coalesce, let's hope, because ShardModel is 13 doubles not 12.
			memset(&(shared_n_shards[threadIdx.x]), 0, sizeof(ShardModel)); // + 13
			memset(&(shared_vie_verts[threadIdx.x]), 0, sizeof(v4));
			memset(&(shared_v_overall_verts[threadIdx.x]), 0, sizeof(f64_vec2));
		};
	};

	__syncthreads();

	v4 our_v, opp_v, prev_v, next_v;
	f64_vec2 our_v_overall, prev_v_overall, next_v_overall, opp_v_overall;
	f64_vec2 opppos, prevpos, nextpos;
	f64 AreaMinor;

	if (threadIdx.x < threadsPerTileMajor) {

		AreaMinor = 0.0;
		three_vec3 ownrates;
		memcpy(&(ownrates.ion), &(p_MAR_ion[iVertex + BEGINNING_OF_CENTRAL]), sizeof(f64_vec3));
		memcpy(&(ownrates.elec), &(p_MAR_elec[iVertex + BEGINNING_OF_CENTRAL]), sizeof(f64_vec3));

		// Now bear in mind:
		// We will often have to do the viscosity calc, and using minor cells.
		// What a bugger.
		// Almost certainly requires lots of stuff like n,T,B. Accept some bus loading.
		// Cross that bridge when we come to it.
		// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

		long izTri[MAXNEIGH];
		char szPBC[MAXNEIGH];
		short tri_len = info.neigh_len;

		memcpy(izTri, p_izTri + iVertex*MAXNEIGH, MAXNEIGH * sizeof(long));
		memcpy(szPBC, p_szPBC + iVertex*MAXNEIGH, MAXNEIGH * sizeof(char));

		our_v = shared_vie_verts[threadIdx.x];
		our_v_overall = shared_v_overall_verts[threadIdx.x];

		if (info.flag == DOMAIN_VERTEX) {

			short iprev = tri_len - 1;
			short i = 0;
			short inext;
			if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMinor))
			{
				prev_v = shared_vie[izTri[iprev] - StartMinor];
				prev_v_overall = shared_v_overall[izTri[iprev] - StartMinor];
				prevpos = shared_pos[izTri[iprev] - StartMinor];
			}
			else {
				prev_v = p_vie_minor[izTri[iprev]];
				prev_v_overall = p_v_overall_minor[izTri[iprev]];
				prevpos = p_info_minor[izTri[iprev]].pos;
			}
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
				prev_v.vxy = Clockwise_d*prev_v.vxy;
				prev_v_overall = Clockwise_d*prev_v_overall;
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
				prev_v.vxy = Anticlockwise_d*prev_v.vxy;
				prev_v_overall = Anticlockwise_d*prev_v_overall;
			}

			if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
			{
				opp_v = shared_vie[izTri[i] - StartMinor];
				opp_v_overall = shared_v_overall[izTri[i] - StartMinor];
				opppos = shared_pos[izTri[i] - StartMinor];
			}
			else {
				opp_v = p_vie_minor[izTri[i]];
				opp_v_overall = p_v_overall_minor[izTri[i]];
				opppos = p_info_minor[izTri[i]].pos;
			}
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
				opp_v.vxy = Clockwise_d*opp_v.vxy;
				opp_v_overall = Clockwise_d*opp_v_overall;
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
				opp_v.vxy = Anticlockwise_d*opp_v.vxy;
				opp_v_overall = Anticlockwise_d*opp_v_overall;
			}

			// Think carefully: DOMAIN vertex cases for n,T ...
			f64 n0 = THIRD*(shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[iprev] + shared_n_shards[threadIdx.x].n_cent);
			f64_vec2 endpt0 = THIRD * (info.pos + opppos + prevpos);

			f64 vez0, viz0, vez1, viz1;
			f64_vec2 vxy0, vxy1, endpt1, edge_normal;

			short iend = tri_len;
			f64_vec2 projendpt0;
			if ((info.flag == INNERMOST) || (info.flag == OUTERMOST)) {
				iend = tri_len - 2;
				if (info.flag == OUTERMOST) {
					endpt0.project_to_radius(projendpt0, FRILL_CENTROID_OUTER_RADIUS_d); // back of cell for Lap purposes
				}
				else {
					endpt0.project_to_radius(projendpt0, FRILL_CENTROID_INNER_RADIUS_d); // back of cell for Lap purposes
				}
				edge_normal.x = endpt0.y - projendpt0.y;
				edge_normal.y = projendpt0.x - endpt0.x;
				AreaMinor += (0.5*projendpt0.x + 0.5*endpt0.x)*edge_normal.x;
			};

			for (i = 0; i < iend; i++)
			{
				// Tri 0 is anticlockwise of neighbour 0, we think
				inext = i + 1; if (inext >= tri_len) inext = 0;

				if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
				{
					next_v = shared_vie[izTri[inext] - StartMinor];
					next_v_overall = shared_v_overall[izTri[inext] - StartMinor];
					nextpos = shared_pos[izTri[inext] - StartMinor];
				}
				else {
					next_v = p_vie_minor[izTri[inext]];
					next_v_overall = p_v_overall_minor[izTri[inext]];
					nextpos = p_info_minor[izTri[inext]].pos;
				}
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
					next_v.vxy = Clockwise_d*next_v.vxy;
					next_v_overall = Clockwise_d*next_v_overall;
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
					next_v.vxy = Anticlockwise_d*next_v.vxy;
					next_v_overall = Anticlockwise_d*next_v_overall;
				}

				f64_vec2 endpt1 = THIRD * (nextpos + info.pos + opppos);
				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

				// ______________________________________________________-

				f64 n1;
				n1 = THIRD*(shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[inext] + shared_n_shards[threadIdx.x].n_cent);

				// Assume neighs 0,1 are relevant to border with tri 0 minor.
				// *********
				// Verify that tri 0 is formed from our vertex, neigh 0 and neigh 1; - tick I think
				// *********
				
				vxy0 = THIRD * (our_v.vxy + prev_v.vxy + opp_v.vxy);
				vxy1 = THIRD * (our_v.vxy + opp_v.vxy + next_v.vxy);

				vez0 = THIRD * (our_v.vez + opp_v.vez + prev_v.vez);
				viz0 = THIRD * (our_v.viz + opp_v.viz + prev_v.viz);

				vez1 = THIRD * (our_v.vez + opp_v.vez + next_v.vez);
				viz1 = THIRD * (our_v.viz + opp_v.viz + next_v.viz);

				f64 relvnormal = 0.5*(vxy0 + vxy1
					- (THIRD * (our_v_overall + next_v_overall + opp_v_overall))
					- (THIRD * (our_v_overall + prev_v_overall + opp_v_overall))
					).dot(edge_normal);

				// In reasonable conditions I suppose that is something sensible.
				// However if we use n v_edge relvnormal then from a fast upwind cell we are always ejecting the slowest material!
				// That is unstable.
				// We could profitably create a minmod model of velocity. 
				// However for now let's try pretending there is a shock front (so use average v for advection) and the upwind nv
				// to advect is just the upwind cell average.

				if (relvnormal > 0.0) {
					// losing stuff : no effect					
				} else {
					ownrates.ion -= 0.5*relvnormal*(n0 + n1)*
						Make3(opp_v.vxy - our_v.vxy, opp_v.viz - our_v.viz);
					
					ownrates.elec -= 0.5*relvnormal*(n0 + n1)*
						Make3(opp_v.vxy - our_v.vxy, opp_v.vez - our_v.vez);					
				};

				// OLD, unstable :
				//ownrates.ion -= 0.5*relvnormal*(n0 *(Make3(vxy0 - our_v.vxy, viz0 - our_v.viz) + n1*(Make3(vxy1 - our_v.vxy, viz1 - our_v.viz))));
					
				//if (TESTADVECT) {
				//			printf("GPUadvect %d izTri[i] %d ownrates.ion.y %1.9E contrib.y %1.9E \n"
				//				"n0 %1.9E n1 %1.9E v0y %1.9E v1y %1.9E \n"
				//				"ourvy %1.9E their vy %1.9E prev %1.9E next %1.9E\n"
				//				"relvnormal %1.12E edgenormal %1.10E %1.10E \n",
				//				VERTCHOSEN, izTri[i], ownrates.ion.y,
				//				-0.5*relvnormal* (n0 * (vxy0.y - our_v.vxy.y) + n1 * (vxy1.y - our_v.vxy.y)),
				//				n0, n1, vxy0.y, vxy1.y, 
				//				our_v.vxy.y, opp_v.vxy.y, prev_v.vxy.y, next_v.vxy.y,								
				//				relvnormal, edge_normal.x, edge_normal.y);
				//		};

				// ______________________________________________________
				//// whether the v that is leaving is greater than our v ..
				//// Formula:
				//// dv/dt = (d(Nv)/dt - dN/dt v) / N
				//// We include the divide by N when we enter the accel routine.

				// Somehow we've created an unstable situ. We are chucking out high-nv at the top. higher n and lower v than in our triangle.
				// Should we insist on upwind v as what is carried?
				// 
				
				AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;

				endpt0 = endpt1;
				n0 = n1;

				prevpos = opppos;
				prev_v = opp_v;
				prev_v_overall = opp_v_overall;

				opppos = nextpos;
				opp_v = next_v;
				opp_v_overall = next_v_overall;
			}; // next i

			   // AreaMinor is not saved, or even calculated for tris.

			   // No neutral stuff in this kernel, momrates should be set now:
			memcpy(p_MAR_ion + iVertex + BEGINNING_OF_CENTRAL, &(ownrates.ion), sizeof(f64_vec3));
			memcpy(p_MAR_elec + iVertex + BEGINNING_OF_CENTRAL, &(ownrates.elec), sizeof(f64_vec3));
		}
		else {
			// NOT domain vertex: Do nothing

			// Then AreaMinor is unset.
			// It's just rubbish!
			// But we do have to calculate LapAz sometimes.

		};
	}; // was it domain vertex or Az-only
	   // This branching is itself a good argument for doing Az in ITS own separate routine with no need for n_shard.

	   // __syncthreads(); // end of first vertex part
	   // Do we need syncthreads? Not overwriting any shared data here...

	   // now the minor with n_ion part:
	info = p_info_minor[iMinor];
	our_v = shared_vie[threadIdx.x];
	our_v_overall = shared_v_overall[threadIdx.x];

	long izNeighMinor[6];
	char szPBC[6];
	memcpy(izNeighMinor, p_izNeighMinor + iMinor * 6, sizeof(long) * 6);
	memcpy(szPBC, p_szPBCtriminor + iMinor * 6, sizeof(char) * 6);

	three_vec3 ownrates_minor;
	memcpy(&(ownrates_minor.ion), &(p_MAR_ion[iMinor]), sizeof(f64_vec3));
	memcpy(&(ownrates_minor.elec), &(p_MAR_elec[iMinor]), sizeof(f64_vec3));

	// this is not a clever way of doing it. Want more careful.

	f64 vez0, viz0, viz1, vez1;
	f64_vec2 vxy0, vxy1;

	if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL)) {
	}
	else {

		AreaMinor = 0.0;
		if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS)) {

			short inext, iprev = 5, i = 0;
			if ((izNeighMinor[iprev] >= StartMinor) && (izNeighMinor[iprev] < EndMinor))
			{
				memcpy(&prev_v, &(shared_vie[izNeighMinor[iprev] - StartMinor]), sizeof(v4));
				prevpos = shared_pos[izNeighMinor[iprev] - StartMinor];
				prev_v_overall = shared_v_overall[izNeighMinor[iprev] - StartMinor];
			}
			else {
				if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					memcpy(&prev_v, &(shared_vie_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(v4));
					prev_v_overall = shared_v_overall_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
					prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					prevpos = p_info_minor[izNeighMinor[iprev]].pos;
					memcpy(&prev_v, &(p_vie_minor[izNeighMinor[iprev]]), sizeof(v4));
					prev_v_overall = p_v_overall_minor[izNeighMinor[iprev]];
				};
			};
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
				prev_v.vxy = Clockwise_d*prev_v.vxy;
				prev_v_overall = Clockwise_d*prev_v_overall;
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
				prev_v.vxy = Anticlockwise_d*prev_v.vxy;
				prev_v_overall = Anticlockwise_d*prev_v_overall;
			}


			i = 0;
			if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
			{
				memcpy(&opp_v, &(shared_vie[izNeighMinor[i] - StartMinor]), sizeof(v4));
				opppos = shared_pos[izNeighMinor[i] - StartMinor];
				opp_v_overall = shared_v_overall[izNeighMinor[i] - StartMinor];
			}
			else {
				if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					memcpy(&opp_v, &(shared_vie_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(v4));
					opp_v_overall = shared_v_overall_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
					opppos = shared_pos_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					opppos = p_info_minor[izNeighMinor[i]].pos;
					memcpy(&opp_v, &(p_vie_minor[izNeighMinor[i]]), sizeof(v4));
					opp_v_overall = p_v_overall_minor[izNeighMinor[i]];
				};
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
				opp_v.vxy = Clockwise_d*opp_v.vxy;
				opp_v_overall = Clockwise_d*opp_v_overall;
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
				opp_v.vxy = Anticlockwise_d*opp_v.vxy;
				opp_v_overall = Anticlockwise_d*opp_v_overall;
			}

			long who_am_I_to_corners[3];
			memcpy(who_am_I_to_corners, &(p_who_am_I_to_corners[iMinor]), sizeof(long) * 3);
			LONG3 cornerindex = p_tricornerindex[iMinor];
			// each corner we want to pick up 3 values off n_shards, as well as n_cent.
			// The three values will not always be contiguous!!!

			// Let's make life easier and load up an array of 6 n's beforehand.
			f64 n_array[6];
			f64 n0, n1;

			short who_am_I = who_am_I_to_corners[0];
			short tri_len = p_info_minor[cornerindex.i1 + BEGINNING_OF_CENTRAL].neigh_len;

			if ((cornerindex.i1 >= StartMajor) && (cornerindex.i1 < EndMajor))
			{
				short who_prev = who_am_I - 1;
				if (who_prev < 0) who_prev = tri_len - 1;
				short who_next = who_am_I + 1;
				if (who_next == tri_len) who_next = 0;
				// Worry about pathological cases later.
				n_array[0] = THIRD*(shared_n_shards[cornerindex.i1 - StartMajor].n[who_next]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n_cent);
				n_array[1] = THIRD*(shared_n_shards[cornerindex.i1 - StartMajor].n[who_prev]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n_cent);
			}
			else {
				// comes from elsewhere
				f64 ncent = p_n_shards[cornerindex.i1].n_cent;
				short who_prev = who_am_I - 1;
				if (who_prev < 0) {
					who_prev = tri_len - 1;
					f64_vec2 temp;
					memcpy(&temp, p_n_shards[cornerindex.i1].n, sizeof(f64_vec2));
					n_array[0] = THIRD*(temp.x + temp.y + ncent);
					n_array[1] = THIRD*(p_n_shards[cornerindex.i1].n[who_prev] + temp.x + ncent);
				}
				else {
					short who_next = who_am_I + 1;
					if (who_next == tri_len) {
						f64_vec2 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i1].n[who_prev]), sizeof(f64_vec2));
						n_array[0] = THIRD*(p_n_shards[cornerindex.i1].n[0] + temp.y + ncent);
						n_array[1] = THIRD*(temp.x + temp.y + ncent);
					}
					else {
						// typical case
						f64_vec3 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i1].n[who_prev]), sizeof(f64) * 3);
						n_array[0] = THIRD*(temp.z + temp.y + ncent);
						n_array[1] = THIRD*(temp.x + temp.y + ncent);
					};
				};
			}

			who_am_I = who_am_I_to_corners[1];
			tri_len = p_info_minor[cornerindex.i2 + BEGINNING_OF_CENTRAL].neigh_len;

			if ((cornerindex.i2 >= StartMajor) && (cornerindex.i2 < EndMajor))
			{
				short who_prev = who_am_I - 1;
				if (who_prev < 0) who_prev = tri_len - 1;
				short who_next = who_am_I + 1;
				if (who_next == tri_len) who_next = 0;
				// Worry about pathological cases later.
				n_array[2] = THIRD*(shared_n_shards[cornerindex.i2 - StartMajor].n[who_next]
					+ shared_n_shards[cornerindex.i2 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i2 - StartMajor].n_cent);
				n_array[3] = THIRD*(shared_n_shards[cornerindex.i2 - StartMajor].n[who_prev]
					+ shared_n_shards[cornerindex.i2 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i2 - StartMajor].n_cent);
			}
			else {
				// comes from elsewhere
				f64 ncent = p_n_shards[cornerindex.i2].n_cent;
				short who_prev = who_am_I - 1;
				if (who_prev < 0) {
					who_prev = tri_len - 1;
					f64_vec2 temp;
					memcpy(&temp, p_n_shards[cornerindex.i2].n, sizeof(f64_vec2));
					n_array[2] = THIRD*(temp.x + temp.y + ncent);
					n_array[3] = THIRD*(p_n_shards[cornerindex.i2].n[who_prev] + temp.x + ncent);
				}
				else {
					short who_next = who_am_I + 1;
					if (who_next == tri_len) {
						f64_vec2 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i2].n[who_prev]), sizeof(f64_vec2));
						n_array[2] = THIRD*(p_n_shards[cornerindex.i2].n[0] + temp.y + ncent);
						n_array[3] = THIRD*(temp.x + temp.y + ncent);
					}
					else {
						// typical case
						f64_vec3 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i2].n[who_prev]), sizeof(f64) * 3);
						n_array[2] = THIRD*(temp.z + temp.y + ncent);
						n_array[3] = THIRD*(temp.x + temp.y + ncent);
					};
				};
			}

			who_am_I = who_am_I_to_corners[2];
			tri_len = p_info_minor[cornerindex.i3 + BEGINNING_OF_CENTRAL].neigh_len;

			if ((cornerindex.i3 >= StartMajor) && (cornerindex.i3 < EndMajor))
			{
				short who_prev = who_am_I - 1;
				if (who_prev < 0) who_prev = tri_len - 1;
				short who_next = who_am_I + 1;
				if (who_next == tri_len) who_next = 0;
				// Worry about pathological cases later.
				n_array[4] = THIRD*(shared_n_shards[cornerindex.i3 - StartMajor].n[who_next]
					+ shared_n_shards[cornerindex.i3 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i3 - StartMajor].n_cent);
				n_array[5] = THIRD*(shared_n_shards[cornerindex.i3 - StartMajor].n[who_prev]
					+ shared_n_shards[cornerindex.i3 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i3 - StartMajor].n_cent);
			}
			else {
				// comes from elsewhere
				f64 ncent = p_n_shards[cornerindex.i3].n_cent;
				short who_prev = who_am_I - 1;
				if (who_prev < 0) {
					who_prev = tri_len - 1;
					f64_vec2 temp;
					memcpy(&temp, p_n_shards[cornerindex.i3].n, sizeof(f64_vec2));
					n_array[4] = THIRD*(temp.x + temp.y + ncent);
					n_array[5] = THIRD*(p_n_shards[cornerindex.i3].n[who_prev] + temp.x + ncent);
				}
				else {
					short who_next = who_am_I + 1;
					if (who_next == tri_len) {
						f64_vec2 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i3].n[who_prev]), sizeof(f64_vec2));
						n_array[4] = THIRD*(p_n_shards[cornerindex.i3].n[0] + temp.y + ncent);
						n_array[5] = THIRD*(temp.x + temp.y + ncent);
					}
					else {
						// typical case
						f64_vec3 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i3].n[who_prev]), sizeof(f64) * 3);
						n_array[4] = THIRD*(temp.z + temp.y + ncent);
						n_array[5] = THIRD*(temp.x + temp.y + ncent);
					};
				};
			}

#pragma unroll 
			for (i = 0; i < 6; i++)
			{
				inext = i + 1; if (inext > 5) inext = 0;
				if ((izNeighMinor[inext] >= StartMinor) && (izNeighMinor[inext] < EndMinor))
				{
					memcpy(&next_v, &(shared_vie[izNeighMinor[inext] - StartMinor]), sizeof(v4));
					nextpos = shared_pos[izNeighMinor[inext] - StartMinor];
					next_v_overall = shared_v_overall[izNeighMinor[inext] - StartMinor];
				}
				else {
					if ((izNeighMinor[inext] >= StartMajor + BEGINNING_OF_CENTRAL) &&
						(izNeighMinor[inext] < EndMajor + BEGINNING_OF_CENTRAL))
					{
						memcpy(&next_v, &(shared_vie_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(v4));
						next_v_overall = shared_v_overall_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
						nextpos = shared_pos_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
					}
					else {
						nextpos = p_info_minor[izNeighMinor[inext]].pos;
						memcpy(&next_v, &(p_vie_minor[izNeighMinor[inext]]), sizeof(v4));
						next_v_overall = p_v_overall_minor[izNeighMinor[inext]];
					};
				};
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
					next_v.vxy = Clockwise_d*next_v.vxy;
					next_v_overall = Clockwise_d*next_v_overall;
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
					next_v.vxy = Anticlockwise_d*next_v.vxy;
					next_v_overall = Anticlockwise_d*next_v_overall;
				}

				// New definition of endpoint of minor edge:
				f64_vec2 endpt0, endpt1, edge_normal;

				endpt0 = THIRD * (prevpos + info.pos + opppos);
				endpt1 = THIRD * (nextpos + info.pos + opppos);

				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

				// ______________________________________________________-
				n0 = n_array[i];
				n1 = n_array[inext]; // 0,1 are either side of corner 0. What is seq of MinorNeigh ? tick

									 // Assume neighs 0,1 are relevant to border with tri 0 minor.

				vxy0 = THIRD * (our_v.vxy + prev_v.vxy + opp_v.vxy);
				vxy1 = THIRD * (our_v.vxy + opp_v.vxy + next_v.vxy);

				vez0 = THIRD * (our_v.vez + opp_v.vez + prev_v.vez);
				viz0 = THIRD * (our_v.viz + opp_v.viz + prev_v.viz);

				vez1 = THIRD * (our_v.vez + opp_v.vez + next_v.vez);
				viz1 = THIRD * (our_v.viz + opp_v.viz + next_v.viz);

				f64 relvnormal = 0.5*(vxy0 + vxy1
					- (THIRD * (our_v_overall + next_v_overall + opp_v_overall))
					- (THIRD * (our_v_overall + prev_v_overall + opp_v_overall))
					).dot(edge_normal);

				if (info.flag == CROSSING_INS) {
					char flag = p_info_minor[izNeighMinor[i]].flag;
					if ((flag == DOMAIN_TRIANGLE) || (flag == DOMAIN_VERTEX))
					{
						// CHANGES 20th August 2019

						// OLD, unstable:
//						ownrates_minor.ion -= 0.5*relvnormal*
//							(n0 * (Make3(vxy0 - our_v.vxy, viz0 - our_v.viz))
//								+ n1 * (Make3(vxy1 - our_v.vxy, viz1 - our_v.viz)));
//						ownrates_minor.elec -= 0.5*relvnormal*
//							(n0 * (Make3(vxy0 - our_v.vxy, vez0 - our_v.vez))
//								+ n1 * (Make3(vxy1 - our_v.vxy, vez1 - our_v.vez)));
//
						if (relvnormal > 0.0) {
							// losing stuff: no effect

							// truly we are changing the amount of momentum in the cell but we have not 
							// programmed it that way. 
							// We are only losing material we assume has same v as cell itself.

						} else {
							ownrates_minor.ion -= 0.5*relvnormal*
								(n0 + n1) * (Make3(opp_v.vxy - our_v.vxy, opp_v.viz - our_v.viz));
							ownrates_minor.elec -= 0.5*relvnormal*
								(n0 + n1) * (Make3(opp_v.vxy - our_v.vxy, opp_v.vez - our_v.vez));
						}
					}
				} else {

					if (relvnormal > 0.0) {
						// losing stuff: no effect

						// truly we are changing the amount of momentum in the cell but we have not 
						// programmed it that way. 
						// We are only losing material we assume has same v as cell itself.

					} else {
						ownrates_minor.ion -= 0.5*relvnormal*
							(n0 + n1) * (Make3(opp_v.vxy - our_v.vxy, opp_v.viz - our_v.viz));
						ownrates_minor.elec -= 0.5*relvnormal*
							(n0 + n1) * (Make3(opp_v.vxy - our_v.vxy, opp_v.vez - our_v.vez));
					}
				};

			//	if (iMinor == 42940) {
			//		printf("49240: CreateMomfluxMinor: %d vxy0  %1.8E %1.8E vxy1 %1.8E %1.8E our_v %1.8E %1.8E\n"
			//			"vez0 %1.8E vez1 %1.8E our_v.vez 1.8E our_v_overall %1.8E %1.8E\n"
			//			"opp_v_overall %1.8E %1.8E\n", izNeighMinor[i],
			//			vxy0.x, vxy0.y, vxy1.x, vxy1.y, our_v.vxy.x, our_v.vxy.y,
			//			vez0, vez1, our_v.vez, our_v_overall.x, our_v_overall.y, opp_v_overall.x, opp_v_overall.y);
			//	}

				if (((TESTTRI))) 
					printf("advectiveGPU %d i %d ownrates_minor.ion.y %1.12E contrib %1.12E relvnormal %1.12E\n"
							"n0 %1.12E n1 %1.12E vxy0.y %1.12E vxy1.y %1.12E vxyours.y %1.12E\n"
							"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n", 
								CHOSEN,i,
								ownrates_minor.ion.y,
								-0.5*relvnormal*(n0 * (vxy0.y - our_v.vxy.y) + n1 * (vxy1.y - our_v.vxy.y)),
								relvnormal,
								n0, n1, vxy0.y, vxy1.y, our_v.vxy.y);
						
				endpt0 = endpt1;

				prevpos = opppos;
				prev_v = opp_v;
				prev_v_overall = opp_v_overall;

				opppos = nextpos;
				opp_v = next_v;
				opp_v_overall = next_v_overall;
			};

			memcpy(&(p_MAR_ion[iMinor]), &(ownrates_minor.ion), sizeof(f64_vec3));
			memcpy(&(p_MAR_elec[iMinor]), &(ownrates_minor.elec), sizeof(f64_vec3));
		}
		else {
			// Not domain, not crossing_ins, not a frill
			// ==========================================
		} // non-domain tri
	}; // was it FRILL
	
	   // Something in the commented code triggers error 77 bad memory access
}


// Not optimized: !!
#define FACTOR_HALL (1.0/0.96)
#define FACTOR_PERP (1.2/0.96)
#define DEBUGNANS


__global__ void kernelCalculate_deps_WRT_beta_Visc(
	f64 const hsub,
	structural * __restrict__ p_info_minor,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtriminor,

	f64 * __restrict__ p_ita_parallel_ion_minor,   // nT / nu ready to look up
	f64 * __restrict__ p_ita_parallel_elec_minor,   // nT / nu ready to look up
	f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
	f64 * __restrict__ p_nu_elec_minor,   // nT / nu ready to look up
	f64_vec3 * __restrict__ p_B_minor,

	nvals * __restrict__ p_n_minor, // got this
	f64 * __restrict__ p_AreaMinor, // got this -> N, Nn

	f64_vec3 * __restrict__ p_Jacobi_ion, 
	f64_vec3 * __restrict__ p_Jacobi_elec, 

	f64_vec3 * __restrict__ p_d_eps_by_d_beta_i_,
	f64_vec3 * __restrict__ p_d_eps_by_d_beta_e_
	)
{
	// We only need 3 in shared now, can re-do when we do elec
	__shared__ f64_vec3 shared_vJ[threadsPerTileMinor]; // sort of thing we want as input
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];
	__shared__ f64_vec2 shared_B[threadsPerTileMinor];
	__shared__ f64 shared_ita_par[threadsPerTileMinor]; // reuse for i,e ; or make 2 vars to combine the routines.
	__shared__ f64 shared_nu[threadsPerTileMinor];

	__shared__ f64_vec3 shared_vJ_verts[threadsPerTileMajor]; // load & reload in Jacobi regressor v instead of v
	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_B_verts[threadsPerTileMajor];
	__shared__ f64 shared_ita_par_verts[threadsPerTileMajor];
	__shared__ f64 shared_nu_verts[threadsPerTileMajor]; // used for creating ita_perp, ita_cross

	// Putting some stuff in shared may speed up if there are spills. !!

	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	long const iVertex = threadsPerTileMajor*blockIdx.x + threadIdx.x; // only meaningful threadIdx.x < threadsPerTileMajor

	long const StartMinor = threadsPerTileMinor*blockIdx.x;
	long const StartMajor = threadsPerTileMajor*blockIdx.x;
	long const EndMinor = StartMinor + threadsPerTileMinor;
	long const EndMajor = StartMajor + threadsPerTileMajor;

	f64_vec3 our_v, opp_v, prev_v, next_v;
	f64_vec2 opppos, prevpos, nextpos;
	f64 nu, ita_par;  // optimization: we always each loop want to get rid of omega, nu once we have calc'd these, if possible!!
	
	f64_vec3 d_eps_by_d_beta;
	
	shared_pos[threadIdx.x] = p_info_minor[iMinor].pos;
	shared_vJ[threadIdx.x] = p_Jacobi_ion[iMinor];    // is memcpy faster or slower than operator= ?
	shared_B[threadIdx.x] = p_B_minor[iMinor].xypart();
	shared_ita_par[threadIdx.x] = p_ita_parallel_ion_minor[iMinor];
	shared_nu[threadIdx.x] = p_nu_ion_minor[iMinor];

	// Perhaps the real answer is this. Advection and therefore advective momflux
	// do not need to be recalculated very often at all. At 1e6 cm/s, we aim for 1 micron,
	// get 1e-10s to actually do the advection !!
	// So an outer cycle. Still limiting the number of total things in a minor tile. We might like 384 = 192*2.

	structural info;
	if (threadIdx.x < threadsPerTileMajor) {
		info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
		shared_pos_verts[threadIdx.x] = info.pos;
		shared_B_verts[threadIdx.x] = p_B_minor[iVertex + BEGINNING_OF_CENTRAL].xypart();
		if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST))
		{
			memcpy(&(shared_vJ_verts[threadIdx.x]), &(p_Jacobi_ion[iVertex + BEGINNING_OF_CENTRAL]), sizeof(f64_vec3));
			shared_ita_par_verts[threadIdx.x] = p_ita_parallel_ion_minor[iVertex + BEGINNING_OF_CENTRAL];
			shared_nu_verts[threadIdx.x] = p_nu_ion_minor[iVertex + BEGINNING_OF_CENTRAL];
			// But now I am going to set ita == 0 in OUTERMOST and agree never to look there because that's fairer than one-way traffic and I don't wanna handle OUTERMOST?
			// I mean, I could handle it, and do flows only if the flag does not come up OUTER_FRILL.
			// OK just do that.
		}
		else {
			// it still manages to coalesce, let's hope, because ShardModel is 13 doubles not 12.
			memset(&(shared_vJ_verts[threadIdx.x]), 0, sizeof(f64_vec3));
			shared_ita_par_verts[threadIdx.x] = 0.0;
			shared_nu_verts[threadIdx.x] = 0.0;
		};
	};

	__syncthreads();

	// How shall we arrange to do v_n, which is isotropic? Handle this first...
	// Is the v_n coefficient negligible? Check.

	// We actually have to think how to handle the x-y dimension. PopOhms will handle it.

	// We can re-use some shared data -- such as pos and B -- to do both ions and electrons
	// But they use different ita_par and different vez, viz. 
	// Often we don't need to do magnetised ion viscosity when we do magnetised electron.

	// IONS FIRST:

	if (threadIdx.x < threadsPerTileMajor) {
	

		long izTri[MAXNEIGH_d];
		char szPBC[MAXNEIGH_d];
		short tri_len = info.neigh_len; // ?!

		if ((info.flag == DOMAIN_VERTEX))
			//|| (info.flag == OUTERMOST)) 
		{
			// We are losing energy if there is viscosity into OUTERMOST.
			memcpy(izTri, p_izTri + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(long));
			memcpy(szPBC, p_szPBC + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(char));

			our_v = shared_vJ_verts[threadIdx.x]; // optimization: use replace or #define to get rid of storing this again.

			d_eps_by_d_beta = our_v; // eps = v_k+1 - v_k - h/N MAR

			f64 Factor = hsub / (p_n_minor[iVertex + BEGINNING_OF_CENTRAL].n * p_AreaMinor[iVertex + BEGINNING_OF_CENTRAL] * m_ion);

			short iprev = tri_len - 1;
			short i = 0;
			short inext;
			if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMinor))
			{
				prev_v = shared_vJ[izTri[iprev] - StartMinor];
				prevpos = shared_pos[izTri[iprev] - StartMinor];
			}
			else {
				prev_v = p_Jacobi_ion[izTri[iprev]];
				prevpos = p_info_minor[izTri[iprev]].pos;
			}
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
				RotateClockwise(prev_v);
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
				RotateAnticlockwise(prev_v);
			}

			if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
			{
				opp_v = shared_vJ[izTri[i] - StartMinor];
				opppos = shared_pos[izTri[i] - StartMinor];
			}
			else {
				opp_v = p_Jacobi_ion[izTri[i]];
				opppos = p_info_minor[izTri[i]].pos;
			}
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
				RotateClockwise(opp_v);
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
				RotateAnticlockwise(opp_v);
			}

			f64_vec2 endpt0 = THIRD * (info.pos + opppos + prevpos);
			f64_vec3 omega_ci;

#pragma unroll 
			for (i = 0; i < tri_len; i++)
			{
				// Tri 0 is anticlockwise of neighbour 0, we think
				inext = i + 1; if (inext >= tri_len) inext = 0;

				// Now sort out anticlock vars:

				if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
				{
					next_v = shared_vJ[izTri[inext] - StartMinor];
					nextpos = shared_pos[izTri[inext] - StartMinor];
				}
				else {
					next_v = p_Jacobi_ion[izTri[inext]];
					nextpos = p_info_minor[izTri[inext]].pos;
				}
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
					RotateClockwise(next_v);
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
					RotateAnticlockwise(next_v);
				}

				// Order of calculations may help things to go out/into scope at the right times so careful with that.
				f64_vec2 gradvx, gradvy, gradviz;

				f64 area_quadrilateral = 0.5*(
					(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
					+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
					+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
					+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
					);

				gradvx.x = 0.5*(
					(our_v.x + next_v.x)*(info.pos.y - nextpos.y)
					+ (prev_v.x + our_v.x)*(prevpos.y - info.pos.y)
					+ (opp_v.x + prev_v.x)*(opppos.y - prevpos.y)
					+ (next_v.x + opp_v.x)*(nextpos.y - opppos.y) // nextpos = pos_anti, assumed
					) / area_quadrilateral;
				gradvx.y = -0.5*(
					(our_v.x + next_v.x)*(info.pos.x - nextpos.x)
					+ (prev_v.x + our_v.x)*(prevpos.x - info.pos.x)
					+ (opp_v.x + prev_v.x)*(opppos.x - prevpos.x)
					+ (next_v.x + opp_v.x)*(nextpos.x - opppos.x) // nextpos = pos_anti, assumed
					) / area_quadrilateral;

				gradvy.x = 0.5*(
					(our_v.y + next_v.y)*(info.pos.y - nextpos.y)
					+ (prev_v.y + our_v.y)*(prevpos.y - info.pos.y)
					+ (opp_v.y + prev_v.y)*(opppos.y - prevpos.y)
					+ (next_v.y + opp_v.y)*(nextpos.y - opppos.y) // nextpos = pos_anti, assumed
					) / area_quadrilateral;
				gradvy.y = -0.5*(
					(our_v.y + next_v.y)*(info.pos.x - nextpos.x)
					+ (prev_v.y + our_v.y)*(prevpos.x - info.pos.x)
					+ (opp_v.y + prev_v.y)*(opppos.x - prevpos.x)
					+ (next_v.y + opp_v.y)*(nextpos.x - opppos.x) // nextpos = pos_anti, assumed
					) / area_quadrilateral;
				//
				//				if (TEST) printf(
				//					"iVertex %d our_v.y next prev opp %1.8E %1.8E %1.8E %1.8E\n"
				//					"area_quad %1.8E \n"
				//					"info.pos %1.8E %1.8E opppos %1.8E %1.8E prev %1.8E %1.8E next %1.8E %1.8E\n",
				//					iVertex, our_v.vxy.y, next_v.vxy.y, prev_v.vxy.y, opp_v.vxy.y,
				//					area_quadrilateral,
				//					info.pos.x, info.pos.y, opppos.x, opppos.y, prevpos.x, prevpos.y, nextpos.x, nextpos.y);
				//
				gradviz.x = 0.5*(
					(our_v.z + next_v.z)*(info.pos.y - nextpos.y)
					+ (prev_v.z + our_v.z)*(prevpos.y - info.pos.y)
					+ (opp_v.z + prev_v.z)*(opppos.y - prevpos.y)
					+ (next_v.z + opp_v.z)*(nextpos.y - opppos.y) // nextpos = pos_anti, assumed
					) / area_quadrilateral;
				gradviz.y = -0.5*(
					(our_v.z + next_v.z)*(info.pos.x - nextpos.x)
					+ (prev_v.z + our_v.z)*(prevpos.x - info.pos.x)
					+ (opp_v.z + prev_v.z)*(opppos.x - prevpos.x)
					+ (next_v.z + opp_v.z)*(nextpos.x - opppos.x) // nextpos = pos_anti, assumed
					) / area_quadrilateral;

				f64_vec2 endpt1 = THIRD * (nextpos + info.pos + opppos);
				{
					f64_vec2 opp_B;
					if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
					{
						opp_B = shared_B[izTri[i] - StartMinor];
						if (shared_ita_par_verts[threadIdx.x] < shared_ita_par[izTri[i] - StartMinor])
						{
							ita_par = shared_ita_par_verts[threadIdx.x];
							nu = shared_nu_verts[threadIdx.x];
						}
						else {
							ita_par = shared_ita_par[izTri[i] - StartMinor];
							nu = shared_nu[izTri[i] - StartMinor];
						};
					}
					else {
						opp_B = p_B_minor[izTri[i]].xypart();
						f64 ita_theirs = p_ita_parallel_ion_minor[izTri[i]];
						f64 nu_theirs = p_nu_ion_minor[izTri[i]];
						if (shared_ita_par_verts[threadIdx.x] < ita_theirs) {
							ita_par = shared_ita_par_verts[threadIdx.x];
							nu = shared_nu_verts[threadIdx.x];
						}
						else {
							ita_par = ita_theirs;
							nu = nu_theirs;
						};
					}
					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opp_B = Clockwise_d*opp_B;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opp_B = Anticlockwise_d*opp_B;
					}
					omega_ci = 0.5*qoverMc*(Make3(opp_B + shared_B_verts[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
				} // Guaranteed DOMAIN_VERTEX never needs to skip an edge; we include CROSSING_INS in viscosity.

				if ((VISCMAG == 0) || (omega_ci.dot(omega_ci) < 0.01*0.1*nu*nu))
				{
					// run unmagnetised case
					f64 Pi_xx, Pi_xy, Pi_yx, Pi_yy, Pi_zx, Pi_zy;

					Pi_xx = -ita_par*THIRD*(4.0*gradvx.x - 2.0*gradvy.y);
					Pi_xy = -ita_par*(gradvx.y + gradvy.x);
					Pi_yx = Pi_xy;
					Pi_yy = -ita_par*THIRD*(4.0*gradvy.y - 2.0*gradvx.x);
					Pi_zx = -ita_par*(gradviz.x);
					Pi_zy = -ita_par*(gradviz.y);

					f64_vec2 edge_normal;
					edge_normal.x = endpt1.y - endpt0.y;
					edge_normal.y = endpt0.x - endpt1.x;

					//visc_contrib.y = -over_m_i*(Pi_yx*edge_normal.x + Pi_yy*edge_normal.y);
					
					d_eps_by_d_beta.x += Factor*(Pi_xx*edge_normal.x + Pi_xy*edge_normal.y); // - h/N visc_contrib I think
					d_eps_by_d_beta.y += Factor*(Pi_yx*edge_normal.x + Pi_yy*edge_normal.y);
					d_eps_by_d_beta.z += Factor*(Pi_zx*edge_normal.x + Pi_zy*edge_normal.y);
					
				} else {

					f64 omegamod;
					f64_vec3 unit_b, unit_perp, unit_Hall;
					{
						f64_vec2 edge_normal;
						edge_normal.x = endpt1.y - endpt0.y;
						edge_normal.y = endpt0.x - endpt1.x; // need to define so as to create unit vectors

						f64 omegasq = omega_ci.dot(omega_ci);
						omegamod = sqrt(omegasq);
						unit_b = omega_ci / omegamod;
						unit_perp = Make3(edge_normal, 0.0) - unit_b*(unit_b.dotxy(edge_normal));
						unit_perp = unit_perp / unit_perp.modulus();
						unit_Hall = unit_b.cross(unit_perp); // Note sign.

															 // store omegamod instead.
															 //	ita_perp = FACTOR_PERP * ita_par * nu*nu / (omegasq + nu*nu);
															 //	ita_cross = FACTOR_HALL * ita_par * nu*omegamod / (omegasq + nu*nu);
					}
					
					f64 W_bb = 0.0, W_bP = 0.0, W_bH = 0.0, W_PP = 0.0, W_PH = 0.0, W_HH = 0.0; // these have to be alive at same time as 9 x partials
																								// but we can make do with 3x partials
																								// 2. Now get partials in magnetic coordinates 
					{
						f64_vec3 intermed;

						// use: d vb / da = b transpose [ dvi/dxj ] a
						// Prototypical element: a.x b.y dvy/dx
						// b.x a.y dvx/dy

						intermed.x = unit_b.dotxy(gradvx);
						intermed.y = unit_b.dotxy(gradvy);
						intermed.z = unit_b.dotxy(gradviz);
						{
							f64 dvb_by_db, dvperp_by_db, dvHall_by_db;

							dvb_by_db = unit_b.dot(intermed);
							dvperp_by_db = unit_perp.dot(intermed);
							dvHall_by_db = unit_Hall.dot(intermed);

							W_bb += 4.0*THIRD*dvb_by_db;
							W_bP += dvperp_by_db;
							W_bH += dvHall_by_db;
							W_PP -= 2.0*THIRD*dvb_by_db;
							W_HH -= 2.0*THIRD*dvb_by_db;
						}
						{
							f64 dvb_by_dperp, dvperp_by_dperp,
								dvHall_by_dperp;
							// Optimize by getting rid of different labels.

							intermed.x = unit_perp.dotxy(gradvx);
							intermed.y = unit_perp.dotxy(gradvy);
							intermed.z = unit_perp.dotxy(gradviz);

							dvb_by_dperp = unit_b.dot(intermed);
							dvperp_by_dperp = unit_perp.dot(intermed);
							dvHall_by_dperp = unit_Hall.dot(intermed);

							W_bb -= 2.0*THIRD*dvperp_by_dperp;
							W_PP += 4.0*THIRD*dvperp_by_dperp;
							W_HH -= 2.0*THIRD*dvperp_by_dperp;
							W_bP += dvb_by_dperp;
							W_PH += dvHall_by_dperp;
						}
						{
							f64 dvb_by_dHall, dvperp_by_dHall, dvHall_by_dHall;

							intermed.x = unit_Hall.dotxy(gradvx);
							intermed.y = unit_Hall.dotxy(gradvy);
							intermed.z = unit_Hall.dotxy(gradviz);

							dvb_by_dHall = unit_b.dot(intermed);
							dvperp_by_dHall = unit_perp.dot(intermed);
							dvHall_by_dHall = unit_Hall.dot(intermed);

							W_bb -= 2.0*THIRD*dvHall_by_dHall;
							W_PP -= 2.0*THIRD*dvHall_by_dHall;
							W_HH += 4.0*THIRD*dvHall_by_dHall;
							W_bH += dvb_by_dHall;
							W_PH += dvperp_by_dHall;
						}
					}

					f64 Pi_b_b = 0.0, Pi_P_b = 0.0, Pi_P_P = 0.0, Pi_H_b = 0.0, Pi_H_P = 0.0, Pi_H_H = 0.0;
					{
						{
							f64 ita_1 = ita_par*(nu*nu / (nu*nu + omegamod*omegamod));

							Pi_b_b += -ita_par*W_bb;
							Pi_P_P += -0.5*(ita_par + ita_1)*W_PP - 0.5*(ita_par - ita_1)*W_HH;
							Pi_H_H += -0.5*(ita_par + ita_1)*W_HH - 0.5*(ita_par - ita_1)*W_PP;
							Pi_H_P += -ita_1*W_PH;
						}
						{
							f64 ita_2 = ita_par*(nu*nu / (nu*nu + 0.25*omegamod*omegamod));
							Pi_P_b += -ita_2*W_bP;
							Pi_H_b += -ita_2*W_bH;
						}
						{
							f64 ita_3 = ita_par*(nu*omegamod / (nu*nu + omegamod*omegamod));
							Pi_P_P -= ita_3*W_PH;
							Pi_H_H += ita_3*W_PH;
							Pi_H_P += 0.5*ita_3*(W_PP - W_HH);
						}
						{
							f64 ita_4 = 0.5*ita_par*(nu*omegamod / (nu*nu + 0.25*omegamod*omegamod));
							Pi_P_b += -ita_4*W_bH;
							Pi_H_b += ita_4*W_bP;
						}
					}

					f64 momflux_b, momflux_perp, momflux_Hall;
					{
						f64_vec3 mag_edge;
						f64_vec2 edge_normal;
						edge_normal.x = endpt1.y - endpt0.y;
						edge_normal.y = endpt0.x - endpt1.x; // need to define so as to create unit vectors

															 // Most efficient way: compute mom flux in magnetic coords
						mag_edge.x = unit_b.x*edge_normal.x + unit_b.y*edge_normal.y;
						mag_edge.y = unit_perp.x*edge_normal.x + unit_perp.y*edge_normal.y;
						mag_edge.z = unit_Hall.x*edge_normal.x + unit_Hall.y*edge_normal.y;

						momflux_b = -(Pi_b_b*mag_edge.x + Pi_P_b*mag_edge.y + Pi_H_b*mag_edge.z);
						momflux_perp = -(Pi_P_b*mag_edge.x + Pi_P_P*mag_edge.y + Pi_H_P*mag_edge.z);
						momflux_Hall = -(Pi_H_b*mag_edge.x + Pi_H_P*mag_edge.y + Pi_H_H*mag_edge.z);
					}

					// ownrates will be divided by N to give dv/dt
					// visc_contrib.z = over_m_i*(unit_b.z*momflux_b + unit_perp.z*momflux_perp + unit_Hall.z*momflux_Hall);
					//	ownrates_visc += visc_contrib;

					d_eps_by_d_beta.x += -Factor*(unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall);
					d_eps_by_d_beta.y += -Factor*(unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall);
					d_eps_by_d_beta.z += -Factor*(unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall);
					// We should have created device function for the visc calc since it is repeated now at least 8 times.

					// Note that momflux here already had -, visc_contrib did not contain -over_m_i as for unmag.

				}

				// MAR_elec -= Make3(0.5*(n0 * T0.Te + n1 * T1.Te)*over_m_e*edge_normal, 0.0);
				// v0.vez = vie_k.vez + h_use * MAR.z / (n_use.n*AreaMinor);

				endpt0 = endpt1;
				prevpos = opppos;
				prev_v = opp_v;
				opppos = nextpos;
				opp_v = next_v;
			}; // next i
			memcpy(p_d_eps_by_d_beta_i_ + iVertex + BEGINNING_OF_CENTRAL, &d_eps_by_d_beta, sizeof(f64_vec3));
		} else {
			// NOT domain vertex: Do nothing			
		};
	};
	// __syncthreads(); // end of first vertex part
	// Do we need syncthreads? Not overwriting any shared data here...

	// Ion , triangle:
	info = p_info_minor[iMinor];
	our_v = shared_vJ[threadIdx.x];
	d_eps_by_d_beta = our_v;

	//if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL)) {
	{
		long izNeighMinor[6];
		char szPBC[6];

		if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS)) {

			f64 Factor = hsub / (p_n_minor[iMinor].n * p_AreaMinor[iMinor] * m_ion);

			memcpy(izNeighMinor, p_izNeighMinor + iMinor * 6, sizeof(long) * 6);
			memcpy(szPBC, p_szPBCtriminor + iMinor * 6, sizeof(char) * 6);

			short inext, iprev = 5, i = 0;
			if ((izNeighMinor[iprev] >= StartMinor) && (izNeighMinor[iprev] < EndMinor))
			{
				memcpy(&prev_v, &(shared_vJ[izNeighMinor[iprev] - StartMinor]), sizeof(f64_vec3));
				prevpos = shared_pos[izNeighMinor[iprev] - StartMinor];
			}
			else {
				if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					memcpy(&prev_v, &(shared_vJ_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(f64_vec3));
					prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					prevpos = p_info_minor[izNeighMinor[iprev]].pos;
					memcpy(&prev_v, &(p_Jacobi_ion[izNeighMinor[iprev]]), sizeof(f64_vec3));
				};
			};
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
				RotateClockwise(prev_v);
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
				RotateAnticlockwise(prev_v);
			}

			i = 0;
			if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
			{
				memcpy(&opp_v, &(shared_vJ[izNeighMinor[i] - StartMinor]), sizeof(f64_vec3));
				opppos = shared_pos[izNeighMinor[i] - StartMinor];
			}
			else {
				if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					memcpy(&opp_v, &(shared_vJ_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(f64_vec3));
					opppos = shared_pos_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					opppos = p_info_minor[izNeighMinor[i]].pos;
					memcpy(&opp_v, &(p_Jacobi_ion[izNeighMinor[i]]), sizeof(f64_vec3));
				};
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
				RotateClockwise(opp_v);
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
				RotateAnticlockwise(opp_v);
			}

			f64_vec2 endpt0 = THIRD * (info.pos + opppos + prevpos);

			f64_vec3 omega_ci;
			// Let's make life easier and load up an array of 6 n's beforehand.
#pragma unroll 
			for (i = 0; i < 6; i++)
			{
				inext = i + 1; if (inext > 5) inext = 0;

				if ((izNeighMinor[inext] >= StartMinor) && (izNeighMinor[inext] < EndMinor))
				{
					memcpy(&next_v, &(shared_vJ[izNeighMinor[inext] - StartMinor]), sizeof(f64_vec3));
					nextpos = shared_pos[izNeighMinor[inext] - StartMinor];
				}
				else {
					if ((izNeighMinor[inext] >= StartMajor + BEGINNING_OF_CENTRAL) &&
						(izNeighMinor[inext] < EndMajor + BEGINNING_OF_CENTRAL))
					{
						memcpy(&next_v, &(shared_vJ_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(f64_vec3));
						nextpos = shared_pos_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
					}
					else {
						nextpos = p_info_minor[izNeighMinor[inext]].pos;
						memcpy(&next_v, &(p_Jacobi_ion[izNeighMinor[inext]]), sizeof(f64_vec3));
					};
				};
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
					RotateClockwise(next_v);
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
					RotateAnticlockwise(next_v);
				}

				//	nu = 1.0e10; // DEBUG
				bool bUsableSide = true;
				{
					f64_vec2 opp_B(0.0, 0.0);
					// newly uncommented:
					if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
					{
						opp_B = shared_B[izNeighMinor[i] - StartMinor];
						if (shared_ita_par[threadIdx.x] < shared_ita_par[izNeighMinor[i] - StartMinor])
						{
							ita_par = shared_ita_par[threadIdx.x];
							nu = shared_nu[threadIdx.x];
						} else {
							ita_par = shared_ita_par[izNeighMinor[i] - StartMinor];
							nu = shared_nu[izNeighMinor[i] - StartMinor];
						};
						if (shared_ita_par[izNeighMinor[i] - StartMinor] == 0.0) bUsableSide = false;
					}
					else {
						if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
							(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
						{
							opp_B = shared_B_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
							if (shared_ita_par[threadIdx.x] < shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL])
							{
								ita_par = shared_ita_par[threadIdx.x];
								nu = shared_nu[threadIdx.x];
							} else {
								ita_par = shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
								nu = shared_nu_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
							};
							if (shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL] == 0.0) bUsableSide = false;
						}
						else {
							opp_B = p_B_minor[izNeighMinor[i]].xypart();
							f64 ita_par_opp = p_ita_parallel_ion_minor[izNeighMinor[i]];
							f64 nu_theirs = p_nu_ion_minor[izNeighMinor[i]];
							if (shared_ita_par[threadIdx.x] < ita_par_opp) {
								ita_par = shared_ita_par[threadIdx.x];
								nu = shared_nu[threadIdx.x]; // why do I deliberately use the corresponding nu? nvm
							} else {
								ita_par = ita_par_opp;
								nu = nu_theirs;
							}
							if (ita_par_opp == 0.0) bUsableSide = false;
						}
					}
					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opp_B = Clockwise_d*opp_B;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opp_B = Anticlockwise_d*opp_B;
					}
					omega_ci = 0.5*qoverMc*(Make3(opp_B + shared_B[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
				}

				f64_vec2 endpt1 = THIRD * (nextpos + info.pos + opppos);
				if (bUsableSide) {
					// New definition of endpoint of minor edge:
					f64_vec2 gradvx, gradvy, gradviz;
					f64 area_quadrilateral = 0.5*(
						(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
						+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
						+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
						+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
						);
					gradvx.x = 0.5*(
						(our_v.x + next_v.x)*(info.pos.y - nextpos.y)
						+ (prev_v.x + our_v.x)*(prevpos.y - info.pos.y)
						+ (opp_v.x + prev_v.x)*(opppos.y - prevpos.y)
						+ (next_v.x + opp_v.x)*(nextpos.y - opppos.y) // nextpos = pos_anti, assumed
						) / area_quadrilateral;
					gradvx.y = -0.5*(
						(our_v.x + next_v.x)*(info.pos.x - nextpos.x)
						+ (prev_v.x + our_v.x)*(prevpos.x - info.pos.x)
						+ (opp_v.x + prev_v.x)*(opppos.x - prevpos.x)
						+ (next_v.x + opp_v.x)*(nextpos.x - opppos.x) // nextpos = pos_anti, assumed
						) / area_quadrilateral;

					gradvy.x = 0.5*(
						(our_v.y + next_v.y)*(info.pos.y - nextpos.y)
						+ (prev_v.y + our_v.y)*(prevpos.y - info.pos.y)
						+ (opp_v.y + prev_v.y)*(opppos.y - prevpos.y)
						+ (next_v.y + opp_v.y)*(nextpos.y - opppos.y) // nextpos = pos_anti, assumed
						) / area_quadrilateral;
					gradvy.y = -0.5*(
						(our_v.y + next_v.y)*(info.pos.x - nextpos.x)
						+ (prev_v.y + our_v.y)*(prevpos.x - info.pos.x)
						+ (opp_v.y + prev_v.y)*(opppos.x - prevpos.x)
						+ (next_v.y + opp_v.y)*(nextpos.x - opppos.x) // nextpos = pos_anti, assumed
						) / area_quadrilateral;

					gradviz.x = 0.5*(
						(our_v.z + next_v.z)*(info.pos.y - nextpos.y)
						+ (prev_v.z + our_v.z)*(prevpos.y - info.pos.y)
						+ (opp_v.z + prev_v.z)*(opppos.y - prevpos.y)
						+ (next_v.z + opp_v.z)*(nextpos.y - opppos.y) // nextpos = pos_anti, assumed
						) / area_quadrilateral;
					gradviz.y = -0.5*(
						(our_v.z + next_v.z)*(info.pos.x - nextpos.x)
						+ (prev_v.z + our_v.z)*(prevpos.x - info.pos.x)
						+ (opp_v.z + prev_v.z)*(opppos.x - prevpos.x)
						+ (next_v.z + opp_v.z)*(nextpos.x - opppos.x) // nextpos = pos_anti, assumed
						) / area_quadrilateral;


					if ((VISCMAG == 0) || (omega_ci.dot(omega_ci) < 0.01*0.1*nu*nu))
					{
						// run unmagnetised case
						f64 Pi_xx, Pi_xy, Pi_yx, Pi_yy, Pi_zx, Pi_zy;

						Pi_xx = -ita_par*THIRD*(4.0*gradvx.x - 2.0*gradvy.y);
						Pi_xy = -ita_par*(gradvx.y + gradvy.x);
						Pi_yx = Pi_xy;
						Pi_yy = -ita_par*THIRD*(4.0*gradvy.y - 2.0*gradvx.x);
						Pi_zx = -ita_par*(gradviz.x);
						Pi_zy = -ita_par*(gradviz.y);

						f64_vec2 edge_normal;
						edge_normal.x = endpt1.y - endpt0.y;
						edge_normal.y = endpt0.x - endpt1.x; // need to define so as to create unit vectors

						d_eps_by_d_beta.x = Factor*(Pi_xx*edge_normal.x + Pi_xy*edge_normal.y);
						d_eps_by_d_beta.y = Factor*(Pi_yx*edge_normal.x + Pi_yy*edge_normal.y);
						d_eps_by_d_beta.z = Factor*(Pi_zx*edge_normal.x + Pi_zy*edge_normal.y);

						// So we are saying if edge_normal.x > 0 and gradviz.x > 0
						// then Pi_zx < 0 then ownrates += a positive amount. That is correct.
					}
					else {
						f64_vec3 unit_b, unit_perp, unit_Hall;
						f64 omegamod;
						{
							f64_vec2 edge_normal;
							edge_normal.x = endpt1.y - endpt0.y;
							edge_normal.y = endpt0.x - endpt1.x; // need to define so as to create unit vectors

							f64 omegasq = omega_ci.dot(omega_ci);
							omegamod = sqrt(omegasq);
							unit_b = omega_ci / omegamod;
							unit_perp = Make3(edge_normal, 0.0) - unit_b*(unit_b.dotxy(edge_normal));
							unit_perp = unit_perp / unit_perp.modulus();
							unit_Hall = unit_b.cross(unit_perp); // Note sign.
																 // store omegamod instead.
						}
						f64 W_bb = 0.0, W_bP = 0.0, W_bH = 0.0, W_PP = 0.0, W_PH = 0.0, W_HH = 0.0; // these have to be alive at same time as 9 x partials
						{
							f64_vec3 intermed;

							// use: d vb / da = b transpose [ dvi/dxj ] a
							// Prototypical element: a.x b.y dvy/dx
							// b.x a.y dvx/dy

							intermed.x = unit_b.dotxy(gradvx);
							intermed.y = unit_b.dotxy(gradvy);
							intermed.z = unit_b.dotxy(gradviz);
							{
								f64 dvb_by_db, dvperp_by_db, dvHall_by_db;

								dvb_by_db = unit_b.dot(intermed);
								dvperp_by_db = unit_perp.dot(intermed);
								dvHall_by_db = unit_Hall.dot(intermed);

								W_bb += 4.0*THIRD*dvb_by_db;
								W_bP += dvperp_by_db;
								W_bH += dvHall_by_db;
								W_PP -= 2.0*THIRD*dvb_by_db;
								W_HH -= 2.0*THIRD*dvb_by_db;
							}
							{
								f64 dvb_by_dperp, dvperp_by_dperp,
									dvHall_by_dperp;
								// Optimize by getting rid of different labels.

								intermed.x = unit_perp.dotxy(gradvx);
								intermed.y = unit_perp.dotxy(gradvy);
								intermed.z = unit_perp.dotxy(gradviz);

								dvb_by_dperp = unit_b.dot(intermed);
								dvperp_by_dperp = unit_perp.dot(intermed);
								dvHall_by_dperp = unit_Hall.dot(intermed);

								W_bb -= 2.0*THIRD*dvperp_by_dperp;
								W_PP += 4.0*THIRD*dvperp_by_dperp;
								W_HH -= 2.0*THIRD*dvperp_by_dperp;
								W_bP += dvb_by_dperp;
								W_PH += dvHall_by_dperp;
							}
							{
								f64 dvb_by_dHall, dvperp_by_dHall, dvHall_by_dHall;

								intermed.x = unit_Hall.dotxy(gradvx);
								intermed.y = unit_Hall.dotxy(gradvy);
								intermed.z = unit_Hall.dotxy(gradviz);

								dvb_by_dHall = unit_b.dot(intermed);
								dvperp_by_dHall = unit_perp.dot(intermed);
								dvHall_by_dHall = unit_Hall.dot(intermed);

								W_bb -= 2.0*THIRD*dvHall_by_dHall;
								W_PP -= 2.0*THIRD*dvHall_by_dHall;
								W_HH += 4.0*THIRD*dvHall_by_dHall;
								W_bH += dvb_by_dHall;
								W_PH += dvperp_by_dHall;
							}
						}

						f64 Pi_b_b = 0.0, Pi_P_b = 0.0, Pi_P_P = 0.0, Pi_H_b = 0.0, Pi_H_P = 0.0, Pi_H_H = 0.0;
						{
							{
								f64 ita_1 = ita_par*(nu*nu / (nu*nu + omegamod*omegamod));

								Pi_b_b += -ita_par*W_bb;
								Pi_P_P += -0.5*(ita_par + ita_1)*W_PP - 0.5*(ita_par - ita_1)*W_HH;
								Pi_H_H += -0.5*(ita_par + ita_1)*W_HH - 0.5*(ita_par - ita_1)*W_PP;
								Pi_H_P += -ita_1*W_PH;
							}
							{
								f64 ita_2 = ita_par*(nu*nu / (nu*nu + 0.25*omegamod*omegamod));
								Pi_P_b += -ita_2*W_bP;
								Pi_H_b += -ita_2*W_bH;
							}
							{
								f64 ita_3 = ita_par*(nu*omegamod / (nu*nu + omegamod*omegamod));
								Pi_P_P -= ita_3*W_PH;
								Pi_H_H += ita_3*W_PH;
								Pi_H_P += 0.5*ita_3*(W_PP - W_HH);
							}
							{
								f64 ita_4 = 0.5*ita_par*(nu*omegamod / (nu*nu + 0.25*omegamod*omegamod));
								Pi_P_b += -ita_4*W_bH;
								Pi_H_b += ita_4*W_bP;
							}
						}

						f64 momflux_b, momflux_perp, momflux_Hall;
						{
							f64_vec3 mag_edge;
							f64_vec2 edge_normal;
							edge_normal.x = endpt1.y - endpt0.y;
							edge_normal.y = endpt0.x - endpt1.x; // need to define so as to create unit vectors
																 // Most efficient way: compute mom flux in magnetic coords
							mag_edge.x = unit_b.x*edge_normal.x + unit_b.y*edge_normal.y;
							mag_edge.y = unit_perp.x*edge_normal.x + unit_perp.y*edge_normal.y;
							mag_edge.z = unit_Hall.x*edge_normal.x + unit_Hall.y*edge_normal.y;
							momflux_b = -(Pi_b_b*mag_edge.x + Pi_P_b*mag_edge.y + Pi_H_b*mag_edge.z);
							momflux_perp = -(Pi_P_b*mag_edge.x + Pi_P_P*mag_edge.y + Pi_H_P*mag_edge.z);
							momflux_Hall = -(Pi_H_b*mag_edge.x + Pi_H_P*mag_edge.y + Pi_H_H*mag_edge.z);
						}
						// visc_contrib.x = over_m_i*(unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall);
						
						// Screen out looking out into insulator:
						// Not really needed since we did bUsableSide, but let's leave it in for now just to be delicate.						
						if (info.flag == CROSSING_INS) {
							char flag = p_info_minor[izNeighMinor[i]].flag;
							if ((flag == DOMAIN_TRIANGLE) || (flag == DOMAIN_VERTEX)) {
								d_eps_by_d_beta.x -= Factor*(unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall);
								d_eps_by_d_beta.y -= Factor*(unit_b.y*momflux_b + unit_perp.y*momflux_perp + unit_Hall.y*momflux_Hall);
								d_eps_by_d_beta.z -= Factor*(unit_b.z*momflux_b + unit_perp.z*momflux_perp + unit_Hall.z*momflux_Hall);
							} else {
								// DO NOTHING -- no additions
							}
						} else {
							d_eps_by_d_beta.x -= Factor*(unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall);
							d_eps_by_d_beta.y -= Factor*(unit_b.y*momflux_b + unit_perp.y*momflux_perp + unit_Hall.y*momflux_Hall);
							d_eps_by_d_beta.z -= Factor*(unit_b.z*momflux_b + unit_perp.z*momflux_perp + unit_Hall.z*momflux_Hall);
						};

					}
				}; // bUsableSide

				endpt0 = endpt1;
				prevpos = opppos;
				prev_v = opp_v;
				opppos = nextpos;
				opp_v = next_v;
			};

			memcpy(&(p_d_eps_by_d_beta_i_[iMinor]), &(d_eps_by_d_beta), sizeof(f64_vec3));
		}
		else {
			// Not domain tri or crossing_ins
			// Did we fairly model the insulator as a reflection of v?
		}
	} // scope

	__syncthreads();

	// Now do electron: overwrite ita and nu, copy-paste the above codes very carefully
	// OVERWRITE REGRESSOR

	shared_ita_par[threadIdx.x] = p_ita_parallel_elec_minor[iMinor];
	shared_nu[threadIdx.x] = p_nu_elec_minor[iMinor];
	shared_vJ[threadIdx.x] = p_Jacobi_elec[iMinor];

	if (threadIdx.x < threadsPerTileMajor) {
		info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];

		if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST))  // keeping consistent with ion above where we did put OUTERMOST here
		{// but we set ita to 0 in the pre routine for outermost.
			shared_ita_par_verts[threadIdx.x] = p_ita_parallel_elec_minor[iVertex + BEGINNING_OF_CENTRAL];
			shared_nu_verts[threadIdx.x] = p_nu_elec_minor[iVertex + BEGINNING_OF_CENTRAL];
			shared_vJ_verts[threadIdx.x] = p_Jacobi_elec[iVertex + BEGINNING_OF_CENTRAL];
		} else {
			shared_ita_par_verts[threadIdx.x] = 0.0;
			shared_nu_verts[threadIdx.x] = 0.0;
			memset(&(shared_vJ_verts[threadIdx.x]), 0, sizeof(f64_vec3));
		};
	};

	__syncthreads();

	if (threadIdx.x < threadsPerTileMajor) {
		
		long izTri[MAXNEIGH];
		char szPBC[MAXNEIGH];
		short tri_len = info.neigh_len; // ?!
		memcpy(izTri, p_izTri + iVertex*MAXNEIGH, MAXNEIGH * sizeof(long));
		memcpy(szPBC, p_szPBC + iVertex*MAXNEIGH, MAXNEIGH * sizeof(char));

		our_v = shared_vJ_verts[threadIdx.x]; // optimization: use replace or #define to get rid of storing this again.
		d_eps_by_d_beta = our_v;

		if ((info.flag == DOMAIN_VERTEX))
			//|| (info.flag == OUTERMOST)) 
		{

			f64 Factor = hsub / (p_n_minor[iVertex + BEGINNING_OF_CENTRAL].n * p_AreaMinor[iVertex + BEGINNING_OF_CENTRAL] * m_e);

			short iprev = tri_len - 1;
			short i = 0;
			short inext;
			if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMinor))
			{
				prev_v = shared_vJ[izTri[iprev] - StartMinor];
				prevpos = shared_pos[izTri[iprev] - StartMinor];
			} else {
				prev_v = p_Jacobi_elec[izTri[iprev]];
				prevpos = p_info_minor[izTri[iprev]].pos;
			}
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
				RotateClockwise(prev_v);
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
				RotateAnticlockwise(prev_v);
			}

			if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
			{
				opp_v = shared_vJ[izTri[i] - StartMinor];
				opppos = shared_pos[izTri[i] - StartMinor];
			} else {
				opp_v = p_Jacobi_elec[izTri[i]];
				opppos = p_info_minor[izTri[i]].pos;
			}
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
				RotateClockwise(opp_v);
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
				RotateAnticlockwise(opp_v);
			}

			f64_vec2 endpt0 = THIRD * (info.pos + opppos + prevpos);
			f64_vec3 omega_ce;
#pragma unroll 
			for (i = 0; i < tri_len; i++)
			{
				// Tri 0 is anticlockwise of neighbour 0, we think
				inext = i + 1; if (inext >= tri_len) inext = 0;

				if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
				{
					next_v = shared_vJ[izTri[inext] - StartMinor];
					nextpos = shared_pos[izTri[inext] - StartMinor];
				}
				else {
					next_v = p_Jacobi_elec[izTri[inext]];
					nextpos = p_info_minor[izTri[inext]].pos;
				}
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
					RotateClockwise(next_v);
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
					RotateAnticlockwise(next_v);
				}
				// All same as ion here:

				// Order of calculations may help things to go out/into scope at the right times so careful with that.
				f64_vec2 gradvx, gradvy, gradvez;

				f64 area_quadrilateral = 0.5*(
					(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
					+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
					+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
					+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
					);

				gradvx.x = 0.5*(
					(our_v.x + next_v.x)*(info.pos.y - nextpos.y)
					+ (prev_v.x + our_v.x)*(prevpos.y - info.pos.y)
					+ (opp_v.x + prev_v.x)*(opppos.y - prevpos.y)
					+ (next_v.x + opp_v.x)*(nextpos.y - opppos.y) // nextpos = pos_anti, assumed
					) / area_quadrilateral;
				gradvx.y = -0.5*(
					(our_v.x + next_v.x)*(info.pos.x - nextpos.x)
					+ (prev_v.x + our_v.x)*(prevpos.x - info.pos.x)
					+ (opp_v.x + prev_v.x)*(opppos.x - prevpos.x)
					+ (next_v.x + opp_v.x)*(nextpos.x - opppos.x) // nextpos = pos_anti, assumed
					) / area_quadrilateral;

				gradvy.x = 0.5*(
					(our_v.y + next_v.y)*(info.pos.y - nextpos.y)
					+ (prev_v.y + our_v.y)*(prevpos.y - info.pos.y)
					+ (opp_v.y + prev_v.y)*(opppos.y - prevpos.y)
					+ (next_v.y + opp_v.y)*(nextpos.y - opppos.y) // nextpos = pos_anti, assumed
					) / area_quadrilateral;
				gradvy.y = -0.5*(
					(our_v.y + next_v.y)*(info.pos.x - nextpos.x)
					+ (prev_v.y + our_v.y)*(prevpos.x - info.pos.x)
					+ (opp_v.y + prev_v.y)*(opppos.x - prevpos.x)
					+ (next_v.y + opp_v.y)*(nextpos.x - opppos.x) // nextpos = pos_anti, assumed
					) / area_quadrilateral;

				gradvez.x = 0.5*(
					(our_v.z + next_v.z)*(info.pos.y - nextpos.y)
					+ (prev_v.z + our_v.z)*(prevpos.y - info.pos.y)
					+ (opp_v.z + prev_v.z)*(opppos.y - prevpos.y)
					+ (next_v.z + opp_v.z)*(nextpos.y - opppos.y) // nextpos = pos_anti, assumed
					) / area_quadrilateral;
				gradvez.y = -0.5*(
					(our_v.z + next_v.z)*(info.pos.x - nextpos.x)
					+ (prev_v.z + our_v.z)*(prevpos.x - info.pos.x)
					+ (opp_v.z + prev_v.z)*(opppos.x - prevpos.x)
					+ (next_v.z + opp_v.z)*(nextpos.x - opppos.x) // nextpos = pos_anti, assumed
					) / area_quadrilateral;

				{
					f64_vec2 opp_B;
					f64 opp_ita, opp_nu;
					if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
					{
						opp_B = shared_B[izTri[i] - StartMinor];
						opp_ita = shared_ita_par[izTri[i] - StartMinor];
						opp_nu = shared_nu[izTri[i] - StartMinor];
						//ita_par = 0.5*(shared_ita_par_verts[threadIdx.x] + shared_ita_par[izTri[i] - StartMinor]);
						//nu = 0.5*(shared_nu_verts[threadIdx.x] + shared_nu[izTri[i] - StartMinor]);
					}
					else {
						opp_B = p_B_minor[izTri[i]].xypart();
						opp_ita = p_ita_parallel_elec_minor[izTri[i]];
						opp_nu = p_nu_elec_minor[izTri[i]];
					}
					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opp_B = Clockwise_d*opp_B;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opp_B = Anticlockwise_d*opp_B;
					}
					if (shared_ita_par_verts[threadIdx.x] < opp_ita) {
						ita_par = shared_ita_par_verts[threadIdx.x];
						nu = shared_nu_verts[threadIdx.x];
					}
					else {
						ita_par = opp_ita;
						nu = opp_nu;
					}
					omega_ce = 0.5*qovermc*(Make3(opp_B + shared_B_verts[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
				}

				f64_vec2 endpt1 = THIRD * (nextpos + info.pos + opppos);
				if ((VISCMAG == 0) || (omega_ce.dot(omega_ce) < 0.01*0.1*nu*nu))
				{
					// run unmagnetised case
					f64 Pi_xx, Pi_xy, Pi_yx, Pi_yy, Pi_zx, Pi_zy;

					// Let's suppose, Pi_yx means the rate of flow of y-momentum in the x direction.
					// Thus when we want to know how much y momentum is flowing through the wall we take
					// Pi_yx.edge_x + Pi_yy.edge_y -- reasonable.

					Pi_xx = -ita_par*THIRD*(4.0*gradvx.x - 2.0*gradvy.y);
					Pi_xy = -ita_par*(gradvx.y + gradvy.x);
					Pi_yx = Pi_xy;
					Pi_yy = -ita_par*THIRD*(4.0*gradvy.y - 2.0*gradvx.x);
					Pi_zx = -ita_par*(gradvez.x);
					Pi_zy = -ita_par*(gradvez.y);

					f64_vec2 edge_normal;
					edge_normal.x = endpt1.y - endpt0.y;
					edge_normal.y = endpt0.x - endpt1.x; // need to define so as to create unit vectors

					d_eps_by_d_beta.x += Factor*(Pi_xx*edge_normal.x + Pi_xy*edge_normal.y);
					d_eps_by_d_beta.y += Factor*(Pi_yx*edge_normal.x + Pi_yy*edge_normal.y);
					d_eps_by_d_beta.z += Factor*(Pi_zx*edge_normal.x + Pi_zy*edge_normal.y);
				} else {
					f64_vec3 unit_b, unit_perp, unit_Hall;
					f64 omegamod;
					{
						f64_vec2 edge_normal;
						edge_normal.x = endpt1.y - endpt0.y;
						edge_normal.y = endpt0.x - endpt1.x; // need to define so as to create unit vectors

						f64 omegasq = omega_ce.dot(omega_ce);
						omegamod = sqrt(omegasq);
						unit_b = omega_ce / omegamod;
						unit_perp = Make3(edge_normal, 0.0) - unit_b*(unit_b.dotxy(edge_normal));
						unit_perp = unit_perp / unit_perp.modulus();
						unit_Hall = unit_b.cross(unit_perp); // Note sign.
															 // store omegamod instead.
					}
					f64 W_bb = 0.0, W_bP = 0.0, W_bH = 0.0, W_PP = 0.0, W_PH = 0.0, W_HH = 0.0; // these have to be alive at same time as 9 x partials
					{
						f64_vec3 intermed;
						// use: d vb / da = b transpose [ dvi/dxj ] a
						// Prototypical element: a.x b.y dvy/dx
						// b.x a.y dvx/dy
						intermed.x = unit_b.dotxy(gradvx);
						intermed.y = unit_b.dotxy(gradvy);
						intermed.z = unit_b.dotxy(gradvez);
						{
							f64 dvb_by_db, dvperp_by_db, dvHall_by_db;
							dvb_by_db = unit_b.dot(intermed);
							dvperp_by_db = unit_perp.dot(intermed);
							dvHall_by_db = unit_Hall.dot(intermed);

							W_bb += 4.0*THIRD*dvb_by_db;
							W_bP += dvperp_by_db;
							W_bH += dvHall_by_db;
							W_PP -= 2.0*THIRD*dvb_by_db;
							W_HH -= 2.0*THIRD*dvb_by_db;
						}
						{
							f64 dvb_by_dperp, dvperp_by_dperp,
								dvHall_by_dperp;
							// Optimize by getting rid of different labels.

							intermed.x = unit_perp.dotxy(gradvx);
							intermed.y = unit_perp.dotxy(gradvy);
							intermed.z = unit_perp.dotxy(gradvez);

							dvb_by_dperp = unit_b.dot(intermed);
							dvperp_by_dperp = unit_perp.dot(intermed);
							dvHall_by_dperp = unit_Hall.dot(intermed);

							W_bb -= 2.0*THIRD*dvperp_by_dperp;
							W_PP += 4.0*THIRD*dvperp_by_dperp;
							W_HH -= 2.0*THIRD*dvperp_by_dperp;
							W_bP += dvb_by_dperp;
							W_PH += dvHall_by_dperp;
						}
						{
							f64 dvb_by_dHall, dvperp_by_dHall, dvHall_by_dHall;

							intermed.x = unit_Hall.dotxy(gradvx);
							intermed.y = unit_Hall.dotxy(gradvy);
							intermed.z = unit_Hall.dotxy(gradvez);

							dvb_by_dHall = unit_b.dot(intermed);
							dvperp_by_dHall = unit_perp.dot(intermed);
							dvHall_by_dHall = unit_Hall.dot(intermed);

							W_bb -= 2.0*THIRD*dvHall_by_dHall;
							W_PP -= 2.0*THIRD*dvHall_by_dHall;
							W_HH += 4.0*THIRD*dvHall_by_dHall;
							W_bH += dvb_by_dHall;
							W_PH += dvperp_by_dHall;
						}
					}

					f64 Pi_b_b = 0.0, Pi_P_b = 0.0, Pi_P_P = 0.0, Pi_H_b = 0.0, Pi_H_P = 0.0, Pi_H_H = 0.0;
					{
						{
							f64 ita_1 = ita_par*(nu*nu / (nu*nu + omegamod*omegamod));

							Pi_b_b += -ita_par*W_bb;
							Pi_P_P += -0.5*(ita_par + ita_1)*W_PP - 0.5*(ita_par - ita_1)*W_HH;
							Pi_H_H += -0.5*(ita_par + ita_1)*W_HH - 0.5*(ita_par - ita_1)*W_PP;
							Pi_H_P += -ita_1*W_PH;
						}
						{
							f64 ita_2 = ita_par*(nu*nu / (nu*nu + 0.25*omegamod*omegamod));
							Pi_P_b += -ita_2*W_bP;
							Pi_H_b += -ita_2*W_bH;
						}
						{
							f64 ita_3 = ita_par*(nu*omegamod / (nu*nu + omegamod*omegamod));
							Pi_P_P -= ita_3*W_PH;
							Pi_H_H += ita_3*W_PH;
							Pi_H_P += 0.5*ita_3*(W_PP - W_HH);
						}
						{
							f64 ita_4 = 0.5*ita_par*(nu*omegamod / (nu*nu + 0.25*omegamod*omegamod));
							Pi_P_b += -ita_4*W_bH;
							Pi_H_b += ita_4*W_bP;
						}
					}

					f64 momflux_b, momflux_perp, momflux_Hall;
					{
						f64_vec3 mag_edge;
						f64_vec2 edge_normal;
						edge_normal.x = endpt1.y - endpt0.y;
						edge_normal.y = endpt0.x - endpt1.x;
						// Most efficient way: compute mom flux in magnetic coords
						mag_edge.x = unit_b.x*edge_normal.x + unit_b.y*edge_normal.y;    // b component
						mag_edge.y = unit_perp.x*edge_normal.x + unit_perp.y*edge_normal.y; // P component
						mag_edge.z = unit_Hall.x*edge_normal.x + unit_Hall.y*edge_normal.y; // H component

						momflux_b = -(Pi_b_b*mag_edge.x + Pi_P_b*mag_edge.y + Pi_H_b*mag_edge.z);
						momflux_perp = -(Pi_P_b*mag_edge.x + Pi_P_P*mag_edge.y + Pi_H_P*mag_edge.z);
						momflux_Hall = -(Pi_H_b*mag_edge.x + Pi_H_P*mag_edge.y + Pi_H_H*mag_edge.z);
					}

					d_eps_by_d_beta.x -= Factor*(unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall);
					d_eps_by_d_beta.y -= Factor*(unit_b.y*momflux_b + unit_perp.y*momflux_perp + unit_Hall.y*momflux_Hall);
					d_eps_by_d_beta.z -= Factor*(unit_b.z*momflux_b + unit_perp.z*momflux_perp + unit_Hall.z*momflux_Hall);
				};

				endpt0 = endpt1;
				prevpos = opppos;
				prev_v = opp_v;
				opppos = nextpos;
				opp_v = next_v;
			}; // next i

			memcpy(p_d_eps_by_d_beta_e_ + iVertex + BEGINNING_OF_CENTRAL, &d_eps_by_d_beta, sizeof(f64_vec3));
		} else {
			// NOT domain vertex: Do nothing			
		};
	};
	
	// Electrons in tris:
	info = p_info_minor[iMinor];
	long izNeighMinor[6];
	char szPBC[6];
	memcpy(izNeighMinor, p_izNeighMinor + iMinor * 6, sizeof(long) * 6);
	memcpy(szPBC, p_szPBCtriminor + iMinor * 6, sizeof(char) * 6);
	
	if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL)) {
	}
	else {
		if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS)) {

			our_v = shared_vJ[threadIdx.x];
			d_eps_by_d_beta = our_v;

			f64 Factor = hsub / (p_n_minor[iMinor].n * p_AreaMinor[iMinor] * m_e);

			short inext, iprev = 5, i = 0;
			if ((izNeighMinor[iprev] >= StartMinor) && (izNeighMinor[iprev] < EndMinor))
			{
				memcpy(&prev_v, &(shared_vJ[izNeighMinor[iprev] - StartMinor]), sizeof(f64_vec3));
				prevpos = shared_pos[izNeighMinor[iprev] - StartMinor];
			}
			else {
				if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					memcpy(&prev_v, &(shared_vJ_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(f64_vec3));
					prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					prevpos = p_info_minor[izNeighMinor[iprev]].pos;
					memcpy(&prev_v, &(p_Jacobi_elec[izNeighMinor[iprev]]), sizeof(f64_vec3));
				};
			};
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
				RotateClockwise(prev_v);
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
				RotateAnticlockwise(prev_v);
			}

			i = 0;
			if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
			{
				memcpy(&opp_v, &(shared_vJ[izNeighMinor[i] - StartMinor]), sizeof(f64_vec3));
				opppos = shared_pos[izNeighMinor[i] - StartMinor];
			}
			else {
				if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					memcpy(&opp_v, &(shared_vJ_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(f64_vec3));
					opppos = shared_pos_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					opppos = p_info_minor[izNeighMinor[i]].pos;
					memcpy(&opp_v, &(p_Jacobi_elec[izNeighMinor[i]]), sizeof(f64_vec3));
				};
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
				RotateClockwise(opp_v);
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
				RotateAnticlockwise(opp_v);
			}

			f64_vec2 endpt0 = THIRD * (info.pos + opppos + prevpos);
			f64_vec3 omega_ce;
			// Let's make life easier and load up an array of 6 n's beforehand.
#pragma unroll 
			for (i = 0; i < 6; i++)
			{
				inext = i + 1; if (inext > 5) inext = 0;

				if ((izNeighMinor[inext] >= StartMinor) && (izNeighMinor[inext] < EndMinor))
				{
					memcpy(&next_v, &(shared_vJ[izNeighMinor[inext] - StartMinor]), sizeof(f64_vec3));
					nextpos = shared_pos[izNeighMinor[inext] - StartMinor];
				}
				else {
					if ((izNeighMinor[inext] >= StartMajor + BEGINNING_OF_CENTRAL) &&
						(izNeighMinor[inext] < EndMajor + BEGINNING_OF_CENTRAL))
					{
						memcpy(&next_v, &(shared_vJ_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(f64_vec3));
						nextpos = shared_pos_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
					}
					else {
						nextpos = p_info_minor[izNeighMinor[inext]].pos;
						memcpy(&next_v, &(p_Jacobi_elec[izNeighMinor[inext]]), sizeof(f64_vec3));
					};
				};
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
					RotateClockwise(next_v);
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
					RotateAnticlockwise(next_v);
				}

				bool bUsableSide = true;
				{
					f64_vec2 opp_B;
					f64 opp_ita, opp_nu;
					if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
					{
						opp_B = shared_B[izNeighMinor[i] - StartMinor];
						opp_ita = shared_ita_par[izNeighMinor[i] - StartMinor];
						opp_nu = shared_nu[izNeighMinor[i] - StartMinor];
						if (shared_ita_par[izNeighMinor[i] - StartMinor] == 0.0) bUsableSide = false;
					}
					else {
						if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
							(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
						{
							opp_B = shared_B_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
							opp_ita = shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
							opp_nu = shared_nu_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
							if (shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL] == 0.0) bUsableSide = false;
						}
						else {
							opp_B = p_B_minor[izNeighMinor[i]].xypart();
							opp_ita = p_ita_parallel_elec_minor[izNeighMinor[i]];
							opp_nu = p_nu_elec_minor[izNeighMinor[i]];
							if (opp_ita == 0.0) bUsableSide = false;
						}
					}
					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opp_B = Clockwise_d*opp_B;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opp_B = Anticlockwise_d*opp_B;
					}
					if (shared_ita_par[threadIdx.x] < opp_ita) {
						ita_par = shared_ita_par[threadIdx.x];
						nu = shared_nu[threadIdx.x];
					}
					else {
						ita_par = opp_ita;
						nu = opp_nu;
					}
					omega_ce = 0.5*qovermc*(Make3(opp_B + shared_B[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
				}

				f64_vec2 endpt1 = THIRD * (nextpos + info.pos + opppos);

				if (bUsableSide) {
					// New definition of endpoint of minor edge:
					f64_vec2 gradvez, gradvx, gradvy;

					f64 area_quadrilateral = 0.5*(
						(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
						+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
						+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
						+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
						);

					gradvx.x = 0.5*(
						(our_v.x + next_v.x)*(info.pos.y - nextpos.y)
						+ (prev_v.x + our_v.x)*(prevpos.y - info.pos.y)
						+ (opp_v.x + prev_v.x)*(opppos.y - prevpos.y)
						+ (next_v.x + opp_v.x)*(nextpos.y - opppos.y) // nextpos = pos_anti, assumed
						) / area_quadrilateral;
					gradvx.y = -0.5*(
						(our_v.x + next_v.x)*(info.pos.x - nextpos.x)
						+ (prev_v.x + our_v.x)*(prevpos.x - info.pos.x)
						+ (opp_v.x + prev_v.x)*(opppos.x - prevpos.x)
						+ (next_v.x + opp_v.x)*(nextpos.x - opppos.x) // nextpos = pos_anti, assumed
						) / area_quadrilateral;

					gradvy.x = 0.5*(
						(our_v.y + next_v.y)*(info.pos.y - nextpos.y)
						+ (prev_v.y + our_v.y)*(prevpos.y - info.pos.y)
						+ (opp_v.y + prev_v.y)*(opppos.y - prevpos.y)
						+ (next_v.y + opp_v.y)*(nextpos.y - opppos.y) // nextpos = pos_anti, assumed
						) / area_quadrilateral;
					gradvy.y = -0.5*(
						(our_v.y + next_v.y)*(info.pos.x - nextpos.x)
						+ (prev_v.y + our_v.y)*(prevpos.x - info.pos.x)
						+ (opp_v.y + prev_v.y)*(opppos.x - prevpos.x)
						+ (next_v.y + opp_v.y)*(nextpos.x - opppos.x) // nextpos = pos_anti, assumed
						) / area_quadrilateral;

					gradvez.x = 0.5*(
						(our_v.z + next_v.z)*(info.pos.y - nextpos.y)
						+ (prev_v.z + our_v.z)*(prevpos.y - info.pos.y)
						+ (opp_v.z + prev_v.z)*(opppos.y - prevpos.y)
						+ (next_v.z + opp_v.z)*(nextpos.y - opppos.y) // nextpos = pos_anti, assumed
						) / area_quadrilateral;
					gradvez.y = -0.5*(
						(our_v.z + next_v.z)*(info.pos.x - nextpos.x)
						+ (prev_v.z + our_v.z)*(prevpos.x - info.pos.x)
						+ (opp_v.z + prev_v.z)*(opppos.x - prevpos.x)
						+ (next_v.z + opp_v.z)*(nextpos.x - opppos.x) // nextpos = pos_anti, assumed
						) / area_quadrilateral;


					if ((VISCMAG == 0) || (omega_ce.dot(omega_ce) < 0.1*0.1*nu*nu))
					{
						// run unmagnetised case
						f64 Pi_xx, Pi_xy, Pi_yx, Pi_yy, Pi_zx, Pi_zy;

						Pi_xx = -ita_par*THIRD*(4.0*gradvx.x - 2.0*gradvy.y);
						Pi_xy = -ita_par*(gradvx.y + gradvy.x);
						Pi_yx = Pi_xy;
						Pi_yy = -ita_par*THIRD*(4.0*gradvy.y - 2.0*gradvx.x);
						Pi_zx = -ita_par*(gradvez.x);
						Pi_zy = -ita_par*(gradvez.y);

						f64_vec2 edge_normal;
						edge_normal.x = endpt1.y - endpt0.y;
						edge_normal.y = endpt0.x - endpt1.x; // need to define so as to create unit vectors

						if (info.flag == CROSSING_INS) {
							char flag = p_info_minor[izNeighMinor[i]].flag;
							if ((flag == DOMAIN_TRIANGLE) || (flag == DOMAIN_VERTEX)) {

								d_eps_by_d_beta.x += Factor *(Pi_xx*edge_normal.x + Pi_xy*edge_normal.y);
								d_eps_by_d_beta.y += Factor *(Pi_yx*edge_normal.x + Pi_yy*edge_normal.y);
								d_eps_by_d_beta.z += Factor *(Pi_zx*edge_normal.x + Pi_zy*edge_normal.y);
							} else {
								// DO NOTHING
							}
						} else {
							d_eps_by_d_beta.x += Factor *(Pi_xx*edge_normal.x + Pi_xy*edge_normal.y);
							d_eps_by_d_beta.y += Factor *(Pi_yx*edge_normal.x + Pi_yy*edge_normal.y);
							d_eps_by_d_beta.z += Factor *(Pi_zx*edge_normal.x + Pi_zy*edge_normal.y);
						}
					} else {
						f64_vec3 unit_b, unit_perp, unit_Hall;
						f64 omegamod;
						{
							f64_vec2 edge_normal;
							edge_normal.x = endpt1.y - endpt0.y;
							edge_normal.y = endpt0.x - endpt1.x; // need to define so as to create unit vectors

							f64 omegasq = omega_ce.dot(omega_ce);
							omegamod = sqrt(omegasq);
							unit_b = omega_ce / omegamod;
							unit_perp = Make3(edge_normal, 0.0) - unit_b*(unit_b.dotxy(edge_normal));
							unit_perp = unit_perp / unit_perp.modulus();
							unit_Hall = unit_b.cross(unit_perp); // Note sign.
																 // store omegamod instead.
						}
						f64 W_bb = 0.0, W_bP = 0.0, W_bH = 0.0, W_PP = 0.0, W_PH = 0.0, W_HH = 0.0; // these have to be alive at same time as 9 x partials
						{
							f64_vec3 intermed;

							// use: d vb / da = b transpose [ dvi/dxj ] a
							// Prototypical element: a.x b.y dvy/dx
							// b.x a.y dvx/dy

							intermed.x = unit_b.dotxy(gradvx);
							intermed.y = unit_b.dotxy(gradvy);
							intermed.z = unit_b.dotxy(gradvez);
							{
								f64 dvb_by_db, dvperp_by_db, dvHall_by_db;

								dvb_by_db = unit_b.dot(intermed);
								dvperp_by_db = unit_perp.dot(intermed);
								dvHall_by_db = unit_Hall.dot(intermed);

								W_bb += 4.0*THIRD*dvb_by_db;
								W_bP += dvperp_by_db;
								W_bH += dvHall_by_db;
								W_PP -= 2.0*THIRD*dvb_by_db;
								W_HH -= 2.0*THIRD*dvb_by_db;
							}
							{
								f64 dvb_by_dperp, dvperp_by_dperp,
									dvHall_by_dperp;
								// Optimize by getting rid of different labels.

								intermed.x = unit_perp.dotxy(gradvx);
								intermed.y = unit_perp.dotxy(gradvy);
								intermed.z = unit_perp.dotxy(gradvez);

								dvb_by_dperp = unit_b.dot(intermed);
								dvperp_by_dperp = unit_perp.dot(intermed);
								dvHall_by_dperp = unit_Hall.dot(intermed);

								W_bb -= 2.0*THIRD*dvperp_by_dperp;
								W_PP += 4.0*THIRD*dvperp_by_dperp;
								W_HH -= 2.0*THIRD*dvperp_by_dperp;
								W_bP += dvb_by_dperp;
								W_PH += dvHall_by_dperp;
							}
							{
								f64 dvb_by_dHall, dvperp_by_dHall, dvHall_by_dHall;

								intermed.x = unit_Hall.dotxy(gradvx);
								intermed.y = unit_Hall.dotxy(gradvy);
								intermed.z = unit_Hall.dotxy(gradvez);

								dvb_by_dHall = unit_b.dot(intermed);
								dvperp_by_dHall = unit_perp.dot(intermed);
								dvHall_by_dHall = unit_Hall.dot(intermed);

								W_bb -= 2.0*THIRD*dvHall_by_dHall;
								W_PP -= 2.0*THIRD*dvHall_by_dHall;
								W_HH += 4.0*THIRD*dvHall_by_dHall;
								W_bH += dvb_by_dHall;
								W_PH += dvperp_by_dHall;
							}
						}

						f64 Pi_b_b = 0.0, Pi_P_b = 0.0, Pi_P_P = 0.0, Pi_H_b = 0.0, Pi_H_P = 0.0, Pi_H_H = 0.0;
						{
							{
								f64 ita_1 = ita_par*(nu*nu / (nu*nu + omegamod*omegamod));

								Pi_b_b += -ita_par*W_bb;
								Pi_P_P += -0.5*(ita_par + ita_1)*W_PP - 0.5*(ita_par - ita_1)*W_HH;
								Pi_H_H += -0.5*(ita_par + ita_1)*W_HH - 0.5*(ita_par - ita_1)*W_PP;
								Pi_H_P += -ita_1*W_PH;
							}
							{
								f64 ita_2 = ita_par*(nu*nu / (nu*nu + 0.25*omegamod*omegamod));
								Pi_P_b += -ita_2*W_bP;
								Pi_H_b += -ita_2*W_bH;
							}
							{
								f64 ita_3 = ita_par*(nu*omegamod / (nu*nu + omegamod*omegamod));
								Pi_P_P -= ita_3*W_PH;
								Pi_H_H += ita_3*W_PH;
								Pi_H_P += 0.5*ita_3*(W_PP - W_HH);
							}
							{
								f64 ita_4 = 0.5*ita_par*(nu*omegamod / (nu*nu + 0.25*omegamod*omegamod));
								Pi_P_b += -ita_4*W_bH;
								Pi_H_b += ita_4*W_bP;
							}
						}

						f64 momflux_b, momflux_perp, momflux_Hall;
						{
							f64_vec3 mag_edge;
							f64_vec2 edge_normal;
							edge_normal.x = endpt1.y - endpt0.y;
							edge_normal.y = endpt0.x - endpt1.x; // need to define so as to create unit vectors
																 // Most efficient way: compute mom flux in magnetic coords
							mag_edge.x = unit_b.x*edge_normal.x + unit_b.y*edge_normal.y;
							mag_edge.y = unit_perp.x*edge_normal.x + unit_perp.y*edge_normal.y;
							mag_edge.z = unit_Hall.x*edge_normal.x + unit_Hall.y*edge_normal.y;

							momflux_b = -(Pi_b_b*mag_edge.x + Pi_P_b*mag_edge.y + Pi_H_b*mag_edge.z);
							momflux_perp = -(Pi_P_b*mag_edge.x + Pi_P_P*mag_edge.y + Pi_H_P*mag_edge.z);
							momflux_Hall = -(Pi_H_b*mag_edge.x + Pi_H_P*mag_edge.y + Pi_H_H*mag_edge.z);
						}

						// unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall
						// is the flow of p_x dotted with the edge_normal
						// ownrates will be divided by N to give dv/dt
						// m N dvx/dt = integral div momflux_x
						// Therefore divide here just by m
						
						
						if (info.flag == CROSSING_INS) {
							char flag = p_info_minor[izNeighMinor[i]].flag;
							if ((flag == DOMAIN_TRIANGLE) || (flag == DOMAIN_VERTEX)) {
								d_eps_by_d_beta.x -= Factor*(unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall);
								d_eps_by_d_beta.y -= Factor*(unit_b.y*momflux_b + unit_perp.y*momflux_perp + unit_Hall.y*momflux_Hall);
								d_eps_by_d_beta.z -= Factor*(unit_b.z*momflux_b + unit_perp.z*momflux_perp + unit_Hall.z*momflux_Hall);
							} else {
								// DO NOTHING
							}
						} else {
							d_eps_by_d_beta.x -= Factor*(unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall);
							d_eps_by_d_beta.y -= Factor*(unit_b.y*momflux_b + unit_perp.y*momflux_perp + unit_Hall.y*momflux_Hall);
							d_eps_by_d_beta.z -= Factor*(unit_b.z*momflux_b + unit_perp.z*momflux_perp + unit_Hall.z*momflux_Hall);
						}
					}
				}; // bUsableSide

				endpt0 = endpt1;
				prevpos = opppos;
				prev_v = opp_v;
				opppos = nextpos;
				opp_v = next_v;
			};
			memcpy(&(p_d_eps_by_d_beta_e_[iMinor]), &(d_eps_by_d_beta), sizeof(f64_vec3));
		}
		else {
			// Not domain, not crossing_ins, not a frill			
		} // non-domain tri
	}; // was it FRILL
}
__global__ void kernelCreate_viscous_contrib_to_MAR_and_NT(

	structural * __restrict__ p_info_minor,
	v4 * __restrict__ p_vie_minor,
	
	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtriminor,

	f64 * __restrict__ p_ita_parallel_ion_minor,   // nT / nu ready to look up
	f64 * __restrict__ p_ita_parallel_elec_minor,   // nT / nu ready to look up
	f64 * __restrict__ p_nu_ion_minor,   // nT / nu ready to look up
	f64 * __restrict__ p_nu_elec_minor,   // nT / nu ready to look up
	f64_vec3 * __restrict__ p_B_minor,

	f64_vec3 * __restrict__ p_MAR_ion,
	f64_vec3 * __restrict__ p_MAR_elec,
	NTrates * __restrict__ p_NT_addition_rate,
	NTrates * __restrict__ p_NT_addition_tri)
{
	__shared__ v4 shared_vie[threadsPerTileMinor]; // sort of thing we want as input
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];
	__shared__ f64_vec2 shared_B[threadsPerTileMinor];
	__shared__ f64 shared_ita_par[threadsPerTileMinor]; // reuse for i,e ; or make 2 vars to combine the routines.
	__shared__ f64 shared_nu[threadsPerTileMinor];
	
	__shared__ v4 shared_vie_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_B_verts[threadsPerTileMajor];
	__shared__ f64 shared_ita_par_verts[threadsPerTileMajor];
	__shared__ f64 shared_nu_verts[threadsPerTileMajor]; // used for creating ita_perp, ita_cross

	// 4+2+2+1+1 *1.5 = 15 per thread. That is possibly as slow as having 24 per thread. 
	// Thus putting some stuff in shared may speed up if there are spills.
	
	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	long const iVertex = threadsPerTileMajor*blockIdx.x + threadIdx.x; // only meaningful threadIdx.x < threadsPerTileMajor

	long const StartMinor = threadsPerTileMinor*blockIdx.x;
	long const StartMajor = threadsPerTileMajor*blockIdx.x;
	long const EndMinor = StartMinor + threadsPerTileMinor;
	long const EndMajor = StartMajor + threadsPerTileMajor;

	v4 our_v, opp_v, prev_v, next_v;
	f64_vec2 opppos, prevpos, nextpos;
	f64 nu, ita_par;  // optimization: we always each loop want to get rid of omega, nu once we have calc'd these, if possible!!
	f64_vec3 ownrates_visc;
	f64 visc_htg;

	shared_pos[threadIdx.x] = p_info_minor[iMinor].pos;
	shared_vie[threadIdx.x] = p_vie_minor[iMinor];
	shared_B[threadIdx.x] = p_B_minor[iMinor].xypart();
	shared_ita_par[threadIdx.x] = p_ita_parallel_ion_minor[iMinor];
	shared_nu[threadIdx.x] = p_nu_ion_minor[iMinor];

	// Perhaps the real answer is this. Advection and therefore advective momflux
	// do not need to be recalculated very often at all. At 1e6 cm/s, we aim for 1 micron,
	// get 1e-10s to actually do the advection !!
	// So an outer cycle. Still limiting the number of total things in a minor tile. We might like 384 = 192*2.

	structural info;
	if (threadIdx.x < threadsPerTileMajor) {
		info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
		shared_pos_verts[threadIdx.x] = info.pos;
		shared_B_verts[threadIdx.x] = p_B_minor[iVertex + BEGINNING_OF_CENTRAL].xypart();
		if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) 
		{
			memcpy(&(shared_vie_verts[threadIdx.x]), &(p_vie_minor[iVertex + BEGINNING_OF_CENTRAL]), sizeof(v4));
			shared_ita_par_verts[threadIdx.x] = p_ita_parallel_ion_minor[iVertex + BEGINNING_OF_CENTRAL];
			shared_nu_verts[threadIdx.x] = p_nu_ion_minor[iVertex + BEGINNING_OF_CENTRAL];
			// But now I am going to set ita == 0 in OUTERMOST and agree never to look there because that's fairer than one-way traffic and I don't wanna handle OUTERMOST?
			// I mean, I could handle it, and do flows only if the flag does not come up OUTER_FRILL.
			// OK just do that.
		} else {
			// it still manages to coalesce, let's hope, because ShardModel is 13 doubles not 12.
			memset(&(shared_vie_verts[threadIdx.x]), 0, sizeof(v4));
			shared_ita_par_verts[threadIdx.x] = 0.0;
			shared_nu_verts[threadIdx.x] = 0.0;
		};
	};
	
	__syncthreads();

	// How shall we arrange to do v_n, which is isotropic? Handle this first...
	// Is the v_n coefficient negligible? Check.

	// We actually have to think how to handle the x-y dimension. PopOhms will handle it.
	
	// We can re-use some shared data -- such as pos and B -- to do both ions and electrons
	// But they use different ita_par and different vez, viz. 
	// Often we don't need to do magnetised ion viscosity when we do magnetised electron.
	
	// IONS FIRST:

	if (threadIdx.x < threadsPerTileMajor) {
		memset(&ownrates_visc, 0, sizeof(f64_vec3));
		visc_htg = 0.0;

		long izTri[MAXNEIGH_d];
		char szPBC[MAXNEIGH_d];
		short tri_len = info.neigh_len; // ?!
		
		if ((info.flag == DOMAIN_VERTEX))
			//|| (info.flag == OUTERMOST)) 
		{
			// We are losing energy if there is viscosity into OUTERMOST.
			memcpy(izTri, p_izTri + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(long));
			memcpy(szPBC, p_szPBC + iVertex*MAXNEIGH_d, MAXNEIGH_d * sizeof(char));
			
			our_v = shared_vie_verts[threadIdx.x]; // optimization: use replace or #define to get rid of storing this again.

			short iprev = tri_len - 1;
			short i = 0;
			short inext;
			if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMinor))
			{
				prev_v = shared_vie[izTri[iprev] - StartMinor];
				prevpos = shared_pos[izTri[iprev] - StartMinor];
			}
			else {
				prev_v = p_vie_minor[izTri[iprev]];
				prevpos = p_info_minor[izTri[iprev]].pos;
			}
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
				prev_v.vxy = Clockwise_d*prev_v.vxy;
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
				prev_v.vxy = Anticlockwise_d*prev_v.vxy;
			}

			if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
			{
				opp_v = shared_vie[izTri[i] - StartMinor];
				opppos = shared_pos[izTri[i] - StartMinor];
			}
			else {
				opp_v = p_vie_minor[izTri[i]];
				opppos = p_info_minor[izTri[i]].pos;
			}
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
				opp_v.vxy = Clockwise_d*opp_v.vxy;
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
				opp_v.vxy = Anticlockwise_d*opp_v.vxy;
			}

			// DROP THIS ONE.

	//		f64_vec2 endpt0 = THIRD * (info.pos + opppos + prevpos);

			// short iend = tri_len;
			//f64_vec2 projendpt0;
			//if ((info.flag == INNERMOST) || (info.flag == OUTERMOST)) {
			//	iend = tri_len - 2;
			//	if (info.flag == OUTERMOST) {
			//		endpt0.project_to_radius(projendpt0, FRILL_CENTROID_OUTER_RADIUS_d); // back of cell for Lap purposes
			//	}
			//	else {
			//		endpt0.project_to_radius(projendpt0, FRILL_CENTROID_INNER_RADIUS_d); // back of cell for Lap purposes
			//	}
			//	edge_normal.x = endpt0.y - projendpt0.y;
			//	edge_normal.y = projendpt0.x - endpt0.x;
			//	AreaMinor += (0.5*projendpt0.x + 0.5*endpt0.x)*edge_normal.x;
			//};

			f64_vec3 omega_ci; 
			
			// ** Be especially vigilant to the changes we need to make to go from ion to electron.
#pragma unroll 
			for (i = 0; i < tri_len; i++)
			{
				// Tri 0 is anticlockwise of neighbour 0, we think
				inext = i + 1; if (inext >= tri_len) inext = 0;
				
				// Now sort out anticlock vars:

				if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
				{
					next_v = shared_vie[izTri[inext] - StartMinor];
					nextpos = shared_pos[izTri[inext] - StartMinor];
				}
				else {
					next_v = p_vie_minor[izTri[inext]];
					nextpos = p_info_minor[izTri[inext]].pos;
				}
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
					next_v.vxy = Clockwise_d*next_v.vxy;
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
					next_v.vxy = Anticlockwise_d*next_v.vxy;
				}

				// Order of calculations may help things to go out/into scope at the right times so careful with that.
				f64_vec2 gradvx, gradvy, gradviz;

				f64 area_quadrilateral = 0.5*(
					(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
					+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
					+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
					+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
					);

				gradvx.x = 0.5*(
					(our_v.vxy.x + next_v.vxy.x)*(info.pos.y - nextpos.y)
					+ (prev_v.vxy.x + our_v.vxy.x)*(prevpos.y - info.pos.y)
					+ (opp_v.vxy.x + prev_v.vxy.x)*(opppos.y - prevpos.y)
					+ (next_v.vxy.x + opp_v.vxy.x)*(nextpos.y - opppos.y) // nextpos = pos_anti, assumed
					) / area_quadrilateral;
				gradvx.y = -0.5*(
					(our_v.vxy.x + next_v.vxy.x)*(info.pos.x - nextpos.x)
					+ (prev_v.vxy.x + our_v.vxy.x)*(prevpos.x - info.pos.x)
					+ (opp_v.vxy.x + prev_v.vxy.x)*(opppos.x - prevpos.x)
					+ (next_v.vxy.x + opp_v.vxy.x)*(nextpos.x - opppos.x) // nextpos = pos_anti, assumed
					) / area_quadrilateral;

				gradvy.x = 0.5*(
					(our_v.vxy.y + next_v.vxy.y)*(info.pos.y - nextpos.y)
					+ (prev_v.vxy.y + our_v.vxy.y)*(prevpos.y - info.pos.y)
					+ (opp_v.vxy.y + prev_v.vxy.y)*(opppos.y - prevpos.y)
					+ (next_v.vxy.y + opp_v.vxy.y)*(nextpos.y - opppos.y) // nextpos = pos_anti, assumed
					) / area_quadrilateral;
				gradvy.y = -0.5*(
					(our_v.vxy.y + next_v.vxy.y)*(info.pos.x - nextpos.x)
					+ (prev_v.vxy.y + our_v.vxy.y)*(prevpos.x - info.pos.x)
					+ (opp_v.vxy.y + prev_v.vxy.y)*(opppos.x - prevpos.x)
					+ (next_v.vxy.y + opp_v.vxy.y)*(nextpos.x - opppos.x) // nextpos = pos_anti, assumed
					) / area_quadrilateral;
//
//				if (TEST) printf(
//					"iVertex %d our_v.y next prev opp %1.8E %1.8E %1.8E %1.8E\n"
//					"area_quad %1.8E \n"
//					"info.pos %1.8E %1.8E opppos %1.8E %1.8E prev %1.8E %1.8E next %1.8E %1.8E\n",
//					iVertex, our_v.vxy.y, next_v.vxy.y, prev_v.vxy.y, opp_v.vxy.y,
//					area_quadrilateral,
//					info.pos.x, info.pos.y, opppos.x, opppos.y, prevpos.x, prevpos.y, nextpos.x, nextpos.y);
//
				gradviz.x = 0.5*(
					(our_v.viz + next_v.viz)*(info.pos.y - nextpos.y)
					+ (prev_v.viz + our_v.viz)*(prevpos.y - info.pos.y)
					+ (opp_v.viz + prev_v.viz)*(opppos.y - prevpos.y)
					+ (next_v.viz + opp_v.viz)*(nextpos.y - opppos.y) // nextpos = pos_anti, assumed
					) / area_quadrilateral;
				gradviz.y = -0.5*(
					(our_v.viz + next_v.viz)*(info.pos.x - nextpos.x)
					+ (prev_v.viz + our_v.viz)*(prevpos.x - info.pos.x)
					+ (opp_v.viz + prev_v.viz)*(opppos.x - prevpos.x)
					+ (next_v.viz + opp_v.viz)*(nextpos.x - opppos.x) // nextpos = pos_anti, assumed
					) / area_quadrilateral;

				//f64_vec2 endpt1 = THIRD * (nextpos + info.pos + opppos);
				
				// we also want to get nu from somewhere. So precompute nu at the time we precompute ita_e = n Te / nu_e, ita_i = n Ti / nu_i. 
				
				{
					f64_vec2 opp_B;
					if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
					{
						opp_B = shared_B[izTri[i] - StartMinor];
						if (shared_ita_par_verts[threadIdx.x] < shared_ita_par[izTri[i] - StartMinor])
						{
							ita_par = shared_ita_par_verts[threadIdx.x];
							nu = shared_nu_verts[threadIdx.x];
						} else {
							ita_par = shared_ita_par[izTri[i] - StartMinor];
							nu = shared_nu[izTri[i] - StartMinor];
						};
					}
					else {
						opp_B = p_B_minor[izTri[i]].xypart();
						f64 ita_theirs = p_ita_parallel_ion_minor[izTri[i]];
						f64 nu_theirs = p_nu_ion_minor[izTri[i]];
						if (shared_ita_par_verts[threadIdx.x] < ita_theirs) {
							ita_par = shared_ita_par_verts[threadIdx.x];
							nu = shared_nu_verts[threadIdx.x];
						} else {
							ita_par = ita_theirs;
							nu = nu_theirs;
						};						
					}
					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opp_B = Clockwise_d*opp_B;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opp_B = Anticlockwise_d*opp_B;
					}
					omega_ci = 0.5*qoverMc*(Make3(opp_B + shared_B_verts[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
				} // Guaranteed DOMAIN_VERTEX never needs to skip an edge; we include CROSSING_INS in viscosity.

				if ((VISCMAG == 0) || (omega_ci.dot(omega_ci) < 0.01*0.1*nu*nu))
				{
					// run unmagnetised case
					f64 Pi_xx, Pi_xy, Pi_yx, Pi_yy, Pi_zx, Pi_zy;

					Pi_xx = -ita_par*THIRD*(4.0*gradvx.x - 2.0*gradvy.y);
					Pi_xy = -ita_par*(gradvx.y + gradvy.x);
					Pi_yx = Pi_xy;
					Pi_yy = -ita_par*THIRD*(4.0*gradvy.y - 2.0*gradvx.x);
					Pi_zx = -ita_par*(gradviz.x);
					Pi_zy = -ita_par*(gradviz.y);

					f64_vec2 edge_normal;
					edge_normal.x = THIRD * (nextpos.y - prevpos.y);
					edge_normal.y = THIRD * (prevpos.x - nextpos.x); 

					f64_vec3 visc_contrib;
					visc_contrib.x = -over_m_i*(Pi_xx*edge_normal.x + Pi_xy*edge_normal.y);
					visc_contrib.y = -over_m_i*(Pi_yx*edge_normal.x + Pi_yy*edge_normal.y);
					visc_contrib.z = -over_m_i*(Pi_zx*edge_normal.x + Pi_zy*edge_normal.y);

					//if (info.flag == OUTERMOST) {
					//	if (p_info_minor[izTri[i]].flag == DOMAIN_TRIANGLE)
					//	{
					//		ownrates_visc += visc_contrib;

					//		visc_htg += -THIRD*m_ion*(
					//			(our_v.vxy.x - opp_v.vxy.x)*visc_contrib.x
					//			+ (our_v.vxy.y - opp_v.vxy.y)*visc_contrib.y
					//			+ (our_v.viz - opp_v.viz)*visc_contrib.z);
					//		// do not look into frill
					//	}
					//	else {
					//		visc_contrib.x = 0.0; visc_contrib.y = 0.0; visc_contrib.z = 0.0;
					//	}
					//} else
					{
						ownrates_visc += visc_contrib;

						visc_htg += -THIRD*m_ion*(
							(our_v.vxy.x - opp_v.vxy.x)*visc_contrib.x
							+ (our_v.vxy.y - opp_v.vxy.y)*visc_contrib.y
							+ (our_v.viz - opp_v.viz)*visc_contrib.z);
					}

//					
					if (TEST)
						printf("iVertex %d tri %d ION ita_par %1.9E \n"
							"gradvx %1.8E %1.8E gradvy %1.8E %1.8E gradvz %1.8E %1.8E\n"
							"edgenormal %1.8E %1.8E  opp_viz %1.10E our_viz %1.10E\n"
							"ourpos %1.8E %1.8E opp pos %1.8E %1.8E\n"
							"Pi_xx %1.8E xy %1.8E yy %1.8E zx %1.8E\n"
							"visc_contrib %1.9E %1.9E %1.9E visc_htg %1.10E\n"
							"===\n",
							iVertex, izTri[i], ita_par, gradvx.x, gradvx.y, gradvy.x, gradvy.y, 
							gradviz.x, gradviz.y,
							edge_normal.x, edge_normal.y, opp_v.viz, our_v.viz,
							info.pos.x,info.pos.y, opppos.x,opppos.y,
							Pi_xx, Pi_xy, Pi_yy, Pi_zx,
							visc_contrib.x, visc_contrib.y, visc_contrib.z, visc_htg
						);
//
					// So we are saying if edge_normal.x > 0 and gradviz.x > 0
					// then Pi_zx < 0 then ownrates += a positive amount. That is correct.
				} else {

					f64_vec3 unit_b, unit_perp, unit_Hall;
					f64 omegamod;
					{
						f64_vec2 edge_normal;
						edge_normal.x = THIRD * (nextpos.y - prevpos.y);
						edge_normal.y = THIRD * (prevpos.x - nextpos.x);

						f64 omegasq = omega_ci.dot(omega_ci);
						omegamod = sqrt(omegasq);
						unit_b = omega_ci / omegamod;
						unit_perp = Make3(edge_normal, 0.0) - unit_b*(unit_b.dotxy(edge_normal));
						unit_perp = unit_perp / unit_perp.modulus();
						unit_Hall = unit_b.cross(unit_perp); // Note sign.

						// store omegamod instead.
					//	ita_perp = FACTOR_PERP * ita_par * nu*nu / (omegasq + nu*nu);
					//	ita_cross = FACTOR_HALL * ita_par * nu*omegamod / (omegasq + nu*nu);
					}
					
					
					f64 W_bb = 0.0, W_bP = 0.0, W_bH = 0.0, W_PP = 0.0, W_PH = 0.0, W_HH = 0.0; // these have to be alive at same time as 9 x partials
					// but we can make do with 3x partials
					// 2. Now get partials in magnetic coordinates 
					{						
						f64_vec3 intermed;

						// use: d vb / da = b transpose [ dvi/dxj ] a
						// Prototypical element: a.x b.y dvy/dx
						// b.x a.y dvx/dy

						intermed.x = unit_b.dotxy(gradvx);
						intermed.y = unit_b.dotxy(gradvy);
						intermed.z = unit_b.dotxy(gradviz);
						{
							f64 dvb_by_db, dvperp_by_db, dvHall_by_db;

							dvb_by_db = unit_b.dot(intermed);
							dvperp_by_db = unit_perp.dot(intermed);
							dvHall_by_db = unit_Hall.dot(intermed);

							W_bb += 4.0*THIRD*dvb_by_db;
							W_bP += dvperp_by_db;
							W_bH += dvHall_by_db;
							W_PP -= 2.0*THIRD*dvb_by_db;
							W_HH -= 2.0*THIRD*dvb_by_db;
						}
						{
							f64 dvb_by_dperp, dvperp_by_dperp,
								dvHall_by_dperp;
							// Optimize by getting rid of different labels.

							intermed.x = unit_perp.dotxy(gradvx);
							intermed.y = unit_perp.dotxy(gradvy);
							intermed.z = unit_perp.dotxy(gradviz);

							dvb_by_dperp = unit_b.dot(intermed);
							dvperp_by_dperp = unit_perp.dot(intermed);
							dvHall_by_dperp = unit_Hall.dot(intermed);

							W_bb -= 2.0*THIRD*dvperp_by_dperp;
							W_PP += 4.0*THIRD*dvperp_by_dperp;
							W_HH -= 2.0*THIRD*dvperp_by_dperp;
							W_bP += dvb_by_dperp;
							W_PH += dvHall_by_dperp;
						}
						{
							f64 dvb_by_dHall, dvperp_by_dHall, dvHall_by_dHall;

							intermed.x = unit_Hall.dotxy(gradvx);
							intermed.y = unit_Hall.dotxy(gradvy);
							intermed.z = unit_Hall.dotxy(gradviz);

							dvb_by_dHall = unit_b.dot(intermed);
							dvperp_by_dHall = unit_perp.dot(intermed);
							dvHall_by_dHall = unit_Hall.dot(intermed);

							W_bb -= 2.0*THIRD*dvHall_by_dHall;
							W_PP -= 2.0*THIRD*dvHall_by_dHall;
							W_HH += 4.0*THIRD*dvHall_by_dHall;
							W_bH += dvb_by_dHall;
							W_PH += dvperp_by_dHall;
						}
					}
					
					f64 Pi_b_b = 0.0, Pi_P_b = 0.0, Pi_P_P = 0.0, Pi_H_b = 0.0, Pi_H_P = 0.0, Pi_H_H = 0.0;
					{
						{
							f64 ita_1 = ita_par*(nu*nu / (nu*nu + omegamod*omegamod));

							Pi_b_b += -ita_par*W_bb;
							Pi_P_P += -0.5*(ita_par + ita_1)*W_PP - 0.5*(ita_par - ita_1)*W_HH;
							Pi_H_H += -0.5*(ita_par + ita_1)*W_HH - 0.5*(ita_par - ita_1)*W_PP;
							Pi_H_P += -ita_1*W_PH;
						}
						{
							f64 ita_2 = ita_par*(nu*nu / (nu*nu + 0.25*omegamod*omegamod));
							Pi_P_b += -ita_2*W_bP;
							Pi_H_b += -ita_2*W_bH;
						}
						{
							f64 ita_3 = ita_par*(nu*omegamod / (nu*nu + omegamod*omegamod));
							Pi_P_P -= ita_3*W_PH;
							Pi_H_H += ita_3*W_PH;
							Pi_H_P += 0.5*ita_3*(W_PP - W_HH);
						}
						{
							f64 ita_4 = 0.5*ita_par*(nu*omegamod / (nu*nu + 0.25*omegamod*omegamod));
							Pi_P_b += -ita_4*W_bH;
							Pi_H_b += ita_4*W_bP;
						}
					}
					
					
					// Now wish to get W out of scope. But we can't easily do this.
					
					// All we want left over at this point is Pi

					f64 momflux_b, momflux_perp, momflux_Hall;
					{
						f64_vec2 edge_normal;
						edge_normal.x = THIRD * (nextpos.y - prevpos.y);
						edge_normal.y = THIRD * (prevpos.x - nextpos.x);  // need to define so as to create unit vectors

						// Most efficient way: compute mom flux in magnetic coords
						f64_vec3 mag_edge;
						mag_edge.x = unit_b.x*edge_normal.x + unit_b.y*edge_normal.y;
						mag_edge.y = unit_perp.x*edge_normal.x + unit_perp.y*edge_normal.y;
						mag_edge.z = unit_Hall.x*edge_normal.x + unit_Hall.y*edge_normal.y;

						momflux_b = -(Pi_b_b*mag_edge.x + Pi_P_b*mag_edge.y + Pi_H_b*mag_edge.z);
						momflux_perp = -(Pi_P_b*mag_edge.x + Pi_P_P*mag_edge.y + Pi_H_P*mag_edge.z);
						momflux_Hall = -(Pi_H_b*mag_edge.x + Pi_H_P*mag_edge.y + Pi_H_H*mag_edge.z);
					}

					// unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall
					// is the flow of p_x dotted with the edge_normal
					// ownrates will be divided by N to give dv/dt
					// m N dvx/dt = integral div momflux_x
					// Therefore divide here just by m

					f64_vec3 visc_contrib;
					visc_contrib.x = over_m_i*(unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall);
					visc_contrib.y = over_m_i*(unit_b.y*momflux_b + unit_perp.y*momflux_perp + unit_Hall.y*momflux_Hall);
					visc_contrib.z = over_m_i*(unit_b.z*momflux_b + unit_perp.z*momflux_perp + unit_Hall.z*momflux_Hall);

					//if (info.flag == OUTERMOST) {
					//	if (p_info_minor[izTri[i]].flag == DOMAIN_TRIANGLE)	{
					//		ownrates_visc += visc_contrib;

					//		visc_htg += -TWOTHIRDS*m_ion*(
					//			(our_v.vxy.x - opp_v.vxy.x)*visc_contrib.x
					//			+ (our_v.vxy.y - opp_v.vxy.y)*visc_contrib.y
					//			+ (our_v.viz - opp_v.viz)*visc_contrib.z);  // Claim all visc htg for this vertcell
					//	}
					//} else 
					{
						ownrates_visc += visc_contrib;

						visc_htg += -TWOTHIRDS*m_ion*(
							(our_v.vxy.x - opp_v.vxy.x)*visc_contrib.x
							+ (our_v.vxy.y - opp_v.vxy.y)*visc_contrib.y
							+ (our_v.viz - opp_v.viz)*visc_contrib.z);  // Claim all visc htg for this vertcell
					}
					

					if (TEST)
						printf("iVertex %d tri %d ION ita_par %1.9E omega %1.9E %1.9E %1.9E nu %1.9E\n"
							"ourpos %1.8E %1.8E opp pos %1.8E %1.8E\n"
							"v %1.9E %1.9E %1.9E their v %1.9E %1.9E %1.9E\n"
							"momflux b %1.8E perp %1.8E cross %1.8E \n"
							"visc_contrib %1.9E %1.9E %1.9E \n"
							"visc_htg %1.10E %1.10E %1.10E \n"
							"===\n",
							iVertex, izTri[i], ita_par, omega_ci.x, omega_ci.y, omega_ci.z, nu,
							info.pos.x, info.pos.y, opppos.x, opppos.y,
							our_v.vxy.x, our_v.vxy.y, our_v.viz, opp_v.vxy.x, opp_v.vxy.y, opp_v.viz,
							momflux_b, momflux_perp, momflux_Hall,
							visc_contrib.x, visc_contrib.y, visc_contrib.z,
							-TWOTHIRDS*m_ion*(our_v.vxy.x - opp_v.vxy.x)*visc_contrib.x,
							-TWOTHIRDS*m_ion*(our_v.vxy.y - opp_v.vxy.y)*visc_contrib.y,
							-TWOTHIRDS*m_ion*(our_v.viz - opp_v.viz)*visc_contrib.z
						);
						

					// sign: ours is gaining visc_contrib so negate that effect to add to NT
					// THIRD because 2/3 factor appears in heat energy, but we divide by 2 because split between this cell and opposing.
										
					// To do:
					// Do heating similarly for 3 more and add to global (discard if < 0)
					// put in test given CROSSING_INS that we do not look into ins
					// Consider changing pressure calcs similarly???
					// Add in heating when we destroy momentum in CROSSING_INS - careful there, center lives on ins.
					// Ideally, add neutral viscosity routine.
				}
				
				// MAR_elec -= Make3(0.5*(n0 * T0.Te + n1 * T1.Te)*over_m_e*edge_normal, 0.0);
				// v0.vez = vie_k.vez + h_use * MAR.z / (n_use.n*AreaMinor);

				prevpos = opppos;
				prev_v = opp_v;
				opppos = nextpos;
				opp_v = next_v;				
			}; // next i
			
			f64_vec3 ownrates;
			memcpy(&ownrates, &(p_MAR_ion[iVertex + BEGINNING_OF_CENTRAL]), sizeof(f64_vec3));
		//	if (TEST) 
		//		printf("%d ion ownrates %1.8E %1.8E %1.8E ownrates_visc %1.8E %1.8E %1.8E our_v %1.8E %1.8E %1.8E\n",
		//		iVertex, ownrates.x, ownrates.y, ownrates.z, ownrates_visc.x, ownrates_visc.y, ownrates_visc.z, our_v.vxy.x, our_v.vxy.y, our_v.viz);
			ownrates += ownrates_visc;
			memcpy(p_MAR_ion + iVertex + BEGINNING_OF_CENTRAL, &ownrates, sizeof(f64_vec3));
			
			p_NT_addition_rate[iVertex].NiTi += visc_htg;
			
#ifdef DEBUGNANS
			if (ownrates.x != ownrates.x)
				printf("iVertex %d NaN ownrates.x\n",iVertex);
			if (ownrates.y != ownrates.y)
				printf("iVertex %d NaN ownrates.y\n", iVertex);
			if (ownrates.z != ownrates.z)
				printf("iVertex %d NaN ownrates.z\n", iVertex);			
			if (visc_htg != visc_htg) printf("iVertex %d NAN VISC HTG\n", iVertex);
#endif

		} else {
			// NOT domain vertex: Do nothing			
		};
	}; 
	// __syncthreads(); // end of first vertex part
	// Do we need syncthreads? Not overwriting any shared data here...
	
	info = p_info_minor[iMinor];

	// memcpy(&(ownrates), &(p_MAR_ion[iMinor]), sizeof(f64_vec3));
	memset(&ownrates_visc, 0, sizeof(f64_vec3));
	
	visc_htg = 0.0;

	our_v = shared_vie[threadIdx.x];

	//if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL)) {
	{	
		long izNeighMinor[6];
		char szPBC[6];

		if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS)) {

			memcpy(izNeighMinor, p_izNeighMinor + iMinor * 6, sizeof(long) * 6);
			memcpy(szPBC, p_szPBCtriminor + iMinor * 6, sizeof(char) * 6);

			short inext, iprev = 5, i = 0;
			if ((izNeighMinor[iprev] >= StartMinor) && (izNeighMinor[iprev] < EndMinor))
			{
				memcpy(&prev_v, &(shared_vie[izNeighMinor[iprev] - StartMinor]), sizeof(v4));
				prevpos = shared_pos[izNeighMinor[iprev] - StartMinor];
			}
			else {
				if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					memcpy(&prev_v, &(shared_vie_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(v4));
					prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					prevpos = p_info_minor[izNeighMinor[iprev]].pos;
					memcpy(&prev_v, &(p_vie_minor[izNeighMinor[iprev]]), sizeof(v4));
				};
			};
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
				prev_v.vxy = Clockwise_d*prev_v.vxy;
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
				prev_v.vxy = Anticlockwise_d*prev_v.vxy;
			}
			
			i = 0;
			if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
			{
				memcpy(&opp_v, &(shared_vie[izNeighMinor[i] - StartMinor]), sizeof(v4));
				opppos = shared_pos[izNeighMinor[i] - StartMinor];
			}
			else {
				if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					memcpy(&opp_v, &(shared_vie_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(v4));
					opppos = shared_pos_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					opppos = p_info_minor[izNeighMinor[i]].pos;
					memcpy(&opp_v, &(p_vie_minor[izNeighMinor[i]]), sizeof(v4));
				};
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
				opp_v.vxy = Clockwise_d*opp_v.vxy;
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
				opp_v.vxy = Anticlockwise_d*opp_v.vxy;
			}

			f64_vec2 endpt0 = THIRD * (info.pos + opppos + prevpos);
			
			f64_vec3 omega_ci;
			// Let's make life easier and load up an array of 6 n's beforehand.
#pragma unroll 
			for (i = 0; i < 6; i++)
			{
				inext = i + 1; if (inext > 5) inext = 0;

				if ((izNeighMinor[inext] >= StartMinor) && (izNeighMinor[inext] < EndMinor))
				{
					memcpy(&next_v, &(shared_vie[izNeighMinor[inext] - StartMinor]), sizeof(v4));
					nextpos = shared_pos[izNeighMinor[inext] - StartMinor];
				}
				else {
					if ((izNeighMinor[inext] >= StartMajor + BEGINNING_OF_CENTRAL) &&
						(izNeighMinor[inext] < EndMajor + BEGINNING_OF_CENTRAL))
					{
						memcpy(&next_v, &(shared_vie_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(v4));
						nextpos = shared_pos_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
					}
					else {
						nextpos = p_info_minor[izNeighMinor[inext]].pos;
						memcpy(&next_v, &(p_vie_minor[izNeighMinor[inext]]), sizeof(v4));
					};
				};
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
					next_v.vxy = Clockwise_d*next_v.vxy;
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
					next_v.vxy = Anticlockwise_d*next_v.vxy;
				}
				
			//	nu = 1.0e10; // DEBUG
				bool bUsableSide = true;
				{
					f64_vec2 opp_B(0.0, 0.0);
					// newly uncommented:
					if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
					{
						opp_B = shared_B[izNeighMinor[i] - StartMinor];
						if (shared_ita_par[threadIdx.x] < shared_ita_par[izNeighMinor[i] - StartMinor])
						{
							ita_par = shared_ita_par[threadIdx.x];
							nu = shared_nu[threadIdx.x];
						} else {
							ita_par = shared_ita_par[izNeighMinor[i] - StartMinor];
							nu = shared_nu[izNeighMinor[i] - StartMinor];
						};
						
						if (shared_ita_par[izNeighMinor[i] - StartMinor] == 0.0) bUsableSide = false;
					}
					else {
						if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
							(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
						{
							opp_B = shared_B_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
							if (shared_ita_par[threadIdx.x] < shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL])
							{
								ita_par = shared_ita_par[threadIdx.x];
								nu = shared_nu[threadIdx.x];
							} else {
								ita_par = shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
								nu = shared_nu_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
							};
							
							if (shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL] == 0.0) bUsableSide = false;
						} else {
							opp_B = p_B_minor[izNeighMinor[i]].xypart();
							f64 ita_par_opp = p_ita_parallel_ion_minor[izNeighMinor[i]];
							f64 nu_theirs = p_nu_ion_minor[izNeighMinor[i]];
							if (shared_ita_par[threadIdx.x] < ita_par_opp) {
								ita_par = shared_ita_par[threadIdx.x];
								nu = shared_nu[threadIdx.x]; // why do I deliberately use the corresponding nu? nvm
							} else {
								ita_par = ita_par_opp;
								nu = nu_theirs;
							}
							 
							if (ita_par_opp == 0.0) bUsableSide = false;
						}
					}
					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opp_B = Clockwise_d*opp_B;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opp_B = Anticlockwise_d*opp_B;
					}
					omega_ci = 0.5*qoverMc*(Make3(opp_B + shared_B[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
				}

			//	f64_vec2 endpt1 = THIRD * (nextpos + info.pos + opppos);

				if (bUsableSide) {

					// New definition of endpoint of minor edge:
					f64_vec2 gradvx, gradvy, gradviz;

					f64 area_quadrilateral = 0.5*(
						(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
						+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
						+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
						+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
						);

					gradvx.x = 0.5*(
						(our_v.vxy.x + next_v.vxy.x)*(info.pos.y - nextpos.y)
						+ (prev_v.vxy.x + our_v.vxy.x)*(prevpos.y - info.pos.y)
						+ (opp_v.vxy.x + prev_v.vxy.x)*(opppos.y - prevpos.y)
						+ (next_v.vxy.x + opp_v.vxy.x)*(nextpos.y - opppos.y) // nextpos = pos_anti, assumed
						) / area_quadrilateral;
					gradvx.y = -0.5*(
						(our_v.vxy.x + next_v.vxy.x)*(info.pos.x - nextpos.x)
						+ (prev_v.vxy.x + our_v.vxy.x)*(prevpos.x - info.pos.x)
						+ (opp_v.vxy.x + prev_v.vxy.x)*(opppos.x - prevpos.x)
						+ (next_v.vxy.x + opp_v.vxy.x)*(nextpos.x - opppos.x) // nextpos = pos_anti, assumed
						) / area_quadrilateral;

					gradvy.x = 0.5*(
						(our_v.vxy.y + next_v.vxy.y)*(info.pos.y - nextpos.y)
						+ (prev_v.vxy.y + our_v.vxy.y)*(prevpos.y - info.pos.y)
						+ (opp_v.vxy.y + prev_v.vxy.y)*(opppos.y - prevpos.y)
						+ (next_v.vxy.y + opp_v.vxy.y)*(nextpos.y - opppos.y) // nextpos = pos_anti, assumed
						) / area_quadrilateral;
					gradvy.y = -0.5*(
						(our_v.vxy.y + next_v.vxy.y)*(info.pos.x - nextpos.x)
						+ (prev_v.vxy.y + our_v.vxy.y)*(prevpos.x - info.pos.x)
						+ (opp_v.vxy.y + prev_v.vxy.y)*(opppos.x - prevpos.x)
						+ (next_v.vxy.y + opp_v.vxy.y)*(nextpos.x - opppos.x) // nextpos = pos_anti, assumed
						) / area_quadrilateral;

					gradviz.x = 0.5*(
						(our_v.viz + next_v.viz)*(info.pos.y - nextpos.y)
						+ (prev_v.viz + our_v.viz)*(prevpos.y - info.pos.y)
						+ (opp_v.viz + prev_v.viz)*(opppos.y - prevpos.y)
						+ (next_v.viz + opp_v.viz)*(nextpos.y - opppos.y) // nextpos = pos_anti, assumed
						) / area_quadrilateral;
					gradviz.y = -0.5*(
						(our_v.viz + next_v.viz)*(info.pos.x - nextpos.x)
						+ (prev_v.viz + our_v.viz)*(prevpos.x - info.pos.x)
						+ (opp_v.viz + prev_v.viz)*(opppos.x - prevpos.x)
						+ (next_v.viz + opp_v.viz)*(nextpos.x - opppos.x) // nextpos = pos_anti, assumed
						) / area_quadrilateral;


					if ((VISCMAG == 0) || (omega_ci.dot(omega_ci) < 0.01*0.1*nu*nu))
					{
						// run unmagnetised case
						f64 Pi_xx, Pi_xy, Pi_yx, Pi_yy, Pi_zx, Pi_zy;

						Pi_xx = -ita_par*THIRD*(4.0*gradvx.x - 2.0*gradvy.y);
						Pi_xy = -ita_par*(gradvx.y + gradvy.x);
						Pi_yx = Pi_xy;
						Pi_yy = -ita_par*THIRD*(4.0*gradvy.y - 2.0*gradvx.x);
						Pi_zx = -ita_par*(gradviz.x);
						Pi_zy = -ita_par*(gradviz.y);

						f64_vec2 edge_normal;
						edge_normal.x = THIRD * (nextpos.y - prevpos.y);
						edge_normal.y = THIRD * (prevpos.x - nextpos.x);  // need to define so as to create unit vectors

						f64_vec3 visc_contrib;
						visc_contrib.x = -over_m_i*(Pi_xx*edge_normal.x + Pi_xy*edge_normal.y);
						visc_contrib.y = -over_m_i*(Pi_yx*edge_normal.x + Pi_yy*edge_normal.y);
						visc_contrib.z = -over_m_i*(Pi_zx*edge_normal.x + Pi_zy*edge_normal.y);

						if (info.flag == CROSSING_INS) {
							char flag = p_info_minor[izNeighMinor[i]].flag;
							if ((flag == DOMAIN_TRIANGLE) || (flag == DOMAIN_VERTEX)) {
								ownrates_visc += visc_contrib;

								if (i % 2 == 0) {
									// vertex : heat collected by vertex
								}
								else {
									visc_htg += -THIRD*m_ion*(
										(our_v.vxy.x - opp_v.vxy.x)*visc_contrib.x
										+ (our_v.vxy.y - opp_v.vxy.y)*visc_contrib.y
										+ (our_v.viz - opp_v.viz)*visc_contrib.z);
									// And we are going to give it to what? Just spread it out after.

								}
							}
							else {
								// DO NOTHING
							}
						} else {
							ownrates_visc += visc_contrib;

							if (i % 2 == 0) {
								// vertex : heat collected by vertex
							}
							else {
								visc_htg += -THIRD*m_ion*(
									(our_v.vxy.x - opp_v.vxy.x)*visc_contrib.x
									+ (our_v.vxy.y - opp_v.vxy.y)*visc_contrib.y
									+ (our_v.viz - opp_v.viz)*visc_contrib.z);
							}

							if (TESTTRI) {
								printf("iMinor %d %d "
									" ita_par %1.11E nu %1.11E omega %1.9E %1.9E %1.9E \n"
									"gradvx %1.9E %1.9E our vx %1.9E theirs %1.9E\n"
									"gradvy %1.9E %1.9E our vy %1.9E theirs %1.9E\n"
									"gradvz %1.9E %1.9E our vz %1.9E theirs %1.9E\n"
									"visc contrib %1.10E %1.10E %1.10E\n"
									"visc htg %1.10E %1.10E %1.10E | running %1.10E \n"
									" *************************************** \n",
									iMinor, izNeighMinor[i],
									ita_par, // Think nu is what breaks it
									nu, omega_ci.x, omega_ci.y, omega_ci.z,
									gradvx.x, gradvx.y, our_v.vxy.x, opp_v.vxy.x,
									gradvy.x, gradvy.y, our_v.vxy.y, opp_v.vxy.y,
									gradviz.x, gradviz.y, our_v.viz, opp_v.viz,
									visc_contrib.x, visc_contrib.y, visc_contrib.z,
									-THIRD*m_ion*(our_v.vxy.x - opp_v.vxy.x)*visc_contrib.x,
									-THIRD*m_ion*(our_v.vxy.y - opp_v.vxy.y)*visc_contrib.y,
									-THIRD*m_ion*(our_v.viz - opp_v.viz)*visc_contrib.z,
									visc_htg
									);

							//	printf("iMinor %d visc_contrib.z %1.10E our-opp %1.10E z htg %1.10E | running %1.10E \n"
							//		" *************************************** \n",
							//		iMinor, visc_contrib.z, our_v.viz - opp_v.viz,
							//		-(our_v.viz - opp_v.viz)*THIRD*m_ion*visc_contrib.z,
							//		visc_htg);
							}
						}

						// So we are saying if edge_normal.x > 0 and gradviz.x > 0
						// then Pi_zx < 0 then ownrates += a positive amount. That is correct.
					} else {
						f64_vec3 unit_b, unit_perp, unit_Hall;
						f64 omegamod;
						{
							f64_vec2 edge_normal;
							edge_normal.x = THIRD * (nextpos.y - prevpos.y);
							edge_normal.y = THIRD * (prevpos.x - nextpos.x); // need to define so as to create unit vectors

							f64 omegasq = omega_ci.dot(omega_ci);
							omegamod = sqrt(omegasq);
							unit_b = omega_ci / omegamod;
							unit_perp = Make3(edge_normal, 0.0) - unit_b*(unit_b.dotxy(edge_normal));
							unit_perp = unit_perp / unit_perp.modulus();
							unit_Hall = unit_b.cross(unit_perp); // Note sign.
																 // store omegamod instead.
						}
						
						f64 W_bb = 0.0, W_bP = 0.0, W_bH = 0.0, W_PP = 0.0, W_PH = 0.0, W_HH = 0.0; // these have to be alive at same time as 9 x partials
						{
							f64_vec3 intermed;

							// use: d vb / da = b transpose [ dvi/dxj ] a
							// Prototypical element: a.x b.y dvy/dx
							// b.x a.y dvx/dy

							intermed.x = unit_b.dotxy(gradvx);
							intermed.y = unit_b.dotxy(gradvy);
							intermed.z = unit_b.dotxy(gradviz);
							{
								f64 dvb_by_db, dvperp_by_db, dvHall_by_db;

								dvb_by_db = unit_b.dot(intermed);
								dvperp_by_db = unit_perp.dot(intermed);
								dvHall_by_db = unit_Hall.dot(intermed);

								W_bb += 4.0*THIRD*dvb_by_db;
								W_bP += dvperp_by_db;
								W_bH += dvHall_by_db;
								W_PP -= 2.0*THIRD*dvb_by_db;
								W_HH -= 2.0*THIRD*dvb_by_db;
							}
							{
								f64 dvb_by_dperp, dvperp_by_dperp,
									dvHall_by_dperp;
								// Optimize by getting rid of different labels.

								intermed.x = unit_perp.dotxy(gradvx);
								intermed.y = unit_perp.dotxy(gradvy);
								intermed.z = unit_perp.dotxy(gradviz);

								dvb_by_dperp = unit_b.dot(intermed);
								dvperp_by_dperp = unit_perp.dot(intermed);
								dvHall_by_dperp = unit_Hall.dot(intermed);

								W_bb -= 2.0*THIRD*dvperp_by_dperp;
								W_PP += 4.0*THIRD*dvperp_by_dperp;
								W_HH -= 2.0*THIRD*dvperp_by_dperp;
								W_bP += dvb_by_dperp;
								W_PH += dvHall_by_dperp;
							}
							{
								f64 dvb_by_dHall, dvperp_by_dHall, dvHall_by_dHall;

								intermed.x = unit_Hall.dotxy(gradvx);
								intermed.y = unit_Hall.dotxy(gradvy);
								intermed.z = unit_Hall.dotxy(gradviz);

								dvb_by_dHall = unit_b.dot(intermed);
								dvperp_by_dHall = unit_perp.dot(intermed);
								dvHall_by_dHall = unit_Hall.dot(intermed);

								W_bb -= 2.0*THIRD*dvHall_by_dHall;
								W_PP -= 2.0*THIRD*dvHall_by_dHall;
								W_HH += 4.0*THIRD*dvHall_by_dHall;
								W_bH += dvb_by_dHall;
								W_PH += dvperp_by_dHall;
							}
						}

						f64 Pi_b_b = 0.0, Pi_P_b = 0.0, Pi_P_P = 0.0, Pi_H_b = 0.0, Pi_H_P = 0.0, Pi_H_H = 0.0;
						{
							{
								f64 ita_1 = ita_par*(nu*nu / (nu*nu + omegamod*omegamod));

								Pi_b_b += -ita_par*W_bb;
								Pi_P_P += -0.5*(ita_par + ita_1)*W_PP - 0.5*(ita_par - ita_1)*W_HH;
								Pi_H_H += -0.5*(ita_par + ita_1)*W_HH - 0.5*(ita_par - ita_1)*W_PP;
								Pi_H_P += -ita_1*W_PH;
							}
							{
								f64 ita_2 = ita_par*(nu*nu / (nu*nu + 0.25*omegamod*omegamod));
								Pi_P_b += -ita_2*W_bP;
								Pi_H_b += -ita_2*W_bH;
							}
							{
								f64 ita_3 = ita_par*(nu*omegamod / (nu*nu + omegamod*omegamod));
								Pi_P_P -= ita_3*W_PH;
								Pi_H_H += ita_3*W_PH;
								Pi_H_P += 0.5*ita_3*(W_PP - W_HH);
							}
							{
								f64 ita_4 = 0.5*ita_par*(nu*omegamod / (nu*nu + 0.25*omegamod*omegamod));
								Pi_P_b += -ita_4*W_bH;
								Pi_H_b += ita_4*W_bP;
							}
						}

						f64 momflux_b, momflux_perp, momflux_Hall;
						{
							f64_vec3 mag_edge;
							f64_vec2 edge_normal;
							edge_normal.x = THIRD * (nextpos.y - prevpos.y);
							edge_normal.y = THIRD * (prevpos.x - nextpos.x);  // need to define so as to create unit vectors

							// Most efficient way: compute mom flux in magnetic coords
							mag_edge.x = unit_b.x*edge_normal.x + unit_b.y*edge_normal.y;
							mag_edge.y = unit_perp.x*edge_normal.x + unit_perp.y*edge_normal.y;
							mag_edge.z = unit_Hall.x*edge_normal.x + unit_Hall.y*edge_normal.y;

							momflux_b = -(Pi_b_b*mag_edge.x + Pi_P_b*mag_edge.y + Pi_H_b*mag_edge.z);
							momflux_perp = -(Pi_P_b*mag_edge.x + Pi_P_P*mag_edge.y + Pi_H_P*mag_edge.z);
							momflux_Hall = -(Pi_H_b*mag_edge.x + Pi_H_P*mag_edge.y + Pi_H_H*mag_edge.z);

							//if (TESTTRI)
							//	printf("iMinor %d %d edge_normal %1.10E %1.10E mag_edge (b,P,H) %1.10E %1.10E %1.10E\n"
							//		"Pi_b_b %1.10E Pi_b_P %1.10E Pi_b_H %1.10E \n"
							//		"Pi_P_b %1.10E Pi_P_P %1.10E Pi_P_H %1.10E \n"
							//		"Pi_H_b %1.10E Pi_H_P %1.10E Pi_H_H %1.10E \n",
							//		iMinor, izNeighMinor[i], edge_normal.x, edge_normal.y, mag_edge.x, mag_edge.y, mag_edge.z,// b,P,H
							//		Pi_b_b, Pi_P_b, Pi_H_b,
							//		Pi_P_b, Pi_P_P, Pi_H_P,
							//		Pi_H_b, Pi_H_P, Pi_H_H);
						}

						// Time to double-check carefully the signs.
						// Pi was defined with - on dv/dx and we then dot that with the edge_normal, so giving + if we are higher than outside.
						
						// unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall
						// is the flow of p_x dotted with the edge_normal
						// ownrates will be divided by N to give dv/dt
						// m N dvx/dt = integral div momflux_x
						// Therefore divide here just by m
						f64_vec3 visc_contrib;
						visc_contrib.x = over_m_i*(unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall);
						visc_contrib.y = over_m_i*(unit_b.y*momflux_b + unit_perp.y*momflux_perp + unit_Hall.y*momflux_Hall);
						visc_contrib.z = over_m_i*(unit_b.z*momflux_b + unit_perp.z*momflux_perp + unit_Hall.z*momflux_Hall);

						if (TESTTRI)
							printf("%d %d over_m_i %1.9E \n"
								"unit_b %1.9E %1.9E %1.9E \n"
								"unit_perp %1.9E %1.9E %1.9E \n"
								"unit_Hall %1.9E %1.9E %1.9E \n"
								"momflux_b %1.9E momflux_perp %1.9E momflux_Hall %1.9E\n"
								"ita_par %1.10E \n",
								
								iMinor, izNeighMinor[i], over_m_i, unit_b.x, unit_b.y, unit_b.z,
								unit_perp.x, unit_perp.y, unit_perp.z,
								unit_Hall.x, unit_Hall.y, unit_Hall.z,
								momflux_b, momflux_perp, momflux_Hall,
								ita_par);

						// Screen out looking out into insulator:

						// Not really needed since we did bUsableSide, but let's leave it in for now just to be delicate.
						if (info.flag == CROSSING_INS) {
							char flag = p_info_minor[izNeighMinor[i]].flag;
							if ((flag == DOMAIN_TRIANGLE) || (flag == DOMAIN_VERTEX)) {
								ownrates_visc += visc_contrib;
								if (i % 2 != 0) // not vertex
								{
									visc_htg += -THIRD*m_ion*(
										(our_v.vxy.x - opp_v.vxy.x)*visc_contrib.x
										+ (our_v.vxy.y - opp_v.vxy.y)*visc_contrib.y
										+ (our_v.viz - opp_v.viz)*visc_contrib.z);
								}
								// NOTE: ISSUES
							} else {
								// DO NOTHING
							}
						} else {
							ownrates_visc += visc_contrib;
							if (i % 2 != 0) // not vertex
								visc_htg += -THIRD*m_ion*((our_v.vxy.x - opp_v.vxy.x)*visc_contrib.x
									+ (our_v.vxy.y - opp_v.vxy.y)*visc_contrib.y
									+ (our_v.viz - opp_v.viz)*visc_contrib.z);

							if (TESTTRI) 
								printf("iMinor %d %d "
									" ita_par %1.11E nu %1.11E omega %1.9E %1.9E %1.9E \n"
									"gradvx %1.9E %1.9E our vx %1.9E theirs %1.9E\n"
									"gradvy %1.9E %1.9E our vy %1.9E theirs %1.9E\n"
									"gradvz %1.9E %1.9E our vz %1.9E theirs %1.9E\n"
									"visc contrib %1.10E %1.10E %1.10E\n"
									"visc htg %1.10E %1.10E %1.10E | running %1.10E \n"
									" *************************************** \n",
									iMinor, izNeighMinor[i],
									ita_par, // Think nu is what breaks it
									nu, omega_ci.x, omega_ci.y, omega_ci.z,
									gradvx.x, gradvx.y, our_v.vxy.x, opp_v.vxy.x,
									gradvy.x, gradvy.y, our_v.vxy.y, opp_v.vxy.y,
									gradviz.x, gradviz.y, our_v.viz, opp_v.viz,
									visc_contrib.x, visc_contrib.y, visc_contrib.z,
									-THIRD*m_ion*(our_v.vxy.x - opp_v.vxy.x)*visc_contrib.x,
									-THIRD*m_ion*(our_v.vxy.y - opp_v.vxy.y)*visc_contrib.y,
									-THIRD*m_ion*(our_v.viz - opp_v.viz)*visc_contrib.z,
									visc_htg
								);
						}

						// MAR_elec -= Make3(0.5*(n0 * T0.Te + n1 * T1.Te)*over_m_e*edge_normal, 0.0);
						// v0.vez = vie_k.vez + h_use * MAR.z / (n_use.n*AreaMinor);
					}
				}; // bUsableSide

//				endpt0 = endpt1;
				prevpos = opppos;
				prev_v = opp_v;
				opppos = nextpos;
				opp_v = next_v;
			};

			f64_vec3 ownrates;
			memcpy(&ownrates,&(p_MAR_ion[iMinor]), sizeof(f64_vec3));
			ownrates += ownrates_visc;
			memcpy(&(p_MAR_ion[iMinor]), &(ownrates), sizeof(f64_vec3));

			p_NT_addition_tri[iMinor].NiTi += visc_htg;

			// We will have to round this up into the vertex heat afterwards.


#ifdef DEBUGNANS
			if (ownrates.x != ownrates.x)
				printf("iMinor %d NaN ownrates.x\n", iMinor);
			if (ownrates.y != ownrates.y)
				printf("iMinor %d NaN ownrates.y\n", iMinor);
			if (ownrates.z != ownrates.z)
				printf("iMinor %d NaN ownrates.z\n", iMinor);

			if (visc_htg != visc_htg) printf("iMinor %d NAN VISC HTG\n", iMinor);
#endif
		
			// We do best by taking each boundary, considering how
			// much heat to add for each one.

		} else {
			// Not domain tri or crossing_ins
			// Did we fairly model the insulator as a reflection of v?
		} 
	} // scope

	__syncthreads();
	
	// Now do electron: overwrite ita and nu, copy-paste the above codes very carefully
	shared_ita_par[threadIdx.x] = p_ita_parallel_elec_minor[iMinor];
	shared_nu[threadIdx.x] = p_nu_elec_minor[iMinor];

	if (threadIdx.x < threadsPerTileMajor) {
		info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];

		if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST))  // keeping consistent with ion above where we did put OUTERMOST here
		{// but we set ita to 0 in the pre routine for outermost.
			shared_ita_par_verts[threadIdx.x] = p_ita_parallel_elec_minor[iVertex + BEGINNING_OF_CENTRAL];
			shared_nu_verts[threadIdx.x] = p_nu_elec_minor[iVertex + BEGINNING_OF_CENTRAL];
		}
		else {
			shared_ita_par_verts[threadIdx.x] = 0.0;
			shared_nu_verts[threadIdx.x] = 0.0;
		};
	};

	__syncthreads();

	if (threadIdx.x < threadsPerTileMajor) {
		memset(&ownrates_visc, 0, sizeof(f64_vec3));
		visc_htg = 0.0;

		long izTri[MAXNEIGH];
		char szPBC[MAXNEIGH];
		short tri_len = info.neigh_len; // ?!
		memcpy(izTri, p_izTri + iVertex*MAXNEIGH, MAXNEIGH * sizeof(long));
		memcpy(szPBC, p_szPBC + iVertex*MAXNEIGH, MAXNEIGH * sizeof(char));

		our_v = shared_vie_verts[threadIdx.x]; // optimization: use replace or #define to get rid of storing this again.

		if ((info.flag == DOMAIN_VERTEX))
			//|| (info.flag == OUTERMOST)) 
		{

			short iprev = tri_len - 1;
			short i = 0;
			short inext;
			if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMinor))
			{
				prev_v = shared_vie[izTri[iprev] - StartMinor];
				prevpos = shared_pos[izTri[iprev] - StartMinor];
			}
			else {
				prev_v = p_vie_minor[izTri[iprev]];
				prevpos = p_info_minor[izTri[iprev]].pos;
			}
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
				prev_v.vxy = Clockwise_d*prev_v.vxy;
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
				prev_v.vxy = Anticlockwise_d*prev_v.vxy;
			}

			if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
			{
				opp_v = shared_vie[izTri[i] - StartMinor];
				opppos = shared_pos[izTri[i] - StartMinor];
			}
			else {
				opp_v = p_vie_minor[izTri[i]];
				opppos = p_info_minor[izTri[i]].pos;
			}
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
				opp_v.vxy = Clockwise_d*opp_v.vxy;
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
				opp_v.vxy = Anticlockwise_d*opp_v.vxy;
			}

		//	f64_vec2 endpt0 = THIRD * (info.pos + opppos + prevpos);

			f64_vec3 omega_ce;
#pragma unroll 
			for (i = 0; i < tri_len; i++)
			{
				// Tri 0 is anticlockwise of neighbour 0, we think
				inext = i + 1; if (inext >= tri_len) inext = 0;

				if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
				{
					next_v = shared_vie[izTri[inext] - StartMinor];
					nextpos = shared_pos[izTri[inext] - StartMinor];
				}
				else {
					next_v = p_vie_minor[izTri[inext]];
					nextpos = p_info_minor[izTri[inext]].pos;
				}
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
					next_v.vxy = Clockwise_d*next_v.vxy;
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
					next_v.vxy = Anticlockwise_d*next_v.vxy;
				}
				// All same as ion here:
				
				// Order of calculations may help things to go out/into scope at the right times so careful with that.
				f64_vec2 gradvx, gradvy, gradvez;

				f64 area_quadrilateral = 0.5*(
					(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
					+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
					+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
					+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
					);

				gradvx.x = 0.5*(
					(our_v.vxy.x + next_v.vxy.x)*(info.pos.y - nextpos.y)
					+ (prev_v.vxy.x + our_v.vxy.x)*(prevpos.y - info.pos.y)
					+ (opp_v.vxy.x + prev_v.vxy.x)*(opppos.y - prevpos.y)
					+ (next_v.vxy.x + opp_v.vxy.x)*(nextpos.y - opppos.y) // nextpos = pos_anti, assumed
					) / area_quadrilateral;
				gradvx.y = -0.5*(
					(our_v.vxy.x + next_v.vxy.x)*(info.pos.x - nextpos.x)
					+ (prev_v.vxy.x + our_v.vxy.x)*(prevpos.x - info.pos.x)
					+ (opp_v.vxy.x + prev_v.vxy.x)*(opppos.x - prevpos.x)
					+ (next_v.vxy.x + opp_v.vxy.x)*(nextpos.x - opppos.x) // nextpos = pos_anti, assumed
					) / area_quadrilateral;

				gradvy.x = 0.5*(
					(our_v.vxy.y + next_v.vxy.y)*(info.pos.y - nextpos.y)
					+ (prev_v.vxy.y + our_v.vxy.y)*(prevpos.y - info.pos.y)
					+ (opp_v.vxy.y + prev_v.vxy.y)*(opppos.y - prevpos.y)
					+ (next_v.vxy.y + opp_v.vxy.y)*(nextpos.y - opppos.y) // nextpos = pos_anti, assumed
					) / area_quadrilateral;
				gradvy.y = -0.5*(
					(our_v.vxy.y + next_v.vxy.y)*(info.pos.x - nextpos.x)
					+ (prev_v.vxy.y + our_v.vxy.y)*(prevpos.x - info.pos.x)
					+ (opp_v.vxy.y + prev_v.vxy.y)*(opppos.x - prevpos.x)
					+ (next_v.vxy.y + opp_v.vxy.y)*(nextpos.x - opppos.x) // nextpos = pos_anti, assumed
					) / area_quadrilateral;

				gradvez.x = 0.5*(
					(our_v.vez + next_v.vez)*(info.pos.y - nextpos.y)
					+ (prev_v.vez + our_v.vez)*(prevpos.y - info.pos.y)
					+ (opp_v.vez + prev_v.vez)*(opppos.y - prevpos.y)
					+ (next_v.vez + opp_v.vez)*(nextpos.y - opppos.y) // nextpos = pos_anti, assumed
					) / area_quadrilateral;
				gradvez.y = -0.5*(
					(our_v.vez + next_v.vez)*(info.pos.x - nextpos.x)
					+ (prev_v.vez + our_v.vez)*(prevpos.x - info.pos.x)
					+ (opp_v.vez + prev_v.vez)*(opppos.x - prevpos.x)
					+ (next_v.vez + opp_v.vez)*(nextpos.x - opppos.x) // nextpos = pos_anti, assumed
					) / area_quadrilateral;

				{
					f64_vec2 opp_B;
					f64 opp_ita, opp_nu;
					if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
					{
						opp_B = shared_B[izTri[i] - StartMinor];
						opp_ita = shared_ita_par[izTri[i] - StartMinor];
						opp_nu = shared_nu[izTri[i] - StartMinor];
						//ita_par = 0.5*(shared_ita_par_verts[threadIdx.x] + shared_ita_par[izTri[i] - StartMinor]);
						//nu = 0.5*(shared_nu_verts[threadIdx.x] + shared_nu[izTri[i] - StartMinor]);
					} else {
						opp_B = p_B_minor[izTri[i]].xypart();
						opp_ita = p_ita_parallel_elec_minor[izTri[i]];
						opp_nu = p_nu_elec_minor[izTri[i]];
					}
					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opp_B = Clockwise_d*opp_B;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opp_B = Anticlockwise_d*opp_B;
					}
					if (shared_ita_par_verts[threadIdx.x] < opp_ita) {
						ita_par = shared_ita_par_verts[threadIdx.x];
						nu = shared_nu_verts[threadIdx.x];
					} else {
						ita_par = opp_ita;
						nu = opp_nu;
					}
					omega_ce = 0.5*qovermc*(Make3(opp_B + shared_B_verts[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
				}

		//		f64_vec2 endpt1 = THIRD * (nextpos + info.pos + opppos);
				if ((VISCMAG == 0) || (omega_ce.dot(omega_ce) < 0.01*0.1*nu*nu))
				{
					// run unmagnetised case
					f64 Pi_xx, Pi_xy, Pi_yx, Pi_yy, Pi_zx, Pi_zy;

					// Let's suppose, Pi_yx means the rate of flow of y-momentum in the x direction.
					// Thus when we want to know how much y momentum is flowing through the wall we take
					// Pi_yx.edge_x + Pi_yy.edge_y -- reasonable.

					Pi_xx = -ita_par*THIRD*(4.0*gradvx.x - 2.0*gradvy.y);
					Pi_xy = -ita_par*(gradvx.y + gradvy.x);
					Pi_yx = Pi_xy;
					Pi_yy = -ita_par*THIRD*(4.0*gradvy.y - 2.0*gradvx.x);
					Pi_zx = -ita_par*(gradvez.x);
					Pi_zy = -ita_par*(gradvez.y);

					f64_vec2 edge_normal;
					edge_normal.x = THIRD * (nextpos.y - prevpos.y);
					edge_normal.y = THIRD * (prevpos.x - nextpos.x); // need to define so as to create unit vectors

					f64_vec3 visc_contrib;
					visc_contrib.x = -over_m_e*(Pi_xx*edge_normal.x + Pi_xy*edge_normal.y);
					visc_contrib.y = -over_m_e*(Pi_yx*edge_normal.x + Pi_yy*edge_normal.y);
					visc_contrib.z = -over_m_e*(Pi_zx*edge_normal.x + Pi_zy*edge_normal.y);

//					if (info.flag == OUTERMOST) {
//						if (p_info_minor[izTri[i]].flag == DOMAIN_TRIANGLE) {
//							ownrates_visc += visc_contrib;
//
//							visc_htg += -TWOTHIRDS*m_e*(
//								(our_v.vxy.x - opp_v.vxy.x)*visc_contrib.x
//								+ (our_v.vxy.y - opp_v.vxy.y)*visc_contrib.y
//								+ (our_v.vez - opp_v.vez)*visc_contrib.z);
//						}
//						else {
//							visc_contrib.x = 0.0; visc_contrib.y = 0.0; visc_contrib.z = 0.0;
//						}
//					} else
					{
						ownrates_visc += visc_contrib;

						visc_htg += -TWOTHIRDS*m_e*(
							(our_v.vxy.x - opp_v.vxy.x)*visc_contrib.x
							+ (our_v.vxy.y - opp_v.vxy.y)*visc_contrib.y
							+ (our_v.vez - opp_v.vez)*visc_contrib.z);
					};
					// The alternative, that may or may not run faster, is to test for ita == 0 before we do all the calcs
					// and then set ita == 0 in all the places not to look, including OUTERMOST, and do not do traffic to or from it.

					if (TEST) printf("iVertex %d tri %d ELEC ita_par %1.9E own ita %1.9E\n"
						"gradvx %1.8E %1.8E gradvy %1.8E %1.8E gradvez %1.8E %1.8E\n"						
						"edgenormal %1.8E %1.8E\n"
						"our_v %1.8E %1.8E %1.10E opp_v %1.8E %1.8E %1.10E\n"
						"Pi_xx %1.8E xy %1.8E yy %1.8E zx %1.8E\n"
						"visc_contrib %1.9E %1.9E %1.9E htg %1.9E\n "
						"heating contribs %1.9E %1.9E %1.9E\n"
						"===\n",
						iVertex, izTri[i], ita_par, shared_ita_par_verts[threadIdx.x],
						gradvx.x, gradvx.y, gradvy.x, gradvy.y, gradvez.x, gradvez.y,
						edge_normal.x, edge_normal.y,
						our_v.vxy.x,our_v.vxy.y,our_v.vez,opp_v.vxy.x,opp_v.vxy.y,opp_v.vez,
						Pi_xx, Pi_xy, Pi_yy, Pi_zx,
						visc_contrib.x, visc_contrib.y, visc_contrib.z, visc_htg,
						-TWOTHIRDS*m_e*((our_v.vxy.x - opp_v.vxy.x)*visc_contrib.x),
						-TWOTHIRDS*m_e*((our_v.vxy.y - opp_v.vxy.y)*visc_contrib.y),
						-TWOTHIRDS*m_e*((our_v.vez - opp_v.vez)*visc_contrib.z)
					);

					// -= !!!
					// So we are saying if edge_normal.x > 0 and gradviz.x > 0
					// then Pi_zx < 0 then ownrates += a positive amount. That is correct.
				}
				else {
					f64_vec3 unit_b, unit_perp, unit_Hall;
					f64 omegamod;
					{
						f64_vec2 edge_normal;
						edge_normal.x = THIRD * (nextpos.y - prevpos.y);
						edge_normal.y = THIRD * (prevpos.x - nextpos.x);  // need to define so as to create unit vectors

						f64 omegasq = omega_ce.dot(omega_ce);
						omegamod = sqrt(omegasq);
						unit_b = omega_ce / omegamod;
						unit_perp = Make3(edge_normal, 0.0) - unit_b*(unit_b.dotxy(edge_normal));
						unit_perp = unit_perp / unit_perp.modulus();
						unit_Hall = unit_b.cross(unit_perp); // Note sign.
															 // store omegamod instead.
					}

					f64 W_bb = 0.0, W_bP = 0.0, W_bH = 0.0, W_PP = 0.0, W_PH = 0.0, W_HH = 0.0; // these have to be alive at same time as 9 x partials
					{
						f64_vec3 intermed;

						// use: d vb / da = b transpose [ dvi/dxj ] a
						// Prototypical element: a.x b.y dvy/dx
						// b.x a.y dvx/dy

						intermed.x = unit_b.dotxy(gradvx);
						intermed.y = unit_b.dotxy(gradvy);
						intermed.z = unit_b.dotxy(gradvez);
						{
							f64 dvb_by_db, dvperp_by_db, dvHall_by_db;

							dvb_by_db = unit_b.dot(intermed);
							dvperp_by_db = unit_perp.dot(intermed);
							dvHall_by_db = unit_Hall.dot(intermed);

							W_bb += 4.0*THIRD*dvb_by_db;
							W_bP += dvperp_by_db;
							W_bH += dvHall_by_db;
							W_PP -= 2.0*THIRD*dvb_by_db;
							W_HH -= 2.0*THIRD*dvb_by_db;
						}
						{
							f64 dvb_by_dperp, dvperp_by_dperp,
								dvHall_by_dperp;
							// Optimize by getting rid of different labels.

							intermed.x = unit_perp.dotxy(gradvx);
							intermed.y = unit_perp.dotxy(gradvy);
							intermed.z = unit_perp.dotxy(gradvez);

							dvb_by_dperp = unit_b.dot(intermed);
							dvperp_by_dperp = unit_perp.dot(intermed);
							dvHall_by_dperp = unit_Hall.dot(intermed);

							W_bb -= 2.0*THIRD*dvperp_by_dperp;
							W_PP += 4.0*THIRD*dvperp_by_dperp;
							W_HH -= 2.0*THIRD*dvperp_by_dperp;
							W_bP += dvb_by_dperp;
							W_PH += dvHall_by_dperp;
						}
						{
							f64 dvb_by_dHall, dvperp_by_dHall, dvHall_by_dHall;

							intermed.x = unit_Hall.dotxy(gradvx);
							intermed.y = unit_Hall.dotxy(gradvy);
							intermed.z = unit_Hall.dotxy(gradvez);

							dvb_by_dHall = unit_b.dot(intermed);
							dvperp_by_dHall = unit_perp.dot(intermed);
							dvHall_by_dHall = unit_Hall.dot(intermed);

							W_bb -= 2.0*THIRD*dvHall_by_dHall;
							W_PP -= 2.0*THIRD*dvHall_by_dHall;
							W_HH += 4.0*THIRD*dvHall_by_dHall;
							W_bH += dvb_by_dHall;
							W_PH += dvperp_by_dHall;
						}
					}

					f64 Pi_b_b = 0.0, Pi_P_b = 0.0, Pi_P_P = 0.0, Pi_H_b = 0.0, Pi_H_P = 0.0, Pi_H_H = 0.0;
					{
						{
							f64 ita_1 = ita_par*(nu*nu / (nu*nu + omegamod*omegamod));

							Pi_b_b += -ita_par*W_bb;
							Pi_P_P += -0.5*(ita_par + ita_1)*W_PP - 0.5*(ita_par - ita_1)*W_HH;
							Pi_H_H += -0.5*(ita_par + ita_1)*W_HH - 0.5*(ita_par - ita_1)*W_PP;
							Pi_H_P += -ita_1*W_PH;
						}
						{
							f64 ita_2 = ita_par*(nu*nu / (nu*nu + 0.25*omegamod*omegamod));
							Pi_P_b += -ita_2*W_bP;
							Pi_H_b += -ita_2*W_bH;
						}
						{
							f64 ita_3 = ita_par*(nu*omegamod / (nu*nu + omegamod*omegamod));
							Pi_P_P -= ita_3*W_PH;
							Pi_H_H += ita_3*W_PH;
							Pi_H_P += 0.5*ita_3*(W_PP - W_HH);
						}
						{
							f64 ita_4 = 0.5*ita_par*(nu*omegamod / (nu*nu + 0.25*omegamod*omegamod));
							Pi_P_b += -ita_4*W_bH;
							Pi_H_b += ita_4*W_bP;
						}
					}

					f64 momflux_b, momflux_perp, momflux_Hall;
					{
						f64_vec3 mag_edge;
						f64_vec2 edge_normal;
						edge_normal.x = THIRD * (nextpos.y - prevpos.y);
						edge_normal.y = THIRD * (prevpos.x - nextpos.x);
						// Most efficient way: compute mom flux in magnetic coords
						mag_edge.x = unit_b.x*edge_normal.x + unit_b.y*edge_normal.y;    // b component
						mag_edge.y = unit_perp.x*edge_normal.x + unit_perp.y*edge_normal.y; // P component
						mag_edge.z = unit_Hall.x*edge_normal.x + unit_Hall.y*edge_normal.y; // H component

						// verify for chosen edge that we obtained a 3-vector of the same length as the original edge!
						// Tick

						momflux_b = -(Pi_b_b*mag_edge.x + Pi_P_b*mag_edge.y + Pi_H_b*mag_edge.z);
						momflux_perp = -(Pi_P_b*mag_edge.x + Pi_P_P*mag_edge.y + Pi_H_P*mag_edge.z);
						momflux_Hall = -(Pi_H_b*mag_edge.x + Pi_H_P*mag_edge.y + Pi_H_H*mag_edge.z);
					}
					
					f64_vec3 visc_contrib;
					visc_contrib.x = over_m_e*(unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall);
					visc_contrib.y = over_m_e*(unit_b.y*momflux_b + unit_perp.y*momflux_perp + unit_Hall.y*momflux_Hall);
					visc_contrib.z = over_m_e*(unit_b.z*momflux_b + unit_perp.z*momflux_perp + unit_Hall.z*momflux_Hall);

					//if (info.flag == OUTERMOST) {
					//	if (p_info_minor[izTri[i]].flag == DOMAIN_TRIANGLE) {
					//		ownrates_visc += visc_contrib;

					//		visc_htg += -TWOTHIRDS*m_e*(
					//			(our_v.vxy.x - opp_v.vxy.x)*visc_contrib.x
					//			+ (our_v.vxy.y - opp_v.vxy.y)*visc_contrib.y
					//			+ (our_v.vez - opp_v.vez)*visc_contrib.z);
					//	}
					//}else
					{
						ownrates_visc += visc_contrib;

						visc_htg += -TWOTHIRDS*m_e*(
							(our_v.vxy.x - opp_v.vxy.x)*visc_contrib.x
							+ (our_v.vxy.y - opp_v.vxy.y)*visc_contrib.y
							+ (our_v.vez - opp_v.vez)*visc_contrib.z);
					};
					//if (TEST) {
					//	f64_vec3 mag_edge;
					//	f64_vec2 edge_normal;
					//	edge_normal.x = endpt1.y - endpt0.y;
					//	edge_normal.y = endpt0.x - endpt1.x;
					//	// Most efficient way: compute mom flux in magnetic coords
					//	mag_edge.x = unit_b.x*edge_normal.x + unit_b.y*edge_normal.y;
					//	mag_edge.y = unit_perp.x*edge_normal.x + unit_perp.y*edge_normal.y;
					//	mag_edge.z = unit_Hall.x*edge_normal.x + unit_Hall.y*edge_normal.y;
					//	
					//	printf("iVertex %d MAGNETIZED elec: visc contrib %1.8E %1.8E %1.8E\n"
					//		"ita_par %1.9E ita_perp %1.9E ita_cross %1.9E\n"
					//		"momflux_b %1.8E momflux_perp %1.8E Hall %1.8E\n"
					//		"mag_edge %1.8E %1.8E %1.8E edge_normal %1.8E %1.8E \n"
					//		"unit_b %1.8E %1.8E %1.8E unit_perp %1.8E %1.8E %1.8E unit_H %1.8E %1.8E %1.8E\n"
					//		"omega_ce %1.8E %1.8E %1.8E mod %1.8E nu %1.8E \n"
					//		"===\n",
					//		iVertex, visc_contrib.x, visc_contrib.y, visc_contrib.z,
					//		ita_par, ita_perp, ita_cross,
					//		momflux_b, momflux_perp, momflux_Hall,
					//		mag_edge.x, mag_edge.y, mag_edge.z,
					//		edge_normal.x, edge_normal.y,
					//		unit_b.x, unit_b.y, unit_b.z, unit_perp.x, unit_perp.y, unit_perp.z, unit_Hall.x, unit_Hall.y, unit_Hall.z,
					//		omega_ce.x, omega_ce.y, omega_ce.z, omega_ce.modulus(), nu);
					//}
					//
					// MAR_elec -= Make3(0.5*(n0 * T0.Te + n1 * T1.Te)*over_m_e*edge_normal, 0.0);
					// v0.vez = vie_k.vez + h_use * MAR.z / (n_use.n*AreaMinor);
				}
		//		endpt0 = endpt1;
				prevpos = opppos;
				prev_v = opp_v;
				opppos = nextpos;
				opp_v = next_v;
			}; // next i

			f64_vec3 ownrates;
			memcpy(&ownrates, &(p_MAR_elec[iVertex + BEGINNING_OF_CENTRAL]), sizeof(f64_vec3));
//
//			if (TEST)
//				printf("iVertex %d ownrates %1.8E %1.8E %1.8E ownrates_visc %1.8E %1.8E %1.8E our_v %1.8E %1.8E %1.8E\n",
//				iVertex, ownrates.x, ownrates.y, ownrates.z, ownrates_visc.x, ownrates_visc.y, ownrates_visc.z, our_v.vxy.x, our_v.vxy.y, our_v.vez);
//
			ownrates += ownrates_visc;
			memcpy(p_MAR_elec + iVertex + BEGINNING_OF_CENTRAL, &ownrates, sizeof(f64_vec3));

			p_NT_addition_rate[iVertex].NeTe += visc_htg;

#ifdef DEBUGNANS
			if (ownrates.x != ownrates.x)
				printf("iVertex e %d NaN ownrates.x\n", iVertex);
			if (ownrates.y != ownrates.y)
				printf("iVertex e %d NaN ownrates.y\n", iVertex);
			if (ownrates.z != ownrates.z)
				printf("iVertex e %d NaN ownrates.z\n", iVertex);

			if (visc_htg != visc_htg) printf("iVertex e %d NAN VISC HTG\n", iVertex);
#endif
		} else {
			 // NOT domain vertex: Do nothing			
		};
	};

	// Electrons in tris:
	info = p_info_minor[iMinor];
	long izNeighMinor[6];
	char szPBC[6];
	memcpy(izNeighMinor, p_izNeighMinor + iMinor * 6, sizeof(long) * 6);
	memcpy(szPBC, p_szPBCtriminor + iMinor * 6, sizeof(char) * 6);
	memset(&ownrates_visc, 0, sizeof(f64_vec3));
	visc_htg = 0.0;

	our_v = shared_vie[threadIdx.x];

	if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL)) {
	}
	else {
		if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS)) {

			short inext, iprev = 5, i = 0;
			if ((izNeighMinor[iprev] >= StartMinor) && (izNeighMinor[iprev] < EndMinor))
			{
				memcpy(&prev_v, &(shared_vie[izNeighMinor[iprev] - StartMinor]), sizeof(v4));
				prevpos = shared_pos[izNeighMinor[iprev] - StartMinor];
			}
			else {
				if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					memcpy(&prev_v, &(shared_vie_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(v4));
					prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					prevpos = p_info_minor[izNeighMinor[iprev]].pos;
					memcpy(&prev_v, &(p_vie_minor[izNeighMinor[iprev]]), sizeof(v4));
				};
			};
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
				prev_v.vxy = Clockwise_d*prev_v.vxy;
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
				prev_v.vxy = Anticlockwise_d*prev_v.vxy;
			}

			i = 0;
			if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
			{
				memcpy(&opp_v, &(shared_vie[izNeighMinor[i] - StartMinor]), sizeof(v4));
				opppos = shared_pos[izNeighMinor[i] - StartMinor];
			}
			else {
				if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					memcpy(&opp_v, &(shared_vie_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(v4));
					opppos = shared_pos_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					opppos = p_info_minor[izNeighMinor[i]].pos;
					memcpy(&opp_v, &(p_vie_minor[izNeighMinor[i]]), sizeof(v4));
				};
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
				opp_v.vxy = Clockwise_d*opp_v.vxy;
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
				opp_v.vxy = Anticlockwise_d*opp_v.vxy;
			}

		//	f64_vec2 endpt0 = THIRD * (info.pos + opppos + prevpos);

			f64_vec3 omega_ce;
			// Let's make life easier and load up an array of 6 n's beforehand.
#pragma unroll 
			for (i = 0; i < 6; i++)
			{
				inext = i + 1; if (inext > 5) inext = 0;

				if ((izNeighMinor[inext] >= StartMinor) && (izNeighMinor[inext] < EndMinor))
				{
					memcpy(&next_v, &(shared_vie[izNeighMinor[inext] - StartMinor]), sizeof(v4));
					nextpos = shared_pos[izNeighMinor[inext] - StartMinor];
				}
				else {
					if ((izNeighMinor[inext] >= StartMajor + BEGINNING_OF_CENTRAL) &&
						(izNeighMinor[inext] < EndMajor + BEGINNING_OF_CENTRAL))
					{
						memcpy(&next_v, &(shared_vie_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(v4));
						nextpos = shared_pos_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
					}
					else {
						nextpos = p_info_minor[izNeighMinor[inext]].pos;
						memcpy(&next_v, &(p_vie_minor[izNeighMinor[inext]]), sizeof(v4));
					};
				};
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
					next_v.vxy = Clockwise_d*next_v.vxy;
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
					next_v.vxy = Anticlockwise_d*next_v.vxy;
				}

				bool bUsableSide = true;
				{
					f64_vec2 opp_B;
					f64 opp_ita, opp_nu;
					if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
					{
						opp_B = shared_B[izNeighMinor[i] - StartMinor];
						opp_ita = shared_ita_par[izNeighMinor[i] - StartMinor];
						opp_nu = shared_nu[izNeighMinor[i] - StartMinor];
						if (shared_ita_par[izNeighMinor[i] - StartMinor] == 0.0) bUsableSide = false;
					}
					else {
						if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
							(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
						{
							opp_B = shared_B_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
							opp_ita = shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
							opp_nu = shared_nu_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL];
							if (shared_ita_par_verts[izNeighMinor[i] - StartMajor - BEGINNING_OF_CENTRAL] == 0.0) bUsableSide = false;
						}
						else {
							opp_B = p_B_minor[izNeighMinor[i]].xypart();
							opp_ita = p_ita_parallel_elec_minor[izNeighMinor[i]];
							opp_nu = p_nu_elec_minor[izNeighMinor[i]];
							if (opp_ita == 0.0) bUsableSide = false;
						}
					}
					if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
						opp_B = Clockwise_d*opp_B;
					}
					if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
						opp_B = Anticlockwise_d*opp_B;
					}
					if (shared_ita_par[threadIdx.x] < opp_ita) {
						ita_par = shared_ita_par[threadIdx.x];
						nu = shared_nu[threadIdx.x];
					} else {
						ita_par = opp_ita;
						nu = opp_nu;
					}
					omega_ce = 0.5*qovermc*(Make3(opp_B + shared_B[threadIdx.x], BZ_CONSTANT)); // NOTE BENE qoverMc
				}

				f64_vec2 endpt1 = THIRD * (nextpos + info.pos + opppos);

				if (bUsableSide) {
					// New definition of endpoint of minor edge:
					f64_vec2 gradvez, gradvx, gradvy;

					f64 area_quadrilateral = 0.5*(
						(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
						+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
						+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
						+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
						);

					gradvx.x = 0.5*(
						(our_v.vxy.x + next_v.vxy.x)*(info.pos.y - nextpos.y)
						+ (prev_v.vxy.x + our_v.vxy.x)*(prevpos.y - info.pos.y)
						+ (opp_v.vxy.x + prev_v.vxy.x)*(opppos.y - prevpos.y)
						+ (next_v.vxy.x + opp_v.vxy.x)*(nextpos.y - opppos.y) // nextpos = pos_anti, assumed
						) / area_quadrilateral;
					gradvx.y = -0.5*(
						(our_v.vxy.x + next_v.vxy.x)*(info.pos.x - nextpos.x)
						+ (prev_v.vxy.x + our_v.vxy.x)*(prevpos.x - info.pos.x)
						+ (opp_v.vxy.x + prev_v.vxy.x)*(opppos.x - prevpos.x)
						+ (next_v.vxy.x + opp_v.vxy.x)*(nextpos.x - opppos.x) // nextpos = pos_anti, assumed
						) / area_quadrilateral;

					gradvy.x = 0.5*(
						(our_v.vxy.y + next_v.vxy.y)*(info.pos.y - nextpos.y)
						+ (prev_v.vxy.y + our_v.vxy.y)*(prevpos.y - info.pos.y)
						+ (opp_v.vxy.y + prev_v.vxy.y)*(opppos.y - prevpos.y)
						+ (next_v.vxy.y + opp_v.vxy.y)*(nextpos.y - opppos.y) // nextpos = pos_anti, assumed
						) / area_quadrilateral;
					gradvy.y = -0.5*(
						(our_v.vxy.y + next_v.vxy.y)*(info.pos.x - nextpos.x)
						+ (prev_v.vxy.y + our_v.vxy.y)*(prevpos.x - info.pos.x)
						+ (opp_v.vxy.y + prev_v.vxy.y)*(opppos.x - prevpos.x)
						+ (next_v.vxy.y + opp_v.vxy.y)*(nextpos.x - opppos.x) // nextpos = pos_anti, assumed
						) / area_quadrilateral;

					gradvez.x = 0.5*(
						(our_v.vez + next_v.vez)*(info.pos.y - nextpos.y)
						+ (prev_v.vez + our_v.vez)*(prevpos.y - info.pos.y)
						+ (opp_v.vez + prev_v.vez)*(opppos.y - prevpos.y)
						+ (next_v.vez + opp_v.vez)*(nextpos.y - opppos.y) // nextpos = pos_anti, assumed
						) / area_quadrilateral;
					gradvez.y = -0.5*(
						(our_v.vez + next_v.vez)*(info.pos.x - nextpos.x)
						+ (prev_v.vez + our_v.vez)*(prevpos.x - info.pos.x)
						+ (opp_v.vez + prev_v.vez)*(opppos.x - prevpos.x)
						+ (next_v.vez + opp_v.vez)*(nextpos.x - opppos.x) // nextpos = pos_anti, assumed
						) / area_quadrilateral;


					if ((VISCMAG == 0) || (omega_ce.dot(omega_ce) < 0.1*0.1*nu*nu))
					{
						// run unmagnetised case
						f64 Pi_xx, Pi_xy, Pi_yx, Pi_yy, Pi_zx, Pi_zy;

						Pi_xx = -ita_par*THIRD*(4.0*gradvx.x - 2.0*gradvy.y);
						Pi_xy = -ita_par*(gradvx.y + gradvy.x);
						Pi_yx = Pi_xy;
						Pi_yy = -ita_par*THIRD*(4.0*gradvy.y - 2.0*gradvx.x);
						Pi_zx = -ita_par*(gradvez.x);
						Pi_zy = -ita_par*(gradvez.y);

						f64_vec2 edge_normal;
						edge_normal.x = THIRD * (nextpos.y - prevpos.y);
						edge_normal.y = THIRD * (prevpos.x - nextpos.x);  // need to define so as to create unit vectors

						f64_vec3 visc_contrib;
						visc_contrib.x = -over_m_e*(Pi_xx*edge_normal.x + Pi_xy*edge_normal.y);
						visc_contrib.y = -over_m_e*(Pi_yx*edge_normal.x + Pi_yy*edge_normal.y);
						visc_contrib.z = -over_m_e*(Pi_zx*edge_normal.x + Pi_zy*edge_normal.y);

						if (info.flag == CROSSING_INS) {
							char flag = p_info_minor[izNeighMinor[i]].flag;
							if ((flag == DOMAIN_TRIANGLE) || (flag == DOMAIN_VERTEX)) {
								ownrates_visc += visc_contrib;
								if (i % 2 != 0) {
									visc_htg += -THIRD*m_e*(
										(our_v.vxy.x - opp_v.vxy.x)*visc_contrib.x
										+ (our_v.vxy.y - opp_v.vxy.y)*visc_contrib.y
										+ (our_v.vez - opp_v.vez)*visc_contrib.z);
								}
							}
							else {
								// DO NOTHING
								visc_contrib.x = 0.0; visc_contrib.y = 0.0; visc_contrib.z = 0.0;
							}
						}
						else {
							ownrates_visc += visc_contrib;
							if (i % 2 != 0)
								visc_htg += -THIRD*m_e*(
								(our_v.vxy.x - opp_v.vxy.x)*visc_contrib.x
									+ (our_v.vxy.y - opp_v.vxy.y)*visc_contrib.y
									+ (our_v.vez - opp_v.vez)*visc_contrib.z);
						}

//						if (iMinor == 42939)
//							printf("\n%d : %d : ita %1.8E our_vez %1.10E opp_vez %1.10E \n"
//								"gradvz %1.9E %1.9E ourpos %1.9E %1.9E opppos %1.9E %1.9E \n"
//								"visc_contrib.z %1.10E visc_htg %1.10E\n", iMinor, izNeighMinor[i],
//								ita_par, our_v.vez, opp_v.vez, 
//								gradvez.x,gradvez.y, info.pos.x,info.pos.y,opppos.x,opppos.y,
//								visc_contrib.z, visc_htg);

						// 42939: Find out why it makes too much heat. Probably a compound error.
				//		if (iMinor == 42939) printf("42939\nour_v %1.8E %1.8E %1.8E \n"
				//			"opp_v %1.8E %1.8E %1.8E \n"
				//			"visc_contrib %1.8E %1.8E %1.8E \n",
				//			our_v.vxy.x, our_v.vxy.y, our_v.vez,
				//			opp_v.vxy.x, opp_v.vxy.y, opp_v.vez,
				//			visc_contrib.x, visc_contrib.y, visc_contrib.z);
				//		
						
					} else {
						f64_vec3 unit_b, unit_perp, unit_Hall;
						f64 omegamod;
						{
							f64_vec2 edge_normal;
							edge_normal.x = THIRD * (nextpos.y - prevpos.y);
							edge_normal.y = THIRD * (prevpos.x - nextpos.x);  // need to define so as to create unit vectors

							f64 omegasq = omega_ce.dot(omega_ce);
							omegamod = sqrt(omegasq);
							unit_b = omega_ce / omegamod;
							unit_perp = Make3(edge_normal, 0.0) - unit_b*(unit_b.dotxy(edge_normal));
							unit_perp = unit_perp / unit_perp.modulus();
							unit_Hall = unit_b.cross(unit_perp); // Note sign.
																 // store omegamod instead.
						}

						f64 W_bb = 0.0, W_bP = 0.0, W_bH = 0.0, W_PP = 0.0, W_PH = 0.0, W_HH = 0.0; // these have to be alive at same time as 9 x partials
						{
							f64_vec3 intermed;

							// use: d vb / da = b transpose [ dvi/dxj ] a
							// Prototypical element: a.x b.y dvy/dx
							// b.x a.y dvx/dy

							intermed.x = unit_b.dotxy(gradvx);
							intermed.y = unit_b.dotxy(gradvy);
							intermed.z = unit_b.dotxy(gradvez);
							{
								f64 dvb_by_db, dvperp_by_db, dvHall_by_db;

								dvb_by_db = unit_b.dot(intermed);
								dvperp_by_db = unit_perp.dot(intermed);
								dvHall_by_db = unit_Hall.dot(intermed);

								W_bb += 4.0*THIRD*dvb_by_db;
								W_bP += dvperp_by_db;
								W_bH += dvHall_by_db;
								W_PP -= 2.0*THIRD*dvb_by_db;
								W_HH -= 2.0*THIRD*dvb_by_db;
							}
							{
								f64 dvb_by_dperp, dvperp_by_dperp,
									dvHall_by_dperp;
								// Optimize by getting rid of different labels.

								intermed.x = unit_perp.dotxy(gradvx);
								intermed.y = unit_perp.dotxy(gradvy);
								intermed.z = unit_perp.dotxy(gradvez);

								dvb_by_dperp = unit_b.dot(intermed);
								dvperp_by_dperp = unit_perp.dot(intermed);
								dvHall_by_dperp = unit_Hall.dot(intermed);

								W_bb -= 2.0*THIRD*dvperp_by_dperp;
								W_PP += 4.0*THIRD*dvperp_by_dperp;
								W_HH -= 2.0*THIRD*dvperp_by_dperp;
								W_bP += dvb_by_dperp;
								W_PH += dvHall_by_dperp;
							}
							{
								f64 dvb_by_dHall, dvperp_by_dHall, dvHall_by_dHall;

								intermed.x = unit_Hall.dotxy(gradvx);
								intermed.y = unit_Hall.dotxy(gradvy);
								intermed.z = unit_Hall.dotxy(gradvez);

								dvb_by_dHall = unit_b.dot(intermed);
								dvperp_by_dHall = unit_perp.dot(intermed);
								dvHall_by_dHall = unit_Hall.dot(intermed);

								W_bb -= 2.0*THIRD*dvHall_by_dHall;
								W_PP -= 2.0*THIRD*dvHall_by_dHall;
								W_HH += 4.0*THIRD*dvHall_by_dHall;
								W_bH += dvb_by_dHall;
								W_PH += dvperp_by_dHall;
							}
						}

						f64 Pi_b_b = 0.0, Pi_P_b = 0.0, Pi_P_P = 0.0, Pi_H_b = 0.0, Pi_H_P = 0.0, Pi_H_H = 0.0;
						{
							{
								f64 ita_1 = ita_par*(nu*nu / (nu*nu + omegamod*omegamod));

								Pi_b_b += -ita_par*W_bb;
								Pi_P_P += -0.5*(ita_par + ita_1)*W_PP - 0.5*(ita_par - ita_1)*W_HH;
								Pi_H_H += -0.5*(ita_par + ita_1)*W_HH - 0.5*(ita_par - ita_1)*W_PP;
								Pi_H_P += -ita_1*W_PH;
							}
							{
								f64 ita_2 = ita_par*(nu*nu / (nu*nu + 0.25*omegamod*omegamod));
								Pi_P_b += -ita_2*W_bP;
								Pi_H_b += -ita_2*W_bH;
							}
							{
								f64 ita_3 = ita_par*(nu*omegamod / (nu*nu + omegamod*omegamod));
								Pi_P_P -= ita_3*W_PH;
								Pi_H_H += ita_3*W_PH;
								Pi_H_P += 0.5*ita_3*(W_PP - W_HH);
							}
							{
								f64 ita_4 = 0.5*ita_par*(nu*omegamod / (nu*nu + 0.25*omegamod*omegamod));
								Pi_P_b += -ita_4*W_bH;
								Pi_H_b += ita_4*W_bP;
							}
						}
						
						f64 momflux_b, momflux_perp, momflux_Hall;
						{
							f64_vec3 mag_edge;
							f64_vec2 edge_normal;
							edge_normal.x = THIRD * (nextpos.y - prevpos.y);
							edge_normal.y = THIRD * (prevpos.x - nextpos.x);  // need to define so as to create unit vectors
							// Most efficient way: compute mom flux in magnetic coords
							mag_edge.x = unit_b.x*edge_normal.x + unit_b.y*edge_normal.y;
							mag_edge.y = unit_perp.x*edge_normal.x + unit_perp.y*edge_normal.y;
							mag_edge.z = unit_Hall.x*edge_normal.x + unit_Hall.y*edge_normal.y;

							momflux_b = -(Pi_b_b*mag_edge.x + Pi_P_b*mag_edge.y + Pi_H_b*mag_edge.z);
							momflux_perp = -(Pi_P_b*mag_edge.x + Pi_P_P*mag_edge.y + Pi_H_P*mag_edge.z);
							momflux_Hall = -(Pi_H_b*mag_edge.x + Pi_H_P*mag_edge.y + Pi_H_H*mag_edge.z);
						}

						// unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall
						// is the flow of p_x dotted with the edge_normal
						// ownrates will be divided by N to give dv/dt
						// m N dvx/dt = integral div momflux_x
						// Therefore divide here just by m
						f64_vec3 visc_contrib;
						visc_contrib.x = over_m_e*(unit_b.x*momflux_b + unit_perp.x*momflux_perp + unit_Hall.x*momflux_Hall);
						visc_contrib.y = over_m_e*(unit_b.y*momflux_b + unit_perp.y*momflux_perp + unit_Hall.y*momflux_Hall);
						visc_contrib.z = over_m_e*(unit_b.z*momflux_b + unit_perp.z*momflux_perp + unit_Hall.z*momflux_Hall);

						if (info.flag == CROSSING_INS) {
							char flag = p_info_minor[izNeighMinor[i]].flag;
							if ((flag == DOMAIN_TRIANGLE) || (flag == DOMAIN_VERTEX)) {
								ownrates_visc += visc_contrib;
								if (i % 2 != 0) visc_htg += -THIRD*m_e*(
									(our_v.vxy.x - opp_v.vxy.x)*visc_contrib.x
									+ (our_v.vxy.y - opp_v.vxy.y)*visc_contrib.y
									+ (our_v.vez - opp_v.vez)*visc_contrib.z);
							}
							else {
								// DO NOTHING
							}
						}
						else {
							ownrates_visc += visc_contrib;
							if (i % 2 != 0)
								visc_htg += -THIRD*m_e*(
								(our_v.vxy.x - opp_v.vxy.x)*visc_contrib.x
									+ (our_v.vxy.y - opp_v.vxy.y)*visc_contrib.y
									+ (our_v.vez - opp_v.vez)*visc_contrib.z);
						}
					}
				}; // bUsableSide

		//		endpt0 = endpt1;
				prevpos = opppos;
				prev_v = opp_v;
				opppos = nextpos;
				opp_v = next_v;
			};
			f64_vec3 ownrates;
			memcpy(&(ownrates), &(p_MAR_elec[iMinor]), sizeof(f64_vec3));
			ownrates += ownrates_visc;
			memcpy(&(p_MAR_elec[iMinor]), &(ownrates), sizeof(f64_vec3));

			p_NT_addition_tri[iMinor].NeTe += visc_htg;

#ifdef DEBUGNANS
			if (ownrates.x != ownrates.x)
				printf("iMinor e %d NaN ownrates.x\n", iMinor);
			if (ownrates.y != ownrates.y)
				printf("iMinor e %d NaN ownrates.y\n", iMinor);
			if (ownrates.z != ownrates.z)
				printf("iMinor e %d NaN ownrates.z\n", iMinor);

			if (visc_htg != visc_htg) printf("iMinor e %d NAN VISC HTG\n", iMinor);
#endif
		} else {
			// Not domain, not crossing_ins, not a frill			
		} // non-domain tri
	}; // was it FRILL
}

// Neutral routine:

// Sort everything else out first and then this. Just a copy of the above routine
// but with 3-vector v_n + Tn in place of 4-vector vie, doing pressure and momflux.

__global__ void kernelNeutral_pressure_and_momflux(
	structural * __restrict__ p_info_minor,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighTriMinor,
	char * __restrict__ p_szPBCtriminor,
	LONG3 * __restrict__ p_who_am_I_to_corners,
	LONG3 * __restrict__ p_tricornerindex,

	T3 * __restrict__ p_T_minor,
	f64_vec3 * __restrict__ p_v_n_minor,
	ShardModel * __restrict__ p_n_shards,
	nvals * __restrict__ p_n_minor, // Just to handle insulator

	f64_vec2 * __restrict__ p_v_overall_minor,
	f64_vec3 * __restrict__ p_MAR_neut
)
{
	__shared__ f64_vec3 shared_v_n[threadsPerTileMinor];
	__shared__ f64_vec2 shared_v_overall[threadsPerTileMinor];
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];
	__shared__ f64 shared_Tn[threadsPerTileMinor]; // 3+2+2+1=8 per thread

	__shared__ ShardModel shared_n_shards[threadsPerTileMajor];

	__shared__ f64_vec3 shared_v_n_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_v_overall_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajor];
	__shared__ f64 shared_Tn_verts[threadsPerTileMajor];  // 1/2( 13+3+2+2+1 = 21) = 10.5 => total 18.5 per minor thread.
	// shame we couldn't get down to 16 per minor thread, and if we could then that might be better even if we load on-the-fly something.

	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	long const iVertex = threadsPerTileMajor*blockIdx.x + threadIdx.x; // only meaningful threadIdx.x < threadsPerTileMajor
	long const StartMinor = threadsPerTileMinor*blockIdx.x;
	long const StartMajor = threadsPerTileMajor*blockIdx.x;
	long const EndMinor = StartMinor + threadsPerTileMinor;
	long const EndMajor = StartMajor + threadsPerTileMajor;

	shared_pos[threadIdx.x] = p_info_minor[iMinor].pos; // QUESTION: DOES THIS LOAD CONTIGUOUSLY?
	shared_v_n[threadIdx.x] = p_v_n_minor[iMinor];
	shared_v_overall[threadIdx.x] = p_v_overall_minor[iMinor];
	shared_Tn[threadIdx.x] = p_T_minor[iMinor].Tn;		// QUESTION: DOES THIS LOAD CONTIGUOUSLY?

	// Advection should be an outer cycle at 1e-10 s.

	structural info;
	if (threadIdx.x < threadsPerTileMajor) {
		info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
		shared_pos_verts[threadIdx.x] = info.pos;
		if (info.flag == DOMAIN_VERTEX) {
			memcpy(&(shared_n_shards[threadIdx.x]), &(p_n_shards[iVertex]), sizeof(ShardModel)); // + 13
			memcpy(&(shared_v_n_verts[threadIdx.x]), &(p_v_n_minor[iVertex + BEGINNING_OF_CENTRAL]), sizeof(f64_vec3));
			shared_v_overall_verts[threadIdx.x] = p_v_overall_minor[iVertex + BEGINNING_OF_CENTRAL];
			shared_Tn_verts[threadIdx.x] = p_T_minor[iVertex + BEGINNING_OF_CENTRAL].Tn;
		}
		else {
			// it still manages to coalesce, let's hope, because ShardModel is 13 doubles not 12.
			memset(&(shared_n_shards[threadIdx.x]), 0, sizeof(ShardModel)); // + 13
			memset(&(shared_v_n_verts[threadIdx.x]), 0, sizeof(f64_vec3));
			memset(&(shared_v_overall_verts[threadIdx.x]), 0, sizeof(f64_vec2));
			shared_Tn_verts[threadIdx.x] = 0.0;
		};
	};

	__syncthreads();

	f64_vec3 our_v, opp_v, prev_v, next_v;
	f64 oppT, prevT, nextT, ourT;
	f64_vec2 our_v_overall, prev_v_overall, next_v_overall, opp_v_overall;
	f64_vec2 opppos, prevpos, nextpos;
	f64 AreaMinor;

	if (threadIdx.x < threadsPerTileMajor) {
		AreaMinor = 0.0;
		f64_vec3 MAR_neut;
		memcpy(&(MAR_neut), &(p_MAR_neut[iVertex + BEGINNING_OF_CENTRAL]), sizeof(f64_vec3));
		
		long izTri[MAXNEIGH];
		char szPBC[MAXNEIGH];
		short tri_len = info.neigh_len;

		memcpy(izTri, p_izTri + iVertex*MAXNEIGH, MAXNEIGH * sizeof(long));
		memcpy(szPBC, p_szPBC + iVertex*MAXNEIGH, MAXNEIGH * sizeof(char));

		our_v = shared_v_n_verts[threadIdx.x];
		our_v_overall = shared_v_overall_verts[threadIdx.x];
		ourT = shared_Tn_verts[threadIdx.x];

		if (info.flag == DOMAIN_VERTEX) {

			short iprev = tri_len - 1;
			short i = 0;
			short inext;
			if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMinor))
			{
				prevT = shared_Tn[izTri[iprev] - StartMinor];
				prev_v = shared_v_n[izTri[iprev] - StartMinor];
				prev_v_overall = shared_v_overall[izTri[iprev] - StartMinor];
				prevpos = shared_pos[izTri[iprev] - StartMinor];
			} else {
				T3 prev_T = p_T_minor[izTri[iprev]];
				prevT = prev_T.Tn;
				prev_v = p_v_n_minor[izTri[iprev]];
				prev_v_overall = p_v_overall_minor[izTri[iprev]];
				prevpos = p_info_minor[izTri[iprev]].pos;
			}
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
				prev_v = Clockwise_rotate3(prev_v);
				prev_v_overall = Clockwise_d*prev_v_overall;
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
				prev_v = Anticlock_rotate3(prev_v);
				prev_v_overall = Anticlockwise_d*prev_v_overall;
			}
			
			if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
			{
				oppT = shared_Tn[izTri[i] - StartMinor];
				opp_v = shared_v_n[izTri[i] - StartMinor];
				opp_v_overall = shared_v_overall[izTri[i] - StartMinor];
				opppos = shared_pos[izTri[i] - StartMinor];
			} else {
				T3 opp_T = p_T_minor[izTri[i]];
				oppT = opp_T.Tn;
				opp_v = p_v_n_minor[izTri[i]];
				opp_v_overall = p_v_overall_minor[izTri[i]];
				opppos = p_info_minor[izTri[i]].pos;
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
				opp_v = Clockwise_rotate3(opp_v);
				opp_v_overall = Clockwise_d*opp_v_overall;
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
				opp_v = Anticlock_rotate3(opp_v);
				opp_v_overall = Anticlockwise_d*opp_v_overall;
			}

			// Think carefully: DOMAIN vertex cases for n,T ...
			f64 n0 = THIRD*(shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[iprev] + shared_n_shards[threadIdx.x].n_cent);
			f64_vec2 endpt0 = THIRD * (info.pos + opppos + prevpos);

			f64_vec2 endpt1, edge_normal;

			short iend = tri_len;
			f64_vec2 projendpt0;
			if ((info.flag == INNERMOST) || (info.flag == OUTERMOST)) {
				iend = tri_len - 2;
				if (info.flag == OUTERMOST) {
					endpt0.project_to_radius(projendpt0, FRILL_CENTROID_OUTER_RADIUS_d); // back of cell for Lap purposes
				} else {
					endpt0.project_to_radius(projendpt0, FRILL_CENTROID_INNER_RADIUS_d); // back of cell for Lap purposes
				}
				edge_normal.x = endpt0.y - projendpt0.y;
				edge_normal.y = projendpt0.x - endpt0.x;
				AreaMinor += (0.5*projendpt0.x + 0.5*endpt0.x)*edge_normal.x;
			};

			for (i = 0; i < iend; i++)
			{
				// Tri 0 is anticlockwise of neighbour 0, we think
				inext = i + 1; if (inext >= tri_len) inext = 0;

				if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
				{
					nextT = shared_Tn[izTri[inext] - StartMinor];
					next_v = shared_v_n[izTri[inext] - StartMinor];
					next_v_overall = shared_v_overall[izTri[inext] - StartMinor];
					nextpos = shared_pos[izTri[inext] - StartMinor];
				}
				else {
					T3 next_T = p_T_minor[izTri[inext]];
					nextT = next_T.Tn;
					next_v = p_v_n_minor[izTri[inext]];
					next_v_overall = p_v_overall_minor[izTri[inext]];
					nextpos = p_info_minor[izTri[inext]].pos;
				}
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
					next_v = Clockwise_rotate3(next_v);
					next_v_overall = Clockwise_d*next_v_overall;
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
					next_v = Anticlock_rotate3(next_v);
					next_v_overall = Anticlockwise_d*next_v_overall;
				}

				f64_vec2 endpt1 = THIRD * (nextpos + info.pos + opppos);
				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

				// ______________________________________________________-

				f64 n1;
				n1 = THIRD*(shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[inext] + shared_n_shards[threadIdx.x].n_cent);
			
				f64 T0, T1;
				T0 = THIRD*(prevT + ourT + oppT);
				T1 = THIRD*(nextT + ourT + oppT);
				f64_vec3 v0 = THIRD*(our_v + prev_v + opp_v);
				f64_vec3 v1 = THIRD*(our_v + opp_v + next_v);

				f64 relvnormal = 0.5*((v0 + v1).xypart()
					- (THIRD * (our_v_overall + next_v_overall + opp_v_overall))
					- (THIRD * (our_v_overall + prev_v_overall + opp_v_overall))
					).dot(edge_normal);
				// CHANGES 20th August 2019
				// OLD, unstable:
				// MAR_neut -= 0.5*relvnormal* (n0 *(v0-our_v) + n1 * (v1 - our_v));

				if (relvnormal < 0.0)
					MAR_neut -= 0.5*relvnormal* (n0 + n1) *(opp_v - our_v);
				// Note: minus a minus so correct sign

				// And we did what? We took n at centre of a triangle WITHIN this major cell 
				// But did not take upwind n ---- is that consistent for all advection?

				MAR_neut -= Make3(0.5*(n0*T0 + n1*T1)*over_m_n*edge_normal, 0.0);

				// ______________________________________________________
				//// whether the v that is leaving is greater than our v ..
				//// Formula:
				//// dv/dt = (d(Nv)/dt - dN/dt v) / N
				//// We include the divide by N when we enter the accel routine.

				AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;
				endpt0 = endpt1;
				n0 = n1;
				prevpos = opppos;
				prevT = oppT;
				prev_v = opp_v;
				prev_v_overall = opp_v_overall;

				opppos = nextpos;
				opp_v = next_v;
				oppT = nextT;
				opp_v_overall = next_v_overall;
			}; // next i

			memcpy(p_MAR_neut + iVertex + BEGINNING_OF_CENTRAL, &(MAR_neut), sizeof(f64_vec3));
		} else {
			// NOT domain vertex: Do nothing
		};
	}; // was it domain vertex or Az-only
	   // This branching is itself a good argument for doing Az in ITS own separate routine with no need for n_shard.

	   // __syncthreads(); // end of first vertex part
	   // Do we need syncthreads? Not overwriting any shared data here...

	   // now the minor with n_ion part:
	info = p_info_minor[iMinor];
	our_v = shared_v_n[threadIdx.x];
	ourT = shared_Tn[threadIdx.x];
	our_v_overall = shared_v_overall[threadIdx.x];

	long izNeighMinor[6];
	char szPBC[6];
	memcpy(izNeighMinor, p_izNeighTriMinor + iMinor * 6, sizeof(long) * 6);
	memcpy(szPBC, p_szPBCtriminor + iMinor * 6, sizeof(char) * 6);

	f64_vec3 MAR_neut;
	memcpy(&(MAR_neut), &(p_MAR_neut[iMinor]), sizeof(f64_vec3));
	
	if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL)) {
		// Do nothing? Who cares what it is.
	} else {
		AreaMinor = 0.0;
		if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS)) {

			short inext, iprev = 5, i = 0;
			if ((izNeighMinor[iprev] >= StartMinor) && (izNeighMinor[iprev] < EndMinor))
			{
				memcpy(&prev_v, &(shared_v_n[izNeighMinor[iprev] - StartMinor]), sizeof(f64_vec3));
				prevT = shared_Tn[izNeighMinor[iprev] - StartMinor];
				prevpos = shared_pos[izNeighMinor[iprev] - StartMinor];
				prev_v_overall = shared_v_overall[izNeighMinor[iprev] - StartMinor];
			} else {
				if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					memcpy(&prev_v, &(shared_v_n_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(f64_vec3));
					prev_v_overall = shared_v_overall_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
					prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
					prevT = shared_Tn_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor]; 
				} else {
					prevpos = p_info_minor[izNeighMinor[iprev]].pos;
					memcpy(&prev_v, &(p_v_n_minor[izNeighMinor[iprev]]), sizeof(f64_vec3));
					prev_v_overall = p_v_overall_minor[izNeighMinor[iprev]];
					prevT = p_T_minor[izNeighMinor[iprev]].Tn;
				};
			};
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
				prev_v = Clockwise_rotate3(prev_v);
				prev_v_overall = Clockwise_d*prev_v_overall;
			};
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
				prev_v = Anticlock_rotate3(prev_v);
				prev_v_overall = Anticlockwise_d*prev_v_overall;
			};
			
			i = 0;
			if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
			{
				memcpy(&opp_v, &(shared_v_n[izNeighMinor[i] - StartMinor]), sizeof(f64_vec3));
				opppos = shared_pos[izNeighMinor[i] - StartMinor];
				opp_v_overall = shared_v_overall[izNeighMinor[i] - StartMinor];
				oppT = shared_Tn[izNeighMinor[i] - StartMinor];
			} else {
				if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					memcpy(&opp_v, &(shared_v_n_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(f64_vec3));
					opp_v_overall = shared_v_overall_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
					opppos = shared_pos_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
					oppT = shared_Tn_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					opppos = p_info_minor[izNeighMinor[i]].pos;
					memcpy(&opp_v, &(p_v_n_minor[izNeighMinor[i]]), sizeof(f64_vec3));
					opp_v_overall = p_v_overall_minor[izNeighMinor[i]];
					T3 opp_T = p_T_minor[izNeighMinor[i]];
					oppT = opp_T.Tn;
				};
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
				opp_v = Clockwise_rotate3(opp_v);
				opp_v_overall = Clockwise_d*opp_v_overall;
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
				opp_v = Anticlock_rotate3(opp_v);
				opp_v_overall = Anticlockwise_d*opp_v_overall;
			}

			long who_am_I_to_corners[3];
			memcpy(who_am_I_to_corners, &(p_who_am_I_to_corners[iMinor]), sizeof(long) * 3);
			LONG3 cornerindex = p_tricornerindex[iMinor];
			// each corner we want to pick up 3 values off n_shards, as well as n_cent.
			// The three values will not always be contiguous!!!

			// Let's make life easier and load up an array of 6 n's beforehand.
			f64 n_array[6];
			f64 n0, n1;

			short who_am_I = who_am_I_to_corners[0];
			short tri_len = p_info_minor[cornerindex.i1 + BEGINNING_OF_CENTRAL].neigh_len;

			if ((cornerindex.i1 >= StartMajor) && (cornerindex.i1 < EndMajor))
			{
				short who_prev = who_am_I - 1;
				if (who_prev < 0) who_prev = tri_len - 1;
				// Worry about pathological cases later.
				short who_next = who_am_I + 1;
				if (who_next == tri_len) who_next = 0;
				n_array[0] = THIRD*(shared_n_shards[cornerindex.i1 - StartMajor].n[who_next]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n_cent);
				n_array[1] = THIRD*(shared_n_shards[cornerindex.i1 - StartMajor].n[who_prev]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n_cent);

			} else {
				// comes from elsewhere
				f64 ncent = p_n_shards[cornerindex.i1].n_cent;
				short who_prev = who_am_I - 1;
				if (who_prev < 0) {
					who_prev = tri_len - 1;
					f64_vec2 temp;
					memcpy(&temp, p_n_shards[cornerindex.i1].n, sizeof(f64_vec2));
					n_array[0] = THIRD*(temp.x + temp.y + ncent);
					n_array[1] = THIRD*(p_n_shards[cornerindex.i1].n[who_prev] + temp.x + ncent);
				}
				else {
					short who_next = who_am_I + 1;
					if (who_next == tri_len) {
						f64_vec2 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i1].n[who_prev]), sizeof(f64_vec2));
						n_array[0] = THIRD*(p_n_shards[cornerindex.i1].n[0] + temp.y + ncent);
						n_array[1] = THIRD*(temp.x + temp.y + ncent);
					}
					else {
						// typical case
						f64_vec3 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i1].n[who_prev]), sizeof(f64) * 3);
						n_array[0] = THIRD*(temp.z + temp.y + ncent);
						n_array[1] = THIRD*(temp.x + temp.y + ncent);
					};
				};
			}

			who_am_I = who_am_I_to_corners[1];
			tri_len = p_info_minor[cornerindex.i2 + BEGINNING_OF_CENTRAL].neigh_len;

			if ((cornerindex.i2 >= StartMajor) && (cornerindex.i2 < EndMajor))
			{
				short who_prev = who_am_I - 1;
				if (who_prev < 0) who_prev = tri_len - 1;
				// Worry about pathological cases later.
				short who_next = who_am_I + 1;
				if (who_next == tri_len) who_next = 0;
				n_array[2] = THIRD*(shared_n_shards[cornerindex.i2 - StartMajor].n[who_next]
					+ shared_n_shards[cornerindex.i2 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i2 - StartMajor].n_cent);
				n_array[3] = THIRD*(shared_n_shards[cornerindex.i2 - StartMajor].n[who_prev]
					+ shared_n_shards[cornerindex.i2 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i2 - StartMajor].n_cent);
			}
			else {
				// comes from elsewhere
				f64 ncent = p_n_shards[cornerindex.i2].n_cent;
				short who_prev = who_am_I - 1;
				if (who_prev < 0) {
					who_prev = tri_len - 1;
					f64_vec2 temp;
					memcpy(&temp, p_n_shards[cornerindex.i2].n, sizeof(f64_vec2));
					n_array[2] = THIRD*(temp.x + temp.y + ncent);
					n_array[3] = THIRD*(p_n_shards[cornerindex.i2].n[who_prev] + temp.x + ncent);
				}
				else {
					short who_next = who_am_I + 1;
					if (who_next == tri_len) {
						f64_vec2 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i2].n[who_prev]), sizeof(f64_vec2));
						n_array[2] = THIRD*(p_n_shards[cornerindex.i2].n[0] + temp.y + ncent);
						n_array[3] = THIRD*(temp.x + temp.y + ncent);
					}
					else {
						// typical case
						f64_vec3 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i2].n[who_prev]), sizeof(f64) * 3);
						n_array[2] = THIRD*(temp.z + temp.y + ncent); 
						n_array[3] = THIRD*(temp.x + temp.y + ncent);
					};
				};
			}

			who_am_I = who_am_I_to_corners[2];
			tri_len = p_info_minor[cornerindex.i3 + BEGINNING_OF_CENTRAL].neigh_len;

			if ((cornerindex.i3 >= StartMajor) && (cornerindex.i3 < EndMajor))
			{
				short who_prev = who_am_I - 1;
				if (who_prev < 0) who_prev = tri_len - 1;
				// Worry about pathological cases later.
				short who_next = who_am_I + 1;
				if (who_next == tri_len) who_next = 0;
				n_array[4] = THIRD*(shared_n_shards[cornerindex.i3 - StartMajor].n[who_next]
					+ shared_n_shards[cornerindex.i3 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i3 - StartMajor].n_cent);
				n_array[5] = THIRD*(shared_n_shards[cornerindex.i3 - StartMajor].n[who_prev]
					+ shared_n_shards[cornerindex.i3 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i3 - StartMajor].n_cent);
			} else {
				// comes from elsewhere
				f64 ncent = p_n_shards[cornerindex.i3].n_cent;
				short who_prev = who_am_I - 1;
				if (who_prev < 0) {
					who_prev = tri_len - 1;
					f64_vec2 temp;
					memcpy(&temp, p_n_shards[cornerindex.i3].n, sizeof(f64_vec2));
					n_array[4] = THIRD*(temp.x + temp.y + ncent);
					n_array[5] = THIRD*(p_n_shards[cornerindex.i3].n[who_prev] + temp.x + ncent);
				} else {
					short who_next = who_am_I + 1;
					if (who_next == tri_len) {
						f64_vec2 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i3].n[who_prev]), sizeof(f64_vec2));
						n_array[4] = THIRD*(p_n_shards[cornerindex.i3].n[0] + temp.y + ncent);
						n_array[5] = THIRD*(temp.x + temp.y + ncent);
					} else {
						// typical case
						f64_vec3 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i3].n[who_prev]), sizeof(f64) * 3);
						n_array[4] = THIRD*(temp.z + temp.y + ncent);
						n_array[5] = THIRD*(temp.x + temp.y + ncent);
					};
				};
			}

#pragma unroll 
			for (i = 0; i < 6; i++)
			{
				inext = i + 1; if (inext > 5) inext = 0;
				if ((izNeighMinor[inext] >= StartMinor) && (izNeighMinor[inext] < EndMinor))
				{
					memcpy(&next_v, &(shared_v_n[izNeighMinor[inext] - StartMinor]), sizeof(f64_vec3));
					nextpos = shared_pos[izNeighMinor[inext] - StartMinor];
					next_v_overall = shared_v_overall[izNeighMinor[inext] - StartMinor];
					nextT = shared_Tn[izNeighMinor[inext] - StartMinor];
				}
				else {
					if ((izNeighMinor[inext] >= StartMajor + BEGINNING_OF_CENTRAL) &&
						(izNeighMinor[inext] < EndMajor + BEGINNING_OF_CENTRAL))
					{
						memcpy(&next_v, &(shared_v_n_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(f64_vec3));
						next_v_overall = shared_v_overall_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
						nextpos = shared_pos_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
						nextT = shared_Tn_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
					}
					else {
						nextpos = p_info_minor[izNeighMinor[inext]].pos;
						memcpy(&next_v, &(p_v_n_minor[izNeighMinor[inext]]), sizeof(f64_vec3));
						next_v_overall = p_v_overall_minor[izNeighMinor[inext]];
						nextT = p_T_minor[izNeighMinor[inext]].Tn;						
					};
				};
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
					next_v = Clockwise_rotate3(next_v);
					next_v_overall = Clockwise_d*next_v_overall;
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
					next_v = Anticlock_rotate3(next_v);
					next_v_overall = Anticlockwise_d*next_v_overall;
				}

				// New definition of endpoint of minor edge:
				f64_vec2 endpt0, endpt1, edge_normal;

				endpt0 = THIRD * (prevpos + info.pos + opppos);
				endpt1 = THIRD * (nextpos + info.pos + opppos);

				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

				n0 = n_array[i];
				n1 = n_array[inext]; // 0,1 are either side of corner 0. What is seq of MinorNeigh ?

				// Assume neighs 0,1 are relevant to border with tri 0 minor.

				f64_vec3 v0 = THIRD*(our_v + prev_v + opp_v);
				f64_vec3 v1 = THIRD*(our_v + next_v + opp_v);

				//if (((izNeighMinor[i] >= NumInnerFrills_d) && (izNeighMinor[i] < FirstOuterFrill_d)))
				{	// Decided not to add test
					f64 relvnormal = 0.5*((v0 + v1).xypart()
						- (THIRD * (our_v_overall + next_v_overall + opp_v_overall))
						- (THIRD * (our_v_overall + prev_v_overall + opp_v_overall))
						).dot(edge_normal);

					// CHANGES 20th August 2019:
					// OLD, unstable:
					// MAR_neut -= 0.5*relvnormal* (n0 *(v0-our_v) + n1 * (v1 - our_v));
					if (relvnormal < 0.0)
						MAR_neut -= 0.5*relvnormal* (n0 + n1) *(opp_v - our_v);

					f64 T0 = THIRD*(ourT + prevT + oppT);
					f64 T1 = THIRD*(ourT + nextT + oppT);
					
					if (info.flag == CROSSING_INS) {
						char flag = p_info_minor[izNeighMinor[i]].flag;
						if ((flag == DOMAIN_TRIANGLE) || (flag == DOMAIN_VERTEX))
						{
							// do nothing
						}
						else {
							// Looking into the insulator we see a reflection of nT. Here we look into an out-of-domain tri or vert below ins.
							// Or allowed a below-ins value to affect something anyway.
							// Just for sanity for now, let's just set our own n,T for the edge:
							n0 = p_n_minor[iMinor].n_n;
							n1 = p_n_minor[iMinor].n_n;
							T0 = ourT;
							T1 = ourT;
						}
					}

					MAR_neut -= Make3(0.5*(n0*T0 + n1*T1)*over_m_n*edge_normal, 0.0);
				}

				endpt0 = endpt1;			
				prevT = oppT;
				prevpos = opppos;
				prev_v = opp_v;
				prev_v_overall = opp_v_overall;
				oppT = nextT;
				opppos = nextpos;
				opp_v = next_v;
				opp_v_overall = next_v_overall;
			};

			if (info.flag == CROSSING_INS) {
				// In this case set v_r = 0 and set a_TP_r = 0 and dv/dt _r = 0 in general
				//f64_vec2 rhat = info.pos / info.pos.modulus();
				MAR_neut -= Make3(
					(MAR_neut.dotxy(info.pos) /
					(info.pos.x*info.pos.x + info.pos.y*info.pos.y))*info.pos, 0.0);

				// Hmm

				// I think we do need to make v_r = 0. It's common sense that it IS 0
				// since we site our v_r estimate on the insulator. Since it is sited there,
				// it is used for traffic into the insulator by n,nT unless we pick out
				// insulator-abutting cells on purpose.

				// However, we then should make an energy correction -- at least if
				// momentum is coming into this minor cell and being destroyed.
			};
			
			memcpy(&(p_MAR_neut[iMinor]), &(MAR_neut), sizeof(f64_vec3));			
		}
		else {
			// Not domain, not crossing_ins, not a frill
			// ==========================================
		} // non-domain tri
	}; // was it FRILL
}

__global__ void kernelNeutral_pressure(
	structural * __restrict__ p_info_minor,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighTriMinor,
	char * __restrict__ p_szPBCtriminor,
	LONG3 * __restrict__ p_who_am_I_to_corners,
	LONG3 * __restrict__ p_tricornerindex,

	T3 * __restrict__ p_T_minor,
	ShardModel * __restrict__ p_n_shards,
	nvals * __restrict__ p_n_minor, // Just to handle insulator

	f64_vec3 * __restrict__ p_MAR_neut
)
{
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];
	__shared__ f64 shared_Tn[threadsPerTileMinor]; // 3+2+2+1=8 per thread

	__shared__ ShardModel shared_n_shards[threadsPerTileMajor];

	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajor];
	__shared__ f64 shared_Tn_verts[threadsPerTileMajor];  // 1/2( 13+3+2+2+1 = 21) = 10.5 => total 18.5 per minor thread.
														  // shame we couldn't get down to 16 per minor thread, and if we could then that might be better even if we load on-the-fly something.

	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	long const iVertex = threadsPerTileMajor*blockIdx.x + threadIdx.x; // only meaningful threadIdx.x < threadsPerTileMajor
	long const StartMinor = threadsPerTileMinor*blockIdx.x;
	long const StartMajor = threadsPerTileMajor*blockIdx.x;
	long const EndMinor = StartMinor + threadsPerTileMinor;
	long const EndMajor = StartMajor + threadsPerTileMajor;

	shared_pos[threadIdx.x] = p_info_minor[iMinor].pos; // QUESTION: DOES THIS LOAD CONTIGUOUSLY?
	shared_Tn[threadIdx.x] = p_T_minor[iMinor].Tn;		// QUESTION: DOES THIS LOAD CONTIGUOUSLY?
														// Advection should be an outer cycle at 1e-10 s.
	structural info;
	if (threadIdx.x < threadsPerTileMajor) {
		info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
		shared_pos_verts[threadIdx.x] = info.pos;
		if (info.flag == DOMAIN_VERTEX) {
			memcpy(&(shared_n_shards[threadIdx.x]), &(p_n_shards[iVertex]), sizeof(ShardModel)); // + 13
			shared_Tn_verts[threadIdx.x] = p_T_minor[iVertex + BEGINNING_OF_CENTRAL].Tn;
		}
		else {
			// it still manages to coalesce, let's hope, because ShardModel is 13 doubles not 12.
			memset(&(shared_n_shards[threadIdx.x]), 0, sizeof(ShardModel)); // + 13
			shared_Tn_verts[threadIdx.x] = 0.0;
		};
	};

	__syncthreads();

	f64 oppT, prevT, nextT, ourT;
	f64_vec2 opppos, prevpos, nextpos;
	f64 AreaMinor;

	if (threadIdx.x < threadsPerTileMajor) {
		AreaMinor = 0.0;
		f64_vec3 MAR_neut;
		memcpy(&(MAR_neut), &(p_MAR_neut[iVertex + BEGINNING_OF_CENTRAL]), sizeof(f64_vec3));

		long izTri[MAXNEIGH];
		char szPBC[MAXNEIGH];
		short tri_len = info.neigh_len;

		memcpy(izTri, p_izTri + iVertex*MAXNEIGH, MAXNEIGH * sizeof(long));
		memcpy(szPBC, p_szPBC + iVertex*MAXNEIGH, MAXNEIGH * sizeof(char));

		ourT = shared_Tn_verts[threadIdx.x];

		if (info.flag == DOMAIN_VERTEX) {

			short iprev = tri_len - 1;
			short i = 0;
			short inext;
			if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMinor))
			{
				prevT = shared_Tn[izTri[iprev] - StartMinor];
				prevpos = shared_pos[izTri[iprev] - StartMinor];
			}
			else {
				T3 prev_T = p_T_minor[izTri[iprev]];
				prevT = prev_T.Tn;
				prevpos = p_info_minor[izTri[iprev]].pos;
			}
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
			}

			if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
			{
				oppT = shared_Tn[izTri[i] - StartMinor];
				opppos = shared_pos[izTri[i] - StartMinor];
			}
			else {
				T3 opp_T = p_T_minor[izTri[i]];
				oppT = opp_T.Tn;
				opppos = p_info_minor[izTri[i]].pos;
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
			}

			// Think carefully: DOMAIN vertex cases for n,T ...
			f64 n0 = THIRD*(shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[iprev] + shared_n_shards[threadIdx.x].n_cent);
			f64_vec2 endpt0 = THIRD * (info.pos + opppos + prevpos);

			f64_vec2 endpt1, edge_normal;

			short iend = tri_len;
			f64_vec2 projendpt0;
			if ((info.flag == INNERMOST) || (info.flag == OUTERMOST)) {
				iend = tri_len - 2;
				if (info.flag == OUTERMOST) {
					endpt0.project_to_radius(projendpt0, FRILL_CENTROID_OUTER_RADIUS_d); // back of cell for Lap purposes
				}
				else {
					endpt0.project_to_radius(projendpt0, FRILL_CENTROID_INNER_RADIUS_d); // back of cell for Lap purposes
				}
				edge_normal.x = endpt0.y - projendpt0.y;
				edge_normal.y = projendpt0.x - endpt0.x;
				AreaMinor += (0.5*projendpt0.x + 0.5*endpt0.x)*edge_normal.x;
			};

			for (i = 0; i < iend; i++)
			{
				// Tri 0 is anticlockwise of neighbour 0, we think
				inext = i + 1; if (inext >= tri_len) inext = 0;

				if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
				{
					nextT = shared_Tn[izTri[inext] - StartMinor];
					nextpos = shared_pos[izTri[inext] - StartMinor];
				}
				else {
					T3 next_T = p_T_minor[izTri[inext]];
					nextT = next_T.Tn;
					nextpos = p_info_minor[izTri[inext]].pos;
				}
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
				}

				f64_vec2 endpt1 = THIRD * (nextpos + info.pos + opppos);
				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

				// ______________________________________________________-

				f64 n1;
				n1 = THIRD*(shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[inext] + shared_n_shards[threadIdx.x].n_cent);

				f64 T0, T1;
				T0 = THIRD*(prevT + ourT + oppT);
				T1 = THIRD*(nextT + ourT + oppT);
	
				// And we did what? We took n at centre of a triangle WITHIN this major cell 
				// But did not take upwind n ---- is that consistent for all advection?

				MAR_neut -= Make3(0.5*(n0*T0 + n1*T1)*over_m_n*edge_normal, 0.0);

				AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;
				endpt0 = endpt1;
				n0 = n1;
				prevpos = opppos;
				prevT = oppT;
				
				opppos = nextpos;
				oppT = nextT;
				
			}; // next i

			memcpy(p_MAR_neut + iVertex + BEGINNING_OF_CENTRAL, &(MAR_neut), sizeof(f64_vec3));
		}
		else {
			// NOT domain vertex: Do nothing
		};
	}; // was it domain vertex or Az-only
	   // This branching is itself a good argument for doing Az in ITS own separate routine with no need for n_shard.

	   // __syncthreads(); // end of first vertex part
	   // Do we need syncthreads? Not overwriting any shared data here...

	   // now the minor with n_ion part:
	info = p_info_minor[iMinor];
	ourT = shared_Tn[threadIdx.x];
	
	long izNeighMinor[6];
	char szPBC[6];
	memcpy(izNeighMinor, p_izNeighTriMinor + iMinor * 6, sizeof(long) * 6);
	memcpy(szPBC, p_szPBCtriminor + iMinor * 6, sizeof(char) * 6);

	f64_vec3 MAR_neut;
	memcpy(&(MAR_neut), &(p_MAR_neut[iMinor]), sizeof(f64_vec3));

	if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL)) {
		// Do nothing? Who cares what it is.
	}
	else {
		AreaMinor = 0.0;
		if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS)) {

			short inext, iprev = 5, i = 0;
			if ((izNeighMinor[iprev] >= StartMinor) && (izNeighMinor[iprev] < EndMinor))
			{
				prevT = shared_Tn[izNeighMinor[iprev] - StartMinor];
				prevpos = shared_pos[izNeighMinor[iprev] - StartMinor];
			} else {
				if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
					prevT = shared_Tn_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					prevpos = p_info_minor[izNeighMinor[iprev]].pos;
					prevT = p_T_minor[izNeighMinor[iprev]].Tn;
				};
			};
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
			};
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
			};

			i = 0;
			if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
			{
				opppos = shared_pos[izNeighMinor[i] - StartMinor];
				oppT = shared_Tn[izNeighMinor[i] - StartMinor];
			}
			else {
				if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					opppos = shared_pos_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
					oppT = shared_Tn_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					opppos = p_info_minor[izNeighMinor[i]].pos;
					T3 opp_T = p_T_minor[izNeighMinor[i]];
					oppT = opp_T.Tn;
				};
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
			}

			long who_am_I_to_corners[3];
			memcpy(who_am_I_to_corners, &(p_who_am_I_to_corners[iMinor]), sizeof(long) * 3);
			LONG3 cornerindex = p_tricornerindex[iMinor];
			// each corner we want to pick up 3 values off n_shards, as well as n_cent.
			// The three values will not always be contiguous!!!

			// Let's make life easier and load up an array of 6 n's beforehand.
			f64 n_array[6];
			f64 n0, n1;

			short who_am_I = who_am_I_to_corners[0];
			short tri_len = p_info_minor[cornerindex.i1 + BEGINNING_OF_CENTRAL].neigh_len;

			if ((cornerindex.i1 >= StartMajor) && (cornerindex.i1 < EndMajor))
			{
				short who_prev = who_am_I - 1;
				if (who_prev < 0) who_prev = tri_len - 1;
				// Worry about pathological cases later.
				short who_next = who_am_I + 1;
				if (who_next == tri_len) who_next = 0;
				n_array[0] = THIRD*(shared_n_shards[cornerindex.i1 - StartMajor].n[who_next]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n_cent);
				n_array[1] = THIRD*(shared_n_shards[cornerindex.i1 - StartMajor].n[who_prev]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n_cent);

			}
			else {
				// comes from elsewhere
				f64 ncent = p_n_shards[cornerindex.i1].n_cent;
				short who_prev = who_am_I - 1;
				if (who_prev < 0) {
					who_prev = tri_len - 1;
					f64_vec2 temp;
					memcpy(&temp, p_n_shards[cornerindex.i1].n, sizeof(f64_vec2));
					n_array[0] = THIRD*(temp.x + temp.y + ncent);
					n_array[1] = THIRD*(p_n_shards[cornerindex.i1].n[who_prev] + temp.x + ncent);
				}
				else {
					short who_next = who_am_I + 1;
					if (who_next == tri_len) {
						f64_vec2 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i1].n[who_prev]), sizeof(f64_vec2));
						n_array[0] = THIRD*(p_n_shards[cornerindex.i1].n[0] + temp.y + ncent);
						n_array[1] = THIRD*(temp.x + temp.y + ncent);
					}
					else {
						// typical case
						f64_vec3 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i1].n[who_prev]), sizeof(f64) * 3);
						n_array[0] = THIRD*(temp.z + temp.y + ncent);
						n_array[1] = THIRD*(temp.x + temp.y + ncent);
					};
				};
			}

			who_am_I = who_am_I_to_corners[1];
			tri_len = p_info_minor[cornerindex.i2 + BEGINNING_OF_CENTRAL].neigh_len;

			if ((cornerindex.i2 >= StartMajor) && (cornerindex.i2 < EndMajor))
			{
				short who_prev = who_am_I - 1;
				if (who_prev < 0) who_prev = tri_len - 1;
				// Worry about pathological cases later.
				short who_next = who_am_I + 1;
				if (who_next == tri_len) who_next = 0;
				n_array[2] = THIRD*(shared_n_shards[cornerindex.i2 - StartMajor].n[who_next]
					+ shared_n_shards[cornerindex.i2 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i2 - StartMajor].n_cent);
				n_array[3] = THIRD*(shared_n_shards[cornerindex.i2 - StartMajor].n[who_prev]
					+ shared_n_shards[cornerindex.i2 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i2 - StartMajor].n_cent);
			}
			else {
				// comes from elsewhere
				f64 ncent = p_n_shards[cornerindex.i2].n_cent;
				short who_prev = who_am_I - 1;
				if (who_prev < 0) {
					who_prev = tri_len - 1;
					f64_vec2 temp;
					memcpy(&temp, p_n_shards[cornerindex.i2].n, sizeof(f64_vec2));
					n_array[2] = THIRD*(temp.x + temp.y + ncent);
					n_array[3] = THIRD*(p_n_shards[cornerindex.i2].n[who_prev] + temp.x + ncent);
				}
				else {
					short who_next = who_am_I + 1;
					if (who_next == tri_len) {
						f64_vec2 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i2].n[who_prev]), sizeof(f64_vec2));
						n_array[2] = THIRD*(p_n_shards[cornerindex.i2].n[0] + temp.y + ncent);
						n_array[3] = THIRD*(temp.x + temp.y + ncent);
					}
					else {
						// typical case
						f64_vec3 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i2].n[who_prev]), sizeof(f64) * 3);
						n_array[2] = THIRD*(temp.z + temp.y + ncent);
						n_array[3] = THIRD*(temp.x + temp.y + ncent);
					};
				};
			}

			who_am_I = who_am_I_to_corners[2];
			tri_len = p_info_minor[cornerindex.i3 + BEGINNING_OF_CENTRAL].neigh_len;

			if ((cornerindex.i3 >= StartMajor) && (cornerindex.i3 < EndMajor))
			{
				short who_prev = who_am_I - 1;
				if (who_prev < 0) who_prev = tri_len - 1;
				// Worry about pathological cases later.
				short who_next = who_am_I + 1;
				if (who_next == tri_len) who_next = 0;
				n_array[4] = THIRD*(shared_n_shards[cornerindex.i3 - StartMajor].n[who_next]
					+ shared_n_shards[cornerindex.i3 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i3 - StartMajor].n_cent);
				n_array[5] = THIRD*(shared_n_shards[cornerindex.i3 - StartMajor].n[who_prev]
					+ shared_n_shards[cornerindex.i3 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i3 - StartMajor].n_cent);
			}
			else {
				// comes from elsewhere
				f64 ncent = p_n_shards[cornerindex.i3].n_cent;
				short who_prev = who_am_I - 1;
				if (who_prev < 0) {
					who_prev = tri_len - 1;
					f64_vec2 temp;
					memcpy(&temp, p_n_shards[cornerindex.i3].n, sizeof(f64_vec2));
					n_array[4] = THIRD*(temp.x + temp.y + ncent);
					n_array[5] = THIRD*(p_n_shards[cornerindex.i3].n[who_prev] + temp.x + ncent);
				}
				else {
					short who_next = who_am_I + 1;
					if (who_next == tri_len) {
						f64_vec2 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i3].n[who_prev]), sizeof(f64_vec2));
						n_array[4] = THIRD*(p_n_shards[cornerindex.i3].n[0] + temp.y + ncent);
						n_array[5] = THIRD*(temp.x + temp.y + ncent);
					}
					else {
						// typical case
						f64_vec3 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i3].n[who_prev]), sizeof(f64) * 3);
						n_array[4] = THIRD*(temp.z + temp.y + ncent);
						n_array[5] = THIRD*(temp.x + temp.y + ncent);
					};
				};
			}

#pragma unroll 
			for (i = 0; i < 6; i++)
			{
				inext = i + 1; if (inext > 5) inext = 0;
				if ((izNeighMinor[inext] >= StartMinor) && (izNeighMinor[inext] < EndMinor))
				{
					nextpos = shared_pos[izNeighMinor[inext] - StartMinor];
					nextT = shared_Tn[izNeighMinor[inext] - StartMinor];
				}
				else {
					if ((izNeighMinor[inext] >= StartMajor + BEGINNING_OF_CENTRAL) &&
						(izNeighMinor[inext] < EndMajor + BEGINNING_OF_CENTRAL))
					{
						nextpos = shared_pos_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
						nextT = shared_Tn_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
					}
					else {
						nextpos = p_info_minor[izNeighMinor[inext]].pos;
						nextT = p_T_minor[izNeighMinor[inext]].Tn;
					};
				};
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
				}

				// New definition of endpoint of minor edge:
				f64_vec2 endpt0, endpt1, edge_normal;

				endpt0 = THIRD * (prevpos + info.pos + opppos);
				endpt1 = THIRD * (nextpos + info.pos + opppos);

				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

				n0 = n_array[i];
				n1 = n_array[inext]; // 0,1 are either side of corner 0. What is seq of MinorNeigh ?

									 // Assume neighs 0,1 are relevant to border with tri 0 minor.

				//if (((izNeighMinor[i] >= NumInnerFrills_d) && (izNeighMinor[i] < FirstOuterFrill_d)))
				{	// Decided not to add test

					f64 T0 = THIRD*(ourT + prevT + oppT);
					f64 T1 = THIRD*(ourT + nextT + oppT);

					if (info.flag == CROSSING_INS) {
						char flag = p_info_minor[izNeighMinor[i]].flag;
						if ((flag == DOMAIN_TRIANGLE) || (flag == DOMAIN_VERTEX))
						{
							// do nothing
						}
						else {
							// Looking into the insulator we see a reflection of nT. Here we look into an out-of-domain tri or vert below ins.
							// Or allowed a below-ins value to affect something anyway.
							// Just for sanity for now, let's just set our own n,T for the edge:
							n0 = p_n_minor[iMinor].n_n;
							n1 = p_n_minor[iMinor].n_n;
							T0 = ourT;
							T1 = ourT;
						}
					}

					MAR_neut -= Make3(0.5*(n0*T0 + n1*T1)*over_m_n*edge_normal, 0.0);
				}

				endpt0 = endpt1;
				prevT = oppT;
				prevpos = opppos;
				oppT = nextT;
				opppos = nextpos;
			};

			if (info.flag == CROSSING_INS) {
				// In this case set v_r = 0 and set a_TP_r = 0 and dv/dt _r = 0 in general
				//f64_vec2 rhat = info.pos / info.pos.modulus();
				MAR_neut -= Make3(
					(MAR_neut.dotxy(info.pos) /
					(info.pos.x*info.pos.x + info.pos.y*info.pos.y))*info.pos, 0.0);

				// Hmm

				// I think we do need to make v_r = 0. It's common sense that it IS 0
				// since we site our v_r estimate on the insulator. Since it is sited there,
				// it is used for traffic into the insulator by n,nT unless we pick out
				// insulator-abutting cells on purpose.

				// However, we then should make an energy correction -- at least if
				// momentum is coming into this minor cell and being destroyed.
			};

			memcpy(&(p_MAR_neut[iMinor]), &(MAR_neut), sizeof(f64_vec3));
		}
		else {
			// Not domain, not crossing_ins, not a frill
			// ==========================================
		} // non-domain tri
	}; // was it FRILL
}


__global__ void kernelNeutral_momflux(
	structural * __restrict__ p_info_minor,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighTriMinor,
	char * __restrict__ p_szPBCtriminor,
	LONG3 * __restrict__ p_who_am_I_to_corners,
	LONG3 * __restrict__ p_tricornerindex,

	f64_vec3 * __restrict__ p_v_n_minor,
	ShardModel * __restrict__ p_n_shards,
	nvals * __restrict__ p_n_minor, // Just to handle insulator

	f64_vec2 * __restrict__ p_v_overall_minor,
	f64_vec3 * __restrict__ p_MAR_neut
)
{
	__shared__ f64_vec3 shared_v_n[threadsPerTileMinor];
	__shared__ f64_vec2 shared_v_overall[threadsPerTileMinor];
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];
	
	__shared__ ShardModel shared_n_shards[threadsPerTileMajor];

	__shared__ f64_vec3 shared_v_n_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_v_overall_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajor];
														  // shame we couldn't get down to 16 per minor thread, and if we could then that might be better even if we load on-the-fly something.

	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	long const iVertex = threadsPerTileMajor*blockIdx.x + threadIdx.x; // only meaningful threadIdx.x < threadsPerTileMajor
	long const StartMinor = threadsPerTileMinor*blockIdx.x;
	long const StartMajor = threadsPerTileMajor*blockIdx.x;
	long const EndMinor = StartMinor + threadsPerTileMinor;
	long const EndMajor = StartMajor + threadsPerTileMajor;

	shared_pos[threadIdx.x] = p_info_minor[iMinor].pos; // QUESTION: DOES THIS LOAD CONTIGUOUSLY?
	shared_v_n[threadIdx.x] = p_v_n_minor[iMinor];
	shared_v_overall[threadIdx.x] = p_v_overall_minor[iMinor];
	
														// Advection should be an outer cycle at 1e-10 s.

	structural info;
	if (threadIdx.x < threadsPerTileMajor) {
		info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
		shared_pos_verts[threadIdx.x] = info.pos;
		if (info.flag == DOMAIN_VERTEX) {
			memcpy(&(shared_n_shards[threadIdx.x]), &(p_n_shards[iVertex]), sizeof(ShardModel)); // + 13
			memcpy(&(shared_v_n_verts[threadIdx.x]), &(p_v_n_minor[iVertex + BEGINNING_OF_CENTRAL]), sizeof(f64_vec3));
			shared_v_overall_verts[threadIdx.x] = p_v_overall_minor[iVertex + BEGINNING_OF_CENTRAL];
		}
		else {
			// it still manages to coalesce, let's hope, because ShardModel is 13 doubles not 12.
			memset(&(shared_n_shards[threadIdx.x]), 0, sizeof(ShardModel)); // + 13
			memset(&(shared_v_n_verts[threadIdx.x]), 0, sizeof(f64_vec3));
			memset(&(shared_v_overall_verts[threadIdx.x]), 0, sizeof(f64_vec2));
		};
	};

	__syncthreads();

	f64_vec3 our_v, opp_v, prev_v, next_v;
		f64_vec2 our_v_overall, prev_v_overall, next_v_overall, opp_v_overall;
	f64_vec2 opppos, prevpos, nextpos;
	f64 AreaMinor;

	if (threadIdx.x < threadsPerTileMajor) {
		AreaMinor = 0.0;
		f64_vec3 MAR_neut;
		memcpy(&(MAR_neut), &(p_MAR_neut[iVertex + BEGINNING_OF_CENTRAL]), sizeof(f64_vec3));

		long izTri[MAXNEIGH];
		char szPBC[MAXNEIGH];
		short tri_len = info.neigh_len;

		memcpy(izTri, p_izTri + iVertex*MAXNEIGH, MAXNEIGH * sizeof(long));
		memcpy(szPBC, p_szPBC + iVertex*MAXNEIGH, MAXNEIGH * sizeof(char));

		our_v = shared_v_n_verts[threadIdx.x];
		our_v_overall = shared_v_overall_verts[threadIdx.x];
		
		if (info.flag == DOMAIN_VERTEX) {

			short iprev = tri_len - 1;
			short i = 0;
			short inext;
			if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMinor))
			{
				prev_v = shared_v_n[izTri[iprev] - StartMinor];
				prev_v_overall = shared_v_overall[izTri[iprev] - StartMinor];
				prevpos = shared_pos[izTri[iprev] - StartMinor];
			}
			else {
				prev_v = p_v_n_minor[izTri[iprev]];
				prev_v_overall = p_v_overall_minor[izTri[iprev]];
				prevpos = p_info_minor[izTri[iprev]].pos;
			}
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
				prev_v = Clockwise_rotate3(prev_v);
				prev_v_overall = Clockwise_d*prev_v_overall;
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
				prev_v = Anticlock_rotate3(prev_v);
				prev_v_overall = Anticlockwise_d*prev_v_overall;
			}

			if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
			{
				opp_v = shared_v_n[izTri[i] - StartMinor];
				opp_v_overall = shared_v_overall[izTri[i] - StartMinor];
				opppos = shared_pos[izTri[i] - StartMinor];
			}
			else {
				opp_v = p_v_n_minor[izTri[i]];
				opp_v_overall = p_v_overall_minor[izTri[i]];
				opppos = p_info_minor[izTri[i]].pos;
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
				opp_v = Clockwise_rotate3(opp_v);
				opp_v_overall = Clockwise_d*opp_v_overall;
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
				opp_v = Anticlock_rotate3(opp_v);
				opp_v_overall = Anticlockwise_d*opp_v_overall;
			}

			// Think carefully: DOMAIN vertex cases for n,T ...
			f64 n0 = THIRD*(shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[iprev] + shared_n_shards[threadIdx.x].n_cent);
			f64_vec2 endpt0 = THIRD * (info.pos + opppos + prevpos);

			f64_vec2 endpt1, edge_normal;

			short iend = tri_len;
			f64_vec2 projendpt0;
			if ((info.flag == INNERMOST) || (info.flag == OUTERMOST)) {
				iend = tri_len - 2;
				if (info.flag == OUTERMOST) {
					endpt0.project_to_radius(projendpt0, FRILL_CENTROID_OUTER_RADIUS_d); // back of cell for Lap purposes
				}
				else {
					endpt0.project_to_radius(projendpt0, FRILL_CENTROID_INNER_RADIUS_d); // back of cell for Lap purposes
				}
				edge_normal.x = endpt0.y - projendpt0.y;
				edge_normal.y = projendpt0.x - endpt0.x;
				AreaMinor += (0.5*projendpt0.x + 0.5*endpt0.x)*edge_normal.x;
			};

			for (i = 0; i < iend; i++)
			{
				// Tri 0 is anticlockwise of neighbour 0, we think
				inext = i + 1; if (inext >= tri_len) inext = 0;

				if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
				{
					next_v = shared_v_n[izTri[inext] - StartMinor];
					next_v_overall = shared_v_overall[izTri[inext] - StartMinor];
					nextpos = shared_pos[izTri[inext] - StartMinor];
				}
				else {
					next_v = p_v_n_minor[izTri[inext]];
					next_v_overall = p_v_overall_minor[izTri[inext]];
					nextpos = p_info_minor[izTri[inext]].pos;
				}
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
					next_v = Clockwise_rotate3(next_v);
					next_v_overall = Clockwise_d*next_v_overall;
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
					next_v = Anticlock_rotate3(next_v);
					next_v_overall = Anticlockwise_d*next_v_overall;
				}

				f64_vec2 endpt1 = THIRD * (nextpos + info.pos + opppos);
				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

				// ______________________________________________________-

				f64 n1;
				n1 = THIRD*(shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[inext] + shared_n_shards[threadIdx.x].n_cent);

				f64_vec3 v0 = THIRD*(our_v + prev_v + opp_v);
				f64_vec3 v1 = THIRD*(our_v + opp_v + next_v);

				f64 relvnormal = 0.5*((v0 + v1).xypart()
					- (THIRD * (our_v_overall + next_v_overall + opp_v_overall))
					- (THIRD * (our_v_overall + prev_v_overall + opp_v_overall))
					).dot(edge_normal);
				// CHANGES 20th August 2019
				// OLD, unstable:
				// MAR_neut -= 0.5*relvnormal* (n0 *(v0-our_v) + n1 * (v1 - our_v));

				if (relvnormal < 0.0)
					MAR_neut -= 0.5*relvnormal* (n0 + n1) *(opp_v - our_v);
				// Note: minus a minus so correct sign

				// And we did what? We took n at centre of a triangle WITHIN this major cell 
				// But did not take upwind n ---- is that consistent for all advection?
				// NO

			
				// ______________________________________________________
				//// whether the v that is leaving is greater than our v ..
				//// Formula:
				//// dv/dt = (d(Nv)/dt - dN/dt v) / N
				//// We include the divide by N when we enter the accel routine.

				AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;
				endpt0 = endpt1;
				n0 = n1;
				prevpos = opppos;
				prev_v = opp_v;
				prev_v_overall = opp_v_overall;

				opppos = nextpos;
				opp_v = next_v;
				opp_v_overall = next_v_overall;
			}; // next i

			memcpy(p_MAR_neut + iVertex + BEGINNING_OF_CENTRAL, &(MAR_neut), sizeof(f64_vec3));
		}
		else {
			// NOT domain vertex: Do nothing
		};
	}; // was it domain vertex or Az-only
	   // This branching is itself a good argument for doing Az in ITS own separate routine with no need for n_shard.

	   // __syncthreads(); // end of first vertex part
	   // Do we need syncthreads? Not overwriting any shared data here...

	   // now the minor with n_ion part:
	info = p_info_minor[iMinor];
	our_v = shared_v_n[threadIdx.x];
	our_v_overall = shared_v_overall[threadIdx.x];

	long izNeighMinor[6];
	char szPBC[6];
	memcpy(izNeighMinor, p_izNeighTriMinor + iMinor * 6, sizeof(long) * 6);
	memcpy(szPBC, p_szPBCtriminor + iMinor * 6, sizeof(char) * 6);

	f64_vec3 MAR_neut;
	memcpy(&(MAR_neut), &(p_MAR_neut[iMinor]), sizeof(f64_vec3));

	if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL)) {
		// Do nothing? Who cares what it is.
	}
	else {
		AreaMinor = 0.0;
		if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS)) {

			short inext, iprev = 5, i = 0;
			if ((izNeighMinor[iprev] >= StartMinor) && (izNeighMinor[iprev] < EndMinor))
			{
				memcpy(&prev_v, &(shared_v_n[izNeighMinor[iprev] - StartMinor]), sizeof(f64_vec3));
				prevpos = shared_pos[izNeighMinor[iprev] - StartMinor];
				prev_v_overall = shared_v_overall[izNeighMinor[iprev] - StartMinor];
			}
			else {
				if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					memcpy(&prev_v, &(shared_v_n_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(f64_vec3));
					prev_v_overall = shared_v_overall_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
					prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					prevpos = p_info_minor[izNeighMinor[iprev]].pos;
					memcpy(&prev_v, &(p_v_n_minor[izNeighMinor[iprev]]), sizeof(f64_vec3));
					prev_v_overall = p_v_overall_minor[izNeighMinor[iprev]];
				};
			};
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
				prev_v = Clockwise_rotate3(prev_v);
				prev_v_overall = Clockwise_d*prev_v_overall;
			};
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
				prev_v = Anticlock_rotate3(prev_v);
				prev_v_overall = Anticlockwise_d*prev_v_overall;
			};

			i = 0;
			if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
			{
				memcpy(&opp_v, &(shared_v_n[izNeighMinor[i] - StartMinor]), sizeof(f64_vec3));
				opppos = shared_pos[izNeighMinor[i] - StartMinor];
				opp_v_overall = shared_v_overall[izNeighMinor[i] - StartMinor];
			}
			else {
				if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					memcpy(&opp_v, &(shared_v_n_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(f64_vec3));
					opp_v_overall = shared_v_overall_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
					opppos = shared_pos_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					opppos = p_info_minor[izNeighMinor[i]].pos;
					memcpy(&opp_v, &(p_v_n_minor[izNeighMinor[i]]), sizeof(f64_vec3));
					opp_v_overall = p_v_overall_minor[izNeighMinor[i]];
				};
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
				opp_v = Clockwise_rotate3(opp_v);
				opp_v_overall = Clockwise_d*opp_v_overall;
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
				opp_v = Anticlock_rotate3(opp_v);
				opp_v_overall = Anticlockwise_d*opp_v_overall;
			}

			long who_am_I_to_corners[3];
			memcpy(who_am_I_to_corners, &(p_who_am_I_to_corners[iMinor]), sizeof(long) * 3);
			LONG3 cornerindex = p_tricornerindex[iMinor];
			// each corner we want to pick up 3 values off n_shards, as well as n_cent.
			// The three values will not always be contiguous!!!

			// Let's make life easier and load up an array of 6 n's beforehand.
			f64 n_array[6];
			f64 n0, n1;

			short who_am_I = who_am_I_to_corners[0];
			short tri_len = p_info_minor[cornerindex.i1 + BEGINNING_OF_CENTRAL].neigh_len;

			if ((cornerindex.i1 >= StartMajor) && (cornerindex.i1 < EndMajor))
			{
				short who_prev = who_am_I - 1;
				if (who_prev < 0) who_prev = tri_len - 1;
				// Worry about pathological cases later.
				short who_next = who_am_I + 1;
				if (who_next == tri_len) who_next = 0;
				n_array[0] = THIRD*(shared_n_shards[cornerindex.i1 - StartMajor].n[who_next]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n_cent);
				n_array[1] = THIRD*(shared_n_shards[cornerindex.i1 - StartMajor].n[who_prev]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n_cent);

			}
			else {
				// comes from elsewhere
				f64 ncent = p_n_shards[cornerindex.i1].n_cent;
				short who_prev = who_am_I - 1;
				if (who_prev < 0) {
					who_prev = tri_len - 1;
					f64_vec2 temp;
					memcpy(&temp, p_n_shards[cornerindex.i1].n, sizeof(f64_vec2));
					n_array[0] = THIRD*(temp.x + temp.y + ncent);
					n_array[1] = THIRD*(p_n_shards[cornerindex.i1].n[who_prev] + temp.x + ncent);
				}
				else {
					short who_next = who_am_I + 1;
					if (who_next == tri_len) {
						f64_vec2 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i1].n[who_prev]), sizeof(f64_vec2));
						n_array[0] = THIRD*(p_n_shards[cornerindex.i1].n[0] + temp.y + ncent);
						n_array[1] = THIRD*(temp.x + temp.y + ncent);
					}
					else {
						// typical case
						f64_vec3 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i1].n[who_prev]), sizeof(f64) * 3);
						n_array[0] = THIRD*(temp.z + temp.y + ncent);
						n_array[1] = THIRD*(temp.x + temp.y + ncent);
					};
				};
			}

			who_am_I = who_am_I_to_corners[1];
			tri_len = p_info_minor[cornerindex.i2 + BEGINNING_OF_CENTRAL].neigh_len;

			if ((cornerindex.i2 >= StartMajor) && (cornerindex.i2 < EndMajor))
			{
				short who_prev = who_am_I - 1;
				if (who_prev < 0) who_prev = tri_len - 1;
				// Worry about pathological cases later.
				short who_next = who_am_I + 1;
				if (who_next == tri_len) who_next = 0;
				n_array[2] = THIRD*(shared_n_shards[cornerindex.i2 - StartMajor].n[who_next]
					+ shared_n_shards[cornerindex.i2 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i2 - StartMajor].n_cent);
				n_array[3] = THIRD*(shared_n_shards[cornerindex.i2 - StartMajor].n[who_prev]
					+ shared_n_shards[cornerindex.i2 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i2 - StartMajor].n_cent);
			}
			else {
				// comes from elsewhere
				f64 ncent = p_n_shards[cornerindex.i2].n_cent;
				short who_prev = who_am_I - 1;
				if (who_prev < 0) {
					who_prev = tri_len - 1;
					f64_vec2 temp;
					memcpy(&temp, p_n_shards[cornerindex.i2].n, sizeof(f64_vec2));
					n_array[2] = THIRD*(temp.x + temp.y + ncent);
					n_array[3] = THIRD*(p_n_shards[cornerindex.i2].n[who_prev] + temp.x + ncent);
				}
				else {
					short who_next = who_am_I + 1;
					if (who_next == tri_len) {
						f64_vec2 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i2].n[who_prev]), sizeof(f64_vec2));
						n_array[2] = THIRD*(p_n_shards[cornerindex.i2].n[0] + temp.y + ncent);
						n_array[3] = THIRD*(temp.x + temp.y + ncent);
					}
					else {
						// typical case
						f64_vec3 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i2].n[who_prev]), sizeof(f64) * 3);
						n_array[2] = THIRD*(temp.z + temp.y + ncent);
						n_array[3] = THIRD*(temp.x + temp.y + ncent);
					};
				};
			}

			who_am_I = who_am_I_to_corners[2];
			tri_len = p_info_minor[cornerindex.i3 + BEGINNING_OF_CENTRAL].neigh_len;

			if ((cornerindex.i3 >= StartMajor) && (cornerindex.i3 < EndMajor))
			{
				short who_prev = who_am_I - 1;
				if (who_prev < 0) who_prev = tri_len - 1;
				// Worry about pathological cases later.
				short who_next = who_am_I + 1;
				if (who_next == tri_len) who_next = 0;
				n_array[4] = THIRD*(shared_n_shards[cornerindex.i3 - StartMajor].n[who_next]
					+ shared_n_shards[cornerindex.i3 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i3 - StartMajor].n_cent);
				n_array[5] = THIRD*(shared_n_shards[cornerindex.i3 - StartMajor].n[who_prev]
					+ shared_n_shards[cornerindex.i3 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i3 - StartMajor].n_cent);
			}
			else {
				// comes from elsewhere
				f64 ncent = p_n_shards[cornerindex.i3].n_cent;
				short who_prev = who_am_I - 1;
				if (who_prev < 0) {
					who_prev = tri_len - 1;
					f64_vec2 temp;
					memcpy(&temp, p_n_shards[cornerindex.i3].n, sizeof(f64_vec2));
					n_array[4] = THIRD*(temp.x + temp.y + ncent);
					n_array[5] = THIRD*(p_n_shards[cornerindex.i3].n[who_prev] + temp.x + ncent);
				}
				else {
					short who_next = who_am_I + 1;
					if (who_next == tri_len) {
						f64_vec2 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i3].n[who_prev]), sizeof(f64_vec2));
						n_array[4] = THIRD*(p_n_shards[cornerindex.i3].n[0] + temp.y + ncent);
						n_array[5] = THIRD*(temp.x + temp.y + ncent);
					}
					else {
						// typical case
						f64_vec3 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i3].n[who_prev]), sizeof(f64) * 3);
						n_array[4] = THIRD*(temp.z + temp.y + ncent);
						n_array[5] = THIRD*(temp.x + temp.y + ncent);
					};
				};
			}

#pragma unroll 
			for (i = 0; i < 6; i++)
			{
				inext = i + 1; if (inext > 5) inext = 0;
				if ((izNeighMinor[inext] >= StartMinor) && (izNeighMinor[inext] < EndMinor))
				{
					memcpy(&next_v, &(shared_v_n[izNeighMinor[inext] - StartMinor]), sizeof(f64_vec3));
					nextpos = shared_pos[izNeighMinor[inext] - StartMinor];
					next_v_overall = shared_v_overall[izNeighMinor[inext] - StartMinor];
				}
				else {
					if ((izNeighMinor[inext] >= StartMajor + BEGINNING_OF_CENTRAL) &&
						(izNeighMinor[inext] < EndMajor + BEGINNING_OF_CENTRAL))
					{
						memcpy(&next_v, &(shared_v_n_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor]), sizeof(f64_vec3));
						next_v_overall = shared_v_overall_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
						nextpos = shared_pos_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
					} else {
						nextpos = p_info_minor[izNeighMinor[inext]].pos;
						memcpy(&next_v, &(p_v_n_minor[izNeighMinor[inext]]), sizeof(f64_vec3));
						next_v_overall = p_v_overall_minor[izNeighMinor[inext]];
					};
				};
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
					next_v = Clockwise_rotate3(next_v);
					next_v_overall = Clockwise_d*next_v_overall;
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
					next_v = Anticlock_rotate3(next_v);
					next_v_overall = Anticlockwise_d*next_v_overall;
				}

				// New definition of endpoint of minor edge:
				f64_vec2 endpt0, endpt1, edge_normal;

				endpt0 = THIRD * (prevpos + info.pos + opppos);
				endpt1 = THIRD * (nextpos + info.pos + opppos);

				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

				n0 = n_array[i];
				n1 = n_array[inext]; // 0,1 are either side of corner 0. What is seq of MinorNeigh ?

									 // Assume neighs 0,1 are relevant to border with tri 0 minor.

				f64_vec3 v0 = THIRD*(our_v + prev_v + opp_v);
				f64_vec3 v1 = THIRD*(our_v + next_v + opp_v);

				//if (((izNeighMinor[i] >= NumInnerFrills_d) && (izNeighMinor[i] < FirstOuterFrill_d)))
				{	// Decided not to add test
					f64 relvnormal = 0.5*((v0 + v1).xypart()
						- (THIRD * (our_v_overall + next_v_overall + opp_v_overall))
						- (THIRD * (our_v_overall + prev_v_overall + opp_v_overall))
						).dot(edge_normal);

					// CHANGES 20th August 2019:
					// OLD, unstable:
					// MAR_neut -= 0.5*relvnormal* (n0 *(v0-our_v) + n1 * (v1 - our_v));
					if (relvnormal < 0.0)
						MAR_neut -= 0.5*relvnormal* (n0 + n1) *(opp_v - our_v);

				}

				endpt0 = endpt1;
				prevpos = opppos;
				prev_v = opp_v;
				prev_v_overall = opp_v_overall;
				opppos = nextpos;
				opp_v = next_v;
				opp_v_overall = next_v_overall;
			};

			if (info.flag == CROSSING_INS) {
				// In this case set v_r = 0 and set a_TP_r = 0 and dv/dt _r = 0 in general
				//f64_vec2 rhat = info.pos / info.pos.modulus();
				MAR_neut -= Make3(
					(MAR_neut.dotxy(info.pos) /
					(info.pos.x*info.pos.x + info.pos.y*info.pos.y))*info.pos, 0.0);

				// Hmm

				// I think we do need to make v_r = 0. It's common sense that it IS 0
				// since we site our v_r estimate on the insulator. Since it is sited there,
				// it is used for traffic into the insulator by n,nT unless we pick out
				// insulator-abutting cells on purpose.

				// However, we then should make an energy correction -- at least if
				// momentum is coming into this minor cell and being destroyed.
			};

			memcpy(&(p_MAR_neut[iMinor]), &(MAR_neut), sizeof(f64_vec3));
		}
		else {
			// Not domain, not crossing_ins, not a frill
			// ==========================================
		} // non-domain tri
	}; // was it FRILL
}

__global__ void kernelAntiAdvect(
	f64 const h_use,
	structural * __restrict__ p_info_minor,

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighTriMinor,
	char * __restrict__ p_szPBCtriminor,

	AAdot * __restrict__ p_AAdot,
	f64_vec2 * __restrict__ p_v_overall_minor,
	AAdot * __restrict__ p_AAdot_dest
)
{
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];
	__shared__ AAdot shared_AAdot[threadsPerTileMinor]; // 3+2+2+1=8 per thread

	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajor];
	__shared__ AAdot shared_AAdot_verts[threadsPerTileMajor];

	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	long const iVertex = threadsPerTileMajor*blockIdx.x + threadIdx.x; // only meaningful threadIdx.x < threadsPerTileMajor
	long const StartMinor = threadsPerTileMinor*blockIdx.x;
	long const StartMajor = threadsPerTileMajor*blockIdx.x;
	long const EndMinor = StartMinor + threadsPerTileMinor;
	long const EndMajor = StartMajor + threadsPerTileMajor;

	structural info = p_info_minor[iMinor];

	shared_pos[threadIdx.x] = info.pos;
	shared_AAdot[threadIdx.x] = p_AAdot[iMinor];
	
	if (threadIdx.x < threadsPerTileMajor) {
		info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
		shared_pos_verts[threadIdx.x] = info.pos;
		shared_AAdot_verts[threadIdx.x] = p_AAdot[iVertex + BEGINNING_OF_CENTRAL];
	};

	__syncthreads();


	AAdot oppA, prevA, nextA, ourA;
	f64_vec2 opppos, prevpos, nextpos, Integ_grad_Az, Integ_grad_Azdot;
	f64 AreaMinor;
	
	if (threadIdx.x < threadsPerTileMajor) 
	{
		if (info.flag == DOMAIN_VERTEX) {// otherwise no move
			AreaMinor = 0.0;

			Integ_grad_Az.x = 0.0;
			Integ_grad_Az.y = 0.0;
			Integ_grad_Azdot.x = 0.0;
			Integ_grad_Azdot.y = 0.0;
			long izTri[MAXNEIGH];
			char szPBC[MAXNEIGH];
			short tri_len = info.neigh_len;

			memcpy(izTri, p_izTri + iVertex*MAXNEIGH, MAXNEIGH * sizeof(long));
			memcpy(szPBC, p_szPBC + iVertex*MAXNEIGH, MAXNEIGH * sizeof(char));

			ourA = shared_AAdot_verts[threadIdx.x];

			short iprev = tri_len - 1;
			short i = 0;
			short inext;
			if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMinor))
			{
				prevA = shared_AAdot[izTri[iprev] - StartMinor];
				prevpos = shared_pos[izTri[iprev] - StartMinor];
			}
			else {
				prevA = p_AAdot[izTri[iprev]];
				prevpos = p_info_minor[izTri[iprev]].pos;
			}
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise_d*prevpos;
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise_d*prevpos;
			}

			if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
			{
				oppA = shared_AAdot[izTri[i] - StartMinor];
				opppos = shared_pos[izTri[i] - StartMinor];
			}
			else {
				oppA = p_AAdot[izTri[i]];
				opppos = p_info_minor[izTri[i]].pos;
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise_d*opppos;
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise_d*opppos;
			}

			// Think carefully: DOMAIN vertex cases for n,T ...
			f64_vec2 endpt0 = THIRD * (info.pos + opppos + prevpos);
			f64_vec2 endpt1, edge_normal;

			short iend = tri_len;
			f64_vec2 projendpt0;
			if ((info.flag == INNERMOST) || (info.flag == OUTERMOST)) {
				iend = tri_len - 2;
				if (info.flag == OUTERMOST) {
					endpt0.project_to_radius(projendpt0, FRILL_CENTROID_OUTER_RADIUS_d); // back of cell for Lap purposes
				}
				else {
					endpt0.project_to_radius(projendpt0, FRILL_CENTROID_INNER_RADIUS_d); // back of cell for Lap purposes
				}
				edge_normal.x = endpt0.y - projendpt0.y;
				edge_normal.y = projendpt0.x - endpt0.x;
				AreaMinor += (0.5*projendpt0.x + 0.5*endpt0.x)*edge_normal.x;
			};

			for (i = 0; i < iend; i++)
			{
				// Tri 0 is anticlockwise of neighbour 0, we think
				inext = i + 1; if (inext >= tri_len) inext = 0;

				if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor))
				{
					nextA = shared_AAdot[izTri[inext] - StartMinor];
					nextpos = shared_pos[izTri[inext] - StartMinor];
				}
				else {
					nextA = p_AAdot[izTri[inext]];
					nextpos = p_info_minor[izTri[inext]].pos;
				}
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise_d*nextpos;
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise_d*nextpos;
				}

				f64_vec2 endpt1 = THIRD * (nextpos + info.pos + opppos);
				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

				// ______________________________________________________-

				// And we did what? We took n at centre of a triangle WITHIN this major cell 
				// But did not take upwind n ---- is that consistent for all advection?

				f64 Az_edge = SIXTH * (2.0*ourA.Az + 2.0*oppA.Az + prevA.Az + nextA.Az);
				Integ_grad_Az += Az_edge*edge_normal;
				f64 Azdot_edge = SIXTH * (2.0*ourA.Azdot + 2.0*oppA.Azdot + prevA.Azdot + nextA.Azdot);
				Integ_grad_Azdot += Azdot_edge*edge_normal;

				AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;
				endpt0 = endpt1;
				prevpos = opppos;
				prevA = oppA;

				opppos = nextpos;
				oppA = nextA;
			}; // next i

			f64_vec2 Grad_Az = Integ_grad_Az / AreaMinor;
			f64_vec2 Grad_Azdot = Integ_grad_Azdot / AreaMinor;

			AAdot AAdot_dest;

			f64_vec2 v_overall = p_v_overall_minor[iVertex + BEGINNING_OF_CENTRAL];

			AAdot_dest.Az = ourA.Az - h_use*Grad_Az.dot(v_overall);
			AAdot_dest.Azdot = ourA.Azdot - h_use*Grad_Azdot.dot(v_overall);

			p_AAdot_dest[iVertex + BEGINNING_OF_CENTRAL] = AAdot_dest;
		} else {
			p_AAdot_dest[iVertex + BEGINNING_OF_CENTRAL] = shared_AAdot_verts[threadIdx.x];
		};

	}; 

	

	// now the minor with n_ion part:
	info = p_info_minor[iMinor];
	ourA = shared_AAdot[threadIdx.x];
	
	long izNeighMinor[6];
	char szPBC[6];
	memcpy(izNeighMinor, p_izNeighTriMinor + iMinor * 6, sizeof(long) * 6);
	memcpy(szPBC, p_szPBCtriminor + iMinor * 6, sizeof(char) * 6);
		
	AreaMinor = 0.0;
	if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS)) {

		Integ_grad_Az.x = 0.0;
		Integ_grad_Az.y = 0.0;
		Integ_grad_Azdot.x = 0.0;
		Integ_grad_Azdot.y = 0.0;

		short inext, iprev = 5, i = 0;
		if ((izNeighMinor[iprev] >= StartMinor) && (izNeighMinor[iprev] < EndMinor))
		{
			prevA = shared_AAdot[izNeighMinor[iprev] - StartMinor];
			prevpos = shared_pos[izNeighMinor[iprev] - StartMinor];
		}
		else {
			if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
				(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL))
			{
				prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
				prevA = shared_AAdot_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
			}
			else {
				prevpos = p_info_minor[izNeighMinor[iprev]].pos;
				prevA = p_AAdot[izNeighMinor[iprev]];
			};
		};
		if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
			prevpos = Clockwise_d*prevpos;
		};
		if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
			prevpos = Anticlockwise_d*prevpos;
		};

		i = 0;
		if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
		{
			opppos = shared_pos[izNeighMinor[i] - StartMinor];
			oppA = shared_AAdot[izNeighMinor[i] - StartMinor];
		}
		else {
			if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
				(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
			{
				opppos = shared_pos_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
				oppA = shared_AAdot_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
			}
			else {
				opppos = p_info_minor[izNeighMinor[i]].pos;
				oppA = p_AAdot[izNeighMinor[i]];
			};
		};
		if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
			opppos = Clockwise_d*opppos;
		}
		if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
			opppos = Anticlockwise_d*opppos;
		}

			
#pragma unroll 
		for (i = 0; i < 6; i++)
		{
			inext = i + 1; if (inext > 5) inext = 0;
			if ((izNeighMinor[inext] >= StartMinor) && (izNeighMinor[inext] < EndMinor))
			{
				nextpos = shared_pos[izNeighMinor[inext] - StartMinor];
				nextA = shared_AAdot[izNeighMinor[inext] - StartMinor];
			}
			else {
				if ((izNeighMinor[inext] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[inext] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					nextpos = shared_pos_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
					nextA = shared_AAdot_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					nextpos = p_info_minor[izNeighMinor[inext]].pos;
					nextA = p_AAdot[izNeighMinor[inext]];
				};
			};
			if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
				nextpos = Clockwise_d*nextpos;
			}
			if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
				nextpos = Anticlockwise_d*nextpos;
			}

			// New definition of endpoint of minor edge:
			f64_vec2 endpt0, endpt1, edge_normal;

			endpt0 = THIRD * (prevpos + info.pos + opppos);
			endpt1 = THIRD * (nextpos + info.pos + opppos);

			edge_normal.x = endpt1.y - endpt0.y;
			edge_normal.y = endpt0.x - endpt1.x;

			f64 Az_edge = SIXTH * (2.0*ourA.Az + 2.0*oppA.Az + prevA.Az + nextA.Az);
			Integ_grad_Az += Az_edge*edge_normal;
			f64 Azdot_edge = SIXTH * (2.0*ourA.Azdot + 2.0*oppA.Azdot + prevA.Azdot + nextA.Azdot);
			Integ_grad_Azdot += Azdot_edge*edge_normal;

			AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;
			
			endpt0 = endpt1;
			prevA = oppA;
			prevpos = opppos;
			oppA = nextA;
			opppos = nextpos;				
		};

		f64_vec2 Grad_Az = Integ_grad_Az / AreaMinor;
		f64_vec2 Grad_Azdot = Integ_grad_Azdot / AreaMinor;

		AAdot AAdot_dest;

		f64_vec2 v_overall = p_v_overall_minor[iMinor];

		AAdot_dest.Az = ourA.Az - h_use*Grad_Az.dot(v_overall);
		AAdot_dest.Azdot = ourA.Azdot - h_use*Grad_Azdot.dot(v_overall);

		p_AAdot_dest[iMinor] = AAdot_dest;

	} else {
		p_AAdot_dest[iMinor] = shared_AAdot[threadIdx.x]; // no move
	};
	

	// DEBUGGING

}


