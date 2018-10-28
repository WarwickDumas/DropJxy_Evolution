
#include "kernel.h"
#include "vector_tensor.cu"
#include "helpers.cu"

__global__ void kernelCalculateOverallVelocitiesVertices(
	v4 * __restrict__ p_vie_major,
	f64_vec3 * __restrict__ p_v_n_major,
	nvals * __restrict__ p_n_major,
	f64_vec2 * __restrict__ p_v_overall_major	)
{
	long const index = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX
	v4 const vie = p_vie_major[index];
	f64_vec3 const v_n = p_v_n_major[index];
	nvals const n = p_n_major[index];
	f64_vec2 v_overall;

	v_overall = (vie.vxy*(m_e + m_i)*n.n +
		v_n.xypart()*m_n*n.n_n) /
		((m_e + m_i)*n.n + m_n*n.n_n);

	p_v_overall_major[index] = v_overall;
}

__global__ void kernelAverageOverallVelocitiesTriangles(
	f64_vec2 * __restrict__ p_overall_v_major,
	f64_vec2 * __restrict__ p_overall_v_minor,
	structural * __restrict__ p_info,
	LONG3 * __restrict__ p_tri_corner_index,
	CHAR3 * __restrict__ p_tri_periodic_corner_flags
)
{
	__shared__ f64_vec2 shared_v[threadsPerTileMajor];

	long const index = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX
	if (threadIdx.x < threadsPerTileMajor)
	{
		long getindex = blockIdx.x * threadsPerTileMajor + threadIdx.x;
		shared_v[threadIdx.x] = p_overall_v_major[getindex];
	};
	long const StartMajor = blockIdx.x*threadsPerTileMajor;
	long const EndMajor = StartMajor + threadsPerTileMajor;
	LONG3 const tri_corner_index = p_tri_corner_index[index];
	CHAR3 const tri_corner_per_flag = p_tri_periodic_corner_flags[index];
	structural info = p_info[index];

	__syncthreads();
	
	if (info.flag == DOMAIN_TRIANGLE) {

		f64_vec2 v(0.0, 0.0);
		f64_vec2 vcorner;
		if ((tri_corner_index.i1 >= StartMajor) && (tri_corner_index.i1 < EndMajor))
		{
			vcorner = THIRD*shared_v[tri_corner_index.i1 - StartMajor];
		}
		else {
			vcorner = THIRD*p_overall_v_major[tri_corner_index.i1];
		};
		if (tri_corner_per_flag.c1 == ROTATE_ME_CLOCKWISE) vcorner = Clockwise*vcorner;
		if (tri_corner_per_flag.c1 == ROTATE_ME_ANTICLOCKWISE) vcorner = Anticlockwise*vcorner;
		v += vcorner;

		if ((tri_corner_index.i2 >= StartMajor) && (tri_corner_index.i2 < EndMajor))
		{
			vcorner = THIRD*shared_v[tri_corner_index.i2 - StartMajor];
		}
		else {
			vcorner = THIRD*p_overall_v_major[tri_corner_index.i2];
		};
		if (tri_corner_per_flag.c2 == ROTATE_ME_CLOCKWISE) vcorner = Clockwise*vcorner;
		if (tri_corner_per_flag.c2 == ROTATE_ME_ANTICLOCKWISE) vcorner = Anticlockwise*vcorner;
		v += vcorner;

		if ((tri_corner_index.i3 >= StartMajor) && (tri_corner_index.i3 < EndMajor))
		{
			vcorner = THIRD*shared_v[tri_corner_index.i3 - StartMajor];
		}
		else {
			vcorner = THIRD*p_overall_v_major[tri_corner_index.i3];
		};
		if (tri_corner_per_flag.c3 == ROTATE_ME_CLOCKWISE) vcorner = Clockwise*vcorner;
		if (tri_corner_per_flag.c3 == ROTATE_ME_ANTICLOCKWISE) vcorner = Anticlockwise*vcorner;
		v += vcorner;
	} else {
		
		// What else?

	}
	p_overall_v_minor[index] = v;
}


__global__ void kernelAdvectPositions_CopyTris(
	f64 h_use,
	structural * __restrict__ p_info_src,
	structural * __restrict__ p_info_dest,
	f64_vec2 * __restrict__ p_overall_v
	// what else is copied?
	// something we can easily copy over
	// with cudaMemcpy, even ahead of steps?
	// Is there a reason we cannot put into the above routine
	// with a split for "this is a vertex->just use its overall v"
	)
{
	long const index = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX
	structural info = p_info_src[index];
	f64_vec2 overall_v = p_overall_v[index];
	info.pos += h_use*overall_v;
	p_info_dest[index] = info;
}

__global__ void kernelAverage_n_T_x_to_tris(
	nvals * __restrict__ p_n_minor,
	nvals * __restrict__ p_n_major,
	T3 * __restrict__ p_T_minor,
	structural * __restrict__ p_info
	)
{
	__shared__ nvals shared_n[threadsPerTileMajor];
	__shared__ T3 shared_T[threadsPerTileMajor];
	__shared__ f64_vec2 shared_pos[threadsPerTileMajor];

	long const index = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX
	if (threadIdx.x < threadsPerTileMajor)
	{
		long getindex = blockIdx.x * threadsPerTileMajor + threadIdx.x;
		shared_n[threadIdx.x] = p_n_major[getindex];
		shared_T[threadIdx.x] = p_T_minor[BEGINNING_OF_CENTRAL + getindex];
		shared_pos[threadIdx.x] = p_info[BEGINNING_OF_CENTRAL + getindex].pos;
	};
	long const StartMajor = blockIdx.x*threadsPerTileMajor; // vertex index
	long const EndMajor = StartMajor + threadsPerTileMajor;
	LONG3 const tri_corner_index = p_tri_corner_index[index];
	CHAR3 const tri_corner_per_flag = p_tri_periodic_corner_flags[index];
	structural info = p_info[index];

	__syncthreads();

	if (info.flag == DOMAIN_TRIANGLE) {

		T3 T(0.0, 0.0, 0.0);
		nvals n(0.0, 0.0);
		f64_vec2 pos(0.0, 0.0); 

		f64_vec2 poscorner;
		if ((tri_corner_index.i1 >= StartMajor) && (tri_corner_index.i1 < EndMajor))
		{
			poscorner = THIRD*shared_pos[tri_corner_index.i1 - StartMajor];
			n += THIRD*shared_n[tri_corner_index.i1 - StartMajor];
			T += THIRD*shared_T[tri_corner_index.i1 - StartMajor];
		}
		else {
			poscorner = THIRD*p_info[tri_corner_index.i1 + BEGINNING_OF_CENTRAL].pos;
			n += THIRD*p_n_major[tri_corner_index.i1];
			T += THIRD*p_T[tri_corner_index.i1 + BEGINNING_OF_CENTRAL];
		};
		if (tri_corner_per_flag.c1 == ROTATE_ME_CLOCKWISE) poscorner = Clockwise*poscorner;
		if (tri_corner_per_flag.c1 == ROTATE_ME_ANTICLOCKWISE) poscorner = Anticlockwise*poscorner;
		pos += poscorner;

		if ((tri_corner_index.i2 >= StartMajor) && (tri_corner_index.i2 < EndMajor))
		{
			poscorner = THIRD*shared_pos[tri_corner_index.i2 - StartMajor];
			n += THIRD*shared_n[tri_corner_index.i2 - StartMajor];
			T += THIRD*shared_T[tri_corner_index.i2 - StartMajor];
		}
		else {
			poscorner = THIRD*p_info[tri_corner_index.i2 + BEGINNING_OF_CENTRAL].pos;
			n += THIRD*p_n_major[tri_corner_index.i2];
			T += THIRD*p_T[tri_corner_index.i2 + BEGINNING_OF_CENTRAL];
		};
		if (tri_corner_per_flag.c2 == ROTATE_ME_CLOCKWISE) poscorner = Clockwise*poscorner;
		if (tri_corner_per_flag.c2 == ROTATE_ME_ANTICLOCKWISE) poscorner = Anticlockwise*poscorner;
		pos += poscorner;

		if ((tri_corner_index.i3 >= StartMajor) && (tri_corner_index.i3 < EndMajor))
		{
			poscorner = THIRD*shared_pos[tri_corner_index.i3 - StartMajor];
			n += THIRD*shared_n[tri_corner_index.i3 - StartMajor];
			T += THIRD*shared_T[tri_corner_index.i3 - StartMajor];
		}
		else {
			poscorner = THIRD*p_info[tri_corner_index.i3 + BEGINNING_OF_CENTRAL].pos;
			n += THIRD*p_n_major[tri_corner_index.i3];
			T += THIRD*p_T[tri_corner_index.i3 + BEGINNING_OF_CENTRAL];
		};
		if (tri_corner_per_flag.c3 == ROTATE_ME_CLOCKWISE) poscorner = Clockwise*poscorner;
		if (tri_corner_per_flag.c3 == ROTATE_ME_ANTICLOCKWISE) poscorner = Anticlockwise*poscorner;
		pos += poscorner;

	} else {
		// What else?



	}
	
	p_n_minor[index] = n;
	p_T[index] = T;
	info.pos = pos;
	p_info[index] = info;
}

__global__ void kernelCreateShardModelOfDensities_And_SetMajorArea(
	structural * __restrict__ p_info_minor, 
	nvals * __restrict__ p_n_minor,
	
	long * __restrict__ p_izTri_vert,
	char * __restrict__ p_szPBCtri_vert,
	ShardModel * __restrict__ p_n_shards,
	ShardModel * __restrict__ p_n_n_shards,
	long * __restrict__ Tri_n_lists,
	long * __restrict__ Tri_n_n_lists	,
	f64 * __restrict__ p_AreaMajor)// sets n_shards_n, n_shards, Tri_n_n_lists, Tri_n_lists
{
	// called for major tile

	// Inputs:
	// n, pTri->cent,  izTri,  pTri->periodic, pVertex->pos

	// Outputs:
	// pVertex->AreaCell
	// n_shards[iVertex]
	// Tri_n_n_lists[izTri[i]][o1 * 2] <--- 0 if not set by domain vertex

	// CALL AVERAGE OF n TO TRIANGLES - WANT QUADRATIC AVERAGE - BEFORE WE BEGIN
	// MUST ALSO POPULATE pVertex->AreaCell with major cell area

	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];
	__shared__ f64 shared_n[threadsPerTileMinor];
	__shared__ f64 shared_n_n[threadsPerTileMinor];

	// Should always put at least 4 doubles in shared. Here 4 doubles/thread.

	ShardModel n_; // to be populated
	int iNeigh, tri_len;
	f64 N_n, N, interpolated_n, interpolated_n_n;
	long i, inext, o1, o2;
	
	//memset(Tri_n_n_lists, 0, sizeof(f64)*NUMTRIANGLES * 6);
	//memset(Tri_n_lists, 0, sizeof(f64)*NUMTRIANGLES * 6);

	// We can afford to stick 6-8 doubles in shared. 8 vars*8 bytes*256 threads = 16*1024.
	{	
		structural info2[2];
		memcpy(info2,info+blockIdx.x*threadsPerTileMinor+2*threadIdx.x, sizeof(structural)*2);
		shared_pos[2*threadIdx.x] = info2[0].pos;
		shared_pos[2*threadIdx.x+1] = info2[1].pos;
		nvals n2[2];
		memcpy(n2,p_n_minor + blockIdx.x*threadsPerTileMinor + 2 * threadIdx.x, sizeof(nvals) * 2);
	}
	long const StartMinor = blockIdx.x*threadsPerTileMinor; // vertex index
	long const EndMinor = StartMinor + threadsPerTileMinor;

	__syncthreads();

	long izTri[MAXNEIGH];
	char szPBC[MAXNEIGH];
	f64 coeff[MAXNEIGH];   // total 21*12 = 252 bytes. 256 max for 192 threads.

   // To fit in Tri_n_n_lists stuff we should first let coeff[] go out of scope.
	long const iVertex = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX

	structural info = p_info_minor[BEGINNING_OF_CENTRAL+iVertex];
	if (info.flag == DOMAIN_VERTEX) {

		memcpy(izTri, p_izTri_vert + MAXNEIGH_d*iVertex, sizeof(long)*MAXNEIGH_d);

		f64 n_avg = p_n_minor[BEGINNING_OF_CENTRAL + iVertex];

		if ((izTri[0] >= StartMinor) && (izTri[0] < EndMinor)) {
			pos0 = shared_pos[izTri[0] - StartMinor];
			ndesire0 = shared_n[izTri[0] - StartMinor];
		}
		else {
			pos0 = p_info_minor[izTri[0]].pos;
			ndesire0 = p_n_minor[izTri[0]];
		}
		if (szPBC[0] == ROTATE_ME_CLOCKWISE) pos0 = Clockwise*pos0;
		if (szPBC[0] == ROTATE_ME_ANTICLOCKWISE) pos0 = Anticlockwise*pos0;
		
		f64 tri_area;
		f64 N0 = 0.0; f64 coeffcent = 0.0;
		memset(coeff, 0, sizeof(f64)*MAXNEIGH_d);
		short i;
		f64 AreaMajor = 0.0;
		f64 high_n = ndesire0;
		f64 low_n = ndesire0;
#pragma unroll MAXNEIGH
		for (i = 0; i < info.tri_len; i++)
		{
			// Temporary setting:
			n_.n[i] = ndesire0;

			inext = i + 1; if (inext == info.tri_len) inext = 0;
			if (izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor)) {
			pos1 = shared_pos[izTri[inext] - StartMinor];
			ndesire1 = shared_n[izTri[inext] - StartMinor];
			}
			else {
				pos1 = p_info_minor[izTri[inext]].pos;
				ndesire1 = p_n_minor[izTri[inext]];
			};
			if (szPBC[inext] == ROTATE_ME_CLOCKWISE) pos1 = Clockwise*pos1;
			if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) pos1 = Anticlockwise*pos1;

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

		if ((n_avg > high_n) || (n_avg < low_n)){
#pragma unroll MAXNEIGH
			for (i = 0; i < info.tri_len; i++)
				n_.n[i] = n_avg;
			n_.n_cent = n_avg;

		} else {
			real n_C_need = (n_avg*AreaMajor - N0) / coeffcent;

			if ((n_C_need > low_n) && (n_C_need < high_n)) {
				n_.n_cent = n_C_need;

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
					n_acceptable = (N - coeffcent*n_C) / (AreaMajor - THIRD*AreaMajor);
					// area-THIRD*area = sum of other coeffs, and of course
					// coeffcent = THIRD*area
					// n_acceptable > N/area since N=area*n_avg > area*low_n.

					// We accept things that are less than this 'max average', and
					// let that increase the threshold; go again until
					// the time we do not find any new lower items ;				
					do {
						bool found = 0;
						f64 coeffremain = 0.0;
						f64 N_attained = coeffcent*low_n;
						for (i = 0; i < info.neigh_len; i++)
						{
							if (fixed[i] == 0) {
								// Go collect ndesire[i]:
								
								f64 ndesire;
								if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
								{
									ndesire = shared_n[izTri[i] - StartMinor];
								} else {
									ndesire = p_n_minor[izTri[i]].n;
								};
								if (ndesire < n_acceptable) { // yes, use ndesire[i] ...
									fixed[i] = true;
									n_.n[i] = ndesire;
									N_attained += n_.n[i] * coeff[i];
									found = true;
								} else {
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

				} else {
					n_C = high_n;
					n_acceptable = (N - coeffcent*n_C) / (AreaMajor - THIRD*AreaMajor);
					do {
						bool found = 0;
						f64 coeffremain = 0.0;
						f64 N_attained = coeffcent*high_n;
						for (i = 0; i < info.neigh_len; i++)
						{
							if (fixed[i] == 0) {
								// Go collect ndesire[i]:

								f64 ndesire;
								if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
								{
									ndesire = shared_n[izTri[i] - StartMinor];
								}
								else {
									ndesire = p_n_minor[izTri[i]].n;
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

		memcpy(p_n_shards[iVertex], &n_, sizeof(ShardModel));
		
	} else { // DOMAIN_VERTEX
		memset(p_n_shards[iVertex], 0, sizeof(ShardModel));
	};






	// NexT: neutral version
	// and tri_n_lists.







	__syncthreads(); // block only

	// What comes next?



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
		};
	}
	else {

		// NOT A DOMAIN VERTEX 
		memset(&(n_shards_n[iVertex]), 0, sizeof(ShardModel));
		memset(&(n_shards[iVertex]), 0, sizeof(ShardModel));
	}
}



__global__ void kernelCalculateNu_eHeartNu_iHeart_nu_nn_visc(
	structural * __restrict__ p_info,
	n2 * __restrict__ p_n,
	T3 * __restrict__ p_T,
	species3 * __restrict__ p_nu 
) {
	// Save nu_iHeart, nu_eHeart, nu_nn_visc.

	species3 nu;
	f64 TeV, sigma_MT, sigma_visc, sqrt_T, nu_en_visc;
	T3 T;
	f64 nu_in_visc, nu_ni_visc, nu_ii;
	n2 our_n;
	long const index = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX
	structural info = p_info[index];
	if (info.flag == DOMAIN_VERTEX) {

		our_n = p_n[index]; // never used again once we have kappa
		T = p_T[index];

		TeV = T.Te * one_over_kB;
		Estimate_Ion_Neutral_Cross_sections_d(TeV, &sigma_MT, &sigma_visc);
		sqrt_T = sqrt(T.Te);
		nu_en_visc = our_n.n_n * sigma_visc * sqrt_T * over_sqrt_m_e;
		nu_eiBar = nu_eiBarconst * kB_to_3halves * our_n.n *
			Get_lnLambda_d(our_n.n, T.Te) / (T.Te*sqrt_T);
		//nu_eHeart:
		nu.e = nu_en_visc + 1.87*nu_eiBar;

		TeV = T.Ti*one_over_kB;
		Estimate_Ion_Neutral_Cross_sections_d(TeV, &sigma_MT, &sigma_visc); // could easily save one call
		sqrt_T = sqrt(T.Ti); // again not that hard to save one call
		nu_in_visc = our_n.n_n * sigma_visc * sqrt(T.Ti / m_ion + T.Tn / m_n);
		nu_ni_visc = nu_in_visc * (our_n.n / our_n.n_n);
		//nu_nn_visc:
		nu.n = our_n.n_n * Estimate_Neutral_Neutral_Viscosity_Cross_section(T.Tn * one_over_kB)
			* sqrt(T.Tn / m_n);
		nu_ii = our_n.n*kB_to_3halves*Get_lnLambda_ion(our_n.n, T.Ti) /
			(2.07e7*SQRT2*sqrt_T*T.Ti);
		// nu_iHeart:
		nu.i = 0.75*nu_in_visc + 0.8*nu_ii - 0.25*(nu_in_visc*nu_ni_visc) / (3.0*nu_ni_visc + nu.n);

		//shared_n_over_nu[threadIdx.x].e = our_n.n / nu.e;
	//	shared_n_over_nu[threadIdx.x].i = our_n.n / nu.i;
	//	shared_n_over_nu[threadIdx.x].n = our_n.n_n / nu.n;
	} else {
		memset(&nu, 0, sizeof(species3));
	}

	p_nu[index] = nu;
}


__global__ void kernelAccumulateDiffusiveHeatRateAndCalcIonisation(
	f64 const h_use,
	structural * __restrict__ p_info_sharing,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,

	n2 * __restrict__ p_n_major,
	T3 * __restrict__ p_T_major,
	f64_vec3 * __restrict__ p_B_major,
	species3 * __restrict__ p_nu_major,

	NTrates * __restrict__ NTadditionrate)
{
	// Inputs:
	// We work from major values of T,n,B
	// Outputs:

	// Aim 16 doubles in shared.
	// 12 long indices counts for 6.

	__shared__ f64_vec2 shared_pos[threadsPerTileMajor]; // 2
	__shared__ T3 shared_T[threadsPerTileMajor];      // +3
	__shared__ species3 shared_n_over_nu[threadsPerTileMajor];   // +3
	// saves a lot of work to compute the relevant nu once for each vertex not 6 or 12 times.
	__shared__ f64_vec2 shared_B[threadsPerTileMajor]; // +2
	// B is smooth. Unfortunately we have not fitted in Bz here.
	// In order to do that perhaps rewrite so that variables are overwritten in shared.
	// We do not need all T and nu in shared at the same time.
	// This way is easier for NOW.
	__shared__ f64 shared_nu_iHeart[threadsPerTileMajor];
	__shared__ f64 shared_nu_eHeart[threadsPerTileMajor];
	
	// Balance of shared vs L1: 16 doubles vs 5 doubles per thread at 384 threads/SM.
	__shared__ long Indexneigh[MAXNEIGH_d*threadsPerTileMajor]; // assume 48 bytes
	char PBCneigh[MAXNEIGH_d*threadsPerTileMajor]; // 12 bytes each from L1. Have 42 per thread at 384 threads.
	  // Note that limiting to 16 doubles actually allows 384 threads in 48K. 128K/(384*8) = 42 f64 registers/thread.
	// We managed this way: 2+3+3+2+2+6+1.5 [well, 12 bytes really] = 19.5
	// 48K/(18*8) = 341 threads. Aim 320 = 2x180? Sadly not room for 384.
	// But nothing to stop making a "clever major block" of 320=256+64, or of 160.
	// We should at any rate try a major block of size 256. If 1 block will run at a time, so be it.

	// Leave L1 in case of register overflow into it. <-- don't know how likely - do we only have 31 doubles in registry
	// regardless # of threads and space? Or can be 63?

	// Remains to be seen if this is best strategy, just having a go.
	
	long const StartMajor = blockIdx.x*blockDim.x;
	long const EndMajor = StartMajor + blockDim.x;
	long const index = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX

	T3 our_T; // know own. Can remove & use shared value if register pressure too great?
	
					 // 1. Load T and n
					 // 2. Create kappa in shared & load B --> syncthreads
					 // 3. Create grad T and create flows
					 // For now without any overwriting, we can do all in 1 pass through neighbours
					 // 4. Ionisation too!
	
	structural info = p_info_sharing[index];
	shared_pos[threadIdx.x] = info.pos;
	species3 our_nu;
	n2 our_n;

	
	if (info.flag == DOMAIN_VERTEX) {
						
		our_n = p_n_major[index]; // never used again once we have kappa
		our_nu = p_nu_major[index];
		our_T = p_T_major[index]; // CAREFUL: Pass vertex array if we use vertex index
		shared_n_over_nu[threadIdx.x].e = our_n.n/our_nu.e;
		shared_n_over_nu[threadIdx.x].i = our_n.n / our_nu.i;
		shared_n_over_nu[threadIdx.x].n = our_n.n_n / our_nu.n;
		shared_nu_iHeart[threadIdx.x] = our_nu.i;
		shared_nu_eHeart[threadIdx.x] = our_nu.e;
		shared_B[threadIdx.x] = p_B[index].xypart();
		shared_T[threadIdx.x] = our_T;

	} else {
			// SHOULD NOT BE LOOKING INTO INS.
			// Is OUTERMOST another thing that comes to this branch? What about it? Should we also rule out traffic?
    
			// How do we avoid?
		memset(shared_B[threadIdx.x], 0, sizeof(f64_vec2));
		memset(shared_n_over_nu[threadIdx.x], 0, sizeof(species3));
		shared_nu_iHeart = 0.0;
		shared_nu_eHeart = 0.0;
		memset(shared_T[threadIdx.x], 0, sizeof(T3));
			// Almost certainly, we take block that is in domain
			// And it will look into ins.
			// Simple criterion: iVertex < value means within ins
			// and therefore no traffic.
		// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	}
	__syncthreads();
	
	f64 Area_quadrilateral;			// + 1
	f64_vec2 grad_T;				// + 2
	T3 T_anti, T_clock, T_out;		// + 9
	// we do need to be able to populate it from outside block!
	// We so prefer not to have to access 3 times but to store it once we read T*3
	f64_vec2 pos_clock, pos_anti, pos_out; // we do need to be able to populate from outside block!
	//species3 nu_clock, nu_anti, nu_out; // + 6 + 9   same logic, we need to store external
	// avoid storing external of this. We are running out of registers.
	//f64_tens2 kappa;				// + 4     
	f64_vec2 //B_clock, B_anti, 
			 B_out; // + 2  
	f64 AreaMajor = 0.0;
	// 29 doubles right there.
	NTrates ourrates;   // 5 more ---> 34
	f64 kappa_parallel_e, kappa_parallel_i, kappa_neut;

	// Need this, we are adding on to existing d/dt N,NT :
	memcpy(&ourrates, NTadditionrates + index, sizeof(NTrates));
	
	if ((info.flag == INNERMOST) || (info.flag == OUTERMOST))
	{
		Area = 1.0; 
		// [ Ignore flux into edge of outermost vertex I guess ???]
	} else {
		if (info.flag == DOMAIN_VERTEX) {
			// The idea of not sending blocks full of non-domain vertices is another idea. Fiddly with indices.

			memcpy(Indexneigh + MAXNEIGH_d*threadIdx.x,
				pIndexNeigh + MAXNEIGH_d*index,
				MAXNEIGH_d * sizeof(long));
			memcpy(PBCneigh,
				pPBCNeigh + MAXNEIGH_d*index,
				MAXNEIGH_d * sizeof(char));

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_clock = shared_pos[indexneigh - StartMajor];
				T_clock = shared_T[indexneigh - StartMajor];
				//	B_clock = shared_B[indexneigh - StartMajor];
			}
			else {
				structural info2 = p_info_sharing[indexneigh];
				pos_clock = info2.pos;
				T_clock = p_T_major[indexneigh];
				//	B_clock = p_B[indexneigh];
					// reconstruct nu_clock:
					//n2 n_clock = p_n[indexneigh];
					// could we save something by using just opposing points instead of 5/12 for nu?
			};

			char PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
			if (PBC == NEEDS_ANTI) {
				pos_clock = Anticlock_rotate2(pos_clock);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_clock = Clockwise_rotate2(pos_clock);
			};

			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_out = shared_pos[indexneigh - StartMajor];
				T_out = shared_T[indexneigh - StartMajor];
			}
			else {
				structural info2 = p_info_sharing[indexneigh];
				pos_out = info2.pos;
				T_out = p_T_major[indexneigh];
			};
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
			if (PBC == NEEDS_ANTI) {
				pos_out = Anticlock_rotate2(pos_out);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_out = Clockwise_rotate2(pos_out);
			};

			short iNeigh;
#pragma unroll MAXNEIGH_d
			for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
			{
				int inext = iNeigh + 1; if (inext == info.neigh_len) inext = 0;
				indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + inext];
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					pos_anti = shared_pos[indexneigh - StartMajor];
					T_anti = shared_T[indexneigh - StartMajor];
					//		B_anti = shared_B[indexneigh - StartMajor];
				}
				else {
					structural info2 = p_info_sharing[indexneigh];
					pos_out = info2.pos;
					T_anti = p_T_major[indexneigh];
					//		B_anti = p_B[indexneigh];
				};
				PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + inext];
				if (PBC == NEEDS_ANTI) {
					pos_anti = Anticlock_rotate2(pos_anti);
					//		B_anti = Anticlock_rotate2(B_anti);
				};
				if (PBC == NEEDS_CLOCK) {
					pos_anti = Clockwise_rotate2(pos_anti);
					//		B_anti = Clockwise_rotate2(B_anti);
				};
				// Do we even really need to be doing with B_anti? Why not just
				// take just once the nu and B from opposite and take 0.5 average with self.
				// It will not make a huge difference to anything.
			
				f64_vec2 edgenormal;
				edgenormal.x = THIRD*(pos_anti.y - pos_clock.y);
				edgenormal.y = THIRD*(pos_clock.x - pos_anti.x);

				AreaMajor += 0.5*edge_normal.x*THIRD*(pos_anti.x + pos_clock.x
					+ info.pos.x + info.pos.x + pos_out.x + pos_out.x);
				//tridata1.pos.x + tridata2.pos.x);

				Area_quadrilateral = 0.5*(
					(info.pos.x + pos_anti.x)*(info.pos.y - pos_anti.y)
					+ (pos_clock.x + info.pos.x)*(pos_clock.y - info.pos.y)
					+ (pos_out.x + pos_clock.x)*(pos_out.y - pos_clock.y)
					+ (pos_anti.x + pos_out.x)*(pos_anti.y - pos_out.y)
					);
				// Te first:
				grad_T.x = 0.5*(
					(our_T.Te + T_anti.Te)*(info.pos.y - pos_anti.y)
					+ (T_clock.Te + our_T.Te)*(pos_clock.y - info.pos.y)
					+ (T_out.Te + T_clock.Te)*(pos_out.y - pos_clock.y)
					+ (T_anti.Te + T_out.Te)*(pos_anti.y - pos_out.y)
					) / Area_quadrilateral;
				grad_T.y = -0.5*( // notice minus
					(our_T.Te + T_anti.Te)*(info.pos.x - pos_anti.x)
					+ (T_clock.Te + our_T.Te)*(pos_clock.x - info.pos.x)
					+ (T_out.Te + T_clock.Te)*(pos_out.x - pos_clock.x)
					+ (T_anti.Te + T_out.Te)*(pos_anti.x - pos_out.x)
					) / Area_quadrilateral;

				//kappa.xx = kappa_parallel * (nu_eHeart*nu_eHeart + omega.x*omega.x) / (nu_eHeart * nu_eHeart + omega_sq);
				//kappa.xy = kappa_parallel * (omega.x*omega.y - nu_eHeart *omega.z) / (nu_eHeart * nu_eHeart + omega_sq);
				//kappa.yx = kappa_parallel * (omega.x*omega.y + nu_eHeart * omega.z) / (nu_eHeart * nu_eHeart + omega_sq);
				//kappa.yy = kappa_parallel * (omega.y*omega.y + nu_eHeart * nu_eHeart) / (nu_eHeart * nu_eHeart + omega_sq);

				indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
				if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
				{
					B_out = shared_B[indexneigh - StartMajor];

					kappa_parallel_e = // 2.5 nT/(m nu)
						2.5*0.5*(shared_n_over_nu[indexneigh - StartMajor].e
							+ shared_n_over_nu[threadIdx.x].e)
						*(0.5*(T_out.Te + our_T.Te)) * over_m_e;
					kappa_parallel_i = 
						KAPPA_ION_FACTOR * 0.5*(shared_n_over_nu[indexneigh - StartMajor].i
							+ shared_n_over_nu[threadIdx.x].i)
							*(0.5*(T_out.Ti + our_T.Ti)) * over_m_i;
					kappa_neut = KAPPA_NEUT_FACTOR  * 0.5*(shared_n_over_nu[indexneigh - StartMajor].n
										+ shared_n_over_nu[threadIdx.x].n)
										*(0.5*(T_out.Tn + our_T.Tn)) * over_m_n;
					// If we don't carry kappa_ion we are carrying shared_n_over_nu because
					// we must load that only once for the exterior neighs. So might as well carry kappa_ion.
					nu_eHeart = 0.5*(our_nu.e + shared_nu_eHeart[indexneigh - StartMajor]); 
					nu_iHeart = 0.5*(our_nu.i + shared_nu_iHeart[indexneigh - StartMajor]);					
				} else {
					n_out = p_n_major[indexneigh];
					B_out = p_B_major[indexneigh];
					T_out = p_T_major[indexneigh];  // reason to combine n,T . How often do we load only 1 of them?
					// Calculate n/nu out there:
					species3 nu_out = p_nu_major[indexneigh];

					kappa_parallel_e =
						2.5*0.5*(n_out.n / nu_out.e + shared_n_over_nu[threadIdx.x].e)
						*(0.5*(T_out.Te + our_T.Te))* over_m_e;
					kappa_parallel_i =
						KAPPA_ION_FACTOR * 0.5*(n_out.n/nu_out.i + shared_n_over_nu[threadIdx.x].i)
						*0.5*(T_out.Ti + our_T.Ti)*over_m_i;
					kappa_neut = KAPPA_NEUT_FACTOR * 0.5*(n_out.n_n / nu_out.n + shared_n_over_nu[threadIdx.x].n)
						*0.5*(T_out.Tn + out_T.Tn)*over_m_n;

					nu_eHeart = 0.5*(our_nu.e + nu_out.e);
					nu_iHeart = 0.5*(our_nu.i + nu_out.i);
					// Could we save register pressure by just calculating these 3 nu values
					// first and doing a load?
				};
				PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
				if (PBC == NEEDS_ANTI) 	B_out = Anticlock_rotate2(B_out);
				if (PBC == NEEDS_CLOCK)	B_out = Clockwise_rotate2(B_out);

				omega = Make3(qovermc * 0.5*(shared_B[threadIdx.x] + B_out),BZ*qovermc);

				// if the outward gradient of T is positive, inwardheatflux is positive.
				//kappa_grad_T_dot_edgenormal = 
				ourrates.NeTe += TWOTHIRDS*kappa_parallel_e*(
					edgenormal.x*(
						//kappa.xx*grad_T.x + kappa.xy*grad_T.y
					(nu_eHeart*nu_eHeart + omega.x*omega.x)*grad_T.x +
						(omega.x*omega.y - nu_eHeart *omega.z)*grad_T.y
						)
					+ edgenormal.y*(
						//kappa.yx*grad_T.x + kappa.yy*grad_T.y
					(omega.x*omega.y + nu_eHeart * omega.z)*grad_T.x +
						(omega.y*omega.y + nu_eHeart * nu_eHeart)*grad_T.y
						))
					/ (nu_eHeart * nu_eHeart + omega.dot(omega));
				// ****************************************************************************************
				// Look: nu_eHeart appeared in kappa formula sep from n/nu in kappa_parallel - we need both

				// Ion:

				grad_T.x = 0.5*(
					(shared_T[threadIdx.x].Ti + T_anti.Ti)*(info.pos.y - pos_anti.y)
					+ (T_clock.Ti + shared_T[threadIdx.x].Ti)*(pos_clock.y - info.pos.y)
					+ (T_out.Ti + T_clock.Ti)*(pos_out.y - pos_clock.y)
					+ (T_anti.Ti + T_out.Ti)*(pos_anti.y - pos_out.y)
					) / Area_quadrilateral;
				grad_T.y = -0.5*( // notice minus
					(shared_T[threadIdx.x].Ti + T_anti.Ti)*(info.pos.x - pos_anti.x)
					+ (T_clock.Ti + shared_T[threadIdx.x].Ti)*(pos_clock.x - info.pos.x)
					+ (T_out.Ti + T_clock.Ti)*(pos_out.x - pos_clock.x)
					+ (T_anti.Ti + T_out.Ti)*(pos_anti.x - pos_out.x)
					) / Area_quadrilateral;

				omega = Make3(qoverMc * 0.5*(shared_B[threadIdx.x] + B_out), BZ*qoverMc);
				
				ourrates.NiTi += TWOTHIRDS * kappa_parallel *(
					edge_normal.x*(
					(nu_iHeart*nu_iHeart + omega.x*omega.x)*grad_T.x +
						(omega.x*omega.y + nu_iHeart * omega.z)*grad_T.y
						)
					+ edge_normal.y*(
					(omega.x*omega.y - nu_iHeart * omega.z)*grad_T.x +
						(omega.y*omega.y + nu_iHeart * nu_iHeart)*grad_T.y
						))
					/ (nu_iHeart * nu_iHeart + omega.dot(omega));

				// Neutral:

				grad_T.x = 0.5*(
					(shared_T[threadIdx.x].Tn + T_anti.Tn)*(info.pos.y - pos_anti.y)
					+ (T_clock.Tn + shared_T[threadIdx.x].Tn)*(pos_clock.y - info.pos.y)
					+ (T_out.Tn + T_clock.Tn)*(pos_out.y - pos_clock.y)
					+ (T_anti.Tn + T_out.Tn)*(pos_anti.y - pos_out.y)
					) / Area_quadrilateral;
				grad_T.y = -0.5*( // notice minus
					(shared_T[threadIdx.x].Tn + T_anti.Tn)*(info.pos.x - pos_anti.x)
					+ (T_clock.Tn + shared_T[threadIdx.x].Tn)*(pos_clock.x - info.pos.x)
					+ (T_out.Tn + T_clock.Tn)*(pos_out.x - pos_clock.x)
					+ (T_anti.Tn + T_out.Tn)*(pos_anti.x - pos_out.x)
					) / Area_quadrilateral;

				ourrates.NnTn += TWOTHIRDS * kappa_neut * grad_T.dot(edge_normal);

				// Now go round:		
				pos_clock = pos_out;
				pos_out = pos_anti;
				T_clock = T_out;
				T_out = T_anti;
				iprev = iNeigh;
			};

			// now add IONISATION:

			TeV = T.Te * one_over_kB;
			f64 sqrtT = sqrt(TeV);
			f64 temp = 1.0e-5*exp(-13.6 / TeV) / (13.6*(6.0*13.6 + TeV));
			// Let h n n_n S be the ionising amount,
			// h n S is the proportion of neutrals! Make sure we do not run out!
			f64 hnS = (h_use*our_n.n*TeV*temp) /
				(sqrtT + h_use * our_n.n_n*our_n.n*temp*SIXTH*13.6);
			f64 ionise_rate = AreaMajor * our_n.n_n*hnS / (h_use*(1 + hnS));
			// ionise_amt / h

			ourrates.N += ionise_rate;
			ourrates.Nn += -ionise_rate;

			// Let nR be the recombining amount, R is the proportion.

			f64 Ttothe5point5 = sqrtT * TeV * TeV*TeV * TeV*TeV;
			f64 hR = h_use * (our_n.n * our_n.n*8.75e-27*TeV) /
				(Ttothe5point5 + h_use * 2.25*TWOTHIRDS*13.6*our_n.n*our_n.n*8.75e-27);

			f64 recomb_rate = AreaMajor * our_n.n * hR / h_use; // could reasonably again take hR/(1+hR) for n_k+1
			ourrates.N -= recomb_rate;
			ourrates.Nn += recomb_rate;

			ourrates.NeTe += -TWOTHIRDS * 13.6*kB*ourrates.N + 0.5*T.Tn*ionise_rate;
			ourrates.NiTi += 0.5*T.Tn*ionise_rate;
			ourrates.NnTn += (T.Te + T.Ti)*recomb_rate;

			memcpy(NTadditionrates + index, &ourrates, sizeof(NTrates));

		} else {
			// Not DOMAIN_VERTEX or INNERMOST or OUTERMOST
			
			// [ Ignore flux into edge of outermost vertex I guess ???]

		};
	};
}


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
	f64 * __restrict__ Integrated_Div_v_overall;
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

	long const index = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX
	structural info = p_info_major[index];
	
	if (info.flag == DOMAIN_VERTEX) {

		n_src_or_use[threadIdx.x] = p_n_major[index];  // used throughout so a good candidate to stick in shared mem
		AreaMajor[threadIdx.x] = p_AreaMajor[index]; // ditto

		NTrates newdata;
		{
			NTrates AdditionNT = NTadditionrates[index];
			newdata.N = n_src_or_use[threadIdx.x].n*AreaMajor[threadIdx.x] + h_use * AdditionNT.N;
			newdata.Nn = n_src_or_use[threadIdx.x].n_n*AreaMajor[threadIdx.x] + h_use * AdditionNT.Nn;
			newdata.NnTn = h_use * AdditionNT.NnTn; // start off without knowing 'factor' so we can ditch AdditionNT
			newdata.NiTi = h_use * AdditionNT.NiTi;
			newdata.NeTe = h_use * AdditionNT.NeTe;
		}

		{
			nvals n_dest; 
			f64 Div_v_overall_integrated = Integrated_Div_v_overall[index];
			n_dest.n = newdata.N / (AreaMajor[threadIdx.x] + h_use*Div_v_overall_integrated); // Do have to worry whether advection steps are too frequent.
			n_dest.n_n = newdata.Nn / (AreaMajor[threadIdx.x] + h_use*Div_v_overall_integrated); // What could do differently: know ROC area as well as mass flux through walls
			p_n_major_dest[index] = n_dest;
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
			f64 Div_v = p_div_v[index];
			f64 Div_v_n = p_div_v_neut[index];
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
			T3 T_src = p_T_major[index];
			newdata.NnTn += n_src_or_use[threadIdx.x].n_n*AreaMajor[threadIdx.x]*T_src.Tn*factor_neut;
			newdata.NiTi += n_src_or_use[threadIdx.x].n*AreaMajor[threadIdx.x]*T_src.Ti*factor;
			newdata.NeTe += n_src_or_use[threadIdx.x].n*AreaMajor[threadIdx.x]*T_src.Te*factor;  // 
		}

		f64 nu_ne_MT, nu_en_MT, nu_ni_MT, nu_in_MT, nu_ei; // optimize after
		{
			// Dimensioning inside a brace allows the following vars to go out of scope at the end of the brace.
			f64 sqrt_Te, ionneut_thermal, electron_thermal, lnLambda, s_in_MT, s_en_MT, s_en_visc;

			n_src_or_use[threadIdx.x] = p_n_use[index];
			T3 T_use = p_T_use[index];

			sqrt_Te = sqrt(T_use.Te); // should be "usedata"
			ionneut_thermal = sqrt(T_use.Ti / m_ion + T_use.Tn / m_n); // hopefully not sqrt(0)
			electron_thermal = sqrt_Te * over_sqrt_m_e;
			lnLambda = Get_lnLambda_d(n_use.n, T_use.Te);

			s_in_MT = Estimate_Ion_Neutral_MT_Cross_section_d(T_use.Ti*one_over_kB);
			Estimate_Ion_Neutral_Cross_sections_d(T_use.Te, // call with T in electronVolts
				&s_en_MT,
				&s_en_visc);
			//s_en_MT = Estimate_Ion_Neutral_MT_Cross_section(T_use.Te*one_over_kB);
			//s_en_visc = Estimate_Ion_Neutral_Viscosity_Cross_section(T_use.Te*one_over_kB);
			// Need nu_ne etc to be defined:
			nu_ne_MT = s_en_MT * n_src_or_use[threadIdx.x].n * electron_thermal; // have to multiply by n_e for nu_ne_MT
			nu_ni_MT = s_in_MT * n_src_or_use[threadIdx.x].n * ionneut_thermal;
			nu_en_MT = s_en_MT * n_src_or_use[threadIdx.x].n_n*electron_thermal;
			nu_in_MT = s_in_MT * n_src_or_use[threadIdx.x].n_n*ionneut_thermal;

			nu_ei = nu_eiBarconst * kB_to_3halves*n_use.n*lnLambda /
				(T_use.Te*sqrt_Te);

	//		nu_ie = nu_ei;

			//	nu_eHeart = 1.87*nu_eiBar + data_k.n_n*s_en_visc*electron_thermal;
		}
		// For now doing velocity-independent resistive heating.
		// Because although we have a magnetic correction Upsilon_zz involved, we ignored it
		// since we are also squashing the effect of velocity-dependent collisions on vx and vy (which
		// would produce a current in the plane) and this squashing should create heat, which
		// maybe means it adds up to velo independent amount of heating. 
		{
			f64_vec3 v_n = p_v_n_use[index];
			v4 vie = p_vie_use[index];

			newdata.NeTe += h_use*(AreaMajor[threadIdx.x]*TWOTHIRDS*nu_en_MT*m_en*(
				(v_n.x - vie.vxy.x)*(v_n.x - vie.vxy.x)
				+ (v_n.y - vie.vxy.y)*(v_n.y - vie.vxy.y)
				+ (v_n.z - vie.vez)*(v_n.z - vie.vez))

				+ AreaMajor[threadIdx.x]*TWOTHIRDS*nu_ei*m_ei*(vie.vez - vie.viz)*(vie.vez - vie.viz));

			newdata.NiTi += h_use*(AreaMajor[threadIdx.x]*TWOTHIRDS*nu_in_MT*M_in*m_n*(
				(v_n.x - vie.vxy.x)*(v_n.x - vie.vxy.x)
				+ (v_n.y - vie.vxy.y)*(v_n.y - vie.vxy.y)
				+ (v_n.z - vie.viz)*(v_n.z - vie.viz)));

			newdata.NnTn += h_use*(AreaMajor[threadIdx.x]*TWOTHIRDS*nu_ni_MT*M_in*m_i*(
				(v_n.x - vie.vxy.x)*(v_n.x - vie.vxy.x)
				+ (v_n.y - vie.vxy.y)*(v_n.y - vie.vxy.y)
				+ (v_n.z - vie.viz)*(v_n.z - vie.viz)));
		}

		f64_tens3 inverted;
		{
			f64_tens3 LHS;
			// x = neutral
			// y = ion
			// z = elec
			// This is for NT
			LHS.xx = 1.0 - h_use * (-M_en * nu_ne_MT - M_in * nu_ni_MT);
			LHS.xy = -h_use * (M_in * nu_in_MT);
			LHS.xz = -h_use *(M_en * nu_en_MT);
			LHS.yx = -h_use *  M_in * nu_ni_MT;
			LHS.yy = 1.0 - h_use * (-M_in * nu_in_MT - M_ei * nu_ie);
			LHS.yz = -h_use * M_ei * nu_ei;
			LHS.zx = -h_use * M_en * nu_ne_MT;
			LHS.zy = -h_use * M_ei * nu_ie;
			LHS.zz = 1.0 - h_use * (-M_en * nu_en_MT - M_ei * nu_ei);

			LHS.Inverse(inverted);
		}

		f64_vec3 RHS;
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

		p_T_major_dest[index] = T_dest;

	} else {
		// nothing to do ??
		if (info.flag == OUTERMOST) {
			p_n_major_dest[index] = p_n_major[index];
			p_T_major_dest[index] = p_T_major[index];
		} else {
			memset(p_n_major_dest + index, 0, sizeof(nvals));
			memset(p_T_major_dest + index, 0, sizeof(T3));
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
	char * __restrict__ p_PBCtriminor, // carries 6, we only need 3
	nvals * __restrict__ p_n_upwind_minor // result
	) 
{
	// The idea is to take the upwind n on each side of each
	// major edge through this tri, weighted by |v.edgenormal|
	// to produce an average.
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor]; // 4 doubles/vertex
	__shared__ f64_12 shared_shards[threadsPerTileMajor];  // + 12
	// 15 doubles right there. Max 21 for 288 vertices. 16 is okay.
	// Might as well stick 1 more double  in there if we get worried about registers.

	// #############################################%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%###############
	// We need a reverse index: this triangle carry 3 indices to know who it is to its corners.
	
	long const iTri = blockDim.x*blockIdx.x + threadIdx.x;
	structural const info = p_info[iTri];
	nvals result;

	shared_pos[threadIdx.x] = info.pos;

	if (threadIdx.x < threadsPerTileMajor)
	{
		shared_shards[threadIdx.x] = p_n_shards_major[threadsPerTileMajor*blockIdx.x + threadIdx.x].n;
		// efficiency vs memcpy? We only need 12 here, not the centre.
	}
	__syncthreads();

	f64 n0, n1, n2;
	f64_vec2 edgenormal0, edgenormal1, edgenormal2;
	LONG3 tricornerindex, trineighindex;
	LONG3 who_am_I;
	f64_vec2 v_overall;

	if (info.flag == DOMAIN_TRIANGLE) // otherwise meaningless
	{
		// Several things we need to collect:
		// . v in this triangle and mesh v at this triangle centre.
		// . edge_normal going each way
		// . n that applies from each corner

		// How to get n that applies from each corner:
		tricornerindex = p_tricornerindex[iTri];
		who_am_I = p_which_iTri_number_am_I[iTri];

		if ((tricornerindex.i1 >= StartMajor) && (tricornerindex.i1 < EndMajor))
		{
			n0 = shared_shards[tricornerindex.i1 - StartMajor][who_am_I.i1];
		} else {
			n0 = p_n_shards_major[tricornerindex.i1].n[who_am_I.i1];
			// at least it's 1 bus journey this way instead of 2 to fetch n_shards.
		}
		if ((tricornerindex.i2 >= StartMajor) && (tricornerindex.i2 < EndMajor))
		{
			n1 = shared_shards[tricornerindex.i2 - StartMajor][who_am_I.i2];
		} else {
			n1 = p_n_shards_major[tricornerindex.i2].n[who_am_I.i2];
			// at least it's 1 bus journey this way instead of 2 to fetch n_shards.
		}
		if ((tricornerindex.i3 >= StartMajor) && (tricornerindex.i3 < EndMajor))
		{
			n2 = shared_shards[tricornerindex.i3 - StartMajor][who_am_I.i3];
		} else {
			n2 = p_n_shards_major[tricornerindex.i3].n[who_am_I.i3];
			// at least it's 1 bus journey this way instead of 2 to fetch n_shards.
		}

		v_overall = p_v_overall_minor[iTri];
		f64_vec2 relv = p_vie_minor[iTri].vxy - v_overall;
		
		trineighindex = p_trineighindex[iTri];
		f64_vec2 nearby_pos;
		if ((trineighindex.i1 >= StartMinor) && (trineighindex.i1 < EndMinor)) {
			nearby_pos = shared_pos[trineighindex.i1 - StartMinor];
		} else {
			nearby_pos = p_info_minor[trineighindex.i1].pos;
		}
		if (sztriPBC_neighs[0] == ROTATE_ME_CLOCKWISE) {
			nearby_pos = Clockwise*nearby_pos;
		} 
		if (sztriPBC_neighs[0] == ROTATE_ME_ANTICLOCKWISE) {
			nearby_pos = Anticlockwise*nearby_pos;
		}
		
		edgenormal0.x = nearby_pos.y - info.pos.y;
		edgenormal0.y = info.pos.x - nearby_pos.x;
		// CAREFUL AS FUCK : which side is which???
		// tri centre 2 is on same side of origin as corner 1 -- I think
		// We don't know if the corners have been numbered anticlockwise?
		// Could arrange it though.
		// So 1 is anticlockwise for edge 0.
		
		f64 numerator = 0.0;
		f64 dot1, dot2;
		f64 dot0 = relv.dot(edgenormal0);
		if (dot0 > 0.0) // v faces anticlockwise
		{
			numerator += dot0*n2;
		} else {
			dot0 = -dot0;
			numerator += dot0*n1;
		}

		if ((trineighindex.i2 >= StartMinor) && (trineighindex.i2 < EndMinor)) {
			nearby_pos = shared_pos[trineighindex.i2 - StartMinor];
		}
		else {
			nearby_pos = p_info_minor[trineighindex.i2].pos;
		}
		if (sztriPBC_neighs[1] == ROTATE_ME_CLOCKWISE) {
			nearby_pos = Clockwise*nearby_pos;
		}
		if (sztriPBC_neighs[1] == ROTATE_ME_ANTICLOCKWISE) {
			nearby_pos = Anticlockwise*nearby_pos;
		}
		edgenormal1.x = nearby_pos.y - info.pos.y;
		edgenormal1.y = info.pos.x - nearby_pos.x;

		dot1 = relv.dot(edgenormal1);
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
		if (sztriPBC_neighs[2] == ROTATE_ME_CLOCKWISE) {
			nearby_pos = Clockwise*nearby_pos;
		}
		if (sztriPBC_neighs[2] == ROTATE_ME_ANTICLOCKWISE) {
			nearby_pos = Anticlockwise*nearby_pos;
		}
		edgenormal2.x = nearby_pos.y - info.pos.y;
		edgenormal2.y = info.pos.x - nearby_pos.x;

		dot2 = relv.dot(edgenormal2);
		if (dot2 > 0.0)
		{
			numerator += dot2*n1;
		} else {
			dot2 = -dot2;
			numerator += dot2*n0;
		}

		result.n = numerator / (dot0 + dot1 + dot2);
		// Argument against fabs in favour of squared weights?
		
	} else {
		if (info.flag == CROSSING_INS) {
			result.n = ;
		} else {
			result.n = 0.0;
		};
	};

	// Now same for upwind neutral density:
	// In order to use syncthreads we had to come out of the branching.
	
	if (threadIdx.x < threadsPerTileMajor)
	{
		shared_shards[threadIdx.x] = p_n_shards_n_major[threadsPerTileMajor*blockIdx.x + threadIdx.x].n;
		// efficiency vs memcpy? We only need 12 here, not the centre.
	}
	__syncthreads();

	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

	if (info.flag == DOMAIN_TRIANGLE) // otherwise meaningless
	{
		if ((tricornerindex.i1 >= StartMajor) && (tricornerindex.i1 < EndMajor))
		{
			n0 = shared_shards[tricornerindex.i1 - StartMajor][who_am_I.i1];
		}
		else {
			n0 = p_n_shards_n_major[tricornerindex.i1].n[who_am_I.i1];
			// at least it's 1 bus journey this way instead of 2 to fetch n_shards.
		}
		if ((tricornerindex.i2 >= StartMajor) && (tricornerindex.i2 < EndMajor))
		{
			n1 = shared_shards[tricornerindex.i2 - StartMajor][who_am_I.i2];
		}
		else {
			n1 = p_n_shards_n_major[tricornerindex.i2].n[who_am_I.i2];
			// at least it's 1 bus journey this way instead of 2 to fetch n_shards.
		}
		if ((tricornerindex.i3 >= StartMajor) && (tricornerindex.i3 < EndMajor))
		{
			n2 = shared_shards[tricornerindex.i3 - StartMajor][who_am_I.i3];
		}
		else {
			n2 = p_n_shards_n_major[tricornerindex.i3].n[who_am_I.i3];
			// at least it's 1 bus journey this way instead of 2 to fetch n_shards.
		}

		f64_vec2 relv = p_v_n_minor[iTri].xypart() - v_overall;
		
		f64 numerator = 0.0;
		f64 dot1, dot2;
		f64 dot0 = relv.dot(edgenormal0);
		if (dot0 > 0.0) // v faces anticlockwise
		{
			numerator += dot0*n2;
		}
		else {
			dot0 = -dot0;
			numerator += dot0*n1;
		}

		dot1 = relv.dot(edgenormal1);
		if (dot1 > 0.0)
		{
			numerator += dot1*n0;
		}
		else {
			dot1 = -dot1;
			numerator += dot1*n2;
		}

		dot2 = relv.dot(edgenormal2);
		if (dot2 > 0.0)
		{
			numerator += dot2*n1;
		}
		else {
			dot2 = -dot2;
			numerator += dot2*n0;
		}

		result.n_n = numerator / (dot0 + dot1 + dot2);
		
	} else {
		if (info.flag == CROSSING_INS) {
			result.n_n = ;
		}
		else {
			result.n_n = 0.0;
		};
	};
	
	p_n_upwind_minor[iTri] = result;

}
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
	)
{
	// Use the upwind density from tris together with v_tri.

	__shared__ f64_vec2 shared_pos[threadsPerTileMinor]; // only reused what, 3 times?
	__shared__ nvals shared_n_upwind[threadsPerTileMinor];
	__shared__ f64_vec2 vxy[threadsPerTileMinor];
	__shared__ f64_vec2 v_n[threadsPerTileMinor];
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
		memcpy(info, p_info_minor + (threadsPerTileMinor*blockDim.x + 2 * threadIdx.x), sizeof(structural) * 2);
		shared_pos[2 * threadIdx.x] = info[0].pos;
		shared_pos[2 * threadIdx.x + 1] = info[1].pos;
		memcpy(&(shared_n_upwind[2*threadIdx.x]), p_n_upwind_minor + (threadsPerTileMinor*blockDim.x + 2 * threadIdx.x), sizeof(nvals) * 2);
		v4 vie[2];
		memcpy(vie,p_vie_minor + (threadsPerTileMinor*blockDim.x + 2 * threadIdx.x), sizeof(v4) * 2);
		shared_vxy[2 * threadIdx.x] = vie[0].vxy;
		shared_vxy[2 * threadIdx.x + 1] = vie[1].vxy;
		f64_vec3 v_n[2];
		memcpy(v_n, p_v_n_minor + (threadsPerTileMinor*blockDim.x + 2 * threadIdx.x), sizeof(f64_vec3) * 2);
		shared_v_n[2 * threadIdx.x] = v_n[0].xypart();
		shared_v_n[2 * threadIdx.x + 1] = v_n[1].xypart();
		memcpy(shared_T[2 * threadIdx.x], p_T_minor + (threadsPerTileMinor*blockDim.x + 2 * threadIdx.x), sizeof(T3) * 2);
	}
	long const StartMinor = threadsPerTileMinor*blockIdx.x;
	long const EndMinor = threadsPerTileMinor + StartMinor;

	__syncthreads();

	// What happens for abutting ins?
	// T defined reasonably at insulator-crossing tri, A defined, v defined reasonably

	structural info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];

	if (info.flag == DOMAIN_VERTEX) {

		T3 Tsrc = p_T_src_major[iVertex];
		nvals nsrc = p_n_src_major[iVertex];
		long izTri[MAXNEIGH];
		memcpy(izTri, p_izTri + iVertex * MAXNEIGH, sizeof(long) * MAXNEIGH);
		short tri_len = info.neigh_len;

		// Now we are assuming what? Neigh 0 is below tri 0, so 0 1 are on neigh 0
		CHECK ASSUMPTION!!;
		//

		f64_vec2 edge_normal;
		f64_vec2 vxy_prev, vxy_next;
		f64_vec2 v_n_prev, v_n_next;
		f64 n_next, n_prev, nn_next, nn_prev;
		f64_vec2 v_overall_prev, v_overall_next;
		f64 Te_next, Te_prev, Ti_next, Ti_prev, Tn_next, Tn_prev;

		short i = 0;
		long iTri = izTri[0];
		f64_vec2 v_overall_prev = p_v_overall_minor[iTri];
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

		}
		else {
			// The volume of random bus accesses means that we would have been better off making a separate
			// neutral routine even though it looks efficient with the shared loading. nvm
			endpt0 = p_info_minor[iTri].pos;
			nvals n_upwind = p_n_upwind_minor[iTri];
			n_prev = n_upwind.n;
			nn_prev = n_upwind.n_n;
			vxy_prev = p_vie_minor[iTri].vxy;
			v_n_prev = p_v_n_minor[iTri].xypart();
			T3 Tuse = p_T_minor[iTri];
			Te_prev = Tuse.Te;
			Ti_prev = Tuse.Ti;
			Tn_prev = Tuse.Tn;
		};
		if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
			endpt0 = Clockwise*endpt0;
			vxy_prev = Clockwise*vxy_prev;
			v_n_prev = Clockwise*v_n_prev;
			v_overall_prev = Clockwise*v_overall_prev;
		};
		if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
			endpt0 = Anticlockwise*endpt0;
			vxy_prev = Anticlockwise*vxy_prev;
			v_n_prev = Anticlockwise*v_n_prev;
			v_overall_prev = Anticlockwise*v_overall_prev;
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
				endpt0 = shared_pos[iTri - StartMinor];
				nvals nvls = shared_n_upwind[iTri - StartMinor];
				n_next = nvls.n;
				nn_next = nvls.n_n;
				vxy_next = shared_vxy[iTri - StartMinor];
				v_n_next = shared_v_n[iTri - StartMinor];
				Te_next = shared_T[iTri - StartMinor].Te;
				Ti_next = shared_T[iTri - StartMinor].Ti;
				Tn_next = shared_T[iTri - StartMinor].Tn;
			}
			else {
				// The volume of random bus accesses means that we would have been better off making a separate
				// neutral routine even though it looks efficient with the shared loading. nvm
				endpt0 = p_info_minor[iTri].pos;
				nvals n_upwind = p_n_upwind_minor[iTri];
				n_next = n_upwind.n;
				nn_next = n_upwind.n_n;
				vxy_next = p_vie_minor[iTri].vxy;
				v_n_next = p_v_n_minor[iTri].xypart();
				T3 Tuse = p_T_minor[iTri];
				Te_next = Tuse.Te;
				Ti_next = Tuse.Ti;
				Tn_next = Tuse.Tn;
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				endpt0 = Clockwise*endpt0;
				vxy_next = Clockwise*vxy_next;
				v_n_next = Clockwise*v_n_next;
				v_overall_next = Clockwise*v_overall_next;
			};
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				endpt0 = Anticlockwise*endpt0;
				vxy_next = Anticlockwise*vxy_next;
				v_n_next = Anticlockwise*v_n_next;
				v_overall_next = Anticlockwise*v_overall_next;
			};

			f64_vec2 edge_normal;
			edge_normal.x = endpt1.y - endpt0.y;
			edge_normal.y = endpt0.x - endpt1.x;

			AreaMajor += 0.5*edge_normal.x*(endpt0.x + endpt1.x);

			Integrated_div_v += 0.5*(vxy_prev + vxy_next).dot(edge_normal);
			Integrated_div_v_n += 0.5*(v_n_prev + v_n_next).dot(edge_normal);
			Integrated_div_v_overall += 0.5*(v_overall_prev + v_overall_next).dot(edge_normal); // Average outward velocity of edge...

			totalmassflux_out.n += 0.5*(n_prev*vxy_prev + n_next*vxy_next).dot(edgenormal);
			totalheatflux_out.Ti += 0.5*(n_prev*Ti_prev*vxy_prev + n_next*Ti_next*vxy_next).dot(edgenormal);
			totalheatflux_out.Te += 0.5*(n_prev*Te_prev*vxy_prev + n_next*Te_next*vxy_next).dot(edgenormal);

			totalmassflux_out.n_n += 0.5*(nn_prev*v_n_prev + nn_next*v_n_next).dot(edgenormal);
			totalmassflux_out.Tn += 0.5*(nn_prev*Tn_prev*v_n_prev + nn_next*Tn_next*v_n_next).dot(edgenormal);

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

		NTplus.N = -h_use*totalmassflux_out.n;
		NTplus.Nn = -h_use*totalmassflux_out.n_n;
		NTplus.NTe = -h_use*totalheatflux_out.Te;
		NTplus.NTi = -h_use*totalheatflux_out.Ti;
		NTplus.NnTn = -h_use*totalheatflux_out.Tn;

		memcpy(p_NTadditionrates + iVertex, NTplus, sizeof(NTadditionrates));

		// What we need now: 
		//	* Cope with non-domain vertex
		p_div_v[iVertex] = Integrated_div_v / AreaMajor;
		p_div_v_n[iVertex] = Integrated_div_v_n / AreaMajor;
		p_div_v_overall[iVertex] = Integrated_div_v_overall / AreaMajor;
		// 3 divisions -- could speed up by creating 1.0/AreaMajor. Except it's bus time anyway.
	} else {
		p_div_v[iVertex] = 0.0;
		p_div_v_n[iVertex] = 0.0;
		p_div_v_overall[iVertex] = 0.0;
	};
}




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
	bool bFeint)
{
	// Don't forget we can use 16KB shared memory to save a bit of overspill:
	// (16*1024)/(512*8) = 4 doubles only for 512 threads. 128K total register space per SM we think.

	__shared__ f64 Iz[threadsPerTileMinor], sigma_zz[threadsPerTileMinor];
	__shared__ f64_vec2 omega[threadsPerTileMinor], grad_Az[threadsPerTileMinor],
		gradTe[threadsPerTileMinor];
		
	// Putting 8 reduces to 256 simultaneous threads. Experiment with 4 in shared.

	// f64 viz0_coeff_on_Lap_Az, vez0_coeff_on_Lap_Az; // THESE APPLY TO FEINT VERSION. ASSUME NOT FEINT FIRST.
	
	v4 v0;
	f64 denom, ROCAzdot_antiadvect, AreaMinor;
	f64_vec3 vn0;
	
	long const index = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX

	// Branch for domain vertex or triangle:
	
	if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE))
	{
		v4 vie_k = p_vie_src[index];
		v_n_src = p_v_n_src[index];
		nvals n_src = p_n_minor_src[index];
		AreaMinor = p_AreaMinor[index];
		// Are we better off with operator = or with memcpy?
		vn0 = v_n_src;		
		{
			three_vec3 MomAddRate;
			memcpy(&MomAddRate, p_AdditionalMomRates + index, sizeof(three_vec3));
			// CHECK IT IS INTENDED TO AFFECT Nv

			vn0.x += h_use * (MomAddRate.neut.x / (n_src.n_n*AreaMinor));
			vn0.y += h_use * (MomAddRate.neut.y / (n_src.n_n*AreaMinor));// MomAddRate is addition rate for Nv. Divide by N.
			
			v0.vxy = vie_k.vxy
				+ h_use * ((m_e*MomAddRate.elec.xypart()
					+ m_i*MomAddRate.ion.xypart())
					/ (n_src.n*(m_i + m_e)*AreaMinor));

			v0.viz = vie_k.viz
				+ h_use * MomAddRate.ion.z / (n_src.n*AreaMinor);
			v0.vez = vie_k.vez
				+ h_use * MomAddRate.elec.z / (n_src.n*AreaMinor);   // UM WHY WAS THIS NEGATIVE

			// + !!!!
		}

		OhmsCoeffs ohm;
		f64 beta_ie_z, Lap_Az;
		f64 cross_section_times_thermal_en, cross_section_times_thermal_in,
			nu_eiBar, nu_eHeart;
		T3 T = p_T_minor[index];

		{
				// Dimensioning inside a brace allows the following vars to go out of scope at the end of the brace.
			f64 sqrt_Te, ionneut_thermal, electron_thermal,
				lnLambda, s_in_MT, s_en_MT, s_en_visc;

			sqrt_Te = sqrt(T.Te);
			ionneut_thermal = sqrt(T.Ti / m_ion + T.Tn / m_n); // hopefully not sqrt(0)
			electron_thermal = sqrt_Te * over_sqrt_m_e;
			lnLambda = Get_lnLambda_d(n_src.n, T.Te);

			s_in_MT = Estimate_Ion_Neutral_MT_Cross_section_d(T.Ti*one_over_kB);
			Estimate_Ion_Neutral_Cross_sections_d(T.Te*one_over_kB, &s_en_MT, &s_en_visc);

			//nu_ne_MT = s_en_MT * electron_thermal * n_src.n; // have to multiply by n_e for nu_ne_MT
			//nu_ni_MT = s_in_MT * ionneut_thermal * n_src.n;
			//nu_in_MT = s_in_MT * ionneut_thermal * n_src.n_n;
			//nu_en_MT = s_en_MT * electron_thermal * n_src.n_n;

			cross_section_times_thermal_en = s_en_MT * electron_thermal;
			cross_section_times_thermal_in = s_in_MT * ionneut_thermal;

			nu_eiBar = nu_eiBarconst * kB_to_3halves*n_src.n*lnLambda / (T.Te*sqrt_Te);
			nu_eHeart = 1.87*nu_eiBar + n_src.n_n*s_en_visc*electron_thermal;
		}

		vn0.x +=  - 0.5*h_use*(M_e_over_en)*(cross_section_times_thermal_en*n_src.n)*(v_n_src.x - vie_k.vxy.x)
				  - 0.5*h_use*(M_i_over_in)*(cross_section_times_thermal_in*n_src.n)*(v_n_src.x - vie_k.vxy.x);
		vn0.x += -0.5*h_use*(M_e_over_en)*(cross_section_times_thermal_en*n_src.n)*(v_n_src.y - vie_k.vxy.y)
				- 0.5*h_use*(M_i_over_in)*(cross_section_times_thermal_in*n_src.n)*(v_n_src.y - vie_k.vxy.y);
		vn0.z += -0.5*h_use*(M_e_over_en)*(cross_section_times_thermal_en*n_src.n)*(v_n_src.z - vie_k.vez)
				- 0.5*h_use*(M_i_over_in)*(cross_section_times_thermal_in*n_src.n)*(v_n_src.z - viee_k.viz);
		denom = 1.0 + h_use * 0.5*M_e_over_en* (cross_section_times_thermal_en*n_src.n)
			+ 0.5*h_use*M_i_over_in* (cross_section_times_thermal_in*n_src.n);

		vn0 /= denom; // It is now the REDUCED value
		
		ohm.beta_ne = 0.5*h_use*(M_e_over_en)*(cross_section_times_thermal_en*n_src.n) / denom;
		ohm.beta_ni = 0.5*h_use*(M_i_over_in)*(cross_section_times_thermal_in*n_src.n) / denom;

				// Now we do vexy:

		grad_Az[threadIdx.x] = p_GradAz[index];
		gradTe[threadIdx.x] = p_GradTe[index];
		LapAz = p_LapAz[index];
		ROCAzdot_antiadvect = ROCAzduetoAdvection[index];

		v0.vxy +=
				- h_use * (q / (2.0*c*(m_i + m_e)))*(vie_k.vez - vie_k.viz)*grad_Az[threadIdx.x]
				- (h_use / (2.0*(m_i + m_e)))*(m_n*M_i_over_in*(cross_section_times_thermal_in*n_src.n_n)
					+ m_n * M_e_over_en*(cross_section_times_thermal_en*n_src.n_n))*
				(vie_k.vxy - v_n_src.xypart() - vn0.xypart());
		
		denom = 1.0 + (h_use / (2.0*(m_i + m_e)))*(
			  m_n* M_i_over_in* (cross_section_times_thermal_in*n_src.n_n) 
			+ m_n * M_e_over_en*(cross_section_times_thermal_en*n_src.n_n))*(1.0 - ohm.beta_ne - ohm.beta_ni);
		v0.vxy /= denom;

		ohm.beta_xy_z = (h_use * q / (2.0*c*(m_i + m_e)*denom)) * grad_Az[threadIdx.x];
		
		omega[threadIdx.x] = qovermc*p_B[index];

		f64 nu_ei_effective = nu_eiBar * (1.0 - 0.9*nu_eiBar*(nu_eHeart*nu_eHeart + qovermc*BZ*qovermc*BZ) /
			(nu_eHeart*(nu_eHeart*nu_eHeart + omega[threadIdx.x].x*omega[threadIdx.x].x + omega[threadIdx.x].y*omega[threadIdx.x].y + qovermc*BZ*qovermc*BZ)));

		f64 Azdot_k = p_AAdot_src[index].Azdot;

		//if ((iPass == 0) || (bFeint == false))
		{
			v0.viz += 
					- 0.5*h_use*qoverMc*(2.0*Azdot_k

						+ h_use * ROCAzdot_antiadvect + h_use * c*c*(Lap_Az
							+ TWOPIoverc * q*n_src.n*(vie_k.viz - vie_k.vez)))
						- 0.5*h_use*qoverMc*(vie_k.vxy + v0.vxy).dot(grad_Az[threadIdx.x]);
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
				1.5*h_use*nu_eiBar*((omega[threadIdx.x].x*qovermc*BZ - nu_eHeart * omega[threadIdx.x].y)*gradTe[threadIdx.x].x +
				(omega[threadIdx.x].y*qovermc*BZ + nu_eHeart * omega.x)*gradTe[threadIdx.x].y) /
					(m_i*nu_eHeart*(nu_eHeart*nu_eHeart + omega[threadIdx.x].dot(omega[threadIdx.x])));

		v0.viz += -h_use * 0.5*M_n_over_ni*nu_in_MT *(vie_k.viz - v_n_src.z - vn0.z) // THIS DOESN'T LOOK RIGHT
				+ h_use * 0.5*(moverM)*nu_ei_effective*(vie_k.vez - vie_k.viz);

		denom = 1.0 + h_use * h_use*PI*qoverM*q*n_src.n + h_use * 0.5*qoverMc*(grad_Az[threadIdx.x].dot(beta_xy_z)) +
					h_use * 0.5*M_n_over_ni*(cross_section_times_thermal_in*n_src.n_n) *(1.0 - beta_ni) + h_use * 0.5*moverM*nu_ei_effective;

//				viz0_coeff_on_Lap_Az = -0.5*h_use*qoverMc*h_use*c*c / denom;

		v0.viz /= denom;
				
		ohm.sigma_i_zz = h_use * qoverM / denom;
		beta_ie_z = (h_use*h_use*PI*qoverM*q*n_src.n
				+ 0.5*h_use*qoverMc*(grad_Az[threadIdx.x].dot(ohm.beta_xy_z))
				+ h_use * 0.5*M_n_over_ni*(cross_section_times_thermal_in*n_src.n_n) *ohm.beta_ne
				+ h_use * 0.5*moverM*nu_ei_effective) / denom;

		v0.vez +=
				 h_use * 0.5*qovermc*(2.0*Azdot_k
				+ h_use * ROCAzdot_antiadvect
						+ h_use * c*c*(Lap_Az
							+ TWOPIoverc * q*n_src.n*(vie_k.viz + v0.viz - vie_k.vez))) // ?????????????????
				+ 0.5*h_use*qovermc*(vie_k.vxy + v0.vxy + v0.viz * ohm.beta_xy_z).dot(grad_Az[threadIdx.x]);
		
		v0.vez -=
			1.5*h_use*nu_eiBar*((omega[threadIdx.x].x*qovermc*BZ - nu_eHeart * omega[threadIdx.x].y)*gradTe[threadIdx.x].x +
			(omega[threadIdx.x].y*qovermc*BZ + nu_eHeart * omega[threadIdx.x].x)*gradTe[threadIdx.x].y) /
				(m_e*nu_eHeart*(nu_eHeart*nu_eHeart + omega[threadIdx.x].dot(omega[threadIdx.x])));
				
				// could store this from above and put opposite -- dividing by m_e instead of m_i

		v0.vez += -0.5*h_use*M_n_over_ne*(cross_section_times_thermal_en*n_src.n_n) *(vie_k.vez - v_n_src.z - vn0.z - beta_ni * v0.viz)
			- 0.5*h_use*nu_ei_effective*(vie_k.vez - vie_k.viz - v0.viz);
		denom = 1.0 + (h_use*h_use*PI*q*qoverm*n_src.n
				+ 0.5*h_use*qovermc*(grad_Az[threadIdx.x].dot(ohm.beta_xy_z)))*(1.0 - beta_ie_z)
				+ 0.5*h_use*M_n_over_ne*(cross_section_times_thermal_en*n_src.n_n) *(1.0 - ohm.beta_ne - ohm.beta_ni * beta_ie_z)
				+ 0.5*h_use*nu_ei_effective*(1.0 - beta_ie_z);

		//		vez0_coeff_on_Lap_Az = h_use * h_use*0.5*qovermc* c*c / denom; 

		ohm.sigma_e_zz = (-h_use * qoverm + h_use * h_use*PI*q*qoverm*n_src.n*ohm.sigma_i_zz
				+ h_use * 0.5*qovermc*(grad_Az[threadIdx.x].dot(ohm.beta_xy_z))*ohm.sigma_i_zz
				+ 0.5*h_use*M_n_over_ne*(cross_section_times_thermal_en*n_src.n_n) *ohm.beta_ni*ohm.sigma_i_zz
				+ 0.5*h_use*nu_ei_effective*ohm.sigma_i_zz)
				/ denom;
				
		v0.vez /= denom;

		// Now update viz(Ez):
		v0.viz += beta_ie_z * v0.vez;
		ohm.sigma_i_zz += beta_ie_z * ohm.sigma_e_zz;

		// sigma_e_zz and sigma_i_zz are change in vz for a change in Ez
		f64 EzShape = GetEzShape_d(data_use.pos.modulus());
		ohm.sigma_i_zz *= EzShape;
		ohm.sigma_e_zz *= EzShape;

		// ==============================================================================================

		p_v0_dest[index] = v0;
		p_ohm_coeff[index] = ohm;
		p_vn0_dest[index] = vn0;

		Iz[threadIdx.x] = q*AreaMinor*p_n_dest[index].n*(viz0 - vez0);
		sigma_zz[threadIdx.x] = q*AreaMinor*p_n_dest[index].n*(sigma_i_zz - sigma_e_zz);
		// Totally need to be skipping the load of an extra n.

		// On iPass == 0, we need to do the accumulate.
		p_Azdot_intermediate[index] = Azdot_k
			+ h_use * ROCAzdot_antiadvect + h_use * c*c*(Lap_Az +
				0.5*FOURPI_OVER_C * q*n_src.n*(data_k.viz - data_k.vez)); // INTERMEDIATE
		//data_1.Azdot = data_k.Azdot
		//	+ h_use * ROCAzdot_antiadvect + h_use * c*c*(Lap_Az +
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
		if ((index < BEGINNING_OF_CENTRAL) && ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL)))
		{
			if (iPass > 0) {
						(pDestMesh->pData + iMinor)->Azdot = 0.0;
						Azdot0[iMinor] = 0.0;
						gamma[iMinor] = 0.0;
			}; // Set Az equal to neighbour in every case, after Accelerate routine.
			// *********************************************// Set Az equal to neighbour in every case, after Accelerate routine.
			// *********************************************// Set Az equal to neighbour in every case, after Accelerate routine.
			// *********************************************// Set Az equal to neighbour in every case, after Accelerate routine.
			// *********************************************// Set Az equal to neighbour in every case, after Accelerate routine.

		} else {
			// Let's make it go right through the middle of a triangle row for simplicity.

			f64 Jz = 0.0;
			if ((iMinor >= numStartZCurrentTriangles) && (iMinor <  numEndZCurrentTriangles))
			{
				// Azdotdot = c^2 (Lap Az + 4pi/c Jz)
				// ASSUME we are fed Iz_prescribed.
				//Jz = -Iz_prescribed / (real)(numEndZCurrentTriangles - numStartZCurrentTriangles);

				AreaMinor = p_AreaMinor[index];
				Jz = negative_Iz_per_triangle/AreaMinor; // Iz would come from multiplying back by area and adding.
			};

			p_Azdot_intermediate[index] = Azdot_k + h_use * c*(c*p_LapAz[index] + 4.0*PI*Jz);
				// + h_use * ROCAzdot_antiadvect // == 0
			
			// FEINT:

			p_Azdot0[index] = Azdot_k + h_use*4.0*PI*c*Jz;
			gamma[iMinor] = h_use * c*c;
		};		
	};
	make
	__constant__ f64 Iz_prescribed;
	__constant__ f64 negative_Iz_per_triangle; // -Iz_prescribed / (real)(numEndZCurrentTriangles - numStartZCurrentTriangles)
	__constant__ long numStartZCurrentTriangles, numEndZCurrentTriangles;

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
		
}

__global__ void kernelUpdateVelocityAndAzdot(
	f64 h_use,
	f64_vec3 * __restrict__ p_vn0,
	v4 * __restrict__ p_v0, 
	OhmsCoeffs * __restrict__ p_OhmsCoeffs,
	f64 * __restrict__ p_Azdot_update,

	v4 * __restrict__ p_vie_out,
	f64_vec3 * __restrict__ p_vn_out
) {
	long index = blockIdx.x*blockDim.x + threadIdx.x;

	OhmsCoeffs ohm = p_OhmsCoeffs[index];
	v4 vie;
	v.vez = v0.vez + ohm.sigma_e_zz * Ez_strength;  // 2
	v.viz = v0.viz + ohm.sigma_i_zz * Ez_strength;  // 2
	v.vxy = v0.vxy + ohm.beta_xy_z * (v.viz - v.vez);   // 4
	f64_vec3 v_n = p_vn0[index];							 // 3 sep
	v_n.x += (ohm.beta_ne + ohm.beta_ni)*v.vxy.x;    // 2
	v_n.y += (ohm.beta_ne + ohm.beta_ni)*v.vxy.y;
	v_n.z += ohm.beta_ne * v.vez + ohm.beta_ni * v.viz;

	//data_1.Azdot = data_k.Azdot
	//	+ h_use * ROCAzdot_antiadvect + h_use * c*c*(Lap_Az +
	//		0.5*FOURPI_OVER_C * q*data_use.n*(data_k.viz + data_1.viz
	//			- data_k.vez - data_1.vez));

	//data_1.Azdot = data_k.Azdot
	//		+ h_use * ROCAzdot_antiadvect + h_use * c*c*(Lap_Az +
	//			0.5*FOURPI_OVER_C * q*data_use.n*(data_k.viz - data_k.vez));
	// intermediate

	// Did we use ROCAzdot_antiadvect anywhere else?
	// LapAz: did we pick this up from anywhere?
	// Can do this update during 1st pass except for v's contribution, add it after.
	nvals n_use = p_n_minor[index];

	memcpy(p_vie_out[index], &vie, sizeof(v4)); // operator = vs memcpy
	p_vn_out[index] = v_n;
	p_Azdot_update[index] += h_use*c*TWOPI*q*n_use.n*(v.viz - v.vez));
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

// Try resetting frills here and ignoring in calculation:
__global__ void kernelResetFrillsAz << <numTriTiles, threadsPerTileMinor >> > (
	structural * __restrict__ p_info,
	LONG3 * __restrict__ trineighbourindex,
	f64 * __restrict__ p_Az) 
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	structural info = p_info[index];
	if (info.flag == INNER_FRILL) || (info.flag == OUTER_FRILL)
	{

		****************************
		Please make sure they have distinct codes to all else.
		
		LONG3 izNeigh = trineighbourindex[index];

		p_Az[index] = p_Az[izNeigh.i1];
	}
}

__global__ void kernelCreateEpsilonAndJacobi(
	structural * __restrict__ p_info,
	f64 * __restrict__ p_Az_array_next,
	f64 * __restrict__ p_Az_array,
	f64 * __restrict__ p_Azdot0,
	f64 * __restrict__ p_gamma,
	f64 * __restrict__ p_LapCoeffself,
	f64 * __restrict__ p_Lap_Aznext,
	f64 * __restrict__ p_epsilon,
	f64 * __restrict__ p_Jacobi_x)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	f64 eps;
	structural info = p_info[index];
	if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL))
	{
		eps = p_Lap_Aznext[index]
		p_Jacobi_x[index] = -eps / p_LapCoeffSelf[index];
	}
	else {
		eps = p_Az_array_next[index] - h_use * p_gamma[index] * p_Lap_Aznext[index]
			- p_Az_array[index]-h_use*p_Azdot0[index];
		p_Jacobi_x[index] = -eps / (1.0 - h_use * p_gamma[index] * p_LapCoeffSelf[index]);
	};
	p_epsilon[index] = eps;

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
		depsbydbeta = (p_Jacobi[index] - h_use * p_gamma[index] * p_LapJacobi[index]);
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

__global__ void kernelGetLap_verts(
	structural * __restrict__ p_info,
	f64 * __restrict__ p_Az,
	long * __restrict__ p_izNeighMinor,
	long * __restrict__ p_izTri,
	f64 * __restrict__ p_LapAz)
{
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];
	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajor];
	__shared__ f64 shared_Az[threadsPerTileMinor];
	__shared__ f64 shared_Az_verts[threadsPerTileMajor];

	// For now, stick with idea that vertices have just major indices that come after tris.
	// Minor indices are not made contiguous - although it might be better ultimately.
	
	long const iVertex = blockDim.x*blockIdx.x+threadIdx.x;

	structural info = p_info[iVertex+BEGINNING_OF_CENTRAL];
	shared_pos_verts[threadIdx.x] = info.pos;
	shared_Az_verts[threadIdx.x] = p_Az[iVertex+BEGINNING_OF_CENTRAL];
	{
		structural info2[2];
		memcpy(info2,p_info[threadsPerTileMinor*blockIdx.x + 2*threadIdx.x,2*sizeof(structural));
		shared_pos[threadIdx.x*2] = info2[0].pos;
		shared_pos[threadIdx.x*2+1] = info2[1].pos;
		memcpy(shared_Az+threadIdx.x*2,p_Az[threadsPerTileMinor*blockIdx.x+2*threadIdx.x,2*sizeof(f64));
	}

	__syncthreads();

	{
		f64 Our_integral_Lap_Az = 0.0;
		f64 AreaMinor = 0.0;
		long tri_len = info.tri_len;
		long izTri[MAXNEIGH];

		memcpy(izTri,p_izTri+MAXNEIGH*index,sizeof(long)*MAXNEIGH);

		iprev = tri_len-1;
		if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMajor))
		{
			prevAz = shared_Az[izTri[iprev]-StartMinor];
			prevpos = shared_pos[izTri[iprev]-StartMinor];
		} else {
			
		}
	}
	// Better if we use same share to do both tris and verts

	// Idea: let's make it called for # minor threads, each loads 1 shared value,
	// and only half the threads run first for the vertex part. That is a pretty good idea.
		
	long const iTri = threadsPerTileMinor*blockIdx.x + 2*threadIdx.x;

	ourAz = shared_Az[2*threadIdx.x];
	
	
}

__global__ void kernelGetLap_minor(
	structural * __restrict__ p_info,
	f64 * __restrict__ p_Az,

	long * __restrict__ p_izTri,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtri_vertex, 
	char * __restrict__ p_szPBCtriminor,

	f64 * __restrict__ p_LapAz)
{
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];
	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajor];
	__shared__ f64 shared_Az[threadsPerTileMinor];
	__shared__ f64 shared_Az_verts[threadsPerTileMajor];
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
		if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) prevpos = Clockwise*prevpos;
		if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) prevpos = Anticlockwise*prevpos;

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
		if (szPBC[i] == ROTATE_ME_CLOCKWISE) opppos = Clockwise*opppos;
		if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) opppos = Anticlockwise*opppos;

		endpt0 = THIRD * (ourdata.pos + oppdata.pos + prevdata.pos);

		short iend = tri_len;
		f64_vec2 projendpt0;
		if ((pVertex->flags == INNERMOST) || (pVertex->flags == OUTERMOST)) {
			
			iend = tri_len - 2;
			if (pVertex->flags == OUTERMOST) {
				endpt0.project_to_radius(projendpt0, OutermostFrillCentroidRadius); // back of cell for Lap purposes
			}
			else {
				endpt0.project_to_radius(projendpt0, InnermostFrillCentroidRadius); // back of cell for Lap purposes
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
				nextpos = shared_pos[izTri[inext] - StartMinor];
			}
			else {
				nextAz = p_Az[izTri[inext]];
				nextpos = p_info[izTri[inext]].pos;
			}
			if (szPBC[inext] == ROTATE_ME_CLOCKWISE) nextpos = Clockwise*nextpos;
			if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) nextpos = Anticlockwise*nextpos;

			endpt1 = THIRD * (nextpos + ourpos + opppos);

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
			area_quadrilateral = 0.5*(
				(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
				+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
				+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
				+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
				);
			//f64_vec2 grad_Az = integ_grad_Az / area_quadrilateral;
			Our_integral_Lap_Az += integ_grad_Az.dot(edge_normal) / area_quadrilateral;

			AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;

			++iprev;
			endpt0 = endpt1;
			prevpos = opppos;
			opppos = nextpos;
			prevAz = oppAz;
			oppAz = nextAz;
		}; // next i

		if ((pVertex->flags == INNERMOST) || (pVertex->flags == OUTERMOST)) {
			// Now add on the final sides to give area:

			//    3     4
			//     2 1 0
			// endpt0=endpt1 is now the point north of edge facing 2 anyway.
			f64_vec2 projendpt1;

			if (pVertex->flags == OUTERMOST) {
				endpt1.project_to_radius(projendpt1, OutermostFrillCentroidRadius);
			} else {
				endpt1.project_to_radius(projendpt1, InnermostFrillCentroidRadius);
			};
			edge_normal.x = projendpt1.y - endpt1.y;
			edge_normal.y = endpt1.x - projendpt1.x;
			AreaMinor += (0.5*projendpt1.x + 0.5*endpt1.x)*edge_normal.x;

			edge_normal.x = projendpt0.y - projendpt1.y;
			edge_normal.y = projendpt1.x - projendpt0.x;
			AreaMinor += (0.5*projendpt1.x + 0.5*projendpt0.x)*edge_normal.x;
			// line between out-projected points
		};

		LapAz_array[iVertex + BEGINNING_OF_CENTRAL] = Our_integral_Lap_Az / AreaMinor;

	}; // was thread in the first half of the block

	info = p_info[iMinor];
	ourAz = shared_Az[threadIdx.x];

	long izNeighMinor[6];
	char szPBC[6];
	memcpy(izNeighMinor, p_izneighminor + iMinor * 6, sizeof(long) * 6);
	memcpy(szPBC, p_szPBCtriminor + iMinor * 6, sizeof(char) * 6);
	if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL)) {
		if (izNeighMinor[0] >= StartMinor) && (izNeighMinor[0] < EndMinor)
		{
			oppAz = shared_Az[izNeighMinor[0]-StartMinor];
		} else {
			oppAz = p_Az[izNeighMinor[0]];
		};
		LapAz_array[iMinor] = oppAz - ourAz;
	}
	else {

		Our_integral_Lap_Az = 0.0;
		AreaMinor = 0.0;

		iprev = 5; i = 0;
		if ((izNeighMinor[iprev] >= StartMinor) && (izNeighMinor[iprev] < EndMinor))
		{
			prevAz = shared_Az[izNeighMinor[iprev] - StartMinor];
			prevpos = shared_pos[izNeighMinor[iprev] - StartMinor];
		} else {
			if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
				(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL)) 
			{
				prevAz = shared_Az_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
				prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
			} else {
				prevAz = p_Az[izNeighMinor[iprev]];
				prevpos = p_info[izNeighMinor[iprev]].pos;
			};
		};
		if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) prevpos = Clockwise*prevpos;
		if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) prevpos = Anticlockwise*prevpos;
	
		i = 0;
		if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
		{
			oppAz = shared_Az[izNeighMinor[i] - StartMinor];
			opppos = shared_pos[izNeighMinor[i] - StartMinor];
		} else {
			if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
			(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
			{
				oppAz = shared_Az_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
				opppos = shared_pos_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
			} else {
				oppAz = p_Az[izNeighMinor[i]];
				opppos = p_info[izNeighMinor[i]].pos;
			};
		};
		if (szPBC[i] == ROTATE_ME_CLOCKWISE) opppos = Clockwise*opppos;
		if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) opppos = Anticlockwise*opppos;
		
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
					nextAz = p_Az[izNeighMinor[inext]];
					nextpos = p_info[izNeighMinor[inext]].pos;
				};
			};
			if (szPBC[inext] == ROTATE_ME_CLOCKWISE) nextpos = Clockwise*nextpos;
			if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) nextpos = Anticlockwise*nextpos;

			// New definition of endpoint of minor edge:

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
			area_quadrilateral = 0.5*(
				(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
				+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
				+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
				+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
				);

			//f64_vec2 grad_Az = integ_grad_Az / area_quadrilateral;
			Our_integral_Lap_Az += integ_grad_Az.dot(edge_normal)/ area_quadrilateral;

			AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;

			endpt0 = endpt1;
			prevAz = oppAz;
			oppAz = nextAz;
			prevpos = opppos;
			opppos = nextpos;
			// There is an even quicker way which is to rotate pointers. No memcpy needed.
		};

		LapAz_array[iMinor] = Our_integral_Lap_Az / AreaMinor;	
	};

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
	long * __restrict__ p_izNeighTriMinor,
	char * __restrict__ p_szPBCtriminor,

	three_vec3 * __restrict__ p_AdditionalMomrates,
	ShardModel * __restrict__ p_n_shards,

	f64_vec2 * __restrict__ p_GradTe,
	f64_vec2 * __restrict__ p_GradAz,
	f64_vec2 * __restrict__ p_LapAz,
	f64_vec3 * __restrict__ p_B
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

	shared_pos[threadIdx.x] = p_info[iMinor].pos;
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
		if (info.flag == DOMAIN_VERTEX) {
			T3 T_ = p_T_minor[iVertex + BEGINNING_OF_CENTRAL];
			shared_T_verts[threadIdx.x].Te = T_.Te;
			shared_T_verts[threadIdx.x].Ti = T_.Ti;
			memcpy(&(shared_n_shards[threadIdx.x]), p_n_shards[iVertex], sizeof(ShardModel)); // + 13
		} else {
			// save several bus trips;
			// it still manages to coalesce, let's hope, because ShardModel is 13 doubles not 12.
			shared_T_verts[threadIdx.x].Te = 0.0;
			shared_T_verts[threadIdx.x].Ti = 0.0;
			memset(&(shared_n_shards[threadIdx.x]), 0, sizeof(ShardModel)); // + 13
		};
	};

	__syncthreads();

	f64 ourAz, oppAz, prevAz, nextAz;
	f64 ourAzdot, oppAzdot, prevAzdot, nextAzdot;
	f64_vec2 opppos, prevpos, nextpos;
	T2 ourT, oppT, prevT, nextT;
	//nvals our_n, opp_n, prev_n, next_n;

	if (threadIdx.x < threadsPerTileMajor) {

		f64_vec2 Our_integral_curl_Az, Our_integral_grad_Az, Our_integral_grad_Te;
		f64 Our_integral_Lap_Az;
		Our_integral_curl_Az.x = 0.0;
		Our_integral_curl_Az.y = 0.0;
		Our_integral_grad_Az.x = 0.0;
		Our_integral_grad_Az.y = 0.0;
		Our_integral_grad_Azdot.x = 0.0;
		Our_integral_grad_Azdot.y = 0.0;
		Our_integral_grad_Te.x = 0.0;
		Our_integral_grad_Te.y = 0.0;
		Our_integral_Lap_Az = 0.0;
		AreaMinor = 0.0;
		three_vec3 ownrates;
		memcpy(&ownrates, p_AdditionalMomrates[iVertex + BEGINNING_OF_CENTRAL], sizeof(three_vec3));

		long izTri[MAXNEIGH];
		char szPBC[MAXNEIGH];
		short tri_len = info.neigh_len;

		memcpy(izTri, long_array_of_izTri + index*MAXNEIGH, MAXNEIGH * sizeof(long));
		memcpy(szPBC, long_array_of_szPBC + index*MAXNEIGH, MAXNEIGH * sizeof(char));)

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
				AAdot temp p_AAdot[izTri[iprev]];
				prevAz = temp.Az;
				prevAzdot = temp.Azdot;
				prevpos = p_info[izTri[iprev]].pos;
			}
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) prevpos = Clockwise*prevpos;
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) prevpos = Anticlockwise*prevpos;

			short i = 0;
			if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
			{
				T3 opp_T = shared_T[izTri[i] - StartMinor];
				oppT.Te = opp_T.Te; oppT.Ti = opp_T.Ti;
				oppAz = shared_Az[izTri[i] - StartMinor];
				oppAzdot = shared_Azdot[izTri[i] - StartMinor];
				opppos = shared_pos[izTri[i] - StartMinor];
			}
			else {
				opp_T = p_T_minor[izTri[i]];
				AAdot temp p_AAdot[izTri[i]];
				oppAz = temp.Az;
				oppAzdot = temp.Azdot;
				opppos = p_info_minor[izTri[i]].pos;
			}
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) opppos = Clockwise*opppos;
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) opppos = Anticlockwise*opppos;

			// Think carefully: DOMAIN vertex cases for n,T ...

			f64 n0 = shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[iprev] + shared_n_shards[threadIdx.x].n_cent;
			endpt0 = THIRD * (ourdata.pos + oppdata.pos + prevdata.pos);

			short iend = tri_len;
			f64_vec2 projendpt0;
			if ((pVertex->flags == INNERMOST) || (pVertex->flags == OUTERMOST)) {
				iend = tri_len - 2;
				if (pVertex->flags == OUTERMOST) {
					endpt0.project_to_radius(projendpt0, OutermostFrillCentroidRadius); // back of cell for Lap purposes
				}
				else {
					endpt0.project_to_radius(projendpt0, InnermostFrillCentroidRadius); // back of cell for Lap purposes
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
					AAdot temp p_AAdot[izTri[inext]];
					nextAz = temp.Az;
					nextAzdot = temp.Azdot;
					nextpos = p_info[izTri[inext]].pos;
				}
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) nextpos = Clockwise*nextpos;
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) nextpos = Anticlockwise*nextpos;

				endpt1 = THIRD * (nextpos + ourpos + opppos);
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
				area_quadrilateral = 0.5*(
					(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
					+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
					+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
					+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
					);
				//f64_vec2 grad_Az = integ_grad_Az / area_quadrilateral;
				Our_integral_Lap_Az += integ_grad_Az.dot(edge_normal) / area_quadrilateral;

				T2 T0, T1;
				f64 n1;
				T0.Te = THIRD* (prevT.Te + shared_T[threadIdx.x].Te + oppT.Te);
				T1.Te = THIRD * (nextT.Te + shared_T[threadIdx.x].Te + oppT.Te);
				T0.Ti = THIRD * (prevT.Ti + shared_T[threadIdx.x].Ti + oppT.Ti);
				T1.Ti = THIRD * (nextT.Ti + shared_T[threadIdx.x].Ti + oppT.Ti);
				n1 = shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[inext] + shared_n_shards[threadIdx.x].n_cent;

				// Assume neighs 0,1 are relevant to border with tri 0 minor.
				// *********
				Verify that tri 0 is formed from our vertex, neigh 0 and neigh 1;
				// *********

				// To get integral grad we add the averages along the edges times edgenormals
				ownrates.ion -= Make3(0.5*(n0 * T0.Ti + n1 * T1.Ti)*over_m_i*edge_normal, 0.0);
				ownrates.elec -= Make3(0.5*(n0 * T0.Te + n1 * T1.Te)*over_m_e*edge_normal, 0.0);

				Our_integral_grad_Te += 0.5*(T0.Te + T1.Te) * edge_normal;

				Az_edge = SIXTH * (2.0*ourAz + 2.0*oppAz + prevAz + nextAz);
				Azdot_edge = SIXTH * (2.0*ourAzdot + 2.0*oppAzdot + prevAzdot + nextAzdot);
				Our_integral_grad_Azdot += Azdot_edge * edge_normal;
				Our_integral_grad_Az += Az_edge * edge_normal;
				Our_integral_curl_Az += Az_edge * (endpt1 - endpt0);

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

			if ((pVertex->flags == INNERMOST) || (pVertex->flags == OUTERMOST)) {
				// Now add on the final sides to give area:

				//    3     4
				//     2 1 0
				// endpt0=endpt1 is now the point north of edge facing 2 anyway.
				f64_vec2 projendpt1;

				if (pVertex->flags == OUTERMOST) {
					endpt1.project_to_radius(projendpt1, OutermostFrillCentroidRadius);
				}
				else {
					endpt1.project_to_radius(projendpt1, InnermostFrillCentroidRadius);
				};
				edge_normal.x = projendpt1.y - endpt1.y;
				edge_normal.y = endpt1.x - projendpt1.x;
				AreaMinor += (0.5*projendpt1.x + 0.5*endpt1.x)*edge_normal.x;

				edge_normal.x = projendpt0.y - projendpt1.y;
				edge_normal.y = projendpt1.x - projendpt0.x;
				AreaMinor += (0.5*projendpt1.x + 0.5*projendpt0.x)*edge_normal.x;
				// line between out-projected points
			};

			GradAz[iVertex + BEGINNING_OF_CENTRAL] = Our_integral_grad_Az / AreaMinor;
			LapAzArray[iVertex + BEGINNING_OF_CENTRAL] = Our_integral_Lap_Az / AreaMinor;
			GradTeArray[iVertex + BEGINNING_OF_CENTRAL] = Our_integral_grad_Te / AreaMinor;
			p_B[iVertex + BEGINNING_OF_CENTRAL] = Make3(Our_integral_curl_Az / AreaMinor, BZ_CONSTANT);
			AreaMinorArray[iVertex + BEGINNING_OF_CENTRAL] = AreaMinor;

			// wow :
			f64_vec2 overall_v_ours = p_overall_v_minor[iVertex + BEGINNING_OF_CENTRAL];
			ROCAzduetoAdvection[iVertex + BEGINNING_OF_CENTRAL] = overall_v_ours.dot(Our_integral_grad_Az / AreaMinor);
			ROCAzdotduetoAdvection[iVertex + BEGINNING_OF_CENTRAL] = overall_v_ours.dot(Our_integral_grad_Azdot / AreaMinor);

			// No neutral stuff in this kernel, momrates should be set now:
			memcpy(pAdditionalMomRates + iVertex + BEGINNING_OF_CENTRAL, &ownrates, sizeof(three_vec3));
		} else {
			// NOT domain vertex: Do Az, Azdot only:


			short iprev = tri_len - 1;
			if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMinor))
			{
				prevAz = shared_Az[izTri[iprev] - StartMinor];
				prevAzdot = shared_Azdot[izTri[iprev] - StartMinor];
				prevpos = shared_pos[izTri[iprev] - StartMinor];
			}
			else {
				AAdot temp p_AAdot[izTri[iprev]];
				prevAz = temp.Az;
				prevAzdot = temp.Azdot;
				prevpos = p_info[izTri[iprev]].pos;
			}
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) prevpos = Clockwise*prevpos;
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) prevpos = Anticlockwise*prevpos;

			short i = 0;
			if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
			{
				oppAz = shared_Az[izTri[i] - StartMinor];
				oppAzdot = shared_Azdot[izTri[i] - StartMinor];
				opppos = shared_pos[izTri[i] - StartMinor];
			}
			else {
				AAdot temp p_AAdot[izTri[i]];
				oppAz = temp.Az;
				oppAzdot = temp.Azdot;
				opppos = p_info_minor[izTri[i]].pos;
			}
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) opppos = Clockwise*opppos;
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) opppos = Anticlockwise*opppos;

			// Think carefully: DOMAIN vertex cases for n,T ...

			f64 n0 = shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[iprev] + shared_n_shards[threadIdx.x].n_cent;
			endpt0 = THIRD * (ourdata.pos + oppdata.pos + prevdata.pos);

			short iend = tri_len;
			f64_vec2 projendpt0;
			if ((pVertex->flags == INNERMOST) || (pVertex->flags == OUTERMOST)) {
				iend = tri_len - 2;
				if (pVertex->flags == OUTERMOST) {
					endpt0.project_to_radius(projendpt0, OutermostFrillCentroidRadius); // back of cell for Lap purposes
				}
				else {
					endpt0.project_to_radius(projendpt0, InnermostFrillCentroidRadius); // back of cell for Lap purposes
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
					AAdot temp p_AAdot[izTri[inext]];
					nextAz = temp.Az;
					nextAzdot = temp.Azdot;
					nextpos = p_info[izTri[inext]].pos;
				}
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) nextpos = Clockwise*nextpos;
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) nextpos = Anticlockwise*nextpos;

				endpt1 = THIRD * (nextpos + ourpos + opppos);
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
				area_quadrilateral = 0.5*(
					(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
					+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
					+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
					+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
					);
				//f64_vec2 grad_Az = integ_grad_Az / area_quadrilateral;
				Our_integral_Lap_Az += integ_grad_Az.dot(edge_normal) / area_quadrilateral;

				// To get integral grad we add the averages along the edges times edgenormals
				Az_edge = SIXTH * (2.0*ourAz + 2.0*oppAz + prevAz + nextAz);
				Azdot_edge = SIXTH * (2.0*ourAzdot + 2.0*oppAzdot + prevAzdot + nextAzdot);
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
			}; // next i

			if ((pVertex->flags == INNERMOST) || (pVertex->flags == OUTERMOST)) {
				// Now add on the final sides to give area:

				//    3     4
				//     2 1 0
				// endpt0=endpt1 is now the point north of edge facing 2 anyway.
				f64_vec2 projendpt1;

				if (pVertex->flags == OUTERMOST) {
					endpt1.project_to_radius(projendpt1, OutermostFrillCentroidRadius);
				}
				else {
					endpt1.project_to_radius(projendpt1, InnermostFrillCentroidRadius);
				};
				edge_normal.x = projendpt1.y - endpt1.y;
				edge_normal.y = endpt1.x - projendpt1.x;
				AreaMinor += (0.5*projendpt1.x + 0.5*endpt1.x)*edge_normal.x;

				edge_normal.x = projendpt0.y - projendpt1.y;
				edge_normal.y = projendpt1.x - projendpt0.x;
				AreaMinor += (0.5*projendpt1.x + 0.5*projendpt0.x)*edge_normal.x;
				// line between out-projected points
			};

			GradAz[iVertex + BEGINNING_OF_CENTRAL] = Our_integral_grad_Az / AreaMinor;
			LapAzArray[iVertex + BEGINNING_OF_CENTRAL] = Our_integral_Lap_Az / AreaMinor;
			p_B[iVertex + BEGINNING_OF_CENTRAL] = Make3(Our_integral_curl_Az / AreaMinor, BZ_CONSTANT);
			AreaMinorArray[iVertex + BEGINNING_OF_CENTRAL] = AreaMinor;

			ROCAzduetoAdvection[iVertex + BEGINNING_OF_CENTRAL] = 0.0;
			ROCAzdotduetoAdvection[iVertex + BEGINNING_OF_CENTRAL] = 0.0;
		};

	}; // was it domain vertex or Az-only
	// This branching is itself a good argument for doing Az in ITS own separate routine with no need for n_shard.

	// __syncthreads(); // end of first vertex part
	// Do we need syncthreads? Not overwriting any shared data here...

	// now the minor with n_ion part:
	info = p_info[iMinor];
	ourAz = shared_Az[threadIdx.x];

	long izNeighMinor[6];
	char szPBC[6];
	memcpy(izNeighMinor, p_izneighminor + iMinor * 6, sizeof(long) * 6);
	memcpy(szPBC, p_szPBCtriminor + iMinor * 6, sizeof(char) * 6);

	T2 prevT, nextT, oppT;
	f64 prevAz, nextAz, oppAz, ourAz;
	f64 prevAzdot, nextAzdot, oppAzdot, ourAzdot;

	three_vec3 ownrates_minor;
	// this is not a clever way of doing it. Want more careful.

	if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL)) {
		if (izNeighMinor[0] >= StartMinor) && (izNeighMinor[0] < EndMinor)
		{
			oppAz = shared_Az[izNeighMinor[0] - StartMinor];
		}
		else {

			AAdot temp = p_AAdot[izNeighMinor[0]];
			oppAz = temp.Az;
		};
		LapAz_array[iMinor] = oppAz - ourAz;

		ROCAzduetoAdvection[iMinor] = 0.0;
		ROCAzdotduetoAdvection[iMinor] = 0.0;
		GradAz[iMinor] = Vector2(0.0, 0.0);
		memset(&(p_B[iMinor]), 0, sizeof(f64_vec3));
		GradTeArray[iMinor] = Vector2(0.0, 0.0);
		AreaMinorArray[iMinor] = 1.0e-12;
		memset(p_AdditionalMomrates[iMinor], 0, sizeof(three_vec3));
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
		AreaMinor = 0.0;

		if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS)) {

			memcpy(&ownrates_minor, p_AdditionalMomRates + iMinor, sizeof(three_vec3));

			iprev = 5; i = 0;
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
					prevpos = p_info[izNeighMinor[iprev]].pos;
				};
			};
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) prevpos = Clockwise*prevpos;
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) prevpos = Anticlockwise*prevpos;

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
					opppos = p_info[izNeighMinor[i]].pos;
				};
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) opppos = Clockwise*opppos;
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) opppos = Anticlockwise*opppos;

			short3 who_am_I_to_corners = p_who_am_I_to_corners[iMinor];
			LONG3 cornerindex = p_tri_cornerindex[iMinor];
			// each corner we want to pick up 3 values off n_shards, as well as n_cent.
			// The three values will not always be contiguous!!!

			// Let's make life easier and load up an array of 6 n's beforehand.
			f64 n_array[6];

			short who_am_I = who_am_I_to_corners[0];
			short tri_len = p_info[cornerindex.i1 + BEGINNING_OF_CENTRAL].tri_len;

			if ((cornerindex.i1 >= StartMajor) && (cornerindex.i1 < EndMajor))
			{
				short who_prev = who_am_I - 1;
				if (who_prev < 0) who_prev = tri_len - 1;
				// Worry about pathological cases later.
				n_array[0] = THIRD*(shared_n_shards[cornerindex.i1 - StartMajor].n[who_prev]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n_cent);
				short who_next = who_am_I + 1;
				if (who_next == tri_len) who_next = 0;
				n_array[1] = THIRD*(shared_n_shards[cornerindex.i1 - StartMajor].n[who_next]
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
					n_array[0] = THIRD*(p_n_shards[cornerindex.i1].n[who_prev] + temp.x + ncent);
					n_array[1] = THIRD*(temp.x + temp.y + ncent);
				}
				else {
					short who_next = who_am_I + 1;
					if (who_next == tri_len) {
						f64_vec2 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i1].n[who_prev]), sizeof(f64_vec2));
						n_array[0] = THIRD*(temp.x + temp.y + ncent);
						n_array[1] = THIRD*(p_n_shards[cornerindex.i1].n[0] + temp.y + ncent);
					}
					else {
						// typical case
						f64_vec3 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i1].n[who_prev]), sizeof(f64) * 3);
						n_array[0] = THIRD*(temp.x + temp.y + ncent);
						n_array[1] = THIRD*(temp.z + temp.y + ncent);
					};
				};
			}

			who_am_I = who_am_I_to_corners[1];
			tri_len = p_info[cornerindex.i2 + BEGINNING_OF_CENTRAL].tri_len;

			if ((cornerindex.i2 >= StartMajor) && (cornerindex.i2 < EndMajor))
			{
				short who_prev = who_am_I - 1;
				if (who_prev < 0) who_prev = tri_len - 1;
				// Worry about pathological cases later.
				n_array[2] = THIRD*(shared_n_shards[cornerindex.i2 - StartMajor].n[who_prev]
					+ shared_n_shards[cornerindex.i2 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i2 - StartMajor].n_cent);
				short who_next = who_am_I + 1;
				if (who_next == tri_len) who_next = 0;
				n_array[3] = THIRD*(shared_n_shards[cornerindex.i2 - StartMajor].n[who_next]
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
					n_array[2] = THIRD*(p_n_shards[cornerindex.i2].n[who_prev] + temp.x + ncent);
					n_array[3] = THIRD*(temp.x + temp.y + ncent);
				}
				else {
					short who_next = who_am_I + 1;
					if (who_next == tri_len) {
						f64_vec2 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i2].n[who_prev]), sizeof(f64_vec2));
						n_array[2] = THIRD*(temp.x + temp.y + ncent);
						n_array[3] = THIRD*(p_n_shards[cornerindex.i2].n[0] + temp.y + ncent);
					}
					else {
						// typical case
						f64_vec3 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i2].n[who_prev]), sizeof(f64) * 3);
						n_array[2] = THIRD*(temp.x + temp.y + ncent);
						n_array[3] = THIRD*(temp.z + temp.y + ncent);
					};
				};
			}

			who_am_I = who_am_I_to_corners[2];
			tri_len = p_info[cornerindex.i3 + BEGINNING_OF_CENTRAL].tri_len;

			if ((cornerindex.i3 >= StartMajor) && (cornerindex.i3 < EndMajor))
			{
				short who_prev = who_am_I - 1;
				if (who_prev < 0) who_prev = tri_len - 1;
				// Worry about pathological cases later.
				n_array[4] = THIRD*(shared_n_shards[cornerindex.i3 - StartMajor].n[who_prev]
					+ shared_n_shards[cornerindex.i3 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i3 - StartMajor].n_cent);
				short who_next = who_am_I + 1;
				if (who_next == tri_len) who_next = 0;
				n_array[5] = THIRD*(shared_n_shards[cornerindex.i3 - StartMajor].n[who_next]
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
					n_array[4] = THIRD*(p_n_shards[cornerindex.i3].n[who_prev] + temp.x + ncent);
					n_array[5] = THIRD*(temp.x + temp.y + ncent);
				}
				else {
					short who_next = who_am_I + 1;
					if (who_next == tri_len) {
						f64_vec2 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i3].n[who_prev]), sizeof(f64_vec2));
						n_array[4] = THIRD*(temp.x + temp.y + ncent);
						n_array[5] = THIRD*(p_n_shards[cornerindex.i3].n[0] + temp.y + ncent);
					}
					else {
						// typical case
						f64_vec3 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i3].n[who_prev]), sizeof(f64) * 3);
						n_array[4] = THIRD*(temp.x + temp.y + ncent);
						n_array[5] = THIRD*(temp.z + temp.y + ncent);
					};
				};
			}

			//
			Check numbering of 0..5 for neighminor: first is edge 0 ? corner 0 ?

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
							nextpos = p_info[izNeighMinor[inext]].pos;
						};
					};
					if (szPBC[inext] == ROTATE_ME_CLOCKWISE) nextpos = Clockwise*nextpos;
					if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) nextpos = Anticlockwise*nextpos;

					// New definition of endpoint of minor edge:

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
					area_quadrilateral = 0.5*(
						(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
						+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
						+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
						+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
						);

					//f64_vec2 grad_Az = integ_grad_Az / area_quadrilateral;
					Our_integral_Lap_Az += integ_grad_Az.dot(edge_normal) / area_quadrilateral;

					T3 T0, T1; // waste of registers
					f64 n1;
					T0.Te = THIRD* (prevT.Te + ourT.Te + oppT.Te);
					T1.Te = THIRD * (nextT.Te + ourT.Te + oppT.Te);
					T0.Ti = THIRD * (prevT.Ti + ourT.Ti + oppT.Ti);
					T1.Ti = THIRD * (nextT.Ti + ourT.Ti + oppT.Ti);

					// Where to get n?

					n1 = n_array[inext]; // !

					// *********
					// Verify that tri 0 is formed from our vertex, neigh 0 and neigh 1;
					// *********

					// To get integral grad we add the averages along the edges times edgenormals
					ownrates.ion -= Make3(0.5*(n0 * T0.Ti + n1 * T1.Ti)*over_m_i*edge_normal, 0.0);
					ownrates.elec -= Make3(0.5*(n0 * T0.Te + n1 * T1.Te)*over_m_e*edge_normal, 0.0);

					Az_edge = SIXTH * (2.0*ourdata.Az + 2.0*oppAz + prevAz + nextAz);
					Azdot_edge = SIXTH * (2.0*ourAzdot + 2.0*oppAzdot + prevAzdot + nextAzdot);
					Our_integral_grad_Azdot += Azdot_edge * edge_normal;
					Our_integral_grad_Az += Az_edge * edge_normal;
					Our_integral_curl_Az += Az_edge * (endpt1 - endpt0);
					Our_integral_grad_Te += 0.5*(T0.Te + T1.Te) * edge_normal;

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
				};

			GradAz[iMinor] = Our_integral_grad_Az / AreaMinor;
			LapAzArray[iMinor] = Our_integral_Lap_Az / AreaMinor;
			GradTeArray[iMinor] = Our_integral_grad_Te / AreaMinor;
			p_B[iMinor] = Make3(Our_integral_curl_Az / AreaMinor, BZ_CONSTANT);
			AreaMinorArray[iMinor] = AreaMinor;

			// wow :
			f64_vec2 overall_v_ours = p_overall_v_minor[iMinor];
			ROCAzduetoAdvection[iMinor] = overall_v_ours.dot(Our_integral_grad_Az / AreaMinor);
			ROCAzdotduetoAdvection[iMinor] = overall_v_ours.dot(Our_integral_grad_Azdot / AreaMinor);

			// No neutral stuff in this kernel, momrates should be set now:
			memcpy(pAdditionalMomRates + iMinor, &ownrates, sizeof(three_vec3));
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
					prevpos = p_info[izNeighMinor[iprev]].pos;
				};
			};
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) prevpos = Clockwise*prevpos;
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) prevpos = Anticlockwise*prevpos;

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
					opppos = p_info[izNeighMinor[i]].pos;
				};
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) opppos = Clockwise*opppos;
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) opppos = Anticlockwise*opppos;

						
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
							nextpos = p_info[izNeighMinor[inext]].pos;
						};
					};
					if (szPBC[inext] == ROTATE_ME_CLOCKWISE) nextpos = Clockwise*nextpos;
					if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) nextpos = Anticlockwise*nextpos;

					// New definition of endpoint of minor edge:

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
					area_quadrilateral = 0.5*(
						(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
						+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
						+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
						+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
						);

					//f64_vec2 grad_Az = integ_grad_Az / area_quadrilateral;
					Our_integral_Lap_Az += integ_grad_Az.dot(edge_normal) / area_quadrilateral;

					Az_edge = SIXTH * (2.0*ourdata.Az + 2.0*oppAz + prevAz + nextAz);
					Azdot_edge = SIXTH * (2.0*ourAzdot + 2.0*oppAzdot + prevAzdot + nextAzdot);
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

			GradAz[iMinor] = Our_integral_grad_Az / AreaMinor;
			LapAzArray[iMinor] = Our_integral_Lap_Az / AreaMinor;
			p_B[iMinor] = Make3(Our_integral_curl_Az / AreaMinor, BZ_CONSTANT);
			AreaMinorArray[iMinor] = AreaMinor;

			ROCAzduetoAdvection[iMinor] = 0.0;
			ROCAzdotduetoAdvection[iMinor] = 0.0;
		} // non-domain tri
	}; // was it FRILL
	
	// Okay. While we have n_shards in memory we could proceed to overwrite with vxy.
	// But get running first before using union and checking same.
}

	// . * Go back and sort out non-domain vertex, domain vs non-domain triangle.
	// ^^ done
	
	// . * Do vxy
	// . * Do neutral variants. n_n T_n, n_n v_n v_n_rel

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
)
{
	__shared__ v4 shared_vie[threadsPerTileMinor];
	__shared__ f64_vec2 shared_v_overall[threadsPerTileMinor];
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];
	__shared__ ShardModel shared_n_shards[threadsPerTileMajor];
	__shared__ T2 shared_vie_verts[threadsPerTileMajor];
	__shared__ f64_vec2 shared_v_overall[threadsPerTileMajor];
	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajor];

	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	long const iVertex = threadsPerTileMajor*blockIdx.x + threadIdx.x; // only meaningful threadIdx.x < threadsPerTileMajor

	long const StartMinor = threadsPerTileMinor*blockIdx.x;
	long const StartMajor = threadsPerTileMajor*blockIdx.x;
	long const EndMinor = StartMinor + threadsPerTileMinor;
	long const EndMajor = StartMajor + threadsPerTileMajor;

	shared_pos[threadIdx.x] = p_info[iMinor].pos;
	shared_vie = p_vie_minor[iMinor];
	shared_v_overall = p_v_overall_minor[iMinor];

	// Perhaps the real answer is this. Advection and therefore advective momflux
	// do not need to be recalculated very often at all. At 1e6 cm/s, we aim for 1 micron,
	// get 1e-10s to actually do the advection !!
	// So an outer cycle. Still limiting the number of total things in a minor tile. We might like 384 = 192*2.

	structural info;
	if (threadIdx.x < threadsPerTileMajor) {
		info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];
		shared_pos_verts[threadIdx.x] = info.pos;
		if (info.flag == DOMAIN_VERTEX) {
			memcpy(&(shared_n_shards[threadIdx.x]), p_n_shards[iVertex], sizeof(ShardModel)); // + 13
			shared_vie_verts[threadIdx.x] = p_vie_minor[iVertex + BEGINNING_OF_CENTRAL];
			shared_v_overall[threadIdx.x] = p_v_overall_minor[iVertex + BEGINNING_OF_CENTRAL];
		}
		else {
			// it still manages to coalesce, let's hope, because ShardModel is 13 doubles not 12.
			memset(&(shared_n_shards[threadIdx.x]), 0, sizeof(ShardModel)); // + 13
			memset(&(shared_vie_verts[threadIdx.x]), 0, sizeof(v4));
			memset(&(shared_v_overall[threadIdx.x]), 0, sizeof(f64_vec2));
		};
	};

	__syncthreads();

	v4 our_v, opp_v, prev_v, next_v;
	f64_vec2 our_v_overall, prev_v_overall, next_v_overall;
	
	if (threadIdx.x < threadsPerTileMajor) {

		AreaMinor = 0.0;
		three_vec3 ownrates;
		memcpy(&ownrates, p_AdditionalMomrates[iVertex + BEGINNING_OF_CENTRAL], sizeof(three_vec3));

		// Now bear in mind:
		// We will often have to do the viscosity calc, and using minor cells.
		// What a bugger.
		// Almost certainly requires lots of stuff like n,T,B. Accept some bus loading.
		// Cross that bridge when we come to it.
		// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

		long izTri[MAXNEIGH];
		char szPBC[MAXNEIGH];
		short tri_len = info.neigh_len;

		memcpy(izTri, long_array_of_izTri + index*MAXNEIGH, MAXNEIGH * sizeof(long));
		memcpy(szPBC, long_array_of_szPBC + index*MAXNEIGH, MAXNEIGH * sizeof(char));)

		our_v = shared_vie[threadIdx.x];
		our_v_overall = shared_v_overall[threadIdx.x];

		if (info.flag == DOMAIN_VERTEX) {
			
			short iprev = tri_len - 1;
			if ((izTri[iprev] >= StartMinor) && (izTri[iprev] < EndMinor))
			{
				prev_v = shared_vie[izTri[iprev] - StartMinor];
				prev_v_overall = shared_v_overall[izTri[iprev] - StartMinor];
				prevpos = shared_pos[izTri[iprev] - StartMinor];
			}
			else {
				prev_v = p_vie_minor[izTri[iprev]];
				prev_v_overall = p_v_overall_minor[izTri[iprev]];
				prevpos = p_info[izTri[iprev]].pos;
			}
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise*prevpos;
				prev_v = Clockwise*prev_v;
				prev_v_overall = Clockwise*prev_v_overall;
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise*prevpos;
				prev_v = Anticlockwise*prev_v;
				prev_v_overall = Anticlockwise*prev_v_overall;
			}

			short i = 0;
			if ((izTri[i] >= StartMinor) && (izTri[i] < EndMinor))
			{
				opp_v = shared_vie[izTri[i] - StartMinor];
				opp_v_overall = shared_v_overall[izTri[iprev] - StartMinor];
				opppos = shared_pos[izTri[i] - StartMinor];				
			}
			else {
				opp_v = p_vie_minor[izTri[i]];
				opp_v_overall = p_v_overall_minor[izTri[i]];
				opppos = p_info[izTri[i]].pos;
			}
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise*opppos;
				opp_v = Clockwise*opp_v;
				opp_v_overall = Clockwise*opp_v_overall;
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise*opppos;
				opp_v = Anticlockwise*opp_v;
				opp_v_overall = Anticlockwise*opp_v_overall;
			}

			// Think carefully: DOMAIN vertex cases for n,T ...
			f64 n0 = shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[iprev] + shared_n_shards[threadIdx.x].n_cent;
			endpt0 = THIRD * (info.pos + opppos + prevpos);

			short iend = tri_len;
			f64_vec2 projendpt0;
			if ((pVertex->flags == INNERMOST) || (pVertex->flags == OUTERMOST)) {
				iend = tri_len - 2;
				if (pVertex->flags == OUTERMOST) {
					endpt0.project_to_radius(projendpt0, OutermostFrillCentroidRadius); // back of cell for Lap purposes
				}
				else {
					endpt0.project_to_radius(projendpt0, InnermostFrillCentroidRadius); // back of cell for Lap purposes
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
					nextpos = p_info[izTri[inext]].pos;
				}
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise*nextpos;
					next_v = Clockwise*next_v;
					next_v_overall = Clockwise*next_v_overall;
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise*nextpos;
					next_v = Anticlockwise*next_v;
					next_v_overall = Anticlockwise*next_v_overall;
				}

				endpt1 = THIRD * (nextpos + ourpos + opppos);
				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

				// ______________________________________________________-

				f64 n1;
				n1 = shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[inext] + shared_n_shards[threadIdx.x].n_cent;

				// Assume neighs 0,1 are relevant to border with tri 0 minor.
				// *********
				Verify that tri 0 is formed from our vertex, neigh 0 and neigh 1;
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

				ownrates.ion -= 0.5*relvnormal*
					(n0 * (Make3(vxy0 - our_v.vxy, viz0 - our_v.viz))
						+ n1 * (Make3(vxy1 - our_v.vxy, viz1 - our_v.viz)));
				ownrates.elec -= 0.5*relvnormal*
					(n0 * (Make3(vxy0 - our_v.vxy, vez0 - our_v.vez))
						+ n1 * (Make3(vxy1 - our_v.vxy, vez1 - our_v.vez)));

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

			
			// No neutral stuff in this kernel, momrates should be set now:
			memcpy(pAdditionalMomRates + iVertex + BEGINNING_OF_CENTRAL, &ownrates, sizeof(three_vec3));
		}
		else {
			// NOT domain vertex: Do nothing
		};
	}; // was it domain vertex or Az-only
	   // This branching is itself a good argument for doing Az in ITS own separate routine with no need for n_shard.

	// __syncthreads(); // end of first vertex part
	// Do we need syncthreads? Not overwriting any shared data here...

	// now the minor with n_ion part:
	info = p_info[iMinor];
	our_v = shared_vie[threadIdx.x];

	long izNeighMinor[6];
	char szPBC[6];
	memcpy(izNeighMinor, p_izneighminor + iMinor * 6, sizeof(long) * 6);
	memcpy(szPBC, p_szPBCtriminor + iMinor * 6, sizeof(char) * 6);

	v4 prev_v, next_v, opp_v;
	three_vec3 ownrates_minor;
	memcpy(&ownrates_minor, p_AdditionalMomRates + iMinor, sizeof(three_vec3));
	// this is not a clever way of doing it. Want more careful.

	if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL)) {
	} else {

		AreaMinor = 0.0;
		if ((info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS)) {


			iprev = 5; i = 0;
			if ((izNeighMinor[iprev] >= StartMinor) && (izNeighMinor[iprev] < EndMinor))
			{
				prev_v = shared_vie[izNeighMinor[iprev] - StartMinor];
				prevpos = shared_pos[izNeighMinor[iprev] - StartMinor];
				prev_v_overall = shared_v_overall[izNeighMinor[iprev] - StartMinor];
			}
			else {
				if ((izNeighMinor[iprev] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[iprev] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					prev_v = shared_vie_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
					prev_v_overall = shared_v_overall_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
					prevpos = shared_pos_verts[izNeighMinor[iprev] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					prevpos = p_info[izNeighMinor[iprev]].pos;
					prev_v = p_vie_minor[izNeighMinor[iprev]];
					prev_v_overall = p_v_overall_minor[izNeighMinor[iprev]];
				};
			};
			if (szPBC[iprev] == ROTATE_ME_CLOCKWISE) {
				prevpos = Clockwise*prevpos;
				prev_v = Clockwise*prev_v;
				prev_v_overall = Clockwise*prev_v_overall;
			}
			if (szPBC[iprev] == ROTATE_ME_ANTICLOCKWISE) {
				prevpos = Anticlockwise*prevpos;
				prev_v = Anticlockwise*prev_v;
				prev_v_overall = Anticlockwise*prev_v_overall;
			}



			i = 0;
			if ((izNeighMinor[i] >= StartMinor) && (izNeighMinor[i] < EndMinor))
			{
				opp_v = shared_vie[izNeighMinor[i] - StartMinor];
				opppos = shared_pos[izNeighMinor[i] - StartMinor];
				opp_v_overall = shared_v_overall[izNeighMinor[i] - StartMinor];
			}
			else {
				if ((izNeighMinor[i] >= StartMajor + BEGINNING_OF_CENTRAL) &&
					(izNeighMinor[i] < EndMajor + BEGINNING_OF_CENTRAL))
				{
					opp_v = shared_vie_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
					opp_v_overall = shared_v_overall_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
					opppos = shared_pos_verts[izNeighMinor[i] - BEGINNING_OF_CENTRAL - StartMajor];
				}
				else {
					opppos = p_info[izNeighMinor[i]].pos;
					opp_v = p_vie_minor[izNeighMinor[i]];
					opp_v_overall = p_v_overall_minor[izNeighMinor[i]];
				};
			};
			if (szPBC[i] == ROTATE_ME_CLOCKWISE) {
				opppos = Clockwise*opppos;
				opp_v = Clockwise*opp_v;
				opp_v_overall = Clockwise*opp_v_overall;
			}
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				opppos = Anticlockwise*opppos;
				opp_v = Anticlockwise*opp_v;
				opp_v_overall = Anticlockwise*opp_v_overall;
			}

			short3 who_am_I_to_corners = p_who_am_I_to_corners[iMinor];
			LONG3 cornerindex = p_tri_cornerindex[iMinor];
			// each corner we want to pick up 3 values off n_shards, as well as n_cent.
			// The three values will not always be contiguous!!!

			// Let's make life easier and load up an array of 6 n's beforehand.
			f64 n_array[6];

			short who_am_I = who_am_I_to_corners[0];
			short tri_len = p_info[cornerindex.i1 + BEGINNING_OF_CENTRAL].tri_len;

			if ((cornerindex.i1 >= StartMajor) && (cornerindex.i1 < EndMajor))
			{
				short who_prev = who_am_I - 1;
				if (who_prev < 0) who_prev = tri_len - 1;
				// Worry about pathological cases later.
				n_array[0] = THIRD*(shared_n_shards[cornerindex.i1 - StartMajor].n[who_prev]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n_cent);
				short who_next = who_am_I + 1;
				if (who_next == tri_len) who_next = 0;
				n_array[1] = THIRD*(shared_n_shards[cornerindex.i1 - StartMajor].n[who_next]
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
					n_array[0] = THIRD*(p_n_shards[cornerindex.i1].n[who_prev] + temp.x + ncent);
					n_array[1] = THIRD*(temp.x + temp.y + ncent);
				}
				else {
					short who_next = who_am_I + 1;
					if (who_next == tri_len) {
						f64_vec2 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i1].n[who_prev]), sizeof(f64_vec2));
						n_array[0] = THIRD*(temp.x + temp.y + ncent);
						n_array[1] = THIRD*(p_n_shards[cornerindex.i1].n[0] + temp.y + ncent);
					}
					else {
						// typical case
						f64_vec3 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i1].n[who_prev]), sizeof(f64) * 3);
						n_array[0] = THIRD*(temp.x + temp.y + ncent);
						n_array[1] = THIRD*(temp.z + temp.y + ncent);
					};
				};
			}

			who_am_I = who_am_I_to_corners[1];
			tri_len = p_info[cornerindex.i2 + BEGINNING_OF_CENTRAL].tri_len;

			if ((cornerindex.i2 >= StartMajor) && (cornerindex.i2 < EndMajor))
			{
				short who_prev = who_am_I - 1;
				if (who_prev < 0) who_prev = tri_len - 1;
				// Worry about pathological cases later.
				n_array[2] = THIRD*(shared_n_shards[cornerindex.i2 - StartMajor].n[who_prev]
					+ shared_n_shards[cornerindex.i2 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i2 - StartMajor].n_cent);
				short who_next = who_am_I + 1;
				if (who_next == tri_len) who_next = 0;
				n_array[3] = THIRD*(shared_n_shards[cornerindex.i2 - StartMajor].n[who_next]
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
					n_array[2] = THIRD*(p_n_shards[cornerindex.i2].n[who_prev] + temp.x + ncent);
					n_array[3] = THIRD*(temp.x + temp.y + ncent);
				}
				else {
					short who_next = who_am_I + 1;
					if (who_next == tri_len) {
						f64_vec2 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i2].n[who_prev]), sizeof(f64_vec2));
						n_array[2] = THIRD*(temp.x + temp.y + ncent);
						n_array[3] = THIRD*(p_n_shards[cornerindex.i2].n[0] + temp.y + ncent);
					}
					else {
						// typical case
						f64_vec3 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i2].n[who_prev]), sizeof(f64) * 3);
						n_array[2] = THIRD*(temp.x + temp.y + ncent);
						n_array[3] = THIRD*(temp.z + temp.y + ncent);
					};
				};
			}

			who_am_I = who_am_I_to_corners[2];
			tri_len = p_info[cornerindex.i3 + BEGINNING_OF_CENTRAL].tri_len;

			if ((cornerindex.i3 >= StartMajor) && (cornerindex.i3 < EndMajor))
			{
				short who_prev = who_am_I - 1;
				if (who_prev < 0) who_prev = tri_len - 1;
				// Worry about pathological cases later.
				n_array[4] = THIRD*(shared_n_shards[cornerindex.i3 - StartMajor].n[who_prev]
					+ shared_n_shards[cornerindex.i3 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i3 - StartMajor].n_cent);
				short who_next = who_am_I + 1;
				if (who_next == tri_len) who_next = 0;
				n_array[5] = THIRD*(shared_n_shards[cornerindex.i3 - StartMajor].n[who_next]
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
					n_array[4] = THIRD*(p_n_shards[cornerindex.i3].n[who_prev] + temp.x + ncent);
					n_array[5] = THIRD*(temp.x + temp.y + ncent);
				}
				else {
					short who_next = who_am_I + 1;
					if (who_next == tri_len) {
						f64_vec2 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i3].n[who_prev]), sizeof(f64_vec2));
						n_array[4] = THIRD*(temp.x + temp.y + ncent);
						n_array[5] = THIRD*(p_n_shards[cornerindex.i3].n[0] + temp.y + ncent);
					}
					else {
						// typical case
						f64_vec3 temp;
						memcpy(&temp, &(p_n_shards[cornerindex.i3].n[who_prev]), sizeof(f64) * 3);
						n_array[4] = THIRD*(temp.x + temp.y + ncent);
						n_array[5] = THIRD*(temp.z + temp.y + ncent);
					};
				};
			}

			//
			Check numbering of 0..5 for neighminor: first is edge 0 ? corner 0 ?

#pragma unroll 
			for (i = 0; i < 6; i++)
			{
				inext = i + 1; if (inext > 5) inext = 0;
				if ((izNeighMinor[inext] >= StartMinor) && (izNeighMinor[inext] < EndMinor))
				{
					next_v = shared_vie[izNeighMinor[inext] - StartMinor];
					nextpos = shared_pos[izNeighMinor[inext] - StartMinor];
					next_v_overall = shared_v_overall[izNeighMinor[inext] - StartMinor];
				}
				else {
					if ((izNeighMinor[inext] >= StartMajor + BEGINNING_OF_CENTRAL) &&
						(izNeighMinor[inext] < EndMajor + BEGINNING_OF_CENTRAL))
					{
						next_v = shared_vie_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
						next_v_overall = shared_v_overall_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
						nextpos = shared_pos_verts[izNeighMinor[inext] - BEGINNING_OF_CENTRAL - StartMajor];
					}
					else {
						nextpos = p_info[izNeighMinor[inext]].pos;
						next_v = p_vie_minor[izNeighMinor[inext]];
						next_v_overall = p_v_overall_minor[izNeighMinor[inext]];
					};
				};
				if (szPBC[inext] == ROTATE_ME_CLOCKWISE) {
					nextpos = Clockwise*nextpos;
					next_v = Clockwise*next_v;
					next_v_overall = Clockwise*next_v_overall;
				}
				if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) {
					nextpos = Anticlockwise*nextpos;
					next_v = Anticlockwise*next_v;
					next_v_overall = Anticlockwise*next_v_overall;
				}

				// New definition of endpoint of minor edge:

				endpt0 = THIRD * (prevpos + info.pos + opppos);
				endpt1 = THIRD * (nextpos + info.pos + opppos);

				edge_normal.x = endpt1.y - endpt0.y;
				edge_normal.y = endpt0.x - endpt1.x;

				// ______________________________________________________-
				n1 = shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[inext] + shared_n_shards[threadIdx.x].n_cent;

				// Assume neighs 0,1 are relevant to border with tri 0 minor.
				// *********
				Verify that tri 0 is formed from our vertex, neigh 0 and neigh 1;
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

				ownrates.ion -= 0.5*relvnormal*
					(n0 * (Make3(vxy0 - our_v.vxy, viz0 - our_v.viz))
						+ n1 * (Make3(vxy1 - our_v.vxy, viz1 - our_v.viz)));
				ownrates.elec -= 0.5*relvnormal*
					(n0 * (Make3(vxy0 - our_v.vxy, vez0 - our_v.vez))
						+ n1 * (Make3(vxy1 - our_v.vxy, vez1 - our_v.vez)));

				endpt0 = endpt1;
				n0 = n1;

				prevpos = opppos;
				prev_v = opp_v;
				prev_v_overall = opp_v_overall;

				opppos = nextpos;
				opp_v = next_v;
				opp_v_overall = next_v_overall;
			};
			f64_vec2 overall_v_ours = p_overall_v_minor[iMinor];
			
			memcpy(pAdditionalMomRates + iMinor, &ownrates, sizeof(three_vec3));
		}
		else {
			// Not domain, not crossing_ins, not a frill
			// ==========================================
		} // non-domain tri
	}; // was it FRILL
}

// Neutral routine:

// Sort everything else out first and then this. Just a copy of the above routine
// but with 3-vector v_n + Tn in place of 4-vector vie, doing pressure and momflux.

__global__ void kernelNeutral_pressure_and_momflux(
	structural * __restrict__ p_info_minor,
	f64_vec3 * __restrict__ p_v_n,
	ShardModel * __restrict__ p_n_shards_n,
	f64_vec2 * __restrict__ p_v_overall,
	three_vec3 * __restrict__ p_AdditionalMomRates
)
{

}


// .. Compare with previous CUDA code to do the alloc & transfer to video memory.
// .. Set up all the constants we want on device. Check through kernels in sequence of calls.
// .. We probably didn't yet fill in feint accel. That should be pretty quick.
// .. Check every line.
// .. 