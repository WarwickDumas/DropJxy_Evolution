#include "kernel.h"
#include "vector_tensor.cu"
#include "cuda_struct.h"
#include "helpers.cu"
#include "constant.h"
#include "FFxtubes.h"

#define FOUR_PI 12.5663706143592


__global__ void kernelCalculateOverallVelocitiesVertices(
	structural * __restrict__ p_info_major,
	v4 * __restrict__ p_vie_major,
	f64_vec3 * __restrict__ p_v_n_major,
	nvals * __restrict__ p_n_major,
	f64_vec2 * __restrict__ p_v_overall_major)
{
	long const index = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX
	structural const info = p_info_major[index];
	f64_vec2 v_overall(0.0, 0.0);

	if (info.flag == DOMAIN_VERTEX) {
		v4 const vie = p_vie_major[index];
		f64_vec3 const v_n = p_v_n_major[index];
		nvals const n = p_n_major[index];
		
		v_overall = (vie.vxy*(m_e + m_i)*n.n +
			v_n.xypart()*m_n*n.n_n) /
			((m_e + m_i)*n.n + m_n*n.n_n);
		if (index == 12078) {
			printf("index %d v_overall %1.10E %1.10E vxy %1.10E %1.10E n %1.10E %1.10E v_n %1.10E %1.10E \n",
				index, v_overall.x, v_overall.y, vie.vxy.x, vie.vxy.y, n.n, n.n_n, v_n.x, v_n.y);
			printf("m_e %1.10E m_i %1.10E m_n %1.10E numer %1.10E denom %1.10E billericay %1.10E",
				m_e, m_i, m_n,
				(vie.vxy*(m_e + m_i)*n.n +
					v_n.xypart()*m_n*n.n_n).x,
					((m_e + m_i)*n.n + m_n*n.n_n),
				billericay);
		}
	};
	p_v_overall_major[index] = v_overall;
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

	long const index = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX
	if (threadIdx.x < threadsPerTileMajor)
	{
		long getindex = blockIdx.x * threadsPerTileMajor + threadIdx.x;
		shared_v[threadIdx.x] = p_overall_v_major[getindex];
	};
	long const StartMajor = blockIdx.x*threadsPerTileMajor;
	long const EndMajor = StartMajor + threadsPerTileMajor;
	LONG3 const tri_corner_index = p_tri_corner_index[index];
	CHAR4 const tri_corner_per_flag = p_tri_periodic_corner_flags[index];
	structural info = p_info[index];

	__syncthreads();

	f64_vec2 v(0.0, 0.0);

	if ( (info.flag == DOMAIN_TRIANGLE) || (info.flag == CROSSING_INS) ){

		f64_vec2 vcorner;
		if ((tri_corner_index.i1 >= StartMajor) && (tri_corner_index.i1 < EndMajor))
		{
			vcorner = THIRD*shared_v[tri_corner_index.i1 - StartMajor];
		}
		else {
			vcorner = THIRD*p_overall_v_major[tri_corner_index.i1];
		};
		if (tri_corner_per_flag.per0 == ROTATE_ME_CLOCKWISE) vcorner = Clockwise_d*vcorner;
		if (tri_corner_per_flag.per0 == ROTATE_ME_ANTICLOCKWISE) vcorner = Anticlockwise_d*vcorner;
		v += vcorner;

		if ((tri_corner_index.i2 >= StartMajor) && (tri_corner_index.i2 < EndMajor))
		{
			vcorner = THIRD*shared_v[tri_corner_index.i2 - StartMajor];
		}
		else {
			vcorner = THIRD*p_overall_v_major[tri_corner_index.i2];
		};
		if (tri_corner_per_flag.per1 == ROTATE_ME_CLOCKWISE) vcorner = Clockwise_d*vcorner;
		if (tri_corner_per_flag.per1 == ROTATE_ME_ANTICLOCKWISE) vcorner = Anticlockwise_d*vcorner;
		v += vcorner;

		if ((tri_corner_index.i3 >= StartMajor) && (tri_corner_index.i3 < EndMajor))
		{
			vcorner = THIRD*shared_v[tri_corner_index.i3 - StartMajor];
		}
		else {
			vcorner = THIRD*p_overall_v_major[tri_corner_index.i3];
		};
		if (tri_corner_per_flag.per2 == ROTATE_ME_CLOCKWISE) vcorner = Clockwise_d*vcorner;
		if (tri_corner_per_flag.per2 == ROTATE_ME_ANTICLOCKWISE) vcorner = Anticlockwise_d*vcorner;
		v += vcorner;

		if (info.flag == CROSSING_INS) {
			// Position is equal to 1/3 avg, projected to ins.				
			// So if we are moving 2 points to the right, it only moves 2/3 as much.

			// Now remove the radial component:
			f64_vec2 r = info.pos;
			//rhat = r / r.modulus();
			//p_v[iMinor] -= rhat.dot(p_v[iMinor])*rhat;
			v = v - r*(r.dot(v)) / (r.x*r.x + r.y*r.y);

		};
	} else {
		// leave it == 0		
	};
	p_overall_v_minor[index] = v;
}


__global__ void kernelAdvectPositions(
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
	long const index = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX
	structural info = p_info_src[index];
	f64_vec2 overall_v = p_v_overall_minor[index];
	info.pos += h_use*overall_v;
	p_info_dest[index] = info;
}

__global__ void kernelAverage_n_T_x_to_tris(
	nvals * __restrict__ p_n_minor,
	nvals * __restrict__ p_n_major,
	T3 * __restrict__ p_T_minor,
	structural * __restrict__ p_info,
	LONG3 * __restrict__ p_tri_corner_index,
	CHAR4 * __restrict__ p_tri_periodic_corner_flags
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
	CHAR4 const tri_corner_per_flag = p_tri_periodic_corner_flags[index];
	structural info = p_info[index];

	__syncthreads();

	T3 T(0.0, 0.0, 0.0);
	nvals n(0.0, 0.0);
	f64_vec2 pos(0.0, 0.0);

	if (info.flag == DOMAIN_TRIANGLE) {

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
			T += THIRD*p_T_minor[tri_corner_index.i1 + BEGINNING_OF_CENTRAL];
		};
		if (tri_corner_per_flag.per0 == ROTATE_ME_CLOCKWISE) poscorner = Clockwise_d*poscorner;
		if (tri_corner_per_flag.per0 == ROTATE_ME_ANTICLOCKWISE) poscorner = Anticlockwise_d*poscorner;
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
			T += THIRD*p_T_minor[tri_corner_index.i2 + BEGINNING_OF_CENTRAL];
		};
		if (tri_corner_per_flag.per1 == ROTATE_ME_CLOCKWISE) poscorner = Clockwise_d*poscorner;
		if (tri_corner_per_flag.per1 == ROTATE_ME_ANTICLOCKWISE) poscorner = Anticlockwise_d*poscorner;
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
			T += THIRD*p_T_minor[tri_corner_index.i3 + BEGINNING_OF_CENTRAL];
		};
		if (tri_corner_per_flag.per2 == ROTATE_ME_CLOCKWISE) poscorner = Clockwise_d*poscorner;
		if (tri_corner_per_flag.per2 == ROTATE_ME_ANTICLOCKWISE) poscorner = Anticlockwise_d*poscorner;
		pos += poscorner;

		if (index == 29427) {
			printf("Domain 29427 n %1.5E %1.5E n3 %1.5E %1.5E \n", n.n, n.n_n,
				p_n_major[tri_corner_index.i3].n,p_n_major[tri_corner_index.i3]);
		}

	} else {
		// What else?
		if (info.flag == CROSSING_INS)
		{
			int iAbove = 0;
			f64_vec2 poscorner;
			if ((tri_corner_index.i1 >= StartMajor) && (tri_corner_index.i1 < EndMajor))
			{
				poscorner = THIRD*shared_pos[tri_corner_index.i1 - StartMajor];
				if (poscorner.dot(poscorner) > DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					n += shared_n[tri_corner_index.i1 - StartMajor];
					T += shared_T[tri_corner_index.i1 - StartMajor];
					iAbove++;
				};
			} else {
				poscorner = THIRD*p_info[tri_corner_index.i1 + BEGINNING_OF_CENTRAL].pos;
				if (poscorner.dot(poscorner) > DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					n += p_n_major[tri_corner_index.i1];
					T += p_T_minor[tri_corner_index.i1 + BEGINNING_OF_CENTRAL];
					iAbove++;
				}
			};
			if (tri_corner_per_flag.per0 == ROTATE_ME_CLOCKWISE) poscorner = Clockwise_d*poscorner;
			if (tri_corner_per_flag.per0 == ROTATE_ME_ANTICLOCKWISE) poscorner = Anticlockwise_d*poscorner;
			pos += poscorner;

			if ((tri_corner_index.i2 >= StartMajor) && (tri_corner_index.i2 < EndMajor))
			{
				poscorner = THIRD*shared_pos[tri_corner_index.i2 - StartMajor];
				if (poscorner.dot(poscorner) > DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					n += shared_n[tri_corner_index.i2 - StartMajor];
					T += shared_T[tri_corner_index.i2 - StartMajor];
					iAbove++;
				};
			} else {
				poscorner = THIRD*p_info[tri_corner_index.i2 + BEGINNING_OF_CENTRAL].pos;
				if (poscorner.dot(poscorner) > DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					n += p_n_major[tri_corner_index.i2];
					T += p_T_minor[tri_corner_index.i2 + BEGINNING_OF_CENTRAL];
					iAbove++;
				};
			};
			if (tri_corner_per_flag.per1 == ROTATE_ME_CLOCKWISE) poscorner = Clockwise_d*poscorner;
			if (tri_corner_per_flag.per1 == ROTATE_ME_ANTICLOCKWISE) poscorner = Anticlockwise_d*poscorner;
			pos += poscorner;

			if ((tri_corner_index.i3 >= StartMajor) && (tri_corner_index.i3 < EndMajor))
			{
				poscorner = THIRD*shared_pos[tri_corner_index.i3 - StartMajor];
				if (poscorner.dot(poscorner) > DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					n += shared_n[tri_corner_index.i3 - StartMajor];
					T += shared_T[tri_corner_index.i3 - StartMajor];
					iAbove++;
				};
			} else {
				poscorner = THIRD*p_info[tri_corner_index.i3 + BEGINNING_OF_CENTRAL].pos;
				if (poscorner.dot(poscorner) > DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
				{
					n += p_n_major[tri_corner_index.i3];
					T += p_T_minor[tri_corner_index.i3 + BEGINNING_OF_CENTRAL];
					iAbove++;
				};
			};
			if (tri_corner_per_flag.per2 == ROTATE_ME_CLOCKWISE) poscorner = Clockwise_d*poscorner;
			if (tri_corner_per_flag.per2 == ROTATE_ME_ANTICLOCKWISE) poscorner = Anticlockwise_d*poscorner;
			pos += poscorner;

			f64_vec2 pos2 = pos;
			pos2.project_to_radius(pos,DEVICE_RADIUS_INSULATOR_OUTER);
			f64 divide = 1.0 / (f64)iAbove;
			n.n *= divide;
			n.n_n *= divide;
			T.Tn *= divide;
			T.Ti *= divide;
			T.Te *= divide;

			if (index == 29427) {
				printf("Crossing ins 29427 n %1.5E %1.5E n3 %1.5E %1.5E \n", n.n, n.n_n,
					p_n_major[tri_corner_index.i3].n, p_n_major[tri_corner_index.i3]);
			}

		}
		else {
			n.n = 0.0;
			n.n_n = 0.0;
			T.Te = 0.0; T.Ti = 0.0; T.Tn = 0.0;

			f64_vec2 poscorner;
			if ((tri_corner_index.i1 >= StartMajor) && (tri_corner_index.i1 < EndMajor))
			{
				poscorner = THIRD*shared_pos[tri_corner_index.i1 - StartMajor];
			}
			else {
				poscorner = THIRD*p_info[tri_corner_index.i1 + BEGINNING_OF_CENTRAL].pos;
			};
			if (tri_corner_per_flag.per0 == ROTATE_ME_CLOCKWISE) poscorner = Clockwise_d*poscorner;
			if (tri_corner_per_flag.per0 == ROTATE_ME_ANTICLOCKWISE) poscorner = Anticlockwise_d*poscorner;
			pos += poscorner;

			if ((tri_corner_index.i2 >= StartMajor) && (tri_corner_index.i2 < EndMajor))
			{
				poscorner = THIRD*shared_pos[tri_corner_index.i2 - StartMajor];
			} else {
				poscorner = THIRD*p_info[tri_corner_index.i2 + BEGINNING_OF_CENTRAL].pos;
			};
			if (tri_corner_per_flag.per1 == ROTATE_ME_CLOCKWISE) poscorner = Clockwise_d*poscorner;
			if (tri_corner_per_flag.per1 == ROTATE_ME_ANTICLOCKWISE) poscorner = Anticlockwise_d*poscorner;
			pos += poscorner;

			if ((info.flag != INNER_FRILL) && (info.flag != OUTER_FRILL))
			{	
				if ((tri_corner_index.i3 >= StartMajor) && (tri_corner_index.i3 < EndMajor))
				{
					poscorner = THIRD*shared_pos[tri_corner_index.i3 - StartMajor];
				}
				else {
					poscorner = THIRD*p_info[tri_corner_index.i3 + BEGINNING_OF_CENTRAL].pos;
				};
				if (tri_corner_per_flag.per2 == ROTATE_ME_CLOCKWISE) poscorner = Clockwise_d*poscorner;
				if (tri_corner_per_flag.per2 == ROTATE_ME_ANTICLOCKWISE) poscorner = Anticlockwise_d*poscorner;
				pos += poscorner;
			} else {
				// FRILL
				pos *= 1.5;
				f64_vec2 pos2 = pos;
				if (info.flag == INNER_FRILL) {
					pos2.project_to_radius(pos, FRILL_CENTROID_INNER_RADIUS_d);
				} else {
					pos2.project_to_radius(pos, FRILL_CENTROID_OUTER_RADIUS_d);
				};
			}
			if (index == 29427) {
				printf("Non domain 29427 n %1.5E %1.5E n3 %1.5E %1.5E \n", n.n, n.n_n,
					p_n_major[tri_corner_index.i3].n, p_n_major[tri_corner_index.i3]);
			}			
		};
		// Outer frills it is thus set to n=0,T=0.
	};

	p_n_minor[index] = n;
	p_T_minor[index] = T;
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
	//	long * __restrict__ Tri_n_lists,
	//	long * __restrict__ Tri_n_n_lists	,
	f64 * __restrict__ p_AreaMajor)// sets n_shards_n, n_shards, Tri_n_n_lists, Tri_n_lists
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
	{
		structural info2[2];
		memcpy(info2, p_info_minor + blockIdx.x*threadsPerTileMinor + 2 * threadIdx.x, sizeof(structural) * 2);
		shared_pos[2 * threadIdx.x] = info2[0].pos;
		shared_pos[2 * threadIdx.x + 1] = info2[1].pos;
		memcpy(&(shared_n[2 * threadIdx.x]), p_n_minor + blockIdx.x*threadsPerTileMinor + 2 * threadIdx.x, sizeof(nvals) * 2);
	}
	long const StartMinor = blockIdx.x*threadsPerTileMinor; // vertex index
	long const EndMinor = StartMinor + threadsPerTileMinor;

	__syncthreads();

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

		f64 n_avg = p_n_minor[BEGINNING_OF_CENTRAL + iVertex].n;

		if ((izTri[0] >= StartMinor) && (izTri[0] < EndMinor)) {
			pos0 = shared_pos[izTri[0] - StartMinor];
			ndesire0 = shared_n[izTri[0] - StartMinor].n;
		}
		else {
			pos0 = p_info_minor[izTri[0]].pos;
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

			inext = i + 1; if (inext == info.neigh_len) inext = 0;
			if ((izTri[inext] >= StartMinor) && (izTri[inext] < EndMinor)) {
				pos1 = shared_pos[izTri[inext] - StartMinor];
				ndesire1 = shared_n[izTri[inext] - StartMinor].n;
			}
			else {
				pos1 = p_info_minor[izTri[inext]].pos;
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

		if ((n_avg > high_n) || (n_avg < low_n)) {
#pragma unroll MAXNEIGH
			for (i = 0; i < info.neigh_len; i++)
				n_.n[i] = n_avg;
			n_.n_cent = n_avg;

		}
		else {
			real n_C_need = (n_avg*AreaMajor - N0) / coeffcent;

			if ((n_C_need > low_n) && (n_C_need < high_n)) {
				n_.n_cent = n_C_need;

			}
			else {
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
									ndesire = shared_n[izTri[i] - StartMinor].n;
								}
								else {
									ndesire = p_n_minor[izTri[i]].n;
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
									ndesire = shared_n[izTri[i] - StartMinor].n;
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

		memcpy(&(p_n_shards[iVertex]), &n_, sizeof(ShardModel));

		// Now start again: neutrals

		n_avg = p_n_minor[BEGINNING_OF_CENTRAL + iVertex].n_n;

		if ((izTri[0] >= StartMinor) && (izTri[0] < EndMinor)) {
			pos0 = shared_pos[izTri[0] - StartMinor];
			ndesire0 = shared_n[izTri[0] - StartMinor].n_n;
		}
		else {
			pos0 = p_info_minor[izTri[0]].pos;
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
			}
			else {
				pos1 = p_info_minor[izTri[inext]].pos;
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

		}
		else {
			real n_C_need = (n_avg*AreaMajor - N0) / coeffcent;

			if ((n_C_need > low_n) && (n_C_need < high_n)) {
				n_.n_cent = n_C_need; // accept desired values

			}
			else {
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

	}
	else { // DOMAIN_VERTEX
		memset(&(p_n_shards[iVertex]), 0, sizeof(ShardModel));
		memset(&(p_n_n_shards[iVertex]), 0, sizeof(ShardModel));
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
	LONG3 * __restrict__ p_who_am_I_to_corner
) {
	// Assume that we do the simplest thing possible.

	long const index = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX
	structural info = p_info[index];
	nvals result;

	if (index >= BEGINNING_OF_CENTRAL)
	{
		if (info.flag == DOMAIN_VERTEX) {
			result.n = p_n_shards[index - BEGINNING_OF_CENTRAL].n_cent;
			result.n_n = p_n_shards_n[index - BEGINNING_OF_CENTRAL].n_cent;
			p_n_minor[index] = result;
		}
		else {
			// Outermost vertex?
			result.n = 0.0;
			result.n_n = 0.0;
			if (info.flag == OUTERMOST) {
				result.n = 1.0e18;
				result.n_n = 1.0e12;
			};
			p_n_minor[index] = result;
		}
	}
	else {
		if (info.flag == DOMAIN_TRIANGLE) {
			LONG3 tri_corner_index = p_tri_corner_index[index];
			LONG3 who_am_I_to_corner = p_who_am_I_to_corner[index];
			result.n = THIRD*
				(p_n_shards[tri_corner_index.i1].n[who_am_I_to_corner.i1]
					+ p_n_shards[tri_corner_index.i2].n[who_am_I_to_corner.i2]
					+ p_n_shards[tri_corner_index.i3].n[who_am_I_to_corner.i3]);
			result.n_n = THIRD*
				(p_n_shards_n[tri_corner_index.i1].n[who_am_I_to_corner.i1]
					+ p_n_shards_n[tri_corner_index.i2].n[who_am_I_to_corner.i2]
					+ p_n_shards_n[tri_corner_index.i3].n[who_am_I_to_corner.i3]);
			p_n_minor[index] = result;
		} else {
			if (info.flag == CROSSING_INS) {
				LONG3 tri_corner_index = p_tri_corner_index[index];
				LONG3 who_am_I_to_corner = p_who_am_I_to_corner[index];
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
				p_n_minor[index] = result;
			}
		}
	}
}

__global__ void kernelCalculateNu_eHeartNu_iHeart_nu_nn_visc(
	structural * __restrict__ p_info,
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
	long const index = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX
	structural info = p_info[index];
	if (info.flag == DOMAIN_VERTEX) {

		our_n = p_n[index]; // never used again once we have kappa
		T = p_T[index];

		TeV = T.Te * one_over_kB;
		Estimate_Ion_Neutral_Cross_sections_d(TeV, &sigma_MT, &sigma_visc);
		sqrt_T = sqrt(T.Te);
		nu_en_visc = our_n.n_n * sigma_visc * sqrt_T * over_sqrt_m_e;
		f64 nu_eiBar = nu_eiBarconst * kB_to_3halves * our_n.n *
			Get_lnLambda_d(our_n.n, T.Te) / (T.Te*sqrt_T);
		//nu_eHeart:
		nu.e = nu_en_visc + 1.87*nu_eiBar;

		TeV = T.Ti*one_over_kB;
		Estimate_Ion_Neutral_Cross_sections_d(TeV, &sigma_MT, &sigma_visc); // could easily save one call
		sqrt_T = sqrt(T.Ti); // again not that hard to save one call
		nu_in_visc = our_n.n_n * sigma_visc * sqrt(T.Ti / m_ion + T.Tn / m_n);
		nu_ni_visc = nu_in_visc * (our_n.n / our_n.n_n);
		//nu_nn_visc:
		nu.n = our_n.n_n * Estimate_Neutral_Neutral_Viscosity_Cross_section_d(T.Tn * one_over_kB)
			* sqrt(T.Tn / m_n);
		nu_ii = our_n.n*kB_to_3halves*Get_lnLambda_ion_d(our_n.n, T.Ti) *Nu_ii_Factor /
			(sqrt_T*T.Ti);
		// nu_iHeart:
		nu.i = 0.75*nu_in_visc + 0.8*nu_ii - 0.25*(nu_in_visc*nu_ni_visc) / (3.0*nu_ni_visc + nu.n);

		//shared_n_over_nu[threadIdx.x].e = our_n.n / nu.e;
		//	shared_n_over_nu[threadIdx.x].i = our_n.n / nu.i;
		//	shared_n_over_nu[threadIdx.x].n = our_n.n_n / nu.n;
	}
	else {
		memset(&nu, 0, sizeof(species3));
	}

	p_nu[index] = nu;
}


__global__ void kernelAccumulateDiffusiveHeatRateAndCalcIonisation(
	f64 const h_use,
	structural * __restrict__ p_info_sharing,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,

	nvals * __restrict__ p_n_major,
	T3 * __restrict__ p_T_major,
	f64_vec3 * __restrict__ p_B_major,
	species3 * __restrict__ p_nu_major,

	NTrates * __restrict__ NTadditionrates)
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
	nvals our_n;


	if (info.flag == DOMAIN_VERTEX) {

		our_n = p_n_major[index]; // never used again once we have kappa
		our_nu = p_nu_major[index];
		our_T = p_T_major[index]; // CAREFUL: Pass vertex array if we use vertex index
		shared_n_over_nu[threadIdx.x].e = our_n.n / our_nu.e;
		shared_n_over_nu[threadIdx.x].i = our_n.n / our_nu.i;
		shared_n_over_nu[threadIdx.x].n = our_n.n_n / our_nu.n;
		shared_nu_iHeart[threadIdx.x] = our_nu.i;
		shared_nu_eHeart[threadIdx.x] = our_nu.e;
		shared_B[threadIdx.x] = p_B_major[index].xypart();
		shared_T[threadIdx.x] = our_T;

	}
	else {
		// SHOULD NOT BE LOOKING INTO INS.
		// Is OUTERMOST another thing that comes to this branch? What about it? Should we also rule out traffic?

		// How do we avoid?
		memset(&(shared_B[threadIdx.x]), 0, sizeof(f64_vec2));
		memset(&(shared_n_over_nu[threadIdx.x]), 0, sizeof(species3));
		shared_nu_iHeart[threadIdx.x] = 0.0;
		shared_nu_eHeart[threadIdx.x] = 0.0;
		memset(&(shared_T[threadIdx.x]), 0, sizeof(T3));
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
	f64_vec2 B_out;
	f64 AreaMajor = 0.0;
	// 29 doubles right there.
	NTrates ourrates;   // 5 more ---> 34
	f64 kappa_parallel_e, kappa_parallel_i, kappa_neut;
	long indexneigh;
	f64 nu_eHeart, nu_iHeart;

	// Need this, we are adding on to existing d/dt N,NT :
	memcpy(&ourrates, NTadditionrates + index, sizeof(NTrates));

	if ((info.flag == INNERMOST) || (info.flag == OUTERMOST))
	{
		// [ Ignore flux into edge of outermost vertex I guess ???]
	}
	else {
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

				f64_vec2 edge_normal;
				edge_normal.x = THIRD*(pos_anti.y - pos_clock.y);
				edge_normal.y = THIRD*(pos_clock.x - pos_anti.x);

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
						(20.0 / 9.0) *
						0.5*(shared_n_over_nu[indexneigh - StartMajor].i
							+ shared_n_over_nu[threadIdx.x].i)
						*(0.5*(T_out.Ti + our_T.Ti)) * over_m_i;

					kappa_neut = NEUTRAL_KAPPA_FACTOR * 0.5*(shared_n_over_nu[indexneigh - StartMajor].n
						+ shared_n_over_nu[threadIdx.x].n)
						*(0.5*(T_out.Tn + our_T.Tn)) * over_m_n;
					// If we don't carry kappa_ion we are carrying shared_n_over_nu because
					// we must load that only once for the exterior neighs. So might as well carry kappa_ion.
					nu_eHeart = 0.5*(our_nu.e + shared_nu_eHeart[indexneigh - StartMajor]);
					nu_iHeart = 0.5*(our_nu.i + shared_nu_iHeart[indexneigh - StartMajor]);
				}
				else {
					nvals n_out = p_n_major[indexneigh];
					f64_vec3 B_out3 = p_B_major[indexneigh];
					B_out = B_out3.xypart();
					T_out = p_T_major[indexneigh];  // reason to combine n,T . How often do we load only 1 of them?
													// Calculate n/nu out there:
					species3 nu_out = p_nu_major[indexneigh];

					kappa_parallel_e =
						2.5*0.5*(n_out.n / nu_out.e + shared_n_over_nu[threadIdx.x].e)
						*(0.5*(T_out.Te + our_T.Te))* over_m_e;
					kappa_parallel_i =
						(20.0 / 9.0) * 0.5*(n_out.n / nu_out.i + shared_n_over_nu[threadIdx.x].i)
						*0.5*(T_out.Ti + our_T.Ti)*over_m_i;
					kappa_neut = NEUTRAL_KAPPA_FACTOR * 0.5*(n_out.n_n / nu_out.n + shared_n_over_nu[threadIdx.x].n)
						*0.5*(T_out.Tn + our_T.Tn)*over_m_n;

					nu_eHeart = 0.5*(our_nu.e + nu_out.e);
					nu_iHeart = 0.5*(our_nu.i + nu_out.i);
					// Could we save register pressure by just calculating these 3 nu values
					// first and doing a load?
				};
				PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
				if (PBC == NEEDS_ANTI) 	B_out = Anticlock_rotate2(B_out);
				if (PBC == NEEDS_CLOCK)	B_out = Clockwise_rotate2(B_out);

				f64_vec3 omega = Make3(qovermc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qovermc);

				// if the outward gradient of T is positive, inwardheatflux is positive.
				//kappa_grad_T_dot_edge_normal = 
				ourrates.NeTe += TWOTHIRDS*kappa_parallel_e*(
					edge_normal.x*(
						//kappa.xx*grad_T.x + kappa.xy*grad_T.y
					(nu_eHeart*nu_eHeart + omega.x*omega.x)*grad_T.x +
						(omega.x*omega.y - nu_eHeart *omega.z)*grad_T.y
						)
					+ edge_normal.y*(
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

				omega = Make3(qoverMc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qoverMc);

				ourrates.NiTi += TWOTHIRDS * kappa_parallel_i *(
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
			};

			// now add IONISATION:

			f64 TeV = shared_T[threadIdx.x].Te * one_over_kB;
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

			ourrates.NeTe += -TWOTHIRDS * 13.6*kB*ourrates.N + 0.5*shared_T[threadIdx.x].Tn*ionise_rate;
			ourrates.NiTi += 0.5*shared_T[threadIdx.x].Tn*ionise_rate;
			ourrates.NnTn += (shared_T[threadIdx.x].Te + shared_T[threadIdx.x].Ti)*recomb_rate;

			memcpy(NTadditionrates + index, &ourrates, sizeof(NTrates));

		}
		else {
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
			f64 Div_v_overall_integrated = p_Integrated_div_v_overall[index];
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
			newdata.NnTn += n_src_or_use[threadIdx.x].n_n*AreaMajor[threadIdx.x] * T_src.Tn*factor_neut;
			newdata.NiTi += n_src_or_use[threadIdx.x].n*AreaMajor[threadIdx.x] * T_src.Ti*factor;
			newdata.NeTe += n_src_or_use[threadIdx.x].n*AreaMajor[threadIdx.x] * T_src.Te*factor;  // 
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
			// Need nu_ne etc to be defined:
			nu_ne_MT = s_en_MT * n_src_or_use[threadIdx.x].n * electron_thermal; // have to multiply by n_e for nu_ne_MT
			nu_ni_MT = s_in_MT * n_src_or_use[threadIdx.x].n * ionneut_thermal;
			nu_en_MT = s_en_MT * n_src_or_use[threadIdx.x].n_n*electron_thermal;
			nu_in_MT = s_in_MT * n_src_or_use[threadIdx.x].n_n*ionneut_thermal;

			nu_ei = nu_eiBarconst * kB_to_3halves*n_src_or_use[threadIdx.x].n*lnLambda /
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

		p_T_major_dest[index] = T_dest;

	}
	else {
		// nothing to do ??
		if (info.flag == OUTERMOST) {
			p_n_major_dest[index] = p_n_major[index];
			p_T_major_dest[index] = p_T_major[index];
		}
		else {
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
	CHAR4 * __restrict__ p_szPBCneigh_tris,
	nvals * __restrict__ p_n_upwind_minor // result 
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

		trineighindex = p_trineighindex[iTri];
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

		edge_normal0.x = nearby_pos.y - info.pos.y;
		edge_normal0.y = info.pos.x - nearby_pos.x;
		// CAREFUL AS FUCK : which side is which???
		// tri centre 2 is on same side of origin as corner 1 -- I think
		// We don't know if the corners have been numbered anticlockwise?
		// Could arrange it though.
		// So 1 is anticlockwise for edge 0.

		f64 numerator = 0.0;
		f64 dot1, dot2;
		f64 dot0 = relv.dot(edge_normal0);
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
		if (dot1 > 0.0)
		{
			numerator += dot1*n0;
		} else {
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
		if (dot2 > 0.0)
		{
			numerator += dot2*n1;
		}
		else {
			dot2 = -dot2;
			numerator += dot2*n0;
		}

		if (dot0 + dot1 + dot2 == 0.0) {
			result.n = THIRD*(n0 + n1 + n2);
		} else {
			result.n = numerator / (dot0 + dot1 + dot2);
		};
		// Argument against fabs in favour of squared weights?

		// Think carefully / debug how it goes for CROSSING_INS.
	} else {
		result.n = 0.0;
	};

	// Now same for upwind neutral density:
	// In order to use syncthreads we had to come out of the branching.

	if (threadIdx.x < threadsPerTileMajor)
	{
		memcpy(&(shared_shards[threadIdx.x].n), &(p_n_shard_n_major[threadsPerTileMajor*blockIdx.x + threadIdx.x].n),
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
		}
		else {
			n1 = p_n_shard_n_major[tricornerindex.i2].n[who_am_I.i2];
			// at least it's 1 bus journey this way instead of 2 to fetch n_shards.
		}
		if ((tricornerindex.i3 >= StartMajor) && (tricornerindex.i3 < EndMajor))
		{
			n2 = shared_shards[tricornerindex.i3 - StartMajor].n[who_am_I.i3];
		}
		else {
			n2 = p_n_shard_n_major[tricornerindex.i3].n[who_am_I.i3];
			// at least it's 1 bus journey this way instead of 2 to fetch n_shards.
		}

		f64_vec2 relv = p_v_n_minor[iTri].xypart() - v_overall;

		f64 numerator = 0.0;
		f64 dot1, dot2;
		f64 dot0 = relv.dot(edge_normal0);
		if (dot0 > 0.0) // v faces anticlockwise
		{
			numerator += dot0*n2;
		}
		else {
			dot0 = -dot0;
			numerator += dot0*n1;
		}

		dot1 = relv.dot(edge_normal1);
		if (dot1 > 0.0)
		{
			numerator += dot1*n0;
		}
		else {
			dot1 = -dot1;
			numerator += dot1*n2;
		}

		dot2 = relv.dot(edge_normal2);
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
		} else {
			result.n_n = numerator / (dot0 + dot1 + dot2);
		};
		// Look carefully at what happens for CROSSING_INS.
		// relv should be horizontal, hence it doesn't give a really low density? CHECK IT IN PRACTICE.

	} else {
		result.n_n = 0.0;		
	};

	p_n_upwind_minor[iTri] = result;

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
	T3 * __restrict__ p_T_minor, // may or may not overlap source: don't we only use from tris? so not overlap

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
		memcpy(info, p_info_minor + (threadsPerTileMinor*blockDim.x + 2 * threadIdx.x), sizeof(structural) * 2);
		shared_pos[2 * threadIdx.x] = info[0].pos;
		shared_pos[2 * threadIdx.x + 1] = info[1].pos;
		memcpy(&(shared_n_upwind[2 * threadIdx.x]), p_n_upwind_minor + (threadsPerTileMinor*blockDim.x + 2 * threadIdx.x), sizeof(nvals) * 2);
		v4 vie[2];
		memcpy(&vie, p_vie_minor + (threadsPerTileMinor*blockDim.x + 2 * threadIdx.x), sizeof(v4) * 2);
		shared_vxy[2 * threadIdx.x] = vie[0].vxy;
		shared_vxy[2 * threadIdx.x + 1] = vie[1].vxy;
		f64_vec3 v_n[2];
		memcpy(v_n, p_v_n_minor + (threadsPerTileMinor*blockDim.x + 2 * threadIdx.x), sizeof(f64_vec3) * 2);
		shared_v_n[2 * threadIdx.x] = v_n[0].xypart();
		shared_v_n[2 * threadIdx.x + 1] = v_n[1].xypart();
		memcpy(&(shared_T[2 * threadIdx.x]), p_T_minor + (threadsPerTileMinor*blockDim.x + 2 * threadIdx.x), sizeof(T3) * 2);
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
				endpt0 = Clockwise_d*endpt0;
				vxy_next = Clockwise_d*vxy_next;
				v_n_next = Clockwise_d*v_n_next;
				v_overall_next = Clockwise_d*v_overall_next;
			};
			if (szPBC[i] == ROTATE_ME_ANTICLOCKWISE) {
				endpt0 = Anticlockwise_d*endpt0;
				vxy_next = Anticlockwise_d*vxy_next;
				v_n_next = Anticlockwise_d*v_n_next;
				v_overall_next = Anticlockwise_d*v_overall_next;
			};

			f64_vec2 edge_normal;
			edge_normal.x = endpt1.y - endpt0.y;
			edge_normal.y = endpt0.x - endpt1.x;

			AreaMajor += 0.5*edge_normal.x*(endpt0.x + endpt1.x);

			Integrated_div_v += 0.5*(vxy_prev + vxy_next).dot(edge_normal);
			Integrated_div_v_n += 0.5*(v_n_prev + v_n_next).dot(edge_normal);
			Integrated_div_v_overall += 0.5*(v_overall_prev + v_overall_next).dot(edge_normal); // Average outward velocity of edge...

			totalmassflux_out.n += 0.5*(n_prev*vxy_prev + n_next*vxy_next).dot(edge_normal);
			totalheatflux_out.Ti += 0.5*(n_prev*Ti_prev*vxy_prev + n_next*Ti_next*vxy_next).dot(edge_normal);
			totalheatflux_out.Te += 0.5*(n_prev*Te_prev*vxy_prev + n_next*Te_next*vxy_next).dot(edge_normal);

			totalmassflux_out.n_n += 0.5*(nn_prev*v_n_prev + nn_next*v_n_next).dot(edge_normal);
			totalheatflux_out.Tn += 0.5*(nn_prev*Tn_prev*v_n_prev + nn_next*Tn_next*v_n_next).dot(edge_normal);

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
		NTplus.NeTe = -h_use*totalheatflux_out.Te;
		NTplus.NiTi = -h_use*totalheatflux_out.Ti;
		NTplus.NnTn = -h_use*totalheatflux_out.Tn;

		memcpy(p_NTadditionrates + iVertex, &NTplus, sizeof(NTrates));

		// What we need now: 
		//	* Cope with non-domain vertex
		p_div_v[iVertex] = Integrated_div_v / AreaMajor;
		p_div_v_n[iVertex] = Integrated_div_v_n / AreaMajor;
		p_Integrated_div_v_overall[iVertex] = Integrated_div_v_overall;
		// 3 divisions -- could speed up by creating 1.0/AreaMajor. Except it's bus time anyway.
	}
	else {
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
	AAdot * __restrict__ p_AAdot_intermediate,
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
		v0.vez -= 0.5*eoverm*h_use*h_use* c* Lap_Az_used;
		v0.viz += 0.5*qoverM*h_use*h_use* c* Lap_Az_used; // adaptation for this.

		OhmsCoeffs Ohms = p_Ohms[iMinor];

		f64 vez_1 = v0.vez + Ohms.sigma_e_zz * Ez_strength;
		f64 viz_1 = v0.viz + Ohms.sigma_i_zz * Ez_strength;

		nvals n_use = p_n_minor[iMinor];

		//AAzdot_k.Azdot +=
		//  h_use * ROCAzdot_antiadvect + h_use * c*c*(Lap_Az +
		//	0.5*FOURPI_OVER_C * q*n_use.n*(vie_k.viz - vie_k.vez)); // INTERMEDIATE
		//	p_AAdot_intermediate[iMinor] = AAzdot_k; // not k any more
													 //	
		p_Azdot0[iMinor] = p_AAdot_intermediate[iMinor].Azdot
			- h_use*c*c*Lap_Az_used // cancel out what PopOhms did!
	// + h_use * ROCAzdot_antiadvect[iMinor]   // we did this as part of PopOhms.
	// + h_use *c*2.0*PI* q*n_use.n*(v_src.viz - v_src.vez) // we did this as part of PopOhms
			+ h_use *c*2.0*M_PI* q*n_use.n*(viz_1 - vez_1);

		//	denom_i = 1.0 + h_use * h_use*PI*qoverM*q_*data_use.n
		//		+ h_use * 0.5*qoverMc*(grad_Az.dot(beta_xy_z)) 
		//		+ h_use * 0.5*M_ni*nu_in_MT*(1.0 - beta_ni) 
		//		+ h_use * 0.5*moverM*nu_ei_effective;
		//
			// Now check what we did with Azdot already to create intermediate. Usable??
		//	denom_e = 1.0 + (h_use*h_use*PI*q_*qoverm*data_use.n
		//		+ 0.5*h_use*eovermc_*(grad_Az.dot(beta_xy_z)))*(1.0 - beta_ie_z)
		//		+ 0.5*h_use*M_ne*nu_en_MT*(1.0 - beta_ne - beta_ni * beta_ie_z)
		//		+ 0.5*h_use*nu_ei_effective*(1.0 - beta_ie_z);


		f64 vez0_coeff_on_Lap_Az = 0.5* h_use*h_use*eoverm*c / p_denom_e[iMinor];
		f64 viz0_coeff_on_Lap_Az = -0.5*h_use*h_use*qoverM*c / p_denom_i[iMinor];

		p_gamma[iMinor] = h_use*c*c*(1.0 + 0.5*FOURPI_OVER_C * q*n_use.n*
			(viz0_coeff_on_Lap_Az - vez0_coeff_on_Lap_Az));
		
	} else {
		// In PopOhms:
		// AAdot temp = p_AAdot_src[iMinor];
		// temp.Azdot += h_use * c*(c*p_LapAz[iMinor] + 4.0*PI*Jz);
		// p_AAdot_intermediate[iMinor] = temp; // 
		
		p_Azdot0[iMinor] = p_AAdot_intermediate[iMinor].Azdot - h_use*c*c*Lap_Az_used;
		p_gamma[iMinor] = h_use * c*c;
		// Note that for frills these will simply not be used.
	};

}

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
	// Now going to need to go through and see this set 0 or sensible every time.

	f64_vec3 * __restrict__ p_vn0_dest,
	v4 * __restrict__ p_v0_dest,
	OhmsCoeffs * __restrict__ p_OhmsCoeffs_dest,
	AAdot * __restrict__ p_AAdot_intermediate,

	f64 * __restrict__ p_Iz0,
	f64 * __restrict__ p_sigma_zz,
	
	f64 * __restrict__ p_denom_i,
	f64 * __restrict__ p_denom_e,
	bool const bSwitchSave) // for turning on save of these denom_ quantities
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

	long const iMinor = threadIdx.x + blockIdx.x * blockDim.x; // INDEX OF VERTEX

	structural info = p_info_minor[iMinor];

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == DOMAIN_TRIANGLE))
	{
		v4 vie_k = p_vie_src[iMinor];
		f64_vec3 v_n_src = p_v_n_src[iMinor];

		nvals n_use = p_n_minor_use[iMinor];
		AreaMinor = p_AreaMinor[iMinor];
		// Are we better off with operator = or with memcpy?
		vn0 = v_n_src;
		{
			f64_vec3 MAR;
			memcpy(&MAR, p_MAR_neut + iMinor, sizeof(f64_vec3));
			// CHECK IT IS INTENDED TO AFFECT Nv

			vn0.x += h_use * (MAR.x / (n_use.n_n*AreaMinor));
			vn0.y += h_use * (MAR.y / (n_use.n_n*AreaMinor));// MomAddRate is addition rate for Nv. Divide by N.

			memcpy(&MAR, p_MAR_ion + iMinor, sizeof(f64_vec3));
			
			v0.vxy = vie_k.vxy
				+ h_use * ( m_i*MAR.xypart()
					/ (n_use.n*(m_i + m_e)*AreaMinor));

			v0.viz = vie_k.viz
				+ h_use * MAR.z / (n_use.n*AreaMinor);

			memcpy(&MAR, p_MAR_elec + iMinor, sizeof(f64_vec3));

			v0.vxy += h_use * ( m_e*MAR.xypart()
					/ (n_use.n*(m_i + m_e)*AreaMinor));

			v0.vez = vie_k.vez
				+ h_use * MAR.z / (n_use.n*AreaMinor);   // UM WHY WAS THIS NEGATIVE
													 // + !!!!
		}

		OhmsCoeffs ohm;
		f64 beta_ie_z, Lap_Az;
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
		}

		vn0.x += -0.5*h_use*(M_e_over_en)*(cross_section_times_thermal_en*n_use.n)*(v_n_src.x - vie_k.vxy.x)
			- 0.5*h_use*(M_i_over_in)*(cross_section_times_thermal_in*n_use.n)*(v_n_src.x - vie_k.vxy.x);
		vn0.x += -0.5*h_use*(M_e_over_en)*(cross_section_times_thermal_en*n_use.n)*(v_n_src.y - vie_k.vxy.y)
			- 0.5*h_use*(M_i_over_in)*(cross_section_times_thermal_in*n_use.n)*(v_n_src.y - vie_k.vxy.y);
		vn0.z += -0.5*h_use*(M_e_over_en)*(cross_section_times_thermal_en*n_use.n)*(v_n_src.z - vie_k.vez)
			- 0.5*h_use*(M_i_over_in)*(cross_section_times_thermal_in*n_use.n)*(v_n_src.z - vie_k.viz);
		denom = 1.0 + h_use * 0.5*M_e_over_en* (cross_section_times_thermal_en*n_use.n)
			+ 0.5*h_use*M_i_over_in* (cross_section_times_thermal_in*n_use.n);

		vn0 /= denom; // It is now the REDUCED value

		ohm.beta_ne = 0.5*h_use*(M_e_over_en)*(cross_section_times_thermal_en*n_use.n) / denom;
		ohm.beta_ni = 0.5*h_use*(M_i_over_in)*(cross_section_times_thermal_in*n_use.n) / denom;

		// Now we do vexy:

		grad_Az[threadIdx.x] = p_GradAz[iMinor];
		gradTe[threadIdx.x] = p_GradTe[iMinor];
		f64 LapAz = p_LapAz[iMinor];
		f64 ROCAzdot_antiadvect = ROCAzdotduetoAdvection[iMinor];

		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		// Here is where we should be using v_use:
		// We do midpoint instead? Why not? Thus allowing us not to load v_use.
		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		v0.vxy +=
			-h_use * (q / (2.0*c*(m_i + m_e)))*(vie_k.vez - vie_k.viz)*grad_Az[threadIdx.x]
			- (h_use / (2.0*(m_i + m_e)))*(m_n*M_i_over_in*(cross_section_times_thermal_in*n_use.n_n)
				+ m_n * M_e_over_en*(cross_section_times_thermal_en*n_use.n_n))*
				(vie_k.vxy - v_n_src.xypart() - vn0.xypart());

		denom = 1.0 + (h_use / (2.0*(m_i + m_e)))*(
			m_n* M_i_over_in* (cross_section_times_thermal_in*n_use.n_n)
			+ m_n * M_e_over_en*(cross_section_times_thermal_en*n_use.n_n))*(1.0 - ohm.beta_ne - ohm.beta_ni);
		v0.vxy /= denom;

		ohm.beta_xy_z = (h_use * q / (2.0*c*(m_i + m_e)*denom)) * grad_Az[threadIdx.x];
		/////////////////////////////////////////////////////////////////////////////// midpoint

		omega[threadIdx.x] = qovermc*p_B[iMinor].xypart();

		f64 nu_ei_effective = nu_eiBar * (1.0 - 0.9*nu_eiBar*(nu_eHeart*nu_eHeart + qovermc*BZ_CONSTANT*qovermc*BZ_CONSTANT) /
			(nu_eHeart*(nu_eHeart*nu_eHeart + omega[threadIdx.x].x*omega[threadIdx.x].x + omega[threadIdx.x].y*omega[threadIdx.x].y + qovermc*BZ_CONSTANT*qovermc*BZ_CONSTANT)));

		AAdot AAzdot_k = p_AAdot_src[iMinor];
				
		//if ((iPass == 0) || (bFeint == false))
		{
			v0.viz +=
				-0.5*h_use*qoverMc*(2.0*AAzdot_k.Azdot

					+ h_use * ROCAzdot_antiadvect + h_use * c*c*(Lap_Az
						+ FOURPI_OVER_C*0.5 * q*n_use.n*(vie_k.viz - vie_k.vez)))
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
			1.5*h_use*nu_eiBar*((omega[threadIdx.x].x*qovermc*BZ_CONSTANT - nu_eHeart * omega[threadIdx.x].y)*gradTe[threadIdx.x].x +
			(omega[threadIdx.x].y*qovermc*BZ_CONSTANT + nu_eHeart * omega[threadIdx.x].x)*gradTe[threadIdx.x].y) /
				(m_i*nu_eHeart*(nu_eHeart*nu_eHeart + omega[threadIdx.x].dot(omega[threadIdx.x])));

		v0.viz += -h_use * 0.5*M_n_over_ni*(cross_section_times_thermal_in*n_use.n_n) *(vie_k.viz - v_n_src.z - vn0.z) // THIS DOESN'T LOOK RIGHT
			+ h_use * 0.5*(moverM)*nu_ei_effective*(vie_k.vez - vie_k.viz);

		denom = 1.0 + h_use * h_use*M_PI*qoverM*q*n_use.n + h_use * 0.5*qoverMc*(grad_Az[threadIdx.x].dot(ohm.beta_xy_z)) +
			h_use * 0.5*M_n_over_ni*(cross_section_times_thermal_in*n_use.n_n) *(1.0 - ohm.beta_ni) + h_use * 0.5*moverM*nu_ei_effective;

		if (bSwitchSave) p_denom_i[iMinor] = denom;
		//				viz0_coeff_on_Lap_Az = -0.5*h_use*qoverMc*h_use*c*c / denom;

		v0.viz /= denom;

		ohm.sigma_i_zz = h_use * qoverM / denom;
		beta_ie_z = (h_use*h_use*M_PI*qoverM*q*n_use.n
			+ 0.5*h_use*qoverMc*(grad_Az[threadIdx.x].dot(ohm.beta_xy_z))
			+ h_use * 0.5*M_n_over_ni*(cross_section_times_thermal_in*n_use.n_n) *ohm.beta_ne
			+ h_use * 0.5*moverM*nu_ei_effective) / denom;

		v0.vez +=
			h_use * 0.5*qovermc*(2.0*AAzdot_k.Azdot
				+ h_use * ROCAzdot_antiadvect
				+ h_use * c*c*(Lap_Az
					+ 0.5*FOURPI_Q_OVER_C*n_use.n*(vie_k.viz + v0.viz - vie_k.vez))) // ?????????????????
			+ 0.5*h_use*qovermc*(vie_k.vxy + v0.vxy + v0.viz * ohm.beta_xy_z).dot(grad_Az[threadIdx.x]);

		v0.vez -=
			1.5*h_use*nu_eiBar*((omega[threadIdx.x].x*qovermc*BZ_CONSTANT - nu_eHeart * omega[threadIdx.x].y)*gradTe[threadIdx.x].x +
			(omega[threadIdx.x].y*qovermc*BZ_CONSTANT + nu_eHeart * omega[threadIdx.x].x)*gradTe[threadIdx.x].y) /
				(m_e*nu_eHeart*(nu_eHeart*nu_eHeart + omega[threadIdx.x].dot(omega[threadIdx.x])));

		// could store this from above and put opposite -- dividing by m_e instead of m_i

		v0.vez += -0.5*h_use*M_n_over_ne*(cross_section_times_thermal_en*n_use.n_n) *(vie_k.vez - v_n_src.z - vn0.z - ohm.beta_ni * v0.viz)
			- 0.5*h_use*nu_ei_effective*(vie_k.vez - vie_k.viz - v0.viz);
		denom = 1.0 + (h_use*h_use*M_PI*q*eoverm*n_use.n
			+ 0.5*h_use*qovermc*(grad_Az[threadIdx.x].dot(ohm.beta_xy_z)))*(1.0 - beta_ie_z)
			+ 0.5*h_use*M_n_over_ne*(cross_section_times_thermal_en*n_use.n_n) *(1.0 - ohm.beta_ne - ohm.beta_ni * beta_ie_z)
			+ 0.5*h_use*nu_ei_effective*(1.0 - beta_ie_z);

		//		vez0_coeff_on_Lap_Az = h_use * h_use*0.5*qovermc* c*c / denom; 

		ohm.sigma_e_zz = (-h_use * eoverm + h_use * h_use*M_PI*q*eoverm*n_use.n*ohm.sigma_i_zz
			+ h_use * 0.5*qovermc*(grad_Az[threadIdx.x].dot(ohm.beta_xy_z))*ohm.sigma_i_zz
			+ 0.5*h_use*M_n_over_ne*(cross_section_times_thermal_en*n_use.n_n) *ohm.beta_ni*ohm.sigma_i_zz
			+ 0.5*h_use*nu_ei_effective*ohm.sigma_i_zz)
			/ denom;

		if (bSwitchSave) p_denom_e[iMinor] = denom;

		v0.vez /= denom;

		// Now update viz(Ez):
		v0.viz += beta_ie_z * v0.vez;
		ohm.sigma_i_zz += beta_ie_z * ohm.sigma_e_zz;

		// sigma_e_zz and sigma_i_zz are change in vz for a change in Ez
		f64 EzShape = GetEzShape(info.pos.modulus());
		ohm.sigma_i_zz *= EzShape;
		ohm.sigma_e_zz *= EzShape;

		// ==============================================================================================

		p_v0_dest[iMinor] = v0;
		p_OhmsCoeffs_dest[iMinor] = ohm;
		p_vn0_dest[iMinor] = vn0;

		Iz[threadIdx.x] = q*AreaMinor*n_use.n*(v0.viz - v0.vez);
		sigma_zz[threadIdx.x] = q*AreaMinor*n_use.n*(ohm.sigma_i_zz - ohm.sigma_e_zz);
		// Totally need to be skipping the load of an extra n.

		// On iPass == 0, we need to do the accumulate.
		//	p_Azdot_intermediate[iMinor] = Azdot_k
		//		+ h_use * ROCAzdot_antiadvect + h_use * c*c*(Lap_Az +
		//			0.5*FOURPI_OVER_C * q*n_use.n*(data_k.viz - data_k.vez)); // INTERMEDIATE

		AAzdot_k.Azdot +=
			 h_use * ROCAzdot_antiadvect + h_use * c*c*(Lap_Az +
				0.5*FOURPI_OVER_C * q*n_use.n*(vie_k.viz - vie_k.vez)); // INTERMEDIATE
		p_AAdot_intermediate[iMinor] = AAzdot_k; // not k any more

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
		if ((iMinor < BEGINNING_OF_CENTRAL) && ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL)))
		{
			p_AAdot_intermediate[iMinor].Azdot = 0.0;
			// Set Az equal to neighbour in every case, after Accelerate routine.
		} else {
			// Let's make it go right through the middle of a triangle row for simplicity.

			f64 Jz = 0.0;
			if ((iMinor >= numStartZCurrentTriangles) && (iMinor <  numEndZCurrentTriangles))
			{
				// Azdotdot = c^2 (Lap Az + 4pi/c Jz)
				// ASSUME we are fed Iz_prescribed.
				//Jz = -Iz_prescribed / (real)(numEndZCurrentTriangles - numStartZCurrentTriangles);

				AreaMinor = p_AreaMinor[iMinor];
				Jz = negative_Iz_per_triangle / AreaMinor; // Iz would come from multiplying back by area and adding.
			};

			AAdot temp = p_AAdot_src[iMinor];
			temp.Azdot += h_use * c*(c*p_LapAz[iMinor] + 4.0*M_PI*Jz);
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
}


__global__ void kernelCalculateVelocityAndAzdot(
	f64 h_use,
	f64_vec3 * __restrict__ p_vn0,
	v4 * __restrict__ p_v0,
	OhmsCoeffs * __restrict__ p_OhmsCoeffs,
	AAdot * __restrict__ p_AAzdot_intermediate,
	nvals * __restrict__ p_n_minor,

	AAdot * __restrict__ p_AAzdot_out,
	v4 * __restrict__ p_vie_out,
	f64_vec3 * __restrict__ p_vn_out ) 
{
	long iMinor = blockIdx.x*blockDim.x + threadIdx.x;

	OhmsCoeffs ohm = p_OhmsCoeffs[iMinor];
	v4 v;
	v4 v0 = p_v0[iMinor];

	v.vez = v0.vez + ohm.sigma_e_zz * Ez_strength;  // 2
	v.viz = v0.viz + ohm.sigma_i_zz * Ez_strength;  // 2
	v.vxy = v0.vxy + ohm.beta_xy_z * (v.viz - v.vez);   // 4
	f64_vec3 v_n = p_vn0[iMinor];							 // 3 sep
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
	nvals n_use = p_n_minor[iMinor];

	memcpy(&(p_vie_out[iMinor]), &v, sizeof(v4)); // operator = vs memcpy
	p_vn_out[iMinor] = v_n;
	AAdot temp = p_AAzdot_intermediate[iMinor];
	temp.Azdot += h_use*c*0.5*FOUR_PI*q*n_use.n*(v.viz - v.vez);

	p_AAzdot_out[iMinor] = temp; 
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
		p_Az[index] = p_Az[izNeigh.i1];
	}
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
	f64 * __restrict__ p_Jacobi_x)
{
	long const index = blockDim.x*blockIdx.x + threadIdx.x;
	f64 eps;
	structural info = p_info[index];
	if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL))
	{
		eps = p_Lap_Aznext[index];
		p_Jacobi_x[index] = -eps / p_LapCoeffself[index];
	}
	else {
		eps = p_Az_array_next[index] - h_use * p_gamma[index] * p_Lap_Aznext[index]
			- p_Az_array[index] - h_use*p_Azdot0[index];
		p_Jacobi_x[index] = -eps / (1.0 - h_use * p_gamma[index] * p_LapCoeffself[index]);
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

/*__global__ void kernelGetLap_verts(
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
long tri_len = info.neigh_len;
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


}*/

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

			++iprev;
			endpt0 = endpt1;
			prevpos = opppos;
			opppos = nextpos;
			prevAz = oppAz;
			oppAz = nextAz;
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

		p_LapAz[iVertex + BEGINNING_OF_CENTRAL] = Our_integral_Lap_Az / AreaMinor;

	}; // was thread in the first half of the block

	info = p_info[iMinor];
	ourAz = shared_Az[threadIdx.x];

	long izNeighMinor[6];
	char szPBC[6];
	memcpy(izNeighMinor, p_izNeighMinor + iMinor * 6, sizeof(long) * 6);
	memcpy(szPBC, p_szPBCtriminor + iMinor * 6, sizeof(char) * 6);
	if ((info.flag == OUTER_FRILL) || (info.flag == INNER_FRILL)) {
		if ((izNeighMinor[0] >= StartMinor) && (izNeighMinor[0] < EndMinor))
		{
			oppAz = shared_Az[izNeighMinor[0] - StartMinor];
		}
		else {
			oppAz = p_Az[izNeighMinor[0]];
		};
		p_LapAz[iMinor] = oppAz - ourAz;
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
			if (szPBC[inext] == ROTATE_ME_CLOCKWISE) nextpos = Clockwise_d*nextpos;
			if (szPBC[inext] == ROTATE_ME_ANTICLOCKWISE) nextpos = Anticlockwise_d*nextpos;

			// New definition of endpoint of minor edge:
			f64_vec2 endpt0, endpt1, edge_normal;
			endpt0 = THIRD * (prevpos + info.pos + opppos);
			endpt1 = THIRD * (nextpos + info.pos + opppos);

			edge_normal.x = endpt1.y - endpt0.y;
			edge_normal.y = endpt0.x - endpt1.x;

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
			Our_integral_Lap_Az += integ_grad_Az.dot(edge_normal) / area_quadrilateral;

			AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;

			endpt0 = endpt1;
			prevAz = oppAz;
			oppAz = nextAz;
			prevpos = opppos;
			opppos = nextpos;
			// There is an even quicker way which is to rotate pointers. No memcpy needed.
		};

		p_LapAz[iMinor] = Our_integral_Lap_Az / AreaMinor;
	};

}


__global__ void kernelGetLapCoeffs(
	structural * __restrict__ p_info,
	long * __restrict__ p_izTri,
	long * __restrict__ p_izNeighMinor,
	char * __restrict__ p_szPBCtri_vertex,
	char * __restrict__ p_szPBCtriminor,
	f64 * __restrict__ p_LapCoeffSelf)
{
	__shared__ f64_vec2 shared_pos[threadsPerTileMinor];
	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajor];

	long const iMinor = blockDim.x*blockIdx.x + threadIdx.x;
	long const iVertex = threadsPerTileMajor*blockIdx.x + threadIdx.x; // only meaningful threadIdx.x < threadsPerTileMajor

	long const StartMinor = threadsPerTileMinor*blockIdx.x;
	long const StartMajor = threadsPerTileMajor*blockIdx.x;
	long const EndMinor = StartMinor + threadsPerTileMinor;
	long const EndMajor = StartMajor + threadsPerTileMajor;

	shared_pos[threadIdx.x] = p_info[iMinor].pos;
	structural info;
	if (threadIdx.x < threadsPerTileMajor) {
		info = p_info[iVertex + BEGINNING_OF_CENTRAL];
		shared_pos_verts[threadIdx.x] = info.pos;
	};

	__syncthreads();

	f64_vec2 opppos, prevpos, nextpos;
	// Better if we use same share to do both tris and verts
	// Idea: let's make it called for # minor threads, each loads 1 shared value,
	// and only half the threads run first for the vertex part. That is a pretty good idea.

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
		}
		else {
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
			}
			else {
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
		};

		p_LapCoeffSelf[iVertex + BEGINNING_OF_CENTRAL] = Our_integral_Lap_Az_contrib_from_own_Az / AreaMinor;

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
	} else {

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

			integ_grad_Az.x = 0.5*(
				(1.0)*(info.pos.y - nextpos.y)
				+ (1.0)*(prevpos.y - info.pos.y));

			integ_grad_Az.y = -0.5*( // notice minus
				(1.0)*(info.pos.x - nextpos.x)
				+ (1.0)*(prevpos.x - info.pos.x));

			f64 area_quadrilateral = 0.5*(
				(info.pos.x + nextpos.x)*(info.pos.y - nextpos.y)
				+ (prevpos.x + info.pos.x)*(prevpos.y - info.pos.y)
				+ (opppos.x + prevpos.x)*(opppos.y - prevpos.y)
				+ (nextpos.x + opppos.x)*(nextpos.y - opppos.y)
				);

			//f64_vec2 grad_Az = integ_grad_Az / area_quadrilateral;
			Our_integral_Lap_Az_contrib_from_own_Az += integ_grad_Az.dot(edge_normal) / area_quadrilateral;

			AreaMinor += (0.5*endpt0.x + 0.5*endpt1.x)*edge_normal.x;

			endpt0 = endpt1;
			prevpos = opppos;
			opppos = nextpos;
			// There is an even quicker way which is to rotate pointers. No memcpy needed.
		};

		p_LapCoeffSelf[iMinor] = Our_integral_Lap_Az_contrib_from_own_Az / AreaMinor;
	};
}

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
	f64_vec3 * __restrict__ p_B)
{
	long const iMinor = blockIdx.x*blockDim.x + threadIdx.x;
	structural info1 = p_info1[iMinor];
	structural info2 = p_info2[iMinor];
	structural info;
	f64 r = 1.0 - ppn;
	info.pos = r*info1.pos + ppn*info2.pos;
	info.flag = info1.flag;
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
	T.Te = r*T1.Te + ppn*T1.Te;
	T.Ti = r*T1.Ti + ppn*T1.Ti;
	T.Tn = r*T1.Tn + ppn*T1.Tn;
	p_T_minor[iMinor] = T;

	f64_vec3 B1 = p_B1[iMinor];
	f64_vec3 B2 = p_B2[iMinor];
	f64_vec3 B = r*B1 + ppn*B2;
	p_B[iMinor] = B;
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
		if (info.flag == DOMAIN_VERTEX) {
			T3 T_ = p_T_minor[iVertex + BEGINNING_OF_CENTRAL];
			shared_T_verts[threadIdx.x].Te = T_.Te;
			shared_T_verts[threadIdx.x].Ti = T_.Ti;
			memcpy(&(shared_n_shards[threadIdx.x]), &(p_n_shards[iVertex]), sizeof(ShardModel)); // + 13
		}
		else {
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

			f64 n0 = shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[iprev] + shared_n_shards[threadIdx.x].n_cent;
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
				T0.Te = THIRD* (prevT.Te + shared_T[threadIdx.x].Te + oppT.Te);
				T1.Te = THIRD * (nextT.Te + shared_T[threadIdx.x].Te + oppT.Te);
				T0.Ti = THIRD * (prevT.Ti + shared_T[threadIdx.x].Ti + oppT.Ti);
				T1.Ti = THIRD * (nextT.Ti + shared_T[threadIdx.x].Ti + oppT.Ti);
				n1 = shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[inext] + shared_n_shards[threadIdx.x].n_cent;

				// Assume neighs 0,1 are relevant to border with tri 0 minor

				// To get integral grad we add the averages along the edges times edge_normals
				MAR_ion -= Make3(0.5*(n0 * T0.Ti + n1 * T1.Ti)*over_m_i*edge_normal, 0.0);
				MAR_elec -= Make3(0.5*(n0 * T0.Te + n1 * T1.Te)*over_m_e*edge_normal, 0.0);

				Our_integral_grad_Te += 0.5*(T0.Te + T1.Te) * edge_normal;

				f64 Az_edge = SIXTH * (2.0*ourAz + 2.0*oppAz + prevAz + nextAz);
				f64 Azdot_edge = SIXTH * (2.0*ourAzdot + 2.0*oppAzdot + prevAzdot + nextAzdot);
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

			p_GradAz[iVertex + BEGINNING_OF_CENTRAL] = Our_integral_grad_Az / AreaMinor;
			p_LapAz[iVertex + BEGINNING_OF_CENTRAL] = Our_integral_Lap_Az / AreaMinor;
			p_GradTe[iVertex + BEGINNING_OF_CENTRAL] = Our_integral_grad_Te / AreaMinor;
			p_B[iVertex + BEGINNING_OF_CENTRAL] = Make3(Our_integral_curl_Az / AreaMinor, BZ_CONSTANT);
			p_AreaMinor[iVertex + BEGINNING_OF_CENTRAL] = AreaMinor;

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
			}
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

			// Think carefully: DOMAIN vertex cases for n,T ...

			f64 n0 = shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[iprev] + shared_n_shards[threadIdx.x].n_cent;
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

			p_GradAz[iVertex + BEGINNING_OF_CENTRAL] = Our_integral_grad_Az / AreaMinor;
			p_LapAz[iVertex + BEGINNING_OF_CENTRAL] = Our_integral_Lap_Az / AreaMinor;
			p_B[iVertex + BEGINNING_OF_CENTRAL] = Make3(Our_integral_curl_Az / AreaMinor, BZ_CONSTANT);
			p_AreaMinor[iVertex + BEGINNING_OF_CENTRAL] = AreaMinor;

			ROCAzduetoAdvection[iVertex + BEGINNING_OF_CENTRAL] = 0.0;
			ROCAzdotduetoAdvection[iVertex + BEGINNING_OF_CENTRAL] = 0.0;
		};

	}; // was it domain vertex or Az-only
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
		if ((izNeighMinor[0] >= StartMinor) && (izNeighMinor[0] < EndMinor))
		{
			oppAz = shared_Az[izNeighMinor[0] - StartMinor];
		}
		else {

			AAdot temp = p_AAdot[izNeighMinor[0]];
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
	} else {
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
				// Worry about pathological cases later.
				// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

				// Pathological case: OUTERMOST vertex where neigh_len is not correct to take as == tri_len

				// !

				n_array[0] = THIRD*(shared_n_shards[cornerindex.i1 - StartMajor].n[who_prev]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n_cent);
				short who_next = who_am_I + 1;
				if (who_next == tri_len) who_next = 0;
				n_array[1] = THIRD*(shared_n_shards[cornerindex.i1 - StartMajor].n[who_next]
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
					n_array[0] = THIRD*(p_n_shards[cornerindex.i1].n[who_prev] + temp.x + ncent);
					n_array[1] = THIRD*(temp.x + temp.y + ncent);
				} else {
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
			tri_len = p_info_minor[cornerindex.i2 + BEGINNING_OF_CENTRAL].neigh_len;

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
			tri_len = p_info_minor[cornerindex.i3 + BEGINNING_OF_CENTRAL].neigh_len;

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
			} else {
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
				//This matches a diagram:
				//            0
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
				Our_integral_Lap_Az += integ_grad_Az.dot(edge_normal) / area_quadrilateral;

				T3 T0, T1; // waste of registers
				f64 n1;
				T0.Te = THIRD* (prevT.Te + ourT.Te + oppT.Te);
				T1.Te = THIRD * (nextT.Te + ourT.Te + oppT.Te);
				T0.Ti = THIRD * (prevT.Ti + ourT.Ti + oppT.Ti);
				T1.Ti = THIRD * (nextT.Ti + ourT.Ti + oppT.Ti);

				// Where to get n?

				n0 = n_array[i];
				n1 = n_array[inext]; // !

				// To get integral grad we add the averages along the edges times edge_normals
				MAR_ion -= Make3(0.5*(n0 * T0.Ti + n1 * T1.Ti)*over_m_i*edge_normal, 0.0);
				MAR_elec -= Make3(0.5*(n0 * T0.Te + n1 * T1.Te)*over_m_e*edge_normal, 0.0);

				f64 Az_edge = SIXTH * (2.0*ourAz + 2.0*oppAz + prevAz + nextAz);
				f64 Azdot_edge = SIXTH * (2.0*ourAzdot + 2.0*oppAzdot + prevAzdot + nextAzdot);
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
		} else {
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
				opp_v_overall = shared_v_overall[izTri[iprev] - StartMinor];
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
			f64 n0 = shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[iprev] + shared_n_shards[threadIdx.x].n_cent;
			f64_vec2 endpt0 = THIRD * (info.pos + opppos + prevpos);

			f64 vez0, viz0, vez1, viz1;
			f64_vec2 vxy0, vxy1, endpt1, edge_normal;

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
					next_v = shared_vie[izTri[inext] - StartMinor];
					next_v_overall = shared_v_overall[izTri[inext] - StartMinor];
					nextpos = shared_pos[izTri[inext] - StartMinor];
				} else {
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
				n1 = shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[inext] + shared_n_shards[threadIdx.x].n_cent;

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
			memcpy(p_MAR_ion + iVertex + BEGINNING_OF_CENTRAL, &(ownrates.ion), sizeof(f64_vec3));
			memcpy(p_MAR_elec + iVertex + BEGINNING_OF_CENTRAL, &(ownrates.elec), sizeof(f64_vec3));
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
	our_v = shared_vie[threadIdx.x];

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
			tri_len = p_info_minor[cornerindex.i2 + BEGINNING_OF_CENTRAL].neigh_len;

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
			tri_len = p_info_minor[cornerindex.i3 + BEGINNING_OF_CENTRAL].neigh_len;

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
				n1 = shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[inext] + shared_n_shards[threadIdx.x].n_cent;

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

				ownrates_minor.ion -= 0.5*relvnormal*
					(n0 * (Make3(vxy0 - our_v.vxy, viz0 - our_v.viz))
						+ n1 * (Make3(vxy1 - our_v.vxy, viz1 - our_v.viz)));
				ownrates_minor.elec -= 0.5*relvnormal*
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
			f64_vec2 overall_v_ours = p_v_overall_minor[iMinor];

			memcpy(&(p_MAR_ion[iMinor]), &(ownrates_minor.ion), sizeof(f64_vec3));
			memcpy(&(p_MAR_elec[iMinor]), &(ownrates_minor.elec), sizeof(f64_vec3));
		} else {
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

	long * __restrict__ p_izTri,
	char * __restrict__ p_szPBC,
	long * __restrict__ p_izNeighTriMinor,
	char * __restrict__ p_szPBCtriminor,
	LONG3 * __restrict__ p_who_am_I_to_corners,
	LONG3 * __restrict__ p_tricornerindex,

	T3 * __restrict__ p_T_minor,
	f64_vec3 * __restrict__ p_v_n_minor,
	ShardModel * __restrict__ p_n_shards,
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
				opp_v_overall = shared_v_overall[izTri[iprev] - StartMinor];
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
			f64 n0 = shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[iprev] + shared_n_shards[threadIdx.x].n_cent;
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
				n1 = shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[inext] + shared_n_shards[threadIdx.x].n_cent;
				f64 T0, T1;
				T0 = THIRD*(prevT + ourT + oppT);
				T1 = THIRD*(nextT + ourT + oppT);
				f64_vec3 v0 = THIRD*(our_v + prev_v + opp_v);
				f64_vec3 v1 = THIRD*(our_v + opp_v + next_v);

				f64 relvnormal = 0.5*((v0 + v1).xypart()
					- (THIRD * (our_v_overall + next_v_overall + opp_v_overall))
					- (THIRD * (our_v_overall + prev_v_overall + opp_v_overall))
					).dot(edge_normal);

				MAR_neut -= 0.5*relvnormal* (n0 *(v0-our_v) + n1 * (v1 - our_v));
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
				n_array[0] = THIRD*(shared_n_shards[cornerindex.i1 - StartMajor].n[who_prev]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n[who_am_I]
					+ shared_n_shards[cornerindex.i1 - StartMajor].n_cent);
				short who_next = who_am_I + 1;
				if (who_next == tri_len) who_next = 0;
				n_array[1] = THIRD*(shared_n_shards[cornerindex.i1 - StartMajor].n[who_next]
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
			tri_len = p_info_minor[cornerindex.i2 + BEGINNING_OF_CENTRAL].neigh_len;

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
			tri_len = p_info_minor[cornerindex.i3 + BEGINNING_OF_CENTRAL].neigh_len;

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
					}
					else {
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

				// ______________________________________________________-
				n1 = shared_n_shards[threadIdx.x].n[i] + shared_n_shards[threadIdx.x].n[inext] + shared_n_shards[threadIdx.x].n_cent;

				// Assume neighs 0,1 are relevant to border with tri 0 minor.

				f64_vec3 v0 = THIRD*(our_v + prev_v + opp_v);
				f64_vec3 v1 = THIRD*(our_v + next_v + opp_v);
				
				f64 relvnormal = 0.5*((v0 + v1).xypart()
					- (THIRD * (our_v_overall + next_v_overall + opp_v_overall))
					- (THIRD * (our_v_overall + prev_v_overall + opp_v_overall))
					).dot(edge_normal);

				MAR_neut -= 0.5*relvnormal*(n0*(v0 - our_v)
					+ n1*(v1 - our_v));
				f64 T0 = THIRD*(ourT + prevT + oppT);
				f64 T1 = THIRD*(ourT + nextT + oppT);
				MAR_neut -= Make3(0.5*(n0*T0 + n1*T1)*over_m_n*edge_normal, 0.0);

				endpt0 = endpt1;
				n0 = n1;
				prevT = oppT;
				prevpos = opppos;
				prev_v = opp_v;
				prev_v_overall = opp_v_overall;
				oppT = nextT;
				opppos = nextpos;
				opp_v = next_v;
				opp_v_overall = next_v_overall;
			};
			f64_vec2 overall_v_ours = p_v_overall_minor[iMinor];

			memcpy(&(p_MAR_neut[iMinor]), &(MAR_neut), sizeof(f64_vec3));			
		}
		else {
			// Not domain, not crossing_ins, not a frill
			// ==========================================
		} // non-domain tri
	}; // was it FRILL
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
	long iTri = blockDim.x*blockIdx.x + threadIdx.x;
	structural info_tri = p_info_minor[iTri];

	LONG3 cornerindex = p_tri_corner_index[iTri];

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

		if ((info[0].pos.x > 0.0) && (info[1].pos.x > 0.0) && (info[2].pos.x > 0.0))
		{
			p_vie_minor[iTri].vxy = Clockwise_d*p_vie_minor[iTri].vxy;
			info_tri.pos = Clockwise_d*info_tri.pos;
			p_v_n_minor[iTri] = Clockwise_rotate3(p_v_n_minor[iTri]);

			// Inform all corners that a triangle got wrapped:
			p_triPBClistaffected[cornerindex.i1] = 1;
			p_triPBClistaffected[cornerindex.i2] = 1;
			p_triPBClistaffected[cornerindex.i3] = 1;
		};
		if (((info[0].pos.x > 0.0) || (flag0 == ROTATE_ME_ANTICLOCKWISE))
			&&
			((info[1].pos.x > 0.0) || (flag1 == ROTATE_ME_ANTICLOCKWISE))
			&&
			((info[2].pos.x > 0.0) || (flag2 == ROTATE_ME_ANTICLOCKWISE))) 
		{
			p_vie_minor[iTri].vxy = Anticlockwise_d*p_vie_minor[iTri].vxy;
			info_tri.pos = Anticlockwise_d*info_tri.pos;
			p_v_n_minor[iTri] = Anticlock_rotate3(p_v_n_minor[iTri]);

			// Inform all corners that a triangle got wrapped:
			p_triPBClistaffected[cornerindex.i1] = 1;
			p_triPBClistaffected[cornerindex.i2] = 1;
			p_triPBClistaffected[cornerindex.i3] = 1;

			// Tell tri neighbours as well?














		}

		p_info_minor[iTri] = info_tri;

		// Now reassess periodic for corners:
		CHAR4 tri_per_corner_flags;
		memset(&tri_per_corner_flags, 0, sizeof(CHAR4));
		tri_per_corner_flags.flag = info_tri.flag;
		if (((info[0].pos.x > 0.0) && (info[1].pos.x > 0.0) && (info[2].pos.x > 0.0))
			||
			((info[0].pos.x < 0.0) && (info[1].pos.x < 0.0) && (info[2].pos.x < 0.0)))
		{
			// 0 is fine
		} else {
			if (info[0].pos.x > 0.0) tri_per_corner_flags.per0 = ROTATE_ME_ANTICLOCKWISE;
			if (info[1].pos.x > 0.0) tri_per_corner_flags.per1 = ROTATE_ME_ANTICLOCKWISE;
			if (info[2].pos.x > 0.0) tri_per_corner_flags.per2 = ROTATE_ME_ANTICLOCKWISE;
		}
		
		p_tri_periodic_corner_flags[iTri] = tri_per_corner_flags;

		// Now reassess periodic complexion of neighbours.
		// We cannot do it until tri rotations are all completed? If we go by tri cent. 
		// Non-periodic tri: assess whether neighbour is periodic, if so then we are looking across iff we are on right. And so on.
	};
}

// This HAS TO APPLY FOR ALL NEIGHBOURS OF AFFECTED TRIS: !! !!!!! ! ! !! ! ! ! !
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

__global__ void kernelReassessTriNeighbourPeriodicFlags_and_populate_PBCIndexneighminor(
	structural * __restrict__ p_info_minor,
	LONG3 * __restrict__ p_tri_neigh_index,
	LONG3 * __restrict__ p_tri_corner_index,
	char * __restrict__ p_was_vertex_rotated,
	CHAR4 * __restrict__ p_tri_periodic_corner_flags,
	CHAR4 * __restrict__ p_tri_periodic_neigh_flags,
	char * __restrict__ p_szPBC_triminor
	)
{
	CHAR4 tri_periodic_neigh_flags;

	long const iTri = blockDim.x*blockIdx.x + threadIdx.x;
	LONG3 cornerindex = p_tri_corner_index[iTri];

	// Inefficient, no shared mem used:
	char flag0 = p_was_vertex_rotated[cornerindex.i1];
	char flag1 = p_was_vertex_rotated[cornerindex.i2];
	char flag2 = p_was_vertex_rotated[cornerindex.i3];

	if ((flag0 == 0) && (flag1 == 0) && (flag2 == 0))
	{
		// typical case: do nothing



		// We should be actually resetting it if we are a neigh of a wrapped tri or a wrapped tri


		// We still want to repopulate indexPBCminor even if only a corner wrapped.

		
		// THINK ABOUT THAT !!!

	}
	else {


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

	}; // was a corner wrapped

	// needs to be OR A NEIGH WRAPPED.
	// gather not scatter for that?
}

__global__ void kernelReset_szPBCtri_vert(
	structural * __restrict__ p_info_minor,
	long * __restrict__ p_izTri_vert,
	long * __restrict__ p_izNeigh_vert,
	char * __restrict__ p_szPBCtri_vert, 
	char * __restrict__ p_szPBCneigh_vert,
	char * __restrict__ p_triPBClistaffected
)
{
	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;

	char szPBCtri[MAXNEIGH];

	structural info = p_info_minor[BEGINNING_OF_CENTRAL + iVertex];
	if (p_triPBClistaffected[iVertex] != 0) {
		long izTri[MAXNEIGH];
		short i;
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
			};
		} else {
#pragma unroll MAXNEIGH
			for (i = 0; i < info.neigh_len; i++)
			{
				infotri = p_info_minor[izTri[i]];
				szPBCtri[i] = 0;
				if (infotri.pos.x > 0.0) szPBCtri[i] = ROTATE_ME_ANTICLOCKWISE;
			};
		};
		memcpy(p_szPBCtri_vert + MAXNEIGH*iVertex, szPBCtri, sizeof(char)*MAXNEIGH);
	} else {
		memcpy(szPBCtri, p_szPBCtri_vert + MAXNEIGH*iVertex, sizeof(char)*MAXNEIGH);
	}
	// Now check all neighbours to see if one wrapped:
	// too difficult
	// Just re-do them all IFF there is a periodic tri found in its list 
	// Otherwise do a memset to 0.
	short i;
	char sum = 0;
#pragma unroll MAXNEIGH
	for (i = 0; i < info.neigh_len; i++)
		sum += szPBCtri[i];
	if (sum == 0) {
		memset(p_szPBCneigh_vert + iVertex*MAXNEIGH, 0, sizeof(char)*MAXNEIGH);
	} else {
		char szPBCneigh[MAXNEIGH];
		long izNeigh[MAXNEIGH];
		structural infoneigh;

		// For those that have per tris, we go through and test for per neighs.
		memcpy(izNeigh, p_izNeigh_vert + MAXNEIGH*iVertex, sizeof(long)*MAXNEIGH);
		if (info.pos.x > 0.0) {
#pragma unroll MAXNEIGH
			for (i = 0; i < info.neigh_len; i++)
			{
				infoneigh = p_info_minor[izNeigh[i] + BEGINNING_OF_CENTRAL];
				szPBCneigh[i] = 0;
				if (infoneigh.pos.x < 0.0) szPBCneigh[i] = ROTATE_ME_CLOCKWISE;
			};
		} else {
#pragma unroll MAXNEIGH
			for (i = 0; i < info.neigh_len; i++)
			{
				infoneigh = p_info_minor[izNeigh[i] + BEGINNING_OF_CENTRAL];
				szPBCneigh[i] = 0;
				if (infoneigh.pos.x > 0.0) szPBCneigh[i] = ROTATE_ME_ANTICLOCKWISE;
			};
		};
		memcpy(p_szPBCneigh_vert + MAXNEIGH*iVertex, szPBCneigh, sizeof(char)*MAXNEIGH);
	};
	
	// Possibly could also argue that if triPBClistaffected == 0 then as it had no wrapping
	// triangle it cannot have a wrapping neighbour. Have to visualise to be sure.
}


// What a bugger it all is!
// Add test for 0 wrapping vertices to cut out all this running.

