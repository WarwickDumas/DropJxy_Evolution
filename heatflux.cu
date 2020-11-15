
#define TESTHEAT (iVertex == VERTCHOSEN)

__global__ void kernelAccumulateDiffusiveHeatRateROC_wrt_T_1species_Geometric(
	structural * __restrict__ p_info_minor,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,
	long * __restrict__ izTri_verts,
	char * __restrict__ szPBCtri_verts,
	f64_vec2 * __restrict__ p_cc,

	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p__T,
	f64_vec3 * __restrict__ p_B_major,

	f64 * __restrict__ p__kappa_major,
	f64 * __restrict__ p__nu_major,

	f64 * __restrict__ p___result, // d/dT of d(NT)/dt in this cell
 
	f64 * __restrict__ p_AreaMajor,
	// scrap masking for now --- but bring it back intelligently???

	bool * __restrict__ p_maskbool,
	bool * __restrict__ p_maskblock,
	bool bUseMask,

	// Just hope that our clever version will converge fast.

	int iSpecies)
{
	
	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajorClever]; // 2
																		 // DO NOT WANT:
	__shared__ f64_vec2 shared_pos[2 * threadsPerTileMajorClever]; // but as far as we know, we are having to use circumcenters.
	// Maybe it works without them now that we have the longitudinal assumptions --- don't know for sure.
	// But it means we are not being consistent with our definition of a cell?
	// Like having major cells Voronoi => velocity living on centroids (which it must, for visc + A) is in slightly the wrong place.

	__shared__ f64 shared_T[threadsPerTileMajorClever];      

	__shared__ f64_vec2 shared_B[threadsPerTileMajorClever]; // +2

															 // B is smooth. Unfortunately we have not fitted in Bz here.
															 // In order to do that perhaps rewrite so that variables are overwritten in shared.
															 // We do not need all T and nu in shared at the same time.
															 // This way is easier for NOW.
	__shared__ f64 shared_kappa[threadsPerTileMajorClever];
	__shared__ f64 shared_nu[threadsPerTileMajorClever];

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
	bool bMask;

	if (bUseMask)
		if (p_maskblock[blockIdx.x] == 0) return;

	// skip out on mask 
	if (bUseMask) bMask = p_maskbool[iVertex];

	structural info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];     // 3 double
	shared_pos_verts[threadIdx.x] = info.pos;
#ifdef CENTROID_HEATCONDUCTION
	{
		structural infotemp[2];
		memcpy(infotemp, p_info_minor + 2 * iVertex, 2 * sizeof(structural));
		shared_pos[threadIdx.x * 2] = infotemp[0].pos;
		shared_pos[threadIdx.x * 2 + 1] = infotemp[1].pos;
		// No nu to set for neutrals - not used
	}
#else
	{
		memcpy(&(shared_pos[threadIdx.x * 2]), p_cc + 2 * iVertex, 2 * sizeof(f64_vec2));		
	}
#endif

	memcpy(&(shared_nu[threadIdx.x]), p__nu_major + iVertex, sizeof(f64));
	memcpy(&(shared_kappa[threadIdx.x]), p__kappa_major + iVertex, sizeof(f64));

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {
		shared_B[threadIdx.x] = p_B_major[iVertex].xypart();
		shared_T[threadIdx.x] = p__T[iVertex];
	} else {
		// SHOULD NOT BE LOOKING INTO INS.
		// How do we avoid?
		memset(&(shared_B[threadIdx.x]), 0, sizeof(f64_vec2));
		shared_T[threadIdx.x] = 0.0;
	}

	__syncthreads();

	f64_vec2 grad_T;
	f64 T_anti, T_clock, T_out, T_outk;		// 5
	f64 x_out, x_anti, x_clock;
	f64_vec2 pos_clock, pos_anti, pos_out;   // +6
	f64_vec2 B_out;       // +2
	//NTrates ourrates;      // +5
	//f64 kappa_parallel; // do we use them all at once or can we save 2 doubles here?
	//f64 nu;                // 20 there  
	f64_vec2 edge_normal;  // 22
	f64_vec2 endpt_anti;    // 24 .. + 6 from above
	long indexneigh;     // into the 2-double buffer in L1
	f64_vec2 endpt_clock;    // As we only use endpt_anti afterwords we could union endpt_clock with edge_normal
							 // Come back and optimize by checking which things we need in scope at the same time?
	f64 kappa_out, nu_out;

	short iNeigh; // only fixed # of addresses so short makes no difference.
	char PBC; // char makes no difference.
	f64 result = 0.0;

	if ((info.flag == DOMAIN_VERTEX) && ((bUseMask == 0) || (bMask == true)) )
	{
		// Need this, we are adding on to existing d/dt N,NT :
		//memcpy(&ourrates, NTadditionrates + iVertex, sizeof(NTrates));

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

			// The idea of not sending blocks full of non-domain vertices is another idea. Fiddly with indices.
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
			T_clock = p__T[indexneigh];
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
			kappa_out = shared_kappa[indexneigh - StartMajor];
			nu_out = shared_nu[indexneigh - StartMajor];
		}
		else {
			structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
			pos_out = info2.pos;
			T_out = p__T[indexneigh]; // saved nothing here, only in loading
			kappa_out = p__kappa_major[indexneigh];
			nu_out = p__nu_major[indexneigh];
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

//		if (T_clock == 0.0) {
//			T_clock = 0.5*(shared_T[threadIdx.x] + T_out);
//		};
//		Mimic


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
				T_anti = p__T[indexneigh];
			};
			if (PBC == NEEDS_ANTI) {
				pos_anti = Anticlock_rotate2(pos_anti);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_anti = Clockwise_rotate2(pos_anti);
			};

			//if (T_anti == 0.0) {
			//	T_anti = 0.5*(shared_T[threadIdx.x] + T_out);
			//}; // So we are receiving 0 then doing this. But how come?
			//Mimic
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
			
			// SMARTY:
			if (TestDomainPos(pos_out))
			{				
				if (iSpecies == 0) {
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						kappa_out = shared_kappa[indexneigh - StartMajor];
					}
					else {
						kappa_out = p__kappa_major[indexneigh];
					};
					f64 kappa_par = sqrt(kappa_out * shared_kappa[threadIdx.x]);
					//if ((!TestDomainPos(pos_clock) ) ||	(!TestDomainPos(pos_anti)))
					{
						f64 edgelen = edge_normal.modulus();
					//	ourrates.NnTn += TWOTHIRDS * kappa_par * edgelen *
					//		(T_out - shared_T[threadIdx.x]) / (pos_out - info.pos).modulus();

						result += TWOTHIRDS * kappa_par *(-1.0)*(edgelen / (pos_out - info.pos).modulus());
						// why are we ever doing anything else?

						// .. for neutral heat cond, it turns out we basically never even use T_anti, T_clock.

#ifdef CENTROID_HEATCONDUCTION
						printf("you know this doesn't match? Need circumcenters for longitudinal heat flux.\n");
#endif
					}					
				} else {
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						B_out = shared_B[indexneigh - StartMajor];
						kappa_out = shared_kappa[indexneigh - StartMajor];
						nu_out = shared_nu[indexneigh - StartMajor];
					}
					else {
						f64_vec3 B_out3 = p_B_major[indexneigh];
						B_out = B_out3.xypart();
						kappa_out = p__kappa_major[indexneigh];
						nu_out = p__nu_major[indexneigh];
					};
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if (PBC == NEEDS_ANTI) 	B_out = Anticlock_rotate2(B_out);
					if (PBC == NEEDS_CLOCK)	B_out = Clockwise_rotate2(B_out);
					
					{ // scoping brace
						f64_vec3 omega;
						if (iSpecies == 1) {
							omega = Make3(qoverMc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qoverMc);
						}
						else {
							omega = Make3(qovermc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qovermc);
						};

						// GEOMETRIC KAPPA:
						f64 kappa_par = sqrt(kappa_out * shared_kappa[threadIdx.x]);
						f64 nu = sqrt(nu_out * shared_nu[threadIdx.x]);

						f64 edgelen = edge_normal.modulus();
						f64 delta_out = sqrt((info.pos.x - pos_out.x)*(info.pos.x - pos_out.x) + (info.pos.y - pos_out.y)*(info.pos.y - pos_out.y));

						// Now it's the nitty-gritty.
					//	f64 sqrt_Tout_Tanti, sqrt_Tout_Tclock, sqrt_Tours_Tanti, sqrt_Tours_Tclock;


						// For the B-diffusive part: infer grad T on the "green hexagon" and dot with omega, 
						// although we could proceed instead by just inferring grad_b T going around the hexagon 
						// and asking how far we went perpendicular to b. Let's not.

						// The hexagon is formed from the opposing vertex positions and midpoints of triangle sides, 
						// assuming values sqrt(T1 T2) at each midpoint.

						//if ((T_out > 0.0) && (T_anti > 0.0)) {
						//	sqrt_Tout_Tanti = sqrt(T_out*T_anti);
						//}
						//else {
						//	sqrt_Tout_Tanti = 0.0;
						//}
						//if ((T_out > 0.0) && (T_clock > 0.0)) {
						//	sqrt_Tout_Tclock = sqrt(T_out*T_clock);
						//}
						//else {
						//	sqrt_Tout_Tclock = 0.0;
						//}

						//if ((shared_T[threadIdx.x] > 0.0) && (T_anti > 0.0)) {
						//	sqrt_Tours_Tanti = sqrt(shared_T[threadIdx.x] * T_anti);
						//}
						//else {
						//	sqrt_Tours_Tanti = 0.0;
						//}
						//if ((shared_T[threadIdx.x] > 0.0) && (T_clock > 0.0)) {
						//	sqrt_Tours_Tclock = sqrt(shared_T[threadIdx.x] * T_clock);
						//}
						//else {
						//	sqrt_Tours_Tclock = 0.0;
						//}
					
						f64_vec2 pos_ours = shared_pos_verts[threadIdx.x];
						
						// Simplify:

						// Simplify:
						f64 Area_hex = 0.25*(
							(1.5*pos_out.x + 0.5*pos_anti.x)*(pos_anti.y - pos_out.y)
							+ (0.5*(pos_out.x + pos_anti.x) + 0.5*(pos_ours.x + pos_anti.x))*(pos_ours.y - pos_out.y)
							+ (0.5*(pos_ours.x + pos_anti.x) + pos_ours.x)*(pos_ours.y - pos_anti.y)
							+ (0.5*(pos_ours.x + pos_clock.x) + pos_ours.x)*(pos_clock.y - pos_ours.y)
							+ (0.5*(pos_ours.x + pos_clock.x) + 0.5*(pos_out.x + pos_clock.x))*(pos_out.y - pos_ours.y)
							+ (0.5*(pos_out.x + pos_clock.x) + pos_out.x)*(pos_out.y - pos_clock.y)
							);

						//grad_T.x = 0.25*(
						//	(T_out + sqrt_Tout_Tanti)*(pos_anti.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + sqrt_Tout_Tanti - sqrt_Tours_Tclock - sqrt_Tout_Tclock)*(pos_ours.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + shared_T[threadIdx.x])*(pos_ours.y - pos_anti.y)
						//	+ (sqrt_Tours_Tclock + shared_T[threadIdx.x])*(pos_clock.y - pos_ours.y)
						//	+ (sqrt_Tout_Tclock + T_out)*(pos_out.y - pos_clock.y)
						//	);
						// could simplify further to just take coeff on each T value.
						f64_vec2 coeffself_grad_T, coeffsqrt_grad_T;

						coeffself_grad_T.x = 0.25*(pos_clock.y - pos_anti.y) / Area_hex;
						coeffself_grad_T.y = 0.25*(pos_anti.x - pos_clock.x) / Area_hex;

						//grad_T.x = 0.25*(
						//	(T_out + sqrt_Tout_Tanti)*(pos_anti.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + sqrt_Tout_Tanti - sqrt_Tours_Tclock - sqrt_Tout_Tclock)*(pos_ours.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + shared_T[threadIdx.x])*(pos_ours.y - pos_anti.y)
						//	+ (sqrt_Tours_Tclock + shared_T[threadIdx.x])*(pos_clock.y - pos_ours.y)
						//	+ (sqrt_Tout_Tclock + T_out)*(pos_out.y - pos_clock.y)
						//	);

						f64 sqrt_Tanti, sqrt_Tclock;
						if (T_anti > 0.0) {
							sqrt_Tanti = sqrt(T_anti);
						} else {
							sqrt_Tanti = 0.0;
						};
						if (T_clock > 0.0) {
							sqrt_Tclock = sqrt(T_clock);
						} else {
							sqrt_Tclock = 0.0;
						};

						coeffsqrt_grad_T.x = 0.25*(						
							(sqrt_Tanti - sqrt_Tclock)*(pos_ours.y - pos_out.y) 
							+ sqrt_Tanti*(pos_ours.y-pos_anti.y) 
							+ sqrt_Tclock*(pos_clock.y-pos_ours.y)							
							) / Area_hex;

						coeffsqrt_grad_T.y = -0.25*(
							(sqrt_Tanti - sqrt_Tclock)*(pos_ours.x - pos_out.x)
							+ sqrt_Tanti*(pos_ours.x - pos_anti.x)
							+ sqrt_Tclock*(pos_clock.x - pos_ours.x)
							) / Area_hex;
							
							// Isotropic part:
						f64 result_coeff_self = TWOTHIRDS * kappa_par *(									
								nu*nu *  //(T_out - shared_T[threadIdx.x]) * (edgelen / delta_out)
								(-1.0)*(edgelen/delta_out)
								+ (omega.dotxy(coeffself_grad_T))*(omega.dotxy(edge_normal))
								)/ (nu * nu + omega.dot(omega))
								;
						f64 result_coeff_sqrt = TWOTHIRDS * kappa_par *(
							 (omega.dotxy(coeffsqrt_grad_T))*(omega.dotxy(edge_normal))
							) / (nu * nu + omega.dot(omega))
							;
						
						f64 over_sqrt_T_ours;

						if (shared_T[threadIdx.x] > 0.0) {
							over_sqrt_T_ours = 1.0 / sqrt(shared_T[threadIdx.x]);
						} else {
							over_sqrt_T_ours = 0.0; // if shared_T wasn't > 0 then all sqrt terms involving it were evaluated 0.
						}

						result += result_coeff_self + 0.5*result_coeff_sqrt*over_sqrt_T_ours;

						// Let's be careful ---- if we ARE dealing with a value below zero, all the sqrt come out as 0,
						// so the contribution of 0 to deps/dT is correct.
					};
				}; // if iSpecies == 0

			} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

				// Now go round:	
			endpt_clock = endpt_anti;
			pos_clock = pos_out;
			pos_out = pos_anti;
			T_clock = T_out;
			T_out = T_anti;
		}; // next iNeigh
				
	}; // DOMAIN vertex active in mask

	// Turned out to be stupid having a struct called NTrates. We just want to modify one scalar at a time.
	
	p___result[iVertex] = result;

	//memcpy(NTadditionrates + iVertex, &ourrates, sizeof(NTrates));
}


__global__ void kernelAccumulateDiffusiveHeatRate_1species_Geometric(
	structural * __restrict__ p_info_minor,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,
	long * __restrict__ izTri_verts,
	char * __restrict__ szPBCtri_verts,
	f64_vec2 * __restrict__ p_cc,

	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p__T,
	f64_vec3 * __restrict__ p_B_major,

	f64 * __restrict__ p__kappa_major,
	f64 * __restrict__ p__nu_major,

	NTrates * __restrict__ NTadditionrates,
	f64 * __restrict__ p_AreaMajor,
	// scrap masking for now --- but bring it back intelligently???

	bool * __restrict__ p_maskbool,
	bool * __restrict__ p_maskblock,
	bool bUseMask,

	// Just hope that our clever version will converge fast.

	int iSpecies)
{

	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajorClever]; // 2
																	 // DO NOT WANT:
	__shared__ f64_vec2 shared_pos[2 * threadsPerTileMajorClever]; // but as far as we know, we are having to use circumcenters.
																   // Maybe it works without them now that we have the longitudinal assumptions --- don't know for sure.
																   // But it means we are not being consistent with our definition of a cell?
																   // Like having major cells Voronoi => velocity living on centroids (which it must, for visc + A) is in slightly the wrong place.

	__shared__ f64 shared_T[threadsPerTileMajorClever];

	__shared__ f64_vec2 shared_B[threadsPerTileMajorClever]; // +2

															 // B is smooth. Unfortunately we have not fitted in Bz here.
															 // In order to do that perhaps rewrite so that variables are overwritten in shared.
															 // We do not need all T and nu in shared at the same time.
															 // This way is easier for NOW.
	__shared__ f64 shared_kappa[threadsPerTileMajorClever];
	__shared__ f64 shared_nu[threadsPerTileMajorClever];

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
	bool bMask;

	if (bUseMask)
		if (p_maskblock[blockIdx.x] == 0) return;

	// skip out on mask 
	if (bUseMask) bMask = p_maskbool[iVertex];

	structural info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];     // 3 double
	shared_pos_verts[threadIdx.x] = info.pos;
#ifdef CENTROID_HEATCONDUCTION
	{
		structural infotemp[2];
		memcpy(infotemp, p_info_minor + 2 * iVertex, 2 * sizeof(structural));
		shared_pos[threadIdx.x * 2] = infotemp[0].pos;
		shared_pos[threadIdx.x * 2 + 1] = infotemp[1].pos;
		// No nu to set for neutrals - not used
	}
#else
	{
		memcpy(&(shared_pos[threadIdx.x * 2]), p_cc + 2 * iVertex, 2 * sizeof(f64_vec2));
	}
#endif

	memcpy(&(shared_nu[threadIdx.x]), p__nu_major + iVertex, sizeof(f64));
	memcpy(&(shared_kappa[threadIdx.x]), p__kappa_major + iVertex, sizeof(f64));

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {
		shared_B[threadIdx.x] = p_B_major[iVertex].xypart();
		shared_T[threadIdx.x] = p__T[iVertex];
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
						   //f64 kappa_parallel; // do we use them all at once or can we save 2 doubles here?
						   //f64 nu;                // 20 there  
	f64_vec2 edge_normal;  // 22
	f64_vec2 endpt_anti;    // 24 .. + 6 from above
	long indexneigh;     // into the 2-double buffer in L1
	f64_vec2 endpt_clock;    // As we only use endpt_anti afterwords we could union endpt_clock with edge_normal
							 // Come back and optimize by checking which things we need in scope at the same time?
	f64 kappa_out, nu_out;

	short iNeigh; // only fixed # of addresses so short makes no difference.
	char PBC; // char makes no difference.

	if ((info.flag == DOMAIN_VERTEX) && ((bUseMask == 0) || (bMask == true)))
	{
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
		memcpy(izTri, //+ MAXNEIGH_d * threadIdx.x,
			izTri_verts + MAXNEIGH_d * iVertex, MAXNEIGH_d * sizeof(long));

		// The idea of not sending blocks full of non-domain vertices is another idea. Fiddly with indices.
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
			T_clock = p__T[indexneigh];
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
			kappa_out = shared_kappa[indexneigh - StartMajor];
			nu_out = shared_nu[indexneigh - StartMajor];
		}
		else {
			structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
			pos_out = info2.pos;
			T_out = p__T[indexneigh]; // saved nothing here, only in loading
			kappa_out = p__kappa_major[indexneigh];
			nu_out = p__nu_major[indexneigh];
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

		//		if (T_clock == 0.0) {
		//			T_clock = 0.5*(shared_T[threadIdx.x] + T_out);
		//		};
		//		Mimic


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
				T_anti = p__T[indexneigh];
			};
			if (PBC == NEEDS_ANTI) {
				pos_anti = Anticlock_rotate2(pos_anti);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_anti = Clockwise_rotate2(pos_anti);
			};

			//if (T_anti == 0.0) {
			//	T_anti = 0.5*(shared_T[threadIdx.x] + T_out);
			//}; // So we are receiving 0 then doing this. But how come?
			//Mimic
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

			// SMARTY:
			if (TestDomainPos(pos_out))
			{
				if (iSpecies == 0) {
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						kappa_out = shared_kappa[indexneigh - StartMajor];
					}
					else {
						kappa_out = p__kappa_major[indexneigh];
					};
					f64 kappa_par = sqrt(kappa_out * shared_kappa[threadIdx.x]);
					//if ((!TestDomainPos(pos_clock) ) ||	(!TestDomainPos(pos_anti)))
					{
						f64 edgelen = edge_normal.modulus();
						ourrates.NnTn += TWOTHIRDS * kappa_par * edgelen *
							(T_out - shared_T[threadIdx.x]) / (pos_out - info.pos).modulus();

						if (0) printf("%d %d kappa_par %1.10E edgelen %1.10E delta %1.10E T %1.10E \n"
							"T_out %1.14E contrib %1.14E flux coefficient on T_out %1.14E\n",
							iVertex, indexneigh, kappa_par, edgelen, (pos_out - info.pos).modulus(), shared_T[threadIdx.x], T_out,
							TWOTHIRDS * kappa_par * edgelen *
							(T_out - shared_T[threadIdx.x]) / (pos_out - info.pos).modulus(),
							TWOTHIRDS * kappa_par * edgelen / (pos_out - info.pos).modulus()	);
						
						// why are we ever doing anything else?

						// .. for neutral heat cond, it turns out we basically never even use T_anti, T_clock.

#ifdef CENTROID_HEATCONDUCTION
						printf("you know this doesn't match? Need circumcenters for longitudinal heat flux.\n");
#endif
					}
				}
				else {
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						B_out = shared_B[indexneigh - StartMajor];
						kappa_out = shared_kappa[indexneigh - StartMajor];
						nu_out = shared_nu[indexneigh - StartMajor];
					}
					else {
						f64_vec3 B_out3 = p_B_major[indexneigh];
						B_out = B_out3.xypart();
						kappa_out = p__kappa_major[indexneigh];
						nu_out = p__nu_major[indexneigh];
					};
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if (PBC == NEEDS_ANTI) 	B_out = Anticlock_rotate2(B_out);
					if (PBC == NEEDS_CLOCK)	B_out = Clockwise_rotate2(B_out);

					{ // scoping brace
						f64_vec3 omega;
						if (iSpecies == 1) {
							omega = Make3(qoverMc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qoverMc);
						}
						else {
							omega = Make3(qovermc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qovermc);
						};

						// GEOMETRIC KAPPA:
						f64 kappa_par = sqrt(kappa_out * shared_kappa[threadIdx.x]);
						f64 nu = sqrt(nu_out * shared_nu[threadIdx.x]);

						f64 edgelen = edge_normal.modulus();
						f64 delta_out = sqrt((info.pos.x - pos_out.x)*(info.pos.x - pos_out.x) + (info.pos.y - pos_out.y)*(info.pos.y - pos_out.y));

						// Now it's the nitty-gritty.
						f64 sqrt_Tout_Tanti, sqrt_Tout_Tclock, sqrt_Tours_Tanti, sqrt_Tours_Tclock;

						// For the B-diffusive part: infer grad T on the "green hexagon" and dot with omega, 
						// although we could proceed instead by just inferring grad_b T going around the hexagon 
						// and asking how far we went perpendicular to b. Let's not.

						// The hexagon is formed from the opposing vertex positions and midpoints of triangle sides, 
						// assuming values sqrt(T1 T2) at each midpoint.

						if ((T_out > 0.0) && (T_anti > 0.0)) {
							sqrt_Tout_Tanti = sqrt(T_out*T_anti);
						}
						else {
							sqrt_Tout_Tanti = 0.0;
						}
						if ((T_out > 0.0) && (T_clock > 0.0)) {
							sqrt_Tout_Tclock = sqrt(T_out*T_clock);
						}
						else {
							sqrt_Tout_Tclock = 0.0;
						}

						if ((shared_T[threadIdx.x] > 0.0) && (T_anti > 0.0)) {
							sqrt_Tours_Tanti = sqrt(shared_T[threadIdx.x] * T_anti);
						}
						else {
							sqrt_Tours_Tanti = 0.0;
						}
						if ((shared_T[threadIdx.x] > 0.0) && (T_clock > 0.0)) {
							sqrt_Tours_Tclock = sqrt(shared_T[threadIdx.x] * T_clock);
						}
						else {
							sqrt_Tours_Tclock = 0.0;
						}
						//grad_T.x = 0.5*(T_out + sqrt_Tout_Tanti)*0.5*(pos_anti.y - pos_out.y)
						//	+ 0.5*(sqrt_Tours_Tanti + sqrt_Tout_Tanti)*
						//	//(0.5*(pos_ours.y + pos_anti.y) - 0.5*(pos_out.y + pos_anti.y))
						//	0.5*(pos_ours.y - pos_out.y)
						//	+ 0.5*(sqrt_Tours_Tanti + T_ours)*0.5*(pos_ours.y - pos_anti.y)
						//	+ 0.5*(sqrt_Tours_Tclock + T_ours)*0.5*(pos_clock.y - pos_ours.y)
						//	+ 0.5*(sqrt_Tours_Tclock + sqrt_Tout_Tclock)*
						//	0.5*(pos_out.y - pos_ours.y)
						//	+ 0.5*(sqrt_Tout_Tclock + T_out)*0.5*(pos_out.y - pos_clock.y);

						f64_vec2 pos_ours = shared_pos_verts[threadIdx.x];
						// Simplify:
						grad_T.x = 0.25*(
							(T_out + sqrt_Tout_Tanti)*(pos_anti.y - pos_out.y)
							+ (sqrt_Tours_Tanti + sqrt_Tout_Tanti - sqrt_Tours_Tclock - sqrt_Tout_Tclock)*(pos_ours.y - pos_out.y)
							+ (sqrt_Tours_Tanti + shared_T[threadIdx.x])*(pos_ours.y - pos_anti.y)
							+ (sqrt_Tours_Tclock + shared_T[threadIdx.x])*(pos_clock.y - pos_ours.y)
							+ (sqrt_Tout_Tclock + T_out)*(pos_out.y - pos_clock.y)
							);
						// could simplify further to just take coeff on each T value.

						grad_T.y = -0.25*(
							(T_out + sqrt_Tout_Tanti)*(pos_anti.x - pos_out.x)
							+ (sqrt_Tours_Tanti + sqrt_Tout_Tanti - sqrt_Tours_Tclock - sqrt_Tout_Tclock)*(pos_ours.x - pos_out.x)
							+ (sqrt_Tours_Tanti + shared_T[threadIdx.x])*(pos_ours.x - pos_anti.x)
							+ (sqrt_Tours_Tclock + shared_T[threadIdx.x])*(pos_clock.x - pos_ours.x)
							+ (sqrt_Tout_Tclock + T_out)*(pos_out.x - pos_clock.x)
							);

						// Integrate 1 : = integral of df/dx for f(x,y) = x.
						//		Area_hex = 0.5*(pos_out.x + 0.5*(pos_out.x+pos_anti.x))*0.5*(pos_anti.y - pos_out.y)
						//			+ 0.5*(0.5*(pos_out.x+pos_anti.x) + 0.5*(pos_ours.x+pos_anti.x))*0.5*(pos_ours.y - pos_out.y)
						//			+ 0.5*(0.5*(pos_ours.x+pos_anti.x) + pos_ours.x)*0.5*(pos_ours.y - pos_anti.y)
						//			+ 0.5*(0.5*(pos_ours.x+pos_clock.x) + pos_ours.x)*0.5*(pos_clock.y - pos_ours.y)
						//			+ 0.5*(0.5*(pos_ours.x + pos_clock.x) + 0.5*(pos_out.x+pos_clock.x))*0.5*(pos_out.y - pos_ours.y)
						//			+ 0.5*(0.5*(pos_out.x + pos_clock.x) + pos_out.x)*0.5*(pos_out.y - pos_clock.y);

						// Simplify:
						f64 Area_hex = 0.25*(
							(1.5*pos_out.x + 0.5*pos_anti.x)*(pos_anti.y - pos_out.y)
							+ (0.5*(pos_out.x + pos_anti.x) + 0.5*(pos_ours.x + pos_anti.x))*(pos_ours.y - pos_out.y)
							+ (0.5*(pos_ours.x + pos_anti.x) + pos_ours.x)*(pos_ours.y - pos_anti.y)
							+ (0.5*(pos_ours.x + pos_clock.x) + pos_ours.x)*(pos_clock.y - pos_ours.y)
							+ (0.5*(pos_ours.x + pos_clock.x) + 0.5*(pos_out.x + pos_clock.x))*(pos_out.y - pos_ours.y)
							+ (0.5*(pos_out.x + pos_clock.x) + pos_out.x)*(pos_out.y - pos_clock.y)
							);

						grad_T.x /= Area_hex;
						grad_T.y /= Area_hex;

						if (iSpecies == 1) {

							// Isotropic part:
							ourrates.NiTi += TWOTHIRDS * kappa_par *(
								nu*nu * (T_out - shared_T[threadIdx.x]) * (edgelen / delta_out)
								+
								(omega.dotxy(grad_T))*(omega.dotxy(edge_normal))
								) / (nu * nu + omega.dot(omega))
								;
						}
						else {
							ourrates.NeTe += TWOTHIRDS * kappa_par *(
								nu*nu * (T_out - shared_T[threadIdx.x]) * (edgelen / delta_out)
								+
								(omega.dotxy(grad_T))*(omega.dotxy(edge_normal))
								) / (nu * nu + omega.dot(omega))
								; // same thing
						};

						//if (TESTHEAT) 
						//	printf("%d %d iSpecies %d contrib %1.10E kappa_par %1.9E\nT_out %1.9E T %1.9E nu %1.9E omega %1.9E %1.9E\n", iVertex, iNeigh, iSpecies,
						//		TWOTHIRDS * kappa_parallel *  (T_out - shared_T[threadIdx.x]) *
						//		(nu*nu*edgelen*edgelen + omega.dotxy(edge_normal)*omega.dotxy(edge_normal))
						//		/ (delta_out*edgelen *(nu * nu + omega.dot(omega))),
						//		kappa_parallel, T_out, shared_T[threadIdx.x], nu, omega.x, omega.y
						//	);

					}
				}; // if iSpecies == 0

			} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

			  // Now go round:	
			endpt_clock = endpt_anti;
			pos_clock = pos_out;
			pos_out = pos_anti;
			T_clock = T_out;
			T_out = T_anti;
		}; // next iNeigh

	}; // DOMAIN vertex active in mask

	   // Turned out to be stupid having a struct called NTrates. We just want to modify one scalar at a time.

	memcpy(NTadditionrates + iVertex, &ourrates, sizeof(NTrates));
}

__global__ void kernelAccumulateDiffusiveHeatRateROC_wrt_regressor_1species_Geometric(
	structural * __restrict__ p_info_minor,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,
	long * __restrict__ izTri_verts,
	char * __restrict__ szPBCtri_verts,
	f64_vec2 * __restrict__ p_cc,

	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p__T,
	f64 const h_use,
	f64 * __restrict__ p__x,
	f64_vec3 * __restrict__ p_B_major,

	f64 * __restrict__ p__kappa_major,
	f64 * __restrict__ p__nu_major,

	f64 * __restrict__ p___result, // d/dbeta of d(NT)/dt in this cell

	f64 * __restrict__ p_AreaMajor,
	// scrap masking for now --- but bring it back intelligently???

	bool * __restrict__ p_maskbool,
	bool * __restrict__ p_maskblock,
	bool bUseMask,

	// Just hope that our clever version will converge fast.

	int iSpecies)
{

	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajorClever]; // 2
																	 // DO NOT WANT:
	__shared__ f64_vec2 shared_pos[2 * threadsPerTileMajorClever]; // but as far as we know, we are having to use circumcenters.
																   // Maybe it works without them now that we have the longitudinal assumptions --- don't know for sure.
																   // But it means we are not being consistent with our definition of a cell?
																   // Like having major cells Voronoi => velocity living on centroids (which it must, for visc + A) is in slightly the wrong place.

	__shared__ f64 shared_T[threadsPerTileMajorClever];

	__shared__ f64_vec2 shared_B[threadsPerTileMajorClever]; // +2

															 // B is smooth. Unfortunately we have not fitted in Bz here.
															 // In order to do that perhaps rewrite so that variables are overwritten in shared.
															 // We do not need all T and nu in shared at the same time.
															 // This way is easier for NOW.
	__shared__ f64 shared_kappa[threadsPerTileMajorClever];
	__shared__ f64 shared_nu[threadsPerTileMajorClever];

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
	bool bMask;
	f64 result;

	if (bUseMask)
		if (p_maskblock[blockIdx.x] == 0) return;

	// skip out on mask 
	if (bUseMask) bMask = p_maskbool[iVertex];

	structural info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];     // 3 double
	shared_pos_verts[threadIdx.x] = info.pos;
#ifdef CENTROID_HEATCONDUCTION
	{
		structural infotemp[2];
		memcpy(infotemp, p_info_minor + 2 * iVertex, 2 * sizeof(structural));
		shared_pos[threadIdx.x * 2] = infotemp[0].pos;
		shared_pos[threadIdx.x * 2 + 1] = infotemp[1].pos;
		// No nu to set for neutrals - not used
	}
#else
	{
		memcpy(&(shared_pos[threadIdx.x * 2]), p_cc + 2 * iVertex, 2 * sizeof(f64_vec2));
	}
#endif

	memcpy(&(shared_nu[threadIdx.x]), p__nu_major + iVertex, sizeof(f64));
	memcpy(&(shared_kappa[threadIdx.x]), p__kappa_major + iVertex, sizeof(f64));

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {
		shared_B[threadIdx.x] = p_B_major[iVertex].xypart();
		shared_T[threadIdx.x] = p__T[iVertex];
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
	f64 our_x, x_clock, x_out, x_anti;
	f64_vec2 pos_clock, pos_anti, pos_out;   // +6
	f64_vec2 B_out;       // +2
	//NTrates ourrates;      // +5
						   //f64 kappa_parallel; // do we use them all at once or can we save 2 doubles here?
						   //f64 nu;                // 20 there  
	f64_vec2 edge_normal;  // 22
	f64_vec2 endpt_anti;    // 24 .. + 6 from above
	long indexneigh;     // into the 2-double buffer in L1
	f64_vec2 endpt_clock;    // As we only use endpt_anti afterwords we could union endpt_clock with edge_normal
							 // Come back and optimize by checking which things we need in scope at the same time?
	f64 kappa_out, nu_out;

	short iNeigh; // only fixed # of addresses so short makes no difference.
	char PBC; // char makes no difference.
	f64 d_by_dbeta = 0.0;

	if ((info.flag == DOMAIN_VERTEX) && ((bUseMask == 0) || (bMask == true)))
	{
		// Need this, we are adding on to existing d/dt N,NT :
	//	memcpy(&ourrates, NTadditionrates + iVertex, sizeof(NTrates));

		our_x = p__x[iVertex];

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

		// The idea of not sending blocks full of non-domain vertices is another idea. Fiddly with indices.
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
			T_clock = p__T[indexneigh];
		};
		PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
		if (PBC == NEEDS_ANTI) {
			pos_clock = Anticlock_rotate2(pos_clock);
		};
		if (PBC == NEEDS_CLOCK) {
			pos_clock = Clockwise_rotate2(pos_clock);
		};
		x_clock = p__x[indexneigh];

		indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
		if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
		{
			pos_out = shared_pos_verts[indexneigh - StartMajor];
			T_out = shared_T[indexneigh - StartMajor];
			kappa_out = shared_kappa[indexneigh - StartMajor];
			nu_out = shared_nu[indexneigh - StartMajor];
		}
		else {
			structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
			pos_out = info2.pos;
			T_out = p__T[indexneigh]; // saved nothing here, only in loading
			kappa_out = p__kappa_major[indexneigh];
			nu_out = p__nu_major[indexneigh];
		};
		PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
		if (PBC == NEEDS_ANTI) {
			pos_out = Anticlock_rotate2(pos_out);
		};
		if (PBC == NEEDS_CLOCK) {
			pos_out = Clockwise_rotate2(pos_out);
		};
		x_out = p__x[indexneigh];

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

		//		if (T_clock == 0.0) {
		//			T_clock = 0.5*(shared_T[threadIdx.x] + T_out);
		//		};
		//		Mimic


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
				T_anti = p__T[indexneigh];
			};
			if (PBC == NEEDS_ANTI) {
				pos_anti = Anticlock_rotate2(pos_anti);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_anti = Clockwise_rotate2(pos_anti);
			};
			x_anti = p__x[indexneigh];

			//if (T_anti == 0.0) {
			//	T_anti = 0.5*(shared_T[threadIdx.x] + T_out);
			//}; // So we are receiving 0 then doing this. But how come?
			//Mimic
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

			// SMARTY:
			if (TestDomainPos(pos_out))
			{
				if (iSpecies == 0) {
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						kappa_out = shared_kappa[indexneigh - StartMajor];
					}
					else {
						kappa_out = p__kappa_major[indexneigh];
					};
					f64 kappa_par = sqrt(kappa_out * shared_kappa[threadIdx.x]);
					//if ((!TestDomainPos(pos_clock) ) ||	(!TestDomainPos(pos_anti)))
					{
						f64 edgelen = edge_normal.modulus();
						//ourrates.NnTn += TWOTHIRDS * kappa_par * edgelen *
						//	(T_out - shared_T[threadIdx.x]) / (pos_out - info.pos).modulus();

						d_by_dbeta += our_x*TWOTHIRDS * kappa_par * edgelen *
							(-1.0) / (pos_out - info.pos).modulus();
						d_by_dbeta += x_out*TWOTHIRDS*kappa_par * edgelen *
							(1.0) / (pos_out - info.pos).modulus();

						// why are we ever doing anything else?

						// .. for neutral heat cond, it turns out we basically never even use T_anti, T_clock.

#ifdef CENTROID_HEATCONDUCTION
						printf("you know this doesn't match? Need circumcenters for longitudinal heat flux.\n");
#endif
					}
				}
				else {
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						B_out = shared_B[indexneigh - StartMajor];
						kappa_out = shared_kappa[indexneigh - StartMajor];
						nu_out = shared_nu[indexneigh - StartMajor];
					}
					else {
						f64_vec3 B_out3 = p_B_major[indexneigh];
						B_out = B_out3.xypart();
						kappa_out = p__kappa_major[indexneigh];
						nu_out = p__nu_major[indexneigh];
					};
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if (PBC == NEEDS_ANTI) 	B_out = Anticlock_rotate2(B_out);
					if (PBC == NEEDS_CLOCK)	B_out = Clockwise_rotate2(B_out);

					{ // scoping brace
						f64_vec3 omega;
						if (iSpecies == 1) {
							omega = Make3(qoverMc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qoverMc);
						}
						else {
							omega = Make3(qovermc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qovermc);
						};

						// GEOMETRIC KAPPA:
						f64 kappa_par = sqrt(kappa_out * shared_kappa[threadIdx.x]);
						f64 nu = sqrt(nu_out * shared_nu[threadIdx.x]);

						f64 edgelen = edge_normal.modulus();
						f64 delta_out = sqrt((info.pos.x - pos_out.x)*(info.pos.x - pos_out.x) + (info.pos.y - pos_out.y)*(info.pos.y - pos_out.y));

						// Now it's the nitty-gritty.
						//	f64 sqrt_Tout_Tanti, sqrt_Tout_Tclock, sqrt_Tours_Tanti, sqrt_Tours_Tclock;



						// For the B-diffusive part: infer grad T on the "green hexagon" and dot with omega, 
						// although we could proceed instead by just inferring grad_b T going around the hexagon 
						// and asking how far we went perpendicular to b. Let's not.

						// The hexagon is formed from the opposing vertex positions and midpoints of triangle sides, 
						// assuming values sqrt(T1 T2) at each midpoint.

						//if ((T_out > 0.0) && (T_anti > 0.0)) {
						//	sqrt_Tout_Tanti = sqrt(T_out*T_anti);
						//}
						//else {
						//	sqrt_Tout_Tanti = 0.0;
						//}
						//if ((T_out > 0.0) && (T_clock > 0.0)) {
						//	sqrt_Tout_Tclock = sqrt(T_out*T_clock);
						//}
						//else {
						//	sqrt_Tout_Tclock = 0.0;
						//}

						//if ((shared_T[threadIdx.x] > 0.0) && (T_anti > 0.0)) {
						//	sqrt_Tours_Tanti = sqrt(shared_T[threadIdx.x] * T_anti);
						//}
						//else {
						//	sqrt_Tours_Tanti = 0.0;
						//}
						//if ((shared_T[threadIdx.x] > 0.0) && (T_clock > 0.0)) {
						//	sqrt_Tours_Tclock = sqrt(shared_T[threadIdx.x] * T_clock);
						//}
						//else {
						//	sqrt_Tours_Tclock = 0.0;
						//}

						f64_vec2 pos_ours = shared_pos_verts[threadIdx.x];
						
						// Simplify:
						f64 Area_hex = 0.25*(
							(1.5*pos_out.x + 0.5*pos_anti.x)*(pos_anti.y - pos_out.y)
							+ (0.5*(pos_out.x + pos_anti.x) + 0.5*(pos_ours.x + pos_anti.x))*(pos_ours.y - pos_out.y)
							+ (0.5*(pos_ours.x + pos_anti.x) + pos_ours.x)*(pos_ours.y - pos_anti.y)
							+ (0.5*(pos_ours.x + pos_clock.x) + pos_ours.x)*(pos_clock.y - pos_ours.y)
							+ (0.5*(pos_ours.x + pos_clock.x) + 0.5*(pos_out.x + pos_clock.x))*(pos_out.y - pos_ours.y)
							+ (0.5*(pos_out.x + pos_clock.x) + pos_out.x)*(pos_out.y - pos_clock.y)
							);

						//grad_T.x = 0.25*(
						//	(T_out + sqrt_Tout_Tanti)*(pos_anti.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + sqrt_Tout_Tanti - sqrt_Tours_Tclock - sqrt_Tout_Tclock)*(pos_ours.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + shared_T[threadIdx.x])*(pos_ours.y - pos_anti.y)
						//	+ (sqrt_Tours_Tclock + shared_T[threadIdx.x])*(pos_clock.y - pos_ours.y)
						//	+ (sqrt_Tout_Tclock + T_out)*(pos_out.y - pos_clock.y)
						//	);
						// could simplify further to just take coeff on each T value.
						f64_vec2 coeffself_grad_T, coeffsqrt_grad_T;

						coeffself_grad_T.x = 0.25*(pos_clock.y - pos_anti.y) / Area_hex;
						coeffself_grad_T.y = 0.25*(pos_anti.x - pos_clock.x) / Area_hex;

						//grad_T.x = 0.25*(
						//	(T_out + sqrt_Tout_Tanti)*(pos_anti.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + sqrt_Tout_Tanti - sqrt_Tours_Tclock - sqrt_Tout_Tclock)*(pos_ours.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + shared_T[threadIdx.x])*(pos_ours.y - pos_anti.y)
						//	+ (sqrt_Tours_Tclock + shared_T[threadIdx.x])*(pos_clock.y - pos_ours.y)
						//	+ (sqrt_Tout_Tclock + T_out)*(pos_out.y - pos_clock.y)
						//	);

						f64 sqrt_Tanti, sqrt_Tclock, sqrt_Tours, sqrt_Tout;
						if (T_anti > 0.0) {
							sqrt_Tanti = sqrt(T_anti);
						}
						else {
							sqrt_Tanti = 0.0;
						};
						if (T_clock > 0.0) {
							sqrt_Tclock = sqrt(T_clock);
						} else {
							sqrt_Tclock = 0.0;
						};
						if (shared_T[threadIdx.x] > 0.0) {
							sqrt_Tours = sqrt(shared_T[threadIdx.x]);
						} else {
							sqrt_Tours = 0.0;
						};
						if (T_out > 0.0) {
							sqrt_Tout = sqrt(T_out);
						} else {
							sqrt_Tout = 0.0;
						};

						coeffsqrt_grad_T.x = 0.25*(
							(sqrt_Tanti - sqrt_Tclock)*(pos_ours.y - pos_out.y)
							+ sqrt_Tanti*(pos_ours.y - pos_anti.y)
							+ sqrt_Tclock*(pos_clock.y - pos_ours.y)
							) / Area_hex;

						coeffsqrt_grad_T.y = -0.25*(
							(sqrt_Tanti - sqrt_Tclock)*(pos_ours.x - pos_out.x)
							+ sqrt_Tanti*(pos_ours.x - pos_anti.x)
							+ sqrt_Tclock*(pos_clock.x - pos_ours.x)
							) / Area_hex;

						// Isotropic part:
						f64 result_coeff_self = TWOTHIRDS * kappa_par *(
							nu*nu *  //(T_out - shared_T[threadIdx.x]) * (edgelen / delta_out)
							(-1.0)*(edgelen / delta_out)
							+ (omega.dotxy(coeffself_grad_T))*(omega.dotxy(edge_normal))
							) / (nu * nu + omega.dot(omega))
							;

						// d/dself:
						f64 result = result_coeff_self;

						if (shared_T[threadIdx.x] > 0.0) {		
											
							f64 result_coeff_sqrt = TWOTHIRDS * kappa_par *(
								(omega.dotxy(coeffsqrt_grad_T))*(omega.dotxy(edge_normal))
								) / (nu * nu + omega.dot(omega));								
							result += 0.5*result_coeff_sqrt / sqrt(shared_T[threadIdx.x]);
						}; // else sqrt term didn't have an effect.
												
						d_by_dbeta += our_x*result;

						//grad_T.x = 0.25*(
						//	(T_out + sqrt_Tout_Tanti)*(pos_anti.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + sqrt_Tout_Tanti - sqrt_Tours_Tclock - sqrt_Tout_Tclock)*(pos_ours.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + shared_T[threadIdx.x])*(pos_ours.y - pos_anti.y)
						//	+ (sqrt_Tours_Tclock + shared_T[threadIdx.x])*(pos_clock.y - pos_ours.y)
						//	+ (sqrt_Tout_Tclock + T_out)*(pos_out.y - pos_clock.y)
						//	);

						// coeff on power 1:
						f64_vec2 ROC_grad_wrt_T_out;
						ROC_grad_wrt_T_out.x = 0.25*(pos_anti.y - pos_clock.y) / Area_hex;
						ROC_grad_wrt_T_out.y = 0.25*(pos_clock.x - pos_anti.x) / Area_hex;

						// stick to format from above :

						// Isotropic part:
						d_by_dbeta += x_out* TWOTHIRDS * kappa_par *(
							nu*nu *  //(T_out - shared_T[threadIdx.x]) * (edgelen / delta_out)
							(1.0)*(edgelen / delta_out)
							+ (omega.dotxy(ROC_grad_wrt_T_out))*(omega.dotxy(edge_normal))
							) / (nu * nu + omega.dot(omega))
							;
						
						if (T_out > 0.0) {
							f64_vec2 coeff_grad_wrt_sqrt_T_out;
							coeff_grad_wrt_sqrt_T_out.x = 0.25*(
								(sqrt_Tanti)*(pos_anti.y - pos_out.y)
								+ (sqrt_Tanti - sqrt_Tclock)*(pos_ours.y - pos_out.y)
								+ (sqrt_Tclock)*(pos_out.y - pos_clock.y)
								) / Area_hex;

							coeff_grad_wrt_sqrt_T_out.y = -0.25*(
								(sqrt_Tanti)*(pos_anti.x - pos_out.x)
								+ (sqrt_Tanti - sqrt_Tclock)*(pos_ours.x - pos_out.x)
								+ (sqrt_Tclock)*(pos_out.x - pos_clock.x)
								) / Area_hex;

							f64 result_coeff_sqrt = TWOTHIRDS * kappa_par *(
								(omega.dotxy(coeff_grad_wrt_sqrt_T_out))*(omega.dotxy(edge_normal))
								) / (nu * nu + omega.dot(omega));
							d_by_dbeta += x_out*0.5*result_coeff_sqrt / sqrt(T_out);
						};

						// T_anti:
						if (T_anti > 0.0) {
							f64_vec2 coeff_grad_wrt_sqrt_T_anti;
							coeff_grad_wrt_sqrt_T_anti.x = 0.25*(
								(sqrt_Tout)*(pos_anti.y - pos_out.y)
								+ (sqrt_Tours + sqrt_Tout)*(pos_ours.y - pos_out.y)
								+ sqrt_Tours*(pos_ours.y - pos_anti.y)
								) / Area_hex;

							coeff_grad_wrt_sqrt_T_anti.y = -0.25*(
								(sqrt_Tout)*(pos_anti.x - pos_out.x)
								+ (sqrt_Tours + sqrt_Tout)*(pos_ours.x - pos_out.x)
								+ sqrt_Tours*(pos_ours.x - pos_anti.x)
								) / Area_hex;

							f64 result_coeff_sqrt = TWOTHIRDS * kappa_par *(
								(omega.dotxy(coeff_grad_wrt_sqrt_T_anti))*(omega.dotxy(edge_normal))
								) / (nu * nu + omega.dot(omega));
							d_by_dbeta += x_anti*0.5*result_coeff_sqrt / sqrt(T_anti);
						};

						if (T_clock > 0.0) {

							//grad_T.x = 0.25*(
							//	(T_out + sqrt_Tout_Tanti)*(pos_anti.y - pos_out.y)
							//	+ (sqrt_Tours_Tanti + sqrt_Tout_Tanti - sqrt_Tours_Tclock - sqrt_Tout_Tclock)*(pos_ours.y - pos_out.y)
							//	+ (sqrt_Tours_Tanti + shared_T[threadIdx.x])*(pos_ours.y - pos_anti.y)
							//	+ (sqrt_Tours_Tclock + shared_T[threadIdx.x])*(pos_clock.y - pos_ours.y)
							//	+ (sqrt_Tout_Tclock + T_out)*(pos_out.y - pos_clock.y)
							//	);
							f64_vec2 coeff_grad_wrt_sqrt_T_clock;
							coeff_grad_wrt_sqrt_T_clock.x = 0.25*(
								-(sqrt_Tours + sqrt_Tout)*(pos_ours.y - pos_out.y)
								+ sqrt_Tours*(pos_clock.y - pos_ours.y)
								+ sqrt_Tout*(pos_out.y - pos_clock.y)
								) / Area_hex;
							coeff_grad_wrt_sqrt_T_clock.y = -0.25*(
								-(sqrt_Tours + sqrt_Tout)*(pos_ours.x - pos_out.x)
								+ sqrt_Tours*(pos_clock.x - pos_ours.x)
								+ sqrt_Tout*(pos_out.x - pos_clock.x)
								) / Area_hex;

							f64 result_coeff_sqrt = TWOTHIRDS * kappa_par *(
								(omega.dotxy(coeff_grad_wrt_sqrt_T_clock))*(omega.dotxy(edge_normal))
								) / (nu * nu + omega.dot(omega));
							d_by_dbeta += x_clock*0.5*result_coeff_sqrt / sqrt(T_clock);
						};

					};
				}; // if iSpecies == 0

			} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

			  // Now go round:	
			endpt_clock = endpt_anti;
			pos_clock = pos_out;
			pos_out = pos_anti;
			T_clock = T_out;
			T_out = T_anti;
			x_clock = x_out;
			x_out = x_anti;
		}; // next iNeigh

		f64 N;
		if (iSpecies == 0) {
			N = p_AreaMajor[iVertex] * p_n_major[iVertex].n_n;
		} else {
			N = p_AreaMajor[iVertex] * p_n_major[iVertex].n;
		};

		result = -d_by_dbeta*(h_use / N) + our_x;

		if (result != result) printf("iVertex %d NaN result. d/dbeta %1.10E N %1.8E our_x %1.8E \n",
			iVertex, d_by_dbeta, N, our_x);


	} else { // was it DOMAIN vertex active in mask
		result = 0.0;
	};
	
	p___result[iVertex] = result;
}


__global__ void kernelAccumulateDiffusiveHeatRate__array_of_deps_by_dxj_1species_Geometric(
	structural * __restrict__ p_info_minor,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,
	long * __restrict__ izTri_verts,
	char * __restrict__ szPBCtri_verts,
	f64_vec2 * __restrict__ p_cc,

	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p__T,
	f64 * __restrict__ p_epsilon,
	f64 const h_use,
	f64_vec3 * __restrict__ p_B_major,

	f64 * __restrict__ p__kappa_major,
	f64 * __restrict__ p__nu_major,

	f64 * __restrict__ p_AreaMajor,
	// scrap masking for now --- but bring it back intelligently???

	bool * __restrict__ p_maskbool,
	bool * __restrict__ p_maskblock,
	bool bUseMask,

	// Just hope that our clever version will converge fast.

	int iSpecies,
	
	f64 * __restrict__ p_array,
	f64 * __restrict__ p_effectself
	)
{

	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajorClever]; // 2
																	 // DO NOT WANT:
	__shared__ f64_vec2 shared_pos[2 * threadsPerTileMajorClever]; // but as far as we know, we are having to use circumcenters.
																   // Maybe it works without them now that we have the longitudinal assumptions --- don't know for sure.
																   // But it means we are not being consistent with our definition of a cell?
																   // Like having major cells Voronoi => velocity living on centroids (which it must, for visc + A) is in slightly the wrong place.

	__shared__ f64 shared_T[threadsPerTileMajorClever];

	__shared__ f64_vec2 shared_B[threadsPerTileMajorClever]; // +2

															 // B is smooth. Unfortunately we have not fitted in Bz here.
															 // In order to do that perhaps rewrite so that variables are overwritten in shared.
															 // We do not need all T and nu in shared at the same time.
															 // This way is easier for NOW.
	__shared__ f64 shared_kappa[threadsPerTileMajorClever];
	__shared__ f64 shared_nu[threadsPerTileMajorClever];

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
	bool bMask;

	if (bUseMask)
		if (p_maskblock[blockIdx.x] == 0) return;
	// skip out on mask 
	if (bUseMask) bMask = p_maskbool[iVertex];

	f64 fzArray[MAXNEIGH_d];  // { deps/dx_j * eps }
	f64 effectself; 
		
	structural info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];     // 3 double
	shared_pos_verts[threadIdx.x] = info.pos;
#ifdef CENTROID_HEATCONDUCTION
	{
		structural infotemp[2];
		memcpy(infotemp, p_info_minor + 2 * iVertex, 2 * sizeof(structural));
		shared_pos[threadIdx.x * 2] = infotemp[0].pos;
		shared_pos[threadIdx.x * 2 + 1] = infotemp[1].pos;
		// No nu to set for neutrals - not used
	}
#else
	{
		memcpy(&(shared_pos[threadIdx.x * 2]), p_cc + 2 * iVertex, 2 * sizeof(f64_vec2));
	}
#endif

	memcpy(&(shared_nu[threadIdx.x]), p__nu_major + iVertex, sizeof(f64));
	memcpy(&(shared_kappa[threadIdx.x]), p__kappa_major + iVertex, sizeof(f64));

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {
		shared_B[threadIdx.x] = p_B_major[iVertex].xypart();
		shared_T[threadIdx.x] = p__T[iVertex];
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
	f64 our_x, x_clock, x_out, x_anti;
	f64_vec2 pos_clock, pos_anti, pos_out;   // +6
	f64_vec2 B_out;       // +2
						  //NTrates ourrates;      // +5
						  //f64 kappa_parallel; // do we use them all at once or can we save 2 doubles here?
						  //f64 nu;                // 20 there  
	f64_vec2 edge_normal;  // 22
	f64_vec2 endpt_anti;    // 24 .. + 6 from above
	long indexneigh;     // into the 2-double buffer in L1
	f64_vec2 endpt_clock;    // As we only use endpt_anti afterwords we could union endpt_clock with edge_normal
							 // Come back and optimize by checking which things we need in scope at the same time?
	f64 kappa_out, nu_out;

	short iNeigh; // only fixed # of addresses so short makes no difference.
	char PBC; // char makes no difference.
	f64 d_by_dbeta = 0.0;

	if ((info.flag == DOMAIN_VERTEX) && ((bUseMask == 0) || (bMask == true)))
	{
		memset(fzArray, 0, sizeof(f64)*MAXNEIGH_d);

		f64 N;
		if (iSpecies == 0) {
			N = p_AreaMajor[iVertex] * p_n_major[iVertex].n_n;
		} else {
			N = p_AreaMajor[iVertex] * p_n_major[iVertex].n;
		}
		f64 epsilon = p_epsilon[iVertex];
		f64 our_fac = -2.0*epsilon* h_use / N; // factor for change in epsilon^2
		// But we missed out the effect of changing T on epsilon directly ! ...

		// need to add 1.0
		effectself = epsilon*2.0; // change in own epsilon by changing T is +1.0 for eps = T_k+1-T_k-hF

		if (TESTHEAT) printf("%d effectself %1.10E our_fac %1.10E \n", iVertex, effectself, our_fac);

		// Need this, we are adding on to existing d/dt N,NT :
		//	memcpy(&ourrates, NTadditionrates + iVertex, sizeof(NTrates));

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

		// The idea of not sending blocks full of non-domain vertices is another idea. Fiddly with indices.
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
			T_clock = p__T[indexneigh];
		};
		PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
		if (PBC == NEEDS_ANTI) {
			pos_clock = Anticlock_rotate2(pos_clock);
		};
		if (PBC == NEEDS_CLOCK) {
			pos_clock = Clockwise_rotate2(pos_clock);
		};
		//x_clock = p__x[indexneigh];

		indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
		if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
		{
			pos_out = shared_pos_verts[indexneigh - StartMajor];
			T_out = shared_T[indexneigh - StartMajor];
			kappa_out = shared_kappa[indexneigh - StartMajor];
			nu_out = shared_nu[indexneigh - StartMajor];
		}
		else {
			structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
			pos_out = info2.pos;
			T_out = p__T[indexneigh]; // saved nothing here, only in loading
			kappa_out = p__kappa_major[indexneigh];
			nu_out = p__nu_major[indexneigh];
		};
		PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
		if (PBC == NEEDS_ANTI) {
			pos_out = Anticlock_rotate2(pos_out);
		};
		if (PBC == NEEDS_CLOCK) {
			pos_out = Clockwise_rotate2(pos_out);
		};
		//x_out = p__x[indexneigh];

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

		//		if (T_clock == 0.0) {
		//			T_clock = 0.5*(shared_T[threadIdx.x] + T_out);
		//		};
		//		Mimic
		short iPrev = info.neigh_len - 1;

#pragma unroll MAXNEIGH_d
		for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
		{
			
			short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNext];
			
			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_anti = shared_pos_verts[indexneigh - StartMajor];
				T_anti = shared_T[indexneigh - StartMajor];
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_anti = info2.pos;
				T_anti = p__T[indexneigh];
			};
			if (PBC == NEEDS_ANTI) {
				pos_anti = Anticlock_rotate2(pos_anti);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_anti = Clockwise_rotate2(pos_anti);
			};
			//x_anti = p__x[indexneigh];

			//if (T_anti == 0.0) {
			//	T_anti = 0.5*(shared_T[threadIdx.x] + T_out);
			//}; // So we are receiving 0 then doing this. But how come?
			//Mimic
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

			// SMARTY:
			if (TestDomainPos(pos_out))
			{
				if (iSpecies == 0) {
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						kappa_out = shared_kappa[indexneigh - StartMajor];
					} else {
						kappa_out = p__kappa_major[indexneigh];
					};
					f64 kappa_par = sqrt(kappa_out * shared_kappa[threadIdx.x]);
					//if ((!TestDomainPos(pos_clock) ) ||	(!TestDomainPos(pos_anti)))
					{
						f64 edgelen = edge_normal.modulus();
						//ourrates.NnTn += TWOTHIRDS * kappa_par * edgelen *
						//	(T_out - shared_T[threadIdx.x]) / (pos_out - info.pos).modulus();

//						d_by_dbeta += our_x*TWOTHIRDS * kappa_par * edgelen *
	//						(-1.0) / (pos_out - info.pos).modulus();
		//				d_by_dbeta += x_out*TWOTHIRDS*kappa_par * edgelen *
			//				(1.0) / (pos_out - info.pos).modulus();

						f64 temp = TWOTHIRDS*kappa_par * edgelen *
							(1.0) / (pos_out - info.pos).modulus();

						fzArray[iNeigh] += temp*our_fac;
						effectself -= temp*our_fac;
						if (TESTHEAT) {
							printf("iVertex %d indexneigh %d temp %1.14E our_fac %1.14E iNeigh %d temp*our_fac %1.14E \n",
								iVertex, indexneigh, temp, our_fac, iNeigh, temp*our_fac);
						}
						// .. for neutral heat cond, it turns out we basically never even use T_anti, T_clock.

#ifdef CENTROID_HEATCONDUCTION
						printf("you know this doesn't match? Need circumcenters for longitudinal heat flux.\n");
#endif
					}
				}
				else {
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						B_out = shared_B[indexneigh - StartMajor];
						kappa_out = shared_kappa[indexneigh - StartMajor];
						nu_out = shared_nu[indexneigh - StartMajor];
					}
					else {
						f64_vec3 B_out3 = p_B_major[indexneigh];
						B_out = B_out3.xypart();
						kappa_out = p__kappa_major[indexneigh];
						nu_out = p__nu_major[indexneigh];
					};
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if (PBC == NEEDS_ANTI) 	B_out = Anticlock_rotate2(B_out);
					if (PBC == NEEDS_CLOCK)	B_out = Clockwise_rotate2(B_out);

					{ // scoping brace
						f64_vec3 omega;
						if (iSpecies == 1) {
							omega = Make3(qoverMc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qoverMc);
						}
						else {
							omega = Make3(qovermc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qovermc);
						};

						// GEOMETRIC KAPPA:
						f64 kappa_par = sqrt(kappa_out * shared_kappa[threadIdx.x]);
						f64 nu = sqrt(nu_out * shared_nu[threadIdx.x]);

						f64 edgelen = edge_normal.modulus();
						f64 delta_out = sqrt((info.pos.x - pos_out.x)*(info.pos.x - pos_out.x) + (info.pos.y - pos_out.y)*(info.pos.y - pos_out.y));

						// Now it's the nitty-gritty.
						//	f64 sqrt_Tout_Tanti, sqrt_Tout_Tclock, sqrt_Tours_Tanti, sqrt_Tours_Tclock;
						
						f64_vec2 pos_ours = shared_pos_verts[threadIdx.x];

						// Simplify:
						f64 Area_hex = 0.25*(
							(1.5*pos_out.x + 0.5*pos_anti.x)*(pos_anti.y - pos_out.y)
							+ (0.5*(pos_out.x + pos_anti.x) + 0.5*(pos_ours.x + pos_anti.x))*(pos_ours.y - pos_out.y)
							+ (0.5*(pos_ours.x + pos_anti.x) + pos_ours.x)*(pos_ours.y - pos_anti.y)
							+ (0.5*(pos_ours.x + pos_clock.x) + pos_ours.x)*(pos_clock.y - pos_ours.y)
							+ (0.5*(pos_ours.x + pos_clock.x) + 0.5*(pos_out.x + pos_clock.x))*(pos_out.y - pos_ours.y)
							+ (0.5*(pos_out.x + pos_clock.x) + pos_out.x)*(pos_out.y - pos_clock.y)
							);

						//grad_T.x = 0.25*(
						//	(T_out + sqrt_Tout_Tanti)*(pos_anti.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + sqrt_Tout_Tanti - sqrt_Tours_Tclock - sqrt_Tout_Tclock)*(pos_ours.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + shared_T[threadIdx.x])*(pos_ours.y - pos_anti.y)
						//	+ (sqrt_Tours_Tclock + shared_T[threadIdx.x])*(pos_clock.y - pos_ours.y)
						//	+ (sqrt_Tout_Tclock + T_out)*(pos_out.y - pos_clock.y)
						//	);
						// could simplify further to just take coeff on each T value.
						f64_vec2 coeffself_grad_T, coeffsqrt_grad_T;

						coeffself_grad_T.x = 0.25*(pos_clock.y - pos_anti.y) / Area_hex;
						coeffself_grad_T.y = 0.25*(pos_anti.x - pos_clock.x) / Area_hex;
						
						f64 sqrt_Tanti, sqrt_Tclock, sqrt_Tours, sqrt_Tout;
						if (T_anti > 0.0) {
							sqrt_Tanti = sqrt(T_anti);
						}
						else {
							sqrt_Tanti = 0.0;
						};
						if (T_clock > 0.0) {
							sqrt_Tclock = sqrt(T_clock);
						}
						else {
							sqrt_Tclock = 0.0;
						};
						if (shared_T[threadIdx.x] > 0.0) {
							sqrt_Tours = sqrt(shared_T[threadIdx.x]);
						}
						else {
							sqrt_Tours = 0.0;
						};
						if (T_out > 0.0) {
							sqrt_Tout = sqrt(T_out);
						}
						else {
							sqrt_Tout = 0.0;
						};

						coeffsqrt_grad_T.x = 0.25*(
							(sqrt_Tanti - sqrt_Tclock)*(pos_ours.y - pos_out.y)
							+ sqrt_Tanti*(pos_ours.y - pos_anti.y)
							+ sqrt_Tclock*(pos_clock.y - pos_ours.y)
							) / Area_hex;

						coeffsqrt_grad_T.y = -0.25*(
							(sqrt_Tanti - sqrt_Tclock)*(pos_ours.x - pos_out.x)
							+ sqrt_Tanti*(pos_ours.x - pos_anti.x)
							+ sqrt_Tclock*(pos_clock.x - pos_ours.x)
							) / Area_hex;

						// Isotropic part:
						f64 result_coeff_self = TWOTHIRDS * kappa_par *(
							nu*nu *  //(T_out - shared_T[threadIdx.x]) * (edgelen / delta_out)
							(-1.0)*(edgelen / delta_out)
							+ (omega.dotxy(coeffself_grad_T))*(omega.dotxy(edge_normal))
							) / (nu * nu + omega.dot(omega))
							;

						// d/dself:
						f64 result = result_coeff_self;

						if (shared_T[threadIdx.x] > 0.0) {

							f64 result_coeff_sqrt = TWOTHIRDS * kappa_par *(
								(omega.dotxy(coeffsqrt_grad_T))*(omega.dotxy(edge_normal))
								) / (nu * nu + omega.dot(omega));
							result += 0.5*result_coeff_sqrt / sqrt(shared_T[threadIdx.x]);
						}; // else sqrt term didn't have an effect.

						//d_by_dbeta += our_x*result;

						effectself += result*our_fac;

						//grad_T.x = 0.25*(
						//	(T_out + sqrt_Tout_Tanti)*(pos_anti.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + sqrt_Tout_Tanti - sqrt_Tours_Tclock - sqrt_Tout_Tclock)*(pos_ours.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + shared_T[threadIdx.x])*(pos_ours.y - pos_anti.y)
						//	+ (sqrt_Tours_Tclock + shared_T[threadIdx.x])*(pos_clock.y - pos_ours.y)
						//	+ (sqrt_Tout_Tclock + T_out)*(pos_out.y - pos_clock.y)
						//	);

						// coeff on power 1:
						f64_vec2 ROC_grad_wrt_T_out;
						ROC_grad_wrt_T_out.x = 0.25*(pos_anti.y - pos_clock.y) / Area_hex;
						ROC_grad_wrt_T_out.y = 0.25*(pos_clock.x - pos_anti.x) / Area_hex;

						// stick to format from above :

						// Isotropic part:
						//d_by_dbeta += x_out* TWOTHIRDS * kappa_par *(
						//	nu*nu *  //(T_out - shared_T[threadIdx.x]) * (edgelen / delta_out)
						//	(1.0)*(edgelen / delta_out)
						//	+ (omega.dotxy(ROC_grad_wrt_T_out))*(omega.dotxy(edge_normal))
						//	) / (nu * nu + omega.dot(omega))
						//	;

						fzArray[iNeigh] += our_fac*TWOTHIRDS * kappa_par *(
							nu*nu *  //(T_out - shared_T[threadIdx.x]) * (edgelen / delta_out)
							(1.0)*(edgelen / delta_out)
							+ (omega.dotxy(ROC_grad_wrt_T_out))*(omega.dotxy(edge_normal))
							) / (nu * nu + omega.dot(omega))
							;

						if (T_out > 0.0) {
							f64_vec2 coeff_grad_wrt_sqrt_T_out;
							coeff_grad_wrt_sqrt_T_out.x = 0.25*(
								(sqrt_Tanti)*(pos_anti.y - pos_out.y)
								+ (sqrt_Tanti - sqrt_Tclock)*(pos_ours.y - pos_out.y)
								+ (sqrt_Tclock)*(pos_out.y - pos_clock.y)
								) / Area_hex;

							coeff_grad_wrt_sqrt_T_out.y = -0.25*(
								(sqrt_Tanti)*(pos_anti.x - pos_out.x)
								+ (sqrt_Tanti - sqrt_Tclock)*(pos_ours.x - pos_out.x)
								+ (sqrt_Tclock)*(pos_out.x - pos_clock.x)
								) / Area_hex;

							f64 result_coeff_sqrt = TWOTHIRDS * kappa_par *(
								(omega.dotxy(coeff_grad_wrt_sqrt_T_out))*(omega.dotxy(edge_normal))
								) / (nu * nu + omega.dot(omega));
							//d_by_dbeta += x_out*0.5*result_coeff_sqrt / sqrt(T_out);

							fzArray[iNeigh] += our_fac*0.5*result_coeff_sqrt / sqrt(T_out);

						};

						// T_anti:
						if (T_anti > 0.0) {
							f64_vec2 coeff_grad_wrt_sqrt_T_anti;
							coeff_grad_wrt_sqrt_T_anti.x = 0.25*(
								(sqrt_Tout)*(pos_anti.y - pos_out.y)
								+ (sqrt_Tours + sqrt_Tout)*(pos_ours.y - pos_out.y)
								+ sqrt_Tours*(pos_ours.y - pos_anti.y)
								) / Area_hex;

							coeff_grad_wrt_sqrt_T_anti.y = -0.25*(
								(sqrt_Tout)*(pos_anti.x - pos_out.x)
								+ (sqrt_Tours + sqrt_Tout)*(pos_ours.x - pos_out.x)
								+ sqrt_Tours*(pos_ours.x - pos_anti.x)
								) / Area_hex;

							f64 result_coeff_sqrt = TWOTHIRDS * kappa_par *(
								(omega.dotxy(coeff_grad_wrt_sqrt_T_anti))*(omega.dotxy(edge_normal))
								) / (nu * nu + omega.dot(omega));
							
							//d_by_dbeta += x_anti*0.5*result_coeff_sqrt / sqrt(T_anti);

							fzArray[iNext] += our_fac*0.5*result_coeff_sqrt / sqrt(T_anti);
						};

						if (T_clock > 0.0) {

							//grad_T.x = 0.25*(
							//	(T_out + sqrt_Tout_Tanti)*(pos_anti.y - pos_out.y)
							//	+ (sqrt_Tours_Tanti + sqrt_Tout_Tanti - sqrt_Tours_Tclock - sqrt_Tout_Tclock)*(pos_ours.y - pos_out.y)
							//	+ (sqrt_Tours_Tanti + shared_T[threadIdx.x])*(pos_ours.y - pos_anti.y)
							//	+ (sqrt_Tours_Tclock + shared_T[threadIdx.x])*(pos_clock.y - pos_ours.y)
							//	+ (sqrt_Tout_Tclock + T_out)*(pos_out.y - pos_clock.y)
							//	);
							f64_vec2 coeff_grad_wrt_sqrt_T_clock;
							coeff_grad_wrt_sqrt_T_clock.x = 0.25*(
								-(sqrt_Tours + sqrt_Tout)*(pos_ours.y - pos_out.y)
								+ sqrt_Tours*(pos_clock.y - pos_ours.y)
								+ sqrt_Tout*(pos_out.y - pos_clock.y)
								) / Area_hex;
							coeff_grad_wrt_sqrt_T_clock.y = -0.25*(
								-(sqrt_Tours + sqrt_Tout)*(pos_ours.x - pos_out.x)
								+ sqrt_Tours*(pos_clock.x - pos_ours.x)
								+ sqrt_Tout*(pos_out.x - pos_clock.x)
								) / Area_hex;

							f64 result_coeff_sqrt = TWOTHIRDS * kappa_par *(
								(omega.dotxy(coeff_grad_wrt_sqrt_T_clock))*(omega.dotxy(edge_normal))
								) / (nu * nu + omega.dot(omega));
							//d_by_dbeta += x_clock*0.5*result_coeff_sqrt / sqrt(T_clock);

							fzArray[iPrev] += our_fac*0.5*result_coeff_sqrt / sqrt(T_clock);
						};

					};
				}; // if iSpecies == 0

			} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

			  // Now go round:	
			endpt_clock = endpt_anti;
			pos_clock = pos_out;
			pos_out = pos_anti;
			T_clock = T_out;
			T_out = T_anti;
			x_clock = x_out;
			x_out = x_anti;
			iPrev = iNeigh;
		}; // next iNeigh

	}; // DOMAIN vertex active in mask

	memcpy(p_array + iVertex*MAXNEIGH_d, fzArray, sizeof(f64)*MAXNEIGH_d);
	p_effectself[iVertex] = effectself;
	//memcpy(NTadditionrates + iVertex, &ourrates, sizeof(NTrates));
}



__global__ void kernelHeat_1species_geometric_coeffself(
	structural * __restrict__ p_info_minor,
	long * __restrict__ pIndexNeigh,
	char * __restrict__ pPBCNeigh,
	long * __restrict__ izTri_verts,
	char * __restrict__ szPBCtri_verts,
	f64_vec2 * __restrict__ p_cc,

	nvals * __restrict__ p_n_major,
	f64 * __restrict__ p__T,
	f64 const h_use,
	f64_vec3 * __restrict__ p_B_major,

	f64 * __restrict__ p__kappa_major,
	f64 * __restrict__ p__nu_major,

	f64 * __restrict__ p_AreaMajor,
	// scrap masking for now --- but bring it back intelligently???

	bool * __restrict__ p_maskbool,
	bool * __restrict__ p_maskblock,
	bool bUseMask,

	// Just hope that our clever version will converge fast.

	int iSpecies,

	f64 * __restrict__ p_effectself // hmmm
)
{
	__shared__ f64_vec2 shared_pos_verts[threadsPerTileMajorClever]; // 2
																	 // DO NOT WANT:
	__shared__ f64_vec2 shared_pos[2 * threadsPerTileMajorClever]; // but as far as we know, we are having to use circumcenters.
																   // Maybe it works without them now that we have the longitudinal assumptions --- don't know for sure.
																   // But it means we are not being consistent with our definition of a cell?
																   // Like having major cells Voronoi => velocity living on centroids (which it must, for visc + A) is in slightly the wrong place.

	__shared__ f64 shared_T[threadsPerTileMajorClever];

	__shared__ f64_vec2 shared_B[threadsPerTileMajorClever]; // +2

															 // B is smooth. Unfortunately we have not fitted in Bz here.
															 // In order to do that perhaps rewrite so that variables are overwritten in shared.
															 // We do not need all T and nu in shared at the same time.
															 // This way is easier for NOW.
	__shared__ f64 shared_kappa[threadsPerTileMajorClever];
	__shared__ f64 shared_nu[threadsPerTileMajorClever];

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
	bool bMask;

	if (bUseMask)
		if (p_maskblock[blockIdx.x] == 0) return;
	// skip out on mask 
	if (bUseMask) bMask = p_maskbool[iVertex];

	f64 effectself;

	structural info = p_info_minor[iVertex + BEGINNING_OF_CENTRAL];     // 3 double
	shared_pos_verts[threadIdx.x] = info.pos;
#ifdef CENTROID_HEATCONDUCTION
	{
		structural infotemp[2];
		memcpy(infotemp, p_info_minor + 2 * iVertex, 2 * sizeof(structural));
		shared_pos[threadIdx.x * 2] = infotemp[0].pos;
		shared_pos[threadIdx.x * 2 + 1] = infotemp[1].pos;
		// No nu to set for neutrals - not used
	}
#else
	{
		memcpy(&(shared_pos[threadIdx.x * 2]), p_cc + 2 * iVertex, 2 * sizeof(f64_vec2));
	}
#endif

	memcpy(&(shared_nu[threadIdx.x]), p__nu_major + iVertex, sizeof(f64));
	memcpy(&(shared_kappa[threadIdx.x]), p__kappa_major + iVertex, sizeof(f64));

	if ((info.flag == DOMAIN_VERTEX) || (info.flag == OUTERMOST)) {
		shared_B[threadIdx.x] = p_B_major[iVertex].xypart();
		shared_T[threadIdx.x] = p__T[iVertex];
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
	f64 our_x, x_clock, x_out, x_anti;
	f64_vec2 pos_clock, pos_anti, pos_out;   // +6
	f64_vec2 B_out;       // +2
						  //NTrates ourrates;      // +5
						  //f64 kappa_parallel; // do we use them all at once or can we save 2 doubles here?
						  //f64 nu;                // 20 there  
	f64_vec2 edge_normal;  // 22
	f64_vec2 endpt_anti;    // 24 .. + 6 from above
	long indexneigh;     // into the 2-double buffer in L1
	f64_vec2 endpt_clock;    // As we only use endpt_anti afterwords we could union endpt_clock with edge_normal
							 // Come back and optimize by checking which things we need in scope at the same time?
	f64 kappa_out, nu_out;

	short iNeigh; // only fixed # of addresses so short makes no difference.
	char PBC; // char makes no difference.
	f64 d_by_dbeta = 0.0;

	if ((info.flag == DOMAIN_VERTEX) && ((bUseMask == 0) || (bMask == true)))
	{
		
		f64 N;
		if (iSpecies == 0) {
			N = p_AreaMajor[iVertex] * p_n_major[iVertex].n_n;
		}
		else {
			N = p_AreaMajor[iVertex] * p_n_major[iVertex].n;
		}
		f64 our_fac = - h_use / N; // factor for change in epsilon^2											 
		effectself = 1.0; // change in own epsilon by changing T is +1.0 for eps = T_k+1-T_k-hF

		if (TESTHEAT) printf("%d effectself %1.10E our_fac %1.10E \n", iVertex, effectself, our_fac);

		// Need this, we are adding on to existing d/dt N,NT :
		//	memcpy(&ourrates, NTadditionrates + iVertex, sizeof(NTrates));

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

		// The idea of not sending blocks full of non-domain vertices is another idea. Fiddly with indices.
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
			T_clock = p__T[indexneigh];
		};
		PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + info.neigh_len - 1];
		if (PBC == NEEDS_ANTI) {
			pos_clock = Anticlock_rotate2(pos_clock);
		};
		if (PBC == NEEDS_CLOCK) {
			pos_clock = Clockwise_rotate2(pos_clock);
		};
		//x_clock = p__x[indexneigh];

		indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + 0];
		if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
		{
			pos_out = shared_pos_verts[indexneigh - StartMajor];
			T_out = shared_T[indexneigh - StartMajor];
			kappa_out = shared_kappa[indexneigh - StartMajor];
			nu_out = shared_nu[indexneigh - StartMajor];
		}
		else {
			structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
			pos_out = info2.pos;
			T_out = p__T[indexneigh]; // saved nothing here, only in loading
			kappa_out = p__kappa_major[indexneigh];
			nu_out = p__nu_major[indexneigh];
		};
		PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + 0];
		if (PBC == NEEDS_ANTI) {
			pos_out = Anticlock_rotate2(pos_out);
		};
		if (PBC == NEEDS_CLOCK) {
			pos_out = Clockwise_rotate2(pos_out);
		};
		//x_out = p__x[indexneigh];

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

		//		if (T_clock == 0.0) {
		//			T_clock = 0.5*(shared_T[threadIdx.x] + T_out);
		//		};
		//		Mimic
		short iPrev = info.neigh_len - 1;

#pragma unroll MAXNEIGH_d
		for (iNeigh = 0; iNeigh < info.neigh_len; iNeigh++)
		{

			short iNext = iNeigh + 1; if (iNext == info.neigh_len) iNext = 0;
			indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNext];
			PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNext];

			if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
			{
				pos_anti = shared_pos_verts[indexneigh - StartMajor];
				T_anti = shared_T[indexneigh - StartMajor];
			}
			else {
				structural info2 = p_info_minor[indexneigh + BEGINNING_OF_CENTRAL];
				pos_anti = info2.pos;
				T_anti = p__T[indexneigh];
			};
			if (PBC == NEEDS_ANTI) {
				pos_anti = Anticlock_rotate2(pos_anti);
			};
			if (PBC == NEEDS_CLOCK) {
				pos_anti = Clockwise_rotate2(pos_anti);
			};
			//x_anti = p__x[indexneigh];

			//if (T_anti == 0.0) {
			//	T_anti = 0.5*(shared_T[threadIdx.x] + T_out);
			//}; // So we are receiving 0 then doing this. But how come?
			//Mimic
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

			// SMARTY:
			if (TestDomainPos(pos_out))
			{
				if (iSpecies == 0) {
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						kappa_out = shared_kappa[indexneigh - StartMajor];
					}
					else {
						kappa_out = p__kappa_major[indexneigh];
					};
					f64 kappa_par = sqrt(kappa_out * shared_kappa[threadIdx.x]);
					//if ((!TestDomainPos(pos_clock) ) ||	(!TestDomainPos(pos_anti)))
					{
						f64 edgelen = edge_normal.modulus();
						//ourrates.NnTn += TWOTHIRDS * kappa_par * edgelen *
						//	(T_out - shared_T[threadIdx.x]) / (pos_out - info.pos).modulus();

						//						d_by_dbeta += our_x*TWOTHIRDS * kappa_par * edgelen *
						//						(-1.0) / (pos_out - info.pos).modulus();
						//				d_by_dbeta += x_out*TWOTHIRDS*kappa_par * edgelen *
						//				(1.0) / (pos_out - info.pos).modulus();

						f64 temp = TWOTHIRDS*kappa_par * edgelen *
							(1.0) / (pos_out - info.pos).modulus();

						effectself -= temp*our_fac;
						
						// .. for neutral heat cond, it turns out we basically never even use T_anti, T_clock.

#ifdef CENTROID_HEATCONDUCTION
						printf("you know this doesn't match? Need circumcenters for longitudinal heat flux.\n");
#endif
					}
				}
				else {
					indexneigh = Indexneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if ((indexneigh >= StartMajor) && (indexneigh < EndMajor))
					{
						B_out = shared_B[indexneigh - StartMajor];
						kappa_out = shared_kappa[indexneigh - StartMajor];
						nu_out = shared_nu[indexneigh - StartMajor];
					}
					else {
						f64_vec3 B_out3 = p_B_major[indexneigh];
						B_out = B_out3.xypart();
						kappa_out = p__kappa_major[indexneigh];
						nu_out = p__nu_major[indexneigh];
					};
					PBC = PBCneigh[MAXNEIGH_d*threadIdx.x + iNeigh];
					if (PBC == NEEDS_ANTI) 	B_out = Anticlock_rotate2(B_out);
					if (PBC == NEEDS_CLOCK)	B_out = Clockwise_rotate2(B_out);

					{ // scoping brace
						f64_vec3 omega;
						if (iSpecies == 1) {
							omega = Make3(qoverMc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qoverMc);
						}
						else {
							omega = Make3(qovermc * 0.5*(shared_B[threadIdx.x] + B_out), BZ_CONSTANT*qovermc);
						};

						// GEOMETRIC KAPPA:
						f64 kappa_par = sqrt(kappa_out * shared_kappa[threadIdx.x]);
						f64 nu = sqrt(nu_out * shared_nu[threadIdx.x]);

						f64 edgelen = edge_normal.modulus();
						f64 delta_out = sqrt((info.pos.x - pos_out.x)*(info.pos.x - pos_out.x) + (info.pos.y - pos_out.y)*(info.pos.y - pos_out.y));

						// Now it's the nitty-gritty.
						//	f64 sqrt_Tout_Tanti, sqrt_Tout_Tclock, sqrt_Tours_Tanti, sqrt_Tours_Tclock;

						f64_vec2 pos_ours = shared_pos_verts[threadIdx.x];

						// Simplify:
						f64 Area_hex = 0.25*(
							(1.5*pos_out.x + 0.5*pos_anti.x)*(pos_anti.y - pos_out.y)
							+ (0.5*(pos_out.x + pos_anti.x) + 0.5*(pos_ours.x + pos_anti.x))*(pos_ours.y - pos_out.y)
							+ (0.5*(pos_ours.x + pos_anti.x) + pos_ours.x)*(pos_ours.y - pos_anti.y)
							+ (0.5*(pos_ours.x + pos_clock.x) + pos_ours.x)*(pos_clock.y - pos_ours.y)
							+ (0.5*(pos_ours.x + pos_clock.x) + 0.5*(pos_out.x + pos_clock.x))*(pos_out.y - pos_ours.y)
							+ (0.5*(pos_out.x + pos_clock.x) + pos_out.x)*(pos_out.y - pos_clock.y)
							);

						//grad_T.x = 0.25*(
						//	(T_out + sqrt_Tout_Tanti)*(pos_anti.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + sqrt_Tout_Tanti - sqrt_Tours_Tclock - sqrt_Tout_Tclock)*(pos_ours.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + shared_T[threadIdx.x])*(pos_ours.y - pos_anti.y)
						//	+ (sqrt_Tours_Tclock + shared_T[threadIdx.x])*(pos_clock.y - pos_ours.y)
						//	+ (sqrt_Tout_Tclock + T_out)*(pos_out.y - pos_clock.y)
						//	);
						// could simplify further to just take coeff on each T value.
						f64_vec2 coeffself_grad_T, coeffsqrt_grad_T;

						coeffself_grad_T.x = 0.25*(pos_clock.y - pos_anti.y) / Area_hex;
						coeffself_grad_T.y = 0.25*(pos_anti.x - pos_clock.x) / Area_hex;

						f64 sqrt_Tanti, sqrt_Tclock, sqrt_Tours, sqrt_Tout;
						if (T_anti > 0.0) {
							sqrt_Tanti = sqrt(T_anti);
						}
						else {
							sqrt_Tanti = 0.0;
						};
						if (T_clock > 0.0) {
							sqrt_Tclock = sqrt(T_clock);
						}
						else {
							sqrt_Tclock = 0.0;
						};
						if (shared_T[threadIdx.x] > 0.0) {
							sqrt_Tours = sqrt(shared_T[threadIdx.x]);
						}
						else {
							sqrt_Tours = 0.0;
						};
						if (T_out > 0.0) {
							sqrt_Tout = sqrt(T_out);
						}
						else {
							sqrt_Tout = 0.0;
						};

						coeffsqrt_grad_T.x = 0.25*(
							(sqrt_Tanti - sqrt_Tclock)*(pos_ours.y - pos_out.y)
							+ sqrt_Tanti*(pos_ours.y - pos_anti.y)
							+ sqrt_Tclock*(pos_clock.y - pos_ours.y)
							) / Area_hex;

						coeffsqrt_grad_T.y = -0.25*(
							(sqrt_Tanti - sqrt_Tclock)*(pos_ours.x - pos_out.x)
							+ sqrt_Tanti*(pos_ours.x - pos_anti.x)
							+ sqrt_Tclock*(pos_clock.x - pos_ours.x)
							) / Area_hex;

						// Isotropic part:
						f64 result_coeff_self = TWOTHIRDS * kappa_par *(
							nu*nu *  //(T_out - shared_T[threadIdx.x]) * (edgelen / delta_out)
							(-1.0)*(edgelen / delta_out)
							+ (omega.dotxy(coeffself_grad_T))*(omega.dotxy(edge_normal))
							) / (nu * nu + omega.dot(omega))
							;

						// d/dself:
						f64 result = result_coeff_self;

						if (shared_T[threadIdx.x] > 0.0) {

							f64 result_coeff_sqrt = TWOTHIRDS * kappa_par *(
								(omega.dotxy(coeffsqrt_grad_T))*(omega.dotxy(edge_normal))
								) / (nu * nu + omega.dot(omega));
							result += 0.5*result_coeff_sqrt / sqrt(shared_T[threadIdx.x]);
						}; // else sqrt term didn't have an effect.

						   //d_by_dbeta += our_x*result;

						effectself += result*our_fac;

						//grad_T.x = 0.25*(
						//	(T_out + sqrt_Tout_Tanti)*(pos_anti.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + sqrt_Tout_Tanti - sqrt_Tours_Tclock - sqrt_Tout_Tclock)*(pos_ours.y - pos_out.y)
						//	+ (sqrt_Tours_Tanti + shared_T[threadIdx.x])*(pos_ours.y - pos_anti.y)
						//	+ (sqrt_Tours_Tclock + shared_T[threadIdx.x])*(pos_clock.y - pos_ours.y)
						//	+ (sqrt_Tout_Tclock + T_out)*(pos_out.y - pos_clock.y)
						//	);

						// coeff on power 1:
						f64_vec2 ROC_grad_wrt_T_out;
						ROC_grad_wrt_T_out.x = 0.25*(pos_anti.y - pos_clock.y) / Area_hex;
						ROC_grad_wrt_T_out.y = 0.25*(pos_clock.x - pos_anti.x) / Area_hex;

						// stick to format from above :

						// Isotropic part:
						//d_by_dbeta += x_out* TWOTHIRDS * kappa_par *(
						//	nu*nu *  //(T_out - shared_T[threadIdx.x]) * (edgelen / delta_out)
						//	(1.0)*(edgelen / delta_out)
						//	+ (omega.dotxy(ROC_grad_wrt_T_out))*(omega.dotxy(edge_normal))
						//	) / (nu * nu + omega.dot(omega))
						//	;
						
						if (T_out > 0.0) {
							f64_vec2 coeff_grad_wrt_sqrt_T_out;
							coeff_grad_wrt_sqrt_T_out.x = 0.25*(
								(sqrt_Tanti)*(pos_anti.y - pos_out.y)
								+ (sqrt_Tanti - sqrt_Tclock)*(pos_ours.y - pos_out.y)
								+ (sqrt_Tclock)*(pos_out.y - pos_clock.y)
								) / Area_hex;

							coeff_grad_wrt_sqrt_T_out.y = -0.25*(
								(sqrt_Tanti)*(pos_anti.x - pos_out.x)
								+ (sqrt_Tanti - sqrt_Tclock)*(pos_ours.x - pos_out.x)
								+ (sqrt_Tclock)*(pos_out.x - pos_clock.x)
								) / Area_hex;

							f64 result_coeff_sqrt = TWOTHIRDS * kappa_par *(
								(omega.dotxy(coeff_grad_wrt_sqrt_T_out))*(omega.dotxy(edge_normal))
								) / (nu * nu + omega.dot(omega));
							//d_by_dbeta += x_out*0.5*result_coeff_sqrt / sqrt(T_out);

						};

						// T_anti:
						if (T_anti > 0.0) {
							f64_vec2 coeff_grad_wrt_sqrt_T_anti;
							coeff_grad_wrt_sqrt_T_anti.x = 0.25*(
								(sqrt_Tout)*(pos_anti.y - pos_out.y)
								+ (sqrt_Tours + sqrt_Tout)*(pos_ours.y - pos_out.y)
								+ sqrt_Tours*(pos_ours.y - pos_anti.y)
								) / Area_hex;

							coeff_grad_wrt_sqrt_T_anti.y = -0.25*(
								(sqrt_Tout)*(pos_anti.x - pos_out.x)
								+ (sqrt_Tours + sqrt_Tout)*(pos_ours.x - pos_out.x)
								+ sqrt_Tours*(pos_ours.x - pos_anti.x)
								) / Area_hex;

							f64 result_coeff_sqrt = TWOTHIRDS * kappa_par *(
								(omega.dotxy(coeff_grad_wrt_sqrt_T_anti))*(omega.dotxy(edge_normal))
								) / (nu * nu + omega.dot(omega));

							//d_by_dbeta += x_anti*0.5*result_coeff_sqrt / sqrt(T_anti);
					};

						if (T_clock > 0.0) {

							//grad_T.x = 0.25*(
							//	(T_out + sqrt_Tout_Tanti)*(pos_anti.y - pos_out.y)
							//	+ (sqrt_Tours_Tanti + sqrt_Tout_Tanti - sqrt_Tours_Tclock - sqrt_Tout_Tclock)*(pos_ours.y - pos_out.y)
							//	+ (sqrt_Tours_Tanti + shared_T[threadIdx.x])*(pos_ours.y - pos_anti.y)
							//	+ (sqrt_Tours_Tclock + shared_T[threadIdx.x])*(pos_clock.y - pos_ours.y)
							//	+ (sqrt_Tout_Tclock + T_out)*(pos_out.y - pos_clock.y)
							//	);
							f64_vec2 coeff_grad_wrt_sqrt_T_clock;
							coeff_grad_wrt_sqrt_T_clock.x = 0.25*(
								-(sqrt_Tours + sqrt_Tout)*(pos_ours.y - pos_out.y)
								+ sqrt_Tours*(pos_clock.y - pos_ours.y)
								+ sqrt_Tout*(pos_out.y - pos_clock.y)
								) / Area_hex;
							coeff_grad_wrt_sqrt_T_clock.y = -0.25*(
								-(sqrt_Tours + sqrt_Tout)*(pos_ours.x - pos_out.x)
								+ sqrt_Tours*(pos_clock.x - pos_ours.x)
								+ sqrt_Tout*(pos_out.x - pos_clock.x)
								) / Area_hex;

							f64 result_coeff_sqrt = TWOTHIRDS * kappa_par *(
								(omega.dotxy(coeff_grad_wrt_sqrt_T_clock))*(omega.dotxy(edge_normal))
								) / (nu * nu + omega.dot(omega));
							//d_by_dbeta += x_clock*0.5*result_coeff_sqrt / sqrt(T_clock);

						};

					};
				}; // if iSpecies == 0

			} // if (pos_out.x*pos_out.x + pos_out.y*pos_out.y > ...)

			  // Now go round:	
			endpt_clock = endpt_anti;
			pos_clock = pos_out;
			pos_out = pos_anti;
			T_clock = T_out;
			T_out = T_anti;
			x_clock = x_out;
			x_out = x_anti;
			iPrev = iNeigh;
		}; // next iNeigh

	}; // DOMAIN vertex active in mask

	p_effectself[iVertex] = effectself;
	//memcpy(NTadditionrates + iVertex, &ourrates, sizeof(NTrates));
}

__global__ void AddFromMyNeighbours(
	structural * __restrict__ p_info_major,
	f64 * __restrict__ p_array,
	f64 * __restrict__ p_arrayself,
	f64 * __restrict__ p_sum,
	long * __restrict__ p_izNeigh_vert,
	short * __restrict__ p_who_am_I_to_you
) {
	//This requires who_am_I to always be well updated.;
	__shared__ f64 pArray[threadsPerTileMajor*MAXNEIGH_d];
	// We can actually fit 24 doubles/thread at 256 threads, 48K - so this is actually running 2 tiles at once.
	__shared__ short who_am_I[threadsPerTileMajor*MAXNEIGH_d];
	// think about memory balance => this got shared

	long const StartMajor = blockIdx.x*blockDim.x;
	long const EndMajor = StartMajor + blockDim.x;
	long const iVertex = blockDim.x*blockIdx.x + threadIdx.x;

	memcpy(pArray + threadIdx.x*MAXNEIGH_d, p_array + iVertex*MAXNEIGH_d, sizeof(f64)*MAXNEIGH_d);

	__syncthreads();

	long indexneigh[MAXNEIGH_d];

	memcpy(indexneigh, p_izNeigh_vert + MAXNEIGH*iVertex, sizeof(long)*MAXNEIGH);
	memcpy(who_am_I + MAXNEIGH_d*threadIdx.x, p_who_am_I_to_you + MAXNEIGH*iVertex, sizeof(short)*MAXNEIGH);

	structural info = p_info_major[iVertex];
	f64 sum = 0.0;
	if (info.flag == DOMAIN_VERTEX) {
		sum = p_arrayself[iVertex]; // 2.0*epsilon _self * deps/dself
		
	//	if ((iVertex == VERTCHOSEN2))
	//		printf("%d sum %1.9E \n", iVertex, sum);

		for (int i = 0; i < info.neigh_len; i++)
		{
			short iWhich = who_am_I[threadIdx.x*MAXNEIGH_d + i];
			long iNeigh = indexneigh[i];
			if ((iNeigh >= StartMajor) && (iNeigh < EndMajor)) {

				sum += pArray[(iNeigh - StartMajor)*MAXNEIGH_d + iWhich];
			//	if ((iVertex == VERTCHOSEN2))
			//		printf("iVertex %d i %d iNeigh %d iWhich %d p_Array[ ] %1.14E sum %1.9E \n", iVertex, i, iNeigh, iWhich,
			//			pArray[(iNeigh - StartMajor)*MAXNEIGH_d + iWhich], sum);

			} else {

				sum += p_array[iNeigh*MAXNEIGH_d + iWhich];
			//	if ((iVertex == VERTCHOSEN2))
			//		printf("iVertex %d i %d iNeigh %d iWhich %d p_array[] %1.14E sum %1.9E \n", iVertex, i, iNeigh, iWhich,
			//			p_array[iNeigh*MAXNEIGH_d + iWhich] , sum);
			};

		};

	}
	p_sum[iVertex] = -sum; // added up eps_j deps_j/dx_i
	// put in minus for steepest descent instead of ascent.

}



