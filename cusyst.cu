
#include "cuda_struct.h"

__host__ bool Call(cudaError_t cudaStatus, char str[])
{
	if (cudaStatus == cudaSuccess) return false;
	printf("Error: %s\nReturned %d : %s\n",
		str, cudaStatus, cudaGetErrorString(cudaStatus));
	printf("Anykey.\n");	getch();
	return true;
}


cuSyst::cuSyst(){
	bInvoked = false;
	bInvokedHost = false;
}

int cuSyst::Invoke(long Numvertices)
{
	 Nverts = Numvertices;
	 Ntris = 2 * Nverts;
	 Nminor = Nverts + Ntris;

	if (bInvoked == false) {

		if (
			   (!CallMAC(cudaMalloc((void**)&p_info, Nminor * sizeof(structural))))

			&& (!CallMAC(cudaMalloc((void**)&p_izTri_vert, Nverts*MAXNEIGH_d * sizeof(long))))
			&& (!CallMAC(cudaMalloc((void**)&p_izNeigh_vert, Nverts*MAXNEIGH_d * sizeof(long))))
			&& (!CallMAC(cudaMalloc((void**)&p_szPBC_vert, Nverts*MAXNEIGH_d * sizeof(char))))
			
			&& (!CallMAC(cudaMalloc((void**)&p_Indexneigh_triminor, Ntris*6 * sizeof(long))))
			&& (!CallMAC(cudaMalloc((void**)&p_szPBC_triminor, Ntris * 6 * sizeof(char))))
			&& (!CallMAC(cudaMalloc((void**)&p_tri_corner_index, Ntris * sizeof(LONG3))))
			&& (!CallMAC(cudaMalloc((void**)&p_who_am_I_to_corner, Ntris * sizeof(LONG3))))
						
			&& (!CallMAC(cudaMalloc((void**)&p_n_major, Nverts * sizeof(nvals))))
			&& (!CallMAC(cudaMalloc((void**)&p_n_minor, Nminor * sizeof(nvals))))
			&& (!CallMAC(cudaMalloc((void**)&p_T_minor, Nminor * sizeof(T3))))

			&& (!CallMAC(cudaMalloc((void**)&p_AAdot, Nminor * sizeof(AAdot))))
			
			&& (!CallMAC(cudaMalloc((void**)&p_v_n, Nminor * sizeof(f64_vec3))))
			&& (!CallMAC(cudaMalloc((void**)&p_vie, Nminor * sizeof(v4))))
			&& (!CallMAC(cudaMalloc((void**)&p_B, Nminor * sizeof(f64_vec3))))

			&& (!CallMAC(cudaMalloc((void**)&p_Lap_Az, Nminor * sizeof(f64))))
			&& (!CallMAC(cudaMalloc((void**)&p_overall_v, Nminor * sizeof(f64_vec2))))
			
			&& (!CallMAC(cudaMalloc((void**)&p_MomAdditionRate_ion, Nminor * sizeof(f64_vec3))))
			&& (!CallMAC(cudaMalloc((void**)&p_MomAdditionRate_elec, Nminor * sizeof(f64_vec3))))
			&& (!CallMAC(cudaMalloc((void**)&p_MomAdditionRate_neut, Nminor * sizeof(f64_vec3))))
			
			&& (!CallMAC(cudaMalloc((void**)&p_AreaMinor, Nminor * sizeof(f64))))
			&& (!CallMAC(cudaMalloc((void**)&p_AreaMajor, Nverts * sizeof(f64))))
			)
		{
			bInvoked = true;
			//Zero();
			printf("Dimensioned for MAXNEIGH_d = %d\n", MAXNEIGH_d);
			return 0;
		}
		else {
			printf("There was an error in dimensioning Systdata object.\n");
			getch();	getch();
			return 1;
		};
	}
	else {
		if (Nverts != N) { printf("Systdata Error - Nverts %d != N %d\n", Nverts, N); getch(); }
		return 2;
	};
}
void cuSyst::InvokeHost(long Numvertices)
{
	Nverts = Numvertices;
	Ntris = 2 * Nverts;
	Nminor = Nverts + Ntris;
	    p_info = ( structural * )malloc(Nminor* sizeof(structural));
		
		p_izTri_vert = ( long *)malloc(Nverts*MAXNEIGH_d * sizeof(long));
		p_izNeigh_vert = (long * )malloc(Nverts*MAXNEIGH_d * sizeof(long));
		p_szPBC_vert = (char * )malloc(Nverts*MAXNEIGH_d * sizeof(char));

		p_Indexneigh_triminor = (long * )malloc(Ntris * 6 * sizeof(long));
		p_szPBC_triminor = (char * )malloc(Ntris * 6 * sizeof(char));
		p_tri_corner_index = ( LONG3 *)malloc(Ntris * sizeof(LONG3));
		p_who_am_I_to_corner = (LONG3 * )malloc(Ntris * sizeof(LONG3));

		p_n_major = (nvals * )malloc(Nverts * sizeof(nvals));
		p_n_minor = (nvals * )malloc(Nminor * sizeof(nvals));
		p_T_minor = (T3 * )malloc(Nminor * sizeof(T3));

		p_AAdot = ( AAdot *)malloc(Nminor * sizeof(AAdot));

		p_v_n = ( f64_vec3 *)malloc(Nminor * sizeof(f64_vec3));
		p_vie = (v4 * )malloc(Nminor * sizeof(v4));
		p_B = ( f64_vec3 *)malloc(Nminor * sizeof(f64_vec3));

		p_Lap_Az = (f64 * )malloc(Nminor * sizeof(f64));
		p_overall_v = (f64_vec2 *)malloc(Nminor * sizeof(f64_vec2));

		p_MomAdditionRate_ion = (f64_vec3 * )malloc(Nminor * sizeof(f64_vec3));
		p_MomAdditionRate_elec = (f64_vec3 *)malloc(Nminor * sizeof(f64_vec3));
		p_MomAdditionRate_neut = (f64_vec3 *)malloc(Nminor * sizeof(f64_vec3));

		p_AreaMinor = (f64 * )malloc(Nminor * sizeof(f64));
		p_AreaMajor = (f64 * )malloc(Nverts * sizeof(f64));

		if (p_Areamajor == 0) {
			printf("failed to invokeHost the cusyst.\n")
			getch();
		}
		else {
			bInvokedHost = true;
		};
}
cuSyst::~cuSyst{
	if (bInvoked)
	{

		cudaFree(p_info);
		cudaFree(p_izTri_vert);
		cudaFree(p_izNeigh_vert);
		cudaFree(p_szPBC_vert);
		cudaFree(p_Indexneigh_triminor);
		cudaFree(p_szPBC_triminor);
		cudaFree(p_tri_corner_index);
		cudaFree(p_who_am_I_to_corner);
		cudaFree(p_n_major);
		cudaFree(p_n_minor);
		cudaFree(p_T_minor);
		cudaFree(p_AAdot);
		cudaFree(p_v_n);
		cudaFree(p_vie);
		cudaFree(p_B);
		cudaFree(p_Lap_Az);
		cudaFree(p_overall_v);
		cudaFree(p_MomAdditionRate_ion);
		cudaFree(p_MomAdditionRate_elec);
		cudaFree(p_MomAdditionRate_neut);
		cudaFree(p_AreaMinor);
		cudaFree(p_AreaMajor);

	}
	if (bInvokedHost) {

free(p_info);
free(p_izTri_vert);
free(p_izNeigh_vert);
free(p_szPBC_vert);
free(p_Indexneigh_triminor);
free(p_szPBC_triminor);
free(p_tri_corner_index);
free(p_who_am_I_to_corner);
free(p_n_major);
free(p_n_minor);
free(p_T_minor);
free(p_AAdot);
free(p_v_n);
free(p_vie);
free(p_B);
free(p_Lap_Az);
free(p_overall_v);
free(p_MomAdditionRate_ion);
free(p_MomAdditionRate_elec);
free(p_MomAdditionRate_neut);
free(p_AreaMinor);
free(p_AreaMajor);

	};
}

void cuSyst::SendToHost(cuSyst & Xhost)
{
	// We are going to need a host-allocated cuSyst in order to
	// do the populating basically.
	(!CallMAC(cudaMemcpy(Xhost.p_info, p_info, Nminor * sizeof(structural), cudaMemcpyDeviceToHost)))

		&& (!CallMAC(cudaMemcpy(Xhost.p_izTri_vert, p_izTri_vert, Nverts*MAXNEIGH_d * sizeof(long), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_izNeigh_vert, p_izNeigh_vert, Nverts*MAXNEIGH_d * sizeof(long), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_szPBC_vert, p_szPBC_vert, Nverts*MAXNEIGH_d * sizeof(char), cudaMemcpyDeviceToHost)))

		&& (!CallMAC(cudaMemcpy(Xhost.p_Indexneigh_triminor, p_Indexneigh_triminor, Ntris * 6 * sizeof(long), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_szPBC_triminor, p_szPBC_triminor, Ntris * 6 * sizeof(char), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_tri_corner_index, p_tri_corner_index, Ntris * sizeof(LONG3), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_who_am_I_to_corner, p_who_am_I_to_corner, Ntris * sizeof(LONG3), cudaMemcpyDeviceToHost)))

		&& (!CallMAC(cudaMemcpy(Xhost.p_n_major, p_n_major, Nverts * sizeof(nvals), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_n_minor, p_n_minor, Nminor * sizeof(nvals), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_T_minor, p_T_minor, Nminor * sizeof(T3), cudaMemcpyDeviceToHost)))

		&& (!CallMAC(cudaMemcpy(Xhost.p_AAdot, p_AAdot, Nminor * sizeof(AAdot), cudaMemcpyDeviceToHost)))

		&& (!CallMAC(cudaMemcpy(Xhost.p_v_n, p_v_n, Nminor * sizeof(f64_vec3), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_vie, p_vie, Nminor * sizeof(v4), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_B, p_B, Nminor * sizeof(f64_vec3), cudaMemcpyDeviceToHost)))

		&& (!CallMAC(cudaMemcpy(Xhost.p_Lap_Az, p_Lap_Az, Nminor * sizeof(f64), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_overall_v, p_overall_v, Nminor * sizeof(f64_vec2), cudaMemcpyDeviceToHost)))

		&& (!CallMAC(cudaMemcpy(Xhost.p_MomAdditionRate_ion, p_MomAdditionRate_ion, Nminor * sizeof(f64_vec3), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_MomAdditionRate_elec, p_MomAdditionRate_elec, Nminor * sizeof(f64_vec3), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_MomAdditionRate_neut, p_MomAdditionRate_neut, Nminor * sizeof(f64_vec3), cudaMemcpyDeviceToHost)))

		&& (!CallMAC(cudaMemcpy(Xhost.p_AreaMinor, p_AreaMinor, Nminor * sizeof(f64), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_AreaMajor, p_AreaMajor, Nverts * sizeof(f64), cudaMemcpyDeviceToHost)))
		Call(cudaThreadSynchronize(), "cudaThreadSynchronize cuSyst::SendToHost");

}
void cuSyst::SendToDevice(cuSyst & Xdevice)
{
	// We are going to need a host-allocated cuSyst in order to
	// do the populating basically.
	if (
		!CallMAC(cudaMemcpy(Xdevice.p_info, p_info, Nminor * sizeof(structural), cudaMemcpyHostToDevice)))

		&& (!CallMAC(cudaMemcpy(Xdevice.p_izTri_vert, p_izTri_vert, Nverts*MAXNEIGH_d * sizeof(long), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_izNeigh_vert, p_izNeigh_vert, Nverts*MAXNEIGH_d * sizeof(long), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_szPBC_vert, p_szPBC_vert, Nverts*MAXNEIGH_d * sizeof(char), cudaMemcpyHostToDevice)))

		&& (!CallMAC(cudaMemcpy(Xdevice.p_Indexneigh_triminor, p_Indexneigh_triminor, Ntris * 6 * sizeof(long), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_szPBC_triminor, p_szPBC_triminor, Ntris * 6 * sizeof(char), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_tri_corner_index, p_tri_corner_index, Ntris * sizeof(LONG3), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_who_am_I_to_corner, p_who_am_I_to_corner, Ntris * sizeof(LONG3), cudaMemcpyHostToDevice)))

		&& (!CallMAC(cudaMemcpy(Xdevice.p_n_major, p_n_major, Nverts * sizeof(nvals), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_n_minor, p_n_minor, Nminor * sizeof(nvals), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_T_minor, p_T_minor, Nminor * sizeof(T3), cudaMemcpyHostToDevice)))

		&& (!CallMAC(cudaMemcpy(Xdevice.p_AAdot, p_AAdot, Nminor * sizeof(AAdot), cudaMemcpyHostToDevice)))

		&& (!CallMAC(cudaMemcpy(Xdevice.p_v_n, p_v_n, Nminor * sizeof(f64_vec3), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_vie, p_vie, Nminor * sizeof(v4), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_B, p_B, Nminor * sizeof(f64_vec3), cudaMemcpyHostToDevice)))

		&& (!CallMAC(cudaMemcpy(Xdevice.p_Lap_Az, p_Lap_Az, Nminor * sizeof(f64), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_overall_v, p_overall_v, Nminor * sizeof(f64_vec2), cudaMemcpyHostToDevice)))

		&& (!CallMAC(cudaMemcpy(Xdevice.p_MomAdditionRate_ion, p_MomAdditionRate_ion, Nminor * sizeof(f64_vec3), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_MomAdditionRate_elec, p_MomAdditionRate_elec, Nminor * sizeof(f64_vec3), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_MomAdditionRate_neut, p_MomAdditionRate_neut, Nminor * sizeof(f64_vec3), cudaMemcpyHostToDevice)))

		&& (!CallMAC(cudaMemcpy(Xdevice.p_AreaMinor, p_AreaMinor, Nminor * sizeof(f64), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_AreaMajor, p_AreaMajor, Nverts * sizeof(f64), cudaMemcpyHostToDevice)))
		)
	{

	}
	Call(cudaThreadSynchronize(), "cudaThreadSynchronize cuSyst::SendToHost");
}

void cuSyst::PopulateFromTriMesh(const TriMesh * pX)
{
	// AsSUMES THIS cuSyst has been allocated on the host.
	nVerts = pX->numVertices;
	nTris = pX->numTriangles;
	nMinor = nVerts + nTris;

	plasma_data data;
	long iMinor;
	for (iMinor = 0; iMinor < NMINOR; iMinor++)
	{
		memcpy(&data, pX->pData[iMinor], sizeof(plasma_data));
		p_n_minor[iMinor].n = data.n;
		p_n_minor[iMinor].n_n = data.n_n;
		if (iMinor > BEGINNING_OF_CENTRAL) {
			p_n_major[iMinor - BEGINNING_OF_CENTRAL].n = data.n;
			p_n_major[iMinor - BEGINNING_OF_CENTRAL].n_n = data.n_n;
		}
		p_T_minor[iMinor].Tn = data.Tn;
		p_T_minor[iMinor].Ti = data.Ti;
		p_T_minor[iMinor].Te = data.Te;
		p_AAdot[iMinor].Az = data.Az;
		p_AAdot[iMinor].Azdot = data.Azdot;
		p_v_n[iMinor] = data.v_n;
		p_vie[iMinor].vxy = data.vxy;
		p_vie[iMinor].vez = data.vez;
		p_vie[iMinor].viz = data.viz;
		p_B[iMinor] = pX->p_B[iMinor];
		p_AreaMinor[iMinor] = pX->p_AreaMinor[iMinor];
	}
	
	// AreaMajor??? pVertex->AreaCell?
	Vertex * X = pX->X;
	long izTri[MAXNEIGH],izNeigh[MAXNEIGH];
	long tri_len;
	structural info;
	for (iVertex = 0; iVertex < nVerts; iVertex++)
	{
		tri_len = pVertex->GetTriangleIndexArray(izTri);
		info.neigh_len = tri_len;
		memset(izTri+tri_len, 0, sizeof(long)*(MAXNEIGH-tri_len));
		memcpy(p_izTri_vert + iVertex*MAXNEIGH, izTri, sizeof(long)*MAXNEIGH);
		tri_len = pVertex->GetNeighbourIndexArray(izNeigh);
		memset(izNeigh + tri_len, 0, sizeof(long)*(MAXNEIGH - tri_len));
		memcpy(p_izNeigh_vert + iVertex*MAXNEIGH,izNeigh, sizeof(long)*MAXNEIGH);
		
		// PBC list????
		info.pos = pVertex->pos;
		p_info_minor[iVertex + BEGINNING_OF_CENTRAL] = info;
		++pVertex;
	};
	

	// Triangle structural?
}

void cuSyst::PopulateTriMesh(const TriMesh * pX)
{
	// AsSUMES THIS cuSyst has been allocated on the host.

	

}
                             