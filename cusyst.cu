
#include "cuda_struct.h"
#pragma once
#ifndef CUSYSTCU
#define CUSYSTCU

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

int cuSyst::Invoke()
{
	 Nverts = NUMVERTICES;
	 Ntris = NUMTRIANGLES; // FFxtubes.h
	 Nminor = Nverts + Ntris;

	if (bInvoked == false) {

		if (
			   (!CallMAC(cudaMalloc((void**)&p_info, Nminor * sizeof(structural))))

			&& (!CallMAC(cudaMalloc((void**)&p_izTri_vert, Nverts*MAXNEIGH_d * sizeof(long))))
			&& (!CallMAC(cudaMalloc((void**)&p_izNeigh_vert, Nverts*MAXNEIGH_d * sizeof(long))))
			&& (!CallMAC(cudaMalloc((void**)&p_szPBCtri_vert, Nverts*MAXNEIGH_d * sizeof(char))))
			&& (!CallMAC(cudaMalloc((void**)&p_szPBCneigh_vert, Nverts*MAXNEIGH_d * sizeof(char))))
			
			&& (!CallMAC(cudaMalloc((void**)&p_izNeigh_TriMinor, Ntris*6 * sizeof(long))))
			&& (!CallMAC(cudaMalloc((void**)&p_szPBC_triminor, Ntris * 6 * sizeof(char))))
			&& (!CallMAC(cudaMalloc((void**)&p_tri_corner_index, Ntris * sizeof(LONG3))))
			&& (!CallMAC(cudaMalloc((void**)&p_tri_periodic_corner_flags, Ntris * sizeof(CHAR4))))
			&& (!CallMAC(cudaMalloc((void**)&p_tri_neigh_index, Ntris * sizeof(LONG3))))
			&& (!CallMAC(cudaMalloc((void**)&p_tri_periodic_neigh_flags, Ntris * sizeof(CHAR4))))
			&& (!CallMAC(cudaMalloc((void**)&p_who_am_I_to_corner, Ntris * sizeof(LONG3))))

			&& (!CallMAC(cudaMalloc((void**)&p_n_major, Nverts * sizeof(nvals))))
			&& (!CallMAC(cudaMalloc((void**)&p_n_minor, Nminor * sizeof(nvals))))
			&& (!CallMAC(cudaMalloc((void**)&p_T_minor, Nminor * sizeof(T3))))

			&& (!CallMAC(cudaMalloc((void**)&p_AAdot, Nminor * sizeof(AAdot))))
			
			&& (!CallMAC(cudaMalloc((void**)&p_v_n, Nminor * sizeof(f64_vec3))))
			&& (!CallMAC(cudaMalloc((void**)&p_vie, Nminor * sizeof(v4))))
			&& (!CallMAC(cudaMalloc((void**)&p_B, Nminor * sizeof(f64_vec3))))

			&& (!CallMAC(cudaMalloc((void**)&p_Lap_Az, Nminor * sizeof(f64))))
			&& (!CallMAC(cudaMalloc((void**)&p_v_overall_minor, Nminor * sizeof(f64_vec2))))
			&& (!CallMAC(cudaMalloc((void**)&p_n_upwind_minor, Nminor * sizeof(nvals))))
						
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
		if (Nverts != NUMVERTICES) { printf("cuSyst Error - Nverts %d != N %d\n", Nverts, NUMVERTICES); getch(); }
		return 2;
	};
}
int cuSyst::InvokeHost()
{
	Nverts = NUMVERTICES;
	Ntris = NUMTRIANGLES;
	Nminor = Nverts + Ntris;
	p_info = ( structural * )malloc(Nminor* sizeof(structural));
		
	p_izTri_vert = ( long *)malloc(Nverts*MAXNEIGH_d * sizeof(long));
	p_izNeigh_vert = (long * )malloc(Nverts*MAXNEIGH_d * sizeof(long));
	p_szPBCtri_vert = (char * )malloc(Nverts*MAXNEIGH_d * sizeof(char));
	p_szPBCneigh_vert = (char *)malloc(Nverts*MAXNEIGH_d * sizeof(char));

	p_izNeigh_TriMinor = (long * )malloc(Ntris * 6 * sizeof(long));
	p_szPBC_triminor = (char * )malloc(Ntris * 6 * sizeof(char));
	p_tri_corner_index = ( LONG3 *)malloc(Ntris * sizeof(LONG3));
	p_tri_periodic_corner_flags = (CHAR4 *)malloc(Ntris * sizeof(CHAR4));
	p_tri_neigh_index = (LONG3 *)malloc(Ntris * sizeof(LONG3));
	p_tri_periodic_neigh_flags = (CHAR4 *)malloc(Ntris * sizeof(CHAR4));
	p_who_am_I_to_corner = (LONG3 * )malloc(Ntris * sizeof(LONG3));

	p_n_major = (nvals * )malloc(Nverts * sizeof(nvals));
	p_n_minor = (nvals * )malloc(Nminor * sizeof(nvals));
	p_T_minor = (T3 * )malloc(Nminor * sizeof(T3));

	p_AAdot = ( AAdot *)malloc(Nminor * sizeof(AAdot));

	p_v_n = ( f64_vec3 *)malloc(Nminor * sizeof(f64_vec3));
	p_vie = (v4 * )malloc(Nminor * sizeof(v4));
	p_B = ( f64_vec3 *)malloc(Nminor * sizeof(f64_vec3));

	p_Lap_Az = (f64 * )malloc(Nminor * sizeof(f64));
	p_v_overall_minor = (f64_vec2 *)malloc(Nminor * sizeof(f64_vec2));
	p_n_upwind_minor = (nvals *)malloc(Nminor * sizeof(nvals));

	p_AreaMinor = (f64 * )malloc(Nminor * sizeof(f64));
	p_AreaMajor = (f64 * )malloc(Nverts * sizeof(f64));

	if (p_AreaMajor == 0) {
		printf("failed to invokeHost the cusyst.\n");
		getch();
		return 1;
	}
	else {
		bInvokedHost = true;
		return 0;
	};
}
cuSyst::~cuSyst(){
	if (bInvoked)
	{

		cudaFree(p_info);
		cudaFree(p_izTri_vert);
		cudaFree(p_izNeigh_vert);
		cudaFree(p_szPBCtri_vert);
		cudaFree(p_szPBCneigh_vert);
		cudaFree(p_izNeigh_TriMinor);
		cudaFree(p_szPBC_triminor);
		cudaFree(p_tri_corner_index);
		cudaFree(p_tri_periodic_corner_flags);
		cudaFree(p_tri_neigh_index);
		cudaFree(p_tri_periodic_neigh_flags);
		cudaFree(p_who_am_I_to_corner);
		cudaFree(p_n_major);
		cudaFree(p_n_minor);
		cudaFree(p_n_upwind_minor);
		cudaFree(p_T_minor);
		cudaFree(p_AAdot);
		cudaFree(p_v_n);
		cudaFree(p_vie);
		cudaFree(p_B);
		cudaFree(p_Lap_Az);
		cudaFree(p_v_overall_minor);
		cudaFree(p_AreaMinor);
		cudaFree(p_AreaMajor);

	}
	if (bInvokedHost) {

free(p_info);
free(p_izTri_vert);
free(p_izNeigh_vert);
free(p_szPBCtri_vert);
free(p_szPBCneigh_vert);
free(p_izNeigh_TriMinor);
free(p_szPBC_triminor);
free(p_tri_corner_index);
free(p_tri_periodic_corner_flags);
free(p_tri_neigh_index);
free(p_tri_periodic_neigh_flags);
free(p_who_am_I_to_corner);
free(p_n_major);
free(p_n_minor);
free(p_n_upwind_minor);
free(p_T_minor);
free(p_AAdot);
free(p_v_n);
free(p_vie);
free(p_B);
free(p_Lap_Az);
free(p_v_overall_minor);
free(p_AreaMinor);
free(p_AreaMajor);

	};
}

void cuSyst::SendToHost(cuSyst & Xhost)
{
	// We are going to need a host-allocated cuSyst in order to
	// do the populating basically.
	if ((!CallMAC(cudaMemcpy(Xhost.p_info, p_info, Nminor * sizeof(structural), cudaMemcpyDeviceToHost)))

		&& (!CallMAC(cudaMemcpy(Xhost.p_izTri_vert, p_izTri_vert, Nverts*MAXNEIGH_d * sizeof(long), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_izNeigh_vert, p_izNeigh_vert, Nverts*MAXNEIGH_d * sizeof(long), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_szPBCtri_vert, p_szPBCtri_vert, Nverts*MAXNEIGH_d * sizeof(char), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_szPBCneigh_vert, p_szPBCneigh_vert, Nverts*MAXNEIGH_d * sizeof(char), cudaMemcpyDeviceToHost)))

		&& (!CallMAC(cudaMemcpy(Xhost.p_izNeigh_TriMinor, p_izNeigh_TriMinor, Ntris * 6 * sizeof(long), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_szPBC_triminor, p_szPBC_triminor, Ntris * 6 * sizeof(char), cudaMemcpyDeviceToHost)))
		
		&& (!CallMAC(cudaMemcpy(Xhost.p_tri_corner_index, p_tri_corner_index, Ntris * sizeof(LONG3), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_tri_periodic_corner_flags, p_tri_periodic_corner_flags, Ntris * sizeof(CHAR4), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_tri_neigh_index, p_tri_neigh_index, Ntris * sizeof(LONG3), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_tri_periodic_neigh_flags, p_tri_periodic_neigh_flags, Ntris * sizeof(CHAR4), cudaMemcpyDeviceToHost)))

		&& (!CallMAC(cudaMemcpy(Xhost.p_who_am_I_to_corner, p_who_am_I_to_corner, Ntris * sizeof(LONG3), cudaMemcpyDeviceToHost)))

		&& (!CallMAC(cudaMemcpy(Xhost.p_n_major, p_n_major, Nverts * sizeof(nvals), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_n_minor, p_n_minor, Nminor * sizeof(nvals), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_T_minor, p_T_minor, Nminor * sizeof(T3), cudaMemcpyDeviceToHost)))

		&& (!CallMAC(cudaMemcpy(Xhost.p_AAdot, p_AAdot, Nminor * sizeof(AAdot), cudaMemcpyDeviceToHost)))

		&& (!CallMAC(cudaMemcpy(Xhost.p_v_n, p_v_n, Nminor * sizeof(f64_vec3), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_vie, p_vie, Nminor * sizeof(v4), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_B, p_B, Nminor * sizeof(f64_vec3), cudaMemcpyDeviceToHost)))

		&& (!CallMAC(cudaMemcpy(Xhost.p_Lap_Az, p_Lap_Az, Nminor * sizeof(f64), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_v_overall_minor, p_v_overall_minor, Nminor * sizeof(f64_vec2), cudaMemcpyDeviceToHost)))

		&& (!CallMAC(cudaMemcpy(Xhost.p_AreaMinor, p_AreaMinor, Nminor * sizeof(f64), cudaMemcpyDeviceToHost)))
		&& (!CallMAC(cudaMemcpy(Xhost.p_AreaMajor, p_AreaMajor, Nverts * sizeof(f64), cudaMemcpyDeviceToHost)))
		)
	{
		// success - do nothing
	}
	else {
		printf("cudaMemcpy error");
		getch();
	}
	Call(cudaThreadSynchronize(), "cudaThreadSynchronize cuSyst::SendToHost");

}
void cuSyst::SendToDevice(cuSyst & Xdevice)
{
	// We are going to need a host-allocated cuSyst in order to
	// do the populating basically.
	if (
		   (!CallMAC(cudaMemcpy(Xdevice.p_info, p_info, Nminor * sizeof(structural), cudaMemcpyHostToDevice)))

		&& (!CallMAC(cudaMemcpy(Xdevice.p_izTri_vert, p_izTri_vert, Nverts*MAXNEIGH_d * sizeof(long), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_izNeigh_vert, p_izNeigh_vert, Nverts*MAXNEIGH_d * sizeof(long), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_szPBCtri_vert, p_szPBCtri_vert, Nverts*MAXNEIGH_d * sizeof(char), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_szPBCneigh_vert, p_szPBCneigh_vert, Nverts*MAXNEIGH_d * sizeof(char), cudaMemcpyHostToDevice)))

		&& (!CallMAC(cudaMemcpy(Xdevice.p_izNeigh_TriMinor, p_izNeigh_TriMinor, Ntris * 6 * sizeof(long), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_szPBC_triminor, p_szPBC_triminor, Ntris * 6 * sizeof(char), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_tri_corner_index, p_tri_corner_index, Ntris * sizeof(LONG3), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_tri_periodic_corner_flags, p_tri_periodic_corner_flags, Ntris * sizeof(CHAR4), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_tri_neigh_index, p_tri_neigh_index, Ntris * sizeof(LONG3), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_tri_periodic_neigh_flags, p_tri_periodic_neigh_flags, Ntris * sizeof(CHAR4), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_who_am_I_to_corner, p_who_am_I_to_corner, Ntris * sizeof(LONG3), cudaMemcpyHostToDevice)))

		&& (!CallMAC(cudaMemcpy(Xdevice.p_n_major, p_n_major, Nverts * sizeof(nvals), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_n_minor, p_n_minor, Nminor * sizeof(nvals), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_T_minor, p_T_minor, Nminor * sizeof(T3), cudaMemcpyHostToDevice)))

		&& (!CallMAC(cudaMemcpy(Xdevice.p_AAdot, p_AAdot, Nminor * sizeof(AAdot), cudaMemcpyHostToDevice)))

		&& (!CallMAC(cudaMemcpy(Xdevice.p_v_n, p_v_n, Nminor * sizeof(f64_vec3), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_vie, p_vie, Nminor * sizeof(v4), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_B, p_B, Nminor * sizeof(f64_vec3), cudaMemcpyHostToDevice)))

		&& (!CallMAC(cudaMemcpy(Xdevice.p_Lap_Az, p_Lap_Az, Nminor * sizeof(f64), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_v_overall_minor, p_v_overall_minor, Nminor * sizeof(f64_vec2), cudaMemcpyHostToDevice)))

		&& (!CallMAC(cudaMemcpy(Xdevice.p_AreaMinor, p_AreaMinor, Nminor * sizeof(f64), cudaMemcpyHostToDevice)))
		&& (!CallMAC(cudaMemcpy(Xdevice.p_AreaMajor, p_AreaMajor, Nverts * sizeof(f64), cudaMemcpyHostToDevice)))

		)
	{

	}
	else {
		printf("SendToDevice error"); getch();
	}
	Call(cudaThreadSynchronize(), "cudaThreadSynchronize cuSyst::SendToHost");
}

void cuSyst::PopulateFromTriMesh(TriMesh * pX)
{
	// USES pTri->cent

	pX->EnsureAnticlockwiseTriangleCornerSequences_SetupTriMinorNeighboursLists();
	// Variables on host are called TriMinorNeighLists and TriMinorPBCLists
	memcpy(p_izNeigh_TriMinor, pX->TriMinorNeighLists, Ntris * 6 * sizeof(long)); // pointless that we duplicate it but nvm
	memcpy(p_szPBC_triminor, pX->TriMinorPBCLists, Ntris * 6 * sizeof(char));

	// AsSUMES THIS cuSyst has been allocated on the host.
	if ((Nverts != pX->numVertices) ||
		(Ntris != pX->numTriangles))
	{
		printf("ERROR (nVerts %d != pX->numVertices %d) || (nTris != pX->numTriangles)\n",
			Nverts, pX->numVertices);
		getch();
		return;
	}

	plasma_data data;
	long iMinor;
	for (iMinor = 0; iMinor < NMINOR; iMinor++)
	{
		memcpy(&data, &(pX->pData[iMinor]), sizeof(plasma_data));
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
		p_B[iMinor] = data.B;
		p_AreaMinor[iMinor] = pX->AreaMinorArray[iMinor];
	}
	
	pX->SetupMajorPBCTriArrays();
	// AreaMajor??? pVertex->AreaCell?
	Vertex * pVertex;
	pVertex = pX->X;
	long izTri[MAXNEIGH],izNeigh[MAXNEIGH];
	char szPBCtri[MAXNEIGH], szPBCneigh[MAXNEIGH];
	long tri_len;
	long iVertex;
	short i;
	structural info;
	for (iVertex = 0; iVertex < Nverts; iVertex++)
	{
		tri_len = pVertex->GetTriIndexArray(izTri);
		info.neigh_len = tri_len;
		memset(izTri+tri_len, 0, sizeof(long)*(MAXNEIGH-tri_len));
		memcpy(p_izTri_vert + iVertex*MAXNEIGH, izTri, sizeof(long)*MAXNEIGH);

		tri_len = pVertex->GetNeighIndexArray(izNeigh);
		memset(izNeigh + tri_len, 0, sizeof(long)*(MAXNEIGH - tri_len));
		memcpy(p_izNeigh_vert + iVertex*MAXNEIGH,izNeigh, sizeof(long)*MAXNEIGH);
		
		// PB lists:
		memset(szPBCtri + tri_len, 0, sizeof(char)*(MAXNEIGH - tri_len));
		memcpy(szPBCtri, pX->MajorTriPBC[iVertex], sizeof(char)*tri_len);
		memcpy(p_szPBCtri_vert + iVertex*MAXNEIGH, szPBCtri, sizeof(char)*MAXNEIGH);
		
		memset(szPBCneigh, 0, sizeof(char)*MAXNEIGH);
		for (i = 0; i < tri_len; i++)
		{
			if ((pX->T + izTri[i])->periodic == 0) {
				// do nothing: neighbour must be contiguous
			} else {
				if (((pX->X + izNeigh[i])->pos.x > 0.0) && (pVertex->pos.x < 0.0))
					szPBCneigh[i] = ROTATE_ME_ANTICLOCKWISE;
				if (((pX->X + izNeigh[i])->pos.x < 0.0) && (pVertex->pos.x > 0.0))
					szPBCneigh[i] = ROTATE_ME_CLOCKWISE;
			};
		}
		memcpy(p_szPBCneigh_vert + iVertex*MAXNEIGH, szPBCneigh, sizeof(char)*MAXNEIGH);

		info.pos = pVertex->pos;
		p_info[iVertex + BEGINNING_OF_CENTRAL] = info;
		++pVertex;
	};
	
	long iTri; 
	// Triangle structural?
	Triangle * pTri = pX->T;
	for (iTri = 0; iTri < Ntris; iTri++)
	{
		LONG3 tri_corner_index;
		CHAR4 tri_periodic_corner_flags;
		LONG3 who_am_I_to_corner;
		LONG3 tri_neigh_index;
		CHAR4 tri_periodic_neigh_flags;

		tri_corner_index.i1 = pTri->cornerptr[0] - pX->X;
		tri_corner_index.i2 = pTri->cornerptr[1] - pX->X;
		tri_corner_index.i3 = pTri->cornerptr[2] - pX->X;
		p_tri_corner_index[iTri] = tri_corner_index;
		tri_neigh_index.i1 = pTri->neighbours[0] - pX->T;
		tri_neigh_index.i2 = pTri->neighbours[1] - pX->T;
		tri_neigh_index.i3 = pTri->neighbours[2] - pX->T;
		p_tri_neigh_index[iTri] = tri_neigh_index;

		tri_len = pTri->cornerptr[0]->GetTriIndexArray(izTri);
		for (i = 0; i < tri_len; i++)
		{
			if (izTri[i] == iTri) who_am_I_to_corner.i1 = i;
		}
		tri_len = pTri->cornerptr[1]->GetTriIndexArray(izTri);
		for (i = 0; i < tri_len; i++)
		{
			if (izTri[i] == iTri) who_am_I_to_corner.i2 = i;
		}
		tri_len = pTri->cornerptr[2]->GetTriIndexArray(izTri);
		for (i = 0; i < tri_len; i++)
		{
			if (izTri[i] == iTri) who_am_I_to_corner.i3 = i;
		}
		p_who_am_I_to_corner[iTri] = who_am_I_to_corner;
		
		memset(&tri_periodic_corner_flags, 0, sizeof(CHAR4));
		tri_periodic_corner_flags.flag = pTri->u8domain_flag;
		if (pTri->periodic != 0) {
			if (pTri->cornerptr[0]->pos.x > 0.0) tri_periodic_corner_flags.per0 = ROTATE_ME_ANTICLOCKWISE;
			if (pTri->cornerptr[1]->pos.x > 0.0) tri_periodic_corner_flags.per1 = ROTATE_ME_ANTICLOCKWISE;
			if (pTri->cornerptr[2]->pos.x > 0.0) tri_periodic_corner_flags.per2 = ROTATE_ME_ANTICLOCKWISE;
		}
		p_tri_periodic_corner_flags[iTri] = tri_periodic_corner_flags;
				
		memset(&tri_periodic_neigh_flags, 0, sizeof(CHAR4));
		tri_periodic_neigh_flags.flag = pTri->u8domain_flag;
		if ((pTri->periodic == 0) && (pTri->cent.x > 0.0)) {
			if (pTri->neighbours[0]->periodic != 0) 
				tri_periodic_neigh_flags.per0 = ROTATE_ME_CLOCKWISE;
			if (pTri->neighbours[1]->periodic != 0)
				tri_periodic_neigh_flags.per1 = ROTATE_ME_CLOCKWISE;
			if (pTri->neighbours[2]->periodic != 0)
				tri_periodic_neigh_flags.per2 = ROTATE_ME_CLOCKWISE;
		} else {
			// if we are NOT periodic but on left, neighs are not rotated rel to us.
			// If we ARE periodic but neigh is not and neigh cent > 0.0 then it is rotated.
			if (pTri->periodic != 0) {
				if ((pTri->neighbours[0]->periodic == 0) && (pTri->neighbours[0]->cent.x > 0.0))
					tri_periodic_neigh_flags.per0 = ROTATE_ME_ANTICLOCKWISE;
				if ((pTri->neighbours[1]->periodic == 0) && (pTri->neighbours[1]->cent.x > 0.0))
					tri_periodic_neigh_flags.per1 = ROTATE_ME_ANTICLOCKWISE;
				if ((pTri->neighbours[2]->periodic == 0) && (pTri->neighbours[2]->cent.x > 0.0))
					tri_periodic_neigh_flags.per2 = ROTATE_ME_ANTICLOCKWISE;
			}
		}
		p_tri_periodic_neigh_flags[iTri] = tri_periodic_neigh_flags;

		++pTri;
	};
	
}

void cuSyst::CopyStructuralDetailsFrom(cuSyst & src) // this assume both live on device
{
	// info contains flag .... do we know that?
	cudaMemcpy(p_info, src.p_info, sizeof(structural)*NMINOR, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_izTri_vert, src.p_izTri_vert, sizeof(long)*MAXNEIGH*Nverts, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_izNeigh_vert, src.p_izNeigh_vert, sizeof(long)*MAXNEIGH*Nverts, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_szPBCtri_vert, src.p_szPBCtri_vert, sizeof(char)*MAXNEIGH*Nverts, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_szPBCneigh_vert, src.p_szPBCneigh_vert, sizeof(char)*MAXNEIGH*Nverts, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_izNeigh_TriMinor, src.p_izNeigh_TriMinor, sizeof(long)*6*Ntris, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_szPBC_triminor, src.p_szPBC_triminor, sizeof(char)*6*Ntris, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_tri_corner_index, src.p_tri_corner_index, sizeof(LONG3) * Ntris, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_tri_periodic_corner_flags, src.p_tri_periodic_corner_flags, sizeof(CHAR4) * Ntris, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_tri_neigh_index, src.p_tri_neigh_index, sizeof(LONG3) * Ntris, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_tri_periodic_neigh_flags, src.p_tri_periodic_neigh_flags, sizeof(CHAR4) * Ntris, cudaMemcpyDeviceToDevice);
	cudaMemcpy(p_who_am_I_to_corner, src.p_who_am_I_to_corner, sizeof(LONG3) * Ntris, cudaMemcpyDeviceToDevice);

	// find another way would be better. Just a waste of memory and processing having duplicate info, creates unnecessary risks.

}
void cuSyst::PopulateTriMesh(TriMesh * pX)
{
	// AsSUMES THIS cuSyst has been allocated on the host.
	
	plasma_data data;
	long iMinor;
	for (iMinor = 0; iMinor < NMINOR; iMinor++)
	{
		data.n = p_n_minor[iMinor].n ;
		data.n_n = p_n_minor[iMinor].n_n ;
		if (iMinor > BEGINNING_OF_CENTRAL) {
			data.n = p_n_major[iMinor - BEGINNING_OF_CENTRAL].n ;
			data.n_n = p_n_major[iMinor - BEGINNING_OF_CENTRAL].n_n ;
		}
		data.Tn = p_T_minor[iMinor].Tn;
		data.Ti = p_T_minor[iMinor].Ti ;
		data.Te = p_T_minor[iMinor].Te ;
		data.Az = p_AAdot[iMinor].Az ;
		data.Azdot = p_AAdot[iMinor].Azdot ;
		data.v_n = p_v_n[iMinor] ;
		data.vxy = p_vie[iMinor].vxy;
		data.vez = p_vie[iMinor].vez;
		data.viz = p_vie[iMinor].viz ;
		data.B = p_B[iMinor] ;

		memcpy(&(pX->pData[iMinor]), &data, sizeof(plasma_data));
		pX->AreaMinorArray[iMinor] = p_AreaMinor[iMinor];
	};
}
                            
#endif
