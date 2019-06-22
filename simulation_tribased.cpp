
#define JACOBI  100
#define CG      200

#define VORONOI_A   

#define MASS_ONLY   0
#define ALL_VARS    1
#define NUMV		2
#define NUMT		3

#define COMBINE_EI_2ND_ORDER_PRESSURE 120
#define SEPARATE_ELECTRON_AND_ION        100

int const FEINT = 1;
int const REAL = 2;

int const PASS1 = 0;
int const COMPUTE_SIGMA_HSQ = 1;
int const ACCELERATE_2NDHALF = 2;
int const COMPUTE_SIGMA_H = 3;
int const FEINT_FOR_PRESSURE_EFFECT = 4;
int const ACCELERATION_AND_HEATING = 5;
int const ACCELERATE = 6;
int const FIRSTPASS = 7;
int const COMPUTE_SIGMA = 8;

int GlobalDebugSwitch = 0;

#include "bandlu.cpp"
#include <math.h>

real TotalEpsilon; // global storage
long iGlobalIteration;

bool bSwitchOffChPatReport;

real GlobalFrictionalHtg_ei, GlobalFrictionalHtg_en, GlobalFrictionalHtg_in, GlobalResistiveHtg, GlobalEnergyInput, GlobalIonisationHtg;
real GlobalViscousheatingtotal;

real Global_Ee_Exchrate_ei,Global_Ee_Exchrate_en,Global_Ee_Te,Global_Ee;
real GlobalFricHtgRate_in, GlobalFricHtgRate_en;

smartlong GlobalTrisVisited;

real GlobalMassReceived, GlobalHeatReceived;
// New and tidy version 21st April 2014
int GlobaliSrcTri, GlobaliSrcPlane,GlobaliSrcShape,GlobalSpecies;
real GlobalPlaneMomy,GlobalPlaneMass, GlobalSrcMomy, GlobalSrcMass;

extern long const NUM_COARSE_LEVELS;

bool GlobalInitial;
real Global_dIz_by_dEz, GlobalDefaultIz, GlobalIzElasticity,GlobalIzPrescribed,
		GlobalEz, GlobalOverall_OldA_Contrib_To_Iz;
real GlobalDefaultIzAux[NUM_COARSE_LEVELS];

bool bZeroSeed;
real scale_Ez;

// avi file -oriented variables
//extern HAVI hAviIon,hAviNeutral;
extern HBITMAP surfbit, dib;
extern HDC surfdc, dibdc;	
extern IDirect3DSurface9* p_backbuffer_surface;

#define NO										0
#define PERIODIC_SRC					1
#define CLIP_AGAINST_TRANCHE	2

FILE * debugfile, *resistfile;

const int VARCODE_MASS = 0;
const int VARCODE_HEAT = 1;
const int VARCODE_V_ALL = 2;
const int VARCODE_V_ALL_NOSMOOTH = 3;
const int VARCODE_VX = 4;
const int VARCODE_VY = 5;
const int VARCODE_VZ = 6;
const int VARCODE_SMOOTH_VISCOUS_HTG = 7;

real cross_T_vals[10] = {0.1,0.501187,1.0,1.99526,3.16228,5.01187,7.94328,12.5893,19.9526,31.6228};

bool boolVerbose = true;

// DEBUG:
real GlobalROC[12];

// momentum-transfer cross section data from http://www-cfadc.phy.ornl.gov/elastic/ddp/tel-DP.html
real cross_s_vals_momtrans_ni[10] = {
	1.210e-14,1.020e-14,9.784e-15,9.076e-15,8.589e-15,8.115e-15,7.653e-15,7.207e-15,6.776e-15,6.351e-15};
	// distinguishable particles:
	//4.408e-15,2.213e-15,1.666e-15,7.625e-16,4.685e-16,2.961e-16,1.878e-16,1.192e-16,7.442e-17,4.083e-17};
// viscosity cross section data from  http://www-cfadc.phy.ornl.gov/elastic/ddp/tel-DP.html
real cross_s_vals_viscosity_ni[10] = {
	4.904e-15,3.023e-15,2.673e-15,1.891e-15,1.203e-15,7.582e-16,4.891e-16,3.185e-16,2.030e-16,1.223e-16};
// viscosity cross section data from http://www-cfadc.phy.ornl.gov/elastic/dd0/tel.html
real cross_s_vals_viscosity_nn[10] = {
	1.753e-15,1.179e-15,9.030e-16,7.650e-16,6.316e-16,4.278e-16,2.685e-16,1.641e-16,9.609e-17,5.550e-17};

real GlobalAreaApportioned;
smartlong GlobalTrisToVisit;

real GlobalMassAttributed;
real GlobalHeatAttributed;

real GlobalMaxVertexRadiusSq;

FILE * fp;

extern int GlobalGraphSetting[4];

//#include "sim_fixed_A_mesh.cpp"		// just for tidiness moved all the FixedMesh code to a different file.


struct debugdata
{
	Vector3 pressure, qoverM_times_E, minus_omega_ci_cross_vi, 
		friction_from_neutrals, friction_e, thermal_force, a;
	Tensor3 Upsilon;
	real nu_ie, nu_in, nu_ei, nu_en, lnLambda;
};
debugdata Globaldebugdata;
bool GlobalDebugRecordIndicator = 0;
	
// Note: the quadrature pressure routine is stored in simulation EXISTING.cpp
#define VORONOI_PRESSURE
int GetRelativeRotation(Triangle * pTri, Triangle *pNeigh)
{
	//	rot2 = GetRelativeRotation(pTri,pNeigh2);
	// == 0 means same; == 1 means Neigh0 was clockwise relative to pTri; == 2 vice versa

	if (pTri->periodic == 0)
	{
		if (pNeigh->periodic == 0) return 0;
		if (pTri->cornerptr[0]->x > 0.0) return 2;
		return 0;
	};
	if (pNeigh->periodic > 0) return 0;
	if (pNeigh->cornerptr[0]->x > 0.0) return 1;
	return 0;
}
Tensor3 RotateCoordsIfNecessary(Triangle * pTri,Triangle * pNeigh,
								Tensor3 tensor3)
{
	// The tensor corresponds to pNeigh; we rotate coordinates if necessary to apply 
	// contiguous with pTri.
	if (pTri->periodic == 0) {
		if ((pNeigh->periodic == 0) || (pTri->cornerptr[0]->x < 0.0)) {
			return tensor3;
		} else {
			// pTri on right, pNeigh on left
			return Clockwise3*tensor3*Anticlockwise3;
		};
	} else {
		if ((pNeigh->periodic > 0) || (pNeigh->cornerptr[0]->x < 0.0)) {
			return tensor3;
		} else {
			// pNeigh on right, pTri on left
			return Anticlockwise3*tensor3*Clockwise3;
		};
	};
}

void GetInterpolationCoeffs( real beta[3],
											real x, real y,
							Vector2 pos0, Vector2 pos1, Vector2 pos2)
{
	// idea is to form a plane that passes through z0,z1,z2.

	// so firstly if we lie on a line between 0 and 1, we know what that is;
	// then we have some gradient in the direction normal to that which is determined by y2

	//relative = pos-pos0;
	//along01 = relative.dot(pos1-pos0)/(pos1-pos0).modulus(); 
	//// by being clever we should be able to avoid the square root since have z0 + (z1-z0)/(pos1-pos0).modulus()
	//perp.x = pos0.y-pos1.y;
	//perp.y = pos1.x-pos0.x;
	//away = relative.dot(perp)/perp.modulus();

	//pos2along01 = (pos2 - pos0).dot(pos1-pos0)/(pos1-pos0).modulus();
	//pos2away = (pos2-pos0).dot(perp)/perp.modulus();

	//real z_ = z0 + pos2along01*(z1-z0)/(pos1-pos0).modulus();
	//gradient_away = (z2-z_)/pos2away;

	//real z = z0 + along01*((z1-z0)/(pos1-pos0).modulus()) + away*gradient_away;
	//*pResult = z;


	// fast version:

	Vector2 pos(x,y);
	Vector2 perp;
	real ratio;//, coeff_on_z0, coeff_on_z1, coeff_on_z2;
	Vector2 relative = pos-pos0;
	Vector2 rel1 = pos1-pos0;
	Vector2 rel2 = pos2-pos0;
	real mod01sq = rel1.dot(rel1);
	real along01_over_mod01 = relative.dot(rel1)/mod01sq;
	real pos2along01_over_mod01 = rel2.dot(rel1)/mod01sq;
	//real z_expect = z0 + pos2along01_over_mod01*(z1-z0);
	//gradient_away = (z2-z_expect)*(perp.modulus()/((pos2-pos0).dot(perp)));
	//away_times_gradient_away = (z2-z_expect)*relative.dot(perp)/((pos2-pos0).dot(perp));
	//real z = z0 + along01_over_mod01*((z1-z0)) + away_times_gradient_away;

	// can we work out coefficients actually on z0,z1,z2 because then can do faster in 2D,3D. :
	
	perp.x = -rel1.y;
	perp.y = rel1.x;
	ratio = relative.dot(perp)/(rel2.dot(perp));
	
	//beta[0] = 1.0 - along01_over_mod01 - ratio + ratio*pos2along01_over_mod01;
	beta[1] =         along01_over_mod01             - ratio*pos2along01_over_mod01;
	beta[2] =                                              ratio;
	beta[0] = 1.0 - beta[1] - beta[2];
	
	//*pResult = coeff_on_z0*z0 + coeff_on_z1*z1 + coeff_on_z2*z2;
}

void Triangle::ReturnCircumcenter(Vector2 & u, Vertex * pVertex)
{
	// assumes pTri->cc already populated
	// returns rotated vector if triangle is peridic and pVertex on right
	if ((periodic == 0) || (pVertex->x < 0.0))
	{
		u = cc;
	} else {
		u = Clockwise*cc;	 
	};
}
	
void SlimTriangle::ReturnCircumcenter(Vector2 & u, SlimVertex * pVertex)
{
	// assumes pTri->cc already populated
	// returns rotated vector if triangle is peridic and pVertex on right
	if ((periodic == 0) || (pVertex->x < 0.0))
	{
		u = cc;
	} else {
		u = Clockwise*cc;	 
	};
}

void Triangle::CreateCoordinates(Vertex * pVertex,Vector2 & u1, Vector2 & u2, Vector2 & u3)
{
	// give positions on same side as pVertex
	if (periodic == 0) 
	{
		PopulatePositions(u1,u2,u3);
	} else {
		if (pVertex->x < 0.0)
		{
			MapLeft(u1,u2,u3);
		} else {
			MapRight(u1,u2,u3);
		};
	};

	//
	//if (flags == 3) {
	//	// Note the coords are expected to go round in a circle:
	//	u2.project_to_ins(u3);
	//	u1.project_to_ins(u4);
	//};
	//if (flags == 4) {
	//	u2.project_to_radius(u3,u2.modulus()+DELTA_NOTIONAL);
	//	u1.project_to_radius(u4,u1.modulus()+DELTA_NOTIONAL);
	//};
}
	
real GetMedian(real * median_vals, long * usable, long * flag, long len, int * index_med)
{
	// pick a random element - make it halfway
	
	// sort to left and right of pivot
	long * pCaret, * pTop, *pPivot, *pTemp, *pEnd;
	long * useptr = usable;
	long * usetopptr = usable + len-1;
	long numTooHigh = 0, numTooLow = 0;
	long pivot_index;
	long numLow, numHigh;	
	real value, temp;
	FILE * debugfile;
	char buffer[1024];

	long sublen = len;
	long start = 0; // start of relevant portion of usable array

	// New plan: we want to return the index of the median element, for sure.
	// So we need array of indices.
	// We choose a way where we keep the reals array fixed and use the index array for lookup.

	for (int i = 0; i < len; i++)
	{
		usable[i] = i; // initially 
	};

	int intermed = 0;
	while (1)
	{
		// Now imagine this when there are some to either side.
		// Use "usable" array from useptr to usetopptr to index the values we want to access.

		numLow = 0;
		numHigh = 0;
		pCaret = usable + start;
		pTop = usable + (start + sublen - 1);
		pivot_index = start + sublen/2; // may wish to randomise
		pPivot = usable + pivot_index;
		value = median_vals[*pPivot];
		
		while (pCaret <= pTop) //(pCaret < pTop) // this was not right - pTop is a testable element and pCaret is what we test
		{
			if (pCaret == pPivot) {
				// what to do for pivot_index ?
				// Do nothing yet - leave it in the lower side
				++pCaret;
			} else {
				if (median_vals[*pCaret] < value) {
					++numLow;
					// keep it where it is
					++pCaret;
				} else {
					++numHigh;
					// swap with the last (unswapped) element - which may be the pivot, or itself
					temp = *pTop;
					*pTop = *pCaret;
					*pCaret = temp;
					if (pTop == pPivot) {
						pPivot = pCaret; // move ptr where value was sent
					}
					
					--pTop;
				};
			};
		}; 
		if (numLow + numHigh != sublen-1)
		{ // debug
			sublen = sublen;
		};
		// 
		
		// swap with pCaret to put *pPivot in the middle --
		// that's no good if pCaret is now on a larger element so walk it backwards 1 if so.
		
		// Depends whether the last element was found to be too high or too low or was the pivot.

		if (pPivot != pCaret){
			// note that pCaret is now always on an element that was too high
			// but the previous element is always too low

			// What if pCaret = start? ie, we never found a low element but kept moving down
			// ptop ? Not possible - we swapped pPivot in here at some stage and then went past it.
			pCaret--;
			temp = *pPivot;
			*pPivot = *pCaret;
			*pCaret = temp;
			pPivot = pCaret; // now move "pPivot" to the place where the pivot element now is.
		};
		// we use < median as criterion for separating... therefore if even #,
		// return the one that is on the high side and use index_med to identify that when we reach it.
		// ie, always put index_med in the high half.

			// DEBUG:
			// Output all those we know to be "below" and their median_vals
		
			//sprintf(buffer,"intermed%d.txt",intermed);
			//intermed++;
			//debugfile = fopen(buffer,"w");
			//fprintf(debugfile,"TooLow: %d Low: %d High: %d TooHigh: %d \n\n",numTooLow,numLow,numHigh,numTooHigh);
			//fprintf(debugfile,"too low!:\n\n:");
			//for (pTemp = usable; pTemp < usable + start; pTemp++)
			//{
			//	fprintf(debugfile,"%d  %1.15E  \n",*pTemp,median_vals[*pTemp]);
			//};

			//fprintf(debugfile,"\nLow:\n\n");
			//
			//for (; pTemp < pPivot; pTemp++)
			//{
			//	fprintf(debugfile,"%d  %1.15E  \n",*pTemp,median_vals[*pTemp]);
			//};

			//fprintf(debugfile,"\nMEDIAN %d %1.15E \n\n",*pPivot,value);
			//++pTemp;

			//// Output all those we know to be "above" and their median_vals

			//fprintf(debugfile,"\nHigh:\n\n");
			//
			//for (; pTemp < usable + start + sublen; pTemp++)
			//{
			//	fprintf(debugfile,"%d  %1.15E  \n",*pTemp,median_vals[*pTemp]);
			//};
			//fprintf(debugfile,"\nToo High:\n\n");

			//for (; pTemp < usable + len; pTemp++)
			//{
			//	fprintf(debugfile,"%d  %1.15E  \n",*pTemp,median_vals[*pTemp]);
			//};

			//fclose(debugfile);



		if ((numHigh + numTooHigh == numLow + numTooLow) || (numHigh + numTooHigh == numLow-1 + numTooLow)) {
			*index_med = *pPivot;

			
			// Then we shall compare to what results we are getting in the caller: is it down to FP?
			// If so then do we need to somehow assign labels within here? Good solution.

			pEnd = usable + len;
			for (pCaret = usable; pCaret < pPivot; pCaret++)
				flag[*pCaret] = 0;
			for (;pCaret < pEnd; pCaret++)
				flag[*pCaret] = 1;
			
			return value;
		};
		
		// We have sorted the 'usable' array into 2 parts
		if (numHigh + numTooHigh > numLow + numTooLow)
		{
			// recurse into higher set;
			// pivot element is part of "too low"
			numTooLow += numLow+1;
			start = (pCaret-usable)+1; // note that pCaret is on pivot
			sublen = sublen - (numLow+1);
		} else {
			numTooHigh += numHigh+1;
			sublen = sublen - (numHigh+1);
		};
		

	}; // while (1)

	
} // end of function

int BilateralLevels = 0;
#define OPTI 1

/*void TriMesh::CreateVolleys(int separation) 
{
	// pick a place to start ... set all neighbours to be the next volley up, unless they are in a volley already counted.
	// insulator vertices are also to be included.

	// Initially just doing this so that volleys are not touching, but we probably need volleys that are separated by at least 2.

	long iVertex;
	Vertex * pVertex, *pNeigh, *pNeighNeigh;
	int i,j;
	bool next_volley_made;
	// THIS ROUTINE ASSUMES that we already refreshed vertex neighbours of vertices.

	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		pVertex->iVolley = 0;
		++pVertex;
	};

	numVolleys = 0;
	next_volley_made = 1;
	while (next_volley_made)
	{
		next_volley_made = 0;
		pVertex = X;
		for (iVertex = 0; iVertex < numVertices; iVertex++)
		{
			if (pVertex->iVolley == numVolleys)
			{
				for (i = 0; i < pVertex->neighbours.len; i++)
				{
					pNeigh = X + pVertex->neighbours.ptr[i];
					if (pNeigh->iVolley == numVolleys)
					{
						pNeigh->iVolley++; // bump all neighbours at this volley up to the next volley
						next_volley_made = 1;
					};
					if (separation == 2) {
						for (j = 0; j < pNeigh->neighbours.len; j++)
						{
							pNeighNeigh = X + pNeigh->neighbours.ptr[j];
							if ((pNeighNeigh->iVolley == numVolleys) && (pNeighNeigh != pVertex))
							{
								pNeighNeigh->iVolley++; // bump all neighbours at this volley up to the next volley
								next_volley_made = 1;
							};
						};
					};
				};
				if (pVertex->flags == 1) { // 3 extra 'neighbours' :
					pNeigh = insulator_verts + pVertex->iScratch;
					if (pNeigh->iVolley == numVolleys)
					{
						pNeigh->iVolley++; // bump all neighbours at this volley up to the next volley
						next_volley_made = 1;
					};
					pNeigh = X + pVertex->neighbours.ptr[0];
					if (pNeigh->flags != 1) { 
						printf("errror\n"); 
						getch(); };
					pNeigh = insulator_verts + pNeigh->iScratch;
					if (pNeigh->iVolley == numVolleys)
					{
						pNeigh->iVolley++; // bump all neighbours at this volley up to the next volley
						next_volley_made = 1;
					};
					if (separation == 2) {
						pNeighNeigh = insulator_verts + pNeigh->neighbours.ptr[0];
						if (pNeighNeigh->iVolley == numVolleys)
						{
							pNeighNeigh->iVolley++; 
							next_volley_made = 1;
						};
						pNeighNeigh = insulator_verts + pNeigh->neighbours.ptr[2];
						if (pNeighNeigh->iVolley == numVolleys)
						{
							pNeighNeigh->iVolley++; 
							next_volley_made = 1;
						};
					};
					pNeigh = X + pVertex->neighbours.ptr[pVertex->neighbours.len-1];
					if (pNeigh->flags != 1) { 
						printf("errror\n"); 
						getch(); };
					pNeigh = insulator_verts + pNeigh->iScratch;
					if (pNeigh->iVolley == numVolleys)
					{
						pNeigh->iVolley++; // bump all neighbours at this volley up to the next volley
						next_volley_made = 1;
					};
					
					if (separation == 2) {
						pNeighNeigh = insulator_verts + pNeigh->neighbours.ptr[0];
						if (pNeighNeigh->iVolley == numVolleys)
						{
							pNeighNeigh->iVolley++; 
							next_volley_made = 1;
						};
						pNeighNeigh = insulator_verts + pNeigh->neighbours.ptr[2];
						if (pNeighNeigh->iVolley == numVolleys)
						{
							pNeighNeigh->iVolley++; 
							next_volley_made = 1;
						};
					};
				};
			}; // otherwise do nothing
			++pVertex;
		};

		pVertex = insulator_verts;
		for (iVertex = 0; iVertex < numVertsLow; iVertex++)
		{
			if (pVertex->iVolley == numVolleys)
			{
				pNeigh = insulator_verts + pVertex->neighbours.ptr[0];
				if (pNeigh->iVolley == numVolleys)
				{
					pNeigh->iVolley++; // bump all neighbours at this volley up to the next volley
					next_volley_made = 1;
				};
				
				if (separation == 2) {
					pNeighNeigh = insulator_verts + pNeigh->neighbours.ptr[0];
					if ((pNeighNeigh != pVertex) && (pNeighNeigh->iVolley == numVolleys))
					{
						pNeighNeigh->iVolley++; // bump all neighbours at this volley up to the next volley
						next_volley_made = 1;
					};
					pNeighNeigh = insulator_verts + pNeigh->neighbours.ptr[2];
					if ((pNeighNeigh != pVertex) && (pNeighNeigh->iVolley == numVolleys))
					{
						pNeighNeigh->iVolley++; // bump all neighbours at this volley up to the next volley
						next_volley_made = 1;
					};
				};
				pNeigh = X + pVertex->neighbours.ptr[1];
				if (pNeigh->iVolley == numVolleys)
				{
					pNeigh->iVolley++; // bump all neighbours at this volley up to the next volley
					next_volley_made = 1;
				};
				pNeigh = insulator_verts + pVertex->neighbours.ptr[2];
				if (pNeigh->iVolley == numVolleys)
				{
					pNeigh->iVolley++; // bump all neighbours at this volley up to the next volley
					next_volley_made = 1;
				};
				
				if (separation == 2) {
					pNeighNeigh = insulator_verts + pNeigh->neighbours.ptr[0];
					if ((pNeighNeigh != pVertex) && (pNeighNeigh->iVolley == numVolleys))
					{
						pNeighNeigh->iVolley++; // bump all neighbours at this volley up to the next volley
						next_volley_made = 1;
					};
					pNeighNeigh = insulator_verts + pNeigh->neighbours.ptr[2];
					if ((pNeighNeigh != pVertex) && (pNeighNeigh->iVolley == numVolleys))
					{
						pNeighNeigh->iVolley++; // bump all neighbours at this volley up to the next volley
						next_volley_made = 1;
					};
				};
				pNeigh = X + pVertex->neighbours.ptr[3];
				if (pNeigh->iVolley == numVolleys)
				{
					pNeigh->iVolley++; // bump all neighbours at this volley up to the next volley
					next_volley_made = 1;
				};
				pNeigh = X + pVertex->neighbours.ptr[4];
				if (pNeigh->iVolley == numVolleys)
				{
					pNeigh->iVolley++; // bump all neighbours at this volley up to the next volley
					next_volley_made = 1;
				};
			}; // otherwise do nothing
			++pVertex;
		};

		numVolleys++;
	};
	// That's our first attempt. It was simple. Want to have ins verts in there also.
}
*/
void TriMesh::CreateTriangleVolleys()
{
	// each volley consists of triangles not linked by a chain with 0 or 1 other neighbour

	// store in "indicator".

	Triangle * pTri, *pNeigh, *pNeighNeigh;
	Vertex *pVertex;
	long iTri;
	int i,j;
	bool next_volley_made;

	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		pTri->iVolley = 0;
		++pTri;
	};
	
	numVolleys = 0;
	next_volley_made = 1;
	while (next_volley_made)
	{
		next_volley_made = 0;
		pTri = T;
		for (iTri = 0; iTri < numTriangles; iTri++)
		{
			if (pTri->iVolley == numVolleys)
			{
				for (i = 0; i < (pTri->flags == 0?3:2); i++) // note: one day triangles may not all have 3 neighbours.
				{
					pVertex = pTri->cornerptr[i];

					for (j = 0; j < pVertex->triangles.len; j++)
					{
						pNeigh = (Triangle *)(pVertex->triangles.ptr[j]);
						if ((pNeigh->iVolley == numVolleys) && (pNeigh != pTri))
						{
							pNeigh->iVolley++;
							next_volley_made = 1;
						};
					};
				};

			}; // otherwise do nothing
			++pTri;
		};
		numVolleys++;
	};

	printf("numVolleys = %d\n",numVolleys); // should be 4 usually.?

}


// Short global functions:
real inline Get_lnLambda(real n_e,real T_e)
{
	real lnLambda, factor, lnLambda_sq, lnLambda1, lnLambda2;

	static real const one_over_kB = 1.0/kB;
	
	real Te_eV = T_e*one_over_kB;
	real Te_eV2 = Te_eV*Te_eV;
	real Te_eV3 = Te_eV*Te_eV2;

	if (n_e*Te_eV3 > 0.0) {
		
		lnLambda1 = 23.0 - 0.5*log(n_e/Te_eV3);
		lnLambda2 = 24.0 - 0.5*log(n_e/Te_eV2);
		// smooth between the two:
		factor = 2.0*fabs(Te_eV-10.0)*(Te_eV-10.0)/(1.0+4.0*(Te_eV-10.0)*(Te_eV-10.0));
		lnLambda = lnLambda1*(0.5-factor)+lnLambda2*(0.5+factor);
		
		// floor at 2 just in case, but it should not get near:
		lnLambda_sq = lnLambda*lnLambda;
		factor = 1.0+0.5*lnLambda+0.25*lnLambda_sq+0.125*lnLambda*lnLambda_sq + 0.0625*lnLambda_sq*lnLambda_sq;
		lnLambda += 2.0/factor;

		// Golant p.40 warns that it becomes invalid when an electron gyroradius is less than a Debye radius. That is something to worry about if  B/400 > n^1/2 , so looks not a big concern.

		// There is also a quantum ceiling. It will not be anywhere near. At n=1e20, 0.5eV, the ceiling is only down to 29; it requires cold dense conditions to apply.

	} else {
		lnLambda = 20.0;
	};
	if (GlobalDebugRecordIndicator)
		Globaldebugdata.lnLambda = lnLambda;
	return lnLambda;
}		


real inline Get_lnLambda_ion(real n_ion,real T_ion)
{
	static real const one_over_kB = 1.0/kB; // multiply by this to convert to eV
	static real const one_over_kB_cubed = 1.0/(kB*kB*kB); // multiply by this to convert to eV
	
	real factor, lnLambda_sq;

	real Tion_eV3 = T_ion*T_ion*T_ion*one_over_kB_cubed;
	
	real lnLambda = 23.0 - 0.5*log(n_ion/Tion_eV3);
	
	// floor at 2:
	lnLambda_sq = lnLambda*lnLambda;
	factor = 1.0+0.5*lnLambda+0.25*lnLambda_sq+0.125*lnLambda*lnLambda_sq + 0.0625*lnLambda_sq*lnLambda_sq;
	lnLambda += 2.0/factor;

	return lnLambda;
}		


real EstimateDistanceBetweenCentres(Triangle * pTri1,Triangle * pTri2)
{
	Vector2 centre1,centre2;
	Vector2 u[3],x[3];
	// Assume always a neighbour tri. 

	if (pTri1->periodic == 0)
	{
		pTri1->PopulatePositions(u[0],u[1],u[2]);
			
		if (pTri2->periodic == 0)
		{
			// typical case:
			pTri2->PopulatePositions(x[0],x[1],x[2]);
		} else {
			if (pTri1->cornerptr[0]->x < 0.0) {
				pTri2->MapLeft(x[0],x[1],x[2]);
			} else {
				pTri2->MapRight(x[0],x[1],x[2]);
			};
		};
	} else {
		if (pTri2->periodic == 0)
		{
			pTri2->PopulatePositions(x[0],x[1],x[2]);

			if (pTri2->cornerptr[0]->x < 0.0) 	{
				pTri1->MapLeft(u[0],u[1],u[2]);
			} else {
				pTri1->MapRight(u[0],u[1],u[2]);
			};
		} else {
			// both periodic
			pTri1->MapLeft(u[0],u[1],u[2]);
			pTri2->MapLeft(x[0],x[1],x[2]);
		};
	};

//	if (pTri1->flags == 0)
//	{
		centre1 = THIRD*(u[0] + u[1] + u[2]);
	//} else {
	//	Vector2 xdash0,xdash1;

	//	if (pTri1->flags == 1)
	//	{
	//		u[0].project_to_ins(xdash0);
	//		u[1].project_to_ins(xdash1);
	//	} else {
	//		// outer wedge - funny case
	//		u[0].project_to_radius(xdash0,HIGH_WEDGE_OUTER_RADIUS);
	//		u[1].project_to_radius(xdash1,HIGH_WEDGE_OUTER_RADIUS);
	//	};
	//	centre1 = 0.25* (u[0] + xdash0 + u[1] + xdash1);
	//};

	//if (pTri2->flags == 0)
	//{
		centre2 = THIRD*(x[0] + x[1] + x[2]);
	//} else {
	//	
	//	Vector2 xdash0,xdash1;

	//	// only pTri2 is a wedge
	//	if (pTri2->flags == 1)
	//	{
	//		x[0].project_to_ins(xdash0);
	//		x[1].project_to_ins(xdash1);
	//	} else {
	//		// outer wedge
	//		x[0].project_to_radius(xdash0,HIGH_WEDGE_OUTER_RADIUS);
	//		x[1].project_to_radius(xdash1,HIGH_WEDGE_OUTER_RADIUS);
	//	};		
	//	centre2 = 0.25*(x[0] + xdash0 + x[1] + xdash1);
	//};
	
	return (centre1-centre2).modulus();
}

real Estimate_Neutral_Neutral_Viscosity_Cross_section(real T) // call with T in electronVolts
{
	if (T > cross_T_vals[9]) return cross_s_vals_viscosity_nn[9];
	if (T < cross_T_vals[0]) return cross_s_vals_viscosity_nn[0];
	int i = 1;
	while (T > cross_T_vals[i]) i++;
	// T lies between i-1,i
	real ppn = (T-cross_T_vals[i-1])/(cross_T_vals[i]-cross_T_vals[i-1]);
	return ppn*cross_s_vals_viscosity_nn[i] + (1.0-ppn)*cross_s_vals_viscosity_nn[i-1];
}

void Estimate_Ion_Neutral_Cross_sections(real T, // call with T in electronVolts
													real * p_sigma_in_MT,
													real * p_sigma_in_visc)
{
	if (T > cross_T_vals[9]) {
		*p_sigma_in_MT = cross_s_vals_momtrans_ni[9];
		*p_sigma_in_visc = cross_s_vals_viscosity_ni[9];
		return;
	}
	if (T < cross_T_vals[0]){
		*p_sigma_in_MT = cross_s_vals_momtrans_ni[0];
		*p_sigma_in_visc = cross_s_vals_viscosity_ni[0];
		return;
	}
	int i = 1;
	while (T > cross_T_vals[i]) i++;
	// T lies between i-1,i
	real ppn = (T-cross_T_vals[i-1])/(cross_T_vals[i]-cross_T_vals[i-1]);

	*p_sigma_in_MT = ppn*cross_s_vals_momtrans_ni[i] + (1.0-ppn)*cross_s_vals_momtrans_ni[i-1];
	*p_sigma_in_visc = ppn*cross_s_vals_viscosity_ni[i] + (1.0-ppn)*cross_s_vals_viscosity_ni[i-1];
	return;
}

real Estimate_Ion_Neutral_MomentumTransfer_Cross_section(real T) // call with T in electronVolts
{
	if (T > cross_T_vals[9]) return cross_s_vals_momtrans_ni[9];
	if (T < cross_T_vals[0]) return cross_s_vals_momtrans_ni[0];
	int i = 1;
	while (T > cross_T_vals[i]) i++;
	// T lies between i-1,i
	real ppn = (T-cross_T_vals[i-1])/(cross_T_vals[i]-cross_T_vals[i-1]);
	return ppn*cross_s_vals_momtrans_ni[i] + (1.0-ppn)*cross_s_vals_momtrans_ni[i-1];
}

bool triplanar_var(real * pn0, real * pn1, real * pn2, real n_avg, real * pnC)
{
	real n[3];
	real n_C;
	bool fail_n_tri, otherabove;
	n[0] = *pn0; n[1] = *pn1; n[2] = *pn2; n_C = *pnC;
	int other, above, below;

	// task of routine is to modify contents of pointers if necessary
	// to attain average n_avg
	// and report failure if that is not possible within the chosen rules

	real minncorner = min(min(n[0],n[1]),n[2]);
	real maxncorner = max(max(n[0],n[1]),n[2]);
	if ((n_avg < minncorner) || (n_avg> maxncorner))
		return true; // fail - have to set to uniform

	// find index of element that does not belong to n_avg's interval - and whether it is higher or lower

	if (n[0] >n[1])
	{
		if (n[2] > n[0]) {
			// 2 > 0 > 1
			if (n_avg > n[0]) {
				above = 2;					below = 0;					other = 1;					otherabove = false;
			} else {
				above = 0;					below = 1;					other = 2;					otherabove = true;
			};
		} else {
			if (n[2] > n[1]) {
				// 0 > 2 > 1
				if (n_avg > n[2]) {
					above = 0;					below = 2;					other = 1;					otherabove = false;
				} else {
					above = 2;					below = 1;					other = 0;					otherabove = true;
				};
			} else {
				// 0 > 1 > 2
				if (n_avg > n[1]) {
					above = 0;					below = 1;					other = 2;					otherabove = false;
				} else {
					above = 1;					below = 2;					other = 0;					otherabove = true;
				};
			};
		};
	} else {
		if (n[2] > n[1]) {
			// 2 > 1 > 0
			if (n_avg > n[1]) {
				above = 2;					below = 1;					other = 0;					otherabove = false;
			} else {
				above = 1;					below = 0;					other = 2;					otherabove = true;
			};
		} else {
			if (n[2] > n[0]) {
				// 1 > 2 > 0
				if (n_avg > n[2]) {
					above = 1;					below = 2;					other = 0;					otherabove = false;
				} else {
					above = 2;					below = 0;					other = 1;					otherabove = true;
				};
			} else {
				// 1 > 0 > 2
				if (n_avg > n[0]) {
					above = 1;					below = 0;					other = 2;					otherabove = false;
				} else {
					above = 0;					below = 2;					other = 1;					otherabove = true;
				};
			};
		};
	};

	real n_C_implied = 3.0 * n_avg - TWOTHIRDS*( n[0]+n[1]+n[2]);
	real maxlow, maxhigh;

#ifdef RELEASE
	if (n_C_implied > n[above]) {
		n_C = n[above];
		if (otherabove == true) {
			n[below] = 4.5*(n_avg - THIRD*n_C - TWONINTHS*(n[above]+n[other]));	
		} else {
			maxlow = 2.25*(n_avg-THIRD*n_C - TWONINTHS*n[above]);
			if (maxlow < n[below])
			{
				n[other] = 4.5*(n_avg-THIRD*n_C - TWONINTHS*(n[above]+n[below]));				
			} else {
				n[below] = maxlow;
				n[other] = maxlow;
			};
		};
	} else {
		if (n_C_implied < n[below]) {
			n_C = n[below];
			if (otherabove == true) {
				maxhigh = 2.25*(n_avg-THIRD*n_C-TWONINTHS*n[below]);
				if (maxhigh > n[above])
				{
					n[other] = 4.5*(n_avg - THIRD*n_C - TWONINTHS*(n[above]+n[below]));					
				} else {
					n[above] = maxhigh;
					n[other] = maxhigh;
				};
			} else {
				n[above] = 4.5*(n_avg-THIRD*n_C - TWONINTHS*(n[below]+n[other]));				
			};
		} else {
			// n_C_implied was within the same interval of corner n as n_avg
			n_C = n_C_implied;				
		};
	};
#else
	if (n_C_implied > n[above]) {
		n_C = n[above];
		if (otherabove == true) {
			// the one above can be attained ; n[below] cannot.
			// infer n[below] :
			nbelow = 4.5*(n_avg - THIRD*n_C - TWONINTHS*(n[above]+n[other]));
			// THIRD*n_C + 2/9 (n[0]+n[1]+n[2]) must = n_avg
			
			// debug:
			if (nbelow < n[below]) {
				printf("code error");
			};
			n[below] = nbelow;
		} else {
			// other cannot be attained. we need to make a downward slope to see if below can be attained.
			
			maxlow = 2.25*(n_avg-THIRD*n_C - TWONINTHS*n[above]);
			if (maxlow < n[below])
			{
				// n[below] can be attained.
				// debug:
				nother = 4.5*(n_avg-THIRD*n_C - TWONINTHS*(n[above]+n[below]));
				if (nother < n[other]) {
					printf("code error");
				}
				n[other] = nother;
			} else {
				n[below] = maxlow;
				n[other] = maxlow;
			};
		};
	} else {
		if (n_C_implied < n[below]) {
			n_C = n[below];
			if (otherabove == true) {
				maxhigh = 2.25*(n_avg-THIRD*n_C-TWONINTHS*n[below]);
				if (maxhigh > n[above])
				{
					nother = 4.5*(n_avg - THIRD*n_C - TWONINTHS*(n[above]+n[below]));
					if (nother > n[other]) {
						printf("code error");
					}
					n[other] = nother;
				} else {
					n[above] = maxhigh;
					n[other] = maxhigh;
				};
			} else {
				nabove = 4.5*(n_avg-THIRD*n_C - TWONINTHS*(n[below]+n[other]));
				if (nabove > n[above]) {
					printf("code error");
				}
				n[above] = nabove;					
			};
		} else {
			// n_C_implied was within the same interval of corner n as n_avg
			// good news! accept.
			n_C = n_C_implied;				
		};
	};

#endif

	*pn0 = n[0];
	*pn1 = n[1];
	*pn2 = n[2];
	*pnC = n_C;

	return false;
}



bool Triplanar(cellvars * pVars, vertvars * pVertvars0, vertvars * pVertvars1,
			   vertvars * pVertvars2, vertvars * pVerts_c, real Area, real Areanew, int code )
{
	
	real n[3];
	real nT[3];
	real n_avg,n_C,nT_avg,nT_C,T_avg,T_C;
	Vector3 nv[3],nv_avg,nv_C,v_C,v_avg; // used for sending, not advection

	real minnvxcorner,maxnvxcorner,minnvycorner,maxnvycorner,minnvzcorner,maxnvzcorner;
	real vx_avg,vy_avg,vz_avg;

	bool fail_nT_tri,fail_nvx_tri,fail_nvy_tri,fail_nvz_tri;
	bool fail_n_tri;
	real maxncorner, minncorner, minnTcorner, maxnTcorner, minTcorner, maxTcorner;

	int above, below, other;
	bool otherabove;

	static real const LOWMINRATIO = 0.95;  // allow n_C to be down to LOWMINRATIO*n_avg if below all corners
	static real const HIGHMAXRATIO = 1.05;

	real scale = Area/Areanew;

	pVertvars0->n *= scale; // that's right, isn't it.
	pVertvars1->n *= scale;
	pVertvars2->n *= scale;
	n[0] =  pVertvars0->n;
	n[1] =  pVertvars1->n;
	n[2] =  pVertvars2->n;

	n_avg = pVars->mass/Areanew; // was /Area
	// mass = (Area) (2/9 (n1 + n2 + n3) + 3/9 n_C)

	// New version - see logic3.lyx

	fail_n_tri = triplanar_var(&(n[0]),&(n[1]),&(n[2]),n_avg,&(n_C));
		
	//n_C = 3.0 * n_avg - TWOTHIRDS*( n0+n1+n2);

	//fail_n_tri = 0;
	//minncorner = n0;
	//if (n1 < n0) minncorner = n1;
	//if (n2 < minncorner) minncorner = n2;
	////if (n_C < 0.0) fail_n_tri = 1; // obviously a fail if it requires the centre n below zero
	//if ((n_C < minncorner) && (n_C < LOWMINRATIO*n_avg)) fail_n_tri = 1;// 0.75*n_avg)) fail_n_tri = 1;

	//maxncorner = n0;
	//if (n1 > n0)  maxncorner = n1;
	//if (n2 > maxncorner) maxncorner = n2;
	//if ((n_C > maxncorner) && (n_C > HIGHMAXRATIO*n_avg)) fail_n_tri = 1;//FOURTHIRDS*n_avg)) fail_n_tri = 1;
			
	if (fail_n_tri)
	{
		// Set n to flat:

		pVertvars0->n = n_avg;
		pVertvars1->n = n_avg;
		pVertvars2->n = n_avg;
		pVerts_c->n = n_avg;

		n[0] = n_avg;
		n[1] = n_avg;
		n[2] = n_avg; // set for rest of this routine...
		n_C = n_avg;
	} else {
		pVertvars0->n = n[0];
		pVertvars1->n = n[1];
		pVertvars2->n = n[2];
		pVerts_c->n = n_C;
	};

	if (code == MASS_ONLY) return fail_n_tri;

	// get nv
	nv_avg = pVars->mom/Areanew; // /Area
	nv[0] = n[0]*pVertvars0->v; // note that this n0 was already spread out
	nv[1] = n[1]*pVertvars1->v;
	nv[2] = n[2]*pVertvars2->v;
	// how does DKE change when n is compressed? # v ^2 = same.
				
	//nv_C_implied = 3.0 * nv_avg - TWOTHIRDS*(nv[0] + nv[1] + nv[2]);

	// Do same thing with nvx :

	fail_nvx_tri = triplanar_var(&(nv[0].x),&(nv[1].x),&(nv[2].x),nv_avg.x,&(nv_C.x));
	
	if (fail_nvx_tri) 
	{
		vx_avg = pVars->mom.x/pVars->mass;
		pVertvars0->v.x = vx_avg;
		pVertvars1->v.x = vx_avg;
		pVertvars2->v.x = vx_avg;
		pVerts_c->v.x = vx_avg;
	} else {
		pVertvars0->v.x = nv[0].x/n[0];
		pVertvars1->v.x = nv[1].x/n[1];
		pVertvars2->v.x = nv[2].x/n[2];
		pVerts_c->v.x = nv_C.x / n_C;
	};
	
	fail_nvy_tri = triplanar_var(&(nv[0].y),&(nv[1].y),&(nv[2].y),nv_avg.y,&(nv_C.y));
	
	if (fail_nvy_tri) 
	{
		vy_avg = pVars->mom.y/pVars->mass;
		pVertvars0->v.y = vy_avg;
		pVertvars1->v.y = vy_avg;
		pVertvars2->v.y = vy_avg;
		pVerts_c->v.y = vy_avg;
	} else {
		pVertvars0->v.y = nv[0].y/n[0];
		pVertvars1->v.y = nv[1].y/n[1];
		pVertvars2->v.y = nv[2].y/n[2];
		pVerts_c->v.y = nv_C.y / n_C;
	};
	
	fail_nvz_tri = triplanar_var(&(nv[0].z),&(nv[1].z),&(nv[2].z),nv_avg.z,&(nv_C.z));
	
	if (fail_nvz_tri) 
	{
		vz_avg = pVars->mom.z/pVars->mass;
		pVertvars0->v.z = vz_avg;
		pVertvars1->v.z = vz_avg;
		pVertvars2->v.z = vz_avg;
		pVerts_c->v.z = vz_avg;
	} else {
		pVertvars0->v.z = nv[0].z/n[0];
		pVertvars1->v.z = nv[1].z/n[1];
		pVertvars2->v.z = nv[2].z/n[2];
		pVerts_c->v.z = nv_C.z / n_C;
	};
	
	fail_nvz_tri = triplanar_var(&(nv[0].z),&(nv[1].z),&(nv[2].z),nv_avg.z,&(nv_C.z));
	
	nT_avg = pVars->heat/Areanew;

	nT[0] = n[0]*pVertvars0->T;
	nT[1] = n[1]*pVertvars1->T;
	nT[2] = n[2]*pVertvars2->T;
		
	//nT_C = 3.0 * nT_avg - TWOTHIRDS*(nT[0] + nT[1] + nT[2]);

	fail_nT_tri = triplanar_var(&(nT[0]),&(nT[1]),&(nT[2]),nT_avg,&(nT_C));

	if (fail_nT_tri) 
	{
		T_avg = pVars->heat/pVars->mass;

		pVertvars0->T = T_avg;
		pVertvars1->T = T_avg;
		pVertvars2->T = T_avg;
		pVerts_c->T = T_avg;
	} else {
		pVertvars0->T = nT[0]/n[0];
		pVertvars1->T = nT[1]/n[1];
		pVertvars2->T = nT[2]/n[2];
		pVerts_c->T = nT_C / n_C;
	};
	
	//// FAIL CASES:
	//// As with n and nT we insist for each dimension that it be within the corner values, or that union with (LOWMINRATIO*avg,HIGHMAXRATIO*avg)
	//// Just bear in mind that since avg may be negative, HIGHMAXRATIO*avg may be lower.

	//fail_nvx_tri = 0;
	//fail_nvy_tri = 0;
	//fail_nvz_tri = 0;

	//minnvxcorner = nv0.x;
	//if (nv1.x < nv0.x) minnvxcorner = nv1.x;
	//if (nv2.x < minnvxcorner) minnvxcorner = nv2.x;
	//if ((nv_C.x < minnvxcorner) && (nv_C.x < LOWMINRATIO*nv_avg.x) && (nv_C.x < HIGHMAXRATIO*nv_avg.x)) fail_nvx_tri = 1;
	//maxnvxcorner = nv0.x;
	//if (nv1.x > nv0.x) maxnvxcorner = nv1.x;
	//if (nv2.x > maxnvxcorner) maxnvxcorner = nv2.x;
	//if ((nv_C.x > maxnvxcorner) && (nv_C.x > HIGHMAXRATIO*nv_avg.x) && (nv_C.x > LOWMINRATIO*nv_avg.x)) fail_nvx_tri = 1;
	//
	//minnvycorner = nv0.y;
	//if (nv1.y < nv0.y) minnvycorner = nv1.y;
	//if (nv2.y < minnvycorner) minnvycorner = nv2.y;
	//if ((nv_C.y < minnvycorner) && (nv_C.y < LOWMINRATIO*nv_avg.y) && (nv_C.y < HIGHMAXRATIO*nv_avg.y)) fail_nvy_tri = 1;
	//maxnvycorner = nv0.y;
	//if (nv1.y > nv0.y) maxnvycorner = nv1.y;
	//if (nv2.y > maxnvycorner) maxnvycorner = nv2.y;
	//if ((nv_C.y > maxnvycorner) && (nv_C.y > HIGHMAXRATIO*nv_avg.y) && (nv_C.y > LOWMINRATIO*nv_avg.y)) fail_nvy_tri = 1;
	//
	//minnvzcorner = nv0.z;
	//if (nv1.z < nv0.z) minnvzcorner = nv1.z;
	//if (nv2.z < minnvzcorner) minnvzcorner = nv2.z;
	//if ((nv_C.z < minnvzcorner) && (nv_C.z < LOWMINRATIO*nv_avg.z) && (nv_C.z < HIGHMAXRATIO*nv_avg.z)) fail_nvz_tri = 1;
	//maxnvzcorner = nv0.z;
	//if (nv1.z > nv0.z) maxnvzcorner = nv1.z;
	//if (nv2.z > maxnvzcorner) maxnvzcorner = nv2.z;
	//if ((nv_C.z > maxnvzcorner) && (nv_C.z > HIGHMAXRATIO*nv_avg.z) && (nv_C.z > LOWMINRATIO*nv_avg.z)) fail_nvz_tri = 1;
	//
	//if (fail_nvx_tri) 
	//{
	//	vx_avg = pVars->mom.x/pVars->mass;
	//	pVertvars0->v.x = vx_avg;
	//	pVertvars1->v.x = vx_avg;
	//	pVertvars2->v.x = vx_avg;
	//	pVerts_c->v.x = vx_avg;
	//} else {
	//	pVerts_c->v.x = nv_C.x / n_C;
	//};

	//if (fail_nvy_tri) 
	//{
	//	vy_avg = pVars->mom.y/pVars->mass;
	//	pVertvars0->v.y = vy_avg;
	//	pVertvars1->v.y = vy_avg;
	//	pVertvars2->v.y = vy_avg;
	//	pVerts_c->v.y = vy_avg;
	//} else {
	//	pVerts_c->v.y = nv_C.y / n_C;
	//};

	//if (fail_nvz_tri) 
	//{
	//	vz_avg = pVars->mom.z/pVars->mass;
	//	pVertvars0->v.z = vz_avg;
	//	pVertvars1->v.z = vz_avg;
	//	pVertvars2->v.z = vz_avg;
	//	pVerts_c->v.z = vz_avg;
	//} else {
	//	pVerts_c->v.z = nv_C.z / n_C;
	//};




	// Note that pVars->heat was already boosted up by ()^2/3 and so was pVertvars0->T.

	//nT_avg = pVars->heat/Areanew;

	//nT[0] = n[0]*pVertvars0->T;
	//nT[1] = n[1]*pVertvars1->T;
	//nT[2] = n[2]*pVertvars2->T;
	//	
	//nT_C = 3.0 * nT_avg - TWOTHIRDS*(nT[0] + nT[1] + nT[2]);

	//fail_nT_tri = 0;
	//minnTcorner = nT0;
	//if (nT1 < nT0) minnTcorner = nT1;
	//if (nT2 < minnTcorner) minnTcorner = nT2;
	//if ((nT_C < minnTcorner) && (nT_C < LOWMINRATIO*nT_avg)) fail_nT_tri = 1;//0.75*nT_avg)) fail_nT_tri = 1;

	//maxnTcorner = nT0;
	//if (nT1 > nT0)  maxnTcorner = nT1;
	//if (nT2 > maxnTcorner) maxnTcorner = nT2;
	//if ((nT_C > maxnTcorner) && (nT_C > HIGHMAXRATIO*nT_avg)) fail_nT_tri = 1;//FOURTHIRDS*nT_avg)) fail_nT_tri = 1;
	//
	//if (fail_nT_tri) 
	//{
	//	T_avg = pVars->heat/pVars->mass;

	//	pVertvars0->T = T_avg;
	//	pVertvars1->T = T_avg;
	//	pVertvars2->T = T_avg;
	//	pVerts_c->T = T_avg;
	//} else {
	//	pVerts_c->T = nT_C / n_C;
	//};

	return (fail_n_tri || fail_nvx_tri || fail_nvy_tri || fail_nvz_tri || fail_nT_tri);
}


bool TestNeighbours(Triangle * pTri1, Triangle * pTri2)
{
	if (pTri1->neighbours[0] == pTri2) return true;
	if (pTri1->neighbours[1] == pTri2) return true;
	if (pTri1->neighbours[2] == pTri2) return true;
	return false;
}
int Triangle::GetCornerIndex(Vertex * pVertex)
{
	if (cornerptr[0] == pVertex) return 0;
	if (cornerptr[1] == pVertex) return 1;
	return 2;
}

real Triangle::ReturnNormalDist(Vertex * pOppVert)
{
	Vector2 u[3];
	real dist;
	if (periodic == 0)
	{
		PopulatePositions(u[0],u[1],u[2]);
	} else {
		MapLeft(u[0],u[1],u[2]);
	};
	if (flags > 0)
	{
		if (pOppVert == HIGH_VERT) return 100.0;
		if (pOppVert == INS_VERT) {
			return min(u[0].modulus(),u[1].modulus())-DEVICE_RADIUS_INSULATOR_OUTER;
		};
	};
	if (pOppVert == cornerptr[0])
	{
		dist = edge_normal[0].dot(u[1]-u[0]);
		return dist;
	};
	if (pOppVert == cornerptr[1])
	{
		dist = edge_normal[1].dot(u[0]-u[1]);
		return dist;
	};
	dist = edge_normal[2].dot(u[0]-u[2]);
	return dist;
}

void Triangle::Return_grad_Area(Vertex *pVertex, real * p_dA_by_dx, real * p_dA_by_dy)
{
	Vector2 u0,u1,u2,opposite,normal,to_ours;
	real rate_of_change_normal;
	static real const r_ins = DEVICE_RADIUS_INSULATOR_OUTER;
//	static real const r_max = HIGH_WEDGE_OUTER_RADIUS;

	// Map to same side as pVertex...
	if (periodic == 0)
	{
		PopulatePositions(u0,u1,u2);
	} else {
		if (pVertex->x < 0.0) {
			MapLeft(u0,u1,u2);
		} else {
			MapRight(u0,u1,u2);
		};
	};
	//if (flags == 0)
	//{
		// Now, rate of change is found by dotting with normal direction.
		// (1,0 ) dot normal = normal.x

		// rate_of_change_normal is 0.5*base
		// since area = 0.5 * normal * base

		if (pVertex == cornerptr[0])
		{
			opposite = u1-u2;
			to_ours = u0-u2;			
		} else {
			if (pVertex == cornerptr[1])
			{
				opposite = u0-u2;
				to_ours = u1-u2;
			} else {
				opposite = u1-u0;
				to_ours = u2-u0;
			};
		};
		rate_of_change_normal = 0.5;// *opposite.modulus();  -- that is wrong. normal isn't normalized so already includes this factor
		normal.x = opposite.y;
		normal.y = -opposite.x;
		if (normal.x*to_ours.x + normal.y*to_ours.y < 0.0) {
			normal.x = -normal.x;
			normal.y = -normal.y;// must face towards our vertex
		}

		*p_dA_by_dx = normal.x*rate_of_change_normal;
		*p_dA_by_dy = normal.y*rate_of_change_normal;
		
		// grad area = 0.5 * | opposite | * grad [normal length] 
		// normal length =  to_ours dot ( normal-direction towards ours so that product is +)
		// so grad normal length = normal-direction 
		// ... If we move right in x when we are to left, we should be reducing normal length. Correct.


	//} else {
	//	if (flags == 1)
	//	{
	//		Vector2 ins0,ins1,rhat,thetahat;
	//		real width, thetadiff,dA_by_dr,dArea_by_dt_plus,dA_by_dt,avgr_outer,r0,r1;

	//		// Now for a wedge we just have to consider both the change due to theta moves and the change due to r moves
	//		
	//		u0.project_to_ins(ins0);
	//		u1.project_to_ins(ins1);

	//		// First let's get thetahat and rhat
	//		rhat = 0.5*(ins0+ins1);
	//		rhat.Normalise();
	//		thetahat.x = rhat.y;
	//		thetahat.y = -rhat.x; // clockwise -- see below

	//		width = fabs((ins0-ins1).x*thetahat.x+(ins0-ins1).y*thetahat.y);
	//			// width... = angle difference times r_ins				
	//		thetadiff = width/DEVICE_RADIUS_INSULATOR_OUTER;
	//		r0 = u0.dot(rhat);
	//		r1 = u1.dot(rhat);
	//		avgr_outer = (r0+r1)*0.5; 
	//		dA_by_dr = thetadiff*0.5*avgr_outer;
	//		dArea_by_dt_plus = 0.5*(avgr_outer*avgr_outer-r_ins*r_ins)/r_ins; 

	//		if (pVertex == cornerptr[0])
	//		{
	//			if (u0.x/u0.y < u1.x/u1.y)
	//			{
	//				dA_by_dt = -dArea_by_dt_plus;
	//			} else {
	//				dA_by_dt = dArea_by_dt_plus;
	//			};
	//		} else {
	//			if (u1.x/u1.y < u0.x/u0.y)
	//			{
	//				dA_by_dt = -dArea_by_dt_plus;
	//			} else {
	//				dA_by_dt = dArea_by_dt_plus;
	//			};
	//		};

	//		*p_dA_by_dx = rhat.x*dA_by_dr + thetahat.x*dA_by_dt;
	//		*p_dA_by_dy = rhat.y*dA_by_dr + thetahat.y*dA_by_dt;
	//	} else {
	//		// outer wedge

	//		Vector2 far0,far1,rhat,thetahat;
	//		real width, thetadiff,dA_by_dr,dArea_by_dt_plus,dA_by_dt,avgr_inner,r0,r1;

	//		// Now for a wedge we just have to consider both the change due to theta moves and the change due to r moves
	//		
	//		u0.project_to_radius(far0,HIGH_WEDGE_OUTER_RADIUS);
	//		u1.project_to_radius(far1,HIGH_WEDGE_OUTER_RADIUS);

	//		// First let's get thetahat and rhat
	//		rhat = 0.5*(far0+far1);
	//		rhat.Normalise();
	//		thetahat.x = rhat.y;
	//		thetahat.y = -rhat.x; // clockwise -- see below

	//		width = fabs((far0-far1).x*thetahat.x+(far0-far1).y*thetahat.y);
	//			// width... = angle difference times r_ins				
	//		thetadiff = width/r_max;
	//		r0 = u0.dot(rhat);
	//		r1 = u1.dot(rhat);
	//		avgr_inner = (r0+r1)*0.5; 
	//		dA_by_dr = thetadiff*0.5*avgr_inner;
	//		dArea_by_dt_plus = 0.5*(r_max*r_max-avgr_inner*avgr_inner)/r_max; 
	//					
	//		if (pVertex == cornerptr[0])
	//		{
	//			if (u0.x/u0.y < u1.x/u1.y)
	//			{
	//				dA_by_dt = -dArea_by_dt_plus;
	//			} else {
	//				dA_by_dt = dArea_by_dt_plus;
	//			};
	//		} else {
	//			if (u1.x/u1.y < u0.x/u0.y)
	//			{
	//				dA_by_dt = -dArea_by_dt_plus;
	//			} else {
	//				dA_by_dt = dArea_by_dt_plus;
	//			};
	//		};

	//		*p_dA_by_dx = rhat.x*dA_by_dr + thetahat.x*dA_by_dt;
	//		*p_dA_by_dy = rhat.y*dA_by_dr + thetahat.y*dA_by_dt;
	//	};
	//};
}




void TriMesh::RecalculateVertexVariables(void)// take angle-weighted averages
{
	// Do for all species.

	Vertex * pVertex = X;
	int lenny;
	real denom,totalweight,weight;
	real dummy;
	Triangle * pTri;
	real area;

	static real const kB_to_3halves = sqrt(kB)*kB;
	real lnLambda;

	// New plan:
	// Instead of angle-weighted average of v we want to take simply
	// sum of momentums / sum of masses
	// so that large area cells do a bigger share of the shoving

	// Seems logical to do likewise for n,T then:
	// let n be just the cluster mass over area
	// let T be just the cluster heat over the mass
	
	// The reason we do this is that it gives the right velocity for advecting stuff instead of giving too much weight to small cells.

	// We could do Voronoi shards if we wished.

	static const Tensor3 Anticlockwise3 (cos(FULLANGLE),-sin(FULLANGLE), 0.0,
														sin(FULLANGLE),cos(FULLANGLE), 0.0,
														0.0, 0.0, 1.0);
	static const Tensor3 Clockwise3 (cos(FULLANGLE),sin(FULLANGLE), 0.0,
														-sin(FULLANGLE),cos(FULLANGLE), 0.0,
														0.0, 0.0, 1.0);

	cellvars iontotal,electotal,neuttotal;
	real r, v_dot_rhat;
	Vector3 rhat;

	for (long iVert = 0; iVert < numVertices; iVert++)
	{
		lenny = pVertex->triangles.len;
		ZeroMemory(&(pVertex->ion),sizeof(vertvars)*3);
		
		ZeroMemory(&iontotal,sizeof(cellvars));
		ZeroMemory(&neuttotal,sizeof(cellvars));
		ZeroMemory(&electotal,sizeof(cellvars));
		
		// Now average B as well because we want it for heat cond/ viscosity
		ZeroMemory(&(pVertex->B),sizeof(Vector3));
		
		area = 0.0;
		
		// BE AWARE THIS KILLS KAPPA
		
		for (int j = 0; j < lenny; j++)
		{
			pTri = (Triangle *)(pVertex->triangles.ptr[j]);			
			
			weight = pTri->ReturnAngle(pVertex); 
			if (pVertex->flags >= 3) weight *= 2.0;// only got half a circle!

			area += pTri->area;
			iontotal.mass += pTri->ion.mass;
			iontotal.heat += pTri->ion.heat;
			neuttotal.mass += pTri->neut.mass;
			neuttotal.heat += pTri->neut.heat;
			electotal.mass += pTri->elec.mass;
			electotal.heat += pTri->elec.heat;
			
			if (
				(pTri->periodic == 0) || 
				((pTri->periodic == 1) && (pVertex == pTri->cornerptr[pTri->GetLeftmostIndex()])) ||
				((pTri->periodic == 2) && (pVertex != pTri->cornerptr[pTri->GetRightmostIndex()]))
			   )
			{
				iontotal.mom += pTri->ion.mom;
				neuttotal.mom += pTri->neut.mom;
				electotal.mom += pTri->elec.mom;

				pVertex->B += weight*pTri->B;
			} else {
				
				iontotal.mom += Clockwise3*pTri->ion.mom;
				neuttotal.mom += Clockwise3*pTri->neut.mom;
				electotal.mom += Clockwise3*pTri->elec.mom;
				
				pVertex->B += weight*(Clockwise3*pTri->B);
			};
		};

		// Now divide :

		pVertex->ion.n = iontotal.mass/area;
		pVertex->ion.T = iontotal.heat/iontotal.mass;
		pVertex->ion.v = iontotal.mom/iontotal.mass;
		pVertex->neut.n = neuttotal.mass/area;
		pVertex->neut.T = neuttotal.heat/neuttotal.mass;
		pVertex->neut.v = neuttotal.mom/neuttotal.mass;
		pVertex->elec.n = electotal.mass/area;
		pVertex->elec.T = electotal.heat/electotal.mass;
		pVertex->elec.v = electotal.mom/electotal.mass;
		
		if (pVertex->flags >= 3)
		{
			// Now kill any inward/outward momentum - assume we do not want it.

			r = sqrt(pVertex->x*pVertex->x+pVertex->y*pVertex->y);
			rhat.x = pVertex->x/r;
			rhat.y = pVertex->y/r;
			rhat.z = 0.0;
			v_dot_rhat = rhat.x*pVertex->ion.v.x + rhat.y*pVertex->ion.v.y;
			pVertex->ion.v -= v_dot_rhat*rhat;
			v_dot_rhat = rhat.x*pVertex->neut.v.x + rhat.y*pVertex->neut.v.y;
			pVertex->neut.v -= v_dot_rhat*rhat;
			v_dot_rhat = rhat.x*pVertex->elec.v.x + rhat.y*pVertex->elec.v.y;
			pVertex->elec.v -= v_dot_rhat*rhat;		
				// It remains to be seen whether that is what we wanted necessarily!
		};
		// Work out nu_ei on vertex:

		// these will be undefined if no ions present:

		//lnLambda = Get_lnLambda(pVertex->ion.n,pVertex->ion.T);
		//pVertex->nu_ei = NU_EI_FACTOR*kB_to_3halves*pVertex->elec.n*lnLambda/
		//														(pVertex->elec.T*sqrt(pVertex->elec.T));

		++pVertex;
	};

	// new way?:

	// First do non-edge vertices;
	// Come back and do edge ones, now assuming that we try to get the mass total on the edge cells
	// kind of thing -- so that it's more likely to give a linear model near the edge.


	// Well, it would be certain.
	
	// I'm not sure about this - come back to when thinking more clearly.
	
	
}



void TriMesh::Recalculate_NuHeart_and_KappaParallel_OnVertices_And_Triangles(short species)
{
	
	static real const one_over_kB = 1.0/kB;
	static real const one_over_kB_cubed = 1.0/(kB*kB*kB);
	static real const over_m_ion = 1.0/m_ion;
	static real const over_m_neutral = 1.0/m_neutral;
	static real const kB_to_3halves = sqrt(kB*kB*kB);
	static real const over_sqrt_m_ion = 1.0/sqrt(m_ion);
	static real const over_sqrt_m_e = 1.0/sqrt(m_e);
	static real const LN_LAMBDA_FLOOR = 4.0;
	static real const half = 0.5;
	static real const expLN_LAMBDA_FLOOR = exp(LN_LAMBDA_FLOOR);
	static real const nu_eiBarconst = //(4.0/3.0)*sqrt(2.0*PI/m_e)*q*q*q*q;
		// don't know in what units but it IS exactly what we already had - see Formulary
									1.0/(3.44e5);

real lnLambda, sqrt_Te, nu_in_visc, nu_en_visc, nu_e_sum, nu_nn_visc,
		nu_ni_visc, nu_eiBar, T_n, T_i, T_e, n_n, n_i, n_e, nu_ii;

	real s_in_visc,s_nn_visc,s_en_visc, electron_thermal, ionneut_thermal;
	real dummy, Tavg, Tion_eV3;
	static real const sqrt_me = sqrt(m_e);

	Vertex * pVertex = X;
	
	// cf CalculateAccelsClass::CalculateCoefficients

	for (long iVertex = 0; iVertex < numVertices; iVertex++)
	{		
		n_i = pVertex->ion.n;
		T_i = pVertex->ion.T;
		T_e = pVertex->elec.T;
		
		if (species == SPECIES_ELECTRON) {
		
			sqrt_Te = sqrt(T_e);
			electron_thermal = sqrt_Te/sqrt_me;

			lnLambda = Get_lnLambda(n_i,T_e);
			Estimate_Ion_Neutral_Cross_sections(pVertex->elec.T*one_over_kB, 
				&dummy, &s_en_visc); // viscosity cross-section
		
			// Get nu_eHeart
			
			nu_eiBar = nu_eiBarconst*kB_to_3halves*n_i*lnLambda/(pVertex->elec.T*sqrt_Te);

			nu_en_visc = pVertex->neut.n*s_en_visc*electron_thermal;
			
			pVertex->elec.nu_Heart = 1.87*nu_eiBar + nu_en_visc; // note, used visc
			
			// store ratio
			pVertex->epsilon = nu_eiBar/pVertex->elec.nu_Heart;

			// Get kappa e parallel

			if (pVertex->elec.n == 0.0) {
				pVertex->elec.kappa = 0.0;
			} else {
				pVertex->elec.kappa = 2.5*pVertex->elec.n*pVertex->elec.T/(m_e*pVertex->elec.nu_Heart);
			};
			// Compute Upsilon on the fly.

		};
		if (species == SPECIES_ION) {

			lnLambda = Get_lnLambda_ion(n_i,T_i);
			// Get viscosity cross-sections with neutrals:
		
			Estimate_Ion_Neutral_Cross_sections(pVertex->ion.T*one_over_kB,
				&dummy, &s_in_visc);
			s_nn_visc =Estimate_Neutral_Neutral_Viscosity_Cross_section(pVertex->neut.T*one_over_kB);

			ionneut_thermal = sqrt(T_i/m_ion
				+pVertex->neut.T/m_n);
			
			nu_in_visc = pVertex->neut.n*s_in_visc*ionneut_thermal;

			nu_nn_visc = pVertex->neut.n*s_nn_visc*sqrt(pVertex->neut.T/m_n);
			// Now not so sure about rel temp thing
			// Why not factor of 2 involved?
			
			nu_ni_visc = n_i*s_in_visc*ionneut_thermal;
			
			nu_ii = NU_II_FACTOR*n_i*lnLambda/(T_i*sqrt(T_i));

			pVertex->ion.nu_Heart = 0.75*nu_in_visc
					+ 0.8*nu_ii-0.25*nu_in_visc*nu_ni_visc/(3.0*nu_ni_visc+nu_nn_visc);

			
			if (pVertex->ion.n == 0.0){
				pVertex->ion.kappa = 0.0;
			} else {
				pVertex->ion.kappa = 2.5*n_i*T_i/(m_i*pVertex->ion.nu_Heart);
			};

		};
		if (species == SPECIES_NEUTRAL) {
			
			Estimate_Ion_Neutral_Cross_sections(T_i*one_over_kB,
				&dummy, &s_in_visc);
			// e-n does not feature.

			ionneut_thermal = sqrt(T_i/m_ion
				+pVertex->neut.T/m_n);
			
			s_nn_visc = Estimate_Neutral_Neutral_Viscosity_Cross_section(pVertex->neut.T*one_over_kB);

			nu_ni_visc = pVertex->ion.n*s_in_visc*ionneut_thermal;
			
			nu_nn_visc = pVertex->neut.n*s_nn_visc*sqrt(pVertex->neut.T/m_n);

			pVertex->neut.nu_Heart = 0.75*nu_ni_visc + 0.25*nu_nn_visc;
			
			pVertex->neut.kappa = NEUTRAL_KAPPA_FACTOR*pVertex->neut.n*pVertex->neut.T/(m_n*pVertex->neut.nu_Heart);
			// NEUTRAL_KAPPA_FACTOR == 10 , then
			// much larger than traditional ---
			// MUST test 2x different cases.
		};
			
	};

	// Now get nu and kappa on triangles also. We average over quadrilaterals.
	Triangle * pTri = T;
	long iTri;

	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		T_e = pTri->elec.heat/pTri->elec.mass;
		T_i = pTri->ion.heat/pTri->ion.mass;
		n_e = pTri->elec.mass/pTri->area;
		n_i = pTri->ion.mass/pTri->area;
		T_n = pTri->neut.heat/pTri->neut.mass;
		n_n = pTri->neut.mass/pTri->area;

		if (species == SPECIES_ELECTRON) {
		
			sqrt_Te = sqrt(T_e);
			lnLambda = Get_lnLambda(n_i,T_e);
			Estimate_Ion_Neutral_Cross_sections(T_e*one_over_kB, 
				&dummy, &s_en_visc); // viscosity cross-section
		
			// Get nu_eHeart
			
			nu_eiBar = nu_eiBarconst*kB_to_3halves*n_i*lnLambda/(T_e*sqrt_Te);

			nu_en_visc = n_n*s_en_visc*electron_thermal;
			
			pTri->scratch[4] = 1.87*nu_eiBar + nu_en_visc; // note, used visc
			// nu heart

			pTri->scratch[6] = nu_eiBar/pTri->scratch[4];

			// Get kappa e parallel

			if (pTri->elec.mass == 0.0) {
				pTri->scratch[5] = 0.0; // use scratch for kappa parallel, nu_Heat
			} else {
				pTri->scratch[5] = 2.5*n_e*T_e/(m_e*pTri->scratch[4]);
			};
			// Compute Upsilon on the fly.
		};
		if (species == SPECIES_ION)
		{

			lnLambda = Get_lnLambda_ion(n_i,T_i);
			// Get viscosity cross-sections with neutrals:
		
			Estimate_Ion_Neutral_Cross_sections(T_i*one_over_kB,
				&dummy, &s_in_visc);
			s_nn_visc =Estimate_Neutral_Neutral_Viscosity_Cross_section(T_n*one_over_kB);

			ionneut_thermal = sqrt(T_i/m_ion+T_n/m_n);
			
			nu_in_visc = n_n*s_in_visc*ionneut_thermal;

			nu_nn_visc = n_n*s_nn_visc*sqrt(T_n/m_n);

			// Now not so sure about rel temp thing
			// Why not factor of 2 involved?
			
			nu_ni_visc = n_i*s_in_visc*ionneut_thermal;
			
			nu_ii = NU_II_FACTOR*n_i*lnLambda/(T_i*sqrt(T_i));

			pTri->scratch[2] = 0.75*nu_in_visc
					+ 0.8*nu_ii-0.25*nu_in_visc*nu_ni_visc/(3.0*nu_ni_visc+nu_nn_visc);
			// nu heart

			if (pTri->ion.mass == 0.0){
				pTri->scratch[3] = 0.0;
			} else {
				pTri->scratch[3] = 2.5*n_i*T_i/(m_i*pTri->scratch[2]);
			};
		};
		if (species == SPECIES_NEUTRAL) {
			
			Estimate_Ion_Neutral_Cross_sections(T_i*one_over_kB,
				&dummy, &s_in_visc);
			s_nn_visc = Estimate_Neutral_Neutral_Viscosity_Cross_section(T_n*one_over_kB);

			// e-n does not feature.

			ionneut_thermal = sqrt(T_i/m_ion+T_n/m_n);
			
			nu_ni_visc = n_i*s_in_visc*ionneut_thermal;
			
			nu_nn_visc = n_n*s_nn_visc*sqrt(T_n/m_n);

			pTri->scratch[0] = 0.75*nu_ni_visc + 0.25*nu_nn_visc; // nu_nheart
			
			if (pTri->neut.mass == 0.0) {
				pTri->scratch[1] = 0.0;
			} else {
				pTri->scratch[1] = NEUTRAL_KAPPA_FACTOR*n_n*T_n/(m_n*pTri->scratch[0]);
			};

			// NEUTRAL_KAPPA_FACTOR == 10 , then
			// much larger than traditional ---
			// MUST test 2x different cases.
		};


		
		++pTri;
	};
		
		// SPEED-UPS:
		// Could we just store this stuff and quickly recalculate kappa on the fly????
		// Store log lambda, sigma_visc and leave dirty for most iterations, or until T changes by 20%.
		// ^^ good plan - log(exp( costs > sqrt and probably EstimateCrossSections costs the same as sqrt.

		// log lambda in particular will hardly change.

}

void TriMesh::Recalculate_KappaOnVerticesOld()
{
	real lnLambda, sqrt_Te, nu_ei, nu_en, nu_e_sum;
	real sigma_in_visc,sigma_nn_visc,sigma_en_visc;
	real dummy, Tavg, Tion_eV3;

	static real const one_over_kB = 1.0/kB;
	static real const one_over_kB_cubed = 1.0/(kB*kB*kB);
	static real const over_m_ion = 1.0/m_ion;
	static real const over_m_neutral = 1.0/m_neutral;
	static real const kB_to_3halves = sqrt(kB*kB*kB);
	static real const over_sqrt_m_ion = 1.0/sqrt(m_ion);
	static real const over_sqrt_m_e = 1.0/sqrt(m_e);
	static real const LN_LAMBDA_FLOOR = 4.0;
	static real const half = 0.5;
	static real const expLN_LAMBDA_FLOOR = exp(LN_LAMBDA_FLOOR);
	
	Vertex * pVertex = X;
	
	for (long iVertex = 0; iVertex < numVertices; iVertex++)
	{		
		// ln Lambda depends on T_ion; 
		// we do not here introduce the transition to the next
		// set of values, yet.
		Tavg = pVertex->ion.T;
		Tion_eV3 = Tavg*Tavg*Tavg*one_over_kB_cubed;
		lnLambda = 23.0 - 0.5*log(pVertex->ion.n*Tion_eV3);
		lnLambda = (log(exp((lnLambda-LN_LAMBDA_FLOOR)*two+LN_LAMBDA_FLOOR)+expLN_LAMBDA_FLOOR)-LN_LAMBDA_FLOOR)*half+LN_LAMBDA_FLOOR;

		// Get viscosity cross-sections with neutrals:
		Estimate_Ion_Neutral_Cross_sections(pVertex->ion.T*one_over_kB, &dummy, &sigma_in_visc);
		Estimate_Ion_Neutral_Cross_sections(pVertex->elec.T*one_over_kB, &dummy, &sigma_en_visc);
		sigma_nn_visc = Estimate_Neutral_Neutral_Viscosity_Cross_section(pVertex->neut.T*one_over_kB);
				
		pVertex->ion.kappa = ION_KAPPA_FACTOR*(pVertex->ion.n)*(pVertex->ion.T)*sqrt(pVertex->ion.T)/
						(m_ion*(
						   pVertex->neut.n*sigma_in_visc*(pVertex->ion.T)*over_sqrt_m_ion  // = nu_ion_neutral_visc * sqrt(T)
						+
						   NU_II_FACTOR*pVertex->ion.n*lnLambda*kB_to_3halves/(pVertex->ion.T) // = nu_ion_ion * sqrt(T)
						));

		if (pVertex->ion.n == 0.0)  pVertex->ion.kappa = 0.0;
		
		// SPEED-UPS:
		// Could we just store this stuff and quickly recalculate kappa on the fly????
		// Store log lambda, sigma_visc and leave dirty for most iterations or until T changes by 20%.
		// ^^ good plan - log(exp( costs > sqrt and probably EstimateCrossSections costs the same as sqrt.

		// zsqrt might help speed up also. But have had trouble getting it faster than sqrt.
		
		// &&, how to put on GPU? We squeeze within small domains but then have to pass data because boundary T is changing next door.
		
				
		pVertex->neut.kappa = THIRD*pVertex->neut.n*pVertex->neut.T/
					(m_neutral*(
							pVertex->ion.n*sigma_in_visc*sqrt(pVertex->ion.T*over_m_ion)
						+
							pVertex->neut.n*sigma_nn_visc*sqrt(pVertex->neut.T*over_m_neutral)
				));

		sqrt_Te = sqrt(pVertex->elec.T);
		nu_ei = NU_EI_FACTOR*pVertex->ion.n*kB_to_3halves*lnLambda/((pVertex->elec.T)*sqrt_Te);
		nu_en = pVertex->neut.n*sigma_en_visc*sqrt_Te*over_sqrt_m_e;
		// Note that we use a different nu_en in kappa than in dyn_conduct
		// and yet we use the same nu_ei. That seems bogus.

		// assume nu_ee = 0.87 nu_ei :
		nu_e_sum = nu_ei*1.87 + nu_en;  

		// Here we have factor 2.5 but it is multiplied by 2/3 in practice
		// NOTE BENE
		// So we get ELECTRON_KAPPA_FACTOR = 5/3
		pVertex->elec.kappa = ELECTRON_KAPPA_FACTOR*(pVertex->elec.n)*(pVertex->elec.T)/(m_e*nu_e_sum);
			
		//pVertex->kappa_e.xx = kappa_factor*(nu_e_sum_sq + omega.x*omega.x);
		//pVertex->kappa_e.xy = kappa_factor*omega.x*omega.y;
		
		// Now here is a thing. We have a scalar electron kappa in the vertvars object
		// That is the independent kappa. Then we have to add on Hall and parallel effects separately, which we shall do.

#ifdef debug_kappa

		if (pVertex->ion.kappa != pVertex->ion.kappa)
		{
			printf("\npVertex->kappa != pVertex->kappa\npVertex->ion.n = %1.10E\n",pVertex->ion.n);
			getch();
		};
#endif

		++pVertex;
	};
}
void TriMesh::RecalculateDisplacementSD()
{
	real lnLambda, sqrt_Te, nu_ei, nu_en, nu_e_sum;
	real sigma_in_visc,sigma_nn_visc,sigma_en_visc;
	real dummy, Tavg, Tion_eV3;

	real sigma_in_MT, sigma_en_MT, sigma_nn_MT,sqrt_neut_T, nu_ion_sum, nu_neut_sum,sqrt_ion_T, variance;

	// THIS ROUTINE NOW SETS kappa == DISPLACEMENT S.D. SO SHOULD BE USED WITH CAUTION

	static real const one_over_kB = 1.0/kB;
	static real const one_over_kB_cubed = 1.0/(kB*kB*kB);
	static real const over_m_ion = 1.0/m_ion;
	static real const over_m_neutral = 1.0/m_neutral;
	static real const kB_to_3halves = sqrt(kB*kB*kB);
	static real const over_sqrt_m_ion = 1.0/sqrt(m_ion);
	static real const over_sqrt_m_e = 1.0/sqrt(m_e);
	static real const over_sqrt_m_neutral = 1.0/sqrt(m_neutral);
	static real const LN_LAMBDA_FLOOR = 4.0;
	static real const half = 0.5;
	static real const expLN_LAMBDA_FLOOR = exp(LN_LAMBDA_FLOOR);
	
	Vertex * pVertex = X;
	
	for (long iVertex = 0; iVertex < numVertices; iVertex++)
	{		

		sqrt_ion_T = sqrt(pVertex->ion.T);
		sqrt_Te= sqrt(pVertex->elec.T);
		sqrt_neut_T= sqrt(pVertex->neut.T);

		Tavg = pVertex->ion.T;
		Tion_eV3 = Tavg*Tavg*Tavg*one_over_kB_cubed;
		lnLambda = 23.0 - 0.5*log(pVertex->ion.n*Tion_eV3);
		lnLambda = (log(exp((lnLambda-LN_LAMBDA_FLOOR)*two+LN_LAMBDA_FLOOR)+expLN_LAMBDA_FLOOR)-LN_LAMBDA_FLOOR)*half+LN_LAMBDA_FLOOR;

		// would be better to make a precalc interpolation for lnLambda !



			// want to get hold of nu_ion_sum and MT is what we'll use 
			// so we can do our hypo-diffusion SD:

		Estimate_Ion_Neutral_Cross_sections(pVertex->ion.T*one_over_kB, &sigma_in_MT, &sigma_in_visc);
		Estimate_Ion_Neutral_Cross_sections(pVertex->elec.T*one_over_kB, &sigma_en_MT, &sigma_en_visc);
		sigma_nn_visc = Estimate_Neutral_Neutral_Viscosity_Cross_section(pVertex->neut.T*one_over_kB);
		
			// looks like we need to guess neutral-neutral cross-section in case of not viscosity
			// need to go get another table from Oak Ridge.
		sigma_nn_MT = sigma_nn_visc*(sigma_in_MT/sigma_in_visc); // stand-in for now
				
		nu_ion_sum = pVertex->neut.n*sigma_in_MT*sqrt_ion_T*over_sqrt_m_ion 
								+ NU_II_FACTOR*pVertex->ion.n*lnLambda*kB_to_3halves/(pVertex->ion.T*sqrt_ion_T)
								+ NU_EI_FACTOR*pVertex->elec.n*lnLambda*kB_to_3halves/((pVertex->elec.T)*sqrt_Te);
			// check in notes that it's same collision freq i-e as e-i
			
		variance = (pVertex->ion.T/(m_ion*nu_ion_sum*nu_ion_sum))*
				(2.0*exp(-nu_ion_sum*h)-2.0+2.0*h*nu_ion_sum);

		pVertex->ion.kappa = sqrt(variance); // Displacement SD
		
		nu_neut_sum = pVertex->ion.n*sigma_in_MT*sqrt_ion_T*over_sqrt_m_ion 
								+ pVertex->neut.n*sigma_nn_MT*sqrt_neut_T*over_sqrt_m_neutral
								+ pVertex->elec.n*sigma_en_MT*sqrt_Te*over_sqrt_m_e;

		variance = (pVertex->neut.T/(m_neutral*nu_neut_sum*nu_neut_sum))*
				(2.0*exp(-nu_neut_sum*h)-2.0+2.0*h*nu_neut_sum);

		pVertex->neut.kappa = sqrt(variance); // Displacement SD
					
		nu_e_sum = pVertex->neut.n*sigma_en_MT*sqrt_Te*over_sqrt_m_e 
							+ 1.87 * NU_EI_FACTOR*pVertex->elec.n*lnLambda*kB_to_3halves/((pVertex->elec.T)*sqrt_Te);
			

		// Would be wise to add something here to avoid a call to exp in case that 
		// basically exp(-x) = 1 - x , or exp(-x) = 1-x + x^2 , or exp(-x) = 0
		// we can neglect from displacement anything < 1.0e-7 

		variance = (pVertex->elec.T/(m_e*nu_e_sum*nu_e_sum))*
				(2.0*exp(-nu_e_sum*h)-2.0+2.0*h*nu_e_sum);
		
		pVertex->elec.kappa = sqrt(variance); // Displacement SD

		++pVertex;
	};
}














void TriMesh::RecalculateEdgeNormals(bool bNormalise)
{
	Triangle * pTri = T;
	for (long iTri = 0; iTri < numTriangles; iTri++)
	{
		pTri->RecalculateEdgeNormalVectors(bNormalise);
		pTri++;
	};    
}
void TriMesh::SurveyCellMassStats(real * pAvgMass, real * pMassSD, real * pMinMass, real * pMaxMass, int * piMin)
{
	long iTri;
	Triangle * pTri = T;
	real mass_sum = 0.0;
	real mass_sum_not_outer = 0.0;
	real mass_sq_sum = 0.0;
	real min_mass = 1.0e100;
	real max_mass = 0.0;
	real mass;
	int iMin = 0;
	long numTriUsed = 0;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		mass = pTri->ion.mass + pTri->neut.mass;
		mass_sum += mass;
		if (pTri->flags < 2)
		{
			mass_sum_not_outer += mass;
			mass_sq_sum += mass*mass;
			if (min_mass > mass){
				iMin = iTri;
				min_mass = mass;
			};
			if (max_mass < mass) max_mass = mass;
			numTriUsed++;
		};
		++pTri;
	};
	*pAvgMass = mass_sum/(real)numTriangles;
	real EMass = mass_sum_not_outer/(real)numTriUsed;
	*pMassSD = sqrt(mass_sq_sum/(real)numTriUsed - EMass*EMass);
	*pMinMass = min_mass;
	*pMaxMass = max_mass;
	*piMin = iMin;
}


void TriMesh::VertexJump(int iVertex, int iTri)
{
	Vertex * pVertex = X + iVertex;
	Triangle * pTri = T + iTri;
	Triangle * pTri2;
	Vector2 u;
	int i;
	for (i = 0; i < pVertex->triangles.len; i++)
	{
		pTri2 = (Triangle *)(pVertex->triangles.ptr[i]);
		pTri2->indicator = 1; // do not send stuff here from now on! it may even have moved!
	};

	// leave new split-up tri as pVertex->triangles.ptr[0], by passing it as pTriContain :
	Disconnect(pVertex, pTri ); 
	
	// May jump into base tri interior but not yet on to boundary:
	pTri->ReturnCentre(&u,pTri->cornerptr[0]);
	if (u.x/u.y > GRADIENT_X_PER_Y)
		u = Anticlockwise*u;
	if (u.x/u.y < -GRADIENT_X_PER_Y)
		u = Clockwise*u;						// take u to be within the tranche.

	pVertex->x = u.x;
	pVertex->y = u.y;

	while (Disconnected.len > 0)
		this->ReconnectLastPointInDiscoArray();
}

int TriMesh::SkipVertices() // return number of verts that skip
{
	long iSkipped = 0;
	Triangle * pTri, *pTriTemp;
	long iTri;
	Vertex * pApex , *pNeigh1, *pNeigh2,* pVertex;
	Vector2 u0,u1,u2,perp;
	int numEdge,i;
	bool skip;
	Vector2 newpos,avg;
	Triangle *pTri2,* pTriNeigh1, *pTriNeigh2, *pTriCopy, *pTriNeigh;


			// NOTE: IF WE EVER MAKE IT SKIP VERTICES OFF EDGE, NEED TO REMEMBER
			// TO SET THE VERTEX FLAG FIRST IF WE THEN GO ON TO CALL RECONNECT

	int i1,i2,iCorner;
	real gradient, gradient1,gradient2, height, ratio_h_over_w,n;

	const real MAXFACTOR_EDGEDELTA = 4.0; // max delta is sqrt(4 m_initial / n)
	pTri = T;
	real MassAverage = 0.0;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		MassAverage += pTri->ion.mass + pTri->neut.mass;
		++pTri;
	};
	MassAverage /= (real)(numTriangles+numEdgeVerts); // this should remain constant throughout.

	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		skip = false;
		if ((pTri->flags == 6) || (pTri->flags == 24)){

			pTri->RecalculateEdgeNormalVectors(true);
	
			// for each base triangle, consider
			// i) is apex above base azimuthally?
			// ii) is apex angle > 90 degrees
			// iii) are there exactly 2 edge neighbours?
			// iv) is base too wide for mass here?
			
			if (pTri->periodic == 0)
			{
				// i) is apex above base azimuthally?

				iCorner = 0;
				while (pTri->cornerptr[iCorner]->flags >= 3) iCorner++;
				i1 = iCorner+1; if (i1 == 3) i1 = 0;
				i2 = i1+1; if (i2 == 3) i2 = 0;
				
				pApex = pTri->cornerptr[iCorner];
				pNeigh1 = pTri->cornerptr[i1];
				pNeigh2 = pTri->cornerptr[i2];

				pApex->PopulatePosition(u0);
				pNeigh1->PopulatePosition(u1);
				pNeigh2->PopulatePosition(u2);

				gradient1 = pNeigh1->x/pNeigh1->y;
				gradient2 = pNeigh2->x/pNeigh2->y;
				gradient = pApex->x/pApex->y;
				if ((gradient1-gradient)*(gradient2-gradient) < 0.0)
				{

					// ii) is apex angle > 90 degrees?
					// Just ask is height less than half width:
					perp.x = u2.y - u1.y;
					perp.y = u1.x - u2.x;
					// now (u0-u2).dot(perp) = (u0-u2) dot normalised outward (+ or - not known) times width
					ratio_h_over_w = fabs(((u0-u2).dot(perp))/(perp.dot(perp)));

					if (ratio_h_over_w < 0.5) {

						// iii) are there exactly 2 edge neighbours?
						numEdge = 0;
						for (i = 0; i < pApex->neighbours.len; i++)
							if ((X+(pApex->neighbours.ptr[i]))->flags >= 3) numEdge++;
						if (numEdge == 2) {

							// iv) is base too wide for mass here?
							
							n = (pTri->ion.mass+pTri->neut.mass)/pTri->area;

							// CRITERION: delta < sqrt (4 m_initial / n) -- see SKIPS.lyx
							if (perp.dot(perp) > MAXFACTOR_EDGEDELTA * MassAverage / n) 
								skip = true;							
							// MassAverage should not change much because total mass is not changing
							// and number of triangles only changes by sending them to base.

						};
					};
				};
			} else {
				// periodic case.
				
			};

			if (skip)
			{
				// change mesh structure so that this is a base vertex. 
				
				pTriNeigh1 = pTri->neighbours[i1];
				pTriNeigh2 = pTri->neighbours[i2];
				
				// Scrap the base triangle :
				// reassign their neighbour to be themselves -- so that the rest of the mesh does not send searches to pTri
				for (i = 0; i < 3; i++)
				{
					if (pTriNeigh1->neighbours[i] == pTri) 
						pTriNeigh1->neighbours[i] = pTriNeigh1;
					if (pTriNeigh2->neighbours[i] == pTri) 
						pTriNeigh2->neighbours[i] = pTriNeigh2;
				}

				// remove from triangle lists of its own vertices:
				for (i = 0; i < 3; i++)
					pTri->cornerptr[i]->triangles.remove((Proto *)pTri);
				
				// The other triangles do not need to change corners at all
				// but we do need to set triangle flags:
				
				// Set flag on vertex and tris:
				pApex->flags = pNeigh1->flags;

				for (i = 0; i < pApex->triangles.len; i++)
				{
					pTriTemp = (Triangle *)(pApex->triangles.ptr[i]);
					pTriTemp->flags = 2; 
					if (pApex->flags == 4) pTriTemp->flags = 8; // ?!
				};

				pTriNeigh1->flags = 6;			pTriNeigh2->flags = 6;
				if (pApex->flags == 4) {
					pTriNeigh1->flags = 24;			pTriNeigh2->flags = 24;
				};
									
				// Reset vertex neighbours: its own are same but now, neighs do not see each other:
				pNeigh1->neighbours.remove(pNeigh2-X);
				pNeigh2->neighbours.remove(pNeigh1-X);
				// We haven't re-done sequence of neighbours but at least now they are the ones listed.

				// if it was a periodic pTri then it's possible we crossed PB by sending to average of u1,u2
				// instead of just projecting down. In this case we changed periodic flags of many triangles.
				// ... need to work on periodic case more generally.

				//  Reinterpolate A,phi at new position --- use old position and value in this though.
				avg = 0.5*(u1+u2);
				if (pTri->flags == 6){
					avg.project_to_ins(newpos);
				} else {
					avg.project_to_radius(newpos,Outermost_r_achieved);
				};

				// easiest way actually: just let A,phi be the average of the two neigh values.
				pApex->A = 0.5*(pNeigh1->A + pNeigh2->A);
				pApex->phi = 0.5*(pNeigh1->phi + pNeigh2->phi);

				pApex->x = newpos.x;
				pApex->y = newpos.y;

				// =======================================================
				// now what? We don't want spare tri to be included in numTriangles
				// so we have to do a swap with the last element.
				// =======================================================

				pTriCopy = T+numTriangles-1;
				// 1. Copy over the data, structural and otherwise
				
				memcpy(pTri,pTriCopy,sizeof(Triangle));
				for (i = 0; i < 3; i++)
				{
					// if neighbour is self, ensure neighbour is still self:
					if (pTriCopy->neighbours[i] == pTriCopy)
						pTri->neighbours[i] = pTri;
				}
				// "delete all dynamic arrays on pTriCopy once they have been copied." -- but we just copied pointers.
				// and, there aren't even any anyway.
				
				// 2. all the other triangles and vertices that looked at pTriCopy have to now be looking at its new location.
				
				// triangles:
				for (i = 0; i < 3; i++)
				{
					pTriNeigh = pTri->neighbours[i];
					for (int ii = 0; ii < 3; ii++)
					{
						if (pTriNeigh->neighbours[ii] == pTriCopy) pTriNeigh->neighbours[ii] = pTri;
					};
				};
				
				// vertices:
				for (i = 0; i < 3; i++)
				{
					pVertex = pTri->cornerptr[i];
					for (int ii = 0; ii < pVertex->triangles.len; ii++)
					{
						pTri2 = (Triangle *)(pVertex->triangles.ptr[ii]);
						if (pTri2 == pTriCopy) pVertex->triangles.ptr[ii] = pTri;
					};
				};
				
				// assign spare triangle index and change counts:
				//pApex->iTriSpare = numTriangles-1;	
				// INDEX NEVER USED FOR ANYTHING
				this->numTriangles--;
				this->numEdgeVerts++;

				iSkipped++;
			};
		}
		++pTri;
	};
	
	return iSkipped;
}

int TriMesh::JumpVertices() // return number of verts that jump
{

	// We want to move from the least efficient place (minimum adjacent mass) to the most efficient
	// (highest mass triangle) 

	// Use existing functions to do disconnect and reconnect
	
	// Criterion:
	// Make a move if 1/3 mass of destination will be greater than 1/(N-2) of total nearby mass
	// or perhaps should be a little more conservative than that:
	// add in factor 1.618 or 1.4

	static const real CONSERVATIVE_FACTOR = 1.25; 

	// Procedure: find which vertices have lowest values of "total nearby / N-2". Create a sorted list of indices. Only bother with say 10% of the list -
	// say we have 10000 points; once we have 1000 entries we can start chucking away anything that exceeds entry 1000.

	int listmax = numVertices/20;

	// Likewise find which triangles have the greatest mass (ion + neutral).

	int * index_verts;
	int * index_tris;
	real * pUseful;
	real * pNeedful;
	int i,j,iIndexVert,iIndexTri;
	long iTri,iVertex,iPopulated;
	Vertex * pVertex;
	Triangle * pTri;
	int iLen;
	real Usefulness,Needfulness;

	index_verts = new int[listmax];
	index_tris = new int[listmax];
	pUseful = new real[listmax];
	pNeedful = new real[listmax]; // Note that we should try still for a speedup by changing arrays to static in general.
		
	// Another way to think: usefulness is the difference made to sum of squares if it is removed.
	// Ask sum of squared masses now and guess what it would be if we removed it. ???
	
	iPopulated = 0;
	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		iLen = pVertex->triangles.len;
		Usefulness = 0.0;
		for (i = 0; i < iLen; i++)
		{
			pTri = (Triangle *)(pVertex->triangles.ptr[i]);
			Usefulness += pTri->ion.mass + pTri->neut.mass;
		};
		if (pVertex->flags < 3) {
			Usefulness /= (real)(iLen-2); // average of masses after removing vertex
		} else {
			Usefulness /= (real)(iLen-1); // for an edge vertex it only reduces # triangles by 1. 
			// [ Does Disconnect know that? ]
			// But only include edge vertex when we also add in a criterion for sqrt(n) delta at edge;
			// maybe move only if edge criterion is still satisfied
			Usefulness = 1.0e100;

			// NOTE: IF WE EVER MAKE IT JUMP VERTICES OFF EDGE, NEED TO REMEMBER
			// TO SET THE VERTEX FLAG BEFORE WE CALL RECONNECT
		};

		// So the question was:
		// Can we move, and still have it that the average mass where we go will STILL be more than
		// the average mass in the triangles we then leave behind?
				
		// Now place this index into our list:
		if (iVertex < listmax) {
			// it will definitely go somewhere
			i = 0;
			while ((i < iPopulated) && (pUseful[i] < Usefulness)) // want least useful as element 0
				i++;
			if (i < iPopulated)
			{
				// shunt up the rest
				for (j = iPopulated-1; j > i; j--)
				{
					index_verts[j] = index_verts[j-1];
					pUseful[j] = pUseful[j-1];
				};
			};
			index_verts[i] = iVertex;
			pUseful[i] = Usefulness;
			iPopulated++;
		} else {
			if (Usefulness < pUseful[listmax-1]) // otherwise do nothing
			{
				i = listmax-1;
				while ((i >= 0) && (Usefulness < pUseful[i])) i--;
				i++;
				// i is now the element it should replace

				for (j = listmax-1; j > i; j--)
				{
					index_verts[j] = index_verts[j-1];
					pUseful[j] = pUseful[j-1];
				};
				index_verts[i] = iVertex;
				pUseful[i] = Usefulness;
			};
		};
		++pVertex;
	};
	

	// Now compare the least useful vertex with most needful triangle
	iPopulated = 0;
	pTri = T;
	
	for (iTri = 0; iTri < numTriangles; iTri++) // numTriangles not to include any spare ones.
	{
		Needfulness = THIRD*(pTri->ion.mass + pTri->neut.mass);
		
		// Now place this index into our list:
		if (iPopulated < listmax) {
			// it will definitely go somewhere
			i = 0;
			while ((i < iPopulated) && (pNeedful[i] > Needfulness))
				i++;
			if (i < iPopulated)
			{
				// shunt up the rest
				for (j = iPopulated-1; j > i; j--)
				{
					index_tris[j] = index_tris[j-1];
					pNeedful[j] = pNeedful[j-1];
				};
			};
			index_tris[i] = iTri;
			pNeedful[i] = Needfulness;
			iPopulated++;
		} else {
			if (Needfulness > pNeedful[listmax-1]) // otherwise do nothing
			{
				i = listmax-1;
				while ((i >= 0) && (Needfulness > pNeedful[i])) i--;
				i++;
				// i is now the element it should replace

				// Faster search would start from previous i since neighbouring triangles may have similar mass.
				for (j = listmax-1; j > i; j--)
				{
					index_tris[j] = index_tris[j-1];
					pNeedful[j] = pNeedful[j-1];
				};
				index_tris[i] = iTri;
				pNeedful[i] = Needfulness;
			};
		};
		++pTri;
	};
	
	// Note that moving should void the possibility of moving a neighbour vertex from its position.
	// Therefore we should have to iterate again if that tries to happen.
	
	// reset vertex flags

	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		pVertex->iVolley = 0;
		++pVertex;
	};
	pTri = T ;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		pTri->indicator = 0;
		++pTri;
	};

	iIndexVert = 0;
	iIndexTri = 0;
	int iJumped = 0;
	bool worth_testing_further = true;
	do
	{
		// find next viable vertex to compare
		while ((iIndexVert < listmax) && (X[index_verts[iIndexVert]].iVolley > 0))
			iIndexVert++;

		if (iIndexVert < listmax) {
			// make the comparison
			if (pUseful[iIndexVert]*CONSERVATIVE_FACTOR < pNeedful[iIndexTri]) // mass is still less even with vertex removed - we imagine
			{
				pVertex = X+index_verts[iIndexVert];
				pTri = T + index_tris[iIndexTri];
				
				// there are some other things that could rule out a jump - e.g. if this tri is actually a neighbour of this vertex...
				// not something we would hope for but we rule it out here anyway:
				
				if (pTri->indicator == 0) {
					if ((pVertex != pTri->cornerptr[0]) &&
						(pVertex != pTri->cornerptr[1]) &&
						(pVertex != pTri->cornerptr[2]))
					{
						// rule out doing any jump for neighbours around where it begins from:
						for (i = 0; i < pVertex->neighbours.len; i++)
						{
							(X+pVertex->neighbours.ptr[i])->iVolley++; 
						};			
						this->VertexJump(index_verts[iIndexVert], index_tris[iIndexTri]);
						// change this so that we set indicator on every triangle that is of jumped vertex originally - do not want smth else jumping in there--?!
						iJumped++;
						// Previously the neighbour hold came here, after it arrived somewhere. Doesn't matter - the jump is in the dest system.
					};					
					iIndexVert++; // move on for next time.
				}
				iIndexTri++;

			} else {
				// the most needful tri is not a better place for X[index_verts[iIndexVert]].
				worth_testing_further = 0;
			};
		} else {
			worth_testing_further = 0; // end of viable vertex list - nothing further we can do
		};
	} while (worth_testing_further);

	// reset vertex flags
	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		pVertex->iVolley = 0;
		++pVertex;
	};

	delete[] index_verts;
	delete[] index_tris;
	delete[] pUseful;
	delete[] pNeedful;

	pTri = T ;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		pTri->indicator = 0; // was never really used? didn't look to understand.
		++pTri;
	};

	return iJumped;
}

/*int TriMesh::JumpVerticesNew(TriMesh * pDestMesh) // return number of verts that jump
{

	// We want to move from the least efficient place (minimum adjacent mass) to the most efficient
	// (highest mass triangle) 

	// Use existing functions to do disconnect and reconnect
	
	// Criterion:
	// Make a move if 1/3 mass of destination will be greater than 1/(N-2) of total nearby mass
	// or perhaps should be a little more conservative than that:
	// add in factor 1.618 or 1.4

	// New way:
	// Compare to 1/4 mass of pair of triangles. More desirable.

	// Do not jump points off the insulator .... creates too many issues .... ? ...
	// ... but know Disconnect MUST handle anyway

	// Alternatively we could put in centre of tri and make 6 tris in place of 4 -- then there is nothing
	// much to choose -- just take most needful one and tell neighbours they are no longer allowed to be 
	// involved after we have done it.

	// Awkward for base triangle. Do edges with quads, then we know if we are picking the base edge ...
	// Maybe just don't do for base triangles after all, is it too much?
	// Would _like_ to consider moving _to_ edge if not away from edge.

	// Feels late at night -- all this takes time.

	
	// We list pairs of { tri with best neigh } and put an indicator once we have messed with a destination.

	
	static const real CONSERVATIVE_FACTOR = 1.5; 

	// Procedure: find which vertices have lowest values of "total nearby / N-2". Create a sorted list of indices. Only bother with say 10% of the list -
	// say we have 10000 points; once we have 1000 entries we can start chucking away anything that exceeds entry 1000.

	int listmax = numVertices/20;

	// Likewise find which triangles have the greatest mass (ion + neutral).

	int * index_verts;
	int * index_tris;
	real * pUseful;
	real * pNeedful;
	int i,j,iIndexVert,iIndexTri;

	this->CopyMesh(pDestMesh); 

	long iTri,iVertex,iPopulated;
	Vertex * pVertex;
	Triangle * pTri;
	int iLen;
	real Usefulness,Needfulness;

	index_verts = new int[listmax];
	index_tris = new int[listmax];
	pUseful = new real[listmax];
	pNeedful = new real[listmax]; // Note that we should try still for a speedup by changing arrays to static in general.
		
	// Another way to think: usefulness is the difference made to sum of squares if it is removed.
	// Ask sum of squared masses now and guess what it would be if we removed it. ???
	
	iPopulated = 0;
	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		iLen = pVertex->triangles.len;
		Usefulness = 0.0;
		for (i = 0; i < iLen; i++)
		{
			pTri = (Triangle *)(pVertex->triangles.ptr[i]);
			Usefulness += pTri->ion.mass + pTri->neut.mass;
		};
		if (pVertex->flags < 3) {
			Usefulness /= (real)(iLen-2); // average of masses after removing vertex
		} else {
			Usefulness /= (real)(iLen-1); // for an edge vertex it only reduces # triangles by 1. 
			// [ Does Disconnect know that? ]
		};
		// So the question was:
		// Can we move, and still have it that the average mass where we go will STILL be more than
		// the average mass in the triangles we then leave behind?
				
		// Now place this index into our list:
		if (iVertex < listmax) {
			// it will definitely go somewhere
			i = 0;
			while ((i < iPopulated) && (pUseful[i] < Usefulness)) // want least useful as element 0
				i++;
			if (i < iPopulated)
			{
				// shunt up the rest
				for (j = iPopulated-1; j > i; j--)
				{
					index_verts[j] = index_verts[j-1];
					pUseful[j] = pUseful[j-1];
				};
			};
			index_verts[i] = iVertex;
			pUseful[i] = Usefulness;
			iPopulated++;
		} else {
			if (Usefulness < pUseful[listmax-1]) // otherwise do nothing
			{
				i = listmax-1;
				while ((i >= 0) && (Usefulness < pUseful[i])) i--;
				i++;
				// i is now the element it should replace

				for (j = listmax-1; j > i; j--)
				{
					index_verts[j] = index_verts[j-1];
					pUseful[j] = pUseful[j-1];
				};
				index_verts[i] = iVertex;
				pUseful[i] = Usefulness;
			};
		};
		++pVertex;
	};
	

	// Now compare the least useful vertex with most needful triangle
	iPopulated = 0;
	pTri = T;
	
	for (iTri = 0; iTri < numTriangles; iTri++) // numTriangles not to include any spare ones.
	{
		// pick neighbour with most mass:

		massmax = 0.0;
		for (i = 0; i < 3; i++)
		{
			mass_i = pTri->neighbours[i]->ion.mass + pTri->neighbours[i].neut.mass);
			if (mass_i > massmax) {
				massmax = mass_i;
				max = i;
			};
		};
		Needfulness = 0.25*(pTri->ion.mass+pTri->neut.mass + massmax);

		// If an edge triangle should be concerning with boundary delta vs sqrt(n) rather than 2*own mass.

		//Needfulness = THIRD*(pTri->ion.mass + pTri->neut.mass);
		
		// Now place this index into our list:
		if (iPopulated < listmax) {
			// it will definitely go somewhere
			i = 0;
			while ((i < iPopulated) && (pNeedful[i] > Needfulness))
				i++;
			if (i < iPopulated)
			{
				// shunt up the rest
				for (j = iPopulated-1; j > i; j--)
				{
					index_tris[j] = index_tris[j-1];
					pNeedful[j] = pNeedful[j-1];
				};
			};
			index_tris[i] = iTri;
			pNeedful[i] = Needfulness;
			iPopulated++;
		} else {
			if (Needfulness > pNeedful[listmax-1]) // otherwise do nothing
			{
				i = listmax-1;
				while ((i >= 0) && (Needfulness > pNeedful[i])) i--;
				i++;
				// i is now the element it should replace

				// Faster search would start from previous i since neighbouring triangles may have similar mass.
				for (j = listmax-1; j > i; j--)
				{
					index_tris[j] = index_tris[j-1];
					pNeedful[j] = pNeedful[j-1];
				};
				index_tris[i] = iTri;
				pNeedful[i] = Needfulness;
			};
		};
		++pTri;
	};
	
	// Note that moving should void the possibility of moving a neighbour vertex from its position.
	// Therefore we should have to iterate again if that tries to happen.
	
	// reset vertex flags

	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		pVertex->iVolley = 0;
		++pVertex;
	};

	iIndexVert = 0;
	iIndexTri = 0;
	int iJumped = 0;
	bool worth_testing_further = true;
	do
	{
		// find next viable vertex to compare
		while ((iIndexVert < listmax) && (X[index_verts[iIndexVert]].iVolley > 0))
			iIndexVert++;

		if (iIndexVert < listmax) {
			// make the comparison
			if (pUseful[iIndexVert]*CONSERVATIVE_FACTOR < pNeedful[iIndexTri])
				// mass is still less even with vertex removed - we imagine
			{
				pVertex = X+index_verts[iIndexVert];
				pTri = T + index_tris[iIndexTri];
				

				// here we make some changes:
				// renew knowledge of which neighbour had max.
				// Only if this tri and that neigh both say they have not been messed with hitherto,
				// then do the move ....
				// and set the indicators for both of them of course.
				// The move is a bit like before as regards disconnect but now we'll choose to
				// reconnect in the centroid of the quadrilateral.
				
				
				
				// there are some other things that could rule out a jump - e.g. if this tri is actually a neighbour of this vertex...
				// not something we would hope for but we rule it out here anyway:
				
				if ((pVertex != pTri->cornerptr[0]) &&
					(pVertex != pTri->cornerptr[1]) &&
					(pVertex != pTri->cornerptr[2]))
				{
					// rule out doing any jump for neighbours around where it begins from:
					for (i = 0; i < pVertex->neighbours.len; i++)
					{
						(X+pVertex->neighbours.ptr[i])->iVolley++; 
					};			
					pDestMesh->VertexJump(index_verts[iIndexVert], index_tris[iIndexTri]);
					iJumped++;
					// Previously the neighbour hold came here, after it arrived somewhere. Doesn't matter - the jump is in the dest system.
				};
				iIndexTri++;
				iIndexVert++;
			} else {
				// the most needful tri is not a better place for X[index_verts[iIndexVert]].
				worth_testing_further = 0;
			};
		} else {
			worth_testing_further = 0; // end of viable vertex list - nothing further we can do
		};
	} while (worth_testing_further);

	// reset vertex flags
	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		pVertex->iVolley = 0;
		++pVertex;
	};

	delete[] index_verts;
	delete[] index_tris;
	delete[] pUseful;
	delete[] pNeedful;

	return iJumped;
}
*/
void TriMesh::SwimMesh(TriMesh * pSrcMesh)
{
	real acceptance, mass_avg,mass_SD,mass_min,mass_max, move, coefficient;

	FILE * swimfile = fopen("swim.txt","a");
	
	// coefficient is the (adaptive) proportion of the steps we try to make....
	// why is it that most of our moves start getting rejected, I do not know
	coefficient = 0.5;
	
	GlobalMaxVertexRadiusSq = 0.0;
	Vertex * pVertex = X;
	for (long iVertex= 0; iVertex < numVertices; iVertex++)
	{
		if (pVertex->x*pVertex->x+pVertex->y*pVertex->y > GlobalMaxVertexRadiusSq) 
			GlobalMaxVertexRadiusSq = pVertex->x*pVertex->x+pVertex->y*pVertex->y;
		++pVertex;
	};
	
	// puzzle why moves are rarely accepted so try something more empirical even if it takes a little longer to run.
	
	for (int j = 0; j < 4; j++) // 3 goes of 1 squeeze, 1 further go
	{
		fprintf(swimfile,"\n\nGSC: %d\n",GlobalStepsCounter);

		printf("%d ",j);
		fprintf(swimfile,"Swim %d  ",j);

		move = this->SwimVertices(pSrcMesh, coefficient, &acceptance);
		printf(" L2 of moves: %1.6E  Squeeze: %1.6E   Acceptance rate: %.2f%%\n",move,coefficient,acceptance*100.0);

		fprintf(swimfile," L2 of moves: %1.6E  Squeeze: %1.6E   Acceptance rate: %.2f%%\n",move,coefficient,acceptance*100.0);

		if ((acceptance > 0.1) && (move < 1.0e-7)) break; // stop if wasting time

		// sometimes move came out 0 indicating acceptance rates had fallen very low ; ..
		if (acceptance == 0.0)
		{
			coefficient *= 0.05;
		} else {		
			if ((acceptance < 0.5) && (j % 3 == 2))
			{
				if (acceptance > 0.025) {
					coefficient *= acceptance/0.5; // Note:quite aggressive - could put sqrt
				} else {
					// too low...
					coefficient *= 0.05;
				};
			};
		};
	};
			
	fclose(swimfile);
}

real TriMesh::SwimVertices(TriMesh * pSrcMesh, real coefficient, real * pAcceptance)
{
	// Redistribute vertices towards where we get more equal masses in cells.
	
	// We must have already positioned vertices in pDestMesh -- position same as existing initially;
	// *this has the existing mesh and values that we keep until we are finished with iterating.
	
	// ( We should then swap pointers: DestMesh is then the current system. )
	
	
	//		1. Swim vertices according to distribution of mass seen on dest mesh;
	//		2. Renew mass distribution for dest mesh by doing advectioncompression with 0 velocity 
	// Then loop.

	// When we are ready to stop, we want those filled in values for dest mesh.
	
	// Here's one thought.
	// Suppose we pick a time when we are doing a move anyway.
	// ...
	//					we created the new mesh
	//				we do advectioncompression for two species
	//				that lands us with mass
	//					let's say we see that vertices can swim profitably
	//				we move them, zero the cells and do the advectioncompression again
	//					is there a criterion to see that it was an improvement?

	// OK so perhaps we want two functions:
	// 1. Establish whether we can make gains
	// 2. Make the moves

	// Remember that we want to FinishAdvectingMesh when we reposition the vertices
	// Might want to look over that also.



		// try to decide a more optimal positioning for this point based on equalising the masses of triangles...
		// So we want density to be inversely proportional to area.

		// If we assume that the density at this vertex is what we add or take away from a triangle, that is 
		// probably not madly wrong.

		
		// Well here is an idea ... assume density in triangles stays fairly constant ... this will be a small move ...
		// or assume that it's somewhere between the two ... 
		// ... either actually might fail if just one of the neighbours is very tall.
		// in that case we need to be assuming that the edge density of the tall cell is what we add when we
		// move the boundaries.

		// Could do the following way: 
		// We need to pick a direction and a magnitude for changing vertex position.
		// Pick the direction that optimises the sum of squared differences from equal mass -- ??

		// Can't remember which way worked best.

		// Maybe we should just consider moving towards average of neighbours --- is it true that if cells
		// are maldistributed then there is always something that can do this?

		// However it might sometimes be, if behaviour locally is very bad, so be careful.


		// OK let's do this -- let's consider just moving towards neighbour average, which makes the grid more equilateral.

		
		// rate of change of triangle area can be found by dotting this move direction with direction perp to other edge
		
		// If we assume density in each triangle is given (this may have proven to be a bad assumption before)
		// then we can try to minimize sum of squared masses ?

		// What we did before was take Sum(mass - avgmass)^2 as objective function and estimated grad empirically,
		// then changed magnitude - to settle on one s.t. both initial value is worse and halving magnitude is also worse.
		

		// So here is what we should do: only move lone vertices: set a flag to each neighbour that it has to be in the next volley.
		// This way we can tell if each move has been an improvement, if we remember the previous objective function at each vertex.

		// We're spoilt for choice: can indicate which volley with Vertex::iScratch or Vertex::flags
		
		// Vertex::e_pm_Heat_denominator can be the objective function stored for the initial position
		// Vertex::eP_Viscous_denominator_x, eP_Viscous_denominator_y can be the stored initial position

		// Vertex::IonP_Viscous_numerator_x, IonP_Viscous_numerator_y is the position we are leapfrogging
		// Vertex::ion_pm_Heat_numerator can be the objective function stored for this position

		// Vertex::iScratch can be the volley to which it is assigned.
		// (and we repeat until everything is assigned status -1 meaning done).
		// Vertex::flags can be which way we are heading -- 0 means more change, 1 means less.
		
		// in this way we bisect to get an improvement:
		// if the new value is worse than e_pm_Heat_denominator then we go smaller until it is better
		// and keep going smaller until it stops getting better
		// if the new value is better, we can try going larger; when it stops being better we stop and accept previous
		
		// Algorithm: 
		// If masses are already within ~8% of each other, it is not worth moving.
		//	If they are somewhat different, do a move:

		//					create 2 first guesses 20% apart --- the magnitude to pick may be based on a number of things,
		// but in particular we might solve the linear equation for d/dt (sum of squared masses) = 0.
		// Note that we also get our grad Objective by assuming that the change is adding (and subtracting) at rate n_vertex.

		// Let's say we take that, reduce it somewhat if necessary according to practical constraints,
		// then consider that vs 80% of that.

		// Now one guess will be better than the other;
		// we walk another 10% that way
		// and continue as long as it keeps getting better.
		// If we end up with a quite small move then stop bothering.
		// When we come to a guess that is worse, we go back to the previous one Viscous_numerator.

		// If that is no better than the original, we fail and stay where we were; if it is better, we accept it.

		// ... so, we need to store which way we are going; for this try Vertex::flags
		
		// We also need to store the direction we are heading in -- make this Pressure_numerator
		// -- since we assume we are not doing this as part of an advance.
		
		// it would be far better to therefore _NOT_ do this re-jig as part of advection
		// There will be a few stubborn points where we do the re-mapping many times -- for these,
		// we want to just re-create masses repeatedly using the triangles locally, not re-doing the whole system.
		
		// ...Seems that it is high time we created a function that returns the triplanar model for a tri
		// or the quadriplanar model for a wedge.
		// We will need it again for smoothings.
		// We will need to do a zero-velocity advection to place triangles of mass and mom on to this new mesh.
		// Wish to do it for particular sets of triangles at a time.

		// Bite the leather.






		// New plan.

		// Do populate masses from source mesh each time.

		// Each volley:
		//				Populate masses for initial position from source mesh;
		//				Calculate objective functions and store them; label neighbour vertices to next volley;
		//					store old positions and create new ones based on grad objective function (store grad and magnitude)
		//				Populate masses again from source mesh;
		//					Create another guess of position: store objective function and our first guess
		//				Populate masses again from source mesh;
		//					Calculate objective functions for 3rd time; now accept the best guess of the 3.

		// so we have 3 populates per volley; we may have 4 volleys I expect. But it could be 5. 
		// .... This is a fairly expensive procedure to run even 1 go of. 
		

// REMEMBER TO CALL VERTEXNEIGHBOURSOFVERTICES BEFORE WE EMBARK ON THIS SWIMVERTICES BUSINESS
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	long * pIndex;
	Vertex * pVertex;
	long iVertex,iTri;
	real length1,length2;
	Vector2 average_pos, changevector;
	int i;
	Triangle * pTri;
	int iVolley = 0;
	bool found_vertices_this_volley;
	real objective, weight,area,minlen, magnitude, mass;
	Vector2 grad, putative;
	real d_mass_by_dx,d_area_by_dx,d_mass_by_dy,d_area_by_dy;
	Vector2 u0,u1,u2,uO;
	real sum_mass_times_rate_of_change ,sum_squared_rates_of_change,sum_squared_move_length,
		d_mass_per_unit_grad , newx,newy;
	int triangles_len;
	bool crush_radial_move;
	Triangle * pDummyTri;
	int which,c1;

	real xdist,ydist,dist, graddotnormal,max, original_dot,original_dist,new_position_dot;
	Vector2 edgenormal, rhat, mingrad, from_here, to1, to2, u;
	Vertex * pNeigh1, * pNeigh2;
	real d_minmass_by_dt, d_mass_by_dt, crossover, minmass,
		normaldistance, tmax;
	int iMin;
	long attempted, accepted;
	attempted = 0; accepted = 0;
	
	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		pVertex->iVolley = 0;
		++pVertex;
	};

	sum_squared_move_length = 0.0; // a crude way to gauge how much impression this call of SwimVertices makes
	
	do // while (found_vertices_this_volley)
	{
		found_vertices_this_volley = false;

		// for each vertex that is assigned to this volley..
		pVertex = X;
		for (iVertex = 0; iVertex < numVertices; iVertex++)
		{
			if (pVertex->iVolley == iVolley)
			{
				found_vertices_this_volley = true;
				
				// If any neighbours are assigned to this volley, send them to the next volley
				pIndex = pVertex->neighbours.ptr;
				for (i = 0; i < pVertex->neighbours.len; i++)
				{
					if (X[*pIndex].iVolley == iVolley)
						X[*pIndex].iVolley++;
					++pIndex;
				};
			};
			++pVertex;
		};

		// Now we did this first so that if this volley is empty, we do not waste time doing the following :
		if (found_vertices_this_volley)
		{	
			pTri = T;
			for (iTri = 0; iTri < numTriangles; iTri++)
			{
				pTri->RecalculateEdgeNormalVectors(true); // these are used below
				++pTri;
			};
			//	Populate masses for initial position from source mesh;
			printf("`");
			pSrcMesh->RepopulateCells(this,MASS_ONLY);
			printf(".");
			
			//	Calculate objective functions and store them; label neighbour vertices to next volley;
			//	store old positions and create new ones based on grad objective function (store grad and magnitude)
			
			pVertex =X;			
			for (iVertex = 0; iVertex < numVertices; iVertex++)
			{
				if (pVertex->iVolley == iVolley)
				{
					objective = 0.0;
					grad.x = 0.0; grad.y = 0.0;
					sum_squared_rates_of_change = 0.0;
					sum_mass_times_rate_of_change = 0.0;

					triangles_len = pVertex->triangles.len;
					pVertex->ion.n = 0.0;
					pVertex->neut.n = 0.0;
					for (i = 0; i <  triangles_len; i++)
					{
						// Get objective function:
						pTri = (Triangle *)(pVertex->triangles.ptr[i]);
						mass = (pTri->ion.mass + pTri->neut.mass);
						objective += mass*mass;

						// Recalculate pVertex->ion.n and pVertex->neut.n
						weight = pTri->ReturnAngle(pVertex); // takes acct of periodic & wedge cases.
						area = pTri->GetArea();
						pVertex->ion.n += weight*(pTri->ion.mass)/area;			
						pVertex->neut.n += weight*(pTri->neut.mass)/area; // really only want the total of course

						// (( Is this how vertex n is calculated elsewhere? ))
					};

					pVertex->e_pm_Heat_denominator = objective; // store for initial positions
					pVertex->eP_Viscous_denominator_x = pVertex->x;
					pVertex->eP_Viscous_denominator_y = pVertex->y; // store initial positions
										
					// Now collect contributions to grad Area, and normalise:
					iMin = 0; minmass = 1.0e100;
					mingrad.x = 0.0; mingrad.y = 0.0;
					for (i = 0; i <  triangles_len; i++)
					{
						pTri = (Triangle *)(pVertex->triangles.ptr[i]);
						pTri->Return_grad_Area(pVertex,&d_area_by_dx,&d_area_by_dy); 		// contiguous with pVertex
						mass = (pTri->ion.mass + pTri->neut.mass);
						
						// Now estimate gradient of objective:
						// d/dx sum of squares = 2 sum (mass )(d/dx mass )
						// To find change of area, dot x with the vector that is normal to the other side						
						// In this whole function we need to map periodically everything to be on same side as pVertex.
						// THAT is rather crucial isn't it 
						
						d_mass_by_dx = (pVertex->ion.n + pVertex->neut.n)*d_area_by_dx;
						d_mass_by_dy = (pVertex->ion.n + pVertex->neut.n)*d_area_by_dy;
						
						if (mass < minmass) {
							minmass = mass;
							iMin = i;
							mingrad.x = d_mass_by_dx;
							mingrad.y = d_mass_by_dy;
						};
						grad.x += mass*d_mass_by_dx*2.0; // gradient of mass*mass ... 
						grad.y += mass*d_mass_by_dy*2.0;
					};
					grad.Normalise(); 

					// Now in place of grad we want our intended move direction.
					if (pVertex->flags >= 3) {
						// delete radial component:
						rhat.x = pVertex->x; rhat.y = pVertex->y; rhat.Normalise();
						grad -= rhat*(grad.dot(rhat)); 
					};

					// Test: if grad is making minimum tri smaller that is bad
					// Bear in mind we expect to head along negative of grad to REDUCE objective function
					if (grad.x*mingrad.x + grad.y*mingrad.y > 0.0)
					{
						// heading against mingrad - no good.
						// we already set pVertex->eP_Viscous_denominator_x = pVertex->x
						 
						pVertex->IonP_Viscous_numerator_x = pVertex->x;
						pVertex->IonP_Viscous_numerator_y = pVertex->y; // store unwrapped first guess positions

						// would be good to record number of times we hit this branch.
					} else {

						// IN EDGE CASE, I think we should be only mooting such moves in the first place
						// We can still take gradient of mass*mass, 2D, but then consider moving azimuthally.
						
						// Decide where to place a guess of a better position:
						// How to find d/dt sum of squares = 0? Modelling area as linear function of progress in this direction,
						// Magnitude :
						//						t = - sum (dA/dt ^2) / sum (A * dA/dt)
						
						// not sure about that??

						// Now get sums

						// This bit was not commented:
						//for (i = 0; i <  triangles_len; i++)
						//{
						//	pTri = (Triangle *)(pVertex->triangles.ptr[i]);
						//	
						//	mass = (pTri->ion.mass + pTri->neut.mass);
						//	pTri->Return_grad_Area(pVertex,&d_area_by_dx,&d_area_by_dy); 			
						//	d_mass_by_dx = (pVertex->ion.n + pVertex->neut.n)*d_area_by_dx;
						//	d_mass_by_dy = (pVertex->ion.n + pVertex->neut.n)*d_area_by_dy;
						//	
						//	d_mass_per_unit_grad = d_mass_by_dx*grad.x + d_mass_by_dy*grad.y; // for a move in direction of grad.
						//	sum_squared_rates_of_change += d_mass_per_unit_grad*d_mass_per_unit_grad;
						//	sum_mass_times_rate_of_change += d_mass_per_unit_grad*mass;
						//};
						//
						//magnitude = - sum_mass_times_rate_of_change / sum_squared_rates_of_change;

						// THAT SEEMS LIKE A BOLD PLAN !
						
						// Alternative idea
						// ______________
						// They are all changing at different rates
						// Stop when one that is moving down becomes the minimum one???
						// How to do?
						// maybe just stop when whichever ones are moving downwards, reach past original average ?
						// might be one close to average moving down. 

						// See when the down-movers cross over the one coming up from least mass.
						// Stop when it crosses over one. Is there a scenario where that is bad? Think it looks pretty good.

						// assume we head in direction MINUS grad
						d_minmass_by_dt = -(mingrad.dot(grad)); // > 0
						
						magnitude = 1.0e100;
						for (i = 0; i <  triangles_len; i++)
						{
							pTri = (Triangle *)(pVertex->triangles.ptr[i]);
							
							mass = (pTri->ion.mass + pTri->neut.mass);
							pTri->Return_grad_Area(pVertex,&d_area_by_dx,&d_area_by_dy); 			
							d_mass_by_dx = (pVertex->ion.n + pVertex->neut.n)*d_area_by_dx;
							d_mass_by_dy = (pVertex->ion.n + pVertex->neut.n)*d_area_by_dy;
							
							d_mass_by_dt = -d_mass_by_dx*grad.x - d_mass_by_dy*grad.y; 
							// ROC for a move in direction of minus grad, we think.
							
							if (d_mass_by_dt < 0.0) {
								// find crossing time; take t = min(t,crossover of this one with min mass)
								crossover = (mass-minmass)/(d_minmass_by_dt - d_mass_by_dt); 
								magnitude = min (magnitude, crossover);
							};
						};
						
						// Just measure, in the first place, the normal distance across a triangle, and travel at most 0.33 of this.
						// Or perhaps just pick the ones where the motion is making that normal shorter - yes.

						if (pVertex->flags < 3) {
							for (i = 0; i < triangles_len; i++)
							{
								// if triangle is periodic then we need to map other vertices to living nearby....
								pTri = (Triangle *)(pVertex->triangles.ptr[i]);

								which = 0; c1 = 1;
								if (pVertex == pTri->cornerptr[1]) { which = 1; c1 = 0;};
								if (pVertex == pTri->cornerptr[2]) which = 2;
								pTri->cornerptr[c1]->PopulatePosition(u1);
								edgenormal = pTri->edge_normal[which]; // We called with (true) so is already normalised
								// this faces across the triangle.
									
								if (pTri->periodic > 0) 
								{
									if (pVertex->x < 0.0) {
										if (u1.x > 0.0) u1 = Anticlockwise*u1;
									} else {
										edgenormal = Clockwise*edgenormal; // make contig with our vertex
										if (u1.x < 0.0) u1 = Clockwise*u1;
									};
								};
								
								if (grad.dot(edgenormal) < 0.0) {
									// only care if we are making the distance shorter by moving in direction _minus grad_
									from_here.x = u1.x-pVertex->x;
									from_here.y = u1.y-pVertex->y;
									normaldistance = from_here.dot(edgenormal); // >0
									// -grad dot edgenormal is the rate of progress in reducing normal distance, dot product of normalised vectors
									tmax = -0.33*(normaldistance/(grad.dot(edgenormal))); // >0
									magnitude = min(tmax,magnitude);
								};
							};
						} else {
							// pVertex->flags >= 3 : test against base neighbour distance only
							pNeigh2 = X + pVertex->neighbours.ptr[0];
							pNeigh1 = X + pVertex->neighbours.ptr[pVertex->neighbours.len-1];
							if ((pNeigh2->flags != pVertex->flags) || (pNeigh1->flags != pVertex->flags))
							{
								printf("\nDid we fail to call RefreshNeighboursOfVerticesOrdered?\n");
								getch();
							};
							// we will only be moving towards one of them. Which one?
							if (pVertex->has_periodic) {
								pNeigh1->PopulatePosition(u1);
								pNeigh2->PopulatePosition(u2);
								if (pVertex->x > 0.0) {
									if (pNeigh1->x < 0.0) u1 = Clockwise*u1;
									if (pNeigh2->x < 0.0) u2 = Clockwise*u2;
								} else {
									if (pNeigh1->x > 0.0) u1 = Anticlockwise*u1;
									if (pNeigh2->x > 0.0) u2 = Anticlockwise*u2;
								};
								to1.x = u1.x - pVertex->x; to1.y = u1.y - pVertex->y;
								to2.x = u2.x - pVertex->x; to2.y = u2.y - pVertex->y;
							} else {
								to1.x = pNeigh1->x-pVertex->x; to1.y = pNeigh1->y-pVertex->y; 
								to2.x = pNeigh2->x-pVertex->x; to2.y = pNeigh2->y-pVertex->y;
							};
							if (grad.dot(to1)*grad.dot(to2) > 0.0)
							{
								printf("summat WEIRD - heading towards/away from both edge neighs \n");
								to1 = to1;
							};
							if (grad.dot(to1) < 0.0) { // bear in mind we move along minus grad
								tmax = -0.33*(to1.dot(to1)/(grad.dot(to1))); // >0
								magnitude = min(tmax,magnitude);
							}
							if (grad.dot(to2) < 0.0) {
								tmax = -0.33*(to2.dot(to2)/(grad.dot(to2))); // >0
								magnitude = min(tmax,magnitude);
							};
						};

						magnitude *= coefficient;	//	Adaptive coefficient. Mysteriously goes small.
						
						if (magnitude < 0.0)
						{
							//  do a warning and try using 0.15 the nearest neighbour length
							printf("	\t magnitude negative -- ");
						};
						
						pVertex->x = pVertex->x - magnitude*grad.x;
						pVertex->y = pVertex->y - magnitude*grad.y;

						if (pVertex->flags == 3)
						{
							pVertex->project_to_ins(u);
							pVertex->x = u.x; pVertex->y = u.y;
						}
						if (pVertex->flags == 4)
						{
							pVertex->project_to_radius(u,Outermost_r_achieved);
							pVertex->x = u.x; pVertex->y = u.y;
						}

						// The following code was too complicated and so was replaced by simply comparing to distances across triangles.


						//// Now we make it at most the nearest neighbour distance.
						//minlen = 1.0; // 1 cm - improbably large
						//for (i = 0; i < triangles_len; i++)
						//{
						//	// if triangle is periodic then we need to map other vertices to living nearby....
						//	pTri = (Triangle *)(pVertex->triangles.ptr[i]);
						//	if (pTri->periodic == 0)
						//	{
						//		pTri->PopulatePositions(u0,u1,u2);								
						//	} else {
						//		// periodic triangle								
						//		if (pVertex->x < 0.0) // bit slapdash but hey, unreasonable for periodic tris to cross centre.
						//		{
						//			pTri->MapLeft(u0,u1,u2);
						//		} else {
						//			pTri->MapRight(u0,u1,u2);
						//		};								
						//	};
						//	
						//	if (pVertex == pTri->cornerptr[0])
						//	{
						//		length1 = (u0-u2).modulus();
						//		length2 = (u0-u1).modulus();
						//	} else {
						//		if (pVertex == pTri->cornerptr[1])
						//		{
						//			length1 = (u1-u0).modulus();
						//			length2 = (u1-u2).modulus();
						//		} else {
						//			length1 = (u2-u0).modulus();
						//			length2 = (u2-u1).modulus();
						//		};
						//	};
						//	minlen = min(minlen,min(length1,length2));
						//}; // Done this way because we need contiguous neighbour image which is harder to get from neighbour array. Okay.
						//
						//// We actually keep it down to 33% of the distance to a neighbour :
						//if (magnitude > 0.33*minlen) magnitude = 0.33*minlen;
						//
						//if (magnitude < 0.0)// && (crush_radial_move == false))
						//{
						//	//  do a warning and try using 0.15 the nearest neighbour length
						//	printf("	\t magnitude negative -- ");
						//	magnitude = 0.15*minlen;
						//};

						//// We do not push out beyond the outermost radius: (hopefully unnecessary check)
						//if (pVertex->x*pVertex->x + pVertex->y*pVertex->y > r_Outermost*r_Outermost)
						//{
						//	real factor = sqrt((pVertex->x*pVertex->x + pVertex->y*pVertex->y) / r_Outermost*r_Outermost);
						//	pVertex->x /= factor;
						//	pVertex->y /= factor;
						//};

						//// Now also verify that this move is not taking us outside the adjacent cells.
						//// This seems pointless: if we did not move more than 33% distance to nearest neighbour then
						//// how can we possibly have exited cells? Perhaps if triangle is extremely flat for some reason. :/

						//// Simpler way then: do not move more than a fraction of normal distance in a triangle!




						//// Use for debug only :

						//magnitude = - magnitude; // old way: coeff on grad not minus grad
						//for (i = 0; i < triangles_len; i++)
						//{
						//	// if triangle is periodic then we need to map other vertices to living nearby....
						//	pTri = (Triangle *)(pVertex->triangles.ptr[i]);

						//	which = 0; c1 = 1;
						//	if (pVertex == pTri->cornerptr[1]) { which = 1; c1 = 0;};
						//	if (pVertex == pTri->cornerptr[2]) which = 2;

						//	// Note that transvec exist for triangle that is mapped left.
						//	if (pTri->periodic == 0) 
						//	{
						//		if (pTri->TestAgainstEdge(pVertex->x,pVertex->y, c1, which, &pDummyTri))
						//		{
						//			// outside!
						//			// how far is it to the edge then?
						//			// Take original position dot normalized tranverse vector

						//			printf("summat strange --- exiting polygon although only move 33% towards neighbours.");

						//			edgenormal = pTri->edge_normal[which]; // DID WE DO NORMALISE TRUE?
						//			edgenormal.Normalise();
						//			xdist = pVertex->eP_Viscous_denominator_x - pTri->cornerptr[c1]->x;
						//			ydist = pVertex->eP_Viscous_denominator_y - pTri->cornerptr[c1]->y;
						//			dist = xdist*edgenormal.x + ydist*edgenormal.y; // may be + or -
						//			// That is the normal distance across the triangle.

						//			// We want to know what multiple of -grad
						//			graddotnormal = grad.x*edgenormal.x+grad.y*edgenormal.y; // dot product of normalized vectors
						//			max = -fabs(dist/graddotnormal);
						//			if (max < magnitude)
						//			{
						//				// error
						//				printf("\nshouldn't be here .. max < magnitude \n");
						//				getch();
						//			} else {
						//				magnitude = max*0.33;
						//				pVertex->x = pVertex->eP_Viscous_denominator_x + magnitude*grad.x;
						//				pVertex->y = pVertex->eP_Viscous_denominator_y + magnitude*grad.y;
						//				if (pVertex->x*pVertex->x + pVertex->y*pVertex->y > r_Outermost*r_Outermost)
						//				{
						//					printf("\n\n\nStrewth! ultimate default \n--------\n\n\n");
						//					// ultimate default:
						//					pVertex->x = pVertex->eP_Viscous_denominator_x;
						//					pVertex->y = pVertex->eP_Viscous_denominator_y;
						//				};
						//				//should be domain interior verts here only.
						//			};
						//		};
						//	} else {
						//		// periodic triangle:
						//		
						//		if (pVertex->x > 0.0)
						//		{
						//			//x_on_left = Anticlockwise.xx*pVertex->x + Anticlockwise.xy*pVertex->y;
						//			//y_on_left = Anticlockwise.yx*pVertex->x + Anticlockwise.yy*pVertex->y;

						//			if (which == 0) pTri->MapRight(uO,u0,u1);
						//			if (which == 1) pTri->MapRight(u0,uO,u1);
						//			if (which == 2) pTri->MapRight(u0,u1,uO);

						//			edgenormal.x = u1.y-u0.y;
						//			edgenormal.y = u0.x-u1.x;
						//			
						//				// want to assess whether pVertex->x,y on same side as pVertex->Viscous_denominator
						//			original_dot = (pVertex->eP_Viscous_denominator_x - u0.x)*edgenormal.x
						//							 + (pVertex->eP_Viscous_denominator_y - u0.y)*edgenormal.y;

						//			new_position_dot = (pVertex->x - u0.x)*edgenormal.x
						//							+ (pVertex->y - u0.y)*edgenormal.y;
						//			if (new_position_dot*original_dot < 0.0)
						//			{
						//				
						//				printf("summat strange --- exiting polygon although only move 33% towards neighbours.");

						//				// not same side of edge.										
						//				edgenormal.Normalise();										
						//				original_dist = (pVertex->eP_Viscous_denominator_x - u0.x)*edgenormal.x
						//											  + (pVertex->eP_Viscous_denominator_y - u0.y)*edgenormal.y;
						//				graddotnormal = grad.x*edgenormal.x + grad.y*edgenormal.y;
						//				
						//				max = -fabs(original_dist/graddotnormal);
						//				if (max < magnitude)
						//				{
						//					// error
						//					printf("\nshouldn't be here .. max < magnitude \n");
						//					getch();
						//				} else {
						//					magnitude = max*0.33;
						//					pVertex->x = pVertex->eP_Viscous_denominator_x + magnitude*grad.x;
						//					pVertex->y = pVertex->eP_Viscous_denominator_y + magnitude*grad.y;
						//					if (pVertex->x*pVertex->x + pVertex->y*pVertex->y > GlobalMaxVertexRadiusSq)
						//					{
						//						printf("\n\n\nStrewth! ultimate default \n--------\n\n\n");
						//						// ultimate default:
						//						pVertex->x = pVertex->eP_Viscous_denominator_x;
						//						pVertex->y = pVertex->eP_Viscous_denominator_y;
						//					};
						//				};
						//			};
						//		} else {
						//			// x is on left so it should be easier
						//			if (pTri->TestAgainstEdge(pVertex->x,pVertex->y, c1, which, &pDummyTri))
						//			{
						//				// outside!
						//				// how far is it to the edge then?
						//				// Take original position dot normalized tranverse vector

						//				printf("summat strange --- exiting polygon although only move 33% towards neighbours.");

						//				if (which == 0) pTri->MapLeft(uO,u0,u1);
						//				if (which == 1) pTri->MapLeft(u0,uO,u1);
						//				if (which == 2) pTri->MapLeft(u0,u1,uO); // may be wedge or tri

						//				edgenormal = pTri->edge_normal[which];
						//				//edgenormal.Normalise();
						//				xdist = pVertex->eP_Viscous_denominator_x - u0.x;
						//				ydist = pVertex->eP_Viscous_denominator_y - u0.y;
						//				original_dist = xdist*edgenormal.x + ydist*edgenormal.y; // may be + or -

						//				graddotnormal = grad.x*edgenormal.x+grad.y*edgenormal.y; // dot product of normalized vectors
						//				// this is how far we travel in direction jim for 1 unit of grad - that's one interpretation
						//			
						//				max = -fabs(original_dist/graddotnormal);

						//				if (max < magnitude)
						//				{
						//					// error
						//					printf("\nshouldn't be here .. max < magnitude \n");
						//					getch();
						//				} else {
						//					magnitude = max*0.33;

						//					pVertex->x = pVertex->eP_Viscous_denominator_x + magnitude*grad.x;
						//					pVertex->y = pVertex->eP_Viscous_denominator_y + magnitude*grad.y;

						//					if (pVertex->x*pVertex->x + pVertex->y*pVertex->y > GlobalMaxVertexRadiusSq)
						//					{
						//						printf("\n\n\nStrewth! ultimate default \n--------\n\n\n");
						//						// ultimate default:
						//						pVertex->x = pVertex->eP_Viscous_denominator_x;
						//						pVertex->y = pVertex->eP_Viscous_denominator_y;
						//					};
						//				};
						//			};
						//		};
						//	};
						//};
						
						// Bear in mind, this may be across PB so ReturnPointerToTriangle would fail.
																
						// In the case it crossed PB, we ought to update periodicity of triangles...					
						// Wrap (x,y) also -- but do not wrap the stored version - this allows us to take an average
						pVertex->IonP_Viscous_numerator_x = pVertex->x;
						pVertex->IonP_Viscous_numerator_y = pVertex->y; // store unwrapped first guess positions
						// ( used for counting up variance and doing periodic tests)
						
						if (pVertex->x/pVertex->y > GRADIENT_X_PER_Y)
						{
							// went off RH side						
							for (i = 0; i < triangles_len; i++)
							{
								// if triangle is periodic then we need to map other vertices to living nearby....
								pTri = (Triangle *)(pVertex->triangles.ptr[i]);
								pTri->IncrementPeriodic();
							};
							newx = Anticlockwise.xx*pVertex->x+Anticlockwise.xy*pVertex->y;
							newy = Anticlockwise.yx*pVertex->x+Anticlockwise.yy*pVertex->y;
							pVertex->x = newx;
							pVertex->y = newy;
						};

						if (pVertex->x/pVertex->y < -GRADIENT_X_PER_Y)
						{
							// went off LH side
							for (i = 0; i < triangles_len; i++)
							{
								// if triangle is periodic then we need to map other vertices to living nearby....
								pTri = (Triangle *)(pVertex->triangles.ptr[i]);
								pTri->DecrementPeriodic();
							};
							newx = Clockwise.xx*pVertex->x+Clockwise.xy*pVertex->y;
							newy = Clockwise.yx*pVertex->x+Clockwise.yy*pVertex->y;
							pVertex->x = newx;
							pVertex->y = newy;
						};				
					
						// DEBUG:
						if (pVertex->x*pVertex->x+pVertex->y*pVertex->y < 11.833599999)
						{
							pVertex->x = pVertex->x;
							// absolutely should not be able to happen
							printf("\nTarnation! point swam inside ins! \n");
							getch();
						};
					
					}; // whether against minimum triangle area grad

				}; // whether iVolley

				++pVertex;
			};
			
		//		Populate masses again from source mesh;
		// Note that we need to update transvec in order to place points and thus triangles into mesh

			pTri = T;
			for (iTri = 0; iTri < numTriangles; iTri++)
			{
				pTri->RecalculateEdgeNormalVectors(false);
				pTri++;
			};    
			printf("`");
			pSrcMesh->RepopulateCells(this,MASS_ONLY);
			printf(".");
			
			// We just test whether we gained an improvement in the objective function, and either accept this or not.

			pVertex = X;			
			for (iVertex = 0; iVertex < numVertices; iVertex++)
			{
				if (pVertex->iVolley == iVolley)
				{
					objective = 0.0;
					triangles_len = pVertex->triangles.len;
					for (i = 0; i < pVertex->triangles.len; i++)
					{
						pTri = (Triangle *)(pVertex->triangles.ptr[i]);
						objective += (pTri->ion.mass + pTri->neut.mass)*(pTri->ion.mass + pTri->neut.mass);
					};
					//pVertex->ion_pm_Heat_numerator = objective; // store for first guess

					// Decide whether to accept move:
					if (objective < pVertex->e_pm_Heat_denominator)
					{
						// improved
						// (x,y) already set so that's it
						
						// but in recording move, need to still remember it may have been wrapped across PB
						// so use the unwrapped coords:						
						sum_squared_move_length += (pVertex->IonP_Viscous_numerator_x - pVertex->eP_Viscous_denominator_x)*(pVertex->IonP_Viscous_numerator_x - pVertex->eP_Viscous_denominator_x)
							+ (pVertex->IonP_Viscous_numerator_y - pVertex->eP_Viscous_denominator_y)*(pVertex->IonP_Viscous_numerator_y - pVertex->eP_Viscous_denominator_y);
					
						accepted++;
						attempted++;
					} else {
						// revert to original position
						attempted++;
						pVertex->x = pVertex->eP_Viscous_denominator_x;
						pVertex->y = pVertex->eP_Viscous_denominator_y;
						// twist back any periodic changes:
						if (pVertex->IonP_Viscous_numerator_x/pVertex->IonP_Viscous_numerator_y > GRADIENT_X_PER_Y)
						{
							// in this case we applied increment periodic
							for (i = 0; i < triangles_len; i++)
							{
								// if triangle is periodic then we need to map other vertices to living nearby....
								pTri = (Triangle *)(pVertex->triangles.ptr[i]);
								pTri->DecrementPeriodic();
							};
						};
						if (pVertex->IonP_Viscous_numerator_x/pVertex->IonP_Viscous_numerator_y < -GRADIENT_X_PER_Y)
						{
							for (i = 0; i < triangles_len; i++)
							{
								// if triangle is periodic then we need to map other vertices to living nearby....
								pTri = (Triangle *)(pVertex->triangles.ptr[i]);
								pTri->IncrementPeriodic();
							};							
						};
					};					
				};
				++pVertex;
			};			


			/*
			// Due to PB, the following became too complicated to be viable !!


		//		Populate masses again from source mesh;

			PopulateMasses(pDestMesh);

		//		Calculate objective functions for 3rd time; now accept the best guess of the 3.

			pVertex = pDestMesh->X;
			for (iVertex = 0; iVertex < numVertices; iVertex++)
			{
				if (pVertex->iScratch == iVolley)
				{
					objective = 0.0;
					for (i = 0; i < pVertex->triangles.len; i++)
					{
						pTri = (Triangle *)(pVertex->triangles.ptr[i]);
						objective += (pTri->ion.mass + pTri->neut.mass)*(pTri->ion.mass + pTri->neut.mass);
					};
					
					if (objective < pVertex->e_pm_Heat_denominator)
					{
						if (objective < pVertex->ion_pm_Heat_numerator)
						{
							// second guess is best
							// don't need to make further changes
						} else {
							// first guess is best
							pVertex->x = pVertex->IonP_Viscous_numerator_x;
							pVertex->y = pVertex->IonP_Viscous_numerator_y;
						};
					} else {
						if (pVertex->ion_pm_Heat_numerator < pVertex->e_pm_Heat_denominator)
						{
							// first guess is best
							pVertex->x = pVertex->IonP_Viscous_numerator_x;
							pVertex->y = pVertex->IonP_Viscous_numerator_y;
						} else {
							// failed: both guesses were exprovements.
							pVertex->x = pVertex->eP_Viscous_denominator_x;
							pVertex->y = pVertex->eP_Viscous_denominator_y;
						};
					};

					// Again, if we have crossed PB relative to presently existing (x,y) then we have to update triangles' periodicity
					// But now we have a difficult problem:

					// We already wrapped points : 
					// confusing but I think we can miss something here. Suppose we end up reverting to original. It
					// may be on the other side from (x,y) which is a wrapped position.


					// This whole thing is too difficult

					// Let's just make one attempted move and take it or leave it ! (   :-(   )


					if (pVertex->x/pVertex->y > GRADIENT_X_PER_Y)
					{
						// went off RH side						
						for (i = 0; i < triangles_len; i++)
						{
							// if triangle is periodic then we need to map other vertices to living nearby....
							pTri = (Triangle *)(pVertex->triangles.ptr[i]);
							pTri->IncrementPeriodic();
						};
						newx = Anticlockwise.xx*pVertex->x+Anticlockwise.xy*pVertex->y;
						newy = Anticlockwise.yx*pVertex->x+Anticlockwise.yy*pVertex->y;
						pVertex->x = newx;
						pVertex->y = newy;
					};

					if (pVertex->x/pVertex->y < -GRADIENT_X_PER_Y)
					{
						for (i = 0; i < triangles_len; i++)
						{
							// if triangle is periodic then we need to map other vertices to living nearby....
							pTri = (Triangle *)(pVertex->triangles.ptr[i]);
							pTri->DecrementPeriodic();
						};
						newx = Clockwise.xx*pVertex->x+Clockwise.xy*pVertex->y;
						newy = Clockwise.yx*pVertex->x+Clockwise.yy*pVertex->y;
						pVertex->x = newx;
						pVertex->y = newy;
					};				
					
					sum_squared_move_length += (pVertex->x - pVertex->eP_Viscous_denominator_x)*(pVertex->x - pVertex->eP_Viscous_denominator_x)
						+ (pVertex->y - pVertex->eP_Viscous_denominator_y)*(pVertex->y - pVertex->eP_Viscous_denominator_y);
					
				}; // whether (pVertex->iScratch == iVolley)
				++pVertex;
			};*/

		}; // whether found any vertices this volley
		iVolley++;
	} while (found_vertices_this_volley);
	
	// One round of SwimVertices will attempt to move every vertex once.
		
	*pAcceptance = ((real)accepted)/((real)attempted);
	return sqrt(sum_squared_move_length/((real)(numVertices)));
}
	
// Routines in some kind of order. 
			
void TriMesh::RepopulateCells(TriMesh * pDestMesh,int code)
{
	long iVertex;
	Vertex * pVertex;

	bSwitchOffChPatReport = 1;

	pDestMesh->ZeroCellData();
	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		pVertex->AdvectedPosition[0].x = pVertex->x;
		pVertex->AdvectedPosition[0].y = pVertex->y;
		pVertex->AdvectedPosition[1].x = pVertex->x;
		pVertex->AdvectedPosition[1].y = pVertex->y;
		pVertex->AdvectedPosition[2].x = pVertex->x;
		pVertex->AdvectedPosition[2].y = pVertex->y;
		++pVertex;
	}
	PlaceAdvected_Triplanar_Conservative_IntoNewMesh(0,pDestMesh,code,0);
	PlaceAdvected_Triplanar_Conservative_IntoNewMesh(1,pDestMesh,code,0);
	if (code == ALL_VARS) 
		PlaceAdvected_Triplanar_Conservative_IntoNewMesh(2,pDestMesh,code,0);

	// Will it fail when we try to place a cell exactly over itself? Remains to be seen.
	
	bSwitchOffChPatReport = 0;
}

void TriMesh::PlaceAdvected_Triplanar_Conservative_IntoNewMesh(
													int which_species,
													TriMesh * pDestMesh,
													 int code,
													 int bDoCompressiveHeating)													
{
	// Fairly straightforward routine:
	// Take each advected triangle of this species;
	// Be careful of periodic-stuff
	// Call triplanar model, chop into 3 planes
	// Send each plane to the new mesh

	vertvars verts_use0, verts_use1, verts_use2, verts_c;
	vertvars * pvertvars[3][3];	// 3 planes
	Vector2 adv0,adv1,adv2,advC;
	real Areanew;
	Triangle * pTri;
	//cellvars * pVars;
	cellvars Vars;
	long iTri;
	int const numPlanes = 3;
	int src_periodic, o;
	ConvexPolygon cpPlane[numPlanes], cpTemp;
	real compression_factor;

	real compressive_heating_total = 0.0;

	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		
		// Note that whether or not tri was periodic before, it is entirely possible that it is now and that
		// a vertex went out of tranche
		// Alternatively a periodic tri can become non-per under advection with an out-of-tranche point to wrap.

		// Chop up into 3 subtri planes:

		// The position solve routine should have populated variables called
		// pVertex->AdvectedPosition[which_species]

		adv0 = pTri->cornerptr[0]->AdvectedPosition[which_species];
		adv1 = pTri->cornerptr[1]->AdvectedPosition[which_species];
		adv2 = pTri->cornerptr[2]->AdvectedPosition[which_species];
		
		if (which_species == SPECIES_ION)
		{
			//pVars = &(pTri->ion);
			Vars = pTri->ion;
			verts_use0 = pTri->cornerptr[0]->ion; // assignment equals, not reference assignment.
			verts_use1 = pTri->cornerptr[1]->ion;
			verts_use2 = pTri->cornerptr[2]->ion;
		} else {
			if (which_species == SPECIES_NEUTRAL)
			{
				//pVars = &(pTri->neut);
				Vars = pTri->neut;
				verts_use0 = pTri->cornerptr[0]->neut; // assignment equals, not reference assignment.
				verts_use1 = pTri->cornerptr[1]->neut;
				verts_use2 = pTri->cornerptr[2]->neut;
			} else {
				//pVars = &(pTri->elec);
				Vars = pTri->elec;
				verts_use0 = pTri->cornerptr[0]->elec; // assignment equals, not reference assignment.
				verts_use1 = pTri->cornerptr[1]->elec;
				verts_use2 = pTri->cornerptr[2]->elec;
			};
		};
				
	//	fail_n_tri = Triplanar( pVars, &verts_use0, &verts_use1, &verts_use2, &verts_c ,
		//									pTri->area,Areanew,ALL_VARS );
		
		// That surely cannot work.
		// We should not be calling with verts_use.v that are not contiguous - that is nonsense.

		// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

		// Now let's decide how to handle periodic stuffs.
		// What is the receiving routine expecting from us :
		
		//if (src_periodic == PERIODIC_SRC)
		//" Given that it's periodic, it appears on the left."
		// We clip against tranche; it is perfectly fine to say src_periodic holds even if it no longer does.
		// However we must ensure all points are a contiguous image and on the left.
		
		// if src_periodic == false then it assume src is not periodic though dest still might be.
		
		src_periodic = (pTri->periodic>0)?PERIODIC_SRC:NO;
		if (src_periodic == false) {
			// detect if advection has made it become periodic.
			if ((adv0.x > GRADIENT_X_PER_Y*adv0.y)
					|| (adv1.x > GRADIENT_X_PER_Y*adv1.y)
					|| (adv2.x > GRADIENT_X_PER_Y*adv2.y))
			{
				src_periodic = true;
				adv0 = Anticlockwise*adv0;
				adv1 = Anticlockwise*adv1;
				adv2 = Anticlockwise*adv2;
				verts_use0.v = Anticlockwise3*verts_use0.v;
				verts_use1.v = Anticlockwise3*verts_use1.v;
				verts_use2.v = Anticlockwise3*verts_use2.v;
				// don't want to affect actual cell vars:				
				//Vars = *pVars;
				Vars.mom = Anticlockwise3*Vars.mom;
				//pVars = &Vars;
			};
			if ((adv0.x < -GRADIENT_X_PER_Y*adv0.y)
					|| (adv1.x < -GRADIENT_X_PER_Y*adv1.y)
					|| (adv2.x < -GRADIENT_X_PER_Y*adv2.y))
			{
				src_periodic = true;
			};
		} else {
			// ensure that we have a contiguous advected set on the left.
			
			// Just take the point(s) that were initially on right and put them
			// on left.
			// Regardless of how anything has wandered.
			
			if (pTri->periodic == 1) {
				o = pTri->GetLeftmostIndex();
				if (o != 0)	{					
					adv0 = Anticlockwise*adv0;
					verts_use0.v = Anticlockwise3*verts_use0.v;
				};
				if (o != 1)	{
					adv1 = Anticlockwise*adv1;
					verts_use1.v = Anticlockwise3*verts_use1.v;
				};
				if (o != 2) {
					adv2 = Anticlockwise*adv2;
					verts_use2.v = Anticlockwise3*verts_use2.v;				
				};
			} else {
				o = pTri->GetRightmostIndex();
				if (o == 0) {
					adv0 = Anticlockwise*adv0;
					verts_use0.v = Anticlockwise3*verts_use0.v;
				};
				if (o == 1) {
					adv1 = Anticlockwise*adv1;
					verts_use1.v = Anticlockwise3*verts_use1.v;				
				};
				if (o == 2) {
					adv2 = Anticlockwise*adv2;
					verts_use2.v = Anticlockwise3*verts_use2.v;				
				};
			};
		};
		
		// If pTri->periodic then cellvars did not need rotating.
		// It only needs doing in one case: that we crossed RH PB.
		
		// adv0,1,2 are now contiguous:
		advC = THIRD*(adv0 + adv1 + adv2); // unwrapped and unbounced so this is OK
		
		cpPlane[0].Clear();
		cpPlane[0].add(adv0);			cpPlane[0].add(adv1);			cpPlane[0].add(advC);
		cpPlane[1].Clear();
		cpPlane[1].add(adv1);			cpPlane[1].add(adv2);			cpPlane[1].add(advC);
		cpPlane[2].Clear();
		cpPlane[2].add(adv0);			cpPlane[2].add(adv2);			cpPlane[2].add(advC);
		
		cpTemp.Clear();
		cpTemp.add(adv0);				cpTemp.add(adv1);				cpTemp.add(adv2);
		Areanew = cpTemp.GetArea();
		

		// Compressive heating:

		// First increase heat integral nT, per the change in scale, so that we are fitting updating nT to updated heat.
		// n should be increased per scale factor area/areanew and nT should of course be increased by that to 5/3
		// Improvement: apply compressive heating on mesh, re-estimate nT on advected mesh, before we run this routine
		// so avoiding any scaling happening here
		// Come back to it. This time around stick with doing comp htg here locally.

		if (bDoCompressiveHeating)
		{
			compression_factor = pow(pTri->area/Areanew,TWOTHIRDS);
			verts_use0.T *= compression_factor;
			verts_use1.T *= compression_factor;
			verts_use2.T *= compression_factor; // and this is why we made a separate variable, amongst other reasons.
			verts_c.T *= compression_factor; 		
			
			compressive_heating_total += Vars.heat*(compression_factor-1.0);

			Vars.heat *= compression_factor;

			
		};

		// Area/Areanew is applied to n, nT during Triplanar. So 2/3 for T does the job here.

		// We could not call triplanar until v are all contiguous:
		Triplanar( &Vars, &verts_use0, &verts_use1, &verts_use2, &verts_c ,
											pTri->area,Areanew, // to compress both n,nT ...
											code );
		
		// Note that this may have amended verts_use0,1,2 as well as c, nowadays.

		// verts_use, verts_c now being populated, we proceed to apply these planes to the new mesh:

		// vertex data pointers:
		pvertvars[0][0] = &verts_use0;
		pvertvars[0][1] = &verts_use1;
		pvertvars[0][2] = &verts_c;
		
		pvertvars[1][0] = &verts_use1;
		pvertvars[1][1] = &verts_use2;			
		pvertvars[1][2] = &verts_c;
		
		pvertvars[2][0] = &verts_use0;
		pvertvars[2][1] = &verts_use2;
		pvertvars[2][2] = &verts_c;
			
		Triangle * pTriSeed;
		Vector2 AdvectedCentre(advC.x,advC.y);
		real debugAttributedMass = 0.0;

		if (AdvectedCentre.x < -GRADIENT_X_PER_Y*AdvectedCentre.y) 
			AdvectedCentre = Clockwise*AdvectedCentre;
		if (AdvectedCentre.x > GRADIENT_X_PER_Y*AdvectedCentre.y) 
			AdvectedCentre = Anticlockwise*AdvectedCentre;

		if (pTri-T < pDestMesh->numTriangles)
		{
			pTriSeed = pDestMesh->ReturnPointerToTriangleContainingPoint
				(pDestMesh->T + (pTri-T), 
				AdvectedCentre.x, AdvectedCentre.y);
		} else {
			pTriSeed = pDestMesh->ReturnPointerToTriangleContainingPoint
				(pDestMesh->T + pDestMesh->numTriangles-1,
				AdvectedCentre.x, AdvectedCentre.y);
		};
		
		for (int iPlane = 0; iPlane < numPlanes; iPlane++)
		{
			// For placing near insulator, allow that anything is in, radially. !!
			// &?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?&?
		
			debugAttributedMass += 
				pDestMesh->SendAllMacroscopicPlanarTriangle(
									cpPlane[iPlane], // convex polygon to send
									pTriSeed, // where to start looking
									src_periodic, // false positives are acceptable
									
									// corner vars to create variable gradients:
									pvertvars[iPlane][0],pvertvars[iPlane][1],pvertvars[iPlane][2],
									which_species, code);
		};		
		
		++pTri;
	};

	if (!bSwitchOffChPatReport)
	{
		fp = fopen(FUNCTIONALFILENAME, "a");
		fprintf(fp, "chPAT %1.11E ",compressive_heating_total);
		fclose(fp);
	};

}

// Remember to go back to mesh merge and ensure edge verts are put back to edge.

real TriMesh::SendAllMacroscopicPlanarTriangle(ConvexPolygon & cp,Triangle * pTriSeed, int src_periodic,
									 vertvars * pvertvars0, vertvars * pvertvars1, vertvars * pvertvars2,
									 int species, int code)
{
	// use the vertex vars and assume linear :
	int i;
	FILE * fp;
	Vector2 origin, pt1, pt2, centre;
	Triangle * pTriSeed1, * pTriSeed2, * pTri;
	cellvars * pDestCellvars;
	real momx, momy;
	char buffer[512];
	vertvars morevars0, morevars1,morevars2;
	real vals1[5];
	real vals2[5];
	real vals3[5];

	// For placement during diffusion:
	real nT0, nT1, nT2, heat_sending;


	real result= 0.0;

	if (src_periodic == PERIODIC_SRC)
	{
		// Given that it's periodic, it appears on the left.
		// So let's map to the right also, to do the split

		ConvexPolygon cp1,cp2,cp3;

		real area_whole = cp.GetArea();

		if (area_whole == 0.0)
		{
			area_whole = area_whole;
		} else {
			cp1.CopyFrom(cp);
			cp2.CopyFrom(cp);
			for (i = 0; i < cp2.numCoords; i++)
			{
				cp2.coord[i] = Clockwise*cp2.coord[i];
			};
			morevars0 = *pvertvars0;
			morevars0.v = Clockwise3*morevars0.v;
			morevars1 = *pvertvars1;
			morevars1.v = Clockwise3*morevars1.v;
			morevars2 = *pvertvars2;
			morevars2.v = Clockwise3*morevars2.v;		

			// We avoid interpolating to the edge of tranche - that would be messy.
			// Instead we send another src_periodic flag that says "clip this against the tranche when we do intersections".
			
			// NOTE THE MEANING OF MOM IN A BOUNDARY CELL is that it's the momentum that appears at the left PB.
			// Therefore if this is the cell contributing to itself, hopefully the mom is going to get rotated back again.
			
			origin.x = 0.0;
			origin.y = 0.0;
			pt1.y = 10.0;
			pt1.x = 10.0*(-GRADIENT_X_PER_Y);
			pt2.y = 10.0;
			pt2.x = 10.0*GRADIENT_X_PER_Y;
			// Note behaviour of ClipAgainstHalfplane: if it returns false that means we did not actually make changes to the data.
			
			if (!cp1.ClipAgainstHalfplane(origin, pt1,pt2))
			{
				cp1.numCoords = 0;
			} else {
				// now compare cp1 to cp to get intercepts. Isn't this tricky?
				// Still send cp
				// Just use cp1 for centre and pTriSeed
				
				cp1.GetCentre(centre);
				pTriSeed1 = ReturnPointerToTriangleContainingPoint(pTriSeed,centre.x,centre.y);				
				GlobalAreaApportioned = 0.0;
				result += SendAllMacroscopicPlanarTriangle(cp, pTriSeed1, CLIP_AGAINST_TRANCHE, 
					pvertvars0, pvertvars1, pvertvars2,
					species, code);
			};

			// Note that if there ARE viable periodic images, we reduce the "coefficient" because otherwise,
			// we will then take GetArea and distribute full mass over the clipped area (and so double-count, counting fully on each side in this tranche)

			cp3.CopyFrom(cp2);
			if (!cp2.ClipAgainstHalfplane(origin,pt2,pt1))
			{
				cp2.numCoords = 0;
			} else {
				cp2.GetCentre(centre);
				pTriSeed2 = ReturnPointerToTriangleContainingPoint(pTriSeed,centre.x,centre.y);
				GlobalAreaApportioned = 0.0;
				result += SendAllMacroscopicPlanarTriangle(cp3, pTriSeed2, CLIP_AGAINST_TRANCHE,	
					&morevars0,&morevars1,&morevars2, species, code);
			};
		};
		// NOTE THE MEANING OF MOM IN A BOUNDARY CELL is that it's the momentum that appears at the left PB.
		// Regardless of where it comes from, it needs to be twisted when it lands there at the right PB.
	} else {
		
		// not periodic src. (though pTriSeed can point to periodic)
		GlobalTrisVisited.clear();
		GlobalAreaApportioned = 0.0;
		real area = cp.GetArea();
		if (area < 1.0e-16) return 0.0; // got no use for snippets - 1e-4 micron x 1e-4 micron.

		//nesting = 0;
		int found_intersection = 0;

		Vector2 u[3];
		real factor,factor2;
		real intersection,intersection2;
		intersection2 = 0.0;

		bool left_intersected, right_intersected, intersected, intersectbase;
		ConvexPolygon cpIntersection, cpIntersectionLeft, cpIntersectionRight,cpDest, cpQuad,
			cpIntersectUnderBase;
		real integrals[5];
		int first,iCorner;

		GlobalTrisToVisit.clear();
		GlobalTrisToVisit.add(pTriSeed-T);
		int search = 0; // to avoid list errors, would be better to find a way to clear back the items that have been processed.
		int caret = 0;
		while (GlobalTrisToVisit.len > caret)
		{
			pTri = T + GlobalTrisToVisit.ptr[caret];
			intersectbase = false;

			if (pTri->indicator == 0)
			{
				// Calculate intersection here:
					
			//	if (pTri->flags == 0)
			//	{
					if (pTri->periodic == 0)
					{
						pTri->PopulatePositions(u[0],u[1],u[2]);

						intersected = cp.GetIntersectionWithTriangle(&cpIntersection,u[0],u[1],u[2]);
						
						// Concern to map inside this tri if it tried to map inside the bottom of it, for an insulator base tri.
						// Created function GetIntersectionWithPolygon.
						// Problem is it's not guaranteed that we will get convex polygon by adding a quad to the bottom of a base tri;
						// apex could be off-base. Consequently we do a separate quadrilateral:
						if (intersected && (pTri->flags == 6)) {
							first = 1;
							for (iCorner =0 ; iCorner < 3; iCorner++)
							{
								if (pTri->cornerptr[iCorner]->flags == 3)
								{
									if (first)
									{
										cpQuad.coord[0] = u[iCorner];  // change to u[] notation in this function.
										first = 0;
									} else {
										cpQuad.coord[1] = u[iCorner];
									};
								};
							};
							cpQuad.coord[1].project_to_radius(cpQuad.coord[2],1.0);
							cpQuad.coord[0].project_to_radius(cpQuad.coord[3],1.0);
							cpQuad.numCoords = 4;
							intersectbase = cp.GetIntersectionWithPolygon(&cpIntersectUnderBase,&cpQuad);
							
						}; // whether base tri and intersected
					} else {
						// periodic tri dest: 				

						// &?&?&?&?&?&&?&?&?&?&&?&?&?&?&?&?&?&?&?&?&?&&?&?&?&?&?&?&?&?&?
						// No time right now to worry about the extra quad underneath a periodic base triangle.
						// It's not hard to add. Come back for it.

						pTri->MapLeft(u[0],u[1],u[2]);
							
						cpDest.Clear();
						cpDest.add(u[0]);
						cpDest.add(u[1]);
						cpDest.add(u[2]);
						left_intersected = cpDest.GetIntersectionWithTriangle(&cpIntersection, cp.coord[0],cp.coord[1],cp.coord[2]);
						if (left_intersected) 
						{
							// do not need to consider intersection on right hand side as well
							intersected = true;

						} else {
							pTri->MapRight(u[0],u[1],u[2]);					
							cpDest.Clear();
							cpDest.add(u[0]);
							cpDest.add(u[1]);
							cpDest.add(u[2]); // note: only dest positions are being rotated, not src

							intersected = cpDest.GetIntersectionWithTriangle(&cpIntersection, cp.coord[0],cp.coord[1],cp.coord[2]);
						};
						// clip intersection against tranche if necessary:
						if ((intersected) && (src_periodic == CLIP_AGAINST_TRANCHE)) 
						{						
							origin.x = 0.0;
							origin.y = 0.0;
							pt1.y = 10.0;
							pt1.x = 10.0*(-GRADIENT_X_PER_Y);
							pt2.y = 10.0;
							pt2.x = 10.0*GRADIENT_X_PER_Y;
							if (cpIntersection.ClipAgainstHalfplane(origin,pt2,pt1) == 0)
							{
								intersected = false;
							};
							if (cpIntersection.ClipAgainstHalfplane(origin,pt1,pt2) == 0)
							{
								intersected = false;
							};
						};
					};				
				//} else {
				//	// Destination is wedge.

				//	if (pTri->periodic == 0)
				//	{
				//		pTri->cornerptr[0]->PopulatePosition(u0);
				//		pTri->cornerptr[1]->PopulatePosition(u1);
				//		if (pTri->flags == 1)
				//		{

				//			// We use a large wedge going inwards through the insulator.
				//			// This is because we do not believe we ever send anything that is not clipped against the domain annulus.

				//			pTri->cornerptr[0]->project_to_radius(u2,DEVICE_RADIUS_INSULATOR_OUTER*0.5);
				//			pTri->cornerptr[1]->project_to_radius(u3,DEVICE_RADIUS_INSULATOR_OUTER*0.5);
				//		} else {
				//			pTri->cornerptr[0]->project_to_radius(u2,HIGH_WEDGE_OUTER_RADIUS);
				//			pTri->cornerptr[1]->project_to_radius(u3,HIGH_WEDGE_OUTER_RADIUS);
				//		};

				//		cpDest.Clear();
				//		cpDest.add(u0);
				//		cpDest.add(u1);
				//		cpDest.add(u3);
				//		cpDest.add(u2); // sequence important
				//		intersected = cpDest.GetIntersectionWithTriangle(&cpIntersection, cp.coord[0],cp.coord[1],cp.coord[2]);
				//		
				//		
				//	} else {
				//		// periodic wedge dest

				//		int o = pTri->GetLeftmostIndex();
				//		pTri->cornerptr[o]->PopulatePosition(u0);
				//		pTri->cornerptr[1-o]->periodic_image(u1,0,1);
				//		if (pTri->flags == 1)
				//		{
				//			//pTri->cornerptr[o]->project_to_ins(u2);
				//			//pTri->cornerptr[1-o]->project_to_ins_periodic(u3,0,1);
				//			pTri->cornerptr[o]->project_to_radius(u2,DEVICE_RADIUS_INSULATOR_OUTER*0.5);
				//			pTri->cornerptr[1-o]->project_to_radius_periodic(u3,DEVICE_RADIUS_INSULATOR_OUTER*0.5,0,1);
				//		} else {
				//			pTri->cornerptr[o]->project_to_radius(u2,HIGH_WEDGE_OUTER_RADIUS);
				//			pTri->cornerptr[1-o]->project_to_radius_periodic(u3,HIGH_WEDGE_OUTER_RADIUS,0,1);
				//		};
				//		
				//		cpDest.Clear();
				//		cpDest.add(u0);
				//		cpDest.add(u1);
				//		cpDest.add(u3);
				//		cpDest.add(u2); // sequence important

				//		left_intersected = cpDest.GetIntersectionWithTriangle(&cpIntersection, cp.coord[0],cp.coord[1],cp.coord[2]);
				//		if (left_intersected)
				//		{
				//			intersected = true;
				//		} else {
				//			u0 = Clockwise*u0;
				//			u1 = Clockwise*u1;
				//			u2 = Clockwise*u2;
				//			u3 = Clockwise*u3;  // periodic dest appearing on RHS

				//			cpDest.Clear();
				//			cpDest.add(u0);
				//			cpDest.add(u1);
				//			cpDest.add(u3);
				//			cpDest.add(u2); // sequence important

				//			intersected = cpDest.GetIntersectionWithTriangle(&cpIntersection, cp.coord[0],cp.coord[1],cp.coord[2]);						
				//		};
				//		
				//		if ((intersected) && (src_periodic == CLIP_AGAINST_TRANCHE))
				//		{
				//			origin.x = 0.0;
				//			origin.y = 0.0;
				//			pt1.y = 10.0;
				//			pt1.x = 10.0*(-GRADIENT_X_PER_Y);
				//			pt2.y = 10.0;
				//			pt2.x = 10.0*GRADIENT_X_PER_Y;
				//			if (cpIntersection.ClipAgainstHalfplane(origin,pt2,pt1) == 0)
				//			{
				//				intersected = false;
				//			};
				//			if (cpIntersection.ClipAgainstHalfplane(origin,pt1,pt2) == 0)
				//			{
				//				intersected = false;
				//			};
				//		};
				//	};
				//};
				//
				if (intersected) {

					

					if ((code == NUMT) || (code == NUMV))
					{
						if (code == NUMT) {
							nT0 = pvertvars0->T*pvertvars0->n; // have to send that way :(
							nT1 = pvertvars1->T*pvertvars1->n;
							nT2 = pvertvars2->T*pvertvars2->n;
							
							cpIntersection.IntegrateMass(cp.coord[0],cp.coord[1],cp.coord[2],
										nT0, nT1, nT2, &(heat_sending));
							pTri->numerator_T += heat_sending;


							if (intersectbase) // add a tiny amt that tried to slip underneath base triangle:
							{
								cpIntersectUnderBase.IntegrateMass
											(cp.coord[0],cp.coord[1],cp.coord[2],
												nT0,nT1,nT2,&(heat_sending));

								pTri->numerator_T += heat_sending;	
							};
							
							// Maybe want to drop this and do reflection instead...


						} else {
							vals1[0] = pvertvars0->n;
							vals1[1] = pvertvars0->n*pvertvars0->T;
							vals1[2] = pvertvars0->n*pvertvars0->v.x;
							vals1[3] = pvertvars0->n*pvertvars0->v.y;
							vals1[4] = pvertvars0->n*pvertvars0->v.z;

							vals2[0] = pvertvars1->n;
							vals2[1] = pvertvars1->n*pvertvars1->T;
							vals2[2] = pvertvars1->n*pvertvars1->v.x;
							vals2[3] = pvertvars1->n*pvertvars1->v.y;
							vals2[4] = pvertvars1->n*pvertvars1->v.z;

							vals3[0] = pvertvars2->n;
							vals3[1] = pvertvars2->n*pvertvars2->T;
							vals3[2] = pvertvars2->n*pvertvars2->v.x;
							vals3[3] = pvertvars2->n*pvertvars2->v.y;
							vals3[4] = pvertvars2->n*pvertvars2->v.z;

							// This is fairly unacceptable. Yet it is what we always do!
							
							cpIntersection.Integrate5Planes(cp.coord[0],cp.coord[1],cp.coord[2],
											vals1, vals2, vals3, integrals);
								
							pTri->numerator_x += integrals[2]; 
							pTri->numerator_y += integrals[3]; 
							pTri->numerator_z += integrals[4];

							if (intersectbase) // add a tiny amt that tried to slip underneath base triangle:
							{
								cpIntersectUnderBase.Integrate5Planes
											(cp.coord[0],cp.coord[1],cp.coord[2],
												vals1, vals2, vals3, integrals);

								pTri->numerator_x += integrals[2]; 
								pTri->numerator_y += integrals[3]; 
								pTri->numerator_z += integrals[4];
							};
						};
					} else {

						if (species == SPECIES_ION)
						{
							pDestCellvars = &(pTri->ion);					
						} else {
							if (species == SPECIES_NEUTRAL)
							{
								pDestCellvars = &(pTri->neut);
							} else {
								pDestCellvars = &(pTri->elec);
							};
						};

						if (code == MASS_ONLY)
						{
							vals1[0] = pvertvars0->n;
							vals2[0] = pvertvars1->n;
							vals3[0] = pvertvars2->n;
							cpIntersection.IntegrateMass(cp.coord[0],cp.coord[1],cp.coord[2],
										vals1[0], vals2[0], vals3[0], &(integrals[0]));
							pDestCellvars->mass += integrals[0];
							// DEBUG:
							result += integrals[0];

							if (intersectbase) // add a tiny amt that tried to slip underneath base triangle:
							{
								cpIntersectUnderBase.IntegrateMass(cp.coord[0],cp.coord[1],cp.coord[2],
											vals1[0], vals2[0], vals3[0], &(integrals[0]));
								pDestCellvars->mass += integrals[0];
								result += integrals[0];
							};
						} else {

							// calculate contributions
							vals1[0] = pvertvars0->n;
							vals1[1] = pvertvars0->n*pvertvars0->T;
							vals1[2] = pvertvars0->n*pvertvars0->v.x;
							vals1[3] = pvertvars0->n*pvertvars0->v.y;
							vals1[4] = pvertvars0->n*pvertvars0->v.z;

							vals2[0] = pvertvars1->n;
							vals2[1] = pvertvars1->n*pvertvars1->T;
							vals2[2] = pvertvars1->n*pvertvars1->v.x;
							vals2[3] = pvertvars1->n*pvertvars1->v.y;
							vals2[4] = pvertvars1->n*pvertvars1->v.z;

							vals3[0] = pvertvars2->n;
							vals3[1] = pvertvars2->n*pvertvars2->T;
							vals3[2] = pvertvars2->n*pvertvars2->v.x;
							vals3[3] = pvertvars2->n*pvertvars2->v.y;
							vals3[4] = pvertvars2->n*pvertvars2->v.z;
							// Look what a waste of time this is --
							// we ought to be just passing nv = p

							cpIntersection.Integrate5Planes(cp.coord[0],cp.coord[1],cp.coord[2],
											vals1, vals2, vals3, integrals);

							// add mass, heat, mom to the destination:
							pDestCellvars->mass += integrals[0];
							pDestCellvars->heat += integrals[1]; 
							pDestCellvars->mom.x += integrals[2]; 
							pDestCellvars->mom.y += integrals[3]; 
							pDestCellvars->mom.z += integrals[4];
	 
							GlobalMassAttributed += integrals[0];
							GlobalHeatAttributed += integrals[1];
							GlobalAreaApportioned += cpIntersection.GetArea();
							
							// DEBUG:
							result += integrals[0];
							
							if (intersectbase) // add a tiny amt that tried to slip underneath base triangle:
							{
								cpIntersectUnderBase.Integrate5Planes(cp.coord[0],cp.coord[1],cp.coord[2],
											vals1, vals2, vals3, integrals);
								pDestCellvars->mass += integrals[0];
								pDestCellvars->heat += integrals[1]; 
								pDestCellvars->mom.x += integrals[2]; 
								pDestCellvars->mom.y += integrals[3]; 
								pDestCellvars->mom.z += integrals[4];
								result += integrals[0];
								GlobalMassAttributed += integrals[0];
								GlobalHeatAttributed += integrals[1];
								GlobalAreaApportioned += cpIntersectUnderBase.GetArea();						
							};
						};
					};

					pTri->indicator = 1; // A searched tri where intersection was now found.
					
					if (pTri->neighbours[0]->indicator == 0)
						GlobalTrisToVisit.add(pTri->neighbours[0]-T);
					if (pTri->neighbours[1]->indicator == 0)
						GlobalTrisToVisit.add(pTri->neighbours[1]-T);
					if (pTri->neighbours[2]->indicator == 0)
						GlobalTrisToVisit.add(pTri->neighbours[2]-T);

					// Note that items may be added multiple times to the list until they are once processed, so
					// we do need to check above that pTri->indicator==0

					found_intersection = 1;
				} else {
					pTri->indicator = 2; // Looked and no intersection found.

					//	GlobalTrisVisited.add(pTri-T);
					if ((found_intersection == 0) && (search < 400)) // If first triangle tried did NOT show an intersection, LOOK AT NEIGHBOURS TO TRY TO FIND IT
						//|| ((GlobalAreaApportioned == 0.0) && (nesting < 20))) // try looking further randomly, for a while
					{			
						if (pTri->neighbours[0]->indicator == 0)
							GlobalTrisToVisit.add(pTri->neighbours[0]-T);
						if (pTri->neighbours[1]->indicator == 0)
							GlobalTrisToVisit.add(pTri->neighbours[1]-T);
						if (pTri->neighbours[2]->indicator == 0)
							GlobalTrisToVisit.add(pTri->neighbours[2]-T);
						search++;
					};
				};
			}; // pTri->indicator==0

			caret++; // next item on list

			// If we already apportioned whole thing, we should
			// really stop now for efficiency.
		};			

		
		if (((GlobalAreaApportioned > area*1.00001) || (GlobalAreaApportioned < area*0.99999)) &&
			(src_periodic != CLIP_AGAINST_TRANCHE))
		{
			caret=caret;
		};

		//if (varcode == VARCODE_MASS) GlobalMassReceived += (GlobalAreaApportioned/area)*coefficient*pVars->mass;

		// Now restore Triangle::indicator to zero for the visited triangles.
		for (int i = 0; i < GlobalTrisToVisit.len; i++)
		{
			pTri = T + GlobalTrisToVisit.ptr[i];
			pTri->indicator = 0;
		};

	}; // whether src_periodic

	return result;
}

void TriMesh::SendMacroscopicPolygon(
									 ConvexPolygon & cp,
									 Triangle * pTriSeed,	
									 bool src_periodic,		// if in doubt send as true
									 real coefficient,			// e.g. Lebesgue weight
									 cellvars * pVars,			// how much stuff lives in cp
									 int varcode)
{
	// First thing here is to deal with periodic.
		
	// If it's periodic, that means we should clip against domain, map to right and clip against domain.
	// For any periodic destination, we always count it on both sides.
	int i;
	FILE * fp;
	Vector2 origin, pt1, pt2, centre;
	Triangle * pTriSeed1, * pTriSeed2, * pTri;
	real momx, momy;
	char buffer[512];

	FILE * contribfile; // debug

	if (src_periodic)
	{
		// Given that it's periodic, it appears on the left.
		// So let's map to the right also, to do the split

		ConvexPolygon cp1,cp2;
		cellvars RotatedVars;

		RotatedVars.mass = pVars->mass;
		RotatedVars.heat = pVars->heat;

		real area_whole = cp.GetArea();

		if (area_whole == 0.0)
		{
			area_whole = area_whole;
		} else {
			cp1.CopyFrom(cp);
			cp2.CopyFrom(cp);
			for (i = 0; i < cp2.numCoords; i++)
			{
				cp2.coord[i] = Clockwise*cp2.coord[i];
				RotatedVars.mom.x = Clockwise.xx * pVars->mom.x + Clockwise.xy* pVars->mom.y;
				RotatedVars.mom.y = Clockwise.yx * pVars->mom.x + Clockwise.yy* pVars->mom.y;
				// NOTE THE MEANING OF MOM IN A BOUNDARY CELL is that it's the momentum that appears at the left PB.
				// Therefore if this is the cell contributing to itself, hopefully the mom is going to get rotated back again.
			};
			RotatedVars.mom.z = pVars->mom.z;

			origin.x = 0.0;
			origin.y = 0.0;
			pt1.y = 10.0;
			pt1.x = 10.0*(-GRADIENT_X_PER_Y);
			pt2.y = 10.0;
			pt2.x = 10.0*GRADIENT_X_PER_Y;
			// Note behaviour of ClipAgainstHalfplane: if it returns false that means we did not actually make changes to the data.

			if ((!cp1.ClipAgainstHalfplane(origin, pt1,pt2))
				||
				(!cp1.ClipAgainstHalfplane(origin, pt2, pt1))) // just for the sake of it
			{
				cp1.numCoords = 0;
			} else {
				cp1.GetCentre(centre);
				pTriSeed1 = ReturnPointerToTriangleContainingPoint(pTriSeed,centre.x,centre.y);
				
				GlobalAreaApportioned = 0.0;
				SendMacroscopicPolygon(cp1,pTriSeed1,false,coefficient * cp1.GetArea() / area_whole,pVars,varcode);
			//	if (varcode == VARCODE_MASS) GlobalMassReceived += (GlobalAreaApportioned/area_whole )*coefficient*pVars->mass;
			};

			// Note that if there ARE viable periodic images, we reduce the "coefficient" because otherwise,
			// we will then take GetArea and distribute full mass over the clipped area (and so double-count, counting fully on each side in this tranche)

			// Taking this approach with reflection may make it easier to allow for making shape smaller to reflect 
			// area between the ins curve and the hinge.

			if ((!cp2.ClipAgainstHalfplane(origin,pt2,pt1))
				||
				(!cp2.ClipAgainstHalfplane(origin,pt1,pt2))) // just for the sake of it
			{
				cp2.numCoords = 0;
			} else {
				cp2.GetCentre(centre);
				pTriSeed2 = ReturnPointerToTriangleContainingPoint(pTriSeed,centre.x,centre.y);
				GlobalAreaApportioned = 0.0;
				SendMacroscopicPolygon(cp2,pTriSeed2,false,coefficient * cp2.GetArea() / area_whole,&RotatedVars,varcode);

			//	if (varcode == VARCODE_MASS) GlobalMassReceived += (GlobalAreaApportioned/area_whole)*coefficient*pVars->mass;
			};
		};
		// BUT, NOTE THE MEANING OF MOM IN A BOUNDARY CELL is that it's the momentum that appears at the left PB.
		// Regardless of where it comes from, it needs to be twisted when it lands there at the right PB.
	} else {
		
		// not periodic src. (though pTriSeed can point to periodic)

		// Set up call to recursive search function:

		// Do recursive calls:
		GlobalTrisVisited.clear();
		GlobalAreaApportioned = 0.0;
		real area = cp.GetArea();
		if (area < 1.0e-16) return; // got no use for snippets - 1e-4 micron x 1e-4 micron.

		//nesting = 0;
		int found_intersection = 0;

		// Let's change to not do recursive calls.

		//SearchIntersectionsForPolygon(cp, pTriSeed, coefficient, pVars, varcode, area);
		
		Vector2 u0,u1,u2,u3;
		real factor,factor2;
		real intersection,intersection2;

		intersection2 = 0.0;
	
	// The way we will do this: have a scrolling list
	// Each investigated triangle adds its neighbours to the list unless they have pTri->indicator

		GlobalTrisToVisit.clear();
		GlobalTrisToVisit.add(pTriSeed-T);
		int search = 0; // to avoid list errors, would be better to find a way to clear back the items that have been processed.
		int caret = 0;
		while (GlobalTrisToVisit.len > caret)
		{
			pTri = T + GlobalTrisToVisit.ptr[caret];			

			if (pTri->indicator == 0)
			{
				// Calculate intersection here:
					
			//	if (pTri->flags == 0)
			//	{
					if (pTri->periodic == 0)
					{
						pTri->cornerptr[0]->PopulatePosition(u0);
						pTri->cornerptr[1]->PopulatePosition(u1);
						pTri->cornerptr[2]->PopulatePosition(u2);

						intersection = cp.FindTriangleIntersectionArea(u0,u1,u2);
					} else {
						// periodic tri dest: 
						
						pTri->MapLeft(u0,u1,u2);
						intersection = cp.FindTriangleIntersectionArea(u0,u1,u2);
						pTri->MapRight(u0,u1,u2);					
						intersection2 = cp.FindTriangleIntersectionArea(u0,u1,u2); // here mom must be rotated because the dest is special
					};
				//} else {
				//	// Destination is wedge.

				//	if (pTri->periodic == 0)
				//	{
				//		pTri->cornerptr[0]->PopulatePosition(u0);
				//		pTri->cornerptr[1]->PopulatePosition(u1);
				//		if (pTri->flags == 1)
				//		{
				//			pTri->cornerptr[0]->project_to_ins(u2);
				//			pTri->cornerptr[1]->project_to_ins(u3);
				//		} else {
				//			pTri->cornerptr[0]->project_to_radius(u2,HIGH_WEDGE_OUTER_RADIUS);
				//			pTri->cornerptr[1]->project_to_radius(u3,HIGH_WEDGE_OUTER_RADIUS);
				//		};

				//		intersection = cp.FindQuadrilateralIntersectionArea(u0,u1,u3,u2); // sequence important
				//	} else {
				//		// periodic wedge dest

				//		int o = pTri->GetLeftmostIndex();
				//		pTri->cornerptr[o]->PopulatePosition(u0);
				//		pTri->cornerptr[1-o]->periodic_image(u1,0,1);
				//		if (pTri->flags == 1)
				//		{
				//			pTri->cornerptr[o]->project_to_ins(u2);
				//			pTri->cornerptr[1-o]->project_to_ins_periodic(u3,0,1);
				//		} else {
				//			pTri->cornerptr[o]->project_to_radius(u2,HIGH_WEDGE_OUTER_RADIUS);
				//			pTri->cornerptr[1-o]->project_to_radius_periodic(u3,HIGH_WEDGE_OUTER_RADIUS,0,1);
				//		};
				//		
				//		intersection = cp.FindQuadrilateralIntersectionArea(u0,u1,u3,u2); 

				//		u0 = Clockwise*u0;
				//		u1 = Clockwise*u1;
				//		u2 = Clockwise*u2;
				//		u3 = Clockwise*u3;  // periodic dest appearing on RHS

				//		intersection2 = cp.FindQuadrilateralIntersectionArea(u0,u1,u3,u2); 
				//	};
				//};
				
		
				if ((intersection > 0.0) || (intersection2 > 0.0))
				{
					pTri->indicator = 1;
				//	GlobalTrisVisited.add(pTri-T);

					if (area > 0.0) // sometimes get called with stupid values of corners on top of each other?
					{
						factor = intersection / area;		
				
						switch(varcode)
						{ 
						case VARCODE_MASS:
							break;
						case VARCODE_HEAT:

							pTri->numerator_T += factor*coefficient * pVars->heat;

							GlobalHeatReceived += factor*coefficient * pVars->heat;

							if (intersection2 > 0.0){
								pTri->numerator_T += (intersection2/area)*coefficient*pVars->heat;
								GlobalHeatReceived += (intersection2/area)*coefficient*pVars->heat;
							};


							break;
						default:


							//// DEBUG:

							//if ((GlobalStepsCounter == 154) && (pTri-T == 1896))
							//{
							//	contribfile = fopen("contrib154_1896_ii.txt","a");

							//	fprintf(contribfile,"iSrc  species  Momy+  Mass+  iPlane  iShape  intersection  area"
							//		" cellmomy cellmass planemomy planemass"
							//		" cum_momy cum_mass pVars->mom.y pVars->mass \n");
							//	fprintf(contribfile,"%d  %d %1.10E  %1.10E  %d  %d  %1.10E %1.10E "
							//		" %1.10E  %1.10E  %1.10E %1.10E "
							//		" %1.10E %1.10E %1.10E %1.10E \n",
							//		GlobaliSrcTri, GlobalSpecies, factor*coefficient * pVars->mom.y, factor*coefficient*pVars->mass,GlobaliSrcPlane,GlobaliSrcShape, intersection, area, 
							//		GlobalSrcMomy, GlobalSrcMass, GlobalPlaneMomy,GlobalPlaneMass,
							//		pTri->numerator_y, pTri->denominator, pVars->mom.y,pVars->mass);


							//	fclose(contribfile);
							//}


							pTri->numerator_x += factor*coefficient * pVars->mom.x;
							pTri->numerator_y += factor*coefficient * pVars->mom.y;
							pTri->numerator_z += factor*coefficient * pVars->mom.z;
							
							if (intersection2 > 0.0) {
								factor2 = intersection2*coefficient / area;
								// Have to rotate anticlockwise to be the momentum here;
						
								// We rotated its own mom clockwise when we made the right image.
								// So rotate back again if things want to stay about the same - makes sense
						
								pTri->numerator_x += Anticlockwise.xx*factor2*pVars->mom.x + Anticlockwise.xy*factor2*pVars->mom.y;
								pTri->numerator_y += Anticlockwise.yx*factor2*pVars->mom.x + Anticlockwise.yy*factor2*pVars->mom.y;
								pTri->numerator_z += factor2*pVars->mom.z;

								// Need to be careful about this.

							};
							break;
						};
						pTri->denominator += factor*coefficient * pVars->mass;// have to quickly go through and infer mass afterwards then.
						GlobalMassReceived += factor*coefficient * pVars->mass;

						if (intersection2 > 0.0) {
							pTri->denominator += (intersection2/area)*coefficient*pVars->mass;				
							GlobalMassReceived += (intersection2/area)*coefficient*pVars->mass;				
						};
						GlobalAreaApportioned += intersection + intersection2;
					};
									
					// if we have not visited neighbours, add them

					if (pTri->neighbours[0]->indicator == 0)
						GlobalTrisToVisit.add(pTri->neighbours[0]-T);
					if (pTri->neighbours[1]->indicator == 0)
						GlobalTrisToVisit.add(pTri->neighbours[1]-T);
					if (pTri->neighbours[2]->indicator == 0)
						GlobalTrisToVisit.add(pTri->neighbours[2]-T);

					found_intersection = 1;
				} else {
					pTri->indicator = 2;
				//	GlobalTrisVisited.add(pTri-T);
					if ((found_intersection == 0) && (search < 200)) // If first triangle tried did NOT show an intersection, LOOK AT NEIGHBOURS TO TRY TO FIND IT
						//|| ((GlobalAreaApportioned == 0.0) && (nesting < 20))) // try looking further randomly, for a while
					{			
						if (pTri->neighbours[0]->indicator == 0)
							GlobalTrisToVisit.add(pTri->neighbours[0]-T);
						if (pTri->neighbours[1]->indicator == 0)
							GlobalTrisToVisit.add(pTri->neighbours[1]-T);
						if (pTri->neighbours[2]->indicator == 0)
							GlobalTrisToVisit.add(pTri->neighbours[2]-T);
						search++;
					};
				};
			};// pTri->indicator == 0
			caret++; // next item on list
		};		

		//if (varcode == VARCODE_MASS) GlobalMassReceived += (GlobalAreaApportioned/area)*coefficient*pVars->mass;

		// Now restore Triangle::indicator to zero for the visited triangles.
		for (int i = 0; i < GlobalTrisToVisit.len; i++)
		{
			pTri = T + GlobalTrisToVisit.ptr[i];
			pTri->indicator = 0;
		};
	};
}

void TriMesh::CopyMesh(TriMesh * pDestMesh)
{
	// copy vertex position and mesh structure; 
	// regenerate neighbour lists
	
	Vertex * XdestX = pDestMesh->X;
	Triangle * XdestT = pDestMesh->T;

	Vertex * pVertex = X;
	Vertex * pVertDest = pDestMesh->X;
	Triangle * pTri = T;
	Proto ** ptr;
	Triangle * pTridest = pDestMesh->T;
	real rr;
	long iVertex;
	real weight1,weight2;
	long iTri, i;
	long * pLong;
	Vector2 temp;
	
	pDestMesh->numTriangles = numTriangles;
	pDestMesh->numVertices = numVertices;
	pDestMesh->numEdgeVerts = numEdgeVerts;
	pDestMesh->Outermost_r_achieved = Outermost_r_achieved;
	
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		pVertDest->x = pVertex->x;
		pVertDest->y = pVertex->y;

		pVertDest->triangles.clear();
		ptr = pVertex->triangles.ptr;
		for (int iii = 0; iii < pVertex->triangles.len; iii++)
		{
			pVertDest->triangles.add(XdestT + (((Triangle *)(*ptr))-T));
			++ptr;
		};

		pVertDest->neighbours.clear();
		pLong = pVertex->neighbours.ptr;
		for (int iii = 0; iii < pVertex->neighbours.len; iii++)
		{
			pVertDest->neighbours.add(*pLong);
			++pLong;
		}; // smartlong should simply have copy function.

		pVertDest->iScratch = pVertex->iScratch;
		pVertDest->flags = pVertex->flags;
		pVertDest->has_periodic = pVertex->has_periodic;
		//pVertDest->iTriSpare = pVertex->iTriSpare;
		
		pVertDest->A = pVertex->A;
		pVertDest->phi = pVertex->phi; // try to keep passing on as well.
		
		++pVertDest;
		++pVertex;
	};
	
	// *********************************************************

	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++) // numTriangles same in every mesh
	{
		// copy over:
			
		// Note : pTri->cornerptr[0]  +  (Xdest.ion.T-ion.T)
		// would quite possibly give the wrong answer
		// we want
		// Xdest.ion.T + (pTri->cornerptr[0] - ion.T)
		
		pTridest->cornerptr[0] = XdestX + (pTri->cornerptr[0] - X);
		pTridest->cornerptr[1] = XdestX + (pTri->cornerptr[1] - X);
		pTridest->cornerptr[2] = XdestX + (pTri->cornerptr[2] - X);

		pTridest->flags = pTri->flags;
		
		pTridest->neighbours[0] = XdestT + (pTri->neighbours[0] - T);
		pTridest->neighbours[1] = XdestT + (pTri->neighbours[1] - T);
		pTridest->neighbours[2] = XdestT + (pTri->neighbours[2] - T);
		
		pTridest->periodic = pTri->periodic;
				
		++pTri;
		++pTridest;
	};
	// *********************************************************

	// Now we need to recalculate triangle transverse vectors for doing overlap tests and other stuff:

	pTri = pDestMesh->T;
	for (iTri = 0; iTri < pDestMesh->numTriangles; iTri++)
	{
		pTri->RecalculateEdgeNormalVectors(false);
		pTri->area = pTri->GetArea();
		++pTri;
	};

	pDestMesh->RefreshVertexNeighboursOfVerticesOrdered();
}


void TriMesh::ZeroVertexPositions()
{
	Vertex * pVertex = X;
	for (int i = 0; i < numVertices; i++)
	{
		pVertex->x = 0.0;
		pVertex->y = 0.0;
		++pVertex;
	};
};

void TriMesh::ZeroCellData()
{
	// This is so that ion.heat etc can be repopulated with contributions afterwards.

	Triangle * pTri = T;
	for (int iTri = 0; iTri < numTriangles; iTri++)
	{
		ZeroMemory(&(pTri->ion),sizeof(cellvars));
		ZeroMemory(&(pTri->neut),sizeof(cellvars));
		ZeroMemory(&(pTri->elec),sizeof(cellvars));
		++pTri;
	};
};

	
struct LLelement
{
	long indexnext;
	long iTri;
	int rela;
	real value;
};
class LinkedList
{
public:
	LLelement * ptr;
	int start, end; // index of first usable element and last element
	int len, dim;
	// the idea is we want to keep list of triangles in ascending order of "value"
	LinkedList()
	{
		len = 0;
		dim = 0;
		ptr = 0;
		start = 0;
		end = 0;
	}
	void Add(long iTri, real value, int rela)
	{
		int newdim;
		// first redimension
		if (len == dim)
		{
			newdim = dim+128;
			ptr = (LLelement *)realloc(ptr,sizeof(LLelement)*newdim);
			if (ptr == 0) { printf("\n\nmemory error\n"); getch(); };
			dim = newdim;
		}
		ptr[len].iTri = iTri;
		ptr[len].rela = rela;
		ptr[len].value = value;

		// First compare to last element. If it is inside this , can go forward from "start"... 			
		if (value >= ptr[end].value) {
			// we hope that if end == 0 and len == 0 then we always go here. Comparing memory to itself should give == ??
			end = len;
			ptr[len].indexnext = len+1; // never had to make changes to the rest
		} else {
			if (value < ptr[start].value) {
				ptr[len].indexnext = start;
				start = len;
				ptr[end].indexnext = len+1;
			} else {			
				int index = start;
				while (value > ptr[ptr[index].indexnext].value)
					index = ptr[index].indexnext;
				ptr[len].indexnext = ptr[index].indexnext;
				ptr[index].indexnext = len;
				ptr[end].indexnext = len+1;	
			};	
		};
		len++;
	}
	void Clear()
	{
		if (dim > 0) free(ptr);
		len = 0;
		dim = 0;
		ptr = 0;
		start = 0;
		end = 0;
	};
	~LinkedList()
	{
		Clear();
	}
	int ShiftStart()
	{
		if (ptr)
		{
			// shift start onwards 1. Don't bother changing memory allocations, just let it grow until we are finished.		
			start = ptr[start].indexnext;
			// watch out for this going past the end.
			if (start >= len) {
				//Clear();
				printf("warning - array used up");
				return 1;
			};
			return 0;
			// be careful -- we might be adding elements and trying to remove the first one, end up removing one of our new ones.
			// So have to remove self first and this may sometimes indeed result in clearing the array.
		} else {
			printf("tried to remove element from non-existent linked list.\n");
			return 2;
		};

		// note that 0 to len-1 are still the populated elements.
	}
	void ReturnFirstData(long * piTri, int * prela)
	{
		if (start == len) {
			printf("read past end of linked list.\n");
			getch();
		} else {
			*piTri = ptr[start].iTri;
			*prela = ptr[start].rela;
		};
	}
};


real Triangle::GetPossiblyPeriodicDistCentres(Triangle * pTri, int * prela)
{
	// check distance from this to pTri centre, looking rotated each way.

	Vector2 cc1,cc2;

	this->ReturnCentre(&cc1,this);
	pTri->ReturnCentre(&cc2,pTri);	// per tri will put centre on left, note bene

	// we return -1 for prela if pTri's closest image is anticlockwise relative to us

	Vector2 anti = Anticlockwise*cc2;
	Vector2 clock = Clockwise*cc2;
	real antidist = (cc1-anti).modulus();
	real dist = (cc1-cc2).modulus();
	real clockdist = (cc1-clock).modulus();
		
//	if ((periodic == 0) && (pTri->periodic == 0))
//	{
	if (antidist < dist) {
		*prela = -1;
		return antidist;	// rotate it anticlockwise rel to us so we'll rotate mom clockwise to be rel to it
	} else {
		if (dist < clockdist) {
			*prela = 0;
			return dist;
		} else {
			*prela = 1;
			return clockdist;
		};
	};
//	}; 
	// consider this further:

	//if ((pTri->periodic > 0) && (periodic == 0))
	//{
	//	// guess that it's either the mapped-left position or shifted clockwise

	//	if (dist < clockdist) {
	//		// mapped-left position is closest
	//		// That basically suggests we should not do any relative rotation
	//		// since in this case we are also left of zero, it stands to reason

	//		*prela = 0;
	//		return dist;
	//	} else {
	//		*prela = 1;   // our stuff appears rotated anticlockwise
	//		return clockdist;
	//	};
	//};
	// conclusion - we don't actually need to do any special code for periodic.



}

bool inline Triangle::has_vertex(Vertex * pVertex)
{
	return ((cornerptr[0] == pVertex) || (cornerptr[1] == pVertex) || (cornerptr[2] == pVertex));
};

bool inline SlimTriangle::has_vertex(SlimVertex * pVertex)
{
	return ((cornerptr[0] == pVertex) || (cornerptr[1] == pVertex) || (cornerptr[2] == pVertex));
};

real TriMesh::GetCircumcenterDistance(Triangle * pTri, Triangle * pTri2)
{
	Vector2 Use;

	// DEBUG:
	if (
		((pTri->flags == 0) && (pTri2->flags == 1))
		||
		((pTri->flags == 1) && (pTri2->flags == 0))
		) {
			pTri = T; pTri2 = pTri->neighbours[2];
	};

	// Triangle to wedge uses same formulation as triangle; position circumcenter for wedge appropriately.
	// We put it on a level with the bottom of the top triangle.
	// Bad?
	// Doesn't matter much how close ....
	// 
	if ( (pTri->flags == 0) || (pTri2->flags == 0) )
	{
		if ( (pTri->periodic > 0) && (pTri2->cc.x > 0.0))
		{
			Use = Anticlockwise*pTri2->cc;
			return (Use-pTri->cc).modulus();
		};
		if ( (pTri->cc.x > 0.0) && (pTri2->periodic > 0) )
		{
			Use = Anticlockwise*pTri->cc;
			return (pTri2->cc-Use).modulus();
		};
		return (pTri->cc-pTri2->cc).modulus();	
	};
	
	// two wedges
	Vector2 Use2, cc1, cc2, r, theta;

	pTri->ReturnCentre(&Use, pTri);
	pTri2->ReturnCentre(&Use2, pTri);
	r = Use+Use2;
	r = r/r.modulus();
	theta.y = -r.x;
	theta.x = r.y;
	return fabs(theta.dot(Use-Use2));
	
}	
void TriMesh::RestoreSpeciesTotals(TriMesh * pSrc)
{
	real factor_neut, factor_ion, factor_elec;
	long iTri;
	Triangle * pTri = T;
	real mass_neut_src = 0.0;
	real mass_ion_src = 0.0;
	real mass_elec_src = 0.0;
	real mass_neut = 0.0;
	real mass_ion = 0.0;
	real mass_elec = 0.0;

	pTri = pSrc->T;
	for (iTri = 0; iTri < pSrc->numTriangles; iTri++)
	{
		mass_neut_src += pTri->neut.mass;
		mass_ion_src += pTri->ion.mass;
		mass_elec_src += pTri->elec.mass;
		++pTri;
	};
	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		mass_neut += pTri->neut.mass;
		mass_ion += pTri->ion.mass;
		mass_elec += pTri->elec.mass;
		++pTri;
	};
	
	factor_neut = mass_neut_src/mass_neut;
	factor_ion = mass_ion_src/mass_ion;
	factor_elec = mass_elec_src/mass_elec;

	printf("Rescale factors: 1+ \nneut: %1.6E ion: %1.6E e: %1.6E \n",
		factor_neut-1.0,factor_ion-1.0, factor_elec-1.0);

	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		pTri->neut = factor_neut*pTri->neut; // temperature, velocity unaffected
		pTri->ion = factor_ion*pTri->ion;
		pTri->elec = factor_elec*pTri->elec;
		++pTri;
	};	

	ReportIonElectronMass();
}

void TriMesh::ReportIonElectronMass()
{
	long iTri;
	Triangle * pTri = T;
	real ionmass = 0.0;
	real emass = 0.0;
	real ionmass0 = 0.0;
	real emass0 = 0.0;
	real ionmass1 = 0.0;
	real emass1 = 0.0;
	real ionmass2 = 0.0;
	real emass2 = 0.0;
	real TotalCharge = 0.0;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		ionmass += pTri->ion.mass;
		emass += pTri->elec.mass;
		if (pTri->flags == 0)
		{
			ionmass0 += pTri->ion.mass;
			emass0 += pTri->elec.mass;
		};
		if (pTri->flags == 1)
		{
			ionmass1 += pTri->ion.mass;
			emass1 += pTri->elec.mass;
		};
		if (pTri->flags == 2)
		{
			ionmass2 += pTri->ion.mass;
			emass2 += pTri->elec.mass;
		};
		TotalCharge += (pTri->ion.mass-pTri->elec.mass);
		++pTri;
	};
	printf("ion mass: \n %1.5E  %1.5E  %1.5E  %1.13E \n",
		ionmass0,ionmass1,ionmass2,ionmass);
	printf("elec mass: \n %1.5E  %1.5E  %1.5E  %1.13E \n",
		emass0,emass1,emass2,emass);
	printf("Difference: %1.5E  but Charge: %1.5E \n",ionmass-emass,TotalCharge);
	
}

void Tensor3::Make3DRotationAboutAxis(Vector3 w, real t)
{
	real cost = cos(t);
	real sint = sin(t);
	real oneminuscost = 1.0-cost;
	xx = w.x*w.x*oneminuscost+cost;
	xy = w.x*w.y*oneminuscost-w.z*sint;
	xz = w.x*w.z*oneminuscost+w.y*sint;
	yx = w.x*w.y*oneminuscost+w.z*sint;
	yy = w.y*w.y*oneminuscost + cost;
	yz = w.y*w.z*oneminuscost - w.x*sint;
	zx = w.z*w.x*oneminuscost - w.y*sint;
	zy = w.z*w.y*oneminuscost + w.x*sint;
	zz = w.z*w.z*oneminuscost + cost;

}

class CalculateAccelsClass
{
public:
	// exists only to do a calculation repeatedly from some stored data

	Vector3 omega_ce, omega_ci;
	Tensor3 omega_ci_cross;
	
	real nu_eiBar, nu_eHeart, nu_ieBar, 
			nu_en_MT, nu_in_MT, nu_ne_MT, nu_ni_MT,
			n_i, n_n, n_e;
			
	real heat_transfer_rate_in,heat_transfer_rate_ni,
		heat_transfer_rate_en,heat_transfer_rate_ne,
		heat_transfer_rate_ei,heat_transfer_rate_ie;
	
	Vector3 ROC_v_ion_due_to_Rie,
		a_ion_pressure_and_E_accel,
		a_neut_pressure,
		a_ion_pressure;

	Vector3 vrel_e, ROC_v_ion_thermal_force; 

		Tensor3 Upsilon_nu_eHeart;

		Tensor3 Rie_thermal_force_matrix;

	real fric_dTe_by_dt_ei;

	Tensor3 Rie_friction_force_matrix;
	Tensor3 Ratio_times_Upsilon_eHeart;
	
	void CalculateCoefficients(Triangle * pTri)
	{
		static Tensor3 const ID3x3 (1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0);
		static real const TWOTHIRDSqsq = 2.0*q*q/3.0;
		static real const one_over_kB = 1.0/kB; // multiply by this to convert to eV
		static real const one_over_kB_cubed = 1.0/(kB*kB*kB); // multiply by this to convert to eV
		static real const kB_to_3halves = sqrt(kB)*kB;
		static real const over_sqrt_m_ion = 1.0/sqrt(m_ion);
		static real const over_sqrt_m_e = 1.0/sqrt(m_e);
		static real const qoverMc = q/(m_ion*c);
		static real const qovermc = q/(m_e*c);

		real const NU_EI_FACTOR = 1.0/(3.44e5);
		static real const nu_eiBarconst = //(4.0/3.0)*sqrt(2.0*PI/m_e)*q*q*q*q;
		// don't know in what units but it IS exactly what we already had - see Formulary
									1.0/(3.44e5);

		real area, det;
		real T_ion, T_n, T_e, sqrt_Te, ionneut_thermal, electron_thermal,
			lnLambda, s_in_MT, s_in_visc, s_en_MT,s_en_visc,
			nu_en_visc;
		
		//Vector3 const E, Vector3 const vrel_e, real * const scratch

		// The first thing we need to do is collect
	
		// nu_eibar, nu_in, nu_en
		// ======================

		// Get nu_eiBar
		// Get nu_en, nu_in, nu_ni, nu_ne, nu_eHeart
	
		area = pTri->area;
		n_i = pTri->ion.mass/area;
		n_e = pTri->elec.mass/area;
		n_n = pTri->neut.mass/area;

		if (pTri->ion.mass > 0.0) {
			T_ion = pTri->ion.heat/pTri->ion.mass;   // may be undefined
		} else {
			T_ion = 0.0;
		};
		if (pTri->neut.mass > 0.0) {
			T_n = pTri->neut.heat/pTri->neut.mass;
		} else {
			T_n = 0.0;
		};
		if (pTri->elec.mass > 0.0) {
			T_e = pTri->elec.heat/pTri->elec.mass;
			sqrt_Te = sqrt(T_e);
		} else {
			T_e = 0.0;
			sqrt_Te = 0.0;
		};

		// Somewhere here: floating point invalid operation
		// Probably divide by zero or sqrt(zero)

		//ion_thermal = sqrt(T_ion/m_ion);
		
		ionneut_thermal = sqrt(T_ion/m_ion+T_n/m_n); // hopefully not sqrt(0)

		electron_thermal = sqrt_Te*over_sqrt_m_e; // possibly == 0

		lnLambda = Get_lnLambda(n_i,T_e); // anything strange in there?

		Estimate_Ion_Neutral_Cross_sections(T_ion*one_over_kB, &s_in_MT, &s_in_visc);
		Estimate_Ion_Neutral_Cross_sections(T_e*one_over_kB,&s_en_MT,&s_en_visc);
		// To use combined temperature looks to be more intelligent -- rel temp GZSB(6.55) for ion, neutral at least.
		
		if (T_e != 0.0) {
			nu_eiBar = nu_eiBarconst*kB_to_3halves*n_i*lnLambda/(T_e*sqrt_Te);
		} else {
			nu_eiBar = 0.0;
		};
		// same as it ever was...
		
		// DO NEED nu_ie != nu_ei because we do
		// have n_e not equal to n_i sometimes, don't want to spuriously
		// generate momentum.

		// This is okay though.

		if (n_i > 0.0) {
			nu_ieBar = nu_eiBar*n_e/n_i;
		} else {
			nu_ieBar = 0.0;
		};
		
		nu_en_MT = n_n*s_en_MT*electron_thermal;
		nu_in_MT = n_n*s_in_MT*ionneut_thermal;
		nu_ne_MT = n_e*s_en_MT*electron_thermal;
		nu_ni_MT = n_i*s_in_MT*ionneut_thermal;
		
		nu_en_visc = n_n*s_en_visc*electron_thermal; 
		
		// those should all be fine though may == 0

		nu_eHeart = 1.87*nu_eiBar + nu_en_visc; // note, used visc
		
		 
		heat_transfer_rate_in = (2.0*m_i*m_n/((m_i+m_n)*(m_i+m_n)))
										*nu_in_MT; // ratio nu_in/nu_ni = n_n/n_i
		heat_transfer_rate_ni = (2.0*m_i*m_n/((m_i+m_n)*(m_i+m_n)))
										*nu_ni_MT;
		heat_transfer_rate_ne = (2.0*m_e*m_n/((m_e+m_n)*(m_e+m_n)))
										*nu_ne_MT;
		heat_transfer_rate_en = (2.0*m_e*m_n/((m_e+m_n)*(m_e+m_n)))
										*nu_en_MT;
		heat_transfer_rate_ei = (2.0*m_e*m_i/((m_e+m_i)*(m_e+m_i)))
										*nu_eiBar;
		heat_transfer_rate_ie = (2.0*m_e*m_i/((m_e+m_i)*(m_e+m_i)))
										*nu_ieBar;
		
		omega_ce = qovermc*pTri->B;
		omega_ci = qoverMc*pTri->B; // note: if ion acceleration stage, we could if we wanted work out B at k+1 first.
		omega_ci_cross.MakeCross(omega_ci);

		// NOTE: Uses GradTe so it better exist.

		// Populate Upsilon(nu_eHeart):
		real nu = nu_eHeart;
		Vector3 omega = omega_ce;

		det = nu*nu + omega.dot(omega);

		// (nu - omega x ) ^-1 :
		Upsilon_nu_eHeart.xx = nu*nu+omega.x*omega.x;
		Upsilon_nu_eHeart.xy = -nu*omega.z + omega.x*omega.y;
		Upsilon_nu_eHeart.xz = nu*omega.y + omega.x*omega.z;
		Upsilon_nu_eHeart.yx = nu*omega.z + omega.x*omega.y;
		Upsilon_nu_eHeart.yy = nu*nu + omega.y*omega.y;
		Upsilon_nu_eHeart.yz = -nu*omega.x + omega.y*omega.z;
		Upsilon_nu_eHeart.zx = -nu*omega.y + omega.z*omega.x;
		Upsilon_nu_eHeart.zy = nu*omega.x + omega.y*omega.z;
		Upsilon_nu_eHeart.zz = nu*nu + omega.z*omega.z;

		Upsilon_nu_eHeart = Upsilon_nu_eHeart/det;
	
		if (nu_eHeart > 0.0) {
			Ratio_times_Upsilon_eHeart = (nu_eiBar/nu_eHeart)*Upsilon_nu_eHeart;
		} else {
			ZeroMemory(&Ratio_times_Upsilon_eHeart,sizeof(Tensor3));
		};

		Rie_friction_force_matrix = 
			
			nu_ieBar*(m_e/m_i)*(ID3x3-0.9*Ratio_times_Upsilon_eHeart);
		// multiply by (v_e-v_i) for ions

		Rie_thermal_force_matrix = 
			
			((1.5/m_i)*(nu_ieBar/nu_eHeart)*Upsilon_nu_eHeart);
		// multiply by + GradTe for ions
		
		// Now what does this need to be:
		
		if (pTri->ion.mass == 0.0) {
			if (pTri->elec.mass == 0.0) {
				ZeroMemory(&vrel_e,sizeof(Vector3));
			} else {
				vrel_e = pTri->elec.mom/pTri->elec.mass;
			};
		} else {
			if (pTri->elec.mass == 0.0) {
				ZeroMemory(&vrel_e,sizeof(Vector3));
			} else {
				vrel_e = pTri->elec.mom/pTri->elec.mass-pTri->ion.mom/pTri->ion.mass;
			};
		};

		// Think about different contexts. :

		// Firstpass - ok
		// Compute sigma - not using existing vrel at all
		// Accelerate - make sure this has been set.

		// Reset v_e per vrel Ohm's Law once ion acceleration done!!!

		// **TEST OF PROGRAM:**
		// SHOULD v_ion 

		
		ROC_v_ion_due_to_Rie = 
				  Rie_thermal_force_matrix * pTri->GradTe
				+ Rie_friction_force_matrix * vrel_e; 
			
		ROC_v_ion_thermal_force = Rie_thermal_force_matrix * pTri->GradTe;
		
		if (pTri->ion.mass != 0.0) {
			a_ion_pressure_and_E_accel.x = pTri->scratch[2]/pTri->ion.mass + qoverM*pTri->E.x;
			a_ion_pressure_and_E_accel.y = pTri->scratch[3]/pTri->ion.mass + qoverM*pTri->E.y;
			a_ion_pressure_and_E_accel.z =  qoverM*pTri->E.z;
		
			a_ion_pressure.x = pTri->scratch[2]/pTri->ion.mass;
			a_ion_pressure.y = pTri->scratch[3]/pTri->ion.mass;
			a_ion_pressure.z = 0.0;
		} else {
			ZeroMemory(&a_ion_pressure_and_E_accel, sizeof(Vector3));
			ZeroMemory(&a_ion_pressure, sizeof(Vector3));
		};

		if (pTri->neut.mass > 0.0) {
			a_neut_pressure.x = pTri->scratch[0]/pTri->neut.mass;
			a_neut_pressure.y = pTri->scratch[1]/pTri->neut.mass;
			a_neut_pressure.z = 0.0;
		} else {
			ZeroMemory(&a_neut_pressure, sizeof(Vector3));
		};
		// All of these things, do not change, because we do not change E, vrel, pressure.
		
		if (n_e > 0.0) {
			fric_dTe_by_dt_ei =
								// - m_i*n_i*v_i*ROC_v_ion_due_to_Rie
								// - m_e*n_e*v_e*ROC_v_e_due_to_Rei
								// but, m_e n_e ROC_e = - m_i n_i ROC_i, so
						TWOTHIRDS * m_i*(n_i/n_e)*ROC_v_ion_due_to_Rie.dot(vrel_e);
		} else {
			fric_dTe_by_dt_ei = 0.0;
		};

		if (fric_dTe_by_dt_ei < 0.0) {
		//	printf("frictional cooling. ");
			
			//getch();
			//fric_dTe_by_dt_ei = 0.0;
			// It seems fair that the vrel part should give + 
			// but how to expect + from the Grad Te thermal force ???
			// It's possible that this thermal force accelerates positively the one that
			// already had the higher velocity. 
			// Still it can't make sense to steal random energy to account for it.
			// Braginskii says 'work done against the thermal force' .. 'sign is indeterminate' .. reversible heating.

		};
		

	}


	void inline CalculateAccels( Vector3 * pa_ion, Vector3 *  pa_neut, 
					real * pDT_ion, real *pDT_neut, real *pDT_e,
				const Vector3 & v_ion,const Vector3 & v_neut, 
				real const T_ion, real const T_neut, real const T_e )
				// the following are not going to change during a subcycle:
				//
				// so we might as well input them once, above.
				//)
	{
		// Pressure accels:
		//	neutmom.x += hsub*pTri->scratch[0];
		//	neutmom.y += hsub*pTri->scratch[1]; 
		//	ionmom.x += hsub*(pTri->scratch[2]);
		//	ionmom.y += hsub*(pTri->scratch[3]);
		//	elecmom.x += hsub*(pTri->scratch[4]);
		//	elecmom.y += hsub*(pTri->scratch[5]);

		Vector3 a_ion, a_neut, v_e_minus_v_n;
		real DT_ion, DT_neut, DT_e;
		real fric_heat_energy_rate_en_over_ne,
			fric_heat_energy_rate_in;

		a_ion = a_ion_pressure_and_E_accel; // constant throughout subcycle :-(
		
		a_ion -= omega_ci.cross(v_ion); 

		a_ion += (m_n/(m_i+m_n))*nu_in_MT*(v_neut - v_ion)
				+ ROC_v_ion_due_to_Rie;

		if (GlobalDebugRecordIndicator) {

			Globaldebugdata.pressure = a_ion_pressure;
			Globaldebugdata.qoverM_times_E = a_ion_pressure_and_E_accel - a_ion_pressure;
			Globaldebugdata.minus_omega_ci_cross_vi = -omega_ci.cross(v_ion); 
			Globaldebugdata.friction_from_neutrals = (m_n/(m_i+m_n))*nu_in_MT*(v_neut - v_ion);

			Globaldebugdata.friction_e = Rie_friction_force_matrix * vrel_e;

			Globaldebugdata.thermal_force = this->ROC_v_ion_thermal_force;
		
			Globaldebugdata.Upsilon = this->Upsilon_nu_eHeart;

			Globaldebugdata.nu_ie = this->nu_ieBar;
			Globaldebugdata.nu_in = this->nu_in_MT;

			Globaldebugdata.nu_en = this->nu_en_MT;
			Globaldebugdata.nu_ei = this->nu_eiBar;

			Globaldebugdata.a = a_ion;

			GlobalDebugRecordIndicator = 0; // do only once then set to off
		};
			

		a_neut = a_neut_pressure;
			
		v_e_minus_v_n = vrel_e + v_ion - v_neut;

		a_neut += (m_i/(m_i+m_n))*nu_ni_MT*(v_ion - v_neut)
			    + (m_e/(m_n+m_e))*nu_ne_MT*v_e_minus_v_n;
			
		// inter-species transfers:
		// ------------------------
		
		// GradTe : assumed rate of momentum transfer e-i governed by value at t_k
		// just same as use nu_en etc from t_k.
		// Don't yet even know what GradTe is at k+1 - heavily affected by heat conduction
		// apart from anything else.
		


		// frictional heating:
		
		// for i-n, e-n:
		// d/dt( ni mi vi^2 + nn mn vn^2)/2 = 
		//   ni mi vi (mn/(mi+mn)) nu_in (v_n-v_i) + nn mn vn (mi/(mi+mn)) nu_ni (v_i-v_n)
		// = (mi mn / (mi+mn)) ni nn (vi-vn) (vn-vi) s_in sqrt(T/mi)

		fric_heat_energy_rate_in = (m_i*m_n/(m_i+m_n))*n_i*nu_in_MT* // *n_n*
								(v_ion-v_neut).dot(v_ion-v_neut);
								//s_in_MT*ion_thermal;

		fric_heat_energy_rate_en_over_ne = (m_e*m_n/(m_e+m_n))*nu_en_MT* 
								(v_e_minus_v_n).dot(v_e_minus_v_n);


		// v_e-v_neut = vrel_e+v_ion-v_neut
			
		// =========
		// will want to estimate energy input -- assume energy input per time is J dot E
		
		// htg from en, ei goes mostly to e; heating from in goes to i and n per particle mass.
		// energy = 3/2 n T. Heat energy in cell = ion.heat * 3/2.
		// =========

		// Included here: frictional htg and heat transfer.
			
		DT_ion = (m_n/(m_i+m_n))*TWOTHIRDS*fric_heat_energy_rate_in/n_i; 


		DT_ion += heat_transfer_rate_in*(T_neut - T_ion)
					+ heat_transfer_rate_ie*(T_e - T_ion);

		DT_neut = (m_n/(m_i+m_n))*TWOTHIRDS*fric_heat_energy_rate_in/n_n; 
		
		DT_neut += heat_transfer_rate_ni*(T_ion - T_neut)
					+ heat_transfer_rate_ne*(T_e - T_neut);

		DT_e = fric_dTe_by_dt_ei + (fric_heat_energy_rate_en_over_ne)*(TWOTHIRDS);


		DT_e +=	heat_transfer_rate_en*(T_neut - T_e)
				  +	heat_transfer_rate_ei*(T_ion - T_e);

		GlobalFricHtgRate_in = TWOTHIRDS*fric_heat_energy_rate_in;

		GlobalFricHtgRate_en = TWOTHIRDS*fric_heat_energy_rate_en_over_ne*n_e;


		*pa_ion = a_ion;
		*pa_neut = a_neut;
		*pDT_ion = DT_ion;
		*pDT_neut = DT_neut;
		*pDT_e = DT_e;
	}

	void inline CalculateAccelsFunctionOfEandvrel( 
				Vector3 * pa_ion, Vector3 *  pa_neut, 
				Tensor3 * pD_ionbetaE, Tensor3 * pD_neutbetaE,
				Tensor3 * pD_ionbetavrel, Tensor3 * pD_neutbetavrel,
		
				Vector3 const v_ion,Vector3 const v_neut, 
				Tensor3 const v_ion_beta_E, Tensor3 const v_neut_beta_E,
				Tensor3 const v_ion_beta_vrel, Tensor3 const v_neut_beta_vrel)
	{
		real temp_in, temp_ni, temp_ne;
		Vector3 a_ion, a_neut, v_e_minus_v_n;
		Tensor3 D_ionbetaE, D_neutbetaE, D_ionbetavrel, D_neutbetavrel;
		
		a_ion = a_ion_pressure; // constant throughout subcycle 
		a_neut = a_neut_pressure;
		
		D_ionbetaE = qoverM*ID3x3; // effect rises at rate q/M
				
		// v_ion is here going to be v_ion_0.
		
		
		a_ion -= omega_ci.cross(v_ion); 
		D_ionbetaE -= omega_ci_cross*v_ion_beta_E;
		D_ionbetavrel = zero3x3 - omega_ci_cross*v_ion_beta_vrel;
		

		// Inter-species transfer:

		temp_in = (m_n/(m_i+m_n))*nu_in_MT;
		a_ion += temp_in*(v_neut - v_ion)
					+ ROC_v_ion_thermal_force;
				//+ ROC_v_ion_due_to_Rie; // a thing that depends on vrel
		
		D_ionbetaE += temp_in*(v_neut_beta_E - v_ion_beta_E);
		D_ionbetavrel += temp_in*(v_neut_beta_vrel - v_ion_beta_vrel)
						+ Rie_friction_force_matrix;
		
		v_e_minus_v_n = vrel_e + v_ion - v_neut;
		
		temp_ni = (m_i/(m_i+m_n))*nu_ni_MT;
		temp_ne = (m_e/(m_n+m_e))*nu_ne_MT;
		
		a_neut += temp_ni*(v_ion - v_neut)
			    + temp_ne*v_e_minus_v_n;
		
		D_neutbetaE = (temp_ni + temp_ne)*(v_ion_beta_E - v_neut_beta_E);
		
		D_neutbetavrel = temp_ni*(v_ion_beta_vrel - v_neut_beta_vrel)
					   + temp_ne*(ID3x3 + v_ion_beta_vrel - v_neut_beta_vrel); 
		
		
		// vrel considered here as an independent quantity in its own right;
		// then we make it a function of E at the end.

		// COMPUTE SIGMA: Don't need to bother with advancing T, it doesn't affect anything.

		
		*pa_ion = a_ion;
		*pa_neut = a_neut;
		*pD_ionbetaE = D_ionbetaE;
		*pD_neutbetaE = D_neutbetaE;
		*pD_ionbetavrel = D_ionbetavrel;
		*pD_neutbetavrel = D_neutbetavrel;
	}



	CalculateAccelsClass(){};
	~CalculateAccelsClass(){};
};

void TriMesh::GetGradTe(Triangle * pTri)
{
	real Tcorner0, Tcorner1, Tcorner2, Tneigh0, Tneigh1, Tneigh2;
	Vector2 u0, u1, u2, uNeigh0, uNeigh1, uNeigh2;
		
	real shoelace;

	// Get grad Te:
	// Use a hexagon with vertices and neighbour centroids
	// Think it does not need to be convex

	// Maybe only get this on the "compute sigma" pass
	
	pTri->neighbours[0]->GenerateContiguousCentroid(&uNeigh0,pTri);
	pTri->neighbours[1]->GenerateContiguousCentroid(&uNeigh1,pTri);
	pTri->neighbours[2]->GenerateContiguousCentroid(&uNeigh2,pTri);
	pTri->GenerateContiguousCentroid(&(pTri->cc),pTri);
	if (pTri->periodic == 0)
	{
		pTri->PopulatePositions(u0,u1,u2);
	} else {
		pTri->MapLeft(u0,u1,u2);
	};

	// Obtain vertex Te beforehand !!! &&&&&&&&&&&&&&&&&&&&&!!!!!!!!!!!!!!!!!!!!!!!

	Tcorner0 = pTri->cornerptr[0]->elec.T;
	Tcorner1 = pTri->cornerptr[1]->elec.T;
	Tcorner2 = pTri->cornerptr[2]->elec.T;

	Tneigh0 = pTri->neighbours[0]->elec.heat/pTri->neighbours[0]->elec.mass;
	Tneigh1 = pTri->neighbours[1]->elec.heat/pTri->neighbours[1]->elec.mass;
	Tneigh2 = pTri->neighbours[2]->elec.heat/pTri->neighbours[2]->elec.mass;
	
	// if neigh == self then so be it.

	// sequence: uNeigh0, u2, uNeigh1, u0, uNeigh2, u1

	shoelace =		uNeigh0.x * (u2.y - u1.y)
				+	u2.x * (uNeigh1.y-uNeigh0.y)
				+	uNeigh1.x * (u0.y-u2.y)
				+	u0.x * (uNeigh2.y-uNeigh1.y)
				+	uNeigh2.x * (u1.y-u0.y)
				+	u1.x * (uNeigh0.y-uNeigh2.y);
				
	pTri->GradTe.x = (	Tneigh0 * (u2.y-u1.y)
				+	Tcorner2 * (uNeigh1.y-uNeigh0.y)
				+	Tneigh1 * (u0.y-u2.y)
				+	Tcorner0 * (uNeigh2.y-uNeigh1.y)
				+	Tneigh2 * (u1.y-u0.y)
				+	Tcorner1 * (uNeigh0.y-uNeigh2.y)
					)/shoelace;

	pTri->GradTe.y = (	Tneigh0 * (u1.x - u2.x)
				+	Tcorner2 * (uNeigh0.x - uNeigh1.x)
				+	Tneigh1 * (u2.x - u0.x)
				+	Tcorner0 * (uNeigh1.x - uNeigh2.x)
				+	Tneigh2 * (u0.x - u1.x)
				+	Tcorner1 * (uNeigh2.x - uNeigh0.x)
					) / shoelace;

	if (pTri->GradTe.y > 1.0e10) {
		pTri = pTri;
	};

	pTri->GradTe.z = 0.0;
	// It doesn't have to be convex for this to work.

	// And probably want to store this - use for both Rei, Rie. Reasonable.
}

void TriMesh::AccelerateIons_or_ComputeOhmsLawForRelativeVelocity(Triangle * pTri, int code)
{
	static Tensor3 const ID3x3 (1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0);
	static real const TWOTHIRDSqsq = 2.0*q*q/3.0;
	static real const one_over_kB = 1.0/kB; // multiply by this to convert to eV
	static real const one_over_kB_cubed = 1.0/(kB*kB*kB); // multiply by this to convert to eV
	static real const kB_to_3halves = sqrt(kB)*kB;
	static real const over_sqrt_m_ion = 1.0/sqrt(m_ion);
	static real const over_sqrt_m_e = 1.0/sqrt(m_e);
	static real const qoverMc = q/(m_ion*c);
	static real const qovermc = q/(m_e*c);

	real const NU_EI_FACTOR = 1.0/(3.44e5);
	static real const nu_eiBarconst = //(4.0/3.0)*sqrt(2.0*PI/m_e)*q*q*q*q;
	// don't know in what units but it IS exactly what we already had - see Formulary
									1.0/(3.44e5);
	
	CalculateAccelsClass Y;
	real hsub;
	int numSubsteps;
	int iSubstep;

	Vector3 v_ion_k, v_neut_k, vhalf1_ion, vhalf2_ion,
		vhalf1_neut,vhalf2_neut,v_ion_predict, v_neut_predict,
		a_ion_k,a_neut_k,a_ion_1,a_neut_1,a_ion_2,a_neut_2,
		a_ion_predict,a_neut_predict,
		electron_pressure_accel;
	real T_ion_k, T_neut_k, T_e_k,
		Thalf1_ion,Thalf1_neut,Thalf1_e,
		Thalf2_ion,Thalf2_neut,Thalf2_e,
		T_ion_predict,T_neut_predict,T_e_predict,
		t,
		DT_ion_k,DT_neut_k,DT_e_k,
		DT_neut_1,DT_ion_1,DT_e_1,
		DT_neut_2,DT_ion_2,DT_e_2,
		DT_neut_predict,DT_ion_predict,DT_e_predict;
	
	real T_ion_kplus1,T_neut_kplus1,T_e_kplus1;
	Vector3 v_ion_kplus1, v_neut_kplus1;

	Vector3 operand3;

	static real const SIXTH = 1.0/6.0;
	
	// called with :
	
	// code == FIRSTPASS
	
	// = Half-accelerate heavy species predicting vrel_k+1/2 = vrel_k
	// and calculate displacement as linear function of pressure accel,
	// which then gets averaged on to vertices.
	
	// Half-evolve temperatures (as in, frictional htg and heat-transfer). -- ?
	// (Well, that allows us to be using better electron thermal pressure - or worse??)
	

	// Aside:
	// We would of course prefer that temps are evolved with linearly changing vrel after vrel_k+1
	// becomes known.
	// But if we chose to evolve v_i on (0,h) in the advected system, we would be giving up 
	// the opportunity to use pressure from t_k. 
	// It seems good to have use of that.
	// That means we can't wait and evolve temps after, since for fric htg we need the value of v_ion
	// during the first half.
	
	// OK - opt for this version: E and vrel flick from k to k+1 at half-time.
	
	
	// Then we go away and do the move, and ionise; create new pressures in advected system;
	// perhaps half the heat conduction should be done also, and half after - ideally.
	
	// code == COMPUTE_SIGMA
	
	// = Predict v_heavy_k+1 using present vrel, (in effect, vrel_k+1=vrel_k) ;
	// use this to create vrel_k+1 as linear function of E_k+1

	// pTri->vrel0 : vrel = vrel0 + sigma*E
	// The idea is to create a definite relationship that will then be actually
	// enforced to give vrel_k+1.
	
	
	// code == ACCELERATE 
	
	// Half-accelerate (second half of step) heavy species v ,
	// now given vrel_k+1 and pressure k+1
	// and half-evolve temperatures.
	
	// ---=

	// What about pressures? E balances pressure ... but mostly e pressure.
	// Not ion pressure.

	// (We should aim to merge with the viscous diffusion, heat conduction.
	//  But there have to be some limits of ambition for now.)

	// ====
	
	// If we come to do backward method for electrons, have to see how that
	// will fit in: if we assume E_k+1 throughout step, as we will likely need to do,
	// we can't do anything with them until we do the solve for E_k+1.
	// Let's see how it goes.
	
	
	// Because this appears here, we ought to do some heat conduction before COMPUTE_SIGMA call :
	if (code != ACCELERATE) GetGradTe(pTri); // for last call of this func, it is already stored.
	
	Y.CalculateCoefficients(pTri); // get things like nu, based on present Te
	
	

	
	// thing is, for prediction we'll want to accelerate v_ion_0 without any E
	// ... and without vrel
	// Have to do different accel routine


	
	// Ion acceleration / v_i_k+1 prediction stage
	// ============================================
	
	// Check timescale for heavy species evolution - should generally only need 1 RK4 step.
	
	// Let max step = min ( h*0.5, 0.1/nu_max inc heat transfer, 0.1/omega_ci ).
	// Usually 1 step.

	//hsub = min (0.1/Y.nu_eiBar, 0.1/Y.nu_en_MT);
	// Doesn't matter how fast momentum transferred from electrons. It's just adding a constant times hsub.

	hsub = 0.1/Y.nu_in_MT;
	hsub = min (hsub, 0.1/Y.omega_ci.modulus());
	
	hsub = min (hsub, 0.1/Y.heat_transfer_rate_ei);
	hsub = min (hsub, 0.1/Y.heat_transfer_rate_en);
	hsub = min (hsub, 0.1/Y.heat_transfer_rate_in); // these only really apply to some subcycles of course.
	
	if (hsub < h*0.5) {
		numSubsteps = (int) (h*0.5/hsub)+1; 
		hsub = h*0.5/(real)numSubsteps;
	} else {
		hsub = h*0.5; numSubsteps = 1;
	};
	// numSubsteps takes us halfway
	// Then if first pass, go numSubsteps again to generate predicted displacement
	
	// If compute sigma, ion v is already at halfway value
	// so just go numSubsteps from here to the end


	// FIRSTPASS:

	// Run to half time, set vars.
	// Run from half time, feint for displacement(pressure). <-- separate code

	// COMPUTE_SIGMA:

	// Run from half time to end, feint for vrel(E)

	// ACCELERATE:

	// Run from half time to end, set vars.

	v_ion_k = pTri->ion.mom/pTri->ion.mass;
	T_ion_k = pTri->ion.heat/pTri->ion.mass;
	v_neut_k = pTri->neut.mom/pTri->neut.mass;
	T_neut_k = pTri->neut.heat/pTri->neut.mass;
	T_e_k = pTri->elec.heat/pTri->elec.mass;
	
	// What is the difference between the runs, apart from setting vars at the end?
	// For compute sigma, feint is basically same code as first pass
	// since we just use existing vrel, E for both
	// I think it's basically same code for all 3 passes.
	
	// Then we put the pressure 2nd half -- have to work out how, after.
	
	if (code == FIRSTPASS)
	{
		// Default displacement:
		ZeroMemory(&(pTri->Displacement0[SPECIES_ION]),sizeof(Vector2));
		ZeroMemory(&(pTri->Displacement0[SPECIES_NEUTRAL]),sizeof(Vector2));
		ZeroMemory(&(pTri->Displacement0[SPECIES_ELECTRON]),sizeof(Vector2)); // used?
	};
	
	if ((code == ACCELERATE) || (code == FIRSTPASS)) 
	{
		
		for (iSubstep = 0; iSubstep < numSubsteps; iSubstep++)
		{
			
			// evolve v_ion, v_neutral, Ti,Tn,Te  RK4
			// including
			// pressure & E accel
			// gyroacceleration
			// momentum transfer & frictional htg
			// inter-species heat transfer
			// ==========================================
					
			Y.CalculateAccels( 
				&a_ion_k, &a_neut_k, &DT_ion_k, &DT_neut_k, &DT_e_k,
				v_ion_k, v_neut_k, T_ion_k, T_neut_k, T_e_k);
			
			// Rough est of htg for functional report:
			GlobalFrictionalHtg_in += hsub*GlobalFricHtgRate_in*pTri->area;
			GlobalFrictionalHtg_en += hsub*GlobalFricHtgRate_en*pTri->area;// 2/3 factor already included in rate
			GlobalFrictionalHtg_ei += hsub*Y.fric_dTe_by_dt_ei*pTri->elec.mass;


			// Now generate v 1/2 1 = v_k + h/2 f(t_k,v_k)
			// -----====----------------------------------

			vhalf1_ion = v_ion_k + a_ion_k*0.5*hsub;
			Thalf1_ion = T_ion_k + DT_ion_k*0.5*hsub;
			vhalf1_neut = v_neut_k + a_neut_k*0.5*hsub;
			Thalf1_neut = T_neut_k + DT_neut_k*0.5*hsub;
			Thalf1_e = T_e_k + DT_e_k*0.5*hsub;
			
			Y.CalculateAccels( 
				&a_ion_1, &a_neut_1, &DT_ion_1, &DT_neut_1, &DT_e_1,
				vhalf1_ion, vhalf1_neut, Thalf1_ion, Thalf1_neut, Thalf1_e);
			
			// Generate v 1/2 2 = v_k + h/2 f(t_k+1/2, v 1/2 1) & T 1/2 2
			
			vhalf2_ion = v_ion_k + a_ion_1*0.5*hsub;
			Thalf2_ion = T_ion_k + DT_ion_1*0.5*hsub;
			vhalf2_neut = v_neut_k + a_neut_1*0.5*hsub;
			Thalf2_neut = T_neut_k + DT_neut_1*0.5*hsub;
			Thalf2_e = T_e_k + DT_e_1*0.5*hsub;
			
			Y.CalculateAccels( 
				&a_ion_2, &a_neut_2, &DT_ion_2, &DT_neut_2, &DT_e_2,
				vhalf2_ion, vhalf2_neut, Thalf2_ion, Thalf2_neut, Thalf2_e);
			
			// want to use same ion pressure throughout move? Check with earlier-on code.
			
			
			// Generate vpredict = v_k + h f (t_k+1/2, v 1/2 2)
			// ------------------------------------------------

			v_ion_predict = v_ion_k + a_ion_2*hsub;
			T_ion_predict = T_ion_k + DT_ion_2*hsub;
			v_neut_predict = v_neut_k + a_neut_2*hsub;
			T_neut_predict = T_neut_k + DT_neut_2*hsub;
			T_e_predict = T_e_k + DT_e_2*hsub;

			Y.CalculateAccels( 
				&a_ion_predict, &a_neut_predict, &DT_ion_predict, &DT_neut_predict, &DT_e_predict,
				v_ion_predict, v_neut_predict, T_ion_predict, T_neut_predict, T_e_predict );

			// Generate v_k+1 = v_k + 1/6( f_k + 2 f_1 + 2 f_2 + f(predict))
			// -------------------------------------------------------------

			v_ion_kplus1 = v_ion_k + hsub*SIXTH*(a_ion_k + 2.0*(a_ion_1+a_ion_2) + a_ion_predict);
			v_neut_kplus1 = v_neut_k + hsub*SIXTH*(a_neut_k + 2.0*(a_neut_1+a_neut_2) + a_neut_predict);
			T_ion_kplus1 = T_ion_k + hsub*SIXTH*(DT_ion_k + 2.0*(DT_ion_1 + DT_ion_2) + DT_ion_predict);
			T_neut_kplus1 = T_neut_k + hsub*SIXTH*(DT_neut_k + 2.0*(DT_neut_1 + DT_neut_2) + DT_neut_predict);
			T_e_kplus1 = T_e_k + hsub*SIXTH*(DT_e_k + 2.0*(DT_e_1 + DT_e_2) + DT_e_predict);
			
			// Set up next substep
			// -------------------
			
			v_ion_k = v_ion_kplus1;
			v_neut_k = v_neut_kplus1;
			T_ion_k = T_ion_kplus1;
			T_neut_k = T_neut_kplus1;
			T_e_k = T_e_kplus1;
			

			// Evolve default displacement, on first pass:
			if (code == FIRSTPASS)
			{
				pTri->Displacement0[SPECIES_ION] += SIXTH*hsub*
					(v_ion_k.xypart() + 2.0*(vhalf1_ion.xypart() + vhalf2_ion.xypart()) + 
					v_ion_predict.xypart());
				pTri->Displacement0[SPECIES_NEUTRAL] += SIXTH*hsub*
					(v_neut_k.xypart() + 2.0*(vhalf1_neut.xypart() + vhalf2_neut.xypart()) +
					v_neut_predict.xypart());
			};
			
		};
		
		// So now we advanced half h.
		
		// Update the triangle actual values, if it's not a v_ion, v_n prediction stage :
		
		// elec mom: 
		// first pass: just add change in vi

		pTri->elec.mom = pTri->elec.mass*(pTri->elec.mom/pTri->elec.mass
						+ v_ion_kplus1-pTri->ion.mom/pTri->ion.mass) ; 
		pTri->ion.heat = T_ion_kplus1*pTri->ion.mass;
		pTri->ion.mom = v_ion_kplus1*pTri->ion.mass;
		pTri->neut.heat = T_neut_kplus1*pTri->neut.mass;
		pTri->neut.mom = v_neut_kplus1*pTri->neut.mass;
		pTri->elec.heat = T_e_kplus1*pTri->elec.mass;


		
		// If it's pre-advection, elec.mom will not be used for anything -- correct?
		// except, say, electron viscosity which gives rise to some viscous heating rate.
		

		// Think about this: Did we want to update that, before advecting?
		
		// Don't think it makes any difference to anything.
		
		// Have to track it out.
		
	} else {
		
		// COMPUTE_SIGMA:
		// Instead of advancing time, we want to advance a linear equation
		// predicting v_ion (1, E, vrel) and v_neut (1, E, vrel)
		
		Tensor3 
			v_ion_beta_E_k,v_ion_beta_vrel_k,v_neut_beta_E_k,v_neut_beta_vrel_k,
			vhalf1_ionbetaE, vhalf1_ionbetavrel, vhalf1_neutbetaE, vhalf1_neutbetavrel,
			vhalf2_ionbetaE, vhalf2_ionbetavrel, vhalf2_neutbetaE, vhalf2_neutbetavrel,
			v_ionbetaE_predict, v_ionbetavrel_predict, v_neutbetaE_predict, v_neutbetavrel_predict,
			
			D_ionbetaE,D_neutbetaE,D_ionbetavrel,D_neutbetavrel,
			D_ionbetaE_1,D_neutbetaE_1,D_ionbetavrel_1,D_neutbetavrel_1,
			D_ionbetaE_2,D_neutbetaE_2,D_ionbetavrel_2,D_neutbetavrel_2,
			D_ionbetaE_predict,D_neutbetaE_predict,D_ionbetavrel_predict,D_neutbetavrel_predict;

		ZeroMemory(&v_ion_beta_E_k,sizeof(Tensor3));
		ZeroMemory(&v_ion_beta_vrel_k,sizeof(Tensor3));
		ZeroMemory(&v_neut_beta_E_k,sizeof(Tensor3));
		ZeroMemory(&v_neut_beta_vrel_k,sizeof(Tensor3));
		
		for (iSubstep = 0; iSubstep < numSubsteps; iSubstep++)
		{
			
			Y.CalculateAccelsFunctionOfEandvrel( 
				&a_ion_k, &a_neut_k, 
				&D_ionbetaE, &D_neutbetaE,
				&D_ionbetavrel, &D_neutbetavrel,
		
				v_ion_k, v_neut_k, 
				v_ion_beta_E_k, v_neut_beta_E_k,
				v_ion_beta_vrel_k, v_neut_beta_vrel_k);
			
			// Now generate v 1/2 1 = v_k + h/2 f(t_k,v_k)
			// -----====----------------------------------
			
			vhalf1_ion = v_ion_k + a_ion_k*0.5*hsub;
			vhalf1_neut = v_neut_k + a_neut_k*0.5*hsub;
			
			vhalf1_ionbetaE = v_ion_beta_E_k + D_ionbetaE*0.5*hsub;
			vhalf1_ionbetavrel = v_ion_beta_vrel_k + D_ionbetavrel*0.5*hsub;
			vhalf1_neutbetaE = v_neut_beta_E_k + D_neutbetaE*0.5*hsub;
			vhalf1_neutbetavrel = v_neut_beta_vrel_k + D_neutbetavrel*0.5*hsub;
			
			Y.CalculateAccelsFunctionOfEandvrel( 
				&a_ion_1, &a_neut_1, 
				&D_ionbetaE_1, &D_neutbetaE_1,
				&D_ionbetavrel_1, &D_neutbetavrel_1,
				
				vhalf1_ion, vhalf1_neut, 
				vhalf1_ionbetaE, vhalf1_neutbetaE,
				vhalf1_ionbetavrel, vhalf1_neutbetavrel);
			
			// Generate v 1/2 2 = v_k + h/2 f(t_k+1/2, v 1/2 1) & T 1/2 2
			
			vhalf2_ion = v_ion_k + a_ion_1*0.5*hsub;
			vhalf2_neut = v_neut_k + a_neut_1*0.5*hsub;
			vhalf2_ionbetaE = v_ion_beta_E_k + D_ionbetaE_1*0.5*hsub;
			vhalf2_ionbetavrel = v_ion_beta_vrel_k + D_ionbetavrel_1*0.5*hsub;
			vhalf2_neutbetaE = v_neut_beta_E_k + D_neutbetaE_1*0.5*hsub;
			vhalf2_neutbetavrel = v_neut_beta_vrel_k + D_neutbetavrel_1*0.5*hsub;
			
			Y.CalculateAccelsFunctionOfEandvrel( 
				&a_ion_2, &a_neut_2, 
				&D_ionbetaE_2, &D_neutbetaE_2,
				&D_ionbetavrel_2, &D_neutbetavrel_2,
				
				vhalf2_ion, vhalf2_neut, 
				vhalf2_ionbetaE, vhalf2_neutbetaE,
				vhalf2_ionbetavrel, vhalf2_neutbetavrel);
						
			// want to use same ion pressure throughout move? Check with earlier-on code.
			
			
			// Generate vpredict = v_k + h f (t_k+1/2, v 1/2 2)
			// ------------------------------------------------
			
			v_ion_predict = v_ion_k + a_ion_2*hsub;
			v_neut_predict = v_neut_k + a_neut_2*hsub;
			v_ionbetaE_predict = v_ion_beta_E_k + D_ionbetaE_2*hsub;
			v_neutbetaE_predict = v_neut_beta_E_k + D_neutbetaE_2*hsub;
			v_ionbetavrel_predict = v_ion_beta_vrel_k + D_ionbetavrel_2*hsub;
			v_neutbetavrel_predict = v_neut_beta_vrel_k + D_neutbetavrel_2*hsub;
			
			Y.CalculateAccelsFunctionOfEandvrel( 
				&a_ion_predict, &a_neut_predict, 
				&D_ionbetaE_predict, &D_neutbetaE_predict,
				&D_ionbetavrel_predict, &D_neutbetavrel_predict,
				
				v_ion_predict, v_neut_predict, 
				v_ionbetaE_predict, v_neutbetaE_predict,
				v_ionbetavrel_predict, v_neutbetavrel_predict);
						
			// Generate v_k+1 = v_k + 1/6( f_k + 2 f_1 + 2 f_2 + f(predict))
			// -------------------------------------------------------------
			
			v_ion_kplus1 = v_ion_k + hsub*SIXTH*(a_ion_k + 2.0*(a_ion_1+a_ion_2) + a_ion_predict);
			v_neut_kplus1 = v_neut_k + hsub*SIXTH*(a_neut_k + 2.0*(a_neut_1+a_neut_2) + a_neut_predict);
			v_ion_beta_E_k = v_ion_beta_E_k + hsub*SIXTH*(D_ionbetaE + 
							2.0*(D_ionbetaE_1 + D_ionbetaE_2)
							+ D_ionbetaE_predict);
			v_ion_beta_vrel_k = v_ion_beta_vrel_k + hsub*SIXTH*(D_ionbetavrel +
							2.0*(D_ionbetavrel_1 + D_ionbetavrel_2)
							+ D_ionbetavrel_predict);
			v_neut_beta_E_k = v_neut_beta_E_k + hsub*SIXTH*(D_neutbetaE + 
							2.0*(D_neutbetaE_1 + D_neutbetaE_2)
							+ D_neutbetaE_predict);
			v_neut_beta_vrel_k = v_neut_beta_vrel_k + hsub*SIXTH*(D_neutbetavrel +
							2.0*(D_neutbetavrel_1 + D_neutbetavrel_2)
							+ D_neutbetavrel_predict);
			
						
			// Set up next substep
			// -------------------
			
			v_ion_k = v_ion_kplus1;
			v_neut_k = v_neut_kplus1;
			//v_ion_beta_E_k = v_ion_beta_E_kplus1;
			//v_ion_beta_vrel_k = v_ion_beta_vrel_kplus1;
			//v_neut_beta_E_k = v_neut_beta_E_kplus1;
			//v_neut_beta_vrel_k = v_neut_beta_vrel_kplus1;
			
			
		};
		
		// v_ion_k is now playing the role of v_ion_0
		
		// Now we want to do some manipulations to get the equation for vrel_i (1, E).
		

		// $@@@@@@@@@!!!!!!!!!!!!@@@@@@@@@@!!!!!!!!!!!*********#@@@@#####################
		// BEAR IN MIND, THE COEFFICIENT FOUND HERE WAS BETA ON VREL_E NOT VREL_I.
		// $@@@@@@@@@!!!!!!!!!!!!@@@@@@@@@@!!!!!!!!!!!*********#@@@@#####################
		
		
		// Now if we are doing the Ohm's Law setup then we'll just use a predicted value of v_i_k+1, v_n_k+1.
		// FOR NOW. We could solve out the k+1 quite easily .. but since here have vrel it isn't crucial?
		
		
		// Ohm's Law
		// ==========

		// Now stick to vrel_e-i throughout

	//	Vector3 vrel_e_0; 
	//	Vector3 vion_0;
	//	Tensor3 sigma_erel;
	//	Tensor3 sigma_i;   

		
		// Here is Ohm's Law for J :
		
		// Operand = E
		//			- B/c x v_i
		//			+ (m/q) grad(neTe)/(n_e m)
		//			- (m/q) (1/2 nu_in - nu_en)*(v_i-v_n))
		//			+ 3/(2qm) nu_eibar/nu_eheart UpsilonHeart grad Te
		
		// Sigma = qqn/m (nu_en + (ID3x3 - 0.9 (nu_eibar/nu_eHeart) UpsilonHeart)nu_eibar - omega_ce x )^-1
		
		// *But we prefer vrel*
		// Because we don't want high ne on one side of a boundary to drag stuff out of another cell.
		
		// So Sigma_erel = -(q/m + q/M) that stuff
		
		// Invert a matrix ...
		
		Tensor3 tempmatrix, invertedmatrix, omega_ce_cross, 
				sigma_inv_0, sigma_inv, B_cross_over_c,
				omega_ci_cross, frictional, to_invert, coeff_on_E, temp3x3;
		Vector3 thermal_force, ion_pressure_accel, vrel_0_operand;
		real nu_in_nu_en_stuff;
		
		int i = 0;
		
		// Start over.
		
		B_cross_over_c.MakeCross(pTri->B/c);
		omega_ce_cross.MakeCross(Y.omega_ce);
		omega_ci_cross.MakeCross(Y.omega_ci);
		
		electron_pressure_accel.x = pTri->scratch[4]/pTri->elec.mass;
		electron_pressure_accel.y = pTri->scratch[5]/pTri->elec.mass;
		electron_pressure_accel.z = 0.0;
		
		ion_pressure_accel.x = pTri->scratch[2]/pTri->ion.mass;
		ion_pressure_accel.y = pTri->scratch[3]/pTri->ion.mass; 
		ion_pressure_accel.z = 0.0;
		//Note: scratch is *cell momentum [pTri->ion.mom] addition rate*. 
		
		nu_in_nu_en_stuff = m_n*Y.nu_in_MT/(m_i+m_n)
						  - m_n*Y.nu_en_MT/(m_e+m_n);
		
		frictional = (m_n*Y.nu_en_MT/(m_e+m_n))*ID3x3 + 
					(1.0+Y.n_e*m_e/(Y.n_i*m_i))*
					(ID3x3 - 0.9*Y.Ratio_times_Upsilon_eHeart)*Y.nu_eiBar;
		// check all exist
		
		thermal_force = (1.0+Y.n_e*m_e/(Y.n_i*m_i))*(-1.5/m_e)*
					Y.Ratio_times_Upsilon_eHeart*pTri->GradTe;
		
		to_invert = - (omega_ce_cross+omega_ci_cross)*v_ion_beta_vrel_k
					- (nu_in_nu_en_stuff)*(v_ion_beta_vrel_k-v_neut_beta_vrel_k)
					+ frictional - omega_ce_cross;
		
		coeff_on_E = -(q/m_e + q/m_i)*ID3x3 + 
			(omega_ce_cross+omega_ci_cross)*v_ion_beta_E_k
			+ (nu_in_nu_en_stuff)*(v_ion_beta_E_k - v_neut_beta_E_k);
		
		to_invert.Inverse(temp3x3);
		
		pTri->sigma_erel = temp3x3*coeff_on_E;
		
		vrel_0_operand = 
				electron_pressure_accel - ion_pressure_accel
			+	(omega_ce_cross+omega_ci_cross)*v_ion_k
			+	(nu_in_nu_en_stuff)*(v_ion_k-v_neut_k) 
			+	thermal_force;
		
		pTri->vrel_e_0 = temp3x3*vrel_0_operand;
		
		// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

		// The following bit no longer can apply: 
		// We no longer will have a simple vrel(1,E) linear equation, with e viscosity.

		// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

		pTri->vion_0 = v_ion_k + v_ion_beta_vrel_k*pTri->vrel_e_0;
		pTri->sigma_i = v_ion_beta_E_k + v_ion_beta_vrel_k*pTri->sigma_erel;
		
		// 
		
		if (pTri->sigma_i.xx < 0.0) {
			i = i;
		}
		if (pTri->sigma_i.zz*(pTri->ion.mass-pTri->elec.mass)
				-pTri->sigma_erel.zz*pTri->elec.mass < 0.0)
		{
			i = i;
		};
		
		// First see if this can be made to do anything worthwhile.
		// Then carefully move to object.
		
		
				
		//
		//
		//tempmatrix = (Y.nu_en_MT + Y.nu_eiBar)*ID3x3
		//			- 0.9*Y.Ratio_times_Upsilon_eHeart
		//			- omega_ce_cross;
		//
		//sigma_inv_0 = ((m_e*m_i)/(q*(m_e+m_i)))*tempmatrix;
		//
		//// now compute default value:
		//
		//// pTri->scratch[4,5] are rate of adding momentum x,y for electrons
		//
		//// so pTri->scratch[4,5]/n_e is rate of adding velocity, we think
		//
		//electron_pressure_accel.x = pTri->scratch[4]/Y.n_e;
		//electron_pressure_accel.y = pTri->scratch[5]/Y.n_e;
		//electron_pressure_accel.z = 0.0;
		//
		//real tempcoefficient = (m_e*m_i/(q*(m_e+m_i)))*( ((m_e+m_n)/(m_e*m_n))*Y.nu_en_MT - ((m_i+m_n)/(m_i*m_n))*Y.nu_in_MT );
		//
		//// Using predicted v_ion,v_neut values:
		//operand3 =  B_cross_over_c*v_ion_kplus1
		//	
		//		+ (m_e*m_i/(q*(m_e+m_i)))*electron_pressure_accel
		//			
		//		- tempcoefficient*(v_ion_kplus1 - v_neut_kplus1)
		//			
		//		- (1.5/(q*m_e))*Y.Ratio_times_Upsilon_eHeart*pTri->GradTe;
		//			
		//Tensor3 U_vrel, U_E;
		//
		//U_vrel = B_cross_over_c*v_ion_beta_vrel_k 
		//	- tempcoefficient*(v_ion_beta_vrel_k - v_neut_beta_vrel_k);
		//
		//U_E = (-ID3x3) + B_cross_over_c*v_ion_beta_E_k 
		//	- tempcoefficient*(v_ion_beta_E_k - v_neut_beta_E_k);
		//	
		//sigma_inv = sigma_inv_0 - U_vrel;
		//sigma_inv.Inverse(pTri->sigma_erel);
		//
		//pTri->vrel_e_0 = pTri->sigma_erel*operand3; 	// v_i - v_e
		//		
		//pTri->sigma_erel = pTri->sigma_erel*U_E;
		//
		//// Now solve it back the other way: what is v_ion_k+1(E)?

		//// We had v_ion_k+1 = v_ion_k + v_ion_beta_E_k E + v_ion_beta_vrel_k (ve-vi)

		//pTri->vion_0 = v_ion_k + v_ion_beta_vrel_k*pTri->vrel_e_0;
		//pTri->sigma_i = v_ion_beta_E_k + v_ion_beta_vrel_k*pTri->sigma_erel;

		// =============================================================================




	};
	
	if (code == FIRSTPASS)
	{
		Tensor2 a_nTP_kplus1_effect_neutv,
			a_nTP_kplus1_effect_neutv_1,
			a_nTP_kplus1_effect_neutv_2,
			a_nTP_kplus1_effect_neutv_predict;
		Tensor3 a_iTP_kplus1_effect_ionv,
			a_iTP_kplus1_effect_ionv_1,
			a_iTP_kplus1_effect_ionv_2,
			a_iTP_kplus1_effect_ionv_predict,
			D_a_iTP_kplus1_effect_ionv,
			D_a_iTP_kplus1_effect_ionv_1,
			D_a_iTP_kplus1_effect_ionv_2,
			D_a_iTP_kplus1_effect_ionv_predict;

		Tensor3 omega_ci_cross;

		omega_ci_cross.MakeCross(Y.omega_ci);

		// Now do 2nd half feint for pressure relationship

		// effect of a_sTP on species momentum, running tally
		ZeroMemory(&a_nTP_kplus1_effect_neutv,sizeof(Tensor2));
		ZeroMemory(&a_iTP_kplus1_effect_ionv,sizeof(Tensor3));
			
		// Effect on displacement:
		ZeroMemory(&(pTri->Displacement0[SPECIES_ION]),sizeof(Vector2));
		ZeroMemory(&(pTri->Displacement0[SPECIES_NEUTRAL]),sizeof(Vector2));
		ZeroMemory(&(pTri->Pressure_a_effect_ion),sizeof(Tensor2));
		ZeroMemory(&(pTri->Pressure_a_effect_neut),sizeof(Tensor2));
			
		Tensor3 hsub_omega_ci_cross;
		hsub_omega_ci_cross = hsub*omega_ci_cross;;
		
		// Simplest:
		t = 0.0;
		for (iSubstep = 0; iSubstep < numSubsteps; iSubstep++)
		{
			// This way no good. Want to RK4 the displacement.

			//pTri->Displacement0[SPECIES_ION] += hsub*v_ion_k.xypart();
			//pTri->Displacement0[SPECIES_NEUTRAL] += hsub*v_neut_k.xypart();
									

			//pTri->Pressure_a_effect_ion.xx += hsub*a_iTP_kplus1_effect_ionv.xx; 
			//pTri->Pressure_a_effect_ion.xy += hsub*a_iTP_kplus1_effect_ionv.xy; 
			//pTri->Pressure_a_effect_ion.yx += hsub*a_iTP_kplus1_effect_ionv.yx; 
			//pTri->Pressure_a_effect_ion.yy += hsub*a_iTP_kplus1_effect_ionv.yy; 
					
			//pTri->Pressure_a_effect_neut += hsub*a_nTP_kplus1_effect_neutv; 
			
			t += hsub;
				
			//a_nTP_kplus1_effect_neutv.xx += hsub;
			//a_nTP_kplus1_effect_neutv.yy += hsub;
			//
			//a_iTP_kplus1_effect_ionv.xx += hsub; // x affects x , y affects y
			//a_iTP_kplus1_effect_ionv.yy += hsub;
			//a_iTP_kplus1_effect_ionv.zz += hsub;
			//	
			//a_iTP_kplus1_effect_ionv -= hsub_omega_ci_cross*a_iTP_kplus1_effect_ionv;

			// Recall logic:

			// v += a
			// v = (1-hsub omega x ) v = (1-hsub omega x ) a


			// Ignore inter-species transfer for additional momentum ...
			// though the effect should be to double a_iTP because of electron
			// momentum soaking to ions? -- What if it is weakly ionised and most
			// e momentum soaks to neutrals?
			
			Y.CalculateAccels( 
				&a_ion_k, &a_neut_k, &DT_ion_k, &DT_neut_k, &DT_e_k,
				v_ion_k, v_neut_k, T_ion_k, T_neut_k, T_e_k);
			D_a_iTP_kplus1_effect_ionv = ID3x3 - omega_ci_cross*a_iTP_kplus1_effect_ionv;
			
			// As well as advancing v_ion_k, we want to advance displacement
			
			// Now generate v 1/2 1 = v_k + h/2 f(t_k,v_k)
			// -----====----------------------------------
			
			vhalf1_ion = v_ion_k + a_ion_k*0.5*hsub;
			Thalf1_ion = T_ion_k + DT_ion_k*0.5*hsub;
			vhalf1_neut = v_neut_k + a_neut_k*0.5*hsub;
			Thalf1_neut = T_neut_k + DT_neut_k*0.5*hsub;
			Thalf1_e = T_e_k + DT_e_k*0.5*hsub;	// Not sure if advancing T serves a purpose, but, big deal.
			
			// not used for anything:
			//Disp_ion_1 = pTri->Displacement0[SPECIES_ION] + 0.5*hsub*v_ion_k.xypart();
			//Disp_neut_1 = pTri->Displacement0[SPECIES_NEUTRAL] + 0.5*hsub*v_neut_k.xypart();
			//Pressure_a_effect_ion_1 = pTri->Pressure_a_effect_ion
			//	+ 0.5*hsub*a_iTP_kplus1_effect_ionv.xy2x2part();
			//Pressure_a_effect_neut_1 = pTri->Pressure_a_effect_neut + 0.5*hsub*a_nTP_kplus1_effect_neutv;
			
			a_nTP_kplus1_effect_neutv_1 = a_nTP_kplus1_effect_neutv
				+ 0.5*hsub*ID2x2;
			a_iTP_kplus1_effect_ionv_1 = a_iTP_kplus1_effect_ionv
				+ 0.5*hsub*D_a_iTP_kplus1_effect_ionv ;
						
			Y.CalculateAccels( 
				&a_ion_1, &a_neut_1, &DT_ion_1, &DT_neut_1, &DT_e_1,
				vhalf1_ion, vhalf1_neut, Thalf1_ion, Thalf1_neut, Thalf1_e);
			D_a_iTP_kplus1_effect_ionv_1 = ID3x3 - omega_ci_cross*a_iTP_kplus1_effect_ionv_1;
			

			// Generate v 1/2 2 = v_k + h/2 f(t_k+1/2, v 1/2 1) & T 1/2 2
			// ----------------------------------------------------------
			
			vhalf2_ion = v_ion_k + a_ion_1*0.5*hsub;
			Thalf2_ion = T_ion_k + DT_ion_1*0.5*hsub;
			vhalf2_neut = v_neut_k + a_neut_1*0.5*hsub;
			Thalf2_neut = T_neut_k + DT_neut_1*0.5*hsub;
			Thalf2_e = T_e_k + DT_e_1*0.5*hsub;
			
			//Disp_ion_2 = pTri->Displacement0[SPECIES_ION] + 0.5*hsub*vhalf1_ion.xypart();
			//Disp_neut_2 = pTri->Displacement0[SPECIES_NEUTRAL] + 0.5*hsub*vhalf1_neut.xypart();
			//Pressure_a_effect_ion_2 = pTri->Pressure_a_effect_ion + 0.5*hsub*a_iTP_kplus1_effect_ionv_1.xy2x2part();
			//Pressure_a_effect_neut_2 = pTri->Pressure_a_effect_neut + 0.5*hsub*a_nTP_kplus1_effect_neutv_1;
			
			a_nTP_kplus1_effect_neutv_2 = a_nTP_kplus1_effect_neutv
				+ 0.5*hsub*ID2x2;
			a_iTP_kplus1_effect_ionv_2 = a_iTP_kplus1_effect_ionv
				+ 0.5*hsub*D_a_iTP_kplus1_effect_ionv_1 ;
			
			Y.CalculateAccels( 
				&a_ion_2, &a_neut_2, &DT_ion_2, &DT_neut_2, &DT_e_2,
				vhalf2_ion, vhalf2_neut, Thalf2_ion, Thalf2_neut, Thalf2_e);
			D_a_iTP_kplus1_effect_ionv_2 = ID3x3 - omega_ci_cross*a_iTP_kplus1_effect_ionv_2;
			
			
			// want to use same ion pressure throughout move? Check with earlier-on code.
			
			
			// Generate vpredict = v_k + h f (t_k+1/2, v 1/2 2)
			// ------------------------------------------------
			
			v_ion_predict = v_ion_k + a_ion_2*hsub;
			T_ion_predict = T_ion_k + DT_ion_2*hsub;
			v_neut_predict = v_neut_k + a_neut_2*hsub;
			T_neut_predict = T_neut_k + DT_neut_2*hsub;
			T_e_predict = T_e_k + DT_e_2*hsub;
			
			//Disp_ion_predict = pTri->Displacement0[SPECIES_ION] + hsub*vhalf2_ion.xypart();
			//Disp_neut_predict = pTri->Displacement0[SPECIES_NEUTRAL] + hsub*vhalf2_neut.xypart();
			//Pressure_a_effect_ion_predict = pTri->Pressure_a_effect_ion + hsub*a_iTP_kplus1_effect_ionv_2.xy2x2part();
			//Pressure_a_effect_neut_predict = pTri->Pressure_a_effect_neut + 0.5*hsub*a_nTP_kplus1_effect_neutv_2;
			
			a_nTP_kplus1_effect_neutv_predict = a_nTP_kplus1_effect_neutv
				+ hsub*ID2x2;
			a_iTP_kplus1_effect_ionv_predict = a_iTP_kplus1_effect_ionv
				+ hsub*D_a_iTP_kplus1_effect_ionv_2 ;
						
			Y.CalculateAccels( 
				&a_ion_predict, &a_neut_predict, &DT_ion_predict, &DT_neut_predict, &DT_e_predict,
				v_ion_predict, v_neut_predict, T_ion_predict, T_neut_predict, T_e_predict);
			D_a_iTP_kplus1_effect_ionv_predict = ID3x3 - omega_ci_cross*a_iTP_kplus1_effect_ionv_predict;
			
			
			
			// Generate v_k+1 = v_k + 1/6( f_k + 2 f_1 + 2 f_2 + f(predict))
			// -------------------------------------------------------------
			
			v_ion_kplus1 = v_ion_k + hsub*SIXTH*(a_ion_k + 2.0*(a_ion_1+a_ion_2) + a_ion_predict);
			v_neut_kplus1 = v_neut_k + hsub*SIXTH*(a_neut_k + 2.0*(a_neut_1+a_neut_2) + a_neut_predict);
			T_ion_kplus1 = T_ion_k + hsub*SIXTH*(DT_ion_k + 2.0*(DT_ion_1 + DT_ion_2) + DT_ion_predict);
			T_neut_kplus1 = T_neut_k + hsub*SIXTH*(DT_neut_k + 2.0*(DT_neut_1 + DT_neut_2) + DT_neut_predict);
			T_e_kplus1 = T_e_k + hsub*SIXTH*(DT_e_k + 2.0*(DT_e_1 + DT_e_2) + DT_e_predict);
			
			
			
			pTri->Displacement0[SPECIES_ION] += hsub*SIXTH*(v_ion_k.xypart() + 
				2.0*(vhalf1_ion.xypart() + vhalf2_ion.xypart()) + v_ion_predict.xypart());
			pTri->Displacement0[SPECIES_NEUTRAL] += hsub*SIXTH*(v_neut_k.xypart() + 
				2.0*(vhalf1_neut.xypart() + vhalf2_neut.xypart()) + v_neut_predict.xypart());
			
			pTri->Pressure_a_effect_ion += hsub*SIXTH*(
				a_iTP_kplus1_effect_ionv.xy2x2part() + 
				2.0*( a_iTP_kplus1_effect_ionv_1.xy2x2part()
					+ a_iTP_kplus1_effect_ionv_2.xy2x2part()) + 
				a_iTP_kplus1_effect_ionv_predict.xy2x2part());
			
			pTri->Pressure_a_effect_neut += hsub*SIXTH*(
				a_nTP_kplus1_effect_neutv + 
				2.0*(	a_nTP_kplus1_effect_neutv_1
					+	a_nTP_kplus1_effect_neutv_2) + 
				a_nTP_kplus1_effect_neutv_predict);
			
			a_iTP_kplus1_effect_ionv += hsub*SIXTH*( 
				D_a_iTP_kplus1_effect_ionv + 
				2.0*(	D_a_iTP_kplus1_effect_ionv_1 
					+	D_a_iTP_kplus1_effect_ionv_2) + 
				D_a_iTP_kplus1_effect_ionv_predict);

			a_nTP_kplus1_effect_neutv += hsub* ID2x2 ;



			// Set up next substep
			// -------------------
			
			v_ion_k = v_ion_kplus1;
			v_neut_k = v_neut_kplus1;
			T_ion_k = T_ion_kplus1;
			T_neut_k = T_neut_kplus1;
			T_e_k = T_e_kplus1;
			
		};

	}; // code == FIRSTPASS


	
	
	// Will want to set up ODE to include viscous mom diffusion of e(-i).
	
	// To work out visc htg? change in DKE as a result of spread of mom = ?
	// (w/o Coulombic part)
	// as a result of flow through boundary from higher
	// to lower, dv1/dt due to boundary flow = a1
	// dv2/dt due to boundary flow = a2
	// usually v2 < v1 if a2 > a1
	// d/dt(n1v1^2+n2v2^2)/2 = n1v1a1+n2v2a2
	// That way makes the most sense really I think. Can it avoid viscous cooling?
	
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

}


void TriMesh::SolveForAdvectedPositions(int species)  // populate advected position data for each vertex using AdvectedPosition0 and Pressure_effect.
{
	//	want to set
	// pVertex->AdvectedPosition[species]_ion = AdvectedPosition0_ion + pVertex->Pressure_a_effect_ion_dis * acceleration due to pressure

	// We then will place the cell data on to the new mesh that we can create from this.

	// Approach to solving:
	// Not sure about Jacobi since we can imagine that one point alone would move very slowly given that it soon feels pressure back from its surroundings.
	// Within reason / the right circumstances, we actually believe that just iterating will do it: pressure pushes us towards where 
	// the equilibrium lies.
	// But look in terms of the eqm of  xdot = -x + (xk + d0(h)) + F(hh) a
	// and using xdotdot we can have a second-order step for that system;
	// if something is going haywire or moving too near its surrounding polygon, we slow down the system trajectory timestep.
	// We have to calculate both a(x) and for a second-order step, a-dot due to xdot. 
	
	real htraj = 0.25; 
	static real const MAXhtraj = 0.25;
	real guess_h, value_new, twoarea_new, twoarea, area_roc_over_area;
	Vector2 temp2, to_existing;
	real compare;
	Vertex * pVertex, *pOther;
	Vector2 acceleration, momrate;
	Triangle * pTri1,* pTri2, *pTri;
	int i, inext;
	Vector2 u[3], U[3], ROC[3];
	Vector2 ROCcc1, cc1, ROCcc2, cc2, ROCmomrate, ROC_accel, ROCu_ins;
	real area, ROCarea, ROCvalue, Average_nT, ROCAverage_nT, value;
	long iTri, iVertex;
	Vector2 u_ins, rhat;
	int iWhich;
	bool broken, not_converged_enough;

	static real const FIVETHIRDS = THIRD*5.0;
	
		// New way for pressure at vertex, justified:
		// area with medians and centroid always equals 1/3 of tri area,
		// and therefore that "Voronoi" mass here can be given for all time.
	
	
	
	
	// Change to make:
	// Somehow incorporate info from ion move (ion and neutral pressure k+1 known)
	// this applies to acceleration
	// but doesn't something also apply to the e move, the E solver?

	// if ion dis is one move only, we want that actual move to give us n_ion
	// is that pointing another way?

	// if ion dis is another pass after E, we want to do what? use estimate of ion pressure k+1 ??

	// That estimate coming from what, for a cell?
	// Things are a bit f'd up here.
	// Pressure at k+1 ought to give rise to, the ion cell displacement, which in turn gives rise to rho_vertex

	// However, should we just say that we know how things are going to move, work that way?
	// To run E on the new mesh we have to know J a function of E_k+1 there.

	// Not clear what cell we map to on the new mesh. Have to put up with E on old cell.
	// 
	long iIterationsConvergence = 0;

	switch (species)
	{
	case SPECIES_ION:
		pVertex = X;
		for (iVertex = 0; iVertex < numVertices; iVertex++)
		{
			pVertex->AdvectedPosition[species].x = pVertex->x;
			pVertex->AdvectedPosition[species].y = pVertex->y;			
			pVertex->Polygon_mass =0.0;
			for (i = 0; i < pVertex->triangles.len; i++)
			{
				pTri1 = ((Triangle *)pVertex->triangles.ptr[i]);
				pVertex->Polygon_mass += THIRD*pTri1->ion.mass;
			};
			pVertex->Polygon_mass *= m_ion;
			++pVertex;
		};
		break;
	case SPECIES_NEUTRAL:
		pVertex = X;
		for (iVertex = 0; iVertex < numVertices; iVertex++)
		{
			pVertex->AdvectedPosition[species].x = pVertex->x;
			pVertex->AdvectedPosition[species].y = pVertex->y;			
			pVertex->Polygon_mass =0.0;
			for (i = 0; i < pVertex->triangles.len; i++)
			{
				pTri1 = ((Triangle *)pVertex->triangles.ptr[i]);
				pVertex->Polygon_mass += THIRD*pTri1->neut.mass;
			};
			pVertex->Polygon_mass *= m_neutral;
			++pVertex;
		};
		break;
	case SPECIES_ELECTRON:
		pVertex = X;
		for (iVertex = 0; iVertex < numVertices; iVertex++)
		{
			pVertex->AdvectedPosition[species].x = pVertex->x;
			pVertex->AdvectedPosition[species].y = pVertex->y;			
			pVertex->Polygon_mass =0.0;
			for (i = 0; i < pVertex->triangles.len; i++)
			{
				pTri1 = ((Triangle *)pVertex->triangles.ptr[i]);
				pVertex->Polygon_mass += THIRD*pTri1->elec.mass;
			};
			pVertex->Polygon_mass *= m_e;
			++pVertex;
		};
		break;
	};
		
	// Loop to solve for positions:

	do
	{
		// Get putative areas and nT_cell:
		pTri = T;
		for (iTri = 0; iTri < numTriangles; iTri++)
		{
			u[0] = pTri->cornerptr[0]->AdvectedPosition[species];
			u[1] = pTri->cornerptr[1]->AdvectedPosition[species];
			u[2] = pTri->cornerptr[2]->AdvectedPosition[species];
			if (pTri->periodic > 0) {
				if (pTri->periodic == 1) {
					// rotate (original) leftmost point to right	
					i = pTri->GetLeftmostIndex();
					u[i] = Clockwise*u[i];
				} else {
					i = pTri->GetRightmostIndex();
					u[i] = Anticlockwise*u[i];
				};
			};
			// Note problem: a triangle can stop or start being periodic as part of the planned advection.
			// What to do about it?
			// LEAVE POINTS UNWRAPPED. Then periodic data of triangle doesn't go stale.

			area = fabs(
							0.5*(	u[0].x*u[1].y - u[1].x*u[0].y
									+	u[1].x*u[2].y - u[2].x*u[1].y
									+	u[2].x*u[0].y - u[0].x*u[2].y	));
			// Does need a branch or fabs because we do not know which way is clockwise.

			// heat/orig (Area now/ orig)^(-5/3) =
			// heat Area_orig^2/3 / Area_now ^5/3
			if (species == SPECIES_ION)
			{
				pTri->nT = pTri->ion.heat*pow((pTri->area/area),FIVETHIRDS)/pTri->area;
			} else {
				if (species == SPECIES_NEUTRAL)
				{
					pTri->nT = pTri->neut.heat*pow((pTri->area/area),FIVETHIRDS)/pTri->area;
				} else {
					pTri->nT = pTri->elec.heat*pow((pTri->area/area),FIVETHIRDS)/pTri->area;
				};
			};
			// or take pTri->ion.heat*pow(pTri->area/area,TWOTHIRDS)/area;
			// Will likely want to precalculate a function like pow (,2/3) and see if quicker.
			// Get rid of exponent first; can store a multiplier that accounts for 10^-13 ^ (2/3)
				
			// pow is expensive - so we did this first for each triangle.			
			++pTri;			
		};

		pVertex = X;
		for (iVertex = 0; iVertex < numVertices; iVertex++)
		{
			// Next value is found from:
			// _________________________

			// 1. calculate acceleration at system present position

			// How we get pressure ordinarily:
	
				//pTri1 = ((Triangle *)pVert->triangles.ptr[i]);
				//pTri2 = ((Triangle *)pVert->triangles.ptr[inext]);
				//pTri1->ReturnCentre(&cc1,pVert); // same tranche as pVert if periodic
				//pTri2->ReturnCentre(&cc2,pVert); 	// Model centroid as where we get nT.
				//Average_nT_neut = 0.5*(pTri1->neut.heat/pTri1->area + pTri2->neut.heat/pTri2->area);
				//ydist = cc2.y-cc1.y; 
				//xdist = cc1.x-cc2.x; 
				//pVert->Pressure_numerator_x -= ydist*Average_nT_neut;
				//pVert->Pressure_numerator_y -= xdist*Average_nT_neut;
			//// that gives momentum addition rate; divide by particle mass and we can have in our units of momentum
			//pTri2->scratch[0] += frac*pVert->Pressure_numerator_x/m_neutral;  // dividing by particle mass

			momrate.x = 0.0; momrate.y = 0.0;
			
			if (pVertex->flags < 3)
			{
				for (i = 0; i < pVertex->triangles.len; i++)
				{
					inext = i+1; if (inext == pVertex->triangles.len) inext = 0;
					pTri1 = ((Triangle *)pVertex->triangles.ptr[i]);
					pTri2 = ((Triangle *)pVertex->triangles.ptr[inext]);
					pTri1->ReturnCentre(&cc1,pVertex); // same tranche as pVert if periodic
					pTri2->ReturnCentre(&cc2,pVertex); 	// Model centroid as where we get nT.
					
					Average_nT = 0.5*(pTri1->nT + pTri2->nT); // had to precompute these - expensive.
					momrate.x -= (cc2.y-cc1.y)*Average_nT;
					momrate.y -= (cc1.x-cc2.x)*Average_nT; 
				}; 
				acceleration = momrate / pVertex->Polygon_mass; // Polygonmass = 1/3 total mass of cells, in our number units * m_species
				
			} else {
				// insulator or outer vertex:
				// special treatment at first and last triangles ...
				pTri1 = ((Triangle *)pVertex->triangles.ptr[0]);
				pTri1->ReturnCentre(&cc1,pVertex); 
				if (pVertex->flags == 3) {
					// new way: median is what we want
					pTri1->GetBaseMedian(&u_ins, pVertex);	
				} else {
					pTri1->GetOuterMedian(&u_ins, pVertex);					
				};				
				Average_nT = pTri1->nT;
				momrate.x -= (cc1.y-u_ins.y)*Average_nT;
				momrate.y -= (u_ins.x-cc1.x)*Average_nT; 
				
				// Now do the usual until we reach the last one:
				
				for (i = 0; i < pVertex->triangles.len - 1; i++)
				{
					inext = i+1;
					pTri1 = ((Triangle *)pVertex->triangles.ptr[i]);
					pTri2 = ((Triangle *)pVertex->triangles.ptr[inext]);
					pTri1->ReturnCentre(&cc1,pVertex); // same tranche as pVert if periodic
					pTri2->ReturnCentre(&cc2,pVertex); 
					// Model centroid as where we get nT.
					Average_nT = 0.5*(pTri1->nT + pTri2->nT);
					momrate.x -= (cc2.y-cc1.y)*Average_nT;
					momrate.y -= (cc1.x-cc2.x)*Average_nT; 
				};		

				if (pVertex->flags == 3) {
					pTri2->GetBaseMedian(&u_ins, pVertex);
				} else {
					pTri2->GetOuterMedian(&u_ins, pVertex);
				};
				Average_nT = pTri2->nT;
				momrate.x -= (u_ins.y-cc2.y)*Average_nT;
				momrate.y -= (cc2.x-u_ins.x)*Average_nT; 
				
				// Now kill the radial component:
				//rhat = pVertex->AdvectedPosition[species]/(pVertex->AdvectedPosition[species].modulus());
				//momrate -= rhat*(momrate.dot(rhat));
				// Note that it's the actual xdot that needs to have the radial component removed.
				// What is the actual intention here?

				acceleration = momrate / pVertex->Polygon_mass; // Polygonmass = 1/3 total mass of cells, in our number units *m_species
				// not sure what we should say about its move;
				// it has to be literally put back on the insulator.
			};

			// 2. calculate position rate of change: xdot
			// ---------------------------------------------------------------
			// seek eqm of  xdot = -x + (xk + d0(h)) + F(hh) a :
			pVertex->xdot = (pVertex->AdvectedPosition0[species] - pVertex->AdvectedPosition[species])
										+ pVertex->Pressure_a_effect_dis[species]*acceleration;

			if (pVertex->flags >= 3) {
				rhat = pVertex->AdvectedPosition[species]/(pVertex->AdvectedPosition[species].modulus());
				pVertex->xdot -= rhat*(pVertex->xdot.dot(rhat));
			}

			// note that we are expecting, for an insulator vertex the accel==0 position should also be on the insulator!
			++pVertex;
		};

		// Now want to get xdotdot.
		// First each cell calculates how fast its new area is changing according to xdot:
		pTri = T;
		for (iTri = 0; iTri < numTriangles; iTri++)
		{
			// Calculate rate of change of area and hence, ROC_nT_ion:
			u[0] = pTri->cornerptr[0]->AdvectedPosition[species];
			ROC[0] = pTri->cornerptr[0]->xdot;
			u[1] = pTri->cornerptr[1]->AdvectedPosition[species];
			ROC[1] = pTri->cornerptr[1]->xdot;
			u[2] = pTri->cornerptr[2]->AdvectedPosition[species];
			ROC[2] = pTri->cornerptr[2]->xdot;
			if (pTri->periodic > 0) {
				if (pTri->periodic == 1) {
					// rotate (original) leftmost point to right	
					i = pTri->GetLeftmostIndex();
					u[i] = Clockwise*u[i];
					ROC[i] = Clockwise*ROC[i]; // from the point of view of this per triangle, how it's moving
				} else {
					i = pTri->GetRightmostIndex(); // wrap the unwrapped point:
					u[i] = Anticlockwise*u[i];
					ROC[i] = Anticlockwise*ROC[i];
				};
			};

			// Note problem: a triangle can stop or start being periodic as part of the planned advection.
			// What to do about it?
			// LEAVE POINTS UNWRAPPED. Then periodic data of triangle doesn't go stale.

			value = 0.5*( u[0].x*u[1].y - u[1].x*u[0].y
									+ u[1].x*u[2].y - u[2].x*u[1].y
									+ u[2].x*u[0].y - u[0].x*u[2].y);

			ROCvalue = 0.5*( ROC[0].x*u[1].y + u[0].x*ROC[1].y - ROC[1].x*u[0].y - u[1].x*ROC[0].y
									+ ROC[1].x*u[2].y + u[1].x*ROC[2].y - ROC[2].x*u[1].y - u[2].x*ROC[1].y
									+ ROC[2].x*u[0].y + u[2].x*ROC[0].y - ROC[0].x*u[2].y - u[0].x*ROC[2].y);

			if (value > 0) {
				area = value;
				ROCarea = ROCvalue;
			} else {
				area = -value;
				ROCarea = -ROCvalue; 
			};
			// Note that change of sign compared to initial during a move is unexpected --
			// that indicates a triangle was flipped, and
			// we should have rejected any such attempted move and never got here.

			//pTri->nT_ion = pTri->ion.heat*pow(pTri->area/area),FIVETHIRDS)/pTri->area;
			
			pTri->ROC_nT = ROCarea*(-FIVETHIRDS)*pTri->nT / area; // f '(g(x))g'(x)

			++pTri;
		};

		pVertex = X;
		for (iVertex = 0; iVertex < numVertices; iVertex++)
		{
			// 3. ROC acceleration:

			// We have to know the combined effect on pressure here from the effects of moving all of, this point and all the neighbours
			// Area is changing, but also the centroid coordinates are changing.
						
			ROCmomrate.x = 0.0; ROCmomrate.y = 0.0;
			if (pVertex->flags < 3)
			{
				for (i = 0; i < pVertex->triangles.len; i++)
				{
					inext = i+1; if (inext == pVertex->triangles.len) inext = 0;
					pTri1 = ((Triangle *)pVertex->triangles.ptr[i]);
					pTri2 = ((Triangle *)pVertex->triangles.ptr[inext]);
					pTri1->ReturnCentre(&cc1,pVertex); // same tranche as pVert if periodic
					pTri2->ReturnCentre(&cc2,pVertex); 	// Model centroid as where we get nT.
					
					ROC[0] = pTri1->cornerptr[0]->xdot;
					ROC[1] = pTri1->cornerptr[1]->xdot;
					ROC[2] = pTri1->cornerptr[2]->xdot;
					if (pTri1->periodic > 0)
					{
						// important that it be relative to our vertex where acceleration is to be found! ..
						// wrapping status of corners is still per the original data
						if (pTri1->periodic == 1)
						{
							iWhich = pTri1->GetLeftmostIndex();
							if (pVertex->x > 0.0) {
								// bring that periodic one back
							    ROC[iWhich] = Clockwise*ROC[iWhich];
							} else {
								if (iWhich != 0) ROC[0] = Anticlockwise*ROC[0];
								if (iWhich != 1) ROC[1] = Anticlockwise*ROC[1];
								if (iWhich != 2) ROC[2] = Anticlockwise*ROC[2];
							};
						} else {
							// pTri1->periodic == 2
							iWhich = pTri1->GetRightmostIndex();
							if (pVertex->x > 0.0) {
								if (iWhich != 0) ROC[0] = Clockwise*ROC[0];
								if (iWhich != 1) ROC[1] = Clockwise*ROC[1];
								if (iWhich != 2) ROC[2] = Clockwise*ROC[2];
							} else {
								ROC[iWhich] = Anticlockwise*ROC[iWhich];
							};
						};
					};
					ROCcc1 = THIRD*(ROC[0] + ROC[1] + ROC[2]);
					// same for pTri2 .... 
					
					ROC[0] = pTri2->cornerptr[0]->xdot;
					ROC[1] = pTri2->cornerptr[1]->xdot;
					ROC[2] = pTri2->cornerptr[2]->xdot;					
					if (pTri2->periodic > 0)
					{
						if (pTri2->periodic == 1)
						{
							iWhich = pTri2->GetLeftmostIndex();
							if (pVertex->x > 0.0) {
								// bring that periodic one back
							    ROC[iWhich] = Clockwise*ROC[iWhich];
							} else {
								if (iWhich != 0) ROC[0] = Anticlockwise*ROC[0];
								if (iWhich != 1) ROC[1] = Anticlockwise*ROC[1];
								if (iWhich != 2) ROC[2] = Anticlockwise*ROC[2];
							};
						} else {
							// pTri2->periodic == 2
							iWhich = pTri2->GetRightmostIndex();
							if (pVertex->x > 0.0) {
								if (iWhich != 0) ROC[0] = Clockwise*ROC[0];
								if (iWhich != 1) ROC[1] = Clockwise*ROC[1];
								if (iWhich != 2) ROC[2] = Clockwise*ROC[2];
							} else {
								ROC[iWhich] = Anticlockwise*ROC[iWhich];
							};
						};
					};
					ROCcc2 = THIRD*(ROC[0] + ROC[1] + ROC[2]);

					// ROC_nT_ion calculated first to save on pow.
					ROCAverage_nT = 0.5*(pTri1->ROC_nT + pTri2->ROC_nT);
					Average_nT = 0.5*(pTri1->nT + pTri2->nT);
					
					//momrate.x -= (cc2.y-cc1.y)*Average_nT_ion/m_ion;
					ROCmomrate.x -= (ROCcc2.y-ROCcc1.y)*Average_nT
											+ (cc2.y-cc1.y)*ROCAverage_nT;
					ROCmomrate.y -= (ROCcc1.x-ROCcc2.x)*Average_nT
											+ (cc1.x-cc2.x)*ROCAverage_nT;
				};
				ROC_accel = ROCmomrate / pVertex->Polygon_mass;
			} else {
				// in this case add on the first and last tri's differently;
				// and note that the rate of change should point ... ??
				pTri1 = ((Triangle *)pVertex->triangles.ptr[0]);
				pTri1->ReturnCentre(&cc1,pVertex); 
				if (pVertex->flags == 3) {
					// new way: median is what we want
					pTri1->GetBaseMedian(&u_ins, pVertex);	
				} else {
					pTri1->GetOuterMedian(&u_ins, pVertex);					
				};				

				ROC[0] = pTri1->cornerptr[0]->xdot;
				ROC[1] = pTri1->cornerptr[1]->xdot;
				ROC[2] = pTri1->cornerptr[2]->xdot;
				if (pTri1->periodic > 0)
				{
					// important that it be relative to our vertex where acceleration is to be found! ..
					// wrapping status of corners is still per the original data
					if (pTri1->periodic == 1)
					{
						iWhich = pTri1->GetLeftmostIndex();
						if (pVertex->x > 0.0) {
							// bring that periodic one back
						    ROC[iWhich] = Clockwise*ROC[iWhich];
						} else {
							if (iWhich != 0) ROC[0] = Anticlockwise*ROC[0];
							if (iWhich != 1) ROC[1] = Anticlockwise*ROC[1];
							if (iWhich != 2) ROC[2] = Anticlockwise*ROC[2];
						};
					} else {
						// pTri1->periodic == 2
						iWhich = pTri1->GetRightmostIndex();
						if (pVertex->x > 0.0) {
							if (iWhich != 0) ROC[0] = Clockwise*ROC[0];
							if (iWhich != 1) ROC[1] = Clockwise*ROC[1];
							if (iWhich != 2) ROC[2] = Clockwise*ROC[2];
						} else {
							ROC[iWhich] = Anticlockwise*ROC[iWhich];
						};
					};
				};
				ROCcc1 = THIRD*(ROC[0] + ROC[1] + ROC[2]);

				//ROCu_ins <-- depends on movement of 2 vertices, this one and the other one that is flags==3.
				pOther = pTri1->GetOtherBaseVertex(pVertex);
				if (pTri1->periodic == 0)
				{
					ROCu_ins = 0.5*(pVertex->xdot + pOther->xdot);					
				} else {
					if (pTri1->periodic == 1) 
					{
						// pTri1->periodic == 1
						iWhich = pTri1->GetLeftmostIndex();
						if (pTri1->cornerptr[iWhich] == pOther) {
							ROCu_ins = 0.5*(pVertex->xdot + Clockwise*(pOther->xdot));	
						} else {
							if (pTri1->cornerptr[iWhich] == pVertex) {
								ROCu_ins = 0.5*(pVertex->xdot + Anticlockwise*(pOther->xdot));	
							} else {
								ROCu_ins = 0.5*(pVertex->xdot + pOther->xdot);	
							};
						};						
					} else {
						iWhich = pTri1->GetRightmostIndex();
						if (pTri1->cornerptr[iWhich] == pOther) {
							ROCu_ins = 0.5*(pVertex->xdot + Anticlockwise*(pOther->xdot));	
						} else {
							if (pTri1->cornerptr[iWhich] == pVertex) {
								ROCu_ins = 0.5*(pVertex->xdot + Clockwise*(pOther->xdot));	
							} else {
								ROCu_ins = 0.5*(pVertex->xdot + pOther->xdot);	
							};
						};
					};
				};

				ROCAverage_nT = pTri1->ROC_nT;
				Average_nT = pTri1->nT;
				ROCmomrate.x -= (ROCcc1.y-ROCu_ins.y)*Average_nT
											+ (cc1.y-u_ins.y)*ROCAverage_nT;
				ROCmomrate.y -= (ROCu_ins.x-ROCcc1.x)*Average_nT
											+ (u_ins.x-cc1.x)*ROCAverage_nT;
								
				// Now the usual triangles:
				for (i = 0; i < pVertex->triangles.len-1; i++)
				{
					inext = i+1; 
					pTri1 = ((Triangle *)pVertex->triangles.ptr[i]);
					pTri2 = ((Triangle *)pVertex->triangles.ptr[inext]);
					pTri1->ReturnCentre(&cc1,pVertex); // same tranche as pVert if periodic
					pTri2->ReturnCentre(&cc2,pVertex); 	// Model centroid as where we get nT.
					
					ROC[0] = pTri1->cornerptr[0]->xdot;
					ROC[1] = pTri1->cornerptr[1]->xdot;
					ROC[2] = pTri1->cornerptr[2]->xdot;
					if (pTri1->periodic > 0)
					{
						// important that it be relative to our vertex where acceleration is to be found! ..
						// wrapping status of corners is still per the original data
						if (pTri1->periodic == 1)
						{
							iWhich = pTri1->GetLeftmostIndex();
							if (pVertex->x > 0.0) {
								// bring that periodic one back
							    ROC[iWhich] = Clockwise*ROC[iWhich];
							} else {
								if (iWhich != 0) ROC[0] = Anticlockwise*ROC[0];
								if (iWhich != 1) ROC[1] = Anticlockwise*ROC[1];
								if (iWhich != 2) ROC[2] = Anticlockwise*ROC[2];
							};
						} else {
							// pTri1->periodic == 2
							iWhich = pTri1->GetRightmostIndex();
							if (pVertex->x > 0.0) {
								if (iWhich != 0) ROC[0] = Clockwise*ROC[0];
								if (iWhich != 1) ROC[1] = Clockwise*ROC[1];
								if (iWhich != 2) ROC[2] = Clockwise*ROC[2];
							} else {
								ROC[iWhich] = Anticlockwise*ROC[iWhich];
							};
						};
					};
					ROCcc1 = THIRD*(ROC[0] + ROC[1] + ROC[2]);
					// same for pTri2 .... 
					
					ROC[0] = pTri2->cornerptr[0]->xdot;
					ROC[1] = pTri2->cornerptr[1]->xdot;
					ROC[2] = pTri2->cornerptr[2]->xdot;
					
					if (pTri2->periodic > 0)
					{
						if (pTri2->periodic == 1)
						{
							iWhich = pTri2->GetLeftmostIndex();
							if (pVertex->x > 0.0) {
								// bring that periodic one back
							    ROC[iWhich] = Clockwise*ROC[iWhich];
							} else {
								if (iWhich != 0) ROC[0] = Anticlockwise*ROC[0];
								if (iWhich != 1) ROC[1] = Anticlockwise*ROC[1];
								if (iWhich != 2) ROC[2] = Anticlockwise*ROC[2];
							};
						} else {
							// pTri2->periodic == 2
							iWhich = pTri2->GetRightmostIndex();
							if (pVertex->x > 0.0) {
								if (iWhich != 0) ROC[0] = Clockwise*ROC[0];
								if (iWhich != 1) ROC[1] = Clockwise*ROC[1];
								if (iWhich != 2) ROC[2] = Clockwise*ROC[2];
							} else {
								ROC[iWhich] = Anticlockwise*ROC[iWhich];
							};
						};
					};
					ROCcc2 = THIRD*(ROC[0] + ROC[1] + ROC[2]);

					// ROC_nT_ion calculated first to save on pow.
					ROCAverage_nT = 0.5*(pTri1->ROC_nT + pTri2->ROC_nT);
					Average_nT = 0.5*(pTri1->nT + pTri2->nT);
					
					//momrate.x -= (cc2.y-cc1.y)*Average_nT_ion/m_ion;
					ROCmomrate.x -= (ROCcc2.y-ROCcc1.y)*Average_nT
											+ (cc2.y-cc1.y)*ROCAverage_nT;
					ROCmomrate.y -= (ROCcc1.x-ROCcc2.x)*Average_nT
											+ (cc1.x-cc2.x)*ROCAverage_nT;
				};

				// Now the last edge:
				
				// Get ROCu_ins and ROCcc2 :

				pOther = pTri2->GetOtherBaseVertex(pVertex);
				ROC[0] = pTri2->cornerptr[0]->xdot;
				ROC[1] = pTri2->cornerptr[1]->xdot;
				ROC[2] = pTri2->cornerptr[2]->xdot;
					
				if (pTri2->periodic == 0) 
				{
					ROCu_ins = 0.5*(pOther->xdot + pVertex->xdot);
				} else {
					if (pTri2->periodic == 1)
					{
						iWhich = pTri2->GetLeftmostIndex();
						if (pVertex->x > 0.0) {
							// bring that periodic one back
							ROC[iWhich] = Clockwise*ROC[iWhich];
						} else {
							if (iWhich != 0) ROC[0] = Anticlockwise*ROC[0];
							if (iWhich != 1) ROC[1] = Anticlockwise*ROC[1];
							if (iWhich != 2) ROC[2] = Anticlockwise*ROC[2];
						};
						if (pTri2->cornerptr[iWhich] == pOther) {
							ROCu_ins = 0.5*(pVertex->xdot + Clockwise*(pOther->xdot));	
						} else {
							if (pTri2->cornerptr[iWhich] == pVertex) {
								ROCu_ins = 0.5*(pVertex->xdot + Anticlockwise*(pOther->xdot));	
							} else {
								ROCu_ins = 0.5*(pVertex->xdot + pOther->xdot);	
							};
						};
					} else {
						iWhich = pTri2->GetRightmostIndex();
						if (pTri2->cornerptr[iWhich] == pOther) {
							ROCu_ins = 0.5*(pVertex->xdot + Anticlockwise*(pOther->xdot));	
						} else {
							if (pTri2->cornerptr[iWhich] == pVertex) {
								ROCu_ins = 0.5*(pVertex->xdot + Clockwise*(pOther->xdot));	
							} else {
								ROCu_ins = 0.5*(pVertex->xdot + pOther->xdot);	
							};
						};
						if (pVertex->x > 0.0) {
							if (iWhich != 0) ROC[0] = Clockwise*ROC[0];
							if (iWhich != 1) ROC[1] = Clockwise*ROC[1];
							if (iWhich != 2) ROC[2] = Clockwise*ROC[2];
						} else {
							ROC[iWhich] = Anticlockwise*ROC[iWhich];
						};
					};					
				};
				ROCcc2 = THIRD*(ROC[0] + ROC[1] + ROC[2]);
	
				ROCAverage_nT = pTri2->ROC_nT;
				Average_nT = pTri2->nT;
				ROCmomrate.x -= (ROCu_ins.y-ROCcc2.y)*Average_nT
											+ (u_ins.y-cc2.y)*ROCAverage_nT;
				ROCmomrate.y -= (ROCcc2.x-ROCu_ins.x)*Average_nT
											+ (cc2.x-u_ins.x)*ROCAverage_nT;
				
				// be careful on this: stuff about staying on insulator etc
				
				// Now kill the radial component:
				//rhat = pVertex->AdvectedPosition[species]_ion/
				//	(pVertex->AdvectedPosition[species]_ion.modulus());
				//momrate -= rhat*(momrate.dot(rhat));
				
				// but we didn't do that, we applied the same to xdot.

				ROC_accel = ROCmomrate / pVertex->Polygon_mass;
			};
			
			// 4. ROC xdot = xdotdot:
			//pVertex->xdot = (pVertex->AdvectedPosition[species]0 - pVertex->AdvectedPosition[species])
			//							+ pVertex->Pressure_a_effect_ion_dis*acceleration;
			// sometimes followed by, xdot_r = 0

			pVertex->xdotdot =  pVertex->Pressure_a_effect_dis[species] * ROC_accel - pVertex->xdot;
			
			if (pVertex->flags >= 3)
			{
				// generally, accel will point dramatically inwards if you are at the insulator,
				// so if we followed up xdot with xdot_r == 0 then this xdotdot may be nonsense.
			
				// For now just try this? :
				rhat = pVertex->AdvectedPosition[species] / (pVertex->AdvectedPosition[species].modulus());
				pVertex->xdotdot -= rhat*(pVertex->xdotdot.dot(rhat));
				// is this true though?
			
				// xdot = f - f_r r^
				// xdotdot = fdot - f_rdot r^ - f_r r^dot
				// but f_r is not available here - it was what got removed from xdot already.
				// fdot is not available either
			};

			// 5. Now set putative coordinates:
			pVertex->temp2 = pVertex->AdvectedPosition[species] + htraj*pVertex->xdot + htraj*htraj*0.5*pVertex->xdotdot;
			
			if ((pVertex->temp2.x-pVertex->x > 2.0e-4) || (pVertex->temp2.x-pVertex->x < -2.0e-4)) {
				i = i;
			};
			if ((pVertex->temp2.x > 4.0) || (pVertex->temp2.x < -4.0)) {
				i = i;
			};
			if (pVertex->flags == 3)
			{
				pVertex->temp2.project_to_ins(temp2);
				pVertex->temp2 = temp2;
			};
			if (pVertex->flags == 4)
			{
				pVertex->temp2.project_to_radius(temp2, Outermost_r_achieved);
				pVertex->temp2 = temp2;
			};
			
			
			// Given that we estimate rate of change of accel, can we estimate there is a point where
			// the equation is actually achieved??
			// Probably not since accel probably heads off to the side as we progress.
			
			// xdot gets small as we get near but what happens to xdotdot?
			// xdot is small so area change is small and so xdotdot is also small?
				
			++pVertex;
		};	
		
		// Now test if that step failed: did something get too near to its surroundings too fast, for instance?
		// _________________________________________________________________________________________
		
		int broken_iterations = 0;
		do 
		{
			broken = false;
			guess_h = htraj;
			// Neue plan:
			// Test shoelace every triangle. See if flipped and/or if area has _diminished_ by too great a factor.
			pTri = T;
			for (iTri = 0; iTri < numTriangles; iTri++)
			{
				u[0] = pTri->cornerptr[0]->AdvectedPosition[species];
				u[1] = pTri->cornerptr[1]->AdvectedPosition[species];
				u[2] = pTri->cornerptr[2]->AdvectedPosition[species];
				
				U[0] = pTri->cornerptr[0]->temp2;
				U[1] = pTri->cornerptr[1]->temp2;
				U[2] = pTri->cornerptr[2]->temp2;
				
				if (pTri->periodic > 0) {
					if (pTri->periodic == 1) {
						// rotate (original) leftmost point to right	
						i = pTri->GetLeftmostIndex();
						u[i] = Clockwise*u[i];
						U[i] = Clockwise*U[i];
					} else {
						i = pTri->GetRightmostIndex();
						u[i] = Anticlockwise*u[i];
						U[i] = Anticlockwise*U[i];
					};
				};
				// LEAVE POINTS UNWRAPPED. Then periodic data of triangle doesn't go stale.
				value = (	u[0].x*u[1].y - u[1].x*u[0].y
									+	u[1].x*u[2].y - u[2].x*u[1].y
									+	u[2].x*u[0].y - u[0].x*u[2].y	);
				value_new = (U[0].x*U[1].y - U[1].x*U[0].y
									+	U[1].x*U[2].y - U[2].x*U[1].y
									+	U[2].x*U[0].y - U[0].x*U[2].y	);
				if (value_new*value < 0.0) {
					broken = true;
					guess_h = min(guess_h, htraj*0.2);
				} else {
					twoarea = fabs(value);
					twoarea_new = fabs(value_new);
					if (twoarea_new < 0.4*twoarea) {
						broken = true;
						guess_h = min(guess_h,htraj*(twoarea_new/(0.4*twoarea)));
					};
				};
				++pTri;			
			};

			if (broken) {
				guess_h *= 0.99;
		//		ratio = guess_h/htraj;
				htraj = guess_h;
				printf("htraj= %1.4E ",htraj);
		//		oneminus = 1.0-ratio;
				// tween halfway back to the existing system position right here until things are acceptable.
				// ^^^^^^^^^^^^^^^^^^^^^^^^^
				// Does not work because we used a quadratic not linear model of how position evolves !!
				
				// Make time half as much and have xdotdot stored for every vertex.

			//	pVertex->temp2 = pVertex->AdvectedPosition[species] + htraj*pVertex->xdot + htraj*htraj*0.5*pVertex->xdotdot;
			
				pVertex = X;
				for (iVertex = 0; iVertex < numVertices; iVertex++)
				{
					pVertex->temp2 = pVertex->AdvectedPosition[species] + htraj*pVertex->xdot + htraj*htraj*0.5*pVertex->xdotdot;
			
					if (pVertex->flags == 3)
					{
						pVertex->temp2.project_to_ins(temp2);
						pVertex->temp2 = temp2;
					};
					if (pVertex->flags == 4)
					{
						pVertex->temp2.project_to_radius(temp2, Outermost_r_achieved);
						pVertex->temp2 = temp2;
					};
					//pVertex->temp2.x = ratio*pVertex->temp2.x+oneminus*pVertex->x;
					//pVertex->temp2.y = ratio*pVertex->temp2.y+oneminus*pVertex->y);
					++pVertex;
				};	
				broken_iterations++;
			};
		} while (broken); 
				
		// If no problems and htraj < some max, increase htraj back to get us to our solution faster ..
		//  set attempt flag : don't attempt again if it fails but shorter step works! know if we are
		//  heading up or down of timestep.
		
		if ((broken_iterations == 0) && (htraj < MAXhtraj))
		{
			htraj *= 1.6;
			if (htraj > MAXhtraj) htraj = MAXhtraj;
		};
			
		// Now accept temporary values ... 
		// ______________________________
		pVertex = X;
		for (iVertex = 0; iVertex < numVertices; iVertex++)
		{
			pVertex->AdvectedPosition[species] = pVertex->temp2;
			++pVertex;
		};	
		
		// Test for convergence: is everything fairly close to converged?
		// _________________________________________________________
		
		// Planned xdot should all have been small compared to the move from pVertex->x,y. 
		// Let's say we should go 99.9% of the way?
		// Also small compared to dist to neighbour - for sure
		// Is that enough on its own?
		// Preferably would say that xdot stayed small this move!
		// Can we say smth about xdotdot ??
		// That xdot is not going to explode in magnitude in time 1 say ?
		
		if (broken_iterations == 0)
		{
			not_converged_enough = false;

			pTri = T;
			for (iTri = 0; iTri < numTriangles; iTri++)
			{
				// Test whether "rate of area change" - the amt it is modelled linearly as going to change
				// during the progress to the implied system position - is > a fraction of new area.

				// pTri->ROC_nT_ion = ROCarea*(-FIVETHIRDS)*pTri->nT_ion / area; // f '(g(x))g'(x)
				area_roc_over_area = fabs(pTri->ROC_nT/(FIVETHIRDS*pTri->nT));
				if (area_roc_over_area > 0.01) {
					not_converged_enough = true;
					break;
				};
				++pTri;
			};
			if (not_converged_enough == false)
			{
				pVertex = X;
				for (iVertex = 0; iVertex < numVertices; iVertex++)
				{
					//xdot is the distance that was seen towards the implied target
					compare = max(pVertex->xdot.modulus(),(pVertex->xdot+pVertex->xdotdot).modulus());
					//neighbour distance at new position is what counts for that:

					//neighdist = 0.0;
					//for (i = 0; i < pVertex->neighbours.len; i++)
					//{
					//	pNeigh = X + pVertex->neighbours.ptr[i];
					//	dist = GetPossiblyPeriodicDist(pVertex->temp2,pNeigh->temp2);
					//	neighdist = max(neighdist,dist);
					//}; // that is super slow.
					//// faster way: take sqrt(area) of each neighbouring triangle, stored. ?

					// Better one: we worked out ROCarea. Let's demand |ROCarea| < 0.01*area.

					to_existing.x = pVertex->AdvectedPosition[species].x - pVertex->x;
					to_existing.y = pVertex->AdvectedPosition[species].y - pVertex->y;
					if (compare > 5.0e-9 + 0.001*to_existing.modulus())
					{
						not_converged_enough = true;
						break;
					};
					++pVertex;
				};				
			};
		};

		iIterationsConvergence++;
	} while (not_converged_enough);
	
	printf("Species %d , did %d iterations.\n",
		species, iIterationsConvergence);

	// At least having got these positions, there is nothing further to do before placing cells on to the new bulk mesh.

	// That placement routine should actually get simpler. We also incorporate energy conservation to do compressive htg.
	// The bulk mesh creation is already done.
	// Completing acceleration stage is clearly not hard.

	// Then we have the E part which was more-or-less there but also now wants multimesh. Hopefully it may all work together.
	// Then we can decide some more how to finalise things and perhaps create E_k+1 on new mesh and create preimage info before we do acceleration stage.
	
	// If I concentrate this much can be completed this week.
	
	// That leaves us to deal then with: diffusion stage (including magnetised). Thermoelectric bit. Thermal force R.
	// MP of everything. Good big deal.
	
};
void TriMesh::FinishAdvectingMesh(TriMesh * pDestMesh)
{
	// Bear in mind that we are being given unwrapped moves --
	// we mapped left initially and we allow that some PB may have been crossed
	// And we do not bounce w0,w1 off boundary either

	// ... Now called directly after AverageVertexPositions.

	Vertex * XdestX = pDestMesh->X;
	Triangle * XdestT = pDestMesh->T;

	Vertex * pVertex = X;
	Vertex * pVertDest = pDestMesh->X;
	Triangle * pTri = T;
	Proto ** ptr;
	Triangle * pTridest = pDestMesh->T;
	real rr;
	long iVertex;
	real weight1,weight2;
	long iTri, i;
	Vector2 temp,pos, u_ins;

	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		pVertDest->flags = pVertex->flags;
		
		// SHOULD the following be necessary? (It does seem to be...)

		if (pVertDest->flags == 3) {
			pos.x = pVertex->x;
			pos.y = pVertex->y;
			pos.project_to_ins(u_ins);
			pVertDest->x = u_ins.x; pVertDest->y = u_ins.y;
		};
		if (pVertDest->flags == 4) {
			pos.x = pVertex->x;
			pos.y = pVertex->y;
			pos.project_to_radius(u_ins, Outermost_r_achieved);
			pVertDest->x = u_ins.x; pVertDest->y = u_ins.y;
		};

		// Apply insulator bounce-off:
		rr = pVertDest->x*pVertDest->x+pVertDest->y*pVertDest->y;
		if (rr < DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
		{
			// SHOULD NEVER HAPPEN!

			// change magnitude of vector:
			pVertDest->x *= DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER/rr;
			pVertDest->y *= DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER/rr;
		};

		//##############################################
				// Note that here we have not restored Vertex::has_periodic

		pVertDest->triangles.clear();
		ptr = pVertex->triangles.ptr;
		for (int iii = 0; iii < pVertex->triangles.len; iii++)
		{
			pVertDest->triangles.add(XdestT + (((Triangle *)(*ptr))-T));
			++ptr;
		};

		++pVertDest;
		++pVertex;
	};
	
	// *********************************************************

	pDestMesh->numTriangles = numTriangles;
	pDestMesh->numEdgeVerts = numEdgeVerts;
	
	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++) 
	{
		// copy over tri set initially, then deal with overlaps:
			
		// Note : pTri->cornerptr[0]  +  (Xdest.ion.T-ion.T)
		// would quite possibly give the wrong answer
		// we want
		// Xdest.ion.T + (pTri->cornerptr[0] - ion.T)
		
		pTridest->cornerptr[0] = XdestX + (pTri->cornerptr[0] - X);
		pTridest->cornerptr[1] = XdestX + (pTri->cornerptr[1] - X);
		pTridest->cornerptr[2] = XdestX + (pTri->cornerptr[2] - X);
		
		//if ((pTri->cornerptr[2] == INS_VERT) || (pTri->cornerptr[2] == HIGH_VERT))
		//{
		//	pTridest->cornerptr[2] = pTri->cornerptr[2];
		//} else {
		//};
		
		pTridest->flags = pTri->flags;
		
		pTridest->neighbours[0] = XdestT + (pTri->neighbours[0] - T);
		pTridest->neighbours[1] = XdestT + (pTri->neighbours[1] - T);
		pTridest->neighbours[2] = XdestT + (pTri->neighbours[2] - T);
		
		pTridest->periodic = pTri->periodic;
		
		++pTri;
		++pTridest;
	};

	// *********************************************************

	pVertDest = XdestX;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		if (pVertDest->x/pVertDest->y < -GRADIENT_X_PER_Y)
		{
			// went beyond left PB
			pVertDest->periodic_image(temp,1); 
			pVertDest->x = temp.x;
			pVertDest->y = temp.y;

			// Need to also rotate interpolated data:

			pVertDest->A = Clockwise3*pVertDest->A;
			pVertDest->AdvectedPosition[SPECIES_ELECTRON] = Clockwise*pVertDest->AdvectedPosition[SPECIES_ELECTRON];
			
			for (i = 0; i < pVertDest->triangles.len; i++)
				((Triangle *)(pVertDest->triangles.ptr[i]))->DecrementPeriodic();
		};
		if (pVertDest->x/pVertDest->y > GRADIENT_X_PER_Y)
		{
			// went beyond right PB
			pVertDest->periodic_image(temp,0);
			pVertDest->x = temp.x;
			pVertDest->y = temp.y;
			
			// Need to also rotate interpolated data:

			pVertDest->A = Anticlockwise3*pVertDest->A;
			pVertDest->AdvectedPosition[SPECIES_ELECTRON] = Anticlockwise*pVertDest->AdvectedPosition[SPECIES_ELECTRON];

			for (i = 0; i < pVertDest->triangles.len; i++)
				((Triangle *)(pVertDest->triangles.ptr[i]))->IncrementPeriodic();
		};
		pVertDest++;
	};

	// *********************************************************

	// Now we need to recalculate triangle transverse vectors for doing overlap tests and other stuff:

	pTri = pDestMesh->T;
	for (iTri = 0; iTri < pDestMesh->numTriangles; iTri++)
	{
		pTri->RecalculateEdgeNormalVectors(false);
		++pTri;
	};

	// *********************************************************

	// Here let's say we keep an index list of the affected triangles.
	GlobalAffectedTriIndexList.clear();

	// Do maintenance:
	int attempts = pDestMesh->DestroyOverlaps(1000);

	pDestMesh->Redelaunerize(true);
}

void TriMesh::AverageVertexPositionsAndInterpolate(TriMesh * pSrcMesh, bool bAveragePositions)
{
	// Create new mesh positions:
	Vertex * pVertex, * pVertSrc;
	Triangle * pTri;
	Vector2 u;
	long iVertex;
	real n_ion, n_neut, factor_ion, factor_neut;
	Vector3 A0,A1,A2;
	Vector2 pos0,pos1,pos2,emove0,emove1,emove2;

	pVertex = X;
	pVertSrc = pSrcMesh->X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		if (bAveragePositions == true)
		{
			n_ion = pVertSrc->ion.n;
			n_neut = pVertSrc->neut.n;
			factor_ion = n_ion/(n_ion+n_neut);
			factor_neut = n_neut/(n_ion+n_neut);
			pVertex->x = factor_ion*pVertSrc->AdvectedPosition0[SPECIES_ION].x + factor_neut*pVertSrc->AdvectedPosition0[SPECIES_NEUTRAL].x;
			pVertex->y = factor_ion*pVertSrc->AdvectedPosition0[SPECIES_ION].y + factor_neut*pVertSrc->AdvectedPosition0[SPECIES_NEUTRAL].y;
			// Note these are unwrapped, so this is not averaging two non-contiguous points.
		};
		
		// ReturnPointerToTriangleContainingPoint function is not intended to work on something that is not in the tranche.
		pVertex->PopulatePosition(u);
		if (u.x/u.y > GRADIENT_X_PER_Y) u = Anticlockwise*u;
		if (u.x/u.y < -GRADIENT_X_PER_Y) u = Clockwise*u;
		// work with u from here on out
		
		pTri = pSrcMesh->ReturnPointerToTriangleContainingPoint(
			(Triangle *)(pVertSrc->triangles.ptr[0]),
			u.x,u.y);
		
		// That won't actually do it!
		// We need to know which ADVECTED ION triangle it resides within!!
		// What a PITA.
		// To test triangle we actually have to test against each edge which is itself a faff.
		// There is NO other way?!
		// Live with using the source triangle's ion-advected image for now. 
		// Ions will not have moved a huge distance so we will be mostly close to the corner anyway.

		// Interpolate A, emove, phi:
		
		real beta[3];

		if (pTri->periodic == 0)
		{

			// More efficient way:

			GetInterpolationCoeffs(beta, u.x, u.y,
							// triangle coordinates:
								pTri->cornerptr[0]->AdvectedPosition[SPECIES_ION],
								pTri->cornerptr[1]->AdvectedPosition[SPECIES_ION],
								pTri->cornerptr[2]->AdvectedPosition[SPECIES_ION]);

			pVertex->AdvectedPosition0[SPECIES_ELECTRON] =
								beta[0] * pTri->cornerptr[0]->AdvectedPosition0[SPECIES_ELECTRON]
						+		beta[1] * pTri->cornerptr[1]->AdvectedPosition0[SPECIES_ELECTRON]
						+		beta[2] * pTri->cornerptr[2]->AdvectedPosition0[SPECIES_ELECTRON];
						// tweening unwrapped positions, no PBC involved in any data

			pVertex->A =
								beta[0] * pTri->cornerptr[0]->A
						+		beta[1] * pTri->cornerptr[1]->A
						+		beta[2] * pTri->cornerptr[2]->A;

			pVertex->phi = 
								beta[0] * pTri->cornerptr[0]->phi
						+		beta[1] * pTri->cornerptr[1]->phi
						+		beta[2] * pTri->cornerptr[2]->phi;
			// phi is most important on cells but in our setup, it lives on vertices also.			


			//Interpolate3( &(pVertex->A), // target address
			//					pVertex->x,pVertex->y,
			//				// triangle coordinates, now contiguous:
			//					pTri->cornerptr[0]->AdvectedPosition[SPECIES_ION],
			//					pTri->cornerptr[1]->AdvectedPosition[SPECIES_ION],
			//					pTri->cornerptr[2]->AdvectedPosition[SPECIES_ION],
			//				// data for tweening:
			//					pTri->cornerptr[0]->A,
			//					pTri->cornerptr[1]->A,
			//					pTri->cornerptr[2]->A);
			
		} else {
			// be careful: both electron move and ion dest position need to be rotated per
			// the parity of the original source triangle corners,
			// to give a contiguous image which will correspond to this unwrapped new position
			int parity[3];
			pTri->GetParity(parity); 
			// parity = 0 means it's on the left
			
			// usually this triangle will be one with this as one of the corners
			// think carefully about that
			
			if (u.x > 0.0) {
				// ensure we put all data on right
				if (parity[0] == 0) {
					pos0 = Clockwise*pTri->cornerptr[0]->AdvectedPosition[SPECIES_ION];
					emove0 = Clockwise*pTri->cornerptr[0]->AdvectedPosition0[SPECIES_ELECTRON];
					A0 = Clockwise3*pTri->cornerptr[0]->A;
				} else {
					pos0 = pTri->cornerptr[0]->AdvectedPosition[SPECIES_ION];
					emove0 = pTri->cornerptr[0]->AdvectedPosition0[SPECIES_ELECTRON];
					A0 = pTri->cornerptr[0]->A;
				};
				if (parity[1] == 0) {
					pos1 = Clockwise*pTri->cornerptr[1]->AdvectedPosition[SPECIES_ION];
					emove1 = Clockwise*pTri->cornerptr[1]->AdvectedPosition0[SPECIES_ELECTRON];
					A1 = Clockwise3*pTri->cornerptr[1]->A;
				} else {
					pos1 = pTri->cornerptr[1]->AdvectedPosition[SPECIES_ION];
					emove1 = pTri->cornerptr[1]->AdvectedPosition0[SPECIES_ELECTRON];
					A1 = pTri->cornerptr[1]->A;
				};
				if (parity[2] == 0) {
					pos2 = Clockwise*pTri->cornerptr[2]->AdvectedPosition[SPECIES_ION];
					emove2 = Clockwise*pTri->cornerptr[2]->AdvectedPosition0[SPECIES_ELECTRON];
					A2 = Clockwise3*pTri->cornerptr[2]->A;
				} else {
					pos2 = pTri->cornerptr[2]->AdvectedPosition[SPECIES_ION];
					emove2 = pTri->cornerptr[2]->AdvectedPosition0[SPECIES_ELECTRON];
					A2 = pTri->cornerptr[2]->A;
				};
			} else {
				// ensure we put all data on left
				if (parity[0] == 0) {
					pos0 = pTri->cornerptr[0]->AdvectedPosition[SPECIES_ION];
					emove0 = pTri->cornerptr[0]->AdvectedPosition0[SPECIES_ELECTRON];
					A0 = pTri->cornerptr[0]->A;
				} else {
					pos0 = Anticlockwise*pTri->cornerptr[0]->AdvectedPosition[SPECIES_ION];
					emove0 = Anticlockwise*pTri->cornerptr[0]->AdvectedPosition0[SPECIES_ELECTRON];
					A0 = Anticlockwise3*pTri->cornerptr[0]->A;
				};
				if (parity[1] == 0) {
					pos1 = pTri->cornerptr[1]->AdvectedPosition[SPECIES_ION];
					emove1 = pTri->cornerptr[1]->AdvectedPosition0[SPECIES_ELECTRON];
					A1 = pTri->cornerptr[1]->A;
				} else {
					pos1 = Anticlockwise*pTri->cornerptr[1]->AdvectedPosition[SPECIES_ION];
					emove1 = Anticlockwise*pTri->cornerptr[1]->AdvectedPosition0[SPECIES_ELECTRON];
					A1 = Anticlockwise3*pTri->cornerptr[1]->A;
				};				
				if (parity[2] == 0) {
					pos2 = pTri->cornerptr[2]->AdvectedPosition[SPECIES_ION];
					emove2 = pTri->cornerptr[2]->AdvectedPosition0[SPECIES_ELECTRON];
					A2 = pTri->cornerptr[2]->A;
				} else {
					pos2 = Anticlockwise*pTri->cornerptr[2]->AdvectedPosition[SPECIES_ION];
					emove2 = Anticlockwise*pTri->cornerptr[2]->AdvectedPosition0[SPECIES_ELECTRON];
					A2 = Anticlockwise3*pTri->cornerptr[2]->A;
				};
			};

			
			GetInterpolationCoeffs(beta,	u.x,u.y,
							// triangle coordinates:
								pos0,pos1,pos2);

			pVertex->AdvectedPosition0[SPECIES_ELECTRON] =
								beta[0] * emove0 +	beta[1] * emove1 + beta[2] * emove2;

			pVertex->A = beta[0] * A0	+	beta[1] * A1	+	beta[2] * A2;

			pVertex->phi = beta[0] * pTri->cornerptr[0]->phi	
								+		beta[1] * pTri->cornerptr[1]->phi		
								+		beta[2] * pTri->cornerptr[2]->phi;
		};

		// when we do wrap the vertex we also rotate emove and A .
		
		++pVertex;
		++pVertSrc;
	};
}

void TriMesh::CollectFunctionals()
{
	FILE * fp;
	long iTri;

	real Smass[3];
	real Smomr[3], Smomtheta[3], Smomz[3];
	real Sheat[3];
	real Sr[3], Srr[3], S_v_x_omega_r, S_v_x_omega_r_within_36, SBth, Snvv[3];
	real EE, BB;
	Vector2 rhat, cc, theta;
	real r;

	Triangle * pTri= T;
	ZeroMemory(&Smass,3*sizeof(real));
	ZeroMemory(&Smomr,3*sizeof(real));
	ZeroMemory(&Smomtheta,3*sizeof(real));
	ZeroMemory(&Smomz,3*sizeof(real));
	ZeroMemory(&Sheat,3*sizeof(real));
	ZeroMemory(&Sr,3*sizeof(real));
	ZeroMemory(&Srr,3*sizeof(real));
	ZeroMemory(&S_v_x_omega_r,sizeof(real));
	ZeroMemory(&S_v_x_omega_r_within_36,sizeof(real));
	ZeroMemory(&SBth,sizeof(real));
	ZeroMemory(&Snvv,3*sizeof(real));
	EE = 0.0;
	BB = 0.0;
	real Area = 0.0;

	
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		pTri->ReturnCentre(&cc, pTri);
		rhat = cc/cc.modulus();
		theta.x = -rhat.y; theta.y = rhat.x;

		Smass[0] += pTri->neut.mass;
		Smass[1] += pTri->ion.mass;
		Smass[2] += pTri->elec.mass;
		// want to know integral n v / integral n to give average ve
		Smomr[0] += pTri->neut.mom.dotxy(rhat);
		Smomtheta[0] += pTri->neut.mom.dotxy(theta);
		Smomz[0] += pTri->neut.mom.z;
		Smomr[1] += pTri->ion.mom.dotxy(rhat);
		Smomtheta[1] += pTri->ion.mom.dotxy(theta);
		Smomz[1] += pTri->ion.mom.z;
		Smomr[2] += pTri->elec.mom.dotxy(rhat);
		Smomtheta[2] += pTri->elec.mom.dotxy(theta);
		Smomz[2] += pTri->elec.mom.z;
		Sheat[0] += pTri->neut.heat;
		Sheat[1] += pTri->ion.heat;
		Sheat[2] += pTri->elec.heat;
		r = cc.modulus();
		Sr[0] += pTri->neut.mass*r;
		Sr[1] += pTri->ion.mass*r;
		Sr[2] += pTri->elec.mass*r;
		Srr[0] += pTri->neut.mass*r*r;
		Srr[1] += pTri->ion.mass*r*r;
		Srr[2] += pTri->elec.mass*r*r;

		// experience of - (v x omega)_r : qBth/c vez --- no m, we divide by M
		
		// rhat dot (p x B) = (x/r) (p x B)_x + (y/r) (p x B)_y 

		S_v_x_omega_r -= q/c* ( (cc.x/r)*( pTri->elec.mom.y*pTri->B.z - pTri->elec.mom.z *pTri->B.y)
												+ (cc.y/r)*( pTri->elec.mom.z*pTri->B.x - pTri->elec.mom.x *pTri->B.z));
		
		SBth += pTri->elec.mass*((-cc.y/r)*pTri->B.x + (cc.x/r)*pTri->B.y);

		if (r < 3.6) {
			S_v_x_omega_r_within_36 -= q/c* ( (cc.x/r)*( pTri->elec.mom.y*pTri->B.z - pTri->elec.mom.z *pTri->B.y)
																		+ (cc.y/r)*( pTri->elec.mom.z*pTri->B.x - pTri->elec.mom.x *pTri->B.z));
		};
		
		// add integral n v^2 : =
		 Snvv[0] += pTri->neut.mom.dot(pTri->neut.mom)/pTri->neut.mass;
		 Snvv[1] += pTri->ion.mom.dot(pTri->ion.mom)/pTri->ion.mass;
		 Snvv[2] += pTri->elec.mom.dot(pTri->elec.mom)/pTri->elec.mass;
		
		 EE += pTri->area * pTri->E.dot(pTri->E);
		 BB += pTri->area * pTri->B.dot(pTri->B);
		
		 Area += pTri->area;

		 // leave \n for when we have added heat changes
		++pTri;
	}

	 fp = fopen(FUNCTIONALFILENAME,"a");

	 fprintf(fp, "%d %1.8E  %1.8E  %1.8E %1.8E %1.8E  "
		 " %1.8E %1.8E %1.8E %1.8E %1.8E %1.8E "
		 " %1.8E %1.8E %1.8E %1.8E %1.8E %1.8E %1.8E %1.8E %1.8E "
		 " %1.8E %1.8E %1.8E  %1.8E %1.8E %1.8E   ",
		 GlobalStepsCounter, evaltime+ZCURRENTBASETIME, Area, Smass[0],Smass[1],Smass[2],
		 Sr[0]/Smass[0], Sr[1]/Smass[1], Sr[2]/Smass[2], 
		 sqrt(Srr[0]/Smass[0]-(Sr[0]/Smass[0])*(Sr[0]/Smass[0])),
		 sqrt(Srr[1]/Smass[1]-(Sr[1]/Smass[1])*(Sr[1]/Smass[1])),
		 sqrt(Srr[2]/Smass[2]-(Sr[2]/Smass[2])*(Sr[2]/Smass[2])),
		 Smomr[0]/Smass[0], Smomtheta[0]/Smass[0], Smomz[0]/Smass[0],
		 Smomr[1]/Smass[1], Smomtheta[1]/Smass[1], Smomz[1]/Smass[1],
		 Smomr[2]/Smass[2], Smomtheta[2]/Smass[2], Smomz[2]/Smass[2],
		 Sheat[0], Sheat[1], Sheat[2], Sheat[0]/Smass[0], Sheat[1]/Smass[1], Sheat[2]/Smass[2]
		 );

	 fprintf(fp, " %1.6E %1.6E %1.6E %1.9E %1.9E %1.8E %1.6E %1.6E  ",
					THIRD*m_neutral*Snvv[0], THIRD*m_ion*Snvv[1],THIRD*m_e*Snvv[2], S_v_x_omega_r/Smass[2], S_v_x_omega_r_within_36/Smass[2],
					SBth/Smass[2], TWOTHIRDS*EE/(8.0*PI), TWOTHIRDS*BB/(8.0*PI));
				
	fclose(fp);
}

void TriMesh::CalculateIndirectNeighbourLists()
{
	// For every cell, we must list every cell that affects it.

	// New system: index into the entire mesh regardless of whether inside/outside insulator boundary.

	long iTri, iVertex;
	Triangle * pTri;
	Vertex * pVertex;
	long index;
	Vector2 u;
	real angle, distance, totalweight;

	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		pTri->indexlist.Clear();
		pTri->numIndirect = 0;
		
		pTri->indexlist.add(iTri);
		pTri->indexlist.add_unique(pTri->neighbours[0]-T);
		pTri->indexlist.add_unique(pTri->neighbours[1]-T);
		pTri->indexlist.add_unique(pTri->neighbours[2]-T);
		
		for (iCorner = 0; iCorner < 3; iCorner++)
		{
			pVertex = pTri->cornerptr[iCorner];
			for (i =0; i < pVertex->neighbours.len; i++)
			{
				index = ((Triangle *)(pVertex->neighbours.ptr[i]))-T;
				pTri->indexlist.add_unique(index);
			};
		};
		pTri->numIndirect = pTri->indexlist.len; // check that works.

		++pTri;
	}

	
	// 0. Each vertex creates averaging coefficients:

	// We use angle divided by distance to centroid??

	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		// Averaging for v, so we should ignore cells outside the plasma domain.
		
		// Weight: angle / distance ??

		// What is sense of that?

		// If we could switch to vertex-based, averaging from 3 is easy because, just create a plane from the 3.
		
		pVertex->coefficients.clear();

		// * Arrange it elsewhere so that
		// first triangle in list is the most clockwise triangle inside plasma domain,
		// and then we go anticlockwise.

		u.x = pVertex->x;
		u.y = pVertex->y;
		
		totalweight = 0.0;
		for (i = 0; i < triangles.len; i++)
		{
			pTri = (Triangle *)(pVertex->triangles.ptr[i]);
		//	if (pTri->u8domain_flag == PLASMA_DOMAIN)
			{
				// Count all triangles at every vertex here.
				// If we want domain-only sometimes, adjust how it is used at those times.

				angle = pTri->ReturnAngle(pVertex);
				pTri->GenerateContiguousCentroid(&centroid, pTri);
				distance = GetPossiblyPeriodicDist(u,centroid);
				pVertex->coefficients.add(angle/distance);
				totalweight += angle/distance;
			};
		};
		
		if (totalweight == 0.0) {
			// maybe this happens if there are no valid triangles for whatever reason.
		} else {
			for (i = 0; i < pVertex->coefficients.len; i++)
				pVertex->coefficients.ptr[i] /= totalweight;
		};
		++pVertex;

		// But insulator vertices also need to set their radial velocity to zero.
		// That has to be borne in mind in the following loop.

	};

}

void TriMesh::Calculate_Electron_Viscous_Momflux_Coefficients()
{
	
	long iTri, iVertex;
	Vertex * pVertex, *pVert1, *pVert2;
	Triangle * pTri, *pNeigh ,*pSrc;
	long index;
	Vector2 u[3], centroid, uNeigh;
	real shoelace;
	int i, iNeigh, iprev, inext;
	
	Vector3 B, unitB, unit_perp, unit_Hall;
	Vector2 coeff_neigh, coeff_self, coeff_inext, coeff_iprev;

	real eta_par, eta_perp, eta_cross;

	real dvx_by_dx[4], dvy_by_dx[4], dvz_by_dx[4], dvx_by_dy[4],
	dvy_by_dy[4], dvz_by_dy[4];

	Vector3 intermed;
	Vector3 dvb_by_db[4], dvperp_by_db[4], dvHall_by_db[4],
		dvb_by_dperp[4], dvperp_by_dperp[4], dvHall_by_dperp[4],
		dvb_by_dHall[4], dvperp_by_dHall[4], dvHall_by_dHall[4];
	Vector3 Pi_b_b[4], Pi_P_b[4], Pi_P_P[4],
			Pi_H_b[4], Pi_H_P[4], Pi_H_H[4];

	Vector3 momflux_px_directionx, momflux_px_directiony,
			momflux_py_directionx, momflux_py_directiony,
			momflux_pz_directionx, momflux_pz_directiony,
			momflux_px[4],momflux_py[4],momflux_pz[4],
			col_b_x[4], col_b_y[4], col_P_x[4],col_P_y[4],
			col_H_x[4], col_H_y[4];

	Tensor2 effect_v_tri_on_v_vertex;
	Vector2 radial;

	real const FACTOR_TO_KAPPA = 0.73/2.5; // anyone's guess ...
	real const FACTOR_PERP = 1.2/0.96;
	real const FACTOR_HALL = 1.0/0.96;

	
	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		if (pTri->visc_coeffs != NULL) delete[] pTri->visc_coeffs;

		pTri->visc_coeffs = new Tensor3[pTri->numIndirect];

		ZeroMemory(pTri->visc_coeffs,sizeof(Tensor3)*pTri->numIndirect);

		// See if we can remove this, in a production version:

		pTri->RecalculateEdgeNormalVectors(false);

		++pTri;
	};

	// Get kappa parallel (on tris) to get eta parallel :
	Recalculate_NuHeart_and_KappaParallel_OnVertices_And_Triangles(SPECIES_ELECTRON);

	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		// How to calc these?

		if (pTri->u8domain_flag != PLASMA_DOMAIN) {
			
			// What coefficients to assign in this case?
			// We should not be creating epsilon Ohm outside this domain
			// so it doesn't matter.

		} else {
			if (pTri->periodic == 0) {
				pTri->PopulatePositions(u[0],u[1],u[2]);
			} else {
				pTri->MapLeft(u[0],u[1],u[2]);
			};

			for (iNeigh = 0; iNeigh < 3; iNeigh++)
			{

				pNeigh = pTri->neighbours[iNeigh] ;
				if ((pNeigh != pTri) && (pNeigh->u8domain_flag == PLASMA_DOMAIN))
				{
					// Note: the reason we do not do pNeigh < pTri and use symmetry
					// at each edge, is that we cannot be sure pNeigh is not across the
					// PB from pTri.


					// use: d vb / da = b transpose [ dvi/dxj ] a
					// Prototypical element: ax by dvy/dx

					// Pick 3 directions:
					// __________________
					// For the sake of argument use edge_normal - b to make perpendicular
					// Since v is not known already --- ???
					// We could use an unknown direction but this way is easier.
					
					// First estimate B_edge:

					B = 0.5*(pTri->B + pNeigh->B);
					unitB = B/B.modulus();
					unit_perp = pTri->edge_normal[iNeigh]-unitB*(pTri->edge_normal[iNeigh].dot(unitB));
					unit_perp = unit_perp/unit_perp.modulus();
					unit_Hall = unitB.cross(unit_perp); // Note sign.

					// pTri->scratch[5] = kappa_e
					eta_par = 0.5*m_e*FACTOR_TO_KAPPA*
						(pTri->scratch[5] + pNeigh->scratch[5]);
						// kappa_e_par = 2.5*n_e*T_e/(m_e*nu_e_heart)
						// eta_par = 0.73*n_e*T_e/nu_e_heart

					eta_perp = FACTOR_PERP * nu*nu/(omegasq + nu*nu);
					eta_cross = FACTOR_HALL * eta_par * nu*omegamod/(omegasq + nu*nu);

					// At each edge, there is a mom flux...

					// 1. Calculate dv_xyz / d_xy as lc of nearby v_xyz

					// Firstly shoelace for grad v :
					// Each component of v is a scalar function:
					
			//shoelace =		uNeigh0.x * (u2.y - u1.y)
			//			+	u2.x * (uNeigh1.y-uNeigh0.y)
			//			+	uNeigh1.x * (u0.y-u2.y)
			//			+	u0.x * (uNeigh2.y-uNeigh1.y)
			//			+	uNeigh2.x * (u1.y-u0.y)
			//			+	u1.x * (uNeigh0.y-uNeigh2.y);
			//			
			//pTri->GradTe.x = (	Tneigh0 * (u2.y-u1.y)
			//			+	Tcorner2 * (uNeigh1.y-uNeigh0.y)
			//			+	Tneigh1 * (u0.y-u2.y)
			//			+	Tcorner0 * (uNeigh2.y-uNeigh1.y)
			//			+	Tneigh2 * (u1.y-u0.y)
			//			+	Tcorner1 * (uNeigh0.y-uNeigh2.y)
			//				)/shoelace;
			//
			//pTri->GradTe.y = (	Tneigh0 * (u1.x - u2.x)
			//			+	Tcorner2 * (uNeigh0.x - uNeigh1.x)
			//			+	Tneigh1 * (u2.x - u0.x)
			//			+	Tcorner0 * (uNeigh1.x - uNeigh2.x)
			//			+	Tneigh2 * (u0.x - u1.x)
			//			+	Tcorner1 * (uNeigh2.x - uNeigh0.x)
			//				) / shoelace;
			
					// create contiguous-image quadrilateral of positions:
					
					iprev = iNeigh-1; if (iprev < 0) iprev = 2;
					inext = iNeigh+1; if (inext > 2) inext = 0;
					pVert1 = pTri->cornerptr[iprev];
					pVert2 = pTri->cornerptr[inext];

					pTri->GenerateContiguousCentroid(&centroid,pTri);
					pNeigh->GenerateContiguousCentroid(&uNeigh,pTri);
					

					shoelace =	uNeigh.x*(u[inext].y-u[iprev].y)
							  +	u[inext].x*(centroid.y-uNeigh.y)
							  + centroid.x*(u[iprev].y-u[inext].y)
							  + u[iprev].x*(uNeigh.y-centroid.y);
					
					// coefficients to produce df/dx
					coeff_neigh.x = (u[inext].y-u[iprev].y)/shoelace;
					coeff_inext.x = (centroid.y-uNeigh.y)/shoelace;
					coeff_self.x = (u[iprev].y-u[inext].y)/shoelace;
					coeff_iprev.x = (uNeigh.y-centroid.y)/shoelace;
					
					// coefficients to produce df/dy
					coeff_neigh.y = (u[iprev].x-u[inext].x)/shoelace;
					coeff_inext.y = (uNeigh.x-centroid.x)/shoelace;
					coeff_self.y = (u[inext].x-u[iprev].x)/shoelace;
					coeff_iprev.y = (centroid.x-uNeigh.x)/shoelace;
					
					//gradvz = coeff_neigh * vz_neigh + coeff_inext * vz_next + etc
					
					// 0 = coeff on self
					// 1 = coeff on vertex 1
					// 2 = coeff on neigh
					// 3 = coeff on vertex 2

					// For clarity, fill them in:
					dvx_by_dx[0] = coeff_self.x;
					dvx_by_dx[1] = coeff_inext.x;
					dvx_by_dx[2] = coeff_neigh.x;
					dvx_by_dx[3] = coeff_iprev.x; // coeff on vx

					dvy_by_dx[0] = coeff_self.x;
					dvy_by_dx[1] = coeff_inext.x;
					dvy_by_dx[2] = coeff_neigh.x;
					dvy_by_dx[3] = coeff_iprev.x; // coeff on vy

					dvz_by_dx[0] = coeff_self.x;
					dvz_by_dx[1] = coeff_inext.x;
					dvz_by_dx[2] = coeff_neigh.x;
					dvz_by_dx[3] = coeff_iprev.x; // coeff on vz

					dvx_by_dy[0] = coeff_self.y;
					dvx_by_dy[1] = coeff_inext.y;
					dvx_by_dy[2] = coeff_neigh.y;
					dvx_by_dy[3] = coeff_iprev.y; // coeff on vx

					dvy_by_dy[0] = coeff_self.y;
					dvy_by_dy[1] = coeff_inext.y;
					dvy_by_dy[2] = coeff_neigh.y;
					dvy_by_dy[3] = coeff_iprev.y; // coeff on vy

					dvz_by_dy[0] = coeff_self.y;
					dvz_by_dy[1] = coeff_inext.y;
					dvz_by_dy[2] = coeff_neigh.y;
					dvz_by_dy[3] = coeff_iprev.y; // coeff on vz

					// We do not here split out the triangles affecting the corner vertices;
					// that would cause us to have to then handle each one separately through all
					// the following. 
					
					// We deal here with contiguous-evaluated v to work out Div Pi so it's not the problem.
					
					// Do we get around the edge vertex vr=0 complications if we worked with vertex-based cells?
					// Then we are concerned with walls between ins verts, but that's okay. So yes.
					
					
					for (i = 0; i < 4; i++) // for each of 4 edge-relevant locations.
					{
						
						// 2. Now get partials in magnetic coordinates as lc of nearby v_xyz
						
						// we want to create 9 partials, as lc's of v _ xyz at the 4 corners.
						// (We can then branch out to what this means for coefficients on triangle v's at the end.)
						
						// suppose we gathered dvx_by_dx[0] meaning the coefficient on vx at location 0
						
						intermed.x = unitB.x * dvx_by_dx[i] + 
									 unitB.y * dvx_by_dy[i] ;
						intermed.y = unitB.x * dvy_by_dx[i] +
									 unitB.y * dvy_by_dy[i] ;
						intermed.z = unitB.x * dvz_by_dx[i] +
									 unitB.y * dvz_by_dy[i] ;
						// These are all the same!!
						
						dvb_by_db[i].x = unitB.x * intermed.x;
						dvb_by_db[i].y = unitB.y * intermed.y;
						dvb_by_db[i].z = unitB.z * intermed.z;
						
						dvperp_by_db[i].x = unit_perp.x * intermed.x;
						dvperp_by_db[i].y = unit_perp.y * intermed.y;
						dvperp_by_db[i].z = unit_perp.z * intermed.z;
						
						dvHall_by_db[i].x = unit_Hall.x * intermed.x;
						dvHall_by_db[i].y = unit_Hall.y * intermed.y;
						dvHall_by_db[i].z = unit_Hall.z * intermed.z;
						
						intermed.x = unit_perp.x * dvx_by_dx[i] +
									 unit_perp.y * dvx_by_dy[i] ;
						intermed.y = unit_perp.x * dvy_by_dx[i] + 
									 unit_perp.y * dvy_by_dy[i] ;
						intermed.z = unit_perp.x * dvz_by_dx[i] + 
									 unit_perp.y * dvz_by_dy[i] ;
						// These are all the same!
						
						dvb_by_dperp[i].x = unitB.x * intermed.x;
						dvb_by_dperp[i].y = unitB.y * intermed.y;
						dvb_by_dperp[i].z = unitB.z * intermed.z;
						
						dvperp_by_dperp[i].x = unit_perp.x * intermed.x;
						dvperp_by_dperp[i].y = unit_perp.y * intermed.y;
						dvperp_by_dperp[i].z = unit_perp.z * intermed.z;
						
						// coeff on vx (contiguous x-y) at locn i :
						dvHall_by_dperp[i].x = unit_Hall.x * intermed.x;

						dvHall_by_dperp[i].y = unit_Hall.y * intermed.y;
						dvHall_by_dperp[i].z = unit_Hall.z * intermed.z;
						
						intermed.x = unit_Hall.x * dvx_by_dx[i] +
									 unit_Hall.y * dvx_by_dy[i] ;
						intermed.y = unit_Hall.x * dvy_by_dx[i] + 
									 unit_Hall.y * dvy_by_dy[i] ;
						intermed.z = unit_Hall.x * dvz_by_dx[i] + 
									 unit_Hall.y * dvz_by_dy[i] ;
						
						// These are all the same!
						
						dvb_by_dHall[i].x = unitB.x * intermed.x;
						dvb_by_dHall[i].y = unitB.y * intermed.y;
						dvb_by_dHall[i].z = unitB.z * intermed.z;
						
						dvperp_by_dHall[i].x = unit_perp.x * intermed.x;
						dvperp_by_dHall[i].y = unit_perp.y * intermed.y;
						dvperp_by_dHall[i].z = unit_perp.z * intermed.z;
						
						dvHall_by_dHall[i].x = unit_Hall.x * intermed.x;
						dvHall_by_dHall[i].y = unit_Hall.y * intermed.y;
						dvHall_by_dHall[i].z = unit_Hall.z * intermed.z;
						
						// used: d vb / da = b transpose [ dvi/dxj ] a
						// Prototypical element: a.x b.y dvy/dx
						// the /dz determines that it's .z for the /d element

						// ******************************************************************
						// 3. Therefore get mom flux vectors that form Pi_bPH as lc's.
					
						// Pi_Perp_b = - eta_perp (dv_perp/db + dvb/dperp) - eta_cross (dv_Hall/db + dv_b/dHall);
								
						// Coefficient on vx at location i comes through coefficient in the contributing 
						// dvb/dperp type terms. 
					
						// Therefore can work out each location's coefficient the same way
					
						//Pi_perp_b[i].x = 
						//Pi_perp_b[i].y = 
						//Pi_perp_b[i].z = 					
						// Happily can combine at least to Pi_perp_b[i]:
						
						Pi_b_b[i] = -eta_par*( 2.0*dvb_by_db[i] - TWOTHIRDS*
							(dvb_by_db[i] + dvperp_by_dperp[i] + dvHall_by_dHall[i] );

						Pi_P_b[i] = -eta_perp*( dvperp_by_db[i] + dvb_by_dperp[i] )
									- eta_cross*(dvHall_by_db[i] + dvb_by_dHall[i]);

						Pi_P_P[i] = -eta_par*(dvperp_by_dperp[i] + dvHall_by_dHall[i] - TWOTHIRDS*
							(dvb_by_db[i] + dvperp_by_dperp[i] + dvHall_by_dHall[i] ))
									
							-0.25*eta_perp*(dvperp_by_dperp[i] - dvHall_by_dHall[i])

							-0.5*eta_cross*(dvHall_by_dperp[i] + dvperp_by_dHall[i]);

						
						Pi_H_b[i] = -eta_perp*(dvHall_by_db[i] + dvb_by_dHall[i])
									+ eta_cross*(dvperp_by_db[i] + dvb_by_dperp[i]);

						Pi_H_P[i] = -0.25*eta_perp*(dvHall_by_dperp[i] + dvperp_by_dHall[i])
									+ 0.5*eta_cross*(dvperp_by_dperp[i] + dvHall_by_dHall[i]);

						Pi_H_H[i] = -eta_par*(dvperp_by_dperp[i] + dvHall_by_dHall[i] - TWOTHIRDS*
							(dvb_by_db[i] + dvperp_by_dperp[i] + dvHall_by_dHall[i] ))
									
							+ 0.25*eta_perp*(dvperp_by_dperp[i] - dvHall_by_dHall[i])
							+ 0.5*eta_cross*(dvHall_by_dperp[i] + dvperp_by_dHall[i]);
									

						// Now instead of calc'ing P_xyz we shall calculate the columns of
						//  ( bx  Px  Hx )  ( Pi_bb Pi_Pb Pi_Hb )
						//  ( by  Py  Hy )  ( Pi_bP Pi_PP Pi_HP )
						//  ( bz  Pz  Hz )  ( Pi_bH Pi_PH Pi_HH )

						//  mn dv_j/dt += - b_j Vec_b - P_j vec_P - H_j vec_H
						// allegedly.


					
						// Each of these 9 elements is a lc of 12 variables. !
						
						col_b_x[i] = unitB.x*Pi_b_b[i] + unit_perp.x*Pi_P_b[i] + unit_Hall.x*Pi_H_b[i];
						col_b_y[i] = unitB.y*Pi_b_b[i] + unit_perp.y*Pi_P_b[i] + unit_Hall.y*Pi_H_b[i];
						//col_b_z[i] = unitB.z*Pi_b_b[i] + unit_perp.z*Pi_P_b[i] + unit_Hall.z*Pi_H_b[i];
						
						// Pi_bPH symmetric

						col_P_x[i] = unitB.x*Pi_P_b[i] + unit_perp.x*Pi_P_P[i] + unit_Hall.x*Pi_H_P[i];
						col_P_y[i] = unitB.y*Pi_P_b[i] + unit_perp.y*Pi_P_P[i] + unit_Hall.y*Pi_H_P[i];
						//col_P_z[i] = unitB.z*Pi_P_b[i] + unit_perp.z*Pi_P_P[i] + unit_Hall.z*Pi_H_P[i];

						col_H_x[i] = unitB.x*Pi_H_b[i] + unit_perp.x*Pi_H_P[i] + unit_Hall.x*Pi_H_H[i];
						col_H_y[i] = unitB.y*Pi_H_b[i] + unit_perp.y*Pi_H_P[i] + unit_Hall.y*Pi_H_H[i];
						//col_H_z[i] = unitB.z*Pi_H_b[i] + unit_perp.z*Pi_H_P[i] + unit_Hall.z*Pi_H_H[i];

						// For px, the momentum flux vector in directions (x,y) depends on vx,y,z at 4 locations.
						// The z-components just calculated are not important in 2D,
						// as only the x,y components of the flux vector will be dotted with edge_normal.

						// Each of these is a 3-vector and the .x indicates dependence on vx at location i.
						momflux_px_directionx = - unitB.x*col_b_x[i] - unit_perp.x*col_P_x[i]
												- unit_Hall.x*col_H_x[i];
							
						momflux_px_directiony = - unitB.x*col_b_y[i] - unit_perp.x*col_P_y[i]
												- unit_Hall.x*col_H_y[i];

						momflux_py_directionx = - unitB.y*col_b_x[i] - unit_perp.y*col_P_x[i]
												- unit_Hall.y*col_H_x[i];

						momflux_py_directiony = - unitB.y*col_b_y[i] - unit_perp.y*col_P_y[i]
												- unit_Hall.y*col_H_y[i];

						momflux_pz_directionx = - unitB.z*col_b_x[i] - unit_perp.z*col_P_x[i]
												- unit_Hall.z*col_H_x[i];

						momflux_pz_directiony = - unitB.z*col_b_y[i] - unit_perp.z*col_P_y[i]
												- unit_Hall.z*col_H_y[i];

					// 4. Therefore get the inward mom flux by dotting with the edge normal -- 
					// make sure it was calculated.

												
						// This will be contribution to m n dvx/dt in cell:
						momflux_px[i] =	momflux_px_directionx*pTri->edge_normal[iNeigh].x
									  +	momflux_px_directiony*pTri->edge_normal[iNeigh].y;

						momflux_py[i] = momflux_py_directionx*pTri->edge_normal[iNeigh].x
									  +	momflux_py_directiony*pTri->edge_normal[iNeigh].y;

						momflux_pz[i] = momflux_pz_directionx*pTri->edge_normal[iNeigh].x
									  +	momflux_pz_directiony*pTri->edge_normal[iNeigh].y;
					};
					
					// momflux_px[i].y = effect of vy at location i on d/dt px = d/dt( m n vx )
				
					// 5. Accumulate coefficients in triangle:
					// dv_xyz/dt = linear [v_self, v_neighs]

					// I think we are preferring to add to this triangle and its neighbour
					// at the same time - so bring in condition above that iTri > iNeighTri -- 
					// even though we will have to go off and do FindIndex for everything twice.

					// Obviously dv/dt = (1/nm) sum of inward mom flows ...
					
					// Want coefficients on v xyz to affect v xyz:
					
					factor = 1.0/(n_e*m_e);

					// The self-effect is the simplest:

					// sign: visc_coeffs shall be + dv/dt 

					pTri->visc_coeffs[0].xx += momflux_px[0].x*factor;
					pTri->visc_coeffs[0].xy += momflux_px[0].y*factor;
					pTri->visc_coeffs[0].xz += momflux_px[0].z*factor;
					// xy is coeff on v.y to get a.x
					
					pTri->visc_coeffs[0].yx += momflux_py[0].x*factor;
					pTri->visc_coeffs[0].yy += momflux_py[0].y*factor;
					pTri->visc_coeffs[0].yz += momflux_py[0].z*factor;

					pTri->visc_coeffs[0].zx += momflux_pz[0].x*factor;
					pTri->visc_coeffs[0].zy += momflux_pz[0].y*factor;
					pTri->visc_coeffs[0].zz += momflux_pz[0].z*factor;
					
					// Now we have to do a mapping: each of the triangles that affected something,
					// has a place in the index list

					iWhich = pTri->indexlist.FindIndex(pNeigh-T);
					pTri->visc_coeffs[iWhich].xx += momflux_px[2].x*factor;
					pTri->visc_coeffs[iWhich].xy += momflux_px[2].y*factor;
					pTri->visc_coeffs[iWhich].xz += momflux_px[2].z*factor;

					pTri->visc_coeffs[iWhich].yx += momflux_py[2].x*factor;
					pTri->visc_coeffs[iWhich].yy += momflux_py[2].y*factor;
					pTri->visc_coeffs[iWhich].yz += momflux_py[2].z*factor;

					pTri->visc_coeffs[iWhich].zx += momflux_pz[2].x*factor;
					pTri->visc_coeffs[iWhich].zy += momflux_pz[2].y*factor;
					pTri->visc_coeffs[iWhich].zz += momflux_pz[2].z*factor;

					// Note, coefficients refer to contiguous velocity.
					

					// And here recall that edge vertices need a special treatment!
					
					for (iSrc = 0; iSrc < pVert1->coefficients.len; iSrc++)
					{
						pSrc = (Triangle *)(pVert1->triangles.ptr[iSrc]);
						index = pTri->indexlist.FindIndex(pSrc-T);
						
						if (pVert1->flags != 3) {
							// otherwise happy to use all tris at the vertex and not manipulate v_vertex
							
							
							coefficient = pVert1->coefficients.ptr[iSrc]*factor;
							
							// We are assuming here that when the contribution is made,
							// we already will have rotated the contributing velocity to be contiguous.
							// coefficients.ptr[iSrc] tells us the ppn at which the (contig) v in tri
							// is averaged to give (contig) v at vertex.
							
							pTri->visc_coeffs[index].xx += momflux_px[1].x*coefficient;
							pTri->visc_coeffs[index].xy += momflux_px[1].y*coefficient;
							pTri->visc_coeffs[index].xz += momflux_px[1].z*coefficient;
							
							pTri->visc_coeffs[index].yx += momflux_py[1].x*coefficient;
							pTri->visc_coeffs[index].yy += momflux_py[1].y*coefficient;
							pTri->visc_coeffs[index].yz += momflux_py[1].z*coefficient;
							
							pTri->visc_coeffs[index].zx += momflux_pz[1].x*coefficient;
							pTri->visc_coeffs[index].zy += momflux_pz[1].y*coefficient;
							pTri->visc_coeffs[index].zz += momflux_pz[1].z*coefficient;
							
							// And life would be nicer if only we were vertex-based.

						} else {
							// In the case that this is an edge vertex: set vr from here = 0
							// factor = 1.0/(n_e*m_e);
							// momflux_pxyz is the amt of additional v_xyz, coefficient on v at corner 1

							// but sometimes the corner has v_r = 0 required

							// which means instead of being a1 v1 + a2 v2
							// it is (a1 v1 + a2 v2) - rhat (a1 v1+a2 v2).dot(rhat)

							// a1 (v1-rhat(v1.rhat)) = a1(v1 - r(v1.r)/r.r)

							if (pSrc->u8domain_flag == PLASMA_DOMAIN) {

								coefficient = 2.0 * pVert1->coefficients.ptr[iSrc]*factor;
		
								// 2 * because we otherwise took the average over ALL tris at this vertex
								// and not just the domain ones.

								radial = u[iprev];

								effect_v_tri_on_v_vertex.xx = coefficient*
									(1.0-radial.x*radial.x/(radial.dot(radial)));
								effect_v_tri_on_v_vertex.xy = coefficient*
									(-radial.x*radial.y/(radial.dot(radial)));
								effect_v_tri_on_v_vertex.yx = effect_v_tri_on_v_vertex.xy;
								effect_v_tri_on_v_vertex.yy = coefficient*
									(1.0-radial.y*radial.y/(radial.dot(radial)));

								pTri->visc_coeffs[index].xx += 
									// effect due to x at vertex: 
									momflux_px[1].x*
									effect_v_tri_on_v_vertex.xx +
									momflux_px[1].y*
									effect_v_tri_on_v_vertex.yx;
								// vx at tri -> vy at vertex -> affects vx
															 
								// effect on x due to y at tri:
								pTri->visc_coeffs[index].xy += momflux_px[1].y* 
									effect_v_tri_on_v_vertex.yy +
									 momflux_px[1].x* 
									 effect_v_tri_on_v_vertex.xy;
								// vy at tri -> vx at vertex -> affects vx
								
								pTri->visc_coeffs[index].xz += momflux_px[1].z*coefficient;
								// vz at tri -> only vz at vertex
								
								pTri->visc_coeffs[index].yx +=
									momflux_py[1].x*
									effect_v_tri_on_v_vertex.xx +
									momflux_py[1].y*
									effect_v_tri_on_v_vertex.yx;

								pTri->visc_coeffs[index].yy +=
									momflux_py[1].x*
									effect_v_tri_on_v_vertex.xy +
									momflux_py[1].y*
									effect_v_tri_on_v_vertex.yy;

								pTri->visc_coeffs[index].yz += momflux_py[1].z*coefficient;

								pTri->visc_coeffs[index].zx +=
									momflux_pz[1].x*
									effect_v_tri_on_v_vertex.xx +
									momflux_pz[1].y*
									effect_v_tri_on_v_vertex.yx;
								
								pTri->visc_coeffs[index].zy +=
									momflux_pz[1].x*
									effect_v_tri_on_v_vertex.xy + 
									momflux_pz[1].y*
									effect_v_tri_on_v_vertex.yy;
								
								pTri->visc_coeffs[index].zz += momflux_pz[i].z*coefficient;
							};
								
							
							// Note: they're coefficients on contiguous velocities!
							// Then can work out what to do with these.
							
							// Coefficients are for every tri at an ins vertex, but we restricted
							// to using only those from in-domain triangles.
							
						};
					}
					
					
					for (iSrc = 0; iSrc < pVert2->coefficients.len; iSrc++)
					{
						pSrc = (Triangle *)(pVert2->triangles.ptr[iSrc]);
						index = pTri->indexlist.FindIndex(pSrc-T);
						
						if (pVert2->flags != 3) {
							// otherwise happy to use all tris at the vertex and not manipulate v_vertex
							
							
							coefficient = pVert2->coefficients.ptr[iSrc]*factor;
							
							// We are assuming here that when the contribution is made,
							// we already will have rotated the contributing velocity to be contiguous.
							// coefficients.ptr[iSrc] tells us the ppn at which the (contig) v in tri
							// is averaged to give (contig) v at vertex.
							
							pTri->visc_coeffs[index].xx += momflux_px[3].x*coefficient;
							pTri->visc_coeffs[index].xy += momflux_px[3].y*coefficient;
							pTri->visc_coeffs[index].xz += momflux_px[3].z*coefficient;
							
							pTri->visc_coeffs[index].yx += momflux_py[3].x*coefficient;
							pTri->visc_coeffs[index].yy += momflux_py[3].y*coefficient;
							pTri->visc_coeffs[index].yz += momflux_py[3].z*coefficient;
							
							pTri->visc_coeffs[index].zx += momflux_pz[3].x*coefficient;
							pTri->visc_coeffs[index].zy += momflux_pz[3].y*coefficient;
							pTri->visc_coeffs[index].zz += momflux_pz[3].z*coefficient;
							
							// And life would be nicer if only we were vertex-based.

						} else {
							// In the case that this is an edge vertex: set vr from here = 0
							// factor = 1.0/(n_e*m_e);
							// momflux_pxyz is the amt of additional v_xyz, coefficient on v at corner 1

							// but sometimes the corner has v_r = 0 required

							// which means instead of being a1 v1 + a2 v2
							// it is (a1 v1 + a2 v2) - rhat (a1 v1+a2 v2).dot(rhat)

							// a1 (v1-rhat(v1.rhat)) = a1(v1 - r(v1.r)/r.r)

							if (pSrc->u8domain_flag == PLASMA_DOMAIN) {

								coefficient = 2.0 * pVert2->coefficients.ptr[iSrc]*factor;
		
								// 2 * because we otherwise took the average over ALL tris at this vertex
								// and not just the domain ones.

								radial = u[inext];

								effect_v_tri_on_v_vertex.xx = coefficient*
									(1.0-radial.x*radial.x/(radial.dot(radial)));
								effect_v_tri_on_v_vertex.xy = coefficient*
									(-radial.x*radial.y/(radial.dot(radial)));
								effect_v_tri_on_v_vertex.yx = effect_v_tri_on_v_vertex.xy;
								effect_v_tri_on_v_vertex.yy = coefficient*
									(1.0-radial.y*radial.y/(radial.dot(radial)));

								pTri->visc_coeffs[index].xx += 
									// effect due to x at vertex: 
									momflux_px[3].x*
									effect_v_tri_on_v_vertex.xx +
									momflux_px[3].y*
									effect_v_tri_on_v_vertex.yx;
								// vx at tri -> vy at vertex -> affects vx
															 
								// effect on x due to y at tri:
								pTri->visc_coeffs[index].xy += momflux_px[3].y* 
									effect_v_tri_on_v_vertex.yy +
									 momflux_px[3].x* 
									 effect_v_tri_on_v_vertex.xy;
								// vy at tri -> vx at vertex -> affects vx
								
								pTri->visc_coeffs[index].xz += momflux_px[3].z*coefficient;
								// vz at tri -> only vz at vertex
								
								pTri->visc_coeffs[index].yx +=
									momflux_py[3].x*
									effect_v_tri_on_v_vertex.xx +
									momflux_py[3].y*
									effect_v_tri_on_v_vertex.yx;

								pTri->visc_coeffs[index].yy +=
									momflux_py[3].x*
									effect_v_tri_on_v_vertex.xy +
									momflux_py[3].y*
									effect_v_tri_on_v_vertex.yy;

								pTri->visc_coeffs[index].yz -= momflux_py[3].z*coefficient;

								pTri->visc_coeffs[index].zx -=
									momflux_pz[3].x*
									effect_v_tri_on_v_vertex.xx +
									momflux_pz[3].y*
									effect_v_tri_on_v_vertex.yx;
								
								pTri->visc_coeffs[index].zy -=
									momflux_pz[3].x*
									effect_v_tri_on_v_vertex.xy + 
									momflux_pz[3].y*
									effect_v_tri_on_v_vertex.yy;
								
								pTri->visc_coeffs[index].zz -= momflux_pz[i].z*coefficient;
							};
								
							
							// Note: they're coefficients on contiguous velocities!
							// Then can work out what to do with these.
							
							// Coefficients are for every tri at an ins vertex, but we restricted
							// to using only those from in-domain triangles.
							
						};
					};

				}; // whether this edge used -- note pTri and pNeigh have to be in plas domain ???
				
			}; // next iNeigh
		
		}; // do not do if not in plasma domain

		++pTri;
	};

	// This routine just about done.
}


// TriMesh::Advance
int TriMesh::Advance(TriMesh * pDestMesh)
{
	Tensor3 sigma_J;
	Vector3 J0, J;
	Vector2 centroid;
	long numTris1D;

	char buffer[256];
	Vertex * pVertex;
	real vx,vy,vr,vtheta, factor_i, factor_e;
	int i;
	real TotalArea, TotalCharge, TotalAbsCharge;
	FILE * fp ;
	Triangle * pTri, * pTriSrc;
	long iTri,iVertex;
	Vector2 centre, rhat;
	real Br, Btheta;
	Vector3 Eplus;
	real temp1, temp2, temp3;
	real ionmass, neutmass, elecmass, Iz_prescribed;
	int species;
	SlimVertex * pInner, * pInnerSrc;
	Vertex * pVertSrc;
	real Iz_existing;
	real store_h;
	
	GlobalFrictionalHtg_en = 0.0;	
	GlobalFrictionalHtg_ei = 0.0;	
	GlobalFrictionalHtg_in = 0.0;
	GlobalResistiveHtg = 0.0;	
	GlobalEnergyInput = 0.0;	
	GlobalIonisationHtg = 0.0;		
	bSwitchOffChPatReport = 0;

	store_h = h;

	real htest = GetMaxTimestep(); 
	if (htest < h) {
		h = htest;
	};


	// 1. Work to create destination mesh.

	this->RecalculateVertexVariables(); // populate vertex v with n-weighted data for each species
	
	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		if ((pTri->ion.heat < 0.0) || (pTri->neut.heat < 0.0) || (pTri->elec.heat < 0.0))
		{
			iTri = iTri;
		};

		// get first installment to guess at energy input:
		GlobalEnergyInput += 0.5*h*q*pTri->E.dot(pTri->ion.mom-pTri->elec.mom);

		++pTri;
	};

	CollectFunctionals();

	report_time(0);
	printf("Setting up displacement linear relations for pressure k+1: ");
	
	this->RefreshVertexNeighboursOfVerticesOrdered();
	// does it also sort triangles?
	this->MakePressureAccelData(SEPARATE_ELECTRON_AND_ION); // pressure for time t_k
	// creates
	// Vertex::Pressure_numerator_x , IonP_Viscous_numerator_x (ion pressure)
	// eP_Viscous_denominator_x (e pressure)
	// "divide by nM Area_voronoi to give additional velocity"
	// "for cell momentum multiply by h/m_species"
	// pTri2->scratch[0] += cellfrac*pVert->Pressure_numerator_x/m_neutral;
	// ^^ cell momentum addition rate
	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		// This will be where this routine still gets used.
 
		if (pTri->flags < 100) // do not do for invalid triangles ! (Are there any?)
			AccelerateIons_or_ComputeOhmsLawForRelativeVelocity(pTri, FIRSTPASS); 

		// Calculates heavy displacement0 and matrix for effect of own pressure.
		++pTri;
	};
		
	// Average matrices and position0's on to vertices:
	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		ionmass = 0.0;
		neutmass = 0.0;
		elecmass = 0.0;
		
		ZeroMemory(&(pVertex->Pressure_a_effect_dis[0]),sizeof(Tensor2));
		ZeroMemory(&(pVertex->Pressure_a_effect_dis[1]),sizeof(Tensor2));
		//ZeroMemory(&(pVertex->Pressure_a_effect_dis[2]),sizeof(Tensor2));
		
		for (species = 0; species < 3; species++)
		{
			pVertex->AdvectedPosition0[species].x = 0.0;//pVertex->x; add at end
			pVertex->AdvectedPosition0[species].y = 0.0;//pVertex->y;
		};
		
		for (i = 0; i < pVertex->triangles.len; i++)
		{
			pTri = (Triangle *)(pVertex->triangles.ptr[i]);

			pVertex->Pressure_a_effect_dis[SPECIES_ION] += pTri->ion.mass*(pTri->Pressure_a_effect_ion);
			pVertex->AdvectedPosition0[SPECIES_ION] += pTri->ion.mass*pTri->Displacement0[SPECIES_ION];
			ionmass += pTri->ion.mass;
			
			pVertex->Pressure_a_effect_dis[SPECIES_NEUTRAL] += pTri->neut.mass*(pTri->Pressure_a_effect_neut);
			pVertex->AdvectedPosition0[SPECIES_NEUTRAL] += pTri->neut.mass*pTri->Displacement0[SPECIES_NEUTRAL];
			neutmass += pTri->neut.mass;
			
			//pVertex->Pressure_a_effect_dis[SPECIES_ELECTRON] += pTri->elec.mass*(pTri->Pressure_a_effect_elec);
			//pVertex->AdvectedPosition0[SPECIES_ELECTRON] += pTri->elec.mass*pTri->Displacement0[SPECIES_ELECTRON];
			//elecmass += pTri->elec.mass;

			// Old news:

			// Stop press: we don't have the two species depend on each others' final pressures. That's because the advected cells aren't in the same place.
			// This could lead to one (high n) thing bouncing off something else somewhere else and affecting the other thing here wrongly. ?

			// For acceleration maybe we should come back, interpolate to get each others' final pressures at destinations ... NOPE
			// That is no good - if we soak momentum then we want to be soaking the same amount to the same place. 
			// Just identify a different place to be soaking it if it comes to that.
			
			// Or put (nT)_vertex on this cell -- that's the only way we're going to get a consistent location to supply pressure after all.
			// confusion!!
			// We do want x_k+1 : at the end of the day that is how we move and we chose that for clear reasons : part of shock-capturing strategy. 
			
			// Try using the dest mesh to do final-time pressure and soak?
			// be creative?
			 
			// Soaking in initial position, including with the advected-position pressures, is our best option I think.
			// It's a second-order inaccuracy. 
			// It seems more accurate than just taking the initial pressure for each species, accelerating fully with that and soaking it. But that would be valid also.
			
			// Maybe we just need to give up on all the fancy business and say that we need to do some advection and re-mapping before soaking again.
			// ie this is fairly pointless -- thermal pressure shouldn't normally change terrifically during a timestep anyway ?
			// (but e pressure force is big)
			// Don't want timesteps on frictional / gyro timescale. Do want timesteps on advective timescale. But occasionally we do hit a wall during a step.

			// Qualitatively the "pressure at start and end, soak wherever" approach is correct: it conserves energy on soak, and creates
			// about the right amount of species momentum. 
			// That will be for acceleration; for displacement it's too tricky to take account of it and we might as well take it (the other species' pressure) to be zero.
			
			// Note: when we did all friction in an instant, that is applied at one time and one place -- as opposed to, just in one place: much harder to get things right
			// given that we do not follow the frictional / gyro timescale.
			
			// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			// Point of including k+1 pressure is very much to avoid silly phenomena to do with over-compression. Otherwise not that helpful.
			// We do want to solve for some mutually consistent vertex positions however. 
			// That is sensible, as is controlling the k+1 effect on displacement intelligently.
			// stick with doing species acceleration per the start and end points for the species;
			// stick with soaking in this original cell even though we understand that is a numerical error. If we want to reduce that, reduce the timestep.
			// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			
		};
		pVertex->Pressure_a_effect_dis[SPECIES_ION] *= 1.0/ionmass;
		pVertex->Pressure_a_effect_dis[SPECIES_NEUTRAL] *= 1.0/neutmass;
		pVertex->AdvectedPosition0[SPECIES_ION] *= 1.0/ionmass;
		pVertex->AdvectedPosition0[SPECIES_NEUTRAL] *= 1.0/neutmass;
		//pVertex->AdvectedPosition0[SPECIES_ELECTRON] *= 1.0/elecmass;
		
		pVertex->AdvectedPosition0[SPECIES_ION].x += pVertex->x;
		pVertex->AdvectedPosition0[SPECIES_ION].y += pVertex->y;
		pVertex->AdvectedPosition0[SPECIES_NEUTRAL].x += pVertex->x;
		pVertex->AdvectedPosition0[SPECIES_NEUTRAL].y += pVertex->y;
		//pVertex->AdvectedPosition0[SPECIES_ELECTRON] = pVertex->Ad;
		//pVertex->AdvectedPosition0[SPECIES_ELECTRON].y += pVertex->y;
		
		++pVertex;
	};

	printf("%s \n",report_time(1));
	report_time(0);

	// Now expect ion to fail for the same reason that it fails when E is not bwd but takes
	// displacement = integral of v over time. v has to reverse to get the required overall
	// displacement - in pressure terms we have to get too close to the object and then next
	// time we will bounce off but we may end up coming too close to another vertex next time
	// .. everything gets wobbly until you change to full backward: displacement = hv
	// ================================================

	//  2. Solve for ion and neutral new vertex positions. 
	
	printf("Advecting ion and neutral vertices: ");
	
	//	Avoiding overcompression means we should never need a bounce when we then do cell contents advection;
	//	whereas having a bounce instead, which would be easier, overcompression always stands a good chance of being a problem,
	//  unless we do what we used to do with anti-overcompression in each cell and multiple corner paths - that then
	//  would mean adiabatic was illegitimate and that we ignored external forces on the cell which was a bit inconsistent.
	
	// For now we did an inefficient way: move as it looks from where we are, the whole system; global timestep changes if we are going too far somewhere.

	SolveForAdvectedPositions(SPECIES_ION);    // populate advected position data for each vertex using AdvectedPosition0 and Pressure_a_effect_on_dis.
	SolveForAdvectedPositions(SPECIES_NEUTRAL);// populate advected position data for each vertex using AdvectedPosition0 and Pressure_a_effect_on_dis.
	

	// The following will no longer be used for anything:
	
	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		// Move electrons as ions for this bit.
		pVertex->AdvectedPosition[SPECIES_ELECTRON] = pVertex->AdvectedPosition[SPECIES_ION];
		// Populate AdvectedPosition0 with data saying "by default, send me back where I came from".
		pVertex->AdvectedPosition0[SPECIES_ELECTRON].x = pVertex->x - pVertex->AdvectedPosition[SPECIES_ELECTRON].x;
		pVertex->AdvectedPosition0[SPECIES_ELECTRON].y = pVertex->y - pVertex->AdvectedPosition[SPECIES_ELECTRON].y;
		
		// I don't really like this way.
		// But we are, at the moment, cancelling a displacement against just hv, so that does muddy the waters.

		++pVertex;
	};

	printf("%s \n",report_time(1));
	report_time(0);
	
	// 3. Create new mesh as bulk average: will need to have n on each vertex of course...
	
	// Incorporate here: interpolate for estimates of A, e rel displacement so far, probably phi for seed
	printf("Creating new mesh: ");
	pDestMesh->AverageVertexPositionsAndInterpolate(this, true);
	// Taking average of new positions is OK as long as we are careful of periodic boundary crossing. 	// DO NOT WRAP 	// WRAP AFTERWARDS
	FinishAdvectingMesh(pDestMesh); 
	
	pInner = pDestMesh->InnerX;
	pInnerSrc = InnerX;
	for (iVertex= 0; iVertex < pDestMesh->numInnerVertices; iVertex++)
	{
		pInner->A = pInnerSrc->A;
		++pInner;
		++pInnerSrc;
	} // not sure it matters but nvm
	
	pTri = pDestMesh->T;
	for (iTri = 0; iTri < pDestMesh->numTriangles; iTri++)
	{
		pTri->area = pTri->GetArea(); // pretty basic and useful, before we go on!
		++pTri;
	};
	pDestMesh->RefreshVertexNeighboursOfVerticesOrdered(); // could be important for B below. Should be in FinishAdvectingMesh. (look there)
	
	printf("%s \n",report_time(1));
	report_time(0);
		
	printf("Placing advected ion and neutral data: ");
	
	// Advection of species into new mesh:
	pDestMesh->ZeroCellData();
	// Be simple and rough about energy conservation on this first attempt.
	// Actually first approximation is to do nothing, since nv gradient usually cancels out with the different nv^2 entering.
	this->PlaceAdvected_Triplanar_Conservative_IntoNewMesh(SPECIES_NEUTRAL,pDestMesh,ALL_VARS,1); 
	this->PlaceAdvected_Triplanar_Conservative_IntoNewMesh(SPECIES_ION,pDestMesh,ALL_VARS,1); 
	this->PlaceAdvected_Triplanar_Conservative_IntoNewMesh(SPECIES_ELECTRON,pDestMesh,ALL_VARS,1); 	// placing electrons as ions
		
	// The following must report what disparities were found:
	pDestMesh->RestoreSpeciesTotals(this); // scale to stamp out tiny changes in species mass etc
	pDestMesh->RecalculateVertexVariables(); 
	pDestMesh->GetBFromA(); // need B to use before we can work out any sigma.
	printf("%s \n",report_time(1));

	pTri = pDestMesh->T;
	for (iTri = 0; iTri < pDestMesh->numTriangles; iTri++)
	{
		if ((pTri->ion.heat < 0.0) || (pTri->neut.heat < 0.0) || (pTri->elec.heat < 0.0))
		{
			iTri = iTri;
		};
		++pTri;
	};

	// Using old B_k of course:
	pDestMesh->ConservativeSmoothings(h*0.25); // do some here ahead of other things - smooth the compressive htg

	pTri = pDestMesh->T;
	for (iTri = 0; iTri < pDestMesh->numTriangles; iTri++)
	{
		if ((pTri->ion.heat < 0.0) || (pTri->neut.heat < 0.0) || (pTri->elec.heat < 0.0))
		{
			iTri = iTri;
		};
		++pTri;
	};

	printf("Do ionisation:");
	// Easiest way for now - get ionisation out of the way before we start messing about with currents.
	// If based on false compressive heating from being moved with ions -- too bad.


	pDestMesh->IoniseAndRecombine(); // check it.
	
	pTri = pDestMesh->T;
	for (iTri = 0; iTri < pDestMesh->numTriangles; iTri++)
	{
		if ((pTri->ion.heat < 0.0) || (pTri->neut.heat < 0.0) || (pTri->elec.heat < 0.0))
		{
			iTri = iTri;
		};
		++pTri;
	};

	//pDestMesh->RecalculateVertexVariables();  // is it in ConservativeSmoothings?

	pDestMesh->ConservativeSmoothings(h*0.25); // do some here ahead of other things - smooth recombination htg
	// Grad Te will be used for momentum transfer etc.
	
	pDestMesh->RecalculateVertexVariables(); 
	
	pTri = pDestMesh->T;
	for (iTri = 0; iTri < pDestMesh->numTriangles; iTri++)
	{
		if ((pTri->ion.heat < 0.0) || (pTri->neut.heat < 0.0) || (pTri->elec.heat < 0.0))
		{
			iTri = iTri;
		};
		++pTri;
	};

	// DEBUG INFO !!!!

	real * radiiArray;
	long * indexArray;
	FILE * fp_1D;
	debugdata * debugdata1D;
	int got1, got0;

	indexArray = new long[10000];
	radiiArray = new real[10000];

	long iCaret = 0;

	pTri = pDestMesh->T;
	for (iTri = 0; iTri < pDestMesh->numTriangles; iTri++)
	{
		// Decide if this is a triangle that contains the cutaway line or not:
		
		pTri->indicator = -1;
		
		if (pTri->periodic == 0) {
		
			got1 = 0; got0 = 0;
			if (pTri->cornerptr[0]->x/pTri->cornerptr[0]->y > -PIOVER32)
			{	got1 = 1; } else { got0 = 1; }
			if (pTri->cornerptr[1]->x/pTri->cornerptr[1]->y > -PIOVER32)
			{	got1 = 1; } else { got0 = 1; }
			if (pTri->cornerptr[2]->x/pTri->cornerptr[2]->y > -PIOVER32)
			{	got1 = 1; } else { got0 = 1; }

			if ((got1 == 1) && (got0 == 1))
			{
				pTri->indicator = 1;
				// Add to a list & list of radii:
				
				pTri->GenerateContiguousCentroid(&centroid, pTri);

				indexArray[iCaret] = iTri;
				radiiArray[iCaret] = sqrt(centroid.x*centroid.x+centroid.y*centroid.y);
				iCaret++;
			}
		};
		
		++pTri;
	};
	
	// Create sorted list of triangle indices according to radius

	QuickSort (indexArray, radiiArray,
		0,iCaret-1); 

	numTris1D = iCaret;

	// Now tell each triangle which position it occupies in this list, as indicator.

	for (iCaret = 0; iCaret < numTris1D; iCaret++)
	{
		pTri = pDestMesh->T + indexArray[iCaret];
		pTri->indicator = iCaret;
	};
	// May have to do 

	debugdata1D = new debugdata[numTris1D];

	// ====================================


	// Now ready to do electron & E :
	printf("Compute Ohm's Law tensors: ");
	
	report_time(0);
	
	// Get pressure before doing subcycle:
	
	Global_Ee_Exchrate_ei = 0.0;
	Global_Ee_Exchrate_en = 0.0;
	Global_Ee_Te = 0.0;
	Global_Ee = 0.0;

	pDestMesh->MakePressureAccelData(SEPARATE_ELECTRON_AND_ION); 

	pDestMesh->CalculateIndirectNeighbourLists();

	pDestMesh->Calculate_Electron_Viscous_Momflux_Coefficients();
	// populate coefficients on cells to give Div Pi



	pTri = pDestMesh->T;
	for (iTri = 0; iTri < pDestMesh->numTriangles; iTri++)
	{
		pDestMesh->AccelerateIons_or_ComputeOhmsLawForRelativeVelocity(pTri, COMPUTE_SIGMA);   
		// To create pTri->sigma = E effect on relative velocity
		// ve-vi will be = pTri->vrel_e_0 + pTri->sigma_erel*E
		// vi will be = pTri->vion_0 + pTri->sigma_i*E

		// Reasoning:
		// . v_ion_k+1 WILL change with E so current might as well anticipate this correctly.
		// . Fix ion displacement with what already done, or we would need still another matrix.


		// May need similar trick to before as regards low timestep for high nu?
		// Only needed a few steps at that length.


		// In a more advanced version we can put here the choice to at least
		// sometimes, do a backward step, that is compatible with same Ohm's Law.


		++pTri;
	};
	



	//// Therefore better stick with having the edge contribution via vertex stored AdvectedPosition0 = default displacement to cancel ion.
	//
	//// Wary of creating "effect on relative momentum"...
	//// easier to end up with negative density?

	//// prefer to create [integral velocity] effect on each _edge_ ... where we also estimate n_edge somehow
	//// then want to use that to get rho(phi)
	//// Alternative view -- to cancel ion we can take difference within the cell, ?? It's a bit garbled! displacement integral vs hv.

	//


	// No reason not to put this inside SolveForAzEz:
	// ------------------------------------------------
	pTri = pDestMesh->T;
	Global_dIz_by_dEz = 0.0;
	GlobalDefaultIz = 0.0;
	for (iTri = 0; iTri < pDestMesh->numTriangles; iTri++)
	{
		// Note that we do need for each species, the effect on cell mom: because we then do species flows.
		// Could just work out again and do flows after, if we wanted, of course.
		
		Global_dIz_by_dEz += q*((pTri->ion.mass-pTri->elec.mass)*pTri->sigma_i.zz
								-pTri->elec.mass*pTri->sigma_erel.zz);
		GlobalDefaultIz += q*((pTri->ion.mass-pTri->elec.mass)*pTri->vion_0.z
								-pTri->elec.mass*pTri->vrel_e_0.z);

			//pTri->p0[SPECIES_ION].z-pTri->p0[SPECIES_ELECTRON].z;
			//					q*pTri->denominator;
		// Used to need, the part of ion velocity that already was taken into account
		// but not in v0 which used to be additional velocity.
		
		++pTri;
	};
	evaltime += h;
	GlobalIzPrescribed = -PEAKCURRENT_STATCOULOMB * sin ((evaltime + ZCURRENTBASETIME) * 0.5*PIOVERPEAKTIME );
		
	pTri = pDestMesh->T;
	for (iTri = 0; iTri < pDestMesh->numTriangles; iTri++)
	{
		if ((pTri->ion.heat < 0.0) || (pTri->neut.heat < 0.0) || (pTri->elec.heat < 0.0))
		{
			iTri = iTri;
		};
		++pTri;
	};

	// Inputs: 
	// _________
	// GlobalIzPrescribed, Global_dIz_by_dEz, GlobalDefaultIz [given E=0], 
	// pTri->denominator [default momdiff]
	// pTri->sigma_e_h,sigma_ion_h [for now, rate of change of cell integral nv wrt E]
	

	pDestMesh->Solve_A_phi_and_J(this);
	
	// Idea is, this should populate phi(on verts); A, E, vrel (on cells).


	// Does it make sense perhaps to solve for elec.mom, also inferring ion mom,
	// given that we apply _electron_ viscosity
	// -- although the assumption is that d/dt(vrel) = 0.

	// Let's work this out in LyX.
	// We can calculate ion v equation based on v_e rather than v_e-v_i, if necessary.

	// Note that just as 
	
	
	printf("Solve for Az,Ez round 1: ");
	pDestMesh->SolveForAzAndEz(this,0);//, GlobalEz);  // work on linear extrapolation for Ez external magnitude.
	
	
	// Now set GlobalEz within routine, so that we can initially have sum of errors = 0.
	
	// Outputs:  pTri->coeff[0,1,2] to create Ez from A,Aold  ;  GlobalEz  ;  pVertex->A.z
	
	// Now adjust default displacement per matrix we worked out:
	Vertex * pVertSrc0, * pVertSrc1, * pVertSrc2;
	real Iz = 0.0;
	pTri = pDestMesh->T;
	for (iTri = 0; iTri < pDestMesh->numTriangles; iTri++)
	{           
		// Now we are going to assume that we have electron displacement = h v_k+1, sadly.
		// Additional ion displacement might as well be the same, h[h/2] q/M E basically. 
		
		pVertSrc0 = X + (pTri->cornerptr[0]-pDestMesh->X);
		pVertSrc1 = X + (pTri->cornerptr[1]-pDestMesh->X);
		pVertSrc2 = X + (pTri->cornerptr[2]-pDestMesh->X);
#ifdef VORONOI_A
		pTri->E.z = GlobalEz - ( pTri->coeff[0]*(pTri->cornerptr[0]->A.z - pVertSrc0->A.z)
							 + pTri->coeff[1]*(pTri->cornerptr[1]->A.z - pVertSrc1->A.z)
							 + pTri->coeff[2]*(pTri->cornerptr[2]->A.z - pVertSrc2->A.z)
							 )/(c*h*pTri->area);
		// coeff/area = fraction of tri?
#else
		pTri->E.z = GlobalEz - THIRD*(
								(pTri->cornerptr[0]->A.z - pVertSrc0->A.z)
							+	(pTri->cornerptr[1]->A.z - pVertSrc1->A.z)
							+	(pTri->cornerptr[2]->A.z - pVertSrc2->A.z)	)/(c*h);
#endif	
		// Adapt putative p0 , k+1 cell momentum going into Gauss solver
		
		// old:
		
		//pTri->p0[SPECIES_ION].x = pTri->sigma_ion_h.xz*pTri->E.z;
		//pTri->p0[SPECIES_ION].y = pTri->sigma_ion_h.yz*pTri->E.z;		
		//pTri->p0[SPECIES_ELECTRON].x += pTri->sigma_e_h.xz*pTri->E.z;
		//pTri->p0[SPECIES_ELECTRON].y += pTri->sigma_e_h.yz*pTri->E.z;
		
		// now:
		
		// Do not wish to ruin our vion_0.x like this !
		pTri->vion_0.x += pTri->sigma_i.xz*pTri->E.z;
		pTri->vion_0.y += pTri->sigma_i.yz*pTri->E.z;
		pTri->vion_0.z += pTri->sigma_i.zz*pTri->E.z;
		
		pTri->vrel_e_0.x += pTri->sigma_erel.xz*pTri->E.z;
		pTri->vrel_e_0.y += pTri->sigma_erel.yz*pTri->E.z;
		pTri->vrel_e_0.z += pTri->sigma_erel.zz*pTri->E.z;
		
		// This is a stupid way. Need to keep v relationship intact -- too easy to make a mistake this way.
		
		++pTri;
	};
	
	pTri = pDestMesh->T;
	for (iTri = 0; iTri < pDestMesh->numTriangles; iTri++)
	{
		if ((pTri->ion.heat < 0.0) || (pTri->neut.heat < 0.0) || (pTri->elec.heat < 0.0))
		{
			iTri = iTri;
		};
		++pTri;
	};

	printf("Putative Iz: %1.10E \n",Iz);	
	printf("%s \n",report_time(1));
	report_time(0); // comes out fine.
		
	//boolVerbose = false;
	printf("Phi Gauss solver: ");
	pDestMesh->SolveGaussForPhi_EulerianCurrent(h,this);  
	// That also includes the flows and compressive heating.

	pTri = pDestMesh->T;
	for (iTri = 0; iTri < pDestMesh->numTriangles; iTri++)
	{           
		// here, restore v_0.

		pTri->vion_0.x -= pTri->sigma_i.xz*pTri->E.z;
		pTri->vion_0.y -= pTri->sigma_i.yz*pTri->E.z;
		pTri->vion_0.z -= pTri->sigma_i.zz*pTri->E.z;
		
		pTri->vrel_e_0.x -= pTri->sigma_erel.xz*pTri->E.z;
		pTri->vrel_e_0.y -= pTri->sigma_erel.yz*pTri->E.z;
		pTri->vrel_e_0.z -= pTri->sigma_erel.zz*pTri->E.z;
		
		// This is still a stupid way. 
		
		++pTri;
	};
		
	boolVerbose = true;

	printf("%s \n",report_time(1));	

	pTri = pDestMesh->T;
	for (iTri = 0; iTri < pDestMesh->numTriangles; iTri++)
	{
		if ((pTri->ion.heat < 0.0) || (pTri->neut.heat < 0.0) || (pTri->elec.heat < 0.0))
		{
			iTri = iTri;
		};
		++pTri;
	};

	// Now we would like to re-set Az, Ez again. But it's not so easy?
	// For seed of A we know we use the existing values; but still old system for comparison.
	// sigma_h.zz is still to be same as before.
	// IzPrescribed is same but initial total Jz needs to be reassessed in view of changed Exy.
	// Actually are coefficients for A all same? Have they been interfered with?
	// Unfortunately yes - overwrote with Exy coefficients.

	// Better do 2nd go after acceleration since it involves ionisation ...

	// =================================================================
	// Ionisation no longer done here but during acceleration step. 
	// Have yet to take account of current reduction due to recombination, in sigma_h.
	
	printf("Acceleration & heating: ");
	report_time(0);


	// Then we can populate some debug array elements with data rather than having to find room in Triangle.

	// Then output some info from both array and triangle itself, in radius order.


	Vector3 Etemp, vi, vrel;
	pDestMesh->MakePressureAccelData(SEPARATE_ELECTRON_AND_ION); // pressure for time t_k+1
	pTri = pDestMesh->T;
	Iz = 0.0;
	real Iz2 = 0.0;
	for (iTri = 0; iTri < pDestMesh->numTriangles; iTri++)
	{
		// ############################################################
		// Bear in mind we wrecked vion_0, v_erel_0 ... already accounted for Ez

		// Here is the place to set electron momentum
		// Ready to use in the ion and neutral acceleration,
		// which we still do again anyway though I'm not sure it should
		// ever come to a different result(?)
		// ############################################################
	
		Etemp = pTri->E;
		
		vi = pTri->vion_0 + pTri->sigma_i*Etemp;
		vrel = pTri->vrel_e_0 + pTri->sigma_erel*Etemp;

		pTri->elec.mom = pTri->elec.mass*(vi+vrel);
		// It, and pTri->E, must also be in date before we call on first pass.
		
		// Now what is actually used here? :
		GlobalDebugRecordIndicator = 0;
		if (pTri->indicator != -1) GlobalDebugRecordIndicator = 1; // whether to collect debug data into Globaldebugdata

		pDestMesh->AccelerateIons_or_ComputeOhmsLawForRelativeVelocity(pTri, ACCELERATE);  
		
		if (pTri->indicator != -1) {
			// Fill in debug data from accels
			
			debugdata1D[pTri->indicator] = Globaldebugdata;
			
		};
		// Now reset elec.mom using accelerated v_ion (same as before) and same Ohm's Law:

		pTri->elec.mom = pTri->elec.mass*(pTri->ion.mom/pTri->ion.mass + vrel);

		Iz2 += q*(pTri->ion.mom.z-pTri->elec.mom.z);

		++pTri;
	};
	printf("Iz from mom %1.10E \n",Iz2);

	// ================================
	
	// Output from debug data 1D array:

	Vector3 unit_radial, unit_theta;

	sprintf(buffer,"array1D%d.txt",GlobalStepsCounter);
	fp_1D = fopen(buffer,"w");

	fprintf(fp_1D,"evaltime: %1.10E \n\n",evaltime);

	debugdata * pddata;

	for (iCaret = 0; iCaret < numTris1D; iCaret++)
	{
		pTri = pDestMesh->T + indexArray[iCaret];
		pddata = &(debugdata1D[iCaret]);

		fprintf(fp_1D,"iTri centroid polar area B polar | "
			" sigma_vrel_e _ xyz | "
			" E polar vrel_e_0 polar | "
			" n_e v_e polar T_e | "
			" n_i v_i polar T_i | "
			" n_n v_n polar T_n | "
			" sigma_i _ xyz | v_i_0 polar | "
			" Upsilon_e _ xyz | "
			" nu_ieBar nu_in_MT lnLambda_ei GradTe polar | "
			" a_pressure polar a_E polar a_omega polar "
			" a_from_neut polar a_frict_ei polar a_thermal polar a_total polar | \n"

			);
		
		// spit out info from pTri
		pTri->GenerateContiguousCentroid(&centroid,pTri);
		unit_radial.x = centroid.x/radiiArray[iCaret];
		unit_radial.y = centroid.y/radiiArray[iCaret];
		unit_radial.z = 0.0;
		unit_theta.x = -unit_radial.y;
		unit_theta.y = unit_radial.x;
		unit_theta.z = 0.0;
		
		fprintf(fp_1D,"%d %1.10E %1.10E %1.10E ",
			indexArray[iCaret],centroid.x,radiiArray[iCaret],pTri->area);
				
		fprintf(fp_1D," %1.9E %1.9E | ", pTri->B.x,pTri->B.dot(unit_radial));

		fprintf(fp_1D," %1.9E %1.9E %1.9E ", 
			pTri->sigma_erel.xx,pTri->sigma_erel.xy,pTri->sigma_erel.xz);
		
		fprintf(fp_1D," | %1.10E %1.10E ",pTri->E.x,
			pTri->E.dot(unit_radial));

		fprintf(fp_1D,"  %1.10E %1.10E | ",pTri->vrel_e_0.x,
			pTri->vrel_e_0.dot(unit_radial));
		
		fprintf(fp_1D," %1.10E %1.10E %1.10E %1.10E | ",
			pTri->elec.mass/pTri->area,
			pTri->elec.mom.x/pTri->elec.mass,
			pTri->elec.mom.dot(unit_radial)/pTri->elec.mass,
			pTri->elec.heat/pTri->elec.mass);
		
		fprintf(fp_1D," %1.10E %1.10E %1.10E %1.10E | ",
			pTri->ion.mass/pTri->area,
			pTri->ion.mom.x/pTri->ion.mass,
			pTri->ion.mom.dot(unit_radial)/pTri->ion.mass,
			pTri->ion.heat/pTri->ion.mass);
		
		fprintf(fp_1D," %1.10E %1.10E %1.10E %1.10E | ",
			pTri->neut.mass,
			pTri->neut.mom.x/pTri->neut.mass,
			pTri->neut.mom.dot(unit_radial)/pTri->neut.mass,
			pTri->neut.heat/pTri->neut.mass);


		fprintf(fp_1D," %1.10E %1.10E %1.10E | ",
			pTri->sigma_i.xx,pTri->sigma_i.xy,pTri->sigma_i.xz);
		
		fprintf(fp_1D," %1.10E %1.10E | ",
			pTri->vion_0.x, pTri->vion_0.dot(unit_radial));

		// now spit out info from debugdata1D[iCaret]
		

		fprintf(fp_1D, " %1.10E %1.10E %1.10E | ",
			pddata->Upsilon.xx,pddata->Upsilon.xy,pddata->Upsilon.xz);

		fprintf(fp_1D, " %1.10E %1.10E %1.10E %1.10E %1.10E | ",pddata->nu_ie,pddata->nu_in, pddata->lnLambda,pTri->GradTe.x,pTri->GradTe.dot(unit_radial));



		fprintf(fp_1D, " %1.10E %1.10E ", 
			pddata->pressure.x,
			pddata->pressure.dot(unit_radial));

		fprintf(fp_1D, " %1.10E %1.10E ", 
			pddata->qoverM_times_E.x,
			pddata->qoverM_times_E.dot(unit_radial));

		fprintf(fp_1D, " %1.10E %1.10E ", 
			pddata->minus_omega_ci_cross_vi.x,
			pddata->minus_omega_ci_cross_vi.dot(unit_radial));

		fprintf(fp_1D, " %1.10E %1.10E ", 
			pddata->friction_from_neutrals.x,
			pddata->friction_from_neutrals.dot(unit_radial));

		fprintf(fp_1D, " %1.10E %1.10E ", 
			pddata->friction_e.x,
			pddata->friction_e.dot(unit_radial));

		fprintf(fp_1D, " %1.10E %1.10E ", 
			pddata->thermal_force.x,
			pddata->thermal_force.dot(unit_radial));
		
		fprintf(fp_1D, " %1.10E %1.10E |", 
			pddata->a.x,
			pddata->a.dot(unit_radial));

		
		//
		fprintf(fp_1D,"\n");
		//

		fprintf(fp_1D,"%d %1.10E %1.10E 0 %1.10E %1.10E | ", 
			indexArray[iCaret],centroid.y,
			centroid.dot(unit_theta.xypart()),				
			pTri->B.y,pTri->B.dot(unit_theta));

		fprintf(fp_1D," %1.10E %1.10E %1.10E  ",
			pTri->sigma_erel.yx,pTri->sigma_erel.yy,pTri->sigma_erel.yz);
		
		fprintf(fp_1D," | %1.10E %1.10E ",pTri->E.y,
			pTri->E.dot(unit_theta));

		fprintf(fp_1D,"  %1.10E %1.10E | ",pTri->vrel_e_0.y,
			pTri->vrel_e_0.dot(unit_theta));
		
		fprintf(fp_1D," %1.10E %1.10E %1.10E %1.10E | ",
			pTri->elec.mass/pTri->area,
			pTri->elec.mom.y/pTri->elec.mass,
			pTri->elec.mom.dot(unit_theta)/pTri->elec.mass,
			pTri->elec.heat/pTri->elec.mass);

		

		fprintf(fp_1D," %1.10E %1.10E %1.10E %1.10E | ",
			pTri->ion.mass/pTri->area,
			pTri->ion.mom.y/pTri->ion.mass,
			pTri->ion.mom.dot(unit_theta)/pTri->ion.mass,
			pTri->ion.heat/pTri->ion.mass);
		
		
		fprintf(fp_1D," %1.10E %1.10E %1.10E %1.10E | ",
			pTri->neut.mass,
			pTri->neut.mom.y/pTri->neut.mass,
			pTri->neut.mom.dot(unit_theta)/pTri->neut.mass,
			pTri->neut.heat/pTri->neut.mass);

		fprintf(fp_1D," %1.10E %1.10E %1.10E | ",
			pTri->sigma_i.yx,pTri->sigma_i.yy,pTri->sigma_i.yz);
		
		fprintf(fp_1D," %1.10E %1.10E | ",
			pTri->vion_0.y, pTri->vion_0.dot(unit_theta));

		// ==

		fprintf(fp_1D, " %1.10E %1.10E %1.10E | ",
			pddata->Upsilon.yx,pddata->Upsilon.yy,pddata->Upsilon.yz);

		fprintf(fp_1D, " 0 0 0 %1.10E %1.10E | ",pTri->GradTe.y,pTri->GradTe.dot(unit_theta));
		

		fprintf(fp_1D, " %1.10E %1.10E ", 
			pddata->pressure.y,
			pddata->pressure.dot(unit_theta));

		fprintf(fp_1D, " %1.10E %1.10E ", 
			pddata->qoverM_times_E.y,
			pddata->qoverM_times_E.dot(unit_theta));

		fprintf(fp_1D, " %1.10E %1.10E ", 
			pddata->minus_omega_ci_cross_vi.y,
			pddata->minus_omega_ci_cross_vi.dot(unit_theta));

		fprintf(fp_1D, " %1.10E %1.10E ", 
			pddata->friction_from_neutrals.y,
			pddata->friction_from_neutrals.dot(unit_theta));

		fprintf(fp_1D, " %1.10E %1.10E ", 
			pddata->friction_e.y,
			pddata->friction_e.dot(unit_theta));

		fprintf(fp_1D, " %1.10E %1.10E ", 
			pddata->thermal_force.y,
			pddata->thermal_force.dot(unit_theta));
		
		fprintf(fp_1D, " %1.10E %1.10E |", 
			pddata->a.y,
			pddata->a.dot(unit_theta));
		
		
		//
		fprintf(fp_1D,"\n");
		//
		
		fprintf(fp_1D,"%d %1.10E %1.10E 0 %1.10E %1.10E | ", 
			indexArray[iCaret],0.0,0.0,				
			pTri->B.z,pTri->B.z);

		fprintf(fp_1D," %1.10E %1.10E %1.10E  ",
			pTri->sigma_erel.zx,pTri->sigma_erel.zy,pTri->sigma_erel.zz);

		fprintf(fp_1D," | %1.10E %1.10E ",pTri->E.z,
			pTri->E.z);
		
		fprintf(fp_1D,"  %1.10E %1.10E | ",pTri->vrel_e_0.z,
			pTri->vrel_e_0.z);
		
		
		fprintf(fp_1D," %1.10E %1.10E %1.10E %1.10E | ",
			pTri->elec.mass/pTri->area,
			pTri->elec.mom.z/pTri->elec.mass,
			pTri->elec.mom.z/pTri->elec.mass,
			pTri->elec.heat/pTri->elec.mass);
		
		
		fprintf(fp_1D," %1.10E %1.10E %1.10E %1.10E | ",
			pTri->ion.mass/pTri->area,
			pTri->ion.mom.z/pTri->ion.mass,
			pTri->ion.mom.z/pTri->ion.mass,
			pTri->ion.heat/pTri->ion.mass);
		
		fprintf(fp_1D," %1.10E %1.10E %1.10E %1.10E | ",
			pTri->neut.mass,
			pTri->neut.mom.z/pTri->neut.mass,
			pTri->neut.mom.z/pTri->neut.mass,
			pTri->neut.heat/pTri->neut.mass);


		fprintf(fp_1D," %1.10E %1.10E %1.10E | ",
			pTri->sigma_i.zx,pTri->sigma_i.zy,pTri->sigma_i.zz);
		
		fprintf(fp_1D," %1.10E %1.10E | ",
			pTri->vion_0.z, pTri->vion_0.z);

		// ==
		
		fprintf(fp_1D, " %1.10E %1.10E %1.10E | ",
			pddata->Upsilon.zx,pddata->Upsilon.zy,pddata->Upsilon.zz);
		
		
		fprintf(fp_1D, " 0 0 0 %1.10E %1.10E | ",pTri->GradTe.z,pTri->GradTe.z);
		
		fprintf(fp_1D, " %1.10E %1.10E ", 
			pddata->pressure.z,
			pddata->pressure.z);

		fprintf(fp_1D, " %1.10E %1.10E ", 
			pddata->qoverM_times_E.z,
			pddata->qoverM_times_E.z);

		fprintf(fp_1D, " %1.10E %1.10E ", 
			pddata->minus_omega_ci_cross_vi.z,
			pddata->minus_omega_ci_cross_vi.z);

		fprintf(fp_1D, " %1.10E %1.10E ", 
			pddata->friction_from_neutrals.z,
			pddata->friction_from_neutrals.z);

		fprintf(fp_1D, " %1.10E %1.10E ", 
			pddata->friction_e.z,
			pddata->friction_e.z);

		fprintf(fp_1D, " %1.10E %1.10E ", 
			pddata->thermal_force.z,
			pddata->thermal_force.z);
		
		fprintf(fp_1D, " %1.10E %1.10E |", 
			pddata->a.z,
			pddata->a.z);
		
		
		fprintf(fp_1D,"\n\n");
		
	};

	fclose(fp_1D);
	
	sprintf(buffer,"graph1D_%d.txt",GlobalStepsCounter);
	fp_1D = fopen(buffer,"w");
	
	fprintf(fp_1D,
			"iTri r area Br Btheta Bz "
			" sigma_erel_xx sigma_erel_xy sigma_erel_xz  sigma_erel_yx  sigma_erel_yy sigma_erel_yz  sigma_erel_zx sigma_erel_zy sigma_erel_zz  Er Etheta Ez "
			" vrel_e_0_r theta z "

			" n_e v_e_r theta z T_e "
			" n_i v_i_r theta z T_i "
			" n_n v_n_r theta z T_n "

			" Upsilon_e_zz Upsilon_e_yz "
			" nu_ieBar nu_in_MT nu_ei nu_en lnLambda_ei "
			" sigma_J xyz  J0 xyz  J xyz \n\n"
			);
	// wish to add: various heating rates

	for (iCaret = 0; iCaret < numTris1D; iCaret++)
	{
		pTri = pDestMesh->T + indexArray[iCaret];
		pddata = &(debugdata1D[iCaret]);
				
		// spit out info from pTri
		pTri->GenerateContiguousCentroid(&centroid,pTri);
		unit_radial.x = centroid.x/radiiArray[iCaret];
		unit_radial.y = centroid.y/radiiArray[iCaret];
		unit_radial.z = 0.0;
		unit_theta.x = -unit_radial.y;
		unit_theta.y = unit_radial.x;
		unit_theta.z = 0.0;
		
		fprintf(fp_1D, "%d %1.10E %1.10E "
			"%1.10E %1.10E %1.10E ",
			iTri, radiiArray[iCaret], pTri->area, pTri->B.dot(unit_radial), pTri->B.dot(unit_theta),pTri->B.z);

		fprintf(fp_1D,"%1.10E %1.10E %1.10E "
			"%1.10E %1.10E %1.10E " "%1.10E %1.10E %1.10E "
			"%1.10E %1.10E %1.10E ",

			pTri->sigma_erel.xx, pTri->sigma_erel.xy, pTri->sigma_erel.xz, 
			pTri->sigma_erel.yx, pTri->sigma_erel.yy, pTri->sigma_erel.yz, 
			pTri->sigma_erel.zx, pTri->sigma_erel.zy, pTri->sigma_erel.zz, 
			pTri->E.dot(unit_radial),pTri->E.dot(unit_theta),pTri->E.z);
		
		fprintf(fp_1D, " %1.10E %1.10E %1.10E ", pTri->vrel_e_0.dot(unit_radial),
			pTri->vrel_e_0.dot(unit_theta),pTri->vrel_e_0.z);
		
		// This is taking place after acceleration, before second round of Ez Az.
		// We just set ve = vi + vrel, vrel = vrel_0 + sigma E -- the E that exists up to now.

		fprintf(fp_1D, "%1.10E %1.10E %1.10E %1.10E %1.10E ",
			pTri->elec.mass/pTri->area,
			(pTri->elec.mom.dot(unit_radial))/pTri->elec.mass,
			(pTri->elec.mom.dot(unit_theta))/pTri->elec.mass,
			pTri->elec.mom.z/pTri->elec.mass,
			pTri->elec.heat/pTri->elec.mass);

		fprintf(fp_1D, "%1.10E %1.10E %1.10E %1.10E %1.10E ",
			pTri->ion.mass/pTri->area,
			(pTri->ion.mom.dot(unit_radial))/pTri->ion.mass,
			(pTri->ion.mom.dot(unit_theta))/pTri->ion.mass,
			pTri->ion.mom.z/pTri->ion.mass,
			pTri->ion.heat/pTri->ion.mass);
		
		fprintf(fp_1D, "%1.10E %1.10E %1.10E %1.10E %1.10E ",
			pTri->neut.mass/pTri->area,
			(pTri->neut.mom.dot(unit_radial))/pTri->neut.mass,
			(pTri->neut.mom.dot(unit_theta))/pTri->neut.mass,
			pTri->neut.mom.z/pTri->neut.mass,
			pTri->neut.heat/pTri->neut.mass);
		
		fprintf(fp_1D, "%1.10E %1.10E %1.10E %1.10E %1.10E %1.10E %1.10E ",
			pddata->Upsilon.zz, pddata->Upsilon.yz, 
			pddata->nu_ie, pddata->nu_in,
			pddata->nu_ei, pddata->nu_en,
			pddata->lnLambda );
		
		// Note, nu_in = about 60 times greater than nu_en (?) 
		// must report both

		sigma_J = (pTri->sigma_i*(pTri->ion.mass-pTri->elec.mass)-pTri->sigma_erel*(pTri->elec.mass))/pTri->area;

		fprintf(fp_1D, "%1.10E %1.10E %1.10E %1.10E %1.10E %1.10E %1.10E %1.10E %1.10E ",
			sigma_J.xx, sigma_J.xy, sigma_J.xz,
			sigma_J.yx, sigma_J.yy, sigma_J.yz,
			sigma_J.zx, sigma_J.zy, sigma_J.zz);

		// but what is/was J0 ?
		
		J0 = (pTri->vion_0*(pTri->ion.mass-pTri->elec.mass)-pTri->vrel_e_0*pTri->elec.mass)/pTri->area;

		fprintf(fp_1D, "%1.10E %1.10E %1.10E ",
			J0.x,J0.y,J0.z);
		
		J = q*(pTri->ion.mom-pTri->elec.mom)/pTri->area;

		fprintf(fp_1D, "%1.10E %1.10E %1.10E ",
								J.x,J.y,J.z);
		// INTERMEDIATE J

		fprintf(fp_1D, "\n");
		
	};
	
	fclose(fp_1D);

	pTri = pDestMesh->T;
	for (iTri = 0; iTri < pDestMesh->numTriangles; iTri++)
	{
		pTri->indicator = 0; // wipe
		++pTri;
	};


	printf("%s \n",report_time(1));

	pTri = pDestMesh->T;
	for (iTri = 0; iTri < pDestMesh->numTriangles; iTri++)
	{	if ((pTri->ion.heat < 0.0) || (pTri->neut.heat < 0.0) || (pTri->elec.heat < 0.0))
		{
			iTri = iTri;
		};
	
		GlobalEnergyInput += 0.5*h*q*pTri->E.dot(pTri->ion.mom-pTri->elec.mom);

		++pTri;
	};
	

	fp = fopen(FUNCTIONALFILENAME,"a");
	fprintf(fp,"fric_in,fric_en,fric_ei,J.E "
		       " %1.9E %1.9E %1.9E %1.9E ",
		GlobalFrictionalHtg_in,GlobalFrictionalHtg_en,GlobalFrictionalHtg_ei,
		GlobalEnergyInput);
	fclose(fp);
	
	
	printf("Second pass for Az Ez: ");
	report_time(0);

	// Want to adjust Ez again. Ignore the extra effects on Jxy.
	// Sensible way would be: when we get Ez the first time, assume Exy_k+1 = Exy_k.
	// ====================================================================
	// altering A slightly means we can then re-set E using the same calculation as before.
	


	// This time around we take the existing found Jz as the initial state and 
	// it's the change in Ez -- both globally and locally -- that 
	//		brings about an ADJUSTMENT to Jz in cell.
		
	

	// Let's look over it.
	// We have set elec.mom
	// and we still have set up an Ohm's Law which says, add a bit more vrel
	// .. and a bit more v_ion ..
	// given _ Ez change _.
		
	
	// What vars are used in the following and shall we set them up here?

	// It will want
	// v_erel_0 and vion_0 --- correct?
	
	pTri= pDestMesh->T;
	for (iTri = 0; iTri < pDestMesh->numTriangles; iTri++)
	{
		pTri->vrel_e_0 = pTri->elec.mom/pTri->elec.mass-pTri->ion.mom/pTri->ion.mass;
		pTri->vion_0 = pTri->ion.mom/pTri->ion.mass;
		++pTri;
	};

	// Now we only add to them given ADDITIONAL Ez

	
	// ????????????????????????????????????
	// ? Is that the way we want to do it ?
	// ????????????????????????????????????
	
	real StoreGlobalEz = GlobalEz;
	GlobalDefaultIz = 0.0;
	Global_dIz_by_dEz = 0.0;
	pTri= pDestMesh->T;
	for (iTri = 0; iTri < pDestMesh->numTriangles; iTri++)
	{
		pTri->ROC_nT = pTri->E.z; // storing
		// pTri->denominator = pTri->ion.mom.z-pTri->elec.mom.z; 
		// what it is now is the default value
		// however, in the routine the default will be set from vion_0, vrel_e_0 -- ???
		GlobalDefaultIz += q*(pTri->ion.mom.z-pTri->elec.mom.z);
		Global_dIz_by_dEz += q*((pTri->ion.mass-pTri->elec.mass)*pTri->sigma_i.zz
								-pTri->elec.mass*pTri->sigma_erel.zz);
		
		// ???
		++pTri;
	};
	pVertex = pDestMesh->X;
	for (iVertex = 0; iVertex < pDestMesh->numVertices; iVertex++)
	{
		pVertex->ion_pm_Heat_numerator = pVertex->A.z; // storing
		++pVertex;
	};
	

	// Here wish to put in a test of what Iz is beforehand:
	
	Iz = 0.0;
	pTri= pDestMesh->T;
	for (iTri = 0; iTri < pDestMesh->numTriangles; iTri++)
	{			
		Iz += q*(pTri->ion.mom.z-pTri->elec.mom.z);
			
		++pTri;
	};
	printf("Iz_prescribed = %1.12E Final Iz = %1.12E ExternalEz: %1.8E\n",
		GlobalIzPrescribed, Iz, GlobalEz); 
	

	printf("Solve for Az,Ez round 2: ");
	pDestMesh->SolveForAzAndEz(pDestMesh,2);
	// Model for Ecell here: pTri->ROC_nT + GlobalEz_extra - (1/ch)(Az_updated_k+1 - Az_k+1)
	
	
	Iz = 0.0;
	pTri= pDestMesh->T;
	for (iTri = 0; iTri < pDestMesh->numTriangles; iTri++)
	{
		// So now we want to assume	we augmented E and this affects J , per sigma:
		// first calculate additional E:
#ifdef VORONOI_A
		pTri->E.z = GlobalEz - ( pTri->coeff[0]*(pTri->cornerptr[0]->A.z - pTri->cornerptr[0]->ion_pm_Heat_numerator)
							 + pTri->coeff[1]*(pTri->cornerptr[1]->A.z - pTri->cornerptr[1]->ion_pm_Heat_numerator)
							 + pTri->coeff[2]*(pTri->cornerptr[2]->A.z -pTri->cornerptr[2]->ion_pm_Heat_numerator)
							 )/(c*h*pTri->area);
#else
		pTri->E.z =  GlobalEz // additional global Ez
						- THIRD*( 
									(pTri->cornerptr[0]->A.z - pTri->cornerptr[0]->ion_pm_Heat_numerator) // = stored Az
							+		(pTri->cornerptr[1]->A.z - pTri->cornerptr[1]->ion_pm_Heat_numerator)
							+		(pTri->cornerptr[2]->A.z - pTri->cornerptr[2]->ion_pm_Heat_numerator)   )/(c*h) ;
#endif

		pTri->ion.mom.z += pTri->ion.mass*pTri->sigma_i.zz*pTri->E.z;
		pTri->elec.mom.z += pTri->elec.mass*(pTri->sigma_erel.zz+pTri->sigma_i.zz)*pTri->E.z;
		
		pTri->E.z += pTri->ROC_nT; // the value it had before we augmented it.
							
		Iz += q*(pTri->ion.mom.z-pTri->elec.mom.z);
		// This should come out finally at the desired value.
		++pTri;
	};

	
	sprintf(buffer,"graph1D_%d.txt",GlobalStepsCounter);
	fp_1D = fopen(buffer,"a");
	
	fprintf(fp_1D,
			"\n\niTri r area  Er Etheta Ez  Jz  Jtheta  Jr  \n\n"
			);
	// wish to add: various heating rates

	for (iCaret = 0; iCaret < numTris1D; iCaret++)
	{
		pTri = pDestMesh->T + indexArray[iCaret];
		pddata = &(debugdata1D[iCaret]);
				
		// spit out info from pTri
		pTri->GenerateContiguousCentroid(&centroid,pTri);
		unit_radial.x = centroid.x/radiiArray[iCaret];
		unit_radial.y = centroid.y/radiiArray[iCaret];
		unit_radial.z = 0.0;
		unit_theta.x = -unit_radial.y;
		unit_theta.y = unit_radial.x;
		unit_theta.z = 0.0;
		
		fprintf(fp_1D, "%d %1.10E %1.10E %1.10E %1.10E %1.10E  ",
			iTri, radiiArray[iCaret], pTri->area,
			pTri->E.dot(unit_radial),pTri->E.dot(unit_theta),pTri->E.z);
		
		fprintf(fp_1D, "%1.10E %1.10E %1.10E ",
			q*(pTri->ion.mom-pTri->elec.mom).dot(unit_radial)/pTri->area,
			q*(pTri->ion.mom-pTri->elec.mom).dot(unit_theta)/pTri->area,
			q*(pTri->ion.mom-pTri->elec.mom).z/pTri->area
				);
		
		fprintf(fp_1D, "\n");
		
	};
	
	fclose(fp_1D);
	// 
		
		real sigma_parallel, sigma_perp, sigma_Hall, tempval;
		Vector3 b;
		Tensor3 sigma_J_E;
		pTri= pDestMesh->T;
		for (iTri = 0; iTri < pDestMesh->numTriangles; iTri++)
		{
			sigma_J_E = q*(
				 (pTri->ion.mass-pTri->elec.mass)*pTri->sigma_i
				 - pTri->elec.mass*pTri->sigma_erel)/pTri->area;
		// note that we divided by area. It's density of current we're interested in, not additional current


		// Idea. 
		// E has some component parallel to the B field.
		// The rest is E perp.
		// Together with the Hall direction that can make
		// an orthonormal basis.

		// However, we have 9 elements in sigma and try to reduce
		// it to 3, par perp and Hall.

		// Sigma_par = (sigma b) dot b

		// Estimate perp component:
		// Using the property that bxbx + byby + bzbz = 1

		// Sxx + Syy + Szz - sigma_par = 2 sigma_perp
		// Take max with zero

		// Estimate Hall component: 
		// Using the fact that the other part is symmetric

		// - Sxy + Syx + Sxz - Szx + Szy - Syz = 2 sigma_Hall (bx + by + bz)

		// Take (Syx-Sxy)^2 + (Sxz-Szx)^2 + (Szy-Syz)^2 = 4 Sigma_Hall ^2

		b = pTri->B;
		b = b/b.modulus();

		sigma_parallel = (sigma_J_E*b).dot(b);
		// if this comes out negative, leave it in there and show it with a special colour.

		sigma_perp = 0.5*(sigma_J_E.xx + sigma_J_E.yy + sigma_J_E.zz - sigma_parallel);
		// if that came out negative it's because the matrix is distorted.

		tempval = (sigma_J_E.yx - sigma_J_E.xy)*(sigma_J_E.yx - sigma_J_E.xy)
				+ (sigma_J_E.xz - sigma_J_E.zx)*(sigma_J_E.xz - sigma_J_E.zx)
				+ (sigma_J_E.zy - sigma_J_E.yz)*(sigma_J_E.zy - sigma_J_E.yz);
		sigma_Hall = 0.5*sqrt(tempval);
		
		// Sigma matrices will not be used further so we can safely mess them up.
		pTri->sigma_i.xx = sigma_parallel;
		pTri->sigma_i.xy = sigma_perp;
		pTri->sigma_i.yy = sigma_Hall;

		++pTri;
		};
	
		// We do this calculation here before B has been updated.

		// Colours to use:

		// violet: parallel dominates perp

		// green (up to 0.5 is enough): Hall/parallel at max
		// -- but also reduce the red component for this

		// 1 red, 0.5 blue: near isotropic

	
	GlobalEz += StoreGlobalEz;
	
	printf("Iz_prescribed = %1.12E Final Iz = %1.12E ExternalEz: %1.8E\n",
		GlobalIzPrescribed, Iz, GlobalEz); 
	//pDestMesh->SolveForA(this);//(this, false); 
	pDestMesh->GetBFromA();
	pDestMesh->RecalculateVertexVariables(); // note vertex variables are sometimes nonzero Jxy even when cells are zero Jxy 
	
	// Not sure if or how this should be different if at all.

	printf("%s  \n",report_time(1));
	report_time(0);
	
	pTri = pDestMesh->T;
	for (iTri = 0; iTri < pDestMesh->numTriangles; iTri++)
	{
		if ((pTri->ion.heat < 0.0) || (pTri->neut.heat < 0.0) || (pTri->elec.heat < 0.0))
		{
			iTri = iTri;
		};
		++pTri;
	};

	// Second half using B_k+1:

	printf("Smoothings: ");	
	pDestMesh->ConservativeSmoothings(0.5*h);	// Note that doing viscous heating should be part of smoothings
	// we bundled the species together because we calculate displacement SD and then fill same memory slot with kappa..
	// and this is done for all species at once to save a few calls to exp.


	// Viscous smoothing has only a weak effect on fields -- my conjecture.
	// Allowing this to happen here means that we don't really affect v_e at all since it still gets thrown away;
	// instead we should have to allow an ODE for vrel in place of tensor Ohm's Law. This would help stabilise the simulation.
	
	
	printf("%s  \n",report_time(1));

	fp = fopen(FUNCTIONALFILENAME," a");
	fprintf(fp,"\n");
	fclose(fp);

	h = store_h;

	printf("TriMesh::Advance completed. \n");
	
	// =================================================

	delete [] indexArray;
	delete [] radiiArray;
	delete [] debugdata1D;

	



//
//
//	// Now do advection: create new mesh and advect into it; do post-advection sweep-up including compressive htg
//	// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
//
//	// Make destination mesh:
//	
//	// Make sure vertex variables are properly recalculated for here?
//
//	printf("Create new mesh: ");	
//	// First populate pDestMesh vertices to have position 0.
//	pDestMesh->ZeroVertexPositions();
//	this->MakePressureAccelData(COMBINE_EI_2ND_ORDER_PRESSURE); 
//	
//	// swizzle:
//	pVertex = X;
//	for (iVertex = 0; iVertex < numVertices; iVertex++)
//	{
//		temp1 = pVertex->Pressure_numerator_x;
//		temp2 = pVertex->Pressure_numerator_y;
//		temp3 = pVertex->Polygon_mass;
//		pVertex->Pressure_numerator_x = pVertex->IonP_Viscous_numerator_x; // ion pressure
//		pVertex->Pressure_numerator_y = pVertex->IonP_Viscous_numerator_y;
//		pVertex->Polygon_mass = pVertex->ion_pm_Heat_numerator;
//		pVertex->IonP_Viscous_numerator_x = temp1;
//		pVertex->IonP_Viscous_numerator_y = temp2;
//		pVertex->ion_pm_Heat_numerator = temp3;
//		++pVertex;
//	};
//	this->ApplyVertexMoves(SPECIES_ION,pDestMesh); 
//	//this->MakePressureAccelData(SPECIES_NEUTRAL,0.0); 
//	
//	// If necessary, swizzle our pressure data into different places to do each one:
//	pVertex = X;
//	for (iVertex = 0; iVertex < numVertices; iVertex++)
//	{
//		temp1 = pVertex->Pressure_numerator_x;
//		temp2 = pVertex->Pressure_numerator_y;
//		temp3 = pVertex->Polygon_mass;
//		pVertex->Pressure_numerator_x = pVertex->IonP_Viscous_numerator_x; // swizzle back
//		pVertex->Pressure_numerator_y = pVertex->IonP_Viscous_numerator_y;
//		pVertex->Polygon_mass = pVertex->ion_pm_Heat_numerator;
//		pVertex->IonP_Viscous_numerator_x = temp1;
//		pVertex->IonP_Viscous_numerator_y = temp2;
//		pVertex->ion_pm_Heat_numerator = temp3;
//		++pVertex;
//	};
//	this->ApplyVertexMoves(SPECIES_NEUTRAL,pDestMesh); 	
//
//	printf("got to here 2");
//	// We have to be careful here - taking average of new positions is OK as long as we are careful of periodic boundary crossing.
//	// No boundary crossings => can take average that's easy.
//	// Some of the moves take it across PB ==> do what?
//	// DO NOT WRAP 	// WRAP AFTERWARDS
//	FinishAdvectingMesh(pDestMesh); // OK I think we might be able to make it work this way.
//	
////	pDestMesh->SetupInsVertsOuterVerts();	
//	// ^^ Need to change: want to
//	// . redim the arrays, 
//	// . set up new positions - this requires to know which vertices we actually are going to project of course.
//	// . Check out FinishAdvectingMesh.
//
//	// delete them if they are already dimensioned, redimension and assign iScratch
//
//
//	printf("got to here 4");
//	// Note that iScratch is corrupted if we ever do DestroyOverlaps, ReDelaunerize, JumpVertices, SwimVertices.
//	// But we think it should remain with the "ins index" meaning as long as a mesh is settled on.
//
//	// ^^^ important bookkeeping thing -- if we do any of the above we cannot then assume
//	// the validity of iScratch. 	
//
//	// What is cleanup for ins verts and outer verts???
//
//	pTri = pDestMesh->T;
//	for (iTri = 0; iTri < pDestMesh->numTriangles; iTri++)
//	{
//		pTri->area = pTri->GetArea(); // pretty basic and useful, before we go on!
//		++pTri;
//	};
//	// Advection of species into new mesh:
//	
//	printf("Advect species: ");
//	pDestMesh->ZeroCellData();
//	// Note that we combine pressure between ions and electrons.
//	//this->MakePressureAccelData();//SPECIES_NEUTRAL,0.0); 
//	// Not necessary to call again here now.
//	// If necessary, swizzle our pressure data into different places to do each one.
//	this->AdvectionCompression(SPECIES_NEUTRAL,pDestMesh); 
//	// swizzle:
//	pVertex = X;
//	for (iVertex = 0; iVertex < numVertices; iVertex++)
//	{
//		temp1 = pVertex->Pressure_numerator_x;
//		temp2 = pVertex->Pressure_numerator_y;
//		temp3 = pVertex->Polygon_mass;
//		pVertex->Pressure_numerator_x = pVertex->IonP_Viscous_numerator_x;
//		pVertex->Pressure_numerator_y = pVertex->IonP_Viscous_numerator_y;
//		pVertex->Polygon_mass = pVertex->ion_pm_Heat_numerator;
//		pVertex->IonP_Viscous_numerator_x = temp1;
//		pVertex->IonP_Viscous_numerator_y = temp2;
//		pVertex->ion_pm_Heat_numerator = temp3;
//		++pVertex;
//	};
//	//this->MakePressureAccelData_And_ApplyAdditionalMomentum(SPECIES_ION,0.0); 
//	this->AdvectionCompression(SPECIES_ION,pDestMesh); 
//	//this->MakePressureAccelData_And_ApplyAdditionalMomentum(SPECIES_ELECTRON,0.0); 
//	pVertex = X;
//	for (iVertex = 0; iVertex < numVertices; iVertex++)
//	{
//		pVertex->Pressure_numerator_x = pVertex->eP_Viscous_denominator_x;
//		pVertex->Pressure_numerator_y = pVertex->eP_Viscous_denominator_y;
//		pVertex->Polygon_mass = pVertex->e_pm_Heat_denominator;
//		++pVertex;
//	};
//	this->AdvectionCompression(SPECIES_ELECTRON,pDestMesh); 
//
//	pDestMesh->RestoreSpeciesTotals(this); // scale to stamp out tiny changes in species mass etc
//	pDestMesh->RecalculateVertexVariables(); // before calling Bwd Gauss
//
//	printf("%s \n",report_time(1));
//	report_time(0);
//
//	// Sweep-up part will now need to be careful that it does not upset data stored for the predictive effort.
//	// particularly pVertex->phi !!
//	// We'll have to look carefully at that. Cross that bridge when we come to it!
//
//	// SWEEP-UP:
//	pDestMesh->BackwardGaussRelaxationPhiOnCellsEedgeWeightedJacobiWithBilaterals_PostAdvectionOnly(this);
//	printf("Solve Exy: %s \n",report_time(1));
//	report_time(0);
//
//	// Magnetic field
//	// ===========================
//
//	printf("Solve for A and hence B: ");	
//	pDestMesh->RecalculateVertexVariables(); // note vertex variables are sometimes nonzero Jxy even when cells are zero Jxy 
//	pDestMesh->Calculate_Magnetic_Field(this, false); 
//
//	// Not sure if or how this should be different if at all.
//
//	printf("\n\nMagnetic: %s  \n",report_time(1));
//	// We also do before smoothing because we shall need to apply magnetic effects during smoothing.
//	report_time(0);
//
//	printf("Smoothings: ");	
//	pDestMesh->ConservativeSmoothings();	// Note that doing viscous heating should be part of smoothings
//	// we bundled the species together because we calculate displacement SD and then fill same memory slot with kappa..
//	// and this is done for all species at once to save a few calls to exp.
//	printf("%s  \n",report_time(1));
//
//	printf("TriMesh::Advance completed. \n");
//	


	// =========================================================================================

	// Electron v sequence: do fastest timescale last.
	// so particularly, do electrostatic alignment of v last.
	// Priority order (reverse of sequence prior to advection) :
	//		Electrostatic : fastest
	//		Lorentz
	//		Pressure
	//		Viscous smoothing
	// Not sure about circuit current increase but bear in mind we want to actually end up with the required z current, so put it near the end of the step.
	// Luckily for now electrostatics does not affect the z current but when it does, they have to be combined.
	
	// For any HYPO-DIFFUSION done after snapback, we keep n_e = n_i and move ambipolar. But basically think using T/M for all is not very wrong.
	
	// electrostatic last and Lorentz matrix second-last, ie apply it to all acceleration.
	// (should do magnetic Lorentz ion accel around the same time as magnetic Lorentz on e?)

	// Those are the main things.

	// Thinking that it might be nice to do Heat Friction immediately after viscous heating and before pressure.
	// But then do Momentum Friction immediately after pressure.
	// This would make it more efficient to store some stuff, but we may need to anyway to do the reduced compression resist above.
	
	// We want to do resistance immediately after e pressure; 
	// meanwhile we want to set B from sensible J, so do after advection

	// Note that on the first step of an actual simulation run, GSC == 1
	
	// =========================================================================================

	// Electron sequence: 
	// we need to apply dyn_conduct after pressure; and after smoothing, to be fussy about it
	// so do smoothing then pressure then friction 

	// =========================================================================================

	// New thinking:

	// Perform substeps because of the interaction between temperature friction and ionisation
	// These two together form the equilibrium

	// Likewise circuit E accel, thermal pressure and momentum friction maintain the balance -
	// friction alone will destroy it. 

	// Thus if h nu >> 1, we are not handling these interactions properly. Big lump of thermal pressure accel
	// followed by big lump of friction followed by big lump of frictioned circuit E might not be accurate? 
	// It's the question, how accurate is a backward step. It's robust but not accurate.
	
	// However, ES forces also destroy pressure very fast but -- elastic transfer of momentum to ions 
	// before moving is actually a good approximation to ES force effect anyway in most cases.
	
	// At any rate it doesn't feel that T friction and ionisation cannot be at all accurate on h >> 1/nu. 

	// We now know that most time will be spent on solving for Exy and Axyz. So we can afford substeps
	// here if we want.

	// Since this is creating a serial task we will have to think carefully how to massively parallelise.

	// Per-cell part:
	// FOR NOW we will just take the same routines and make a subcycle within this routine
	// For now, no adaptive substep. On GPU, adapt substep to cell.

		//printf("Per-cell inner cycle. ");
		//this->RecalculateVertexVariables(true); // populate vertex v with n-weighted data for each species - for time t_k

		//// First create the pressure acceleration data.

		//this->MakePressureAccelData_And_ApplyAdditionalMomentum(SPECIES_ION,1.0); // get 4x sums and differences at each vertex, and Voronoi mass
		//this->MakePressureAccelData_And_ApplyAdditionalMomentum(SPECIES_NEUTRAL,1.0); 
		//this->MakePressureAccelData_And_ApplyAdditionalMomentum(SPECIES_ELECTRON,1.0); 
		//
		//// Put the rest all into one per-cell routine for CUDA.

		////this->Calculate_Ez_linear_for_Circuit_Current();	// well, let's be careful. 
		//// We are going to come up with a matrix within the friction procedure that will give rise to some acceleration given E.
		//// For now, we can set Ez within there to set the circuit current.
		//// NO - it has to be set globally given all of these matrices! Oh dear !!
		//// Well we just need to calculate one data value for all: given a certain Ez what is the addition to Jz there?

		//// So perform a feint --- this way will be only for 2D; for 3D we set it at the same time as Gauss.

		//pTri = T; // this bit we 
		//for (iTri = 0; iTri < numTriangles; iTri++)
		//{
		//	
		//	++pTri;
		//};
		////this->CalculateFrictionCoefficients(); // makes sense - no point recalculating these _or_ pressure accel.
		//
		//// Regarding substeps, note that the ionisation <-> T friction timescale is likely to be much longer 
		//// than the momentum friction <-> acceleration timescale.

		//printf("F");	
		//this->FrictionAndResistance();  // creates dynamic conductivity matrix
		//printf("I");		
		//this->IoniseAndRecombine();			
		//evaltime += SUBSTEP; // time that this system will now be accelerated for
		//
		//// Work out linear increase in Ez during step; get it right again at start of step (one day it will be towards the end)

		//printf("Ez ");	// do circuit current, applying magnetic Lorentz 
		//
	
	return 0;
}
real TriMesh::GetMaxTimestep()
{
	// simple initially:
	// find max velocity; take 1e-3 / max velocity
	// so v= 1e-9 => h = 1e-12 => hv = 1e-3
	
	long iTri;
	Triangle * pTri = T;
	real maxvsq = 0.0;
	real vsq;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		if ((pTri->ion.mass > 0.0) && (pTri->elec.mass > 0.0))
		{
			vsq = max((pTri->ion.mom.x*pTri->ion.mom.x+pTri->ion.mom.y*pTri->ion.mom.y)/
						(pTri->ion.mass*pTri->ion.mass),
					(pTri->elec.mom.x*pTri->elec.mom.x+pTri->elec.mom.y*pTri->elec.mom.y)/
						(pTri->elec.mass*pTri->elec.mass));
			maxvsq = max(maxvsq, vsq);
		};
		++pTri;
	};
	real v = sqrt(maxvsq);
	return 1.0e-3 / v;
}
void Triangle::GenerateContiguousCentroid(Vector2 * pCentre, Triangle * pContig)
{
	Vector2 centroid;
	// first get our own centroid:
	if (periodic == 0) {
		centroid.x = THIRD*(cornerptr[0]->x + cornerptr[1]->x + cornerptr[2]->x);
		centroid.y = THIRD*(cornerptr[0]->y + cornerptr[1]->y + cornerptr[2]->y);
	} else {
		Vector2 u0,u1,u2;
		MapLeft(u0,u1,u2);
		centroid = THIRD*(u0+u1+u2);
	};
	
	if ( (pContig->periodic) && (centroid.x > 0.0) )
		centroid = Anticlockwise*centroid;
	
	if ((periodic > 0) && (pContig->periodic == 0) && (pContig->cornerptr[0]->x > 0.0))
		centroid = Clockwise*centroid; // match the one on the right
	
	*pCentre = centroid;
	
}
void TriMesh::MakePressureAccelData(int code)
{
	real height, r;
	Vertex * pVert;
	Triangle * pTri, * pTri1, * pTri2, * pTri3;
	real xdist, ydist,
		Average_nT_neut,Average_nT_ion,Average_nT_e,
		Voronoi_area, Voronoi_mass, Polygon_mass_ion, Polygon_mass_elec, Polygon_mass_neut,
		Average_nT1_neut, Average_nT2_neut,
		Average_nT1_ion,Average_nT2_ion,
		Average_nT1_e,Average_nT2_e;
	Vector2 u,u2,u12,intercept12,intercept23,u23;
	long iVert, iTri, i, j,inext, iprev,k;
	Vector2 cc, cc1, cc2, cc3, rhat;
	real neut_dot_rhat, ion_dot_rhat, e_dot_rhat;
	real theta, frac;
	real angle[100];
	long index[100];
	Proto * tempptr[100];
	real tri_intersection_area[100];
	real Neutral_intersection_mass[100];
	real Ion_intersection_mass[100];
	real Electron_intersection_mass[100];
	Vector2 u0,u1,u_ins;
	real r0,r1,r2,maxr,minr;
	real Polygon_area;
	Vector2 proj1,proj2;
	ConvexPolygon cp;

	static real const kB_to_3halves = sqrt(kB)*kB;
	static real const one_over_kB = 1.0/kB; // multiply by this to convert to eV
	static real const over_sqrt_m_e = 1.0/sqrt(m_e);

	// We no longer refresh vertex neighbours of vertices here. We should be coming in with an intact mesh that has ordered
	// triangle arrays and ordered vertex neighbour arrays.

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// 1. Populate putative accelerations per the accel at the present position (or maybe it would make more sense to consider the advance with v)
	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	

	// We do treat boundary vertices differently since the cell-centres-and-medians cell has to include
	// projected coordinates.
	// And note that boundary vertices are only to feel pressure along the boundary.

	// NEW JUSTIFICATION:
	// MEDIANS AND A POINT BETWEEN CENTROID AND CC GIVES 1/3 AREAS
	// SO QUITE JUSTIFIED IF WE TAKE 1/3 CELL MASS TOWARDS THE "VORONOI"
	// MASS HERE

	pVert = X;
	for (iVert = 0; iVert < numVertices; iVert++)
	{
		// For x-pressure: each edge of Voronoi needs to decide if it is left or right of 
		// shape. So find lowest and highest cell centre; in fact forming a circle of neighs is easiest.
 
		// First form circle, then decide which elements are lowest and highest and therefore
		// which edges are left and right; similarly, which are leftmost and rightmost and therefore 
		// which edges are up and down.
		// From there it's simple.

		// So when we get here, we have our circle of adjacent triangles and can form our Voronoi cell.

		// OK so we have the indices for highest and lowest.
		
		// We know that we increase angle by increasing index.
		// Therefore left hand side is after imaxy, before iminy.

		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		// In fact, we can always take (y value - previous y value) and it will have the right sign.
		// For y we take (previous x value - x value)
		// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		// Now add the edges: for x momentum addition, we add the vertical distance for the edge
		// times the average of the two vertex nT.
		pVert->Pressure_numerator_x = 0.0;
		pVert->Pressure_numerator_y = 0.0;
		pVert->IonP_Viscous_numerator_x = 0.0;
		pVert->IonP_Viscous_numerator_y = 0.0;
		pVert->eP_Viscous_denominator_x = 0.0;
		pVert->eP_Viscous_denominator_y = 0.0;

		if (pVert->flags < 3)
		{
			for (i = 0; i < pVert->triangles.len; i++)
			{
				inext = i+1; if (inext == pVert->triangles.len) inext = 0;

				// For each edge: 
				// take average vertex nT and have decided whether we are adding + or - for x,y

				pTri1 = ((Triangle *)pVert->triangles.ptr[i]);
				pTri2 = ((Triangle *)pVert->triangles.ptr[inext]);
				pTri1->ReturnCentre(&cc1,pVert); // same tranche as pVert if periodic
				pTri2->ReturnCentre(&cc2,pVert); 
				// Model centroid as where we get nT.

				Average_nT_neut = 0.5*(pTri1->neut.heat/pTri1->area + pTri2->neut.heat/pTri2->area);

				if (code == COMBINE_EI_2ND_ORDER_PRESSURE)
				{
					Average_nT_ion = 0.5*((pTri1->ion.heat + pTri1->elec.heat)/pTri1->area
										  + (pTri2->ion.heat + pTri2->elec.heat)/pTri2->area);		
					Average_nT_e = Average_nT_ion; // unweighted because 1/m in pressure cancels with weight in friction
					// ?!!!
				} else {
					Average_nT_ion = 0.5*(pTri1->ion.heat/pTri1->area + pTri2->ion.heat/pTri2->area);
					Average_nT_e = 0.5*(pTri1->elec.heat/pTri1->area + pTri2->elec.heat/pTri2->area);
				};
		
				ydist = cc2.y-cc1.y; 
				xdist = cc1.x-cc2.x; 

				pVert->Pressure_numerator_x -= ydist*Average_nT_neut;
				pVert->Pressure_numerator_y -= xdist*Average_nT_neut;
				pVert->IonP_Viscous_numerator_x -= ydist*Average_nT_ion;
				pVert->IonP_Viscous_numerator_y -= xdist*Average_nT_ion;
				pVert->eP_Viscous_denominator_x -= ydist*Average_nT_e;
				pVert->eP_Viscous_denominator_y -= xdist*Average_nT_e;
				
			};		
			// """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
			// Divide by particle mass since our "mom" recorded is just nv
			// No - do it in Voronoi mass.
			// """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

			// Divide by nM * Voronoi area to give additional velocity around here
			
			// To apply to cell momentum, multiply by h/m_species and we can
			// distribute it between cells according to whatever criteria we see fit.
		} else {
							
			pTri1 = ((Triangle *)pVert->triangles.ptr[0]);
			pTri1->ReturnCentre(&cc1,pVert); // same tranche as pVert if periodic
			
			if (pVert->flags == 3) {

				//cc1.project_to_ins(u_ins);

				// New way: it's not the projection of the centroid, but the median, that we want

				pTri1->GetBaseMedian(&u_ins, pVert);

			} else {
				// project to back of triangle only.
				if (pTri1->flags != 24) {
					printf("oh dear, (pTri1->flags != 24) \n"); getch();
				};
				
				// just take half of triangle height
				//pTri1->PopulatePositions(u0,u1,u2); // doesn't matter if periodic!
				//r0 = u0.modulus();
				//r1 = u1.modulus();
				//r2 = u2.modulus();
				//maxr = max(max(r1,r2),r0);
				//minr = min(min(r1,r2),r0);				
				//cc1.project_to_radius(u_ins, cc1.modulus() + 0.5*(maxr-minr));
				pTri1->GetOuterMedian(&u_ins, pVert);
			};
			Average_nT_neut = pTri1->neut.heat/pTri1->area;
			if (code == COMBINE_EI_2ND_ORDER_PRESSURE)
			{
				Average_nT_ion = (pTri1->ion.heat + pTri1->elec.heat)/pTri1->area;		
				Average_nT_e = Average_nT_ion; // unweighted because 1/m in pressure cancels with weight in friction
			} else {
				Average_nT_ion = (pTri1->ion.heat/pTri1->area);
				Average_nT_e = (pTri1->elec.heat/pTri1->area);
			};
		
			ydist = cc1.y-u_ins.y; 
			xdist = u_ins.x-cc1.x; 
			pVert->Pressure_numerator_x -= ydist*Average_nT_neut;
			pVert->Pressure_numerator_y -= xdist*Average_nT_neut;
			pVert->IonP_Viscous_numerator_x -= ydist*Average_nT_ion;
			pVert->IonP_Viscous_numerator_y -= xdist*Average_nT_ion;
			pVert->eP_Viscous_denominator_x -= ydist*Average_nT_e;
			pVert->eP_Viscous_denominator_y -= xdist*Average_nT_e;
			
			
			
			// Now do the usual:
			// Note: only do until we reach the last one.

			for (i = 0; i < pVert->triangles.len - 1; i++)
			{
				inext = i+1;

				// For each edge: 
				// take average vertex nT and have decided whether we are adding + or - for x,y

				pTri1 = ((Triangle *)pVert->triangles.ptr[i]);
				pTri2 = ((Triangle *)pVert->triangles.ptr[inext]);
				pTri1->ReturnCentre(&cc1,pVert); // same tranche as pVert if periodic
				pTri2->ReturnCentre(&cc2,pVert); 
				// Model centroid as where we get nT.

				Average_nT_neut = 0.5*(pTri1->neut.heat/pTri1->area + pTri2->neut.heat/pTri2->area);

				if (code == COMBINE_EI_2ND_ORDER_PRESSURE)
				{
					Average_nT_ion = 0.5*((pTri1->ion.heat + pTri1->elec.heat)/pTri1->area
										  + (pTri2->ion.heat + pTri2->elec.heat)/pTri2->area);		
					Average_nT_e = Average_nT_ion; // unweighted because 1/m in pressure cancels with weight in friction
				} else {
					Average_nT_ion = 0.5*(pTri1->ion.heat/pTri1->area + pTri2->ion.heat/pTri2->area);
					Average_nT_e = 0.5*(pTri1->elec.heat/pTri1->area + pTri2->elec.heat/pTri2->area);
				};
		
				ydist = cc2.y-cc1.y; 
				xdist = cc1.x-cc2.x; 

				pVert->Pressure_numerator_x -= ydist*Average_nT_neut;
				pVert->Pressure_numerator_y -= xdist*Average_nT_neut;
				pVert->IonP_Viscous_numerator_x -= ydist*Average_nT_ion;
				pVert->IonP_Viscous_numerator_y -= xdist*Average_nT_ion;
				pVert->eP_Viscous_denominator_x -= ydist*Average_nT_e;
				pVert->eP_Viscous_denominator_y -= xdist*Average_nT_e;
				
			};		

			if (pVert->flags == 3) {
				//cc2.project_to_ins(u_ins);
				pTri2->GetBaseMedian(&u_ins, pVert);
			} else {
				// project to back of triangle only.
				if (pTri2->flags != 24) {
					printf("oh dear, (pTri1->flags != 24) \n"); getch();
				};	
				pTri2->GetOuterMedian(&u_ins, pVert);
				// just take half of triangle height
				//height = pTri2->GetRoughRadialHeight();
				//cc2.project_to_radius(u_ins, cc2.modulus() + 0.5*height);
			};
			
			Average_nT_neut = pTri2->neut.heat/pTri2->area;
			if (code == COMBINE_EI_2ND_ORDER_PRESSURE)
			{
				Average_nT_ion = (pTri2->ion.heat + pTri2->elec.heat)/pTri2->area;		
				Average_nT_e = Average_nT_ion; // unweighted because 1/m in pressure cancels with weight in friction
			} else {
				Average_nT_ion = (pTri2->ion.heat/pTri2->area);
				Average_nT_e = (pTri2->elec.heat/pTri2->area);
			};
		
			ydist = u_ins.y-cc2.y;
			xdist = cc2.x-u_ins.x; 
			pVert->Pressure_numerator_x -= ydist*Average_nT_neut;
			pVert->Pressure_numerator_y -= xdist*Average_nT_neut;
			pVert->IonP_Viscous_numerator_x -= ydist*Average_nT_ion;
			pVert->IonP_Viscous_numerator_y -= xdist*Average_nT_ion;
			pVert->eP_Viscous_denominator_x -= ydist*Average_nT_e;
			pVert->eP_Viscous_denominator_y -= xdist*Average_nT_e;
			
			// """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
			// Divide by particle mass since our "mom" recorded is just nv
			// No - do it in Voronoi mass.
			// """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

			// Divide by nM * Voronoi area to give additional velocity around here
			
			// To apply to cell momentum, multiply by h/m_species and we can
			// distribute it between cells according to whatever criteria we see fit.

			// As for inner/outer pressure, it is considered zero at a boundary vertex.
			// So let's get rid of any radial components.

			r = sqrt(pVert->x*pVert->x+pVert->y*pVert->y);
			rhat.x = pVert->x/r;
			rhat.y = pVert->y/r;
			neut_dot_rhat = pVert->Pressure_numerator_x*rhat.x + pVert->Pressure_numerator_y*rhat.y;
			ion_dot_rhat = pVert->IonP_Viscous_numerator_x*rhat.x + pVert->IonP_Viscous_numerator_y*rhat.y;
			e_dot_rhat = pVert->eP_Viscous_denominator_x*rhat.x + pVert->eP_Viscous_denominator_y*rhat.y;
			pVert->Pressure_numerator_x -= rhat.x*neut_dot_rhat;
			pVert->Pressure_numerator_y -= rhat.y*neut_dot_rhat;
			pVert->IonP_Viscous_numerator_x -= rhat.x*ion_dot_rhat;
			pVert->IonP_Viscous_numerator_y -= rhat.y*ion_dot_rhat;
			pVert->eP_Viscous_denominator_x -= rhat.x*e_dot_rhat;
			pVert->eP_Viscous_denominator_y -= rhat.y*e_dot_rhat;
                                                                        
		}; // flags < 3 ?
		++pVert;
	};
	
	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		ZeroMemory(pTri->scratch,6*sizeof(real));
		++pTri;
	};

	Vector3 utemp;
	Vector2 centre;
	pVert = X;
	for (iVert = 0; iVert < numVertices; iVert++)
	{

		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		// Now we need to increase momentum in the cells that overlap this Voronoi cell.
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		// Assume that pVert->Pressure_numerator has been populated with the desired extra _momentum_
		// or maybe it's the desired extra acceleration.

		// The Vertex::triangles.ptr array is still sorted anticlockwise.

		// To calculate intersection area, we identify the intercepts on cell edges. Then we can create a quadrilateral and get area.

		// NEW change 10th April 2014:  let us attempt to stabilise by applying more
		// force on the side where there is more mass. 
		// Not sure how wrong (or even right) this will be.

		// Note: "Voronoi" does not mean Voronoi, it means using centroids and medians.

		// Voronoi might make sense from the perspective that it's the locus of closest points
		// and we are then applying pressure into this region .. thing is, the vertex is not
		// really involved at all here - it just served to store some information.

		Polygon_area = 0.0;
		Polygon_mass_ion = 0.0;
		Polygon_mass_elec = 0.0;
		Polygon_mass_neut = 0.0;

		// scrapped version with intercepts, use medians --> just 1/3 of triangle area each time.

		for (i = 0; i < pVert->triangles.len; i++)
		{
			pTri = ((Triangle *)(pVert->triangles.ptr[i]));
			Polygon_area += pTri->area*THIRD;
			Neutral_intersection_mass[i] = m_neutral*pTri->neut.mass*THIRD;
			Polygon_mass_neut += Neutral_intersection_mass[i];
			Ion_intersection_mass[i] = m_ion*pTri->ion.mass*THIRD;
			Polygon_mass_ion += Ion_intersection_mass[i];
			Electron_intersection_mass[i] = m_e*pTri->elec.mass*THIRD;
			Polygon_mass_elec += Electron_intersection_mass[i];

			//
			//
			//tri_intersection_area[i] = cp.GetArea();
			//Polygon_area += tri_intersection_area[i];
			//
			//Neutral_intersection_mass[i] = m_neutral*pTri2->neut.mass*tri_intersection_area[i]/pTri2->area;
			//Polygon_mass_neut += Neutral_intersection_mass[i];
			//Ion_intersection_mass[i] = m_ion*pTri2->ion.mass*tri_intersection_area[i]/pTri2->area;
			//Polygon_mass_ion += Ion_intersection_mass[i];
			//Electron_intersection_mass[i] = m_e*pTri2->elec.mass*tri_intersection_area[i]/pTri2->area;
			//Polygon_mass_elec += Electron_intersection_mass[i];

		};
				
		if (code == COMBINE_EI_2ND_ORDER_PRESSURE)
		{
			// collect total mass: for both species we divide total mom by total mass
			Polygon_mass_ion += Polygon_mass_elec;
			Polygon_mass_elec = Polygon_mass_ion;
		};
		pVert->Polygon_mass = Polygon_mass_neut; // used?
		pVert->ion_pm_Heat_numerator = Polygon_mass_ion;
		pVert->e_pm_Heat_denominator = Polygon_mass_elec;

		// None of the above used for anything but determining frac :




			// note that we have to be careful not to overwrite this for graphics etc 
			// while we are doing advections.
		    
		// be careful here.
		// Check what is used in AdvectionCompression :
		// Pressure_numerator_x, Pressure_numerator_y, Polygon_mass
		// So we are going to need to swizzle and store data for all species,
		// the way we are now doing it.		// Yep.

		// Fill in the following on tris for acceleration stage:
		// Note that we should not call with code COMBINE_EI_ if this is what it is for.

		for (i = 0; i < pVert->triangles.len; i++)
		{                                   
			pTri2 = ((Triangle *)(pVert->triangles.ptr[i]));			
		
			frac = Neutral_intersection_mass[i]/Polygon_mass_neut;
			// no h : it's momentum addition rate. Divide by mass of particle.
			pTri2->scratch[0] += frac*pVert->Pressure_numerator_x/m_neutral;
			pTri2->scratch[1] += frac*pVert->Pressure_numerator_y/m_neutral; 
			frac = Ion_intersection_mass[i]/Polygon_mass_ion;
			pTri2->scratch[2] += frac*pVert->IonP_Viscous_numerator_x/m_ion;
			pTri2->scratch[3] += frac*pVert->IonP_Viscous_numerator_y/m_ion;
			frac = Electron_intersection_mass[i]/Polygon_mass_elec;
			pTri2->scratch[4] += frac*pVert->eP_Viscous_denominator_x/m_e;
			pTri2->scratch[5] += frac*pVert->eP_Viscous_denominator_y/m_e;
//			pTri2->ion.mom.x += frac*pVert->Pressure_numerator_x/m_ion; // included h
//			pTri2->ion.mom.y += frac*pVert->Pressure_numerator_y/m_ion; 
		};

		++pVert;
	};


}



real TriMesh::CalculateIonisationTimestep(Triangle * pTri)
{
	real hsub;
	real minimum_mass;
	int surplusflag;

	static real const E0 = 13.6; // eV
	static real const one_over_kB = 1.0/kB;

	static real const MAX_IONISATION = 0.005;
	static real const MAX_PPN_OF_ATOMS_IONISE = 0.08;
	static real const MAX_PPN_OF_IONS_RECOMBINE = 0.08;
	static real const MINIMUM_IONISATION_STEP = 1.0e-15;

	if (pTri->ion.mass < pTri->elec.mass)
	{
		minimum_mass = pTri->ion.mass;
		surplusflag = 1; // electrons had the surplus
	} else {
		minimum_mass = pTri->elec.mass;
		surplusflag = 0; // ions had the surplus
	};
	
	real Te_eV = (pTri->elec.heat/pTri->elec.mass)*one_over_kB;
	real sqrtTe_eV = sqrt(Te_eV);
	real Tn_eV = (pTri->neut.heat/pTri->neut.mass)*one_over_kB;
		
			// current ionisation ppn:
	real area = pTri->area;
	real ntot = (minimum_mass + pTri->neut.mass)/area;
	real lambda = minimum_mass/(minimum_mass + pTri->neut.mass);

	real S = 1.0e-5*sqrtTe_eV*exp(-E0/Te_eV)/(E0*(6.0*E0 + Te_eV));
	real a_r = 2.7e-13/sqrtTe_eV;
	real a_3 = 8.75e-27/(Te_eV*Te_eV*Te_eV*Te_eV*sqrtTe_eV);

			// Find out the net ionisation rate at the new values to determine if the sign has changed.
	real d_lambda_by_dt = lambda*ntot*(1.0-lambda)*S - 
											ntot*lambda*lambda*(a_r + lambda*ntot*a_3);
			
			// Want to limit the substep so that we change by at most 1% in a substep let's say.
			// and so that the net ionised amount is at most 10% of remaining neutrals
			// or net recombination amount is at most 10% of ions.
	
			// We should generally then find that we get quite a few explicit steps 
			// (SHOULD be using RK2 not RK1 of course!)
			// before we have to do implicit solves, which is not a bad thing.
			// Respecting the ionisation timescale can be necessary to maintain accuracy of simulation results.

	hsub = fabs(MAX_IONISATION/d_lambda_by_dt);
	if (d_lambda_by_dt > 0.0) {
				// max hsub puts hsub dlambda/dt ntot = 10% n_a = 10% ntot (1-lambda)
				// so hsub dlambda/dt = 10
		hsub = min (hsub, (1.0-lambda)*MAX_PPN_OF_ATOMS_IONISE/d_lambda_by_dt);
	} else {
				// max hsub puts hsub dlambda/dt ntot = 10% n_i = 10% lambda ntot
		hsub = min (hsub, fabs(lambda*MAX_PPN_OF_IONS_RECOMBINE/d_lambda_by_dt));
	};

	if (hsub > h) hsub = h;
	if (hsub < MINIMUM_IONISATION_STEP) hsub = MINIMUM_IONISATION_STEP;
			
	return hsub;
}
void TriMesh::IoniseAndRecombine()
{
	Triangle * pTri;
	long iTri;
	real Te_eV, sqrtTe_eV, Tn_eV, area, ntot, lambda, lambda_new, Te_new,
		S,a_r,a_3,d_lambda_by_dt, d_lambda_by_dt_new, masstot,T_e,T_i,T_n,
		sqrtTe_eVnew;
	Vector3 v_n,v_e,v_i;
	real lambda_low,lambda_high,lambda_test,dbydt;
	real minimum_mass;
	int surplusflag;
	long num_substeps, iSubstep;
	real hsub;

	static real const E0 = 13.6; // eV
	static real const one_over_kB = 1.0/kB;
	static real const THRESHOLD_FOR_BOTHERING = 5.0e4;
	// if it would take 2.0e-7 seconds to change ionisation by 1%, it's not worth bothering with.

	real ionisation_htg = 0.0;

	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		// The first job is to calculate net ionisation rate
		// If it is positive or negative then the bwd solve is different.

		// However, it can also be used to first try an explicit step.

		// 0. Test for n_e = 0, n_n = 0, things like that.

		if ((pTri->elec.mass > 0.0) && (pTri->neut.mass > 0.0) && (pTri->ion.mass > 0.0))
		{

			// Quickly edited for now: repeat some calcs twice initially
			
			if (pTri->ion.mass < pTri->elec.mass)
			{
				minimum_mass = pTri->ion.mass;
				surplusflag = 1; // electrons had the surplus
			} else {
				minimum_mass = pTri->elec.mass;
				surplusflag = 0; // ions had the surplus
			};
			Te_eV = (pTri->elec.heat/pTri->elec.mass)*one_over_kB;
			sqrtTe_eV = sqrt(Te_eV);
			Tn_eV = (pTri->neut.heat/pTri->neut.mass)*one_over_kB;
		
			// current ionisation ppn:
			area = pTri->area;
			ntot = (minimum_mass + pTri->neut.mass)/area;
			lambda = minimum_mass/(minimum_mass + pTri->neut.mass);

			S = 1.0e-5*sqrtTe_eV*exp(-E0/Te_eV)/(E0*(6.0*E0 + Te_eV));
			a_r = 2.7e-13/sqrtTe_eV;
			a_3 = 8.75e-27/(Te_eV*Te_eV*Te_eV*Te_eV*sqrtTe_eV);

			// Find out the net ionisation rate at the new values to determine if the sign has changed.
			d_lambda_by_dt = lambda*ntot*(1.0-lambda)*S - 
											ntot*lambda*lambda*(a_r + lambda*ntot*a_3);
			
			if (
				((d_lambda_by_dt < 0.0) && (minimum_mass < 1.0e11*pTri->area)) 
				|| 
				((d_lambda_by_dt > 0.0) && (pTri->neut.mass < 1.0e9*pTri->area)) 
					// will want to come back and alter that case to go fully ionised - but it probably will cause a few hiccups
				||
				(fabs(d_lambda_by_dt) < THRESHOLD_FOR_BOTHERING)
				)
			{
				// do nothing
			} else {


				// Want to limit the substep so that we change by at most 1% in a substep let's say.
				// and so that the net ionised amount is at most 10% of remaining neutrals
				// or net recombination amount is at most 10% of ions.
				// We should generally then find that we get quite a few explicit steps 
				// (SHOULD be using RK2 not RK1 of course!)
				// before we have to do implicit solves, which is not a bad thing.
				// Respecting the ionisation timescale can be necessary to maintain accuracy of simulation results.

				hsub = fabs(0.01/d_lambda_by_dt);
				if (d_lambda_by_dt > 0.0) {
					// max hsub puts hsub dlambda/dt ntot = 10% n_a = 10% ntot (1-lambda)
					// so hsub dlambda/dt = 10
					hsub = min (hsub, (1.0-lambda)*0.1/d_lambda_by_dt);
				} else {
					// max hsub puts hsub dlambda/dt ntot = 10% n_i = 10% lambda ntot
					hsub = min (hsub, fabs(lambda*0.1/d_lambda_by_dt));
				};

				if (hsub < h) {
					if (hsub < 1.0e-16) hsub = 1.0e-16; // have to have some minimum hsub!
					num_substeps = (int)(h/hsub)+1;
					hsub = h/(real)num_substeps;
				} else {
					hsub = h;
					num_substeps = 1;
				};

				for (iSubstep = 0; iSubstep < num_substeps; iSubstep++)
				{
				// Want to allow that n_e does not exactly equal n_ion.
				// ______________________________
				// INTERIM FIX 19TH JULY :
				// Deal with the less populous of the two species 

					if (pTri->ion.mass < pTri->elec.mass)
					{
						minimum_mass = pTri->ion.mass;
						surplusflag = 1; // electrons had the surplus
					} else {
						minimum_mass = pTri->elec.mass;
						surplusflag = 0; // ions had the surplus
					};
								
					// 1. Explicit step
					// %%%%%%%%%

					// Work out rate:

					// Bear in mind T is in eV

					Te_eV = (pTri->elec.heat/pTri->elec.mass)*one_over_kB;
					sqrtTe_eV = sqrt(Te_eV);
					Tn_eV = (pTri->neut.heat/pTri->neut.mass)*one_over_kB;
					
					// current ionisation ppn:
					area = pTri->area;
					ntot = (minimum_mass + pTri->neut.mass)/area;
					lambda = minimum_mass/(minimum_mass + pTri->neut.mass);
					
					S = 1.0e-5*sqrtTe_eV*exp(-E0/Te_eV)/(E0*(6.0*E0 + Te_eV));
					a_r = 2.7e-13/sqrtTe_eV;
					a_3 = 8.75e-27/(Te_eV*Te_eV*Te_eV*Te_eV*sqrtTe_eV);
					
					// Find out the net ionisation rate at the new values to determine if the sign has changed.
					d_lambda_by_dt = lambda*ntot*(1.0-lambda)*S - 
													ntot*lambda*lambda*(a_r + lambda*ntot*a_3);
					
					// Simple Euler:				
					lambda_new = lambda + hsub*d_lambda_by_dt;	
					
					// 2. Test for overshooting
					// %%%%%%%%%%%%
					
					bool failed = false;
					if ( (lambda_new < 0.0) || (lambda_new > 1.0))
					{
						// failed - go to step 3
						failed = true;
					} else {
						
						if (d_lambda_by_dt > 0.0) {
							// net ionisation:
							Te_new = Te_eV*lambda/lambda_new + 
								(0.5*Tn_eV-E0*2.0/3.0)*((lambda_new-lambda)/lambda_new);
						} else {
							// net recombination:
							Te_new = Te_eV - E0*(2.0/3.0)*((lambda_new-lambda)/lambda_new);
						};		
						if (Te_new < 0.0)
						{
							failed = true;
						} else {
							sqrtTe_eVnew = sqrt(Te_new);
							
							S = 1.0e-5*sqrtTe_eVnew*exp(-E0/Te_new)/(E0*(6.0*E0 + Te_new));
							a_r = 2.7e-13/sqrtTe_eVnew;
							a_3 = 8.75e-27/(Te_new*Te_new*Te_new*Te_new*sqrtTe_eVnew);
							
							d_lambda_by_dt_new = lambda_new*ntot*(1.0-lambda_new)*S - 
															ntot*lambda_new*lambda_new*(a_r + lambda_new*ntot*a_3);
							// MORE FRIENDLY TEST: ALLOW OVERSHOOT OF EQUILIBRIUM
				
							// AS LONG AS MAGNITUDE OF D/DT IS DECREASING ENOUGH.
				
							if ( (d_lambda_by_dt_new*d_lambda_by_dt < 0.0) &&	( fabs(d_lambda_by_dt_new) > 0.5*fabs(d_lambda_by_dt) ))
								failed = true;
						};
					};		

					// If it did not overshoot equilibrium (or go outside (0,1)) then we accept the value
					// Otherwise now go to step 3.

					if (failed)
					{
				//		printf("\n\nlambda = %1.10E   Te_eV = %1.10E   d/dt = %1.10E \n"
				//						 "lambda_new = %1.10E   Te_eV = %1.10E   d/dt = %1.10E\n\n",
				//			lambda,Te_eV,d_lambda_by_dt,
				//			lambda_new,Te_new,d_lambda_by_dt_new);

						// Well, it might be worth a quick go of having a forward step of theta=1/lambda
						// just to see if this saves runtime. NOTE BENE.

						// 3. Implicit first-order step (any value closer to eqm than bwd
						// can be accepted since it is then an lc of fwd and bwd steps)
						// This is seen when we still move same way as initial but bwd soln is seen to be nearer initial
						// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
						
						// Can work with theta = 1/lambda ; this is very like working with T
						

						// Here is another thought.

						// Go between lambda_old and min(1,lambda_new_explicit) and take a selection of values
						// A value is accepted if:
						
						//     * Te > 0 (or lambda_putative is too high)
							// * lambda_putative <= eqm lambda, or >= if falling;
						//				ie, dl/dt has the same sign dl/dt as for lambda_old
							// * lambda_putative >= bwd lambda, or <= if falling;
						//				ie, -h dl/dt will not get us back as far as lambda_old

						// We can weaken the conditions: allow overshooting if | dl/dt | is less?
						// would have to think carefully about what variable to actually use.

						// can bisection method find such a value?

						// for ease of programming, split out two cases:
						if (d_lambda_by_dt > 0.0)
						{

							// Create initial interval:

							lambda_high = min(lambda_new,1.0);
							lambda_low = lambda;

							bool carry_on = true;

							while (carry_on)
							{
								// bisect existing two points
								lambda_test = 0.5*(lambda_low + lambda_high);
								// Take derivative d/dt at this lambda, to test sign:

								if (lambda_high-lambda_test < 1.0e-15) {
									lambda_new = lambda_high;
									carry_on = false;
								} else {
									Te_new = Te_eV*lambda/lambda_test + 
										(0.5*Tn_eV-E0*2.0/3.0)*((lambda_test-lambda)/lambda_test);

									if (Te_new < 0.0)
									{
										// overshooting eqm
										lambda_high = lambda_test;
									} else {

										sqrtTe_eVnew = sqrt(Te_new);
									
										S = 1.0e-5*sqrtTe_eVnew*exp(-E0/Te_new)/(E0*(6.0*E0 + Te_new));
										a_r = 2.7e-13/sqrtTe_eVnew;
										a_3 = 8.75e-27/(Te_new*Te_new*Te_new*Te_new*sqrtTe_eVnew);
									
										dbydt = lambda_test*ntot*(1.0-lambda_test)*S - 
																	ntot*lambda_test*lambda_test*(a_r + lambda_test*ntot*a_3);
									
										if (dbydt * d_lambda_by_dt < 0.0)
										{
											// overshooting eqm
											lambda_high = lambda_test;
										} else {
											// same sign deriv as original lambda
											// now ask if it is past the backward solution:

											if (lambda_test - dbydt*hsub >= lambda+1.0e-15)
											{
												// accept solution: we are between bwd lambda and eqm lambda

												carry_on = false;
												lambda_new = lambda_test;
											} else {
												lambda_low = lambda_test;

											};
										};
									};
								};
							};
						} else {

							// Create initial interval:

							lambda_low = max(lambda_new,0.0);
							lambda_high = lambda;

							bool carry_on = true;

							while (carry_on)
							{
								// bisect existing two points
								lambda_test = 0.5*(lambda_low + lambda_high);
								if (lambda_high-lambda_test < 1.0e-15) {
									lambda_new = lambda_high;
									carry_on = false;
								} else {

									// Take derivative d/dt at this lambda, to test sign:

									Te_new = Te_eV - E0*(2.0/3.0)*((lambda_test-lambda)/lambda_test);

									sqrtTe_eVnew = sqrt(Te_new);
								
									S = 1.0e-5*sqrtTe_eVnew*exp(-E0/Te_new)/(E0*(6.0*E0 + Te_new));
									a_r = 2.7e-13/sqrtTe_eVnew;
									a_3 = 8.75e-27/(Te_new*Te_new*Te_new*Te_new*sqrtTe_eVnew);
								
									dbydt = lambda_test*ntot*(1.0-lambda_test)*S - 
																ntot*lambda_test*lambda_test*(a_r + lambda_test*ntot*a_3);
								
									if (dbydt * d_lambda_by_dt < 0.0)
									{
										// overshooting eqm
										lambda_low = lambda_test;
									} else {
										// same sign deriv as original lambda
										// now ask if it is past the backward solution:

										if (lambda_test - dbydt*hsub <= lambda+1.0e-15) // remember in this case dbydt < 0 so we added a bit
										{
											// accept solution: we are between bwd lambda (approximately) and eqm lambda
											carry_on = false;
											lambda_new = lambda_test;
										} else {
											lambda_high = lambda_test;
										};
									};
								};
							};
						};
						// Nothing can be accepted if it is further from eqm, ie as indicated by the direction of initial d/dt
					};

					// 4. Now do the ionisation variable updates
					// inc. contribution to momentum, Ti, etc:
					// %%%%%%%%%%%%%%%%%%%%%
					
					T_n = pTri->neut.heat/pTri->neut.mass;
					T_i = pTri->ion.heat/pTri->ion.mass;
					T_e = pTri->elec.heat/pTri->elec.mass;

					if (surplusflag == 0) {
						// ion surplus; lambda was based on e's
						masstot = pTri->elec.mass + pTri->neut.mass;
					} else {
						// electron surplus
						masstot = pTri->ion.mass + pTri->neut.mass;
					};

		 			pTri->elec.mass += (lambda_new-lambda)*masstot;				
					pTri->ion.mass += (lambda_new-lambda)*masstot;	
					pTri->neut.mass = (1.0-lambda_new)*masstot;	

					// The following bit always works because (lambdanew-lambda)*masstot is always the number converting (per area)
					if (lambda_new > lambda)
					{
						// net ionisation:

		//			lambda = pTri->elec.mass/(pTri->elec.mass + pTri->neut.mass);

						v_n = pTri->neut.mom/pTri->neut.mass;
						pTri->ion.mom += (lambda_new-lambda)*masstot*v_n; 
						pTri->elec.mom += (lambda_new-lambda)*masstot*v_n;
						pTri->neut.mom -= (lambda_new-lambda)*masstot*v_n;
						// Note that as we adjusted neutral mass, we adjust neutral momentum to keep v_n same.

						pTri->elec.heat += 0.5*(lambda_new-lambda)*masstot*T_n
													- kB*E0*(2.0/3.0)*(lambda_new-lambda)*masstot;

						ionisation_htg += - kB*E0*(2.0/3.0)*(lambda_new-lambda)*masstot;
						
						// Decide if we think this is right
						// Debug:
						// compare Te_new with Te now:
						
						Te_eV = (pTri->elec.heat/pTri->elec.mass)*one_over_kB;
						if ((Te_new > 1.01*Te_eV) || (Te_new < 0.99*Te_eV))
							Te_new = Te_new;
						
						pTri->ion.heat += 0.5*(lambda_new-lambda)*masstot*T_n;
						pTri->neut.heat -= (lambda_new-lambda)*masstot*T_n;
						
					} else {
						// net recombination:
						v_e = pTri->elec.mom/pTri->elec.mass;
						v_i = pTri->ion.mom/pTri->ion.mass;

						pTri->elec.mom += (lambda_new-lambda)*masstot*v_e; // negative coefficient
						pTri->ion.mom += (lambda_new-lambda)*masstot*v_i; // negative coefficient
						// This should leave elec.mom/elec.mass and ion.mom/ion.mass the same
						pTri->neut.mom -= (lambda_new-lambda)*masstot*
														((m_ion/m_neutral)*v_i + (m_e/m_neutral)*v_e);
						
						// May wish to do a debug check that total momentum stayed same.
						
						pTri->ion.heat += (lambda_new-lambda)*masstot*T_i;
						pTri->elec.heat += (lambda_new-lambda)*masstot*T_e
													- kB*E0*(2.0/3.0)*(lambda_new-lambda)*masstot;
						
						ionisation_htg += - kB*E0*(2.0/3.0)*(lambda_new-lambda)*masstot;

						// both negative contributions - take away its own heat that left and take away heat the was taken up for ionisation also.
						
						pTri->neut.heat -= (lambda_new-lambda)*masstot*(T_i + T_e);
							// note: sum of ion and electron temperatures
												  
					};
					
					if (pTri->elec.heat < 0.0)
					{
						pTri = pTri;
					};
				};// next substep

				/*

				BACKUP:
				
				
				// 1. Explicit step
				// %%%%%%%%%

				// Work out rate:

				// Bear in mind T is in eV

				Te_eV = (pTri->elec.heat/pTri->elec.mass)*one_over_kB;
				sqrtTe_eV = sqrt(Te_eV);
				Tn_eV = (pTri->neut.heat/pTri->neut.mass)*one_over_kB;
			
				// current ionisation ppn:
				area = pTri->GetArea();
				ntot = (pTri->elec.mass + pTri->neut.mass)/area;
				lambda = pTri->elec.mass/(pTri->elec.mass + pTri->neut.mass);

				S = 1.0e-5*sqrtTe_eV*exp(-E0/Te_eV)/(E0*(6.0*E0 + Te_eV));
				a_r = 2.7e-13/sqrtTe_eV;
				a_3 = 8.75e-27/(Te_eV*Te_eV*Te_eV*Te_eV*sqrtTe_eV);

				// Find out the net ionisation rate at the new values to determine if the sign has changed.
				d_lambda_by_dt = lambda*ntot*(1.0-lambda)*S - 
												ntot*lambda*lambda*(a_r + lambda*ntot*a_3);
				
				// Simple Euler:

				lambda_new = lambda + h*d_lambda_by_dt;	
				
				// 2. Test for overshooting
				// %%%%%%%%%%%%

				bool failed = false;
				if ( (lambda_new < 0.0) || (lambda_new > 1.0))
				{
					// failed - go to step 3
					failed = true;
				} else {
					
					if (d_lambda_by_dt > 0.0) {
						// net ionisation:
						Te_new = Te_eV*lambda/lambda_new + 
							(0.5*Tn_eV-E0*2.0/3.0)*((lambda_new-lambda)/lambda_new);
					} else {
						// net recombination:
						Te_new = Te_eV - E0*(2.0/3.0)*((lambda_new-lambda)/lambda_new);
					};		
				
					if (Te_new < 0.0)
					{
						failed = true;
					} else {
						sqrtTe_eVnew = sqrt(Te_new);
						
						S = 1.0e-5*sqrtTe_eVnew*exp(-E0/Te_new)/(E0*(6.0*E0 + Te_new));
						a_r = 2.7e-13/sqrtTe_eVnew;
						a_3 = 8.75e-27/(Te_new*Te_new*Te_new*Te_new*sqrtTe_eVnew);
						
						d_lambda_by_dt_new = lambda_new*ntot*(1.0-lambda_new)*S - 
														ntot*lambda_new*lambda_new*(a_r + lambda_new*ntot*a_3);
						
						if (d_lambda_by_dt_new*d_lambda_by_dt < 0.0)
							failed = true;
					};
				};		

				// If it did not overshoot equilibrium (or go outside (0,1)) then we accept the value
				// Otherwise now go to step 3.

				if (failed)
				{
			//		printf("\n\nlambda = %1.10E   Te_eV = %1.10E   d/dt = %1.10E \n"
			//						 "lambda_new = %1.10E   Te_eV = %1.10E   d/dt = %1.10E\n\n",
			//			lambda,Te_eV,d_lambda_by_dt,
			//			lambda_new,Te_new,d_lambda_by_dt_new);

					// Well, it might be worth a quick go of having a forward step of theta=1/lambda
					// just to see if this saves runtime. NOTE BENE.

					// 3. Implicit first-order step (any value closer to eqm than bwd
					// can be accepted since it is then an lc of fwd and bwd steps)
					// This is seen when we still move same way as initial but bwd soln is seen to be nearer initial
					// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
					
					// Can work with theta = 1/lambda ; this is very like working with T
					

					// Here is another thought.

					// Go between lambda_old and min(1,lambda_new_explicit) and take a selection of values
					// A value is accepted if:
					
					//     * Te > 0 (or lambda_putative is too high)
						// * lambda_putative <= eqm lambda, or >= if falling;
					//				ie, dl/dt has the same sign dl/dt as for lambda_old
						// * lambda_putative >= bwd lambda, or <= if falling;
					//				ie, -h dl/dt will not get us back as far as lambda_old

					// We can weaken the conditions: allow overshooting if | dl/dt | is less?
					// would have to think carefully about what variable to actually use.

					// can bisection method find such a value?

					// for ease of programming, split out two cases:
					if (d_lambda_by_dt > 0.0)
					{

						// Create initial interval:

						lambda_high = min(lambda_new,1.0);
						lambda_low = lambda;

						bool carry_on = true;

						while (carry_on)
						{
							// bisect existing two points
							lambda_test = 0.5*(lambda_low + lambda_high);
							// Take derivative d/dt at this lambda, to test sign:

							if (lambda_high-lambda_test < 1.0e-15) {
								lambda_new = lambda_high;
								carry_on = false;
							} else {
								Te_new = Te_eV*lambda/lambda_test + 
									(0.5*Tn_eV-E0*2.0/3.0)*((lambda_test-lambda)/lambda_test);

								if (Te_new < 0.0)
								{
									// overshooting eqm
									lambda_high = lambda_test;
								} else {

									sqrtTe_eVnew = sqrt(Te_new);
								
									S = 1.0e-5*sqrtTe_eVnew*exp(-E0/Te_new)/(E0*(6.0*E0 + Te_new));
									a_r = 2.7e-13/sqrtTe_eVnew;
									a_3 = 8.75e-27/(Te_new*Te_new*Te_new*Te_new*sqrtTe_eVnew);
								
									dbydt = lambda_test*ntot*(1.0-lambda_test)*S - 
																ntot*lambda_test*lambda_test*(a_r + lambda_test*ntot*a_3);
								
									if (dbydt * d_lambda_by_dt < 0.0)
									{
										// overshooting eqm
										lambda_high = lambda_test;
									} else {
										// same sign deriv as original lambda
										// now ask if it is past the backward solution:

										if (lambda_test - dbydt*h >= lambda+1.0e-15)
										{
											// accept solution: we are between bwd lambda and eqm lambda

											carry_on = false;
											lambda_new = lambda_test;
										} else {
											lambda_low = lambda_test;

										};
									};
								};
							};
						};
					} else {

						// Create initial interval:

						lambda_low = max(lambda_new,0.0);
						lambda_high = lambda;

						bool carry_on = true;

						while (carry_on)
						{
							// bisect existing two points
							lambda_test = 0.5*(lambda_low + lambda_high);
							if (lambda_high-lambda_test < 1.0e-15) {
								lambda_new = lambda_high;
								carry_on = false;
							} else {

								// Take derivative d/dt at this lambda, to test sign:

								Te_new = Te_eV - E0*(2.0/3.0)*((lambda_test-lambda)/lambda_test);

								sqrtTe_eVnew = sqrt(Te_new);
							
								S = 1.0e-5*sqrtTe_eVnew*exp(-E0/Te_new)/(E0*(6.0*E0 + Te_new));
								a_r = 2.7e-13/sqrtTe_eVnew;
								a_3 = 8.75e-27/(Te_new*Te_new*Te_new*Te_new*sqrtTe_eVnew);
							
								dbydt = lambda_test*ntot*(1.0-lambda_test)*S - 
															ntot*lambda_test*lambda_test*(a_r + lambda_test*ntot*a_3);
							
								if (dbydt * d_lambda_by_dt < 0.0)
								{
									// overshooting eqm
									lambda_low = lambda_test;
								} else {
									// same sign deriv as original lambda
									// now ask if it is past the backward solution:

									if (lambda_test - dbydt*h <= lambda+1.0e-15) // remember in this case dbydt < 0 so we added a bit
									{
										// accept solution: we are between bwd lambda (approximately) and eqm lambda
										carry_on = false;
										lambda_new = lambda_test;
									} else {
										lambda_high = lambda_test;
									};
								};
							};
						};
					};
					// Nothing can be accepted if it is further from eqm, ie as indicated by the direction of initial d/dt
				};

				// 4. Now do the ionisation variable updates
				// inc. contribution to momentum, Ti, etc:
				// %%%%%%%%%%%%%%%%%%%%%
				
				masstot = pTri->elec.mass + pTri->neut.mass;
				T_n = pTri->neut.heat/pTri->neut.mass;
				T_i = pTri->ion.heat/pTri->ion.mass;
				T_e = pTri->elec.heat/pTri->elec.mass;

		 		pTri->elec.mass = lambda_new*masstot;				// more
				pTri->neut.mass = (1.0-lambda_new)*masstot;	// less
				pTri->ion.mass += (lambda_new-lambda)*masstot;	// more

				if (lambda_new > lambda)
				{
					// net ionisation:

	//			lambda = pTri->elec.mass/(pTri->elec.mass + pTri->neut.mass);

					v_n = pTri->neut.mom/pTri->neut.mass;
					pTri->ion.mom += (lambda_new-lambda)*masstot*v_n; 
					pTri->elec.mom += (lambda_new-lambda)*masstot*v_n;
					pTri->neut.mom -= (lambda_new-lambda)*masstot*v_n;
					// Note that as we adjusted neutral mass, we adjust neutral momentum to keep v_n same.

					pTri->elec.heat += 0.5*(lambda_new-lambda)*masstot*T_n
												- kB*E0*(2.0/3.0)*(lambda_new-lambda)*masstot;
					// Decide if we think this is right
					// Debug:
					// compare Te_new with Te now:
					
					Te_eV = (pTri->elec.heat/pTri->elec.mass)*one_over_kB;
					if ((Te_new > 1.01*Te_eV) || (Te_new < 0.99*Te_eV))
						Te_new = Te_new;

					pTri->ion.heat += 0.5*(lambda_new-lambda)*masstot*T_n;
					pTri->neut.heat -= (lambda_new-lambda)*masstot*T_n;
					
				} else {
					// net recombination:
					v_e = pTri->elec.mom/pTri->elec.mass;
					v_i = pTri->ion.mom/pTri->ion.mass;

					pTri->elec.mom += (lambda_new-lambda)*masstot*v_e; // negative coefficient
					pTri->ion.mom += (lambda_new-lambda)*masstot*v_i; // negative coefficient
					// This should leave elec.mom/elec.mass and ion.mom/ion.mass the same
					pTri->neut.mom -= (lambda_new-lambda)*masstot*
													((m_ion/m_neutral)*v_i + (m_e/m_neutral)*v_e);
					
					// May wish to do a debug check that total momentum stayed same.
					
					pTri->ion.heat += (lambda_new-lambda)*masstot*T_i;
					pTri->elec.heat += (lambda_new-lambda)*masstot*T_e
												- kB*E0*(2.0/3.0)*(lambda_new-lambda)*masstot;
					// both negative contributions - take away its own heat that left and take away heat the was taken up for ionisation also.
					
					pTri->neut.heat -= (lambda_new-lambda)*masstot*(T_i + T_e);
						// note: sum of ion and electron temperatures
											  
				};
				
				if (pTri->elec.heat < 0.0)
				{
					pTri = pTri;
				};
	*/
			}; // whether there existed any ions, neutrals

			// Note that we probably want another speed-up:
			// if Te is high enough and ionisation is near complete, may be best just to jump to
			// full ionisation rather than doing lots of backward solves near 1.
			// Or if low and close to full recombination, ... jump to 0.
			
			// This will cause a whole raft of issues with places that have zero density of a species; - be prepared for them.
			// e.g. sum of heat / sum of mass = 0/0 at a vertex
			// just for starters.
			// Will have to go through whole program.

		}; // whether hardly any ions, trying to disappear, 
		// hardly any neutrals and trying to fully ionise,
		// or d/dt lambda just too small of a rate to bother, etc

		++pTri;
	};

	fp = fopen(FUNCTIONALFILENAME, "a");
	fprintf(fp," ionise %1.13E ",ionisation_htg);
	fclose(fp);
}

void TriMesh::IonisationSubSubcycle(Triangle * pTri, real hsub, long num_substeps)
{
	long iTri;
	real Te_eV, sqrtTe_eV, Tn_eV, area, ntot, lambda, lambda_new, Te_new,
		S,a_r,a_3,d_lambda_by_dt, d_lambda_by_dt_new, masstot,T_e,T_i,T_n,
		sqrtTe_eVnew;
	Vector3 v_n,v_e,v_i;
	real lambda_low,lambda_high,lambda_test,dbydt;
	real minimum_mass;
	int surplusflag;
	long iSubstep;
	
	static real const E0 = 13.6; // eV
	static real const one_over_kB = 1.0/kB;
	
	if ((pTri->elec.mass <= 1.0e9*pTri->area) || (pTri->neut.mass <= 1.0e9*pTri->area) || (pTri->ion.mass <= 1.0e9*pTri->area)) return;
	// if n < 1e9 then don't bother!
	
	for (long iSubstep = 0; iSubstep < num_substeps; iSubstep++)
	{
		// Want to allow that n_e does not exactly equal n_ion.
		// ______________________________
		// INTERIM FIX 19TH JULY :
		// Deal with the less populous of the two species 

		if (pTri->ion.mass < pTri->elec.mass)
		{
			minimum_mass = pTri->ion.mass;
			surplusflag = 1; // electrons had the surplus
		} else {
			minimum_mass = pTri->elec.mass;
			surplusflag = 0; // ions had the surplus
		};
					
		// 1. Explicit step
		// %%%%%%%%%

		// Work out rate:

		// Bear in mind T is in eV

		Te_eV = (pTri->elec.heat/pTri->elec.mass)*one_over_kB;
		sqrtTe_eV = sqrt(Te_eV);
		Tn_eV = (pTri->neut.heat/pTri->neut.mass)*one_over_kB;
	
		// current ionisation ppn:
		area = pTri->area;
		ntot = (minimum_mass + pTri->neut.mass)/area;
		lambda = minimum_mass/(minimum_mass + pTri->neut.mass);

		S = 1.0e-5*sqrtTe_eV*exp(-E0/Te_eV)/(E0*(6.0*E0 + Te_eV));
		a_r = 2.7e-13/sqrtTe_eV;
		a_3 = 8.75e-27/(Te_eV*Te_eV*Te_eV*Te_eV*sqrtTe_eV);

		// Find out the net ionisation rate at the new values to determine if the sign has changed.
		d_lambda_by_dt = lambda*ntot*(1.0-lambda)*S - 
										ntot*lambda*lambda*(a_r + lambda*ntot*a_3);
		
		// Simple Euler:

		lambda_new = lambda + hsub*d_lambda_by_dt;	
		
		// 2. Test for overshooting
		// %%%%%%%%%%%%

		bool failed = false;
		if ( (lambda_new < 0.0) || (lambda_new > 1.0))
		{
			// failed - go to step 3
			failed = true;
		} else {
			
			if (d_lambda_by_dt > 0.0) {
				// net ionisation:
				Te_new = Te_eV*lambda/lambda_new + 
					(0.5*Tn_eV-E0*2.0/3.0)*((lambda_new-lambda)/lambda_new);
			} else {
				// net recombination:
				Te_new = Te_eV - E0*(2.0/3.0)*((lambda_new-lambda)/lambda_new);
			};		
		
			if (Te_new < 0.0)
			{
				failed = true;
			} else {
				sqrtTe_eVnew = sqrt(Te_new);
				
				S = 1.0e-5*sqrtTe_eVnew*exp(-E0/Te_new)/(E0*(6.0*E0 + Te_new));
				a_r = 2.7e-13/sqrtTe_eVnew;
				a_3 = 8.75e-27/(Te_new*Te_new*Te_new*Te_new*sqrtTe_eVnew);
				
				d_lambda_by_dt_new = lambda_new*ntot*(1.0-lambda_new)*S - 
												ntot*lambda_new*lambda_new*(a_r + lambda_new*ntot*a_3);
				
				// WOULD PREFER MORE FRIENDLY TEST: ALLOW OVERSHOOT OF EQUILIBRIUM
				// AS LONG AS MAGNITUDE OF D/DT IS DECREASING ENOUGH.
				
				if (		(d_lambda_by_dt_new*d_lambda_by_dt < 0.0) 
					&&	( fabs(d_lambda_by_dt_new) > 0.5*fabs(d_lambda_by_dt) ))
					failed = true;
			};
		};		

		// If it did not overshoot equilibrium (or go outside (0,1)) then we accept the value
		// Otherwise now go to step 3.

		if (failed)
		{
	//		printf("\n\nlambda = %1.10E   Te_eV = %1.10E   d/dt = %1.10E \n"
	//						 "lambda_new = %1.10E   Te_eV = %1.10E   d/dt = %1.10E\n\n",
	//			lambda,Te_eV,d_lambda_by_dt,
	//			lambda_new,Te_new,d_lambda_by_dt_new);

			// Well, it might be worth a quick go of having a forward step of theta=1/lambda
			// just to see if this saves runtime. NOTE BENE.

			// 3. Implicit first-order step (any value closer to eqm than bwd
			// can be accepted since it is then an lc of fwd and bwd steps)
			// This is seen when we still move same way as initial but bwd soln is seen to be nearer initial
			// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			
			// Can work with theta = 1/lambda ; this is very like working with T
			

			// Here is another thought.

			// Go between lambda_old and min(1,lambda_new_explicit) and take a selection of values
			// A value is accepted if:
			
			//     * Te > 0 (or lambda_putative is too high)
				// * lambda_putative <= eqm lambda, or >= if falling;
			//				ie, dl/dt has the same sign dl/dt as for lambda_old
				// * lambda_putative >= bwd lambda, or <= if falling;
			//				ie, -h dl/dt will not get us back as far as lambda_old

			// We can weaken the conditions: allow overshooting if | dl/dt | is less?
			// would have to think carefully about what variable to actually use.

			// can bisection method find such a value?

			// for ease of programming, split out two cases:
			if (d_lambda_by_dt > 0.0)
			{

				// Create initial interval:

				lambda_high = min(lambda_new,1.0);
				lambda_low = lambda;

				bool carry_on = true;

				while (carry_on)
				{
					// bisect existing two points
					lambda_test = 0.5*(lambda_low + lambda_high);
					// Take derivative d/dt at this lambda, to test sign:

					if (lambda_high-lambda_test < 1.0e-15) {
						lambda_new = lambda_high;
						carry_on = false;
					} else {
						Te_new = Te_eV*lambda/lambda_test + 
							(0.5*Tn_eV-E0*2.0/3.0)*((lambda_test-lambda)/lambda_test);

						if (Te_new < 0.0)
						{
							// overshooting eqm
							lambda_high = lambda_test;
						} else {

							sqrtTe_eVnew = sqrt(Te_new);
						
							S = 1.0e-5*sqrtTe_eVnew*exp(-E0/Te_new)/(E0*(6.0*E0 + Te_new));
							a_r = 2.7e-13/sqrtTe_eVnew;
							a_3 = 8.75e-27/(Te_new*Te_new*Te_new*Te_new*sqrtTe_eVnew);
						
							dbydt = lambda_test*ntot*(1.0-lambda_test)*S - 
														ntot*lambda_test*lambda_test*(a_r + lambda_test*ntot*a_3);
						
							if (dbydt * d_lambda_by_dt < 0.0)
							{
								// overshooting eqm
								lambda_high = lambda_test;
							} else {
								// same sign deriv as original lambda
								// now ask if it is past the backward solution:

								if (lambda_test - dbydt*hsub >= lambda+1.0e-15)
								{
									// accept solution: we are between bwd lambda and eqm lambda

									carry_on = false;
									lambda_new = lambda_test;
								} else {
									lambda_low = lambda_test;

								};
							};
						};
					};
				};
			} else {

				// Create initial interval:

				lambda_low = max(lambda_new,0.0);
				lambda_high = lambda;

				bool carry_on = true;

				while (carry_on)
				{
					// bisect existing two points
					lambda_test = 0.5*(lambda_low + lambda_high);
					if (lambda_high-lambda_test < 1.0e-15) {
						lambda_new = lambda_high;
						carry_on = false;
					} else {

						// Take derivative d/dt at this lambda, to test sign:

						Te_new = Te_eV - E0*(2.0/3.0)*((lambda_test-lambda)/lambda_test);

						sqrtTe_eVnew = sqrt(Te_new);
					
						S = 1.0e-5*sqrtTe_eVnew*exp(-E0/Te_new)/(E0*(6.0*E0 + Te_new));
						a_r = 2.7e-13/sqrtTe_eVnew;
						a_3 = 8.75e-27/(Te_new*Te_new*Te_new*Te_new*sqrtTe_eVnew);
					
						dbydt = lambda_test*ntot*(1.0-lambda_test)*S - 
													ntot*lambda_test*lambda_test*(a_r + lambda_test*ntot*a_3);
					
						if (dbydt * d_lambda_by_dt < 0.0)
						{
							// overshooting eqm
							lambda_low = lambda_test;
						} else {
							// same sign deriv as original lambda
							// now ask if it is past the backward solution:

							if (lambda_test - dbydt*hsub <= lambda+1.0e-15) // remember in this case dbydt < 0 so we added a bit
							{
								// accept solution: we are between bwd lambda (approximately) and eqm lambda
								carry_on = false;
								lambda_new = lambda_test;
							} else {
								lambda_high = lambda_test;
							};
						};
					};
				};
			};
			// Nothing can be accepted if it is further from eqm, ie as indicated by the direction of initial d/dt
		};

		// 4. Now do the ionisation variable updates
		// inc. contribution to momentum, Ti, etc:
		// %%%%%%%%%%%%%%%%%%%%%
		
		T_n = pTri->neut.heat/pTri->neut.mass;
		T_i = pTri->ion.heat/pTri->ion.mass;
		T_e = pTri->elec.heat/pTri->elec.mass;

		if (surplusflag == 0) {
			// ion surplus; lambda was based on e's
			masstot = pTri->elec.mass + pTri->neut.mass;
		} else {
			// electron surplus
			masstot = pTri->ion.mass + pTri->neut.mass;
		};

 		pTri->elec.mass += (lambda_new-lambda)*masstot;				
		pTri->ion.mass += (lambda_new-lambda)*masstot;	
		pTri->neut.mass = (1.0-lambda_new)*masstot;	

		// The following bit always works because (lambdanew-lambda)*masstot is always the number converting (per area)
		if (lambda_new > lambda)
		{
			// net ionisation:

//			lambda = pTri->elec.mass/(pTri->elec.mass + pTri->neut.mass);

			v_n = pTri->neut.mom/pTri->neut.mass;
			pTri->ion.mom += (lambda_new-lambda)*masstot*v_n; 
			pTri->elec.mom += (lambda_new-lambda)*masstot*v_n;
			pTri->neut.mom -= (lambda_new-lambda)*masstot*v_n;
			// Note that as we adjusted neutral mass, we adjust neutral momentum to keep v_n same.

			pTri->elec.heat += 0.5*(lambda_new-lambda)*masstot*T_n
										- kB*E0*(2.0/3.0)*(lambda_new-lambda)*masstot;
			
			GlobalIonisationHtg += - kB*E0*(2.0/3.0)*(lambda_new-lambda)*masstot;
			// Decide if we think this is right
			// Debug:
			// compare Te_new with Te now:
			
			Te_eV = (pTri->elec.heat/pTri->elec.mass)*one_over_kB;
			if ((Te_new > 1.01*Te_eV) || (Te_new < 0.99*Te_eV))
				Te_new = Te_new;

			pTri->ion.heat += 0.5*(lambda_new-lambda)*masstot*T_n;
			pTri->neut.heat -= (lambda_new-lambda)*masstot*T_n;
			
		} else {
			// net recombination:
			v_e = pTri->elec.mom/pTri->elec.mass;
			v_i = pTri->ion.mom/pTri->ion.mass;

			pTri->elec.mom += (lambda_new-lambda)*masstot*v_e; // negative coefficient
			pTri->ion.mom += (lambda_new-lambda)*masstot*v_i; // negative coefficient
			// This should leave elec.mom/elec.mass and ion.mom/ion.mass the same
			pTri->neut.mom -= (lambda_new-lambda)*masstot*
											((m_ion/m_neutral)*v_i + (m_e/m_neutral)*v_e);
			
			// May wish to do a debug check that total momentum stayed same.
			
			pTri->ion.heat += (lambda_new-lambda)*masstot*T_i;
			pTri->elec.heat += (lambda_new-lambda)*masstot*T_e
										- kB*E0*(2.0/3.0)*(lambda_new-lambda)*masstot;
			// both negative contributions - take away its own heat that left and take away heat the was taken up for ionisation also.
			GlobalIonisationHtg += - kB*E0*(2.0/3.0)*(lambda_new-lambda)*masstot;
			
			pTri->neut.heat -= (lambda_new-lambda)*masstot*(T_i + T_e);
				// note: sum of ion and electron temperatures
									  
		};
		
		if (pTri->elec.heat < 0.0)
		{
			pTri = pTri;
		};
	};// next substep

}

void TriMesh::EstimateInitialSigmazz(void)
{
	long iTri;
	Triangle *pTri;
	real area, n_ion, n_e, n_n, T_ion, T_n, T_e, sqrt_Te, ion_neut_thermal, electron_thermal,
		lnLambda, sigma_in_MT, sigma_en_MT, sigma_in_visc, sigma_en_visc, nu_en, nu_ei, nu_in;
	real sigma_e_zz;
	static real const TWOTHIRDSqsq = 2.0*q*q/3.0;
	static real const one_over_kB = 1.0/kB; // multiply by this to convert to eV
	static real const one_over_kB_cubed = 1.0/(kB*kB*kB); // multiply by this to convert to eV
	static real const kB_to_3halves = sqrt(kB)*kB;
	static real const over_sqrt_m_ion = 1.0/sqrt(m_ion);
	static real const over_sqrt_m_e = 1.0/sqrt(m_e);
	static real const qoverMc = q/(m_ion*c);
	static real const qovermc = q/(m_e*c);

	real nu_ni, nu_ne, nu_plusbun, denom, numer;

	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		area = pTri->area;
		n_ion = pTri->ion.mass/area;
		n_e = pTri->elec.mass/area;
		n_n = pTri->neut.mass/area;
		T_ion = pTri->ion.heat/pTri->ion.mass;   // may be undefined
		T_n = pTri->neut.heat/pTri->neut.mass;
		T_e = pTri->elec.heat/pTri->elec.mass;
		sqrt_Te = sqrt(T_e);
		ion_neut_thermal = sqrt(T_ion/m_ion+T_n/m_n);
		electron_thermal = sqrt_Te*over_sqrt_m_e;
		
		lnLambda = Get_lnLambda(n_ion,T_e);
		Estimate_Ion_Neutral_Cross_sections(T_ion*one_over_kB, &sigma_in_MT, &sigma_in_visc);
		Estimate_Ion_Neutral_Cross_sections(T_e*one_over_kB,&sigma_en_MT,&sigma_en_visc);
		
		nu_en = n_n*sigma_en_MT*electron_thermal;
		nu_in = n_n*sigma_in_MT*ion_neut_thermal;
		nu_ni = n_ion*sigma_in_MT*ion_neut_thermal;
		nu_ne = n_e*sigma_en_MT*electron_thermal;
		nu_ei = n_ion*NU_EI_FACTOR*kB_to_3halves*lnLambda/(T_e*sqrt_Te);
		
		nu_plusbun = m_i*m_n*nu_ni*nu_en/(m_e*(m_i+m_n)*nu_ne+m_i*(m_e+m_n)*nu_ni)
					+ nu_ei - 0.9*nu_ei/(1.87*nu_ei+nu_en);

		ZeroMemory(&pTri->sigma_erel,sizeof(Tensor3));
		ZeroMemory(&pTri->sigma_i,sizeof(Tensor3));

		// Note minus! erel is for ve - vi

		pTri->sigma_erel.zz = -(q/(m_e*nu_plusbun));
		pTri->sigma_erel.xx = pTri->sigma_erel.zz;
		pTri->sigma_erel.yy = pTri->sigma_erel.zz; // don't know why, don't care

		pTri->vrel_e_0.x = 0.0;
		pTri->vrel_e_0.y = 0.0;
		pTri->vrel_e_0.z = 0.0;
		pTri->vion_0.x = 0.0;
		pTri->vion_0.y = 0.0;
		pTri->vion_0.z = 0.0;

		// That is the information to go into the Az solve.
		// Then we also want to then set v_ion.

		denom = n_e*m_e + (n_n*m_n*m_e*(m_i+m_n)*nu_ne)/(m_e*(m_i+m_n)*nu_ne+m_i*(m_e+m_n)*nu_ni);
		numer = denom + n_ion*m_ion + (n_n*m_n*m_i*(m_e+m_n)*nu_ni)/
			(m_e*(m_i+m_n)*nu_ne + m_i*(m_e+m_n)*nu_ni);

		pTri->sigma_i.zz = (denom/numer)*(q/(m_e*nu_plusbun));
		pTri->sigma_i.xx = pTri->sigma_i.zz;
		pTri->sigma_i.yy = pTri->sigma_i.yy;
		
		// neutral vz = (lambda) vez + (1-lambda) viz

		pTri->GradTe.z = m_e*(m_i+m_n)*nu_ne/(m_e*(m_i+m_n)*nu_ne + m_i*(m_e+m_n)*nu_ni);
		
		if (pTri->GradTe.z > 0.1) {
			// summat dodgy
			pTri = pTri;
		};
			
		//// Where did I work this out?
		//pTri->sigma_i.zz= q/(
		//					nu_in*m_n*m_i/(m_i+m_n)
		//				+ (nu_ei*nu_in/nu_en)*(m_i*m_i*(m_e+m_n))/((m_i+m_n)*(m_e+m_i))
		//				+ nu_ei*m_e*m_i/(m_i+m_e)
		//					); // for v_ion
		//
		//
		//sigma_e_zz = -q/(
		//					nu_en*m_n*m_e/(m_e+m_n)
		//				+ (nu_ei*nu_en/nu_in)*(m_e*m_e*(m_i+m_n))/((m_i+m_e)*(m_e+m_n))
		//				+ nu_ei*m_e*m_i/(m_i+m_e)
		//					); // for v_n

		//pTri->sigma_erel.zz = sigma_e_zz - pTri->sigma_i.zz; 
		
		//dIzbydEz += q*q*pTri->elec.mass/
		//	(
		//	(m_i*m_e/(m_i+m_e))*nu_ei +
		//	(m_i*m_e*m_n/(nu_in*m_i*(m_e+m_n) + nu_en*m_e*(m_i+m_n)))*nu_en*nu_in 
		//	);
		// q / () is the rate of change of relative velocity
		


		++pTri;
	};
	
}

void TriMesh::EstimateInitialEandJz(void)
{
	long iTri;
	Triangle *pTri;
	real area, n_ion, n_e, n_n, T_ion, T_n, T_e, sqrt_Te, ion_thermal, electron_thermal,
		lnLambda, sigma_in_MT, sigma_en_MT, sigma_in_visc, sigma_en_visc, nu_en, nu_ei, nu_in;
	static real const TWOTHIRDSqsq = 2.0*q*q/3.0;
	static real const one_over_kB = 1.0/kB; // multiply by this to convert to eV
	static real const one_over_kB_cubed = 1.0/(kB*kB*kB); // multiply by this to convert to eV
	static real const kB_to_3halves = sqrt(kB)*kB;
	static real const over_sqrt_m_ion = 1.0/sqrt(m_ion);
	static real const over_sqrt_m_e = 1.0/sqrt(m_e);
	static real const qoverMc = q/(m_ion*c);
	static real const qovermc = q/(m_e*c);

	// 1. Work out d Iz / dEz for constant profile Ez:

	real dIzbydEz = 0.0;

	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		area = pTri->area;
		n_ion = pTri->ion.mass/area;
		n_e = pTri->elec.mass/area;
		n_n = pTri->neut.mass/area;
		T_ion = pTri->ion.heat/pTri->ion.mass;   // may be undefined
		T_n = pTri->neut.heat/pTri->neut.mass;
		T_e = pTri->elec.heat/pTri->elec.mass;
		sqrt_Te = sqrt(T_e);
		ion_thermal = sqrt(T_ion/m_ion);
		electron_thermal = sqrt_Te*over_sqrt_m_e;
		
		lnLambda = Get_lnLambda(n_ion,T_e);
		Estimate_Ion_Neutral_Cross_sections(T_ion*one_over_kB, &sigma_in_MT, &sigma_in_visc);
		Estimate_Ion_Neutral_Cross_sections(T_e*one_over_kB,&sigma_en_MT,&sigma_en_visc);
		
		nu_en = n_n*sigma_en_MT*electron_thermal;
		nu_in = n_n*sigma_en_MT*ion_thermal;
		nu_ei = n_ion*NU_EI_FACTOR*kB_to_3halves*lnLambda/(T_e*sqrt_Te);
		
		dIzbydEz += q*q*pTri->elec.mass/
			(
			(m_i*m_e/(m_i+m_e))*nu_ei +
			(m_i*m_e*m_n/(nu_in*m_i*(m_e+m_n) + nu_en*m_e*(m_i+m_n)))*nu_en*nu_in 
			);
		// q / () is the rate of change of relative velocity
		
		++pTri;
	};
	// 2. Calculate Ez for the prescribed current

	real Iz_prescribed = PEAKCURRENT_STATCOULOMB * sin ((evaltime + ZCURRENTBASETIME) * 0.5*PIOVERPEAKTIME );
	real Ez = Iz_prescribed/dIzbydEz; // ASSUME existing velocity is zero coming into this routine.

	// 3. Infer v_i, v_e 

	real v_e, v_ion;
	real Iz = 0.0;
	Triangle * pTrimax = T+numTriangles;
	for (pTri = T; pTri < pTrimax; ++pTri)
	{
		pTri->E.z = Ez;

		area = pTri->area;
		n_ion = pTri->ion.mass/area;
		n_e = pTri->elec.mass/area;
		n_n = pTri->neut.mass/area;
		T_ion = pTri->ion.heat/pTri->ion.mass;   // may be undefined
		T_n = pTri->neut.heat/pTri->neut.mass;
		T_e = pTri->elec.heat/pTri->elec.mass;
		sqrt_Te = sqrt(T_e);
		ion_thermal = sqrt(T_ion/m_ion);
		electron_thermal = sqrt_Te*over_sqrt_m_e;
		
		lnLambda = Get_lnLambda(n_ion,T_e);
		Estimate_Ion_Neutral_Cross_sections(T_ion*one_over_kB, &sigma_in_MT, &sigma_in_visc);
		Estimate_Ion_Neutral_Cross_sections(T_e*one_over_kB,&sigma_en_MT,&sigma_en_visc);
		
		nu_en = n_n*sigma_en_MT*electron_thermal;
		nu_in = n_n*sigma_en_MT*ion_thermal;
		nu_ei = n_ion*NU_EI_FACTOR*kB_to_3halves*lnLambda/(T_e*sqrt_Te);
		
		v_ion = q*Ez/(
							nu_in*m_n*m_i/(m_i+m_n)
						+ (nu_ei*nu_in/nu_en)*(m_i*m_i*(m_e+m_n))/((m_i+m_n)*(m_e+m_i))
						+ nu_ei*m_e*m_i/(m_i+m_e)
						);
		
		v_e = -q*Ez/(
							nu_en*m_n*m_e/(m_e+m_n)
						+ (nu_ei*nu_en/nu_in)*(m_e*m_e*(m_i+m_n))/((m_i+m_e)*(m_e+m_n))
						+ nu_ei*m_e*m_i/(m_i+m_e)
						);
		
		// alter velocity :
		pTri->ion.mom.z = pTri->ion.mass*v_ion;
		pTri->elec.mom.z = pTri->elec.mass*v_e;
				
		Iz += q*(pTri->ion.mom.z-pTri->elec.mom.z);
	};
	
	printf("Prescribed Iz: %1.14E  Iz: %1.14E  Ez: %1.14E \n",
		Iz_prescribed, Iz, Ez);
}

void TriMesh::RefreshHasPeriodic()
{
	Vertex * pVertex;
	long iVertex, i;
	Triangle *pTri;

	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		pVertex->has_periodic = 0;
		for (i = 0; i < pVertex->triangles.len; i++)
		{
			pTri = (Triangle *)(pVertex->triangles.ptr[i]);
			if (pTri->periodic > 0) pVertex->has_periodic = 1;
		};
		++pVertex;
	};
	// This only looked at neighbours within the domain.
}

void TriMesh::RefreshHasPeriodic_Inner()
{
	SlimVertex * pVertex;
	long iVertex, i;
	SlimTriangle *pTri;

	pVertex = InnerX;
	for (iVertex = 0; iVertex < numInnerVertices; iVertex++)
	{
		pVertex->has_periodic = 0;
		for (i = 0; i < pVertex->tri_len; i++)
		{
			pTri = InnerT + pVertex->iTriangles[i];
			if (pTri->periodic > 0) pVertex->has_periodic = 1;
		};
		++pVertex;
	};
	// This only looked at neighbours within the inner mesh.
	// Joining the two meshes we have to set more has_periodic == 1.
}

void TriMesh::RefreshHasPeriodicAux(int iLevel)
{
	SlimVertex * pVertex;
	long iVertex, i;
	SlimTriangle *pTri;

	pVertex = AuxX[iLevel];
	for (iVertex = 0; iVertex < numAuxVertices[iLevel]; iVertex++)
	{
		pVertex->has_periodic = 0;
		for (i = 0; i < pVertex->tri_len; i++)
		{
			pTri = AuxT[iLevel] + pVertex->iTriangles[i];
			if (pTri->periodic > 0) pVertex->has_periodic = 1;
		};
		++pVertex;
	};
	// This only looked at neighbours within the inner mesh.
	// Joining the two meshes we have to set more has_periodic == 1.
}

void TriMesh::Solve_A_phi_and_J(TriMesh * pMesh_with_A_k)
{
	Triangle * pTri;
	Triangle * pNeigh;

	long iTri, i, j;
	Vertex * pVertex;
	long iVertex;
	int iEdge;
	Vector3 A_k, A_k_neigh;
	Tensor3 Effect_contiguous, sigma_use;
	int wnum[3];
	real LapCoeff[7]; // scalar effect on Laplacian of scalar
	Vector2 gradcoeff[3]; // scalar effect on grad of scalar
	real shoelace, n_e, n_i;
	Vector2 u[3];


	// Solve for A, phi, v_e simultaneously


	// This is where we shall set up the ODE coefficient arrays based on the tensors passed
	// out of COMPUTE_SIGMA and the known coefficients for electron viscous acceleration
	
#define PHI  0
#define GAUSS 0
#define AX   1
#define AY   2
#define AZ   3
#define AMPX  1
#define AMPY  2
#define AMPZ  3
#define VX   4
#define VY   5
#define VZ   6
#define OHMX  4
#define OHMY  5
#define OHMZ  6
#define UNITY 7

	// epsilon_Ampere depends on :
	// A (component effect) in indirect neighbours
	//  [but PB will make it a 2x2 x-y effect at least]
	//  & MM will have to recognise as such, so... do 2x2

	// phi at corners
	// v_e (3x3 effect) here
	// and a constant


	// epsilon_Gauss depends on :
	// A (x,y,z) here and direct neighbours
	// phi at corners of direct neighbours
	// v_e (x,y,z) here and direct neighbours
	// and a constant


	// epsilon_Ohm depends on :
	// A (3x3 effect) here 
	// phi at corners
	// v_e (3x3 effect) in indirect neighbours
	// and a constant


	printf("Setting up ODE coefficients.\n");

	// Dimension:

	long len;
	
	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++) 
	{	
		// Every triangle? No, some are within domain.
		// We should recognise this and only want Ampere for these.
		// In fact our Ampere needs to include outer current.

		len = pTri->indexlist.len;

		pTri->coefficients[GAUSS][PHI] = new real[6]; // here and direct neighbours corners
		pTri->coefficients[GAUSS][AX] = new real[4];
		pTri->coefficients[GAUSS][AY] = new real[4];
		pTri->coefficients[GAUSS][AZ] = new real[4];
		pTri->coefficients[GAUSS][VX] = new real[4];
		pTri->coefficients[GAUSS][VY] = new real[4];
		pTri->coefficients[GAUSS][VZ] = new real[4];
		pTri->coefficients[GAUSS][UNITY] = new real[1];
		pTri->coefficients[GAUSS][EZEXT] = new real[1];
		// None of those truly need to be redimensioned every time!

		pTri->coefficients[AMPX][PHI] = new real[3];		
		pTri->coefficients[AMPX][AX] = new real[len];
		pTri->coefficients[AMPX][AY] = new real[len];
		pTri->coefficients[AMPX][AZ] = new real[1]; // not used
		
		ZeroMemory(pTri->coefficients[AMPX][AX],len*sizeof(real));
		ZeroMemory(pTri->coefficients[AMPX][AY],len*sizeof(real));

		pTri->coefficients[AMPX][VX] = new real[1];
		pTri->coefficients[AMPX][VY] = new real[1];
		pTri->coefficients[AMPX][VZ] = new real[1];
		pTri->coefficients[AMPX][UNITY] = new real[1];
		pTri->coefficients[AMPX][EZEXT] = new real[1];

		pTri->coefficients[AMPY][PHI] = new real[3];
		pTri->coefficients[AMPY][AX] = new real[len];
		pTri->coefficients[AMPY][AY] = new real[len];
		pTri->coefficients[AMPY][AZ] = new real[1]; // not used

		ZeroMemory(pTri->coefficients[AMPY][AX],len*sizeof(real));
		ZeroMemory(pTri->coefficients[AMPY][AY],len*sizeof(real));

		pTri->coefficients[AMPY][VX] = new real[1];
		pTri->coefficients[AMPY][VY] = new real[1];
		pTri->coefficients[AMPY][VZ] = new real[1];
		pTri->coefficients[AMPY][UNITY] = new real[1];
		pTri->coefficients[AMPY][EZEXT] = new real[1];

		pTri->coefficients[AMPZ][PHI] = new real[3];
		pTri->coefficients[AMPZ][AX] = new real[1]; // not used
		pTri->coefficients[AMPZ][AY] = new real[1];
		pTri->coefficients[AMPZ][AZ] = new real[len]; 

		ZeroMemory(pTri->coefficients[AMPZ][AZ],len*sizeof(real));

		pTri->coefficients[AMPZ][VX] = new real[1];
		pTri->coefficients[AMPZ][VY] = new real[1];
		pTri->coefficients[AMPZ][VZ] = new real[1];
		pTri->coefficients[AMPZ][UNITY] = new real[1];
		pTri->coefficients[AMPZ][EZEXT] = new real[1];

		pTri->coefficients[OHMX][PHI] = new real[3];
		pTri->coefficients[OHMX][AX] = new real[1];
		pTri->coefficients[OHMX][AY] = new real[1];
		pTri->coefficients[OHMX][AZ] = new real[1];
		pTri->coefficients[OHMX][VX] = new real[len];
		pTri->coefficients[OHMX][VY] = new real[len];
		pTri->coefficients[OHMX][VZ] = new real[len];

		ZeroMemory(pTri->coefficients[OHMX][VX],len*sizeof(real));
		ZeroMemory(pTri->coefficients[OHMX][VY],len*sizeof(real));
		ZeroMemory(pTri->coefficients[OHMX][VZ],len*sizeof(real));

		pTri->coefficients[OHMX][UNITY] = new real[1];
		pTri->coefficients[OHMX][EZEXT] = new real[1];

		pTri->coefficients[OHMY][PHI] = new real[3];
		pTri->coefficients[OHMY][AX] = new real[1];
		pTri->coefficients[OHMY][AY] = new real[1];
		pTri->coefficients[OHMY][AZ] = new real[1];
		pTri->coefficients[OHMY][VX] = new real[len];
		pTri->coefficients[OHMY][VY] = new real[len];
		pTri->coefficients[OHMY][VZ] = new real[len];
				
		ZeroMemory(pTri->coefficients[OHMY][VX],len*sizeof(real));
		ZeroMemory(pTri->coefficients[OHMY][VY],len*sizeof(real));
		ZeroMemory(pTri->coefficients[OHMY][VZ],len*sizeof(real));

		pTri->coefficients[OHMY][UNITY] = new real[1];
		pTri->coefficients[OHMY][EZEXT] = new real[1];

		pTri->coefficients[OHMZ][PHI] = new real[3];
		pTri->coefficients[OHMZ][AX] = new real[1];
		pTri->coefficients[OHMZ][AY] = new real[1];
		pTri->coefficients[OHMZ][AZ] = new real[1];
		pTri->coefficients[OHMZ][VX] = new real[len];
		pTri->coefficients[OHMZ][VY] = new real[len];
		pTri->coefficients[OHMZ][VZ] = new real[len];

		ZeroMemory(pTri->coefficients[OHMZ][VX],len*sizeof(real));
		ZeroMemory(pTri->coefficients[OHMZ][VY],len*sizeof(real));
		ZeroMemory(pTri->coefficients[OHMZ][VZ],len*sizeof(real));

		pTri->coefficients[OHMZ][UNITY] = new real[1];
		pTri->coefficients[OHMZ][EZEXT] = new real[1];

		// Maybe a coefficients object would be able to have a method to handle getting epsilon.
		// Not sure if that is viable or not for performance!
		++pTri;
	}
	
	// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

	// Sensibly need to be setting ALL to zero to initialise.

	// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



		
	// Reasonably -- once it runs -- can use realloc instead of new/delete.
	// In addition to not playing with ones that are fine and always same dimension.

	// Calculate coefficient values:
	// This is the big job.


	// Let's start with Ampere. 
	
	// Coefficients on phi :
	// (Note that going via E to calculate epsilon makes a whole lot more sense.)

	n_e = pTri->elec.mass/pTri->area;

	n_i = pTri->ion.mass/pTri->area;

	wnum[0] = WindingNumber(pTri->neighbours[0],pTri); 
	wnum[1] = WindingNumber(pTri->neighbours[1],pTri); 
	wnum[2] = WindingNumber(pTri->neighbours[2],pTri); 

		// 0: contiguous
		// 1: data from first argument has to be rotated right to apply to second arg
		// -1: data from first argument has to be rotated left to apply to second arg

	// create vectors : gradphicoeff[0].x = effect of phi[0] on gradphi.x

	if (pTri->periodic == 0) {
		pTri->PopulatePositions(u[0],u[1],u[2]);
	} else {
		pTri->MapLeft(u[0],u[1],u[2]);
	};
	shoelace = u[0].x*(u[1].y-u[2].y)
			 + u[1].x*(u[2].y-u[0].y)
			 + u[2].x*(u[0].y-u[1].y);

	gradcoeff[0].x = (u[1].y-u[2].y)/shoelace;
	gradcoeff[1].x = (u[2].y-u[0].y)/shoelace;
	gradcoeff[2].x = (u[0].y-u[1].y)/shoelace;

	gradcoeff[0].y = (u[2].x-u[1].x)/shoelace;
	gradcoeff[1].y = (u[0].x-u[2].x)/shoelace;
	gradcoeff[2].y = (u[1].x-u[0].x)/shoelace;
	// to get grad of a scalar that is given at each corner, over this tri


	// Self should always be first element in indexlist.
	// (and neighbours are next 3 elements.)

	// First let's create the coefficients on corners that we have
	// for affecting Lap A:

	// = sum [grad A dot edge normal]
	// there are 7 A's to concern about: 1 + 3 neighs + 3 corners

	ZeroMemory(LapCoeff,7*sizeof(real));

	// 3 edges:


	// Edge opposite 0:

	if (pTri->neighbours[0] == pTri) {
		// do nothing
		
		// Our only concern here is, if neighbour even existed.
		// If uNeigh = uCent then Lapcoeff[6] = 0 whereas
		// Lapcoeff[1] + Lapcoeff[0] = 0
		// but in that case better off just saying both apply to Lapcoeff[0]
	} else {
		pTri->neighbours[0]->GenerateContiguousCentroid(&uNeigh,pTri);
			
		// add, grad Ax dot edge_normal on the edge opposite 0:

		shoelace = u[1].x*(uNeigh.y-uCent.y)
				 + uNeigh.x*(u[2].y-u[1].y)
				 + u[2].x*(uCent.y-uNeigh.y)
				 + uCent.x*(u[1].y-u[2].y);
		
		//grad_x = T[1]*(uNeigh.y-uCent.y)/shoelace + ..

		Lapcoeff[0] += ((u[1].y-u[2].y)*pTri->edge_normal[0].x +
						(u[2].x-u[1].x)*pTri->edge_normal[0].y)
							/shoelace;
		
		Lapcoeff[5] += // coeff on corner 1
						((uNeigh.y-uCent.y)*pTri->edge_normal[0].x +
						(uCent.x-uNeigh.x)*pTri->edge_normal[0].y)
							/shoelace;

		Lapcoeff[1] += // coeff on neighbour 0
						((u[2].y-u[1].y)*pTri->edge_normal[0].x +
						(u[1].x-u[2].x)*pTri->edge_normal[0].y)
							/shoelace;
		
		Lapcoeff[6] += // coeff on corner 2
						((uCent.y-uNeigh.y)*pTri->edge_normal[0].x +
						(uNeigh.x-uCent.x)*pTri->edge_normal[0].y)
							/shoelace;
	};

	// Edge opposite 1:

	if (pTri->neighbours[1] != pTri) {

		pTri->neighbours[1]->GenerateContiguousCentroid(&uNeigh,pTri);
		
		shoelace = u[0].x*(uNeigh.y-uCent.y)
				 + uNeigh.x*(u[2].y-u[0].y)
				 + u[2].x*(uCent.y-uNeigh.y)
				 + uCent.x*(u[0].y-u[2].y);

		Lapcoeff[0] += ((u[0].y-u[2].y)*pTri->edge_normal[1].x +
						(u[2].x-u[0].x)*pTri->edge_normal[1].y )/shoelace;

		Lapcoeff[4] += // coeff on corner 0
						((uNeigh.y-uCent.y)*pTri->edge_normal[1].x +
						 (uCent.x-uNeigh.x)*pTri->edge_normal[1].y )/shoelace;
		
		Lapcoeff[2] += // coeff on neighbour 1
						((u[2].y-u[0].y)*pTri->edge_normal[1].x +
						 (u[0].x-u[2].x)*pTri->edge_normal[1].y )/shoelace;

		Lapcoeff[6] += // coeff on corner 2
						((uCent.y-uNeigh.y)*pTri->edge_normal[1].x +
						 (uNeigh.x-uCent.x)*pTri->edge_normal[1].y )/shoelace;
	};


	// Edge opposite 2:

	if (pTri->neighbours[2] != pTri) {
		pTri->neighbours[2]->GenerateContiguousCentroid(&uNeigh,pTri);

		shoelace = u[0].x*(uNeigh.y-uCent.y)
				 + uNeigh.x*(u[1].y-u[0].y)
				 + u[1].x*(uCent.y-uNeigh.y)
				 + uCent.x*(u[0].y-u[1].y);

		Lapcoeff[0] += ((u[0].y-u[1].y)*pTri->edge_normal[2].x +
						(u[1].y-u[0].y)*pTri->edge_normal[2].y)/shoelace;

		Lapcoeff[5] += // coeff on corner 1
			((uCent.y-uNeigh.y)*pTri->edge_normal[2].x +
			(uNeigh.x-uCent.x)*pTri->edge_normal[2].y)/shoelace;

		Lapcoeff[3] += // coeff on neighbour 2
			((u[1].y-u[0].y)*pTri->edge_normal[2].x +
			(u[0].x-u[1].x)*pTri->edge_normal[2].y)/shoelace;

		Lapcoeff[4] += // coeff on corner 0
			((uNeigh.y-uCent.y)*pTri->edge_normal[2].x +
			(uCent.x-uNeigh.x)*pTri->edge_normal[2].y)/shoelace;
	};

	// ***************************************************************
	// Have to work out A_k by interpolation -- a longwinded addition
	// which would barely be necessary for vertex-based.
	// ***************************************************************
	// How to go about it?
	// Existing A values came from centroids in old mesh.

	// Try: create tri from old vertex and old centroids
	A_k = pMesh_With_A_k->EstimateA(uCent);


	// These are coefficients on contiguous. ie,
	// In actually using them we have to be aware that for things that live on
	// different tranches, we need to rotate before applying the effect.


	pTri->coefficients[AMPX][PHI][0] = FOUR_PI_Q_OVER_C*n_i*
			(	pTri->sigma_ion.xx*gradcoeff[0].x
			+	pTri->sigma_ion.xy*gradcoeff[0].y );
	pTri->coefficients[AMPX][PHI][1] = FOUR_PI_Q_OVER_C*n_i*
			(	pTri->sigma_ion.xx*gradcoeff[1].x
			+	pTri->sigma_ion.xy*gradcoeff[1].y );
	pTri->coefficients[AMPX][PHI][2] = FOUR_PI_Q_OVER_C*n_i*
			(	pTri->sigma_ion.xx*gradcoeff[2].x
			+	pTri->sigma_ion.xy*gradcoeff[2].y );

	// NB: In order to calculate E=-grad phi in neighbour we should use 
	// corner positions that are contiguous with this triangle.

	pTri->coefficients[AMPX][AX][0] = Lapcoeff[0]
										+ FOUR_PI_Q_OVER_C*n_i*(
										  pTri->sigma_ion.xx/(c*h));
	pTri->coefficients[AMPX][AY][0] = FOUR_PI_Q_OVER_C*n_i*(
										  pTri->sigma_ion.xy/(c*h));
	pTri->coefficients[AMPX][AZ][0] = FOUR_PI_Q_OVER_C*n_i*(
										  pTri->sigma_ion.xz/(c*h));
	
	tempvec3 = pTri->sigma_ion*A_k/(c*h);
	pTri->coefficients[AMPX][UNITY][0] = FOUR_PI_Q_OVER_C*n_i*(-pTri->v_ion_0.x
										- tempvec3.x);
	
	// Direct neighbours: same PBC tranche or not?

	// It is always the case that we can look into an (existent) neighbour for A,
	// so that's all right. 

	for (iEdge = 0; iEdge < 3; iEdge++)
	{
		if (wnum[iEdge] == 0) {
			pTri->coefficients[AMPX][AX][1+iEdge] = Lapcoeff[1+iEdge];
			pTri->coefficients[AMPX][AY][1+iEdge] = 0.0;
		} else {
			if (wnum[iEdge] == -1) {
				pTri->coefficients[AMPX][AX][1+iEdge] = Anticlockwise.xx*Lapcoeff[1+iEdge];
				pTri->coefficients[AMPX][AY][1+iEdge] = Anticlockwise.xy*Lapcoeff[1+iEdge];
				// Idea is: first we applied anticlockwise then for new x, apply Lapcoeff[1].
			} else {
				pTri->coefficients[AMPX][AX][1+iEdge] = Clockwise.xx*Lapcoeff[1+iEdge];
				pTri->coefficients[AMPX][AY][1+iEdge] = Clockwise.xy*Lapcoeff[1+iEdge];
			};
		};
		pTri->coefficients[AMPX][AZ][1+iEdge] = 0.0;
	};

	// Now to get the coefficients on A-values we add through, with the coefficients
	// by which triangles affect vertices.
	
	// Each corner contribution to Lap, we want to ask does this triangle affect that
	// corner and if so, with what coefficient?
	
	for (iCorner = 0; iCorner < 3; iCorner++)
	{
		pVertex = pTri->cornerptr[iCorner];
		for (i = 0; i < pVertex->triangles.len; i++)
		{
			pTri2 = (Triangle *)(pVertex->triangles.ptr[i]);
			iTri2 = pTri2-T;
			index = pTri->indexlist.FindIndex(iTri2);

			wnum = WindingNumber(pTri2,pTri);
			if (wnum == 0) {
				pTri->coefficients[AMPX][AX][index] += 
					pVertex->coefficients.ptr[i]*Lapcoeff[4+iCorner];

				pTri->coefficients[AMPY][AY][index] +=
					pVertex->coefficients.ptr[i]*Lapcoeff[4+iCorner];

				// suppose we are at the edge of memory. Does it matter?
				// No.
				// If vertex is included it's because of a valid in-domain side
				// and we do indeed then get its value from the triangles it does have.

			} else {
				if (wnum == -1) {
					// Doesn't matter where vertex was located; need to rotate
					// src triangle data into contiguous image.

					pTri->coefficients[AMPX][AX][index] += 
						Anticlockwise.xx*pVertex->coefficients.ptr[i]*Lapcoeff[4+iCorner];
					pTri->coefficients[AMPX][AY][index] += 
						Anticlockwise.xy*pVertex->coefficients.ptr[i]*Lapcoeff[4+iCorner];

					pTri->coefficients[AMPY][AX][index] +=
						Anticlockwise.yx*pVertex->coefficients.ptr[i]*Lapcoeff[4+iCorner];
					pTri->coefficients[AMPY][AY][index] +=
						Anticlockwise.yy*pVertex->coefficients.ptr[i]*Lapcoeff[4+iCorner];

				} else {

					pTri->coefficients[AMPX][AX][index] += 
						Clockwise.xx*pVertex->coefficients.ptr[i]*Lapcoeff[4+iCorner];
					pTri->coefficients[AMPX][AY][index] += 
						Clockwise.xy*pVertex->coefficients.ptr[i]*Lapcoeff[4+iCorner];

					pTri->coefficients[AMPY][AX][index] +=
						Clockwise.yx*pVertex->coefficients.ptr[i]*Lapcoeff[4+iCorner];
					pTri->coefficients[AMPY][AY][index] +=
						Clockwise.yy*pVertex->coefficients.ptr[i]*Lapcoeff[4+iCorner];

				};
			};
			
			pTri->coefficients[AMPZ][AZ][index] +=
					pVertex->coefficients.ptr[i]*Lapcoeff[4+iCorner];
		}
	};
	

	pTri->coefficients[AMPX][VX][0] = FOUR_PI_Q_OVER_C*(n_e - n_i*pTri->beta_i_e.xx);
	pTri->coefficients[AMPX][VY][0] = FOUR_PI_Q_OVER_C*(-n_i*pTri->beta_i_e.xy);
	pTri->coefficients[AMPX][VZ][0] = FOUR_PI_Q_OVER_C*(-n_i*pTri->beta_i_e.xz);
	
	pTri->coefficients[AMPX][EZEXT][0] = FOUR_PI_Q_OVER_C*(-n_i*pTri->sigma_ion.xz);

	// =========================================================================

	pTri->coefficients[AMPY][PHI][0] = FOUR_PI_Q_OVER_C*n_i*
			(	pTri->sigma_ion.yx*gradphicoeff[0].x
			+	pTri->sigma_ion.yy*gradphicoeff[0].y );
	pTri->coefficients[AMPY][PHI][1] = FOUR_PI_Q_OVER_C*n_i*
			(	pTri->sigma_ion.yx*gradphicoeff[1].x
			+	pTri->sigma_ion.yy*gradphicoeff[1].y );
	pTri->coefficients[AMPY][PHI][2] = FOUR_PI_Q_OVER_C*n_i*
			(	pTri->sigma_ion.yx*gradphicoeff[2].x
			+	pTri->sigma_ion.yy*gradphicoeff[2].y );
	
	pTri->coefficients[AMPY][AX][0] = FOUR_PI_Q_OVER_C*n_i*(
										  pTri->sigma_ion.yx/(c*h));
	pTri->coefficients[AMPY][AY][0] = Lapcoeff[0]
										+ FOUR_PI_Q_OVER_C*n_i*(
										  pTri->sigma_ion.yy/(c*h));
	pTri->coefficients[AMPY][AZ][0] = FOUR_PI_Q_OVER_C*n_i*(
										  pTri->sigma_ion.yz/(c*h));
	
	if (wnum0 == 0) {
		pTri->coefficients[AMPY][AX][1] = 0.0;
		pTri->coefficients[AMPY][AY][1] = Lapcoeff[1];
	} else {
		if (wnum0 == -1) {
			pTri->coefficients[AMPY][AX][1] = Anticlockwise.yx*Lapcoeff[1];
			pTri->coefficients[AMPY][AY][1] = Anticlockwise.yy*Lapcoeff[1];
		} else {

			pTri->coefficients[AMPY][AX][1] = Clockwise.yx*Lapcoeff[1];
			pTri->coefficients[AMPY][AY][1] = Clockwise.yy*Lapcoeff[1];
			// remember: apply rotation first (x -> y) then apply Lapcoeff to y element; so Clockwise.yx
			// to give coefficient from x to y.
		};
	};
	pTri->coefficients[AMPY][AZ][1] = 0.0;

	if (wnum1 == 0) {
		pTri->coefficients[AMPY][AX][2] = 0.0;
		pTri->coefficients[AMPY][AY][2] = Lapcoeff[2];
	} else {
		if (wnum1 == -1) {
			pTri->coefficients[AMPY][AX][2] = Anticlockwise.yx*Lapcoeff[2];
			pTri->coefficients[AMPY][AY][2] = Anticlockwise.yy*Lapcoeff[2];
		} else {
			pTri->coefficients[AMPY][AX][2] = Clockwise.yx*Lapcoeff[2];
			pTri->coefficients[AMPY][AY][2] = Clockwise.yy*Lapcoeff[2];
		};
	};
	pTri->coefficients[AMPY][AZ][2] = 0.0;

	if (wnum2 == 0) {
		pTri->coefficients[AMPY][AX][3] = 0.0;
		pTri->coefficients[AMPY][AY][3] = Lapcoeff[3];
	} else {
		if (wnum2 == -1) {
			pTri->coefficients[AMPY][AX][3] = Anticlockwise.yx*Lapcoeff[3];
			pTri->coefficients[AMPY][AY][3] = Anticlockwise.yy*Lapcoeff[3];
		} else {
			pTri->coefficients[AMPY][AX][3] = Clockwise.yx*Lapcoeff[3];
			pTri->coefficients[AMPY][AY][3] = Clockwise.yy*Lapcoeff[3];
		};
	};
	pTri->coefficients[AMPY][AZ][3] = 0.0;

	
	pTri->coefficients[AMPY][VX][0] = FOUR_PI_Q_OVER_C*(-n_i*pTri->beta_i_e.yx);
	pTri->coefficients[AMPY][VY][0] = FOUR_PI_Q_OVER_C*(n_e -n_i*pTri->beta_i_e.yy);
	pTri->coefficients[AMPY][VZ][0] = FOUR_PI_Q_OVER_C*(-n_i*pTri->beta_i_e.yz);
	
	pTri->coefficients[AMPY][UNITY][0] = FOUR_PI_Q_OVER_C*n_i*(-pTri->v_ion_0.y - tempvec3.y);
	

	pTri->coefficients[AMPY][EZEXT][0] = FOUR_PI_Q_OVER_C*(-n_i*pTri->sigma_ion.yz);

	// =========================================================================
	

	pTri->coefficients[AMPZ][PHI][0] = FOUR_PI_Q_OVER_C*n_i*
			(	pTri->sigma_ion.zx*gradphicoeff[0].x
			+	pTri->sigma_ion.zy*gradphicoeff[0].y );
	pTri->coefficients[AMPZ][PHI][1] = FOUR_PI_Q_OVER_C*n_i*
			(	pTri->sigma_ion.zx*gradphicoeff[1].x
			+	pTri->sigma_ion.zy*gradphicoeff[1].y );
	pTri->coefficients[AMPZ][PHI][2] = FOUR_PI_Q_OVER_C*n_i*
			(	pTri->sigma_ion.zx*gradphicoeff[2].x
			+	pTri->sigma_ion.zy*gradphicoeff[2].y );
	
	
	pTri->coefficients[AMPZ][AX][0] = FOUR_PI_Q_OVER_C*n_i*(
										  pTri->sigma_ion.zx/(c*h));
	pTri->coefficients[AMPZ][AY][0] = FOUR_PI_Q_OVER_C*n_i*(
										  pTri->sigma_ion.zy/(c*h));
	pTri->coefficients[AMPZ][AZ][0] = Lapcoeff[0]
										+ FOUR_PI_Q_OVER_C*n_i*(
										  pTri->sigma_ion.zz/(c*h));
	
	pTri->coefficients[AMPZ][AX][1] = 0.0;
	pTri->coefficients[AMPZ][AY][1] = 0.0;
	pTri->coefficients[AMPZ][AZ][1] = Lapcoeff[1];

	pTri->coefficients[AMPZ][AX][2] = 0.0;
	pTri->coefficients[AMPZ][AY][2] = 0.0;
	pTri->coefficients[AMPZ][AZ][2] = Lapcoeff[2];

	pTri->coefficients[AMPZ][AX][3] = 0.0;
	pTri->coefficients[AMPZ][AY][3] = 0.0;
	pTri->coefficients[AMPZ][AZ][3] = Lapcoeff[3];

	pTri->coefficients[AMPZ][VX][0] = FOUR_PI_Q_OVER_C*(- n_i*pTri->beta_i_e.zx);
	pTri->coefficients[AMPZ][VY][0] = FOUR_PI_Q_OVER_C*(-n_i*pTri->beta_i_e.zy);
	pTri->coefficients[AMPZ][VZ][0] = FOUR_PI_Q_OVER_C*(n_e-n_i*pTri->beta_i_e.zz);
	
	pTri->coefficients[AMPZ][UNITY][0] = FOUR_PI_Q_OVER_C*n_i*(-pTri->v_ion_0.z - tempvec3.z);
	
	pTri->coefficients[AMPZ][EZEXT][0] = FOUR_PI_Q_OVER_C*(-n_i*pTri->sigma_ion.zz);
	
	
	/////////////////////////////////////////////////////////////////////
	
	
	// :::::::::::::::::::::::::::::::::::::::::::::::::::
	// II. Now let's do Gauss.
	// :::::::::::::::::::::::::::::::::::::::::::::::::::


	// epsilon_Gauss depends on :
	// A (x,y,z) here and direct neighbours
	// phi at corners of direct neighbours
	// v_e (x,y,z) here and direct neighbours
	// and a constant

	// does not depend on Ez.


	if (pTri->u8domain_flag == PLASMA_DOMAIN) {

		// phi index 3,4,5 indicates the points through sides 0,1,2
		// Div E gives us - Lap phi.

		pTri->coefficients[GAUSS][PHI][0] = -Lapcoeff[0];
		pTri->coefficients[GAUSS][PHI][1] = -Lapcoeff[1];
		pTri->coefficients[GAUSS][PHI][2] = -Lapcoeff[2];
		pTri->coefficients[GAUSS][PHI][3] = -Lapcoeff[3];
		pTri->coefficients[GAUSS][PHI][4] = -Lapcoeff[4];
		pTri->coefficients[GAUSS][PHI][5] = -Lapcoeff[5];
		pTri->coefficients[GAUSS][PHI][6] = -Lapcoeff[6]; 
		// They are set to no contrib from edge of memory.

		pTri->coefficients[GAUSS][AX][0] = 0.0;
		pTri->coefficients[GAUSS][AY][0] = 0.0;
		pTri->coefficients[GAUSS][AZ][0] = 0.0;
		// Div E gives us 0 on A_self, since edge_normals sum to 0.
		// For via v_i see below
		
		pTri->coefficients[GAUSS][UNITY][0] = - FOUR_PI_Q*(pTri->ion.mass-pTri->elec.mass);
		
		pTri->coefficients[GAUSS][EZEXT][0] = 0.0;

		// Those pTri-> masses should be set to give us the default charge in the cell.


		// coeff on A in neigh 0 for Div E comes from
		// -(A-Ak)/(2ch) . edge_normal

		for (iEdge = 0; iEdge < 3; iEdge++)
		{
		
			if (pTri->neighbours[iEdge] != pTri) {
				
				// Now check: does this edge take us out of the plasma domain?
				pNeigh = pTri->neighbours[iEdge];
				if (pNeigh->u8domain_flag == PLASMA_DOMAIN) {

					pNeigh->GenerateContiguousCentroid(&uNeigh,pTri->neighbours[iEdge]);
					// A_k here is from neighbour's tranche.
					A_k_neigh = pMesh_With_A_k->EstimateA(uNeigh);

					pTri->coefficients[GAUSS][VX][0] -= FOUR_PI_Q*h*pTri->elec.mass*
						( pTri->edge_normal[iEdge].x/(pTri->area + pTri->neighbours[iEdge]->area);		
					pTri->coefficients[GAUSS][VY][0] -= FOUR_PI_Q*h*pTri->elec.mass*
						( pTri->edge_normal[iEdge].y/(pTri->area + pTri->neighbours[0]->area);
					// add effect of local ion velocity : the MOVE of electrons
					// will use v_e - v_i based on this existing v_i. For simplicity.

					pTri->coefficients[GAUSS][UNITY][0] += FOUR_PI_Q*h*pTri->elec.mass*
						(pTri->ion.mom/pTri->ion.mass).dotxy(pTri->edge_normal[iEdge])
						/(pTri->area + pTri->neighbours[0]->area);

		// A self -->
		// E = - (A-Ak)/ch - grad phi
		// n_i v_i = ((N_ion)(v_0 + sigma_ion E + beta_ion_e v_e) + same from neigh)/(area sum)
		// eps += 4piq n_i v_i dot edge

					//factor = pTri->ion.mass/(pTri->area+pNeigh->area)/(c*h);

					//pTri->coefficients[GAUSS][AX][0] += -factor*
					//	(pTri->sigma_ion.xx*pTri->edge_normal[iEdge].x
					//	+pTri->sigma_ion.yx*pTri->edge_normal[iEdge].y);
					//// using pTri->sigma_ion in own triangle for our part of the velocity.

					//pTri->coefficients[GAUSS][AY][0] += -factor*
					//	(pTri->sigma_ion.xy*pTri->edge_normal[iEdge].x
					//	+pTri->sigma_ion.yy*pTri->edge_normal[iEdge].y);
					//
					//pTri->coefficients[GAUSS][AZ][0] += -factor*
					//	(pTri->sigma_ion.xz*pTri->edge_normal[iEdge].x
					//	+pTri->sigma_ion.yz*pTri->edge_normal[iEdge].y);

					//pTri->coefficients[GAUSS][UNITY][0] += 
					//	  factor*A_k.x*
					//	(pTri->sigma_ion.xx*pTri->edge_normal[iEdge].x
					//	+pTri->sigma_ion.yx*pTri->edge_normal[iEdge].y)
					//	+ factor*A_k.y*
					//	(pTri->sigma_ion.xy*pTri->edge_normal[iEdge].x
					//	+pTri->sigma_ion.yy*pTri->edge_normal[iEdge].y)
					//	+ factor*A_k.z*
					//	(pTri->sigma_ion.xz*pTri->edge_normal[iEdge].x
					//	+pTri->sigma_ion.yz*pTri->edge_normal[iEdge].y);

					//factor = pTri->ion.mass/(pTri->area+pNeigh->area); 
					//// now do phi
					//pTri->coefficients[GAUSS][PHI][0] += -factor*
					//	(
					//	(pTri->sigma_ion.xx*gradcoeff[0].x
					//	+pTri->sigma_ion.xy*gradcoeff[0].y)*
					//	pTri->edge_normal[iEdge].x + 

					//	(pTri->sigma_ion.yx*gradcoeff[0].x
					//	+pTri->sigma_ion.yy*gradcoeff[0].y)*
					//	pTri->edge_normal[iEdge].y
					//	);
					//pTri->coefficients[GAUSS][PHI][1] += -factor*
					//	(
					//	(pTri->sigma_ion.xx*gradcoeff[1].x
					//	+pTri->sigma_ion.xy*gradcoeff[1].y)*
					//	pTri->edge_normal[iEdge].x + 

					//	(pTri->sigma_ion.yx*gradcoeff[1].x
					//	+pTri->sigma_ion.yy*gradcoeff[1].y)*
					//	pTri->edge_normal[iEdge].y
					//	);

					//pTri->coefficients[GAUSS][PHI][2] += -factor*
					//	(
					//	(pTri->sigma_ion.xx*gradcoeff[2].x
					//	+pTri->sigma_ion.xy*gradcoeff[2].y)*
					//	pTri->edge_normal[iEdge].x + 

					//	(pTri->sigma_ion.yx*gradcoeff[2].x
					//	+pTri->sigma_ion.yy*gradcoeff[2].y)*
					//	pTri->edge_normal[iEdge].y
					//	);

					// Commented: no longer wish to have dependence on v_i(v_e) in Gauss.

					// Instead the move of electrons shall be based on hv_e - hv_i as already estimated.

					//factor = pNeigh->ion.mass/(pTri->area+pNeigh->area)/(c*h);
					
					
					if (wnum[iEdge] == 0) {

						pTri->coefficients[GAUSS][AX][1+iEdge] = -pTri->edge_normal[iEdge].x/(2.0*c*h);
						pTri->coefficients[GAUSS][AY][1+iEdge] = -pTri->edge_normal[iEdge].y/(2.0*c*h);

						pTri->coefficients[GAUSS][UNITY][0] += 
							A_k_neigh.dotxy(pTri->edge_normal[iEdge])/(2.0*c*h);
						
						// Contrib from v_e at this edge:
						// 4pi q (- (mass_1_or_2)/(area1+area2)) dot edge_normal
						pTri->coefficients[GAUSS][VX][1+iEdge] = -FOUR_PI_Q*h*pNeigh->elec.mass*
							( pTri->edge_normal[iEdge].x/(pTri->area + pTri->neighbours[iEdge]->area);

						pTri->coefficients[GAUSS][VY][1+iEdge] = -FOUR_PI_Q*h*pNeigh->elec.mass*
							( pTri->edge_normal[iEdge].y/(pTri->area + pTri->neighbours[iEdge]->area);
						
						pTri->coefficients[GAUSS][UNITY][0] += FOUR_PI_Q*h*pNeigh->elec.mass*
							(pNeigh->ion.mom/pNeigh->ion.mass).dotxy(pTri->edge_normal[iEdge])
							/(pTri->area + pTri->neighbours[iEdge]->area);

						//
						
						
					} else {
						if (wnum[iEdge] == -1) {

							// Careful here: now need to think about contiguous across PBC

							// First apply Anticlockwise then let the result have the same
							// coefficients as above.

							Effect_contiguous = -pTri->edge_normal[iEdge]/(2.0*c*h);
							
							pTri->coefficients[GAUSS][AX][1+iEdge] = Effect_contiguous.x*Anticlockwise.xx + Effect_contiguous.y*Anticlockwise.yx; // Does this match with above?
								
							pTri->coefficients[GAUSS][AY][1+iEdge] = Effect_contiguous.x*Anticlockwise.xy + Effect_contiguous.y*Anticlockwise.yy;

							pTri->coefficients[GAUSS][UNITY][0] += 
								(Anticlockwise*A_k_neigh).dotxy(pTri->edge_normal[iEdge])/(2.0*c*h);

							Effect_contiguous = -FOUR_PI_Q*h*pTri->neighbours[iEdge]->elec.mass*
								( pTri->edge_normal[iEdge]/(pTri->area + pTri->neighbours[iEdge]->area);

							pTri->coefficients[GAUSS][VX][1+iEdge] = Effect_contiguous.x*Anticlockwise.xx + Effect_contiguous.y*Anticlockwise.yx; 
							pTri->coefficients[GAUSS][VY][1+iEdge] = Effect_contiguous.x*Anticlockwise.xy + Effect_contiguous.y*Anticlockwise.yy;

							pTri->coefficients[GAUSS][UNITY][0] += FOUR_PI_Q*h*pNeigh->elec.mass*
							(Anticlockwise3*(pNeigh->ion.mom/pNeigh->ion.mass)).dotxy(pTri->edge_normal[iEdge])
							/(pTri->area + pTri->neighbours[iEdge]->area);

						} else {

							Effect_contiguous = -pTri->edge_normal[iEdge]/(2.0*c*h);
							
							pTri->coefficients[GAUSS][AX][1+iEdge] = Effect_contiguous.x*Clockwise.xx + Effect_contiguous.y*Clockwise.yx; 
							pTri->coefficients[GAUSS][AY][1+iEdge] = Effect_contiguous.x*Clockwise.xy + Effect_contiguous.y*Clockwise.yy;
							
							pTri->coefficients[GAUSS][UNITY][0] += 
								(Clockwise*A_k_neigh).dotxy(pTri->edge_normal[iEdge])/(2.0*c*h);

							Effect_contiguous = -FOUR_PI_Q*h*pNeigh->elec.mass*
								( pTri->edge_normal[iEdge]/(pTri->area + pTri->neighbours[iEdge]->area);

							pTri->coefficients[GAUSS][VX][1+iEdge] = Effect_contiguous.x*Clockwise.xx + Effect_contiguous.y*Clockwise.yx; 
							pTri->coefficients[GAUSS][VY][1+iEdge] = Effect_contiguous.x*Clockwise.xy + Effect_contiguous.y*Clockwise.yy;

							pTri->coefficients[GAUSS][UNITY][0] += FOUR_PI_Q*h*pNeigh->elec.mass*
							(Clockwise3*(pNeigh->ion.mom/pNeigh->ion.mass)).dotxy(pTri->edge_normal[iEdge])
							/(pTri->area + pTri->neighbours[iEdge]->area);


						};
					};
					pTri->coefficients[GAUSS][AZ][1] = 0.0;
					

				} else {
					
					// The neighbour exists but it's outside the domain.

					// Do nothing. ?

					// Check: Boundary conditions?

				};
			} else {
				// do nothing: no contribution to div E at edge of memory.

				// Check that. Boundary conditions?
			};
		}; // next iEdge
		
		// ========================================================================

		//
		// III. Now let's do Ohm.
		//
		
		// We want it to match scaling of the other equations.
		// What is factor on v_e since we used 4pi/c J above?
		// Put 4piq/c n , for now.

		pTri->coefficients[OHMX][UNITY][0] = FOUR_PI_Q_OVER_C*n_e*pTri->v_e_0.x;
		pTri->coefficients[OHMY][UNITY][0] = FOUR_PI_Q_OVER_C*n_e*pTri->v_e_0.y;
		pTri->coefficients[OHMZ][UNITY][0] = FOUR_PI_Q_OVER_C*n_e*pTri->v_e_0.z;
		// formula for v_e_0 given in lyx file serious.lyx.

		pTri->coefficients[OHMX][VX][0] = -FOUR_PI_Q_OVER_C*n_e;
		pTri->coefficients[OHMX][VY][0] = 0.0;
		pTri->coefficients[OHMX][VZ][0] = 0.0;

		pTri->coefficients[OHMY][VX][0] = 0.0;
		pTri->coefficients[OHMY][VY][0] = -FOUR_PI_Q_OVER_C*n_e;
		pTri->coefficients[OHMY][VZ][0] = 0.0;

		pTri->coefficients[OHMZ][VX][0] = 0.0;
		pTri->coefficients[OHMZ][VY][0] = 0.0;
		pTri->coefficients[OHMZ][VZ][0] = -FOUR_PI_Q_OVER_C*n_e;

		for (i = 1; i < pTri->numIndirect; i++)
		{
			pTri2 = T + pTri->indexlist.ptr[i];
			wnum_ = WindingNumber(pTri2,pTri); // check arguments CHECK

			if (wnum_ == 0 ) {
				Effect_of_tri = FOUR_PI_Q_OVER_C*n_e*
							pTri->chi*pTri->visc_coeffs[i];
			} else {
				if (wnum_ == -1) {
					// visc_coeffs tensor applies for contiguous velocity.
					Effect_of_tri = FOUR_PI_Q_OVER_C*n_e*
							pTri->chi*pTri->visc_coeffs[i]*Anticlockwise3;
				} else {
					Effect_of_tri = FOUR_PI_Q_OVER_C*n_e*
							pTri->chi*pTri->visc_coeffs[i]*Clockwise3;
					// logic:
					// first apply rotation, then visc_coeffs, then chi
				};
			};
			pTri->coefficients[OHMX][VX][i] = Effect_of_tri.xx;
			pTri->coefficients[OHMX][VY][i] = Effect_of_tri.xy;
			pTri->coefficients[OHMX][VZ][i] = Effect_of_tri.xz;
			pTri->coefficients[OHMY][VX][i] = Effect_of_tri.yx;
			pTri->coefficients[OHMY][VY][i] = Effect_of_tri.yy;
			pTri->coefficients[OHMY][VZ][i] = Effect_of_tri.yz;
			pTri->coefficients[OHMZ][VX][i] = Effect_of_tri.zx;
			pTri->coefficients[OHMZ][VY][i] = Effect_of_tri.zy;
			pTri->coefficients[OHMZ][VZ][i] = Effect_of_tri.zz;
		};
		
		// coefficients on A, phi come from sigma E

		sigma_use = FOUR_PI_Q_OVER_C*n_e*pTri->sigma_e;
		pTri->coefficients[OHMX][AX][0] = -sigma_use.xx/(c*h);
		pTri->coefficients[OHMX][AY][0] = -sigma_use.xy/(c*h);
		pTri->coefficients[OHMX][AZ][0] = -sigma_use.xz/(c*h);
		pTri->coefficients[OHMX][UNITY][0] = 
			 sigma_use.xx*A_k.x/(c*h)
			+ sigma_use.xy*A_k.y/(c*h)
			+ sigma_use.xz*A_k.z/(c*h);

		pTri->coefficients[OHMY][AX][0] = -sigma_use.yx/(c*h);
		pTri->coefficients[OHMY][AY][0] = -sigma_use.yy/(c*h);
		pTri->coefficients[OHMY][AZ][0] = -sigma_use.yz/(c*h);
		pTri->coefficients[OHMY][UNITY][0] = 
			  sigma_use.yx*A_k.x/(c*h)
			+ sigma_use.yy*A_k.y/(c*h)
			+ sigma_use.yz*A_k.z/(c*h);

		pTri->coefficients[OHMZ][AX][0] = -sigma_use.zx/(c*h);
		pTri->coefficients[OHMZ][AY][0] = -sigma_use.zy/(c*h);
		pTri->coefficients[OHMZ][AZ][0] = -sigma_use.zz/(c*h);
		pTri->coefficients[OHMZ][UNITY][0] = 
			  sigma_use.zx*A_k.x/(c*h)
			+ sigma_use.zy*A_k.y/(c*h)
			+ sigma_use.zz*A_k.z/(c*h);

		// gradcoeff[0].x = effect on grad_x of value at corner 0
		pTri->coefficients[OHMX][PHI][0] = -sigma_use.xx*gradcoeff[0].x
										   -sigma_use.xy*gradcoeff[0].y;
		pTri->coefficients[OHMX][PHI][1] = -sigma_use.xx*gradcoeff[1].x
										   -sigma_use.xy*gradcoeff[1].y;
		pTri->coefficients[OHMX][PHI][2] = -sigma_use.xx*gradcoeff[2].x
										   -sigma_use.xy*gradcoeff[2].y;

		pTri->coefficients[OHMY][PHI][0] = -sigma_use.yx*gradcoeff[0].x
										   -sigma_use.yy*gradcoeff[0].y;
		pTri->coefficients[OHMY][PHI][1] = -sigma_use.yx*gradcoeff[1].x
										   -sigma_use.yy*gradcoeff[1].y;
		pTri->coefficients[OHMY][PHI][2] = -sigma_use.yx*gradcoeff[2].x
										   -sigma_use.yy*gradcoeff[2].y;

		pTri->coefficients[OHMZ][PHI][0] = -sigma_use.zx*gradcoeff[0].x
										   -sigma_use.zy*gradcoeff[0].y;
		pTri->coefficients[OHMZ][PHI][1] = -sigma_use.zx*gradcoeff[1].x
										   -sigma_use.zy*gradcoeff[1].y;
		pTri->coefficients[OHMZ][PHI][2] = -sigma_use.zx*gradcoeff[2].x
										   -sigma_use.zy*gradcoeff[2].y;

		pTri->coefficients[OHMX][EZEXT][0] = sigma_use.xz;
		pTri->coefficients[OHMY][EZEXT][0] = sigma_use.yz;
		pTri->coefficients[OHMZ][EZEXT][0] = sigma_use.zz;



		// To handle:

		// E external? Do we want a further coefficient -- any other way???

		// edge of plasma for Gauss. For eps_Ohm?

		// v_i effect in Gauss.

	} else {
		// Non-domain triangle.

		// We left code same for Ampere. --- ???

		// What to do for Gauss, Ohm ?? 


		// ---

	};

	//
	printf("Doing iterations.\n");
	
	// Given this nontrivial coefficient set, we now have a set of defined
	// linear equations.
	
	IterateFinest();

	// Now apply electron move

	// Set E in cells for doing acceleration stage






	// Free coefficients:

	printf("Freeing coefficient memory.\n");
	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		for (i = 0; i < 7; i++) // 7 residual equations
			for (j = 0; j < 8; j++) // 8 things to have a coefficient on, including unity
				delete[] pTri->coefficients[i][j];
			
		++pTri;
	};
	// Whether this is best approach, not clear. Maybe it's more sensible that
	// coefficients[i] always remains -- only the finest arrays get destroyed, and really
	// it only needs to be those that are for indirect neighbour list.

}

void TriMesh::IterateFinest() 
{

	// Newton-Richardson LS:

	
	JacobiRichardsonLS_A_phi_v(6);

	//LSGS_A_phi_v(2);


}

void TriMesh::JacobiRichardsonLS_A_phi_v(int iterations)
{
	long iIteration;
	real old_RSS, SS1, SS1, L2Rich, L2Jacobi;

	// The point of putting this loop inside is to retain any calcs and reduce overheads.
	for (iIteration = 0; iIteration < iterations; iIteration++)
	{
		// . Calculate epsilons

		old_RSS = CalculateEpsilons_Gauss_Ampere_Ohm();

		for (iTri = 0; iTri < numTriangles; iTri++)
		{
			// Careful here: on which cells does solution take place?
			// For non-domain cells, we take v = 0 -- for now -- ???

			// Only Ampere applies there, and J = 0 for that. Can we set up coefficients so that this is clear?


			// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
			// First try:
			// In serial, first update A, then update v, then update phi
			// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


			// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
			// phi has to live on vertices, for triangle mesh.
			// For a vertex-centred mesh we could let phi live on vertices alongside everything else.
			// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


			if (pTri->coefficients[AX][AX][SELF] == 0.0)
			{
				// ?
				pTri->regressor[JACOBI_REGR][AX] = 0.0;
				pTri->regressor[RICHARDSON_REGR][AX] = 0.0;
			} else {
				pTri->regressor[JACOBI_REGR][AX] = pTri->epsilon_Ampere.x / pTri->coefficients[AX][AX][SELF]; 
				pTri->regressor[RICHARDSON_REGR][AX] = pTri->epsilon_Ampere.x;
			};

			
				pTri->regressor[JACOBI_REGR][AY] = pTri->epsilon_Ampere.y / pTri->coefficients[AY][AY][SELF];
				
				pTri->regressor[JACOBI_REGR][AZ] = pTri->epsilon_Ampere.z / pTri->coefficients[AZ][AZ][SELF];
			
			pTri->regressor[RICHARDSON_REGR][AY] = pTri->epsilon_Ampere.y;
			pTri->regressor[RICHARDSON_REGR][AZ] = pTri->epsilon_Ampere.z;

			SS1 += pTri->regressor[JACOBI_REGR]*pTri->regressor[JACOBI_REGR];
			SS2 += pTri->regressor[RICHARDSON_REGR]*pTri->regressor[RICHARDSON_REGR];

			++pTri;
		};

		printf("RSS %1.4E ",old_RSS);
		if (SS1 == 0.0) return;
		if (SS2 == 0.0) return; // one of the regressors a zero vector -- ?!

		// Note: working with numTriangles instead of numnonzero.
		// Consequences?
		L2Jacobi = sqrt(SS1/(real)(numTriangles));
		L2Rich = sqrt(SS2/(real)(numTriangles));
		
		pTri = T;
		for (iTri = 0; iTri < numTriangles; iTri++)
		{
			pTri->regressor[JACOBI_REGR][AX] /= L2Jacobi;
			pTri->regressor[JACOBI_REGR][AY] /= L2Jacobi;
			pTri->regressor[JACOBI_REGR][AZ] /= L2Jacobi;
			pTri->regressor[RICHARDSON_REGR][AX] /= L2Rich;
			pTri->regressor[RICHARDSON_REGR][AY] /= L2Rich;
			pTri->regressor[RICHARDSON_REGR][AZ] /= L2Rich;
			++pTri;
		};
		
		//
		//
		//




		Sum11 = 0.0; Sum22 = 0.0; Sum12 = 0.0; Sum1eps = 0.0; Sum2eps = 0.0;
		Ax1 = 0.0; Ax2 = 0.0;

		pTri = T;
		for (iTri = 0; iTri < numTriangles; iTri++)
		{

			// Collect the vector found by multiplying coefficient matrix on to the regressors,
			// ie, get epsilon if the variables being changed, equalled the regressors.

			// This will be a 7-vector for each regressor.

			// Fill in CalculateEpsilon function first.


			////Ax1 = pAux->coeff_self*pAux->vLanczos[0];
			////Ax2 = pAux->coeff_self*pAux->vLanczos[1];

			////if (pAux->coeff_self == 0.0)
			////{
			////	// Almost certainly means it is in the inner mesh where we don't populate because we solve on the domain only.
			////} else {
			////	coeffptr = pAux->coefficients;
			////	neighlen = pAux->neigh_len;

			////	for (i = 0; i < neighlen; i++)
			////	{
			////		Ax1 += (*coeffptr)*((AuxX[iLevel]+pAux->iNeighbours[i])->vLanczos[0]);
			////		Ax2 += (*coeffptr)*((AuxX[iLevel]+pAux->iNeighbours[i])->vLanczos[1]);
			////		++coeffptr;
			////	};		
			////	for (i = 0; i < pAux->index_extra.len; i++)
			////	{
			////		Ax1 += (pAux->coeff_extra.ptr[i])*((AuxX[iLevel]+pAux->index_extra.ptr[i])->vLanczos[0]);
			////		Ax2 += (pAux->coeff_extra.ptr[i])*((AuxX[iLevel]+pAux->index_extra.ptr[i])->vLanczos[1]);
			////	};		
			////	
			////	Sum11 += Ax1*Ax1;
			////	Sum12 += Ax2*Ax1;
			////	Sum22 += Ax2*Ax2;
			////	Sum1eps += Ax1*pAux->epsilon;
			////	Sum2eps += Ax2*pAux->epsilon;

			////};

			// Probably:
			Sum11 += Ax1Ampere.dot(Ax1Ampere) + Ax1Ohm.dot(Ax1Ohm) + Ax1Gauss*Ax1Gauss;

			Sum1eps += Ax1Ampere.dot(pTri->epsilon_Ampere) + 
						Ax1Ohm.dot(pTri->epsilon_Ohm) +
						Ax1Gauss*pTri->epsilon_Gauss;
			
			++pTri;
		};



		// Calculate coefficients gamma

		det = Sum11*Sum22-Sum12*Sum12;
		
		if (det == 0.0) return;

		gamma1 = -(Sum22*Sum1eps-Sum12*Sum2eps)/det;
		gamma2 = -(-Sum12*Sum1eps+Sum11*Sum2eps)/det;// - because - on rhs
		

		// Work out 

		pTri = T;
		for (iTri = 0; iTri < numTriangles; iTri++)
		{
			pTri->A += gamma1*pTri->regressor[JACOBI_REGR] + gamma2*pTri->regressor[RICHARDSON_REGR];

			++pTri;
		};

		// Clearly we need to go over a little algebra for this also.

		

		// Now update vrel




		// Now update phi on verts
		// Jacobi direction comes from sum of affected cluster.





	}; // next iteration
}

real TriMesh::CalculateEpsilons_Gauss_Ampere_Ohm()
{

	real RSS = 0.0;

	Triangle * pTri;
	long iTri;

	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		// 7-dimensional epsilon vector


		// epsilon_Ampere depends on :
		// A (component effect) in indirect neighbours

		//  [but PB will make it a 2x2 x-y effect at least]
		//  & MM will have to recognise as such, so... do 3x3?

		// phi at corners
		// v_e (3x3 effect) here
		// and a constant


		// epsilon_Gauss depends on :
		// A (x,y,z) here and direct neighbours
		// phi at corners of direct neighbours
		// v_e (x,y,z) here and direct neighbours
		// and a constant


		// epsilon_Ohm depends on :
		// A (3x3 effect) here 
		// phi at corners
		// v_e (3x3 effect) in indirect neighbours
		// and a constant


		// If we could use vertex-based, there would be far more symmetry.
		// Lap A, Lap v, Lap phi would come from direct neighbours all.


		// First the things that do not use the indirect neighbour list:

		pTri->epsilon_Ampere.x = pTri->coefficients[AMPX][UNITY][0]

								+ pTri->coefficients[AMPX][PHI][0]*pTri->cornerptr[0]->phi
								+ pTri->coefficients[AMPX][PHI][1]*pTri->cornerptr[1]->phi
								+ pTri->coefficients[AMPX][PHI][2]*pTri->cornerptr[2]->phi
								
								+ pTri->coefficients[AMPX][VX][0]*(pTri->v_e.x)
								+ pTri->coefficients[AMPX][VY][0]*(pTri->v_e.y)
								+ pTri->coefficients[AMPX][VZ][0]*(pTri->v_e.z);

		pTri->epsilon_Ampere.y = pTri->coefficients[AMPY][UNITY][0]

								+ pTri->coefficients[AMPY][PHI][0]*pTri->cornerptr[0]->phi
								+ pTri->coefficients[AMPY][PHI][1]*pTri->cornerptr[1]->phi
								+ pTri->coefficients[AMPY][PHI][2]*pTri->cornerptr[2]->phi
								
								+ pTri->coefficients[AMPY][VX][0]*(pTri->v_e.x)
								+ pTri->coefficients[AMPY][VY][0]*(pTri->v_e.y)
								+ pTri->coefficients[AMPY][VZ][0]*(pTri->v_e.z);

		pTri->epsilon_Ampere.z = pTri->coefficients[AMPZ][UNITY][0]

								+ pTri->coefficients[AMPZ][PHI][0]*pTri->cornerptr[0]->phi
								+ pTri->coefficients[AMPZ][PHI][1]*pTri->cornerptr[1]->phi
								+ pTri->coefficients[AMPZ][PHI][2]*pTri->cornerptr[2]->phi
								
								+ pTri->coefficients[AMPZ][VX][0]*(pTri->v_e.x)
								+ pTri->coefficients[AMPZ][VY][0]*(pTri->v_e.y)
								+ pTri->coefficients[AMPZ][VZ][0]*(pTri->v_e.z);

		// Should try to speed up the following using local intermediate variables and pointers.

		// As it stands, pTri refers to a globally created variable and everything here is a global access.
		// Quite possibly a simple memcpy of pTri->coefficients to a local array would be faster!
		// Tricky : have to know how to address each dynamic array.
		// Maybe do a memcpy for every indexlist set of coefficients, at least.

		// And here do locals to speed up, such as
		// real ** pTemp = pTri->coefficients[GAUSS];
		// real * pTempPhi = pTemp[PHI];
		// real * pTempAx = pTemp[AX];
		// etc;
		// A = pTri->A, phi0 = pTri->cornerptr[0]->phi, etc

		// Is there any way to declare that *pTri is read-only?

		pTri->epsilon_Gauss = pTri->coefficients[GAUSS][UNITY][0]

							+ pTri->coefficients[GAUSS][PHI][0]*pTri->cornerptr[0]->phi
							+ pTri->coefficients[GAUSS][PHI][1]*pTri->cornerptr[1]->phi
							+ pTri->coefficients[GAUSS][PHI][2]*pTri->cornerptr[2]->phi
							+ pTri->coefficients[GAUSS][PHI][3]*pTri->neighbours[0]->ReturnUnsharedVertex(pTri)->phi
							+ pTri->coefficients[GAUSS][PHI][4]*pTri->neighbours[1]->ReturnUnsharedVertex(pTri)->phi
							+ pTri->coefficients[GAUSS][PHI][5]*pTri->neighbours[2]->ReturnUnsharedVertex(pTri)->phi

							// possibly the following 3 coefficients are zero, for self?
							+ pTri->coefficients[GAUSS][AX][0]*pTri->A.x
							+ pTri->coefficients[GAUSS][AY][0]*pTri->A.y
							+ pTri->coefficients[GAUSS][AZ][0]*pTri->A.z

							+ pTri->coefficients[GAUSS][AX][1]*pNeigh0->A.x
							+ pTri->coefficients[GAUSS][AY][1]*pNeigh0->A.y
							+ pTri->coefficients[GAUSS][AZ][1]*pNeigh0->A.z

							+ pTri->coefficients[GAUSS][AX][2]*pNeigh1->A.x
							+ pTri->coefficients[GAUSS][AY][2]*pNeigh1->A.y
							+ pTri->coefficients[GAUSS][AZ][2]*pNeigh1->A.z

							+ pTri->coefficients[GAUSS][AX][3]*pNeigh2->A.x
							+ pTri->coefficients[GAUSS][AY][3]*pNeigh2->A.y
							+ pTri->coefficients[GAUSS][AZ][3]*pNeigh2->A.z

							// rho is affected by v here and in neighbours.
							+ pTri->coefficients[GAUSS][VX][0]*pTri->vrel.x
							+ pTri->coefficients[GAUSS][VY][0]*pTri->vrel.y
							+ pTri->coefficients[GAUSS][VZ][0]*pTri->vrel.z

							+ pTri->coefficients[GAUSS][VX][1]*pNeigh0->vrel.x
							+ pTri->coefficients[GAUSS][VY][1]*pNeigh0->vrel.y
							+ pTri->coefficients[GAUSS][VZ][1]*pNeigh0->vrel.z

							+ pTri->coefficients[GAUSS][VX][2]*pNeigh1->vrel.x
							+ pTri->coefficients[GAUSS][VY][2]*pNeigh1->vrel.y
							+ pTri->coefficients[GAUSS][VZ][2]*pNeigh1->vrel.z

							+ pTri->coefficients[GAUSS][VX][3]*pNeigh2->vrel.x
							+ pTri->coefficients[GAUSS][VY][3]*pNeigh2->vrel.y
							+ pTri->coefficients[GAUSS][VZ][3]*pNeigh2->vrel.z
							;

		// Can we save some costs here: calculate E_cell and we get to save on calcs, at least in this routine. 
		// (That sounds like it makes a lot of sense! Still fill in coeffs to know
		// putative effects of changing e.g. phi.)
		// Only used in Ohm ?

		pTri->epsilon_Ohm.x = pTri->coefficients[OHMX][UNITY][0]
							
							+ pTri->coefficients[OHMX][PHI][0]*pTri->cornerptr[0]->phi
							+ pTri->coefficients[OHMX][PHI][1]*pTri->cornerptr[1]->phi
							+ pTri->coefficients[OHMX][PHI][2]*pTri->cornerptr[2]->phi

							+ pTri->coefficients[OHMX][AX][0]*pTri->A.x
							+ pTri->coefficients[OHMX][AY][0]*pTri->A.y
							+ pTri->coefficients[OHMX][AZ][0]*pTri->A.z;

		pTri->epsilon_Ohm.y = pTri->coefficients[OHMY][UNITY][0]
							
							  pTri->coefficients[OHMY][PHI][0]*pTri->cornerptr[0]->phi
							+ pTri->coefficients[OHMY][PHI][1]*pTri->cornerptr[1]->phi
							+ pTri->coefficients[OHMY][PHI][2]*pTri->cornerptr[2]->phi

							+ pTri->coefficients[OHMY][AX][0]*pTri->A.x
							+ pTri->coefficients[OHMY][AY][0]*pTri->A.y
							+ pTri->coefficients[OHMY][AZ][0]*pTri->A.z;

		pTri->epsilon_Ohm.z = pTri->coefficients[OHMZ][UNITY][0]
							
							+ pTri->coefficients[OHMZ][PHI][0]*pTri->cornerptr[0]->phi
							+ pTri->coefficients[OHMZ][PHI][1]*pTri->cornerptr[1]->phi
							+ pTri->coefficients[OHMZ][PHI][2]*pTri->cornerptr[2]->phi

							+ pTri->coefficients[OHMZ][AX][0]*pTri->A.x
							+ pTri->coefficients[OHMZ][AY][0]*pTri->A.y
							+ pTri->coefficients[OHMZ][AZ][0]*pTri->A.z;


		// i == 0 is for self.
		for (i = 0; i < pTri->indexlist.len; i++)
		{
			pTri2 = T + pTri->indexlist.ptr[i];

		// epsilon_Ampere depends on :
		// A (component effect) in indirect neighbours
		//  [but PB will make it a 2x2 x-y effect at least]
		//  & MM will have to recognise as such, so... do 2x2

		// epsilon_Ohm depends on :
		// v_e (3x3 effect) in indirect neighbours

			pTri->epsilon_Ampere.x += pTri->coefficients[AMPX][AX][i]*(pTri2->A.x)
									+ pTri->coefficients[AMPX][AY][i]*(pTri2->A.y);
									//+ pTri->coefficients[AX][AZ][i]*(pTri2->A.z);
			// What else does Ampere epsilon depend upon?
			
			pTri->epsilon_Ampere.y += pTri->coefficients[AMPY][AX][i]*(pTri2->A.x)
									+ pTri->coefficients[AMPY][AY][i]*(pTri2->A.y);
									//+ pTri->coefficients[AY][AZ][i]*(pTri2->A.z);

			pTri->epsilon_Ampere.z += pTri->coefficients[AMPZ][AZ][i]*(pTri2->A.z);


			pTri->epsilon_Ohm.x += pTri->coefficients[OHMX][VX][i]*(pTri2->v_e.x)
								 + pTri->coefficients[OHMX][VY][i]*(pTri2->v_e.y)
								 + pTri->coefficients[OHMX][VZ][i]*(pTri2->v_e.z);

			pTri->epsilon_Ohm.y += pTri->coefficients[OHMY][VX][i]*(pTri2->v_e.x)
								 + pTri->coefficients[OHMY][VY][i]*(pTri2->v_e.y)
								 + pTri->coefficients[OHMY][VZ][i]*(pTri2->v_e.z);
			
			pTri->epsilon_Ohm.z += pTri->coefficients[OHMZ][VX][i]*(pTri2->v_e.x)
								 + pTri->coefficients[OHMZ][VY][i]*(pTri2->v_e.y)
								 + pTri->coefficients[OHMZ][VZ][i]*(pTri2->v_e.z);
		};

		RSS += pTri->epsilon_Ampere.dot(pTri->epsilon_Ampere)
				+ pTri->epsilon_Gauss*pTri->epsilon_Gauss
				+ pTri->epsilon_Ohm.dot(pTri->epsilon_Ohm);
		
		++pTri;
	}

	return RSS;
}

void TriMesh::GetBFromA()
{
	bool twosides_above;
	int ileft,iright,iother;
	real yexpected, Integral1, Integral2, Integral3, Integral1y,Integral2y,Integral3y,Integral4,Integral4y;
	Vector2 u_left,u_mid,u_right,u_leftins,u_rightins,u_leftout,u_rightout;
	real Az_left,Az_mid,Az_right,Az_leftins,Az_leftout,Az_rightins,Az_rightout;
	Vector3 A_left, A_mid, A_right, A_leftins,A_leftout,A_rightins,A_rightout;
	Vector3 sum_A;
	Vector2 diff_pos;
	Triangle * pTri;
	long iTri;


	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		pTri->B.x = 0.0;
		pTri->B.y = 0.0;
		pTri->B.z = 0.0;
		
		++pTri;
	};

	
	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	//														III. Infer B on cells from A on vertices.
	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	// This relies on the fact that the flux through a surface at velocity curl A is equal to the line integral of
	// A around the edge. 

	// Bz

	// The following code could now be simplified as follows.
	// When we call RecalculateEdgeNormalVectors(false), these have the side length as their length;
	// rotating them anticlockwise gives anticlockwise vectors.
	// (not sure about periodic)

	// 
	
	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		pTri->B.z = 0.0;

		ileft = pTri->GetLeftmostIndex();
		iright = pTri->GetRightmostIndex();
		// ^^ Note -- this ignores mapping left for periodic. Bear in mind.
				
		iother = 0; while ((iother == ileft) || (iother == iright)) iother++;	

		if (pTri->periodic == 0)
		{
			pTri->cornerptr[ileft]->PopulatePosition(u_left);
			pTri->cornerptr[iother]->PopulatePosition(u_mid);
			pTri->cornerptr[iright]->PopulatePosition(u_right);
			A_right = pTri->cornerptr[iright]->A;
			A_mid = pTri->cornerptr[iother]->A;
			A_left = pTri->cornerptr[ileft]->A;
		} else {
			if (pTri->periodic == 1)
			{ 
				// map left:
				pTri->cornerptr[ileft]->PopulatePosition(u_right);
				pTri->cornerptr[iother]->periodic_image(u_left,0,1);
				pTri->cornerptr[iright]->periodic_image(u_mid,0,1);
					
				// Note that where vertices are mapped out of our tranche, we must
				// multiply by Anticlockwise matrix:

				A_right = pTri->cornerptr[ileft]->A;
				A_left = Anticlockwise3*(pTri->cornerptr[iother]->A);
				A_mid = Anticlockwise3*(pTri->cornerptr[iright]->A);
			} else {
				// Note that by mapping all left, 
				// we are therefore going to get the B that applies when this cell is situated on the left side.
					
				pTri->cornerptr[ileft]->PopulatePosition(u_mid);
				pTri->cornerptr[iother]->PopulatePosition(u_right);
				pTri->cornerptr[iright]->periodic_image(u_left,0,1);
					
				A_left = Anticlockwise3*(pTri->cornerptr[iright]->A);
				A_mid = pTri->cornerptr[ileft]->A;
				A_right = pTri->cornerptr[iother]->A;
			};
		};
		// got to make sure it is anticlockwise. 
		// What to do about that?
		// start from rightmost point
		// If the middle point is above the left-right line, then we move to the middle point; otherwise go the other way.

		// In practice always go right->mid->left, just multiply by -1 if necessary.

		sum_A = A_right + A_mid;
		diff_pos = u_mid - u_right;
		pTri->B.z += (sum_A.x*diff_pos.x + sum_A.y*diff_pos.y);

		// [A_avg dot edge vector]

		sum_A = A_left + A_mid;
		diff_pos = u_left - u_mid;
		pTri->B.z +=  (sum_A.x*diff_pos.x + sum_A.y*diff_pos.y);

		 sum_A = A_left + A_right;
		diff_pos = u_right - u_left;
		 pTri->B.z += (sum_A.x*diff_pos.x + sum_A.y*diff_pos.y);

		yexpected =u_left.y + (u_mid.x-u_left.x)*(u_right.y-u_left.y)/(u_right.x-u_left.x); // this had a mistake!
		twosides_above = (u_mid.y > yexpected);
		if (!(twosides_above)) pTri->B.z = -pTri->B.z; 
		// if middle x value is above line, we went anti-clockwise round.
		// if below, we went clockwise.

		pTri->B.z /= (2.0*pTri->area);

 
		pTri->B.z += BZ_CONSTANT;

		++pTri;
	};

	// DEBUG:

	// Output Axy for the first rows - vertex 0 to 415, tri to 834

	//FILE * file = fopen("debugAB.txt","w");

	//Vertex * pVertex = X;
	//for (long iVertex = 0; iVertex < 415; iVertex++)
	//{
	//	fprintf(file,"%d %1.14E %1.14E %1.14E %1.14E \n",
	//		iVertex, pVertex->x,pVertex->y,pVertex->A.x,pVertex->A.y);
	//	++pVertex;
	//};

	//fprintf(file,"\n\n");

	//pTri = T;
	//for (iTri = 0; iTri < 834; iTri++)
	//{
	//	fprintf(file,"%d %d %d %d %d  %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E  %1.14E %1.14E %1.14E %1.14E %1.14E %1.14E \n",
	//		iTri, pTri->cornerptr[0]-X,pTri->cornerptr[1]-X,pTri->cornerptr[2]-X,pTri->periodic,
	//		pTri->B.z, pTri->area,
	//		pTri->cornerptr[0]->A.x, pTri->cornerptr[1]->A.x,pTri->cornerptr[2]->A.x,pTri->cornerptr[0]->A.y,pTri->cornerptr[1]->A.y,pTri->cornerptr[2]->A.y,
	//		pTri->cornerptr[0]->x,pTri->cornerptr[0]->y,pTri->cornerptr[1]->x,pTri->cornerptr[1]->y,pTri->cornerptr[2]->x,pTri->cornerptr[2]->y);
	//	
	//	++pTri;
	//};

	//fclose(file);

	// Then ask why that happens .. or why it is not shown ...



	// Bxy
	
	// Meanwhile, Bx = d/dy Az   <-- integrate over area and we get something with x-distances - go positive along the top and negative along bottom
	// By = - d/dx Az
	// Integral over cell Bx = integral along x-projection (Az(top)-Az(bottom))
	// Integral over cell By = MINUS integral along y-projection (Az(right)-Az(left))
	
	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		ileft = pTri->GetLeftmostIndex();
		iright = pTri->GetRightmostIndex(); 

		iother = 0;
		while ((iother == ileft) || (iother == iright)) iother++;
			
		if (pTri->periodic == 0)
		{
			// establish whether the other point is above line or below
			yexpected = pTri->cornerptr[ileft]->y + (pTri->cornerptr[iother]->x-pTri->cornerptr[ileft]->x)*
				(pTri->cornerptr[iright]->y-pTri->cornerptr[ileft]->y)/(pTri->cornerptr[iright]->x-pTri->cornerptr[ileft]->x);
			twosides_above = (pTri->cornerptr[iother]->y > yexpected);

			// assume that A changes linearly along edges
			// => Integral dx is x length multiplied by average
			// Integral dy is y length multiplied by average

			Integral1 = 0.5*(pTri->cornerptr[ileft]->A.z + pTri->cornerptr[iother]->A.z)*(pTri->cornerptr[iother]->x-pTri->cornerptr[ileft]->x);
			Integral2 = 0.5*(pTri->cornerptr[iright]->A.z + pTri->cornerptr[iother]->A.z)*(pTri->cornerptr[iright]->x-pTri->cornerptr[iother]->x);
			Integral3 = 0.5*(pTri->cornerptr[iright]->A.z + pTri->cornerptr[ileft]->A.z)*(pTri->cornerptr[iright]->x-pTri->cornerptr[ileft]->x);

			Integral1y = 0.5*(pTri->cornerptr[ileft]->A.z + pTri->cornerptr[iother]->A.z)*(pTri->cornerptr[iother]->y-pTri->cornerptr[ileft]->y);
			Integral2y = 0.5*(pTri->cornerptr[iright]->A.z + pTri->cornerptr[iother]->A.z)*(pTri->cornerptr[iright]->y-pTri->cornerptr[iother]->y);
			Integral3y = 0.5*(pTri->cornerptr[iright]->A.z + pTri->cornerptr[ileft]->A.z)*(pTri->cornerptr[iright]->y-pTri->cornerptr[ileft]->y);

			// positive across the top. The three 'integral' are all positive.
			pTri->B.x = (twosides_above)?(Integral1+Integral2-Integral3):-(Integral1+Integral2-Integral3);

			// Integral over cell By = MINUS integral along y-projection (Az(right)-Az(left))
				
			// case by case analysis comes down to this:
			if (twosides_above)
			{							// Note that we are counting positive y-difference for things that appear on left side of shape:
				pTri->B.y = Integral1y + Integral2y - Integral3y; 
			} else {
				pTri->B.y = -Integral1y - Integral2y + Integral3y; 
			};
			// we then divide by area.

		} else {
			
			if (pTri->periodic == 1)
			{
				// map the two unmapped ones (not ileft) to left
				// ileft becomes iright
				// iright becomes iother
				// iother becomes ileft
				
				pTri->cornerptr[ileft]->PopulatePosition(u_right);
				pTri->cornerptr[iright]->periodic_image(u_mid,0,1);
				pTri->cornerptr[iother]->periodic_image(u_left,0,1);
				Az_left = pTri->cornerptr[iother]->A.z;
				Az_right = pTri->cornerptr[ileft]->A.z;
				Az_mid = pTri->cornerptr[iright]->A.z;
			} else {
				// map the unmapped one (iright) to left
				// ileft becomes iother
				// iother becomes iright
				// iright becomes ileft

				pTri->cornerptr[ileft]->PopulatePosition(u_mid);
				pTri->cornerptr[iother]->PopulatePosition(u_right);
				pTri->cornerptr[iright]->periodic_image(u_left,0,1);
				Az_left = pTri->cornerptr[iright]->A.z;
				Az_mid = pTri->cornerptr[ileft]->A.z;
				Az_right = pTri->cornerptr[iother]->A.z;
			};
			
			yexpected = u_left.y + (u_mid.x-u_left.x)*(u_right.y-u_left.y)/(u_right.x-u_left.x);
			twosides_above = (u_mid.y > yexpected);
			
			Integral1 = 0.5*(Az_left + Az_mid)*(u_mid.x-u_left.x);
			Integral2 = 0.5*(Az_right + Az_mid)*(u_right.x-u_mid.x);
			Integral3 = 0.5*(Az_right + Az_left)*(u_right.x-u_left.x);

			Integral1y = 0.5*(Az_left + Az_mid)*(u_mid.y-u_left.y);
			Integral2y = 0.5*(Az_right + Az_mid)*(u_right.y-u_mid.y);
			Integral3y = 0.5*(Az_right + Az_left)*(u_right.y-u_left.y);

			pTri->B.x = (twosides_above)?(Integral1+Integral2-Integral3):-(Integral1+Integral2-Integral3);
			// case by case analysis comes down to this:
			if (twosides_above)
			{							// Note that we are counting positive y-difference for things that appear on left side of shape:
				pTri->B.y = Integral1y + Integral2y - Integral3y; 
			} else {
				pTri->B.y = -Integral1y - Integral2y + Integral3y; 
			};
		};
		
		pTri->B.x /= pTri->area;
		pTri->B.y /= pTri->area;
		
		++pTri;
	};
	
}


void TriMesh::GetCurlBcOver4Pi()
{
	bool twosides_above;
	int ileft,iright,iother;
	real yexpected, Integral1, Integral2, Integral3, Integral1y,Integral2y,Integral3y,Integral4,Integral4y;
	Vector2 u_left,u_mid,u_right,u_leftins,u_rightins,u_leftout,u_rightout;
	real Bz_left,Bz_mid,Bz_right,Bz_leftins,Bz_leftout,Bz_rightins,Bz_rightout;
	Vector3 B_left, B_mid, B_right, B_leftins,B_leftout,B_rightins,B_rightout;
	Vector3 sum_B;
	Vector2 diff_pos;
	Triangle * pTri;
	long iTri;
	static real const FOURPI = 4.0*PI;

	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		pTri->numerator_x = 0.0;
		pTri->numerator_y = 0.0;
		pTri->numerator_z = 0.0;
		
		++pTri;
	};

	
	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	//														III. Infer B on cells from A on vertices.
	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	// This relies on the fact that the flux through a surface at velocity curl A is equal to the line integral of
	// A around the edge. 
	// Integral of curl A _z over cell = integral dR A dot dR
	// Bz = (1/Area) sum over sides
	//         Integral {t,0,1} [ ( t A1 + (1-t) A0 ) . (x1 - x0)/|| x1-x0 || ]
	// = (1/Area) (1/2) sum ((A1 + A0) . (x1 -x0)/ || x1-x0 || )
	
	// I think this looks like we are wrongly dividing by side length. 


	// Bz


	// The following code could now be simplified as follows.
	// When we call RecalculateEdgeNormalVectors(false), these have the side length as their length;
	// rotating them anticlockwise gives anticlockwise vectors.
	// (not sure about periodic)
	
	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		pTri->numerator_z = 0.0;

		ileft = pTri->GetLeftmostIndex();
		iright = pTri->GetRightmostIndex();
		// ^^ Note -- this ignores mapping left for periodic. Bear in mind.
				
		iother = 0; while ((iother == ileft) || (iother == iright)) iother++;	

		if (pTri->periodic == 0)
		{
			pTri->cornerptr[ileft]->PopulatePosition(u_left);
			pTri->cornerptr[iother]->PopulatePosition(u_mid);
			pTri->cornerptr[iright]->PopulatePosition(u_right);
			B_right = pTri->cornerptr[iright]->B;
			B_mid = pTri->cornerptr[iother]->B;
			B_left = pTri->cornerptr[ileft]->B;
		} else {
			if (pTri->periodic == 1)
			{ 
				// map left:
				pTri->cornerptr[ileft]->PopulatePosition(u_right);
				pTri->cornerptr[iother]->periodic_image(u_left,0,1);
				pTri->cornerptr[iright]->periodic_image(u_mid,0,1);
					
				// Note that where vertices are mapped out of our tranche, we must
				// multiply by Bnticlockwise matrix:

				B_right = pTri->cornerptr[ileft]->B;
				B_left = Anticlockwise3*(pTri->cornerptr[iother]->B);
				B_mid = Anticlockwise3*(pTri->cornerptr[iright]->B);
			} else {
				// Note that by mapping all left, 
				// we are therefore going to get the B that applies when this cell is situated on the left side.
					
				pTri->cornerptr[ileft]->PopulatePosition(u_mid);
				pTri->cornerptr[iother]->PopulatePosition(u_right);
				pTri->cornerptr[iright]->periodic_image(u_left,0,1);
					
				B_left = Anticlockwise3*(pTri->cornerptr[iright]->B);
				B_mid = pTri->cornerptr[ileft]->B;
				B_right = pTri->cornerptr[iother]->B;
			};
		};
		// got to make sure it is anticlockwise. 
		// What to do about that?
		// start from rightmost point
		// If the middle point is above the left-right line, then we move to the middle point; otherwise go the other way.

		// In practice always go right->mid->left, just multiply by -1 if necessary.

		sum_B = B_right + B_mid;
		diff_pos = u_mid - u_right;
		pTri->numerator_z += (sum_B.x*diff_pos.x + sum_B.y*diff_pos.y);

		// [B_avg dot edge vector]

		sum_B = B_left + B_mid;
		diff_pos = u_left - u_mid;
		pTri->numerator_z +=  (sum_B.x*diff_pos.x + sum_B.y*diff_pos.y);

		 sum_B = B_left + B_right;
		diff_pos = u_right - u_left;
		 pTri->numerator_z += (sum_B.x*diff_pos.x + sum_B.y*diff_pos.y);

		yexpected =u_left.y + (u_mid.x-u_left.x)*(u_right.y-u_left.y)/(u_right.x-u_left.x); // this had a mistake!
		twosides_above = (u_mid.y > yexpected);
		if (!(twosides_above)) pTri->numerator_z = -pTri->numerator_z; 
		// if middle x value is above line, we went anti-clockwise round.
		// if below, we went clockwise.

		pTri->numerator_z /= (2.0*pTri->area);

		pTri->numerator_z *= c/(FOURPI);

		++pTri;
	};

	// Bxy
	
	// Meanwhile, Bx = d/dy Az   <-- integrate over area and we get something with x-distances - go positive along the top and negative along bottom
	// By = - d/dx Az
	// Integral over cell Bx = integral along x-projection (Az(top)-Az(bottom))
	// Integral over cell By = MINUS integral along y-projection (Az(right)-Az(left))
	
	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		ileft = pTri->GetLeftmostIndex();
		iright = pTri->GetRightmostIndex(); 

		iother = 0;
		while ((iother == ileft) || (iother == iright)) iother++;
			
		if (pTri->periodic == 0)
		{
			// establish whether the other point is above line or below
			yexpected = pTri->cornerptr[ileft]->y + (pTri->cornerptr[iother]->x-pTri->cornerptr[ileft]->x)*
				(pTri->cornerptr[iright]->y-pTri->cornerptr[ileft]->y)/(pTri->cornerptr[iright]->x-pTri->cornerptr[ileft]->x);
			twosides_above = (pTri->cornerptr[iother]->y > yexpected);

			// assume that A changes linearly along edges
			// => Integral dx is x length multiplied by average
			// Integral dy is y length multiplied by average

			Integral1 = 0.5*(pTri->cornerptr[ileft]->B.z + pTri->cornerptr[iother]->B.z)*(pTri->cornerptr[iother]->x-pTri->cornerptr[ileft]->x);
			Integral2 = 0.5*(pTri->cornerptr[iright]->B.z + pTri->cornerptr[iother]->B.z)*(pTri->cornerptr[iright]->x-pTri->cornerptr[iother]->x);
			Integral3 = 0.5*(pTri->cornerptr[iright]->B.z + pTri->cornerptr[ileft]->B.z)*(pTri->cornerptr[iright]->x-pTri->cornerptr[ileft]->x);

			Integral1y = 0.5*(pTri->cornerptr[ileft]->B.z + pTri->cornerptr[iother]->B.z)*(pTri->cornerptr[iother]->y-pTri->cornerptr[ileft]->y);
			Integral2y = 0.5*(pTri->cornerptr[iright]->B.z + pTri->cornerptr[iother]->B.z)*(pTri->cornerptr[iright]->y-pTri->cornerptr[iother]->y);
			Integral3y = 0.5*(pTri->cornerptr[iright]->B.z + pTri->cornerptr[ileft]->B.z)*(pTri->cornerptr[iright]->y-pTri->cornerptr[ileft]->y);

			// positive across the top. The three 'integral' are all positive.
			pTri->numerator_x = (twosides_above)?(Integral1+Integral2-Integral3):-(Integral1+Integral2-Integral3);

			// Integral over cell By = MINUS integral along y-projection (Az(right)-Az(left))
				
			// case by case analysis comes down to this:
			if (twosides_above)
			{							// Note that we are counting positive y-difference for things that appear on left side of shape:
				pTri->numerator_y = Integral1y + Integral2y - Integral3y; 
			} else {
				pTri->numerator_y = -Integral1y - Integral2y + Integral3y; 
			};
			// we then divide by area.

		} else {
			
			if (pTri->periodic == 1)
			{
				// map the two unmapped ones (not ileft) to left
				// ileft becomes iright
				// iright becomes iother
				// iother becomes ileft
				
				pTri->cornerptr[ileft]->PopulatePosition(u_right);
				pTri->cornerptr[iright]->periodic_image(u_mid,0,1);
				pTri->cornerptr[iother]->periodic_image(u_left,0,1);
				Bz_left = pTri->cornerptr[iother]->B.z;
				Bz_right = pTri->cornerptr[ileft]->B.z;
				Bz_mid = pTri->cornerptr[iright]->B.z;
			} else {
				// map the unmapped one (iright) to left
				// ileft becomes iother
				// iother becomes iright
				// iright becomes ileft

				pTri->cornerptr[ileft]->PopulatePosition(u_mid);
				pTri->cornerptr[iother]->PopulatePosition(u_right);
				pTri->cornerptr[iright]->periodic_image(u_left,0,1);
				Bz_left = pTri->cornerptr[iright]->B.z;
				Bz_mid = pTri->cornerptr[ileft]->B.z;
				Bz_right = pTri->cornerptr[iother]->B.z;
			};
			
			yexpected = u_left.y + (u_mid.x-u_left.x)*(u_right.y-u_left.y)/(u_right.x-u_left.x);
			twosides_above = (u_mid.y > yexpected);
			
			Integral1 = 0.5*(Bz_left + Bz_mid)*(u_mid.x-u_left.x);
			Integral2 = 0.5*(Bz_right + Bz_mid)*(u_right.x-u_mid.x);
			Integral3 = 0.5*(Bz_right + Bz_left)*(u_right.x-u_left.x);

			Integral1y = 0.5*(Bz_left + Bz_mid)*(u_mid.y-u_left.y);
			Integral2y = 0.5*(Bz_right + Bz_mid)*(u_right.y-u_mid.y);
			Integral3y = 0.5*(Bz_right + Bz_left)*(u_right.y-u_left.y);

			pTri->numerator_x = (twosides_above)?(Integral1+Integral2-Integral3):-(Integral1+Integral2-Integral3);
			// case by case analysis comes down to this:
			if (twosides_above)
			{							// Note that we are counting positive y-difference for things that appear on left side of shape:
				pTri->numerator_y = Integral1y + Integral2y - Integral3y; 
			} else {
				pTri->numerator_y = -Integral1y - Integral2y + Integral3y; 
			};
		};
		
		pTri->numerator_x /= pTri->area;
		pTri->numerator_y /= pTri->area;
		pTri->numerator_x *= c/(FOURPI);
		pTri->numerator_y *= c/(FOURPI);

		++pTri;
	};	
}




void TriMesh::SetupJBEGraphing()
{
	// Set Vertex Temp = J, Vertex B = B

	Vertex * pVertex = X;
	int lenny;
	real denom,totalweight,weight;
	real dummy;
	Triangle * pTri;
	real area;
	//Vector3 vnum;
	//real Tnum;

	static const Tensor3 Anticlockwise3 (cos(FULLANGLE),-sin(FULLANGLE), 0.0,
														sin(FULLANGLE),cos(FULLANGLE), 0.0,
														0.0, 0.0, 1.0);
	static const Tensor3 Clockwise3 (cos(FULLANGLE),sin(FULLANGLE), 0.0,
														-sin(FULLANGLE),cos(FULLANGLE), 0.0,
														0.0, 0.0, 1.0);


	for (long iVert = 0; iVert < numVertices; iVert++)
	{
		lenny = pVertex->triangles.len;
		denom = 0.0;
		totalweight = 0.0;
		
		ZeroMemory(&(pVertex->Temp),sizeof(Vector3));
		ZeroMemory(&(pVertex->B),sizeof(Vector3));
		ZeroMemory(&(pVertex->E),sizeof(Vector3));

		for (int j = 0; j < lenny; j++)
		{
			pTri = (Triangle *)(pVertex->triangles.ptr[j]);
			
			weight = pTri->ReturnAngle(pVertex); // takes acct of periodic & wedge cases.
			area = pTri->area;
			if (pVertex->flags >= 3) weight *= 2.0; // only got half a circle!

			// Now here is a thing.
			// PB triangles store momentum meaning what it is on the left side. !!
			// This means if the triangle is being used on the RHS, then we need to rotate the momentum vector.

			if (
				(pTri->periodic == 0) || 
				((pTri->periodic == 1) && (pVertex == pTri->cornerptr[pTri->GetLeftmostIndex()])) ||
				((pTri->periodic == 2) && (pVertex != pTri->cornerptr[pTri->GetRightmostIndex()]))
			   )
			{
				pVertex->Temp += (weight/area)*q*(pTri->ion.mom - pTri->elec.mom);

				pVertex->B += weight*pTri->B;
				pVertex->E += weight*pTri->E;

			} else {
				Vector3 ionclock,elecclock;
				ionclock = Clockwise3*pTri->ion.mom;
				elecclock = Clockwise3*pTri->elec.mom;
				
				pVertex->Temp += (weight/area)*q*(ionclock - elecclock);
				pVertex->B += weight*(Clockwise3*pTri->B);
				pVertex->E += weight*(Clockwise3*pTri->E);

			};

			totalweight += weight;
		};
		++pVertex;
	};

	// do once pVertex->B are known:

	GetCurlBcOver4Pi(); // Triangle::numerator_x,y,z

	pVertex= X;
	for (long iVert = 0; iVert < numVertices; iVert++)
	{
		lenny = pVertex->triangles.len;
		denom = 0.0;
		totalweight = 0.0;
		
		pVertex->Pressure_numerator_x = 0.0;
		pVertex->Pressure_numerator_y = 0.0;
		pVertex->ion_pm_Heat_numerator = 0.0;

		for (int j = 0; j < lenny; j++)
		{
			pTri = (Triangle *)(pVertex->triangles.ptr[j]);
			
			weight = pTri->ReturnAngle(pVertex); // takes acct of periodic & wedge cases.
			area = pTri->area;
			if (pVertex->flags >= 3) weight *= 2.0; // only got half a circle!

			// Now here is a thing.
			// PB triangles store momentum meaning what it is on the left side. !!
			// This means if the triangle is being used on the RHS, then we need to rotate the momentum vector.

			if (
				(pTri->periodic == 0) || 
				((pTri->periodic == 1) && (pVertex == pTri->cornerptr[pTri->GetLeftmostIndex()])) ||
				((pTri->periodic == 2) && (pVertex != pTri->cornerptr[pTri->GetRightmostIndex()]))
			   )
			{
				pVertex->Pressure_numerator_x += weight*pTri->numerator_x; 
				pVertex->Pressure_numerator_y += weight*pTri->numerator_y;
				pVertex->ion_pm_Heat_numerator += weight*pTri->numerator_z;

			} else {
				Vector3 ionclock,elecclock;
				
				pVertex->Pressure_numerator_x += weight*(Clockwise.xx*pTri->numerator_x+Clockwise.xy*pTri->numerator_y); 
				pVertex->Pressure_numerator_y += weight*(Clockwise.yy*pTri->numerator_y+Clockwise.yx*pTri->numerator_x);
				pVertex->ion_pm_Heat_numerator += weight*pTri->numerator_z;

			};

			totalweight += weight;
		};
		++pVertex;
	};


}


void TriMesh::SetupSigmaJEGraphing()
{
	
	Vertex * pVertex = X;
	int lenny;
	real denom,totalweight,weight;
	real dummy;
	Triangle * pTri;
	real area;

	static const Tensor3 Anticlockwise3 (cos(FULLANGLE),-sin(FULLANGLE), 0.0,
														sin(FULLANGLE),cos(FULLANGLE), 0.0,
														0.0, 0.0, 1.0);
	static const Tensor3 Clockwise3 (cos(FULLANGLE),sin(FULLANGLE), 0.0,
														-sin(FULLANGLE),cos(FULLANGLE), 0.0,
														0.0, 0.0, 1.0);

	// Job of this routine:
	// Set
	
	// pVertex->Temp = J
	// pVertex->E = E
	//pVertex->ion_pm_Heat_numerator
	//pVertex->Pressure_numerator_x,y  =  sigma par, perp, hall
	
	for (long iVert = 0; iVert < numVertices; iVert++)
	{
		lenny = pVertex->triangles.len;
		denom = 0.0;
		totalweight = 0.0;
		
		ZeroMemory(&(pVertex->Temp),sizeof(Vector3));
		ZeroMemory(&(pVertex->E),sizeof(Vector3));
		ZeroMemory(&(pVertex->eps),sizeof(Vector3));
		
		for (int j = 0; j < lenny; j++)
		{
			pTri = (Triangle *)(pVertex->triangles.ptr[j]);
			
			weight = pTri->ReturnAngle(pVertex); // takes acct of periodic & wedge cases.
			area = pTri->area;
			if (pVertex->flags >= 3) weight *= 2.0; // only got half a circle!

			// Now here is a thing.
			// PB triangles store momentum meaning what it is on the left side. !!
			// This means if the triangle is being used on the RHS, then we need to rotate the momentum vector.

			if (
				(pTri->periodic == 0) || 
				((pTri->periodic == 1) && (pVertex == pTri->cornerptr[pTri->GetLeftmostIndex()])) ||
				((pTri->periodic == 2) && (pVertex != pTri->cornerptr[pTri->GetRightmostIndex()]))
			   )
			{
				pVertex->Temp += (weight/area)*q*(pTri->ion.mom - pTri->elec.mom);
				pVertex->E += weight*pTri->E;
				
			} else {
				Vector3 ionclock,elecclock;
				ionclock = Clockwise3*pTri->ion.mom;
				elecclock = Clockwise3*pTri->elec.mom;
				
				pVertex->Temp += (weight/area)*q*(ionclock - elecclock);
				pVertex->E += weight*(Clockwise3*pTri->E);
				
			};
			
			pVertex->eps.x += weight*pTri->sigma_i.xx; // parallel
			pVertex->eps.y += weight*pTri->sigma_i.xy;	// perp
			pVertex->eps.z += weight*pTri->sigma_i.yy;	// Hall
			
			totalweight += weight;
		};
		++pVertex;
	};
	
}

void TriMesh::CalculateTotalGraphingData()
{
	// already have ion.n, elec.T, etc, at vertex

	// populate
	Vertex * pVertex;
	long iVertex;
	real min_lambda = 1.0e100;
	real max_lambda = -1.0;

	pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
		pVertex->coeff_self = pVertex->ion.n+pVertex->neut.n;

		pVertex->eP_Viscous_denominator_x = pVertex->ion.n/pVertex->coeff_self; // use for ionisation colouring
	
		min_lambda = min (pVertex->eP_Viscous_denominator_x, min_lambda);
		max_lambda = max (pVertex->eP_Viscous_denominator_x, max_lambda);

		pVertex->Pressure_numerator_x = (pVertex->ion.n*m_ion*pVertex->ion.v.x
																+ pVertex->neut.n*m_neutral*pVertex->neut.v.x
																+ pVertex->elec.n*m_e*pVertex->elec.v.x)/
																(pVertex->ion.n*m_ion+pVertex->neut.n*m_neutral+pVertex->elec.n*m_e);
		pVertex->Pressure_numerator_y = (pVertex->ion.n*m_ion*pVertex->ion.v.y
																+ pVertex->neut.n*m_neutral*pVertex->neut.v.y
																+ pVertex->elec.n*m_e*pVertex->elec.v.y)/
																(pVertex->ion.n*m_ion+pVertex->neut.n*m_neutral+pVertex->elec.n*m_e);
		
		pVertex->eP_Viscous_denominator_y = (pVertex->ion.n*pVertex->ion.T
																+ pVertex->neut.n*pVertex->neut.T
																+ pVertex->elec.n*pVertex->elec.T)/
																(pVertex->ion.n+pVertex->neut.n+pVertex->elec.n);
		
		
		++pVertex;
	}
	printf("\n\nmin lambda %f max lambda %f \n\n",min_lambda,max_lambda);

}

void TriMesh::ReturnCharge()
{
	Triangle * pTri = T;
	real charge =0.0;
	real netcharge = 0.0;
	real ionmass = 0.0;
	for (long iTri = 0; iTri < numTriangles; iTri++)
	{
		charge += fabs(pTri->ion.mass-pTri->elec.mass);
		netcharge += pTri->ion.mass-pTri->elec.mass;
		ionmass += pTri->ion.mass;
		++pTri;
	};
	printf("\n\ncharge: %1.14E   netcharge: %1.14E  ionmass: %1.14E \n\n",charge,netcharge,ionmass);
};




void GetInsulatorIntercept(Vector2 *result, const Vector2 & x1, const Vector2 & x2)
{
	// find where line x1->x2 crosses r = DEVICE_RADIUS_INSULATOR_OUTER

	// x = x1.x + t(x2.x-x1.x)
	// y = x1.y + t(x2.y-x1.y)
	// x^2+y^2 = c^2

	// (x1.x + t(x2.x-x1.x))^2 + (x1.y + t(x2.y-x1.y))^2 = c^2

	// or, y = x1.y + dy/dx (x-x1.x)

	// x^2 + (x1.y - dy/dx x1.x + dy/dx x)^2 = c^2


	// (x1.x + t(x2.x-x1.x))^2 + (x1.y + t(x2.y-x1.y))^2 = c^2

	// t^2 ( (x2.x-x1.x)^2 + (x2.y - x1.y)^2 ) + 2t (x1.x (x2.x-x1.x) + x1.y (x2.y-x1.y) )
	//    + x1.x^2 + x1.y^2 = c^2
	// t^2 + 2t ( -- ) / (-- ) = (c^2 - x1.x^2 - x1.y^2)/ (-- )
	
	real den = (x2.x-x1.x)*(x2.x-x1.x) + (x2.y - x1.y)*(x2.y - x1.y) ;
	real a = (x1.x * (x2.x-x1.x) + x1.y * (x2.y-x1.y) ) / den;

	// (t + a)^2 - a^2 = (  c^2 - x1.x^2 - x1.y^2  )/den
	
	real root = sqrt( (DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER
							- x1.x*x1.x - x1.y*x1.y)/den + a*a ) ;
	
	real t1 = root - a;
	real t2 = -root - a;
	
	// since this is a sufficient condition to satisfy the circle, this probably means that
	// the other solution is on the other side of the circle.
	
	// Which root is within x1, x2 ? Remember x2 would be t = 1.

	if (t1 > 1.0) 
	{
		if ((t2 < 0.0) || (t2 > 1.0))
		{	
			// This usually means one of the points actually is on the curve.

			// ********************************************************************************************
			// What we really need to do is recognise in our routines that boundary points stay there and the whole tri is reflected.
			// ********************************************************************************************
			
			real dist1 = min(fabs(t1-1.0),fabs(t1));
			real dist2 = min(fabs(t2-1.0),fabs(t2));

			if (dist1 < dist2)
			{
				// use t1
				
				if (dist1 > 0.00000001)
				{
					printf("\n\nError.\n"); 
					getch();
				};
				
				result->x = x1.x + t1*(x2.x-x1.x);
				result->y = x1.y + t1*(x2.y-x1.y);
			} else {
				// use t2
				
				if (dist2 > 0.00000001)
				{
					printf("\n\nError.\n"); 
					printf("t1 = %1.10E , \nt2 = %1.10E , \nx1.x= %1.10E ,\nx1.y = %1.10E ,\nx2.x = %1.10E ,\nx2.y = %1.10E\n",
								t1,t2,x1.x,x1.y,x2.x,x2.y);
					getch();
				};
				
				result->x = x1.x + t2*(x2.x-x1.x);
				result->y = x1.y + t2*(x2.y-x1.y);
			};
		} else {		
			// use t2:
		
			result->x = x1.x + t2*(x2.x-x1.x);
			result->y = x1.y + t2*(x2.y-x1.y);
		};
	} else {
		if (t1 < -1.0e-13) 
		{	
			printf("\n\nError.KL\n"); 
			printf("t1 = %1.10E , \nt2 = %1.10E , \nx1.x= %1.10E ,\nx1.y = %1.10E ,\nx2.x = %1.10E ,\nx2.y = %1.10E\n",
				t1,t2,x1.x,x1.y,x2.x,x2.y);
			getch(); 
		};

		result->x = x1.x + t1*(x2.x-x1.x);
		result->y = x1.y + t1*(x2.y-x1.y);		
	};

#ifdef DEBUG
	if (result->x*result->x + result->y*result->y > 1.000001*DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
	{
		result = result;
	};
	if (result->y < 0.0)
	{
		result = result;
	};
#endif
}


void TriMesh::ConservativeSmoothings(real dt)
{
	long iTri,iVertex;
	Triangle * pTri;
	Vertex * pVertex;
	real OldTotalHeat, Boost;
	real add1,add2,heat1,heat2;
	int iHot[3];
	real rate, expfactor, difference, new_directed_energy;
	
	FILE * debugfile = fopen("visc_errors.txt","a");
	
	static real const EXPFACTOR = 0.8; // move 0.8 of the way towards equilibration
	
	real store_h = h;
	h = dt;
	
	real maxbadness1,maxbadness2;
	
	real t_step = 0.01;				// heat density nT could be 10^6 in principle. Or could just as easily be 10^-2.
	
	real maxvxvx,maxvyvy,maxv,newmaxv,newmaxvxvx,newmaxvyvy;
	int iOldMaxvxvx,iNewMaxvxvx,iOldMaxvyvy,iNewMaxvyvy,iOldMaxv,iNewMaxv;
	real v,vx,vy,vxvx,vyvy;
	
	RecalculateVertexVariables();
	
	GlobalViscousheatingtotal = 0.0;
	
	// Hypo-diffusion of mass:
	// --------------------------
	if (HYPODIFFUSION)
	{
		//RecalculateDisplacementSD(); // store displacement SD as "kappa" for each species

		//Smoothing(SPECIES_ION,VARCODE_MASS);
		//InferMass(SPECIES_ION); // this simply will set mass = e_pm_Heat_denominator
		//// this is conservative automatically.
		//
		//Smoothing(SPECIES_NEUTRAL,VARCODE_MASS);
		//InferMass(SPECIES_NEUTRAL); // this simply will set mass = e_pm_Heat_denominator
		//
		//Smoothing(SPECIES_ELECTRON,VARCODE_MASS);
		//InferMass(SPECIES_ELECTRON); // this simply will set mass = e_pm_Heat_denominator
	};
	
	// Smoothing Momentum and viscous heating:
	
	this->RecalculateVertexVariables(); // Now gets B on vertices also.
	
	for (int species = 0; species < 3; species++)
	{
		Recalculate_NuHeart_and_KappaParallel_OnVertices_And_Triangles(species);
		// Because it's only called within CFLSmoothing for the heat conduction case.
	
		if (species != 2) {
			printf("CFLSmoothing viscosity Species %d\n",species);
			CFLSmoothing(species,VARCODE_V_ALL);
			

		} else {
			// Wrongness:

			// Scrapped electron viscosity for now because it's taking too long.
			
			// Change to an ODE Ohm's Law anyway.

			// Just do a little bit:
			//h = 1.0e-12;

			//printf("CFLSmoothing viscosity Species %d\n",species);
			//CFLSmoothing(species,VARCODE_V_ALL);
			//
			//h = dt;

			// see hsub=1e-19. Seriously.

		};
		
		printf("CFLSmoothing heatcond Species %d\n",species);
		CFLSmoothing(species,VARCODE_HEAT);
	};
	
	fclose(debugfile);
	
	h = store_h;
	
	fp = fopen(FUNCTIONALFILENAME, "a");
	fprintf(fp, " visc %1.10E ",GlobalViscousheatingtotal);
	fclose(fp);
	
	return;
	
	// That's it.
	
}

void TriMesh::AdvancedSmoothing(short species, int varcode, real hSmooth)
{
	
//	static const Tensor2 R6th(cos(TWOPI/6.0),sin(TWOPI/6.0),-sin(TWOPI/6.0),cos(TWOPI/6.0));
/*

	// 0a. Create model of nT or nv on each cell
	// 0b. Create kappa on centroids and vertices

	// For each cell distribute heat, store in numerator_T. [initialise to 0]
	// 1. Divide cell into regions where we map to self or only to one other cell.
	//	a. Know kappa, 1/n at each place on cell. 
	//	b. Infer max radius = 3SD for each subtri, using max variance from cell together with opposite centroid.
	//	c. Would like 1D area to be rectangle preferably, so choose rectangles, that do not overlap
	//	because their corners always lie on the lines from corner to centre.
	//	// Does that make super awkward remainder regions? What is the self-map region now?

	// In practice they would match up EXCEPT for one thing: because that line bisects the corner angle. 
	// The rectangle corner point is where the radius inferred, for that line, only just hits the other edge.
	// It sometimes has to be pulled inward because if this is ACentroid, BCentroid may not be as far towards B.
	// Each side the height of rect is the minimum from the two corner-to-centroid lines.

	// However there is another consideration(?): a short opposing cell?
	// Take radius at edge points: it should not go beyond the edges of the opposing cell.
	// Think that under Delaunay we can get away with not checking.

	// Case that even at centroid we exceed boundaries of this cell: assign whole thing to MC. No big loss of efficiency.

	// 2. Handle areas where there is a fast way of apportioning between 2 cells.
	//	

	// 3. Handle areas that are for MC SDE.
	//	a. Be able to evaluate kappa and 1/n quickly  <--- this itself depends on finding tri & subtri we are in.
	//	b. 

	// What if we draw some circles, more for greater variance, and see how large an arc lies within each dest
	// triangle?
	// Idea for path: each and every Wiener difference, rotate 8 ways. Thus if we do 3 steps we do have to place
	// 8^3 paths. Maybe do 2 arms of 2. 64 to place. Seeking dest tri is quite slow? 3 dot products.
	// Choose timestep of SDE to make typical move smaller than fraction of cell width.
	// Therefore number of moves is not predetermined. But number of arms can be.

	// More stratified sampling: scale Wiener draws and do pdf for relative weight. This leads to a certain
	// number of effective samples, which we divide by to show how on average the heat is apportioned.

	// 64 dests per point. Possibly slower to identify which triangles than to generate them.
	// What to do about that? Can make sure we have good seed for each triangle search. Anything else?
	// Can make list : know a priori where dests can lie based on radius, in many cases. No point searching
	// anywhere else until we searched all dest tris on the list.
	// That doesn't really help: we already have a directed search.
	//
	// Better to be placing lots of points with dot products or to be computing some kind of angles? 
	// Can't see decent way with angles given that there are multiple steps and they ping in and out
	// with Euler-Heun.
	// 

	// BUT TWO SPEED-UPS to search:
	// 1. Remember where we just came from - saves at least 1/3
	// 2. Blue flag vertices: know that the dest is probably nearby a triangle, call it pTriSrc;
	//		then if a pTriSrc vertex is available, we may pick that as the first edge to check.
	//		Not sure how beneficial; can compare speeds with and without.


	// 1D 2-way:
	
	// We def don't want heat transferred to ever be greater than what is available in the cell,
	// so it makes sense to do each side separately instead of looking over an edge to do dT/dt...
	// Each region in a cell should know how much heat it had initially and apportion exactly this 
	// amount to different cells. How much heat comes from triplanar; probability of a cell we
	// should estimate as well as we can.
	// Still could give transferred net = (T1-T0)/dist kappa_edge -- idea, take nT, take var = kappa/n
	// take 1/2 for whether heading in or out, 
	// only normal dimension matters ,
	
	// 

	
	// For MC we should be wary of particulation : amt of heat transfer may be 1/1000 ;
	// do not prefer for it to be sampled as 0,0,0,0,1,0,0,.. 

	// Not a lot we can do about that?



	
	// For each cell distribute heat, store in numerator_T. [initialise to 0]
	// 1. Divide cell into regions where we map to self or only to one other cell.
	//	a. Know kappa, 1/n at each place on cell. 
	//	b. Infer max radius = 3SD for each subtri, using max variance from cell together with opposite centroid.
	//	c. Would like 1D area to be rectangle preferably, so choose rectangles, that do not overlap
	//	because their corners always lie on the lines from corner to centre.
	//	// Does that make super awkward remainder regions? What is the self-map region now?


	// each cell store?

	Triangle * pTri , *pNeigh;	
	vertvars verts_use[3], Verts_c;
	cellvars Vars;
	bool src_periodic;
	int o;
	real nT[3],nT_C;
	long iTri;
	ConvexPolygon cp;
	Vector2 uCorner[3], rectcnr1, rectcnr2, project1, project2, projectedcentroid,
		unit_normal;
	real mod, nmin, kappa_edge,F,cc_dist_to_edge,y,xsrc,xsrc_out,
		nT_left, nT_out,P,P_out,dF_by_dy,target,dF_by_dx,
		n_edge, sidelenfactor, sidelength, factor3;
	real MCheat, Heat_to_send, heat_subtri, heat_rectangle;
	int iSwitch, iInterval, iEdge, numMCs, iMC;
	Vector2 MC[2][3];

	int const NUM_GAUSS_RADII = 3;
	real const GaussRadius[3] = {0.0, 0.5, 1.8668942438};
	real const GaussWeight[3] = {0.156118143459, 0.6, 0.24388185654};
	// Magic numbers chosen to get variance = 1 and kurtosis = 3
	
	
	// pTri->numerator_x,y,z,T are to receive the momentum or heat.
	
	// need to have kappa on centroid, corners I think
	// want n on centroid, corners
	// and nT for source

	// We need to allow sometimes it is discontinuous --- cannot always get mass if we
	// fix the corners.
	// One day, splines.
	// For now, we need to store 4 kappa per triangle.

	// Can work out n, nT when in use; they are not the problem.
	// Except for MC -- when we do need n to be stored over every triangle!!

	// Kappa:
	// Tensor2 pTri->Effect_Exy_displacement_e 
	// xx = 0, xy = 1, yx = 2, yy = centroid

	//printf("Pre-generating kappa for kappa(dest):\n");
	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		pTri->numerator_T = 0.0; // write first for heat. !
		pTri->numerator_x = 0.0;
		pTri->numerator_y = 0.0;
		pTri->numerator_z = 0.0;

		pTri->GetContiguousCentroid(pTri->cc,pTri);
		pTri->RecalculateEdgeNormals(false); 

		++pTri;
	};

	// The rest: only use kappa centroid?
	// Just take T in cell and calc that way ...
	// Need CONTINUOUS KAPPA, 
	// That's the rub.



		// Calculate n, nT triplanar
	//	
	//	if (species == SPECIES_ION)
	//	{
	//		//pVars = &(pTri->ion);
	//		Vars = pTri->ion;
	//		verts_use0 = pTri->cornerptr[0]->ion; // assignment equals, not reference assignment.
	//		verts_use1 = pTri->cornerptr[1]->ion;
	//		verts_use2 = pTri->cornerptr[2]->ion;
	//	} else {
	//		if (species == SPECIES_NEUTRAL)
	//		{
	//			//pVars = &(pTri->neut);
	//			Vars = pTri->neut;
	//			verts_use0 = pTri->cornerptr[0]->neut; // assignment equals, not reference assignment.
	//			verts_use1 = pTri->cornerptr[1]->neut;
	//			verts_use2 = pTri->cornerptr[2]->neut;
	//		} else {
	//			//pVars = &(pTri->elec);
	//			Vars = pTri->elec;
	//			verts_use0 = pTri->cornerptr[0]->elec; // assignment equals, not reference assignment.
	//			verts_use1 = pTri->cornerptr[1]->elec;
	//			verts_use2 = pTri->cornerptr[2]->elec;
	//		};
	//	};				
	//			
	//	src_periodic = (pTri->periodic > 0)?true:false;
	//	if (src_periodic)
	//	{
	//		if (pTri->periodic == 1) {
	//			o = pTri->GetLeftmostIndex();
	//			if (o != 0)	verts_use0.v = Anticlockwise3*verts_use0.v;
	//			if (o != 1)	verts_use1.v = Anticlockwise3*verts_use1.v;
	//			if (o != 2) verts_use2.v = Anticlockwise3*verts_use2.v;				
	//		} else {
	//			o = pTri->GetRightmostIndex();
	//			if (o == 0) verts_use0.v = Anticlockwise3*verts_use0.v;
	//			if (o == 1) verts_use1.v = Anticlockwise3*verts_use1.v;				
	//			if (o == 2) verts_use2.v = Anticlockwise3*verts_use2.v;				
	//		};
	//	};
	//	
	//	Triplanar(&Vars, &verts_use0, &verts_use1, &verts_use2,&Verts_c,
	//		pTri->area,pTri->area,ALL_VARS); // all vars for now...

	////	pTri->Effect_Exy_displacement_e.xx = CalculateScalarKappa(verts_use0.n, verts_use0.T, varcode);
	////	pTri->Effect_Exy_displacement_e.xy = CalculateScalarKappa(verts_use1,n, verts_use1.T, varcode);
	////	pTri->Effect_Exy_displacement_e.yx = CalculateScalarKappa(verts_use2.n, verts_use2.T, varcode);
	////	pTri->Effect_Exy_displacement_e.yy = CalculateScalarKappa(Verts_c.n, Verts_c.T, varcode);

	//	// Not used.

	//	// Used for Euler-Heun step to deal with kappa(destination).

	//	++pTri;
	//};
	//printf("done. Assigning macroscopic: ");

	
	// Ought to set up a fast means of returning interpolated kappa value on each triangle also.
	// Can do that.

	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		// Subtris are formed with centroid.
		
		// Calculate n, nT triplanar, again:

		if (species == SPECIES_ION)
		{
			//pVars = &(pTri->ion);
			Vars = pTri->ion;
			verts_use[0] = pTri->cornerptr[0]->ion; // assignment equals, not reference assignment.
			verts_use[1] = pTri->cornerptr[1]->ion;
			verts_use[2] = pTri->cornerptr[2]->ion;
		} else {
			if (species == SPECIES_NEUTRAL)
			{
				//pVars = &(pTri->neut);
				Vars = pTri->neut;
				verts_use[0] = pTri->cornerptr[0]->neut; // assignment equals, not reference assignment.
				verts_use[1] = pTri->cornerptr[1]->neut;
				verts_use[2] = pTri->cornerptr[2]->neut;
			} else {
				//pVars = &(pTri->elec);
				Vars = pTri->elec;
				verts_use[0] = pTri->cornerptr[0]->elec; // assignment equals, not reference assignment.
				verts_use[1] = pTri->cornerptr[1]->elec;
				verts_use[2] = pTri->cornerptr[2]->elec;
			};
		};
				
		src_periodic = (pTri->periodic > 0)?true:false;
		if (src_periodic)
		{
			if (pTri->periodic == 1) {
				o = pTri->GetLeftmostIndex();
				if (o != 0)	verts_use[0].v = Anticlockwise3*verts_use[0].v;
				if (o != 1)	verts_use[1].v = Anticlockwise3*verts_use[1].v;
				if (o != 2) verts_use[2].v = Anticlockwise3*verts_use[2].v;				
			} else {
				o = pTri->GetRightmostIndex();
				if (o == 0) verts_use[0].v = Anticlockwise3*verts_use[0].v;
				if (o == 1) verts_use[1].v = Anticlockwise3*verts_use[1].v;				
				if (o == 2) verts_use[2].v = Anticlockwise3*verts_use[2].v;				
			};
		};
		Triplanar(pVars, &verts_use[0], &verts_use[1], &verts_use[2], pVerts_c, pTri->area,pTri->area,ALL_VARS);

		if (pTri->periodic == 0) {
			pTri->PopulatePositions(uCorner[0],uCorner[1],uCorner[2]);
		} else {
			pTri->MapLeft(uCorner[0],uCorner[1],uCorner[2]);
		};

		// get back divided out values:
		
		nT[0] = verts_use[0].T*verts_use[0].n; // incorrect of course - it's T that should be modelled as linear.
		nT[1] = verts_use[1].T*verts_use[1].n;
		nT[2] = verts_use[2].T*verts_use[2].n;
		nT_C = Verts_c.T*Verts_c.n;
		
		cc_dist_to_edge = (pTri->cc-uCorner[0]).dot(pTri->edge_normal[2])/pTri->edge_normal[2].modulus(); // distance times edge length
			
		// Vertex kappa is already defined beforehand and is what we'll use for 1D. For now. Because
		// it is less volatile and may be therefore preferable...
		
		
		for (iEdge = 0; iEdge < 3; iEdge++)
		{
			// Note that when we have planes of nT & n we do not have planes of T, never mind kappa. :(
			
			iprev = iEdge - 1; if (iEdge == 0) iprev = 2;
			inext = iEdge + 1; if (iEdge == 2) inext = 0;
			
			pNeigh = pTri->neighbours[iEdge];
			
			heat_subtri = THIRD*(nT_C+nT[iprev]+nT[inext]) // average nT in subtri
							*THIRD*pTri->area; // Is it true that we get 1/3 area ? Think yes.
			
			// For now we treated nT as linear. Once again, that is something we want to change.
			
			if (pNeigh == pTri) {
				
				pTri->numerator_T += heat_subtri;	
				
			} else {

				// Decide for this subtri 3 choices:
				// =================================
				
				// 1. High variance case: Just do MC for this whole tri. Which will involve placing 12-24 triangles.
				// 2. Intermediate case: 1D rectangle, self-map in centre, MC at corners.
				// 3. Low variance case: 1D rectangle considered to extend across whole edge. Self map otherwise.
				
				// Try 1D procedure (this time): 
				// -----------------------------
				// Do just use vertex-based kappa edge, because 
				// _ we will need a continuous kappa, and
				// _ kappa_edge is less volatile as heat flows.
				
				// 17/06/15: In the case that n constant and using kappa_edge,
				// we can show that Kolmogorov gives same answer as Divergence Theorem.
				
				// Let n = n_C + y(n_0-n_C)
				
				kappa_edge = 0.5*(verts_use[iprev].kappa+verts_use[inext].kappa);
				// Note that the vertex kappa has not probably been modified by the
				// Triplanar routine. // (Correct? Check.)

				var_C = 2.0*hSmooth*kappa_edge/Verts_c.n;
				
				if (2.0*2.0*var_C > cc_dist_to_edge) { // testing 2.0 SD
				
					iSwitch = HIGH_VARIANCE; // MC only.. may be faster but don't much prefer in case that var is too low.
				
					MC[0][0] = uCorner[iprev];
					MC[0][1] = pTri->cc;
					MC[0][2] = uCorner[inext];
					numMCs = 1;

				} else {
					mod = pTri->edge_normal[iEdge].modulus();
					unit_normal = pTri->edge_normal[iEdge]/mod;
					nmin = min(verts_use[iprev].n,verts_use[inext].n);
					
					// y^2 = 4 * 2 h kappa / (n = nmin+y(nC-nmin)/(ccdist)) 
					// y^2 (nmin + y ()/() ) = 8h kappa
					
					target = 8.0*hSmooth*kappa_edge;
					factor3 = (Verts_c.n-nmin)/cc_dist_to_edge;

					// use NR - can't be bothered with completing the cube or annoying cubic formula.
					
					do {
						F = y*y*(y+factor3 + nmin) - target;
						dF_by_dy = 3.0*y*y*factor3 + 2.0*y*nmin;
						y -= F/dF_by_dy;
					} while (F/target > 0.0000000001);
					
					//F_low = ylow*ylow*(ylow*factor3 + nmin);
					//F_high = yhigh*yhigh*(yhigh*factor3 + nmin);
	//				for (i = 0; i < 20; i++) {
	//					y = (ylow+yhigh)*0.5;
	//					F = y*y*(y+factor3 + nmin);
	//					if (F < target) {
	//						ylow = y;
	//					} else {
	//						yhigh = y;
	//					};
	//				}; // 0.5^20 = 1/1048576.
				 	
					// Now how large is area at corner?
					

					if (y < 0.05*cc_dist_to_edge) {
						// only 1% of area in 1/10 x 1/10 .. but it would be 10% of what is sent somewhere. Go with 5%.
						iSwitch = LOW_VARIANCE;					
						sidelenfactor = 1.0; // use pTri->edge_normal[iEdge].modulus();					
						numMCs = 0;
					} else {
						iSwitch = INTERMEDIATE;
						sidelenfactor = (1.0-y/cc_dist_to_edge); 
						// if moved to y = 0.3 of the dist to centre inwards, multiplied side length by 0.7
						
						// Intermediate case central triangle:
						// ===================================
						
						rectcnr1 = (1.0-y/cc_dist_to_edge)*uCorner[iprev] + (y/cc_dist_to_edge)*pTri->cc;
						rectcnr2 = (1.0-y/cc_dist_to_edge)*uCorner[inext] + (y/cc_dist_to_edge)*pTri->cc;
						
						cp.Clear();
						cp.add(pTri->cc);
						cp.add(rectcnr1);
						cp.add(rectcnr2); // integrate over this triangle.
						
						cp.IntegrateMass (pTri->cc,uCorner[iprev],uCorner[iNext],
												nT_C,nT[iprev],nT[iNext], 	&heat_central);
						pTri->numerator_T += heat_central;
						
						// Now get rectangle:
						
						projectedcentroid = pTri->cc + cc_dist_to_edge*unit_normal;							
						 
						project1 = (1.0-y/cc_dist_to_edge)*uCorner[iprev] + (y/cc_dist_to_edge)*projectedcentroid;
						project2 = (1.0-y/cc_dist_to_edge)*uCorner[inext] + (y/cc_dist_to_edge)*projectedcentroid;
						
						cp.Clear();
						cp.add(rectcnr1);
						cp.add(rectcnr2);
						cp.add(project2);
						cp.add(project1);
						
						// also want to work out heat of rectangle:
						
						cp.IntegrateMass (pTri->cc,uCorner[iprev],uCorner[iNext],
											nTC,nT[iprev],nT[iNext], &heat_rectangle);
						
						// For low variance case, just add it after as whatever was not sent across the border.
						
						MC[0][0] = uCorner[iprev];
						MC[0][1] = rectcnr1;
						MC[0][2] = project1;
						MC[1][0] = uCorner[inext];
						MC[1][1] = rectcnr2;
						MC[1][2] = project2;
						numMCs = 2;
						
					};

					// 1D heat transfer computation:
					// =============================

					//
					//			Just to get going:
					// 

					// Using kappa_edge only, and n_src
					// Solve cubic: where are we 0.1 SD back from edge?
					
					n_edge = 0.5*(verts_use[iprev].n + verts_use[inext].n);
					// linearly interp between n_edge and Verts_c.n

					// Solve for 0.2 sigma = delta, 0.4 sigma = delta, and so on.

					sidelength = mod*sidelenfactor;

					int const MAX025SDs = 10;
					xsrc_out = 0.0; // 0 sigma = delta <=> delta = 0
					P_out = 0.5; // one Gaussian draw, for now - though it should be SDE path.

					nTproject1 = nT[iprev]+distppn1*(nT[inext]-nT[iprev]);
					nTproject2 = nT[inext]+distppn2*(nT[iprev]-nT[inext]);
					// distppn2 looks back the other way.
					nT_out = 0.5*(nTproject1+nTproject2);

					nTrectcnr1 = // Interpolate within plane ...

					nTrectcnr2 = 
					
					Heat_to_send = 0.0;
					for (iInterval = 1; iInterval <= MAX025SDs; iInterval++) {
						
						stddist = 0.25*(real)iInterval;

						// Solve for xsrc that has stddist SD to edge

						// variance = 2 h kappa_edge / n_src

						// Newton-Raphson:
						// Solving x^2 n_src - stddist^2 2 h kappa_edge = 0
						// x^3 dndx + x^2 n_edge - stddist^2 2 h kappa_edge = 0

						// CAREFUL WITH SIGNS

						xsrc = xsrc_out;

						target = stddist*stddist*2.0*hSmooth*kappa_edge;
						dndx_in = (Verts_c.n-n_edge)/cc_dist_to_edge;
						
						do {
							F = xsrc*xsrc*(n_edge + xsrc*dndx_in) - target;
							dF_by_dx = 3.0*xsrc*xsrc*dndx_in + 2.0*xsrc*n_edge;
							xsrc -= F/dF_by_dx; 					
						} while (fabs(F/target) > 0.00000000001);
						
						// Work out P((x>0)|xsrc):
						// Because variance is constant, given src, it's just Gaussian cdf.
						
						P = Ncdf_stored[i];

						// let's admit, it would be faster to do calls to Ncdf, with intervals that are fixed...

						// FOR NOW still assuming nT is linear - though this is wrong approach.
						// So integrate 2 lines against each other:

						// P from 'P' to 'P_out', x from 'xsrc' to 'xsrc_out', nT from nT_left to nT_out

						// Important --- take average nT along sidelength						
						// infer nT(point) for both ends of line :
						
						nT1 = //Interpolate(nT[iprev], nT_C, nT[inext],
								//		  uCorner[iprev], pTri->cc, uCorner[inext],
								//		  project1 - xsrc*unit_normal);
							// can do as follows: store values at corners of rectangle and interpolate in 1D:
							  nTproject1 + (xsrc/y)*(nTrectcnr1-nTproject1);
						nT2 = nTproject2 + (xsrc/y)*(nTrectcnr2-nTproject2);
						nT_left = 0.5*(nT1+nT2);
						
						IntegralPnT = (xsrc-xsrc_out)*THIRD*(nT_out*P_out + nT_left*P + 0.5*(nT_out*P + nT_left*P_out));
					
						IntegralnT = (xsrc-xsrc_out)*0.5*(nT_out + nT_left);
						Heat_to_send += sidelength * IntegralPnT;
						
						// Check remainder from this rect was positive: by preference, do IntegrateMass to check values.
						
						if (IntegralnT - IntegralPnT < 0.0) {
							printf("error.(IntegralnT - IntegralPnT < 0.0)\n");
							getch();
						};

						xsrc_out = xsrc;
						P_out = P;
						nT_out = nT_left;

						// Careful -- only guaranteed that it extends back to centre;
						// if we go back beyond centre, stop here:

						if (xsrc > cc_dist_to_edge) break;
						// CAREFUL... see about what heat added to self.


					}; // next interval

					pNeigh->numerator_T += Heat_to_send;

					if (iSwitch == LOW_VARIANCE) {
						// add remainder to self:
						//
						//
						if (heat_subtri > Heat_to_send) {
							pTri->numerator_T += heat_subtri - Heat_to_send;
						} else {
							printf("sent more heat than was present LV. \n");
							getch();
						};
					} else {
						
						if (heat_rectangle > Heat_to_send) {
							pTri->numerator_T += heat_rectangle - Heat_to_send;
						} else {
							printf("sent more heat than was present Intermed. \n");
							getch();
						};
						
					};
					
				}; // whether high variance case.
				
				
				// MC Procedure: work out, in general, X_T(w) each corner and use this to send triangle
				// stretched for each path w(t). Or to begin with, one Euler-Heun Gaussian step.
				
				// Impeccable logic for sending distorted tris according to same w at each corner: we could chunk
				// up tri into particles and do same thing, send each with same set of Wiener paths. This gives same result.
				// As long as we are allowed to tween the X_T(w), which arguably we are.
				
				// If intermediate case, need to do twice. If high, need to do once:
				for (iMC = 0; iMC < num_MCs; iMC++)
				{
					cp.Clear();
					cp.add(MC[iMC][0]);
					cp.add(MC[iMC][1]);
					cp.add(MC[iMC][2]);
					cp.IntegrateMass(pTri->cc, uCorner[iprev], uCorner[inext],
						nT_C,nT[iprev],nT[inext], & MCheat );


					// Create destinations at each corner.

					// For now, do just 1 Euler-Heun step.
					// Take set of radii weighted to represent 1D Gaussian.
					// Each radius (e.g. = 0.2 sigma) is found according to [  kappa(src)+kappa(dest) / nsrc ]
					// Take 6 rotations, each radius tri set is randomly rotated.
					// r = 0 is always one of the radii.
					// Take +- 0.6 sigma, +- 1.2 sigma, +- 1.8 sigma
					// Weights to get E[X^2] -- work out these in advance -- ?

					// There are many ways of doing it; this is just one of them for now.

					
					
					// NOTE: StdRadii[0] = 0
					// GaussWeight[0] is then defined accordingly.

					// First add to own triangle, what we get with the sample W(t) = 0 :
					p = GaussWeight[0]; 
					pTri->numerator_T += MCheat*p;
					// May check total contributed heat for debug..

					for (iRadius = 1; iRadius < NUM_GAUSS_RADII; iRadius++) {
		
						GaussRadius = StdRadii[iRadius];
						theta = UniformDraw()*TWOPI;
						Use.x = cos(theta)*GaussRadius;
						Use.y = sin(theta)*GaussRadius;

						p = GaussWeight[iRadius]/6.0; // fade as we go out.

						for (iRotate = 0; iRotate < 6; iRotate++)
						{
							W = R6th*Use; // rotate 6 times
							
							for (iCorner = 0; iCorner < 3; iCorner++)
							{							
								// Notice that we pretty much require kappa to be continuous; use kappa vertex
								// from vertex T calcs, and kappa at centroids.

								InitialSD = sqrt(hUse*2.0*kappasrc[iCorner]/nsrc[iCorner]);				
								XHeun = Xsrc + Use*InitialSD;
								kappaHeun = GetScalarKappa(XHeun,pTri);

								X[iCorner] = Xsrc + Use*sqrt(hUse*(kappasrc[iCorner] + kappaHeun)/nsrc[iCorner]);

								// let's consider where this iteration might really be taking us. 
								// Would be nice to actually find the kappa dest where it more-or-less holds.

								// Let's say kappa changes fairly slowly, then XHeun is not a bad approximation;
								// then kappanext is a better approximation to kappa(dest)

								// Pick best of 3?
								// initial SD -> x	// Then kappadest(x) + kappasrc -> x2	// kappadest(x2) + kappasrc -> x3

							}; // next corner

							// Now place the triangle 

							// First squeeze up nT plane

							cp.Clear();
							cp.add(X[0]);
							cp.add(X[1]);
							cp.add(X[2]);							
							factor = p * MCarea/cp.GetArea();
							
							// must affect verts with factor:
							tempvvars0.n *= factor;
							tempvvars1.n *= factor;
							tempvvars2.n *= factor;

							SendAllMacroscopicPlanarTriangle(
								cp,
								pTri, 
								int src_periodic,
								&tempvvars0, &tempvvars1, &tempvvars2,
								0, // not used
								CODE_NUMT);

							// Detect periodic: how?
							
							
						}; // next rotation

					}; // next radius 

				
					// Note that while changing to SDE trajectory with multiple steps is important, it may pale
					// by comparison with the effect of changing T on kappa over time.
					// Small timestep diffusion reveals rate of change of kappa. -- ? Future improvements.				
				
				}; // next MC region.


				// 1D & MC:

				// Prefer to use [kappa average over move / nsrc]
				// -> more sensible behaviour than if use endpoints only. Halfway to SDE solutions.
								
				// We don't do dT/dx. But should worry about whether getting the appropriate thing in the right limit.
				// Can show it if n is constant.

			}; // whether pNeigh == pTri

		} // next iEdge
		
		


		// bollox:


		
		// Use same Wiener paths, with rotations, for each one. But the expensive part is
		// to locate points and evaluate kappa. What to do about that? Can't do anything much except take
		// fewer segments per path. But we should not be leaping cells.
		
		// Calculate appropriate timestep for paths (how many segments in hSmooth/2):

		SD = sqrt(hSmooth*kappamax/nmin); // cancel 1/2
		cellmindelta = min side length of this cell;

		//SD will be sqrt(hUse * kappamax / nmin)
		// want <= cellmindelta*0.5;
		// (hSmooth/numSegs) * kappamax/nmin < cellmindelta^2*0.25
		// 4 hSmooth * kappamax/(nmin*cellmindelta)  <  numSegs
		numSegs = (int)(4.0*hSmooth*kappamax/(nmin*cellmindelta*cellmindelta))+1;
		hUse = hSmooth*0.5/(real)numSegs;
		// Hopefully numSegs is generally 1, at most 2 -- or we should survey it and reduce hSmooth. ?

		// Always do in two halves to use two arms.
		// Get sequence of Gaussian draws:

		for (iStep = 0; iStep < numSegs; iStep++)
		{
			GaussDraw_MT(& (Wienerdiff[iStep]));
		};
		// Running Wiener paths is kind of taking precision beyond accuracy: for instance, ignoring
		// the rate of change of kappa, although kappa ppnl= T^2.5. If there are features of kappa that
		// need to be taken account of, they are not static anyway.
		
		// better just to take a couple of Gaussian draws, and rotate.
		
		// Then how to apply information for sending heat from triangle?

		// Can go further: can we just expand and know how much goes into some different subregions?

		// Split out source plane into 3 basic planes. (y,0,0).
		
		
		
		// Note: Stop worrying and love the things that can be -soon- put on GPU.
		// Get this simulation to work soon -> start porting.
		// How about this - do however, then get GPU going, then can broaden horizons.
		
		
		++pTri;
	};
	
	// Now round up the accumulated numerator_T

	pTri= T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		if (species == SPECIES_ION) {
			pTri->ion.heat = pTri->numerator_T; // ?
		} else {
			// ?
		};

		++pTri;
	};
*/

}


void TriMesh::CFLSmoothing(short species, int varcode)
{
	real m_s, energy;
	Vector2 flux_px, flux_py, flux_pz;
	real Tcorner1, Tcorner2;
	real maxabschange;
	real T_change;
	real width,height;
	long iVertex,iTri;
	Triangle * pTri, * pTriSeed;
	Vertex * pVertex;
	bool fail_n_tri;
	real over_nPlanes, factor;
	vertvars * pvertvars0, * pvertvars1, *pvertvars2; // vertex data of original
	cellvars * pVars;
	Vector3 vsrc,vdest, momflow_outward;
	real our_mass, nu_heart_sq;
	Vector3 vneigh0,vneigh1,vneigh2,v_existing,v_change,putative_v,v_lowest,v_highest;
	real maxchange;
	static real const qoverMc = q/(m_i*c);
	real visc_coeff;

	int iNext,iPrev;
	vertvars * pvertvarsNext, * pvertvarsPrev;
	real kappa_edge_times_two;						// careful ....
	real normaldist_src, normaldist_dest,n_src,n_dest,diffusivity_edge_times_two,SD_provisional,SD;
	Triangle * pTriDest;
	int flag_normaldist_against_ins;
	Vector2 u[3];
	Vertex * pOppVert;
	int iOpposite;
	real variance, maxdist,sidelength,Tsrc,Tdest;
	cellvars * pVarsDest;
	real delta, dTbydx, heatflow_outward,T_avg,T_existing,putative_T;
	real eta;
	real T_lowest,T_highest,Tneigh0,Tneigh1,Tneigh2,Thighworst,Tlowworst;
	vertvars * pvv1, * pvv2;

	real normaldist, gradient, intercept,rsq;

		Vertex *pCorner1, *pCorner2;
		real shoelace;
		Vector2 gradT, uNeigh, pos1, pos2, cent, flux;
		real kappa_parallel, nu_heart;

		Tensor2 kappa; // note that only the xy-part of kappa is here used

		Vector3 B, omega, vcorner1, vcorner2,  dvbydx, dvbydy;

		Vector3 vrel;
		Vector2 vuse;
		real nTuse, ratio;

	long iWorst;
	real Tworst, vworst, ROCworst, nworst;
	int dimworst;


	static real const DISTANCE_FACTOR_CFL = 0.2; // allow SD to be 1/5 of normal distance, inwards and outwards
	// not used I think
	
	static real const ABSOLUTE_THRESHOLD_VELOCITY_CHANGE = 2.0e4; 
	static real const MAX_CHANGE_PPN_V = 0.4;
	
	static real const MAX_PPN_CHANGE_T = 0.4; // do we get oscillations because it doesn't get close to eqm?
	static real const ABS_THRESHOLD_T_UPWARD_CHANGE = 4.0e-14;


	bool completed_timestep = false;

	real time_remaining = h;

	pTri = T;
	for (iTri = 0; iTri < numTriangles; iTri++)
	{
		pTri->RecalculateEdgeNormalVectors(false); // why true?
		++pTri;
	};

	// This way is different -- see jibboleth.lyx -- 
	// we shall just let the heat flow on an edge be dT/dx_normal edgelength kappa h


	do	// do substeps until we reach the end of the timestep
	{

		if (varcode == VARCODE_HEAT)
		{
			// 1. Set kappa !
			this->RecalculateVertexVariables();		// needed for RecalculateKappaOnVertices
			this->Recalculate_NuHeart_and_KappaParallel_OnVertices_And_Triangles(species);
			
			// Might well be where most of our time actually ends up spent.
			// -- almost without a doubt.
			// But on GPU, won't care.

			
			// We should avoid recalculating ln Lambda too frequently.

		};

		// {___________________________________________________}
		//			    2. Estimate viable substep length eta
		// {___________________________________________________}

		eta = time_remaining*2.0; // > time_remaining
		
		
		if (varcode == VARCODE_HEAT)
		{
			pTri = T;
			for (iTri = 0; iTri < numTriangles; iTri++)
			{
				pTri->numerator_T = 0.0;
				pTri++;
			};
		} else {
			// for momentum, pTri->numerator_T accumulates the viscous heating or cooling?
			// more easily could just store the kinetic energy beforehand... ?	// that is cleaner
			// still, do not affect pTri->numerator_T

			pTri = T;
			for (iTri = 0; iTri < numTriangles; iTri++)
			{
				pTri->numerator_x = 0.0;
				pTri->numerator_y = 0.0;
				pTri->numerator_z = 0.0;
				pTri->numerator_T = 0.0;
				pTri++;
			};
		};
		
		
		pTri = T;
		for (iTri = 0; iTri < numTriangles; iTri++)
		{
			if (species == SPECIES_ION) pVars = &(pTri->ion);
			if (species == SPECIES_NEUTRAL) pVars = &(pTri->neut);
			if (species == SPECIES_ELECTRON) pVars = &(pTri->elec);

			// use pTri->numerator_T for RATE OF CHANGE OF nT 
			
			for (int iEdge = 0; iEdge < 3; iEdge++)
			{
				// First get SD applicable, pretty much as before.
				// Should probably parcel off into a routine - no reason to do otherwise.

				pTriDest = pTri->neighbours[iEdge];

				if (pTriDest != pTri)
				{
					if (species == SPECIES_ION) pVarsDest = &(pTriDest->ion);
					if (species == SPECIES_NEUTRAL) pVarsDest = &(pTriDest->neut);
					if (species == SPECIES_ELECTRON) pVarsDest = &(pTriDest->elec);
					
					if (pTri->periodic > 0) {
						pTri->MapLeft(u[0],u[1],u[2]);
					} else {
						pTri->PopulatePositions(u[0],u[1],u[2]);
					};
									
					iNext = iEdge+1;
					if (iNext == 3) iNext = 0;
					iPrev = iEdge-1;
					if (iPrev == -1) iPrev = 2;
					
					pCorner1 = pTri->cornerptr[iNext];
					pCorner2 = pTri->cornerptr[iPrev];
					pos1 = u[iNext];
					pos2 = u[iPrev];
					pTriDest->GenerateContiguousCentroid(&uNeigh,pTri);
					pTri->GenerateContiguousCentroid(&cent,pTri);
					
					B = 0.25*(pTri->B + pTriDest->B + pCorner1->B + pCorner2->B);

					if (species == SPECIES_ION)
					{
						kappa_parallel = 0.25*(pTri->scratch[3] + pTriDest->scratch[3] +
							pCorner1->ion.kappa + pCorner2->ion.kappa);
						nu_heart = 0.25*(pTri->scratch[2] + pTriDest->scratch[2] + 
							pCorner1->ion.nu_Heart + pCorner2->ion.nu_Heart);

						
						pvv1 = &(pCorner1->ion);
						pvv2 = &(pCorner2->ion);
					} else {
						if (species == SPECIES_NEUTRAL)
						{
							kappa_parallel = 0.25*(pTri->scratch[1] + pTriDest->scratch[1] +
								pCorner1->neut.kappa + pCorner2->neut.kappa);
							nu_heart = 0.25*(pTri->scratch[0] + pTriDest->scratch[0] +
								pCorner1->neut.nu_Heart + pCorner2->neut.nu_Heart);
							
							pvv1 = &(pCorner1->neut);
							pvv2 = &(pCorner2->neut);
						} else {								
							kappa_parallel = 0.25*(pTri->scratch[5] + pTriDest->scratch[5] +
								pCorner1->elec.kappa + pCorner2->elec.kappa);
							nu_heart = 0.25*(pTri->scratch[4] + pTriDest->scratch[4] +
								pCorner1->elec.nu_Heart + pCorner2->elec.nu_Heart);
														
							pvv1 = &(pCorner1->elec);
							pvv2 = &(pCorner2->elec);
						};
					};

					if (varcode == VARCODE_HEAT)
					{
						

						// estimate grad T:

						Tsrc = pVars->heat / pVars->mass;
						Tdest = pVarsDest->heat / pVarsDest->mass;
						Tcorner1 = pvv1->T;
						Tcorner2 = pvv2->T;

						shoelace = uNeigh.x*(pos2.y-pos1.y)
								+ pos2.x*(cent.y-uNeigh.y)
								+ cent.x*(pos1.y-pos2.y)
								+ pos1.x*(uNeigh.y-cent.y);

						gradT.x = (	Tdest*(pos2.y-pos1.y)
								+	Tcorner2*(cent.y-uNeigh.y)
								+	Tsrc*(pos1.y-pos2.y)
								+	Tcorner1*(uNeigh.y-cent.y)
									)/shoelace;

						gradT.y = (	Tdest*(pos1.x-pos2.x)
								+	Tcorner2*(uNeigh.x-cent.x)
								+	Tsrc*(pos2.x-pos1.x)
								+	Tcorner1*(cent.x-uNeigh.x)
									)/shoelace;


						// Create kappa tensor:

						nu_heart_sq = nu_heart*nu_heart;

						if (species == SPECIES_ION) 
						{
							omega = qoverMc*B;

							factor = kappa_parallel / (nu_heart_sq + omega.dot(omega));
							// Upsilon + 
							kappa.xx = factor*(nu_heart_sq + omega.x*omega.x);
							kappa.xy = factor*(omega.x*omega.y + nu_heart*omega.z);
							kappa.yx = factor*(omega.x*omega.y - nu_heart*omega.z);
							kappa.yy = factor*(nu_heart_sq + omega.y*omega.y);
						} else {
							if (species == SPECIES_ELECTRON) 
							{
								omega = qovermc*B;

								factor = kappa_parallel / (nu_heart_sq + omega.dot(omega));

								// Upsilon - 

								kappa.xx = factor*(nu_heart_sq + omega.x*omega.x);
								kappa.xy = factor*(omega.x*omega.y - nu_heart*omega.z);
								kappa.yx = factor*(omega.x*omega.y + nu_heart*omega.z);
								kappa.yy = factor*(nu_heart_sq + omega.y*omega.y);

								// Hall direction basically vertical; unimportant!

							} else {
								
								kappa = ID2x2*kappa_parallel;
							};
						};
						// Compute flux vector

						flux = kappa*gradT;

						if (species == SPECIES_ELECTRON) {
							// Thermoelectric heat flux:

							// estimate relative v:

							vrel = (pTri->elec.mom + pTriDest->elec.mom) / (pTri->elec.mass + pTriDest->elec.mass) - (pTri->ion.mom + pTriDest->ion.mom)/(pTri->ion.mass + pTriDest->ion.mass);
							nTuse = (pTri->elec.heat + pTriDest->elec.heat)/(pTri->area+pTriDest->area);
							ratio = 0.25*(pTri->scratch[6] + pTriDest->scratch[6] +
								pCorner1->epsilon + pCorner2->epsilon);
											
							//factor = 1.5*(nu_ei_bar/nu_heart)/(nu_heart_sq + omega.dot(omega));
							factor = 1.5*ratio/(nu_heart_sq + omega.dot(omega));
							vuse.x = factor*
								( (nu_heart_sq + omega.x*omega.x)*vrel.x
								+ (omega.x*omega.y - nu_heart*omega.z)*vrel.y
								+ (omega.x*omega.z + nu_heart*omega.y)*vrel.z);

							vuse.y = factor*
								( (omega.x*omega.y + nu_heart*omega.z)*vrel.x
								+ (nu_heart_sq + omega.y*omega.y)*vrel.y
								+ (omega.y*omega.z - nu_heart*omega.x)*vrel.z
								);
						
							flux += vuse*nTuse; 

						};



						heatflow_outward = -flux.dot(pTri->edge_normal[iEdge]);

						// rate of heat energy transfer across boundary
						// dT/dt = 2/3n [sum of flowrates]
						// our "ion.heat" = integral nT so is affected how?
						// = 2/3 heat energy in cell
						// T = ion.heat/ion.mass
						
						pTriDest->numerator_T += heatflow_outward;
						pTri->numerator_T -= heatflow_outward;	
						
					} else {


						// estimate grad v:

						vsrc = pVars->mom / pVars->mass;
						vdest = pVarsDest->mom / pVarsDest->mass;
						vcorner1 = pvv1->v;
						vcorner2 = pvv2->v;

						shoelace = uNeigh.x*(pos2.y-pos1.y)
								+ pos2.x*(cent.y-uNeigh.y)
								+ cent.x*(pos1.y-pos2.y)
								+ pos1.x*(uNeigh.y-cent.y);
						
						dvbydx = (	vdest*(pos2.y-pos1.y)
								+	vcorner2*(cent.y-uNeigh.y)
								+	vsrc*(pos1.y-pos2.y)
								+	vcorner1*(uNeigh.y-cent.y)
									)/shoelace;
						
						dvbydy = (	vdest*(pos1.x-pos2.x)
								+	vcorner2*(uNeigh.x-cent.x)
								+	vsrc*(pos2.x-pos1.x)
								+	vcorner1*(cent.x-uNeigh.x)
									)/shoelace;
						
						// dvbydy.x = dvx / dy

						// For now, do isotropic viscosity. Revisit.

						// Factor from thermal conductivity depends on species.

						if (species == SPECIES_ION) 
						{
							visc_coeff = kappa_parallel*0.96/3.9;
							// ratios from Braginskii B=0 case
							m_s = m_i;
						};
						if (species == SPECIES_ELECTRON)
						{
							visc_coeff = kappa_parallel*0.73/3.2;
							// ratios from Braginskii B=0 case
							m_s = m_e;
						};
						if (species == SPECIES_NEUTRAL)
						{
							visc_coeff = kappa_parallel*TWOTHIRDS; // traditional...
							m_s = m_n;
						};
						// our visc_coeff = Formulary ita / m_s
						
						flux_px.x = dvbydx.x * visc_coeff;
						flux_px.y = dvbydy.x * visc_coeff; 
						flux_py.x = dvbydx.y * visc_coeff;
						flux_py.y = dvbydy.y * visc_coeff;
						flux_pz.x = dvbydx.z * visc_coeff;
						flux_pz.y = dvbydy.z * visc_coeff;

						momflow_outward.x = -pTri->edge_normal[iEdge].dot(flux_px);
						momflow_outward.y = -pTri->edge_normal[iEdge].dot(flux_py);
						momflow_outward.z = -pTri->edge_normal[iEdge].dot(flux_pz);

						// For now, explicitly prohibit moves that send nv from lower to higher v
						if (momflow_outward.x*(vdest.x-vsrc.x) > 0.0)
							momflow_outward.x = 0.0;
						if (momflow_outward.y*(vdest.y-vsrc.y) > 0.0)
							momflow_outward.y = 0.0;
						if (momflow_outward.z*(vdest.z-vsrc.z) > 0.0)
							momflow_outward.z = 0.0;

						// These moves seem to come about when vertices are the thing that dominates
						// the grad v calculation. Generally momentum should move the right way for 
						// viscous heating not cooling.

						// Magnetic should hopefully not change that a lot -- but Coulombic part might??

						energy = -m_s*momflow_outward.dot(vdest-vsrc);
						// Now MUST be positive!

						pTriDest->numerator_x += momflow_outward.x;
						pTriDest->numerator_y += momflow_outward.y;
						pTriDest->numerator_z += momflow_outward.z;
						pTri->numerator_x -= momflow_outward.x;
						pTri->numerator_y -= momflow_outward.y;
						pTri->numerator_z -= momflow_outward.z;
				
						pTriDest->numerator_T += 0.5*energy;
						pTri->numerator_T += 0.5*energy; // share heat change between both cells.

						// or just give heat energy mostly to the one that is losing kinetic energy.

					}; // whether heat conduction or viscosity

				};// whether its own neighbour

			}; // which edge
			pTri++;
		};

		// ===================================================================
		// Want eta such that the change of T = heat/mass in cell will not take us 
		// far the other side of the average of neighbours, or reduce / increase by > 10%
		// Now go through: in 1 pass, decide what step (not more than our existing eta) is viable.
		// ===================================================================

		if (varcode == VARCODE_HEAT)
		{
			pTri = T;
			for (iTri = 0; iTri < numTriangles; iTri++)
			{
				if (species == SPECIES_ION) {
					our_mass = pTri->ion.mass;
					Tneigh0 = pTri->neighbours[0]->ion.heat / pTri->neighbours[0]->ion.mass;
					Tneigh1 = pTri->neighbours[1]->ion.heat / pTri->neighbours[1]->ion.mass;
					Tneigh2 = pTri->neighbours[2]->ion.heat / pTri->neighbours[2]->ion.mass;
					// left in for debug only.

					T_existing = pTri->ion.heat/pTri->ion.mass;
				} else {
					if (species == SPECIES_NEUTRAL) {
						our_mass = pTri->neut.mass;
						Tneigh0 = pTri->neighbours[0]->neut.heat / pTri->neighbours[0]->neut.mass;
						Tneigh1 = pTri->neighbours[1]->neut.heat / pTri->neighbours[1]->neut.mass;
						Tneigh2 = pTri->neighbours[2]->neut.heat / pTri->neighbours[2]->neut.mass;		
						T_existing = pTri->neut.heat/pTri->neut.mass;			
					} else {
						our_mass = pTri->elec.mass;					
						Tneigh0 = pTri->neighbours[0]->elec.heat / pTri->neighbours[0]->elec.mass;
						Tneigh1 = pTri->neighbours[1]->elec.heat / pTri->neighbours[1]->elec.mass;
						Tneigh2 = pTri->neighbours[2]->elec.heat / pTri->neighbours[2]->elec.mass;		
						T_existing = pTri->elec.heat/pTri->elec.mass;			
					};
				} ;
		

				T_change = eta*TWOTHIRDS*pTri->numerator_T/our_mass; 
				

				if ((T_change > 1.0e-13*T_existing) || (T_change < -1.0e-13*T_existing)) 
					// note: T_change == 0 is not something we want to handle, for whatever reason.
				{
					T_highest = max(max(Tneigh0,Tneigh1),Tneigh2);
					T_lowest = min(min(Tneigh0,Tneigh1),Tneigh2);				
					// Keeping those in for debug only.


					putative_T = T_existing + T_change;

					if (T_existing > putative_T)
					{
						// Two lower limits for T decreasing that we are interested in:
						// . T should not decrease by more than 10%
						// . If we cross average, let the new distance from average be half as much at most
						if (putative_T < T_existing*(1.0-MAX_PPN_CHANGE_T))
						{
							// want T_existing + eta*pTri->numerator_T/mass = 0.9*T_existing
							// eta = mass*-0.1*T_existing/pTri->numerator_T
							eta = -(MAX_PPN_CHANGE_T)*T_existing*our_mass/pTri->numerator_T;

							putative_T = (1.0-MAX_PPN_CHANGE_T)*T_existing;

							iWorst = iTri;
							Tworst = T_existing;
							ROCworst = pTri->numerator_T/our_mass;
							nworst = our_mass/pTri->area;
							Thighworst = T_highest;
							Tlowworst = T_lowest;
						};
						
						// I reckon this actually is adequate alone. 
						// For a spike of heat, this means we can spread it out how fast?
						// 1.5^20 = 3325. in 20 steps.
						// If we get 1e-17 that is because the density there is super low and it wants to spread
						// across the whole thing. Whatever method, it must always be able to do that.
						// But as T decreases so will kappa.
						
						// I do suspect eventually we hit high T, low n so that everything wants to spread out
						// at a ridiculous rate and spread across 1cm if it were allowed to. This happens inside the halo. 
						// But as long as T profile is near equilibrium it might be okay - we just ask that the actual ROC is small.


					} else {
						if (putative_T > T_existing*(1.0+MAX_PPN_CHANGE_T) + ABS_THRESHOLD_T_UPWARD_CHANGE)
						{
							eta = (MAX_PPN_CHANGE_T*T_existing + ABS_THRESHOLD_T_UPWARD_CHANGE)*our_mass/pTri->numerator_T;
							putative_T = (1.0+MAX_PPN_CHANGE_T)*T_existing;

							
							iWorst = iTri;
							Tworst = T_existing;
							ROCworst = pTri->numerator_T/our_mass;
							nworst = our_mass/pTri->area;
							Thighworst = T_highest;
							Tlowworst = T_lowest;
						};
						
					};
				};
				++pTri;
			};
				
			printf("heat iTri %d T %1.8E ROC %1.8E n %1.6E Thigh %1.6E Tlow %1.6E\n",iWorst, Tworst, ROCworst, nworst,
				Thighworst,Tlowworst);

			if (eta > time_remaining) 
			{
				completed_timestep = true;
				eta = time_remaining;
			} else {
				completed_timestep = false;
			};
			printf("eta used = %1.5E  time_remaining = %1.5E\n",eta,time_remaining);
			

			// 3. Perform the step!
			
			switch(species)
			{
			case SPECIES_ION:
				pTri = T;
				for (iTri = 0; iTri < numTriangles; iTri++)
				{
					pTri->ion.heat += TWOTHIRDS*eta*pTri->numerator_T;
					++pTri;
				};
				break;
			case SPECIES_NEUTRAL:
				pTri = T;
				for (iTri = 0; iTri < numTriangles; iTri++)
				{
					pTri->neut.heat += TWOTHIRDS*eta*pTri->numerator_T;
					++pTri;
				};
				break;
			case SPECIES_ELECTRON:
				pTri = T;
				for (iTri = 0; iTri < numTriangles; iTri++)
				{
					pTri->elec.heat += TWOTHIRDS*eta*pTri->numerator_T;
					++pTri;
				};
				break;
			};
		} else {
			if (varcode == VARCODE_V_ALL)
			{
				// We'll apply the same constraints for every dimension.
				// It's at most a 3x worsening if we do all at once.
				pTri = T;
				for (iTri = 0; iTri < numTriangles; iTri++)
				{
					if (species == SPECIES_ION) {
						our_mass = pTri->ion.mass;
						vneigh0 = pTri->neighbours[0]->ion.mom / pTri->neighbours[0]->ion.mass;
						vneigh1 = pTri->neighbours[1]->ion.mom / pTri->neighbours[1]->ion.mass;
						vneigh2 = pTri->neighbours[2]->ion.mom / pTri->neighbours[2]->ion.mass;
						v_existing = pTri->ion.mom/pTri->ion.mass;
					} else {
						if (species == SPECIES_NEUTRAL) {
							our_mass = pTri->neut.mass;
							vneigh0 = pTri->neighbours[0]->neut.mom / pTri->neighbours[0]->neut.mass;
							vneigh1 = pTri->neighbours[1]->neut.mom / pTri->neighbours[1]->neut.mass;
							vneigh2 = pTri->neighbours[2]->neut.mom / pTri->neighbours[2]->neut.mass;	
							v_existing = pTri->neut.mom/pTri->neut.mass;				
						} else {
							our_mass = pTri->elec.mass;					
							vneigh0 = pTri->neighbours[0]->elec.mom / pTri->neighbours[0]->elec.mass;
							vneigh1 = pTri->neighbours[1]->elec.mom / pTri->neighbours[1]->elec.mass;
							vneigh2 = pTri->neighbours[2]->elec.mom / pTri->neighbours[2]->elec.mass;	
							v_existing = pTri->elec.mom/pTri->elec.mass;				
						};
					} ;
			
					v_change.x = eta*pTri->numerator_x/our_mass;	
					
					if ((fabs(v_change.x)-ABSOLUTE_THRESHOLD_VELOCITY_CHANGE)/fabs(v_existing.x) > MAX_CHANGE_PPN_V)
					{
						maxabschange = fabs(v_existing.x)*MAX_CHANGE_PPN_V + ABSOLUTE_THRESHOLD_VELOCITY_CHANGE;
						maxchange = (v_change.x > 0.0)?maxabschange:-maxabschange;
						eta = maxchange*(our_mass / pTri->numerator_x);
						
						iWorst = iTri;
						vworst = v_existing.x;
						dimworst = 0;
						ROCworst = pTri->numerator_x/our_mass;
						
					};
					
					v_change.y = eta*pTri->numerator_y/our_mass;	
					
					if ((fabs(v_change.y)-ABSOLUTE_THRESHOLD_VELOCITY_CHANGE)/fabs(v_existing.y) > MAX_CHANGE_PPN_V) 
					{
						maxabschange = fabs(v_existing.y)*MAX_CHANGE_PPN_V + ABSOLUTE_THRESHOLD_VELOCITY_CHANGE;
						maxchange = (v_change.y > 0.0)?maxabschange:-maxabschange;
						eta = maxchange*(our_mass / pTri->numerator_y);
						
						iWorst = iTri;
						vworst = v_existing.y;
						dimworst = 1;
						ROCworst = pTri->numerator_y/our_mass;
						
					};
					
					v_change.z = eta*pTri->numerator_z/our_mass;	

					if ((fabs(v_change.z)-ABSOLUTE_THRESHOLD_VELOCITY_CHANGE)/fabs(v_existing.z) > MAX_CHANGE_PPN_V) 
					{
						maxabschange = fabs(v_existing.z)*MAX_CHANGE_PPN_V + ABSOLUTE_THRESHOLD_VELOCITY_CHANGE;
						maxchange = (v_change.z > 0.0)?maxabschange:-maxabschange;
						eta = maxchange*(our_mass / pTri->numerator_z);
						
						iWorst = iTri;
						vworst = v_existing.z;
						dimworst = 2;
						ROCworst = pTri->numerator_z/our_mass;
						
					};					
					
					++pTri;
				};
				printf("iTri %d %c %1.8E ROC %1.8E \n",iWorst,
					(dimworst==0)?'x':(dimworst==1)?'y':'z',
					vworst, ROCworst);

				if (eta > time_remaining) 
				{
					completed_timestep = true;
					eta = time_remaining;
				} else {
					completed_timestep = false;
				};
				printf("eta: %1.5E remain: %1.5E |~| \n",eta,time_remaining);
				
				// 3. Perform the step!
				
				switch(species)
				{
				case SPECIES_ION:
					pTri = T;
					for (iTri = 0; iTri < numTriangles; iTri++)
					{
						
						pTri->ion.mom.x += eta*pTri->numerator_x;
						pTri->ion.mom.y += eta*pTri->numerator_y;
						pTri->ion.mom.z += eta*pTri->numerator_z;
						pTri->ion.heat += TWOTHIRDS*eta*pTri->numerator_T;
						GlobalViscousheatingtotal += TWOTHIRDS*eta*pTri->numerator_T;
						++pTri;
					};
					break;
				case SPECIES_NEUTRAL:
					pTri = T;
					for (iTri = 0; iTri < numTriangles; iTri++)
					{
						pTri->neut.mom.x += eta*pTri->numerator_x;
						pTri->neut.mom.y += eta*pTri->numerator_y;
						pTri->neut.mom.z += eta*pTri->numerator_z;
						pTri->neut.heat += TWOTHIRDS*eta*pTri->numerator_T;
						GlobalViscousheatingtotal += TWOTHIRDS*eta*pTri->numerator_T;
						++pTri;
					};
					break;
				case SPECIES_ELECTRON:
					pTri = T;
					for (iTri = 0; iTri < numTriangles; iTri++)
					{
						pTri->elec.mom.x += eta*pTri->numerator_x;
						pTri->elec.mom.y += eta*pTri->numerator_y;
						pTri->elec.mom.z += eta*pTri->numerator_z;
						pTri->elec.heat += TWOTHIRDS*eta*pTri->numerator_T;
						GlobalViscousheatingtotal += TWOTHIRDS*eta*pTri->numerator_T;
						// Heating: how?

						// use n for quadrilateral

						// rate of change of v2 * v2 / n
						// + rate of change of v1 * v1 / n

						// half heating here?
						
						++pTri;
					};
					break;
				};
			} else {
				// hypo-diffusion
			};
		};

		
		time_remaining -= eta;

		// 4. Hopefully we skip the boost of nT to restore total heat, but we could choose to add it in here.

		// 4. Likewise, viscous heating  -- 
		// although to do nv and nT smoothing together might make more sense. Timesteps almost certainly similar and then we can smooth the viscous heating.

	} while (completed_timestep == false);
	
}



int TriMesh::DebugCheckTemperaturesPositive ()
{
	Vertex * pVert = X;
	long countbad = 0;
	long last;
	for (long iVert = 0; iVert < numVertices; iVert++)
	{
		if ((pVert->ion.T < 0.0) || (pVert->neut.T < 0.0) || (pVert->elec.T < 0.0))
		{
			countbad++;
			last = iVert;
		};
		pVert++;
	};

	if (countbad > 0) return countbad;

	countbad = 0;
	long clast;
	Triangle * pTri = T;
	for (long iTri = 0; iTri < numTriangles; iTri++)
	{
		if ((pTri->ion.heat < 0.0) || (pTri->neut.heat < 0.0) || (pTri->elec.heat < 0.0))
		{
			countbad++;
			clast = iTri;
		};
		pTri++;
	};
	
	return countbad;

}


void TriMesh::DebugLocateHighestDensity()
{
	Vertex * pVert = X;
	real maxn = 0.0;
	long which = 0;
	for (long iVert = 0; iVert < numVertices; iVert++)
	{
		if (pVert->ion.n > maxn)
		{
			maxn = pVert->ion.n;
			which = iVert;
		};

		++pVert;
	};
	
	printf("\n\nvertex index for high n = %d\n\n",which);
}




