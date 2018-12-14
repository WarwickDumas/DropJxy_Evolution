#define real double
#define f64 double

#define HISTORY										4

#include "headers.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <time.h>
#include <stdio.h>
#include <windows.h>
#include "resource.h"
#include "flags.h"
#include "FFxtubes.h"
//#include "cppconst.h"
#include "cuda_struct.h"
#include "constant.h"

#include "d3d.h"    
#include <d3dx9.h> 
#include <dxerr.h>

#include <commdlg.h>    // probably used by avi_utils

#include "surfacegraph_tri.h"

#include "avi_utils.cpp"     // for making .avi


//=======================================================
// Declarations of functions:

void RefreshGraphs(TriMesh & X, const int iGraphsFlag);
LRESULT CALLBACK	WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK	About(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK	SetupBox(HWND, UINT, WPARAM, LPARAM);
extern f64 GetEzShape__(f64 r);
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);



// Global variables:
// =================

extern D3D Direct3D;

float xzscale;

bool bCullNone = false;
bool bGlobalsave = false;
int GlobalSwitchBox = 0;
int iGlobalScratch;

real GlobalHeightScale;

int GlobalSpeciesToGraph = SPECIES_ELECTRON;
int GlobalWhichLabels = 0;
bool GlobalRenderLabels = false;
int GlobalColoursPlanView = 0;
bool GlobalBothSystemsInUse;

bool GlobalCutaway = true;

unsigned int cw; // control word for floating point hardware exception hiding

TriMesh * pX, *pXnew;
TriMesh X1, X2, X3;
cuSyst cuSyst_host, cuSyst_host2;

D3DXVECTOR3 GlobalEye, GlobalLookat, GlobalPlanEye, GlobalPlanEye2, GlobalPlanLookat,
GlobalPlanLookat2, GlobalEye2, GlobalLookat2;

IDirect3DSurface9* p_backbuffer_surface;

long steps_remaining, GlobalStepsCounter, steps_remaining_CPU;
real evaltime, h;

extern real GlobalIzElasticity;
FILE * massfile, *maxfile;

// Global Variables:
HINSTANCE hInst;   // current instance
				   // window vars:
HWND hWnd;
WNDCLASSEX wcex;
TCHAR szTitle[1024];					// The title bar text
TCHAR szWindowClass[1024];			// the main window class name

char Functionalfilename[1024];
int GlobalGraphSetting[7];
surfacegraph Graph[7]; // why was it 5? // 5th one can be whole thing.


float Historic_max[100][HISTORY]; // if max is falling, use historic maximum for graph.
float Historic_min[100][HISTORY];
int Historic_powermax[200];
int Historic_powermin[200]; // just store previous value only.

bool boolGlobalHistory, GlobalboolDisplayMeshWireframe;

// avi file -oriented variables
int const NUMAVI = 1;
HAVI hAvi[NUMAVI + 1];
int const GraphFlags[5] = { LAPAZ_AZ, VIZVEZJZAZDOT,
							SPECIES_ELECTRON, JZAZBXYEZ, SPECIES_NEUTRAL};
char szAvi[5][128] = { "LapAz_Az","VIZVEZJZAZDOT","Elec","JzAzBxyEz","Neut" };

AVICOMPRESSOPTIONS opts;
int counter;
HBITMAP surfbit, dib;
HDC surfdc, dibdc;
LPVOID lpvBits;
BITMAPINFO bitmapinfo;



char * report_time(int action)
{
	/* action = 0: reset ; action = 1: report */
	/* requires timebuffy to be defined as char[255] globally */

	static char timebuffer[255];
	static clock_t start;
	double timenow;
	long ops;

	if (action == 0)
	{
		start = clock();
	}
	else
	{
		timenow = ((double)(clock() - start) / (double)CLOCKS_PER_SEC);
		ops = (long)(clock() - start);
		/* create a null-terminated string */
		sprintf(timebuffer, "%6.4f sec.", timenow);
	};
	return &(timebuffer[0]);
};


f64 GetTriangleArea(f64_vec2 pos0, f64_vec2 pos1, f64_vec2 pos2)
{
	f64 area = 0.5*((pos0.x + pos1.x)*(pos1.y - pos0.y) + (pos1.x + pos2.x)*(pos2.y - pos1.y)
		+ (pos0.x + pos2.x)*(pos0.y - pos2.y));
	return fabs(area);
}


void TriMesh::CalculateTotalGraphingData()
{
	/*long iVertex;
	Vertex * pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
	if ((pVertex->flags == DOMAIN_VERTEX) || (pVertex->flags == OUTERMOST))
	{
	pVertex->n = (pVertex->Neut.mass + pVertex->Ion.mass) / pVertex->AreaCell;
	pVertex->v = (m_n*pVertex->Neut.mom + m_ion * pVertex->Ion.mom + m_e * pVertex->Elec.mom) /
	(m_n*pVertex->Neut.mass + m_ion * pVertex->Ion.mass + m_e * pVertex->Elec.mass);
	pVertex->T = (pVertex->Neut.heat + pVertex->Ion.heat + pVertex->Elec.heat) /
	(pVertex->Neut.mass + pVertex->Ion.mass + pVertex->Elec.mass);
	pVertex->Temp.x = pVertex->Ion.mass / (pVertex->Neut.mass + pVertex->Ion.mass);
	};
	++pVertex;
	}*/
}

void TriMesh::Setup_J()
{
	/*long iVertex;
	Vertex * pVertex = X;
	for (iVertex = 0; iVertex < numVertices; iVertex++)
	{
	if ((pVertex->flags == DOMAIN_VERTEX) || (pVertex->flags == OUTERMOST))
	{
	pVertex->Temp = q * (pVertex->Ion.mom - pVertex->Elec.mom) / pVertex->AreaCell;
	}
	else {
	memset(&(pVertex->Temp), 0, sizeof(Vector3));
	}
	++pVertex;
	}*/
}

void surfacegraph::DrawSurface(const char * szname,
	const int heightflag,
	const real * var_ptr_0,
	const int colourflag,
	const real * var_ptr_c,
	const bool bDisplayInner,
	const int code, // graph code, to pass to called routines - sometimes useful
	const TriMesh * pX // for passing to SetDataWithColour and Render
					   // and for working out offsets
)
{
	// replaced CreateSurfaceGraphs.
	// I think this is about the right balance.

	char buff[256];
	real * temprealptr = (real *)(pX->pData);
	long offset = var_ptr_0 - temprealptr;
	long offset_c = var_ptr_c - temprealptr;

	// Does shader always go with colour type?? yes I think.
	switch (colourflag) {
	case VELOCITY_COLOUR:
		this->mhTech = mFX->GetTechniqueByName("VelociTech");
		break;
	case SEGUE_COLOUR:
		this->mhTech = mFX->GetTechniqueByName("SegueTech");
		break;
	case CURRENT_COLOUR:
		this->mhTech = mFX->GetTechniqueByName("XYZTech");
		break;
	case AZSEGUE_COLOUR:
		mhTech = mFX->GetTechniqueByName("AzSegueTech");
		break;
	case IONISE_COLOUR:
		mhTech = mFX->GetTechniqueByName("IoniseTech");
		break;
	};

	// Usual settings:
	//if (GlobalGraphSetting[i] != GRAPH_NONE) {

	this->boolDisplayShadow = true;
	this->boolDisplayMainMesh = true;
	this->boolDisplayMeshWireframe = GlobalboolDisplayMeshWireframe;
	this->boolClearZBufferBeforeWireframe = false;
	// Or try setting true and CULL_CCW to see if this stops it showing "the back of the wireframe"
	this->SetEyeAndLookat(GlobalEye, GlobalLookat);
	this->boolDisplayScales = true;
	this->boolDisplayInnerMesh = bDisplayInner;

	// work out whether to display key button:
	if (((colourflag == FLAG_VELOCITY_COLOUR) || (colourflag == FLAG_CURRENT_COLOUR))
		&& (bDisplayInner == 0))
	{
		this->boolDisplayKeyButton = true;
	}
	else {
		this->boolDisplayKeyButton = false;
	};
	//int const FLAG_COLOUR_MESH = 0;
	//int const FLAG_SEGUE_COLOUR = 1;
	//int const FLAG_VELOCITY_COLOUR = 2;
	//int const FLAG_CURRENT_COLOUR = 3;
	//int const FLAG_AZSEGUE_COLOUR = 4;
	//int const FLAG_IONISE_COLOUR = 5;

	this->SetDataWithColour(*pX,
		colourflag, heightflag, // apparently it's that way round
		offset, offset_c,
		code);

	if (this->bDisplayTimestamp) {
		sprintf(buff, "%6.2f ps", evaltime*1.0e12);
		this->Render(szname, false, pX, buff);
	}
	else {
		this->Render(szname, false, pX);
	};
}

// Here we make a function that we can call to tidy up graph calling code:

// How we are going to do graphs:
// Use plasma_data objects from BEGINNING_OF_CENTRAL.

void RefreshGraphs(TriMesh & X, // only not const because of such as Reset_vertex_nvT
	const int iGraphsFlag)
{
	Vertex * pVertex;
	long iVertex;
	plasma_data * pdata;
	int offset_v, offset_T;
	char buff[256];
	sprintf(buff, "%5.2f ps", evaltime*1.0e12);

	switch (iGraphsFlag) {
		/*
		case JXY_RHO_EXY_GRADPHI_AXYDOTOC_AXY:

		X.Setup_J(); // the others can already exist.

		Graph[4].bDisplayTimestamp = true;

		pVertex = pX->X;
		for (iVertex = 0; iVertex < pX->numVertices; iVertex++)
		{
		if (pVertex->flags == DOMAIN_VERTEX) {
		pVertex->temp2.x = q * (pVertex->Ion.mass - pVertex->Elec.mass) / pVertex->AreaCell;
		}
		else {
		pVertex->temp2.x = 0.0;
		};
		pVertex->Adot /= c;
		++pVertex;
		}

		Graph[0].DrawSurface("Exy[statV/cm]",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].E)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].E)),
		true,
		GRAPH_EXY, &X);

		Graph[1].DrawSurface("Adotxy/c[statV/cm]",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].Adot.x)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].Adot.x)),
		true,
		GRAPH_ADOTXY, &X);

		Graph[2].DrawSurface("Jxy",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].Temp.x)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].Temp.x)),
		false, // no inner mesh display.
		GRAPH_JXY, &X);

		Graph[3].DrawSurface("phidot[statV/s]",
		DATA_HEIGHT, (real *)(&(X.X[0].phidot)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].phidot)),
		true,
		GRAPH_PHIDOT, &X);

		Graph[4].DrawSurface("rho",
		DATA_HEIGHT, (real *)(&(X.X[0].temp2.x)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].temp2.x)),
		false, // no inner mesh display.
		GRAPH_RHO, &X);

		Graph[5].DrawSurface("phi[statV]",
		DATA_HEIGHT, (real *)(&(X.X[0].phi)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].phi)),
		true,
		GRAPH_PHI, &X);

		pVertex = pX->X;
		for (iVertex = 0; iVertex < pX->numVertices; iVertex++)
		{
		pVertex->Adot *= c;
		++pVertex;
		}
		break;
		case JZ_AZ_BXY_EZ_ADOTZOC_NVZ:

		X.Reset_vertex_nvT(SPECIES_ELEC);

		X.Setup_J(); // the others can already exist.

		pVertex = pX->X;
		for (iVertex = 0; iVertex < pX->numVertices; iVertex++)
		{
		pVertex->Adot /= c;
		++pVertex;
		}
		Graph[4].bDisplayTimestamp = true;

		Graph[0].DrawSurface("Ez[statV/cm]",
		DATA_HEIGHT, (real *)(&(X.X[0].E.z)),
		FLAG_SEGUE_COLOUR, (real *)(&(X.X[0].E.z)),
		false, // ??
		GRAPH_EZ, &X);

		Graph[1].DrawSurface("Az",
		DATA_HEIGHT, (real *)(&(X.X[0].A.z)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].A.z)),
		true,
		GRAPH_AZ, &X);
		Graph[2].DrawSurface("Jz",
		DATA_HEIGHT, (real *)(&(X.X[0].Temp.z)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].Temp.z)),
		false, // no inner mesh display.
		GRAPH_JZ, &X);
		Graph[3].DrawSurface("Bxy[Gs]",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].B)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].B)),
		true, // no inner mesh display: ??
		GRAPH_BXY, &X);

		Graph[4].DrawSurface("Adotz/c [statV/cm]",
		DATA_HEIGHT, (real *)(&(X.X[0].Adot.z)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].Adot.z)),
		true,
		GRAPH_AZ, &X);

		Graph[5].colourmax = Graph[2].colourmax;
		Graph[5].DrawSurface("Elec n",
		DATA_HEIGHT, (real *)(&(X.X[0].n)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].Temp.z)),
		false, // no inner mesh display
		GRAPH_ELEC_N, &X);

		pVertex = pX->X;
		for (iVertex = 0; iVertex < pX->numVertices; iVertex++)
		{
		pVertex->Adot *= c;
		++pVertex;
		}

		break;
		case SPECIES_ELECTRON2:

		X.Reset_vertex_nvT(SPECIES_ELEC);

		Graph[0].DrawSurface("Elec n [/cc]",
		DATA_HEIGHT, (real *)(&(X.X[0].n)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].v)),
		false, // no inner mesh display
		GRAPH_ELEC_N, &X);

		Graph[1].DrawSurface("v_e_xy[cm/s]",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].v)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].v)),
		false, // no inner mesh display
		GRAPH_ELEC_V, &X);

		Graph[3].DrawSurface("v_e_z[cm/s]",
		DATA_HEIGHT, (real *)(&(X.X[0].v.z)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].v.z)),
		false, // no inner mesh display.
		GRAPH_VEZ, &X);

		pVertex = pX->X;
		for (iVertex = 0; iVertex < pX->numVertices; iVertex++)
		{
		if (pVertex->flags == DOMAIN_VERTEX) {
		pVertex->temp2.x = (pVertex->Ion.mass - pVertex->Elec.mass) / pVertex->AreaCell;
		}
		else {
		pVertex->temp2.x = 0.0;
		};
		++pVertex;
		}
		Graph[2].bDisplayTimestamp = false;
		Graph[2].DrawSurface("n_i-n_e",
		DATA_HEIGHT, (real *)(&(X.X[0].temp2.x)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].temp2.x)),
		false, // no inner mesh display.
		GRAPH_NINE, &X);

		Graph[5].TickRescaling = 1.0 / kB;
		Graph[5].DrawSurface("Elec T [eV]",
		DATA_HEIGHT, (real *)(&(X.X[0].T)),
		SEGUE_COLOUR, (real *)(&(X.X[0].T)),
		false, // no inner mesh display
		GRAPH_ELEC_T, &X);
		Graph[5].TickRescaling = 1.0;

		offset_v = (real *)(&(X.X[0].v)) - (real *)(&(X.X[0]));
		offset_T = (real *)(&(X.X[0].T)) - (real *)(&(X.X[0]));

		Graph[4].SetEyePlan(GlobalPlanEye);
		Graph[4].boolDisplayMeshWireframe = true;
		Graph[4].boolClearZBufferBeforeWireframe = true;
		Graph[4].boolDisplayMainMesh = true;
		Graph[4].boolDisplayInnerMesh = false;
		Graph[4].boolDisplayScales = false;
		Graph[4].boolDisplayShadow = false;
		Graph[4].mhTech = Graph[4].mFX->GetTechniqueByName("VelociTech");
		Graph[4].colourmax = Graph[0].colourmax; // match colours
		Graph[4].SetDataWithColour(X, FLAG_VELOCITY_COLOUR, FLAG_FLAT_MESH, offset_v, offset_v,
		GRAPH_FLAT_WIRE_MESH);
		Graph[4].Render(buff, GlobalRenderLabels, &X);

		break;
		*/
	case SPECIES_NEUTRAL:


		Graph[0].DrawSurface("Neutral n",
			DATA_HEIGHT, (real *)(&(X.pData[0].n_n)),
			VELOCITY_COLOUR, (real *)(&(X.pData[0].v_n)),
			false, // no inner mesh display
			GRAPH_NEUT_N, &X);
		Graph[1].DrawSurface("Neutral v",
			VELOCITY_HEIGHT, (real *)(&(X.pData[0].v_n)),
			VELOCITY_COLOUR, (real *)(&(X.pData[0].v_n)),
			false, // no inner mesh display
			GRAPH_NEUT_V, &X);
		Graph[3].TickRescaling = 1.0 / kB_;
		Graph[3].DrawSurface("Neutral T",
			DATA_HEIGHT, (real *)(&(X.pData[0].Tn)),
			SEGUE_COLOUR, (real *)(&(X.pData[0].Tn)),
			false, // no inner mesh display
			GRAPH_NEUT_T, &X);
		Graph[3].TickRescaling = 1.0;
		// How to handle Graph[2] ?

		break;
	case SPECIES_ION:


		Graph[3].TickRescaling = 1.0 / kB_;
		Graph[3].DrawSurface("Ion T",
			DATA_HEIGHT, (real *)(&(X.pData[0].Ti)),
			SEGUE_COLOUR, (real *)(&(X.pData[0].Ti)),
			false, // no inner mesh display
			GRAPH_ION_T, &X);
		Graph[3].TickRescaling = 1.0;

		// labels only appear on first 1 called.

		Graph[0].DrawSurface("Ion n",
			DATA_HEIGHT, (real *)(&(X.pData[0].n)),
			VELOCITY_COLOUR, (real *)(&(X.pData[0].vxy)),
			false, // no inner mesh display
			GRAPH_ION_T, &X);
		Graph[1].DrawSurface("Ion v",
			VELOCITY_HEIGHT, (real *)(&(X.pData[0].vxy)),
			VELOCITY_COLOUR, (real *)(&(X.pData[0].vxy)),
			false, // no inner mesh display
			GRAPH_ION_V, &X);

		break;

	case SPECIES_ELEC:

		Graph[0].DrawSurface("Elec n",
			DATA_HEIGHT, (real *)(&(X.pData[0].n)),
			VELOCITY_COLOUR, (real *)(&(X.pData[0].vxy)),
			false, // no inner mesh display
			GRAPH_ELEC_T, &X);
		// colours == 0.0 ... because v = 0
		// First........... let's understand why surface normals come out unpredictable.
		// Then............ let's go and see what it does with y values (in Render and .fx)

		Graph[1].DrawSurface("Elec v",
			VELOCITY_HEIGHT, (real *)(&(X.pData[0].vxy)),
			VELOCITY_COLOUR, (real *)(&(X.pData[0].vxy)),
			false, // no inner mesh display
			GRAPH_ELEC_V, &X);
		Graph[3].TickRescaling = 1.0 / kB_;
		Graph[3].DrawSurface("Elec T",
			DATA_HEIGHT, (real *)(&(X.pData[0].Te)),
			SEGUE_COLOUR, (real *)(&(X.pData[0].Te)),
			false, // no inner mesh display
			GRAPH_ELEC_T, &X);
		Graph[3].TickRescaling = 1.0;
		break;

		// In other cases, (and even for the above),
		// here is a good place to call the 
		// setup routines for temp variables.

	case OVERALL:
		break;

	case JZAZBXYEZ:

		/*
		X.Setup_J(); // the others can already exist.

		printf("About to do Az\n");
		Graph[0].DrawSurface("Az",
		DATA_HEIGHT, (real *)(&(X.X[0].A.z)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].A.z)),
		true,
		GRAPH_AZ, &X);
		printf("About to do Bxy\n");
		Graph[1].DrawSurface("Bxy",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].B)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].B)),
		true, // no inner mesh display: ??
		GRAPH_BXY, &X);
		printf("About to do Ez\n");
		Graph[2].DrawSurface("Ez",
		DATA_HEIGHT, (real *)(&(X.X[0].E.z)),
		FLAG_SEGUE_COLOUR, (real *)(&(X.X[0].E.z)),
		false, // ??
		GRAPH_EZ, &X);
		printf("About to do Jz\n");
		Graph[3].DrawSurface("Jz",
		DATA_HEIGHT, (real *)(&(X.X[0].Temp.z)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].Temp.z)),
		false, // no inner mesh display.
		GRAPH_JZ, &X);
		break;
		*/
		break;
	case VIZVEZJZAZDOT:

		// Set Jz:
		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.x = q_ * pdata->n*(pdata->viz - pdata->vez);
			++pdata;
		};

		Graph[0].DrawSurface("viz",
			DATA_HEIGHT, (real *)(&(X.pData[0].viz)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
			false, GRAPH_VIZ, &X);

		Graph[1].DrawSurface("vez",
			DATA_HEIGHT, (real *)(&(X.pData[0].vez)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
			false, GRAPH_VEZ, &X);

		Graph[2].DrawSurface("Azdot",
			DATA_HEIGHT, (real *)(&(X.pData[0].Azdot)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].Azdot)),
			true, GRAPH_AZDOT, &X);

		Graph[3].DrawSurface("Jz",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
			false, GRAPH_JZ, &X);

		break;
	case LAPAZ_AZ:
		// Assume temp.x contains Lap Az
		Graph[0].DrawSurface("Lap Az",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.x)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.x)),
			true, GRAPH_LAPAZ, &X);
		Graph[1].DrawSurface("Az",
			DATA_HEIGHT, (real *)(&(X.pData[0].Az)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].Az)),
			true, GRAPH_AZ, &X);
		Graph[2].DrawSurface("Azdot",
			DATA_HEIGHT, (real *)(&(X.pData[0].Azdot)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].Azdot)),
			true, GRAPH_AZDOT, &X);
		pdata = X.pData + BEGINNING_OF_CENTRAL;
		for (iVertex = 0; iVertex < NUMVERTICES; iVertex++)
		{
			pdata->temp.y = q_ * pdata->n*(pdata->viz - pdata->vez);
			++pdata;
		};
		Graph[3].DrawSurface("Jz",
			DATA_HEIGHT, (real *)(&(X.pData[0].temp.y)),
			AZSEGUE_COLOUR, (real *)(&(X.pData[0].temp.y)),
			false, GRAPH_JZ, &X);

		break;
	case EXYCOMPONENTS:
		/*
		X.Setup_J(); // the others can already exist.

		Graph[0].DrawSurface("Adotxy",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].Adot.x)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].Adot.x)),
		true,
		GRAPH_ADOTXY, &X);
		Graph[1].DrawSurface("Grad phi",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].GradTe)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].GradTe)),
		true, // no inner mesh display: ??
		GRAPH_GRADPHI, &X);
		Graph[2].DrawSurface("Exy",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].E)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].E)),
		true,
		GRAPH_EXY, &X);
		Graph[3].DrawSurface("Jxy",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].Temp.x)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].Temp.x)),
		false, // no inner mesh display.
		GRAPH_JXY, &X);

		*/
		// Set GradTe to grad phi
		break;
	case JXYAXYBZEXY:
		/*
		X.Setup_J(); // the others can already exist.

		Graph[0].DrawSurface("Axy",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].A.x)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].A.x)),
		true,
		GRAPH_AXY, &X);
		Graph[1].DrawSurface("Bz",
		DATA_HEIGHT, (real *)(&(X.X[0].B.z)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].B.z)),
		true, // no inner mesh display: ??
		GRAPH_BZ, &X);
		Graph[2].DrawSurface("Exy",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].E)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].E)),
		true,
		GRAPH_EXY, &X);
		Graph[3].DrawSurface("Jxy",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].Temp.x)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].Temp.x)),
		false, // no inner mesh display.
		GRAPH_JXY, &X);
		*/
		break;
	case EXY_RHO_PHI_PHIDOT:
		/*
		// For this one do n_i-n_e
		pVertex = pX->X;
		for (iVertex = 0; iVertex < pX->numVertices; iVertex++)
		{
		if (pVertex->flags == DOMAIN_VERTEX) {
		pVertex->temp2.x = (pVertex->Ion.mass - pVertex->Elec.mass) / pVertex->AreaCell;
		}
		else {
		pVertex->temp2.x = 0.0;
		};
		++pVertex;
		}

		Graph[0].DrawSurface("phi",
		DATA_HEIGHT, (real *)(&(X.X[0].phi)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].phi)),
		true,
		GRAPH_PHI, &X);
		Graph[1].DrawSurface("phidot",
		DATA_HEIGHT, (real *)(&(X.X[0].phidot)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].phidot)),
		true,
		GRAPH_PHIDOT, &X);
		Graph[2].DrawSurface("Exy",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].E)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].E)),
		true,
		GRAPH_EXY, &X);
		Graph[3].DrawSurface("n_i-n_e",
		DATA_HEIGHT, (real *)(&(X.X[0].temp2.x)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].temp2.x)),
		false, // no inner mesh display.
		GRAPH_NINE, &X);
		*/
		break;
	case EXY_RHO_PHI_JXY:
		// create rho on pVertex->temp2.x ... 
		/*
		pVertex = pX->X;
		for (iVertex = 0; iVertex < pX->numVertices; iVertex++)
		{
		if (pVertex->flags == DOMAIN_VERTEX) {
		pVertex->temp2.x = q * (pVertex->Ion.mass - pVertex->Elec.mass) / pVertex->AreaCell;
		}
		else {
		pVertex->temp2.x = 0.0;
		};
		++pVertex;
		}

		X.Setup_J();

		Graph[0].DrawSurface("phi",
		DATA_HEIGHT, (real *)(&(X.X[0].phi)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].phi)),
		false,
		GRAPH_PHI, &X);
		Graph[1].DrawSurface("Jxy",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].Temp)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].Temp)),
		false, // no inner mesh display: ??
		GRAPH_JXY, &X);
		Graph[2].DrawSurface("Exy",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].E)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].E)),
		false,
		GRAPH_EXY, &X);
		Graph[3].DrawSurface("rho",
		DATA_HEIGHT, (real *)(&(X.X[0].temp2.x)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].temp2.x)),
		false, // no inner mesh display.
		GRAPH_RHO, &X);
		*/
		break;

	case EXY_RHO_BZ_JXY:
		/*
		// create rho on pVertex->temp2.x ...
		pVertex = pX->X;
		for (iVertex = 0; iVertex < pX->numVertices; iVertex++)
		{
		if (pVertex->flags == DOMAIN_VERTEX) {
		pVertex->temp2.x = q * (pVertex->Ion.mass - pVertex->Elec.mass) / pVertex->AreaCell;
		}
		else {
		pVertex->temp2.x = 0.0;
		};
		++pVertex;
		}

		X.Setup_J();

		Graph[0].DrawSurface("Bz",
		DATA_HEIGHT, (real *)(&(X.X[0].B.z)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].B.z)),
		true, // no inner mesh display: ??
		GRAPH_BZ, &X);
		Graph[1].DrawSurface("Jxy",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].Temp)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].Temp)),
		false, // no inner mesh display: ??
		GRAPH_JXY, &X);
		Graph[2].DrawSurface("Exy",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].E)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].E)),
		false,
		GRAPH_EXY, &X);
		Graph[3].DrawSurface("rho",
		DATA_HEIGHT, (real *)(&(X.X[0].temp2.x)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].temp2.x)),
		false, // no inner mesh display.
		GRAPH_RHO, &X);
		*/
		break;

	case SIGMA_E_J:
		/*
		X.Setup_J(); // the others can already exist.

		Graph[0].DrawSurface("sigma_e_zz",
		DATA_HEIGHT, (real *)(&(X.X[0].sigma_e.zz)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].sigma_e.zz)),
		true,
		GRAPH_SIGMA_E, &X);
		//Graph[1].DrawSurface("v_e_0.z",
		//	DATA_HEIGHT,(real *)(&(X.X[0].v_e_0.z)),
		//	AZSEGUE_COLOUR,(real *)(&(X.X[0].v_e_0.z)),
		//false, // no inner mesh display: ??
		// GRAPH_VE0Z, &X);
		Graph[1].DrawSurface("nsigma",
		DATA_HEIGHT, (real *)(&(X.X[0].xdotdot.x)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].xdotdot.x)),
		true, GRAPH_SIGMATEMP, &X);
		Graph[2].DrawSurface("Ez",
		DATA_HEIGHT, (real *)(&(X.X[0].E.z)),
		FLAG_AZSEGUE_COLOUR, (real *)(&(X.X[0].E.z)), // how to make SEGUE_COLOUR work?
		false, // ??
		GRAPH_EZ, &X);
		Graph[3].DrawSurface("Jz",
		DATA_HEIGHT, (real *)(&(X.X[0].Temp.z)),
		AZSEGUE_COLOUR, (real *)(&(X.X[0].Temp.z)),
		false, // no inner mesh display.
		GRAPH_JZ, &X);
		*/
		break;

	case TOTAL:
		break;
		// In this case we have to create data,
		// as we go.
		/*
		// Best put it here so we can see where
		// data is being populated.

		X.CalculateTotalGraphingData();

		// ought to change this to use variables n,v,T !

		Graph[0].DrawSurface("n_n+n_ion",
		DATA_HEIGHT, (real *)(&(X.X[0].n)),
		IONISE_COLOUR, (real *)(&(X.X[0].Temp.y)),
		false,
		GRAPH_TOTAL_N, &X);
		Graph[1].DrawSurface("[n_s v_s m_s]/[n_s m_s]",
		VELOCITY_HEIGHT, (real *)(&(X.X[0].v)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].v)),
		false, // no inner mesh display
		GRAPH_TOTAL_V, &X);
		Graph[2].DrawSurface("n_n+n_ion",
		DATA_HEIGHT, (real *)(&(X.X[0].n)),
		VELOCITY_COLOUR, (real *)(&(X.X[0].v)),
		false,
		GRAPH_TOTAL_N_II, &X);
		Graph[3].TickRescaling = 1.0 / kB;
		Graph[3].DrawSurface("[n_s T_s]/[n_s]",
		DATA_HEIGHT, (real *)(&(X.X[0].T)),
		SEGUE_COLOUR, (real *)(&(X.X[0].T)),
		false,
		GRAPH_TOTAL_T, &X);
		Graph[3].TickRescaling = 1.0;
		break;
		*/
	};

	// Graph 2, in case of species graphs:

	switch (iGraphsFlag) {
	case SPECIES_NEUTRAL:
	case SPECIES_ION:
	case SPECIES_ELEC:

		int offset_v, offset_T;
		offset_v = (real *)(&(X.pData[0].vxy)) - (real *)(&(X.pData[0]));
		offset_T = (real *)(&(X.pData[0].Te)) - (real *)(&(X.pData[0]));

		Graph[2].SetEyePlan(GlobalPlanEye);
		Graph[2].boolDisplayMeshWireframe = true;
		Graph[2].boolClearZBufferBeforeWireframe = true;
		Graph[2].boolDisplayMainMesh = true;
		Graph[2].boolDisplayInnerMesh = false;
		Graph[2].boolDisplayScales = false;

		if (GlobalColoursPlanView == 0)
		{
			// nothing
			Graph[2].mhTech = Graph[2].mFX->GetTechniqueByName("MeshTech");
			Graph[2].SetDataWithColour(X, FLAG_COLOUR_MESH, FLAG_FLAT_MESH, 0, 0,
				GRAPH_FLAT_WIRE_MESH);
			Graph[2].Render(buff, GlobalRenderLabels, &X);

		}
		else {
			// Tell SDWC not to mess with colourmax if it's a flat mesh.

			if (GlobalColoursPlanView == 1)
			{
				// velocity
				Graph[2].mhTech = Graph[2].mFX->GetTechniqueByName("VelociTech");
				Graph[2].colourmax = Graph[0].colourmax; // match colours

				Graph[2].SetDataWithColour(X, FLAG_VELOCITY_COLOUR, FLAG_FLAT_MESH, offset_v, offset_v,
					GRAPH_FLAT_WIRE_MESH);
				Graph[2].Render(buff, GlobalRenderLabels, &X);
			}
			else {
				// temperature
				Graph[2].mhTech = Graph[2].mFX->GetTechniqueByName("SegueTech");
				// SegueVS should take maximum as a parameter;
				// at least for colours we should prefer an absolute scale for T
				// Is it ever used for anything else? Not so far? eps?

				Graph[2].SetDataWithColour(X, FLAG_SEGUE_COLOUR, FLAG_FLAT_MESH, offset_T, offset_T,
					GRAPH_FLAT_WIRE_MESH);
				Graph[2].Render(buff, GlobalRenderLabels, &X);
			};
		};
		break;
	}
}




__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };
	
	HINSTANCE hInstance = GetModuleHandle(NULL);
	HWND hwndConsole = GetConsoleWindow();
	char szInitialFilenameAvi[512];
	MSG msg;
	HDC hdc;
	//	HACCEL hAccelTable;
	real x, y, temp;
	int i, j;
	float a1, a2, a3, a4;
	//HWND hwndConsole;
	FILE * fp;
	extern char Functionalfilename[1024];
	
	h = TIMESTEP;
	evaltime = 0.0; // gets updated before advance

	memset(Historic_powermax, 0, 200 * sizeof(int));
	memset(Historic_powermin, 0, 200 * sizeof(int));

	ZeroMemory(Historic_max, 100 * HISTORY * sizeof(float));
	ZeroMemory(Historic_min, 100 * HISTORY * sizeof(float));
	GlobalStepsCounter = 0; steps_remaining = 0; steps_remaining_CPU = 0;

	SetConsoleTitle("2D 1/16 annulus DPF simulation");
	Sleep(40);
	//hwndConsole = FindWindow(NULL, "2D 1/16 annulus DPF simulation");
	MoveWindow(hwndConsole, 0, 0, SCREEN_WIDTH - VIDEO_WIDTH - 10, SCREEN_HEIGHT - 30, TRUE);

	report_time(0);

	int filetag = 0;
	do {
		filetag++;
		sprintf(Functionalfilename, FUNCTIONALFILE_START "%03d.txt", filetag);
	} while ((_access(Functionalfilename, 0)) != -1);
	printf("\n\nopening %s \n", Functionalfilename);
	fp = fopen(Functionalfilename, "w");
	if (fp == 0) {
		printf("error with %s \n", Functionalfilename);
		getch();
	}
	else {
		printf("opened %s \n", Functionalfilename);
	};
	fprintf(fp, "GSC evaltime Area neut.N ion.N elec.N neut.r ion.r elec.r SDneut.r SDion.r SDelec.r "
		" neut.vr neut.vth neut.vz  ion.vr ion.vth ion.vz elec.vr elec.vth elec.vz neut.heat ion.heat elec.heat neut.T ion.T elec.T "
		" neut.mnvv/3 ion.mnvv/3 elec.mnvv/3 elec.force(vxB)r within3.6 elec.Bth EE BB Heatings and dT changes - see code \n");
	fclose(fp);

	X1.Initialise(1); // Set evaltime first
	X2.Initialise(2);
	X3.Initialise(3);
	pX = &X1;
	pXnew = &X2;

	GlobalBothSystemsInUse = 0;

	printf(report_time(1));
	printf("\n");
	report_time(0);



	// Window setup
	LoadString(hInstance, IDS_APP_TITLE, szTitle, 1024);
	LoadString(hInstance, IDC_F2DVALS, szWindowClass, 1024);
	wcex.cbSize = sizeof(WNDCLASSEX);
	wcex.style = CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc = WndProc;
	wcex.cbClsExtra = 0;
	wcex.cbWndExtra = 0;
	wcex.hInstance = hInstance;
	wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_F2DVALS));
	wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
	wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wcex.lpszMenuName = MAKEINTRESOURCE(IDR_MENU1);
	wcex.lpszClassName = szWindowClass;
	wcex.hIconSm = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));
	if (RegisterClassEx(&wcex) == 0) {
		char buff[128];
		MessageBox(NULL, "RegisterClassEx failed", itoa(GetLastError(), buff, 10), MB_OK);
	};


	printf("SCREEN_WIDTH %d VIDEO_WIDTH %d VIDEO_HEIGHT %d \n",
		SCREEN_WIDTH, VIDEO_WIDTH, VIDEO_HEIGHT);

	hWnd = CreateWindowEx(NULL, szWindowClass, szTitle, WS_BORDER | WS_POPUP,
		SCREEN_WIDTH - VIDEO_WIDTH - 5, 0, VIDEO_WIDTH + 5, VIDEO_HEIGHT + 20, NULL, NULL, hInstance, NULL);
	if (!hWnd) {
		DWORD dword = GetLastError();
		char buff[128];
		MessageBox(NULL, "CreateWindowEx failed", itoa(dword, buff, 10), MB_OK);
		return dword;
	}
	// This is sending a message to WndProc before any of the following happens.

	ShowWindow(hWnd, SW_SHOWNORMAL);
	UpdateWindow(hWnd);

	xzscale = 2.0 / 0.1; // very zoomed in. Now what?

	DXChk(Direct3D.Initialise(hWnd, hInstance, VIDEO_WIDTH, VIDEO_HEIGHT));

	// With Field Of View = PI/4 used this:

	GlobalEye.x = 0.0f;
	GlobalEye.y = 12.4f;  //7.2f;
	GlobalEye.z = -18.0f + 2.5*xzscale;//DEVICE_RADIUS_INSULATOR_OUTER*xzscale;//-17.8f+

	GlobalLookat.x = 0.4f;
	GlobalLookat.y = 3.0f;
	GlobalLookat.z = DEVICE_RADIUS_INITIAL_FILAMENT_CENTRE * xzscale;

	GlobalPlanEye.x = 0.0f;
	GlobalPlanEye.y = 35.0f;
	GlobalPlanEye.z = (3.44 + 4.1)*0.5*xzscale;

	GlobalPlanEye2.x = -0.1f;
	GlobalPlanEye2.y = 19.5f;
	GlobalPlanEye2.z = 2.8*xzscale;

	GlobalPlanLookat.x = GlobalPlanEye.x;
	GlobalPlanLookat.y = 0.0f;
	GlobalPlanLookat.z = GlobalPlanEye.z + 0.0001;

	GlobalPlanLookat2.x = GlobalPlanEye2.x;
	GlobalPlanLookat2.y = 0.0f;
	GlobalPlanLookat2.z = GlobalPlanEye2.z + 0.0001;

						 // Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	if (DXChk(Graph[0].InitialiseWithoutBuffers(0, 0, GRAPH_WIDTH, GRAPH_HEIGHT, GlobalEye, GlobalLookat)) +
		DXChk(Graph[0].InitialiseBuffers(X1))
		)
	{
		PostQuitMessage(200);
	};
	if (DXChk(Graph[1].InitialiseWithoutBuffers(0, GRAPH_HEIGHT, GRAPH_WIDTH, GRAPH_HEIGHT, GlobalEye, GlobalLookat)) +
		DXChk(Graph[1].InitialiseBuffers(X1))
		)
	{
		PostQuitMessage(201);
	};
	if (DXChk(Graph[2].InitialiseWithoutBuffers(GRAPH_WIDTH, 0, GRAPH_WIDTH, GRAPH_HEIGHT, GlobalPlanEye, GlobalPlanLookat)) +
		DXChk(Graph[2].InitialiseBuffers(X1))
		)
	{
		PostQuitMessage(202);
	};
	if (DXChk(Graph[3].InitialiseWithoutBuffers(GRAPH_WIDTH, GRAPH_HEIGHT, GRAPH_WIDTH, GRAPH_HEIGHT, GlobalEye, GlobalLookat)) +
		DXChk(Graph[3].InitialiseBuffers(X1))
		)
	{
		PostQuitMessage(203);
	};

	if (NUMGRAPHS > 4) {

		if (DXChk(Graph[4].InitialiseWithoutBuffers(GRAPH_WIDTH * 2, 0, GRAPH_WIDTH, GRAPH_HEIGHT, GlobalPlanEye, GlobalPlanLookat)) +
			DXChk(Graph[4].InitialiseBuffers(X1))
			)
		{
			PostQuitMessage(204);
		};

		if (DXChk(Graph[5].InitialiseWithoutBuffers(GRAPH_WIDTH * 2, GRAPH_HEIGHT, GRAPH_WIDTH, GRAPH_HEIGHT, GlobalPlanEye, GlobalPlanLookat)) +
			DXChk(Graph[5].InitialiseBuffers(X1))
			)
		{
			PostQuitMessage(204);
		};

	};

	Graph[0].bDisplayTimestamp = false;
	Graph[1].bDisplayTimestamp = false;
	Graph[2].bDisplayTimestamp = true;
	Graph[3].bDisplayTimestamp = false;
	Graph[4].bDisplayTimestamp = false;
	Graph[5].bDisplayTimestamp = false;

	Direct3D.pd3dDevice->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &p_backbuffer_surface);

	if (DXChk(p_backbuffer_surface->GetDC(&surfdc), 1000))
		MessageBox(NULL, "GetDC failed", "oh dear", MB_OK);

	surfbit = CreateCompatibleBitmap(surfdc, VIDEO_WIDTH, VIDEO_HEIGHT); // EXTRAHEIGHT = 90
	SelectObject(surfdc, surfbit);
	dibdc = CreateCompatibleDC(surfdc);

	long VideoWidth = VIDEO_WIDTH;
	long VideoHeight = VIDEO_HEIGHT;

	// pasted here just to set up format:
	bitmapinfo.bmiHeader.biSize = sizeof(BITMAPINFO);
	bitmapinfo.bmiHeader.biWidth = VideoWidth;
	bitmapinfo.bmiHeader.biHeight = VideoHeight;
	bitmapinfo.bmiHeader.biPlanes = 1;
	bitmapinfo.bmiHeader.biBitCount = 24;
	bitmapinfo.bmiHeader.biCompression = BI_RGB; // uncompressed  
	bitmapinfo.bmiHeader.biSizeImage = bitmapinfo.bmiHeader.biHeight;
	bitmapinfo.bmiHeader.biXPelsPerMeter = 3000;
	bitmapinfo.bmiHeader.biYPelsPerMeter = 3000;
	bitmapinfo.bmiHeader.biClrUsed = 0;
	bitmapinfo.bmiHeader.biClrImportant = 0;
	bitmapinfo.bmiColors->rgbBlue = 0;
	bitmapinfo.bmiColors->rgbRed = 0;
	bitmapinfo.bmiColors->rgbGreen = 0;
	bitmapinfo.bmiColors->rgbReserved = 0;
	// dimension DIB and set up pointer to bits
	dib = CreateDIBSection(dibdc, &bitmapinfo, DIB_RGB_COLORS, &lpvBits, NULL, 0);
	SelectObject(dibdc, dib);

	BitBlt(dibdc, 0, 0, VIDEO_WIDTH, VIDEO_HEIGHT, surfdc, 0, 0, SRCCOPY);


	for (i = 0; i < NUMAVI; i++)
	{
		sprintf(szInitialFilenameAvi, "%s%s_%s", FOLDER, szAvi[i], INITIALAVI);
		hAvi[i] = CreateAvi(szInitialFilenameAvi, AVIFRAMEPERIOD, NULL);
	};

	// 1000/25 = 40
	ZeroMemory(&opts, sizeof(opts));
	opts.fccHandler = mmioFOURCC('D', 'I', 'B', ' ');//('d','i','v','x');
	opts.dwFlags = 8;

	for (i = 0; i < NUMAVI; i++)
		SetAviVideoCompression(hAvi[i], dib, &opts, false, hWnd); // always run this for every avi file but can
																  // call with false as long as we know opts contains valid information. 
	counter = 0;
	//ReleaseDC(hWnd,surfdc);
	p_backbuffer_surface->ReleaseDC(surfdc);

	GlobalCutaway = true;
	
	RefreshGraphs(*pX, GlobalSpeciesToGraph); 

	//printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
	//	c[0], c[1], c[2], c[3], c[4]);
	Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);

						 // Main message loop:
	memset(&msg, 0, sizeof(MSG));
	while (msg.message != WM_QUIT)
	{
		if (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		else {
		
			Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);

		};
	};

	UnregisterClass(szWindowClass, wcex.hInstance);


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}


LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	static bool bInvoked_cuSyst = false;
	static long GSCCPU = 0;
	int iAntiskips;
	int wmId, wmEvent;
	int i, j, ctr;
	PAINTSTRUCT ps;
	HDC hdc;
	real time_back_for_Adot;
	FILE * file, *fp;
	int maxeerr, count, iMin;
	char buf1000[1024];
	int attempts;
	real store_h;
	char ch, o;
	int failed;
	RECT rect;
	real TotalArea, TotalCharge;
	long iVertex;
	real mass_avg, mass_SD, mass_min, mass_max;
	OPENFILENAME ofn;       // common dialog box structure
	char szFile[260];       // buffer for file name
	char szFilter[1000]; // buffer for file filter
	char szfilter[256];
	char buffer[256];

	TriMesh * temp;
	Vertex * pVertex;

	static const real XCENTRE2 = DEVICE_RADIUS_INITIAL_FILAMENT_CENTRE * sin(PI / 32.0);
	static const real XCENTRE1 = -XCENTRE2;
	static const real YCENTRE = DEVICE_RADIUS_INITIAL_FILAMENT_CENTRE * cos(PI / 32.0);

	switch (message)
	{
	case WM_CREATE:

		// Don't ever try doing initialisation here;
		// That should be done manually from the menus.		

		break;

	case WM_COMMAND:
		wmId = LOWORD(wParam);
		wmEvent = HIWORD(wParam);

		// Ensure that display menu items are consecutive IDs.
		// Parse the menu selections:
		switch (wmId)
		{
		case ID_DISPLAY_NEUT:
		case ID_DISPLAY_ELECTRON:
		case ID_DISPLAY_ION:
		case ID_DISPLAY_TOTAL:
		case ID_DISPLAY_JZAZBXYEZ:
		case ID_DISPLAY_VIZVEZJZAZDOT:

			GlobalSpeciesToGraph = wmId;

			i = wmId - ID_DISPLAY_NEUT;
			printf("\nGraph: %d %s", i, szAvi[i]);
			RefreshGraphs(*pX, i);
			Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);
			break;

		case ID_HELP_ABOUT:
			DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
			break;
		case ID_FILE_EXIT:
			DestroyWindow(hWnd);
			break;

		case ID_FILE_SAVECAMERA:
			ZeroMemory(&ofn, sizeof(ofn));
			ofn.lStructSize = sizeof(ofn);
			ofn.hwndOwner = hWnd;
			ofn.lpstrFile = szFile;
			// Set lpstrFile[0] to '\0' so that GetOpenFileName does not 
			// use the contents of szFile to initialize itself.
			ofn.lpstrFile[0] = '\0';
			ofn.nMaxFile = sizeof(szFile);
			memcpy(szfilter, "All\0*.*\0cam\0*.CAM\0\0", 19); // strcpy stops at first null !!
			ofn.lpstrFilter = szfilter; //"All\0*.*\0Dat\0*.DAT\0\0";	// summat weird about that example code
			ofn.nFilterIndex = 1;
			ofn.lpstrFileTitle = NULL;
			ofn.nMaxFileTitle = 0;
			ofn.lpstrInitialDir = NULL;
			ofn.Flags = OFN_OVERWRITEPROMPT;
			ofn.lpstrTitle = NULL;

			if (GetSaveFileName(&ofn) == TRUE)
			{
				printf("\nsaving camera...");
				fp = fopen(ofn.lpstrFile, "wt");
				if (fp == 0) {
					printf("save failed.\n");
				}
				else {
					fprintf(fp, "%f %f %f ", GlobalPlanEye.x, GlobalPlanEye.y, GlobalPlanEye.z);
					fprintf(fp, "%f %f %f ", GlobalLookat.x, GlobalLookat.y, GlobalLookat.z);
					fprintf(fp, "%f %f %f ", GlobalEye.x, GlobalEye.y, GlobalEye.z);
					fprintf(fp, "%f %f %f ", GlobalPlanLookat.x, GlobalPlanLookat.y, GlobalPlanLookat.z);
					fclose(fp);
					printf("done\n");
				};
			}
			else {
				printf("there was an issue\n");
			};
			break;
		case ID_FILE_LOADCAMERA:
			// Initialize OPENFILENAME
			ZeroMemory(&ofn, sizeof(ofn));
			ofn.lStructSize = sizeof(ofn);
			ofn.hwndOwner = hWnd;
			ofn.lpstrFile = szFile;
			// Set lpstrFile[0] to '\0' so that GetOpenFileName does not 
			// use the contents of szFile to initialize itself.
			ofn.lpstrFile[0] = '\0';
			ofn.nMaxFile = sizeof(szFile);
			memcpy(szfilter, "All\0*.*\0*.cam\0*.Cam\0\0", 21); // strcpy stops at first null !!
			ofn.lpstrFilter = szfilter; //"All\0*.*\0*.Dat\0*.DAT\0\0";	// summat weird about that example code
			ofn.nFilterIndex = 1;
			ofn.lpstrFileTitle = NULL;
			ofn.nMaxFileTitle = 0;
			ofn.lpstrInitialDir = NULL;
			ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
			// Display the Open dialog box. 
			if (GetOpenFileName(&ofn) == TRUE)
			{
				printf("\nloading camera...");
				fp = fopen(ofn.lpstrFile, "rt");
				if (fp == 0) {
					printf("failed.\n");
				}
				else {
					rewind(fp);
					fscanf(fp, "%f %f %f ", &(GlobalPlanEye.x), &(GlobalPlanEye.y), &(GlobalPlanEye.z));
					fscanf(fp, "%f %f %f ", &(GlobalLookat.x), &(GlobalLookat.y), &(GlobalLookat.z));
					fscanf(fp, "%f %f %f ", &(GlobalEye.x), &(GlobalEye.y), &(GlobalEye.z));
					fscanf(fp, "%f %f %f ", &(GlobalPlanLookat.x), &(GlobalPlanLookat.y), &(GlobalPlanLookat.z));
					fclose(fp);
				};
				RefreshGraphs(*pX, GlobalSpeciesToGraph); // sends data to graphs AND renders them
				Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);

			}
			else {
				printf("file error camera\n");
			};
			break; 
		case ID_FILE_SAVEBINARY:
			// Initialize OPENFILENAME
			ZeroMemory(&ofn, sizeof(ofn));
			ofn.lStructSize = sizeof(ofn);
			ofn.hwndOwner = hWnd;
			ofn.lpstrFile = szFile;
			// Set lpstrFile[0] to '\0' so that GetOpenFileName does not 
			// use the contents of szFile to initialize itself.
			ofn.lpstrFile[0] = '\0';
			ofn.nMaxFile = sizeof(szFile);
			memcpy(szfilter, "All\0*.*\0*.dat\0*.Dat\0\0", 21); // strcpy stops at first null !!
			ofn.lpstrFilter = szfilter; //"All\0*.*\0Dat\0*.DAT\0\0";	// summat weird about that example code
			ofn.nFilterIndex = 1;
			ofn.lpstrFileTitle = NULL;
			ofn.nMaxFileTitle = 0;
			ofn.lpstrInitialDir = NULL;
			ofn.Flags = OFN_OVERWRITEPROMPT;
			ofn.lpstrTitle = NULL;
			// Display the Open dialog box. 
			if (GetSaveFileName(&ofn) == TRUE)
			{
				printf("\nsaving system...");
				pX->Save(ofn.lpstrFile);
				printf("done\n");
			}
			else {
				printf("there was an issue\n");
			};
			break;
		case ID_FILE_SAVETEXT:
			// Initialize OPENFILENAME
			ZeroMemory(&ofn, sizeof(ofn));
			ofn.lStructSize = sizeof(ofn);
			ofn.hwndOwner = hWnd;
			ofn.lpstrFile = szFile;
			//
			// Set lpstrFile[0] to '\0' so that GetOpenFileName does not 
			// use the contents of szFile to initialize itself.
			ofn.lpstrFile[0] = '\0';
			ofn.nMaxFile = sizeof(szFile);
			//strcpy(szFilter,"All\0*.*\0Text\0*.TXT\0");
			memcpy(szfilter, "All\0*.*\0Dat\0*.DAT\0\0", 19); // strcpy stops at first null !!
			ofn.lpstrFilter = szfilter; //"All\0*.*\0Dat\0*.DAT\0\0";	// summat weird about that example code
			ofn.nFilterIndex = 1;
			ofn.lpstrFileTitle = NULL;
			ofn.nMaxFileTitle = 0;
			ofn.lpstrInitialDir = NULL;
			ofn.Flags = OFN_OVERWRITEPROMPT;
			ofn.lpstrTitle = NULL;
			// Display the Open dialog box. 
			if (GetSaveFileName(&ofn) == TRUE)
			{
				printf("\nsaving system...");
				pX->SaveText(ofn.lpstrFile);
				printf("done\n");
			}
			else {
				printf("there was an issue\n");
			};
			break;
		case ID_FILE_LOAD:

			// Initialize OPENFILENAME:
			ZeroMemory(&ofn, sizeof(ofn));
			ofn.lStructSize = sizeof(ofn);
			ofn.hwndOwner = hWnd;
			ofn.lpstrFile = szFile;
			//
			// Set lpstrFile[0] to '\0' so that GetOpenFileName does not 
			// use the contents of szFile to initialize itself.
			ofn.lpstrFile[0] = '\0';
			ofn.nMaxFile = sizeof(szFile);
			//strcpy(szFilter, "All\0*.*\0Dat\0*.DAT\0\0");
			memcpy(szfilter, "All\0*.*\0Dat\0*.DAT\0\0", 19); // strcpy stops at first null !!
			ofn.lpstrFilter = szfilter; //"All\0*.*\0Dat\0*.DAT\0\0";	// summat weird about that example code
			ofn.nFilterIndex = 1;
			ofn.lpstrFileTitle = NULL;
			ofn.nMaxFileTitle = 0;
			ofn.lpstrInitialDir = NULL;
			ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

			// Display the Open dialog box. 
			if (GetOpenFileName(&ofn) == TRUE)
			{
				pX->Load(ofn.lpstrFile);
				printf("\ndoin nothing...");
			};
		break;

		case ID_RUN_SIMULATIONSTEPS:

			GlobalSwitchBox = 0;
			DialogBox(hInst, MAKEINTRESOURCE(IDD_DIALOG1), hWnd, SetupBox);
			// that will not return with steps_remaining unset.

			if (steps_remaining > 0)
				SetTimer(hWnd, 1, 1, NULL); // 1 millisecond delay

			break;

		case ID_RUN_SIMULATIONSTEPS_CPU:

			GlobalSwitchBox = 0;
			steps_remaining_CPU = 1;
			// that will not return with steps_remaining unset.

			if (steps_remaining_CPU > 0)
				SetTimer(hWnd, 2, 1, NULL); // 1 millisecond delay

			break;

		case ID_RUN_STOP:

			steps_remaining = 0;
			steps_remaining_CPU = 0;
			break;
		case ID_INITIALISE_IONISATIONSTEPS:
			break;

		default:
			return DefWindowProc(hWnd, message, wParam, lParam);
		}
		break;

	case WM_TIMER:
		
		KillTimer(hWnd, wParam);
		report_time(0);
		if (wParam == 1)
		{
			if (bInvoked_cuSyst == false) {
				bInvoked_cuSyst = true;
				cuSyst_host.InvokeHost();

				pX->EnsureAnticlockwiseTriangleCornerSequences_SetupTriMinorNeighboursLists();
				pX->Average_n_T_to_tris_and_calc_centroids_and_minorpos();
				cuSyst_host.PopulateFromTriMesh(pX);
				cuSyst_host2.InvokeHost();
				cuSyst_host2.PopulateFromTriMesh(pX);
				
				cuSyst_host.Output("n0.txt");

				PerformCUDA_Invoke_Populate(
					&cuSyst_host,
					NUMVERTICES,
					pX->InnermostFrillCentroidRadius,
					pX->OutermostFrillCentroidRadius,
					pX->numStartZCurrentTriangles,
					pX->numEndZCurrentTriangles);
			}

			// Run 1 step:
			printf("evaltime %1.8E\n", evaltime);
			
			PerformCUDA_RunStepsAndReturnSystem_Debug(&cuSyst_host, &cuSyst_host2, pX, &X3);
			cuSyst_host.PopulateTriMesh(pX);

			steps_remaining--;
			GlobalStepsCounter++;

			printf("Done steps: %d   ||   Remaining this run: %d\n\n", GlobalStepsCounter, steps_remaining);
		}
		else {
			pX->Advance(pXnew, &X3);
			temp = pX;
			pX = pXnew;
			pXnew = temp;

			steps_remaining_CPU--;
			GSCCPU++;
			printf("Done steps CPU: %d   ||   Remaining this run: %d\n\n", GSCCPU, steps_remaining_CPU);

			sprintf(buf1000, "autosaveCPU%d.dat", GSCCPU);
			pX->Save(buf1000);
			printf("saved as %s\n", buf1000);
		};
		printf("%s\n", report_time(1));

		/*	
		if (GlobalStepsCounter % GRAPHICS_FREQUENCY == 0)
		{
			// make video frames:
			for (i = 0; i < NUMAVI; i++)
			{
				RefreshGraphs(*pX, GraphFlags[i]); // sends data to graphs AND renders them
															   //	::PlanViewGraphs1(*pX);

															   //RefreshGraphs(*pX,GraphFlags[i]); // sends data to graphs AND renders them
				Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);

				if (DXChk(p_backbuffer_surface->GetDC(&surfdc), 100))
					MessageBox(NULL, "GetDC failed", "oh dear", MB_OK);

							//SelectObject(surfdc,surfbit);
				BitBlt(dibdc, 0, 0, VIDEO_WIDTH, VIDEO_HEIGHT, surfdc, 0, 0, SRCCOPY);
					p_backbuffer_surface->ReleaseDC(surfdc);
					AddAviFrame(hAvi[i], dib);

			};
						sprintf(szFile, "System_%d", GlobalStepsCounter);
						//			pX->SaveText(szFile);
		};*/

		/*
		if (GlobalStepsCounter % (AVI_FILE_PINCHOFF_FREQUENCY * GRAPHICS_FREQUENCY) == 0)
		{
		for (i = 0; i < NUMAVI; i++)
		{
		// now have to pinch out avi file and make a new one
		CloseAvi(hAvi[i]);
		sprintf(buf1000, "%s%s_%d.avi", FOLDER, szAvi[i], GlobalStepsCounter);
		hAvi[i] = CreateAvi(buf1000, AVIFRAMEPERIOD, NULL);
		SetAviVideoCompression(hAvi[i], dib, &opts, false, hWnd);
		};
		};
		*/
		RefreshGraphs(*pX,GlobalSpeciesToGraph); // sends data to graphs AND renders them
		Direct3D.pd3dDevice->Present( NULL, NULL, NULL, NULL );
		
		if (steps_remaining > 0) {
			SetTimer(hWnd, 1, DELAY_MILLISECS, NULL);
			printf("Waiting %d milliseconds to allow user input.\n", DELAY_MILLISECS);
		};
		if (steps_remaining_CPU > 0) {
			SetTimer(hWnd, 2, DELAY_MILLISECS, NULL);
			printf("Waiting %d milliseconds to allow user input.\n", DELAY_MILLISECS);
		};
		if (wParam == 1) {
			sprintf(buf1000, "autosaveGPU%d.dat", GlobalStepsCounter);
		} else {
			sprintf(buf1000, "autosaveCPU%d.dat", GSCCPU);
		}
		pX->Save(buf1000);
		printf("saved as %s\n", buf1000);
		printf("save ascii?");
		
		do {
			o = getch();
		} while ((o != 'y') && (o != 'n'));
		printf("%c\n", o);
		if (o == 'y') {
			pX->SaveText(buf1000);
			printf("Ascii file saved.\n");
		}
		printf("steps_remaining GPU: %d  CPU: %d\n",steps_remaining, steps_remaining_CPU);
		
			// Auto-save system:
		//if (GlobalStepsCounter % DATA_SAVE_FREQUENCY == 0)
		
		break;

	case WM_KEYDOWN:

		switch (wParam)
		{
		case 'W':
			GlobalEye.z += 1.0f;
			printf("GlobalEye %f %f %f  \n",
				GlobalEye.x, GlobalEye.y, GlobalEye.z);
			break;
		case 'S':
			GlobalEye.z -= 1.0f;
			printf("GlobalEye %f %f %f  \n",
				GlobalEye.x, GlobalEye.y, GlobalEye.z);
			break;
		case 'A':
			GlobalEye.x -= 0.8f;
			printf("GlobalEye %f %f %f  \n",
				GlobalEye.x, GlobalEye.y, GlobalEye.z);
			break;
		case 'D':
			GlobalEye.x += 0.8f;
			printf("GlobalEye %f %f %f  \n",
				GlobalEye.x, GlobalEye.y, GlobalEye.z);
			break;
		case 'E':
			GlobalEye.y += 0.8f;
			printf("GlobalEye %f %f %f  \n",
				GlobalEye.x, GlobalEye.y, GlobalEye.z);
			break;
		case 'C':
			GlobalEye.y -= 0.8f;
			printf("GlobalEye %f %f %f  \n",
				GlobalEye.x, GlobalEye.y, GlobalEye.z);
			break;

		case 'V':
			GlobalLookat.z -= 0.4f;
			printf("GlobalEye %f %f %f  GlobalLookat %f %f %f\n",
				GlobalEye.x, GlobalEye.y, GlobalEye.z, GlobalLookat.x, GlobalLookat.y, GlobalLookat.z);
			break;
		case 'R':
			GlobalLookat.z += 0.4f;
			printf("GlobalEye %f %f %f  GlobalLookat %f %f %f\n",
				GlobalEye.x, GlobalEye.y, GlobalEye.z, GlobalLookat.x, GlobalLookat.y, GlobalLookat.z);
			break;
		case 'F':
			GlobalLookat.x -= 0.4f;
			printf("GlobalEye %f %f %f  GlobalLookat %f %f %f\n",
				GlobalEye.x, GlobalEye.y, GlobalEye.z, GlobalLookat.x, GlobalLookat.y, GlobalLookat.z);
			break;
		case 'G':
			GlobalLookat.x += 0.4f;
			printf("GlobalEye %f %f %f  GlobalLookat %f %f %f\n",
				GlobalEye.x, GlobalEye.y, GlobalEye.z, GlobalLookat.x, GlobalLookat.y, GlobalLookat.z);
			break;
		case 'T':
			GlobalLookat.y += 0.4f;
			printf("GlobalLookat %f %f %f\n",
				GlobalLookat.x, GlobalLookat.y, GlobalLookat.z);
			break;
		case 'B':
			GlobalLookat.y -= 0.4f;
			printf("GlobalLookat %f %f %f\n",
				GlobalLookat.x, GlobalLookat.y, GlobalLookat.z);
			break;
		case 'Q':
			GlobalCutaway = !GlobalCutaway;
			break;
		case 'Y':
		case '<':
			GlobalEye.x = -10.4; GlobalEye.y = 16.4; GlobalEye.z = 44.0;
			GlobalLookat.x = -3.6; GlobalLookat.y = 3.0; GlobalLookat.z = 72.2;
			printf("GlobalEye %f %f %f  GlobalLookat %f %f %f\n",
				GlobalEye.x, GlobalEye.y, GlobalEye.z, GlobalLookat.x, GlobalLookat.y, GlobalLookat.z);

			GlobalPlanEye.x = 7.1; GlobalPlanEye.y = 11.5; GlobalPlanEye.z = 71.35;
			printf("GlobalPlanEye %f %f %f\n",
				GlobalPlanEye.x, GlobalPlanEye.y, GlobalPlanEye.z);

			break;
		case '_':
		case '-':
		case '>':
			GlobalPlanEye.x = 7.0; GlobalPlanEye.y = 14.0; GlobalPlanEye.z = 71.0;
			printf("GlobalPlanEye %f %f %f\n",
				GlobalPlanEye.x, GlobalPlanEye.y, GlobalPlanEye.z);
			break;

		case 'U':
			GlobalPlanEye.z += 0.6f;
			printf("GlobalPlanEye %f %f %f\n",
				GlobalPlanEye.x, GlobalPlanEye.y, GlobalPlanEye.z);
			break;
		case 'J':
			GlobalPlanEye.z -= 0.6f;
			printf("GlobalPlanEye %f %f %f\n",
				GlobalPlanEye.x, GlobalPlanEye.y, GlobalPlanEye.z);
			break;
		case 'H':
			GlobalPlanEye.x -= 0.6f;
			printf("GlobalPlanEye %f %f %f\n",
				GlobalPlanEye.x, GlobalPlanEye.y, GlobalPlanEye.z);
			break;
		case 'K':
			GlobalPlanEye.x += 0.6f;
			printf("GlobalPlanEye %f %f %f\n",
				GlobalPlanEye.x, GlobalPlanEye.y, GlobalPlanEye.z);
			break;
		case 'I':
			GlobalPlanEye.y *= 1.25f;
			printf("GlobalPlanEye %f %f %f\n",
				GlobalPlanEye.x, GlobalPlanEye.y, GlobalPlanEye.z);
			break;
		case 'M':
			GlobalPlanEye.y *= 0.8f;
			printf("GlobalPlanEye %f %f %f\n",
				GlobalPlanEye.x, GlobalPlanEye.y, GlobalPlanEye.z);
			break;
		case 'N':
			GlobalboolDisplayMeshWireframe = !GlobalboolDisplayMeshWireframe;
			//Graph1.boolDisplayMeshWireframe = (!(Graph1.boolDisplayMeshWireframe));
			break;
		case '9':
			GlobalRenderLabels = false;
			break;
		case '5':
			GlobalRenderLabels = true;
			GlobalWhichLabels = 0;// iTri
			break;
		case '8':
			GlobalRenderLabels = true;
			GlobalWhichLabels = 1;//T
			break;
		case '7':
			GlobalRenderLabels = true;
			GlobalWhichLabels = 2;//v
			break;
		case '6':
			GlobalRenderLabels = true;
			GlobalWhichLabels = 3;	//n
			break;
		case '1':
			GlobalColoursPlanView = 1;//v
			break;
		case '4':
			GlobalColoursPlanView = 0;//nothing
			break;
		case '2':
			GlobalColoursPlanView = 2;//T
			break;
		case '0':
			steps_remaining = 0;
			break;

		default:
			return DefWindowProc(hWnd, message, wParam, lParam);

		};

		//PlanViewGraphs1(*pX);

		RefreshGraphs(*pX, GlobalSpeciesToGraph); // sends data to graphs AND renders them
		Direct3D.pd3dDevice->Present(NULL, NULL, NULL, NULL);

		break;
	case WM_PAINT:
		// Not sure, do we want to do this?
		//	RefreshGraphs(*pX,); // sends data to graphs AND renders them
		GetUpdateRect(hWnd, &rect, FALSE);
		if (Direct3D.pd3dDevice != NULL)
			Direct3D.pd3dDevice->Present(&rect, &rect, NULL, NULL);

		ValidateRect(hWnd, NULL);
		break;
	case WM_DESTROY:
		DeleteObject(dib);
		DeleteDC(dibdc);
		for (i = 0; i < NUMAVI; i++)
			CloseAvi(hAvi[i]);

		//  _controlfp_s(0, cw, _MCW_EM); // Line A
		PerformCUDA_Revoke();


		PostQuitMessage(0);
		break;
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}
	return 0;
}

// Message handler for about box.
INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
	UNREFERENCED_PARAMETER(lParam);
	switch (message)
	{
	case WM_INITDIALOG:
		return (INT_PTR)TRUE;

	case WM_COMMAND:
		if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
		{
			EndDialog(hDlg, LOWORD(wParam));
			return (INT_PTR)TRUE;
		}
		break;
	}
	return (INT_PTR)FALSE;
}


INT_PTR CALLBACK SetupBox(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
	UNREFERENCED_PARAMETER(lParam);
	char buffer[2048];
	char string[1024];
	real newh;

	switch (message)
	{
	case WM_INITDIALOG:
		sprintf(buffer, "New h? (present = %1.10E)", h);
		if (GlobalSwitchBox)
			SetDlgItemText(hDlg, IDC_STATIC, buffer);
		return (INT_PTR)TRUE;

	case WM_COMMAND:
		if (LOWORD(wParam) == IDOK)
		{
			// try to read data from edit control:
			GetDlgItemText(hDlg, IDC_EDIT1, buffer, 2048);
			if (GlobalSwitchBox == 0)
			{
				// 
				steps_remaining = atoi(buffer);
				if (steps_remaining >= 0)
				{
					EndDialog(hDlg, LOWORD(wParam));
				}
				else {
					MessageBox(NULL, "incorrect value", "Enter a nonnegative integer.", MB_OK);
				};
			}
			else {
				newh = atof(buffer);
				if (newh > 0.0)
				{
					EndDialog(hDlg, LOWORD(wParam));
					sprintf(string, "h = %1.10E\n", newh);
					h = newh;
					MessageBox(NULL, string, "New value of h", MB_OK);
				}
				else {
					MessageBox(NULL, "no good", "Negative h entered", MB_OK);
				};
			};
			return (INT_PTR)TRUE;
		}
		break;
	}
	return (INT_PTR)FALSE;
}

