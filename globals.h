#ifndef globals_h
#define globals_h


#define f64 double
#define qd_or_d dd_real

#include "surfacegraph_tri.h"
#include "vector_tensor.cu"

// This is for extern declarations of the global variables used.
// Definitions should live in DropJxy.cpp

// Can also put extern declaration of global functions here.

extern bool GlobalFlagNeedPeriodicImage;
extern smartlong GlobalVertexScratchList;
//extern int nesting;
extern long GlobalVerticesInRange;
extern long GlobalTrianglesInfluencing;
extern bool GlobalPeriodicSearch;
extern float maximum_v;

extern bool bCullNone,bGlobalsave;
extern f64 evaltime, h;                             
extern int GlobalSwitchBox, iGlobalScratch;

extern f64 GlobalHeightScale;
extern int GlobalSpeciesToGraph;
extern int GlobalWhichLabels;
extern bool GlobalRenderLabels;
extern int GlobalColoursPlanView;
extern bool GlobalBothSystemsInUse;
extern f64 GlobalRescaling;
extern bool GlobalCutaway;
extern f64 GlobalIzElasticity;

// Global pointers to meshes:
extern TriMesh * pX, * pXnew;
extern TriMesh X1, X2;
//FixedMesh Fixed;			// the auxiliary fixed mesh for solving A

//#include "MeshUtil.cpp" 
//#include "simulation.cpp"  // defines simulation methods

extern long steps_remaining, GlobalStepsCounter, steps_remaining_CPU;
extern int GlobalGraphSetting[8];
extern surfacegraph Graph[8];
//extern D3DXVECTOR3 GlobalPlanEye2;
extern float xzscale; // for graphs -- how to handle?
extern bool boolDisplayInnerMesh;
extern smartlong GlobalVertexScratchList;
extern float Historic_max[512][HISTORY]; // if max is falling, use historic maximum for graph.
extern float Historic_min[512][HISTORY];
extern int GlobalWedgeSwitch; 
extern smartlong GlobalAffectedTriIndexList;

extern Tensor2 const Anticlockwise;
extern Tensor2 const Clockwise;
extern Tensor3 const Clockwise3;
extern Tensor3 const Anticlockwise3;
extern Tensor2 const HalfAnticlockwise;
extern Tensor2 const HalfClockwise;


// *************************
// Global functions:

extern int test_dependency(void);

extern void RefreshGraphs(TriMesh & X, const int graphflag);
extern void CreateSurfaceGraphs(TriMesh & X);
extern void CreateSurfaceGraphsAux(TriMesh & X, int iLevel);

extern void QuickSort (long VertexIndexArray[], f64 radiusArray[],
		long lo, long hi); // in meshutil.cpp

extern Vector2 CreateOutwardNormal(f64 x1, f64 y1,
					f64 x2, f64 y2,
					f64 x, f64 y); // in basics.cpp
extern void GetInterpolationCoefficients( f64 beta[3],
							f64 x, f64 y,
							Vector2 pos0, Vector2 pos1, Vector2 pos2); // basics.cpp

extern void GetInsulatorIntercept(Vector2 *result, const Vector2 & x1, const Vector2 & x2);

extern void Get_ROC_InsulatorIntercept(Vector2 * pROCintercept1,
								Vector2 lower , Vector2 moving,Vector2 ROC);

extern f64 GetCos(const Vector2 & u1, const Vector2 & centre, const Vector2 & u2);
extern f64 GetCos(const Vector2 & v1, const Vector2 & v2);

extern f64 CalculateTriangleIntersectionArea(Vector2 & x1, Vector2 & x2, Vector2 & x3,					          
									   Vector2 & r1, Vector2 & r2, Vector2 & r3);

#endif