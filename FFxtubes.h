

#ifndef FFXTUBES_h
#define FFXTUBES_h

// Should have thought of better name.
// Lagrangian deterministic 2D plasma filament simulation;
// vertex-based version.

#define real double

#include "constant.h"
#include "resource.h"
#include <math.h>

bool const bScrewPinch = false;

#define FOLDER "C:/outputs/"
#define INITIALAVI "start.avi"		

#define FUNCTIONALFILE_START FOLDER "functionals"
#define DATAFILENAME FOLDER "Data_"
#define AUTOSAVE_FILENAME FOLDER "autosave.dat"
#define RUNTIME_FILENAME FOLDER "runtime.dat"

#define DELAY_MILLISECS      100 // pause

// steps per frame
#define GRAPHICS_FREQUENCY				2
#define STEPS_PER_LOOP               1    // soon change to 500
// frames between file pinch-offs:
#define AVI_FILE_PINCHOFF_FREQUENCY     120
// milliseconds between frames:
#define AVIFRAMEPERIOD         50 // 20 fps which is ok with our setting of 25 frames/ns [running 2e-11 x 2 = 4e-11 per frame]

#define DATA_SAVE_FREQUENCY					  10

// Program Mechanics:
// ==================

#define MAXNEIGH 12 // please keep < 12? It will help.

// 12*32768*5 = 2MB .. just to keep things in perspective.
// We should keep the number down just to reduce fetch size.
// Let's keep it real. nvT is best for our fetches and therefore is best.
#define SWITCH_TO_CENTRE_OF_INTERSECTION_WITH_INSULATOR_FOR_TRI_CENTROID_CPU 0

long const numTriTiles = 288; // note that there are also centrals
long const numTilesMajor = 288;
long const numTilesMinor = 432; // 432 = 288+144
								// 456*256 = 304*256 + 304*128

								// numTriTiles == numTilesMajor because the two sets are bijective.
								// Then we also have to assign central minors to tiles, twice the size of the major tiles...

long const threadsPerTileMinor = 256; // PopOhmsLaw ASSUMES THIS IS A POWER OF 2 !!!
long const threadsPerTileMajor = 128; // see about it - usually we take info from minor.

long const threadsPerTileMajorClever = 256;
long const numTilesMajorClever = 144;

long const SIZE_OF_MAJOR_PER_TRI_TILE = 128;
long const SIZE_OF_TRI_TILE_FOR_MAJOR = 256;
long const BEGINNING_OF_CENTRAL = threadsPerTileMinor*numTriTiles;

long const NUMVERTICES = numTilesMajor*threadsPerTileMajor;//36864; //36000; // particularly applies for polar?
											 // = 288*128
long const NMINOR = threadsPerTileMinor * numTilesMinor;
long const NUMTRIANGLES = NMINOR - NUMVERTICES;


// Model parameters:
//===============================

 // radii in cm from anode centre : 
#define DOMAIN_OUTER_RADIUS   4.64

// reality: cathode rod is 0.47625 cm radius, at 5cm device radius
real const CATHODE_ROD_R = 5.0;
real const CATHODE_ROD_RADIUS = 0.47625;
// Let's say some vertcells will be within it and we acknowledge that asap.

real const DEVICE_RADIUS_INITIAL_FILAMENT_CENTRE = 3.61;   
#define  DEVICE_RADIUS_INSULATOR_OUTER  3.44
#define  REVERSE_ZCURRENT_RADIUS   2.8
#define  INSULATOR_HEIGHT          2.8
							// around the edge of the anode
#define OVER_RADIAL_DIST_ANODE_TO_BACK   0.5434782609
									// 1.0/(4.64-2.8)
#define  INNER_A_BOUNDARY   2.72
				// Safe to assume that within this, Az is constant and Axy declines with r 

#define PHI_EDGE_PPN     0.1172
#define GRADIENT_INSIDE  1.3793
#define MAXSINR          3.5734715649245
				// = 3.44 + pi*0.5/(GRADIENT_INSIDE/PHI_EDGE_PPN)

// Tooth is height 0.63
int const NUM_PLANES = 8;
real const PLANE_INTERCEPT_LINEAR[8] = {0.68,1.1,1.6,2., 2.72, 2.725, 2.73, 2.74}; // 0.5+2.9=3.4
real const PLANE_DZBYDR[8] = {0.0, 0.0, 0.0, 0.0, 0.414213562, 1.0, 2.414213562,-1.0 // vertical
								};
// We probably also need to solve for A near cathode, no?
// But that means leaving points inside tooth - more work.
// Just assume looking down that A is constant. (Current within cathode is being ignored...)

// It certainly could be easier if we could avoid problem with otherwise intersecting or nonlinear dataplanes.

// 22.5 deg, 45 deg, 67.5 deg.

//// real const PLANE_INITIAL_STRETCHFROM = 3.8; // below r = 3.8, use radius in plane as
// DO NOT bother: we can assume instead that tubes do emerge from cathode rod.
// The simulated volume is a cylinder with a dome.


// position for vertex, matching horizontal planes
// Outside this r, stretch out to match x in horizontal :
// where the tube next to the insulator basically
// goes nearly square. <-- different idea.

// To set N need volume of vertcell. To get volume need to set the offset locations.
// OR assume something: linear towards next position. AND cross-section area changes linearly.

//0.577350269,2.747477419};
// 0.577350269 <-- sin/cos 30 degrees
// 1.732050808 <-- sin/cos 60 degrees
// 2.747477419 <-- sin/cos 70 degrees
// 70 chosen as 2/3 point between 30 and 90, so that top cell extends equal amount each way.

//real const PLANE_ACTUAL_INTERCEPT[8] = {0.68, 0.82, 1.6, 2.7, 2.74, 2.75, 2.76, 2.77};
// NOTE: FOR THE ONES THAT ARE RAISED, THESE SHOULD BE WELL BELOW WHERE PLANE INTERCEPTS INSULATOR.

// Problem with flat planes: planes intersect. ? 
// BEST IDEA: Add bend *inside insulator* so that they bend to horizontal.
// We def have to live with tubes that are not orthogonal to planes, 
// though we can optionally move "planes" for efficiency. z(r) or more general.
// BEAR THIS IN MIND as we write the program.
// Yes that's fine but it's easier to start with planar dataplanes nonetheless.

// HAVE 2 EXTRA (inner + outer) vertical sets of A,A-dot values WITHIN ANODE north of insulator.
// THE POSITIONS ARE CHOSEN TO MAKE EVERY TUBE PERPENDICULAR TO THE ANODE.
// THEY SUBTEND THE PLANES THAT BECOME FLAT. 
real const ANODE_VERTICAL_R2 = 2.79;
real const ANODE_VERTICAL_R1 = 2.75; // on the inner side, the cell assumes Az constant and Axy ~ r.
// The positions are projected along tubes from the highest plane.

// Note that the corresponding values must be loaded as "above" the top plane.


long const POINTS_PER_PLANE = 36864; // To have just under is OK
// Initially do we want to plan for unused points? Not sure.
// We can build that it - but we want the total dimensioned to be divisible by 128 etc.



// For info: horizontal planar area at 2.8cm inner, 5cm outer radius = 3.37cm^2 with 1.18cm^2 inside objects.
// So here we are going for roughly 100 micron spacing, 200000 points.


// 2D :
real const ZTOP = 2.9; // OLD
#define EFFECTIVE_PLATEAU_HEIGHT       0.5
// assume near tooth phi=0 at this level.
#define EFFECTIVE_LOGIT_CENTRE         3.84
// assume phi=0 at plateau out to this radius.
#define EFFECTIVE_LOGIT_COEFFICIENT    16.0
// assume phi=0 slopes down to 0 height at this radius.


real const PLANE_Z = 1.0;

// change to logistic: maybe for values too.


//real const INSULATOR_RADIUS_SQ = DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER;
real const PPN_CIRCLE = 1.0/16.0;  // the proportion of a circle for this simulation

//real const HALFANGLE = PI*PPN_CIRCLE; // half the base angle of the slice
#define HALFANGLE 0.196349541

//real const GRADIENT_X_PER_Y = tan(HALFANGLE); 
#define GRADIENT_X_PER_Y    0.198912367

real const GRADIENT_Y_PER_X = 1.0/GRADIENT_X_PER_Y;
real const CUTAWAYANGLE = -0.005;//-GRADIENT_X_PER_Y * 0.5;
// reality: cathode rod is 0.47625 cm radius, at 5cm device radius
// IT DOES NOT LIKE 0!!
// **Clearly need dimensioned index array to achieve some overlap.**
// **Fix problem by going to look for numTrianglesTotal[ **

real const n_INITIAL_SD = 0.04; // 400 micron
real const INITIALnSD = n_INITIAL_SD; // 400 micron
real const INITIALTSD = 0.035; // if n is Gaussian then T is a cone. So make n more spread out than T (?)

real const FILAMENT_OWN_PEAK_n = 9.9e17;
real const UNIFORM_n = 1.0e12;  // ionisation fraction at room temp would be tiny;
							// this is here to help avoid density = zero which causes division issues
real const FILAMENT_OWN_PEAK_T = 4.8e-12;
real const UNIFORM_T= 4.0e-14; // 300K

#define BZ_CONSTANT     2.7  
							   // Constant Bz before any plasma Bz:
					   // 0.3 G from Earth and 2.4 G from coil
real const RELATIVEINITIALJZUNIFORM = 0.0; 
// note it is not wise to have uniform current w/o uniform ion density - can revisit this

real const  ZCURRENTBASETIME = 0.0;//18.0e-9; // start from 0 -- 16/10/17
real const  PEAKCURRENT_AMP = 88000.0;   // 88000 Amp - 2 filaments
// Note that the current is negative from this.
real const  PEAKCURRENT_STATCOULOMB = PEAKCURRENT_AMP * sC_;
real const  PEAKTIME  = 18.0e-7;
real const  PIOVERPEAKTIME = PI/PEAKTIME;

real const SimArea = PI*(DOMAIN_OUTER_RADIUS*DOMAIN_OUTER_RADIUS-DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)/16.0; // pi r^2 -- note to self pls do not keep adding a 2.
real const FULLANGLE = 2.0*PI/16.0;

// " integral Jz = I_peak sin (pi (t + tBASE) / tPEAK ) " - rough_model_with_layer_methods_33a.lyx

// Simulation parameters:
//==================================

real const h_INNER_INIT = 0.5e-14; // initial value
real const h_RECALCULATE = 1.0e-12;
long const INNERMOST_STEPS = 200;
//real const h_MOTION = 1.0e-12; // could make it longer. 1e-12+6 = 1e-6 cm = 0.01 micron.
// Need to be careful in case of thermal pressure messing up mesh.
// Let's assume we do recalculation of parameters as often as we do recalculation of pressure.


real const TIMESTEP = 2.0e-12; //
real const SUBSTEP = 1.0e-13;
int const SUBCYCLES = 20; // Try instead of 1e-13 and see if different.


//long const NUM_VERTICES_PER_CM_SQ = 10000; // 60000; //12000; // use 262144 = 2^18. 2^9 = 512
//long const NUM_VERTICES_PER_CM_SQ_INSIDE_INS = 9000; // 24000; // 8000;
// Make life easier: both same.

int const NUM_SUBSTEPS_IN_h_over_2 = 3; // Visc+Accel, Ionisation+Heating

int const VERTICES_PER_ARRAY = 65536; // for graphics. Just sends warning at the moment.

// We are still going to need to sometimes do multimesh, for a reset maybe, or at least initially.
real const ODE_ABSTHRESHOLD[4] = {1.0e-9,1.0e-9,1.0e-9,1.0e-9};
									// threshold for sqrt(avg squared residual)

// 1.0e-6 --> easily visible errors circa 100ns. 1.0e-8 calms somewhat.
real const ODE_FPRATIO = 1.0e-14; // smth seemed to be giving a too large threshold for Gauss ?!
real const ODE_IZRELTOL = 1.0e-10; // said 1e-10 previously
// We are going to have to chase the current by inputting a fixed amount of electrons.

long const NUM_COARSE_LEVELS = 3; 

//long const NUM_AUX_VERTS_PER_CM_SQ[3] = {1800, 240, 40}; // *total* verts 18000 ..
//long const NUM_AUX_VERTS_PER_CM_SQ[3] = {3600, 420, 60}; // 24000 per cm sq
//long const NUM_AUX_VERTS_PER_CM_SQ[3] = {1800, 220, 28}; // 16000 per cm sq ..
long const ITERATIONS_at_levels[7] = {4,8,12,16,20,24,256}; // {3, 15, 75}
	// IterationsJRLS(iLevel,ITERATIONS_at_levels[iLevel+1]);				
			
int const CYCLEPEAKS = 1;
int const COARSE_LIMIT = 12;
bool const QUADWEIGHTS = false;

int const SWIM_FREQUENCY = 20; // how many turns between swims
int const SWIM_COUNT = 2; // max # of swims to do

real const hLARGE = 1.0e-9; // choose a large time for flows
// to be used initially.
real const maximum_spacing = 0.025;  // cm: do not make cell widths larger than this

// Graphics related
// ================


float const NEAR_CLIPPING_PLANE = 0.1f;
float const FAR_CLIPPING_PLANE = 200.0f;

extern float xzscale;

#define NEGATE_JZ

//long const VIDEO_WIDTH = 1440;
//long const VIDEO_HEIGHT = 900;  // the client area size; used for 4 graphs(at present)

long const VIDEO_WIDTH = 1440;
long const VIDEO_HEIGHT = 1050;  // the client area size; used for 4 graphs(at present)

								 // CHANGE THIS TO MAX DIMS ---- *** ??? *** ???

long const SCREEN_WIDTH = 1920;//1440;
long const SCREEN_HEIGHT = 1080;//900; // affects how window and console are placed

long const GRAPH_WIDTH = VIDEO_WIDTH/2; // /3
long const GRAPH_HEIGHT = VIDEO_HEIGHT/2;
long const NUMGRAPHS = 4;

float const GRAPHIC_MIN_Y = 0.0;
float const GRAPHIC_MAX_Y = 4.5;

real const GRAPH_SCALE_GEOMETRIC_INCREMENT = 1.333521452; // 10^(1/8)


// SCREW PINCH / Z PINCH
// =====================

real const IZ_SCREW_PINCH = 44000.0*sC_; // 44000 Amp in statCoulomb

real const N0_centre_SP = 1.0e18; // central n initially
real const n_UNIFORM_SP = 1.0e15;
real const a_nSD = 0.05;	// cm		// n = N0 exp(-r^2/ a_nSD^2)  -- so it's not really the SD
real const OUTER_RADIUS = 0.6;		 // cm
real const REVERSE_CURRENT_RADIUS_SP = 0.58; // cm
real const SCREWPINCH_INITIAL_T = 15.0*1.6e-12; // about 15eV in erg
real const SP_CENTRE_Y = 3.61;			// translate origin

long const NUM_AUX_VERTS[3] = {6000, 720, 100}; // total verts ?


#endif

