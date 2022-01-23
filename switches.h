
//#define REMOTE
//#define NOGRAPHICS
//#define OSCILLATE_IZ
//#define AZCG

#include <stdlib.h>
#include <stdio.h> 
#include <conio.h>
#include <math.h>
#include <time.h>
#include <windows.h>  

#define LAPACKE
#define PRECISE_VISCOSITY 
#define DEBUGTE               0
#define SQRTNT           1
#define SQRTNV 
#define EQNS_TOTAL 384 // 1024
#define INNER_EQNS 256 // 512   // half inside, half outside
#define MAXRINGS    96
#define MAXRINGLEN  512  // Note that this is also the max # on the periphery.

#define LIBERALIZED_VISC_SOLVER

#define CHOSEN 8
#define CHOSEN1 14332
#define CHOSEN2 14334 

#define VERTCHOSEN 8
#define VERTCHOSEN2 50

#define REGRESSORS 8
#define SQUASH_POINTS  24

#define VISCMAG 1 
#define MIDPT_A

#define BWDSIDET
#define LONGITUDINAL
#define TEST_OVERALL_V (0) //index == 38799)
#define FOUR_PI 12.5663706143592

#define TEST  (0) // iVertex == VERTCHOSEN) 
#define TEST_ELEC_VISC_TRI (0) //iMinor == CHOSEN)
#define TESTNEUTVISC2 (0) // iMinor == CHOSEN)
#define TESTPRESSUREY (0) //iVertex == VERTCHOSEN)
#define TEST_T (0) // iVertex == VERTCHOSEN) 
#define TEST3  (0)
#define TEST1 (0)
#define TESTTRI (0) // iMinor == VERTCHOSEN + BEGINNING_OF_CENTRAL) // thermal pressure output & infer minor density & momflux_minor
#define TESTADVECT (0)
#define TESTADVECTZ (0)//iVertex == VERTCHOSEN)
#define TESTADVECTNEUT (0) //iVertex == VERTCHOSEN)
#define TESTIONVERTVISC (0)//(iVertex == VERTCHOSEN)
#define TESTNEUTVISC (0) // iVertex == VERTCHOSEN) 
#define TESTVISC (0) // iMinor == VERTCHOSEN + BEGINNING_OF_CENTRAL)
#define TESTIONVISC (0) 
#define TESTHEAT (0)
#define TESTHEATFULL (0)
#define TESTHEAT1 (0)
#define TESTTRI2 (0)
#define TESTTRI3 (0)
#define TESTHEAT2 (0)
#define TESTIONISE (0)
#define TESTOHMS (0) //iMinor == VERTCHOSEN + BEGINNING_OF_CENTRAL)
#define TEST_IONIZE (0) //iVertex == VERTCHOSEN)
#define TESTACCEL (0) //iMinor == VERTCHOSEN + BEGINNING_OF_CENTRAL)
#define TESTACCEL2 (0) //iMinor - BEGINNING_OF_CENTRAL == VERTCHOSEN)
#define TESTACCEL_X (0) // PopOhms output
#define TESTLAP (0)//iVertex == VERTCHOSEN)
#define TESTLAP2 (0) //(iMinor == CHOSEN1) || (iMinor == CHOSEN2))
#define TESTVEZ (0) // iMinor == VERTCHOSEN + BEGINNING_OF_CENTRAL)
#define TEST_VS_MATRIX (0) //iMinor == VERTCHOSEN + BEGINNING_OF_CENTRAL)
#define TEST_VS_MATRIX2 (0) // iVertex == VERTCHOSEN
#define TESTVNX (0)
#define TESTVNY (0) //iMinor == CHOSEN)//PopOhms
#define TESTVNY2 (0) //iMinor == VERTCHOSEN+BEGINNING_OF_CENTRAL) //neutral momflux
#define TESTVNY3 (0)// || (iVertex == VERTCHOSEN2))
#define TESTVNZ (0)//iMinor == VERTCHOSEN + BEGINNING_OF_CENTRAL)
#define TEST_ADV_HEAT_FLAG 0
#define TEST_ADV_MASS_FLAG 0
#define TESTVNXVERT (0)
#define TESTVNYVERT (0)
#define TEST_ACCEL_Y (0) // iMinor == VERTCHOSEN + BEGINNING_OF_CENTRAL)
#define TEST_ACCEL_EZ (0)//iMinor == CHOSEN)
#define TEST_EPSILON_Y (0)
#define TEST_EPSILON_X (0)
#define TEST_EPSILON_Y_IMINOR (0)//iMinor == lChosen)
#define TEST_EPSILON_X_MINOR (0) // iMinor == CHOSEN)

#define ARTIFICIAL_RELATIVE_THRESH  1.0e10 // if we let it be more strict than heat thresh then it drives a difference generating heat!
#define ARTIFICIAL_RELATIVE_THRESH_HEAT  1.0e10   // typical initial density is 1e8 vs 1e18
#define LOW_THRESH_FOR_VISC_CALCS 1.0e10 // density. This should not be too much greater than the density where we do not soak away velocity and heat. At the moment it's 100 times.
#define MINIMUM_NU_EI_DENSITY       1.0e12

