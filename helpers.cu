
// Device routines that can be #included by the kernels file.

#include "mesh.h" 
#include "FFxtubes.h"
#include "cuda_struct.h"
#include "flags.h"
#include "kernel.h"
#include "matrix_real.h"
#include "globals.h"
#include "headers.h"
#include <stdlib.h>
#include <stdio.h> 
#include <conio.h>
#include <math.h>
#include <time.h>
#include <windows.h>

//extern __constant__ long NumInnerFrills_d, FirstOuterFrill_d;
//extern __constant__ long DebugFlag;
//extern __constant__ long nBlocks, Nverts, uDataLen_d; // Nverts == numVertices 
//extern __constant__ f64_tens2 Anticlockwise_d, Clockwise_d; // use this to do rotation.   
//extern __constant__ f64 kB, c, q, m_e, m_ion, m_i, m_n,
//eoverm, qoverM, moverM, qovermc, qoverMc,
//FOURPI_Q_OVER_C, FOURPI_Q, FOURPI_OVER_C,
//one_over_kB, one_over_kB_cubed, kB_to_3halves,
//NU_EI_FACTOR, nu_eiBarconst, Nu_ii_Factor,
//M_i_over_in,// = m_i / (m_i + m_n);
//M_e_over_en,// = m_e / (m_e + m_n);
//M_n_over_ni,// = m_n / (m_i + m_n);
//M_n_over_ne,// = m_n / (m_e + m_n);
//M_en, //= m_e * m_n / ((m_e + m_n)*(m_e + m_n));
//M_in, // = m_i * m_n / ((m_i + m_n)*(m_i + m_n));
//M_ei, // = m_e * m_i / ((m_e + m_i)*(m_e + m_i));
//m_en, // = m_e * m_n / (m_e + m_n);
//m_ei, // = m_e * m_i / (m_e + m_i);
//over_sqrt_m_ion, over_sqrt_m_e, over_sqrt_m_neutral,
//over_m_e, over_m_i, over_m_n,
//four_pi_over_c_ReverseJz, RELTHRESH_AZ_d,
//FRILL_CENTROID_OUTER_RADIUS_d, FRILL_CENTROID_INNER_RADIUS_d;
//extern __constant__ long lChosen;
//extern __constant__ f64 UNIFORM_n_d;
//extern __constant__ f64 cross_s_vals_viscosity_ni_d[10], cross_s_vals_viscosity_nn_d[10],
//cross_T_vals_d[10], cross_s_vals_MT_ni_d[10];
//extern __constant__ f64 beta_n_c[32], beta_i_c[8], beta_e_c[8];
//extern __constant__ long iEqnsDevice, ActualInnerEqnsDevice;
//extern __constant__ bool bSwitch;
//extern __constant__ f64 recomb_coeffs[32][3][5];
//extern __constant__ f64 ionize_coeffs[32][5][5];
//extern __constant__ f64 ionize_temps[32][10];
//extern __constant__ long MyMaxIndex;
//__device__ extern __constant__ f64 billericay;
//extern __constant__ f64 Ez_strength;
//extern __constant__ f64 negative_Iz_per_triangle; // -Iz_prescribed / (f64)(numEndZCurrentTriangles - numStartZCurrentTriangles)
//extern __constant__ long numStartZCurrentTriangles, numEndZCurrentTriangles;




#ifdef __CUDACC__
__device__ __forceinline__ f64 GetEzShape(f64 x, f64 y) {


	f64 rr = x*x + y*y;
	f64 r = sqrt(rr);
	f64 retval = 1.0 - 1.0 / (1.0 + exp(-24.0*(r - 4.4)));

	// This is max 1. Now let's add 29% over each tooth.

	// Let's assume that based on radius we linearly increase & decrease the % of full tooth.

	// It's 45 degrees we are told.

	// And 0.04 x 0.08 plateau.

	// Thus take 3.61-0.04 = 3.57, and we then only drop by 0.13 (to 0.5cm height) as we go back
	// to the insulator.
	// 3.61+0.04 = 3.65, and it takes us 0.63 cm to fall 0.63 cm. So 4.28 

#define TOOTH_RADIUS_START  3.4925
#define TOOTH_RADIUS_PEAK1   3.58
#define TOOTH_RADIUS_PEAK2   3.64  // Doing 0.06 cm long plateau !
#define TOOTH_RADIUS_OUTER  3.8989
#define TOOTH_HEIGHT        0.63
#define VALLEY_HEIGHT       0.30

	//#define VALLEY_YoverX       0.0
	//#define PEAK_YoverX         10.15317039

	// where north = pi/2
#define VALLEY_ANGLE        1.374446786
#define VALLEY_ANGLE2       1.767145868

#define PLATEAU_ANGLE_11    1.467070459
	//#define PEAK_ANGLE          1.472621556
#define PLATEAU_ANGLE_12    1.478172459
#define PLATEAU_ANGLE_21    1.66342
	//#define PEAK_ANGLE2         1.668971097
#define PLATEAU_ANGLE_22    1.674522

#define VALLEY_YoverX_2     5.02734

	// Note that if we use division we don't save anything.

	f64 theta = atan2(y, x); // might want to check what it returns.
	f64 maxh, maxh2;

	if ((r > TOOTH_RADIUS_START) && (r < TOOTH_RADIUS_OUTER))
	{
		if (r < TOOTH_RADIUS_PEAK1)
		{
			maxh = (r - TOOTH_RADIUS_START)*TOOTH_HEIGHT / (TOOTH_RADIUS_PEAK1 - TOOTH_RADIUS_START);
			// That's for the plane facing up to the point radially.
		}
		else {
			if (r > TOOTH_RADIUS_PEAK2) {
				maxh = (TOOTH_RADIUS_OUTER - r)*TOOTH_HEIGHT / (TOOTH_RADIUS_OUTER - TOOTH_RADIUS_PEAK2);
			}
			else {
				maxh = TOOTH_HEIGHT;
			};
		};

		// Consider plane facing the other way : azimuthal :
		if (theta < M_PI*0.5)
		{
			if (theta > PLATEAU_ANGLE_12)
			{
				maxh2 = VALLEY_HEIGHT +
					(M_PI*0.5 - theta)*(TOOTH_HEIGHT - VALLEY_HEIGHT) /
					(M_PI*0.5 - PLATEAU_ANGLE_12);
			}
			else {
				if (theta > PLATEAU_ANGLE_11)
				{
					maxh2 = TOOTH_HEIGHT;
				}
				else {
					maxh2 = VALLEY_HEIGHT +
						(theta - VALLEY_ANGLE)*(TOOTH_HEIGHT - VALLEY_HEIGHT) /
						(PLATEAU_ANGLE_11 - VALLEY_ANGLE);
				};
			};
		}
		else {
			if (theta < PLATEAU_ANGLE_21)
			{
				maxh2 = VALLEY_HEIGHT +
					(theta - M_PI*0.5)*(TOOTH_HEIGHT - VALLEY_HEIGHT) /
					(PLATEAU_ANGLE_21 - M_PI*0.5);
			}
			else {
				if (theta < PLATEAU_ANGLE_22) {
					maxh2 = TOOTH_HEIGHT;
				}
				else {
					maxh2 = VALLEY_HEIGHT +
						(theta - VALLEY_ANGLE2)*(TOOTH_HEIGHT - VALLEY_HEIGHT) /
						(PLATEAU_ANGLE_22 - VALLEY_ANGLE2);
				};
			};
		};
		if (maxh2 < maxh) maxh = maxh2;// = min(maxh, maxh2); // cut one plane with the other.

								   // Uplift:

		retval *= INSULATOR_HEIGHT / (INSULATOR_HEIGHT - maxh);
		// 2.8/(2.8-0.63) = 1.29		
	};

	//retval /= (1.0 - 0.1*exp((r - 3.61)*(r - 3.61)));

	return retval;
	// 4.48: At 4.0cm it is 96% as strong as at tooth. At 4.4 it is 4%.
	// 4.48 -- used to say 4.32 before 377 ns



}
#else
f64 inline GetEzShape_(f64 r) {
	return 1.0 - 1.0 / (1.0 + exp(-16.0*(r - 4.2))); // At 4.0cm it is 96% as strong as at tooth. 4.2 50%. At 4.4 it is 4%.
}
#endif

__device__ f64 ArtificialUpliftFactor_MT(f64 n_i, f64 n_n)
{
	if (n_i > 1.0e13) return 1.0;
	// Used in crushing v to be hydrodynamic and in viscous ita.

	f64 additional_nn = min(exp(-n_i*n_i / 0.5e24)*(1.0e30 / (n_i)), 1.0e20); // high effective density to produce hydrodynamics
																			  // n <= 1e10 : additional_nn ~= 1e20
																			  // n == 1e11 : additional_nn ~= 1e19
																			  // n == 1e12 : additional_nn ~= 1e17
																			  // n == 1e13 : additional_nn ~= 1e-70
	return 1.0 + additional_nn / n_n;
}


__device__ __forceinline__ f64 Get_lnLambda_ion_d(f64 n_ion,f64 T_ion)
{
	// Assume static f64 const is no good in kernel.

	f64 factor, lnLambda_sq;
	f64 Tion_eV3 = T_ion*T_ion*T_ion*one_over_kB_cubed;
	f64 lnLambda = 23.0 - 0.5*log(n_ion/Tion_eV3); 

	// floor at 2.0:
	lnLambda_sq = lnLambda*lnLambda;
	factor = 1.0+0.5*lnLambda+0.25*lnLambda_sq+0.125*lnLambda*lnLambda_sq + 0.0625*lnLambda_sq*lnLambda_sq;
	lnLambda += 2.0/factor;

	return lnLambda;
}		

__device__ __forceinline__ f64 Get_lnLambda_d(f64 n_e,f64 T_e)
{
	f64 lnLambda, factor, lnLambda_sq, lnLambda1, lnLambda2;

	f64 Te_eV = T_e*one_over_kB;
	f64 Te_eV2 = Te_eV*Te_eV;
	f64 Te_eV3 = Te_eV*Te_eV2;

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

		// Golant p.40 warns that it becomes invalid when an electron gyroradius is less than a Debye radius.
		// That is something to worry about if  B/400 > n^1/2 , so looks not a big concern.

		// There is also a quantum ceiling. It will not be anywhere near. At n=1e20, 0.5eV, the ceiling is only down to 29; it requires cold dense conditions to apply.

		if (lnLambda < 2.0) lnLambda = 2.0; // deal with negative inputs

	} else {
		lnLambda = 20.0;
	};
	return lnLambda;
}		



__device__ __forceinline__ void CalculateCircumcenter(f64_vec2 * p_cc, f64_vec2 poscorner0, f64_vec2 poscorner1, f64_vec2 poscorner2)
{
	f64_vec2 Bb = poscorner1 - poscorner0;
	f64_vec2 C = poscorner2 - poscorner0;
	f64 D = 2.0*(Bb.x*C.y - Bb.y*C.x);
	f64 modB = Bb.x*Bb.x + Bb.y*Bb.y;
	f64 modC = C.x*C.x + C.y*C.y;
	p_cc->x = (C.y*modB - Bb.y*modC) / D + poscorner0.x;
	p_cc->y = (Bb.x*modC - C.x*modB) / D + poscorner0.y;
	// formula agrees with wikipedia so why does it give a stupid result.
}


__device__ __forceinline__ bool TestDomainPos(f64_vec2 pos)
{
	return (
		(pos.x*pos.x + pos.y*pos.y > DEVICE_RADIUS_INSULATOR_OUTER*DEVICE_RADIUS_INSULATOR_OUTER)
		&&
		(pos.x*pos.x + (pos.y - CATHODE_ROD_R_POSITION)*(pos.y - CATHODE_ROD_R_POSITION) > CATHODE_ROD_RADIUS*CATHODE_ROD_RADIUS)
		);
}

__device__ __forceinline__ f64_vec2 Anticlock_rotate2(const f64_vec2 arg)
{
	f64_vec2 result;
	result.x = Anticlockwise_d.xx*arg.x+Anticlockwise_d.xy*arg.y;
	result.y = Anticlockwise_d.yx*arg.x+Anticlockwise_d.yy*arg.y;
	return result;
}
__device__ __forceinline__ f64_vec2 Clockwise_rotate2(const f64_vec2 arg)
{
	f64_vec2 result;
	result.x = Clockwise_d.xx*arg.x+Clockwise_d.xy*arg.y;
	result.y = Clockwise_d.yx*arg.x+Clockwise_d.yy*arg.y;
	return result;
}

__device__ __forceinline__ f64_vec3 Anticlock_rotate3(const f64_vec3 arg)
{
	f64_vec3 result;
	result.x = Anticlockwise_d.xx*arg.x+Anticlockwise_d.xy*arg.y;
	result.y = Anticlockwise_d.yx*arg.x+Anticlockwise_d.yy*arg.y;
	result.z = arg.z;
	return result;
}
__device__ __forceinline__ f64_vec3 Clockwise_rotate3(const f64_vec3 arg)
{
	f64_vec3 result;
	result.x = Clockwise_d.xx*arg.x+Clockwise_d.xy*arg.y;
	result.y = Clockwise_d.yx*arg.x+Clockwise_d.yy*arg.y;
	result.z = arg.z;
	return result;
}

__device__  __forceinline__ void Estimate_Ion_Neutral_Cross_sections_d(f64 T, // call with T in electronVolts
	f64 * p_sigma_in_MT,
	f64 * p_sigma_in_visc)
{
	if (T > cross_T_vals_d[9]) {
		*p_sigma_in_MT = cross_s_vals_MT_ni_d[9];
		*p_sigma_in_visc = cross_s_vals_viscosity_ni_d[9];
		return;
	}
	if (T < cross_T_vals_d[0]) {
		*p_sigma_in_MT = cross_s_vals_MT_ni_d[0];
		*p_sigma_in_visc = cross_s_vals_viscosity_ni_d[0];
		return;
	}
	int i = 1;
	//while (T > cross_T_vals_d[i]) i++;

	if (T > cross_T_vals_d[5]) {
		if (T > cross_T_vals_d[7]) {
			if (T > cross_T_vals_d[8])
			{
				i = 9; // top of interval
			}
			else {
				i = 8;
			};
		}
		else {
			if (T > cross_T_vals_d[6]) {
				i = 7;
			}
			else {
				i = 6;
			};
		};
	}
	else {
		if (T > cross_T_vals_d[3]) {
			if (T > cross_T_vals_d[4]) {
				i = 5;
			}
			else {
				i = 4;
			};
		}
		else {
			if (T > cross_T_vals_d[2]) {
				i = 3;
			}
			else {
				if (T > cross_T_vals_d[1]) {
					i = 2;
				}
				else {
					i = 1;
				};
			};
		};
	};
	// T lies between i-1,i
	f64 ppn = (T - cross_T_vals_d[i - 1]) / (cross_T_vals_d[i] - cross_T_vals_d[i - 1]);

	*p_sigma_in_MT = ppn * cross_s_vals_MT_ni_d[i] + (1.0 - ppn)*cross_s_vals_MT_ni_d[i - 1];
	*p_sigma_in_visc = ppn * cross_s_vals_viscosity_ni_d[i] + (1.0 - ppn)*cross_s_vals_viscosity_ni_d[i - 1];
	return;
}

__device__ __forceinline__ f64 Estimate_Neutral_MT_Cross_section_d(f64 T)
{
	// CALL WITH T IN eV

	if (T > cross_T_vals_d[9]) return cross_s_vals_MT_ni_d[9];		
	if (T < cross_T_vals_d[0]) return cross_s_vals_MT_ni_d[0];
	
	int i = 1;
	//while (T > cross_T_vals_d[i]) i++;
	
	if (T > cross_T_vals_d[5]) {
		if (T > cross_T_vals_d[7]) {
			if (T > cross_T_vals_d[8])
			{
				i = 9; // top of interval
			} else {
				i = 8;
			};
		} else {
			if (T > cross_T_vals_d[6]) {
				i = 7;
			} else {
				i = 6;
			};
		};
	} else {
		if (T > cross_T_vals_d[3]) {
			if (T > cross_T_vals_d[4]) {
				i = 5;
			} else {
				i = 4;
			};
		} else {
			if (T > cross_T_vals_d[2]) {
				i = 3;
			} else {
				if (T > cross_T_vals_d[1]) {
					i = 2;
				} else {
					i = 1;
				};
			};
		};
	}; 
	
	// T lies between i-1,i
	f64 ppn = (T-cross_T_vals_d[i-1])/(cross_T_vals_d[i]-cross_T_vals_d[i-1]);
	return ppn*cross_s_vals_MT_ni_d[i] + (1.0-ppn)*cross_s_vals_MT_ni_d[i-1];

}

__device__ __forceinline__ f64 Estimate_Neutral_Neutral_Viscosity_Cross_section_d(f64 T) 
{
	// call with T in electronVolts
	
	if (T > cross_T_vals_d[9]) return cross_s_vals_viscosity_nn_d[9];
	if (T < cross_T_vals_d[0]) return cross_s_vals_viscosity_nn_d[0];

	int i = 1;
	//while (T > cross_T_vals_d[i]) i++;
	
	if (T > cross_T_vals_d[5]) {
		if (T > cross_T_vals_d[7]) {
			if (T > cross_T_vals_d[8])
			{
				i = 9; // top of interval
			} else {
				i = 8;
			};
		} else {
			if (T > cross_T_vals_d[6]) {
				i = 7;
			} else {
				i = 6;
			};
		};
	} else {
		if (T > cross_T_vals_d[3]) {
			if (T > cross_T_vals_d[4]) {
				i = 5;
			} else {
				i = 4;
			};
		} else {
			if (T > cross_T_vals_d[2]) {
				i = 3;
			} else {
				if (T > cross_T_vals_d[1]) {
					i = 2;
				} else {
					i = 1;
				};
			};
		};
	}; 

	// T lies between i-1,i
	f64 ppn = (T-cross_T_vals_d[i-1])/(cross_T_vals_d[i]-cross_T_vals_d[i-1]);
	return ppn*cross_s_vals_viscosity_nn_d[i] + (1.0-ppn)*cross_s_vals_viscosity_nn_d[i-1];
}

__device__ __forceinline__ f64 Estimate_Ion_Neutral_Viscosity_Cross_section(f64 T)
{
	if (T > cross_T_vals_d[9]) return cross_s_vals_viscosity_ni_d[9];		
	if (T < cross_T_vals_d[0]) return cross_s_vals_viscosity_ni_d[0];
	
	int i = 1;
	//while (T > cross_T_vals_d[i]) i++;
	
	if (T > cross_T_vals_d[5]) {
		if (T > cross_T_vals_d[7]) {
			if (T > cross_T_vals_d[8])
			{
				i = 9; // top of interval
			} else {
				i = 8;
			};
		} else {
			if (T > cross_T_vals_d[6]) {
				i = 7;
			} else {
				i = 6;
			};
		};
	} else {
		if (T > cross_T_vals_d[3]) {
			if (T > cross_T_vals_d[4]) {
				i = 5;
			} else {
				i = 4;
			};
		} else {
			if (T > cross_T_vals_d[2]) {
				i = 3;
			} else {
				if (T > cross_T_vals_d[1]) {
					i = 2;
				} else {
					i = 1;
				};
			};
		};
	}; 
	
	// T lies between i-1,i
	f64 ppn = (T-cross_T_vals_d[i-1])/(cross_T_vals_d[i]-cross_T_vals_d[i-1]);
	return ppn*cross_s_vals_viscosity_ni_d[i] + (1.0-ppn)*cross_s_vals_viscosity_ni_d[i-1];
}


__device__ __forceinline__ f64 Calculate_Kappa_Neutral(f64 n_i, f64 T_i, f64 n_n, f64 T_n)
{
	// NOTE:
	// It involves sqrt and we could easily find a way to calculate only once.
		
	if (n_n == 0.0) return 0.0;

	f64 s_in_visc, s_nn_visc;

	s_in_visc = Estimate_Ion_Neutral_Viscosity_Cross_section(T_i*one_over_kB);
	s_nn_visc = Estimate_Neutral_Neutral_Viscosity_Cross_section_d(T_n*one_over_kB);

	// Oh. So there's another two we have to port.
	// Yet for ion eta it's so different, apparently.
	
	f64 ionneut_thermal = sqrt(T_i/m_ion+T_n/m_n);
	f64	nu_ni_visc = n_i*s_in_visc*ionneut_thermal;
	f64	nu_nn_visc = n_n*s_nn_visc*sqrt(T_n/m_n);
	f64	nu_nheart = 0.75*nu_ni_visc + 0.25*nu_nn_visc;
	f64 kappa_n = NEUTRAL_KAPPA_FACTOR*n_n*T_n/(m_n*nu_nheart);
	// NEUTRAL_KAPPA_FACTOR should be in constant.h
	// e-n does not feature.
	return kappa_n;
}


__device__ __forceinline__ void Get_kappa_parallels_and_nu_hearts
				(f64 n_n,f64 T_n,f64 n_i,f64 T_i,f64 n_e,f64 T_e,
				f64 * pkappa_neut, f64 * pnu_nheart, 
				f64 * pkappa_ion_par, f64 * pnu_iheart,
				f64 * pkappa_e_par, f64 * pnu_eheart,
				f64 * pratio)
{
	f64 s_in_visc, s_nn_visc, s_en_visc;

	f64 ionneut_thermal, 
		nu_ni_visc, nu_nn_visc, nu_nheart,
		nu_in_visc, nu_en_visc, nu_ii, nu_iheart, nu_eheart,
		sqrt_Te, electron_thermal, nu_eiBar;
	 
	f64 lnLambda = Get_lnLambda_ion_d(n_i,T_i);

	ionneut_thermal = sqrt(T_i/m_ion+T_n/m_n);
	sqrt_Te = sqrt(T_e);
	
	s_in_visc = Estimate_Ion_Neutral_Viscosity_Cross_section(T_i*one_over_kB);
	s_nn_visc = Estimate_Neutral_Neutral_Viscosity_Cross_section_d(T_n*one_over_kB);
	
	nu_in_visc = n_n*s_in_visc*ionneut_thermal;
	nu_nn_visc = n_n*s_nn_visc*sqrt(T_n/m_n);
	nu_ni_visc = n_i*s_in_visc*ionneut_thermal;
	
	nu_ii = Nu_ii_Factor*kB_to_3halves*n_i*lnLambda/(T_i*sqrt(T_i));

	nu_iheart = 0.75*nu_in_visc
			+ 0.8*nu_ii-0.25*nu_in_visc*nu_ni_visc/(3.0*nu_ni_visc+nu_nn_visc);
	*pkappa_ion_par = 2.5*n_i*T_i/(m_ion*(nu_iheart));
	*pnu_iheart = nu_iheart;

	s_en_visc = Estimate_Ion_Neutral_Viscosity_Cross_section(T_e*one_over_kB);
	electron_thermal = (sqrt_Te*over_sqrt_m_e);
	
	lnLambda = Get_lnLambda_d(n_e,T_e);
	
	nu_eiBar = nu_eiBarconst*kB_to_3halves*n_i*lnLambda/(T_e*sqrt_Te);
	nu_en_visc = n_n*s_en_visc*electron_thermal;
	nu_eheart = 1.87*nu_eiBar + nu_en_visc;
	*pnu_eheart = nu_eheart;
	*pkappa_e_par =  2.5*n_e*T_e/(m_e*nu_eheart);

	// Store ratio for thermoelectric use:
	*pratio = nu_eiBar/nu_eheart;


	if (n_n == 0.0){
		*pkappa_neut = 0.0;
	} else {

		nu_nheart = 0.75*nu_ni_visc + 0.25*nu_nn_visc;
		*pkappa_neut = NEUTRAL_KAPPA_FACTOR*n_n*T_n/(m_n*nu_nheart);
		*pnu_nheart = nu_nheart;
		// NEUTRAL_KAPPA_FACTOR should be in constant.h
		// e-n does not feature.
	};
	 
}
__device__ __forceinline__ void RotateClockwise(f64_vec3 & v)
{
	f64 temp = Clockwise_d.xx*v.x + Clockwise_d.xy*v.y;
	v.y = Clockwise_d.yx*v.x + Clockwise_d.yy*v.y;
	v.x = temp;
}
__device__ __forceinline__ void RotateAnticlockwise(f64_vec3 & v)
{
	f64 temp = Anticlockwise_d.xx*v.x + Anticlockwise_d.xy*v.y;
	v.y = Anticlockwise_d.yx*v.x + Anticlockwise_d.yy*v.y;
	v.x = temp;
}

__device__ __forceinline__ f64_vec2 GetRadiusIntercept(f64_vec2 x1,f64_vec2 x2,f64 r)
{
	// where we meet radius r on the line passing through u0 and u1?
	f64_vec2 result;
	
	f64 den = (x2.x-x1.x)*(x2.x-x1.x) + (x2.y - x1.y)*(x2.y - x1.y) ;
	f64 a = (x1.x * (x2.x-x1.x) + x1.y * (x2.y-x1.y) ) / den;
	// (t + a)^2 - a^2 = (  c^2 - x1.x^2 - x1.y^2  )/den
	f64 root = sqrt( (r*r- x1.x*x1.x - x1.y*x1.y)/den + a*a ) ;
	f64 t1 = root - a;
	f64 t2 = -root - a;
	
	// since this is a sufficient condition to satisfy the circle, this probably means that
	// the other solution is on the other side of the circle.
	// Which root is within x1, x2 ? Remember x2 would be t = 1.

	if (t1 > 1.0) 
	{
		if ((t2 < 0.0) || (t2 > 1.0))
		{	
			// This usually means one of the points actually is on the curve.
			f64 dist1 = min(fabs(t1-1.0),fabs(t1));
			f64 dist2 = min(fabs(t2-1.0),fabs(t2));
			if (dist1 < dist2)
			{
				// use t1				
				result.x = x1.x + t1*(x2.x-x1.x);
				result.y = x1.y + t1*(x2.y-x1.y);
		//		printf("t1@@");
			} else {
				// use t2				
				result.x = x1.x + t2*(x2.x-x1.x);
				result.y = x1.y + t2*(x2.y-x1.y);
		//		printf("t2@@");
			};
		} else {		
			// use t2:		
			result.x = x1.x + t2*(x2.x-x1.x);
			result.y = x1.y + t2*(x2.y-x1.y);
		//	printf("t2~");
		};
	} else {
		result.x = x1.x + t1*(x2.x-x1.x);
		result.y = x1.y + t1*(x2.y-x1.y);	
		//printf("t1~");
	};

	// For some reason this is only hitting the radius to single precision.

	// printf to compare difference between achieved radius and r.
	
	//if ((result.x < -0.145) && (result.x > -0.155))
	//{
	//	f64 achieve = result.modulus();
	//	printf("ach %1.12E r %1.2f t1 %1.10E \nx %1.12E y %1.12E\n",achieve,r,t1,result.x,result.y);
	//}

	// So what do we do?

	// We could boost back but there seem to be bigger problems thereafter.

	// Ideally we'd go through and compare and see, is it t1 that is a bit wrong here?
	// 

	return result;
}


__device__ void Augment_JacobeanNeutral(
	f64_tens3 * pJ,
	f64 Factor, //h_over (N m_i)
	f64_vec2 edge_normal,
	f64 ita_par, f64 nu, f64_vec3 omega,
	f64 grad_vjdx_coeff_on_vj_self,
	f64 grad_vjdy_coeff_on_vj_self
) {

	//Pi_zx = -ita_par*(gradviz.x);
	//Pi_zy = -ita_par*(gradviz.y);		
	//	visc_contrib.x = -over_m_i*(Pi_xx*edge_normal.x + Pi_xy*edge_normal.y);
	// The z direction doesn't feature vx --- that is because dvx/dz == 0

	pJ->xx += Factor*
		((
			// Pi_zx
			-ita_par*grad_vjdx_coeff_on_vj_self
			)*edge_normal.x + (
				// Pi_zy
				-ita_par*grad_vjdy_coeff_on_vj_self
				)*edge_normal.y);

	pJ->yy += Factor*
		((
			// Pi_zx
			-ita_par*grad_vjdx_coeff_on_vj_self
			)*edge_normal.x + (
				// Pi_zy
				-ita_par*grad_vjdy_coeff_on_vj_self
				)*edge_normal.y);

	pJ->zz += Factor*
		((
			// Pi_zx
			-ita_par*grad_vjdx_coeff_on_vj_self
			)*edge_normal.x + (
				// Pi_zy
				-ita_par*grad_vjdy_coeff_on_vj_self
				)*edge_normal.y);


	// We are storing a whole matrix when it's just a scalar. !!!

}

__device__ void Augment_Jacobean(
	f64_tens3 * pJ,
	f64 Factor, //h_over (N m_i)
	f64_vec2 edge_normal,
	f64 ita_par, f64 nu, f64_vec3 omega,
	f64 grad_vjdx_coeff_on_vj_self,
	f64 grad_vjdy_coeff_on_vj_self
) {
	if ((VISCMAG == 0) || (omega.dot(omega) < 0.01*0.1*nu*nu))
	{
		// run unmagnetised case

		//Pi_xx = -ita_par*THIRD*(4.0*gradvx.x - 2.0*gradvy.y);
		//Pi_xy = -ita_par*(gradvx.y + gradvy.x);
		//Pi_yx = Pi_xy;
		//Pi_yy = -ita_par*THIRD*(4.0*gradvy.y - 2.0*gradvx.x);
		//Pi_zx = -ita_par*(gradviz.x);
		//Pi_zy = -ita_par*(gradviz.y);		
		//	visc_contrib.x = -over_m_i*(Pi_xx*edge_normal.x + Pi_xy*edge_normal.y);


		// eps_x = vx_k+1 - vx_k - h MAR.x / N
		pJ->xx += Factor*
			((
				// Pi_xx
				-ita_par*THIRD*(4.0*grad_vjdx_coeff_on_vj_self)
				) *edge_normal.x + (
					//Pi_xy
					-ita_par*(grad_vjdy_coeff_on_vj_self)
					)*edge_normal.y);

		pJ->yx += Factor*
			((
				// Pi_yx
				-ita_par*(grad_vjdy_coeff_on_vj_self)
				)*edge_normal.x + (
					// Pi_yy
					-ita_par*THIRD*(-2.0*grad_vjdx_coeff_on_vj_self)
					)*edge_normal.y);

		// The z direction doesn't feature vx --- that is because dvx/dz == 0

		pJ->xy += Factor*
			((
				// Pi_xx
				-ita_par*THIRD*(-2.0*grad_vjdy_coeff_on_vj_self)
				)*edge_normal.x + (
					// Pi_xy
					-ita_par*(grad_vjdx_coeff_on_vj_self)
					)*edge_normal.y);

		pJ->yy += Factor*
			((
				// Pi_yx
				-ita_par*(grad_vjdx_coeff_on_vj_self)
				)*edge_normal.x + (
					// Pi_yy
					-ita_par*THIRD*(4.0*grad_vjdy_coeff_on_vj_self)
					)*edge_normal.y);

		pJ->zz += Factor*
			((
				// Pi_zx
				-ita_par*grad_vjdx_coeff_on_vj_self
				)*edge_normal.x + (
					// Pi_zy
					-ita_par*grad_vjdy_coeff_on_vj_self
					)*edge_normal.y);

		// In this way we develop let's say the J matrix, J for Jacobean
		// Then we could wish to do LU decomp of 4x4 matrix J so that
		// Regressor = J^-1 epsilon[loaded]
		// But we'll do 2 x 3 x 3.
	}
	else {

		f64 omegamod;
		f64_vec3 unit_b, unit_perp, unit_Hall;
		{
			f64 omegasq = omega.dot(omega);
			omegamod = sqrt(omegasq);
			unit_b = omega / omegamod;
			unit_perp = Make3(edge_normal, 0.0) - unit_b*(unit_b.dotxy(edge_normal));
			unit_perp = unit_perp / unit_perp.modulus();
			unit_Hall = unit_b.cross(unit_perp); // Note sign.
		}

		//f64 momflux_b, momflux_perp, momflux_Hall;
		f64 ita_1 = ita_par*(nu*nu / (nu*nu + omegamod*omegamod));
		f64 ita_2 = ita_par*(nu*nu / (nu*nu + 0.25*omegamod*omegamod));
		f64 ita_3 = ita_par*(nu*omegamod / (nu*nu + omegamod*omegamod));
		f64 ita_4 = 0.5*ita_par*(nu*omegamod / (nu*nu + 0.25*omegamod*omegamod));

		f64_vec3mag mag_edge;
		mag_edge.b = unit_b.x*edge_normal.x + unit_b.y*edge_normal.y;
		mag_edge.P = unit_perp.x*edge_normal.x + unit_perp.y*edge_normal.y;
		mag_edge.H = unit_Hall.x*edge_normal.x + unit_Hall.y*edge_normal.y;

		// ==================================================================

		// Our approach is going to be to populate the "Flux Jacobean".
		// Let F_P = PI_Pb edgenormal_b + PI_PP edgenormal_P + PI_PH edgenormal_H

		// *********************
		//  Accumulate dF_b/dvx 
		// *********************

		f64_tens3mag F;
		memset(&F, 0, sizeof(f64_tens3mag));
		f64 bdotPsi = unit_b.x*grad_vjdx_coeff_on_vj_self + unit_b.y*grad_vjdy_coeff_on_vj_self;
		f64 PdotPsi = unit_perp.x*grad_vjdx_coeff_on_vj_self + unit_perp.y*grad_vjdy_coeff_on_vj_self;
		f64 HdotPsi = unit_Hall.x*grad_vjdx_coeff_on_vj_self + unit_Hall.y*grad_vjdy_coeff_on_vj_self;

		f64 d_Pi_by_dvx;
		// how to use union? Can just put in and out of scope.
		// d_Pi_bb_by_dvx =
		d_Pi_by_dvx = -ita_par*THIRD*(4.0*unit_b.x*bdotPsi - 2.0*unit_perp.x*PdotPsi - 2.0*unit_Hall.x*HdotPsi);

		F.bx += d_Pi_by_dvx * mag_edge.b; // Pi_bb

		d_Pi_by_dvx = -ita_2*(unit_b.x*PdotPsi + unit_perp.x*bdotPsi)
			- ita_4*(unit_b.x*HdotPsi + unit_Hall.x*bdotPsi);
		// Pi_bP

		F.bx += d_Pi_by_dvx * mag_edge.P; // Pi_bP
		F.Px += d_Pi_by_dvx * mag_edge.b; // Pi_Pb

		d_Pi_by_dvx = -(ita_2*(unit_b.x*HdotPsi + unit_Hall.x*bdotPsi) + ita_4*(unit_b.x*PdotPsi + unit_perp.x*bdotPsi));
		// Pi_bH

		F.bx += d_Pi_by_dvx * mag_edge.H; // Pi_bH 
		F.Hx += d_Pi_by_dvx * mag_edge.b; // Pi_Hb	

		d_Pi_by_dvx = -0.5*(ita_par + ita_1)*THIRD*(-2.0*unit_b.x*bdotPsi + 4.0*unit_perp.x*PdotPsi - 2.0*unit_Hall.x*HdotPsi)
			- 0.5*(ita_par - ita_1)*THIRD*(-2.0*unit_b.x*bdotPsi - 2.0*unit_perp.x*PdotPsi + 4.0*unit_Hall.x*HdotPsi)
			- ita_3*(unit_perp.x*HdotPsi + unit_Hall.x*PdotPsi);
		// Pi_PP

		F.Px += d_Pi_by_dvx * mag_edge.P;

		d_Pi_by_dvx = -ita_1*(unit_perp.x*HdotPsi + unit_Hall.x*PdotPsi) + ita_3*(unit_perp.x*PdotPsi - unit_Hall.x*HdotPsi);
		// Pi_PH

		F.Px += d_Pi_by_dvx * mag_edge.H;
		F.Hx += d_Pi_by_dvx * mag_edge.P; // Pi_PH

		d_Pi_by_dvx = -0.5*(ita_par + ita_1)*THIRD*(-2.0*unit_b.x*bdotPsi - 2.0*unit_perp.x*PdotPsi + 4.0*unit_Hall.x*HdotPsi)
			- 0.5*(ita_par - ita_1)*THIRD*(-2.0*unit_b.x*bdotPsi + 4.0*unit_perp.x*PdotPsi - 2.0*unit_Hall.x*HdotPsi)
			+ ita_3*(unit_perp.x*HdotPsi + unit_Hall.x*PdotPsi);
		// Pi_HH

		F.Hx += d_Pi_by_dvx*mag_edge.H;

		// That was the x column.
		// Repeat exact same thing again replacing .x ..
		// first get it sensible.
		f64 d_Pi_by_dvy;

		// d_Pi_bb_by_dvy =
		d_Pi_by_dvy = -ita_par*THIRD*(4.0*unit_b.y*bdotPsi - 2.0*unit_perp.y*PdotPsi - 2.0*unit_Hall.y*HdotPsi);

		F.by += d_Pi_by_dvy * mag_edge.b; // Pi_bb

		d_Pi_by_dvy = -ita_2*(unit_b.y*PdotPsi + unit_perp.y*bdotPsi)
			- ita_4*(unit_b.y*HdotPsi + unit_Hall.y*bdotPsi);
		// Pi_bP

		F.by += d_Pi_by_dvy * mag_edge.P; // Pi_bP
		F.Py += d_Pi_by_dvy * mag_edge.b; // Pi_Pb

		d_Pi_by_dvy = -(ita_2*(unit_b.y*HdotPsi + unit_Hall.y*bdotPsi) + ita_4*(unit_b.y*PdotPsi + unit_perp.y*bdotPsi));
		// Pi_bH

		F.by += d_Pi_by_dvy * mag_edge.H; // Pi_bH 
		F.Hy += d_Pi_by_dvy * mag_edge.b; // Pi_Hb	

		d_Pi_by_dvy = -0.5*(ita_par + ita_1)*THIRD*(-2.0*unit_b.y*bdotPsi + 4.0*unit_perp.y*PdotPsi - 2.0*unit_Hall.y*HdotPsi)
			- 0.5*(ita_par - ita_1)*THIRD*(-2.0*unit_b.y*bdotPsi - 2.0*unit_perp.y*PdotPsi + 4.0*unit_Hall.y*HdotPsi)
			- ita_3*(unit_perp.y*HdotPsi + unit_Hall.y*PdotPsi);
		// Pi_PP

		F.Py += d_Pi_by_dvy * mag_edge.P;

		d_Pi_by_dvy = -ita_1*(unit_perp.y*HdotPsi + unit_Hall.y*PdotPsi) + ita_3*(unit_perp.y*PdotPsi - unit_Hall.y*HdotPsi);
		// Pi_PH

		F.Py += d_Pi_by_dvy * mag_edge.H;
		F.Hy += d_Pi_by_dvy * mag_edge.P; // Pi_PH

		d_Pi_by_dvy = -0.5*(ita_par + ita_1)*THIRD*(-2.0*unit_b.y*bdotPsi - 2.0*unit_perp.y*PdotPsi + 4.0*unit_Hall.y*HdotPsi)
			- 0.5*(ita_par - ita_1)*THIRD*(-2.0*unit_b.y*bdotPsi + 4.0*unit_perp.y*PdotPsi - 2.0*unit_Hall.y*HdotPsi)
			+ ita_3*(unit_perp.y*HdotPsi + unit_Hall.y*PdotPsi);
		// Pi_HH

		F.Hy += d_Pi_by_dvy*mag_edge.H;

		f64 d_Pi_by_dvz;

		// d_Pi_bb_by_dvz =
		d_Pi_by_dvz = -ita_par*THIRD*(4.0*unit_b.z*bdotPsi - 2.0*unit_perp.z*PdotPsi - 2.0*unit_Hall.z*HdotPsi);

		F.bz += d_Pi_by_dvz * mag_edge.b; // Pi_bb

		d_Pi_by_dvz = -ita_2*(unit_b.z*PdotPsi + unit_perp.z*bdotPsi)
			- ita_4*(unit_b.z*HdotPsi + unit_Hall.z*bdotPsi);
		// Pi_bP

		F.bz += d_Pi_by_dvz * mag_edge.P; // Pi_bP
		F.Pz += d_Pi_by_dvz * mag_edge.b; // Pi_Pb

		d_Pi_by_dvz = -(ita_2*(unit_b.z*HdotPsi + unit_Hall.z*bdotPsi) + ita_4*(unit_b.z*PdotPsi + unit_perp.z*bdotPsi));
		// Pi_bH

		F.bz += d_Pi_by_dvz * mag_edge.H; // Pi_bH 
		F.Hz += d_Pi_by_dvz * mag_edge.b; // Pi_Hb	

		d_Pi_by_dvz = -0.5*(ita_par + ita_1)*THIRD*(-2.0*unit_b.z*bdotPsi + 4.0*unit_perp.z*PdotPsi - 2.0*unit_Hall.z*HdotPsi)
			- 0.5*(ita_par - ita_1)*THIRD*(-2.0*unit_b.z*bdotPsi - 2.0*unit_perp.z*PdotPsi + 4.0*unit_Hall.z*HdotPsi)
			- ita_3*(unit_perp.z*HdotPsi + unit_Hall.z*PdotPsi);
		// Pi_PP

		F.Pz += d_Pi_by_dvz * mag_edge.P;

		d_Pi_by_dvz = -ita_1*(unit_perp.z*HdotPsi + unit_Hall.z*PdotPsi) + ita_3*(unit_perp.z*PdotPsi - unit_Hall.z*HdotPsi);
		// Pi_PH

		F.Pz += d_Pi_by_dvz * mag_edge.H;
		F.Hz += d_Pi_by_dvz * mag_edge.P; // Pi_PH

		d_Pi_by_dvz = -0.5*(ita_par + ita_1)*THIRD*(-2.0*unit_b.z*bdotPsi - 2.0*unit_perp.z*PdotPsi + 4.0*unit_Hall.z*HdotPsi)
			- 0.5*(ita_par - ita_1)*THIRD*(-2.0*unit_b.z*bdotPsi + 4.0*unit_perp.z*PdotPsi - 2.0*unit_Hall.z*HdotPsi)
			+ ita_3*(unit_perp.z*HdotPsi + unit_Hall.z*PdotPsi);
		// Pi_HH

		F.Hz += d_Pi_by_dvz*mag_edge.H;

		// *************************
		//  Now use it to create J
		// *************************

		pJ->xx += Factor*(unit_b.x*F.bx + unit_perp.x*F.Px + unit_Hall.x*F.Hx);
		pJ->xy += Factor*(unit_b.x*F.by + unit_perp.x*F.Py + unit_Hall.x*F.Hy); // d eps x / d vy
		pJ->xz += Factor*(unit_b.x*F.bz + unit_perp.x*F.Pz + unit_Hall.x*F.Hz);

		pJ->yx += Factor*(unit_b.y*F.bx + unit_perp.y*F.Px + unit_Hall.y*F.Hx);
		pJ->yy += Factor*(unit_b.y*F.by + unit_perp.y*F.Py + unit_Hall.y*F.Hy);
		pJ->yz += Factor*(unit_b.y*F.bz + unit_perp.y*F.Pz + unit_Hall.y*F.Hz);

		pJ->zx += Factor*(unit_b.z*F.bx + unit_perp.z*F.Px + unit_Hall.z*F.Hx);
		pJ->zy += Factor*(unit_b.z*F.by + unit_perp.z*F.Py + unit_Hall.z*F.Hy);
		pJ->zz += Factor*(unit_b.z*F.bz + unit_perp.z*F.Pz + unit_Hall.z*F.Hz);

	}
}
