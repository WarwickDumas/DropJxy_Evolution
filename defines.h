
#ifndef constant_h
#define constant_h

#define f64 double


// decided to stick with this
// but move math.h functions out of header file
// into "constant.cpp"


f64 const zero = 0.0;
f64 const unity = 1.0;
f64 const two = 2.0;
f64 const twothirds = 2.0/3.0;
f64 const THIRTEENSIX = 13.6;
f64 const TWOTHIRDS = 2.0/3.0;
f64 const FOURTHIRDS = 4.0/3.0;
f64 const TWONINTHS = 2.0/9.0;
f64 const NINTH = 1.0/9.0;
f64 const THIRD = 1.0/3.0;

f64 const PI = 3.14159265358979323846;
f64 const PI2 = 2.0*PI;     
f64 const PI2inv = unity/PI2; 

f64 const sC = 2997924580.0;    // Coulomb to statCoulomb
f64 const kB = 1.60217646e-12;  // erg per eV

f64 const c = 29979245800.0 ;   // speed of light in vacuum (cm/s)
f64 const Z = unity ;           // number of charges in ion species
f64 const e = 4.8e-10  ;        // charge of electrons (statcoul)
f64 const q = 4.8e-10  ;        // ion charge [unsigned]
f64 const m_e = 9.10953e-28;    // electron mass in grams
f64 const m = m_e;

// NOTE CHANGE : 3 x mass :

f64 const m_i = 3.0*3.67291e-24;    // deuteron mass in grams

f64 const M = m_i;
f64 const m_ion = m_i;
f64 const M_ion = m_ion;
f64 const m_neutral = m_e + m_ion;
f64 const M_neutral = m_neutral;
f64 const m_n = m_neutral;
f64 const m_neut = m_n;

f64 const Minv = unity/m_i;
f64 const two_pi_over_c = PI*two/c;

//f64 const OVER_SQRT_2PI = 1.0/sqrt(two*PI);
// no sqrt in .h

f64 const PIOVER32 = PI/32.0;

f64 const eoverm  = e/m;//5.26920708313162e+17 ;         // e/me, electron statcoul/gram
f64 const qoverM  = q/M;//130686567326725.0     ;        // q/mi, ion statcoul/gram
f64 const moverM  = m/M;//0.000248019417845795    ;      // electron to ion mass ratio
f64 const qoverm  = q/m;
f64 const qqoverm = q*q/m;
f64 const overM = unity/M;
f64 const qovermc = qoverm/c;
f64 const qoverMc = qoverM/c;
f64 const TWOTHIRDSqsq = 2.0*q*q/3.0;
f64 const FOUR_PI_Q_OVER_C = 4.0*PI*q/c;
f64 const FOUR_PI_Q = 4.0*PI*q;
f64 const FOURPI_OVER_C = 4.0*PI/c;
f64 const FOURPIOVERC = 4.0*PI/c;
f64 const FOUR_PI_OVER_C = FOURPIOVERC;

f64 const NEUTRAL_KAPPA_FACTOR = 10.0;


f64 const NU_EI_FACTOR = 1.0/(3.44e5);
//f64 const NU_II_FACTOR = 1.0/(sqrt(2.0)*2.09e7);
// sqrt in .h file == bad

f64 const nu_eiBarconst = //(4.0/3.0)*sqrt(2.0*PI/m_e)*q*q*q*q;
		// don't know in what units but it IS exactly what we already had - see Formulary
									1.0/(3.44e5);

f64 const  ION_KAPPA_FACTOR = 20.0/9.0; // 10/3 times 2/3
f64 const  ELECTRON_KAPPA_FACTOR = 5.0/3.0; // 5/2 times 2/3

f64 const ALPHA_ION = 0.96; // this factor appears in the viscosity coefficient
f64 const ALPHA_e   = 0.73;
f64 const ALPHA_ELECTRON = 0.73;

f64 const cross_T_vals[10] = {0.1,0.501187,1.0,1.99526,3.16228,5.01187,7.94328,12.5893,19.9526,31.6228};

// momentum-transfer cross section data from http://www-cfadc.phy.ornl.gov/elastic/ddp/tel-DP.html
f64 const cross_s_vals_momtrans_ni[10] = {
	1.210e-14,1.020e-14,9.784e-15,9.076e-15,8.589e-15,8.115e-15,7.653e-15,7.207e-15,6.776e-15,6.351e-15};
// distinguishable particles:
	//4.408e-15,2.213e-15,1.666e-15,7.625e-16,4.685e-16,2.961e-16,1.878e-16,1.192e-16,7.442e-17,4.083e-17};

// viscosity cross section data from  http://www-cfadc.phy.ornl.gov/elastic/ddp/tel-DP.html
f64 const cross_s_vals_viscosity_ni[10] = {
	4.904e-15,3.023e-15,2.673e-15,1.891e-15,1.203e-15,7.582e-16,4.891e-16,3.185e-16,2.030e-16,1.223e-16};
// viscosity cross section data from http://www-cfadc.phy.ornl.gov/elastic/dd0/tel.html
f64 const cross_s_vals_viscosity_nn[10] = {
	1.753e-15,1.179e-15,9.030e-16,7.650e-16,6.316e-16,4.278e-16,2.685e-16,1.641e-16,9.609e-17,5.550e-17};

#endif
