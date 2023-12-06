
#ifndef constant_h
#define constant_h

#define f64 double


// decided to stick with this
// but move math.h functions out of header file
// into "constant.cpp"


#define PI 3.1415926535897932;

f64 const sC_ = 2997924580.0;    // Coulomb to statCoulomb
f64 const kB_ = 1.60217646e-12;  // erg per eV

f64 const c_ = 29979245800.0 ;   // speed of light in vacuum (cm/s)
f64 const Z_ = 1.0 ;           // number of charges in ion species
f64 const e_ = 4.8e-10  ;        // charge of electrons (statcoul)
f64 const q_ = 4.8e-10  ;        // ion charge [unsigned]
f64 const m_e_ = 9.10953e-28;    // electron mass in grams

// NOTE CHANGE : 3 x mass :

f64 const m_ion_ = 3.0*3.67291e-24;    // deuteron mass in grams
f64 const m_n_ = m_e + m_ion;

f64 const eoverm_ = e_/m_e_;//5.26920708313162e+17 ;         // e/me, electron statcoul/gram
f64 const qoverM_ = q_/m_ion_;//130686567326725.0     ;        // q/mi, ion statcoul/gram
f64 const moverM_  = m_e_/m_ion_;//0.000248019417845795    ;      // electron to ion mass ratio
f64 const eovermc_ = eoverm_/c_;
f64 const qoverMc_ = qoverM_/c_;
f64 const TWOTHIRDSqsq_ = 2.0*q_*q_/3.0;
f64 const FOURPI_Q_OVER_C_ = 4.0*PI*q_/c_;
f64 const FOURPI_Q_ = 4.0*PI*q_;
f64 const FOURPI_OVER_C_ = 4.0*PI/c_;
f64 const FOURPI_ = 4.0*PI;

f64 const NU_EI_FACTOR_ = 1.0/(3.44e5); // NOTE: this goes with T in eV

//f64 const NU_II_FACTOR = 1.0/(sqrt(2.0)*2.09e7);
// sqrt in .h file == bad

f64 const nu_eiBarconst = //(4.0/3.0)*sqrt(2.0*PI/m_e)*q*q*q*q;
		// don't know in what units but it IS exactly what we already had - see Formulary
									1.0/(3.44e5);

f64 const NEUTRAL_KAPPA_FACTOR_ = 10.0; // debatable
f64 const  ION_KAPPA_FACTOR_ = 20.0/9.0; // 10/3 times 2/3
f64 const  ELECTRON_KAPPA_FACTOR_ = 5.0/3.0; // 5/2 times 2/3

f64 const ALPHA_ION_ = 0.96; // this factor appears in the viscosity coefficient
f64 const ALPHA_e_   = 0.73;

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
