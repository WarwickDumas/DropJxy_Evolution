
#ifndef cppconst_h
#define cppconst_h
ftyryrt

f64 const sC = 2997924580.0;    // Coulomb to statCoulomb
f64 const kB = 1.60217646e-12;  // erg per eV

f64 const c = 29979245800.0 ;   // speed of light in vacuum (cm/s)
f64 const Z = unity ;           // number of charges in ion species
f64 const e = 4.8e-10  ;        // charge of electrons (statcoul)
f64 const q = 4.8e-10  ;        // ion charge [unsigned]
//f64 const m_e = 9.10953e-28;    // electron mass in grams
f64 const qovermc = q / (m_e_*c);

f64 const m_ion = m_i_;
f64 const m_neutral = m_e_ + m_ion;

f64 const eoverm  = e/m_e_;//5.26920708313162e+17 ;         // e/me, electron statcoul/gram
f64 const qoverM  = q/m_ion;//130686567326725.0     ;        // q/mi, ion statcoul/gram
f64 const moverM  = m_e_/m_ion;//0.000248019417845795    ;      // electron to ion mass ratio
f64 const qoverm  = q/m_e_;
f64 const over_sqrt_m_e = 33132332783135.3;

f64 const eovermc = eoverm/c;
f64 const qoverMc = qoverM/c;
f64 const FOUR_PI_Q_OVER_C = 4.0*PI*q/c;
f64 const FOUR_PI_Q = 4.0*PI*q;
f64 const FOURPI_OVER_C = 4.0*PI/c;
f64 const FOURPIOVERC = 4.0*PI/c;
f64 const FOUR_PI_OVER_C = FOURPIOVERC;
f64 const TWOPIoverc = 2.0*PI / c;

// Having done this, it will certainly cause issues when we are
// messing around with Az.

f64 const NU_EI_FACTOR = 1.0/(3.44e5);
//f64 const NU_II_FACTOR = 1.0/(sqrt(2.0)*2.09e7);
// sqrt in .h file == bad

f64 const nu_eiBarconst = //(4.0/3.0)*sqrt(2.0*PI/m_e)*q*q*q*q;
		// don't know in what units but it IS exactly what we already had - see Formulary
									1.0/(3.44e5);

#endif
