
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

	f64 theta = atan2(y, x); // might want to check what it returns.

	if ((r > TOOTH_RADIUS_START) && (r < TOOTH_RADIUS_OUTER))
	{
		if (r < TOOTH_RADIUS_PEAK1)
		{
			maxh = (r - TOOTH_RADIUS_START)*TOOTH_HEIGHT / (TOOTH_RADIUS_PEAK1 - TOOTH_RADIUS_START);
			// That's for the plane facing up to the point radially.
		} else {
			if (r > TOOTH_RADIUS_PEAK2) {
				maxh = (TOOTH_RADIUS_OUTER - r)*TOOTH_HEIGHT / (TOOTH_RADIUS_OUTER - TOOTH_RADIUS_PEAK2);
			} else {
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
				} else {
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
				} else {
					maxh2 = VALLEY_HEIGHT +
						(theta - VALLEY_ANGLE2)*(TOOTH_HEIGHT - VALLEY_HEIGHT) /
						(PLATEAU_ANGLE_22 - VALLEY_ANGLE2);
				};
			};
		};
		height = min(maxh, maxh2); // cut one plane with the other.

		// Uplift:

		retval *= INSULATOR_HEIGHT / (INSULATOR_HEIGHT - height);
		// 2.8/(2.8-0.63) = 1.29		
	};