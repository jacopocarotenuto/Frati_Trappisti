import batman
import numpy as np
import matplotlib.pyplot as plt
#HAT-P-18b hartman et al 2011
def calculate_transit(parameters):
	params = batman.TransitParams()       #object to store transit parameters
	params.t0 = 0.                        #time of inferior conjunction
	params.per = parameters.get("p")                    #orbital period
	params.rp = parameters.get("r")                    #planet radius (in units of stellar radii)
	params.a = parameters.get("a")                      #semi-major axis (in units of stellar radii)
	params.inc = parameters.get("i")                     #orbital inclination (in degrees)
	params.ecc = parameters.get("e")                    #eccentricity
	params.w = parameters.get("w")                      #longitude of periastron (in degrees)
	params.limb_dark = parameters.get("l")        #limb darkening model
	params.u = parameters.get("u")             #limb darkening coefficients [u1, u2, u3, u4]

	t = np.linspace(-0.10,0.10,100)              #times at which to calculate light curve
	m = batman.TransitModel(params, t)

	flux = m.light_curve(params)
	radii = np.array([params.rp/2, params.rp, params.rp*2])
	fig = plt.figure(figsize=(10, 6))
	for r in radii:
			params.rp = r                           #updates planet radius
			new_flux = m.light_curve(params)        #recalculates light curve
			plt.plot(t, new_flux, label=f'Light curve r='+str(r))

	plt.xlabel('Time from transit center (Days)')
	plt.ylabel('Relative Flux')
	plt.title(parameters.get("name") + ' Transit Light Curve')
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	plt.show()
	fig.savefig('assignment2_taskBC.png')