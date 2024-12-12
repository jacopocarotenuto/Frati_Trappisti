import batman
import numpy as np
import matplotlib.pyplot as plt
#HAT-P-18b hartman et al 2011

params = batman.TransitParams()       #object to store transit parameters
params.t0 = 0.                        #time of inferior conjunction
params.per = 5.508023               #orbital period
params.rp = 0.13650                     #planet radius (in units of stellar radii)
params.a = 16.04                       #semi-major axis (in units of stellar radii)
params.inc = 88.8                    #orbital inclination (in degrees)
params.ecc = 0.084                   #eccentricity
params.w = 120                      #longitude of periastron (in degrees)
params.limb_dark = "quadratic"        #limb darkening model
params.u = [0.506, 0.128]             #limb darkening coefficients [u1, u2]

t = np.linspace(-0.25,0.25,100)              #times at which to calculate ligh>
m = batman.TransitModel(params, t)


flux = m.light_curve(params)
#radii = np.linspace(0.09, 0.11, 20)
#for r in radii:
#        params.rp = r                           #updates planet radius
#        new_flux = m.light_curve(params)        #recalculates light curve
fig = plt.figure()
plt.plot(t, flux, label='Transit Light Curve')
plt.xlabel('Time from transit center (Days)')
plt.ylabel('Relative Flux')
plt.title('HAT-P-18b Transit Light Curve')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('HAT-P-18b_assignment1_taskF_try.png')

	
