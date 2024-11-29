import batman
import numpy as np
import matplotlib.pyplot as plt
#HAT-P-18b hartman et al 2011
params = batman.TransitParams()       #object to store transit parameters
params.t0 = 0.                        #time of inferior conjunction
params.per = 5.508                    #orbital period
params.rp = 0.1365                    #planet radius (in units of stellar radii)
params.a = 16.04                      #semi-major axis (in units of stellar radii)
params.inc = 88.8                     #orbital inclination (in degrees)
params.ecc = 0.084                    #eccentricity
params.w = 120.0                      #longitude of periastron (in degrees)
params.limb_dark = "quadratic"        #limb darkening model
params.u = [0.506, 0.128]             #limb darkening coefficients [u1, u2, u3, u4]

t = np.linspace(-0.10,0.10,100)              #times at which to calculate light curve
m = batman.TransitModel(params, t)

flux = m.light_curve(params)
radii = np.array([0.06825, 0.1365, 0.273])
fig = plt.figure(figsize=(10, 6))
for r in radii:
        params.rp = r                           #updates planet radius
        new_flux = m.light_curve(params)        #recalculates light curve
        plt.plot(t, new_flux, label=f'Light curve r='+str(r))

plt.xlabel('Time from transit center (Days)')
plt.ylabel('Relative Flux')
plt.title('Exoplanet Transit Light Curve')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('assignment2_taskBC.png')
