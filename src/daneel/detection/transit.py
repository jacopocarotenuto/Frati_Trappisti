import batman
import numpy as np
import matplotlib.pyplot as plt

params = batman.TransitParams()       #object to store transit parameters
params.t0 = 0.                        #time of inferior conjunction
params.per = 1.09141937               #orbital period
params.rp = 0.195                     #planet radius (in units of stellar radii)
params.a = 5.04                       #semi-major axis (in units of stellar radii)
params.inc = 83.37                    #orbital inclination (in degrees)
params.ecc = 0.0447                   #eccentricity
params.w = 272.7                      #longitude of periastron (in degrees)
params.limb_dark = "quadratic"        #limb darkening model
params.u = [0.574, 0.197]             #limb darkening coefficients [u1, u2, u3, u4]

t = np.linspace(-0.25,0.25,100)              #times at which to calculate light curve
m = batman.TransitModel(params, t)

flux = m.light_curve(params)
radii = np.linspace(0.09, 0.11, 20)
for r in radii:
        params.rp = r                           #updates planet radius
        new_flux = m.light_curve(params)        #recalculates light curve
fig = plt.figure()
plt.plot(t, flux)
plt.show()
fig.savefig('WASP12-b_assignment1_taskF.png')
