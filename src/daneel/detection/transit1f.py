import batman
import numpy as np
import matplotlib.pyplot as plt

# HAT-P-18b Hartman et al. 2011
params = batman.TransitParams()       # Object to store transit parameters
params.t0 = 0.                        # Time of inferior conjunction
params.per = 5.508023                 # Orbital period (days)
params.rp = 0.13650                   # Planet radius (in units of stellar radii)
params.a = 16.04                      # Semi-major axis (in units of stellar radii)
params.inc = 88.8                     # Orbital inclination (degrees)
params.ecc = 0.084                    # Eccentricity
params.w = 120                        # Longitude of periastron (degrees)
params.limb_dark = "quadratic"        # Limb darkening model
params.u = [0.506, 0.128]             # Limb darkening coefficients [u1, u2]

t = np.linspace(-0.25, 0.25, 100)     # Times at which to calculate light curve
m = batman.TransitModel(params, t)

flux = m.light_curve(params)

fig = plt.figure(figsize=(8, 6))
plt.plot(t, flux, label='Light Curve')
plt.xlabel('Time from transit center (Days)')
plt.ylabel('Relative Flux')
plt.title('HAT-P-18b Transit Light Curve')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('HAT-P-18b_assignment1_taskF.png')

