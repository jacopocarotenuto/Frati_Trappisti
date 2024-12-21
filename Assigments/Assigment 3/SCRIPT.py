# Import the required libraries
import taurex
import numpy as np
import matplotlib.pyplot as plt
from taurex.cache import OpacityCache,CIACache
import json
import pickle
# Set general paths
path_to_cia = "./cia/hitran/"
path_to_xsec = "./xsecs/"

# Setup paths
OpacityCache().clear_cache()
OpacityCache().set_opacity_path(path_to_xsec)
CIACache().set_cia_path(path_to_cia)

# We choose HAT-P-18b
HAT_P_18b_PARAMS = {
    "time_of_conjunction": 0.0,               # Time of inferior conjunction (days)
    "orbital_period": 5.508,                  # Orbital period (days)
    "planet_radius": 0.995,                   # Planet radius (Jupiter radii)
    "semi_major_axis": 16.04,                 # Semi-major axis (in units of stellar radii)
    "orbital_inclination": 88.8,              # Orbital inclination (degrees)
    "orbital_eccentricity": 0.084,            # Orbital eccentricity
    "longitude_of_periastron": 120.0,         # Longitude of periastron (degrees)
    "planet_mass": 0.197,                       # Planet Mass (Jupiter Mass)
    "star_temperature": 4803,                 # Kelvin
    "star_radius": 0.765,                      # Stellar Radius (Solar Radii)
    "atm_min_pressure": 1,
    "atm_max_pressure": 1e6,
    "equilibrium_temperature": 852,            # Planet equilibrium temperature in Kelvin
    "stellar_mass": 0.770,                       # Solar Mass
    "stellar_metallicity": 0.10,                 
    
}

# Now  we randomize the abundances of the molecules composing this planet
H2O = np.random.uniform(1e-8,1e-2)
print(f"H2O: {H2O:.3e}")
CH4 = np.random.uniform(1e-8,1e-2)
print(f"CH4: {CH4:.3e}")
CO2 = np.random.uniform(1e-8,1e-2)
print(f"CO2: {CO2:.3e}")
CO = np.random.uniform(1e-8,1e-2)
print(f"CO: {CO:.3e}")

# We add them to the parameter list
HAT_P_18b_PARAMS["H2O"] = H2O
HAT_P_18b_PARAMS["CH4"] = CH4
HAT_P_18b_PARAMS["CO2"] = CO2
HAT_P_18b_PARAMS["CO"] = CO

# Save parameters
with open('HAT_P_18b_assignment3_taskA_parameters.json', 'w') as file:
    json.dump(HAT_P_18b_PARAMS, file, indent=4)


# Import Taurex Relevant Functions
from taurex.temperature import Guillot2010
from taurex.planet import Planet
from taurex.stellar import BlackbodyStar
from taurex.chemistry import TaurexChemistry
from taurex.model import TransmissionModel
from taurex.contributions import AbsorptionContribution
from taurex.contributions import CIAContribution
from taurex.contributions import RayleighContribution
from taurex.chemistry import ConstantGas
from taurex.binning import FluxBinner,SimpleBinner

# Define our model
temperature_profile = Guillot2010(T_irr=HAT_P_18b_PARAMS["equilibrium_temperature"])
planet = Planet(planet_mass = HAT_P_18b_PARAMS["planet_mass"], planet_radius = HAT_P_18b_PARAMS["planet_radius"])
star = BlackbodyStar(temperature=HAT_P_18b_PARAMS["star_temperature"], radius=HAT_P_18b_PARAMS["star_radius"], mass=HAT_P_18b_PARAMS["stellar_mass"],metallicity=HAT_P_18b_PARAMS["stellar_metallicity"])
chemistry = TaurexChemistry()
chemistry.addGas(ConstantGas('H2O',mix_ratio=np.power(10, -4.47)))
# chemistry.addGas(ConstantGas('CH4',mix_ratio=CH4))
chemistry.addGas(ConstantGas('CO2',mix_ratio=np.power(10, -4.86)))
# chemistry.addGas(ConstantGas('CO',mix_ratio=CO))
chemistry.addGas(ConstantGas("Na", mix_ratio=np.power(10, -6.99)))

# "H2O": np.power(10, -4.47),  # Water vapor abundance
# "CO2": np.power(10, -4.86),  # Carbon dioxide abundance
# "Na": np.power(10, -6.99),   # Sodium abundance

model = TransmissionModel(planet=planet,
                       temperature_profile=temperature_profile,
                       chemistry=chemistry,
                       star=star,
                        atm_min_pressure=1e-0,
                       atm_max_pressure=1e6,
                       nlayers=30)

model.add_contribution(AbsorptionContribution())
model.add_contribution(CIAContribution(cia_pairs=['H2-H2','H2-He']))
model.add_contribution(RayleighContribution())

model.build()
results = model.model()

binned_fig = plt.figure()
#Make a logarithmic grid
wngrid = np.sort(10000/np.logspace(-0.22,0.44,1000))
bn = FluxBinner(wngrid=wngrid)
bin_wn, bin_rprs,_,_  = bn.bin_model(model.model(wngrid=wngrid))
plt.plot(10000/bin_wn,bin_rprs)
plt.xscale('log')
plt.title("Spettro di Hat-18-b")
plt.xlabel('Lunghezza d\'onda [$\\mu$m]')
plt.ylabel(r'$(R_p / R_s)^2$')
plt.savefig('HAT_P_18b_assignment3_taskA_spectrum.png')

# Save the matrix
matrix = np.array([10000/bin_wn, bin_rprs, np.ones(bin_rprs.shape)*20e-6])
np.savetxt('HAT18b_assignment3_taskA_spectrum.dat', matrix.T)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

df = pd.read_csv('/home/ubuntu/comp_astro_24/Assigments/Assigment 3/HAT18P_139points.csv')

df = df[['CENTRALWAVELNG', 'BANDWIDTH', 'PL_TRANDEP', 'PL_TRANDEPERR1']]
df["PL_TRANDEP"] = df["PL_TRANDEP"] / 100
df['PL_TRANDEPERR1'] = df['PL_TRANDEPERR1']**2
df.to_csv('/home/ubuntu/comp_astro_24/Assigments/Assigment 3/HATP18_spectrum_assignment3_taskB.csv')
np.savetxt('HATP18_spectrum_assignment3_taskB.dat', df[['CENTRALWAVELNG', 'PL_TRANDEP', 'PL_TRANDEPERR1']].to_numpy())
# Estrai i dati per il plot
wavelength = df['CENTRALWAVELNG']  # Lunghezza d'onda in micron
wavelength_error = df['BANDWIDTH'] # Errore su lunghezza d'onda
transit_depth = df['PL_TRANDEP']   # Profondità di transito (Rp/Rs)^2
error = df['PL_TRANDEPERR1']       # Errore sulla profondità di transito

# Crea una figura più larga
plt.figure()  # Aumenta il valore della larghezza (primo parametro)

# Crea il grafico dello spettro
#plt.errorbar(wavelength, transit_depth, xerr=wavelength_error, yerr=error, fmt='o', markersize=4, capsize=2, label='Dati osservati')
plt.plot(wavelength,transit_depth, label="Observation")
# Configura il grafico
plt.xlabel('Lunghezza d\'onda [$\\mu$m]')
plt.ylabel('Profondità di transito $(R_p / R_s)^2$')
plt.title('Spettro di trasmissione di HAT-P-18')
plt.xscale('log')
plt.plot(10000/bin_wn,bin_rprs, label="Simulated Data")
plt.grid(True)
plt.legend()

# Salva il grafico in un file PNG
plt.savefig('/home/ubuntu/comp_astro_24/Assigments/Assigment 3/HATP18_spectrum_assignment3.png', dpi=300)

# Update the abundances

REAL_HAT_P_18b_PARAMS = HAT_P_18b_PARAMS
REAL_HAT_P_18b_PARAMS.update({
    "H2O": np.power(10, -4.47),  # Water vapor abundance
    "CO2": np.power(10, -4.86),  # Carbon dioxide abundance
    "Na": np.power(10, -6.99),   # Sodium abundance
})
# Save to file
with open('HATP18b_NearInfraredTransmission_report_assignment3_taskb.json', 'w') as file:
    json.dump(REAL_HAT_P_18b_PARAMS, file, indent=4)


from taurex.optimizer.nestle import NestleOptimizer
from taurex.data.spectrum.observed import ObservedSpectrum
opt = NestleOptimizer(num_live_points=200)

opt.set_model(model)
obs = ObservedSpectrum("HAT18b_assignment3_taskA_spectrum.dat")
opt.set_observed(obs)
obin = obs.create_binner()

# Define parameters to fit
opt.enable_fit('planet_radius')
opt.set_boundary('planet_radius',[0.8,1.2])

opt.enable_fit("T_irr")
opt.set_boundary('T_irr',[750,950])

opt.enable_fit("H2O")
opt.set_boundary('H2O',[1e-4,1e-5])

opt.enable_fit("CO2")
opt.set_boundary('CO2',[1e-4,1e-5])

opt.enable_fit("Na")
opt.set_boundary('Na',[1e-6,1e-8])
# Perform the fit
DO_FIT = True

if DO_FIT:
    solution = opt.fit()
    taurex.log.disableLogging()
    with open('Retrieval_Simulated_Solution.pkl', 'wb') as file:  # Use 'wb' for write binary mode
        for data in opt.get_solution():
            pickle.dump(data, file)
else:
    with open('Retrieval_Simulated_Solution.pkl', 'rb') as file:  # Use 'wb' for write binary mode
        data = pickle.load(file)
    
    #####################
    
    
# Create new model

# Define our model
temperature_profile = Guillot2010(T_irr=REAL_HAT_P_18b_PARAMS["equilibrium_temperature"])
planet = Planet(planet_mass = REAL_HAT_P_18b_PARAMS["planet_mass"], planet_radius = REAL_HAT_P_18b_PARAMS["planet_radius"])
star = BlackbodyStar(temperature=REAL_HAT_P_18b_PARAMS["star_temperature"], radius=REAL_HAT_P_18b_PARAMS["star_radius"])
chemistry = TaurexChemistry()
chemistry.addGas(ConstantGas('H2O',mix_ratio=REAL_HAT_P_18b_PARAMS["H2O"]))
chemistry.addGas(ConstantGas('Na',mix_ratio=REAL_HAT_P_18b_PARAMS["Na"]))
chemistry.addGas(ConstantGas('CO2',mix_ratio=REAL_HAT_P_18b_PARAMS["CO2"]))

model = TransmissionModel(planet=planet,
                       temperature_profile=temperature_profile,
                       chemistry=chemistry,
                       star=star,
                       atm_min_pressure=1e-0,
                       atm_max_pressure=1e6,
                       nlayers=30)

model.add_contribution(AbsorptionContribution())
model.add_contribution(CIAContribution(cia_pairs=['H2-H2','H2-He']))
model.add_contribution(RayleighContribution())

model.build()
results = model.model()

from taurex.optimizer.nestle import NestleOptimizer
from taurex.data.spectrum.observed import ObservedSpectrum
opt = NestleOptimizer(num_live_points=50)

opt.set_model(model)
obs = ObservedSpectrum("HATP18_spectrum_assignment3_taskB.dat")
opt.set_observed(obs)
obin = obs.create_binner()

# Define parameters to fit
opt.enable_fit('planet_radius')
opt.set_boundary('planet_radius',[0.8,1.2])

opt.enable_fit("T_irr")
opt.set_boundary('T_irr',[750,950])

opt.enable_fit("H2O")
opt.set_boundary('H2O',[1e-4,1e-5])

opt.enable_fit("CO2")
opt.set_boundary('CO2',[1e-4,1e-5])

opt.enable_fit("Na")
opt.set_boundary('Na',[1e-6,1e-8])

        
# Perform the fit
DO_FIT = True

if DO_FIT:
    solution = opt.fit()
    taurex.log.disableLogging()
    with open('Retrieval_Real_Solution.pkl', 'wb') as file:  # Use 'wb' for write binary mode
        for data in opt.get_solution():
            pickle.dump(data, file)
else:
    with open('Retrieval_Real_Solution.pkl', 'rb') as file:  # Use 'wb' for write binary mode
        data = pickle.load(file)