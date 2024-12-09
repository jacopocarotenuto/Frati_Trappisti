import taurex
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
import numpy as np
from taurex.cache import OpacityCache,CIACache

def create_forward_model(param, save_to_file=True):
    # Set general paths
    path_to_cia = param.get("path_to_cia")
    path_to_xsec = param.get("path_to_xsec")

    # Setup paths
    OpacityCache().clear_cache()
    OpacityCache().set_opacity_path(path_to_xsec)
    CIACache().set_cia_path(path_to_cia)
    
    
    
    # Get parameters as dictionary
    params = param.get("ForwardModelParameters")
    print(params)
    assert isinstance(params, dict), "Error in retrieving model parameters from file"
    
    ### MODEL DEFINITION ##
    temperature_profile = Guillot2010(T_irr=params["equilibrium_temperature"])
    print(f"Using Guillot Temperature Model with Equilibrium Temperature of {params['equilibrium_temperature']} K")
    
    planet = Planet(planet_mass = params['planet_mass'],
                    planet_radius = params['planet_radius'])
    print(f"Planet Mass: {params['planet_mass']} Solar Masses\nPlanet Radius: {params['planet_radius']} Jovian Radii")
    
    star = BlackbodyStar(temperature=params["star_temperature"],
                         radius=params["star_radius"], mass=params["stellar_mass"],
                         metallicity=params["stellar_metallicity"])
    print(f"Star Temperature: {params['star_temperature']} K\nStar Radius: {params['star_radius']} Solar Radii\nStar Mass: {params['stellar_mass']} Solar Masses\nStar Metallicity: {params['stellar_metallicity']}")
    
    
    
    # Assuming that params["gas"] is a dictionary
    assert isinstance(params["gas"], dict), "Gases not correctly set in parameter files"
    assert len(params["gas"]) > 0, "No gases specified in config file"
    chemistry = TaurexChemistry()
    
    for gas in params["gas"].keys():
        chemistry.addGas(ConstantGas(gas, mix_ratio=params["gas"][gas]))
        print(f"Added {gas} with density {params['gas'][gas]:2e}")
    
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
    
    print("Added Absorption Contribution, CIA Contribution (H2-H2, H2-He) and Rayleight Contribution")
    
    model.build()
    
    wngrid = np.sort(10000/np.logspace(-0.22,0.44,1000))
    bn = FluxBinner(wngrid=wngrid)
    bin_wn, bin_rprs,_,_  = bn.bin_model(model.model(wngrid=wngrid))
    matrix = np.array([10000/bin_wn, bin_rprs, np.sqrt(bin_rprs)])
    
    if save_to_file:
        try:
            np.savetxt(param.get("ForwardModelFilePath"), matrix.T)
            print("Forward Model Saved Successfully")
        except:
            print("Error in saving forward model data")
    
    return model