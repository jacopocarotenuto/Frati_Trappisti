from taurex.optimizer.nestle import NestleOptimizer
from taurex.data.spectrum.observed import ObservedSpectrum
from daneel.atmosphere.model_creation import create_forward_model


def perform_atmospheric_retrieval(param):
    fit_parameters = param.get("fit_parameters")
    assert isinstance(fit_parameters, dict), "Fitting Parameters incorrectly read from file. Must be a dictionary"

    observation_data_file = param.get("ObservationDataFile")
    
    opt = NestleOptimizer(num_live_points=200)
    model = create_forward_model(param, save_to_file=False)
    opt.set_model(model)
    obs = ObservedSpectrum(observation_data_file)
    opt.set_observed(obs)
    obin = obs.create_binner()
    
    # Define Fitting Parameters
    for fit_par in fit_parameters.keys():
        opt.enable_fit(fit_par)
        opt.set_boundary(fit_par, fit_parameters[fit_par])

    # Fit
    solution = opt.fit()
    for solution,optimized_map,optimized_value,values in opt.get_solution():
        sol = values[1][1]
    for key in sol.keys():
        print(f"Fitted {key}: ",sol[key]["value"])