# Computational Astrophysics 2024 - Group 16

## How to Use

The "Daneel" package has different commands that execute common tasks in computational astrophysics.

### Transit Calculations

Daneel can calculate and plot the transit of a planet easily, provided with the parameters of the planet.
Usage:
`daneel -i PATH_TO_PARAMETER_FILE.yaml -t`
Where the `PATH_TO_PARAMETER_FILE.yaml` is a .yaml file containing all the required parameters for the transit calculation, that are:
- Period [days]
- Planet Radius [solar radii]
- Semi-Major Axis [solar radii]
- Inclination [degree]
- Eccentricity
- Longitude of the periastron [degree]
- Limb darkening model (eg: "quadratic")
- Limb darkening coefficients (array)





## Credits

Repository for the "Computational Astrophysics" course at Unipd in academic year 2024/2025.

Group Members: 
- Jacopo Carotenuto
- Andrea Semenzato
- Lorenzo Fiaccadori
- Linda Giuliani
