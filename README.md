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

### Detection Calculations

Daneel can detect planets in lightcurve data with different methods. The methods available are: Support Vector Machine (svm), Neural Network (nn), Convolutional Neural Network (cnn).
The parameters of the detection method (such as epochs, training files, learning rate...) are to be specified in a .yaml file.

The detection is invoked like this:
`daneel -i PATH_TO_PARAMETERS.yaml -d DETECTION_METHOD`

Also, daneel can generate new lightflux images with the command
`daneel -i PATH_TO_PARAMETERS.yamls --dream`




## Credits

Repository for the "Computational Astrophysics" course at Unipd in academic year 2024/2025.

Group Members: 
- Jacopo Carotenuto
- Andrea Semenzato
- Lorenzo Fiaccadori
- Linda Giuliani
