# Project "ICOS data and modeling at Col du Lautaret"

As carbon dioxide (CO<sub>2</sub>) concentration into the amosphere is increasing, causing global warming, it becomes essential to mitigate the negative impacts of this phenomenon. Acting as natural carbon sinks, grasslands sequester up to 300 kg of carbon per hectare per year. However, their role and long-term sustainability could be threatened by climate change, especially in the Alps where the global warming is occurring faster than elsewhere. To understand the exchanges of CO<sub>2</sub> and H<sub>2</sub>O between the ecosystem and the atmosphere, the IGE (Institute of Environmental Geosciences) supervises an ICOS-associated flux tower at Col du Lautaret (France). Installed on an alpine grassland characterized by varied plant species and topography, the main problematic is: **to what extent are the datatsets representative of the different elements and fluxes in the landscape ?** To answer this question, flux footprint models can be used to describe the spatial extent and position of the surface area that is contributing to flux measurements.

---

## Objectives

The goals of this work are:  
- **To characterize the fluxes origin depending on different periods**  
- **To determine the contribution of certain areas (road, wetland, ...) to the measured CO<sub>2</sub>**  
- **To facilitate the comparison with other models, in particular hydrological**

---

## Usage 

Before running the notebook, the mentioned computing environment must be installed (**`environment.yml`**).  

### 1. Data importation

The input data is half-hourly meteorogical and fluxes variables for the Lautaret station, following ICOS standards. As some measurements are missing for footprint calculation on the [ICOS data portal] (https://data.icos-cp.eu/portal/), raw csv files and metadata are available on _shared storage_. 

### 2. Data preparation
### 3. Isolate certain periods
### 4. Calculate footprints with contours from 10 to 90%
### 5. Plot the footprint
### 6. Reproject the footprint onto another grid
### 7. Save in daily netCDF files for further work

---

## Material
...

---

## Contributors 
...

---

## Licenses and accessibility

The code and this repository are licensed under the MIT License. See the [MIT_LICENSE](MIT_LICENSE) file for details.
The input (csv) and output data (netCDF and figures) are licensed under a [Creative Commons Attribution 4.0 International license] (https://creativecommons.org/licenses/by/4.0/legalcode). 
This project uses the footprint model by Kljun et al. (2015), which is licensed under the ISC License. See [ISC_MODEL_LICENSE](ISC_MODEL_LICENSE) for details.  

The data and scripts are shared under open licenses that support the principles of open science. This ensures that others can freely access, use and build upon the material for any purpose, promoting reproducibility and collaborative research while still giving the original authors appropriate credit.  

**Conservation strategies:** For now, output datasets and figures will be used as part of the internship deliverables and are not meant to be stored on a public repository like Zenodo. In the long term, the code could be adapted for more general use and applied for other ICOS stations. Then it will be stored on software heritage or zenodo.

---

## Acknowledgments

Kljun et al. (2015) for the footprint model