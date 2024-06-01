# StoBe-Data-Loader
This repository contains routines that will extract data from the StoBe output files associated with a transition potential calculation in preparation for the clustering algorithm.

The routines have been parallelized wherever possible. This is still very much a prototype and is not as fleshed out as the IGOR implementation. 

## Usage 
1. Create either a conda or a venv environment called stobe_loader
2. Ensure you have the python packages listed below installed in your environment. They can all be readily installed using pip.
  - pandas
  - streamlit
  - natsort
  - py3Dmol
  - stmol
  - matplotlib
  - numpy
  - seaborn
  - scipy
  - joblib
3.Run the application by navigating to the location of the stobe_loader.py file via
```
streamlit run stobe_loader.py
```
4. The loading function requires you to provide a path to the directory containing the folders that have the StoBe output files from the transition potential calculations.
   For each excitation center, it will look for 3 different files
   
   a. A ground State calculation denoted by a 'gnd.out' file naming scheme
   
   b. An excited State calculation denoted by a 'exc.out' file naming scheme
   
   c. A transition potential calculation denoted by a 'tp.out' file naming scheme
   
   **IMPORTANT: Make sure that the files follow this naming scheme or you'll have to modify the loading function to accommodate whatever scheme you used.**

5. Set the desired values for the energy dependent broadening scheme using the Width 1, Width 2 and Maximum DFT Energy input boxes. The default values work very well for the Carbon edge NEXAFS.
6. Set the name of your molecule.
7. For the Clustering Algorithm, set an oscillator strength threshold using the given input box
8. OVP Threshold not implemented yet.
