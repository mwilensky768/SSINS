# SSINS Readme

SSINS is a python package intended for radio frequency interference flagging in radio astronomy datasets. It stands for Sky-Subtracted Incoherent Noise Spectra. If you use this software for published work, please cite this paper that describes the SSINS method: (doi:10.1088/1538-3873/ab3cad) (arxiv version: https://arxiv.org/abs/1906.01093).

## Dependencies

**python3** is now required. Support for python2 has been dropped.<br/>
**pyuvdata 2.4.1 or newer**.<br/>
**pyuvdata has its own dependencies!**<br/>
See https://github.com/RadioAstronomySoftwareGroup/pyuvdata.  
**pyyaml 5.3.1 or newer** also for reading and writing SSINS outputs.  

### Optional Dependencies

**matplotlib** will be necessary if the user wants to use the Catalog_Plot and plot_lib libraries.  
**pytest** if you want to run unit tests


## Installation

Once pyuvdata and all dependencies of that are installed, simply clone this repo and run `pip install .` from the top level `SSINS` directory.

## Documentation

Docs are available at: https://ssins.readthedocs.io/en/latest/  

There are also tutorials available here: https://ssins.readthedocs.io/en/latest/tutorial.html?highlight=tutorial  

## Contact

Please feel free to email questions to michael.wilensky@manchester.ac.uk
