# SSINS Readme

SSINS is a python package intended for radio frequency interference flagging in radio astronomy datasets. It stands for Sky-Subtracted Incoherent Noise Spectra. If you use this software for published work, please cite this paper that describes the SSINS method: (doi:10.1088/1538-3873/ab3cad) (arxiv version: https://arxiv.org/abs/1906.01093).

## Dependencies

**python3** is now required. Support for python2 has been dropped.
**pyuvdata 2.0.2 or better**.  
**pyuvdata has its own dependencies!** and some of those listed below are shared.  
See https://github.com/RadioAstronomySoftwareGroup/pyuvdata.  

**numpy** (see pyuvdata dependencies).  
**scipy** (see pyuvdata dependencies).  
**six** (see pyuvdata dependencies).  
**h5py** for reading and writing SSINS outputs.  
**pyyaml** also for reading and writing SSINS outputs.  

### Optional Dependencies

**matplotlib** will be necessary if the user wants to use the Catalog_Plot and plot_lib libraries.  
**astropy** will be necessary if desiring to write mwaf files. (required for pyuvdata anyway, see pyuvdata dependencies)


## Installation

Once pyuvdata and all dependencies of that are installed, simply clone this repo and either add it to your python path, or run `python setup.py install`.

## Documentation

Docs are available at: https://ssins.readthedocs.io/en/latest/  

There are also tutorials available here: https://ssins.readthedocs.io/en/latest/tutorial.html?highlight=tutorial  
