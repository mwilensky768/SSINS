# SSINS Readme

SSINS is a python package intended for radio frequency interference flagging in radio astronomy datasets. It stands for Sky-Subtracted Incoherent Noise Spectra.

## Dependencies

**pyuvdata 1.3.8 or better**
**pyuvdata has its own dependencies!**
See https://github.com/RadioAstronomySoftwareGroup/pyuvdata

**h5py** for reading and writing SSINS outputs
**pyyaml** also for reading and writing SSINS outputs

### Optional Dependencies

**matplotlib** will be necessary if the user wants to use the Catalog_Plot and plot_lib libraries
**astropy** will be necessary if desiring to write mwaf files

## Installation

Once pyuvdata and all dependencies of that are installed, simply clone this repo and either add it to your python path, or run `python setup.py install`.

## Documentation

Docs are available at: https://ssins.readthedocs.io/en/latest/

There are also tutorials available here: https://ssins.readthedocs.io/en/latest/tutorial.html?highlight=tutorial
