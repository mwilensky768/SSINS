# SSINS Readme

SSINS is a python package intended for radio frequency interference flagging in radio astronomy datasets. It stands for Sky-Subtracted Incoherent Noise Spectra.

## Dependencies

**pyuvdata** **master branch** which is available at https://github.com/RadioAstronomySoftwareGroup/pyuvdata
**pyuvdata has its own dependencies!**

**h5py** for reading and writing SSINS outputs
**pyyaml** also for reading and writing SSINS outputs

### Optional Dependencies

**matplotlib** will be necessary if the user wants to use the Catalog_Plot and plot_lib libraries
**astropy** will be necessary if desiring to write mwaf files

## Installation

Once pyuvdata and all dependencies of that are installed, simply clone this repo and either add it to your python path, or run `python setup.py install`.

## Documentation

Functions and classes have docstrings available. There are also tutorials available here: https://ssins.readthedocs.io/en/doc_writing_refactor/tutorial.html?highlight=tutorial
