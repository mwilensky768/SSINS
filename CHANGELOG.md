# SSINS Change Log

## Unrealeased
- Updates pyuvdata requirements to 3.1.3, which always uses future_array_shapes.
- Updates python requirement to 3.10

## 1.4.7
- Added handling for negative `sig_thresh` such that when set, the associated shape will only trigger
on a negative sig smaller than `sig_thresh`. Behavior for positive `sig_thresh` is unchanged;
this will continue to use the absolute value of sig. Setting a negative `sig_thresh` for the
`narrow` shape is not supported and will trigger a warning.
- Now requires pyuvdata 2.4.1, which introduces several new required parameters for 
UVData (thus SS) and UVFlag (thus INS). This required refactoring the INS constructor,
though the API has been preserved. However, if reading SSINS h5 files written with an earlier 
pyuvdata version, there may be an error issued from pyuvdata about the new required keywords.
In this case, one will need to specify the originating telescope of the file, or disable run_check to proceed.
- Changed how `_data_params` property is handled.
- Moved installation totally over to pyroject
- Remove PendingDeprecation warning in SS.read



## 1.4.6
- Requires pyuvdata 2.2.8 or greater, which changes the astropy dependency. Since
this version of pyuvdata uses a version of astropy that no longer supports python 3.7,
we have also dropped that from our CI. However, python 3.10 has been added to the CI.
- Make unit tests use tmp_path pytest fixtures, which makes running them much cleaner
- **notable api change** Got rid of old matplotlib functionality that was deprecated.
Must now supply a _string_ for the cmap argument in all the plotting code,
which is the name of the colormap you would like to use, rather than the actual colormap object.
- Update diff unit test to be more flexible with pyuvdata antenna_number conventions
- Call SS.apply_flags at end of SS.diff so that object is always ready to pass
to INS after diff
- Fixed xscale kwarg passing bug in plotting code
- Updated MWA_EoR_High_uvfits_write and then moved it to the EoRImaging/pipeline_scripts repo
- Change version handling to use setuptools_scm.
- Update Run_HERA_SSINS.py to take auto_metrics and ant_metrics files to calculate
a set of xants, which will be excluded from the data when flagging.


## 1.4.5
- Replaced is with == when comparing to string literals
- Added write_meta util function for easy metadata writing.
- Added match_filter writeout to yaml format.
- Added extent options to plotting libraries so that default ticks on plots would
  be sensible and easy.
- Added inplace keyword to select
- Added extend and alpha keyword for image_plot colorbars
- Add warning filters to unit tests.

## 1.4.4

- Added change log
