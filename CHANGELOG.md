# SSINS Change Log

## Unreleased
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
