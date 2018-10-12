Tutorial
========

.. testsetup::
   from __future__ import absolute_import, division, print_function

--
SS
--

sky_subtract: Initializing
--------------------------
Initializing by using pyuvdata

(a) Initializing from a UVData object
*************************************
::

  >>> from pyuvdata import UVData
  >>> from SSINS import SS
  >>> UV = UVData()

  # This is an MWA uvfits file trimmed to a single pol, 99 baselines, and only
  # half of the observing time. We will read in only cross correlations.
  # This is typical since the autocorrelations have slightly different statistics
  # than the cross correlations.
  >>> UV.read('SSINS/data/1061313128_99bl_1pol_half_time.uvfits',
              file_type='uvfits', ant_str='cross')

  # Now that we have a UVData object, we can initialize the SS class.
  # We will give it an outpath for saving data as well as an obs name.
  # We will also tell it to reshape the UVData object's data array and do the
  # sky subtraction, since we have not done it ourselves. This is actually the
  # default setting, but we set it explicitly here.
  >>> ss = SS(obs=1061313128_99bl_1pol_half_time,
              outpath='SSINS/data/test_outputs' UV=UV, diff=True)

  # Now the SS instance has its own UVData object that it carries around that
  # can be modified in the usual ways. Note the following.
  >>> ss.UV is UV
  True

(b) Initializing with a Path
****************************
::

  >>> from SSINS import SS
  >>> import numpy as np

  # We can also just read in the data upon initialization in the following way
  >>> inpath = 'SSINS/data/1061313128_99bl_1pol_half_time.uvfits'
  >>> outpath = 'SSINS/data/test_outputs'
  >>> obs = '1061313128_99bl_1pol_half_time'

  # We set up a keyword dictionary for the select on read functionality
  # This time we will only keep the first 16 frequency channels
  >>> read_kwargs = {'ant_str': 'cross',
                     'freq_chans': np.arange(16)}

  # We will also remove some known problematic times, by index
  # This is accomplished by reading in the metadata first and finding the times
  # by index to remove, then doing a select on read.
  >>> bad_time_indices = [0, -1, -2, -3]
  >>> ss = SS(obs=obs, outpath=outpath, inpath=inpath, read_kwargs=read_kwargs,
              bad_time_indices=bad_time_indices)

(c) Applying Flags
**********************************************
::

  >>> from SSINS import SS
  >>> inpath = 'SSINS/data/1061313128_99bl_1pol_half_time.uvfits'
  >>> outpath = 'SSINS/data/test_outputs'
  >>> obs = '1061313128_99bl_1pol_half_time'
  >>> read_kwargs = {'ant_str': 'cross'}

  # The SSINS package utilizes numpy masked arrays for a number of purposes.
  # This allows us to work with an additional set of flags along with the original flag array.
  # We can apply the original flags on read in the following way.

  >>> ss = SS(obs=obs, outpath=outpath, inpath=inpath, read_kwargs=read_kwargs,
              flag_choice='original')
  >>> np.all(ss.UV.data_array.mask == ss.UV.flag_array)
  True

  # There are multiple ways to unflag the data, but one way is to use the
  # apply_flags() method.

  >>> ss.apply_flags(choice=None)
  >>> np.any(ss.UV.data_array.mask)
  False

  # Custom flag masks can also be applied by either manually manipulating the mask
  # or on read by supplying a custom flag array.

  custom = np.zeros_like(ss.UV.flag_array)
  custom[:, :, 0, 0, :] = 1
  ss.apply_flags(choice='custom', custom=custom)

  # This flags everything in the zeroth frequency channel.
  # apply_flags() is called on initialization, so any flag manipulation can be
  # done on read or later as is fit for the situation
