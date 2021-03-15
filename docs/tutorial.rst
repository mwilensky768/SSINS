Tutorial
========

.. testsetup::
   from __future__ import absolute_import, division, print_function

------------------------
Basic SSINS Construction
------------------------
There are three main classes in the software: SS (sky_subtract),
INS (incoherent_noise_spectrum), and MF (match_filter). We will use these
classes to navigate the various steps in the SSINS process detailed in the
paper (https://arxiv.org/abs/1906.01093).

Generating the sky-subtracted visibilities
------------------------------------------

(a) Initializing an SS object and reading in raw data
*****************************************************
::
  >>> from SSINS import SS

  >>> # The SS object is a subclass of a UVData object, and therefore has all of its attributes and methods
  >>> # We initialize it identically
  >>> # See UVData documentation on https://pyuvdata.readthedocs.io/en/latest/
  >>> ss = SS()

  >>> # Read data by specifying a filepath as an argument to the read method
  >>> filepath = 'SSINS/data/1061313128_99bl_1pol_half_time.uvfits'
  >>> # By default, the visibilities are NOT differenced in time on read (see paper). This is for compatibility with multi-file reading.
  >>> ss.read(filepath, diff=True)

(b) Passing keyword arguments to SS.read
****************************************
::
  >>> import numpy as np

  >>> # SS.read is actually a small wrapper around UVData.read; they share keywords
  >>> # In particular, select on read and reading only metadata function as usual (see UVData.select documentation)
  >>> ss = SS()
  >>> ss.read(filepath, read_data=False)

  >>> # The following lines make use of the time_array attribute (metadata) to
  >>> # read in all but the first and last integrations
  >>> times = np.unique(ss.time_array)[1:-1]
  >>> # This read() call uses the `diff` keyword to difference the data automatically along the time axis
  >>> ss.read(filepath, read_data=True, times=times, diff=True)

(c) Differencing along the frequency axis
*****************************************
::
  >>> # We can opt to difference visibilities along the frequency axis instead of the default time axis
  >>> # To do this, use the separate keyword `diff_freq` and set it to true when you call the read() function
  >>> # The two modes work otherwise the same except for the axis the visibilities are differenced upon.

  >>> # Note that this will override the time-differenced data if you ran ss.read() from section (b)
  >>> ss.read(filepath, read_data=True, times=times, diff_freq=True)

(d) Applying flags
******************
::
  >>> # SS.data_array is a numpy masked array. To "apply flags" is to change the mask of the data_array.
  >>> # The proper way to apply flags to the sky-subtracted data is to use the apply_flags method
  >>> # To apply the original flags in the raw data file, make the following call
  >>> ss.apply_flags(flag_choice='original')
  >>> # Note that the original flags are always stored in the flag_array attribute
  >>> # The flag_choice keyword is stored in an attribute
  >>> print(ss.flag_choice)
  original

  >>> # You can apply flags from a custom flag array that is the same shape as the data
  >>> custom = np.zeros(ss.data_array.shape, dtype=bool)
  >>> # Let us make it so that only the first frequency channel is flagged and nothing else
  >>> custom[:, 0, 0, :] = True
  >>> # Apply these flags in the following way
  >>> ss.apply_flags(flag_choice='custom', custom=custom)
  >>> print(ss.flag_choice)
  custom

  >>> # Unflag the data by setting flag_choice=None (note this is actually the default!!)
  >>> ss.apply_flags(flag_choice=None)
  >>> # Check if anything is flagged, for demonstration purposes
  >>> print(np.any(ss.data_array.mask))
  False

(e) Plotting using Catalog_Plot
*******************************
::
  >>> from SSINS import Catalog_Plot as cp
  >>> import os

  >>> # The Catalog_Plot library contains wrappers around plot_lib functions for basic plotting needs
  >>> # See the documentation: https://ssins.readthedocs.io/en/latest/Catalog_Plot.html
  >>> # Each function in Catalog_Plot requires a class instance and a filename prefix as arguments (a suffix is appended by the wrapper)
  >>> # Whatever unique identifying information for the plot should be specified in the prefix
  >>> prefix = 'SSINS/data/test_data'

  >>> # To make a Histogram of the Visibility Differences (a VDH, figure 1 of paper), and save it as a pdf, do the following
  >>> # This also plots a fit estimated from the data
  >>> cp.VDH_plot(ss, prefix, file_ext='pdf', post_flag=False)
  >>> # Check to see that the file exists
  >>> print(os.path.exists('%s_VDH.pdf' % (prefix)))
  True

  >>> # Let's apply flags and plot the flagged data alongside the unflagged data, without fits
  >>> # We also want legend labels and a legend
  >>> ss.apply_flags('original')
  >>> new_prefix = '%s_flag_unflag_nofits' % prefix
  >>> cp.VDH_plot(ss, new_prefix, file_ext='pdf', pre_flag=True,
  ...             post_flag=True, pre_model=False, post_model=False,
  ...             post_label='Post-Flag Data', pre_label='Pre-Flag Data',
  ...             legend=True)
  >>> print(os.path.exists('%s_VDH.pdf' % (new_prefix)))
  True

Making and writing an incoherent noise spectrum
-----------------------------------

(a) Making an incoherent noise spectrum from sky-subtracted data
****************************************************************
::
  >>> from SSINS import INS

  >>> # Making an INS from sky-subtracted data is as simple as passing an SS instance as an argument
  >>> ins = INS(ss)
  >>> # This averages the amplitudes of the sky-subtracted data over the baselines, taking into account flags that were applied

(b) Making an incoherent noise spectrum out of autocorrelations
***************************************************************
::
    >>> auto_ss = SS()

    >>> # Read data by specifying a filepath as an argument to the read method
    >>> auto_filepath = 'SSINS/data/1061312640_autos.uvfits'
    >>> auto_ss.read(auto_filepath, diff=True)
    >>> auto_ins = INS(auto_ss, spectrum_type="auto")

(c) Plotting using Catalog_Plot
*******************************
::
  >>> # Plotting INS is similar to plotting a VDH, just with a different function
  >>> # This plots all polarizations present in the file separately
  >>> # The first column are the baseline-averaged amplitudes, while the second column shows the mean-subtracted data (z-scores)
  >>> cp.INS_plot(ins, prefix, file_ext='pdf')
  >>> print(os.path.exists('%s_SSINS.pdf' % prefix))
  True

  >>> # You can specify various plotting nuances with keywords
  >>> # Let's set some frequency ticks every 50 channels
  >>> xticks = np.arange(0, len(ins.freq_array), 50)
  >>> xticklabels = ['%.1f' % (ins.freq_array[tick]* 10 ** (-6)) for tick in xticks]
  >>> tick_prefix = '%s_ticks' % prefix
  >>> cp.INS_plot(ins, tick_prefix, file_ext='pdf', xticks=xticks, xticklabels=xticklabels)
  >>> print(os.path.exists('%s_SSINS.pdf' % tick_prefix))
  True

(d) Plotting using the plot_lib library
***************************************
::
  >>> import matplotlib.pyplot as plt
  >>> from matplotlib import cm
  >>> from SSINS import plot_lib
  >>> # Let's plot the first polarization data and z-scores
  >>> fig, ax = plt.subplots(nrows=2, figsize=(16, 9))
  >>> # The averaged amplitudes are stored in the metric_array parameter
  >>> plot_lib.image_plot(fig, ax[0], ins.metric_array[:, :, 0],
  ...                     title='XX Amplitudes', xticks=xticks,
  ...                     xticklabels=xticklabels)
  >>> # The z-scores are stored in the metric_ms parameter.
  >>> # Let's choose a diverging colorbar and center it on zero using the cmap and midpoint keywords.
  >>> plot_lib.image_plot(fig, ax[1], ins.metric_ms[:, :, 0],
  ...                     title='XX z-scores', xticks=xticks,
  ...                     xticklabels=xticklabels, cmap=cm.coolwarm,
  ...                     midpoint=True)
  >>> fig.savefig('%s_plot_lib_SSINS.pdf' % prefix)
  >>> print(os.path.exists('%s_plot_lib_SSINS.pdf' % prefix))
  True

(e) Saving out and reading in a spectrum
****************************************
::
  >>> # The INS.write method saves out h5 files that can be read both by INS objects and UVFlag objects
  >>> # By default it saves out the metric_array in the file, z-scores must be saved separately
  >>> # Set clobber=True to overwrite files with the same prefix (default is False)
  >>> ins.write(prefix, clobber=True)
  >>> ins.write(prefix, output_type='z_score', clobber=True)
  >>> print(os.path.exists('%s_SSINS_data.h5' % prefix))
  True
  >>> print(os.path.exists('%s_SSINS_z_score.h5' % prefix))
  True

  >>> # This file can later be read upon instantiation of a new object
  >>> # The z-scores will be recalculated on instantiation, so no need to read in the z-scores
  >>> new_ins = INS('%s_SSINS_data.h5' % prefix)
  >>> # Check equality
  >>> print(np.all(ins.metric_array == new_ins.metric_array))
  True

Flagging an INS using a match_filter (MF)
-----------------------------------------

(a) Constructing a filter with no additional sub-bands
************************************************
::
  >>> from SSINS import MF

  >>> # The MF class requires a frequency array and significance threshold as positional arguments
  >>> # We will disable searching for broadband streaks and provide no additional sub-bands for the filter
  >>> # First we need to define a sig_thresh dictionary for our only shape (narrowband)
  >>> sig_thresh = 5
  >>> mf = MF(ins.freq_array, sig_thresh, streak=False, narrow=True, shape_dict={})

(b) Constructing a filter for streaks and Western Australian DTV in MWA EoR Highband
************************************************************************************
::
  >>> # Use the shape_dict keyword to provide custom sub-bands to search over during flagging
  >>> # The input should be a dictionary, where the key is the name of the shape and the value are the lower/upper frequencies in hz
  >>> shape_dict = {'TV6': [1.74e8, 1.81e8],
  ...               'TV7': [1.81e8, 1.88e8],
  ...               'TV8': [1.88e8, 1.95e8],
  ...               'TV9': [1.95e8, 2.02e8]}
  >>> # We also need to apply significance thresholds for each shape, including 'narrow' and 'streak'
  >>> # In principle, these can be different values per shape, see advanced techniques.
  >>> sig_thresh = 5
  >>> mf = MF(ins.freq_array, sig_thresh, shape_dict=shape_dict, streak=True, narrow=True)

(c) Constructing a filter for streaks and South African DTV in HERA below 200 Mhz
*********************************************************************************
::
  >>> # Use the shape_dict keyword to provide custom sub-bands to search over during flagging
  >>> # The input should be a dictionary, where the key is the name of the shape and the value are the lower/upper frequencies in hz
  >>> shape_dict = {'TV4': [1.74e8, 1.82e8],
  ...               'TV5': [1.82e8, 1.9e8],
  ...               'TV6': [1.9e8, 1.98e8]}
  >>> # Technically 2 Mhz of channel 7 should appear, but we omit that in this example
  >>> # We also need to apply significance thresholds for each shape, including 'narrow' and 'streak'
  >>> # In principle, these can be different values per shape
  >>> sig_thresh = 5
  >>> mf = MF(ins.freq_array, sig_thresh, shape_dict=shape_dict, streak=True)

(d) Using the filter to flag the noise spectrum
***********************************************
::
  >>> # Construct the filter that you want to use.
  >>> # For the test data we will use the MWA DTV example above
  >>> shape_dict = {'TV6': [1.74e8, 1.81e8],
  ...               'TV7': [1.81e8, 1.88e8],
  ...               'TV8': [1.88e8, 1.95e8],
  ...               'TV9': [1.95e8, 2.02e8]}
  >>> sig_thresh = 5
  >>> mf = MF(ins.freq_array, sig_thresh, shape_dict=shape_dict, streak=True)

  >>> # Use the apply_match_test method to flag the INS (this applies the flags to the mask of the metric array)
  >>> mf.apply_match_test(ins) # doctest: +SKIP

(e) Saving the INS mask out to an h5 file
*****************************************
::
  >>> # Just use the write method as above, with the right output_type
  >>> ins.write(prefix, output_type='mask', clobber=True)
  >>> print(os.path.exists('%s_SSINS_mask.h5' % prefix))
  True


(f) Getting time propagated flags from the INS mask
***************************************************
::
  >>> from pyuvdata import UVData, UVFlag
  >>> # Each integration in the SSINS is a result of a difference of paired integrations
  >>> # To get flags for the raw data, we have to propagate flagged INS samples in time to all possible contributing times
  >>> # The mask_to_flags method returns an array where we have done this. This is useful for comparing to other UVFlag objects
  >>> flags = ins.mask_to_flags()

  >>> # We can write these out to an h5 file as well, but we need to make a UVFlag object from the original data
  >>> uvd = UVData()
  >>> uvd.read(filepath, times=times)
  >>> uvf = UVFlag(uvd, waterfall=True, mode='flag')
  >>> ins.write(prefix, output_type='flags', clobber=True, uvf=uvf)
  >>> print(os.path.exists('%s_SSINS_flags.h5' % prefix))
  True

(g) Applying time-propagated flags from INS to a UVData object and write new file
*********************************************************************************
::
  >>> from pyuvdata import utils as uvutils
  >>> uvutils.apply_uvflag(uvd, uvf)
  >>> uvd.write_uvfits('SSINS/data/tutorial_test_writeout.uvfits')

(h) Writing flags to an mwaf file
*********************************
::
  >>> # We can add or replace flags from an existing mwaf file
  >>> # An mwaf file is a special fits file for storing flags of raw MWA data
  >>> # A special keyword option in ins.write() helps write them
  >>> # You must supply a list of existing mwaf files from which to gather the header data
  >>> # Currently you must flag at the same time/freq resolution as the data in the existing mwaf_files

  >>> # For instance if you wanted to flag just the first two coarse bands for an obsid
  >>> mwaf_files = ['/path/to/obsid_01.mwaf', '/path/to/obsid/obsid_02.mwaf'] # doctest: +SKIP



  >>> # As usual you must supply a prefix for the file.
  >>> # You can choose to add flags to the file from SSINS flagging, or totally replace them
  >>> prefix_add = '/path/to/obsid_SSINS_add' # doctest: +SKIP
  >>> prefix_replace = '/path/to/obsid_SSINS_replace' # doctest: +SKIP
  >>> # Can use Ncoarse keyword if input data does not have 24 coarse channels in it (default is 24)
  >>> ins.write(prefix_add, output_type='mwaf', mwaf_files=mwaf_files, # doctest: +SKIP
  ...           mwaf_method='add', Ncoarse=24) # doctest: +SKIP
  >>> ins.write(prefix_replace, output_type='mwaf', mwaf_files=mwaf_files, # doctest: +SKIP
  ...           mwaf_method='replace', Ncoarse=24) # doctest: +SKIP

  >>> # Be sure to set clobber=False (default) if using the same prefix
  >>> # as the original file and you don't want to overwrite


--------------------
Getting Version Info
--------------------

SSINS uses setuptools_scm to get the version from git tags.

(a) Printing the version.
*************************
::
  >>> from SSINS import __version__ as SSINS_version
  >>> # This string will be recorded in the history string of written outputs
  >>> print(SSINS_version) # doctest: +SKIP

-------------------
Advanced Techniques
-------------------
The techniques below are for users who are already familiar with the basic tutorials above.

Using INS.mean_subtract
-----------------------

(a) Basic functionality
***********************
::
  >>> from SSINS import INS
  >>> ins = INS('SSINS/data/1061313128_99bl_1pol_half_time_SSINS.h5')

  >>> # The mean_subtract method returns the result of mean_subtraction
  >>> # It does NOT automatically change the metric_ms attribute
  >>> ms_arr = ins.mean_subtract()

  >>> # You can do mean subtraction on just a subset of the frequencies to get a smaller output
  >>> # This functionality is used to speed up match filtering
  >>> # Let's just do the first ten frequency channels
  >>> ms_arr = ins.mean_subtract(freq_slice=slice(0, 10))

(b) Subtracting a polynomial fit instead of the mean
****************************************************
::
  >>> # If the noise levels are expected to change over the course of the obs (due to a refrigeration cycle for instance)
  >>> # then may want to subtract a polynomial fit that describes the drift
  >>> # The mean_subtract method uses INS.order to determine what degree of polynomial to subtract
  >>> # Default order is 0, which just does mean subtraction
  >>> ins.order = 1
  >>> ms_arr_ord_1 = ins.mean_subtract()
  >>> ins.order = 0
  >>> ms_arr_ord_0 = ins.mean_subtract()

  >>> # Can ask for the fit coefficients on a per-frequency basis
  >>> ins.order = 2
  >>> ms_arr_ord_2, coeffs_ord_2 = ins.mean_subtract(return_coeffs=True)
  >>> # The shape is (INS.order + 1, Nfreqs, Npols) where Nfreqs is the number of frequencies in the slice
  >>> # It goes from higher degree coefficients to lower degree

Extra Flagging Bits
-------------------

(a) Flagging all times for highly contaminated channels
*******************************************************
::
  >>> # Suppose you want to flag any channels with less than 40% clean data
  >>> # Construct a MF as follows
  >>> sig_thresh = {'narrow': 5, 'streak': 5}
  >>> mf = MF(ins.freq_array, sig_thresh, tb_aggro=0.4)
  >>> mf.apply_match_test(ins, time_broadcast=True) # doctest: +SKIP

(b) Broadcasting flags over subbands
************************************
::
  >>> # Suppose you want to spread flags over certain subbands if RFI is found in those subbands
  >>> # For instance: maybe you want to flag a whole TV band if anything is found in it
  >>> # Make a broadcast_dict (this is a South Africa example)
  >>> broadcast_dict = {'TV4': [174e6, 182e6], 'TV5': [182e6, 190e6], 'TV6': [190e6, 192e6]}
  >>> mf = MF(ins.freq_array, 5, broadcast_dict=broadcast_dict)
  >>> # Note that intervals in SSINS are INCLUSIVE on both ends

(c) Broadcasting flags over subbands, with guard bands
******************************************************
  >>> # Depending on the channelization, these subbands may overlap
  >>> # This means events found at the very edge of one subbands may induce flags in the other, unless a guard band is thrown in
  >>> # An example 100 kHz guard band program might look like this
  >>> guard_width = 100e3
  >>> broadcast_dict = {}
  >>> broadcast_dict['TV4'] = [174e6, 182e6 - guard_width]
  >>> broadcast_dict['guard_4_5'] = [182e6 - guard_width, 182e6 + guard_width]
  >>> broadcast_dict['TV5'] = [182e6 + guard_width, 190e6 - guard_width]
  >>> mf = MF(ins.freq_array, 5, broadcast_dict=broadcast_dict)


(d) Calculating occupancy
*************************
::
  >>> # A dictionary that reports the occupancy of the shapes found by the flagger can be calculated
  >>> # See util.calc_occ docs
  >>> occ_dict = util.calc_occ(ins, mf, 0) # doctest: +SKIP
  >>> # This can then be written to a yaml
  >>> with open("SSINS/data/test_occ.yml", "w") as yaml_file: # doctest: +SKIP
  ...    yaml.safe_dump(occ_dict, yaml_file) # doctest: +SKIP


(e) Setting different significance thresholds per shape
*******************************************************
::
  >>> # One may pass a dictionary of significance thresholds to set different
  >>> # thresholds per shape. All desired shapes, including narrow and broad if
  >>> # desired, must be included.
  >>> sig_thresh = {'narrow': 5, 'streak': 20, 'TV6': 5, 'TV7': 5, 'TV8': 5, 'TV9': 5}
  >>> shape_dict = {'TV6': [1.74e8, 1.81e8],
  ...               'TV7': [1.81e8, 1.88e8],
  ...               'TV8': [1.88e8, 1.95e8],
  ...               'TV9': [1.95e8, 2.02e8]}
  >>> mf = MF(ins.freq_array, sig_thresh, shape_dict=shape_dict)

(f) Writing out flags to a visibility file from a UVData object
***************************************************************
::
  >>> ss = SS()
  >>> ss.read(filepath, times=times, flag_choice='original', diff=True)
  >>> ss.write('SSINS/data/tutorial_test_writeout_2.uvfits', 'uvfits', UV=uvd)
