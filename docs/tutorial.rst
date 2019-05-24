Tutorial
========

.. testsetup::
   from __future__ import absolute_import, division, print_function

--
SS
--

sky_subtract: Initializing
--------------------------
Initializing the sky_subtract class by using pyuvdata

(a) Initializing an SS object
*****************************
::

  >>> from SSINS import SS

  # The SS object is a subclass of a UVData object and is initialized in an identical way
  >>> ss = SS()

  # This is an MWA uvfits file trimmed to a single pol, 99 baselines, and only
  # half of the observing time. We can pass the same keywords for UVData.read to SS.read.
  # For instance, we read in only cross correlations here.
  >>> ss.read('SSINS/data/1061313128_99bl_1pol_half_time.uvfits', ant_str='cross')

  # We time-difference the visibilities during initialization.
  # We adjust the attributes during differencing to ensure the SS object is still a UVData object
  >>> ss.check()
  True

(b) Applying Flags
**********************************************
::

  >>> from SSINS import SS
  >>> inpath = 'SSINS/data/1061313128_99bl_1pol_half_time.uvfits'

  # The SSINS package utilizes numpy masked arrays for a number of purposes.
  # This allows us to work with an additional set of flags along with the original flag array.
  # We can apply the original flags (propagated through differencing) on read in the following way.

  >>> ss = SS()
  # This omits the  autocorrelations on read - they have very different statistics than the cross correlations
  >>> ss.read(inpath, flag_choice='original', ant_str='cross')
  >>> np.all(ss.UV.data_array.mask == ss.UV.flag_array)
  True

  # There are multiple ways to change the flags on the data, but the proper way
  # is to use the apply_flags method. Custom flag masks can be applied.
  # This leaves the flag_array attribute untouched, and simply changes the mask
  # on the data.

  >>> custom = np.zeros_like(ss.UV.flag_array)
  >>> custom[:, 0, 0, :] = 1
  >>> ss.apply_flags(flag_choice='custom', custom=custom)

  # This flagged everything in the zeroth frequency channel.
  # We can also apply masks from a flagged INS as follows (see INS and MF tutorials for details)
  >>> ss.apply_flags(flag_choice='custom', INS=ins)

  # The following lines unflag the data.
  >>> ss.apply_flags(flag_choice=None)
  >>> np.any(ss.UV.data_array.mask)
  False

  # apply_flags() is called on read (default choice is None), so any flag
  # manipulation can be done on read or later as is fit for the situation

sky_subtract: Plotting
----------------------
A useful preliminary diagnostic plot is a histogram of the sky-subtracted
visibility amplitudes. We cover that procedure here.

(a) Invoking Catalog_Plot
*************************
::

  >>> from SSINS import Catalog_Plot as cp
  >>> import os

  # We use the VDH_plot (Visibility Difference Histogram) function from Catalog_Plot
  # We need a prefix for the output file onto which will be attached a tag saying what the plot is

  >>> prefix = 'SSINS/data/tutorial_'

  # Let's save a png, plot in log-log, only plot data that is not flagged, and
  # generate a model for the thermal background
  >>> cp.VDH_plot(ss, prefix, file_ext='png', xlabel='Amplitude (~Jy)', xscale='log',
  >>>             yscale='log', pre_flag=False, post_model=True)

  # Check for the file SSINS/data/tutorial_VDH.png. The obs we used earlier in
  # the tutorial has DTV RFI in it which will have contaminated the maximumum
  # likelihood estimate for the model, so the thermal model may look a little funny

---
INS
---

incoherent_noise_spectrum: Initializing
---------------------------------------
An INS can be initialized from an SS object, as well as from a previously
saved INS.

(a) From an SS object
*********************
::

  >>> from SSINS import INS

  # Simply pass the SS object from which the INS will be made
  >>> ins = INS(ss)

(b) From a saved file
*********************
::

  # This will read in a saved INS specified by inpath
  >>> inpath = 'SSINS/data/1061313128_99_bl_1pol_half_time_SSINS.h5'
  >>> ins = INS(inpath)

incoherent_noise_spectrum: Writing
----------------------------------
We can write the information from an INS out to h5 files using the write method.
There are three main data products to write out: (1) The baseline averaged
visibility difference amplitudes, (2) The z-scores from mean-subtraction, and (3)
any mask that may have come from flagging.

(a) Writing the three main data products
****************************************
::

  # We need to specify a prefix for the files
  >>> prefix = 'SSINS/data/tutorial_'

  # Now lets write the data
  >>> ins.write(prefix, output_type='data')
  # And lets write the z-scores
  >>> ins.write(prefix, output_type='z_score')

  # We detail how to use the match_filter to flag an INS in the match_filter section
  # This will apply masks to the data, which we write as follows
  >>> ins.write(prefix, output_type='mask')
  # We can apply these on read from the output file using the mask_file keyword on init

(b) Writing time-propagated flags
*********************************
::

  # The time-propagated flags (extending them back across the time-difference)
  # are calculated using the mask_to_flags method
  >>> tp_flags = ins.mask_to_flags()

  # This generates a flag array of the original length of the data where
  # any samples that would have contributed to a flagged difference are flagged

  # We can write these flags out (readable by UVFlag!)
  # It automatically calls this method when writing flags (different than writing mask)
  >>> ins.write(data_output='flags')

(c) Writing an mwaf file
************************
::

  # An mwaf file is a special fits file for storing flags of raw MWA data
  # A special keyword option in ins.write() helps write them
  # You must supply a list of existing mwaf files from which to gather the header data
  # Currently you must flag at the same time/freq resolution as the data in the existing mwaf_files

  # For instance if you wanted to flag just the first two coarse bands for an obsid
  >>> mwaf_files = ['/path/to/obsid_01.mwaf', '/path/to/obsid/obsid_02.mwaf']



  # As usual you must supply a prefix for the file.
  # You can choose to add flags to the file from SSINS flagging, or totally replace them
  >>> prefix_add = '/path/to/obsid_SSINS_add'
  >>> prefix_replace = '/path/to/obsid_SSINS_replace'
  >>> ins.write(prefix_add, output_type='mwaf', mwaf_files=mwaf_files,
                mwaf_method='add')
  >>> ins.write(prefix_replace, output_type='mwaf', mwaf_files=mwaf_files,
                mwaf_method='replace')

  # Be sure to set clobber=False (default) if using the same prefix
  # as the original file and you don't want to overwrite

incoherent_noise_spectrum: Using the mean_subtract() Method
-----------------------------------------------------------

(a) Basic Use
*************
::

  # The method does not automatically set the data_ms attribute, so the assignment
  # must be done manually
  >>> ins.data_ms = ins.mean_subtract()

  # A slice of the array can be calculated by using the f keyword (f for frequencies)
  # Set up a slice object for frequency channels 100 to 199 inclusive
  >>> f = slice(100, 200)
  >>> ins.data_ms[:, :, f] = ins.mean_subtract(f=f)

(b) Using the order Parameter
*****************************
::

  # Sometimes the mean appears to drift in time to linear or higher order
  # A polynomial fit to each channel can be constructed using the order parameter
  >>> ins.order = 2
  >>> ins.data_ms = ins.mean_subtract(order=2)

  # That made a quadratic fit for each channel

  # This can also be done on initialization in the same way
  >>> ins = INS(inpath, order=1)

  # That made a linear fit
  # The order parameter defaults to 0 (just take a mean)


incoherent_noise_spectrum: Plotting
-----------------------------------

There exists a small plotting library in the repo called plot_lib which exists
for the sake of convenience. There are some wrappers around these functions in
the repo contained in Catalog_Plot.

(a) Using Catalog_Plot
**********************
::

  >>> from SSINS import Catalog_Plot as cp
  >>> from matplotlib import cm

  # Let's make ticklabels (in Mhz) using the frequency array
  >>> prefix = 'SSINS/data/tutorial_'
  >>> xticks = np.arange(0, len(ins.freq_array), 50)
  >>> xticklabels = ['%.1f' % (ins.freq_array[tick] * 10 ** (-6)) for tick in xticks]

  # We will plot images of the data and the z-scores as png's (default is pdf)
  # We clip all data above 150 and all z-scores whose absolute value is greater than 5
  # We also prescribe a colormap for the data
  >>> cp.INS_plot(ins, prefix, data_cmap=cm.plasma, vmin=0, vmax=150, ms_vmin=-5,
  >>>             ms_vmax=5, xticks=xticks, xticklabels=xticklabels,
  >>>             xlabel='Frequency (Mhz)')

  # If using the original data in the above tutorials with no flags applied to
  # make the INS, there should be some DTV visible in the middle of the plot
  # in all polarizations in the output file.

(b) Using plot_lib
******************
::

  # Finer control over which plots come out can be obtained without the
  # Catalog_Plot wrapper using just plot_lib
  >>> from SSINS import plot_lib
  >>> from matplotlib import cm
  >>> import matplotlib.pyplot as plt

  >>> fig, ax = plt.subplots(nrows=2)
  >>> prefix = 'SSINS/data/figs/tutorial_order_compare'

  # Here we take an INS and plot its mean-subtracted data in the first
  # polarization with different order parameters

  >>> for i in range(2):
  ...     ins.ms = ins.mean_subtract(order=i)
  ...     plot_lib.image_plot(fig, ax[i], ins.metric_ms[:, 0, :, 0],
  ...                         cmap=cm.coolwarm, freq_array=ins.freq_array[0],
  ...                         title='order = %i' % i, vmin=-5, vmax=5)
  >>> fig.savefig('%s/tutorial_order_compare.png' % (prefix, ins.obs))

  # This particular example is useful when the overall noise level appears to be
  # drifting over the course of the observation and you want to ignore that drift
  # If using the usual tutorial file from above, this may appear to
  # spread the DTV contamination in time - it can still be flagged reasonably
  # since the match_filter is iterative

--
MF
--

match_filter: initialization
----------------------------

(a) Initializing
****************
::

  >>> from SSINS import MF

  # Initialization involves setting desired parameters (reasonable defaults exist)
  # RFI shapes are passed with a dictionary (this example is digital TV in
  # Western Australia, where the MWA is located)
  >>> shape_dict = {'TV6': [1.74e8, 1.81e8],
                    'TV7': [1.81e8, 1.88e8],
                    'TV8': [1.88e8, 1.96e8]}

  # sig_thresh governs the maximal strength of outlier to leave unflagged
  # A reasonable value can be estimated from the size of the data,
  # as detailed in the paper, section (section): (arxiv link)
  >>> sig_thresh = 5

  # The single-frequency and broadband streak flaggers can be turned off (default on)
  >>> narrow = False
  >>> streak = False

  # An frequency array is required for initialization (typically taken from an INS to be flagged)
  >>> mf = MF(ins.freq_array, sig_thresh, shape_dict=shape_dict, narrow=narrow, streak=streak)

match_filter: Applying Tests
----------------------------

(a) Basic Match-Shape Test:
***************************
::

  >>> from SSINS import Catalog_Plot as cp

  # Here, the shapes in the shape_dictionary are tested for
  # This method will automatically apply flags to samples which match the flagging criterion
  # We will also append events to the ins.match_events attribute
  >>> mf.apply_match_test(ins, event_record=True)

  # We can plot the results for the INS and the mask will be applied to the plot automatically
  >>> cp.INS_plot(ins, prefix, ms_vmin=-mf.sig_thresh, ms_vmax=mf.sig_thresh)

  # We can write the match_events out to a yml file
  >>> ins.write(prefix, output_type='match_events')
  # We can read these back in from the output file on initializing an INS using
  # the match_events_file keyword

(b) Flagging All Times for Highly Contaminated Channels:
********************************************************
::

  >>> from SSINS import MF

  # the N_thresh parameter must be set on initialization
  # If a channel has less than N_thresh clean samples remaining, all times will be flagged
  >>> mf = MF(ins.freq_array, sig_thresh=5, N_samp_thresh=20)

  # One must simply set the apply_N_thresh keyword for the apply_match_test() method
  >>> mf.apply_match_test(ins, apply_samp_thresh=True)
