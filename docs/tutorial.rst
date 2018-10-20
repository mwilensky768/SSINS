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

(a) Initializing from a UVData object
*************************************
::

  >>> from pyuvdata import UVData
  >>> from SSINS import SS
  >>> UV = UVData()

  # This is an MWA uvfits file trimmed to a single pol, 99 baselines, and only
  # half of the observing time. We will read in only cross correlations, which
  # is preferred.
  >>> UV.read('SSINS/data/1061313128_99bl_1pol_half_time.uvfits',
  ...         file_type='uvfits', ant_str='cross')

  # Now that we have a UVData object, we can initialize the SS class.
  # We will also explicitly tell it to reshape the UVData object's data array
  # and do the sky subtraction.
  >>> ss = SS(obs='1061313128_99bl_1pol_half_time',
  ...         outpath='SSINS/data/test_outputs' UV=UV, diff=True)

  # Now the SS instance has its own UVData object that it carries around.
  # Note the following.
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

  # We set up a keyword dictionary for the select on read functionality.
  # This time we will only keep the first 16 frequency channels.
  >>> read_kwargs = {'ant_str': 'cross', 'freq_chans': np.arange(16)}

  # We will also remove some known problematic times, by index.
  >>> bad_time_indices = [0, -1, -2, -3]
  >>> ss = SS(obs=obs, outpath=outpath, inpath=inpath, read_kwargs=read_kwargs,
  ...         bad_time_indices=bad_time_indices)

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
  ...         flag_choice='original')
  >>> np.all(ss.UV.data_array.mask == ss.UV.flag_array)
  True

  # There are multiple ways to unflag the data, but one way is to use the
  # apply_flags() method.

  >>> ss.apply_flags(choice=None)
  >>> np.any(ss.UV.data_array.mask)
  False

  # Custom flag masks can also be applied by either manually manipulating the mask
  # or on read by supplying a custom flag array.

  >>> custom = np.zeros_like(ss.UV.flag_array)
  >>> custom[:, :, 0, 0, :] = 1
  >>> ss.apply_flags(choice='custom', custom=custom)

  # This flags everything in the zeroth frequency channel.
  # apply_flags() is called on initialization, so any flag manipulation can be
  # done on read or later as is fit for the situation

sky_subtract: Data Products
---------------------------
Forming and plotting data products using the sky_subtract class. Options are
incoherent_noise_spectrum (INS), vis_diff_hist (VDH), event_stat (ES),
match_filter (MF).

(a) Forming and Plotting Data Products
**************************************
::

  >>> from SSINS import Catalog_Plot as cp

  # The INS_prepare() method attaches an INS instance to the SS instance
  >>> ss.INS_prepare()
  flag_choice is set to None. If this does not reflect the flag_choice of the original data, then saved arrays will be mislabled
  # This issues a warning about the flag_choice attribute, which defaults to None

  # Similarly a Visibility Difference Histogram (VDH) instance can be formed with
  >>> ss.VDH_prepare()

  # We can save relevant data and metadata to ss.outpath with the following
  >>> ss.INS.save()
  >>> ss.VDH.save()

  # Useful plots can be made using the Catalog_Plot module
  # They are saved to ss.outpath
  >>> cp.INS_plot(ss.INS, ms_vmax=5, ms_vmin=-5)
  >>> cp.VDH_plot(ss.VDH, xscale='linear')

---
INS
---

incoherent_noise_spectrum: Reading From Saved Data
--------------------------------------------------
If data and metadata are saved they can be read back in using the read_paths
keyword. This dictionary can be set manually, but also one can be set up using a
function in util if they are saved in the same manner as is done by INS.save()

(a) Using util
**************
::

  >>> import util
  >>> from SSINS import INS
  >>> basedir = SSINS/data
  >>> obs = '1061313128_99bl_1pol_half_time'
  >>> outpath = '%s/test_outputs' % basedir

  # This function works for multiple data products, so we specify the product in
  # the function call, along with other important metadata
  >>> read_paths = util.read_paths_construct(basedir, 'original', obs, 'INS')

  # This makes a dictionary which is used as follows
  >>> ins = INS(obs=obs, outpath=outpath, read_paths=read_paths,
                flag_choice='original')

  # If events are caught by a filter, then there will be a tag on the filename
  # This tag needs to be specified to the util function
  >>> read_paths = util.read_paths_construct(basedir, None, obs, 'INS',
                                             tag='match')
  >>> ins2 = INS(read_paths=read_paths, obs=obs, outpath=outpath)

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

  # By default, this plots 2-d colormaps of the INS.data and INS.data_ms,
  # Using INS.freq_array to determine the ticklabels, and saving to INS.outpath
  >>> ins.outpath = 'SSINS/data/figs/default'
  >>> cp.INS_plot(ins)

  # Other typical matplotlib settings can be chosen, such as the colormap or the
  # bounds of the colorbar
  >>> ins.outpath = 'SSINS/data/figs/cmap_cbar'
  >>> cp.INS_plot(ins, data_cmap=cm.plasma, vmin=0, vmax=150, ms_vmin=-5, ms_vmax=5)

(b) Using plot_lib
******************
::

  # Finer control over which plots come out can be obtained without the
  # Catalog_Plot wrapper using just plot_lib
  >>> from SSINS import plot_lib
  >>> from matplotlib import cm
  >>> import matplotlib.pyplot as plt

  >>> fig, ax = plt.subplots(nrows=2)
  >>> ins.outpath = 'SSINS/data/figs/order_compare'

  # Here we take an INS and plot its mean-subtracted data in the first
  # polarization with different order parameters

  >>> for i in range(2):
  ...     ins.mean_subtract(order=i)
  ...     plot_lib.image_plot(fig, ax[i], ins.data_ms[:, 0, :, 0],
  ...                         cmap=cm.coolwarm, freq_array=ins.freq_array[0],
  ...                         title='order = %i' % i, vmin=-5, vmax=5)
  >>> fig.savefig('%s/%s_order_compare.png' % (ins.outpath, ins.obs))

  # This particular example is useful when the overall noise level appears to be
  # drifting over the course of the observation and you want to ignore that drift

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
  >>> ins.data_ms = ins.mean_subtract(order=2)

  # That made a quadratic fit for each channel
  # This can also be done on initialization in the same way
  >>> ins = INS(obs='obsid', outpath='test_outpath', read_paths=read_paths,
                order=1)

  # That made a linear fit

--
MF
--

match_filter: initialization
----------------------------

(a) Initializing from Scratch
*****************************
::

  >>> from SSINS import MF

  # Initialization involves setting desired parameters (reasonable defaults exist)
  # RFI shapes are passed with a dictionary (this example is digital TV in South Africa, where HERA is located)
  >>> shape_dict = {'TV4': [1.74e8, 1.82e8],
                    'TV5': [1.82e8, 1.9e8],
                    'TV6': [1.9e8, 1.98e8]}

  # sig_thresh governs the maximal strength of outlier to leave unflagged
  # The default is estimated from the size of the data
  >>> sig_thresh = 5

  # The single-frequency and broadband streak flaggers can be turned off (default on)
  >>> point = False
  >>> streak = False

  # An INS() instance to operate on is required
  >>> mf = MF(ins, shape_dict=shape_dict, sig_thresh=sig_thresh, point=point,
              streak=streak)

(b) Initializing from a SS() instance
*************************************
::

  # As explained above, the SS class has thin wrappers around the useful classes in this module
  # simply pass the keywords to the method as you would to the initialization
  >>> ss.MF_prepare(sig_thresh=5, shape_dict={'TV4': [1.74e8, 1.82e8],
                                              'TV5': [1.82e8, 1.9e8],
                                              'TV6': [1.9e8, 1.98e8]})

  # If the SS() does not have an INS() attached, it will construct one and pass
  # it to the MF()

match_filter: Applying Tests
----------------------------

(a) Basic Match-Shape Test:
***************************
::

  >>> from SSINS import Catalog_Plot as cp

  # Here, the shapes in the shape_dictionary are tested for
  # This method will automatically apply flags to samples which match the flagging criterion
  >>> mf.apply_match_test()

  # There is a method in Catalog_Plot for plotting MF results
  >>> cp.MF_plot(mf, ms_vmin=-mf.sig_thresh, ms_vmax=mf.sig_thresh)

(b) Flagging All Times for Highly Contaminated Channels:
********************************************************
::

  >>> from SSINS import MF

  # the N_thresh parameter must be set on initialization
  # If a channel has less than N_thresh clean samples remaining, all times will be flagged
  >>> mf = MF(ins, N_thresh=20)

  # One must simply set the apply_N_thresh keyword for the apply_match_test() method
  >>> mf.apply_match_test(apply_N_thresh=True)

(c) Flagging with Polynomial Fitting:
*************************************
::

  >>> from SSINS import Catalog_Plot as cp
  >>> order = 1

  # Initialize the ins of the mf with the desired order
  >>> mf.ins.data_ms = mf.ins.mean_subtract(order=order)

  # Apply flagging at that order
  >>> mf.apply_match_test(order=order)

  # RFI shapes can be smeared at fits of nonzero order
  # This is generally unproblematic for the filter, but looking at the results
  # at different orders can be more or less informative depending on the observation
  >>> mf.ins.data_ms = mf.ins.mf.ins.mean_subtract(order=0)
  >>> cp.MF_plot(mf, ms_vmin=-mf.sig_thresh, ms_vmax=mf.sig_thresh)
