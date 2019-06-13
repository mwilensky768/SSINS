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
     from SSINS import SS

     # The SS object is a subclass of a UVData object, and therefore has all of its attributes and methods
     # We initialize it identically
     # See UVData documentation on https://pyuvdata.readthedocs.io/en/latest/
     >>> ss = SS()

     # Read data by specifying a filepath as an argument to the read method
     >>> filepath = 'SSINS/data/1061313128_99bl_1pol_half_time.uvfits'
     # By default, the visibilities are differenced in time on read (see paper)
     >>> ss.read(filepath)
     # Setting diff=False saves the differencing for later (not useful in most situations)

  (b) Passing keyword arguments to SS.read
  ****************************************
  ::
    # SS.read is actually a small wrapper around UVData.read; they share keywords
    # In particular, select on read and reading only metadata function as usual (see UVData.select documentation)
    >>> ss = SS()
    >>> ss.read(filepath, read_data=False)

    # The following lines make use of the time_array attribute (metadata) to
    # read in all but the first and last integrations
    >>> times = np.unique(ss.time_array)[1:-1]
    >>> ss.read(filepath, read_data=True, times=times)

  (c) Applying flags
  ******************
  ::
    # SS.data_array is a numpy masked array. To "apply flags" is to change the mask of the data_array.
    # The proper way to apply flags to the sky-subtracted data is to use the apply_flags method
    # To apply the original flags in the raw data file, make the following call
    >>> ss.apply_flags(flag_choice='original')
    # Note that the original flags are always stored in the flag_array attribute
    # The flag_choice keyword is stored in an attribute
    >>> print(ss.flag_choice)
    'original'

    # You can apply flags from a custom flag array that is the same shape as the data
    >>> custom = np.zeros(ss.data_array.shape, dtype=bool)
    # Let us make it so that only the first frequency channel is flagged and nothing else
    >>> custom[:, 0, 0, :] = True
    # Apply these flags in the following way
    >>> ss.apply_flags(flag_choice='custom', custom=custom)
    >>> print(ss.flag_choice)
    'custom'

    # Unflag the data by setting flag_choice=None (note this is actually the default!!)
    >>> ss.apply_flags(flag_choice=None)
    # Check if anything is flagged, for demonstration purposes
    >>> print(np.any(ss.data_array.mask))
    False

  (c) Plotting using Catalog_Plot
  *******************************
