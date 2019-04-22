"""
The sky_subtract (SS) class is defined here. This is the backbone of the analysis
pipeline when working with raw datafiles. It is a subclass of UVData. See UVData
documentation for attributes that are not listed here.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
from pyuvdata import UVData
import os
from SSINS import util, INS, MF
import scipy.stats
import warnings
import time


class SS(UVData):

    """
    Defines the SS class.
    """

    def __init__(self):

        """
        Initializes identically to a UVData object, except for one additional attribute.
        """
        super(SS, self).__init__()
        self.MLE = None
        """Array of length Nfreqs that stores maximum likelihood estimators for
        each frequency, calculated using the MLE_calc method"""

    def read(self, filename, diff=True, flag_choice=None, INS=None, custom=None,
             **kwargs):

        """
        Reads in a file that is compatible with UVData object by first calling
        UVData.read(). See UVData documentation for list of kwargs that can be
        passed to UVData.read()

        Args:
            filename (str): The filepath to read in.
            diff (bool): If True, and data was read in, then difference the visibilities in time
            flag_choice: Sets flags for the data array on read using apply_flags method.
            INS: An INS object for apply_flags()
            custom: A custom flag array for apply_flags()
            kwargs: Additional kwargs are passed to UVData.read()
        """

        super(SS, self).read(filename, **kwargs)
        if (self.data_array is not None) and diff:
            self.diff()
            self.apply_flags(flag_choice=flag_choice)

    def apply_flags(self, flag_choice=None, INS=None, custom=None):
        """
        A function which applies flags to the data via numpy masked arrays. Also
        changes the SS.flag_choice attribute.

        Args:
            flag_choice (None, 'original', 'INS', 'custom'):
                Applies flags according to the choice. None unflags the data.
                'original' applies flags based on the flag_array attribute.
                'INS' applies flags from an INS object specified by the INS keyword.
                'custom' applies a custom flag array specified by the custom keyword
                - it must be the same shape as the data.
            INS: An INS from which to apply flags - only used if flag_choice='INS'
            custom: A custom flag array from which to apply flags - only used if flag_choice='custom'
        """
        self.flag_choice = flag_choice
        self.MLE = None
        if flag_choice is 'original':
            self.data_array.mask = np.copy(self.flag_array)
        elif flag_choice is 'INS':
            self.data_array.mask[:] = False
            ind = np.where(INS.metric_array.mask)
            for i in range(len(ind[0])):
                self.data_array[ind[0][i] * self.Nbls:(ind[0][i] + 1) * self.Nbls,
                                :, ind[1][i], ind[2][i]] = np.ma.masked
        elif flag_choice is 'custom':
            self.data_array.mask[:] = False
            if custom is not None:
                self.data_array[custom] = np.ma.masked
            else:
                warnings.warn('Custom flags were chosen, but custom flags were None type. Setting flags to None.')
                self.flag_choice = None
        elif flag_choice is None:
            self.data_array.mask = np.zeros(self.data_array.shape, dtype=bool)
        else:
            raise ValueError('flag_choice of %s is unacceptable, aborting.' % flag_choice)

    def diff(self):

        """
        Differences the visibilities in time. Only supported if all baselines
        have the same integration time, all baselines report at each time, and
        the baseline-time axis is in the same baseline ordering at each integration.
        In other words, this is not yet functional with baseline dependent averaging.
        The flags are propagated by taking the boolean OR of the entries that correspond
        to the visibilities that are differenced from one another. Other metadata
        attributes are also adjusted so that the resulting SS object passes
        UVData.check()
        """

        assert self.Nblts == self.Nbls * self.Ntimes, 'Nblts != Nbls * Ntimes'
        cond = np.all([self.baseline_array[:self.Nbls] == self.baseline_array[k * self.Nbls:(k + 1) * self.Nbls]
                       for k in range(1, self.Ntimes - 1)])
        assert cond, 'Baseline array slices do not match in each time! The baselines are out of order.'

        # Difference in time and OR the flags
        self.data_array = np.ma.masked_array(self.data_array[self.Nbls:] - self.data_array[:-self.Nbls])
        """The time-differenced visibilities. Complex array of shape (Nblts, Nspws, Nfreqs, Npols)."""
        self.flag_array = np.logical_or(self.flag_array[self.Nbls:], self.flag_array[:-self.Nbls])
        """The flag array, which results from boolean OR of the flags corresponding to visibilities that are differenced from one another."""

        # Adjust the UVData attributes.
        self.Nblts -= self.Nbls
        """Number of baseline-times. For now, this must be equal to the number of baselines times the number of times."""
        self.ant_1_array = self.ant_1_array[:-self.Nbls]
        self.ant_2_array = self.ant_2_array[:-self.Nbls]
        self.baseline_array = self.baseline_array[:-self.Nbls]
        self.integration_time = self.integration_time[self.Nbls:] + self.integration_time[:-self.Nbls]
        """Total amount of integration time (sum of the differenced visibilities) at each baseline-time (length Nblts)"""
        self.Ntimes -= 1
        """Total number of integration times in the data. Equal to the original Ntimes-1."""
        self.nsample_array = 0.5 * (self.nsample_array[self.Nbls:] + self.nsample_array[:-self.Nbls])
        """See pyuvdata documentation. Here we average the nsample_array of the visibilities that are differenced"""
        self.time_array = 0.5 * (self.time_array[self.Nbls:] + self.time_array[:-self.Nbls])
        """The center time of the differenced visibilities. Length Nblts."""
        self.uvw_array = 0.5 * (self.uvw_array[self.Nbls:] + self.uvw_array[:-self.Nbls])
        super(SS, self).set_lsts_from_time_array()

    def MLE_calc(self):

        """
        Calculates maximum likelihood estimators for Rayleigh fits at each
        frequency. Used for developing a mixture fit.
        """

        self.MLE = np.sqrt(0.5 * np.mean(np.absolute(self.data_array)**2, axis=(0, 1, -1)))

    def mixture_prob(self, bins):
        """
        Calculates the probabilities of landing in each bin for a given set of
        bins.

        Args:
            bins: The bin edges of the bins to calculate the probabilities for.
        Returns:
            prob: The probability to land in each bin based on the maximum likelihood model
        """

        if self.MLE is None:
            self.MLE_calc()
        if bins is 'auto':
            _, bins = np.histogram(np.abs(self.data_array[np.logical_not(self.data_array.mask)]))

        N_spec = np.sum(np.logical_not(self.data_array.mask), axis=(0, 1, -1))
        N_total = np.sum(N_spec)

        # Calculate the fraction belonging to each frequency
        chi_spec = N_spec / N_total

        # initialize the probability array
        prob = np.zeros(len(bins) - 1)
        # Calculate the mixture distribution
        # If this could be vectorized over frequency, that would be better.
        for chan in range(self.Nfreqs):
            if self.MLE[chan] > 0:
                quants = scipy.stats.rayleigh.cdf(bins, scale=self.MLE[chan])
                prob += chi_spec[chan] * (quants[1:] - quants[:-1])

        return(prob)

    def rev_ind(self, band):

        """
        Reverse indexes sky-subtracted visibilities whose amplitudes are within
        a band given by the band argument. Collapses along the baselines to
        return a time-frequency waterfall per polarization. For example, setting
        a band of [1e3, 1e4] reports the number of baselines at each
        time/frequency/polarization whose sky-subtracted visibility amplitude
        was between 1e3 and 1e4. Includes flags.

        Args:
            band: The minimum and maximum amplitudes to be sought
        Returns:
            rev_ind_hist:
                A time-frequency waterfall per polarization counting the number
                of baselines whose sky-subtracted visibility amplitude fell
                within the band argument.

        """

        where_band = np.logical_and(np.absolute(self.data_array) > min(band),
                                    np.absolute(self.data_array) < max(band))
        where_band_mask = np.logical_and(np.logical_not(self.data_array.mask),
                                         where_band)
        shape = [self.Ntimes, self.Nbls, self.Nfreqs, self.Npols]
        rev_ind_hist = np.sum(where_band_mask.reshape(shape), axis=1)
        return(rev_ind_hist)

    def write(self, filename_out, file_type_out, UV=None, filename_in=None,
              read_kwargs={}, combine=True, nsample_default=1, write_kwargs={}):

        """
        Lets one write out the flags to a new file. This requires extending the
        flags in time. The same convention is used as in INS.flags_to_mask().
        The rest of the data for writing the file is pulled from an existing
        UVData object passed using the UV keyword, or read in to a new UVData
        object using the filename_in keyword. Due to how the nsample_array
        and flag_array get combined into the weights when writing uvfits,
        areas where the nsample_array == 0 are set to nsample_default so that
        new flags can actually be propagated to those data in the new uvfits file.

        Args:
            filename_out: The name of the file to write to. *Required*
            file_type_out: The typle of file to write out. See pyuvdata documentation for options. *Required*
            UV: A UVData object whose data and metadata to use to write the file.
            filename_in: A file from which to read data in order to write the new file. Not used if UV is not None.
            read_kwargs: A keyword dictionary for the UVData read method if reading from a file. See pyuvdata documentation for read keywords.
            combine (bool): If True, combine the original flags with the new flags (OR them), else just use the new flags.
            nsample_default:
                Used for writing uvfits when elements of the nsample_array are 0.
                This is necessary due to the way the nsample_array and flag_array
                are combined into the weights when writing uvfits, otherwise
                flags do not actually get propagated to the new file where nsample_array is 0.
            write_kwargs: A keyword dictionary for the selected UVData write method. See pyuvdata documentation for write keywords.
        """

        if UV is None:
            UV = UVData()
            UV.read(filename_in, **read_kwargs)

        # Test that assumptions abouts blts axis are ok
        assert UV.Nblts == UV.Nbls * UV.Ntimes, 'Nblts != Nbls * Ntimes for UV object.'
        cond = np.all([UV.baseline_array[:UV.Nbls] == UV.baseline_array[k * UV.Nbls:(k + 1) * UV.Nbls]
                       for k in range(1, UV.Ntimes)])
        assert cond, 'Baseline array slices do not match in each time! The baselines are out of order.'

        # Check nsample_array for issue
        if np.any(UV.nsample_array == 0) and (file_type_out is 'uvfits'):
            warnings.warn("Writing uvfits file with some nsample == 0. This will"
                          " result in a failure to propagate flags. Changing "
                          " nsample value to nsample_default parameter (default is 1)")
            UV.nsample_array[UV.nsample_array == 0] = nsample_default

        # Option to keep old flags
        if not combine:
            UV.flag_array[:] = 0

        # Propagate the new flags
        for i in range(self.Ntimes):
            # This actually does not invert properly but I think it's the best way
            UV.flag_array[i * self.Nbls: (i + 1) * self.Nbls][self.data_array.mask[i * self.Nbls: (i + 1) * self.Nbls]] = 1
            UV.flag_array[(i + 1) * self.Nbls: (i + 2) * self.Nbls][self.data_array.mask[i * self.Nbls: (i + 1) * self.Nbls]] = 1

        # Write file
        getattr(UV, 'write_%s' % file_type_out)(filename_out, **write_kwargs)
