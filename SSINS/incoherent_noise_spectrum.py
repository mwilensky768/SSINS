from __future__ import absolute_import, division, print_function

"""
The incoherent noise spectrum class.
"""

import numpy as np
from scipy.special import erfcinv
import os
import warnings
import pickle
from hera_qm import UVFlag


class INS(UVFlag):
    """
    Defines the incoherent noise spectrum (INS) class.
    """

    def __init__(self, input, history='', label='', order=0):

        """
        init function for the INS class. Can set the attributes manually, or
        read some in using the read_paths dictionary. The keys for read_paths
        are the attribute names as strings, while the values are paths to
        numpy loadable binary files (pickle is used for masked arrays). The init
        function will calculate the Calculated Attributes (see below).

        Args:
            data: The data which will be assigned to the data attribute. (Required)
            Nbls: The number of baselines that went into each element of the
                  data array. (Required)
            freq_array: The frequencies (in hz) that describe the data, as found
                        in a UVData object. (Required)
            flag_choice: The flag choice used in the original SS object.
            vis_units: The units for the visibilities.
            obs: The obsid for the data.
            outpath: The base directory for data outputs.
            match_events: A list of events found by the filter in the MF class.
                          Usually not assigned initially.
            match_hists: Histograms describing the match_events.
                         Usually not assigned initially.
            chsq_events: Events found by the chisq_test in the MF class.
                         Usually not assigned initially.
            chisq_hists: Histograms describing the chisq events.
                         Usually not assigned initially.
            read_paths: A dictionary that can be used to read in a match filter,
                        rather than passing attributes to init or constructing
                        from an SS object. The keys are the attributes to be
                        passed in, while the values are paths to files that
                        contain the attribute data.
            samp_thresh_events: Events using the samp_thresh_test in the MF class.
                                Usually not assigned initially.
            order: The order of polynomial fit for each frequency channel when
                   calculating the mean-subtracted spectrum. Setting order=0
                   just calculates the mean in each frequency channel.
            coeff_write: An option to write out the coefficients of the polynomial
                         fit to each frequency channel.
        """

        super(INS, self).__init__(input, mode='metric', copy_flags=False,
                                  waterfall=False, history='', label='')
        if self.type is 'baseline':
            # Set the metric array to the data array without the spw axis
            self.metric_array = input.data_array
            self.weights_array = np.logical_not(input.data_array.mask)
            super(INS, self).to_waterfall(method='mean')

        self.order = order
        self.metric_ms = self.mean_subtract()
        """The mean-subtracted data."""

    def mean_subtract(self, f=slice(None)):

        """
        A function which calculated the mean-subtracted spectrum from the
        regular spectrum. A spectrum made from a perfectly clean observation
        will be standardized (written as a z-score) by this operation.

        Args:
            f: The frequency slice over which to do the calculation. Usually not
               set by the user.

        Returns:
            MS (masked array): The mean-subtracted data array.
        """

        # This constant is determined by the Rayleigh distribution, which
        # describes the ratio of its rms to its mean
        C = 4 / np.pi - 1
        if not self.order:
            MS = (self.metric_array[:, f] / self.metric_array[:, f].mean(axis=0) - 1) * np.sqrt(self.weights_array[:, f] / C)
        else:
            MS = np.zeros_like(self.metric_array[:, f])
            # Make sure x is not zero so that np.polyfit can proceed without nans
            x = np.arange(1, self.metric_array.shape[0] + 1)
            for i in range(self.metric_array.shape[-1]):
                y = self.metric_array[:, f, i]
                # Only iterate over channels that are not fully masked
                good_chans = np.where(np.logical_not(np.all(y.mask, axis=0)))[0]
                # Create a subarray mask of only those good channels
                good_data = y[:, good_chans]
                # Find the set of unique masks so that we can iterate over only those
                unique_masks, mask_inv = np.unique(good_data.mask, axis=1,
                                                   return_inverse=True)
                for k in range(unique_masks.shape[1]):
                    # Channels which share a mask
                    chans = np.where(mask_inv == k)[0]
                    coeff = np.ma.polyfit(x, good_data[:, chans], self.order)
                    mu = np.sum([np.outer(x**(self.order - k), coeff[k]) for k in range(self.order + 1)],
                                axis=0)
                    MS[:, good_chans[chans], i] = (good_data[:, chans] / mu - 1) * np.sqrt(self.weights_array[:, f, i][:, good_chans[chans]] / C)

        return(MS)

    def write(self, filename, clobber=False, data_compression='lzf'):
        self.metric_array = self.metric_array.data
        super(INS, self).write(filename, clobber=clobber, data_compression=data_compression)
        self.metric_array = np.ma.masked_where(self.metric_ms.mask, self.metric_array)
