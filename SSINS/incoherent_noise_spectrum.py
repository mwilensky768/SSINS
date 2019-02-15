from __future__ import absolute_import, division, print_function

"""
The incoherent noise spectrum class.
"""

import numpy as np
from scipy.special import erfcinv
import os
import warnings
import pickle


class INS(object):
    """
    Defines the incoherent noise spectrum (INS) class.
    """

    def __init__(self, data=None, Nbls=None, freq_array=None, pols=None,
                 flag_choice=None, vis_units=None, obs=None, outpath=None,
                 read_paths={}, order=0, coeff_write=False):

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
            pols: The polarizations present in the data, in the order of the data array.
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

        opt_args = {'obs': obs, 'pols': pols, 'vis_units': vis_units,
                    'outpath': outpath, 'flag_choice': flag_choice}
        for attr in ['obs', 'outpath']:
            if opt_args[attr] is None:
                warnings.warn('%s%s' % ('In order to save outputs and use Catalog.py,',
                                        'please supply %s attribute' % (attr)))
            else:
                setattr(self, attr, opt_args[attr])

        if flag_choice is None:
            warnings.warn('%s%s%s%s' % ('flag_choice is set to None. If this ',
                                        'does not reflect the flag_choice of ',
                                        'the original data, then saved arrays ',
                                        'will be mislabled'))
        self.flag_choice = flag_choice
        """The flag choice for the original SS object."""

        if not read_paths:
            args = (data, Nbls, freq_array)
            assert all([arg is not None for arg in args]),\
                '%s%s%s' % ('Insufficient data given. You must supply a data',
                            ' array, a Nbls array of matching shape, and a freq',
                            '_array of matching sub-shape')
            self.data = data
            """The sky-subtracted visibilities averaged over the baselines.
               Axes are (time, spw, freq, pol)."""
            if not len(self.data.mask.shape):
                self.data.mask = np.zeros(self.data.shape, dtype=bool)
            self.Nbls = Nbls
            """The number of baselines that went into each element of the data array."""
            self.freq_array = freq_array
            """The frequencies (in hz) describing the data."""

            for attr in ['pols', 'vis_units']:
                if opt_args[attr] is None:
                    warnings.warn('%s%s%s' % ('In order to use Catalog_Plot.py',
                                              ' with appropriate labels, please',
                                              ' supply %s attribute' % (attr)))
                else:
                    setattr(self, attr, opt_args[attr])

        else:
            self._read(read_paths)

        self.data_ms = self.mean_subtract(order=order, coeff_write=coeff_write)
        """The mean-subtracted data."""
        self.counts, self.bins, self.sig_thresh = self.hist_make()
        """Histogram data for the mean-subtracted data array."""

    def mean_subtract(self, f=slice(None), order=0, coeff_write=False):

        """
        A function which calculated the mean-subtracted spectrum from the
        regular spectrum. A spectrum made from a perfectly clean observation
        will be standardized (written as a z-score) by this operation.

        Args:
            f: The frequency slice over which to do the calculation. Usually not
               set by the user.
            order: The order of the polynomial fit for each frequency channel, by LLSE.
                   Setting order=0 just calculates the mean.
            coeff_write: Option to write out the polynomial fit coefficients for
                         each frequency channel when this function is run.

        Returns:
            MS (masked array): The mean-subtracted data array.
        """

        # This constant is determined by the Rayleigh distribution, which
        # describes the ratio of its rms to its mean
        C = 4 / np.pi - 1
        if not order:
            MS = (self.data[:, :, f] / self.data[:, :, f].mean(axis=0) - 1) * np.sqrt(self.Nbls[:, :, f] / C)
        else:
            MS = np.zeros_like(self.data[:, :, f])
            # Make sure x is not zero so that np.polyfit can proceed without nans
            x = np.arange(1, self.data.shape[0] + 1)
            for i in range(self.data.shape[-1]):
                y = self.data[:, 0, f, i]
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
                    coeff = np.ma.polyfit(x, good_data[:, chans], order)
                    if coeff_write:
                        with open('%s/%s_ms_poly_coeff_order_%i_%s.npy' %
                                  (self.outpath, self.obs, order, self.pols[i]), 'wb') as file:
                            pickle.dump(coeff, file)
                    mu = np.sum([np.outer(x**(order - k), coeff[k]) for k in range(order + 1)],
                                axis=0)
                    MS[:, 0, good_chans[chans], i] = (good_data[:, chans] / mu - 1) * np.sqrt(self.Nbls[:, 0, f, i][:, good_chans[chans]] / C)

        return(MS)

    def hist_make(self, sig_thresh=None, event=None):

        """
        A function which will make histograms of the mean-subtracted data.

        Args:
            sig_thresh: The significance threshold within which to make the
                        primary bins. Bins are of unit width. Data outside the
                        sig_thresh will be placed in a single outlier bin.
                        Will calculate a reasonable one by default.
            event: Used to histogram a single shape or frequency channel.
                   Providing an event as found in the INS.match_events list
                   will histogram the data corresponding to the shape for that
                   event, where the data is averaged over that shape before
                   histogramming.

        Returns:
            counts (array): The counts in each bin.
            bins (array): The bin edges.
            sig_thresh (float): The sig_thresh parameter. Will be the calculated value
                                if sig_thresh is None, else it will be what sig_thresh
                                was set to.
        """

        if sig_thresh is None:
            sig_thresh = np.sqrt(2) * erfcinv(1. / np.prod(self.data.shape))
        bins = np.linspace(-sig_thresh, sig_thresh,
                           num=int(2 * np.ceil(2 * sig_thresh)))
        if event is None:
            dat = self.data_ms
        else:
            N = np.count_nonzero(np.logical_not(self.data_ms.mask[:, 0, event[2]]), axis=1)
            dat = self.data_ms[:, 0, event[2]].mean(axis=1) * np.sqrt(N)
        if dat.min() < -sig_thresh:
            bins = np.insert(bins, 0, dat.min())
        if dat.max() > sig_thresh:
            bins = np.append(bins, dat.max())
        counts, _ = np.histogram(dat[np.logical_not(dat.mask)],
                                 bins=bins)

        return(counts, bins, sig_thresh)

    def save(self, sig_thresh=None):
        """
        Writes out relevant data products.

        Args:
            sig_thresh: Can give a little sig_thresh tag at the end of the
                        filename if desired. (Technically this does not have
                        to be an integer, so you can tag it however you want.)
        """
        tags = ['match', 'chisq', 'samp_thresh']
        tag = ''
        if sig_thresh is not None:
            tag += '_%s' % sig_thresh
        for subtag in tags:
            if len(getattr(self, '%s_events' % (subtag))):
                tag += '_%s' % subtag

        for string in ['arrs', 'metadata']:
            if not os.path.exists('%s/%s' % (self.outpath, string)):
                os.makedirs('%s/%s' % (self.outpath, string))

        for attr in ['data', 'data_ms', 'Nbls']:
            with open('%s/arrs/%s_%s_INS_%s%s.npym' %
                      (self.outpath, self.obs, self.flag_choice, attr, tag), 'wb') as f:
                pickle.dump(getattr(self, attr), f)
        for attr in ['counts', 'bins']:
            np.save('%s/arrs/%s_%s_INS_%s%s.npy' %
                    (self.outpath, self.obs, self.flag_choice, attr, tag),
                    getattr(self, attr))

        for attr in ['freq_array', 'pols', 'vis_units']:
            if hasattr(self, attr):
                np.save('%s/metadata/%s_%s.npy' % (self.outpath, self.obs, attr),
                        getattr(self, attr))

    def _read(self, read_paths):
        """
        Reads in attributes from numpy loadable (or depicklable) files.

        Args:
            read_paths: A dictionary whose keys are the attributes to be read in
                        and whose values are paths to files where the attribute
                        data is written.
        """

        for arg in ['data', 'Nbls', 'freq_array']:
            assert arg in read_paths,\
                'You must supply a path to a numpy loadable %s file for read_paths entry' % (arg)
            setattr(self, arg, np.load(read_paths[arg]))
        if not len(self.data.mask.shape):
            data.mask = np.zeros(self.data.shape, dtype=bool)
        for attr in ['pols', 'vis_units']:
            if attr not in read_paths:
                warnings.warn('In order to use Catalog_Plot.py, please supply\
                               path to numpy loadable %s attribute for read_paths entry' % (attr))
            else:
                setattr(self, attr, np.load(read_paths[attr]))
