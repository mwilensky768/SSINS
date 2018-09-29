"""
The visibility difference histogram class.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.stats
import os
import warnings
import pickle
import time


class VDH(object):

    """
    Defines the VDH class. This class just contains data relevant to a histogram
    of the sky-subtracted visibility (visibility difference) amplitudes.
    """

    def __init__(self, data=None, flag_choice=None, freq_array=None, pols=None,
                 vis_units=None, obs=None, outpath=None, bins=None, fit_hist=False,
                 MLE=True, read_paths={}):

        """
        init function for the VDH class. This grabs all the relevant metadata
        that was passed, makes a histogram of the data and adds fits if desired.
        The fit is made using maximum likelihood estimation for each baseline,
        frequency, and polarization.

        Keywords: data: The data to be histogrammed. Can be circumvented if
                        other attributes are being read in.

                  flag_choice: The flagging choice for the data. The histogram
                               for no flags is always computed. If flag_choice
                               is not None, then a second histogram will be
                               computed with the flags applied.

                  freq_array: The frequency array which describes axes 2 and 3
                              of the data.

                  pols: The polarizations present in the data

                  vis_units: The units for the visibilities

                  obs: The OBSID for the data

                  outpath: The base directory for saving outputs

                  bins: The bin edges for the histograms. If None, will use
                        logarithmically spaced bins. If 'auto,' then the same
                        thing happens as when passing 'auto' to np.histogram.

                  fit_hist: Specifies whether or not to calculate a rayleigh
                            mixture fit to the histograms via maximum likelihood
                            estimation.

                  MLE: Specify whether to calculate an MLE for each baseline,
                       frequency, and polarization. This must be enabled in
                       order to calculate a fit.

                 read_paths: One can read in saved histograms using this keyword.
                             Set the keys to the keywords to read in, and set
                             the values as paths to the attributes to be read in.

        Attributes: counts: The number of data points which lie in the bins
                            whose edges are described by the corresponding entry
                            in bins.

                    bins: Will be an array whose entries are the bin edges for
                          the data without and with flags applied, respectively.

                    fits: Rayleigh-mixture expected counts for the bins in bins,
                          found by maximum likelihood estimation.

                    errors: The 1-sigma error bars for the fits.

                    MLEs: The estimators for those corresponding entries in
                          the fits.

                    W_hist: A waterfall histogram may be calculated with
                            rev_ind(data, window) which finds the number of
                            baselines in a time/frequency/polarization which
                            have amplitudes within the window given. Useful for
                            seeing where bright RFI is spectrally located.
        """

        self.flag_choice = flag_choice

        opt_args = {'freq_array': freq_array, 'pols': pols,
                    'vis_units': vis_units, 'obs': obs, 'outpath': outpath}

        for attr in ['obs', 'outpath']:
            if opt_args[attr] is None:
                warnings.warn('In order to save outputs, and use SSINS.Catalog_Plot,\
                               please supply a value other than None for %s keyword' % (attr))
            else:
                setattr(self, attr, opt_args[attr])

        if data is not None:
            for attr in ['freq_array', 'pols', 'vis_units']:
                if opt_args[attr] is None:
                    warnings.warn('In order to use SSINS.Catalog_Plot, please supply\
                                   a value other than None for %s keyword' % (attr))
                else:
                    setattr(self, attr, opt_args[attr])
            self.counts, self.bins = self.hist_make(data, bins=bins)
            if MLE:
                self.MLEs, self.fits, self.errors = self.rayleigh_mixture_fit(data, fit_hist=fit_hist)
        else:
            self.read(read_paths)

        for string in ['arrs', 'figs']:
            if not os.path.exists('%s/%s' % (self.outpath, string)):
                os.makedirs('%s/%s' % (self.outpath, string))

    def save(self):
        """
        Saves metadata and data products to the outpath
        """

        for string in ['arrs', 'metadata']:
            if not os.path.exists('%s/%s' % (self.outpath, string)):
                os.makedirs('%s/%s' % (self.outpath, string))

        for attr in ['counts', 'bins', 'fits', 'errors']:
            if hasattr(self, attr):
                np.save('%s/arrs/%s_%s_VDH_%s.npy' %
                        (self.outpath, self.obs, self.flag_choice, attr),
                        getattr(self, attr))

        for attr in ['MLEs', 'W_hist']:
            if hasattr(self, attr):
                with open('%s/arrs/%s_%s_VDH_%s.npym' %
                          (self.outpath, self.obs, self.flag_choice, attr), 'wb') as f:
                    pickle.dump(getattr(self, attr), f)

        for attr in ['freq_array', 'pols', 'vis_units']:
            if hasattr(self, attr):
                np.save('%s/metadata/%s_%s.npy' %
                        (self.outpath, self.obs, attr),
                        getattr(self, attr))

    def read(self, read_paths):
        """
        Reads in metadata and data products using read_paths
        """

        for attr in ['freq_array', 'pols', 'vis_units']:
            if attr in read_paths and read_paths[attr] is not None:
                setattr(self, attr, np.load(read_paths[attr]))
            else:
                warnings.warn('In order to use SSINS.Catalog_Plot, please supply\
                               path to numpy loadable file for %s read_paths entry' % attr)
        for attr in ['counts', 'bins']:
            assert attr in read_paths and read_paths[attr] is not None, \
                'You must supply a path to a numpy loadable file for %s read_paths entry' % attr
            setattr(self, attr, np.load(read_paths[attr]))
        for attr in ['fits', 'errors']:
            if attr in read_paths and read_paths[attr] is not None:
                setattr(self, attr, np.load(read_paths[attr]))
        for attr in ['MLEs', 'W_hist']:
            if attr in read_paths and attr is not None:
                with open(read_paths[attr], 'rb') as f:
                    setattr(self, attr, pickle.load(f))

    def hist_make(self, data, bins=None):
        """
        Makes a histogram given the data, flag_choice, and bins choice.
        """
        if bins is None:
            bins = np.logspace(np.floor(np.log10(np.amin(data.data[data.data > 0]))),
                               np.ceil(np.log10(np.amax(data.data))),
                               num=1001)
        counts = np.zeros(1 + bool(self.flag_choice), dtype=object)
        bins_arr = np.copy(counts)
        for i in range(1 + bool(self.flag_choice)):
            if i:
                temp_counts, temp_bins = np.histogram(data[np.logical_not(data.mask)],
                                                      bins=bins)
            else:
                temp_counts, temp_bins = np.histogram(data, bins=bins)
            counts[i] = temp_counts
            bins_arr[i] = temp_bins
        return(counts, bins_arr)

    def rayleigh_mixture_fit(self, data, fit_hist=False):
        """
        Makes a rayleigh-mixture fit via maximum likelihood estimation.
        """
        print('Beginning fit at %s' % time.strftime("%H:%M:%S"))
        MLEs = []
        fits = np.zeros(1 + bool(self.flag_choice), dtype=object)
        errors = np.copy(fits)
        for i in range(1 + bool(self.flag_choice)):
            if i:
                MLE = 0.5 * np.mean(data**2, axis=0)
                N = np.count_nonzero(np.logical_not(data.mask), axis=0)
            else:
                # Just use the data without the mask
                MLE = 0.5 * np.mean(data.data**2, axis=0)
                N = np.count_nonzero(data.data, axis=0)
            MLEs.append(MLE)
            P = np.zeros(len(self.bins[i]) - 1)
            Ntot = np.sum(N)
            if fit_hist:
                for mle, n in zip(MLE.flatten(), N.flatten()):
                    P += n / Ntot * (np.exp(-self.bins[i][:-1]**2 / (2 * mle)) -
                                     np.exp(-self.bins[i][1:]**2 / (2 * mle)))
                fit = Ntot * P
                error = np.sqrt(Ntot * P * (1 - P))
            else:
                fit = None
                error = None
            fits[i] = fit
            errors[i] = error
        MLEs = np.array(MLEs)
        print('Done with fit at %s' % time.strftime("%H:%M:%S"))
        return(MLEs, fits, errors)

    def rev_ind(self, data, window):
        """
        Finds the number of baselines in a time-frequency-polarization bin which
        had data whose brightness is within the window. Useful for finding the
        spectral location of bright RFI.
        """
        self.W_hist = []
        self.window = window
        for i in range(1 + bool(self.flag_choice)):
            W = np.zeros(data.shape)
            if i:
                dat = data
            else:
                # Used the masked array's data without its mask
                dat = data.data
            W[np.logical_and(min(window) < dat, dat < max(window))] = 1
            self.W_hist.append(W.sum(axis=1))
        self.W_hist = np.array(self.W_hist)
