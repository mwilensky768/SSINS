from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.stats
import os
import warnings
import pickle
import time


class VDH:

    def __init__(self, data=None, flag_choice=None, freq_array=None, pols=None,
                 vis_units=None, obs=None, outpath=None, bins=None, fit_hist=False,
                 MLE=True, read_paths={}):

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
        for attr in ['fits', 'errors', 'MLEs']:
            if attr in read_paths and read_paths[attr] is not None:
                setattr(self, attr, np.load(read_paths[attr]))

    def hist_make(self, data, bins=None):
        if bins is None:
            bins = np.logspace(np.floor(np.log10(np.amin(data.data))),
                               np.ceil(np.log10(np.amax(data.data))),
                               num=1001)
        counts = np.zeros(1 + bool(self.flag_choice), dtype=object)
        bins_arr = np.copy(counts)
        for i in range(1 + bool(self.flag_choice)):
            if i:
                temp_counts, temp_bins = np.histogram(data[:, :, 0][np.logical_not(data[:, :, 0].mask)],
                                                      bins=bins)
            else:
                temp_counts, temp_bins = np.histogram(data[:, :, 0], bins=bins)
            counts[i] = temp_counts
            bins_arr[i] = temp_bins
        return(counts, bins_arr)

    def rayleigh_mixture_fit(self, data, fit_hist=False):
        print('Beginning fit at %s' % time.strftime("%H:%M:%S"))
        MLEs = []
        fits = np.zeros(1 + bool(self.flag_choice), dtype=object)
        errors = np.copy(fits)
        for i in range(1 + bool(self.flag_choice)):
            if i:
                MLE = 0.5 * np.mean(data**2, axis=0)
                N = np.count_nonzero(np.logical_not(data.mask), axis=0)
            else:
                # copy does not copy the mask
                dat = np.copy(data)
                MLE = 0.5 * np.mean(dat**2, axis=0)
                N = np.count_nonzero(dat, axis=0)
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
        self.W_hist = []
        self.window = window
        for i in range(1 + bool(self.flag_choice)):
            W = np.zeros(data.shape)
            if i:
                dat = data
            else:
                # Copying the array does not copy the mask
                dat = np.copy(data)
            W[np.logical_and(min(window) < dat, dat < max(window))] = 1
            self.W_hist.append(W.sum(axis=1))
        self.W_hist = np.array(self.W_hist)
