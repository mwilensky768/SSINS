from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.stats
import os
import warnings


class VDH:

    def __init__(self, data=None, flag_choice=None, freq_array=None, pols=None,
                 vis_units=None, obs=None, outpath=None, bins='auto', fit=True,
                 read_paths={}):

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
            if fit:
                self.MLEs, self.fits, self.errors = self.rayleigh_mixture_fit(data)
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
                np.ma.dump(getattr(self, attr), '%s/arrs/%s_%s_VDH_%s.npym' %
                           (self.outpath, self.obs, self.flag_choice, attr))

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

    def hist_make(self, data, bins='auto'):
        counts = np.zeros([data.shape[2], 1 + bool(self.flag_choice)], dtype=object)
        bins_arr = np.copy(counts)
        for spw in range(data.shape[2]):
            for i in range(1 + bool(self.flag_choice)):
                if i:
                    temp_counts, temp_bins = np.histogram(data[:, :, spw][np.logical_not(data[:, :, spw].mask)], bins=bins)
                else:
                    temp_counts, temp_bins = np.histogram(data[:, :, spw], bins=bins)
                counts[spw, i] = temp_counts
                bins_arr[spw, i] = temp_bins
        return(counts, bins_arr)

    def rayleigh_mixture_fit(self, data):
        MLEs = []
        fits = np.zeros([data.shape[2], 1 + bool(self.flag_choice)], dtype=object)
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
            for spw in range(data.shape[2]):
                P = np.zeros(len(self.bins[spw, i]) - 1)
                Ntot = np.sum(N[:, spw])
                for mle, n in zip(MLE[:, spw].flatten(), N[:, spw].flatten()):
                    P += n / Ntot * (scipy.stats.rayleigh.cdf(self.bins[spw, i][1:], scale=np.sqrt(mle)) -
                                     scipy.stats.rayleigh.cdf(self.bins[spw, i][:-1], scale=np.sqrt(mle)))
                fit = Ntot * P
                error = np.sqrt(Ntot * P * (1 - P))
                fits[spw, i] = fit
                errors[spw, i] = error
        MLEs = np.array(MLEs)
        return(MLEs, fits, errors)

    def rev_ind(data, window):
        self.W_hist = []
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
