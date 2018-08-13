from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.special import erfcinv
import os
import warnings
import pickle


class INS:
    """
    This incoherent noise spectrum class, formed from an RFI object as an input.
    Outputs a data array
    """

    def __init__(self, data=None, Nbls=None, freq_array=None, pols=None,
                 flag_choice=None, vis_units=None, obs=None, outpath=None,
                 match_events=[], match_hists=[], chisq_events=[],
                 chisq_hists=[], read_paths={}, samp_thresh_events=[]):

        opt_args = {'obs': obs, 'pols': pols, 'vis_units': vis_units,
                    'outpath': outpath, 'flag_choice': flag_choice}
        for attr in ['obs', 'outpath']:
            if opt_args[attr] is None:
                warnings.warn('In order to save outputs and use Catalog.py, \
                               please supply %s attribute' % (attr))
            else:
                setattr(self, attr, opt_args[attr])

        if flag_choice is None:
            warnings.warn('flag_choice is set to None. If this does not ' +
                          'reflect the flag_choice of the original data, ' +
                          'then saved arrays will be mislabled')
        self.flag_choice = flag_choice

        if not read_paths:
            args = (data, Nbls, freq_array)
            assert all([arg is not None for arg in args]),\
                'Insufficient data given. You must supply a data array,\
                 a Nbls array of matching shape, and a freq_array of matching sub-shape'
            self.data = data
            self.Nbls = Nbls
            self.freq_array = freq_array

            for attr in ['pols', 'vis_units']:
                if opt_args[attr] is None:
                    warnings.warn('In order to use Catalog_Plot.py, with \
                                   appropriate labels please supply %s attribute' % (attr))
                else:
                    setattr(self, attr, opt_args[attr])

            kwargs = {'match_events': match_events, 'match_hists': match_hists,
                      'chisq_events': chisq_events, 'chisq_hists': chisq_hists,
                      'samp_thresh_events': samp_thresh_events}
            for kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])
        else:
            self.read(read_paths)

        self.data_ms = self.mean_subtract()
        self.counts, self.bins, self.sig_thresh = self.hist_make()

    def mean_subtract(self):

        # This constant is determined by the Rayleigh distribution, which
        # describes the ratio of its rms to its mean
        C = 4 / np.pi - 1
        MS = (self.data / self.data.mean(axis=0) - 1) * np.sqrt(self.Nbls / C)
        return(MS)

    def hist_make(self, sig_thresh=None, event=None):

        if sig_thresh is None:
            sig_thresh = np.sqrt(2) * erfcinv(1. / np.prod(self.data.shape))
        bins = np.linspace(-sig_thresh, sig_thresh,
                           num=int(2 * np.ceil(2 * sig_thresh)))
        if event is None:
            dat = self.data_ms
        else:
            N = np.count_nonzero(np.logical_not(self.data_ms.mask[:, event[0], event[1]]), axis=1)
            dat = self.data_ms[:, event[0], event[1]].mean(axis=1) * np.sqrt(N)
        if dat.min() < -sig_thresh:
            bins = np.insert(bins, 0, dat.min())
        if dat.max() > sig_thresh:
            bins = np.append(bins, dat.max())
        counts, _ = np.histogram(dat[np.logical_not(dat.mask)],
                                 bins=bins)

        return(counts, bins, sig_thresh)

    def save(self):
        tags = ['match', 'chisq', 'samp_thresh']
        tag = ''
        for subtag in tags:
            if len(getattr(self, '%s_events' % (subtag))):
                tag += '_%s' % subtag

        for string in ['arrs', 'metadata']:
            if not os.path.exists('%s/%s' % (self.outpath, string)):
                os.makedirs('%s/%s' % (self.outpath, string))

        for attr in ['data', 'data_ms', 'Nbls']:
            pickle.dump(getattr(self, attr).data,
                        open('%s/arrs/%s_%s_INS_%s%s.npym' %
                        (self.outpath, self.obs, self.flag_choice, attr, tag), 'wb'))
        for attr in ['counts', 'bins']:
            np.save('%s/arrs/%s_%s_INS_%s%s.npy' %
                    (self.outpath, self.obs, self.flag_choice, attr, tag),
                    getattr(self, attr))
        for attr in ['match_events', 'match_hists', 'chisq_events',
                     'chisq_hists', 'samp_thresh_events']:
            if len(getattr(self, attr)):
                np.save('%s/arrs/%s_%s_INS_%s.npy' %
                        (self.outpath, self.obs, self.flag_choice, attr),
                        getattr(self, attr))

        for attr in ['freq_array', 'pols', 'vis_units']:
            if hasattr(self, attr):
                np.save('%s/metadata/%s_%s.npy' % (self.outpath, self.obs, attr),
                        getattr(self, attr))

    def read(self, read_paths):

        for arg in ['data', 'Nbls', 'freq_array']:
            assert arg in read_paths,\
                'You must supply a path to a numpy loadable %s file for read_paths entry' % (arg)
            setattr(self, arg, np.load(read_paths[arg]))
        for attr in ['pols', 'vis_units']:
            if attr not in read_paths:
                warnings.warn('In order to use Catalog_Plot.py, please supply\
                               path to numpy loadable %s attribute for read_paths entry' % (attr))
            else:
                setattr(self, attr, np.load(read_paths[attr]))
        for attr in ['match', 'chisq']:
            for subattr in ['events', 'hists']:
                attribute = '%s_%s' % (attr, subattr)
                if attribute in read_paths:
                    setattr(self, attribute, np.load(read_paths[attribute]).tolist())
                else:
                    setattr(self, attribute, [])
        if 'samp_thresh_events' in read_paths:
            self.samp_thresh_events = np.load(read_paths['samp_thresh_events'])
        else:
            self.samp_thresh_events = []
