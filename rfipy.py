import numpy as np
from pyuvdata import UVData
import os
import Util
import scipy.stats
from VDH import Hist
from INS import Spectrum
from MF import match_filter
from ES import event_stats


class RFI:

    def __init__(self, obs, inpath, outpath, filetype, bad_time_indices=None,
                 read_kwargs={}, flag_choice=None, INS=None, custom=None):

        # These lines establish the most basic attributes of the class, namely
        # its base UVData object and the obsid
        self.obs = obs
        self.UV = UVData()
        self.outpath = outpath
        self.flag_choice = flag_choice
        getattr(self.UV, 'read_%s' % (filetype))(inpath, **read_kwargs)

        # These ensure that every baseline reports at every time so that subtraction
        # can go off without a hitch
        assert self.UV.Nblts == self.UV.Nbls * self.UV.Ntimes, 'Nblts != Nbls * Ntimes'
        cond = np.all([self.UV.baseline_array[:self.UV.Nbls] ==
                       self.UV.baseline_array[k * self.UV.Nbls:(k + 1) * self.UV.Nbls]
                       for k in range(1, self.UV.Ntimes - 1)])
        assert cond, 'Baseline array slices do not match!'

        if bad_time_indices is not None:
            bool_ind = np.ones(self.UV.Ntimes, dtype=bool)
            bool_ind[bad_time_indices] = 0
            times = np.unique(self.UV.Ntimes)[bool_ind]
            self.UV.select(times=times)

        # This generalizes polarization references during plotting
        pol_keys = range(-8, 5)
        pol_keys.remove(0)
        pol_values = ['YX', 'XY', 'YY', 'XX', 'LR', 'RL', 'LL', 'RR', 'I', 'Q',
                      'U', 'V']
        pol_dict = dict(zip(pol_keys, pol_values))
        self.pols = [pol_dict[self.UV.polarization_array[k]] for k in
                     range(self.UV.Npols)]

        self.UV.data_array = np.ma.masked_array(np.absolute(np.diff(np.reshape(self.UV.data_array,
                                                [self.UV.Ntimes, self.UV.Nbls, self.UV.Nspws,
                                                 self.UV.Nfreqs, self.UV.Npols]), axis=0)))

        self.UV.flag_array = np.reshape((self.UV.flag_array[:-self.UV.Nbls] +
                                         self.UV.flag_array[self.UV.Nbls:]) > 0,
                                        [self.UV.Ntimes - 1, self.UV.Nbls,
                                         self.UV.Nspws, self.UV.Nfreqs,
                                         self.UV.Npols]).astype(bool)

        for subdir in ['arrs', 'figs', 'metadata']:
            path = '%s/%s/' % (self.outpath, subdir)
            if not os.path.exists(path):
                os.makedirs(path)
            assert os.path.exists(path), 'Output directories could not be created. Check permissions.'

        for meta in ['pols', 'obs']:
            np.save('%s/metadata/%s_%s.npy' % (self.outpath, self.obs, meta), getattr(self, meta))
        for meta in ['vis_units', 'freq_array']:
            np.save('%s/metadata/%s_%s.npy' % (self.outpath, self.obs, meta), getattr(self.UV, meta))

        if self.flag_choice is not None:
            self.apply_flags(choice=flag_choice, INS=INS, custom=custom)

    def apply_flags(self, choice=None, INS=None, custom=None):
        if choice is 'Original':
            self.UV.data_array.mask = self.UV.flag_array
        elif choice is 'INS':
            ind = np.where(INS.data.mask)
            self.UV.data_array[ind[0], :, ind[1], ind[2], ind[3]] = np.ma.masked
        elif choice is 'custom':
            self.UV.data_array[custom] = np.ma.masked
        elif np.any(self.UV.data_array.mask):
            self.UV.data_array.mask = False

    def INS_prepare(self):
        data = self.UV.data_array.mean(axis=1)
        Nbls = np.count_nonzero(np.logical_not(self.UV.data_array.mask(axis=1)))
        args = (data, Nbls, self.UV.freq_array, self.pols, self.UV.vis_units,
                self.obs, self.outpath)
        self.INS = Spectrum(*args)

    def VDH_prepare(self, bins='auto', MLE_axis=0, window=None, rev_ind_axis=None):

        args = (self.UV.data_array, self.flag_choice, self.UV.freq_array,
                self.pols, self.UV.vis_units, self.obs, self.outpath)
        self.VDH = Hist(*args, bins=bins, MLE_axis=MLE_axis)
        if window is not None:
            self.VDH.rev_ind(self.UV.data_array, window=window, axis=rev_ind_axis)

    def MF_prepare(self, sig_thresh=None, shape_dict={}, N_thresh=0, alpha=None,
                   tests=['match_filter']):

        if not hasattr(self, 'INS'):
            self.INS_prepare()
        self.MF = match_filter(self.INS, sig_thresh=None, shape_dict={},
                               N_thresh=0, alpha=None)
        if tests is not None:
            for test in tests:
                getattr(self.MF, 'apply_%s_test' % (test))

    def BA_prepare(self, grid_lim=None, INS=None, sig_thresh=None, shape_dict={},
                   N_thresh=0, alpha=None, tests=['match_filter'], choice=None,
                   custom=None, MC_iter=int(1e4), grid_dim=[50, 50], R_thresh=10):

        # Make a match filtered noise spectrum if one is not already passed
        if INS is None:
            self.MF_prepare(sig_thresh=sig_thresh, shape_dict=shape_dict,
                            N_thresh=N_thresh, alpha=alpha, tests=tests)
        else:
            self.INS = INS

        # Calculate MLE's with the INS flags in mind, and then apply choice of
        # non-INS flags to the data
        self.apply_flags(choice='INS', INS=self.INS)
        self.VDH_prepare(self.UV.data_array, 'INS', self.freq_array, self.pols,
                         self.UV.vis_units, self.obs, self.outpath)
        self.apply_flags(choice=choice, custom=custom)

        BA_args = (self.UV.data_array, self.INS.events, self.VDH.MLEs[-1],
                   self.UV.uvw_array, self.UV.vis_units, self.obs, self.pols,
                   self.outpath)

        self.BA = bl_avg(*BA_args, MC_iter=MC_iter, grid_dim=grid_dim,
                         grid_lim=grid_lim, R_thresh=R_thresh)

        self.UV.data_array.mask = np.logical_and(self.UV.data_array.mask, self.BA.mask)
