import numpy as np
from pyuvdata import UVData
import os
import scipy.linalg
from scipy.special import erfinv
import rfiutil
import warnings
import scipy.stats
from VDH import Hist
from INS import Spectrum
from MF import match_filter


class RFI:

    def __init__(self, obs, inpath, outpath, filetype, bad_time_indices=None,
                 read_kwargs={}, flag_choice=None, INS=None, custom=None):

        # These lines establish the most basic attributes of the class, namely
        # its base UVData object and the obsid
        self.obs = obs
        self.UV = UVData()
        self.outpath = outpath
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

        if flag_choice is not None:
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

        self.VDH = Hist(self.UV.data_array, bins=bins, MLE_axis=MLE_axis)
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

    def bl_flag(self, INS=None, sig_thresh=None, shape_dict={}, N_thresh=0,
                alpha=None, tests=['match_filter'], choice=None, custom=None,
                MC_iter=int(1e4)):

        # Make a match filtered noise spectrum if one is not already passed
        if INS is None:
            self.MF_prepare(sig_thresh=sig_thresh, shape_dict=shape_dict,
                            N_thresh=N_thresh, alpha=alpha, tests=tests)
        else:
            self.INS = INS

        # Calculate MLE's with the INS flags in mind, and then apply choice of
        # non-INS flags to the data
        self.apply_flags(choice='INS', INS=self.INS)
        self.VDH_prepare(self.UV.data_array)
        self.apply_flags(choice=choice, custom=custom)

        # Make a temporary mask that all eventual changes will first be applied to
        # The reason for this thing's existence is that we don't want to alter
        # events later in the stack by dynamically altering the data's actual mask
        temp_mask = np.zeros(self.UV.data_array.shape, dtype=bool)
        temp_mask[self.UV.data_array.mask] = 1

        bl_hist = []
        sim_hist = []
        cutoffs = []

        for event in self.INS.events:
            bl_avg = self.UV.data_array[event[2], :, event[0], event[1]]
            init_shape = bl_avg.shape
            init_mask = bl_avg.mask
            bl_avg = bl_avg.mean(axis=1)
            counts, bins = np.histogram(bl_avg[np.logical_not(bl_avg.mask)], bins='auto')
            sim_counts = np.zeros((MC_iter, len(counts)))
            # Simulate some averaged rayleigh data and histogram - take averages/variances of histograms
            for i in range(MC_iter):
                sim_data = np.random.rayleigh(size=init_shape,
                                              scale=np.sqrt(self.VDH.MLE[:, event[0], event[1]]))
                sim_data = np.ma.masked_where(init_mask, sim_data)
                sim_data = sim_data.mean(axis=1)
                sim_counts[i, :], _ = np.histogram(sim_data, bins=bins)
            exp_counts = sim_counts.mean(axis=0)
            exp_error = np.sqrt(sim_counts.var(axis=0))
            # Find where the expected counts are 0.1 * the observed counts
            max_loc = bins[:-1][exp_counts.argmax()] + 0.5 * (bins[1] - bins[0])
            R = counts.astype(float) / exp_counts
            lcut_cond = np.logical_and(R > 10, bins[1:] < max_loc)
            rcut_cond = np.logical_and(R > 10, bins[:-1] > max_loc)
            if np.any(lcut_cond):
                lcut = bins[1:][max(np.where(lcut_cond)[0])]
            else:
                lcut = bins[0]
            if np.any(rcut_cond):
                rcut = bins[:-1][min(np.where(rcut_cond)[0])]
            else:
                rcut = bins[-1]
            cut_cond = np.logical_or(bl_avg > rcut, bl_avg < lcut)
            cut_ind = np.where(cut_cond)
            # Flag the temp_mask
            if np.any(cut_cond):
                temp_mask[event[2], cut_ind[0], event[0], event[1]] = 1

            bl_hist.append([counts, bins])
            sim_hist.append([exp_counts, exp_error])
            cutoffs.append([lcut, rcut])
        self.UV.data_array.mask = temp_mask

        return(bl_hist, sim_hist, cutoffs)
