"""
The sky_subtract (SS) class is defined here. This is the backbone of the analysis
pipeline when working with raw datafiles or UVData objects.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
from pyuvdata import UVData
import os
from SSINS import util
from SSINS import INS
from SSINS import MF
from SSINS import VDH
from SSINS import ES
import scipy.stats
import warnings
import time


class SS(UVData):

    """
    Defines the SS class.
    """

    def __init__(self):

        """
        """
        super(SS, self).__init__()

    def read(filename, **kwargs):
        super(SS, self).read(filename, **kwargs)
        if self.data_array is not None:
            self.diff()

    def apply_flags(self, flag_choice=None, INS=None, custom=None):
        """
        A function which applies flags to the data via numpy masked arrays. Also
        changes the SS.flag_choice attribute, which will affect saved outputs,
        so it is convenient to change flags using this function.

        keywords: choice: Options are None, 'original', 'INS', and 'custom'

                          None: No flags are applied to the data

                          'original': The "differenced flags" from the original
                                      flag_array are applied to the data

                          'custom': A custom flag array will be applied to the
                                    data.

                          'INS': A flag_array developed from an INS will be
                                 applied to the data. All flags in the INS will
                                 be extended across the baseline axis of the SS
                                 data array.

                  custom: The custom flags to be applied. Must be used in
                          conjunction with choice='custom'

                  INS: The INS whose flags will be applied. Must be used in
                       conjunction with choice='INS'
        """
        self.flag_choice = flag_choice
        if choice is 'original':
            self.UV.data_array.mask = self.UV.flag_array
        elif choice is 'INS':
            ind = np.where(INS.data.mask)
            self.UV.data_array[ind[0], :, ind[1], ind[2], ind[3]] = np.ma.masked
        elif choice is 'custom':
            if custom is not None:
                self.UV.data_array[custom] = np.ma.masked
            else:
                warnings.warn('Custom flags were chosen, but custom flags were None type. Not applying flags.')
        elif np.any(self.UV.data_array.mask):
            self.UV.data_array.mask = False

    def diff():

        assert self.Nblts == self.Nbls * self.Ntimes, 'Nblts != Nbls * Ntimes'
        cond = np.all([self.baseline_array[:self.Nbls] == self.baseline_array[k * self.Nbls:(k + 1) * self.Nbls]
                       for k in range(1, self.Ntimes - 1)])
        assert cond, 'Baseline array slices do not match in each time! The baselines are out of order.'

        # Difference in time and OR the flags
        self.data_array = np.ma.masked_array(np.absolute(self.data_array[self.Nbls:] - self.data_array[:-self.Nbls]))
        self.flag_array = np.logical_or(self.flag_array[self.Nbls:], self.flag_array[:-self.Nbls])

        # Adjust the UVData attributes.
        self.ant_1_array = self.ant_1_array[:-self.Nbls]
        self.ant_2_array = self.ant_2_array[:-self.Nbls]
        self.baseline_array = self.baseline_array[:-self.Nbls]
        self.integration_time = self.integration_time[self.Nbls:] + self.integration_time[:-self.Nbls]
        self.Ntimes -= 1
        self.nsample_array = 0.5 * (self.nsample_array[self.Nbls:] + self.nsample_array[:-self.Nbls])
        self.time_array = 0.5 * (self.time_array[self.Nbls:] + self.time_array[:-self.Nbls])
        self.uvw_array = 0.5 * (self.uvw_array[self.Nbls:] + self.uvw_array[:-self.Nbls])
        super(SS, self).set_lsts_from_time_array()

    def save_data(self):
        """
        Saves formed data products to the outpath using their respective save()
        functions.
        """

        for attr in ['INS', 'VDH', 'ES']:
            if hasattr(self, attr):
                getattr(getattr(self, attr), 'save')()

    def INS_prepare(self, order=0):

        """
        Prepares an INS. Passes all possible relevant non-conflicting attributes.
        """

        data = self.UV.data_array.mean(axis=1)
        if np.any(self.UV.data_array.mask):
            Nbls = np.count_nonzero(np.logical_not(self.UV.data_array.mask), axis=1)
        else:
            Nbls = self.UV.Nbls * np.ones(data.shape)
        kwargs = {'data': data,
                  'Nbls': Nbls,
                  'freq_array': self.UV.freq_array,
                  'pols': self.pols,
                  'vis_units': self.UV.vis_units,
                  'obs': self.obs,
                  'outpath': self.outpath,
                  'flag_choice': self.flag_choice,
                  'order': order}
        self.INS = INS(**kwargs)

    def VDH_prepare(self, bins=None, fit_hist=False, MLE=True, window=None):

        """
        Prepares a VDH. Passes all possible relevant non-conflicting attributes.

        Keywords: bins: The bins to use for the histogram. Options are None,
                        'auto', sequence

                        None: Logarithmically spaced bins spanning the nonzero
                              data are made.

                        'auto': Same as passing 'auto' to np.histogram()

                        sequence: The sequence is used to define the bin edges.

                  fit_hist: Make a Rayleigh-mixture fit to the histograms.
                            Requires MLE=True

                  MLE: Calculate the Rayleigh MLE for each baseline, frequency,
                       and polarization.

                  window: Provide upper and lower limits for VDH.rev_ind()
        """

        kwargs = {'data': self.UV.data_array,
                  'flag_choice': self.flag_choice,
                  'freq_array': self.UV.freq_array,
                  'pols': self.pols,
                  'vis_units': self.UV.vis_units,
                  'obs': self.obs,
                  'outpath': self.outpath,
                  'bins': bins,
                  'fit_hist': fit_hist,
                  'MLE': MLE}
        self.VDH = VDH(**kwargs)
        if window is not None:
            self.VDH.rev_ind(self.UV.data_array, window)

    def MF_prepare(self, sig_thresh=None, shape_dict={}, N_thresh=0, alpha=None,
                   point=True, streak=True):

        """
        Prepares a MF. Since a MF requires an INS, if the SS does not have an
        INS yet, then an INS is prepared with the current settings.

        Keywords: sig_thresh: Passes this as the sig_thresh attribute of the MF.
                              sig_thresh=None will force the MF to calculate a
                              reasonable one based on the data size.

                  shape_dict: Gives the shape dictionary to the MF. Keys are
                              only used internally, but they refer to the names
                              of the RFI shapes being looked for. Values are
                              upper and lower frequency limits for the
                              corresponding shapes.

                  N_thresh: Sets the N_thresh parameter used in the
                            samp_thresh_test

                  alpha: Sets the significance level for the chisq_test.
                         alpha=None will force the MF to calculate one based on
                         the data size.

                  tests: The tests performed by the match filter, in the order
                         of the sequence given. Options are 'match', 'chisq',
                         'samp_thresh'

                  point: Instruct the MF to look for single-point outliers if
                         point=True. Else omit this search.

                  streak: Instruct the MF to look for broadband (possibly not
                          band-limited) features in the data. Else omit this
                          search.
        """

        self.MF = MF(self.UV.freq_array, sig_thresh=sig_thresh, shape_dict=shape_dict,
                     N_thresh=N_thresh, alpha=alpha, point=point, streak=streak)

    def ES_prepare(self, apply_tests=True, tests=None, order=0, sig_thresh=None,
                   shape_dict={}, N_thresh=0, alpha=None, point=True, streak=True,
                   test_kwargs=None):

        """

        """

        ES_kwargs = {'flag_choice': self.flag_choice,
                     'obs': self.obs,
                     'outpath': self.outpath}

        self.ES = ES(**ES_kwargs)
        if not hasattr(self, INS):
            self.INS_prepare(order=order)
        if not hasattr(self.MF):
            self.MF_prepare(sig_thresh=sig_thresh, shape_dict=shape_dict,
                            N_thresh=N_thresh, alpha=alpha, point=point,
                            streak=streak)
        if apply_tests:
            if tests is None:
                raise TypeError("To apply tests, must supply types of test to apply")
            else:
                for i, test in enumerate(tests):
                    getattr(self.MF, 'apply_%s_test' % test)(self.INS, self.ES, **test_kwargs[i])

    def write(self, outpath, file_type_out, UV=None, inpath=None, read_kwargs={},
              bad_time_indices=None, combine=True, nsample_default=1, write_kwargs={}):

        """
        Lets one write out a newly flagged file. Data is recovered by reading
        in the original file or using the original UV object. If passing a UV
        object, be careful that the original UV object was not changed by any
        operations due to typical confusing python binding issues. The operation
        used to make "differenced flags" is actually not invertible in some
        cases, so this just extends flags as much as possible.

        Keywords: outpath: The name of the file to write out to.

                  file_type_out: The file_type to write out to.

                  UV: If using this, make sure it is the original UV object
                      intended without any extra flagging or differencing or
                      reshaped arrays.

                  inpath: The file to read in to get the original data from.

                  read_kwargs: The UVData.read keyword dict for the original
                               UVData object

                  bad_time_indices: Bad time indices to remove from original
                                    UVData object.
        """

        if UV is None:
            UV = self.read(inpath, read_kwargs=read_kwargs,
                           bad_time_indices=bad_time_indices)
        UV.nsample_array[UV.nsample_array == 0] = nsample_default
        UV.flag_array = UV.flag_array.reshape([UV.Ntimes, UV.Nbls, UV.Nspws,
                                               UV.Nfreqs, UV.Npols])
        if not combine:
            UV.flag_array[:] = 0
        for i in range(UV.Ntimes - 1):
            # This actually does not invert properly but I think it's the best way
            UV.flag_array[i][self.UV.data_array.mask[i]] = 1
            UV.flag_array[i + 1][self.UV.data_array.mask[i]] = 1
        UV.flag_array = UV.flag_array.reshape([UV.Nblts, UV.Nspws, UV.Nfreqs,
                                               UV.Npols])
        getattr(UV, 'write_%s' % file_type_out)(outpath, **write_kwargs)
