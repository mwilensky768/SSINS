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


class SS(object):

    """
    Defines the SS class.
    """

    def __init__(self, obs=None, outpath=None, UV=None, inpath=None,
                 bad_time_indices=None, read_kwargs={}, flag_choice=None,
                 INS=None, custom=None, diff=True):

        """
        init function for the SS class. One may pass it a UVData object with
        the UV keyword. The typical mode of operation is to pass an inpath
        which leads to a datafile in a pyuvdata-readable format. Currently only
        data with Nbls * Ntimes = Nblts are supported.

        Keywords: obs: The OBSID of the data in question. Only necessary if
                       saved outputs are desired.

                  outpath: The location to a directory where saved outputs ought
                           to go. Only necessary if saved outputs are desired.

                  UV: A UVData object to pass. If this is set to None, then a
                      path to a pyuvdata-readable file must be supplied for the
                      inpath keyword. Otherwise, this will be the working UVData
                      object in all analysis.

                  inpath: A path to a pyuvdata-readable file with which to form
                          the working UVData object during analysis. Necessary
                          if the UV keyword is not set to a valid UVData object.

                  bad_time_indices: One may remove times from the observation
                                    by index rather than JD by setting this
                                    keyword equal to a sequence of indices to be
                                    removed from the UVData object.

                  read_kwargs: If the UVData object is read-in rather than
                               manually supplied, then this keyword dictionary
                               will be passed to pyuvdata.UVData.read() so as to
                               allow for relatively robust UVData object setup.

                  flag_choice: The flag choice to apply to the data. See
                               apply_flags() function for more explanation.

                  INS: A keyword for apply_flags().

                  custom: A keyword for apply_flags().

                  diff: Whether or not to difference the data. This should be
                        left true unless differences were already formed by the
                        user. This will also take the original UVData flag_array
                        and form "differenced flags," where a difference is
                        flagged if either of its contributing visibilities were
                        flagged. The data and flag arrays are also reshaped so
                        that they have separate time and baseline axes.
        """

        self.obs = obs
        self.outpath = outpath
        for attr in ['obs', 'outpath']:
            if getattr(self, attr) is None:
                warnings.warn('%s%s' % ('In order to save outputs and use Catalog_Plot.py,',
                                        'please supply %s keyword other than None' % (attr)))

        if UV is None:
            self.UV = self.read(inpath, read_kwargs=read_kwargs,
                                bad_time_indices=bad_time_indices)

            self.flag_choice = flag_choice
        else:
            self.UV = UV
            self.flag_choice = flag_choice

        pol_keys = list(range(-8, 5))
        pol_keys.remove(0)
        pol_values = ['YX', 'XY', 'YY', 'XX', 'LR', 'RL', 'LL', 'RR', 'I', 'Q',
                      'U', 'V']
        pol_dict = dict(zip(pol_keys, pol_values))
        self.pols = np.array([pol_dict[self.UV.polarization_array[k]] for k in
                              range(self.UV.Npols)])

        if diff:
            assert self.UV.Nblts == self.UV.Nbls * self.UV.Ntimes, 'Nblts != Nbls * Ntimes'
            cond = np.all([self.UV.baseline_array[:self.UV.Nbls] == self.UV.baseline_array[k * self.UV.Nbls:(k + 1) * self.UV.Nbls]
                           for k in range(1, self.UV.Ntimes - 1)])
            assert cond, 'Baseline array slices do not match in each time! The baselines are out of order.'

            reshape = [self.UV.Ntimes, self.UV.Nbls, self.UV.Nspws,
                       self.UV.Nfreqs, self.UV.Npols]

            self.UV.data_array = np.reshape(self.UV.data_array, reshape)
            self.UV.data_array = np.diff(self.UV.data_array, axis=0)
            self.UV.data_array = np.ma.masked_array(np.absolute(self.UV.data_array))

            self.UV.flag_array = np.reshape(self.UV.flag_array, reshape)
            self.UV.flag_array = np.logical_or(self.UV.flag_array[:-1],
                                               self.UV.flag_array[1:])

        if self.flag_choice is not None:
            self.apply_flags(choice=self.flag_choice, INS=INS, custom=custom)

    def apply_flags(self, choice=None, INS=None, custom=None):
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
        self.flag_choice = choice
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

    def save_meta(self):

        """
        Saves useful metadata to the outpath.
        """

        path = '%s/metadata' % self.outpath
        if not os.path.exists(path):
            os.makedirs(path)
        assert os.path.exists(path), 'Output directory, %s, could not be created.\
                                      Check permissions.' % (path)
        np.save('%s/%s_pols.npy' % (path, self.obs), self.pols)
        for meta in ['vis_units', 'freq_array']:
            np.save('%s/metadata/%s_%s.npy' %
                    (self.outpath, self.obs, meta), getattr(self.UV, meta))
        for meta in ['time_array', 'lst_array']:
            np.save('%s/metadata/%s_%s.npy' % (self.outpath, self.obs, meta),
                    np.unique(getattr(self.UV, meta)))

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

        if not hasattr(self, 'INS'):
            self.INS_prepare()
        self.MF = MF(self.INS, sig_thresh=sig_thresh, shape_dict=shape_dict,
                     N_thresh=N_thresh, alpha=alpha, point=point, streak=streak)

    def ES_prepare(self, grid_lim=None, INS=None, sig_thresh=None, shape_dict={},
                   N_thresh=0, alpha=None, tests=['match'], choice=None,
                   fit_hist=False, bins=None, custom=None,
                   MC_iter=int(1e4), grid_dim=50, R_thresh=10):

        """
        Creates an ES class to work with. If a filtered INS is not already
        made, one is made according to the corresponding passed keywords. The
        filtered INS is used to make an improved MLE using the VDH class. The
        MLE is then used to simulate thermal distributions for the data,
        possibly flagged, averaged over those events which were located in the
        filtered INS. This allows for INS-informed flagging while still
        preserving some of the possibly uncontaminated baselines.

        Keywords: grid_lim: Sets the limits of the uv-grid (in meters)

                  INS: If None, will either prepare or use the one which is
                       already prepared. Otherwise the passed one will be used
                       for the MLE.

                  sig_thresh: keyword for the MF preparation step

                  shape_dict: keyword for the MF preparation step

                  N_thresh: keyword for the MF preparation step

                  alpha: keyword for the MF preparation step

                  tests: keyword for the MF preparation step

                  choice: keyword for apply_flags(). Only 'custom', None, and
                          'original' choices can be used in this context.

                  custom: keyword for apply_flags().

                  fit_hist: keyword for the VDH preparation step

                  bins: keyword for the VDH preparation step

                  MC_iter: How many thermal histograms to simulate and average
                           together. Anything more then ~10 takes a long time.

                  grid_dim: The number of pixels in each dimension of the grid.

                  R_thresh: Aggression parameter for flagging. Should be at
                            least 2. A higher number is less aggressive.
        """

        # Make a match filtered noise spectrum if one is not already passed and
        # one is not already made.
        if INS is None:
            if not hasattr(self, 'MF'):
                MF_kwargs = {'sig_thresh': sig_thresh,
                             'shape_dict': shape_dict,
                             'N_thresh': N_thresh,
                             'alpha': alpha}
                self.MF_prepare(**MF_kwargs)
                for test in tests:
                    getattr(self.MF, 'apply_%s_test' % test)()
        else:
            self.INS = INS

        # Calculate MLE's with the INS flags in mind, and then apply choice of
        # non-INS flags to the data
        self.apply_flags(choice='INS', INS=self.INS)
        VDH_kwargs = {'bins': bins,
                      'fit_hist': fit_hist}
        print('Preparing VDH at %s' % time.strftime("%H:%M:%S"))
        self.VDH_prepare(**VDH_kwargs)
        print('Done preparing VDH at %s ' % time.strftime("%H:%M:%S"))
        self.apply_flags(choice=choice, custom=custom)

        ES_kwargs = {'data': self.UV.data_array,
                     'flag_choice': choice,
                     'events': self.INS.match_events,
                     'MLE': self.VDH.MLEs[-1],
                     'uvw_array': self.UV.uvw_array,
                     'vis_units': self.UV.vis_units,
                     'obs': self.obs,
                     'pols': self.pols,
                     'outpath': self.outpath,
                     'MC_iter': MC_iter,
                     'grid_dim': grid_dim,
                     'grid_lim': grid_lim,
                     'R_thresh': R_thresh,
                     'freq_array': self.UV.freq_array}

        self.ES = ES(**ES_kwargs)

    def read(self, inpath, read_kwargs={}, bad_time_indices=None):

        """
        Essentially a wrapper around UVData.read() with some extra bells and
        whistles.

        Keywords: inpath: path to pyuvdata-readable datafile

                  read_kwargs: Keyword dictionary to pass to UVData.read()

                  bad_time_indices: If a sequence is passed, removes data from
                                    the UVData object based on index rather than
                                    JD.
        """

        assert inpath is not None, 'Supply a path to a valid UVData file for the inpath keyword'

        UV = UVData()
        if bad_time_indices is not None:
            UV.read(inpath, read_data=False)
            time_arr = np.unique(UV.time_array)
            good_ind = np.ones(time_arr.shape, dtype=bool)
            good_ind[bad_time_indices] = 0
            times = time_arr[good_ind]
            read_kwargs['times'] = times
        UV.read(inpath, **read_kwargs)
        if np.any(UV.ant_1_array == UV.ant_2_array):
            warnings.warn('%s%s%s' % ('Autocorrelations are still present in the',
                                      ' UVData object. User may want to remove',
                                      ' these before analysis.'))

        return(UV)

    def write(self, outpath, file_type_out, UV=None, inpath=None, read_kwargs={},
              bad_time_indices=None, combine=True):

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
        getattr(UV, 'write_%s' % file_type_out)(outpath)
