"""
Class for more deeply examining events which were identified in INS.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import warnings
import os
import pickle
import time


class ES:

    """
    ES stands for Event Statistics. While the INS averages over baselines to
    locate RFI events in time-frequency, this class takes those events and
    averages across time-frequency for those baselines to see if particular
    baselines are more contaminated than others. This is often used more like a
    function than an interactive class.
    """

    def __init__(self, data=None, flag_choice=None, events=None, MLE=None,
                 uvw_array=None, vis_units=None, obs=None, pols=None,
                 outpath=None, MC_iter=int(1e4), grid_dim=50, grid_lim=None,
                 R_thresh=10, read_paths={}, freq_array=None):

        """
        init function for ES class. Attributes can be read in from numpy loadable
        files using the read_paths dictionary keyword, otherwise they can be set
        manually.

        Required Parameters: data: The sky-subtracted visibility amplitudes to
                                   be examined.

                             events: The events found by some of the filter
                                     tests in the MF class.

                             MLE: The maximimum likelihood estimator for the
                                  Rayleigh width as calculated in the VDH class.

                             uvw_array: The uvw_array as found in a pyuvdata
                                        object

                             MC_iter: The number of simulated histograms to
                                      average together when flagging averaged
                                      events. Event just 100 iterations can be
                                      extremely time-intensive.

                             grid_dim: The number of pixels to have in each
                                       dimension of the uv-grid.

                             grid_lim: The u and v bounds for the uv-grid.

                             R_thresh: Describes the sensitivity of the flagger.
                                       Should be greater than 2.

        Optional Parameters: flag_choice: The flag choice used for the
                                          sky-subtracted data.

                             vis_units: The units for the sky-subtracted
                                        visibility amplitudes.

                             obs: The OBSID for the observation in question.

                             pols: The polarizations present in the data.

                             outpath: A base directory to save results to.

                             freq_array: The frequencies present in the data.

        The new flags are generated upon instantiation. This process uses
        Monte-Carlo to approximate the thermal distribution and draw
        a cutoff between the baselines exhibiting reasonable thermal noise and
        those exhibiting possible RFI contamination. This is a very slow
        algorithm at present.
        """

        if not read_paths:

            kwargs = {'events': events, 'MLE': MLE, 'uvw_array': uvw_array,
                      'vis_units': vis_units, 'obs': obs, 'pols': pols,
                      'outpath': outpath, 'flag_choice': flag_choice,
                      'MC_iter': MC_iter, 'grid_dim': grid_dim,
                      'R_thresh': R_thresh, 'freq_array': freq_array}

            for attr in kwargs:
                setattr(self, attr, kwargs[attr])

            if grid_lim is None:
                self.grid_lim = [self.uvw_array[:, :-1].min(),
                                 self.uvw_array[:, :-1].max()]
            else:
                self.grid_lim = grid_lim

            self.grid = np.linspace(min(self.grid_lim), max(self.grid_lim),
                                    num=grid_dim + 1)
            self.Nbls = data.shape[1]
            temp_mask = np.zeros(data.shape, dtype=bool)

            attr_list = ['avgs', 'counts', 'exp_counts', 'exp_error', 'bins',
                         'uv_grid', 'cutoffs']
            for attr in attr_list:
                setattr(self, attr, [])

            if events is not None and len(events):
                print('Beginning bl_avg flagging at %s' % time.strftime("%H:%M:%S"))
                for event in events:
                    avg, counts, exp_counts, exp_error, bins = self.event_avg(data, event)
                    lcut, rcut = self.cutoff(counts, bins, exp_counts, R_thresh)
                    cut_cond = np.logical_or(avg > rcut, avg < lcut)
                    cut_ind = np.where(cut_cond)
                    temp_mask[event[0], cut_ind[0], 0, event[2]] = 1
                    uv_grid = self.bl_grid(avg, event)
                    for attr, calc in zip(attr_list, (avg, counts, exp_counts,
                                                      exp_error, bins, uv_grid,
                                                      np.array([lcut, rcut]))):
                        getattr(self, attr).append(calc)
                for attr in attr_list:
                    setattr(self, attr, np.array(getattr(self, attr)))
                self.mask = temp_mask
                print('Done with bl_avg flagging at %s' % time.strftime("%H:%M:%S"))
            else:
                print('No events given to ES class. Not computing flags.')
        else:
            self.read(read_paths)

    def save(self):
        """
        Writes out the attributes of this class.
        """
        for subdir in ['arrs', 'metadata']:
            path = '%s/%s' % (self.outpath, subdir)
            if not os.path.exists(path):
                os.makedirs(path)

        for attr in ['vis_units', 'pols', 'grid', 'freq_array']:
            if hasattr(self, attr):
                np.save('%s/metadata/%s_%s.npy' % (self.outpath, self.obs, attr),
                        getattr(self, attr))

        for attr in ['counts', 'exp_counts', 'exp_error', 'bins', 'cutoffs']:
            np.save('%s/arrs/%s_%s_ES_%s.npy' %
                    (self.outpath, self.obs, self.flag_choice, attr),
                    getattr(self, attr))
        for attr in ['avgs', 'uv_grid']:
            with open('%s/arrs/%s_%s_ES_%s.npym' %
                      (self.outpath, self.obs, self.flag_choice, attr), 'wb') as f:
                pickle.dump(getattr(self, attr), f)

    def read(self, read_paths):
        """
        Reads in the attributes which are present in the read_paths dictionary.
        See init docstring for details on which attributes are optional.
        """
        for attr in ['vis_units', 'pols', 'grid', 'freq_array']:
            if attr not in read_paths or read_paths[attr] is None:
                warnings.warn('In order to use SSINS.Catalog_Plot, please supply\
                               numpy loadable path for %s read_paths entry' % attr)
            else:
                setattr(self, attr, np.load(read_paths[attr]))
        for attr in ['counts', 'exp_counts', 'exp_error', 'bins', 'cutoffs',
                     'avgs', 'uv_grid']:
            assert attr in read_paths and read_paths[attr] is not None, \
                'Insufficient data. You must supply numpy loadable path for %s \
                 read_paths entry'
            setattr(self, attr, np.load(read_paths[attr]))

    def event_avg(self, data, event):

        """
        This takes an event (time-frequency) and averages the visibility
        difference amplitudes across that event for each baseline. Then,
        simulated noise is averaged across that same event, where the noise
        is estimated using the maximum-likelihood estimator (calculated in a VDH
        instance) for the noise in each baseline/frequency/polarization. This
        simulated averaged noise is histogrammed, and this process is repeated
        MC_iter times. The histograms are averaged together to draw an
        empirical thermal distribution.
        """

        avg = data[event[0], :, 0, event[2]]
        init_shape = avg.shape
        init_mask = avg.mask
        avg = avg.mean(axis=1)
        counts, bins = np.histogram(avg[np.logical_not(avg.mask)], bins='auto')
        sim_counts = np.zeros((self.MC_iter, len(counts)))
        # Simulate some averaged rayleigh data and histogram - take averages/variances of histograms
        for i in range(self.MC_iter):
            sim_data = np.random.rayleigh(size=init_shape,
                                          scale=np.sqrt(self.MLE[0, event[2]]))
            sim_data = sim_data.mean(axis=0)
            sim_counts[i, :], _ = np.histogram(sim_data, bins=bins)
        exp_counts = sim_counts.mean(axis=0)
        exp_error = np.sqrt(sim_counts.var(axis=0))

        return(avg, counts, exp_counts, exp_error, bins)

    def cutoff(self, counts, bins, exp_counts, R_thresh):

        """
        This function takes the histogrammed, averaged data, and compares it to
        the empirical distribution drawn from event_avg(). Cutoffs are drawn
        based on R_thresh, which is a ratio of counts.
        """

        max_loc = bins[:-1][exp_counts.argmax()] + 0.5 * (bins[1] - bins[0])
        R = counts.astype(float) / exp_counts
        lcut_cond = np.logical_and(R > self.R_thresh, bins[1:] < max_loc)
        rcut_cond = np.logical_and(R > self.R_thresh, bins[:-1] > max_loc)
        if np.any(lcut_cond):
            lcut = bins[1:][max(np.where(lcut_cond)[0])]
        else:
            lcut = bins[0]
        if np.any(rcut_cond):
            rcut = bins[:-1][min(np.where(rcut_cond)[0])]
        else:
            rcut = bins[-1]

        return(lcut, rcut)

    def bl_grid(self, avg, event):

        """
        This takes time-frequency averaged data from event_avg() and coarsely
        grids it in the UV-plane at the time of the event. Each pixel is
        averaged across the baselines whose centers lie within the pixel.
        """

        u = self.uvw_array[event[0] * self.Nbls:(event[0] + 1) * self.Nbls, 0]
        v = self.uvw_array[event[0] * self.Nbls:(event[0] + 1) * self.Nbls, 1]
        uv_grid = np.zeros((len(self.pols), self.grid_dim, self.grid_dim))
        for i in range(self.grid_dim):
            for k in range(self.grid_dim):
                uv_grid[:, -k, i] = avg[np.logical_and(np.logical_and(u < self.grid[i + 1], self.grid[i] < u),
                                                       np.logical_and(v < self.grid[k + 1], self.grid[k] < v))].sum()

        return(uv_grid)
