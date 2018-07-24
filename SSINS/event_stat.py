from __future__ import absolute_import, division, print_function

import numpy as np


class ES:

    def __init__(self, data, events, MLE, uvw_array, vis_units, obs,
                 pols, outpath, MC_iter=int(1e4), grid_dim=50, grid_lim=None,
                 R_thresh=10):

        args = {'events': events, 'MLE': MLE, 'uvw_array': uvw_array,
                'vis_units': vis_units, 'obs': obs, 'pols': pols, 'outpath': outpath}
        kwargs = {'MC_iter': MC_iter, 'grid_dim': grid_dim, 'R_thresh': R_thresh}

        for attr in args:
            setattr(self, attr, args[attr])
        for attr in kwargs:
            setattr(self, attr, kwargs[attr])

        if grid_lim is None:
            self.grid_lim = [self.uvw_array[:, :-1].min(), self.uvw_array[:, :-1].max()]
        else:
            self.grid_lim = grid_lim

        self.grid = np.linspace(min(grid_lim), max(grid_lim), num=grid_dim + 1)
        self.Nbls = data.shape[1]
        self.Nfreqs = data.shape[3]
        temp_mask = np.zeros(data.shape)

        attr_list = ['avgs', 'counts', 'exp_counts', 'exp_error', 'bins', 'uv_grid', 'cutoffs']
        for attr in attr_list:
            setattr(self, attr, [])

        for event in events:
            avg, counts, exp_counts, exp_error, bins = self.event_avg(data, event)
            lcut, rcut = self.cutoff(counts, bins, exp_counts, event, R_thresh)
            cut_cond = np.logical_or(avg > rcut, avg < lcut)
            cut_ind = np.where(cut_cond)
            temp_mask[event[2], cut_ind[0], event[0], event[1]] = 1
            uv_grid = self.bl_grid(avg, event)
            for attr, calc in zip(attr_list, (avg, counts, exp_counts, exp_error, bins, uv_grid, [lcut, rcut])):
                getattr(self, attr).append(calc)
        self.mask = temp_mask

    def event_avg(self, data, event):

        avg = data[event[2], :, event[0], event[1]]
        init_shape = bl_avg.shape
        init_mask = bl_avg.mask
        avg = bl_avg.mean(axis=1)
        counts, bins = np.histogram(bl_avg[np.logical_not(bl_avg.mask)], bins='auto')
        sim_counts = np.zeros((self.MC_iter, len(counts)))
        # Simulate some averaged rayleigh data and histogram - take averages/variances of histograms
        for i in range(self.MC_iter):
            sim_data = np.random.rayleigh(size=init_shape,
                                          scale=np.sqrt(self.MLE[:, event[0], event[1]]))
            sim_data = np.ma.masked_where(init_mask, sim_data)
            sim_data = sim_data.mean(axis=1)
            sim_counts[i, :], _ = np.histogram(sim_data, bins=bins)
        exp_counts = sim_counts.mean(axis=0)
        exp_error = np.sqrt(sim_counts.var(axis=0))

        return(avg, counts, exp_counts, exp_error, bins)

    def cutoff(self, counts, bins, exp_counts, event, R_thresh):

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

        return(l_cut, r_cut)

    def bl_grid(self, avg, event):

        u = self.uvw_array[event[2] * self.Nbls:(event[2] + 1) * self.Nbls, 0]
        v = self.uvw_array[event[2] * self.Nbls:(event[2] + 1) * self.Nbls, 1]
        uv_grid = np.zeros(len(self.pols), self.grid_dim, self.grid_dim)
        for i in range(self.grid_dim):
            for k in range(self.grid_dim):
                uv_grid[:, -k, i] = avg[np.logical_and(np.logical_and(u < self.grid[i + 1], self.grid[i] < u),
                                                       np.logical_and(v < self.grid[k + 1], self.grid[k] < v))].sum()

        return(uv_grid)
