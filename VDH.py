import numpy as np
import scipy.stats


class Hist:

    def __init__(self, data, bins='auto', MLE_axis=0):
        self.counts, self.bins = np.histogram(data[np.logical_not(data.mask)], bins=bins)
        self.MLE, self.fit, self.error = self.rayleigh_mixture_fit(data, MLE_axis)

    def rayleigh_mixture_fit(self, data, axis):
        MLE = 0.5 * np.mean(data**2, axis=axis)
        N = np.count_nonzero(np.logical_not(data.mask), axis=axis)
        Ntot = np.sum(N)
        axes = range(data.ndims)
        for mle, n in zip(MLE.flatten(), N.flatten()):
            P += n / Ntot * (scipy.stats.rayleigh.cdf(self.bins[1:], scale=np.sqrt(mle)) -
                             scipy.stats.rayleigh.cdf(self.bins[:-1], scale=np.sqrt(mle)))
        fit = Ntot * P
        error = np.sqrt(Ntot * P * (1 - P))
        return(MLE, fit, error)

    def rev_ind(data, window, axis):
        W = np.zeros(data.shape)
        W[np.logical_and(min(window) < data, data < max(window))] = 1
        self.W_hist = W.sum(axis=axis)
