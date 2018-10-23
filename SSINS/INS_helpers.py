"""
Some useful functions specifically involving INS analysis.
"""

from __future__ import absolute_import, division, print_function

from SSINS import util, INS


def INS_concat(INS_sequence, axis, metadata_kwargs={}):
    """
    This function is used to concatenate spectra. For instance, if polarizations
    are in separate files or if different parts of the observing band are in
    separate files.
    """
    data = np.concatenate([ins.data for ins in INS_sequence], axis=axis)
    Nbls = np.concatenate([ins.Nbls for ins in INS_sequence], axis=axis)
    # This is the frequency axis
    if axis is 2:
        freq_array = np.concatenate([ins.freq_array for ins in INS_sequence], axis=1)
    else:
        freq_array = INS_sequence[0].freq_array
    ins = INS(data=data, Nbls=Nbls, freq_array=freq_array, **metadata_kwargs)

    return(ins)
