"""
Some utility functions. These are not used in the main code but are potentially
useful for scripts.
"""
import numpy as np
import os
from SSINS.match_filter import Event
import copy
import warnings


def get_unique_event_tf(ins, event_list):
    """
    Get unique time/frequency combinations from an event list for a given ins.

    Args:
        ins: An incoherent noise spectrum to which the events belong.
        event_list: List of events from the match filter.
    """

    time_range = np.arange(ins.Ntimes)
    freq_range = np.arange(ins.Nfreqs)

    tf_set = set()
    for event in event_list:
        ntimes_event = event.time_slice.stop - event.time_slice.start
        tf_subset = set(zip(time_range[event.time_slice],
                            np.repeat(freq_range[event.freq_slice], ntimes_event)))
        tf_set = tf_set.union(tf_subset)

    return(tf_set)


def get_sum_z_score(ins, tf_set):
    """
    Get the sum of the z-scores for a set of time-frequency combinations as
    returned by get_unique_event_tf. Used as a brightness metric.

    Args:
        ins: INS that has been flagged and has events contained in tf_set.
        tf_set: Time-frequency combinations as reported by get_unique_event_tf
    """

    total_z = 0
    # Just do a for loop since the array is small.
    for time_ind, freq_range in tf_set:
        # Careful with infs/nans
        z_arr = np.ma.masked_invalid(ins.metric_ms.data)

        # Just do the z-score calculation here.
        sliced_z_arr = z_arr[time_ind, freq_range]
        N_unmasked = np.count_nonzero(sliced_z_arr.mask, axis=1)
        z_subband = np.ma.sum(sliced_z_arr, axis=1) / np.sqrt(N_unmasked)

        total_z += z_subband

    return(total_z)


def calc_occ(ins, mf, num_init_flag, num_int_flag=0, lump_narrowband=False,
             return_z_scores=False):
    """
    Calculates the fraction of times an event was caught by the flagger for
    each type of event. Does not take care of frequency broadcasted events.

    Args:
        ins: The flagged incoherent noise spectrum in question
        mf: The match filter used to flag the INS
        num_init_flag: The number of initially flagged samples
        num_int_flag: The number of fully flagged integrations in the starting flags
        lump_narrowband: Whether to combine narrowband occupancies into a single number.
            Will slightly overestimate if n_int_flag > 0.
        return_brightness: Whether to return a flux density dictionary as well.

    Returns:
        occ_dict: A dictionary with shapes for keys and occupancy fractions for values
        z_dict: A dictionary with sum of z-scores for different shapes.
            z-scores are calculated from clean data only.
    """

    occ_dict = {shape: 0. for shape in mf.slice_dict}
    z_dict = {shape: 0 for shape in mf.bright_dict}

    # Figure out the total occupancy sans initial flags
    total_data = np.prod(ins.metric_array.shape)
    total_valid = total_data - num_init_flag
    total_flag = np.sum(ins.metric_array.mask)
    total_RFI = total_flag - num_init_flag
    total_occ = total_RFI / total_valid

    init_shapes = list(occ_dict.keys())

    for shape in init_shapes:
        relevant_events = [event for event in ins.match_events if shape in event.shape]

        if shape == "narrow":
            if lump_narrowband:
                # The length of the shape_set tells you how many points were flagged, without overcounting
                occ_dict[shape] = len(tf_set) / total_valid
                tf_set = get_unique_event_tf(ins, relevant_events)
                z_dict[shape] = get_sum_z_score(ins, tf_set)
            else:
                # Need to pull out the unique frequencies that were identified
                new_event_shapes = []
                for event in relevant_events:
                    _ind = event.shape.rfind("_")
                    if event.shape[_ind + 1:] not in new_event_shapes:
                        new_event_shapes.append(event.shape[_ind + 1:])
                # need to iterate over unique frequencies
                for subshape in new_event_shapes:
                    subshape_key = f"narrow_{subshape}"
                    subshape_relevant_events = [event for event in relevant_events if subshape in event.shape]
                    tf_set = get_unique_event_tf(ins, subshape_relevant_events)
                    occ_dict[subshape_key] = len(tf_set) / (ins.Ntimes - num_int_flag)
                    z_dict[shape] = get_sum_z_score(ins, tf_set)
                    # Sometimes broadcast events can make this bigger than 1.
                    # Does not apply to lump case.
                    if occ_dict[subshape_key] > 1:
                        occ_dict[subshape_key] = 1
        else:
            tf_set = get_unique_event_tf(ins, relevant_events)
            occ_dict[shape] = len(tf_set) / (ins.Ntimes - num_int_flag)
            z_dict[shape] = get_sum_z_score(ins, tf_set)
            # Sometimes broadcast events can make this bigger than 1.
            if occ_dict[shape] > 1:
                occ_dict[shape] = 1

    if not lump_narrowband:
        occ_dict.pop("narrow")

    occ_dict['total'] = total_occ
    for item in occ_dict:
        occ_dict[item] = float(occ_dict[item])

    if not return_z_dict:
        return(occ_dict)
    else:
        return(occ_dict, z_dict)


def make_obslist(obsfile):
    # due to the newline character, raw string is needed `r"""`
    r"""
    Makes a python list from a text file whose lines are separated by "\\n"

    Args:
        obsfile: A text file with an obsid on each line

    Returns:
        obslist: A list whose entries are obsids
    """
    with open(obsfile) as f:
        obslist = f.read().split("\n")
    while '' in obslist:
        obslist.remove('')
    obslist.sort()
    return(obslist)


def make_obsfile(obslist, outpath):
    """
    Makes a text file from a list of obsids

    Args:
        obslist: A list of obsids
        outpath: The filename to write to
    """
    with open(outpath, 'w') as f:
        for obs in obslist:
            f.write("%s\n" % obs)


def make_ticks_labels(freqs, freq_array, sig_fig=1):
    """
    Makes xticks from desired frequencies to be ticked and the freq_array.

    Args:
        freqs: The desired frequencies to be ticked, in Hz
        freq_array: The frequency array that is to be ticked.
        sig_fig: Precision of the label - number of digits after decimal point (in Mhz)
    """

    # Find the channel numbers closest to each desired frequency
    ticks = np.array([np.argmin(np.abs(freq_array - freq)) for freq in freqs])
    labels = [('%.' + str(sig_fig) + 'f') % (10**-6 * freq) for freq in freqs]

    return(ticks, labels)


def combine_ins(ins1, ins2, inplace=False):
    """
    This utility function combines INS for the same obs that have been averaged
    over different baselines.

    Args:
        ins1: The first spectrum for the combination
        ins2: The second spectrum for the combination
        inplace: Whether to do the operation inplace on ins1 or not.
    """

    if not np.array_equal(ins1.time_array, ins2.time_array):
        raise ValueError("The spectra do not have matching time arrays.")

    if not np.array_equal(ins1.freq_array, ins2.freq_array):
        raise ValueError("The spectra do not have the same frequencies.")

    if not np.array_equal(ins1.polarization_array, ins2.polarization_array):
        raise ValueError("The spectra do not have the same pols.")

    if not ins1.spectrum_type == ins2.spectrum_type:
        raise ValueError(f"ins1 is of type {ins1.spectrum_type} while ins2 is of type {ins2.spectrum_type}")

    if (not np.array_equal(ins1.metric_ms, ins1.sig_array)) or (not np.array_equal(ins2.metric_ms, ins2.sig_array)):
        warnings.warn("sig_array attribute will be reset after combinging INS")

    if inplace:
        this = ins1
    else:
        this = copy.deepcopy(ins1)

    new_weights = this.weights_array + ins2.weights_array
    new_weights_square = this.weights_square_array + ins2.weights_square_array
    new_metric = (this.weights_array * this.metric_array + ins2.weights_array * ins2.metric_array) / new_weights

    this.metric_array = new_metric
    this.weights_array = new_weights
    this.weights_square_array = new_weights_square

    this.metric_ms = this.mean_subtract()
    this.sig_array = np.ma.copy(this.metric_ms)

    return(this)


def write_meta(prefix, ins, uvf=None, mf=None, sep="_", clobber=False,
               data_compression="lzf"):
    """
    Wrapper around several calls to ins.write so that a standard set of
    metadata can be written.

    Args:
        prefix: The filepath prefix to write outputs to.
        ins: The INS for which to write metadata.
        uvf: A UVFlag object for which to generate diff-propagated flags, if desired.
        mf: The MF (match filter) that was run on the INS object, if any.
        sep: The separator character between the prefix and rest of output filenames.
        clobber: Whether to overwrite existing files.
        data_compression: The type of data compression to use for hdf5 outputs.
    """

    ins.write(prefix, sep=sep, clobber=clobber,
              data_compression=data_compression)
    ins.write(prefix, output_type="mask", sep=sep, clobber=clobber,
              data_compression=data_compression)
    ins.write(prefix, output_type="match_events", sep=sep, clobber=clobber,
              data_compression=data_compression)
    if uvf is not None:
        ins.write(prefix, output_type="flags", uvf=uvf, sep=sep, clobber=clobber,
                  data_compression=data_compression)
    if mf is not None:
        mf.write(prefix, sep=sep, clobber=clobber)
