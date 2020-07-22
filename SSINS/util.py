"""
Some utility functions. Some are utilized within the main code and some are just
nice to have around.
"""
import numpy as np
import os


def red_event_sort(match_events, shape_tuples):

    """
    This is a function that is used to filter redundant events from a match_events
    list, in case events with different names (or even shapes) are physically caused
    by the same mechanism.

    Args:
        match_events: A list of match_events from an INS found by a match filter
        shape_tuples:
            A list of tuples where each tuple indicates which events are
            considered redundant with one another. Each tuple must be ordered
            according to the priority in which the shapes are to be kept.

    Returns:
        match_events: The filtered match_event list.
    """

    removal_events = []
    # Iterate over the redundancy sets
    for shape_tup in shape_tuples:
        # Filter the shapes that are in the current shape tuple
        relevant_events = [event for event in match_events if event[-2] in shape_tup]
        all_times = [event[0] for event in relevant_events]
        # Generate an array of unique times and counts
        unique_times, counts = np.unique(all_times, return_counts=True)
        # find times with redundancies
        red_times = unique_times[counts > 1]
        # iterate over the times with redundancies
        for time in unique_times:
            # Collect relevant events that occurred at each time
            red_events = [event for event in relevant_events if event[0] == time]
            # Sort them according to the priority in the shape tuple
            sorted_red_events = sorted(red_events, key=lambda x: shape_tup.index(x[-2]))
            # Add the tail of the list to the removal_list
            removal_events += sorted_red_events[1:]

    for event in removal_events:
        match_events.remove(event)

    return(match_events)


def calc_occ(ins, mf, num_init_flag, num_int_flag=0, num_chan_flag=0):
    """
    Calculates the fraction of times an event was caught by the flagger for
    each type of event. Does not take care of frequency broadcasted events.

    Args:
        ins: The flagged incoherent noise spectrum in question
        mf: The match filter used to flag the INS
        num_init_flag: The number of initially flagged samples
        num_int_flag: The number of fully flagged integrations in the starting flags
        num_chan_flag: The number of channels fully flagged before the match filter.

    Returns:
        occ_dict: A dictionary with shapes for keys and occupancy fractions for values
    """
    occ_dict = {shape: 0. for shape in mf.slice_dict}

    for shape in occ_dict:
        relevant_events = [event for event in ins.match_events if shape in event.shape]
        num_time_event = sum([(event.time_slice.stop - event.time_slice.start) for event in relevant_events])
        occ_dict[shape] = num_time_event / (ins.Ntimes - num_int_flag)
        if shape == "narrow":
            occ_dict[shape] /= (ins.Nfreqs - num_chan_flag)

    # Figure out the total occupancy sans initial flags
    total_data = np.prod(ins.metric_array.shape)
    total_valid = total_data - num_init_flag
    total_flag = np.sum(ins.metric_array.mask)
    total_RFI = total_flag - num_init_flag
    total_occ = total_RFI / total_valid
    occ_dict['total'] = total_occ

    return(occ_dict)


def make_obslist(obsfile):
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
