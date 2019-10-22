from SSINS import INS, util
import argparse
import csv


def csv_to_dict_list(csv_filepath):

    with open(csv_filepath, newline='') as csv_file:
        dict_reader = csv.DictReader(csv_file)
        dict_list = []
        for row in dict_reader:
            dict_list.append(row)

    return(dict_list)


def dict_list_to_csv(dict_list, outpath, fieldnames):

    with open(outpath, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames)

        writer.writeheader()
        for row in dict_list:
            writer.writerow(row)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--shapes', nargs='*', help='The shapes to calculate occupancies for')
    parser.add_argument('-c', '--csv', help='A csv whose columns are obsids, ins data files, ins masks, and yml files')
    parser.add_argument('-o', '--outfile', help='The name of the output csv that contains occupancy information')
    parser.add_argument('-i', '--ch_ignore', help='A text file of fine frequency channels to ignore in occupancy calculation')
    args = parser.parse_args()

    dict_list = csv_to_dict_list(args.csv)

    if args.ch_ignore is not None:
        fine_channels_ignore = util.make_obslist(args.ch_ignore)
        fine_channels_ignore = np.array(fine_channels_ignore).astype(int)
    else:
        fine_channels_ignore = None

    occ_dict_list = []
    for obs in dict_list:
        obsid = obs['obsid']
        ins = INS(obs['ins_file'], mask_file=obs['mask_file'],
                  match_events_file=obs['yml_file'])

        if fine_channels_ignore is not None:
            freq_inds = np.arange(len(ins.freq_array))
            freq_inds = np.delete(freq_inds, fine_channels_ignore)
        else:
            freq_inds = slice(None)

        total_occ = np.mean(ins.metric_array.mask[:, freq_inds, 0])

        obs_occ_dict = {'obsid': obsid, 'total_occ': total_occ}
        if args.shapes is not None:
            for shapes in args.shapes:
                obs_occ_dict[shape] = len([event for event in match_events if event[2] == shape]) / ins.metric_array.shape[0]

        occ_dict_list.append(obs_occ_dict)

    dict_list_to_csv(occ_dict_list, args.outfile, ['obsid', 'total_occ'] + [shape for shapes in args.shapes])
