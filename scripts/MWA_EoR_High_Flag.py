from SSINS import INS, util, Catalog_Plot, MF
import argparse
from astropy.time import Time
from matplotlib import cm
import numpy as np
from time import time

# note on performance: enabling the -p plots flag takes approximately an order
# longer than only -w writing the data files: for raw processing of data, disabling
# -p is recommended.

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Processes a list of SSINS data files into plots, CSV for occ_csv, and/or processed data.')
    parser.add_argument('-i', '--ins_file_list', help="A text file of SSINS h5 data files.")
    parser.add_argument('-o', '--outdir', help="The output directory for the files.")
    parser.add_argument('-w', '--write', action='store_true', help="Toggles creation of output data files.")
    parser.add_argument('-v', '--verbose', action='store_true', help="Toggles verbose console output of file processing progress.")
    parser.add_argument('-p', '--plots', action='store_true', help="Toggles creation of plots.")
    parser.add_argument('-g', '--gencsv', action='store_true', help="Toggles creation of CSV for occ_csv script.")
    args = parser.parse_args()

    ins_file_list = util.make_obslist(args.ins_file_list)

    if args.gencsv:
        f = open("occcsv.csv", "w") #wipe old file
        f.write("obsid,ins_file,mask_file,yml_file\n")#header
        f = open("occcsv.csv", "a")

    #time the full length of the run if -v passed
    if args.verbose:
        start = time()

    for ins_filepath in ins_file_list:
        slash_ind = ins_filepath.rfind('/')
        obsid = ins_filepath[slash_ind + 1: slash_ind + 11]

        ins = INS(ins_filepath)
        ins.select(times=ins.time_array[3:-3])
        ins.metric_ms = ins.mean_subtract()
        shape_dict = {'TV6': [1.74e8, 1.81e8],
                      'TV7': [1.81e8, 1.88e8],
                      'TV8': [1.88e8, 1.95e8],
                      'TV9': [1.95e8, 2.02e8]}
        sig_thresh = {shape: 5 for shape in shape_dict}
        sig_thresh['narrow'] = 5
        sig_thresh['streak'] = 8
        mf = MF(ins.freq_array, sig_thresh, shape_dict=shape_dict,
                N_samp_thresh=len(ins.time_array) // 2)

        ins.metric_array[ins.metric_array == 0] = np.ma.masked
        ins.metric_ms = ins.mean_subtract()
        ins.sig_array = np.ma.copy(ins.metric_ms)

        prefix = '%s/%s_trimmed_zeromask' % (args.outdir, obsid)
        freqs = np.arange(1.7e8, 2e8, 5e6)
        xticks, xticklabels = util.make_ticks_labels(freqs, ins.freq_array, sig_fig=0)
        yticks = [0, 20, 40]
        yticklabels = []
        for tick in yticks:
            yticklabels.append(Time(ins.time_array[tick], format='jd').iso[:-4])

        #write plots if command flagged to do so
        if args.plots:
            Catalog_Plot.INS_plot(ins, prefix, xticks=xticks, yticks=yticks,
                                  xticklabels=xticklabels, yticklabels=yticklabels,
                                  data_cmap=cm.plasma, ms_vmin=-5, ms_vmax=5,
                                  title=obsid, xlabel='Frequency (Mhz)', ylabel='Time (UTC)')
            if args.verbose:
                print("wrote trimmed zeromask for "+obsid)

        mf.apply_match_test(ins, apply_samp_thresh=False)
        mf.apply_samp_thresh_test(ins, event_record=True)

        flagged_prefix = '%s/%s_trimmed_zeromask_MF_s8' % (args.outdir, obsid)

        #write data/mask/match/ if command flagged to do so
        if args.write:
            ins.write(flagged_prefix, output_type='data', clobber=True)
            ins.write(flagged_prefix, output_type='mask', clobber=True)
            ins.write(flagged_prefix, output_type='match_events')
            if args.verbose:
                print("wrote data/mask/match files for "+obsid)

        #write plots if command flagged to do so
        if args.plots:
            Catalog_Plot.INS_plot(ins, flagged_prefix, xticks=xticks, yticks=yticks,
                                  xticklabels=xticklabels, yticklabels=yticklabels,
                                  data_cmap=cm.plasma, ms_vmin=-5, ms_vmax=5,
                                  title=obsid, xlabel='Frequency (Mhz)', ylabel='Time (UTC)')

            if args.verbose:
                print("wrote trimmed zeromask (w/ match filter) for "+obsid)

        #a hardcoded csv generator for occ_csv
        if args.gencsv:
            csv = ""+obsid+","+flagged_prefix+"_SSINS_data.h5,"+flagged_prefix+"_SSINS_mask.h5,"+flagged_prefix+"_SSINS_match_events.yml\n"
            f.write(csv)
            print("wrote entry for "+obsid)
            print(csv)

    #print out full length of run
    if args.verbose:
        print(f'Time taken: {time() - start}')
