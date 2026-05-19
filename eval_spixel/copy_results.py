"""
Code modified from https://github.com/fuy34/superpixel_fcn/blob/master/eval_spixel/copy_resCSV.py
"""

import shutil
import os
import argparse
import sys

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str,
                        help="source dir the output of the benchmark code", required=True)
    parser.add_argument("--dst_dir", type=str,
                        help="the final results for plotting", required=True)
    parser.add_argument("--data_name", type=str,
                        help="which data to use", required=True)

    cmdargs = parser.parse_args()

    src = cmdargs.src_dir
    dst = cmdargs.dst_dir
    data_name = cmdargs.data_name

    if data_name == 'bsd':
        splist = ["54", "96", "150" ,"216" ,"294", "384","486", "600", "726" ,"864", "1014", "1176", "1350"]
    elif data_name == 'nyu':
        splist = ["300", "432", "588", "768", "972", "1200", "1452", "1728", "2028", "2352"]
    elif data_name == 'voc':
        splist = ["81", "144", "225", "324", "441", "576", "729", "900", "1089", "1296"]
    else:
        raise Exception("unknown data_name")
    for l in splist:
        src_pth = src + '/SPixelNet_nSpixel_' + l +'/map_csv/results.csv'
        dst_pth = dst + '/' + l
        if not os.path.isdir(dst_pth):
            os.makedirs(dst_pth)
        dst_path =dst_pth + '/results.csv'
        shutil.copy(src_pth, dst_path)

    sys.exit(0)