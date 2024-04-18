#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:47:50 2024

@author: pg496
"""

import os
import pdb
import scipy.io
from tqdm import tqdm

import util
import proc_behav
import plotter


##########
## MAIN ##
##########

behav_root = "/gpfs/milgram/project/chang/pg496/data_dir/social_gaze/social_gaze_eyetracking"
pos_subfolder_path = 'aligned_raw_samples/position'
roi_rects_subfolder_path = 'roi_rect_tables'
time_subfolder_path = 'aligned_raw_samples/time'
pupil_subfolder_path = 'aligned_raw_samples/pupil_size'
plot_root = "/gpfs/milgram/project/chang/pg496/data_dir/social_gaze/plots"
plot_dir_name = "pupil_heatmaps"

sorted_time_files = util.list_mat_files_sorted(
    behav_root, time_subfolder_path)
sorted_pos_files = util.list_mat_files_sorted(
    behav_root, pos_subfolder_path)
sorted_pupil_files = util.list_mat_files_sorted(
    behav_root, pupil_subfolder_path)
sorted_rect_files = util.list_mat_files_sorted(
    behav_root, roi_rects_subfolder_path)
sorted_rect_files = [file for file in sorted_rect_files \
                     if os.path.basename(file) in [os.path.basename(file) \
                     for file in sorted_time_files]]
assert [os.path.basename(file) for file in sorted_time_files] == \
    [os.path.basename(file) for file in sorted_pos_files]
assert [os.path.basename(file) for file in sorted_time_files] == \
    [os.path.basename(file) for file in sorted_rect_files]
assert [os.path.basename(file) for file in sorted_time_files] == \
    [os.path.basename(file) for file in sorted_pupil_files]
    
for pos_file, time_file, rect_file, pupil_file in \
    tqdm(zip(
        sorted_pos_files, sorted_time_files, sorted_rect_files, sorted_pupil_files),
        total=len(sorted_pos_files)):
    loaded_pos_file = scipy.io.loadmat(pos_file)
    loaded_time_file = scipy.io.loadmat(time_file)
    loaded_rect_file = scipy.io.loadmat(rect_file)
    loaded_pupil_file = scipy.io.loadmat(pupil_file)
    session = util.extract_session_date(pos_file)
    run_number = util.extract_run_number(pos_file)
    try:
        m1_pos_cleaned, m2_pos_cleaned, time_vec_cleaned, \
            m1_pupil_cleaned, m2_pupil_cleaned, rects_m1, rects_m2 = \
                proc_behav.extract_pos_time(
                    loaded_pos_file, loaded_time_file,
                    loaded_rect_file, loaded_pupil_file)
        plotter.plot_mean_pupil_size_distribution(
            m1_pos_cleaned, m2_pos_cleaned, rects_m1, rects_m2,
            m1_pupil_cleaned, m2_pupil_cleaned, session, run_number,
            pos_file, plot_root, plot_dir_name)
    except:
        continue

