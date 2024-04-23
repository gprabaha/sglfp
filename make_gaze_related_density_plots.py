#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:47:50 2024

@author: pg496
"""

import pdb

import util
import proc_behav
import plotter


def generate_pupil_distrubutions_parallel(ordered_gaze_files, plot_root,
                                          plot_dir_name, stretch_factor, n_bins):
    sorted_time_files, sorted_pos_files, sorted_pupil_files, sorted_rect_files \
        = ordered_gaze_files
    args_list = []
    for pos_file, time_file, rect_file, pupil_file in \
            zip(sorted_pos_files, sorted_time_files,
                sorted_rect_files, sorted_pupil_files):
        args_list.append((pos_file, time_file, rect_file, pupil_file,
                          plot_root, plot_dir_name, stretch_factor, n_bins))
    
    # Parallel
    #'''
    util.run_parallel_function(
        plotter.plot_pupil_dustribution_for_one_file, args_list)
    #'''
    
    # One serial iteration
    '''
    plotter.plot_pupil_dustribution_for_one_file(args_list[5])
    '''

##########
## MAIN ##
##########

behav_root = "/gpfs/milgram/project/chang/pg496/data_dir/social_gaze/social_gaze_eyetracking"
pos_subfolder_path = 'aligned_raw_samples/position'
roi_rects_subfolder_path = 'roi_rect_tables'
time_subfolder_path = 'aligned_raw_samples/time'
pupil_subfolder_path = 'aligned_raw_samples/pupil_size'
plot_root = "/gpfs/milgram/project/chang/pg496/data_dir/social_gaze/plots"
plot_dir_name = "pupil_heatmaps_cropped_by_roi"
stretch_factor = 1.2
n_bins = 100

args_to_acquire_gaze_files = (behav_root, time_subfolder_path, pos_subfolder_path,
                              pupil_subfolder_path, roi_rects_subfolder_path)
ordered_gaze_files = proc_behav.sort_and_match_gaze_files(args_to_acquire_gaze_files)
generate_pupil_distrubutions_parallel(ordered_gaze_files, plot_root,
                                      plot_dir_name, stretch_factor, n_bins)


