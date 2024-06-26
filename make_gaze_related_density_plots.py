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


def generate_fixation_density_plots_for_each_session(session_files, plot_root,
                                                     plot_dir_name, stretch_factor, n_bins, sampling_rate, do_parallel=False):
    if do_parallel:
        #pdb.set_trace()
        args_list = []
        for file_tuple in session_files:
            args_list.append((file_tuple, plot_root, plot_dir_name,
                              stretch_factor, n_bins, sampling_rate))
        #[(file_tuple, plot_root, plot_dir_name, stretch_factor, n_bins, sampling_rate) for file_tuple in session_files]
        util.run_parallel_without_progressbar(plotter.plot_gaze_fixation_and_pupil_heatmap_for_session, args_list)
    else:
        for file_tuple in session_files:
            plotter.plot_gaze_fixation_and_pupil_heatmap_for_session(file_tuple, plot_root,
                                                                     plot_dir_name, stretch_factor, n_bins, sampling_rate)





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
sampling_rate = 1e3

args_to_acquire_gaze_files = (behav_root, time_subfolder_path, pos_subfolder_path,
                              pupil_subfolder_path, roi_rects_subfolder_path)
ordered_gaze_files = proc_behav.sort_and_match_gaze_files(args_to_acquire_gaze_files)
'''
generate_pupil_distrubutions_parallel(ordered_gaze_files, plot_root,
                                      plot_dir_name, stretch_factor, n_bins)
'''

session_files = proc_behav.group_files_by_session(ordered_gaze_files)


generate_fixation_density_plots_for_each_session(session_files, plot_root,
                                      plot_dir_name, stretch_factor, n_bins, sampling_rate)
