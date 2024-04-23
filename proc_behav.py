#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:47:30 2024

@author: pg496
"""

import os
import numpy as np

import util

import pdb


def sort_and_match_gaze_files(behav_root, time_subfolder, pos_subfolder,
                              pupil_subfolder, roi_rects_subfolder):
    """
    Sort and match gaze files from subfolders.
    Args:
    - behav_root (str): Root directory of behavioral data.
    - time_subfolder (str): Subfolder containing time files.
    - pos_subfolder (str): Subfolder containing position files.
    - pupil_subfolder (str): Subfolder containing pupil files.
    - roi_rects_subfolder (str): Subfolder containing ROI rects files.
    Returns:
    - sorted_time_files (list): Sorted time file paths.
    - sorted_pos_files (list): Sorted position file paths.
    - sorted_pupil_files (list): Sorted pupil file paths.
    - sorted_rect_files (list): Sorted ROI rects file paths.
    """
    sorted_time_files = util.list_mat_files_sorted(behav_root, time_subfolder)
    sorted_pos_files = util.list_mat_files_sorted(behav_root, pos_subfolder)
    sorted_pupil_files = util.list_mat_files_sorted(behav_root, pupil_subfolder)
    sorted_rect_files = util.list_mat_files_sorted(behav_root, roi_rects_subfolder)
    sorted_rect_files = util.match_with_time_files(sorted_rect_files, sorted_time_files)
    # Assert basenames equality
    assert [os.path.basename(f) for f in sorted_time_files] == \
           [os.path.basename(f) for f in sorted_pos_files] == \
           [os.path.basename(f) for f in sorted_rect_files] == \
           [os.path.basename(f) for f in sorted_pupil_files]
    return sorted_time_files, sorted_pos_files, sorted_pupil_files, sorted_rect_files


def use_roi_to_create_frame_and_crop_pos_time(args):
    m1_pos_cleaned, m2_pos_cleaned, time_vec_cleaned, m1_pupil_cleaned, \
        m2_pupil_cleaned, rects_m1, rects_m2, stretch_factor = args
    m1_rois = rects_m1.dtype.names
    m2_rois = rects_m2.dtype.names
    m1_frame, m1_scale = util.get_frame_rect_and_scales_for_m1(
        rects_m1, m1_rois, stretch_factor)
    m2_frame = util.get_frame_for_m2(rects_m2, m2_rois, m1_scale)
    # Filter m1_pos_cleaned within m1_frame
    m1_pos_within_frame, m1_time_within_frame, m1_pupil_within_frame = \
        util.filter_positions_within_frame(m1_pos_cleaned, time_vec_cleaned, 
                                           m1_pupil_cleaned, m1_frame)
    # Filter m2_pos_cleaned within m2_frame
    m2_pos_within_frame, m2_time_within_frame, m2_pupil_within_frame = \
        util.filter_positions_within_frame(m2_pos_cleaned, time_vec_cleaned,
                                           m2_pupil_cleaned, m2_frame)
    return (m1_pos_within_frame, m1_time_within_frame, m1_pupil_within_frame, rects_m1, m1_rois,
            m2_pos_within_frame, m2_time_within_frame, m2_pupil_within_frame, rects_m2, m2_rois)



def remove_nans_in_pos_time(loaded_pos_file, loaded_time_file,
                     loaded_rect_file, loaded_pupil_file):
    """
    Extract position, time, pupil, and ROI rects data.
    Args:
    - loaded_pos_file: Loaded position file.
    - loaded_time_file: Loaded time file.
    - loaded_rect_file: Loaded ROI rects file.
    - loaded_pupil_file: Loaded pupil file.
    Returns:
    - m1_pos_cleaned (numpy.ndarray): Cleaned position data for m1.
    - m2_pos_cleaned (numpy.ndarray): Cleaned position data for m2.
    - time_vec_cleaned (numpy.ndarray): Cleaned time vector.
    - m1_pupil_cleaned (numpy.ndarray): Cleaned pupil data for m1.
    - m2_pupil_cleaned (numpy.ndarray): Cleaned pupil data for m2.
    - rects_m1 (list): ROI rects for m1.
    - rects_m2 (list): ROI rects for m2.
    """
    m1_pos = loaded_pos_file['aligned_position_file']['m1'][0][0]
    m2_pos = loaded_pos_file['aligned_position_file']['m2'][0][0]
    time_vec = loaded_time_file['time_file']['t'][0][0]
    m1_pupil = loaded_pupil_file['var'][0][0]['m1'][0]
    m2_pupil = loaded_pupil_file['var'][0][0]['m2'][0]
    # Extract ROI rects
    rects_m1 = loaded_rect_file['roi_rects']['m1'][0][0][0]
    rects_m2 = loaded_rect_file['roi_rects']['m2'][0][0][0]
    # Remove NaN values
    time_vec_cleaned = time_vec[~np.isnan(time_vec).squeeze()]
    m1_pos_cleaned = m1_pos[:, ~np.isnan(time_vec.T)[0]]
    m2_pos_cleaned = m2_pos[:, ~np.isnan(time_vec.T)[0]]
    m1_pupil_cleaned = m1_pupil[~np.isnan(time_vec).squeeze()]
    m2_pupil_cleaned = m2_pupil[~np.isnan(time_vec).squeeze()]
    # Remove NaN values from position data
    valid_indices_m1 = ~np.isnan(m1_pos_cleaned).any(axis=0)
    valid_indices_m2 = ~np.isnan(m2_pos_cleaned).any(axis=0)
    m1_pos_cleaned = m1_pos_cleaned[:, valid_indices_m1 & valid_indices_m2].T
    m2_pos_cleaned = m2_pos_cleaned[:, valid_indices_m1 & valid_indices_m2].T
    m1_pupil_cleaned = m1_pupil_cleaned[valid_indices_m1 & valid_indices_m2]
    m2_pupil_cleaned = m2_pupil_cleaned[valid_indices_m1 & valid_indices_m2]
    time_vec_cleaned = time_vec_cleaned[valid_indices_m1 & valid_indices_m2]
    return (m1_pos_cleaned, m2_pos_cleaned, time_vec_cleaned,
           m1_pupil_cleaned, m2_pupil_cleaned, rects_m1, rects_m2)


def calculate_gaze_avg_pupil_size(pos_x, pos_y, pupil, bins):
    """
    Calculate average pupil size based on position.
    Args:
    - pos_x (numpy.ndarray): x-coordinates of gaze positions.
    - pos_y (numpy.ndarray): y-coordinates of gaze positions.
    - pupil (numpy.ndarray): Pupil size data.
    - bins (int or sequence of scalars): Bins for histogram.
    Returns:
    - heatmap (numpy.ndarray): Heatmap of gaze positions.
    - avg_pupil (numpy.ndarray): Average pupil size based on position.
    - xedges (numpy.ndarray): Bin edges along the x-axis.
    - yedges (numpy.ndarray): Bin edges along the y-axis.
    """
    heatmap, xedges, yedges = np.histogram2d(pos_x, pos_y, bins=bins)
    avg_pupil = np.zeros_like(heatmap)
    for i in range(len(xedges) - 1):
        for j in range(len(yedges) - 1):
            indices = np.where((pos_x >= xedges[i]) & (pos_x < xedges[i+1]) &
                               (pos_y >= yedges[j]) & (pos_y < yedges[j+1]))[0]
            if len(indices) > 0:
                avg_pupil[i, j] = np.mean(pupil[indices])
    return heatmap, avg_pupil, xedges, yedges
