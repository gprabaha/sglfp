#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:47:30 2024

@author: pg496
"""

import os
import numpy as np
import scipy
from tqdm import tqdm

import util
import fix

import pdb


def sort_and_match_gaze_files(args):
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
    behav_root, time_subfolder, pos_subfolder, pupil_subfolder, roi_rects_subfolder = args
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
    return (sorted_time_files, sorted_pos_files, sorted_pupil_files, sorted_rect_files)


def use_roi_to_create_frame_and_crop_pos_time(args):
    """
    Use regions of interest (ROIs) to create frames and crop position-time data accordingly.
    Args:
    - args (tuple): A tuple containing the following elements in order:
        - m1_pos_cleaned (list): Cleaned position data for M1.
        - m2_pos_cleaned (list): Cleaned position data for M2.
        - time_vec_cleaned (list): Cleaned time vector.
        - m1_pupil_cleaned (list): Cleaned pupil data for M1.
        - m2_pupil_cleaned (list): Cleaned pupil data for M2.
        - rects_m1 (numpy.ndarray): Detected rectangles for M1.
        - rects_m2 (numpy.ndarray): Detected rectangles for M2.
        - stretch_factor (float): Stretch factor for frame adjustment.
    Returns:
    - tuple: A tuple containing the following elements in order:
        - m1_pos_within_frame (np.array): Position data for M1 within M1 frame.
        - m1_time_within_frame (np.array): Time vector for M1 within M1 frame.
        - m1_pupil_within_frame (np.array): Pupil data for M1 within M1 frame.
        - rects_m1 (numpy.ndarray): Detected rectangles for M1.
        - m1_rois (list): List of regions of interest for M1.
        - m2_pos_within_frame (np.array): Position data for M2 within M2 frame.
        - m2_time_within_frame (np.array): Time vector for M2 within M2 frame.
        - m2_pupil_within_frame (np.array): Pupil data for M2 within M2 frame.
        - rects_m2 (numpy.ndarray): Detected rectangles for M2.
        - m2_rois (list): List of regions of interest for M2.
    """
    # Unpack input arguments
    m1_pos_cleaned, m2_pos_cleaned, time_vec_cleaned, m1_pupil_cleaned, \
        m2_pupil_cleaned, rects_m1, rects_m2, stretch_factor = args
    # Get regions of interest for M1 and M2
    m1_rois = rects_m1.dtype.names
    m2_rois = rects_m2.dtype.names
    # Get frame and scaling information for M1
    m1_frame, m1_scale = util.get_frame_rect_and_scales_for_m1(
        rects_m1, m1_rois, stretch_factor)
    # Get frame for M2 using M1 scaling
    m2_frame = util.get_frame_for_m2(rects_m2, m2_rois, m1_scale, stretch_factor)
    # Filter M1 position data within M1 frame
    m1_pos_within_frame, m1_time_within_frame, m1_pupil_within_frame = \
        util.filter_positions_within_frame(m1_pos_cleaned, time_vec_cleaned, 
                                           m1_pupil_cleaned, m1_frame)
    # Filter M2 position data within M2 frame
    m2_pos_within_frame, m2_time_within_frame, m2_pupil_within_frame = \
        util.filter_positions_within_frame(m2_pos_cleaned, time_vec_cleaned,
                                           m2_pupil_cleaned, m2_frame)
    # Return the cropped data and relevant information
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
    m1_pos_cleaned = np.array(m1_pos_cleaned[:, valid_indices_m1 & valid_indices_m2].T)
    m2_pos_cleaned = np.array(m2_pos_cleaned[:, valid_indices_m1 & valid_indices_m2].T)
    m1_pupil_cleaned = np.array(m1_pupil_cleaned[valid_indices_m1 & valid_indices_m2])
    m2_pupil_cleaned = np.array(m2_pupil_cleaned[valid_indices_m1 & valid_indices_m2])
    time_vec_cleaned = np.array(time_vec_cleaned[valid_indices_m1 & valid_indices_m2])
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
    

def group_files_by_session(ordered_gaze_files):
    sorted_time_files, sorted_pos_files, sorted_pupil_files, sorted_rect_files \
        = ordered_gaze_files
    # Ensure that filenames in each list are exactly the same and in the same order
    filenames_time = [os.path.basename(file) for file in sorted_time_files]
    filenames_pos = [os.path.basename(file) for file in sorted_pos_files]
    filenames_pupil = [os.path.basename(file) for file in sorted_pupil_files]
    filenames_rect = [os.path.basename(file) for file in sorted_rect_files]
    if filenames_time != filenames_pos or filenames_time != filenames_pupil or filenames_time != filenames_rect:
        raise ValueError("Filenames in the lists are not the same or not in the same order")
    # Group files by session id
    sessions = {}
    for file_path in sorted_time_files + sorted_pos_files + sorted_pupil_files + sorted_rect_files:
        filename = os.path.basename(file_path)
        session_id = filename.split('_')[0]  # Assuming session id is the part before the first underscore
        if session_id not in sessions:
            sessions[session_id] = [[], [], [], []]  # 4 lists for each file group
        if file_path in sorted_time_files:
            sessions[session_id][0].append(file_path)
        elif file_path in sorted_pos_files:
            sessions[session_id][1].append(file_path)
        elif file_path in sorted_pupil_files:
            sessions[session_id][2].append(file_path)
        elif file_path in sorted_rect_files:
            sessions[session_id][3].append(file_path)
    # Convert the dictionary to a list of tuples
    session_files = [tuple(session_data) for session_data in sessions.values()]
    return session_files


def get_pos_time_pupil_fix_and_rois_within_session(file_tuple, stretch_factor, sampling_rate):
    time_files, pos_files, pupil_files, rect_files = file_tuple
    m1_pos_in_session = np.empty((0,2))
    m2_pos_in_session = np.empty((0,2))
    m1_pupil_in_session = []
    m2_pupil_in_session = []
    m1_fix_in_session = []
    m2_fix_in_session = []
    time_in_session = np.empty((0,1))
    m1_roi_rects = []
    m2_roi_rects = []
    
    # Extract session here and return it
    
    for i, (time_file, pos_file, pupil_file, rect_file) in \
        enumerate(
            tqdm(
                zip(time_files, pos_files, pupil_files, rect_files),
                total=len(time_files), desc="Processing file in session:"),
            1):
        loaded_pos_file = scipy.io.loadmat(pos_file)
        loaded_time_file = scipy.io.loadmat(time_file)
        loaded_rect_file = scipy.io.loadmat(rect_file)
        loaded_pupil_file = scipy.io.loadmat(pupil_file)
        nan_removed_files = remove_nans_in_pos_time(
            loaded_pos_file, loaded_time_file, loaded_rect_file, loaded_pupil_file)
        # Unpack cleaned data
        m1_pos_cleaned, m2_pos_cleaned, time_vec_cleaned, m1_pupil_cleaned, \
            m2_pupil_cleaned, rects_m1, rects_m2 = nan_removed_files
        # Find fixations here
        m1_fix = fix.is_fixation(util.px2deg(m1_pos_cleaned), time_vec_cleaned, sampling_rate=sampling_rate)
        m2_fix = fix.is_fixation(util.px2deg(m2_pos_cleaned), time_vec_cleaned, sampling_rate=sampling_rate)
        # Update arrays
        m1_pos_in_session = np.concatenate((m1_pos_in_session, np.array(m1_pos_cleaned)), axis=0)
        m2_pos_in_session = np.concatenate((m2_pos_in_session, np.array(m2_pos_cleaned)), axis=0)
        m1_fix_in_session = np.concatenate((m1_fix_in_session, np.array(m1_fix)), axis=0)
        m2_fix_in_session = np.concatenate((m2_fix_in_session, np.array(m2_fix)), axis=0)
        m1_pupil_in_session = np.concatenate((m1_pupil_in_session, np.array(m1_pupil_cleaned)), axis=0)
        m2_pupil_in_session = np.concatenate((m2_pupil_in_session, np.array(m2_pupil_cleaned)), axis=0)
        time_in_session = np.concatenate((time_in_session, np.array(time_vec_cleaned)), axis=0)
        if len(m1_roi_rects) == 0:
            m1_roi_rects = rects_m1
        if len(m2_roi_rects) == 0:
            m2_roi_rects = rects_m2
    m1_rois = m1_roi_rects.dtype.names
    m2_rois = m2_roi_rects.dtype.names
    # Get frame and scaling information for M1
    m1_frame, m1_scale = util.get_frame_rect_and_scales_for_m1(
        m1_roi_rects, m1_rois, stretch_factor)
    # Get frame for M2 using M1 scaling
    m2_frame = util.get_frame_for_m2(m2_roi_rects, m2_rois, m1_scale, stretch_factor)
    
    m1_data_within_frame = util.filter_data_within_frame(
        (m1_pos_in_session, time_in_session, m1_pupil_in_session, m1_fix_in_session),
        m1_frame)
    # Filter M2 position data within M2 frame
    m2_data_within_frame = util.filter_data_within_frame(
        (m2_pos_in_session, time_in_session, m2_pupil_in_session, m2_fix_in_session),
        m2_frame)
    
    # All fixation values are coming to be 0 now. Something needs fixing
    pdb.set_trace()
    # Return the cropped data and relevant information
    return m1_data_within_frame, rects_m1, m1_rois, m2_data_within_frame, rects_m2, m2_rois


'''
## FIXATIONS ##

def fixation_detection(data, t1, t2, minDur, s):
    n = len(data)
    if n == 0:
        return []  # Return empty list if data is empty
    fixations = np.zeros((n, 4))  # Initialize fixations array
    # Spatial clustering
    fixid = 1
    mx, my, d = 0, 0, 0
    fixpointer = 1
    for i in range(n):
        mx = np.mean(data[fixpointer:i+1, 0])
        my = np.mean(data[fixpointer:i+1, 1])
        d = distance2p(mx, my, data[i, 0], data[i, 1])
        if d > t1:
            fixid += 1
            fixpointer = i
        fixations[i, 3] = fixid
    # Temporal filtering
    number_fixations = fixations[-1, 3]
    fixation_list = []
    for i in range(1, int(number_fixations) + 1):
        centerx_t2, centery_t2, n_t1_t2, n_t2, t1_t2, t2_t2, d_t2, out_points = fixations_t2(fixations, i, t2)
        fixation_list.append([centerx_t2, centery_t2, n_t1_t2, n_t2, t1_t2, t2_t2, d_t2])
    # Duration thresholding
    fixation_list = min_duration(fixation_list, minDur)
    # Final output
    fix_ranges = []
    for fix in fixation_list:
        s_ind = np.where(data[:, 2] == fix[4])[0][0]
        e_ind = np.where(data[:, 2] == fix[5])[0][-1]
        fix_ranges.append([s_ind, e_ind])
    return fix_ranges


def distance2p(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def fixations_t2(fixations, fixation_id, t2):
    fixations_id = fixations[fixations[:, 3] == fixation_id]
    number_t1 = len(fixations_id)
    # Clustering according to criterion t2
    fixx, fixy = np.mean(fixations_id[:, :2], axis=0)
    for i in range(number_t1):
        d = distance2p(fixx, fixy, fixations_id[i, 0], fixations_id[i, 1])
        if d > t2:
            fixations_id[i, 3] = 0
    # Initialize lists
    fixations_list_t2 = np.empty((0, 4))  # Initialize fixations_list_t2 as an empty 2D array
    list_out_points = np.empty((0, 4))  # Initialize list_out_points as an empty 2D array
    for i in range(number_t1):
        if fixations_id[i, 3] > 0:
            fixations_list_t2 = np.vstack((fixations_list_t2, fixations_id[i, :]))
        else:
            list_out_points = np.vstack((list_out_points, fixations_id[i, :]))
    # Compute number of t2 fixations
    number_t2 = fixations_list_t2.shape[0]
    fixx, fixy = np.mean(fixations_list_t2[:, :2], axis=0)
    if number_t2 > 0:
        start_time = fixations_list_t2[0, 2]
        end_time = fixations_list_t2[-1, 2]
        duration = end_time - start_time
    else:
        start_time, end_time, duration = 0, 0, 0
    return fixx, fixy, number_t1, number_t2, start_time, end_time, duration, list_out_points


def min_duration(fixation_list, minDur):
    return [fix for fix in fixation_list if fix[6] >= minDur]


def is_fixation(pos, time, t1=None, t2=None, minDur=None, sampling_rate=None):
    # Combine position and time into a single data matrix
    data = np.column_stack((pos, time))
    # Calculate sampling rate if not provided
    if sampling_rate is None:
        sampling_rate = 1 / (time[1,:] - time[0,:])
    # Set default values
    if minDur is None:
        minDur = 0.01
    if t2 is None:
        t2 = 15
    if t1 is None:
        t1 = 30
    # Add NaN padding based on sampling rate
    dt = 1 / sampling_rate
    # Initialize fix_vector
    fix_vector = np.zeros(data.shape[0])
    # Find segments based on NaN or time interval
    diff_time = np.diff(time, axis=0)
    start_idc = np.where(diff_time > dt)[0]  # Find indices where time interval is greater than dt
    # Include the first data point index
    if start_idc[0] != 0:
        start_idc = np.insert(start_idc, 0, 0)
    # Loop through segments
    for i_segment in range(len(start_idc)):
        start_idx = start_idc[i_segment]
        end_idx = start_idc[i_segment + 1] - 1 if i_segment + 1 < len(start_idc) else -1
        # Extract segment data
        segment_data = data[start_idx:end_idx + 1, :]
        # Call fixation_detection function on segment data
        t_ind = fixation_detection(segment_data, t1, t2, minDur, start_idx)
        # Mark fixations in fix_vector
        for t_range in t_ind:
            fix_vector[t_range[0]:t_range[1] + 1] = 1
    return fix_vector
'''


