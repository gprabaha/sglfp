#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:46:08 2024

@author: pg496
"""

import os
from datetime import datetime
from tqdm import tqdm  # Importing tqdm for progress bar
import multiprocessing
import numpy as np
import math

import pdb

def fetch_monitor_info():
    return {'height': 27, 'distance': 50, 'vertical_resolution': 768}


def px2deg(px, monitor_info=None):
    if monitor_info is None:
        monitor_info = fetch_monitor_info() # in defaults
    h = monitor_info['height']
    d = monitor_info['distance']
    r = monitor_info['vertical_resolution']
    deg_per_px = math.degrees(math.atan2(0.5 * h, d)) / (0.5 * r)
    deg = px * deg_per_px
    return deg


def filter_data_within_frame(data_tuple, frame):
    """
    Filter positions, time, and pupil data within a specified frame.
    Args:
    - data_tuple (tuple): Tuple containing positions, time, pupil, and additional variables.
    - frame (list): List containing frame coordinates [x1, y1, x2, y2].
    Returns:
    - filtered_data (tuple): Tuple containing filtered positions and the rest of the elements
                             of the input tuple filtered using positions.
    """
    positions, *additional_variables = data_tuple
    x1, y1, x2, y2 = frame
    # Filter data within the specified frame
    filtered_data = [(pos, *additional) for pos, *additional in zip(positions, *additional_variables)
                     if x1 <= pos[0] <= x2 and y1 <= pos[1] <= y2]
    # Unzip the filtered data
    filtered_positions, *filtered_additional = zip(*filtered_data)
    # Convert to NumPy arrays
    filtered_positions = np.array(filtered_positions)
    # Convert filtered additional variables to NumPy arrays
    filtered_additional = [np.array(var) for var in filtered_additional]
    # Construct the filtered data tuple
    filtered_data = (filtered_positions, *filtered_additional)
    return filtered_data


def find_center(rect):
    """
    Find the center point of a rectangle.

    Args:
    - rect (list): List containing rectangle coordinates [x1, y1, x2, y2].

    Returns:
    - center (list): List containing the center coordinates [x_center, y_center].
    """
    x1, y1, x2, y2 = rect
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    return [x_center, y_center]


def get_frame_rect_and_scales_for_m1(rects_m1, m1_rois, stretch_factor):
    """
    Get frame rectangle and scales for M1 based on object boundaries.

    Args:
    - rects_m1 (dict): Dictionary containing M1 rectangle data.
    - m1_rois (list): List of regions of interest for M1.
    - stretch_factor (float): Scaling factor.

    Returns:
    - frame_rect (list): List containing frame rectangle coordinates [x_left, y_top, x_right, y_bottom].
    - scales (list): List containing scaling factors [x_scale, y_scale].
    """
    if ('left_nonsocial_object' in m1_rois) and ('right_nonsocial_object' in m1_rois) and ('face' in m1_rois):
        l_obj_rect = rects_m1['left_nonsocial_object'][0][0]
        r_obj_rect = rects_m1['right_nonsocial_object'][0][0]
        face_rect = rects_m1['face'][0][0]
        face_center = find_center(face_rect)
        y_top = min(l_obj_rect[1], r_obj_rect[1])
        y_scale = face_center[1] - y_top
        y_bottom = face_center[1] + stretch_factor * y_scale * 1.2
        y_top = face_center[1] - stretch_factor * y_scale * 1.2
        x_scale = max(face_center[0] - l_obj_rect[0], r_obj_rect[0] - face_center[0])
        x_left = face_center[0] - stretch_factor * x_scale
        x_right = face_center[0] + stretch_factor * x_scale
        frame_rect = [x_left, y_top, x_right, y_bottom]
        scales = [x_scale, y_scale]
        return frame_rect, scales
    else:
        return None, None
    

def get_frame_for_m2(rects_m2, m2_rois, m1_scale, stretch_factor):
    """
    Get frame rectangle for M2 based on M1's scale.
    Args:
    - rects_m2 (dict): Dictionary containing M2 rectangle data.
    - m2_rois (list): List of regions of interest for M2.
    - m1_scale (list): List containing scaling factors from M1 [x_scale, y_scale].
    - stretch_factor (float): Scaling factor.
    Returns:
    - frame_rect (list): List containing frame rectangle coordinates [x_left, y_top, x_right, y_bottom].
    """
    if 'face' in m2_rois:
        face_rect = rects_m2['face'][0][0]
        face_center = find_center(face_rect)
        x_left = face_center[0] - stretch_factor * m1_scale[0]
        y_top = face_center[1] - stretch_factor * m1_scale[1] * 1.2
        x_right = face_center[0] + stretch_factor * m1_scale[0]
        y_bottom = face_center[1] + stretch_factor * m1_scale[1] * 1.2
        frame_rect = [x_left, y_top, x_right, y_bottom]
        return frame_rect
    else:
        return None



def match_with_time_files(sorted_rect_files, sorted_time_files):
    """
    Filter rectangle files matching time files by basename.
    Args:
    - sorted_rect_files (list): Sorted rectangle file paths.
    - sorted_time_files (list): Sorted time file paths.
    Returns:
    - filtered_files (list): Filtered rectangle file paths.
    """
    filtered_files = [file for file in sorted_rect_files
                      if os.path.basename(file) in [os.path.basename(f)
                                                    for f in sorted_time_files]]
    return filtered_files


def extract_session_date(filepath):
    """
    Extract the session date from the filename.
    Args:
    - filepath (str): Path of the file containing the session date.
    Returns:
    - session_date (datetime): The session date extracted from the filename.
    """
    filename = os.path.basename(filepath)
    session_date_str = filename[:8]  # Extract the first 8 characters representing the date
    return datetime.strptime(session_date_str, '%m%d%Y')


def extract_run_number(filename):
    """
    Extract the run number from the filename.
    Args:
    - filename (str): Name of the file containing the run number.
    Returns:
    - run_number (int): The run number extracted from the filename.
    """
    parts = filename.split('_')
    if len(parts) < 3:  # Check if the filename has required parts
        return 0
    run_number_str = parts[-1].split('.')[0]  # Extract run number from the last part of the filename
    return int(run_number_str)


def list_mat_files_sorted(behav_root, rel_subfolder_path):
    """
    List .mat files sorted by session date and run number.
    Args:
    - behav_root (str): Root directory containing the behavioral data.
    - rel_subfolder_path (str): Relative path to the subfolder containing .mat files.

    Returns:
    - mat_files (list): List of .mat files sorted by session date and run number.
    """
    combined_path = os.path.join(behav_root, rel_subfolder_path)
    mat_files = []
    files = os.listdir(combined_path)
    for file in files:
        if file.endswith('.mat') and 'dot' not in file:  # Filter out files not ending with .mat and containing 'dot'
            mat_files.append(os.path.join(combined_path, file))
    mat_files.sort(key=lambda x: (extract_session_date(x), extract_run_number(x)))  # Sort files by session date and run number
    return mat_files


def run_parallel_function(function, args_list, num_processes=None):
    """
    Run a function in parallel with multiprocessing.
    Args:
    - function: The function to run in parallel.
    - args_list (list): List of arguments to pass to the function for each parallel process.
    - num_processes (int): Number of processes to run in parallel. If None, it defaults to the number of CPU cores.
    Returns:
    - result_list (list): List of results returned by the function for each parallel process.
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()  # Get the number of CPU cores
    with multiprocessing.Pool(processes=num_processes) as pool:
        result_list = list(tqdm(pool.imap_unordered(function, args_list), total=len(args_list)))  # Run the function in parallel using multiprocessing.Pool
    return result_list


def create_rects_dict(rects_array):
    rects_dict = {}
    for field_name in rects_array.dtype.names:
        rects_dict[field_name] = [rects_array[field_name][0][0] for rects_array[field_name] in rects_array]
    return rects_dict
    
    
    
    
    