#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:47:50 2024

@author: pg496
"""

import os
import numpy as np
import scipy.io
from datetime import datetime

def extract_session_date(filepath):
    filename = os.path.basename(filepath)
    session_date_str = filename[:8]
    return datetime.strptime(session_date_str, '%m%d%Y')

def extract_run_number(filename):
    parts = filename.split('_')
    if len(parts) < 3:
        return 0
    run_number_str = parts[-1].split('.')[0]
    return int(run_number_str)

def list_mat_files_sorted(behav_root, rel_subfolder_path):
    combined_path = os.path.join(behav_root, rel_subfolder_path)
    mat_files = []
    files = os.listdir(combined_path)
    for file in files:
        if file.endswith('.mat'):
            mat_files.append(os.path.join(combined_path, file))
    mat_files.sort(key=lambda x: (extract_session_date(x), extract_run_number(x)))
    return mat_files

# Example usage:
behav_root = "/gpfs/milgram/project/chang/pg496/data_dir/social_gaze/social_gaze_eyetracking"
pos_subfolder_path = 'aligned_raw_samples/position'
roi_bounds_subfolder_path = 'aligned_raw_samples/bounds'
time_subfolder_path = 'aligned_raw_samples/time'
pupil_subfolder_path = 'pupil_size'

sorted_pos_files = list_mat_files_sorted(behav_root, pos_subfolder_path)
sorted_bound_files = list_mat_files_sorted(behav_root, roi_bounds_subfolder_path)
sorted_time_files = list_mat_files_sorted(behav_root, time_subfolder_path)
sorted_pupil_files = list_mat_files_sorted(behav_root, pupil_subfolder_path)

assert [os.path.basename(file) for file in sorted_time_files] == [os.path.basename(file) for file in sorted_pos_files]
assert [os.path.basename(file) for file in sorted_time_files] == [os.path.basename(file) for file in sorted_bound_files]
assert [os.path.basename(file) for file in sorted_time_files] == [os.path.basename(file) for file in sorted_pupil_files]

for pos_file, time_file in zip(sorted_pos_files, sorted_time_files):
    loaded_pos_file = scipy.io.loadmat(pos_file)
    loaded_time_file = scipy.io.loadmat(time_file)
    session = extract_session_date(pos_file)
    run_number = extract_run_number(pos_file)
    
    m1_pos = loaded_pos_file['aligned_position_file']['m1'][0][0] 
    m2_pos = loaded_pos_file['aligned_position_file']['m2'][0][0]
    time_vec = loaded_time_file['time_file']['t'][0][0]
