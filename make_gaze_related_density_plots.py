#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:47:50 2024

@author: pg496
"""

import os
import numpy as np
import pdb
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
from tqdm import tqdm


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
        if file.endswith('.mat') and 'dot' not in file:
            mat_files.append(os.path.join(combined_path, file))
    mat_files.sort(key=lambda x: (extract_session_date(x), extract_run_number(x)))
    return mat_files


def extract_pos_time(loaded_pos_file, loaded_time_file, loaded_rect_file):
    m1_pos = loaded_pos_file['aligned_position_file']['m1'][0][0]
    m2_pos = loaded_pos_file['aligned_position_file']['m2'][0][0]
    time_vec = loaded_time_file['time_file']['t'][0][0]
    rects_mat = loaded_rect_file['roi_rects'][0][0]['m1']
    rects_m1 = [];
    for i in range(len(rects_mat)):
        rects_m1.append(rects_mat[i][1].squeeze())
    rects_mat = loaded_rect_file['roi_rects'][0][0]['m2']
    rects_m2 = [];
    for i in range(len(rects_mat)):
        rects_m2.append(rects_mat[i][1].squeeze())
    # pdb.set_trace()
    # Remove NaN values from time_vec and corresponding columns in m1_pos and m2_pos
    time_vec_cleaned = time_vec[~np.isnan(time_vec)]
    m1_pos_cleaned = m1_pos[:, ~np.isnan(time_vec.T)[0]]
    m2_pos_cleaned = m2_pos[:, ~np.isnan(time_vec.T)[0]]
    # Remove NaN values from m1_pos_cleaned
    valid_indices_m1 = ~np.isnan(m1_pos_cleaned).any(axis=0)
    # Remove NaN values from m2_pos_cleaned
    valid_indices_m2 = ~np.isnan(m2_pos_cleaned).any(axis=0)
    m1_pos_cleaned = m1_pos_cleaned[:, valid_indices_m1 & valid_indices_m2]
    m2_pos_cleaned = m2_pos_cleaned[:, valid_indices_m1 & valid_indices_m2]
    time_vec_cleaned = time_vec_cleaned[valid_indices_m1 & valid_indices_m2]
    return m1_pos_cleaned, m2_pos_cleaned, time_vec_cleaned, rects_m1, rects_m2


def plot_and_save_gaze_heatmaps(m1_pos_cleaned, m2_pos_cleaned, rects_m1, rects_m2, session, run_number, pos_file, plot_root, plot_dir_name):
    # Create 2D histogram for m1
    heatmap_m1, xedges_m1, yedges_m1 = \
        np.histogram2d(m1_pos_cleaned[0], m1_pos_cleaned[1], bins=100)
    # Create 2D histogram for m2
    heatmap_m2, xedges_m2, yedges_m2 = \
        np.histogram2d(m2_pos_cleaned[0], m2_pos_cleaned[1], bins=100)
    # Plot subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # Plot heatmap for m1
    img1 = axs[0].imshow( 
        heatmap_m1.T,
        extent=[xedges_m1[0], xedges_m1[-1], yedges_m1[0], yedges_m1[-1]],
        origin='lower')
    axs[0].set_title('m1_pos_cleaned')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].grid(False)
    axs[0].set_aspect('equal')
    axs[0].invert_yaxis()  # Invert y-axis direction
    # Plot rectangles for m1
    for rect in rects_m1:
        x1, y1, x2, y2 = rect
        width = x2 - x1
        height = y2 - y1
        rect_patch = patches.Rectangle((x1, y1), width, height, edgecolor='r', facecolor='none')
        axs[0].add_patch(rect_patch)
    # Plot heatmap for m2
    img2 = axs[1].imshow(
        heatmap_m2.T,
        extent=[xedges_m2[0], xedges_m2[-1], yedges_m2[0], yedges_m2[-1]],
        origin='lower')
    axs[1].set_title('m2_pos_cleaned')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    axs[1].grid(False)
    axs[1].set_aspect('equal')
    axs[1].invert_yaxis()  # Invert y-axis direction
    # Plot rectangles for m2
    for rect in rects_m2:
        x1, y1, x2, y2 = rect
        width = x2 - x1
        height = y2 - y1
        rect_patch = patches.Rectangle((x1, y1), width, height, edgecolor='r', facecolor='none')
        axs[1].add_patch(rect_patch)
    # Set super-title
    super_title = f"Session: {session.strftime('%Y-%m-%d')} - Run: {run_number}"
    fig.suptitle(super_title, fontsize=14)
    # Save plot
    plot_name = os.path.basename(pos_file).replace('.mat', '.png')
    session_folder = session.strftime('%Y-%m-%d')
    save_dir = os.path.join(plot_root, plot_dir_name, session_folder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, plot_name))
    plt.close(fig)  # Close the figure to release memory

##########
## MAIN ##
##########

behav_root = "/gpfs/milgram/project/chang/pg496/data_dir/social_gaze/social_gaze_eyetracking"
pos_subfolder_path = 'aligned_raw_samples/position'
roi_rects_subfolder_path = 'roi_rect_tables'
time_subfolder_path = 'aligned_raw_samples/time'
pupil_subfolder_path = 'pupil_size'
plot_root = "/gpfs/milgram/project/chang/pg496/data_dir/social_gaze/plots"
plot_dir_name = "gaze_loc_heatmaps"

sorted_time_files = list_mat_files_sorted(behav_root, time_subfolder_path)
sorted_pos_files = list_mat_files_sorted(behav_root, pos_subfolder_path)
sorted_rect_files = list_mat_files_sorted(behav_root, roi_rects_subfolder_path)
sorted_rect_files = [file for file in sorted_rect_files if os.path.basename(file) in [os.path.basename(file) for file in sorted_time_files]]
sorted_pupil_files = list_mat_files_sorted(behav_root, pupil_subfolder_path)
assert [os.path.basename(file) for file in sorted_time_files] == [os.path.basename(file) for file in sorted_pos_files]
assert [os.path.basename(file) for file in sorted_time_files] == [os.path.basename(file) for file in sorted_rect_files]
assert [os.path.basename(file) for file in sorted_time_files] == [os.path.basename(file) for file in sorted_pupil_files]
for pos_file, time_file, rect_file in \
    tqdm(zip(sorted_pos_files, sorted_time_files, sorted_rect_files),
         total=len(sorted_pos_files)):
    loaded_pos_file = scipy.io.loadmat(pos_file)
    loaded_time_file = scipy.io.loadmat(time_file)
    loaded_rect_file = scipy.io.loadmat(rect_file)
    session = extract_session_date(pos_file)
    run_number = extract_run_number(pos_file)
    try:
        m1_pos_cleaned, m2_pos_cleaned, time_vec_cleaned, rects_m1, rects_m2 = \
            extract_pos_time(loaded_pos_file, loaded_time_file, loaded_rect_file)
        plot_and_save_gaze_heatmaps(
            m1_pos_cleaned, m2_pos_cleaned, rects_m1, rects_m2, session, run_number,
            pos_file, plot_root, plot_dir_name)
    except:
        continue
plt.close('all')

