#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:47:30 2024

@author: pg496
"""

import numpy as np
import pdb


def extract_pos_time(loaded_pos_file, loaded_time_file,
                     loaded_rect_file, loaded_pupil_file):
    m1_pos = loaded_pos_file['aligned_position_file']['m1'][0][0]
    m2_pos = loaded_pos_file['aligned_position_file']['m2'][0][0]
    time_vec = loaded_time_file['time_file']['t'][0][0]
    rects_mat_m1 = loaded_rect_file['roi_rects'][0][0]['m1']
    m1_pupil = loaded_pupil_file['var'][0][0]['m1'][0]
    m2_pupil = loaded_pupil_file['var'][0][0]['m2'][0]
    rects_m1 = [];
    for i in range(len(rects_mat_m1)):
        rects_m1.append(rects_mat_m1[i][1].squeeze())
    rects_mat_m2 = loaded_rect_file['roi_rects'][0][0]['m2']
    rects_m2 = [];
    for i in range(len(rects_mat_m2)):
        rects_m2.append(rects_mat_m2[i][1].squeeze())
    # Remove NaN values from time_vec and corresponding columns in m1_pos and m2_pos
    time_vec_cleaned = time_vec[~np.isnan(time_vec).squeeze()]
    m1_pos_cleaned = m1_pos[:, ~np.isnan(time_vec.T)[0]]
    m2_pos_cleaned = m2_pos[:, ~np.isnan(time_vec.T)[0]]
    m1_pupil_cleaned = m1_pupil[~np.isnan(time_vec).squeeze()]
    m2_pupil_cleaned = m2_pupil[~np.isnan(time_vec).squeeze()]
    # Remove NaN values from m1_pos_cleaned
    valid_indices_m1 = ~np.isnan(m1_pos_cleaned).any(axis=0)
    # Remove NaN values from m2_pos_cleaned
    valid_indices_m2 = ~np.isnan(m2_pos_cleaned).any(axis=0)
    m1_pos_cleaned = m1_pos_cleaned[:, valid_indices_m1 & valid_indices_m2]
    m2_pos_cleaned = m2_pos_cleaned[:, valid_indices_m1 & valid_indices_m2]
    m1_pupil_cleaned = m1_pupil_cleaned[valid_indices_m1 & valid_indices_m2]
    m2_pupil_cleaned = m2_pupil_cleaned[valid_indices_m1 & valid_indices_m2]
    time_vec_cleaned = time_vec_cleaned[valid_indices_m1 & valid_indices_m2]
    return m1_pos_cleaned, m2_pos_cleaned, time_vec_cleaned, \
        m1_pupil_cleaned, m2_pupil_cleaned, rects_m1, rects_m2


def calculate_gaze_avg_pupil_size(pos_x, pos_y, pupil, bins):
    heatmap, xedges, yedges = np.histogram2d(pos_x, pos_y, bins=bins)
    avg_pupil = np.zeros_like(heatmap)
    for i in range(len(xedges) - 1):
        for j in range(len(yedges) - 1):
            indices = np.where((pos_x >= xedges[i]) & (pos_x < xedges[i+1]) &
                               (pos_y >= yedges[j]) & (pos_y < yedges[j+1]))[0]
            if len(indices) > 0:
                avg_pupil[i, j] = np.mean(pupil[indices])
    return heatmap, avg_pupil, xedges, yedges