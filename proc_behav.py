#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:47:30 2024

@author: pg496
"""

import numpy as np


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