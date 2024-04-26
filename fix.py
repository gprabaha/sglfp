#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 10:50:18 2024

@author: pg496
"""

import numpy as np


def distance2p(x1, y1, x2, y2):
    """
    Calculate the distance between two points.
    Args:
    x1, y1: Coordinates of the first point.
    x2, y2: Coordinates of the second point.
    Returns:
    The distance between the two points.
    """
    dx = x2 - x1
    dy = y2 - y1
    distance2p = np.sqrt(dx**2 + dy**2)
    return distance2p


def fixations_t2(fixations, fixation_id, t2):
    """
    Cluster fixations based on spatial criteria and apply t2 threshold.
    Args:
    fixations: Array containing fixations.
    fixation_id: ID of the fixation.
    t2: Spatial parameter t2.
    Returns:
    Fixation information after applying t2 threshold.
    """
    fixations_id = fixations[fixations[:, 3] == fixation_id]
    number_t1 = len(fixations_id)
    # Clustering according to criterion t2
    fixx, fixy = np.nanmean(fixations_id[:, :2], axis=0)
    for i in range(number_t1):
        d = distance2p(fixx, fixy, fixations_id[i, 0], fixations_id[i, 1])
        if d > t2:
            fixations_id[i, 3] = 0
    # Initialize lists
    fixations_list_t2 = np.empty((0, 4))
    list_out_points = np.empty((0, 4))
    for i in range(number_t1):
        if fixations_id[i, 3] > 0:
            fixations_list_t2 = np.vstack((fixations_list_t2, fixations_id[i, :]))
        else:
            list_out_points = np.vstack((list_out_points, fixations_id[i, :]))
    # Compute number of t2 fixations
    number_t2 = fixations_list_t2.shape[0]
    fixx, fixy = np.nanmean(fixations_list_t2[:, :2], axis=0)
    if number_t2 > 0:
        start_time = fixations_list_t2[0, 2]
        end_time = fixations_list_t2[-1, 2]
        duration = end_time - start_time
    else:
        start_time, end_time, duration = 0, 0, 0
    return fixx, fixy, number_t1, number_t2, start_time, end_time, duration, list_out_points


def min_duration(fixation_list, minDur):
    """
    Apply duration threshold to fixation list.
    Args:
    fixation_list: List of fixations.
    minDur: Minimum fixation duration.
    Returns:
    Fixation list after applying duration threshold.
    """
    return [fix for fix in fixation_list if fix[6] >= minDur]


def fixation_detection(data, t1, t2, minDur):
    """
    Detect fixations from raw data.
    Args:
    data: Raw data (x, y, t).
    t1: Spatial parameter t1.
    t2: Spatial parameter t2.
    minDur: Minimum fixation duration.
    Returns:
    Fixation list computed with t1, t2, minDur criteria.
    """
    n = len(data)
    if n == 0:
        return []  # Return empty list if data is empty
    fixations = np.zeros((n, 4))  # Initialize fixations array
    # Spatial clustering
    fixid = 1
    mx, my, d = 0, 0, 0
    fixpointer = 1
    for i in range(n):
        mx = np.nanmean(data[fixpointer:i+1, 0])
        my = np.nanmean(data[fixpointer:i+1, 1])
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


def is_fixation(pos, time, t1=None, t2=None, minDur=None, sampling_rate=None):
    """
    Determine fixations based on position and time data.

    Args:
    pos: Position data (x, y).
    time: Time data.
    t1: Spatial parameter t1.
    t2: Spatial parameter t2.
    minDur: Minimum fixation duration.
    sampling_rate: Sampling rate.
    Returns:
    Binary vector indicating fixations (1) and non-fixations (0).
    """
    # Combine position and time into a single data matrix
    data = np.column_stack((pos, time))
    # Call fixation_detection function to detect fixations
    fix_ranges = fixation_detection(data, t1, t2, minDur)
    # Initialize fixation vector
    fixation_vector = np.zeros(len(time))
    # Mark fixations in the fixation vector
    for range in fix_ranges:
        fixation_vector[range[0]:range[1] + 1] = 1
    return fixation_vector



# Example usage:
# data = np.loadtxt("data.txt")
# fixation_list = fixation_detection(data, 0.250, 0.100, 150, 1.25, 1.00)
# print(fixation_list)
