#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:48:36 2024

@author: pg496
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import scipy
import numpy as np

import util
import proc_behav

import pdb

def plot_roi_rects(ax, rects, roi, make_legend=False):
    """
    Plot rectangles on the given axis.
    Args:
    - ax (matplotlib.axes.Axes): The axis object to plot on.
    - rects (dict): Dictionary of rectangles, each defined by (x1, y1, x2, y2).
    - roi (list): List of strings indicating the ROI names.
    - make_legend (bool): Whether to include a legend in the plot (default False).
    Returns:
    - None
    """
    default_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, roi_name in enumerate(roi):
        rect = rects[roi_name][0][0]
        x1, y1, x2, y2 = rect
        width = x2 - x1
        height = y2 - y1
        color = default_colors[i % len(default_colors)]  # Use modulo to cycle through default colors
        rect_patch = patches.Rectangle(
            (x1, y1), width, height, edgecolor=color, facecolor='none', label=roi_name)
        ax.add_patch(rect_patch)
    if make_legend:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=3)


def plot_gaze_density(ax, heatmap, xedges, yedges, title):
    """
    Plot gaze density heatmap.
    Args:
    - ax (matplotlib.axes.Axes): The axis object to plot on.
    - heatmap (np.array): 2D array representing the heatmap.
    - xedges (np.array): Array representing the bin edges along the x-axis.
    - yedges (np.array): Array representing the bin edges along the y-axis.
    - title (str): Title for the plot.
    Returns:
    - None
    """
    ax.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin="lower")
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(False)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Invert y-axis direction


def plot_average_pupil(ax, avg_pupil, xedges, yedges, title):
    """
    Plot average pupil size distribution.
    Args:
    - ax (matplotlib.axes.Axes): The axis object to plot on.
    - avg_pupil (np.array): 2D array representing the average pupil size distribution.
    - xedges (np.array): Array representing the bin edges along the x-axis.
    - yedges (np.array): Array representing the bin edges along the y-axis.
    - title (str): Title for the plot.
    Returns:
    - None
    """
    ax.imshow(avg_pupil.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin="lower")
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(False)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Invert y-axis direction


def plot_mean_pupil_size_distribution(pruned_files, session, run_number,
                                      pos_file, plot_root, plot_dir_name, n_bins):
    """
    Plot mean pupil size distribution.
    Args:
    - pruned_files (tuple): Tuple containing cleaned data and metadata.
    - session (datetime.datetime): Date of the session.
    - run_number (int): Run number of the session.
    - pos_file (str): File path for position data.
    - plot_root (str): Root directory for saving plots.
    - plot_dir_name (str): Directory name for saving plots.
    - n_bins (int): Number of bins for histogram.
    Returns:
    - None
    """
    # Extract cleaned data and metadata
    m1_pos_within_frame, m1_time_within_frame, m1_pupil_within_frame, rects_m1, m1_rois, \
    m2_pos_within_frame, m2_time_within_frame, m2_pupil_within_frame, rects_m2, m2_rois = pruned_files
    bins = n_bins
    # Calculate gaze density and average pupil size for M1
    heatmap_m1, avg_pupil_m1, xedges_m1, yedges_m1 = \
        proc_behav.calculate_gaze_avg_pupil_size(
            m1_pos_within_frame[:,0], m1_pos_within_frame[:,1], m1_pupil_within_frame, bins)
    # Calculate gaze density and average pupil size for M2
    heatmap_m2, avg_pupil_m2, xedges_m2, yedges_m2 = \
        proc_behav.calculate_gaze_avg_pupil_size(
            m2_pos_within_frame[:,0], m2_pos_within_frame[:,1], m2_pupil_within_frame, bins)
    # Plot subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    # Plot M1 gaze density
    plot_gaze_density(axs[0, 0], heatmap_m1, xedges_m1, yedges_m1, 'm1 Gaze Density')
    plot_roi_rects(axs[0, 0], rects_m1, m1_rois, make_legend=True)
    # Plot M2 gaze density
    plot_gaze_density(axs[0, 1], heatmap_m2, xedges_m2, yedges_m2, 'm2 Gaze Density')
    plot_roi_rects(axs[0, 1], rects_m2, m2_rois)
    # Plot M1 average pupil size
    plot_average_pupil(axs[1, 0], avg_pupil_m1, xedges_m1, yedges_m1, 'm1 Average Pupil Size')
    plot_roi_rects(axs[1, 0], rects_m1, m1_rois)
    # Plot M2 average pupil size
    plot_average_pupil(axs[1, 1], avg_pupil_m2, xedges_m2, yedges_m2, 'm2 Average Pupil Size')
    plot_roi_rects(axs[1, 1], rects_m2, m2_rois)
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


def plot_pupil_dustribution_for_one_file(args):
    """
    Plot pupil distribution for one file.
    Args:
    - args (tuple): Tuple containing file paths and plot directories.
    Returns:
    - None
    """
    pos_file, time_file, rect_file, pupil_file, plot_root, plot_dir_name, stretch_factor, n_bins = args
    # Load data files
    loaded_pos_file = scipy.io.loadmat(pos_file)
    loaded_time_file = scipy.io.loadmat(time_file)
    loaded_rect_file = scipy.io.loadmat(rect_file)
    loaded_pupil_file = scipy.io.loadmat(pupil_file)
    # Extract session date and run number
    session = util.extract_session_date(pos_file)
    run_number = util.extract_run_number(pos_file)
    try:
        # Remove NaNs from position-time data
        nan_removed_files = proc_behav.remove_nans_in_pos_time(
            loaded_pos_file, loaded_time_file, loaded_rect_file, loaded_pupil_file)
        # Unpack cleaned data
        m1_pos_cleaned, m2_pos_cleaned, time_vec_cleaned, m1_pupil_cleaned, \
            m2_pupil_cleaned, rects_m1, rects_m2 = nan_removed_files
        # Prepare parameters for plotting
        plot_params_tuple = nan_removed_files + (stretch_factor, )
        # Create frames and crop position-time data
        pruned_files = proc_behav.use_roi_to_create_frame_and_crop_pos_time(
            plot_params_tuple)
        # Plot mean pupil size distribution
        plot_mean_pupil_size_distribution(pruned_files, session, run_number,
                                          pos_file, plot_root, plot_dir_name, n_bins)
    except:
        # Do nothing if there's an error
        pass


def plot_gaze_fixation_and_pupil_heatmap_for_session(file_tuple, plot_root, 
                                                     plot_dir_name, stretch_factor,
                                                     n_bins, sampling_rate):
    """
    Plot mean pupil size distribution.
    Args:
    - pruned_files (tuple): Tuple containing cleaned data and metadata.
    - session (datetime.datetime): Date of the session.
    - run_number (int): Run number of the session.
    - pos_file (str): File path for position data.
    - plot_root (str): Root directory for saving plots.
    - plot_dir_name (str): Directory name for saving plots.
    - n_bins (int): Number of bins for histogram.
    Returns:
    - None
    """
    time_files, pos_files, pupil_files, rect_files = file_tuple
    rep_file_path = pos_files[0]
    session = util.extract_session_date(rep_file_path)  # Extract session date
    gaze_events_in_session = \
        proc_behav.get_pos_time_pupil_fix_and_rois_within_session(
            file_tuple, stretch_factor, sampling_rate)
    # Extract cleaned data and metadata
    m1_pos_in_session, m1_time_in_session, m1_pupil_in_session, \
        m1_fix_in_session, rects_m1, m1_rois, \
            m2_pos_in_session, m2_time_in_session, m2_pupil_in_session, \
                m2_fix_in_session, rects_m2, m2_rois = \
                    gaze_events_in_session
    bins = n_bins
    m1_fix_in_session = [bool(x) for x in m1_fix_in_session]
    m2_fix_in_session = [bool(x) for x in m2_fix_in_session]
    # Calculate gaze density and average pupil size for M1 and M2
    print('Generating heatmaps...')
    heatmap_m1, avg_pupil_m1, xedges_m1, yedges_m1 = \
        proc_behav.calculate_gaze_avg_pupil_size(
            m1_pos_in_session[:,0], m1_pos_in_session[:,1],
            m1_pupil_in_session, bins)
    heatmap_m2, avg_pupil_m2, xedges_m2, yedges_m2 = \
        proc_behav.calculate_gaze_avg_pupil_size(
            m2_pos_in_session[:,0], m2_pos_in_session[:,1],
            m2_pupil_in_session, bins)
    # Calculate fixation density for M1 and M2
    fix_heatmap_m1, fix_pupil_m1, fix_xedges_m1, fix_yedges_m1 = \
        proc_behav.calculate_gaze_avg_pupil_size(
            m1_pos_in_session[m1_fix_in_session,0],
            m1_pos_in_session[m1_fix_in_session,1],
            m1_pupil_in_session[m1_fix_in_session],
            bins)
    fix_heatmap_m2, fix_pupil_m2, fix_xedges_m2, fix_yedges_m2 = \
        proc_behav.calculate_gaze_avg_pupil_size(
            m2_pos_in_session[m2_fix_in_session,0],
            m2_pos_in_session[m2_fix_in_session,1],
            m2_pupil_in_session[m2_fix_in_session],
            bins)
    # Plot gaze density and normalized pupil size
    fig1, axs1 = plt.subplots(2, 2, figsize=(12, 8))
    # Plot M1 gaze density
    plot_gaze_density(axs1[0, 0], heatmap_m1, xedges_m1, yedges_m1, 'm1 Gaze Density')
    plot_roi_rects(axs1[0, 0], rects_m1, m1_rois, make_legend=True)
    # Plot M2 gaze density
    plot_gaze_density(axs1[0, 1], heatmap_m2, xedges_m2, yedges_m2, 'm2 Gaze Density')
    plot_roi_rects(axs1[0, 1], rects_m2, m2_rois)
    # Plot M1 average pupil size
    plot_average_pupil(axs1[1, 0], avg_pupil_m1, xedges_m1, yedges_m1, 'm1 Average Pupil Size')
    plot_roi_rects(axs1[1, 0], rects_m1, m1_rois)
    # Plot M2 average pupil size
    plot_average_pupil(axs1[1, 1], avg_pupil_m2, xedges_m2, yedges_m2, 'm2 Average Pupil Size')
    plot_roi_rects(axs1[1, 1], rects_m2, m2_rois)
    # Set super-title
    super_title = f"Gaze Density for Entire Session: {session.strftime('%Y-%m-%d')}"
    fig1.suptitle(super_title, fontsize=14)
    # Save plot
    session_folder = session.strftime('%Y-%m-%d')
    plot_name1 = f"gaze_dens_{session_folder}.png"
    save_dir1 = os.path.join(plot_root, plot_dir_name)
    if not os.path.exists(save_dir1):
        os.makedirs(save_dir1)
    plt.savefig(os.path.join(save_dir1, plot_name1))
    plt.close(fig1)  # Close the figure to release memory
    # Plot fixation density
    fig2, axs2 = plt.subplots(2, 2, figsize=(12, 8))
    # Plot M1 fixation density
    plot_gaze_density(axs2[0, 0], fix_heatmap_m1, fix_xedges_m1, fix_yedges_m1, 'm1 Fixation Density')
    plot_roi_rects(axs2[0, 0], rects_m1, m1_rois, make_legend=True)
    # Plot M2 fixation density
    plot_gaze_density(axs2[0, 1], fix_heatmap_m2, fix_xedges_m2, fix_yedges_m2, 'm2 Fixation Density')
    plot_roi_rects(axs2[0, 1], rects_m2, m2_rois)
    # Plot M1 average pupil size
    plot_average_pupil(axs2[1, 0], fix_pupil_m1, fix_xedges_m1, fix_yedges_m1, 'm1 Average Pupil Size')
    plot_roi_rects(axs2[1, 0], rects_m1, m1_rois)
    # Plot M2 average pupil size
    plot_average_pupil(axs2[1, 1], fix_pupil_m2, fix_xedges_m2, fix_yedges_m2, 'm2 Average Pupil Size')
    plot_roi_rects(axs2[1, 1], rects_m2, m2_rois)
    # Set super-title
    super_title = f"Fixation Density for Entire Session: {session.strftime('%Y-%m-%d')}"
    fig2.suptitle(super_title, fontsize=14)
    # Save plot
    plot_name2 = f"fixations_{session_folder}.png"
    save_dir2 = os.path.join(plot_root, plot_dir_name)
    if not os.path.exists(save_dir2):
        os.makedirs(save_dir2)
    plt.savefig(os.path.join(save_dir2, plot_name2))
    plt.close(fig2)  # Close the figure to release memory





