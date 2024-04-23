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
import util
import proc_behav

import pdb

def plot_roi_rects(ax, rects, roi):
    """
    Plot rectangles on the given axis.
    Args:
    - ax (matplotlib.axes.Axes): The axis object to plot on.
    - rects (dict): Dictionary of rectangles, each defined by (x1, y1, x2, y2).
    - roi (list): List of strings indicating the ROI names.
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
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)

def plot_gaze_density(ax, heatmap, xedges, yedges, title):
    ax.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin="lower")
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(False)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Invert y-axis direction

def plot_average_pupil(ax, avg_pupil, xedges, yedges, title):
    ax.imshow(avg_pupil.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin="lower")
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(False)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Invert y-axis direction

def plot_mean_pupil_size_distribution(pruned_files, session, run_number,
                                      pos_file, plot_root, plot_dir_name, n_bins):
    m1_pos_within_frame, m1_time_within_frame, m1_pupil_within_frame, rects_m1, m1_rois, \
    m2_pos_within_frame, m2_time_within_frame, m2_pupil_within_frame, rects_m2, m2_rois = pruned_files
    bins = n_bins
    # Calculate average pupil for m1
    heatmap_m1, avg_pupil_m1, xedges_m1, yedges_m1 = \
        proc_behav.calculate_gaze_avg_pupil_size(
            m1_pos_within_frame[:,0], m1_pos_within_frame[:,1], m1_pupil_within_frame, bins)
    # Calculate average pupil for m2
    heatmap_m2, avg_pupil_m2, xedges_m2, yedges_m2 = \
        proc_behav.calculate_gaze_avg_pupil_size(
            m2_pos_within_frame[:,0], m2_pos_within_frame[:,1], m2_pupil_within_frame, bins)
    # Plot subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    # Plot m1 gaze density
    plot_gaze_density(axs[0, 0], heatmap_m1, xedges_m1, yedges_m1, 'm1 Gaze Density')
    plot_roi_rects(axs[0, 0], rects_m1, m1_rois)
    # Plot m2 gaze density
    plot_gaze_density(axs[0, 1], heatmap_m2, xedges_m2, yedges_m2, 'm2 Gaze Density')
    plot_roi_rects(axs[0, 1], rects_m2, m2_rois)
    # Plot m1 average pupil size
    plot_average_pupil(axs[1, 0], avg_pupil_m1, xedges_m1, yedges_m1, 'm1 Average Pupil Size')
    plot_roi_rects(axs[1, 0], rects_m1, m1_rois)
    # Plot m2 average pupil size
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
    loaded_pos_file = scipy.io.loadmat(pos_file)
    loaded_time_file = scipy.io.loadmat(time_file)
    loaded_rect_file = scipy.io.loadmat(rect_file)
    loaded_pupil_file = scipy.io.loadmat(pupil_file)
    session = util.extract_session_date(pos_file)
    run_number = util.extract_run_number(pos_file)
    try:
        nan_removed_files = proc_behav.remove_nans_in_pos_time(
            loaded_pos_file, loaded_time_file, loaded_rect_file, loaded_pupil_file)
        m1_pos_cleaned, m2_pos_cleaned, time_vec_cleaned, m1_pupil_cleaned, \
            m2_pupil_cleaned, rects_m1, rects_m2 = nan_removed_files
        plot_params_tuple = nan_removed_files + (stretch_factor, )
        pruned_files = proc_behav.use_roi_to_create_frame_and_crop_pos_time(
            plot_params_tuple)
        plot_mean_pupil_size_distribution(pruned_files, session, run_number,
                                          pos_file, plot_root, plot_dir_name, n_bins)
    except:
        # Do nothing if there's an error
        pass


