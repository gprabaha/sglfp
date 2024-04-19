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


def plot_roi_rects(ax, rects, color='r'):
    """
    Plot rectangles on the given axis.
    Args:
    - ax (matplotlib.axes.Axes): The axis object to plot on.
    - rects (list): List of rectangles, each defined by (x1, y1, x2, y2).
    - color (str): Color of the rectangles. Default is 'r' (red).
    Returns:
    - None
    """
    for rect in rects:
        x1, y1, x2, y2 = rect
        width = x2 - x1
        height = y2 - y1
        rect_patch = patches.Rectangle(
            (x1, y1), width, height, edgecolor=color, facecolor='none')
        ax.add_patch(rect_patch)


def plot_mean_pupil_size_distribution(
        m1_pos_cleaned, m2_pos_cleaned, rects_m1, rects_m2,
        m1_pupil_cleaned, m2_pupil_cleaned, session, run_number,
        pos_file, plot_root, plot_dir_name):
    """
    Plot mean pupil size distribution.
    Args:
    - m1_pos_cleaned (numpy.ndarray): Cleaned position data for m1.
    - m2_pos_cleaned (numpy.ndarray): Cleaned position data for m2.
    - rects_m1 (list): ROI rects for m1.
    - rects_m2 (list): ROI rects for m2.
    - m1_pupil_cleaned (numpy.ndarray): Cleaned pupil data for m1.
    - m2_pupil_cleaned (numpy.ndarray): Cleaned pupil data for m2.
    - session (datetime.datetime): Session date.
    - run_number (int): Run number.
    - pos_file (str): Path of the position file.
    - plot_root (str): Root directory for saving plots.
    - plot_dir_name (str): Name of the plot directory.
    Returns:
    - None
    """
    bins = 100
    # Calculate average pupil for m1
    heatmap_m1, avg_pupil_m1, xedges_m1, yedges_m1 = \
        proc_behav.calculate_gaze_avg_pupil_size(
            m1_pos_cleaned[0], m1_pos_cleaned[1], m1_pupil_cleaned, bins)
    # Calculate average pupil for m2
    heatmap_m2, avg_pupil_m2, xedges_m2, yedges_m2 = \
        proc_behav.calculate_gaze_avg_pupil_size(
            m2_pos_cleaned[0], m2_pos_cleaned[1], m2_pupil_cleaned, bins)
    # Plot subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    # Plot heatmap for m1
    axs[0, 0].imshow(heatmap_m1.T, extent=[xedges_m1[0], xedges_m1[-1], yedges_m1[0], yedges_m1[-1]], origin='lower')
    axs[0, 0].set_title('m1 Gaze Density')
    axs[0, 0].set_xlabel('X')
    axs[0, 0].set_ylabel('Y')
    axs[0, 0].grid(False)
    axs[0, 0].set_aspect('equal')
    axs[0, 0].invert_yaxis()  # Invert y-axis direction
    # Plot ROIs for m1
    plot_roi_rects(axs[0, 0], rects_m1)
    # Plot heatmap for m2
    axs[0, 1].imshow(heatmap_m2.T, extent=[xedges_m2[0], xedges_m2[-1], yedges_m2[0], yedges_m2[-1]], origin='lower')
    axs[0, 1].set_title('m2 Gaze Density')
    axs[0, 1].set_xlabel('X')
    axs[0, 1].set_ylabel('Y')
    axs[0, 1].grid(False)
    axs[0, 1].set_aspect('equal')
    axs[0, 1].invert_yaxis()  # Invert y-axis direction
    # Plot ROIs for m2
    plot_roi_rects(axs[0, 1], rects_m2)
    # Plot average pupil for m1
    axs[1, 0].imshow(avg_pupil_m1.T, extent=[xedges_m1[0], xedges_m1[-1], yedges_m1[0], yedges_m1[-1]], origin='lower')
    axs[1, 0].set_title('m1 Average Pupil Size')
    axs[1, 0].set_xlabel('X')
    axs[1, 0].set_ylabel('Y')
    axs[1, 0].grid(False)
    axs[1, 0].set_aspect('equal')
    axs[1, 0].invert_yaxis()  # Invert y-axis direction
    # Plot ROIs for m1
    plot_roi_rects(axs[1, 0], rects_m1)
    # Plot average pupil for m2
    axs[1, 1].imshow(avg_pupil_m2.T, extent=[xedges_m2[0], xedges_m2[-1], yedges_m2[0], yedges_m2[-1]], origin='lower')
    axs[1, 1].set_title('m2 Average Pupil Size')
    axs[1, 1].set_xlabel('X')
    axs[1, 1].set_ylabel('Y')
    axs[1, 1].grid(False)
    axs[1, 1].set_aspect('equal')
    axs[1, 1].invert_yaxis()  # Invert y-axis direction
    # Plot ROIs for m2
    plot_roi_rects(axs[1, 1], rects_m2)
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
    pos_file, time_file, rect_file, pupil_file, plot_root, plot_dir_name = args
    loaded_pos_file = scipy.io.loadmat(pos_file)
    loaded_time_file = scipy.io.loadmat(time_file)
    loaded_rect_file = scipy.io.loadmat(rect_file)
    loaded_pupil_file = scipy.io.loadmat(pupil_file)
    session = util.extract_session_date(pos_file)
    run_number = util.extract_run_number(pos_file)
    try:
        m1_pos_cleaned, m2_pos_cleaned, time_vec_cleaned, \
            m1_pupil_cleaned, m2_pupil_cleaned, rects_m1, rects_m2 = \
                proc_behav.extract_pos_time(
                    loaded_pos_file, loaded_time_file,
                    loaded_rect_file, loaded_pupil_file)
        plot_mean_pupil_size_distribution(
            m1_pos_cleaned, m2_pos_cleaned, rects_m1, rects_m2,
            m1_pupil_cleaned, m2_pupil_cleaned, session, run_number,
            pos_file, plot_root, plot_dir_name)
    except:
        # Do nothing if there's an error
        pass


