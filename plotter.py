#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:48:36 2024

@author: pg496
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

import pdb

import proc_behav

def plot_and_save_gaze_heatmaps(
        m1_pos_cleaned, m2_pos_cleaned, rects_m1, rects_m2,
        session, run_number, pos_file, plot_root, plot_dir_name):
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



def plot_mean_pupil_size_distribution(
        m1_pos_cleaned, m2_pos_cleaned, rects_m1, rects_m2,
        m1_pupil_cleaned, m2_pupil_cleaned, session, run_number,
        pos_file, plot_root, plot_dir_name):
    
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
    # Plot rectangles for m1
    for rect in rects_m1:
        x1, y1, x2, y2 = rect
        width = x2 - x1
        height = y2 - y1
        rect_patch = patches.Rectangle((x1, y1), width, height, edgecolor='r', facecolor='none')
        axs[0, 0].add_patch(rect_patch)

    # Plot heatmap for m2
    axs[0, 1].imshow(heatmap_m2.T, extent=[xedges_m2[0], xedges_m2[-1], yedges_m2[0], yedges_m2[-1]], origin='lower')
    axs[0, 1].set_title('m2 Gaze Density')
    axs[0, 1].set_xlabel('X')
    axs[0, 1].set_ylabel('Y')
    axs[0, 1].grid(False)
    axs[0, 1].set_aspect('equal')
    axs[0, 1].invert_yaxis()  # Invert y-axis direction
    # Plot rectangles for m2
    for rect in rects_m2:
        x1, y1, x2, y2 = rect
        width = x2 - x1
        height = y2 - y1
        rect_patch = patches.Rectangle((x1, y1), width, height, edgecolor='r', facecolor='none')
        axs[0, 1].add_patch(rect_patch)

    # Plot average pupil for m1
    axs[1, 0].imshow(avg_pupil_m1.T, extent=[xedges_m1[0], xedges_m1[-1], yedges_m1[0], yedges_m1[-1]], origin='lower')
    axs[1, 0].set_title('m1 Average Pupil Size')
    axs[1, 0].set_xlabel('X')
    axs[1, 0].set_ylabel('Y')
    axs[1, 0].grid(False)
    axs[1, 0].set_aspect('equal')
    axs[1, 0].invert_yaxis()  # Invert y-axis direction
    # Plot rectangles for m1
    for rect in rects_m1:
        x1, y1, x2, y2 = rect
        width = x2 - x1
        height = y2 - y1
        rect_patch = patches.Rectangle((x1, y1), width, height, edgecolor='r', facecolor='none')
        axs[1, 0].add_patch(rect_patch)

    # Plot average pupil for m2
    axs[1, 1].imshow(avg_pupil_m2.T, extent=[xedges_m2[0], xedges_m2[-1], yedges_m2[0], yedges_m2[-1]], origin='lower')
    axs[1, 1].set_title('m2 Average Pupil Size')
    axs[1, 1].set_xlabel('X')
    axs[1, 1].set_ylabel('Y')
    axs[1, 1].grid(False)
    axs[1, 1].set_aspect('equal')
    axs[1, 1].invert_yaxis()  # Invert y-axis direction
    # Plot rectangles for m2
    for rect in rects_m2:
        x1, y1, x2, y2 = rect
        width = x2 - x1
        height = y2 - y1
        rect_patch = patches.Rectangle((x1, y1), width, height, edgecolor='r', facecolor='none')
        axs[1, 1].add_patch(rect_patch)

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
