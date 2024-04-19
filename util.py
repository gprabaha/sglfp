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
