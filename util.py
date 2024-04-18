#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:46:08 2024

@author: pg496
"""

import os
import datetime

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