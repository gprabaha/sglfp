#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:02:58 2024

@author: pg496
"""

import os
import glob
import numpy as np

lfp_data_dir = "/gpfs/milgram/project/chang/pg496/data_dir/social_gaze/social_gaze_lfp"

# Use glob to find all .npz files in the directory
npz_files = glob.glob(os.path.join(lfp_data_dir, '*.npz'))

random_seed = 40
np.random.seed(random_seed)
ex_file = np.random.choice(npz_files)
print(ex_file)

ex_lfp, ex_lfp_key = np.load(ex_file)
ex_lfp = ex_lfp['lfp']