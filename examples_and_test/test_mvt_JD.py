 #!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This script requires the nipy-data package to run. It is an example of
simultaneous motion correction and slice timing correction in
multi-session fMRI data from the FIAC 2005 dataset. Specifically, it
uses the first two sessions of subject 'fiac0'.

Usage:
python space_time_realign [iterations]

where iterations is a positive integer (set to 1 by default). If
zero, no motion correction is performed and the data is only
corrected for slice timing. The larger iterations, the more accurate
motion correction.

Two images will be created in the working directory for the realigned
series:

rarun2.nii

Il y a 2 lignes importantes. La première pour instancier l'algorithme:

R = FmriRealign4d(runs, tr=2.5, slice_order='ascending', interleaved=True)

Dans ton cas, `interleaved=False`, tr à ajuster. La deuxième ligne pour lancer le calcul proprement dit:

R.estimate(refscan=None)

Tu peux omettre `refscan=None` et écrire donc juste R.estimate() si tu veux que toutes tes images soient recalées sur la première image de la première session.

Author: Alexis Roche, 2009.
"""

from nipy.algorithms.registration import FmriRealign4d, SpaceTimeRealign
from nipype.interfaces.nipy import SpaceTimeRealigner

from nibabel import load
from nipy import save_image
#from nipy.utils import example_data
import os
from os.path import join, split
#import tempfile
from nibabel import Nifti1Image
import numpy as np
from nipy.algorithms.registration.groupwise_registration import tr_from_header
from nibabel import (load, Nifti1Image, io_orientation)
# Input images are provided with the nipy-data package
# runnames = [example_data.get_filename('fiac', 'fiac0', run + '.nii.gz')\
#                for run in ('run1', 'run2')]
db_dir = '/media/db242421/db242421_data/data_JD'

# subjects = [each for each in os.listdir(db_dir) if os.path.isdir(join(db_dir, each)) and each[0:3] == 'sub']
# ['sub01_eb110475',
# subjects = ['sub11_lp130475']
# 'sub10_cb120364',...

# subjects = open('/neurospin/grip/protocols/MRI/Resting_state_Victor_2014/ressources_txt/sub1.txt','r').read().split()
subjects = open('/media/db242421/db242421_data/data_JD/listRbb.txt', 'r').read().split()

### choisir
session = 'RightHand'
seq = 'fmri'
# run_prefix = 'fluphra'
# run_prefix = 'RightHand'
TR = 1  # 2.4 pour resting mmx 2.5 pour flumot et fluphra mmx
ext = 'hand.nii'

for subject in subjects:


    run_prefix = subject
    # subject = subjects[6]
    print('Subject %s :' % subject)
    data_dir = join(db_dir, subject, seq, session)

    filenames = [join(data_dir, each) for each in os.listdir(data_dir)]

    runnames = []
    runnames = [filename for filename in filenames if
             split(filename)[1][0:len(run_prefix)] == run_prefix and split(filename)[1][-8:] == ext]
    print(runnames)

    runs = [load(run) for run in runnames]
    slice_axis = int(np.where(io_orientation(runs[0].affine)[:, 0] == 2)[0])
    # Spatio-temporal realigner
   # R = FmriRealign4d(runs, tr=TR, slice_order='descending', interleaved='False')
    R = SpaceTimeRealign(runs, tr=TR,
                         slice_times='descending',
                         slice_info=slice_axis)
    R.estimate(refscan=None,
               optimizer='bfgs')
   # R.inputs.in_file = runnames
   # R.inputs.tr = TR
   # R.inputs.slice_times = 'descending'
   # R.inputs.slice_info = 2
   # R.output_spec.out_file = data_dir

    #R.run()

    # Estimate motion within- and between-sessions
    # R.estimate(refscan=None) R = FmriRealign4d(runs, tr=2.5, slice_order='ascending', interleaved='True')
    R.estimate(refscan=None)
    # Resample data on a regular space+time lattice using 4d interpolation
    # Save images
    # savedir = tempfile.mkdtemp()
    savedir = join(db_dir, subject, seq, session)
    # savedir = data_dir +'interleaved_preproc/'
    # os.mkdir(savedir)
    print('Saving results in: %s' % savedir)

    for i in range(len(runs)):


        corr_run = R.resample(i)
        fname = 'ra' + split(runnames[i])[1]
        save_image(corr_run, join(savedir, fname))
        transforms = R._transforms[i]
        translations = np.array([t.translation for t in transforms])
        rotations = np.array([t.rotation for t in transforms])
        mvt = np.concatenate((translations, rotations), axis=1)
        mvtname = 'rp_' + split(runnames[i])[1][:-3] + 'txt'
        np.savetxt(join(savedir, mvtname), mvt)
