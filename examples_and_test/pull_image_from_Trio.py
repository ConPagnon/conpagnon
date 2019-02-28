"""
 Created by db242421 at 27/02/19

 """

import os
import pandas as pd
import re
import warnings
# TrioTim scanner base directory
TrioTim_directory = '/neurospin/acquisition/database/TrioTim'

# root directory: copy images folder in this directory
root_directory = '/media/db242421/db242421_data/DTI_preproc_state_Ines/images/patients'

# acquisition date and nip in the corresponding order.
acquisition_date_nip = pd.read_csv(os.path.join(root_directory, 'patients_acq_date_nip.txt'))
n_subjects = len(acquisition_date_nip)
# List of regular expression to fetch the right images
acquisition_identifiers = ['(.*)t1mpr(.*)', r'(.*)_b1000-dw30(.*)\d+$']
acquisition_output_path = ['T1/dicom', 'diffusion/dicom']

for subject in range(n_subjects):
    # Locate the acquisition data folder and dive into
    # the corresponding nip
    subject_acquisition_date_directory = os.path.join(TrioTim_directory,
                                                      str(acquisition_date_nip.loc[subject]['acquisition_date']))
    # List the corresponding folders
    acquisition_date_folder_list = os.listdir(subject_acquisition_date_directory)
    # Find the subject folder based on NIP
    nip_pattern = re.compile('{}(.*)'.format(acquisition_date_nip.loc[subject]['NIP']))
    matching_nip_folders = \
        [nip_pattern.search(f).group() for f in acquisition_date_folder_list
         if nip_pattern.search(f) is not None]
    if len(matching_nip_folders) == 1:
        # Dive into the subject directory
        subject_dicom = os.path.join(TrioTim_directory, str(acquisition_date_nip.loc[subject]['acquisition_date']),
                                     matching_nip_folders[0])
        subject_dicom_list = os.listdir(subject_dicom)
        # Loop over the different acquisition we want to fetch
        for acq in range(len(acquisition_identifiers)):
            # Find the folder corresponding to the current identifiers
            acq_pattern = re.compile('{}'.format(acquisition_identifiers[acq]))
            matching_acquisition_folders = \
                [acq_pattern.search(f).group() for f in subject_dicom_list
                    if acq_pattern.search(f) is not None]
            # If two or more matching pattern are found, take the first one
            # by defaults and raise a warning
            if len(matching_acquisition_folders) >= 2:
                warnings.warn('Careful! \n Find {} acquisitions folders matching you\'re request: {}. Taking '
                              'the first one.'.format(len(matching_acquisition_folders), matching_acquisition_folders))
                matching_acquisition_folder = matching_acquisition_folders[0]
