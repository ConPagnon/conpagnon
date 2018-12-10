"""
 Created by Dhaif BEKHA.

This routine aims to remove the lesion driven artifact in resting state fmri
in case of a stroke. Adapted in python from [1].

.. [1] Connectome-based lesion-symptom mapping (CLSM): A novel approach to map neurological function,
       NeuroImage Clinical, 2018, Ezequiel et al.

 """
import os
import numpy as np
from nilearn.image import load_img, resample_to_img, iter_img
import pandas as pd
import glob
from subprocess import Popen, PIPE
from nilearn.plotting import plot_stat_map
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


# Set data path
subjects_list = '/media/db242421/db242421_data/ConPagnon_data/text_data/acm_patients.txt'
image_data_directory = '/media/db242421/db242421_data/identify_lesion_driven_artifact/preprocessed_patients'
fmri_image_directory = 'fmri'
lesion_image_directory = 'lesion'
fmri_image_filter = 'art*.nii'
lesion_image_filter = 'w*.nii'
mean_controls_image = '/media/db242421/db242421_data/ConPagnon_data/lesion_behavior_mapping/mean_controls.nii'
# Get FSL path via current environment
fsl_path = os.path.join(os.getenv('FSLDIR'), 'bin')
fsl_outtype = os.getenv('FSLOUTPUTTYPE')
if not fsl_outtype:
    fsl_outtype = 'NIFTI'
else:
    pass

# Set repetition time
TR = 2.40

# Read subjects list
subjects = list(pd.read_csv(subjects_list, header=None)[0])
# Loop the whole algorithm on all the subjects
for subject in subjects:
    print('Identifying and removing independent components associated with '
          'lesion for subject {} \n'.format(subject))
    # Full path of subject fmri image
    subject_fmri_path = glob.glob(os.path.join(
        image_data_directory, subject,
        fmri_image_directory,
        fmri_image_filter))[0]
    # Load the subject lesion image, affine, and array
    subject_lesion_path = glob.glob(os.path.join(
        image_data_directory, subject, lesion_image_directory,
        lesion_image_filter))[0]
    subject_lesion_image = load_img(subject_lesion_path)
    subject_lesion_affine = subject_lesion_image.affine
    subject_lesion_data = subject_lesion_image.get_data()

    # Run melodic ICA from FSL
    melodic_command = 'melodic -i {} -o {}/melodic_output --tr={}'.format(subject_fmri_path,
                                                                          os.path.join(image_data_directory,
                                                                                       subject),
                                                                          TR)
    melodic_process = Popen(melodic_command.split(), stdout=PIPE)
    melodic_output, melodic_error = melodic_process.communicate()

    # find the independent components that overlap
    # with the subject lesion mask with the Jaccard index
    # significant overlap is when Jaccard index is over 5%.

    # Read independent components image
    all_independent_components_img = load_img(os.path.join(
        image_data_directory,
        subject, 'melodic_output/melodic_IC.nii.gz'))
    all_independent_components_data = all_independent_components_img.get_data()
    # Resample the independent component image to the
    # subject lesion image
    all_resampled_independent_components_img = \
        resample_to_img(source_img=all_independent_components_img,
                        target_img=subject_lesion_image)
    all_resampled_independent_components_data = all_resampled_independent_components_img.get_data()

    # Plot on glass brain IC map
    # Create a illustration directory
    os.mkdir(path=os.path.join(image_data_directory, subject, 'plot'))
    iterator_ic_map = list(
        iter_img(os.path.join(image_data_directory, subject, 'melodic_output/melodic_IC.nii.gz')))
    with PdfPages(os.path.join(image_data_directory, subject, 'plot', 'ic_maps.pdf')) as pdf:
        for ICA_map in iterator_ic_map:
            t = np.percentile(a=all_independent_components_data[..., iterator_ic_map.index(ICA_map)],
                              q=97.5)
            plt.figure(num=iterator_ic_map.index(ICA_map))
            display = plot_stat_map(
                stat_map_img=ICA_map,
                bg_img=mean_controls_image,
                title='Melodic map # {}'.format(iterator_ic_map.index(ICA_map)),
                threshold=t)
            display.add_contours(img=subject_lesion_path, colors=['black'], contours=4)
            pdf.savefig()
            plt.close(iterator_ic_map.index(ICA_map))

    # Initialize jaccard vector
    jaccard = np.zeros(all_resampled_independent_components_img.shape[3])
    jaccard_ = np.zeros(all_resampled_independent_components_img.shape[3])
    for component in range(all_independent_components_data.shape[3]):
        # Find threshold voxel value corresponding to the
        # to a two tailed p value
        threshold = np.percentile(a=all_independent_components_data[..., component],
                                  q=97.5)
        # Compute the t-score map, i.e find voxels with value
        # between -threshold and +threshold
        t_component_data = np.logical_or(
            all_resampled_independent_components_data[..., component] >= threshold,
            all_resampled_independent_components_data[..., component] <= -threshold)
        # Compute the Jaccard index between lesion and the resampled
        # independent component map
        # Intersection
        jaccard_numerator = np.sum(np.logical_and(t_component_data.flatten(), subject_lesion_data.flatten()))
        # Union
        jaccard_denominator = np.sum(np.logical_or(t_component_data.flatten(), subject_lesion_data.flatten()))

        # Jaccard index : intersection(X, Y) / union(X, Y) in absolute value (no need here)
        jaccard[component] = jaccard_numerator / jaccard_denominator
    # Find the independent components presenting more than
    # 5% of overlap with the lesion image
    lesioned_components,  = np.where(jaccard > 0.05)

    print('{} lesion driven artifact identified \n'.format(lesioned_components.size))
    # If they are some lesioned component we regress them
    # with FSL:
    if lesioned_components.size != 0:
        print('Regression of lesion artifact...')
        subject_directory, fmri_filename = os.path.split(subject_fmri_path)
        lesion_filtered_subject_fmri = 'm' + fmri_filename
        fsl_reg_filt_command = 'fsl_regfilt -i {} -o {} -d {}/melodic_output/melodic_mix -f {}'.format(
            subject_fmri_path,
            os.path.join(subject_directory, lesion_filtered_subject_fmri),
            os.path.join(image_data_directory, subject),
            ','.join(map(str, lesioned_components.tolist()))
             )
        fsl_reg_filt_process = Popen(fsl_reg_filt_command.split(), stdout=PIPE)
        fsl_reg_filt_output, fsl_reg_filt_error = fsl_reg_filt_process.communicate()
        print('Done.')
    else:
        print('No lesion driven artifact detected for {}'.format(subject))

    print('Lesion driven artifact removal done for subject {} \n'.format(subject))



