"""
 Created by Dhaif BEKHA, (dhaif.bekha@cea.fr).

 """
import pandas as pd
import os
from nipype.interfaces.mrtrix.convert import MRTrix2TrackVis
from dipy.io.streamline import load_trk
from nilearn.image import load_img
import dipy.segment.bundles as dsb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

"""
This code compute tract profile 
following [1]. One must install
the Dipy library to use this code.
"""

# Root directory of the architecture of you're database
root_directory = "/neurospin/grip/protocols/MRI/Ines_2018/images/controls_v2"

# Path to the subjects list in text file, with one column per
# subject identifiers.
subjects_list_txt = "/neurospin/grip/protocols/MRI/Ines_2018/images/controls_v2_nip"
subjects = list(pd.read_csv(subjects_list_txt, header=None)[0])

# List of bundles of interest: name of the tractography tract
# files
bundles_to_tract = sorted(['AF_right.tck', 'AF_left.tck'])


# Tracking directory architecture
tracking_directories = "TractSeg_outputs/tractseg_output/TOM_trackings"

# Dwi image directory: just needed for the conversion
# of .tck file format in .trk file format.
dwi_image_directory = "motion_corrected_data"

# Dtifit output directory: directory containing
# the FA map of the subject
FA_map_directory = "dtifit"

# Sampling rate along the bundle
nb_points = 500

# iterate over the subjects:
for subject in subjects:
    print("Tract analysis for subject {}".format(subject))
    # iterate over desired bundles
    for bundle in bundles_to_tract:
        print(bundle)
        # Check if bundle file exist
        bundles_list = os.listdir(os.path.join(root_directory, subject, tracking_directories))
        if bundle not in bundles_list:
            print("Warning !! The bundle {} is missing and will not be processed.".format(bundle))
            continue
        bundle_file = os.path.join(root_directory, subject, tracking_directories, bundle)
        dwi_image = os.path.join(root_directory, subject, dwi_image_directory,
                                 "distortion_and_eddy_corrected_dti_" + subject + ".nii.gz")
        # Create a directory for the tract profile analysis, and a directory for each tract
        tract_analysis_directory = os.path.join(root_directory, subject, "tract_analysis")
        if tract_analysis_directory.split(sep="/")[-1:][0] not in os.listdir(os.path.join(root_directory, subject)):
            os.mkdir(tract_analysis_directory)
        # Create a directory for the bundles
        if bundle[:-4] not in os.listdir(tract_analysis_directory):

            os.mkdir(os.path.join(tract_analysis_directory, bundle[:-4]))

        bundle_analysis_directory = os.path.join(tract_analysis_directory, bundle[:-4])
        # need to convert them
        tck2trk = MRTrix2TrackVis()
        tck2trk.inputs.in_file = bundle_file
        tck2trk.inputs.image_file = dwi_image
        tck2trk.inputs.out_filename = os.path.join(bundle_analysis_directory, bundle[:-4] + ".trk")
        tck2trk.run()

        # Full path to the .trk bundle file
        bundle_trk_file = os.path.join(bundle_analysis_directory, bundle[:-4] + ".trk")
        # Full path to the subject FA map
        fa = os.path.join(root_directory, subject, FA_map_directory, "dtifit_" + subject + "_FA.nii.gz")

        # Load the bundle streamlines with DIPY
        bundle_trk, bundle_header = load_trk(bundle_trk_file)
        transform = bundle_header['voxel_to_rasmm']
        # Load FA map and get data
        fa_img = load_img(fa)
        fa_data = fa_img.get_data()
        fa_affine = fa_img.affine

        # Extracting bundle feature (FA values)

        # Compute weights for the bundle
        bundle_weights = dsb.gaussian_weights(bundle=bundle_trk, n_points=nb_points)

        # Compute the tract profile along the bundle
        bundle_profile = dsb.afq_profile(data=fa_data, bundle=bundle_trk, affine=fa_affine,
                                         weights=bundle_weights, n_points=nb_points)

        with PdfPages(os.path.join(bundle_analysis_directory, bundle[:-4] + ".pdf")) as bundle_tract:
            plt.figure()
            plt.plot(bundle_profile, 'o', markersize=0.5)
            plt.xlabel('Node along {}'.format(bundle[:-4]))
            plt.ylabel('Fractional Anisotropy')
            plt.title('FA {} sampled along {} points'.format(bundle[:-4], nb_points))
            bundle_tract.savefig()
            plt.show()