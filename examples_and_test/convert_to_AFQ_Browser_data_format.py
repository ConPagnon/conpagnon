"""
 Created by Dhaif BEKHA, (dhaif.bekha@cea.fr).

 """
import numpy as np
from nibabel import streamlines
import pandas as pd


"""
Convert tractography file (tck, trk) to
a the data format requested by AFQ browser


"""

tck_file = "/media/db242421/db242421_data/tracto_to_AFQ_browser/TOM_trackings/UF_left.tck"

bundle = streamlines.load(tck_file)

bundle_array = bundle.streamlines

tract_name = ["UF_left"]

# Create a dictionary for the bundle
bundle_dictionary = dict.fromkeys(tract_name)

# Create the tract the subsequent key.
# The first sub-key is always named "coreFiber", and
# the rest can be a number
bundle_keys = list(range(0, len(bundle_array)))
bundle_dictionary[tract_name[0]] = dict.fromkeys(bundle_keys)

# Fill the bundle keys with the
# coordinate of each point for
# each streamlines
for streamline in range(len(bundle_array)):
    bundle_dictionary[tract_name[0]][streamline] = [list(bundle_array[streamline][point])
                                                 for point in range(bundle_array[streamline].shape[0])]

# Get an FA profile
FA_profile = pd.read_csv('/neurospin/grip/protocols/MRI/MEMODEV_AB_2018/MEMODEV_MRI/diffusion/FA_profiles.csv')
subject_FA_profile = FA_profile["ab190042_UF_left.tck"]