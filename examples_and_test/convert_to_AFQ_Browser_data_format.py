"""
 Created by Dhaif BEKHA, (dhaif.bekha@cea.fr).

 """
import numpy as np
from nibabel import streamlines
import pandas as pd
import os
import csv
import json
import collections

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
#bundle_keys = [str(bundle_keys[i]) for i in range(len(bundle_keys))]
bundle_dictionary[tract_name[0]] = dict.fromkeys(bundle_keys)

# Fill the bundle keys with the
# coordinate of each point for
# each streamlines
for streamline in range(len(bundle_array)):
    bundle_dictionary[tract_name[0]][streamline] = [list(bundle_array[streamline][point])
                                                    for point in range(bundle_array[streamline].shape[0])]

# JSONIFY the dictionary
for streamline in range(len(bundle_array)):
    list_of_point = bundle_dictionary[tract_name[0]][streamline]
    new_list_of_point = [[list_of_point[p][0].item(), list_of_point[p][1].item(), list_of_point[p][2].item()] for p in range(len(list_of_point))]
    bundle_dictionary[tract_name[0]][streamline] = new_list_of_point




# "coreFiber" as the first key
old_key, new_key = 0, "coreFiber"

# Get an FA profile
FA_profile = pd.read_csv('/neurospin/grip/protocols/MRI/MEMODEV_AB_2018/MEMODEV_MRI/diffusion/FA_profiles.csv')
subject_FA_profile = list(FA_profile["ab190042_UF_left.tck"])

# Create a node.csv file
header = ["subjectID", "tractID", "nodeID", "FA"]
with open(os.path.join("/media/db242421/db242421_data/tracto_to_AFQ_browser", "nodes.csv"), "w") as nodes_csv:
    node_csv_writer = csv.writer(nodes_csv)
    node_csv_writer.writerow(header)
    for i in range(len(subject_FA_profile)):
        node_csv_writer.writerow(["ab190042", tract_name[0], i, subject_FA_profile[i]])


from functools import singledispatch

@singledispatch
def keys_to_strings(ob):
    return ob

@keys_to_strings.register(dict)
def _handle_dict(ob):
    return {str(k): keys_to_strings(v) for k, v in ob.items()}

@keys_to_strings.register(list)
def _handle_list(ob):
    return [keys_to_strings(v) for v in ob]

bundle_dictionary[tract_name[0]] = keys_to_strings(bundle_dictionary[tract_name[0]])
with open(os.path.join("/media/db242421/db242421_data/tracto_to_AFQ_browser", "streamlines.json"), "w") \
        as streamline_json:
    json.dump(bundle_dictionary, streamline_json)