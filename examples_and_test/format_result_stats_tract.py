import pandas as pd
import re
import os
"""
Format R results from statistical 
analysis of bundle parameters from
NAIS cohort.

"""

def _parse_line(line, rx_dict):
    """

    """
    for key, rx in rx_dict.items():

        match = rx.search(line)
        if match:
            return key, match

    return None, None


stat_results_directory = "/media/db242421/Samsung_T5/Work/Neurospin/AVCnn/Ines_2018/stats_tracts_2"
model_names = ["_Ipsi_Contra_Parole_Zscore_laure"]
parameters = ["FA"]
sides = ["ipsi", "contra"]
bundle_names = ["AF", "MLF", "SLF"]

variables = ["Lesion_ratio"]

for parameter in parameters:
    for side in sides:
        for model in model_names:
            for bundle in bundle_names:
                # Build the regular expression dictionary to
                # search for in the result text file
                rx_dict = {"Lesion_ratio": re.compile(r"Lesion_ratio\s\s\s[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?.*$"),
                           bundle + "_" + side + "_" + parameter:
                               re.compile(r"{}\s\s\s[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?.*$".format(
                                   bundle + "_" + side + "_" + parameter)),
                           "Lesion_ratio_": re.compile(r"Lesion_ratio\s\s\s\s[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?.*$")}

                # Open the text file containing the
                # result for the current model
                bundle_model_result = os.path.join(stat_results_directory, bundle + "_" + side + "_" + parameter +
                                                   "_" + model + ".txt")

                # read the result txt file
                with open(bundle_model_result) as bundle_result_file:
                    # Go through each line, to have a match for the
                    # regular expression
                    for line in bundle_result_file:
                        key, match = _parse_line(line, rx_dict=rx_dict)
                        if key == "Lesion_ratio":
                            model_bundle_lesion_ratio = match.group()
                            t_lesion_ratio, p_lesion_ratio = model_bundle_lesion_ratio.split()[3:5]
                            # Format the t and p couple: t={}, p={}
                            lesion_ratio_data = "t={}, p={}".format(t_lesion_ratio, p_lesion_ratio)
                        if key == "Lesion_ratio_":
                            model_bundle_lesion_ratio = match.group()
                            t_lesion_ratio, p_lesion_ratio = model_bundle_lesion_ratio.split()[3:5]
                            # Format the t and p couple: t={}, p={}
                            lesion_ratio_data = "t={}, p={}".format(t_lesion_ratio, p_lesion_ratio)
                        if key == bundle + "_" + side + "_" + parameter:
                            model_bundle_parameter = match.group()
                            t_bundle_parameter, p_bundle_parameter = model_bundle_parameter.split()[3:5]
                            # Format the t and p couple: t={}, p={}
                            model_bundle_parameter_data = "t={}, p={}".format(t_bundle_parameter,
                                                                              p_bundle_parameter)








