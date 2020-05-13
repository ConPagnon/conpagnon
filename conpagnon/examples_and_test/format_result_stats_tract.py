import pandas as pd
import re
import os
from conpagnon.data_handling.data_management import concatenate_dataframes, shift_index_column
from statsmodels.stats.multitest import multipletests
"""
Format R results from statistical 
analysis of bundle parameters from
NAIS cohort.
Author: Dhaif BEKHA (dhaif@dhaifbekha.com)

"""


def _parse_line(line, rx_dict):
    """

    """
    for key, rx in rx_dict.items():

        match = rx.search(line)
        if match:
            return key, match

    return None, None


stat_results_directory = "/media/dhaif/Samsung_T5/Work/Neurospin/AVCnn/Ines_2018/stats_tracts_2"
# model names: the same model name you find in the R script
model_names = ["_Parole_LD", "_SyntExp_LD", "_SyntComp_LD", "_LexExp_LD", "_LexComp_LD", "_PC1_LD"]
# parameters of interest
parameters = ["FA"]
# sides: ipsi, contra, etc...
sides = ["lat"]
# bundle of interest
bundle_names = ["MLF", "SLF", "AF"]


parameters_results_dict = dict.fromkeys(parameters)
for parameter in parameters:
    parameters_results_dict[parameter] = dict.fromkeys(sides)
    for side in sides:

        parameters_results_dict[parameter][side] = dict.fromkeys(model_names)
        for model in model_names:
            model_bundle_results = []
            for bundle in bundle_names:
                # Build the regular expression dictionary to
                # search for in the result text file
                rx_dict = {"Lesion_ratio": re.compile(r"Lesion_ratio\s\s\s[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?.*$"),
                           bundle + "_" + side + "_" + parameter:
                               re.compile(r"{}\s\s\s[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?.*$".format(
                                   bundle + "_" + side + "_" + parameter)),
                           "Lesion_ratio_": re.compile(r"Lesion_ratio\s\s\s\s[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?.*$"),
                           "Lesion_ratio__": re.compile(r"Lesion_ratio\s\s[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?.*$"),
                           "Lesion_ratio___":re.compile(r"Lesion_ratio\s[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?.*$"),
                           "Lesion_ratio____":re.compile(r"Lesion_ratio\s\s\s\s\s[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?.*$"),
                           bundle + "_" + side + "_" + parameter + "_":
                               re.compile(r"{}\s\s\s\s\s[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?.*$".format(
                                   bundle + "_" + side + "_" + parameter)),
                           bundle + "_" + side + "_" + parameter + "__":
                               re.compile(r"{}\s\s\s\s\s\s[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?.*$".format(
                                   bundle + "_" + side + "_" + parameter)),
                           bundle + "_" + side + "_" + parameter + "___":
                               re.compile(r"{}\s\s\s\s[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?.*$".format(
                                   bundle + "_" + side + "_" + parameter)),
                           bundle + "_" + side + "_" + parameter + "____":
                               re.compile(r"{}\s\s[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?.*$".format(
                                   bundle + "_" + side + "_" + parameter)),
                           bundle + "_" + side + "_" + parameter + "_____":
                               re.compile(r"{}\s[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?.*$".format(
                                   bundle + "_" + side + "_" + parameter))


                           }

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
                        print(key)
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
                        if key == "Lesion_ratio__":
                            model_bundle_lesion_ratio = match.group()
                            t_lesion_ratio, p_lesion_ratio = model_bundle_lesion_ratio.split()[3:5]
                        if key == "Lesion_ratio___":
                            model_bundle_lesion_ratio = match.group()
                            t_lesion_ratio, p_lesion_ratio = model_bundle_lesion_ratio.split()[3:5]
                        if key == "Lesion_ratio____":
                            model_bundle_lesion_ratio = match.group()
                            t_lesion_ratio, p_lesion_ratio = model_bundle_lesion_ratio.split()[3:5]
                            # Format the t and p couple: t={}, p={}
                            lesion_ratio_data = "t={}, p={}".format(t_lesion_ratio, p_lesion_ratio)
                        if key == bundle + "_" + side + "_" + parameter:
                            model_bundle_parameter = match.group()
                            t_bundle_parameter, p_bundle_parameter = model_bundle_parameter.split()[3:5]
                            # Format the t and p couple: t={}, p={}
                            model_bundle_parameter_data = "t={}, p={}, q=".format(t_bundle_parameter,
                                                                              p_bundle_parameter)
                        if key == bundle + "_" + side + "_" + parameter + "_":
                            model_bundle_parameter = match.group()
                            t_bundle_parameter, p_bundle_parameter = model_bundle_parameter.split()[3:5]
                            # Format the t and p couple: t={}, p={}
                            model_bundle_parameter_data = "t={}, p={}, q=".format(t_bundle_parameter,
                                                                              p_bundle_parameter)
                        if key == bundle + "_" + side + "_" + parameter + "__":
                            model_bundle_parameter = match.group()
                            t_bundle_parameter, p_bundle_parameter = model_bundle_parameter.split()[3:5]
                            # Format the t and p couple: t={}, p={}
                            model_bundle_parameter_data = "t={}, p={}, q=".format(t_bundle_parameter,
                                                                              p_bundle_parameter)
                        if key == bundle + "_" + side + "_" + parameter + "___":
                            model_bundle_parameter = match.group()
                            t_bundle_parameter, p_bundle_parameter = model_bundle_parameter.split()[3:5]
                            # Format the t and p couple: t={}, p={}
                            model_bundle_parameter_data = "t={}, p={}, q=".format(t_bundle_parameter,
                                                                              p_bundle_parameter),
                        if key == bundle + "_" + side + "_" + parameter + "____":
                            model_bundle_parameter = match.group()
                            t_bundle_parameter, p_bundle_parameter = model_bundle_parameter.split()[3:5]
                            # Format the t and p couple: t={}, p={}
                            model_bundle_parameter_data = "t={}, p={}, q=".format(t_bundle_parameter,
                                                                                  p_bundle_parameter)
                        if key == bundle + "_" + side + "_" + parameter + "_____":
                            model_bundle_parameter = match.group()
                            t_bundle_parameter, p_bundle_parameter = model_bundle_parameter.split()[3:5]
                            # Format the t and p couple: t={}, p={}
                            model_bundle_parameter_data = "t={}, p={}, q=".format(t_bundle_parameter,
                                                                                  p_bundle_parameter)

                    # Build a small dataframe for the current model, bundle, side.
                    model_bundle_parameter_df = pd.DataFrame(index=[model_bundle_lesion_ratio.split()[0],
                                                                    model_bundle_parameter.split()[0]],
                                                             columns=[bundle + model])
                    model_bundle_parameter_df.loc[model_bundle_lesion_ratio.split()[0]] = lesion_ratio_data
                    model_bundle_parameter_df.loc[model_bundle_parameter.split()[0]] = model_bundle_parameter_data
                    model_bundle_results.append(model_bundle_parameter_df)
                parameters_results_dict[parameter][side][model] = model_bundle_results


# Build clean results table
for parameter in parameters:
    for side in sides:
        for model in model_names:
            model_side_dict = parameters_results_dict[parameter][side][model]
            for df in model_side_dict:
                print(df)
                df.to_excel(os.path.join(stat_results_directory, df.columns[0] + "_" + side + ".xlsx"))

# concatenate the excel file
for bundle in bundle_names:

    for side in sides:
        df_list = []
        for model in model_names:
            df = pd.read_excel(os.path.join(stat_results_directory, bundle + "_" + model[1:] + "_" + side + ".xlsx"))
            df_cols = list(df.columns)
            df = df.rename(columns={df_cols[0]: "variables", df_cols[1]: df_cols[1]})
            df = shift_index_column(df, ["variables"])
            print(df)
            df_list.append(df)

        results_concatenate = concatenate_dataframes(df_list, 1)
        results_concatenate.to_excel(os.path.join(stat_results_directory, bundle + "_" + side + ".xlsx"), index=True)

# Finally correct the p value and write it in the right case
for bundle in bundle_names:
    for side in sides:
        df_results = pd.read_excel(os.path.join(stat_results_directory, bundle + "_" + side + ".xlsx"), index_col=0)
        # read all the columns
        df_results_columns = list(df_results.columns)
        p_values = []
        for col in df_results_columns:

            p_value = float(df_results[col].iloc[1].strip().split(", ")[1][2:])
            p_values.append(p_value)

        corrected_p_values = multipletests(pvals=p_values, alpha=0.05, method="fdr_bh")[1]

        for col in df_results_columns:
            df_results[col].iloc[1] = df_results[col].iloc[1].replace("q=", "q={}".format(corrected_p_values[df_results_columns.index(col)]))

        df_results.to_excel(os.path.join(stat_results_directory, bundle + "_" + side + ".xlsx"), index=True)


