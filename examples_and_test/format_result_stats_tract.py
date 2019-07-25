import pandas as pd
import re

"""
Format R results from statistical 
analysis of bundle parameters from
NAIS cohort.

"""

stat_results_directory = "/media/db242421/Samsung_T5/Work/Neurospin/AVCnn/Ines_2018/stats_tracts_2"
model_name = "_Ipsi_contra_Parole_Zscore_laure"
parameter = ["FA"]
side = ["ipsi", "contra"]
bundle_name = ["AF", "MLF", "SLF"]

variables = ["Lesion_ratio"]

one_result_file = "/media/db242421/Samsung_T5/Work/Neurospin/AVCnn/Ines_2018/stats_tracts_2/" \
                  "AF_contra_FA__Ipsi_Contra_Parole_Zscore_laure.txt"

with open(one_result_file) as result_file:
    result_file_content = result_file.read()
    print(result_file_content)

lesion_ratio_pattern = r"Lesion_ratio\s\s\s[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?.*$"

test = re.compile(lesion_ratio_pattern)

with open(one_result_file) as result_file:
    for line in result_file:
        match = test.search(line)
        if match:
            match_result = line
            parse_line = match_result.split()[-2:]


