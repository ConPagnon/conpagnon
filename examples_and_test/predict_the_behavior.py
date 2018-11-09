import importlib
from data_handling import atlas, data_architecture, dictionary_operations
from utils import folders_and_files_management
from computing import compute_connectivity_matrices as ccm
from sklearn import covariance
from machine_learning import classification
from connectivity_statistics import parametric_tests
from plotting import display
from matplotlib.backends import backend_pdf
from data_handling import data_management
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from connectivity_statistics import regression_analysis_model
from plotting.display import t_and_p_values_barplot
from scipy.stats import ttest_rel
from matplotlib.backends.backend_pdf import PdfPages
# Reload all module
importlib.reload(data_management)
importlib.reload(atlas)
importlib.reload(display)
importlib.reload(parametric_tests)
importlib.reload(ccm)
importlib.reload(folders_and_files_management)
importlib.reload(classification)
importlib.reload(data_architecture)
importlib.reload(dictionary_operations)

# Some path
root_directory = "/media/db242421/db242421_data/ConPagnon_data/language_study_ANOVA_ACM_controls"
kinds = ['correlation', 'partial correlation', 'tangent']
network_name = ['DMN', 'Executive', 'Language',
                'MTL', 'Primary_Visual',
                'Salience', 'Secondary_Visual',
                'Sensorimotor', 'Visuospatial']

subjects_connectivity_matrices = folders_and_files_management.load_object(
    full_path_to_object=os.path.join(root_directory,
                                     "dictionary/z_fisher_transform_subjects_connectivity_matrices.pkl")
)
groups = list(subjects_connectivity_matrices.keys())
# Load the behavior data
behavior_data = pd.read_csv(os.path.join(root_directory, 'behavioral_data.csv'))
behavior_data = data_management.shift_index_column(panda_dataframe=behavior_data,
                                                   columns_to_index=['subjects'])
# Load the atlas data
atlas_excel_file = '/media/db242421/db242421_data/atlas_AVCnn/atlas_version2.xlsx'
sheetname = 'complete_atlas'

atlas_information = pd.read_excel(atlas_excel_file, sheetname=sheetname)

# Load connectivity data
model_to_load = ['contra_intra']
network_to_load = 'Visuospatial'

# Choose the connectivity measure for the analysis
kind = 'tangent'

# Asymmetry show the visuospatial network in the contralesional hemisphere
# stands out. can we classify the impaired language and
# non impaired language based on that ?
network_indices = np.array(atlas_information[atlas_information['network'] == network_to_load]['atlas4D index'])
# Compute the network matrices
network_connectivity_matrices = ccm.extract_sub_connectivity_matrices(
    subjects_connectivity_matrices=subjects_connectivity_matrices,
    kinds=[kind],
    regions_index=network_indices,
    vectorize=True,
    discard_diagonal=True
)
class_labels = np.hstack((1*np.ones(len(subjects_connectivity_matrices[groups[0]].keys())),
                          2*np.ones(len(subjects_connectivity_matrices[groups[1]].keys())),
                          3*np.ones(len(subjects_connectivity_matrices[groups[2]].keys()))))
# Let perform some basic classification
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, LeaveOneOut, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


mean_scores = []
svc = LinearSVC()
sss = StratifiedShuffleSplit(n_splits=10000)
loo = LeaveOneOut()
lr = LogisticRegression()
# Final mean accuracy scores will be stored in a dictionary
mean_score_dict = {}
groups_classification = ['controls', 'non_impaired_language']


features = np.array([network_connectivity_matrices[group][subject][kind]
                    for group in groups_classification
                    for subject in network_connectivity_matrices[group].keys()],
                                      )
features_labels = np.hstack((1*np.ones(len(subjects_connectivity_matrices[groups_classification[0]].keys())),
                             2*np.ones(len(subjects_connectivity_matrices[groups_classification[1]].keys()))))

search_C = GridSearchCV(estimator=lr,param_grid={'C': np.linspace(start=0.001, stop=1e3, num=100)},
             scoring='accuracy', cv=sss, n_jobs=16, verbose=1)
if __name__ == '__main__':
    search_C.fit(X=features, y=features_labels)

print("Classification between {} and {} with a L2 linear SVM achieved the best score of {} % accuracy "
      "with C={}".format(groups_classification[0], groups_classification[1],
                         search_C.best_score_*100, search_C.best_params_['C']))

C = search_C.best_params_['C']
cv_scores = cross_val_score(estimator=LinearSVC(C=C), X=features,
                            y=features_labels, cv=sss,
                            scoring='accuracy', n_jobs=16,
                            verbose=1)
print('mean accuracy {} % +- {} %'.format(cv_scores.mean()*100, cv_scores.std()*100))

