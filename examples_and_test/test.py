import importlib
from data_handling import atlas, data_architecture, dictionary_operations
from utils import folders_and_files_management
from computing import compute_connectivity_matrices as ccm
from machine_learning import classification
from connectivity_statistics import parametric_tests
from plotting import display
from data_handling import data_management
from utils.folders_and_files_management import load_object
import os
import numpy as np
import pandas as pd
from nilearn.connectome import sym_matrix_to_vec
from sklearn.model_selection import LeaveOneOut, StratifiedShuffleSplit, KFold, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
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


# data_directory = '/media/db242421/db242421_data/ConPagnon_data/patients_ACM_controls/dictionary'
data_directory = 'D:\\test_language_ml\\dictionary'
output_directory = 'D:\\test_language_ml'
kind = 'tangent'
# Load subjects matrices: just ACM lesion here
Z_subjects_matrices = load_object(os.path.join(data_directory,
                                               'z_fisher_transform_subjects_connectivity_matrices.pkl'))

# Load behavioral data file
behavioral_data = 'D:\\test_language_ml\\behavioral_data.xlsx'
regression_data_file = data_management.read_excel_file(
    excel_file_path=behavioral_data,
    sheetname='cohort_functional_data')

# Load atlas ROIs labels for writing purpose
# Atlas excel information file
atlas_excel_file = 'D:\\atlas_AVCnn\\atlas_version2.xlsx'
sheetname = 'complete_atlas'
atlas_information = pd.read_excel(atlas_excel_file, sheetname='complete_atlas')
atlas_roi_labels = list(atlas_information['anatomical label'])

# Select a subset of patients
# Compute the connectivity matrices dictionary with factor as keys.
group_by_factor_subjects_connectivity, population_df_by_factor, factor_keys =\
    dictionary_operations.groupby_factor_connectivity_matrices(
        population_data_file=behavioral_data,
        sheetname='cohort_functional_data',
        subjects_connectivity_matrices_dictionnary=Z_subjects_matrices,
        groupes=['patients'], factors=['langage_clinique'], drop_subjects_list=None)

# In this dataset, I have Impaired and non Impaired language
groups = ['Non_impaired_language', 'Impaired_language', 'controls']
language_connectivity_dictionary = dict.fromkeys(groups)
# Subjects connectivity matrices for both group
language_connectivity_dictionary['Impaired_language'] = group_by_factor_subjects_connectivity['A']
language_connectivity_dictionary['Non_impaired_language'] = group_by_factor_subjects_connectivity['N']
language_connectivity_dictionary['controls'] = Z_subjects_matrices['controls']
# Subject list for both group
impaired_language_subjects_list = list(language_connectivity_dictionary['Impaired_language'].keys())
non_impaired_language_subjects_list = list(language_connectivity_dictionary['Non_impaired_language'].keys())
controls_subjects_list = list(language_connectivity_dictionary['controls'].keys())

# Stack matrices
class_name_to_classify = ['Non_impaired_language','controls']
vectorized_connectivity_matrices = sym_matrix_to_vec(
    np.array([language_connectivity_dictionary[class_name][s][kind]
              for class_name in class_name_to_classify
              for s in language_connectivity_dictionary[class_name].keys()]),
    discard_diagonal=True)

# Subjects labels
language_subjects_labels = np.hstack((np.zeros(len(non_impaired_language_subjects_list)),
                                      np.ones(len(controls_subjects_list))))

# Run SVM classification
from sklearn import svm
# Leave one out Cross validation scheme
loo = LeaveOneOut()
# Stratified cross validation scheme
sss = StratifiedShuffleSplit(n_splits=10000)
# K-fold evaluation scheme
kfold = KFold(n_splits=5, shuffle=True)
# SVM kernel
kernel = 'linear'
C = 1
# Support Vector Classification object
svc = svm.SVC(C=C, kernel=kernel)
from sklearn.multiclass import OneVsRestClassifier
# Compute classification score between impaired language
# and non impaired language groups
svm_scores = cross_val_score(estimator=svc,
                             X=vectorized_connectivity_matrices,
                             y=language_subjects_labels,
                             cv=loo, scoring='accuracy')

mean_accuracy = round(np.mean(svm_scores), 2)*100

print('Mean prediction accuracy : {}%'.format(mean_accuracy))

# Feature ranking and elimination, and classification: binary task
for train, test in loo.split(X=vectorized_connectivity_matrices, y=language_subjects_labels):
    # Perform two sample t test in training test between
    # the two classes
    class_one
