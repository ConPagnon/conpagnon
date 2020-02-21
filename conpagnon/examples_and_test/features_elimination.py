from conpagnon.utils.folders_and_files_management import load_object
import os
import numpy as np
from conpagnon.data_handling import atlas
from nilearn.connectome import sym_matrix_to_vec
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from conpagnon.utils.array_operation import array_rebuilder
from conpagnon.machine_learning.features_indentification import remove_reversed_duplicates
from conpagnon.data_handling.data_management import read_excel_file
import csv
"""
Example of features selection using recursive features elimination (RFE).


Author: Dhaif BEKHA.
"""


# Atlas set up
atlas_folder = '/media/db242421/db242421_data/ConPagnon_data/atlas/atlas_reference'
atlas_name = 'atlas4D_2.nii'
labels_text_file = '/media/db242421/db242421_data/ConPagnon_data/atlas/atlas_reference/atlas4D_2_labels.csv'
colors = ['navy', 'sienna', 'orange', 'orchid', 'indianred', 'olive',
          'goldenrod', 'turquoise', 'darkslategray', 'limegreen', 'black',
          'lightpink']
# Number of regions in each user defined networks
networks = [2, 10, 2, 6, 10, 2, 8, 6, 8, 8, 6, 4]
# Atlas path
# Read labels regions files
atlas_nodes, labels_regions, labels_colors, n_nodes = atlas.fetch_atlas(
    atlas_folder=atlas_folder,
    atlas_name=atlas_name,
    network_regions_number=networks,
    colors_labels=colors,
    labels=labels_text_file,
    normalize_colors=True)


# Load connectivity matrices
data_folder = '/media/db242421/db242421_data/ConPagnon_data/features_identification_results/' \
              'All_impaired_non_impaired_lang'
connectivity_dictionary_name = 'connectivity_matrices_all_impaired_non_impaired.pkl'
subjects_connectivity_matrices = load_object(os.path.join(data_folder,
                                                          connectivity_dictionary_name))

class_names = list(subjects_connectivity_matrices.keys())
metric = 'tangent'

# Vectorize the connectivity for classification
vectorized_connectivity_matrices = sym_matrix_to_vec(
    np.array([subjects_connectivity_matrices[class_name][s][metric] for class_name
              in class_names for s in subjects_connectivity_matrices[class_name].keys()]),
    discard_diagonal=True)

# Stacked the 2D array of connectivity matrices for each subjects
stacked_connectivity_matrices = np.array([subjects_connectivity_matrices[class_name][s][metric]
                                          for class_name in class_names
                                          for s in subjects_connectivity_matrices[class_name].keys()])

# Labels vectors
class_labels = np.hstack((1*np.ones(len(subjects_connectivity_matrices[class_names[0]].keys())),
                          -1*np.ones(len(subjects_connectivity_matrices[class_names[1]].keys()))))

# Compute mean connectivity matrices for each class
first_class_mean_matrix = np.array([subjects_connectivity_matrices[class_names[0]][s][metric] for s in
                                    subjects_connectivity_matrices[class_names[0]].keys()]).mean(axis=0)
second_class_mean_matrix = np.array([subjects_connectivity_matrices[class_names[1]][s][metric] for s in
                                     subjects_connectivity_matrices[class_names[1]].keys()]).mean(axis=0)


estimator = LinearSVC()

# Number of most informative features you want
top_features_number = 4
selector = RFE(estimator, n_features_to_select=top_features_number)
selector = selector.fit(X=vectorized_connectivity_matrices,
                        y=class_labels)

# Ranking array
ranking_array = array_rebuilder(vectorized_array=selector.ranking_,
                                array_type='numeric',
                                diagonal=np.zeros(n_nodes))


top_ranks_indices_array = np.array(np.where(ranking_array == 1)).T

top_ranks_indices = np.array(list(remove_reversed_duplicates(top_ranks_indices_array)))
labels_regions = np.array(labels_regions)
most_informative_features = np.array(
    [(labels_regions[top_ranks_indices[i][0]], labels_regions[top_ranks_indices[i][1]])
     for i in range(top_ranks_indices.shape[0])])

# print the most informative features in you're data
print('The {} most informative features are:'.format(top_features_number))
for i in range(top_ranks_indices.shape[0]):
    print('{} <---> {}'.format(most_informative_features[i, 0], most_informative_features[i, 1]))

# Build a csv file with indices of interest
save_directory = '/media/db242421/db242421_data/ConPagnon_data/RFE/Lang_impaired_non_impaired'
subjects_data = '/media/db242421/db242421_data/ConPagnon_data/regression_data/regression_data.xlsx'

regions_to_write = top_ranks_indices
patients_ids = []
patients_matrices = []
for class_name in class_names:
    n_subject = len(subjects_connectivity_matrices[class_name].keys())
    class_subject_list = list(subjects_connectivity_matrices[class_name].keys())
    for s in class_subject_list:
        # append the id
        patients_ids.append(s)
        # append the matrices
        patients_matrices.append(subjects_connectivity_matrices[class_name][s][metric])

patients_matrices = np.array(patients_matrices)
language_scores = read_excel_file(subjects_data,
                                  sheetname='cohort_functional_data')['language_score']

gender = read_excel_file(subjects_data,
                         sheetname='cohort_functional_data')['Sexe']

lesion_volume = read_excel_file(subjects_data,
                                sheetname='cohort_functional_data')['lesion_normalized']

language_profil = read_excel_file(subjects_data,
                                  sheetname='cohort_functional_data')['langage_clinique']
lesion_side =  read_excel_file(subjects_data,
                                  sheetname='cohort_functional_data')['Lesion']
header = ['subjects', 'connectivity', 'language_score', 'gender', 'lesion_volume', 'language_profil',
          'lesion_side']
for region in regions_to_write:
    region_csv = os.path.join(save_directory,
                              labels_regions[region[0]] + '_' + labels_regions[region[1]] + '.csv')
    with open(os.path.join(region_csv), "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(header)
        for line in range(len(patients_ids)):
            writer.writerow([patients_ids[line], patients_matrices[line, region[0], region[1]],
                             language_scores.loc[patients_ids[line]], gender[patients_ids[line]],
                            lesion_volume[patients_ids[line]], language_profil[patients_ids[line]],
                             lesion_side[patients_ids[line]]])
