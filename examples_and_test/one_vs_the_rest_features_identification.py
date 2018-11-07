from utils.folders_and_files_management import load_object
import os
import numpy as np
from machine_learning.features_indentification import bootstrap_svc, \
     permutation_bootstrap_svc, features_weights_max_t_correction, \
     features_weights_parametric_correction, find_significant_features_indices, find_top_features
import psutil
import time
import matplotlib.pyplot as plt
from utils.array_operation import array_rebuilder
from nilearn.plotting import plot_connectome
from data_handling import atlas
from plotting.display import plot_matrix
from utils.folders_and_files_management import save_object
from matplotlib.backends.backend_pdf import PdfPages
from nilearn.connectome import sym_matrix_to_vec
from machine_learning.features_indentification import one_against_all_bootstrap, one_against_all_permutation_bootstrap

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
data_folder = '/media/db242421/db242421_data/ConPagnon_data/language_study_ANOVA_ACM_controls/dictionary'
connectivity_dictionary_name = 'z_fisher_transform_subjects_connectivity_matrices.pkl'
subjects_connectivity_matrices = load_object(os.path.join(data_folder,
                                                          connectivity_dictionary_name))
save_figures = '/media/db242421/db242421_data/ConPagnon_data/language_study_ANOVA_ACM_controls/' \
               'discriminative_connection_identification/one_vs_the_rest'

class_names = ['controls', 'impaired_language', 'non_impaired_language']
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
                          2*np.ones(len(subjects_connectivity_matrices[class_names[1]].keys())),
                          3*np.ones(len(subjects_connectivity_matrices[class_names[2]].keys()))))


from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier




svc = LinearSVC()
one_vs_the_rest_classifier = OneVsRestClassifier(estimator=svc)
one_vs_one_classifier = OneVsOneClassifier(estimator=svc)

one_vs_the_rest_fit = one_vs_the_rest_classifier.fit(X=vectorized_connectivity_matrices,
                                                     y=class_labels)
one_vs_one_fit = one_vs_one_classifier.fit(X=vectorized_connectivity_matrices,
                                           y=class_labels)

one_vs_the_rest_weights = one_vs_the_rest_fit.coef_
# Plot the weight for each class
with PdfPages(os.path.join(save_figures, 'one_vs_the_rest.pdf')) as pdf:

    for class_name in class_names:
        # Reshape the array
        class_weights = array_rebuilder(
            vectorized_array=one_vs_the_rest_weights[class_names.index(class_name)],
            array_type='numeric',
            diagonal=np.zeros(stacked_connectivity_matrices.shape[1]))
        # Find the top features weights
        class_weight_array_top_features, top_weights, top_coefficients_indices, top_weight_labels = \
            find_top_features(normalized_mean_weight_array=class_weights,
                              labels_regions=labels_regions,
                              top_features_number=5)
        plt.figure()
        plot_connectome(adjacency_matrix=class_weights,
                        node_coords=atlas_nodes, colorbar=True,
                        title='{} weights'.format(class_name),
                        node_size=15,
                        node_color=labels_colors)
        pdf.savefig()
        plt.show()

        plt.figure()
        plot_connectome(adjacency_matrix=class_weight_array_top_features,
                        node_coords=atlas_nodes, colorbar=True,
                        title='{} top weights'.format(class_name),
                        node_size=15,
                        node_color=labels_colors)
        pdf.savefig()
        plt.show()

from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=10000)

scores = cross_val_score(estimator=one_vs_the_rest_classifier, X=vectorized_connectivity_matrices,
                         y=class_labels, cv=sss, verbose=1, n_jobs=4)
mean_score = scores.mean()
std_score = scores.std()

print('One vs All classification accuracy: {} % +- {} %'.format(mean_score*100,
                                                                std_score*100))

n_subjects = vectorized_connectivity_matrices.shape[0]
bootstrap_number = 100
n_permutations = 10
n_classes = len(np.unique(class_labels))
# Indices to bootstrap
indices = np.arange(n_subjects)
# Generate a matrix containing all bootstrapped indices
bootstrap_matrix = np.random.choice(a=indices, size=(bootstrap_number, n_subjects),
                                    replace=True)

# Generate a permuted class labels array
class_labels_permutation_matrix = np.array([np.random.permutation(class_labels)
                                            for n in range(n_permutations)])

bootstrap_array_perm = np.random.choice(a=indices,
                                        size=(n_permutations, bootstrap_number,
                                              n_subjects),
                                        replace=True)

bootstrap_weights = one_against_all_bootstrap(features=vectorized_connectivity_matrices,
                                                        class_labels=class_labels,
                                                        bootstrap_array_indices=bootstrap_matrix,
                                                        n_cpus_bootstrap=8,
                                                        verbose=1)

normalized_mean_weight = bootstrap_weights.mean(axis=0)/bootstrap_weights.std(axis=0)

save_object(object_to_save=normalized_mean_weight, saving_directory=save_figures,
            filename='normalized_mean_weight.pkl')

null_distribution = one_against_all_permutation_bootstrap(features=vectorized_connectivity_matrices,
                                                          class_labels_perm=class_labels_permutation_matrix,
                                                          bootstrap_array_perm=bootstrap_array_perm,
                                                          n_classes=3,
                                                          n_permutations=n_permutations,
                                                          n_cpus_bootstrap=8,
                                                          verbose_bootstrap=1,
                                                          verbose_permutation=1)

# Save the null distribution to avoid
save_object(object_to_save=null_distribution, saving_directory=save_figures,
            filename='null_distribution.pkl')

# Rebuild a symmetric array from normalized mean weight vector for each class
normalized_mean_weight_array = np.zeros((n_classes, n_nodes, n_nodes))
for group in range(n_classes):

    normalized_mean_weight_array[group, ...] = array_rebuilder(normalized_mean_weight[group, ...],
                                                   'numeric', diagonal=np.zeros(n_nodes))
# Find top features for each class
all_classes_normalized_mean_weight_array_top_features = np.zeros((n_classes, n_nodes, n_nodes))
all_classes_top_weights = []
all_classes_top_coefficients_indices = []
all_classes_top_weight_labels = []
for group in range(n_classes):

    normalized_mean_weight_array_top_features, top_weights, top_coefficients_indices, top_weight_labels = \
        find_top_features(normalized_mean_weight_array=normalized_mean_weight_array[group, ...],
                          labels_regions=labels_regions)
    all_classes_normalized_mean_weight_array_top_features[group, ...] = normalized_mean_weight_array_top_features
    all_classes_top_weights.append(top_weights)
    all_classes_top_coefficients_indices.append(top_coefficients_indices)
    all_classes_top_weight_labels.append(top_weight_labels)

# Convert list to array for later purposes
all_classes_top_weights = np.array(all_classes_top_weights)
all_classes_top_coefficients_indices = np.array(all_classes_top_coefficients_indices)
all_classes_top_weight_labels = np.array(all_classes_top_weight_labels)
