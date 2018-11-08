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
# TODO: Add function to wrap the generation of a report, correction, and classifier fitting 
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

from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, cross_validate

sss = StratifiedShuffleSplit(n_splits=10000)

scores = cross_val_score(estimator=one_vs_the_rest_classifier, X=vectorized_connectivity_matrices,
                         y=class_labels, cv=sss, verbose=1, n_jobs=4, groups=class_labels)
mean_score = scores.mean()
std_score = scores.std()

print('One vs All classification accuracy: {} % +- {} %'.format(mean_score*100,
                                                                std_score*100))

n_subjects = vectorized_connectivity_matrices.shape[0]
bootstrap_number = 500
n_permutations = 1000
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
# Correction method
correction = 'fdr_bh'
n_physical = psutil.cpu_count(logical=False)
alpha = .05

top_features_number = 50

node_size = 15
bootstrap_weights = one_against_all_bootstrap(features=vectorized_connectivity_matrices,
                                              class_labels=class_labels,
                                              bootstrap_array_indices=bootstrap_matrix,
                                              n_cpus_bootstrap=n_physical,
                                              verbose=1)

normalized_mean_weight = bootstrap_weights.mean(axis=0)/bootstrap_weights.std(axis=0)

save_object(object_to_save=normalized_mean_weight, saving_directory=save_figures,
            filename='normalized_mean_weight.pkl')

null_distribution = one_against_all_permutation_bootstrap(features=vectorized_connectivity_matrices,
                                                          class_labels_perm=class_labels_permutation_matrix,
                                                          bootstrap_array_perm=bootstrap_array_perm,
                                                          n_classes=3,
                                                          n_permutations=n_permutations,
                                                          n_cpus_bootstrap=n_physical,
                                                          verbose_bootstrap=1,
                                                          verbose_permutation=1)

# Save the null distribution to avoid
save_object(object_to_save=null_distribution, saving_directory=save_figures,
            filename='null_distribution.pkl')

# Rebuild a symmetric array from normalized mean weight vector for each class
normalized_mean_weight_array = np.zeros((n_classes, n_nodes, n_nodes))
for group in range(n_classes):

    normalized_mean_weight_array[group, ...] = array_rebuilder(normalized_mean_weight[group, ...],
                                                               'numeric',
                                                               diagonal=np.zeros(n_nodes))
# Find top features for each class
all_classes_normalized_mean_weight_array_top_features = np.zeros((n_classes, n_nodes, n_nodes))
all_classes_top_weights = []
all_classes_top_coefficients_indices = []
all_classes_top_weight_labels = []
for group in range(n_classes):

    normalized_mean_weight_array_top_features, top_weights, top_coefficients_indices, top_weight_labels = \
        find_top_features(normalized_mean_weight_array=normalized_mean_weight_array[group, ...],
                          labels_regions=labels_regions,
                          top_features_number=top_features_number)
    all_classes_normalized_mean_weight_array_top_features[group, ...] = normalized_mean_weight_array_top_features
    all_classes_top_weights.append(top_weights)
    all_classes_top_coefficients_indices.append(top_coefficients_indices)
    all_classes_top_weight_labels.append(top_weight_labels)

# Convert list to array for later purposes
all_classes_top_weights = np.array(all_classes_top_weights)
all_classes_top_coefficients_indices = np.array(all_classes_top_coefficients_indices)
all_classes_top_weight_labels = np.array(all_classes_top_weight_labels)

if correction == 'max_t':
    all_classes_sorted_null_maximum_dist = []
    all_classes_sorted_null_minimum_dist = []
    all_classes_p_value_positive_weights = []
    all_classes_p_value_negative_weights = []
    all_classes_p_negative_features_significant = []
    all_classes_p_positive_features_significant = []
    all_classes_significant_positive_features_indices = []
    all_classes_significant_negative_features_indices = []
    all_classes_significant_positive_features_labels = []
    all_classes_significant_negative_features_labels = []
    all_classes_p_all_significant_features = []
    for group in range(n_classes):

        # Corrected p values with the maximum statistic for each classes
        sorted_null_maximum_dist, sorted_null_minimum_dist, p_value_positive_weights, p_value_negative_weights = \
            features_weights_max_t_correction(null_distribution_features_weights=null_distribution[:, group, :],
                                              normalized_mean_weight=normalized_mean_weight[group, ...])
        all_classes_sorted_null_maximum_dist.append(sorted_null_maximum_dist)
        all_classes_sorted_null_minimum_dist.append(sorted_null_minimum_dist)
        all_classes_p_value_positive_weights.append(p_value_positive_weights)
        all_classes_p_value_negative_weights.append(p_value_negative_weights)

        # Rebuild vectorized p values array
        p_max_values_array = array_rebuilder(vectorized_array=p_value_positive_weights,
                                             array_type='numeric',
                                             diagonal=np.ones(n_nodes))

        p_min_values_array = array_rebuilder(vectorized_array=p_value_negative_weights,
                                             array_type='numeric',
                                             diagonal=np.ones(n_nodes))

        # Find p-values under the alpha threshold
        p_negative_features_significant = np.array(p_min_values_array < alpha, dtype=int)
        p_positive_features_significant = np.array(p_max_values_array < alpha, dtype=int)

        p_all_significant_features = p_positive_features_significant + p_negative_features_significant

        all_classes_p_all_significant_features.append(p_all_significant_features)
        all_classes_p_negative_features_significant.append(p_negative_features_significant)
        all_classes_p_positive_features_significant.append(p_positive_features_significant)

        significant_positive_features_indices, significant_negative_features_indices, \
        significant_positive_features_labels, significant_negative_features_labels = find_significant_features_indices(
            p_positive_features_significant=p_positive_features_significant,
            p_negative_features_significant=p_negative_features_significant,
            features_labels=labels_regions)

        all_classes_significant_positive_features_indices.append(significant_positive_features_indices)
        all_classes_significant_negative_features_indices.append(significant_negative_features_indices)
        all_classes_significant_positive_features_labels.append(significant_positive_features_labels)
        all_classes_significant_negative_features_labels.append(significant_negative_features_labels)

    # Convert all output to array for convenience
    all_classes_sorted_null_maximum_dist = np.array(all_classes_sorted_null_maximum_dist)
    all_classes_sorted_null_minimum_dist = np.array(all_classes_sorted_null_minimum_dist)
    all_classes_p_value_positive_weights = np.array(all_classes_p_value_positive_weights)
    all_classes_p_value_negative_weights = np.array(all_classes_p_value_negative_weights)
    all_classes_p_negative_features_significant = np.array(all_classes_p_negative_features_significant)
    all_classes_p_positive_features_significant = np.array(all_classes_p_positive_features_significant)
    all_classes_significant_positive_features_indices = np.array(all_classes_significant_positive_features_indices)
    all_classes_significant_negative_features_indices = np.array(all_classes_significant_negative_features_indices)
    all_classes_significant_positive_features_labels = np.array(all_classes_significant_positive_features_labels)
    all_classes_significant_negative_features_labels = np.array(all_classes_significant_negative_features_labels)
    all_classes_p_all_significant_features = np.array(all_classes_p_all_significant_features)

    with PdfPages(os.path.join(save_figures, 'one_against_all_controls_impaired_non_impaired.pdf')) as pdf:
        for group in range(n_classes):

            # Plot the estimated null distribution
            plt.figure(constrained_layout=True)
            plt.hist(all_classes_sorted_null_maximum_dist[group, ...], 'auto', histtype='bar', alpha=0.5,
                     edgecolor='black')
            # The five 5% extreme values among maximum distribution
            p95 = np.percentile(all_classes_sorted_null_maximum_dist[group, ...], q=95)
            plt.axvline(x=p95, color='black')
            plt.title('Null distribution of maximum normalized weight mean for {}'.format(class_names[group]))
            pdf.savefig()

            plt.figure()
            plt.hist(all_classes_sorted_null_minimum_dist[group, ...], 'auto', histtype='bar',
                     alpha=0.5, edgecolor='black')
            # The five 5% extreme values among minimum distribution
            p5 = np.percentile(all_classes_sorted_null_minimum_dist[group, ...], q=5)
            plt.axvline(x=p5, color='black')
            plt.title('Null distribution of minimum normalized weight mean for {}'.format(class_names[group]))
            pdf.savefig()

            plt.figure()
            plot_connectome(adjacency_matrix=normalized_mean_weight_array[group, ...],
                            node_coords=atlas_nodes, colorbar=True,
                            title='Weights connectome for {}'.format(class_names[group]),
                            node_size=node_size,
                            node_color=labels_colors)
            pdf.savefig()

            # plot the top weight in a histogram fashion
            fig = plt.figure(figsize=(15, 10), constrained_layout=True)

            weight_colors = ['blue' if weight < 0 else 'red' for weight in all_classes_top_weights[group, ...]]
            plt.bar(np.arange(len(all_classes_top_weights[group, ...])), list(all_classes_top_weights[group, ...]),
                    color=weight_colors,
                    edgecolor='black',
                    alpha=0.5)
            plt.xticks(np.arange(0, len(all_classes_top_weights[group, ...])),
                       all_classes_top_weight_labels[group, ...],
                       rotation=60,
                       ha='right')
            for label in range(len(plt.gca().get_xticklabels())):
                plt.gca().get_xticklabels()[label].set_color(weight_colors[label])
            plt.xlabel('Features names')
            plt.ylabel('Features weights')
            plt.title('Top {} features ranking of normalized mean weight for {}'.format(top_features_number,
                                                                                        class_names[group]))
            pdf.savefig()

            plt.figure()
            plot_connectome(adjacency_matrix=all_classes_normalized_mean_weight_array_top_features[group, ...],
                            node_coords=atlas_nodes,
                            colorbar=True,
                            title='Top {} features weight for {}'.format(top_features_number, class_names[group]),
                            node_size=node_size,
                            node_color=labels_colors)
            pdf.savefig()

            if np.where(all_classes_p_positive_features_significant[group, ...] == 1)[0].size != 0:
                # Plot on glass brain the significant positive features weight
                plt.figure()
                plot_connectome(adjacency_matrix=all_classes_p_positive_features_significant[group, ...],
                                node_coords=atlas_nodes, colorbar=True,
                                title='Significant positive weight for {}'.format(class_names[group]),
                                edge_cmap='Reds',
                                node_size=node_size,
                                node_color=labels_colors)
                pdf.savefig()

                # Matrix view of significant positive and negative weight
                plt.figure()
                plot_matrix(matrix=all_classes_p_positive_features_significant[group, ...],
                            labels_colors=labels_colors, mpart='all',
                            colormap='Blues', linecolor='black',
                            title='Significant negative weight for {}'.format(class_names[group]),
                            vertical_labels=labels_regions, horizontal_labels=labels_regions)
                pdf.savefig()

            if np.where(all_classes_p_negative_features_significant[group, ...] == 1)[0].size != 0:
                # Plot on glass brain the significant negative features weight
                plt.figure()
                plot_connectome(adjacency_matrix=all_classes_p_negative_features_significant[group, ...],
                                node_coords=atlas_nodes, colorbar=True,
                                title='Significant negative weight for {}'.format(class_names[group]),
                                edge_cmap='Blues',
                                node_size=node_size,
                                node_color=labels_colors)
                pdf.savefig()

                plt.figure()
                plot_matrix(matrix=all_classes_p_negative_features_significant[group, ...],
                            labels_colors=labels_colors, mpart='all',
                            colormap='Reds', linecolor='black',
                            title='Significant positive weight for {}'.format(class_names[group]),
                            vertical_labels=labels_regions, horizontal_labels=labels_regions)
                pdf.savefig()

                plt.close("all")
else:
    all_classes_p_values_corrected = []
    all_classes_p_values_corrected_array = []
    all_classes_p_negative_features_significant = []
    all_classes_p_positive_features_significant = []
    all_classes_significant_positive_features_indices = []
    all_classes_significant_negative_features_indices = []
    all_classes_significant_positive_features_labels = []
    all_classes_significant_negative_features_labels = []
    all_classes_p_all_significant_features = []

    # Perform another type of correction like FDR, ....
    for group in range(n_classes):

        p_values_corrected = features_weights_parametric_correction(
            null_distribution_features_weights=null_distribution[:, group, :],
            normalized_mean_weight=normalized_mean_weight[group, ...],
            method=correction)

        p_values_corrected_array = array_rebuilder(vectorized_array=p_values_corrected,
                                                   array_type='numeric',
                                                   diagonal=np.ones(n_nodes))

        # Find p values under alpha threshold for negative and positive weight features
        p_negative_features_significant = np.array(
            (p_values_corrected_array < alpha) & (normalized_mean_weight_array[group, ...] < 0),
            dtype=int)
        p_positive_features_significant = np.array(
            (p_values_corrected_array < alpha) & (normalized_mean_weight_array[group, ...] > 0),
            dtype=int)

        significant_positive_features_indices, significant_negative_features_indices, \
        significant_positive_features_labels, significant_negative_features_labels = find_significant_features_indices(
            p_positive_features_significant=p_positive_features_significant,
            p_negative_features_significant=p_negative_features_significant,
            features_labels=labels_regions)

        # Mask for all significant connection, for both,
        # positive and negative weight
        p_all_significant_features = p_positive_features_significant + p_negative_features_significant

        all_classes_p_values_corrected.append(p_values_corrected)
        all_classes_p_values_corrected_array.append(p_values_corrected_array)
        all_classes_p_positive_features_significant.append(p_positive_features_significant)
        all_classes_p_negative_features_significant.append(p_negative_features_significant)
        all_classes_significant_positive_features_indices.append(significant_positive_features_indices)
        all_classes_significant_negative_features_indices.append(significant_negative_features_indices)
        all_classes_significant_positive_features_labels.append(significant_positive_features_labels)
        all_classes_significant_negative_features_labels.append(significant_negative_features_labels)
        all_classes_p_all_significant_features.append(p_all_significant_features)

    all_classes_p_values_corrected = np.array(all_classes_p_values_corrected)
    all_classes_p_values_corrected_array = np.array(all_classes_p_values_corrected_array)
    all_classes_p_positive_features_significant = np.array(all_classes_p_positive_features_significant)
    all_classes_p_negative_features_significant = np.array(all_classes_p_negative_features_significant)
    all_classes_significant_positive_features_indices = np.array(all_classes_significant_positive_features_indices)
    all_classes_significant_negative_features_indices = np.array(all_classes_significant_negative_features_indices)
    all_classes_significant_positive_features_labels = np.array(all_classes_significant_positive_features_labels)
    all_classes_significant_negative_features_labels = np.array(all_classes_significant_negative_features_labels)
    all_classes_p_all_significant_features = np.array(all_classes_p_all_significant_features)

    # plot the top weight in a histogram fashion
    with PdfPages(os.path.join(save_figures, correction + '_one_against_all.pdf')) as pdf:
        for group in range(n_classes):

            plt.figure()
            plot_connectome(adjacency_matrix=normalized_mean_weight_array[group, ...],
                            node_coords=atlas_nodes,
                            colorbar=True,
                            title='Features weight for {}'.format(class_names[group]),
                            node_size=node_size,
                            node_color=labels_colors)
            pdf.savefig()

            fig = plt.figure(figsize=(15, 10), constrained_layout=True)
            weight_colors = ['blue' if weight < 0 else 'red' for weight in all_classes_top_weights[group, ...]]
            plt.bar(np.arange(len(all_classes_top_weights[group, ...])), list(all_classes_top_weights[group, ...]),
                    color=weight_colors,
                    edgecolor='black',
                    alpha=0.5)
            plt.xticks(np.arange(0, len(all_classes_top_weights[group, ...])),
                       all_classes_top_weight_labels[group, ...],
                       rotation=60,
                       ha='right')
            for label in range(len(plt.gca().get_xticklabels())):
                plt.gca().get_xticklabels()[label].set_color(weight_colors[label])
            plt.xlabel('Features names')
            plt.ylabel('Features weights')
            plt.title('Top {} features ranking of normalized mean weight for {}'.format(top_features_number,
                                                                                        class_names[group]))
            pdf.savefig()

            plt.figure()
            plot_connectome(adjacency_matrix=all_classes_normalized_mean_weight_array_top_features[group, ...],
                            node_coords=atlas_nodes, colorbar=True,
                            title='Top {} features weight for {}'.format(top_features_number,
                                                                         class_names[group]),
                            node_size=node_size,
                            node_color=labels_colors)
            pdf.savefig()
            if np.where(all_classes_p_positive_features_significant[group, ...] == 1)[0].size != 0:

                # Plot on glass brain the significant positive features weight
                plt.figure()
                plot_connectome(adjacency_matrix=all_classes_p_positive_features_significant[group, ...],
                                node_coords=atlas_nodes, colorbar=True,
                                title='Significant positive weight after {} correction for {}'.format(
                                    correction,
                                    class_names[group]),
                                edge_cmap='Reds',
                                node_size=node_size,
                                node_color=labels_colors)

                pdf.savefig()

                plt.figure()
                plot_matrix(matrix=all_classes_p_positive_features_significant[group, ...],
                            labels_colors=labels_colors, mpart='all',
                            colormap='Reds', linecolor='black',
                            title='Significant positive weight after {} correction for {}'.format(correction,
                                                                                                  class_names[group]),
                            vertical_labels=labels_regions, horizontal_labels=labels_regions)
                pdf.savefig()
            if np.where(all_classes_p_negative_features_significant[group, ...] == 1)[0].size != 0:
                # Plot on glass brain the significant negative features weight
                plt.figure()
                plot_connectome(adjacency_matrix=all_classes_p_negative_features_significant[group, ...],
                                node_coords=atlas_nodes, colorbar=True,
                                title='Significant negative weight after {} correction for {}'.format(correction,
                                                                                                      class_names[group]),
                                edge_cmap='Blues',
                                node_size=node_size,
                                node_color=labels_colors)
                pdf.savefig()

                # Matrix view of significant positive and negative weight
                plt.figure()
                plot_matrix(matrix=all_classes_p_negative_features_significant[group, ...],
                            labels_colors=labels_colors, mpart='all',
                            colormap='Blues', linecolor='black',
                            title='Significant negative weight after {} correction for {}'.format(
                                correction,
                                class_names[group]),
                            vertical_labels=labels_regions, horizontal_labels=labels_regions)
                pdf.savefig()

                plt.close("all")



