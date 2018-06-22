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
data_folder = '/media/db242421/db242421_data/ConPagnon_data/features_identification_results/LG_ACM_patients_controls'
connectivity_dictionary_name = 'LG_acm_controls_connectivity_matrices.pkl'
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

# Compute mean connectivity matrices for each class
first_class_mean_matrix = np.array([subjects_connectivity_matrices[class_names[0]][s][metric] for s in
                                    subjects_connectivity_matrices[class_names[0]].keys()]).mean(axis=0)
second_class_mean_matrix = np.array([subjects_connectivity_matrices[class_names[1]][s][metric] for s in
                                     subjects_connectivity_matrices[class_names[1]].keys()]).mean(axis=0)


# choose the top features weight number to plot
top_features_number = 50
node_size = 10
# Labels vectors
class_labels = np.hstack((1*np.ones(len(subjects_connectivity_matrices[class_names[0]].keys())),
                          -1*np.ones(len(subjects_connectivity_matrices[class_names[1]].keys()))))

# Number of Bootstrap (with replacement)
bootstrap_number = 500

# Number of permutation
n_permutations = 1000

# Number of subjects
n_subjects = vectorized_connectivity_matrices.shape[0]

# Type one error rate
alpha = 0.05

# Indices to bootstrap
indices = np.arange(n_subjects)
# Generate a matrix containing all bootstrapped indices
bootstrap_matrix = np.random.choice(a=indices, size=(bootstrap_number, n_subjects),
                                    replace=True)

# Choose multiple comparison correction
correction = 'fdr_bh'

# Compute number of physical and logical core
n_physical = psutil.cpu_count(logical=False)
n_cpu_with_logical = psutil.cpu_count(logical=True)

# Generate a permuted class labels array
class_labels_permutation_matrix = np.array([np.random.permutation(class_labels)
                                            for n in range(n_permutations)])

bootstrap_array_perm = np.random.choice(a=indices,
                                        size=(n_permutations, bootstrap_number,
                                              n_subjects),
                                        replace=True)

save_directory = '/media/db242421/db242421_data/ConPagnon_data/features_identification_results/' \
                 'LG_ACM_patients_controls'

# report name for features visualisation and parameters text report
report_filename = 'features_identification_' + class_names[0] + '_' + class_names[1] + '_' + str(alpha) + \
                  '_' + correction + '.pdf'
text_report_filename = 'features_identification_' + class_names[0] + '_' + class_names[1] + '_' + str(alpha) + \
                  '_' + correction + '.txt'
# Sanity check fo bootstrapped sample class labels for each permutation
print('Check Bootstrapped class labels for each permutation...')
for n in range(n_permutations):
    for b in range(bootstrap_number):
        bootstrapped_permuted_labels = class_labels_permutation_matrix[n, bootstrap_array_perm[n, b, ...]]
        count_labels_occurence = len(np.unique(bootstrapped_permuted_labels))
        if count_labels_occurence == 2:
            pass
        else:
            print('For the permutation # {}, the bootstrapped samples # {} is ill...'.format(n, b))
            print(b)
            # We replace the problematic bootstrap
            new_bootstrap_indices = np.random.choice(a=indices, size=len(indices),
                                                     replace=True)
            bootstrap_array_perm[n , b, ...] = new_bootstrap_indices
            new_bootstrap_class_labels_permuted = class_labels_permutation_matrix[n, new_bootstrap_indices]
            class_labels_permutation_matrix[n, ...] = new_bootstrap_class_labels_permuted

print('Verifying that bootstrapped sample labels classes contain two labels....')
for n in range(n_permutations):
    for b in range(bootstrap_number):
        bootstrapped_permuted_labels = class_labels_permutation_matrix[n, bootstrap_array_perm[n, b, ...]]
        count_labels_occurence = len(np.unique(bootstrapped_permuted_labels))
        if count_labels_occurence < 2:
            raise ValueError('Sample size seems too small to generate clean bootstrapped class labels \n '
                             'with at least two classes !')
        else:
            pass
print('Done checking bootstrapped class labels for each permutation.')


if __name__ == '__main__':
    # True bootstrap weight
    tic_bootstrap = time.time()
    print('Performing classification on {} bootstrap sample...'.format(bootstrap_number))
    bootstrap_weight = bootstrap_svc(features=vectorized_connectivity_matrices,
                                     class_labels=class_labels,
                                     bootstrap_array_indices=bootstrap_matrix,
                                     n_cpus_bootstrap=n_physical,
                                     verbose=1)
    tac_bootstrap = time.time()
    t_bootstrap = tac_bootstrap - tic_bootstrap

    normalized_mean_weight = bootstrap_weight.mean(axis=0)/bootstrap_weight.std(axis=0)

    save_object(object_to_save=normalized_mean_weight, saving_directory=save_directory,
                filename='normalized_mean_weight_' + class_names[0] + '_' + class_names[1] + '.pkl')
    # Estimation of null distribution of normalized mean weight
    null_distribution = permutation_bootstrap_svc(features=vectorized_connectivity_matrices,
                                                  class_labels_perm=class_labels_permutation_matrix,
                                                  bootstrap_array_perm=bootstrap_array_perm,
                                                  n_permutations=n_permutations,
                                                  n_cpus_bootstrap=n_physical,
                                                  verbose_bootstrap=1,
                                                  verbose_permutation=1)

    # Save the null distribution to avoid
    save_object(object_to_save=null_distribution, saving_directory=save_directory,
                filename='null_distribution_' + class_names[0] + '_' + class_names[1] + '.pkl')

    # Rebuild a symmetric array from normalized mean weight vector
    normalized_mean_weight_array = array_rebuilder(normalized_mean_weight,
                                                   'numeric', diagonal=np.zeros(n_nodes))
    # Find top features
    normalized_mean_weight_array_top_features, top_weights, top_coefficients_indices, top_weight_labels = \
        find_top_features(normalized_mean_weight_array=normalized_mean_weight_array,
                          labels_regions=labels_regions)

    if correction == 'max_t':
        null_distribution = load_object(full_path_to_object='/media/db242421/db242421_data/ConPagnon_data/'
                                                            'features_identification_results/'
                                                            'LG_ACM_patients_controls/null_distribution.pkl')
        # Corrected p values with the maximum statistic
        sorted_null_maximum_dist,  sorted_null_minimum_dist, p_value_positive_weights, p_value_negative_weights = \
            features_weights_max_t_correction(null_distribution_features_weights=null_distribution,
                                              normalized_mean_weight=normalized_mean_weight)

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

        significant_positive_features_indices, significant_negative_features_indices, \
        significant_positive_features_labels, significant_negative_features_labels = find_significant_features_indices(
            p_positive_features_significant=p_positive_features_significant,
            p_negative_features_significant=p_negative_features_significant,
            features_labels=labels_regions)

        # Take the mean difference and masking all connection except the surviving ones for
        # surviving negative and positive features weight
        mean_difference_positive_mask = np.multiply(first_class_mean_matrix - second_class_mean_matrix,
                                                    p_positive_features_significant)
        mean_difference_negative_mask = np.multiply(first_class_mean_matrix - second_class_mean_matrix,
                                                    p_negative_features_significant)

        with PdfPages(os.path.join(save_directory, report_filename)) as pdf:
            # Plot the estimated null distribution
            plt.figure(constrained_layout=True)
            plt.hist(sorted_null_maximum_dist, 'auto',  histtype='bar', alpha=0.5,
                     edgecolor='black')
            # The five 5% extreme values among maximum distribution
            p95 = np.percentile(sorted_null_maximum_dist, q=95)
            plt.axvline(x=p95, color='black')
            plt.title('Null distribution of maximum normalized weight mean')
            pdf.savefig()

            plt.figure()
            plt.hist(sorted_null_minimum_dist, 'auto',  histtype='bar', alpha=0.5, edgecolor='black')
            # The five 5% extreme values among minimum distribution
            p5 = np.percentile(sorted_null_minimum_dist, q=5)
            plt.axvline(x=p5, color='black')
            plt.title('Null distribution of minimum normalized weight mean')
            pdf.savefig()

            plt.figure()
            plot_connectome(adjacency_matrix=normalized_mean_weight_array,
                            node_coords=atlas_nodes, colorbar=True,
                            title='Features weight', node_size=node_size,
                            node_color=labels_colors)
            pdf.savefig()

            # plot the top weight in a histogram fashion
            fig = plt.figure(figsize=(15, 10), constrained_layout=True)

            weight_colors = ['blue' if weight < 0 else 'red' for weight in top_weights]
            plt.bar(np.arange(len(top_weights)), list(top_weights),
                    color=weight_colors,
                    edgecolor='black',
                    alpha=0.5)
            plt.xticks(np.arange(0,  len(top_weights)), top_weight_labels,
                       rotation=60,
                       ha='right')
            for label in range(len(plt.gca().get_xticklabels())):
                plt.gca().get_xticklabels()[label].set_color(weight_colors[label])
            plt.xlabel('Features names')
            plt.ylabel('Features weights')
            plt.title('Top {} features ranking of normalized mean weight'.format(top_features_number))
            pdf.savefig()

            plt.figure()
            plot_connectome(adjacency_matrix=normalized_mean_weight_array_top_features,
                            node_coords=atlas_nodes,
                            colorbar=True,
                            title='Top {} features weight'.format(top_features_number),
                            node_size=node_size,
                            node_color=labels_colors)
            pdf.savefig()

            # Plot on glass brain the significant positive features weight
            plt.figure()
            plot_connectome(adjacency_matrix=p_positive_features_significant,
                            node_coords=atlas_nodes, colorbar=True,
                            title='Significant positive weight', edge_cmap='Reds',
                            node_size=node_size,
                            node_color=labels_colors)
            pdf.savefig()

            # Plot on glass brain the mean difference in connectivity between
            # the two groups for surviving positive weight
            plt.figure()
            plot_connectome(adjacency_matrix=mean_difference_positive_mask,
                            node_coords=atlas_nodes, colorbar=True,
                            title='Difference in connectivity {} - {} (positive weights)'.format(
                                class_names[0],
                                class_names[1]),
                            node_size=node_size,
                            node_color=labels_colors)
            pdf.savefig()

            # Plot on glass brain the significant negative features weight
            plt.figure()
            plot_connectome(adjacency_matrix=p_negative_features_significant,
                            node_coords=atlas_nodes, colorbar=True,
                            title='Significant negative weight', edge_cmap='Blues',
                            node_size=node_size,
                            node_color=labels_colors)
            pdf.savefig()

            plt.figure()
            plot_connectome(adjacency_matrix=mean_difference_negative_mask,
                            node_coords=atlas_nodes, colorbar=True,
                            title='Difference in connectivity {} - {} (negative weights)'.format(
                                class_names[0],
                                class_names[1]),
                            node_size=node_size,
                            node_color=labels_colors)
            pdf.savefig()

            # Matrix view of significant positive and negative weight
            plt.figure()
            plot_matrix(matrix=p_negative_features_significant, labels_colors='auto', mpart='all',
                        colormap='Blues', linecolor='black', title='Significant negative weight',
                        vertical_labels=labels_regions, horizontal_labels=labels_regions)
            pdf.savefig()

            plt.figure()
            plot_matrix(matrix=p_positive_features_significant, labels_colors='auto', mpart='all',
                        colormap='Reds', linecolor='black', title='Significant positive weight',
                        vertical_labels=labels_regions, horizontal_labels=labels_regions)
            pdf.savefig()

            plt.close("all")

    else:
        null_distribution = load_object(full_path_to_object='/media/db242421/db242421_data/ConPagnon_data/'
                                                            'features_identification_results/'
                                                            'LG_ACM_patients_controls/null_distribution.pkl')
        # Perform another type of correction like FDR, ....
        p_values_corrected = features_weights_parametric_correction(
            null_distribution_features_weights=null_distribution,
            normalized_mean_weight=normalized_mean_weight,
            method=correction)

        p_values_corrected_array = array_rebuilder(vectorized_array=p_values_corrected,
                                                   array_type='numeric',
                                                   diagonal=np.ones(n_nodes))

        # Find p values under alpha threshold for negative and positive weight features
        p_negative_features_significant = np.array(
            (p_values_corrected_array < alpha) & (normalized_mean_weight_array < 0),
            dtype=int)
        p_positive_features_significant = np.array(
            (p_values_corrected_array < alpha) & (normalized_mean_weight_array > 0),
            dtype=int)

        significant_positive_features_indices, significant_negative_features_indices, \
        significant_positive_features_labels, significant_negative_features_labels = find_significant_features_indices(
            p_positive_features_significant=p_positive_features_significant,
            p_negative_features_significant=p_negative_features_significant,
            features_labels=labels_regions)

        # Take the mean difference and masking all connection except the surviving ones for
        # surviving negative and positive features weight
        mean_difference_positive_mask = np.multiply(first_class_mean_matrix - second_class_mean_matrix,
                                                    p_positive_features_significant)
        mean_difference_negative_mask = np.multiply(first_class_mean_matrix - second_class_mean_matrix,
                                                    p_negative_features_significant)

        # plot the top weight in a histogram fashion
        with PdfPages(os.path.join(save_directory, report_filename)) as pdf:

            plt.figure()
            plot_connectome(adjacency_matrix=normalized_mean_weight_array,
                            node_coords=atlas_nodes, colorbar=True,
                            title='Features weight',
                            node_size=node_size,
                            node_color=labels_colors)
            pdf.savefig()

            fig = plt.figure(figsize=(15, 10), constrained_layout=True)
            weight_colors = ['blue' if weight < 0 else 'red' for weight in top_weights]
            plt.bar(np.arange(len(top_weights)), list(top_weights),
                    color=weight_colors,
                    edgecolor='black',
                    alpha=0.5)
            plt.xticks(np.arange(0,  len(top_weights)), top_weight_labels,
                       rotation=60,
                       ha='right')
            for label in range(len(plt.gca().get_xticklabels())):
                plt.gca().get_xticklabels()[label].set_color(weight_colors[label])
            plt.xlabel('Features names')
            plt.ylabel('Features weights')
            plt.title('Top {} features ranking of normalized mean weight'.format(top_features_number))
            pdf.savefig()

            plt.figure()
            plot_connectome(adjacency_matrix=normalized_mean_weight_array_top_features,
                            node_coords=atlas_nodes, colorbar=True,
                            title='Top {} features weight'.format(top_features_number),
                            node_size=node_size,
                            node_color=labels_colors)
            pdf.savefig()
            # Plot on glass brain the significant positive features weight
            plt.figure()
            plot_connectome(adjacency_matrix=p_positive_features_significant,
                            node_coords=atlas_nodes, colorbar=True,
                            title='Significant positive weight after {} correction'.format(correction),
                            edge_cmap='Reds',
                            node_size=node_size,
                            node_color=labels_colors)

            pdf.savefig()

            # the two groups for surviving positive weight
            plt.figure()
            plot_connectome(adjacency_matrix=mean_difference_positive_mask,
                            node_coords=atlas_nodes, colorbar=True,
                            title='Difference in connectivity {} - {} (positive weights)'.format(
                                class_names[0],
                                class_names[1]),
                            node_size=node_size,
                            node_color=labels_colors)
            pdf.savefig()

            # Plot on glass brain the significant negative features weight
            plt.figure()
            plot_connectome(adjacency_matrix=p_negative_features_significant,
                            node_coords=atlas_nodes, colorbar=True,
                            title='Significant negative weight after {} correction'.format(correction),
                            edge_cmap='Blues',
                            node_size=node_size,
                            node_color=labels_colors)
            pdf.savefig()

            plt.figure()
            plot_connectome(adjacency_matrix=mean_difference_negative_mask,
                            node_coords=atlas_nodes, colorbar=True,
                            title='Mean difference in connectivity {} - {} (negative weights)'.format(
                                class_names[0],
                                class_names[1]),
                            node_size=node_size,
                            node_color=labels_colors)
            pdf.savefig()

            # Matrix view of significant positive and negative weight
            plt.figure()
            plot_matrix(matrix=p_negative_features_significant, labels_colors=labels_colors, mpart='all',
                        colormap='Blues', linecolor='black',
                        title='Significant negative weight after {} correction'.format(correction),
                        vertical_labels=labels_regions, horizontal_labels=labels_regions)
            pdf.savefig()

            plt.figure()
            plot_matrix(matrix=p_positive_features_significant, labels_colors=labels_colors, mpart='all',
                        colormap='Reds', linecolor='black',
                        title='Significant positive weight after {} correction'.format(correction),
                        vertical_labels=labels_regions, horizontal_labels=labels_regions)
            pdf.savefig()

            plt.close("all")

    # Write a small report in a text file
    with open(os.path.join(save_directory, text_report_filename), 'w') as output_results:
        # Write parameters
        output_results.write('------------ Parameters ------------')
        output_results.write('\n')
        output_results.write('Number of subjects: {}'.format(n_subjects))
        output_results.write('\n')
        output_results.write('Groups: {}'.format(class_names))
        output_results.write('\n')
        output_results.write('Groups labels: {}'.format(class_labels))
        output_results.write('\n')
        output_results.write('Bootstrap number: {}'.format(bootstrap_number))
        output_results.write('\n')
        output_results.write('Number of permutations: {}'.format(n_permutations))
        output_results.write('\n')
        output_results.write('Alpha threshold: {}'.format(alpha))
        output_results.write('\n')
        output_results.write('Connectivity metric: {}'.format(metric))
        output_results.write('\n')
        output_results.write('\n')
        output_results.write('\n')
        output_results.write('------------ Results ------------')
        output_results.write('\n')
        output_results.write('------------ Discriminative connections for negative features weight ------------')
        output_results.write('\n')
        # Write the labels of features with negative weight identified
        for negative_feature in range(len(significant_negative_features_labels)):

            output_results.write('\n')
            output_results.write('{} <-> {}, (indices: {} <-> {})'.format(
                significant_negative_features_labels[negative_feature][0],
                significant_negative_features_labels[negative_feature][1],
                significant_negative_features_indices[negative_feature][0],
                significant_negative_features_indices[negative_feature][1]))

        output_results.write('\n')
        output_results.write('\n')
        output_results.write('------------ Discriminative connections for positive features weight ------------')
        output_results.write('\n')
        # Write the labels of features with negative weight identified
        for positive_feature in range(len(significant_positive_features_labels)):
            output_results.write('\n')
            output_results.write('{} <-> {}, (indices: {} <-> {})'.format(
                significant_positive_features_labels[positive_feature][0],
                significant_positive_features_labels[positive_feature][1],
                significant_positive_features_indices[positive_feature][0],
                significant_positive_features_indices[positive_feature][1]))

    # write in text file some significant pairs, with language scores
#    regions_to_write = significant_positive_features_indices
#    patients_ids = list(subjects_connectivity_matrices['patients'].keys())
#    patients_matrices = np.array([subjects_connectivity_matrices['patients'][s][metric]
#                                  for s in patients_ids])
#    language_scores = read_excel_file('D:\\FunConnect\\regression_data.xlsx',
#                                    sheetname='cohort_functional_data')['language_score']

#    gender = read_excel_file('D:\\FunConnect\\regression_data.xlsx',
#                            sheetname='cohort_functional_data')['Sexe']

#    lesion_volume = read_excel_file('D:\\FunConnect\\regression_data.xlsx',
#                                    sheetname='cohort_functional_data')['lesion_normalized']

#    language_profil = read_excel_file('D:\\FunConnect\\regression_data.xlsx',
#                                      sheetname='cohort_functional_data')['langage_clinique']
#    header = ['subjects', 'connectivity', 'language_score', 'gender', 'lesion_volume', 'language_profil']
#    for region in regions_to_write:
#        region_csv = os.path.join(save_directory, 'pos_connection',
#                                  labels_regions[region[0]] + '_' + labels_regions[region[1]] + '.csv')
#        with open(os.path.join(region_csv), "w", newline='') as csv_file:
#            writer = csv.writer(csv_file, delimiter=',')
#            writer.writerow(header)
#            for line in range(len(patients_ids)):
#                writer.writerow([patients_ids[line], patients_matrices[line, region[0], region[1]],
#                                 language_scores.loc[patients_ids[line]], gender[patients_ids[line]],
#                                 lesion_volume[patients_ids[line]], language_profil[patients_ids[line]]])
