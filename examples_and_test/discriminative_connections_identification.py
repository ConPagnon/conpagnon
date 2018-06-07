from utils.folders_and_files_management import load_object
import os
import numpy as np
from nilearn.connectome import sym_matrix_to_vec
from machine_learning.features_indentification import bootstrap_svc, \
     rank_top_features_weight, remove_reversed_duplicates, \
     permutation_bootstrap_svc, features_weights_max_t_correction, \
     features_weights_parametric_correction
import psutil
import time
import matplotlib.pyplot as plt
from utils.array_operation import array_rebuilder
from nilearn.plotting import plot_connectome
from data_handling import atlas
from plotting.display import plot_matrix
from utils.folders_and_files_management import save_object
from matplotlib.backends.backend_pdf import PdfPages


# Atlas set up
atlas_folder = '/neurospin/grip/protocols/MRI/AVCnn_Dhaif_2018/atlas_reference'
atlas_name = 'atlas4D_2.nii'
monAtlas = atlas.Atlas(path=atlas_folder,
                       name=atlas_name)
# Atlas path
atlas_path = monAtlas.fetch_atlas()
# Read labels regions files
labels_text_file = '/neurospin/grip/protocols/MRI/AVCnn_Dhaif_2018/atlas_reference/atlas4D_2_labels.csv'
labels_regions = monAtlas.GetLabels(labels_text_file)
# User defined colors for labels ROIs regions
colors = ['navy', 'sienna', 'orange', 'orchid', 'indianred', 'olive',
          'goldenrod', 'turquoise', 'darkslategray', 'limegreen', 'black',
          'lightpink']
# Number of regions in each user defined networks
networks = [2, 10, 2, 6, 10, 2, 8, 6, 8, 8, 6, 4]
# Transformation of string colors list to an RGB color array,
# all colors ranging between 0 and 1.
labels_colors = (1./255)*monAtlas.UserLabelsColors(networks=networks,
                                                   colors=colors)
# Fetch nodes coordinates
atlas_nodes = monAtlas.GetCenterOfMass()
# Fetch number of nodes in the atlas
n_nodes = monAtlas.GetRegionNumbers()

# Load connectivity matrices
data_folder = '/media/db242421/db242421_data/ConPagnon_data/patient_controls/dictionary'
connectivity_dictionary_name = 'raw_subjects_connectivity_matrices.pkl'
subjects_connectivity_matrices = load_object(os.path.join(data_folder, connectivity_dictionary_name))

# Optional: drop some subjects of the subjects matrices dictionary
subjects_to_drop = ['sub18_mg110111', 'sub21_yg120001', 'sub23_lf120459', 'sub26_as110192',
                    'sub41_sa130332', 'sub20_hd120032', 'sub24_ed110159', 'sub25_ec110149',
                    'sub38_mv130274', 'sub37_la130266', 'sub02_rf110332', 'sub03_mc120272',
                    'sub01_rm110247']

for subject in subjects_to_drop:
    subjects_connectivity_matrices['patients'].pop(subject, None)

class_names = ['controls', 'patients']
metric = 'tangent'
vectorized_connectivity_matrices = sym_matrix_to_vec(
    np.array([subjects_connectivity_matrices[class_name][s][metric] for class_name
              in class_names for s in subjects_connectivity_matrices[class_name].keys()]),
    discard_diagonal=True)
# Labels vectors
class_labels = np.hstack((-1*np.ones(len(subjects_connectivity_matrices[class_names[0]].keys())),
                          1*np.ones(len(subjects_connectivity_matrices[class_names[1]].keys()))))

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
bootstrap_matrix = np.random.choice(a=indices, size=(bootstrap_number, n_subjects), replace=True)

# Choose multiple comparison correction
correction = 'fdr_bh'

# Compute number of physical and logical core
n_physical = psutil.cpu_count(logical=False)
n_cpu_with_logical = psutil.cpu_count(logical=True)

# Generate a permuted class labels array
class_labels_permutation_matrix = np.array([np.random.permutation(class_labels) for n in range(n_permutations)])

save_directory = '/media/db242421/db242421_data/ConPagnon_data/features_identification_results'
report_filename = 'features_identification_controls_LG_ACM_patients_labels_FDR.pdf'
text_report_filename = 'features_identification_controls_LG_ACM_patients_FDR.txt'
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
    # Estimation of null distribution of normalized mean weight
    null_distribution = permutation_bootstrap_svc(features=vectorized_connectivity_matrices,
                                                  class_labels_perm=class_labels_permutation_matrix,
                                                  n_permutations=n_permutations,
                                                  n_cpus_bootstrap=n_physical,
                                                  verbose_bootstrap=1,
                                                  verbose_permutation=1)

    # Save the null distribution to avoid
    save_object(object_to_save=null_distribution, saving_directory=save_directory,
                filename='null_distribution.pkl')

    if correction == 'max_t':

        # Corrected p values with the maximum statistic
        sorted_null_maximum_dist,  sorted_null_minimum_dist, p_values_max, p_values_min = \
            features_weights_max_t_correction(null_distribution_features_weights=null_distribution,
                                              normalized_mean_weight=normalized_mean_weight)

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

            # Rebuild vectorized p values array
            p_max_values_array = array_rebuilder(vectorized_array=p_values_max,
                                                 array_type='numeric',
                                                 diagonal=np.ones(72))

            p_min_values_array = array_rebuilder(vectorized_array=p_values_min,
                                                 array_type='numeric',
                                                 diagonal=np.ones(72))

            # Find p-values under the alpha threshold
            p_min_significant = np.array(p_min_values_array < alpha, dtype=int)
            p_max_significant = np.array(p_max_values_array < alpha, dtype=int)

            # Plot on glass brain the normalized weight mean
            normalized_mean_weight_array = array_rebuilder(normalized_mean_weight,
                                                           'numeric', diagonal=np.zeros(n_nodes))

            plt.figure()
            plot_connectome(adjacency_matrix=normalized_mean_weight_array,
                            node_coords=atlas_nodes, node_color=labels_colors, colorbar=True,
                            title='Features weight')
            pdf.savefig()

            # Find the top features among the normalized mean weight distribution
            top_features_number = 50
            top_weights, top_coefficients_indices, top_weight_labels = rank_top_features_weight(
                coefficients_array=normalized_mean_weight_array,
                top_features_number=top_features_number,
                features_labels=labels_regions)

            # plot the top weight in a histogram fashion
            fig = plt.figure(figsize=(15, 10), constrained_layout=True)

            weight_colors = ['blue' if weight < 0 else 'red' for weight in top_weights]
            plt.bar(np.arange(len(top_weights)), list(top_weights), color=weight_colors,
                    edgecolor='black',
                    alpha=0.5)
            plt.xticks(np.arange(0,  len(top_weights)), top_weight_labels, rotation=60,
                       ha='right')
            for label in range(len(plt.gca().get_xticklabels())):
                plt.gca().get_xticklabels()[label].set_color(weight_colors[label])
            plt.xlabel('Features names')
            plt.ylabel('Features weights')
            plt.title('Top {} features ranking of normalized mean weight'.format(top_features_number))
            pdf.savefig()

            # Plot the top features weight on glass brain
            top_weights_mask = np.zeros((n_nodes, n_nodes), dtype=bool)
            top_weights_mask[top_coefficients_indices[:, 0], top_coefficients_indices[:, 1]] = True
            normalized_mean_weight_array_top_features = np.multiply(normalized_mean_weight_array, top_weights_mask)
            normalized_mean_weight_array_top_features += normalized_mean_weight_array_top_features.T

            plt.figure()
            plot_connectome(adjacency_matrix=normalized_mean_weight_array_top_features,
                            node_coords=atlas_nodes, node_color=labels_colors, colorbar=True,
                            title='Top {} features weight'.format(top_features_number))
            pdf.savefig()

            # Plot on glass brain the significant positive features weight
            plt.figure()
            plot_connectome(adjacency_matrix=p_max_significant,
                            node_coords=atlas_nodes, node_color=labels_colors, colorbar=True,
                            title='Significant positive weight', edge_cmap='Reds')
            pdf.savefig()

            # Plot on glass brain the significant negative features weight
            plt.figure()
            plot_connectome(adjacency_matrix=p_min_significant,
                            node_coords=atlas_nodes, node_color=labels_colors, colorbar=True,
                            title='Significant negative weight', edge_cmap='Blues')
            pdf.savefig()

            # Matrix view of significant positive and negative weight
            plt.figure()
            plot_matrix(matrix=p_min_significant, labels_colors=labels_colors, mpart='lower',
                        colormap='Blues', linecolor='black', title='Significant negative weight',
                        vertical_labels=labels_regions, horizontal_labels=labels_regions)
            pdf.savefig()

            plt.figure()
            plot_matrix(matrix=p_max_significant, labels_colors=labels_colors, mpart='lower',
                        colormap='Reds', linecolor='black', title='Significant positive weight',
                        vertical_labels=labels_regions, horizontal_labels=labels_regions)
            pdf.savefig()

            plt.close("all")

            # Print the significant features, that survive permutation testing
            significant_positive_features_indices = \
                np.array(list(remove_reversed_duplicates(np.array(np.where(p_max_significant != 0)).T)))
            significant_positive_features_labels = \
                np.array([labels_regions[significant_positive_features_indices[i]] for i
                          in range(significant_positive_features_indices.shape[0])])

            significant_negative_features_indices = \
                np.array(list(remove_reversed_duplicates(np.array(np.where(p_min_significant != 0)).T)))

            significant_negative_features_labels = \
                np.array([labels_regions[significant_negative_features_indices[i]] for i
                          in range(significant_negative_features_indices.shape[0])])
    else:
        # Perform another type of correction
        p_values_corrected = features_weights_parametric_correction(
            null_distribution_features_weights=null_distribution,
            normalized_mean_weight=normalized_mean_weight,
            method=correction)

        p_values_corrected_array = array_rebuilder(vectorized_array=p_values_corrected,
                                                   array_type='numeric',
                                                   diagonal=np.ones(n_nodes))

        # Plot on glass brain the normalized weight mean
        normalized_mean_weight_array = array_rebuilder(normalized_mean_weight,
                                                       'numeric', diagonal=np.zeros(n_nodes))

        # Find p values under alpha threshold for negative and positive weight features
        p_min_significant = np.array((p_values_corrected_array < alpha) & (normalized_mean_weight_array < 0),
                                     dtype=int)
        p_max_significant = np.array((p_values_corrected_array < alpha) & (normalized_mean_weight_array > 0),
                                     dtype=int)

        with PdfPages(os.path.join(save_directory, report_filename)) as pdf:

            plt.figure()
            plot_connectome(adjacency_matrix=normalized_mean_weight_array,
                            node_coords=atlas_nodes, node_color=labels_colors, colorbar=True,
                            title='Features weight')
            pdf.savefig()

            # Find the top features among the normalized mean weight distribution
            top_features_number = 50
            top_weights, top_coefficients_indices, top_weight_labels = rank_top_features_weight(
                coefficients_array=normalized_mean_weight_array,
                top_features_number=top_features_number,
                features_labels=labels_regions)

            # plot the top weight in a histogram fashion
            fig = plt.figure(figsize=(15, 10), constrained_layout=True)

            weight_colors = ['blue' if weight < 0 else 'red' for weight in top_weights]
            plt.bar(np.arange(len(top_weights)), list(top_weights), color=weight_colors,
                    edgecolor='black',
                    alpha=0.5)
            plt.xticks(np.arange(0,  len(top_weights)), top_weight_labels, rotation=60,
                       ha='right')
            for label in range(len(plt.gca().get_xticklabels())):
                plt.gca().get_xticklabels()[label].set_color(weight_colors[label])
            plt.xlabel('Features names')
            plt.ylabel('Features weights')
            plt.title('Top {} features ranking of normalized mean weight'.format(top_features_number))
            pdf.savefig()

            # Plot the top features weight on glass brain
            top_weights_mask = np.zeros((n_nodes, n_nodes), dtype=bool)
            top_weights_mask[top_coefficients_indices[:, 0], top_coefficients_indices[:, 1]] = True
            normalized_mean_weight_array_top_features = np.multiply(normalized_mean_weight_array, top_weights_mask)
            normalized_mean_weight_array_top_features += normalized_mean_weight_array_top_features.T

            plt.figure()
            plot_connectome(adjacency_matrix=normalized_mean_weight_array_top_features,
                            node_coords=atlas_nodes, node_color=labels_colors, colorbar=True,
                            title='Top {} features weight'.format(top_features_number))
            pdf.savefig()
            # Plot on glass brain the significant positive features weight
            plt.figure()
            plot_connectome(adjacency_matrix=p_max_significant,
                            node_coords=atlas_nodes, node_color=labels_colors, colorbar=True,
                            title='Significant positive weight after {} correction'.format(correction),
                            edge_cmap='Reds')
            pdf.savefig()

            # Plot on glass brain the significant negative features weight
            plt.figure()
            plot_connectome(adjacency_matrix=p_min_significant,
                            node_coords=atlas_nodes, node_color=labels_colors, colorbar=True,
                            title='Significant negative weight after {} correction'.format(correction),
                            edge_cmap='Blues')
            pdf.savefig()

            # Matrix view of significant positive and negative weight
            plt.figure()
            plot_matrix(matrix=p_min_significant, labels_colors=labels_colors, mpart='lower',
                        colormap='Blues', linecolor='black',
                        title='Significant negative weight after {} correction'.format(correction),
                        vertical_labels=labels_regions, horizontal_labels=labels_regions)
            pdf.savefig()

            plt.figure()
            plot_matrix(matrix=p_max_significant, labels_colors=labels_colors, mpart='lower',
                        colormap='Reds', linecolor='black',
                        title='Significant positive weight after {} correction'.format(correction),
                        vertical_labels=labels_regions, horizontal_labels=labels_regions)
            pdf.savefig()

            plt.close("all")

            # Print the significant features, that survive permutation testing
            significant_positive_features_indices = \
                np.array(list(remove_reversed_duplicates(np.array(np.where(p_max_significant != 0)).T)))
            significant_positive_features_labels = \
                np.array([labels_regions[significant_positive_features_indices[i]] for i
                          in range(significant_positive_features_indices.shape[0])])

            significant_negative_features_indices = \
                np.array(list(remove_reversed_duplicates(np.array(np.where(p_min_significant != 0)).T)))

            significant_negative_features_labels = \
                np.array([labels_regions[significant_negative_features_indices[i]] for i
                          in range(significant_negative_features_indices.shape[0])])

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
