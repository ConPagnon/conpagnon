import importlib
from data_handling import atlas, data_architecture, dictionary_operations
from utils import folders_and_files_management
from computing import compute_connectivity_matrices as ccm
from machine_learning import classification
from connectivity_statistics import parametric_tests
from plotting import display
from data_handling import data_management
import os
import numpy as np
import pandas as pd
from scipy.stats import t
from plotting.display import plot_matrix
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from nilearn.connectome import sym_matrix_to_vec, vec_to_sym_matrix
import webcolors
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

# Build a giant dataset: all connectitivity type in columns, and subjects id in lines

root_data_directory = '/media/db242421/db242421_data/ConPagnon_data/patients_behavior/language_score/tangent'


models = ['contra_intra', 'ipsi_intra', 'intra_homotopic']
networks = ['DMN', 'Executive', 'Language',  'MTL',
           'Salience', 'Sensorimotor', 'Visuospatial',
           'Primary_Visual', 'Secondary_Visual']
data_to_merge = []

for network in networks:
    for model in models:
        # Read the model dataframe
        raw_model_data = pd.read_csv(os.path.join(root_data_directory, network, 'patients_' + model + '_' + network +
                                              '_connectivity.csv'))

        # Shift index to subjects nip
        model_data = data_management.shift_index_column(panda_dataframe=raw_model_data,
                                                        columns_to_index=['subjects'])
        # rename the columns
        model_data.columns = [model + '_' + network + '_connectivity']
        # Stack the dataframe
        data_to_merge.append(model_data)t
# build the giant dataframe
all_models_data = data_management.merge_list_dataframes(data_to_merge)
all_models_data_T = all_models_data.T
all_models_data_T.to_csv('/media/db242421/db242421_data/ConPagnon_data/test/all_connectivity_T.csv')
all_models_data.to_csv('/media/db242421/db242421_data/ConPagnon_data/test/all_connectivity.csv')

from scipy.stats import pearsonr
# Some statistic test
alpha = 0.05

R_mat = all_models_data.corr()
labels = list(R_mat.index)
df = 25
# Convert in T mat
T_mat = (np.sqrt(df) * np.abs(R_mat)) / (np.sqrt(np.ones(R_mat.shape[0]) - R_mat ** 2))
# compute p values
P_mat = t.sf(np.abs(T_mat), df) * 2

# Vectorize the P matrix

P_mat_vec = sym_matrix_to_vec(P_mat, discard_diagonal=True)


P_mat_vec_corrected = multipletests(pvals=P_mat_vec, method='fdr_bh', alpha=alpha)[1]

P_mat_corrected = vec_to_sym_matrix(P_mat_vec_corrected, diagonal=np.ones(R_mat.shape[0])*(1/np.sqrt(2)))
np.savetxt(fname='/media/db242421/db242421_data/ConPagnon_data/test/pcorrected.csv', X=P_mat_corrected,
           delimiter=',', header="")


labels_color = ['goldenrod', 'indianred', 'sienna', 'lightpink', 'turquoise', 'black', 'darkslategray',
                'orchid', 'limegreen']
labels_color = [webcolors.name_to_rgb(i) for i in labels_color]

# Re-index the matrices, sorted the rows and columns by connectivity type
sorted_labels = sorted(labels)
#new_sorted_labels = sorted_labels[9:18] + sorted_labels[18:27] + sorted_labels[27:36] + sorted_labels[0:9]
new_sorted_labels = sorted_labels[9:18] + sorted_labels[18:27] + sorted_labels[0:9]

sorted_labels_colors = (1/255)*np.array(labels_color + labels_color + labels_color + labels_color)
new_index_order = np.array([labels.index(i) for i in new_sorted_labels])

# Re-index correlation matrix
R_mat = np.array(R_mat)
R_mat_reindex = R_mat[new_index_order, :]
R_mat_reindex = R_mat_reindex[:, new_index_order]

# Re-index p corrected matrices
P_mat_corrected_reindex = P_mat_corrected[new_index_order, :]
P_mat_corrected_reindex = P_mat_corrected_reindex[:, new_index_order]

with PdfPages('/media/db242421/db242421_data/ConPagnon_data/test/P_mat_corrected_fdr_bh_wo_intra_' +
              str(alpha) + '.pdf') as pdf:

    plot_matrix(matrix=R_mat_reindex, mpart='all',
                vmin=-1, vmax=1, horizontal_labels=new_sorted_labels,
                vertical_labels=new_sorted_labels, linecolor='black',
                title='Correlation between connectivity types in patients',
                labels_colors=sorted_labels_colors)
    pdf.savefig()
    plt.show()

    plot_matrix(matrix=P_mat_corrected_reindex, mpart='lower',
                colormap='hot', vmin=0, vmax=alpha, horizontal_labels=new_sorted_labels,
                vertical_labels=new_sorted_labels, linecolor='black',
                title='Significant P values, alpha = {}'.format(alpha),
                labels_colors=sorted_labels_colors)
    pdf.savefig()
    plt.show()

    # Corresponding correlation matrix
    mask = P_mat_corrected_reindex < alpha
    R_mat_masked = np.multiply(mask, R_mat_reindex)

    plot_matrix(matrix=R_mat_masked, mpart='lower',
                vmin=-1, vmax=1, horizontal_labels=new_sorted_labels,
                vertical_labels=new_sorted_labels, linecolor='black',
                title='Significant correlation',
                labels_colors=sorted_labels_colors)
    pdf.savefig()
    plt.show()

