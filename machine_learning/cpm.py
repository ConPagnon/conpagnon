from data_handling import atlas, data_management
from utils.folders_and_files_management import load_object
import numpy as np
from nilearn.connectome import sym_matrix_to_vec, vec_to_sym_matrix
from patsy import dmatrix
from data_handling import dictionary_operations
from machine_learning.CPM_method import predict_behavior
from scipy.io import savemat
from connectivity_statistics.parametric_tests import partial_corr
from scipy.stats import t

# Atlas set up
atlas_folder = '/media/db242421/db242421_data/ConPagnon_data/atlas/atlas_reference'
atlas_name = 'atlas4D_2.nii'
monAtlas = atlas.Atlas(path=atlas_folder,
                       name=atlas_name)
# Atlas path
atlas_path = monAtlas.fetch_atlas()
# Read labels regions files
labels_text_file = '/media/db242421/db242421_data/ConPagnon_data/atlas/atlas_reference/atlas4D_2_labels.csv'
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
# Fetch number of nodes in the parcellation
n_nodes = monAtlas.GetRegionNumbers()

# Load raw and Z-fisher transform matrix
subjects_connectivity_matrices = load_object(
    full_path_to_object='/media/db242421/db242421_data/ConPagnon_data/patient_controls/dictionary/'
                        'raw_subjects_connectivity_matrices.pkl')
Z_subjects_connectivity_matrices = load_object(
    full_path_to_object='/media/db242421/db242421_data/ConPagnon_data/patient_controls/dictionary/'
                        'z_fisher_transform_subjects_connectivity_matrices.pkl')
# Load behavioral data file
regression_data_file = data_management.read_excel_file(
    excel_file_path='/media/db242421/db242421_data/ConPagnon_data/regression_data/regression_data.xlsx',
    sheetname='cohort_functional_data')

# Type of subjects connectivity matrices
subjects_matrices = subjects_connectivity_matrices

# Select a subset of patients
# Compute the connectivity matrices dictionary with factor as keys.
group_by_factor_subjects_connectivity, population_df_by_factor, factor_keys, =\
    dictionary_operations.groupby_factor_connectivity_matrices(
        population_data_file='/media/db242421/db242421_data/ConPagnon_data/regression_data/regression_data.xlsx',
        sheetname='cohort_functional_data',
        subjects_connectivity_matrices_dictionnary=subjects_matrices,
        groupes=['patients'], factors=['Lesion'], drop_subjects_list=['sub40_np130304'])

subjects_matrices = {}
subjects_matrices['patients'] = group_by_factor_subjects_connectivity['G']

# Fetch patients matrices, and one behavioral score
kind = 'tangent'
patients_subjects_ids = list(subjects_matrices['patients'].keys())
# Patients matrices stack
patients_connectivity_matrices = np.array([subjects_matrices['patients'][s][kind] for
                                           s in patients_subjects_ids])

# Behavioral score
behavioral_scores = regression_data_file['language_score'].loc[patients_subjects_ids]
# Vectorized connectivity matrices of shape (n_samples, n_features)
vectorized_connectivity_matrices = sym_matrix_to_vec(patients_connectivity_matrices, discard_diagonal=True)

# Build confounding variable
confounding_variables = ['lesion_normalized', 'Sexe']
confounding_variables_data = regression_data_file[confounding_variables].loc[patients_subjects_ids]
# Encode the confounding variable in an array
confounding_variables_matrix = dmatrix(formula_like='+'.join(confounding_variables), data=confounding_variables_data,
                                       return_type='dataframe').drop(['Intercept'], axis=1)

add_predictive_variables = confounding_variables_matrix
significance_selection_threshold = 0.01

# Features selection by leave one out cross validation scheme
# Clean behavioral data
drop_subject_in_data = ['sub40_np130304']
try:
    regression_data_file = regression_data_file.drop(drop_subject_in_data)
except:
    pass

# Save the matrices for matlab utilisation
# Transpose the shape to (n_features, n_features, n_subjects)
# patients_connectivity_matrices_t = np.transpose(patients_connectivity_matrices, (1,2,0))
# Put ones on the diagonal
# for i in range(patients_connectivity_matrices_t.shape[2]):
    # np.fill_diagonal(patients_connectivity_matrices_t[..., i], 1)
# Save the matrix in .mat format
# patients_matrices_dict = {'patients_matrices': patients_connectivity_matrices_t}
# savemat('/media/db242421/db242421_data/CPM_matlab_ver/patients_LG_mat.mat', patients_matrices_dict)
# Save gender in .mat format
# gender_dict = {'gender': np.array(confounding_variables_data['Sexe'])}
# savemat('/media/db242421/db242421_data/CPM_matlab_ver/gender_LG_mat.mat', gender_dict)
# Save lesion normalized
# lesion_dict = {'lesion': np.array(confounding_variables_data['lesion_normalized'])}
# savemat('/media/db242421/db242421_data/CPM_matlab_ver/lesion_LG_mat.mat', lesion_dict)
# Save behavior
# behavior_dict = {'behavior': np.array(behavioral_scores)}
# savemat('/media/db242421/db242421_data/CPM_matlab_ver/behavior_LG_mat.mat', behavior_dict)

P, N = predict_behavior(vectorized_connectivity_matrices=vectorized_connectivity_matrices,
                        behavioral_scores=np.array(behavioral_scores),
                        selection_predictor_method='correlation',
                        significance_selection_threshold=significance_selection_threshold,
                        confounding_variables_matrix=None,
                        add_predictive_variables=add_predictive_variables,
                        verbose=1)



