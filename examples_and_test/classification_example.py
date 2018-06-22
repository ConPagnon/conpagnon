from utils.folders_and_files_management import load_object, save_object
import numpy as np
from nilearn.connectome import sym_matrix_to_vec
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, LeaveOneOut
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os

save_figures_dir = '/media/db242421/db242421_data/ConPagnon_data/features_identification_results/LesionFlip_controls'

# Load the connectivity matrices dictionary
subjects_connectivity_matrices = load_object(os.path.join(save_figures_dir,
                                                          'connectivity_matrices_LesionFlip_controls.pkl'))


# Class name, i.e the name of the groups to differentiate
class_names = ['LesionFlip', 'controls']
# Labels vectors: 1 for the first class, -1 for the second
class_labels = np.hstack((np.zeros(len(subjects_connectivity_matrices[class_names[0]].keys())),
                          np.ones(len(subjects_connectivity_matrices[class_names[1]].keys()))))

n_subjects = len(class_labels)

# I want to see if just homotopic connectivity make a good marker
homotopic_roi_indices = np.array([
    (1, 0), (2, 3), (4, 5), (6, 7), (8, 11), (9, 10), (13, 12), (14, 15), (16, 17), (18, 19), (20, 25),
    (21, 26), (22, 29), (23, 28), (24, 27), (30, 31), (32, 33), (35, 34), (36, 37), (38, 39), (44, 40),
    (41, 45), (42, 43), (46, 49), (47, 48), (50, 53), (53, 54), (54, 57), (55, 56), (58, 61), (59, 60),
    (62, 63), (64, 65), (66, 67), (68, 69), (70, 71)])

# Stratified Shuffle and Split cross validation:
cv = StratifiedShuffleSplit(n_splits=10000,
                            random_state=0,
                            test_size="default")

# cv = LeaveOneOut()
# Instance initialization of SVM classifier with a linear kernel
svc = LinearSVC()
# Compare the classification accuracy accross multiple metric
metrics = ['tangent', 'correlation', 'partial correlation']
mean_scores = []
# Final mean accuracy scores will be stored in a dictionary
mean_score_dict = {}
for metric in metrics:
    print('Evaluate classification performance on {}...'.format(metric))
    features = np.array([subjects_connectivity_matrices[group][subject][metric][
                             homotopic_roi_indices[:, 0],
                             homotopic_roi_indices[:, 1]] for group in class_names
                         for subject in subjects_connectivity_matrices[group].keys()])
    cv_scores = cross_val_score(estimator=svc, X=features,
                                y=class_labels, cv=cv,
                                scoring='accuracy')

    mean_scores.append(cv_scores.mean())
    mean_score_dict[metric] = cv_scores.mean()
    print('Done for {}'.format(metric))


save_object(object_to_save=mean_score_dict,
            saving_directory=save_figures_dir,
            filename=class_names[0] + '_' + class_names[1] + '.pkl')

# bar plot of classification results for the different metrics
with PdfPages(os.path.join(save_figures_dir, 'lesion_flip_controls_homotopic.pdf')) as pdf:

    plt.figure()
    sns.barplot(x=list(mean_score_dict.keys()), y=list(mean_score_dict.values()))
    plt.xlabel('Connectivity metrics')
    plt.ylabel('Mean scores of classification')
    plt.title('Mean scores of classification using different kind of connectivity')
    pdf.savefig()
    plt.show()