from conpagnon.utils.folders_and_files_management import load_object, save_object
import numpy as np
from nilearn.connectome import sym_matrix_to_vec
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os
from conpagnon.plotting import display

save_figures_dir = '/media/db242421/db242421_data/Presentation/Royaumont'

# Load the connectivity matrices dictionary
subjects_connectivity_matrices = load_object(os.path.join(save_figures_dir,
                                                          'connectivity_matrices_patients_controls.pkl'))


# Plot all times series
for group in subjects_connectivity_matrices.keys():
    subject_list = list(subjects_connectivity_matrices[group].keys())
    for subject in subject_list:
        matrix = subjects_connectivity_matrices[group][subject]['correlation']
        plt.figure()
        display.plot_matrix(matrix=matrix, mpart='all')
        plt.savefig(os.path.join(save_figures_dir, group + 'subject_' + str(subject_list.index(subject))
                                 + '.png'))


# Class name, i.e the name of the groups to differentiate
class_names = ['patients', 'controls']
# Labels vectors: 1 for the first class, -1 for the second
class_labels = np.hstack((np.zeros(len(subjects_connectivity_matrices[class_names[0]].keys())),
                          np.ones(len(subjects_connectivity_matrices[class_names[1]].keys()))))

n_subjects = len(class_labels)


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
    features = sym_matrix_to_vec(np.array([subjects_connectivity_matrices[group][subject][metric]
                                           for group in class_names
                                           for subject in subjects_connectivity_matrices[group].keys()],
                                          ), discard_diagonal=True)
    print(features.shape)
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
with PdfPages(os.path.join(save_figures_dir, 'patients_controls.pdf')) as pdf:

    plt.figure()
    sns.barplot(x=list(mean_score_dict.keys()), y=list(mean_score_dict.values()))
    plt.xlabel('Connectivity metrics')
    plt.ylabel('Mean scores of classification')
    plt.title('Mean scores of classification using different kind of connectivity')
    pdf.savefig()
    plt.show()