from utils.folders_and_files_management import load_object, save_object
import numpy as np
from nilearn.connectome import sym_matrix_to_vec
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn as sns

# Load the connectivity matrices dictionary
subjects_connectivity_matrices = load_object('/media/db242421/db242421_data/AICHA_test/'
                                             'aicha_connectivity_matrices.pkl')
# Class name, i.e the name of the groups to differentiate
class_names = ['controls', 'patients']
# Labels vectors: 1 for the first class, -1 for the second
class_labels = np.hstack((np.ones(len(subjects_connectivity_matrices[class_names[0]].keys())),
                          -1*np.ones(len(subjects_connectivity_matrices[class_names[1]].keys()))))

# Stratified Shuffle and Split cross validation:
cv = StratifiedShuffleSplit(n_splits=10000,
                            random_state=0,
                            test_size="default")
# Instance initialization of SVM classifier with a linear kernel
svc = LinearSVC()
# Compare the classification accuracy accross multiple metric
metrics = ['tangent', 'correlation', 'partial correlation']
mean_scores = []
# Final mean accuracy scores will be stored in a dictionary
mean_score_dict = {}
for metric in metrics:
    print('Evaluate classification performance on {}...'.format(metric))
    vectorized_connectivity_matrices = sym_matrix_to_vec(
        np.array([subjects_connectivity_matrices[class_name][s][metric] for class_name
                  in class_names for s in subjects_connectivity_matrices[class_name].keys()]),
        discard_diagonal=True)
    cv_scores = cross_val_score(estimator=svc, X=vectorized_connectivity_matrices,
                                y=class_labels, cv=cv,
                                scoring='accuracy')

    mean_scores.append(cv_scores.mean())
    mean_score_dict[metric] = cv_scores.mean()
    print('Done for {}'.format(metric))

save_object(object_to_save=mean_score_dict,
            saving_directory='/media/db242421/db242421_data/AICHA_test',
            filename='mean_accuracy_score.pkl')

# bar plot of classification results for the different metrics
plt.figure()
sns.barplot(x=list(mean_score_dict.keys()), y=list(mean_score_dict.values()))
plt.xlabel('Connectivity metrics')
plt.ylabel('Mean scores of classification')
plt.title('Mean scores of classification using different kind of connectivity')
plt.show()