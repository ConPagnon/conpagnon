"""
 Created by db242421 at 14/01/19

 """
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import numpy as np
from scipy.stats import pearsonr
import scipy.special as special


def _betai(a, b, x):
    x = np.asarray(x)
    x = np.where(x < 1.0, x, 1.0)  # if x > 1 then return 1.0
    return special.betainc(a, b, x)


def vcorrcoef(X, y):
    Xm = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
    ym = np.mean(y)
    r_num = np.sum((X - Xm) * (y - ym), axis=1)
    r_den = np.sqrt(np.sum((X - Xm) ** 2, axis=1) * np.sum((y - ym) ** 2))
    r = r_num / r_den
    df = len(y) - 2

    t_squared = r ** 2 * (df / ((1.0 - r) * (1.0 + r)))
    prob = _betai(0.5 * df, 0.5, df / (df + t_squared))

    return r, prob


def predict_scores(connectivity_matrices, raw_score, c_grid, scoring,
                   alpha=0.01,
                   n_jobs=10,
                   C=1,
                   optimize_regularization=False):

    # Scale the scores
    sc = StandardScaler()
    # Store the score prediction for each subject
    score_pred = []
    # Store the prediction weight
    score_pred_weights = []
    # Store the selected features indices
    score_selected_features = []
    scores = sc.fit_transform(np.array(raw_score).reshape(-1, 1))
    # initialize a leave one subject out cross validation scheme
    loo = LeaveOneOut()
    for train, test in loo.split(np.arange(0, connectivity_matrices.shape[0], 1)):
        # Split data in train set, and test set
        train_matrices = connectivity_matrices[train]
        train_scores = scores[train, :]

        test_matrices = connectivity_matrices[test]

        # Feature selection with a correlation between score and
        # connectivity for each brain connections
        r_mat, p_mat = vcorrcoef(X=train_matrices.T,
                                 y=np.squeeze(train_scores))

        # Search brain connection surviving the features selection step
        significant_features = np.where(p_mat < alpha)[0]

        # Select the selected brain connection in the train and test set
        test_selected_features = test_matrices[..., significant_features]
        train_selected_features = train_matrices[..., significant_features]

        if optimize_regularization is True:
            # Search for the best regularization parameters
            search_c_train_set = GridSearchCV(estimator=SVR(kernel='linear'),
                                              param_grid={'C': c_grid},
                                              cv=LeaveOneOut(),
                                              scoring=scoring,
                                              n_jobs=n_jobs,
                                              verbose=0)
            search_c_train_set.fit(X=train_selected_features, y=train_scores.ravel())

            best_c_parameter = search_c_train_set.best_params_['C']
        else:
            # Take the default in scikit-learn for C in SVR
            best_c_parameter = C

        # Initialize the estimator with the best regularization parameters
        svr = SVR(kernel='linear', C=best_c_parameter)
        # Fit the model on the whole training set
        svr.fit(X=train_selected_features, y=train_scores.ravel())
        # Predict the score of the left out subjects
        score_pred.append(svr.predict(X=test_selected_features))
        # Compute the weight of features of the estimator
        score_pred_weights.append(svr.coef_)
        # Append the indices of selected features
        score_selected_features.append(significant_features)
    # Compute R squared, as the squared correlation coefficient between predicted values, and
    # true scores values.
    r2 = pearsonr(np.array(score_pred), scores)[0] ** 2

    return r2, score_pred, score_pred_weights, score_selected_features
