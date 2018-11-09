# -*- coding: utf-8 -*-
"""Display functions

Author: Dhaif BEKHA
"""

from nilearn.plotting import plot_connectome
import matplotlib.pyplot as plt
#import matplotlib
import seaborn as sns
import numpy as np
from matplotlib.backends import backend_pdf
from scipy import stats


def plot_ttest_results(t_test_dictionnary, groupes, contrast, node_coords, node_color,
                       output_pdf, node_size=50,
                       colormap='bwr', colorbar=False,
                       annotate=True,
                       display_mode='ortho'):
    """Plot on a glass brain the significant mean effect resulting of a
    two samples t-test and save resulting figures in a PDF files.
    
    Parameters
    ----------
    t_test_dictionnary : dict
        A dictionnary containing multiple keys :
            - 'tstatistic' : The raw statistic t-map for the choosen contrast
            2D numpy array of shape (number of regions, number of regions).
            
            - 'uncorrected pvalues' : The raw pvalues, 2D numpy array of shape
            (number of regions, number of regions).
            
            - 'corrected pvalues' : The corrected pvalues with the chosen methods.
            2D numpy array of shape (number of regions, number of regions).
            
            - 'significant edges' : The significant t-values after masking
            for the non significant pvalues at alpha level. 2D numpy array of shape 
            (number of regions, number of regions).
            
            - 'significant pvalues' : The significant pvalues at level alpha.
            2D numpy array of shape (number of regions, number of regions)
            
            - 'significant mean effect' : The differences of mean connectivity
            between the two groups according to the chosen contrast, and non-significant
            connexion are mask at alpha level. 2D numpy array of shape 
            (number of regions, number of regions).
    groupes : list
        The list of groupes involved in the computed
        contrast.
    contrast : list
        The contrast vector used in the two samples t-test.
    colormap: str, optional
        The colormap to use for the matrix heatmap.
        Default is 'bwr'.
    node_coords : list
        The list of centroids coordinates of each regions
        in the atlas used for regions signals extraction.
    node_color : numpy.ndarray shape(number of regions, 3)
        The color in the RGB normalized space, that is each
        color are represented by triplet of float ranging between
        0 and 1.
    output_pdf : str
        The full path including .pdf extension to the a pdf file
        for saving the glass brain projection plot.
    node_size : int, optional
        The node size. Default is 50.
    colorbar : bool, optional
        If True, the colorbar of the chosen colormap
        is displayed alongside the glass brain plot.
        Default is False.
    annotate : bool, optional
        If True, letters at the top of the glass brain
        indicates the left or right hemisphere location.
        Default is True.
    display_mode : str, optional
        The slice configuration you want to display
        Possible values are: 'ortho','x', 'y', 'z', 'xz' 
        'yx', 'yz', 'l', 'r', 'lr', 'lzr', 'lyr','lzry', 'lyrz'.
        Default if 'ortho'.
        
    See Also 
    --------
    parametric_tests.two_sample_t_test : 
        This function return a dictionnary resulting
        of the comparison of two groupe using a t-test. This
        the input of this display function.
        
    nilearn.plotting.plot_connectome :
        This function in the Nilearn packages is used here
        for the plot of the glass brain. I encourage
        the user to read the corresponding docstring in
        the Nilearn git repo.
        
    Notes
    -----
    The variables `groupes` and `contrast` are needed only
    the title. You shoul have already used this variables when
    you compute the two sample t-test between you're two groups.
    
    
    
    """

    # Modify the dictionnary of parameters for matplotlib
    import matplotlib.pylab as pylab
    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (10, 4.5),
              'axes.labelsize': 'x-large',
              'axes.titlesize':'x-large',
              'xtick.labelsize':'x-large',
              'ytick.labelsize':'x-large'
              }
    pylab.rcParams.update(params)
    
    # For all kinds present in the t_test_dictionnary keys:
    with backend_pdf.PdfPages(output_pdf) as pdf:
        for kind in t_test_dictionnary.keys():
            # Write title for the current kind according contrast
            if contrast == [-1.0, 1.0]:
                title = 't-test for {} and contrast {} - {}'.format(kind, groupes[1], groupes[0])
            elif contrast == [1.0, -1.0]:
                title = 't-test for {} contrast {} - {}'.format(kind, groupes[0], groupes[1])
            else:
                raise ValueError('Unrecognized contrast vector !')

            # plot the connectome on a glass brain using Nilearn plot_connectome function
            fig = plt.figure()
            plot_connectome(adjacency_matrix=t_test_dictionnary[kind]['significant mean effect'],
                            node_coords=node_coords,
                            node_color=node_color, node_size=node_size,
                            edge_cmap=colormap, colorbar=colorbar,
                            annotate=annotate,
                            display_mode=display_mode, figure=fig, title=title)
            pdf.savefig()


def plot_matrix(matrix, labels_colors='auto', mpart='lower', k=0, colormap='RdBu_r',
                colorbar_params={'shrink': .5}, center=0,
                vmin=None, vmax=None, labels_size=8, horizontal_labels='auto',
                vertical_labels='auto',
                linewidths=.5, linecolor='white', title='Untitled',
                figure_size=(12, 9)):
    """Plot a entire matrix, or it's lower part with a chosen heatmap.
    
    Parameters
    ----------
    matrix : numpy.array shape(numbers of regions, numbers of regions)
        The numerical array you want to plot.
    labels_colors : str, or numpy.ndarray shape(number of regions, 3)
        The color in the RGB normalized space, that is each
        color are represented by triplet of float ranging between
        0 and 1. Default is 'auto', random colors are automatically
        generated.
    mpart : str
        The part of the array you want to plot, that is 
        the entire matrix or just it's lower part.
    k : int
        The position from the main diagonal you want 
        to cut. Default is 0.
    colormap : str
        The seaborn colormap you want to apply on the matrix
        when displaying it. Default is 'RdBu_r'.
    colorbar_params : dict, optional
        Additional parameters concerning the parameters
        of the colorbar : position, size...
        Keyword arguments for `fig.colorbar`.
    vmin, vmax : float, optional
        The extrema of the colormap. all display values
        are threshold according too those values. If None,
        these values are inferred from the matrix.
        Default is 'None'.
    labels_size : int, optional
        The font size of the labels.
        Default is 8.
    horizontal_labels : str or list; optional
        The labels list of each rows. If 'auto', labels
        are replaced by the row index.
    vertical_labels : str or list, optional
        The labels list of each columns. If 'auto', labels
        are replaced by the column index.
    linewidths : float, optional
        The width that divide each cell. Default
        is 0.5.
    linecolor : str, optional
        The color of the lines that divide each cell. Default
        is 'white'.
    title : str, optional
        The figure title. Default is 'untitled'.
    figure_size : tuple, optional
        Figure size in a tuple format (height, width).
        Default is (12, 9).
    center: float, optional
        The center of the colormap, for divergent data.
        
    See Also
    --------
    sns.heatmap :
        See the docstring of this function for a deeper
        insight of plotting function of a numerical array using
        seaborn.
        
    """
    
    # Compute matrices mask according to the chosen part of the matrix to plot.
    if mpart == 'lower':
        # Compute a boolean mask for the lower part of the array.
        matrice_mask = np.invert(np.ma.make_mask(np.tril(matrix, k=k)))
    elif mpart == 'all':
        # Compute a boolean  mask for all the array
        matrice_mask = np.invert(np.ma.make_mask(matrix))
    else:
        # Raise a value error, if the option are not among the authorized ones.
        raise ValueError('Unrecognized option for matrices mask.')
    # Generate random labels colors if labels_colors is set to auto:
    if labels_colors is 'auto':
        # Get numbers of labels
        n_labels = matrix.shape[0]
        labels_colors = np.random.rand(n_labels, 3)

    # plot matrices with the choosing cmap
    f, ax = plt.subplots(figsize=figure_size)
    for tick in ax.get_xticklabels():
        # Set label rotation to 90° for the x axis.
        tick.set_rotation(90)
    sns.heatmap(data=matrix, square=True, cmap=colormap, mask=matrice_mask, cbar_kws=colorbar_params,
                center=center,
                vmin=vmin, vmax=vmax, xticklabels=horizontal_labels,
                yticklabels=vertical_labels,
                linewidths=linewidths,
                linecolor=linecolor)
    # Set rotation angle to zero for the y axis.
    plt.yticks(rotation=0)
    for xtick, color in zip(ax.get_xticklabels(), labels_colors):
        xtick.set_color(color)
        xtick.set_fontsize(labels_size)
    for ytick, color in zip(ax.get_yticklabels(), labels_colors):
        ytick.set_color(color)
        ytick.set_fontsize(labels_size)
    plt.title(title)
    f.tight_layout()


def display_gaussian_connectivity_fit(vectorized_connectivity, estimate_mean, estimate_std,
                                      raw_data_colors='blue', line_width=2, alpha=0.5, normed=True,
                                      bins='auto', fitted_distribution_color='black', title=None,
                                      xtitle=None, ytitle=None, legend_fitted='Fitted Gaussian Distribution',
                                      legend_data=None, display_fit='yes', ms=5):
    """Display a vectorized connectivity matrices histogram, along with a Gaussian
    fit with an estimated mean, and standard deviation.

    Parameters
    ----------
    vectorized_connectivity: numpy.array, shape 0.5*n_features*(n_features + 1)
        A array of connectivity coefficient.
    estimate_mean: float
        The estimated mean of the connectivity coefficient distribution.
    estimate_std: float
        The estimated standard deviation of the connectivity coefficient distribution.
    raw_data_colors: string, optional
        The colors of the histogram for the connectivity coefficient distribution.
        Default is blue.
    line_width: int, optional
        The width of line for the fit of the data. Default is 2.
    alpha: float, optional
        The opacity coefficient for histogram, between 0 and 1. Default is 0.5.
    normed: bool, optional
        Normalized the histogram. This is mandatory for displaying
        the gaussian fit over the data, since it represent a probability density function.
    bins: string, optional
        How the bins edges of the histogram are computed, choices among different estimators
        {'fd', 'doane', 'scott', 'rice', 'sturges'}
        Default is 'auto', the maximum of the sturges and fd estimators are taken.
    fitted_distribution_color: string, optional
        The color of the curves representing the Gaussian fit. Default is black.
    title: string, optional
        The overall title of the figure. Default is None.
    xtitle: string, optional
        The legend of the x-axis. Default is None.
    ytitle: string, optional
        The legend of the y-axis. Default is None.
    legend_fitted: string, optional
        The legend for the Gaussian fit. Default is 'Fitted Gaussian Distribution'.
    legend_data: string, optional
        The legend for the histogram of the data. Default is None.
    display_fit: bool, optional
        If True, the gaussian fit of the data is displayed over the normalized histogram.
    ms: float, optional
        The size of the dot on the graph, default is 5.

    See Also
    --------
    matplotlib.pyplot.hist:
        This function, used here, compute and display the histogram of the data.

    """
    # Sort the vectorized vector of connectivity entered
    sorted_vectorized_connectivity = sorted(vectorized_connectivity)
    
    # Call matplotlib histogram function to draw the histogram of the data
    if display_fit is 'yes':
        distribution_density = stats.norm.pdf(sorted_vectorized_connectivity,
                                              estimate_mean, estimate_std)
        plt.plot(sorted_vectorized_connectivity, distribution_density, '-o',
                 lw=line_width, label=legend_fitted,
                 color=fitted_distribution_color, ms=ms)
        plt.legend(prop={'size': 7})
        plt.title(title)
        plt.xlabel(xtitle)
        plt.ylabel(ytitle)
    elif display_fit is 'no':
        count_array, bins_array, _ = plt.hist(sorted_vectorized_connectivity, label=legend_data,
                                              bins=bins, alpha=alpha, lw=line_width,
                                              color=raw_data_colors, normed=False)

        plt.legend(prop={'size': 7})
        plt.title(title)
        plt.xlabel(xtitle)
        plt.ylabel(ytitle)
    else:
        raise ValueError('Unrecognized display option ! '
                         'Choices are yes or no and you entered {}'.format(display_fit))


def t_and_p_values_barplot(t_values, p_values, alpha_level, xlabel_color, bar_labels,
                           t_xlabel, t_ylabel, p_xlabel, p_ylabel,
                           t_title, p_title, xlabel_size=10,
                           ylabel_size=10, size_label=8):
    """Plot a barplot representation of t and p values resulting
    of a two sample Student t-test in two separates figure.



    """
    # Conversion in numpy array format of t and p values list
    pvalues = np.array(p_values)
    tvalues = np.array(t_values)
    # Plot a seaborn bar plot for p-values

    # Put in grey the bar above the type I error threshold
    bar_color = [xlabel_color[i] if (pvalues[i] < alpha_level) else 'grey'
                 for i in range(len(bar_labels))]
    plt.figure(constrained_layout=True)
    ax = sns.barplot(x=bar_labels, y=pvalues, palette=bar_color)
    plt.xlabel(p_xlabel, size=xlabel_size)
    plt.ylabel(p_ylabel, size=ylabel_size)
    # Set font size for the x axis, and rotate the labels of 90° for visibility
    for xtick, color in zip(ax.get_xticklabels(), bar_color):
        xtick.set_color(color)
        xtick.set_fontsize(size_label)
        xtick.set_rotation(90)
    # Plot a asymptote for p-value above alpha threshold
    plt.hlines(y=alpha_level, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1],
               label=str(alpha_level) + ' threshold', colors='red')
    plt.title(p_title)
    plt.legend()
    plt.show()
    # Plot the T statistic distribution
    plt.figure(constrained_layout=True)
    ax = sns.barplot(x=bar_labels, y=tvalues, palette=bar_color)
    plt.xlabel(t_xlabel, size=xlabel_size)
    plt.ylabel(t_ylabel, size=ylabel_size)
    for xtick, color in zip(ax.get_xticklabels(), bar_color):
        xtick.set_color(color)
        xtick.set_fontsize(size_label)
        xtick.set_rotation(90)
    plt.title(t_title)
   # plt.show()


def seaborn_scatterplot(x, y, data, figure_title, **kwargs):

    plt.figure()
    sns.lmplot(x=x, y=y, data=data, **kwargs)
    plt.title(figure_title)



