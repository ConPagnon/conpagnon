from nipype.interfaces.nipy import SpaceTimeRealigner
import numpy as np
import matplotlib.pyplot as plt

functional_image = 'D:\\RS1\\fMRIstar\\RS1_rm110247-2677_20110601_08.nii'

realigner = SpaceTimeRealigner()
realigner.inputs.in_file = [functional_image]
realigner.inputs.tr = 2.4
realigner.inputs.slice_times = 'ascending'
realigner.inputs.slice_info = 2
res = realigner.run()


# Load and plot translation and rotation
movements_parameters = 'D:\\RS1\\fMRIstar\\RS1_rm110247-2677_20110601_08.nii.par'
movements = np.loadtxt(movements_parameters, delimiter=' ')
translations = movements[:, 0:3]
rotations = movements[:, -3:]

# Plot the translation


translations_labels = ['X-translation',
                       'Y-translation',
                       'Z-translation']
translations_colors = ['red', 'green', 'blue']
for t in range(translations.shape[1]):
    plt.plot(translations.shape[0], translations[:, t], translations_colors[t])
    plt.legend(translations_labels[t])
    plt.show()