from nipype.interfaces.nipy import SpaceTimeRealigner

functional_image = 'D:\\RS1\\fMRIstar\\RS1_rm110247-2677_20110601_08.nii'

realigner = SpaceTimeRealigner()
realigner.inputs.in_file = [functional_image]
realigner.inputs.tr = 2.4
realigner.inputs.slice_times = 'ascending'
realigner.inputs.slice_info = 2
res = realigner.run()