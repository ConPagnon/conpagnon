#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 17:32:38 2017

@author: Dhaif BEKHA

ComPagnon version 2.0

"""
import os
import pandas as pd
import re
import glob

from conpagnon.utils import folders_and_files_management as ffm


def read_text_data_file(file_path, colname=None, header=None):
    """Read a data file 
    
    The data file can be a .csv, .txt, .xlsx or .xls file
    
    Parameters
    ----------
    
    file_path : str
        Full path to the file to read.
    colname : None or str
        The column name to extract.
    header : None of int
        Row number to use as the column names, and the start of the data.
        
    Returns
    -------
    
    output : pandas.core.frame.DataFrame
        The extracted column if form of a panda dataframe.
    
    """
    
    if file_path.lower().endswith(('.csv', '.txt')):
        data = pd.read_csv(file_path, header = header)
    elif file_path.lower().endswith(('.xlsx', '.xls')):
        data = pd.read_excel(file_path, header = header)
        
    if colname:
        data = data[colname]
    
    return data


def list_fmri_data(root_fmri_data_directory):
    """Fetch all functional images found in sub-directories
    at a root directory.
    
    Parameters
    ----------
    root_fmri_data_directory : str
        The full path of a root directory containing one or numerous
        sub-directories where functional images are.
    
    
    Returns
    -------
    output : dict
        A dictionnary with sub-directories names as keys and full path
        to functional images as values.
    """
    # Initialisation of list which will contain full path to functional images
    fmri_files = []
    
    # Check if sub-directories are empty, delete them if this is the case
    ffm.remove_empty_directories(root_directory = root_fmri_data_directory)
    
    for path, dirs, files in os.walk(root_fmri_data_directory):
        # List of the different sub-directories
        if dirs != []:
            groupes = dirs
        # Append fmri files
        if files != []:
            fmri_files.append(files)
                
    # Creating the dictionnary with sub-directories names as keys, i.e
    # the different groups of the study and full path to functional images
    # as values
    n_groupes = len(groupes)
    fmri_files_dict = dict([(groupes[groupe], fmri_files[fmri_file])
                            for groupe, fmri_file in zip(range(n_groupes),range(n_groupes)) ])

    # Filling the dictionnary values with full path to fmri files for each keys
    for groupe in fmri_files_dict.keys():
        # Name of the images
        list_name_img = fmri_files_dict[groupe]
        # Appending the full path
        list_img_fullpath = [ os.path.join(root_fmri_data_directory, groupe) + '/' + name_img 
                             for name_img in list_name_img]
        fmri_files_dict[groupe] = list_img_fullpath
        del list_name_img
        
    return fmri_files_dict


def fetch_fmri_data(root_fmri_data_directory, groupes):
    """Fetch functional images found in a list of 
    sub-directories.
    
    Parameters
    ----------
    
    root_fmri_data_directory : str
        The full path of a root directory containing one or numerous
        sub-directories where functional images are.
        
    groupes : list
        List of sub-directories names containing fmri files you want.
    
    
    Returns
    -------
    
    output : dict
        A dictionnary with sub-directories names as keys and full path
        to functional images as values.
        
    See Also
    --------
    
    list_fmri_data: Fetch functional images in all sub-directories.
    
    """
    # We fetch all fmri data thank to list_fmri_data
    fmri_files_dict = list_fmri_data(root_fmri_data_directory=root_fmri_data_directory)
    
    # We extract a sub-dictionnary according to the list entered in groupes:
    # We check that the elements of groupes are existing keys in fmri_files_dict
    for i in range(len(groupes)):
        if groupes[i] not in fmri_files_dict.keys():
            raise ValueError('Group {} doesn\'t exist !. available groups in {} are \n {}'.format(groupes[i],
                             root_fmri_data_directory, fmri_files_dict.keys()))
        else:
            pass
            
    # If keys exist, we can extract the sub-dictionnary to return the new ones
    group_fmri_dict = dict((g, fmri_files_dict[g]) for g in groupes)
    
    return group_fmri_dict


def create_group_dictionnary(subjects_id_data_path, root_fmri_data_directory, groupes):
    """Initialise a dictionnary containing groups as keys and subjects IDs as values
    
    Parameters
    ----------
    
    subjects_id_data_path : str
        The full path to the data file containing the subjects IDs.
            
    root_fmri_data_directory : str
        The full path of a root directory containing one or numerous
        sub-directories where functional images are
        
    groupes : list
        List of sub-directories names containing fmri files you want.
    
    
    Returns
    -------
    
    output : dict
        A dictionnary groupes as keys and subjects IDs as values for each
        groups
        
    See Also
    --------
    
    list_fmri_data : Fetch functional images in all sub-directories.
    read_text_data_file : read a text data file.
    
    Notes
    -----
    
    Whatever the format for the subjects IDs datafile, it should not
    contains any header. it should consist of one raw columns of subject 
    IDs.
    
    
    """
    dict2 = dict.fromkeys(groupes)
    subjects_ID = read_text_data_file(file_path=subjects_id_data_path, colname=None, header=None)
    for groupe in groupes:
        # List available functional images
        a = glob.glob(os.path.join(root_fmri_data_directory, groupe, '*'))
        # Search and extraction of subjects identifier
        b= [re.search(s_id,f_name) for f_name in a for s_id in subjects_ID[0]]
        #  Fetch the subjects identifier present
        c = [b[i].group(0) for i in range(len(b)) if b[i] != None]
        # Create a subject identifier dictionary per group
        dict_sub_id = dict.fromkeys(c)

        dict2[groupe] = dict_sub_id
        
    return dict2


def fetch_data_with_individual_atlases(subjects_id_data_path, root_fmri_data_directory, groupes,
                                       individual_atlases_directory, individual_atlases_labels_directory,
                                       individual_atlas_file_extension, individual_atlas_labels_extension,
                                       individual_counfounds_directory=None):
    """Fetch a complete organised structure for a groups study require the use
    of individual atlases
    
    
    Parameters
    ----------
    subjects_id_data_path: str
        The full path to the data file containing the subjects IDs.
        
    root_fmri_data_directory: str
        The full path of a root directory containing one or numerous
        sub-directories where functional images are
        
    groupes: list
        List of sub-directories names containing fmri files you want.
        
    individual_atlases_directory: str
        Full path to individual atlases directory for all subjects.
        
    individual_atlases_labels_directory: str
        Full path to individual atlases labels directory for all subjects.
        
    individual_atlas_file_extension: str
        Extension of individuals atlases images for all subjects.
        
    individual_atlas_labels_extension: str
        Extension of text data containing individual atlases labels file for
        all subjects.
        
    individual_counfounds_directory : None or str
        Full path to counfounds files for all subjects.

    Returns
    -------
    output: dict
        A multi level dictionnary containing all the data. 
        The first level is the groups keys. The second levels is the subjects 
        IDs. The last level, is all the relevant file for one subjects: fmri image,
        subject atlas, subject atlas labels file, and confound file if required.
        A keys called 'discarded_rois' contain the excluded rois, see Notes.
    
    See Also  
    --------
    fetch_data : Fetch a complete organised structure for a groups study
    on a common atlas for all subjects.
    
    Notes
    -----
    Whatever the format for the subjects IDs datafile, it should not
    contains any header. it should consist of one raw columns of subject 
    IDs.
    
    A discarded_rois is a subject atlas ROI where the corresponding
    labels is 'void'. This ROIS can be a empty ROIs, a ROI you doesnt need
    for the analysis. The discarded rois will be discarded (!) in the 
    connectivity analysis when computing t-test for example. 
    
    
    """
    
    group_fmri_dict = fetch_fmri_data(root_fmri_data_directory=root_fmri_data_directory,
                                      groupes=groupes)

    subjects_ID = read_text_data_file(file_path=subjects_id_data_path, colname=None,
                                      header=None)
    
    # Full path to individual atlases files and labels file directory
    individual_atlases_files = glob.glob(os.path.join(individual_atlases_directory,
                                                      individual_atlas_file_extension))
    individual_atlases_labels_files = glob.glob(os.path.join(individual_atlases_labels_directory,
                                                             individual_atlas_labels_extension))
    
    # Check if counfounds files is required
    if individual_counfounds_directory != None:
        individual_confounds_files = glob.glob(os.path.join(individual_counfounds_directory, '*'))
    else:
        individual_confounds_files = False

    dict1 = {}
    n_groupes = len(groupes)

    for s_id in subjects_ID[0]:
        for groupe in range(n_groupes):
            # Subjects ID to look for
            regex = re.compile(".*("+str(s_id)+").*")
            # Fetch the correspond fmri file
            subject_functional_file = [m.group(0) for l in group_fmri_dict[groupes[groupe]]
                                       for m in [regex.search(l)] if m]
            # If fmri exist, fetch the corresponding atlas file, and atlas labels files
            if subject_functional_file != []:
                # Subject atlas
                subject_atlas_file = [m.group(0) for l in individual_atlases_files
                                      for m in [regex.search(l)] if m]
                # Subject atlas labels
                subject_atlas_labels_file = [m.group(0) for l in individual_atlases_labels_files
                                             for m in [regex.search(l)] if m]
                
                # Append confound if they exist:
                if individual_confounds_files:
                    subject_counfounds_file = [m.group(0) for l in individual_confounds_files
                                               for m in [regex.search(l)] if m]
                    
                else:
                    subject_counfounds_file = []
                
                # Creation of a sub-dictionary for each subject containing the full
                # path to files as values, and file descriptions as keys
                subject_files = {'functional_file': subject_functional_file,
                                 'atlas_file': subject_atlas_file,
                                 'label_file': subject_atlas_labels_file,
                                 'counfounds_file': subject_counfounds_file}
                # Appending the sub-dictionnary to the corresponding subjects entries
                dict1[s_id] = subject_files
    
    # We initialize the final dictionnary with groups as keys and subject ID as values
    dict2 = create_group_dictionnary(subjects_id_data_path=subjects_id_data_path,
                                     root_fmri_data_directory=root_fmri_data_directory,
                                     groupes=groupes)

    # We fill the final dictionnary with the information gathered for each subject
    for groupe in groupes:
        for s_id in subjects_ID[0]:
            if s_id in dict2[groupe].keys():
                 dict2[groupe][s_id] = dict1[s_id]

    return dict2


def fetch_data(subjects_id_data_path, root_fmri_data_directory, groupes,
               individual_confounds_directory=None):
    """Fetch a complete organised structure for a groups study on a common
    atlas for all subjects
    
    
    Parameters
    ----------
    subjects_id_data_path : str
        The full path to the data file containing the subjects IDs.
        
    root_fmri_data_directory : str
        The full path of a root directory containing one or numerous
        sub-directories where functional images are
        
    groupes : list
        List of sub-directories names containing fmri files you want.
        
    individual_confounds_directory : None or str
        Full path to confounds files for all subjects.


    Returns
    -------
    output : dict
        A multi level dictionnary containing all the data. 
        The first level is the groups keys. The second levels is the subjects 
        IDs. The last level, is all the relevant file for one subjects: fmri image,
        and confound file if required.
    
    See Also  
    --------
    fetch_data_with_individual_atlases : Fetch a complete organised
    structure for a groups study which require an individual atlas per subjects.
    
    Notes
    -----
    Whatever the format for the subjects IDs datafile, it should not
    contains any header. it should consist of one raw columns of subject 
    IDs.

    """
    
    group_fmri_dict = fetch_fmri_data(root_fmri_data_directory=root_fmri_data_directory,
                                      groupes=groupes)

    subjects_ID = read_text_data_file(file_path=subjects_id_data_path,
                                      colname=None, header=None)
    
    if individual_confounds_directory != None:
        individual_confounds_files = glob.glob(os.path.join(individual_confounds_directory, '*'))
    else:
        individual_confounds_files = False

    dict1 = {}

    n_groupes = len(groupes)

    for s_id in subjects_ID[0]:
        for groupe in range(n_groupes):
            # the subjects ID to look for
            regex = re.compile(".*("+str(s_id)+").*")
            # Fetch of fmri image
            subject_functional_file = [m.group(0) for l in group_fmri_dict[groupes[groupe]]
                                       for m in [regex.search(l)] if m]
            # If fmri exist we look for a confound file
            if subject_functional_file != []:                
                # If a confound file we append it:
                if individual_confounds_files:
                    subject_counfounds_file = [m.group(0) for l in individual_confounds_files
                                               for m in [regex.search(l)] if m]
                else:
                    subject_counfounds_file = []
                
                # Creation of a sub-dictionary for each subject containing the full path
                # to files as values, and file descriptions as keys
                subject_files = {'functional_file': subject_functional_file,
                                 'counfounds_file': subject_counfounds_file}
                # We append the sub-dictionnary to the subject dictionnary for each subject
                dict1[s_id] = subject_files
    
    # We initialize the final dictionnary with groups as keys and subject ID as values
    dict2 = create_group_dictionnary(subjects_id_data_path=subjects_id_data_path,
                                     root_fmri_data_directory=root_fmri_data_directory,
                                     groupes=groupes)

    # We fill the final dictionnary with the information gathered for each subject
    for groupe in groupes:
        for s_id in subjects_ID[0]:
            if s_id in dict2[groupe].keys():
                 dict2[groupe][s_id] = dict1[s_id]

    return dict2


