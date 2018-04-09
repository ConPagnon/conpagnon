#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 13:34:12 2017

@author: db242421


Utilitary modules.

"""


import os
import pickle
from PyPDF2 import PdfFileWriter, PdfFileReader
import shutil


def remove_empty_directories(root_directory):

    """Erase all empty directory at a root location.

    Parameters
    ----------
    root_directory : str
        The full path to a directory.

    """
    sub_dir_list = os.listdir(root_directory)

    for sub_dir in sub_dir_list:
        if not os.listdir(os.path.join(root_directory, sub_dir)):
            os.rmdir(os.path.join(root_directory, sub_dir))


def check_directories_existence(root_directory, directories_list):
    """Check if all directories supplied exist, raises an exception if not.

    Parameters
    ----------
    root_directory : str
        The full path to a root directory, containing multiple folders.

    directories_list : list
        List of directories you want to check for, in root_directory.

    Raises
    ------
    Exception
        Raise an exception if one of the directories supplied
        do not exist.

    """

    # la liste des dossiers a la racine root_directory
    sub_dir_list = os.listdir(root_directory)

    for d in directories_list:
        if d not in sub_dir_list:
            raise Exception('le dossier {} n\'existe pas ! Les dossiers '
                            'disponibles dans {} sont {}'.format(d, root_directory, sub_dir_list))


def save_object(object_to_save, saving_directory, filename):
    """Save a python object in pickle format.

    Parameters
    ----------
    object_to_save : a Python object structure
        A python that can be pickled : list, dict, integer,
        tuples, etc...

    saving_directory : str
        Full path to a saving directory.

    filename : str
        Filename of the saved object, including the extension .pkl.

    See Also
    --------
    load_object :
        This function load a python object save
        by this function.

    Notes
    -----
    See the pickle module for python object to see
    a exhausting list of what object can be saved.

    """
    full_path_to_file = os.path.join(saving_directory, filename)
    with open(full_path_to_file, 'wb+') as output:
        pickle.dump(obj=object_to_save, file=output, protocol=pickle.HIGHEST_PROTOCOL )


def load_object(full_path_to_object):

    """Load a python object saved by a pickler operator.

    Parameters
    ----------
    full_path_to_object : str
        The full path to the saved python object, including
        the .pkl extension.

    Returns
    -------
    output : python object
        The loaded python object.

    See Also
    --------
    save_object :
        This function save a python object
        using the pickle module.

    """
    with open(full_path_to_object, 'rb') as pickleObj:

        object_to_load = pickle.load(pickleObj)
    return object_to_load


# Creating a routine that appends files to the output file
def append_pdf(input, output):
    """Append a input pdf file to an output empty pdf file.

    """
    [output.addPage(input.getPage(page_num)) for page_num in range(input.numPages)]


def merge_pdfs(pdfs, output_filename):
    """Merge together a list of pdf files.

    """

    # Creating an object where pdf pages are appended to
    output = PdfFileWriter()

    # Appending two pdf-pages from two different files
    for pdf in pdfs:
        # read the pdf file
        pdf_ = PdfFileReader(pdf)
        append_pdf(pdf_,output)

    # Writing all the collected pages to a file
    output.write(open(output_filename, "wb"))


def create_directory(directory, erase_previous=False):
    """Create a directory, erase it and create it
    if it exist.
    """

    # If directory doesn't exist we create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        # It it exist, and erase_previous is True we erase it
        if erase_previous:
            shutil.rmtree(directory)
            os.makedirs(directory)

    return directory
