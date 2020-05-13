Contributing to ConPagnon
=========================

Makes changes to the code
-------------------------

ConPagnon is a open-source Python library
and anyone can contribute to it, adding
functionality, or tailored some algorithm
to your own study with some functionality
that is too specific to end up in the
official repository.

Forking and cloning
~~~~~~~~~~~~~~~~~~~

Make sure you have **Git** installed in your
system. Once you have set up Git, please
go the official `GitHub repository <https://github.com/ConPagnon/conpagnon>`_
and hit the **Fork** button in the top right corner. If you have any trouble,
do not hesitate to report to the official instruction on how `to fork a
repository on GitHub <https://help.github.com/articles/fork-a-repo/>`_.
This step then open your fork page. It's like you own safe copy of
the folder containing the full code of ConPagnon. Now you can
work on your own copy using the following command lines:

.. code-block:: bash

    git clone https://github.com/your-user-name/conpagnon
    cd conpagnon
    git remote add upstream https://github.com/conpagnon/conpagnon
    git fetch --all

The third line sets-up a read-only connection to the main ConPagnon
repository. This will allow you to update whenever you need your local code with
changes in the official ConPagnon repository.  The final command
fetches both your repository and the upstream ConPagnon repository.

Create a Branch
~~~~~~~~~~~~~~~

It's good practice, that every changes you made are made inside
a **branch**. To create a branch, enter the following command
lines:

.. code-block:: bash

    git checkout master
    git rebase upstream/master
    git checkout -b your-new-feature-name

Those command ensure you are starting from an up-to-date version of the official
ConPagnon repository, and finally create a new branch.

.. code-block:: bash

    git branch

will output the following scheme:

.. code-block:: bash

    * your-new-feature-name
      master

to indicate that you are now on a new branch, named after *your-new-features-name*.

Making changes
~~~~~~~~~~~~~~

You are ready to make new changes. The only rules is to keep
your changes in your branches. Every time you modify a file
you can see the changes your made by entering the following
command line:

.. code-block:: bash

    git status

Pushing your modifications
~~~~~~~~~~~~~~~~~~~~~~~~~~

Once your are satisfied of the changes
your made, and if it's the first time
you've made changes, please enter the
following command line:

.. code-block:: bash

    git push

and, for all the other time you want to push changes:

.. code-block:: bash

    git push --set-upstream origin your-new-feature-name

that tell git to set the current branch to track its corresponding branch in
your github repository.

You can see the remote repositories by

.. code-block:: bash

    git remote -v

Pull Requests
~~~~~~~~~~~~~

When you final and definitive changes are committed
to your repository, you are ready to make a
**pull request**, and ask for a code review
by maintainers of ConPagnon. Please follow
the below instructions to make a pull
request:


#. Navigate to your repository on github.
#. Click on `Branch List`
#. Click on the `Compare` button for your feature branch, `your-new-feature-name`.
#. Select the `base` and `compare` branches, if necessary.
#. Check the overview of your changes

Now you can make a `pull request <https://help.github.com/articles/about-pull-requests/>`__.

#. Navigate to your own repository on github.
#. Click on the `Pull Request` button.
#. Write a description of your changes in the `Preview Discussion` tab.
#. Click `Send Pull Request`.

Your request will then be reviewed. If you need to go back and make more
changes, you can make them in your branch and push them to github and the pull
request will be automatically updated.
