Debugging Numba CUDA Programs with Visual Studio Code and CUDA GDB
==================================================================

Introduction
------------

With the release of the CUDA Toolkit (CTK) 13.1, CUDA GDB now includes beta support for debugging Numba CUDA programs on Linux. CUDA GDB is included in the CTK and is the backend debugger for the Nsight Visual Studio Code Edition extension to support debugging in Microsoft Visual Studio Code (``VSCode``).

Features included in this release:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Linux support
* Debugging of Numba CUDA programs with CUDA GDB using ``VSCode``.
* Debugging of Numba CUDA programs with CUDA GDB from the command line.
* Variable inspection and modification.
* Execution control (continue, step over, step into, step out, restart, stop).
* Setting breakpoints in the VSCode GUI.
* Setting breakpoints programmatically in Numba CUDA code by inserting a ``breakpoint()`` call.
* Formatting arrays in a more human readable format.
* Basic support for polymorphic variables.

These directions are for debugging with CUDA GDB using ``VSCode``, but CLI debugging with CUDA GDB is also supported. A more detailed description of debugging with VSCode in general can be found here: https://code.visualstudio.com/docs/editor/debugging.

Installation and Environment Setup
----------------------------------

To begin you’ll need an installation of CUDA Toolkit 13.1 and corresponding CUDA Driver for Linux. The installer for Linux can be downloaded from https://developer.nvidia.com/cuda-downloads. Follow the instructions provided on the download page to install it in ``/usr/local/cuda-13.1`` and create a symlink to it called ``/usr/local/cuda`` (which is handled by the install script).

You will also need a working Numba CUDA development environment. The documentation for Numba CUDA including the installation instructions can be found at: https://nvidia.github.io/numba-cuda. It is highly recommended that you use Anaconda, venv, or another Python virtual environment. The examples in this documentation are based on an Anaconda environment. In particular, make sure the required packages are installed in the virtual environment you wish to debug in.

The following commands can be used to create a Numba CUDA development environment using Anaconda. This example creates a virtual environment called ``numba-cuda-debug`` using Python 3.12.

.. code-block:: bash

    conda create --name numba-cuda-debug python=3.12
    conda activate numba-cuda-debug
    pip install cuda-python numba numba-cuda

Configure and Launch VSCode
---------------------------

* Clone the Numba CUDA repository from GitHub.

.. code-block:: bash

    git clone https://github.com/NVIDIA/numba-cuda.git

This is necessary to access the Numba CUDA debugging example code and the Numba CUDA pretty printer extension for CUDA GDB. The pretty printer extension is used to support formatting Numba CUDA arrays in a more human readable format. This is in addition to the install of ``numba-cuda`` with ``pip`` that was done earlier.

* Start VSCode using the ``debugging.code-workspace`` workspace file in the ``numba-cuda/examples/debugging`` folder:

.. code-block:: bash

    cd numba-cuda/examples/debugging
    code debugging.code-workspace

* In the VSCode Extensions View search for and install or update the following extensions:

    * Python (from Microsoft)
    * Nsight Visual Studio Code Edition (from NVIDIA)

* Press Ctrl+Shift+P to open the VSCode Command Palette and type "Python: Select Interpreter" to select the Python interpreter for the Numba CUDA virtual environment you wish to debug in (e.g. ``numba-cuda-debug``).

Review the Debug Configuration (launch.json)
--------------------------------------------

Using the Explorer Pane on the left hand side of the VSCode window open the provided ``.vscode/launch.json`` file by double clicking on it. You may have to expand the ``.vscode`` directory first in order to see it. This file contains the launch configuration for debugging Numba CUDA programs with VSCode.

.. image:: ../_static/launch-json.png
   :alt: The launch.json file

Since Numba CUDA programs are Python programs, they are executed within the python interpreter executable which is the program that CUDA GDB needs to debug. This configuration fragment selects the Python interpreter pointed to by the Python extension for VSCode by using the ``${command:python.interpreterPath}`` variable.

The Python script/program to run and any additional arguments are specified by the ``args`` entry. Here we are debugging the ``hello.py`` Numba CUDA program.

This launch configuration fragment can be used as a starting point for customizing the debugging of other Numba CUDA programs.

Starting Debugging
------------------

On the left hand side of the VSCode window look for the debugging icon (a right pointing arrow with a bug) and click on it. A drop-down menu will appear near the top. Select the ``Numba: hello.py Example`` menu entry and then click on the right pointing green arrow or press ``F5`` to start the program running under cuda-gdb.

.. image:: ../_static/starting-debugging.png
   :alt: Starting debugging.

Because ``breakOnLaunch`` is set to ``true`` in the ``launch.json`` configuration file, the program will automatically stop on the first source line of any launched kernel, which can be useful for getting started debugging. If this is not desired, change ``breakOnLaunch`` to be ``false``.

After starting the program, the debugger will stop at the first source line of the kernel launch. Your VSCode window should look like the following image. The program is stopped on line 18, which is indicated by the yellow arrow to the left of the line number.

.. image:: ../_static/kernel-entry.png
   :alt: Stopped at kernel entry

Controlling Execution, Setting Breakpoints, and Inspecting Variables
---------------------------------------------------------------------

After the program is stopped in the kernel, the user can use the buttons near the top center of the VSCode window to control program execution. From left to right these icons are: ``Continue`` / ``Step Over`` / ``Step Into`` / ``Step Out`` / ``Restart`` / ``Stop`` program execution. These buttons have hover text that shows their function.

.. image:: ../_static/run-control.png
   :alt: Visual Studio Code run control buttons

The following is a description of the functionality of each of the run control buttons and their corresponding CUDA GDB CLI commands.

* ``Continue`` will continue the program running until the next breakpoint or the program terminates. Equivalent to the ``continue`` command in CUDA GDB. When pressed, this icon will change to a pause icon. Pressing the pause icon will pause the program and return control to the user for further debugging.
* ``Step Over`` will step over one line of source code, stepping over function calls instead of into them. Equivalent to the ``next`` command in CUDA GDB.
* ``Step Into`` will step over one line of source code, stepping into any function calls made by that line of code. Equivalent to the ``step`` command in CUDA GDB.
* ``Step Out`` will step out of the current function, returning to the line of code that called it. Equivalent to the ``finish`` command in CUDA GDB.
* ``Restart`` will restart the program from the beginning. Equivalent to the ``run`` command in CUDA GDB.
* ``Stop`` will terminate the program and end the debugging session. Equivalent to the ``kill`` command in CUDA GDB.

Breakpoints can be set by clicking to the left of the source line numbers in the Editor window, and are shown as red dots next to the line numbers. Breakpoints can also be inserted programmatically by calling ``breakpoint()`` in the Numba CUDA code. See the ``hello.py`` program for an example.

From here, you can continue debugging the kernel by pressing the ``Continue`` button, stepping through the code one line at a time using the ``Step Over``, ``Step Into``, or ``Step Out`` buttons, or restarting the program from the beginning using the ``Restart`` button. You can also stop the program at any time using the ``Pause`` button.

Known Issues and Limitations
----------------------------

Polymorphic Variables
^^^^^^^^^^^^^^^^^^^^^

Unlike statically typed languages such as C or C++, Python variables are inherently polymorphic in nature. Any assignment to the variable can change its type as well as changing its value. Polymorphic variables are handled by Numba CUDA by creating a union. This union is exposed to the debugger as a single variable with the different types being the different members of the union. With this beta release, the user will have to manually determine which member of the union is the current one based on the code context. This limitation will be addressed in a future release.

CUDA GDB Pretty Printer Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA GDB supports extensions to the debugger written in Python. The Numba CUDA cuda-gdb pretty printer is such an extension. This is used to provide a more readable representation of Numba CUDA arrays in the debugger.

The Numba CUDA pretty printer extension is located in the ``numba-cuda/misc/gdb_print_extension.py`` file. This extension is loaded automatically by the ``launch.json`` file at the beginning of the debugging session. The `launch.json` example below assumes that you've opened the `debugging.code-workspace` workspace file in the `numba-cuda/examples/debugging` directory (due to the use of `..`). If the path to the ``gdb_print_extension.py`` file is not correct, simply edit the path to the ``misc/`` directory below.

.. code-block:: json-object

    {
        "environment": [
            {
                "name": "PYTHONPATH",
                "value": "${workspaceFolder}/../../misc:${env:PYTHONPATH}"
            }
        ]
    }

The CUDA GDB pretty printer uses the Python ``numpy`` package to inspect and print numpy arrays. The ``launch.json`` file sets up the environment necessary for debugging. However, the pretty printer runs inside of cuda-gdb and not as part of the Python program being debugged. This means that the ``numpy`` package must be installed in both:
* The Python environment where VSCode was started in (required by the Numba CUDA cuda-gdb pretty printer).
* The Python environment where the Numba CUDA program is being debugged (required by Numba CUDA).

These can both use the same Python environment, but that is not required.

Automatic loading of the pretty printer extension is done by the following in the ``launch.json`` file. This command is executed before CUDA GDB is started. Failure to find the extension is ignored in case ``PYTHONPATH`` is not set correctly.

.. code-block:: json-object

    {
        "setupCommands": [
            {
                "description": "Load the Numba CUDA cuda-gdb pretty-printer extension",
                "text": "python import gdb_print_extension",
                "ignoreFailures": true
            }
        ]
    }

Debugging Host Python Code
^^^^^^^^^^^^^^^^^^^^^^^^^^

Debugging host Python code using the ``debugpy`` package is not supported when also debugging with CUDA GDB. Numba CUDA programs are executed within the Python interpreter, which is the program that CUDA GDB controls during the debugging session. The ``debugpy`` python module also executes in the Python interpreter, which means that it requires that the interpreter be actively running in order for the VSCode Python debugger to communicate with it. However, CUDA GDB will stop both the CUDA host application and any code executing on the GPU while debugging, which prevents the ``debugpy`` package from running.

Debugging host Python code manually from the command line with the ``pdb`` package is supported, with the limitation that while cuda-gdb has the host application stopped pdb commands will not function (until the program is resumed).

