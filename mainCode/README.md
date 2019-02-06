## Prerequisites
Everthing should be already installed except for pymultinest. This is installed under the home directory (`cd ~`) the following commands:

`git clone https://github.com/JohannesBuchner/PyMultiNest/`

`cd PyMultiNest`

`python3 setup.py install --user`

Then cd back to the top directory above PyMutliNest

`git clone https://github.com/JohannesBuchner/MultiNest`

`cd MultiNest/build`

`cmake ..`

`make`

The hard part is getting pymultinest to recognise multinest. One of these should do the job...

`echo 'export LD_LIBRARY_PATH=/home/nicholas.farrow/MultiNest/lib/:$LD_LIBRARY_PATH'`

`echo 'export LD_LIBRARY_PATH=/home/FIRSTNAME.LASTNAME/MultiNest/lib/:$LD_LIBRARY_PATH' >>~/.bashrc`

Check if this worked by using `python3` then `import pymultinest`. If nothing shows up then everthing is working.


## Individual Sampling
This should be done first to check that all the required packages are installed (especially pymultinest).
Sampling using a specific model (or model pairs) can be done through calling

`python3 hypothesisB.py 3`

Which runs hypothesis B for the 3rd model pair (single - uniform).


## Distributed Computing
As sampling can take very long periods of time (12 combinations of sub-hypotheses ranging between 2 and 7 parameters), it is required that sampling be done through a distributed computing system. 
The method of providing a number argument to the code allows for easy batch running of the script.

`python3 hypothesisA.py [1-3]`.
`python3 hypothesisB.py [1-9]`.

##  Basic analysis of output
`python3 hypothesisAnalyse.py A hypothesisA_out/`

`python3 hypothesisAnalyse.py B hypothesisB_out/`

This creates files like evidence lists, relative evidence tables and totals, and also latex tables.
