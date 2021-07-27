# -*- coding: utf-8 -*-
"""

@author: JBC @ Northwestern University (jibril.coulibaly@gmail.com)

script: main_script_post3d.py

Python 2 post-processing scripts for LAMMPS 3-dimensional DEM simulations post-processing
The purpose of this script is to get a very clean working directory
Call the functions from a base location
get the input from a target directory
save the output in a target directory
rename conventions etc


The naming of the variables in the dump dictionaries MUST respect the following standards:
- Geometry dump:
    x,y,z: particle coordinates
    radius, diameter: radius, diameter of the particle
    TO BE COMPLETED
Usual geometry dictionaries, user-defined:
    {"id":1, "x":2, "y":3, "z":4, "diameter":5}
    {"id":1, "x":2, "y":3, "z":4, "radius":5}
    {"id":1, "x":2, "y":3, "z":4, "radius":5, "diameter":6}
    {"id":1, "type":2, "x":3, "y":4, "z":5, "diameter":6}
    {"id":1, "type":2, "x":3, "y":4, "z":5, "radius":6}  
    TO BE COMPLETED     

- Topology dump:
    id1, id2: the atom ID of particles 1 and 2 in the given contact
    fn: the normal force of the contact
    ft: the tangent force of the contact
    dist: the distance between particles id1 and id2
    lx,ly,lz: the components of the branch vector from particle id2 towards particle id1
    TO BE COMPLETED
Usual topology dictionaries, user-defined:
    {"id1":1, "id2":2, "dist":3, "lx":4, "ly":5, "lz":6}
    {"id1":1, "id2":2, "fn":3, "lx":4, "ly":5, "lz":6}
    {"id1":1, "id2":2, "fn":3, "ft":4, "lx":5, "ly":6, "lz":7}
    {"id1":1, "id2":2, "dist":3, "fn":4, "ft":5, "lx":6, "ly":7, "lz":8}
    TO BE COMPLETED

The naming of the variables in the log dictionaries has no fixed standard.
Only the values provided in the dictionary are read. Not all keys are necessary
The column number in the dictionary must match that of the intended variable in
the log file
Here are examples of dictionaries for usual simulations (as of 7/3/2020)
(these may change when the corresponding input files are changed to save different variables)
- Usual Heating-Cooling log dictionary:
    {"step":1, "temperature":2, "epsx":3, "epsy":4, "epsz":5, "epsv":6, "pf":7, "pxx":8, "pyy":9, "pzz":10, "press":11, "qdev":12, "pxy":13, "pxz":14, "pyz":15}
    
Usual workflow:

(1) select and load the files

dic = {} # Dictionary for the dump files
input_path = "" # Path were dump files are located, must use raw string (r"") or replace backslash "\" with slash "/" on Windows
input_fnames =[] # list of the names of the dump files
with cd(input_path):
    dmp = dump(" ".join(input_fnames)) # Reading of the selected dump files
post3d.map_dic(dmp,dic)
# dump() automatically sorts by increasing timestep and deletes duplicates

(2) apply selected functions

results = post3d.density(dmp)
results = post3d.coordination(dmp,Npart)
results = post3d.fabric_tensor(dmp)
results = post3d.stiffness_tensor(dmptopo,dmpco,E,nu)
results = post3d.force_distribution(dmptopo,Nsample,fn_range,mob_range,mu)

(3) format save results

output_csv(data,headers,outstyle,fnames)

"""

from dump import dump # The folder of Pizza sources must be in the pythonpath
import post3d # The folder of post3d.py sources must be in the pythonpath
from log import log # The folder of pizza.py sources must be in the pythonpath
from log_duplicate import log_duplicate # The folder of pizza.py sources must be in the pythonpath
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

import glob
import os, errno

class cd:
    """Context manager for changing the current working directory
    (https://stackoverflow.com/questions/431684/how-do-i-change-the-working-directory-in-python)
    We need this because the dump function of the Pizza package
    reads spaces as file separators and we cannot use paths with spaces"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)
        
"""############################################################################
###############################################################################
############################################################################"""

# ====
# INPUT AND OUTPUT PATHS
# ====
input_path = r"<path_to_dump/log_files>"

output_path = r"<path_to_saved_output>"
if not os.path.exists(output_path):
    try:
        os.makedirs(output_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

# ====
# SAMPLE SCRIPTS
# ====

""" # Example 1: read log file from oedometric test simulation

input_path = r"C:\Users\jibri\Desktop\test_log"

output_path = input_path+r"/outdir"
if not os.path.exists(output_path):
    try:
        os.makedirs(output_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


fname = "log.oedo3"
dic = {"pzz":4, "press":5, "qdev":6, "epsz":12, "evoid":14}
with cd(input_path):
    lg = log(fname) # Reading of the selected log file

results = post3d.read_log(lg,dic)
plt.plot(results["pzz"],results["evoid"])
plt.xscale('log')

out_values = np.transpose(results.values())
out_header = results.keys()
output_fnames = [fname+'_out.csv'] # list of the names of the saved files
outstyle = 'single'
with cd(output_path):
    post3d.output_csv(out_values,out_header,outstyle,output_fnames)

"""



""" # Example 2: read log file from triaxial test simulation

input_path = r"C:\Users\jibri\Desktop\test_log"

output_path = input_path+r"/outdir"
if not os.path.exists(output_path):
    try:
        os.makedirs(output_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

fname = "log.triax3"
dic = {"pxx":2, "pyy":3, "pzz":4, press":5, "qdev":6, "epsx":10, "epsy":11, "epsz":12, "epsv":13}
with cd(input_path):
    lg = log(fname) # Reading of the selected log file

results = post3d.read_log(lg,dic)
plt.figure(1)
plt.plot(-100*results["epsz"],results["qdev"]/1000)
plt.xlabel('Axial strain [%]')
plt.ylabel('Deviatoric stress [kPa]')

plt.figure(2)
plt.plot(-100*results["epsz"],-100results["epsv"])
plt.xlabel('Axial strain [%]')
plt.ylabel('Volumetric strain [%]')

out_values = np.transpose(results.values())
out_header = results.keys()
output_fnames = [fname+'_out.csv'] # list of the names of the saved files
outstyle = 'single'
with cd(output_path):
    post3d.output_csv(out_values,out_header,outstyle,output_fnames)
"""



""" # Example 3: read log file from heating-cooling simulation

input_path = r"C:\Users\jibri\Desktop\test_log"

output_path = input_path+r"/outdir"
if not os.path.exists(output_path):
    try:
        os.makedirs(output_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

fname = "log.hc3"
dic = {"temperature":2, "epsv":3}
with cd(input_path):
    lg = log(fname) # Reading of the selected log file

results = post3d.read_log(lg,dic)
plt.plot(-100*results["epsv"],results["temperature"])
plt.xlabel('Volumetric strain [%]')
plt.ylabel('Temperature [C]')

out_values = np.transpose(results.values())
out_header = results.keys()
output_fnames = [fname+'_out.csv'] # list of the names of the saved files
outstyle = 'single'
with cd(output_path):
    post3d.output_csv(out_values,out_header,outstyle,output_fnames)

"""



""" # Example 4: Obtain geometry and topology information from one sample

input_path = r"C:\Users\jibri\Desktop\test_geo_top"

output_path = input_path+r"/outdir"
if not os.path.exists(output_path):
    try:
        os.makedirs(output_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

# Geometry
input_fname = "dump.co.lammpstrj"
dic = {"id":1, "x":2, "y":3, "z":4, "diameter":5}
with cd(input_path):
    dmp = dump(input_fname) # Reading of the selected dump files
post3d.map_dic(dmp,dic)

dens = post3d.density(dmp)
Npart = dmp.snaps[0].natoms # Number of particles

output_fnames = ["sample_geom.csv"] # Must be a list for output_csv() to work
output_values = np.transpose(dens.values())
output_header = dens.keys()
output_style = 'single'
with cd(output_path):
    post3d.output_csv(output_values,output_header,output_style,output_fnames)

# Topology
input_fname = "dump.topo.lammpstrj"
dic = {"id1":1, "id2":2, "fn":3, "ft":4, "lx":5, "ly":6, "lz":7}
with cd(input_path):
    dmp = dump(input_fname) # Reading of the selected dump files
post3d.map_dic(dmp,dic)

coor = post3d.coordination(dmp,Npart)

output_fnames = ["sample_topo.csv"] # Must be a list for output_csv() to work
output_values = np.transpose(coor.values())
output_header = coor.keys()
output_style = 'single'
with cd(output_path):
    post3d.output_csv(output_values,output_header,output_style,output_fnames)

"""




""" # Example 5: Obtain topology evolution through thermal cycles


input_path = r"C:\Users\jibri\Desktop\Recalculation_rattlers\CONFIG_1_mono"

output_path = input_path+r"/outdir"
if not os.path.exists(output_path):
    try:
        os.makedirs(output_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


Nparticle = 8000

with cd(input_path):
    input_fnames = glob.glob("dump.topo_sample_after_cooling*") # Contact topology at the end of the HC cycle
    input_fnames.insert(0,glob.glob("dump.topo_sample_before_cycles*")[0]) # initial state before cycles
    # No need to sort, dump() sorts snapshots automatically

dic = {"id1":1, "id2":2, "fn":3, "ft":4, "lx":5, "ly":6, "lz":7}
with cd(input_path):
    dmp = dump(" ".join(input_fnames)) # Reading of all the selected dump files
post3d.map_dic(dmp,dic)

coor = post3d.coordination(dmp,Nparticle)

Ncycle = np.arange(1001)
output_fnames = ["heat_cool_test_ID_x0_and_z.csv"]

output_values = np.concatenate((Ncycle[:,np.newaxis],np.transpose(coor.values())),axis=1)
output_header = ["Ncycle"]+coor.keys()
output_style = 'single'
with cd(output_path):
    post3d.output_csv(output_values,output_header,output_style,output_fnames)


plt.figure(1)
plt.plot(Ncycle,100*coor["rattlers fraction"])
plt.xlabel('Number of cycles')
plt.ylabel('rattlers proportion [%]')

plt.figure(2)
plt.plot(Ncycle,coor["corrected coordination"])
plt.xlabel('Number of cycles')
plt.ylabel('Corrected coordination [-]')

"""





















"""
EXAMPLE N: Obtain fabric tensor for many snaps of a simulation

TO BE WRITTEN


out_values = np.transpose(results.values()) # append the range variable values
out_header = results.keys()
#output_fnames =["outfname.csv"] # list of the names of the saved files
output_fnames = ['{0}_out.csv'.format(i) for i in input_fnames]
outstyle = 'multi' # 'single' or 'multi'
with cd(output_path):
    post3d.output_csv(out_values,out_header,outstyle,output_fnames)

"""




# LEGACY HELP STUFF, DOES NOT MATTER

# Example 1 : get the#
# (1) select and load the files

#dic = {"id":1, "x":2, "y":3, "z":4, "radius":5, "diameter":6} # Dictionary for the dump files
#dic = {"id1":1, "id2":2, "dist":3, "fn":4, "ft":5, "lx":6, "ly":7, "lz":8}
#input_path = r"C:\Users\jibri\Desktop\test_code_Manan\campaign_results\CONFIG_1_mono" # Path were dump files are located
#input_path = r"C:\Users\jibri\Documents\POSTDOC_JBC\001 - CODES\LAMMPS\sample_database\SPHERE_Foundry\sphere_foundry_N8000_P1kPa_PBC_01_pf0.6169"
#input_fnames =["dump.topo_sample_3.lammpstrj"] # list of the names of the dump files
#with cd(input_path):
#    input_fnames = glob.glob("dump.co*") # Reading a list of filenames directly from the input folder
    

#output_path = input_path+r"/outdir"
#output_fnames =["outfname.csv"] # list of the names of the saved files
#output_fnames = ['{0}_out.csv'.format(i) for i in input_fnames]
#if not os.path.exists(output_path):
#    try:
#        os.makedirs(output_path)
#    except OSError as e:
#        if e.errno != errno.EEXIST:
#            raise
#
#with cd(input_path):
#    dmp = dump(" ".join(input_fnames)) # Reading of the selected dump files
#post3d.map_dic(dmp,dic)
#timestep = np.array([dmp.snaps[t].time for t in range(len(input_fnames))])


#(2) apply selected functions



#results = post3d.density(dmp)
#results = post3d.fabric_tensor(dmp)
#results = post3d.coordination(dmp,8000)
#results = post3d.stiffness_tensor(dmptopo,dmpco,50e9,0.25)

# (3) format and save results

""" Can we generalize the definition of the output variables?

all dimensions
all concatenation
all manual range values
"""

#range_header = "Ncycle" # Variable of range (optional)
#range_values = range(0,10,1) # Values of the range variable (optional)

#out_values = np.transpose(density_results.values()) # append the range variable values
#out_header = density_results.keys() # append the range variable header

#out_values = np.transpose(results.values()) # append the range variable values
#out_header = results.keys()


#outstyle = 'multi' # 'single' or 'multi'
#with cd(output_path):
#    post3d.output_csv(out_values,out_header,outstyle,output_fnames)































