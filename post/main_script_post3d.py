# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 09:37:31 2020

@author: JBC

script: main_script_post3d.py

Python 2 post-processing scripts for LAMMPS 3-dimensional DEM simulations post-processing
The purpose of this script is to get a very clean working directory
Call the functions from a base location
get the input from a target directory
save the output in a target directory
rename conventions etc


The naming of the variables in the dictionaries MUST respect the following standards:
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
import numpy as np
from numpy import linalg as LA

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



"""
#EXAMPLE 1: Obtain geometry and topology information from one sample

input_path = r"C:\Users\jibri\Desktop\test_geo_top"

output_path = input_path+r"/outdir"
if not os.path.exists(output_path):
    try:
        os.makedirs(output_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

# Geometry
input_fnames = ["dump.co.lammpstrj"]
dic = {"id":1, "x":2, "y":3, "z":4, "diameter":5}
with cd(input_path):
    dmp = dump(" ".join(input_fnames)) # Reading of the selected dump files
post3d.map_dic(dmp,dic)

dens = post3d.density(dmp)
Npart = dmp.snaps[0].natoms # Number of particles

output_fnames = ["sample_geom.csv"]
output_values = np.transpose(dens.values())
output_header = dens.keys()
output_style = 'single'
with cd(output_path):
    post3d.output_csv(output_values,output_header,output_style,output_fnames)

# Topology
input_fnames = ["dump.topo.lammpstrj"]
dic = {"id1":1, "id2":2, "dist":3, "fn":4, "ft":5, "lx":6, "ly":7, "lz":8}
with cd(input_path):
    dmp = dump(" ".join(input_fnames)) # Reading of the selected dump files
post3d.map_dic(dmp,dic)

coor = post3d.coordination(dmp,Npart)

output_fnames = ["sample_topo.csv"]
output_values = np.transpose(coor.values())
output_header = coor.keys()
output_style = 'single'
with cd(output_path):
    post3d.output_csv(output_values,output_header,output_style,output_fnames)
"""



"""
EXAMPLE 2: Obtain fabric tensor for many snaps of a simulation

TO BE WRITTEN


out_values = np.transpose(results.values()) # append the range variable values
out_header = results.keys()
#output_fnames =["outfname.csv"] # list of the names of the saved files
output_fnames = ['{0}_out.csv'.format(i) for i in input_fnames]
outstyle = 'multi' # 'single' or 'multi'
with cd(output_path):
    post3d.output_csv(out_values,out_header,outstyle,output_fnames)

"""



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































