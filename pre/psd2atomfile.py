# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 09:54:46 2019

Creation of particle size distribution atomfile for LAMMPS based on actual soils PSD

@author: jibri
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

#use something like random.shuffle(x) to create multiple PSDs

# ----------------------------------#
# ACTUAL PSD OF SOIL, BASED ON MASS #
# ----------------------------------#

fname = "PSD_Fonderie.csv" # Tabulated PSD for actual soil: must be formated as a .csv file with columns 'passing' and 'size'
passing_data = np.genfromtxt(fname, delimiter=',',names=True)['passing'] # Percent pasing in terms of mass or volume
size_data = np.genfromtxt(fname, delimiter=',',names=True)['size'] # Size of the particle in mm

Ninterp = 1000 # Linear interpolation of the sieve data
size_interp = np.linspace(size_data[0],size_data[-1],Ninterp)
passing_interp = np.interp(size_interp,size_data,passing_data)

delta_passing_interp = np.insert(np.diff(passing_interp),0,0)
vol_interp = size_interp**3
nparticle_interp = delta_passing_interp/vol_interp # en Y ?
density_interp = nparticle_interp/np.trapz(nparticle_interp,size_interp)
cumulative_interp = integrate.cumtrapz(density_interp,size_interp,initial=0)

# ------------------------------#
# APPROXIMATE PSD OF DEM SAMPLE #
# ------------------------------#

Nmax = 100000 # Number of particles to sample for
cumulative_DEM = np.random.rand(Nmax) # Random value to pick from the soil cumulative mass PSD
size_DEM = np.interp(cumulative_DEM,cumulative_interp,size_interp)# Perform Inverse transform sampling with linear interpolation on the soil PSD
size_DEM = size_DEM/1000 # Conversion to meters
vol_DEM = np.sort(size_DEM)**3


passing_DEM = np.cumsum(vol_DEM)/np.sum(vol_DEM)
plt.figure()
plt.semilogx(size_data,100*passing_data,'ko',label='Sieve data')
plt.semilogx(np.sort(size_DEM)*1000,100*passing_DEM,'r',label='DEM sample')
plt.title('Particle size distribution of Foundry sand')
plt.xlabel('Particle size [mm]')
plt.ylabel('Percent passing by weight [%]')
plt.legend()
plt.savefig('PSD_Foundy_sieve_DEM.jpg',format='jpg',bbox_inches='tight',dpi=300)

# ---------------------------#
# WRITING OF LAMMPS ATOMFILE #
# ---------------------------#

file = open("PSD_Foundry.atom",'w')
file.write("# LAMMPS atom file for the PSD of XXX soil. %i particles (upper bound)\n" % Nmax)
file.write("%i\n" % Nmax)
for i in range(Nmax):
    file.write("%i %e\n" % (i+1,size_DEM[i]))
file.close()


