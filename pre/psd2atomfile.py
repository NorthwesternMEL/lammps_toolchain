# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 09:54:46 2019

Creation of particle size distribution atomfile for LAMMPS based on actual soils PSD

author: Jibril B. Coulibaly

#
# Copyright (C) 2023 Mechanics and Energy Laboratory, Northwestern University
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
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

# The input PSD in the file might be truncated
# e.g., if large fraction of small grains, or very large grains.
# Once this fraction is discarded, we get a new PSD to fit. To do so, we first
# re-normalize the PSD to span from 0% to 100% passing
psdmassfraction = passing_data[-1] - passing_data[0]
normalized_passing_data = (passing_data - passing_data[0]) / psdmassfraction

Ninterp = 1000 # Linear interpolation of the normalized sieve data
size_interp = np.linspace(size_data[0],size_data[-1],Ninterp)
passing_interp = np.interp(size_interp,size_data,passing_data)
passing_interp = (passing_interp - passing_interp[0])/psdmassfraction
diff_passing = np.diff(passing_interp)
delta_passing_interp = 0.5*(np.insert(diff_passing,0,0.0) + np.append(diff_passing,0.0))
vol_interp = size_interp**3
nparticle_interp = delta_passing_interp/vol_interp
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
plt.semilogx(np.sort(size_DEM)*1000,100*(passing_DEM*psdmassfraction + passing_data[0]),'r',label='de-normalized DEM sample')
plt.title('Particle size distribution of Foundry sand')
plt.xlabel('Particle size [mm]')
plt.ylabel('Percent passing by weight [%]')
plt.legend()
plt.savefig('PSD_Foundy_sieve_DEM.jpg',format='jpg',bbox_inches='tight',dpi=300)

plt.figure(2)
plt.semilogx(size_data,100*normalized_passing_data,'ko',label='Normalized sieve data')
plt.semilogx(np.sort(size_DEM)*1000,100*passing_DEM,'r',label='DEM sample')
plt.title('Particle size distribution of Foundry sand')
plt.xlabel('Particle size [mm]')
plt.ylabel('Percent passing by weight [%]')
plt.legend()
plt.savefig('PSD_Foundry_normalized_sieve_DEM.jpg',format='jpg',bbox_inches='tight',dpi=300)

# ---------------------------#
# WRITING OF LAMMPS ATOMFILE #
# ---------------------------#

file = open("PSD_Foundry.atom",'w')
file.write("# LAMMPS atom file for the PSD of XXX soil. %i particles (upper bound)\n" % Nmax)
file.write("%i\n" % Nmax)
for i in range(Nmax):
    file.write("%i %e\n" % (i+1,size_DEM[i]))
file.close()


