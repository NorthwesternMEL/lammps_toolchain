# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 09:37:31 2020

@author: jibri

module: post3d.py

Python 2 post-processing functions for LAMMPS 3-dimensional DEM simulations
Obtain geometry and topology descriptors from LAMMPS fump files
The geometry dump *MUST* be ordered by atom id (dump_modify sort id)

The naming of the variables in the dump columns is overwritten using dictionaries as:
    - Geometry dump:
        x,y,z: particle coordinates
        radius, diameter: radius, diameter of the particle




    - Topology dump:
        id1, id2: the atom ID of particles 1 and 2 in the given contact
        fn: the normal force of the contact
        ft: the tangent force of the contact
        dist: the distance between particles id1 and id2
        lx,ly,lz: the components of the branch vector from particle id2 towards particle id1

The naming of the variables in the log columns has no fixed standard.
It is recommended to use the following (out of habit):
    - epsx,epsy,epsz,epsv: x, y, z and volumetric engineering strain
    - evoid: void ratio
    - pxx,pyy,pzz,pxy,pxz,pyz: xx, yy, zz, xy, xz and yz stress tensor components
    - press,qdev: mean stress and deviatoric stress ( q = sqrt(3*J_2) )

TODO:
    - 
"""

#from dump import dump # The folder of pizza.py sources must be in the pythonpath
import numpy as np
from numpy import linalg as LA


# --------------------------------------------------------------------------- #

def map_dic(dmp,dic):
    """
    Maps the dictionary to the columns of the dump file
    The naming of the variables in the dump columns is overwritten using standardized labels defined here.
    the user MUST use these standardized labels
    - Geometry dump:
        x,y,z: particle coordinates
        radius, diameter: radius, diameter of the particle
        TO BE COMPLETED
 
    - Topology dump:
        id1, id2: the atom ID of particles 1 and 2 in the given contact
        fn: the normal force of the contact
        ft: the tangent force of the contact
        dist: the distance between particles id1 and id2
        lx,ly,lz: the components of the branch vector from particle id2 towards particle id1
    todo:
        - 
        
    INPUT:
    dmp: Dump file [dump class from Pizza]
    dic: Dictionary of column variables (keys) and columns number (value, starting from 1) of the coordinates dump `dmp` [Python Dictionary] 
    
    OUTPUT:
    None
    """
    for i in range(len(dic)):
        dmp.map(dic.values()[i],dic.keys()[i])
    
# --------------------------------------------------------------------------- # 
    
def density(dmpco):
    """
    Computes information related to the packing density of the system
    todo:
        - relative density ? (have emin, emax as inputs?)
        - read walls as optional arguments ?
        
    INPUT:
    dmpco: Dump of the particles coordinates (already mapped using `map_dic`) [dump class from Pizza]
    
    OUTPUT:
    volume (vol): volume of the system (does not include walls)
    packing fraction (pf): packing fraction
    porosity (poro): porosity
    void ratio (evoid): void ratio
    """
    
    Nsnaps = dmpco.nsnaps # Number of snapshot in the dump `dmpco`
    vol = np.empty(Nsnaps) # volume of the periodic cell
    sumd3 = np.empty(Nsnaps) # sum of cubic diameter of the particles
    flag_radius=flag_diameter=False # If diameter or radius is read
    if 'diameter' in dmpco.names.keys():
        flag_diameter=True
    elif 'radius' in dmpco.names.keys():
        flag_radius=True
    else:
        raise KeyError("The dump file and dictionary must define the 'radius' or 'diameter' keys")

    for t in range(Nsnaps):
        vol[t] = (dmpco.snaps[t].xhi - dmpco.snaps[t].xlo) * (dmpco.snaps[t].yhi - dmpco.snaps[t].ylo) * (dmpco.snaps[t].zhi - dmpco.snaps[t].zlo) # Volume of the periodic cell
        if flag_diameter:
            sumd3[t] = np.sum(dmpco.snaps[t].atoms[:,dmpco.names['diameter']]**3)
        elif flag_radius:
            sumd3[t] = np.sum(2.0*dmpco.snaps[t].atoms[:,dmpco.names['radius']]**3)
    pf = np.pi*sumd3/(6*vol) # Packing fraction Vs / Vt
    
    return {'volume': vol,
            'packing fraction': pf,
            'porosity': 1.0-pf,
            'void ratio':1.0/pf-1.0}
    
# --------------------------------------------------------------------------- #
    
def touch(dmptopo,t):
    """
    Determines the pairs in contact at snapshot t from a Dump local file
    This function might be useless if the topology information is exported with radius cutoff instead of neighbor cutoff
    todo:
        - 
        - 
        
    INPUT:
    dmptopo: Dump of the contacts topology [dump class from Pizza]
    t: index of the snapshot to get pairs in contact for 
    
    OUTPUT:
    touchflag: bool array of the contacts actually touching at snapshot t
    ncontact: number of particles actually touching
    """
    
    if 'fn' in dmptopo.names.keys():
        touchflag = dmptopo.snaps[t].atoms[:,dmptopo.names['fn']] > 0.0 # Pairs in strict contact have a strictly positive normal force
    elif all(branch in dmptopo.names.keys() for branch in ['lx','ly','lz']):
        touchflag = np.logical_and.reduce((dmptopo.snaps[t].atoms[:,dmptopo.names['lx']] != 0.0, dmptopo.snaps[t].atoms[:,dmptopo.names['ly']] != 0.0, dmptopo.snaps[t].atoms[:,dmptopo.names['lz']] != 0.0)) # Pairs in strict contact have all branch vector components non-zero
    else:
        raise KeyError("The dump file and dictionary lack keys to determine particles in contact")
    ncontact = np.count_nonzero(touchflag) # Number of contacts
    
    return touchflag,ncontact
    
# --------------------------------------------------------------------------- #
    
def backbone(pair):
    """
    Determines the backbone particles at snapshot t from a Dump local file
    This function is necessary as recursive determination of rattlers is needed
    todo:
        - 
        - 
        
    INPUT:
    pair: array of pairs of particles in contact at snapshot t [numpy (npair x 2)]
    
    OUTPUT:
    flag_pair: array indicating the pairs to be counted
    idp0: id (starting at 1) of the contacting particles at snapshot t
    zcount0: number of contacts of the contacting particles idp0 at snapshot t
    idpbb: id (starting at 1) of the backbone particles at snapshot t
    zcountbb: number of contacts of the backbone particles idpbb at snapshot t
    """
    
    flag_pair = np.full(len(pair),True) # flags backbone in the pair array
    stillrat=True
    idp0,zcount0 = np.unique(pair[flag_pair],return_counts=True)
    while stillrat:
        idp,zcount = np.unique(pair[flag_pair],return_counts=True)
        if np.count_nonzero(zcount<2) == 0:
            stillrat = False
        else:
            flag_pair = np.logical_and(np.in1d(pair[:,0],idp[zcount>=2]), np.in1d(pair[:,1],idp[zcount>=2]))
    
    return flag_pair,idp0,zcount0,idp,zcount

# --------------------------------------------------------------------------- #
    
def coordination(dmptopo,Npart):
    """
    Computes information related to the coordination of the system
    todo:
        - add exception handling for id1, id2
        - Get individual coordination of all particles (in 2D matrix)
        
    INPUT:
    dmptopo: Dump of the contacts topology [dump class from Pizza]
    Npart: number of particles (constant)
    
    OUTPUT:
    rattlers fraction (rat): rattlers proportion (z<2, i.e. including divalent particles)
    mean coordination (zmean): average number of contacts for all particles
    corrected coordination (zstar=zmean/(1-rat)): corrected coordination (includes rattler-backbone contacts)
    backbone coordination (zbb): average number of contact among backbone particles (only backbone-backbone contacts)
    """
    
    Nsnaps = dmptopo.nsnaps # Number of snapshot in the dump `dmptopo`
    rat = np.empty(Nsnaps) # fraction of rattlers
    zmean = np.empty(Nsnaps) # mean coordination
    zbb = np.empty(Nsnaps) # backbone coordination

    for t in range(Nsnaps):
        pair = np.asarray(dmptopo.snaps[t].atoms[:,[dmptopo.names['id1'],dmptopo.names['id2']]],dtype=int) # Pairs within max cutoff
        touchflag = touch(dmptopo,t)[0] # Get particles actually touching
        # Rattlers
        zcount_pair,idp_bb,zcount_bb = backbone(pair[touchflag])[2:5] # Determination of the coordination of all contacting particles, and elimination of the rattlers
        Nbb = len(idp_bb) # Number of backbone particles
        rat[t] = 1.0-float(Nbb)/Npart # Fraction of all particles that are rattlers
        # Coordination number
        zmean[t] = float(np.sum(zcount_pair))/Npart # Coordination number of all contacting particles
        zbb[t] = float(np.sum(zcount_bb))/Nbb # Coordination number of backbone particles (does not account of contact between backbone and rattler particles)
    
    return {'rattlers fraction': rat,
            'mean coordination': zmean,
            'corrected coordination': zmean/(1.0-rat),
            'backbone coordination':zbb}

# --------------------------------------------------------------------------- #
    
def fabric_tensor(dmptopo):
    """
    Computes the fabric tensor of the system
    todo:
        - 
        - 
        
    INPUT:
    dmptopo: Dump of the contacts topology [dump class from Pizza]
    Npart: number of particles (constant)
    
    OUTPUT:
    xx,yy,zz,xy,xz,yz (fabric_tensor[i,j]): components of the 2nd-order symmetric fabric tensor of normal contact directions
    dev (dev): deviatoric eigenvalue of the 2nd-order symmetric fabric tensor of normal contact directions
    """
    
    Nsnaps = dmptopo.nsnaps # Number of snapshot in the dump `dmptopo`
    fabric_tensor = np.zeros((Nsnaps,3,3)) # fabric_tensor
    dev = np.empty(Nsnaps) # mean coordination
    
    if not all(branch in dmptopo.names.keys() for branch in ['lx','ly','lz']):
        raise KeyError("The dump file and dictionary must define all branch vector keys 'lx', 'ly' and 'lz'")

    for t in range(Nsnaps):
        pair = np.asarray(dmptopo.snaps[t].atoms[:,[dmptopo.names['id1'],dmptopo.names['id2']]],dtype=int) # Pairs within max cutoff
        touchflag = touch(dmptopo,t)[0] # Get particles actually touching
        backboneflag = backbone(pair[touchflag])[0] # Determination the backbone pairs among the contacts
        pair = pair[touchflag][backboneflag] # Pair in the backbone
        branch = dmptopo.snaps[t].atoms[:,[dmptopo.names['lx'],dmptopo.names['ly'],dmptopo.names['lz']]][touchflag][backboneflag] # unwrapped branch vector from patom2 towards patom1
        normal = branch / np.sqrt(np.sum(branch**2,axis=1))[:,np.newaxis]
        
        # Calculation of the fabric tensor
        for i in range(len(normal)):
            for j in [0,1,2]:
                for k in [0,1,2]:
                    fabric_tensor[t,j,k] += normal[i][j]*normal[i][k]
        fabric_tensor[t] = fabric_tensor[t] / len(normal)
        w = LA.eig(fabric_tensor[t])[0]
        dev[t] = np.sqrt(0.5*((w[0]-w[1])**2 + (w[0]-w[2])**2 + (w[1]-w[2])**2))
        
    return {'xx': fabric_tensor[:,0,0],
            'yy': fabric_tensor[:,1,1],
            'zz': fabric_tensor[:,2,2],
            'xy': fabric_tensor[:,0,1],
            'xz': fabric_tensor[:,0,2],
            'yz': fabric_tensor[:,1,2],
            'dev': dev}
    
# --------------------------------------------------------------------------- #
    
def stiffness_tensor(dmptopo,dmpco,E,nu):
    """
    Computes the stiffness tensor of the system
    the dumps must be saved at matching timesteps
    todo:
        - check for id1 id2
        - determine the overlap from distance if fn is not saved (i.e. if branch and dist only are saved)
        
    INPUT:
    dmptopo: Dump of the contacts topology [dump class from Pizza]
    dmpco: Ordered dump of the particles coordinates [dump class from Pizza]
    E: Young's modulus of the constituting material of the solid particles [Pa]
    nu: Poisson's ratio of the constituting material of the solid particles [-]

    
    OUTPUT:
    11,12,...,65,66 (Voigt[i,j]): components of the 2nd-order Voigt stiffness tensor
    """
    
    Nsnaps = dmpco.nsnaps # Number of snapshot in the dump `dmpco`
    if Nsnaps != dmptopo.nsnaps:
        raise ValueError("The coordinates and topology dumps must have the same number of snapshots")

    flag_radius=flag_diameter=False # If diameter or radius is read
    if 'diameter' in dmpco.names.keys():
        flag_diameter=True
    elif 'radius' in dmpco.names.keys():
        flag_radius=True
    else:
        raise KeyError("The coordinates dump file and dictionary must define the 'radius' or 'diameter' keys")
    
    flag_fn=flag_dist=False
    if 'fn' in dmptopo.names.keys():
        flag_fn=True
    elif 'dist' in dmptopo.names.keys():
        flag_dist=True
    else:
        raise KeyError("The dump file and dictionary lack keys to determine contact sitffness")
    
    if not all(branch in dmptopo.names.keys() for branch in ['lx','ly','lz']):
        raise KeyError("The dump file and dictionary must define all branch vector keys 'lx', 'ly' and 'lz'")

    vol = np.empty(Nsnaps) # volume of the periodic cell
    stiff_Voigt = np.empty((Nsnaps,6,6)) # Stiffness tensor under Voigt notation
    stiff_4th  = np.empty((Nsnaps,3,3,3,3)) # 4th order Stiffness tensor with 81 entries
    
    for t in range(Nsnaps):
        if dmpco.snaps[t].time != dmptopo.snaps[t].time:
            raise ValueError("The coordinates and topology dumps must have matching snapshots")           
        pair = np.asarray(dmptopo.snaps[t].atoms[:,[dmptopo.names['id1'],dmptopo.names['id2']]],dtype=int) # Pairs within max cutoff
        touchflag = touch(dmptopo,t)[0] # Get particles actually touching
        backboneflag = backbone(pair[touchflag])[0] # Determination the backbone pairs among the contacts
        pair = pair[touchflag][backboneflag] # Pair in the backbone
        branch = dmptopo.snaps[t].atoms[:,[dmptopo.names['lx'],dmptopo.names['ly'],dmptopo.names['lz']]][touchflag][backboneflag] # unwrapped branch vector from patom2 towards patom1
        normal = branch / np.sqrt(np.sum(branch**2,axis=1))[:,np.newaxis] # Normalize branch vector
        tangent1 = np.copy(normal)
        tangent1[:,2] = tangent1[:,0];tangent1[:,0]=-tangent1[:,1];tangent1[:,1]=tangent1[:,2];tangent1[:,2]=0.0 # Create a tangent vector t that is normal to the normal vector n
        tangent1 = tangent1 / np.sqrt(np.sum(tangent1**2,axis=1))[:,np.newaxis] # Normalize the tangent vector t
        tangent2 = np.cross(normal,tangent1) # second tangent vector s
               
        if flag_diameter:
            rad = 0.5*dmpco.snaps[t].atoms[:,dmpco.names['diameter']]
        elif flag_radius:
            rad = dmpco.snaps[t].atoms[:,dmpco.names['radius']]
        Reff = rad[pair[:,dmptopo.names['id1']]-1]*rad[pair[:,dmptopo.names['id2']]-1]/(rad[pair[:,dmptopo.names['id1']]-1]+rad[pair[:,dmptopo.names['id2']]-1])
    
        if flag_fn:
            fn = dmptopo.snaps[t].atoms[:,dmptopo.names['fn']][touchflag][backboneflag] # Normal force between particles
            kn = 1.5**(1.0/3.0)*fn**(1.0/3.0)*(E*np.sqrt(Reff)/(1.0-nu**2))**(2.0/3.0) # Normal contact stiffness
        elif flag_dist:
            dist = dmptopo.snaps[t].atoms[:,dmptopo.names['dist']][touchflag][backboneflag] # distance between particles in contact, to normalize branch vector
            delta = rad[pair[:,dmptopo.names['id1']]-1]+rad[pair[:,dmptopo.names['id2']]-1] - dist # positive overlap between particles in contact
            kn = E*np.sqrt(Reff*delta)/(1.0-nu**2) # Normal contact stiffness
        kt = (2.0-2.0*nu)/(2.0-nu)*kn
        
        vol = (dmpco.snaps[t].xhi - dmpco.snaps[t].xlo) * (dmpco.snaps[t].yhi - dmpco.snaps[t].ylo) * (dmpco.snaps[t].zhi - dmpco.snaps[t].zlo) # Volume of the periodic cell
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        stiff_4th[t,i,j,k,l] = np.sum(kn*normal[:,i]*branch[:,j]*normal[:,k]*branch[:,l] + kt*tangent1[:,i]*branch[:,j]*tangent1[:,k]*branch[:,l] + kt*tangent2[:,i]*branch[:,j]*tangent2[:,k]*branch[:,l]) / vol
        
        # For Voigt notation, we make the assumption that both stress and strains are symmetric.
        # Because the stress is not strictly symmetric, we also take the average of the shear stress e.g. sig_avg_0,1 = sig_avg_1,0 = 1/2 * (sig_0,1 + sig_1,0)
        # As a result, the componenets of the differential operator are summed for, e.g., eps_0,1 and eps_1,0
        # For the Voigt notation, the arrangement of the stress/strain tensor componenet sinto a vector are: 00,11,22,21,20,10

        for i in range(3):
            for j in range(3):
                stiff_Voigt[t,i,j] = stiff_4th[t,i,i,j,j] # First diagonal 3x3 quadrant [0:2][0:2]
                stiff_Voigt[t,i,j+3] = stiff_4th[t,i,i,(j-1)%3,(j+1)%3] + stiff_4th[t,i,i,(j+1)%3,(j-1)%3] # First extra-diagonal 3x3 quadrant [0:2][3:5]
                stiff_Voigt[t,i+3,j+3] = 0.5*((stiff_4th[t,(i-1)%3,(i+1)%3,(j-1)%3,(j+1)%3] + stiff_4th[t,(i-1)%3,(i+1)%3,(j+1)%3,(j-1)%3]) + (stiff_4th[t,(i+1)%3,(i-1)%3,(j-1)%3,(j+1)%3] + stiff_4th[t,(i+1)%3,(i-1)%3,(j+1)%3,(j-1)%3])) # Second diagonal 3x3 quadrant [3:5][3:5]
                stiff_Voigt[t,i+3,j] = 0.5*(stiff_4th[t,(i-1)%3,(i+1)%3,j,j] + stiff_4th[t,(i+1)%3,(i-1)%3,j,j]) # Second extra-diagonal 3x3 quadrant [3:5][0:2]
    
    Voigt = {'11': stiff_Voigt[:,0,0], '12': stiff_Voigt[:,0,1], '13': stiff_Voigt[:,0,2], '14': stiff_Voigt[:,0,3], '15': stiff_Voigt[:,0,4], '16': stiff_Voigt[:,0,5],
             '21': stiff_Voigt[:,1,0], '22': stiff_Voigt[:,1,1], '23': stiff_Voigt[:,1,2], '24': stiff_Voigt[:,1,3], '25': stiff_Voigt[:,1,4], '26': stiff_Voigt[:,1,5],
             '31': stiff_Voigt[:,2,0], '32': stiff_Voigt[:,2,1], '33': stiff_Voigt[:,2,2], '34': stiff_Voigt[:,2,3], '35': stiff_Voigt[:,2,4], '36': stiff_Voigt[:,2,5],
             '41': stiff_Voigt[:,3,0], '42': stiff_Voigt[:,3,1], '43': stiff_Voigt[:,3,2], '44': stiff_Voigt[:,3,3], '45': stiff_Voigt[:,3,4], '46': stiff_Voigt[:,3,5],
             '51': stiff_Voigt[:,4,0], '52': stiff_Voigt[:,4,1], '53': stiff_Voigt[:,4,2], '54': stiff_Voigt[:,4,3], '55': stiff_Voigt[:,4,4], '56': stiff_Voigt[:,4,5],
             '61': stiff_Voigt[:,5,0], '62': stiff_Voigt[:,5,1], '63': stiff_Voigt[:,5,2], '64': stiff_Voigt[:,5,3], '65': stiff_Voigt[:,5,4], '66': stiff_Voigt[:,5,5]}
        
    #return {'Voigt':Voigt} We could use this to return different tensors, each with their own components
    return Voigt

# --------------------------------------------------------------------------- #

def force_distribution(dmptopo,Nsample,fn_range,mob_range,mu):
    """
    Computes the force distributions, normal force and mobilized friction
    todo:
        - check for id1 id2
        - figure out a smart way to get the force distributions without saving the same data multiple times in memory. Is it done? I lost track...

    INPUT:
    dmptopo: Dump of the contacts topology [dump class from Pizza]
    Nsample: number of sampling points (bins) for the distributions
    fn_range: range of the investigated normal force (normalized by its mean), tuple (min(fn/fnmean) , max(fn/fnmean)): bounds [0;Large finite number]
    mob_range: range of the investigated mobilized friction, tuple (min(mob) , max(mob)),: bounds [0;1]
    mu: Coulomb friction limit for the contact between the solid particles [-]


    OUTPUT:
    fn_bins: Nsample+1 values of the bin edges of normal force (normalized by mean value), common to all snapshots
    fn_mean: mean normal force of all backbone contacts, one per snapshot
    fn_cumul: cumulative probability of normal force at the Nsample values of fn_val, one per snapshot
    fn_density: probability density function of normal force at the Nsample values of fn_val, one per snapshot
    mob_bins: Nsample+1 values of the bin edges of normal force (normalized by mean value), common to all snapshots, center of bins (if key 'ft' is defined, otherwise returns None)
    mob_cumul: cumulative probability of mobilized friction (if key 'ft' is defined, otherwise returns None)
    mob_density: probability density function of mobilized friction (if key 'ft' is defined, otherwise returns None)
    """        
        
    if fn_range[1] <= fn_range[0]:
        raise ValueError("Range must be strictly increasing")
    if mob_range[1] <= mob_range[0]:
        raise ValueError("Range must be strictly increasing")
    if 'fn' not in dmptopo.names.keys():
        raise KeyError("The dump file and dictionary must define the normal force key 'fn'")
    
    flag_ft = False
    if 'ft' in dmptopo.names.keys():
        flag_ft = True
    
    Nsnaps = dmptopo.nsnaps # Number of snapshot in the dump `dmptopo`
    fn_mean = np.empty(Nsnaps) # mean normal force for normalization
    fn_cumul = np.zeros((Nsnaps,Nsample+1)) # cumulative distribution of normal forces
    fn_density = np.empty((Nsnaps,Nsample)) # probability density function of normal forces
    
    mob_cumul = np.empty((Nsnaps,Nsample+1)) # cumulative distribution of normal forces
    mob_density = np.empty((Nsnaps,Nsample)) # probability density function of normal forces
    
    fn_bins = np.linspace(fn_range[0], fn_range[1], Nsample+1) # Edges of bins to evaluate distributions
    if flag_ft:
        mob_bins = np.linspace(mob_range[0], mob_range[1], Nsample+1) # Edges of bins to evaluate distributions
    else:
        mob_bins = None
        mob_cumul = None
        mob_density = None

# Use the mean normal force to normalize and set a normalization range

    # Multiple output solution: eventually, these should be users decisions to make and not influence the way this function outputs results
        # - We output the bounds of the interval and note that the values are linearly distributed. Hence can be reconstructed form the files
        # - We output 2 files: one with the densities, one with the values
        # - We output one shared density/cumulative line at the top of the file that is shared by all snaps
        # - We output multiple files with 2 columns of Nsample, 1 file per snapshot
        # - Solutions using distributions for all snapshots can be memory consuming, especially if the values are always the same
    # 2 values per distribution: the value of the variable and the density/cumulative value
    # identical density, should I add one line at the top? like an NaN. This is possibly for cumulative, nor or density
    for t in range(Nsnaps):
        pair = np.asarray(dmptopo.snaps[t].atoms[:,[dmptopo.names['id1'],dmptopo.names['id2']]],dtype=int) # Pairs within max cutoff
        touchflag = touch(dmptopo,t)[0] # Get particles actually touching
        backboneflag = backbone(pair[touchflag])[0] # Determination the backbone pairs among the contacts
        
        fn = dmptopo.snaps[t].atoms[:,dmptopo.names['fn']][touchflag][backboneflag] # Normal force between particles of the backbone
        fn_mean[t] = np.mean(fn) # Mean value of the normal contact force between backbone particles
        fn_norm = fn/fn_mean[t] # normal force normalized by its mean value
        fn_density[t] = np.histogram(fn_norm, bins=fn_bins, density=True)[0] * (fn_range[1]-fn_range[0])/np.amax(fn_norm) # Probability density function of normalized normal force in given range
        fn_cumul[t,1:] = np.cumsum(fn_density[t]) * (fn_bins[1]-fn_bins[0]) # Cumulative distribution function of normal force in given range
        fn_cumul[t] += float(np.count_nonzero(fn_norm < fn_range[0]))/len(fn) # Cumulative distribution function of normal force in given range

        if flag_ft:
            ft = dmptopo.snaps[t].atoms[:,dmptopo.names['ft']][touchflag][backboneflag] # Tangent force between particles of the backbone
            mob = np.minimum(np.maximum(ft/(fn*mu),0.0),1.0) # mobilized friction between particles of the backbone
            mob_density[t] = np.histogram(mob, bins=mob_bins, density=True)[0] *(mob_range[1]-mob_range[0]) # Probability density function of mobilized friction in given range. Maximum range expected to be [0,1]
            mob_cumul[t,1:] = np.cumsum(mob_density[t]) * (mob_bins[1]-mob_bins[0]) # Cumulative distribution function of mobilized friction in given range
            mob_cumul[t] += float(np.count_nonzero(mob < mob_range[0]))/len(mob) # Cumulative distribution function of mobilized friction in given range
    
    return {'fn_bins': fn_bins,
            'fn_mean': fn_mean,
            'fn_cumul': fn_cumul,
            'fn_density': fn_density,
            'mob_bins': mob_bins,
            'mob_cumul': mob_cumul,
            'mob_density':mob_density}
    
# --------------------------------------------------------------------------- #

def read_log(lg,dic):
    """
    Reads and saves the data of a single log file.
    Only the data from the dictionary are saved
    todo:
        - figure out a smart way to get the force distributions without saving the same data multiple times in memory

    INPUT:
    lg: Log file, MUST be single file [log (or log_duplicate) class from Pizza]
    dic: Dictionary of column variables (keys) and columns number (value, starting from 1) of the log file `lg` that are saved [Python Dictionary]



    OUTPUT:
    <key>: values of the corresponding key of the input dictionary
    """


    Nsnaps = lg.nlen # Number of snapshot in the log `lg`
    list_data = [np.array([])]*len(dic) # List of output data

    for t in range(Nsnaps):
        for m in range(len(dic)):
            icol = dic.values()[m]-1
            list_data[m] = np.append(list_data[m],lg.data[t][icol])

    return {dic.keys()[i]:list_data[i] for i in range(len(dic))}

# --------------------------------------------------------------------------- #


def output_csv(data,headers,outstyle,fnames):
    """
    Outputs post-processing data to one single comma-separated file
    todo:
        - 
        - 
        
    INPUT:
    data: 2D numpy arrays of size (Nsnaps,Ncol) data to be output together per column.
    headers: list of headers for each column (list of strings, even if outstyle=='single')
    outstyle: output to a single file or to multiple files (string: 'single' or 'multi')
    fname: list of the names of the output files (list of strings, even if outstyle=='single')
    
    OUTPUT:
    No Python output, creates a file `fname` in the OS
    """
    if len(data.shape) != 2:
        raise ValueError('data array must be 2D with shape (Nsnaps,Ncol)')
    elif len(headers) != len(data[0]):
        raise ValueError('The number of headers and the number of data columns must be equal')
    
    if outstyle == 'single':
        np.savetxt(fnames[0], data, header=",".join(headers), delimiter=',')
    elif outstyle == 'multi' and len(data) == len(fnames):
        for i in range(len(fnames)):
            np.savetxt(fnames[i], data[np.newaxis,i], header=",".join(headers), delimiter=',')
    else:
        raise IndexError("Cannot write multiple output files: the number of files and the number of data rows must be equal")
