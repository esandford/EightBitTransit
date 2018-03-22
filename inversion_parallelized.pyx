# cython: profile=True
from __future__ import division
import numpy as np
cimport numpy as np
import copy
import math
import itertools
import matplotlib.pyplot as plt
import time
import sys
from schwimmbad import MPIPool

from .cTransitingImage import *
from .cGridFunctions import *
from .inversion import *
from .misc import *

__all__ = ['nCr', 'makeArc', 'callMakeArc']

cpdef int nCr(int n, int r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

cpdef makeArc(list arguments):
    """
    the "worker" function for schwimmbad parallelization of makeArcBasis, below
    """
    cdef:
        np.ndarray[np.double_t, ndim=2] SARTimage = arguments[0]
        np.ndarray[np.double_t, ndim=1] times = arguments[1]
        np.ndarray[np.double_t, ndim=3] LCdecrements = arguments[2]
        np.ndarray[np.double_t, ndim=1] obsLC = arguments[3]
        np.ndarray[np.double_t, ndim=1] obsLCerr = arguments[4]
        np.ndarray[np.double_t, ndim=3] areas = arguments[5]
        np.ndarray[np.double_t, ndim=1] delta_fluxes = arguments[6]
        np.ndarray[np.double_t, ndim=2] recombined 
        
        int k = arguments[7], k_idx = arguments[8]
        int N=np.shape(arguments[0])[0], M=np.shape(arguments[0])[1], k_interval, Nmid, nOpacityUnits, nLimbPixelSpaces, nCombinations, comboIdx, p, northern_i, southern_i, ii, jj, kk
        
        double w = arguments[9]
        double t_interval, bestRMS, RMS_

        double[:,:,:] LCdecrements_C = arguments[2]
        double[:] decrements_1D = np.zeros_like(arguments[1])
        double[:] trial_LC = np.ones_like(arguments[1])
        double[:] trial_delta_fluxes = np.zeros((len(arguments[1])-1))


    i_arr = (np.tile(np.arange(N),(M,1))).T
    j_arr = (np.tile(np.arange(N),(M,1)))
    
    onPixelMask = (SARTimage > 0.)
    onPixel_is = i_arr[onPixelMask]
    onPixel_js = j_arr[onPixelMask]
    
    if (N>1) & (N%2 == 0): #N even
        Nmid = int(N/2) 
    elif (N>1) & (N%2 != 0): #N odd
        Nmid = int((N-1)/2 + 1)
    
    recombined = np.zeros_like(SARTimage)

    #get indices, xy positions, and angular positions of "on" pixels that overlap the stellar limb
    limbPixelMask = ((areas[k] > 0.) & (areas[k] < (w**2)/np.pi))

    limbPixel_is = i_arr[limbPixelMask & onPixelMask]
    limbPixel_js = j_arr[limbPixelMask & onPixelMask]

    limbPixel_is_half = limbPixel_is[limbPixel_is < Nmid]
    limbPixel_js_half = limbPixel_js[limbPixel_is < Nmid]
        
    #if there are limb pixels and dF/dt > 0.5 pixels' worth of opacity:
    if (len(limbPixel_is_half) > 0) & (np.abs(delta_fluxes[k_idx]) > ((ti.w)**2/(2.*np.pi))):
        #print np.abs(delta_fluxes[k_idx])
            
        nOpacityUnits = int(np.ceil(np.abs(delta_fluxes[k_idx])/np.mean(ti.areas[k][limbPixelMask & onPixelMask]))) #number of "units" of 0.5 opacity that 
                                                                                           #need to be distributed among the limb pixels. 
        print "nOpacityUnits is {0}".format(nOpacityUnits)
        nLimbPixelSpaces = len(limbPixel_is_half)*2 #available spaces that can hold a "unit" of 0.5 opacity
        print "nLimbPixelSpaces is {0}".format(nLimbPixelSpaces)
            
            
        nCombinations = nCr(nLimbPixelSpaces, nOpacityUnits)
        if nCombinations == 0:
            nCombinations = int(1e6)
            
        print "nCombinations is {0}".format(nCombinations)
            
        combinations = itertools.combinations(iterable = np.arange(nLimbPixelSpaces), r = nOpacityUnits)

        bestRMS = 1000000.0
        best_whichOn = np.zeros((len(limbPixel_is_half)),dtype=int)
            
        #if there are too many combinations, just take a random subset
        if nCombinations > 1e5:
            combinations = []

            for nc in range(int(1e5)):
                combo = tuple(np.sort(np.random.choice(np.arange(nLimbPixelSpaces), size=nOpacityUnits, replace=False)))
                combinations.append(combo)

        for comboIdx, combo in enumerate(combinations):
            t0 = time.time()
            grid = np.zeros_like(SARTimage)
                
            limbPixels_to_p05 = np.array(combo) % len(limbPixel_is_half)
                
            for p in limbPixels_to_p05:
                northern_i = limbPixel_is_half[p]
                southern_i = N - limbPixel_is_half[p] - 1
                grid[northern_i,limbPixel_js_half[p]] += 0.5
                grid[southern_i,limbPixel_js_half[p]] += 0.5
                
            foldedGrid = foldOpacities(grid)

            for ii in range(0, N):
                for jj in range(0, M):
                    if foldedGrid[ii,jj]==1.:
                        for kk in range(0, len(times)):
                            trial_LC[kk] -= LCdecrements_C[ii][jj][kk]

            for kk in range(0, len(times)-1):
                trial_delta_fluxes[kk] = trial_LC[kk+1]-trial_LC[kk]
            
            RMS_ = ((delta_fluxes[k_idx] - trial_delta_fluxes[k_idx])**2/obsLCerr[k]**2)
                
            if RMS_ < bestRMS:
                bestRMS = RMS_
                best_whichOn = limbPixels_to_p05

            t1 = time.time()
            #print "1 combination test: {0} seconds".format(t1-t0)

        for p in best_whichOn:
            northern_i = limbPixel_is_half[p]
            southern_i = N - limbPixel_is_half[p] - 1
            recombined[northern_i,limbPixel_js_half[p]] += 0.5
            recombined[southern_i,limbPixel_js_half[p]] += 0.5
            
    
    foldedGrid = foldOpacities(recombined)
    
    for ii in range(0, N):
        for jj in range(0, M):
            if foldedGrid[ii,jj]==1.:
                for kk in range(0, len(times)):
                    trial_LC[kk] -= LCdecrements_C[ii][jj][kk]

    for kk in range(0, len(times)-1):
        trial_delta_fluxes[kk] = trial_LC[kk+1]-trial_LC[kk]

    return (np.ravel(recombined), RMS(obsLC,obsLCerr,trial_LC))


cpdef callMakeArc(np.ndarray[double, ndim=2] SARTimage, np.ndarray[double, ndim=1] times, 
    np.ndarray[double, ndim=3] LCdecrements, np.ndarray[double, ndim=1] obsLC, 
    np.ndarray[double, ndim=1] obsLCerr):
    """
    Do genetic recombination *along arcs*
    """
    cdef:
        np.ndarray[np.int64_t, ndim=2] i_arr
        np.ndarray[np.int64_t, ndim=2] j_arr
        np.ndarray[np.int64_t, ndim=1] ks
        np.ndarray[np.int64_t, ndim=1] onPixel_is
        np.ndarray[np.int64_t, ndim=1] onPixel_js
        np.ndarray[np.int64_t, ndim=1] limbPixel_is
        np.ndarray[np.int64_t, ndim=1] limbPixel_js
        np.ndarray[np.int64_t, ndim=1] limbPixel_is_half
        np.ndarray[np.int64_t, ndim=1] limbPixel_js_half
        np.ndarray[np.double_t, ndim=1] time_points
        np.ndarray[np.double_t, ndim=1] delta_times
        np.ndarray[np.double_t, ndim=1] middle_times
        np.ndarray[np.double_t, ndim=1] flux_points
        np.ndarray[np.double_t, ndim=1] delta_fluxes
        np.ndarray[np.double_t, ndim=2] basis
        np.ndarray[np.double_t, ndim=1] basisRMSs

        int N=np.shape(SARTimage)[0], M=np.shape(SARTimage)[1], k=len(times), k_interval, Nmid, k_idx, nOpacityUnits, nLimbPixelSpaces, nCombinations, comboIdx, p, northern_i, southern_i, ii, jj, kk
        
        double t_interval, bestRMS, RMS_

        double[:,:,:] LCdecrements_C = LCdecrements
        double[:] decrements_1D = np.zeros_like(times)
        double[:] trial_LC = np.ones_like(times)
        double[:] trial_delta_fluxes = np.zeros((len(times)-1))
   
    
    #print type(LCdecrements)    #np.ndarray
    #print type(LCdecrements_C)  #EightBitTransit.inversion._memoryviewslice
    N = np.shape(SARTimage)[0]
    M = np.shape(SARTimage)[1]

    ti = TransitingImage(opacitymat=np.zeros((N,M)), LDlaw="uniform", v=0.4, t_ref=0., t_arr=times)
    ti.gen_LC(times)
    
    #How long does it take for the grid to move laterally by a distance of w/2 (i.e., 1/2 pixel width)?
    t_interval = (ti.w)/(2.*ti.v)
    
    #What indices in the time array correspond to these time intervals?
    k_interval = int(t_interval/np.mean(times[1:] - times[0:-1]))
    
    i_arr = (np.tile(np.arange(N),(M,1))).T
    j_arr = (np.tile(np.arange(N),(M,1)))
    
    onPixelMask = (SARTimage > 0.)
    onPixel_is = i_arr[onPixelMask]
    onPixel_js = j_arr[onPixelMask]
    
    if (N>1) & (N%2 == 0): #N even
        Nmid = int(N/2) 
    elif (N>1) & (N%2 != 0): #N odd
        Nmid = int((N-1)/2 + 1)
    
    #Get delta(light curve), calculated between time points (2*t_interval) apart (i.e., when the grid has moved a distance of w)
    ks = np.arange(0,np.shape(ti.areas)[0],k_interval)
    
    time_points = np.zeros((len(ks)+1))
    time_points[0:-1] = times[np.arange(0,len(times),k_interval)]
    time_points[-1] = times[-1]
    delta_times = time_points[1:] - time_points[0:-1]
    middle_times = time_points[0:-1] + (delta_times/2.)
    
    flux_points = np.zeros((len(ks)+1))
    flux_points[0:-1] = obsLC[np.arange(0,len(obsLC),k_interval)]
    flux_points[-1] = obsLC[-1]
    
    delta_fluxes = flux_points[1:] - flux_points[0:-1]
    
    pool = MPIPool()

    if not pool.is_master():
        pool.wait()
        sys.exit(0)


    tasks = []
    for k_idx, k in enumerate(ks):
        tasks.append([SARTimage, times, LCdecrements, obsLC, obsLCerr, areas, delta_fluxes, k, k_idx, w])

    results = pool.map(makeArc, tasks)
    
    return results
