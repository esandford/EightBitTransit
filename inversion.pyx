# cython: profile=True
from __future__ import division
import numpy as np
cimport numpy as np
import copy
import math
import itertools
import matplotlib.pyplot as plt
import time
#from libcpp cimport bool as cybool

from .cTransitingImage import *
from .cGridFunctions import *
from .misc import *

__all__ = ['nCr', 'makeArcBasis', 'whoAreMyArcNeighbors','arcRearrange','discreteFourierTransform_2D','inverseDiscreteFourierTransform_2D','Gaussian2D_PDF','simultaneous_ART', 'wedgeRearrange','wedgeNegativeEdge', 'wedgeOptimize_sym',
'foldOpacities','round_ART','invertLC']
#define 2D discrete fourier transform and 2D inverse discrete fourier transform

cpdef int nCr(int n, int r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

cpdef makeArcBasis(np.ndarray[double, ndim=2] SARTimage, np.ndarray[double, ndim=1] times, 
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
        np.ndarray[np.int64_t, ndim=1] best_whichOn
        np.ndarray[np.double_t, ndim=2] recombined
        np.ndarray[np.double_t, ndim=2] grid
        np.ndarray[np.int64_t, ndim=1] limbPixels_to_p05
        np.ndarray[np.double_t, ndim=2] foldedGrid
        np.ndarray[np.double_t, ndim=2] decrements
        #np.ndarray[np.double_t, ndim=1] decrements_1D
        #np.ndarray[np.double_t, ndim=1] trial_LC
        #np.ndarray[np.double_t, ndim=1] trial_flux_points
        #np.ndarray[np.double_t, ndim=1] trial_delta_fluxes

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
        
    #fig = plt.figure(figsize=(8,6))
    #plt.plot(times,obsLC,'m-',lw=2)
    #plt.plot(time_points,flux_points,'ko',ls='-',mec='None',lw=1)
    #plt.bar(x=middle_times,height=delta_fluxes,width=0.2,bottom=1.,color='b',edgecolor='None',alpha=0.5)
    #plt.ylim(0.8,1.06)
    #plt.show()
    
    basis = np.zeros((len(ks),N*M))
    basisRMSs = np.zeros((len(ks)))

    for k_idx,k in enumerate(ks): #at every interval during which the grid moves a distance of w
        
        #start from an empty grid at every time step
        recombined = np.zeros_like(SARTimage)

        print k

        #get indices, xy positions, and angular positions of "on" pixels that overlap the stellar limb
        limbPixelMask = ((ti.areas[k] > 0.) & (ti.areas[k] < ((ti.w)**2)/np.pi))

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
                            for kk in range(0, k):
                                trial_LC[kk] -= LCdecrements_C[ii][jj][kk]

                for kk in range(0, k-1):
                    trial_delta_fluxes[kk] = trial_LC[kk+1]-trial_LC[kk]
                #decrements = LCdecrements_C[foldedGrid.astype(bool)]
                #decrements_1D = np.sum(decrements,axis=0)

                #trial_LC = ones_decrements_1D - decrements_1D

                #trial_flux_points = np.zeros((len(ks)+1))
                #trial_flux_points[0:-1] = trial_LC[np.arange(0,len(trial_LC),k_interval)]
                #trial_flux_points[-1] = trial_LC[-1]

                #trial_delta_fluxes = trial_flux_points[1:] - trial_flux_points[0:-1]

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
            
        #plot it
        foldedGrid = foldOpacities(recombined)
        #decrements = LCdecrements[foldedGrid.astype(bool)]
        #decrements_1D = np.sum(decrements,axis=0)
        #trial_LC = np.ones_like(decrements_1D) - decrements_1D
        #trial_flux_points = np.zeros((len(ks)+1))
        #trial_flux_points[0:-1] = trial_LC[np.arange(0,len(trial_LC),k_interval)]
        #trial_flux_points[-1] = trial_LC[-1]
        #trial_delta_fluxes = trial_flux_points[1:] - trial_flux_points[0:-1]
        
        for ii in range(0, N):
            for jj in range(0, M):
                if foldedGrid[ii,jj]==1.:
                    for kk in range(0, k):
                        trial_LC[kk] -= LCdecrements_C[ii][jj][kk]

        for kk in range(0, k-1):
            trial_delta_fluxes[kk] = trial_LC[kk+1]-trial_LC[kk]

        #fig,axes = plt.subplots(1,2,figsize=(12,6))
        #axes[0].imshow((recombined+recombined[::-1,:])/2.,cmap='bwr_r',interpolation='none',vmin=-1.,vmax=1.)
        #axes[1].axvline(middle_times[k_idx],color='k',ls='--',alpha=0.5)
        #axes[1].plot(times,obsLC,'m-',lw=2)
        #axes[1].plot(time_points,trial_flux_points,'ko',ls='-',mec='None',lw=1)
        #axes[1].bar(x=middle_times,height=delta_fluxes,width=0.4,bottom=1.,color='b',edgecolor='None',alpha=0.8)
        #axes[1].bar(x=middle_times,height=trial_delta_fluxes,width=0.2,bottom=1.,color='r',edgecolor='None',alpha=0.7)
        #axes[1].set_ylim(0.8,1.06)
        #plt.show()

        basis[k_idx] = np.ravel(recombined)
        basisRMSs[k_idx] = RMS(obsLC,obsLCerr,trial_LC) #np.min(costs)
        
    return basis, basisRMSs

def whoAreMyArcNeighbors(N,M,i,j):
    """
    Find the other pixels that also belong to the double-arc pattern centered at pixel i,j of grid of size N,M.
    """
    leftArcGrid = np.zeros((N,M))
    rightArcGrid = np.zeros((N,M))
    arcGrid_positions = positions(N, M, np.array((-0.1,0.,0.1)), tref=0., v=0.05)[1]
    #print np.shape(arcGrid_positions)
    r = 1. #radius of arcs, because star is radius unity
    
    w = 2./N
    
    x_int = arcGrid_positions[i,j,0] #x-coordinate of intersection point of the two arcs
    y_int = arcGrid_positions[i,j,1] #y-coordinate of intersection point of the two arcs
    
    #print x_int, y_int
    
    xc1 = x_int - np.sqrt((r**2 - y_int**2))
    xc2 = x_int + np.sqrt((r**2 - y_int**2))
    
    #equation of left-opening arc: (x - xc1)**2 + y**2 = r**2; x > xc1
    #equation of right-opening arc: (x - xc2)**2 + y**2 = r**2; x < xc2
    
    #(if xc1 > -1, add negative opacity along this arc: (x - xc1)**2 + y**2 = r**2; x < xc1 ?)
    #(if xc2 < 1, add negative opacity along this arc: (x - xc2)**2 + y**2 = r**2; x > xc2 ?)
    
    #print xc1, xc2
    
    leftArc_xs = xc1 + np.sqrt((r**2 - arcGrid_positions[:,:,1]**2))
    leftArc_lb = leftArc_xs - (w/2.)*np.sqrt(2.)
    leftArc_ub = leftArc_xs + (w/2.)*np.sqrt(2.)
    
    leftArc_mask = (arcGrid_positions[:,:,0] >= leftArc_lb) & (arcGrid_positions[:,:,0] < leftArc_ub) 
    
    
    rightArc_xs = xc2 - np.sqrt((r**2 - arcGrid_positions[:,:,1]**2))
    rightArc_lb = rightArc_xs - (w/2.)*np.sqrt(2.)
    rightArc_ub = rightArc_xs + (w/2.)*np.sqrt(2.)
    
    rightArc_mask = (arcGrid_positions[:,:,0] >= rightArc_lb) & (arcGrid_positions[:,:,0] < rightArc_ub) 
    
    leftArcGrid[leftArc_mask] = 1.
    rightArcGrid[rightArc_mask] = 1.
    
    return leftArcGrid.astype(bool), rightArcGrid.astype(bool)

def arcRearrange(SART, times):
    """
    Rearrange opacity to remove unphysical pixels in the raw SART solution.
    
    We only want to deal with  pixels which are < 0. or > 1. 
    """
    
    arcRearranged = copy.deepcopy(SART)
    
    ti = TransitingImage(opacitymat=SART, LDlaw="uniform", v=0.4, t_ref=0., t_arr=times)
    ti.gen_LC(times)
    
    N = np.shape(SART)[0]
    M = np.shape(SART)[1]

    i_arr = (np.tile(np.arange(N),(M,1))).T
    j_arr = (np.tile(np.arange(N),(M,1)))
    
    negativePixelMask = (SART < 0.)
    negativePixel_is = i_arr[negativePixelMask]
    negativePixel_js = j_arr[negativePixelMask]
    #print len(negativePixel_is)
    
    for n, op in enumerate(SART[negativePixelMask]):
        i = negativePixel_is[n]
        j = negativePixel_js[n]
        
        #print i,j
        leftArcNeighborMask, rightArcNeighborMask = whoAreMyArcNeighbors(N,M,i,j)
        #plt.imshow(arcNeighborMask.astype(int),cmap='Blues',interpolation='None',vmin=0.,vmax=1.,)
        #plt.show()
        
        leftArcOpacity = np.sum(SART[leftArcNeighborMask])
        rightArcOpacity = np.sum(SART[rightArcNeighborMask])
        
        #set unphysical arcs at the edges equal to 0.
        if (leftArcOpacity < 0.) & (j < M/2.):
            arcRearranged[leftArcNeighborMask] = 0.
        if (rightArcOpacity < 0.) & (j > M/2.):
            arcRearranged[rightArcNeighborMask] = 0.
            
    negativePixelMask = (arcRearranged < 0.)
    negativePixel_is = i_arr[negativePixelMask]
    negativePixel_js = j_arr[negativePixelMask]
    
    for n, op in enumerate(SART[negativePixelMask]):
        i = negativePixel_is[n]
        j = negativePixel_js[n]
        
        #print i,j
        leftArcNeighborMask, rightArcNeighborMask = whoAreMyArcNeighbors(N,M,i,j)
        #plt.imshow(arcNeighborMask.astype(int),cmap='Blues',interpolation='None',vmin=0.,vmax=1.,)
        #plt.show()
        
        leftArcOpacity = np.sum(SART[leftArcNeighborMask])
        rightArcOpacity = np.sum(SART[rightArcNeighborMask])
        
        arcNeighborMask = (leftArcNeighborMask | rightArcNeighborMask)
        
        arcRearranged[i,j] = 0.
        arcRearranged[arcNeighborMask] += op/len(arcRearranged[arcNeighborMask])
        
        
    tooBigPixelMask = (SART > 1.)
    tooBigPixel_is = i_arr[tooBigPixelMask]
    tooBigPixel_js = j_arr[tooBigPixelMask]
    
    for n, op in enumerate(SART[tooBigPixelMask]):
        i = tooBigPixel_is[n]
        j = tooBigPixel_js[n]
        
        leftArcNeighborMask, rightArcNeighborMask = whoAreMyArcNeighbors(N,M,i,j)

        leftArcOpacity = np.sum(SART[leftArcNeighborMask])
        rightArcOpacity = np.sum(SART[rightArcNeighborMask])
        
        arcNeighborMask = (leftArcNeighborMask | rightArcNeighborMask)
        
        arcRearranged[i,j] = 1.
        arcRearranged[arcNeighborMask] += (op - 1.)/len(arcRearranged[arcNeighborMask])
    
    
    arcRearranged = (arcRearranged + arcRearranged[::-1,:])/2.
    
    return arcRearranged

def discreteFourierTransform_2D(matrix):
    """
    Return the 2D discrete Fourier transform of a matrix.
    """
    return np.fft.fft2(matrix)

def inverseDiscreteFourierTransform_2D(matrix):
    """
    Return the inverse discrete Fourier transform of matrix in frequency space ("matrix").
    """
    return np.fft.ifft2(matrix)

def Gaussian2D_PDF(xVec, muVec, sigmaMatrix):
    """
    Return the value of the PDF of the multivariate normal at the position vector xVec.
    """
    sigmaDet = np.linalg.det(sigmaMatrix)
    sigmaInv = np.linalg.inv(sigmaMatrix)
    
    return np.exp(-0.5 * np.dot((xVec-muVec).T, np.dot(sigmaInv, (xVec - muVec))))/np.sqrt((2.*np.pi)**2 * sigmaDet)

cpdef np.ndarray simultaneous_ART(int n_iter, np.ndarray[double, ndim=2] tau_init, 
    np.ndarray[double, ndim=2] A, np.ndarray[double, ndim=1] obsLC, np.ndarray[double, ndim=1] obsLCerr, 
    double reg, bint threshold, str filename, str window):
    """
    Use the algebraic reconstruction technique to solve the system A*tau = np.ones_like(obsLC) - obsLC.
    
    Inputs:
    n_iter = number of iterations 
    tau_init = initial guess for the raveled opacity vector tau (shape = NM = k)
    A = matrix of pixel overlap areas (shape = (NM, NM) = (k, k))
    obsLC = observed normalized flux values in the light curve (shape = NM = k)
    
    Outputs:
    tau = opacity vector
    """
    cdef:
        np.ndarray[np.double_t, ndim=1] tau=np.ravel(tau_init) #tau_init = (Nhalf, M)
        np.ndarray[np.double_t, ndim=1] RHS
        np.ndarray[np.double_t, ndim=1] tau_update=np.zeros_like(tau, dtype=float)
        np.ndarray[np.double_t, ndim=2] testtau=np.zeros_like(tau_init)
        np.ndarray[np.double_t, ndim=2] testLC=np.zeros_like(np.atleast_2d(obsLC))
        np.ndarray[np.double_t, ndim=1] testLC1d=np.zeros_like(obsLC)
        np.ndarray[np.double_t, ndim=2] Asquare=np.zeros((len(tau),len(tau)))
        np.ndarray[np.double_t, ndim=1] origRHS=np.zeros(len(obsLC))
        np.ndarray[np.double_t, ndim=1] windowarr=np.ones_like(tau, dtype=float)

        np.ndarray[np.double_t, ndim=2] windowarr2D=np.zeros((np.shape(tau_init)[0]*4 - 1, np.shape(tau_init)[1]*2 - 1), dtype=float)
        np.ndarray[np.double_t, ndim=2] SART_zeropadded=np.zeros((np.shape(tau_init)[0]*4 - 1, np.shape(tau_init)[1]*2 - 1), dtype=float)
        np.ndarray[np.double_t, ndim=2] truth_zeropadded=np.zeros((np.shape(tau_init)[0]*4 - 1, np.shape(tau_init)[1]*2 - 1), dtype=float)
        
        int q, N, M, tau_entry, entry_idx
        double outer_numerator, outer_denominator, inner_numerator, inner_denominator, testRMS
        list RMSs, taus, tau_updates

    #tau = np.ravel(tau_init)
    N = np.shape(tau_init)[0]
    M = np.shape(tau_init)[1]
    #RHS = np.ones_like(obsLC) - obsLC

    if ((np.shape(A)[0] == np.shape(A)[1]) & (reg==0.)):
        RHS = np.ones_like(obsLC) - obsLC
        Asquare = A
        
    else:
        origRHS = np.ones_like(obsLC) - obsLC
        RHS = np.dot(A.T, np.ones_like(obsLC) - obsLC)
        Asquare = np.dot(A.T, A) + (reg * np.eye(N=np.shape(A)[1]))

    RMSs = []
    taus = []
    taus.append(np.zeros_like(tau))
    taus.append(tau)
    tau_updates = []
    tau_updates.append(np.zeros_like(tau))

    if window=="none" or window=="None" or window is None:
        pass 
    elif window=="hann":
        for n in range(0,len(windowarr)):
            windowarr[n] = 0.5*(1.-np.cos((2.*np.pi*n)/(len(windowarr)-1)))
    elif window=="hamming":
        for n in range(0,len(windowarr)):
            windowarr[n] = 0.54 - 0.46*np.cos((2.*np.pi*n)/(len(windowarr)-1))
    
    elif window=="2DGaussian":
        jmid = (M-1)/2.
        w = 2./(2*N)
        for i in range(0,N*4-1):
            for j in range(0,M*2-1):
                x = (j-jmid)*w
                y = 1. - (w/2.) - i*w
                windowarr2D[i,j] = 0.025*Gaussian2D_PDF(np.array((x,y)), np.array((0.,0.)), 0.1*np.eye(2))
    
    for q in range(0, n_iter):
        tau_update = np.zeros_like(tau, dtype=float)
        
        for j in range(0, np.shape(Asquare)[0]):
            outer_numerator = 0. 
            outer_denominator = np.sum(Asquare[:,j])

            for i in range(0, np.shape(Asquare)[0]):
                inner_denominator = np.sum(Asquare[i])
                inner_numerator = (RHS[i] - np.dot(Asquare[i], tau)) * Asquare[i,j] * windowarr[i]
                outer_numerator = outer_numerator + (inner_numerator/inner_denominator)
            
            tau_update[j] = (outer_numerator/outer_denominator)
            
        tau = tau + tau_update
        
        if window == '2DGaussian':
            Nquarter = 4*N / 4
            Mquarter = 2*M / 4
            SART_zeropadded[Nquarter:Nquarter+N,Mquarter:Mquarter+M] = tau.reshape(N,M)
            SART_zeropadded[Nquarter+N:Nquarter+(2*N),Mquarter:Mquarter+M] = tau.reshape(N,M)[::-1,:]
            truth_zeropadded = np.fft.fftshift(inverseDiscreteFourierTransform_2D(discreteFourierTransform_2D(SART_zeropadded)/discreteFourierTransform_2D(windowarr2D))).real
            tau = np.ravel(truth_zeropadded[Nquarter:Nquarter+(N*2),Mquarter:Mquarter+M][0:N,:]) #cut in half again, then ravel

        if threshold is True:
            for entry_idx in range(0,len(tau)):
                tau[entry_idx] = np.max((np.min((tau[entry_idx], 1.)),0.))
        
        else:
            #testtau = np.round(wedgeRearrange(np.round(wedgeRearrange(np.round(wedgeRearrange(np.reshape(tau,(N,M))),2)),2)),2)
            #testtau = np.round(wedgeRearrange(wedgeOptimize_sym(wedgeOptimize_sym(wedgeOptimize_sym(testtau, obsLC=obsLC, obsLCerr=obsLCerr, areas=A), obsLC=obsLC, obsLCerr=obsLCerr, areas=A), obsLC=obsLC, obsLCerr=obsLCerr, areas=A)),2)
            #testtau = np.round(wedgeNegativeEdge(testtau),2)
            #testtau = np.round(wedgeRearrange(wedgeOptimize_sym(wedgeOptimize_sym(wedgeOptimize_sym(testtau, obsLC=obsLC, obsLCerr=obsLCerr, areas=A), obsLC=obsLC, obsLCerr=obsLCerr, areas=A), obsLC=obsLC, obsLCerr=obsLCerr, areas=A)),2)
            testtau = np.reshape(tau, (N,M))

        testLC = np.atleast_2d(np.ones_like(origRHS)).T - np.dot(A,np.reshape(testtau,(N*M,1)))
        testLC1d = testLC[:,0]
        testRMS = RMS(LC_obs=obsLC,LC_obs_err=obsLCerr,LC_model=testLC1d)
        RMSs.append(testRMS)
        taus.append(tau)
        tau_updates.append(tau_update)
    
    fig = plt.figure(figsize=(8,6))
    plt.plot(RMSs)
    plt.yscale('log')
    plt.xscale('log')
    plt.title("RMS",fontsize=14)
    #plt.show()
    plt.savefig("{0}_chiSquareds.png".format(filename),fmt="png")

    taus_arr = np.array(taus)
    tau_updates_arr = np.array(tau_updates)

    np.savetxt("{0}_chiSquareds.txt".format(filename), np.array(RMSs))
    np.savetxt("{0}_taus.txt".format(filename), taus)
    np.savetxt("{0}_tauUpdates.txt".format(filename), tau_updates)

    fig = plt.figure(figsize=(8,6))
    for tau_entry in range(0,np.shape(tau_updates_arr)[1]):
        plt.plot(np.abs(tau_updates_arr[:,tau_entry]),alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0,np.shape(tau_updates_arr)[0])
    plt.ylim(1.e-10,1.e0)
    plt.title("Tau updates", fontsize=14)
    #plt.show()
    plt.savefig("{0}_tauUpdates.png".format(filename),fmt="png")

    fig = plt.figure(figsize=(8,6))
    for tau_entry in range(0,np.shape(taus_arr)[1]):
        plt.plot(taus_arr[:,tau_entry],alpha=0.7)
    plt.xscale('log')
    plt.xlim(0,np.shape(tau_updates_arr)[0])
    plt.ylim(-0.1,1.1)
    plt.title("Taus", fontsize=14)
    #plt.show()
    plt.savefig("{0}_taus.png".format(filename),fmt="png")

    return tau

def wedgeRearrange(tau):
    """
    Exploit the "wedge degeneracy" to shift opacity around and prevent unphysical opacities.
    
    Strategy: Move too-high & too-low opacities out, so they can be distributed across wider pixel blocks
    (When you reach the edges of the grid, turn around and push remaining unphysical opacities in?? Or else just
    round to 1s, 0s.)
    """
    newtau = copy.copy(tau)
    
    # Start at the middle of the grid
    N = np.shape(tau)[0]
    M = np.shape(tau)[1]
    middleN = int(np.floor((N-1)/2.))
    
    w = 2./N
    #N even
    if N%2 == 0:
        #print "even"
        northRows = np.arange(middleN, -1, -1)
        southRows = np.arange(N-1-middleN, N, 1)
        b = w/2.
        nextRow_b = (3.*w)/2.
        
    #N odd
    else:
        #print "odd"
        northRows = np.arange(middleN-1, -1, -1)
        southRows = np.arange(N-middleN, N, 1)
        
        #impact parameter of middle pixel row is 0. for an odd-N grid
        middleRow = tau[middleN]
        middleRow_unphys = np.arange(0,M)[(middleRow > 1.0) | (middleRow < 0.0)]
        
        #propagate unphysical opacities out to neighboring rows
        b = 0.
        nextRow_b = w
        #width of pixel block with same transit duration [units of PIXELS]
        sameDuration = (w + 2.*np.sqrt(1.-b**2) - 2.*np.sqrt(1.-nextRow_b**2)) / w
        
        sameDuration_int = 0
        sameDuration_leftover = sameDuration
        while sameDuration_leftover > 1.:
            sameDuration_int += 1
            sameDuration_leftover -= 1

        if sameDuration_int%2 == 0:
            sameDuration_int = sameDuration_int - 1
            sameDuration_leftover = sameDuration_leftover + 1.
        
        for j in middleRow_unphys:
            #get spillover column idxs
            spillover_j = np.arange(j-(int(np.floor(sameDuration_int/2))), j+(int(np.floor(sameDuration_int/2))) + 1)
            spillover_j = spillover_j[(spillover_j >= 0.) &  (spillover_j < M)]
            
            #let unphysical opacities overflow, where the distribution of overflows is proportional to
            # the pixel's "contribution" to the transit duration
            
            if middleRow[j] > 1.:
                amtOverflow = middleRow[j] - 1.
                newtau[middleN, j] = 1.
            elif middleRow[j] < 0.:
                amtOverflow = middleRow[j]
                newtau[middleN, j] = 0.
                
            directOverflowWeight = (1./sameDuration)
            edgeOverflowWeight = (sameDuration_leftover/2.)
            
            for col in spillover_j:
                newtau[middleN+1,col] += (directOverflowWeight*amtOverflow)/2. #divide by 2 because middle row overflows both north and south
                newtau[middleN-1,col] += (directOverflowWeight*amtOverflow)/2.
            
            leftCol = j - int(np.floor(sameDuration_int/2)) - 1
            rightCol = j + int(np.floor(sameDuration_int/2)) + 1
                    
            while leftCol < 0:
                leftCol = leftCol + 1
            while rightCol > M-1:
                rightCol = rightCol - 1
                
            newtau[middleN+1, leftCol] += (edgeOverflowWeight*amtOverflow)/2.
            newtau[middleN+1, rightCol] += (edgeOverflowWeight*amtOverflow)/2.
            
            newtau[middleN-1, leftCol] += (edgeOverflowWeight*amtOverflow)/2.
            newtau[middleN-1, rightCol] += (edgeOverflowWeight*amtOverflow)/2.
            
            b = w
            nextRow_b = 2.*w
            
    
    for row in northRows[:-1]:
        northRow = tau[row]
        northRow_unphys = np.arange(0,M)[(northRow > 1.0) | (northRow < 0.0)]
        
        southRow = tau[N-1-row]
        southRow_unphys = np.arange(0,M)[(southRow > 1.0) | (southRow < 0.0)]
        
        #propagate unphysical opacities out to neighboring rows
        #width of pixel block with same transit duration [units of PIXELS]
        sameDuration = (w + 2.*np.sqrt(1.-b**2) - 2.*np.sqrt(1.-nextRow_b**2)) / w
        
        sameDuration_int = 0
        sameDuration_leftover = sameDuration
        while sameDuration_leftover > 1.:
            sameDuration_int += 1
            sameDuration_leftover -= 1

        if sameDuration_int%2 == 0:
            sameDuration_int = sameDuration_int - 1
            sameDuration_leftover = sameDuration_leftover + 1.
        
        for j in northRow_unphys:
            #get spillover column idxs
            spillover_j = np.arange(j-(int(np.floor(sameDuration_int/2))), j+(int(np.floor(sameDuration_int/2))) + 1)
            spillover_j = spillover_j[(spillover_j >= 0.) &  (spillover_j < M)]
            
            #let unphysical opacities overflow, where the distribution of overflows is proportional to
            # the pixel's "contribution" to the transit duration
            
            if northRow[j] > 1.:
                amtOverflow = northRow[j] - 1.
                newtau[row, j] = 1.
            elif northRow[j] < 0.:
                amtOverflow = northRow[j]
                newtau[row, j] = 0.
                
            directOverflowWeight = (1./sameDuration)
            edgeOverflowWeight = (sameDuration_leftover/2.)
            
            for col in spillover_j:
                newtau[row-1,col] += (directOverflowWeight*amtOverflow)
                
            leftCol = j - int(np.floor(sameDuration_int/2)) - 1
            rightCol = j + int(np.floor(sameDuration_int/2)) + 1
                    
            while leftCol < 0:
                leftCol = leftCol + 1
            while rightCol > M-1:
                rightCol = rightCol - 1
                
            newtau[row-1, leftCol] += (edgeOverflowWeight*amtOverflow)
            newtau[row-1, rightCol] += (edgeOverflowWeight*amtOverflow)
            
        for j in southRow_unphys:
            #print j
            #get spillover column idxs
            spillover_j = np.arange(j-(int(np.floor(sameDuration_int/2))), j+(int(np.floor(sameDuration_int/2))) + 1)
            spillover_j = spillover_j[(spillover_j >= 0.) &  (spillover_j < M)]
            
            #let unphysical opacities overflow, where the distribution of overflows is proportional to
            # the pixel's "contribution" to the transit duration
            
            if southRow[j] > 1.:
                amtOverflow = southRow[j] - 1.
                newtau[N-1-row, j] = 1.
            elif southRow[j] < 0.:
                amtOverflow = southRow[j]
                newtau[N-1-row, j] = 0.
                
            directOverflowWeight = (1./sameDuration)
            edgeOverflowWeight = (sameDuration_leftover/2.)
            
            for col in spillover_j:
                newtau[N-row,col] += (directOverflowWeight*amtOverflow)
                
            leftCol = j - int(np.floor(sameDuration_int/2)) - 1
            rightCol = j + int(np.floor(sameDuration_int/2)) + 1
                    
            while leftCol < 0:
                leftCol = leftCol + 1
            while rightCol > M-1:
                rightCol = rightCol - 1
                
            newtau[N-row, leftCol] += (edgeOverflowWeight*amtOverflow)
            newtau[N-row, rightCol] += (edgeOverflowWeight*amtOverflow)
            
        b += w
        nextRow_b += w
    
    return newtau

def wedgeNegativeEdge(tau):
    """
    Exploit the "wedge degeneracy" to shift opacity around, outside-in.
    """
    
    # Start at the middle of the grid
    N = np.shape(tau)[0]
    M = np.shape(tau)[1]
    middleN = int(np.floor((N-1)/2.))
    #print "middleN is {0}".format(middleN)
    
    w = 2./N
    
    proptau = copy.copy(tau)
    
    #N even
    if N%2 == 0:
        #print "even"
        northRows = np.arange(middleN, -1, -1)
        southRows = np.arange(N-1-middleN, N, 1)
        b = w/2.
        outermost_b = 1. - w/2.
        
    #N odd
    else:
        #print "odd"
        northRows = np.arange(middleN-1, -1, -1)
        southRows = np.arange(N-middleN, N, 1)
        
        #pull opacity from outer rows to central row
        b = 0.
        #propPrior = (1.-b**2)**0.25 * w**2 # (1-b^2)^(1/4) * p^2, from Kipping & Sandford 2016
        
        #row that negative opacity is pulled from: loop from outermost to innermost rows
        for fillop in [1.0]:
            for outerRow in range(0, middleN):
                #re-evaluate which pixels are full
                middleRow = proptau[middleN]
                #print middleRow
                middleRow_notempty = np.arange(0,M)[middleRow > 0.]

                #print outerRow
                #print N-1-outerRow

                outer_b = 1. - w/2. - outerRow*w
                #print outer_b

                #get diameter of the star at that position in the limb
                outer_x = (2.*np.sqrt(1.-outer_b**2))/w
                #print "central row outer_x is {0}".format(outer_x)

                #width of pixel block with same transit duration [units of PIXELS]
                sameDuration = (w + 2.*np.sqrt(1.-b**2) - 2.*np.sqrt(1.-outer_b**2)) / w

                sameDuration_forOpacity = copy.copy(sameDuration)
                #prevent "same duration" block from becoming wider than the grid
                while sameDuration > M:
                    sameDuration = sameDuration - 2.

                while sameDuration > outer_x:
                    sameDuration = sameDuration - 2.

                sameDuration_int = 0
                sameDuration_leftover = copy.copy(sameDuration)
                while sameDuration_leftover > 1.:
                    sameDuration_int += 1
                    sameDuration_leftover -= 1

                if sameDuration_int%2 == 0:
                    sameDuration_int = sameDuration_int - 1
                    sameDuration_leftover = sameDuration_leftover + 1.

                sameDuration_forOpacity_int = 0
                sameDuration_forOpacity_leftover = copy.deepcopy(sameDuration_forOpacity)

                while sameDuration_forOpacity_leftover > 1.:
                    sameDuration_forOpacity_int += 1
                    sameDuration_forOpacity_leftover -= 1

                if sameDuration_forOpacity_int%2 == 0:
                    sameDuration_forOpacity_int = sameDuration_forOpacity_int - 1
                    sameDuration_forOpacity_leftover = sameDuration_forOpacity_leftover + 1.

                for j in middleRow_notempty:
                    #get spill-in column idxs (relative to idx of the pixel they're spilling into)
                    spillin_j = np.arange(j-(int(np.floor(sameDuration_int/2))), j+(int(np.floor(sameDuration_int/2))) + 1)
                    #print j
                    #print spillin_j
                    #eliminate columns outside the bounds of the grid
                    spillin_j = spillin_j[(spillin_j >= 0.) &  (spillin_j < M)]
                    #print spillin_j
                    extra_spillin_j = np.arange(j-(int(np.floor(sameDuration_forOpacity_int/2))), j+(int(np.floor(sameDuration_forOpacity_int/2))) + 1)
                    extra_spillin_j = extra_spillin_j[(extra_spillin_j >= 0.) &  (extra_spillin_j < M)]
                    extra_spillin_j = extra_spillin_j[np.where(np.in1d(extra_spillin_j, spillin_j, invert=True))[0]]

                    #let outermost negative opacities flow in, where the distribution of where the opacities come from is proportional to
                    # the pixel's "contribution" to the transit duration
                    amtToFill = middleRow[j]

                    #print "amtToFill is {0}".format(amtToFill)

                    directOverflowWeight = (1./sameDuration_forOpacity)
                    edgeOverflowWeight = (sameDuration_leftover/2.)

                    for col in spillin_j: #This only works if the input grid is symmetrical!!!!
                        if ((proptau[outerRow, col] < 0.) & (proptau[N-1-outerRow,col] < 0.) & (proptau[middleN, j] + proptau[outerRow, col] + proptau[N-1-outerRow,col] >= 0.)):
                            proptau[middleN, j] += (proptau[outerRow, col] + proptau[N-1-outerRow,col])
                            proptau[outerRow, col] = 0.
                            proptau[N-1-outerRow, col] = 0.

                        elif ((proptau[outerRow, col] < 0.) & (proptau[N-1-outerRow,col] < 0.) & (proptau[middleN, j] + proptau[outerRow, col] + proptau[N-1-outerRow,col] < 0.)):
                           excess = proptau[middleN, j] + proptau[outerRow, col] + proptau[N-1-outerRow,col]
                           proptau[middleN, j] = 0.
                           proptau[outerRow, col] -= excess/2.
                           proptau[N-1-outerRow, col] -= excess/2.
                    """
                    for col in extra_spillin_j:
                        if proptau[outerRow, col] < 0:
                            proptau[outerRow, col] = 0.
                    """
        
    #do the same for the next-middlemost rows, out toward the top and bottom of the grid.
    for fillop in [1.0]:
        for nrow in northRows[:-1][::-1]: #no need to do it for the top row
            northRow = proptau[nrow]
            northRow_notempty = np.arange(0,M)[(northRow > 0.)]

            #pull opacity from outermost row first
            b = 1. - w/2. - nrow*w

            #print b

            #row that opacity is pulled from: loop from outermost to innermost rows
            for outerRow in range(0, nrow):
                #re-evaluate which pixels are empty
                northRow = proptau[nrow]
                northRow_notempty = np.arange(0,M)[(northRow > 0.)]

                #get impact parameter of outer transiting row
                outer_b = 1. - w/2. - outerRow*w

                #get stellar diameter at that impact parameter
                outer_x = (2.*np.sqrt(1.-outer_b**2))/w

                #width of pixel block with same transit duration [units of PIXELS]
                sameDuration = (w + 2.*np.sqrt(1.-b**2) - 2.*np.sqrt(1.-outer_b**2)) / w

                sameDuration_forOpacity = copy.deepcopy(sameDuration)

                #prevent "same duration" block from becoming wider than the grid
                while sameDuration > M:
                    sameDuration = sameDuration - 2.

                while sameDuration > outer_x:
                    sameDuration = sameDuration - 2.

                sameDuration_int = 0
                sameDuration_leftover = copy.deepcopy(sameDuration)
                while sameDuration_leftover > 1.:
                    sameDuration_int += 1
                    sameDuration_leftover -= 1

                if sameDuration_int%2 == 0:
                    sameDuration_int = sameDuration_int - 1
                    sameDuration_leftover = sameDuration_leftover + 1.

                sameDuration_forOpacity_int = 0
                sameDuration_forOpacity_leftover = copy.deepcopy(sameDuration_forOpacity)

                while sameDuration_forOpacity_leftover > 1.:
                    sameDuration_forOpacity_int += 1
                    sameDuration_forOpacity_leftover -= 1

                if sameDuration_forOpacity_int%2 == 0:
                    sameDuration_forOpacity_int = sameDuration_forOpacity_int - 1
                    sameDuration_forOpacity_leftover = sameDuration_forOpacity_leftover + 1.

                for j in northRow_notempty:
                    #get spill-in column idxs (relative to idx of the pixel they're spilling into)
                    spillin_j = np.arange(j-(int(np.floor(sameDuration_int/2))), j+(int(np.floor(sameDuration_int/2))) + 1)
                    #eliminate columns outside the bounds of the grid
                    spillin_j = spillin_j[(spillin_j >= 0.) &  (spillin_j < M)]
                    #print "spillin_j is {0}".format(spillin_j)

                    extra_spillin_j = np.arange(j-(int(np.floor(sameDuration_forOpacity_int/2))), j+(int(np.floor(sameDuration_forOpacity_int/2))) + 1)
                    extra_spillin_j = extra_spillin_j[(extra_spillin_j >= 0.) &  (extra_spillin_j < M)]
                    extra_spillin_j = extra_spillin_j[np.where(np.in1d(extra_spillin_j, spillin_j, invert=True))[0]]
                    #print "extra_spillin_j is {0}".format(extra_spillin_j)

                    #let outermost opacities flow in, where the distribution of where the opacities come from is proportional to
                    # the pixel's "contribution" to the transit duration
                    amtToFill = northRow[j]

                    directOverflowWeight = (fillop/sameDuration_forOpacity)
                    edgeOverflowWeight = (sameDuration_forOpacity_leftover/(2./fillop))

                    for col in spillin_j: #This only works if the input grid is symmetrical!!!!
                        if ((proptau[outerRow, col] < 0.) & (proptau[nrow, j] + proptau[outerRow, col] >= 0.)):
                            proptau[nrow, j] += proptau[outerRow, col]
                            proptau[outerRow, col] = 0.

                        elif ((proptau[outerRow, col] < 0.) & (proptau[nrow, j] + proptau[outerRow, col] < 0.)):
                           excess = proptau[nrow, j] + proptau[outerRow, col]
                           proptau[nrow, j] = 0.
                           proptau[outerRow, col] -= excess

                    
                    """
                    for col in extra_spillin_j:
                        if proptau[outerRow, col] < 0:
                            proptau[outerRow, col] = 0.
                    """
                        
                    #make proposed tau grid symmetrical
                    for srowidx, srow in enumerate(southRows):
                        proptau[srow] = proptau[northRows[srowidx]]
                
        
    return proptau

def wedgeOptimize_sym(tau, obsLC, obsLCerr, areas):
    """
    Exploit the "wedge degeneracy" to shift opacity around. This is different from wedgeRearrange because here, we're
    starting from a grid of physical opacities (0 <= tau <= 1).
    
    Strategy: Start from the middle, and pull opacity from the outermost row until the middle-most pixels are full or the outermost
    pixels are empty. Then move outward to the next-middlemost row, pulling opacity from the outermost row and then the next-outermost row, etc.
    """
    
    # Start at the middle of the grid
    N = np.shape(tau)[0]
    M = np.shape(tau)[1]
    middleN = int(np.floor((N-1)/2.))
    #print "middleN is {0}".format(middleN)
    
    w = 2./N
    
    newtau = copy.copy(tau)
    newtauLC = np.atleast_2d(np.ones_like(obsLC)).T - np.dot(areas,np.reshape(newtau,(N*M,1)))
    newtauLC = newtauLC[:,0]
    newtauCost = RMS(obsLC, obsLCerr, newtauLC)
    proptau = copy.copy(newtau)
    
    #N even
    if N%2 == 0:
        #print "even"
        northRows = np.arange(middleN, -1, -1)
        southRows = np.arange(N-1-middleN, N, 1)
        b = w/2.
        outermost_b = 1. - w/2.
        
    #N odd
    else:
        #print "odd"
        northRows = np.arange(middleN-1, -1, -1)
        southRows = np.arange(N-middleN, N, 1)
        
        #pull opacity from outer rows to central row
        b = 0.
        #propPrior = (1.-b**2)**0.25 * w**2 # (1-b^2)^(1/4) * p^2, from Kipping & Sandford 2016
        
        #row that opacity is pulled from: loop from outermost to innermost rows
        for fillop in [1.0, 0.5]:
            for outerRow in range(0, middleN):
                #re-evaluate which pixels are full
                middleRow = proptau[middleN]
                #print middleRow
                middleRow_notfull = np.arange(0,M)[(middleRow > (fillop-0.5)) & (middleRow < fillop)]

                #print outerRow
                #print N-1-outerRow

                outer_b = 1. - w/2. - outerRow*w
                #print outer_b

                #get diameter of the star at that position in the limb
                outer_x = (2.*np.sqrt(1.-outer_b**2))/w
                #print "central row outer_x is {0}".format(outer_x)

                #width of pixel block with same transit duration [units of PIXELS]
                sameDuration = (w + 2.*np.sqrt(1.-b**2) - 2.*np.sqrt(1.-outer_b**2)) / w

                sameDuration_forOpacity = copy.copy(sameDuration)
                #prevent "same duration" block from becoming wider than the grid
                while sameDuration > M:
                    sameDuration = sameDuration - 2.

                while sameDuration > outer_x:
                    sameDuration = sameDuration - 2.

                sameDuration_int = 0
                sameDuration_leftover = copy.copy(sameDuration)
                while sameDuration_leftover > 1.:
                    sameDuration_int += 1
                    sameDuration_leftover -= 1

                if sameDuration_int%2 == 0:
                    sameDuration_int = sameDuration_int - 1
                    sameDuration_leftover = sameDuration_leftover + 1.

                sameDuration_forOpacity_int = 0
                sameDuration_forOpacity_leftover = copy.deepcopy(sameDuration_forOpacity)

                while sameDuration_forOpacity_leftover > 1.:
                    sameDuration_forOpacity_int += 1
                    sameDuration_forOpacity_leftover -= 1

                if sameDuration_forOpacity_int%2 == 0:
                    sameDuration_forOpacity_int = sameDuration_forOpacity_int - 1
                    sameDuration_forOpacity_leftover = sameDuration_forOpacity_leftover + 1.

                for j in middleRow_notfull:
                    #get spill-in column idxs (relative to idx of the pixel they're spilling into)
                    spillin_j = np.arange(j-(int(np.floor(sameDuration_int/2))), j+(int(np.floor(sameDuration_int/2))) + 1)
                    #print j
                    #print spillin_j
                    #eliminate columns outside the bounds of the grid
                    spillin_j = spillin_j[(spillin_j >= 0.) &  (spillin_j < M)]
                    #print spillin_j
                    extra_spillin_j = np.arange(j-(int(np.floor(sameDuration_forOpacity_int/2))), j+(int(np.floor(sameDuration_forOpacity_int/2))) + 1)
                    extra_spillin_j = extra_spillin_j[(extra_spillin_j >= 0.) &  (extra_spillin_j < M)]
                    extra_spillin_j = extra_spillin_j[np.where(np.in1d(extra_spillin_j, spillin_j, invert=True))[0]]

                    #let outermost opacities flow in, where the distribution of where the opacities come from is proportional to
                    # the pixel's "contribution" to the transit duration
                    amtToFill = fillop - middleRow[j]

                    #print "amtToFill is {0}".format(amtToFill)

                    directOverflowWeight = (fillop/sameDuration_forOpacity)
                    edgeOverflowWeight = (sameDuration_leftover/(2./fillop))

                    for col in spillin_j: #This only works if the input grid is symmetrical!!!!
                        if ((proptau[outerRow,col] - (directOverflowWeight*amtToFill)/2.) >= 0.) & (proptau[N-1-outerRow,col] - (directOverflowWeight*amtToFill)/2. >= 0.) & (proptau[middleN, j] + directOverflowWeight*amtToFill <= fillop):
                            proptau[middleN, j] += directOverflowWeight*amtToFill
                            proptau[outerRow,col] -= (directOverflowWeight*amtToFill)/2. #divide by 2 because middle row overflows both north and south
                            proptau[N-1-outerRow,col] -= (directOverflowWeight*amtToFill)/2. #divide by 2 because middle row overflows both north and south
                        elif ((proptau[outerRow,col] - (directOverflowWeight*amtToFill)/2.) < 0.) | (proptau[N-1-outerRow,col] - (directOverflowWeight*amtToFill)/2. < 0.):
                            proptau[middleN, j] += proptau[outerRow,col]
                            proptau[middleN, j] += proptau[N-1-outerRow,col]
                            proptau[outerRow, col] = 0.
                            proptau[N-1-outerRow, col] = 0.
                        elif (proptau[middleN, j] + directOverflowWeight*amtToFill > fillop):
                            excess = (fillop - proptau[middleN, j])/2.
                            proptau[middleN, j] = fillop
                            proptau[outerRow, col] -= excess
                            proptau[N-1-outerRow, col] -= excess 


                        leftCol = j - int(np.floor(sameDuration_int/2)) - 1
                        rightCol = j + int(np.floor(sameDuration_int/2)) + 1

                        while leftCol < 0:
                            leftCol = leftCol + 1
                        while rightCol > M-1:
                            rightCol = rightCol - 1

                        #left col
                        if ((proptau[outerRow,leftCol] - (edgeOverflowWeight*amtToFill)/2.) >= 0.) & (proptau[N-1-outerRow,leftCol] - (edgeOverflowWeight*amtToFill)/2. >= 0.) & (proptau[middleN, j] + edgeOverflowWeight*amtToFill <= fillop):
                            proptau[middleN, j] += edgeOverflowWeight*amtToFill
                            proptau[outerRow,leftCol] -= (edgeOverflowWeight*amtToFill)/2. #divide by 2 because middle row overflows both north and south
                            proptau[N-1-outerRow,leftCol] -= (edgeOverflowWeight*amtToFill)/2. #divide by 2 because middle row overflows both north and south
                        elif ((proptau[outerRow,leftCol] - (edgeOverflowWeight*amtToFill)/2.) < 0.) | (proptau[N-1-outerRow,leftCol] - (edgeOverflowWeight*amtToFill)/2. < 0.):
                            proptau[middleN, j] += proptau[outerRow,leftCol]
                            proptau[middleN, j] += proptau[N-1-outerRow,leftCol]
                            proptau[outerRow,leftCol] = 0.
                            proptau[N-1-outerRow,leftCol] = 0.
                        elif ((proptau[outerRow,leftCol] - (edgeOverflowWeight*amtToFill)/2.) >= 0.) & (proptau[N-1-outerRow,leftCol] - (edgeOverflowWeight*amtToFill)/2. >= 0.) & (proptau[middleN, j] + edgeOverflowWeight*amtToFill > fillop):
                            excess = (fillop - proptau[middleN, j])/2.
                            proptau[middleN, j] = fillop
                            proptau[outerRow,leftCol] -= excess 
                            proptau[N-1-outerRow,leftCol] -= excess

                        #right col
                        if ((proptau[outerRow,rightCol] - (edgeOverflowWeight*amtToFill)/2.) >= 0.) & (proptau[N-1-outerRow,rightCol] - (edgeOverflowWeight*amtToFill)/2. >= 0.) & (proptau[middleN, j] + edgeOverflowWeight*amtToFill <= fillop):
                            proptau[middleN, j] += edgeOverflowWeight*amtToFill
                            proptau[outerRow,rightCol] -= (edgeOverflowWeight*amtToFill)/2. #divide by 2 because middle row overflows both north and south
                            proptau[N-1-outerRow,rightCol] -= (edgeOverflowWeight*amtToFill)/2. #divide by 2 because middle row overflows both north and south       
                        elif ((proptau[outerRow,rightCol] - (edgeOverflowWeight*amtToFill)/2.) < 0.) | (proptau[N-1-outerRow,rightCol] - (edgeOverflowWeight*amtToFill)/2. < 0.):
                            proptau[middleN, j] += proptau[outerRow,rightCol]
                            proptau[middleN, j] += proptau[N-1-outerRow,rightCol]
                            proptau[outerRow,rightCol] = 0.
                            proptau[N-1-outerRow,rightCol] = 0.
                        elif ((proptau[outerRow,rightCol] - (edgeOverflowWeight*amtToFill)/2.) >= 0.) & (proptau[N-1-outerRow,rightCol] - (edgeOverflowWeight*amtToFill)/2. >= 0.) & (proptau[middleN, j] + edgeOverflowWeight*amtToFill > fillop):
                            excess = (fillop - proptau[middleN, j])/2.
                            proptau[middleN, j] = fillop
                            proptau[outerRow,rightCol] -= excess 
                            proptau[N-1-outerRow,rightCol] -= excess

                    for col in extra_spillin_j:
                        proptau[outerRow, col] = 0.

                    #account for prior in deciding whether to accept
                    propPrior = (1.-b**2)**0.25 * w**2 # (1-b^2)^(1/4) * p^2, from Kipping & Sandford 2016
                    oldPrior = 2*(1.-outer_b**2)**0.25 * (w**2 * sameDuration_forOpacity) #use area of spill-in pixel blocks to calculate ratio-of-radii proxy
            
                    proptauLC = np.atleast_2d(np.ones_like(obsLC)).T - np.dot(areas,np.reshape(proptau,(N*M,1)))
                    proptauLC = proptauLC[:,0]
                    proptauCost = RMS(obsLC, obsLCerr, proptauLC)
                    
                    deltaRMS = np.exp(-0.5*(proptauCost**2 - newtauCost**2))
                    postRatio = deltaRMS * (propPrior/oldPrior)
                    
                    testProb = np.random.uniform(0.,1.)
                    if testProb < postRatio:
                        newtau = proptau
                        newtauCost = proptauCost
                        proptau = copy.copy(newtau)
                    else:
                        proptau = copy.copy(newtau)
            
        #do not account for prior in deciding whether to accept
        """proptauLC = np.atleast_2d(np.ones_like(obsLC)).T - np.dot(areas,np.reshape(proptau,(N*M,1)))
        proptauLC = proptauLC[:,0]
        proptauCost = RMS(obsLC, obsLCerr, proptauLC)
        
        newtau = copy.copy(proptau)
        newtauCost = proptauCost
        proptau = copy.copy(newtau)"""
        
    #do the same for the next-middlemost rows, out toward the top and bottom of the grid.
    for fillop in [1.0, 0.5]:
        for nrow in northRows[:-1][::-1]: #no need to do it for the top row
            northRow = proptau[nrow]
            northRow_notfull = np.arange(0,M)[(northRow > (fillop-0.5)) & (northRow < fillop)]

            #pull opacity from outermost row first
            b = 1. - w/2. - nrow*w

            #print b

            #row that opacity is pulled from: loop from outermost to innermost rows
            for outerRow in range(0, nrow):
                #re-evaluate which pixels are full
                northRow = proptau[nrow]
                northRow_notfull = np.arange(0,M)[(northRow > (fillop-0.5)) & (northRow < fillop)]

                #get impact parameter of outer transiting row
                outer_b = 1. - w/2. - outerRow*w

                #get stellar diameter at that impact parameter
                outer_x = (2.*np.sqrt(1.-outer_b**2))/w

                #width of pixel block with same transit duration [units of PIXELS]
                sameDuration = (w + 2.*np.sqrt(1.-b**2) - 2.*np.sqrt(1.-outer_b**2)) / w

                sameDuration_forOpacity = copy.deepcopy(sameDuration)

                #prevent "same duration" block from becoming wider than the grid
                while sameDuration > M:
                    sameDuration = sameDuration - 2.

                while sameDuration > outer_x:
                    sameDuration = sameDuration - 2.

                sameDuration_int = 0
                sameDuration_leftover = copy.deepcopy(sameDuration)
                while sameDuration_leftover > 1.:
                    sameDuration_int += 1
                    sameDuration_leftover -= 1

                if sameDuration_int%2 == 0:
                    sameDuration_int = sameDuration_int - 1
                    sameDuration_leftover = sameDuration_leftover + 1.

                sameDuration_forOpacity_int = 0
                sameDuration_forOpacity_leftover = copy.deepcopy(sameDuration_forOpacity)

                while sameDuration_forOpacity_leftover > 1.:
                    sameDuration_forOpacity_int += 1
                    sameDuration_forOpacity_leftover -= 1

                if sameDuration_forOpacity_int%2 == 0:
                    sameDuration_forOpacity_int = sameDuration_forOpacity_int - 1
                    sameDuration_forOpacity_leftover = sameDuration_forOpacity_leftover + 1.

                for j in northRow_notfull:
                    #get spill-in column idxs (relative to idx of the pixel they're spilling into)
                    spillin_j = np.arange(j-(int(np.floor(sameDuration_int/2))), j+(int(np.floor(sameDuration_int/2))) + 1)
                    #eliminate columns outside the bounds of the grid
                    spillin_j = spillin_j[(spillin_j >= 0.) &  (spillin_j < M)]
                    #print "spillin_j is {0}".format(spillin_j)

                    extra_spillin_j = np.arange(j-(int(np.floor(sameDuration_forOpacity_int/2))), j+(int(np.floor(sameDuration_forOpacity_int/2))) + 1)
                    extra_spillin_j = extra_spillin_j[(extra_spillin_j >= 0.) &  (extra_spillin_j < M)]
                    extra_spillin_j = extra_spillin_j[np.where(np.in1d(extra_spillin_j, spillin_j, invert=True))[0]]
                    #print "extra_spillin_j is {0}".format(extra_spillin_j)

                    #let outermost opacities flow in, where the distribution of where the opacities come from is proportional to
                    # the pixel's "contribution" to the transit duration
                    amtToFill = fillop - northRow[j]

                    directOverflowWeight = (fillop/sameDuration_forOpacity)
                    edgeOverflowWeight = (sameDuration_forOpacity_leftover/(2./fillop))

                    for col in spillin_j: 
                        if ((proptau[outerRow,col] - (directOverflowWeight*amtToFill)) >= 0.) & (proptau[nrow,j] + directOverflowWeight*amtToFill <= fillop):
                            proptau[nrow, j] += directOverflowWeight*amtToFill
                            proptau[outerRow,col] -= (directOverflowWeight*amtToFill)
                        elif (proptau[outerRow,col] - (directOverflowWeight*amtToFill) < 0.):
                            proptau[nrow, j] += proptau[outerRow,col]
                            proptau[outerRow, col] = 0.
                        elif (proptau[nrow,j] + directOverflowWeight*amtToFill > fillop):
                            excess = fillop - proptau[nrow, j]
                            proptau[nrow, j] = fillop
                            proptau[outerRow, col] -= excess

                        leftCol = j - int(np.floor(sameDuration_int/2)) - 1
                        rightCol = j + int(np.floor(sameDuration_int/2)) + 1

                        while leftCol < 0:
                            leftCol = leftCol + 1
                        while rightCol > M-1:
                            rightCol = rightCol - 1

                        if ((proptau[outerRow,leftCol] - (edgeOverflowWeight*amtToFill)) >= 0.) & (proptau[nrow, j] + edgeOverflowWeight*amtToFill <= fillop) :
                            proptau[nrow, j] += edgeOverflowWeight*amtToFill
                            proptau[outerRow,leftCol] -= (edgeOverflowWeight*amtToFill)    
                        elif (proptau[outerRow,leftCol] - (edgeOverflowWeight*amtToFill) < 0.):
                            proptau[nrow, j] += proptau[outerRow,leftCol]
                            proptau[outerRow,leftCol] = 0.
                        elif (proptau[nrow, j] + edgeOverflowWeight*amtToFill > fillop):
                            excess = fillop - proptau[nrow,j]
                            proptau[nrow, j] = fillop
                            proptau[outerRow,leftCol] -= excess

                        if ((proptau[outerRow,rightCol] - (edgeOverflowWeight*amtToFill)) >= 0.) & (proptau[nrow, j] + edgeOverflowWeight*amtToFill <= fillop) :
                            proptau[nrow, j] += edgeOverflowWeight*amtToFill
                            proptau[outerRow,rightCol] -= (edgeOverflowWeight*amtToFill)    
                        elif (proptau[outerRow,rightCol] - (edgeOverflowWeight*amtToFill) < 0.):
                            proptau[nrow, j] += proptau[outerRow,rightCol]
                            proptau[outerRow,rightCol] = 0.
                        elif (proptau[nrow, j] + edgeOverflowWeight*amtToFill > fillop):
                            excess = fillop - proptau[nrow,j]
                            proptau[nrow, j] = fillop
                            proptau[outerRow,rightCol] -= excess

                    for col in extra_spillin_j:
                        proptau[outerRow, col] = 0.
                        
                    #make proposed tau grid symmetrical
                    for srowidx, srow in enumerate(southRows):
                        proptau[srow] = proptau[northRows[srowidx]]
                    
                    #account for prior in deciding whether to accept
                    propPrior = (1.-b**2)**0.25 * w**2 # (1-b^2)^(1/4) * p^2, from Kipping & Sandford 2016
                    oldPrior = 2.*(1.-outer_b**2)**0.25 * (w**2 * sameDuration_forOpacity) #use area of spill-in pixel blocks to calculate ratio-of-radii proxy
            
                    proptauLC = np.atleast_2d(np.ones_like(obsLC)).T - np.dot(areas,np.reshape(proptau,(N*M,1)))
                    proptauLC = proptauLC[:,0]
                    proptauCost = RMS(obsLC, obsLCerr, proptauLC)
                    
                    deltaRMS = np.exp(-0.5*(proptauCost**2 - newtauCost**2))
                    postRatio = deltaRMS * (propPrior/oldPrior)
                    
                    testProb = np.random.uniform(0.,1.)
                    if testProb < postRatio:
                        #print "north better"
                        newtau = proptau
                        newtauCost = proptauCost
                        proptau = copy.copy(newtau)
                    else:
                        proptau = copy.copy(newtau)
                
        
    return proptau

cpdef foldOpacities(np.ndarray[double, ndim=2] tau):
    """
    Start with a 2D opacity grid and fold opacities over to northern hemisphere
    """
    cdef:
        np.ndarray[np.double_t, ndim=2] foldedTau
        int N, M, i, j
    
    foldedTau = copy.copy(tau)
    N = np.shape(tau)[0]
    M = np.shape(tau)[1]
    
    for i in range(0, int(np.floor(N/2.))):
        for j in range(0,M):
            if (tau[i,j] + tau[N-1-i, j]) <= 1.:
                foldedTau[i,j] = foldedTau[i,j] + foldedTau[N-1-i, j]
                foldedTau[N-1-i,j] = 0.
                
            else:
                foldedTau[N-1-i,j] = (foldedTau[i,j] + foldedTau[N-1-i, j]) - 1.
                foldedTau[i,j] = 1.
                
    return foldedTau

def round_ART(ARTsoln):
    """
    Rounds each entry in best-fit ART opacity grid to 0, 0.5, or 1.
    """
    roundedtau = copy.copy(ARTsoln)
    
    N = np.shape(roundedtau)[0]
    M = np.shape(roundedtau)[1]
    
    #for i in range(0,N):
    #    for j in range(0,M):
    #        if roundedtau[i,j] > 1.0:
    #            roundedtau[N-1-i,j] = roundedtau[i,j]/2.
    #            roundedtau[i,j] = roundedtau[i,j]/2.
                
    toobigmask = (roundedtau > 1.0)
    roundedtau[toobigmask] = 1.0
    
    toosmallmask = (roundedtau < 0.0)
    roundedtau[toosmallmask] = 0.0
    
    #0, 0.5, or 1.
    """
    nearzeromask = (roundedtau > 0.0) & (roundedtau <= (1./10.))
    roundedtau[nearzeromask] = 0.0
    
    nearpt5mask = (roundedtau > (1./10.)) & (roundedtau <= (9./10.))
    roundedtau[nearpt5mask] = 0.5
    
    nearonemask = (roundedtau > (9./10.)) & (roundedtau <= 1.0)
    roundedtau[nearonemask] = 1.0
    """
    #0. or 1.
    nearzeromask = (roundedtau > 0.0) & (roundedtau < 0.5)
    roundedtau[nearzeromask] = 0.0
    
    nearonemask = (roundedtau >= 0.5) & (roundedtau <= 1.0)
    roundedtau[nearonemask] = 1.0
    
    roundedtau=np.abs(roundedtau)
    return roundedtau

def invertLC(N, M, v, t_ref, t_arr, obsLC, obsLCerr, nTrial, filename, window=None, LDlaw="uniform",LDCs=[],fac=0.001,RMSstop=1.e-6, n_iter=0, WR=True, WO=True, initstate="uniform",reg=0.,threshold=False):
    """
    Run the following algorithms in sequence:
        - Simultaneous ART
        - wedgeRearrange (if WR is True)
        - wedgeOptimize (if WO is True)
        - fold/round

    and output results at two stages:
        - after simultaneous ART and optional WR and optional WO
        - after folding/rounding of above.
    
    """
    
    if LDlaw == "uniform":
        ti = TransitingImage(opacitymat=np.zeros((N,M)), LDlaw="uniform", v=v, t_ref=t_ref, t_arr=t_arr)
        ti_LC = ti.gen_LC(t_arr)

        raveledareas = np.zeros((np.shape(ti.areas)[0],np.shape(ti.areas)[1]*np.shape(ti.areas)[2])) 

        for i in range(0,np.shape(ti.areas)[0]): #time axis
            for j in range(0,np.shape(ti.areas)[1]): #N axis
                raveledareas[i,M*j:M*(j+1)] = ti.areas[i,j,:]

    elif LDlaw == "quadratic":
        #the nonlinear case reduces to the quadratic case by the following equations:
        c1 = 0.
        c3 = 0.
        
        c2 = LDCs[0] + 2.*LDCs[1]
        c4 = -1.*LDCs[1]

        nonlinearLDCs = [c1,c2,c3,c4]

        ti = TransitingImage(opacitymat=np.ones((N,M)), LDlaw="nonlinear", LDCs=nonlinearLDCs, v=v, t_ref=t_ref, t_arr=t_arr)
        ti_LC = ti.gen_LC(t_arr)

        raveledareas = np.zeros((np.shape(ti.LD)[0],np.shape(ti.LD)[1]*np.shape(ti.LD)[2])) 

        for i in range(0,np.shape(ti.LD)[0]): #time axis
            for j in range(0,np.shape(ti.LD)[1]): #N axis
                raveledareas[i,M*j:M*(j+1)] = ti.LD[i,j,:]

    elif LDlaw == "nonlinear":
        ti = TransitingImage(opacitymat=np.ones((N,M)), LDlaw="nonlinear", LDCs=LDCs, v=v, t_ref=t_ref, t_arr=t_arr)
        ti_LC = ti.gen_LC(t_arr)

        raveledareas = np.zeros((np.shape(ti.LD)[0],np.shape(ti.LD)[1]*np.shape(ti.LD)[2])) 

        for i in range(0,np.shape(ti.LD)[0]): #time axis
            for j in range(0,np.shape(ti.LD)[1]): #N axis
                raveledareas[i,M*j:M*(j+1)] = ti.LD[i,j,:]
        
    
    # previously:
    # try:
    #     raveledtau = ART(tau_init=0.5*np.ones((N,M)), A=raveledareas, obsLC=obsLC, mirrored=False, RMSstop=RMSstop, reg=0., n_iter=n_iter)
    # except ValueError:
    #     raveledtau = ART_normal(tau_init=0.5*np.ones((N,M)), A=raveledareas, obsLC=obsLC, mirrored=False, RMSstop=RMSstop, reg=0.)
            
    
    #take only half of area matrix to avoid dealing with flip degeneracy
    if (N>1) & (N%2 == 0):
        Nhalf = int(N/2)
    elif (N>1) & (N%2 != 0):
        Nhalf = int((N-1)/2 + 1)
    
    halfAreas = raveledareas[:,0:(Nhalf*M)]
    
    # Run simultaneous ART according to user's choice of initial grid.
    if initstate=="uniform":  
        raveledtau = simultaneous_ART(n_iter=n_iter, tau_init=0.5*np.ones((Nhalf,M)), A=halfAreas, obsLC=obsLC, obsLCerr=obsLCerr, reg=reg, threshold=threshold, filename=filename,window=window)
    elif initstate=="empty":
        raveledtau = simultaneous_ART(n_iter=n_iter, tau_init=np.zeros((Nhalf,M)), A=halfAreas, obsLC=obsLC, obsLCerr=obsLCerr, reg=reg, threshold=threshold, filename=filename,window=window)
    elif initstate=="random":
        raveledtau = simultaneous_ART(n_iter=n_iter, tau_init=np.random.uniform(0.,1.,(Nhalf,M)), A=halfAreas, obsLC=obsLC, obsLCerr=obsLCerr, reg=reg, threshold=threshold, filename=filename,window=window)
    else: #allow for user to input an initial state matrix
        initstateHalf = copy.deepcopy(initstate)[0:Nhalf,:]
        initstateHalf = np.ravel(initstateHalf)
        raveledtau = simultaneous_ART(n_iter=n_iter, tau_init=initstateHalf, A=halfAreas, obsLC=obsLC, obsLCerr=obsLCerr, reg=reg, threshold=threshold, filename=filename,window=window)
    
    
    if (N>1) & (N%2 == 0): #N even
    #raveledtau = top pixels only 
        raveledtauHalf = np.reshape(copy.deepcopy(raveledtau), (int(N/2), M))
        raveledtau = np.zeros((N, M))
        raveledtau[0:int(N/2)] = raveledtauHalf
        for rowIdx in np.arange(N-1, int(N/2) - 1, -1):
            raveledtau[rowIdx] = raveledtauHalf[N - rowIdx - 1]
            
    elif (N>1) & (N%2 != 0):
        #raveledtau = top pixels + 1 row only 
        raveledtauHalf = np.reshape(copy.deepcopy(raveledtau), (int((N-1)/2 + 1), M))
        raveledtau = np.zeros((N, M))
        raveledtau[0:int((N-1)/2 + 1)] = raveledtauHalf
        for rowIdx in np.arange(N-1, int((N-1)/2), -1):
            raveledtau[rowIdx] = raveledtauHalf[N - rowIdx - 1]
    
    raveledtau = raveledtau/2.
    raveledtau = np.round(raveledtau,2)
                                             
    bestWOCost = np.inf
    bestWOGrid = np.ones_like(wedgeRearrange(raveledtau))
    
    bestBinaryCost = np.inf
    bestBinaryGrid = np.ones_like(wedgeRearrange(raveledtau))
                                             
    for i in range(nTrial):
        wo = raveledtau
        wo = np.round(wedgeNegativeEdge(wo),2)
        np.save("{0}_SARTfinal.npy".format(filename),wo)

        if threshold is False:
            if WR is True:
                wo = np.round(wedgeRearrange(np.round(wedgeRearrange(np.round(wedgeRearrange(raveledtau),2)),2)),2)
                wo = np.round(wedgeNegativeEdge(wo),2)
                wo = np.round(wedgeRearrange(wo),2)
                wo = np.round(wedgeNegativeEdge(wo),2)
                np.save("{0}_WR.npy".format(filename),wo)
            elif WO is True:
                wo = np.round(wedgeRearrange(np.round(wedgeRearrange(np.round(wedgeRearrange(raveledtau),2)),2)),2)
                wo = np.round(wedgeRearrange(wedgeOptimize_sym(wedgeOptimize_sym(wedgeOptimize_sym(wo, obsLC=obsLC, obsLCerr=obsLCerr, areas=raveledareas), obsLC=obsLC, obsLCerr=obsLCerr, areas=raveledareas), obsLC=obsLC, obsLCerr=obsLCerr, areas=raveledareas)),2)
                wo = np.round(wedgeNegativeEdge(wo),2)
                wo = np.round(wedgeRearrange(wedgeOptimize_sym(wedgeOptimize_sym(wedgeOptimize_sym(wo, obsLC=obsLC, obsLCerr=obsLCerr, areas=raveledareas), obsLC=obsLC, obsLCerr=obsLCerr, areas=raveledareas), obsLC=obsLC, obsLCerr=obsLCerr, areas=raveledareas)),2)
                wo = np.round(wedgeOptimize_sym(wo, obsLC=obsLC, obsLCerr=obsLCerr, areas=raveledareas),2)
                np.save("{0}_WRWO.npy".format(filename),wo)

        woLC = np.atleast_2d(np.ones_like(t_arr)).T - np.dot(raveledareas,np.reshape(wo,(N*M,1)))
        woLC = woLC[:,0]
        
        woCost = RMS(LC_obs=obsLC,LC_obs_err=obsLCerr,LC_model=woLC)
        
        if woCost < bestWOCost:
            bestWOCost = woCost
            bestWOGrid = wo
            bestWOLC = woLC
        
        foldedART = foldOpacities(wo)

        np.save("{0}_SARTfinal_fold.npy".format(filename),foldedART)

        roundedART = round_ART(foldedART)

        np.save("{0}_SARTfinal_foldround.npy".format(filename),roundedART)

        testLC = np.atleast_2d(np.ones_like(t_arr)).T - np.dot(raveledareas,np.reshape(roundedART,(N*M,1)))
        testLC = testLC[:,0]

        cost = RMS(LC_obs=obsLC,LC_obs_err=obsLCerr,LC_model=testLC)
                                             
        if cost < bestBinaryCost:
            bestBinaryCost = cost
            bestBinaryGrid = roundedART
            bestBinaryLC = testLC
        
    return bestWOGrid, bestWOLC, bestWOCost, bestBinaryGrid, bestBinaryLC, bestBinaryCost