# cython: profile=True
from __future__ import division
import numpy as np
cimport numpy as np
import copy
import matplotlib.pyplot as plt
#from libcpp cimport bool as cybool

from .cTransitingImage import *
from .cGridFunctions import *
from .misc import *

__all__ = ['simultaneous_ART', 'wedgeRearrange','wedgeNegativeEdge', 'wedgeOptimize_sym',
'foldOpacities','round_ART','invertLC']

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
        np.ndarray[np.double_t, ndim=1] tau=np.ravel(tau_init)
        np.ndarray[np.double_t, ndim=1] RHS
        np.ndarray[np.double_t, ndim=1] tau_update=np.zeros_like(tau, dtype=float)
        np.ndarray[np.double_t, ndim=2] testtau=np.zeros_like(tau_init)
        np.ndarray[np.double_t, ndim=2] testLC=np.zeros_like(np.atleast_2d(obsLC))
        np.ndarray[np.double_t, ndim=1] testLC1d=np.zeros_like(obsLC)
        np.ndarray[np.double_t, ndim=2] Asquare=np.zeros((len(tau),len(tau)))
        np.ndarray[np.double_t, ndim=1] origRHS=np.zeros(len(obsLC))
        np.ndarray[np.double_t, ndim=1] windowarr=np.ones_like(tau, dtype=float)
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

    np.savetxt("{0}_chiSquareds.txt".format(filename), RMSs)
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

def foldOpacities(tau):
    """
    Start with a 2D opacity grid and fold opacities over to northern hemisphere
    """
    tau = copy.copy(tau)
    N = np.shape(tau)[0]
    M = np.shape(tau)[1]
    
    for i in range(0, int(np.floor(N/2.))):
        for j in range(0,M):
            if tau[i,j] + tau[N-1-i, j] <= 1.:
                tau[i,j] = tau[i,j] + tau[N-1-i, j]
                tau[N-1-i,j] = 0
                
            else:
                tau[i,j] = 1.
                tau[N-1-i,j] = (tau[i,j] + tau[N-1-i, j]) - 1.

            #if ((roundedtau[i,j]==0.5) & (roundedtau[N-1-i, j]==0.5)):
            #    roundedtau[i,j]=1.
            #    roundedtau[N-1-i,j] = 0.
            
            #if ((roundedtau[i,j]==0.) & (roundedtau[N-1-i, j]==1.)):
            #    roundedtau[i,j]=1.
            #    roundedtau[N-1-i,j] = 0.
                
    #roundedtau=np.abs(roundedtau)            
    return tau

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
            
    
    # Run simultaneous ART according to user's choice of initial grid.
    if initstate=="uniform":  
        raveledtau = simultaneous_ART(n_iter=n_iter, tau_init=0.5*np.ones((N,M)), A=raveledareas, obsLC=obsLC, obsLCerr=obsLCerr, reg=reg, threshold=threshold, filename=filename,window=window)
    elif initstate=="empty":
        raveledtau = simultaneous_ART(n_iter=n_iter, tau_init=np.zeros((N,M)), A=raveledareas, obsLC=obsLC, obsLCerr=obsLCerr, reg=reg, threshold=threshold, filename=filename,window=window)
    elif initstate=="random":
        raveledtau = simultaneous_ART(n_iter=n_iter, tau_init=np.random.uniform(0.,1.,(N,M)), A=raveledareas, obsLC=obsLC, obsLCerr=obsLCerr, reg=reg, threshold=threshold, filename=filename,window=window)
    elif initstate=="lowrestriangle":
        lowrestriangle_init = np.load("//Users/Emily/Documents/Columbia/lightcurve_imaging/lowrestriangle_init.npy")
        raveledtau = simultaneous_ART(n_iter=n_iter, tau_init=lowrestriangle_init, A=raveledareas, obsLC=obsLC, obsLCerr=obsLCerr, reg=reg, threshold=threshold, filename=filename,window=window)
    else: #allow for user to input an initial state matrix
        raveledtau = simultaneous_ART(n_iter=n_iter, tau_init=initstate, A=raveledareas, obsLC=obsLC, obsLCerr=obsLCerr, reg=reg, threshold=threshold, filename=filename,window=window)

    raveledtau = np.reshape(raveledtau,(N,M))
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