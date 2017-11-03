# cython: profile=True
import numpy as np
import copy
from .cTransitingImage import *
from .cGridFunctions import *
from .misc import *


__all__ = ['ART','ART_normal','simultaneous_ART', 'neighborPermutations', 'AStarGeneticStep', 'LCfromSummedPixels', 'perimeter', 'AStarGeneticStep_pixsum']

def ART(tau_init, A, obsLC, mirrored=False, RMSstop=1.e-6):
    """
    Use the algebraic reconstruction technique to solve the system A*tau = np.ones_like(obsLC) - obsLC.
    
    Inputs:
    tau_init = initial guess for the raveled opacity vector tau (shape = NM = k)
    A = matrix of pixel overlap areas (shape = (NM, NM) = (k, k))
    obsLC = observed normalized flux values in the light curve (shape = NM = k)
    mirrored = whether tau is strictly mirrored about the horizontal midplane or not
    RMSstop = stopping criterion
    
    Outputs:
    tau = opacity vector
    """
    tau = np.ravel(tau_init)
    N = np.shape(tau_init)[0]
    M = np.shape(tau_init)[1]
    RHS = np.ones_like(obsLC) - obsLC
    
    RMSarr = np.ones((10,))
    RMSmean = np.mean(RMSarr)
    q = 0
    RMSidx = 0
    
    testLC = np.atleast_2d(np.ones_like(RHS)).T - np.dot(A,np.reshape(tau,(10*10,1)))
    testRMS = RMS(LC_obs=obsLC,LC_model=testLC,temperature=1)
    RMSdiff = 1.
    
    if mirrored is False:
        while RMSmean > RMSstop:
        #for q in range(0, n_iter):
            if q % 10000 == 0:
                testLC = np.atleast_2d(np.ones_like(RHS)).T - np.dot(A,np.reshape(tau,(10*10,1)))
                testRMS = RMS(LC_obs=obsLC,LC_model=testLC,temperature=1)
            
            rowidx = q % (np.shape(A)[0])
            #rowidx = np.random.randint(0,np.shape(A)[0])
        
            dotp = np.dot(A[rowidx], tau)
            denom = (np.linalg.norm(A[rowidx]))**2
        
            tau_update = ((RHS[rowidx] - dotp)/denom) * A[rowidx]
        
            tau = tau + tau_update
            
            if q % 10000 == 0:
                #print q
                testLC = np.atleast_2d(np.ones_like(RHS)).T - np.dot(A,np.reshape(tau,(10*10,1)))
                testRMSp1 = RMS(LC_obs=obsLC,LC_model=testLC,temperature=1)
                RMSdiff = np.abs(testRMSp1 - testRMS)
                RMSarr[RMSidx%10] = RMSdiff
                RMSmean = np.mean(RMSarr)
                RMSidx = RMSidx + 1
                
            
            q = q + 1
            
    else:
        while RMSmean > RMSstop:
        #for q in range(0, n_iter):
            if q % 10000 == 0:
                testLC = np.atleast_2d(np.ones_like(RHS)).T - np.dot(A,np.reshape(tau,(10*10,1)))
                testRMS = RMS(LC_obs=obsLC,LC_model=testLC,temperature=1)
                
            rowidx = q % (np.shape(A)[0])
            #rowidx = np.random.randint(0,np.shape(A)[0])
        
            dotp = np.dot(A[rowidx], tau)
            denom = (np.linalg.norm(A[rowidx]))**2
        
            tau_update = ((RHS[rowidx] - dotp)/denom) * A[rowidx]
            
            tau_update = tau_update.reshape((N,M))
            
            for row in range(int(np.floor(N/2)),N):
                tau_update[row] = tau_update[N-1-row]
                
            tau_update = tau_update.reshape(len(tau))
            
            tau = tau + tau_update
            
            if q % 10000 == 0:
                #print q
                testLC = np.atleast_2d(np.ones_like(RHS)).T - np.dot(A,np.reshape(tau,(10*10,1)))
                testRMSp1 = RMS(LC_obs=obsLC,LC_model=testLC,temperature=1)
                RMSdiff = np.abs(testRMSp1 - testRMS)
                RMSarr[RMSidx%10] = RMSdiff
                RMSmean = np.mean(RMSarr)
                RMSidx = RMSidx + 1
                
            q = q + 1
    
    #print q
    return tau


def ART_normal(tau_init, A, obsLC, reg=0., mirrored=False, RMSstop=1.e-6):
    """
    Use the algebraic reconstruction technique to solve the system A.T * A * tau = A.T * (np.ones_like(obsLC) - obsLC).
    
    Inputs:
    tau_init = initial guess for the raveled opacity vector tau (shape = NM = k)
    A = matrix of pixel overlap areas (shape = (NM, NM) = (k, k))
    obsLC = observed normalized flux values in the light curve (shape = NM = k)
    reg = regularization parameter (scalar)
    mirrored = whether tau is strictly mirrored about the horizontal midplane or not
    RMSstop = stopping criterion
    
    
    Outputs:
    tau = opacity vector
    """
    tau = np.ravel(tau_init)
    N = np.shape(tau_init)[0]
    M = np.shape(tau_init)[1]
    
    if ((np.shape(A)[0] == np.shape(A)[1]) & (reg==0.)):
        RHS = np.ones_like(obsLC) - obsLC
        
    else:
        RHS = np.dot(A.T, np.ones_like(obsLC) - obsLC)
        A = np.dot(A.T, A) + (reg * np.eye(N=np.shape(A)[1]))
    
    RMSarr = np.ones((10,))
    RMSmean = np.mean(RMSarr)
    q = 0
    RMSidx = 0
    
    testLC = np.atleast_2d(np.ones_like(RHS)).T - np.dot(A,np.reshape(tau,(10*10,1)))
    testRMS = RMS(LC_obs=obsLC,LC_model=testLC,temperature=1)
    RMSdiff = 1.
    if mirrored is False:
        while RMSmean > RMSstop:
        #for q in range(0, n_iter):
            if q % 10000 == 0:
                testLC = np.atleast_2d(np.ones_like(RHS)).T - np.dot(A,np.reshape(tau,(10*10,1)))
                testRMS = RMS(LC_obs=obsLC,LC_model=testLC,temperature=1)
            
            rowidx = q % (np.shape(A)[0])
            #rowidx = np.random.randint(0,np.shape(A)[0])
        
            dotp = np.dot(A[rowidx], tau)
            denom = (np.linalg.norm(A[rowidx]))**2
        
            tau_update = ((RHS[rowidx] - dotp)/denom) * A[rowidx]
        
            tau = tau + tau_update
            
            if q % 10000 == 0:
                #print q
                testLC = np.atleast_2d(np.ones_like(RHS)).T - np.dot(A,np.reshape(tau,(10*10,1)))
                testRMSp1 = RMS(LC_obs=obsLC,LC_model=testLC,temperature=1)
                RMSdiff = np.abs(testRMSp1 - testRMS)
                RMSarr[RMSidx%10] = RMSdiff
                RMSmean = np.mean(RMSarr)
                RMSidx = RMSidx + 1
                
            
            q = q + 1

    else:
        while RMSmean > RMSstop:
        #for q in range(0, n_iter):
            if q % 10000 == 0:
                testLC = np.atleast_2d(np.ones_like(RHS)).T - np.dot(A,np.reshape(tau,(10*10,1)))
                testRMS = RMS(LC_obs=obsLC,LC_model=testLC,temperature=1)
                
            rowidx = q % (np.shape(A)[0])
            #rowidx = np.random.randint(0,np.shape(A)[0])
        
            dotp = np.dot(A[rowidx], tau)
            denom = (np.linalg.norm(A[rowidx]))**2
        
            tau_update = ((RHS[rowidx] - dotp)/denom) * A[rowidx]
            
            tau_update = tau_update.reshape((N,M))
            
            for row in range(int(np.floor(N/2)),N):
                tau_update[row] = tau_update[N-1-row]
                
            tau_update = tau_update.reshape(len(tau))
            
            tau = tau + tau_update
            
            if q % 10000 == 0:
                #print q
                testLC = np.atleast_2d(np.ones_like(RHS)).T - np.dot(A,np.reshape(tau,(10*10,1)))
                testRMSp1 = RMS(LC_obs=obsLC,LC_model=testLC,temperature=1)
                RMSdiff = np.abs(testRMSp1 - testRMS)
                RMSarr[RMSidx%10] = RMSdiff
                RMSmean = np.mean(RMSarr)
                RMSidx = RMSidx + 1
                
            q = q + 1
    
    #print q
    return tau

def simultaneous_ART(n_iter, tau_init, A, obsLC):
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
    tau = tau_init
    RHS = np.ones_like(obsLC) - obsLC
    
    for q in range(0, n_iter):
        tau_update = np.zeros_like(tau, dtype=float)
        
        for j in range(0, np.shape(A)[0]):
            outer_numerator = 0. 
            outer_denominator = np.sum(A[:,j])

            for i in range(0, np.shape(A)[0]):
                inner_denominator = np.sum(A[i])
                inner_numerator = (RHS[i] - np.dot(A[i], tau)) * A[i,j]
                outer_numerator = outer_numerator + (inner_numerator/inner_denominator)
            
            tau_update[j] = (outer_numerator/outer_denominator)
            
        tau = tau + tau_update
        
    return tau

def neighborPermutations(grid,grid_i,grid_j,wraparound=False):
    """
    Inputs:
    grid = a grid of 1s and 0s
    grid_i = the row index of a particular pixel
    grid_j = the column index of the same pixel
    
    Returns:
    neighbors = Set of (at maximum) 1024 (=2**(3*3 + 1)) base-10 numbers describing every permutation of this pixel 
    and its 8 immediate neighbors, plus its counterpart across the horizontal midplane, on and off. For small grids, 
    the neighbor across the midplane may be an immediate neighbor; it won't be counted twice, and the output array will 
    be of length 512.
    
    At the edges of the grid, this algorithm will "wrap," so pixels from the opposite edge will be counted as neighbors.
    """
    
    neighbors = set()
    
    N = np.shape(grid)[0]
    M = np.shape(grid)[1]
    #print N,M
    assert grid_i < N, "i out of bounds"
    assert grid_j < M, "j out of bounds"
    
    #find counterpart across the horizontal midplane
    mirror_i = int(((N/2. - (grid_i+0.5)) + N/2.) - 0.5)
    mirror_j = grid_j
    
    #find neighbors
    #vertical
    if grid_i == 0:
        if wraparound is True:
            neighbors_i = [N-1, 0, 1]
        else:
            neighbors_i = [0, 1]
    elif grid_i == N-1:
        if wraparound is True:
            neighbors_i = [N-2, N-1, 0]
        else:
            neighbors_i = [N-2, N-1]
    else:
        neighbors_i = [grid_i - 1, grid_i, grid_i + 1]
    
    #horizontal
    if grid_j == 0:
        if wraparound is True:
            neighbors_j = [M-1, 0, 1]
        else:
            neighbors_j = [0, 1]
    elif grid_j == M-1:
        if wraparound is True:
            neighbors_j = [M-2, M-1, 0]
        else:
            neighbors_j = [M-2, M-1]
    else:
        neighbors_j = [grid_j - 1, grid_j, grid_j + 1]
    
    #print neighbors_i
    #print neighbors_j
    #get indices of these 9 neighbor grid positions into the raveled pixel grid (1D array of length NM)
    ravelidxs = []
    for k in range(0,len(neighbors_i)):
        for m in range(0,len(neighbors_j)):
            thisneighbor_i = neighbors_i[k]
            thisneighbor_j = neighbors_j[m]
            
            ravelidxs.append(thisneighbor_j + M*thisneighbor_i)
    #add in index of the counterpart across the midplane
    ravelidxs.append(mirror_j + M*mirror_i)
    
    #sort these and make sure you're not double-counting the midplane counterpart
    ravelidxs = np.sort(np.unique(np.array(ravelidxs)))
    
    #print ravelidxs
    #Take the raveled pixel grid and substitute in all possible 2**10 (or 2**9) pixel arrangements at the positions
    #specified by ravelidxs.
    
    for i in range(2**(len(ravelidxs))):
        ravelgrid = np.ravel(grid)
        #print ravelgrid
        permute_arr = str(bin(i))[2:].zfill(len(ravelidxs))
        #print permute_arr
        permute_arr = np.array([int(n) for n in permute_arr])
        #print permute_arr
        
        for j in range(0,len(ravelidxs)):
            ravelgrid[ravelidxs[j]] = permute_arr[j]
        
        #print ravelgrid
        #convert to base-10 number
        ravelgrid = ravelgrid[::-1]
        #print ravelgrid
        
        base10number = ''.join(str(int(k)) for k in ravelgrid).lstrip('0')
        
        #print base10number
        #catch case where all are 0, and hence all are stripped
        try:
            base10number = int(base10number,2)
        except ValueError:
            base10number = 0
        
        #print base10number
        #add to "neighbors" set
        neighbors.add(base10number)
        
    return neighbors

def AStarGeneticStep(currentgrid, obsLC, times, temperature=0.01, saveplots=False, filename="astrofestpetridish"):
    """
    Find the grid that best matches obsLC.
    """
    
    N = np.shape(currentgrid)[0]
    M = np.shape(currentgrid)[1]
    #print N, M
    
    currentti = TransitingImage(opacitymat=currentgrid, LDlaw="uniform", v=0.4, t_ref=0., t_arr=rt_times)
    currentLC = currentti.gen_LC(rt_times)   
    currentcost = RMS(currentLC, obsLC, temperature)
    
    numfig=0
    nside=N
    
    if saveplots==True:
        fig, ax = plt.subplots(1,1,figsize=(6,6))
        ax.imshow(currentgrid,cmap="Greys",aspect="equal",origin="upper",interpolation='nearest',vmin=0,vmax=1)
        ax.set_xlim(-0.5,nside-0.5)
        ax.set_ylim(nside-0.5,-0.5)
        ax.set_xticks(np.arange(-0.5,nside+0.5,1));
        ax.set_yticks(np.arange(-0.5,nside+0.5,1));
        ax.axes.get_xaxis().set_ticklabels([])
        ax.axes.get_yaxis().set_ticklabels([])
        #print "saved {0}".format(numfig)
        plt.grid(which='major', color='k', linestyle='-', linewidth=1)
        plt.savefig("./{0}{1}.pdf".format(filename,numfig), fmt="pdf")
    
        numfig+=1
    
    #print currentcost
    
    alreadyEvaluated = set()
    alreadyEvaluated.add(fromBinaryGrid(currentgrid))
    
    #print alreadyEvaluated
    
    #choose random "on" pixel
    onmask = np.ravel((currentgrid > 0.))
    
    onidxs = np.arange(0,len(np.ravel(currentgrid)))[onmask]
    
    randompix = np.random.choice(onidxs,1)[0]
    pixAlready = [randompix]
    
    randompix_i = randompix // N
    randompix_j = randompix % M
    
    #print randompix_i, randompix_j
    
    
    #why isn't the below working for grids bigger than 4x4??
    notYetEvaluated = neighborPermutations(grid=copy.copy(currentgrid),grid_i=randompix_i,grid_j=randompix_j)
    #print notYetEvaluated
    
    while len(notYetEvaluated) > 0:
        for checkBase10 in notYetEvaluated:
            checkGrid = toBinaryGrid(checkBase10,N,M)
            ti = TransitingImage(opacitymat=checkGrid, LDlaw="uniform", v=0.4, t_ref=0., t_arr=times)
            LC = ti.gen_LC(times)   
            cost = RMS(LC, obsLC, temperature)
            
            alreadyEvaluated.add(checkBase10)
            
            if cost == 0:
                currentgrid = checkGrid
                currentcost = cost
                
                if saveplots==True:
                    fig, ax = plt.subplots(1,1,figsize=(6,6))
                    ax.imshow(currentgrid,cmap="Greys",aspect="equal",origin="upper",interpolation='nearest',vmin=0,vmax=1)
                    ax.set_xlim(-0.5,nside-0.5)
                    ax.set_ylim(nside-0.5,-0.5)
                    ax.set_xticks(np.arange(-0.5,nside+0.5,1));
                    ax.set_yticks(np.arange(-0.5,nside+0.5,1));
                    ax.axes.get_xaxis().set_ticklabels([])
                    ax.axes.get_yaxis().set_ticklabels([])
                    plt.grid(which='major', color='k', linestyle='-', linewidth=1)
                    plt.savefig("./{0}{1}.pdf".format(filename,numfig), fmt="pdf")
                    #print "saved {0}".format(numfig)
    
                    numfig+=1
                #print len(alreadyEvaluated)
                return currentgrid, currentcost
            
            elif cost <= currentcost:
                currentgrid = checkGrid
                currentcost = cost
                
                if saveplots==True:
                    fig, ax = plt.subplots(1,1,figsize=(6,6))
                    ax.imshow(currentgrid,cmap="Greys",aspect="equal",origin="upper",interpolation='nearest',vmin=0,vmax=1)
                    ax.set_xlim(-0.5,nside-0.5)
                    ax.set_ylim(nside-0.5,-0.5)
                    ax.set_xticks(np.arange(-0.5,nside+0.5,1));
                    ax.set_yticks(np.arange(-0.5,nside+0.5,1));
                    ax.axes.get_xaxis().set_ticklabels([])
                    ax.axes.get_yaxis().set_ticklabels([])
                    plt.grid(which='major', color='k', linestyle='-', linewidth=1)
                    plt.savefig("./{0}{1}.pdf".format(filename,numfig), fmt="pdf")
                    #print "saved {0}".format(numfig)
    
                    numfig+=1
            
            else:
                pass
        
        #print currentgrid
        #print currentcost
        notYetEvaluated.difference_update(alreadyEvaluated)
        
        #select a new "on" pixel to test
        onmask = np.ravel((currentgrid > 0.))
    
        onidxs = np.arange(0,len(np.ravel(currentgrid)))[onmask]
        
        #if we've already tried all of the choices
        if np.all(np.in1d(onidxs,np.array(pixAlready))):
            #print pixAlready
            #print "try off pixel"
            onidxs = np.arange(0,len(np.ravel(currentgrid)))[~onmask]
        
        randompix = np.random.choice(onidxs,1)[0]
        
        #make sure we haven't already tried this
        while randompix in pixAlready:
            randompix = np.random.choice(onidxs,1)[0]
            #if we've tried every pixel
            if len(pixAlready)==len(np.ravel(currentgrid)):
                #print len(alreadyEvaluated)
                return currentgrid, currentcost
        
        pixAlready.append(randompix)
    
        randompix_i = randompix // N
        randompix_j = randompix % M
        #print randompix_i, randompix_j
        #update with permutations about this new pixel
        newPermutations = neighborPermutations(grid=copy.copy(currentgrid),grid_i=randompix_i,grid_j=randompix_j)
        
        newPermutations.difference_update(alreadyEvaluated)
        notYetEvaluated.update(newPermutations)
    
    #print len(alreadyEvaluated)
    return currentgrid, currentcost

def LCfromSummedPixels(base10number, LCdecrements):
    """
    Takes base10number, figures out which pixels are "on", sums up those pixels' contributions from
    LCdecrements, outputs resulting light curve.
    """

    grid = toBinaryGrid(base10number,N=np.shape(LCdecrements)[0],M=np.shape(LCdecrements)[1])
    
    onpixdecrements = LCdecrements[grid.astype(bool)]
    
    onpixdecrements = np.sum(onpixdecrements,axis=0)
    
    LC = np.ones_like(onpixdecrements) - onpixdecrements
    return LC


def perimeter(grid):
    """
    Calculate the perimeter of a grid arrangement.
    """

    Pc = 0.
    
    N = np.shape(grid)[0]
    M = np.shape(grid)[1]

    onmask = np.ravel((grid > 0.))
    onidxs = np.arange(0,len(np.ravel(grid)))[onmask]
    N_on = len(onidxs)
    
    #vertical
    for i in range (N-1):
        for j in range (M):
            if grid[i,j] > 0. and grid[i+1,j] > 0.:
                Pc += 1.
    #horizontal
    for i in range(N):
        for j in range(M-1):
            if grid[i,j] > 0. and grid[i,j+1] > 0.:
                Pc += 1.

    return ((4.*N_on) + (2.*Pc))

def AStarGeneticStep_pixsum(currentgrid, obsLC, times, temperature=0.01, saveplots=False, filename="astrofestpetridish",costfloor=1.e-10,perimeterFac=0.1):
    """
    Find the grid that best matches obsLC.
    
    Instead of recalculating the light curve from the grid every time, just sum up the light curves produced by individual
    transiting pixels.
    """
    
    N = np.shape(currentgrid)[0]
    M = np.shape(currentgrid)[1]
    
    costList = []
    #get array of flux decrements due to each individual pixel being "on"
    LCdecrements = np.zeros((N, M, len(times)))
    
    for i in range(N):
        for j in range(M):
            onepixgrid = np.zeros((N,M),dtype=int)
            onepixgrid[i,j] = 1
            onepix_ti = TransitingImage(opacitymat=onepixgrid, LDlaw="uniform", v=0.4, t_ref=0., t_arr=times)
            onepix_LC = onepix_ti.gen_LC(times)
            onepixcost = RMS(onepix_LC,obsLC,temperature) + perimeterFac*perimeter(onepixgrid)/(N*M) #(100.*(1./(N*M)))
            if onepixcost <= costfloor:
                costList.append(onepixcost)
                return onepixgrid, onepixcost,costList
            
            LCdecrements[i,j,:] = np.ones_like(onepix_LC) - onepix_LC
    
    currentLC = LCfromSummedPixels(fromBinaryGrid(currentgrid),LCdecrements)  
    onmask = np.ravel((currentgrid > 0.))
    onidxs = np.arange(0,len(np.ravel(currentgrid)))[onmask]
    currentcost = RMS(currentLC, obsLC, temperature) + perimeterFac*perimeter(currentgrid)/(N*M) #(100.*(len(onidxs)/(N*M)))
    costList.append(currentcost)
    
    numfig=0
    nside=N
    
    if saveplots==True:
        fig, ax = plt.subplots(1,1,figsize=(6,6))
        ax.imshow(currentgrid,cmap="Greys",aspect="equal",origin="upper",interpolation='nearest',vmin=0,vmax=1)
        ax.set_xlim(-0.5,nside-0.5)
        ax.set_ylim(nside-0.5,-0.5)
        ax.set_xticks(np.arange(-0.5,nside+0.5,1));
        ax.set_yticks(np.arange(-0.5,nside+0.5,1));
        ax.axes.get_xaxis().set_ticklabels([])
        ax.axes.get_yaxis().set_ticklabels([])
        #print "saved {0}".format(numfig)
        plt.grid(which='major', color='k', linestyle='-', linewidth=1)
        plt.savefig("./{0}{1}.pdf".format(filename,numfig), fmt="pdf")
    
        numfig+=1
    
    #print currentcost
    
    alreadyEvaluated = set()
    alreadyEvaluated.add(fromBinaryGrid(currentgrid))
    
    #print alreadyEvaluated
    
    #choose random "on" pixel
    onmask = np.ravel((currentgrid > 0.))
    
    onidxs = np.arange(0,len(np.ravel(currentgrid)))[onmask]
    
    randompix = np.random.choice(onidxs,1)[0]
    pixAlready = [randompix]
    
    randompix_i = randompix // N
    randompix_j = randompix % M
    
    #print randompix_i, randompix_j
    
    
    #why isn't the below working for grids bigger than 4x4??
    notYetEvaluated = neighborPermutations(grid=copy.copy(currentgrid),grid_i=randompix_i,grid_j=randompix_j)
    #print notYetEvaluated
    
    while len(notYetEvaluated) > 0:
        for checkBase10 in notYetEvaluated:
            checkGrid = toBinaryGrid(checkBase10,N,M)
            
            LC = LCfromSummedPixels(checkBase10,LCdecrements)
            checkonmask = np.ravel((checkGrid > 0.))
            checkonidxs = np.arange(0,len(np.ravel(checkGrid)))[checkonmask]
            cost = RMS(LC, obsLC, temperature) + perimeterFac*perimeter(checkGrid)/(N*M) #(100.*(len(checkonidxs)/(N*M)))
            
            alreadyEvaluated.add(checkBase10)
            
            if cost <= costfloor:
                currentgrid = checkGrid
                currentcost = cost
                costList.append(currentcost)
                
                if saveplots==True:
                    fig, ax = plt.subplots(1,1,figsize=(6,6))
                    ax.imshow(currentgrid,cmap="Greys",aspect="equal",origin="upper",interpolation='nearest',vmin=0,vmax=1)
                    ax.set_xlim(-0.5,nside-0.5)
                    ax.set_ylim(nside-0.5,-0.5)
                    ax.set_xticks(np.arange(-0.5,nside+0.5,1));
                    ax.set_yticks(np.arange(-0.5,nside+0.5,1));
                    ax.axes.get_xaxis().set_ticklabels([])
                    ax.axes.get_yaxis().set_ticklabels([])
                    plt.grid(which='major', color='k', linestyle='-', linewidth=1)
                    plt.savefig("./{0}{1}.pdf".format(filename,numfig), fmt="pdf")
                    #print "saved {0}".format(numfig)
    
                    numfig+=1
                #print len(alreadyEvaluated)
                return currentgrid, currentcost, costList
            
            elif cost <= currentcost:
                currentgrid = checkGrid
                currentcost = cost
                costList.append(currentcost)
                
                if saveplots==True:
                    fig, ax = plt.subplots(1,1,figsize=(6,6))
                    ax.imshow(currentgrid,cmap="Greys",aspect="equal",origin="upper",interpolation='nearest',vmin=0,vmax=1)
                    ax.set_xlim(-0.5,nside-0.5)
                    ax.set_ylim(nside-0.5,-0.5)
                    ax.set_xticks(np.arange(-0.5,nside+0.5,1));
                    ax.set_yticks(np.arange(-0.5,nside+0.5,1));
                    ax.axes.get_xaxis().set_ticklabels([])
                    ax.axes.get_yaxis().set_ticklabels([])
                    plt.grid(which='major', color='k', linestyle='-', linewidth=1)
                    plt.savefig("./{0}{1}.pdf".format(filename,numfig), fmt="pdf")
                    #print "saved {0}".format(numfig)
    
                    numfig+=1
            
            else:
                pass
        
        #print currentgrid
        #print currentcost
        notYetEvaluated.difference_update(alreadyEvaluated)
        
        #select a new "on" pixel to test
        onmask = np.ravel((currentgrid > 0.))
    
        onidxs = np.arange(0,len(np.ravel(currentgrid)))[onmask]
        
        #if we've already tried all of the choices
        if np.all(np.in1d(onidxs,np.array(pixAlready))):
            #print pixAlready
            #print "try off pixel"
            onidxs = np.arange(0,len(np.ravel(currentgrid)))[~onmask]
        
        randompix = np.random.choice(onidxs,1)[0]
        
        #make sure we haven't already tried this
        while randompix in pixAlready:
            randompix = np.random.choice(onidxs,1)[0]
            #if we've tried every pixel
            if len(pixAlready)==len(np.ravel(currentgrid)):
                #print len(alreadyEvaluated)
                return currentgrid, currentcost, costList
        
        pixAlready.append(randompix)
    
        randompix_i = randompix // N
        randompix_j = randompix % M
        #print randompix_i, randompix_j
        #update with permutations about this new pixel
        newPermutations = neighborPermutations(grid=copy.copy(currentgrid),grid_i=randompix_i,grid_j=randompix_j)
        
        newPermutations.difference_update(alreadyEvaluated)
        notYetEvaluated.update(newPermutations)
    
    #print len(alreadyEvaluated)
    return currentgrid, currentcost, costList