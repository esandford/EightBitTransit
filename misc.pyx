# cython: profile=True
from __future__ import division
import numpy as np
from scipy import stats

__all__ = ['lowres_grid_ti','change_res_grid','sigmoid_opacities', 'continuous_opacities', 
'RMS', 'RMS_penalty', 'toBinaryGrid', 'fromBinaryGrid']

def lowres_grid_ti(ti, nside, method='mode', rounding=True):
    """
    Inputs:
    opacitymat = matrix of 0 = transparent, 1 = opaque
    nside = None or number of desired pixels per side (generates a square grid)
    method = 'mode' or 'mean'
    
    Outputs:
    lowres_tau = (nside, (M*nside)/N) matrix of opacities, where each entry is either 0 (transparent) or 1 (opaque)
    """
    
    imshape = np.shape(ti.opacitymat)
    
    diff = np.abs(imshape[0] - imshape[1])
    side1 = diff//2
    side2 = diff - side1
    
    if imshape[0] > imshape[1]: #height > width
        side1 = np.zeros((imshape[0], side1))
        side2 = np.zeros((imshape[0], side2))
        ti.opacitymat = np.hstack((side1,ti.opacitymat,side2))
        
        
    imshape = np.shape(ti.opacitymat)
    
    mside = int(np.round((imshape[1]*nside)/float(imshape[0])))
    
    tau_orig_pos = ti.positions[0]
    
    w = 2./imshape[0]
    
    lowres_tau = np.zeros((nside,mside))
    
    newpix_height = float(imshape[0]*w)/float(nside)
    newpix_width = float(imshape[1]*w)/float(mside)
    
    for (i,j), value in np.ndenumerate(lowres_tau):
        topedge = (tau_orig_pos[0,0,1] + w/2.) - i*newpix_height
        bottomedge = (tau_orig_pos[0,0,1] + w/2.) - (i+1)*newpix_height
        
        leftedge = (tau_orig_pos[0,0,0] - w/2.) + j*newpix_width
        rightedge = (tau_orig_pos[0,0,0] - w/2.) + (j+1)*newpix_width
        
        thisneighborhoodmask = ((tau_orig_pos[:,:,0] > leftedge) & (tau_orig_pos[:,:,0] < rightedge) & (tau_orig_pos[:,:,1] > bottomedge) & (tau_orig_pos[:,:,1] < topedge))
        
        if method=='mode':
            lowres_tau[i,j] = np.round(stats.mode(ti.opacitymat[thisneighborhoodmask],axis=None)[0][0])
        elif method=='mean':
            if rounding==True:
                lowres_tau[i,j] = np.round(np.mean(ti.opacitymat[thisneighborhoodmask]))
            else:
                lowres_tau[i,j] = np.mean(ti.opacitymat[thisneighborhoodmask])
        
    return lowres_tau


def change_res_grid(grid, newdim, rounding=True):
    """
    Takes a grid and outputs a new grid of the same opacity content but different shape, where opacities in 
    the output grid are assigned by subdividing or grouping the original grid pixels, then taking the mean of the resulting
    opacities.
    
    FOR NOW, GRIDS MUST BE SQUARE
    
    Inputs:
    grid = grid of 0s and 1s
    newdim = (height, width) tuple of desired output grid
    rounding = True to round output grid to 0s and 1s.
    
    Outputs:
    highres_grid
    """
    
    N = np.shape(grid)[0]
    M = np.shape(grid)[1]
    
    assert N==M, "input grid must be square"
    
    Nnew = newdim[0]
    Mnew = newdim[1]
    
    assert Nnew==Mnew, "output grid must be square"
    
    #old pixel height and width in "physical units"
    w = 2./N
    
    # positions of centers of pixels in original grid in "physical units" at t=tref
    origpos = np.zeros((N,M,2))
    
    for i in range(1, N+1):
        origpos[i-1,:,1] = 1. - (w/2.) - (i-1.)*w
    
    for j in range(1, M+1):
        #origpos[:,j-1,0] = w*(j - 1. - (M-1.)/2.)
        #above is equivalent, in the case of a square grid (w = 2./M), to:
        origpos[:,j-1,0] = -1. + (w/2.) + (j-1.)*w
    
    
    newGrid = np.zeros((Nnew, Mnew))
    
    #new pixel height and width in "physical units"
    wnew = float(N*w)/float(Nnew)
    
    #high res to low res
    if Nnew < N:
        for (i, j), value in np.ndenumerate(newGrid):
            #edges of new pixels in "physical units"
            topedge = origpos[0,0,1] + w/2. - i*wnew
            bottomedge = origpos[0,0,1] + w/2. - (i+1)*wnew
        
            leftedge = origpos[0,0,0] - w/2. + j*wnew
            rightedge = origpos[0,0,0] - w/2. + (j+1)*wnew        
        
            thisneighborhoodmask = ((origpos[:,:,0] > leftedge) & (origpos[:,:,0] < rightedge) & (origpos[:,:,1] > bottomedge) & (origpos[:,:,1] < topedge))

            if rounding==True:
                newGrid[i,j] = np.round(np.mean(grid[thisneighborhoodmask]))
            else:
                newGrid[i,j] = np.mean(grid[thisneighborhoodmask])
    
    #low res to high res
    elif Nnew > N:
        for (i, j), value in np.ndenumerate(newGrid):
            #edges of new pixels in "physical units"
            topedge = origpos[0,0,1] + w/2. - i*wnew
            bottomedge = origpos[0,0,1] + w/2. - (i+1)*wnew
        
            leftedge = origpos[0,0,0] - w/2. + j*wnew
            rightedge = origpos[0,0,0] - w/2. + (j+1)*wnew
            
            center_y = origpos[0,0,1] + w/2. - i*wnew - wnew/2.
            center_x = origpos[0,0,0] - w/2. + j*wnew + wnew/2.
                
            # for each old pixel, calculate its overlapping area with each new pixel
            overlapareas = np.zeros((N, M, Nnew, Mnew))
            for a in range(0,N):
                for b in range(0,M):
                    #if new left edge > old right edge, overlap = 0
                    if leftedge > origpos[a,b,0] + w/2.:
                        overlapareas[a,b,i,j] = 0.
                    #if new bottom edge > old top edge, overlap = 0
                    elif bottomedge > origpos[a,b,1] + w/2.:
                        overlapareas[a,b,i,j] = 0.
                        
                    #if old left edge > new right edge, overlap = 0
                    elif origpos[a,b,0] - w/2. > rightedge:
                        overlapareas[a,b,i,j] = 0.
                        
                    #if old bottom edge > new top edge, overlap = 0
                    elif origpos[a,b,1] - w/2.> topedge:
                        overlapareas[a,b,i,j] = 0.
                        
                    else:
                        #if new pixel's center is left of old's, overlap width is new right minus old left
                        if center_x < origpos[a,b,0]:
                            overlap_width = rightedge - (origpos[a,b,0] - w/2.)
                        #else, overlap width is old right minus new left
                        else:
                            overlap_width = (origpos[a,b,0] + w/2.) - leftedge
                        
                        #if new pixel's center is below old's, overlap height is new top minus old bottom
                        if center_y < origpos[a,b,1]:
                            overlap_height = topedge - (origpos[a,b,1] - w/2.)
                        #else, overlap height is old top minus new bottom
                        else:
                            overlap_height = (origpos[a,b,1] + w/2.) - bottomedge
                        
                        overlapareas[a,b,i,j] = overlap_width * overlap_height
                        
            #normalize by areas of new pixels
            overlapareas = overlapareas/wnew**2
            
            thisneighborhoodmask = (overlapareas[:,:,i,j] != 0.)
            
            #in going from low to higher res, we want to weight the opacities by the area of pixel overlap so we can "dilute" them appropriately
            if rounding==True:
                newGrid[i,j] = np.round(np.average(grid[thisneighborhoodmask], weights = overlapareas[:,:,i,j][thisneighborhoodmask]))
            else:
                newGrid[i,j] = np.average(grid[thisneighborhoodmask], weights = overlapareas[:,:,i,j][thisneighborhoodmask])
    return newGrid

def sigmoid_opacities(q_ravel,output_shape):
    """
    Takes a vector of continuous opacities and returns a matrix of opacity values between 0 and 1.
    
    Inputs:
    q_ravel = 1D array of shape NMx1
    output_shape = (N,M)
    
    Outputs:
    tau = N x M matrix of  opacity values: tau_ij = 1./(1. + e^(-q_ij))
    """
    
    q = np.reshape(q_ravel, output_shape)
    tau = 1./(1.+np.exp(-1.*q))

    return tau

def continuous_opacities(tau):
    """
    Transform discrete opacity matrix to q matrix (continuous opacity values)

    Inputs:
    tau = N x M matrix of discrete opacity values, where 0 = transparent and 1 = opaque
    
    Outputs:
    q = NM x 1 matrix of transformed opacity values: tau_ij = 1./(1. + e^(-q_ij)). if element tau_ij is 0, 
        then element q_ij is a negative number, and if element tau_ij is 1, then element q_ij is a  
        positive number.
    """
    
    q = np.zeros_like(tau,dtype=float)
    
    lowbound = (1./(1.+np.e))
    highbound = (np.e/(1.+np.e))

    
    continuousmask = (tau >= lowbound) & (tau <= highbound)
    q[continuousmask] = np.log((tau[continuousmask])/(np.ones_like(tau[continuousmask]) - tau[continuousmask]))
    
    near_transparent_mask = (tau >= 0.) & (tau < lowbound)
    q[near_transparent_mask] = -1.0
    
    near_opaque_mask = (tau > highbound) & (tau <= 1.)
    q[near_opaque_mask] = 1.0
    
    #q[opmask] = 1.0 # reverse: opacity_transform(1.) = 0.73. choose s.t. amoeba has a chance of walking in either direction
    #q[~opmask] = -1.0 # reverse: opacity_transform(-1.) = 0.27. choose s.t. amoeba has a chance of walking in either direction

    return np.ravel(q)

def RMS(LC_obs,LC_obs_err,LC_model):
    """
    Calculates chi-squared of model light curve.
    
    Inputs:
    LC_obs = observed light curve
    LC_obs_err = uncertainty on each observed light curve data point (to calculate RMS, make this a vector of ones!)
    LC_model = model light curve (same shape as LC_obs)
    
    Outputs:
    RMS = RMS error
    chisquared = same as RMS in the case where LC_obs_err is a vector of ones.
    """
    #N = len(LC_obs)
    
    chisquared = np.sum((LC_model - LC_obs)**2/LC_obs_err**2)

    return chisquared


def RMS_penalty(grid,LC_obs,LC_model,temperature):
    """
    Calculates RMS error of model light curve.
    
    Inputs:
    LC_obs = observed light curve
    LC_model = model light curve (same shape as LC_obs)
    
    Outputs:
    RMS = RMS error
    """
    N = len(LC_obs)
    
    RMS = (1./np.sqrt(N)) * np.sqrt(np.sum((LC_model - LC_obs)**2)/temperature**2)
    
    RMS = RMS + np.sum(np.abs(grid))
    
    return RMS


def toBinaryGrid(base10number,N,M):
    """
    Takes a base-10 number describing an arrangement of 0s and 1s as input; 
    returns a grid of 0s and 1s of shape N rows by M columns.
    """
    
    binary = str(bin(base10number))[2:].zfill(N*M)
    binary = list(binary)
    binary = [int(j) for j in binary]
    binary = binary[::-1] #reverse it so that it starts from the upper left rather than lower right corner
    binary = np.array(binary).reshape(N,M)
    
    #fig = plt.figure(figsize=(2,2))
    #plt.imshow(binary,cmap="Greys",aspect="equal",origin="upper",interpolation='none',vmin=0,vmax=1)
    #plt.show()
    
    return binary

def fromBinaryGrid(binarygrid):
    """
    Takes a pixel grid of 0s and 1s and outputs a base-10 number describing the arrangement.
    """
    
    binarygrid = np.ravel(binarygrid)[::-1]
    base10number = ''.join(str(int(i)) for i in binarygrid).lstrip('0')
    base10number = int(base10number,2)
    
    return base10number




