# cython: profile=True
from __future__ import division
import numpy as np
import copy
from scipy import misc, stats
from imageio import imread
from collections import Counter
from .pixeloverlap import positions
__all__ = ['pixelate_image', 'lowres_grid', 'LDfluxsmall']


def pixelate_image(imfile, nside, method='mode', rounding=False):
    """
    Inputs:
    imfile = image file
    nside = None or number of desired pixels per side (generates a square grid)
    #imgrid = (N, M, 3) RGB matrix describing image

    Outputs:
    if nside = None:
        tau = (N, M) matrix of opacities, where each entry is either 0 (transparent) or 1 (opaque)
    else:
        tau = (nside, (M*nside)/N) matrix of opacities, where each entry is either 0 (transparent) or 1 (opaque)
    """

    imgrid = imread(imfile)
    imgrid = imgrid[:,:,0:3]
    if nside is None:
        #print np.shape(imgrid[:,:,0])
        tau = np.ones_like(imgrid[:,:,0]) - (np.sqrt((np.sum((imgrid/256.)**2,axis=2))/3.))
        #return tau
    else:
        imshape = np.shape(imgrid[:,:,0])
        #print imshape
        diff = np.abs(imshape[0] - imshape[1])
        side1 = diff//2
        side2 = diff - side1

        if imshape[0] > imshape[1]: #height > width
            side1 = np.ones((imshape[0], side1, 3))*255
            side2 = np.ones((imshape[0], side2, 3))*255
            imgrid = np.hstack((side1,imgrid,side2))

        #else: #width > height
            #side1 = np.ones((side1, imshape[1], 3))*255
            #side2 = np.ones((side2, imshape[1], 3))*255
            #imgrid = np.vstack((side1,imgrid,side2))

        imshape = np.shape(imgrid[:,:,0])

        mside = int(np.round((imshape[1]*nside)/float(imshape[0])))

        tau_orig = np.ones_like(imgrid[:,:,0]) - (np.sqrt((np.sum((imgrid/256.)**2,axis=2))/3.))
        tau_orig_pos = positions(n=imshape[0],m=imshape[1],t=np.atleast_1d(np.array((0))),tref=0,v=0.)[0][0]

        w = 2./imshape[0]

        tau = np.zeros((nside,mside))

        newpix_height = float(imshape[0]*w)/float(nside)
        newpix_width = float(imshape[1]*w)/float(mside)

        for (i,j), value in np.ndenumerate(tau):
            topedge = (tau_orig_pos[0,0,1] + w/2.) - i*newpix_height
            bottomedge = (tau_orig_pos[0,0,1] + w/2.) - (i+1)*newpix_height

            leftedge = (tau_orig_pos[0,0,0] - w/2.) + j*newpix_width
            rightedge = (tau_orig_pos[0,0,0] - w/2.) + (j+1)*newpix_width

            thisneighborhoodmask = ((tau_orig_pos[:,:,0] > leftedge) & (tau_orig_pos[:,:,0] < rightedge) & (tau_orig_pos[:,:,1] > bottomedge) & (tau_orig_pos[:,:,1] < topedge))

            if method=='mode':
                tau[i,j] = np.round(stats.mode(tau_orig[thisneighborhoodmask],axis=None)[0][0])
            elif method=='mean':
                if rounding==True:
                    tau[i,j] = np.round(np.mean(tau_orig[thisneighborhoodmask]))
                else:
                    tau[i,j] = np.mean(tau_orig[thisneighborhoodmask])

    tau[tau <= 0.004] = 0.
    return tau

def lowres_grid(opacitymat, positions, nside, method='mean', rounding=False):
    """
    Inputs:
    opacitymat = matrix of 0 = transparent, 1 = opaque
    nside = None or number of desired pixels per side (generates a square grid)
    method = 'mode' or 'mean'
    rounding = if method is mean, round each big pixel opacity to 0 or 1. otherwise, allow for continuous opacities between 0 and 1.

    Outputs:
    lowres_tau = (nside, (M*nside)/N) matrix of opacities, where each entry is either 0 (transparent) or 1 (opaque)
    """

    imshape = np.shape(opacitymat)

    diff = np.abs(imshape[0] - imshape[1])
    side1 = diff//2
    side2 = diff - side1

    if imshape[0] > imshape[1]: #height > width
        side1 = np.zeros((imshape[0], side1))
        side2 = np.zeros((imshape[0], side2))
        opacitymat = np.hstack((side1,opacitymat,side2))

    imshape = np.shape(opacitymat)

    mside = int(np.round((imshape[1]*nside)/float(imshape[0])))

    tau_orig_pos = positions[0]

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
            lowres_tau[i,j] = np.round(stats.mode(opacitymat[thisneighborhoodmask],axis=None)[0][0])
        elif method=='mean':
            if rounding==True:
                lowres_tau[i,j] = np.round(np.mean(opacitymat[thisneighborhoodmask]))
            else:
                lowres_tau[i,j] = np.mean(opacitymat[thisneighborhoodmask])

    return lowres_tau

def LDfluxsmall(x, y, t, Ar_occ, c1, c2, c3, c4, w):
    """
    computes limb-darkened flux assuming small-planet approximation
    Inputs:
    x = array of pixel center x-coordinates at each time t, same shape as t
    y = array of pixel center y-coordinates at each time t, same shape as t
    t = array of times at which to calculate the flux
    w = pixel width
    Ar_occ = array of areas of star occulted by pixel at each time t; same shape as t
    c1, c2, c3, c4 = LDCs


    Outputs:
    I0 = array of relative flux obscured by each pixel; same shape as t
    """

    pi=3.1415926535

    Ftot = 1. - 0.2*c1 - (1./3.)*c2 - (3./7.)*c3 - 0.5*c4

    r = w/(pi**0.5) #set r such that area = w^2= pi*r^2 => r = w/sqrt(pi), i.e. r is the radius of the circle with area w**2

    n = len(t)

    #Ar_ann = np.zeros(n) #area of annulus
    #Fl_ann = np.zeros(n) #flux within annulus
    I0 = np.zeros(n)

    for i in range(0,n):
        S = (x[i]**2 + y[i]**2)**0.5 #distance from stellar center
        am = (S - r)**2 #inner part of annulus, centered at stellar center, which contains this pixel
        bm = (S + r)**2 #outer part of annulus, centered at stellar center, which contains this pixel

        amR = (1. - am)**0.25
        bmR = (1. - bm)**0.25

        if S > (1. + r): #case I: pixel is outside of stellar disk
            Ar_ann = 0.
            Fl_ann = 0.
            I0[i] = 0.

        elif (S > r) and (S < (1.-r)): #case III: pixel fully overlaps stellar disk
            Ar_ann = pi*(bm - am)
            Fl_ann = (am - bm)*(c1 + c2 + c3 + c4 - 1.) + 0.8*c1*amR**5 + (2./3.)*c2*amR**6 + (4./7.)*c3*amR**7 + 0.5*c4*amR**8 - 0.8*c1*bmR**5 - (2./3.)*c2*bmR**6 -(4./7.)*c3*bmR**7 - 0.5*c4*bmR**8
            I0[i] = pi*(Ar_occ[i]/Ar_ann)*(Fl_ann/Ftot)
            #flux blocked by this pixel = pi*(area occulted by pixel/area of annulus)*(flux in annulus/flux of whole star)
            # I think this is the "effective area" of the pixel for the purposes of inversion with non-uniform LD
        elif (S < r): #case IV: pixel is very close to stellar center
            Ar_ann = pi*bm		
            Fl_ann = -1.*bm*(c1 + c2 + c3 + c4 - 1.) + 0.8*c1 + (2./3.)*c2 + (4./7.)*c3 + 0.5*c4 -0.8*c1*bmR**5 - (2./3.)*c2*bmR**6 - (4./7.)*c3*bmR**7 - 0.5*c4*bmR**8
            I0[i] = pi*(Ar_occ[i]/Ar_ann)*(Fl_ann/Ftot)


        else: #if S[i] > (1.-r) and S[i] < (1.+r), case II: pixel overlaps edge of stellar disk
            Ar_ann = pi*(1.-am)
            Fl_ann = (am - 1.)*(c1 + c2 + c3 + c4 - 1.) + 0.8*c1*amR**5 + (2./3.)*c2*amR**6 +(4./7.)*c3*amR**7 + 0.5*c4*amR**8
            I0[i] = pi*(Ar_occ[i]/Ar_ann)*(Fl_ann/Ftot)


    return I0

