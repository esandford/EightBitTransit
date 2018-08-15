# cython: profile=True
from __future__ import division
import numpy as np
import time
import copy
from scipy import misc, stats
from collections import Counter

__all__ = ['positions', 'pixelate_image', 'lowres_grid', 'pixeloverlaparea', 'LDfluxsmall']

def positions(n, m, t, tref, v):
	"""
	Calculates the (x,y) position of the n x m pixel grid at each time t.
	
	Inputs:
	n = height of grid, in pixels
	m = width of grid, in pixels
	t = array of times at which to output pixel grid positions
	tref = reference transit midpoint time [time units]
	v = grid velocity [1/(time units)] (since distance is in units of R*)
	
	Outputs:
	pos = array of shape (len(t), n, m, 2) containing thepositions of pixel (n,m) at each time t. pos[k,:,:,0] are
		the x-positions of the whole grid at time t=k; pos[k,:,:,1] are the y-positions.
	"""
	pos = np.zeros((len(t),n,m,2))
	w = (2./n)
	
	#y positions are constant in time
	#y = np.zeros(n)
	for i in range(1,n+1):
		#y[i] = 1. - (w/2.) - (i-1.)*w
		pos[:,i-1,:,1] = 1. - (w/2.) - (i-1.)*w
		
	#x positions are time-evolving
	jmid = 1. + (m-1.)/2.
	xmidref = 0. #x position of reference pixel at tref
	
	xref = np.zeros(m)
	#at time t=tref:
	for j in range(1,m+1):
		xref[j-1] = xmidref + (j-jmid)*w
	
	#pixels shift linearly in time
	for k in range(0,len(t)):
		for j in range(1,m+1):
			pos[k,:,j-1,0] = xref[j-1] + (t[k] - tref)*v
	
	tMin = tref - (2. + w*(m-1))/(2.*v)
	tMax = tref + (2. + w*(m-1))/(2.*v)

	overlappingTimesMask = (t > tMin) & (t < tMax)

	overlappingTimes = t[overlappingTimesMask]
	overlappingPos = pos[overlappingTimesMask]

	return overlappingPos, overlappingTimes

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
	
	imgrid = misc.imread(imfile)
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
		tau_orig_pos = positions(n=imshape[0],m=imshape[1],t=np.atleast_1d(np.array((0))),tref=0,v=0)[0]
		
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

cpdef pixeloverlaparea(double x0, double y0, double w):
	"""
	Calculates the area of overlap of a square pixel at centered at (x0, y0) of width w with the stellar disk.
	
	Inputs:
	x0 = x-coordinate of pixel center
	y0 = y-coordinate of pixel center
	w = pixel width
	
	Outputs:
	area = approximate area of overlap between pixel and stellar disk
	"""
	cdef:
		double rsq, chord, area, norm_area, unobscured_area, edgedist, h, triarea, xd00, xd01, triarea0, triarea1, chord0, chord1, yd00, yd01
		double xarr[8]
		double yarr[8]
		double verts[8]
		double sqrt2=1.41421356
		int i, jj, num_intersections
		list xg, yg, vg
	
	#radial distance squared minus 1
	rsq = x0**2 + y0**2 - 1.
	
	#calculate area of overlap between pixel and sky-projected stellar disk
	if rsq > ((sqrt2*w)):
		#pixel is guaranteed to be fully outside the star
		area = 0.0
		#print "outside"
	
	elif rsq < (-1.*sqrt2*w):
		#pixel is guaranteed to be fully inside the star
		area = w**2
		#print "inside"

	else:
		#pixel overlap possible, but not guaranteed
		
		#possible intersections:
		# 0: x0**2 + (y0 + w/2)**2 = 1. (top side of pixel intersects)
		# 1: negative root of same
		# 2: (x0 + w/2)**2 + y0**2 = 1. (right side of pixel intersects)
		# 3: negative root of same
		# 4: x0**2 + (y0 - w/2)**2 = 1. (bottom side of pixel intersects)
		# 5: negative root of same
		# 6: (x0 - w/2)**2 + y0**2 = 1. (left side of pixel intersects)
		# 7: negative root of same
		
		#calculate x-coordinates of intersection solutions
		#xarr = np.zeros(8)
		xarr[0] = ((1. - 0.5*w - y0)*(1. + 0.5*w + y0))**0.5
		xarr[1] = -1.*xarr[0]
		
		xarr[2] = x0 + 0.5*w
		xarr[3] = xarr[2]
		
		xarr[4] = -1.*((1. + 0.5*w - y0)*(1. - 0.5*w + y0))**0.5
		xarr[5] = -1.*xarr[4]
		
		xarr[6] = x0 - 0.5*w
		xarr[7] = xarr[6]
	
		#calculate y-coordinates of intersection solutions
		#yarr = np.zeros(8)
		yarr[0] = y0 + 0.5*w
		yarr[1] = yarr[0]
		
		yarr[2] = -1.*((1. - 0.5*w - x0)*(1. + 0.5*w + x0))**0.5
		yarr[3] = -1.*yarr[2]
		
		yarr[4] = y0 - 0.5*w
		yarr[5] = yarr[4]
		
		yarr[6] = ((1. + 0.5*w - x0)*(1. - 0.5*w + x0))**0.5
		yarr[7] = -1.*yarr[6]
		
		#label vertices as 1 = top, 2 = right, 3 = bottom, 4 = left
		#verts = np.zeros(8)
		for i in range(1,9):
			verts[i-1] = float(np.ceil(0.5*i))
		
		#count up valid solutions
		xg = []
		yg = []
		vg = []
		num_intersections = 0
		for jj in range(0,8):
			#see if the distance between the pixel center and the intersection point is less than or equal to
			# w/sqrt(2), which is the distance between the pixel center and any corner
			if ((xarr[jj] - x0)**2 + (yarr[jj] - y0)**2) <= (0.5*w**2):
				num_intersections +=1
				xg.append(xarr[jj])
				yg.append(yarr[jj])
				vg.append(verts[jj])
		
		#print num_intersections
		#print xg
		#print yg
		#print vg
		if num_intersections==2: #pixel partially overlaps with stellar disk
			#print "2 called"
			if (vg[0]%2 == vg[1]%2) and (vg[0] != vg[1]): #either the points of intersection are top/bottom sides of pixel, or they are left/right
				#calculate overlap area as a trapezoid
				if vg[0]%2 == 1: #points of intersection are in top/bottom
					edgedist = x0 - (0.5*w)*np.sign(x0) #distance from x = 0 to nearest vertical edge of box
					area = 0.5*w*abs(xg[0] + xg[1] - 2*edgedist)
				else: #left/right
					edgedist = y0 - (0.5*w)*np.sign(y0) #distance from y = 0 to nearest horizontal edge of box
					area = 0.5*w*abs(yg[0] + yg[1] - 2*edgedist)
			
			elif (vg[0]%2 == vg[1]%2) and (vg[0] == vg[1]): #points of intersection are both on the same face, e.g. for the middle pixel of an odd-N pixel grid at first ingress
				area = 0. #will add chord correction later.
			
			elif (vg[0]%2 != vg[1]%2): #calculate overlap area as a triangle
				#print "triangle"
				area = 0.5*abs(xg[1] - xg[0])*abs(yg[1] - yg[0])
				chord = ((xg[1] - xg[0])**2 + (yg[1] - yg[0])**2)**0.5
				h = 1. - (1. - (chord**2/4.))**0.5 # 1**2 = (chord/2)**2 + (1-h)**2

				if ((x0**2 + y0**2)**0.5 < (1. - h)):
				#	print "x0,y0: ({0},{1})".format(x0,y0)
				#	print "mostly inside"
					#most of pixel is inside the circle
					area = w**2 - area
			
			#append area with arc correction
			chord = ((xg[1] - xg[0])**2 + (yg[1] - yg[0])**2)**0.5
			area = area + np.arcsin(0.5*chord) - 0.5*np.sin(2.*np.arcsin(0.5*chord))

		elif num_intersections==3:
			#print "3 called"
			#possibilities:
			#    left, top, right     = 4, 1, 2. verts_mod2 = 0, 1, 0
			#    left, bottom, right  = 4, 3, 2. verts_mod2 = 0, 1, 0
			#    bottom, left, top    = 3, 4, 1. verts_mod2 = 1, 0, 1
			#    bottom, right, top   = 3, 2, 1. verts_mod2 = 1, 0, 1
			
			#catch cases where pixel corner touches edge of stellar disk (happens for pixels touching the midplane, if N is even)
			corners = np.zeros(3)
			for i in range(0,3):
				if (abs(xg[i]) == 1.0) and (abs(yg[i]) == 0.0):
					corners[i] = 1.
			
			if np.sum(corners)==3.:
				area = 0.
				
			else:
				verts_mod2 = [v%2 for v in vg]
			
				if np.sum(verts_mod2) == 1: #left, top, right or left, bottom, right
					middleidx = np.arange(3)[np.array(verts_mod2) == 1]
					outeridx = np.arange(3)[np.array(verts_mod2) == 0]
				
				elif np.sum(verts_mod2) == 2:
					middleidx = np.arange(3)[np.array(verts_mod2) == 0]
					outeridx = np.arange(3)[np.array(verts_mod2) == 1]
			
				area = w**2
			
				middleidx = int(middleidx)
			
				for idx in outeridx:
					triarea = 0.5*abs(xg[idx] - xg[middleidx])*abs(yg[idx] - yg[middleidx])
				
					chord = ((xg[idx] - xg[middleidx])**2 + (yg[idx] - yg[middleidx])**2)**0.5
					h = 1. - (1. - (chord**2/4.))**0.5
					if ((x0**2 + y0**2)**0.5 < (1. - h)):
						#most of pixel is inside the circle
						triarea = -1*triarea
					
					area += triarea
					area += np.arcsin(0.5*chord) - 0.5*np.sin(2.*np.arcsin(0.5*chord))
			
		elif num_intersections==4:
			#print "4 called"
			#possibilities:
			#    left, top, top, right        = 4, 1, 1, 2
			#    left, bottom, bottom, right  = 4, 3, 3, 2
			#    bottom, left, left, top      = 3, 4, 4, 1
			#    bottom, right, right, top    = 3, 2, 2, 1
			
			middlevert = Counter(vg).most_common(3)[0][0]

			middleidxs = np.arange(4)[np.array(vg) == middlevert]
			outeridxs = np.arange(4)[~(np.array(vg) == middlevert)]
			
			area = w**2

			if middlevert%2 == 1: #middle intersection is on top or bottom
				outeridx0 = outeridxs[0]
				outeridx1 = outeridxs[1]
				
				middleidx0 = middleidxs[0]
				middleidx1 = middleidxs[1]
				
				xd00 = abs(xg[outeridx0] - xg[middleidx0])
				xd01 = abs(xg[outeridx0] - xg[middleidx1])
				
				if xd00 <= xd01:
					triarea0 = 0.5*xd00*abs(yg[outeridx0] - yg[middleidx0])
					chord0 = (xd00**2 + (yg[outeridx0] - yg[middleidx0])**2)**0.5
					#print triarea0
					#print chord0
					area = area - triarea0 + np.arcsin(0.5*chord0) - 0.5*np.sin(2.*np.arcsin(0.5*chord0))
					
					triarea1 = 0.5*abs(xg[outeridx1] - xg[middleidx1])*abs(yg[outeridx1] - yg[middleidx1])
					chord1 = ((xg[outeridx1] - xg[middleidx1])**2 + (yg[outeridx1] - yg[middleidx1])**2)**0.5
					area = area - triarea1 + np.arcsin(0.5*chord1) - 0.5*np.sin(2.*np.arcsin(0.5*chord1))
					#print triarea1
					#print chord1
				else:
					triarea0 = 0.5*xd01*abs(yg[outeridx0] - yg[middleidx1])
					chord0 = (xd01**2 + (yg[outeridx0] - yg[middleidx1])**2)**0.5
					area = area - triarea0 + np.arcsin(0.5*chord0) - 0.5*np.sin(2.*np.arcsin(0.5*chord0))
					#print triarea0
					triarea1 = 0.5*abs(xg[outeridx1] - xg[middleidx0])*abs(yg[outeridx1] - yg[middleidx0])
					chord1 = ((xg[outeridx1] - xg[middleidx0])**2 + (yg[outeridx1] - yg[middleidx0])**2)**0.5
					area = area - triarea1 + np.arcsin(0.5*chord1) - 0.5*np.sin(2.*np.arcsin(0.5*chord1))
					#print triarea1
					
				#print area
				
			else: #middle intersection is on left or right
				outeridx0 = outeridxs[0]
				outeridx1 = outeridxs[1]
				
				middleidx0 = middleidxs[0]
				middleidx1 = middleidxs[1]
				
				yd00 = abs(yg[outeridx0] - yg[middleidx0])
				yd01 = abs(yg[outeridx0] - yg[middleidx1])
				
				if yd00 <= yd01:
					triarea0 = 0.5*yd00*abs(xg[outeridx0] - xg[middleidx0])
					chord0 = (yd00**2 + (xg[outeridx0] - xg[middleidx0])**2)**0.5
					#print triarea0
					area = area - triarea0 + np.arcsin(0.5*chord0) - 0.5*np.sin(2.*np.arcsin(0.5*chord0))
					
					triarea1 = 0.5*abs(xg[outeridx1] - xg[middleidx1])*abs(yg[outeridx1] - yg[middleidx1])
					chord1 = ((xg[outeridx1] - xg[middleidx1])**2 + (yg[outeridx1] - yg[middleidx1])**2)**0.5
					area = area - triarea1 + np.arcsin(0.5*chord1) - 0.5*np.sin(2.*np.arcsin(0.5*chord1))
					#print triarea1
				else:
					triarea0 = 0.5*yd01*abs(xg[outeridx0] - xg[middleidx1])
					chord0 = (yd01**2 + (xg[outeridx0] - xg[middleidx1])**2)**0.5
					area = area - triarea0 + np.arcsin(0.5*chord0) - 0.5*np.sin(2.*np.arcsin(0.5*chord0))
					#print triarea0
					triarea1 = 0.5*abs(xg[outeridx1] - xg[middleidx0])*abs(yg[outeridx1] - yg[middleidx0])
					chord1 = ((xg[outeridx1] - xg[middleidx0])**2 + (yg[outeridx1] - yg[middleidx0])**2)**0.5
					area = area - triarea1 + np.arcsin(0.5*chord1) - 0.5*np.sin(2.*np.arcsin(0.5*chord1))
					#print triarea1
				
		elif num_intersections==6: #only called in the case of a 1x1 grid
			#eliminate intersections at the top and bottom of the stellar disk, which add no information
			non_tb_x = []
			non_tb_y = []

			if w == 2.: #one-row grid
				for i in range(0,6):
					if (abs(xg[i]) == 0.0) and (abs(yg[i]) == 1.0):
						pass
					else:
						non_tb_x.append(xg[i])
						non_tb_y.append(yg[i])

				non_tb_x = np.array(non_tb_x)
				non_tb_y = np.array(non_tb_y)

				if (len(non_tb_x) == 0) & (len(non_tb_y)==0):
					if ((x0**2 + y0**2) < 1.): #pixel is fully inside stellar disk, so the occulted area is the area of the star
						#print "called"
						area = np.pi
					elif ((x0**2 + y0**2) == 1.): #pixel is exactly halfway inside stellar disk, so the occulted area is half the area of the star
						area = np.pi/2.
					else:
						area = 0.

				else:
					chord = ((non_tb_x[1] - non_tb_x[0])**2 + (non_tb_y[1] - non_tb_y[0])**2)**0.5
					unobscured_area = np.arcsin(0.5*chord) - 0.5*np.sin(2.*np.arcsin(0.5*chord))
					if ((x0**2 + y0**2) < 1.):
						return 1.0 - (unobscured_area/np.pi)
					else:
						area = 0.

			elif w == 1.: #two-row grid
				tbandside = [0,0]

				for i in range(0,6):
					if (abs(xg[i]) == 0.0) and (abs(yg[i]) == 1.0):
						tbandside[0] = 1
					if (abs(xg[i]) == 1.0) and (abs(yg[i]) == 0.0):
						tbandside[1] = 1

				if sum(tbandside) == 2:
					area = np.pi/4.

				else:
					if ((x0**2 + y0**2) < 1.): #pixel is fully inside stellar disk
						area = w**2
					else:
						area = 0.


		else:    #no valid intersection points found
			if ((x0**2 + y0**2) < 1.): #pixel is fully inside stellar disk
				area = np.min([w**2,np.pi]) #to account for the case of a one-row grid
		
			else: #pixel is fully outside stellar disk
				area = 0.
	
	norm_area = area/np.pi #normalize by stellar area, which is pi*1**2
	#print norm_area
	return norm_area

cpdef LDfluxsmall(x, y, t, Ar_occ, double c1, double c2, double c3, double c4, double w):
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

	cdef:
		double Ftot, r, S, am, bm, amR, bmR, Ar_ann, Fl_ann, pi=3.1415926535
		int n, i
	
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
	
