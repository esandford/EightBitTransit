# cython: profile=True
from __future__ import division
import numpy as np
import time
import copy
import matplotlib.pyplot as plt
from scipy import misc, stats
from .cGridFunctions import *

__all__ = ['TransitingImage']

class TransitingImage(object):
	def __init__(self,**kwargs):
		
		#all initial values are None
		self.imfile = None
		self.lowres = None
		self.lowrestype = "mean" #also allowed: "mode"
		self.lowresround = False #also allowed: True
		self.opacitymat = None
		self.LDlaw = "uniform" #also allowed: "nonlinear"
		self.LDCs = None
		self.positions = None
		self.areas = None
		self.blockedflux = None
		self.LD = None
		
		allowed_keys = ["imfile","lowres","lowrestype","lowresround","opacitymat","v","t_ref","LDlaw","LDCs", "t_arr", "positions", "areas"]
		
		#update with values passed in kwargs. values not passed in kwargs will remain None
		self.__dict__.update((k,v) for k,v in kwargs.iteritems() if k in allowed_keys)
		
		#check for required kwargs
		if not (("imfile" in kwargs) or ("opacitymat" in kwargs)):
			raise Exception("Must initialize TransitingImage object with either imfile or opacitymat")
		
		if not ("v" in kwargs):
			raise Exception("Must initialize TransitingImage object with velocity v")
			
		if not ("t_ref" in kwargs):
			raise Exception("Must initialize TransitingImage object with reference time t_ref")
			
		if not ("t_arr" in kwargs):
			raise Exception("Must initialize TransitingImage object with time array t_arr")
		
		if self.LDlaw == "linear":
			if not ("LDCs" in kwargs):
				raise Exception("Must specify array of 1 limb-darkening coefficient for linear LD law")
			elif len(self.LDCs) != 1:
				raise Exception("Incorrect number of limb-darkening coefficients for linear LD law")

		if self.LDlaw == "quadratic":
			if not ("LDCs" in kwargs):
				raise Exception("Must specify array of 2 limb-darkening coefficients for quadratic LD law")
			elif len(self.LDCs) != 2:
				raise Exception("Incorrect number of limb-darkening coefficients for quadratic LD law")
		
		if self.LDlaw == "nonlinear":
			if not ("LDCs" in kwargs):
				raise Exception("Must specify array of 4 limb-darkening coefficients for nonlinear LD law")
			elif len(self.LDCs) != 4:
				raise Exception("Incorrect number of limb-darkening coefficients for nonlinear LD law")
		
		if self.LDlaw not in ["nonlinear","linear","uniform","quadratic"]:
			raise Exception("Only uniform, linear, quadratic, or 4-parameter nonlinear LD laws are supported")
		#set opacity matrix if it's not passed in
		if "imfile" in kwargs:
			self.opacitymat = pixelate_image(imfile=self.imfile, nside=self.lowres, method=self.lowrestype, rounding=self.lowresround)
		
		self.w = 2./(np.shape(self.opacitymat)[0])
		
		gridshape = np.shape(self.opacitymat)
		
		if self.positions is None:
			self.positions = positions(n=gridshape[0], m=gridshape[1], t=self.t_arr, tref=self.t_ref, v=self.v)
		
		#if opacity matrix is passed in but the desired pixel resolution is smaller, lower the resolution
		if (("opacitymat" in kwargs) and ("lowres" in kwargs)):
			self.opacitymat = lowres_grid(opacitymat=self.opacitymat, positions=self.positions, nside=self.lowres, method=self.lowrestype, rounding=self.lowresround)
			self.w = 2./(np.shape(self.opacitymat)[0])
			gridshape = np.shape(self.opacitymat)
			self.positions = positions(n=gridshape[0], m=gridshape[1], t=self.t_arr, tref=self.t_ref, v=self.v)
		
	def gen_LC(self, t_arr):
		cdef:
			int i,j,k
		#updates self.t_arr if the passed t_arr is different
		gridshape = np.shape(self.opacitymat)
		
		if ~np.all(t_arr == self.t_arr):
			#print "new times"
			self.t_arr = t_arr
			self.positions = positions(n=gridshape[0], m=gridshape[1], t=self.t_arr, tref=self.t_ref, v=self.v)
		
		if self.LDlaw == "uniform":
			if self.areas is None:
				self.areas = np.zeros((len(self.t_arr), gridshape[0], gridshape[1]), dtype=float)
			self.blockedflux = np.zeros((len(self.t_arr), gridshape[0], gridshape[1]), dtype=float)
			
			#t0 = time.time()

			if np.count_nonzero(self.areas) == 0:
				for i in range(0,gridshape[0]):
					for j in range(0,gridshape[1]):
						for k in range(0,len(self.t_arr)):
							#print (k, i, j)
							# allow for opacities between 0 and 1
							#print self.t_arr[k]
							self.areas[k,i,j] = pixeloverlaparea(x0=self.positions[k,i,j,0], y0=self.positions[k,i,j,1], w=self.w)
							self.blockedflux[k,i,j] = self.areas[k,i,j]*self.opacitymat[i,j]
			
			else:
				for i in range(0,gridshape[0]):
					for j in range(0,gridshape[1]):
						for k in range(0,len(self.t_arr)):
							self.blockedflux[k,i,j] = self.areas[k,i,j]*self.opacitymat[i,j]
						
			
			fluxtot = np.zeros(len(self.t_arr))
			for k in range(0,len(self.t_arr)):
				#fluxtot[k] = 1. - np.sum(self.areas[k,:,:])
				fluxtot[k] = 1. - np.sum(self.blockedflux[k,:,:])

		elif (((self.LDlaw == "nonlinear") | (self.LDlaw == "linear") | (self.LDlaw == "quadratic")) & (self.w <= 0.2)):
			self.areas = np.zeros((len(self.t_arr), gridshape[0], gridshape[1]), dtype=float)
			self.blockedflux = np.zeros((len(self.t_arr), gridshape[0], gridshape[1]), dtype=float)
			
			#t0 = time.time()
			for i in range(0,gridshape[0]):
				for j in range(0,gridshape[1]):
					for k in range(0,len(self.t_arr)):
						# allow for opacities between 0 and 1
						self.areas[k,i,j] = pixeloverlaparea(x0=self.positions[k,i,j,0], y0=self.positions[k,i,j,1], w=self.w)
						self.blockedflux[k,i,j] = self.areas[k,i,j]*self.opacitymat[i,j]
						
			#t1 = time.time()
			#print(t1-t0)
			
			
			#t2 = time.time()
			self.LD = np.zeros_like(self.areas)
			for i in range(0,gridshape[0]):
				for j in range(0,gridshape[1]):
					if self.LDlaw == "nonlinear":
						self.LD[:,i,j] = LDfluxsmall(x=self.positions[:,i,j,0], y=self.positions[:,i,j,1], t=self.t_arr, Ar_occ=self.blockedflux[:,i,j], c1=self.LDCs[0], c2=self.LDCs[1], c3=self.LDCs[2], c4=self.LDCs[3], w=self.w)
					elif self.LDlaw == "quadratic":
						self.LD[:,i,j] = LDfluxsmall(x=self.positions[:,i,j,0], y=self.positions[:,i,j,1], t=self.t_arr, Ar_occ=self.blockedflux[:,i,j], c1=0., c2=(self.LDCs[0] + 2.*self.LDCs[1]), c3=0., c4=(-1.*self.LDCs[1]), w=self.w)
					elif self.LDlaw == "linear":
						self.LD[:,i,j] = LDfluxsmall(x=self.positions[:,i,j,0], y=self.positions[:,i,j,1], t=self.t_arr, Ar_occ=self.blockedflux[:,i,j], c1=0., c2=self.LDCs[0], c3=0., c4=0., w=self.w)
					
			#t3 = time.time()
			#print (t3-t2)
			
			fluxtot = np.zeros(len(self.t_arr))
			for k in range(0,len(self.t_arr)):
				fluxtot[k] = 1. - np.sum(self.LD[k,:,:])
				#if np.isnan(fluxtot[k]):
					#print self.t_arr[k]
		
		elif (((self.LDlaw == "nonlinear") | (self.LDlaw == "linear") | (self.LDlaw == "quadratic")) & (self.w > 0.2)):
			raise Exception("Small-planet approximation for LD calculation is inappropriate. Choose higher resolution")
			fluxtot = None

		return fluxtot
		
	def plot_grid(self,save=False, filename=None):
		nside_y = np.shape(self.opacitymat)[0]
		nside_x = np.shape(self.opacitymat)[1]
		
		#nside = np.max(np.shape(self.opacitymat))
		fig, ax= plt.subplots(1,1,figsize=(8,8))
		ax.set_aspect('equal')
		
		ax.imshow(self.opacitymat,cmap="Greys",aspect="equal",origin="upper",interpolation='none',vmin=0.,vmax=1.)
		#plt.colorbar()
		ax.set_xlabel("j",fontsize=16)
		ax.set_ylabel("i",fontsize=16)
		ax.set_xlim(-0.5,nside_x-0.5)
		ax.set_ylim(nside_y-0.5,-0.5)
		
		if save is False:
			plt.show()
		elif save is True:
			plt.savefig(filename, fmt="png")
		
		return None
