#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-08 10:52:59
# @Author  : Lzh (lizhaoh2015@gmail.com)
# @Link    : http://songlaoshi.github.io
# @Version : $Id$

import os
import numpy as np
import datetime


class VI_container(object):
	"""docstring for VI_container:
	植被指数类，保存植被指数计算结果
	NDVI=(Rnir-Rred)/(Rnir+Rred),
	EVI=(2.5 * (Rnir - Rred) / (Rnir + 6 * Rred - 7.5 * Rblue + 1))
	MTCI=(R754-R709)/(R709-R681)
	MTVI2=1.5[1.2(R800-R550)-2.5(R670-R550)]/sqrt((2R800+1)^2-(6R800-5sqrt(R670))-0.5)
	CVI=(Rnir/Rgreen)*(Rred/Rgreen) where nir=760-900,red=630-690,green=520-600
    NIRv = NDVI * Rnir
	FCVI = Rnir - Rvis
	NIRVR = NDVI * Rad_nir
	"""

	def __init__(self):
		super(VI_container, self).__init__()
	wl = np.array([])
	irrad = np.array([])
	rad = np.array([])
	ref = np.array([])
	ftime = np.array([])
	NDVI = np.array([])
	EVI = np.array([])
	MTCI = np.array([])
	MTVI2 = np.array([])
	PRI = np.array([])
	greenNDVI = np.array([])
	rededgeNDVI = np.array([])
	OSAVI = np.array([])
	CIgreen = np.array([])
	CIrededge = np.array([])
	CVI = np.array([])
	SR = np.array([])
	Rblue=np.array([])
	Rgreen=np.array([])
	Rred=np.array([])
	Rnir=np.array([])
	Rrededge=np.array([])
	# Ra_blue=np.array([])
	# Ra_green=np.array([])
	# Ra_red=np.array([])
	Ra_nir=np.array([])
	# Ra_rededge=np.array([])
	NIRv = np.array([])
	FCVI = np.array([])
	mNDI705 = np.array([])
	NIRVR = np.array([])


def get_vegetation_indices(wl, ref, radiance):
	'''
	计算植被指数
	'''
	iblue = (wl > 430) & (wl < 450)
	igreen = (wl > 540) & (wl < 570)
	ired = (wl > 650) & (wl < 680)
	iNIR = (wl > 770) & (wl < 790)
	irededge = (wl > 700) & (wl < 710)
	iband10 = (wl > 753.75 - 7.5 / 2) & (wl < 753.75 + 7.5 / 2)
	iband9 = (wl > 708.75 - 10 / 2) & (wl < 708.75 + 10 / 2)
	iband8 = (wl > 681.25 - 7.5 / 2) & (wl < 681.25 + 7.5 / 2)
	i_lamda_800 = (wl > 800 - 1.5) & (wl < 800 + 1.5)
	i_lamda_550 = (wl > 550 - 1.5) & (wl < 550 + 1.5)
	i_lamda_670 = (wl > 670 - 1.5) & (wl < 670 + 1.5)
	i_lamda_531 = (wl > 531 - 0.5) & (wl < 531 + 0.5)
	i_lamda_570 = (wl > 570 - 0.5) & (wl < 570 + 0.5)
	ivis = (wl>400) & (wl<700)
	# i_lamda_750 = (wl > 750 - 12.5) & (wl < 750 + 12.5)
	# i_lamda_705 = (wl > 705 - 12.5) & (wl < 705 + 12.5)
	# i_lamda_445 = (wl > 445 - 12.5) & (wl < 445 + 12.5)

	# get average ref
	Rblue = np.nanmean(ref[:,:, iblue], axis=2)
	Rgreen = np.nanmean(ref[:,:, igreen], axis=2)
	Rred = np.nanmean(ref[:,:, ired], axis=2)
	Rnir = np.nanmean(ref[:,:, iNIR], axis=2)
	Rrededge = np.nanmean(ref[:,:, irededge], axis=2)
	R10 = np.nanmean(ref[:,:, iband10], axis=2)
	R9 = np.nanmean(ref[:,:, iband9], axis=2)
	R8 = np.nanmean(ref[:,:, iband8], axis=2)
	R_lamda_800 = np.nanmean(ref[:,:, i_lamda_800], axis=2)
	R_lamda_550 = np.nanmean(ref[:,:, i_lamda_550], axis=2)
	R_lamda_670 = np.nanmean(ref[:,:, i_lamda_670], axis=2)
	R_lamda_531 = np.nanmean(ref[:,:, i_lamda_531], axis=2)
	R_lamda_570 = np.nanmean(ref[:,:, i_lamda_570], axis=2)
	Rvis = np.nanmean(ref[:,:,ivis], axis = 2)
    # get VIS
	setup = VI_container()
	setup.NDVI = (Rnir - Rred) / (Rnir + Rred)
	setup.EVI = (2.5 * (Rnir - Rred) / (Rnir + 6 * Rred - 7.5 * Rblue + 1))
	setup.MTCI = (R10 - R9) / (R9 - R8)
	temp = (2 * R_lamda_800 + 1)**2 - \
	        (6 * R_lamda_800 - 5 * np.sqrt(R_lamda_670)) - 0.5
	setup.MTVI2 = 1.5 * (1.2 * (R_lamda_800 - R_lamda_550) -
	                     2.5 * (R_lamda_670 - R_lamda_550)) / np.sqrt(temp)
	setup.PRI = (R_lamda_531 - R_lamda_570) / (R_lamda_531 + R_lamda_570)
	setup.greenNDVI = (Rnir - Rgreen) / (Rnir + Rgreen)
	setup.rededgeNDVI = (Rnir - Rrededge) / (Rnir + Rrededge)
	setup.OSAVI = (Rnir - Rred)/(Rnir + Rred +0.16)
	setup.CIgreen = Rnir / Rgreen - 1
	setup.CIrededge = Rnir / Rrededge - 1
	setup.CVI = Rnir * Rred / Rgreen / Rgreen
	setup.SR = Rnir / Rred
	setup.Rblue=Rblue
	setup.Rgreen=Rgreen
	setup.Rred=Rred
	setup.Rnir=Rnir
	setup.Rrededge=Rrededge
	# setup.Ra_blue=Ra_blue
	# setup.Ra_green=Ra_green
	# setup.Ra_red=Ra_red
	# setup.Ra_nir=Ra_nir
	# setup.Ra_rededge=Ra_rededge
	setup.NIRv = setup.NDVI * Rnir
	setup.FCVI = Rnir - Rvis

	return setup

def get_vegetation_indices2D(setup, wl, ref, *radiance):
	'''
	计算植被指数
	'''
	iblue = (wl > 430) & (wl < 450)
	igreen = (wl > 540) & (wl < 570)
	ired = (wl > 650) & (wl < 680)
	iNIR = (wl > 770) & (wl < 790)
	irededge = (wl > 700) & (wl < 710)
	iband10 = (wl > 753.75 - 7.5 / 2) & (wl < 753.75 + 7.5 / 2)
	iband9 = (wl > 708.75 - 10 / 2) & (wl < 708.75 + 10 / 2)
	iband8 = (wl > 681.25 - 7.5 / 2) & (wl < 681.25 + 7.5 / 2)
	i_lamda_800 = (wl > 800 - 1.5) & (wl < 800 + 1.5)
	i_lamda_550 = (wl > 550 - 1.5) & (wl < 550 + 1.5)
	i_lamda_670 = (wl > 670 - 1.5) & (wl < 670 + 1.5)
	i_lamda_531 = (wl > 531 - 0.5) & (wl < 531 + 0.5)
	i_lamda_570 = (wl > 570 - 0.5) & (wl < 570 + 0.5)
	ivis = (wl>400) & (wl<700)
	i_lamda_750 = (wl > 750 - 12.5) & (wl < 750 + 12.5)
	i_lamda_705 = (wl > 705 - 12.5) & (wl < 705 + 12.5)
	i_lamda_445 = (wl > 445 - 12.5) & (wl < 445 + 12.5)
	
	# get average ref
	Rblue = np.nanmean(ref.loc[iblue,:], axis=0)
	Rgreen = np.nanmean(ref.loc[igreen,:], axis=0)
	Rred = np.nanmean(ref.loc[ired,:], axis=0)
	Rnir = np.nanmean(ref.loc[iNIR,:], axis=0)
	Rrededge = np.nanmean(ref.loc[irededge,:], axis=0)
	R10 = np.nanmean(ref.loc[iband10,:], axis=0)
	R9 = np.nanmean(ref.loc[iband9,:], axis=0)
	R8 = np.nanmean(ref.loc[iband8,:], axis=0)
	R_lamda_800 = np.nanmean(ref.loc[i_lamda_800,:], axis=0)
	R_lamda_550 = np.nanmean(ref.loc[i_lamda_550,:], axis=0)
	R_lamda_670 = np.nanmean(ref.loc[i_lamda_670,:], axis=0)
	R_lamda_531 = np.nanmean(ref.loc[i_lamda_531,:], axis=0)
	R_lamda_570 = np.nanmean(ref.loc[i_lamda_570,:], axis=0)
	Rvis = np.nanmean(ref.loc[ivis,:], axis = 0)
	R_lamda_750 = np.nanmean(ref.loc[i_lamda_750,:], axis=0)
	R_lamda_705 = np.nanmean(ref.loc[i_lamda_705,:], axis=0)
	R_lamda_445 = np.nanmean(ref.loc[i_lamda_445,:], axis=0)
    # get VIS
	setup.NDVI = (Rnir - Rred) / (Rnir + Rred)
	setup.EVI = (2.5 * (Rnir - Rred) / (Rnir + 6 * Rred - 7.5 * Rblue + 1))
	setup.MTCI = (R10 - R9) / (R9 - R8)
	temp = (2 * R_lamda_800 + 1)**2 - \
	        (6 * R_lamda_800 - 5 * np.sqrt(R_lamda_670)) - 0.5
	setup.MTVI2 = 1.5 * (1.2 * (R_lamda_800 - R_lamda_550) -
	                     2.5 * (R_lamda_670 - R_lamda_550)) / np.sqrt(temp)
	setup.PRI = (R_lamda_531 - R_lamda_570) / (R_lamda_531 + R_lamda_570)
	setup.greenNDVI = (Rnir - Rgreen) / (Rnir + Rgreen)
	setup.rededgeNDVI = (Rnir - Rrededge) / (Rnir + Rrededge)
	setup.OSAVI = (Rnir - Rred)/(Rnir + Rred +0.16)
	setup.CIgreen = Rnir / Rgreen - 1
	setup.CIrededge = Rnir / Rrededge - 1
	setup.CVI = Rnir * Rred / Rgreen / Rgreen
	setup.SR = Rnir / Rred
	setup.Rblue=Rblue
	setup.Rgreen=Rgreen
	setup.Rred=Rred
	setup.Rnir=Rnir
	setup.Rrededge=Rrededge
	# setup.Ra_blue=Ra_blue
	# setup.Ra_green=Ra_green
	# setup.Ra_red=Ra_red
	# setup.Ra_nir=Ra_nir
	# setup.Ra_rededge=Ra_rededge
	setup.NIRv = setup.NDVI * Rnir
	setup.FCVI = Rnir - Rvis
	setup.mNDI705 = (R_lamda_750 - R_lamda_705) / (R_lamda_750 + R_lamda_705 - 2 * R_lamda_445)
	if radiance:
		setup.Ra_nir = np.nanmean(radiance[0].loc[iNIR,:], axis=0)
		setup.NIRVR = setup.NDVI * setup.Ra_nir

	return setup

def get_doy(datestr):
	'''
	计算day of year
	datestr: e.g. 20170707
	'''
	year=int(datestr[0:4])
	month=int(datestr[4:6])
	day=int(datestr[6:8])
	doy=datetime.date(year,month,day).timetuple().tm_yday
	return doy