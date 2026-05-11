# -*- coding: utf-8 -*- 

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import leastsq
# import matplotlib.pyplot as plt
# import sys
# sys.path.append(r'.\functions')
import SVD_functions

# -------------------class--------------------------
class container:
    wl = np.array([])
    irrad = np.array([])
    rad = np.array([])
    ref = np.array([])
    ftime = np.array([])
    
    # (attention！-%) the minimum SD (normalized singular value) to automatically include SV_n
    variance_threshold = 1
    poly1degree = 2  # the degree to use for the reflectance function polynomial for SV1
    poly2degree = 1  # the degree to use for the reflectance function polynomial for SV2
    min_rank = 2  # the minimum # of SV's to include
    # the # of fluorescence SV's to include (Ari's data) or -1 for Christian's
    # fluorescence curve
    # NOTE: numFLv = 0 computes the best-fit w/o any SIF.
    numFlv = 1
    meanwl = 0
    norm_S = np.array([])
    EV = np.array([])
    U = np.array([])
    S = np.array([])
    K = np.array([])
    hF = np.array([])
    Flidx = np.array([])
    Kstd = np.array([])
    Kmean = np.array([])
    pixels = np.array([])
    # irrCoefs=np.array([])

# -------------------func--------------------------

def findwl(wl,wltarget):
    tmp = np.abs(wl - wltarget).tolist()
    idx = tmp.index(np.min(tmp))
    return idx,wl[idx]

def FLD3(SIFcon, wl3fld, widths):
    wlout1,wlin,wlout2 = wl3fld[0],wl3fld[1],wl3fld[2]
    # find the indices of the three wavelengths
    idx1 = abs(SIFcon.wl - wlout1) < widths[0]
    idxin = abs(SIFcon.wl - wlin) < widths[1]
    idx2 = abs(SIFcon.wl - wlout2) < widths[2]

    irradout1 = np.nanmean(SIFcon.irrad[idx1])
    radout1 = np.nanmean(SIFcon.rad[idx1])
    irradin = np.nanmean(SIFcon.irrad[idxin])
    radin = np.nanmean(SIFcon.rad[idxin])
    irradout2 = np.nanmean(SIFcon.irrad[idx2])
    radout2 = np.nanmean(SIFcon.rad[idx2])

    w21 = (wlout2-wlin)/(wlout2-wlout1)
    w22 = (wlin-wlout1)/(wlout2-wlout1)
    e_term = irradin/(w21*irradout1+w22*irradout2)
    fs = (radin-e_term*(w21*radout1+w22*radout2))/(1-e_term)

    return fs

def iFLD(SIFcon,wlin,fwhm):
    wl = SIFcon.wl
    E = SIFcon.irrad
    L = SIFcon.rad
    R = SIFcon.ref

    wl_in = wlin
    bufferin, bufferout = 5, 1
    out_in = 0.7535*fwhm+2.8937
    wlout1 = wl_in-bufferin
    wlout2 = wl_in+bufferin
    #
    idx_1 = (wl<wlout1) & (wl>750)
    idx_in = (wl>wlout1) & (wl<wlout2)
    idx_2 = (wl>wlout2) & (wl<780)
    wl_for_fit = list(wl[idx_1]) + list(wl[idx_2])
    wl_for_polyval = wl[idx_in]
    E_for_fit = np.array(list(E[idx_1])+list(E[idx_2]))
    R_for_fit = np.array(list(R[idx_1])+list(R[idx_2]))
    # 拟合
    f = interpolate.interp1d(wl_for_fit,E_for_fit,kind = 'quadratic')
    E_fitted = f(wl_for_polyval)
    p1 = np.polyfit(wl_for_fit, R_for_fit, 4)
    z = np.poly1d(p1)
    R_fitted = z(wl_for_polyval)

    # fig,axs = plt.subplots(1,1)
    # # axs.plot(wl_for_fit,E_for_fit,'b-',marker='.')
    # axs.plot(wl[(wl>750) & (wl<780)],E[(wl>750) & (wl<780)],'b-',marker='.')
    # axs.plot(wl_for_polyval,E_fitted,'r-')
    # ax = axs.twinx()
    # ax.plot(wl[(wl>750) & (wl<780)],R[(wl>750) & (wl<780)],'orange',ls='-',marker='.')
    # ax.plot(wl_for_polyval,R_fitted,'k-')
    # plt.show()

    Ein = E[idx_in]
    tmp = np.abs(Ein - np.min(Ein)).tolist()
    idx_Ein = tmp.index(np.min(tmp)) # E最小值对应的波长序号
    Rin_fitted = R_fitted[idx_Ein]
    Iin = Ein[idx_Ein]
    Iin_fitted = np.mean(E_fitted)
    wl_out = wl_for_polyval - out_in
    out, Rout= [], []
    for m in range(len(wl_out)):
        tmp = np.abs(wl - wl_out[m]).tolist()
        idx_for_wl_out = tmp.index(np.min(tmp)) 
        out.append( wl[idx_for_wl_out])
        Rout.append(R[idx_for_wl_out])
    Rout = np.mean(Rout)
    aR = Rout/Rin_fitted

    Iout = np.mean(E[(wl>np.mean(wl_out)-bufferout) & (wl<np.mean(wl_out))])
    Lin = np.min(L[(wl>wl_in-bufferin) & (wl<wl_in+bufferin)])
    Lout = np.mean(L[(wl>np.mean(wl_out)-bufferout) & (wl<np.mean(wl_out))])
    aF = (Iout/Iin_fitted)*aR
    fs = (aR*Iout*Lin - Iin*Lout)/(aR*Iout - aF*Iin)

    return fs

# %% SFM拟合函数和残差函数定义
def sfmlinearfit(x,wl,p):
    k1,b1,k2,b2 = p
    y = (k1*wl+b1)*x + (k2*wl+b2)
    return y

# def sfmquadraticfit(x,wl,p):
#     a1,b1,c1,a2,b2,c2 = p
#     y = (a1*wl**2+b1*wl+c1) * x + (a2*wl**2+b2*wl+c2)
#     return y

def sfmquadraticfit(x,wl,p):
    a1,b1,c1,a2,b2 = p
    y = (a1*wl**2+b1*wl+c1) * x + (a2*wl+b2)
    return y

def residuals_lin(p,wl,ydata,xdata):
    return ydata - sfmlinearfit(xdata,wl,p)

def residuals_qua(p,wl,ydata,xdata):
    return ydata - sfmquadraticfit(xdata,wl,p)


# %%
def SFMfit(SIFcon,wlout1,wlin,wlout2):
    roi = (SIFcon.wl>=wlout1) & (SIFcon.wl<=wlout2)
    wl = SIFcon.wl[roi]
    xdata = SIFcon.irrad[roi]
    ydata = SIFcon.rad[roi]
    # --------ploynomial-----------
    # linear 
    p0 = [0.01,-5,-0.0005,-0.1]
    plsq = leastsq(residuals_lin,p0,args = (wl,ydata,xdata))
    # yfit = sfmlinearfit(xdata,wl,plsq[0])
    _,_,k2,b2 = plsq[0]
    fs_linear = k2*wlin+b2
    # quadratic
    # p0 = [0.01,1,-0.0001,0.01,-0.1,1]
    p0 = [0.01,1,-0.0001,0.01,-0.1]
    plsq = leastsq(residuals_qua,p0,args = (wl,ydata,xdata))
    _,_,_,a2,b2 = plsq[0]
    fs_quadratic = a2*wlin+b2
    return fs_linear,fs_quadratic

def SFM(SIFcon,wlsfm):
    wl = SIFcon.wl
    wlout1a,wlina,wlout2a = wlsfm[0],wlsfm[1],wlsfm[2]
    wlout1b,wlinb,wlout2b = wlsfm[3],wlsfm[4],wlsfm[5]
    if np.nanmin(wl)>687:
        fs_linear_a, fs_quadratic_a = SFMfit(SIFcon,wlout1a,wlina,wlout2a)
        fs_linear_b, fs_quadratic_b = np.nan,np.nan
    else:
        fs_linear_a, fs_quadratic_a = SFMfit(SIFcon,wlout1a,wlina,wlout2a)
        fs_linear_b, fs_quadratic_b = SFMfit(SIFcon,wlout1b,wlinb,wlout2b)
    return fs_linear_a, fs_quadratic_a, fs_linear_b, fs_quadratic_b
    
    

# %%
def get_train(trainpath):
    io=pd.ExcelFile(trainpath)
    irrad = pd.read_excel(io)
    irrad = irrad.dropna()
    training_spec=irrad.iloc[:,1:].values
    new_wl_sky = np.array((irrad.columns[1:]).tolist()).astype('float')

    rangetemp = ['all', 'fraun', 'o2a',
                 'o2a_narrow', 'fraun_o2a', 'new1', 'new2']
    # new_wl_sky=wl
    chosen_range = rangetemp[6]  # namely, new2
    numFlv = -1  # use fspectrum
    # get setup and FL760
    setup = container()
    setup, FL760 = SVD_functions.preprocess(
        new_wl_sky, chosen_range, setup, numFlv, 0.04, 2, 2)

    min_rank = 2
    # train = SVD_functions.trainSVD_SIF(
    #         training_spec[:, setup.pixels], setup, min_rank)  # exspec

    # return train,setup,min_rank,FL760
    return setup,min_rank,FL760

def solveSVD(train, full_spec, setup,min_rank,FL760):
    #
    userdata = 1
    R = container()
    R, _ = SVD_functions.mysolveSVD_SIF(train,full_spec, userdata, R, setup, min_rank, FL760, rolling=False, var_m=0, var_b=1)
    # R.SIF[R.SIF<0]=np.NaN
    return R.SIF