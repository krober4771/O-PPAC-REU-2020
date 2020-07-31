import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
    

#gaussian fit function
def gaussianpdf(x, k, sigma, mu):
    return k*np.exp(-0.5*((x-mu)/sigma)**2)


#for some reason I am not able to name the function "full_width_half_max" as I wanted to
#not sure if this is something someone else can fix
#used to find the fwhm of a residual histogram
#allows for separate fwhm for x and y when comparing to the true label
def full_half(data):
    bin_heights, bin_borders, _ = plt.hist(data, bins='auto', label='Residual')
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    popt, _ = curve_fit(gaussianpdf, bin_centers, bin_heights, p0=[60., 10., 30.])
    fwhm = popt[1]*2.355
    
    x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000) #the x interval the fit is put over
    plt.plot(x_interval_for_fit, gaussianpdf(x_interval_for_fit, *popt), label='Gaussian Fit')
    #gauss = gaussianpdf(x_interval_for_fit, *popt) #y data for the gaussian fit if needed to be plotted separately
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    return np.abs(fwhm)