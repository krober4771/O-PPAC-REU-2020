import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import tqdm
from tqdm import tqdm



#adds 10 units of padding to the front and back of dataset diodes for fitting
def padding(data_arr):
    return np.pad(data_arr, [(0,0), (10, 10)], 'constant')

#gaussian fit function
def gaussianpdf(x, k, sigma, mu):
    return k*np.exp(-0.5*((x-mu)/sigma)**2)

#function to (hopefully) fit gaussians (and thus, centroids which is what we want) onto all data sets
    #pulls out all the centroid values
#to be used with fit_peaks
def centroiddn(x, y):
    try:
        if np.max(y) > 20:
            kvalue = np.max(y)+(np.max(y)/2)
        elif np.max(y) == 0:
            kvalue = 1
        else:
            kvalue = np.max(y)
        popt, _ = curve_fit(gaussianpdf, x, y, bounds=(0, [kvalue,10., 33.]))
        return popt[2]
    except RuntimeError: #getting some runtime errors from it fitting incorrectly; this keeps it from freezing
        return -5555
    
#fits peaks of data sets (only one peak at a time; data must be broken into the 4 sides)
#xdata is an array of integers from 0-n with n number of diodes (for 33 diodes it is: np.arange(0,33,1))
def fit_peaks(x_data, y_data):
    fit_peaks = []
    for i in tqdm(range(y_data.shape[0])):
        fit_peaks.append(centroiddn(x_data, y_data[i,:]))
    return np.array(fit_peaks)

#locates the -5555 runtime errors in 2 arrays, then finds the unique indices between them
def find_peak_error(arr1data, arr2data):
    error_indices1 = np.where(arr1data == -5555 )
    error_indices2 = np.where(arr2data == -5555 )
        
    error_indices = np.unique(np.concatenate([error_indices1, error_indices2], axis = 1))
        
    return error_indices


#the next 2 functions remove the indices of error as defined above
#the first works with 2 peaks simultaneously while the second works with a single peak 
def remove_peak_error(arr1data, arr2data, removearr):
    
    arr1data_df = pd.DataFrame(arr1data)
    arr2data_df = pd.DataFrame(arr2data)
        
    arr1data_df_fixed = arr1data_df.drop(arr1data_df.index[removearr])
    arr2data_df_fixed = arr2data_df.drop(arr2data_df.index[removearr])
        
    arr1data_fixed = arr1data_df_fixed.to_numpy()
    arr2data_fixed = arr2data_df_fixed.to_numpy()
    
    arr1data = np.reshape(arr1data_fixed, -1)
    arr2data = np.reshape(arr2data_fixed, -1)
    
    return arr1data, arr2data
#defined above
def remove_peak_error_single(arr1data, removearr):
    arr1data_df = pd.DataFrame(arr1data)
    arr1data_df_fixed = arr1data_df.drop(arr1data_df.index[removearr])
    arr1data_fixed = arr1data_df_fixed.to_numpy()
    arr1data = np.reshape(arr1data_fixed, -1)
    return arr1data

#this error remove is specifically for labels since the above error remove reshapes the array at the end
    #reshaping with labels means you lose all the data
def remove_label_indices_noresh(labels, indices):
    labels = pd.DataFrame(labels)
    labels_fixed = labels.drop(labels.index[indices])
    return labels_fixed.to_numpy()


#conversions from diode number to millimeters and vice versa
#might need to be updated if there is a better method of converting
def diodenum_to_mm(arr1data):
    return (arr1data /33)*100 - 48

def mm_to_diodenum(arr1data):
    return ((arr1data + 48)/100)*33

#finds the averages between 2 arrays; pretty straightforward
def average(arr1data, arr2data):
    avg = (arr1data + arr2data)/2
    return avg

#uses remove_peak_error, average, and diodenum_to_mm to recompile peak data for comparison and graphing
def peak_recompile(arr1, arr2, arrerror):
    arr1, arr2 = remove_peak_error(arr1, arr2, arrerror)
    avg_peaks = average(arr1, arr2)
    mm_peaks = diodenum_to_mm(avg_peaks)
    return avg_peaks, mm_peaks
    #avg_peaks is in diode number and good for graphing while mm_peaks is in mm and good for comparing against labels


