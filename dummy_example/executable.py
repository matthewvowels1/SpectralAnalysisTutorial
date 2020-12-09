'''
(1) Assumes dyadic data are in the following format:
    row 1: [pA1 t=0, .... pA1 t=30, pB1 t=0 ... pB1 t=30]
    row 2: [pA2 t=0, .... pA2 t=30, pB2 t=0 ... pB2 t=30]
    ...
    i.e. matrix shape is (num_participants, 2*T)

    and that non-dyadic data are in the following format:
    row 1: [ind1 t=0, .... ind1 t=30]
    row 2: [ind2 t=0, .... ind2 t=30]
    ...
    i.e. matrix shape is (num_participants, T)

(2) Assume data have been individually mean centred.

(3) Assumes the sampling rate is *T per month*
    (e.g. T=30 times in a month for 3 days of daily diary data)

(4) CPSD is calculated as    1/num_couples * sum_i (abs(cpsd_i))  for couple 'i'.
    i.e. the average of the absolute cpsds

'''

print('############ SETTING GLOBAL PARAMETERS ##############')

filename = './DummyIndividualDailyDiaryFamilySet_12.4.20.csv'
analysis_type = 'ind'  # 'dyadic' or 'ind'
significance_test = True  # turns on or off bootstrapped significance testing
num_bootstraps = 10000  # number of bootstraps to run when deriving p-value estimates
T = 30  # number of timepoints for ALL participants
Fs = 30  # number of samples in a MONTH (cycles per month cpm are the assumed frequency units)
windowing = True  # whether or not to do time windowing (Hanning)
missing_value_code = 999 # this is the code for missing values that will be replaced using linear interpolation
interpolation = True  # whether or not to scrape the data and interpolate missing values
# If interpolation = True then a new, interpolated
# version of the dataset will be saved, which can be loaded in for future analyses (and then
# set interpolation = False to avoid re-computing it.)


print('########### INSTALLING AND IMPORTING PACKAGES ###############')
import subprocess
import sys
import pip


def install(package):
	if hasattr(pip, 'main'):
		pip.main(['install', package])
	else:
		pip._internal.main(['install', package])


install('numpy')
install('matplotlib')
install('scipy')
install('pandas')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.stats import circmean


#  Defining useful function for significance testing
def calculate_significance(reference, bootstraps, f):
	counts = np.zeros(f.shape[0])
	n = bootstraps.shape[0]
	for i in range(n):
		for j in range(f.shape[0]):
			to_compare = bootstraps[i, j]
			coeff = reference[j]
			if coeff > to_compare:
				counts[j] += 1
	counts /= (n * 100)
	return counts

# define function that is useful for interpolating nan values
def nan_helper(y):
    # https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def interp_helper(seq_to_interp):
    nans, x = nan_helper(seq_to_interp)
    seq_to_interp[nans]= np.interp(x(nans), x(~nans), seq_to_interp[~nans])
    return seq_to_interp

def convert_value_to_NaN(x, value):
    indices = np.where(x==value)
    x[indices] = np.nan
    return x



print('######## LOADING DATA ##########')
data = pd.read_csv(filename)
columns = data.columns

if analysis_type == 'dyadic':
    assert data.shape[1] == int(2 * T), 'dyadic time variable dimension does not match 2*T, check T against data'
elif analysis_type == 'ind':
    assert data.shape[1] == T, 'time variable dimension does not match T, check T against data'

num_participants = data.shape[0]  # if data are dyadic, this is equal to the number of couples
# convert data to numpy arrays
data = np.asarray(data.values)


time_index = np.linspace(0, 1 - (1 / T), T)
f = Fs * np.linspace(0, int(T / 2), int(T / 2 + 1)) / T

window = np.hanning(T)
acf = T / (window.sum())
window = np.repeat(window.reshape(-1, 1), num_participants, 1)

if analysis_type == 'dyadic':
    print('############ RUNNING DYADIC ANALYSIS #################')

    pAs = data[:, :T].T
    pBs = data[:, T:].T

    pAs_interp = []
    pBs_interp = []
    if interpolation:
        for time_series_ in range(num_participants):
            time_series_A = interp_helper(convert_value_to_NaN(pAs[:, time_series_], missing_value_code))
            time_series_B = interp_helper(convert_value_to_NaN(pBs[:, time_series_], missing_value_code))
            pAs_interp.append(time_series_A)
            pBs_interp.append(time_series_B)

        pAs = np.asarray(pAs_interp).T
        pBs = np.asarray(pBs_interp).T
        both = np.concatenate((pAs, pBs), 0)

        # save new dataframe
        interp_df = pd.DataFrame(both).T
        interp_df.columns = list(columns)
        interp_df.to_csv(filename.split('.csv')[0] + '_interp.csv', index=False)

    if windowing:
        pAs = pAs * window * acf
        pBs = pBs * window * acf

    f1, Pxy = signal.csd(pAs, pBs, nperseg=T, axis=0)
    Pxy_mean = np.mean(np.abs(Pxy), axis=1)
    cpsd_mean = pd.DataFrame()
    cpsd_mean['cpsd'] = Pxy_mean
    cpsd_mean['f (cpm)'] = f
    cpsd_mean.to_csv('./cpsd.csv', index=False)

    cpsd_phase = np.angle(Pxy)

    # unwrap phase:
    for i in range(cpsd_phase.shape[1]):
        for j in range(cpsd_phase.shape[1]):
            if cpsd_phase[i, j] <= 0:
                cpsd_phase[i, j] += 2 * np.pi
    couple_av_phase = circmean(cpsd_phase, axis=1)

    av_phase = pd.DataFrame()
    all_phase = pd.DataFrame(cpsd_phase)
    av_phase['average_phase'] = couple_av_phase
    av_phase['f (cpm)'] = f
    all_phase['f (cpm)'] = f

    av_phase.to_csv('./average_phase.csv', index=False)
    all_phase.to_csv('./all_phase.csv', index=False)

    if significance_test:
        print('############ RUNNING BOOTSTRAPS #################')
        cpsd_mean_rands = np.zeros([num_bootstraps, f.shape[0]])

        for i in range(num_bootstraps):
            if i % 100 == 0:
                print('Running bootstrap number {} out of '.format(i), num_bootstraps)
            pAs_rand = np.random.permutation(pAs)
            pBs_rand = np.random.permutation(pBs)
            f1, Pxy = signal.csd(pAs, pBs, nperseg=T, axis=0)
            cpsd_mean_rands[i, :] = np.mean(np.abs(Pxy), axis=1)

        counts = calculate_significance(Pxy_mean, cpsd_mean_rands, f)
        counts_df = pd.DataFrame()
        counts_df['p_vales'] = counts
        counts_df['f (cpm)'] = f
        counts_df.to_csv('dyadic_p_values.csv', index=False)


elif analysis_type == 'ind':

    pAs = data.T

    pAs_interp = []
    if interpolation:
        for time_series_ in range(num_participants):
            time_series_A = interp_helper(convert_value_to_NaN(pAs[:, time_series_], missing_value_code))
            pAs_interp.append(time_series_A)

        pAs = np.asarray(pAs_interp).T

        # save new dataframe
        interp_df = pd.DataFrame(pAs).T
        interp_df.columns = list(columns)
        interp_df.to_csv(filename.split('.csv')[0] + '_interp.csv', index=False)

    if windowing:
        pAs = pAs * window * acf

    fft_pAs = np.fft.fft(pAs, axis=0)
    fft_pAs = 2 * np.abs(fft_pAs[0:int(T / 2) + 1] / T)
    fft_pAs_mean = fft_pAs.mean(1)

    fft_df = pd.DataFrame()
    fft_df['amplitudes'] = fft_pAs_mean
    fft_df['f (cpm)'] = f
    fft_df.to_csv('./individual_fft_mean.csv', index=False)

    if significance_test:
        print('############ RUNNING BOOTSTRAPS #################')
        fft_mean_rands = np.zeros([num_bootstraps, f.shape[0]])

        for i in range(num_bootstraps):
            if i % 100 == 0:
                print('Running bootstrap number {} out of '.format(i), num_bootstraps)
            pAs_rand = np.random.permutation(pAs)
            fft_pAs_rand = np.fft.fft(pAs_rand, axis=0)
            fft_pAs_rand = 2 * np.abs(fft_pAs_rand[0:int(T / 2) + 1] / T)
            fft_mean_rands[i, :] = fft_pAs_rand.mean(1)

        counts = calculate_significance(fft_pAs_mean, fft_mean_rands, f)
        counts_df = pd.DataFrame()
        counts_df['p_vales'] = counts
        counts_df['f (cpm)'] = f
        counts_df.to_csv('individual_p_values.csv', index=False)
