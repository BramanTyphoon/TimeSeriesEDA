#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 22:51:40 2020

@author: bramante
"""

import numpy as np
import argparse
from matplotlib import pyplot as plt

def EOF(arr,normalize=True):
    """
    Calculates the empirical orthogonal function of a matrix

    Parameters
    ----------
    arr : numpy.ndarray
        n-dimensional array, where the first dimension is time and subsequent
        dimensions represent different sampling points (in space, between
        subjects, or otherwise) among which you want to analyze covariance
    normalize : bool, optional
        Flag whether to normalize the covariance matrix by the number of time
        slices examined (unbiased). Default is True

    Returns
    -------
    tuple of numpy.ndarray
        The single-value decomposition of the covariance matrix of arr (u,s,v),
        where s is the singular values, u cols are the eigenvectors of the
        matrix COV*COV^T, and v rows are the eigenvectors of the matrix 
        COV^T*COV

    """
    
    # If input array has more than 2 dimensions, collapse to 2
    in_shape = arr.shape
    if np.ndim(arr) > 2:
        arr = np.reshape(arr,(in_shape[0],np.cumprod(in_shape[1:])[-1]),order='F')
    
    arr_dev = arr - np.mean(arr, axis=0)
    
    # Calculate the covariance matrix
    cov = np.matmul(arr_dev.T,arr_dev)
    if normalize:
        cov = cov / (in_shape[0] - 1)
        
    u,s,v = np.linalg.svd(cov)
    
    # Transform the output
    eof = np.matmul(arr,u)
    eof_maps = np.reshape(u,(in_shape[1],in_shape[2],u.shape[1]),order='F')
    rsquared = s / np.sum(s)
    
    return(eof,eof_maps,rsquared)

def MCA(arr1,arr2,normalize=True):
    
    # Input arrays must have the same lenth 0th axis
    s1 = arr1.shape
    s2 = arr2.shape
    if not (s1[0] == s2[0]):
        raise ValueError('Input arrays to MCA must have the same length 0th axis.')
    # If input arrays have more than 2 dimensions, collapse to 2
    if np.ndim(arr1) > 2:
        arr1 = np.reshape(arr1,(s1[0],np.cumprod(s1[1:])[-1]),order='F')
    if np.ndim(arr2) > 2:
        arr2 = np.reshape(arr2,(s1[0],np.cumprod(s2[1:])[-1]),order='F')
    
    # Calculate the cross-covariance matrix
    arr1_dev = arr1 - np.mean(arr1,axis=0)
    arr2_dev = arr2 - np.mean(arr2,axis=0)
    cov = np.matmul(arr1_dev.T,arr2_dev)
    if normalize:
        cov = cov / (s1[0] - 1)
        
    u,s,v = np.linalg.svd(cov)
    
    # Transform the output
    eof1 = np.matmul(arr1,u)
    eof2 = np.matmul(arr2,v.T)
    eof_map1 = np.reshape(u,(s1[1],s1[2],u.shape[1]),order='F')
    eof_map2 = np.reshape(v.T,(s2[1],s2[2],v.shape[1]),order='F')
    rsquared = s / np.sum(s)
    
    return(eof1,eof2,eof_map1,eof_map2,rsquared)

def parse_arguments():
    """
    Parse command line arguments. Specifically, allow tests to be run
    automatically
    
    Returns
    -------
    dict
    """
    # Instantiate an argument parser, and return the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--test',type=str, metavar='NAME', nargs='+',help='Flag for running tests on a specific method')
    args = parser.parse_args()
    return vars(args)

def test(tests):
    ## Final variables
    SQUARE_SIZE = 50
    TIME_LENGTH = 100
    RANDOM_SEED = 42
    BASE_MAGNITUDE = 0.01
    np.random.seed(RANDOM_SEED)
    
    if type(tests) == list:
        for t in tests:
            test(t)
    else:
        # Generate a spatial variable that varies over time according to a
        # linear combination of trends and periodic components
        time = np.arange(TIME_LENGTH)
        gridx,gridy,gridt = np.meshgrid(np.arange(SQUARE_SIZE+1),
                                        np.arange(SQUARE_SIZE+1),
                                        time)
        slope = BASE_MAGNITUDE / TIME_LENGTH * \
            (-np.sqrt(np.power(SQUARE_SIZE/2 - gridx, 2) + 
               np.power(SQUARE_SIZE/2 - gridy, 2)) + SQUARE_SIZE/4)
        spatial_var1a = np.random.random(list(gridx.shape[:2])+[1]) + slope * (gridt)
        spatial_var1b = BASE_MAGNITUDE / 5 * (gridx - SQUARE_SIZE/2) * \
            np.cos(2 * 4 * np.pi / TIME_LENGTH * gridt)
        spatial_var1c = BASE_MAGNITUDE / 10 * (gridy - SQUARE_SIZE/2) * \
            np.cos(2 * 10 * np.pi / TIME_LENGTH * gridt)
        spatial_var1 = spatial_var1a + spatial_var1b + spatial_var1c
        
        gridx2,gridy2,gridt2 = np.meshgrid(np.arange(SQUARE_SIZE*1.5),
                                           np.arange(SQUARE_SIZE*1.5),
                                           time)
        slope2 = BASE_MAGNITUDE * 5 / TIME_LENGTH * \
            (np.sqrt(np.power(SQUARE_SIZE*1.5/2 - gridx2, 2) + 
               np.power(SQUARE_SIZE*1.5/2 - gridy2, 2)) - SQUARE_SIZE*1.5/4)
        spatial_var2a = np.random.random(list(gridx2.shape[:2])+[1]) + slope2 * (gridt2)
        spatial_var2b = -BASE_MAGNITUDE * (gridx2 - SQUARE_SIZE*1.5/2) * \
            np.cos(2 * 4 * np.pi / TIME_LENGTH * gridt2)
        spatial_var2c = BASE_MAGNITUDE / 2 * (gridy2 - SQUARE_SIZE*1.5/2) * \
            np.cos(2 * 9 * np.pi / TIME_LENGTH * gridt2)
        spatial_var2 = spatial_var2a + spatial_var2b + spatial_var2c
        
        # Perform an empirical orthogonal function analysis to separate the
        # components of variability
        eof, maps, rs = EOF(np.transpose(spatial_var1,(2,0,1)))
        
        # Perform maximum cross-covariance analysis instead
        eof1,eof2,eof1m,eof2m,rs2 = MCA(np.transpose(spatial_var1,(2,0,1)),
                                        np.transpose(spatial_var2,(2,0,1)))
        
        if tests.lower() == 'eof':
            # Plot the top 5 EOFs
            for ii in range(5):
                fig = plt.figure(constrained_layout=True)
                gs = fig.add_gridspec(4,1)
                ax = []
                ax += [fig.add_subplot(gs[0])]
                ax += [fig.add_subplot(gs[1:])]
                ax[0].plot(eof[:,ii])
                ax[0].set_xlabel('Time')
                pcm = ax[1].pcolor(gridx[:,:,0],gridy[:,:,0],maps[:,:,ii],cmap='seismic')
                fig.colorbar(pcm,ax=ax[1])
        
        if tests.lower() == 'mca':
            # Plot the top 5 MCA EOFs
            for ii in range(5):
                fig = plt.figure(constrained_layout=True)
                gs = fig.add_gridspec(4,2)
                ax = []
                ax += [fig.add_subplot(gs[0,0])]
                ax += [fig.add_subplot(gs[1:,0])]
                ax += [fig.add_subplot(gs[0,1])]
                ax += [fig.add_subplot(gs[1:,1])]
                ax[0].plot(eof1[:,ii])
                ax[0].set_xlabel('Time')
                pcm = ax[1].pcolor(gridx[:,:,0],gridy[:,:,0],eof1m[:,:,ii],cmap='seismic')
                fig.colorbar(pcm,ax=ax[1])
                ax[1].set_aspect('equal')
                ax[2].plot(eof2[:,ii])
                ax[2].set_xlabel('Time')
                pcm = ax[3].pcolor(gridx2[:,:,0],gridy2[:,:,0],eof2m[:,:,ii],cmap='seismic')
                fig.colorbar(pcm,ax=ax[3])
                ax[3].set_aspect('equal')
        
        
        



def main(args):
    if args['test'] is not None:
        for arg in args['test']:
            test(arg)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)