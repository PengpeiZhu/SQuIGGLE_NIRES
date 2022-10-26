#!/usr/bin/env python
# coding: utf-8

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import sys
import glob
from scipy.ndimage.interpolation import shift
from scipy.signal import find_peaks
from scipy.signal import find_peaks_cwt
from scipy.signal import peak_prominences
from scipy.signal import peak_widths
import itertools
from scipy.optimize import curve_fit
import csv

def find_edge_coord(slices, i):
    edge_coord1 = []
    edge_coord2 = []
    x_lim = []
    # define y coordinates of the slice range.
    lo = slices[i][0]
    up = slices[i][1]
    # go through the y axis, find the edges by finding the peaks along the y-axis
    # average over every 3 pixels on the x-aixs.
    for x in range(2, len(flat_data[1]), 3):
        # if x is the last element, pass.
        if x == 2048: pass
        else:
            # average over every 3 pixels along x-axis.
            edge = (edges[lo:up, x - 1] + edges[lo:up, x] +
                    edges[lo:up, x + 1]) / 3
            # find the peaks and prominences along y-axis.
            peaks, properties = find_peaks(edge, width=(2, 4), prominence=0.04)
            prominences = properties['prominences']
            # define the y-coordinate of a peak to be the middle value of that peak's FWHM.
            peak_coord = (properties['right_ips'] +
                          properties['left_ips']) / 2 + lo - (shift_pix / 2)
            ########################################################################
            ########################################################################
            # for the first and last slice range there's only one edge to be found.
            if i == 0:
                # provide a limit on the x-axis for the spectrum
                # to avoid the over scanned area.
                edge_coord2.append(np.nan)
                x_lim.append(1100)
                if x <= 1100:
                    # if no peak is found, append zero value to the list.
                    if peak_coord.size == 0:
                        edge_coord1.append(0)
                    elif prominences.size >= 1:
                        # the peaks with the max values.
                        peak_id1 = np.argmax(prominences)
                        edge_coord1.append(peak_coord[peak_id1])
                else:
                    pass
                    ########################################################################
            ########################################################################
            # for the first and last slice range there's only one edge to be found.
            if i == 5:
                x_lim.append(2048)
                edge_coord2.append(np.nan)
                # if no peak is found, append zero value to the list.
                if peak_coord.size == 0:
                    edge_coord1.append(0)
                elif prominences.size >= 1:
                    # find the peaks with the max values.
                    peak_id1 = np.argmax(prominences)
                    edge_coord1.append(peak_coord[peak_id1])
            ########################################################################
            ########################################################################
            # for the second slice range, there are two edeges but the lower edge
            # ends before the upper edge.
            elif i == 1:
                ########################################################################
                # provide a limit on the x-axis for the spectrum
                # to avoid the over scanned area.
                x_lim.append(1000)
                if x <=1000:
                    # if found less than two peaks, append zero value to the list.
                    if peak_coord.size <= 1:
                        edge_coord1.append(0)
                        edge_coord2.append(0)
                    # if found exact two peaks
                    elif prominences.size == 2:
                        # find the max and 2nd max peak values
                        # for the first slice range peaks of the upper edge are always greater
                        peak_id1 = np.argmax(prominences)
                        peak_id2 = np.where(prominences == np.partition(
                            prominences, -2)[-2])[0][0]
                        edge_coord1.append(peak_coord[peak_id1])
                        edge_coord2.append(peak_coord[peak_id2])

                    elif prominences.size > 2:
                        peak_id1 = np.argmax(prominences)
                        peak_id2 = np.where(prominences == np.partition(
                            prominences, -2)[-2])[0][0]
                        peak_id3 = np.where(prominences == np.partition(
                            prominences, -3)[-3])[0][0]

                        if peak_coord[peak_id1] > peak_coord[
                                peak_id2] and peak_coord[
                                    peak_id1] < peak_coord[peak_id3]:
                            edge_coord1.append(peak_coord[peak_id1])
                            edge_coord2.append(peak_coord[peak_id2])
                        elif peak_coord[peak_id1] < peak_coord[
                                peak_id2] and peak_coord[
                                    peak_id1] > peak_coord[peak_id3]:
                            edge_coord1.append(peak_coord[peak_id1])
                            edge_coord2.append(peak_coord[peak_id3])

                        elif peak_coord[peak_id1] > peak_coord[
                                peak_id2] and peak_coord[
                                    peak_id1] > peak_coord[peak_id3]:
                            edge_coord1.append(peak_coord[peak_id1])
                            edge_coord2.append(peak_coord[peak_id2])

                        elif peak_coord[peak_id1] < peak_coord[
                                peak_id2] and peak_coord[
                                    peak_id1] < peak_coord[peak_id3]:
                            peak_id4 = np.where(prominences == np.partition(
                                prominences, -4)[-4])[0][0]
                            edge_coord1.append(peak_coord[peak_id1])
                            edge_coord2.append(peak_coord[peak_id4])
                ########################################################################
                elif x>1000:
                    if peak_coord.size == 0:
                        edge_coord1.append(0)
                    elif prominences.size >= 1:
                        # the peak with the max value
                        peak_id1 = np.argmax(prominences)
                        edge_coord1.append(peak_coord[peak_id1])
            ########################################################################
            ########################################################################
            # for all the other slice ranges
            elif i == 2 or i == 3 or i == 4:
                x_lim.append(2048)
                if x<=2048:
                    if peak_coord.size <= 1:
                        edge_coord1.append(0)
                    elif prominences.size > 1:
                        # find the max and 2nd max peaks
                        peak_id1 = np.argmax(prominences)
                        peak_id2 = np.where(
                            prominences == np.partition(prominences, -2)[-2])[0][0]
                        # determine which edge the peak belongs two by comparing
                        # their coordinates.
                        if peak_coord[peak_id1] > peak_coord[peak_id2]:
                            edge_coord1.append(peak_coord[peak_id1])
                            edge_coord2.append(peak_coord[peak_id2])
                        else:
                            edge_coord1.append(peak_coord[peak_id2])
                            edge_coord2.append(peak_coord[peak_id1])
    return edge_coord1, edge_coord2, x_lim[0]
    
##################################################################################
##################################################################################
    
def eliminate_outliers(edge_coord1, out):
    # eliminate the outliers by
    # giving them zero values
    for i in range(len(edge_coord1)):
        local_median_elements = []
        # the first 100 elements
        if i <= 42:
            # define the elements equal or greater than the 'out' parameter
            # comparing with the local median of 40 values
            for n in range(80):
                local_median_elements.append(edge_coord1[i + n])
            if abs(edge_coord1[i] - np.median(local_median_elements)) >= out:
                edge_coord1[i] = 0
        # the last 100 elements
        elif i >= len(edge_coord1) - 41:
            for n in range(80):
                local_median_elements.append(edge_coord1[i - n])
            if abs(edge_coord1[i] - np.median(local_median_elements)) >= out:
                edge_coord1[i] = 0
        # everything else
        else:
            for n in range(40):
                local_median_elements.append(edge_coord1[i - n])
                local_median_elements.append(edge_coord1[i + n])
            if abs(edge_coord1[i] - np.median(local_median_elements)) >= out:
                edge_coord1[i] = 0
    return edge_coord1

##################################################################################
##################################################################################

def replace_zeros(edge_coord1, replace_range):
    # replace the zero values with the mean of replace_range(number) pxiel values
    for index, value in enumerate(edge_coord1):
        if value == 0:
            replace = []
            # for the last elements
            if index + replace_range >= len(edge_coord1) - 1:
                # determine if the designated replacement is zero or not
                # if not zero then write to the replace list
                for r in range(replace_range):
                    if not edge_coord1[index - r] == 0:
                        replace.append(edge_coord1[index - r])
                # calculate the mean of the nonzero values in the replace list
                replace = np.mean(replace)
                # if no mean value is avaliable, remain the origional value
                if np.isnan(replace): edge_coord1[index] = value
                # else replace the zero value with the replacement value
                else:
                    edge_coord1[index] = value + replace
            # for the first elements
            elif index <= replace_range:
                for r in range(replace_range):
                    if not edge_coord1[index + r] == 0:
                        replace.append(edge_coord1[index + r])
                replace = np.mean(replace)
                if np.isnan(replace): edge_coord1[index] = value
                else:
                    edge_coord1[index] = value + replace
            # for everything else
            else:
                for r in range(replace_range):
                    if not edge_coord1[index -
                                       r] == 0 and not edge_coord1[index +
                                                                   r] == 0:
                        replace.append(edge_coord1[index - r])
                        replace.append(edge_coord1[index + r])
                replace = np.mean(replace)
                if np.isnan(replace): edge_coord1[index] = value
                else:
                    edge_coord1[index] = value + replace

        else:
            pass

    return edge_coord1
