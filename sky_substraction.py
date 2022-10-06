#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import sys
import glob
from scipy.ndimage.interpolation import shift


#define a functioon that saves the fits
def write_hdulist_to(hdulist, fileobj, overwrite=True, **kwargs):
    hdulist.writeto(fileobj, overwrite=True, **kwargs)


''' NIRES header keywords of interest
Dither pattern name: DPATNAME
Dither pattern step size in X: DPATSTPX
Dither pattern step size in Y: DPATSTPY
Current frame dither position X offset: XOFFSET
Current frame dither position Y offset: YOFFSET
Pixel scale: PSCALE
ITIME: Integration time
DATAFILE: image saved data file name
OBJECT: object name
TARGNAME: target name
OBSTYPE: observation type
'''

"""
+
 NAME:
   SKY_SUBSTRACTION

 PURPOSE:
   
 CALLING SEQUENCE:
   sky_substraction, date, name, save = True

 INPUT PARAMETERS:
    <date> - string in the form of 20yy-mm-dd
    <name> - string in the form of Jxxxx+xxxx or SDSS_Jxxxx+xxxx, find the name in the observation log
    <save> - save the final fits file or not. Defult is yes
    <plot> - plot a preview of the fits file. Defult is no
    
 OUTPUT:
    A fits file that saubstracted the sky frames and stacked together with median
"""


def sky_substraction(date, name, save=True, plot=False):

    # import all flat frames
    flat_list = glob.glob('NIRES/' + date + '/Flat/*.fits')

    # define a zero array for master flat
    m_flat = np.zeros((fits.getdata(flat_list[0]).shape[0],
                       fits.getdata(flat_list[0]).shape[1], len(flat_list)))
    # get a master flat
    for i in range(len(flat_list)):
        flat_data = fits.getdata(flat_list[i])
        m_flat[:, :, i] = flat_data
    # normalize the flats, then combine them together with median
    m_flat = m_flat / np.median(m_flat)
    m_flat = np.median(m_flat, axis=2)

    # import all the science and telluric frames from one observation
    frame_list = glob.glob('NIRES/' + date + '/' + 'Science&Telluric/*.fits')

    all_headers = []
    all_data = []
    for frame in range(len(frame_list)):
        hdul = fits.open(frame_list[frame])
        all_headers.append(hdul[0].header)
        all_data.append(hdul[0].data)
        hdul.close()

    # pick all the frames from frame list for
    # the designated object, and apply master flat
    data = []
    headers = []
    for i in range(len(all_headers)):
        if all_headers[i]['TARGNAME'] == name and all_headers[i]['DATAFILE'][
                0] == 's':
            headers.append(all_headers[i])
            # apply master flat on data, ignore the 0 values in the flat
            data.append(
                np.divide(all_data[i],
                          m_flat,
                          out=all_data[i],
                          where=m_flat != 0))

    # perform data substraction
    data_sub = []
    if headers[0]['DPATNAME'] == 'ABC' or 'NONE':
        for i in range(len(headers)):
            # first frame: use only second frame as sky
            if i == 0:
                data_sub.append(data[0] - data[1])
            # last frame: use only previous frame as sky
            elif i == len(headers) - 1:
                data_sub.append(data[-1] - data[-2])
            # everything in between: take mean of two surrounding exposures as the sky
            # also make sure every surrounding exposures have offsets
            else:
                if abs(headers[i]['YOFFSET'] - headers[i - 1]['YOFFSET']
                       ) >= 0 and abs(headers[i]['YOFFSET'] -
                                      headers[i + 1]['YOFFSET']) >= 0:

                    sky = np.mean([data[i - 1], data[i + 1]], axis=0)
                    data_sub.append(data[i] - sky)
    # else for ABBA dither pattern
    elif headers[0]['DPATNAME'] == 'ABBA':
        for i in range(len(headers)):
            if not i % 2:  # this is 0/2/4 exposure-- ie, start of a pair. use following one to subtract
                data_sub.append(data[i] - data[i + 1])
            else:
                data_sub.append(data[i] - data[i - 1])

    # make list of offsets from headers
    offsets = []
    for i in range(len(headers)):
        offsets.append(np.array([headers[i]['YOFFSET'],
                                 headers[i]['XOFFSET']]))
    pix_scale = headers[0]['PSCALE']

    # shift frames and add them back together
    # use the offset of the first frame as a base! don't offset things that are the same offset as this
    data_shift = []
    for i in range(len(headers)):
        # if same offset as first frame, don't shift
        if np.array_equal(offsets[i], offsets[0]):
            data_shift.append(data_sub[i])
        # otherwise, shift to where the first frame is
        else:
            data_shift.append(
                shift(data_sub[i], (offsets[i] - offsets[0]) / pix_scale))
    med = np.median(data_shift, axis=0)

    if plot:
        # for a graph preview
        plt.figure(figsize=(16, 8))
        plt.imshow(med,
                   origin='lower',
                   interpolation='nearest',
                   cmap='Greys',
                   norm=LogNorm(),
                   aspect='auto')
        plt.colorbar()

    if save:
        med_hdu = fits.PrimaryHDU(med)
        filebase = headers[0]['DATAFILE'][1:8]
        write_hdulist_to(
            med_hdu,
            'NIRES/' + date + '/' + 'skysub_' + filebase + name + '.fits')
        
        
    
def find_targnames(date):
    frame_list = glob.glob('NIRES/' + date + '/' + 'Science&Telluric/*.fits')

    all_headers = []
    for frame in range(len(frame_list)):
        hdul = fits.open(frame_list[frame])
        all_headers.append(hdul[0].header)
        hdul.close()
    targnames = []
    for i in range(len(all_headers)):
        targnames.append(all_headers[i]['TARGNAME'])
    
    targnames = set(targnames)

    return targnames

