#!/usr/bin/env python
# coding: utf-8

def strighten(data,x_lim,polyfit_parameters):
    # first caculate pn with poly1d form the polyfit_parameters
    pn = []
    for n in range (polyfit_parameters.shape[0]):
        pn.append(np.poly1d(polyfit_parameters[n,:]))
    # generate the x axis based on the right-side x limit
    x = []
    for n in range (len(x_lim)):
        x.append(np.linspace(0,x_lim[n],int(x_lim[n])))
    # strighten every spectrum, from lower to upper
    strighten =[]
    for n in range(len(x)):
        # strighten based on the lower edge
        if n%2 ==0:
            # define lower and upper limits for spectrm width
            y_lo = pn[n](x[n]).astype(int)
            y_up = pn[n+1](x[n+1]).astype(int)
            # caculate spectrum width based on the left edge
            width = y_up[0]-y_lo[0]
            # define a stright spectrum zero array
            stright_n = np.zeros((width,int(x_lim[n])))
            for i in range(int(x_lim[n])):
                # determine if the ith column have the same size
                # as the width defined based on the left edge
                if (y_up[i]-y_lo[i]) == width:
                    # rewrite the data in between two edges into the new array
                    stright_n[:,i] = data[y_lo[i]:y_up[i],i]
                # if size not macth then corret to make
                # the ith column to have the same width
                else:
                    delta = (y_up[i]-y_lo[i])-width
                    stright_n[:,i] = data[y_lo[i]:(y_up[i]-delta),i]
            strighten.append(stright_n)
        else: pass
    return strighten,x
