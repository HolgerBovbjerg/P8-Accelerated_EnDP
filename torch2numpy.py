# -*- coding: utf-8 -*-
"""
Created on Mon May  3 10:33:24 2021

@author: holge
"""

import numpy as np


def torch2numpy(image, filter):
    npimage = np.array(image)
    npfilter = np.array([filter])
    return (npimage, npfilter)
