# -*- coding: utf-8 -*-

import numpy as np

# RX Anomaly Detector
def RX_anomaly_detector(img):
    '''
    RX anomaly detector

    Find anomalies (different data from the background) in the image.

    Parameters
    ----------
    img: array-like of shape (n_pixels_row, n_pixels_col, n_features_per_pixel)


    Returns
    ----------
    res: array-like of shape (n_pixels_row, n_pixels_col)
    '''
    if len(img.shape) != 3:
        print("input shape must be equal to 3")
        return

    X = img.copy()
    X = X.reshape(-1, X.shape[-1])
    c = np.cov(X.T)
    inv_c = np.linalg.inv(c)
    res = np.empty((X.shape[0]))
    res = np.apply_along_axis(lambda x: x @ inv_c @ x, 1, X)
    res = np.abs(res)
    return res.reshape((img.shape[0], img.shape[1]))

