import os
from glob import glob
import pydicom
import numpy as np
from matplotlib import pyplot as plt
import cv2

def set_outside_scanner_to_air(hu_pixelarrays):
    """
    Pixel Padding Value Attribute(0028,0120) -> air
    """
    hu_pixelarrays[hu_pixelarrays < -1024] = -1024
    
    return hu_pixelarrays
    

def dcms_to_imgs(dcms):
    dcms.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    images = np.stack([file.pixel_array for file in dcms])
    images = images.astype(np.int16)

    # convert to HU
    intercept = dcms[0].RescaleIntercept
    slope = dcms[0].RescaleSlope
    hu_images = images.astype(np.float64) * slope + intercept
    hu_images = set_outside_scanner_to_air(hu_images.astype(np.int16))
    return hu_images


def get_frames(BASE_DIR, series_num):

    # Get all DICOM files in the base directory
    path = f"{glob(BASE_DIR)[2]}/*.dcm"
    
    # Save all slices corresponding the series number
    slices = []
    for fname in sorted(glob(path)):
        try:
            if pydicom.dcmread(fname, force=True)[(0x020, 0x037)].value[4] == series_num:
                slices.append(pydicom.dcmread(fname, force=True))
        except:
            pass
    
    slices = sorted(slices, key=lambda s: s[0x020, 0x032][-1], reverse=True)

    return slices

def make_and_save_drr(slices):
    ps = slices[0].PixelSpacing
    ss = slices[0][0x020, 0x032][-1] - slices[1][0x020, 0x032][-1]
    ax_aspect = ps[1]/ps[0]
    sag_aspect = ss/ps[1]
    cor_aspect = ss/ps[0]
    
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape)

    for i, s in enumerate(slices):
        try:
            img2d = s.pixel_array
            img3d[:, :, i] = img2d
        except:
            print(i, sorted(glob(path)[i]))

    img_numpy = np.mean(img3d.T, axis=2)[-512:]

    drr_arr = (img_numpy - np.min(img_numpy)) / (np.max(img_numpy) - np.min(img_numpy))
    drr_arr = np.array(drr_arr*255).astype(np.uint8)

    cubic_img = cv2.resize(img, (img_numpy.shape[1], int(img_numpy.shape[0]*cor_aspect)), interpolation = cv2.INTER_CUBIC)
    
    cv2.imwrite("rescale_2.png", cubic_img)





