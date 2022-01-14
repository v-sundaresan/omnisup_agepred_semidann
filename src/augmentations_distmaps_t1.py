# Nicola Dinsdale 2018
# Augment the harp dataset
# Vaanathi Sundaresan 2018
# Adapted for WMH segmnetation - OPTIMA, MWSC datasets
########################################################################################################################
# Dependencies
import random
from scipy.ndimage.interpolation import shift
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import rotate, zoom
from skimage.util import random_noise
from skimage import exposure
import numpy as np
########################################################################################################################


def translate_it(image, t1, label, gmdist, ventdist):
    offsetx = random.randint(-5, 5)
    offsety = random.randint(-5, 5)
    is_seg = False
    order = 0 if is_seg is True else 5
    translated_im = shift(image, (offsetx, offsety, 0), order=order, mode='nearest')
    translated_t1 = shift(t1, (offsetx, offsety, 0), order=order, mode='nearest')
    translated_label = shift(label,(offsetx, offsety, 0), order=order, mode='nearest')
    translated_gmdist = shift(gmdist,(offsetx, offsety, 0), order=order, mode='nearest')
    translated_ventdist = shift(ventdist,(offsetx, offsety, 0), order=order, mode='nearest')
    return translated_im, translated_t1, translated_label, translated_gmdist, translated_ventdist

def translate_it2d(image, imaget1, label, gmdist, ventdist):
    offsetx = random.randint(-15, 15)
    offsety = random.randint(-15, 15)
    is_seg = False
    order = 0 if is_seg == True else 5
    translated_im = shift(image, (offsetx, offsety), order=order, mode='nearest')
    translated_imt1 = shift(imaget1, (offsetx, offsety), order=order, mode='nearest')
    translated_label = shift(label,(offsetx, offsety), order=order, mode='nearest')
    translated_gmdist = shift(gmdist,(offsetx, offsety), order=order, mode='nearest')
    translated_ventdist = shift(ventdist,(offsetx, offsety), order=order, mode='nearest')
    return translated_im, translated_imt1, translated_label, translated_gmdist, translated_ventdist

def scale_it(image, label):
    factor = random.uniform(0.8, 1.5)
    is_seg = False

    order = 0 if is_seg == True else 3

    height, width, depth = image.shape
    zheight = int(np.round(factor * height))
    zwidth = int(np.round(factor * width))
    zdepth = depth

    if factor < 1.0:
        newimg = np.zeros_like(image)
        newlab = np.zeros_like(label)
        row = (height - zheight) // 2
        col = (width - zwidth) // 2
        layer = (depth - zdepth) // 2
        newimg[row:row+height, col:col+zwidth, layer:layer+zdepth] = zoom(image, (float(factor), float(factor), 1.0),
                                                                          order=order, mode='nearest')[0:zheight, 0:zwidth, 0:zdepth]
        newlab[row:row+height, col:col+zwidth, layer:layer+zdepth] = zoom(label, (float(factor), float(factor), 1.0),
                                                                          order=order, mode='nearest')[0:zheight, 0:zwidth, 0:zdepth]
        return newimg, newlab

    elif factor > 1.0:
        row = (zheight - height) // 2
        col = (zwidth - width) // 2
        layer = (zdepth - depth) // 2

        newimg = zoom(image[row:row+zheight, col:col+zwidth, layer:layer+zdepth], (float(factor), float(factor), 1.0),
                      order=order, mode='nearest')
        newlab = zoom(label[row:row+zheight, col:col+zwidth, layer:layer+zdepth], (float(factor), float(factor), 1.0),
                      order=order, mode='nearest')

        extrah = (newimg.shape[0] - height) // 2
        extraw = (newimg.shape[1] - width) // 2
        extrad = (newimg.shape[2] - depth) // 2
        newimg = newimg[extrah:extrah+height, extraw:extraw+width, extrad:extrad+depth]

        extrah = (newlab.shape[0] - height) // 2
        extraw = (newlab.shape[1] - width) // 2
        extrad = (newlab.shape[2] - depth) // 2
        newlab = newlab[extrah:extrah+height, extraw:extraw+width, extrad:extrad+depth]
        return newimg, newlab

    else:
        return image, label

def scale_it2d(image, imaget1, label, gmdist, ventdist):
    factor = random.uniform(0.7, 1.3)
    order = 0

    height, width = image.shape
    zheight = int(np.round(factor * height))
    zwidth = int(np.round(factor * width))

    if factor < 1.0:
        newimg = np.zeros_like(image)
        newimgt1 = np.zeros_like(imaget1)
        newlab = np.zeros_like(label)
        newgmdist = np.zeros_like(gmdist)
        newventdist = np.zeros_like(ventdist)
        row = (height - zheight) // 2
        col = (width - zwidth) // 2
        newimg[row:row+height, col:col+zwidth] = zoom(image, (float(factor), float(factor)),
                                                                          order=order, mode='nearest')[0:zheight, 0:zwidth]
        newimgt1[row:row+height, col:col+zwidth] = zoom(imaget1, (float(factor), float(factor)),
                                                                          order=order, mode='nearest')[0:zheight, 0:zwidth]
        newlab[row:row+height, col:col+zwidth] = zoom(label, (float(factor), float(factor)),
                                                                          order=order, mode='nearest')[0:zheight, 0:zwidth]
        newgmdist[row:row+height, col:col+zwidth] = zoom(gmdist, (float(factor), float(factor)),
                                                                          order=order, mode='nearest')[0:zheight, 0:zwidth]
        newventdist[row:row+height, col:col+zwidth] = zoom(ventdist, (float(factor), float(factor)),
                                                                          order=order, mode='nearest')[0:zheight, 0:zwidth]
        return newimg, newimgt1, newlab, newgmdist, newventdist

    elif factor > 1.0:
        row = (zheight - height) // 2
        col = (zwidth - width) // 2
        newimg = zoom(image[row:row+zheight, col:col+zwidth], (float(factor), float(factor)),
                      order=order, mode='nearest')
        newimgt1 = zoom(imaget1[row:row+zheight, col:col+zwidth], (float(factor), float(factor)),
                      order=order, mode='nearest')
        newlab = zoom(label[row:row+zheight, col:col+zwidth], (float(factor), float(factor)),
                      order=order, mode='nearest')
        newgmdist = zoom(gmdist[row:row+zheight, col:col+zwidth], (float(factor), float(factor)),
                      order=order, mode='nearest')
        newventdist = zoom(ventdist[row:row+zheight, col:col+zwidth], (float(factor), float(factor)),
                      order=order, mode='nearest')
       
        extrah = (newimg.shape[0] - height) // 2
        extraw = (newimg.shape[1] - width) // 2
        newimg = newimg[extrah:extrah+height, extraw:extraw+width]
        
        extrah = (newimgt1.shape[0] - height) // 2
        extraw = (newimgt1.shape[1] - width) // 2
        newimgt1 = newimgt1[extrah:extrah+height, extraw:extraw+width]

        extrah = (newlab.shape[0] - height) // 2
        extraw = (newlab.shape[1] - width) // 2
        newlab = newlab[extrah:extrah+height, extraw:extraw+width]

        extrah = (newgmdist.shape[0] - height) // 2
        extraw = (newgmdist.shape[1] - width) // 2
        newgmdist = newgmdist[extrah:extrah+height, extraw:extraw+width]

        extrah = (newventdist.shape[0] - height) // 2
        extraw = (newventdist.shape[1] - width) // 2
        newventdist = newventdist[extrah:extrah+height, extraw:extraw+width]
        return newimg, newimgt1, newlab, newgmdist, newventdist

    else:
        return image, imaget1, label, gmdist, ventdist

def rotate_it2d(image, imaget1, label,gmdist, ventdist):
    theta = random.uniform(-15, 15)
    is_seg = False
    order = 0 if is_seg == True else 5
    new_img = rotate(image, float(theta), reshape=False, order=order, mode='nearest')
    new_imgt1 = rotate(imaget1, float(theta), reshape=False, order=order, mode='nearest')
    new_lab = rotate(label, float(theta), reshape=False, order=order, mode='nearest')
    new_lab = (new_lab > 0.5).astype(float)
    return new_img, new_imgt1, new_lab, gmdist, ventdist


def blur_it2d(image, imaget1, label, gmdist, ventdist):
    sigma = random.uniform(0.1,0.2)
    new_img = gaussian_filter(image, sigma)
    new_imgt1 = gaussian_filter(imaget1, sigma)
    return new_img, new_imgt1, label, gmdist, ventdist

def flip_it(image, t1, label, gmdist, ventdist):
    new_img = image[:,:, ::-1]
    new_t1 = t1[:,:, ::-1]
    new_lab = label[:,:, ::-1]
    new_gmdist = gmdist[:,:, ::-1]
    new_ventdist = ventdist[:,:, ::-1]
    return new_img, new_t1, new_lab, new_gmdist, new_ventdist

def add_noise_to_it(image,imaget1,label,gmdist,ventdist):
    new_img = random_noise(image,clip=False)
    new_imgt1 = random_noise(imaget1,clip=False)
    new_lab = label
    return new_img, new_imgt1, new_lab, gmdist, ventdist

def rescale_its_intensity(image,imaget1,label, gmdist, ventdist):
    min_limit = random.uniform(50,80)
    v_min, v_max = np.percentile(image, (min_limit, 100))
    new_img = exposure.rescale_intensity(image, in_range=(v_min, v_max))
    new_imgt1 = exposure.rescale_intensity(imaget1, in_range=(v_min, v_max))
    new_lab = label
    return new_img, new_imgt1, new_lab, gmdist, ventdist

def augment(image_to_transform, imaget1, label, gmdist, ventdist):
    # Image applies a random number of the possible transformations to the input image. Returns the transformed image
    # If label is none also applies to the image --> important for segmentation or similar
    """
    :param image_to_transform: input image as array
    :param label: optional: label for input image to also transform
    :return: transformed image, if label also returns transformed label
    """
    if len(image_to_transform.shape) == 3:
        # Add to the available transformations any functions from 3d you want to be applied
        available_transformations = {'flip': flip_it, 'noise': add_noise_to_it, 'translate': translate_it}
        # Decide how many of these transformations to apply
        num_transformations_to_apply = random.randint(1, len(available_transformations))
        num_transformations = 0
        transformed_image = None
        transformed_label = None

        while num_transformations <= num_transformations_to_apply:
            # choose which transformations to apply at random
            key = random.choice(list(available_transformations))
            transformed_image, transformed_imaget1, transformed_label, transformed_gmdist, transformed_ventdist = available_transformations[key](image_to_transform, imaget1, label, gmdist, ventdist)
            num_transformations += 1
        return transformed_image, transformed_imaget1, transformed_label, transformed_gmdist, transformed_ventdist


    if len(image_to_transform.shape) == 2:
        # Add to the available transformations any functions from 3d you want to be applied
        available_transformations = {'noise': add_noise_to_it, 'translate': translate_it2d, 'rotate': rotate_it2d, 'blur': blur_it2d}
        # Decide how many of these transformations to apply
        num_transformations_to_apply = random.randint(1, len(available_transformations))
        num_transformations = 0
        transformed_image = None
        transformed_label = None

        while num_transformations <= num_transformations_to_apply:
            # choose which transformations to apply at random
            key = random.choice(list(available_transformations))
            transformed_image, transformed_imaget1, transformed_label, transformed_gmdist, transformed_ventdist = available_transformations[key](image_to_transform, imaget1, label, gmdist, ventdist)
            num_transformations += 1
        return transformed_image, transformed_imaget1, transformed_label, transformed_gmdist, transformed_ventdist
    else:
        raise Exception('Invalid dimensions for image augmentation - currently only supported in 3d')
