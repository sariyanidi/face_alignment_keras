#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 16:46:16 2021

@author: sariyanidi
"""

import numpy as np
import gzip
import cv2


def recolor_image(image, c1, c2, c3):
    image[:,:,0] = c1*image[:,:,0]
    image[:,:,1] = c2*image[:,:,1]
    image[:,:,2] = c3*image[:,:,2]
    
    image[image>1.0] = 1.0
    image[image<0.0] = 0.0

    return image


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def scale_image(image, s):
    image_center = tuple(np.array(image.shape[1::-1], dtype=np.float32) / 2)
    image_center_after = tuple(s*np.array(image.shape[1::-1], dtype=np.float32) / 2)
    image_shift = (image_center[0]-image_center_after[0],
                 image_center[1]-image_center_after[1])
  
    M = np.float32([[s,0,image_shift[0]],[0,s,image_shift[1]]])
  
    result = cv2.warpAffine(image, M, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def perturb_image(image, s, angle):
    return scale_image(rotate_image(image, angle), s)


def tfdata_generator(imagepaths, labelpaths, is_training, batch_size=20):
    import tensorflow as tf
    '''Construct a data generator using tf.Dataset'''

    def im_file_to_tensor(impath, labelpath, is_training):
        def _im_file_to_tensor(impath, labelpath, is_training):
            image_decoded = tf.image.decode_png(tf.io.read_file(impath), channels=3)
            image_decoded = np.array(image_decoded, dtype=np.float32)/ 255.0
            
            f = gzip.GzipFile(labelpath.numpy(), "r")
            arr_gz = np.load(f)
            f.close()
            
            if is_training:
                scale_factor = 0.3
                rot_factor = 30
                
                rand_angle = np.clip(np.random.randn()*30, -2*rot_factor, 2*rot_factor)
                rand_scale = np.clip(np.random.randn()*scale_factor + 1, 1-scale_factor, 1+scale_factor)
                
                c1 = np.random.rand()*0.9+0.2
                c2 = np.random.rand()*0.9+0.2
                c3 = np.random.rand()*0.9+0.2
                
                image_decoded = recolor_image(perturb_image(image_decoded, rand_scale, rand_angle), c1, c2, c3)
                arr_gz = perturb_image(arr_gz, rand_scale, rand_angle)
            
            im = tf.cast(image_decoded, tf.float32) 
            arr_gz = tf.cast(arr_gz, tf.float32)
            label = tf.convert_to_tensor(arr_gz)
            
            return im, label
        
        return tf.py_function(_im_file_to_tensor, 
                              inp=(impath, labelpath, is_training), 
                              Tout=(tf.float32, tf.float32))
    
    dataset = tf.data.Dataset.from_tensor_slices((imagepaths, labelpaths, [is_training for i in range(len(imagepaths))]))
        
    if is_training: 
        dataset = dataset.repeat().shuffle(len(imagepaths)).map(im_file_to_tensor).batch(batch_size)#
    else:
        dataset = dataset.repeat().map(im_file_to_tensor).batch(batch_size)#
        
    return dataset
    #return dataset.prefetch(tf.data.experimental.AUTOTUNE)


def transform(point, center, scale, resolution, invert=False):
    """
    The code below is directly adapted from the original method's repo, 
    i.e., from https://github.com/1adrianb/face-alignment
    
    We simply remove the dependence to torch from the original code
    """
    """Generate and affine transformation matrix.

    Given a set of points, a center, a scale and a targer resolution, the
    function generates and affine transformation matrix. If invert is ``True``
    it will produce the inverse transformation.

    Arguments:
        point {torch.tensor} -- the input 2D point
        center {torch.tensor or numpy.array} -- the center around which to perform the transformations
        scale {float} -- the scale of the face/object
        resolution {float} -- the output resolution

    Keyword Arguments:
        invert {bool} -- define wherever the function should produce the direct or the
        inverse transformation matrix (default: {False})
    """
    _pt = np.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = np.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)
    
    if invert:
        t = np.linalg.inv(t)

    new_point = (np.matmul(t, _pt))[0:2]
    

    return np.array(new_point, dtype=int)#.int()


def crop(image, center, scale, resolution=256.0):
    """
    The code below is directly adapted from the original method's repo, 
    i.e., from https://github.com/1adrianb/face-alignment
    
    We simply remove the dependence to torch from the original code
    """
    
    """
    Center crops an image or set of heatmaps

    Arguments:
        image {numpy.array} -- an rgb image
        center {numpy.array} -- the center of the object, usually the same as of the bounding box
        scale {float} -- scale of the face

    Keyword Arguments:
        resolution {float} -- the size of the output cropped image (default: {256.0})

    Returns:
        [type] -- [description]
    """  # Crop around the center point
    """ Crops the image around the center. Input is expected to be an np.ndarray """
    ul = transform([1, 1], center, scale, resolution, True)
    br = transform([resolution, resolution], center, scale, resolution, True)
    
    if image.ndim > 2:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0],
                           image.shape[2]], dtype=np.int32)
        newImg = np.zeros(newDim, dtype=np.uint8)
    else:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
        newImg = np.zeros(newDim, dtype=np.uint8)
        
    ht = image.shape[0]
    wd = image.shape[1]
    newX = np.array(
        [max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
    newY = np.array(
        [max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
    
    oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
    oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
    newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1]
           ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]
    newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)),
                        interpolation=cv2.INTER_LINEAR)
    
    return newImg


def get_landmarks(imp, landmark_net, ptCenter, scale):
    landmark_net.setInput(cv2.dnn.blobFromImage(np.float32(cv2.cvtColor(imp, cv2.COLOR_BGR2RGB))/255))
    y_dnn = landmark_net.forward()
    
    return heatmaps_to_landmarks(y_dnn, ptCenter, scale)


def preprocess_image(im, x0, y0, xf, yf):
    face_width = xf - x0
    face_height = yf - y0
    
    ptCenter = np.array([(x0+x0+face_width)/2.0, (y0+y0+face_height)/2.0])
    scale = (face_width+face_height)/195.0
    
    return (crop(im, ptCenter, scale), ptCenter, scale)


def heatmaps_to_landmarks(hm, ptCenter, scale, resolution=64):
    """
    The code below is directly adapted from the original method's repo, 
    i.e., from https://github.com/1adrianb/face-alignment
    
    We simply remove the dependence to torch from the original code
    and make it run with OpenCV
    """
    p = np.zeros((68,2), dtype=int)
    
    for i in range(68):
        (_, _, _, maxLoc) = cv2.minMaxLoc(hm[0,i,:,:])
        
        px = maxLoc[0]
        py = maxLoc[1]
        
        if px > 0 and px < 63 and py > 0 and py < 63:
            diffx = hm[0,i,py,px+1] - hm[0,i,py,px-1]
            diffy = hm[0,i,py+1,px] - hm[0,i,py-1, px];
            
            px += 1
            py += 1 
            
            if diffx > 0:
                px += 0.25
            else:
                px -= 0.25
        
            if diffy > 0:
                py += 0.25
            else:
                py -= 0.25
    
        px -= 0.5
        py -= 0.5

        ptOrig = transform((px, py), ptCenter, scale, resolution, True)
        p[i,0] = ptOrig[0]
        p[i,1] = ptOrig[1]

    return p

