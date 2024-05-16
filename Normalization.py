import numpy as np
#-----------------------------Z-score Normalization-------------------------------#
#Normalize pixel distribution of image 
def normalization(image_array):
    max_pixel_image = np.amax(image_array)
    image_array = (max_pixel_image - image_array) / max_pixel_image
    return image_array


