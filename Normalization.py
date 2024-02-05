import numpy as np
import torch
#-----------------------------Z-score Normalization-------------------------------#

def normalization(image_array):
    max_pixel_image = np.amax(image_array)
    image_array = (max_pixel_image - image_array) / max_pixel_image
    return image_array

# Example usage
#image_array = np.random.randint(0, 10 , size=(5,5))  # Replace with your actual image array
#normalized_array = normalization(image_array)

#print ('Array')
#print(image_array)

#print("Normalized array:")
#print(normalized_array)


