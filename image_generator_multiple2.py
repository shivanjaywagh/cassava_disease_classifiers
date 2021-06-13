# https://youtu.be/ccdssX4rIh8
"""
@author: Sreenivas Bhattiprolu
Image shifts via the width_shift_range and height_shift_range arguments.
Image flips via the horizontal_flip and vertical_flip arguments.
Image rotations via the rotation_range argument
Image brightness via the brightness_range argument.
Image zoom via the zoom_range argument.
"""





from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from skimage import io

# Construct an instance of the ImageDataGenerator class
# Pass the augmentation parameters through the constructor. 

datagen = ImageDataGenerator(
        rotation_range=30,     #Random rotation between 0 and 45
        width_shift_range=0.1,   #% shift
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.05,
        horizontal_flip=True,
		fill_mode='constant', cval=0)    #Also try nearest, constant, reflect, wrap




#x = io.imread("D:/fortiate/Google-Image-Scraper-master-Copy/aadhar_selected/aadhar_card14.jpg")  #Array with shape (256, 256, 3)

# Reshape the input image because ...
#x: Input data to datagen.flow must be Numpy array of rank 4 or a tuple.
#First element represents the number of images
#x = x.reshape((1, ) + x.shape)  #Array with shape (1, 256, 256, 3)







#Multiple images.
#Manually read each image and create an array to be supplied to datagen via flow method


import numpy as np
from skimage import io
import os
from PIL import Image
import piexif

image_directory = "D:/cassava/class_4_2/"
SIZE = 256
dataset = []



#import piexif
# suppose im_path is a valid image path
#piexif.remove(image_directory)



my_images = os.listdir(image_directory)
for i, image_name in enumerate(my_images):
    if (image_name.split('.')[1] == 'jpg'):
        image = io.imread(image_directory + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE,SIZE))
        dataset.append(np.array(image))




x = np.array(dataset)

#Let us save images to get a feel for the augmented images.
#Create an iterator either by using image dataset in memory (using flow() function)
#or by using image dataset from a directory (using flow_from_directory)
#from directory can beuseful if subdirectories are organized by class
   
# Generating and saving 10 augmented samples  
# using the above defined parameters.  
#Again, flow generates batches of randomly augmented images
  
i = 0
for batch in datagen.flow(x, batch_size=1230,  
                          save_to_dir="D:/cassava/class_4_augmented", 
                          save_prefix='aug', 
                          save_format='jpg'):
    i += 1
    if i > 5:
        break  # otherwise the generator would loop indefinitely  




























