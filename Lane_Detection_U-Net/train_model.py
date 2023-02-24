
import numpy as np
import os 
import cv2
import numpy as np
import os 
import matplotlib.pyplot as plt
import tensorflow as tf
from focal_loss import BinaryFocalLoss
from keras import backend as K

os.chdir('/home/eeavserver/parth_stuff/newdataset/')
dataset_dir = os.getcwd()

seed = 24 
batch_size = 64 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

img_data_gen_args = dict(rescale = 1/255., brightness_range=[0.2,0.8],
                         horizontal_flip=True)

mask_data_gen_args = dict(rescale = 1/255., brightness_range=[0.2,0.8],
                         horizontal_flip=True)

image_data_generator = ImageDataGenerator(**img_data_gen_args)
image_generator = image_data_generator.flow_from_directory('dataset/train_images/', 
                                                            seed=seed,
                                                            target_size = (256,256),
                                                            batch_size=batch_size,
                                                            class_mode = None)
mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
mask_generator = mask_data_generator.flow_from_directory('dataset/train_masks/',
                                                         target_size = (256,256),
                                                         seed=seed,
                                                         batch_size=batch_size,
                                                         color_mode='grayscale',
                                                         class_mode = None)

val_img_generator = image_data_generator.flow_from_directory(('dataset/val_images/'),
                                                            target_size = (256,256), 
                                                            seed=seed,
                                                            batch_size=batch_size,
                                                            class_mode = None)


val_mask_generator = mask_data_generator.flow_from_directory(('dataset/val_masks/'),
                                                         target_size = (256,256),
                                                         seed=seed,
                                                         batch_size=batch_size,
                                                         color_mode='grayscale',
                                                         class_mode = None)

train_img_generator = zip(image_generator,mask_generator)
val_img_generator = zip(val_img_generator, val_mask_generator)

## Sanity Check on some images and masks

#x = image_generator.next()
#y = mask_generator.next()

#for i in range(0,1):
#    image = x[i]
#    mask = y[i]
#    plt.subplot(1,2,1)
#    plt.imshow(image[:,:,:])
#    plt.subplot(1,2,2)
#    plt.imshow(mask[:,:,0])
#    plt.show()


def dice_metric(y_pred,y_true):
    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
    union = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))

    return 2*intersection/union


IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS = 256,256,3  #should add argparse here 

inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

c1 = tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same')(inputs)
c1 = tf.keras.layers.Dropout(0.2)(c1)
c1 = tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

c2 = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same')(p1)
c2 = tf.keras.layers.Dropout(0.2)(c2)
c2 = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

c3 = tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

c4 = tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same')(c4)
p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)

c5 = tf.keras.layers.Conv2D(512, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same')(p4)
c5 = tf.keras.layers.Dropout(0.2)(c5)
c5 = tf.keras.layers.Conv2D(512, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same')(c5)

## Expansive Path

u6 = tf.keras.layers.Conv2DTranspose(256, (2,2), strides=(2,2), padding ='same')(c5)
u6 = tf.keras.layers.concatenate([u6,c4])
c6 = tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same')(u6)
c6 = tf.keras.layers.Dropout(0.3)(c6)
c6 = tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding ='same')(c6)
u7 = tf.keras.layers.concatenate([u7,c3])
c7 = tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same')(u7)
c7 = tf.keras.layers.Dropout(0.3)(c7) 
c7 = tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding ='same')(c7)
u8 = tf.keras.layers.concatenate([u8,c2])
c8 = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same')(u8)
c8 = tf.keras.layers.Dropout(0.3)(c8) 
c8 = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding ='same')(c8)
u9 = tf.keras.layers.concatenate([u9,c1], axis =3)
c9 = tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same')(u9)
c9 = tf.keras.layers.Dropout(0.3)(c9) 
c9 = tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same')(c9)

outputs = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid', name='lane_lines')(c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics =[dice_metric, 'acc'])

model.summary()

num_train_imgs = len(os.listdir(dataset_dir + '/dataset/train_images/train/'))

steps_per_epoch = num_train_imgs // batch_size

history = model.fit(train_img_generator, 
                              validation_data = val_img_generator,
                              steps_per_epoch=steps_per_epoch,
                              validation_steps=steps_per_epoch, epochs=60)

model.save(dataset_dir + "/model_256x256v9.h5")

