import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import wget
import zipfile
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import pathlib
import numpy as np
import random
import datetime

#Create helper function
from Helper_functions import plot_loss_curves, unzip_file, create_tensorboard_callback, download_and_unzip_data, walk_through_dir,\
    view_random_image

#Get data
# data_url="https://storage.googleapis.com/ztm_tf_course/food_vision/101_food_classes_10_percent.zip"
# filename="101_food_classes_10_percent.zip"
# download_and_unzip_data(data_url,filename)
walk_through_dir("101_food_classes_10_percent")

#View random image
import random
target_dir ="101_food_classes_10_percent/train/"
target_class=random.choice(os.listdir(target_dir))
print(target_class)
target_image=random.choice(os.listdir(target_dir+target_class))
print(target_image)
img_path=target_dir+target_class+"/"+target_image
img=mpimg.imread(img_path)
plt.imshow(img)
plt.title(f"{target_class}")
# plt.show()

#Define directories
train_dir="101_food_classes_10_percent/train/"
test_dir="101_food_classes_10_percent/test/"

IMAGE_SIZE=(224,224)
BATCH_SIZE=32
#Define train and test data
train_data_101_class_10_percent = tf.keras.preprocessing.image_dataset_from_directory(directory=train_dir,
                                                                                      label_mode="categorical",
                                                                                      image_size=IMAGE_SIZE,
                                                                                      batch_size=BATCH_SIZE)
test_data_101_class_10_percent = tf.keras.preprocessing.image_dataset_from_directory(directory=test_dir,
                                                                                     label_mode="categorical",
                                                                                     image_size=IMAGE_SIZE,
                                                                                     batch_size=BATCH_SIZE,
                                                                                     shuffle=False)
#Train the model
"""
1. Create a model checkpoint callback
2. Create data augmentation layer 
3. Build headless (no top layer) functional efficientnet model
4. Compile the model
5. Feature extract for 5 full passes (5 epochs on the train dataset and validate on 15% of test data)
"""
#1.Create model checkpoint callback
checkpoint_path = "101_classes_10_percent_checkpoint.ckpt"
from Helper_functions import create_model_checkpoint_callback

#2.Create data augmentation layer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

data_augmentation=keras.Sequential([
    preprocessing.RandomFlip("horizontal"),
    preprocessing.RandomRotation(0.2),
    preprocessing.RandomHeight(0.2),
    preprocessing.RandomWidth(0.2),
    preprocessing.RandomZoom(0.2)
], name="data_augmentation")

#3.Build model from functional api
input_shape=IMAGE_SIZE+(3,)
base_model=tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False)
base_model.trainable=False
inputs=layers.Input(shape=input_shape)
x=data_augmentation(inputs)
x=base_model(x, training=False)
x=layers.GlobalAveragePooling2D(name="global_avg_pool_layer")(x)
outputs=layers.Dense(len(train_data_101_class_10_percent.class_names), activation="softmax", name="output_layer")(x)
model=tf.keras.Model(inputs,outputs)

#4.Compile the model
model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)

#5.Fit the model
history=model.fit(train_data_101_class_10_percent,
                  epochs=5,
                  validation_data=test_data_101_class_10_percent,
                  validation_steps=int(0.15*len(test_data_101_class_10_percent)),
                  callbacks=create_model_checkpoint_callback(checkpoint_path))

#6.Evaluate the model
results_model=model.evaluate(test_data_101_class_10_percent)
print(results_model)