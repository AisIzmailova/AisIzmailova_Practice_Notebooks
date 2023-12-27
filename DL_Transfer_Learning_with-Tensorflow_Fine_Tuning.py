import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import wget
import zipfile
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib
import numpy as np
import random
import datetime

#Create helper function
from Helper_functions import plot_loss_curves, unzip_file, create_tensorboard_callback

#----------------------------Model_0-----------------------------------------------------------------------------------
#We are going to use pretrained models from tf.keras.applications and apply them to our own problem
# https://www.tensorflow.org/api_docs/python/tf/keras/applications

#Get the data
#Walk through 10 classes of food image data
for dirpath, dirnames, filenames in os.walk("10_food_classes_10_percent"):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in {dirpath}.")

#Create train and test directory paths
train_dir="10_food_classes_10_percent/train/"
test_dir="10_food_classes_10_percent/test/"

IMAGE_SIZE=(224,224)
BATCH_SIZE=32

#Get data using image_dataset_from_directory
train_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(
  directory=train_dir, image_size=IMAGE_SIZE, label_mode="categorical", batch_size=BATCH_SIZE
)

test_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(
  directory=test_dir, image_size=IMAGE_SIZE, label_mode="categorical", batch_size=BATCH_SIZE
)

#Checkout class names
print(train_data_10_percent.class_names)

#Check one batch
for images, labels in train_data_10_percent.take(1):
  print(images, labels)

#Create a baseline model using functional api
"""
The sequential API is straight-forward, it runs our layers in sequential order. But the functional API gives us more flexibility with our models.
"""

#1.Create base transfer learning feature extraction model with tf.keras.applications
base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False)
#2.Freeze the base model (so the underlying pre-trained patterns are not updated)
base_model.trainable=False
#3.Create inputs into our model
inputs = tf.keras.layers.Input(shape=(224,224,3), name="input_layer")
#4.If using Resnetv250 model we will need to normalize inputs, no need to normalize for Efficientnet model
#x=tf.keras.layers.experimental.preprocessing.Rescaling(1/255.)(inputs)
#5.Pass the inputs to base model
x=base_model(inputs)
print(f"Shape after passing inputs through base model: {x.shape}")
#6.Average pool the outputs of the base model
x=tf.keras.layers.GlobalAvgPool2D(name="global_avg_pooling_layer")(x)
print(f"Shape after Global Average Pooling 2D: {x.shape}")
#7.Create the output layer
outputs = tf.keras.layers.Dense(10, activation="softmax", name="output_layer")(x)
model_0=tf.keras.Model(inputs,outputs)
#8.Compile the mode
model_0.compile(
  loss=tf.keras.losses.categorical_crossentropy,
  optimizer=tf.keras.optimizers.Adam(),
  metrics=["accuracy"])
#9.Fit the model
history_0 = model_0.fit(train_data_10_percent,
                        epochs=5,
                        steps_per_epoch=len(train_data_10_percent),
                        validation_data=test_data_10_percent,
                        validation_steps=len(test_data_10_percent),
                        callbacks=create_tensorboard_callback(dir_name="transfer_learning",
                                                              experiment_name="fine_tuning_base_model_0"))
print("Base model evaluation ", model_0.evaluate(test_data_10_percent))
plot_loss_curves(history_0)

#Check layers in our base model
# for layer_number, layer in enumerate(base_model.layers):
#     print(layer_number, layer.name)

#Check the summary of EfficientNetV2B0
# print(base_model.summary())

#Get the feature vector
"""
We have a tensor after our model goes through `base_model` of shape (None, 7,7,1280). When it passes through GlobalAveragePooling2D, it turns into (None, 1280).
Let's use a similar shaped tensor (1,4,4,3) and the pass it to GlobalAveragePooling2D.
"""

#Create a random tensor
input_shape=(1,4,4,3)
tf.random.set_seed(42)
input_tensor=tf.random.normal(input_shape)
print(f"Random input tensor: {input_tensor}")
#Pass the random tensor through GlobalAveragePooling2D
global_avg_pooled_tensor = tf.keras.layers.GlobalAvgPool2D()(input_tensor)
print(f"2D global avg pooled random tensor: {global_avg_pooled_tensor}")

#GlobalAvgPool2D() converts a tensor into a feature vector by condensing the middle dimensions
print(f"Shape of input tensor: {input_tensor.shape}")
print(f"Shape of the global avg pooled 2D tensor: {global_avg_pooled_tensor}")

#Let's replicate the GlobalAveragePooling2D layer
tf.reduce_mean(input_tensor, axis=[1,2])

max_pooled_tensor = tf.keras.layers.MaxPool2D()(input_tensor)
print(f"Max Pooled random tensor {max_pooled_tensor}")
print(f"Shape of the max pooled tensor: {max_pooled_tensor.shape}")

"""
One of the reasons feature extraction transfer learning is names how it is is because what often happens is pretrained model outputs
feature vector (a long tensor of numbers which represents the learned representation of the model on a particular sample, in our case, 
this is the output of the GlobalAveragePooling2D() layer which can then be used to extract patterns out of our own specific problem. 
"""
#----------------------------Model_1-----------------------------------------------------------------------------------
#Create a model using feature extraction transfer learning with 1% of the training data with data augmentation
#Get food data but just 1 percent
# data_url = "https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_1_percent.zip"
# food_data = wget.download(data_url)
# print(food_data)
#
# #Unzip the data
# zip_ref = zipfile.ZipFile("10_food_classes_1_percent.zip", "r")
# zip_ref.extractall()
# zip_ref.close()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import random
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Helper_functions import create_tensorboard_callback, plot_loss_curves

IMAGE_SIZE=(224,224)
BATCH_SIZE=32


#Create data directories
train_1_percent_dir = "10_food_classes_1_percent/train/"
test_1_percent_dir = "10_food_classes_1_percent/test/"

from Helper_functions import walk_through_dir
walk_through_dir("10_food_classes_1_percent")

#Load data
train_1_percent_data = tf.keras.preprocessing.image_dataset_from_directory(directory=train_1_percent_dir,
                                                                           label_mode='categorical',
                                                                           image_size=IMAGE_SIZE,
                                                                           batch_size=BATCH_SIZE)
test_1_percent_data = tf.keras.preprocessing.image_dataset_from_directory(directory=test_1_percent_dir,
                                                                          label_mode='categorical',
                                                                          image_size=IMAGE_SIZE,
                                                                          batch_size=BATCH_SIZE)

#Add data augmentation right into the model
"""
To add data augmentation right into the mode, we can use the layers inside:
tf.keras.layers.experimental.preprocessing()
"""


data_augmentation = keras.Sequential([
    preprocessing.RandomFlip("horizontal"),
    preprocessing.RandomRotation(0.2),
    preprocessing.RandomZoom(0.2),
    preprocessing.RandomHeight(0.2),
    preprocessing.RandomWidth(0.2)
    #preprocessing.Rescale(1/255.), #required for Resnet model
], name="data_augmentation")

"""
There are two important points to be aware when using data augmentation as a layer in our model:
- Data augmentation will run on-devide, synchronously with the rest of your layers, and beenfit from GPU acceleration.
- When you export your model using model.save, the preprocessing layers will be saved along with the rest of your model.
  If you later deploy this model, it will automatically standardize images (according to the configuration of your layers).
  This can save you from the effort of having to reimplement that logic server-side.
"""

#Visualize our data augmentation layer
#View a random image
target_class = random.choice(train_1_percent_data.class_names)
target_dir = "10_food_classes_1_percent/train/"+target_class
print(target_dir)
random_image = random.choice(os.listdir(target_dir))
random_image_path = target_dir+"/"+random_image
img = mpimg.imread(random_image_path)
plt.imshow(img)
plt.title(f"Original image from random class {target_class}")
plt.axis(False)
#Let's visualize augmented image
plt.figure()
augmented_img = data_augmentation(img, training=True)
plt.imshow(augmented_img/255.)
plt.title(f"Augmented image from random class {target_class}")
#plt.show()

#Create a model with feature extraction transfer learning with data augmentation on 1 percent data
input_shape = (224,224,3)
#Setup base model and freeze the learning
base_model= tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False)
base_model.trainable=False
#Create the input layer
inputs = layers.Input(shape=input_shape, name="input_layer")
#Add augmentation layer
x = data_augmentation(inputs)
#Give base_model the inputs
x = base_model(x, training=False)
#Pool the output features
x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
#Put an output layer
outputs = layers.Dense(10, activation="softmax", name="output_layer")(x)
#Define the model_1
model_1=keras.Model(inputs, outputs)
#Compile the model
model_1.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer = tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)
#Fit the model
history_1 = model_1.fit(train_1_percent_data,
                        epochs=5,
                        steps_per_epoch=len(train_1_percent_data),
                        validation_data=test_1_percent_data,
                        validation_steps=len(test_1_percent_data),
                        callbacks=[create_tensorboard_callback(dir_name="transfer_learning",
                                                               experiment_name="fine_tuning_model_1")])
#Evaluate the model
print("Model 1 evaluation ", model_1.evaluate(test_1_percent_data))

from Helper_functions import plot_loss_curves
#Plot the loss curves
plot_loss_curves(history_1)

#----------------------------Model_2-----------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import random
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Helper_functions import create_tensorboard_callback, plot_loss_curves

IMAGE_SIZE=(224,224)
BATCH_SIZE=32
#Create train and test directory paths
train_dir="10_food_classes_10_percent/train/"
test_dir="10_food_classes_10_percent/test/"

#Get data using image_dataset_from_directory
train_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(
  directory=train_dir, image_size=IMAGE_SIZE, label_mode="categorical", batch_size=BATCH_SIZE
)

test_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(
  directory=test_dir, image_size=IMAGE_SIZE, label_mode="categorical", batch_size=BATCH_SIZE
)

#Add data augmentation right into the model
data_augmentation = keras.Sequential([
    preprocessing.RandomFlip("horizontal"),
    preprocessing.RandomRotation(0.2),
    preprocessing.RandomZoom(0.2),
    preprocessing.RandomHeight(0.2),
    preprocessing.RandomWidth(0.2),
    #preprocessing.Rescale(1/255.),
], name="data_augmentation")

#Create Model checkpoint callback
"""
The ModelCheckpoint callback intermediately saves our model (the full model or just the weights) during the training.
This is useful so we can come and start where we left off.
"""

#Set path

#Save weights of the model only
checkpoint_path = "ten_percent_model_weights/checkpoint.ckpt"
def create_model_checkpoint_callback(filepath):
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                       save_weights_only=True,
                                       save_best_only=False,
                                       save_freq="epoch",
                                       verbose=1)
    return checkpoint_callback

#Create a model with feature extraction transfer learning with data augmentation on 10 percent data
input_shape = (224,224,3)
#Setup base model and freeze the learning
base_model= tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False)
base_model.trainable=False
#Create the input layer
inputs = layers.Input(shape=input_shape, name="input_layer")
#Add augmentation layer
x = data_augmentation(inputs)
#Give base_model the inputs
x = base_model(x, training=False)
#Pool the output features
x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
#Put an output layer
outputs = layers.Dense(10, activation="softmax", name="output_layer")(x)
#Define the model_1
model_2=keras.Model(inputs, outputs)
#Compile the model
model_2.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer = tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)
#Fit the model
history_2 = model_2.fit(train_data_10_percent,
                        epochs=5,
                        steps_per_epoch=len(train_data_10_percent),
                        validation_data=test_data_10_percent,
                        validation_steps=len(test_data_10_percent),
                        callbacks=[create_tensorboard_callback(dir_name="transfer_learning",
                                                               experiment_name="fine_tuning_model_2"),
                                   create_model_checkpoint_callback(checkpoint_path)])
#Evaluate the model
print("Model 2 evaluation ", model_2.evaluate(test_data_10_percent))
#Plot the loss curves
plot_loss_curves(history_2)
# plt.show()


"""
Loading in checkpointed weights returns a model to a specific checkpoint.
"""
#Load in saved model weights and evaluate the model
model_2.load_weights(checkpoint_path)
loaded_weights_model_results = model_2.evaluate(test_data_10_percent)
print(loaded_weights_model_results)

#----------------------------Model_3-----------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import random
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Helper_functions import create_tensorboard_callback, plot_loss_curves

IMAGE_SIZE=(224,224)
BATCH_SIZE=32

#Create train and test directory paths
train_dir="10_food_classes_10_percent/train/"
test_dir="10_food_classes_10_percent/test/"

#Get data using image_dataset_from_directory
train_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(
  directory=train_dir, image_size=IMAGE_SIZE, label_mode="categorical", batch_size=BATCH_SIZE
)

test_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(
  directory=test_dir, image_size=IMAGE_SIZE, label_mode="categorical", batch_size=BATCH_SIZE
)
"""
Fine-tuning usually works best after training the feature extraction model for a few epochs with large amounts of custom data.
"""
#Check layers in loaded model
print(model_2.layers)
#Check whether layers are trainable
for layer in model_2.layers:
    print(layer, layer.trainable)
#How many trainable variables on our base model
print("Trainable variables: ", len(model_2.layers[2].trainable_variables))

#1.Create base transfer learning feature extraction model with tf.keras.applications
# base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False)
#Make top 10 layers trainable
base_model.trainable=True
for layer in base_model.layers[:-10]:
    layer.trainable=False

#General rule of thumb for fine-tuning is to lower learning_rate by 10x because our model has already learned weights.
#We do not want to update those weights too much, otherwise it may result in overfitting.
#Learning rate dictates how much the model should update its internal patterns/weights.
#Chekout ULM Fit paper

#Recompile the model
model_2.compile(loss= tf.keras.losses.categorical_crossentropy,
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                metrics=["accuracy"]
)

print("Trainable variables: ", len(model_2.layers[2].trainable_variables))
#Fit the model
history_2_trainable = model_2.fit(test_data_10_percent,
                                  epochs=10,
                                  steps_per_epoch=len(train_data_10_percent),
                                  validation_data=test_data_10_percent,
                                  validation_steps=int(0.25*len(test_data_10_percent)),
                                  initial_epoch=history_2.epoch[-1], #start training from previous last epoch
                                  callbacks=[create_tensorboard_callback(dir_name="transfer_learning",
                                                                         experiment_name="fine_tuning_model_2_trainable")])
#Evaluate the model
print("Evaluate Model 2 trainable ", model_2.evaluate(test_data_10_percent))

#Create a fucntion to compare training histories

def compare_histories(original_history,new_history, initial_epochs=5):
    """
    Compares two tensorflow history objects
    """
    #Get original_history metrics
    acc=original_history.history["accuracy"]
    loss=original_history.history["loss"]
    val_acc=original_history.history["val_accuracy"]
    val_loss=original_history.history["val_loss"]
    #Combine original hisotry metrics with new_history metrics
    total_acc=acc+new_history.history["accuracy"]
    total_loss=loss+new_history.history["loss"]
    total_val_acc=val_acc+new_history.history["val_accuracy"]
    total_val_loss=val_loss+new_history.history["val_loss"]
    #Plot accuracy
    plt.figure(figsize=(8,8))
    plt.subplot(2,1,1)
    plt.plot(total_acc, label="Training Accuracy")
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label="Start Fine Tuning")
    plt.legend(loc='lower right')
    plt.title("Training and Validation Accuracy")

    # Plot loss
    #plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label="Training Loss")
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label="Start Fine Tuning")
    plt.legend(loc='upper right')
    plt.title("Training and Validation Loss")

compare_histories(history_2, history_2_trainable, initial_epochs=5)

#----------------------------Model_3-----------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import random
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Helper_functions import create_tensorboard_callback, plot_loss_curves

IMAGE_SIZE=(224,224)
BATCH_SIZE=32

#Create train and test directory paths
train_dir="10_food_classes_all_data/train/"
test_dir="10_food_classes_all_data/test/"

#Get data using image_dataset_from_directory
train_data_all_data = tf.keras.preprocessing.image_dataset_from_directory(
  directory=train_dir, image_size=IMAGE_SIZE, label_mode="categorical", batch_size=BATCH_SIZE
)

test_data_all_data = tf.keras.preprocessing.image_dataset_from_directory(
  directory=test_dir, image_size=IMAGE_SIZE, label_mode="categorical", batch_size=BATCH_SIZE
)
#To train a fine tuning model we need to revert it back to its feature extraction weight
# model_2.load_weights(checkpoint_path)
print("Evaluate model with reverted weights ", model_2.evaluate(test_data_all_data))
model_2.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=["accuracy"]
)

history_4 = model_2.fit(train_data_all_data,
                        epochs=10,
                        steps_per_epoch=len(train_data_all_data),
                        validation_data=test_data_all_data,
                        validation_steps=int(0.25*len(test_data_all_data)),
                        initial_epoch=history_2.epoch[-1], #start training from previous last epoch
                        callbacks=[create_tensorboard_callback(dir_name="transfer_learning",
                                                               experiment_name="fine_tuning_model_4")])
#Evaluate the model
print("Evaluate Model 4 ", model_2.evaluate(test_data_all_data))

#Check trainable layers in model_4
# for layer_number, layer in enumerate(model_2.layers):
#     print(layer_number, layer.name, layer.trainable)
#
# for layer_number in enumerate(model_2.layers[2].layers):
#     print(layer_number, layer.name, layer.trainable)

compare_histories(history_2_trainable, history_4, initial_epochs=5)

