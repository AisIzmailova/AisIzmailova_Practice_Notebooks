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

def plot_loss_curves(history):
  """
  Returns separate loss curves for training any validation metrics.
  """

  loss= history.history["loss"]
  val_loss= history.history["val_loss"]
  accuracy= history.history["accuracy"]
  val_accuracy= history.history["val_accuracy"]
  epochs =range(len(history.history["loss"]))

  #Plot the loss
  plt.figure()
  plt.plot(epochs,loss,  label="training_loss")
  plt.plot(epochs, val_loss, label="val_loss")
  plt.title("Loss")
  plt.xlabel("Epochs")
  plt.legend()
  #Plot the accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label="accuracy")
  plt.plot(epochs, val_accuracy,  label="val_accuracy")
  plt.title("Accuracy")
  plt.xlabel("Epochs")
  plt.legend()

def load_and_prep_image(filename,img_shape=224):
  """
  Reads an image from filename, turns it into a tensor and reshapes it to (img_shape, img_shape, colour_channels).
  """
  #Read the image
  img = tf.io.read_file(filename)
  #Decode the read file into a tensor
  img = tf.image.decode_image(img)
  #Resize the image
  img = tf.image.resize(img, size=[img_shape, img_shape])
  #Rescale the image (get all values btwn 0 and 1)
  img=img/255.
  return img

def pred_and_plot(model, filename, class_names):
  """
  Imports an image located at filename, makes prediction with the model and plots the image with the predicted class the title.
  """

  #Import the target image and preprocess it
  img = load_and_prep_image(filename)

  #Make prediction
  pred = model.predict(tf.expand_dims(img, axis=0))
  print(pred[0])
  print(tf.argmax(pred[0]))
  print(class_names[tf.argmax(pred[0])])
  #Add logic for multiclass classification
  if len(pred[0])>1:
    pred_class = class_names[tf.argmax(pred[0])]
  else:
    #Get the predicted class for binary classification
    pred_class = class_names[int(tf.round(pred))]

  #Plot the image
  plt.figure()
  plt.imshow(img)
  plt.title(f"{pred_class}")
  plt.axis(False)


def view_random_image(target_dir, target_class):

  #Setup the target directory
  target_folder = target_dir+target_class

  #Get a random image path
  random_image = random.sample(os.listdir(target_folder), 1)
  print(random_image)

  #Read in the image and plot it using matplotlib
  img = mpimg.imread(target_folder+"/"+random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off")

  print(f"Image shape:  {img.shape}")
  return img

def create_tensorboard_callback(dir_name, experiment_name):
    log_dir=dir_name+"/"+experiment_name+"/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback

def create_model(model_url, num_classes):
    """
    Takes a Tensorflow Hub url and creates a Keras Sequential model.
    Args:
    model_url (str): a tensorflow hub feature extraction url
    num_classes (int): number of output neurons in the output layer, should be equal to number of target classes, default 10.
    Returns:
    An uncompiled Keras Sequential model with model_url as feature extractor layer and Dense output layer with num_classes output neurons.
    """
    #Download the pretrained model and save it as Keras layer
    feature_extraction_layer = hub.KerasLayer(model_url,
                                       trainable=False,
                                       name="feature_extraction_layer",
                                       input_shape=IMAGE_SHAPE+(3,))
    #Create our model
    model=tf.keras.Sequential([
        feature_extraction_layer,
        layers.Dense(num_classes, activation="softmax", name="output_layer")
    ])
    return model

def unzip_file(filename):
    #Unzip the data
    zip_ref = zipfile.ZipFile(filename, "r")
    zip_ref.extractall()
    zip_ref.close()

def walk_through_dir(directory):
    # Walk through 10 classes of food image data
    for dirpath, dirnames, filenames in os.walk(directory):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in {dirpath}.")

def create_model_checkpoint_callback(filepath):
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                       save_weights_only=True,
                                       save_best_only=False,
                                       save_freq="epoch",
                                       verbose=1)
    return checkpoint_callback


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
    plt.title("Training and Validation Loss")

    # Plot loss
    #plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label="Training Loss")
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label="Start Fine Tuning")
    plt.legend(loc='upper right')
    plt.title("Training and Validation Loss")