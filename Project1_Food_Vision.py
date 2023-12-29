import tensorflow as tf
from tensorflow import keras
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

from Helper_functions import create_tensorboard_callback, create_model_checkpoint_callback

#Download data from tensorflow datasets
import tensorflow_datasets as tfds
#List all available datasets
dataset_list = tfds.list_builders() #get all available datasets
print(dataset_list)
print("food101" in dataset_list)

#Load Food101 dataset
(train_data, test_data), ds_info = tfds.load(name = "food101", split=["train", "validation"], shuffle_files=True, as_supervised=True, with_info=True)

"""
##Exploring the Food101 data from Tensorflow Datasets

To become one with data, we want to find:
* Class names
* Shape of our input data
* Datatype of the input data
* Labels (one-hot encoded or label encoded)
* Do labels match with class names?
"""

#Features of Food101 from tfds
print("Dataset features", ds_info.features)
print("Number of classes", ds_info.features["label"].num_classes)
print("Image shape", ds_info.features["image"].shape)
#Get class names
class_names = ds_info.features["label"].names
print("Class names", class_names[:10])
train_one_sample = train_data.take(1)
print("One sample", train_one_sample)

for image, label in train_one_sample:
  print(f"""
  Image shape: {image.shape}
  Image datatype: {image.dtype}
  Target class (tensor form): {label}
  Class name (str form): {class_names[label.numpy()]}
  """)

#Plot an image
plt.imshow(image)
plt.title(f"{class_names[label.numpy()]}")
plt.axis(False)

"""
##Create preprocessing functions

Neural networks perform best when the data is in a certain way (batched, normalized, etc.). However, not all data (including data from Tensorflow datasets) comes like this. So in order to get it ready for a neural network, we will often have to write preprocessing functions and map it to our data.

What we know about our data:
* In `uint8` datatype
* Comprised of all different sizes tensors
* Not scaled (the pixel values are btwn 0 and 255)

Principles for preprocessing functions:
* Data in `float32` datatype (or mixed precision `float16` and `float32`).
* For batches, Tensorflow likes all of the tensors to be of the same size.
* Scaled (values btwn 0 and 1), also called normalized tensors.
"""

def preprocess_image(image, label, img_shape=224):
    """
    Converts image datatype from 'uint8' to 'float32' and reshapes image to [img_shape,img_shape, colour_channels]
    """
    image=tf.image.resize(image,[img_shape, img_shape])
    return tf.cast(image, tf.float32), label #returns an image,label tuple

preprocessed_img = preprocess_image(image, label)[0]
print(f"Preprocessed image shape {preprocessed_img.shape}, Preprocessed datatype {preprocessed_img.dtype}")

"""
##Batching & preparing images from dataset
Read: https://www.tensorflow.org/guide/data_performance
We are going to map preprocessing function across our training dataset, then shuffle a number of 
elements, and then batch them together, and finally prepare new batches (prefetch) whilst the model 
will be looking through the current batch.
"""

#Map preprocessing function to training data (and parallelize)
train_data = train_data.map(map_func=preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
#Shuffle train data and turn it into batches and prefetch it
train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

test_data = test_data.map(map_func=preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
test_data = test_data.batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE) #.cache() if you have enough memory

##Create modelling callbacks
checkpoint_path="checkpoints/model_checkpoint.ckpt"

"""
#Set up Mixed Precision
Read: https://www.tensorflow.org/guide/mixed_precision
Mixed precision utilizes a combination of float32 and float16 data types to speed up model 
performance. Layers use `float16` computations and `float32` variables. Computations are done 
in `float16` for performance, but variables must be kept in `float32` for numeric stability.
"""

# #Turn on mixed precision training
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

##Build feature extraction model
# from tensorflow.keras import layers
# from tensorflow.keras.layers.experimental import preprocessing
#from tensorflow import keras
#Download base model from tf.keras.applications
base_model=tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False)
base_model.trainable=False
input_shape=(224,224,3)
inputs=tf.keras.layers.Input(shape=input_shape)
#x=preprocessing.Rescaling(1./255)(x)
x=base_model(inputs,training=False)
x=tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pooling")(x)
x=tf.keras.layers.Dense(len(class_names))(x)
outputs=tf.keras.layers.Activation('softmax', dtype='float32', name="output_layer")(x)
model=tf.keras.Model(inputs, outputs)

#If we have labels provided as integers, use Sparse Categorical Cross Entropy loss,
# otherwise if labels are provided using one-hot encoded representation, use Categorical
# Cross Entropy loss.
#Compile the model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])
#Check datatypes for each of the layer
for layer in model.layers:
  print(f"Layer name: {layer.name}, Layer trainable? {layer.trainable}, Layer datatype: {layer.dtype}, Layer datatype policy:  {layer.dtype_policy}")

#Fit the model
history=model.fit(train_data,
                  epochs=3,
                  steps_per_epoch=len(train_data),
                  validation_data=test_data,
                  validation_steps=int(0.15*len(test_data)),
                  callbacks=[create_tensorboard_callback("experiment_logs", "101_food_classes"),create_model_checkpoint_callback(checkpoint_path)])
#Evaluate the model
model_results=model.evaluate(test_data)
print(model_results)

#Fine tune the model

#Make last 5 layers of the base model trainable
base_model.trainable=True
for layer in base_model.layers[:-5]:
    layer.trainable=False

#Compile the model
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=["accuracy"]
)
#Fit the model
history_fine_tuned=model.fit(train_data,
                  epochs=6,
                  validation_data=test_data,
                  validation_steps=int(0.15*len(test_data)),
                  initial_epoch=history.epoch[-1],
                  callbacks=create_tensorboard_callback("experiment_logs",
                                                        experiment_name="101_food_classes"))

fine_tuned_model_results=model.evaluate(test_data)
print("Model Evaluation ", model_results)
print("Fine Tuned Model Evaluation ", fine_tuned_model_results)

from Helper_functions import compare_histories
compare_histories(history, history_fine_tuned, initial_epochs=3)

#Load in saved model weights and evaluate the model
model.load_weights(checkpoint_path)
loaded_weights_model_results = model.evaluate(test_data)
print(loaded_weights_model_results)

#Make last 10 layers of the base model trainable
base_model.trainable=True
for layer in base_model.layers[:-10]:
    layer.trainable=False

#Compile the model
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=["accuracy"]
)
#Fit the model
history_fine_tuned_10_layers=model.fit(train_data,
                  epochs=6,
                  validation_data=test_data,
                  validation_steps=int(0.15*len(test_data)),
                  initial_epoch=history.epoch[-1],
                  callbacks=create_tensorboard_callback("experiment_logs",
                                                        experiment_name="101_food_classes"))

fine_tuned_model_10_layers_results=model.evaluate(test_data)
compare_histories(history, history_fine_tuned_10_layers, initial_epochs=3)
print("Model Evaluation ", model_results)
print("Fine Tuned Model Evaluation ", fine_tuned_model_results)
print("Fine Tuned Model Evaluation 10 layers ", fine_tuned_model_10_layers_results)

#Making predictions with our model
preds_probs= model.predict(test_data, verbose=1)
pred_classes=preds_probs.argmax(axis=1)
print(pred_classes[:10])


y_labels=[]
for images, labels in test_data.unbatch():
    y_labels.append(labels.numpy().argmax())
print(y_labels[:10])


from sklearn.metrics import accuracy_score
sklearn_accuracy = accuracy_score(y_labels, pred_classes)
print("Sklearn accuracy ", sklearn_accuracy)

from sklearn.metrics import classification_report
print(classification_report(y_labels, pred_classes))


#2.Create DataFrame
pred_df=pd.DataFrame({"y_true": y_labels,
                      "y_pred": pred_classes,
                      "pred_conf": preds_probs.max(axis=1),#get the max prediction probability value
                      "y_true_classname": [class_names[i] for i in y_labels],
                      "y_pred_classname": [class_names[i] for i in pred_classes]
                      })

#3.Find all the wrong predictions
pred_df["pred_correct"]=pred_df["y_true"]==pred_df["y_pred"]
print(pred_df.head())

#4.Sort the dataframe
top_100_wrong=pred_df[pred_df["pred_correct"]==False].sort_values("pred_conf", ascending=False)[:100]
print(top_100_wrong[["y_true_classname","y_pred_classname","pred_conf"]].head(20))


def view_and_predict_one_sample(model,dataset):

    #Pick one random sample
    train_one_sample=dataset.unbatch().take(1)
    for image, label in train_one_sample:
        print(f"""
        Image shape: {image.shape}
        Image datatype: {image.dtype}
        Target class (tensor form): {label}
        Class name (str form): {class_names[label.numpy()]}
        """)

        #Plot an image
        plt.imshow(image/255.)
        plt.title(f"{class_names[label.numpy()]}")
        plt.axis(False)

        #Make a prediction
        pred_prob=model.predict(train_one_sample)
        pred_class=class_names[pred_prob.argmax()]
        print(f"Predicted food class {pred_class}, actual food class {class_names[label.numpy()]}")

view_and_predict_one_sample(model, train_data)
plt.show()
