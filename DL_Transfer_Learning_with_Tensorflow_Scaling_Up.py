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
from Helper_functions import plot_loss_curves, unzip_file, create_tensorboard_callback, download_and_unzip_data, walk_through_dir, view_random_image

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
checkpoint_path = "checkpoints/101_classes_10_percent_checkpoint.ckpt"
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

# from Helper_functions import plot_loss_curves
# plot_loss_curves(history)

base_model.trainable=True
for layer in base_model.layers[:-5]:
    layer.trainable=False

#Compile the model
model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=["accuracy"]
)

#Fit the model
history_fine_tuned=model.fit(train_data_101_class_10_percent,
                  epochs=10,
                  validation_data=test_data_101_class_10_percent,
                  validation_steps=int(0.15*len(test_data_101_class_10_percent)),
                  initial_epoch=history.epoch[-1],
                  callbacks=create_tensorboard_callback("transfer_learning",
                                                        experiment_name="101_classes_10_percent"))

results_fine_tuned_model=model.evaluate(test_data_101_class_10_percent)

print("Fine Tuned Model Evaluation ", results_fine_tuned_model)
print("Model Evaluation ", results_model)

from Helper_functions import compare_histories
compare_histories(history, history_fine_tuned, initial_epochs=5)

model.save("fine_tuned_model")

loaded_model=tf.keras.models.load_model("fine_tuned_model")
results_loaded_model=loaded_model.evaluate(test_data_101_class_10_percent)
print(results_loaded_model)

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
from Helper_functions import download_and_unzip_data

data_url="https://storage.googleapis.com/ztm_tf_course/food_vision/06_101_food_class_10_percent_saved_big_dog_model.zip"
download_and_unzip_data(data_url=data_url, filename="06_101_food_class_10_percent_saved_big_dog_model.zip")

#Download pretrained model
model_retrieved=tf.keras.models.load_model("06_101_food_class_10_percent_saved_big_dog_model")

results_retrieved_model=model_retrieved.evaluate(test_data_101_class_10_percent)
print(results_retrieved_model)

#Making predictions with our model
preds_probs= model_retrieved.predict(test_data_101_class_10_percent, verbose=1)
print(f"The shape of preds_probs is: {preds_probs.shape} and the length is {len(preds_probs)}")
class_names=train_data_101_class_10_percent.class_names
print(class_names)
print(tf.argmax(preds_probs[0]))
class_names[tf.argmax(preds_probs[0])]

pred_classes=preds_probs.argmax(axis=1)
print(pred_classes[:10])

"""
Now we've got a predictions array of all of our model's predictions, to evaluate them, we need to compare them to the original dataset labels.
"""
#To get our test labels we need to unravel our test_data batchdataset
y_labels=[]
for images, labels in test_data_101_class_10_percent.unbatch():
    y_labels.append(labels.numpy().argmax())
print(y_labels[:10])

from sklearn.metrics import accuracy_score
sklearn_accuracy = accuracy_score(y_labels, pred_classes)
print("Sklearn accuracy ", sklearn_accuracy)

import numpy as np
print(np.isclose(results_retrieved_model[1], sklearn_accuracy))

#Draw confusion matrix
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False):
    import itertools
    from sklearn.metrics import confusion_matrix
    """Makes a labelled confusion matrix comparing predictions and ground truth labels.
  
    If classes is passed, confusion matrix will be labelled, if not, integer class values
    will be used.
  
    Args:
      y_true: Array of truth labels (must be same shape as y_pred).
      y_pred: Array of predicted labels (must be same shape as y_true).
      classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
      figsize: Size of output figure (default=(10, 10)).
      text_size: Size of output figure text (default=15).
      norm: normalize values or not (default=False).
      savefig: save confusion matrix to file (default=False).
  
    Returns:
      A labelled confusion matrix plot comparing y_true and y_pred.
  
    Example usage:
      make_confusion_matrix(y_true=test_labels, # ground truth test labels
                            y_pred=y_preds, # predicted labels
                            classes=class_names, # array of class label names
                            figsize=(15, 15),
                            text_size=10)
    """
    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # normalize it
    n_classes = cm.shape[0]  # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)  # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)

    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes),  # create enough axis slots for each class
           yticks=np.arange(n_classes),
           xticklabels=labels,  # axes will labeled with class names (if they exist) or ints
           yticklabels=labels)

    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)

    # Save the figure to the current working directory
    if savefig:
        fig.savefig("confusion_matrix.png")

class_names=train_data_101_class_10_percent.class_names
make_confusion_matrix(y_labels, pred_classes, class_names, figsize=(10,10), text_size=10, savefig=True)

#Evaluate the model using other metrics
"""
Scikit-learn has a helpful function for acquiring many different classification metrics per class called classification report.
Classification report gives a great class-by-class evaluation of our model's predictions. 
"""
from sklearn.metrics import classification_report
print(classification_report(y_labels, pred_classes))

#Visualize the classification report
classification_report_dict=classification_report(y_labels, pred_classes, output_dict=True)
#Plot all classes' F1-score

#Create empty dictionary
class_f1_scores={}
#Loop through classification report dictionary items
for k,v in classification_report_dict.items():
    if k=="accuracy":
        break
    else:
        #Add class names and f1-scores to out dictionary
        class_f1_scores[class_names[int(k)]] = v["f1-score"]
print(class_f1_scores)

#Turn f1-scores into dataframe
import pandas as pd
f1_scores= pd.DataFrame({"class_names": list(class_f1_scores.keys()),
                         "f1_score": list(class_f1_scores.values())}).sort_values("f1_score", ascending=False)

f1_scores.head()
#Plot bar chart
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12,25))
scores = ax.barh(range(len(f1_scores)), f1_scores["f1_score"].values)
ax.set_yticks(range(len(f1_scores)))
ax.set_yticklabels(f1_scores["class_names"])
ax.set_xlabel("F1-score")
ax.set_title("F1-score for 101 food classes")
ax.invert_yaxis() #reverse the order of our plot
# plt.show()

"""
Visualizing predictions on test images

To visualize our model's predictions on our own images, we will need to load and preprocess images, 
specifically will need to:
1. Read a target image using tf.io.read_file()
2. Turn image into a Tensor using tf.io.decode_image()
3. Resize the image tensor to be the same size as the images our model has been trained on
4. Scale the image to get all of the pixel values btwn 0 and 1
"""

from Helper_functions import load_and_prep_image, pred_and_plot
import random
class_names=train_data_101_class_10_percent.class_names
target_dir="101_food_classes_10_percent/train/"
target_class = random.choice(os.listdir(target_dir))
print(target_class)
target_image = random.choice(os.listdir(target_dir + target_class))
img_path = target_dir + target_class + "/" + target_image
# img=load_and_prep_image(img_path)
# pred_prob=loaded_model.predict(tf.expand_dims(img, axis=0))
# print(pred_prob[0])
# print(pred_prob.argmax())
# pred_class=class_names[pred_prob.argmax()]
# print(f"Predicted food class {pred_class}, actual food class {target_class}")
pred_and_plot(loaded_model, img_path, class_names)

"""
Finding the most wrong predictions

To find out where our model us most wrong, let's write some code to find out the following:
1. Get all of the image file paths in the test dataset using list_files() method
2. Create pandas dataframe of the image filepaths, ground truth labels, predicted classes, max prediction probabilities.
3. Use our dataframe to find all the wrong predictions.
4. Sort the dataframe based on wrong predictions (have highest prediction probabilities at the top).
5. Visualize the images with highest prediction probabilities but have the wrong prediction.
"""

#1.Get all the image filepaths
filepaths=[]
for filepath in test_data_101_class_10_percent.list_files("101_food_classes_10_percent/test/*/*.jpg", shuffle=False):
    filepaths.append(filepath.numpy())
print(filepaths[:10])

#2.Create DataFrame
pred_df=pd.DataFrame({"img_path": filepaths,
                      "y_true": y_labels,
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
print(top_100_wrong.head(20))

#5.Visualize
images_to_view=9
start_index=0
plt.figure(figsize=(15,10))
for i, row in enumerate(top_100_wrong[start_index:start_index+images_to_view].itertuples()):
    plt.subplot(3,3,i+1)
    img=load_and_prep_image(row[1])
    _,_,_,_,pred_conf, y_true_classname, y_pred_classname, _= row# only interested in few values from the row
    plt.imshow(img)
    plt.title(f"Actual {y_true_classname}, prediction {y_pred_classname}  \n probability {pred_conf}")
    plt.axis(False)
plt.show()

