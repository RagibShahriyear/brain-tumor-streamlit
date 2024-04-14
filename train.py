import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from warnings import filterwarnings
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from tensorflow.keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPool2D,
    BatchNormalization,
    MaxPooling2D,
)
from keras import models

import tensorflow as tf
import os
import os.path
from pathlib import Path
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import regularizers
from keras.optimizers import RMSprop, Adam

import glob
from PIL import Image

# Ignoring unnecessary warnings
filterwarnings("ignore", category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=UserWarning)


# Path process
## Train

No_Data_Path = Path("./no/")
Yes_Data_Path = Path("./yes/")

No_JPG_Path = list(No_Data_Path.glob(r"*.jpg"))
Yes_JPG_Path = list(Yes_Data_Path.glob(r"*.jpg"))

print(No_JPG_Path[0:5])
print("_____" * 20)
print(Yes_JPG_Path[0:5])

Yes_No_List = []

for No_JPG in No_JPG_Path:
    Yes_No_List.append(No_JPG)

for Yes_JPG in Yes_JPG_Path:
    Yes_No_List.append(Yes_JPG)

print(Yes_No_List[0])

JPG_Labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], Yes_No_List))
len(JPG_Labels)

print(JPG_Labels[2995:2999])

print("NO COUNTING: ", JPG_Labels.count("no"))
print("YES_COUNTING: ", JPG_Labels.count("yes"))

JPG_Path_Series = pd.Series(Yes_No_List, name="JPG").astype(str)
JPG_Category_Series = pd.Series(JPG_Labels, name="TUMOR_CATEGORY")

Main_Train_Data = pd.concat([JPG_Path_Series, JPG_Category_Series], axis=1)

print(Main_Train_Data.head(-1))

# Test

Prediction_Path = Path("./pred")
Test_JPG_Path = list(Prediction_Path.glob(r"*.jpg"))
print(Test_JPG_Path[0:5])

Test_JPG_Labels = list(
    map(lambda x: os.path.split(os.path.split(x)[0])[1], Test_JPG_Path)
)
print(Test_JPG_Labels[0:5])

Test_JPG_Path_Series = pd.Series(Test_JPG_Path, name="JPG").astype(str)
Test_JPG_Labels_Series = pd.Series(Test_JPG_Labels, name="TUMOR_CATEGORY")

Test_Data = pd.concat([Test_JPG_Path_Series, Test_JPG_Labels_Series], axis=1)

print(Test_Data.head())

Main_Train_Data = Main_Train_Data.sample(frac=1).reset_index(drop=True)
print(Main_Train_Data.head(-1))

image_num = 0
figure = plt.figure(figsize=(5, 5))
plt.imshow(plt.imread(Main_Train_Data["JPG"][image_num]))
plt.title(Main_Train_Data["TUMOR_CATEGORY"][image_num])

fig, axes = plt.subplots(
    nrows=5, ncols=5, figsize=(10, 10), subplot_kw={"xticks": [], "yticks": []}
)


for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(Main_Train_Data["JPG"][i]))
    ax.set_title(Main_Train_Data["TUMOR_CATEGORY"][i])

plt.tight_layout()
plt.show()

# Model Training Data

train_data, test_data = train_test_split(
    Main_Train_Data, train_size=0.9, random_state=42
)


print(train_data.shape, test_data.shape)

# Image Data Generator Without Diversification

Generator_Basic = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.1)

Train_Set = Generator_Basic.flow_from_dataframe(
    dataframe=train_data,
    x_col="JPG",
    y_col="TUMOR_CATEGORY",
    color_mode="grayscale",
    class_mode="categorical",
    subset="training",
    batch_size=20,
    target_size=(200, 200),
)

Validation_Set = Generator_Basic.flow_from_dataframe(
    dataframe=train_data,
    x_col="JPG",
    y_col="TUMOR_CATEGORY",
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation",
    batch_size=20,
    target_size=(200, 200),
)

Test_Set = Generator_Basic.flow_from_dataframe(
    dataframe=test_data,
    x_col="JPG",
    y_col="TUMOR_CATEGORY",
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=20,
    target_size=(200, 200),
)

# checking
for data_batch, label_batch in Train_Set:
    print("DATA SHAPE :", data_batch.shape)
    print("LABEL SHAPE :", label_batch.shape)
    break


for data_batch, label_batch in Validation_Set:
    print("DATA SHAPE :", data_batch.shape)
    print("LABEL SHAPE :", label_batch.shape)
    break

for data_batch, label_batch in Test_Set:
    print("DATA SHAPE :", data_batch.shape)
    print("LABEL SHAPE :", label_batch.shape)
    break

print(Train_Set.class_indices)
print(Train_Set.classes[0:5])
print(Train_Set.image_shape)


print(Validation_Set.class_indices)
print(Validation_Set.classes[0:5])
print(Validation_Set.image_shape)

print(Test_Set.class_indices)
print(Test_Set.classes[0:5])
print(Test_Set.image_shape)


# CNN Model for non-diversification


Model = Sequential()
Model.add(Conv2D(32, (5, 5), activation="relu", input_shape=(200, 200, 1)))
Model.add(MaxPool2D((2, 2)))
Model.add(Dropout(0.2))
#
Model.add(Conv2D(64, (3, 3), activation="relu"))
Model.add(MaxPool2D((2, 2)))
Model.add(Dropout(0.2))
#
Model.add(Conv2D(128, (3, 3), activation="relu"))
Model.add(MaxPool2D((2, 2)))
Model.add(Dropout(0.2))
#
Model.add(Conv2D(256, (3, 3), activation="relu"))
Model.add(MaxPool2D((2, 2)))
Model.add(Dropout(0.2))
#
Model.add(Flatten())
Model.add(Dropout(0.5))
Model.add(Dense(512, activation="relu"))
Model.add(Dense(2, activation="softmax"))

Model.summary()

Model.compile(
    optimizer=RMSprop(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Model Fitting
ANN_Model = Model.fit(
    Train_Set, validation_data=Validation_Set, epochs=30, steps_per_epoch=120
)


Model.summary()

# Checking model with Graphs
HistoryDict = ANN_Model.history

val_losses = HistoryDict["val_loss"]
val_acc = HistoryDict["val_accuracy"]
acc = HistoryDict["accuracy"]
losses = HistoryDict["loss"]
epochs = range(1, len(val_losses) + 1)

plt.plot(ANN_Model.history["accuracy"])
plt.plot(ANN_Model.history["val_accuracy"])
plt.ylabel("ACCURACY")
plt.legend()
plt.show()

plt.plot(epochs, losses, "k-", label="LOSS")
plt.plot(epochs, val_losses, "ro", label="LOSS VALIDATION")
plt.title("LOSS & LOSS VAL")
plt.xlabel("EPOH")
plt.ylabel("LOSS & LOSS VAL")
plt.legend()
plt.show()

plt.plot(epochs, acc, "k-", label="ACCURACY")
plt.plot(epochs, val_acc, "ro", label="VALIDATION ACCURACY")
plt.title("TRAINING AND VALIDATION ACCURACY")
plt.xlabel("EPOH")
plt.ylabel("TRAINING AND VALIDATIN ACCURACY")
plt.legend()
plt.show()

Dict_Summary = pd.DataFrame(ANN_Model.history)
Dict_Summary.plot()

# Prediction Score on Divided Data
Model_Results = Model.evaluate(Test_Set, verbose=False)
print("LOSS: " + "%.4f" % Model_Results[0])
print("ACCURACY: " + "%.4f" % Model_Results[1])

# Prediction Process
Main_Data_Prediction = pd.DataFrame({"JPG": Test_JPG_Path_Series})
print(Main_Data_Prediction.head())

Main_Test_Generator = ImageDataGenerator(rescale=1.0 / 255)

Main_Test_Set = Main_Test_Generator.flow_from_dataframe(
    dataframe=Main_Data_Prediction,
    x_col="JPG",
    y_col=None,
    color_mode="grayscale",
    class_mode=None,
    bath_size=20,
    target_size=(200, 200),
)

Model_Test_Prediction = Model.predict(Main_Test_Set)

Model_Test_Prediction = Model_Test_Prediction.argmax(axis=-1)

Last_Prediction = []
[
    Last_Prediction.append("NO") if i == 1 else Last_Prediction.append("TUMOR")
    for i in Model_Test_Prediction
]
print(Last_Prediction)

fig, axes = plt.subplots(
    nrows=5, ncols=5, figsize=(20, 20), subplot_kw={"xticks": [], "yticks": []}
)


for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(Main_Data_Prediction["JPG"].loc[i]))
    ax.set_title(f"PREDICTION: {Last_Prediction[i]}")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(
    nrows=5, ncols=5, figsize=(20, 20), subplot_kw={"xticks": [], "yticks": []}
)


for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(Main_Data_Prediction["JPG"].loc[i]))
    ax.set_title(f"PREDICTION: {Last_Prediction[i]}")
plt.tight_layout()
plt.show()

Model.save("trained_model.h5")
