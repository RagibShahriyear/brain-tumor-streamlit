import os
import pandas as pd
import tensorflow as tf
from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

model = load_model("trained_model.h5")


# new_test_path = Path("test/")

# new_test_jpg = list(new_test_path.glob(r"*.jpg"))

# print(new_test_jpg)


# def get_jpg_label(x):
#     return os.path.split(os.path.split(x)[0])[1]


# ragib_test_JPG_labels = list(map(get_jpg_label, new_test_jpg))
# print(ragib_test_JPG_labels)

# Ragib_Test_JPG_Path_Series = pd.Series(new_test_jpg, name="JPG").astype(str)
# Ragib_Test_JPG_Labels_Series = pd.Series(ragib_test_JPG_labels, name="TUMOR_CATEGORY")

# Test_Ragib = pd.concat(
#     [Ragib_Test_JPG_Path_Series, Ragib_Test_JPG_Labels_Series], axis=1
# )

# print(Test_Ragib)

# figure = plt.figure(figsize=(5, 5))
# plt.imshow(plt.imread(Test_Ragib["JPG"][3]))

# Generator_Basic = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.1)

# Test_Set_2 = Generator_Basic.flow_from_dataframe(
#     dataframe=Test_Ragib,
#     x_col="JPG",
#     y_col="TUMOR_CATEGORY",
#     color_mode="grayscale",
#     class_mode="categorical",
#     target_size=(200, 200),
# )

# # for data_batch, label_batch in Test_Set_2:
# #     print("data shape: ", data_batch.shape)
# #     print("label shape: ", label_batch.shape)
# #     break

# # print(Test_Set_2.class_indices)
# # print(Test_Set_2.classes)
# # print(Test_Set_2.image_shape)

# Ragib_Data_Prediction = pd.DataFrame({"JPG": Ragib_Test_JPG_Path_Series})

# # print(Ragib_Data_Prediction)

# Main_Test_Generator = ImageDataGenerator(rescale=1.0 / 255)

# Ragibs_Test_Set = Main_Test_Generator.flow_from_dataframe(
#     dataframe=Ragib_Data_Prediction,
#     x_col="JPG",
#     y_col=None,
#     color_mode="grayscale",
#     class_mode=None,
#     target_size=(200, 200),
# )

# Ragib_Test_Prediction = model.predict(Ragibs_Test_Set)

# Ragib_Test_Prediction = Ragib_Test_Prediction.argmax(axis=-1)
# print(Ragib_Test_Prediction)

# Last_Prediction = []
# [
#     Last_Prediction.append("NO") if i == 1 else Last_Prediction.append("TUMOR")
#     for i in Ragib_Test_Prediction
# ]
# print(Last_Prediction)


def RetPreds(model):
    new_test_path = Path("./uploaded_test")
    new_test_jpg = list(new_test_path.glob(r"*.jpg"))

    Ragib_Test_JPG_Path_Series = pd.Series(new_test_jpg, name="JPG").astype(str)

    Ragib_Data_Prediction = pd.DataFrame({"JPG": Ragib_Test_JPG_Path_Series})
    Main_Test_Generator = ImageDataGenerator(rescale=1.0 / 255)
    Ragibs_Test_Set = Main_Test_Generator.flow_from_dataframe(
        dataframe=Ragib_Data_Prediction,
        x_col="JPG",
        y_col=None,
        color_mode="grayscale",
        class_mode=None,
        target_size=(200, 200),
    )

    Ragib_Test_Prediction = model.predict(Ragibs_Test_Set)

    Ragib_Test_Prediction = Ragib_Test_Prediction.argmax(axis=-1)
    print(Ragib_Test_Prediction)

    Last_Prediction = []
    [
        Last_Prediction.append("Tumor") if i == 1 else Last_Prediction.append("Normal")
        for i in Ragib_Test_Prediction
    ]
    return Last_Prediction
