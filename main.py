import pandas as pd

from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

model = load_model("trained_model.h5")


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
