import streamlit as st
import os
from keras.models import load_model
from main import RetPreds

# File Processing Pkgs
from PIL import Image

model = load_model("trained_model.h5")


# Load Images
@st.cache_data
def load_image(image_file):
    img = Image.open(image_file)
    return img


def main():
    st.title("Brain Tumor Detection System")

    menu = ["Home", "About"]

    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Upload a brain MRI image for prediction")
        image_file = st.file_uploader("Upload an Image", type=["jpg"])
        if image_file is not None:
            # file_details = {"FileName": image_file.name, "FileType": image_file.type}
            # st.write(file_details)
            # st.write(dir(image_file))
            # st.write(type(image_file))
            img = load_image(image_file)
            st.image(img)
            # saving file (needed for making prediction)
            with open(os.path.join("uploaded_test", image_file.name), "wb") as f:
                f.write(image_file.getbuffer())

            # st.success("File Saved")
            prediction = RetPreds(model)
            st.write("Looks like this MRI image is:", prediction[0])
            # st.write(prediction[0])

            path = os.path.join("uploaded_test/", image_file.name)
            os.remove(path)

    else:
        st.subheader("About")
        st.write(
            "Welcome to our Brain MRI Predictor! This Streamlit app utilizes cutting-edge Convolutional Neural Network (CNN) technology to analyze brain MRI images and predict the presence of abnormalities with an impressive 98% accuracy rate. Simply upload your MRI image, and our powerful model will swiftly process it, providing you with insights into the presence of any anomalies. Whether you're a healthcare professional or an individual concerned about your brain health, our app offers a reliable and accessible tool for early detection and diagnosis."
        )
        st.subheader("Samples of data: ")
        st.image("resource/examples.png")
        st.subheader("Model Parameteres: ")
        st.image("resource/parameters.png", use_column_width=True)
        st.subheader("Loss vs Accuracy plot over Training and validation set:")
        st.image("resource/training.png", use_column_width=True)
        code = """
        Model_Results = Model.evaluate(Test_Set, verbose=False)
        print("LOSS: " + "%.4f" % Model_Results[0])
        print("ACCURACY: " + "%.4f" % Model_Results[1])
        """

        st.code(code, language="python", line_numbers=False)
        st.text("LOSS: 0.0965\nACCURACY: 0.9800")


if __name__ == "__main__":
    main()
