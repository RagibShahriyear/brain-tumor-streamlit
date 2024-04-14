import streamlit as st
import os


from keras.models import load_model
from main import RetPreds

model = load_model("trained_model.h5")


# File Processing Pkgs
from PIL import Image


# Load Images
@st.cache_data
def load_image(image_file):
    img = Image.open(image_file)
    return img


def main():
    st.title("FIle Uploads & Saved File to Directory App")

    menu = ["Home", "Datasets", "About"]

    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Upload Images")

        # new feature
        if "file_uploader_key" not in st.session_state:
            st.session_state["file_uploader_key"] = 0

        if "uploaded_files" not in st.session_state:
            st.session_state["uploaded_files"] = []

        image_file = st.file_uploader(
            "Upload an Image",
            type=["png", "jpg", "jpeg"],
            key=st.session_state["file_uploader_key"],
        )
        if image_file is not None:
            file_details = {"FileName": image_file.name, "FileType": image_file.type}
            st.write(file_details)
            # st.write(dir(image_file))
            # st.write(type(image_file))
            img = load_image(image_file)
            st.image(img)
            # saving file
            with open(os.path.join("uploaded_test", image_file.name), "wb") as f:
                f.write(image_file.getbuffer())

            # st.success("File Saved")
            prediction = RetPreds(model)
            st.write(prediction)

            path = os.path.join("uploaded_test/", image_file.name)
            os.remove(path)

        # if st.button("Clear uploaded files"):
        #     # Clear the session state list of uploaded files
        #     st.session_state["uploaded_files"] = []
        #     # Increment the key to reset the file uploader
        #     st.session_state["file_uploader_key"] += 1
        #     st.experimental_rerun()

        # st.write("Uploaded files:", st.session_state["uploaded_files"])

    else:
        st.subheader("About")


if __name__ == "__main__":
    main()
