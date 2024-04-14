import streamlit as st
import matplotlib.pyplot as plt


# File Processing Pkgs
from PIL import Image


# Load Images
@st.cache_data
def load_image(image_file):
    img = Image.open(image_file)
    return img


def main():
    st.title("File Uplaod Tutorial")

    menu = ["Home", "Dataset", "DocumentFiles", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])

        if image_file is not None:
            # To see details
            st.write(type(image_file))
            st.write(dir(image_file))
            file_details = {
                "filename": image_file.name,
                "filetype": image_file.type,
                "filesize": image_file.size,
            }
            st.write(file_details)

            st.image(load_image(image_file))

    elif choice == "Dataset":
        st.subheader("Dataset")

    elif choice == "DocumentFiles":
        st.subheader("DocumentFiles")

    else:
        st.subheader("About")


if __name__ == "__main__":
    main()
