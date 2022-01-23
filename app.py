#Importing the core packages

import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np 
import os

@st.cache
def load_image(img):
    im = Image.open(img)
    return im


face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_smile.xml')

def detect_faces(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    #Draw rectangle
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
    return img, faces

def detect_eyes(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    for(ex,ey,ew,eh) in eyes:
        cv2.rectangle(img, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
    return img

def detect_smile(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Detect smiles
    smiles = smile_cascade.detectMultiScale(gray, 1.3, 5)
    for(sx,sy,sw,sh) in smiles:
        cv2.rectangle(img, (sx,sy), (sx+sw, sy+sh), (0,0,255), 2)
    return img


def main():
    """ Face Detection App"""

    st.title("Face Detection App")
    st.text("Build with Streamlit and OpenCV")

    activities = ["Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice =="Detection":
        st.subheader("Face Detection")

        image_file = st.file_uploader("Upload Image", type=['jpg', 'png','jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original Image")
            st.write(type(our_image))
            st.image(our_image)

        enhance_type = st.sidebar.radio("Enhance Type", ["Original", "Gray-Scale", "Contrast", "Brightness", "Blurring"])
        if enhance_type =="Gray-Scale":
            new_img = np.array(our_image.convert('RGB'))
            img = cv2.cvtColor(new_img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #st.write(new_img)
            st.image(gray)

        elif enhance_type =="Contrast":
            c_rate = st.sidebar.slider("Contrast", 0.0, 10.0)
            enhancer = ImageEnhance.Contrast(our_image)
            img_output= enhancer.enhance(c_rate)
            st.image(img_output)

        elif enhance_type =="Brightness":
            c_rate = st.sidebar.slider("Brightness", 0.0, 10.0)
            enhancer = ImageEnhance.Brightness(our_image)
            img_output= enhancer.enhance(c_rate)
            st.image(img_output)

        elif enhance_type =="Blurring":
            new_img = np.array(our_image.convert('RGB'))
            blur_rate = st.sidebar.slider("Blurriness", 0.1, 10.0)
            img = cv2.cvtColor(new_img, 1)
            blur_image = cv2.GaussianBlur(img, (11,11),blur_rate)
            st.image(blur_image)



        #Face Detection
        task = ["Faces","Smiles","Eyes"]
        feature_choice  = st.sidebar.selectbox("Find Features", task)

        if st.button("Process"):

            if feature_choice =="Faces":
                result_img, result_faces = detect_faces(our_image)
                st.image(result_img)

                st.success(f"Found {len(result_faces)} faces ")

            elif feature_choice =="Eyes":
                result_img = detect_eyes(our_image)
                st.image(result_img)

            elif feature_choice =="Smiles":
                result_img = detect_smile(our_image)
                st.image(result_img)

        
    elif choice =="About":
        st.subheader("About")
        st.write("""
        This app is made with the use of Streamlit and Open CV library by Saurjayan Bhattacharjee. 
        Python 3.8.7 was used to build this app.
        This app can successfully draw rectangles around faces , smiles and eyes of the subject image.
        Apart from this it can process images into different levels of basic editing which include change the image 
        to its gray scaled version, adjust brightness of the image, adjust Gaussian Blur and contrast of the image
        """)

if __name__=='__main__':
    main()