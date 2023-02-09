"""
Creator: Morsinaldo Medeiros
Date: 09-02-2023
Description: This is a Streamlit app that allows you to upload an image and make a prediction
"""

import torch
import wandb
import joblib
import logging
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO
from helpers import MyVisionTransformerModel

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()

wandb.login()

@st.cache(suppress_st_warning=True)
def load_model():
    run = wandb.init(job_type="Inference")
    model_artifact = run.use_artifact('morsinaldo/water_bottle_classifier/vit_l_32.pth:v0', type='model')
    model_path = model_artifact.file()

    logger.info("Downloading the encoder")
    encoder_artifact = run.use_artifact('morsinaldo/water_bottle_classifier/target_encoder:v0', type='encoder')
    encoder_path = encoder_artifact.file()
    encoder = joblib.load(encoder_path)

    logger.info("Downloading the model")
    model = MyVisionTransformerModel(len_encoding=len(encoder.classes_), logger=logger, pretrained=False)
    model.load_model(model_path)

    return model, encoder


STYLE = """
    <style>
    img {
        max-width: 100%;
    }
    </style>
"""

def main():
    """Run this function to display the Streamlit app"""
    
    file = st.file_uploader("Upload file", type=["csv", "png", "jpg"])
    show_file = st.empty()
 
    if not file:
        show_file.info("Please upload a file of type: " + ", ".join(["png", "jpg", "jpeg"]))
        return
 
    if isinstance(file, BytesIO):
        show_file.image(file)

    array = np.array(Image.open(file))
    
    # add a button to make a prediction
    if st.button("Predict"):
        
        # load the model and the encoder
        logger.info("Loading the model and the encoder")
        model, encoder = load_model()

        logger.info("Making a prediction")
        prediction = model.predict_image(array)

        logger.info("Decoding the prediction")
        pred_class = encoder.inverse_transform([torch.argmax(prediction).item()])[0]

        st.write("Predicted class: {}".format(pred_class))
    file.close()
 
main()