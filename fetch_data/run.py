"""
Creator: Morsinaldo Medeiros
Date: 05-02-2023
Description: This script is responsible for downloading 
the dataset from Google Drive and upload it to W&B.
"""

import os
import wandb
import gdown
import logging
import zipfile
import argparse
from imutils import paths

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()

def process_args(args):
    """
    Arguments:
      args - command line arguments
      args.dataset - path to the dataset
      args.url - url to download the dataset
      args.artifact_name - name of the artifact containing the raw data
    """

    # download the dataset
    logger.info("[INFO] downloading dataset...")
    output = '../'
    gdown.download(args.url, output, quiet=False)

    # unzip the dataset
    logger.info("[INFO] unzipping dataset...")
    with zipfile.ZipFile("../data.zip", 'r') as zip_ref:
        zip_ref.extractall("../")
      
    # login to wandb
    wandb.login()

    run = wandb.init(job_type="fetch_data")

    # create an artifact for all the raw data
    raw_data = wandb.Artifact(args.artifact_name, type="raw_data")

    # grab the list of images that we'll be describing
    logger.info("[INFO] loading images...")
    image_paths = list(paths.list_images(args.dataset))

    # append all images to the artifact
    for img in image_paths:
      label = img.split(os.path.sep)
      raw_data.add_file(img, name=os.path.join(label[-2],label[-1]))

    # save artifact to W&B
    run.log_artifact(raw_data)
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--artifact_name", type=str, required=True,
                        help="name of the artifact containing the raw data")
    parser.add_argument("--dataset", type=str, required=True,
                        help="path to the dataset")
    parser.add_argument("--url", type=str, required=True,
                        help="url to download the dataset")

    ARGS = parser.parse_args()

    process_args(ARGS)

# run the script
    # mlflow run . -P project_name=water_bottle_classifier \
    #               -P artifact_name=water_bottle_raw_dataset \
    #               -P dataset=data \
    #               -P url=https://drive.google.com/uc?id=1hb9P1KVMcMBLHhJKKU-_FWX_g7uHb74A