import wandb
import gdown
import os
import logging
from imutils import paths

# login to wandb
wandb.login()

# download the dataset
url = 'https://drive.google.com/uc?id=1hb9P1KVMcMBLHhJKKU-_FWX_g7uHb74A'
output = '../'
gdown.download(url, output, quiet=False)

# unzip the dataset
import zipfile

with zipfile.ZipFile("../data.zip", 'r') as zip_ref:
    zip_ref.extractall("../")

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()

args = {
	"dataset": "data",
    "project_name": "water_bottle_classifier",
    "artifact_name": "water_bottle_raw_dataset",
}

run = wandb.init(entity="morsinaldo",project=args["project_name"], job_type="fetch_data")

# create an artifact for all the raw data
raw_data = wandb.Artifact(args["artifact_name"], type="raw_data")

# grab the list of images that we'll be describing
logger.info("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# append all images to the artifact
for img in imagePaths:
  label = img.split(os.path.sep)
  raw_data.add_file(img, name=os.path.join(label[-2],label[-1]))

# save artifact to W&B
run.log_artifact(raw_data)
run.finish()