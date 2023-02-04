# import the necessary packages
import logging
import joblib
import wandb

from imutils import paths
from helpers import *

# login to wandb
wandb.login()

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()

# since we are using Jupyter Notebooks we can replace our argument
# parsing code with *hard coded* arguments and values
args = {
	"dataset": "data",
    "project_name": "water_bottle_classifier",
    "artifact_name": "water_bottle_raw_dataset:latest",
    "features": "clean_features",
    "target": "labels",
}

# open the W&B project created in the Fetch step
run = wandb.init(entity="morsinaldo",project=args["project_name"], job_type="preprocessing")

# download the raw data from W&B
raw_data = run.use_artifact(args["artifact_name"])
data_dir = raw_data.download()
logger.info("Path: {}".format(data_dir))

# grab the list of images that we'll be describing
logger.info("[INFO] preprocessing images...")
imagePaths = list(paths.list_images(data_dir))

# initialize the image preprocessors
sp = SimplePreprocessor(224,224)

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp], logger=logger)
(data, labels) = sdl.load(imagePaths, verbose=50)
data = data.astype("float") / 255.0

# show some information on memory consumption of the images
logger.info("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1024)))
logger.info("[INFO] labels vector: {:.1f}MB".format(labels.nbytes / (1024 * 1024)))
logger.info("[INFO] features shape: {}, labels shape: {}".format(data.shape,labels.shape))

# Save the feature artifacts using joblib
joblib.dump(data, args["features"])

# Save the target using joblib
joblib.dump(labels, args["target"])

logger.info("Dumping the clean data artifacts to disk")

# clean data artifact
artifact = wandb.Artifact(args["features"],
                          type="CLEAN_DATA",
                          description="A json file representing the clean features data"
                          )

logger.info("Logging clean data artifact")
artifact.add_file(args["features"])
run.log_artifact(artifact)

# clean label artifact
artifact = wandb.Artifact(args["target"],
                          type="CLEAN_DATA",
                          description="A json file representing the clean target"
                          )

logger.info("Logging clean target artifact")
artifact.add_file(args["target"])
run.log_artifact(artifact)

run.finish()