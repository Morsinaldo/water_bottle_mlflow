"""
Creator: Morsinaldo Medeiros
Date: 04-02-2023
Description: This script will download the clean data and labels artifacts from 
the Fetch step and will split the data into training, validation and test sets. 
The training and validation sets will be used to train the model and the test set 
will be used to evaluate the model. The training, validation and test sets will 
be saved as artifacts and uploaded to W&B.
"""

# import the necessary packages
import argparse
import logging
import os
import joblib
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split

# login to wandb
wandb.login()

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
      args.project_name - name of the project
      args.artifact_name_feature - name of the artifact containing the clean features
      args.artifact_name_target - name of the artifact containing the labels
      args.train_feature_artifact - name of the artifact containing the training features
      args.train_target_artifact - name of the artifact containing the training labels
      args.val_feature_artifact - name of the artifact containing the validation features
      args.val_target_artifact - name of the artifact containing the validation labels
      args.test_feature_artifact - name of the artifact containing the test features
      args.test_target_artifact - name of the artifact containing the test labels
      args.test_size - percentage of the data to be used for testing
      args.val_size - percentage of the data to be used for validation
      args.seed - seed for random number generator
    """
    # open the W&B project created in the Fetch step
    run = wandb.init(entity="morsinaldo",project=args.project_name, job_type="data_segregation")

    logger.info("Downloading and reading clean data artifact")
    clean_data = run.use_artifact(args.artifact_name_feature)
    clean_data_path = clean_data.file()

    logger.info("Downloading and reading label data artifact")
    label_data = run.use_artifact(args.artifact_name_target)
    label_data_path = label_data.file()

    # unpacking the artifacts
    data = joblib.load(clean_data_path)
    label = joblib.load(label_data_path)

    # partition the data into training, test splits using 75% of
    # the data for training and the remaining 25% for test
    (train_x, test_x, train_y, test_y) = train_test_split(data, label,test_size=args.test_size, random_state=args.seed)

    # partition the training into training, validation splits using 75% of
    # the training set for training and the remaining 25% for validation
    (train_x, val_x, train_y, val_y) = train_test_split(train_x, train_y,test_size=args.val_size, random_state=args.seed)

    logger.info("Train x: {}".format(train_x.shape))
    logger.info("Train y: {}".format(train_y.shape))
    logger.info("Validation x: {}".format(val_x.shape))
    logger.info("Validation y: {}".format(val_y.shape))
    logger.info("Test x: {}".format(test_x.shape))
    logger.info("Test y: {}".format(test_y.shape))

    # Save the artifacts using joblib
    joblib.dump(train_x, args.train_feature_artifact)
    joblib.dump(train_y, args.train_target_artifact)
    joblib.dump(val_x, args.val_feature_artifact)
    joblib.dump(val_y, args.val_target_artifact)
    joblib.dump(test_x, args.test_feature_artifact)
    joblib.dump(test_y, args.test_target_artifact)

    logger.info("Dumping the train and validation data artifacts to the disk")

    # train_x artifact
    artifact = wandb.Artifact(args.train_feature_artifact,
                              type="TRAIN_DATA",
                              description="A json file representing the train_x"
                              )

    logger.info("Logging train_x artifact")
    artifact.add_file(args.train_feature_artifact)
    run.log_artifact(artifact)

    # train_y artifact
    artifact = wandb.Artifact(args.train_target_artifact,
                              type="TRAIN_DATA",
                              description="A json file representing the train_y"
                              )

    logger.info("Logging train_y artifact")
    artifact.add_file(args.train_target_artifact)
    run.log_artifact(artifact)

    # val_x artifact
    artifact = wandb.Artifact(args.val_feature_artifact,
                              type="VAL_DATA",
                              description="A json file representing the val_x"
                              )

    logger.info("Logging val_x artifact")
    artifact.add_file(args.val_feature_artifact)
    run.log_artifact(artifact)

    # val_y artifact
    artifact = wandb.Artifact(args.val_target_artifact,
                              type="VAL_DATA",
                              description="A json file representing the val_y"
                              )

    logger.info("Logging val_y artifact")
    artifact.add_file(args.val_target_artifact)
    run.log_artifact(artifact)

    # test_x artifact
    artifact = wandb.Artifact(args.test_feature_artifact,
                              type="TEST_DATA",
                              description="A json file representing the test_x"
                              )

    logger.info("Logging test_x artifact")
    artifact.add_file(args.test_feature_artifact)
    run.log_artifact(artifact)

    # test_y artifact
    artifact = wandb.Artifact(args.test_target_artifact,
                              type="TEST_DATA",
                              description="A json file representing the test_y"
                              )

    logger.info("Logging test_y artifact")
    artifact.add_file(args.test_target_artifact)
    run.log_artifact(artifact)

    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Segregation")

    parser.add_argument("--project_name", type=str, default="water_bottle_classifier", required=False)
    parser.add_argument("--artifact_name_feature", type=str, default="clean_features:latest", required=False)
    parser.add_argument("--artifact_name_target", type=str, default="labels:latest", required=False)
    parser.add_argument("--train_feature_artifact", type=str, default="train_x", required=False)
    parser.add_argument("--train_target_artifact", type=str, default="train_y", required=False)
    parser.add_argument("--val_feature_artifact", type=str, default="val_x", required=False)
    parser.add_argument("--val_target_artifact", type=str, default="val_y", required=False)
    parser.add_argument("--test_feature_artifact", type=str, default="test_x", required=False)
    parser.add_argument("--test_target_artifact", type=str, default="test_y", required=False)
    parser.add_argument("--test_size", type=float, default=0.25, required=False)
    parser.add_argument("--val_size", type=float, default=0.25, required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)

    ARGS = parser.parse_args()

    process_args(ARGS)

# Run the script
# mlflow run . -P project_name="water_bottle_classifier" \
#               -P artifact_name_feature="clean_features:latest" \
#               -P artifact_name_target="labels:latest" \
#               -P train_feature_artifact="train_x" \
#               -P train_target_artifact="train_y" \
#               -P val_feature_artifact="val_x" \
#               -P val_target_artifact="val_y" \
#               -P test_feature_artifact="test_x" \
#               -P test_target_artifact="test_y" \
#               -P test_size=0.25 \
#               -P val_size=0.25 \
#               -P seed=42