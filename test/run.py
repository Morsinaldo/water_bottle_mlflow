"""
Creator: Morsinaldo Medeiros
Date: 05-02-2023
Description: This script is responsible for running the test step of the pipeline.
"""
import wandb
import logging
import joblib
import argparse
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from torch.utils.data import DataLoader

from helpers.images_processing import ImageDataset
from helpers.model import MyVisionTransformerModel

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
        args.test_feature_artifact - name of the artifact containing the test features
        args.test_target_artifact - name of the artifact containing the test target
        args.encoder - name of the artifact containing the encoder
        args.inference_model - name of the artifact containing the inference model
        args.batch_size - batch size for the dataloader
    """

    wandb.login()

    # open the W&B project created in the Fetch step
    run = wandb.init(job_type="Test")

    logger.info("Downloading the test data")
    # test x
    test_x_artifact = run.use_artifact(args.test_feature_artifact)
    test_x_path = test_x_artifact.file()

    # test y
    test_y_artifact = run.use_artifact(args.test_target_artifact)
    test_y_path = test_y_artifact.file()

    # unpacking the artifacts
    test_x = joblib.load(test_x_path)
    test_y = joblib.load(test_y_path)

    # download the encoder
    logger.info("Downloading the encoder")
    encoder_artifact = run.use_artifact(args.encoder)
    encoder_path = encoder_artifact.file()
    encoder = joblib.load(encoder_path)

    # download the inference model
    logger.info("Downloading the inference model")
    model_artifact = run.use_artifact(args.inference_model)
    model_path = model_artifact.file()

    # encoding the test target
    logger.info("Encoding the test target")
    test_y = encoder.transform(test_y)
    
    # create the test dataloader
    logger.info("Creating the test dataloader")
    test_dataset = ImageDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = MyVisionTransformerModel(len_encoding=len(encoder.classes_), logger=logger, pretrained=False)
    model.load_model(model_path)

    test_predictions, test_labels = model.predict(test_loader)

    plt.style.use("ggplot")
    fig_confusion_matrix, ax = plt.subplots(1,1,figsize=(7,4))
    ConfusionMatrixDisplay(confusion_matrix(test_predictions,
                                            test_labels),
                          display_labels=encoder.classes_).plot(values_format=".0f",ax=ax)
    ax.set_xlabel("True Label")
    ax.set_ylabel("Predicted Label")
    ax.grid(False)

    # log the confusion matrix
    logger.info("Logging the confusion matrix")
    run.log({"Confusion Matrix": wandb.Image(fig_confusion_matrix)})

    # compute the metrics
    logger.info("Test Evaluation metrics")
    precision = precision_score(test_labels, test_predictions, average='weighted')
    recall = recall_score(test_labels, test_predictions, average='weighted')
    accuracy = accuracy_score(test_labels, test_predictions)
    f1 = f1_score(test_labels, test_predictions, average='weighted')

    # log the metrics
    logger.info("Test Accuracy: {}".format(accuracy))
    logger.info("Test Precision: {}".format(precision))
    logger.info("Test Recall: {}".format(recall))
    logger.info("Test F1: {}".format(f1))

    run.summary["Acc"] = accuracy
    run.summary["Precision"] = precision
    run.summary["Recall"] = recall
    run.summary["F1"] = f1

    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test step of the pipeline")

    parser.add_argument("--test_feature_artifact", type=str, required=True,
                        help="name of the artifact containing the test features")
    parser.add_argument("--test_target_artifact", type=str, required=True,
                        help="name of the artifact containing the test target")
    parser.add_argument("--encoder", type=str, required=True,
                        help="name of the artifact containing the encoder")
    parser.add_argument("--inference_model", type=str, required=True,
                        help="name of the artifact containing the inference model")
    parser.add_argument("--batch_size", type=int, required=True,
                        help="batch size for the dataloader")

    ARGS = parser.parse_args()

    process_args(ARGS)
