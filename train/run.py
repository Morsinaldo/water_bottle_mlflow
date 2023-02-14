"""
Creator: Morsinaldo Medeiros
Date: 06-02-2023
Description: This script is responsible for running the train step of the pipeline.
"""

import wandb
import joblib
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

from codecarbon import EmissionsTracker
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import torch
from torch.utils.data import DataLoader
from torch import nn

from helpers.images_processing import ImageDataset
from helpers.model import MyVisionTransformerModel

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()

# classes = {'Full  Water level': 0, 'Half water level': 1, 'Overflowing': 2}

def process_args(args):
    """
    Arguments:
        args - command line arguments
        args.train_feature_artifact - name of the artifact containing the train features
        args.train_target_artifact - name of the artifact containing the train target
        args.val_feature_artifact - name of the artifact containing the validation features
        args.val_target_artifact - name of the artifact containing the validation target
        args.encoder - name of the artifact containing the encoder
        args.inference_model - name of the artifact containing the inference model
        args.batch_size - batch size for the dataloader
        args.seed - seed for the random number generator
        args.epochs - number of epochs to train the model
        args.learning_rate - learning rate for the optimizer
    """

    wandb.login()

    # open the W&B project created in the Fetch step
    run = wandb.init(job_type="Train")

    logger.info("Downloading the train and validation data")
    # train x
    train_x_artifact = run.use_artifact(args.train_feature_artifact)
    train_x_path = train_x_artifact.file()

    # train y
    train_y_artifact = run.use_artifact(args.train_target_artifact)
    train_y_path = train_y_artifact.file()

    # validation x
    val_x_artifact = run.use_artifact(args.val_feature_artifact)
    val_x_path = val_x_artifact.file()

    # validation y
    val_y_artifact = run.use_artifact(args.val_target_artifact)
    val_y_path = val_y_artifact.file()

    # unpacking the artifacts
    train_x = joblib.load(train_x_path)
    train_y = joblib.load(train_y_path)
    val_x = joblib.load(val_x_path)
    val_y = joblib.load(val_y_path)

    # log the shape of the data
    logger.info("Train x: {}".format(train_x.shape))
    logger.info("Train y: {}".format(train_y.shape))
    logger.info("Validation x: {}".format(val_x.shape))
    logger.info("Validation y: {}".format(val_y.shape))

    # encode the target
    logger.info("Encoding the target")
    encoder = LabelEncoder()

    # fit the encoder
    train_y = encoder.fit_transform(train_y)
    val_y = encoder.transform(val_y)

    # save the encoder
    logger.info("Saving the encoder")
    joblib.dump(encoder, args.encoder)

    # create the dataset and dataloader
    train_dataset = ImageDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = ImageDataset(val_x, val_y)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # compute class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_y), y=train_y)
    logger.info("Class weights: {}".format(class_weights))

    # total steps
    total_steps = len(train_loader)
    logger.info("Total steps: {}".format(total_steps))

    model = MyVisionTransformerModel(len_encoding=len(encoder.classes_), logger=logger, pretrained=True)

    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights.astype(np.float32)).to(model.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # create codecarbon tracker
    tracker = EmissionsTracker(log_level="critical")
    tracker.start()

    history = model.train(train_loader, test_loader, criterion, optimizer, total_steps, args.epochs)

    emissions = tracker.stop()

    test_predictions, test_labels = model.predict(test_loader)

    fig_confusion_matrix, ax = plt.subplots(1,1,figsize=(7,4))
    ConfusionMatrixDisplay(confusion_matrix(test_predictions,
                                            test_labels),
                          display_labels=encoder.classes_).plot(values_format=".0f",ax=ax)

    ax.set_xlabel("True Label")
    ax.set_ylabel("Predicted Label")
    ax.grid(False)

    # plot the training and validation loss and accuracy
    plt.style.use("ggplot")
    fig_loss, ax = plt.subplots(1,1,figsize=(7,4))
    ax.plot(history['train_acc'], label='train_acc')
    ax.plot(history['train_loss'], label='train_loss', linestyle='--')
    ax.plot(history['val_acc'], label='val_acc')
    ax.plot(history['val_loss'], label='val_loss', linestyle='--')
    ax.set_title("Training Loss and Accuracy")
    ax.set_xlabel("Epoch #")
    ax.set_ylabel("Loss/Accuracy")
    ax.legend()

    fig_confusion_matrix.savefig("confusion_matrix.png")
    fig_loss.savefig("loss.png")

    # save the model
    torch.save(model.state_dict(), args.inference_model)

    # send the model to the wandb as an artifact
    logger.info("Saving the model")
    model_artifact = wandb.Artifact(args.inference_model, type='model')
    model_artifact.add_file(args.inference_model)
    wandb.log_artifact(model_artifact)

    # send the encoder to wandb as an artifact
    logger.info("Saving the encoder")
    encoder_artifact = wandb.Artifact(args.encoder, type='encoder')
    encoder_artifact.add_file(args.encoder)
    wandb.log_artifact(encoder_artifact)

    # Uploading figures
    logger.info("Uploading figures")
    run.log(
        {
            "confusion_matrix": wandb.Image(fig_confusion_matrix),
            "loss": wandb.Image(fig_loss),
            # "other_figure": wandb.Image(other_fig)
        }
    )

    # log the metrics
    logger.info("Logging metrics")
    run.log({"train_loss": history['train_loss'][-1],
                "val_loss": history['val_loss'][-1],
                "train_acc": history['train_acc'][-1],
                "val_acc": history['val_acc'][-1],
                "emissions": emissions
                })
    
    precision = precision_score(test_predictions, test_labels, average='weighted')
    recall = recall_score(test_predictions, test_labels, average='weighted')
    f1 = f1_score(test_predictions, test_labels, average='weighted')
    accuracy = accuracy_score(test_predictions, test_labels)

    run.summary["precision"] = precision
    run.summary["recall"] = recall
    run.summary["f1"] = f1
    run.summary["accuracy"] = accuracy

    # energy unit is kWh
    run.summary["Energy_Consumed"] = tracker.final_emissions_data.energy_consumed
    run.summary["Energy_RAM"] = tracker.final_emissions_data.ram_energy
    run.summary["Energy_GPU"] = tracker.final_emissions_data.gpu_energy
    run.summary["Energy_CPU"] = tracker.final_emissions_data.cpu_energy
    # kg
    run.summary["CO2_Emissions"] = tracker.final_emissions_data.emissions

    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_feature_artifact', type=str, required=True,
                        help="Name of the artifact containing the training features")
    parser.add_argument('--train_target_artifact', type=str, required=True,
                        help="Name of the artifact containing the training target")
    parser.add_argument('--val_feature_artifact', type=str, required=True,
                        help="Name of the artifact containing the validation features")
    parser.add_argument('--val_target_artifact', type=str, required=True,
                        help="Name of the artifact containing the validation target")
    parser.add_argument('--encoder', type=str, required=True,
                        help="Name of the artifact containing the target encoder")
    parser.add_argument('--inference_model', type=str, required=True,
                        help="Name of the artifact containing the inference model")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size")
    parser.add_argument('--seed', type=int, required=True, help="Random seed")
    parser.add_argument('--epochs', type=int, required=True, help="Number of epochs")
    parser.add_argument('--learning_rate', type=float, required=True, help="Learning rate")

    ARGS = parser.parse_args()

    process_args(ARGS)