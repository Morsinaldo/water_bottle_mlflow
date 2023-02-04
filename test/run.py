import wandb
import logging
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import fbeta_score, precision_score, recall_score, accuracy_score

import torch
from torch.utils.data import DataLoader
from torchvision import models
from torch import nn

from helpers import ImageDataset, MyVisionTransformerModel

HYP = {
        'seed': 44,
        'batch_size': 50,
        'img_size': (225,225),
        'epochs': 50,
        'patience': 5,
        'learning_rate': 0.0001,
    }

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()

args = {
  "project_name": "water_bottle_classifier",
  "test_feature_artifact": "test_x:latest",
  "test_target_artifact": "test_y:latest",
  "encoder": "target_encoder:latest",
  "inference_model": "vit_l_32.pth:latest"
}

wandb.login()

# open the W&B project created in the Fetch step
run = wandb.init(entity="morsinaldo",project=args["project_name"], job_type="Test")

logger.info("Downloading the test data")
# test x
test_x_artifact = run.use_artifact(args["test_feature_artifact"])
test_x_path = test_x_artifact.file()

# test y
test_y_artifact = run.use_artifact(args["test_target_artifact"])
test_y_path = test_y_artifact.file()

# unpacking the artifacts
test_x = joblib.load(test_x_path)
test_y = joblib.load(test_y_path)

classes = {'Full  Water level': 0, 'Half water level': 1, 'Overflowing': 2}
test_y = np.array([classes[i] for i in test_y])

test_dataset = ImageDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=HYP['batch_size'], shuffle=False)

# load the model
# model = models.vit_l_32(pretrained=False)
# for param in model.parameters():
#     param.requires_grad = False

# model.heads.head = nn.Linear(1024, 3)
# model.load_state_dict(torch.load('./vit_l_32.pth'))

# device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
# model.to(device)

model = MyVisionTransformerModel(len_encoding=3, logger=logger, pretrained=False)
model.load_model("./vit_l_32.pth")

# make predictions
# test_predictions = []
# test_labels = []
# with torch.no_grad():
#     for i, (t_images, t_labels) in enumerate(test_loader):
#         t_images = t_images.to(device)
#         t_labels = t_labels.to(device)
#         outputs = model(t_images.float())
#         _, predictions = torch.max(outputs, 1)
#         predictions = predictions.to('cpu')
#         t_labels = t_labels.to('cpu')
#         test_predictions.append(predictions.numpy())
#         test_labels.append(t_labels.numpy())

# test_predictions = np.hstack(test_predictions)
# test_labels = np.hstack(test_labels)

test_predictions, test_labels = model.predict(test_loader)

plt.style.use("ggplot")
fig_confusion_matrix, ax = plt.subplots(1,1,figsize=(7,4))
ConfusionMatrixDisplay(confusion_matrix(test_predictions,
                                        test_labels),
                       display_labels=classes).plot(values_format=".0f",ax=ax)

ax.set_xlabel("True Label")
ax.set_ylabel("Predicted Label")
ax.grid(False)
# plt.show()

# compute the metrics
logger.info("Test Evaluation metrics")
precision = precision_score(test_labels, test_predictions, average='macro')
recall = recall_score(test_labels, test_predictions, average='macro')
accuracy = accuracy_score(test_labels, test_predictions)
fbeta = fbeta_score(test_labels, test_predictions, beta=0.5, average='macro')

# log the metrics
logger.info("Test Accuracy: {}".format(accuracy))
logger.info("Test Precision: {}".format(precision))
logger.info("Test Recall: {}".format(recall))
logger.info("Test F1: {}".format(fbeta))

run.summary["Acc"] = accuracy
run.summary["Precision"] = precision
run.summary["Recall"] = recall
run.summary["F1"] = fbeta

# log the confusion matrix
logger.info("Logging the confusion matrix")
run.log({"Confusion Matrix": wandb.Image(fig_confusion_matrix)})

run.finish()

