import os
import wandb
import joblib
import logging
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch import nn

from utils.helpers import ImageDataset, MyVisionTransformerModel

# batch_size = 50

# class ImageDataset(Dataset):
#     def __init__(self, X, y):
#         self.x = X
#         self.y = y
#         self.n_samples = len(X)
        
#     def __getitem__(self, index):
#         transform = transforms.Compose([transforms.ToTensor()])
#         return transform(self.x[index]), torch.tensor(self.y[index], dtype=torch.int64)
    
#     def __len__(self):
#         return self.n_samples

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()

classes = {'Full  Water level': 0, 'Half water level': 1, 'Overflowing': 2}

# since we are using Jupyter Notebooks we can replace our argument
# parsing code with *hard coded* arguments and values
args = {
  "project_name": "water_bottle_classifier",
  "train_feature_artifact": "train_x:latest",
  "train_target_artifact": "train_y:latest",
  "val_feature_artifact": "val_x:latest",
  "val_target_artifact": "val_y:latest",
  "encoder": "target_encoder",
  "inference_model": "vit_l_32.pth"
}

HYP = {
        'seed': 44,
        'batch_size': 50,
        'img_size': (225,225),
        'epochs': 50,
        'patience': 5,
        'learning_rate': 0.0001,
    }

# open the W&B project created in the Fetch step
run = wandb.init(entity="morsinaldo",project=args["project_name"], job_type="Train")

logger.info("Downloading the train and validation data")
# train x
train_x_artifact = run.use_artifact(args["train_feature_artifact"])
train_x_path = train_x_artifact.file()

# train y
train_y_artifact = run.use_artifact(args["train_target_artifact"])
train_y_path = train_y_artifact.file()

# validation x
val_x_artifact = run.use_artifact(args["val_feature_artifact"])
val_x_path = val_x_artifact.file()

# validation y
val_y_artifact = run.use_artifact(args["val_target_artifact"])
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

# map train_y and val_y to 0, 1, 2
train_y = np.array([classes[i] for i in train_y])
val_y = np.array([classes[i] for i in val_y])

# create the dataset and dataloader
train_dataset = ImageDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=HYP['batch_size'], shuffle=True)
test_dataset = ImageDataset(val_x, val_y)
test_loader = DataLoader(test_dataset, batch_size=HYP['batch_size'])

# compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_y), y=train_y)
logger.info("Class weights: {}".format(class_weights))

# total steps
total_steps = len(train_loader)
logger.info("Total steps: {}".format(total_steps))

# define the model
# model = models.vit_l_32(pretrained=True)
# for param in model.parameters():
#     param.requires_grad = False

water_level_encoding = {}
for index, label in enumerate(train_y):
    water_level_encoding[label] = index
water_level_encoding

# model.heads.head = nn.Linear(1024, len(water_level_encoding))

# device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
# logger.info("Device: {}".format(device))
# model.to(device)

model = MyVisionTransformerModel(num_classes=len(water_level_encoding))

criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights.astype(np.float32)).to(model.device))
optimizer = torch.optim.Adam(model.parameters(), lr=HYP['learning_rate'])

# history = {
#     'train_loss': [],
#     'train_acc': [],
#     'val_loss': [],
#     'val_acc': []
# }

# epochs = HYP['epochs']

# for epoch in range(HYP['epochs']):
#     for i, (train_images, train_labels) in enumerate(train_loader):
#         train_images = train_images.to(device)
#         train_labels = train_labels.to(device)
#         outputs = model(train_images.float())
#         loss = criterion(outputs, train_labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         with torch.no_grad():
#             train_images.to('cpu')
#             train_labels.to('cpu')
#             predicted_outputs = model(train_images.float())
#             _, train_predictions = torch.max(predicted_outputs, 1)
#             n_correct = (train_predictions == train_labels).sum().item()
#             acc = n_correct/len(train_images)
            
#         if (i+1) % 4 == 0:
#             with torch.no_grad():
#                 n_test_correct = 0
#                 n_test = 0
#                 for test_images, test_labels in test_loader:
#                     test_images = test_images.to(device)
#                     test_labels = test_labels.to(device)
#                     test_outputs = model(test_images.float())
#                     _, test_predictions = torch.max(test_outputs, 1)
#                     test_images.to('cpu')
#                     test_labels.to('cpu')
#                     n_test_correct += (test_predictions == test_labels).sum().item()
#                     n_test += len(test_images)
#                 val_acc = n_test_correct/n_test
#             logger.info(f'epoch {epoch+1} / {epochs}, step {i+1} / {total_steps}, loss = {loss.item():.4f}, accuracy = {acc:.4f}, val_accuracy = {val_acc:.4f}')
#             history['train_loss'].append(loss.item())
#             history['train_acc'].append(acc)
#             history['val_loss'].append(loss.item())
#             history['val_acc'].append(val_acc)

model.train(train_loader, test_loader, criterion, optimizer, HYP['epochs'])

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

fig_confusion_matrix, ax = plt.subplots(1,1,figsize=(7,4))
ConfusionMatrixDisplay(confusion_matrix(test_predictions,
                                        test_labels),
                       display_labels=water_level_encoding).plot(values_format=".0f",ax=ax)

ax.set_xlabel("True Label")
ax.set_ylabel("Predicted Label")
ax.grid(False)
# plt.show()

# plot the training and validation loss
fig_loss, ax = plt.subplots(1,1,figsize=(7,4))
ax.plot(history['train_loss'], label='train')
ax.plot(history['val_loss'], label='validation')
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
# plt.show()

fig_confusion_matrix.savefig("confusion_matrix.png")
fig_loss.savefig("loss.png")

# save the model
torch.save(model.state_dict(), args['inference_model'])

# send the model to the wandb as an artifact
model_artifact = wandb.Artifact(args['inference_model'], type='model')
model_artifact.add_file(args['inference_model'])
wandb.log_artifact(model_artifact)

# send the confusion matrix and loss plot to wandb
wandb.log_artifact(wandb.Image(fig_confusion_matrix, caption="Confusion Matrix"))
wandb.log_artifact(wandb.Image(fig_loss, caption="Loss"))

# send the training and validation loss figures to wandb
wandb.log({"train_loss": wandb.Image(fig_loss, caption="Train Loss")})
wandb.log({"val_loss": wandb.Image(fig_loss, caption="Validation Loss")})

run.finish()