"""
Creator: Morsinaldo Medeiros
Date: 06-02-2023
Description: This script contains the model class.
"""

import numpy as np
import torch
from torchvision import models
from torchvision import transforms

class MyVisionTransformerModel():
    """ MyVisionTransformerModel class to load the model and predict the image. """
    def __init__(self, len_encoding, logger=None, pretrained=True):
        self.logger = logger
        self.model = models.vit_l_32(pretrained=pretrained)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.heads.head = torch.nn.Linear(1024, len_encoding)
        self.device = None
        self.set_device()
        
    def set_device(self):
        self.logger.info("Setting device")
        if torch.cuda.is_available():
            self.logger.info("Using GPU")
            self.device = torch.device('cuda')
        else:
            self.logger.info("Using CPU")
            self.device = torch.device('cpu')
        self.model.to(self.device)

    def load_model(self, model_path):
        self.logger.info(f"Loading model from {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def train(self, train_loader, test_loader, criterion, optimizer, total_steps, epochs=20):
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.logger.info("Training model")
        for epoch in range(epochs):
            for i, (train_images, train_labels) in enumerate(train_loader):
                train_images = train_images.to(self.device)
                train_labels = train_labels.to(self.device)
                outputs = self.model(train_images.float())
                loss = criterion(outputs, train_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    train_images.to('cpu')
                    train_labels.to('cpu')
                    predicted_outputs = self.model(train_images.float())
                    _, train_predictions = torch.max(predicted_outputs, 1)
                    n_correct = (train_predictions == train_labels).sum().item()
                    acc = n_correct/len(train_images)
                    
                if (i+1) % 4 == 0:
                    with torch.no_grad():
                        n_test_correct = 0
                        n_test = 0
                        for test_images, test_labels in test_loader:
                            test_images = test_images.to(self.device)
                            test_labels = test_labels.to(self.device)
                            test_outputs = self.model(test_images.float())
                            _, test_predictions = torch.max(test_outputs, 1)
                            test_images.to('cpu')
                            test_labels.to('cpu')
                            n_test_correct += (test_predictions == test_labels).sum().item()
                            n_test += len(test_images)
                        val_acc = n_test_correct/n_test
                    self.logger.info(f'epoch {epoch+1} / {epochs}, step {i+1} / {total_steps}, loss = {loss.item():.4f}, accuracy = {acc:.4f}, val_accuracy = {val_acc:.4f}')
                    history['train_loss'].append(loss.item())
                    history['train_acc'].append(acc)
                    history['val_loss'].append(loss.item())
                    history['val_acc'].append(val_acc)
        return history

    def predict(self, test_loader):
        self.logger.info("Predicting")
        test_predictions = []
        test_labels = []
        with torch.no_grad():
            for i, (t_images, t_labels) in enumerate(test_loader):
                t_images = t_images.to(self.device)
                t_labels = t_labels.to(self.device)
                outputs = self.model(t_images.float())
                _, predictions = torch.max(outputs, 1)
                predictions = predictions.to('cpu')
                t_labels = t_labels.to('cpu')
                test_predictions.append(predictions.numpy())
                test_labels.append(t_labels.numpy())
        test_predictions = np.hstack(test_predictions)
        test_labels = np.hstack(test_labels)
        return test_predictions, test_labels

    def predict_image(self, image):
        self.logger.info("Predicting image")
        self.model.eval()
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((224, 224)),])
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)
        with torch.no_grad():
            outputs = self.model(image.float())
            _, predictions = torch.max(outputs, 1)
            predictions = predictions.to('cpu')
        return predictions