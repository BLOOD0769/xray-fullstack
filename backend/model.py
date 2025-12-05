# backend/model.py
import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=2, pretrained=True):
    model = models.densenet121(pretrained=pretrained)
    # replace classifier
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path, num_classes=2, device='cpu'):
    model = get_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
