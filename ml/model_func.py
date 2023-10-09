import torch
import torch.nn as nn
from torchvision.models import resnet101
import torchvision.transforms as T
import pandas as pd

df = pd.read_csv('ml/sports.csv')
list_1 = list(df['labels'].unique())
label_to_idx = {label: idx for idx, label in enumerate(list_1)}

def load_model():
    resnet_sport = resnet101()
    num_features = resnet_sport.fc.in_features
    resnet_sport.fc = nn.Linear(num_features, 100)

    resnet_sport.load_state_dict(torch.load('ml/resnet50weight.pt', map_location='cpu'))
    resnet_sport.eval()
    return resnet_sport

def transform_image(img):
    trnsfrms = T.Compose(
      [
          T.Resize((224, 224)),
          T.ToTensor(),
          T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ]
    )

    return trnsfrms(img).unsqueeze(0)

def load_classes():
    return list_1

def predict_image(model, image):
    with torch.no_grad():
      image = transform_image(image)
      output = model(image)
      _, predicted = torch.max(output, 1)
      return predicted.item()