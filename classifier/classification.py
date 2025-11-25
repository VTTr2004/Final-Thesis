import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import vgg16, VGG16_Weights

class TrafficCNN(nn.Module):
    def __init__(self, input_size=(64, 64), pretrained=True):
        super(TrafficCNN, self).__init__()

        if pretrained:
            vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        else:
            vgg = vgg16(weights=None)

        # Thay ─æß╗òi input conv ─æß║ºu th├ánh 1 channel
        vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        self.features = vgg.features
        for param in self.features.parameters():
            param.requires_grad = False

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_size)
            out = self.features(dummy_input)
            flatten_dim = out.view(1, -1).size(1)

        # Thay ─æß╗òi classifier ─æß╗â output 3 lß╗¢p
        self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(flatten_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 3)  # output 3 lß╗¢p
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class Classifier:
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TrafficCNN()
        # Tß║úi Th├┤ng Sß╗æ Model
        checkpoint = torch.load(config.model_classify_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def classify(self, imgs):
        if len(imgs) == 0:
            return []
        img_tensor = torch.stack(imgs).float()/255.0
        img_tensor = img_tensor.to(self.device)
        # Bß║»t ─Éß║ºu Ph├ón Loß║íi
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            prebs = torch.argmax(probs, dim=1)
            top_probs = torch.max(probs, dim=1).values
            result = [[pred.item(), prob.item()] for pred, prob in zip(prebs, top_probs)]
            return result
