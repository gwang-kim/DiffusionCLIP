import torch
from torch import nn
from configs.paths_config import MODEL_PATHS
from models.insight_face.model_irse import Backbone, MobileFaceNet


class IDLoss(nn.Module):
    def __init__(self, use_mobile_id=False):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(MODEL_PATHS['ir_se50']))

        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, x, x_hat):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        x_feats = x_feats.detach()

        x_hat_feats = self.extract_feats(x_hat)
        losses = []
        for i in range(n_samples):
            loss_sample = 1 - x_hat_feats[i].dot(x_feats[i])
            losses.append(loss_sample.unsqueeze(0))

        losses = torch.cat(losses, dim=0)
        return losses
