# 必要なモジュールのインポート
from torchvision import transforms
import pytorch_lightning as pl
import torch.nn as nn
# 学習済みモデルはResnet34を使用する
from torchvision.models import resnet34

# データ前処理（データ拡張）
transform = transforms.Compose([
    transforms.Resize(256), # 256pxにresize
    transforms.CenterCrop(224), # 224pxにクロップ
    transforms.RandomHorizontalFlip(), # 左右反転
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # 画像の色をランダムに変更
    transforms.RandomRotation(degrees=15), # ランダムに回転
    transforms.ToTensor(), # Tensorに変換
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ノーマライズ
])

# ネットワーク構造の定義
# 畳み込み・プーリング：ResNet34を使用（ファインチューニング）
# 全結合：1000⇒2

class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.feature = resnet34(pretrained=True)
        self.fc = nn.Linear(1000, 2)

    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h