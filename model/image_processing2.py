import os
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np

# 警告を抑制（必要に応じて）
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# テスト画像の前処理 (ResNetに合わせた前処理)
test_preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# NN準備
# モデルのパスとラベル名
weights = '/Users/imurayuuji/geeksalon/GSDC/static/rn50_photo1.pth'
label_names = ['1_good', '2_notgood']

# モデルのロード
net = models.resnet50(weights=None)  # 修正点: 'pretrained' ではなく 'weights=None' を使用
num_features = net.fc.in_features
net.fc = nn.Linear(num_features, len(label_names))

# 学習したモデルのロード
try:
    # PyTorchバージョンが対応している場合
    state_dict = torch.load(weights, map_location=torch.device('cpu'), weights_only=True)
except TypeError:
    # 'weights_only' がサポートされていない場合
    state_dict = torch.load(weights, map_location=torch.device('cpu'))

net.load_state_dict(state_dict)
net.eval()

# GPUが使えれば使う
use_gpu = torch.cuda.is_available()
if use_gpu:
    print('GPU is available')
    net = net.cuda()
else:
    print('GPU is not available, using CPU.')

def judge(img_cv2):
    # OpenCVの画像をRGBに変換し、PIL Imageに変換
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # 前処理を適用
    img_preprocessed = test_preprocess(img_pil)
    img_preprocessed = img_preprocessed.unsqueeze_(0)  # バッチ次元の追加

    if use_gpu:
        img_preprocessed = img_preprocessed.cuda()

    # モデルに画像を入力して予測を実行
    with torch.no_grad():
        outputs = net(img_preprocessed)
        softmax = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()

    # ソフトマックス後のクラス確率を取得
    good_prob = softmax[0] * 100
    notgood_prob = softmax[1] * 100

    # 結果を画像に描画
    result_text = f"Good: {good_prob:.2f}%, Not Good: {notgood_prob:.2f}%"
    cv2.putText(img_cv2, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return img_cv2



# --------------------------------------------------------------------------------------------------------------------

import os
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np

# テスト画像の前処理 (ResNetに合わせた前処理)
test_preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# NN準備
# モデルのパスとラベル名
weights = '/Users/imurayuuji/geeksalon/GSDC/static/rn50_photo1.pth'
label_names = ['1_good', '2_notgood']

# モデルのロード
net = models.resnet50(pretrained=False)
num_features = net.fc.in_features
net.fc = nn.Linear(num_features, len(label_names))
net.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
net.eval()

# GPUが使えれば使う
use_gpu = torch.cuda.is_available()
if use_gpu:
    print('GPU is available')
    net = net.cuda()
else:
    print('GPU is not available, using CPU.')

def judge(img_cv2):
    # OpenCVの画像をRGBに変換し、PIL Imageに変換
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # 前処理を適用
    img_preprocessed = test_preprocess(img_pil)
    img_preprocessed = img_preprocessed.unsqueeze_(0)  # バッチ次元の追加

    if use_gpu:
        img_preprocessed = img_preprocessed.cuda()

    # モデルに画像を入力して予測を実行
    with torch.no_grad():
        outputs = net(img_preprocessed)
        softmax = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()

    # ソフトマックス後のクラス確率を取得
    good_prob = softmax[0] * 100
    notgood_prob = softmax[1] * 100

    # 結果を画像に描画
    result_text = f"Good: {good_prob:.2f}%, Not Good: {notgood_prob:.2f}%"
    cv2.putText(img_cv2, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return img_cv2


















# -----------------------------------------------------------------------------------------------------------------------


import os
import glob
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

def judge():
    # ローカル実行用のモデルのパスとテストデータのパス
    weights = '/Users/imurayuuji/geeksalon/GSDC/static/rn50_photo1.pth'  # ローカルに保存されたモデルのパス
    test_dir = '/Users/imurayuuji/Desktop/IMG_0904.jpeg'  # ローカルに保存されたテストデータのディレクトリ
    label_names = ['1_good', '2_notgood']

    # テストディレクトリ内の画像ファイルを取得 (jpg ファイルのみ)
    files = glob.glob(os.path.join(test_dir, '*.jpg'))

    # NN準備
    net = models.resnet50(pretrained=False)
    num_features = net.fc.in_features
    net.fc = nn.Linear(num_features, len(label_names))

    # 学習したモデルのロード
    net.load_state_dict(torch.load(weights))
    net.eval()

    # # GPUが使えれば使う
    # use_gpu = torch.cuda.is_available()
    # if use_gpu:
    #     print('GPU is available')
    #     net = net.cuda()
    # else:
    #     print('GPU is not available, using CPU.')

    # テスト画像の枚数を出力
    print('N =', len(files))

    # 各画像に対して推論を実行
    for filename in files:
        img = Image.open(filename).convert('RGB')  # 画像をRGBに変換して読み込み
        img_preprocessed = test_preprocess(img)  # 前処理
        img_preprocessed = img_preprocessed.unsqueeze_(0)  # バッチ次元の追加

        if use_gpu:
            img_preprocessed = img_preprocessed.cuda()

        # モデルに画像を入力して予測を実行
        outputs = net(img_preprocessed).cpu()
        softmax = torch.nn.functional.softmax(outputs.detach(), dim=1)[0].numpy()

        # ソフトマックス後のクラス0 (1_good) の確率を出力
        print(f"{filename}\t{softmax[0] * 100:.2f}%")

# テスト画像の前処理 (ResNetに合わせた前処理)
test_preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

####
if __name__ == '__main__':
    judge()

