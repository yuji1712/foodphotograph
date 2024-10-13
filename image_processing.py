import os
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import warnings
import requests  # 追加

# 警告を抑制
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ラベル名の定義
label_names = ['1_good', '2_notgood']

# テスト画像の前処理
test_preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 現在のファイルのディレクトリを取得
current_dir = os.path.dirname(os.path.abspath(__file__))

# モデルファイルへのパスを構築
weights_path = os.path.join(current_dir, 'static', 'rn50_photo1.pth')

# モデルファイルのダウンロード URL を設定
model_url = 'https://drive.google.com/file/d/1-4t8soY8tZbM2Xjy-SnUX73krzxz0CUH/view?usp=sharing'

# モデルファイルが存在しない場合はダウンロード
if not os.path.exists(weights_path):
    print("モデルファイルをダウンロードしています...")
    response = requests.get(model_url)
    if response.status_code == 200:
        with open(weights_path, 'wb') as f:
            f.write(response.content)
        print("モデルファイルのダウンロードが完了しました。")
    else:
        print(f"モデルファイルのダウンロードに失敗しました。ステータスコード: {response.status_code}")
        exit(1)
else:
    print("モデルファイルは既に存在します。")

# ファイルサイズを確認
file_size = os.path.getsize(weights_path)
print(f"モデルファイルのサイズ: {file_size} バイト")

# モデルのロード
net = models.resnet50(weights=None)  # または `pretrained=False` を使用
num_features = net.fc.in_features
net.fc = nn.Linear(num_features, len(label_names))

# 学習したモデルのロード
try:
    # モデルの状態をロード
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    net.load_state_dict(state_dict)
except Exception as e:
    print(f"モデルの読み込み中にエラーが発生しました: {e}")
    exit(1)

net.eval()

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)

def judge(img_cv2):
    # OpenCVの画像をRGBに変換し、PIL Imageに変換
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # 前処理を適用
    img_preprocessed = test_preprocess(img_pil)
    img_preprocessed = img_preprocessed.unsqueeze_(0)  # バッチ次元の追加
    img_preprocessed = img_preprocessed.to(device)

    # モデルに画像を入力して予測を実行
    with torch.no_grad():
        outputs = net(img_preprocessed)
        print(f"Model outputs: {outputs}")  # デバッグ用
        softmax = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
        print(f"Softmax outputs: {softmax}")  # デバッグ用

    # クラス確率を取得
    good_prob = softmax[0] * 100
    notgood_prob = softmax[1] * 100

    # 結果を表示
    print(f"Good: {good_prob:.2f}%, Not Good: {notgood_prob:.2f}%")

    # 画像に結果を描画
    result_text = f"Good: {good_prob:.2f}%, Not Good: {notgood_prob:.2f}%"
    cv2.putText(img_cv2, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 処理した画像とスコアを返す
    return img_cv2, good_prob, notgood_prob