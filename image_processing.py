import os
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import warnings
import gdown

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

# モデルファイルの期待される最小サイズ（バイト単位）
MIN_FILE_SIZE = 100 * 1024 * 1024  # 90MB（モデルファイルの実際のサイズに合わせてください）

# Google ドライブの共有リンクまたはファイルID
file_id = '1-4t8soY8tZbM2Xjy-SnUX73krzxz0CUH'
url = f'https://drive.google.com/uc?id={file_id}'

# モデルファイルが存在しない場合、ダウンロードを実行
if not os.path.exists(weights_path) or os.path.getsize(weights_path) < MIN_FILE_SIZE:
    print("モデルファイルをダウンロードしています...")
    gdown.download(url, weights_path, quiet=False)
    print("モデルファイルのダウンロードが完了しました。")
else:
    print("モデルファイルは既に存在します。")

# モデルの定義
net = models.resnet50(weights=None)  # または`pretrained=False`を使用
num_features = net.fc.in_features
net.fc = nn.Linear(num_features, len(label_names))

# モデルのロード
try:
    # モデルの状態をロード
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    net.load_state_dict(state_dict)
    print("モデルの読み込みが完了しました。")
except Exception as e:
    print(f"モデルの読み込み中にエラーが発生しました: {e}")
    # 詳細なエラーメッセージを表示
    with open(weights_path, 'r', errors='ignore') as f:
        file_head = f.read(1024)
        print(f"ファイルの先頭部分：\n{file_head}")
    exit(1)

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