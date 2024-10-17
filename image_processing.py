import os
import json
import gc
from google.oauth2 import service_account
from google.cloud import storage
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2

# デバイスの設定（GPUが利用可能ならGPUを使用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 環境変数からサービスアカウントの情報を取得
credentials_info = json.loads(os.environ.get('GOOGLE_CREDENTIALS'))
credentials = service_account.Credentials.from_service_account_info(credentials_info)

# Google Cloud Storageクライアントを初期化
storage_client = storage.Client(credentials=credentials)

# Google Cloud Storageからモデルをダウンロードする関数
def download_model_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Google Cloud Storageからモデルをダウンロードする関数"""

    # バケットオブジェクトを取得
    bucket = storage_client.bucket(bucket_name)

    # バケット内のファイル（Blob）を指定
    blob = bucket.blob(source_blob_name)

    # ファイルをローカルにダウンロード
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded model to {destination_file_name}")

# GCSからのダウンロードに関する設定
bucket_name = 'foodphotograph'  # あなたのバケット名
multi_task_model_blob = 'model/multi_task_model.pth'  # multi_task_modelのGCS内のパス
score_model_blob = 'model/rn50_photo1.pth'  # score_modelのGCS内のパス
multi_task_model_local = 'multi_task_model.pth'  # ローカルに保存するファイル名
score_model_local = 'rn50_photo1.pth'  # ローカルに保存するファイル名

# Google Cloud Storageからモデルをダウンロード
download_model_from_gcs(bucket_name, multi_task_model_blob, multi_task_model_local)
download_model_from_gcs(bucket_name, score_model_blob, score_model_local)

# 前処理の定義
test_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# クラスのマッピング（予測結果をラベルに変換）
composition_classes = {0: '俯瞰構図', 1: '対角線構図', 2: '三角構図', 3: 'その他'}
angle_classes = {0: '真上', 1: '目線'}
sizzle_classes = {0: 'なし', 1: 'あり'}
label_names = ['Good', 'Not Good']  # スコアのクラスラベル

# マルチタスクモデルの定義
class MultiTaskResNet(nn.Module):
    def __init__(self):
        super(MultiTaskResNet, self).__init__()
        # ResNet18をプリトレーニング済みの重みでロード（軽量化のため）
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        # 各要素の予測用の出力層を追加
        self.fc_composition = nn.Linear(num_features, 4)  # 構図（4クラス）
        self.fc_angle = nn.Linear(num_features, 2)        # 角度（2クラス）
        self.fc_sizzle_shiny = nn.Linear(num_features, 2) # シズル効果（光沢）
        self.fc_sizzle_motion = nn.Linear(num_features, 2) # シズル効果（動き）
        self.fc_sizzle_steam = nn.Linear(num_features, 2) # シズル効果（蒸気）

    def forward(self, images):
        x = self.resnet(images)

        # 各要素の予測結果を出力
        composition_output = self.fc_composition(x)
        angle_output = self.fc_angle(x)
        sizzle_shiny_output = self.fc_sizzle_shiny(x)
        sizzle_motion_output = self.fc_sizzle_motion(x)
        sizzle_steam_output = self.fc_sizzle_steam(x)

        return {
            'composition': composition_output,
            'angle': angle_output,
            'sizzle_shiny': sizzle_shiny_output,
            'sizzle_motion': sizzle_motion_output,
            'sizzle_steam': sizzle_steam_output
        }

# スコア用のモデルを定義（ResNet18に変更）
class ScoreResNet(nn.Module):
    def __init__(self):
        super(ScoreResNet, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 2)  # クラス数は2

    def forward(self, images):
        return self.resnet(images)

# judge関数の定義
def judge(img_cv2):
    try:
        print("Judge function called")  # デバッグ用

        # モデルのロード（マルチタスクモデル）
        net = MultiTaskResNet().to(device)
        checkpoint = torch.load(multi_task_model_local, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()

        # スコアモデルのロード
        score_net = ScoreResNet().to(device)
        checkpoint_score = torch.load(score_model_local, map_location=device)
        score_net.load_state_dict(checkpoint_score['model_state_dict'])
        score_net.eval()

        # OpenCVの画像をRGBに変換し、PIL Imageに変換
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # 前処理を適用
        img_preprocessed = test_preprocess(img_pil)
        img_preprocessed = img_preprocessed.unsqueeze_(0).to(device)  # バッチ次元を追加

        # モデルに画像を入力して予測を実行
        with torch.no_grad():
            # マルチタスクモデルによる予測
            outputs = net(img_preprocessed)
            print("Multi-task model inference completed")  # デバッグ用

            # 各要素の予測結果を取得
            composition_pred = torch.argmax(outputs['composition'], dim=1).item()
            angle_pred = torch.argmax(outputs['angle'], dim=1).item()
            sizzle_shiny_pred = torch.argmax(outputs['sizzle_shiny'], dim=1).item()
            sizzle_motion_pred = torch.argmax(outputs['sizzle_motion'], dim=1).item()
            sizzle_steam_pred = torch.argmax(outputs['sizzle_steam'], dim=1).item()

            # 結果をラベルに変換
            composition_label = composition_classes.get(composition_pred, '不明')
            angle_label = angle_classes.get(angle_pred, '不明')
            sizzle_shiny_label = sizzle_classes.get(sizzle_shiny_pred, '不明')
            sizzle_motion_label = sizzle_classes.get(sizzle_motion_pred, '不明')
            sizzle_steam_label = sizzle_classes.get(sizzle_steam_pred, '不明')

            # スコアモデルによる予測
            score_outputs = score_net(img_preprocessed)
            score_softmax = torch.nn.functional.softmax(score_outputs, dim=1)[0].cpu().numpy()
            score_percent = score_softmax[0] * 100  # クラス0（Good）の確率を取得
            score_pred = torch.argmax(score_outputs, dim=1).item()
            score_label = label_names[score_pred]

        # 結果を画像に描画（必要に応じて）
        result_text = f"構図: {composition_label}, 角度: {angle_label}"
        cv2.putText(img_cv2, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        sizzle_text = f"シズル効果(光沢): {sizzle_shiny_label}, (動き): {sizzle_motion_label}, (蒸気): {sizzle_steam_label}"
        cv2.putText(img_cv2, sizzle_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        score_text = f"スコア: {score_label} ({score_percent:.2f}%)"
        cv2.putText(img_cv2, score_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # モデルの出力を表示（デバッグ用）
        print("Model outputs:", outputs)

        # 結果を辞書としてまとめる
        results = {
            'score': f"{score_label} ({score_percent:.2f}%)",
            'composition': composition_label,
            'angle': angle_label,
            'sizzle_shiny': sizzle_shiny_label,
            'sizzle_motion': sizzle_motion_label,
            'sizzle_steam': sizzle_steam_label
        }

        # メモリ解放
        del net
        del checkpoint
        del score_net
        del checkpoint_score
        torch.cuda.empty_cache()
        gc.collect()

        # 処理した画像と結果を返す
        print("Returning results from judge function")  # デバッグ用
        return img_cv2, results
    except Exception as e:
        print(f"An error occurred in judge function: {e}")
        import traceback
        traceback.print_exc()
        raise
