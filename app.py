from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
from PIL import Image
import io
import base64
from google.cloud import storage  # GCSからモデルをダウンロードするためのライブラリ
from image_processing import judge  # image_processing.py から judge 関数をインポート

app = Flask(__name__)

# Google Cloud Storageからモデルをダウンロードする関数
def download_model_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Google Cloud Storageからモデルをダウンロードする関数"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded model to {destination_file_name}")

# モデルのセットアップを行う関数
def setup_models():
    bucket_name = 'foodphotograph'  # Google Cloud Storageに作成したバケットの名前
    multi_task_model_blob = 'model/multi_task_model.pth'  # GCS内のmulti-taskモデルファイルのパス
    score_model_blob = 'model/rn50_photo1.pth'  # GCS内のスコアモデルファイルのパス
    multi_task_model_local = 'multi_task_model.pth'  # ローカルに保存するファイル名
    score_model_local = 'rn50_photo1.pth'  # ローカルに保存するファイル名

    # Google Cloud Storageからモデルをダウンロード
    download_model_from_gcs(bucket_name, multi_task_model_blob, multi_task_model_local)
    download_model_from_gcs(bucket_name, score_model_blob, score_model_local)
    print("Models have been downloaded and are ready for use.")

# アプリケーション起動時にモデルをセットアップ
setup_models()

# 画像を受け取って処理し、結果を返すエンドポイント
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # アップロードされた画像を取得
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        file = request.files['image']

        # 画像をPIL形式で読み込む
        img_pil = Image.open(file.stream)

        # 画像をOpenCV形式に変換
        img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # 画像を処理して結果を取得
        processed_img, results = judge(img_cv2)

        # スコアが results に含まれているか確認
        if 'score' not in results:
            return jsonify({'error': 'Score calculation failed'}), 500

        # OpenCV画像をJPEGにエンコード
        _, buffer = cv2.imencode('.jpg', processed_img)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        # 結果をJSONで返す
        return jsonify({
            'results': results,
            'image_data': jpg_as_text
        }), 200

    except Exception as e:
        print(f"Error occurred during processing: {e}")
        return jsonify({'error': f'Processing error occurred: {e}'}), 500

# メインページのルーティング
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # サーバーの実行
    app.run(debug=True)
