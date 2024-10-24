# app.py

from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
from PIL import Image
import io
import base64
from image_processing import judge  # image_processing.py から judge 関数をインポート

app = Flask(__name__)

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








