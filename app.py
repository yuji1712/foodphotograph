from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
import io
import base64
from image_processing import judge

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    # アップロードされた画像を取得
    image_file = request.files['image']
    
    # OpenCVが処理できる形式に変換
    image_stream = io.BytesIO(image_file.read())
    file_bytes = np.frombuffer(image_stream.getvalue(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # カスタム処理関数を呼び出し（画像とスコアを取得）
    processed_img, good_prob, notgood_prob = judge(img)

    # 画像をJPEG形式にエンコード
    _, img_encoded = cv2.imencode('.jpg', processed_img)

    # エンコードした画像をBase64に変換
    img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
    # デバッグ用に値を表示
    print(f"Good: {good_prob}%, Not Good: {notgood_prob}%")
    print(f"Type of good_prob: {type(good_prob)}, Value: {good_prob}")
    print(f"Type of notgood_prob: {type(notgood_prob)}, Value: {notgood_prob}")
    
    # JSONデータとして画像とスコアを返す
    response_data = {
        'image_data': img_base64,
        'good_prob': float(good_prob),
        'notgood_prob': float(notgood_prob)
    }
    print(f"Response data: {response_data}")  # デバッグ用
    return jsonify(response_data)


if __name__ == '__main__':
    app.run(debug=True)


