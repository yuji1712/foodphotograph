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


# -----------------------------------------------------

# import os
# import cv2
# import torch
# from flask import Flask, request, jsonify, render_template
# from werkzeug.utils import secure_filename
# from image_processing import judge, calculate_score  # image_processing.py からインポート
# from PIL import Image
# import numpy as np

# # Flaskアプリケーションのインスタンス化
# app = Flask(__name__)

# # アップロードされる画像の保存先ディレクトリ
# UPLOAD_FOLDER = 'static/uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # 画像の拡張子許可
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# # 許可されたファイルかを確認する関数
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # ルートページの表示
# @app.route('/')
# def index():
#     return render_template('index.html')

# # 画像アップロードと処理のエンドポイント
# @app.route('/upload', methods=['POST'])
# def upload_image():
#     # 画像が正しく送信されたか確認
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image part in the request'}), 400

#     file = request.files['image']

#     # ファイル名が有効かを確認
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     # 画像ファイルが許可されたものであるかを確認
#     if file and allowed_file(file.filename):
#         # 画像を保存
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)

#         # 画像を読み込んでOpenCV形式に変換
#         img = Image.open(filepath)
#         img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

#         try:
#             # 画像処理を行う（judge関数とスコア計算関数を呼び出し）
#             processed_image, results = judge(img_cv2)
#             score = calculate_score(processed_image)

#             # 処理した画像と結果を返す
#             return jsonify({
#                 "results": results,
#                 "score": score
#             })
#         except Exception as e:
#             return jsonify({'error': str(e)}), 500

#     return jsonify({'error': 'Invalid file format'}), 400

# if __name__ == '__main__':
#     # アップロードフォルダが存在しない場合は作成
#     if not os.path.exists(app.config['UPLOAD_FOLDER']):
#         os.makedirs(app.config['UPLOAD_FOLDER'])

#     # Flaskアプリケーションの実行
#     app.run(debug=True)
