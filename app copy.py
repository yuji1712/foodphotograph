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

    # JSONデータとして画像とスコアを返す
    return jsonify({
        'image_data': img_base64,
        'good_prob': float(good_prob),
        'notgood_prob': float(notgood_prob)
    })

if __name__ == '__main__':
    app.run(debug=True)


# -------------------------------------------------------------------

from flask import Flask, request, render_template
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

    # テンプレートに画像データとスコアを渡してレンダリング
    return render_template('result.html', image_data=img_base64, good_prob=good_prob, notgood_prob=notgood_prob)

if __name__ == '__main__':
    app.run(debug=True)



# --------------------------------------------------------------------------------------------------------------------------------

from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
import io
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
    file_bytes = np.frombuffer(image_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 別ファイルのカスタム処理関数を呼び出し
    processed_img = judge(img)

    # 結果をバイト形式に変換して返送
    _, img_encoded = cv2.imencode('.jpg', processed_img)
    return send_file(
        io.BytesIO(img_encoded.tobytes()),
        mimetype='image/jpeg',
        as_attachment=False,
        download_name='result.jpg'
    )

if __name__ == '__main__':
    app.run(debug=True)


# -------------------------------------------------------------------------------------------------------


from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
import io


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
    file_bytes = np.frombuffer(image_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 顔認識処理
    haarcascade_path = '/Users/imurayuuji/opt/anaconda3/envs/geek/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haarcascade_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # 顔の周りに四角形を描画
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        message = "認識できました"
    else:
        message = "認識できませんでした"

    # 結果をバイト形式に変換して返送
    _, img_encoded = cv2.imencode('.jpg', img)
    return send_file(
        io.BytesIO(img_encoded.tobytes()),
        mimetype='image/jpeg',
        as_attachment=False,
        download_name='result.jpg'
    )

if __name__ == '__main__':
    app.run(debug=True)
