from flask import Flask, render_template, request
import tensorflow as tf
import cv2
import  numpy as np
import json

# 플라스크 시작
app = Flask(__name__)

model = tf.keras.models.load_model('model/model.h5')

path = './image/Zzim.jpg'
with open(path, 'rb') as f:
	data = f.read()

# 루트 파일
@app.route('/')
def index():
	return render_template('index.html')

# HTML 파일 추가
@app.route('/home')
def home():
	return render_template('index.html')

@app.route('/upload')
def load_file():
	return render_template('upload.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print(request.files)
    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (416, 416))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    print(predictions.shape)
    print(predictions)
    class_names = {0: 'Mandu', 1: 'KKennip', 2: 'Jabgokbab', 3: 'Jeyukbokum', 4: 'Gimchizzigae', 5: 'Samgyupsal', 6: 'Duinjangzzigae', 7: 'Gamjatang',
                   8: 'Ramyun', 9: 'Pizza', 10: 'Yangnyumchicken', 11: 'Friedchicken', 12: 'BaechuKimchi', 13: 'Kkakdugi', 14: 'Bulgogi',
                   15: 'Godeungeogui', 16: 'Zzajangmyun', 17: 'Zzambbong', 18: 'Friedegg', 19: 'Gyeranjjim'}
    results = []
    for output in predictions:
        for detection in output:
            max_class = int(np.argmax(detection[5:]))
            confidence = float(detection[5 + max_class])
            if confidence >= 0.1:
                class_name = class_names[max_class + 1]
                obj = {
                    "class": class_name,
                    "confidence": confidence,
                    "x": float(detection[0]),
                    "y": float(detection[1]),
                    "w": float(detection[2] - detection[0]),
                    "h": float(detection[3] - detection[1])
                }
                results.append(obj)
    results = sorted(results, key=lambda x: x['confidence'], reverse=True)
    return json.dumps(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True, port=80)

'''
    "x": int(detection[0]),
    "y": int(detection[1]),
    "w": int(detection[2] - detection[0]),
    "h": int(detection[3] - detection[1])
'''

