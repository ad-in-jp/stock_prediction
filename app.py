from flask import Flask, render_template, request, jsonify
import yfinance as yf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# 1. Colabで保存したモデルを読み込む
# 事前に my_model.save('my_model.h5') で保存しておく必要があります
model = load_model('my_model.h5')

@app.route('/')
def index():
    # 最初にサイトにアクセスした時に HTML を表示する
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # HTMLから送られてきた銘柄コード(ticker)を取得
    data = request.get_json()
    ticker = data.get('ticker', 'AAPL')
    
    try:
        # 2. データの取得
        stock_data = yf.download(ticker, period="1y")
        if stock_data.empty:
            return jsonify({'error': '銘柄が見つかりません'})

        # 3. 直近30日の画像化 (学習時と同じ処理)
        latest_30 = stock_data.tail(30)
        plt.figure(figsize=(1, 1), dpi=64)
        plt.plot(latest_30['Close'].values, color='black')
        plt.axis('off')
        plt.savefig('temp_predict.png')
        plt.close()

        # 4. AI予測の実行
        img = cv2.imread('temp_predict.png', cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        input_data = np.array(img).reshape(1, 64, 64, 1) / 255.0
        
        prediction = model.predict(input_data)
        prob = float(prediction[0][0]) # 上昇する確率
        
        # 5. 結果をブラウザに返す
        result = "上昇" if prob > 0.5 else "下落・停滞"
        confidence = prob * 100 if prob > 0.5 else (1 - prob) * 100
        
        return jsonify({
            'ticker': ticker,
            'prediction': result,
            'confidence': f"{confidence:.1f}"
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)