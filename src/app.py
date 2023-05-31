# Web動作用 View file

# 必要なモジュールのインポート
import torch
from animal import transform, Net
from flask import Flask, request, render_template, redirect
import io
from PIL import Image
import base64

# [推論処理]
# 学習済みモデルを使用して推論する
def predict(img):
    # ネットワークの準備
    net = Net().cpu().eval()
    # 学習済みモデル（resnet34）を読み込み
    net.load_state_dict(torch.load("./dog-cat.pt", map_location=torch.device("cpu")))
    # データ前処理
    img = transform(img)
    img = img.unsqueeze(0) # 推論できるように1次元増やす
    # 推論
    y = torch.argmax(net(img), dim=1).cpu().detach().numpy()
    return y

# [推論結果のラベル付け]
# 推論結果から「犬or猫」を返す
def getName(label):
    if label == 0:
        return "猫"
    elif label == 1:
        return "犬"

# [Web APP制御]
# FLASKインスタンス作成
app = Flask(__name__)

# アップロード可能な拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg'])

# 拡張子が適切かどうかのチェック
def allwed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Webページ上で何かアクションが発生した時の挙動
@app.route('/', methods = ['GET', 'POST'])
def predicts():

    # Request : POSTの場合
    if request.method == 'POST':
        # ファイルが存在しない場合：Request元のURLにRedirect
        if 'filename' not in request.files:
            return redirect(request.url)
        # ファイルが存在する場合：データの取り出し
        file = request.files['filename']
        # ファイル拡張子が問題ないかチェック→問題なければ以下処理実行
        if file and allwed_file(file.filename):

            #　画像ファイルに対する処理
            #　画像書き込み用バッファを確保
            buf = io.BytesIO()
            image = Image.open(file).convert('RGB')
            #　画像データをバッファに書き込む
            image.save(buf, 'png')
            #　バイナリデータをbase64でエンコードしてutf-8でデコード
            base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            #　HTML側のsrcの記述に合わせるために付帯情報付与する
            base64_data = 'data:image/png;base64,{}'.format(base64_str)

            # 入力された画像に対して推論、推論結果と画像データをresult.htmlに渡す
            pred = predict(image)
            animalName_ = getName(pred)
            return render_template('result.html', animalName=animalName_, image=base64_data)
        return redirect(request.url)

    # Request：GETの場合
    elif request.method == 'GET':
        return render_template('index.html')

# アプリケーションの実行定義
if __name__ == '__main__':
    app.run(debug=True)