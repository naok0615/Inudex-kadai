# Inudex

Robot programming 2019 task

## 概要

授業課題で作成したアプリケーション．

入力された犬または猫の画像から犬と猫の種類を識別し,画像へペイントします．

1. server.pyを実行することでwebアプリケーションが立ち上がります．

　　　$ python sever.py

2. http://0.0.0.0:8888/ にアクセスします．
3. "ファイルを選択"から画像を選択するか"ファイルを選択"へ画像をドラッグ&ドロップすることで画像を選択
4. "submit"ボタンを押すことでserver.pyを実行しているPCへ画像がアップロードされます．
5. 約1秒後にアップロードした画像へ犬または猫の種類がペイントされた画像が表示されます．

## ディレクトリの概要
inputdata/data： アップロードした画像が保存されるディレクトリ．

static/images： 犬または猫の種類がペイントされた画像が保存されるディレクトリ．

templates： webのデザイン． 

Arial Unicode.ttf： ペイントに用いられる文字フォント．

image_main.py： 犬と猫の種類識別と画像へのペイント．

kadai-weight-cpu.pth： 学習済みの重み．

server.py： サーバーの立ち上げ．

## 環境
* Python 3.7.1
* Pytorch 1.0.0
* Flask 1.0.2
* numpy 1.15.4
* opencv-python 4.1.0.25
* Pillow 5.3.0

## 例
https://user-images.githubusercontent.com/49013079/61851202-45a05e80-aef1-11e9-95d1-b51ce113a405.jpg
