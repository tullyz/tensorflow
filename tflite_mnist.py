from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
start = time.time()

import argparse
import numpy as np

from PIL import Image

#from tensorflow.lite.python.interpreter import Interpreter
from tflite_runtime.interpreter import Interpreter




# ラベルの読み込み
def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

# メインの実行
if __name__ == '__main__':
    # 引数のパース
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--image',
        default='img/test02854.png',
        help='image to be classified')
    parser.add_argument(
        '-m',
        '--model_file',
        default='tf_lite_model/mnist.tflite',
        help='.tflite model to be executed')
    parser.add_argument(
        '-l',
        '--label_file',
        default='tf_lite_model/labels.txt',
        help='name of file containing labels')
    parser.add_argument(
        '--input_mean',
        default=127.5, type=float,
        help='input_mean')
    parser.add_argument(
        '--input_std',
        default=127.5, type=float,
        help='input standard deviation')
    args = parser.parse_args()

    lap = time.time() - start
    print('{:2.2f}'.format(lap), "sec for preparation")

    # インタプリタの生成
    interpreter = Interpreter(model_path=args.model_file)
    interpreter.allocate_tensors()

    # 入力情報と出力情報の取得
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 入力テンソル種別の取得(Floatingモデルかどうか)
    floating_model = input_details[0]['dtype'] == np.float32

    # 幅と高さの取得(NxHxWxC, H:1, W:2)
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # 入力画像のリサイズ
    img = Image.open(args.image).resize((width, height))

    # 入力データの生成
    input_data = np.expand_dims(img, axis=0)

    # Floatingモデルのデータ変換
    if floating_model:
        input_data = (np.float32(input_data) - args.input_mean) / args.input_std
 
    # 入力をインタプリタに指定
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # 推論の実行
    interpreter.invoke()

    # 出力の取得
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)


    # 出力の上位5件の表示
    top_k = results.argsort()[-1:][::-1]
    labels = load_labels(args.label_file)
    for i in top_k:
        if floating_model:
            #print('{:2.1f}: {}'.format(float(results[i]), labels[i]))
            print(labels[i])
        else:
            #print('{:2.1f}: {}'.format(float(results[i] / 255.0), labels[i]))
            print(labels[i])

    print('{:2.2f}'.format(time.time() - start - lap), "sec for inference")


