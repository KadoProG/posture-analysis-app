### モデルの導入

https://miyashinblog.com/openpose/#toc5

pose 以降をディレクトリ直下に貼り付け

```bash
.
└── pose
    ├── coco
    │   ├── pose_deploy_linevec.prototxt
    │   └── pose_iter_440000.caffemodel
    └── mpi
        ├── pose_deploy_linevec.prototxt
        ├── pose_deploy_linevec_faster_4_stages.prototxt
        └── pose_iter_160000.caffemodel

```

### ライブラリのインストール

```bash
pip install -r requirements.txt
```

### アプリの実行

#### 事前に画像を用意する必要があります(現在は `sample.jpg`)

```bash
python -m app
```

#### chat_gpt が作成したコードはカメラモジュールを使用してリアルタイムに映像を分析するらしいです

```bash
python chatgpt_code.py
```
