# ５章 推薦アルゴリズムの詳細
書籍の第５章では、各種推薦アルゴリズムを紹介しました。
そこで紹介した各種アルゴリズムをjupyter notebookで実行する方法を説明します。

ご自身のPCで実行する方法と、Google Colabのnotebookで実行する方法があります。
環境構築をせずに手っ取り早くアルゴリズムをお試ししたい方は、Google Colabをご利用ください。
ご自身のPCで、環境構築する場合は、poetryを利用して構築する方法と、Dockerを利用して構築する方法の２通りを紹介しますので、ご参考ください。


## フォルダ構成
```
chapter5
├─── src
│   ├── base_recommender.py
│   ├── association.py
│   └── etc...
├─── notebook
│   ├── Association.ipynb
│   ├── BPR.ipynb
│   └── etc...
├─── util
│   ├── __init__.py
│   ├── data_loader.py
│   ├── metric_calculator.py
│   └── models.py
└─── colab
    ├── Association.ipynb
    ├── BPR.ipynb
    └── etc...
```
`chapter5`フォルダ配下には、`src`, `notebook`, `util`,`colab`の４つのフォルダがあります。

`src`, `notebook`, `util`は、統一モジュールを利用して、アルゴリズムを実行するコードが格納されています。`src`には、`base_recommender.py`のクラス設計に沿って、各種アルゴリズムが実装されています。`util`には、データの読み込みや評価の統一モジュールが実装されています。`notebook`には、`src`で実装したアルゴリズムを利用して、アルゴリズムの動作確認をするコードが記述されています。一部のnotebookには、アルゴリズムのパラメーターを変えて、予測精度の変化を確認しているものもありますので、ご参考ください。

`colab`には、Google Colabで動くnotebookが格納されています。
書籍の中では、データの読み込みや評価をするのに統一モジュールを利用したり、各種推薦アルゴリズムをクラス設計したりしていました。統一フォーマットがあることで、推薦アルゴリズムをシステムに組み込むときに便利です。しかし、初学者にとっては、とっつきにくいところもあります。
そこで、統一モジュールを利用せずに、１枚のnotebookにデータの読み込みとアルゴリズムのコードを記述し、その中だけで完結するnotebookを用意しました。notebook内では、各種ステップごとに、データを表示するようにしてますので、推薦アルゴリズムの理解の手助けになるかと思います。

## Google Colabで動かす
`chapter5/colab`というフォルダに入っています。
github上から、notebookを開くと、Colabへのリンクが表示されるので、それをクリックするとcolab上で実行することができます。（Colabではなく、ご自身のPCで実行も可能です。その際には、ご自身のPCで環境構築する必要があります。）
まずは、`Association.ipynb`や`Item2vec.ipynb`などを見ていただくと、レコメンドアルゴリズムのイメージが掴みやすいです。

## ご自身のPCで環境構築
### データのダウンロード
アルゴリズムの学習に使用するMovielensのデータを`https://files.grouplens.org/datasets/movielens/ml-10m.zip`から手動でダウンロードして、解凍したものを`chapter5/data`に格納してください。
または、下記のコードでダウンロードして解凍ください。
```
# MovieLensのデータセットをdataディレクトリにダウンロードして展開
$ wget -nc --no-check-certificate https://files.grouplens.org/datasets/movielens/ml-10m.zip -P chapter5/data
$ unzip -n chapter5/data/ml-10m.zip -d chapter5/data
```

### Poetryを利用した環境構築
pythonのバージョンは、python3.7.8を利用します。
python3.7.8のインストールには、[pyenv](https://github.com/pyenv/pyenv)などのバージョン管理ツールをご利用ください。
また、今回、[poetry](https://python-poetry.org/)をパッケージ管理ツールとして利用しますので、インストールください。
ご参考に、macOS Montereyに、`pyenv`と`poetry`をインストールした手順を示します。windowsやlinuxの方は、ご自身のOSのコマンドに置き換えいただくか、Dockerを利用した環境構築をご参考ください。

#### python3.7.8のインストール
pyenvをインストールします。
```
$ brew install pyenv
```

pyenvの設定を普段使用しているシェルの設定に書き込みます。
`bash`をお使いの方は、`~/.bash_profile`に、`zsh`をお使いの方は、`~/.zshrc`に以下のコードを追加します。
```
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
```

シェルの設定を読み込みます。
```
# bashの場合
$ source ~/.bash_profile

# zshの場合
$ source ~/.zshrc
```
python3.7.8をインストールします。
```
$ pyenv install 3.7.8
```
python3.7.8をローカルで使用するように設定します。

```
$ pyenv local 3.7.8
```

#### poetryとライブラリのインストール
poetryをインストールします。
```
$ brew install poetry
```

ライブラリをインストールします。
```
$ poetry install
```
※ macでxlearnライブラリのインストールに失敗し、`Exception: Please install CMake first`と表示された場合は、`xcode-select --install`や`cmake`のインストールをしてから、再度実行します。
```
$ brew install cmake
```

#### jupyter notebookの起動
jupyter notebookの起動は以下になります。
```
$ poetry run jupyter notebook 
$ poetry run jupyter lab # jupyter labが好みの方はこちらを
```

### Dockerを利用した環境構築
dockerを利用したjupyter notebookの起動は以下になります。
(※ macOS: Monterey, docker: 1.29.2, docker-compose: 1.29.2で動作を確認しています。)
```
$ docker-compose up -d
$ docker-compose exec app poetry run jupyter notebook --allow-root --ip=0.0.0.0
$ docker-compose exec app poetry run jupyter lab --allow-root --ip=0.0.0.0 # jupyter labが好みの方はこちらを
```
dockerのプロセスは次のコマンドで停止できます。
```
$ docker-compose stop
```
