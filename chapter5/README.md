# ５장 추천 알고리즘 상세

이 책의 5장에서는 각종 추천 알고리즘을 소개합니다. 5장에서 소개한 각종 알고리즘을 jupyter notebook으로 실행하는 방법을 설명합니다.

로컬 PC에서 실행하는 방법과 Google Colab의 notebook에서 실행하는 방법이 있습니다. 별도의 환경을 구축하지 않고 빠르게 알고리즘을 실행하고 싶다면 Google Colab을 사용하기 바랍니다. 로컬 PC에서 환경을 구축할 때는 `poetry`를 사용하는 방법과 도커(Docker)를 사용하는 방법을 설명했으므로 참고하기 바랍니다.

## 폴더 구성
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

`chapter5` 폴더 아래에는 `src`, `notebook`, `util`,`colab`라는 4개의 폴더가 있습니다.

`src`, `notebook`, `util`은 공통의 모듈을 사용해서 알고리즘을 실행하는 코드가 저장되어 있습니다. `src`에는 `base_recommender.py` 클래스 설계에 맞춰 각종 알고리즘이 구현되어 있습니다. `util`에는 데이터 읽기와 평가 공통 모듈이 구현되어 있습니다. `notebook`에는 `src`에 구현한 알고리즘을 사용해서, 알고리즘의 동작을 확인하는 코드가 기술되어 있습니다. 일부 notebook에는 알고리즘의 파라미터를 바꾸어, 예측 정확도의 변화를 확인할 수 있도록 되어 있으므로 참고하기 바랍니다.

`colab`에는 Google Colab에서 동작하는 notebook이 저장되어 있습니다. 책에서는 데이터 읽기 또는 평가를 할 때 공통 모듈을 사용하거나, 각종 추천 알고리즘을 클래스로 설계했습니다. 공통 포맷을 사용하면 추천 알고리즘을 시스템에 조합할 때 편리합니다. 하지만 초보자에게는 다소 어려운 부분도 있습니다. 그래서 공통 모듈을 사용하지 않고 하나의 notebook에 데이터 읽기와 알고리즘 코드를 기술해, 그 자체로 완결되도록 구성했습니다. notebook 안에서는 각 단계 별로 데이터를 표시하도록 했으므로, 추천 알고리즘을 이해하는 데 도움이 될 것입니다.

## Google Colab에서 동작시키기

`chapter5/colab` 폴더에 들어있습니다. github에서 notebook을 열면 Colab 링크가 표시됩니다. 해당 링크를 클릭하면 colab 상에서 실행할 수 있습니다.(Colab이 아닌 로컬 PC에서도 실행할 수 있습니다. 이 때는 로컬 PC에 환경을 구축해야 합니다.) 먼저 `Association.ipynb`, `Item2vec.ipynb` 등을 보면 추천 알고리즘에 관한 이미지를 잡기 쉬울 것입니다.

## 5장 샘플 파일 설명(장/절 순)

5장의 `colab`, `notebook` 디렉터리 안의 노트북(`.ipynb`) 파일은 다음과 같이 구성되어 있습니다.

|파일명|분석/알고리즘|관련 장/절|관련 페이지|
|:--|:--|:--|:--|
|`data_download.ipynb`|MovieLens 데이터셋 다운로드|5.2|p.103|
|`Random.ipynb`|무작위 추천<sup>Random Recommendation</sup>|5.3|p.118|
|`Popularity.ipynb`|인기도순 추천|5.4|p.120|
|`Association.ipynb`|연관 규칙(어소시에이션 분석) - Apriori 알고리즘|5.5|p.127|
|`UMCF.ipynb`|사용자-사용자 메모리 기반 방법 협조 필터링<sup>User-User Memory Based Collaborative Filtering, UMCF</sup>|5.6|p.131|
|`RF.ipynb`|회귀 모델, 랜덤 포레스트<sup>Random Forest, RF</sup>|5.7|p.137|
|`SVD.ipynb`|특잇값 분해<sup>Singular Value Decomposition, SVD</sup>|5.8.2|p.144|
|`NMF.ipynb`|비음수 행렬 분해<sup>Non-negative Matrix Factorization, NMF</sup>|5.8.3|p.148|
|`MF.ipynb`|행렬 분해<sup>Matrix Factorization, MF</sup>|5.8.4|p.150|
|`IMF.ipynb`|암묵적 행렬 분해<sup>Implicit Matrix Factorization, IMF</sup>|5.8.5|p.154|
|`BPR.ipynb`|개인화 된 랭킹 문제<sup>Bayesian Personalized Ranking, BPR</sup>|5.8.6|p.158|
|`FM.ipynb`|Factorization Machines, FM|5.8.7|p.160|
|`LDA_content.ipynb`|잠재 디리클레 할당<sup>Latent Dirichlet Allocation, LDA</sup>|5.9.1|p.165|
|`LDA_collaboration.ipynb`|LDA를 행동 데이터에 적용|5.9.3|p.169|
|`Word2vec.ipynb`|`word2vec`|5.9.5|p.172|
|`Item2vec.ipynb`|`word2vec`을 사용한 협조 필터링 추천(`item2vec`)|5.9.6|p.176|


## 로컬 PC에서 환경 구축하기

### 데이터 다운로드

알고리즘 학습에 사용하는 MovieLens 데이터를 `https://files.grouplens.org/datasets/movielens/ml-10m.zip`에서 다운로드한 뒤, 압축을 풀어서 `chapter5/data`에 저장합니다. 또는 다음 코드를 실행해서 다운로드한 뒤, 압축을 풉니다.

```
# MovieLens 데이터셋을 data 디렉터리에 다운로드한 뒤, 압축을 푼다
$ wget -nc --no-check-certificate https://files.grouplens.org/datasets/movielens/ml-10m.zip -P chapter5/data
$ unzip -n chapter5/data/ml-10m.zip -d chapter5/data
```

### Poetry를 사용해서 환경 구축하기

python 버전은 python3.7.8을 사용합니다. python3.7.8 설치에는 [pyenv](https://github.com/pyenv/pyenv) 등의 버전 관리 도구를 사용하십시오. 또한, 여기에서는 [poetry](https://python-poetry.org/)를 패키지 관리 도구로 사용하므로 설치하기 바랍니다. 참고로 MacOS Monterey에 `pyenv`와 `poetry`를 설치하는 순서를 설명합니다. Windows나 Linux의 경우에는 여러분이 사용하는 OS에 해당하는 명령어로 변경하거나, 도커를 사용한 환경 구축하기를 참조하기 바랍니다.

#### python3.7.8 설치

pyenv를 설치합니다.

```
$ brew install pyenv
```

`pyenv` 설정을 일반적으로 사용하는 셸의 설정에 추가합니다. `bash`를 사용한다면 `~/.bash_profile`, `zsh`를 사용한다면 `~/.zshrc`에 다음 코드를 추가합니다.

```
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
```

셸 설정을 읽습니다.

```
# bash의 경우
$ source ~/.bash_profile

# zsh의 경우
$ source ~/.zshrc
```

python3.7.8을 설치합니다.

```
$ pyenv install 3.7.8
```

python3.7.8을 로컬에서 사용할 수 있도록 설정합니다

```
$ pyenv local 3.7.8
```

#### poetry와 라이브러리 설치하기

poetry를 설치합니다.

```
$ brew install poetry
```

라이브러리를 설치합니다.

```
$ poetry install
```

※ mac에서 xlearn 라이브러리 설치 시 실패하면서 `Exception: Please install CMake first` 메시지가 표시될 때는, `xcode-select --install`이나 `cmake`를 설치한 뒤 재 실행합니다.

```
$ brew install cmake
```

#### jupyter notebook 기동하기

jupyter notebook을 다음과 같이 기동합니다.

```
$ poetry run jupyter notebook 
$ poetry run jupyter lab # jupyter lab을 선호한다면 이 명령어를 실행
```

### Docker를 사용해 환경 구축하기

docker를 사용한 jupyter notebook은 다음과 같이 기동합니다(※ MacOS: Monterey, docker: 1.29.2, docker-compose: 1.29.2에서 동작을 확인했습니다).

```
$ docker-compose up -d
$ docker-compose exec app poetry run jupyter notebook --allow-root --ip=0.0.0.0
$ docker-compose exec app poetry run jupyter lab --allow-root --ip=0.0.0.0 # jupyter lab을 선호한다면 이 명령어를 실행
```

docker 프로세스는 다음 명령어로 정지할 수 있습니다.

```
$ docker-compose stop
```
