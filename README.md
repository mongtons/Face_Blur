# Face_Blur
sklearn의 KNeighborClassifier를 이용한 사람 얼굴 분류 및 분류된 얼굴 블러처리

## 개요
&nbsp;영상처리프로그래밍의 핵심인 머신러닝을 학습하고 직접 부가적 요소인 블러처리나 트랙바를 지정하여 스스로 학습할 수 있는 계기를 만들자고파 프로그램을 제작하는 계기가 되었습니다.

## Python OpenCV와 haarcascade를 이용한 얼굴분류 프로그램 설명
&nbsp;하르 분류 파일 'haarcascade_frontalface_alt2.xml', 'haarcascade_profileface.xml'을 이용하여 전자인 frontalface 분류 xml파일은 사람의 정면 얼굴을 검출하고 후자파일인 profileface은 얼굴의 옆면을 검출하여 분류한 학습된 파일이다. Python의 OpenCV를 이용하여 유사 하르 파일로 특징을 잡아준다. 그 특징을 이용하여 프로그램은 학습을 하여 사람의 얼굴을 분류하게 된다. 기초적인 머신러닝 프로그램이다.   

&nbsp;위의 두 xml파일을 이용하여 영상 속 사람들의 얼굴을 검출하여 흐릿하게 블러처리하는 프로그램이다.   


### Haar cascade file Link
* https://github.com/opencv/opencv/tree/master/data/haarcascades

## 개발 환경
* Jupyter Lab
* OpenCV `pip3 install opencv-python`
> OpenCV version 4.5.5
* numpy
> NumPy version 1.21.5
* skLearn

## 라이브러리
```python
import cv2
import numpy as np
from sklearn.neighbors import KNeighborClassifier
```

## 실행 요약


```python
classifier=cv2.CascadeClassifier(filename)
result=classifier.dectMultiScale(image, scaleFactor=None, minNeighbors=None, minSize=None, maxSize=None)
```

