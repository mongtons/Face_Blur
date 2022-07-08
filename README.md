# Face_Blur
sklearn의 KNeighborClassifier를 이용한 사람 얼굴 분류 및 분류된 얼굴 블러처리

<h3>Python OpenCV와 haarcascade를 이용한 얼굴분류 프로그램</h3>
하르 분류 파일 'haarcascade_frontalface_alt2.xml', 'haarcascade_profileface.xml'을 이용하여 전자인 frontalface 분류 xml파일은 사람의 정면 얼굴을 검출하고 후자파일인 profileface은 얼굴의 옆면을 검출하여 분류한 학습된 파일이다.  




`result=cv2.CascadeClassifier(filename).dectMultiScale(image, scaleFactor=None, minNeighbors=None, minSize=None, maxSize=None`
