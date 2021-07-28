# 강화학습 포물선 슈팅
## PYQT5,Tensorflow 를 활용한 포물선 궤적 맞추기
<ul>
  <li>PyQt를 이용한 Window 생성 -> 시각화</li>
  <li>Tensorflow를 활용한 인공신경망을 생성 후 강화학습</li>
  <li>Model을 Window에 적용하여 강화학습 시각화</li>
</ul>

#### 주요 내용
<p>
  Pyqt5 를 이용하여 Window를 생성 Window 안에는 Target과 Bollet이 존재.
</p>
<p>
  Window는 무한루프를 사용해서 돌아가기 때문에 강화학습을 시켜주기 위해서는 Thread를 사용해야 하고<br>
  Thread를 만들어서 Window를 돌린후에 강화학습의 데이터를 생성하는 과정이 필요.
</p>
<p>
  발사여부를 판단하는 Model과 radian각도를 구하는 Model을 사용하여 발사각 발사여부를 지정 해야하고<br>
  그러기 위해서는 표적에 맞췄을때의 각도와 발사여부가 필요.
</p>
<p>
  강화학습의 데이터를 추출하기 위해 무작위로 발사각과,발사여부를 지정해서 발사함.
</p>
<p>
  데이터를 추출후 발사여부를 판단하는 Model은 OutputLayer에 softmax를 활용하여 1에 가까운값을 사용하고<br>
  발사각을 계산하는 Model은 relu와,linear을 사용해 발사각을 구함.
</p>
<p>
  학습된 모델을 Pyqt Window에 적용하여 시각화.
</p>
![image](https://user-images.githubusercontent.com/82852526/127259175-541858de-2657-4c61-be37-a7dc10e7a8d3.png)

### 파일항목
- Radians_Shooting - Windw가 없는 포물선 슈팅
- action_model.h5 - 발사여부 가중치
- rad_model.h5 - 발사각도 가중치
- Parabolic_Shooting Window를 활용한 포물선 슈팅
