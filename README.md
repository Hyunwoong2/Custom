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
  발사여부를 판단하는 Model과 radian각도를 구하는 Model을 사용하여 발사각 발사여부를 지정 해야하고
  그러기 위해서는 표적에 맞췄을때의 각도와 발사여부가 필요.
</p>
<p>
  강화학습의 데이터를 추출하기 위해 무작위로 발사각과,발사여부를 지정해서 발사함.
</p>
<p>
  데이터를 추출후
</p>
