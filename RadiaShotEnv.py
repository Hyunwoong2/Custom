#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math

sx = math.cos( math.radians(25)) #25도의  x스피드
sy = math.sin( math.radians(25)) #25도의 y 스피드
print(sx,sy)


# In[2]:


sx *= 3
sy *= 3
print(sx,sy)


# In[3]:


# 현재 Env의 포물선 모양
import math
import numpy as np
import matplotlib.pyplot as plt
# 포물선을 따라 이동하는 물체의 궤적 구하기
# y = x^2
# y = ax^2 + bx + c
# x의 값 구하기

x = np.arange(0,100)
a=1/45      # 상하 선 길이 조절
b=3      # 좌우 선 길이 조절
c=0   # 상하Y 값 조절

#y = a*x**2 + b*x + c   # 아래로 굽어지는 포물선
y = -(1/25)*(x-50)**2 +100 #위로 굽어지는 포물선
#-a(x-5)^2 이라면 형태를 유지한채로 우측으로 5이동하게 된다
plt.plot(x,y)

print(f'x가 40일때 y값 {-a*40**2+100}')


# In[105]:


#Env 안의 구성 요소 (Target,Bullet)
import random
import numpy as np
import math
from gym import Env
from gym.spaces import Discrete, Box

class Target:
    def __init__(self):
        self.x = 0                               # 왼쪽에서 출발해서 포물선을 그리는 Target x 좌표
        self.y = 0                               # y는 0 바닥에서 시작한다
        self.dia = 10                            # Target의 크기
    def move(self):
        self.x += 1                              # x 는 1씩 늘어나며 y는 2차함수를 통한 포물선을 그리게 된다,
        self.y = -(1/25)*(self.x-50)**2 + 100   # a에 -를 붙이면 음수영역에서 시작하며 위로 그리는 포물선이 나타나며 양수 영역으로
                                                 # x에 -50을 주어서 x를 -100에서 시작하는데 c가 100이기 때문에 (100~100) 공간이다
    def get_state(self):                         # 만들기 위해서는 c에 y좌표가 0에서 시작할 수 있도록 값을 주면 된다.
        return (self.x,self.y)                   # Target의 현재 x좌표와 y좌표를 나중에 총알에 저장하기 위한 함수
    def printState(self):
        print('Target x=',self.x,'y=',self.y)     # 현재의 x좌표와 y좌표를 출력하기 위한 함수
    def reset(self):                             # 영역을 벗어났을때
        self.x = 0                               # x와 y의 좌표를 다시 0 0으로 되돌린다
        self.y = 0
        
class Bullet:
    def __init__(self,rad):                     # 총을 쏘는 agent는 가운데에 있기 때문에 x와  y는 0에 있다 (-50 ~ 50)
        self.x = 50
        self.y = 0.0
        self.rad = rad                          # 나중에 표적을 맞췄을때 각도(radian)를 알기위해서
        self.sx = math.cos(rad)*2               # x좌표의 속도는 현재 radian 각에 cos()*2 이다
        self.sy = math.sin(rad)*2             
        self.dia = 10
        self.target_state = (0,0)
    def move(self):
        self.x += self.sx
        self.y += self.sy
    def printState(self):
        print('Bullet x=',self.x, 'y=',self.y)
    def get_rad(self):
        return self.rad 
    def set_t_state(self,state):
        self.target_state = state
    def get_t_state(self):
        return self.target_state


# In[106]:


#Env
class Shot3Env(Env):
        
    def __init__(self):
        self.action_space = Discrete(2) # (0 1 2) stay,bang
        self.observation_space = Box(low=0,high=100,shape=(2,),dtype=int) # 0 ~100 사이의 상태값     
        self.target = Target()                         # 움직이는 Target 변수
        self.bullets = []
        self.state = (0,0)                             # 현재 상태
        self.episodes = 100                            # Episode 횟수
        self.hit = 0                                   # 맞춘 횟수
        self.memory = []                               # 맞췄을때의 상태정보를 저장하기위한 리스트
        self.bangCnt = 0                               # 총 발사횟수
        self.radm = []                                 # 맞췄을때의 rad각을 저장하기 위한 리스트
        
    def step(self, action, rad):                       # action과 rad를 같이받아서 발사의 조건을 정한다
        self.episodes -= 1                             # 한 episode에 60번 step인데
        self.target.move()                             # 한 step마다 target 이 이동함
            
        if action==0:
            pass                                      # 액션에 1이 들어와야 총이발사 됨으로 0일땐 아무것도 안함
        elif action==1: 
            self.bang(rad)
            self.bangCnt +=1
        for b in self.bullets :
            b.move()
            if self.boom(b):                          # a^2 = b^2+c^2
                self.hit +=1
                self.memory.append(b.get_t_state())
                self.radm.append(b.get_rad())
                self.bullets.remove(b)                # 총알을 맞췄을때 삭제하지 않으면 target에 계속 hit를 하여 데이터에 이상값이 생김                                                 
            if b.x<=0:                               # 왼쪽 경계를 벗어난 경우 제거
                try:
                    self.bullets.remove(b)
                except:
                    pass
                
        #Check if episode is done   
        if self.episodes<=0:
            done = True
        else:
            done = False
        info = {}
        self.state = (self.target.x,self.target.y)        
        return self.state,0, done, info
    
    def bang(self,rad):                                   # 총알 발사 함수
        # 외부(무작위 또는 신경망)에서 전달된 각도로 총알을 발사한다.
        b = Bullet(rad)                                   # 총알 만들기
        b.set_t_state(self.target.get_state())            # 만든순간에 target의 상태정보를 기억
        self.bullets.append(b)                            # list에 appned로 총알을 넣어줌
        
    def boom(self,b) :
        if math.pow(((self.target.dia/2)+(b.dia/2)),2) >= (
        math.pow((self.target.x-b.x),2)
        + math.pow((self.target.y-b.y),2)) :
            return True
        else : return False
        
    def render(self):
        pass
    
    def reset(self):                                       # 한 에피소드가 끝나면 모든걸 다시 처음값으로 초기화 해줌
        self.episodes = 100
        self.bullets.clear()
        self.state = (0,0)
        self.target.reset()                  # x와 y를 동시에 0 으로 만드는 target.reset() 함수
        self.hit=0
        self.bangCnt=0
        return self.state                    #reset은 반드시 상태정보를 return 해줘야 함


# In[138]:


# Test Agent
import random
import time

env = Shot3Env()
episodes = 5000
states = []
for e in range(episodes):
    env.reset()
    done=False
    while not done :
        action = env.action_space.sample()
        rad = math.radians(np.random.randint(0,180))
        state,reward,done,info = env.step(action,rad)
        states.append(state)
        if done:
            print(f'Episode{e+1} bangs:{env.bangCnt} hits:{env.hit}')   # 한 Episode 마다 맞춘 횟수를 알려줌


# In[124]:


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

def action_model() :
    model = Sequential()
    model.add(Dense(24,activation='relu',input_shape = (2,))) 
    #들어오는 값이 음수가 있으면 안되기 때문에 0이하를 버리는 relu를 사용함
    model.add(Dense(24,activation='relu'))
    model.add(Dense(24,activation='relu'))
    model.add(Dense(2,activation='softmax'))
    #softmax를 사용하여 출력할때 1쪽에 더 가까운 값을 사용함
    model.compile(
        loss="categorical_crossentropy",
        #범주형 데이터가 나올때 손실
        optimizer="adam",
        #대부분의 optimizer
        metrics=["accuracy"])
        #metrics= 정확도(accuracy)를 사용하겠다
    return model

def rad_model() :
    model = Sequential()
    model.add(Dense(24,activation='relu',input_shape = (2,))) 
    #들어오는 값이 음수가 있으면 안되기 때문에 0이하를 버리는 relu를 사용함
    model.add(Dense(24,activation='relu'))
    model.add(Dense(24,activation='relu'))
    model.add(Dense(1,activation='linear'))
    #각도는 한개의 값이기 때문에 linear로 1개의 output을 받는다
    model.compile(
        loss="mse",
        #loss는 mse를 사용한다.
        optimizer="adam",
        #대부분의 optimizer
        metrics=["mae"])
        #metrics= 정확도(mean absolute error)를 사용하여 error의 절대값을 표시하겠다
    return model


# In[125]:


import numpy as np

a_labels = []                          # 원핫 인코딩을 넣어줄 리스트 생성

for s in states :                   # 무작위로 뽑은 액션의 상태정보
    ohe = np.zeros(2)               # 원핫 인코딩을 위한 생성
    if s in env.memory :            # 상태정보가 정답과 맞은 경우라면
        ohe[1] = 1                  # 두번째 공간에 1로 setting
    else : ohe[0] = 1               # 첫번째 공간에 1로 setting
    a_labels.append(ohe)              # 원핫인코딩된 데이터를 리스트에 추가
trainX = np.array(states)           #신경망에 문제로 들어갈 데이터
trainY = np.array(a_labels)           #신경망에 답으로 들어갈 데이터
print( type(trainX),trainX.shape) 
print( type(trainY),trainY.shape)


# In[126]:


model_a= action_model()
model_a.summary()


# In[127]:


import numpy as np

b_labels = []
for s in states :                   # 무작위로 뽑은 액션의 상태정보
    ohe = np.zeros(1)               # 원핫 인코딩을 위한 생성
    if s in env.memory :            # 상태정보가 정답과 맞은 경우라면
        b_labels.append(env.radm)                  # 두번째 공간에 1로 setting
    else : b_labels.append(0)
trainXX = np.array(env.memory)           #신경망에 문제로 들어갈 데이터
trainYY = np.array(env.radm).reshape(-1,1)           #신경망에 답으로 들어갈 데이터
print( type(trainXX),trainXX.shape) 
print( type(trainYY),trainYY.shape)


# In[128]:


print(len(trainYY))


# In[129]:


cnt = 0
test = []
for s in states:              #상태정보를 하나씩 빼서
    if s in env.memory :    # 그상태정보가 정답(env.memory)과 일치한다면
        test.append(s)
        cnt+=1 
print(cnt)
print(len(test))
print(len(set(test)))
env.radm


# In[130]:


model_b= rad_model()
model_b.summary()


# In[131]:


model_a.fit(trainX,trainY,epochs=100)


# In[133]:


model_a.save('Shot3Env_DQN_h5',overwrite=True)
model_b.fit(trainXX,trainYY,epochs=200)
model_b.save('Shot3Env_rad_h5',overwrite=True)


# In[139]:


import random
import time

env = Shot3Env()
episodes = 100

state = (0,0)

for e in range(episodes):
    env.reset()
    done=False
    while not done :
        action = model_a.predict(np.array(state).reshape(-1,2))
        #튜플인 state를 np.array로 변환후 reshape()를 사용해 2차원배열로 변환
        action = np.argmax(action[0].reshape(-1,2))
        rad = model_b.predict(np.array(state).reshape(-1,2))
        state,reward,done,info = env.step(action,rad)
            #states.append(state)
        if done:
            print(f'Episode{e+1} bangs:{env.bangCnt} hits:{env.hit}')   # 한 Episode 마다 맞춘 횟수를 알려줌
print('테스트 종료')      


# In[137]:


rad = model_b.predict(np.array((32,50)).reshape(-1,2))
print(rad)


# In[30]:




