# coding=utf-8 
# 使用Python构建FNN S-T
from decimal import *
import numpy as np 
import matplotlib.pyplot as plt
from itertools import product
from itertools import cycle
# 数据归一化
def Normalize(data):
  for i in range(data.shape[0]):
    m = np.mean(data[i,:])
    mx = np.max(data[i,:])
    mn = np.min(data[i,:])
    for j in range(len(data[i,:])):
      data[i][j]=(data[i][j]-m)/(mx-mn)
    M=[m,mx,mn]
  return data,M
 
# 数据反归一化
def Normalize_deriv(data,M):
  for i in range(data.shape[0]):
    m = M[0]
    mx = M[1]
    mn = M[2]
    for j in range(len(data[i,:])):
      data[i][j]=data[i][j]*(mx-mn)+m
  return data

# 使用类 面向对象的技巧 建立FNN 
class NeuralNetwork: 
  # 构造函数 layers指的是每层内有多少个神经元 
  def __init__(self,layers): 

    self.layers=layers

    # 连接权、中心和宽度的学习效率
    self.alfa=0.01
    self.beta=0.005
    self.xite=0.005
    
    # 循环数
    self.loop=1

    self.loss=0.01

    self.I=self.layers[0]
    self.m=self.layers[1]
    self.O=self.layers[2]

    self.M=1
    ############## 沒搞懂
    for i in range(len(m)):
      self.M=self.M*m[i]
    ##############

    # 权重初始化
    self.w = np.random.uniform(0,1,(self.O,self.M))
    
    # 隶属函数参数初始化
    self.c=[]
    self.b=[]
    for i in range(self.I):
      cc=np.random.uniform(0,1,self.m[i])
      bb=np.random.uniform(0,1,self.m[i])
      # print(i,m[i])
      self.c.append(cc)
      self.b.append(bb)
    
 
  def fit(self,X,Y):
   
    I=self.I
    M=self.M
    O=self.O
    m=self.m
    #网络出差 
    yn=np.zeros((O,self.loop))
    # 实际输出
    y=np.zeros((O,self.loop))
    #误差
    e=np.zeros((O,self.loop))
    E=np.zeros((self.loop,1))

    for k in range(self.loop):

      ###### 改成外部輸入
      #任取训练数据中的一组数据
      ii = np.random.randint(X.shape[1])
      ######
      
      ###### 外部輸入 : 行轉列
      # 第一层
      x=X[:,ii].reshape(-1,1)
      y[:,k]=np.round(Y[:,ii])
      ######

      # 第二层
      u=[]
      for i in range(I):
        uu=[]
        for j in range(m[i]):
          uu.append(np.exp(-(x[i]-self.c[i][j])**2/self.b[i][j]**2))
        uu=np.asarray(uu).reshape(-1)
        u.append(np.asarray(uu))
      # u=np.asarray(u).reshape()
      u=np.asarray(u)

      # 第三层
      a=np.ones((M,1))
      dd=list(product(*u))
      for i in range(len(dd)):
        a[i]=np.min(dd[i])

      ###### 要改成分七類
      # 第四層
      a_mean=np.ones((M,1))
      add_a=np.sum(a)
      a_mean=a/add_a
      ######
      
      # 第五层，输出层
      yi = np.sum(self.w * a_mean)
      # print(yi)

      ee=[]
      for i in range(O):
        yn[i][k]=yi[i]
        ee.append((y[i][k]-yi[i])**2)
      # 误差代价
      E[k]=sum(ee)/2

      ###### 可以刪掉
      if E[k] <= self.loss:
        break
      ######


      e[:,k]=y[:,k]-yn[:,k]
      # 第五层误差
      g_5=e[:,k].reshape(-1,1)
      # 第四层误差
      g_4=np.dot(self.w.T,g_5)

      g_3=np.zeros((M,1))
      for i in range(M):
        g_3[i]=g_4[i]*(add_a-a[i])/add_a
        
      g_c=[]
      g_b=[]
      uu=0*u 
      for i in range(I):
        cc=[]
        bb=[]
        for j in range(m[i]):
          g_u=0
          for mm in range(M):
            if u[i][j]==a[mm]:
              g_u=g_u+g_3[mm]
          gg=g_u*np.exp(-(x[i]-self.c[i][j])**2/self.b[i][j]**2)
          
          cc.append(-2*gg*(x[i]-self.c[i][j])/self.b[i][j]**2)
          
          bb.append(-2*gg*(x[i]-self.c[i][j])**2/self.b[i][j]**3)

        g_c.append(cc)
        g_b.append(bb)

      # print(g_c)

      self.w=self.w+self.alfa*(np.dot(g_5,a_mean.T))

      for i in range(I):
        for j in range(m[i]):
          # print(self.c[i][j],g_c[i][j])
          self.c[i][j]=self.c[i][j]-self.beta*g_c[i][j]
          # print('c[i][j]:',self.c[i][j])
          self.b[i][j]=self.b[i][j]-self.xite*g_b[i][j]

    # 误差
    x1 = np.linspace(0,self.loop,self.loop)
    # plt.plot(x1, y, 'r.-')
    # plt.plot(x1, yn, 'yo-')
    plt.plot(x1, E, 'g-')
    plt.show()
    


  # 预测过程 
  def predict(self,X): 
    I=self.I;
    M=self.M;
    O=self.O;
    m=self.m
    #网络输出 
    print('c:',self.c,'b:',self.b,'w:',self.w)
    yn=np.zeros((O,X.shape[1]))
    for k in range(X.shape[1]): 
      x=X[:,k].reshape(-1,1)
      

      a=np.ones((M,1))
      a_mean=np.ones((M,1))
      
      u=[]
      for i in range(I):
        uu=[0 for n in range(m[i])]
        for j in range(m[i]):
          uu[j]=np.exp(-(x[i]-self.c[i][j])**2/self.b[i][j]**2)
        u.append(uu)
      # print(u,u[1][1])

      dd=list(product(*u))
      # print(dd)
      
      for i in range(M):
        a[i]=min(dd[i])
      add_a=np.sum(a)
      a_mean=a/add_a
      # print(self.w,a_mean)
      yi=np.dot(self.w,a_mean)
      # print(type(yi),yi,type(yn[:,k]),yn[:,k])
      for i in range(O):
        yn[i][k]=yi[i]
    return yn



if __name__ == "__main__": 

  input_train = np.loadtxt(r'input_train.csv', delimiter=',', skiprows=0)
  output_train = np.loadtxt(r'output_train.csv', delimiter=',', skiprows=0)
  input_train = np.atleast_2d(input_train)
  output_train = np.atleast_2d(output_train)
  # print(type(output_train),output_train.shape)
  [X,m1]=Normalize(input_train)
  # [Y,m2]=Normalize(output_train)
  Y=output_train
  # [Y,m2]=Normalize(output_train)

  I=6;
  m=[2,2,2,2,2,2];
  # M=12;
  O=1;

  layers=[I,m,O]

  nn = NeuralNetwork(layers) 
  nn.fit(X,Y)


  input_test = np.loadtxt(r'input_test.csv', delimiter=',', skiprows=0)
  output_test = np.loadtxt(r'output_test.csv', delimiter=',', skiprows=0)
  input_test = np.atleast_2d(input_test)
  [Xt,mt]=Normalize(input_test)
  y=nn.predict(Xt)
  print(output_train,y)
  # Yt=Normalize_deriv(y,m2)
  # print(output_test,Yt)
