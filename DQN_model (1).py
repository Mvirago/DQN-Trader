#!/usr/bin/env python
# coding: utf-8

# #  IMPORTING LIBRARIES

# In[1]:


from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential , load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import math
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque
import pandas as pd


# # LOADING DATA

# In[2]:


data = pd.read_csv('GOOG (4).csv')
data=np.array(data['Close'])
data=data[:850]
print(len(data))


# In[3]:


plt.figure(figsize=(20,10))
plt.plot(range(len(data)),data)
plt.show()


# # AGENT CLASS

# In[4]:


class Agent :
  def __init__(self,Money,MAXT, state_size,model_name=""):
        self.state_size = state_size+2 #(days + money + no. of transactions)
        self.action_size = 3 # buy sell hold
        self.memory = deque(maxlen=1000)
        self.inventory = 0. #no. of stock in possesion
        self.initial_money=Money 
        self.money=float(Money) #money agent have after every transaction
        self.money_before=float(Money) 
        self.transactions=0 
        self.max_t=MAXT
        self.is_eval = False #true when model is used for prediction
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._model() if model_name=="" else load_model(model_name)
  def _model(self):
        model = Sequential()
        model.add(Dense(units=16, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))
        return model
  def act(self, state):
        if not self.is_eval and random.random()<= self.epsilon:
            s = " random " #just to check whether decisions are taken by model or randomly
            return random.randrange(self.action_size) , s
        input = np.reshape(state,(1,-1))
        options = self.model.predict(input)
        s = " not random "
        return np.argmax(options[0]) , s
  def expReplay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size+1, l):
            mini_batch.append(self.memory[i])
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            input = np.reshape(state,(1,-1))
            if not done:
                next_input=np.reshape(state,(1,-1))
                target = reward + self.gamma * np.amax(self.model.predict(next_input)[0])#Bellman Equation
            target_f = self.model.predict(input)
            target_f[0][action] = target
            self.model.fit(input, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            


# # FUNCTIONS - BUY , SELL , GET_STATE

# In[5]:


def formatPrice(n):
    return("-Rs." if n<0 else "Rs.")+"{0:.2f}".format(abs(n)) 
def sigmoid(x):
    return 1/(1+math.exp(-x))

def buy(agent,price):
    if(agent.transactions>=agent.max_t or agent.money<=0):
        return -1
    x=agent.money/(agent.max_t-agent.transactions)
    agent.money=agent.money-x
    stock=x/price
    agent.transactions+=1
    agent.inventory=stock+agent.inventory
    return 0
    
def sell(agent,price):
    if(agent.inventory==0) :
        return -1
    agent.money+=price*agent.inventory
    value =max(agent.initial_money,agent.money_before)
    reward=max(agent.money-value,0)
    agent.money_before=agent.money
    agent.inventory=0
    agent.transactions=0
    return reward

def get_state(agent ,data):
    value = np.reshape(data,(-1,1))
    ss=StandardScaler()
    value=ss.fit_transform(value)

    value=value[:,0]
    for i in range(len(value)):
        value[i]=sigmoid(value[i])
    value=np.append(value,[agent.transactions/agent.max_t,agent.money/(agent.money_before+1)])
    return np.array(value)


# In[6]:


window_size = 8
episode_count = 6
Money=10000.0;
MAXT=5;


# In[7]:


agent = Agent(Money,MAXT,window_size)


# In[ ]:





# # TRAINING 

# In[8]:


agent.is_eval=False
episode_count = 12
l = len(data)
agent.max_t=5
batch_size = 32
for e in range(episode_count):
    print("Episode " + str(e) + "/" + str(episode_count))
    state = get_state(agent,data[0:window_size])
    agent.money=Money
    code ={0:'b',1:'r',2:'g'}
    decisions=[]
    actions=[0,0,0]
    agent.inventory=0.
    agent.transactions=0.
    for t in range(window_size,l-1):
        action , s = agent.act(state)
        decisions.append(code[action])
        reward = 0
        if action==0:
            actions[0]=actions[0]+1
            actions[1]=0
            actions[2]=0
            reward = -1*(actions[0])
        if action == 1: # buy
            reward=buy(agent,float(data[t-1]))
            
            actions[1]=actions[1]+1
            actions[0]=0
            actions[2]=0
            reward=reward*actions[1]
        elif action == 2 :
            reward=sell(agent,data[t-1])
            actions[2]=actions[2]+1
            actions[1]=0
            actions[0]=0
            if reward <0:
                  reward = -1*(2**(actions[2]))
        next_state = get_state(agent,data[t-window_size+1:t+1])
        done = True if t == l - 2 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            agent.money+=agent.inventory*float(data[t])
            print("--------------------------------")
            print("Total Profit: " + formatPrice(agent.money-Money))
            print("--------------------------------")
        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)
    
        print(str(t)+ "    "+str(agent.money)+" " +str(action)+s)
    plt.figure(figsize=(20,30))
    for t in range(window_size,l-1):
        plt.scatter(t,data[t-1],color=decisions[t-window_size])
    plt.plot(range(len(data)),data)
    plt.show()
    agent.memory.clear()
    
    
agent.model.save('lastplease')


# # TESTING

# In[9]:


stocks=['GOOG','AMZN','AAPL','TSLA','^NSEI']


# In[15]:


for stock in stocks:
    test = pd.read_csv(stock+'.csv')
    l = len(test)
    agent.money=Money
    agent.inventory=0.
    agent.is_eval=True
    agent.transactions=0.
    test=np.array(test['Close'])
    state=get_state(agent,test[0:window_size])
    decisions={0:[],1:[],2:[]}
    for t in range(window_size,l-1):
            action,_ = agent.act(state)
            next_state = get_state(agent,test[t-window_size+1:t+1])
            decisions[action].append([t-1,test[t-1]])
            if action == 1: # buy
                buy(agent,float(test[t-1]))
            elif action == 2 :
                sell(agent,test[t-1])
            done = True if t == l - 2 else False
            state = next_state
            if done:
                agent.money+=agent.inventory*float(test[t])
                print("--------------------------------")
                print("Total Profit: " + formatPrice(agent.money-Money))
                print("--------------------------------")
    h = np.array(decisions[0])
    b = np.array(decisions[1])
    s = np.array(decisions[2])
    
    plt.figure(figsize=(20,10))
    plt.title(stock)
    Buy=plt.scatter(b[:,0],b[:,1],color='r')
    Hold=plt.scatter(h[:,0],h[:,1],color='b')
    Sell=plt.scatter(s[:,0],s[:,1],color='g')
    plt.plot(range(len(test)),test,color='lightblue')
    plt.legend(handles = [Buy,Hold,Sell], 
               labels  = ['Buy'+'('+str(len(b))+')', 'Hold'+'('+str(len(h))+')', 'Sell'+'('+str(len(s))+')'])
    plt.show()
    


# # TESTING WITH VALUE CHANGE

# In[14]:


Money=20000 
agent.max_t=7


# In[12]:


for stock in stocks:
    test = pd.read_csv(stock+'.csv')
    l = len(test)
    agent.money=Money
    agent.inventory=0.
    agent.is_eval=True
    agent.transactions=0.
    test=np.array(test['Close'])
    state=get_state(agent,test[0:window_size])
    decisions={0:[],1:[],2:[]}
    for t in range(window_size,l-1):
            action,_ = agent.act(state)
            next_state = get_state(agent,test[t-window_size+1:t+1])
            decisions[action].append([t-1,test[t-1]])
            if action == 1: # buy
                buy(agent,float(test[t-1]))
            elif action == 2 :
                sell(agent,test[t-1])
            done = True if t == l - 2 else False
            state = next_state
            if done:
                agent.money+=agent.inventory*float(test[t])
                print("--------------------------------")
                print("Total Profit: " + formatPrice(agent.money-Money))
                print("--------------------------------")
    h = np.array(decisions[0])
    b = np.array(decisions[1])
    s = np.array(decisions[2])
    
    plt.figure(figsize=(20,10))
    plt.title(stock)
    Buy=plt.scatter(b[:,0],b[:,1],color='r')
    Hold=plt.scatter(h[:,0],h[:,1],color='b')
    Sell=plt.scatter(s[:,0],s[:,1],color='g')
    plt.plot(range(len(test)),test,color='lightblue')
    plt.legend(handles = [Buy,Hold,Sell], 
               labels  = ['Buy'+'('+str(len(b))+')', 'Hold'+'('+str(len(h))+')', 'Sell'+'('+str(len(s))+')'])
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




