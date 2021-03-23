from gridEnv import GridEnv
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from collections import deque
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

#env = GridEnv()
#valeur possible
#print(env.observation_space.sample()[0])

#action possible
#print(env.action_space.sample())
#print(states,actions)

def process_log(observation):
    observation = np.array(observation)
    observation_temp = np.where(observation <= 0, 1, observation)
    processed_observation = np.log2(observation_temp)/np.log2(2**16)
    return processed_observation.reshape(1,4,4)

def get_grids_next_step(grid):
    env1 = GridEnv()
    env1.setMatrix(grid)
    return env1.getAvailableMovesForMax()

class DQN:
    def __init__(self,env):
        #Defining the hyperparameters for the model
        self.env=env.getMatrix()
        #The replay memory will be stored in a Deque
        self.memory=deque(maxlen=2000)
        self.gamma=0.90
        self.epsilon=1.0
        self.epsilon_min=0.01
        self.epsilon_decay=0.995
        self.learning_rate=0.005
        self.tau=0.125
        #We use 2 models to prevent Bootstrapping
        self.model=self.create_model()
        self.target_model=self.create_model()
        
        
    def create_model(self):
        model=Sequential()
        state_shape=4
                
        model.add(Flatten(input_shape=(4,4)))
        model.add(Dense(units=1024,activation="relu"))
        model.add(Dense(units=512,activation="relu"))
        model.add(Dense(units=256,activation="relu"))
        model.add(Dense(units=4))
        model.compile(loss="mean_squared_error",optimizer=Adam(lr=self.learning_rate))
        print(model.summary())
        return model
                  
    def act(self,state):
        #Epsilon value decays as model gains experience
        self.epsilon*=self.epsilon_decay
        self.epsilon=max(self.epsilon_min,self.epsilon)
        if np.random.random()<self.epsilon:
                  return np.random.randint(0,4)
        else:
            #Getting the 4 future states
            allstates=get_grids_next_step(state)
            
            res=[]
            for i in range(len(allstates)):
                if (allstates[i]==state).all():
                    res.append(0)
                else:
                    processed_state=process_log(allstates[i])
                    #max from the 4 future Q_Values is appended in res
                    res.append(np.max(self.model.predict(processed_state)))
            
            a=self.model.predict(process_log(state))
            #Final Q_Values are the sum of Q_Values of current state andfuture states
            final=np.add(a,res)
            
            return np.argmax(final)
    
    def remember(self, state, action, reward, new_state, done):
        #Replay Memory stores tuple(S, A, R, S')
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size=32
        if len(self.memory)<batch_size:
            return
        samples=random.sample(self.memory,batch_size)
        for sample in samples:
            
            state,action,reward,new_state,done=sample
            
            target=self.target_model.predict(process_log(state))
            
            
            if done:
                target[0][action]=reward
            else:
                #Bellman Equation for update
                Q_future=max(self.target_model.predict(process_log(new_state))[0])
                
                #The move which was selected, its Q_Value gets updated
                target[0][action]=reward+Q_future*self.gamma
            self.model.fit((process_log(state)),target,epochs=1,verbose=0)
                  
                  
    def target_train(self):
        weights=self.model.get_weights()
        target_weights=self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i]=weights[i]*self.tau+target_weights[i]*(1-self.tau)
        self.target_model.set_weights(target_weights)
                  
                  
    def save_model(self,fn):
        self.model.save(fn)


def main():
    env=GridEnv()
    gamma=0.9
    epsilon=0.95
    
    trials=1000
    trial_len=500
    dqn_agent=DQN(env=env)
    steps=[]
    
    
    
    for trial in range(trials):
        env.reset()
        cur_state=env.getMatrix()
        
        cur_state=np.array(cur_state)
        stepno=0
        
        for step in range(trial_len):
            stepno+=1
            
            action=dqn_agent.act(cur_state)
            
            _,reward,done,_=env.step(action)
            new_state=env.getMatrix()
            
            dqn_agent.remember(cur_state,action,reward,new_state,done)
            if step%10==0:
                dqn_agent.replay()
                dqn_agent.target_train()
            
            cur_agent=new_state
            
            if done:
                
                break
        
        if stepno<500:
            
            if env.highest()==2048:
                print("Completed in --",trial)
                print(env.getMatrix())
                max_tile.append(env.maxValue())

                dqn_agent.save_model("success.model")

            
            else:
                print(f"Trial number {trial} Failed to complete in 500 steps")
                print(env.getMatrix())
                max_tile.append(env.maxValue())
                if env.maxValue()>=512:
                    dqn_agent.save_model("trial num-{}.model".format(trial))
                    
        else:
            print(f"Failed to complete in 500 steps")
            print(env.getMatrix())
            max_tile.append(env.maxValue())

            
if __name__ == "__main__":
    main()

"""
matrix = [
    [256,64,4,2],
    [128,16,8,0],
    [0,0,0,0],
    [0,0,0,0],
]
print(get_grids_next_step(matrix))
"""