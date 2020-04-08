import gym
import tensorflow as tf
import numpy as np
import random, time
from matplotlib import pyplot as plt

class DQN:
  def __init__(self,env_name,episodes,alpha,alpha_decay,gamma,epsilon,epsilon_decay,epsilon_min,batch_size):
    self.env_name = env_name
    self.env = gym.make(env_name)
    self.memory = []
    self.episodes = episodes
    self.scores = []
    self.start_time = time.time()

    self.alpha = alpha # learning rate
    self.alpha_decay = alpha_decay
    self.gamma = gamma # discount rate on future rewards
    self.epsilon = epsilon # exploration rate
    self.epsilon_decay = epsilon_decay # the decay of epsilon after each training batch
    self.epsilon_min = epsilon_min # the minimum exploration rate permissible
    self.batch_size = batch_size # maximum size of the batches sampled from memory

    # Init model
    self.model = tf.keras.Sequential()
    self.model.add(tf.keras.layers.Dense(32, input_dim=self.env.observation_space.shape[0], activation='relu'))
    self.model.add(tf.keras.layers.Dense(16, activation='relu'))
    self.model.add(tf.keras.layers.Dense(self.env.action_space.n))
    self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.alpha, decay=self.alpha_decay))

    print("Time to initialize DQN:",self.elapsed_time(self.start_time),end="\n\n\n")


  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))


  #This DQN has 2 networks

  # The first network (online network/Q network) is for the DQN to predict the next optimal movement with the highest Q-value based on the state
  # def act(self, state):
  #   return self.env.action_space.sample() if np.random.random() <= self.epsilon else np.argmax(self.model.predict(state))

  #for some reason this method works better although they're the same
  def act(self, state):
    if np.random.random() <= self.epsilon:
      return self.env.action_space.sample()
    return np.argmax(self.model.predict(state))

  """
  The second network (target network) is for the DQN to learn and improve based on the memory of the previous actions
  Called after each timestep in an episode after there is enough memory

  1) retrieves random set of experience (called a minibatch) from the agent's memory
  2) calculates Q-value for each experience in the minibatch (using Bellman equation)
  3) network is fit
  """
  def experience_replay(self):
    if len(self.memory) < self.batch_size:
      return

    minibatch = random.sample(self.memory, self.batch_size)
    states, new_q_values = [],[]

    for state, action, reward, next_state, done in minibatch:
      q_values = self.model.predict(state)
      q_values[0][action] = reward if done else reward + self.gamma * np.amax(self.model.predict(next_state)[0])
      states.append(state[0])
      new_q_values.append(q_values[0])

    self.model.fit(np.array(states), np.array(new_q_values), verbose=0)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

  def reshape(self,state):
    return np.reshape(state, (1,self.env.observation_space.shape[0]))

  
  def run(self):
    scores = []
    self.start_time = time.time()

    for episode in range(self.episodes):
      done = False
      score = 0
      observation = self.reshape(self.env.reset())
      episode_start = time.time()

      while not done:
        action = self.act(observation)
        next_observation, reward, done, info = self.env.step(action)
        next_observation = self.reshape(next_observation)
        score += reward
        self.remember(observation, action, reward, next_observation, done)
        observation = next_observation

        self.experience_replay()

      self.scores.append(score)
      print(f"episode: {episode+1}/{self.episodes} | score: {score} | epsilon: {self.epsilon:.3f}")
      print(f"episode time: {self.elapsed_time(episode_start)} | total time: {self.elapsed_time(self.start_time)}",end="\n\n")
      if episode % 5 == 0:
        self.save("CartPole-v1")
        print("SAVED!!!")


  def plot(self,save=False, show=True):
    x = np.arange(1,len(self.scores)+1)
    y = self.scores
    
    plt.plot(x,y,label="Scores")
    plt.xlabel("Episode")
    plt.ylabel("Score")

    # trend line
    if len(x) > 1:
      z = np.polyfit(x, y, 1)
      p = np.poly1d(z)
      plt.plot(x,p(x),"r--",label="Trend")
      
    # average
    if len(y) > 1:
      avg = np.mean(y)
      plt.plot(x,np.full(len(x),avg),"g:",label=f"Average Score: {avg:.3f}")
      
      # average of last 100 episodes
      if len(y) > 100:
        avg_last_100 = np.mean(y[-100:])
        plt.plot(x[-100:],np.full(len(x[-100:]),avg_last_100),"m-.",label=f"Last 100 Episode Average Score: {avg_last_100:.3f}",linewidth=2)
    
    plt.title("Score vs Episode")
    plt.legend()
    
    if save:
      plt.savefig(f"plots/{self.env_name}/plot.png")

    if show:
      plt.show()
    else:
      plt.close()


  def save(self):
    self.model.save_weights(f"models/{self.env_name}/model.h5", overwrite=True)
    np.save(f"plots/{self.env_name}/data/scores.npy",self.scores)

  def load(self):
    self.model.load_weights(f"models/{self.env_name}/model.h5")
    self.scores = np.load(f"plots/{self.env_name}/data/scores.npy")

  def play(self):
    while True:
      done = False
      score = 0
      observation = self.env.reset()
      temp = self.epsilon
      self.epsilon = 0 # always choose the greedy action over random

      while not done:
        action = self.act(self.reshape(observation))
        observation, reward, done, info = self.env.step(action)
        score += reward
        self.env.render()

      print("Score:", score,end="\n\n")
      self.epsilon = temp

      if input('Do you want the agent to play another game? (Y/N): ').lower()[0] == 'n':
        break;


  def elapsed_time(self,start_time):
    return time.strftime("%Hh, %Mm, %Ss", time.gmtime(time.time()-start_time)) if time.time()-start_time > 1 else f"{time.time()-start_time:.3f}μs"




