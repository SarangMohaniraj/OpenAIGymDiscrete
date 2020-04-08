from DQN import DQN
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
  env_name="CartPole-v1"
  dqn = DQN(env_name=env_name,episodes=500,alpha=.001,alpha_decay=.001,gamma=.99,epsilon=1.0,epsilon_decay=.999,epsilon_min=0.01,batch_size=32)
  
  # try:
  #   dqn.run()
  # except KeyboardInterrupt:
  #   sys.exit(1)

  # dqn.plot(save=True)
  # dqn.save()
  
  
  dqn.load()
  dqn.plot()
  dqn.play()
  