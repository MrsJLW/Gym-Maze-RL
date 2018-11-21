import gym
import gym_maze
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle

EPISODES = 500
model_file = 'Charts_and_Models/part1'
load_model = True


class Agent():
    def __init__(self, state_list, action_size):
        self.state_list = state_list
        self.action_size = action_size
        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_min = 0.1  # changed this from 0.01
        self.epsilon_decay = 0.995
        self.alpha = 0.5
        self.q = self.init_q()

    def init_q(self):
        q = {}
        for i in self.state_list:
            q[i] = {}
            for j in range(self.action_size):
                if i == (4, 4):  # Hardcoded this
                    q[i][j] = 0
                else:
                    q[i][j] =  0
                    # q[i][j] = np.random.normal(0, 0.1)
        return q

    def act(self, state):
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        if random.random() < self.epsilon:
            return np.random.randint(4)
        val = -10000000
        action = 0
        if random.random() <= 0.5:
            for i in range(self.action_size):
                if self.q[state][i] > val:
                    action = i
                    val = self.q[state][i]
        else:
            for i in range(self.action_size - 1, -1, -1):
                if self.q[state][i] > val:
                    action = i
                    val = self.q[state][i]
        return action

    def update_q(self, state, action, reward, next_state):
        val = -10000000
        for i in range(self.action_size):
            if self.q[next_state][i] > val:
                val = self.q[next_state][i]
        old = self.q[state][action]
        self.q[state][action] = self.q[state][action] + self.alpha * (reward + self.gamma * val - self.q[state][action])
        # if state == next_state:
        #     if old < self.q[state][action]:
        #         print((state, action), old, self.q[state][action])


def print_table(q):
    for i in range(5):
        for j in range(5):
            val = []
            for action in range(4):
                # val.append(float(str(q[(i,j)][action])[:4]))
                val.append(q[(i, j)][action])
            print(val, end="\t")
        print()
    print()

    for i in range(5):
        for j in range(5):
            val = -10000000
            a = 0
            for action in range(4):
                if q[(i, j)][action] > val:
                    a = action
                    val = q[(i, j)][action]
            if a == 0:
                print("UP   \t", end='')
            elif a == 1:
                print("DOWN \t", end='')
            elif a == 2:
                print("LEFT \t", end='')
            elif a == 3:
                print("RIGHT\t", end='')
        print()
    print()


if __name__ == '__main__':
    env = gym.make("maze-random-5x5-v0")
    time_list= []
    state_list = []
    for i in range(5):
        for j in range(5):
            state_list.append((i, j))
    epsilon = [1]
    time = []
    agent = Agent(state_list, 4)
    if load_model:
        infile = open(model_file, 'rb')
        agent.q = pickle.load(infile)
        infile.close()
        # agent.epsilon = 0
    for e in range(EPISODES):
        # if e > 5 :
        #     agent.epsilon = 0
        state = env.reset()
        state = (state[0], state[1])
        for t in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = (next_state[0], next_state[1])
            agent.update_q(state, action, reward, next_state)
            state = next_state
            if done or t == 499:
                print("episode: {}/{}, Took = {}, Epsilon = {}"
                      .format(e, EPISODES, t, agent.epsilon))
                time.append(t)

                # print_table(agent.q)
                # plt.plot(time)
                # plt.savefig('Charts_and_Models/part2.png')
                # if e % 100 == 0:
                #     outfile = open(model_file, 'wb')
                #     pickle.dump(agent.q, outfile)
                #     outfile.close()
                break
    plt.plot(time, 'r')
    plt.savefig('Charts_and_Models/Question4_part2.png')
    # plt.plot(time_list[0], 'r', time_list[1], 'g', time_list[2], 'b', time_list[3], 'y', time_list[4], 'k' )
    # plt.savefig('Charts_and_Models/part5_state_4_0.png')