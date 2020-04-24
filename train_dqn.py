#!/usr/bin/env python3

import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

COLAB = False
CUDA = True

if not COLAB:
    from lib import wrappers
    from lib import dqn_model

    import argparse
    from tensorboardX import SummaryWriter

#ENV_NAME = "PongNoFrameskip-v4"
ENV_NAME = "Riverraid-v0"
#MEAN_REWARD_BOUND = 19.5 # for PongNoFrameskip env
MEAN_REWARD_BOUND = 5000  # for Riverraid-v0

GAMMA = 0.99   # used in loss_calc, ??????????????????????
BATCH_SIZE = 32
# if REPLAY_SIZE IS BIG itS POSSIBLY CAUSES RAM BLOATING
# SO I TRIED TO REDUCE IT
#REPLAY_SIZE = 10 ** 4 * 2
REPLAY_SIZE = 4000        # just a max len of deque to store frames (ExperienceReplay.buffer)
LEARNING_RATE = 1e-4
TARGET_UPDATE_FREQ = 1000  # update target_net each 1000 frames
#LEARNING_STARTS = 10000
LEARNING_STARTS = 3000 # How many frames to be in replay buffer at begining of learning

EPSILON_DECAY = 10**5      # 10**5 = 100000 so epsilon will be decaying for 100 000 frames.
EPSILON_START = 1.0
EPSILON_FINAL = 0.01       # after 100 000 frames epsilon will stay at this value


MODEL = "PretrainedModels/PongNoFrameskip-v4-407.dat"
LOAD_MODEL = True

# its like a container to store data for single frame
# we will pass this "containers" into ExperienceReplay.buffer
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

# class to create replay_memory
class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        #print('buffer size is:', len(self.buffer))
        return len(self.buffer)

    def append(self, experience):
        #print('expirience:', experience)
        self.buffer.append(experience)

    # dtype=np.uint8 to be replaced with dtype=np.bool ?????
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, replay_memory):
        self.env = env
        self.replay_memory = replay_memory
        self._reset()
        self.last_action = 0

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        """
        Select action
        Execute action and step environment
        Add state/action/reward to experience replay
        """
        done_reward = None
        # np.random.random() is random number from 0 to 1.
        if np.random.random() < epsilon:
            # take random action
            action = env.action_space.sample()
        else:
            # get pixels for current frame
            state_a = np.array([self.state], copy=False)
            # convert to torch datatype and send to GPU
            state_v = torch.tensor(state_a).to(device)
            # get Q values for state
            q_vals_v = net(state_v)

            _, act_v = torch.max(q_vals_v, dim=1)
            #print('act_v', act_v, 'act_v.item():', act_v.item())
            # action to be done based on state_v
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        new_state = new_state

        # just store dato for current frame in exp variable
        exp = Experience(self.state, action, reward, is_done, new_state)
        # append this data to replay_memory
        self.replay_memory.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            # reset the environment if done
            self._reset()
        return done_reward

# we use 2 nets (net, target_net) to calculate loss ???????????????????
def calculate_loss(batch, net, target_net, device="cpu"):
    """
    Calculate MSE between actual state action values,
    and expected state action values from DQN
    """
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done = torch.ByteTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.long().unsqueeze(-1)).squeeze(-1)
    next_state_values = target_net(next_states_v).max(1)[0]
    next_state_values[done] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)

print("ReplayMemory will require {}gb of GPU RAM".format(round(REPLAY_SIZE * 32 * 84 * 84 / 1e+9, 2)))

if torch.cuda.is_available():
    #device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    #device = torch.device("cpu")
    print("Running on the CPU")



if __name__ == "__main__":
    if COLAB:
        pass
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
        parser.add_argument("--env", default=ENV_NAME,
                            help="Name of the environment, default=" + ENV_NAME)
        parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                            help="Mean reward to stop training, default={}".format(round(MEAN_REWARD_BOUND, 2)))
        parser.add_argument("-m", "--model", help="Model file to load")
        args = parser.parse_args()

    device = torch.device("cuda:0")
    #device = torch.device("cuda:0" if args.cuda else "cpu")
    #device = torch.device("cpu")

    # Make Gym environement and DQNs
    if COLAB:
        pass
    else:
        env = wrappers.make_env(args.env)
        net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
        target_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
        writer = SummaryWriter(comment="-" + args.env)

    print(net)

    replay_memory = ExperienceReplay(REPLAY_SIZE)
    print('replay_memory initialized')
    agent = Agent(env, replay_memory)
    print('agent initialized')
    epsilon = EPSILON_START


    if LOAD_MODEL:
        net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
        target_net.load_state_dict(net.state_dict())
        print("Models loaded from disk!")

        # Lower exploration rate
        EPSILON_START = EPSILON_FINAL


    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    best_mean_reward = None
    frame_idx = 0
    timestep_frame = 0
    timestep = time.time()

    while True:

        #monitor how many namedtupels we have in replay_memory
        #print(len(replay_memory))


        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY)
        #print('epsilon',epsilon)

        # agent makes action every frame
        reward = agent.play_step(net, epsilon, device=device)

        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - timestep_frame) / (time.time() - timestep)
            timestep_frame = frame_idx
            timestep = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            print("{} frames: done {} games, mean reward {}, best_reward {}, eps {}, speed {} f/s".format(
                frame_idx, len(total_rewards), round(mean_reward, 3), round(best_mean_reward, 3), round(epsilon,2), round(speed, 2)))
            if not COLAB:
                writer.add_scalar("epsilon", epsilon, frame_idx)
                writer.add_scalar("speed", speed, frame_idx)
                writer.add_scalar("reward_100", mean_reward, frame_idx)
                writer.add_scalar("reward", reward, frame_idx)
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), args.env + "-" + str(len(total_rewards)) + ".dat")
                if COLAB:
                    pass
                if best_mean_reward is not None:
                    print("New best mean reward {} -> {}, model saved".format(round(best_mean_reward, 3), round(mean_reward, 3)))
                best_mean_reward = mean_reward
            if mean_reward > args.reward and len(total_rewards) > 10:
                print("Game solved in {} frames! Average score of {}".format(frame_idx, mean_reward))
                break

            # show memory RAM usage
            # pip install psutil
            import os
            import psutil
            def show_RAM_usage():
                py = psutil.Process(os.getpid())
                print('RAM usage: {} GB'.format(py.memory_info()[0] / 2. ** 30))
            show_RAM_usage()

        # iterating through this loop until buffer is filled with
        # specified number of frames(LEARNING_STARTS) for replaying .
        if len(replay_memory) < LEARNING_STARTS:
            print('len(replay_memory):',len(replay_memory),LEARNING_STARTS)
            continue

        # update target_net every 1000 frames
        if frame_idx % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = replay_memory.sample(BATCH_SIZE)
        loss_t = calculate_loss(batch, net, target_net, device=device)
        loss_t.backward()
        optimizer.step()

    env.close()
    writer.close()
