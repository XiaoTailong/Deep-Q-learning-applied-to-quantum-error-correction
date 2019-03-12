from torch import from_numpy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

import random
from collections import namedtuple
import os
import matplotlib.pyplot as plt
from copy import deepcopy
import heapq
import pandas as pd
import time

from toric_model import Toric_code
from toric_model import Action
from toric_model import Perspective

from Replay_memory import Replay_memory
from Replay_memory import Transition

from NN import NN_0
from NN import NN_1
from NN import NN_2
from NN import NN_3
from NN import NN_4
from NN import NN_5


class RL():
    def __init__(self, Network=str, system_size=int, p_error=0.1, capacity=int, dropout=0.0, learning_rate=float,
                discount_factor=float, number_of_actions=3, terminate_sequence=50, batch_size=32, 
                replay_start_size=32, device='cpu'):
        # device
        self.device = device
        # Toric code
        self.toric = Toric_code(system_size)
        self.grid_shift = int(system_size/2)
        self.terminate_sequence = terminate_sequence
        self.system_size = system_size
        self.p_error = p_error
        # Replay Memory
        self.capacity = capacity
        self.memory = Replay_memory(capacity)
        # Network
        self.network_name = Network
        self.network = self.select_network(Network)
        self.policy_net = self.network(system_size, dropout, number_of_actions, device)
        self.policy_net.to(self.device)
        self.target_net = self.network(system_size, dropout, number_of_actions, device)
        self.target_net.to(self.device)
        self.learning_rate = learning_rate
        # hyperparameters RL
        self.discount_factor = discount_factor
        self.number_of_actions = number_of_actions


    def select_network(self, network):
        if network == 'NN_0':
            return NN_0
        if network == 'NN_1':
            return NN_1
        if network == 'NN_2':
            return NN_2
        if network == 'NN_3':
            return NN_3
        if network == 'NN_4':
            return NN_4
        if network == 'NN_5':
            return NN_5


    def save_network(self, PATH):
        torch.save(self.policy_net, PATH)


    def load_network(self, PATH):
        self.policy_net = torch.load(PATH)
        self.target_net = deepcopy(self.policy_net)

    
    def plot_average_action_value(self, mean_q):
        plt.title('Average predicted action-value')
        plt.ylabel('Average action value (Q)')
        plt.xlabel('Training epochs')
        plt.plot(mean_q)
        plt.savefig('plots/action_q_value.png')
        plt.close()


    def plot_error_correction(self, error_corrected_list=float, ground_state_list=float, list_p_error=float):
        ax = plt.subplot(111)
        plt.title('Error correction success rate')
        plt.ylabel('success rate')
        plt.xlabel('p_error')
        ax.plot(list_p_error, error_corrected_list, label='error corrected')
        ax.plot(list_p_error, ground_state_list, label='ground state conserved')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=False, ncol=5)
        plt.savefig('plots/error_correction.png')
        plt.close()


    def experience_replay(self, criterion, optimizer, discount_factor, batch_size, clip_error_term):
        def generate_input(tensor):
            tensor = from_numpy(tensor)
            tensor = tensor.type('torch.Tensor')
            return tensor
        if len(self.memory) < batch_size:
            return
        self.policy_net.train()
        self.target_net.eval()
        # get transitions and unpack them to minibatch
        transitions = self.memory.sample(batch_size)
        mini_batch = Transition(*zip(*transitions))
        # unpack action batch
        batch_actions = Action(*zip(*mini_batch.action))
        batch_actions = np.array(batch_actions.action) - 1
        batch_actions = torch.Tensor(batch_actions).long()
        batch_actions = batch_actions.to(self.device)
        # preprocess batch_input and batch_target_input for the network
        batch_input = np.stack(mini_batch.state, axis=0)
        batch_input = generate_input(batch_input)
        batch_input = batch_input.to(self.device)
        batch_target_input = np.stack(mini_batch.next_state, axis=0)
        batch_target_input = generate_input(batch_target_input)
        batch_target_input = batch_target_input.to(self.device)
        # preprocess batch_terminal and batch reward
        batch_terminal = generate_input(np.array(mini_batch.terminal)) 
        batch_terminal = batch_terminal.to(self.device)
        batch_reward = generate_input(np.array(mini_batch.reward))
        batch_reward = batch_reward.to(self.device)
        # compute output from network and gather q value for selected action
        output = self.policy_net(batch_input)
        output = output.gather(1, batch_actions.view(-1, 1))
        output = output[:, 0]
        # compute target output from network and take max q value
        with torch.no_grad():
            target_output = self.target_net(batch_target_input)
            target_output = target_output.max(1)[0].detach()
        # batch_terminal is an array 0 for terminal and 1 for not terminal
        y = batch_reward + (batch_terminal * discount_factor * target_output)
        # clip error term
        ind = (y > clip_error_term)
        y[ind] = clip_error_term
        ind = (y < -clip_error_term)
        y[ind] = -clip_error_term
        # backpropagate loss        
        loss = criterion(y, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    def select_action(self, number_of_actions=int, epsilon=float, grid_shift=int, prev_action=float):
        # set network in evluation mode 
        self.policy_net.eval()
        # generate perspectives 
        self.toric.generate_perspective(grid_shift)
        number_of_perspectives = len(self.toric.perspectives)
        # preprocess batch of perspectives and actions 
        perspectives = Perspective(*zip(*self.toric.perspectives))
        batch_perspectives = np.array(perspectives.perspective)
        batch_perspectives = from_numpy(batch_perspectives)
        batch_perspectives = batch_perspectives.type('torch.Tensor')
        batch_perspectives = batch_perspectives.to(self.device)
        batch_position_actions = perspectives.position
        #choose action using epsilon greedy approach
        rand = random.random()
        if(1 - epsilon > rand):
            # select greedy action 
            with torch.no_grad():        
                policy_net_output = self.policy_net(batch_perspectives)
                q_values_table = np.array(policy_net_output.cpu())
                row, col = np.where(q_values_table == np.max(q_values_table))
                perspective = row[0]
                max_q_action = col[0] + 1
                step = Action(batch_position_actions[perspective], max_q_action)
                # avoid loop of actions
                if prev_action == step:
                    res = heapq.nlargest(2, q_values_table.flatten())
                    row, col = np.where(q_values_table == res[1])
                    perspective = row[0]
                    max_q_action = col[0] + 1
                    step = Action(batch_position_actions[perspective], max_q_action)
        # select random action
        else:
            random_perspective = random.randint(0, number_of_perspectives-1)
            random_action = random.randint(1, number_of_actions)
            step = Action(batch_position_actions[random_perspective], random_action)   
        return step


    def train(self, training_steps=int, target_update=int, epsilon_start=1.0, num_of_epsilon_steps=10, 
        epsilon_end=0.1, clip_error_term=1, reach_final_epsilon=0.3, reward_definition=int, optimizer=str, batch_size=int,
        replay_start_size=int):
        # set network to train mode
        self.policy_net.train()
        # define criterion and optimizer
        criterion = nn.MSELoss()
        if optimizer == 'RMSprop':
            optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate)
        elif optimizer == 'Adam':    
            optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        # init counters
        steps_counter = 0
        update_counter = 1
        # define epsilon steps 
        epsilon = epsilon_start
        num_of_steps = np.round(training_steps/num_of_epsilon_steps)
        epsilon_decay = np.round((epsilon_start-epsilon_end)/num_of_epsilon_steps, 5)
        epsilon_update = num_of_steps * reach_final_epsilon
        # main loop over training steps 
        while update_counter < training_steps:
            num_of_steps_per_episode = 0
            # initialize syndrom
            self.toric = Toric_code(self.system_size)
            terminal_state = 0
            while terminal_state == 0:
                self.toric.generate_random_error(self.p_error)
                terminal_state = self.toric.terminal_state(self.toric.state)
            # solve one episode
            prev_action = 0
            while terminal_state == 1 and num_of_steps_per_episode < self.terminate_sequence and update_counter < training_steps:
                num_of_epsilon_steps += 1
                steps_counter += 1
                # select action using epsilon greedy policy
                action = self.select_action(number_of_actions=self.number_of_actions,
                                            epsilon=epsilon, 
                                            grid_shift=self.grid_shift,
                                            prev_action=prev_action)
                #prev_action = action
                self.toric.step(action)
                reward = self.get_reward(reward_definition=reward_definition)
                # generate memory entry
                perspective, action_memory, reward, next_perspective, terminal = self.toric.generate_memory_entry(
                    action, reward, self.grid_shift)    
                # save transition in memory
                self.memory.save(perspective, action_memory, reward, next_perspective, terminal)
                # experience replay
                if steps_counter > replay_start_size:
                    update_counter += 1
                    self.experience_replay(criterion, optimizer, self.discount_factor, batch_size, clip_error_term)
                # set target_net to policy_net
                if update_counter % target_update == 0:
                    self.target_net = deepcopy(self.policy_net)
                # update epsilon
                if (update_counter % epsilon_update == 0):
                    epsilon = np.round(np.maximum(epsilon - epsilon_decay, epsilon_end), 3)
                # set next_state to new state and update terminal state
                self.toric.state = self.toric.next_state
                terminal_state = self.toric.terminal_state(self.toric.state)


    def get_reward(self, reward_definition=int):
        if reward_definition == 0:
            defects_state = np.sum(self.toric.state)
            defects_next_state = np.sum(self.toric.next_state)
            reward = defects_state - defects_next_state
            if reward == 0:
                reward = -1
            terminal = np.all(self.toric.next_state==0)
            if terminal == True:
                reward = 5

        elif reward_definition == 1:
            terminal = self.toric.terminal_state(self.toric.next_state)
            if terminal == 0:
                reward = 10
            else:
                reward = 0

        return reward


    def select_action_evaluation(self, number_of_actions=int, epsilon=float, grid_shift=int, prev_action=float):
        # set network in eval mode
        self.policy_net.eval()
        # generate perspectives
        self.toric.generate_perspective(grid_shift)
        number_of_perspectives = len(self.toric.perspectives)
        # preprocess batch of perspectives and actions 
        perspectives = Perspective(*zip(*self.toric.perspectives))
        batch_perspectives = np.array(perspectives.perspective)
        batch_perspectives = from_numpy(batch_perspectives)
        batch_perspectives = batch_perspectives.type('torch.Tensor')
        batch_perspectives = batch_perspectives.to(self.device)
        batch_position_actions = perspectives.position
        # batch_position_actions = batch_position_actions.to(self.device)
        # generate action value for different perspectives 
        with torch.no_grad():
            policy_net_output = self.policy_net(batch_perspectives)
            q_values_table = np.array(policy_net_output.cpu())
        #choose action using epsilon greedy approach
        rand = random.random()
        if(1 - epsilon > rand):
            # select greedy action 
            row, col = np.where(q_values_table == np.max(q_values_table))
            perspective = row[0]
            max_q_action = col[0] + 1
            step = Action(batch_position_actions[perspective], max_q_action)
            if prev_action == step:
                res = heapq.nlargest(2, q_values_table.flatten())
                row, col = np.where(q_values_table == res[1])
                perspective = row[0]
                max_q_action = col[0] + 1
                step = Action(batch_position_actions[perspective], max_q_action)
            q_value = q_values_table[row[0], col[0]]
        # select random action
        else:
            random_perspective = random.randint(0, number_of_perspectives-1)
            random_action = random.randint(1, number_of_actions)
            q_value = q_values_table[random_perspective, random_action-1]
            step = Action(batch_position_actions[random_perspective], random_action)
        return step, q_value


    def evaluation(self, evaluation_steps=int, epsilon=float):
        def incremental_mean(x, mu, N):
            return mu + (x - mu) / N
        self.policy_net.eval()           
        iteration = 0
        mean_q = 0
        while iteration < evaluation_steps:
            num_of_steps_per_episode = 0
            # generate random syndrom
            self.toric = Toric_code(self.system_size)
            terminal_state = 0
            while terminal_state == 0:
                self.toric.generate_random_error(self.p_error)
                terminal_state = self.toric.terminal_state(self.toric.state)
            prev_action = 0
            while terminal_state == 1 and num_of_steps_per_episode < self.terminate_sequence and iteration < evaluation_steps:
                iteration += 1
                num_of_steps_per_episode += 1
                action, q_value = self.select_action_evaluation(number_of_actions=self.number_of_actions, 
                                                                epsilon=epsilon,
                                                                grid_shift=self.grid_shift,
                                                                prev_action=prev_action)
                prev_action = action
                mean_q = incremental_mean(q_value, mean_q, iteration)
                self.toric.step(action)
                self.toric.state = self.toric.next_state
                terminal_state = self.toric.terminal_state(self.toric.state)
                
        return mean_q


    def prediction(self, num_of_predictions=1, epsilon=0.0, num_of_steps_for_solving_one_episode=15, PATH=None, plot_one_episode=False, 
        show_network=False, show_plot=False):
        def incremental_mean(x, mu, N):
            return mu + (x - mu) / (N)
        # load network for prediction and set eval mode 
        if PATH != None:
            self.load_network(PATH)
        self.policy_net.eval()
        # print network architecture
        if show_network == True:
            summary(self.policy_net, (2,self.system_size,self.system_size))
        list_p_error = [0.05, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18]
        ground_state_list = np.zeros(len(list_p_error))
        error_corrected_list = np.zeros(len(list_p_error))
        average_number_of_steps = np.zeros(len(list_p_error))
        # loop through different p_error
        for i, p_error in enumerate(list_p_error):
            ground_state = np.ones(num_of_predictions, dtype=bool)
            error_corrected = np.zeros(num_of_predictions)
            mean_steps_per_p_error = 0
            for j in range(num_of_predictions):
                num_of_steps_per_episode = 0
                # generate random syndrom
                terminal_state = 0
                self.toric = Toric_code(self.system_size)
                while terminal_state == 0:
                    self.toric.generate_random_error(p_error)
                    terminal_state = self.toric.terminal_state(self.toric.state)
                # plot solution
                if plot_one_episode == True and j == 0 and i == 3:
                    self.toric.plot_toric_code(self.toric.state, 'initial_syndrom')
                # loop for action prediction
                prev_action = 0
                while terminal_state == 1 and num_of_steps_per_episode < num_of_steps_for_solving_one_episode:
                    # choose greedy action
                    action = self.select_action(number_of_actions=self.number_of_actions, 
                                                epsilon=epsilon,
                                                grid_shift=self.grid_shift,
                                                prev_action=prev_action)
                    prev_action = action
                    self.toric.step(action)
                    self.toric.state = self.toric.next_state
                    terminal_state = self.toric.terminal_state(self.toric.state)
                    if plot_one_episode == True and j == 0 and i == 3:
                        self.toric.plot_toric_code(self.toric.state, 'step_'+str(num_of_steps_per_episode))
                    num_of_steps_per_episode += 1
                mean_steps_per_p_error = incremental_mean( num_of_steps_per_episode, mean_steps_per_p_error, j+1)
                error_corrected[j] = self.toric.terminal_state(self.toric.state) # 0: error corrected 
                                                                                 # 1: error not corrected       
                self.toric.eval_ground_state()                                                          
                ground_state[j] = self.toric.ground_state
            success_rate = (num_of_predictions - np.sum(error_corrected)) / num_of_predictions
            error_corrected_list[i] = success_rate            
            ground_state_change = (num_of_predictions - np.sum(ground_state)) / num_of_predictions
            ground_state_list[i] =  1 - ground_state_change
            average_number_of_steps[i] = np.round(mean_steps_per_p_error, 1)
        print(list_p_error, 'list p_error')
        print(error_corrected_list, 'error corrected')
        print(ground_state_list, 'ground state conserved')
        print(average_number_of_steps, 'average number of steps')
        if show_plot == True:
            self.plot_error_correction(ground_state_list=ground_state_list,
                                    error_corrected_list=error_corrected_list,
                                    list_p_error=list_p_error)

        return error_corrected_list, ground_state_list, average_number_of_steps


    def train_for_n_epochs(self, training_steps=int, evaluation_steps=int, epochs=int, PATH=str, prediction_steps=100, 
        clip_error_term=1, target_update=100, reward_definition=int, optimizer=str, save_model_each_epoch=False, data=[],
        batch_size=32, replay_start_size=32, localtime=str):
        # print network summary
        summary(self.policy_net, (2, self.system_size, self.system_size))
        # evaluate random network
        mean_q = np.zeros(epochs + 1)
        mean_q[0]= self.evaluation(evaluation_steps=evaluation_steps, epsilon=0.05)
        # loop over epochs
        for i in range(epochs):
            # train model
            self.train(training_steps=training_steps, 
                    clip_error_term=clip_error_term,    
                    target_update=target_update, 
                    reward_definition=reward_definition,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    replay_start_size=replay_start_size)
            # evaluate network 
            print('epoch: ', i+1)
            mean_q_temp = self.evaluation(evaluation_steps=evaluation_steps, epsilon=0.05)
            mean_q[i + 1] = np.round(mean_q_temp, 4)        
            error_corrected_list, ground_state_list, average_number_of_steps = self.prediction(
                    num_of_predictions=prediction_steps)
            print(np.round(mean_q, 3), 'mean q value')
            print()
            # save information about the training run 
            data.append((self.system_size, self.network_name, self.learning_rate, target_update, optimizer, reward_definition, 
                training_steps, i, mean_q[i+1], error_corrected_list[3], ground_state_list[3], average_number_of_steps[3]))
            df = pd.DataFrame(data, columns=['system size', 'network', 'learning rate', 'target update', 'optimizer', 'reward', 
                'training steps', 'epoch', 'mean q', 'corrected error (1=100%)', 'conserved groundstate (1=100%)', 'average number of steps'])
            df.to_csv(str(localtime[0:10])+' .csv', sep='\t', encoding='utf-8')
            # save network after each epoch
            if save_model_each_epoch == True:
                step = (i + 1) * training_steps
                PATH = 'network/network_epoch/size_{3}_{2}_target_update_{5}_optimizer_{6}_reward_{7}_epoch_{0}_steps_{4}_q_{1}.pt'.format(i+1, np.round(mean_q[i+1], 4), 
                    self.network_name, self.system_size, step, target_update, optimizer, reward_definition)
                self.save_network(PATH)
        # save model and plot average action value
        steps = training_steps * epochs
        PATH2 = 'network/size_{0}_{1}_steps_{2}.pt'.format(self.system_size, self.network_name, steps)
        self.save_network(PATH2)
        PATH2 = 'network/size_{0}_{1}_target_update_{3}_optimizer_{4}_reward_{5}_steps_{2}_mean_q'.format(self.system_size, 
                                self.network_name, epochs*training_steps, target_update, optimizer, reward_definition)
        np.save(PATH2, np.round(mean_q, 3))
        self.plot_average_action_value(mean_q)
