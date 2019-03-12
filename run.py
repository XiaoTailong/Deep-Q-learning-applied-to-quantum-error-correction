import numpy as np
import os
import torch
import _pickle as cPickle
from RL import RL
import time 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
localtime = time.asctime(time.localtime(time.time()))
os.system('clear')
data = []

rl = RL(Network='NN_2',
        system_size=3,
        p_error=0.1,
        capacity=200,
        dropout=0.0,
        learning_rate=0.00025,     
        discount_factor=0.95)

rl.train_for_n_epochs(training_steps=100, 
                        evaluation_steps=100, 
                        prediction_steps=10,
                        epochs=10, 
                        clip_error_term=5,
                        target_update=10,
                        reward_definition=0,
                        optimizer='Adam',
                        save_model_each_epoch=True,
                        data=data,
                        localtime=localtime)

# load network for predictions
PATH = 'network/test.pt'
error_corrected_list, ground_state_list, average_number_of_steps = rl.prediction(num_of_predictions=20, 
                                                                                num_of_steps_for_solving_one_episode=30, 
                                                                                PATH=PATH, 
                                                                                show_network=True, 
                                                                                plot_one_episode=True)


