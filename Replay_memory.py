from collections import namedtuple
import random
import _pickle as cPickle

Transition = namedtuple('Transition',
                        ['state', 'action', 'reward', 'next_state', 'terminal'])


class Replay_memory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def save(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def save_memory(self, size):
        file = open('replay_memory/replay_memory_size_'+str(size)+'_capacity_'+str(self.capacity)+'.txt','wb+')
        file.write(cPickle.dumps(self.__dict__))
        file.close()

    def load_memory(self, name):
        file = open('replay_memory/'+str(name)+'.txt' ,'rb')
        dataPickle = file.read()
        file.close()
        self.__dict__ = cPickle.loads(dataPickle)

    def __len__(self):
        return len(self.memory)