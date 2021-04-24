import random
import torch
from collections import namedtuple, deque

Transition = namedtuple('Transition', 's a r ns d')
Batch = namedtuple('Batch', 's a r ns d')

class ReplayBuffer(object):

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.memory.appendleft(transition)

    def ready_for(self, batch_size: int) -> bool:
        if len(self.memory) >= batch_size:
            return True
        return False

    def sample(self, batch_size: int) -> Batch:
        batch = random.sample(self.memory, batch_size)
        batch = Batch(*zip(*batch))
        s  = torch.tensor(batch.s, dtype  = torch.float).view(batch_size, -1)
        a  = torch.tensor(batch.a, dtype  = torch.float).view(batch_size, -1)  # continuous, multi-dim action
        r  = torch.tensor(batch.r, dtype  = torch.float).view(batch_size,  1)
        ns = torch.tensor(batch.ns, dtype = torch.float).view(batch_size, -1)
        d  = torch.tensor(batch.d, dtype  = torch.float).view(batch_size,  1)
        return Batch(s, a, r, ns, d)
