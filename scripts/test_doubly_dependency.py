import torch
from torch.distributions import Normal, Independent

torch.manual_seed(42)

means = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=torch.float).view(2, -1)
log_stds  = torch.tensor([[-20, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=torch.float).view(2, -1)

means.requires_grad = True
log_stds.requires_grad = True

stds = torch.exp(log_stds)
print(stds)

dist = Independent(Normal(loc=means, scale=stds), 1)
# dist = Normal(loc=means, scale=stds)

print(dist.batch_shape, dist.event_shape)  # torch.Size([2]) torch.Size([5])

samples_with_grad = dist.rsample(sample_shape=torch.Size([]))

print(samples_with_grad.shape)  # torch.Size([2, 5])

log_prob = - dist.log_prob(samples_with_grad)

print(log_prob)

result = torch.sum(log_prob)
result.backward()

print(means.grad)
print(log_stds.grad)

# tensor([[0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.]])
# tensor([[1.0000, 0.5000, 0.3333, 0.2500, 0.2000],
#         [1.0000, 0.5000, 0.3333, 0.2500, 0.2000]])


