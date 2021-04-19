# Soft actor-critic

### The SAC algorithm

TODO: algorithm screenshto and link from spinning up

#### How to calculate <img src="https://render.githubusercontent.com/render/math?math=\log \pi_{\theta} (\tilde{a}_\theta(s)\mid s)"> properly?

What confused me initially in this step is that, in the entropy term (the second term), both the action and the log-probability of the action depends on parameter theta. How can this be? To better understand this, I wrote the following snippet.

Basically, you can see that the gradient with respect to the means 

```python
import torch
from torch.distributions import Normal, Independent

means = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=torch.float).view(2, -1)  # two mean vectors
stds  = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=torch.float).view(2, -1)  # two std vectors

means.requires_grad = True
stds.requires_grad = True

dist = Independent(Normal(loc=means, scale=stds), 1)

print(dist.batch_shape, dist.event_shape)  # torch.Size([2]) torch.Size([5])

samples_with_grad = dist.rsample()  # rsample using reparameterization trick; use sample instead if you don't want samples to be back

print(samples_with_grad.shape)  # torch.Size([2, 5])

log_prob = - dist.log_prob(samples_with_grad)
result = torch.sum(log_prob)
result.backward()

print(means.grad)
print(stds.grad)

# tensor([[0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.]])
# tensor([[1.0000, 0.5000, 0.3333, 0.2500, 0.2000],
#         [1.0000, 0.5000, 0.3333, 0.2500, 0.2000]])
```
