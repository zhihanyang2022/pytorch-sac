# Soft actor-critic

### What is SAC?

TODO: algorithm screenshto and link from spinning up

### How to perform step 14?

What confused me initially $`a^2+b^2=c^2`$

```{python}
import torch
from torch.distributions import Normal, Independent

means = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=torch.float).view(2, -1)
stds  = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=torch.float).view(2, -1)

means.requires_grad = True
stds.requires_grad = True

dist = Independent(Normal(loc=means, scale=stds), 1)

print(dist.batch_shape, dist.event_shape)  # torch.Size([2]) torch.Size([5])

samples_with_grad = dist.rsample(sample_shape=torch.Size([]))

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
