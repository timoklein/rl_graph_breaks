# Examples torch.compile graph breaks in RL code

Code is based on [cleanRL's SAC-discrete implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_atari.py).

## `sac_atari_breaks_simplified.py`

A simple example where `Categorical.sample()` breaks the graph. Minimal dependencies: Only `pytorch==2.0.1`.

### Explain printout for `critic_update`

Here the graph breaks twice, both times at `policy_dist.sample()`. Note that I'm not trying to backprop through the sampling operation.

```python
('Dynamo produced 3 graphs with 2 graph break and 31 ops\n'
 ' Break reasons: \n'
 '\n'
 '1. call_function ConstantVariable(NoneType) [ConstantVariable(int), '
 'ConstantVariable(int)] {}\n'
 '  File "/media/data/timok34dm/cleanrl/cleanrl/sac_atari_breaks.py", line 91, '
 'in critic_update\n'
 '    _, next_state_log_pi, next_state_action_probs = '
 'actor.get_action(data.next_observations)\n'
 '  File "/media/data/timok34dm/cleanrl/cleanrl/sac_atari_breaks.py", line 82, '
 'in get_action\n'
 '    action = policy_dist.sample()\n'
 '  File '
 '"/export/home/timok34dm/mambaforge/envs/cleanrlt/lib/python3.10/site-packages/torch/distributions/categorical.py", '
 'line 117, in sample\n'
 '    probs_2d = self.probs.reshape(-1, self._num_events)\n'
 ' \n'
 '2. step_unsupported\n'
 '  File "/media/data/timok34dm/cleanrl/cleanrl/sac_atari_breaks.py", line 82, '
 'in get_action\n'
 '    action = policy_dist.sample()\n'
 ' \n'
 '3. inline in skipfiles: Optimizer.zero_grad  | _fn '
 '/export/home/timok34dm/mambaforge/envs/cleanrlt/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py\n'
 '  File "/media/data/timok34dm/cleanrl/cleanrl/sac_atari_breaks.py", line '
 '111, in <graph break in critic_update>\n'
 '    q_optimizer.zero_grad(True)\n'
 ' \n'
 'TorchDynamo compilation metrics:\n'
 'Function, Runtimes (s)\n'
 '_compile, 0.1561, 0.0490, 0.1694, 0.0122, 0.0357, 0.0067, 0.0050, 0.0307, '
 '0.1172, 0.0008, 0.0063\n'
 'OutputGraph.call_user_compiler, 0.0000, 0.0000, 0.0001')
```

### Explain printout for `actor_update`

Graph breaks at `policy_dist.sample()` again.

```python
('Dynamo produced 2 graphs with 1 graph break and 16 ops\n'
 ' Break reasons: \n'
 '\n'
 '1. step_unsupported\n'
 '  File "/media/data/timok34dm/cleanrl/cleanrl/sac_atari_breaks.py", line 82, '
 'in get_action\n'
 '    action = policy_dist.sample()\n'
 ' \n'
 '2. inline in skipfiles: Optimizer.zero_grad  | _fn '
 '/export/home/timok34dm/mambaforge/envs/cleanrlt/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py\n'
 '  File "/media/data/timok34dm/cleanrl/cleanrl/sac_atari_breaks.py", line '
 '126, in <graph break in actor_update>\n'
 '    actor_optimizer.zero_grad(True)\n'
 ' \n'
 'TorchDynamo compilation metrics:\n'
 'Function, Runtimes (s)\n'
 '_compile, 0.0550, 0.0548, 0.0825, 0.0071, 0.0120, 0.0061, 0.0050, 0.0176, '
 '0.0648, 0.0007, 0.0032\n'
 'OutputGraph.call_user_compiler, 0.0000, 0.0001')
```

### Explain printout for `alpha_update`

Graph breaks when computing `alpha = log_alpha.exp()` before returning it. Can be fixed easily.

```python
('Dynamo produced 2 graphs with 1 graph break and 4 ops\n'
 ' Break reasons: \n'
 '\n'
 '1. inline in skipfiles: Optimizer.zero_grad  | _fn '
 '/export/home/timok34dm/mambaforge/envs/cleanrlt/lib/python3.10/site-packages/torch/_dynamo/eval_frame.py\n'
 '  File "/media/data/timok34dm/cleanrl/cleanrl/sac_atari_breaks.py", line '
 '135, in alpha_update\n'
 '    a_optimizer.zero_grad(True)\n'
 ' \n'
 '2. return_value\n'
 '  File "/media/data/timok34dm/cleanrl/cleanrl/sac_atari_breaks.py", line '
 '140, in <graph break in alpha_update>\n'
 '    return alpha, alpha_loss\n'
 ' \n'
 'TorchDynamo compilation metrics:\n'
 'Function, Runtimes (s)\n'
 '_compile, 0.0145, 0.0045, 0.0097, 0.0060, 0.0050, 0.0051, 0.0177, 0.0007, '
 '0.0050\n'
 'OutputGraph.call_user_compiler, 0.0000, 0.0000')
```

## `sac_atari_breaks_full.py`

Full algorithm code with gym interaction loop, logging, etc.
