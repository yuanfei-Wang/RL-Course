# DQN_2048

The gym environment of 2048 should be used as:

```
import gym2048
env = gym.make("Env2048-v0")
```



### PPO & A2C

```bash
cd PPO-batch
bash train2048.sh
```



### MCTS

For MCTS&N-Tuple experiment:

```bash
python mcts-test.py
```

Due to the large size of tuple.npy, we didn't upload the file. So the command will not work.
