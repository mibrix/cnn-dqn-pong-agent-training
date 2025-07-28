# Deep Q-Network (DQN) for Pong

## I. Introduction

This project implements the Deep Q-Network (DQN) introduced by Minh et al. (2015) [1] to learn how to play the classic arcade game Pong.

Deep Q-Network is an advanced reinforcement learning algorithm that merges Q-learning with deep convolutional neural networks (CNNs). This allows it to learn effective policies directly from high-dimensional sensory inputs, such as raw pixel data. DQN incorporates two key innovations to enhance learning stability and efficiency: **experience replay** and a **target network**.

* **Experience Replay**: The agent's experiences (state $s_t$, action $a_t$, reward $r_t$, next state $s_{t+1}$) are stored in a replay memory $D$. During training, mini-batches of experiences are randomly sampled from $D$. This process decorrelates consecutive experiences, reduces variance in updates, and prevents divergence often seen with correlated data, thereby stabilizing training.

* **Target Network**: A separate network, $Q'$ with parameters $\theta^-$, is used to generate stable target Q-values. The parameters $\theta^-$ are periodically copied from the main Q-network $Q$ (with parameters $\theta$) every $C$ iterations and remain fixed between updates. This strategy mitigates instability and divergence caused by rapidly fluctuating target values in Q-learning updates.

The DQN algorithm approximates the optimal action-value function $Q(s, a)$ using a CNN. Q-value updates are governed by the Bellman equation:

$$Q(s, a; \theta) \leftarrow Q(s, a; \theta) + \alpha (r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))$$

where:
* $Q(s, a; \theta)$ is the action-value function parameterized by $\theta$.
* $\alpha$ is the learning rate.
* $\gamma$ is the discount factor.
* $r$ is the reward received after taking action $a$ in state $s$.
* $s'$ is the subsequent state.
* $a'$ is the action taken in state $s'$.
* $\theta^-$ are the parameters of the target network.

The neural network architecture for DQN takes an $84 \times 84 \times 4$ preprocessed image stack as input, representing the current state. This is followed by:
* Three convolutional layers:
    * 32 filters of size $8 \times 8$ with stride 4.
    * 64 filters of size $4 \times 4$ with stride 2.
    * 64 filters of size $3 \times 3$ with stride 1.
    * Each convolutional layer is followed by a Rectified Linear Unit (ReLU) activation.
* Two fully connected layers:
    * 512 units with ReLU activation.
    * An output layer with a single output for each valid action, representing the Q-values.

The agent uses an $\epsilon$-greedy policy, choosing a random action with probability $\epsilon$ and the action that maximizes the Q-value with probability $1 - \epsilon$. As training progresses, $\epsilon$ linearly decreases from a high initial value to a lower fixed value, balancing exploration (trying new actions) and exploitation (using known good actions).

The network is trained using stochastic gradient descent with the Adam optimizer, minimizing the mean-squared error between predicted Q-values and target Q-values.

## B. Pong

### 1) Theoretical Introduction

To effectively track the ball's velocity, a convolutional neural network (CNN) is employed, receiving four consecutive frames as input. Thus, each "datapoint" consists of four consecutive images, all rewarded and stored in memory. The architecture is based on the convolutional neural network described in the Nature DQN paper [1].

Necessary code modifications included ensuring the input tensor to the CNN was appropriately dimensioned as `[1, number of pictures, height, width]` to accommodate four consecutive $84 \times 84$ images. The leading dimension of 1 is crucial for correct neural network input formatting. These adjustments were applied to both `train.py` and `evaluate.py`. Additionally, due to differences in actions compared to the cart pole model, actions 0 and 1 were converted to 2 and 3.

The following hyperparameters were adhered to as suggested:

| Hyperparameter            | Value    |
| :------------------------ | :------- |
| Observation stack size    | 4        |
| Replay memory capacity    | 10000    |
| Batch size                | 32       |
| Target update frequency   | 1000     |
| Training frequency        | 4        |
| Discount factor           | 0.99     |
| Learning rate             | 1e-4     |
| Initial epsilon           | 1.0      |
| Final epsilon             | 0.01     |
| Anneal length             | $10^6$   |

### 2) Results

Training was conducted multiple times in Google Colab. An initial CPU-only attempt achieved 485 training episodes in approximately 11 hours. Switching to GPU significantly reduced this to about 1 hour and 10 minutes for the same number of episodes.

Despite encountering interruptions, the model that achieved the highest number of episodes, totaling 503, was preserved. Examination of the training graph illustrates a positive correlation between the number of episodes and the mean return, suggesting further improvement could be achieved with extended training. The absence of overfitting indications also supports the potential for enhanced results with additional training.

Evaluation during training was performed every 25th episode.

**Fig. 4. Training performance**
![Training Performance](training_performance.png)

The graph above illustrates the evaluation results of the best-performing model recorded during training. The model achieved a mean reward of 12.2 over 10 evaluation episodes.

**Fig. 5. Evaluation performance**
![Evaluation Performance](evaluation_performance.png)

**A video demonstrating the agent beating the "standard pong player" without letting a single point to him is available!**

📺 [Watch the agent's flawless victory!](test-video-episode-600.mp4)

This video was saved after the 500th training episode, capturing the agent's impressive performance where it outperformed its opponent by 21 points.

## III. Discussion

For future work on Pong, it would be beneficial to train the model without limits on the number of rounds. Additionally, exploring different hyperparameters, such as a lower learning rate (e.g., less than 1e-4), holds promise for achieving even better results by reducing the likelihood of overshooting the minimum.

## References

[1] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis, “Human-level control through deep reinforcement learning,” *Nature*, vol. 518, no. 7540, pp. 529–533, Feb 2015. [Online]. Available: https://doi.org/10.1038/nature14236
