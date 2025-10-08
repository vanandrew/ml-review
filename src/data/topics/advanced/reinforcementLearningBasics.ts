import { Topic } from '../../../types';

export const reinforcementLearningBasics: Topic = {
  id: 'reinforcement-learning-basics',
  title: 'Reinforcement Learning Basics',
  category: 'advanced',
  description: 'Learning through interaction with an environment to maximize rewards',
  content: `
    <h2>Reinforcement Learning: Learning from Interaction</h2>
    <p>Reinforcement Learning (RL) represents a fundamentally different paradigm from supervised and unsupervised learning. Instead of learning from labeled examples, an RL agent learns by directly interacting with an environment, receiving feedback in the form of rewards, and discovering through trial and error which actions lead to desirable outcomes. This interactive learning framework mirrors how humans and animals learn—through consequences of actions rather than explicit instruction. RL has achieved remarkable successes, from mastering complex games like Go and StarCraft to controlling robotic systems and optimizing data center efficiency. The field combines ideas from optimal control, dynamic programming, temporal difference learning, and function approximation to tackle the challenge of sequential decision-making under uncertainty.</p>

    <h3>The Reinforcement Learning Framework</h3>

    <h4>The Agent-Environment Interaction Loop</h4>
    <p>At each time step t:</p>
    <ol>
      <li><strong>Agent observes state s_t:</strong> Receives information about environment's current situation</li>
      <li><strong>Agent selects action a_t:</strong> Based on its policy π(a|s)</li>
      <li><strong>Environment transitions:</strong> Moves to new state s_{t+1} according to dynamics</li>
      <li><strong>Agent receives reward r_{t+1}:</strong> Scalar feedback signal</li>
      <li><strong>Loop repeats:</strong> Process continues, agent learns from experience</li>
    </ol>

    <h5>Visual Representation</h5>
    <pre>
    Agent                       Environment
      │                            │
      │  ──────  State s_t  ─────► │
      │ ◄─────  Reward r_t  ─────  │
      │                            │
      │  ──────  Action a_t ─────► │
      │                            │
      │  ◄────  State s_{t+1} ──── │
      │  ◄───  Reward r_{t+1} ──── │
      │                            │
      └────────────────────────────┘
    </pre>

    <h4>Key Components Defined</h4>

    <h5>1. State (s ∈ S)</h5>
    <ul>
      <li><strong>Definition:</strong> Complete description of environment at time t</li>
      <li><strong>Markov property:</strong> State contains all information needed to predict future (no hidden dependencies on past)</li>
      <li><strong>State space S:</strong> Set of all possible states (discrete or continuous)</li>
      <li><strong>Examples:</strong> Chess board position, robot joint angles and velocities, pixel observations</li>
      <li><strong>Observation vs state:</strong> Agent may observe o_t (partial information) rather than full state s_t</li>
    </ul>

    <h5>2. Action (a ∈ A)</h5>
    <ul>
      <li><strong>Definition:</strong> Decision or control available to agent</li>
      <li><strong>Action space A(s):</strong> Set of actions available in state s</li>
      <li><strong>Discrete actions:</strong> Finite set (e.g., {Up, Down, Left, Right})</li>
      <li><strong>Continuous actions:</strong> Real-valued vectors (e.g., robot motor torques)</li>
      <li><strong>Constraints:</strong> Some actions only valid in certain states</li>
    </ul>

    <h5>3. Reward (r ∈ ℝ)</h5>
    <ul>
      <li><strong>Definition:</strong> Scalar feedback signal indicating immediate desirability of action</li>
      <li><strong>Timing:</strong> Received after taking action in state</li>
      <li><strong>Reward function:</strong> R(s, a, s') = expected immediate reward</li>
      <li><strong>Reward hypothesis:</strong> All goals can be expressed as maximizing cumulative reward</li>
      <li><strong>Sparse vs dense:</strong> Reward every step vs only at goal</li>
    </ul>

    <h5>4. Policy (π)</h5>
    <ul>
      <li><strong>Definition:</strong> Agent's strategy for selecting actions</li>
      <li><strong>Deterministic policy:</strong> a = π(s), map state to single action</li>
      <li><strong>Stochastic policy:</strong> π(a|s) = P(A_t=a | S_t=s), probability distribution over actions</li>
      <li><strong>Goal:</strong> Find optimal policy π* that maximizes expected return</li>
      <li><strong>Representation:</strong> Table (small spaces), neural network (large spaces)</li>
    </ul>

    <h5>5. Transition Dynamics (P)</h5>
    <ul>
      <li><strong>Definition:</strong> P(s'|s,a) = probability of reaching state s' after action a in state s</li>
      <li><strong>Model-based:</strong> Agent learns/knows P and R</li>
      <li><strong>Model-free:</strong> Agent doesn't learn dynamics, learns directly from experience</li>
      <li><strong>Stochastic:</strong> Same action can lead to different outcomes (inherent randomness)</li>
    </ul>

    <h3>Core Concepts in Reinforcement Learning</h3>

    <h4>Return: Cumulative Reward</h4>
    <p><strong>Objective:</strong> Maximize expected return, not just immediate reward.</p>

    <h5>Discounted Return</h5>
    <p style="text-align: center; font-size: 1.1em;">
      $G_t = R_{t+1} + \\gamma R_{t+2} + \\gamma^2 R_{t+3} + ... = \\sum_{k=0}^{\\infty} \\gamma^k R_{t+k+1}$
    </p>

    <h5>Discount Factor $\\gamma \\in [0, 1]$</h5>
    <ul>
      <li><strong>$\\gamma = 0$:</strong> Myopic agent, only cares about immediate reward</li>
      <li><strong>$\\gamma$ close to 1:</strong> Far-sighted agent, considers long-term consequences</li>
      <li><strong>$\\gamma = 1$:</strong> Undiscounted (only for episodic tasks with guaranteed termination)</li>
      <li><strong>Typical values:</strong> 0.9, 0.95, 0.99</li>
      <li><strong>Mathematical necessity:</strong> $\\gamma < 1$ ensures infinite sum converges</li>
      <li><strong>Economic interpretation:</strong> Reward now worth more than reward later (time value)</li>
    </ul>

    <h4>Value Functions: Expected Returns</h4>

    <h5>State-Value Function V^π(s)</h5>
    <p><strong>Definition:</strong> Expected return starting from state s, following policy π</p>
    <p style="text-align: center;">
      $V^\\pi(s) = \\mathbb{E}_\\pi[G_t | S_t = s] = \\mathbb{E}_\\pi[\\sum \\gamma^k R_{t+k+1} | S_t = s]$
    </p>
    <ul>
      <li><strong>Interpretation:</strong> "How good is it to be in state s under policy π?"</li>
      <li><strong>Higher V(s):</strong> More desirable state</li>
      <li><strong>Policy evaluation:</strong> Computing V^π for given π</li>
    </ul>

    <h5>Action-Value Function Q^π(s, a)</h5>
    <p><strong>Definition:</strong> Expected return starting from state s, taking action a, then following π</p>
    <p style="text-align: center;">
      $Q^\\pi(s, a) = \\mathbb{E}_\\pi[G_t | S_t = s, A_t = a]$
    </p>
    <ul>
      <li><strong>Interpretation:</strong> "How good is it to take action a in state s, then follow π?"</li>
      <li><strong>Action selection:</strong> Choose $a = \\arg\\max_a Q(s, a)$</li>
      <li><strong>Relationship:</strong> $V^\\pi(s) = \\mathbb{E}_{a\\sim\\pi}[Q^\\pi(s,a)]$</li>
    </ul>

    <h4>Bellman Equations: Recursive Structure</h4>

    <h5>Visual: Bellman Backup Diagram</h5>
    <pre class="code-block">
Bellman Backup: How value propagates backward

Current State s:
     V(s)
       │
  [π(s) chooses action a]
       │
    r + γ·V(s')
       │
       ▼
  Next State s'

One-Step Lookahead:
V(s) = ∑ π(a|s) ∑ P(s'|s,a)[R(s,a,s') + γ·V(s')]
     │         │        │              │
  policy  transition immediate   discounted
          dynamics   reward      future value
    </pre>

    <h5>Example: Simple GridWorld</h5>
    <pre class="code-block">
GridWorld (4x4):
┌───┬───┬───┬───┐
│ S │ -1│ -1│ -1│  S = Start
├───┼───┼───┼───┤  G = Goal (+10)
│ -1│ XX│ -1│ -1│  XX = Wall
├───┼───┼───┼───┤  -1 = Step cost
│ -1│ -1│ -1│ -1│
├───┼───┼───┼───┤
│ -1│ -1│ -1│ G │
└───┴───┴───┴───┘

Optimal Values (with γ=0.9):
┌────┬────┬────┬────┐
│ 6.1│ 7.3│ 8.1│ 7.3│
├────┼────┼────┼────┤
│ 5.4│ XX │ 9.0│ 8.1│  Values decrease
├────┼────┼────┼────┤  with distance
│ 4.9│ 5.4│ 7.3│ 9.0│  from goal
├────┼────┼────┼────┤
│ 4.4│ 4.9│ 6.1│ 10 │
└────┴────┴────┴────┘

Optimal Policy (↑↓←→):
┌───┬───┬───┬───┐
│ → │ → │ ↓ │ ↓ │
├───┼───┼───┼───┤
│ ↑ │ XX│ ↓ │ ↓ │  All paths
├───┼───┼───┼───┤  lead to goal
│ ↑ │ → │ → │ ↓ │
├───┼───┼───┼───┤
│ ↑ │ → │ → │ G │
└───┴───┴───┴───┘
    </pre>

    <h5>Bellman Expectation Equation</h5>
    <p>Value functions satisfy recursive relationships:</p>

    <h6>For V^π:</h6>
    <p style="text-align: center;">
      $V^\\pi(s) = \\mathbb{E}_\\pi[R_{t+1} + \\gamma V^\\pi(S_{t+1}) | S_t = s]$
    </p>
    <p>Current value = immediate reward + discounted future value</p>

    <h6>For Q^π:</h6>
    <p style="text-align: center;">
      $Q^\\pi(s, a) = \\mathbb{E}[R_{t+1} + \\gamma Q^\\pi(S_{t+1}, A_{t+1}) | S_t=s, A_t=a]$
    </p>

    <h5>Bellman Optimality Equation</h5>
    <p>Optimal value functions satisfy:</p>

    <h6>Optimal State-Value:</h6>
    <p style="text-align: center;">
      $V^*(s) = \\max_a \\mathbb{E}[R_{t+1} + \\gamma V^*(S_{t+1}) | S_t=s, A_t=a]$
    </p>

    <h6>Optimal Action-Value:</h6>
    <p style="text-align: center;">
      $Q^*(s, a) = \\mathbb{E}[R_{t+1} + \\gamma \\max_{a\'} Q^*(S_{t+1}, a\') | S_t=s, A_t=a]$
    </p>
    <ul>
      <li><strong>Key insight:</strong> Optimal policy π* takes action that maximizes Q*(s,a)</li>
      <li><strong>Fixed point:</strong> Bellman optimality is fixed point equation</li>
    </ul>

    <h3>The Exploration-Exploitation Dilemma</h3>

    <h4>The Trade-off</h4>
    <ul>
      <li><strong>Exploitation:</strong> Choose action known to yield high reward (use current knowledge)</li>
      <li><strong>Exploration:</strong> Try new actions to discover potentially better options (gather more information)</li>
      <li><strong>Dilemma:</strong> Can't do both simultaneously—must balance</li>
      <li><strong>Consequence:</strong> Too much exploitation → stuck in suboptimal local optimum; too much exploration → never leverage good strategies</li>
    </ul>

    <h4>Exploration Strategies</h4>

    <h5>1. ε-Greedy</h5>
    <ul>
      <li><strong>Mechanism:</strong> With probability ε explore (random action), with 1-ε exploit (best known action)</li>
      <li><strong>Simple and effective:</strong> Most common approach</li>
      <li><strong>ε decay:</strong> Start high (e.g., 0.5), decay to low value (0.01) as training progresses</li>
      <li><strong>Pros:</strong> Easy to implement, guarantees exploration</li>
      <li><strong>Cons:</strong> Explores uniformly (doesn't consider action quality)</li>
    </ul>

    <h5>2. Softmax / Boltzmann Exploration</h5>
    <ul>
      <li><strong>Mechanism:</strong> Select actions probabilistically based on Q-values: $\\pi(a|s) \\propto \\exp(Q(s,a)/\\tau)$</li>
      <li><strong>Temperature $\\tau$:</strong> Controls randomness (high $\\tau$ → uniform, low $\\tau$ → greedy)</li>
      <li><strong>Pros:</strong> Better actions explored more often</li>
      <li><strong>Cons:</strong> Sensitive to Q-value scale</li>
    </ul>

    <h5>3. Upper Confidence Bound (UCB)</h5>
    <ul>
      <li><strong>Principle:</strong> Optimism in face of uncertainty—prefer actions with uncertain values</li>
      <li><strong>Bonus term:</strong> Select $a = \\arg\\max_a [Q(s,a) + c\\sqrt{\\ln t / N(s,a)}]$</li>
      <li><strong>Exploration bonus:</strong> Higher for less-visited actions</li>
      <li><strong>Theoretical guarantees:</strong> Logarithmic regret bounds</li>
    </ul>

    <h5>4. Thompson Sampling</h5>
    <ul>
      <li><strong>Bayesian approach:</strong> Maintain distribution over Q-values, sample from posterior</li>
      <li><strong>Naturally balances:</strong> Exploration proportional to uncertainty</li>
      <li><strong>Effective:</strong> Often outperforms simpler strategies</li>
    </ul>

    <h3>Core RL Algorithms</h3>

    <h4>Dynamic Programming (Model-Based)</h4>

    <h5>Policy Iteration</h5>
    <ol>
      <li><strong>Policy Evaluation:</strong> Compute $V^\\pi$ for current policy $\\pi$ (solve Bellman expectation)</li>
      <li><strong>Policy Improvement:</strong> Update policy: $\\pi(s) = \\arg\\max_a Q^\\pi(s,a)$</li>
      <li><strong>Repeat:</strong> Until policy converges</li>
      <li><strong>Guarantee:</strong> Converges to optimal policy $\\pi^*$</li>
    </ol>

    <h5>Value Iteration</h5>
    <ul>
      <li><strong>Direct optimization:</strong> Iterate Bellman optimality: $V(s) \\leftarrow \\max_a \\mathbb{E}[R + \\gamma V(s')]$</li>
      <li><strong>Single pass:</strong> Combines evaluation and improvement</li>
      <li><strong>Converges to $V^*$:</strong> Extract optimal policy $\\pi^*(s) = \\arg\\max_a Q^*(s,a)$</li>
    </ul>

    <h4>Monte Carlo Methods (Model-Free)</h4>

    <h5>Principle</h5>
    <ul>
      <li><strong>Learn from complete episodes:</strong> Sample full trajectories, observe returns</li>
      <li><strong>Average returns:</strong> Estimate V(s) or Q(s,a) by averaging observed returns</li>
      <li><strong>No bootstrapping:</strong> Don't rely on value estimates of other states</li>
      <li><strong>High variance:</strong> Return depends on entire trajectory</li>
    </ul>

    <h5>Monte Carlo Control</h5>
    <ol>
      <li><strong>Generate episode:</strong> Follow policy π, record states, actions, rewards</li>
      <li><strong>For each (s,a) in episode:</strong> Update $Q(s,a)$ toward observed return $G_t$</li>
      <li><strong>Improve policy:</strong> $\\pi(s) = \\arg\\max_a Q(s,a)$</li>
      <li><strong>Repeat:</strong> Generate new episodes, converge to optimal policy</li>
    </ol>

    <h4>Temporal Difference (TD) Learning (Model-Free)</h4>

    <h5>Core Idea: Bootstrap</h5>
    <ul>
      <li><strong>Update immediately:</strong> After each step, don't wait for episode end</li>
      <li><strong>TD target:</strong> $R_{t+1} + \\gamma V(S_{t+1})$ (estimate return using current V estimate)</li>
      <li><strong>TD error:</strong> $\\delta_t = R_{t+1} + \\gamma V(S_{t+1}) - V(S_t)$</li>
      <li><strong>Update:</strong> $V(S_t) \\leftarrow V(S_t) + \\alpha \\delta_t$</li>
      <li><strong>Combines MC and DP:</strong> Samples like MC, bootstraps like DP</li>
    </ul>

    <h5>SARSA (On-Policy TD Control)</h5>
    <ul>
      <li><strong>Algorithm name:</strong> State-Action-Reward-State-Action</li>
      <li><strong>On-policy:</strong> Learn Q for policy being followed</li>
      <li><strong>Update:</strong> $Q(S_t, A_t) \\leftarrow Q(S_t, A_t) + \\alpha[R_{t+1} + \\gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$</li>
      <li><strong>Action selection:</strong> $A_{t+1}$ actually taken from policy (e.g., $\\varepsilon$-greedy)</li>
      <li><strong>Conservative:</strong> Learns safe policy accounting for exploration</li>
    </ul>

    <h5>Q-Learning (Off-Policy TD Control)</h5>
    <ul>
      <li><strong>Off-policy:</strong> Learn optimal $Q^*$ while following exploratory policy</li>
      <li><strong>Update:</strong> $Q(S_t, A_t) \\leftarrow Q(S_t, A_t) + \\alpha[R_{t+1} + \\gamma \\max_a Q(S_{t+1}, a) - Q(S_t, A_t)]$</li>
      <li><strong>Key difference:</strong> Uses $\\max_a Q(S_{t+1}, a)$ regardless of action actually taken</li>
      <li><strong>Aggressive:</strong> Learns optimal policy, assumes greedy actions even if exploring</li>
      <li><strong>Convergence:</strong> Guaranteed to find optimal $Q^*$ under certain conditions</li>
    </ul>

    <h5>Comparison: SARSA vs Q-Learning</h5>
    <table >
      <tr>
        <th>Aspect</th>
        <th>SARSA (On-Policy)</th>
        <th>Q-Learning (Off-Policy)</th>
      </tr>
      <tr>
        <td>Update Rule</td>
        <td>Uses actual next action $A_{t+1}$</td>
        <td>Uses $\\max_a Q(s',a)$</td>
      </tr>
      <tr>
        <td>Policy Learned</td>
        <td>Policy being followed ($\\varepsilon$-greedy)</td>
        <td>Optimal policy (greedy)</td>
      </tr>
      <tr>
        <td>Behavior</td>
        <td>Conservative, risk-aware</td>
        <td>Aggressive, risk-seeking</td>
      </tr>
      <tr>
        <td>Convergence</td>
        <td>To policy being followed</td>
        <td>To optimal policy</td>
      </tr>
      <tr>
        <td>Cliff Walking Example</td>
        <td>Learns safe path away from cliff</td>
        <td>Learns risky path near cliff (optimal but dangerous during learning)</td>
      </tr>
      <tr>
        <td>Use Case</td>
        <td>When exploration is risky</td>
        <td>When want optimal policy</td>
      </tr>
    </table>

    <h4>Deep Reinforcement Learning</h4>

    <h5>Deep Q-Networks (DQN)</h5>
    <ul>
      <li><strong>Function approximation:</strong> Use neural network to approximate $Q(s,a; \\theta)$</li>
      <li><strong>Handles large state spaces:</strong> Images, continuous states</li>
      <li><strong>Challenge:</strong> Correlated data, non-stationary targets cause instability</li>
    </ul>

    <h6>DQN Innovations</h6>
    <ol>
      <li><strong>Experience Replay:</strong> Store transitions in buffer, sample random minibatches for training (breaks correlation)</li>
      <li><strong>Target Network:</strong> Separate network $\\hat{Q}$ for targets, updated periodically (stabilizes learning)</li>
      <li><strong>Loss:</strong> $L(\\theta) = \\mathbb{E}[(R + \\gamma \\max_{a'} \\hat{Q}(s',a'; \\theta^-) - Q(s,a; \\theta))^2]$</li>
      <li><strong>Breakthrough:</strong> Played Atari games from raw pixels at human level</li>
    </ol>

    <h5>Policy Gradient Methods</h5>
    <ul>
      <li><strong>Direct policy optimization:</strong> Parameterize policy $\\pi(a|s; \\theta)$, optimize $\\theta$ directly</li>
      <li><strong>Objective:</strong> $J(\\theta) = \\mathbb{E}_\\pi[G_t]$, maximize expected return</li>
      <li><strong>Policy gradient:</strong> $\\nabla_\\theta J(\\theta) = \\mathbb{E}_\\pi[\\nabla_\\theta \\log \\pi(a|s; \\theta) Q^\\pi(s,a)]$</li>
      <li><strong>REINFORCE algorithm:</strong> Monte Carlo estimate of gradient</li>
      <li><strong>Advantages:</strong> Handles continuous actions, stochastic policies, better convergence properties</li>
    </ul>

    <h5>Actor-Critic Methods</h5>
    <ul>
      <li><strong>Hybrid approach:</strong> Combine policy gradient (actor) with value function (critic)</li>
      <li><strong>Actor:</strong> Policy network $\\pi(a|s; \\theta)$, updated via policy gradient</li>
      <li><strong>Critic:</strong> Value network $V(s; w)$, estimates returns (reduces variance)</li>
      <li><strong>Advantage:</strong> $A(s,a) = Q(s,a) - V(s)$, measures how good action is relative to average</li>
      <li><strong>Update actor:</strong> $\\nabla_\\theta J \\approx \\nabla_\\theta \\log \\pi(a|s; \\theta) A(s,a)$</li>
      <li><strong>Examples:</strong> A3C, PPO, SAC (state-of-the-art algorithms)</li>
    </ul>

    <h3>Applications of Reinforcement Learning</h3>

    <h4>Game Playing</h4>
    <ul>
      <li><strong>Atari games:</strong> DQN achieves human-level performance from pixels</li>
      <li><strong>Go:</strong> AlphaGo defeats world champions using RL + tree search</li>
      <li><strong>Chess, Shogi:</strong> AlphaZero learns from self-play, surpasses human knowledge</li>
      <li><strong>Poker:</strong> Pluribus defeats top professionals in multi-player Texas Hold'em</li>
      <li><strong>StarCraft II:</strong> AlphaStar reaches Grandmaster level in real-time strategy</li>
      <li><strong>Dota 2:</strong> OpenAI Five competes with professional players</li>
    </ul>

    <h4>Robotics</h4>
    <ul>
      <li><strong>Locomotion:</strong> Learning to walk, run, backflip (simulated and real robots)</li>
      <li><strong>Manipulation:</strong> Grasping, insertion tasks, dexterous hand control</li>
      <li><strong>Navigation:</strong> Obstacle avoidance, path planning</li>
      <li><strong>Sim-to-real transfer:</strong> Train in simulation, deploy on hardware</li>
    </ul>

    <h4>Autonomous Vehicles</h4>
    <ul>
      <li><strong>Driving policy:</strong> Lane keeping, lane changing, merging</li>
      <li><strong>Traffic light control:</strong> Optimize flow in urban networks</li>
      <li><strong>Routing:</strong> Dynamic path planning</li>
    </ul>

    <h4>Resource Management</h4>
    <ul>
      <li><strong>Data center cooling:</strong> Google DeepMind reduces energy by 40%</li>
      <li><strong>Power grid optimization:</strong> Load balancing, demand response</li>
      <li><strong>Job scheduling:</strong> Cluster resource allocation</li>
    </ul>

    <h4>Recommendation Systems</h4>
    <ul>
      <li><strong>Sequential recommendations:</strong> Model user interaction as MDP</li>
      <li><strong>Long-term engagement:</strong> Optimize for sustained user satisfaction, not just clicks</li>
    </ul>

    <h4>Finance</h4>
    <ul>
      <li><strong>Algorithmic trading:</strong> Learn trading strategies from market data</li>
      <li><strong>Portfolio management:</strong> Dynamic asset allocation</li>
      <li><strong>Option pricing:</strong> Hedging strategies</li>
    </ul>

    <h3>Challenges in Reinforcement Learning</h3>

    <h4>Sample Efficiency</h4>
    <ul>
      <li><strong>Problem:</strong> RL often requires millions of interactions (expensive, time-consuming)</li>
      <li><strong>Especially difficult:</strong> Real-world applications with costly interactions (robotics)</li>
      <li><strong>Solutions:</strong> Model-based RL, transfer learning, imitation learning, offline RL</li>
    </ul>

    <h4>Credit Assignment</h4>
    <ul>
      <li><strong>Temporal credit assignment:</strong> Which action (among many) led to eventual reward?</li>
      <li><strong>Structural credit assignment:</strong> Which features/factors were relevant?</li>
      <li><strong>Solutions:</strong> Value functions, eligibility traces, hindsight experience replay</li>
    </ul>

    <h4>Sparse Rewards</h4>
    <ul>
      <li><strong>Problem:</strong> Reward only received after long sequence of actions (e.g., winning game)</li>
      <li><strong>Exploration difficulty:</strong> Hard to discover rewarding behavior</li>
      <li><strong>Solutions:</strong> Reward shaping, curiosity-driven exploration, hierarchical RL</li>
    </ul>

    <h4>Partial Observability</h4>
    <ul>
      <li><strong>POMDP:</strong> Agent doesn't observe full state, only partial observations</li>
      <li><strong>Challenge:</strong> History matters, need memory</li>
      <li><strong>Solutions:</strong> Recurrent networks (LSTM), attention mechanisms, belief states</li>
    </ul>

    <h4>Training Stability</h4>
    <ul>
      <li><strong>Non-stationarity:</strong> Target changes as policy improves</li>
      <li><strong>High variance:</strong> Gradient estimates noisy</li>
      <li><strong>Catastrophic forgetting:</strong> Policy can suddenly degrade</li>
      <li><strong>Solutions:</strong> Target networks, trust region methods (PPO, TRPO), experience replay</li>
    </ul>

    <h3>The Future of Reinforcement Learning</h3>
    <p>Reinforcement learning stands at the frontier of AI, offering a framework for agents to learn complex behaviors through experience. Recent advances—PPO for stable policy optimization, model-based RL for sample efficiency, multi-task and meta-learning for generalization—continue to expand RL's capabilities. Challenges remain: sample efficiency in real-world domains, safe exploration (ensuring agent doesn't take catastrophic actions during learning), and scaling to long-horizon tasks. As these challenges are addressed, RL promises to unlock autonomous systems that adapt and improve continuously, from personalized healthcare to scientific discovery and beyond.</p>
  `,
  codeExamples: [
    {
      language: 'Python',
      code: `import numpy as np
import gym

# Q-Learning implementation
class QLearning:
  def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99, epsilon=0.1):
      self.Q = np.zeros((n_states, n_actions))
      self.lr = lr
      self.gamma = gamma
      self.epsilon = epsilon
      self.n_actions = n_actions

  def choose_action(self, state):
      # ε-greedy policy
      if np.random.random() < self.epsilon:
          return np.random.randint(self.n_actions)  # Explore
      else:
          return np.argmax(self.Q[state])  # Exploit

  def update(self, state, action, reward, next_state):
      # Q-learning update rule
      best_next_action = np.argmax(self.Q[next_state])
      td_target = reward + self.gamma * self.Q[next_state, best_next_action]
      td_error = td_target - self.Q[state, action]
      self.Q[state, action] += self.lr * td_error

# Train on simple environment
env = gym.make('FrozenLake-v1')
n_states = env.observation_space.n
n_actions = env.action_space.n

agent = QLearning(n_states, n_actions, lr=0.1, gamma=0.99, epsilon=0.1)

# Training loop
n_episodes = 10000
rewards = []

for episode in range(n_episodes):
  state = env.reset()[0]
  total_reward = 0
  done = False

  while not done:
      action = agent.choose_action(state)
      next_state, reward, terminated, truncated, _ = env.step(action)
      done = terminated or truncated

      agent.update(state, action, reward, next_state)

      state = next_state
      total_reward += reward

  rewards.append(total_reward)

  # Decay epsilon
  agent.epsilon = max(0.01, agent.epsilon * 0.995)

  if episode % 1000 == 0:
      avg_reward = np.mean(rewards[-100:])
      print(f"Episode {episode}, Avg Reward: {avg_reward:.3f}, ε: {agent.epsilon:.3f}")

# Test learned policy
state = env.reset()[0]
done = False
total_reward = 0

agent.epsilon = 0  # Greedy policy
while not done:
  action = agent.choose_action(state)
  state, reward, terminated, truncated, _ = env.step(action)
  done = terminated or truncated
  total_reward += reward

print(f"\\nTest episode reward: {total_reward}")`,
      explanation: 'Q-Learning implementation with ε-greedy exploration on a simple grid world environment.'
    },
    {
      language: 'Python',
      code: `import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# Deep Q-Network (DQN)
class DQN(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_dim=128):
      super().__init__()
      self.network = nn.Sequential(
          nn.Linear(state_dim, hidden_dim),
          nn.ReLU(),
          nn.Linear(hidden_dim, hidden_dim),
          nn.ReLU(),
          nn.Linear(hidden_dim, action_dim)
      )

  def forward(self, x):
      return self.network(x)

class ReplayBuffer:
  def __init__(self, capacity=10000):
      self.buffer = deque(maxlen=capacity)

  def push(self, state, action, reward, next_state, done):
      self.buffer.append((state, action, reward, next_state, done))

  def sample(self, batch_size):
      batch = random.sample(self.buffer, batch_size)
      states, actions, rewards, next_states, dones = zip(*batch)
      return (
          torch.FloatTensor(states),
          torch.LongTensor(actions),
          torch.FloatTensor(rewards),
          torch.FloatTensor(next_states),
          torch.FloatTensor(dones)
      )

  def __len__(self):
      return len(self.buffer)

class DQNAgent:
  def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0):
      self.action_dim = action_dim
      self.gamma = gamma
      self.epsilon = epsilon

      # Q-network and target network
      self.q_network = DQN(state_dim, action_dim)
      self.target_network = DQN(state_dim, action_dim)
      self.target_network.load_state_dict(self.q_network.state_dict())

      self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
      self.replay_buffer = ReplayBuffer()

  def choose_action(self, state):
      if np.random.random() < self.epsilon:
          return np.random.randint(self.action_dim)

      with torch.no_grad():
          state = torch.FloatTensor(state).unsqueeze(0)
          q_values = self.q_network(state)
          return q_values.argmax().item()

  def train(self, batch_size=64):
      if len(self.replay_buffer) < batch_size:
          return

      states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

      # Current Q-values
      q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

      # Target Q-values
      with torch.no_grad():
          next_q_values = self.target_network(next_states).max(1)[0]
          target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

      # Loss and optimization
      loss = nn.MSELoss()(q_values.squeeze(), target_q_values)

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      return loss.item()

  def update_target_network(self):
      self.target_network.load_state_dict(self.q_network.state_dict())

# Usage
state_dim = 4
action_dim = 2
agent = DQNAgent(state_dim, action_dim)

# Training loop example
for episode in range(1000):
  state = np.random.randn(state_dim)  # Replace with env.reset()
  done = False

  while not done:
      action = agent.choose_action(state)
      # next_state, reward, done = env.step(action)
      next_state = np.random.randn(state_dim)
      reward = np.random.rand()
      done = np.random.rand() > 0.95

      agent.replay_buffer.push(state, action, reward, next_state, done)
      loss = agent.train()

      state = next_state

  # Update target network periodically
  if episode % 10 == 0:
      agent.update_target_network()

  # Decay epsilon
  agent.epsilon = max(0.01, agent.epsilon * 0.995)

  if episode % 100 == 0:
      print(f"Episode {episode}, ε: {agent.epsilon:.3f}")`,
      explanation: 'Deep Q-Network (DQN) implementation with experience replay and target network for stable training.'
    }
  ],
  interviewQuestions: [
    {
      question: 'Explain the difference between on-policy and off-policy learning.',
      answer: `On-policy methods learn about the policy they follow during exploration (e.g., SARSA learns Q-values for the ε-greedy policy it uses). Off-policy methods can learn about a target policy while following a different behavior policy (e.g., Q-learning learns optimal policy while following ε-greedy). Off-policy enables: (1) Learning from historical data, (2) Sample efficiency through experience replay, (3) Exploratory behavior separate from target policy. On-policy methods are often more stable but less sample efficient.`
    },
    {
      question: 'What is the exploration-exploitation trade-off?',
      answer: `The exploration-exploitation trade-off balances between taking known good actions (exploitation) and trying new actions to discover potentially better ones (exploration). Too much exploitation leads to suboptimal policies; too much exploration wastes time on poor actions. Strategies include: (1) ε-greedy - random exploration with probability ε, (2) UCB - upper confidence bound selection, (3) Thompson sampling - probabilistic exploration, (4) Optimistic initialization, (5) Decay schedules reducing exploration over time.`
    },
    {
      question: 'How does Q-learning differ from SARSA?',
      answer: `Q-learning (off-policy): Updates Q(s,a) using maximum next action: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]. SARSA (on-policy): Updates using actual next action taken: Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]. Q-learning learns optimal policy regardless of exploration strategy; SARSA learns policy for the actual behavior being followed. SARSA is often safer in practice as it accounts for exploration during learning.`
    },
    {
      question: 'What is the role of the discount factor γ?',
      answer: `Discount factor γ ∈ [0,1] determines how much future rewards are valued relative to immediate rewards. γ = 0: only immediate rewards matter (myopic). γ = 1: all future rewards equally weighted (infinite horizon). Practical effects: (1) Controls learning horizon, (2) Ensures convergence in infinite scenarios, (3) Represents uncertainty about future, (4) Balances short vs long-term planning. Typical values: 0.9-0.99. Lower γ for shorter episodes, higher for long-term planning tasks.`
    },
    {
      question: 'Explain experience replay and why it\'s important for DQN.',
      answer: `Experience replay stores transitions (s, a, r, s') in a buffer and randomly samples batches for training. Benefits: (1) Breaks temporal correlations in sequential data, (2) Enables multiple learning updates per environment step, (3) Stabilizes training by reducing variance, (4) Allows off-policy learning from historical experiences. Critical for DQN because neural networks require i.i.d. data for stable learning, but RL naturally produces correlated sequential experiences that can cause catastrophic forgetting.`
    },
    {
      question: 'What is the credit assignment problem in RL?',
      answer: `Credit assignment determines which actions were responsible for observed rewards, especially challenging when rewards are delayed or sparse. Problems: (1) Temporal credit assignment - which past actions led to current reward, (2) Structural credit assignment - which components of complex actions were important. Solutions: (1) Eligibility traces for temporal credit, (2) Advantage estimation, (3) Return decomposition, (4) Hierarchical RL, (5) Attention mechanisms in neural policies, (6) Causal inference methods.`
    }
  ],
  quizQuestions: [
    {
      id: 'rl1',
      question: 'What is the goal of reinforcement learning?',
      options: ['Classify data', 'Maximize cumulative reward', 'Minimize loss', 'Cluster samples'],
      correctAnswer: 1,
      explanation: 'The goal of RL is for an agent to learn a policy that maximizes the expected cumulative reward over time through interaction with an environment.'
    },
    {
      id: 'rl2',
      question: 'What does the discount factor γ control?',
      options: ['Learning rate', 'Importance of future rewards', 'Exploration rate', 'Batch size'],
      correctAnswer: 1,
      explanation: 'The discount factor γ (0 ≤ γ ≤ 1) controls how much the agent values future rewards. γ near 0 makes the agent myopic (only immediate rewards), while γ near 1 makes future rewards important.'
    },
    {
      id: 'rl3',
      question: 'What problem does experience replay solve in DQN?',
      options: ['Slow training', 'Correlated samples', 'Large memory', 'Overfitting'],
      correctAnswer: 1,
      explanation: 'Experience replay stores transitions in a buffer and samples randomly for training, breaking the correlation between consecutive samples and stabilizing learning.'
    }
  ]
};
