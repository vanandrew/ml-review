import { Topic } from '../../../types';

export const federatedLearning: Topic = {
  id: 'federated-learning',
  title: 'Federated Learning',
  category: 'advanced',
  description: 'Collaborative training across decentralized devices without sharing data',
  content: `
    <h2>Federated Learning: Privacy-Preserving Collaborative AI</h2>
    <p>Federated Learning (FL), pioneered by Google in 2016, revolutionizes machine learning by enabling model training across millions of decentralized devices without centralizing data. Instead of bringing data to the model (traditional approach), federated learning brings the model to the data. Each participating device—smartphone, hospital server, IoT sensor—trains locally on its private data, sending only model updates (not raw data) to a central server that aggregates improvements. This paradigm addresses growing privacy concerns, regulatory requirements (GDPR, HIPAA), and practical constraints where data cannot be centralized due to size, ownership, or security. Google's Gboard keyboard uses FL to improve next-word predictions from billions of user interactions while keeping typing data on-device. Healthcare institutions collaborate on disease prediction without sharing patient records. Federated learning enables the next generation of AI: privacy-preserving, decentralized, and collaborative.</p>

    <h3>The Federated Learning Paradigm</h3>

    <h4>Traditional vs Federated Learning</h4>

    <h5>Traditional Centralized Learning</h5>
    <ol>
      <li><strong>Collect data:</strong> Aggregate training data from all sources into central server/database</li>
      <li><strong>Train model:</strong> Use centralized data to train ML model</li>
      <li><strong>Deploy:</strong> Distribute trained model to devices</li>
      <li><strong>Challenges:</strong> Privacy concerns, data transfer costs, regulatory barriers, single point of failure</li>
    </ol>

    <h5>Federated Learning Approach</h5>
    <ol>
      <li><strong>Initialize:</strong> Server creates initial global model</li>
      <li><strong>Distribute:</strong> Send model to participating clients (devices)</li>
      <li><strong>Local training:</strong> Each client trains on its local data independently</li>
      <li><strong>Upload updates:</strong> Clients send model updates (weights or gradients) to server</li>
      <li><strong>Aggregate:</strong> Server combines updates into improved global model</li>
      <li><strong>Iterate:</strong> Repeat distribution-training-aggregation cycle</li>
      <li><strong>Benefits:</strong> Data never leaves devices, privacy preserved, reduced bandwidth, regulatory compliance</li>
    </ol>

    <h5>Visual: Federated Learning Communication Round</h5>
    <pre class="code-block">
                  Round t: Global Model w_t
                            │
                            │ 1. Distribute
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
      ┌──────────┐                      ┌──────────┐
      │ Client 1 │  ...                 │ Client K │
      │          │                      │          │
      └──────────┘                      └──────────┘
          │                                   │
          │ 2. Local Training                 │
          │    (On private data)              │
          │    Data never sent!               │
          ▼                                   ▼
      Local Model                        Local Model
        w_t^1                               w_t^k
          │                                   │
          │ 3. Send Updates                   │
          │    (Model weights only)           │
          └───────────────┬───────────────────┘
                          │
                          ▼
                    ┌──────────┐
                    │  SERVER  │
                    │          │
                    └──────────┘
                          │
                          │ 4. Aggregate
                          │    w_{t+1} = Σ (n_k/n) × w_t^k
                          ▼
                  Round t+1: Global Model w_{t+1}
                          │
                          │ 5. Repeat...

Key Benefits:
✓ Privacy: Raw data stays on device
✓ Efficiency: Reduced data transfer
✓ Compliance: GDPR, HIPAA compatible
✓ Scalability: Millions of devices
    </pre>

    <h4>Core Principles</h4>
    <ul>
      <li><strong>Data locality:</strong> Training data remains on client devices, never transmitted</li>
      <li><strong>Decentralization:</strong> No central data repository, distributed computation</li>
      <li><strong>Privacy-by-design:</strong> Built-in privacy preservation through architecture</li>
      <li><strong>Collaborative:</strong> Multiple parties contribute without trust requirements</li>
      <li><strong>Model-centric communication:</strong> Exchange model parameters, not data samples</li>
    </ul>

    <h3>Federated Averaging (FedAvg): The Foundation</h3>

    <h4>Algorithm Overview</h4>
    <p>FedAvg, proposed by McMahan et al. (2017), is the most widely used FL algorithm.</p>

    <h5>Server Algorithm</h5>
    <pre>
<strong>Input:</strong> K clients, learning rate η, communication rounds T
<strong>Initialize:</strong> global model weights w_0

for round t = 1 to T:
  # Sample clients
  S_t ← random subset of K clients (e.g., 10%)
  
  # Distribute model to selected clients
  for each client k in S_t:
      send w_t to client k
  
  # Clients train locally (parallel)
  for each client k in S_t:
      w_t^k ← ClientUpdate(k, w_t)
  
  # Aggregate updates
  w_{t+1} ← Σ (n_k / n) × w_t^k
  
  where n_k = number of data points on client k
        n = Σ n_k (total across selected clients)

<strong>Output:</strong> final global model w_T
    </pre>

    <h5>Client Update Algorithm</h5>
    <pre>
<strong>ClientUpdate(client k, initial weights w):</strong>
  B ← split local data into batches
  
  for epoch e = 1 to E:
      for batch b in B:
          # Standard SGD update
          w ← w - η × ∇L(w; b)
  
  return w
    </pre>

    <h4>Key Components Explained</h4>

    <h5>1. Client Sampling</h5>
    <ul>
      <li><strong>Fraction C:</strong> Proportion of clients per round (e.g., C=0.1 → 10%)</li>
      <li><strong>Minimum clients:</strong> Ensure statistical significance (e.g., at least 100 clients)</li>
      <li><strong>Random selection:</strong> Unbiased sampling</li>
      <li><strong>Rationale:</strong> Millions of clients, can't wait for all; some may be offline</li>
    </ul>

    <h5>2. Local Training</h5>
    <ul>
      <li><strong>Local epochs E:</strong> Number of passes through local data (typically 1-5)</li>
      <li><strong>Local batch size B:</strong> Mini-batch size for SGD</li>
      <li><strong>Trade-off:</strong> More local epochs → fewer communication rounds but risk of overfitting to local data</li>
      <li><strong>Computation vs communication:</strong> Leverage local compute to reduce expensive communication</li>
    </ul>

    <h5>3. Weighted Aggregation</h5>
    <p><strong>Weight by data size:</strong></p>
    <p style="text-align: center; font-size: 1.1em;">
      $w_{\\text{global}} = \\sum_{k\\in S} \\frac{n_k}{n} \\times w_k$
    </p>
    <ul>
      <li><strong>Rationale:</strong> Clients with more data should contribute more</li>
      <li><strong>Alternative:</strong> Uniform weighting (1/K) if data sizes unknown or similar</li>
      <li><strong>Fairness consideration:</strong> May disadvantage minority groups with less data</li>
    </ul>

    <h3>Major Challenges in Federated Learning</h3>

    <h4>1. Non-IID (Non-Independently and Identically Distributed) Data</h4>

    <h5>The Problem</h5>
    <ul>
      <li><strong>Heterogeneous distributions:</strong> Each client's data follows different distribution</li>
      <li><strong>Examples:</strong>
        - Keyboard: User A types in English, User B in Chinese
        - Healthcare: Hospital A serves elderly, Hospital B serves children
        - Images: User prefers cats, another prefers dogs
      </li>
      <li><strong>Impact:</strong> Slow convergence, reduced accuracy, client drift (local models diverge)</li>
    </ul>

    <h5>Types of Non-IID</h5>
    <ul>
      <li><strong>Feature distribution skew:</strong> Different input distributions</li>
      <li><strong>Label distribution skew:</strong> Different class prevalence (e.g., one client mostly class 0)</li>
      <li><strong>Same label, different features:</strong> Digit "7" written differently across cultures</li>
      <li><strong>Temporal shift:</strong> Data distribution changes over time</li>
      <li><strong>Quantity skew:</strong> Vastly different data sizes per client</li>
    </ul>

    <h5>Solutions</h5>
    <ul>
      <li><strong>Data sharing (limited):</strong> Share small public dataset for regularization</li>
      <li><strong>FedProx:</strong> Add proximal term to keep local models close to global: L + (μ/2)||w - w_global||²</li>
      <li><strong>SCAFFOLD:</strong> Use control variates to correct client drift</li>
      <li><strong>Personalization:</strong> Accept heterogeneity, personalize models per client</li>
    </ul>

    <h4>2. Communication Efficiency</h4>

    <h5>The Bottleneck</h5>
    <ul>
      <li><strong>Model size:</strong> Modern models 100MB-1GB (ResNet50: 97MB, BERT-base: 420MB)</li>
      <li><strong>Limited bandwidth:</strong> Mobile networks slow, expensive</li>
      <li><strong>Energy consumption:</strong> Communication drains battery 2-3x faster than computation</li>
      <li><strong>Latency:</strong> Round-trip delays accumulate</li>
    </ul>

    <h5>Communication Costs</h5>
    <ul>
      <li><strong>Per round:</strong> Download model (server→client) + Upload updates (client→server)</li>
      <li><strong>Total rounds:</strong> Often 100s-1000s rounds needed for convergence</li>
      <li><strong>Calculation:</strong> For 1000 rounds, 100MB model: 100MB × 2 × 1000 = 200GB per client!</li>
    </ul>

    <h5>Solutions</h5>

    <h6>Gradient Compression</h6>
    <ul>
      <li><strong>Quantization:</strong> Reduce gradient precision (FP32 → INT8)</li>
      <li><strong>Sparsification:</strong> Send only top-k largest gradients</li>
      <li><strong>Sketching:</strong> Random projections, count sketches</li>
      <li><strong>Typical savings:</strong> 10-100x compression</li>
    </ul>

    <h6>Reduce Communication Rounds</h6>
    <ul>
      <li><strong>More local epochs:</strong> Train longer before communicating</li>
      <li><strong>Better optimizers:</strong> Adaptive methods (Adam) converge faster</li>
      <li><strong>Knowledge distillation:</strong> Distill into smaller model</li>
    </ul>

    <h6>Structured Updates</h6>
    <ul>
      <li><strong>Federated Dropout:</strong> Update only subset of parameters</li>
      <li><strong>Low-rank adaptation:</strong> Send low-rank updates</li>
      <li><strong>Submodel sampling:</strong> Different clients update different parts</li>
    </ul>

    <h4>3. Systems Heterogeneity</h4>

    <h5>Device Variability</h5>
    <ul>
      <li><strong>Compute power:</strong> High-end phones vs low-end IoT (1000x difference)</li>
      <li><strong>Memory:</strong> 128MB to 8GB RAM</li>
      <li><strong>Network:</strong> 5G to 2G, WiFi to cellular</li>
      <li><strong>Battery:</strong> Some plugged in, others battery-constrained</li>
      <li><strong>Availability:</strong> Devices come online/offline unpredictably</li>
    </ul>

    <h5>The Straggler Problem</h5>
    <ul>
      <li><strong>Issue:</strong> Slowest device determines round completion time</li>
      <li><strong>Example:</strong> 99 fast devices finish in 1 min, 1 slow device takes 10 min → round takes 10 min</li>
      <li><strong>Impact:</strong> Drastically slows training</li>
    </ul>

    <h5>Solutions</h5>
    <ul>
      <li><strong>Asynchronous updates:</strong> Don't wait for all clients (FedAsync)</li>
      <li><strong>Deadline-based:</strong> Only aggregate updates received by deadline</li>
      <li><strong>Adaptive aggregation:</strong> Weight by computation time</li>
      <li><strong>Client tiering:</strong> Different requirements for fast/slow devices</li>
    </ul>

    <h4>4. Privacy and Security Risks</h4>

    <h5>Privacy Attacks</h5>

    <h6>Model Inversion</h6>
    <ul>
      <li><strong>Attack:</strong> Reconstruct training data from model updates</li>
      <li><strong>Example:</strong> Given gradient of image classifier, reconstruct training images</li>
      <li><strong>Risk:</strong> Especially severe for small batches or distinct data points</li>
    </ul>

    <h6>Membership Inference</h6>
    <ul>
      <li><strong>Attack:</strong> Determine if specific data point was in training set</li>
      <li><strong>Method:</strong> Query model, observe confidence patterns</li>
      <li><strong>Risk:</strong> Privacy breach (knowing patient was in medical dataset)</li>
    </ul>

    <h5>Security Attacks</h5>

    <h6>Poisoning Attacks</h6>
    <ul>
      <li><strong>Data poisoning:</strong> Malicious client trains on corrupted data</li>
      <li><strong>Model poisoning:</strong> Send crafted malicious updates</li>
      <li><strong>Goal:</strong> Degrade accuracy or insert backdoors</li>
      <li><strong>Example:</strong> Cause model to misclassify specific trigger patterns</li>
    </ul>

    <h6>Sybil Attacks</h6>
    <ul>
      <li><strong>Attack:</strong> Adversary creates multiple fake clients</li>
      <li><strong>Goal:</strong> Gain majority, control aggregation</li>
      <li><strong>Mitigation:</strong> Authentication, reputation systems</li>
    </ul>

    <h3>Privacy-Enhancing Technologies</h3>

    <h4>Differential Privacy (DP)</h4>

    <h5>Definition</h5>
    <p><strong>Informal:</strong> Adding/removing single data point changes output distribution negligibly.</p>
    <p><strong>Formal:</strong> Algorithm A is (ε, δ)-differentially private if for all neighboring datasets D, D' and all outputs S:</p>
    <p style="text-align: center;">
      $P(A(D) \\in S) \\leq e^{\\varepsilon} \\times P(A(D') \\in S) + \\delta$
    </p>

    <h5>Implementation: DP-SGD</h5>
    <ol>
      <li><strong>Compute gradient:</strong> g_i = ∇L(w; x_i) for each sample</li>
      <li><strong>Clip gradients:</strong> ḡ_i = g_i / max(1, ||g_i|| / C) (bound sensitivity)</li>
      <li><strong>Add noise:</strong> g̃ = (1/n) Σ ḡ_i + N(0, σ²C²I) (Gaussian noise)</li>
      <li><strong>Update:</strong> w ← w - η × g̃</li>
    </ol>

    <h5>Parameters</h5>
    <ul>
      <li><strong>ε (epsilon):</strong> Privacy budget (smaller → more private, e.g., ε=1 strong, ε=10 weak)</li>
      <li><strong>δ (delta):</strong> Failure probability (typically $10^{-5}$)</li>
      <li><strong>C (clip norm):</strong> Gradient clipping threshold</li>
      <li><strong>σ (noise scale):</strong> Standard deviation of noise</li>
    </ul>

    <h5>Trade-offs</h5>
    <ul>
      <li><strong>Privacy vs accuracy:</strong> More noise → better privacy but lower accuracy</li>
      <li><strong>Privacy budget composition:</strong> Multiple accesses consume budget</li>
      <li><strong>Practical impact:</strong> 2-5% accuracy drop for strong privacy (ε<1)</li>
    </ul>

    <h4>Secure Aggregation</h4>

    <h5>Goal</h5>
    <p>Server learns only aggregate Σw_k, never individual updates w_k.</p>

    <h5>Protocol (Simplified)</h5>
    <ol>
      <li><strong>Key exchange:</strong> Clients establish pairwise shared secrets</li>
      <li><strong>Masking:</strong> Client k sends w_k + mask_k where mask_k = Σ (shared secret with other clients)</li>
      <li><strong>Aggregation:</strong> Server sums: Σ(w_k + mask_k) = Σw_k + Σmask_k</li>
      <li><strong>Unmask:</strong> Σmask_k = 0 by construction (masks cancel out)</li>
      <li><strong>Result:</strong> Server learns Σw_k without seeing individual w_k</li>
    </ol>

    <h5>Properties</h5>
    <ul>
      <li><strong>Server learns nothing:</strong> About individual clients (unless all but one drop out)</li>
      <li><strong>Cryptographic security:</strong> Provable guarantees</li>
      <li><strong>Overhead:</strong> Additional communication for key exchange</li>
      <li><strong>Dropout tolerance:</strong> Advanced protocols handle client dropouts</li>
    </ul>

    <h4>Homomorphic Encryption</h4>
    <ul>
      <li><strong>Concept:</strong> Compute on encrypted data without decrypting</li>
      <li><strong>FL application:</strong> Aggregate encrypted updates</li>
      <li><strong>Advantage:</strong> Strong cryptographic guarantees</li>
      <li><strong>Disadvantage:</strong> Very high computational overhead (1000-10000x slower)</li>
      <li><strong>Practical:</strong> Limited to specific operations (addition), ongoing research</li>
    </ul>

    <h3>Federated Learning Variants</h3>

    <h4>Cross-Device vs Cross-Silo: Comparison</h4>
    <table >
      <tr>
        <th>Aspect</th>
        <th>Cross-Device FL</th>
        <th>Cross-Silo FL</th>
      </tr>
      <tr>
        <td>Scale</td>
        <td>Millions to billions of devices</td>
        <td>Few to hundreds of organizations</td>
      </tr>
      <tr>
        <td>Participants</td>
        <td>Smartphones, IoT devices, edge devices</td>
        <td>Hospitals, companies, data centers</td>
      </tr>
      <tr>
        <td>Data Size per Participant</td>
        <td>Small (KB to MB)</td>
        <td>Large (GB to TB)</td>
      </tr>
      <tr>
        <td>Availability</td>
        <td>Unpredictable, intermittent</td>
        <td>Reliable, always online</td>
      </tr>
      <tr>
        <td>Communication</td>
        <td>Slow, expensive, limited bandwidth</td>
        <td>Fast, high bandwidth available</td>
      </tr>
      <tr>
        <td>Compute Power</td>
        <td>Limited (mobile processors)</td>
        <td>High (server-grade hardware)</td>
      </tr>
      <tr>
        <td>Client Selection</td>
        <td>Random sampling (e.g., 0.1% per round)</td>
        <td>All or most participants per round</td>
      </tr>
      <tr>
        <td>Privacy Concerns</td>
        <td>Individual user privacy</td>
        <td>Organizational confidentiality</td>
      </tr>
      <tr>
        <td>Examples</td>
        <td>Gboard keyboard, Siri suggestions</td>
        <td>Multi-hospital research, financial consortiums</td>
      </tr>
      <tr>
        <td>Main Challenge</td>
        <td>Scale, stragglers, non-IID data</td>
        <td>Trust, fairness, coordination</td>
      </tr>
    </table>

    <h4>Cross-Device FL</h4>
    <ul>
      <li><strong>Scale:</strong> Millions to billions of devices (smartphones, IoT)</li>
      <li><strong>Characteristics:</strong> Highly unbalanced, unreliable, limited resources</li>
      <li><strong>Data per device:</strong> Small (KB to MB)</li>
      <li><strong>Participation:</strong> Unpredictable, devices often offline</li>
      <li><strong>Communication:</strong> Expensive, limited bandwidth</li>
      <li><strong>Examples:</strong> Google Gboard, Apple Siri, mobile keyboard prediction</li>
    </ul>

    <h4>Cross-Silo FL</h4>
    <ul>
      <li><strong>Scale:</strong> Few to hundreds of organizations (hospitals, companies)</li>
      <li><strong>Characteristics:</strong> Reliable servers, stable connections</li>
      <li><strong>Data per silo:</strong> Large (GB to TB)</li>
      <li><strong>Participation:</strong> Predictable, always online</li>
      <li><strong>Communication:</strong> High bandwidth available</li>
      <li><strong>Examples:</strong> Multi-hospital research, inter-bank fraud detection</li>
    </ul>

    <h4>Personalized Federated Learning</h4>

    <h5>Motivation</h5>
    <p>Single global model may not fit all clients due to heterogeneity.</p>

    <h5>Approaches</h5>
    <ul>
      <li><strong>Meta-learning (MAML):</strong> Learn initialization that quickly adapts to local data</li>
      <li><strong>Multi-task learning:</strong> Shared parameters + personalized parameters</li>
      <li><strong>Fine-tuning:</strong> Train global model, fine-tune last layers locally</li>
      <li><strong>Mixture of experts:</strong> Combine global and local models</li>
      <li><strong>Clustering:</strong> Group similar clients, train per-cluster models</li>
    </ul>

    <h3>Applications and Impact</h3>

    <h4>Mobile Keyboards (Google Gboard)</h4>
    <ul>
      <li><strong>Task:</strong> Next-word prediction, emoji suggestions</li>
      <li><strong>Scale:</strong> 100s of millions of devices</li>
      <li><strong>Privacy:</strong> Typing data never leaves device</li>
      <li><strong>Deployment:</strong> Production since 2017</li>
      <li><strong>Impact:</strong> Personalized predictions without compromising privacy</li>
    </ul>

    <h4>Healthcare Collaboration</h4>
    <ul>
      <li><strong>Disease prediction:</strong> Multi-hospital models without sharing patient data</li>
      <li><strong>Drug discovery:</strong> Pharma companies collaborate on molecular models</li>
      <li><strong>Radiology:</strong> Federated training of imaging models (X-ray, MRI analysis)</li>
      <li><strong>Compliance:</strong> Satisfies HIPAA, GDPR regulations</li>
      <li><strong>Example:</strong> MELLODDY project (10 pharma companies, federated drug discovery)</li>
    </ul>

    <h4>Financial Services</h4>
    <ul>
      <li><strong>Fraud detection:</strong> Banks collaborate without sharing transactions</li>
      <li><strong>Credit scoring:</strong> Improved models from distributed data</li>
      <li><strong>Anti-money laundering:</strong> Cross-institution pattern detection</li>
    </ul>

    <h4>Autonomous Vehicles</h4>
    <ul>
      <li><strong>Collaborative perception:</strong> Vehicles share learned features without raw sensor data</li>
      <li><strong>Rare events:</strong> Learn from collective experiences (accidents, edge cases)</li>
      <li><strong>Privacy:</strong> Keep driving patterns confidential</li>
    </ul>

    <h4>Internet of Things</h4>
    <ul>
      <li><strong>Smart homes:</strong> Personalized automation without cloud dependence</li>
      <li><strong>Industrial IoT:</strong> Predictive maintenance across factories</li>
      <li><strong>Edge intelligence:</strong> On-device learning with global knowledge</li>
    </ul>

    <h3>The Future of Federated Learning</h3>
    <p>Federated learning is transitioning from research to widespread deployment. Future directions include: vertical FL (different features across parties, not samples), federated reinforcement learning for multi-agent systems, FL for foundation models (collaboratively pre-training large models), and integration with blockchain for decentralized aggregation. As privacy regulations tighten globally and data localization laws proliferate, federated learning's importance grows. It enables AI to respect privacy, comply with regulations, and harness decentralized data—unlocking insights while keeping sensitive information secure. Federated learning represents the future of collaborative, privacy-preserving machine learning.</p>
  `,
  codeExamples: [
    {
      language: 'Python',
      code: `import torch
import torch.nn as nn
import torch.optim as optim
import copy
from typing import List

# Simple Federated Learning implementation

class SimpleModel(nn.Module):
  def __init__(self):
      super().__init__()
      self.fc1 = nn.Linear(784, 128)
      self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
      x = torch.relu(self.fc1(x))
      return self.fc2(x)

class FederatedServer:
  def __init__(self, model):
      self.global_model = model

  def aggregate(self, client_models, client_weights):
      """
      Federated Averaging: weighted average of client models

      Args:
          client_models: List of client model state dicts
          client_weights: List of weights (e.g., data size per client)
      """
      # Normalize weights
      total_weight = sum(client_weights)
      weights = [w / total_weight for w in client_weights]

      # Initialize aggregated state dict
      global_dict = self.global_model.state_dict()

      # Average each parameter
      for key in global_dict.keys():
          global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
          for client_dict, weight in zip(client_models, weights):
              global_dict[key] += weight * client_dict[key].float()

      self.global_model.load_state_dict(global_dict)

  def get_global_model(self):
      return copy.deepcopy(self.global_model)

class FederatedClient:
  def __init__(self, client_id, train_data, train_labels):
      self.client_id = client_id
      self.train_data = train_data
      self.train_labels = train_labels
      self.model = None

  def download_model(self, global_model):
      """Download global model from server"""
      self.model = copy.deepcopy(global_model)

  def train(self, epochs=1, lr=0.01):
      """Train on local data"""
      criterion = nn.CrossEntropyLoss()
      optimizer = optim.SGD(self.model.parameters(), lr=lr)

      self.model.train()
      for epoch in range(epochs):
          optimizer.zero_grad()
          outputs = self.model(self.train_data)
          loss = criterion(outputs, self.train_labels)
          loss.backward()
          optimizer.step()

      return loss.item()

  def upload_model(self):
      """Upload model to server"""
      return self.model.state_dict()

  def get_data_size(self):
      return len(self.train_data)

# Federated Learning simulation
def federated_learning(server, clients, num_rounds=10, clients_per_round=5, local_epochs=1):
  """
  Simulate federated learning

  Args:
      server: FederatedServer instance
      clients: List of FederatedClient instances
      num_rounds: Number of communication rounds
      clients_per_round: Number of clients selected per round
      local_epochs: Local training epochs per client
  """
  for round_num in range(num_rounds):
      print(f"\\n=== Round {round_num + 1}/{num_rounds} ===")

      # Select random subset of clients
      import random
      selected_clients = random.sample(clients, min(clients_per_round, len(clients)))

      client_models = []
      client_weights = []

      # Each selected client trains locally
      for client in selected_clients:
          # Download global model
          client.download_model(server.get_global_model())

          # Train locally
          loss = client.train(epochs=local_epochs)
          print(f"Client {client.client_id}: loss = {loss:.4f}")

          # Upload model and data size
          client_models.append(client.upload_model())
          client_weights.append(client.get_data_size())

      # Server aggregates updates
      server.aggregate(client_models, client_weights)

# Example usage
global_model = SimpleModel()
server = FederatedServer(global_model)

# Create clients with different data distributions (simulated)
clients = []
for i in range(10):
  # Each client has different amount of data (non-IID simulation)
  n_samples = torch.randint(50, 200, (1,)).item()
  data = torch.randn(n_samples, 784)
  labels = torch.randint(0, 10, (n_samples,))
  clients.append(FederatedClient(i, data, labels))

# Run federated learning
federated_learning(server, clients, num_rounds=5, clients_per_round=3, local_epochs=2)`,
      explanation: 'Basic federated learning implementation with FedAvg algorithm, showing client-server architecture.'
    },
    {
      language: 'Python',
      code: `import torch
import torch.nn as nn

# Differential Privacy in Federated Learning

class DPFederatedClient:
  def __init__(self, client_id, train_data, train_labels, epsilon=1.0, delta=1e-5):
      self.client_id = client_id
      self.train_data = train_data
      self.train_labels = train_labels
      self.model = None
      self.epsilon = epsilon  # Privacy budget
      self.delta = delta
      self.clip_norm = 1.0  # Gradient clipping threshold

  def download_model(self, global_model):
      self.model = copy.deepcopy(global_model)

  def clip_gradients(self):
      """Clip gradients to bound sensitivity"""
      total_norm = torch.nn.utils.clip_grad_norm_(
          self.model.parameters(),
          self.clip_norm
      )
      return total_norm

  def add_noise_to_gradients(self, noise_scale):
      """Add Gaussian noise to gradients for differential privacy"""
      with torch.no_grad():
          for param in self.model.parameters():
              if param.grad is not None:
                  noise = torch.randn_like(param.grad) * noise_scale
                  param.grad += noise

  def train_with_dp(self, epochs=1, lr=0.01):
      """Train with differential privacy"""
      criterion = nn.CrossEntropyLoss()
      optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

      # Compute noise scale based on privacy budget
      sensitivity = self.clip_norm
      noise_scale = (sensitivity / self.epsilon) * torch.sqrt(
          torch.tensor(2.0 * torch.log(torch.tensor(1.25 / self.delta)))
      )

      self.model.train()
      for epoch in range(epochs):
          optimizer.zero_grad()
          outputs = self.model(self.train_data)
          loss = criterion(outputs, self.train_labels)
          loss.backward()

          # Clip gradients
          self.clip_gradients()

          # Add noise
          self.add_noise_to_gradients(noise_scale)

          optimizer.step()

      return loss.item()

# Secure Aggregation (simplified simulation)
class SecureAggregationServer:
  def __init__(self, model):
      self.global_model = model

  def secure_aggregate(self, encrypted_models, client_weights):
      """
      Simplified secure aggregation
      In practice, uses cryptographic protocols like homomorphic encryption
      """
      # Here we just aggregate, but imagine each model is encrypted
      # and we can only see the aggregate

      total_weight = sum(client_weights)
      weights = [w / total_weight for w in client_weights]

      global_dict = self.global_model.state_dict()

      for key in global_dict.keys():
          global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
          for encrypted_dict, weight in zip(encrypted_models, weights):
              # In real secure aggregation, this operation happens
              # on encrypted data using homomorphic encryption
              global_dict[key] += weight * encrypted_dict[key].float()

      self.global_model.load_state_dict(global_dict)
      return self.global_model

# Personalized Federated Learning
class PersonalizedFLClient:
  def __init__(self, client_id, train_data, train_labels):
      self.client_id = client_id
      self.train_data = train_data
      self.train_labels = train_labels
      self.global_model = None
      self.personal_model = None

  def download_and_personalize(self, global_model, personal_epochs=5):
      """
      Download global model and personalize on local data
      """
      # Start with global model
      self.personal_model = copy.deepcopy(global_model)

      # Fine-tune on local data (only last layer)
      criterion = nn.CrossEntropyLoss()

      # Freeze all layers except last
      for param in self.personal_model.parameters():
          param.requires_grad = False
      for param in self.personal_model.fc2.parameters():
          param.requires_grad = True

      optimizer = torch.optim.SGD(
          filter(lambda p: p.requires_grad, self.personal_model.parameters()),
          lr=0.01
      )

      self.personal_model.train()
      for epoch in range(personal_epochs):
          optimizer.zero_grad()
          outputs = self.personal_model(self.train_data)
          loss = criterion(outputs, self.train_labels)
          loss.backward()
          optimizer.step()

      print(f"Client {self.client_id} personalized model, loss: {loss.item():.4f}")

# Example: Differential Privacy FL
global_model = SimpleModel()
dp_clients = [
  DPFederatedClient(i, torch.randn(100, 784), torch.randint(0, 10, (100,)), epsilon=1.0)
  for i in range(5)
]

for client in dp_clients:
  client.download_model(global_model)
  loss = client.train_with_dp(epochs=1)
  print(f"DP Client {client.client_id} loss: {loss:.4f}")`,
      explanation: 'Differential privacy in federated learning and personalized FL, showing privacy-preserving techniques.'
    }
  ],
  interviewQuestions: [
    {
      question: 'What problem does federated learning solve?',
      answer: `Federated learning enables training ML models across decentralized data without centralizing raw data. Solves: (1) Privacy concerns - data remains on client devices, (2) Regulatory compliance - GDPR, healthcare regulations, (3) Communication costs - sending models vs. data, (4) Data sovereignty - organizations keep control over their data. Applications include mobile keyboard prediction, healthcare analytics, financial fraud detection where data sharing is prohibited or impractical.`
    },
    {
      question: 'Explain the FedAvg algorithm.',
      answer: `FedAvg (Federated Averaging) is the foundational federated learning algorithm: (1) Server broadcasts global model to clients, (2) Each client trains locally for E epochs on local data, (3) Clients send model updates (parameters) to server, (4) Server aggregates updates using weighted averaging based on data size, (5) Process repeats. Benefits: communication efficient, simple implementation. Challenges: non-IID data across clients, varying computational capabilities, client dropouts during training.`
    },
    {
      question: 'What challenges arise from non-IID data in federated learning?',
      answer: `Non-IID (independent and identically distributed) data occurs when clients have different data distributions. Challenges: (1) Model divergence - local updates may conflict, (2) Slower convergence, (3) Reduced final accuracy, (4) Client drift - models become specialized to local data. Solutions: (1) Data sharing (privacy-preserving), (2) Personalization techniques, (3) FedProx algorithm with proximal term, (4) Scaffold algorithm correcting for client drift, (5) Clustered federated learning grouping similar clients.`
    },
    {
      question: 'How does differential privacy work in federated learning?',
      answer: `Differential privacy adds calibrated noise to protect individual privacy while preserving aggregate patterns. In federated learning: (1) Local differential privacy - clients add noise before sending updates, (2) Central differential privacy - server adds noise during aggregation. Privacy budget (ε, δ) parameters control privacy-utility trade-off. Techniques include Gaussian noise addition, gradient clipping, and privacy accounting across multiple rounds. Stronger privacy (lower ε) reduces model accuracy but provides formal privacy guarantees.`
    },
    {
      question: 'What is secure aggregation and why is it important?',
      answer: `Secure aggregation allows the server to compute aggregate statistics (sum/average) of client updates without seeing individual contributions. Uses cryptographic techniques like homomorphic encryption or secure multi-party computation. Benefits: (1) Protects against honest-but-curious servers, (2) Prevents inference attacks on individual updates, (3) Enables stronger privacy guarantees. Trade-offs: increased computational and communication overhead. Essential for sensitive applications where even encrypted model updates could leak information.`
    },
    {
      question: 'Compare cross-device vs cross-silo federated learning.',
      answer: `Cross-device: Millions of clients (phones, IoT devices) with limited data each. Characteristics: high client churn, limited communication windows, heterogeneous hardware. Use cases: mobile keyboards, recommendation systems. Cross-silo: Fewer participants (organizations, hospitals) with substantial datasets. Characteristics: stable participation, better connectivity, similar hardware. Use cases: healthcare collaborations, financial consortiums. Different algorithms and communication strategies needed for each setting.`
    }
  ],
  quizQuestions: [
    {
      id: 'fl1',
      question: 'What is the main advantage of federated learning?',
      options: ['Faster training', 'Data privacy - data stays local', 'Better accuracy', 'Smaller models'],
      correctAnswer: 1,
      explanation: 'Federated learning enables collaborative training without sharing raw data. Data remains on local devices, preserving privacy while still benefiting from collective learning.'
    },
    {
      id: 'fl2',
      question: 'In FedAvg, how are client models combined?',
      options: ['Take best model', 'Weighted average', 'Concatenate', 'Use last client'],
      correctAnswer: 1,
      explanation: 'FedAvg (Federated Averaging) combines client models using a weighted average, where weights are typically proportional to the amount of data each client has.'
    },
    {
      id: 'fl3',
      question: 'What is a challenge unique to federated learning?',
      options: ['Overfitting', 'Non-IID data distribution', 'Vanishing gradients', 'Mode collapse'],
      correctAnswer: 1,
      explanation: 'Non-IID (non-independent and identically distributed) data is a key challenge in FL. Different clients have different data distributions, making convergence slower and more difficult.'
    }
  ]
};
