import { Topic } from '../../../types';

export const regularization: Topic = {
  id: 'regularization',
  title: 'Regularization (L1, L2, Dropout)',
  category: 'foundations',
  description: 'Techniques to prevent overfitting and improve model generalization',
  content: `
    <h2>Regularization: Controlling Model Complexity</h2>
    <p>Regularization is the practice of adding constraints or penalties to a model to prevent overfitting and improve generalization. The core idea is to discourage overly complex models that fit training data too closely, including its noise and idiosyncrasies. By penalizing complexity, regularization guides the learning algorithm toward simpler models that capture true underlying patterns rather than memorizing training examples.</p>
    
    <p>Without regularization, complex models with many parameters can achieve perfect training accuracy while performing poorly on new data. Regularization provides a principled way to control this by modifying the loss function to include not just prediction error, but also a measure of model complexity. The result is a bias-variance tradeoff: some increase in training error (bias) in exchange for better generalization (reduced variance).</p>

    <div class="info-box info-box-cyan">
      <h4>⚡ Regularization Quick Guide</h4>
      <table>
        <tr>
          <th>Technique</th>
          <th>What It Does</th>
          <th>When to Use</th>
        </tr>
        <tr>
          <td><strong>L2 (Ridge)</strong></td>
          <td>Shrinks all weights, keeps all features</td>
          <td>Default choice, all features relevant</td>
        </tr>
        <tr>
          <td><strong>L1 (Lasso)</strong></td>
          <td>Drives weights to zero, feature selection</td>
          <td>Many irrelevant features, need sparsity</td>
        </tr>
        <tr>
          <td><strong>Elastic Net</strong></td>
          <td>Combines L1 + L2</td>
          <td>Correlated features, unsure L1 vs L2</td>
        </tr>
        <tr>
          <td><strong>Dropout</strong></td>
          <td>Randomly drops neurons during training</td>
          <td>Neural networks (p=0.5 for FC layers)</td>
        </tr>
        <tr>
          <td><strong>Early Stopping</strong></td>
          <td>Stops when validation plateaus</td>
          <td>Any iterative algorithm, always use</td>
        </tr>
      </table>
      <p><strong>💡 Pro Tip:</strong> Start with L2 + Early Stopping. Add Dropout for neural networks. Use L1 only if you need feature selection.</p>
    </div>

    <h3>L2 Regularization (Ridge Regression / Weight Decay)</h3>
    
    <p><strong>Mathematical Formulation:</strong></p>
    <p>$\\text{Loss} = \\text{Original Loss} + \\lambda \\sum w^2$</p>
    <p>where $\\lambda$ (lambda) is the regularization strength parameter and $w$ represents model weights.</p>
    
    <p>L2 regularization adds a penalty term proportional to the sum of squared weights to the loss function. During training, the optimization algorithm must balance minimizing prediction error (original loss) with keeping weights small (regularization term). This creates "weight decay" because the gradient of the squared weights term always pushes weights toward zero.</p>
    
    <p><strong>How It Works:</strong></p>
    <p>Large weights indicate the model is relying heavily on specific features or parameters, which can lead to overfitting—the model becomes too sensitive to particular input values. L2 regularization penalizes large weights quadratically, so doubling a weight quadruples its penalty. This strongly discourages extreme weight values while allowing many small non-zero weights. The effect is that weights shrink toward zero but rarely become exactly zero; instead, you get many small weights distributed across features.</p>
    
    <p><strong>The Role of λ (Lambda):</strong></p>
    <ul>
      <li><strong>λ = 0:</strong> No regularization; model can use full capacity (risk of overfitting)</li>
      <li><strong>Small λ (e.g., 0.001):</strong> Weak penalty; model nearly unconstrained</li>
      <li><strong>Moderate λ (e.g., 0.01-1.0):</strong> Balanced regularization; typical sweet spot</li>
      <li><strong>Large λ (e.g., 10-100):</strong> Strong penalty; weights driven very small (risk of underfitting)</li>
      <li><strong>Very large λ:</strong> Model becomes too simple, potentially just predicting the mean</li>
    </ul>
    
    <p>Finding optimal λ requires hyperparameter tuning via cross-validation. Plot validation performance vs λ: as λ increases from zero, validation performance improves (reducing overfitting), reaches a peak (optimal regularization), then degrades (causing underfitting).</p>
    
    <p><strong>Weight Decay in Neural Networks:</strong></p>
    <p>In the context of neural networks trained with gradient descent, L2 regularization is often called "weight decay." The gradient of the L2 penalty term is $2\\lambda w$, which when subtracted during the weight update acts as exponential decay: weights multiplicatively shrink by a factor of $(1-2\\lambda\\eta)$ each iteration (where $\\eta$ is learning rate). This equivalence between L2 regularization and weight decay holds for standard gradient descent.</p>
    
    <p><strong>When to Use L2:</strong></p>
    <ul>
      <li>When you believe all or most features are relevant and should be kept</li>
      <li>When you want stable, continuous weight adjustments</li>
      <li>As default regularization for neural networks and linear models</li>
      <li>When features are correlated (L2 spreads weight across correlated features)</li>
      <li>When you need computationally efficient optimization (differentiable everywhere)</li>
    </ul>
    
    <p><strong>Advantages:</strong></p>
    <ul>
      <li>Smooth, differentiable penalty enables efficient optimization</li>
      <li>Closed-form solutions exist for some models (Ridge regression)</li>
      <li>Generally provides good generalization improvements</li>
      <li>Handles multicollinearity by distributing weights among correlated features</li>
      <li>Numerically stable</li>
    </ul>

    <h3>L1 Regularization (Lasso Regression)</h3>
    
    <p><strong>Mathematical Formulation:</strong></p>
    <p>$\\text{Loss} = \\text{Original Loss} + \\lambda \\sum |w|$</p>
    
    <p>L1 regularization adds a penalty proportional to the sum of absolute values of weights. Unlike L2's quadratic penalty, L1's linear penalty treats all weight magnitudes equally—doubling a weight doubles its penalty. Critically, the absolute value function creates a non-smooth penalty at zero that encourages exact sparsity.</p>
    
    <p><strong>Sparsity and Feature Selection:</strong></p>
    <p>L1's defining characteristic is that it drives many weights to exactly zero, effectively performing automatic feature selection. As λ increases, more weights become zero until, at very high λ, all weights are zero. The surviving non-zero weights identify the most important features. This makes L1 invaluable for interpretability—a model using 10 out of 1000 features is much easier to understand and deploy than one using all 1000 with tiny weights.</p>
    
    <p><strong>Geometric Intuition:</strong></p>
    <p>Visualize the optimization as finding where the loss contours touch the regularization constraint region. For L2, this region is a circle/sphere (smooth), so the touching point typically has non-zero values in all dimensions. For L1, the region is a diamond/polytope with sharp corners along the axes—solutions often land on these corners where some coordinates are exactly zero. The corners correspond to sparse solutions.</p>
    
    <p><strong>The Role of λ in L1:</strong></p>
    <ul>
      <li><strong>λ = 0:</strong> No regularization; all features retained</li>
      <li><strong>Small λ:</strong> Few weights zeroed out; modest sparsity</li>
      <li><strong>Moderate λ:</strong> Significant sparsity; many features eliminated</li>
      <li><strong>Large λ:</strong> Most weights are zero; very sparse model</li>
      <li><strong>λ → ∞:</strong> All weights become zero; model predicts constant</li>
    </ul>
    
    <p>You can plot the "regularization path"—how weights change as λ varies—to see which features remain non-zero at different regularization strengths. This reveals feature importance ordering.</p>
    
    <p><strong>When to Use L1:</strong></p>
    <ul>
      <li>When you suspect many features are irrelevant and want automatic selection</li>
      <li>When interpretability and model simplicity are priorities</li>
      <li>In high-dimensional settings (p >> n) to identify relevant features</li>
      <li>When you need very sparse models for deployment efficiency</li>
      <li>For feature discovery in exploratory analysis</li>
    </ul>
    
    <p><strong>Challenges:</strong></p>
    <ul>
      <li>Non-differentiable at zero (requires specialized optimization algorithms)</li>
      <li>Can be unstable with highly correlated features (arbitrarily picks one)</li>
      <li>No closed-form solution (unlike Ridge regression)</li>
      <li>More computationally expensive than L2</li>
    </ul>

    <h3>Elastic Net: Combining L1 and L2</h3>
    
    <p><strong>Formula:</strong> $\\text{Loss} = \\text{Original Loss} + \\lambda_1 \\sum |w| + \\lambda_2 \\sum w^2$</p>
    <p>Or equivalently: $\\text{Loss} = \\text{Original Loss} + \\lambda [\\alpha \\sum |w| + (1-\\alpha) \\sum w^2]$</p>
    
    <p>Elastic Net combines L1 and L2 regularization, getting benefits of both: L1's sparsity and feature selection with L2's stability and ability to keep groups of correlated features. The mixing parameter $\\alpha$ controls the balance: $\\alpha=1$ is pure L1, $\\alpha=0$ is pure L2, and intermediate values blend them.</p>
    
    <p><strong>Why Elastic Net?</strong></p>
    <ul>
      <li><strong>Grouped selection:</strong> When features are correlated, L1 picks one arbitrarily; L2 includes all. Elastic Net includes groups of correlated features together.</li>
      <li><strong>Stability:</strong> More stable than pure L1 in presence of highly correlated features</li>
      <li><strong>Sparsity with control:</strong> Get sparse solutions (from L1) without sacrificing too much stability (from L2)</li>
      <li><strong>Flexibility:</strong> Tune $\\alpha$ to adjust sparsity-stability tradeoff for your specific problem</li>
    </ul>
    
    <p><strong>Practical Usage:</strong></p>
    <p>Start with Elastic Net when you're unsure whether L1 or L2 is better. Use grid search or cross-validation to find optimal $\\alpha$ and $\\lambda$. Common $\\alpha$ values to try: [0.1, 0.3, 0.5, 0.7, 0.9]. In practice, Elastic Net often outperforms both pure L1 and pure L2, especially with correlated features.</p>

    <h3>Dropout: Regularization for Neural Networks</h3>
    
    <p>Dropout is a powerful regularization technique specifically designed for neural networks. During training, dropout randomly "drops out" (sets to zero) a fraction of neurons in each layer for each training batch. This prevents neurons from co-adapting and forces the network to learn more robust, distributed representations.</p>
    
    <p><strong>How Dropout Works:</strong></p>
    <p>For each training iteration:</p>
    <ol>
      <li>For each layer with dropout, randomly select p% of neurons (typically 20-50%)</li>
      <li>Set the selected neurons' outputs to zero for this iteration</li>
      <li>Forward propagate using the remaining active neurons</li>
      <li>Backward propagate and update weights only for active neurons</li>
      <li>Next iteration, select a different random set of neurons to drop</li>
    </ol>
    
    <p>Each training batch effectively trains a different sub-network (different neurons dropped). Over many iterations, this is like training an ensemble of 2^n possible networks (where n is the number of neurons) and averaging their predictions.</p>
    
    <p><strong>Why Dropout Prevents Overfitting:</strong></p>
    <ul>
      <li><strong>Breaks co-adaptation:</strong> Neurons can't rely on specific other neurons being present, forcing them to learn more generally useful features</li>
      <li><strong>Ensemble effect:</strong> Training many sub-networks and averaging them reduces variance, like bagging</li>
      <li><strong>Distributes representations:</strong> Information must be spread across many neurons, not concentrated in a few</li>
      <li><strong>Adds noise:</strong> The random dropping acts as noise injection, a known regularizer</li>
    </ul>
    
    <p><strong>Dropout During Inference (Standard Dropout):</strong></p>
    <p>At test time, dropout is turned off—all neurons are active. However, because we trained with only (1-p) fraction of neurons active on average, using all neurons at test would make activations larger than during training. To compensate:</p>
    <ul>
      <li><strong>Standard dropout:</strong> Multiply all neuron outputs by (1-p) at inference time</li>
      <li>If p=0.5 (50% dropout), multiply outputs by 0.5 at test time</li>
      <li>This ensures expected activation magnitudes match training conditions</li>
    </ul>
    
    <p><strong>Inverted Dropout (Modern Standard):</strong></p>
    <p>To avoid extra computation at inference time, modern implementations use "inverted dropout":</p>
    <ul>
      <li><strong>During training:</strong> After dropping neurons, divide remaining neurons' outputs by (1-p)</li>
      <li>This scales up activations to compensate for dropped neurons</li>
      <li><strong>During inference:</strong> Use all neurons with no scaling—simpler and faster</li>
      <li>Mathematically equivalent to standard dropout but more convenient</li>
    </ul>
    
    <p><strong>Choosing Dropout Rate (p):</strong></p>
    <ul>
      <li><strong>p = 0:</strong> No dropout; no regularization</li>
      <li><strong>p = 0.1-0.2:</strong> Light regularization; use for convolutional layers or when overfitting is mild</li>
      <li><strong>p = 0.5:</strong> Standard for fully-connected layers; good default</li>
      <li><strong>p = 0.6-0.8:</strong> Strong regularization; use when overfitting is severe</li>
      <li><strong>p > 0.8:</strong> Usually too much; can cause underfitting</li>
    </ul>
    
    <p><strong>Where to Apply Dropout:</strong></p>
    <ul>
      <li><strong>Fully-connected layers:</strong> Most beneficial here; use p=0.5</li>
      <li><strong>Convolutional layers:</strong> Less prone to overfitting; use lower p=0.1-0.2 or none</li>
      <li><strong>Recurrent connections:</strong> Can use dropout, but requires careful application (don't drop across time steps)</li>
      <li><strong>After activation functions:</strong> Typically applied after ReLU/tanh</li>
      <li><strong>Not on output layer:</strong> Never apply dropout to final predictions</li>
    </ul>
    
    <p><strong>Dropout vs Batch Normalization:</strong></p>
    <p>Batch normalization has some regularization effects (the batch statistics add noise), and in some architectures, adding both dropout and batch normalization can conflict. Modern architectures often use batch normalization for training stability and reduce dropout usage, or skip dropout in layers with batch normalization.</p>

    <h3>Other Important Regularization Techniques</h3>
    
    <p><strong>Early Stopping:</strong></p>
    <p>Monitor validation loss during training and stop when it stops improving (or starts increasing), even if training loss could continue decreasing. This prevents overfitting by halting at the point of best generalization.</p>
    <ul>
      <li>Simple and effective—works with any iterative algorithm</li>
      <li>Use patience parameter (stop after N epochs without improvement)</li>
      <li>Save checkpoints to revert to best validation performance</li>
      <li>Acts as implicit regularization by limiting model capacity to fit noise</li>
    </ul>
    
    <p><strong>Data Augmentation (Implicit Regularization):</strong></p>
    <p>Create synthetic training examples through transformations that preserve the label. For images: rotations, crops, flips, color jittering. For text: synonym replacement, back-translation. For audio: time stretching, pitch shifting. Data augmentation acts as regularization by:</p>
    <ul>
      <li>Increasing effective dataset size, reducing overfitting</li>
      <li>Teaching invariances (rotation-invariant object recognition)</li>
      <li>Adding noise/variation that prevents memorization</li>
      <li>Improving model robustness to real-world variations</li>
    </ul>
    
    <p><strong>Batch Normalization (Side Effect Regularization):</strong></p>
    <p>Batch normalization normalizes layer activations using batch statistics (mean and variance). Its primary purpose is stabilizing and accelerating training, but it has regularization side effects:</p>
    <ul>
      <li>Batch statistics introduce noise (different for each mini-batch), acting like dropout</li>
      <li>Reduces need for other regularization in some architectures</li>
      <li>Can sometimes replace dropout in modern networks</li>
      <li>The regularization effect is weaker than dropout but helps</li>
    </ul>
    
    <p><strong>Label Smoothing:</strong></p>
    <p>Instead of hard targets (0 or 1), use soft targets (0.1 or 0.9). Prevents the model from becoming overconfident and improves generalization, especially in classification.</p>
    
    <p><strong>Mixup and CutMix:</strong></p>
    <p>Create training examples by mixing two samples and their labels. Forces the model to learn smoother decision boundaries and improves robustness.</p>

    <h3>Comparing Regularization Techniques</h3>
    
    <table>
      <tr>
        <th>Technique</th>
        <th>Best For</th>
        <th>Computational Cost</th>
        <th>Sparsity</th>
      </tr>
      <tr>
        <td>L2 (Ridge)</td>
        <td>General use, all features relevant</td>
        <td>Low</td>
        <td>No</td>
      </tr>
      <tr>
        <td>L1 (Lasso)</td>
        <td>Feature selection, high dimensions</td>
        <td>Medium</td>
        <td>Yes</td>
      </tr>
      <tr>
        <td>Elastic Net</td>
        <td>Correlated features, unsure L1 vs L2</td>
        <td>Medium</td>
        <td>Partial</td>
      </tr>
      <tr>
        <td>Dropout</td>
        <td>Neural networks, especially deep</td>
        <td>Low (inverted)</td>
        <td>No</td>
      </tr>
      <tr>
        <td>Early Stopping</td>
        <td>Any iterative algorithm</td>
        <td>None</td>
        <td>No</td>
      </tr>
      <tr>
        <td>Data Aug</td>
        <td>Images, audio, text, limited data</td>
        <td>Medium-High</td>
        <td>No</td>
      </tr>
    </table>

    <h3>Practical Guidelines</h3>
    
    <p><strong>Choosing λ (Regularization Strength):</strong></p>
    <ul>
      <li>Use cross-validation to find optimal λ</li>
      <li>Try logarithmic grid: [0.001, 0.01, 0.1, 1.0, 10, 100]</li>
      <li>Plot validation performance vs λ (regularization path)</li>
      <li>If underfitting: decrease λ</li>
      <li>If overfitting: increase λ</li>
      <li>Can use different λ for different layers in neural networks</li>
    </ul>
    
    <p><strong>Combining Multiple Regularization Techniques:</strong></p>
    <ul>
      <li>L2 + Dropout is standard for neural networks</li>
      <li>L2 + Early Stopping works well for most models</li>
      <li>Data Augmentation + Dropout for computer vision</li>
      <li>Start with one technique, add more if overfitting persists</li>
      <li>Be careful combining dropout + batch normalization (can conflict)</li>
    </ul>
    
    <p><strong>When NOT to Regularize:</strong></p>
    <ul>
      <li>When underfitting (high training and validation error)</li>
      <li>When you have abundant data relative to model complexity</li>
      <li>During initial model development (add regularization after confirming overfitting)</li>
      <li>When interpretability requires using all features (avoid L1)</li>
    </ul>

    <h3>Summary</h3>
    <p>Regularization is essential for building models that generalize well. L2 regularization (weight decay) is the most common baseline, providing stable, continuous shrinkage of weights. L1 performs feature selection through sparsity, ideal when you have many irrelevant features. Elastic Net combines both for flexibility. Dropout is specifically powerful for neural networks, preventing co-adaptation through random neuron dropping. Complement these with early stopping and data augmentation for comprehensive overfitting prevention. The key is matching the regularization technique to your problem: feature selection needs L1, neural networks benefit from dropout, and most problems benefit from L2 as a starting point.</p>
  `,
  codeExamples: [
    {
      language: 'Python',
      code: `from sklearn.linear_model import Ridge, Lasso, ElasticNet
import numpy as np

X = np.random.randn(200, 20)
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(200) * 0.5

# L2 Regularization (Ridge)
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
print(f"Ridge - Non-zero coefficients: {np.sum(np.abs(ridge.coef_) > 0.01)}/20")

# L1 Regularization (Lasso)
lasso = Lasso(alpha=1.0)
lasso.fit(X, y)
print(f"Lasso - Non-zero coefficients: {np.sum(np.abs(lasso.coef_) > 0.01)}/20")
print(f"Lasso - Exactly zero: {np.sum(lasso.coef_ == 0)}")

# Elastic Net (combines L1 and L2)
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic.fit(X, y)`,
      explanation: 'L2 keeps all features but shrinks coefficients. L1 performs feature selection by setting some coefficients to zero. Elastic Net combines both.'
    },
    {
      language: 'Python',
      code: `import tensorflow as tf
from tensorflow.keras import layers, models

# Model with L2 regularization and Dropout
model = models.Sequential([
  layers.Dense(128, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
  layers.Dropout(0.3),  # 30% dropout
  layers.Dense(64, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
  layers.Dropout(0.3),
  layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')`,
      explanation: 'Neural networks typically use both L2 regularization (weight decay) and Dropout for effective regularization.'
    }
  ],
  interviewQuestions: [
    {
      question: 'What is the difference between L1 and L2 regularization?',
      answer: 'L1 (Lasso) and L2 (Ridge) regularization both penalize large weights but in fundamentally different ways with distinct consequences. L2 regularization adds a penalty term proportional to the sum of squared weights ($\\lambda \\sum w^2$) to the loss function. This encourages weights to be small but doesn\'t force them exactly to zero—weights shrink proportionally toward zero but rarely reach it exactly. The penalty is differentiable everywhere, making optimization straightforward with gradient descent. L2 tends to spread weights across all features, giving many features small non-zero weights.\n\nL1 regularization adds a penalty proportional to the sum of absolute values of weights ($\\lambda \\sum |w|$). The key difference is that L1 actively drives some weights exactly to zero, performing automatic feature selection. The absolute value creates a non-differentiable point at zero, which geometrically favors sparse solutions—many weights become exactly zero while others remain relatively large. This makes L1 useful when you suspect many features are irrelevant or want an interpretable model with fewer active features. L1 can be more computationally expensive to optimize due to the non-smooth penalty.\n\nThe geometric intuition helps: visualize the loss surface and the constraint region (where the penalty equals a constant). For L2, this region is a circle/sphere (smooth), so the optimal point tends to have non-zero values in all dimensions. For L1, the region is a diamond/polytope with sharp corners along axes—solutions often land on these corners where some coordinates are exactly zero. Practically, L2 is the default choice for general regularization (stable, easy to optimize, good generalization), while L1 is chosen when you want sparsity/feature selection or suspect the true model uses only a subset of available features. Elastic Net combines both, getting benefits of each: L1\'s sparsity and L2\'s grouping of correlated features.'
    },
    {
      question: 'How does dropout work and why does it prevent overfitting?',
      answer: 'Dropout is a regularization technique for neural networks where, during each training step, we randomly "drop" (set to zero) a fraction of neurons (typically 20-50%) along with their connections. For each training batch, a different random subset of neurons is dropped, meaning each forward/backward pass uses a different sub-network. This randomness prevents neurons from co-adapting—they can\'t rely on the presence of other specific neurons, forcing each to learn more robust features independently useful for making predictions.\n\nDropout prevents overfitting through multiple mechanisms. First, it acts like training an ensemble of exponentially many different sub-networks (2^n possible networks for n neurons), then averaging their predictions. Ensembles reduce variance by averaging out individual model errors, similar to how random forests average many decision trees. Second, it prevents complex co-adaptations where specific combinations of neurons fire together to memorize training data. Without dropout, a neuron might learn to correct another neuron\'s mistakes on training data, creating brittle dependencies that don\'t generalize. Dropout breaks these dependencies, forcing more distributed representations.\n\nDuring training, dropped neurons don\'t participate in forward propagation or backpropagation for that iteration. The remaining neurons must compensate, learning to make good predictions even when their partners are absent. At inference time, dropout is turned off (all neurons active), but their outputs are scaled by the dropout probability to account for more neurons being active than during training. This ensures expected output magnitude matches training conditions. Modern implementations often use "inverted dropout" which scales up during training instead, avoiding extra computation at inference. The dropout rate is a hyperparameter: higher rates provide stronger regularization but can lead to underfitting; typical values are 0.2-0.5 for hidden layers, 0.5 for fully-connected layers, and lower (0.1-0.2) or zero for convolutional layers which are less prone to overfitting.'
    },
    {
      question: 'When would you use L1 over L2 regularization?',
      answer: 'Choose L1 regularization when you want automatic feature selection and suspect many features are irrelevant. L1 drives weights exactly to zero, effectively removing features from the model, producing sparse solutions where only important features have non-zero weights. This is valuable when interpretability matters—a model using 10 out of 1000 features is much easier to understand and deploy than one using all features with small weights. In domains like genomics or text analysis where you have thousands or millions of features but believe only a few drive the outcome, L1\'s sparsity is crucial.\n\nL1 is also preferable when features are highly correlated. L2 tends to give correlated features similar weights (spreading penalty across both), while L1 typically picks one and zeros out the others. This arbitrary selection among correlated features isn\'t ideal for inference but can improve computational efficiency (fewer active features) and prevent multicollinearity issues. For high-dimensional datasets where p > n (more features than samples), L1 can identify a small subset of predictive features, making the problem tractable.\n\nUse L2 in most other scenarios: when you want to use all features but prevent overfitting, when features aren\'t clearly categorizable as relevant/irrelevant, when you need stable gradient-based optimization, or when computationally cheaper solutions matter (L2 has closed-form solutions for some models like linear/ridge regression). L2 tends to give slightly better predictive performance when most features are at least weakly relevant. Elastic Net combines both penalties (αL1 + (1-α)L2), letting you tune between sparsity and stable shrinkage, often outperforming either alone. In neural networks, L2 (weight decay) is more common than L1 because the network\'s architecture already provides feature learning, but L1 can be used for structured sparsity (e.g., pruning entire channels). The choice ultimately depends on your goals: predictive performance only → L2 or Elastic Net; interpretability and feature selection → L1 or Elastic Net with high α.'
    },
    {
      question: 'What happens to dropout during inference?',
      answer: 'During inference (making predictions on new data), dropout is turned off entirely—all neurons are active and contribute to the prediction. However, to maintain consistent output magnitudes, the neuron outputs must be scaled appropriately. During training with dropout probability p, each neuron\'s output is randomly set to zero with probability p, so the expected value of its output is (1-p) times its actual computed value. To match this expected behavior at inference where all neurons are active, we need to scale outputs.\n\nThere are two equivalent approaches. Standard dropout scales neuron outputs at inference by multiplying them by (1-p). If you trained with p=0.5 dropout, at inference you multiply each neuron\'s output by 0.5, ensuring the magnitude matches training expectations. The alternative, inverted dropout (more common in modern implementations), does the scaling during training instead: when a neuron isn\'t dropped during training, its output is divided by (1-p), scaling it up to compensate for other neurons being dropped. At inference with inverted dropout, you simply use all neurons without any scaling—cleaner and computationally cheaper since inference happens more frequently than training.\n\nThe mathematical justification is maintaining E[output] consistent between training and inference. During training, each neuron has probability (1-p) of being active with scaled output, and probability p of being inactive (zero output). The expected output is (1-p) × (scaled value). At inference, all neurons are always active, so without adjustment, the expected output would be higher, creating a train-test mismatch. The scaling correction ensures the network sees similar activation magnitudes whether in training or inference mode, preventing unexpected behavior when deploying the model. Frameworks like TensorFlow and PyTorch handle this automatically—you set model.train() for training mode (dropout active) or model.eval() for evaluation mode (dropout off, appropriate scaling applied), and the framework manages the details.'
    },
    {
      question: 'If your model is underfitting, should you increase or decrease regularization?',
      answer: 'If your model is underfitting (high bias), you should decrease regularization or remove it entirely. Regularization penalizes model complexity, intentionally constraining the model to prevent overfitting. When underfitting, the problem is the opposite—your model is too simple and can\'t capture the underlying patterns in the data. Adding more constraints through regularization makes this worse, further limiting the model\'s capacity to fit the training data. Reducing regularization allows the model more freedom to learn complex patterns and fit the training data better.\n\nConcretely, if using L1 or L2 regularization, reduce the regularization parameter $\\lambda$ (sometimes called alpha). Smaller $\\lambda$ means less penalty on large weights, allowing the model to use its full capacity. If using dropout in neural networks, reduce the dropout rate or remove dropout from some layers. If applying early stopping, train for more epochs to let the model fully learn available patterns. The extreme case is $\\lambda=0$ or dropout rate=0, meaning no regularization at all, which is appropriate when underfitting is severe.\n\nThe diagnostic pattern is: if you see poor performance on both training and validation sets with a small gap between them, you have high bias (underfitting). The solution is to increase model capacity, which includes reducing regularization but also adding features, using more complex model architectures (deeper networks, higher polynomial degrees, more trees in ensemble), or training longer. After reducing regularization and increasing capacity, you might then see overfitting (large train-test gap), at which point you\'d reintroduce regularization at a moderate level. The goal is finding the sweet spot: enough model capacity to capture patterns (low bias) with sufficient regularization to prevent fitting noise (low variance). This is typically found through cross-validation across different regularization strengths, choosing the value that minimizes validation error.'
    }
  ],
  quizQuestions: [
    {
      id: 'reg1',
      question: 'What is the main advantage of L1 over L2 regularization?',
      options: ['Faster computation', 'Automatic feature selection', 'Always better accuracy', 'Works better with neural networks'],
      correctAnswer: 1,
      explanation: 'L1 regularization performs automatic feature selection by driving some coefficients to exactly zero, creating sparse models.'
    },
    {
      id: 'reg2',
      question: 'During inference in a neural network with dropout, what happens?',
      options: ['30% of neurons are dropped', 'All neurons are active', 'Random neurons are dropped', 'Only training neurons are used'],
      correctAnswer: 1,
      explanation: 'During inference, all neurons are active and dropout is turned off. Weights or outputs are scaled to account for this.'
    }
  ]
};
