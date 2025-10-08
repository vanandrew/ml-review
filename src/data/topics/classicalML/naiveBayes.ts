import { Topic } from '../../../types';

export const naiveBayes: Topic = {
  id: 'naive-bayes',
  title: 'Naive Bayes',
  category: 'classical-ml',
  description: 'Probabilistic classifier based on Bayes theorem with strong independence assumptions',
  content: `
    <h2>Naive Bayes</h2>
    <p>Naive Bayes is a probabilistic classification algorithm based on Bayes' Theorem with the "naive" assumption that features are conditionally independent given the class label. Despite this strong assumption, it performs surprisingly well in many real-world applications.</p>

    <h3>Bayes' Theorem</h3>
    <p><strong>P(C|X) = [P(X|C) × P(C)] / P(X)</strong></p>
    <ul>
      <li><strong>P(C|X):</strong> Posterior probability (probability of class C given features X)</li>
      <li><strong>P(X|C):</strong> Likelihood (probability of features X given class C)</li>
      <li><strong>P(C):</strong> Prior probability (probability of class C)</li>
      <li><strong>P(X):</strong> Evidence (probability of features X, acts as normalizing constant)</li>
    </ul>

    <h3>Naive Independence Assumption</h3>
    <p>Assumes features are conditionally independent given the class:</p>
    <ul>
      <li><strong>P(X|C) = P(x₁|C) × P(x₂|C) × ... × P(xₙ|C)</strong></li>
      <li>Simplifies computation significantly</li>
      <li>"Naive" because features are usually dependent in practice</li>
      <li>Works well despite violated assumption</li>
    </ul>

    <h3>The "Naive" Independence Assumption: Why It's Both Wrong and Useful</h3>
    <p>The "naive" assumption states that all features are <strong>conditionally independent</strong> given the class label. Mathematically: P(x₁, x₂, ..., xₙ | y) = P(x₁|y) × P(x₂|y) × ... × P(xₙ|y). This means once you know the class, knowing one feature's value tells you nothing about another feature's value.</p>
    
    <p><strong>Why It's "Naive" (Usually Wrong):</strong></p>
    <p>In reality, features are often correlated. In spam classification with features ["contains 'free'", "contains 'winner'", "length > 100 words"], the presence of "free" and "winner" together is more common in spam than their individual probabilities would suggest—they're not independent. Spam emails use templates that include both words. Similarly, in medical diagnosis, symptoms often co-occur (fever and cough together in flu), violating independence.</p>
    
    <p><strong>Why It Works Anyway:</strong></p>
    <p>Despite being false, the assumption often doesn't hurt classification accuracy much because:</p>
    <ul>
      <li><strong>Classification uses ranking, not absolute probabilities:</strong> You only need P(spam|email) > P(ham|email), not accurate probability values. Even if Naive Bayes estimates P(spam|email) = 0.9 when true value is 0.7, the classification is correct.</li>
      <li><strong>Redundancy helps:</strong> Correlated features provide overlapping evidence pointing to the correct class. Even if the model double-counts evidence, all classes are affected similarly, preserving relative rankings.</li>
      <li><strong>Simplicity prevents overfitting:</strong> With few parameters (linear in features), Naive Bayes generalizes well despite bias. Complex models that capture correlations might overfit those correlations if they're noisy or training-specific.</li>
      <li><strong>High dimensions dilute correlations:</strong> In text with 10,000+ words, most feature pairs are only weakly correlated, making the assumption less harmful.</li>
    </ul>
    
    <p><strong>When It Fails:</strong> Strongly dependent features where dependency is crucial (e.g., "patient has symptom A" matters only if "patient has symptom B"). Feature interactions (effect of A depends on value of B). In these cases, consider Decision Trees (explicitly model interactions), Logistic Regression (captures some dependencies via coefficients), or Bayesian Networks (relax independence).</p>

    <h3>Classification</h3>
    <p>Predict the class with highest posterior probability:</p>
    <ul>
      <li><strong>ŷ = argmax_c P(C=c|X)</strong></li>
      <li>Since P(X) is constant, we maximize: P(X|C) × P(C)</li>
      <li>Taking log for numerical stability: log P(C) + Σ log P(xᵢ|C)</li>
    </ul>

    <h3>Types of Naive Bayes</h3>

    <h4>Gaussian Naive Bayes</h4>
    <ul>
      <li>For continuous features</li>
      <li>Assumes features follow Gaussian (normal) distribution</li>
      <li>P(xᵢ|C) = (1/√(2πσ²)) × exp(-(xᵢ-μ)²/(2σ²))</li>
      <li>Learn mean μ and variance σ² for each feature per class</li>
      <li>Best for: Continuous numerical features</li>
    </ul>

    <h4>Multinomial Naive Bayes</h4>
    <ul>
      <li>For discrete count features</li>
      <li>Originally for document classification (word counts)</li>
      <li>P(xᵢ|C) = (count of feature i in class C) / (total count in class C)</li>
      <li>Laplace smoothing prevents zero probabilities</li>
      <li>Best for: Text classification, count data</li>
    </ul>

    <h4>Bernoulli Naive Bayes</h4>
    <ul>
      <li>For binary/boolean features</li>
      <li>Models presence/absence of features</li>
      <li>Explicitly models non-occurrence of features</li>
      <li>Best for: Binary document classification (word present/absent)</li>
    </ul>

    <h3>Laplace Smoothing: Solving the Zero-Probability Problem</h3>
    <p>Laplace smoothing (add-one smoothing) prevents catastrophic failure when a feature-class combination never appears in training data.</p>
    
    <p><strong>The Problem:</strong> If word "blockchain" never appeared in training spam emails, the estimated P("blockchain"|spam) = 0/1000 = 0. During classification, P(spam|email) = P(spam) × P(word₁|spam) × ... × P("blockchain"|spam) × ... = P(spam) × ... × 0 × ... = 0, regardless of other evidence. A single zero eliminates the class entirely. This is overly harsh—absence from training data doesn't mean impossibility.</p>
    
    <p><strong>The Solution:</strong> Add pseudo-counts to all feature-class combinations:</p>
    <p><strong>P(xᵢ|C) = (count(xᵢ, C) + α) / (count(C) + α × |vocabulary|)</strong></p>
    <ul>
      <li><strong>α:</strong> Smoothing parameter (typically α=1, hence "add-one")</li>
      <li><strong>Numerator:</strong> Actual count + α gives every combination at least α "virtual" occurrences</li>
      <li><strong>Denominator:</strong> Total count + (α × vocab size) normalizes probabilities to sum to 1</li>
    </ul>
    
    <p><strong>Example (Multinomial NB for text):</strong></p>
    <p>Training data: 1000 spam emails, vocabulary of 10,000 words. Word "blockchain" appears 0 times in spam.</p>
    <ul>
      <li><strong>Without smoothing:</strong> P("blockchain"|spam) = 0/1000 = 0 ← Problem!</li>
      <li><strong>With α=1:</strong> P("blockchain"|spam) = (0+1)/(1000+1×10000) = 1/11000 ≈ 0.00009 ← Small but non-zero</li>
    </ul>
    
    <p>Now a spam email containing "blockchain" won't be automatically classified as ham just because this word is unseen in training spam.</p>
    
    <p><strong>Choosing α:</strong></p>
    <ul>
      <li><strong>α=0:</strong> No smoothing (risky—zero probabilities possible)</li>
      <li><strong>α=1:</strong> Laplace/add-one smoothing (standard, works well in most cases)</li>
      <li><strong>α<1:</strong> Light smoothing (e.g., α=0.1) when you have lots of data</li>
      <li><strong>α>1:</strong> Heavy smoothing (e.g., α=10) for very sparse data or small vocabularies</li>
      <li>Tune α via cross-validation for optimal performance on your specific dataset</li>
    </ul>
    
    <p><strong>Why It Matters More for Text:</strong> Text data is sparse—vocabulary is large (10k-100k words) but documents are short (100-1000 words), so most word-class combinations are unseen. Without smoothing, Naive Bayes fails on any test document containing new words. With smoothing, it gracefully handles novel vocabulary.</p>

    <h3>Advantages</h3>
    <ul>
      <li>Fast training and prediction (O(nd) complexity)</li>
      <li>Works well with small training sets</li>
      <li>Naturally handles multi-class problems</li>
      <li>Provides probability estimates</li>
      <li>Handles high-dimensional data well (curse of dimensionality less severe)</li>
      <li>Simple to implement and interpret</li>
      <li>Requires minimal hyperparameter tuning</li>
      <li>Excellent for text classification</li>
    </ul>

    <h3>Disadvantages</h3>
    <ul>
      <li>Strong independence assumption (rarely true in practice)</li>
      <li>Poor probability estimates (though classifications can still be good)</li>
      <li>Zero-frequency problem (mitigated by smoothing)</li>
      <li>Correlated features reduce performance</li>
      <li>Sensitive to irrelevant features</li>
      <li>Cannot learn feature interactions</li>
    </ul>

    <h3>Step-by-Step Classification Example</h3>
    <p>Let's classify an email as spam/ham using Multinomial Naive Bayes with a tiny vocabulary.</p>
    
    <p><strong>Training Data:</strong></p>
    <ul>
      <li><strong>Spam (3 emails):</strong>
        <ul>
          <li>Email 1: "buy free now" (words: buy×1, free×1, now×1)</li>
          <li>Email 2: "free offer now" (words: free×1, offer×1, now×1)</li>
          <li>Email 3: "buy free offer" (words: buy×1, free×1, offer×1)</li>
        </ul>
      </li>
      <li><strong>Ham (2 emails):</strong>
        <ul>
          <li>Email 4: "meeting tomorrow" (words: meeting×1, tomorrow×1)</li>
          <li>Email 5: "call me tomorrow" (words: call×1, me×1, tomorrow×1)</li>
        </ul>
      </li>
    </ul>
    
    <p><strong>Vocabulary:</strong> {buy, free, now, offer, meeting, tomorrow, call, me} (8 words)</p>
    
    <p><strong>Step 1: Estimate Priors</strong></p>
    <ul>
      <li>P(spam) = 3/5 = 0.6</li>
      <li>P(ham) = 2/5 = 0.4</li>
    </ul>
    
    <p><strong>Step 2: Count Word Occurrences</strong></p>
    <table>
      <thead><tr><th>Word</th><th>Count in Spam</th><th>Count in Ham</th></tr></thead>
      <tbody>
        <tr><td>buy</td><td>2</td><td>0</td></tr>
        <tr><td>free</td><td>3</td><td>0</td></tr>
        <tr><td>now</td><td>2</td><td>0</td></tr>
        <tr><td>offer</td><td>2</td><td>0</td></tr>
        <tr><td>meeting</td><td>0</td><td>1</td></tr>
        <tr><td>tomorrow</td><td>0</td><td>2</td></tr>
        <tr><td>call</td><td>0</td><td>1</td></tr>
        <tr><td>me</td><td>0</td><td>1</td></tr>
        <tr><td><strong>Total</strong></td><td><strong>9</strong></td><td><strong>5</strong></td></tr>
      </tbody>
    </table>
    
    <p><strong>Step 3: Estimate Likelihoods (with α=1 Laplace smoothing)</strong></p>
    <p>P(word|class) = (count + 1) / (total_count + vocabulary_size) = (count + 1) / (total + 8)</p>
    
    <p><strong>Spam:</strong></p>
    <ul>
      <li>P(buy|spam) = (2+1)/(9+8) = 3/17 ≈ 0.176</li>
      <li>P(free|spam) = (3+1)/(9+8) = 4/17 ≈ 0.235</li>
      <li>P(tomorrow|spam) = (0+1)/(9+8) = 1/17 ≈ 0.059</li>
    </ul>
    
    <p><strong>Ham:</strong></p>
    <ul>
      <li>P(buy|ham) = (0+1)/(5+8) = 1/13 ≈ 0.077</li>
      <li>P(free|ham) = (0+1)/(5+8) = 1/13 ≈ 0.077</li>
      <li>P(tomorrow|ham) = (2+1)/(5+8) = 3/13 ≈ 0.231</li>
    </ul>
    
    <p><strong>Step 4: Classify Test Email "buy free tomorrow"</strong></p>
    
    <p><strong>Spam score:</strong></p>
    <p>P(spam) × P(buy|spam) × P(free|spam) × P(tomorrow|spam)</p>
    <p>= 0.6 × 0.176 × 0.235 × 0.059 = 0.00147</p>
    
    <p><strong>Ham score:</strong></p>
    <p>P(ham) × P(buy|ham) × P(free|ham) × P(tomorrow|ham)</p>
    <p>= 0.4 × 0.077 × 0.077 × 0.231 = 0.00055</p>
    
    <p><strong>Prediction:</strong> Spam (0.00147 > 0.00055)</p>
    <p>Despite "tomorrow" being a ham word, "buy" and "free" strongly indicate spam, leading to correct classification.</p>
    
    <p><strong>Note on Log Probabilities:</strong> In practice, we use log probabilities to avoid underflow with many features:
    <ul>
      <li>log P(spam|X) = log P(spam) + Σ log P(wᵢ|spam)</li>
      <li>Predict argmax [log P(spam|X), log P(ham|X)]</li>
    </ul>

    <h3>Common Pitfalls and Solutions</h3>
    <ul>
      <li><strong>Forgetting Laplace smoothing:</strong> Always use α>0 to avoid zero probabilities. Default α=1 works well.</li>
      <li><strong>Not using log probabilities:</strong> With many features, probabilities underflow to 0.0. Always use log-space: log P(y) + Σ log P(xᵢ|y).</li>
      <li><strong>Using wrong variant:</strong> Gaussian for continuous features, Multinomial for counts, Bernoulli for binary. Mismatches hurt performance.</li>
      <li><strong>Keeping highly correlated features:</strong> Naive Bayes double-counts correlated evidence. Remove redundant features for better calibration.</li>
      <li><strong>Treating it as black-box:</strong> Naive Bayes is interpretable! Inspect P(word|spam) to see which words indicate spam. Use this for feature engineering.</li>
      <li><strong>Expecting well-calibrated probabilities:</strong> Predicted probabilities are often over-confident (too close to 0 or 1). Use for ranking/classification, not confidence estimation. Apply calibration (Platt scaling, isotonic regression) if you need accurate probabilities.</li>
      <li><strong>Applying to non-text non-independent data:</strong> Naive Bayes excels on text (high-dimensional, sparse, somewhat independent features). For other domains with strong feature dependencies, consider alternatives.</li>
    </ul>

    <h3>Applications</h3>
    <ul>
      <li><strong>Spam filtering:</strong> Classic use case (spam vs ham). Gmail's early spam filter used Naive Bayes.</li>
      <li><strong>Text classification:</strong> Sentiment analysis, topic categorization, language detection, author identification</li>
      <li><strong>Real-time prediction:</strong> Fast training (O(n)) and prediction (O(d)) enable real-time systems with millions of requests</li>
      <li><strong>Document classification:</strong> News articles into categories, support tickets by topic, medical records by diagnosis</li>
      <li><strong>Recommendation systems:</strong> As baseline or feature ("users who liked X also liked Y")</li>
      <li><strong>Medical diagnosis:</strong> Disease prediction from symptoms (though violated independence is more problematic here)</li>
      <li><strong>Fraud detection:</strong> Flagging suspicious transactions based on features (amount, location, time)</li>
      <li><strong>Online learning:</strong> Easy to update with new data incrementally (just update counts)</li>
    </ul>

    <h3>Visual Understanding</h3>
    <p>Imagine you're trying to identify whether an email is spam based on the words it contains. Naive Bayes asks: "For each word, how much more often does it appear in spam vs ham?" Words like "free," "offer," and "buy" appear frequently in spam, so seeing them increases the spam score. Words like "meeting" or "tomorrow" appear more in ham, decreasing spam score. The algorithm multiplies these individual word "votes" together (in practice, adds their log probabilities) to get a final prediction.</p>
    
    <p><strong>Key visualizations to understand:</strong></p>
    <ul>
      <li><strong>Conditional probability heatmap:</strong> For text classification, show a table where rows are words ("free", "meeting", "offer") and columns are classes (Spam, Ham). Cell values are P(word|class), color-coded (red = high probability). Words like "free" are red under Spam, "meeting" is red under Ham. This shows which words are discriminative.</li>
      <li><strong>Feature contribution bar chart:</strong> For a specific prediction, show bars for each feature with signed contribution: +2.3 (word "free" pushes toward spam), -1.1 (word "tomorrow" pushes toward ham), +1.8 (word "buy" toward spam). Final sum determines class. Illustrates additive log-probability model.</li>
      <li><strong>Class prior and likelihood decomposition:</strong> Pie chart showing prior P(spam)=60%, then multiply by likelihoods from each word. Visual shows how prior belief is updated by evidence from each feature.</li>
      <li><strong>Decision boundary for 2D continuous features:</strong> Scatter plot with Gaussian NB decision boundary. For two features (e.g., height and weight for gender classification), show ellipses representing Gaussian distributions for each class, and the boundary where P(male|x) = P(female|x). Boundary is curved but simple (products of Gaussians).</li>
      <li><strong>Comparison of independence assumption:</strong> Side-by-side: Left shows actual feature correlations (scatter plot with strong correlation between features). Right shows Naive Bayes' assumption (overlaid vertical/horizontal lines, treating features independently). Gap between them explains when NB underperforms.</li>
    </ul>

    <h3>Common Mistakes to Avoid</h3>
    <ul>
      <li><strong>❌ Forgetting Laplace smoothing:</strong> Without it, a single unseen word in test data causes P(word|class)=0, making entire probability 0. Always use α≥1 (default in sklearn). This is critical for text data with large vocabularies.</li>
      <li><strong>❌ Using wrong variant for data type:</strong> Gaussian for continuous, Multinomial for counts (word frequencies), Bernoulli for binary (word presence/absence). Using Gaussian on count data or Multinomial on continuous data severely hurts performance.</li>
      <li><strong>❌ Not using log probabilities:</strong> Multiplying many small probabilities (e.g., 100 features each with P≈0.1) causes underflow to 0.0. Sklearn handles this internally, but if implementing yourself, ALWAYS work in log-space: log P(y) + Σ log P(xᵢ|y).</li>
      <li><strong>❌ Including highly correlated features:</strong> If features X1 and X2 are highly correlated (e.g., "buy" and "purchase"), Naive Bayes counts their evidence twice, over-weighting it. Remove redundant features via correlation analysis or feature selection.</li>
      <li><strong>❌ Expecting calibrated probabilities:</strong> Naive Bayes often outputs extreme probabilities (99.9% or 0.1%) due to violated independence. Use predicted class for classification, but don't trust raw probabilities. Apply Platt scaling or isotonic regression if calibrated probabilities are needed.</li>
      <li><strong>❌ Using on data with strong feature dependencies:</strong> If features are highly dependent (e.g., image pixels, where neighboring pixels are correlated), Naive Bayes underperforms. Use models that capture dependencies: logistic regression with interactions, tree-based methods, neural networks.</li>
      <li><strong>❌ Not handling class imbalance:</strong> If 95% of emails are ham, predicting "ham" for everything gives 95% accuracy but is useless. Use stratified splits, class weights, or evaluate with F1/AUC, not just accuracy.</li>
      <li><strong>❌ Applying Gaussian NB without checking feature distributions:</strong> Gaussian NB assumes features follow normal distributions. If features are heavily skewed or multimodal, transform them (log, Box-Cox) or use a different variant/algorithm.</li>
    </ul>

    <h3>Best Practices</h3>
    <ul>
      <li>Use appropriate variant for your data type</li>
      <li>Apply Laplace smoothing to avoid zero probabilities</li>
      <li>Remove highly correlated features</li>
      <li>Feature selection improves performance</li>
      <li>Consider as baseline before complex models</li>
      <li>Works better for balanced datasets</li>
    </ul>
  `,
  codeExamples: [
    {
      language: 'Python',
      code: `from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np

# Gaussian Naive Bayes for continuous features
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                         n_redundant=5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gaussian NB (for continuous features)
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
y_proba = gnb.predict_proba(X_test)

print("Gaussian Naive Bayes:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Class priors: {gnb.class_prior_}")
print(f"\\nProbability estimates (first 3 samples):")
print(y_proba[:3])

# Cross-validation
cv_scores = cross_val_score(gnb, X, y, cv=5, scoring='accuracy')
print(f"\\nCV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Compare with and without feature scaling
# Note: Gaussian NB doesn't require scaling, but let's see the effect
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

gnb_scaled = GaussianNB()
gnb_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = gnb_scaled.predict(X_test_scaled)

print(f"\\nWith scaling: {accuracy_score(y_test, y_pred_scaled):.4f}")
print(f"Without scaling: {accuracy_score(y_test, y_pred):.4f}")`,
      explanation: 'Demonstrates Gaussian Naive Bayes for continuous features. Shows probability estimates, class priors, and cross-validation. Unlike distance-based methods, NB doesn\'t strictly require feature scaling.'
    },
    {
      language: 'Python',
      code: `from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Text classification example
docs_train = [
  'python is great for data science',
  'machine learning with python',
  'deep learning and neural networks',
  'this movie was terrible',
  'worst film ever made',
  'great acting and cinematography'
]
labels_train = [1, 1, 1, 0, 0, 1]  # 1=positive, 0=negative

docs_test = [
  'python for machine learning',
  'terrible acting in this movie'
]
labels_test = [1, 0]

# Convert text to word count vectors
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(docs_train)
X_test_counts = vectorizer.transform(docs_test)

print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"Feature names: {vectorizer.get_feature_names_out()[:10]}...")

# Multinomial Naive Bayes with Laplace smoothing
mnb = MultinomialNB(alpha=1.0)  # alpha=1 is Laplace smoothing
mnb.fit(X_train_counts, labels_train)

y_pred = mnb.predict(X_test_counts)
y_proba = mnb.predict_proba(X_test_counts)

print(f"\\nPredictions: {y_pred}")
print(f"True labels: {labels_test}")
print(f"\\nProbabilities:")
for i, (doc, proba) in enumerate(zip(docs_test, y_proba)):
  print(f"'{doc}'")
  print(f"  Negative: {proba[0]:.3f}, Positive: {proba[1]:.3f}")

# Feature log probabilities (most important words per class)
feature_names = vectorizer.get_feature_names_out()
log_probs = mnb.feature_log_prob_

print(f"\\nTop 5 words for each class:")
for class_idx in [0, 1]:
  top_features = np.argsort(log_probs[class_idx])[-5:]
  print(f"Class {class_idx}: {[feature_names[i] for i in top_features]}")`,
      explanation: 'Text classification with Multinomial Naive Bayes. Shows how to convert text to count vectors, apply Laplace smoothing, and interpret feature probabilities. Common for spam detection and sentiment analysis.'
    }
  ],
  interviewQuestions: [
    {
      question: 'What is the "naive" assumption in Naive Bayes?',
      answer: 'The "naive" assumption is that all features are **conditionally independent** given the class label. In other words, knowing the value of one feature provides no information about the value of another feature, once you know the class. Mathematically: P(x₁, x₂, ..., xₙ | y) = P(x₁|y) × P(x₂|y) × ... × P(xₙ|y). This simplifies computation dramatically—instead of estimating the joint distribution P(x₁, ..., xₙ | y) which requires exponentially many parameters, you estimate n separate conditional distributions P(xᵢ|y), which is linear in the number of features.\n\nFor example, in spam classification with features [contains "free", contains "winner", length > 100 words], Naive Bayes assumes that whether an email contains "free" is independent of whether it contains "winner," given that we know it\'s spam. In reality, this is often false—spam emails that contain "free" are more likely to also contain "winner" because they come from the same template or scam strategy. The features are correlated. Yet Naive Bayes ignores these correlations and treats each feature independently when computing probabilities.\n\nDespite this strong (and often violated) assumption, Naive Bayes works surprisingly well in practice. The reason is subtle: while the estimated probabilities P(y|x) are often inaccurate (poorly calibrated), the class rankings tend to be correct. For classification, you only need to know which class has the highest probability, not the exact probability values. Even if Naive Bayes incorrectly estimates P(spam|email) = 0.9 when the true value is 0.7, as long as P(spam|email) > P(ham|email), the classification is correct. The independence assumption creates bias in probability estimates but doesn\'t necessarily hurt classification accuracy. This makes Naive Bayes a practical and efficient classifier despite its "naive" simplification.'
    },
    {
      question: 'Explain Bayes\' Theorem and how it\'s used in Naive Bayes classification.',
      answer: '**Bayes\' Theorem** relates conditional probabilities: P(y|x) = [P(x|y) × P(y)] / P(x), where y is the class label and x is the feature vector. In words: the probability of class y given features x (posterior) equals the probability of features x given class y (likelihood) times the prior probability of class y, divided by the probability of features x (evidence). For classification, we want to find the class with maximum posterior probability: argmax_y P(y|x) = argmax_y [P(x|y) × P(y)], dropping P(x) since it\'s constant across classes.\n\nNaive Bayes applies Bayes\' theorem with the independence assumption: P(x|y) = P(x₁, x₂, ..., xₙ|y) = ∏P(xᵢ|y). This transforms the problem into estimating simpler probabilities from training data: **P(y)** (prior) is the fraction of training examples with class y; **P(xᵢ|y)** (likelihood) is estimated differently depending on feature type (Gaussian for continuous, multinomial for counts, Bernoulli for binary). For a test example, compute P(y) × ∏P(xᵢ|y) for each class and predict the class with the highest value.\n\nFor example, classifying an email as spam/ham with features [word counts of "free", "winner", "meeting"]. First, estimate priors: P(spam) = 0.3 (30% of training emails are spam), P(ham) = 0.7. Then estimate likelihoods: P("free"|spam) = 0.8 (word appears in 80% of spam), P("free"|ham) = 0.05. Do this for all words and classes. For a test email with specific word counts, compute: P(spam) × P(words|spam) and P(ham) × P(words|ham). If the first is larger, predict spam.\n\nThe beauty of Bayes\' theorem is it inverts the problem: rather than directly modeling P(y|x) (discriminative), which is hard, we model P(x|y) (generative), which is easier because we can process one feature at a time under the independence assumption. This is why Naive Bayes is a **generative classifier**—it models how data is generated (features given class) and uses Bayes\' theorem to infer classification probabilities. The approach is principled, probabilistically interpretable, and computationally efficient.'
    },
    {
      question: 'What is Laplace smoothing and why is it necessary?',
      answer: 'Laplace smoothing (add-one smoothing) addresses the **zero-probability problem** in Naive Bayes. When estimating P(xᵢ|y) from training data, if a particular feature value never appears with class y in the training set, the estimated probability is 0. This causes problems: when classifying a test example, if any P(xᵢ|y) = 0, the entire product ∏P(xᵢ|y) becomes 0, making P(y|x) = 0, regardless of other features. A single unseen feature-class combination zeroes out the entire prediction, which is overly harsh and leads to poor generalization.\n\nFor example, in spam classification, suppose the word "blockchain" never appeared in any training spam emails. The estimated P("blockchain"|spam) = 0. Now a test spam email about cryptocurrency (containing "blockchain") will be incorrectly classified as ham because P(spam|email) = P(spam) × 0 × ... = 0, even if all other words strongly suggest spam. The model is too confident that spam can\'t contain "blockchain" based on limited training data.\n\n**Laplace smoothing** adds a small count (typically 1) to all feature-class combinations: P(xᵢ|y) = (count(xᵢ, y) + α) / (count(y) + α × k), where α is the smoothing parameter (usually 1), and k is the number of possible values for feature xᵢ. This ensures no probability is exactly 0—even unseen combinations get a small non-zero probability. With α = 1 (add-one smoothing), if "blockchain" never appeared in spam training data (count = 0 out of 1000 spam emails, vocabulary size 10,000), we get: P("blockchain"|spam) = (0 + 1) / (1000 + 1×10000) ≈ 0.0001, a small but non-zero value.\n\nThe amount of smoothing (α) is a hyperparameter: **α = 0** (no smoothing) risks zero probabilities; **α = 1** (Laplace) is standard and works well; **α > 1** (more aggressive smoothing) for very small datasets or high sparsity; **α < 1** (lighter smoothing) when you have ample data. Smoothing is especially critical for text classification where vocabulary is large (10,000+ words) and training data is sparse—many word-class combinations are unseen. Without smoothing, Naive Bayes fails catastrophically on test data containing any new feature values. With smoothing, it gracefully handles unseen data by assigning plausible low probabilities rather than impossible zeros. Other smoothing variants include **Lidstone smoothing** (generalized Laplace with tunable α) and **Good-Turing smoothing** (more sophisticated, adjusts based on frequency-of-frequency statistics), but Laplace is simplest and most commonly used.'
    },
    {
      question: 'When would you use Gaussian vs Multinomial vs Bernoulli Naive Bayes?',
      answer: 'The choice depends on your feature types and data distribution. **Gaussian Naive Bayes** assumes features are continuous and follow a Gaussian (normal) distribution for each class. It estimates P(xᵢ|y) as a Gaussian with class-specific mean μᵢ,y and variance σ²ᵢ,y. Use it for continuous features like height, weight, sensor readings, or measurements. For example, classifying iris flowers based on petal length/width, predicting disease based on lab test values, or anomaly detection with sensor data. It works well when features are roughly normally distributed, but can still perform reasonably even when they\'re not, due to the robustness of Naive Bayes to assumption violations.\n\n**Multinomial Naive Bayes** is designed for discrete count data, typically word counts or term frequencies in text. It models P(xᵢ|y) as a multinomial distribution: features represent counts (how many times each word appears). The model estimates the probability that word i appears in class y. Use it for text classification (spam detection, sentiment analysis, topic classification) with bag-of-words or TF-IDF features, document categorization, or any task with count-based features. For example, an email with word counts [3 "free", 0 "meeting", 1 "winner"] is treated as drawing words from the spam/ham multinomial distributions.\n\n**Bernoulli Naive Bayes** is for binary (presence/absence) features. Each feature xᵢ is 0 or 1, indicating whether a word appears (not how many times). It models P(xᵢ=1|y) and P(xᵢ=0|y), explicitly accounting for absent features. Use it for text classification with binary features (word presence), document filtering where you only care if a term appears, or any binary feature domain (yes/no questions, has_symptom features). Bernoulli is particularly good when documents are short and word frequency is less informative than mere presence.\n\n**Comparison for text**: Multinomial uses counts ("free" appears 3 times matters), Bernoulli uses presence ("free" appears, regardless of count). For long documents, multinomial is typically better (frequency information helps). For short documents (tweets, SMS), Bernoulli may work better since counts are less reliable. In practice, try both on your data via cross-validation. **Gaussian for non-text**: Use Gaussian for numerical features, never for text (word counts aren\'t Gaussian). You can mix: use Gaussian Naive Bayes for numerical features and Multinomial for text features in different classifiers, though you\'d need to combine them carefully (or just apply appropriate preprocessing). Scikit-learn provides GaussianNB, MultinomialNB, and BernoulliNB—experiment to find the best fit for your data.'
    },
    {
      question: 'Why does Naive Bayes work well despite the independence assumption being violated?',
      answer: 'Naive Bayes often performs well in practice even though features are usually correlated, violating the independence assumption. The key insight is that **classification depends on ranking classes, not on accurate probability estimates**. Naive Bayes predicts argmax_y P(y|x), so you only need the relative ordering of P(y|x) across classes to be correct, not the absolute values. The independence assumption creates biased probability estimates (usually over-confident: predicted probabilities too close to 0 or 1), but the ranking of classes often remains correct because the bias affects all classes similarly.\n\nFormally, if features are correlated, the true posterior is P(y|x) ∝ P(x|y) × P(y), while Naive Bayes computes P_NB(y|x) ∝ [∏P(xᵢ|y)] × P(y). These aren\'t equal, but they may be **monotonically related**: if P(y₁|x) > P(y₂|x), then P_NB(y₁|x) > P_NB(y₂|x). When this holds, classifications are identical even though probabilities differ. This is more likely when features are **redundant** (many features provide overlapping information) rather than **complementary**—redundancy makes correlations less impactful because each feature independently points toward the correct class.\n\n**Empirical reasons for success**: (1) **Simplicity helps generalization**: Naive Bayes has few parameters (linear in features, not exponential), reducing overfitting risk. With limited data, a simple biased model often outperforms a complex unbiased model. (2) **Robustness to noise**: Correlations between features might be noisy or inconsistent across train/test, so ignoring them can actually help. (3) **High dimensionality**: In high-dimensional spaces (text with 10,000+ features), the effective amount of correlation is diluted—many features are only weakly correlated with each other. (4) **Class separation**: If classes are well-separated in feature space, even a crude approximation to the decision boundary (via independent features) suffices.\n\n**When Naive Bayes fails**: Strongly dependent features where the dependency is critical for classification (e.g., medical diagnosis where symptom combinations matter more than individual symptoms). If feature A only matters when feature B is present, Naive Bayes misses this interaction. In such cases, use discriminative models (logistic regression captures feature interactions via coefficients; decision trees explicitly model interactions via sequential splits) or relax the independence assumption (Tree-Augmented Naive Bayes, Bayesian Networks). Despite its "naive" assumption, Naive Bayes remains a competitive baseline, especially for high-dimensional sparse data like text, where its simplicity and speed make it highly practical.'
    },
    {
      question: 'What are the advantages of Naive Bayes for text classification?',
      answer: '**Speed and efficiency**: Naive Bayes is one of the fastest machine learning algorithms. Training computes simple frequency counts (O(n×d), linear in samples and features) with no optimization required. Prediction multiplies probabilities, which is also O(d), extremely fast. For large text corpora (millions of documents, 100,000+ vocabulary), Naive Bayes trains and predicts orders of magnitude faster than SVM, neural networks, or ensemble methods. This makes it ideal for real-time systems, prototyping, or resource-constrained environments.\n\n**Handles high dimensionality well**: Text data is inherently high-dimensional (vocabulary size = features), often 10,000-100,000 dimensions. Many algorithms struggle with high dimensions (overfitting, computational cost), but Naive Bayes thrives because: (1) it makes the independence assumption, reducing parameters to O(d) instead of O(d²) or worse; (2) sparsity is natural (most words don\'t appear in most documents), and Naive Bayes handles sparse data efficiently; (3) high dimensions often mean features are less correlated (many weak signals instead of few strong correlated ones), making the naive assumption more reasonable.\n\n**Works with limited training data**: Naive Bayes is a low-variance, high-bias estimator. It makes strong assumptions (independence) and has few parameters, so it doesn\'t require massive training data to generalize well. With just hundreds or thousands of labeled examples, Naive Bayes can achieve decent performance, while deep learning might need millions. This is crucial for domains where labeling is expensive (medical, legal text classification). It also provides a strong baseline: always try Naive Bayes first to establish minimum acceptable performance before trying more complex models.\n\n**Naturally handles multi-class problems**: Extends trivially to many classes (not just binary). Compute P(y|x) for each class y and predict the max, regardless of how many classes exist. Many other algorithms require one-vs-rest or pairwise strategies for multi-class, adding complexity. **Interpretability**: Probabilities have clear meanings; you can inspect P(word|spam) to see which words are indicative of spam. Feature importance is transparent: high P(xᵢ|y) means feature xᵢ strongly indicates class y. This helps debugging and understanding model decisions.\n\n**Robust to irrelevant features**: If many features are noise (common in text with large vocabulary), Naive Bayes is relatively unaffected. Irrelevant words have similar probabilities across classes, contributing little to the classification decision. Other models might overfit to these features. **Online learning**: Easy to update with new data incrementally—just update counts without retraining from scratch. Important for evolving text streams (news, social media). The combination of speed, efficiency with high-dimensional sparse data, and minimal tuning requirements makes Naive Bayes a go-to baseline for text classification tasks like spam filtering, sentiment analysis, and topic categorization.'
    }
  ],
  quizQuestions: [
    {
      id: 'nb-q1',
      question: 'You are building a spam filter and encounter a word in the test email that never appeared in training data. Without Laplace smoothing, what happens?',
      options: [
        'The word is ignored',
        'P(word|spam) = 0, making P(spam|email) = 0, incorrectly ruling out spam',
        'Naive Bayes automatically handles this',
        'The model predicts randomly'
      ],
      correctAnswer: 1,
      explanation: 'Zero probability for any feature makes the entire product P(X|C) = 0, eliminating that class from consideration regardless of other evidence. Laplace smoothing (alpha > 0) prevents this by adding small pseudo-counts.'
    },
    {
      id: 'nb-q2',
      question: 'You have a dataset with highly correlated features. How will this affect Naive Bayes?',
      options: [
        'No effect - Naive Bayes handles correlation well',
        'Performance degrades because independence assumption is violated more severely',
        'Naive Bayes will fail to train',
        'Training time increases significantly'
      ],
      correctAnswer: 1,
      explanation: 'Naive Bayes assumes features are independent. With highly correlated features, the assumption is violated more severely, and the model may over-weight correlated evidence. Consider removing redundant features or using a different algorithm.'
    },
    {
      id: 'nb-q3',
      question: 'Which scenario is BEST suited for Naive Bayes?',
      options: [
        'Small dataset with complex feature interactions',
        'Large text dataset for spam classification with real-time prediction requirements',
        'Image classification with pixel correlations',
        'Time series forecasting'
      ],
      correctAnswer: 1,
      explanation: 'Naive Bayes excels at text classification: (1) handles high-dimensional sparse data well, (2) works with small training sets, (3) very fast prediction, (4) text features are somewhat independent. Poor for images (pixel correlations) or time series (temporal dependencies).'
    }
  ]
};
