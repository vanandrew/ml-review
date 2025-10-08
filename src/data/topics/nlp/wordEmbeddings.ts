import { Topic } from '../../../types';

export const wordEmbeddings: Topic = {
  id: 'word-embeddings',
  title: 'Word Embeddings',
  category: 'nlp',
  description: 'Dense vector representations of words that capture semantic meaning',
  content: `
    <h2>Word Embeddings: Representing Language in Continuous Space</h2>
    <p>Word embeddings represent one of the most influential innovations in natural language processing, transforming the way computers understand and process language. By mapping discrete word symbols to continuous vector representations, embeddings enable neural networks to capture semantic and syntactic relationships in ways that traditional symbolic approaches could not achieve. This foundational technique underlies virtually all modern NLP systems, from search engines and recommendation systems to machine translation and conversational AI.</p>

    <h3>The Representation Problem in NLP</h3>
    <p>Natural language processing faces a fundamental challenge: computers process numbers, but language consists of discrete symbols (words, characters, phrases). The question of how to represent linguistic units numerically has profound implications for what relationships models can learn and how effectively they generalize.</p>

    <h4>Traditional Approaches and Their Limitations</h4>
    <p><strong>One-hot encoding</strong> represents each word as a binary vector with a single 1 and all other elements 0. For a vocabulary of size V, each word becomes a V-dimensional vector. Critical limitations include extreme sparsity (99.999% zeros), high dimensionality (vocabulary size = dimensions), no semantic relationships (all words orthogonal), and no generalization across related words.</p>

    <h3>The Word Embedding Revolution</h3>
    <p>Word embeddings solve these limitations by mapping words to <strong>dense, low-dimensional, continuous vectors</strong> where semantic and syntactic relationships are captured through geometric relationships in the embedding space.</p>

    <p><strong>Key properties:</strong> Dense representations (50-300 dimensions vs 50,000+), semantic similarity through proximity, compositional semantics (vec(king) - vec(man) + vec(woman) ≈ vec(queen)), learned from data automatically, and transfer learning across tasks.</p>

    <p><strong>The distributional hypothesis:</strong> Words occurring in similar contexts tend to have similar meanings (Firth, 1957). Embedding methods implicitly capture semantic similarity through contextual similarity.</p>

    <h3>Word2Vec: Neural Word Embeddings at Scale</h3>
    <p>Word2Vec, introduced by Mikolov et al. (2013) at Google, democratized word embeddings by providing efficient algorithms that could train high-quality embeddings on billion-word corpora in hours.</p>

    <h4>CBOW (Continuous Bag of Words)</h4>
    <ul>
      <li><strong>Objective:</strong> Predict center word given context words</li>
      <li><strong>Example:</strong> Context ["the", "cat", "on", "the"] → Target "sat"</li>
      <li><strong>Characteristics:</strong> Faster training, works well for frequent words, better for syntactic tasks</li>
      <li><strong>Mathematical:</strong> Maximize $P(w_t | w_{t-c}, ..., w_{t+c})$</li>
    </ul>

    <h4>Skip-gram</h4>
    <ul>
      <li><strong>Objective:</strong> Predict context words given center word</li>
      <li><strong>Example:</strong> Target "sat" → Context ["the", "cat", "on", "the"]</li>
      <li><strong>Characteristics:</strong> Slower but better for rare words, better for semantic tasks</li>
      <li><strong>Mathematical:</strong> Maximize $P(w_{t-c}, ..., w_{t+c} | w_t)$</li>
    </ul>

    <h4>Negative Sampling: Training Efficiency</h4>
    <p>Standard softmax over entire vocabulary is computationally intractable (requires V dot products per example where V=50,000+). Negative sampling reformulates as binary classification: is this word-context pair "correct" or "noise"?</p>
    
    <p><strong>Method:</strong> For each positive (word, context) pair, sample k=5-20 negative words from noise distribution $P_n(w) \\propto \\text{count}(w)^{3/4}$. This reduces computation from V to k+1 dot products (1000× speedup).</p>

    <h4>Additional Techniques</h4>
    <ul>
      <li><strong>Subsampling:</strong> Randomly discard frequent words ("the", "a") with probability $P(w) = 1 - \\sqrt{\\frac{t}{f(w)}}$ to balance dataset</li>
      <li><strong>Window size:</strong> Small (2-5) for syntactic, large (5-10+) for semantic relationships</li>
      <li><strong>Learning rate:</strong> Start 0.025, decay to 0.0001</li>
      <li><strong>Epochs:</strong> 5-15 iterations over corpus</li>
    </ul>

    <h3>GloVe: Global Vectors for Word Representation</h3>
    <p>GloVe (Pennington et al., 2014) takes a different approach: explicitly model global word-word co-occurrence statistics from the entire corpus rather than predicting local context.</p>

    <p><strong>Core insight:</strong> Ratios of co-occurrence probabilities encode meaning. For "ice" vs "steam": $\\frac{P(\\text{"solid"}|\\text{"ice"})}{P(\\text{"solid"}|\\text{"steam"})}$ is large, $\\frac{P(\\text{"gas"}|\\text{"ice"})}{P(\\text{"gas"}|\\text{"steam"})}$ is small.</p>

    <p><strong>Objective:</strong> Learn word vectors such that $w_i^T w_j + b_i + b_j = \\log(X_{ij})$, where $X_{ij}$ is co-occurrence count.</p>

    <p><strong>Loss function:</strong> $J = \\sum f(X_{ij})(w_i^T w_j + b_i + b_j - \\log X_{ij})^2$, with weighting $f(x) = (x/x_{\\text{max}})^\\alpha$ preventing very frequent co-occurrences from dominating.</p>

    <p><strong>GloVe vs Word2Vec:</strong> Global vs local context, requires pre-computing co-occurrence matrix vs sequential processing, comparable performance but slight differences by task.</p>

    <h3>FastText: Subword Embeddings</h3>
    <p>FastText (Facebook AI Research, 2016) extends Word2Vec by representing words as bags of character n-grams, addressing several critical limitations.</p>

    <p><strong>Problems solved:</strong> Out-of-vocabulary words have no representation, morphological relationships ignored, rare words have poor embeddings, compound words cannot be handled.</p>

    <p><strong>Approach:</strong> Represent each word as sum of its character n-gram embeddings (typically n=3-6). Example: "where" with n=3 yields n-grams <wh, whe, her, ere, re>, plus <where>.</p>

    <p><strong>Key advantages:</strong></p>
    <ul>
      <li><strong>OOV handling:</strong> Generate embeddings for unseen words by composing their n-grams</li>
      <li><strong>Morphology:</strong> "play", "playing", "played" share many n-grams, producing similar embeddings</li>
      <li><strong>Rare words:</strong> Share n-grams with common words for better generalization</li>
      <li><strong>Rich morphology:</strong> Valuable for Turkish, Finnish, Arabic where words have complex structure</li>
      <li><strong>Robustness:</strong> Handles typos and informal text</li>
    </ul>

    <p><strong>Trade-offs:</strong> Higher memory (millions of n-grams vs thousands of words), slightly slower training, may conflate words with similar form but different meanings.</p>

    <h3>Beyond Static Embeddings: Contextual Representations</h3>
    <p>Static embeddings assign single fixed vector per word, ignoring context, creating fundamental limitations.</p>

    <h4>The Polysemy Problem</h4>
    <p>"Bank" has multiple meanings: riverside, financial institution, airplane tilt. Static embeddings produce single vector averaging across all meanings, capturing none precisely.</p>

    <h4>Contextual Embeddings</h4>
    <ul>
      <li><strong>ELMo (2018):</strong> Deep bidirectional LSTMs trained on language modeling generate context-dependent embeddings. "Bank" in "river bank" gets different embedding than in "bank account".</li>
      <li><strong>BERT, GPT (2018+):</strong> Transformer-based contextual embeddings achieve even better representations, forming foundation of modern NLP.</li>
    </ul>

    <p><strong>Trade-offs:</strong> Static embeddings are fast, simple, interpretable. Contextual embeddings handle polysemy, achieve state-of-the-art results, but require full forward pass through deep network per sentence.</p>

    <h3>Evaluation: Measuring Embedding Quality</h3>

    <h4>Intrinsic Evaluation</h4>
    <p><strong>1. Word Similarity:</strong> Correlate embedding similarities with human judgments using datasets like WordSim-353, SimLex-999. Compute cosine similarity between embedding pairs, correlate with human ratings using Spearman's $\\rho$.</p>

    <p><strong>2. Word Analogies:</strong> Test compositional semantics through "a:b :: c:?" format. Compute vec(b) - vec(a) + vec(c), find nearest word. Google analogy dataset has 19,544 questions covering semantic ("Athens:Greece :: Baghdad:Iraq") and syntactic ("apparent:apparently :: rapid:rapidly") categories.</p>

    <p><strong>3. Visualization:</strong> t-SNE or UMAP projection to 2D, verify semantic clusters, examine nearest neighbors.</p>

    <h4>Extrinsic Evaluation</h4>
    <p>Ultimate test: do embeddings improve performance on real NLP tasks?</p>
    <ul>
      <li><strong>Tasks:</strong> Text classification, named entity recognition, question answering, machine translation, information retrieval</li>
      <li><strong>Protocol:</strong> Fix embeddings or allow fine-tuning, train downstream model, measure task-specific metrics (accuracy, F1, BLEU)</li>
    </ul>

    <h3>Practical Considerations and Best Practices</h3>

    <h4>Dimensionality Selection</h4>
    <ul>
      <li><strong>50-100:</strong> Fast, efficient, sufficient for simple tasks or small datasets</li>
      <li><strong>200-300:</strong> Sweet spot for most applications, good performance/efficiency balance</li>
      <li><strong>300+:</strong> Diminishing returns, may overfit, slower computation</li>
    </ul>

    <h4>Pre-training vs Training from Scratch</h4>
    <ul>
      <li><strong>Use pre-trained when:</strong> Limited data (< 100K sentences), general domain, want faster development</li>
      <li><strong>Train from scratch when:</strong> Highly specialized domain (medical, legal), very large dataset, specific vocabulary</li>
      <li><strong>Fine-tuning:</strong> Start with pre-trained, continue training on domain-specific data—often best approach</li>
    </ul>

    <h4>Handling OOV Words</h4>
    <ul>
      <li><strong>FastText:</strong> Generate from subword units (best)</li>
      <li><strong>Random initialization:</strong> Assign random vector (poor but simple)</li>
      <li><strong>UNK token:</strong> Map all OOV to single <UNK> embedding (loses information)</li>
      <li><strong>Character-level models:</strong> Represent words as character sequences</li>
    </ul>

    <h4>Implementation Recommendations</h4>
    <ul>
      <li><strong>Libraries:</strong> Gensim (Python, easy), fastText (C++, fast), TensorFlow/PyTorch (custom)</li>
      <li><strong>Pre-trained:</strong> GloVe (840B tokens, 2.2M vocab), fastText (Common Crawl, 600B tokens, 2M vocab), Word2Vec (Google News, 100B tokens, 3M vocab)</li>
      <li><strong>Normalization:</strong> Often beneficial to L2-normalize embeddings</li>
      <li><strong>Freezing vs fine-tuning:</strong> Small datasets freeze, large datasets fine-tune</li>
    </ul>

    <h3>Applications and Impact</h3>
    <ul>
      <li><strong>Search engines:</strong> Semantic search, query understanding, document relevance</li>
      <li><strong>Recommendation systems:</strong> Content similarity, user-item matching</li>
      <li><strong>Chatbots:</strong> Intent classification, entity extraction, response generation</li>
      <li><strong>Machine translation:</strong> Input representations, attention mechanisms</li>
      <li><strong>Sentiment analysis:</strong> Feature extraction for classification</li>
      <li><strong>Named entity recognition:</strong> Character and word-level features</li>
      <li><strong>Document clustering:</strong> Represent documents as embedding averages</li>
    </ul>

    <h3>Limitations and Future Directions</h3>
    <ul>
      <li><strong>Static representation:</strong> Single vector per word ignores polysemy → Contextual embeddings (ELMo, BERT)</li>
      <li><strong>Lack of compositionality:</strong> Simple averaging doesn't capture phrasal meanings → Tree-based or attention-based composition</li>
      <li><strong>Social biases:</strong> Embeddings learn stereotypes from data (gender bias: "doctor" closer to "man" than "woman") → Debiasing techniques</li>
      <li><strong>Cross-lingual:</strong> Separate embeddings per language → Multilingual embeddings (mBERT, XLM-R)</li>
      <li><strong>Domain adaptation:</strong> May not transfer across domains → Domain-specific training</li>
    </ul>
  `,
  codeExamples: [
    {
      language: 'Python',
      code: `from gensim.models import Word2Vec
import numpy as np

# Sample corpus (list of tokenized sentences)
corpus = [
  ['machine', 'learning', 'is', 'awesome'],
  ['deep', 'learning', 'is', 'a', 'subset', 'of', 'machine', 'learning'],
  ['neural', 'networks', 'are', 'used', 'in', 'deep', 'learning'],
  ['word', 'embeddings', 'capture', 'semantic', 'meaning'],
  ['word2vec', 'and', 'glove', 'are', 'popular', 'embeddings']
]

# Train Word2Vec model (Skip-gram)
model = Word2Vec(
  sentences=corpus,
  vector_size=100,      # Embedding dimension
  window=5,             # Context window size
  min_count=1,          # Minimum word frequency
  sg=1,                 # 1 = Skip-gram, 0 = CBOW
  negative=5,           # Negative sampling
  epochs=100
)

# Get embedding for a word
embedding = model.wv['learning']
print(f"Embedding shape: {embedding.shape}")
print(f"Learning vector: {embedding[:10]}...")  # First 10 dimensions

# Find similar words
similar_words = model.wv.most_similar('learning', topn=5)
print(f"\\nWords similar to 'learning':")
for word, similarity in similar_words:
  print(f"  {word}: {similarity:.3f}")

# Word analogy: king - man + woman = ?
# result = model.wv.most_similar(positive=['king', 'woman'], negative=['man'])

# Compute similarity between two words
similarity = model.wv.similarity('machine', 'deep')
print(f"\\nSimilarity between 'machine' and 'deep': {similarity:.3f}")

# Visualize embeddings (dimensionality reduction with t-SNE)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

words = list(model.wv.index_to_key)[:20]  # Top 20 words
vectors = np.array([model.wv[word] for word in words])

# Reduce to 2D
tsne = TSNE(n_components=2, random_state=42)
vectors_2d = tsne.fit_transform(vectors)

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])
for i, word in enumerate(words):
  plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]))
plt.title('Word Embeddings Visualization (t-SNE)')
plt.show()`,
      explanation: 'This example demonstrates training a Word2Vec Skip-gram model, retrieving word embeddings, finding similar words, computing similarity, and visualizing embeddings in 2D using t-SNE.'
    },
    {
      language: 'Python',
      code: `import torch
import torch.nn as nn

# Using pre-trained GloVe embeddings
# First download: wget http://nlp.stanford.edu/data/glove.6B.zip

def load_glove_embeddings(glove_file, vocab):
  """Load GloVe embeddings for given vocabulary"""
  embeddings_index = {}
  with open(glove_file, 'r', encoding='utf-8') as f:
      for line in f:
          values = line.split()
          word = values[0]
          vector = np.array(values[1:], dtype='float32')
          embeddings_index[word] = vector

  # Create embedding matrix
  embedding_dim = len(next(iter(embeddings_index.values())))
  embedding_matrix = np.zeros((len(vocab), embedding_dim))

  for i, word in enumerate(vocab):
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
          embedding_matrix[i] = embedding_vector
      else:
          # Word not found, use random vector
          embedding_matrix[i] = np.random.randn(embedding_dim) * 0.01

  return embedding_matrix

# Using embeddings in PyTorch
class TextClassifier(nn.Module):
  def __init__(self, vocab_size, embedding_dim, num_classes, pretrained_embeddings=None):
      super().__init__()

      # Embedding layer
      self.embedding = nn.Embedding(vocab_size, embedding_dim)

      # Initialize with pre-trained embeddings
      if pretrained_embeddings is not None:
          self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
          # Option 1: Freeze embeddings (don't update during training)
          # self.embedding.weight.requires_grad = False
          # Option 2: Fine-tune embeddings (default behavior)

      self.fc1 = nn.Linear(embedding_dim, 128)
      self.fc2 = nn.Linear(128, num_classes)
      self.dropout = nn.Dropout(0.5)

  def forward(self, x):
      # x: [batch_size, seq_len] with word indices
      embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]

      # Average pooling over sequence
      pooled = embedded.mean(dim=1)  # [batch_size, embedding_dim]

      x = torch.relu(self.fc1(pooled))
      x = self.dropout(x)
      x = self.fc2(x)
      return x

# Example usage
vocab_size = 10000
embedding_dim = 100
num_classes = 5

# Option 1: Random initialization
model = TextClassifier(vocab_size, embedding_dim, num_classes)

# Option 2: Pre-trained embeddings
# vocab = ['the', 'a', 'learning', 'machine', ...]
# glove_embeddings = load_glove_embeddings('glove.6B.100d.txt', vocab)
# model = TextClassifier(vocab_size, embedding_dim, num_classes, glove_embeddings)

# Forward pass
batch_size = 32
seq_len = 50
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
output = model(input_ids)
print(f"Output shape: {output.shape}")  # [32, 5]`,
      explanation: 'This example shows how to load pre-trained GloVe embeddings and use them in a PyTorch model for text classification, with options to freeze or fine-tune the embeddings.'
    }
  ],
  interviewQuestions: [
    {
      question: 'What are word embeddings and how do they differ from one-hot encoding?',
      answer: `Word embeddings are dense vector representations of words in a continuous vector space where semantically similar words are mapped to nearby points. They transform discrete word symbols into continuous vectors that can be processed effectively by neural networks, representing one of the most fundamental breakthroughs in natural language processing.

One-hot encoding represents each word as a sparse binary vector with exactly one element set to 1 and all others set to 0. The dimensionality equals the vocabulary size, typically 10,000-100,000+ for real applications, making these vectors extremely sparse and high-dimensional. More critically, one-hot vectors treat all words as equally distant from each other - there's no notion of semantic similarity between "king" and "queen" versus "king" and "apple."

Word embeddings solve these fundamental limitations by mapping words to dense, low-dimensional vectors (typically 50-300 dimensions) that capture semantic and syntactic relationships. Similar words have similar vector representations, enabling the model to understand that "happy" and "joyful" are more related than "happy" and "table." This semantic understanding enables powerful capabilities like analogical reasoning (king - man + woman ≈ queen).

The key differences include: (1) Dimensionality - embeddings use 100-300 dimensions versus vocabulary-sized one-hot vectors, (2) Density - embeddings have meaningful values in all dimensions while one-hot vectors are 99.999% zeros, (3) Semantic meaning - embeddings capture relationships while one-hot encoding treats words as atomic symbols, and (4) Generalization - embeddings enable models to handle unseen word combinations by leveraging learned relationships.

Embeddings can be learned from large text corpora using methods like Word2Vec, GloVe, or FastText, capturing statistical relationships between words based on their co-occurrence patterns. This learned representation transfers across different NLP tasks, making pre-trained embeddings a powerful foundation for various applications. The transition from one-hot encoding to word embeddings marked a paradigm shift in NLP, enabling the development of more sophisticated and effective language models.`
    },
    {
      question: 'Explain the difference between Word2Vec CBOW and Skip-gram architectures.',
      answer: `Word2Vec offers two distinct architectures for learning word embeddings: Continuous Bag of Words (CBOW) and Skip-gram, which differ fundamentally in their prediction tasks and learning objectives, each with specific advantages depending on the dataset characteristics and application requirements.

CBOW (Continuous Bag of Words) predicts a target word from its surrounding context words. Given a window of context words around a target position, CBOW averages the embeddings of context words and uses this averaged representation to predict the center word. For example, given "The cat sat on the ___", CBOW would use the embeddings of ["The", "cat", "sat", "on", "the"] to predict "mat". The architecture involves averaging context word vectors and passing them through a single hidden layer to produce output probabilities over the vocabulary.

Skip-gram takes the opposite approach, predicting context words from a given target word. Given a center word, Skip-gram tries to predict each of the surrounding words independently. Using the same example, given "mat" as input, Skip-gram would try to predict each context word: "The", "cat", "sat", "on", "the". This results in multiple training examples from each window - one for each context position.

The key architectural differences include: (1) Input-output relationship - CBOW uses multiple inputs (context) to predict single output (target), while Skip-gram uses single input (target) to predict multiple outputs (context), (2) Training efficiency - CBOW is generally faster because it processes each window once, while Skip-gram creates multiple training examples per window, (3) Model complexity - CBOW averages context embeddings while Skip-gram maintains separate predictions for each context position.

Performance characteristics differ significantly: CBOW works better with frequent words and larger datasets because the averaging effect provides more stable gradients for common words. Skip-gram excels with rare words and smaller datasets because each word gets more training examples as both target and context, providing more learning opportunities for infrequent terms. CBOW tends to focus on syntactic relationships due to its averaging nature, while Skip-gram often captures more semantic relationships.

The choice between architectures depends on specific requirements: use CBOW for computational efficiency and when working with large corpora containing mostly frequent words, or choose Skip-gram when dealing with domain-specific vocabulary, rare words, or when semantic relationships are more important than syntactic ones.`
    },
    {
      question: 'What is negative sampling and why is it used in Word2Vec training?',
      answer: `Negative sampling is a crucial optimization technique in Word2Vec training that dramatically improves computational efficiency by approximating the full softmax computation with a much simpler binary classification problem. This technique was essential for making Word2Vec practical on large vocabularies and datasets.

The fundamental challenge in Word2Vec training lies in the output layer computation. The standard approach requires computing softmax over the entire vocabulary to predict the target word (CBOW) or context words (Skip-gram). For a vocabulary of size V, this means computing V dot products and normalizing across all vocabulary words for each training example. With vocabularies often containing 100,000+ words, this becomes computationally prohibitive.

Negative sampling transforms this multi-class problem into a series of binary classification problems. Instead of predicting the correct word from the entire vocabulary, the model learns to distinguish between positive word pairs (actual word-context pairs from the corpus) and negative pairs (randomly sampled word-context pairs that don't occur together). For each positive example, the algorithm samples a small number (typically 5-20) of negative examples by randomly selecting words from the vocabulary.

The training objective becomes: maximize the probability of observing actual word-context pairs while minimizing the probability of randomly sampled negative pairs. This is implemented using sigmoid functions instead of softmax, where each word-context pair is treated as an independent binary classification problem. The loss function becomes the sum of log-sigmoid for positive pairs and log(1-sigmoid) for negative pairs.

Key advantages include: (1) Computational efficiency - reduces computation from O(V) to O(k) where k is the number of negative samples, (2) Scalability - enables training on large vocabularies and datasets, (3) Quality preservation - maintains embedding quality while dramatically reducing training time, and (4) Theoretical justification - approximates the original softmax objective under certain conditions.

The sampling strategy for negative examples is crucial for performance. Word2Vec uses unigram distribution raised to the 3/4 power, which down-samples very frequent words while ensuring rare words have reasonable sampling probability. This sampling distribution balances between random sampling and frequency-based sampling, leading to better embedding quality.

Negative sampling has become a standard technique beyond Word2Vec, influencing many subsequent embedding methods and demonstrating how clever approximations can make previously intractable problems computationally feasible while maintaining solution quality.`
    },
    {
      question: 'How does GloVe differ from Word2Vec in its approach to learning embeddings?',
      answer: `GloVe (Global Vectors for Word Representation) represents a fundamentally different approach to learning word embeddings compared to Word2Vec, combining the advantages of global matrix factorization methods with the benefits of local context window methods to create a unified framework that leverages corpus-wide statistical information.

Word2Vec learns embeddings through local context windows using either CBOW or Skip-gram architectures. It processes one context window at a time, making predictions based on immediate word neighborhoods. While this captures local semantic relationships effectively, it doesn't directly utilize global corpus statistics like overall word co-occurrence frequencies across the entire dataset.

GloVe takes a global approach by first constructing a word co-occurrence matrix from the entire corpus, then factorizing this matrix to obtain word vectors. The co-occurrence matrix X records how frequently word i appears in the context of word j across all documents. GloVe then learns embeddings by fitting a log-bilinear model that explains these global co-occurrence statistics.

The key insight behind GloVe is that ratios of co-occurrence probabilities can encode semantic relationships. For example, the ratio P(ice|solid)/P(ice|gas) should be much larger than P(steam|solid)/P(steam|gas), capturing the relationship between states of matter. GloVe's objective function is designed to encode these ratio relationships in the learned vector space.

The GloVe objective function combines several terms: (1) A weighted least squares model that fits word vectors to co-occurrence log-probabilities, (2) Bias terms for individual words to handle frequency differences, (3) A weighting function that down-weights very frequent word pairs while ensuring rare pairs still contribute to learning. This creates a balance between local and global information.

Key differences include: (1) Information utilization - GloVe uses global corpus statistics while Word2Vec focuses on local contexts, (2) Training approach - GloVe factorizes a pre-computed matrix while Word2Vec uses online learning, (3) Theoretical foundation - GloVe has clearer theoretical justification based on co-occurrence ratios, and (4) Computational characteristics - GloVe requires storing co-occurrence matrices but can leverage efficient matrix factorization techniques.

Performance-wise, GloVe often produces embeddings with better performance on word analogy tasks and tends to capture more nuanced semantic relationships due to its global perspective. However, Word2Vec can be more memory-efficient and easier to scale to very large corpora since it doesn't require storing large co-occurrence matrices.

Both methods have influenced subsequent embedding techniques, with GloVe's global perspective inspiring methods that combine local and global information, while Word2Vec's efficiency has influenced many scalable embedding approaches.`
    },
    {
      question: 'What are the advantages of FastText over Word2Vec?',
      answer: `FastText, developed by Facebook Research, extends Word2Vec with several crucial innovations that address fundamental limitations of traditional word-level embeddings, particularly the out-of-vocabulary (OOV) problem and the inability to leverage morphological information within words.

The primary innovation in FastText is its subword-aware approach. Instead of treating words as atomic units like Word2Vec, FastText represents each word as a bag of character n-grams (typically 3-6 characters) plus the word itself. For example, "running" might be decomposed into character trigrams: "<ru", "run", "unn", "nni", "nin", "ing", "ng>", where < and > represent word boundaries. The final word embedding is the sum of embeddings for all these subword units.

This subword approach provides several critical advantages: (1) OOV handling - FastText can generate embeddings for words not seen during training by composing them from learned subword embeddings, (2) Morphological awareness - related words like "run", "running", "runner" share subword components, enabling the model to understand morphological relationships, (3) Rare word handling - even rare words benefit from shared subword information with more frequent words, and (4) Multilingual effectiveness - particularly beneficial for morphologically rich languages like German, Turkish, or Finnish.

FastText retains Word2Vec's efficient training algorithms (both CBOW and Skip-gram) while extending them to operate on subword units. During training, the model learns embeddings for all character n-grams that appear in the vocabulary, then represents words as sums of their constituent subword embeddings. This approach maintains Word2Vec's computational efficiency while adding subword-level understanding.

Additional advantages include: (1) Better performance on word similarity tasks, especially for morphologically related words, (2) Improved handling of misspellings and typos through subword overlap, (3) More robust embeddings for domain-specific or technical vocabularies, (4) Better transfer learning capabilities across related languages or domains, and (5) Reduced vocabulary size requirements since many concepts can be represented through subword combinations.

The subword approach particularly shines in practical applications where vocabulary coverage is crucial. Web-scale applications, social media analysis, and multilingual systems benefit significantly from FastText's ability to handle previously unseen words and variations. Medical and scientific domains, which contain many compound words and technical terms, also see substantial improvements.

However, FastText also has trade-offs including slightly increased computational complexity due to subword processing, larger model sizes when storing all n-gram embeddings, and potential noise from irrelevant character combinations. Despite these limitations, FastText's innovations have influenced many subsequent embedding methods, establishing subword modeling as a standard practice in modern NLP systems.`
    },
    {
      question: 'What is the difference between static and contextual word embeddings?',
      answer: `Static and contextual word embeddings represent two fundamentally different paradigms for word representation that reflect the evolution of NLP from simpler lookup-based approaches to sophisticated context-aware models that understand language nuance and ambiguity.

Static word embeddings, exemplified by Word2Vec, GloVe, and FastText, assign a single, fixed vector representation to each word regardless of context. Once trained, the word "bank" has the same embedding whether it appears in "river bank" or "savings bank." These embeddings capture general semantic properties and relationships but cannot distinguish between different senses or meanings of polysemous words.

Contextual word embeddings, introduced by models like ELMo, BERT, and GPT, generate different vector representations for the same word based on its surrounding context. The word "bank" would receive different embeddings in different sentences, with the model learning to emphasize financial meanings in one context and geographical meanings in another. This represents a paradigm shift from static lookup tables to dynamic, context-dependent representations.

The key technical differences include: (1) Representation generation - static embeddings use simple lookup operations while contextual embeddings require processing entire sequences through neural networks, (2) Context sensitivity - static embeddings ignore context while contextual embeddings are computed based on surrounding words, (3) Computational requirements - static embeddings enable fast lookup while contextual embeddings require expensive inference through large neural networks, and (4) Model architecture - static embeddings are typically learned through simpler models while contextual embeddings emerge from sophisticated transformer architectures.

Contextual embeddings provide several crucial advantages: (1) Polysemy resolution - different word senses receive different representations, (2) Better compositional understanding - meanings emerge from word interactions rather than isolated semantics, (3) Dynamic adaptation - representations adapt to specific contexts and domains, (4) Improved downstream performance - contextual representations typically yield better results on NLP tasks, and (5) Syntactic awareness - embeddings capture grammatical roles and relationships.

However, static embeddings still have important advantages: (1) Computational efficiency - constant-time lookup versus expensive neural network inference, (2) Interpretability - static relationships are easier to analyze and understand, (3) Stability - consistent representations enable reliable similarity comparisons, (4) Storage efficiency - single vector per word versus context-dependent computation, and (5) Simplicity - easier to integrate into traditional ML pipelines.

The choice between static and contextual embeddings depends on application requirements. Static embeddings work well for tasks where context is less critical and computational efficiency is paramount, such as basic similarity search or simple classification. Contextual embeddings excel in complex NLP tasks requiring nuanced understanding like question answering, reading comprehension, and sophisticated text analysis. Modern applications often use hybrid approaches, leveraging contextual embeddings for complex reasoning while maintaining static embeddings for efficient similarity operations.`
    }
  ],
  quizQuestions: [
    {
      id: 'embed1',
      question: 'What is the main advantage of word embeddings over one-hot encoding?',
      options: ['Faster computation', 'Capture semantic relationships', 'Use less memory', 'More interpretable'],
      correctAnswer: 1,
      explanation: 'Word embeddings capture semantic relationships by mapping words to dense vectors where similar words are nearby in the vector space. One-hot vectors treat all words as equally distant with no semantic information.'
    },
    {
      id: 'embed2',
      question: 'In Word2Vec, what does the Skip-gram model predict?',
      options: ['Target word from context', 'Context words from target', 'Next word in sequence', 'Word sentiment'],
      correctAnswer: 1,
      explanation: 'Skip-gram predicts context words given a target word, while CBOW does the opposite. Skip-gram works better with rare words and smaller datasets.'
    },
    {
      id: 'embed3',
      question: 'Which embedding method can handle out-of-vocabulary words by using character n-grams?',
      options: ['Word2Vec', 'GloVe', 'FastText', 'One-hot encoding'],
      correctAnswer: 2,
      explanation: 'FastText represents words as bags of character n-grams, allowing it to generate embeddings for unseen words by combining n-gram embeddings. This is particularly useful for rare words and morphologically rich languages.'
    }
  ]
};
