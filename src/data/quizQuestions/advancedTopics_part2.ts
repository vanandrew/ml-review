import { QuizQuestion } from '../../types';

// Model Compression - 20 questions
export const modelCompressionQuestions: QuizQuestion[] = [
  {
    id: 'mc1',
    question: 'Why is model compression important?',
    options: ['Not important', 'Reduces size/memory/compute for deployment on edge devices', 'Makes models worse', 'Only for training'],
    correctAnswer: 1,
    explanation: 'Compression enables deployment on mobile, embedded devices with limited resources.'
  },
  {
    id: 'mc2',
    question: 'What are the main compression techniques?',
    options: ['Only one method', 'Quantization, pruning, knowledge distillation, low-rank factorization', 'No methods', 'Only quantization'],
    correctAnswer: 1,
    explanation: 'Multiple approaches: reduce precision, remove weights, train smaller models, decompose layers.'
  },
  {
    id: 'mc3',
    question: 'What is quantization?',
    options: ['Increasing precision', 'Reducing numerical precision of weights/activations (e.g., float32 → int8)', 'No change', 'Pruning'],
    correctAnswer: 1,
    explanation: 'Quantization maps high-precision values to lower precision (8-bit, 4-bit) reducing memory 4-8×.'
  },
  {
    id: 'mc4',
    question: 'What is post-training quantization?',
    options: ['Training from scratch', 'Quantizing pre-trained model without retraining', 'Requires retraining', 'No quantization'],
    correctAnswer: 1,
    explanation: 'Post-training quantization converts trained model to lower precision with minimal/no retraining.'
  },
  {
    id: 'mc5',
    question: 'What is quantization-aware training?',
    options: ['Post-training only', 'Training with quantization in forward pass to adapt weights', 'No training', 'Same as post-training'],
    correctAnswer: 1,
    explanation: 'QAT simulates quantization during training, allowing model to adapt for better low-precision accuracy.'
  },
  {
    id: 'mc6',
    question: 'What is pruning?',
    options: ['Adding weights', 'Removing unnecessary weights/neurons to reduce model size', 'No removal', 'Increasing size'],
    correctAnswer: 1,
    explanation: 'Pruning identifies and removes low-importance weights, creating sparse networks.'
  },
  {
    id: 'mc7',
    question: 'What is magnitude pruning?',
    options: ['Random removal', 'Removing weights with smallest absolute values', 'Largest weights', 'No criterion'],
    correctAnswer: 1,
    explanation: 'Magnitude pruning assumes small weights contribute less, removing them first.'
  },
  {
    id: 'mc8',
    question: 'What is structured vs unstructured pruning?',
    options: ['Same thing', 'Structured: remove entire channels/layers; Unstructured: individual weights', 'Only structured', 'Only unstructured'],
    correctAnswer: 1,
    explanation: 'Unstructured creates sparse matrices (needs special hardware); structured removes whole units (hardware friendly).'
  },
  {
    id: 'mc9',
    question: 'What is the Lottery Ticket Hypothesis?',
    options: ['Random networks work', 'Dense networks contain sparse subnetworks that train to same accuracy', 'All tickets lose', 'No hypothesis'],
    correctAnswer: 1,
    explanation: 'Lottery Ticket: random initialization contains "winning tickets" (subnetworks) that match full network performance.'
  },
  {
    id: 'mc10',
    question: 'What is knowledge distillation?',
    options: ['Training from scratch', 'Training smaller student model to mimic larger teacher model', 'No transfer', 'Same size models'],
    correctAnswer: 1,
    explanation: 'Distillation transfers knowledge from large teacher to compact student via soft targets.'
  },
  {
    id: 'mc11',
    question: 'What are soft targets in distillation?',
    options: ['Hard labels only', 'Teacher\'s softmax probabilities with temperature, not just argmax', 'Binary', 'One-hot'],
    correctAnswer: 1,
    explanation: 'Soft targets (temperature-scaled probabilities) convey richer information than hard labels.'
  },
  {
    id: 'mc12',
    question: 'What is the temperature parameter in distillation?',
    options: ['Physical temperature', 'Softens probability distribution: higher T → more uniform, revealing dark knowledge', 'No effect', 'Learning rate'],
    correctAnswer: 1,
    explanation: 'Temperature T>1 smooths softmax, exposing relationships between classes (dark knowledge).'
  },
  {
    id: 'mc13',
    question: 'What is low-rank factorization?',
    options: ['High-rank only', 'Decomposing weight matrices into products of smaller matrices', 'No decomposition', 'Increasing rank'],
    correctAnswer: 1,
    explanation: 'Factorization replaces W (m×n) with U (m×r) and V (r×n) where r << min(m,n).'
  },
  {
    id: 'mc14',
    question: 'What is neural architecture search (NAS) for compression?',
    options: ['Manual design', 'Automatically finding efficient architectures with fewer parameters', 'No search', 'Only large models'],
    correctAnswer: 1,
    explanation: 'NAS explores architecture space to find compact, efficient networks (e.g., MobileNet, EfficientNet).'
  },
  {
    id: 'mc15',
    question: 'What is MobileNet?',
    options: ['Desktop model', 'Efficient CNN using depthwise separable convolutions for mobile deployment', 'Largest model', 'No efficiency'],
    correctAnswer: 1,
    explanation: 'MobileNet uses depthwise + pointwise convolutions, drastically reducing parameters and FLOPs.'
  },
  {
    id: 'mc16',
    question: 'What is mixed precision training?',
    options: ['Single precision', 'Using float16 for computation, float32 for gradients to speed training', 'Only int8', 'No mixing'],
    correctAnswer: 1,
    explanation: 'Mixed precision (FP16/FP32) accelerates training on modern GPUs while maintaining accuracy.'
  },
  {
    id: 'mc17',
    question: 'What accuracy-size tradeoffs exist?',
    options: ['No tradeoff', 'Aggressive compression reduces accuracy; must balance size reduction with performance', 'No impact', 'Always better'],
    correctAnswer: 1,
    explanation: 'Compression trades accuracy for efficiency; goal is minimal accuracy loss with maximal compression.'
  },
  {
    id: 'mc18',
    question: 'What is int8 quantization benefit?',
    options: ['No benefit', '4× memory reduction, 2-4× speedup on CPUs/edge devices', 'Slower', 'More memory'],
    correctAnswer: 1,
    explanation: 'Int8 reduces model size 4×, enables faster inference on CPUs and edge accelerators.'
  },
  {
    id: 'mc19',
    question: 'Can you combine compression techniques?',
    options: ['Only one at a time', 'Yes: quantization + pruning + distillation for maximum compression', 'Never combine', 'No benefit'],
    correctAnswer: 1,
    explanation: 'Techniques are complementary: distill large model, then quantize and prune for extreme compression.'
  },
  {
    id: 'mc20',
    question: 'What tools exist for compression?',
    options: ['No tools', 'TensorFlow Lite, PyTorch quantization, ONNX Runtime, TensorRT, Neural Compressor', 'Manual only', 'One tool'],
    correctAnswer: 1,
    explanation: 'Many frameworks provide compression: TFLite, PyTorch (torch.quantization), NVIDIA TensorRT, Intel Neural Compressor.'
  }
];

// Federated Learning - 20 questions
export const federatedLearningQuestions: QuizQuestion[] = [
  {
    id: 'fl1',
    question: 'What is federated learning?',
    options: ['Centralized training', 'Training models across decentralized devices without centralizing data', 'Cloud only', 'No distribution'],
    correctAnswer: 1,
    explanation: 'Federated learning trains on distributed data (phones, hospitals) while keeping data local.'
  },
  {
    id: 'fl2',
    question: 'Why is federated learning important?',
    options: ['Not important', 'Preserves privacy, enables training on distributed sensitive data', 'No privacy', 'Only speed'],
    correctAnswer: 1,
    explanation: 'FL crucial for privacy-sensitive applications (healthcare, mobile keyboards) where data can\'t be centralized.'
  },
  {
    id: 'fl3',
    question: 'What is the basic FL process?',
    options: ['Send data to server', 'Server sends model → devices train locally → send updates → aggregate', 'Centralized', 'No aggregation'],
    correctAnswer: 1,
    explanation: 'FL: distribute model, train on local data, collect gradient/weight updates, aggregate, repeat.'
  },
  {
    id: 'fl4',
    question: 'What is Federated Averaging (FedAvg)?',
    options: ['No averaging', 'Average model weights from clients weighted by local dataset size', 'Median', 'Random selection'],
    correctAnswer: 1,
    explanation: 'FedAvg: weighted average of client models proportional to local data amounts.'
  },
  {
    id: 'fl5',
    question: 'What challenge does non-IID data pose?',
    options: ['No challenge', 'Heterogeneous data distributions across clients degrade convergence', 'Perfect IID', 'No impact'],
    correctAnswer: 1,
    explanation: 'Non-IID data (different distributions per client) causes slower, less stable FL convergence.'
  },
  {
    id: 'fl6',
    question: 'What is data heterogeneity?',
    options: ['Identical data', 'Clients have different data distributions, quantities, features', 'Homogeneous', 'Same everywhere'],
    correctAnswer: 1,
    explanation: 'Heterogeneity: clients may have different classes, skewed distributions, or varying data volumes.'
  },
  {
    id: 'fl7',
    question: 'What is the communication bottleneck?',
    options: ['No bottleneck', 'Sending models/gradients is expensive; need compression and fewer rounds', 'Unlimited bandwidth', 'No cost'],
    correctAnswer: 1,
    explanation: 'FL requires many communication rounds; bandwidth/latency is major bottleneck, not computation.'
  },
  {
    id: 'fl8',
    question: 'How to reduce communication cost?',
    options: ['More communication', 'Gradient compression, quantization, sparse updates, fewer rounds', 'No reduction', 'Send more'],
    correctAnswer: 1,
    explanation: 'Reduce communication via: quantization, sparsification, local epochs, model compression.'
  },
  {
    id: 'fl9',
    question: 'What is differential privacy in FL?',
    options: ['No privacy', 'Adding noise to updates to protect individual data points', 'Perfect privacy', 'No noise'],
    correctAnswer: 1,
    explanation: 'Differential privacy adds calibrated noise to gradients, preventing inference of individual training samples.'
  },
  {
    id: 'fl10',
    question: 'What is secure aggregation?',
    options: ['No security', 'Cryptographic protocol ensuring server sees only aggregate, not individual updates', 'Plain aggregation', 'No encryption'],
    correctAnswer: 1,
    explanation: 'Secure aggregation uses crypto (e.g., secret sharing) so server computes sum without seeing individual updates.'
  },
  {
    id: 'fl11',
    question: 'What is client selection?',
    options: ['All clients always', 'Choosing subset of clients per round (availability, resources)', 'Fixed clients', 'No selection'],
    correctAnswer: 1,
    explanation: 'FL selects subset of available clients each round due to resource constraints and device availability.'
  },
  {
    id: 'fl12',
    question: 'What is system heterogeneity?',
    options: ['Identical devices', 'Clients have varying compute, memory, network capabilities', 'Homogeneous', 'Same hardware'],
    correctAnswer: 1,
    explanation: 'Devices differ in CPU, memory, battery, connectivity; must handle stragglers and dropouts.'
  },
  {
    id: 'fl13',
    question: 'What are FL applications?',
    options: ['None', 'Mobile keyboards, healthcare, IoT, finance (privacy-sensitive domains)', 'Public data only', 'No applications'],
    correctAnswer: 1,
    explanation: 'FL deployed in: Gboard (Google Keyboard), medical imaging, smart homes, fraud detection.'
  },
  {
    id: 'fl14',
    question: 'What is cross-device vs cross-silo FL?',
    options: ['Same thing', 'Cross-device: millions of phones; Cross-silo: few organizations/datacenters', 'Only devices', 'Only silos'],
    correctAnswer: 1,
    explanation: 'Cross-device (mobile FL): many unreliable clients; Cross-silo (enterprise): few reliable institutions.'
  },
  {
    id: 'fl15',
    question: 'What is personalized FL?',
    options: ['One global model', 'Adapting global model to individual clients\' data distributions', 'No personalization', 'Identical models'],
    correctAnswer: 1,
    explanation: 'Personalized FL creates client-specific models by fine-tuning or multi-task learning from global model.'
  },
  {
    id: 'fl16',
    question: 'What is model poisoning attack?',
    options: ['No attacks', 'Malicious clients send bad updates to corrupt global model', 'Harmless', 'No security risk'],
    correctAnswer: 1,
    explanation: 'Adversarial clients can send malicious gradients to degrade model or inject backdoors.'
  },
  {
    id: 'fl17',
    question: 'How to defend against poisoning?',
    options: ['No defense', 'Byzantine-robust aggregation, outlier detection, secure multi-party computation', 'Trust all', 'No mitigation'],
    correctAnswer: 1,
    explanation: 'Defenses: robust aggregation (Krum, median), detecting anomalous updates, cryptographic verification.'
  },
  {
    id: 'fl18',
    question: 'What frameworks support FL?',
    options: ['No frameworks', 'TensorFlow Federated, PySyft, Flower, NVIDIA FLARE', 'Manual only', 'One framework'],
    correctAnswer: 1,
    explanation: 'FL frameworks: TFF (TensorFlow Federated), PySyft, Flower (framework-agnostic), NVIDIA FLARE (healthcare).'
  },
  {
    id: 'fl19',
    question: 'What is the convergence challenge?',
    options: ['Fast convergence', 'Non-IID data and limited communication slow convergence vs centralized', 'Instant', 'No challenge'],
    correctAnswer: 1,
    explanation: 'FL converges slower than centralized training due to heterogeneity and communication constraints.'
  },
  {
    id: 'fl20',
    question: 'What is the future of FL?',
    options: ['Declining', 'Growing importance for privacy-preserving ML, edge AI, regulatory compliance', 'No future', 'Obsolete'],
    correctAnswer: 1,
    explanation: 'FL increasingly critical as privacy regulations (GDPR) strengthen and edge computing grows.'
  }
];

// Few-Shot Learning - 20 questions
export const fewShotLearningQuestions: QuizQuestion[] = [
  {
    id: 'fsl1',
    question: 'What is few-shot learning?',
    options: ['Large data training', 'Learning from very few examples (typically 1-10 per class)', 'No examples', 'Always 1000s'],
    correctAnswer: 1,
    explanation: 'Few-shot learning trains models that generalize from minimal labeled data per new class.'
  },
  {
    id: 'fsl2',
    question: 'What is N-way K-shot classification?',
    options: ['Single class', 'N classes with K examples each', 'Unlimited examples', 'One example total'],
    correctAnswer: 1,
    explanation: 'N-way K-shot: classify among N classes given K training examples per class (e.g., 5-way 1-shot).'
  },
  {
    id: 'fsl3',
    question: 'What is one-shot learning?',
    options: ['Many examples', 'Learning from single example per class (K=1)', '10 examples', 'No examples'],
    correctAnswer: 1,
    explanation: 'One-shot: classify new instances given only one labeled example of each class.'
  },
  {
    id: 'fsl4',
    question: 'What is zero-shot learning?',
    options: ['Few examples', 'Classifying classes never seen during training using side information', 'One example', 'Many examples'],
    correctAnswer: 1,
    explanation: 'Zero-shot: recognize new classes without any examples, using attributes or embeddings.'
  },
  {
    id: 'fsl5',
    question: 'Why is few-shot learning hard?',
    options: ['Easy problem', 'Insufficient data for conventional training; high risk of overfitting', 'No challenges', 'Too much data'],
    correctAnswer: 1,
    explanation: 'Standard deep learning needs thousands of examples; few-shot must generalize from handful.'
  },
  {
    id: 'fsl6',
    question: 'What is meta-learning?',
    options: ['Standard learning', 'Learning to learn: training across many tasks to quickly adapt to new tasks', 'Single task', 'No learning'],
    correctAnswer: 1,
    explanation: 'Meta-learning trains on distribution of tasks, learning initialization/optimizer for fast adaptation.'
  },
  {
    id: 'fsl7',
    question: 'What is MAML?',
    options: ['Architecture', 'Model-Agnostic Meta-Learning: finds initialization good for quick fine-tuning', 'Dataset', 'Loss function'],
    correctAnswer: 1,
    explanation: 'MAML optimizes for initialization that reaches high accuracy with few gradient steps on new tasks.'
  },
  {
    id: 'fsl8',
    question: 'What is the MAML objective?',
    options: ['Single task loss', 'Minimize loss after few gradient steps across task distribution', 'Pre-training', 'No objective'],
    correctAnswer: 1,
    explanation: 'MAML meta-trains to find θ such that θ - α∇L achieves low loss on new tasks after one/few updates.'
  },
  {
    id: 'fsl9',
    question: 'What is prototypical networks?',
    options: ['No prototypes', 'Learn embedding where classes cluster around prototypes (class means)', 'No embeddings', 'Random centers'],
    correctAnswer: 1,
    explanation: 'Prototypical networks compute class prototype (mean embedding), classify by nearest prototype.'
  },
  {
    id: 'fsl10',
    question: 'How do prototypical networks classify?',
    options: ['Linear classifier', 'Nearest prototype in embedding space', 'Random', 'No classification'],
    correctAnswer: 1,
    explanation: 'Embed query and support examples, compute class prototypes, assign query to nearest prototype.'
  },
  {
    id: 'fsl11',
    question: 'What is siamese networks?',
    options: ['Single network', 'Twin networks learning similarity metric between examples', 'Three networks', 'No comparison'],
    correctAnswer: 1,
    explanation: 'Siamese networks learn embedding where similar pairs are close, dissimilar are far.'
  },
  {
    id: 'fsl12',
    question: 'What is matching networks?',
    options: ['No matching', 'Attention-based approach using support set as memory', 'No attention', 'Single example'],
    correctAnswer: 1,
    explanation: 'Matching networks use attention over support set to classify queries via weighted nearest neighbors.'
  },
  {
    id: 'fsl13',
    question: 'What is transfer learning for few-shot?',
    options: ['Training from scratch', 'Pre-train on large dataset, fine-tune/adapt to few-shot target', 'No transfer', 'Random init'],
    correctAnswer: 1,
    explanation: 'Transfer learning provides strong initialization; fine-tune with few examples on new classes.'
  },
  {
    id: 'fsl14',
    question: 'What is episodic training?',
    options: ['Batch training', 'Training on episodes (few-shot tasks) sampled from base classes', 'Single batch', 'No episodes'],
    correctAnswer: 1,
    explanation: 'Episodic training simulates few-shot scenarios during meta-training, sampling N-way K-shot tasks.'
  },
  {
    id: 'fsl15',
    question: 'What is data augmentation for few-shot?',
    options: ['No augmentation', 'Critical for few-shot to artificially increase training diversity', 'Not useful', 'Harmful'],
    correctAnswer: 1,
    explanation: 'Augmentation essential in few-shot to prevent overfitting; includes hallucination, mixup.'
  },
  {
    id: 'fsl16',
    question: 'What are few-shot learning applications?',
    options: ['None', 'Medical imaging, rare species recognition, personalization, drug discovery', 'Only common classes', 'No applications'],
    correctAnswer: 1,
    explanation: 'Few-shot useful when labels expensive: medical diagnosis, endangered species, customized ML.'
  },
  {
    id: 'fsl17',
    question: 'What is the difference between few-shot and transfer learning?',
    options: ['Same thing', 'Few-shot explicitly trains for quick adaptation with minimal data', 'No difference', 'Unrelated'],
    correctAnswer: 1,
    explanation: 'Transfer learning pre-trains then fine-tunes; few-shot meta-learns specifically for low-data scenarios.'
  },
  {
    id: 'fsl18',
    question: 'What is transductive few-shot learning?',
    options: ['No test data', 'Using unlabeled test examples during inference to improve predictions', 'Only training', 'No unlabeled'],
    correctAnswer: 1,
    explanation: 'Transductive approaches leverage unlabeled query set structure during few-shot classification.'
  },
  {
    id: 'fsl19',
    question: 'What are common few-shot benchmarks?',
    options: ['ImageNet only', 'Omniglot, miniImageNet, tieredImageNet', 'CIFAR-10', 'MNIST'],
    correctAnswer: 1,
    explanation: 'Benchmarks: Omniglot (character recognition), miniImageNet (100 classes, 600 images each), tieredImageNet.'
  },
  {
    id: 'fsl20',
    question: 'What is the relationship with prompting in LLMs?',
    options: ['Unrelated', 'Few-shot prompting in LLMs is form of few-shot learning via in-context learning', 'No relationship', 'Different fields'],
    correctAnswer: 1,
    explanation: 'LLM few-shot prompting demonstrates few-shot learning capability through in-context examples.'
  }
];

// Multi-Modal Learning - 20 questions
export const multiModalQuestions: QuizQuestion[] = [
  {
    id: 'mm1',
    question: 'What is multi-modal learning?',
    options: ['Single modality', 'Learning from multiple data types: text, images, audio, video', 'Text only', 'Images only'],
    correctAnswer: 1,
    explanation: 'Multi-modal learning combines information from different modalities for richer representations.'
  },
  {
    id: 'mm2',
    question: 'What are examples of modalities?',
    options: ['Only images', 'Vision, text, audio, video, sensor data, graphs', 'Single type', 'No variety'],
    correctAnswer: 1,
    explanation: 'Modalities: images, natural language, speech, time-series, 3D scans, etc.'
  },
  {
    id: 'mm3',
    question: 'Why combine modalities?',
    options: ['No benefit', 'Complementary information improves robustness and understanding', 'Worse performance', 'No synergy'],
    correctAnswer: 1,
    explanation: 'Different modalities capture different aspects; fusion yields more complete understanding.'
  },
  {
    id: 'mm4',
    question: 'What is early fusion?',
    options: ['Late fusion', 'Combining raw inputs or low-level features before processing', 'Output fusion', 'No fusion'],
    correctAnswer: 1,
    explanation: 'Early fusion concatenates inputs/features at start, processing jointly through network.'
  },
  {
    id: 'mm5',
    question: 'What is late fusion?',
    options: ['Early fusion', 'Processing modalities separately, combining predictions/high-level features', 'Input fusion', 'No fusion'],
    correctAnswer: 1,
    explanation: 'Late fusion processes each modality independently, merges final representations or outputs.'
  },
  {
    id: 'mm6',
    question: 'What is intermediate fusion?',
    options: ['Only early/late', 'Combining features at intermediate layers', 'No intermediate', 'One fusion'],
    correctAnswer: 1,
    explanation: 'Intermediate fusion merges mid-level representations, balancing early and late approaches.'
  },
  {
    id: 'mm7',
    question: 'What is CLIP?',
    options: ['CNN only', 'Contrastive Language-Image Pre-training: aligns vision and language', 'Text only', 'No alignment'],
    correctAnswer: 1,
    explanation: 'CLIP (OpenAI) learns joint vision-language space via contrastive learning on 400M image-text pairs.'
  },
  {
    id: 'mm8',
    question: 'How does CLIP work?',
    options: ['Classification only', 'Learns to match images with corresponding text descriptions via contrastive loss', 'No matching', 'Supervised'],
    correctAnswer: 1,
    explanation: 'CLIP maximizes similarity between correct image-text pairs, minimizes for incorrect pairs.'
  },
  {
    id: 'mm9',
    question: 'What are CLIP capabilities?',
    options: ['Only training classes', 'Zero-shot classification using text descriptions as class labels', 'Requires fine-tuning', 'No flexibility'],
    correctAnswer: 1,
    explanation: 'CLIP enables zero-shot classification by comparing image to text descriptions of classes.'
  },
  {
    id: 'mm10',
    question: 'What is image captioning?',
    options: ['Classification', 'Generating text descriptions of images', 'No text', 'Only labels'],
    correctAnswer: 1,
    explanation: 'Image captioning combines vision (CNN/ViT) with language generation (LSTM/Transformer decoder).'
  },
  {
    id: 'mm11',
    question: 'What is visual question answering (VQA)?',
    options: ['No questions', 'Answering natural language questions about images', 'Only captions', 'No interaction'],
    correctAnswer: 1,
    explanation: 'VQA requires understanding both image content and question to generate correct answer.'
  },
  {
    id: 'mm12',
    question: 'What is text-to-image generation?',
    options: ['Image-to-text', 'Synthesizing images from text descriptions', 'No generation', 'Only retrieval'],
    correctAnswer: 1,
    explanation: 'Text-to-image (DALL-E, Stable Diffusion, Midjourney) creates images matching text prompts.'
  },
  {
    id: 'mm13',
    question: 'What is DALL-E?',
    options: ['Text only', 'OpenAI\'s text-to-image model using diffusion or autoregressive approaches', 'Image only', 'No generation'],
    correctAnswer: 1,
    explanation: 'DALL-E generates photorealistic images from text; DALL-E 2/3 use diffusion models.'
  },
  {
    id: 'mm14',
    question: 'What is Stable Diffusion?',
    options: ['Proprietary', 'Open-source text-to-image diffusion model', 'Closed', 'No generation'],
    correctAnswer: 1,
    explanation: 'Stable Diffusion (Stability AI) is open-source latent diffusion model for text-to-image synthesis.'
  },
  {
    id: 'mm15',
    question: 'What is cross-modal retrieval?',
    options: ['Same modality', 'Retrieving one modality using queries from another (text→image or image→text)', 'No retrieval', 'Single mode'],
    correctAnswer: 1,
    explanation: 'Cross-modal retrieval finds images given text query or vice versa via shared embedding space.'
  },
  {
    id: 'mm16',
    question: 'What is audio-visual learning?',
    options: ['Audio only', 'Learning from synchronized audio and video', 'Visual only', 'No sync'],
    correctAnswer: 1,
    explanation: 'Audio-visual models align speech with faces, sound with source objects (e.g., SoundNet).'
  },
  {
    id: 'mm17',
    question: 'What is the alignment problem?',
    options: ['Perfect alignment', 'Different modalities have different structures, semantics, granularity', 'No problem', 'Easy alignment'],
    correctAnswer: 1,
    explanation: 'Alignment challenge: modalities differ in dimensionality, timing, and semantic representation.'
  },
  {
    id: 'mm18',
    question: 'What is attention for multi-modal fusion?',
    options: ['No attention', 'Cross-modal attention aligns and weights modalities dynamically', 'Fixed weights', 'No weighting'],
    correctAnswer: 1,
    explanation: 'Cross-attention lets modalities attend to each other, learning which parts correspond.'
  },
  {
    id: 'mm19',
    question: 'What are multi-modal applications?',
    options: ['None', 'Autonomous driving, medical imaging, video understanding, accessibility, robotics', 'Single mode sufficient', 'No applications'],
    correctAnswer: 1,
    explanation: 'Multi-modal crucial for: self-driving (camera+lidar+radar), medical (MRI+CT+text), assistive tech.'
  },
  {
    id: 'mm20',
    question: 'What is the future of multi-modal AI?',
    options: ['Single modality', 'Unified models like GPT-4 handling multiple modalities seamlessly', 'Declining', 'No progress'],
    correctAnswer: 1,
    explanation: 'Trend toward unified multi-modal models (GPT-4, Gemini) processing text, images, audio, video together.'
  }
];
