import { QuizQuestion } from '../../types';

// Transfer Learning - 25 questions
export const transferLearningQuestions: QuizQuestion[] = [
  {
    id: 'tl1',
    question: 'What is transfer learning?',
    options: ['Training from scratch', 'Using knowledge from one task to improve learning on another task', 'Transferring data', 'Moving models between devices'],
    correctAnswer: 1,
    explanation: 'Transfer learning leverages pre-trained models from one task (usually on large datasets) to accelerate and improve learning on related tasks.'
  },
  {
    id: 'tl2',
    question: 'Why is transfer learning useful?',
    options: ['Always required', 'Saves time, requires less data, often achieves better performance', 'Only for small models', 'Slows training'],
    correctAnswer: 1,
    explanation: 'Transfer learning enables training with limited data by starting from features learned on large datasets, often achieving better results faster.'
  },
  {
    id: 'tl3',
    question: 'What is a pre-trained model?',
    options: ['Random model', 'Model trained on large dataset (e.g., ImageNet), used as starting point', 'Untrained model', 'Final model'],
    correctAnswer: 1,
    explanation: 'Pre-trained models have learned useful features from large datasets and serve as initialization for new tasks.'
  },
  {
    id: 'tl4',
    question: 'What is fine-tuning?',
    options: ['Training from scratch', 'Continuing training of pre-trained model on new task', 'Testing only', 'Data preprocessing'],
    correctAnswer: 1,
    explanation: 'Fine-tuning adjusts the weights of a pre-trained model by training it on a new task, usually with a smaller learning rate.'
  },
  {
    id: 'tl5',
    question: 'What is feature extraction in transfer learning?',
    options: ['Training all layers', 'Using pre-trained layers as fixed feature extractor, only training new layers', 'Data augmentation', 'Model compression'],
    correctAnswer: 1,
    explanation: 'Feature extraction freezes pre-trained layers and only trains new layers (e.g., classifier), useful when data is very limited.'
  },
  {
    id: 'tl6',
    question: 'When should you freeze more layers vs fewer layers?',
    options: ['Always freeze all', 'Freeze more with less data and similar tasks; freeze less with more data or different tasks', 'Never freeze', 'Random'],
    correctAnswer: 1,
    explanation: 'With little data or similar tasks, freeze more layers. With more data or different tasks, fine-tune more layers.'
  },
  {
    id: 'tl7',
    question: 'What layers typically transfer well between tasks?',
    options: ['Output layers', 'Early layers (low-level features like edges)', 'All layers equally', 'No layers'],
    correctAnswer: 1,
    explanation: 'Early layers learn general features (edges, textures) that transfer well. Later layers are more task-specific.'
  },
  {
    id: 'tl8',
    question: 'What layers are usually replaced or retrained?',
    options: ['Early layers', 'Final classification layers specific to original task', 'Middle layers', 'All layers'],
    correctAnswer: 1,
    explanation: 'The final fully connected/classification layers are task-specific and usually replaced with new layers for the target task.'
  },
  {
    id: 'tl9',
    question: 'What learning rate should you use for fine-tuning?',
    options: ['Very high', 'Lower than training from scratch to avoid destroying learned features', 'Same as from scratch', 'No learning rate'],
    correctAnswer: 1,
    explanation: 'Use a smaller learning rate (e.g., 10-100× smaller) to gently adjust pre-trained weights without destroying useful features.'
  },
  {
    id: 'tl10',
    question: 'Can you use different learning rates for different layers?',
    options: ['No', 'Yes, often use lower rates for early layers and higher for new layers', 'Only one rate allowed', 'Only in RNNs'],
    correctAnswer: 1,
    explanation: 'Discriminative learning rates: use small rates for pre-trained layers (fine-tune gently) and larger rates for new layers.'
  },
  {
    id: 'tl11',
    question: 'What is the most common source dataset for ImageNet pre-training?',
    options: ['MNIST', 'ImageNet (1.2M images, 1000 classes)', 'CIFAR-10', 'Single image'],
    correctAnswer: 1,
    explanation: 'ImageNet is the most popular pre-training dataset, containing diverse images that help models learn general visual features.'
  },
  {
    id: 'tl12',
    question: 'What if your target task is very different from the pre-training task?',
    options: ['Transfer learning useless', 'Transfer learning still helps but may need more fine-tuning or less freezing', 'Always perfect', 'Never use transfer learning'],
    correctAnswer: 1,
    explanation: 'Even for different tasks, low-level features transfer. Fine-tune more layers or use lower learning rates throughout.'
  },
  {
    id: 'tl13',
    question: 'Can transfer learning work across domains (e.g., images to text)?',
    options: ['Perfectly', 'Limited; works best within similar domains', 'Never', 'Always better'],
    correctAnswer: 1,
    explanation: 'Transfer learning is most effective within similar domains. Cross-domain transfer (images↔text) requires specialized approaches.'
  },
  {
    id: 'tl14',
    question: 'What is zero-shot learning?',
    options: ['Training from scratch', 'Model predicts classes never seen during training', 'Transfer learning', 'No learning'],
    correctAnswer: 1,
    explanation: 'Zero-shot learning uses auxiliary information (e.g., class descriptions) to classify unseen classes without training examples.'
  },
  {
    id: 'tl15',
    question: 'What is few-shot learning?',
    options: ['Large dataset training', 'Learning from very few examples per class (e.g., 1-5 examples)', 'No examples', 'Transfer learning'],
    correctAnswer: 1,
    explanation: 'Few-shot learning trains models to generalize from only a handful of examples per class, often using meta-learning.'
  },
  {
    id: 'tl16',
    question: 'What is domain adaptation?',
    options: ['Training from scratch', 'Adapting model trained on source domain to perform well on different target domain', 'Data cleaning', 'Model compression'],
    correctAnswer: 1,
    explanation: 'Domain adaptation addresses distribution shift between training (source) and deployment (target) domains.'
  },
  {
    id: 'tl17',
    question: 'What is a common strategy for transfer learning with limited data?',
    options: ['Train all layers', 'Freeze most layers, only train final classifier', 'Increase learning rate', 'Remove layers'],
    correctAnswer: 1,
    explanation: 'With limited data, freeze early/middle layers to use as feature extractor and only train the final classification layer(s).'
  },
  {
    id: 'tl18',
    question: 'What is a common strategy for transfer learning with moderate data?',
    options: ['Freeze all', 'Fine-tune later layers while keeping early layers frozen or with low learning rate', 'Train from scratch', 'Random'],
    correctAnswer: 1,
    explanation: 'With moderate data, fine-tune the later (more task-specific) layers while keeping early (general) layers mostly frozen.'
  },
  {
    id: 'tl19',
    question: 'What is a common strategy for transfer learning with large data?',
    options: ['Only feature extraction', 'Fine-tune all or most layers with small learning rate', 'Freeze all', 'Train from scratch'],
    correctAnswer: 1,
    explanation: 'With sufficient data, fine-tune the entire network (or most of it) with a reduced learning rate for best performance.'
  },
  {
    id: 'tl20',
    question: 'Can transfer learning be used for medical imaging?',
    options: ['No, only natural images', 'Yes, even ImageNet features help despite domain difference', 'Never works', 'Only for X-rays'],
    correctAnswer: 1,
    explanation: 'ImageNet pre-training helps even for medical images (X-rays, CT, MRI), though domain-specific pre-training can be better.'
  },
  {
    id: 'tl21',
    question: 'What is multi-task learning?',
    options: ['Single task only', 'Training one model on multiple related tasks simultaneously', 'Sequential learning', 'Transfer learning'],
    correctAnswer: 1,
    explanation: 'Multi-task learning trains a shared model on multiple tasks together, helping the model learn more robust and general features.'
  },
  {
    id: 'tl22',
    question: 'What are popular pre-trained models for computer vision?',
    options: ['Random models', 'ResNet, VGG, Inception, EfficientNet, MobileNet', 'BERT, GPT', 'Only LeNet'],
    correctAnswer: 1,
    explanation: 'Many CNN architectures pre-trained on ImageNet are available: ResNet, VGG, Inception, EfficientNet, MobileNet, etc.'
  },
  {
    id: 'tl23',
    question: 'What is catastrophic forgetting?',
    options: ['Better memory', 'Model forgets previous knowledge when learning new tasks', 'Improved learning', 'No forgetting'],
    correctAnswer: 1,
    explanation: 'Catastrophic forgetting occurs when a model loses previously learned knowledge while learning new tasks, mitigated by careful fine-tuning.'
  },
  {
    id: 'tl24',
    question: 'How can you mitigate catastrophic forgetting?',
    options: ['Train faster', 'Use lower learning rates, freeze layers, or use regularization techniques', 'Use higher learning rate', 'Remove layers'],
    correctAnswer: 1,
    explanation: 'Techniques include: lower learning rates, freezing layers, elastic weight consolidation, or continual learning methods.'
  },
  {
    id: 'tl25',
    question: 'What is self-supervised pre-training?',
    options: ['Supervised learning', 'Pre-training without labels using pretext tasks on unlabeled data', 'Transfer learning', 'Zero-shot learning'],
    correctAnswer: 1,
    explanation: 'Self-supervised learning creates supervision from the data itself (e.g., predicting rotation, colorization) for pre-training without labels.'
  }
];

// Object Detection - 25 questions
export const objectDetectionQuestions: QuizQuestion[] = [
  {
    id: 'od1',
    question: 'What is object detection?',
    options: ['Image classification', 'Locating and classifying multiple objects in an image with bounding boxes', 'Segmentation', 'Only classification'],
    correctAnswer: 1,
    explanation: 'Object detection identifies what objects are in an image (classification) and where they are (localization with bounding boxes).'
  },
  {
    id: 'od2',
    question: 'How does object detection differ from classification?',
    options: ['No difference', 'Detection localizes multiple objects; classification only labels the whole image', 'Detection is easier', 'Detection is older'],
    correctAnswer: 1,
    explanation: 'Classification assigns a single label to an image; detection finds and labels multiple objects with their locations.'
  },
  {
    id: 'od3',
    question: 'What is a bounding box?',
    options: ['Image frame', 'Rectangle defined by coordinates (x, y, width, height) around an object', 'Object label', 'Pixel mask'],
    correctAnswer: 1,
    explanation: 'A bounding box is a rectangle specified by coordinates that encloses an object in the image.'
  },
  {
    id: 'od4',
    question: 'What is IoU (Intersection over Union)?',
    options: ['Loss function', 'Metric measuring overlap between predicted and ground truth boxes', 'Optimizer', 'Activation function'],
    correctAnswer: 1,
    explanation: 'IoU = (Area of Overlap) / (Area of Union) measures how well a predicted box matches the ground truth box.'
  },
  {
    id: 'od5',
    question: 'What IoU threshold is commonly used to determine a "correct" detection?',
    options: ['0.1', '0.5 or 0.75', '0.99', '1.0'],
    correctAnswer: 1,
    explanation: 'IoU ≥ 0.5 is the standard threshold for PASCAL VOC; COCO uses multiple thresholds (0.5 to 0.95).'
  },
  {
    id: 'od6',
    question: 'What is Non-Maximum Suppression (NMS)?',
    options: ['Loss function', 'Removes duplicate detections by keeping highest confidence box per object', 'Data augmentation', 'Optimizer'],
    correctAnswer: 1,
    explanation: 'NMS eliminates overlapping boxes for the same object, keeping only the box with highest confidence score.'
  },
  {
    id: 'od7',
    question: 'How does NMS work?',
    options: ['Random selection', 'Sort by confidence, keep highest, remove overlapping boxes with IoU > threshold', 'Keep all boxes', 'Average boxes'],
    correctAnswer: 1,
    explanation: 'NMS sorts boxes by confidence, iteratively keeps the highest and suppresses boxes with high IoU overlap with it.'
  },
  {
    id: 'od8',
    question: 'What are two-stage detectors?',
    options: ['One pass models', 'First propose regions, then classify them (e.g., R-CNN, Faster R-CNN)', 'Real-time detectors', 'No stages'],
    correctAnswer: 1,
    explanation: 'Two-stage detectors: stage 1 generates region proposals; stage 2 classifies and refines these proposals.'
  },
  {
    id: 'od9',
    question: 'What is R-CNN?',
    options: ['Recurrent CNN', 'Region-based CNN: uses selective search for proposals, then CNN for classification', 'Real-time CNN', 'Residual CNN'],
    correctAnswer: 1,
    explanation: 'R-CNN (2014) uses selective search to propose ~2000 regions, then classifies each with a CNN. Slow but accurate.'
  },
  {
    id: 'od10',
    question: 'What improvement did Fast R-CNN make?',
    options: ['Slower processing', 'Shares CNN computation across proposals using RoI pooling', 'More proposals', 'No CNN'],
    correctAnswer: 1,
    explanation: 'Fast R-CNN processes the entire image once with CNN, then pools features for each proposal (RoI pooling), much faster than R-CNN.'
  },
  {
    id: 'od11',
    question: 'What improvement did Faster R-CNN make?',
    options: ['Removed CNN', 'Replaces selective search with learned Region Proposal Network (RPN)', 'Slower', 'Manual proposals'],
    correctAnswer: 1,
    explanation: 'Faster R-CNN introduces RPN that learns to generate proposals, making the entire pipeline end-to-end trainable and faster.'
  },
  {
    id: 'od12',
    question: 'What are one-stage detectors?',
    options: ['Two-stage models', 'Directly predict classes and boxes in single pass (e.g., YOLO, SSD)', 'Slower models', 'No predictions'],
    correctAnswer: 1,
    explanation: 'One-stage detectors predict all objects in a single forward pass without separate proposal generation, enabling real-time speed.'
  },
  {
    id: 'od13',
    question: 'What does YOLO stand for?',
    options: ['You Only Look Once', 'You Only Look Once', 'Yellow Object Location', 'Your Optimal Learning Object'],
    correctAnswer: 1,
    explanation: 'YOLO (You Only Look Once) is a one-stage detector that divides the image into a grid and predicts boxes directly.'
  },
  {
    id: 'od14',
    question: 'What is the main advantage of YOLO?',
    options: ['Highest accuracy', 'Very fast, enabling real-time object detection', 'Most parameters', 'Slowest'],
    correctAnswer: 1,
    explanation: 'YOLO\'s single-pass architecture enables real-time detection (30+ FPS), making it suitable for video and embedded systems.'
  },
  {
    id: 'od15',
    question: 'What is SSD (Single Shot Detector)?',
    options: ['Two-stage detector', 'One-stage detector using multiple feature maps at different scales', 'Image classifier', 'Segmentation model'],
    correctAnswer: 1,
    explanation: 'SSD detects objects at multiple scales by making predictions from multiple feature layers in the network.'
  },
  {
    id: 'od16',
    question: 'What are anchor boxes?',
    options: ['Fixed boxes', 'Pre-defined boxes of various sizes/ratios used as references for predictions', 'Ground truth boxes', 'Random boxes'],
    correctAnswer: 1,
    explanation: 'Anchor boxes are predefined templates at different scales and aspect ratios; models predict offsets from these anchors.'
  },
  {
    id: 'od17',
    question: 'Why use anchor boxes?',
    options: ['No reason', 'Handle objects of different sizes and aspect ratios', 'Slow down model', 'Complicate training'],
    correctAnswer: 1,
    explanation: 'Anchor boxes provide multiple reference shapes, allowing the model to detect objects of varying sizes and proportions.'
  },
  {
    id: 'od18',
    question: 'What is RetinaNet known for?',
    options: ['Slowest detector', 'Introducing Focal Loss to address class imbalance in one-stage detectors', 'Oldest detector', 'No innovation'],
    correctAnswer: 1,
    explanation: 'RetinaNet uses Focal Loss to down-weight easy negatives, addressing the extreme class imbalance in dense detection.'
  },
  {
    id: 'od19',
    question: 'What is Feature Pyramid Network (FPN)?',
    options: ['Single scale', 'Multi-scale feature hierarchy for detecting objects at different sizes', 'Data augmentation', 'Loss function'],
    correctAnswer: 1,
    explanation: 'FPN builds a feature pyramid with both bottom-up and top-down pathways, improving detection across scales.'
  },
  {
    id: 'od20',
    question: 'What is the COCO dataset?',
    options: ['10 classes', '80-class object detection dataset with ~330k images', 'MNIST variant', 'Text dataset'],
    correctAnswer: 1,
    explanation: 'Microsoft COCO (Common Objects in Context) is a large-scale dataset for object detection, segmentation, and captioning.'
  },
  {
    id: 'od21',
    question: 'What metric does COCO use for evaluation?',
    options: ['Accuracy', 'mAP (mean Average Precision) averaged over multiple IoU thresholds', 'MSE', 'F1'],
    correctAnswer: 1,
    explanation: 'COCO mAP averages AP across IoU thresholds from 0.5 to 0.95, providing a more comprehensive evaluation.'
  },
  {
    id: 'od22',
    question: 'What is the speed-accuracy tradeoff in object detection?',
    options: ['No tradeoff', 'Two-stage detectors are more accurate but slower; one-stage are faster but less accurate', 'Both equal', 'One-stage always better'],
    correctAnswer: 1,
    explanation: 'Historically, two-stage (Faster R-CNN) were more accurate but slow; one-stage (YOLO) faster but less accurate. Gap has narrowed.'
  },
  {
    id: 'od23',
    question: 'What is EfficientDet?',
    options: ['Old detector', 'Efficient detector using compound scaling and weighted feature fusion', 'Slow detector', 'Text detector'],
    correctAnswer: 1,
    explanation: 'EfficientDet applies compound scaling to object detection, achieving state-of-the-art accuracy with fewer parameters.'
  },
  {
    id: 'od24',
    question: 'What is DETR?',
    options: ['CNN-only', 'Detection Transformer using attention instead of anchors/NMS', 'Old method', 'RNN-based'],
    correctAnswer: 1,
    explanation: 'DETR treats object detection as a set prediction problem using transformers, eliminating hand-crafted components like anchors and NMS.'
  },
  {
    id: 'od25',
    question: 'What are common challenges in object detection?',
    options: ['Too easy', 'Scale variation, occlusion, class imbalance, small objects', 'No challenges', 'Only classification'],
    correctAnswer: 1,
    explanation: 'Challenges include: objects at different scales, partial occlusion, extreme class imbalance (few objects, many background), and small objects.'
  }
];

// Image Segmentation - 20 questions
export const imageSegmentationQuestions: QuizQuestion[] = [
  {
    id: 'seg1',
    question: 'What is image segmentation?',
    options: ['Object detection', 'Partitioning image into regions/pixels belonging to different objects or classes', 'Classification', 'Edge detection'],
    correctAnswer: 1,
    explanation: 'Segmentation assigns a label to every pixel in the image, creating a detailed map of objects and regions.'
  },
  {
    id: 'seg2',
    question: 'What is semantic segmentation?',
    options: ['Instance-aware', 'Classifying each pixel into categories, not distinguishing instances', 'Object detection', 'Counting objects'],
    correctAnswer: 1,
    explanation: 'Semantic segmentation labels pixels by class (e.g., all "person" pixels) without distinguishing between individual instances.'
  },
  {
    id: 'seg3',
    question: 'What is instance segmentation?',
    options: ['Semantic only', 'Segmenting and distinguishing between individual object instances', 'No instances', 'Classification'],
    correctAnswer: 1,
    explanation: 'Instance segmentation identifies each individual object separately (e.g., person1, person2) with pixel-level masks.'
  },
  {
    id: 'seg4',
    question: 'What is panoptic segmentation?',
    options: ['Semantic only', 'Combination of semantic segmentation (stuff) and instance segmentation (things)', 'Instance only', 'No segmentation'],
    correctAnswer: 1,
    explanation: 'Panoptic segmentation unifies semantic seg (background/stuff like sky, road) and instance seg (countable things like cars, people).'
  },
  {
    id: 'seg5',
    question: 'What is U-Net?',
    options: ['Object detector', 'Encoder-decoder architecture with skip connections for segmentation', 'Classifier', 'GAN'],
    correctAnswer: 1,
    explanation: 'U-Net has a contracting path (encoder) and expansive path (decoder) with skip connections, designed for biomedical image segmentation.'
  },
  {
    id: 'seg6',
    question: 'What is the purpose of skip connections in U-Net?',
    options: ['Slow training', 'Combine high-res features from encoder with decoder for precise localization', 'Add parameters', 'No purpose'],
    correctAnswer: 1,
    explanation: 'Skip connections pass fine-grained spatial information from encoder to decoder, improving localization accuracy.'
  },
  {
    id: 'seg7',
    question: 'What is the encoder part of U-Net?',
    options: ['Upsampling', 'Downsampling path that extracts features (like typical CNN)', 'Skip connections', 'Output layer'],
    correctAnswer: 1,
    explanation: 'The encoder is a contracting path with convolutions and pooling that captures context and features.'
  },
  {
    id: 'seg8',
    question: 'What is the decoder part of U-Net?',
    options: ['Downsampling', 'Upsampling path that reconstructs spatial resolution', 'Encoder', 'Input layer'],
    correctAnswer: 1,
    explanation: 'The decoder uses upsampling/transposed convolutions to gradually recover spatial resolution for pixel-wise predictions.'
  },
  {
    id: 'seg9',
    question: 'What is FCN (Fully Convolutional Network)?',
    options: ['Has fully connected layers', 'Network with only convolutional layers, no fully connected layers', 'RNN variant', 'Transformer'],
    correctAnswer: 1,
    explanation: 'FCN replaces fully connected layers with convolutional layers, allowing arbitrary input sizes and dense predictions.'
  },
  {
    id: 'seg10',
    question: 'What is transposed convolution (deconvolution)?',
    options: ['Standard convolution', 'Upsampling operation to increase spatial resolution', 'Pooling', 'Activation'],
    correctAnswer: 1,
    explanation: 'Transposed convolution performs learnable upsampling, increasing spatial dimensions in decoder networks.'
  },
  {
    id: 'seg11',
    question: 'What is Mask R-CNN?',
    options: ['Semantic segmentation only', 'Extends Faster R-CNN with mask prediction branch for instance segmentation', 'Classification model', 'Text model'],
    correctAnswer: 1,
    explanation: 'Mask R-CNN adds a mask prediction branch to Faster R-CNN, producing pixel-level masks for each detected instance.'
  },
  {
    id: 'seg12',
    question: 'What is the difference between Faster R-CNN and Mask R-CNN?',
    options: ['No difference', 'Mask R-CNN adds a segmentation mask branch', 'Mask R-CNN is slower', 'Faster R-CNN segments'],
    correctAnswer: 1,
    explanation: 'Mask R-CNN extends Faster R-CNN by adding a parallel branch that predicts segmentation masks for each bounding box.'
  },
  {
    id: 'seg13',
    question: 'What loss function is used for semantic segmentation?',
    options: ['MSE', 'Cross-entropy per pixel, or Dice loss', 'Hinge loss', 'Triplet loss'],
    correctAnswer: 1,
    explanation: 'Pixel-wise cross-entropy is common; Dice loss (based on IoU) is also used, especially for imbalanced classes.'
  },
  {
    id: 'seg14',
    question: 'What is the Dice coefficient?',
    options: ['Accuracy', 'Similarity metric: 2×|A∩B| / (|A|+|B|)', 'Loss function only', 'Optimizer'],
    correctAnswer: 1,
    explanation: 'Dice coefficient measures overlap between predicted and ground truth segmentation, similar to IoU but more sensitive to small objects.'
  },
  {
    id: 'seg15',
    question: 'What is DeepLab?',
    options: ['Simple CNN', 'Segmentation architecture using atrous (dilated) convolution and ASPP', 'Object detector', 'Classifier'],
    correctAnswer: 1,
    explanation: 'DeepLab uses atrous/dilated convolutions to increase receptive field and Atrous Spatial Pyramid Pooling (ASPP) for multi-scale features.'
  },
  {
    id: 'seg16',
    question: 'What is atrous/dilated convolution?',
    options: ['Standard convolution', 'Convolution with gaps (dilation) to increase receptive field without pooling', 'Pooling', 'Activation'],
    correctAnswer: 1,
    explanation: 'Dilated convolution inserts spaces between kernel elements, expanding receptive field while maintaining resolution.'
  },
  {
    id: 'seg17',
    question: 'Why use dilated convolutions in segmentation?',
    options: ['No reason', 'Increase receptive field without reducing resolution', 'Reduce parameters', 'Speed up training'],
    correctAnswer: 1,
    explanation: 'Dilated convolutions capture larger context without downsampling, preserving spatial resolution crucial for segmentation.'
  },
  {
    id: 'seg18',
    question: 'What is a common challenge in segmentation?',
    options: ['Too easy', 'Class imbalance (background vs objects), boundary precision', 'Too fast', 'No challenges'],
    correctAnswer: 1,
    explanation: 'Challenges include severe class imbalance (most pixels are background), accurate boundary delineation, and small object detection.'
  },
  {
    id: 'seg19',
    question: 'What is the difference between segmentation and object detection?',
    options: ['No difference', 'Segmentation provides pixel-level masks; detection provides bounding boxes', 'Segmentation is easier', 'Detection is more accurate'],
    correctAnswer: 1,
    explanation: 'Detection localizes objects with rectangular boxes; segmentation provides precise pixel-level object boundaries.'
  },
  {
    id: 'seg20',
    question: 'What are common segmentation datasets?',
    options: ['MNIST only', 'PASCAL VOC, COCO, Cityscapes, ADE20K', 'ImageNet only', 'No datasets'],
    correctAnswer: 1,
    explanation: 'Popular datasets: PASCAL VOC (21 classes), COCO (segmentation), Cityscapes (urban scenes), ADE20K (scene parsing).'
  }
];
