import { QuizQuestion } from '../../types';

// CNNs - 25 questions
export const cnnQuestions: QuizQuestion[] = [
  {
    id: 'cnn1',
    question: 'What does CNN stand for?',
    options: ['Central Neural Network', 'Convolutional Neural Network', 'Computed Neural Network', 'Complex Neural Network'],
    correctAnswer: 1,
    explanation: 'CNN stands for Convolutional Neural Network, designed to process grid-like data such as images.'
  },
  {
    id: 'cnn2',
    question: 'What makes CNNs particularly suitable for image processing?',
    options: ['More layers', 'They preserve spatial structure and detect local patterns through convolution', 'Faster training', 'Less parameters always'],
    correctAnswer: 1,
    explanation: 'CNNs use convolution to detect local spatial patterns (edges, textures) and preserve spatial hierarchies in images.'
  },
  {
    id: 'cnn3',
    question: 'What is a convolutional layer?',
    options: ['Fully connected layer', 'Layer that applies filters to detect features', 'Pooling layer', 'Output layer'],
    correctAnswer: 1,
    explanation: 'A convolutional layer applies learnable filters (kernels) that slide over the input to detect features like edges or textures.'
  },
  {
    id: 'cnn4',
    question: 'What is a filter/kernel in CNNs?',
    options: ['Activation function', 'Small matrix of weights that slides over the image to detect patterns', 'Loss function', 'Optimizer'],
    correctAnswer: 1,
    explanation: 'A filter (kernel) is a small matrix (e.g., 3×3, 5×5) of learnable weights used to detect specific features through convolution.'
  },
  {
    id: 'cnn5',
    question: 'What operation does a convolutional layer perform?',
    options: ['Matrix multiplication', 'Element-wise multiplication and sum (convolution)', 'Division', 'Sorting'],
    correctAnswer: 1,
    explanation: 'Convolution performs element-wise multiplication between the filter and patches of input, then sums the results.'
  },
  {
    id: 'cnn6',
    question: 'What is stride in convolution?',
    options: ['Filter size', 'Number of pixels the filter moves at each step', 'Number of filters', 'Learning rate'],
    correctAnswer: 1,
    explanation: 'Stride is the step size for moving the filter. Stride of 1 moves one pixel at a time; stride of 2 skips pixels.'
  },
  {
    id: 'cnn7',
    question: 'What is padding in CNNs?',
    options: ['Adding layers', 'Adding pixels around the border of the input', 'Removing pixels', 'Dropout'],
    correctAnswer: 1,
    explanation: 'Padding adds border pixels (usually zeros) to control output size and preserve edge information.'
  },
  {
    id: 'cnn8',
    question: 'What is "valid" padding?',
    options: ['Padding that maintains size', 'No padding', 'Maximum padding', 'Random padding'],
    correctAnswer: 1,
    explanation: 'Valid (no) padding means the filter only moves over valid positions, reducing the output size.'
  },
  {
    id: 'cnn9',
    question: 'What is "same" padding?',
    options: ['No padding', 'Padding to keep output size same as input', 'Padding that reduces size', 'Maximum padding'],
    correctAnswer: 1,
    explanation: 'Same padding adds enough border pixels so the output spatial dimensions match the input (with stride=1).'
  },
  {
    id: 'cnn10',
    question: 'What is parameter sharing in CNNs?',
    options: ['Sharing between networks', 'Same filter weights used across all spatial locations', 'Sharing data', 'Transfer learning'],
    correctAnswer: 1,
    explanation: 'Parameter sharing means the same filter is applied across the entire image, drastically reducing parameters compared to fully connected layers.'
  },
  {
    id: 'cnn11',
    question: 'What is translation invariance in CNNs?',
    options: ['Language translation', 'Detecting features regardless of their position in the image', 'Moving images', 'Rotating images'],
    correctAnswer: 1,
    explanation: 'Translation invariance means CNNs can detect features like edges anywhere in the image, achieved through weight sharing.'
  },
  {
    id: 'cnn12',
    question: 'What are feature maps?',
    options: ['Input images', 'Outputs of convolutional layers showing detected features', 'Weight matrices', 'Loss values'],
    correctAnswer: 1,
    explanation: 'Feature maps are the outputs of applying filters, showing where certain features (edges, textures) are detected in the image.'
  },
  {
    id: 'cnn13',
    question: 'How many feature maps does one filter produce?',
    options: ['One per pixel', 'One feature map per filter', 'Multiple maps', 'None'],
    correctAnswer: 1,
    explanation: 'Each filter produces one feature map showing where that filter\'s pattern appears in the input.'
  },
  {
    id: 'cnn14',
    question: 'What do early convolutional layers typically detect?',
    options: ['Objects', 'Low-level features like edges and corners', 'Text', 'Colors only'],
    correctAnswer: 1,
    explanation: 'Early layers detect simple patterns like edges, corners, and colors. Deeper layers combine these into complex patterns and objects.'
  },
  {
    id: 'cnn15',
    question: 'What do deeper convolutional layers typically detect?',
    options: ['Edges', 'Complex patterns and high-level features like object parts', 'Nothing', 'Pixels'],
    correctAnswer: 1,
    explanation: 'Deeper layers learn hierarchical representations, combining low-level features into higher-level concepts like faces or wheels.'
  },
  {
    id: 'cnn16',
    question: 'What is the receptive field?',
    options: ['Filter size', 'Region of input that influences a particular neuron\'s output', 'Output size', 'Number of layers'],
    correctAnswer: 1,
    explanation: 'The receptive field is the input region that affects a neuron. It grows larger in deeper layers, allowing detection of larger patterns.'
  },
  {
    id: 'cnn17',
    question: 'What is a 1×1 convolution?',
    options: ['Invalid operation', 'Convolutional filter with 1×1 size, used for channel-wise transformations', 'Identity operation', 'Pooling'],
    correctAnswer: 1,
    explanation: '1×1 convolutions adjust the number of channels (depth) without spatial convolution, useful for dimensionality reduction.'
  },
  {
    id: 'cnn18',
    question: 'How do CNNs reduce parameters compared to fully connected networks for images?',
    options: ['They don\'t', 'Through parameter sharing and local connectivity', 'Using dropout', 'Smaller images'],
    correctAnswer: 1,
    explanation: 'CNNs use weight sharing (same filter everywhere) and local connectivity (filters only see small patches), massively reducing parameters.'
  },
  {
    id: 'cnn19',
    question: 'What is the typical architecture of a CNN?',
    options: ['Only convolution layers', 'Stacked conv layers → pooling → fully connected layers → output', 'Random structure', 'Single layer'],
    correctAnswer: 1,
    explanation: 'CNNs typically stack convolutional and pooling layers for feature extraction, then use fully connected layers for classification.'
  },
  {
    id: 'cnn20',
    question: 'What activation function is commonly used in CNNs?',
    options: ['Sigmoid', 'ReLU', 'Linear', 'Step function'],
    correctAnswer: 1,
    explanation: 'ReLU is the most common activation in CNNs due to its simplicity and effectiveness in avoiding vanishing gradients.'
  },
  {
    id: 'cnn21',
    question: 'Can CNNs be used for non-image data?',
    options: ['No, only images', 'Yes, any grid-structured data (1D signals, 3D volumes)', 'Only 2D data', 'Only RGB images'],
    correctAnswer: 1,
    explanation: 'CNNs work on any grid-like data: 1D convolutions for audio/text, 2D for images, 3D for video or medical scans.'
  },
  {
    id: 'cnn22',
    question: 'What is a depth-wise separable convolution?',
    options: ['Standard convolution', 'Factorizes convolution into spatial and channel-wise operations for efficiency', 'Very deep convolution', '3D convolution'],
    correctAnswer: 1,
    explanation: 'Depth-wise separable convolution splits standard convolution into depth-wise (spatial) and point-wise (channel) operations, reducing computation.'
  },
  {
    id: 'cnn23',
    question: 'What is the purpose of multiple filters in a convolutional layer?',
    options: ['Redundancy', 'To detect different features (edges, textures, patterns)', 'To slow down training', 'No purpose'],
    correctAnswer: 1,
    explanation: 'Multiple filters allow the network to learn diverse features simultaneously, creating a rich representation.'
  },
  {
    id: 'cnn24',
    question: 'How is the output size of a convolution calculated?',
    options: ['Same as input', 'Output = (Input - Filter + 2×Padding) / Stride + 1', 'Random', 'Filter size'],
    correctAnswer: 1,
    explanation: 'The formula accounts for filter size, stride, and padding: out_size = (in_size - kernel + 2×pad) / stride + 1.'
  },
  {
    id: 'cnn25',
    question: 'What are dilated/atrous convolutions?',
    options: ['Standard convolutions', 'Convolutions with gaps between kernel elements, increasing receptive field', 'Compressed convolutions', 'Random convolutions'],
    correctAnswer: 1,
    explanation: 'Dilated convolutions insert spaces in the filter, expanding the receptive field without increasing parameters.'
  }
];

// Pooling Layers - 20 questions
export const poolingQuestions: QuizQuestion[] = [
  {
    id: 'pool1',
    question: 'What is pooling in CNNs?',
    options: ['Adding layers', 'Downsampling operation to reduce spatial dimensions', 'Data augmentation', 'Weight initialization'],
    correctAnswer: 1,
    explanation: 'Pooling reduces the spatial size (height and width) of feature maps, reducing parameters and computation.'
  },
  {
    id: 'pool2',
    question: 'What is Max Pooling?',
    options: ['Taking average', 'Taking maximum value from each pooling window', 'Taking minimum', 'Summing values'],
    correctAnswer: 1,
    explanation: 'Max pooling selects the maximum value from each pooling region, preserving the strongest activations.'
  },
  {
    id: 'pool3',
    question: 'What is Average Pooling?',
    options: ['Taking maximum', 'Taking average value from each pooling window', 'Taking median', 'Random selection'],
    correctAnswer: 1,
    explanation: 'Average pooling computes the mean of values in each pooling window, providing smoother downsampling.'
  },
  {
    id: 'pool4',
    question: 'Which is more common: Max Pooling or Average Pooling?',
    options: ['Average pooling', 'Max pooling', 'Equally common', 'Neither is used'],
    correctAnswer: 1,
    explanation: 'Max pooling is more common in CNNs as it preserves the most prominent features and provides some translation invariance.'
  },
  {
    id: 'pool5',
    question: 'What is a typical pooling window size?',
    options: ['1×1', '2×2 or 3×3', '10×10', '100×100'],
    correctAnswer: 1,
    explanation: '2×2 with stride 2 is most common, halving spatial dimensions. 3×3 is also used but less frequently.'
  },
  {
    id: 'pool6',
    question: 'Does pooling have learnable parameters?',
    options: ['Yes, many', 'No, it\'s a fixed operation', 'Only in max pooling', 'Only with stride'],
    correctAnswer: 1,
    explanation: 'Pooling operations have no learnable parameters; they apply a fixed function (max, average) to each window.'
  },
  {
    id: 'pool7',
    question: 'What is the main purpose of pooling?',
    options: ['Increase size', 'Reduce spatial dimensions, provide translation invariance, and reduce overfitting', 'Add parameters', 'Slow training'],
    correctAnswer: 1,
    explanation: 'Pooling reduces spatial size (fewer parameters), provides some translation invariance, and acts as a regularizer.'
  },
  {
    id: 'pool8',
    question: 'Does pooling affect the number of channels (depth)?',
    options: ['Yes, reduces channels', 'No, operates independently on each channel', 'Yes, increases channels', 'Removes all channels'],
    correctAnswer: 1,
    explanation: 'Pooling is applied separately to each channel, reducing height and width but preserving the number of channels.'
  },
  {
    id: 'pool9',
    question: 'What is Global Average Pooling?',
    options: ['Small window pooling', 'Averages entire feature map into single value per channel', 'No pooling', 'Random pooling'],
    correctAnswer: 1,
    explanation: 'Global Average Pooling (GAP) averages each feature map to a single number, often used before the final classification layer.'
  },
  {
    id: 'pool10',
    question: 'What is Global Max Pooling?',
    options: ['Standard max pooling', 'Takes maximum value from entire feature map per channel', 'No pooling', 'Local pooling'],
    correctAnswer: 1,
    explanation: 'Global Max Pooling takes the maximum activation from each entire feature map, producing one value per channel.'
  },
  {
    id: 'pool11',
    question: 'Why use Global Average Pooling over fully connected layers?',
    options: ['No difference', 'Fewer parameters, more robust to spatial translations', 'Faster only', 'Better accuracy always'],
    correctAnswer: 1,
    explanation: 'GAP eliminates the need for large fully connected layers, dramatically reducing parameters and overfitting risk.'
  },
  {
    id: 'pool12',
    question: 'Can you backpropagate through pooling layers?',
    options: ['No', 'Yes, gradients flow back through pooling', 'Only max pooling', 'Only average pooling'],
    correctAnswer: 1,
    explanation: 'Both max and average pooling are differentiable. Max pooling routes gradients to the maximum value location.'
  },
  {
    id: 'pool13',
    question: 'What is the stride typically used with pooling?',
    options: ['1', 'Usually same as pool size (e.g., 2 for 2×2)', 'Always 3', 'Random'],
    correctAnswer: 1,
    explanation: 'Stride typically equals pool size (e.g., 2×2 pool with stride 2) to create non-overlapping windows and halve dimensions.'
  },
  {
    id: 'pool14',
    question: 'What is overlapping pooling?',
    options: ['Standard pooling', 'Pooling with stride < pool size, creating overlap', 'No pooling', 'Global pooling'],
    correctAnswer: 1,
    explanation: 'Overlapping pooling uses stride smaller than pool size (e.g., 3×3 pool with stride 2), providing smoother downsampling.'
  },
  {
    id: 'pool15',
    question: 'Is pooling necessary in modern CNNs?',
    options: ['Yes, always required', 'Not always; some use strided convolutions instead', 'Never used', 'Only for images'],
    correctAnswer: 1,
    explanation: 'Modern architectures sometimes replace pooling with strided convolutions, which learn how to downsample.'
  },
  {
    id: 'pool16',
    question: 'What is stochastic pooling?',
    options: ['Random pooling size', 'Samples from pooling window based on activation probabilities', 'Average pooling', 'No pooling'],
    correctAnswer: 1,
    explanation: 'Stochastic pooling randomly samples values from the pooling window weighted by their magnitude, adding regularization.'
  },
  {
    id: 'pool17',
    question: 'How does max pooling provide translation invariance?',
    options: ['It doesn\'t', 'Small shifts in input don\'t change output if max value remains in window', 'Through averaging', 'Through padding'],
    correctAnswer: 1,
    explanation: 'Max pooling makes the network robust to small translations because the maximum activation in a region is preserved.'
  },
  {
    id: 'pool18',
    question: 'What happens to spatial information during pooling?',
    options: ['Perfectly preserved', 'Some spatial information is lost through downsampling', 'Increased', 'No change'],
    correctAnswer: 1,
    explanation: 'Pooling discards spatial resolution. This is a trade-off: reduce computation but lose fine-grained location information.'
  },
  {
    id: 'pool19',
    question: 'Where is pooling typically placed in CNN architecture?',
    options: ['Input layer', 'After convolutional layers', 'Before convolutional layers', 'Output layer'],
    correctAnswer: 1,
    explanation: 'Pooling is typically placed after convolutional layers to progressively reduce spatial dimensions through the network.'
  },
  {
    id: 'pool20',
    question: 'Can pooling help reduce overfitting?',
    options: ['No effect', 'Yes, by reducing spatial dimensions and parameters', 'Increases overfitting', 'Only with dropout'],
    correctAnswer: 1,
    explanation: 'Pooling reduces the dimensionality of feature maps, which reduces the risk of overfitting by constraining the model.'
  }
];

// Classic CNN Architectures - 25 questions
export const classicArchitecturesQuestions: QuizQuestion[] = [
  {
    id: 'arch1',
    question: 'What was LeNet-5?',
    options: ['Modern architecture', 'One of the first CNNs (1998) for handwritten digit recognition', 'Transformer model', 'RNN variant'],
    correctAnswer: 1,
    explanation: 'LeNet-5 by Yann LeCun (1998) was an early CNN used for digit recognition in the MNIST dataset.'
  },
  {
    id: 'arch2',
    question: 'What was significant about AlexNet (2012)?',
    options: ['First neural network', 'Won ImageNet 2012 with deep CNN, popularizing deep learning', 'Smallest network', 'First RNN'],
    correctAnswer: 1,
    explanation: 'AlexNet won ImageNet 2012 by a large margin, demonstrating the power of deep CNNs and GPUs, sparking the deep learning revolution.'
  },
  {
    id: 'arch3',
    question: 'What innovations did AlexNet introduce?',
    options: ['Attention mechanism', 'ReLU activation, dropout, GPU training, data augmentation', 'Transformers', 'GANs'],
    correctAnswer: 1,
    explanation: 'AlexNet popularized ReLU, used dropout for regularization, leveraged GPU acceleration, and employed extensive data augmentation.'
  },
  {
    id: 'arch4',
    question: 'What is VGGNet known for?',
    options: ['Complexity', 'Using very small (3×3) filters consistently throughout deep network', 'Few layers', 'Attention'],
    correctAnswer: 1,
    explanation: 'VGGNet (2014) showed that depth matters by stacking many layers with small 3×3 filters, achieving strong performance.'
  },
  {
    id: 'arch5',
    question: 'What is a drawback of VGGNet?',
    options: ['Too shallow', 'Very large number of parameters (memory intensive)', 'Too fast', 'No convolutions'],
    correctAnswer: 1,
    explanation: 'VGGNet has ~138M parameters, making it memory-intensive and slow to train compared to more efficient modern architectures.'
  },
  {
    id: 'arch6',
    question: 'What innovation did GoogLeNet (Inception) introduce?',
    options: ['Deeper only', 'Inception modules with multiple filter sizes in parallel', 'Attention', 'Recurrence'],
    correctAnswer: 1,
    explanation: 'GoogLeNet introduced Inception modules that apply multiple filter sizes (1×1, 3×3, 5×5) in parallel, capturing multi-scale features.'
  },
  {
    id: 'arch7',
    question: 'What is the purpose of 1×1 convolutions in Inception modules?',
    options: ['No purpose', 'Dimensionality reduction before expensive larger convolutions', 'Increase size', 'Pooling'],
    correctAnswer: 1,
    explanation: '1×1 convolutions reduce the number of channels before applying larger filters, reducing computational cost.'
  },
  {
    id: 'arch8',
    question: 'What problem does ResNet address?',
    options: ['Too fast training', 'Degradation problem: very deep networks are hard to train', 'Too few parameters', 'Data scarcity'],
    correctAnswer: 1,
    explanation: 'ResNet (2015) addressed the degradation problem where deeper networks performed worse due to optimization difficulties.'
  },
  {
    id: 'arch9',
    question: 'What is a residual connection (skip connection)?',
    options: ['Standard connection', 'Shortcut that adds input directly to output: F(x) + x', 'Pooling', 'Dropout'],
    correctAnswer: 1,
    explanation: 'Skip connections add the input to the layer output, allowing gradients to flow directly and enabling training of very deep networks.'
  },
  {
    id: 'arch10',
    question: 'How do residual connections help training?',
    options: ['Add parameters', 'Enable easier gradient flow and identity mapping learning', 'Slow training', 'Reduce accuracy'],
    correctAnswer: 1,
    explanation: 'Skip connections provide direct gradient paths, mitigating vanishing gradients and allowing networks with 100+ layers.'
  },
  {
    id: 'arch11',
    question: 'What is the residual block formula?',
    options: ['y = F(x)', 'y = F(x) + x', 'y = F(x) × x', 'y = F(x) - x'],
    correctAnswer: 1,
    explanation: 'Residual learning: y = F(x) + x, where F(x) is the learned residual mapping and x is the identity shortcut.'
  },
  {
    id: 'arch12',
    question: 'How deep was the original ResNet that won ImageNet 2015?',
    options: ['18 layers', '152 layers', '5 layers', '1000 layers'],
    correctAnswer: 1,
    explanation: 'ResNet-152 won ImageNet 2015. ResNet variants include ResNet-18, ResNet-34, ResNet-50, ResNet-101, and ResNet-152.'
  },
  {
    id: 'arch13',
    question: 'What is DenseNet?',
    options: ['Sparse network', 'Connects each layer to all subsequent layers in a block', 'Standard CNN', 'RNN variant'],
    correctAnswer: 1,
    explanation: 'DenseNet creates dense connections where each layer receives feature maps from all previous layers, encouraging feature reuse.'
  },
  {
    id: 'arch14',
    question: 'What is the advantage of DenseNet?',
    options: ['Fewer connections', 'Better gradient flow, feature reuse, fewer parameters', 'Slower training', 'Simpler architecture'],
    correctAnswer: 1,
    explanation: 'DenseNet\'s dense connectivity improves gradient flow, reduces parameters through feature reuse, and mitigates overfitting.'
  },
  {
    id: 'arch15',
    question: 'What is MobileNet designed for?',
    options: ['Desktop GPUs', 'Efficient inference on mobile and edge devices', 'Large servers', 'Research only'],
    correctAnswer: 1,
    explanation: 'MobileNet uses depth-wise separable convolutions for efficient mobile deployment with minimal accuracy loss.'
  },
  {
    id: 'arch16',
    question: 'What technique does MobileNet use for efficiency?',
    options: ['More layers', 'Depth-wise separable convolutions', 'Larger filters', 'More parameters'],
    correctAnswer: 1,
    explanation: 'Depth-wise separable convolutions factorize standard convolutions, drastically reducing parameters and computation.'
  },
  {
    id: 'arch17',
    question: 'What is EfficientNet?',
    options: ['Random architecture', 'Systematically scales depth, width, and resolution with compound coefficient', 'Old architecture', 'Text model'],
    correctAnswer: 1,
    explanation: 'EfficientNet uses neural architecture search and compound scaling to efficiently scale all dimensions (depth, width, resolution) together.'
  },
  {
    id: 'arch18',
    question: 'What is NASNet?',
    options: ['Hand-designed', 'Architecture found through Neural Architecture Search (automated)', 'Random design', 'Simple CNN'],
    correctAnswer: 1,
    explanation: 'NASNet uses automated Neural Architecture Search to discover optimal architectures, achieving state-of-the-art performance.'
  },
  {
    id: 'arch19',
    question: 'What is Squeeze-and-Excitation (SE) block?',
    options: ['Pooling layer', 'Attention mechanism for channels, recalibrating channel importance', 'Convolution type', 'Loss function'],
    correctAnswer: 1,
    explanation: 'SE blocks adaptively recalibrate channel-wise feature responses by modeling interdependencies between channels (channel attention).'
  },
  {
    id: 'arch20',
    question: 'Which architecture introduced batch normalization?',
    options: ['AlexNet', 'Inception/GoogLeNet variants (Inception v2)', 'LeNet', 'VGG'],
    correctAnswer: 1,
    explanation: 'Batch Normalization was introduced in 2015 and popularized in Inception v2/v3, becoming standard in modern architectures.'
  },
  {
    id: 'arch21',
    question: 'What is ResNeXt?',
    options: ['Simpler ResNet', 'ResNet variant with cardinality dimension (grouped convolutions)', 'Older than ResNet', 'No residual connections'],
    correctAnswer: 1,
    explanation: 'ResNeXt adds a cardinality dimension (multiple parallel paths with grouped convolutions) to ResNet for improved performance.'
  },
  {
    id: 'arch22',
    question: 'What is the general trend in CNN architecture evolution?',
    options: ['Simpler over time', 'Deeper, more efficient, with better optimization (skip connections, normalization)', 'Shallower', 'Random'],
    correctAnswer: 1,
    explanation: 'CNNs evolved from shallow (LeNet) to deeper (VGG, ResNet), incorporating skip connections, normalization, and efficiency improvements.'
  },
  {
    id: 'arch23',
    question: 'What is the purpose of auxiliary classifiers in GoogLeNet?',
    options: ['Primary output', 'Provide additional gradient signal during training to fight vanishing gradients', 'Data augmentation', 'Regularization only'],
    correctAnswer: 1,
    explanation: 'Auxiliary classifiers at intermediate layers inject gradients deeper in the network, helping with training (usually removed at inference).'
  },
  {
    id: 'arch24',
    question: 'What dataset was ImageNet competition based on?',
    options: ['MNIST', '1000-class image classification with ~1.2M training images', 'CIFAR-10', 'Fashion-MNIST'],
    correctAnswer: 1,
    explanation: 'ImageNet Large Scale Visual Recognition Challenge (ILSVRC) used 1000 object classes with ~1.2M training images.'
  },
  {
    id: 'arch25',
    question: 'Which architecture first achieved superhuman performance on ImageNet?',
    options: ['LeNet', 'ResNet (surpassed human-level 5% top-5 error)', 'AlexNet', 'VGG'],
    correctAnswer: 1,
    explanation: 'ResNet-152 achieved ~3.5% top-5 error on ImageNet in 2015, surpassing estimated human performance (~5%).'
  }
];
