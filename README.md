# Site Page

https://nononsense.me/ml-review

# ML## Features

- 📚 **Comprehensive Content**: 52 topics covering foundations to advanced ML concepts
- 💬 **342+ Interview Questions**: Detailed 3-5 paragraph answers for each question
- 🧪 **Practice Quizzes**: Random 10-question quizzes from large question pools (20+ questions per topic)
- 💻 **Code Examples**: Python implementations for key concepts
- 📈 **Smart Progress Tracking**: Automatic status updates based on quiz performance
- 🎯 **Mastery System**: Earn "mastered" status by demonstrating consistent performance
- 🎮 **Interactive Demo**: Bias-Variance Tradeoff visualization
- 🎲 **Quiz Variety**: Different random questions each time you take a quiz
- 🌙 **Dark/Light Theme**: Toggle between themes for comfortable learning
- 📱 **Responsive Design**: Works seamlessly on desktop and mobile devices
- 🔍 **Category Navigation**: Organized into 8 major ML categories
- 💾 **Auto-Save**: Progress automatically saved to browser localStorage

## Quiz System

The app features a comprehensive quiz system with:
- **Large question pools** (20-25 questions per topic)
- **Random selection** of 10 questions per quiz
- **Fresh experience** every time you retake a quiz
- **Progress tracking** with score history

## Mastery System

The app features an **automatic mastery tracking system** that updates your progress based on quiz performance:

### Status Levels

- **Not Started**: You haven't taken any quizzes for this topic yet
- **Reviewing**: You're actively learning and practicing
- **Mastered**: You've demonstrated consistent high performance

### How to Achieve Mastery

Mastery status is automatically earned when you meet one of these criteria:

1. **Consistent Performance**: 
   - Complete at least 2 quizzes
   - Achieve 80%+ average on your last 3 quizzes

2. **Perfect Streak**:
   - Score 100% on 2 consecutive quizzes

### Maintaining Mastery

### Maintaining Mastery

Your mastery status is dynamic and reflects your current knowledge level:
- If you score below 70% on a quiz, your status may return to "Reviewing"
- If your average drops below 75%, you'll need to practice more
- This ensures your progress reflects actual mastery, not just completion

### Mastery Decay System

To ensure you retain knowledge over time, mastered topics use a **smart decay system**:

- **Mastery Strength**: Each mastered topic has a strength rating (0-100) based on:
  - Average quiz performance
  - Consistency of high scores (90%+)
  - Number of quizzes taken
  - Current high score streak

- **Decay Timeline**: Topics automatically downgrade from "Mastered" to "Reviewing" after a period without practice
  - **Strong mastery** (80-100 strength): Up to 90 days before decay
  - **Medium mastery** (50-79 strength): 30-60 days before decay
  - **Weak mastery** (0-49 strength): 7-30 days before decay

- **Decay Adjustments**: 
  - Perfect scores (100%) extend decay time by 20%
  - High scores (90%+) extend decay time by 10%
  - Lower scores (<80%) reduce decay time by 30%

- **Dashboard Warnings**: Get notified about:
  - Topics decaying within 7 days
  - Topics that have already decayed and need review
  - Your average mastery strength across all topics

This system ensures you regularly review mastered content and maintain long-term retention!

### Benefits

- 🎯 **Clear Goals**: Know exactly what it takes to master each topic
- 📊 **Honest Progress**: Status reflects your actual performance
- 🎓 **Bonus XP**: Earn 50 bonus XP when you achieve mastery
- 🏆 **Challenge Access**: Master topics to unlock advanced challenge modes
- 📈 **Motivated Learning**: Visual feedback encourages consistent practice

A comprehensive platform to master machine learning concepts for technical interviews, built with React and TypeScript.

## Features

- 📚 **Comprehensive Content**: 57 topics covering foundations to advanced ML concepts
- 💬 **342+ Interview Questions**: Detailed 3-5 paragraph answers for each question
- 🧪 **Practice Quizzes**: Multiple choice questions with score tracking and explanations
- � **Code Examples**: Python implementations for key concepts
- �📈 **Progress Tracking**: Mark topics as reviewing or mastered, with localStorage persistence
- 🎮 **Interactive Demo**: Bias-Variance Tradeoff visualization
- 🌙 **Dark/Light Theme**: Toggle between themes for comfortable learning
- 📱 **Responsive Design**: Works seamlessly on desktop and mobile devices
- 🔍 **Category Navigation**: Organized into 8 major ML categories
- 💾 **Auto-Save**: Progress automatically saved to browser localStorage

## Topics Covered

### 1. Foundations (7 topics)
- Supervised vs Unsupervised vs Reinforcement Learning
- Train/Validation/Test Split
- Bias-Variance Tradeoff (with interactive demo)
- Overfitting & Underfitting
- Regularization Techniques
- Cross-Validation
- Evaluation Metrics

### 2. Classical ML (10 topics)
- Linear Regression
- Logistic Regression
- Decision Trees
- Random Forests
- Gradient Boosting
- Support Vector Machines (SVMs)
- K-Nearest Neighbors (KNN)
- K-Means Clustering
- Principal Component Analysis (PCA)
- Naive Bayes

### 3. Neural Networks (7 topics)
- Perceptron
- Multi-Layer Perceptron (MLP)
- Activation Functions
- Backpropagation
- Gradient Descent & Optimization
- Batch Normalization
- Loss Functions

### 4. Computer Vision (6 topics)
- Convolutional Neural Networks (CNNs)
- Pooling Layers
- Classic CNN Architectures (VGG, ResNet, Inception)
- Transfer Learning
- Object Detection
- Image Segmentation

### 5. Natural Language Processing (6 topics)
- Word Embeddings (Word2Vec, GloVe)
- Recurrent Neural Networks (RNNs)
- LSTM & GRU
- Sequence-to-Sequence Models
- Attention Mechanisms
- Encoder-Decoder Architectures

### 6. Transformers & Modern NLP (8 topics)
- Transformer Architecture
- Self-Attention & Multi-Head Attention
- Positional Encoding
- BERT & Variants
- GPT & Variants
- T5 & BART
- Fine-tuning vs Prompt Engineering
- Large Language Models (LLMs)

### 7. Advanced Topics (7 topics)
- Generative Adversarial Networks (GANs)
- Variational Autoencoders (VAEs)
- Reinforcement Learning Basics
- Model Compression (Quantization, Pruning, Distillation)
- Federated Learning
- Few-Shot Learning
- Multi-Modal Models

### 8. ML Systems & Production (7 topics)
- Feature Engineering
- Data Preprocessing & Normalization
- Handling Imbalanced Data
- Model Deployment
- A/B Testing
- Model Monitoring & Drift Detection
- Scaling & Optimization

**Total: 57 comprehensive topics with 342+ interview questions**

## Tech Stack

- **Frontend**: React 18 + TypeScript
- **Styling**: Tailwind CSS
- **Build Tool**: Vite
- **Deployment**: GitHub Pages
- **Icons**: Lucide React

## Development

### Prerequisites

- Node.js 18+
- npm

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ml_review.git
   cd ml_review
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start development server:
   ```bash
   npm run dev
   ```

4. Open http://localhost:5173/ml_review/ in your browser

### Build

```bash
npm run build
```

### Preview build locally

```bash
npm run preview
```

## Deployment

This project is configured for automatic deployment to GitHub Pages using GitHub Actions.

### Setup GitHub Pages

1. Go to your repository Settings
2. Navigate to Pages section
3. Set Source to "GitHub Actions"
4. Push to main branch to trigger deployment

The site will be available at: `https://yourusername.github.io/ml_review/`

### Manual Deployment

You can also deploy manually using gh-pages:

```bash
npm run deploy
```

## Project Structure

```
src/
├── components/          # React components
│   ├── Sidebar.tsx     # Navigation sidebar
│   ├── TopicView.tsx   # Main topic display
│   ├── Quiz.tsx        # Quiz functionality
│   └── ...
├── data/               # Content and data
│   ├── topics.ts       # Topic content
│   └── categories.ts   # Category definitions
├── types/              # TypeScript type definitions
└── hooks/              # Custom React hooks
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests if applicable
5. Commit your changes: `git commit -am 'Add some feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## Adding New Topics

To add new ML topics:

1. Add the topic to `src/data/topics.ts`
2. Include comprehensive content, code examples, and quiz questions
3. Update category mappings in `src/data/categories.ts`
4. For interactive demos, create components in `src/components/`

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Inspired by NeetCode's excellent interview prep platform
- Built with modern React and TypeScript best practices
- Uses Tailwind CSS for responsive design