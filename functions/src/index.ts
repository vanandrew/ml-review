import { onRequest } from 'firebase-functions/v2/https';
import Anthropic from '@anthropic-ai/sdk';

interface GenerateQuestionsRequest {
  topicId: string;
  topicTitle: string;
  topicContent: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  count: number;
  apiKey: string;
}

// Interface removed - questions are generated as an array now

/**
 * Cloud Function to generate quiz questions using Claude API
 * HTTP endpoint with built-in CORS support (v2 API)
 */
export const generateQuestions = onRequest(
  {
    cors: true, // Enable CORS for all origins
    timeoutSeconds: 540, // 9 minutes max timeout
    memory: '512MiB',
    region: 'us-central1'
  },
  async (req, res) => {
      // Handle preflight OPTIONS request
      if (req.method === 'OPTIONS') {
        res.status(204).send('');
        return;
      }

      if (req.method !== 'POST') {
        res.status(405).json({ error: 'Method not allowed' });
        return;
      }

      try {
        const data: GenerateQuestionsRequest = req.body.data || req.body;
        const { topicId, topicTitle, topicContent, difficulty, count, apiKey } = data;

        // Validate input
        if (!topicTitle || !topicContent || !difficulty || !count || !apiKey) {
          res.status(400).json({ error: 'Missing required parameters' });
          return;
        }

        // Validate API key format
        if (typeof apiKey !== 'string' || !apiKey.startsWith('sk-ant-')) {
          res.status(400).json({ error: 'Invalid API key format' });
          return;
        }

        // Check for invalid characters in API key
        if (/[\n\r\t]/.test(apiKey) || apiKey.length < 20) {
          res.status(400).json({ error: 'API key contains invalid characters' });
          return;
        }

        if (count < 1 || count > 20) {
          res.status(400).json({ error: 'Count must be between 1 and 20' });
          return;
        }

        const client = new Anthropic({ apiKey });

        // Generate all questions in a SINGLE call for better variety and coherence
        const prompt = `Generate exactly ${count} diverse ${difficulty} level quiz questions about: ${topicTitle}

Context:
${topicContent}

CRITICAL VARIETY AND CREATIVITY REQUIREMENTS:
- Generate ${count} questions that cover DIFFERENT aspects of the topic
- Be CREATIVE and SURPRISING - avoid obvious or textbook-style questions
- Use diverse question formats: scenarios, edge cases, counterintuitive situations, debugging scenarios, design choices, trade-offs
- Ensure questions cover a balanced mix of:
  1. Fundamental concepts and definitions (but in novel ways)
  2. Practical applications and use cases (with unexpected twists)
  3. Comparing different approaches (with nuanced distinctions)
  4. Advantages and disadvantages (in specific contexts)
  5. When to use which technique (with real constraints)
  6. Common pitfalls and misconceptions (counter-intuitive ones)
  7. Real-world scenarios (with complexity and ambiguity)
  8. Technical implementation details (specific and technical)
  9. Problem-solving strategies (requiring critical thinking)
  10. Evaluation metrics (in context of trade-offs)

- Each question MUST focus on a different aspect - NO REPETITION
- If the topic has multiple subtopics (e.g., "Supervised vs Unsupervised vs Reinforcement"), ensure BALANCED coverage across ALL subtopics
- Avoid asking the same type of question multiple times (e.g., don't just ask "which paradigm" for every question)
- Make questions thought-provoking and interesting - surprise the learner!
- Test deep understanding and critical thinking, not just memorization
- Each question should have exactly 4 plausible options with only ONE clearly correct answer
- Include detailed explanations that provide insight

Return ONLY a valid JSON array with NO markdown, NO code blocks, in this exact format:
[
  {"question": "...", "options": ["...", "...", "...", "..."], "correctAnswer": 0, "explanation": "..."},
  {"question": "...", "options": ["...", "...", "...", "..."], "correctAnswer": 1, "explanation": "..."}
]`;

        const message = await client.messages.create({
          model: 'claude-sonnet-4-5-20250929',
          max_tokens: 8000, // Increased for multiple questions
          temperature: 1.0, // Maximum temperature for Anthropic API (0-1.0 range) = most creative and varied
          messages: [{ role: 'user', content: prompt }],
        });

        // Extract token usage for accurate cost calculation
        const inputTokens = message.usage.input_tokens;
        const outputTokens = message.usage.output_tokens;

        const contentBlock = message.content[0];
        if (contentBlock.type !== 'text') {
          throw new Error('Unexpected response type from Claude API');
        }
        const content = contentBlock.text;

        let questionsData;
        try {
          const cleanContent = content
            .replace(/```json\n?/g, '')
            .replace(/```\n?/g, '')
            .trim();
          questionsData = JSON.parse(cleanContent);
        } catch (parseError) {
          console.error('Failed to parse questions JSON:', content);
          throw new Error('Invalid response format from Claude API');
        }

        if (!Array.isArray(questionsData) || questionsData.length === 0) {
          throw new Error('Expected an array of questions but got invalid format');
        }

        // Shuffle answer options to randomize correct answer position
        const shuffleAnswers = (question: any) => {
          const correctAnswerText = question.options[question.correctAnswer];
          const shuffledOptions = [...question.options];

          // Fisher-Yates shuffle algorithm
          for (let i = shuffledOptions.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffledOptions[i], shuffledOptions[j]] = [shuffledOptions[j], shuffledOptions[i]];
          }

          // Find new position of correct answer
          const newCorrectIndex = shuffledOptions.indexOf(correctAnswerText);

          return {
            ...question,
            options: shuffledOptions,
            correctAnswer: newCorrectIndex,
          };
        };

        // Add metadata and shuffle each question
        const questions = questionsData.map((q, index) => {
          const shuffled = shuffleAnswers(q);
          return {
            ...shuffled,
            id: `ai-${topicId}-${Date.now()}-${index}`,
            type: 'multiple-choice',
            source: 'ai-generated',
          };
        });

        // Calculate actual cost based on real token usage
        // Sonnet 4.5 pricing: $3/1M input tokens, $15/1M output tokens
        const inputCost = (inputTokens / 1_000_000) * 3.0;   // $3 per million
        const outputCost = (outputTokens / 1_000_000) * 15.0; // $15 per million
        const actualCost = inputCost + outputCost;

        res.status(200).json({
          data: {
            questions,
            metadata: {
              count: questions.length,
              actualCost: actualCost,
              estimatedCost: actualCost, // Backward compatibility - now using actual cost
              costPerQuestion: actualCost / questions.length,
              inputTokens: inputTokens,
              outputTokens: outputTokens,
              totalTokens: inputTokens + outputTokens,
              model: 'claude-sonnet-4-5-20250929',
              generatedAt: new Date().toISOString(),
            },
          }
        });
      } catch (error: any) {
        console.error('Error generating questions:', error);

        if (error.status === 401) {
          res.status(401).json({ error: 'Invalid API key' });
          return;
        }

        if (error.status === 429) {
          res.status(429).json({ error: 'Rate limit exceeded. Please try again later.' });
          return;
        }

        res.status(500).json({ error: 'Failed to generate questions: ' + error.message });
      }
  }
);
