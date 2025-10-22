"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.generateQuestions = void 0;
const https_1 = require("firebase-functions/v2/https");
const sdk_1 = __importDefault(require("@anthropic-ai/sdk"));
// Interface removed - questions are generated as an array now
/**
 * Cloud Function to generate quiz questions using Claude API
 * HTTP endpoint with built-in CORS support (v2 API)
 */
exports.generateQuestions = (0, https_1.onRequest)({
    cors: true, // Enable CORS for all origins
    timeoutSeconds: 540, // 9 minutes max timeout
    memory: '512MiB',
    region: 'us-central1'
}, async (req, res) => {
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
        const data = req.body.data || req.body;
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
        const client = new sdk_1.default({ apiKey });
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
- Randomize the position of the correct answer among the options
- Include detailed explanations that provide insight

Return ONLY a valid JSON array with NO markdown, NO code blocks, in this exact format:
[
  {"question": "...", "options": ["...", "...", "...", "..."], "correctAnswer": 0, "explanation": "..."},
  {"question": "...", "options": ["...", "...", "...", "..."], "correctAnswer": 1, "explanation": "..."}
]`;
        const message = await client.messages.create({
            model: 'claude-sonnet-4-5-20250929',
            max_tokens: 8000, // Increased for multiple questions
            top_p: 0.95, // When thinking is enabled, only top_p between 0.95-1.0 is allowed (no temperature/top_k)
            thinking: {
                type: 'enabled',
                budget_tokens: 5000, // Allow up to 5000 tokens for thinking (must be >= 1024 and < max_tokens)
            },
            messages: [{ role: 'user', content: prompt }],
        });
        // Extract token usage for accurate cost calculation
        const inputTokens = message.usage.input_tokens;
        const outputTokens = message.usage.output_tokens;
        // With extended thinking, response includes thinking blocks followed by text blocks
        // Find the first text content block (skip thinking/redacted_thinking blocks)
        const textBlock = message.content.find(block => block.type === 'text');
        if (!textBlock || textBlock.type !== 'text') {
            throw new Error('No text content block found in Claude API response');
        }
        const content = textBlock.text;
        let questionsData;
        try {
            const cleanContent = content
                .replace(/```json\n?/g, '')
                .replace(/```\n?/g, '')
                .trim();
            questionsData = JSON.parse(cleanContent);
        }
        catch (parseError) {
            console.error('Failed to parse questions JSON:', content);
            throw new Error('Invalid response format from Claude API');
        }
        if (!Array.isArray(questionsData) || questionsData.length === 0) {
            throw new Error('Expected an array of questions but got invalid format');
        }
        // Add metadata to each question
        const questions = questionsData.map((q, index) => (Object.assign(Object.assign({}, q), { id: `ai-${topicId}-${Date.now()}-${index}`, type: 'multiple-choice', source: 'ai-generated' })));
        // Calculate actual cost based on real token usage
        // Sonnet 4.5 pricing: $3/1M input tokens, $15/1M output tokens
        const inputCost = (inputTokens / 1000000) * 3.0; // $3 per million
        const outputCost = (outputTokens / 1000000) * 15.0; // $15 per million
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
    }
    catch (error) {
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
});
//# sourceMappingURL=index.js.map