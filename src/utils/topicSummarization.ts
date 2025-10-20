import { Topic } from '../types';
import { TopicContext } from '../types/ai';

/**
 * Strip HTML tags from content
 */
const stripHTML = (html: string): string => {
  // Create a temporary div element
  const tmp = document.createElement('div');
  tmp.innerHTML = html;
  return tmp.textContent || tmp.innerText || '';
};

/**
 * Extract bullet points from text
 */
const extractBulletPoints = (text: string): string[] => {
  const lines = text.split('\n');
  const bullets: string[] = [];

  for (const line of lines) {
    const trimmed = line.trim();
    // Match lines starting with -, *, •, or numbered lists
    if (/^[-*•]\s/.test(trimmed) || /^\d+\.\s/.test(trimmed)) {
      bullets.push(trimmed.replace(/^[-*•]\s/, '').replace(/^\d+\.\s/, ''));
    }
  }

  return bullets;
};

/**
 * Extract definitions (text that looks like "Term: definition")
 */
const extractDefinitions = (text: string): string[] => {
  const lines = text.split('\n');
  const definitions: string[] = [];

  for (const line of lines) {
    const trimmed = line.trim();
    // Match patterns like "Term: definition" or "**Term**: definition"
    if (/:/.test(trimmed) && trimmed.length < 200) {
      definitions.push(trimmed);
    }
  }

  return definitions.slice(0, 5); // Limit to 5 definitions
};

/**
 * Extract mathematical formulas and equations
 */
const extractFormulas = (text: string): string[] => {
  const formulas: string[] = [];

  // Extract content between $ signs (LaTeX math)
  const mathMatches = text.match(/\$([^$]+)\$/g);
  if (mathMatches) {
    formulas.push(...mathMatches.map(m => m.replace(/\$/g, '')));
  }

  // Extract lines with common math symbols
  const lines = text.split('\n');
  for (const line of lines) {
    if (/[=+\-*/^∂∇∑∏∫]/.test(line) && line.length < 100) {
      formulas.push(line.trim());
    }
  }

  // Remove duplicates and limit to 3
  return [...new Set(formulas)].slice(0, 3);
};

/**
 * Extract key concepts (words in bold, headers, etc.)
 */
const extractKeyConcepts = (html: string): string[] => {
  const concepts: string[] = [];

  // Extract text in <strong> or <b> tags
  const strongMatches = html.match(/<(?:strong|b)>(.*?)<\/(?:strong|b)>/g);
  if (strongMatches) {
    concepts.push(...strongMatches.map(m => stripHTML(m)));
  }

  // Extract headers
  const headerMatches = html.match(/<h[1-6]>(.*?)<\/h[1-6]>/g);
  if (headerMatches) {
    concepts.push(...headerMatches.map(m => stripHTML(m)));
  }

  // Remove duplicates and limit to 10
  return [...new Set(concepts)].slice(0, 10);
};

/**
 * Create a condensed context from topic content for AI prompts
 * This reduces token usage while preserving important information
 */
export const createTopicContext = (topic: Topic): TopicContext => {
  const textContent = stripHTML(topic.content);
  const keyConcepts = extractKeyConcepts(topic.content);
  const bulletPoints = extractBulletPoints(textContent);
  const definitions = extractDefinitions(textContent);
  const formulas = extractFormulas(textContent);

  // Combine unique key points from concepts and bullets
  const keyPoints = [...new Set([...keyConcepts, ...bulletPoints])].slice(0, 10);

  return {
    title: topic.title,
    keyPoints,
    definitions,
    formulas,
  };
};

/**
 * Format topic context as a condensed string for AI prompts
 * Maximum ~500 tokens
 */
export const formatTopicContextForPrompt = (context: TopicContext): string => {
  const parts: string[] = [`Topic: ${context.title}`];

  if (context.keyPoints.length > 0) {
    parts.push('\nKey Concepts:');
    context.keyPoints.forEach((point, i) => {
      parts.push(`${i + 1}. ${point}`);
    });
  }

  if (context.definitions.length > 0) {
    parts.push('\nDefinitions:');
    context.definitions.forEach(def => {
      parts.push(`- ${def}`);
    });
  }

  if (context.formulas.length > 0) {
    parts.push('\nImportant Formulas:');
    context.formulas.forEach(formula => {
      parts.push(`- ${formula}`);
    });
  }

  return parts.join('\n');
};

/**
 * Get condensed topic content for AI question generation
 * This is the main function to use when preparing content for AI
 */
export const getCondensedTopicContent = (topic: Topic): string => {
  const context = createTopicContext(topic);
  return formatTopicContextForPrompt(context);
};

/**
 * Estimate token count for a string (rough approximation)
 * 1 token ≈ 4 characters for English text
 */
export const estimateTokenCount = (text: string): number => {
  return Math.ceil(text.length / 4);
};

/**
 * Calculate token savings from using condensed content
 */
export const calculateTokenSavings = (
  originalContent: string,
  condensedContent: string
): { original: number; condensed: number; savings: number; savingsPercent: number } => {
  const originalTokens = estimateTokenCount(originalContent);
  const condensedTokens = estimateTokenCount(condensedContent);
  const savings = originalTokens - condensedTokens;
  const savingsPercent = (savings / originalTokens) * 100;

  return {
    original: originalTokens,
    condensed: condensedTokens,
    savings,
    savingsPercent,
  };
};

/**
 * Hash content for cache invalidation
 */
export const hashContent = (content: string): string => {
  let hash = 0;
  for (let i = 0; i < content.length; i++) {
    const char = content.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  return hash.toString(36);
};
