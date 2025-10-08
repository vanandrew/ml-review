import { Topic } from '../../../types';

// Import all topics
import { transformerArchitecture } from './transformerArchitecture';
import { selfAttentionMultiHead } from './selfAttentionMultiHead';
import { positionalEncoding } from './positionalEncoding';
import { visionTransformers } from './visionTransformers';
import { bert } from './bert';
import { gpt } from './gpt';
import { t5Bart } from './t5Bart';
import { fineTuningVsPromptEngineering } from './fineTuningVsPromptEngineering';
import { largeLanguageModels } from './largeLanguageModels';

export const transformersTopics: Record<string, Topic> = {
  'transformer-architecture': transformerArchitecture,
  'self-attention-multi-head': selfAttentionMultiHead,
  'positional-encoding': positionalEncoding,
  'vision-transformers': visionTransformers,
  'bert': bert,
  'gpt': gpt,
  't5-bart': t5Bart,
  'fine-tuning-vs-prompt-engineering': fineTuningVsPromptEngineering,
  'large-language-models': largeLanguageModels,
};
