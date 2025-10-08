import { Topic } from '../../../types';

// Import all topics
import { wordEmbeddings } from './wordEmbeddings';
import { recurrentNeuralNetworks } from './recurrentNeuralNetworks';
import { lstmGru } from './lstmGru';
import { seq2seqModels } from './seq2seqModels';
import { attentionMechanism } from './attentionMechanism';
import { encoderDecoderArchitecture } from './encoderDecoderArchitecture';

export const nlpTopics: Record<string, Topic> = {
  'word-embeddings': wordEmbeddings,
  'recurrent-neural-networks': recurrentNeuralNetworks,
  'lstm-gru': lstmGru,
  'seq2seq-models': seq2seqModels,
  'attention-mechanism': attentionMechanism,
  'encoder-decoder-architecture': encoderDecoderArchitecture,
};
