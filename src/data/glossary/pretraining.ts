// Module 3A · One Machine, Two Jobs
// Terms for pretraining: self-supervision, the loss we train on, and the two games
// (masked-word and next-word) that produced BERT and GPT-2.

import type { GlossaryEntry } from '../glossary';

export const terms: Record<string, GlossaryEntry> = {
  'supervised-learning': {
    term: 'Supervised learning',
    def: 'Training on (input, label) pairs where a person wrote every label: a photo and "cat", a review and "negative". Accurate, but slow and expensive to scale, and each dataset teaches one task.',
    lesson: 'pretraining/self-supervision',
  },
  'self-supervised-learning': {
    term: 'Self-supervised learning',
    def: 'Hide part of the data and train the model to predict the hidden part from the rest. The hidden part is the label, so raw text grades itself and no annotator is needed. How every LLM is pretrained.',
    lesson: 'pretraining/self-supervision',
  },
  'training-objective': {
    term: 'Training objective',
    def: 'The exact game the model is trained to play, and the loss that scores it: "predict the next token from the left context" or "predict masked tokens from both sides". The objective decides what the model is good at and how it is used.',
    lesson: 'pretraining/self-supervision',
  },
  pretraining: {
    term: 'Pretraining',
    def: 'The first, expensive phase of training: learning from a huge amount of raw text with a self-supervised objective. Nearly all the compute goes here, once. Finetuning or prompting then adapts the result cheaply.',
    lesson: 'pretraining/self-supervision',
  },
  'foundation-model': {
    term: 'Foundation model',
    def: 'A model pretrained once on broad data that can be adapted to many downstream tasks by finetuning or prompting. The term was popularised by a 2021 Stanford report; BERT and GPT-2 are early examples.',
    lesson: 'pretraining/self-supervision',
  },
  'cross-entropy': {
    term: 'Cross-entropy loss',
    def: 'The loss every language model is trained on: the average of −ln p(true token) over the targets. Also called negative log-likelihood. Measured in nats (natural log) or bits (log₂); 1 nat = 1.443 bits.',
    lesson: 'pretraining/loss-perplexity',
  },
  'bits-per-byte': {
    term: 'Bits per byte (BPB)',
    def: 'Total loss on a text in bits, divided by the text’s size in UTF-8 bytes: (loss per token ÷ ln 2) × tokens ÷ bytes. Unlike per-token loss or perplexity, it is fair across different tokenizers.',
    lesson: 'pretraining/loss-perplexity',
  },
  'data-contamination': {
    term: 'Data contamination',
    def: 'When a benchmark’s test text was in a model’s training data, so the model has partly memorised the answers. An unusually low perplexity on the test set, compared with fresh text of the same style, is a warning sign.',
    lesson: 'pretraining/loss-perplexity',
  },
  'masked-language-model': {
    term: 'Masked language model (MLM)',
    def: 'BERT’s training game: hide about 15% of the tokens (80% become [MASK], 10% a random token, 10% stay) and predict the originals using both sides of the context. Only the hidden positions are graded.',
    lesson: 'pretraining/bert',
  },
  'encoder-only': {
    term: 'Encoder (encoder-only transformer)',
    def: 'A transformer stack with no causal mask: every token attends to every other, producing a contextual vector per input token. Great for reading tasks (classification, search embeddings), but it cannot generate text. BERT is one.',
    lesson: 'pretraining/bert',
  },
  wordpiece: {
    term: 'WordPiece',
    def: 'BERT’s subword tokenizer, close to BPE, with a vocabulary of about 30,000. Pieces that continue a word are marked with ##, as in "play ##ing".',
    lesson: 'pretraining/bert',
  },
  'cls-token': {
    term: '[CLS] and [SEP]',
    def: 'BERT’s special tokens. [CLS] sits at the front and its final vector summarises the input for classification heads; [SEP] ends each of the (up to two) text segments.',
    lesson: 'pretraining/bert',
  },
  'next-sentence-prediction': {
    term: 'Next-sentence prediction (NSP)',
    def: 'BERT’s second pretraining task: decide from [CLS] whether segment B really followed segment A (50%) or is random text (50%). RoBERTa later showed it could be dropped without loss.',
    lesson: 'pretraining/bert',
  },
};
