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
};
