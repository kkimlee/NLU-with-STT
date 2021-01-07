# coding=utf-8
#
# Modified by TwoBlock Ai
# youngcho@tbai.info, youngcho@gmail.com
#
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import ast
import math
import os
import random
import modeling
import optimization
import tokenization
import six
import tensorflow as tf

import sys
import json
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.flags
FLAGS = flags.FLAGS

global eval_examples, eval_features
eval_features = list()
eval_examples = list()

## Required parameters
flags.DEFINE_string(
    "bert_config_file", "HanBert-54kN-MRC/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", "HanBert-54kN-MRC/vocab_54k.txt",
                    "The vocabulary file that the BERT model was trained on.")


flags.DEFINE_string(
    "init_checkpoint", 'HanBert-54kN-MRC',
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "mod_kor", True,
    "Whether to use bug fixing code for Korean. "
    "Will be removed later after validity.")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_integer(
    "n_best_size", 5,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

class SquadExample(object):
  """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

  def __init__(self,
               question_text,
               doc_tokens,
               start_position=None,
               end_position=None,
               is_impossible=False):
    self.question_text = question_text
    self.doc_tokens = doc_tokens
    self.start_position = start_position
    self.end_position = end_position
    self.is_impossible = is_impossible

  def __str__(self):
    return self.__repr__()
    
  def __repr__(self):
    s = ""
    s += "question_text: %s" % (
        tokenization.printable_text(self.question_text))
    s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
    if self.start_position:
      s += ", start_position: %d" % (self.start_position)
    if self.start_position:
      s += ", end_position: %d" % (self.end_position)
    if self.start_position:
      s += ", is_impossible: %r" % (self.is_impossible)
    return s


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               doc_span_index,
               tokens,
               token_to_orig_map,
               token_is_max_context,
               input_ids,
               input_mask,
               segment_ids,
               start_position=None,
               end_position=None,
               is_impossible=None):
    self.unique_id = unique_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.tokens = tokens
    self.token_to_orig_map = token_to_orig_map
    self.token_is_max_context = token_is_max_context
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.start_position = start_position
    self.end_position = end_position
    self.is_impossible = is_impossible


def read_kpmg_examples(input_context, input_question):
  """Read a SQuAD json file into a list of SquadExample."""
  def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
      return True
    return False

  examples = list()

  question_text = input_question
  start_position = None
  end_position = None
  is_impossible = False

  paragraph_text = input_context
  doc_tokens = []
  prev_is_whitespace = True
  for c in paragraph_text:
    if is_whitespace(c):
      prev_is_whitespace = True
    else:
      if prev_is_whitespace:
        doc_tokens.append(c)
      else:
        doc_tokens[-1] += c
      prev_is_whitespace = False

#  print(f"doc_tokens = {doc_tokens}")

  example = SquadExample(
    question_text=question_text,
    doc_tokens=doc_tokens,  # list of tokenized words
    start_position=start_position,  # start word index
    end_position=end_position,  # end word index
    is_impossible=is_impossible)
  examples.append(example)

  return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length):
  """Loads a data file into a list of `InputBatch`s."""

  all_features = list()

  unique_id = 1000000000

  for (example_index, example) in enumerate(examples):
    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > max_query_length:
      query_tokens = query_tokens[0:max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []

    # TODO : doc_tokens need to be analyzed by Mecab or else
    # TODO : need to substitute "for" block to make different "all_doc_tokens"
    for (i, token) in enumerate(example.doc_tokens):
      orig_to_tok_index.append(len(all_doc_tokens))  # index : word index, value : subtoken index
      sub_tokens = tokenizer.tokenize(token)
      for sub_token in sub_tokens:
        tok_to_orig_index.append(i)  # index : subtoken index, value : word index
        all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
      length = len(all_doc_tokens) - start_offset
      if length > max_tokens_for_doc:
        length = max_tokens_for_doc
      doc_spans.append(_DocSpan(start=start_offset, length=length))
      if start_offset + length == len(all_doc_tokens):
        break
      start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
      tokens = []
      token_to_orig_map = {}
      token_is_max_context = {}
      segment_ids = []
      tokens.append("[CLS]")
      segment_ids.append(0)
      for token in query_tokens:
        tokens.append(token)
        segment_ids.append(0)
      tokens.append("[SEP]")
      segment_ids.append(0)

      for i in range(doc_span.length):
        split_token_index = doc_span.start + i
        token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

        is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                               split_token_index)
        token_is_max_context[len(tokens)] = is_max_context
        tokens.append(all_doc_tokens[split_token_index])
        segment_ids.append(1)
      tokens.append("[SEP]")
      segment_ids.append(1)

#      print(f"Think : {tokens}")
      input_ids = tokenizer.convert_tokens_to_ids(tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length

      start_position = None
      end_position = None


      if not (start_position == end_position == 0 and not example.is_impossible):
        feature = InputFeatures(
            unique_id=unique_id,
            example_index=example_index,
            doc_span_index=doc_span_index,
            tokens=tokens,
            token_to_orig_map=token_to_orig_map,
            token_is_max_context=token_is_max_context,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            start_position=start_position,
            end_position=end_position,
            is_impossible=example.is_impossible)

        # Run callback
        all_features.append(feature)
        unique_id += 1

    return all_features

def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids)

  final_hidden = model.get_sequence_output()

  final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]
  hidden_size = final_hidden_shape[2]

  output_weights = tf.get_variable(
      "cls/squad/output_weights", [2, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

  final_hidden_matrix = tf.reshape(final_hidden,
                                   [batch_size * seq_length, hidden_size])
  logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)

  logits = tf.reshape(logits, [batch_size, seq_length, 2]) # (b, n, 2)
  logits = tf.transpose(logits, [2, 0, 1]) # (2, b, n)

  unstacked_logits = tf.unstack(logits, axis=0)

  (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1]) # (b, n), (b, n)

  return (start_logits, end_logits)


def model_fn_builder(bert_config, init_checkpoint, num_train_steps, num_warmup_steps) :
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (start_logits, end_logits) = create_model(
        bert_config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids)

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
        # tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          "unique_ids": unique_ids,
          "start_logits": start_logits,
          "end_logits": end_logits,
      }
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode, predictions=predictions, scaffold=scaffold_fn)
    else:
      raise ValueError(
          "Only TRAIN and PREDICT modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def input_fn_builder(input_data, tokenizer, seq_length, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  def gen() :

    global input_idx
    input_idx = -1

    while input_idx < len(input_data)-1 :
      # receive request
      input_idx += 1
      while True :
        input_context = input_data[input_idx]
        print(f"* 명령어  : 1) 내용보기, 2) 다음 예제, 3) 끝내기")
        print(f"* 질의문  : ", end='')
        input_question = input()
        if input_question == 'show' or input_question == '1' :
          print(f"본문내용 : {input_context}\n\n")
          continue
        if input_question == 'next' or input_question == '2' :
          print("다음 문서로 갑니다.\n")
          if input_idx == len(input_data)-1 :
            print("처음 문서로 갑니다.\n")
            input_idx = -1
          break
        if input_question == 'quit' or input_question == '3' or input_question == 'q' :
          return

        examples = read_kpmg_examples(input_context, input_question)
        features = convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            doc_stride=FLAGS.doc_stride,
            max_query_length=FLAGS.max_query_length)
        eval_examples.append(examples)
        eval_features.append(features)
        print(f"... Reading : {len(features)} pages Start ... ",  end='')

        yield {'unique_ids'     : [f.unique_id for f in features],
               'input_ids'      : [f.input_ids for f in features],
               'input_mask'     : [f.input_mask for f in features],
               'segment_ids'    : [f.segment_ids for f in features]}

  def input_fn():
      return (tf.data.Dataset.from_generator(gen,
        output_types={'unique_ids': tf.int32, 'input_ids': tf.int32, 'input_mask': tf.int32, 'segment_ids': tf.int32},
        output_shapes={'unique_ids': [None], 'input_ids': [None, seq_length], 'input_mask': [None, seq_length],'segment_ids': [None, s
eq_length]}))

  return input_fn


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, analyze_kor_morph, tokenizer):
  """Write final predictions to the json file and log-odds of null if needed."""

  example_index_to_features = collections.defaultdict(list)
  for feature in all_features:
    example_index_to_features[feature.example_index].append(feature)

  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
      "PrelimPrediction",
      ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

  for (example_index, example) in enumerate(all_examples):
    features = example_index_to_features[example_index]

    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    score_null = 1000000  # large and positive
    min_null_feature_index = 0  # the paragraph slice with min mull score
    null_start_logit = 0  # the start logit at the slice with min null score
    null_end_logit = 0  # the end logit at the slice with min null score
    for (feature_index, feature) in enumerate(features):
      result = unique_id_to_result[feature.unique_id]
      start_indexes = _get_best_indexes(result.start_logits, n_best_size)
      end_indexes = _get_best_indexes(result.end_logits, n_best_size)
      for start_index in start_indexes:
        for end_index in end_indexes:
          if start_index >= len(feature.tokens):
            continue
          if end_index >= len(feature.tokens):
            continue
          if start_index not in feature.token_to_orig_map:
            continue
          if end_index not in feature.token_to_orig_map:
            continue
          if end_index < start_index:
            continue
          length = end_index - start_index + 1
          prelim_predictions.append(
              _PrelimPrediction(
                  feature_index=feature_index,
                  start_index=start_index,
                  end_index=end_index,
                  start_logit=result.start_logits[start_index],
                  end_logit=result.end_logits[end_index]))

    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_logit", "end_logit"])

    seen_predictions = {}
    nbest = []
    for pred_index, pred in enumerate(prelim_predictions):
      if len(nbest) >= n_best_size:
        break
      feature = features[pred.feature_index]
      if pred.start_index > 0:  # this is a non-null prediction
        tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
        orig_doc_start = feature.token_to_orig_map[pred.start_index]
        orig_doc_end = feature.token_to_orig_map[pred.end_index]
        orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
        tok_text = " ".join(tok_tokens)

        # De-tokenize WordPieces that have been split off.
        tok_text = tok_text.replace(" ##", "")
        tok_text = tok_text.replace("##", "")  # "##" at the start of prediction(maybe wrong)

        tok_text_backup = tok_text

        tok_text = tok_text.replace(" ~~", "")
        tok_text = tok_text.replace("~~", "")  # "##" at the start of prediction(maybe wrong)
        tok_text = tok_text.replace(" ~", "")
        tok_text = tok_text.replace("~", "")  # "##" at the start of prediction(maybe wrong)

        tok_text = tok_text.strip()
        tok_text = " ".join(tok_text.split())
        orig_text = " ".join(orig_tokens)
        orig_text_no_space = "".join(orig_tokens)
        tok_text_no_space  = tok_text.replace(" ", "")  # "##" at the start of prediction(maybe wrong)

        final_text = get_final_text(tok_text_backup, orig_text, tokenizer)

        if final_text in seen_predictions:
          continue
        seen_predictions[final_text] = True

      else:
        final_text = ""
        seen_predictions[final_text] = True

      nbest.append(
          _NbestPrediction(
              text=final_text,
              start_logit=pred.start_logit,
              end_logit=pred.end_logit))

    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
      nbest.append(
          _NbestPrediction(text="N/A", start_logit=0.0, end_logit=0.0))

    assert len(nbest) >= 1

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
      total_scores.append(entry.start_logit + entry.end_logit)
      if not best_non_null_entry:
        if entry.text:
          best_non_null_entry = entry

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_logit"] = entry.start_logit
      output["end_logit"] = entry.end_logit
      nbest_json.append(output)

    assert len(nbest_json) >= 1

  return nbest_json[0]["text"], nbest_json[0]['start_logit'] + nbest_json[0]['end_logit'], nbest_json[0]['probability']


def get_final_text(pred_text, orig_text, tokenizer):
  output_text = tokenizer.match(orig_text, pred_text)

  return output_text


def _get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs

def read_korquad_examples(input_file):
  with tf.gfile.Open(input_file, "r") as reader:
    input_data = json.load(reader)["data"]

  examples = []
  for entry in input_data:
    for paragraph in entry["paragraphs"]:
      paragraph_text = paragraph["context"]
      examples.append(paragraph_text)

  return examples


def validate_flags_or_throw(bert_config):
  """Validate the input FLAGS or throw an exception."""
  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_query_length "
        "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))

def main(argv):
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  validate_flags_or_throw(bert_config)

  tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case, use_moran=True)

  num_train_steps = None
  num_warmup_steps = None

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=tf.estimator.RunConfig(session_config=config))

  basic_tokenizer = tokenization.BasicTokenizer(use_moran=False)

  examples = read_korquad_examples(input_file=FLAGS.example_file)
  rnd = random.Random(12345)
  rnd.shuffle(examples)

  print(f"\n=========== 기계독해 예문 {len(examples)}건 Loadind Done ...  by TBai ==============\n")
  all_results = list()
  output_results = dict()
  for result in estimator.predict(
          input_fn_builder(
              input_data=examples,
              tokenizer=tokenizer,
              seq_length=FLAGS.max_seq_length,
              drop_remainder=False),
          yield_single_examples=False) :

    for idx in range(len(eval_features[0])) :
      unique_id = int(result["unique_ids"][idx])
      start_logits = [float(x) for x in result["start_logits"][idx].flat]
      end_logits = [float(x) for x in result["end_logits"][idx].flat]

      all_results.append(RawResult(
          unique_id=unique_id,
          start_logits=start_logits,
          end_logits=end_logits))

    if len(eval_examples) != 0 and len(eval_features) != 0 :
      answer, score, prob = write_predictions(eval_examples[0], eval_features[0], all_results,
                    FLAGS.n_best_size, FLAGS.max_answer_length,
                    FLAGS.do_lower_case, FLAGS.analyze_kor_morph, basic_tokenizer)

      tscore = str(score)[:5]
      tprob  = str(prob*100)[:5]
      if answer != 'N/A' :
        print(f"Done ... 답변 신뢰도 :  ({tscore}, {tprob}%)")
      else :
        print(f"Done ...")

      res = "독해결과  : "  + answer + "\n\n"
      print(f"{res}")

    all_results.clear()
    eval_features.clear()
    eval_examples.clear()

  print("\n\n\t투블럭에이아이에서 제공하여 드렸습니다. https://twoblockai.com/\n\n")

if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  tf.app.run()



