# coding=utf-8
# Modified by TwoBlock AI.
#
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

import os
import tempfile
import six
import tensorflow as tf
import os
import tokenization

if __name__ == "__main__":
    mtokenizer = tokenization.BasicTokenizer(use_moran=True)
    text = "Cat.12가 600 Mbps,"
    toks = "12 ~~가 600 ~mbps"
    output_text = mtokenizer.match(text, toks)
    moran_tokens  = mtokenizer.tokenize(text)
    moran_text = " ".join(moran_tokens)
    print(text)
    print(toks)
    print(moran_tokens)
    print(moran_text)
    print(output_text);


    text = "나는 걸어가고 있는 중입니다. 나는걸어 가고있는 중입니다. 잘 분류되기도 한다. 잘 먹기도 한다."
    btokenizer = tokenization.BasicTokenizer(use_moran=True)
    tokens = btokenizer.tokenize(text)
    print(tokens)
    tok_text = " ".join(tokens)
    print(tok_text)
    tokenizer = tokenization.FullTokenizer("HanBert-54kN/vocab_54k.txt", use_moran=True)
    tokens = tokenizer.tokenize(text)
    print(tokens)
