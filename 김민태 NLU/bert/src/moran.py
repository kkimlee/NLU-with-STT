# coding=utf-8
# Copyright 2020 The TwoBlock AI.
#
# youngcho@tbai.info, youngcho@gmail.com


from ctypes import *

moran = CDLL('/usr/local/moran/libmoran4dnlp.so')
moran.Moran4dnlp.restype=c_char_p
moran.Moran4dnlp.argtypes=[c_char_p, c_char_p, c_int]

moran.Moran4match.restype=c_char_p
moran.Moran4match.argtypes=[c_char_p, c_char_p, c_char_p, c_int]

moran_input  = create_string_buffer(102400)
moran_output = create_string_buffer(102400)

all_html_tag_list = '<table> <caption> </caption> <tbody> <tr> <th> </th> </tr> <span> </span> <td> <a> </a> <br/> </td> </tbody> </ta
ble> <p> <b> </b> </p> <div> <input> <h2> </h2> <label> </label> </div> <ul> <li> </li> </ul> <img> <hr/> <abbr> </abbr> <s> </s> <!>
<!!> <sup> </sup> <big> </big> <i> </i> <ol> </ol> <dl> <dt> </dt> <dd> </dd> </dl> <sub> </sub> <strong> </strong> <small> </small> <
br> <a!!> <map> <area> </map> <pre> </pre> <noscript> </noscript> <mokcha> </mokcha>'

class MorAn16(object):
    def __init__(self):
        moran.MorAn16_open_dbs()

    def close(self):
        moran.MorAn_close_dbs()

    def run(self, text):
        result = list()
        moran_input.value = text.encode()
        x = moran.Moran4dnlp(moran_input, moran_output, 102400)
        y = x.decode()
        result = y.split()
        return result
class MoranTokenizer(object):

  def __init__(self) :
    self.moran = MorAn16()

  def tokenize(self, text):
    output_tokens = self.moran.run(text)
    return output_tokens

def text2moran(text, tokenizer) :
  res = ''
  text_list = text.split()
  sub_text = ''
  for x in text_list :
    if x[0] == '<' and x[-1] == '>' and all_html_tag_list.find(x) != -1 :
      if sub_text != '' :
        y = tokenizer.tokenize(sub_text)
        if res == '' : res = ' '.join(y)
        else : res += ' ' + ' '.join(y)
        sub_text = ''
      if res == '' : res = x
      else : res += ' ' + x
    else :
      if sub_text == '' : sub_text = x
      else : sub_text += ' ' + x
      if (len(sub_text) > 100 and x.find('다.') != -1) or (len(sub_text) > 150 and  x[-1] == ',') :
        y = tokenizer.tokenize(sub_text)
        if res == '' : res = ' '.join(y)
        else : res += ' ' + ' '.join(y)
        sub_text = ''
  if sub_text != '' :
    y = tokenizer.tokenize(sub_text)
    if res == '' : res = ' '.join(y)
    else : res += ' ' + ' '.join(y)

  result = res.split()
  return result
