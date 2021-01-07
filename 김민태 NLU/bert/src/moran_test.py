# Copyright 2020 The TwoBlock AI.
#
# youngcho@tbai.info, youngcho@gmail.com


import moran

moran_tokenizer = moran.MoranTokenizer()
x = '한국어 BERT를 공개합니다.'
moran_line = moran.text2moran(x, moran_tokenizer)
print(moran_line)
x = '<table> <tr> <td> 한국어 BERT를 공개합니다. </td> </tr> </table>'
moran_line = moran.text2moran(x, moran_tokenizer)
print(moran_line)

print(f"{x} -> {moran_line}")

