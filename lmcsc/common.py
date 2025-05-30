from string import punctuation

RADICAL_INDEX = 0
COMPONENT_INDEX = 1
OOV_CHAR = "□"

MIN = -1e32
MAX = 1e32
EPS = 1e-7

chinese_punct = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟–—‘'‛“”„‟…‧."
english_punct = punctuation
PUNCT = set(chinese_punct + english_punct)