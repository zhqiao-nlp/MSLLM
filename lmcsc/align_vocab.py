from collections import defaultdict
import pickle
import json
from tqdm import tqdm
import torch

from lmcsc.common import chinese_punct

class LLM_to_TraditionalModel_Vocab:
    def __init__(self, vocab, is_bytes_level, vocab_size):
        self.vocab = vocab
        self.is_bytes_level = is_bytes_level
        # self.align_vocab = torch.ones(len(self.vocab), 10).int().cuda()
        self.align_vocab = torch.ones(vocab_size, 10).int().cuda()
        self.relm_vocab = [i.strip() for i in open("/bert-base-chinese/vocab.txt")]
        self.traditional_model_vocab = defaultdict(int)
        for i, s in enumerate(self.relm_vocab):
            self.traditional_model_vocab[s] = i
        self.build_index()

    def isChinese(self, s):
        for char in s:
            # 判断字符的 Unicode 编码是否在中文字符范围内
            if not ('\u4e00' <= char <= '\u9fff'):
                return False
        return True

    def build_index(self):
        for k, idx in tqdm(self.vocab.items()):
            ori_token = k
            if self.is_bytes_level:
                try:
                    ori_token = k.decode("utf-8")
                    if len(ori_token) < 10:
                        for i, char in enumerate(ori_token):
                            self.align_vocab[idx, i] = self.traditional_model_vocab.get(char, 100)
                except UnicodeDecodeError:
                    continue
            else:
                if len(ori_token) < 10:
                    for i, char in enumerate(ori_token):
                        self.align_vocab[idx, i] = self.traditional_model_vocab.get(char, 100)