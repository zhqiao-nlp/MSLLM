import argparse
import pickle
import time
import datetime

import os
import torch
import torch.nn.functional as F
from lmcsc.model import LMCSCModel
from lmcsc.common import OOV_CHAR
from lmcsc.utils import clean_sentences, rebuild_sentences
from accelerate import Accelerator
from lmcsc.autocsc import *
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
from tqdm.auto import tqdm, trange


class InputExample(object):
    def __init__(self, guid, src, trg):
        self.guid = guid
        self.src = src
        self.trg = trg


class InputFeatures(object):
    def __init__(self, src_ids, attention_mask, trg_ids, trg_ref_ids=None):
        self.src_ids = src_ids
        self.attention_mask = attention_mask
        self.trg_ids = trg_ids
        self.trg_ref_ids = trg_ref_ids


class DataProcessor:
    """
    Processor for the data set:
    a) in a .tsv format, i.e. src\ttrg; b) separate Chinese characters from each other by spaces; c) without headlines.
    """

    def get_train_examples(self, data_dir, filename):
        return self._create_examples(self._read(os.path.join(data_dir, filename)), "train")

    def get_dev_examples(self, data_dir, filename):
        return self._create_examples(self._read(os.path.join(data_dir, filename)), "dev")

    def get_test_examples(self, lines):
        return self._create_examples(lines, "test")

    @staticmethod
    def _read(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                src, trg = line.strip().split("\t")
                lines.append((src.split(), trg.split()))
            return lines

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (src, trg) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if len(src) == len(trg):
                examples.append(InputExample(guid=guid, src=src, trg=trg))
        return examples


class DataProcessorForRephrasing(DataProcessor):
    @staticmethod
    def convert_examples_to_features(examples, max_seq_length, tokenizer, verbose=True):
        features = []
        for i, example in enumerate(examples):
            src_ids = tokenizer(example.src,
                                max_length=max_seq_length // 2 - 2,
                                truncation=True,
                                is_split_into_words=True,
                                add_special_tokens=False).input_ids
            trg_ids = tokenizer(example.trg,
                                max_length=max_seq_length // 2 - 2,
                                truncation=True,
                                is_split_into_words=True,
                                add_special_tokens=False).input_ids
            input_ids = [tokenizer.cls_token_id] + src_ids + [tokenizer.sep_token_id] + [tokenizer.mask_token_id for
                                                                                         _ in trg_ids] + [
                            tokenizer.sep_token_id]
            label_ids = [tokenizer.cls_token_id] + src_ids + [tokenizer.sep_token_id] + trg_ids + [
                tokenizer.sep_token_id]
            attention_mask = [1] * len(input_ids)
            ref_ids = [tokenizer.cls_token_id] + trg_ids + [tokenizer.sep_token_id] + trg_ids + [
                tokenizer.sep_token_id]

            offset_length = max_seq_length - len(input_ids)
            if offset_length > 0:
                input_ids = input_ids + [tokenizer.pad_token_id] * offset_length
                attention_mask = attention_mask + [0] * offset_length
                label_ids = label_ids + [tokenizer.pad_token_id] * offset_length
                ref_ids = ref_ids + [tokenizer.pad_token_id] * offset_length
            input_ids, attention_mask, label_ids, ref_ids = input_ids[:max_seq_length], attention_mask[
                                                                                        :max_seq_length], label_ids[
                                                                                                          :max_seq_length], ref_ids[
                                                                                                                            :max_seq_length]

            assert len(input_ids) == max_seq_length
            assert len(attention_mask) == max_seq_length
            assert len(label_ids) == max_seq_length

            if verbose and i < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("src_tokens: %s" % " ".join(tokenizer.convert_ids_to_tokens(input_ids)))
                logger.info("trg_tokens: %s" % " ".join(tokenizer.convert_ids_to_tokens(label_ids)))
                logger.info("src_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("trg_ids: %s" % " ".join([str(x) for x in label_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))

            features.append(
                InputFeatures(src_ids=input_ids,
                              attention_mask=attention_mask,
                              trg_ids=label_ids,
                              trg_ref_ids=ref_ids)
            )
        return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_slow_tokenizer", action="store_true",
                        help="A slow tokenizer will be used if passed.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Whether not to use CUDA when available.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use mixed precision.")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum total input sequence length after word-piece tokenization.")
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--bert-name", type=str, required=True)
    # decoding parameters
    parser.add_argument(
        "--batch-size", type=int, default=200, help="Number of characters in each batch"
    )
    parser.add_argument(
        "--max-sentences-per-batch", type=int, default=128, help="Number of sentences in each batch"
    )
    parser.add_argument(
        "--n-beam", type=int, default=8, help="Number of beams in beam search"
    )
    parser.add_argument(
        "--n-beam-hyps-to-keep",
        type=int,
        default=1,
        help="Number of beams to keep in beam search",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum length of the corrected sentence",
    )
    parser.add_argument(
        "--deocode-prefix",
        type=str,
        default="",
        help="Prefix to add to the input sentence",
    )
    # noise distortion model parameters
    parser.add_argument(
        "--n-observed-chars",
        type=int,
        default=8,
        help="How many next characters to observe",
    )
    parser.add_argument(
        "--shape-similar-threshold",
        type=float,
        default=0.45,
        help="Threshold for shape similarity",
    )
    parser.add_argument(
        "--distortion-model-smoothing",
        type=float,
        default=-15.0,
        help="Smoothing for distortion model",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Hyperparameter for the small model probs"
    )
    parser.add_argument(
        "--beta", type=float, default=0.5, help="Hyperparameter for the distortion probs"
    )
    parser.add_argument(
        "--use-faithfulness-reward",
        action="store_true",
        help="Whether to use faithfulness reward",
    )

    args = parser.parse_args()

    processor = DataProcessorForRephrasing()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device: {}, n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, "-accelerate", args.fp16))
    tokenizer = AutoTokenizer.from_pretrained("/bert-base-chinese",
                                              do_lower_case=args.do_lower_case,
                                              use_fast=not args.use_slow_tokenizer,
                                              add_prefix_space=True)

    accelerator = Accelerator(cpu=args.no_cuda, mixed_precision="fp16" if args.fp16 else "no")

    model_state_dict = ""

    # AutoCSCfinetune AutoCSCSoftMasked AutoCSCReLM
    if args.bert_name == "relm":
        model = AutoCSCReLM.from_pretrained("/bert-base-chinese",
                                        state_dict=torch.load(model_state_dict))
    elif args.bert_name == "ftbert":
        model = AutoCSCfinetune.from_pretrained("/bert-base-chinese",
                                            state_dict=torch.load(model_state_dict))
    if args.bert_name == "ftbert":
        args.max_seq_length = 256
    elif args.bert_name == "relm":
        args.max_seq_length = 256
    model = accelerator.prepare(model)
    print("using model:", model_state_dict)



    dataset = "_".join(args.input_file.split("/")[1:])
    dataset = ".".join(dataset.split(".")[:-1])
    print(f"Dataset: {dataset}")
    args.output_file = f"{args.path}/prediction.txt"
    os.makedirs(args.path, exist_ok=True)

    sources = []
    for line in open(args.input_file, "r"):
        source, _ = line.split("\t")
        sources.append(source.strip())

    # baichuan-inc/Baichuan2-7B-Base, Baichuan2 is the model_family
    model_family = args.model_name.split("/")[-1].split("-")[0]
    lmcsc_model = LMCSCModel(
        args.model_name,
        n_beam=args.n_beam,
        n_beam_hyps_to_keep=args.n_beam_hyps_to_keep,
        alpha=args.alpha,
        beta=args.beta,
        n_observed_chars=args.n_observed_chars,
        shape_similar_threshold=args.shape_similar_threshold,
        distortion_model_smoothing=args.distortion_model_smoothing,
        use_faithfulness_reward=args.use_faithfulness_reward,
    )

    # reorder sources by length, from longest to shortest
    src_index, reordered_sources = zip(
        *sorted(enumerate(sources), key=lambda x: len(x[1]), reverse=True)
    )
    reorder_index, _ = zip(*sorted(enumerate(src_index), key=lambda x: x[1]))

    hypos = []

    batch_size = args.batch_size
    i = 0
    sentence_id = 0
    batch = []
    ori_batch = []
    cur_batch_size = 0
    start = time.time()
    while i < len(sources):
        batch_start = time.time()
        sentence_id = i
        sentence_ids = []
        # Build batch
        while True:
            if len(batch) == 0 or (
                    (cur_batch_size + len(reordered_sources[i])) < batch_size
            ):
                ori_batch.append(reordered_sources[i])
                sentence_ids.append(src_index[i])
                if "uer" in args.model_name:
                    batch.append("".join(reordered_sources[i][: args.max_length].split()))
                else:
                    batch.append(reordered_sources[i][: args.max_length])
                cur_batch_size += min(len(reordered_sources[i]), args.max_length)
                i += 1
                if i >= len(reordered_sources):
                    break
                if len(batch) > args.max_sentences_per_batch:
                    break
            else:
                break
        print(ori_batch[0])
        batch, changes = clean_sentences(batch)

        small_model_input = [(' '.join(list(ori_batch[i].lower())).split(), ' '.join(list(ori_batch[i].lower())).split()) for i in range(len(ori_batch))]
        eval_examples = processor.get_test_examples(small_model_input)
        eval_features = processor.convert_examples_to_features(eval_examples, args.max_seq_length, tokenizer, False)

        all_input_ids = torch.tensor([f.src_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.trg_ids for f in eval_features], dtype=torch.long)
        all_ref_ids = torch.LongTensor([f.trg_ref_ids for f in eval_features])
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_ref_ids)

        eval_dataloader = DataLoader(eval_data, shuffle=False, batch_size=200)
        eval_dataloader = accelerator.prepare(eval_dataloader)

        def decode(input_ids):
            return tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True)

        model.eval()

        for small_model_batch in eval_dataloader:
            small_model_batch = tuple(t.to(device) for t in small_model_batch)
            src_ids, attention_mask, trg_ids = small_model_batch[:3]
            # print(src_ids[0])
            with torch.no_grad():
                outputs = model(src_ids=src_ids,
                                attention_mask=attention_mask,
                                trg_ids=trg_ids)
                prob_distribute = outputs["prob_distribute"]
                prob_distribute = F.pad(prob_distribute, (0, 0, 0, 10), "constant", -15)
                prob_distribute[:, :, 0] = -15
                prob_distribute[:, :, 100] = -15
        output = lmcsc_model(batch, sentence_ids, prob_distribute, [args.deocode_prefix] * len(batch))
        output = rebuild_sentences(output, changes)
        output = [
            (o + s[len(o):]) if (len(o) < len(s)) else o
            for o, s in zip(output, ori_batch)
        ]

        # Handle oov characters
        output = [
            "".join(
                [
                    o_char if o_char != OOV_CHAR else s_char
                    for o_char, s_char in zip(o, s)
                ]
            )
            for o, s in zip(output, ori_batch)
        ]

        print(output[0])
        hypos.extend(output)
        speed = len(batch) / (time.time() - batch_start)
        print(
            f"Processed: {i}, Speed: {speed:.2f} sentences/sec, Time to go: {datetime.timedelta(seconds=(len(sources) - i) / speed)}"
        )
        print()

        # reset batch
        ori_batch = []
        batch = []
        cur_batch_size = 0

    hypos = [hypos[i].strip().replace("\n", " ") for i in reorder_index]
    output_file = open(args.output_file, "w", encoding="utf-8")
    output_file.write("\n".join(hypos))
    output_file.close()
    print(f"Total time: {datetime.timedelta(seconds=(time.time() - start))}")
