import torch
import torch.nn as nn
from copy import deepcopy
from transformers import BertPreTrainedModel, BertModel, BertForMaskedLM, AutoTokenizer
import numpy as np


class AutoCSCReLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.vocab_size
        self.softmax = nn.Softmax(-1)

    def forward(self, src_ids, trg_ids, attention_mask, alpha=1.1, beta=0.7):
        prob_distribute = torch.full((src_ids.shape[0], 62, 21128), -1.0).cuda()
        labels = trg_ids.clone()
        labels[(src_ids == trg_ids)] = -100

        outputs = self.bert(
            input_ids=src_ids,
            attention_mask=attention_mask,
        )

        sequence_output = outputs[0]
        logits = self.cls(sequence_output)
        loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        probs = nn.functional.log_softmax(logits, -1)

        src = src_ids.clone()
        length = [attention_mask[i].sum().item() for i in range(attention_mask.shape[0])]
        for i in range(src.shape[0]):
            src[i, (length[i] + 1) // 2:length[i] - 1] = src[i, 1:(length[i] - 1) // 2]
            
        for i in range(src.shape[0]):
            prob_distribute[i, ...] = probs[i, (length[i] + 1) // 2: (length[i] + 1) // 2 + 62]

        _, predict_ids = torch.max(probs, -1)

        return {
            "loss": loss,
            "predict_ids": predict_ids,
            "prob_distribute": prob_distribute,
        }


class AutoCSCfinetune(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.vocab_size

        self.bert = BertModel(config, add_pooling_layer=False)
        self.generate_linear = nn.Linear(config.hidden_size, self.num_labels)
        self.softmax = nn.Softmax(-1)

        self.post_init()

    def forward(self, src_ids, trg_ids, attention_mask, alpha=1.1, beta=0.7):
        prob_distribute = torch.full((src_ids.shape[0], 192, 21128), -1.0).cuda()
        outputs = self.bert(
            src_ids,
            attention_mask=attention_mask,
        )

        sequence_output = outputs[0]

        logits = self.generate_linear(sequence_output)
        probs = nn.functional.log_softmax(logits, -1)

        # modification
        src = src_ids.clone()
        for i in range(src.shape[0]):
            prob_distribute[i, ...] = probs[i, 1: 193]

        _, predict_ids = torch.max(probs, -1)
        predict_ids = predict_ids.masked_fill(attention_mask == 0, 0)

        loss_fct = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
        loss = loss_fct(logits.view(-1, self.num_labels), trg_ids.view(-1))

        return {
            "loss": loss,
            "predict_ids": predict_ids,
            "prob_distribute": prob_distribute,
        }


class AutoCSCSoftMasked(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.vocab_size

        self.bert = BertModel(config, add_pooling_layer=False)
        self.generate_linear = nn.Linear(config.hidden_size, self.num_labels)

        self.rnn = nn.GRU(config.hidden_size, 256, batch_first=True,
                          bidirectional=True, num_layers=2, dropout=0.2)

        self.mask_embed = nn.Linear(1, config.hidden_size, bias=False)
        self.copy_linear = nn.Linear(256 * 2, 1)

        self.softmax = nn.Softmax(-1)
        self.sigmoid = nn.Sigmoid()

        self.post_init()

    def forward(self, src_ids, trg_ids, attention_mask, alpha=1.1, beta=0.7):
        prob_distribute = torch.full((src_ids.shape[0], 62, 21128), -1.0).cuda()
        embedding_output = self.bert.embeddings(src_ids)
        detect_embed = self.bert.embeddings.word_embeddings(src_ids)

        rnn_hidden_states, _ = self.rnn(detect_embed)
        copy_logits = self.copy_linear(rnn_hidden_states)
        copy_probs = self.sigmoid(copy_logits)
        embedding_output = copy_probs * embedding_output + self.mask_embed(1 - copy_probs)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_output = self.bert.encoder(embedding_output, extended_attention_mask)[0]
        sequence_output = sequence_output + embedding_output

        logits = self.generate_linear(sequence_output)
        probs = nn.functional.log_softmax(logits, -1)

        # modification
        src = src_ids.clone()
        for i in range(src.shape[0]):
            prob_distribute[i, ...] = probs[i, 1: 63]

        _, predict_ids = torch.max(probs, -1)
        predict_ids = predict_ids.masked_fill(attention_mask == 0, 0)

        loss_fct = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
        loss = loss_fct(logits.view(-1, self.num_labels), trg_ids.view(-1))

        b_loss_fct = nn.BCEWithLogitsLoss(reduction="none")
        b_logits_loss = b_loss_fct(copy_logits.view(-1), (src_ids == trg_ids).float().view(-1))
        b_loss = (b_logits_loss * attention_mask.view(-1)).mean()
        loss = 0.8 * loss + (1 - 0.8) * b_loss

        return {
            "loss": loss,
            "predict_ids": predict_ids,
            "prob_distribute": prob_distribute,
        }


class AutoCSCMDCSpell(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.vocab_size

        self.bert = BertModel(config, add_pooling_layer=False)
        self.generate_linear = nn.Linear(config.hidden_size, self.num_labels)
        self.softmax = nn.Softmax(-1)

        self.detect_layers = deepcopy(self.bert.encoder.layer[:2])
        self.detect_sigmoid = nn.Sequential(nn.Linear(config.hidden_size, 1), nn.Sigmoid())
        self.emb = self.bert.embeddings

        self.post_init()

    def forward(self, src_ids, trg_ids, attention_mask):
        outputs = self.bert(
            src_ids,
            attention_mask=attention_mask,
        )

        sequence_output = outputs[0]

        position_ids = self.emb.position_ids[:, : src_ids.shape[1]]
        token_type_ids = torch.zeros(*src_ids.shape, dtype=torch.long, device=position_ids.device)
        detect_output = self.emb.word_embeddings(src_ids) + self.emb.position_embeddings(
            position_ids) + self.emb.token_type_embeddings(token_type_ids)
        for layer in self.detect_layers:
            detect_output = layer(detect_output)[0]
        detect_probs = self.detect_sigmoid(detect_output)

        logits = self.generate_linear(sequence_output + detect_output)
        probs = self.softmax(logits)

        print("src_ids", src_ids, src_ids.shape)
        print("probs", probs, probs.shape)

        _, predict_ids = torch.max(probs, -1)
        predict_ids = predict_ids.masked_fill(attention_mask == 0, 0)

        loss_fct = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
        loss = loss_fct(logits.view(-1, self.num_labels), trg_ids.view(-1))

        detect_labels = (src_ids != trg_ids).float()
        detect_loss_fct = nn.BCEWithLogitsLoss(size_average=True)
        detect_loss = detect_loss_fct(detect_probs.squeeze(-1) * attention_mask, detect_labels)
        loss = 0.85 * loss + 0.15 * detect_loss

        return {
            "loss": loss,
            "predict_ids": predict_ids,
        }
