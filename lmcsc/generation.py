from typing import Tuple, Union, List, Optional
import torch
import torch.distributed as dist
from torch import nn
from copy import deepcopy
from collections import defaultdict
import warnings

from transformers import BertPreTrainedModel, BertModel, BertForMaskedLM, AutoTokenizer
from transformers import (
    BeamScorer,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from transformers.generation.stopping_criteria import validate_stopping_criteria
from transformers.generation.utils import (
    GenerateBeamOutput,
    GenerateBeamEncoderDecoderOutput,
    GenerateBeamDecoderOnlyOutput,
)

from lmcsc.common import MIN, MAX
from lmcsc.obversation_generator import BaseObversationGenerator

def token_transformation_to_probs(self, observed_sequence: str) -> Tuple[List[int], List[float]]:
    token_transformation = self.transformation_type.get_transformation_type(
        observed_sequence
    )
    cache = self.transformation_type_cache
    indices = list(token_transformation.keys())
    # print(len(indices))
    weight = []
    for trans in token_transformation.values():
        trans = tuple(trans)
        if trans in cache:
            w = cache[trans]
        else:
            w = 0
            for tran in trans:
                w += self.distortion_probs[tran]
            cache[trans] = w
        weight.append(w)
    return indices, weight


def small_model_to_probs(self, sentence_id: int, step: int, observed_sequence: str, prob_distribute, align_vocab) -> Tuple[List[int], List[float]]:
    relm_prob_distributes = prob_distribute
    indices = torch.arange(step, step + 10).cuda().view(1, -1)
    mask = align_vocab != 1
    second_dim_first_pos = align_vocab[:, 0].view(-1)
    mask2 = second_dim_first_pos == 1
    align_probs = relm_prob_distributes[indices, align_vocab]
    align_probs = align_probs * mask  # 过滤
    align_probs = align_probs.sum(dim=-1)
    align_probs[mask2] = -15
    return align_probs


def get_distortion_probs(
    self, batch_observed_sequences: List[List[str]], eos_token_id: int
) -> Tuple[List[int], List[int], List[int], List[float]]:
    distortion_probs = []
    cache = self.cache
    batch_indices = []
    beam_indices = []
    token_indices = []
    for batch_index, observed_sequences in enumerate(batch_observed_sequences):
        for beam_index, observed_sequence in enumerate(observed_sequences):
            if observed_sequence in cache:
                indices, weight = cache[observed_sequence]
            else:
                if len(observed_sequence) != 0:
                    indices, weight = self.token_transformation_to_probs(
                        observed_sequence
                    )
                else:
                    if isinstance(eos_token_id, list):
                        indices = eos_token_id
                        weight = [0.0] * len(eos_token_id)
                    else:
                        indices = [eos_token_id]
                        weight = [0.0]
                cache[observed_sequence] = indices, weight
            batch_indices.extend([batch_index] * len(indices))
            beam_indices.extend([beam_index] * len(indices))
            token_indices.extend(indices)
            distortion_probs.extend(weight)
    return batch_indices, beam_indices, token_indices, distortion_probs


def get_small_model_probs(
    self, batch_observed_sequences: List[List[str]], steps: List[List[int]], eos_token_id: int, sentence_id: List[int],
        prob_distribute, align_vocab
) -> Tuple[List[int], List[int], List[int], List[float]]:
    cache = self.align_vocab_cache
    batch_beam_probs = torch.zeros(len(steps), len(steps[0]), align_vocab.shape[0]).cuda()
    eos_probs = torch.ones(align_vocab.shape[0]).cuda() * -15
    for batch_index, step_sequences in enumerate(steps):
        for beam_index, step in enumerate(step_sequences):
            # if (sentence_id + batch_index, step) in cache:
            if (sentence_id[batch_index], step) in cache:
                batch_beam_probs[batch_index, beam_index] = cache[(sentence_id[batch_index], step)]
            else:
                if len(batch_observed_sequences[batch_index][beam_index]) != 0:
                    batch_beam_probs[batch_index, beam_index] = self.small_model_to_probs(sentence_id[batch_index], step,
                                                                batch_observed_sequences[batch_index][beam_index],
                                                                prob_distribute[batch_index], align_vocab)
                else:
                    tmp = eos_probs.clone()
                    if isinstance(eos_token_id, list):
                        for indice in eos_token_id:
                            tmp[indice] = 0.0
                    else:
                        indice = eos_token_id
                        tmp[indice] = 0.0
                    batch_beam_probs[batch_index, beam_index] = tmp
                cache[(sentence_id[batch_index], step)] = batch_beam_probs[batch_index, beam_index]

    return batch_beam_probs.view(len(steps) * len(steps[0]), align_vocab.shape[0])


@torch.jit.script
def distortion_probs_to_cuda(
    template_tensor: torch.Tensor, 
    batch_size: int, 
    num_beams: int, 
    batch_beam_size: int, 
    vocab_size: int, 
    _batch_indices: List[int], 
    _beam_indices: List[int], 
    _token_indices: List[int], 
    _distortion_probs: List[float]) -> torch.Tensor:

    distortion_probs = template_tensor.repeat(batch_size, num_beams, 1)
    distortion_probs[_batch_indices, _beam_indices, _token_indices] = torch.tensor(
        _distortion_probs, device=template_tensor.device
    )

    return distortion_probs.view(batch_beam_size, vocab_size)

def distortion_guided_beam_search(
    self,
    observed_sequence_generator: BaseObversationGenerator,
    beam_scorer: BeamScorer,
    sentence_id: List[int] = None,
    prob_distribute = None,
    align_vocab = None,
    input_ids: torch.LongTensor = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GenerateBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    <Tip warning={true}>

    In most cases, you do not need to call [`~generation.GenerationMixin.beam_search`] directly. Use generate()
    instead. For an overview of generation strategies and code examples, check the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        observed_sequence_generator (`BaseObversationGenerator`):
            An instance of [`BaseObversationGenerator`] that defines how observed sequences are generated.
        beam_scorer (`BeamScorer`):
            An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
            sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`generation.GenerateBeamDecoderOnlyOutput`], [`~generation.GenerateBeamEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateBeamEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.


    Examples:

    ```python
    >>> from transformers import (
    ...     AutoTokenizer,
    ...     AutoModelForSeq2SeqLM,
    ...     LogitsProcessorList,
    ...     MinLengthLogitsProcessor,
    ...     BeamSearchScorer,
    ... )
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

    >>> encoder_input_str = "translate English to German: How old are you?"
    >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


    >>> # lets run beam search using 3 beams
    >>> num_beams = 3
    >>> # define decoder start token ids
    >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
    >>> input_ids = input_ids * model.config.decoder_start_token_id

    >>> # add encoder_outputs to model keyword arguments
    >>> model_kwargs = {
    ...     "encoder_outputs": model.get_encoder()(
    ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
    ...     )
    ... }

    >>> # instantiate beam scorer
    >>> beam_scorer = BeamSearchScorer(
    ...     batch_size=1,
    ...     num_beams=num_beams,
    ...     device=model.device,
    ... )

    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList(
    ...     [
    ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
    ...     ]
    ... )

    >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ['Wie alt bist du?']
    ```"""
    # init values
    logits_processor = (
        logits_processor if logits_processor is not None else LogitsProcessorList()
    )
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = (
        pad_token_id
        if pad_token_id is not None
        else self.generation_config.pad_token_id
    )
    eos_token_id = (
        eos_token_id
        if eos_token_id is not None
        else self.generation_config.eos_token_id
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    output_scores = (
        output_scores
        if output_scores is not None
        else self.generation_config.output_scores
    )
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )
    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    ## Modification 0:
    ## Initialization

    if input_ids is None:
        # In this case, we don't provide prompt or context, so we need to generate the first token
        input_ids = torch.ones((batch_size, 1), device=self.device, dtype=torch.long)
        input_ids = input_ids * self.config.decoder_start_token_id
    input_ids, model_kwargs = self._expand_inputs_for_generation(
        input_ids=input_ids, expand_size=num_beams, **model_kwargs
    )

    try:
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
    except:
        pass
    vocab_size = self.vocab_size

    # template for the distortion model
    template_weight = self.probs_template * self.distortion_model_smoothing
    template_weight[self.token_length > 1] = MIN
    small_model_template_weight = self.probs_template * 0.0

    # clear the cache
    self.cache = {}
    self.align_vocab_cache = {}
    self.beam_cache = {}
    self.cached_observed_sequences = []
    self.max_cached_observed_sequences = num_beams * batch_size

    ## END of modification

    batch_beam_size, cur_len = input_ids.shape
    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    beam_indices = (
        tuple(() for _ in range(batch_beam_size))
        if (return_dict_in_generate and output_scores)
        else None
    )
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = (
            model_kwargs["encoder_outputs"].get("attentions")
            if output_attentions
            else None
        )
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states")
            if output_hidden_states
            else None
        )
    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.
    beam_scores = torch.zeros(
        (batch_size, num_beams), dtype=torch.float, device=input_ids.device
    )
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))
    this_peer_finished = False  # used by synced_gpus only

    decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(
                0.0 if this_peer_finished else 1.0
            ).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        # outputs.logits.shape: [bz * n_beam, 1, vocab]

        ## Modification 1.0:
        observed_sequences = observed_sequence_generator.get_observed_sequences()
        true_steps = observed_sequence_generator.get_true_steps()
        steps = observed_sequence_generator.get_steps()
        _batch_indices, _beam_indices, _token_indices, _distortion_probs = (
            self.get_distortion_probs(observed_sequences, eos_token_id)
        )
        ## END of modification

        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        next_token_scores = nn.functional.log_softmax(
            next_token_logits, dim=-1
        )  # (batch_size * num_beams, vocab_size)

        ## Modification 1.1:
        ## Intervention of decoding process
        # get the observed sequences and calculate the distortion probs
        distortion_probs = distortion_probs_to_cuda(
            template_weight,
            batch_size,
            num_beams,
            batch_beam_size,
            vocab_size,
            _batch_indices,
            _beam_indices,
            _token_indices,
            _distortion_probs
        )

        if self.is_bytes_level:
            small_model_probs = self.get_small_model_probs(observed_sequences, true_steps, eos_token_id, sentence_id, prob_distribute, align_vocab)
        else:
            small_model_probs = self.get_small_model_probs(observed_sequences, steps, eos_token_id, sentence_id, prob_distribute, align_vocab)

        # faithfulness reward
        faithfulness_coefficient = 1.0
        if self.use_faithfulness_reward:
            entropy = -torch.sum(
                next_token_scores * torch.exp(next_token_scores), dim=-1, keepdim=True
            )
            entropy = entropy / self.max_entropy
            faithfulness_coefficient = 1.0 + entropy * 1.0

        length_reward = 2.5 * (self.token_length[None] - 1).clamp(min=0.0)
        
        next_token_scores = next_token_scores + faithfulness_coefficient * (
                self.alpha * small_model_probs + self.beta * distortion_probs
        )
        
        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + beam_scores[
            :, None
        ].expand_as(next_token_scores_processed)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores_processed,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,)
                    if self.config.is_encoder_decoder
                    else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        n_eos_tokens = len(eos_token_id) if eos_token_id else 0
        next_token_scores, next_tokens = torch.topk(
            next_token_scores,
            max(2, 1 + n_eos_tokens) * num_beams,
            dim=1,
            largest=True,
            sorted=True,
        )

        # # 这个indices 是在判断tokens属于第几个indices,也就是是beams中第几个candidate延伸出来的结果.
        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        # # 这个tokens 是在判断tokens属于第next_indices个beam的第几个token.
        next_tokens = next_tokens % vocab_size

        # stateless
        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        input_ids = torch.cat(
            [input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1
        )

        ## Modification 2:
        ## Update the observed sequences
        observed_sequence_generator.reorder(
            (beam_idx % num_beams).view(batch_size, num_beams)
        )
        predicted_tokens = [
            self.convert_ids_to_tokens(t)
            for t in beam_next_tokens.view(batch_size, num_beams).tolist()
        ]
        observed_sequence_generator.step(predicted_tokens)
        if streamer is not None:
            streamer.put((beam_scorer, input_ids.cpu()))
        # If you want to what's happening in the decoding process, you can uncomment the following line:
        # observed_sequence_generator.show_steps()
        # print()
        ## END of modification
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        if model_kwargs["past_key_values"] is not None:
            model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                model_kwargs["past_key_values"], beam_idx
            )

        if return_dict_in_generate and output_scores:
            beam_indices = tuple(
                (
                    beam_indices[beam_idx[i]] + (beam_idx[i],)
                    for i in range(len(beam_indices))
                )
            )

        # increase cur_len
        cur_len = cur_len + 1

        ## Modification 3:
        ## Remove stopping_criteria
        if beam_scorer.is_done:
            if not synced_gpus:
                break
            else:
                this_peer_finished = True
        ## END of modification

    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=beam_indices,
        decoder_prompt_len=decoder_prompt_len,
    )

    ## Modification 4:
    if streamer is not None:
        streamer.put((beam_scorer, input_ids.cpu()))
        streamer.end()
    ## END of modification

    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None

        if self.config.is_encoder_decoder:
            return GenerateBeamEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateBeamDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return sequence_outputs["sequences"]
