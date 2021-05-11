import os
import re
import torch
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers import PreTrainedModel, PretrainedConfig, BertPreTrainedModel
from transformers.file_utils import (TF_WEIGHTS_NAME, TF2_WEIGHTS_NAME, WEIGHTS_NAME)
from transformers.file_utils import (is_remote_url, hf_bucket_url, cached_path, is_torch_tpu_available)
from transformers.modeling_bert import (
  BertEmbeddings,
  BertLayer,
  BertOnlyMLMHead,
  ModelOutput,
  BaseModelOutput,
  MaskedLMOutput,
  SequenceClassifierOutput,
  TokenClassifierOutput,
  load_tf_weights_in_bert,
)
from transformers.utils import logging
from configuration_kebio import KebioConfig

logger = logging.get_logger(__name__)


class KGMaskedLMOutput(MaskedLMOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    mention_detection_loss: Optional[torch.FloatTensor] = None
    entity_linking_loss: Optional[torch.FloatTensor] = None
    mlm_loss: Optional[torch.FloatTensor] = None


@dataclass
class KebioModelOutput(ModelOutput):
  last_hidden_state: torch.FloatTensor
  entity_logits: torch.FloatTensor
  mention_detection_logits: torch.FloatTensor
  hidden_states: Optional[Tuple[torch.FloatTensor]] = None
  attentions: Optional[Tuple[torch.FloatTensor]] = None


class KebioContextEncoder(torch.nn.Module):
  def __init__(self, config: KebioConfig):
    super().__init__()
    self.config = config
    self.layer = torch.nn.ModuleList([
      BertLayer(config) for _ in range(config.num_context_layers)])

  def forward(
      self,
      hidden_states,
      attention_mask=None,
      head_mask=None,
      encoder_hidden_states=None,
      encoder_attention_mask=None,
      output_attentions=False,
      output_hidden_states=False,
      return_dict=False,
  ):
    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None
    for i, layer_module in enumerate(self.layer):
      if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

      layer_head_mask = head_mask[i] if head_mask is not None else None

      if getattr(self.config, "gradient_checkpointing", False):
        def create_custom_forward(module):
          def custom_forward(*inputs):
            return module(*inputs, output_attentions)

          return custom_forward

        layer_outputs = torch.utils.checkpoint.checkpoint(
          create_custom_forward(layer_module),
          hidden_states,
          attention_mask,
          layer_head_mask,
          encoder_hidden_states,
          encoder_attention_mask,
        )
      else:
        layer_outputs = layer_module(
          hidden_states,
          attention_mask,
          layer_head_mask,
          encoder_hidden_states,
          encoder_attention_mask,
          output_attentions,
        )
      hidden_states = layer_outputs[0]
      if output_attentions:
        all_attentions = all_attentions + (layer_outputs[1],)

    if output_hidden_states:
      all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
      return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
    return BaseModelOutput(
      last_hidden_state=hidden_states,
      hidden_states=all_hidden_states,
      attentions=all_attentions
    )


class KebioContextEntityEncoder(torch.nn.Module):
  def __init__(self, config: KebioConfig,):
    super().__init__()
    self.config = config
    self.layer = torch.nn.ModuleList([
      BertLayer(config)
      for _ in range(config.num_hidden_layers - config.num_context_layers)])

  def forward(
      self,
      hidden_states,
      attention_mask=None,
      head_mask=None,
      encoder_hidden_states=None,
      encoder_attention_mask=None,
      output_attentions=False,
      output_hidden_states=False,
      return_dict=False,
  ):
    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    for i, layer_module in enumerate(self.layer):
      if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

      layer_head_mask = head_mask[i] if head_mask is not None else None

      if getattr(self.config, "gradient_checkpointing", False):
        def create_custom_forward(module):
          def custom_forward(*inputs):
            return module(*inputs, output_attentions)

          return custom_forward

        layer_outputs = torch.utils.checkpoint.checkpoint(
          create_custom_forward(layer_module),
          hidden_states,
          attention_mask,
          layer_head_mask,
          encoder_hidden_states,
          encoder_attention_mask,
        )
      else:
        layer_outputs = layer_module(
          hidden_states,
          attention_mask,
          layer_head_mask,
          encoder_hidden_states,
          encoder_attention_mask,
          output_attentions,
        )
      hidden_states = layer_outputs[0]
      if output_attentions:
        all_attentions = all_attentions + (layer_outputs[1],)

    if output_hidden_states:
      all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
      return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
    return BaseModelOutput(
      last_hidden_state=hidden_states,
      hidden_states=all_hidden_states,
      attentions=all_attentions
    )


class KebioLinker(torch.nn.Module):
  def __init__(self, config: KebioConfig):
    super().__init__()
    self.num_entities = config.num_entities
    self.entity_embeddings = torch.nn.Linear(in_features=config.entity_size,
                                             out_features=config.num_entities,
                                             bias=False)

    self.mention_to_entity_projection = torch.nn.Linear(in_features=config.hidden_size * 2,
                                                        out_features=config.entity_size)

  def forward(self, hidden_states: torch.Tensor):
    batch_size, max_mentions, mention_size = hidden_states.shape
    hidden_states = self.mention_to_entity_projection(hidden_states)
    hidden_states = hidden_states.view(batch_size * max_mentions, -1)
    hidden_states = self.entity_embeddings(hidden_states)
    hidden_states = hidden_states.view(batch_size, max_mentions, self.num_entities)
    return hidden_states


class KebioModel(BertPreTrainedModel):
  def __init__(self, config: KebioConfig):
    super().__init__(config)
    self.config = config
    # context encoder
    self.embeddings = BertEmbeddings(config)
    self.context_encoder = KebioContextEncoder(config)

    # mention detector
    self.num_labels = 3
    self.mention_detector = torch.nn.Linear(in_features=config.hidden_size,
                                            out_features=3)

    #
    self.entity_linker = KebioLinker(config)
    self.entity_context_projection = torch.nn.Linear(in_features=config.entity_size,
                                                     out_features=config.hidden_size)

    self.layer_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    #
    self.recontext_encoder = KebioContextEntityEncoder(config)

    self.init_weights()

  def get_input_embeddings(self):
    return self.embeddings.word_embeddings

  def set_input_embeddings(self, value):
    self.embeddings.word_embeddings = value

  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      token_type_ids=None,
      position_ids=None,
      mention_detection_labels=None,
      head_mask=None,
      inputs_embeds=None,
      encoder_hidden_states=None,
      encoder_attention_mask=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=None,
  ):
    # The encoder_hidden_states and encoder_attention_mask are for text generation.
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
      output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
      raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
      input_shape = input_ids.size()
    elif inputs_embeds is not None:
      input_shape = inputs_embeds.size()[:-1]
    else:
      raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if attention_mask is None:
      attention_mask = torch.ones(input_shape, device=device)
    if token_type_ids is None:
      token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if self.config.is_decoder and encoder_hidden_states is not None:
      encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
      encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
      if encoder_attention_mask is None:
        encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
      encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    else:
      encoder_extended_attention_mask = None

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
    # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

    embedding_output = self.embeddings(
      input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
    )
    context_encoder_outputs = self.context_encoder(
      embedding_output,
      attention_mask=extended_attention_mask,
      head_mask=head_mask,
      encoder_hidden_states=encoder_hidden_states,
      encoder_attention_mask=encoder_extended_attention_mask,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
    )
    context_sequence_output = context_encoder_outputs[0]

    # Do mention detection
    mention_detection_logits = self.mention_detector(context_sequence_output)
    if mention_detection_labels is None:
      mention_detection_labels = torch.argmax(mention_detection_logits, dim=-1)

    mention_detection_labels = mention_detection_labels.cpu().numpy()
    lengths = torch.sum(input_ids != self.config.pad_token_id, dim=-1, dtype=torch.long).tolist()

    batch_spans = []
    for bid, labels in enumerate(mention_detection_labels):
      result_starts, result_ends = [], []
      prev_label = None
      for position in range(1, lengths[bid]):
        label = labels[position]
        if label == 1 or (label == 2 and (not prev_label or prev_label == 0)):
          result_starts.append(position)
          result_ends.append(position + 1)
        elif label == 2:
          if len(result_starts) == 0:
            result_starts.append(position)
            result_ends.append(position)
          result_ends[-1] = position + 1
        prev_label = label
      spans = [(result_start, result_end) for result_start, result_end in zip(result_starts, result_ends)]
      batch_spans.append(spans)

    max_mentions = max([len(spans) for spans in batch_spans])
    entity_states = torch.zeros_like(context_sequence_output)

    if max_mentions > 0:
      if max_mentions > self.config.max_mentions:
        max_mentions = self.config.max_mentions

      for i in range(len(batch_spans)):
        if len(batch_spans[i]) > max_mentions:
          batch_spans[i] = batch_spans[i][:max_mentions]
        else:
          while len(batch_spans[i]) < max_mentions:
            batch_spans[i].append((0, 1))

      batch_size, seq_length, hidden_size = context_sequence_output.shape
      batch_span_offsets = torch.arange(
        0, batch_size * seq_length, seq_length, dtype=torch.long).view(batch_size, 1).repeat(1, max_mentions)

      batch_span_start_offsets = torch.tensor(
        [[span[0] for span in spans] for i, spans in enumerate(batch_spans)], dtype=torch.long) + batch_span_offsets
      batch_span_end_offsets = torch.tensor(
        [[span[1] - 1 for span in spans] for i, spans in enumerate(batch_spans)], dtype=torch.long) + batch_span_offsets

      flat_context_sequence_output = context_sequence_output.view(batch_size * seq_length, -1)
      span_head_states = flat_context_sequence_output[batch_span_start_offsets.view(-1)]
      span_tail_states = flat_context_sequence_output[batch_span_end_offsets.view(-1)]

      mention_context_states = torch.cat([span_head_states, span_tail_states], dim=1).view(batch_size, max_mentions, -1)

      entity_logits = self.entity_linker.forward(mention_context_states)

      topk_logits, topk_indices = torch.topk(entity_logits,
                                             min(self.config.max_candidate_entities, self.config.num_entities),
                                             dim=-1)
      a = torch.nn.Softmax(dim=-1)(topk_logits)
      batch_size, max_mentions, depth = a.shape

      flat_topk_indices = topk_indices.view(-1)
      entity_embeddings = torch.index_select(self.entity_linker.entity_embeddings.weight, dim=0, index=flat_topk_indices)
      entity_embeddings = entity_embeddings.view(batch_size, max_mentions, depth, -1)
      entity_embeddings = torch.sum(a.unsqueeze(-1) * entity_embeddings, dim=-2)

      for i in range(len(batch_spans)):
        for j, (start, end) in enumerate(batch_spans[i]):
          entity_states[i, start: end + 1, :] = self.entity_context_projection(entity_embeddings[i, j, :])
    else:
      entity_logits = None

    context_sequence_output = self.layer_norm(context_sequence_output + entity_states)
    recontext_encoder_outputs = self.recontext_encoder(
      context_sequence_output,
      attention_mask=extended_attention_mask,
      head_mask=head_mask,
      encoder_hidden_states=encoder_hidden_states,
      encoder_attention_mask=encoder_extended_attention_mask,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,)

    recontext_sequence_output = recontext_encoder_outputs[0]

    if not return_dict:
      return (mention_detection_logits, entity_logits, recontext_sequence_output, ) + \
             recontext_encoder_outputs[1:] + context_encoder_outputs

    return KebioModelOutput(
      entity_logits=entity_logits,
      last_hidden_state=recontext_sequence_output,
      mention_detection_logits=mention_detection_logits,
      hidden_states=(recontext_encoder_outputs.hidden_states + context_encoder_outputs.hidden_states),
      attentions=(recontext_encoder_outputs.attentions + context_sequence_output.attentions)
    )


class KebioPreTrainedModel(PreTrainedModel):
  config_class = KebioConfig
  load_tf_weights = load_tf_weights_in_bert
  base_model_prefix = "bert"
  authorized_missing_keys = [r"position_ids"]

  def _init_weights(self, module):
    """ Initialize the weights """
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
      # Slightly different from the TF version which uses truncated_normal for initialization
      # cf https://github.com/pytorch/pytorch/pull/5617
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    elif isinstance(module, torch.nn.LayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)
    if isinstance(module, torch.nn.Linear) and module.bias is not None:
      module.bias.data.zero_()


class KebioForPreTraining(KebioPreTrainedModel):
  def __init__(self, config: KebioConfig):
    super().__init__(config)

    self.bert = KebioModel(config)
    self.cls = BertOnlyMLMHead(config)

    self.init_weights()

  def get_output_embeddings(self):
    # NOTE: this is needed to resize the embeddings!
    return self.cls.predictions.decoder

  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      token_type_ids=None,
      position_ids=None,
      mention_detection_labels=None,
      gold_entity_ids=None,
      head_mask=None,
      inputs_embeds=None,
      mlm_labels=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=None,
      **kwargs
  ):
    if "masked_lm_labels" in kwargs:
      warnings.warn(
        "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
        FutureWarning,
      )
      mlm_labels = kwargs.pop("masked_lm_labels")
    assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.bert(
      input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      position_ids=position_ids,
      mention_detection_labels=mention_detection_labels,
      head_mask=head_mask,
      inputs_embeds=inputs_embeds,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
    )
    mention_detection_logits, entity_logits, recontext_sequence_output = outputs[:3]
    
    mention_detection_loss = None
    entity_linking_loss = None
    masked_lm_loss = None

    total_loss = None
    if mention_detection_labels is not None:

      loss_fct = torch.nn.CrossEntropyLoss()
      # Only keep active parts of the loss
      if attention_mask is not None:
        active_loss = attention_mask.view(-1) == 1
        active_logits = mention_detection_logits.view(-1, 3)
        active_labels = torch.where(
          active_loss, mention_detection_labels.view(-1),
          torch.tensor(loss_fct.ignore_index).type_as(mention_detection_labels)
        )
        mention_detection_loss = loss_fct(active_logits, active_labels)
        total_loss = mention_detection_loss
      else:
        mention_detection_loss = loss_fct(mention_detection_logits.view(-1, self.config.num_labels),
                              mention_detection_labels.view(-1))
        total_loss = mention_detection_loss

    if gold_entity_ids is not None and entity_logits is not None:
      num_mentions = gold_entity_ids.shape[1]
      if num_mentions > self.config.max_mentions:
        gold_entity_ids = gold_entity_ids[:, :self.config.max_mentions].contiguous()

      loss_fct = torch.nn.CrossEntropyLoss()
      
      entity_linking_loss = loss_fct(entity_logits.view(-1, self.config.num_entities),
                                     gold_entity_ids.view(-1))
      if total_loss is None:
        total_loss = entity_linking_loss
      else:
        total_loss = total_loss + entity_linking_loss
    else:
      entity_linking_loss = None

    prediction_scores = self.cls(recontext_sequence_output)

    if mlm_labels is not None:
      loss_fct = torch.nn.CrossEntropyLoss()
      masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
      if total_loss is None:
        total_loss = masked_lm_loss
      else:
        total_loss += masked_lm_loss
    else:
      masked_lm_loss = None

    if not return_dict:
      output = (prediction_scores, ) + outputs[2:] + (mention_detection_loss, entity_linking_loss, masked_lm_loss)
      return ((total_loss,) + output) if total_loss is not None else output

    return KGMaskedLMOutput(
      loss=total_loss,
      logits=prediction_scores,
      hidden_states=outputs.hidden_states,
      attentions=outputs.attentions,
      mention_detection_loss=mention_detection_loss,
      entity_linking_loss=entity_linking_loss,
      mlm_loss=masked_lm_loss,
    )

  @classmethod
  def from_bert_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    config = kwargs.pop("config", None)
    state_dict = kwargs.pop("state_dict", None)
    cache_dir = kwargs.pop("cache_dir", None)
    from_tf = kwargs.pop("from_tf", False)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    output_loading_info = kwargs.pop("output_loading_info", False)
    local_files_only = kwargs.pop("local_files_only", False)
    use_cdn = kwargs.pop("use_cdn", True)
    mirror = kwargs.pop("mirror", None)

    # Load config if we don't provide a configuration
    if not isinstance(config, PretrainedConfig):
      config_path = config if config is not None else pretrained_model_name_or_path
      config, model_kwargs = cls.config_class.from_pretrained(
        config_path,
        *model_args,
        cache_dir=cache_dir,
        return_unused_kwargs=True,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        local_files_only=local_files_only,
        **kwargs,
      )
    else:
      model_kwargs = kwargs

    # Load model
    if pretrained_model_name_or_path is not None:
      if os.path.isdir(pretrained_model_name_or_path):
        if from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")):
          # Load from a TF 1.0 checkpoint
          archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
        elif from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
          # Load from a TF 2.0 checkpoint
          archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
        elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
          # Load from a PyTorch checkpoint
          archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
        else:
          raise EnvironmentError(
            "Error no file named {} found in directory {} or `from_tf` set to False".format(
              [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME + ".index"],
              pretrained_model_name_or_path,
            )
          )
      elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
        archive_file = pretrained_model_name_or_path
      elif os.path.isfile(pretrained_model_name_or_path + ".index"):
        assert (
          from_tf
        ), "We found a TensorFlow checkpoint at {}, please set from_tf to True to load from this checkpoint".format(
          pretrained_model_name_or_path + ".index"
        )
        archive_file = pretrained_model_name_or_path + ".index"
      else:
        archive_file = hf_bucket_url(
          pretrained_model_name_or_path,
          filename=(TF2_WEIGHTS_NAME if from_tf else WEIGHTS_NAME),
          use_cdn=use_cdn,
          mirror=mirror,
        )

      try:
        # Load from URL or cache if already cached
        resolved_archive_file = cached_path(
          archive_file,
          cache_dir=cache_dir,
          force_download=force_download,
          proxies=proxies,
          resume_download=resume_download,
          local_files_only=local_files_only,
        )
        if resolved_archive_file is None:
          raise EnvironmentError
      except EnvironmentError:
        msg = (
          f"Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
          f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
          f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named one of {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME}.\n\n"
        )
        raise EnvironmentError(msg)

      if resolved_archive_file == archive_file:
        logger.info("loading weights file {}".format(archive_file))
      else:
        logger.info("loading weights file {} from cache at {}".format(archive_file, resolved_archive_file))
    else:
      resolved_archive_file = None

    # Instantiate model.
    model = cls(config, *model_args, **model_kwargs)

    if state_dict is None and not from_tf:
      try:
        state_dict = torch.load(resolved_archive_file, map_location="cpu")
      except Exception:
        raise OSError(
          "Unable to load weights from pytorch checkpoint file. "
          "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. "
        )

    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    if from_tf:
      if resolved_archive_file.endswith(".index"):
        # Load from a TensorFlow 1.X checkpoint - provided by original authors
        model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
      else:
        # Load from our TensorFlow 2.0 checkpoints
        try:
          from .modeling_tf_pytorch_utils import load_tf2_checkpoint_in_pytorch_model

          model = load_tf2_checkpoint_in_pytorch_model(model, resolved_archive_file, allow_missing_keys=True)
        except ImportError:
          logger.error(
            "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
            "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
          )
          raise
    else:
      # Convert old format to new format if needed from a PyTorch state_dict
      old_keys = []
      new_keys = []
      for key in state_dict.keys():
        new_key = None
        if "gamma" in key:
          new_key = key.replace("gamma", "weight")
        if "beta" in key:
          new_key = key.replace("beta", "bias")
        if new_key:
          old_keys.append(key)
          new_keys.append(new_key)
      for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

      # copy state_dict so _load_from_state_dict can modify it
      metadata = getattr(state_dict, "_metadata", None)
      state_dict = state_dict.copy()
      if metadata is not None:
        state_dict._metadata = metadata

      # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
      # so we need to apply the function recursively.
      def load(module: torch.nn.Module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
          state_dict,
          prefix,
          local_metadata,
          True,
          missing_keys,
          unexpected_keys,
          error_msgs,
        )
        for name, child in module._modules.items():
          if child is not None:
            load(child, prefix + name + ".")

      # Make sure we are able to load base models as well as derived models (with heads)
      start_prefix = ""
      model_to_load = model
      has_prefix_module = any(s.startswith(cls.base_model_prefix) for s in state_dict.keys())
      if not hasattr(model, cls.base_model_prefix) and has_prefix_module:
        start_prefix = cls.base_model_prefix + "."
      if hasattr(model, cls.base_model_prefix) and not has_prefix_module:
        model_to_load = getattr(model, cls.base_model_prefix)

      #
      key_changes = []
      for key in state_dict.keys():
        if key.startswith('bert.encoder.layer.'):
          n_layer = int(key.split('.')[3])
          if n_layer < config.num_context_layers:
            new_key = key.replace('.encoder.', '.context_encoder.')
          else:
            new_key = key.replace('.encoder.layer.{}'.format(n_layer),
                                  '.recontext_encoder.layer.{}'.format(n_layer - config.num_context_layers))
          logger.info('state_dict mapping: {:60s} -> {:70s} {}'.format(key, new_key, list(state_dict[key].shape)))
          key_changes.append((key, new_key))

      for key, new_key in key_changes:
        state_dict[new_key] = state_dict.pop(key)

      load(model_to_load, prefix=start_prefix)

      if model.__class__.__name__ != model_to_load.__class__.__name__:
        base_model_state_dict = model_to_load.state_dict().keys()
        head_model_state_dict_without_base_prefix = [
          key.split(cls.base_model_prefix + ".")[-1] for key in model.state_dict().keys()
        ]
        missing_keys.extend(head_model_state_dict_without_base_prefix - base_model_state_dict)

      # Some models may have keys that are not in the state by design, removing them before needlessly warning
      # the user.
      if cls.authorized_missing_keys is not None:
        for pat in cls.authorized_missing_keys:
          missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

      if cls.authorized_unexpected_keys is not None:
        for pat in cls.authorized_unexpected_keys:
          unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

      if len(unexpected_keys) > 0:
        logger.warning(
          f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when "
          f"initializing {model.__class__.__name__}: {unexpected_keys}\n"
          f"- This IS expected if you are initializing {model.__class__.__name__} from the checkpoint of a model trained on another task "
          f"or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPretraining model).\n"
          f"- This IS NOT expected if you are initializing {model.__class__.__name__} from the checkpoint of a model that you expect "
          f"to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
        )
      else:
        logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
      if len(missing_keys) > 0:
        logger.warning(
          f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} "
          f"and are newly initialized: {missing_keys}\n"
          f"You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
        )
      else:
        logger.info(
          f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at {pretrained_model_name_or_path}.\n"
          f"If your task is similar to the task the model of the checkpoint was trained on, "
          f"you can already use {model.__class__.__name__} for predictions without further training."
        )
      if len(error_msgs) > 0:
        raise RuntimeError(
          "Error(s) in loading state_dict for {}:\n\t{}".format(
            model.__class__.__name__, "\n\t".join(error_msgs)
          )
        )
    # make sure token embedding weights are still tied if needed
    model.tie_weights()

    # Set model in evaluation mode to deactivate DropOut modules by default
    model.eval()

    if output_loading_info:
      loading_info = {
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "error_msgs": error_msgs,
      }
      return model, loading_info

    if hasattr(config, "xla_device") and config.xla_device and is_torch_tpu_available():
      import torch_xla.core.xla_model as xm

      model = xm.send_cpu_data_to_device(model, xm.xla_device())
      model.to(xm.xla_device())

    return model


class KebioForSequenceClassification(BertPreTrainedModel):
  def __init__(self, config: KebioConfig):
    super().__init__(config)
    self.num_labels = config.num_labels

    self.bert = KebioModel(config)
    self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
    self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    self.init_weights()

  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      token_type_ids=None,
      position_ids=None,
      head_mask=None,
      inputs_embeds=None,
      labels=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=None,
  ):
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.bert(
      input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      position_ids=position_ids,
      head_mask=head_mask,
      inputs_embeds=inputs_embeds,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
    )

    sequence_output = outputs[2]
    pooled_output = sequence_output[:, 0]

    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)

    loss = None
    if labels is not None:
      if self.num_labels == 1:
        #  We are doing regression
        loss_fct = torch.nn.MSELoss()
        loss = loss_fct(logits.view(-1), labels.view(-1))
      else:
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

    if not return_dict:
      output = (logits,) + outputs[2:]
      return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
      loss=loss,
      logits=logits,
      hidden_states=outputs.hidden_states,
      attentions=outputs.attentions,
    )


class KebioForRelationExtraction(BertPreTrainedModel):

  authorized_unexpected_keys = [r"pooler"]

  def __init__(self, config: KebioConfig):
    super().__init__(config)
    self.num_labels = config.num_labels

    self.bert = KebioModel(config)
    self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
    self.classifier = torch.nn.Linear(config.hidden_size * 2, config.num_labels)

    self.init_weights()

  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      token_type_ids=None,
      position_ids=None,
      first_entity_position=None,
      second_entity_position=None,
      head_mask=None,
      inputs_embeds=None,
      labels=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=None,
  ):
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.bert(
      input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      position_ids=position_ids,
      head_mask=head_mask,
      inputs_embeds=inputs_embeds,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
    )
    sequence_output = outputs[2]
    batch_size = sequence_output.shape[0]
    pooled_output = torch.cat(
      [sequence_output[torch.arange(batch_size), first_entity_position, :],
       sequence_output[torch.arange(batch_size), second_entity_position, :]], dim=1)

    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)

    loss = None
    if labels is not None:
      if self.num_labels == 1:
        #  We are doing regression
        loss_fct = torch.nn.MSELoss()
        loss = loss_fct(logits.view(-1), labels.view(-1))
      else:
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

    if not return_dict:
      output = (logits,) + outputs[2:]
      return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
      loss=loss,
      logits=logits,
      hidden_states=outputs.hidden_states,
      attentions=outputs.attentions,
    )


class KebioForTokenClassification(BertPreTrainedModel):

  authorized_unexpected_keys = [r"pooler"]

  def __init__(self, config):
    super().__init__(config)
    self.num_labels = config.num_labels

    self.bert = KebioModel(config)
    self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
    self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    self.init_weights()

  def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
      r"""
      labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
          Labels for computing the token classification loss.
          Indices should be in ``[0, ..., config.num_labels - 1]``.
      """
      return_dict = return_dict if return_dict is not None else self.config.use_return_dict

      outputs = self.bert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
      )

      sequence_output = outputs[2]

      sequence_output = self.dropout(sequence_output)
      logits = self.classifier(sequence_output)

      loss = None
      if labels is not None:
        loss_fct = torch.nn.CrossEntropyLoss()
        # Only keep active parts of the loss
        if attention_mask is not None:
          active_loss = attention_mask.view(-1) == 1
          active_logits = logits.view(-1, self.num_labels)
          active_labels = torch.where(
            active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
          )
          loss = loss_fct(active_logits, active_labels)
        else:
          loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

      if not return_dict:
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

      return TokenClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
      )
