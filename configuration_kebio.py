from transformers import BertConfig


class KebioConfig(BertConfig):
  """Configuration for `KebioModel`."""

  def __init__(self,
               vocab_size,
               num_entities,
               max_mentions=15,
               max_candidate_entities=100,
               hidden_size=768,
               entity_size=50,
               num_hidden_layers=12,
               num_context_layers=8,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02, **kwargs):
    super(KebioConfig, self).__init__(vocab_size=vocab_size,
                                      hidden_size=hidden_size,
                                      num_hidden_layers=num_hidden_layers,
                                      num_attention_heads=num_attention_heads,
                                      intermediate_size=intermediate_size,
                                      hidden_act=hidden_act,
                                      hidden_dropout_prob=hidden_dropout_prob,
                                      attention_probs_dropout_prob=attention_probs_dropout_prob,
                                      max_position_embeddings=max_position_embeddings,
                                      type_vocab_size=type_vocab_size,
                                      initializer_range=initializer_range, **kwargs)
    self.num_context_layers = num_context_layers
    self.entity_size = entity_size
    self.num_entities = num_entities
    self.max_mentions = max_mentions
    self.max_candidate_entities = max_candidate_entities
