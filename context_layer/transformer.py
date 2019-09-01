import torch
from pytorch_transformers.modeling_bert import BertEncoder

import utils.io as io


class ContextLayerConstants(io.JsonSerializableClass):
    """
        Arguments:
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
    """
    def __init__(self):
        super().__init__()
        self.hidden_size = 768
        self.num_hidden_layers = 3
        self.num_attention_heads = 12
        self.hidden_act = "gelu"
        self.intermediate_size = 768
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.initializer_range = 0.02
        self.layer_norm_eps = 1e-12
        self.output_attentions = False
        self.output_hidden_states = False


class ContextLayer(BertEncoder):
    def __init__(self,config):
        super().__init__(config)
        self.config = config

    def forward(self,hidden_states,attention_mask=None,head_mask=None):
        """
        Returns a dict with the following:
        :last_hidden_states: B x T x hidden_size
        :all_hidden_states: tuple of len num_hidden_layers + 1 where 0th item 
            is input hidden_states and -1th item is last_hidden_states
        :attention: B x num_attention_heads x T x T
        """
        B,T,D = hidden_states.size()
        if attention_mask is None:
            attention_mask = torch.ones(B,1,1,T).to(hidden_states.device)
        
        extended_attention_mask = (1-attention_mask) * -10000

        if head_mask is None:
            head_mask = [None]*self.config.num_attention_heads

        output = super().forward(
            hidden_states,
            extended_attention_mask,
            head_mask)

        while(len(output)<3):
            output += (None,)
        
        if self.output_hidden_states==False:
            output = (output[0],None,output[1])

        to_return = {
            'last_hidden_states': output[0],
            'all_hidden_states': output[1],
            'attention': output[2],
        }
        return to_return


if __name__=='__main__':
    const = ContextLayerConstants()
    context_layer = ContextLayer(const)
    B,T,D = (5,7,768)
    hidden_states = torch.rand(B,T,D)
    output = context_layer(hidden_states)
    import pdb; pdb.set_trace()

