
from .common import *

def convert_Module_Embedding(module=nn.Embedding(num_embeddings=0, embedding_dim=0, padding_idx=None, max_norm=None, norm_type=2., scale_grad_by_freq=False, sparse=False), inputs=None, output=None):
    sd = module.state_dict()
    return convert_demonstrate(module, inputs, output)

def convert_Module_MultiheadAttention(module=nn.MultiheadAttention(embed_dim=8, num_heads=2, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=8, vdim=8, batch_first=False), inputs=None, output=None):
    sd = module.state_dict()
    if not module._qkv_same_embed_dim:
        # info._q_weight = encode_tensor_to_str(sd['q_proj_weight'])
        # info._k_weight = encode_tensor_to_str(sd['k_proj_weight'])
        # info._v_weight = encode_tensor_to_str(sd['v_proj_weight'])
        pass
    else:
        # info._q_weight = encode_tensor_to_str(sd['in_proj_weight'][:module.embed_dim])
        # info._k_weight = encode_tensor_to_str(sd['in_proj_weight'][module.embed_dim:module.embed_dim*2])
        # info._v_weight = encode_tensor_to_str(sd['in_proj_weight'][module.embed_dim*2:])
        pass
    return convert_demonstrate(module, inputs, output)

def convert_Module_MultiHeadAttentionRelativePosition(module=FileNotFoundError, inputs=None, output=None):
    sd = module.state_dict()
    return convert_demonstrate(module, inputs, output)
