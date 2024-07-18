import os
import torch
from torch import nn
import copy
import math
import config as config

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
#
def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


#神经网络正则化技术,实现了层归一化（Layer Normalization）操作,帮助模型更好地学习和提取特征。
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

#构造位置嵌入
class CrossEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(CrossEmbeddings, self).__init__()
        self.position_embeddings = nn.Embedding(config.SIMS.downStream.max_position_embeddings, config.SIMS.downStream.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.SIMS.downStream.type_vocab_size, config.SIMS.downStream.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(config.SIMS.downStream.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.SIMS.downStream.hidden_dropout_prob)

    def forward(self, concat_embeddings, concat_type=None):
        # print(concat_embeddings.size())
        batch_size, seq_length = concat_embeddings.size(0), concat_embeddings.size(1)
        #batch_size：32，seq_length：768
        if concat_type is None:
            concat_type = torch.zeros(batch_size, concat_type).to(concat_embeddings.device)

        #创建了一个形状为(seq_length,)的张量，其中的元素是从0到seq_length-1的整数
        position_ids = torch.arange(seq_length, dtype=torch.long, device=concat_embeddings.device)
        #(batch_size, seq_length)
        position_ids = position_ids.unsqueeze(0).expand(concat_embeddings.size(0), -1)
        #concat_type是一个形状为(batch_size, seq_length)的张量，表示了输入序列中每个元素的类型。
        token_type_embeddings = self.token_type_embeddings(concat_type)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = concat_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


#实现跨模态的编码器，它可以处理多模态的输入数据，并输出融合后的特征
class CrossLayer(nn.Module):
    def __init__(self, config):
        super(CrossLayer, self).__init__()
        self.attention = CrossAttention(config)
        self.intermediate = CrossIntermediate(config)
        self.output = CrossOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

#对输入的多模态数据进行逐层的处理和融合，最终得到融合后的特征表示。
class CrossEncoder(nn.Module):
    def __init__(self, config):
        super(CrossEncoder, self).__init__()
        layer = CrossLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.SIMS.downStream.num_hidden_layers)])


    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

# 定义了一个基本的模型，可以得到输入数据的编码表示和池化输出，可以用于各种自然语言处理任务。
class CrossModel(nn.Module):
    def __init__(self, config):
        super(CrossModel, self).__init__()
        self.embeddings = CrossEmbeddings(config)
        self.encoder = CrossEncoder(config)

    #获取模块参数的数据类型，如果模块没有参数，那么就返回模块的第一个张量属性的数据类型。
    @property #将方法转为属性
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def forward(self, concat_input, concat_type=None, attention_mask=None, output_all_encoded_layers=True):

        if attention_mask is None:
            attention_mask = torch.ones(concat_input.size(0), concat_input.size(1))
        if concat_type is None:
            concat_type = torch.zeros_like(attention_mask)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        #转为了long型
        embedding_output = self.embeddings(concat_input, concat_type.long())

        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers


#用于实现前向传播。
class CrossOutput(nn.Module):
    def __init__(self, config):
        super(CrossOutput, self).__init__()
        # 定义了一个全连接层（self.dense），
        # 输入维度为config.MOSI.downStream.intermediate_size， #256
        # 输出维度为config.MOSI.downStream.hidden_size。 #256
        self.dense = nn.Linear(config.SIMS.downStream.intermediate_size, config.SIMS.downStream.hidden_size)
        #层归一化（self.LayerNorm），输入维度为config.MOSI.downStream.hidden_size。 256
        self.LayerNorm = LayerNorm(config.SIMS.downStream.hidden_size, eps=1e-12)
        #dropout层（self.dropout），丢弃概率为config.MOSI.downStream.hidden_dropout_prob。 0.1 or 0.3
        self.dropout = nn.Dropout(config.SIMS.downStream.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


#实现了跨注意力机制
class CrossAttention(nn.Module):
    def __init__(self, config):
        super(CrossAttention, self).__init__()
        self.self = CrossSelfAttention(config)
        self.output = CrossSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


#用于实现自注意力机制
class CrossSelfAttention(nn.Module):
    def __init__(self, config):
        super(CrossSelfAttention, self).__init__()
        #检查隐藏层的大小是否可以被注意力头的数量整除，如果不能整除，就抛出错误
        if config.SIMS.downStream.hidden_size % config.SIMS.downStream.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.SIMS.downStream.hidden_size, config.SIMS.downStream.num_attention_heads))

        self.num_attention_heads = config.SIMS.downStream.num_attention_heads  #8 or 12
        self.attention_head_size = int(config.SIMS.downStream.hidden_size / config.SIMS.downStream.num_attention_heads) #768/8 96
        self.all_head_size = self.num_attention_heads * self.attention_head_size   # 8*96=768
        #定义三个线性变换，分别用于生成查询、键和值
        self.query = nn.Linear(config.SIMS.downStream.hidden_size, self.all_head_size) #torch.Size([768, 768])
        self.key = nn.Linear(config.SIMS.downStream.hidden_size, self.all_head_size) #torch.Size([768, 768])
        self.value = nn.Linear(config.SIMS.downStream.hidden_size, self.all_head_size) #torch.Size([768, 768])
        #用于在训练过程中随机丢弃一部分神经元，以防止过拟合。
        self.dropout = nn.Dropout(config.SIMS.downStream.attention_probs_dropout_prob) #0.1 or 0.3

    #用于将输入的张量x转置，以便进行注意力打分。
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # print(x.size())
        #permute用于重新排列张量的维度
        return x.permute(0,2,1,3) #(0,2,1,3)


    def forward(self, hidden_states, attention_mask):
        #使用之前定义的线性变换生成查询、键和值。
        #hidden_states:[32, 768]
        # print(hidden_states.size())
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        #将查询、键和值进行转置。
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)


        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        #将结果除以注意力头大小的平方根，得到原始的注意力分数。
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_mask = attention_mask.float()
        attention_scores = attention_scores + attention_mask

        #使用softmax函数将注意力分数归一化为概率。
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        #计算上下文层，即注意力概率和值的点积。
        context_layer = torch.matmul(attention_probs, value_layer)

        #首先将上下文层的维度进行重新排序，然后将其形状改变为新的形状。
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

#对输入数据进行线性变换、dropout处理和层归一化处理，帮助模型更好地学习和提取特征
class CrossSelfOutput(nn.Module):
    def __init__(self, config):
        super(CrossSelfOutput, self).__init__()
        #全连接层 hidden_size：768
        self.dense = nn.Linear(config.SIMS.downStream.hidden_size, config.SIMS.downStream.hidden_size)
        #层归一化
        self.LayerNorm = LayerNorm(config.SIMS.downStream.hidden_size, eps=1e-12)
        #dropout层 hidden_dropout_prob：0.1
        self.dropout = nn.Dropout(config.SIMS.downStream.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


#用于实现前向传播
# 对输入数据进行线性变换和非线性激活，帮助模型更好地学习和提取特征
class CrossIntermediate(nn.Module):
    def __init__(self, config):
        super(CrossIntermediate, self).__init__()
        #全连接层，输入输出为256
        self.dense = nn.Linear(config.SIMS.downStream.hidden_size, config.SIMS.downStream.intermediate_size)
        #激活函数，从ACT2FN中获取对应的激活函数
        #如果config.MOSI.downStream.hidden_act是一个字符串，
        # 那么从ACT2FN字典中获取对应的激活函数；否则，直接使用config.MOSI.downStream.hidden_act作为激活函数。
        self.intermediate_act_fn = ACT2FN[config.SIMS.downStream.hidden_act] \
            if isinstance(config.SIMS.downStream.hidden_act, str) else config.SIMS.downStream.hidden_act

    #hidden_states为一个张量，应该为输入数据
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


