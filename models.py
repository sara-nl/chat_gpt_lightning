import math
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from transformers import GPT2Model,GPT2LMHeadModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import loralib as lora
from torch.utils.checkpoint import checkpoint
from tokenizer import TiktokenTokenizer


# [1] Attention is all you need
# [2] Improving Language Understanding by Generated Pre-Training
# [3] Note 10: Self-Attention & Transformers

class MaskedMultiheadSelfAttention(nn.Module):

    def __init__(self, 
                use_bias: bool = True,
                embedding_dim: int = 4,
                lora_rank: int = 0,
                dropout_rate: bool = 0.1,
                n_heads: int = 4,
                sequence_length: int = 128):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.lora_rank = lora_rank
        self.n_heads = n_heads

        # Figure 2 in [1]
        if self.lora_rank > 0:
            self.qkv_projection = lora.Linear(self.embedding_dim,
                                              3 * self.embedding_dim,
                                              bias=self.use_bias,
                                              r=self.lora_rank)
            self.output_projection = lora.Linear(self.embedding_dim,
                                                 self.embedding_dim,
                                                 bias=self.use_bias,
                                                 r=self.lora_rank)
        else:
            self.qkv_projection = nn.Linear(self.embedding_dim,
                                            3 * self.embedding_dim,
                                            bias=self.use_bias)

                                
            self.output_projection = nn.Linear(self.embedding_dim,
                                               self.embedding_dim,
                                               bias=self.use_bias)
        self.attention_dropout = nn.Dropout(self.dropout_rate)
        self.output_dropout = nn.Dropout(self.dropout_rate)

        # construct a mask like this
        # [[1, 0, 0]
        #  [1, 1, 0]]
        #  [1, 1, 1]] when sequence_length is 3
        mask = torch.tril(torch.ones(self.sequence_length, self.sequence_length))
        # insert (B, T) dimension for broadcasting later
        mask = mask.view(1, 1, self.sequence_length, self.sequence_length)
        # mask is a constant and shouldn't be considered as parameters
        # (1, 1, sequence_length, sequence_length)
        self.register_buffer("mask", mask)

    def forward(self, x: Tensor, attention_mask: Tensor = None):
        """
        x: shape of (B, T, C)
        """
        B, T, C = x.shape
        # Project x three times and split into Q,K,V
        x3 = self.qkv_projection(x)  # (B, T, 3C)
        
        #Q, K, V = x3.split(self.embedding_dim,
        #                   dim=2)  # (B, T, C) for each

        Q, K, V = x3.chunk(chunks=3, dim=-1)

        # Prepare Q,K,V into desired shape for multi-head attention
        # Multi-head attention is equivalent to single-head attention on sequence-tensor form
        # see 3.1 in [3]
        Q = Q.view(B, T, self.n_heads,
                   C // self.n_heads)  # (B, T, h, h_dim)
        Q = Q.transpose(1, 2)  # (B, h, T, h_dim)
        K = K.view(B, T, self.n_heads,
                   C // self.n_heads)  # (B, T, h, h_dim)
        K = K.transpose(1, 2)  # (B, h, T, h_dim)
        V = V.view(B, T, self.n_heads,
                   C // self.n_heads)  # (B, T, h, h_dim)
        V = V.transpose(1, 2)  # (B, h, T, h_dim)

        # (B, h, T, h_dim) @ (B, h, h_dim, T) -> (B, h, T, T)
        attention = Q @ K.transpose(2, 3)
        attention *= 1.0 / math.sqrt(K.size(-1))
        # In transformer decoder, one word can only attend to words before itself
        attention[torch.where(self.mask[:,:,:T,:T]==0)] = float("-inf")
        #attention = attention.masked_fill(self.mask[:, :, :T, :T] == 0,
        #                                  float('-inf'))  # (B, h, T, T)
        if attention_mask is not None:
            # https://github.com/huggingface/transformers/blob/c7f3abc257af9dfb6006a76f2b09b48355322d4d/src/transformers/models/gpt2/modeling_gpt2.py#L805
            # also, we don't need attend to padding tokens
            attention_mask = attention_mask[:, None,
                             None, :]  # (B, T) -> (B, 1, 1, T)
            attention_mask = (1.0 - attention_mask) * torch.finfo(
                attention.dtype).min
            # This will broadcast to each row of the last dimension of attention map
            # [[[[1, -inf, -inf],
            #    [1, 1,    -inf],
            #    [1, 1,    1   ]]]]]  + [[[[0, 0, -float.min]]]]]
            # =
            # [[[[1, -inf, -inf       ],
            #    [1, 1,    -inf       ],
            #    [1, 1,    1-float.min]]]]]
            attention = attention + attention_mask

        attention = F.softmax(attention, dim=-1)  # (B, h, T, T)
        attention = self.attention_dropout(attention)
        # (B, h, T, T) @ (B, h, T, h_dim) -> (B, h, T, h_dim)
        # V weighted by attention
        weighted_value = attention @ V
        # restore the original shape (B, T, C)
        weighted_value = weighted_value.transpose(1, 2).contiguous().view(
            B, T, C)

        # Finally, linearly project the weighted value to get the output
        y = self.output_projection(weighted_value)
        y = self.output_dropout(y)
        return y


class FeedForwardNetworks(nn.Module):

    def __init__(self, 
                lora_rank: int=0,
                dropout_rate: float = 0.1,
                use_bias: bool = True,
                embedding_dim: int = 4,
                ):
        super().__init__()
        self.lora_rank = lora_rank

        if self.lora_rank > 0:
            self.fc1 = lora.Linear(embedding_dim,
                                   4 * embedding_dim,
                                   bias=use_bias,
                                   r=lora_rank)
            self.fc2 = lora.Linear(4 * embedding_dim,
                                   embedding_dim,
                                   bias=use_bias,
                                   r=lora_rank)
        else:
            self.fc1 = nn.Linear(embedding_dim,
                                 4 * embedding_dim,
                                 bias=use_bias)
            self.fc2 = nn.Linear(4 * embedding_dim,
                                 embedding_dim,
                                 bias=use_bias)
        self.dropout = nn.Dropout(dropout_rate)

    def gelu(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        y = self.dropout(x)
        return y


class TransformerDecoderBlock(nn.Module):

    def __init__(self,
                use_bias: bool = True,
                embedding_dim: int = 4,
                lora_rank: int = 0,
                dropout_rate: bool = 0.1,
                n_heads: int = 4,
                sequence_length: int = 128):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dim,
                                elementwise_affine=use_bias)
        self.mmsa = MaskedMultiheadSelfAttention(use_bias,
                                                embedding_dim,
                                                lora_rank,
                                                dropout_rate,
                                                n_heads,
                                                sequence_length)
        self.ln2 = nn.LayerNorm(embedding_dim,
                                elementwise_affine=use_bias)
        self.ffn = FeedForwardNetworks(lora_rank,
                                        dropout_rate,
                                        use_bias,
                                        embedding_dim)

    def forward(self, x: Tensor, attention_mask: Tensor = None):
        identity1 = x
        x = self.ln1(x)
        x = self.mmsa(x, attention_mask)
        x = identity1 + x

        identity2 = x
        x = self.ln2(x)
        x = self.ffn(x)
        y = identity2 + x
        return y


class TransformerDecoder(nn.Module):

    def __init__(self, vocab_size: int=50257,
                        embedding_dim: int = 1024,
                        n_layers: int = 24,
                        sequence_length: int = 256,
                        dropout_rate: float = 0.1,
                        activation_checkpointing: bool=False,
                        lora_rank: int = 0,
                        use_bias: bool=True,
                        n_heads: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.dropout_rate = dropout_rate
        self.activation_checkpointing = activation_checkpointing
        self.use_bias = use_bias

        self.token_embedding_layer = nn.Embedding(
            self.vocab_size, self.embedding_dim)  # (Vocab, d)

        #self.postion_embedding_layer = nn.Embedding(self.vocab_size,
        #                                            self.embedding_dim)

        self.postion_embedding_layer = nn.Parameter(torch.empty(self.sequence_length, self.embedding_dim))
        
        self.input_dropout = nn.Dropout(self.dropout_rate)
        self.decoder_blocks = nn.ModuleList([TransformerDecoderBlock(use_bias,embedding_dim,lora_rank,dropout_rate,n_heads,sequence_length) for _ in range(self.n_layers)])
        self.ln = nn.LayerNorm(self.embedding_dim,
                               elementwise_affine=self.use_bias)

    def forward(self, x: Tensor, attention_mask: Tensor = None):
        B, T = x.shape
        offset = 0
        #pos = torch.arange(0, T, dtype=torch.long).cuda().unsqueeze(0)  # (1, T)
        token_embeddings = self.token_embedding_layer(x)  # (B, T, d)
        #pos_embeddings = self.postion_embedding_layer[:T,:] # (B, T, d)
        pos_embeddings = self.postion_embedding_layer[offset : offset + x.shape[-1]]
        x = self.input_dropout(token_embeddings + pos_embeddings)

        # N decoder blocks
        for block in self.decoder_blocks:
            if self.activation_checkpointing:
                x = checkpoint(block, x, attention_mask)
            else:
                x = block(x, attention_mask)

        y = self.ln(x)
        return y


class GPT(nn.Module):

    def __init__(self, lora_rank: int=0,
                        vocab_size: int=50257,
                        n_heads: int = 16,
                        embedding_dim: int =  1024,
                        n_layers: int = 24,
                        sequence_length: int = 256,
                        dropout_rate: float = 0.1,
                        activation_checkpointing: bool = False,
                        use_bias: bool = True
                        ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.dropout_rate = dropout_rate
        self.lora_rank = lora_rank
        self.n_heads = n_heads
        self.activation_checkpointing = activation_checkpointing

        self.tokenizer = TiktokenTokenizer("gpt2")

        self.transformer = TransformerDecoder(vocab_size,
                                            embedding_dim,
                                            n_layers,
                                            sequence_length,
                                            dropout_rate,
                                            activation_checkpointing,
                                            lora_rank,
                                            use_bias,
                                            n_heads)
        # Final linear layer as language model head w/o softmax
        if self.lora_rank > 0:
            self.lm_head = lora.Linear(self.embedding_dim,
                                       self.vocab_size,
                                       bias=False,
                                       r=self.lora_rank)
        else:
            self.lm_head = nn.Linear(self.embedding_dim,
                                     self.vocab_size,
                                     bias=False)

    def forward(self, x: Tensor, attention_mask: Tensor = None):
        """
        x: Shape of (B, T)
        """
        x = self.transformer(x, attention_mask)  # x = (B, T, embedding_dim)
        logits = self.lm_head(x)  # logits = (B, T, voca_size)
        return logits

    @classmethod
    def from_pretrained(self, 
                        hf_model: str = "gpt2-medium",
                        lora_rank: int=0,
                        vocab_size: int=50257,
                        n_heads: int = 16,
                        embedding_dim: int =  1024,
                        n_layers: int = 24,
                        sequence_length: int = 256,
                        dropout_rate: float = 0.1,
                        activation_checkpointing: bool = False,
                        use_bias: bool = True):
        """
        https://github.com/karpathy/nanoGPT/blob/master/model.py#L213
        """

        def convert_state_key(k):
            huggingface_names = {
                "token_embedding_layer": "wte",
                "postion_embedding_layer": "wpe.weight",
                "decoder_blocks": "h",
                "mmsa": "attn",
                "ln1": "ln_1",
                "ln2": "ln_2",
                "ffn": "mlp",
                "fc1": "c_fc",
                "fc2": "c_proj",
                "qkv_projection": "c_attn",
                "output_projection": "c_proj",
                "ln": "ln_f",
            }
            hf_key = []
            for name in k.split('.'):
                hf_key.append(huggingface_names.get(name, name))
            return '.'.join(hf_key)

        def should_transpose(k):
            transposed = [
                'attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight',
                'mlp.c_proj.weight'
            ]
            for t in transposed:
                if k.endswith(t):
                    return True
            return False

        model = GPT(lora_rank,
                    vocab_size,
                    n_heads,
                    embedding_dim,
                    n_layers,
                    sequence_length,
                    dropout_rate,
                    activation_checkpointing,
                    use_bias)

        model_states = model.state_dict()
        model_states_keys = [
            k for k in model_states.keys() if not k.endswith('.mmsa.mask')
        ]
        model_states_keys = [k for k in model_states_keys if not 'lora' in k]

        model_pretrained = GPT2LMHeadModel.from_pretrained(hf_model)
        pretrained_states = model_pretrained.state_dict()

        pretrained_states_keys = [
            k for k in pretrained_states.keys()
            if not k.endswith('.attn.masked_bias')
        ]
        pretrained_states_keys = [
            k for k in pretrained_states_keys if not k.endswith('.attn.bias')
        ]

        #assert len(pretrained_states_keys) == len(
        #    model_states_keys
        #), f"mismatched keys: {len(pretrained_states_keys)} != {len(model_states_keys)}"

        for dst_key in model_states_keys:
            src_key = convert_state_key(dst_key)
            if should_transpose(src_key):
                assert pretrained_states[src_key].shape[::-1] == model_states[
                    dst_key].shape
                with torch.no_grad():
                    model_states[dst_key].copy_(pretrained_states[src_key].t())
            else:
                assert pretrained_states[src_key].shape == model_states[
                    dst_key].shape
                with torch.no_grad():
                    model_states[dst_key].copy_(pretrained_states[src_key])

        return model

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        https://github.com/karpathy/nanoGPT/blob/master/model.py#L343
    
        Take a conditioning sequence of idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at sequence_length
            idx_cond = idx if idx.size(
                1) <= self.sequence_length else idx[:, -self.sequence_length:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            next_id = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, next_id), dim=1)

        return idx

    @torch.no_grad()
    def batch_generate(self,
                       idx: torch.Tensor,
                       input_masks: torch.Tensor,
                       input_lengths: torch.Tensor,
                       max_new_tokens: int,
                       temperature=1.0,
                       top_k=None):
        """
        idx: (B, T)
        input_masks: (B, T)
        """
        B, T = idx.size()
        min_input_length = torch.min(input_lengths)  # (B)
        max_input_length = torch.max(input_lengths)  # (B)
        total_length = min(self.sequence_length,
                           max_input_length + max_new_tokens)
        if T < total_length:
            idx = F.pad(idx, (0, total_length - T), value=self.tokenizer.pad_token)
            input_masks = F.pad(input_masks, (0, total_length - T), value=0.0)
        input_masks = input_masks.bool()

        for curr_pos in range(min_input_length, total_length):
            # forward the model to get the logits for the index in the sequence
            logits = self(idx[:, :curr_pos])
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            next_id = torch.multinomial(probs, num_samples=1).view(-1)
            next_id = torch.where(input_masks[:, curr_pos], idx[:, curr_pos],
                                  next_id)
            # append sampled index to the running sequence and continue
            idx[:, curr_pos] = next_id

        return idx


class GPTRewardModel(nn.Module):

    def __init__(self, lora_rank: int,
                        vocab_size: int=50257,
                        n_heads: int = 12,
                        embedding_dim: int = 768,
                        n_layers: int = 12,
                        sequence_length: int = 1024,
                        dropout_rate: float = 0.1,
                        activation_checkpointing: bool = False,
                        use_bias: bool = True):
        super().__init__()

        self.backbone = GPT(lora_rank,
                    vocab_size,
                    n_heads,
                    embedding_dim,
                    n_layers,
                    sequence_length,
                    dropout_rate,
                    activation_checkpointing,
                    use_bias)
                    
        self.lora_rank = lora_rank

        # no need for LoRA here as we won't have weights anyway
        self.value_head = nn.Linear(embedding_dim, 1, bias=False)

    def forward(self, x: Tensor, attention_mask: Tensor = None):
        hidden = self.backbone(x, attention_mask)
        out = self.value_head(hidden).mean(dim=1)
        return out


    def freeze_weights(self):

        if self.lora_rank > 0:
            lora.mark_only_lora_as_trainable(self)

        else:
            # fine tune last block, hard-coded the 11th layer, being the last one in 12 layer gpt-model
            trainable_params = [
                "backbone.transformer.decoder_blocks.11.mmsa.mask",
                "backbone.transformer.decoder_blocks.11.mmsa.qkv_projection.weight",
                "backbone.transformer.decoder_blocks.11.mmsa.qkv_projection.bias",
                "backbone.transformer.decoder_blocks.11.mmsa.output_projection.weight",
                "backbone.transformer.decoder_blocks.11.mmsa.output_projection.bias",
                "backbone.transformer.decoder_blocks.11.ln2.weight",
                "backbone.transformer.decoder_blocks.11.ln2.bias",
                "backbone.transformer.decoder_blocks.11.ffn.fc1.weight",
                "backbone.transformer.decoder_blocks.11.ffn.fc1.bias",
                "backbone.transformer.decoder_blocks.11.ffn.fc2.weight",
                "backbone.transformer.decoder_blocks.11.ffn.fc2.bias",
                "backbone.transformer.ln.weight",
                "backbone.transformer.ln.bias", "value_head.weight"
            ]
            for name, param in self.named_parameters():
                if name not in trainable_params:
                    param.requires_grad = False
                else:
                    print(f"{name} is trainable.")


class GPTCritic(nn.Module):

    def __init__(self, lora_rank: int,
                vocab_size: int=50257,
                n_heads: int = 12,
                embedding_dim: int = 768,
                n_layers: int = 12,
                sequence_length: int = 1024,
                dropout_rate: float = 0.1,
                activation_checkpointing: bool = False,
                use_bias: bool = True):

        super().__init__()

        self.model = GPTRewardModel(lora_rank,
                                    vocab_size,
                                    n_heads,
                                    embedding_dim,
                                    n_layers,
                                    sequence_length,
                                    dropout_rate,
                                    activation_checkpointing,
                                    use_bias)
            

    def forward_critic(self,
                       x: Tensor,
                       attention_mask: Tensor = None,
                       num_actions=0) -> torch.Tensor:
        """
        x: (B, T)
        """
        hidden = self.model.backbone(x, attention_mask)  # (B, T, vocab_size)
        values = self.model.value_head(hidden).squeeze(-1)  # (B, T, 1)
        # Vt only depends on st
        values = values * attention_mask
        values = values[:, :-num_actions].mean(dim=1)
        return values  # (B, 1)

class GPTActor(nn.Module):

    def __init__(self,
                lora_rank: int=0,
                vocab_size: int=50257,
                n_heads: int = 16,
                embedding_dim: int =  1024,
                n_layers: int = 24,
                sequence_length: int = 256,
                dropout_rate: float = 0.1,
                activation_checkpointing: bool = False,
                use_bias: bool = True):

        super().__init__()
        self.model = GPT(lora_rank,
                vocab_size,
                n_heads,
                embedding_dim,
                n_layers,
                sequence_length,
                dropout_rate,
                activation_checkpointing,
                use_bias)

        self.tokenizer = TiktokenTokenizer("gpt2")

    def forward_actor(self,
                      x: Tensor,
                      attention_mask: Tensor = None,
                      num_actions=1):
        """
        x (B, T)
        """
        logits = self.model(x, attention_mask)  # logits = (B, T, voca_size)
        log_prob_all_vocab = F.log_softmax(logits[:, :-1, :],
                                           dim=2)  # (B, T-1, vocab_size)
        # no need to know the logits of last token because we don't have the index of that token in x
        index = x[:, 1:].unsqueeze(-1)  # (B, T-1, 1)
        log_prob_output = log_prob_all_vocab.gather(
            dim=2,
            index=index)  # teacher-forcing, get the prob of each gt token
        return log_prob_output[:, -num_actions:, 0]  # (B, T)

    def batch_generate(self,
                       idx,
                       input_masks: torch.Tensor,
                       input_lengths: torch.Tensor,
                       max_new_tokens,
                       temperature=1,
                       top_k=None):
        """
        idx: Shape of (B, T)
        """
        B, T = idx.size()
        completions = self.model.batch_generate(idx, 
                                                input_masks, 
                                                input_lengths, 
                                                max_new_tokens,
                                                temperature,
                                                top_k)  # completions = (B, T)

        attention_mask = torch.where(completions != self.tokenizer.pad_token,
                                     torch.ones_like(completions),
                                     torch.zeros_like(completions))
        action_mask = torch.ones_like(completions, dtype=torch.bool)
        action_mask[:, :T] = 0.0
        action_mask = action_mask[:, 1:]
        # we can only take the minimum among all instances in this batch as common num_actions
        num_actions = completions.size(1) - T
        return completions, attention_mask, num_actions, action_mask[:, -num_actions:]