import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, NLLLoss, KLDivLoss
from transformers import *
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_utils import Conv1D
from transformers.modeling_utils import find_pruneable_heads_and_indices, prune_conv1d_layer
from typing import Tuple, Optional



class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False, is_cross_attention=False):
        super().__init__()

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer(
            "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.is_cross_attention = is_cross_attention
        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * n_state, nx)
            self.q_attn = Conv1D(n_state, nx)
        else:
            self.c_attn = Conv1D(3 * n_state, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_head, self.split_size // self.n_head, self.pruned_heads
        )
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / (float(v.size(-1)) ** 0.5)
        nd, ns = w.size(-2), w.size(-1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            mask = self.bias[:, :, ns - nd : ns, :ns]
            w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        if encoder_hidden_states is not None:
            assert hasattr(
                self, "q_attn"
            ), "If class is used as cross attention, the weights `q_attn` have to be defined. Please make sure to instantiate class with `Attention(..., is_cross_attention=True)`."
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        else:
            present = (None,)

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions) 



class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)



class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        hidden_size = config.n_embd
        inner_dim = 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = Attention(hidden_size, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(inner_dim, config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        attn_outputs = self.attn(
            self.ln_1(hidden_states),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + hidden_states

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attn_outputs = self.crossattention(
                self.ln_cross_attn(hidden_states),
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = hidden_states + attn_output
            outputs = outputs + cross_attn_outputs[1:]  # add cross attentions if we output attention weights

        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states))
        # residual connection
        hidden_states = hidden_states + feed_forward_hidden_states

        outputs = [hidden_states] + outputs
        return outputs  # hidden_states, present, (cross_attentions, attentions) 




class VisualProject(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(VisualProject, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)



class PriorNetwork(nn.Module): 
    def __init__(self, d_in, d_out):
        super(PriorNetwork, self).__init__() 
        self.mean = nn.Linear(d_in, d_out) 
        self.logvar = nn.Linear(d_in, d_out) 
    
    def forward(self, input_emb): 
        mean = self.mean(input_emb) 
        logvar = self.logvar(input_emb) 

        outputs = (mean, logvar,) 
        return outputs 



class LatentDynamicModule(nn.Module): 
    def __init__(self, args, config): 
        super(LatentDynamicModule, self).__init__()
        self.args = args 
        self.config = config 
        self.wpe = nn.Embedding(256, config.n_embd) 
        self.drop = nn.Dropout(config.embd_pdrop) 
        self.h = nn.ModuleList([Block(256, config, scale=True) for _ in range(2)]) 
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.loss = nn.MSELoss()

    def forward(self, sentence_hidden):
        hidden_states = sentence_hidden
        position_ids = torch.arange(0, hidden_states.size(1), dtype=torch.long, device=sentence_hidden.device)
        position_ids = position_ids.unsqueeze(0).view(-1, hidden_states.size(1))
        position_embeds = self.wpe(position_ids)
        hidden_states = hidden_states + position_embeds
        hidden_states = self.drop(hidden_states)

        for i, block in enumerate(self.h):
            outputs = block(hidden_states)
            hidden_states, present = outputs[:2]
        hidden_states = self.ln_f(hidden_states)
        loss = self.loss(hidden_states[0, :-1, :], sentence_hidden[0, 1:, :])
        return hidden_states, loss 



class MutiCaptionGenerator(nn.Module): 
    def __init__(self, config, args, tokenizer):
        super(MutiCaptionGenerator, self).__init__()
        self.args = args 
        self.config = config 
        self.captioner = GPT2Model.from_pretrained(args.tokenizer_path) 
        self.captioner.resize_token_embeddings(len(tokenizer))

        self.visual_project = VisualProject((args.prefix_size, (config.n_embd * args.prefix_length) // 2,
                                     config.n_embd * args.prefix_length)) 
        
        self.latent_project = PriorNetwork(config.n_embd, config.n_latent_variables)
        self.n_latent_variables = config.n_latent_variables
        self.lm_head = nn.Linear(config.n_embd + config.n_latent_variables, len(tokenizer), bias=False) 
        self.bow_head = nn.Linear(config.n_embd, len(tokenizer), bias=False)
        self.latent_dynamic_module = LatentDynamicModule(args, config) 

        self.loss_fct = CrossEntropyLoss(ignore_index=-100)
        self.loss_bow = CrossEntropyLoss(ignore_index=-100)



    def step(self, input_embs, sentence_index, token_type_ids, attention_mask=None): 
        transformer_outputs_outputs = self.captioner(
            inputs_embeds=input_embs, 
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        ) 
        hidden_states = transformer_outputs_outputs[0]  # （bsz, seq_len, model_d）

        for j in range(hidden_states.size(0)): 
            latent_dynamic_hiddents = hidden_states[j:j+1].index_select(1, sentence_index[1:]) 
            latent_output, _ = self.latent_dynamic_module(latent_dynamic_hiddents) # (bsz, num_of_dyn, model_d)  
            latent_mean, _ = self.latent_project(latent_output) 
            z = latent_mean
        lm_logits = self.lm_head(torch.cat([hidden_states[:, -1, :], z[:, -1, :]], dim=-1)) 
        return lm_logits 


    def forward(self, input_embs, sentence_index, labels, token_type_ids=None, attention_mask=None, from_mean=False): 
        transformer_outputs_outputs = self.captioner(
            inputs_embeds=input_embs, 
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        ) 
        hidden_states = transformer_outputs_outputs[0] # （bsz, seq_len, model_d）
        
        loss_gen_cum = 0 
        loss_dyn_cum = 0 
        loss_bow_cum = 0
        loss_kl_cum = 0

        for j in range(hidden_states.size(0)): 
            latent_dynamic_hiddents = hidden_states[j:j+1].index_select(1, sentence_index[1:]) # (bsz, num_of_dyn, model_d) 

            latent_output, loss_dyn = self.latent_dynamic_module(latent_dynamic_hiddents) 
            
            hidden_states_list = [] 
            labels_list = [] 
            for i in range(sentence_index.size(0) - 1): 
                start = sentence_index[i] + 1 + self.args.prefix_length
                end = sentence_index[i+1] 
                hidden_states_list.append(hidden_states[:, start: end]) 
                labels_list.append(labels[:, start: end]) 
            
            hidden_logits_list = [] 
            bow_logits_list = [] 
            kl_latent_list = []

            for i in range(len(labels_list)): 
                temp_hidden_states = hidden_states_list[i] 
                t_latent_hidden = latent_output[j][i].expand(temp_hidden_states.size(1), -1).unsqueeze(0) 
                
                latent_mean, latent_logvar = self.latent_project(t_latent_hidden) 
                if from_mean: 
                    z = latent_mean 
                else:
                    z = self.reparameterize(latent_mean, latent_logvar)  
                kl_latent_list.append([latent_mean, latent_logvar])

                lm_logits = self.lm_head(torch.cat([z, temp_hidden_states], dim=-1))
                hidden_logits_list.append(lm_logits) 
                
                bow_logits = self.bow_head(latent_output[j][i].unsqueeze(0) )
                bow_logits_list.append(bow_logits) 
            
            loss_gen_cum += self._compute_generation_loss(hidden_logits_list, labels_list) 
            loss_bow_cum += self._compute_bow_loss(bow_logits_list, labels_list) 
            loss_dyn_cum += loss_dyn 
            loss_kl_cum += self._compute_kl_loss(kl_latent_list)
        return loss_gen_cum + loss_dyn_cum + loss_bow_cum + loss_kl_cum 


    def _compute_generation_loss(self, hidden_logits_list, labels_list):
        loss = 0
        for hidden, target in zip(hidden_logits_list, labels_list):
            loss = loss + self.loss_fct(hidden[:, :-1, :].contiguous().view(-1, hidden.size(-1)),
                                        target[:, 1:].contiguous().view(-1))
        loss = loss / len(labels_list)
        return loss
    
    def _compute_bow_loss(self, predictions, labels_list):
        loss = 0
        for pred, target in zip(predictions, labels_list):
            pred = pred.expand(target.size(1), -1).unsqueeze(0)
            loss = loss + self.loss_bow(pred[:, :-1, :].contiguous().view(-1, pred.size(-1)),
                                        target[:, 1:].contiguous().view(-1))
        loss = loss / len(labels_list)
        return loss
    
    def reparameterize(self, mean, logvar, z=None):
        std = logvar.mul(0.5).exp()
        if z is None:
            z = torch.randn(std.size(), device=mean.device, dtype=mean.dtype)
        return z.mul(std) + mean 
    
    def kl_loss(self, mean1, logvar1, mean2, logvar2):
        exponential = logvar1 - logvar2 - torch.pow(mean1 - mean2, 2) / logvar2.exp() - torch.exp(logvar1 - logvar2) + 1
        result = -0.5 * torch.sum(exponential, tuple(range(1, len(exponential.shape))))
        return result.mean()


    def _compute_kl_loss(self, kl_latent_list): 
        loss = 0 
        for i in range(len(kl_latent_list)): 
            mean, logvar = kl_latent_list[i]
            prior_mean, prior_logvar = mean[:, :-1, :], logvar[:, :-1, :]
            posterior_mean, posterior_logvar = mean[:, 1:, :], logvar[:, 1:, :]
            loss += self.kl_loss(posterior_mean.view(-1, self.n_latent_variables), posterior_logvar.view(-1, self.n_latent_variables), \
                                prior_mean.view(-1, self.n_latent_variables), prior_logvar.view(-1, self.n_latent_variables)).unsqueeze(0) 
        return loss
        
