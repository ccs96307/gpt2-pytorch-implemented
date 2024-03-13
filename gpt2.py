from typing import OrderedDict, Optional, List, Tuple, Union

from dataclasses import dataclass
import json
import math

import regex as re
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file


@dataclass
class BaseModelOutputWithPastAndCrossAttentions:
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class CausalLMOutputWithCrossAttentions:
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


class GPT2Config:
    def __init__(
        self,
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_inner: Optional[int] = None,
        activation_function: str = "gelu_new",
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        summary_type: str = "cls_index",
        summary_use_proj: bool = True,
        summary_activation: Optional[str] = None,
        summary_proj_to_labels: bool = True,
        summary_first_dropout: bool = 0.1,
        scale_attn_weights: bool = True,
        use_cache: bool = True,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        scale_attn_by_inverse_layer_idx: bool = False,
        reorder_and_upcast_attn: bool = False,
        output_attention: bool = False,
        output_hidden_states: bool = False,
        **kargs,
    ) -> None:
        # Mapping
        self.hidden_size = n_embd
        self.max_position_embeddings = n_positions
        self.num_attention_heads = n_positions
        self.num_hidden_layers = n_layer

        # Init
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_proj_to_labels = summary_proj_to_labels
        self.summary_first_dropout = summary_first_dropout
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn
        self.output_attention = output_attention
        self.output_hidden_states = output_hidden_states

    @staticmethod
    def from_pretrained_model_or_path(pretrained_model_name_or_path: str) -> "GPT2Config":
        resolved_archive_file = cached_file(
            path_or_repo_id=pretrained_model_name_or_path,
            filename=CONFIG_NAME,
            _raise_exceptions_for_missing_entries=False,
        )
        
        config_content = json.load(open(resolved_archive_file))
        return GPT2Config(**config_content)


class NewGELUActivation(torch.nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class Conv1D(torch.nn.Module):
    """This is a special customized linear layer, it is used for GPT-2 model."""
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(input_dim, output_dim))
        self.bias = torch.nn.Parameter(torch.empty(output_dim))
        self.reset_paramter()

    def reset_paramter(self):
        torch.nn.init.normal_(self.weight, std=0.02)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        # Calculate outout is: (batch_size, seq_len, output_dim)
        output_shape = x.size()[:-1] + (self.weight.size(1),)
        
        # x shape: (batch_size * seq_len, input_dim)
        x = x.view(-1, x.size(-1))

        # Output shape: (batch_size * seq_len, output_dim)
        # x = torch.addmm(self.bias, x, self.weight)
        x = F.linear(input=x, weight=self.weight.T, bias=self.bias)
        x = x.view(output_shape)
        return x


class GPT2Attention(torch.nn.Module):
    def __init__(self, config: GPT2Config) -> torch.Tensor:
        super().__init__()
        # Init
        self.n_head = config.n_head
        self.head_size = int(config.hidden_size / self.n_head)
        self.scale = 1 / (self.head_size ** 0.5)
        self.hidden_size = config.hidden_size

        self.c_attn = Conv1D(input_dim=config.hidden_size, output_dim=3*config.hidden_size)
        self.c_proj = Conv1D(input_dim=config.hidden_size, output_dim=config.hidden_size)
        self.attn_dropout = torch.nn.Dropout(p=config.attn_pdrop)
        self.resid_dropout = torch.nn.Dropout(p=config.resid_pdrop)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> torch.Tensor:
        batch_size = hidden_states.size(0)

        # QKV
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(self.hidden_size, dim=-1)

        # Reshape
        q = q.contiguous().view(batch_size, -1, self.n_head, self.head_size).permute(0, 2, 1, 3)
        k = k.contiguous().view(batch_size, -1, self.n_head, self.head_size).permute(0, 2, 1, 3)
        v = v.contiguous().view(batch_size, -1, self.n_head, self.head_size).permute(0, 2, 1, 3)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        
        if use_cache:
            present = (k, v)
        else:
            present = None

        # Compute Q @ K^T
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores * self.scale

        # Causal mask
        seq_len = hidden_states.size(-2)
        mask_value = torch.finfo(hidden_states.dtype).min
        causal_mask = torch.triu(torch.full((seq_len, seq_len), mask_value), diagonal=1)
        attention_scores = attention_scores + causal_mask

        # Attention mask
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
            attention_scores = attention_scores + attention_mask

        # Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)

        # Compute V
        attention_scores = torch.matmul(attention_weights, v)

        # Reshape
        context_layer = attention_scores.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(batch_size, -1, self.head_size * self.n_head)
        attention_output = self.c_proj(context_layer)

        # Skip connection & Dropout
        attention_output = self.resid_dropout(attention_output)

        outputs = (attention_output, present)
        if output_attentions:
            outputs += (attention_weights,)
        
        return outputs


class GPT2MLP(torch.nn.Module):
    def __init__(self, inner_dim: int, config: GPT2Config) -> None:
        super().__init__()
        self.c_fc = Conv1D(input_dim=config.hidden_size, output_dim=inner_dim)
        self.c_proj = Conv1D(input_dim=inner_dim, output_dim=config.hidden_size)
        self.act = NewGELUActivation()
        self.dropout = torch.nn.Dropout(p=config.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x        


class GPT2Block(torch.nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size

        self.ln_1 = torch.nn.LayerNorm(normalized_shape=config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config=config)
        self.ln_2 = torch.nn.LayerNorm(normalized_shape=config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config=config)

    def forward(
        self,
        hidden_states: torch.LongTensor,
        layer_past: Optional[Tuple[torch.LongTensor]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states

        # Self-Attention
        hidden_states = self.ln_1(hidden_states)
        attention_outputs = self.attn(
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attention_output = attention_outputs[0]  # output_attn: (attention_output, present, all_attentions)
        outputs = attention_outputs[1:]
        
        # Residual connection
        hidden_states = attention_output + residual
        residual = hidden_states

        # MLP
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        # Cache
        if use_cache:
            outputs = (hidden_states,) + outputs  # outputs: (hidden_states, present, all_attentions)
        else:
            outputs = (hidden_states,) + outputs  # outputs: (hidden_states, all_attentions)

        return outputs
    

class GPT2Model(torch.nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.wte = torch.nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = torch.nn.Embedding(config.n_positions, config.n_embd)
        self.dropout = torch.nn.Dropout(p=config.embd_pdrop)
        self.h = torch.nn.ModuleList([GPT2Block(config=config) for _ in range(config.n_layer)])
        self.ln_f = torch.nn.LayerNorm(normalized_shape=config.n_embd, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        # Token embeddings
        token_embeddings = self.wte(input_ids)

        # Position embeddings
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.size(1))
            position_embeddings = self.wpe(position_ids).view(1, -1, token_embeddings.size(-1))
        else:
            position_embeddings = self.wpe(position_ids)

        # Sum the embeddings
        embeddings = token_embeddings + position_embeddings

        # KV Cache
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        # Computation
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        
        hidden_states = self.dropout(embeddings)

        for block, layer_past in zip(self.h, past_key_values):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states)

            outputs = block(
                hidden_states=hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]

            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache is True else 1],)
        
        # LayerNorm
        hidden_states = self.ln_f(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=None,
        )


class MyGPT2LMHeadModel(torch.nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        # Cache
        self.use_cache = config.use_cache
        self.output_hidden_states = config.output_hidden_states

        self.eos_token_id = config.eos_token_id

        self.transformer = GPT2Model(config=config)
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tie_weights()

    def tie_weights(self) -> None:
        self.lm_head.weight = self.transformer.wte.weight

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> CausalLMOutputWithCrossAttentions:
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = transformer_outputs.last_hidden_state

        lm_logits = self.lm_head(hidden_states)

        # Loss
        loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
    
    @staticmethod
    def from_pretrained(pretrained_model_name_or_path: str) -> "MyGPT2LMHeadModel":
        """Load pretrained weights from HuggingFace into model.
        
        Args:
            pretrained_model_name_or_path: One of
                * "openai-community/gpt2"
                ...

        Returns:
            model: BertModel model with weights loaded
        """

        def load_state_dict_hf(path_or_repo_id: str) -> OrderedDict:
            resolved_archive_file = cached_file(
                path_or_repo_id=path_or_repo_id,
                filename=WEIGHTS_NAME,
            )
            return torch.load(resolved_archive_file, weights_only=True)

        # Load config
        config = GPT2Config.from_pretrained_model_or_path(pretrained_model_name_or_path=pretrained_model_name_or_path)

        # Load weights
        state_dict = load_state_dict_hf(pretrained_model_name_or_path)

        new_state_dict = {}
        for key in state_dict:
            if not re.findall(r"h.\d+.attn.[wb]", key):
                new_key = "transformer." + key
                new_state_dict[new_key] = state_dict[key]

        # Load model
        model = MyGPT2LMHeadModel(config=config)
        model.load_state_dict(new_state_dict, strict=False)

        return model

    @torch.no_grad
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 20,
        top_p: float = 0.9,
        top_k: int = 10,
        no_repeat_ngram_size: int = 2,
        early_stopping: bool = False,
        use_cache: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        # Prepare input
        batch_size = input_ids.shape[0]
        past_key_values = None
        generation_mode = "greedy"
        finished = torch.zeros(batch_size, dtype=torch.bool)
        all_sequences = input_ids

        # Position ids
        position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)
        position_ids.masked_fill_(attention_mask == 0, 1)
        
        # Greedy search
        if generation_mode == "greedy":
            for idx in range(max_length):
                outputs = self(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=self.use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )

                past_key_values = outputs.past_key_values
                lm_logits = outputs.logits[:, -1, :]

                # Next token
                next_token = torch.argmax(lm_logits, dim=-1, keepdim=True)

                # Determine finished
                just_finished = next_token.squeeze(-1) == self.eos_token_id
                finished = finished | just_finished

                # Update input_ids
                next_token = torch.where(
                    condition=finished.unsqueeze(-1),
                    input=torch.full_like(next_token, self.eos_token_id),
                    other=next_token,
                )
                all_sequences = torch.cat([all_sequences, next_token], dim=1)
                input_ids = next_token

                # Update position_ids
                position_ids = position_ids[:, -1:] + 1
                position_ids = torch.where(
                    condition=finished.unsqueeze(-1),
                    input=torch.ones_like(position_ids),
                    other=position_ids,
                )

                # Update attention_mask
                new_attention_mask_column = torch.ones((batch_size, 1), device=input_ids.device, dtype=torch.long)
                attention_mask = torch.cat([attention_mask, new_attention_mask_column], dim=1)

                if finished.all():
                    break
            
            return all_sequences
        


if __name__ == "__main__":
    # Settings
    pretrained_model_name_or_path = "openai-community/gpt2"
    config = GPT2Config.from_pretrained_model_or_path(pretrained_model_name_or_path=pretrained_model_name_or_path)

    # Model & Tokenizer
    model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path).eval()
    custom_model = MyGPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path).eval()
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Test data
    sentences = [
        "Today is a nice day",
        "I want to go to play",
        "Hello",
        "Nice to meet you too",
    ]

    inputs = tokenizer(
        sentences,
        max_length=512,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )

    # Compare two model
    print("Logits:", custom_model(**inputs).logits == model(**inputs).logits)
    print("My Model:", "\n".join(tokenizer.batch_decode(custom_model.generate(**inputs))))
    print()
    print("HF Model:", "\n".join(tokenizer.batch_decode(model.generate(**inputs))))
