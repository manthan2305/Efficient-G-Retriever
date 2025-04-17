import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_scatter import scatter
import torch.nn.functional as F
from src.model.gnn import load_gnn_model
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'

IGNORE_INDEX = -100

class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, hidden_dim, num_heads, device):
        super().__init__()
        self.num_heads = num_heads
        self.attentions = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_heads)])
        self.device = device
        self.to(device)

    def forward(self, node_embeds, batch):
        node_embeds = node_embeds.to(self.device)
        batch = batch.to(self.device)

        all_graph_embeds = []
        for attention in self.attentions:
            attn_scores = attention(node_embeds).squeeze(-1)
            attn_scores = attn_scores - scatter(attn_scores, batch, dim=0, reduce='max')[batch]
            attn_scores = attn_scores.exp()
            attn_scores = attn_scores / (scatter(attn_scores, batch, dim=0, reduce='sum')[batch] + 1e-9)
            attn_scores = attn_scores.unsqueeze(-1)

            graph_embeds = scatter(node_embeds * attn_scores, batch, dim=0, reduce='sum')
            all_graph_embeds.append(graph_embeds)

        # Concatenate all head outputs
        graph_embeds = torch.cat(all_graph_embeds, dim=-1)
        return graph_embeds

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim, device):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        self.device = device
        self.to(device)

    def forward(self, node_embeds, batch):
        # Ensure inputs are on the correct device
        node_embeds = node_embeds.to(self.device)
        batch = batch.to(self.device)

        # Compute attention scores
        attn_scores = self.attention(node_embeds).squeeze(-1)
        
        # Compute softmax within each graph
        attn_scores = attn_scores - scatter(attn_scores, batch, dim=0, reduce='max')[batch]
        attn_scores = attn_scores.exp()
        attn_scores = attn_scores / (scatter(attn_scores, batch, dim=0, reduce='sum')[batch] + 1e-9)
        attn_scores = attn_scores.unsqueeze(-1)

        # Weighted sum of node embeddings
        graph_embeds = scatter(node_embeds * attn_scores, batch, dim=0, reduce='sum')

        return graph_embeds

class GraphLLM(torch.nn.Module):

    def __init__(
        self,
        args,
        **kwargs
    ):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        print('Loading LLAMA')
        kwargs = {
                "max_memory": {0: '40GiB', 1: '40GiB',  2: '40GiB',  3: '40GiB'},
            "device_map": "auto",
            "revision": "main",
        }

        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **kwargs
        )

        if args.llm_frozen == 'True':
            print("Freezing LLAMA!")
            for name, param in model.named_parameters():
                param.requires_grad = False
        else:
            print("Training LLAMA with LORA!")
            model = prepare_model_for_kbit_training(model)
            lora_r: int = 8
            lora_alpha: int = 16
            lora_dropout: float = 0.05
            lora_target_modules = [
                "q_proj",
                "v_proj",
            ]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

            # Reducing memory usage by recomputing certain gradients during backprop
            model.gradient_checkpointing_enable()

        self.model = model
        print('Finish loading LLAMA!')

        self.graph_encoder = load_gnn_model[args.gnn_model_name](
            in_channels=args.gnn_in_dim,
            out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(self.model.device)

        self.projector = nn.Sequential(
            nn.Linear(4096, 8192),
            nn.GELU(),                                  # Replace with sigmoid
            nn.LayerNorm(8192),                         # Added New LayerNorm
            nn.Linear(8192, 4096),
        ).to(self.model.device)

        self.word_embedding = self.model.model.get_input_embeddings()

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def encode_graphs(self, samples):
        graphs = samples['graph']
        graphs = graphs.to(self.model.device)
        # n_embeds, _ = self.graph_encoder(graphs.x, graphs.edge_index.long(), graphs.edge_attr)
        n_embeds, e_embeds = self.graph_encoder(graphs.x, graphs.edge_index.long(), graphs.edge_attr)   # consider node and edge embeddings

        # mean pooling
        # g_embeds = scatter(n_embeds, graphs.batch, dim=0, reduce='mean')

        # If e_embeds are available, we will combine
        row, col = graphs.edge_index
        node_edge_agg = torch.zeros_like(n_embeds)

        # Aggregate incoming edge features
        node_edge_agg.index_add_(0, row, e_embeds)

        # combined node and edge features
        combined_embeds = n_embeds + node_edge_agg

        # Attention Pooling
        if not hasattr(self, 'attention_pool'):         # Initialize attention pooling layer once
            self.attention_pool = MultiHeadAttentionPooling(hidden_dim=combined_embeds.size(-1), num_heads=4, device=self.model.device)
            
        g_embeds = self.attention_pool(combined_embeds, graphs.batch)

        return g_embeds

    def forward(self, samples):

        # encode description, questions and labels
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)

        # encode special tokens
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0])
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)

        # encode graphs
        graph_embeds = self.encode_graphs(samples)
        graph_embeds = self.projector(graph_embeds)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):
            # Add bos & eos token
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
            input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0]-len(label_input_ids))+label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length+batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss

    def inference(self, samples):

        # encode description and questions
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)

        # encode special tokens
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0])
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)

        # encode graphs
        graph_embeds = self.encode_graphs(samples)
        graph_embeds = self.projector(graph_embeds)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            # Add bos & eos token
            input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                # do_sample=True,
                use_cache=True  # IMPORTANT!
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {'id': samples['id'],
                'pred': pred,
                'label': samples['label'],
                'question': samples['question'],
                'desc': samples['desc'], }

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
