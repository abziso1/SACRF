from torch import nn
import torch
from typing import Tuple, List, Dict
from qutils import *
from abc import ABC, abstractmethod
from tqdm import tqdm

class KBCModel(nn.Module, ABC):
    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1,
            verbose: bool = True
    ):
        ranks = torch.ones(len(queries))
        with tqdm(total=queries.shape[0], unit='ex', disable=not verbose) as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    target_idxs = these_queries[:, 2].cpu().tolist()
                    scores, _, _ = self.forward(these_queries)
                    targets = torch.stack([scores[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())] + [queries[b_begin + i, 2].item()]
                        scores[i, torch.as_tensor(filter_out, dtype=torch.long, device=scores.device)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()
                    b_begin += batch_size
                    bar.update(batch_size)
        return ranks

class BiQUEScorer(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def forward(self, head, relation, all):
        rel_add = relation[:, self.rank * 8:]
        rel_mul = relation[:, :self.rank * 8]
        
        head += rel_add
        
        w_a, x_a, y_a, z_a = torch.chunk(head, 4, dim=-1)
        w_b, x_b, y_b, z_b = torch.chunk(rel_mul, 4, dim=-1)
        
        A = complex_mul(w_a,w_b) - complex_mul(x_a,x_b) - complex_mul(y_a,y_b) - complex_mul(z_a,z_b)
        B = complex_mul(w_a,x_b) + complex_mul(x_a,w_b) + complex_mul(y_a,z_b) - complex_mul(z_a,y_b)
        C = complex_mul(w_a,y_b) - complex_mul(x_a,z_b) + complex_mul(y_a,w_b) + complex_mul(z_a,x_b)
        D = complex_mul(w_a,z_b) + complex_mul(x_a,y_b) - complex_mul(y_a,x_b) + complex_mul(z_a,w_b)
        
        res = torch.cat([A,B,C,D], dim=-1)
        score = res @ all.T
        return score
        

class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
    
    def forward(self, x):
        return self.encoder(x)

class SACRF(KBCModel):
    def __init__(
        self, sizes: Tuple[int, int, int], srank: int, vrank: int, trank: int, gate_dim:int, alpha_max: float = 0.2,
        img_features: torch.FloatTensor = None, text_features: torch.FloatTensor = None
    ):

        super().__init__()
        
        self.srank = srank
        self.vrank = vrank
        self.trank = trank
        
        nentity, nrelation, _ = sizes

        img_dim = 8 * vrank
        self.img_embeddings = nn.Embedding.from_pretrained(img_features).requires_grad_(False)
        self.img_encoder = ModalityEncoder(img_features.shape[1],  img_dim)
        self.img_relation_embeddings = nn.Embedding(nrelation, 2 * img_dim)
        self.img_scorer = BiQUEScorer(vrank)
        
        text_dim = 8 * trank
        self.text_embeddings = nn.Embedding.from_pretrained(text_features).requires_grad_(False)
        self.text_encoder = ModalityEncoder(text_features.shape[1], text_dim)
        self.text_relation_embeddings = nn.Embedding(nrelation, 2 * text_dim)
        self.text_scorer = BiQUEScorer(trank)

        self.entity_embeddings = nn.Embedding(nentity, srank * 8)
        self.relation_embeddings = nn.Embedding(nrelation, srank * 16)
        self.stru_scorer = BiQUEScorer(srank)
        
        self.gate_dim = gate_dim
        self.gate_relation = nn.Embedding(nrelation, gate_dim)
        self.img_gate = nn.Linear(gate_dim, 1)
        self.text_gate = nn.Linear(gate_dim, 1)
        
        self.alpha_max = alpha_max
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
        nn.init.xavier_uniform_(self.img_relation_embeddings.weight.data)
        nn.init.xavier_uniform_(self.text_relation_embeddings.weight)

        for encoder in (self.img_encoder, self.text_encoder):
            for module in encoder.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
       
    @staticmethod
    def normalize_score(score):
        mean = score.mean(dim=-1, keepdim=True)
        std = score.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (score - mean) / std        
    
    def forward(self, x):
        h_ids = x[:, 0]
        r_ids = x[:, 1]
        t_ids = x[:, 2]
        
        all_s = self.entity_embeddings.weight
        head_s = self.entity_embeddings(h_ids)
        relation_s = self.relation_embeddings(r_ids)
        tail_s = self.entity_embeddings(t_ids)
        score_s = self.stru_scorer(head_s, relation_s, all_s)
        factors_s = [(get_norm(head_s, 8), get_norm(relation_s[:, :self.srank*8], 8), get_norm(tail_s, 8))]

        all_v = self.img_encoder(self.img_embeddings.weight)
        head_v = all_v[h_ids]
        relation_v = self.img_relation_embeddings(r_ids)
        score_v = self.img_scorer(head_v, relation_v, all_v)
        
        all_t = self.text_encoder(self.text_embeddings.weight)
        head_t = all_t[h_ids]
        relation_t = self.text_relation_embeddings(r_ids)
        score_t = self.text_scorer(head_t, relation_t, all_t)
        
        gate_rel = self.gate_relation(r_ids) # [B, modal_dim]
        alpha_v = self.alpha_max * torch.sigmoid(
            self.img_gate(gate_rel)
        )                                                # [B, 1]

        alpha_t = self.alpha_max * torch.sigmoid(
            self.text_gate(gate_rel)
        )                                                # [B, 1]

        struct_scale = score_s.detach().std(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)

        delta_v = self.normalize_score(score_v) * struct_scale
        delta_t = self.normalize_score(score_t) * struct_scale

        score_final = (
            score_s
            + alpha_v * delta_v
            + alpha_t * delta_t
        )
        
    
        branch_scores = {
            "stru": score_s,
            "img": score_v,
            "text": score_t
        }
        return score_final, factors_s, branch_scores
