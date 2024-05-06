import json
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from collections import defaultdict

import sys
sys.path.append("..")

from src.model.barcor.kg_bart import KGForBART
from src.model.barcor.barcor_model import BartForSequenceClassification

class TREA():

    def __init__(self, opt, dictionary,con2word,n_kind_set, is_finetune='mov', padding_idx=0, start_idx=1, end_idx=2, longest_label=1):

        super().__init__()
        self.classify=False
        self.device='cuda'
        self.dim=opt['dim']
        self.batch_size = opt['batch_size']
        self.max_r_length = opt['max_r_length']
        self.entity_max=opt['n_entity']
        self.NULL_IDX = padding_idx
        self.END_IDX = end_idx
        self.register_buffer('START', torch.LongTensor([start_idx]))
        self.longest_label = longest_label
        self.entity_label=[6] * 6924 + [5] * 12803 + [4] * 10707 + [1] * 18 + [0] * 7 + [2] * 12

        self.pad_idx = padding_idx
        self.embeddings = _create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )

        self.concept_embeddings=_create_entity_embeddings(
            opt['n_concept']+1, opt['dim'], 0)
        self.concept_padding=0

        self.kg=json.load(open('data/redial_kg.json'))

        if opt.get('n_positions'):
            # if the number of positions is explicitly provided, use that
            n_positions = opt['n_positions']
        else:
            # else, use the worst case from truncate
            n_positions = max(
                opt.get('truncate') or 0,
                opt.get('text_truncate') or 0,
                opt.get('label_truncate') or 0
            )
            if n_positions == 0:
                # default to 1024
                n_positions = 1024

        if n_positions < 0:
            raise ValueError('n_positions must be positive')

        self.encoder = _build_encoder(
            opt, dictionary, self.embeddings, self.pad_idx, reduction=False,
            n_positions=n_positions,
        )
        self.decoder = _build_decoder4kg(
            opt, dictionary, self.embeddings, self.pad_idx,
            n_positions=n_positions,
        )
        self.db_norm = nn.Linear(opt['dim'], opt['embedding_size'])
        self.kg_norm = nn.Linear(opt['dim'], opt['embedding_size'])

        self.db_attn_norm=nn.Linear(opt['dim'],opt['embedding_size'])
        self.kg_attn_norm=nn.Linear(opt['dim'],opt['embedding_size'])

        self.criterion1=nn.BCELoss(reduce=False)
        self.criterion = nn.CrossEntropyLoss(reduce=False)
        self.criterion2 = nn.CrossEntropyLoss(reduce=False,ignore_index=self.entity_max)
        self.criterion3=nn.CrossEntropyLoss(reduce=False,ignore_index=2)

        self.self_attn1 = SelfAttentionLayer_batch(opt['dim'], opt['dim'])
        self.self_attn2 = SelfAttentionLayer_batch(opt['dim'], opt['dim'])
        self.self_attn_emb1 = SelfAttentionLayer_batch(opt['dim'], opt['dim'])
        self.self_attn_emb2 = SelfAttentionLayer_batch(opt['dim'], opt['dim'])

        self.user_norm4gen=nn.Linear(opt['dim']*3,opt['dim'])
        self.output_en4gen=nn.Linear(opt['dim'],opt['n_entity'])

        self.self_attn_batch_db = SelfAttentionLayer_batch(opt['dim'],opt['dim'])
        self.self_attn_batch_his1 = SelfAttentionLayer_batch(opt['dim'], opt['dim'])
        self.self_attn_batch_his2 = SelfAttentionLayer_batch(opt['dim'], opt['dim'])
        self.self_attn_batch_hism1 = SelfAttentionLayer_batch(opt['dim'], opt['dim'])
        self.self_attn_batch_hism2 = SelfAttentionLayer_batch(opt['dim'], opt['dim'])
        self.self_attn_batch_leaf = SelfAttentionLayer_batch(opt['dim'], opt['dim'])

        self.self_attn_db = SelfAttentionLayer(opt['dim'], opt['dim'])
        self.self_attn_word=SelfAttentionLayer_batch(opt['dim'],opt['dim'])
        self.semantic_drag=nn.Sequential(
            nn.Linear(opt['embedding_size'],opt['dim']),
            nn.Tanh(),
            nn.Linear(opt['dim'],opt['embedding_size'])
        )
        self.semantic_drag_new = nn.Sequential(
            nn.Linear(opt['embedding_size'], opt['dim']),
            nn.Tanh(),
            nn.Linear(opt['dim'], opt['dim'])
        )
        self.self_attn_word_new = SelfAttentionLayer_batch(opt['dim'],opt['dim'])
        self.self_conattn=SelfAttentionLayer_batch(opt['dim'],opt['dim'])
        self.user_norm = nn.Linear(opt['dim']*2, opt['dim'])
        self.gate_norm = nn.Linear(opt['dim'], 1)
        self.copy_norm = nn.Linear(opt['embedding_size']*2+opt['embedding_size'], opt['embedding_size'])
        self.representation_bias = nn.Linear(opt['embedding_size'], len(dictionary) + 4)

        self.info_con_norm = nn.Sequential(nn.Linear(opt['dim'], opt['dim']),nn.LeakyReLU())
        self.info_db_norm = nn.Sequential(nn.Linear(opt['dim'], opt['dim']),nn.LeakyReLU())
        self.state_norm1 = nn.Sequential(nn.Linear(opt['dim']*24,opt['dim']*12),nn.LeakyReLU(),nn.Linear(opt['dim']*12,opt['dim']*24))
        self.state_norm2 = nn.Linear(opt['dim']*24, opt['dim'])
        self.con_word_norm=nn.Linear(opt['dim'],opt['embedding_size'])
        self.info_output_db = nn.Linear(opt['dim'], opt['n_entity'])
        self.info_output_con = nn.Linear(opt['dim'], opt['n_concept']+1)
        self.info_output_word=nn.Linear(opt['embedding_size'],self.embeddings.weight.shape[0])
        self.dim2dimSe=nn.Sequential(nn.Linear(self.dim,self.dim),nn.LeakyReLU())
        self.dim2dim = nn.Linear(self.dim,self.dim)
        self.info_con_loss = nn.MSELoss(size_average=False,reduce=False)
        self.info_db_loss = nn.MSELoss(size_average=False,reduce=False)
        self.score_loss = nn.MSELoss(size_average=False, reduce=False)

        self.user_representation_to_bias_1 = nn.Linear(opt['dim'], 512)
        self.user_representation_to_bias_2 = nn.Linear(512, len(dictionary) + 4)
        self.gate_norm_3_ele = GateLayer_3_eles(self.dim)
        self.gate_layer1 = GateLayer(self.dim)
        self.gate_layer2 = GateLayer(self.dim)
        self.gate_layer3 = GateLayer(self.dim)
        self.gate_layer_minus = GateLayer(self.dim,is_minus = True)
        self.output_en = nn.Linear(opt['dim'], opt['n_entity'])

        self.embedding_size=opt['embedding_size']
        self.dim=opt['dim']

        self.db_edge_idx,self.db_edge_type,self.n_relation=_edge_list(self.kg)
        # self.con_edge_idx,self.con_edge_type=concept_egde_list4GCN_new()
        self.con_edge_idx, self.con_edge_type,self.con_kind = concept_egde_list4GCN_newnew()
        self.con_re_emb = nn.Embedding(self.con_kind, self.dim)
        self.concept_relation=2
        self.con_classify = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Tanh(),
            nn.Linear(self.dim, self.concept_relation)
        )

        self.dbpedia_RGCN=RGCNConv(opt['n_entity'], self.dim, self.n_relation+1, num_bases=opt['num_bases'])
        self.concept_RGCN=RGCNConv(opt['n_concept']+1, self.dim, self.concept_relation, num_bases=opt['num_bases'])
        self.concept_edge_sets=concept_edge_list4GCN()
        self.concept_GCN=GCNConv(self.dim, self.dim)
        self.n_kind=6
        self.kind_emb=nn.Embedding(self.n_kind,self.embedding_size)
        self.wordfusekind=nn.Linear(self.embedding_size*2,self.embedding_size)

        w2i=json.load(open('word2index_redial.json',encoding='utf-8'))
        self.i2w={w2i[word]:word for word in w2i}
        self.con2word=torch.tensor(con2word,device=self.device)
        self.n_kind_set=n_kind_set
        self.stopwords_rd=torch.tensor([dictionary[word.strip()] for word in open('stopwords.txt', encoding='utf-8') if word in dictionary],device=self.device).long()
        self.emb2ent=nn.Sequential(
            nn.Linear(self.embedding_size,self.embedding_size//2),
            nn.ReLU(),
            nn.Linear(self.embedding_size//2,self.embedding_size)
        )
        self.ent2rec=nn.Linear(2*self.dim+self.embedding_size,self.dim)
        self.mask4=torch.ones(len(dictionary) + 4).to(self.device)


        params = [self.dbpedia_RGCN.parameters(), self.concept_GCN.parameters(),self.concept_embeddings.parameters()]

        for param in params:
            for pa in param:
                pa.requires_grad = False

    def _starts(self, bsz):
        """Return bsz start tokens."""
        return self.START.detach().expand(bsz, 1)

    def get_rec(self, conv_dict):
        
        # dataset
        text_list = []
        turn_idx = 0
        
        for utt in conv_dict['context']:
            if utt != '':
                text = ''
                if turn_idx % 2 == 0:
                    text += 'User: '
                else:
                    text += 'System: '
                text += utt
                text_list.append(text)
            turn_idx += 1
        
        context = f'{self.tokenizer.sep_token}'.join(text_list)
        context_ids = self.tokenizer.encode(context, truncation=True, max_length=self.context_max_length)

        data_list = []
        
        for rec in conv_dict['rec']:
            if rec in self.entity2id:
                data_dict = {
                    'context': context_ids,
                    'entity': [self.entity2id[ent] for ent in conv_dict['entity'] if ent in self.entity2id],
                    'rec': self.entity2id[rec]
                }
                if 'template' in conv_dict:
                    data_dict['template'] = conv_dict['template']
                data_list.append(data_dict)
        
        # dataloader
        input_dict = defaultdict(list)
        label_list = []
        
        for data in data_list:
            input_dict['input_ids'].append(data['context'])
            label_list.append(data['rec'])
        
        input_dict = self.tokenizer.pad(
            input_dict, max_length=self.context_max_length,
            padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of
        )
        
        input_dict['labels'] = label_list
        
        for k, v in input_dict.items():
            if not isinstance(v, torch.Tensor):
                input_dict[k] = torch.as_tensor(v, device=self.device)
        
        labels = input_dict['labels'].tolist()
        self.crs_rec_model.eval()
        outputs = self.crs_rec_model(**input_dict) 
        item_ids = torch.as_tensor(self.kg['item_ids'], device=self.device)
        logits = outputs['logits'][:, item_ids]
        ranks = torch.topk(logits, k=50, dim=-1).indices
        preds = item_ids[ranks].tolist()
        
        return preds, labels
    
    def get_conv(self, conv_dict):
        
        text_list = []
        turn_idx = 0
        for utt in conv_dict['context']:
            if utt != '':
                text = ''
                if turn_idx % 2 == 0:
                    text += 'User: '
                else:
                    text += 'System: '
                text += utt
                text_list.append(text)
            turn_idx += 1
        context = f'{self.tokenizer.sep_token}'.join(text_list)
        context_ids = self.tokenizer.encode(context, truncation=True, max_length=self.context_max_length)

        if turn_idx % 2 == 0:
            user_str = 'User: '
        else:
            user_str = 'System: '
        resp = user_str + conv_dict['resp']
        resp_ids = self.tokenizer.encode(resp, truncation=True, max_length=self.resp_max_length)

        data_dict = {
            'context': context_ids,
            'resp': resp_ids,
        }
        
        input_dict = defaultdict(list)
        label_dict = defaultdict(list)
        
        input_dict['input_ids'] = data_dict['context']
        label_dict['input_ids'] = data_dict['resp']
        
        input_dict = self.tokenizer.pad(
            input_dict, max_length=self.context_max_length,
            padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of
        )

        label_dict = self.tokenizer.pad(
            label_dict, max_length=self.context_max_length,
            padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of
        )['input_ids']
        
        # input_dict['labels'] = label_dict
        
        for k, v in input_dict.items():
            if not isinstance(v, torch.Tensor):
                input_dict[k] = torch.as_tensor(v, device=self.device).unsqueeze(0)
                
        self.crs_conv_model.eval()
        
        gen_args = {
            'min_length': 0,
            'max_length': self.resp_max_length,
            'num_beams': 1,
            'no_repeat_ngram_size': 3,
            'encoder_no_repeat_ngram_size': 3
        }
        
        gen_seqs = self.accelerator.unwrap_model(self.crs_conv_model).generate(**input_dict, **gen_args)
        gen_str = self.tokenizer.decode(gen_seqs[0], skip_special_tokens=True)
        
        return input_dict, gen_str
    
    def get_choice(self, gen_inputs, options, state, conv_dict=None):
        outputs = self.accelerator.unwrap_model(self.crs_conv_model).generate(
            **gen_inputs,
            min_new_tokens=5, max_new_tokens=5, num_beams=1,
            return_dict_in_generate=True, output_scores=True
        )
        option_token_ids = [self.tokenizer.encode(f" {op}", add_special_tokens=False)[0] for op in options]
        option_scores = outputs.scores[-2][0][option_token_ids]
        state = torch.as_tensor(state, device=self.device, dtype=option_scores.dtype)
        option_scores += state
        option_with_max_score = options[torch.argmax(option_scores)]
        
        return option_with_max_score