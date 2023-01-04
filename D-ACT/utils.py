from itertools import cycle
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, PLBartTokenizer, PLBartForConditionalGeneration, T5ForConditionalGeneration 


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler, TensorDataset

from seq2seq_models import *

from parser import DFG_java, remove_comments_and_docstrings, tree_to_token_index, index_to_code_token

from tree_sitter import Language, Parser

from tqdm import tqdm
import numpy as np
import os

## read input

## for all models


### for codebert
class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

class InputFeatures_CodeBERT(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask  

def read_examples(input_file_path, gt_file_path):
    """Read examples from filename."""
    examples = []

    idx = 0

    with open(input_file_path, encoding = "ISO-8859-1") as f:
        patch1 = f.readlines()
    
    with open(gt_file_path, encoding = "ISO-8859-1") as f:
        targets = f.readlines()
    
    for input, target in zip(patch1, targets):
        examples.append(
            Example(idx, input, target)
        )
        idx += 1

        # just for testing
        # if idx >= 5:
        #     break

    return examples

def convert_examples_to_features_for_CodeBERT(examples, tokenizer, max_source_length = 512, max_target_length = 512, stage=None):
    features = []

    for example_index, example in tqdm(enumerate(examples)):
                #source
        source_tokens = tokenizer.tokenize(example.source)[:max_source_length-2]
        source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
        source_mask = [1] * (len(source_tokens))
        padding_length = max_source_length - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        source_mask+=[0]*padding_length
 
        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:max_target_length-2]
        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length =max_target_length - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length   

        features.append(
            InputFeatures_CodeBERT(
                 example_index,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
            )
        )

        # just for testing
        # if example_index >= 500:
        #     break

    return features

### for Graphcodebert

class InputFeatures_GraphCodeBERT(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,                 
                 target_ids,
                 source_mask,
                 target_mask,

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.position_idx = position_idx
        self.dfg_to_code = dfg_to_code
        self.dfg_to_dfg = dfg_to_dfg
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask       
        
class TextDataset(Dataset):
    def __init__(self, examples, max_source_length):
        self.examples = examples
        self.max_source_length = max_source_length
        # self.args=args  
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):
        #calculate graph-guided masked function
        attn_mask=np.zeros((self.max_source_length,self.max_source_length),dtype=np.bool)
        #calculate begin index of node and max length of input
        node_index=sum([i>1 for i in self.examples[item].position_idx])
        max_length=sum([i!=1 for i in self.examples[item].position_idx])
        #sequence can attend to sequence
        attn_mask[:node_index,:node_index]=True
        #special tokens attend to all tokens
        for idx,i in enumerate(self.examples[item].source_ids):
            if i in [0,2]:
                attn_mask[idx,:max_length]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(self.examples[item].dfg_to_code):
            if a<node_index and b<node_index:
                attn_mask[idx+node_index,a:b]=True
                attn_mask[a:b,idx+node_index]=True
        #nodes attend to adjacent nodes         
        for idx,nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a+node_index<len(self.examples[item].position_idx):
                    attn_mask[idx+node_index,a+node_index]=True  
                    
        return (torch.tensor(self.examples[item].source_ids),
                torch.tensor(self.examples[item].source_mask),
                torch.tensor(self.examples[item].position_idx),
                torch.tensor(attn_mask), 
                torch.tensor(self.examples[item].target_ids),
                torch.tensor(self.examples[item].target_mask),)

dfg_function={
    'java':DFG_java
}

parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser

#remove comments, tokenize code and extract dataflow     
def extract_dataflow(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens,dfg


def convert_examples_to_features_for_GraphCodeBERT(examples, tokenizer, max_source_length = 512, max_target_length = 512,stage=None):
    features = []
    # examples = examples[:100] # just for testing

    for example_index, example in enumerate(tqdm(examples,total=len(examples))):
        ##extract data flow
        code_tokens,dfg=extract_dataflow(example.source,parsers["java"],"java")
        code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]

        ori2cur_pos={}
        ori2cur_pos[-1]=(0,0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
        code_tokens=[y for x in code_tokens for y in x]  
        
        #truncating
        code_tokens=code_tokens[:max_source_length-3][:512-3]
        source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
        dfg=dfg[:max_source_length-len(source_tokens)]
        source_tokens+=[x[0] for x in dfg]
        position_idx+=[0 for x in dfg]
        source_ids+=[tokenizer.unk_token_id for x in dfg]
        padding_length=max_source_length-len(source_ids)
        position_idx+=[tokenizer.pad_token_id]*padding_length
        source_ids+=[tokenizer.pad_token_id]*padding_length      
        source_mask = [1] * (len(source_tokens))
        source_mask+=[0]*padding_length        
        
        #reindex
        reverse_index={}
        for idx,x in enumerate(dfg):
            reverse_index[x[1]]=idx
        for idx,x in enumerate(dfg):
            dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
        dfg_to_dfg=[x[-1] for x in dfg]
        dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
        length=len([tokenizer.cls_token])
        dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]        
      

        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:max_target_length-2]
        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = max_target_length - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length   
       
        features.append(
            InputFeatures_GraphCodeBERT(
                 example_index,
                 source_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,
                 target_ids,
                 source_mask,
                 target_mask,
            )
        )
    return features


### for PLBART/CodeT5
def get_data(input_file_path, gt_file_path, tokenizer):

    with open(input_file_path) as f:
        init_ver = f.readlines()

    with open(gt_file_path) as f:
        approved_ver = f.readlines()

    all_source_ids, all_source_mask, all_target_ids, all_target_mask = [], [], [], []

    # get all_new_patch1, target from file

    for (p, t) in tqdm(zip(init_ver, approved_ver)):
        # print('string:',p)
        encoded_input = tokenizer.batch_encode_plus([p], add_special_tokens=True, return_tensors='pt', padding='max_length', max_length=512, truncation=True)
        
        encoded_output = tokenizer.batch_encode_plus([t], add_special_tokens=True, return_tensors='pt', padding='max_length', max_length=512, truncation=True)
        
        source_ids = encoded_input["input_ids"].squeeze().tolist()
        source_mask = encoded_input["attention_mask"].squeeze().tolist()
        target_ids = encoded_output["input_ids"].squeeze().tolist()
        target_mask = encoded_output["attention_mask"].squeeze().tolist()

        all_source_ids.append(source_ids)
        all_source_mask.append(source_mask)
        all_target_ids.append(target_ids)
        all_target_mask.append(target_mask)

    all_source_ids = torch.tensor(all_source_ids, dtype=torch.long)
    all_source_mask = torch.tensor(all_source_mask, dtype=torch.long)
    all_target_ids = torch.tensor(all_target_ids, dtype=torch.long)
    all_target_mask = torch.tensor(all_target_mask, dtype=torch.long)

    return (all_source_ids, all_source_mask, all_target_ids, all_target_mask)

def load_data(model_type, input_file_path, gt_file_path, tokenizer, stage, batch_size, gradient_accumulation_steps = 1, max_source_length = 512, max_target_length = 512):

    if model_type == 'codebert':
        print('loading data for CodeBERT')

        examples = read_examples(input_file_path, gt_file_path)
        features = convert_examples_to_features_for_CodeBERT(examples, tokenizer, max_source_length = max_source_length, max_target_length = max_target_length, stage = stage)

        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in features], dtype=torch.long)

        all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in features], dtype=torch.long)  

        data_tensor = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)

    elif model_type == 'graphcodebert':
        print('loading data for GraphCodeBERT')

        examples = read_examples(input_file_path, gt_file_path)
        features = convert_examples_to_features_for_GraphCodeBERT(examples, tokenizer,max_source_length = max_source_length, max_target_length = max_target_length, stage = stage)

        data_tensor = TextDataset(features,max_source_length=max_source_length)


    else:   # for PLBART/CodeT5
        print('loading data for',model_type)
        data = get_data(input_file_path, gt_file_path, tokenizer)
        data_tensor = TensorDataset(data[0], data[1], data[2], data[3])

    if stage == 'train':
        sampler = RandomSampler(data_tensor)
    else:
        sampler = SequentialSampler(data_tensor)

    dataloader = DataLoader(data_tensor, sampler=sampler, batch_size=batch_size //gradient_accumulation_steps)

    if stage == 'train':
        dataloader = cycle(dataloader)

    return dataloader


pre_trained_model_dirs = {
    'codebert': 'microsoft/codebert-base', 
    'graphcodebert': 'microsoft/graphcodebert-base',
    'plbart': 'uclanlp/plbart-base',
    'codet5': 'Salesforce/codet5-base'
}

def load_model(model_type, tokenizer, model_dir = '', ckpt_num = 0, beam_size = 1, max_target_length = 512, DEVICE = torch.device('cuda'), use_special_tokens = False, no_cuda = False):
    
    # print('loading model of', model_type)
    
    if ckpt_num == 0:
        print('loading pre-trained model')
    else:
        print('loading fine-tuned model')

    HUGGINGFACE_REPO = pre_trained_model_dirs[model_type]

    if model_type == 'codebert':

        config = RobertaConfig.from_pretrained(HUGGINGFACE_REPO)

        encoder = RobertaModel.from_pretrained(HUGGINGFACE_REPO, config=config)

        if use_special_tokens:
            encoder.resize_token_embeddings(len(tokenizer))

        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)

        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        model = Seq2Seq_CB(encoder=encoder, decoder=decoder, config=config,
                            beam_size=beam_size, max_length=max_target_length,
                            sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
    elif model_type == 'graphcodebert':

        config = RobertaConfig.from_pretrained(HUGGINGFACE_REPO)

        if use_special_tokens:
            encoder.resize_token_embeddings(len(tokenizer))

        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        model = Seq2Seq_GCB(encoder=encoder, decoder=decoder, config=config,
                            beam_size=beam_size, max_length=max_target_length,
                            sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)

    elif model_type == 'plbart':
        model = PLBartForConditionalGeneration.from_pretrained(HUGGINGFACE_REPO)

        if use_special_tokens:
            model.resize_token_embeddings(len(tokenizer))


    else:
        model = T5ForConditionalGeneration.from_pretrained(HUGGINGFACE_REPO)#.to(DEVICE)

        if use_special_tokens:
            model.resize_token_embeddings(len(tokenizer))

    # print('device', DEVICE)
    model.to(DEVICE)

    if model_dir != '' and ckpt_num > 0:
        actual_model_dir = os.path.join(model_dir, 'pytorch_model-'+str(ckpt_num)+'-steps.bin')

        if not os.path.exists(actual_model_dir):
            print('model is not found')
            exit(0)

        print('loading model from',actual_model_dir)

        if no_cuda:
            checkpoint = torch.load(actual_model_dir, map_location=torch.device('cpu'))
        
        else:
            checkpoint = torch.load(actual_model_dir)

        model.load_state_dict(checkpoint['model_state_dict'])

        del checkpoint

        # pass # load checkpoint here

    return model


special_tokens_dict = {'additional_special_tokens': ['<START_MOD>','<END_MOD>']}

def load_tokenizer(model_type, use_special_tokens = False):
    # print('loading tokenizer of', model_type)
    
    HUGGINGFACE_REPO = pre_trained_model_dirs[model_type]

    print('loading tokenizer of',model_type)

    if model_type != 'plbart':
        tokenizer = RobertaTokenizer.from_pretrained(HUGGINGFACE_REPO)
    else:
        tokenizer = PLBartTokenizer.from_pretrained(HUGGINGFACE_REPO, language_codes="base")

    if use_special_tokens:
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    return tokenizer


def get_model_input(batch):

    if model_type == 'codebert':
        batch = tuple(t.to(DEVICE) for t in batch)
        source_ids,source_mask,target_ids,target_mask = batch

        if train_flag:
            input_dict = {
                'source_ids': source_ids,
                'source_mask': source_mask,
                'target_ids': target_ids,
                'target_mask': target_mask
            }
        elif test_flag:
            input_dict = {
                'source_ids': source_ids,
                'source_mask': source_mask
            }

    elif model_type == 'graphcodebert':
        batch = tuple(t.to(DEVICE) for t in batch)
        source_ids,source_mask,position_idx,att_mask,target_ids,target_mask = batch

        if train_flag:
            input_dict = {
                'source_ids': source_ids,
                'source_mask': source_mask,
                'target_ids': target_ids,
                'target_mask': target_mask,
                'position_idx': position_idx,
                'att_mask': att_mask
            }
        elif test_flag:
            input_dict = {
                'source_ids': source_ids,
                'source_mask': source_mask,
                'position_idx': position_idx,
                'att_mask': att_mask
            }

    else:
        input_id = batch[0].to(DEVICE)
        input_attention_mask = batch[1].to(DEVICE)

        y = batch[2].to(DEVICE)
        output_attention_mask = batch[3].to(DEVICE)

        lm_labels = y.clone().detach()
        lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100

        if train_flag:
            input_dict = {
                'input_ids': input_id,
                'attention_mask': input_attention_mask,
                'labels': lm_labels,
                'decoder_attention_mask': output_attention_mask
            }
        elif test_flag:
            input_dict = {
                'input_ids': input_id,
                'attention_mask': input_attention_mask
            }

    return input_dict