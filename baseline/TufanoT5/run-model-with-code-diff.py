from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration, get_linear_schedule_with_warmup, RobertaTokenizer

import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler, TensorDataset

import pandas as pd
from tqdm import tqdm
from difflib import SequenceMatcher
import sys, os, re, time, argparse
from itertools import cycle

import javalang

parser = argparse.ArgumentParser()


parser.add_argument("--output_dir", default='./from-pytorch/output/', type=str, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--proj", type=str)
parser.add_argument("--load_model_path", default=None, type=str,
                    help="Path to trained model: Should contain the .bin files")
## Other parameters
parser.add_argument("--train_file_dir", default=None, type=str,
                    help="The train folder.")
parser.add_argument("--eval_file_dir", default=None, type=str,
                    help="The dev folder.")
parser.add_argument("--test_file_dir", default=None, type=str,
                    help="The test folder.")

parser.add_argument("--max_source_length", default=512, type=int,
                    help="The maximum total source sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--max_target_length", default=512, type=int,
                    help="The maximum total target sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")

parser.add_argument("--do_train", action='store_true',
                    help="Whether to run training.")
parser.add_argument("--do_eval", action='store_true',
                    help="Whether to run eval on the dev set.")
parser.add_argument("--do_test", action='store_true',
                    help="Whether to run eval on the dev set.")

parser.add_argument("--no_cuda", action='store_true',
                    help="Avoid using CUDA when available")

parser.add_argument("--train_batch_size", default=6, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--eval_batch_size", default=32, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--learning_rate", default=5e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--beam_size", default=1, type=int,
                    help="beam size for beam search")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=3.0, type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--eval_steps", default=-1, type=int,
                    help="")
parser.add_argument("--train_steps", default=300000, type=int,
                    help="") # same as baseline...


parser.add_argument('--exp_name', default='no_name')
parser.add_argument('--last_train_step', type=int, default=0) # just for resuming model training
parser.add_argument('--selected_train_step', type=int)

args = parser.parse_args()

output_dir = args.output_dir+args.proj
exp_name = args.exp_name
last_train_step = args.last_train_step


extended_output_dir = os.path.join(output_dir, exp_name)
model_dir = os.path.join(extended_output_dir, 'model')
last_output_dir = os.path.join(model_dir, 'checkpoint-last')

os.makedirs(args.output_dir, exist_ok=True)    
os.makedirs(model_dir, exist_ok=True)
os.makedirs(extended_output_dir, exist_ok=True)
os.makedirs(last_output_dir, exist_ok=True)


def tokenize_java(code):
    token_gen = javalang.tokenizer.tokenize(code)
    tokens = []
    indexes = []
    while (True):
        try:
            token = next(token_gen)
        except:
            break
        tokens.append(token)

    pure_tokens = [token.value for token in tokens]

    return pure_tokens

def get_tokenized_code(c):
    try:
        return " ".join(tokenize_java(c))
    except:
        return c


def get_target(data_dir_path,file_suffix):
    with open(data_dir_path+'/approved_ver'+file_suffix+'.txt') as f:
        target = f.readlines()

    return target

def get_data(data_dir_path, file_suffix_init):

    with open(data_dir_path+'/initial_ver'+file_suffix_init+'.txt', encoding = "ISO-8859-1") as f:
        init_ver = f.readlines()

    with open(data_dir_path+'/approved_ver.txt', encoding = "ISO-8859-1") as f:
        approved_ver = f.readlines()


    all_source_ids, all_source_mask, all_target_ids, all_target_mask = [], [], [], []

    for (p, t) in tqdm(zip(init_ver, approved_ver)):
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



DEVICE = torch.device('cuda')
task = 'code2code: '

num_train_optimization_steps = args.train_steps
save_every_step = 2000

tokenizer_name = "./tokenizer/TokenizerModel.model"
config_name = "./model_dump/pre-training/pytorch_model/config.json"
model_name_or_path = './model_dump/pre-training/pytorch_model/pytorch_model.bin'

tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)


special_tokens_dict = {'additional_special_tokens': ['<START_MOD>','<END_MOD>']}


num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)


config = T5Config.from_pretrained(config_name)


if args.no_cuda:

    DEVICE = torch.device('cpu')
    os.environ["CUDA_VISIBLE_DEVICES"]='' # just for testing

else:
    DEVICE = torch.device('cuda')
    os.environ["CUDA_VISIBLE_DEVICES"]='1'

model = T5ForConditionalGeneration.from_pretrained(model_name_or_path, config=config).to(DEVICE)

model.resize_token_embeddings(len(tokenizer))

def get_model_input(batch):

    input_id = batch[0].to(DEVICE)
    input_attention_mask = batch[1].to(DEVICE)

    y = batch[2].to(DEVICE)
    output_attention_mask = batch[3].to(DEVICE)

    lm_labels = y.clone().detach()
    lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100

    return input_id, input_attention_mask, output_attention_mask, lm_labels


if args.do_train:

    train_data_dir = args.train_file_dir
    eval_data_dir = args.eval_file_dir

    train_data = get_data(train_data_dir, '_initial_ver_code_diff_info')
    eval_data = get_data(eval_data_dir, '_initial_ver_code_diff_info')

    train_data_tensor = TensorDataset(train_data[0], train_data[1], train_data[2], train_data[3])
    train_sampler = RandomSampler(train_data_tensor)

    train_dataloader = DataLoader(train_data_tensor, sampler=train_sampler, batch_size=args.train_batch_size // args.gradient_accumulation_steps, pin_memory=True)

    eval_data = TensorDataset(eval_data[0], eval_data[1], eval_data[2], eval_data[3])

    eval_sampler = SequentialSampler(eval_data)

    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, pin_memory=True)

    train_dataloader = cycle(train_dataloader)

    nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = 0, 0, 0, 1, 0, 1e6

    loss_dir = os.path.join(extended_output_dir, 'eval_loss.csv')

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-5, eps=1e-8)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_optimization_steps)

    train_losses = []
    valid_losses = []
    train_steps = []

    total_train_time = 0

    bar = range(1, num_train_optimization_steps+1)

    start_time = time.time()

    if last_train_step > 0:
        state_dict_path = os.path.join(last_output_dir, "pytorch_model-"+str(last_train_step)+"-steps.bin")
        print('loading checkpoint from',state_dict_path)

        checkpoint = torch.load(state_dict_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        total_train_time = checkpoint['total_train_time']

        loss_df = pd.read_csv(loss_dir)
        train_losses = loss_df['train_loss'].tolist()
        all_eval_perfect_match = loss_df['perfect_match'].tolist()
        train_steps = loss_df['train_step'].tolist()

        del checkpoint
    
    best_eval_perfect_match = 0
    early_stop_count = 0

    for step in tqdm(bar):

        batch = next(train_dataloader)

        if step <= last_train_step:
            global_step += 1
            continue

        model.train()

        input_id, input_attention_mask, output_attention_mask, lm_labels = get_model_input(batch)

        outputs = model(input_ids=input_id, attention_mask=input_attention_mask,labels=lm_labels, decoder_attention_mask=output_attention_mask)

        loss = outputs.loss
        tr_loss += loss.item()
        train_loss = tr_loss / (nb_tr_steps + 1)

        nb_tr_steps += 1

        loss.backward()
        optimizer.step()
        scheduler.step()

        model.zero_grad()
        optimizer.zero_grad()

        if global_step % save_every_step == 0:

            torch.cuda.empty_cache()

            model.eval()
            eval_loss, tokens_num = 0, 0
            nb_eval_step = 0

            predictions = []
            
            for batch in eval_dataloader:
                
                input_id, input_attention_mask, output_attention_mask, lm_labels = get_model_input(batch)

                with torch.no_grad():
                    outputs = model(input_ids=input_id, attention_mask=input_attention_mask,labels=lm_labels, decoder_attention_mask=output_attention_mask)

                loss = outputs.loss
                eval_loss += loss.item()

                nb_eval_step = nb_eval_step + 1

            final_eval_loss = eval_loss/nb_eval_step
            valid_losses.append(final_eval_loss)

            end_time = time.time()

            train_time = end_time - start_time

            total_train_time = total_train_time + train_time
            
            train_losses.append(train_loss)
            train_steps.append(global_step)

            loss_df = pd.DataFrame()
            loss_df['train_loss'] = train_losses
            loss_df['eval_loss'] = valid_losses
            loss_df['train_step'] = train_steps

            loss_df.to_csv(loss_dir, index=False)

            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
            output_model_file = os.path.join(last_output_dir, "pytorch_model-"+str(global_step)+"-steps.bin")

            # torch.save(model_to_save.state_dict(), output_model_file)

            torch.save({
                            'train_step': global_step,
                            'total_train_time': total_train_time,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()
                        }, output_model_file)

            text = 'train time: ' + str(total_train_time) + ' secs'

            with open(os.path.join(extended_output_dir, 'train_time.txt'),'w') as f:
                f.write(text)

            start_time = time.time() # for next round

        global_step += 1


if args.do_test:

    beam_size = args.beam_size
    print("Running Test")
    test_data_dir = args.test_file_dir

    selected_train_step = str(args.selected_train_step)

    saved_prediction_dir = os.path.join('./from-pytorch/output/',args.proj,exp_name)

    os.makedirs(saved_prediction_dir, exist_ok=True)

    print('saved prediction dir:',saved_prediction_dir)

    actual_model_dir = os.path.join(last_output_dir, 'pytorch_model-'+selected_train_step+'-steps.bin')

    print('abs path:', os.path.abspath(actual_model_dir))
        
    if args.no_cuda:
        checkpoint = torch.load(actual_model_dir, map_location=torch.device('cpu'))
        
    else:
        checkpoint = torch.load(actual_model_dir)

    model.load_state_dict(checkpoint['model_state_dict'])

    test_data = get_data(test_data_dir, '_init_diff_code_diff_representation')


    test_data_tensor = TensorDataset(test_data[0], test_data[1], test_data[2], test_data[3])

    test_sampler = SequentialSampler(test_data_tensor)
    test_dataloader = DataLoader(test_data_tensor, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)

    f_pred = open(os.path.join(saved_prediction_dir, "prediction-beam-"+str(beam_size)+".txt"), 'a+')

    for batch in tqdm(test_dataloader):

        input_id, input_attention_mask, _, _ =  get_model_input(batch)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_id,
                max_length=512,
                num_beams=beam_size,
                attention_mask=input_attention_mask,
                early_stopping=True,
                num_return_sequences=beam_size).to(DEVICE)

        prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for i in range(0,len(prediction)):
            f_pred.write(prediction[i].replace('\n','')+'\n')