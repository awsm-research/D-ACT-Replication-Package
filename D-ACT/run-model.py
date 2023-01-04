from utils import *

from transformers import get_linear_schedule_with_warmup

import pandas as pd
from tqdm import tqdm
import time, argparse, os

parser = argparse.ArgumentParser()


parser.add_argument("--proj_name", type=str)
parser.add_argument("--model_type", type=str)
parser.add_argument("--use_special_tokens", action='store_true')
parser.add_argument("--output_dir", default='./output/', type=str, help="The output directory where the model predictions and checkpoints will be written.")

parser.add_argument("--train_input_file_path", default=None, type=str)
parser.add_argument("--train_gt_file_path", default=None, type=str)
parser.add_argument("--eval_input_file_path", default=None, type=str)
parser.add_argument("--eval_gt_file_path", default=None, type=str)
parser.add_argument("--test_input_file_path", default=None, type=str)
parser.add_argument("--test_gt_file_path", default=None, type=str)

parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
parser.add_argument("--do_test", action='store_true', help="Whether to run eval on the dev set.")

parser.add_argument("--no_cuda_flag", action='store_true', help="Avoid using CUDA when available")
parser.add_argument("--gpu_num", default='0', type = str)

parser.add_argument("--train_batch_size", default=6, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--eval_batch_size", default=16, type=int,
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

parser.add_argument("--train_steps", default=300000, type=int) 
parser.add_argument('--exp_name', default='no_name')
parser.add_argument('--last_train_step', type=int, default=0) # just for resuming model training
parser.add_argument('--selected_train_step', type=int, default=0)


args = parser.parse_args()
    # logger.info(args)

## show arguments to the screen
print()
for arg in vars(args): 
    print('{: <30} {}'.format(arg, getattr(args, arg)))
    # print(arg, ':', getattr(args, arg)) 

print('-'*80)

proj_name = args.proj_name

train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
last_train_step = args.last_train_step
beam_size = args.beam_size
ckpt = args.selected_train_step

num_train_optimization_steps = args.train_steps
save_every_step = 2000 # 2000

train_input_file_path = args.train_input_file_path
train_gt_file_path = args.train_gt_file_path
eval_input_file_path = args.eval_input_file_path
eval_gt_file_path = args.eval_gt_file_path
test_input_file_path = args.test_input_file_path
test_gt_file_path = args.test_gt_file_path

model_type = args.model_type.lower()
use_special_tokens = args.use_special_tokens

train_flag = args.do_train
test_flag = args.do_test

no_cuda_flag = args.no_cuda_flag
gpu_num = args.gpu_num

if model_type not in ['codebert', 'graphcodebert', 'plbart', 'codet5']:
    print('wrong model type...')
    print('the supported models are CodeBERT, GraphCodeBERT, PLBART, CodeT5')
    exit(0)

if proj_name is None:
    print('project name is missing')
    print('project name is either android, google, or ovirt')
    exit(0)


if train_flag and test_flag:
    print('both train and test modes are activated')
    print('choose either train or test mode')
    exit(0)
elif not train_flag and not test_flag:
    print('select either train or test mode')
    exit(0)

output_dir = args.output_dir+proj_name
exp_name = args.exp_name

extended_output_dir = os.path.join(output_dir, exp_name)
model_dir = os.path.join(extended_output_dir, 'model')

os.makedirs(args.output_dir, exist_ok=True)    
os.makedirs(model_dir, exist_ok=True)
os.makedirs(extended_output_dir, exist_ok=True)

if no_cuda_flag:

    DEVICE = torch.device('cpu')
    os.environ["CUDA_VISIBLE_DEVICES"]='' # just for testing

else:
    DEVICE = torch.device('cuda')
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num


print('running model based on',model_type)

if use_special_tokens:
    print('run model with special tokens')

tokenizer = load_tokenizer(model_type, use_special_tokens = use_special_tokens)

model = load_model(model_type, tokenizer, model_dir = model_dir, ckpt_num = ckpt, beam_size = beam_size, use_special_tokens = use_special_tokens, no_cuda = no_cuda_flag, DEVICE=DEVICE)

if train_flag:

    if None in [train_input_file_path, train_gt_file_path, eval_input_file_path, eval_gt_file_path]:
        print('input or output file(s) are missing')
        exit(0)

    print('loading training data')

    train_dataloader = load_data(model_type, train_input_file_path, train_gt_file_path, tokenizer, 'train', train_batch_size)

    print('loading validation data')
    
    eval_dataloader = load_data(model_type, eval_input_file_path, eval_gt_file_path, tokenizer, 'eval', eval_batch_size)

    nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = 0, 0, 0, 1, 0, 1e6

    loss_dir = os.path.join(extended_output_dir, 'loss.csv')

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-5, eps=1e-8)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_optimization_steps)

    train_losses, valid_losses, train_steps = [], [], []

    total_train_time = 0

    bar = range(1, num_train_optimization_steps+1)

    print('training',model_type)

    start_time = time.time()
    
    best_eval_perfect_match = 0
    early_stop_count = 0


    for step in tqdm(bar):
        # remove_old_models(last_output_dir)

        batch = next(train_dataloader)

        if step <= last_train_step:
            global_step += 1
            continue

        model.train()

        input_dict = get_model_input(batch)
 
        if model_type in ['codebert','graphcodebert']:
            loss, _, __ = model(**input_dict)
            loss_num = loss.item()
        else:
            outputs = model(**input_dict)
            loss = outputs.loss
            loss_num = loss.item()

        tr_loss += loss_num
        train_loss = tr_loss / (nb_tr_steps + 1)

        nb_tr_steps += 1

        loss.backward()
        optimizer.step()
        scheduler.step()

        model.zero_grad()
        optimizer.zero_grad()

        if global_step % save_every_step == 0:

            del input_dict

            torch.cuda.empty_cache()

            # Start Evaling model
            model.eval()
            eval_loss, tokens_num = 0, 0
            nb_eval_step = 0
            predictions = []

            for batch in eval_dataloader:
                # get model's input here

                input_dict = get_model_input(batch)

                with torch.no_grad():

                    if model_type in ['codebert','graphcodebert']:
                        _,loss,num  = model(**input_dict)

                        eval_loss += loss.sum().item()
                        tokens_num += num.sum().item()

                    else:
                        outputs = model(**input_dict)
                        loss = outputs.loss
                        eval_loss += loss.item()

                nb_eval_step = nb_eval_step + 1

                if model_type in ['codebert','graphcodebert']:
                    final_eval_loss = eval_loss / tokens_num
                else:
                    final_eval_loss = eval_loss/nb_eval_step

            final_eval_loss = eval_loss/nb_eval_step

            end_time = time.time()

            train_time = end_time - start_time

            total_train_time = total_train_time + train_time
            
            train_losses.append(train_loss)
            train_steps.append(global_step)
            valid_losses.append(final_eval_loss)

            loss_df = pd.DataFrame()
            loss_df['train_loss'] = train_losses
            loss_df['valid_loss'] = valid_losses
            loss_df['train_step'] = train_steps

            loss_df.to_csv(loss_dir, index=False)

            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
            output_model_file = os.path.join(model_dir, "pytorch_model-"+str(global_step)+"-steps.bin")

            print('save model to',output_model_file)

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


if test_flag:

    if None in [test_input_file_path, test_gt_file_path]:
        print('input or output file(s) are missing')
        exit(0)

    print("Running Test")
    print('loading test data')

    test_dataloader = load_data(model_type, test_input_file_path, test_gt_file_path, tokenizer, 'test', eval_batch_size)

    saved_prediction_dir = os.path.join('./output/',proj_name,exp_name)

    f_pred = open(os.path.join(extended_output_dir, "prediction-beam-"+str(beam_size)+".txt"), 'a+')

    model.eval()

    print('generating prediction of',model_type)

    for batch in tqdm(test_dataloader):
        # get model's input here

        input_dict = get_model_input(batch)

        with torch.no_grad():
            if model_type in ['codebert','graphcodebert']:

                prediction = []

                preds = model(**input_dict)

                for pred in preds:

                    for i in range(0,beam_size):

                        t=pred[i].cpu().numpy()
                        t=list(t)
                        if 0 in t:
                            t=t[:t.index(0)]
                            
                        text = tokenizer.decode(t,skip_special_tokens=True)

                        f_pred.write(text.replace('\n','')+'\n')
            
            else:
                outputs = model.generate(**input_dict,
                    max_length=512,  
                    num_beams=beam_size,
                    early_stopping=True,
                    num_return_sequences=beam_size).to(DEVICE)

                prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                for i in range(0,len(prediction)):
                    f_pred.write(prediction[i].replace('\n','')+'\n')
