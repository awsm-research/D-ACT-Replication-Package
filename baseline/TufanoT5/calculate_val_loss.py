from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
import torch
from tqdm import tqdm
import sys, os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class EvalDataset(torch.utils.data.Dataset):
    samples = []

    # filename: eval.tsv or test.tsv

    def __init__(self, data_dir_path, task, filename):

        self.samples = []

        df = pd.read_csv(data_dir_path + filename, sep='\t', names=['source', 'target'])
        source = df['source'].tolist()
        target = df['target'].tolist()

        f_source = open('test.source', 'w+')
        f_target = open('test.target', 'w+')
        for j in range(len(df)):
            f_source.write(task + source[j] + '\n')
            f_target.write(target[j] + '\n')
        f_source.close()
        f_target.close()

        input_file = open('test.source', 'r')
        output_file = open('test.target', 'r')

        lines_input = input_file.readlines()
        output_lines = output_file.readlines()
        print('data: ', len(lines_input))

        for (inp, out) in zip(lines_input, output_lines):
            self.samples.append((inp.rstrip(), out.rstrip()))

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

os.environ["CUDA_VISIBLE_DEVICES"]='0' 

DEVICE = torch.device('cuda')

beam_size = int(sys.argv[1])
batch_size = 8
task = 'code2code: '
data_dir = sys.argv[2] 
tokenizer_name = "./tokenizer/TokenizerModel.model"
model_name_or_path = sys.argv[3]
config_name = "./config.json"

filename = sys.argv[5]

dataset = EvalDataset(data_dir, task, filename)
dloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=0)

t5_tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)

t5_config = T5Config.from_pretrained(config_name)

# GENERATE PREDICTIONS
prediction_dir = sys.argv[4]
# f_pred = open(data_dir + '/predictions_' + str(beam_size) + '.txt', 'w+')

os.makedirs(prediction_dir, exist_ok=True)



all_loss = []

steps = list(np.arange(202000,400001,2000)) # without special tokens

step_list = []

for step in steps:

    eval_loss = 0
    eval_step = 0

    print(model_name_or_path+str(step)+'/pytorch_model.bin')

    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path+str(step)+'/pytorch_model.bin', config=t5_config).to(DEVICE)

    for batch in tqdm(dloader):
        encoded_input = t5_tokenizer.batch_encode_plus(batch[0], add_special_tokens=False, return_tensors='pt', padding='max_length', max_length=512, truncation=True)

        encoded_output = t5_tokenizer.batch_encode_plus(batch[1], add_special_tokens=False, return_tensors='pt', padding='max_length', max_length=512, truncation=True)

        input_ids = encoded_input['input_ids'].to(DEVICE)
        attention_mask = encoded_input['attention_mask'].to(DEVICE)

        output_ids = encoded_input['input_ids'].to(DEVICE)
        output_attention_mask = encoded_input['attention_mask'].to(DEVICE)

        lm_labels = output_attention_mask.clone().detach()
        lm_labels[lm_labels[:, :] == t5_tokenizer.pad_token_id] = -100

        with torch.no_grad():

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,labels=lm_labels, decoder_attention_mask=output_attention_mask)

        loss = outputs.loss
        eval_loss += loss.item()
        eval_step = eval_step+1

    final_loss = eval_loss/eval_step

    all_loss.append(final_loss)
    step_list.append(step)

    df = pd.DataFrame()
    df['step'] = step_list
    df['val_loss'] = all_loss

    df.to_csv(prediction_dir+'/val_loss.csv',index=False)


print(all_loss)

