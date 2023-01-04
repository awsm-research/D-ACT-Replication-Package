# TODO : check file paths obtained from sys.argv[...]


from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
import torch
from tqdm import tqdm
import sys, os
from torch.utils.data import Dataset, DataLoader
import pandas as pd

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

## run model on cpu (take long time but it's ok)

os.environ["CUDA_VISIBLE_DEVICES"]='0' # just for testing in m4 server

DEVICE = torch.device('cuda')
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu',)  # GPU recommended
# DEVICE = torch.device('cpu') # just for testing

# beam_size = 1
beam_size = int(sys.argv[1])
batch_size = 4 # default is 10
task = 'code2code: '  # possible options: 'code2code: ', 'code&comment2code: ', 'code2comment: '
# data_dir = "../../dataset/fine-tuning/large/code-to-code/"  # change the path if needed
data_dir = sys.argv[2] # something like this??  ../../combined-method-pairs/android/small/test/ 
tokenizer_name = "./tokenizer/TokenizerModel.model"
# model_name_or_path = "./dumps/pytorch_model.bin" # may change later
model_name_or_path = sys.argv[3]
config_name = "./config.json"

filename = sys.argv[5]

dataset = EvalDataset(data_dir, task, filename)
dloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=0)

t5_tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)

t5_config = T5Config.from_pretrained(config_name)
t5_mlm = T5ForConditionalGeneration.from_pretrained(model_name_or_path, config=t5_config).to(DEVICE)

# GENERATE PREDICTIONS
prediction_dir = sys.argv[4]
# f_pred = open(data_dir + '/predictions_' + str(beam_size) + '.txt', 'w+')

os.makedirs(prediction_dir, exist_ok=True)

f_pred = open(os.path.join(prediction_dir, 'prediction_beam_'+ str(beam_size) + '.txt'), 'a+')
predictions = []

# indexes for batches
old = 0
new = batch_size * beam_size


for batch in tqdm(dloader):
    encoded = t5_tokenizer.batch_encode_plus(batch[0], add_special_tokens=False, return_tensors='pt', padding='max_length', max_length=512, truncation=True)

    input_ids = encoded['input_ids'].to(DEVICE)
    attention_mask = encoded['attention_mask'].to(DEVICE)

    outputs = t5_mlm.generate(
        input_ids=input_ids,
        max_length=512,  # Change here
        num_beams=beam_size,
        attention_mask=attention_mask,
        early_stopping=True,
        num_return_sequences=beam_size).to(DEVICE)

    predictions.extend(t5_tokenizer.batch_decode(outputs, skip_special_tokens=True))

    to_analyze = predictions[old:new]
    target_list = batch[1]
    input_list = batch[0]

    idx = 0
    for (input_item, target_item) in zip(input_list, target_list):
        target_item = " ".join(target_item.split(' '))
        for i in range(beam_size):
            f_pred.write(to_analyze[idx] + '\n')
            idx += 1

    old = new
    new = new + (batch_size * beam_size)

f_pred.close()
