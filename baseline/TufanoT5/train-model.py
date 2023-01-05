import functools, sys
import os
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

from t5.data import postprocessors as t5_postprocessors
from t5.seqio import Feature,SentencePieceVocabulary

from mesh_tensorflow.transformer.learning_rate_schedules import slanted_triangular 

from mesh_tensorflow.transformer.learning_rate_schedules import truncated_rsqrt
 
from tensorflow.keras.optimizers.schedules import PolynomialDecay

import t5
import gin

tf.disable_v2_behavior()

# Improve logging.
from contextlib import contextmanager
import logging as py_logging

@contextmanager
def tf_verbosity_level(level):
	og_level = tf.logging.get_verbosity()
	tf.logging.set_verbosity(level)
	yield
	tf.logging.set_verbosity(og_level)


project = sys.argv[1] # android, google, or ovirt

base_data_dir = sys.argv[2]+'/'+project+'/'

nq_tsv_path_code_code = {
    "train":      base_data_dir+'train.tsv',
    "validation": base_data_dir+'eval.tsv'
}

data_train = len([line for line in open(base_data_dir+'train.tsv', 'r')])
data_val = len([line for line in open(base_data_dir+'eval.tsv', 'r')])


num_nq_examples_code_code = dict(train=data_train, validation=data_val)




vocab_model_path = './tokenizer/TokenizerModel.model'
vocab_path = './tokenizer/TokenizerModel.vocab'

TaskRegistry = t5.data.TaskRegistry
TfdsTask = t5.data.TfdsTask

def get_default_vocabulary():
  return SentencePieceVocabulary(vocab_model_path, 100)

DEFAULT_OUTPUT_FEATURES = {
    "inputs": Feature(
        vocabulary=get_default_vocabulary(), add_eos=True, required=False),

    "targets": Feature(
        vocabulary=get_default_vocabulary(), add_eos=True)
}

def nq_dataset_code_code(split, shuffle_files=True):
  # We only have one file for each split.
  del shuffle_files

  # Load lines from the text file as examples.
  ds = tf.data.TextLineDataset(nq_tsv_path_code_code[split])
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["string","string"],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  
  ds = ds.map(lambda *ex: dict(zip(["input", "output"], ex)))
  return ds


def code_code_preprocessing(ds):
  def to_inputs_and_targets(ex):
        inputs = tf.strings.join(['code2code: ' + ex['input']], separator=' ')
        class_label = tf.strings.join([ex['output']], separator=' ')
        return {'inputs': inputs, 'targets': class_label }
    
  return ds.map(to_inputs_and_targets, 
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
  
t5.data.TaskRegistry.remove('code_to_code')
t5.data.TaskRegistry.add(
    "code_to_code",
    dataset_fn=nq_dataset_code_code,
    splits=["train", "validation"],
    text_preprocessor=[code_code_preprocessing],
    output_features = DEFAULT_OUTPUT_FEATURES,
    metric_fns=[t5.evaluation.metrics.accuracy],
    num_input_examples=num_nq_examples_code_code
)


# Setting up fine tuning tasks
def _rate_num_input_examples(task):
  if "train" in task.splits:
    return float(task.num_input_examples("train"))
  elif "validation" in task.splits:
    return float(task.num_input_examples("validation"))
  else:
    raise ValueError("Task %s does not have a train or validation split." % (task.name))


t5.data.MixtureRegistry.remove("code_to_code")

t5.data.MixtureRegistry.add(
    "code_to_code",
    ["code_to_code"],
    default_rate=_rate_num_input_examples
)

# our T5 selected architecture
MODEL_SIZE = "small"

#@title Select fine-tuning with or without pre-training
fine_tuning = "fine-tuning_with_pre-training/" #@param ["fine-tuning_with_pre-training/", "fine-tuning_without_pre-training/"]

#@title Select small or large dataset
# dataset = "small" #@param ["Tufano_etal_ICSE21", "new_large"]

task = "code-to-code/"

#@title Selecte the task
task_to_train = "code_to_code"


############ output path ############
MODEL_DIR = './model_dump/' +sys.argv[3] + '/' + project

tf.io.gfile.makedirs(MODEL_DIR)

model_parallelism, train_batch_size, keep_checkpoint_max = {
    "small": (1, 6, 20),
    "base": (2, 128, 8),
    "large": (8, 64, 4),
    "3B": (8, 16, 1),
    "11B": (8, 16, 1)}[MODEL_SIZE]



starter_learning_rate = 0.05
end_learning_rate = 0.001
decay_steps = 10000

learning_rate_fn = PolynomialDecay(
    starter_learning_rate,
    decay_steps,
    end_learning_rate,
    power=0.5)

#@title Select a learning rate scheduler
learning_rate_scheduler_picker = "slanted" #@param ["slanted", "isr", "polynomial", "constant"]

if learning_rate_scheduler_picker == "slanted":
  selected_learning_rate_scheduler = slanted_triangular
  PATH_GIN_FILE = './utils/operative_config_slanted.gin'
elif learning_rate_scheduler_picker == "isr":
  selected_learning_rate_scheduler = truncated_rsqrt
  PATH_GIN_FILE = './utils/operative_config_isr.gin'
elif learning_rate_scheduler_picker == "polynomial":
  selected_learning_rate_scheduler = learning_rate_fn
  PATH_GIN_FILE = './utils/operative_config_polynomial.gin'
elif learning_rate_scheduler_picker == "constant":
  selected_learning_rate_scheduler = 0.001
  PATH_GIN_FILE = './utils/operative_config_constant.gin'

#@title Select a learning rate scheduler
# number_of_steps = 500 #@param {type:"integer"}
number_of_steps = 300000 # default is 300000

tf.io.gfile.makedirs(MODEL_DIR)

model = t5.models.MtfModel(
    model_dir=MODEL_DIR,
    tpu='',
    tpu_topology='',
    model_parallelism=model_parallelism,
    batch_size=train_batch_size,
    learning_rate_schedule = selected_learning_rate_scheduler,
    sequence_length={"inputs": 512, "targets": 512},
    save_checkpoints_steps=2000, # default is 10000
    keep_checkpoint_max=None,
    iterations_per_loop=100,
)

if learning_rate_scheduler_picker == "slanted":
  gin_lines = [line for line in open("./config.gin")]
  f = open("./config.gin", "w+")
  for i in range(len(gin_lines)):
    if i == 196 and fine_tuning == "fine-tuning_without_pre-training/":
      line = "slanted_triangular.start_step = 0\n"
      f.write(line)
      continue
    if i == 197:
      line = "slanted_triangular.total_train_steps = " + str(number_of_steps) + '\n'
      f.write(line)
      continue
    f.write(gin_lines[i])
  f.close()


# Specify the pre-trained dir which must contain the pre-trained models, the operative_config.gin file and the checkpoint file as well
PRETRAINED_DIR= './model_dump/pre-training/'

os.makedirs(PRETRAINED_DIR, exist_ok=True)

start = time.time()

if fine_tuning == "fine-tuning_without_pre-training/":
  # NON PRETRAINED
  with gin.unlock_config():    
      gin.parse_config_file("./config.gin")
      TRAIN_STEPS = number_of_steps
      model.train(task_to_train, steps=number_of_steps)

else:
  # PRETRAINED
  with gin.unlock_config():
      gin.parse_config_file("./config.gin")
      #RUN FINE-TUNING
      model.finetune(
          mixture_or_task_name=task_to_train,
          # pretrained_model_dir=PRETRAINED_DIR,
          pretrained_model_dir=PRETRAINED_DIR,
          finetune_steps=number_of_steps
      )

end = time.time()

time_diff = end-start

with open(os.path.join(MODEL_DIR,'train_time.txt'),'a') as f:
  f.write('train time at {} steps: {} secs\n'.format(str(number_of_steps), str(time_diff)))
