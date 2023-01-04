# D-ACT-Replication-Package

## Environmental Setup

  

In the main folder of the replication package, run the command below in terminal to install the required libraries (the libraries are for conda environment)

  

conda env create -f requirement.yml

  

## A Replication Package

  

We provide scripts for our approach (D-ACT) and the baselines (i.e., AutoTransform and TufanoT5) in our study. There are 3 main directories:

  

-  `baseline`: the directory that stores source code for the baselines

-  `D-ACT`: the directory that stores source code for our approach

-  `dataset`: the directory that stores dataset

To obtain the experiment results presented in our paper, we provide the generated predictions and the script for evaluation (for more detail, please have a look at "**How to get evaluation result**" in each section).
However, you can generate the predictions yourself by follow the steps in the below sections.

### Obtaining the fine-tuned models

The fine-tuned models of all approaches (i.e., D-ACT, TufanoT5 and AutoTransform) can be obtained from this google drive: https://drive.google.com/file/d/14MQCF5YUYX1vH2iuUf_RJX9QFtaNtQec/view?usp=sharing

To use the fine-tuned models to generate predictions, follow the below steps
1. Put the obtained fine-tuned models in the 'supplementary-material' directory

    - D-ACT
        - Put the directories 'model' to the corresponding directories of `supplementary-material`

        - For example, the directory `models-for-D-ACT-papers/D-ACT/output/android/model-time-wise-with-token-level-code-diff-info/model` should be put in `supplementary-material-for-D-ACT-papers/D-ACT/output/android/model-time-wise-with-token-level-code-diff-info/`

    - TufanoT5
        - Put the directories beginning with `pytorch_dump` in the directory `supplementary-material-for-D-ACT-papers/baseline/TufanoT5`

        - Put the directories `model` to the corresponding directories of `supplementary-material`

        - For example, the directory `models-for-D-ACT-papers/baseline/TufanoT5/from-pytorch/output/android/T5_with_token_level_code_diff_info_time_wise/model` should be put in `supplementary-material-for-D-ACT-papers/baseline/TufanoT5/from-pytorch/output/android/T5_with_token_level_code_diff_info_time_wise/`

    - AutoTransform
        - Put the directories beginning with `model` in the directory `supplementary-material-for-D-ACT-papers/baseline/AutoTransform`
    
2. Follow the instructions below

### Using Source Code of Our D-ACT

 
**Steps to generate prediction**

  

To generate the prediction from D-ACT for RQ1, please follow the below steps:

  

1. Go to directory `D-ACT`

2. Run the below script

  

		bash run-inference-with-code-diff.sh <PROJECT> <MODEL_TYPE> <CKPT> <BEAM_SIZE>

  

where `<PROJECT>` is android, google or ovirt; `<MODEL_TYPE>` is CodeT5; `<CKPT>` is the number of train steps; `<BEAM_SIZE>` is the beam size

  

To generate the prediction from D-ACT for RQ2, please follow the below steps:

  

1. Go to directory `D-ACT`

2. Run the below script

	2.1 get prediction with code diff information

		bash run-inference-with-code-diff.sh <PROJECT> <MODEL_TYPE> <CKPT> <BEAM_SIZE>

  

	where `<MODEL_TYPE>` is CodeBERT, GraphCodeBERT or PLBART
	
	<br />

	2.2 get prediction without code diff information

  

		bash run-inference-without-code-diff.sh <PROJECT> <MODEL_TYPE> <CKPT> <BEAM_SIZE>

	where `<MODEL_TYPE>` is CodeT5

  

**How to reproduce the experiment result**

  

Run the script for generating prediction by using the following checkpoints (the fine-tuned models are already provided).

  

- Train steps for RQ1

  

	| **Project** | D-ACT|
	|:-----------:|:---------------:|
	| Android | 20,000 |
	| Google | 16,000 |
	| Ovirt | 16,000 |

  

- Train steps for RQ3 (with code diff information)

	| **Project** | **CodeBERT** | **GraphCodeBERT** | **PLBART** |
	|:-----------:|:---------------:|:-------------:|:------------------------:|
	| Android | 54,000 | 56,000 | 8,000 |
	| Google | 80,000 | 40,000 | 4,000 |
	| Ovirt | 80,000 | 58,000 | 6,000 |

  

- Train steps for RQ3 (without code diff information)

  

	| **Project** | **CodeT5** |
	|:-----------:|:---------------:|
	| Android | 12,000 |
	| Google | 8,000 |
	| Ovirt | 18,000 |

  

**How to get evaluation result**

  

1. Go to directory `D-ACT`

2. Run the below script (you can run in an interactive mode): `get_perfect_match.py`

  
  

### Using Source Code of the Baselines

  

#### AutoTransform

  

**Steps to prepare data**

  

AutoTransform is implemented in the [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) library. Thus, to run AutoTransform, we need to convert the dataset to binary files.

  

To create binary files of the datasets for RQ1, please follow the below steps:

  

1. Go to directory `baseline/AutoTransform/scripts`

2. Run the following command

  

		bash prepare_data.sh <PROJECT>

  

where `<PROJECT>` is android, google or ovirt

  

The script will call subword_toknize.sh to tokenize the datasets by using BPE subword tokenization (for more info, please refer to [this Github](https://github.com/rsennrich/subword-nmt)). The tokenized dataset is stored in BPE-2000-random-split (for time-ignore evaluation) and BPE-2000-time-wise (for time-wise evaluation).

After the dataset is tokenized, the script will call generate_binary_data.py to create binary files from the tokenized dataset. The generated binary files are stored in binary-data-random-split (for time-ignore evaluation) and binary-data-time-wise (for time-wise evaluation) .

  
  

To create binary files of the datasets for RQ2, please follow the below steps:

  

1. Go to directory `baseline/AutoTransform/scripts-with-token-level-code-diff-info`

2. Run the following script

  

		bash prepare_data.sh <PROJECT>

  

	where `<PROJECT>` is android, google or ovirt

	<br />	

	The tokenized dataset is stored in BPE-2000-time-wise-with-code-diff-representation. After the dataset is tokenized, the script will call generate_binary_data.py to create binary files from the tokenized dataset. The generated binary files are stored in binary-data-time-wise-with-code-diff-representation.


**Steps to generate prediction**

  

To generate the prediction from AutoTransform for RQ1, please follow the below steps:

  

1. Go to directory `baseline/AutoTransform/scripts`

2. Run the below script

	2.1 Time-ignore evaluation

		bash call_inference_random_split.sh <PROJECT> <TRAIN_STEP>

  

	where `<PROJECT>` is android, google or ovirt; and `<TRAIN_STEP>` is the number of checkpoint of the model.

  	<br />

	2.2 Time-wise evaluation

  

		bash call_inference_time_wise.sh <PROJECT> <TRAIN_STEP>

  

To generate the prediction from AutoTransform for RQ2, please follow the below steps:

  

1. Go to directory `baseline/AutoTransform/scripts-with-token-level-code-diff-info`

2. Run the below script

  

		bash call_inference.sh <PROJECT> <TRAIN_STEP>

  

**How to reproduce the experiment result**

  

Run the script for generating prediction by using the following checkpoints (the fine-tuned models are already provided).

  

| **Project** | **Time-ignore** | **Time-wise** | **Train with code diff** |
|:-----------:|:---------------:|:-------------:|:------------------------:|
| Android | 28,000 | 20,000 | 25,000 |
| Google | 23,000 | 22,000 | 22,000 |
| Ovirt | 22,000 | 17,000 | 22,000 |

  
  

**How to get evaluation result**

  

1. Go to directory `baseline/AutoTransform/`
2. Run the below script (you can run in an interactive mode): `get_perfect_match.py`

  

#### TufanoT5

  

TufanoT5 is implemented in the [t5](https://github.com/google-research/text-to-text-transfer-transformer) library.

**Steps to generate prediction**

  

To generate the prediction from TufanoT5 for RQ1, please follow the below steps:

  

1. Go to directory `baseline/TufanoT5`
3. Run the below script

	2.1 Time-ignore evaluation

		bash generate_prediction_for_test_time_ignore.sh <PROJECT> <TRAIN_STEP>

  

	where `<PROJECT>` is android, google or ovirt; and `<TRAIN_STEP>` is the number of checkpoint of the model.

	<br />

	2.2 Time-wise evaluation

  

		bash generate_prediction_for_test_time_wise.sh <PROJECT> <TRAIN_STEP>

  

To generate the prediction from TufanoT5 for RQ2, please follow the below steps:

  

1. Go to directory `baseline/TufanoT5`

2. Run the below script

  

		bash generate-prediction-with-code-diff.sh <PROJECT> <TRAIN_STEP>

  

**How to reproduce the experiment result**

  

Run the script for generating prediction by using the following checkpoints (the fine-tuned models are already provided).

  

| **Project** | **Time-ignore** | **Time-wise** | **Train with code diff** |
|:-----------:|:---------------:|:-------------:|:------------------------:|
| Android | 26,000 | 44,000 | 66,000 |
| Google | 26,000 | 26,000 | 78,000 |
| Ovirt | 16,000 | 34,000 | 114,000 |

  

**How to get evaluation result**

  

1. Go to directory `baseline/TufanoT5/`

2. Run the below script (you can run in an interactive mode): `get_perfect_match.py`