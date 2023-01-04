#%%

import re, os
import pandas as pd

import javalang

exp_name = ''

base_data_dir = '../dataset/final-dataset-no-space-special-chars-latest-version-time-wise/'


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

def load_ground_truth(proj_name):
    with open(base_data_dir+proj_name+'/test/approved_ver.txt') as f:
        gt = f.readlines()

    gt = [re.sub('\s+',' ',s).strip() for s in gt]

    return gt

def load_init_ver(proj_name):
    with open(base_data_dir+proj_name+'/test/initial_ver.txt') as f:
        init_ver = f.readlines()

    init_ver = [re.sub('\s+',' ',s).strip() for s in init_ver]

    return init_ver


def get_perfect_match(dataset_name, beam_size):
    global ground_truth
    global init_ver

    beam_size = str(beam_size)


    output_file_path = os.path.join('./output/',dataset_name,exp_name,'prediction-beam-'+beam_size+'.txt')


    if os.path.exists(output_file_path):
        with open(output_file_path,'r') as f:
            output = f.readlines()

        perfect_match = 0

        if int(beam_size) == 1:

            pred_flag = []

            prediction_df = pd.DataFrame()
            
            for init, op, gt in zip(init_ver, output, ground_truth):
                
                current_init = get_tokenized_code(init)
                current_pred = get_tokenized_code(op)
                current_tgt = get_tokenized_code(gt)

                if " ".join(current_pred.split()) == " ".join(current_tgt.split()):
                    perfect_match = perfect_match + 1
                    pred_flag.append(1)
                else:
                    pred_flag.append(0)

            prediction_df['initial-ver'] = init_ver[:len(output)]
            prediction_df['pred-initial-ver'] = output
            prediction_df['approved-ver'] = ground_truth[:len(output)]
            prediction_df['correct-predict'] = pred_flag[:len(output)]


            # prediction_df.to_csv(os.path.join('./output/',dataset_name,exp_name,'prediction_df_'+beam_size+'.csv'), index=False,sep='\t')

        else:
            start = 0
            end = int(beam_size)

            correct_pred = []

            for i in range(0,len(ground_truth)):
                # print(start,end)
                gt = ground_truth[i]
                current_tgt = get_tokenized_code(gt)

                for j in range(start,end):

                    try:
                        op = output[j]

                        current_pred = get_tokenized_code(op)

                        if " ".join(current_pred.split()) == " ".join(current_tgt.split()):
                            correct_pred.append(op)
                            perfect_match = perfect_match + 1
                            break
                    except:
                        pass

                start = end
                end = end + int(beam_size)

        print('beam size {}: {} '.format(beam_size, perfect_match))


    else:
        print('file not exist')




#%%

proj_names = ['google','ovirt','android']

beam_sizes = [1,5,10]

# for proj_name in proj_names:

ground_truth = None
init_ver = None

def get_prediction(proj_name):
    # for ds in data_sizes:

    global ground_truth, init_ver

    ground_truth = load_ground_truth(proj_name)
    init_ver = load_init_ver(proj_name)

    print('total test',len(ground_truth))
    for bs in beam_sizes:
        # print(proj_name, bs)
        get_perfect_match(proj_name, bs)

#%%

## get result of our D-ACT
def get_result_RQ1_time_wise():
    global exp_name

    exp_name = 'D-ACT'

    print('RQ2 D-ACT')
    for proj in proj_names:
        print(proj)
        get_prediction(proj)



## get result of the variants of our D-ACT
def get_result_RQ2_time_wise():
    global exp_name

    exp_names = ['codeT5_without_token_level_code_diff_info_time_wise', 'codebert_with_token_level_code_diff_info_time_wise', 'GraphCodebert_with_token_level_code_diff_info_time_wise', 'PLBART_with_token_level_code_diff_info_time_wise']

    # exp_name = 'D-ACT'

    print('RQ2 ablation study')
    for exp_name in exp_names:
        print(exp_name)

        for proj in proj_names:
            print(proj)
            
            get_prediction(proj)
            print('-'*30)
        print('*'*50)


# %%
get_result_RQ1_time_wise()
get_result_RQ2_time_wise()