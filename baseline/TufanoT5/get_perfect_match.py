#%%

import re, os
import pandas as pd

import javalang

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

def get_perfect_match(dataset_name, beam_size, output_file_path):
    global ground_truth
    global init_ver

    beam_size = str(beam_size)

    if os.path.exists(output_file_path):
        with open(output_file_path,'r') as f:
            output = f.readlines()

        output = [o.strip() for o in output]

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
            prediction_df['pred-approved-ver'] = output
            prediction_df['approved-ver'] = ground_truth[:len(output)]
            prediction_df['correct-predict'] = pred_flag

            ## just in case you want to use the prediction for other purposes
            # prediction_df.to_csv(os.path.join(base_prediction_dir, 'prediction_df_'+beam_size+'.csv'), index=False,sep='\t')

        else:
            start = 0
            end = int(beam_size)

            correct_pred = []

            for i in range(0,len(ground_truth)-1):
                gt = ground_truth[i]
                current_tgt = get_tokenized_code(gt)

                for j in range(start,end):
                    op = output[j]

                    current_pred = get_tokenized_code(op)

                    if " ".join(current_pred.split()) == " ".join(current_tgt.split()):
                        correct_pred.append(op)
                        perfect_match = perfect_match + 1
                        break

                start = end
                end = end + int(beam_size)

        print('beam size {}: {}'.format(beam_size, perfect_match))

    else:
        print('file not exist')



#%%

proj_names = ['google','ovirt','android']

beam_sizes = [1,5,10]


ground_truth = None
init_ver = None

def get_prediction(proj_name, base_prediction_dir, file_suffix):

    global ground_truth, init_ver

    ground_truth = load_ground_truth(proj_name)
    init_ver = load_init_ver(proj_name)

    print('total test',len(ground_truth))
    for bs in beam_sizes:
        output_file_path = os.path.join(base_prediction_dir, file_suffix +str(bs)+'.txt')
        get_perfect_match(proj_name, bs, output_file_path)



# %%

def get_result_RQ1_time_ignore():
    global base_data_dir
    base_data_dir = '../../dataset/dataset-time-ignore'
    
    print('RQ1 time-ignore')
    for proj in proj_names:
        base_prediction_dir = './prediction-time-ignore/'+proj+'/final_prediction/'
        get_prediction(proj,base_prediction_dir,'prediction_beam_')

def get_result_RQ1_time_wise():
    global base_data_dir
    base_data_dir = '../../dataset/dataset-time-wise/'

    print()
    print('RQ1 time-wise')
    for proj in proj_names:
        base_prediction_dir = './prediction-time-wise/'+proj+'/final_prediction/'
        get_prediction(proj,base_prediction_dir,'prediction_beam_')
    # get_prediction('android')
    # get_prediction('google')
    # get_prediction('ovirt')

def get_result_RQ2():
    global base_data_dir
    
    base_data_dir = '../../dataset/dataset-time-wise/'

    print()
    print('RQ2: time-wise with token-level code diff information')
    for proj in proj_names:
        base_prediction_dir = './from-pytorch/output/'+proj+'/T5_with_token_level_code_diff_info_time_wise/'
        get_prediction(proj,base_prediction_dir,'prediction-beam-')


#%%

get_result_RQ1_time_ignore()
get_result_RQ1_time_wise()
get_result_RQ2()
# %%
