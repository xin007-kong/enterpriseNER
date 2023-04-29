import pickle
import sys
sys.path.append("..")  # 将上级目录添加到Python路径
from utils import load_model, extend_maps, prepocess_data_for_lstmcrf
from data import build_corpus,build_map
from evaluating import Metrics
from evaluate import ensemble_evaluate
from os.path import join
from codecs import open
# 用户输入句子
inputStr = input("请输入句子：")
def text_to_bmes(input_text, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for char in input_text:
            if char == '\n':
                f.write('\n')
            else:
                f.write(f'{char} O\n')
        f.write('\n')


input_text = inputStr

output_file = "./ResumeNER/predict.char.bmes"

text_to_bmes(input_text, output_file)

predict_word_lists, predict_tag_lists = build_corpus("predict", make_vocab=False,data_dir="./ResumeNER")

train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train",data_dir="./ResumeNER")

crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)

bilstm_model = load_model('./ckpts/bilstm_crf.pkl')

bilstm_model.model.bilstm.bilstm.flatten_parameters()  # remove warning

predict_word_lists, predict_tag_lists = prepocess_data_for_lstmcrf(
        predict_word_lists, predict_tag_lists, test=True
    )

lstmcrf_pred, target_tag_list = bilstm_model.test(predict_word_lists, predict_tag_lists,
                                                  crf_word2id, crf_tag2id)

lstmcrf_pred
