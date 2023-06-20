from data import build_corpus, build_map
from codecs import open
from os.path import join
from evaluate import ensemble_evaluate
from evaluating import Metrics
from utils import load_model, extend_maps, prepocess_data_for_lstmcrf
import pickle
import sys
sys.path.append("..")  # 将上级目录添加到Python路径
import re


def text_to_bmes(input_text, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for char in input_text:
            if char == '\n':
                f.write('\n')
            else:
                f.write(f'{char} O\n')
        f.write('\n')  #保证bmes文件里面的空行


def build_corpus(split, make_vocab=True, data_dir="./ResumeNER"):
    """读取数据"""
    print(f"Provided split value: {split}")  # 添加输出以查看传入的split值
    assert split in ['train', 'dev', 'test','predict']

    word_lists = []
    tag_lists = []
    with open(join(data_dir, split+".char.bmes"), 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            # if line != '\r\n':
            #     word, tag = line.strip('\n').split()
            #     word_list.append(word)
            #     tag_list.append(tag)
            if line.strip():  # 如果不是空行
                word, tag = line.strip().split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists
    
def predict():
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

    # print(lstmcrf_pred)
    return lstmcrf_pred


# 定义一个函数，将标签列表转换为实体字典
def tags_to_dict(text, tags):
    entities = {}
    entity = ''
    entity_type = ''
    for i, tag in enumerate(tags):
        if tag.startswith('B-'):
            # 如果是实体的开始，将当前实体清空，并开始一个新的实体
            entity = text[i]
            entity_type = tag[2:]  # 获取实体类型
        elif tag.startswith('M-'):
            # 如果是实体的中间部分，将当前字加入实体
            entity += text[i]
        elif tag.startswith('E-'):
            # 如果是实体的结束，将当前字加入实体，并将实体加入实体字典
            entity += text[i]
            entities[entity_type] = entity
            entity = ''
            entity_type = ''
        else:
            # 如果不是实体，将当前实体清空
            entity = ''
            entity_type = ''
    return entities



def extract_dates(text):
    # 正则表达式，用于识别常见的日期格式
    date_pattern = r"""(?P<date>
        (?P<year_only>\d{4}(?![\d年]))|                                       # 单独的年份，如 2012，2017
        # YYYY-MM-DD 或 YYYY/MM/DD 或 YYYY.MM.DD 或 YYYY年MM月DD日
        (?P<year>\d{4})[-/年.](?P<month>\d{1,2})[-/月.]?(?P<day>\d{1,2})?[日]?|
        # DD-MM-YYYY 或 DD/MM/YYYY 或 DD.MM.YYYY
        (?P<day2>\d{1,2})[-/.](?P<month2>\d{1,2})[-/.](?P<year2>\d{4})|
        # MM-DD-YYYY 或 MM/DD/YYYY 或 MM.DD.YYYY
        (?P<month3>\d{1,2})[-/.](?P<day3>\d{1,2})[-/.](?P<year3>\d{2,4})
    )"""

    # 使用正则表达式查找日期

    matches = re.finditer(date_pattern, text, re.VERBOSE)
    result = []

    # 将匹配的日期添加到结果列表中
    for match in matches:

        date_str = match.group('date')
        # append的内容是 "日期":"date_str"，如"日期":"2010年3月3日"
        result.append({"日期": date_str})

    return result



class CaseNumberRecognizer:

    def __init__(self):
        self.pattern = r"[（(]\d{4}[）)]\S+\d+号|第\d+号"

    def recognize(self, text):
        results = re.findall(self.pattern, text)
        if results:
            return [{"案号":result} for result in results]
        else:
            return None






# 用户输入句子
inputStr = input("请输入句子：")
input_text = inputStr.strip() # 去掉首尾空格
output_file = "./ResumeNER/predict.char.bmes"
text_to_bmes(input_text, output_file)

predict()

text = input_text
tags = predict()[0]

# 调用函数，将标签列表转换为实体字典
entities = tags_to_dict(text, tags)

# 示例
textDate = input_text
# entities = entities + extract_dates(textDate)
for item in extract_dates(textDate):
    entities.update(item)

text = input_text
recognizer = CaseNumberRecognizer()
results = recognizer.recognize(text)
# 如果返回不为空，将结果添加到实体字典中
if results:
    for res in results:
        entities.update(res)

print(entities)