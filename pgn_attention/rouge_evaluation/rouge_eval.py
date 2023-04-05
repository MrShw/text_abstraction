# 导入工具包
import os
import sys
from rouge import Rouge
import jieba

# 设定项目的root路径, 方便相关代码文件的导入
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

# 导入项目的相关代码文件
from pgn_attention.src.predict import Predict
from pgn_attention.utils.func_utils import timer
from pgn_attention.utils import config


# 构建ROUGE评估类
class RougeEval(object):
    def __init__(self, path):
        self.path = path
        self.scores = None
        self.rouge = Rouge()
        self.sources = []
        self.hypos = []
        self.refs = []
        self.process()

    # 预处理待评估的数据集
    def process(self):
        print('Reading from ', self.path)
        with open(self.path, 'r') as test:
            for line in test:
                source, ref = line.strip().split('<SEP>')
                ref = ref.replace('。', '.')
                self.sources.append(source)
                self.refs.append(ref)

        print('self.refs[]包含的样本数: ', len(self.refs))
        print(f'Test set contains {len(self.sources)} samples.')

    # 生成预测集合
    @timer('building hypotheses')
    def build_hypos(self, predict):
        print('Building hypotheses.')
        count = 0
        for source in self.sources:
            count += 1
            if count % 1000 == 0:
                print('count=', count)
            self.hypos.append(predict.predict(source.split()))

    def get_average(self):
        assert len(self.hypos) > 0, '需要首先构建hypotheses'
        print('Calculating average rouge scores.')
        return self.rouge.get_scores(self.hypos, self.refs, avg=True)


if __name__ == '__main__':
    # 真实的测试机是val_data_path: dev.txt
    print('实例化Rouge对象......')
    rouge_eval = RougeEval(config.val_data_path)
    print('实例化Predict对象......')
    predict = Predict()

    # 利用模型对article进行预测
    print('利用模型对article进行预测, 并通过Rouge对象进行评估......')
    rouge_eval.build_hypos(predict)
    # 将预测结果和标签abstract进行ROUGE规则计算
    print('开始用Rouge规则进行评估......')
    result = rouge_eval.get_average()
    print('rouge1: ', result['rouge-1'])
    print('rouge2: ', result['rouge-2'])
    print('rougeL: ', result['rouge-l'])

    # 最后将计算评估结果写入文件中
    print('将评估结果写入结果文件中......')
    with open('./eval_result/rouge_result.txt', 'a') as f:
        for r, metrics in result.items():
            f.write(r + '\n')
            for metric, value in metrics.items():
                f.write(metric + ': ' + str(value * 100) + '\n')

# 1: 关键的第一步是在rouge_eval.py文件中, 设置好RougeEval(config.val_data_path)中待评估的文件配置路径,
# 也就是config.py中val_data_path的设置调整成当前待评估的数据

# 2: 关键的第二步是在predict.py文件中, 设置好当前要评估的模型, 使用self.model.load_state_dict(torch.load(config.model_save_path)),
# 将config.py中的model_save_path指定成当前待评估的模型存储路径.

# 3: 执行评估代码即可
# cd /home/ec2-user/text_summary/pgn/src/
# python rouge_eval.py