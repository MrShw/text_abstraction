import os

train_writer = open('../data/train.txt', 'w', encoding='utf-8')
test_writer = open('../data/test.txt', 'w', encoding='utf-8')

n = 0
with open('../data/train_seg_data.csv', 'r', encoding='utf-8') as f1:
    for line in f1.readlines():
        line = line.strip().strip('\n')
        article, abstract = line.split('<sep>')
        text = article + '<SEP>' + abstract + '\n'
        train_writer.write(text)
        n += 1

print('train n=', n)
n = 0

with open('../data/test_seg_data.csv', 'r', encoding='utf-8') as f2:
    for line in f2.readlines():
        line = line.strip().strip('\n')
        text = line + '\n'
        test_writer.write(text)
        n += 1

print('test n=', n)

os.system("tail -n 12871 ../data/train.txt > ../data/dev.txt")
os.system("head -n 70000 ../data/train.txt > ../data/train.txt")
# 传递一个参数
# os.system("python test.py -i %s" % input_param)

