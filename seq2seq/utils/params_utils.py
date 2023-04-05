# step3
# argparse是Python标准库中的一个模块，用于解析命令行参数。
# 可以使用argparse来定义程序所需的参数，并且可以自动生成帮助信息。
# argparse可以轻松地处理参数的解析和错误处理，让代码更加清晰和易于维护。
import argparse


def get_params():
    parser = argparse.ArgumentParser()
    # 编码器和解码器的最大序列长度
    parser.add_argument("--max_enc_len", default=300, help="Encoder input max sequence length", type=int)
    parser.add_argument("--max_dec_len", default=50, help="Decoder input max sequence length", type=int)
    # 一个训练批次的大小
    parser.add_argument("--batch_size", default=32, help="Batch size", type=int)
    # seq2seq训练轮数
    parser.add_argument("--seq2seq_train_epochs", default=20, help="Seq2seq model training epochs", type=int)
    # 词嵌入大小
    parser.add_argument("--embed_size", default=500, help="Words embeddings dimension", type=int)
    # 编码器、解码器以及attention的隐含层单元数
    parser.add_argument("--enc_units", default=512, help="Encoder GRU cell units number", type=int)
    parser.add_argument("--dec_units", default=512, help="Decoder GRU cell units number", type=int)
    parser.add_argument("--attn_units", default=20, help="Used to compute the attention weights", type=int)
    # 学习率
    parser.add_argument("--learning_rate", default=0.001, help="Learning rate", type=float)

    # param是一个字典类型的变量，键：参数名，值：参数值
    args = parser.parse_args()
    params = vars(args)

    return params


if __name__ == '__main__':
    res = get_params()
    print(res)
