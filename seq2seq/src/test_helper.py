# step3
# 对于生成式任务, 经典模式就是"贪心解码"
import torch
import torch.nn as nn
import tqdm


def greedy_decode(model, test_x, word_to_id, id_to_word, params):
    batch_size = params['batch_size']
    results = []
    total_test_num = len(test_x)
    step_epoch = total_test_num // batch_size + 1
    for i in range(step_epoch):
        batch_data = test_x[i * batch_size: (i + 1) * batch_size]
        results += batch_predict(model, batch_data, word_to_id, id_to_word, params)
        if (i + 1) % 10 == 0:
            print('i = ', i + 1)

    return results


def batch_predict(model, inputs, word_to_id, id_to_word, params):
    data_num = len(inputs)
    predicts = [''] * data_num

    # 输入参数inputs是从文件中加载的numpy类型数据, 需要转换成tensor类型
    inputs = torch.from_numpy(inputs)
    initial_hidden_state = torch.zeros(1, data_num, model.encoder.enc_units)
    enc_output, enc_hidden = model.encoder(inputs, initial_hidden_state)
    dec_hidden = enc_hidden
    dec_input = torch.tensor([word_to_id['<START>']] * data_num)
    dec_input = dec_input.unsqueeze(1)
    context_vector, _ = model.attention(dec_hidden, enc_output)

    for t in range(params['max_dec_len']):
        context_vector, attention_weights = model.attention(dec_hidden, enc_output)
        predictions, dec_hidden = model.decoder(dec_input, context_vector)
        # id转换成字符, 采用贪心解码
        predict_ids = torch.argmax(predictions, dim=1)
        # 内层for循环是为了处理batch中的每一条数据.
        for index, p_id in enumerate(predict_ids.numpy()):
            predicts[index] += id_to_word[p_id] + ' '

        dec_input = predict_ids.unsqueeze(1)

    results = []
    for pred in predicts:
        pred = pred.strip()
        # 句子小于max_len就结束, 直接截断
        if '<STOP>' in pred:
            pred = pred[:pred.index('<STOP>')]
        results.append(pred)

    return results
