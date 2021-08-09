import numpy as np
from lstm import LSTM


def get_initial_loss(vocab_size, seq_length):
    """
    初始化损失，个人理解：softmax损失函数L=-sum(yi * log(yi_hat))，i=0,1,...,vocab_size
    在预测下一个字符实验，下面公式相当于每个cell预测每个字符概率相等，都是1/vocab_size。
    y是vocab_size维向量，第i个位置是标记正确的是1，其余位置是0。
    有seq_length个cell。
    :param vocab_size: 字符（或单词）数量
    :param seq_length: cell数量
    :return:
    """
    return -np.log(1.0/vocab_size)*seq_length


def smooth(loss, cur_loss):
    """
    loss平滑公式，相当于取1000次平均损失
    :param loss:
    :param cur_loss:
    :return:
    """
    return loss * 0.999 + cur_loss * 0.001


def sample(lstm_model, char_to_ix, seed):
    """
    根据输出的概率分布序列对字符序列进行抽样
    生成名称字符串，每生成一个字符作为下一个cell的输入，直到字符是\n或者长度到50结束
    参数：
    lstm_model -- python字典，包含参数Waa, Wax, Wya, by和b。
    char_to_ix -- 将每个字符映射到索引的Python字典。
    seed -- 用于评分

    返回：
    indices -- 一个包含采样字符的索引列表。
    """

    # Step 1: 初始化序列生成
    x = np.zeros((vocab_size, 1))
    # Step 1': 初始化a_prev为零
    a_prev = np.zeros((lstm_model.n_a, 1))
    c_prev = np.zeros((lstm_model.n_a, 1))

    # 创建一个空的索引列表，这个列表将包含要生成的字符的索引列表
    indices = []

    # Idx: 一个检测换行符的标志，初始化为-1
    idx = -1

    # 循环时间步长t。在每个时间步长中，从概率分布中抽取一个字符并附加
    # 它的索引为“indices”。如果达到50个字符，就会停止
    # trained model: 这有助于调试并防止进入无限循环。
    counter = 0
    newline_character = char_to_ix['\n']

    while idx != newline_character and counter != 50:

        # Step 2: 向前传播
        a, c, y, cache = lstm_model.lstm_cell_forward(x, a_prev, c_prev)

        np.random.seed(counter+seed)

        # Step 3: 从概率分布y中对词汇表中的一个字符的索引进行抽样
        idx = np.random.choice(range(vocab_size), p=y.ravel())

        # 将索引添加到“indices”
        indices.append(idx)

        # Step 4: 将输入字符改写为与采样索引相对应的字符。
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        a_prev = a
        c_prev = c

        seed += 1
        counter += 1

    if counter == 50:
        indices.append(char_to_ix['\n'])

    return indices


def print_sample(sample_ix, ix_to_char):
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]
    print('%s' % (txt, ), end='')


def model(data, ix_to_char, char_to_ix, n_a=50, iter_num=35000, dino_names=7, vocab_size=27):
    """
    训练模型并生成恐龙名称。

    参数：
    data -- 文本库
    ix_to_char -- 将索引映射到字符的字典
    char_to_ix -- 将字符映射到索引的字典
    n_a -- 细胞的单位数
    iter_num -- 训练模型的迭代次数
    dino_names -- 希望在每次迭代中采样的恐龙名称的数量。
    vocab_size -- 词汇量的大小，文本中字符的数量（26个字母+换行符）

    返回值：
    parameters -- 学参数
    """
    lstm = LSTM(n_a=n_a, batch_size=1)
    n_x, n_y = vocab_size, vocab_size

    # 初始化参数
    parameters = lstm.initialize_parameters(n_a, n_x, n_y)

    # 初始化loss，平滑loss
    loss = get_initial_loss(vocab_size, dino_names)

    # 创建新联样本集
    with open("dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    # 打乱恐龙的名字
    np.random.seed(0)
    np.random.shuffle(examples)

    # 初始化隐藏状态
    a_prev = np.zeros((n_a, 1))

    for j in range(iter_num):

        # 定义训练示例 (X,Y) (≈ 2 lines)
        index = j % len(examples)
        x = [None] + [char_to_ix[ch] for ch in examples[index]]  # 输入的名字example是名字list
        y = x[1:] + [char_to_ix["\n"]]  # 对应的输出名字，x左移一位后补\n
        X_batch = np.zeros((n_x, 1, len(x)))  # x转为输入矩阵
        Y_batch = np.zeros((n_y, 1, len(x)))  # y转为label
        # 字符对应位置补1
        for t in range(len(x)):
            if x[t] is not None:
                X_batch[x[t], 0, t] = 1
            Y_batch[y[t], 0, t] = 1

        # 每个序列输入初始化loss=0
        lstm.loss = 0
        # 训练
        curr_loss, gradients, a_prev = lstm.optimize(X=X_batch, Y=Y_batch, a_prev=a_prev)

        # 加速训练
        loss = smooth(loss, curr_loss)

        # 每number次迭代，通过sample()生成n个字符，以检查模型是否正确学习
        number = 2000
        if j % number == 0:

            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

            # 要打印的恐龙名字数目
            seed = 0
            for name in range(dino_names):

                # 抽样索引并打印
                sampled_indices = sample(lstm, char_to_ix, seed)
                print_sample(sampled_indices, ix_to_char)

                seed += 1

            print('\n')

    return parameters


data = open('dinos.txt', 'r').read()
data = data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)

char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}
parameters = model(data, ix_to_char, char_to_ix)
