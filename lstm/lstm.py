import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LSTM:
    def __init__(self, epochs=20, n_a=16, alpha=0.01, batch_size=32):
        """
        :param epochs: 迭代次数
        :param n_a: 隐藏层节点数
        :param alpha: 梯度下降参数
        :param batch_size: 每个batch大小
        """
        self.epochs = epochs
        self.n_a = n_a
        self.alpha = alpha
        self.parameters = {}
        self.loss = 0.0
        self.n_x = 2
        self.n_y = 2
        self.m = batch_size

    def initialize_parameters(self, n_a, n_x, n_y):
        """
        :param n_a: 每个cell输出a的维度
        :param n_x: 每个cell输入xi的维度
        :param n_y: 每个cell输出yi的维度
        """
        np.random.seed(1)
        Wf = np.random.randn(n_a, n_a + n_x)*0.01
        bf = np.zeros((n_a, 1))
        Wi = np.random.randn(n_a, n_a + n_x)*0.01
        bi = np.zeros((n_a, 1))
        Wc = np.random.randn(n_a, n_a + n_x)*0.01
        bc = np.zeros((n_a, 1))
        Wo = np.random.randn(n_a, n_a + n_x)*0.01
        bo = np.zeros((n_a, 1))
        Wy = np.random.randn(n_y, n_a)*0.01
        by = np.zeros((n_y, 1))

        self.parameters = {
            "Wf": Wf,
            "bf": bf,
            "Wi": Wi,
            "bi": bi,
            "Wc": Wc,
            "bc": bc,
            "Wo": Wo,
            "bo": bo,
            "Wy": Wy,
            "by": by,
        }
        self.n_x = n_x
        self.n_y = n_y

    def lstm_cell_forward(self, xt, a_prev, c_prev):
        """
        实现lstm单元的单个前向步骤

        Arguments:
        xt -- 你的输入数据在时间步长"t"， numpy数组的形状(n_x, m)。
        a_prev -- 时间步"t-1"的隐藏状态，numpy数组的形状(n_a, m)
        c_prev -- 在时间步长“t-1”时的内存状态，numpy数组的形状(n_a, m)
        parameters -- dictionary containing:
                            Wf -- 权重矩阵的遗忘门，numpy数组的形状 (n_a, n_a + n_x)
                            bf -- 遗忘门的偏置，numpy数组的形状 (n_a, 1)
                            Wi -- 更新门的权值矩阵，numpy数组的形状 (n_a, n_a + n_x)
                            bi -- 更新门的偏置，numpy数组的形状 (n_a, 1)
                            Wc -- 权重矩阵的第一个“tanh”，numpy数组的形状 (n_a, n_a + n_x)
                            bc --  第一偏差 "tanh", numpy数组的形状 (n_a, 1)
                            Wo -- 输出门的权值矩阵，numpy数组的形状 (n_a, n_a + n_x)
                            bo --  输出门的偏置，numpy数组的形状 (n_a, 1)
                            Wy -- 将隐藏状态与输出numpy形状数组相关联的权值矩阵 (n_y, n_a)
                            by -- 隐藏状态与输出numpy形状数组相关的偏置 (n_y, 1)

        Returns:
        a_next -- 下一个隐藏状态，形状 (n_a, m)
        c_next -- 下一个记忆状态，形状 (n_a, m)
        yt_pred -- 预测时间步长“t”，numpy形状数组 (n_y, m)
        cache -- 包含向后传递所需的值的元组 (a_next, c_next, a_prev, c_prev, xt, parameters)

        """

        # Retrieve parameters from "parameters"
        Wf = self.parameters["Wf"]
        bf = self.parameters["bf"]
        Wi = self.parameters["Wi"]
        bi = self.parameters["bi"]
        Wc = self.parameters["Wc"]
        bc = self.parameters["bc"]
        Wo = self.parameters["Wo"]
        bo = self.parameters["bo"]
        Wy = self.parameters["Wy"]
        by = self.parameters["by"]

        # 从xt和Wy的形状检索尺寸
        n_x, m = xt.shape
        n_y, n_a = Wy.shape

        # 链接 a_prev 和 xt (≈3 lines)
        concat = np.zeros((n_a + n_x, m))
        concat[: n_a, :] = a_prev
        concat[n_a :, :] = xt

        # 计算
        ft = sigmoid(np.dot(Wf, concat) + bf)
        it = sigmoid(np.dot(Wi, concat) + bi)
        cct = np.tanh(np.dot(Wc, concat) + bc)
        c_next = np.multiply(ft, c_prev) + np.multiply(it, cct)
        ot = sigmoid(np.dot(Wo, concat) + bo)
        a_next = np.multiply(ot, np.tanh(c_next))

        # 计算LSTM单元的预测
        yt_pred = softmax(np.dot(Wy, a_next) + by)

        # 在缓存中存储向后传播所需的值
        cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt)

        return a_next, c_next, yt_pred, cache

    def lstm_forward(self, x, a0):
        """
        使用lstm细胞实现递归神经网络的正向传播。

        Arguments:
        x -- 输入每个时间步长的数据,  (n_x, m, T_x).
        a0 -- 初始隐藏状态,  (n_a, m)
        parameters -- dictionary containing:
                            Wf -- 权重矩阵的遗忘门，numpy数组的形状 (n_a, n_a + n_x)
                            bf -- 遗忘门的偏置，numpy数组的形状 (n_a, 1)
                            Wi -- 更新门的权值矩阵，numpy数组的形状 (n_a, n_a + n_x)
                            bi -- 更新门的偏置，numpy数组的形状 (n_a, 1)
                            Wc -- 权重矩阵的第一个“tanh”，numpy数组的形状 (n_a, n_a + n_x)
                            bc -- 第一偏差 "tanh", numpy 数组的形状 (n_a, 1)
                            Wo -- 输出门的权值矩阵，numpy数组的形状 (n_a, n_a + n_x)
                            bo -- 输出门的偏置，numpy数组的形状 (n_a, 1)
                            Wy -- 将隐藏状态与输出numpy形状数组相关联的权值矩阵 (n_y, n_a)
                            by -- 隐藏状态与输出numpy形状数组相关的偏置 (n_y, 1)

        Returns:
        a -- 每个时间步的隐藏状态，numpy形状数组(n_a, m, T_x)
        y -- 预测每一个时间步长，numpy数组的形状 (n_y, m, T_x)
        caches -- 包含向后传递所需的值的元组
        """

        # 初始化“缓存”
        caches = []

        # 从x的形状和参数中检索尺寸['Wy']
        n_x, m, T_x = x.shape
        n_y, n_a = self.parameters['Wy'].shape

        # 初始化“a”，“c”和“y”为零
        a = np.zeros((n_a, m, T_x))
        c = np.zeros((n_a, m, T_x))
        y = np.zeros((n_y, m, T_x))

        # ：初始化a_next和c_next
        a_next = a0
        c_next = np.zeros((n_a, m))

        # 循环所有的时间步骤
        for t in range(T_x):
            # 更新下一个隐藏状态，下一个内存状态，计算预测，获取缓存
            a_next, c_next, yt, cache = self.lstm_cell_forward(xt=x[:, :, t], a_prev=a_next, c_prev=c_next)
            # 将新的“next”隐藏状态的值保存在"a"
            a[:, :, t] = a_next
            # 将预测的值保存为"y"
            y[:, :, t] = yt
            # 保存下一个单元格状态的值
            c[:, :, t] = c_next
            # 将缓存追加到缓存中
            caches.append(cache)

        # 在缓存中存储向后传播所需的值
        caches = (caches, x)

        return a, y, c, caches

    def compute_loss(self, y_hat, y):
        """
        计算损失函数
        :param y_hat: (n_y, m, T_x),经过rnn正向传播得到的值
        :param y: (n_y, m, T_x),标记的真实值
        :return: loss
        """
        n_y, m, T_x = y.shape
        for t in range(T_x):
            self.loss -= 1/m * np.sum(np.multiply(y[:, :, t], np.log(y_hat[:, :, t])))
        return self.loss

    def lstm_cell_backward(self, dz, da_next, dc_next, cache):
        """
        实现LSTM-cell的向后传递(单时间步长)。

        Arguments:
        da_next -- 下一个隐藏状态的梯度，形状(n_a, m)
        dc_next -- 下一个单元格状态的梯度，形状(n_a, m)
        cache -- 缓存存储来自forward pass的信息

        Returns:
        gradients -- dictionary containing:
            dxt -- 输入数据在时间步长t时的梯度，形状 (n_x, m)
            da_prev -- 前面的隐藏状态，numpy数组的形状 (n_a, m)
            dc_prev -- 之前的内存状态, 形状(n_a, m, T_x)
            dWf -- 遗忘门的权重矩阵，numpy数组的形状 (n_a, n_a + n_x)
            dWi --  更新门的权值矩阵，numpy数组的形状 (n_a, n_a + n_x)
            dWc --  记忆门的权值矩阵，numpy数组的形状 (n_a, n_a + n_x)
            dWo -- 输出门的权值矩阵，numpy数组的形状 (n_a, n_a + n_x)
            dbf -- 遗忘门的偏差，形状 (n_a, 1)
            dbi --  更新门的偏差，形状 (n_a, 1)
            dbc --  记忆门的偏差，形状 (n_a, 1)
            dbo --  输出门的偏差，形状 (n_a, 1)
        """

        # 从“缓存”检索信息
        (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt) = cache

        # 从xt和a_next的形状中提取尺寸
        n_a, m = a_next.shape

        dWy = np.dot(dz, a_next.T)
        dby = np.sum(dz, axis=1, keepdims=True)
        # cell的da由两部分组成，
        da_next = np.dot(self.parameters['Wy'].T, dz) + da_next

        # 计算门相关的导数
        dot = da_next * np.tanh(c_next) * ot * (1 - ot)
        dcct = (dc_next * it + ot * (1 - np.square(np.tanh(c_next))) * it * da_next) * (1 - np.square(cct))
        dit = (dc_next * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * da_next) * it * (1 - it)
        dft = (dc_next * c_prev + ot * (1 - np.square(np.tanh(c_next))) * c_prev * da_next) * ft * (1 - ft)

        # 计算参数相关的导数。
        concat = np.vstack((a_prev, xt)).T
        dWf = np.dot(dft, concat)
        dWi = np.dot(dit, concat)
        dWc = np.dot(dcct, concat)
        dWo = np.dot(dot, concat)
        dbf = np.sum(dft, axis=1, keepdims=True)
        dbi = np.sum(dit, axis=1, keepdims=True)
        dbc = np.sum(dcct, axis=1, keepdims=True)
        dbo = np.sum(dot, axis=1, keepdims=True)

        # 根据先前的隐藏状态、先前的存储状态和输入计算导数。
        da_prev = np.dot(self.parameters['Wf'][:, :n_a].T, dft) + np.dot(self.parameters['Wi'][:, :n_a].T, dit) + np.dot(self.parameters['Wc'][:, :n_a].T, dcct) + np.dot(self.parameters['Wo'][:, :n_a].T, dot)
        dc_prev = dc_next * ft + ot * (1 - np.square(np.tanh(c_next))) * ft * da_next
        dxt = np.dot(self.parameters['Wf'][:, n_a:].T, dft) + np.dot(self.parameters['Wi'][:, n_a:].T, dit) + np.dot(self.parameters['Wc'][:, n_a:].T, dcct) + np.dot(self.parameters['Wo'][:, n_a:].T, dot)

        # 在字典中保存梯度。
        gradients = {"dxt": dxt, "da_next": da_prev, "dc_next": dc_prev, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                     "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo, "dWy": dWy, "dby": dby}

        return gradients

    def lstm_backward(self, y, y_hat, caches):

        """
        使用LSTM-cell(在整个序列上)实现向后传递。

        Arguments:
        :param y: one_hot ,(n_y, m, T_x)
        :param y_hat: lstm_forward 计算结果, (n_y, m, T_x)
        caches -- 缓存存储来自forward pass的信息 (lstm_forward)

        Returns:
        gradients -- dictionary containing:
                dx -- 输入的梯度，形状 (n_x, m, T_x)
                da0 --  前面的隐藏状态，numpy数组的形状 (n_a, m)
                dWf -- 遗忘门的权重矩阵，numpy数组的形状 (n_a, n_a + n_x)
                dWi -- 更新门的权值矩阵，numpy数组的形状 (n_a, n_a + n_x)
                dWc -- 记忆门的权值矩阵，numpy数组的形状 (n_a, n_a + n_x)
                dWo --  权矩阵的保存门，numpy数组的形状 (n_a, n_a + n_x)
                dbf -- 遗忘门的偏差，形状(n_a, 1)
                dbi -- 更新门的偏差，形状 (n_a, 1)
                dbc -- 记忆门的偏差，形状 (n_a, 1)
                dbo -- 存储门的偏差，形状 (n_a, 1)
                dWy -- 输出的权值矩阵，numpy数组的形状 (n_y, n_a)
                dby -- 输出偏差，numpy数组的形状 (n_y, 1)
        """

        # 从第一个缓存(t=1)中检索值。
        (caches, x) = caches

        # 从da和x1的形状中检索尺寸
        n_x, m, T_x = x.shape
        n_a = self.n_a
        # 初始化梯度
        dx = np.zeros((n_x, m, T_x))
        da0 = np.zeros((n_a, m))
        da_next = np.zeros((n_a, m))
        dc_next = np.zeros((n_a, m))
        dWf = np.zeros((n_a, n_a + n_x))
        dWi = np.zeros((n_a, n_a + n_x))
        dWc = np.zeros((n_a, n_a + n_x))
        dWo = np.zeros((n_a, n_a + n_x))
        dWy = np.zeros((self.n_y, n_a))
        dbf = np.zeros((n_a, 1))
        dbi = np.zeros((n_a, 1))
        dbc = np.zeros((n_a, 1))
        dbo = np.zeros((n_a, 1))
        dby = np.zeros((self.n_y, 1))
        dz = y_hat - y  # y_hat=softmax(z), dz=dl/dy_hat * dy_hat/dz


        for t in reversed(range(T_x)):
            # 使用lstm_cell_backward计算所有梯度
            gradients = self.lstm_cell_backward(dz=dz[:, :, t], da_next=da_next, dc_next=dc_next, cache=caches[t])
            # 存储或添加梯度到参数的前一步的梯度
            dx[:, :, t] = gradients["dxt"]
            dWf = dWf+gradients["dWf"]
            dWi = dWi+gradients["dWi"]
            dWc = dWc+gradients["dWc"]
            dWo = dWo+gradients["dWo"]
            dWy = dWy+gradients["dWy"]
            dbf = dbf+gradients["dbf"]
            dbi = dbi+gradients["dbi"]
            dbc = dbc+gradients["dbc"]
            dbo = dbo+gradients["dbo"]
            dby = dby+gradients["dby"]
            da_next = gradients['da_next']
            dc_next = gradients['dc_next']

        # 将第一个激活的梯度设置为反向传播的梯度da_prev。
        da0 = gradients['da_next']

        gradients = {"dx": dx, "da0": da0, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                     "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo, "dWy": dWy, "dby": dby}

        return gradients

    def update_parameters(self, gradients):
        """
        梯度下降
        :param gradients:
        :return:
        """
        self.parameters['Wf'] += -self.alpha * gradients["dWf"]
        self.parameters['Wi'] += -self.alpha * gradients["dWi"]
        self.parameters['Wc'] += -self.alpha * gradients['dWc']
        self.parameters['Wo'] += -self.alpha * gradients["dWo"]
        self.parameters['Wy'] += -self.alpha * gradients['dWy']

        self.parameters['bf'] += -self.alpha * gradients['dbf']
        self.parameters['bi'] += -self.alpha * gradients['dbi']
        self.parameters['bc'] += -self.alpha * gradients['dbc']
        self.parameters['bo'] += -self.alpha * gradients['dbo']
        self.parameters['by'] += -self.alpha * gradients['dby']

    def optimize(self, X, Y, a_prev):
        """
        执行优化的一个步骤来训练模型。

        Arguments:
        X -- 输入数据序列，维度(n_x, m, T_x)，n_x是每个step输入xi的维度，m是一个batch数据量，T_x一个序列长度
        Y -- 每个输入xi对应的输出yi (n_y, m, T_x)，n_y是输出向量（分类数，只有一位是1）
        a_prev -- previous hidden state.

        Returns:
        loss -- 损失函数的值 (互熵)
        gradients -- dictionary containing:
                            dWax -- 形状的输入到隐藏权值的梯度 (n_a, n_x)
                            dWaa -- 渐变的隐藏权重到隐藏权重，形状 (n_a, n_a)
                            dWya -- 隐藏到输出权重的梯度，形状 (n_y, n_a)
                            db -- 偏置向量的梯度，形状的梯度 (n_a, 1)
                            dby -- 输出偏置向量的梯度，形状 (n_y, 1)
        a[len(X)-1] -- 最后一种隐藏的状态，形状 (n_a, 1)
        """

        # 正向传播
        a, y_pred, c, caches = self.lstm_forward(X, a_prev)
        # 计算损失
        loss = self.compute_loss(y_hat=y_pred, y=Y)

        gradients = self.lstm_backward(Y, y_pred, caches)

        self.update_parameters(gradients)

        return loss, gradients, a[:, :, -1]
