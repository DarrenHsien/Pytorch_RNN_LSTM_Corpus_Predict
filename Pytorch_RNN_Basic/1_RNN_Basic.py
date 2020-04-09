
import numpy as np

"""
RNN詳細介紹：https://karpathy.github.io/2015/05/21/rnn-effectiveness/
網絡層架構

1.固定大小的輸入與輸出
1->1->1
--cnn圖像分類
2.單向輸入序列輸出
1->3->3
--圖像字幕拍攝圖像解析出單個詞句
3.順序輸入單向輸出
3->3->1
--給定句子並歸類表達的為正面情感或負面情感
4.序列輸入與序列輸出
3->5->3
--讀取英文句子並翻譯程中文句子
5.同步的序列輸入與輸出
--標記視屏每一禎的分類

RNN將輸入向量及其狀態向量與固定（但學習）的函數結合在一起以產生新的狀態向量。
"""


class RNN:

    """
    step為內部每次訓練更新都會調用的函式

    內部包含三個重要參數
    -W_xh : input -> hidden layer 中間的權重
    -W_hh : hidden layer -> hidden layer 中間橫向傳遞的權重
    -W_hy : hidden layer -> output 中間的權重

    np.tanh -> Activation Function

    np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x) -> tanh activation -> dot(self.W_hy) -> predicted
    """
    def step(self, x):
        # update the hidden state
        self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
        # compute the output vector
        y = np.dot(self.W_hy, self.h)
        return y



rnn = RNN()
# x為輸入向量 ; y為rnn模型輸出向量
y = rnn.step(x) 



# 將兩個RNN組合
y1 = rnn1.step(x)
y = rnn2.step(y1)

"""
目標: 我們將為RNN提供大量文本，並要求其在給定先前字符序列的情況下，對序列中下一個字符的概率分佈進行建模。
然後，這將使我們能夠一次生成一個字符的新文本。
作為一個工作示例，假設我們只有四個可能的字母“ helo”的詞彙，並且想在訓練序列“ hello”上訓練RNN。
實際上，該訓練序列來自4個單獨的訓練示例：
1.在“h”的情況下，“ e”的概率應該很可能； 
2.在“he”的情況下，“ l”的可能性很可能； 
3.“l”也應該考慮到“ hel”的上下文，
4.“o”應該考慮給定“hell”的上下文。


假設RNN架構為4-4-4
我們餵入"hell"

rnn的第一步將文字轉化為向量形式
h -> 1 0 0 0 
e -> 0 1 0 0
l -> 0 0 1 0
l -> 0 0 1 0

經過隱藏層的縱向與橫向傳遞
縱向 + 橫向
h -> h下一個的可能
e -> e下一個的可能+he下一個的可能
l -> l下一個的可能+hel下一個的可能
l -> l下一個的可能+hell下一個的可能

"""
























