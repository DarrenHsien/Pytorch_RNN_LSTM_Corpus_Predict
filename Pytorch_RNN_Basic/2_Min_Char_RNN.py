

"""
Minimal character-level Vanilla RNN model.
相關觀念請參考 ： https://karpathy.github.io/2015/05/21/rnn-effectiveness/
"""
import numpy as np

# data I/O
data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print( 'data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
# 內層網絡節點數
hidden_size = 100 # size of hidden layer of neurons
# 預計單次導入學習的字元數
# 學習器每輪會導入的數字量
seq_length = 5 # number of steps to unroll the RNN for
# 預計預測字元數組合
answerLength = 30
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  #print("total word length :",len(inputs))
  #print(xs)
  #print(targets)
  for t in range(0,len(inputs),1):
    #print(t)
    # --字元編碼--
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    # input到隱藏層前向傳播 Activation( (xs dot Wxh)縱向 + (Whh dot hs[t-1])橫向 )
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    # 隱藏層到輸出層前向傳播 (Why dot hs[t])
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    # 預測每個字元接續下個字元的機率矩陣
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    # 計算損失
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  
  # 反向傳播梯度計算
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(range(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in range(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

# 輸入值 : 讀取文檔 -> 針對預計導入學習的數目擷取出文檔對應字元,並將他們轉化為序列順序值
inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
# 預測值 : 讀取文檔 -> 針對預計導入學習的數目擷取出文檔對應字元+1,並將他們轉化為序列順序值
targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
print("inputs : ",inputs)
print("targets : ",targets)

while True:
    # 用來判斷序列生成input與target是否超出文檔範圍
    # 如果超出須重新開始
    # 須清除隱藏層節點值（因為重頭開始讀取同一句字詞了）
    if p+seq_length+1 >= len(data) or n == 0: 
        hprev = np.zeros((hidden_size,1)) # reset RNN memory
        p = 0 # go from start of data
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
    # 無關訓練;僅在中間過程查看目前當下預測文檔輸出狀況
    if n % 1000 == 0:
        sample_ix = sample(hprev, inputs[0], answerLength)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print( '----\n %s \n----' % (txt, ))

    # forward seq_length characters through the net and fetch gradient
    # 前向傳播
    # inputs 導入當前input
    # targets 導入當前input 對應 解答
    # hprev 導入當前中間隱藏層節點數值
    # loss 輸出當次損失
    # dWxh 輸出當次權重反向梯度
    # dWhh 輸出當次權重反向梯度
    # dWhy 輸出當次權重反向梯度
    # dbh 輸出當次bias反向梯度
    # dby 輸出當次bias反向梯度
    # hprev 輸出當前中間隱藏層節點數值
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 1000 == 0: print( 'iter %d, loss: %f' % (n, smooth_loss)) # print progress
  
    # perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
    # 推進讀去序列文本index
    p += seq_length # move data pointer
    n += 1 # iteration counter 