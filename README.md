# Resurrecting the Dead - Chinese

Text generation system based on a mixed corpus of Kobe bryant<勵志名言> and《論語》(Confucian Analects).

|Framework|Model|Optimizer|
|:-:|:-:|:-:|
| PyTorch | RNN (LSTM) | Adam |

原始連接 : (https://pyliaorachel.github.io/blog/tech/nlp/2017/12/24/resurrecting-the-dead-chinese.html)

## Usage

###### 解說
###### src/data.py
- parse_corpus -> 生成input 長度為 seq_length，target 長度為 1
- format_data -> 採用 mini-batch，尾巴不足 batch_size 的直接捨棄,每 batch_size 筆資料包成一組，並包成 tensor
###### model.py
- 輸入的每個中文字都會先轉成 embedding vector，也就是用一個 vector 來表示各個中文字
- Dropout 則是常見的防止過擬合 (overfitting) 的手段，也就是在訓練過程中三不五時捨棄/忽略一些神經元，來減弱他們彼此間的聯合適應性 (co-adaptation)
- LSTM 設定於hidden layer
###### train.py
- 加速器 : Adam
- Loss : Cross_Entropy
- 模型主架構 : LSTM


###### 混何語言庫
###### Direct Run
$ python3 mix.py ../../corpus/mao_sent.txt ../../corpus/luen_yu_sent.txt --output ../../corpus/corpus.txt

```bash
$ cd src/corpus
$ python3 mix.py <first-corpus> <second-corpus> --output <output-corpus-text-file>

###### Train
###### Direct Run
$ python3 -m train.train ../corpus/corpus.txt --output ../output/model/model.bin --output-c ../output/model/corpus.bin --seq-length 50 --batch-size 4 --embedding-dim 256 --hidden-dim 256 --lr 0.0001 --dropout 0.2 --epochs 30

```bash
$ cd src
$ python3 -m train.train <corpus-text-file> 

Outputs:

- `model.bin`: torch 模型
- `corpus.bin`: parsed corpus, mapping, & vocabulary

###### Text generation

```bash
$ cd src
$ python3 -m generate_text.gen <corpus-bin-file> <model-bin-file>

# For more options
$ python3 -m generate_text.gen -h

# Or directly run
$ ./gen.sh
```

## Structure

```
├── corpus                                          # Raw & parsed corpus
│   ├── corpus.txt                                      # Main corpus file for training
│   ├── luen_yu_clean.txt                               # Raw corpus with irrelevant words removed
│   ├── luen_yu_raw.txt                                 # Raw corpus
│   ├── luen_yu_sent.txt                                # Clean corpus seperated into sentences
│   ├── mao_clean.txt                                   # Raw corpus with irrelevant words removed
│   ├── mao_raw.txt                                     # Raw corpus
│   └── mao_sent.txt                                    # Clean corpus seperated into sentences
├── output                                          # Results
│   ├── log                                             # Log files
│   └── model                                           # Pretrained models
│       └── slxx-bsxx-edxx-hdxx-lrxx-drxx-epxx              # seq_length, batch_size, embedding_dim, hidden_dim, 
│                                                           # learning_rate, dropout, epochs
└── src                                             # Source codes
    ├── corpus                                          # Corpus processing
    │   ├── mix.py                                          # Mix two corpora
    │   └── run.sh                                          # Running the script
    ├── generate_text                                   # Text generation
    │   └── gen.py                                          # Text generation
    ├── train                                           # Model training
    │   ├── data.py                                         # Parse data
    │   ├── model.py                                        # Main LSTM model
    │   └── train.py                                        # Training
    ├── gen.sh                                          # Running text generation script
    └── train.sh                                        # Running training script
```
