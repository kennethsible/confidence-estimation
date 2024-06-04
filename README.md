# Dictionary-Augmented Machine Translation
**Ken Sible | [NLP Group](https://nlp.nd.edu)** | **University of Notre Dame**

Note, any option in `config.toml` can also be passed as a command line argument,
```
$ python translation/translate.py --model de-en.pt --beam-size 10 "Guten Tag!"
```

and any output from `stdout` can be diverted using `tee` or output redirection.
```
$ python translation/translate.py --model de-en.pt --input data.src > data.hyp
```

German-English Dictionary from Technische Universität Chemnitz ([Source](https://ftp.tu-chemnitz.de/pub/Local/urz/ding/de-en-devel/))<br>
Biomedical Test Set from WMT22 Biomedical Translation Task ([Source](https://www.statmt.org/wmt22/biomedical-translation-task.html))

## Data Preprocessing
```
usage: preprocess.py [-h] --lang-pair LANG_PAIR --data-dir DATA_DIR [--lem-model LEM_MODEL] --max-length MAX_LENGTH --len-ratio LEN_RATIO {bpe,spm} ...

positional arguments:
  {bpe,spm}             BPE or SentencePiece

options:
  -h, --help            show this help message and exit
  --lang-pair LANG_PAIR
                        language pair
  --data-dir DATA_DIR   data directory
  --lem-model LEM_MODEL
                        lemmatizer model
  --max-length MAX_LENGTH
                        maximum length
  --len-ratio LEN_RATIO
                        length ratio
```

## Model Training
```
usage: main.py [-h] --lang-pair LANG_PAIR --train-data FILE_PATH --val-data FILE_PATH [--lem-train FILE_PATH] [--lem-val FILE_PATH] [--dict FILE_PATH] [--freq FILE_PATH] --sw-vocab FILE_PATH --sw-model FILE_PATH --model FILE_PATH --log FILE_PATH [--seed SEED]

options:
  -h, --help            show this help message and exit
  --lang-pair LANG_PAIR
                        source-target language pair
  --train-data FILE_PATH
                        parallel training data
  --val-data FILE_PATH  parallel validation data
  --lem-train FILE_PATH
                        lemmatized training data
  --lem-val FILE_PATH   lemmatized validation data
  --dict FILE_PATH      bilingual dictionary
  --freq FILE_PATH      frequency statistics
  --sw-vocab FILE_PATH  subword vocab
  --sw-model FILE_PATH  subword model
  --model FILE_PATH     translation model
  --log FILE_PATH       logger output
  --seed SEED           random seed
```

## Model Inference
```
usage: translate.py [-h] [--dict FILE_PATH] [--freq FILE_PATH] --sw-vocab FILE_PATH --sw-model FILE_PATH --model FILE_PATH [--input FILE_PATH]

options:
  -h, --help            show this help message and exit
  --dict FILE_PATH      bilingual dictionary
  --freq FILE_PATH      frequency statistics
  --sw-vocab FILE_PATH  subword vocab
  --sw-model FILE_PATH  subword model
  --model FILE_PATH     translation model
  --input FILE_PATH     detokenized input
```

## Model Configuration (Default)
```
embed_dim           = 512   # dimensions of embedding sublayers
ff_dim              = 2048  # dimensions of feed-forward sublayers
num_heads           = 8     # number of parallel attention heads
dropout             = 0.1   # dropout for emb/ff/attn sublayers
num_layers          = 6     # number of encoder/decoder layers
max_epochs          = 250   # maximum number of epochs, halt training
lr                  = 3e-4  # learning rate (step size of the optimizer)
patience            = 3     # number of epochs tolerated w/o improvement
decay_factor        = 0.8   # if patience reached, lr *= decay_factor
min_lr              = 5e-5  # minimum learning rate, halt training
max_patience        = 20    # maximuim patience for early stopping
label_smoothing     = 0.1   # label smoothing (regularization technique)
clip_grad           = 1.0   # maximum allowed value of gradients
batch_size          = 4096  # number of tokens per batch (source/target)
max_length          = 256   # maximum sentence length (during training)
beam_size           = 4     # size of decoding beam (during inference)
threshold           = 10    # frequency threshold, append definitions
max_append          = 10    # maximum number of definitions/headword
```
