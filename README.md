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

German-English Dictionary from Technische Universität Chemnitz ([Source](https://ftp.tu-chemnitz.de/pub/Local/urz/ding/de-en-devel/))

## Data Preprocessing
```
usage: preprocess.py [-h] --lang-pair LANG_PAIR --merge-ops MERGE_OPS --val-set VAL_SET --test-set TEST_SET --data-dir DATA_DIR

options:
  -h, --help            show this help message and exit
  --lang-pair LANG_PAIR
                        source-target language pair
  --merge-ops MERGE_OPS
                        merge operations (subword-nmt)
  --val-set VAL_SET     validation set (sacrebleu)
  --test-set TEST_SET   test set (sacrebleu)
  --data-dir DATA_DIR   output directory
```

## Model Training
```
usage: main.py [-h] --lang-pair LANG_PAIR --train-data FILE_PATH --val-data FILE_PATH [--lem-train FILE_PATH] [--lem-val FILE_PATH] [--dict FILE_PATH] [--freq FILE_PATH] --vocab FILE_PATH --codes FILE_PATH --model FILE_PATH [--seed SEED]

options:
  -h, --help            show this help message and exit
  --lang-pair LANG_PAIR
                        source-target language pair
  --train-data FILE_PATH
                        parallel training
  --val-data FILE_PATH  parallel validation
  --lem-train FILE_PATH
                        lemmatized training
  --lem-val FILE_PATH   lemmatized validation
  --dict FILE_PATH      bilingual dictionary
  --freq FILE_PATH      frequency statistics
  --vocab FILE_PATH     shared vocabulary
  --codes FILE_PATH     subword-nmt codes
  --model FILE_PATH     translation model
  --seed SEED           random seed
```

## Model Inference
```
usage: translate.py [-h] [--dict FILE_PATH] [--freq FILE_PATH] --model FILE_PATH --input FILE_PATH

options:
  -h, --help         show this help message and exit
  --dict FILE_PATH   bilingual dictionary
  --freq FILE_PATH   frequency statistics
  --model FILE_PATH  translation model
  --input FILE_PATH  detokenized input
```

## Model Configuration (Default)
```
embed_dim           = 512   # dimensions of embedding sublayers
ff_dim              = 2048  # dimensions of feed-forward sublayers
num_heads           = 8     # number of parallel attention heads
dropout             = 0.1   # dropout for emb/ff/attn sublayers
num_layers          = 6     # number of encoder/decoder layers
max_epochs          = 200   # maximum number of epochs, halt training
lr                  = 3e-4  # learning rate (step size of the optimizer)
patience            = 3     # number of epochs tolerated w/o improvement
decay_factor        = 0.8   # if patience reached, lr *= decay_factor
min_lr              = 5e-5  # minimum learning rate, halt training
max_patience        = 20    # maximuim patience for early stopping
label_smoothing     = 0.1   # label smoothing (regularization technique)
clip_grad           = 1.0   # maximum allowed value of gradients
batch_size          = 4096  # number of tokens per batch (source/target)
max_length          = 256   # maximum sentence length (during training)
len_ratio           = 2.0   # source-target length ratio (during training)
beam_size           = 4     # size of decoding beam (during inference)
threshold           = 10    # frequency threshold, append definitions
max_append          = 10    # maximum number of definitions/headword
dpe_embed           = 0     # dictionary positional encodings (boolean)
```
