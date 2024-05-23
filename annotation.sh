#!/bin/bash

python annotation.py annotate < data_annotatioin/gpt_input.txt > data_annotation/gpt_output.txt
python annotation.py collate < data_annotation/gpt_output.txt > data_annotation/mistranslated.txt
python annotation.py score --threshold 20 data_annotation/mistranslated.txt data_annotation/news-test2008.json
