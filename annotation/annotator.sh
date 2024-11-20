#!/bin/bash

python annotation.py annotate < annotation/gpt_input.txt > annotation/gpt_output.txt
python annotation.py collate < annotation/gpt_output.txt > annotation/mistranslated.txt
