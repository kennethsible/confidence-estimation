#!/bin/bash

python annotation.py annotate < data_gpt-4o/gpt_input.txt > data_gpt-4o/gpt_output.txt
python annotation.py collate < data_gpt-4o/gpt_output.txt > data_gpt-4o/mistranslated.txt
