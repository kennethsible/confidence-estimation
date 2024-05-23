import json

with open('freq.tsv') as freq_f:
    freq_dict: dict[str, int] = {}
    for line in freq_f.readlines():
        word, freq = line.split()
        freq_dict[word] = int(freq)

with open('data_annotation/news-test2008.json') as json_f:
    json_list = []
    for conf_list in json.load(json_f):
        freq_list = []
        for word, _ in conf_list:
            frequency = freq_dict[word] if word in freq_dict else 0
            freq_list.append([word, frequency])
        json_list.append(freq_list)
with open('data_annotation/news-test2008.freq.json', 'w') as json_f:
    json.dump(json_list, json_f, indent=4)
