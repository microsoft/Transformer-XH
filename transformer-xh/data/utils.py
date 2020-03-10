# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json


## read json file into list
def load_data(data_file, isTrain):  
    data = list()
    with open(data_file) as f:
        for line in f.readlines():
            ex = json.loads(line)
            data.append(ex)

    return data


def truncate_input_sequence(tokens_a, tokens_b, max_num_tokens):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b

        if trunc_tokens[-1] == "[SEP]":
            del trunc_tokens[-2]
        else:
            trunc_tokens.pop()
