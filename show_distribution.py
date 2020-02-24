import sys
import json
import math

def count_distribution(path):
    """
    Returns a dictionary of all classes in the json-file at 'path' with their respective amounts.
    """
    count = {}
    json_data = json.load(open(path, 'r'))
    
    for speaker in json_data:
        for sequence in json_data[speaker]:
            for frame in json_data[speaker][sequence]:
                frame_content = json_data[speaker][sequence][frame]
                truth = frame_content[1]
                if truth in count.keys():
                    count[truth] = count[truth] + 1
                else:
                    count[truth] = 1
    return count

def find_top_class(count):
    """
    Returns the most common class and its frequency
    """
    top_class = ''
    n = 0

    for key in count.keys():
        if count[key] > n:
            top_class = key
            n = count[key]

    return top_class, n

def proportion_weight(count, top_class):
    """
    Returns a dictionary that gives weights for all classes x in the given dictionary using the formula: n_max / n_x
    """
    weight_dict = {}
    for key in count.keys():
        weight_dict[key] = count[top_class]/count[key]

    return weight_dict

def sqrt_proportion_weight(count, top_class):
    """
    Returns a dictionary that gives weights for all classes x in the given dictionary using the formula: sqrt(n_max / n_x)
    """
    weight_dict = {}
    for key in count.keys():
        weight_dict[key] = math.sqrt(count[top_class]/count[key])
    return weight_dict

def inverse_weight(count):
    """
    Returns a dictionary that gives weights for all classes x in the given dictionary using the formula: 1 / x
    """
    weight_dict = {}
    for key in count.keys():
        weight_dict[key] = 1 / count[key]
    return weight_dict

# if show_distribution is called directly:
if sys.argv[0] == 'show_distribution.py':
    path = sys.argv[1]
    
    count = count_distribution(path)

    total = 0
    for value in count.values():
        total = total + value

    print('Total frames: ' + str(total))
    for key in count.keys():
        value = count[key] * 100 / total
        if len(key) == 1:
            spacer = '  '
        elif len(key) == 2:
            spacer = ' '
        else:
            spacer = ''
        print(key + ': ' + spacer + str(round(value, 2)))
    print('\n', flush=True)

    top_class, number = find_top_class(count)

    print('\nTop class: ', top_class)

    print('\nWeights (x_max/x):')
    weights = proportion_weight(count, top_class)
    for key in weights:
        print(key, weights[key])    

    print('\nWeights (1/x):')
    weights = inverse_weight(count)
    for key in count.keys():
        print(key, weights[key])
        
    import torch
    import torch.nn.functional as F
    weight_vector = [x for x in count.values()]
    weight_vector = torch.Tensor(weight_vector)
    weight_vector = F.softmax(weight_vector, 0)
    print(weight_vector)