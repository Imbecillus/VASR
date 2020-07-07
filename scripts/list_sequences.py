import sys
if sys.argv[1] == '-full':
    full = True
else:
    full = False

for json_path in sys.argv[2:]:
    print(f'Enumerating all SEQUENCES in {json_path}')

    import json
    data = json.load(open(json_path, 'r'))

    thelist = []
    for speaker in data.keys():
        for sequence in data[speaker].keys():
            length = len(data[speaker][sequence])
            thelist.append([f'{speaker}_{sequence}', length])

    for i in range(len(thelist)):
        if not full:
            if thelist[i][1] > 0:
                continue
        print(f'{i}: {thelist[i][0]} - {thelist[i][1]}')