from viseme_list import phonemes_to_woodwarddisney as mapping

newmap = {}

for phoneme in mapping:
    viseme = mapping[phoneme]
    
    if not viseme in newmap:
        newmap[viseme] = []
    newmap[viseme].append(phoneme)

print(newmap)