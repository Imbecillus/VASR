train_set_speakers = [
    '24M',
    '04M',
    '26M',
    '02M',
    '32F',
    '47M',
    '06M',
    '50F',
    '59F',
    '23M',
    '19M',
    '05F',
    '31F',
    '22M',
    '01M',
    '39M',
    '46F',
    '11F',
    '42M',
    '57M',
    '43F',
    '29M',
    '17F',
    '37F',
    '21M',
    '12M',
    '38F',
    '48M',
    '16M',
    '52M',
    '40F',
    '13F',
    '14M',
    '03F',
    '20M',
    '51F',
    '30F',
    '10M',
    '07F'
]

test_set_speakers = [
    '28M',
    '55F',
    '25M',
    '56M',
    '49F',
    '44F',
    '33F',
    '09F',
    '18M',
    '54M',
    '45F',
    '36F',
    '34M',
    '15F',
    '58F',
    '08F',
    '41M'
]

train_set_male = 0
train_set_female = 0
test_set_male = 0
test_set_female = 0

for speaker in train_set_speakers:
    if 'F' in speaker:
        train_set_female += 1
    else:
        train_set_male += 1

for speaker in test_set_speakers:
    if 'F' in speaker:
        test_set_female += 1
    else:
        test_set_male += 1

print('Train set: ')
print(round(train_set_female / len(train_set_speakers) * 100, 0), '% female')
print(round(train_set_male / len(train_set_speakers) * 100, 0), '% male\n')
print('Test set: ')
print(round(test_set_female / len(test_set_speakers) * 100, 0), '% female')
print(round(test_set_male / len(test_set_speakers) * 100, 0), '% male')