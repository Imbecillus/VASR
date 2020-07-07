import sys

infile = sys.argv[1]
outfile = sys.argv[2]

with open(infile, 'r') as f:
    with open(outfile, 'w') as o:
        line = f.readline()
        while line:
            if line.startswith('"'):
                line = line.split('/')
                speaker = line[7]
                seq = line[10]

                seq = seq.split('.')[0].upper()
                line = f'"{speaker}_{seq}.rec"\n'
                o.write(line)
            elif line.startswith('#') or line.startswith('.'):
                o.write(line)
            else:
                line = line.split(' ')
                start = line[0][0:-3] if not line[0] == '0' else '0'
                stop = line[1][0:-3]
                phone = line[2]
                line = f'{start} {stop} {phone}'
                o.write(line)

            line = f.readline()
