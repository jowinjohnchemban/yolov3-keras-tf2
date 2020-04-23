import re

with open('something.txt') as some:
    with open('out.txt', 'a') as new:
        for line in some.readlines():
            print(line)
            values = re.findall('[0.-9]+', line)
            minimum = float(values[8])
            maximum = float(values[10])
            new.write(f'{line.strip()}{minimum + maximum}\n')