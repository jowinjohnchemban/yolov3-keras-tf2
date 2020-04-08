with open('model_layers.txt') as lay:
    lines = [line.strip().replace("'", '')
             for line in lay.readlines()]
    for item in lines:
        print(item)