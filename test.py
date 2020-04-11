with open('../../mod_layers.txt') as not_mine:
    with open('../../mod_layers_me.txt') as mine:
        nm = [item.strip() for item in not_mine.readlines()]
        m = [item.strip() for item in mine.readlines()]
        for i in range(len(nm)):
            try:
                print(f'{i}- {nm[i]}   {m[i]}')
            except IndexError:
                print(f'{i}- {nm[i]}   None')