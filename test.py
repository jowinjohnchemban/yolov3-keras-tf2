def make_multiplier_of(n):
    def multiplier(x):
        return x * n
    return multiplier


print(make_multiplier_of(5)(7))