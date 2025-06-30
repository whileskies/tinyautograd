

class Ops:
    def __init__(self):
        pass 

    @staticmethod
    def add(a , b):
        print(a._shape, b._shape)
        if a._device == 'cpu' and b._device == 'cpu':
            return a._data + b._data
        else:
            pass
        print(a._shape, b._shape)