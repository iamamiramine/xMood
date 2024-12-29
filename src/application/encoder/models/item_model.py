# define "Item" for general storage
class Item(object):
    def __init__(self, name, start, end, velocity=None, pitch=None, instrument=None):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch
        self.instrument = instrument

    def __repr__(self):
        return 'Item(name={}, start={}, end={}, velocity={}, pitch={}, instrument={})'.format(
            self.name, self.start, self.end, self.velocity, self.pitch, self.instrument)