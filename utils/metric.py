class Metric(object):
    def __init__(self, correct, total):
        self.correct = correct
        self.total = total
    
    def acc(self):
        return self.correct / self.total
