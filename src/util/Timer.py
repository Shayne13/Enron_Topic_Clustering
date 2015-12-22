import time

class Timer:
    def __init__(self, name = None):
        self.name = name
        print 'Starting %s' % name
        self.start = time.time()

    def markEvent(self, eventName):
        print eventName
        print 'Current time elapsed: %f' % (time.time() - self.start)

    def finish(self):
        print 'Finished %s' % self.name
        print 'Total time elapsed: %f' % (time.time() - self.start)
