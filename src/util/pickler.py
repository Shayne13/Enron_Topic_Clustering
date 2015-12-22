import pickle

def save(obj, filename):
    with open(filename, 'w') as f:
        pickle.dump(obj, f)

def load(filename):
    with open(filename, 'r') as f:
        return pickle.load(f)
