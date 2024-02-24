import os

def find_deepest_folder(directory):
    max_depth = -1
    deepest_folder = None
    for root, dirs, files in os.walk(directory):
        depth = root.count(os.sep)
        if depth > max_depth:
            max_depth = depth
            deepest_folder = root
    return deepest_folder