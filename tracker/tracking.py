import numpy as np
from scipy.optimize import linear_sum_assignment

class Tracker:
    ATTRIBUTE = {
        'x, y, a, h': [0, 0, 0, 0],
        "x', y', a', h'": [0, 0, 0, 0],
        'last': [0],
        'label': [0]
    }
    NEW_CHAR = []
    for val in ATTRIBUTE.values():
        NEW_CHAR += val
    NEW_CHAR = np.array(NEW_CHAR)

    @staticmethod
    def new_char(box):
        temp = np.copy(Tracker.NEW_CHAR)
        temp[:4] = box
        return temp

    def get_value_future(self, box):
        return box[:4] + box[4:8]
    
    def get_char_old(self, file_path):
        result = []
        with open(file_path, 'r') as file:
            for line in file:
                temp = line.strip().split(' ')  
                temp = list(map(float, temp))
                result.append(temp)
        return np.array(result)
    
    def get_min_couple(self, char_old, char_new):
        size_old = len(char_old)
        size_new = len(char_new)
        size_max = max(size_old, size_new)
        matrix_distance = np.linalg.norm(np.array(char_old)[:, None, :] -\
                                          np.array(char_new)[None, :, :],\
                                          axis=2)
        matrix_padded = np.full((size_max, size_max), fill_value = 1e9)
        matrix_padded[:size_old, :size_new] = matrix_distance
        old_ind, new_ind = linear_sum_assignment(matrix_padded)
        xs = []
        ys = []
        for i, j in zip(old_ind, new_ind):
            if i < size_old and j < size_new:
                threshold = 0.3 * char_old[i][3]
                if matrix_distance[i, j] < threshold:
                    xs.append(i)
                    ys.append(j)
        return xs, ys
    
    def tracking(self, file_path, char_new, return_objs = False):
        chars = self.get_char_old(file_path)
        if len(chars) != 0:
            if len(char_new) == 0:
                chars[:, -2] += 1
            else:
                char_old = np.array([self.get_value_future(char) for char in chars])
                char_new = np.array(char_new)
                old_id, new_id = self.get_min_couple(char_old, char_new)
                
                chars[old_id, 4:8] = char_new[new_id] - chars[old_id, :4]
                chars[old_id, :4] = char_new[new_id]
                
                chars[[i for i in range(len(chars)) if i not in old_id], -2] += 1
                chars[old_id, -2] = 0
                chars = chars[chars[:, -2] < 15]
                
                char_new = char_new[[i for i in range(len(char_new)) if i not in new_id]]
                if len(char_new) != 0:
                    char_new = [self.new_char(np.array(char)) for char in char_new]
                    chars = np.vstack((chars, char_new))
        else:
            chars = [self.new_char(np.array(char)) for char in char_new]
        
        with open(file_path, 'w') as file:
            for char in chars:
                file.write(' '.join(map(str, char.tolist())) + '\n')
        if return_objs:
            return chars
