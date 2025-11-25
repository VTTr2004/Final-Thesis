import torch
import numpy as np

class Visualer:
    def __init__(self, size=63, channel=1):
        self.size = size
        self.channel = channel

    def get_street_feature(self, file_path):
        result = []
        with open(file_path, 'r') as file:
            for line in file:
                result.append(list(map(int, line.strip().split(' '))))
        result = np.array(result)
        result[result==1] = 50
        return result

    def get_visual(self, file_path, h_w, chars, for_check=False):
        img = self.get_street_feature(file_path)
        h, w = h_w
        # Vß║╜ Tß║Ñt Cß║ú Vß║¡t Thß╗â L├¬n
        def max_distance(val):
            if val < -2:
                return -2
            if val > 2:
                return 2
            return int(val)
        for char in chars:
            if char[-2] != 0:
                continue
            x, y = char[:2]
            x = int(x/w*self.size)
            y = int(y/h*self.size)
            dx = max_distance(char[4])
            dy = max_distance(char[5])
            x2 = int(x-dx)
            y2 = int(y-dy)
            img[y2:y2+2, x2:x2+2]=100
            img[y:y+2, x:x+2]=150
        if for_check:
            return 0, img
        # Trß║ú Vß╗ü Danh S├ích ß║ónh
        result = []
        idxes = []
        for idx, char in enumerate(chars):
            if char[-2] != 0 or char[-1] != 0:
                continue
            temp = img.copy()
            x, y = char[:2]
            x = int(x/w*self.size)
            y = int(y/h*self.size)
            dx = max_distance(char[4])
            dy = max_distance(char[5])
            x2 = int(x-dx)
            y2 = int(y-dy)
            temp[y2:y2+2, x2:x2+2]=200
            temp[y:y+2, x:x+2]=250
            tensor_img = torch.tensor(temp, dtype=torch.float32).unsqueeze(0)
            idxes.append(idx)
            result.append(tensor_img)
        return idxes, result
