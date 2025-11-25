import cv2

class Drawer:
    LINE_1 = [0,255,0]
    LINE_2 = [0,0,255]
    BOX = [255,0,0]
    THICKNESS = 3

    def draw_way_obj(self, img, chars):
        for char in chars:
            if char[-2] != 0:
                continue
            x, y = list(map(int, char[:2]))
            dx, dy = list(map(int, char[4:6]))
            # Vß║╜ ─É╞░ß╗¥ng Thß║│ng
            cv2.line(img, (x, y), (x-dx, y-dy), Drawer.LINE_1, Drawer.THICKNESS)
            cv2.line(img, (x-dx, y-dy), (x-2*dx, y-2*dy), Drawer.LINE_2, Drawer.THICKNESS)
            # Vß║╜ Khung
            if char[-1] != 0:
                x, y, a, h = list(map(int, char[:4]))
                w = a * h
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), Drawer.BOX, Drawer.THICKNESS)
        return img
