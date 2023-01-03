from math import sqrt, atan, pi
class Line:
    def __init__(self, p1, p2) -> None:
        self.len = sqrt((abs(p1[0]-p2[0])) + abs(p1[1]-p2[1]))
        self.p2 = p2
        self.p1 = p1
        
        if (p1[0]>p2[0]):
            if (p1[1]>p2[1]):
                self.angle = atan((p1[0]-p2[0])/(p1[1]-p2[1])) # 1st Cuadrant
            else:
                self.angle = atan(2*pi-abs(p1[0]-p2[0])/abs(p1[1]-p2[1])) # 4th Cuadrant
        else:
            if (p1[1]>p2[1]):
                self.angle = atan(pi - abs(p1[0]-p2[0])/abs(p1[1]-p2[1])) # 2nd Cuadrant
            else:
                self.angle = atan(pi + abs(p1[0]-p2[0])/abs(p1[1]-p2[1])) # 3rd Cuadrant
            
            
        