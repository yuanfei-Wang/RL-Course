import numpy as np
import random
import os

class Tuple_Agent():
    def __init__(self):
        self.net = []
        if os.path.isfile("tupleNet/tuple1.npy"):
            print("Found tuple network")
            print("Loading...")
            self.load_tupleNet("tupleNet/tuple")
        else:
            print("Building tuple Network...")
            self.build_tupleNet()
            print("Completed")


    def build_tupleNet(self):
        self.net.append(np.zeros(shape=(20, 20, 20, 20, 20, 20), dtype=np.float32))
        self.net.append(np.zeros(shape=(20, 20, 20, 20, 20, 20), dtype=np.float32))
        self.net.append(np.zeros(shape=(20, 20, 20, 20), dtype=np.float32))
        self.net.append(np.zeros(shape=(20, 20, 20, 20), dtype=np.float32))
        
    def load_tupleNet(self, filename):
        for i in range(4):
            self.net.append(np.load(filename+str(i+1)+".npy"))
        
    def getV(self, state):
        v = 0.0
        tmp = Board()
        for i in range(8):
            tmp.copyState(state)
            tmp.morphBoard(i)
            v += self.net[0][tmp.getTile(0)][tmp.getTile(4)][tmp.getTile(8)][tmp.getTile(1)][tmp.getTile(5)][tmp.getTile(9)]
            v += self.net[1][tmp.getTile(1)][tmp.getTile(5)][tmp.getTile(9)][tmp.getTile(2)][tmp.getTile(6)][tmp.getTile(10)]
            v += self.net[2][tmp.getTile(2)][tmp.getTile(6)][tmp.getTile(10)][tmp.getTile(14)]
            v += self.net[3][tmp.getTile(3)][tmp.getTile(7)][tmp.getTile(11)][tmp.getTile(15)]
        return v

class Board():
    def __init__(self):
        self.tile = np.zeros((4, 4), dtype = np.uint32)
        self.initialize()

    def initialize(self):
        self.tile = np.zeros((4, 4), dtype = np.uint32)
        init_places = np.random.choice(a=np.array([5, 6, 9, 10], dtype=np.int32), size=2, replace=False)
        init_digitals = np.random.choice(a=np.array([1, 2], dtype=np.int32), size=2, replace=True)
        for i in range(2):
            self.tile[init_places[i] // 4][init_places[i] % 4] = init_digitals[i]

    def copyBoard(self, tmp):
        self.tile = tmp.getBoard().copy()

    def copyState(self, state):
        self.tile = state.copy()

    def transpose(self):
        self.tile = self.tile.transpose()

    def reflect_horizontal(self):
        self.tile = np.fliplr(self.tile)
    
    def reflect_vertical(self):
        self.tile = np.flipud(self.tile)

    def rotate_right(self):
        self.tile = np.rot90(self.tile, 1, (1, 0))
    
    def rotate_left(self):
        self.tile = np.rot90(self.tile)

    def reverse(self):
        self.reflect_horizontal()
        self.reflect_vertical()

    def morphBoard(self, i):
        #if i == 0: keep the same board
        if i == 1:
            self.reflect_horizontal()
        elif i == 2:
            self.reflect_vertical()
        elif i == 3:
            self.reflect_horizontal()
            self.reflect_vertical()
        elif i == 4:
            self.rotate_right()
        elif i == 5:
            self.rotate_right()
            self.reflect_horizontal()
        elif i == 6:
            self.rotate_right()
            self.reflect_vertical()
        elif i == 7:
            self.rotate_right()
            self.reflect_horizontal()
            self.reflect_vertical()		

    def getBoard(self):
        return self.tile
    
    def getTile(self, pos):
        return self.tile[pos // 4][pos % 4]

agent = Tuple_Agent()

def getValue(state):
    # state: np.array((4,4), dtype = np.int32)
    # value: after log()
    state = state.astype(np.uint32)
    return agent.getV(state)

if __name__ == "__main__":
    state = np.zeros((4,4), dtype = np.int32)
    state = np.array([[2,2,2,2],[2,2,2,2],[1,1,1,1],[0,0,1,1]])
    print(getValue(state))