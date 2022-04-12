import math
import numpy as np
import pandas as pd


class MatrixGame:
    def __init__(self, payoffs, actions):
        self.matrix_size = int(math.sqrt(len(payoffs)))
        self.payoffs = np.array(payoffs, dtype=[("row", object), ("col", object)]).reshape(self.matrix_size,
                                                                                           self.matrix_size)
        self.actions = actions

    def __str__(self):
        game = pd.DataFrame(np.nan, self.actions, self.actions, dtype=object)
        for row in range(self.matrix_size):
            for col in range(self.matrix_size):
                game.iat[row, col] = self.payoffs[row][col]
        return str(game)

