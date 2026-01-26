import numpy as np
def select_k_from_n(self,n: int, k: int,rng: np.random.Generator) -> list[int]:
        selected = []
        remaining_slots = k
        for i in range(n):
            prob_select = remaining_slots / (n - i)
            if rng.random() < prob_select:
                selected.append(i)
                remaining_slots -= 1

                if remaining_slots == 0:
                    break
        return selected
class DynamicMatrix:
    def __init__(self, initial_n=1):
        self.n = initial_n
        self.matrix = [[0] * self.n for _ in range(self.n)]

    def increment(self, x=1):
        self.n += x
        # Add new columns to existing rows
        for row in self.matrix:
            row.extend([0] * x)
        # Add new rows
        for _ in range(x):
            self.matrix.append([0] * self.n)

    def set_value(self, row, col, value):
        if 0 <= row < self.n and 0 <= col < self.n:
            self.matrix[row][col] = value
        else:
            raise IndexError("Index out of bounds")

    def get_value(self, row, col):
        if 0 <= row < self.n and 0 <= col < self.n:
            return self.matrix[row][col]
        raise IndexError("Index out of bounds")

    def __getitem__(self, idx):
        row, col = idx
        return self.get_value(row, col)

    def __setitem__(self, idx, value):
        row, col = idx
        self.set_value(row, col, value)

    def get_qubo(self):
        return  np.array(self.matrix,dtype=np.float64)

    def __str__(self):
        return '\n'.join(' '.join(str(x) for x in row) for row in self.matrix)


def add_equality(Q:DynamicMatrix, V:list[float], I:list[int], C:float, penalty=1000):
    """
    Add equality constraint: sum(V[i] * x[I[i]]) = C to QUBO matrix Q.
    
    Q[i,j] is modified in-place.
    """
    n = len(I)
    
    for i in range(n):
        for j in range(n):
            idx_i = I[i]
            idx_j = I[j]
            
            if idx_i == idx_j:
                Q[idx_i, idx_i] += penalty * (V[i]**2 - 2 * C * V[i])
            else:
                Q[idx_i, idx_j] += penalty * V[i] * V[j]

def make_symetric(Q:DynamicMatrix):
    for i in range(Q.n):
        for j in range(i+1,Q.n):
            Q[i,j]=Q[j,i]=(Q[i,j]+Q[j,i])/2