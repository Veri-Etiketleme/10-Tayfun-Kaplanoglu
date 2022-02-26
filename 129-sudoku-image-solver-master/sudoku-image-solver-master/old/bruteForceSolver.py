import numpy as np

def possible(p, r, c, x):
    sq = (r // 3, c // 3)
    sq_set = set(p[sq[0]*3:sq[0]*3+3, sq[1]*3:sq[1]*3+3].flatten())
    rowset = set(p[r, :])
    colset = set(p[:, c])
    return x not in (sq_set | rowset | colset)

def _solvePuzzle(p):
    for r in range(9):
        for c in range(9):
            if p[r, c] == 0:
                for x in range(1,10):
                    if possible(p, r, c, x):
                        p[r, c] = x
                        if r == 8 and c == 8:
                            return True
                        if _solvePuzzle(p):
                            return True
                        else:
                            p[r, c] = 0
                return False
    return False

def solvePuzzle(p):
    c = p.copy()
    _solvePuzzle(c)
    return c

if __name__ == "__main__":
    """
    test = [[0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,3,0,8,5],
            [0,0,1,0,2,0,0,0,0],
            [0,0,0,5,0,7,0,0,0],
            [0,0,4,0,0,0,1,0,0],
            [0,9,0,0,0,0,0,0,0],
            [5,0,0,0,0,0,0,7,3],
            [0,0,2,0,1,0,0,0,0],
            [0,0,0,0,4,0,0,0,9]]
    """
    test = [[0,0,0,6,0,4,7,0,0],
            [7,0,6,0,0,0,0,0,9],
            [0,0,0,0,0,5,0,8,0],
            [0,7,0,0,2,0,0,9,3],
            [9,0,0,0,0,0,0,0,5],
            [4,3,0,0,1,0,0,7,0],
            [0,5,0,2,0,0,0,0,0],
            [3,0,0,0,0,0,2,0,8],
            [0,0,2,3,0,1,0,0,0]]
    t = np.array(test)
    s = solvePuzzle(t)
    print(s)
