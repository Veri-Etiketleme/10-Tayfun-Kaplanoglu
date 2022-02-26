import numpy as np

def cross(A, B):
    """
    Cross product of A and B
    """
    return [a + b for a in A for b in B]

digits = '123456789'
rows = 'ABCDEFGHI'
cols = digits
squares = cross(rows, cols)

# List of all possible units (rows, columns, squares
unitlist = ([cross(rows, c) for c in cols] +
            [cross(r, cols) for r in rows] +
            [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')])

# Dictionary mapping squares to their units
units = dict((s, [u for u in unitlist if s in u]) for s in squares)
# Dictionary mapping squares to their peers (all other squares in thier unit) 
peers = dict((s, set(sum(units[s], [])) - set([s])) for s in squares)

def grid_values(grid):
    """
    Create dictionary from ndarray, mapping squares to their initial values
    """
    return dict(zip(squares, map(str, grid.flatten())))

def parse_grid(grid):
    """
    Parses grid into dictionary of key = sq, val = string of possible values
    """
    values = dict((s, digits) for s in squares)
    for s, d in grid_values(grid).items():
        if d in digits and not assign(values, s, d):
            return False # Can't assign value to this square!
    return values

def eliminate(values, s, d):
    """
    Eliminate d from values[s].  Propogate when values or places is <= 2.
    Return False if contradiction is detected, else return values.
    """
    if d not in values[s]:
        return values # d has already been eliminated
    values[s] = values[s].replace(d, '') # eliminate d

    # If a square is reduced to 0 values (no legal values), there is a contradiction.
    if len(values[s]) == 0:
        return False 

    # If a square is reduced to only one value, d2, then eliminate d2 from all of its peers
    # and ensure that all of them still have legal values
    elif len(values[s]) == 1:
        d2 = values[s]
        if not all(eliminate(values, s2, d2) for s2 in peers[s]):
            return False

    # If a unit is reduced to only one possible place for a value d, 
    # then place d there and propigate
    for u in units[s]: 
        # s = square
        # d = digit in question
        # units[s] = units containing square s
        # u = a unit (row, col, square)
        # values[s] = possible values for square s

        # Find all places in the each unit s belongs to where d can go
        dplaces = [s for s in u if d in values[s]]
        # If there are no places d can go, there is a contradiction
        if len(dplaces) == 0:
            return False 
        if len(dplaces) == 1:
            if not assign(values, dplaces[0], d):
                return False
    return values

def assign(values, s, d):
    """
    Eliminate all values (except d) from values[s] and propogate.
    Return False if a contradiction is detected, else return values.
    """
    other_values = values[s].replace(d, '')
    if all(eliminate(values, s, d2) for d2 in other_values):
        return values
    else:
        return False

def search(values):
    # Failed somewhere earlier in search
    if values is False:
        return False 
    # Solved!
    if all(len(values[s]) == 1 for s in squares):
        return values 
    # Chose the unfilled square with the fewest possibilities
    n, s = min((len(values[s]), s) for s in squares if len(values[s]) > 1)
    # Run through all possiblities for the square, returning a correct 
    # solution if it exists.  Copy the values for each attempted value 
    # to avoid dealing with bookkeeping for backtracking.
    return some(search(assign(values.copy(), s, d)) for d in values[s])
   
def some(seq):
    for e in seq:
        if e:
            return e
    return False

def formatOutput(values):
    if not values:
        return False
    return np.array([int(x[1]) for x in sorted((s, d) for s, d in values.items())]).reshape(9,9)

def solve(grid):
    return formatOutput(search(parse_grid(grid)))

def test():
    """
    A set of unit tests.
    """

    assert len(squares) == 81
    assert len(unitlist) == 27
    assert all(len(units[s]) == 3 for s in squares)
    assert all(len(peers[s]) == 20 for s in squares)
    assert units['C2'] == [['A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2', 'I2'],
                           ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'],
                           ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']]
    assert peers['C2'] == set(['A2', 'B2', 'D2', 'E2', 'F2', 'G2', 'H2', 'I2',
                               'C1', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
                               'A1', 'A3', 'B1', 'B3'])
    print('All basic tests passed.')
    with open('./test/top95.txt') as f:
        line = True
        count = 0
        while line:
            line = f.readline().strip()
            if len(line) == 81:
                count += 1
                parsed = (np.array([int(x) if ord(x) >= ord('1') and ord(x) <= ord('9') else 0 for x in line]).reshape((9,9)))
                print('Puzzle {}\n'.format(count), parsed)
                print('Puzzle {} Solution\n'.format(count), solve(parsed))
        print('Solved {} puzzles successfully'.format(count))

if __name__ == "__main__":
    test()
