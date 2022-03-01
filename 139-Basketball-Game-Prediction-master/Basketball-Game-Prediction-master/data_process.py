#-*- coding: utf-8 -*-

import random
import numpy as np

__all__ = ['MatchData']


def train_data_func():
    with open('data/train.csv', 'r') as openfile:
        lines = openfile.readlines()
    i = 0
    data = []
    for line in lines[1:]:
        data.append(map(int, line.split(',')))
        data[i][2] = -data[i][2]
        data[i][5] = -data[i][5]
        data[i][6] = -data[i][6]
        i += 1
    random.shuffle(data)
    return data


def test_data_func():
    with open('data/test.csv', 'r') as openfile:
        lines = openfile.readlines()

    data = []
    i = 0
    for line in lines[1:]:
        data.append(map(int, line.split(',')))
        data[i][2] = -data[i][2]
        data[i][5] = -data[i][5]
        data[i][6] = -data[i][6]
        i += 1
    random.shuffle(data)
    return data


def load_team_data():

    team_data = []

    with open('data/teamData.csv', 'r') as open_file:
        lines = open_file.readlines()
    lines = lines[1:]

    def get_num(_str):
        if len(_str) == 0:
            return 0.0
        _rec = _str.replace('%','')
        return float(_rec)

    for line in lines:
        info = [get_num(x) for x in line.split(',')]
        team_data.append(info)
    return team_data


def load_competitions():
    """
    Load Comepetions
    Competition: [(Away, Home, Away_Ago_Win, Away_Ago_Lose, Home_Ago_Win, Home_Ago_Lose, Away_Score, Home_Score, Home_Win)]
    """

    data = []
    with open('data/matchDataTrain.csv') as open_file:
        lines = open_file.readlines()

    def get_record(rec_str):
        rec_str = rec_str.replace('胜', ',')
        rec_str = rec_str.replace('负', ',')
        records = map(int, rec_str.split(',')[:-1])
        records = [records[0], records[-1]]

        return records

    for line in lines[1:]:
        elements = line.split(',')
        away = int(elements[0])
        home = int(elements[1])

        away_ago = get_record(elements[2])
        home_ago = get_record(elements[3])

        score = map(int, elements[4].split(':'))

        if score[0] > score[1]:
            result = [0]
        else:
            result = [1]

        parts = [away, home] + away_ago + home_ago + score + result

        data.append(parts)

    return data


def tmp_load():
    i = 0

    data = []
    with open('./data/matchDataTest.csv') as open_file:
        lines = open_file.readlines()

    def get_record(rec_str):
        rec_str = rec_str.replace('胜', ',')
        rec_str = rec_str.replace('负', ',')
        records = map(int, rec_str.split(',')[:-1])

        return records

    for line in lines[1:]:
        temp = []
        temp.append(i)
        i += 1
        elements = line.split(',')
        parts = map(int, elements[0:2])

        away_ago = get_record(elements[2])
        home_ago = get_record(elements[3])
        away_ago[0] = -away_ago[0]
        home_ago[1] = -home_ago[1]
        parts = parts + away_ago + home_ago
        data.append(parts)
    return data


def tmp_write():
    s = MatchData(1000)
    s.roll_data()
    s.dump_matches_to_file('data/')


class TeamData(object):

    def __init__(self):
        self._team = dict()

    def get_team(self):
        return self._team

    def process(self):

        team_data = load_team_data()
        current_arr = []
        current_team = 0
        for member in team_data:
            if member[0] != current_team:
                self._team[current_team] = np.array(current_arr)
                current_arr = []
                current_team += 1
            current_arr.append(np.array(member[2:]))
        self._team[current_team] = np.array(current_arr)

    def test(self):

        for key in self._team.keys():
            print("KEY: %s NUMS: %s CONTENT: %s" %(key, len(self._team[key]), self._team[key]))


class MatchData(object):

    def __init__(self, testing_size):
        self.data = load_competitions()
        random.shuffle(self.data)
        self.current_index = 0
        self.testing_size = testing_size

    def roll_data(self):
        """
        :description: cross-validation shuffle data when the index > size 
        """
        if self.current_index > len(self.data):
            self.current_index = 0
            random.shuffle(self.data)
        self.testing_data = self.data[self.current_index:self.current_index+self.testing_size]
        self.training_data = self.data[:self.current_index]+self.data[self.current_index+self.testing_size:]
        self.current_index += self.testing_size

    def get_train_data(self):
        self.roll_data()
        return self.training_data

    def get_test_data(self):
        return self.testing_data

    def dump_matches_to_file(self, file_dir):

        csv_line = 'away,home,away_ago_win,away_ago_lose,home_ago_win,home_ago_lose,score_away,score_home,result\n'
        with open(file_dir+'train.csv', 'w+') as f:
            f.write(csv_line)
            for match in self.training_data:
                line = ','.join(['%s' % x for x in match])
                line += '\n'
                f.write(line)

        with open(file_dir+'test.csv', 'w+') as f:
            f.write(csv_line)
            for match in self.testing_data:
                line = ','.join(['%s' % x for x in match])
                line += '\n'
                f.write(line)


def test_load_competitions():

    data = load_competitions()
    print(data[1:10])


def test_match_data():

    s = MatchData(1000)
    s.roll_data()
    s.dump_matches_to_file('data/')


def test_team_data():

    s = TeamData()
    s.process()












