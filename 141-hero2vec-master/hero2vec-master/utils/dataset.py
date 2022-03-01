import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class DataFrameIterator(Dataset):

    def __init__(self, df, hero2ix):
        """
        inputs:
            df: pandas Dataframe
            hero2ix: dictionary
        """
        self.df = df
        self.hero2ix = hero2ix

    def __len__(self):
        """
        Each team compostions can result in 6 center hero (context heroes)
        """
        return int(len(self.df)*6)

    def __getitem__(self, idx):
        """
        inputs:
            idx: int
        returns:
            inputs: torch.LongTensor, size = (5, )
            targets: int
        """
        # Each team composition can give 6 center hero (context heroes)
        # So a specific (context heroes, center_hero) is determined by the team
        # composition and the position of the center hero
        team, center_hero = divmod(idx, 6)

        #locate the team
        heroes = list(self.df.iloc[team])

        #divide context and center hero
        context_heroes = heroes[:center_hero] + heroes[center_hero + 1:]

        team_idxs = list(map(lambda x: int(self.hero2ix[x]), context_heroes))
        center_hero_idx = int(self.hero2ix[heroes[center_hero]])
        inputs = torch.LongTensor(team_idxs)
        targets = center_hero_idx
        return inputs, targets

class MapDataFrameIterator(Dataset):

    def __init__(self, df, hero2ix, map2ix):
        """
        inputs:
            df: pandas Dataframe
            hero2ix: dictionary
            map2ix: dictionary
        """
        self.df = df
        self.hero2ix = hero2ix
        self.map2ix = map2ix

    def __len__(self):
        """
        returns:
            length of DataFrame
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        inputs:
            idx: int
        returns:
            inputs: torch.LongTensor, size = (6, )
            targets: int
        """
        #locate the team and map
        row = self.df.iloc[idx]
        team, map_name = list(row[1:]), row[0]

        team_idxs = list(map(lambda x: int(self.hero2ix[x]), team))
        map_idx = int(self.map2ix[map_name])
        inputs = torch.LongTensor(team_idxs)
        targets = map_idx
        return inputs, targets
