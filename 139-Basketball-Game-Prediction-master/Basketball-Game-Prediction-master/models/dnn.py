import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_COMP_VECTOR_SIZE = 2
TEAM_VECTOR_SIZE = 14


class DNN(nn.Module):
    
    def __init__(self):
        super(DNN, self).__init__()
        self.input_team_home_layer = nn.Linear(CURRENT_COMP_VECTOR_SIZE+TEAM_VECTOR_SIZE, 64)
        self.input_team_away_layer = nn.Linear(CURRENT_COMP_VECTOR_SIZE+TEAM_VECTOR_SIZE, 64)
        self.home_team_layer = nn.Linear(64, 128)
        self.away_team_layer = nn.Linear(64, 128)
        self.comp_layer_1 = nn.Linear(256, 256)
        self.comp_layer_2 = nn.Linear(256, 512)
        self.comp_layer_3 = nn.Linear(512, 256)
        self.comp_layer_4 = nn.Linear(256, 128)
        self.out_prob = nn.Linear(128, 2)
        self.out_score = nn.Linear(128, 2)

    def forward(self, home_vector, away_vector, home_state, away_state):
        """
        
        :param home_vector: 
        :type home_vector: 
        :param away_vector: 
        :type away_vector: 
        :param home_state: 
        :type home_state: 
        :param away_state: 
        :type away_state: 
        :return: 
        :rtype: 
        """

        home_representation = F.leaky_relu(
            self.input_team_home_layer(torch.cat([home_vector, home_state], dim=1)),
            negative_slope=-0.1
        )

        away_representation = F.leaky_relu(
            self.input_team_away_layer(torch.cat([away_vector, away_state], dim=1)),
            negative_slope=-0.1
        )

        home_ready = F.relu(
            self.home_team_layer(home_representation)
        )

        away_ready = F.relu(
            self.away_team_layer(away_representation)
        )

        competition_round = F.relu(self.comp_layer_1(torch.cat([home_ready, away_ready], dim=1)))
        competition_round = F.relu(self.comp_layer_2(competition_round))
        # competition_round = self.dropout(competition_round)
        competition_round = F.relu(self.comp_layer_3(competition_round))
        # competition_round = self.dropout(competition_round)
        competition_round = F.relu(self.comp_layer_4(competition_round))

        output_prob = F.softmax(self.out_prob(competition_round))
        output_score = self.out_score(competition_round)

        return output_prob


class SimDNN(nn.Module):

    def __init__(self, input_size):
        super(SimDNN, self).__init__()
        self.input_home_vector = nn.Linear(input_size, 120)
        self.input_away_vector = nn.Linear(input_size, 120)
        self.input_home_state = nn.Linear(2, 8)
        self.input_away_state = nn.Linear(2, 8)
        self.home_layer = nn.Linear(128, 256)
        self.away_layer = nn.Linear(128, 256)
        self.comp_layer_1 = nn.Linear(512, 512)
        self.comp_layer_2 = nn.Linear(512, 256)
        self.comp_layer_3 = nn.Linear(256, 128)
        self.comp_layer_4 = nn.Linear(128, 64)

        self.out_prob = nn.Linear(64, 2)

    def forward(self, home_vector, home_state, away_vector, away_state):
        home_vector = F.leaky_relu(
            self.input_home_vector(home_vector),
            negative_slope=0.2
        )
        away_vector = F.leaky_relu(
            self.input_away_vector(away_vector),
            negative_slope=0.2
        )
        home_state = F.relu(
            self.input_home_state(home_state)
        )
        away_state = F.relu(
            self.input_away_state(away_state)
        )

        home_representation = F.leaky_relu(
            self.home_layer(torch.cat((home_vector, home_state), dim=1)),
            negative_slope=0.2
        )

        away_representation = F.leaky_relu(
            self.home_layer(torch.cat((away_vector, away_state), dim=1)),
            negative_slope=0.2
        )

        competition_round = F.leaky_relu(self.comp_layer_1(torch.cat([home_representation, away_representation], dim=1)))
        competition_round = F.leaky_relu(self.comp_layer_2(competition_round), negative_slope=-0.5)
        competition_round = F.relu(self.comp_layer_3(competition_round))
        competition_round = F.relu(self.comp_layer_4(competition_round))

        output_prob = F.softmax(
            self.out_prob(competition_round)
        )
        return output_prob


class AllInputModel(nn.Module):
    def __init__():
        pass
    def forward(self, home_team_vector, away_team_vector, home_current_comp_vector, away_current_comp_vector):
        pass