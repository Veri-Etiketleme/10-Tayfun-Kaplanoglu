# -*- coding: utf-8 -*-

"""
Team Model
"""
from sklearn.decomposition import PCA
import numpy as np
import Queue
from pprint import pprint


def member_process(member):
    """
    0出场次数,1首发次数,2上场时间,3投篮命中率,4投篮命中次数,5投篮出手次数,6三分命中率,7三分命中次数,8三分出手次数,
    9罚球命中率,10罚球命中次数,11罚球出手次数,12篮板总数,13前场篮板,14后场篮板,15助攻,16抢断,17盖帽,18失误,19犯规,20得分

    :param member: 
    :type member: 
    :return:0上场时间,1投篮命中率,2投篮命中次数,3投篮出手次数,4三分命中率,5三分命中次数,6三分出手次数,
    7罚球命中率,8罚球命中次数,9罚球出手次数,10篮板总数,11前场篮板,12后场篮板,13助攻,14抢断,15盖帽,16失误,17犯规,18得分 
    :rtype: 
    """

    member[3] = member[3]/100
    member[6] = member[6]/100
    member[9] = member[9]/100

    member[18] = -member[18]
    member[19] = -member[19]

    return member[2:]


def get_top_k(k, team):
    """
    Return the Top-K team members of a Team
    :param k: 
    :type k: 
    :param team: 
    :type team: 
    :return: 
    :rtype: 
    """
    #def rank(member):


    priority_queue = Queue.PriorityQueue()
    for i in range(len(team)):
        member = team[i]
        priority_queue.put((-(member[20] + 0.001 * i), member))
    priority_team_data = []
    for i in range(k):
        s = priority_queue.get()
        priority_team_data.append(s[-1])
    return priority_team_data


def average(team_raw_data, topk=-1):
    """
    
    :param team_raw_data: 
    :type team_raw_data: 
    :param topk: 
    :type topk: 
    :return: 
    :rtype: 
    """
    team_data = dict()

    for key in team_raw_data.keys():
        team = team_raw_data[key]
        if topk != -1:
            team = get_top_k(topk, team)
        show_time = 0.0
        times = []
        # for i in range(0, len(team)):
        #     show_time += team[i][2]
        #     times.append(team[i][2])
        team_vector = member_process(team[0]) # times[0]/show_time*
        for i in range(1, len(team)):
            team_vector += member_process(team[i])#*times[0]/show_time
        team_data[key] = team_vector
    return team_data


def extract_essentials(member):

    """
     Member: [投篮命中率,投篮次数, 三分命中率,三分次数, 罚球命中率, 篮板总数, 前场篮板, 后场篮板, 助攻, 抢断, 盖帽, 失误, 犯规, 得分]
#                     [0出场次数,1首发次数,2上场时间,3投篮命中率,4投篮命中次数,5投篮出手次数,6三分命中率,7三分命中次数,8三分出手次数,9罚球命中率,
#                     10罚球命中次数,11罚球出手次数,12篮板总数,13前场篮板,14后场篮板,15助攻,16抢断,17盖帽,18失误,19犯规,20得分]
    
    0上场时间,1投篮命中率,2投篮命中次数,3投篮出手次数,4三分命中率,5三分命中次数,6三分出手次数,
    7罚球命中率,8罚球命中次数,9罚球出手次数,10篮板总数,11前场篮板,12后场篮板,13助攻,14抢断,15盖帽,16失误,17犯规,18得分 
    
    :param member: 
    :type member: 
    :return: 
    :rtype: 
    """
    member = member_process(member)
    new_member = [member[1], member[3], member[4], member[6], member[7]] + member[10:].tolist()

    return np.array(new_member)


def reduce_(team_raw_data, topk=-1):

    team_data = dict()

    for key in team_raw_data.keys():
        team = team_raw_data[key]
        if topk != -1:
            team = get_top_k(topk, team)
        show_time = 0.0
        times = []
        # for i in range(0, len(team)):
        #     show_time += team[i][2]
        #     times.append(team[i][2])
        team_vector = extract_essentials(team[0]) #times[0]/show_time*
        for i in range(1, len(team)):
            team_vector += extract_essentials(team[i])#*times[0]/show_time
        team_data[key] = team_vector
    return team_data

def six_vector(team_raw_data, topk=-1):
    team_data = dict()

    for key in team_raw_data.keys():
        team = team_raw_data[key]
        if topk != -1:
            team = get_top_k(topk, team)
        show_time = 0.0
        times = []
        # for i in range(0, len(team)):
        #     show_time += team[i][2]
        #     times.append(team[i][2])
        team_vector = six_features(team[0]) #times[0]/show_time*
        for i in range(1, len(team)):
            team_vector += six_features(team[i])#*times[0]/show_time
        team_data[key] = team_vector
    return team_data

def six_features(member):
    
    """
     Member: [投篮命中率,投篮次数, 三分命中率,三分次数, 罚球命中率, 篮板总数, 前场篮板, 后场篮板, 助攻, 抢断, 盖帽, 失误, 犯规, 得分]
#                     [0出场次数,1首发次数,2上场时间,3投篮命中率,4投篮命中次数,5投篮出手次数,6三分命中率,7三分命中次数,8三分出手次数,9罚球命中率,
#                     10罚球命中次数,11罚球出手次数,12篮板总数,13前场篮板,14后场篮板,15助攻,16抢断,17盖帽,18失误,19犯规,20得分]
    
    0上场时间,1投篮命中率,2投篮命中次数,3投篮出手次数,4三分命中率,5三分命中次数,6三分出手次数,
    7罚球命中率,8罚球命中次数,9罚球出手次数,10篮板总数,11前场篮板,12后场篮板,13助攻,14抢断,15盖帽,16失误,17犯规,18得分 
    
    :param member: 
    :type member: 
    :return: 
    :rtype: 
    """
    member = member_process(member)
    new_member = [member[4], member[7], member[10], member[12], member[16] + member[17]]

    return np.array(new_member)


def team_representations(team_raw_data, type, topk=-1):

    if type == "average":
        return average(team_raw_data, 8)

    elif type == "reduce":
        return reduce_(team_raw_data, 8)

    elif type == "six_v":
        return six_vector(team_raw_data, 8)


    else:
        print("Not Implemented!")
        exit(-1)


# def team_representations(team_raw_data, type):
#     """
#     Process Team Data to Retrieve its Representations
#     :return Tensor [1x21]
#     """
#     team_data = dict()
#     # Average
#     if type == "rank_8":
#         def retrieve_essentials(member):
#             member[-2] = -member[-2]
#             member[-3] = -member[-3]
#             return member
#
#         for key in team_raw_data.keys():
#             team = team_raw_data[key]
#             team = get_top_k(8, team)
#             team_vector = retrieve_essentials(team[0])
#             for i in range(1, 8):
#                 team_vector += retrieve_essentials(team[i])
#             team_vector = team_vector/8
#             team_data[key] = team_vector
#     elif type == "pca":
#
#         for key in team_raw_data.keys():
#             team = team_raw_data[key]
#
#             team = get_top_k(8, team)
#             # for member in team:
#             #     member = __pca.fit_transform(member)
#             #     print(member.shape)
#             # __pca.fit(team)
#             # team = __pca.transform(team)
#             team_vector = team[0]
#             for i in range(1, 8):
#                 team_vector += team[i]
#             team_data[key] = team_vector
#         return team_data
#     elif type == 'reduce':
#
#         """
#             Member: [投篮命中率, 三分命中率, 罚球命中率, 篮板总数, 前场篮板, 后场篮板, 助攻, 抢断, 盖帽, 失误, 犯规, 得分]
#                     [0出场次数,1首发次数,2上场时间,3投篮命中率,4投篮命中次数,5投篮出手次数,6三分命中率,7三分命中次数,8三分出手次数,9罚球命中率,
#                     10罚球命中次数,11罚球出手次数,12篮板总数,13前场篮板,14后场篮板,15助攻,16抢断,17盖帽,18失误,19犯规,20得分]
#         """
#
#         weights = [0.14, 0.14, 0.14, 0.14, 0.14, 0.1, 0.1, 0.1]
#
#         def retrieve_essentials(member):
#             member[18] = -member[18]
#             member[19] = -member[19]
#             result = [2*member[3], 3*member[6], 2*member[9]]
#             result.extend(member[12:])
#             return np.array(result)
#
#         for key in team_raw_data.keys():
#
#             team = team_raw_data[key]
#             team = get_top_k(8, team)
#             team_vector = retrieve_essentials(team[0])
#             for i in range(1, 8):
#                 team_vector += retrieve_essentials(team[i])
#             team_data[key] = team_vector/8
#         return team_data
#     else:
#         print("Not Implemented!")
#
#     print("Team Data Prepared Finished!")
#     return team_data



