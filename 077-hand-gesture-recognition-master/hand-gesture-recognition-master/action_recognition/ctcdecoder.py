import torch
from itertools import groupby

class CTCDecoder:

    def decode(self):
        """Abstract method
        """
        pass


class BestPathDecoder(CTCDecoder):
    """Simple best path decoding
    """

    def __init__(self, classes, blank_idx=0):
        """classes (int): number of class labels including blank label
        """
        self.classes = classes
        self.blank_idx = blank_idx

    def decode(self, probs):
        """
        probs: softmax network probabilities Tensor, format: BxTxC
        returns:
            labels: BxT, with -1 labels for padding if sequence is shorter than T
            label_scores: BxT, with -1 for padding (same lenghts as labels)
        """
        # get best path
        best_path_values, best_path_indices = probs.topk(1, axis=2)
        best_path_values = best_path_values.squeeze()
        best_path_indices = best_path_indices.squeeze()

        # decode by first removing duplicates, then removing blanks and calculating label scores
        labels = best_path_indices.new_full(best_path_indices.shape, -1)
        label_scores = best_path_values.new_full(best_path_values.shape, -1)

        # iterate over batch
        for batch in range(best_path_indices.size(0)):
            # reorganize in (value, index) pairs
            data = [(index, value) for index, value in zip(best_path_indices[batch], best_path_values[batch])]
            # group by equal indices to remove duplicates
            step = 0
            for key, value in groupby(data, key=lambda x: x[0]):
                # remove blanks in the result aswell
                if key == self.blank_idx:
                    continue

                # calculate the label score as the average of the scores of the grouped predictions
                group_scores = [v for _, v in value]
                label_score = sum(group_scores) / len(group_scores)
                labels[batch, step] = key
                label_scores[batch, step] = label_score
                step += 1

        return labels, label_scores
