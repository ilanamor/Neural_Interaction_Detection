import numpy as np
import bisect
import operator

###################
#Interpret Weights
###################

def preprocess_weights(w_dict):
    hidden_layers = [int(layer[1:]) for layer in w_dict.keys() if layer.startswith('h')]
    output_h = ['h' + str(x) for x in range(max(hidden_layers),1,-1)]
    w_agg = np.abs(w_dict['out'])
    w_h1 = np.abs(w_dict['h1'])

    for h in output_h:
        w_agg = np.matmul( np.abs(w_dict[h]), w_agg)

    return w_h1, w_agg


def get_interaction_ranking(w_dict):
    xdim = w_dict['h1'].shape[0]
    w_h1, w_agg = preprocess_weights(w_dict)

    # rank interactions
    interaction_strengths = dict()

    for i in range(len(w_agg)):
        sorted_fweights = sorted(enumerate(w_h1[:, i]), key=lambda x: x[1], reverse=True)
        interaction_candidate = []
        weight_list = []
        for j in range(len(w_h1)):
            bisect.insort(interaction_candidate, sorted_fweights[j][0] + 1)
            weight_list.append(sorted_fweights[j][1])
            if len(interaction_candidate) == 1:
                continue
            interaction_tup = tuple(interaction_candidate)
            if interaction_tup not in interaction_strengths:
                interaction_strengths[interaction_tup] = 0
            inter_agg = min(weight_list)
            interaction_strengths[interaction_tup] += np.abs(inter_agg * np.sum(w_agg[i]))

    interaction_sorted = sorted(interaction_strengths.items(), key=operator.itemgetter(1), reverse=True)
    # forward prune the ranking of redundant interactions
    interaction_ranking_pruned = []
    existing_largest = []
    for i, inter in enumerate(interaction_sorted):
        if len(interaction_ranking_pruned) > 20000: break
        skip = False
        indices_to_remove = set()
        for inter2_i, inter2 in enumerate(existing_largest):
            # if this is not the existing largest
            if set(inter[0]) < set(inter2[0]):
                skip = True
                break
            # if this is larger, then need to recall this index later to remove it from existing_largest
            if set(inter[0]) > set(inter2[0]):
                indices_to_remove.add(inter2_i)
        if skip:
            assert len(indices_to_remove) == 0
            continue
        prevlen = len(existing_largest)
        existing_largest[:] = [el for el_i, el in enumerate(existing_largest) if el_i not in indices_to_remove]
        existing_largest.append(inter)
        interaction_ranking_pruned.append((inter[0], inter[1]))

        curlen = len(existing_largest)

    return interaction_ranking_pruned

def get_pairwise_ranking(w_dict):
    xdim = w_dict['h1'].shape[0]
    w_h1, w_agg = preprocess_weights(w_dict)

    input_range = range(1,xdim+1)
    pairs = [(xa,yb) for xa in input_range for yb in input_range if xa != yb]
    for entry in pairs:
        if (entry[1], entry[0]) in pairs:
            pairs.remove((entry[1],entry[0]))

    pairwise_strengths = []
    for pair in pairs:
        a = pair[0]
        b = pair[1]
        wa = w_h1[a-1].reshape(w_h1[a-1].shape[0],1)
        wb = w_h1[b-1].reshape(w_h1[b-1].shape[0],1)
        wz = np.abs(np.minimum(wa , wb))*w_agg
        cab = np.sum(np.abs(wz))
        pairwise_strengths.append((pair, cab))
#     list(zip(pairs, pairwise_strengths))

    pairwise_ranking = sorted(pairwise_strengths,key=operator.itemgetter(1), reverse=True)

    return pairwise_ranking

w_dict = sess.run(weights)

# Variable-Order Interaction Ranking
print(get_interaction_ranking(w_dict))

# Pairwise Interaction Ranking
print(get_pairwise_ranking(w_dict))