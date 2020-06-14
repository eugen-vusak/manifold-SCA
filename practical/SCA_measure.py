import numpy as np
from pprint import pprint

# translate probabilities of label classes to keys guesses
# HW/HD: 9 classes => 256 key guesses
# value model: 256 classes => 256 key guesses
# input: label probabilities, key guesses (depending on the crypto alg/operation attacked, externally computed)
# output: probabilties of each key
def calculate_key_proba(label_proba, key_guess):
    return label_proba[key_guess]


# computing guessing entropy, and success rate
# SR: is the most probable key == secret key
# GE: ranking position of the correct key
# input: key probabilties, secret key
# output: guessing entropy, success rate (for one sample)
def GE_SR(key_proba, secret_key):
    ranking = key_proba.argsort()

    SR = 0
    if ranking[-1] == secret_key:
        SR = 1

    ind = np.argmax(ranking == secret_key)
    GE = key_proba.shape[0] - ind

    return GE, SR


def compute_guessing_entropy(y_proba, key_guesses, secret_key, number_of_traces, number_of_experiments=50):
    y_proba_log = np.ma.log(y_proba).filled(0)

    GE = np.zeros(number_of_traces)
    SR = np.zeros(number_of_traces)

    for exp in range(number_of_experiments):
        indexes = np.arange(number_of_traces)

        np.random.shuffle(indexes)
        key_proba_total = np.zeros(256)

        for i in range(number_of_traces):
            ind = indexes[i]

            label_proba_log = y_proba_log[ind, :]
            key_guess = key_guesses[:, ind]

            key_proba_total += calculate_key_proba(label_proba_log, key_guess)

            ge, sr = GE_SR(key_proba_total, secret_key)

            GE[i] += ge
            SR[i] += sr

    GE /= float(number_of_experiments)
    SR /= float(number_of_experiments)

    return GE, SR


def main():
    # my code here

    number_of_traces = 100  # DPAv4: 100, all others: 25000
    number_of_exp = 50  # 50 should also be sufficient in case speed up is needed
    # path to key guesses and secret key
    path_to_guesses = 'SCA_datasets/Random Delay/'
    # path where label probabilities are stored (delimiter=',', or to be changed in function above)
    path_to_files = 'Manifold_datasets/Random Delay/15D probs/'
    method = 'RF_value_0.csv'  # filename of label probabilities

    y_pred_proba = np.loadtxt("ltsa1530predproba.csv", delimiter=',')
    key_guesses = np.loadtxt("data/Random Delay/value/key_guesses_ALL.csv", dtype=np.int)
    secret_key = np.loadtxt("data/Random Delay/secret_key.csv", dtype=np.int)
    n_traces = number_of_traces

    np.random.seed(42)

    GE, SR = compute_guessing_entropy(
        y_pred_proba,
        key_guesses, secret_key,
        n_traces
    )

    # saving
    # steps = range(number_of_traces)
    # np.savetxt(path_to_files+method[:-4]+'_SR.csv', [steps, SR], delimiter=',')
    # np.savetxt(path_to_files+method[:-4]+'_GE.csv', [steps, GE], delimiter=',')

    print(GE)
    print(SR)


if __name__ == "__main__":
    main()
