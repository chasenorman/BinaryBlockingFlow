from goldberg_rao.test_algorithm import test_correctness
import random

if __name__ == "__main__":
    i = 0
    while True:
        random.seed(17)
        print(i)
        test_correctness()
        i += 1

