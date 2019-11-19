import numpy as np
import copy
import sys
import os.path
from tqdm import tqdm
from time import time, sleep
import matplotlib.pyplot as plt
import xlsxwriter
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=300)


class SingleTrackTS:
    """Single Track Local Search Algorithm.

    It reads an instance and uses local search to find optimal schedule, with best accept strategy.

    Initial solution: same departing sequence at all tracks, sequence based on the departure times at the first track.
    Neighbours: swapping adjacent trains in a departing sequence at a track.
    Objective function: the sum of departure time delays at each location.

    In a 5-station-4-train instance, there is 9 locations (stations and tracks).
    location    0   1   2   3   4   5   6   7   8
    type        s   t   s   t   s   t   s   t   s

    Attributes:
        num_sta(int): Number of stations in the instance
        num_trn(int): Number of trains in the instance
        num_tot(int): Number of locations
        p(numpy.ndarray[float]): Processing times of all trains at all locations
        d(numpy.ndarray[float]): Ideal departure times of all trains at all locations
        inc_A(numpy.ndarray[int]): Incumbent sequences of departing trains at all tracks (5 stations -> 4 sequences)
        inc_departures(numpy.ndarray[float]): Incumbent departure times
        inc_obj(float): Incumbent objective value
        best_A(numpy.ndarray[int]): Best sequences of departing trains at all tracks
        best_obj(int): Best objective value
        best_departures(numpy.ndarray[float]): Best departure times
        iter(int): Number of iterations in local search
        max_iter(int): Maximum number of iteration in local search
        tabu(): Tabu list of moves
                A move (t, i) in a "swap" neighbours means swapping A[t, i] and A[t, i+1]
        tabu_size(int): size of tabu list
        obj_his(numpy.ndarray[float]): History of objective values in each iteration
    """

    def __init__(self, folder_loc, file_name, max_iter=500, tabu_size=8):
        """Initialize the problem by reading an instance and find initial solution"""

        self.iter = 0
        self.max_iter = max_iter
        self.tabu = []
        self.tabu_size = tabu_size
        self.obj_his = []
        self.filename = file_name

        # read instance
        with open(os.path.join(folder_loc, file_name), "r") as f:
            # read number of stations and number of trains
            self.num_sta = int(f.readline())
            self.num_trn = int(f.readline())
            # num_tot is the total number of stations and tracks (5 stations -> num_tot = 9)
            self.num_tot = 2 * self.num_sta - 1

            # read processing times
            self.p = np.zeros((self.num_trn, self.num_tot - 1), dtype=np.double)
            for t in range(0, self.num_trn):
                line = f.readline().split(" ")
                for s in range(0, self.num_tot - 1):
                    self.p[t, s] = line[s]

            # read ideal departure times
            self.d = np.zeros((self.num_trn, self.num_tot), dtype=np.double)
            for t in range(0, self.num_trn):
                line = f.readline().split(" ")
                for s in range(0, self.num_tot):
                    self.d[t, s] = line[s]

        # moves
        self.moves = [(t, i) for t in range(0, self.num_sta - 1) for i in range(0, self.num_trn - 1)]

        # initial solution: same departing sequence at all tracks
        # sequence based on the departure times at the first track (location 1 d[:, 1])
        sequence = self.d[:, 1].argsort(axis=0)
        init_A = np.repeat(np.expand_dims(sequence, axis=0), repeats=self.num_sta - 1, axis=0)
        init_obj, init_departures = self.obj_and_departures(init_A)

        # incumbent solution
        self.inc_A = copy.deepcopy(init_A)
        self.inc_obj, self.inc_departures = copy.deepcopy(init_obj), copy.deepcopy(init_departures)

        # best solution
        self.best_A = copy.deepcopy(init_A)
        self.best_obj, self.best_departures = copy.deepcopy(init_obj), copy.deepcopy(init_departures)

    def neighbour(self, move):
        t = move[0]
        i = move[1]
        new_A = copy.deepcopy(self.inc_A)
        new_A[t, i], new_A[t, i + 1] = new_A[t, i + 1], new_A[t, i]
        return new_A

    def obj_and_departures(self, A, m=None):
        """Calculate and return objective value and actual departure times based on A, d and p

            A[k]: sequence of train No. at track k (5 stations -> k = 4)
        """
        if m is None:
            departures = np.zeros(self.d.shape, dtype=np.double)
            # departure times at station 0
            for i in range(0, self.num_trn):
                departures[i, 0] = self.d[i, 0]
            # departure time at the remaining locations
            for s in range(0, self.num_sta - 1):
                t = 2 * s + 1
                # a queue of trains ordered by departure times at t+1 (track) in ascending order
                q = A[s]
                # first train's departure times at location t (track) and location t+1 (station)
                departures[q[0], t] = departures[q[0], t - 1] + self.p[q[0], t - 1]
                departures[q[0], t + 1] = departures[q[0], t] + self.p[q[0], t]
                # remaining trains' departure times at location t (track) and location t+1 (station)
                for i in range(1, self.num_trn):
                    departures[q[i], t] = max(departures[q[i - 1], t + 1], departures[q[i], t - 1] + self.p[q[i], t - 1])
                    departures[q[i], t + 1] = departures[q[i], t] + self.p[q[i], t]
            obj = np.sum(departures-self.d)
            return obj, departures
        else:
            departures = copy.deepcopy(self.inc_departures)
            # departure time at the remaining locations
            for s in range(0, self.num_sta - 1):
                t = 2 * s + 1
                # a queue of trains ordered by departure times at t+1 (track) in ascending order
                q = A[s]
                if s >= m[0]:
                    # first train's departure times at location t (track) and location t+1 (station)
                    departures[q[0], t] = departures[q[0], t - 1] + self.p[q[0], t - 1]
                    departures[q[0], t + 1] = departures[q[0], t] + self.p[q[0], t]
                    # remaining trains' departure times at location t (track) and location t+1 (station)
                    for i in range(1, self.num_trn):
                        departures[q[i], t] = max(departures[q[i - 1], t + 1], departures[q[i], t - 1] + self.p[q[i], t - 1])
                        departures[q[i], t + 1] = departures[q[i], t] + self.p[q[i], t]
            obj = np.sum(departures-self.d)
            return obj, departures

    def solve(self, verbose=False):
        """Solve the local search problem using best accept strategy."""
        if self.iter == 0:
            # if verbose, print initial solution
            if verbose:
                print("Initial solution:")
                print(self.inc_A)
                print(self.inc_obj)
                print()
            else:
                # if not verbose, initialize progress bar
                pbar = tqdm(total=self.max_iter)

        while self.iter < self.max_iter:
            self.iter += 1
            if verbose:
                print("====Iteration", self.iter, "====")
                print("Neighbours:")
            else:
                pbar.update(1)

            # list of candidate solutions (objective values and A)
            candidate_m = []
            candidate_A = []
            candidate_obj = []
            moves = copy.deepcopy(self.moves)
            for i in range(0, len(moves)):
                # swapping two trains x and y for at track t if this move is not in tabu list
                _m = moves[i]
                if _m not in self.tabu:
                    if verbose:
                        print(moves[i], " not in tabu list", self.tabu)
                    _A = self.neighbour(_m)
                    _obj, _ = self.obj_and_departures(_A, m=_m)
                    candidate_m.append(_m)
                    candidate_A.append(_A)
                    candidate_obj.append(_obj)
                else:
                    if verbose:
                        print(_m, " IS  in tabu list", self.tabu)
                        # modification
                        # _A = neighbours[i]
                        # _obj, _ = self.obj_and_departures(_A)
                        # if _obj < self.best_obj:
                        #     candidate_m.append(_m)
                        #     candidate_A.append(_A)
                        #     candidate_obj.append(_obj)
            assert len(candidate_m) == len(candidate_A) == len(candidate_obj)

            # if there is no candidates, stop solving
            if not candidate_obj:
                self.iter -= 1
                if verbose:
                    print("No move can be made")
                    print("Tabu list:")
                    print(self.tabu)
                    print("Best solution:")
                    print(self.best_A)
                    print(self.best_obj)
                    print()
                else:
                    pbar.close()
                break
            else:
                # pick the move, update tabu list
                m = candidate_m[int(np.argmin(candidate_obj))]
                A = candidate_A[int(np.argmin(candidate_obj))]
                self.tabu.append(m)
                if len(self.tabu) > self.tabu_size:
                    self.tabu.pop(0)
                # no duplicates in tabu list (no move in tabu list is made)
                assert len(self.tabu) == len(set(self.tabu))

                # make the move, update incumbent
                self.inc_A = copy.deepcopy(A)
                self.inc_obj, self.inc_departures = copy.deepcopy(self.obj_and_departures(A))
                self.obj_his.append(self.inc_obj)

                # update best solution, best objective value and best departures
                if self.inc_obj < self.best_obj:
                    self.best_A = copy.deepcopy(self.inc_A)
                    self.best_obj = copy.deepcopy(self.inc_obj)
                    self.best_departures = copy.deepcopy(self.inc_departures)

                if verbose:
                    print("Move:", _m)
                    print("Tabu list:")
                    print(self.tabu)
                    print("Incumbent solution:")
                    print(self.inc_A)
                    print(self.inc_obj)
                    print()
        if not verbose:
            pbar.close()
        sleep(0.01)

    def display_result(self, plot=False):
        """Display local search information."""
        # print("\n\n====Problem setup====")
        # print("Number of stations: ", self.num_sta)
        # print("Number of trains:", self.num_trn)
        print("\n====Result====")
        print("Best solution:\n", self.best_A)
        print("Best objective value:\n", self.best_obj)
        # print("Best departures:\n", self.best_departures)

        # plot evolution of objective values
        if plot:
            x = np.arange(self.iter, dtype=int)
            plt.plot(x, self.obj_his)
            plt.title(str(self.num_sta)+" stations, "+str(self.num_trn)+" trains, tabu list size: "+str(self.tabu_size))
            plt.xlabel("Iteration")
            plt.ylabel("Objective value")
            plt.savefig(self.filename + "_iter_" + str(self.max_iter) + "_size_" + str(self.tabu_size) + '.png')
            plt.clf()


if __name__ == "__main__":
    m_iter = 100
    tb_size = 7
    folder_location = os.getcwd() + "\mie562_instances"
    instance_name = "8_12.txt"
    ts = SingleTrackTS(folder_location, instance_name, max_iter=m_iter, tabu_size=tb_size)
    start_time = time()
    ts.solve(verbose=False)
    solving_time = time() - start_time
    ts.display_result()
    print(solving_time)
