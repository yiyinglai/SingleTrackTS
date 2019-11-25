import copy
import sys
import os.path
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
import xlsxwriter
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=300)


class SingleTrackTS:
    """Single Track tabu Search Algorithm.
    It reads an instance and find optimal schedule using tabu search with specified strategy.

    Initial solution: same departing sequence at all tracks, sequence based on the departure times at the first track.
    Neighbours: swapping adjacent trains in a departing sequence on a track.
    Objective function: the sum of departure time delays at each location.

    In a 5-station-4-train instance, there is 9 locations (stations and tracks).
    location    0   1   2   3   4   5   6   7   8
    type        s   t   s   t   s   t   s   t   s

    Attributes:
        strategy(str): Strategy of TS, "first accept" or "best accept"
        iter(int): Number of iterations in tabu search
        max_iter(int): Maximum number of iteration in tabu search
        tabu([move]): A list of moves, where a move (t, i) means swapping A[t, i] and A[t, i+1]
        tabu_size(int): size of tabu list
        obj_his(numpy.ndarray[float]): History of objective values at each iteration
        num_sta(int): Number of stations of the instance
        num_trn(int): Number of trains of the instance
        num_tot(int): Number of locations of the instance
        p(numpy.ndarray[float]): Processing times of all trains at all locations
        d(numpy.ndarray[float]): Ideal departure times of all trains at all locations
        inc_A(numpy.ndarray[int]): Incumbent departing sequences of trains on all tracks (5 stations -> 4 sequences)
        inc_departures(numpy.ndarray[float]): Incumbent departure times
        inc_obj(float): Incumbent objective value
        best_A(numpy.ndarray[int]): Best departing sequences of trains on all tracks
        best_obj(int): Best objective value
        best_departures(numpy.ndarray[float]): Best departure times
    """

    def __init__(self, folder_loc, file_name, max_iter=500, tabu_size=8, strategy="best accept"):
        """Initialize the problem by reading an instance and find initial solution"""

        self.strategy = strategy
        self.iter = 0
        self.max_iter = max_iter
        self.tabu = []
        self.tabu_size = tabu_size
        self.obj_his = []
        self.filename = file_name
        self.solving_time = None

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

        # initial solution: same departing sequence on all tracks
        # sequence based on the departure times on the first track (location 1 d[:, 1])
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
        """Return new departing sequences based on incumbent solution and move"""
        t = move[0]
        i = move[1]
        new_A = copy.deepcopy(self.inc_A)
        new_A[t, i], new_A[t, i + 1] = new_A[t, i + 1], new_A[t, i]
        return new_A

    def obj_and_departures(self, A, m=None):
        """Calculate and return objective value and actual departure times based on A, d and p
            A[k]: sequence of train No. at track k (5 stations -> k = 4)
        """
        # this block is used for solving initial solution, when m=None
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

        # this block is for evaluating new solution, it uses incumbent information and runs faster
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

    def solve(self):
        """Solve the SingleTrack scheduling problem using tabu search with specified strategy."""
        start_solving = time()
        # initialize progress bar
        pbar = tqdm(total=self.max_iter)

        # "best accept" strategy
        if self.strategy == "best accept":
            while self.iter < self.max_iter:
                self.iter += 1
                pbar.update(1)

                # list of candidate solutions (objective values and A)
                candidate_m = []
                candidate_A = []
                candidate_obj = []
                moves = copy.deepcopy(self.moves)

                for i in range(0, len(moves)):
                    # calculate obj and departures regardless of the legitimacy of the move
                    _m = moves[i]
                    _A = self.neighbour(_m)
                    # calculate obj and departures in a faster way
                    _obj, _ = self.obj_and_departures(_A, m=_m)
                    # if this move is not on tabu list
                    if _m not in self.tabu:
                        candidate_m.append(_m)
                        candidate_A.append(_A)
                        candidate_obj.append(_obj)
                assert len(candidate_m) == len(candidate_A) == len(candidate_obj)

                # if a move can be made
                if candidate_obj:
                    # end of each exploration, pick a move
                    m = candidate_m[int(np.argmin(candidate_obj))]
                    A = candidate_A[int(np.argmin(candidate_obj))]
                    # update tabu list
                    self.tabu.append(m)
                    if len(self.tabu) > self.tabu_size:
                        self.tabu.pop(0)
                    # update incumbent solution
                    self.inc_A = copy.deepcopy(A)
                    self.inc_obj, self.inc_departures = copy.deepcopy(self.obj_and_departures(A))
                    self.obj_his.append(self.inc_obj)
                    # update best solution
                    if self.inc_obj <= self.best_obj:
                        self.best_A = copy.deepcopy(self.inc_A)
                        self.best_obj = copy.deepcopy(self.inc_obj)
                        self.best_departures = copy.deepcopy(self.inc_departures)
                # no move can be made (when tabu size is greater than neighbour size)
                else:
                    self.iter -= 1
                    break

        # "first accept" strategy
        elif self.strategy == "first accept":
            while self.iter < self.max_iter:
                self.iter += 1
                pbar.update(1)

                moves = copy.deepcopy(self.moves)
                for i in range(0, len(moves)):
                    # calculate obj and departures regardless of the legitimacy of the move
                    _m = moves[i]
                    _A = self.neighbour(_m)
                    _obj, _departures = self.obj_and_departures(_A, m=_m)
                    # if this move is not on tabu list
                    if _m not in self.tabu:
                        # better than incumbent solution, accept immediately and break the for loop
                        if _obj <= self.inc_obj:
                            self.inc_A = _A.copy()
                            self.inc_obj = _obj.copy()
                            self.inc_departures = _departures.copy()
                            # update best solution
                            if _obj <= self.best_obj:
                                self.best_A = _A.copy()
                                self.best_obj = _obj.copy()
                                self.best_departures = _departures.copy()
                            break
                        # no better than incumbent solution, evaluate next move in the for loop
                        else:
                            continue
                    # if this move is on tabu list and yields best result, accept immediately and break the for loop
                    elif _obj <= self.best_obj:
                        self.inc_A = _A.copy()
                        self.inc_obj = _obj.copy()
                        self.inc_departures = _departures.copy()
                        # update best solution
                        self.best_A = _A.copy()
                        self.best_obj = _obj.copy()
                        self.best_departures = _departures.copy()
                        break
                    # the move is on tabu list, and no better than best solution, evaluate next move in the for loop
                    else:
                        continue
                # finished evaluating neighbours of incumbent solution, record incumbent objective value
                self.obj_his.append(self.inc_obj)

        # finish solving
        self.solving_time = time() - start_solving
        pbar.close()

    def display_result(self):
        """Display tabu search result and solving time."""
        print("\n====Result====")
        print("Best solution:\n", self.best_A)
        print("Best departures:\n", self.best_departures)
        print("Best objective value:\n", self.best_obj)
        print("Solving time:\n", self.solving_time)

    def save_plot(self, show=False):
        """Save the evolution of objective values in a plot."""
        x = np.arange(self.iter, dtype=int)
        plt.plot(x, self.obj_his)
        plt.title(self.filename[:-4] + "(" + self.strategy +
                  ",iter=" + str(self.max_iter) + ",size=" + str(self.tabu_size) + ")")
        plt.xlabel("Iteration")
        plt.ylabel("Objective value")
        figurename = self.filename[:-4] + "(" + self.strategy + ",iter=" + str(self.max_iter) + ",size=" + str(self.tabu_size) + ').png'
        figurepath = os.getcwd() + r"\Final report\results"
        path_filename = os.path.join(figurepath, figurename)
        # save figure if it does not exist
        if not os.path.exists(path_filename):
            plt.savefig(path_filename)
        if show:
            plt.show()
        plt.clf()


if __name__ == "__main__":
    # same maximum iteration and tabu list size for all instances
    m_iter = 300
    tb_size = 16
    strategy = 'best accept'

    # paths and names
    instances_path = os.getcwd() + r"\Final report\instances"
    file_dirs = os.listdir(instances_path)
    results_path = os.getcwd() + r"\Final report\results"
    # noinspection SpellCheckingInspection
    xls_name = r"\Results for (" + strategy + ',iter=' + str(m_iter) + ',size=' + str(tb_size) + ').xlsx'

    with xlsxwriter.Workbook(os.path.join(results_path, xls_name)) as workbook:
        worksheet = workbook.add_worksheet()
        worksheet.write('A1', 'Instance')
        worksheet.write('B1', 'Objective value')
        worksheet.write('C1', 'Time to solve')
        for i, instance_name in enumerate(file_dirs):
            ts = SingleTrackTS(instances_path, instance_name, max_iter=m_iter, tabu_size=tb_size, strategy=strategy)
            ts.solve()
            # ts.display_result()
            ts.save_plot()
            worksheet.write('A' + str(i+2), instance_name[:-4])
            worksheet.write('B' + str(i+2), getattr(ts, 'best_obj'))
            worksheet.write('C' + str(i+2), getattr(ts, 'solving_time'))
