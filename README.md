# SingleTrackTS
A Tabu search algorithm for solving one-way-single-track scheduling problem.

Problem Setup:\
A brief description of a one-way-single-trakc system. When a train departs at a station, it enters that stations and stops for some time. When a train departs at a segment, it enters the segment and travels to the next station.

Each train stops at each station for a certain amount of time ideally, and travels between two stations using a certain amount of time. Ideal departure times at each segment are given and the travel times between two stations are also given.

Rules:\
A train cannot depart earlier than its ideal departure time.
A train can stay in a station longer than its ideal docking time.
Unlimited number of trains can dock in a station, while at most one train can be present on a track segment between two stations.
Trains can pass each other in the station.

Objective:\
The objective is to minmize the total delays of departure times.

Instance Format:\
First line is the number of stations.\
Second line is the number of trains.\
In the following (number of trains) lines, each line has ((number of stations - 1) * 2) time points, sepcifying the travel times between 
two stations (for a track segment between two sations) or the ideal docking time in a station (for a station).\
In the following (number of trains) lines, each line has (number of stations * 2 - 1) time points, sepcifying the depature times at each segment.
