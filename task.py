import matplotlib.pyplot as plt
import random

class ProblemInstance : 
    def __init__(self, n1, n2, m, nodes, distance_matrix) : 
        self.n1 = n1
        self.n2 = n2
        self.m = m
        self.nodes = nodes
        self.distance_matrix = distance_matrix

    def from_file(path) :
        # open the file
        with open(path, "r") as file : 
            # on the first line, read n1, n1 and m
            line = file.readline().strip().split(" ")
            n1 = int(line[0])
            n2 = int(line[1])
            m = int(line[2])

            # read the next n1+n2 lines to find all the node coordinates
            nodes = []
            for i in range(n1+n2) :
                l = file.readline().strip().split(" ")
                nodes.append([float(l[0]), float(l[1])])

            # read the next n1+n2 lines to find the rows for the distance matrix
            distance_matrix = []
            for i in range(n1+n2) : 
                l = file.readline().strip().split(" ")
                row = [float(c) for c in l]
                distance_matrix.append(row)

            # return this data as a problem instance object
            return ProblemInstance(n1, n2, m, nodes, distance_matrix)
        
    def length_of_solution(self, S) : 
        # keep track of the total length
        length = 0

        # for each route s in S
        for s in S : 
            # for every node in s
            for i in range(len(s)) :   
                # find the next node from this one, and add the distance from this node to the next one 
                next = (i+1)%len(s)
                length += self.distance_matrix[s[i]][s[next]]

        # return the total length
        return length
    
    def random_solution(self) : 
        # initialise n1 routes, where each carer starts at their own home
        S = [[i] for i in range(self.n1)]

        # keep track of which carers are still available for more appointments (they have less than m visits schedules)
        available_carers = list(range(0,self.n1))

        # for each client...
        for i in range(self.n1, self.n1+self.n2) : 
            # assign them a random carer; add this client to the carer's route
            carer = random.choice(available_carers)
            S[carer].append(i)

            # if adding this client makes the carer have m clients (ie len(s) >= m+1), then remove this carer from the list of available carers
            if len(S[carer]) > self.m : 
                available_carers.remove(carer)

        # return the random solution
        return S
    
    def random_tweak(self, S) : 
        # copy the solution
        new_S = [s.copy() for s in S]

        # chose a client at random to move
        carer = random.randint(0,self.n1-1)
        client = random.choice(new_S[carer])

        # find a carer which can accomodate this client
        possible_new_carers = []
        for i in range(self.n1) : 
            if i == carer or len(new_S[i]) <= self.m : 
                possible_new_carers.append(i)

        # remove the client from the old carer, and add it randomly somwhere to a new one
        new_S[carer].remove(client)
        new_carer = random.choice(possible_new_carers)
        new_S[new_carer].insert(random.randint(0,len(new_S[new_carer])), client)

        return new_S
            

    def draw(self, S=[]) : 
        plt.figure()

        # for each route...
        for s in S : 
            # for each node in this route...
            for i in range(len(s)) : 
                # draw a blue line from this node to the next node in the route
                next = (i+1)%len(s)
                plt.plot([self.nodes[s[i]][0], self.nodes[s[next]][0]], [self.nodes[s[i]][1], self.nodes[s[next]][1]], "b")

        # draw all of the carer nodes in black
        for i in range(self.n1) : 
            plt.plot(self.nodes[i][0], self.nodes[i][1], "ko")

        # draw all of the client nodes in red
        for i in range(self.n2) : 
            plt.plot(self.nodes[self.n1+i][0], self.nodes[self.n1+i][1], "ro")

        # show the figure
        plt.show()