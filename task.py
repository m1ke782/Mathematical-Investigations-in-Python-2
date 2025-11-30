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
        
    def cost(self, S) : 
        # keep track of the total length
        length = 0

        # for each carer
        for carer in range(self.n1) :
            # add lengths for each client the carer visits
            last = carer
            for client in S[carer] : 
                length += self.distance_matrix[last][client]
                last = client
            # add the final length for the carer to go back home
            length += self.distance_matrix[last][carer]

        return length
    
    def random_solution(self) : 
        # initialise n1 empty routes
        S = [[] for i in range(self.n1)]

        # keep track of which carers are still available for more appointments (they have less than m visits schedules)
        available_carers = list(range(0,self.n1))

        # for each client...
        for i in range(self.n1, self.n1+self.n2) : 
            # assign them a random carer; add this client to the carer's route
            carer = random.choice(available_carers)
            S[carer].append(i)

            # if adding this client makes the carer have m clients (ie len(s) >= m), then remove this carer from the list of available carers
            if len(S[carer]) >= self.m : 
                available_carers.remove(carer)

        # return the random solution
        return S
    
    def random_tweak(self, S) : 
        # copy the solution
        new_S = [s.copy() for s in S]

        # choose a client to move
        client = random.randint(self.n1,self.n1+self.n2)

        # find the carer responsible for this client
        carer = None
        for i in range(self.n1) :
            if client in S[i] : 
                carer = i
                break

        # find a carer which can accomodate this client
        possible_new_carers = []
        for i in range(self.n1) : 
            if i == carer or len(new_S[i]) < self.m : 
                possible_new_carers.append(i)

        # remove the client from the old carer, and add it randomly somwhere to a new one
        new_S[carer].remove(client)
        new_carer = random.choice(possible_new_carers)
        new_S[new_carer].insert(random.randint(0,len(new_S[new_carer])), client)

        return new_S
            

    def draw(self, S=[]) : 
        plt.figure()

        # for each carer
        for carer in range(self.n1) :
            # draw lines from the last client to the next one
            last = carer
            for client in S[carer] : 
                plt.plot([self.nodes[last][0], self.nodes[client][0]], [self.nodes[last][1], self.nodes[client][1]], "b")
                last = client
            # draw the final line going back to the start
            plt.plot([self.nodes[last][0], self.nodes[carer][0]], [self.nodes[last][1], self.nodes[carer][1]], "b")

        # draw all of the carer nodes in black
        for i in range(self.n1) : 
            plt.plot(self.nodes[i][0], self.nodes[i][1], "ko")
            plt.text(self.nodes[i][0], self.nodes[i][1], "v"+str(i), color="g")

        # draw all of the client nodes in red
        for i in range(self.n1, self.n1+self.n2) : 
            plt.plot(self.nodes[i][0], self.nodes[i][1], "ro")
            plt.text(self.nodes[i][0], self.nodes[i][1], "v"+str(i), color="g")

        # show the figure
        plt.show()