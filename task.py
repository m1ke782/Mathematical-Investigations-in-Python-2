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
        client = random.randint(self.n1,self.n1+self.n2-1)

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
    
    def all_tweaks(self, S) :
        # keep track of all of the tweaks
        tweaks = []

        # for every carer
        for carer in range(self.n1) : 
            # for each client this carer cares for
            for original_client_position in range(len(S[carer])) : 
                # for each carer
                for new_carer in range(self.n1) : 
                    # ignore if this carer can't take on another client
                    if new_carer != carer and len(S[new_carer]) >= self.m : 
                        continue

                    for new_client_position in range(len(S[new_carer])) :
                        # not an interesting tweak
                        if new_carer == carer and original_client_position == new_client_position : 
                            continue

                        # create a new empty route
                        new_S = [s.copy() for s in S]

                        # remove the client from its old position
                        client = new_S[carer][original_client_position]
                        del new_S[carer][original_client_position]

                        # add the client to its new position
                        new_S[new_carer].insert(new_client_position, client)

                        # add the tweak
                        tweaks.append(new_S)
                
        return tweaks
            

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

        # add a title
        plt.title("Cost : " + str(self.cost(S)))

        # show the figure
        plt.show()
        
    def basic_local_search(self, trials) : 
        # start with a random solution
        solution = self.random_solution()
        cost = self.cost(solution)
        
        # keep track of the cost over many iterations
        cost_over_iterations = []
        
        # many many times...
        for i in range(trials) : 
            # find a random neighbour, and compare its cost to the 
            random_neighbour = self.random_tweak(solution)
            random_neighbour_cost = self.cost(random_neighbour)
        
            # if it has a lower cost, keep it
            if random_neighbour_cost < cost : 
                solution = random_neighbour
                cost = random_neighbour_cost
                
            # add this cost to our graph
            cost_over_iterations.append(cost)
        
        # print the output
        print("We found the solution : ", solution)
        print("Its solution has cost : ", cost)
        
        # draw the solution
        self.draw(solution)
        
        # draw the cost against iterations
        plt.figure()
        plt.plot(cost_over_iterations)
        plt.show()
        
    def mutliple_basic_local_search(self, trials, restarts) : 
        # keep track of the best solution
        best_solution = None
        best_cost = None
        
        # keep track of the cost over many iterations
        cost_over_iterations = []
        
        # many many times...
        for j in range(restarts) : 
            # start with a random solution
            solution = self.random_solution()
            cost = self.cost(solution)
            
            for i in range(trials) : 
                # find a random neighbour, and compare its cost to the 
                random_neighbour = self.random_tweak(solution)
                random_neighbour_cost = self.cost(random_neighbour)
            
                # if it has a lower cost, keep it
                if random_neighbour_cost < cost : 
                    solution = random_neighbour
                    cost = random_neighbour_cost
                
            # change the best solution if this solution is better
            if best_cost == None or cost < best_cost :
                best_solution = solution
                best_cost = cost
                
            # add this cost to our graph
            cost_over_iterations.append(best_cost)
        
        # print the output
        print("We found the solution, : ", best_solution)
        print("Its solution has cost : ", best_cost)
        
        # draw the solution
        self.draw(best_solution)
        
        # draw the cost against iterations
        plt.figure()
        plt.plot(cost_over_iterations)
        plt.show()
        
    def best_neighbour(self) : 
        # start with a random solution
        solution = self.random_solution()
        cost = self.cost(solution)
        
        # keep track of the cost over many iterations
        cost_over_iterations = []
        
        # many many times...
        while True : 
            # find all neighbours of the current solution
            all_neighbours = self.all_tweaks(solution)
            
            # keep track of the best neighbour
            best_neighbour = None
            best_neighbour_cost = None
            
            # find the best neighbour
            for neighbour in all_neighbours : 
                this_neighbour_cost = self.cost(neighbour)
                if best_neighbour_cost == None or this_neighbour_cost < best_neighbour_cost :
                    best_neighbour = neighbour
                    best_neighbour_cost = this_neighbour_cost
            
            # terminate if the best neighbour doesn't exist or is worse than the current one
            if best_neighbour_cost == None or best_neighbour_cost >= cost :
                break
            
            # otherwise, move to this neighbour
            solution = best_neighbour
            cost = best_neighbour_cost
            cost_over_iterations.append(cost)
            
        # print the output
        print("We found the solution : ", solution)
        print("Its solution has cost : ", cost)
            
        # draw the solution
        self.draw(solution)
        
        # draw the cost against iterations
        plt.figure()
        plt.plot(cost_over_iterations)
        plt.show()

    def multiple_best_neighbour(self, restarts) : 
        # keep track of the best solution
        best_solution = None
        best_cost = None
        
        # keep track of the cost over many iterations
        cost_over_iterations = []
        
        # many many times...
        for i in range(restarts) : 
            # get a random solution
            solution = self.random_solution()
            cost = self.cost(solution)

            while True : 
                # find all neighbours of the current solution
                all_neighbours = self.all_tweaks(solution)
                
                # keep track of the best neighbour
                best_neighbour = None
                best_neighbour_cost = None
                
                # find the best neighbour
                for neighbour in all_neighbours : 
                    this_neighbour_cost = self.cost(neighbour)
                    if best_neighbour_cost == None or this_neighbour_cost < best_neighbour_cost :
                        best_neighbour = neighbour
                        best_neighbour_cost = this_neighbour_cost
                
                # terminate if the best neighbour doesn't exist or is worse than the current one
                if best_neighbour_cost == None or best_neighbour_cost >= cost :
                    break
                
                # otherwise, move to this neighbour
                solution = best_neighbour
                cost = best_neighbour_cost

                # change the best solution if this solution is better
                if best_cost == None or cost < best_cost :
                    best_solution = solution
                    best_cost = cost

                # add this cost to our graph
                cost_over_iterations.append(best_cost)
            
        # print the output
        print("We found the solution : ", best_solution)
        print("Its solution has cost : ", best_cost)
            
        # draw the solution
        self.draw(best_solution)
        
        # draw the cost against iterations
        plt.figure()
        plt.plot(cost_over_iterations)
        plt.show()
        
    def tabu_search(self, trials) :
        # keep track of the best solution
        best_solution = None
        best_cost = None
        
        # start with a random solution
        solution = self.random_solution()
        cost = self.cost(solution)
        
        # start with a list of banned solutions
        banned_solutions = set()
        
        # keep track of the cost against iterations
        cost_over_iterations = []
        
        # many many times...
        for i in range(trials) : 
            # find all neighbours of the current solution
            all_neighbours = self.all_tweaks(solution)
            
            # keep track of the best neighbour
            best_neighbour = None
            best_neighbour_cost = None
            
            # find the best non-baned neighbour
            for neighbour in all_neighbours : 
                # ignore banned neighbour
                if str(neighbour) in banned_solutions : 
                    continue
                
                this_neighbour_cost = self.cost(neighbour)
                if best_neighbour_cost == None or this_neighbour_cost < best_neighbour_cost :
                    best_neighbour = neighbour
                    best_neighbour_cost = this_neighbour_cost
            
            # ban the current solution
            banned_solutions.add(str(solution))
            
            # otherwise, move to this neighbour
            solution = best_neighbour
            cost = best_neighbour_cost
            
            # keep track of the best solution
            if best_cost == None or cost < best_cost : 
                best_solution = solution
                best_cost = cost
                
            # keep track of the cost per iterations
            cost_over_iterations.append(best_cost)
            
            
        # print the output
        print("We found the solution : ", best_solution)
        print("Its solution has cost : ", best_cost)
             
        # draw the solution
        self.draw(best_solution)
         
        # draw the cost against iterations
        plt.figure()
        plt.plot(cost_over_iterations)
        plt.show()