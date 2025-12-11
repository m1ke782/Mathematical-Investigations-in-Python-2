import matplotlib.pyplot as plt
import random
import math
import scipy
import time
import statistics


def convergence_order(cost) : 
    return -scipy.stats.linregress([math.log(i) for i in range(1,len(cost)+1)], [math.log(cost[i-1]) for i in range(1,len(cost)+1)]).slope

def convergence_speed(cost, t) : 
    if t == 0 : 
        return None
    return (cost[0] - cost[-1]) / t

def exponential_probability_selector(costs, T, alpha) : 
    return [math.exp(-alpha*cost/T) for cost in costs]

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

    def length_of_route(self, S, carer) : 
        # keep track of the length of this route
        length = 0

        # add lengths for each client the carer visits
        last = carer
        for client in S[carer] : 
            length += self.distance_matrix[last][client]
            last = client

        # add the final length for the carer to go back home
        length += self.distance_matrix[last][carer]

        # return the length of this route
        return length

        
    def cost(self, S) : 
        # the cost is the sum of the lengths of each route
        return sum(self.length_of_route(S, i) for i in range(self.n1))
    
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
    
    def random_tweak(self, S, cost=False) : 
        # copy the solution
        new_S = [s.copy() for s in S]

        # choose a random client to move
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

        # give the cost if they asked for it
        if cost : 
            if new_carer == carer : 
                return (new_S, self.length_of_route(new_S, carer) - self.length_of_route(S, carer))
            else : 
                return (new_S, self.length_of_route(new_S, carer) + self.length_of_route(new_S, new_carer) - self.length_of_route(S, carer) - self.length_of_route(S, new_carer))
            
        # otherwise, just give the tweak
        return new_S
    
    def all_tweaks(self, S, costs=False) :
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

                        # add the cost if the user asks for it
                        if costs : 
                            if new_carer == carer : 
                                tweaks.append((new_S, self.length_of_route(new_S, carer) - self.length_of_route(S, carer)))
                            else : 
                                tweaks.append((new_S, self.length_of_route(new_S, carer) + self.length_of_route(new_S, new_carer) - self.length_of_route(S, carer) - self.length_of_route(S, new_carer)))

                        # otherwise, just add the tweak
                        else : 
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
            random_neighbour, delta_cost = self.random_tweak(solution, True)
        
            # if it has a lower cost, keep it
            if delta_cost < 0 : 
                solution = random_neighbour
                cost += delta_cost
                
            # add this cost to our graph
            cost_over_iterations.append(cost)

        # return the output
        return solution, cost, cost_over_iterations
    
        
    def best_neighbour(self, start_from=None) : 
        # start with a random solution
        solution = start_from if start_from != None else self.random_solution()
        cost = self.cost(solution)
        
        # keep track of the cost over many iterations
        cost_over_iterations = []
        
        # many many times...
        while True : 
            # find all neighbours of the current solution
            all_neighbours = self.all_tweaks(solution, True)
            
            # keep track of the best neighbour
            best_neighbour = None
            best_neighbour_cost = None
            
            # find the best neighbour
            for neighbour in all_neighbours : 
                this_neighbour_cost = neighbour[1] + cost
                if best_neighbour_cost == None or this_neighbour_cost < best_neighbour_cost :
                    best_neighbour = neighbour[0]
                    best_neighbour_cost = this_neighbour_cost
            
            # terminate if the best neighbour doesn't exist or is worse than the current one
            if best_neighbour_cost == None or best_neighbour_cost >= cost :
                break
            
            # otherwise, move to this neighbour
            solution = best_neighbour
            cost = best_neighbour_cost
            cost_over_iterations.append(cost)
            
        return solution, cost, cost_over_iterations
        
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
            all_neighbours = self.all_tweaks(solution, True)
            
            # keep track of the best neighbour
            best_neighbour = None
            best_neighbour_cost = None
            
            # find the best non-baned neighbour
            for neighbour in all_neighbours : 
                # ignore banned neighbour
                if str(neighbour[0]) in banned_solutions : 
                    continue
                
                this_neighbour_cost = neighbour[1] + cost
                if best_neighbour_cost == None or this_neighbour_cost < best_neighbour_cost :
                    best_neighbour = neighbour[0]
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
            
            
        # return the output
        return best_solution, best_cost, cost_over_iterations

    def simulated_annealing(self, trials, T_0, alpha) : 
        # keep track of the best solution
        best_solution = None
        best_cost = None

        # start with a random solution
        solution = self.random_solution()
        cost = self.cost(solution)
        
        # keep track of the cost over many iterations
        cost_over_iterations = []
        
        # many many times...
        for i in range(trials) : 
            # find all neighbours and their weighted probabilities
            neighbours = self.all_tweaks(solution, True)
            weights = exponential_probability_selector([neighbour[1]+cost for neighbour in neighbours], (T_0*(i+1)/trials), alpha)

            # select one at random with these weights
            move_to = random.choices(neighbours, weights)[0]
            solution = move_to[0]
            cost += move_to[1]
                
            # update the best solution
            if best_cost == None or cost < best_cost : 
                best_cost = cost
                best_solution = solution

            # add this cost to our graph
            cost_over_iterations.append(best_cost)
        
        # return the output
        return best_solution, best_cost, cost_over_iterations

    def great_deluge(self, trials, delta_level) : 
        # keep track of the best solution
        best_solution = None
        best_cost = None

        # start with a random solution
        solution = self.random_solution()
        cost = self.cost(solution)

        # initialise the water level
        water_level = cost

        # keep track of the cost over many iterations
        cost_over_iterations = []

         # many many times...
        for i in range(trials) : 
            # find a random neighbour and its cost
            neighbour, delta_cost = self.random_tweak(solution, True)
            neighbour_cost = cost + delta_cost

            # if its an improvement or lower than the water level, move to it
            if neighbour_cost < max(cost, water_level): 
                solution = neighbour
                cost = neighbour_cost
                
            # lower the water level
            water_level -= delta_level

            # keep track of the best solution
            if best_cost == None or cost < best_cost : 
                best_cost = cost
                best_solution = solution

            # add this cost to our graph
            cost_over_iterations.append(best_cost)

        # return the output
        return best_solution, best_cost, cost_over_iterations
        
    def test_cost_speedup(self, trials) : 
        # do this the old fashioned way
        start = time.time_ns()
        for i in range(trials) : 
            S = self.random_solution()
            cost = self.cost(S)
            S_star = self.all_tweaks(S)
            for Sp in S_star : 
                c = self.cost(Sp)
        print("The old method takes ", (time.time_ns()-start)/trials, "ns")

        # do this the new fashioned way
        start = time.time_ns()
        for i in range(trials) : 
            S = self.random_solution()
            cost = self.cost(S)
            S_star = self.all_tweaks(S,True)
            for Sp,c in S_star : 
                c =c+cost
        print("The new method takes ", (time.time_ns()-start)/trials, "ns")

    def test_optimiser(self, optimisier, sample_size) : 
        # keep track of the best solution
        best_solution = None
        best_cost = None

        # keep track of the samples and time
        cost_samples = []
        speed_samples = []
        start_time = time.time_ns()
        
        # keep track of the cost over many iterations
        cost_over_iterations = []
        
        # many many times...
        for i in range(sample_size) : 
            # perform optimisier
            time_for_iteration = time.time_ns()
            solution, cost, cost_over_iterations_for_sample = optimisier()
            time_for_iteration = time.time_ns() - time_for_iteration

            # change the best solution if this solution is better
            if best_cost == None or cost < best_cost :
                best_solution = solution
                best_cost = cost

            # add this cost to our graph
            cost_samples.append(cost)
            speed_samples.append(convergence_speed(cost_over_iterations_for_sample, time_for_iteration))
            cost_over_iterations.append(best_cost)


        # calculate the time taken
        time_taken = time.time_ns() - start_time

        # remove invalid speed samples
        speed_samples = [speed for speed in speed_samples if speed != None]

        # print some information about the solution
        print("Found the solution   : ", best_solution)
        print("With cost            : ", best_cost)
        print("Time taken           : ", time_taken)
        
        if sample_size > 1 : 
            print("Convergence order    : ", convergence_order(cost_over_iterations))
            print("Convergence speed    : ", convergence_speed(cost_over_iterations, time_taken))
            print("Cost mean            : ", statistics.mean(cost_samples))
            print("Cost variance        : ", statistics.variance(cost_samples))
            print("Sample size          : ", sample_size)
            print("Speed mean           : ", statistics.mean(speed_samples))
            print("Speed variance       : ", statistics.variance(speed_samples))
            print("Sample size          : ", len(speed_samples))

        # make a figure of the best solution
        self.draw(best_solution)

        # make a plot of the cost against the number of iterations
        plt.figure()
        plt.title("Cost against Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.plot(cost_over_iterations)
        plt.show()

        
        if sample_size > 1 :
            # make a boxplot of the cost samples 
            plt.figure()
            plt.title("Costs")
            plt.boxplot(cost_samples)
            plt.show()

            # make a boxplot of the time samples
            plt.figure()
            plt.title("Convergence Speeds")
            plt.boxplot(speed_samples)
            plt.show()

    def test_all(self, sample_size) : 
        # keep track of all statistics
        all_cost_samples = []
        all_speed_samples = []
        
        # perform tests for all optimisers
        optimisers = [
            lambda : self.basic_local_search(10000),
            lambda : self.best_neighbour(),
            lambda : self.tabu_search(100),
            lambda : self.simulated_annealing(100, 1000, 2),
            lambda : self.great_deluge(10000, 0.1)
        ]

        for optimiser in optimisers : 
            # keep track of the samples and time
            cost_samples = []
            speed_samples = []
            
            # many many times...
            for i in range(sample_size) : 
                # perform optimisier
                time_for_iteration = time.time_ns()
                solution, cost, cost_over_iterations_for_sample = optimiser()
                time_for_iteration = time.time_ns() - time_for_iteration

                # add this cost to our graph
                cost_samples.append(cost)
                speed_samples.append(convergence_speed(cost_over_iterations_for_sample, time_for_iteration))

            # remove invalid speed samples
            speed_samples = [speed for speed in speed_samples if speed != None]

            # add samples to make picture
            all_cost_samples.append(cost_samples)
            all_speed_samples.append(speed_samples)

            # print some information about the solution
            print("-----~-----")
            print("Best cost            : ", min(cost_samples))
            print("Cost mean            : ", statistics.mean(cost_samples))
            print("Cost variance        : ", statistics.variance(cost_samples))
            print("Sample size          : ", sample_size)
            print("Speed mean           : ", statistics.mean(speed_samples))
            print("Speed variance       : ", statistics.variance(speed_samples))
            print("Sample size          : ", len(speed_samples))

        # draw boxplots
        plt.figure()
        plt.title("Costs")
        plt.boxplot(all_cost_samples, positions=[i*100 for i in range(len(all_cost_samples))], widths=50)
        plt.show()

        plt.figure()
        plt.title("Speeds")
        plt.boxplot(all_speed_samples, positions=[i*100 for i in range(len(all_speed_samples))], widths=50)
        plt.show()


    def test_simulated_annealing_sharpness(self, trials, T_0) : 
        plt.figure()
        plt.title("Cost against Iterations for several values of alpha")

        for alpha in range(1,10) : 
            solution, cost, cost_against_iterations = self.simulated_annealing(trials, T_0, alpha)
            plt.plot(cost_against_iterations, label=str(alpha))

        plt.legend()
        plt.show()


def main() : 
    # read the problem file
    file_name = input("Enter problem file name >>> ")
    problem = ProblemInstance.from_file(file_name)

    # here are all of the things the user may want to do
    tasks = [
        {
            "func" : problem.test_cost_speedup, 
            "name" : "Test Cost Speedup", 
            "args" : [
                {"name":"Number of trials", "type":int}
            ]
        },
        {
            "func" : lambda samples : problem.test_all(samples), 
            "name" : "Test Everything", 
            "args" : [
                {"name":"Number of samples", "type":int}
            ]
        },
        {
            "func" : lambda trials, samples : problem.test_optimiser(lambda : problem.basic_local_search(trials), samples), 
            "name" : "Basic Local Search", 
            "args" : [
                {"name":"Number of trials", "type":int},
                {"name":"Number of samples", "type":int}
            ]
        },
        {
            "func" : lambda samples: problem.test_optimiser(problem.best_neighbour, samples), 
            "name" : "Best Neighbour", 
            "args" : [
                {"name":"Number of samples", "type":int}
            ]
        },
        {
            "func" : lambda trials, samples: problem.test_optimiser(lambda : problem.tabu_search(trials), samples), 
            "name" : "Tabu Search", 
            "args" : [
                {"name":"Number of trials", "type":int},
                {"name":"Number of samples", "type":int}
            ]
        },
        {
            "func" : lambda trials, t_0, alpha, samples: problem.test_optimiser(lambda : problem.simulated_annealing(trials, t_0, alpha), samples), 
            "name" : "Simulated Annealing", 
            "args" : [
                {"name":"Number of trials", "type":int},
                {"name":"Initial temperature T_0", "type":float},
                {"name":"Sharpness alpha", "type":float},
                {"name":"Number of samples", "type":int}
            ]
        },
        {
            "func" : lambda trials, delta_level, samples: problem.test_optimiser(lambda : problem.great_deluge(trials, delta_level), samples), 
            "name" : "Great Deluge", 
            "args" : [
                {"name":"Number of trials", "type":int},
                {"name":"Change in water level per iteration", "type":float},
                {"name":"Number of samples", "type":int}
            ]
        },
    ]

    # ask the user which task they wish to explore
    print("Which task do you wish to explore?")
    for i in range(len(tasks)) : 
        print(i, ") ", tasks[i]["name"])
    task = int(input(">>> "))

    # take in all of the arguments needed
    args = [arg["type"](input(arg["name"] + " >>> ")) for arg in tasks[task]["args"]]

    # run the task with all arguments
    tasks[task]["func"](*args)

if __name__ == "__main__" : 
    main()