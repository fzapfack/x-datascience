import itertools
import heapq
import operator
import igraph
import copy
import time

def terms_to_graph(terms, w):
    # returns a graph from a list of ordered terms
    # edges are weighted based on term co-occurence within a fixed-size sliding window

    t = time.time()
    counter = 0

    from_to = {}
    
    # create initial graph (first w terms)
    terms_temp = terms[0:w]
    indexes = list(itertools.combinations(range(w), r=2))
    
    new_edges = list()
    
    for i in xrange(len(indexes)):
        new_edges.append(" ".join(list(terms_temp[i] for i in indexes[i])))
       
    for i in xrange(0,len(new_edges)):
        from_to[new_edges[i].split()[0],new_edges[i].split()[1]] = 1

    # then iterate over the remaining terms
    for i in xrange(w, len(terms)):
        counter += 1
        # term to consider
        considered_term = terms[i]
        # all terms within sliding window
        terms_temp = terms[(i-w+1):(i+1)]
        
        # edges to try
        candidate_edges = list()
        for p in xrange(w-1):
            candidate_edges.append((terms_temp[p],considered_term))
    
        for try_edge in candidate_edges:
            
            # if not self-edge
            if (try_edge[1] != try_edge[0]):
            
                boolean1 = (try_edge[0],try_edge[1]) in from_to  
                boolean2 = (try_edge[1],try_edge[0]) in from_to
                
                # if edge has already been seen, update its weight
                if boolean1:
                    from_to[try_edge[0],try_edge[1]] += 1
                   
                elif boolean2:
                    from_to[try_edge[1],try_edge[0]] += 1
                
                # if edge has never been seen, create it and assign it a unit weight     
                else:
                    from_to[try_edge] = 1
    
        #if counter % 10000 == 0:
        #    print counter, 'terms have been processed in', time.time() - t
    
    # print 'creating graph'    
    
    # create empty graph
    g = igraph.Graph(directed=True)
    
    # add vertices
    g.add_vertices(sorted(set(terms)))
    
    # add edges, direction is preserved since the graph is directed
    g.add_edges(from_to.keys())
    
    # set edge and vertice weights
    g.es["weight"] = from_to.values() # based on co-occurence within sliding window
    g.vs["weight"] = g.strength(weights=from_to.values()) # weighted degree
    
    return(g)

def core_dec(g, weighted = True):
    # return core numbers of all nodes of the graph and main core
  
    # work on clone of g to preserve g 
    gg = copy.deepcopy(g)    
    
    # initialize dictionary that will contain the core numbers
    cores_g = dict(zip(gg.vs["name"],[0]*len(gg.vs["name"])))
    
    if weighted == True:
        # k-core decomposition for weighted graphs (generalized k-cores)
        # based on Batagelj and Zaversnik's (2002) algorithm #4
    
        # initialize min heap of degrees
        heap_g = zip(gg.vs["weight"],gg.vs["name"])
        heapq.heapify(heap_g)   

        while len(heap_g)>0:
            top = heap_g[0][1]
            # find vertice index of heap top element
            index_top = gg.vs["name"].index(top)
            # save names of the top element's neighbors
            neighbors_top = gg.vs[gg.neighbors(top)]["name"]
            # for safety reasons, remove top element from list of neighbors            
            neighbors_top = [neighbor for neighbor in neighbors_top if (neighbor!=top)]
            # set core number of heap top element as its weighted degree
            cores_g[top] = gg.vs["weight"][index_top]
            # delete top vertice
            gg.delete_vertices(index_top)
            
            if len(neighbors_top)>0:
            # iterate over neighbors of top element
                for name_n in neighbors_top:
                    index_n = gg.vs["name"].index(name_n)
                    max_n = max(cores_g[top],gg.strength(weights=gg.es["weight"])[index_n])
                    gg.vs[index_n]["weight"] = max_n
                    # update heap
                    heap_g = zip(gg.vs["weight"],gg.vs["name"])
                    heapq.heapify(heap_g)
            else:
                # update heap
                heap_g = zip(gg.vs["weight"],gg.vs["name"])
                heapq.heapify(heap_g)
                
    else:
        # k-core decomposition for unweighted graphs
        # based on Batagelj and Zaversnik's (2002) algorithm #1
         cores_g = dict(zip(gg.vs["name"],g.coreness()))
    
    # sort vertices by decreasing core number
    sorted_cores_g = sorted(cores_g.items(), key=operator.itemgetter(1), reverse=True)
    
    # find out the order of the highest/main core
    max_core_number = max([element[1] for element in sorted_cores_g])    
    
    # extract main core
    main_core = [element[0] for element in sorted_cores_g if element[1] == max_core_number]
    
    mc_subgraph = g.subgraph(main_core, implementation="create_from_scratch")

    return {'core_numbers':sorted_cores_g, 'main_core':mc_subgraph}

def accuracy_metrics(candidate, truth):
    
    # true positives ("hits") are both in candidate and in truth
    tp = len(set(candidate).intersection(truth))
    
    # false positives ("false alarms") are in candidate but not in truth
    fp = len([element for element in candidate if element not in truth])
    
    # false negatives ("misses") are in truth but not in candidate
    fn = len([element for element in truth if element not in candidate])

    # precision
    prec = round(float(tp)/(tp+fp),5)
    
    # recall
    rec = round(float(tp)/(tp+fn),5)
    
    if prec+rec != 0:
        # F1 score
        f1 = round(2 * float(prec*rec)/(prec+rec),5)
    else:
        f1=0
       
    return (prec, rec, f1)
	
def compute_node_centrality(graph):
    # degree
    degrees = graph.degree()
    degrees = [round(float(degree)/(len(graph.vs)-1),5) for degree in degrees]    
    
    # weighted degree
    w_degrees = graph.strength(weights=graph.es["weight"])
    w_degrees = [round(float(degree)/(len(graph.vs)-1),5) for degree in w_degrees]
    
    # closeness
    closeness = graph.closeness(normalized=True)
    closeness = [round(value,5) for value in closeness]
    
    # weighted closeness
    w_closeness = graph.closeness(normalized=True, weights=graph.es["weight"])
    w_closeness = [round(value,5) for value in w_closeness]

    return(zip(graph.vs["name"],degrees,w_degrees,closeness,w_closeness))