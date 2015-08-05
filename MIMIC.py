__author__ = 'Henry'


import sys
import copy
import random
import graphviz
import operator
import itertools
import numpy as np
import networkx as nx
from collections import Counter
from string import ascii_lowercase
from bayesian.bbn import build_bbn
from matplotlib import pyplot as plt


def declare_function(header, body):
    """
    Declares the header function using the body code.

    :type header: str
    :param header: The name of the function.

    :type body: str
    :param body: The code of the function.

    :rtype: function
    :return: A pointer to the function.
    """
    exec body
    return locals()['f_' + str(header)]


class Color(object):
    hexadecimal_color_length = 6

    def __init__(self):
        pass

    @staticmethod
    def randomize_colors(n_colors, colors=[], p=[]):
        return [Color.randomize_color(colors, p) for x in range(n_colors)]

    @staticmethod
    def randomize_color(colors=[], p=[]):
        values = '0123456789ABCDEF'

        if not colors:  # invalid color array; may choose any color
            color = np.random.choice(list(values), size=Color.hexadecimal_color_length, replace=True)
            return '#' + ''.join(color)
        else:
            if not p:
                color = np.random.choice(colors, size=1)
            else:
                color = np.random.choice(a=colors, size=1, replace=True, p=p)
            return color


class MIMIC(object):
    population_size = 0
    model_graph = None
    palette = []
    population = []
    bbn = None

    dependencies = []
    _bbn_functions = []
    node_names = []

    def __init__(self, population_size, model_graph, colors):
        """
        Initializes the MIMIC algorithm;

        :type population_size: int
        :param population_size: The size of the population.

        :type model_graph: ModelGraph
        :param model_graph: The ModelGraph which the population will be copied of.

        :type colors: list
        :param colors: a list of valid colors to choose from.
        """
        self.population_size = population_size
        self.model_graph = model_graph
        self.palette = colors

        self.node_names = self.model_graph.names

        self.population = np.array(
            map(
                lambda x: copy.deepcopy(self.model_graph),
                xrange(self.population_size)
            )
        )

        self.dependencies = dict(list(itertools.izip_longest(self.node_names[1:], self.node_names, fillvalue=None)))
        self.__build_bbn__()

    def solve(self, max_iter=100):
        """
        Solves the k-max coloring problem. Exports the best individual
        along with the bayesian belief network to a pdf file.

        :rtype max_iter: int
        :param max_iter: Max number of iterations.
        """
        i = 1
        while i <= max_iter:
            sys.stdout.write('\r' + 'Iterations: ' + "%03d" % (i,))
            i += 1

            self.__sample__()
            fitness = map(lambda x: x.fitness, self.population)
            median = np.median(fitness)
            fittest = list(itertools.ifilter(lambda x: x.fitness >= median, self.population))

            # former depends on latter in the tuple
            self.dependencies = self.__search_dependencies__(fittest)
            self.__build_bbn__(depends_on=self.dependencies, fittest=fittest)

            if self.__has_converged__():
                break

        print '\n'
        self.__export__(i, screen=True, pdf=False, file=True)

    def __build_bbn__(self, depends_on=None, fittest=[], one_optimal=False):
        """
        Build a bayesian belief network and sets it to self._bbn_functions.

        :type depends_on: list
        :param depends_on: the dependency chain between attributes.

        :type fittest: list
        :param fittest: A list of the fittest individuals (denoted as ModelGraph's) for this generation.

        :type one_optimal: bool
        :param one_optimal: Wheter to solve this problem with only one optimal individual or not.
        """
        _nodes = self.node_names

        functions = []
        if not depends_on:
            for i, node in enumerate(_nodes[:-1]):
                _str = "def f_" + node + "(" + node + ", " + _nodes[i + 1] + "):\n    return " + str(1. / float(len(self.palette)))
                func = declare_function(node, _str)
                functions += [(node, func)]

            _str = "def f_" + _nodes[-1] + "(" + _nodes[-1] + "):\n"
            if one_optimal:
                _str += "    " + "return 1. if " + _nodes[-1] + " == '" + self.palette[0] + "' else 0."
            else:
                _str = "def f_" + _nodes[-1] + "(" + _nodes[-1] + "):\n    return " + str(1. / float(len(self.palette)))

            func = declare_function(_nodes[-1], _str)
            functions += [(_nodes[-1], func)]

        else:
            # delete former functions to avoid any overlap in the next executions
            if len(self._bbn_functions) > 0:
                for func in self._bbn_functions:
                    del func

            # fittest_dict is a collection of the fittest individuals, grouped by their attributes in a dictionary
            fittest_dict = dict(
                zip(
                    _nodes,
                    np.array(map(lambda x: x.colors, fittest)).T.tolist()
                )
            )

            functions = []
            drawn = None
            set_names = set(depends_on.values())

            while len(set_names) > 0:
                _str = self.__build_function__(depends_on[drawn], len(fittest), fittest_dict, drawn)
                func = declare_function(depends_on[drawn], _str)
                functions += [(depends_on[drawn], func)]
                drawn = depends_on[drawn]
                set_names -= {drawn}

        self._bbn_functions = dict(functions)

    def __build_function__(self, drawn, fittest_count, fittest_dict, dependency=None):
        """
        Builds a function, which later will be used to infer values for attributes.

        :type drawn: str
        :param drawn: The attribute which function will be defined.

        :type fittest_count: int
        :param fittest_count: The number of fittest individuals for this generation.

        :type fittest_dict: dict
        :param fittest_dict: The fittest individuals grouped by attributes.

        :type dependency: str
        :param dependency: the attribute which the drawn attribute depends on, or None otherwise.

        :rtype: str
        :return: The function body.
        """

        carriage = "    "

        _str = "def f_" + drawn + "(" + drawn + (', ' + dependency if dependency else '') + "):\n"

        if not dependency:
            count_drawn = Counter(fittest_dict[drawn])

            for i, color in enumerate(self.palette):
                _str += carriage + "if " + drawn + " == '" + color + "':\n" + (carriage * 2) + "return " + \
                    str(float(count_drawn.get(color) if color in count_drawn else 0.) / float(fittest_count)) + "\n"

        else:
            iterated = itertools.product(self.palette, self.palette)
            count_dependency = Counter(zip(fittest_dict[drawn], fittest_dict[dependency]))

            for ct in iterated:
                denominator = np.sum(map(lambda x: count_dependency.get(x) if x in count_dependency else 0., itertools.product(self.palette, [ct[1]])))

                value = 0. if denominator == 0. else float(count_dependency.get((ct[0], ct[1])) if (ct[0], ct[1]) in count_dependency else 0.) / denominator

                _str += carriage + "if " + drawn + " == '" + ct[0] + "' and " + dependency + " == '" + ct[1] + "':\n"
                _str += (carriage * 2) + "return " + str(value) + "\n"

        return _str

    def __sample__(self):
        """
        Assigns colors to the population of graphs.
        """

        drawn = self.dependencies[None]
        values = map(lambda x: self._bbn_functions[drawn](x), self.palette)
        sample = np.random.choice(self.palette, size=self.population_size, replace=True, p=values).reshape((1, self.population_size))

        remaining_nodes = set(self.node_names) - {drawn}

        while len(remaining_nodes) > 0:
            drawn = self.dependencies[drawn]

            # probs are reversed, with the free parameter first and the conditioned later.
            values = np.array(
                map(
                    lambda x: self._bbn_functions[drawn](x[1], x[0]),
                    itertools.product(sample[-1, :], self.palette)
                )
            ).reshape((self.population_size, len(self.palette)))

            _samples = map(lambda x: np.random.choice(self.palette, p=x), values)
            sample = np.vstack((sample, _samples))

            remaining_nodes -= {drawn}

        sample = sample.T

        for graph, colors in itertools.izip(self.population, sample):
            graph.colors = colors

    def __search_dependencies__(self, fittest):
        """
        Infers the dependencies between attributes.

        :type fittest: list
        :param fittest: A list of the fittest individuals for this generation.

        :rtype: dict
        :return: A list of tuples containing each pair of dependencies.
        """
        min_key = None
        remaining = set(self.node_names)
        dict_dependencies = dict()

        while len(remaining) > 0:
            entropies = dict(map(lambda x: self.__entropy__(x, fittest, min_key), remaining))
            new_min_key = entropies[min(entropies)]
            dict_dependencies[min_key] = new_min_key
            remaining -= {new_min_key}
            min_key = new_min_key

        return dict_dependencies

    @staticmethod
    def __marginalize__(dependencies, p_value):
        """
        Marginalizes the distribution given the p_value: P(p_value | any_q_value). Returns the frequency.
        :param dependencies: All possible configurations of values and the global frequencies.
        :param p_value: The value of the dependent variable.
        :return:
        """
        keys = []
        for item in dependencies.keys():
            if item[0] == p_value:
                keys.append(item)

        total = float(reduce(operator.add, dependencies.values()))
        _sum = np.sum(map(lambda x: dependencies[x], keys)) / total
        return _sum

    def __entropy__(self, name, sample, dependency=None):
        """
        Calculates the entropy of a given attribute.
        :type name: str
        :param name: The attribute name.

        :type dependency: str
        :param dependency: optional -- If the attribute is dependent on other attribute. In this case,
            it shall be provided the name of the free attribute.

        :rtype: tuple
        :return: A tuple containing the name of the attribute alongside its entropy.
        """
        if not dependency:
            return -1. * np.sum(
                map(
                    lambda x: (float(x) / len(sample)) * np.log2(float(x) / len(sample)),
                    Counter(map(lambda y: y.nodes[name].color, sample)).values()
                )
            ), name
        else:
            conditionals = Counter(map(lambda x: (x.nodes[name].color, x.nodes[dependency].color), sample))

            entropy = 0.
            for value in set(
                    map(lambda x: x[0], conditionals.keys())):  # iterates over the values of the conditioned attribute
                marginal = self.__marginalize__(conditionals, value)
                entropy += marginal * np.log2(marginal)

            return -1. * entropy, name

    def __has_converged__(self):
        fitness = map(
            lambda x: x.fitness,
            self.population
        )
        median = np.median(fitness)
        result = np.all(fitness == median)
        return result

    def __best_individual__(self):
        fitness = map(lambda x: x.fitness, self.population)
        return self.population[np.argmax(fitness)]

    def __export__(self, iterations, screen=True, file=True, pdf=True):
        _str = 'Finished inference in ' + str(iterations) + ' iterations.\n'
        _str += 'Evaluations: ' + str(iterations * self.population_size) + '\n'
        _str += 'Population size: ' + str(self.population_size) + '\n'
        _str += 'Nodes: ' + str(self.model_graph.count_nodes) + '\n'
        _str += 'Colors: ' + str(len(self.palette)) + '\n'
        _str += 'Best individual fitness: ' + str(round(self.__best_individual__().fitness, 2)) + "\n"
        _str += 'Marginalization matrix:\n'

        some_bbn = build_bbn(
            *self._bbn_functions.values(),  # unzips list
            domains=dict([(x, self.palette) for x in self.node_names])
        )

        print _str
        some_bbn.q()

        _dict = copy.deepcopy(self.dependencies)
        del _dict[None]

        if pdf:
            bbn = graphviz.Digraph()
            [bbn.node(x) for x in self.node_names]

            bbn.edges(_dict.items())
            bbn.render('bbn.gv')

            self.__best_individual__().export('optimal')

        if file:
            query = some_bbn.query().items()
            query = '\n'.join([str(x) for x in query])

            with open('output.txt', 'w') as file:
                file.write(_str)
                file.write(query)

        if screen:
            def plot_bbn():
                plt.figure(1)
                G = nx.DiGraph()
                G.add_edges_from(_dict.items())
                layout = nx.shell_layout(G)
                nx.draw_networkx(G, pos=layout, cmap=plt.get_cmap('jet'), node_color=list(itertools.repeat('cyan', len(self.node_names))))

            def plot_optimal():
                plt.figure(2)
                individual = self.__best_individual__()
                G = nx.Graph()
                G.add_nodes_from(individual.names)
                some_edges = [tuple(list(x)) for x in individual.edges]
                G.add_edges_from(some_edges)
                layout = nx.shell_layout(G)
                nx.draw_networkx(G, pos=layout, cmap=plt.get_cmap('jet'), node_color=individual.colors)

            plot_bbn()
            plot_optimal()
            plt.show()


class Node(object):
    name = ''
    color = None
    neighbours = []

    def __init__(self, name, **kwargs):
        self.name = name
        self.color = "#ffffff" if 'color' not in kwargs else kwargs['color']
        self.neighbours = [] if 'neighbours' not in kwargs else kwargs['neighbours']

    def __deepcopy__(self, memo):
        _dict = {
            'name': copy.deepcopy(self.name),
            'color': copy.deepcopy(self.color),  # shallow copy
            'neighbours': copy.copy(self.neighbours)  # shallow copy
        }

        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in _dict.items():
            setattr(result, k, v)
        return result

    def __copy__(self):
        _dict = {
            'color': copy.copy(self.color),  # shallow copy
            'neighbours': copy.copy(self.neighbours)  # shallow copy
        }
        name = copy.copy(self.name)
        return Node(name, **_dict)

    def link(self, node):
        if node not in self.neighbours and self not in node.neighbours and node != self:
            self.neighbours.append(node)
            node.neighbours.append(self)

    def clear_linkage(self):
        self.neighbours = []
        return self

    def str_edges(self):
        return [self.name + x.name for x in self.neighbours]


class ModelGraph(object):
    """
    Attributes:

    :type nodes: list
    :param nodes: a list of the nodes.

    :type edges: list
    :param edges: a list of the edges between the nodes.
    """
    nodes = dict()
    edges = []
    _fitness = 0.
    names = []

    def __init__(self, *args, **kwargs):
        """
        Constructor for MGraph class.

        :type node: Node
        :param node: optional -- The nodes of this MGraph.

        :type neighborhood: list
        :param neighborhood: optional -- To pass the nodes of this MGraph as a list.

        :type visual: graphviz.graph
        :param visual: optional - visual engine of the graph

        :type edges: list
        :param edges: optional - edges of the graph
        """

        self.edges = kwargs['edges'] if 'edges' in kwargs else list()

        map(self.__extend_node_collection__, args)

        if 'neighborhood' in kwargs:
            map(self.__extend_node_collection__, kwargs['neighborhood'])

        self.edges = list(set(self.edges))
        self._fitness = self.fitness

        self.names = map(lambda x: x.name, self.nodes.values())

    def __copy__(self):
        _dict = dict()
        _dict['edges'] = copy.copy(self.edges)
        _dict['neighborhood'] = copy.copy(self.nodes)

        # pass dictionary as kwargs:
        # http://stackoverflow.com/questions/5710391/converting-python-dict-to-kwargs
        return ModelGraph(**_dict)

    def __deepcopy__(self, memo):
        _dict = dict()
        _dict['edges'] = copy.deepcopy(self.edges)

        nodes = [Node(copy.deepcopy(x.name), color=copy.deepcopy(x.color)) for x in self.nodes.values()]
        _dict['names'] = map(lambda x: copy.deepcopy(x.name), nodes)

        neighborhood = dict(zip(_dict['names'], nodes))

        for edge in _dict['edges']:
            links = list(edge)
            neighborhood[links[0]].link(neighborhood[links[1]])

        _dict['nodes'] = neighborhood

        # pass dictionary as kwargs:
        # http://stackoverflow.com/questions/5710391/converting-python-dict-to-kwargs
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in _dict.items():
            setattr(result, k, v)
        return result

    def __extend_node_collection__(self, node, make_edges=True):
        self.nodes[node.name] = node
        if make_edges:
            self.__make__edges__(node)

    def __make__edges__(self, node):
        _sorted = [''.join(sorted(x)) for x in node.str_edges()]
        self.edges.extend(_sorted)

    def randomize_edges(self, chain=False):
        """
        Randomly creates edges between nodes.

        :type chain: bool
        :param chain: Whether to chain attributes or not.
        """
        selectable = set(self.nodes)
        nodes = self.nodes.values()

        for i, node in enumerate(self.nodes):
            selectable -= {node}  # prevents a node from linking with itself
            to_connect = None
            if not chain:  # random linkage
                to_connect = np.random.choice(selectable)
            elif (i + 1) < len(self.nodes):  # chain linkage
                to_connect = nodes[i + 1]

            if to_connect is not None:
                nodes[i].link(to_connect)
                self.__make__edges__(nodes[i])

        self.edges = list(set(self.edges))

    @property
    def fitness(self):
        """
        The fitness of this graph - the number of edges that it's nodes
        doesn't have the same color, normalized to [0,1].

        :rtype: float
        :return: Ranges from 0 (worst fitness) to 1 (best fitness).
        """
        return self._fitness

    @property
    def count_nodes(self):
        """
        :rtype: int
        :return: Number of nodes in this graph.
        """
        return len(self.nodes.keys())

    @property
    def colors(self):
        return map(lambda x: x.color, self.nodes.values())

    @colors.setter
    def colors(self, value):
        if isinstance(value, list) or isinstance(value, np.ndarray):
            for i, node in enumerate(self.nodes.itervalues()):
                node.color = value[i]
        elif isinstance(value, dict):
            for key in value.iterkeys():
                self.nodes[key].color = value[key]

        diff_colors = reduce(
            operator.add,
            map(
                lambda y: len(set(
                    map(
                        lambda x: self.nodes[x].color,
                        list(y)
                    )
                )) > 1,
                self.edges
            )
        )
        self._fitness = float(diff_colors) / len(self.edges)

    def export(self, filename):
        visual = graphviz.Graph()

        for node in self.nodes.values():
            visual.node(node.name, node.name, style='filled', fillcolor=node.color)

        visual.edges(self.edges)
        visual.render(filename + '.gv')


def main():
    _nodes = []
    count_nodes = 26  # max number of nodes = letters in the alphabet
    population_size = 200  # size of the population
    seed = None  # use None for random or any integer for predetermined randomization
    max_iter = 100000  # max iterations to search for optima
    n_colors = 2  # number of colors to use

    random.seed(seed)
    np.random.seed(seed)

    for char in list(ascii_lowercase)[:count_nodes]:
        _nodes.append(Node(char))

    my_graph = ModelGraph(neighborhood=_nodes)
    my_graph.randomize_edges(chain=True)

    colors = Color.randomize_colors(n_colors=n_colors)

    mr_mime = MIMIC(population_size, my_graph, colors)
    mr_mime.solve(max_iter=max_iter)


main()
