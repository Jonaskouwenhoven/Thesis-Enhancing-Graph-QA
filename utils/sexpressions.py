import inspect
import networkx as nx
from collections import OrderedDict
from EoN import hierarchy_pos
from matplotlib import pyplot as plt
from rdflib import Graph, QB, SKOS, RDF, Literal
from types import NoneType
from typing import TypeVar, Set

from utils import logical_forms
from utils.logical_forms import *
from odata_graph.sparql_controller import SparqlEngine
import pandas as pd

T = TypeVar("T", bound=Node)

tabledf = pd.read_pickle("data/tabledf.pkl")
class ObservationMap:
    def __init__(self):
        self.dim_groups: Set[DimId] = set()
        self.Measure: Set[Code] = set()
        self.dims: Set[Code] = set()


class SExpression:
    def __init__(self, table_dict, query, graph: Optional[Graph] = None):
        self.root: Optional[Node] = None  # The root (i.e. aggregation function) of the syntax tree
        self._current_node: Optional[Node] = None
        self.expression: str = ''
        self.graph: Graph = graph
        self.table_dict = table_dict
        self.alpaca_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. ### Instruction: {{}} ### Input: {{}} ### Response:{'<s>'}"""
        self.query = query
        # Convenience maps for the different obs    ervation elements present in syntax tree
        self.obs_map: OrderedDict[TABLE, ObservationMap] = OrderedDict()
        self._valid_dim_groups = None
        self._valid_dim_tokens = None

    def get_admissible_tokens(self) -> List[str]:
        """
            Return the possible tokens from the current state of the expression tree

            :returns: list containing all currently possible tokens while building the S-expression sequentially
        """
        admissible_tokens = []
        new_prompt_total = None
        dict = None
        table_branch_node = False
        if self.root is None:
            # Only a root node (aggregation function) is possible starting from nothing
            # print( [c.__name__ for c in getattr(logical_forms, 'AGGREGATION').__subclasses__()])
            return [c.__name__ for c in getattr(logical_forms, 'AGGREGATION').__subclasses__()]

        if self._current_node is None:
            # Done if all brackets match
            return [] if self.expression.count(SOS) - self.expression.count(EOS) == 0 else [EOS]

        candidates = self._current_node.get_admissible_nodes()
        if len(candidates) == 0:
            return [EOS]  # no options for the current node, i.e. statement must be closed

        for T in candidates:
            if isinstance(T, str):
                if T == EOS and self.expression[-1] == SOS:
                    pass  # EOS may not come directly after an SOS
                else:
                    admissible_tokens.append(T)
                continue

            match T.__name__:
                case TABLE.__name__:
                    ids, new_prompt, dict = self.get_table_ids()
                    admissible_tokens.extend(ids)
                    new_prompt_total = new_prompt
                    table_branch_node = True
                case DimId.__name__:
                    admissible_tokens.extend(self._valid_dim_groups)
                case Code.__name__:
                    # Get all possible Measure or dims corresponding with table as admissible tokens
                    root = self._current_node.parent if isinstance(self._current_node, OR) else self._current_node
                    if isinstance(root, MSR):
                        ids, new_prompt, dict = self.get_valid_msr_tokens()
                        admissible_tokens.extend(ids - set(self._current_node.children))
                        if new_prompt:
                            new_prompt_total = new_prompt

                    if isinstance(root, DIM):
                        self._valid_dim_tokens, prompt = self.get_valid_dim_tokens(root.children[0])
                        admissible_tokens.extend(self._valid_dim_tokens)
                        if prompt:
                            new_prompt_total = prompt
                    if OR.__name__ in admissible_tokens and len(admissible_tokens) <= 2:
                        admissible_tokens.remove(OR.__name__)
                case DIM.__name__ | WHERE.__name__:
                    # Check if there are also admissible dim groups before giving the option of a DIM token
                    self._valid_dim_groups = self.get_valid_dim_groups()
                    if len(self._valid_dim_groups) > 0:
                        admissible_tokens.append(T())
                case NoneType.__name__:
                    admissible_tokens.append(EOS)
                case _:
                    admissible_tokens.append(T())
        return ([str(c) for c in admissible_tokens], new_prompt_total, dict, table_branch_node)
        ## try and make this work
        # return ([str(c) for c in admissible_tokens], new_prompt_total)

    def add_token(self, token: str) -> Union[Node, str]:
        """
            Add given token to the actual s-expression (sting) and syntax tree.

            :param token: token string to add. Should be obtained with `get_admissible_tokens()`
            :returns: the node added based on the token given
        """

        admissible_tokens = self.get_admissible_tokens()[0] if len(self.get_admissible_tokens()) > 1 else self.get_admissible_tokens() 
        
        parse_mode = inspect.stack()[1].function == self.parse.__name__
        if not parse_mode and (self.graph is None or len(self.graph) == 0):
            raise ValueError(f"Cant validate next token {token} in {self.expression}. No graph available.")
        if not parse_mode and token not in (admissible_tokens + [SOS]):
            print(f"we are trying to add {token} to the expression {self.expression} btu the admissible tokens are {admissible_tokens}")
            raise ValueError(f"Invalid next token {token} in {self.expression}.")

        match token:
            case logical_forms.SOS:  # (
                self.expression += ' ' + SOS
                return SOS
            case logical_forms.EOS:  # )
                if self._current_node is not None:
                    self._current_node = self._current_node.parent
                    if isinstance(self._current_node, WHERE):
                        self._valid_dim_tokens = None  # DIM is closed, reset tokens
                self.expression += EOS
                return EOS
            case _:
                type_ = getattr(logical_forms, token, None)
                if type_ is None:
                    # Not an operator is given, assume it is a code (str)
                    if isinstance(self._current_node, AGGREGATION):
                        # TABLE code node must follow after the aggregation root
                        node = self._add_node(TABLE, table_id=token)
                        self.obs_map[node] = ObservationMap()

                    elif isinstance(self._current_node, DIM) and len(self._current_node.children) == 0:
                        # First code node of DIM must always be a DimId
                        node = self._add_node(DimId, value=token)
                        self.obs_map[self._current_table].dim_groups.add(node)
                        if not parse_mode:
                            self._valid_dim_groups.remove(token)

                    elif isinstance(self._current_node, MSR):
                        node = self._add_node(Code, value=token)
                        self.obs_map[self._current_table].Measure.add(node)

                    elif self._current_node.__class__ in [DIM, OR]:
                        # Atm only DIMS support OR nodes. This might change in the future
                        node = self._add_node(Code, value=token)
                        self.obs_map[self._current_table].dims.add(node)
                        if not parse_mode:
                            self._valid_dim_tokens.remove(token)

                    else:
                        raise f"Encountered unknown token {type_}"
                elif type_ == TC or type_ == GC:
                    assert isinstance(self._current_node, DIM)
                    node = self._add_node(type_, value=uri_to_code(self._tc_gc_to_dim_id(type_=type_)))
                    self.obs_map[self._current_table].dim_groups.add(node)
                    if not parse_mode:
                        self._valid_dim_groups.remove(token)
                elif issubclass(type_, AGGREGATION):
                    self._current_node = self.root = node = type_()
                else:
                    # Operator is given, add to the syntax tree
                    node = self._add_node(type_)

                self.expression += ' ' if isinstance(node, TerminalNode) or not parse_mode else ''
                self.expression += SOS if not isinstance(node, TerminalNode) and not parse_mode else ''
                self.expression += token
                # print(f"We successfully added the token {token} to the expression {self.expression}")
                return node

    def _add_node(self, node: type[T], **kwargs) -> T:
        """
            Add an arbitrary node to the syntax tree following the s-expression

            :param node: class type from which a node can be instantiated
            :param **kwargs: arbitrary extra arguments that could be passed to a class instance if applicable
            :returns: node that was added to the syntax tree
        """
        operator = node(parent=self._current_node, **kwargs)
        self._current_node.add_child(operator)
        if not isinstance(operator, TerminalNode):
            self._current_node = operator

        return operator

    # def get_table_ids(self) -> Set[str]:
    #     """Get all possible table options from the graph as admissible nodes"""
    #     tables = set(self.graph.subjects(QB.measure, None))
    #     return {uri_to_code(t) for t in tables}


    def get_table_ids(self) -> Set[str]:
        table_ids = []
        table_prompt_Str = f"Based on the table id's and table descriptions, find the table that best fits the {self.query}. \n"
        table_dict = {}
        for table in self.table_dict:
            table_df_slice = tabledf[tabledf['table_id'] == table['table_id']]
            newsamen = table_df_slice['Summary'].values[0]
            # print(newsamen)
            table_ids.append(table['table_id'])
            table_prompt_Str += f"Table ID: {table['table_id']} Description {newsamen}\n"
            table_dict[table['table_id']] = table['table_desc']
        # print(new_prompt)
        prompt = self.alpaca_prompt.format("Find the most appropriate table based on the table descriptions and table ids", table_prompt_Str)
        return set(table_ids), prompt, table_dict


    def get_possibleMeasure(self, tableid: TABLE) -> Set[str]:
        """
            Return all valid Measure of table corresponding with the current expression.
            Make sure every measure can only be added once to a WHERE clause.
        """
        idstoreturn = None
        new_prompt = f"Based on the measure id's and measure descriptions, find the measure that best fits the {self.query}."
        measure_dict = {}
        for table in self.table_dict:
            if table['table_id'] == str(tableid):
                ids = list(table['table_info']['Measure'].keys())
                idstoreturn = set(ids)
                for id in ids:
                    new_prompt += f"Measure ID: {id}: Description: {table['table_info']['Measure'][id]}\n"
                    measure_dict[id] = table['table_info']['Measure'][id]
        
        prompt = self.alpaca_prompt.format("Find the most appropriate measure based on the measure descriptions and measure ids", new_prompt)
        return idstoreturn, prompt, measure_dict
    
    
    def get_valid_msr_tokens(self) -> Set[str]:
        """Return all valid Measure of table corresponding with the current expression"""
        # Measure = set(self.graph.objects(self._current_table.uri, e/QB.measure))
        Measure_2, new_prompt, dict = self.get_possibleMeasure(self._current_table)

        original_returned = {uri_to_code(msr) for msr in Measure_2} - set(str(m) for m in self.obs_map[self._current_table].Measure)
        # Exclude Measure already present in expression
        return original_returned, new_prompt, dict
    

    def get_valid_dim_groups(self) -> Set[str]:
        """
            Return all valid dimension IDs of table corresponding with the current expression.
            Make sure every ID can only be added once to a WHERE clause.
        """
        if self._valid_dim_groups is not None:
            return self._valid_dim_groups

        # Filter dims highest in hierarchy (i.e. groups) and exclude groups already present in expression
        # TODO: the SKOS.broader should probably be SKOS.narrower with the new graph
        # print(self._current_table.uri)


        groups: Set[Union[Node, str]] = (set(self.graph.objects(self._current_table.uri, QB.dimension)) -
                                         set(self.graph.subjects(SKOS.broader, None)))
        valid_dim_groups = ({uri_to_code(dim) for dim in groups} -
                            set(str(d) for d in self.obs_map[self._current_table].dim_groups))

        # Substitute TimeDimensions for TC and GeoDimensions for GC if needed
        for g in valid_dim_groups & {uri_to_code(d) for d in self.graph.subjects(RDF.type, Literal('TimeDimension'))}:
            valid_dim_groups.remove(g)
            valid_dim_groups.add('TC')

        for g in valid_dim_groups & {uri_to_code(d) for d in self.graph.subjects(RDF.type, Literal('GeoDimension'))}:
            valid_dim_groups.remove(g)
            valid_dim_groups.add('GC')

        return valid_dim_groups
    

    def get_possiblegroups(self, tableid: TABLE, dim_group: Union[DimId, term.Node, URIRef]) -> Set[str]:
        """
            Return all valid dimension IDs of table corresponding with the current expression.
            Make sure every ID can only be added once to a WHERE clause.
        """
        import random
        try:
            idstoreturn = None
            new_prompt = f"Based on the dimension id's and dimension descriptions, find the dimension that best fits the {self.query} if unclear always pick the dimension that corresponds to a Total"
            for table in self.table_dict:
                if table['table_id'] == str(tableid):
                    # print(f"WE ARE At a tbale {table}, with dimgroup {str(dim_group)}")
                    ids = list(table['table_info']['dimensionGroups'][str(dim_group)].keys())
                    ## shuffling the ids
                    # print(f"did we find any ids: {ids}")
                    for dim in table['table_info']['dimensionGroups'][str(dim_group)]:
                        new_prompt += f"Dimension ID {dim}: Description {table['table_info']['dimensionGroups'][str(dim_group)][dim]}\n"
                    # ids = random.shuffle(ides)
                    idstoreturn = set(ids)


            prompt = self.alpaca_prompt.format("You find the most appropriate dimension based on the dimension descriptions and dimension id's", new_prompt)
            return idstoreturn, prompt
        
        except:
            return set(), None



    def get_valid_dim_tokens(self, dim_group: Union[DimId, term.Node, URIRef]) -> Set[str]:
        """Return all valid  codes of dimension corresponding for a given dimension group (DimID)"""
        if self._valid_dim_tokens is not None:
            return self._valid_dim_tokens, None


        possible_geo_dims = ['GC', 'RegioS', "Regio", "WijkenEnBuurten"]
        # Filter dims lowest in hierarchy (i.e. codes) corresponding with the current dim group
        # and exclude groups already present in expression
        if isinstance(dim_group, TC):
            # print("PERIODEm DIMENSION SHOULD BE RETURNED")
            return {'<TC>'}, None
        if str(dim_group) == 'Perioden':
            return {'<TC>'}, None
        elif str(dim_group) in possible_geo_dims:
            return {'<GC>'}, None
        elif isinstance(dim_group, GC):
            return {'<GC>'}, None
        elif isinstance(dim_group, term.Node) or isinstance(dim_group, URIRef):
            uri = dim_group
        else:
            uri = dim_group.uri


        possible_groups, prompt = self.get_possiblegroups(self._current_table, dim_group)
        if len(possible_groups) == 0:
            possible_groups = (set(self.graph.objects(uri, SKOS.narrower)) -
                set(self.graph.subjects(SKOS.narrower, None)) &
                set(self.graph.objects(self._current_table.uri, QB.dimension)))
        
        return possible_groups, prompt
    


    def _tc_gc_to_dim_id(self, type_: Union[TC.__class__, GC.__class__]) -> term.Node:
        """
            Get the correct DimId (str) of a corresponding Time- or GeoDimension from a TC/GC token
            for the current active DIM node.

            :param type_: type of conversion to do (TimeDimension or GeoDimension)
        """
        # TODO: the SKOS.broader should probably be SKOS.narrower with the new graph
        groups: Set[Union[Node, str]] = (set(self.graph.objects(self._current_table.uri, QB.dimension)) -
                                         set(self.graph.subjects(SKOS.broader, None)))
        tc_gc_dims = set(self.graph.subjects(RDF.type, Literal('TimeDimension' if type_ == TC else 'GeoDimension')))

        uri = list(tc_gc_dims & groups)[0]
        return uri

    # TODO: make public or create a proper solution
    @property
    def _current_table(self) -> TABLE:
        """Return the currently active table node (for referencing the obs_map)"""
        return next(reversed(self.obs_map)) if len(self.obs_map) > 0 else None

    def print_tree(self):
        """Generate and draw a visual tree following the S-expression"""
        G = nx.Graph()

        def _traverse(node: Node = None):
            G.add_node(node)
            if node.parent is not None:
                G.add_edge(node.parent, node)

            for child in node.children:
                _traverse(child)

        _traverse(self.root)

        pos = hierarchy_pos(G, self.root)
        plt.figure(figsize=(12, 12))
        nx.draw(G, pos=pos, with_labels=True, node_size=2500)
        plt.show()

    @staticmethod
    def parse(sexp: str, graph: Optional[Graph] = None, engine: Optional[SparqlEngine] = None) -> 'SExpression':
        """
            Parse a string into a valid S-expression.
            Parser taken from https://rosettacode.org/wiki/S-expressions#Python

            :param sexp: S-expression in string format
            :param graph: optional subgraph for validating parsing of S-expression (recommended).
                          Attempts to explode the subgraph once a TABLE node is parsed if none given.
            :param engine: TODO
            :returns: parsed S-expression instance following the given input
        """
        if graph is None and engine is None:
            raise AssertionError('Either a graph of SparqlEngine must be provided when parsing the S-expression.')

        TERM_REGEX = rf'''(?mx)
            \s*(?:
                (?P<brackl>\{SOS})|
                (?P<brackr>\{EOS})|
                (?P<sq>"[^"]*")|
                (?P<s>[^(^)\s]+)
               )'''

        assert sexp[0] == '(' and sexp[-1] == ')'
        # sexp = sexp[1:-1]

        node = None
        expression = SExpression(graph)
        for termtypes in re.finditer(TERM_REGEX, sexp):
            _, value = [(t, v) for t, v in termtypes.groupdict().items() if v][0]

            if isinstance(node, DIM):  # i.e. adding dim group, translate group to TC/GC if needed
                uri = DimId(value).uri
                if uri in set(expression.graph.subjects(RDF.type, Literal('TimeDimension'))):
                    value = 'TC'

                if uri in set(expression.graph.subjects(RDF.type, Literal('GeoDimension'))):
                    value = 'GC'

            node = expression.add_token(value)
            if isinstance(node, TABLE):
                if expression.graph is None or len(expression.graph) == 0:
                    expression.graph = engine.get_table_graph(node)
                if expression.graph is None or len(expression.graph) == 0:
                    raise ValueError(f"Graph for table {node} is empty. Can't validate tokens to add during parsing.")

        return expression

    def __str__(self):
        return self.expression.strip()

    def __repr__(self):
        return self.expression.strip()
