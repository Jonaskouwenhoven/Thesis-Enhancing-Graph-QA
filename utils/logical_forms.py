import re
from inspect import isclass
from rdflib import Namespace, URIRef, term
from typing import Tuple, List, Union, Optional, get_type_hints, get_origin, get_args

SOS = '('  # Start Of Sub-expression
EOS = ')'  # End of Sub-expression

# List of special tokens possible in an S-expression
SPECIAL_TOKENS = {
    SOS, EOS, 'VALUE', 'TABLE', 'MSR', 'DIM', 'WHERE', 'OR', 'TC', 'GC', '<GC>', '<TC>', " (", " )"
}


class Node:
    """
        Main abstract node type of syntax tree for s-expressions. Possible children
        of a node deriving from Node should contain type hints to explain what and how
        many children are allowed.

        * Tuple[] defines a fixed number of required children (unless overriden with Optional[])
        * List[] defines zero or more children of the type contained within
        * Union[] specifies multiple types that are all applicable as child
    """
    rdf_ns: Namespace  # optional URI namespace for (children of) node

    def __init__(self, parent=None, children=None):
        self.parent: Optional['Node'] = parent
        self.children: List['Node'] = children or []

    def add_child(self, child: 'Node'):
        self.children.append(child)

    def get_admissible_nodes(self) -> set:
        """Get all node class types that are applicable to add as children to this node"""
        type_hint = get_type_hints(self)['children']
        type_structure = get_origin(type_hint)

        if type_structure == tuple:
            # A Tuple child structure must contain each element in the given order.
            # Return therefore the first tuple entry not yet in children.
            try:
                type_hint = get_args(type_hint)[len(self.children)]
            except IndexError:
                return set()  # all children are present, none should be added

        return set(self._flatten_type_hint(type_hint))

    def _flatten_type_hint(self, type_hint) -> list:
        """
            Unpack the type annotation for an attribute to return the possible types
            applicable to add as children to this node.

            :param type_hint: type hint obtained with `get_type_hints(<class instance>)['children']`
            :returns: flattened list containing all node class types applicable following from the type hints
        """
        if isclass(type_hint):
            return [type_hint]

        flattened = []
        for item in get_args(type_hint):
            if not isclass(item):
                flattened.extend(self._flatten_type_hint(item))
            else:
                flattened.append(item)

        return flattened

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.__class__.__name__

    def __eq__(self, other: Union['Node', str]):
        """Compare string representation of nodes to determine equality"""
        return str(self) == str(other)

    def __ne__(self, other: Union['Node', str]):
        return not self == other

    def __hash__(self):
        return id(self)


class TerminalNode(Node):
    def add_child(self, _):
        raise TypeError("Can't add a child to a terminal node")

    def get_admissible_nodes(self):
        raise TypeError("Can't add admissible nodes to a terminal node")


class Code(TerminalNode):
    value: str

    def __init__(self, value: str, parent=None, children=None):
        super().__init__(parent, children)
        self.value = value

    @property
    def uri(self):
        return self.parent.rdf_ns.term(self.value)

    def __repr__(self):
        return self.value


class DimId(Code):
    # Special variant of code node to distinguish between de DIM code and value code
    rdf_ns = Namespace("https://vocabs.cbs.nl/def/dimensie/")

    @property
    def uri(self):
        return self.rdf_ns.term(self.value)


class TC(DimId):
    pass


class GC(DimId):
    pass


class OR(Node):
    children: List[Code]

    def get_admissible_nodes(self) -> set:
        admissible_nodes = super().get_admissible_nodes()
        if len(self.children) >= 2:  # Or must contain at least two options
            admissible_nodes.add(EOS)
        return admissible_nodes


class DIM(Node):
    children: Tuple[DimId, Union[OR, Code]]
    rdf_ns = Namespace("https://vocabs.cbs.nl/def/dimensie/")


class WHERE(Node):
    children: List[DIM]

    def get_admissible_nodes(self) -> set:
        admissible_nodes = super().get_admissible_nodes()
        if len(self.children) >= 1:  # Where must contain at least one dimension
            admissible_nodes.add(EOS)
        return admissible_nodes


class MSR(Node):
    children: Tuple[Code, Optional[WHERE]]
    rdf_ns = Namespace("https://vocabs.cbs.nl/def/onderwerp/")


class TABLE(Node):
    children: MSR
    table_id: str
    rdf_ns = Namespace("https://opendata.cbs.nl/#/CBS/nl/dataset/")

    def __init__(self, table_id: str, parent=None, children=None):
        super().__init__(parent, children)
        self.table_id = table_id

    @property
    def uri(self):
        return self.rdf_ns.term(self.table_id)

    def get_admissible_nodes(self) -> set:
        # Node should contain only one child
        return set() if len(self.children) > 0 else super().get_admissible_nodes()

    def __repr__(self):
        return self.table_id


class AGGREGATION(Node):
    children: TABLE

    def get_admissible_nodes(self) -> set:
        # Aggregation nodes should contain only one child
        return set() if len(self.children) > 0 else super().get_admissible_nodes()


class VALUE(AGGREGATION):
    pass


def uri_to_code(uri: Union[URIRef, term.Node]) -> str:
    return re.split(fr"{TABLE.rdf_ns}|{MSR.rdf_ns}|{DIM.rdf_ns}", str(uri))[-1]
