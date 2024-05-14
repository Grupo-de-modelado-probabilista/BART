cimport cython
import numpy as np
cimport numpy as cnp

from typing import Optional


ctypedef np.float_t DTYPE_t
ctypedef np.int_t INT_t
cdef class Node:
    """Node of a binary tree.

    Attributes
    ----------
    value : float
    idx_data_points : Optional[npt.NDArray[np.int_]]
    idx_split_variable : Optional[npt.NDArray[np.int_]]
    """

    def __init__(
        self,
        value: DTYPE_t = -1.0,
        idx_data_points: Optional[np.ndarray[np.int_]] = None,
        idx_split_variable: INT_t = -1,
    ) -> None:
        self.value = value
        self.idx_data_points = idx_data_points
        self.idx_split_variable = idx_split_variable
    cdef public DTYPE_t value
    cdef public np.ndarray idx_data_points
    cdef public INT_t idx_split_variable

    def __init__(self, DTYPE_t value=0.0, INT_t idx_data_points, INT_t idx_split_variable=-1):
        self.value = value
        self.idx_data_points = idx_data_points
        self.idx_split_variable = idx_split_variable
    @cython.classmethod
    def new_leaf_node(cls, value: float, idx_data_points: Optional[np.ndarray[np.int_]]) -> "Node":
        return cls(value=value, idx_data_points=idx_data_points)

    @cython.classmethod
    def new_split_node(cls, split_value: float, idx_split_variable: int) -> "Node":
        return cls(value=split_value, idx_split_variable=idx_split_variable)

#TODO: inline
    def is_split_node(self) -> bool:
        return self.idx_split_variable != -1

    #TODO: inline
    def is_leaf_node(self) -> bool:
        return self.idx_split_variable == -1


    #TODO: inline
def get_idx_left_child(index) -> int:
    return index * 2 + 1


    #TODO: inline
def get_idx_right_child(index) -> int:
    return index * 2 + 2

    #TODO: inline

def get_depth(index: int) -> int:
    return (index + 1).bit_length() - 1


cdef class Tree:
    cdef public dict tree_structure
    cdef public cnp.ndarray[cnp.float_t, ndim=2] output
    cdef public list idx_leaf_nodes

    def __init__(
        self,
        dict tree_structure,
        cnp.ndarray[cnp.float_t, ndim=2] output,
        list idx_leaf_nodes=None,
    ):
        self.tree_structure = tree_structure
        self.idx_leaf_nodes = idx_leaf_nodes
        self.output = output

    @staticmethod
    cdef Tree new_tree(
        DTYPE_t leaf_node_value,
        INT_t[:] idx_data_points,
        INT_t num_observations,
        INT_t shape,
    ):
        cdef dict tree_structure = {0: Node.new_leaf_node(value=leaf_node_value, idx_data_points=idx_data_points)}
        cdef list idx_leaf_nodes = [0]
        cdef cnp.ndarray[cnp.float_t, ndim=2] output = np.zeros((num_observations, shape), dtype=np.float32).squeeze()
        return Tree(tree_structure, output, idx_leaf_nodes)

    def __getitem__(self, index) -> Node:
        return self.get_node(index)

    def __setitem__(self, index, node) -> None:
        self.set_node(index, node)

    def copy(self) -> "Tree":
        tree: dict = {
            k: Node(v.value, v.idx_data_points, v.idx_split_variable)
            for k, v in self.tree_structure.items()
        }
        idx_leaf_nodes = self.idx_leaf_nodes.copy() if self.idx_leaf_nodes is not None else None
        return Tree(tree_structure=tree, idx_leaf_nodes=idx_leaf_nodes, output=self.output)

    def get_node(self, INT_t index) -> Node:
        return
