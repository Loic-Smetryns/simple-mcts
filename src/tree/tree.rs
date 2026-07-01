use std::mem::swap;
use std::slice::{Iter, IterMut};
use super::arena::*;

/// A strongly typed identifier for a node within the [`Tree`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(ArenaIndex);

/// Represents a single node in the [`Tree`].
///
/// A node contains its own data, an optional reference to its parent,
/// and a fixed-size array of optional children.
/// The maximum number of children is determined by the const generic `N`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Node<T, const N: usize> {
    data: T,
    parent: Option<NodeId>,
    children: [Option<NodeId>; N],
}

impl<T, const N: usize> Node<T, N> {
    /// Creates a new `Node`.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to store in this node.
    /// * `parent` - The optional [`NodeId`] of the parent node.
    /// * `children` - An array of optional [`NodeId`]s representing the children.
    ///
    /// # Returns
    ///
    /// A newly initialized `Node` instance.
    pub fn new(data: T, parent: Option<NodeId>, children: [Option<NodeId>; N]) -> Self {
        Node { data, parent, children }
    }

    /// Retrieves the ID of the parent node, if it exists.
    ///
    /// # Returns
    ///
    /// An `Option<NodeId>` containing the parent's ID, or `None` if this is the root node.
    pub fn parent(&self) -> Option<NodeId> {
        self.parent
    }

    /// Retrieves the ID of the child at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - The slot index (0 to N-1) of the child.
    ///
    /// # Returns
    ///
    /// An `Option<NodeId>` containing the child's ID if present, or `None`.
    ///
    /// # Panics
    ///
    /// Panics if the `index` is greater than or equal to `N`.
    pub fn child(&self, index: usize) -> Option<NodeId> {
        self.children[index]
    }

    /// Retrieves a mutable reference to the child slot at the specified index.
    ///
    /// This is an internal helper method used by the [`Tree`] to modify its topology
    /// (e.g., when adding a new child).
    ///
    /// # Arguments
    ///
    /// * `index` - The slot index (0 to N-1) of the child to mutate.
    ///
    /// # Returns
    ///
    /// A mutable reference to the `Option<NodeId>` at the given index.
    ///
    /// # Panics
    ///
    /// Panics if the `index` is greater than or equal to `N`.
    fn child_mut(&mut self, index: usize) -> &mut Option<NodeId> {
        &mut self.children[index]
    }

    /// Retrieves a reference to the array of children IDs.
    ///
    /// # Returns
    ///
    /// A reference to the fixed-size array `[Option<NodeId>; N]`.
    pub fn children(&self) -> &[Option<NodeId>; N] {
        &self.children
    }

    /// Retrieves a mutable reference to the entire array of children IDs.
    ///
    /// This is an internal helper method used for bulk operations on a node's children.
    ///
    /// # Returns
    ///
    /// A mutable reference to the fixed-size array `[Option<NodeId>; N]`.
    #[allow(dead_code)]
    fn children_mut(&mut self) -> &mut [Option<NodeId>; N] {
        &mut self.children
    }

    /// Returns an iterator over the children's IDs.
    ///
    /// # Returns
    ///
    /// An `Iter` yielding references to the `Option<NodeId>` of each child.
    pub fn iter_children(&self) -> Iter<'_, Option<NodeId>> {
        self.children.iter()
    }

    /// Returns a mutable iterator over the children's IDs.
    ///
    /// This is an internal helper method used to iterate and potentially modify
    /// the children links directly.
    ///
    /// # Returns
    ///
    /// An `IterMut` yielding mutable references to the `Option<NodeId>` of each child.
    #[allow(dead_code)]
    fn iter_children_mut(&mut self) -> IterMut<'_, Option<NodeId>> {
        self.children.iter_mut()
    }

    /// Retrieves a reference to the node's internal data.
    ///
    /// # Returns
    ///
    /// A reference to the data of type `T`.
    pub fn data(&self) -> &T {
        &self.data
    }

    /// Retrieves a mutable reference to the node's internal data.
    ///
    /// # Returns
    ///
    /// A mutable reference to the data of type `T`.
    pub fn data_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

impl<T: Default, const N: usize> Default for Node<T, N> {
    fn default() -> Self {
        Node {
            data: Default::default(),
            parent: None,
            children: [ None; N]
        }
    }
}

/// Represents all possible errors that can occur during [`Tree`] operations.
///
/// This enum is used as the error type in `Result` returns across the tree API
/// to ensure safe and predictable handling of invalid operations or invalid [`NodeId`]s.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TreeError{
    /// Returned when attempting to add a child to a slot that is already occupied.
    ChildAlreadyExists,

    /// Returned when the provided `parent_id` does not point to a valid node in the arena.
    ParentDoesntExist,

    /// Returned when attempting to set a root on a tree that already has one.
    RootAlreadyExists,

    /// Returned when attempting to access or mutate a node ID that does not exist in the arena.
    NodeDoesntExist,
}

/// An arena-backed tree structure with a fixed maximum number of children per node.
///
/// The `Tree` uses an [`Arena`] for contiguous memory allocation. This approach guarantees
/// excellent cache locality (Data-Oriented Design) and totally avoids the overhead of
/// individual heap allocations (`Box` or `Rc`) per node..
///
/// # Type Parameters
///
/// * `T` - The generic data type stored inside each node.
/// * `N` - The maximum branching factor (e.g., `2` for a binary tree, `8` for an octree).
pub struct Tree<T, const N: usize> {
    arena: Arena<Node<T, N>>,
    root: Option<NodeId>,
}

impl<T, const N: usize> Tree<T, N> {
    /// Creates a new, empty `Tree`.
    ///
    /// # Returns
    ///
    /// A new `Tree<T, N>` instance with no allocated capacity and no root.
    pub fn new() -> Self {
        Tree{
            arena: Arena::new(),
            root: None,
        }
    }

    /// Creates a new, empty `Tree` with at least the specified capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The initial capacity of the underlying arena.
    ///
    /// # Returns
    ///
    /// A new `Tree<T, N>` instance pre-allocated with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Tree{
            arena: Arena::with_capacity(capacity),
            root: None,
        }
    }

    /// Returns the ID of the root node, if it exists.
    ///
    /// # Returns
    ///
    /// An `Option<NodeId>` containing the root's ID, or `None` if the tree is empty.
    pub fn root(&self) -> Option<NodeId> {
        self.root
    }

    /// Returns a mutable reference to the root node Option.
    ///
    /// # Returns
    ///
    /// A `&mut Option<NodeId>` allowing to modify the root ID directly.
    #[allow(dead_code)]
    pub fn root_mut(&mut self) -> &mut Option<NodeId> {
        &mut self.root
    }

    /// Sets the root of the tree with the given data.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to store in the new root node.
    ///
    /// # Returns
    ///
    /// The `NodeId` of the newly created root.
    ///
    /// # Errors
    ///
    /// Returns [`TreeError::RootAlreadyExists`] if the tree already has a root.
    pub fn set_root(&mut self, data: T) -> Result<NodeId, TreeError> {
        if self.root.is_some() { return Err(TreeError::RootAlreadyExists); }

        let node = Node::new(data, None, [None; N]);
        let node_id = NodeId(self.arena.alloc(node));
        self.root = Some(node_id);

        Ok(node_id)
    }

    /// Adds a new child node to an existing parent at the specified index.
    ///
    /// # Arguments
    ///
    /// * `parent_id` - The ID of the parent node.
    /// * `index` - The slot index (0 to N-1) where the child should be placed.
    /// * `data` - The data to store in the new child node.
    ///
    /// # Returns
    ///
    /// The `NodeId` of the newly created child node.
    ///
    /// # Errors
    ///
    /// Returns a [`TreeError`] if:
    /// - The parent does not exist ([`TreeError::ParentDoesntExist`])
    /// - The parent already has a child at `index` ([`TreeError::ChildAlreadyExists`])
    ///
    /// # Panics
    ///
    /// Panics if the `index` is greater than or equal to the tree's maximum capacity `N`.
    pub fn add(&mut self, parent_id: NodeId, index: usize, data: T) -> Result<NodeId, TreeError> {
        assert!(index < N, "Index out of bounds");
        let parent = self.arena.get(parent_id.0).ok_or(TreeError::ParentDoesntExist)?;

        if parent.child(index).is_some() { return Err(TreeError::ChildAlreadyExists); }

        let node = Node::new(data, Some(parent_id), [None; N]);
        let node_id = NodeId(self.arena.alloc(node));

        let parent = self.arena.get_mut(parent_id.0).unwrap();

        let child = parent.child_mut(index);
        *child = Some(node_id);
        Ok(node_id)
    }

    /// Clears the tree, removing all nodes and resetting the root.
    ///
    /// The underlying capacity of the arena is preserved, preventing OS reallocations.
    pub fn clear(&mut self) {
        self.arena.clear();
        self.root = None;
    }

    /// Retrieves a reference to the complete node with the specified ID.
    ///
    /// # Arguments
    ///
    /// * `index` - The `NodeId` of the node to retrieve.
    ///
    /// # Returns
    ///
    /// A reference to the [`Node`].
    ///
    /// # Errors
    ///
    /// Returns [`TreeError::NodeDoesntExist`] if the ID is invalid.
    pub fn get(&self, index: NodeId) -> Result<&Node<T, N>, TreeError> {
        self.arena.get(index.0).ok_or(TreeError::NodeDoesntExist)
    }

    /// Retrieves a mutable reference to the complete node with the specified ID.
    ///
    /// # Arguments
    ///
    /// * `index` - The `NodeId` of the node to retrieve.
    ///
    /// # Returns
    ///
    /// A mutable reference to the [`Node`].
    ///
    /// # Errors
    ///
    /// Returns [`TreeError::NodeDoesntExist`] if the ID is invalid.
    pub fn get_mut(&mut self, index: NodeId) -> Result<&mut Node<T, N>, TreeError> {
        self.arena.get_mut(index.0).ok_or(TreeError::NodeDoesntExist)
    }

    /// Retrieves a reference to the data of the specified node.
    ///
    /// # Arguments
    ///
    /// * `node` - The `NodeId` of the target node.
    ///
    /// # Returns
    ///
    /// A reference to the data `T`.
    ///
    /// # Errors
    ///
    /// Returns [`TreeError::NodeDoesntExist`] if the ID is invalid.
    pub fn data(&self, node: NodeId) -> Result<&T, TreeError> {
        self.arena.get(node.0).map(|node| node.data()).ok_or(TreeError::NodeDoesntExist)
    }

    /// Retrieves a mutable reference to the data of the specified node.
    ///
    /// # Arguments
    ///
    /// * `node` - The `NodeId` of the target node.
    ///
    /// # Returns
    ///
    /// A mutable reference to the data `T`.
    ///
    /// # Errors
    ///
    /// Returns [`TreeError::NodeDoesntExist`] if the ID is invalid.
    #[allow(dead_code)]
    pub fn data_mut(&mut self, node: NodeId) -> Result<&mut T, TreeError> {
        self.arena.get_mut(node.0).map(|node| node.data_mut()).ok_or(TreeError::NodeDoesntExist)
    }

    /// Retrieves the ID of the parent of the specified node.
    ///
    /// # Arguments
    ///
    /// * `node` - The `NodeId` of the target node.
    ///
    /// # Returns
    ///
    /// An `Option<NodeId>` containing the parent's ID, or `None` if it is the root.
    ///
    /// # Errors
    ///
    /// Returns [`TreeError::NodeDoesntExist`] if the requested child ID is invalid.
    #[allow(dead_code)]
    pub fn parent(&self, node: NodeId) -> Result<Option<NodeId>, TreeError> {
        self.arena.get(node.0).map(|node| node.parent()).ok_or(TreeError::NodeDoesntExist)
    }

    /// Retrieves a reference to the array of children IDs for the specified node.
    ///
    /// # Arguments
    ///
    /// * `node` - The `NodeId` of the target node.
    ///
    /// # Returns
    ///
    /// A reference to the fixed-size array `[Option<NodeId>; N]`.
    ///
    /// # Errors
    ///
    /// Returns [`TreeError::NodeDoesntExist`] if the ID is invalid.
    #[allow(dead_code)]
    pub fn children(&self, node: NodeId) -> Result<&[Option<NodeId>; N], TreeError> {
        self.arena.get(node.0).map(|node| node.children()).ok_or(TreeError::NodeDoesntExist)
    }

    /// Returns an iterator over the children IDs of the specified node.
    ///
    /// # Arguments
    ///
    /// * `node` - The `NodeId` of the target node.
    ///
    /// # Returns
    ///
    /// An `Iter` over the `Option<NodeId>`s of the children.
    ///
    /// # Errors
    ///
    /// Returns [`TreeError::NodeDoesntExist`] if the ID is invalid.
    #[allow(dead_code)]
    pub fn iter_children(&self, node: NodeId) -> Result<Iter<'_, Option<NodeId>>, TreeError> {
        self.arena.get(node.0).map(|node| node.iter_children()).ok_or(TreeError::NodeDoesntExist)
    }

    /// Retrieves the ID of a specific child for a given node.
    ///
    /// # Arguments
    ///
    /// * `node` - The `NodeId` of the parent node.
    /// * `i` - The slot index (0 to N-1) of the child to retrieve.
    ///
    /// # Returns
    ///
    /// An `Option<NodeId>` containing the child's ID if present, or `None`.
    ///
    /// # Errors
    ///
    /// Returns [`TreeError::NodeDoesntExist`] if the requested parent node ID is invalid.
    ///
    /// # Panics
    ///
    /// Panics if the child index `i` is greater than or equal to the tree's maximum branching factor `N`.
    pub fn child(&self, node: NodeId, i: usize) -> Result<Option<NodeId>, TreeError> {
        self.arena.get(node.0).map(|node| node.child(i)).ok_or(TreeError::NodeDoesntExist)
    }

    /// Changes the root of the tree to the specified node.
    ///
    /// This operation severs the link between the target node and its former parent,
    /// making it the new absolute root of the tree.
    ///
    /// # Note
    ///
    /// This method does *not* free the memory of the abandoned nodes. To reclaim
    /// memory from unreachable branches, call [`Tree::compact`] after moving the root.
    ///
    /// # Arguments
    ///
    /// * `node` - The `NodeId` that will become the new root.
    ///
    /// # Errors
    ///
    /// Returns [`TreeError::NodeDoesntExist`] if the target node ID is invalid.
    pub fn move_root_to(&mut self, node: NodeId) -> Result<(), TreeError> {
        self.get_mut(node)?.parent = None;
        self.root = Some(node);
        Ok(())
    }

    /// Returns the total number of nodes the tree can hold without reallocating.
    ///
    /// # Returns
    ///
    /// A `usize` representing the current physical capacity of the underlying arena.
    #[allow(dead_code)]
    pub fn capacity(&self) -> usize {
        self.arena.capacity()
    }

    /// Returns the total number of nodes currently stored in memory.
    ///
    /// This includes both the reachable nodes in the active tree and the unreachable
    /// nodes left behind after calling [`Tree::move_root_to`].
    /// To reclaim memory from unreachable nodes, call [`Tree::compact`].
    ///
    /// # Returns
    ///
    /// A `usize` representing the exact number of nodes allocated in the arena.
    pub fn allocated_nodes(&self) -> usize {
        self.arena.len()
    }
}

impl<T: Copy, const N: usize> Tree<T, N> {
    /// Rebuilds the tree to reclaim memory from unreachable nodes.
    ///
    /// Because the tree is backed by an append-only arena structure [`Arena`], changing the root
    /// leaves old nodes (parents and unchosen sibling branches) stranded in memory.
    /// This method resolves memory exhaustion by creating a fresh internal arena and
    /// selectively copying only the currently reachable nodes (starting from the
    /// current root) using an iterative Depth-First Search.
    ///
    /// # Performance
    ///
    /// This operation traverses the entire active subtree and allocates a new arena.
    /// While optimized to avoid recursion overhead, it is a heavy operation.
    pub fn compact(&mut self) {
        let mut tree = Self::with_capacity(self.arena.capacity());

        if let Some(root) = self.root{
            let new_root = tree.set_root(self.get(root).unwrap().data).unwrap();

            let mut stack = Vec::<(NodeId, NodeId, usize)>::with_capacity(64);
            stack.push((root, new_root, 0));

            while let Some((current_id, new_current_id, i)) = stack.last_mut() {
                if *i >= N{
                    stack.pop();
                }
                else if let Some(node) = self.child(*current_id, *i).unwrap() {
                    let child_id = node;
                    let new_child_id = tree.add(*new_current_id, *i, *self.data(child_id).unwrap()).unwrap();

                    *i += 1;
                    stack.push((child_id, new_child_id, 0));
                }
                else{
                    *i += 1;
                }
            }
        }

        swap(self, &mut tree);
    }
}

impl<T, const N: usize> std::ops::Index<NodeId> for Tree<T, N> {
    type Output = Node<T, N>;

    /// Retrieves a reference to the node at the given index.
    ///
    /// # Arguments
    ///
    /// * `index` - The `NodeId` of the target node.
    ///
    /// # Panics
    ///
    /// Panics if the `index` does not exist in the underlying arena.
    fn index(&self, index: NodeId) -> &Self::Output {
        &self.arena[index.0]
    }
}

impl<T, const N: usize> std::ops::IndexMut<NodeId> for Tree<T, N> {
    /// Retrieves a mutable reference to the node at the given index.
    ///
    /// # Arguments
    ///
    /// * `index` - The `NodeId` of the target node.
    ///
    /// # Panics
    ///
    /// Panics if the `index` does not exist in the underlying arena.
    fn index_mut(&mut self, index: NodeId) -> &mut Self::Output {
        &mut self.arena[index.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_tree() -> Tree<i32, 2> {
        Tree::new()
    }

    #[test]
    fn test_set_root() {
        let mut tree = setup_tree();
        let root_id = tree.set_root(100).expect("Should set root");

        assert_eq!(tree.root(), Some(root_id));
        assert_eq!(tree.data(root_id).unwrap(), &100);
    }

    #[test]
    fn test_set_root_twice_fails() {
        let mut tree = setup_tree();
        tree.set_root(1).unwrap();
        let err = tree.set_root(2).unwrap_err();
        assert_eq!(err, TreeError::RootAlreadyExists);
    }

    #[test]
    fn test_add_child() {
        let mut tree = setup_tree();
        let root = tree.set_root(0).unwrap();

        let child = tree.add(root, 0, 10).expect("Should add child");

        assert_eq!(tree.data(child).unwrap(), &10);
        assert_eq!(tree.parent(child).unwrap(), Some(root));
        assert_eq!(tree.children(root).unwrap()[0], Some(child));
    }

    #[should_panic(expected = "Index out of bounds")]
    #[test]
    fn test_add_child_out_of_range() {
        let mut tree = setup_tree(); // N=2
        let root = tree.set_root(0).unwrap();

        let err = tree.add(root, 2, 10).unwrap_err();
    }

    #[test]
    fn test_add_child_already_exists() {
        let mut tree = setup_tree();
        let root = tree.set_root(0).unwrap();
        tree.add(root, 0, 10).unwrap();

        let err = tree.add(root, 0, 20).unwrap_err();
        assert_eq!(err, TreeError::ChildAlreadyExists);
    }

    #[test]
    fn test_add_to_non_existent_parent() {
        let mut tree = setup_tree();
        let fake_id = NodeId(ArenaIndex::from_usize(1));

        let err = tree.add(fake_id, 0, 10).unwrap_err();
        assert_eq!(err, TreeError::ParentDoesntExist);
    }

    #[test]
    fn test_data_mut() {
        let mut tree = setup_tree();
        let root = tree.set_root(10).unwrap();

        *tree.data_mut(root).unwrap() = 20;
        assert_eq!(tree.data(root).unwrap(), &20);
    }

    #[test]
    fn test_clear() {
        let mut tree = setup_tree();
        tree.set_root(10).unwrap();
        tree.clear();

        assert!(tree.root().is_none());

        let fake_id = NodeId(ArenaIndex::from_usize(1));
        assert_eq!(tree.get(fake_id).unwrap_err(), TreeError::NodeDoesntExist);
    }

    #[test]
    fn test_iter_children() {
        let mut tree = setup_tree();
        let root = tree.set_root(0).unwrap();
        tree.add(root, 0, 1).unwrap();

        let children: Vec<_> = tree.iter_children(root).unwrap().collect();
        assert_eq!(children.len(), 2);
        assert!(children[0].is_some());
        assert!(children[1].is_none());
    }

    #[test]
    fn test_index_traits() {
        let mut tree = setup_tree();
        let root = tree.set_root(10).unwrap();

        assert_eq!(tree[root].data(), &10);

        tree[root].data_mut();
    }

    #[test]
    fn test_capacity_and_allocated_nodes() {
        let mut tree = Tree::<i32, 2>::with_capacity(10);

        assert!(tree.capacity() >= 10);
        assert_eq!(tree.allocated_nodes(), 0);

        let root = tree.set_root(42).unwrap();
        assert_eq!(tree.allocated_nodes(), 1);

        tree.add(root, 0, 10).unwrap();
        assert_eq!(tree.allocated_nodes(), 2);
    }

    #[test]
    fn test_child_favorable_and_unfavorable() {
        let mut tree = Tree::<i32, 2>::new();
        let root = tree.set_root(1).unwrap();
        let child_0 = tree.add(root, 0, 2).unwrap();

        assert_eq!(tree.child(root, 0), Ok(Some(child_0)));
        assert_eq!(tree.child(root, 1), Ok(None));

        let old_root = root;
        tree.clear();
        assert_eq!(tree.child(old_root, 0), Err(TreeError::NodeDoesntExist));
    }

    #[test]
    #[should_panic]
    fn test_child_out_of_bounds_panics() {
        let mut tree = Tree::<i32, 2>::new();
        let root = tree.set_root(1).unwrap();

        let _ = tree.child(root, 2);
    }

    #[test]
    fn test_move_root_to() {
        let mut tree = Tree::<i32, 2>::new();
        let root = tree.set_root(1).unwrap();
        let child = tree.add(root, 0, 2).unwrap();

        assert!(tree.move_root_to(child).is_ok());
        assert_eq!(tree.root(), Some(child));

        assert_eq!(tree.parent(child), Ok(None));
        tree.clear();
        assert_eq!(tree.move_root_to(child), Err(TreeError::NodeDoesntExist));
    }

    #[test]
    fn test_compact_empty_or_single_node() {
        let mut tree = Tree::<i32, 2>::new();

        tree.compact();
        assert_eq!(tree.allocated_nodes(), 0);

        tree.set_root(42).unwrap();
        tree.compact();
        assert_eq!(tree.allocated_nodes(), 1);

        let root = tree.root().unwrap();
        assert_eq!(*tree.data(root).unwrap(), 42);
    }

    #[test]
    fn test_compact_complex_scenario() {
        let mut tree = Tree::<i32, 2>::new();

        //      Root (0)
        //      /     \
        //    A(10)   B(20)
        //    /   \
        // C(30) D(40)

        let root = tree.set_root(0).unwrap();
        let a = tree.add(root, 0, 10).unwrap();
        let _b = tree.add(root, 1, 20).unwrap();
        let _c = tree.add(a, 0, 30).unwrap();
        let _d = tree.add(a, 1, 40).unwrap();

        assert_eq!(tree.allocated_nodes(), 5);

        tree.move_root_to(a).unwrap();

        assert_eq!(tree.allocated_nodes(), 5);

        tree.compact();

        assert_eq!(tree.allocated_nodes(), 3);

        let new_root = tree.root().unwrap();
        assert_eq!(*tree.data(new_root).unwrap(), 10);

        let new_c = tree.child(new_root, 0).unwrap().unwrap();
        assert_eq!(*tree.data(new_c).unwrap(), 30);

        let new_d = tree.child(new_root, 1).unwrap().unwrap();
        assert_eq!(*tree.data(new_d).unwrap(), 40);
    }
}