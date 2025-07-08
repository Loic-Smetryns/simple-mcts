//! Tree data structure implementation for MCTS

use std::{cell::RefCell, rc::{Rc, Weak}};

/// Strong reference to a tree node
pub type NodeRef<T, const N: usize> = Rc<RefCell<Node<T, N>>>;
/// Weak reference to a tree node (to break reference cycles)
pub type WeakNodeRef<T, const N: usize> = Weak<RefCell<Node<T, N>>>;

/// A node in the tree structure
///
/// # Type Parameters
/// - `T`: The data type stored in the node
/// - `N`: The number of child slots
pub struct Node<T, const N: usize>{
    parent: Option<WeakNodeRef<T, N>>,
    children: [Option<NodeRef<T, N>>; N],
    data: T
}

#[allow(dead_code)]
impl<T, const N: usize> Node<T, N>{
    /// Creates a new node with given parent and data
    ///
    /// # Parameters
    /// - `parent`: The parent node (None for root)
    /// - `data`: The data to store in this node
    #[inline]
    pub fn new(parent: Option<WeakNodeRef<T, N>>, data: T) -> Self{
        Node { parent: parent, children: std::array::from_fn(|_| None), data: data }
    }

    /// Creates a new root node with given data
    ///
    /// # Parameters
    /// - `data`: The data to store in the root node
    #[inline]
    pub fn new_root(data: T) -> NodeRef<T, N>{
        Rc::new(RefCell::new(Node{ parent: None, children: std::array::from_fn(|_| None), data: data }))
    }

    /// Checks if this node is the root (has no parent)
    #[inline]
    pub fn is_root(&self) -> bool{
        self.parent.is_none()
    }

    /// Gets the parent node if it exists
    #[inline]
    pub fn get_parent(&self) -> Option<NodeRef<T, N>>{
        self.parent.as_ref().and_then(Weak::upgrade)
    }

    /// Transform node to root
    #[inline]
    pub fn detach(&mut self){
        self.parent = None;
    }

    /// Gets a child node at the specified index
    ///
    /// # Parameters
    /// - `i`: The child index (must be < N)
    #[inline]
    pub fn get_child(&self, i: usize) -> Option<NodeRef<T, N>>{
        if i >= N{
            None
        }
        else{
            self.children[i].as_ref().and_then(|x| Some(x.clone()))
        }
    }

    /// Adds a new child node at the specified index
    ///
    /// # Parameters
    /// - `node`: The parent node
    /// - `i`: The index to add the child at
    /// - `data`: The data for the new child
    ///
    /// # Returns
    /// Reference to the newly created child node
    #[inline]
    pub fn add_child(node: &NodeRef<T, N>, i: usize, data: T) -> NodeRef<T, N>{
        let ref_node = Rc::new(
            RefCell::new(Node::<T, N>::new(Some(Rc::downgrade(node)), data))
        );

        node.borrow_mut().children[i] = Some(Rc::clone(&ref_node));
        ref_node
    }

    /// Removes the child node at the specified index
    ///
    /// # Parameters
    /// - `node`: The parent node
    /// - `i`: The index of the child to remove
    #[inline]
    pub fn remove_child(node: &NodeRef<T, N>, i: usize){
        node.borrow_mut().children[i] = None;
    }

    /// Gets a reference to the node's data
    #[inline]
    pub fn get(&self) -> &T{
        &self.data
    }

    /// Gets a mutable reference to the node's data
    #[inline]
    pub fn get_mut(&mut self) -> &mut T{
        &mut self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_root_1() {
        let root = Node::<u32, 4>::new_root(5);
        let node = & *root.borrow();

        assert_eq!(*node.get(), 5);

        assert!(node.get_parent().is_none());
        assert!(node.is_root());

        assert!(node.get_child(0).is_none());
        assert!(node.get_child(1).is_none());
        assert!(node.get_child(2).is_none());
        assert!(node.get_child(3).is_none());
    }

    #[test]
    fn test_root_2() {
        let root = Node::<u32, 3>::new_root(2);
        let node = & *root.borrow();

        assert_eq!(*node.get(), 2);

        assert!(node.get_parent().is_none());
        assert!(node.is_root());

        assert!(node.get_child(0).is_none());
        assert!(node.get_child(1).is_none());
        assert!(node.get_child(2).is_none());
    }

    #[test]
    fn test_add_node_1() {
        let root = Node::<u32, 4>::new_root(6);

        Node::add_child(&root, 0, 1);
        Node::add_child(&root, 1, 2);
        Node::add_child(&root, 3, 8);

        let root_node = &*root.borrow();

        assert_eq!(*root_node.get(), 6);
        assert!(root_node.get_child(0).is_some());
        assert!(root_node.get_child(1).is_some());
        assert!(root_node.get_child(2).is_none());
        assert!(root_node.get_child(3).is_some());

        let ptr_node_0 = root_node.get_child(0).unwrap();
        let ptr_node_1 = root_node.get_child(1).unwrap();
        let ptr_node_3 = root_node.get_child(3).unwrap();

        let node_0 = &*ptr_node_0.borrow();
        let node_1 = &*ptr_node_1.borrow();
        let node_3 = &*ptr_node_3.borrow();

        assert_eq!(*node_0.get(), 1);
        assert_eq!(*node_1.get(), 2);
        assert_eq!(*node_3.get(), 8);
    }

    #[test]
    fn test_remove_node_1() {
        let root = Node::<u32, 4>::new_root(6);

        Node::add_child(&root, 0, 1);
        Node::add_child(&root, 1, 2);
        Node::add_child(&root, 3, 8);

        Node::remove_child(&root, 3);

        let root_node = &*root.borrow();

        assert_eq!(*root_node.get(), 6);
        assert!(root_node.get_child(0).is_some());
        assert!(root_node.get_child(1).is_some());
        assert!(root_node.get_child(2).is_none());
        assert!(root_node.get_child(3).is_none());
    }
}
