//! N-dimensional naive octree implementation

use std::cmp::Ordering;

pub type BTree<T> = Tree<T, dims::Dimension<1, 2>, 1, 2>;
pub type QuadTree<T> = Tree<T, dims::Dimension<2, 4>, 2, 4>;
pub type OctTree<T> = Tree<T, dims::Dimension<3, 8>, 3, 8>;

/// Represents which dimension to use. Workaround for rust compiler's not supporting const generics
/// yet. `D` is the dimensionality of the tree, `N` is the number of children per node. Therefore,
/// `N` must be `2^D`
///
/// You can of course extend tree types by implementing this trait for your own types, but don't
/// forget to appropriately set `D` and `N`. See alias definitions [`BTree`], [`QuadTree`],
/// [`OctTree`]
pub trait Dimension<const D: usize, const N: usize>: Default + Clone {
    fn children(&self) -> &[usize; N];
    fn children_mut(&mut self) -> &mut [usize; N];
}

/// Comparable coordinate item.
///
/// To implement your own comparable node type, you must implement this trait for your type.
///
/// ```
/// struct MyPoint {
///     x: i32,
///     y: i32,
///     value: String,
/// }
///
/// impl ndtree::TreeItem<2> for MyPoint {
///     fn compare(&self, other: &Self) -> [std::cmp::Ordering; 2] {
///         [self.x.cmp(&other.x), self.y.cmp(&other.y)]
///     }
/// }
///
/// let quad_tree = ndtree::QuadTree::<MyPoint>::new();
/// ```
///
/// Otherwise, the library provides default implementation for every tuple that has comparable first
/// entity or comparable items array.
///
/// ```
/// let quad_tree = ndtree::QuadTree::<[i32;2]>::new();
/// let quad_tree = ndtree::QuadTree::<([i32;2], String)>::new();
/// ```
pub trait TreeItem<const D: usize> {
    fn compare(&self, other: &Self) -> [Ordering; D];
}

/// Multi-dimensional tree implementation.
///
/// Type parameter `D` is the dimensionality of the tree, `N` is the number of children per node.
/// `N` must be equal to `2^D`.
///
/// The key of tree argument must implement [`TreeItem`].
///
/// ```
/// let quad_tree = ndtree::Tree::<([i32;2], bool), ndtree::dims::Dimension<2, 4>, 2, 4>::new();
/// ```
///
/// ---
///
/// It is advised to use the type aliases [`BTree`], [`QuadTree`], [`OctTree`] instead of manually
/// specifying the type parameters like above.
///
/// ```
/// let quad_tree = ndtree::QuadTree::<([i32;2], String)>::new();
/// let quad_tree = ndtree::QuadTree::<[i32;2]>::new();
/// ```
pub struct Tree<T, C, const D: usize, const N: usize>
where
    C: Dimension<D, N>,
{
    nodes_pool: Vec<Option<TreeNode<T, C, D, N>>>,
    unused_nodes: Vec<usize>,

    root: usize,
}

struct TreeNode<T, C, const D: usize, const N: usize>
where
    C: Dimension<D, N>,
{
    children: C,
    data: T,
}

pub mod items {
    use crate::TreeItem;

    #[cfg(not(feature = "assert-partial-ord"))]
    impl<const D: usize, C, T> TreeItem<D> for ([C; D], T)
    where
        C: Clone + Ord,
    {
        fn compare(&self, other: &Self) -> [std::cmp::Ordering; D] {
            // SAFETY: `D` is a const generic parameter, so it is always known at compile time
            std::array::from_fn(|index| unsafe {
                self.0
                    .get_unchecked(index)
                    .cmp(other.0.get_unchecked(index))
            })
        }
    }

    #[cfg(not(feature = "assert-partial-ord"))]
    impl<const D: usize, C> TreeItem<D> for [C; D]
    where
        C: Clone + Ord,
    {
        fn compare(&self, other: &Self) -> [std::cmp::Ordering; D] {
            // SAFETY: `D` is a const generic parameter, so it is always known at compile time
            std::array::from_fn(|index| unsafe {
                self.get_unchecked(index).cmp(other.get_unchecked(index))
            })
        }
    }

    #[cfg(feature = "assert-partial-ord")]
    impl<const D: usize, C, T> TreeItem<D> for ([C; D], T)
    where
        C: Clone + PartialOrd,
    {
        fn compare(&self, other: &Self) -> [std::cmp::Ordering; D] {
            // SAFETY: `D` is a const generic parameter, so it is always known at compile time
            std::array::from_fn(|index| unsafe {
                self.0
                    .get_unchecked(index)
                    .partial_cmp(other.0.get_unchecked(index))
                    .unwrap()
            })
        }
    }

    #[cfg(feature = "assert-partial-ord")]
    impl<const D: usize, C> TreeItem<D> for [C; D]
    where
        C: Clone + PartialOrd,
    {
        fn compare(&self, other: &Self) -> [std::cmp::Ordering; D] {
            // SAFETY: `D` is a const generic parameter, so it is always known at compile time
            std::array::from_fn(|index| unsafe {
                self.get_unchecked(index)
                    .partial_cmp(other.get_unchecked(index))
                    .unwrap()
            })
        }
    }
}

pub mod dims {
    /// Generic dimension implementation. All children are initialized to `usize::MAX`
    #[derive(Clone)]
    pub struct Dimension<const D: usize, const N: usize>([usize; N]);

    impl<const D: usize, const N: usize> Default for Dimension<D, N> {
        fn default() -> Self {
            Self([usize::MAX; N])
        }
    }

    impl<const D: usize, const N: usize> crate::Dimension<D, N> for Dimension<D, N> {
        fn children(&self) -> &[usize; N] {
            &self.0
        }

        fn children_mut(&mut self) -> &mut [usize; N] {
            &mut self.0
        }
    }
}

mod inner {
    use crate::{Dimension, Tree, TreeItem, TreeNode};

    impl<T, C, const D: usize, const N: usize> Default for Tree<T, C, D, N>
    where
        C: Dimension<D, N>,
        T: TreeItem<D>,
    {
        fn default() -> Self {
            Self::new()
        }
    }

    impl<T, C, const D: usize, const N: usize> Clone for Tree<T, C, D, N>
    where
        C: Dimension<D, N>,
        T: Clone,
    {
        fn clone(&self) -> Self {
            Self {
                nodes_pool: self.nodes_pool.clone(),
                unused_nodes: self.unused_nodes.clone(),
                root: self.root.clone(),
            }
        }
    }

    impl<T, C, const D: usize, const N: usize> Tree<T, C, D, N>
    where
        T: TreeItem<D>,
        C: Dimension<D, N>,
    {
        pub fn with_capacity(capacity: usize) -> Self {
            Self {
                nodes_pool: {
                    let mut value = Vec::with_capacity(capacity);
                    value.resize_with(capacity, || None);
                    value
                },
                unused_nodes: Vec::with_capacity(capacity),
                root: usize::MAX,
            }
        }

        pub fn new() -> Self {
            Self {
                nodes_pool: Vec::new(),
                unused_nodes: Vec::new(),
                root: usize::MAX,
            }
        }
    }

    impl<T, C, const D: usize, const N: usize> TreeNode<T, C, D, N>
    where
        C: Dimension<D, N>,
    {
        fn new(data: T) -> Self {
            Self {
                children: {
                    let default = C::default();
                    debug_assert!(default.children().iter().all(|&x| x == usize::MAX));
                    default
                },
                data,
            }
        }
    }

    impl<T, C, const D: usize, const N: usize> Clone for TreeNode<T, C, D, N>
    where
        C: Dimension<D, N>,
        T: Clone,
    {
        fn clone(&self) -> Self {
            Self {
                children: self.children.clone(),
                data: self.data.clone(),
            }
        }
    }
}
