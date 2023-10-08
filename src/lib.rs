//! N-dimensional naive octree implementation. This only provides the basic functionality of
//! multi-dimensional tree, and is not very well optimized.

// - TODO: Tree re-balancing

use std::cmp::Ordering;
pub type BTreeMap<K, V> = TreeMap<K, V, dims::Dimension<1, 2>, 1, 2>;
pub type QuadTreeMap<K, V> = TreeMap<K, V, dims::Dimension<2, 4>, 2, 4>;
pub type OctTreeMap<K, V> = TreeMap<K, V, dims::Dimension<3, 8>, 3, 8>;

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
/// ```
/// struct MyPoint {
///     x: i32,
///     y: i32,
/// }
///
/// impl ndtree::TreeKey<2> for MyPoint {
///     fn compare(&self, other: &Self) -> [std::cmp::Ordering; 2] {
///         [self.x.cmp(&other.x), self.y.cmp(&other.y)]
///     }
/// }
///
/// let quad_tree = ndtree::QuadTreeMap::<MyPoint, String>::new();
/// ```
///
/// Otherwise, the library provides default implementation for every tuple that has comparable first
/// entity or comparable items array.
///
/// ```
/// let quad_tree = ndtree::QuadTreeMap::<[i32;2], String>::new();
/// let quad_tree = ndtree::QuadTreeMap::<(i32, f32), String>::new();
/// ```
pub trait TreeKey<const D: usize> {
    fn compare(&self, other: &Self) -> [Ordering; D];
}

/// Multi-dimensional tree implementation.
///
/// Type parameter `D` is the dimensionality of the tree, `N` is the number of children per node.
/// `N` must be equal to `2^D`.
///
/// The key of tree argument must implement [`TreeKey`].
///
/// ```
/// let quad_tree = ndtree::TreeMap::<[i32;2], Vec<String>, ndtree::dims::Dimension<2, 4>, 2, 4>::new();
/// ```
///
/// ---
///
/// It is advised to use the type aliases [`BTree`], [`QuadTree`], [`OctTree`] instead of manually
/// specifying the type parameters like above.
///
/// ```
/// let quad_tree = ndtree::QuadTreeMap::<(i32, f64), String>::new();
/// let quad_tree = ndtree::QuadTreeMap::<[i32;2], bool>::new();
/// ```
///
/// ---
///
/// As the map manages internal nodes as contiguous pool of array elements, various node index based
/// operations are provided for advanced control of the tree.
pub struct TreeMap<K, V, C, const D: usize, const N: usize>
where
    C: Dimension<D, N>,
{
    nodes_pool: Vec<Option<TreeNode<K, V, C, D, N>>>,
    unused_nodes: Vec<usize>,

    root: usize,
}

struct TreeNode<K, V, C, const D: usize, const N: usize>
where
    C: Dimension<D, N>,
{
    children: C,
    key: K,
    value: V,
}

pub mod key_types {
    use crate::TreeKey;

    #[cfg(not(feature = "assert-partial-ord"))]
    impl<const D: usize, C> TreeKey<D> for [C; D]
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
    impl<const D: usize, C> TreeKey<D> for [C; D]
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

    /* ---------------------------------------- Tuple Key --------------------------------------- */

    #[rustfmt::skip]
    mod tup_impl {
        use super::*;
        
        macro_rules! tup_compare {
            ($n:literal, [$($numbers:tt),*], [$($args:ident),*]) => {
                #[cfg(not(feature = "assert-partial-ord"))]
                impl <$($args),*> TreeKey<$n> for ($($args,)*)
                    where $($args: Clone + Ord),*
                {
                    #[allow(non_snake_case)]
                    fn compare(&self, other: &Self) -> [std::cmp::Ordering; $n] {
                        [$(self.$numbers.cmp(&other.$numbers),)*]
                    }
                }

                #[cfg(feature = "assert-partial-ord")]
                impl <$($args),*> TreeKey<$n> for ($($args,)*)
                    where $($args: Clone + PartialOrd),*
                {
                    #[allow(non_snake_case)]
                    fn compare(&self, other: &Self) -> [std::cmp::Ordering; $n] {
                        [$(self.$numbers.partial_cmp(&other.$numbers).unwrap(),)*]
                    }
                }
            };
        }    
        
        tup_compare!(1, [0], [T1]);
        tup_compare!(2, [0, 1], [T1, T2]);
        tup_compare!(3, [0, 1, 2], [T1, T2, T3]);
        tup_compare!(4, [0, 1, 2, 3], [T1, T2, T3, T4]);
        tup_compare!(5, [0, 1, 2, 3, 4], [T1, T2, T3, T4, T5]);
        tup_compare!(6, [0, 1, 2, 3, 4, 5], [T1, T2, T3, T4, T5, T6]);
        tup_compare!(7, [0, 1, 2, 3, 4, 5, 6], [T1, T2, T3, T4, T5, T6, T7]);
        tup_compare!(8, [0, 1, 2, 3, 4, 5, 6, 7], [T1, T2, T3, T4, T5, T6, T7, T8]);
        tup_compare!(9, [0, 1, 2, 3, 4, 5, 6, 7, 8], [T1, T2, T3, T4, T5, T6, T7, T8, T9]);
        tup_compare!(10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [T1, T2, T3, T4, T5, T6, T7, T8, T9, T10]);
        tup_compare!(11, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11]);
        tup_compare!(12, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12]);
        tup_compare!(13, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13]);
        tup_compare!(14, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14]);
        tup_compare!(15, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15]);
        tup_compare!(16, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16]);
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
    use std::borrow::Borrow;

    use crate::{Dimension, TreeKey, TreeMap, TreeNode};

    impl<K, V, C, const D: usize, const N: usize> Default for TreeMap<K, V, C, D, N>
    where
        C: Dimension<D, N>,
        K: TreeKey<D>,
    {
        fn default() -> Self {
            Self::new()
        }
    }

    impl<K, V, C, const D: usize, const N: usize> Clone for TreeMap<K, V, C, D, N>
    where
        C: Dimension<D, N>,
        K: Clone,
        V: Clone,
    {
        fn clone(&self) -> Self {
            Self {
                nodes_pool: self.nodes_pool.clone(),
                unused_nodes: self.unused_nodes.clone(),
                root: self.root.clone(),
            }
        }
    }

    impl<K, V, C, const D: usize, const N: usize> TreeMap<K, V, C, D, N>
    where
        K: TreeKey<D>,
        C: Dimension<D, N>,
    {
        /* ---------------------------------------- Public Apis --------------------------------------- */

        /// Create tree with a given capacity. This is useful if you know the approximate number of
        /// nodes in the tree beforehand.
        pub fn with_capacity(capacity: usize) -> Self {
            Self {
                nodes_pool: Vec::with_capacity(capacity),
                unused_nodes: Vec::with_capacity(capacity),
                root: usize::MAX,
            }
        }

        /// Create empty tree. This does not allocate any memory.
        pub fn new() -> Self {
            Self {
                nodes_pool: Vec::new(),
                unused_nodes: Vec::new(),
                root: usize::MAX,
            }
        }

        /// Returns the number of nodes in the tree.
        pub fn len(&self) -> usize {
            self.nodes_pool.len() - self.unused_nodes.len()
        }

        /// Returns `true` if the tree is empty.
        pub fn is_empty(&self) -> bool {
            self.len() == 0
        }

        /// Returns the capacity of the tree.
        pub fn capacity(&self) -> usize {
            self.nodes_pool.capacity()
        }

        /// Clear nodes
        pub fn clear(&mut self) {
            self.nodes_pool.clear();
            self.unused_nodes.clear();
            self.root = usize::MAX;
        }

        /// Shrink to fit
        pub fn shrink_to_fit(&mut self) {
            self.nodes_pool.shrink_to_fit();
            self.unused_nodes.shrink_to_fit();
        }

        /// Fill holes in the tree nodes pool. This is useful if you have deleted a lot of nodes
        /// and want to reclaim the memory.
        ///
        /// May make slight improvement on cache locality.
        pub fn make_compact(&mut self) {
            // TODO:
        }

        /// Reserve space for nodes
        pub fn reserve(&mut self, additional: usize) {
            self.nodes_pool.reserve(additional);
            self.unused_nodes.reserve(additional);
        }

        pub fn insert(&mut self, key: K, value: V) {
            if self.root == usize::MAX {
                self.root = self.allocate_node(key, value);
            } else {
            }
        }

        pub fn get<Q>(&self, key: &Q) -> Option<&V>
        where
            K: Borrow<Q>,
        {
            None
        }

        /* ------------------------------ Index-based Manipulation ------------------------------ */

        pub fn root(&self) -> usize {
            self.root
        }

        pub fn get_at(&self, index: usize) -> Option<(&K, &V)> {
            self.nodes_pool
                .get(index)
                .and_then(|x| x.as_ref().map(|x| (&x.key, &x.value)))
        }

        pub fn get_at_mut(&mut self, index: usize) -> Option<(&K, &mut V)> {
            self.nodes_pool
                .get_mut(index)
                .and_then(|x| x.as_mut().map(|x| (&x.key, &mut x.value)))
        }

        /* ------------------------------------- Inner Impl ------------------------------------- */

        /// Allocate empty node
        fn allocate_node(&mut self, key: K, value: V) -> usize {
            let at = if let Some(index) = self.unused_nodes.pop() {
                index
            } else {
                let index = self.nodes_pool.len();
                self.nodes_pool.push(None);
                index
            };

            let new_node = TreeNode::new(key, value);

            if cfg!(debug_assertions) {
                assert!(self.nodes_pool[at].replace(new_node).is_none());
            } else {
                self.nodes_pool[at] = Some(new_node);
            }

            at
        }

        /// Release given node
        fn release_node(&mut self, index: usize) {
            if cfg!(debug_assertions) {
                let Some(node) = self.nodes_pool[index].take() else {
					panic!("Node at index {} is already released", index);
				};

                assert!(
                    node.children.children().iter().all(|&x| x == usize::MAX),
                    "Node at index {} is not empty",
                    index
                );
            } else {
                self.nodes_pool[index] = None;
            }

            self.unused_nodes.push(index);
        }
    }

    impl<K, V, C, const D: usize, const N: usize> TreeNode<K, V, C, D, N>
    where
        C: Dimension<D, N>,
    {
        fn new(key: K, data: V) -> Self {
            Self {
                children: {
                    let default = C::default();
                    debug_assert!(default.children().iter().all(|&x| x == usize::MAX));
                    default
                },
                key,
                value: data,
            }
        }
    }

    impl<K, V, C, const D: usize, const N: usize> Clone for TreeNode<K, V, C, D, N>
    where
        C: Dimension<D, N>,
        K: Clone,
        V: Clone,
    {
        fn clone(&self) -> Self {
            Self {
                children: self.children.clone(),
                key: self.key.clone(),
                value: self.value.clone(),
            }
        }
    }
}
