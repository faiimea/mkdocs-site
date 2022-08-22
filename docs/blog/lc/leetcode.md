# Leetcode 101

## 贪心算法

贪心算法或贪心思想采用贪心的策略，保证每次操作都是局部最优的，从而使最后得到的结果是全局最优的。

在每个人身上都是**局部最优**的。又因为全局结果是局部结果的**简单求和**，且局部结果**互不相干**，因此局部最优的策略也同样是全局最优的策略。

### 455.Assign Cookies(E)

#### 关于sort函数

void sort (**RandomAccessIterator** first, **RandomAccessIterator** last, Compare comp);

Sorts the elements in the range `[first,last)` into ascending order.

The elements are compared using `operator<` for the first version, and comp for the second.

```c++
std::sort (myvector.begin(), myvector.end(), myobject); 
```

常用的数组排序方式见上，由于`[first,last)`，所以涉及到的`vetcor`容器中，`vector.end()`返回的iterator是所有元素的下一位。

### 135.Candy(H)

做完了题目 455，会不会认为存在比较关系的贪心策略一定需要排序或是选择？虽然这一道题也是运用贪心策略，但我们只需要简单的两次遍历即可。

事实上本题采用双遍历的贪心算法会带来O(n)的空间复杂度，若通过讨论递增递减数列的长度可以优化至O(1)，不过较难以理解。

其中关于递减递增序列的转换需要点思维强度，考虑一下 [3, 4, 9, 8, 7, 6] 的排列。