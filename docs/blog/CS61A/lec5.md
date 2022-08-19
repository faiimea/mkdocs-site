# Lecture 6/7 Recursion

## Read 1.7

### 递归函数

*如果对于第一次学习编程的人来说，递归的确是个不好理解的概念

函数的主体调用函数本身，则为递归函数

```python
# 一个典型的递归例子：求数字各位之和
>>> def sum_digits(n):
        """Return the sum of the digits of positive integer n."""
        if n < 10:
            return n
        else:
            all_but_last, last = n // 10, n % 10
            return sum_digits(all_but_last) + last
```

在许多递归函数的主体中可以找到一种常见的模式。

正文以*基本情况开始，基本情况*是一个条件语句，用于定义最简单处理的输入的函数的行为。

然后，基本情况后跟一个或多个*递归调用*。递归调用始终具有特定字符：它们简化了原始问题。递归函数通常以与我们以前使用的迭代方法不同的方式解决问题：递归函数通过逐步简化问题来表达计算。

将递归调用视为函数式抽象被称为***信念的递归飞跃***。我们根据函数本身来定义一个函数，但只需相信在验证函数的正确性时，更简单的情况将正常工作。在这个例子中，我们相信事实`（n-1）`会正确计算`（n-1）！`;如果这个假设成立，我们只需要检查`n！`是否正确计算。通过这种方式，**验证递归函数的正确性是归纳证明的一种形式。**(归纳证明induction的内容会在离散数学中学习)

每个主体最多调用一次自己的函数为线性递归函数

### 相互递归

当递归过程分为两个相互调用的函数时，这些函数被称为*相互递归的*。

SICP中的示例为`is_even`与`is_odd`函数的相互调用，通过打破两个函数之间的抽象边界，可以将互递归函数转换为单个递归函数。在这个例子中，`is_odd`的主体可以合并到`is_even`的主体中，确保在`is_odd`的主体中将`n`替换为`n-1`，以反映传递给它的参数

互递归提供了一种在复杂的递归程序中维护抽象的机制。（不过是可替代的

这种维护抽象的办法在编程中可以起到作用，例如鹅卵石一例

```python
>>> def play_alice(n):
        if n == 0:
            print("Bob wins!")
        else:
            play_bob(n-1)
>>> def play_bob(n):
        if n == 0:
            print("Alice wins!")
        elif is_even(n):
            play_alice(n-2)
        else:
            play_alice(n-1)
```

互递归可以很好的契合回合制：

自然分解是将每个策略封装在自己的函数中。这允许我们修改一种策略而不影响另一种策略，从而保持两者之间的抽象障碍。为了融入游戏的逐回合性质，这两个函数在每个回合结束时相互调用。

### 树递归

树递归中函数多次调用自身（最典型的就是斐波那契数列）

这个递归定义相对于我们之前的尝试非常有吸引力：它完全反映了斐波那契数列的熟悉定义。具有多个递归调用的函数被称为*树递归函数*，因为每个调用分支为多个较小的调用，每个调用分支到更小的调用中，就像树的分支从主干延伸时变得更小但更多一样。（树的比喻很恰当）

*事实上对于这个问题，采用dp的思想其实比树递归好很多（我记得

同时，二分法也是一个常用的树递归应用（当然也可以用迭代实现）

而树递归为最优算法的问题是分区问题

*一个语法糖，用and和or可以代替if分支

#### 分区问题

正整数 `n` 的分区数（使用大小不超过 `m` 的部分）是 `n` 可以表示为正整数部分之和（以递增的顺序排列）的方式数。例如，使用最多 4 个部分的 6 个分区数为 9。

我们将定义一个函数` count_partitions（n， m），`该函数使用最多 `m` 的部分返回 `n` 的不同分区数。此函数具有一个简单的解决方案，即基于以下观察结果的树递归函数：

使用最大 `m` 的整数对 `n` 进行分区的方法数等于

1. 使用最大为 `m` 的整数对 `n-m` 进行分区的方法数，以及
2. 使用最大 `m-1` 的整数对 `n` 进行分区的方法数。

对于这道题的解法，模拟一棵树是很易得的

所有划分`n`的方法都可以分为两组：至少包含一个`m的方法`和不包含m的方法。此外，第一组中的每个分区都是 `n-m` 的分区，最后添加 `m`。在上面的示例中，前两个分区包含 4 个，其余分区不包含。

因此，我们可以递归地将使用最大 `m` 的整数对 `n` 进行分区的问题简化为两个更简单的问题：（1） 分区一个较小的 `n-m`，（包括m）以及 （2） 使用较小的分量进行分区，直到 `m-1`。（不包括m）

**同时，由于m为0（甚至为1）的情况下是可以直接得到答案的，所以作为基本情况前置**

```python
>>> def count_partitions(n, m):
        """Count the ways to partition n using parts up to m."""
        if n == 0:
            return 1
        elif n < 0:
            return 0
        elif m == 0:
            return 0
        else:
            return count_partitions(n-m, m) + count_partitions(n, m-1)
```

**我们可以将树递归函数视为探索不同的可能性。**在这种情况下，我们探讨了使用`m`尺寸的一部分的可能性以及不使用的可能性。第一个和第二个递归调用对应于这些可能性。

*其实递归是很符合思维方式的编程方法，分类讨论，探索可能性，分离基本情况



## Lecture 6 Recursion

```python
# Tail-recursive sum_squares
def sum_squares2(N):
    def part_sum(accum, k):
        """Return the sum of ACCUM + K**2 + (K+1)**2 + ... + N**2."""
        if k > N:
            return accum
        else:
            return part_sum(accum + k**2, k + 1)
    return part_sum(0, 1)
```

一个格式完整，逻辑清晰的递归函数

inner def可以在函数定义中起到预定义的作用，如本题中就预定义了accum与k的值，这层抽象给了函数更大的泛用性

（tail recursion指的是调用主体是主体的最后一句的函数）

### Recursive Thinking

Don't look at its implementation,Just its documentation.

It is a recursive leap of faith(abstraction). What we need to notice is to prevent an infinite recursion. You eventually need a value to return (in some condition)

Recursive thinking = Induction thinking

* 一个简单的走迷宫程序

```python
def is_path(blocked, x0, y0):
   if blocked(x0, y0):
       return False
   elif y0 == 0:
       return True
   else:
       return (is_path(blocked, x0-1, y0-1) \
              or is_path(blocked, x0, y0-1) \
              or is_path(blocked, x0+1, y0-1))
```

​		通过自然语言的分析实现递归，来达成目标

​		同时，将false，true，or改为0，1，+可以求出路线数目

* 一个简单的删除数字中对应数字的程序

  可以想到一个很基础的迭代写法，但是如果要是用递归的话

  ```python
  def remove_digit(n, digit):
      if n == 0:
          return 0
      if n % 10 == digit:
          return remove_digit(n // 10, digit)
      return n % 10 + remove_digit(n // 10, digit) * 10
  
  ```

  还是一样，对吧 ^ ^

  这里比较困难的地方就是想到分类的条件，最后一位是否可以删除，分别考虑两种情况就比较好办了

