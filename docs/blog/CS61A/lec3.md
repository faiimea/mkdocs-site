# Lecture 3 + read 3

## Summary

在环境中，由下至上寻找变量绑定
To sum up, the environment reflect the scope of variables.More precisely, it is **the scope of declaration**

![env](media\bc9a43ce2c3f6c2735c566a7d6a89d8.png)
```py
# Create nested environment
x = 1
y = 12
def g1(x):
    def g2(x):
        # Stop here
        print(x)
    g2(x + 1)
g1(2)
```
When you call the user-defined function, you create a new local frame, binding formal parameters to actual parameters.

And this new local frame ia attached to parent frame.

When a function calls other function, (differing from def in the function), the new function's parent is global(like it).

```py
def id(x):
  return x
print(id(id)(id(13)))
```
Such as this, it won't reflect error, and this is a good example that function is value

in `id(id)` ,x is binding to id(functoin)"<function __main__.id(x)>", and it equals to id(13)

```py
# Illustration of returing a nested function.
def incr(n):
    def f(x):
        return n + x
    return f

>>>incr(5)(6)
11
# 5+x(6)
```
**why we use return f instead of return f() ?**

Because when we return f, it is a **reference** to the function, while return f(), it will **calculate** the value. So here it means 5+x(6)

(I seem it as the location is replaced by the function referrence)

## Control

### Conditional evaluation

In python, we can use this

`Truepart if Condition else Falsepart`

such as ```1 / x if x!=0 else 1```

False None 0 Emptythings are all false values, which means [] is false values as well.

### and/or

`Left and Right`

- evaluate left
- if left is false, that becomes the value of the whole expression
- otherwise the value of the expression is that of Right

for ex: `[] and 1/0`:is []

`or` 如果 x 是非 0，它返回 x 的计算值，否则它返回 y 的计算值。

```py
>>> True and 13
13

>>> False or 0
0

>>> not 10
False

>>> not None
True
```

### if-else

```py
if con:
  state1
elif con:
  state2
else:
  staten
```

What is unusual is py use indentation to divide the group. So take good use of tab

### Iteration

we can use recursive function to do it
```py
def add_sq(accum, k, n):
    """Return ACCUM + K ** 2 + (K+1)**2 + ... + N**2."""
    if k > n:
        return accum
    else:
        return add_sq(accum + k ** 2, k + 1, n)
print(add_sq(0, 1, 100))

```

### while

(just the same as cpp)
```py
while k <= n:
    accum = accum + k ** 2
    k += 1    # Another way to write k = k + 1
```

a Syntactic sugar: k+=1, in py it works as well.\

## read

### 函数

python解释器工作就是执行由语句组成的程序，
复合语句一般占据多行，一个头部和一组缩进的代码叫做子句
ex：
* 表达式，返回语句，赋值语句是简单语句
* `def`是复合语句，头部后组成函数组

最简单的函数中，函数体只有带有表达式的一个返回语句
但由于复合语句结构，函数体可以拓展到多个函数

例如，在函数体内部出现赋值语句，赋值效果仅限当前环境的帧，不会影响global frame(局部赋值)

无论函数何时被调用，`return`语句会重定向控制，停止调用流程，返回函数值

### 函数测试

**断言** ：程序员使用`assert`语句来验证预期，测试函数输出。assert语句在布尔上下文中只有一个表达式，后面是带引号的一行文本（单引号或双引号都可以，但是要一致）如果表达式求值为假，它就会显示。
```py
assert fib(8) == 13, 'The 8th Fibonacci number should be 13'
```
当被断言的表达式求值为真时，断言语句的执行没有任何效果。当它是假时，`assert`会造成执行中断。

在文件中而不是直接在解释器中编写 Python 时，测试可以写在同一个文件，或者后缀为_test.py的相邻文件中。

**Doctest**: Python 提供了一个便利的方法，将简单的测试直接写到函数的文档字符串内。文档字符串的第一行应该包含单行的函数描述，后面是一个空行。参数和行为的详细描述可以跟随在后面。此外，文档字符串可以包含调用该函数的简单交互式会话：

```py
>>> def sum_naturals(n):
        """Return the sum of the first n natural numbers
        >>> sum_naturals(10)
        55
        >>> sum_naturals(100)
        5050
        """
        total, k = 0, 1
        while k <= n:
          total, k = total + k, k + 1
        return total
```
[doctest模块](https://docs.python.org/zh-cn/3/library/doctest.html)可以用来验证交互

在文件中编写 Python 时，可以通过以下面的命令行选项启动 Python 来运行一个文档中的所有 doctest。

```py
python3 -m doctest <python_source_file>
```

只调用一个函数的测试叫做单元测试，**详尽的单元测试是良好程序设计的标志。**