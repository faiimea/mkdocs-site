# Lecture 4/5 Higher-Order Functions/Environment

## Preface
Becaues I have learned how to understand the relationship of functions in C++, so the content of higher-order functions is not quiet complex for me. Therefore, I just select part of lecture and take it down.

## Functions as templates
A function can be seemde as a template for computation. So we should DRY.  
**Fuction parameters allow us to pass into a computation**, like `sum_mation(N,sin)`  
The sin is the function that passed in.
```py
# term is a funtion
def summation(N, term):
    k = 1
    sum = 0
    while k <= N:
        sum += term(k)
        k += 1
    return sum
```
```py
def sum_squares(N):
    def square(x):
        return x*x
    return summation(N, square)

# or(cheaper)

def sum_squares(N):
    return summation(N, lambda x: x*x)
```

## Functions produce functions
Functions are vlaues,meaning that we can **assign them to variables,pass them to functions, and return them from functions**

```py
# This is cooler than cpp
def add_func(f, g):
    """Return function that returns F(x)+G(x) for argument x."""
    def adder(x):           #
        return f(x) + g(x)  # or return lambda x: f(x) + g(x)
    return adder            # 

h = add_func(abs, neg)
print(h(-5))

# Generalizing still more:

def combine_funcs(op):
    def combined(f, g):
        def val(x):
            return op(f(x), g(x))
        return val
    return combined

add_func = combine_funcs(add)

h = add_func(abs, neg)  

```
The adder here is a function on function. And, well, this is a complex example. We can divide it into different parts.
<img src="https://s2.loli.net/2022/08/02/di9H3PnegsoOvxI.png" ></a>

Q: Why we return `val` instead of returning `val(x)`?  
A: We don't want to call the function, we just need to **return a referrence** of it. So we can gain a function instead of gaining a value computed.

Q: Can we use function to simulate the branch?
A: 
The struct like `if_func(1/x, x>0, 0)` makes no sense because x>0 isn't a value.
We can use it like
```py
def if_func(then_expr, cond, else_expr):
  return then_expr() if condition else else_expr()
if_func(lambda: 1/x, x>0, lambda: 0)
```
Here we call the function because what we need is value. It has no parameter because we define it in lambda.

```py
 def f():
        return 0

    def g():
        print(f())

    def h():
        def f():
            return 1
        g()

    h()
```
**The answer is 0, not 1 !**(I was wrong)
Because the parent frame of g frame is **global**. So when we define g, we use the definition of f in global frame, which return 0.(不是简单的近者优先)
```py
def f():
        return 0

    g = f

    def f():
        return 1

    print(g())
```
I was wrong again !!!
The answer should be 0. We can understand it like
```py
x=3
y=x
x=4
```

**All the complex code could be plodding through step by step**, becuase that is what pc does. In this lecture, It means analyze the environment and frame connection.

```py
    def f():
        return 0

    def g():
        return f()

    def h(k):
        def f():
            return 1
        p = k
        return p()

    print(h(g))
```
Answer is 0 as well, because the g can only see the global frame.

```py
def eg6():
    def f(p, k):
        def g():
            print(k)
        if k == 0:
            f(g, 1)
        else:
            p()

    f(None, 0)
```
The answer is 0, the meaning of this example is that even though the frame resulted from a call(f), its parent frame is still global frame.

## Read-1.6

**Functions are a method of abstraction that describe compound operations independent of the particular values of their arguments.**

One of the things we should demand from a powerful programming language is the ability to build abstractions by assigning names to common patterns and then to work in terms of the names directly.

为了将特定的通用模式表达为具名概念，我们需要构造可以接受其他函数作为参数的函数，或者将函数作为返回值的函数。操作函数的函数叫做**高阶函数**。

### 作为参数的函数

这三个函数在背后都具有相同模式。它们大部分相同，只是名字、用于计算被加项的k的函数，以及提供k的下一个值的函数不同。我们可以通过向相同的模板中填充槽位来生成每个函数：

我们可以在 Python 中使用上面展示的通用模板，并且把槽位变成形式参数来轻易完成它。

### 作为一般方法的函数

首先，命名和函数允许我们抽象而远离大量的复杂性。当每个函数定义不重要时，由求值过程触发的计算过程是相当复杂的，并且我们甚至不能展示所有东西。

其次，基于事实，我们拥有了非常通用的求值过程，小的组件组合在复杂的过程中。理解这个过程便于我们验证和检查我们创建的程序。

### 定义函数 III：嵌套定义

上面的例子演示了将函数作为参数传递的能力如何提高了编程语言的表现力。每个通用的概念或方程都能映射为自己的小型函数

这一方式的一个负面效果是全局帧会被小型函数弄乱。

另一个问题是我们限制于特定函数的签名：iter_improve的update参数必须只接受一个参数。Python 中，**嵌套函数**的定义解决了这些问题，这些问题的解决方案是把**函数放到其他定义的函数体中。**

就像局部赋值，局部的def语句仅仅影响当前的**局部帧**。这些函数仅仅当square_root求值时在作用域内。和求值过程一致，局部的def语句在square_root调用之前并不会求值。

```PY
>>> def square_root(x):
        def update(guess):
            return average(guess, x/guess)
        def test(guess):
            return approx_eq(square(guess), x)
        return iter_improve(update, test)
```

**词法作用域**。局部定义的函数也可以访问它们定义所在作用域的名称绑定。这个例子中，update引用了名称x，它是外层函数square_root的一个形参。这种在嵌套函数中共享名称的规则叫做词法作用域。**严格来说，内部函数能够访问定义所在环境（而不是调用所在位置）的名称。**(这里很重要，内部函数访问的是定于所在的环境，但由于调用时看起来比较接近，常常会出错误)

我们需要两个对我们环境的扩展来兼容词法作用域。

每个用户定义的函数都有一个关联环境：它的定义所在的环境。
当一个用户定义的函数调用时，它的局部帧扩展于函数所关联的环境。

词法作用域的两个关键优势。

* 局部函数的名称并不影响定义所在函数外部的名称，因为局部函数的名称绑定到了定义处的当前局部环境中，而不是全局环境。
  
* 局部函数可以访问外层函数的环境。这是因为局部函数的函数体的求值环境扩展于定义处的求值环境

在环境内部定义的函数，由于封装性，也被称为**闭包**

### 作为返回值的函数
（比较有趣的一个点，个人理解为返回一个函数的引用）

定义复合函数：
```py
>>> def compose1(f, g):
        def h(x):
            return f(g(x))
        return h
>>> add_one_and_square = compose1(square, successor)
>>> add_one_and_square(12)
169
```
由于这个函数的作用是要生成一个新的，复合的，仅有一个参数的函数，所以在内部定义函数h，并且将`add_one_and_square`与函数h（引用）绑定，此后`add_one_and_square`便可以做完函数使用
* 关于这里为什么不直接return f(g)
  
  语法不清，没法把x传进来

###  Lambda 表达式

Python 中，我们可以使用 Lambda 表达式凭空创建函数，它会求值为匿名函数。Lambda 表达式是函数体**具有单个返回表达式的函数**，不允许出现赋值和控制语句。
```py
>>> def compose1(f,g):
        return lambda x: f(g(x))

>>> compose1 = lambda f,g: lambda x: f(g(x))
```
（实际上这里和上面的compose1具有相同的作用）

*这里介绍了通过py编程实现**牛顿法**的步骤，牛顿法是一个迭代改进算法，是一个用于解决微分方程的强大的通用计算方法。（通过切线模拟）


### *函数装饰器

```py
>>> def trace1(fn):
        def wrapped(x):
            print('-> ', fn, '(', x, ')')
            return fn(x)
        return wrapped
>>> @trace1
    def triple(x):
        return 3 * x
>>> triple(12)
->  <function triple at 0x102a39848> ( 12 )
36
```
这个例子中，定义了高阶函数trace1，它返回一个函数，这个函数在调用它的参数之前执行print语句来输出参数。

triple的def语句拥有一个注解，@trace1，它会影响def的执行规则。像通常一样，函数triple被创建了，但是，triple的名称并没有绑定到这个函数上，而是绑定到了在新定义的函数triple上调用trace1的返回函数值上。

在代码中，这个装饰器等价于：

```py
>>> def triple(x):
        return 3 * x
>>> triple = trace1(triple)
```
