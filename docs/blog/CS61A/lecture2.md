# lecture 2 + text

## Fuctions as values

* New functions definitions
  ```py
  def saxb(a,x,b) #header: name and formal parameters
    return a*x+b #body: computation
  ```
  **The second line must be indented**

  in cpp, the signature of function includes other information such as the type of input and output, while in python, just the number and name of parameters

* Name of function
  * in python, name is part of the function
  * there are anonymous function, we call `lambda`
  * `lambda a, x, b: a * x + b`

* Type of function
  * pure function
    * `add(a,b)`
  * impure function
    * side effects `print(1)`
    * randint function: interactive

* Expression
  * **recursively defined concept**, which means exp can consists of other expression

## Environment

* Substitution
  * assignment  `x=3`
  * def statement is kind of assignment
    * a good way to understand the logic of function: substitution:
    ```py
    def incr(n)
      def f(x)
        return n+x
      return f
    ```
    `func incr(n):|def f...return f |(5)`

    so incr(5):return func f(x):return 5+x

    btw even if there exists same name in packed expression, you can just substityte the **free occurrence variable** which is not defined or bound

    EX:
    ```py
    def hmmm(x)
      def f(x) # here is the definition of x, x is the parameter of f
        return x
      return f
    ```
* Environment
  * in environment, **names bound to values**
  * ways include pre-defined(abs), imported(pi), assigned, assigned by def
  * In environment, name has meaning
* Frame
  * 环境由帧的序列组成，每一帧包含了一些绑定(binding)
  * An environment in which an expression is evaluated consists of a sequence of frames, depicted as boxes. Each frame contains bindings, each of which associates a name with its corresponding value. There is a single global frame. 
  * Calling User-Defined Functions
    * **Bind the arguments** to the names of the function's formal parameters in a new local frame.
    * Execute the body of the function in the environment that starts with this frame.
  * 在帧中，形参与实参相绑定。局部帧在唯一的全局帧中，共享名称。而在不同的帧中，相同的名称绑定的值不同（说实话，这个看起来是很显然的事情，但是仔细思考后就没那么简单）

## Designing Functions
* Each function should have exactly one job.
* **Don't repeat yourself** is a central tenet of software engineering. If you find yourself copying and pasting a block of code, you have probably found an opportunity for functional abstraction.

A function definition will often include documentation describing the function, called a docstring, which must be indented along with the function body.

Docstrings are conventionally triple quoted.(" is quote)

When you call help with the name of a function as an argument, you see its docstring (type q to quit Python help)

*This really amazes me, what an interesting language

## Default Argument Values

(类似于cpp中的缺省)

In Python, we can provide default values for the arguments of a function. When calling that function, arguments with default values are optional. 

If they are not provided, then the default value is bound to the formal parameter name instead.

(Some values that never change, such as the fundamental constant k, can be bound in the function body or in the global frame.)

`def pressure(v, t, n=6.022e23):`

## hw01:

``` py
return if_function(cond(), true_func(), false_func())
```

``` py
 if cond():
        return true_func()
    else:
        return false_func()
```

what is the diffrence?
in A code, because the parameters are functions, 会首先计算函数的值，再返回给主函数，所以所有函数都会运行，而在B code，只会运行cond对应的函数