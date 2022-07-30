# Assembly
>Nand to Teris_Chapter 6.lab

## 功能
**将汇编语言(.asm)翻译为机器语言(.hack)**

## 模块构造
- 语法分析器：Parser
  - 对输入代码的访问操作

- 编码：Code
    - 将Hack汇编语言（获取的字符串）翻译为二进制码

- 符号表：Symbol Table
    - 建立和维持符号与地址之间的关联
    - hash表为经典数据结构，通过STL实现(map)

- 主程序：main
    - 标签化解决：编写两遍汇编-编译器
    - 初始化：预定义符号初始化
    - 第一pass：构建符号表，不生成代码，只处理标签（ROM）
    - 第二pass：发现变量符号时，在符号表查找，替换或定义+分配内存（RAM）
    *由于Hack架构中，程序内存与数据内存分开管理，所以不会出现内存冲突


## 有趣的东西
- iterator
    - [iterator_help](https://blog.csdn.net/weixin_44737923/article/details/104838288)
    - [auto help](https://blog.csdn.net/weixin_44737923/article/details/105597049)


- string::to_string的用法  
    - `_symbolTable["R" + std::to_string(i)] = i;`


- string::erase的用法
    - Erases part of the string, reducing its length:
```cpp
std::string str ("This is an example sentence.");
str.erase (10,8);// "This is an sentence."
str.erase (str.begin()+9);// "This is a sentence."
str.erase (str.begin()+5, str.end()-9);// "This sentence."
```

- `str.insert(str.begin(), '0');`：典型的c++式写法，比我在todo里面用的字符数组不知道好了多少倍
  

- string::substr
    - Returns a newly constructed string object **with its value initialized** to a copy of a substring of this object.
    - `str.substr (3,5);`:(pos,len)
    - `size_t pos = str.find("live");`
    - `string str3 = str.substr (pos);`:Another way to initialize it.  

- string::find_first_of
    - Searches the string for the first character that matches any of the characters specified in its arguments.
    - **return** : The position of the first character that matches.If no matches are found, the function returns **string::npos.**

- string::npos
    - Maximum value for size_t
    - **npos** is a static member constant value with the greatest possible value for an element of type size_t.
    - This constant is **defined with a value of -1**, which because size_t is an unsigned integral type, it is the largest possible representable value for this type.

- `string+="xxx"`在字符串尾部添加的简单重载


- 常量引用与STL中map查找函数的应用  
    - `map.find(const std::string &symbol)!=map.end()`  

- find_first_of
    - Returns an iterator to the first element in the range [first1,last1) that matches any of the elements in [first2,last2). If no such element is found, the function returns last1.  

- remove_if
    - Transforms the range [first,last) into a range with all the elements for which **pred returns true removed**, and returns an iterator to the **new end of that range**.
    - return : An iterator to the element that follows the last element not removed.
    - eg:`auto iter = remove_if(tmpStr.begin(), tmpStr.end(), ::isspace);`
    * auto & iterator   

- stoi ：Parses str interpreting its content as an integral number of the specified base, which is returned as an int value.  

- isspace : Check if character is a white-space  

- variant : 变体
    - Variant 数据类型是所有没被显式声明为其他类型变量的数据类型。Variant 数据类型并没有类型声明字符。
    - Variant 是一种特殊的数据类型，除了定长 String 数据及用户定义类型外，可以包含任何种类的数据。
  ```cpp
  std::variant<AL_Command,C_Command> val;
  _command.SetType(Command::A_COMMAND);
  _command.val = Line.substr(1);
  &&
  _command.SetType(Command::C_COMMAND);
  _command.val = Command::C_Command{dest, comp, jump};
  ```

- 二进制转换函数(string)
  ```cpp
  int n = std::stoi(tmpStr);
    std::string str;
    while(n!=0) { str = (n % 2 == 0 ? "0" : "1") + str; n/=2;}
    while(str.size() < 15)
    {
        str.insert(str.begin(), '0');
    }
    return str;
  ```

- 字符串的格式化（去空格，去注释）
   - 由于一句话的注释段仅有可能出现在原句后，若在原句前则失去了代码作用，因此用一行substr足够了
   ```cpp
   std::string Parser::RemoveSpaceAndComment(const std::string &str)
  {
    return RemoveSpace(RemoveComment(str));
  }

  std::string Parser::RemoveSpace(const std::string &str)
  {
    auto tmpStr = str;
    auto iter = remove_if(tmpStr.begin(), tmpStr.end(), ::isspace);
    tmpStr.erase(iter, tmpStr.end());
    return tmpStr;
  }

  std::string Parser::RemoveComment(const std::string &str)
  {
    return str.substr(0, str.find_first_of("//"));
  }
   ```
## 问题

Q：   
`c++17`的特殊数据结构`variant`在目前的IDE(Clion+WSL)中无法被编译   
A：   
在`cmakelist`中将`set(CMAKE_CXX_STANDARD 14)`改为`set(CMAKE_CXX_STANDARD 17)`则可以编译成功
（不得不说现在的cmake做的确实智能）

