# C语言

> CS50

## 前言

由于在自学CS50_C之前，我已经学习过SJTU的程设与数据结构两门课，有一年左右的编程经历，因此在这里并没有收录全部的讲座内容，而只会记录一些我比较感兴趣的东西。  

说实话，在看完这些内容之后，我心情比较糟糕。原因很简单，我确实深刻的认识到了，很早之前看到的一句话：  

**“国内绝大部分大学的本科教学，不是濒临崩溃，而是早已崩溃。”**

我大一的时候，本来是怀揣着对编程的热情来学习计算机相关的课程的的，然而一学期过后，我旷了将近半学期的程设课程。并且虽然我大摆特摆，仍然取得了接近满绩的成绩。同时，我并不觉得我的C++能力能达到优秀的水平。

在完成这一部分的学习后，我发现，即使是一个没有提前学习，没有极高的天赋的学生（甚至是没有热情的），都可以通过十小时以内的课程（其中还包括了一些游戏和动画）来掌握c语言编程的基础能力。  

而且以我的经验来看，SJTU工科平台的学生，在一学期的课程之后，C语言的平均水平，恐怕不如认真听了这十小时课的高中生。这实在是令人遗憾而悲哀的事情。



## C语言

* 一个简单的函数，有询问和导入字符串的功能
  ```cpp
  string answer = get_string("Here is the question\n");
  printf("hello，%s",answer);
  ```
  （`get_int()`等函数也是如此）

* Compiler
在Linux中，如果不使用IDE的话，可以手动编译生成汇编文件a.out
CS50用的是clang，我自己比较常用的是GCC+GDB，而且这两者的语法是很类似的
`
gcc -o test main.c # 将main.c编译并生成名字为test的文件
`
这里的-o就是-out的缩写，非常简单易懂

* 编译
这里其实是NIS1336的内容，但当时没有深刻理解，但在预习完计组之后，再看这些内容就很清晰了
    * pre-processing：将头文件用实际文件替换
    * compiling：高级语言转换为汇编指令
    * assembling：汇编指令转换为机器语言
    * linking：将需要的多个文件整合

* 一些关于linux命令的冷知识
  rm：remove
  mkdir：make dictionary
  rmdir：guess it
  十六进制以0x开头，八进制以0开头，二进制以b结尾
  地址的计数都是十六进制，可以与字节匹配

* 字符串
  字符串是以`\0`结尾的字符数组
  ```cpp
  //s=string/char*
  printf("%s",s); //打印字符串 faii
  printf("%p",s); //打印char*的地址 0x21398
  ```

### 字符串&字符数组

* C风格字符串：

用`\0`终止的一个一维字符数组,C++ 编译器会在初始化数组时，自动把 ‘\0’ 放在字符串的末尾。所以也可以利用下面的形式进行初始化
```cpp
char greeting[6] = {'H', 'e', 'l', 'l', 'o', '\0'};
char greeting[] = "Hello";
```

* C++ 字符串
    * String : C++类库中的一个类,它较char*的优势是内容可以动态拓展，以及对字符串操作的方便快捷，用+号进行字符串的连接是最常用的操作
    * char* : char* 是指向字符串的指针(其实严格来说，它是指向字符串的首个字母)，你可以让它指向一串常量字符串。
    * const char* : 该声明指出，指针指向的是一个const char类型，即不能通过当前的指针对字符串的内容作出修改(char * const是指向字符的静态指针)
    * char[] ：代表字符数组，可以对应一个字符串

* char* & char[]
一，char*是变量，值可以改变， char[]是常量，值不能改变
a是一个char型指针变量，其值（指向）可以改变；
b是一个char型数组的名字，也是该数组首元素的地址，是常量，其值不可以改变

二，char[]对应的内存区域总是可写，char*指向的区域有时可写，有时只读
```cpp
char * a="string1";
char b[]="string2";
gets(a); //试图将读入的字符串保存到a指向的区域，运行崩溃！ 
gets(b) //OK
```
解释： a指向的是一个字符串常量，即指向的内存区域只读；
b始终指向他所代表的数组在内存中的位置，始终可写

总结：char *本身是一个字符指针变量，但是它既可以指向字符串常量，又可以指向字符串变量，指向的类型决定了对应的字符串能不能改变
  
三，char * 和char[]的初始化操作有着根本区别：

char * a=”string1”;是实现了3个操作：

1: 声明一个char*变量(也就是声明了一个指向char的指针变量);  
2: 在内存中的文字常量区中开辟了一个空间存储字符串常量”string1”  
3: 返回这个区域的地址，作为值，赋给这个字符指针变量a  

实际上， char * a=”string1”; 的写法是不规范的！

`const char * a="string1";`:类型相同赋值

可以理解为`char[]`是char const类型，作为函数的声明的参数的时候，char []是被当做char *来处理的！两种形参声明写法完全等效！

关于各个格式之间的转换，[见这里](https://blog.csdn.net/ksws0292756/article/details/79432329?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165959942616782390549700%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=165959942616782390549700&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-79432329-null-null.142^v39^pc_rank_34_1&utm_term=c%2B%2B%E5%AD%97%E7%AC%A6%E4%B8%B2&spm=1018.2226.3001.4187)

变成string,直接赋值。

char[]变成别的，直接赋值。

char*变constchar*容易，const char*变char*麻烦。
`<const_cast><char*>(constchar*)`;

string变char*要通过const char*中转。

变成char[]。string逐个赋值，char* const char* strncpy_s()。

[->更多关于字符串的操作<-](http://c.biancheng.net/view/400.html)（string是功能很强大的类w）
```cpp
string s1("Hello");
s1="now"; //与常量字符串不一样,可以用 char* 类型的变量、常量，以及 char 类型的变量、常量对 string 对象进行赋值
s3.assign(s1); //赋值
s2.assign(s1, 1, 2);
int len=s3.length(); //长度
s1.append(s2); //连接
s1.append(s2, 1, 2);
int n = s1.compare(s2); //比较
n = s1.compare(1, 2, s2, 0, 3);
string s2 = s1.substr(2, 4); //字串
s1.swap(s2);//交换
if ((n = s1.find('u')) != string::npos) //查找
//rfind,find_first_of,find_last_of,find_first_not_of
s1.replace(1, 3, "123456", 2, 4);//替换
s1.erase(1, 3); //删除
s1.insert(2, "123");//插入
string s("afgcbed");//STL操作（迭代器）
    string::iterator p = find(s.begin(), s.end(), 'c');
    if (p!= s.end())
        cout << p - s.begin() << endl;  //输出 3
    sort(s.begin(), s.end());
    cout << s << endl;  //输出 abcdefg
```

### 字符串读入
一直是比较关注的一个话题，现在统一整理一下
1.cin>>

用法一：最常用、最基本的用法，输入一个数字：

用法二：接受一个字符串，遇“空格”、“Tab”、“回车”都结束

2：cin.get()

用法一：cin.get(字符变量名)可以用来接收字符

用法二：cin.get(字符数组名，接收字符数)用来接收一行字符串，可以接收空格`cin.get(a,20);`

用法三：cin.get(无参数)没有参数,主要是用于舍弃输入流中的不需要的字符, 或者舍弃回车, 弥补cin.get(字符数组名,接收字符数目)的不足.

3：cin.getline()

这个是主要的输入字符串的方法。

cin.getline()实际上有三个参数，cin.getline(接受字符串到m,接受个数5,结束字符)当第三个参数省略时，系统默认为’\0’ 是‘/n’
接受个数可以设置多一些。

4：getline()

接收一个字符串，可以接收空格并输出，不过要包含#include<string>
这也就弥补了之前cin.getline()和cin.get（）的不能读取string的一个小的弊端
这里还要将这个写法稍微改正一下getline(cin,字符串数组名);

```cpp
#include<iostream>
#include<string>
using namespace std;
int main()
{
	string s;
	getline(cin,s);
	cout << s;
	return 0;
}
```

5：gets()

gets()： 接受一个字符串，可以接收空格并输出，需包含 #include<string>.

```cpp
char m[20];
gets(m); //不能写成m=gets();
cout<<m<<endl;
//输入：abcabcabc
//输出：abcabcabc

//输入：abc abc abc
//输出：abc abc abc

```

### C++ IO

[->关于C++文件读写<-](https://www.runoob.com/cplusplus/cpp-files-streams.html)

* 打开文件
  ```cpp
  ifstream  afile;
  afile.open("file.dat", ios::out | ios::in );
  ```
* 关闭文件
  `infile.close();`
* 写入文件
  `outfile << data << endl;`
* 读取文件
  `infile >> data; `
* 文件位置指针
  istream 和 ostream 都提供了用于重新定位文件位置指针的成员函数。这些成员函数包括关于 istream 的 seekg（"seek get"）和关于 ostream 的 seekp（"seek put"）

  seekg 和 seekp 的参数通常是一个长整型。第二个参数可以用于指定查找方向。查找方向可以是 ios::beg（默认的，从流的开头开始定位），也可以是 ios::cur（从流的当前位置开始定位），也可以是 ios::end（从流的末尾开始定位）。
