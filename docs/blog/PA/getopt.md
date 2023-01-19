# Getopt

## Func 'getopt'

### 解析
一个典型的 unix 命令行有着如下的形式。

```sh
command [options] arguments
```
选项的形式为连字符 (-) 紧跟着一个唯一的字符用来标识该选项，以及一个针对该选项的可选参数。带有一个参数的选项能够以可选的方式在参数和选项之间用空格分开。多个选项可以在一个单独的连字符后归组在一起,而组中最后一个选项可能会带有一个参数。根据这些规则,下面这些命令都是等同的
```sh
grep -l -i -f patterns *.c
grep -lif patterns *.c
grep -lifpatterns *.c
```
在上面这些命令中，-l 和 -i 选项没有参数,而 -f 选项将字符串 patterns 当做它的参数。 因为许多程序都需要按照上述格式来解析选项,相关的机制被封装在了一个标准库函数中,这就是 getopt()。
```c
#include <unistd.h>

extern int optind, opterr, optopt;
extern char *optarg;

int getopt(int argc, char *const argv[], const char *optstring);

//See main text for description of return value
```
函数 getopt() 解析给定在参数argc和argv中的命令行参数集合。这两个参数通常是从 main() 函数的参数列表中获取。参数 optstring 指定了函数 getopt() 应该寻找的命令行选项集合,该参数由一组字符组成，每个字符标识一个选项。SUSv3中规定了 getopt() 至少应该可以接受62个字符[a-zA-Z0-9]作为选项。除了: ? - 这几个对 getopt() 来说有着特殊意义的字符外,大多数实现还允许其他的字符也作为选项出现。

每个选项字符后可以跟一个冒号字符,表示这个选项带有一个参数。

我们通过连续调用 getopt() 来解析命令行。每次调用都会返回下一个未处理选项的信息。 如果找到了选项,那么代表该选项的字符就作为函数结果返回。如果到达了选项列表的结尾 getopt() 就返回-1。如果选项带有参数, getopt() 就把全局变量 optarg 设为指向这个参数。

如果选项不带参数,那么 glibc的 getopt() 实现(同大多数实现一样)会将 optarg设为 NULL。但是,SUSv3并没有对这种行为做出规定。因此基于可移植性的考虑,应用程序不能依赖这种行为(通常也不需要)。

每次调用 getopt() 时,全局变量 optind 都得到更新,其中包含着参数列表 argv 中未处理的下一个元素的索引。(当把多个选项归组到一个单独的单词中时, getopt() 内部会做一些记录工作,以此跟踪该单词,找出下一个待处理的部分。)在首次调用 getopt()之前,变量 optind 会自动设为1。在如下两种情况中我们可能会用到这个变量。

1.如果 getopt() 返回了-1,表示目前没有更多的选项可解析了,且 optind 的值比argc要小,那么 argv[optind]就表示命令行中下一个非选项单词。
2.如果我们处理多个命令行向量或者重新扫描相同的命令行,那么我们必须手动将 optind 重新设为1。

在下列情况中, getopt() 函数会返回-1,表示已到达选项列表的结尾。

1.由 argc 加上 argv 所代表的列表已到达结尾(即 argv[optind]为NULL)。

2.argv中下一个未处理的单字不是以选项分隔符打头的(即, argv[optind][0]不是连字符)。

3.argv中下一个未处理的单字只由一个单独的连字符组成(即, argvloptind] 为 -)。 有些命令可以理解这种参数,该单字本身代表了特殊的意义。

4.argv中下一个未处理的单字由两个连字符(-)组成。在这种情况下, getopt() 会悄悄地读取这两个连字符,并将 optind 调整为指向双连字符之后的下一个单字。就算命令行中的下一个单字(在双连字符之后)看起来像一个选项(即,以一个连字符开头), 这种语法也能让用户指出命令的选项结尾。比如,如果我们想利用grep在文件中查找字符串 -k,那么我们可以写成

`grep -- -k myfile`

当 getopt() 在处理选项列表时,可能会出现两种错误。一种错误是当遇到某个没有指定在 optstring 中的选项时会出现。另一种错误是当某个选项需要一个参数，而参数却未提供时会出现(即,选项出现在命令行的结尾)。有关 getopt() 是如何处理并上报这些错误的规则如下。

1.默认情况下, getopt()在标准错误输出上打印出一条恰当的错误消息,并将字符 ? 作为函数返回的结果。在这种情况下,全局变量 optopt 返回出现错误的选项字符(即,未能识别出来的或缺少参数的那个选项)。

2.全局变量 opterr 可用来禁止显示由 getopt() 打印出的错误消息。默认情况下,这个变量被设为1。如果我们将它设为0,那么 getopt() 将不再打印错误消息,而是表现的如同 上一条所描述的那样。程序可以通过检查函数返回值是否为?字符来判断是否出错, 并打印出用户自定义的错误消息。

3.此外,还有一种方法可以用来禁止显示错误消息。可以在参数 optstring 中将第一个字符指定为冒号(这么做会重载将 opterr 设为0的效果)。在这种情况下,错误上报的规则同将 opterr 设为0时一样,只是此时缺失参数的选项会通过函数返回冒号:来报告。如果需要的话,我们可以根据不同的返回值来区分这两类错误(未识别的选项,以及缺失参数的选项)。


### 函数示例

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <stdarg.h>

#define printable(ch) (isprint((unsigned char) ch) ? ch : '#')

void fatal(char *msg){
    printf(msg);
    printf("\n");
    exit(1);
}

static void usageError(char *progName, char *msg, int opt){
    if (msg != NULL && opt != 0)
        fprintf(stderr, "%s (-%c)\n", msg, printable(opt));
    fprintf(stderr, "Usage: %s [-p arg] [-x]\n", progName);
    exit(EXIT_FAILURE);
}

int
main(int argc, char *argv[])
{
    int opt, xfnd;
    char *pstr;

    xfnd = 0;
    pstr = NULL;

    while ((opt = getopt(argc, argv, ":p:x")) != -1) {
        printf("opt =%3d (%c); optind = %d", opt, printable(opt), optind);
        if (opt == '?' || opt == ':')
            printf("; optopt =%3d (%c)", optopt, printable(optopt));
        printf("\n");

        switch (opt) {
        case 'p': pstr = optarg;        break;
        case 'x': xfnd++;               break;
        case ':': usageError(argv[0], "Missing argument", optopt);
        case '?': usageError(argv[0], "Unrecognized option", optopt);
        default:  fatal("Unexpected case in switch()");
        }
    }

    if (xfnd != 0)
        printf("-x was specified (count=%d)\n", xfnd);
    if (pstr != NULL)
        printf("-p was specified with the value \"%s\"\n", pstr);
    if (optind < argc)
        printf("First no option argument is \"%s\" at argv[%d]\n",
                argv[optind], optind);
    exit(EXIT_SUCCESS);
}
```

```sh
[root@izj6cfw9yi1iqoik31tqbgz c]# ./a.out -xxxp para freecls
opt =120 (x); optind = 1
opt =120 (x); optind = 1
opt =120 (x); optind = 1
opt =112 (p); optind = 3
-x was specified (count=3)
-p was specified with the value "para"
First no option argument is "freecls" at argv[3]
[root@izj6cfw9yi1iqoik31tqbgz c]# ./a.out -x -p para -c freecls
opt =120 (x); optind = 2
opt =112 (p); optind = 4
opt = 63 (?); optind = 5; optopt = 99 (c)
Unrecognized option (-c)
Usage: ./a.out [-p arg] [-x]
[root@izj6cfw9yi1iqoik31tqbgz c]# ./a.out -x -p freecls -- -c
opt =120 (x); optind = 2
opt =112 (p); optind = 4
-x was specified (count=1)
-p was specified with the value "freecls"
First no option argument is "-c" at argv[5]

[root@izj6cfw9yi1iqoik31tqbgz c]# ./a.out -x -p freecls -c
opt =120 (x); optind = 2
opt =112 (p); optind = 4
opt = 63 (?); optind = 5; optopt = 99 (c)
Unrecognized option (-c)
Usage: ./a.out [-p arg] [-x]
```

## Func 'getopt_long'

### 解析

getopt函数只能处理短选项，而getopt_long函数两者都可以，可以说getopt_long已经包含了getopt_long的功能。因此，这里就只介绍getopt_long函数。

```C
#include <unistd.h>  
extern char *optarg;  
extern int optind, opterr, optopt;  
#include <getopt.h>
int getopt(int argc, char * const argv[],const char *optstring);  
int getopt_long(int argc, char * const argv[], const char *optstring, const struct option *longopts, int *longindex);  
int getopt_long_only(int argc, char * const argv[], const char *optstring, const struct option *longopts, int *longindex);
```

参数以及返回值介绍（以上三个函数都适用）：

1、argc和argv和main函数的两个参数一致。

2、optstring: 表示短选项字符串。

    形式如“a:b::cd:“，分别表示程序支持的命令行短选项有-a、-b、-c、-d，冒号含义如下：
    (1)只有一个字符，不带冒号——只表示选项， 如-c 
    (2)一个字符，后接一个冒号——表示选项后面带一个参数，如-a 100
    (3)一个字符，后接两个冒号——表示选项后面带一个可选参数，即参数可有可无， 如果带参数，则选项与参数直接不能有空格
        形式应该如-b200
3、longopts：表示长选项结构体。结构如下：

```C
struct option 
{  
     const char *name;  
     int         has_arg;  
     int        *flag;  
     int         val;  
};  
eg:
 static struct option longOpts[] = {
      { "daemon", no_argument, NULL, 'D' },
      { "dir", required_argument, NULL, 'd' },
      { "out", required_argument, NULL, 'o' },
      { "log", required_argument, NULL, 'l' },
      { "split", required_argument, NULL, 's' },
      { "http-proxy", required_argument, &lopt, 1 },
      { "http-user", required_argument, &lopt, 2 },
      { "http-passwd", required_argument, &lopt, 3 },
      { "http-proxy-user", required_argument, &lopt, 4 },
      { "http-proxy-passwd", required_argument, &lopt, 5 },
      { "http-auth-scheme", required_argument, &lopt, 6 },
      { "version", no_argument, NULL, 'v' },
      { "help", no_argument, NULL, 'h' },
      { 0, 0, 0, 0 }
    };
```
  (1)name:表示选项的名称,比如daemon,dir,out等。

  (2)has_arg:表示选项后面是否携带参数。该参数有三个不同值，如下：

    a:  no_argument(或者是0)时   ——参数后面不跟参数值，eg: --version,--help
    b: required_argument(或者是1)时 ——参数输入格式为：--参数 值 或者 --参数=值。eg:--dir=/home
    c: optional_argument(或者是2)时  ——参数输入格式只能为：--参数=值
  (3)flag:这个参数有两个意思，空或者非空。

    a:如果参数为空NULL，那么当选中某个长选项的时候，getopt_long将返回val值。
            eg，可执行程序 --help，getopt_long的返回值为h.             
    b:如果参数不为空，那么当选中某个长选项的时候，getopt_long将返回0，并且将flag指针参数指向val值。

            eg: 可执行程序 --http-proxy=127.0.0.1:80 那么getopt_long返回值为0，并且lopt值为1。

  (4)val：表示指定函数找到该选项时的返回值，或者当flag非空时指定flag指向的数据的值val。

4、longindex：longindex非空，它指向的变量将记录当前找到参数符合longopts里的第几个元素的描述，即是longopts的下标值。

5、全局变量：

    （1）optarg：表示当前选项对应的参数值。

    （2）optind：表示的是下一个将被处理到的参数在argv中的下标值。

    （3）opterr：如果opterr = 0，在getopt、getopt_long、getopt_long_only遇到错误将不会输出错误信息到标准输出流。opterr在非0时，向屏幕输出错误。

    （4）optopt：表示没有被未标识的选项。

6、返回值：

    （1）如果短选项找到，那么将返回短选项对应的字符。

    （2）如果长选项找到，如果flag为NULL，返回val。如果flag不为空，返回0

    （3）如果遇到一个选项没有在短字符、长字符里面。或者在长字符里面存在二义性的，返回“？”

    （4）如果解析完所有字符没有找到（一般是输入命令参数格式错误，eg： 连斜杠都没有加的选项），返回“-1”

    （5）如果选项需要参数，忘了添加参数。返回值取决于optstring，如果其第一个字符是“：”，则返回“：”，否则返回“？”。
注意：

    （1）longopts的最后一个元素必须是全0填充，否则会报段错误

    （2）短选项中每个选项都是唯一的。而长选项如果简写，也需要保持唯一性。

###  示例
```C
#include <stdio.h>     /* for printf */
#include <stdlib.h>    /* for exit */
#include <getopt.h>
 
int
main(int argc, char **argv)
{
    int c;
    int digit_optind = 0;
 
   while (1) {
        int this_option_optind = optind ? optind : 1;
        int option_index = 0;
        static struct option long_options[] = {
            {"add",     required_argument, 0,  0 },
            {"append",  no_argument,       0,  0 },
            {"delete",  required_argument, 0,  0 },
            {"verbose", no_argument,       0,  0 },
            {"create",  required_argument, 0, 'c'},
            {"file",    required_argument, 0,  0 },
            {0,         0,                 0,  0 }
        };
 
       c = getopt_long(argc, argv, "abc:d:012",
                 long_options, &option_index);
        if (c == -1)
            break;
 
       switch (c) {
        case 0:
            printf("option %s", long_options[option_index].name);
            if (optarg)
                printf(" with arg %s", optarg);
            printf("\n");
            break;
 
       case '0':
        case '1':
        case '2':
            if (digit_optind != 0 && digit_optind != this_option_optind)
              printf("digits occur in two different argv-elements.\n");
            digit_optind = this_option_optind;
            printf("option %c\n", c);
            break;
 
       case 'a':
            printf("option a\n");
            break;
 
       case 'b':
            printf("option b\n");
            break;
 
       case 'c':
            printf("option c with value '%s'\n", optarg);
            break;
 
       case 'd':
            printf("option d with value '%s'\n", optarg);
            break;
 
       case '?':
            break;
 
       default:
            printf("?? getopt returned character code 0%o ??\n", c);
        }
    }
 
   if (optind < argc) {
        printf("non-option ARGV-elements: ");
        while (optind < argc)
            printf("%s ", argv[optind++]);
        printf("\n");
    }
 
   exit(EXIT_SUCCESS);
}
```