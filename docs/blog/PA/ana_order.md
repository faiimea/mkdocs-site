# 解析命令

> From PA1

## Readline
>为了让简易调试器易于使用, NEMU通过readline库与用户交互, 使用readline()函数从键盘上读入命令. 与gets()相比, readline()提供了"行编辑"的功能, 最常用的功能就是通过上, 下方向键翻阅历史记录. 事实上, shell程序就是通过readline()读入命令的. 

### Description
readline will read a line from the terminal and return it, using
**prompt** as a prompt.  If prompt is NULL or the empty string, no
prompt is issued.  The line returned is allocated with malloc(3);
the caller must free it when finished.  The line returned has the
final newline removed, so only the text of the line remains.

(在下面的示例中，"nemu"充当了prompt的角色)

readline offers editing capabilities while the user is entering
the line.  By default, the line editing commands are similar to
those of emacs.  A vi-style line editing interface is also
available.

This manual page describes only the most basic use of readline.
Much more functionality is available; see The GNU Readline
Library and The GNU History Library for additional information.

### Example

```c
static char* rl_gets() {
  static char *line_read = NULL;

  if (line_read) {
    free(line_read);
    line_read = NULL;
  }

  line_read = readline("(nemu) ");

  if (line_read && *line_read) {
    add_history(line_read);
  }

  return line_read;
}

```


## Strtok
>从键盘上读入命令后, NEMU需要解析该命令, 然后执行相关的操作. 解析命令的目的是识别命令中的参数, 例如在si 10的命令中识别出si和10, 从而得知这是一条单步执行10条指令的命令. 解析命令的工作是通过一系列的字符串处理函数来完成的, 例如框架代码中的strtok(). strtok()是C语言中的标准库函数

### Description
The C library function char *strtok(char *str, const char *delim) breaks string str into a series of tokens using the delimiter delim.

### Declaration
Following is the declaration for strtok() function.

char *strtok(char *str, const char *delim)
Parameters
str − The contents of this string are modified and broken into smaller strings (tokens).

delim − This is the C string containing the delimiters. These may vary from one call to another.

strtok()函数将一个字符串分割成一系列非空的字符组。第一次调用strtok，要parse的字符串由str指定。后面每次调用strtok就是parse同一个字符串，str必须是NULL。delim参数确定一组字节，这组字节用于分割字符串中的符号组。在后面的调用的时候，strtok可以指定不同的delim来分割字符串。

每次调用strtok，strtok都返回一个指向包含下一个tokens，以null结尾的字符串。这个字符串不包括定界字节。如果没有找到跟多的字符组，strtok返回NULL。

一系列的操作在同一个字符串上的strtok调用，维护了一个指向下一次要搜索符号组的开始位置的指针。第一次调用strtok把指针指向字符串的第一个字节。下一个符号组的开始位置是通过向前扫描下一个非delim字符。如果找到这样的一个字节，这就作为下一个符号组的开始位置。如果没有找到这样的字节，那就是没有更多的字符了。strtok就返回NULL。（因此，如果一个字符串是空的或者只包括了定界符那么它就会在第一次调用的时候返回NULL）。

### Return Value
This function returns a pointer to the first token found in the string. A null pointer is returned if there are no tokens left to retrieve.

### Example
The following example shows the usage of strtok() function.

```c
#include <string.h>
#include <stdio.h>

int main () {
   char str[80] = "This is - www.tutorialspoint.com - website";
   const char s[2] = "-";
   char *token;
   
   /* get the first token */
   token = strtok(str, s);
   
   /* walk through other tokens */
   while( token != NULL ) {
      printf( " %s\n", token );
    
      token = strtok(NULL, s);
   }
   
   return(0);
}
```
Let us compile and run the above program that will produce the following result −
```
This is 
  www.tutorialspoint.com 
  website
```