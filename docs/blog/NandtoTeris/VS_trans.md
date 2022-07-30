## 有趣的东西
- ofstream 和 fstream 的 write() 成员方法实际上继承自 ostream 类，其功能是将内存中 buffer 指向的 count 个字节的内容写入文件，基本格式如下：
  ```cpp
  ostream & write(char* buffer, int count);
  ```
    其中，buffer 用于指定要写入文件的二进制数据的起始位置；count 用于指定写入字节的个数。
    
    如果使用了c++的string类，再使用write函数的时候，需要用`cstr`函数将`string`转换为`char*`
    
    实例如下
    ```cpp
    std::string outPut = "@261\nD=A\n"
                         "@SP\nM=D\n" // Esp init
                         "@Sys.init\n" // call
                         "0;JMP\n";
    _writer.write(outPut.c_str(), outPut.size());
    ```
  
- pair
    - Pair of values
    This class couples together a pair of values, which may be of different types ( and ). The individual values can be accessed through its public members and .
    - make_pair():```std::make_pair("+", "Binary");```