# Web



##如何通过网络传输数据？
* IP协议：
    * **Internet Protocol**
    IPV4代表2^32的IP地址，40亿左右，其实不足
    目前正在启用IPV6

* URL:Uniform Resource Locator(统一资源定位符)
  
* DNS:Domain Name Systrm(域名系统)
    * 互联网上的一种服务，将URL与IP之间转换
    * www.faii.com
        * com：商业网站
        * www：万维网，服务器的名称
        * HTTP：超文本传输协议
        （浏览器与服务器沟通的协议）
        * https://：代表了加密
    * 一般来说，访问这个网站要返回的数据为 www.faii.com/index.html index.html是一个文档的文件名。

* 数据传输
    * TCP协议允许在IP后增加数字来声明服务内容（序列号）
    * 服务器返回的数据中(一部分信息)
        * 200 OK
        * 403 FORBIDDEN
        * 404 NOT FOUND
        * 418 I AM A TEAPOT（是真的）

## HTML：超文本标记语言
  * 并不是编程语言，而是标记语言。只是将物品列出来
  * [一个还不错的学习网站](https://www.w3school.com.cn/html/index.asp)，以前还有一个更亲民的，但因为我清空收藏夹导致再也找不到了（悲
  * HTML是基于标签的语言。开启标签，关闭标签
      ```html
      <head>
        <title>
          hello
        </title>
      </head>
      ```
      是很不错的对称
  * 在数据传输中，浏览器会依据服务器传回的html操作，生成网页
  * HTML不需要翻译为机器语言，但要在网络服务器上运行
  * HTML也包含图像与声音，影像，网页链接等内容，有对应的格式(href为超链接)，也可以对文本进行属性修改，设置标题，表格等等
    ```html
    <img src="ss.jpg">
    <a href="http://www.faii.com"paypal.com>
    ```
  * 进入一个网页，获取这个网页的源码并部署到自己的服务器上，就复制了一个网页（没有内容）任何人都可以获得HTML文件。所以用这个来诈骗也很简单。

  * 动态网页
    对于类似Google的网页，没有一个静态的HTML来维护，而是一个动态网页  
    在Google中有一个可输入表单，在输入后会跳转到特殊的URL
    `www.google.com/search?q=cats`
    内容为：运行搜索程序，传递一个用户输入的值。  
    对于html来说，`<form>`用来代表可输入表单（搜索栏）


## CSS：Cascading Style Sheets:层叠样式表
  * CSS相较于HTML，类似于Latex/word与txt
  * CSS通过特性来美观网页，包括层叠特性，class，标签属性值，形式内容分离等

## JS：JavaScript
* **java和js本质上没有关系**
* js允许在客户端执行代码，而不是每次从服务器刷新，客户端收到代码时就会改变网页（无需刷新）
* js会内置在html中，`<script>`区分，可以编写一些函数（响应函数）并在网站中运行
* 可以实现很多交互功能（甚至包括获取用户的IP等）