# Latex攻破

## 结构

### 导言区

```latex
\documentclass{cumcmthesis}	文章类型
\usepackage	调用宏
\title{xxx}	文章标题
```

### 正文区

```latex
\begin{document} 开始正文
\maketitle	生成标题。在标准类中，标题出现在单独的页面上，但在 article 类中它位于第一页的顶部。
\section{xx}	章节标题
\subsection{xx}	副标题
\end{document} 结束正文
```

### 附录区

```latex
\begin{appendices}
\section{模板所用的宏包}
```

## 字体

使用字体命令来改变字体（关于字体可以不必太关心，模板中已经设计好了）

```latex
\textrm{罗马}
\textsf{无衬线}
\texttt{等宽}
\textbf{加粗}
\textit{斜体}
```

## 特殊符号

<img src="https://pic2.zhimg.com/80/v2-d9fecf976d5fedbd67d4b7900b522ff5_720w.jpg" alt="img" style="zoom:50%;" />

<img src="https://pic4.zhimg.com/v2-0a827e7a6f1fc1fa6880df2e479f391b_r.jpg" alt="preview" style="zoom:50%;" />

<img src="https://pic4.zhimg.com/v2-7f8dccdfe117dfac270483b9865cd743_r.jpg" alt="preview" style="zoom:50%;" />

```latex
\^ 上标	\_ 下标

无序列表是这样的：
\begin{itemize}
    \item one
    \item two
    \item ...
\end{itemize}

有序列表是这样子的：
\begin{enumerate}
    \item one
    \item two
    \item ...
\end{enumerate}
```

## 表格

标准三线表：

```latex
\begin{table}[!htbp]
    \caption{标准三线表格}\label{tab:001} \centering
    \begin{tabular}{ccc}
        \toprule[1.5pt]
        符号 & 意义 & 单位 \\
        \midrule[1pt]
        5 & 269.8 & 0.000674\\
        10 & 第 i 次移动后标号 j 的主索节点附近标记点偏离距离的均值 & 0.001035\\
        20 & 640.2 & 0.001565\\
        \bottomrule[1.5pt]
    \end{tabular}
\end{table}
```

## 图片

```latex
\begin{figure}[!h]
    \centering
    \includegraphics[width=.6\textwidth]{smokeblk}
    \caption{电路图}
    \label{fig:circuit-diagram}
\end{figure}
```



```LATEX
\begin{figure}
    \centering
    \begin{minipage}[c]{0.48\textwidth}
        \centering
        \includegraphics[height=0.2\textheight]{cat}
        \subcaption{一只猫}
    \end{minipage}
    \begin{minipage}[c]{0.48\textwidth}
        \centering
        \includegraphics[height=0.2\textheight]{smokeblk}
        \subcaption{电路图}
    \end{minipage}
    \caption{多图并排示例}
\end{figure}
```



## 公式

[在线LaTeX公式编辑器-编辑器 (latexlive.com)](https://www.latexlive.com/##)

## 参考文献

```latex
\LaTeX{}的入门书籍可以看《\LaTeX{}入门》\cite{liuhaiyang2013latex}。

%参考文献
\begin{thebibliography}{9}%宽度9
    \bibitem[1]{liuhaiyang2013latex}
    刘海洋.
    \newblock \LaTeX {}入门\allowbreak[J].
    \newblock 电子工业出版社, 北京, 2013.
    \bibitem[2]{mathematical-modeling}
    全国大学生数学建模竞赛论文格式规范 (2020 年 8 月 25 日修改).
    \bibitem{3} \url{https://www.latexstudio.net}
\end{thebibliography}
```

## 引用代码

```latex
\begin{lstlisting}[language=xxx]
xxx
 \end{lstlisting}
```

