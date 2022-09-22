# 基于Csharp和OpenVINO部署PaddleOCR模型

## 项目简介

&emsp;该项目基于OpenVINOTM模型推理库，在C#语言下，调用封装的OpenVINOTM动态链接库，部署推理PP-OCR中的文字识别模型；实现了在C#平台调用OpenVINOTM部署PP-OCR文字识别模型。

&emsp;如图所示，对于一个完整的文字识别流程主要分为三个流程：识别文字区域、识别文字方向、识别文字内容。其中识别文字区域主要是将一张图片上的所有文字区域识别出来，并将其处理成一个个小的文字区域；文本方向识别主要是识别当前文字的方向，并调整文本方向到正确方向；识别文字内容主要是将前两步处理后的文字区域中的文字识别出来。通过以上三个步骤，就可以实现提取一张图片上文字内容。

&emsp;当前新项目主要实现了文本区域识别以及文本内容识别两个步骤，由于所识别的文字都是正向。因此对文本方向未作过多的研究。下面的讲解主要围绕文本区域识别与文本内容识别展开。

​                               ![image-20220922095836258](E:\Git_space\基于Csharp和OpenVINO部署PaddleOCR模型\doc\image\image-20220922095836258.png)

 

## 项目编码环境

&emsp;为了防止复现代码出现问题，列出以下代码开发环境，可以根据自己需求设置，注意OpenVINOTM一定是2022版本，其他依赖项可以根据自己的设置修改。

- 操作系统：Windows 11

- OpenVINOTM：2022.1

- OpenCV：4.5.5

- Visual Studio：2022

-  C#框架：.NET 6.0

- OpenCvSharp：OpenCvSharp4

## 源码下载方式

&emsp;项目所使用的源码均已经在Github和Gitee上开源，

```
Github:

git clone https://github.com/guojin-yan/Csharp_and_OpenVINO_deploy_PaddleOCR.git

Gitee:

git clone https://gitee.com/guojin-yan/Csharp_and_OpenVINO_deploy_PaddleOCR.git
```



