![OpenVINO™ C# API](https://socialify.git.ci/guojin-yan/OpenVINO-CSharp-API/image?description=1&descriptionEditable=💞%20OpenVINO%20wrapper%20for%20.NET💞%20&forks=1&issues=1&logo=https%3A%2F%2Fs2.loli.net%2F2023%2F01%2F26%2FylE1K5JPogMqGSW.png&name=1&owner=1&pattern=Circuit%20Board&pulls=1&stargazers=1&theme=Light)

简体中文| [English](README.md)

# 使用 OpenVINO<sup>TM </sup> C# API 部署 PaddleOCR

该项目主要基于开发的[OpenVINO<sup>TM </sup> C# API](OpenVINO<sup>TM </sup> C# API)项目，基于 C# 编程语言在.NET框架下使用[OpenVINO<sup>TM </sup>](https://github.com/openvinotoolkit/openvino)部署工具部署百度飞桨下的 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 系列模型，实现图片文字识别、版面分析以及表格识别等功能。

项目提供了简单的案例以及二次开发的API接口i，大家可以根据自己需求进行再次开发与使用。

# 🛠 项目环境

在本项目中主要使用的是自己开发的**OpenVINO<sup>TM </sup> C# API**项目以及**OpenCvSharp4**项目，所使用**NuGet Package**程序包以及安装方式如下所示

## <img title="NuGet" src="https://s2.loli.net/2023/08/08/jE6BHu59L4WXQFg.png" alt="" width="40"> NuGet Package

- **OpenVINO.CSharp.API >= 2023.2.0.2**

- **OpenVINO.runtime.win >= 2023.2.0.1**
- **OpenCvSharp4.Windows >= 4.8.0.20230708**
- **OpenCvSharp4.Extensions >= 4.8.0.20230708**

## 😇 安装方式

NuGet Package 可以通过Visual Studio 安装或者通过**dotnet**命令安装，安装方式如下：

```shell
dotnet add package OpenVINO.CSharp.API
dotnet add package OpenVINO.runtime.win
dotnet add package OpenCvSharp4.Windows
dotnet add package OpenCvSharp4.Extensions
```

# 🎯 快速开始

## 获取项目源码

```bash
git clone https://github.com/guojin-yan/PaddleOCR-OpenVINO-CSharp.git
cd PaddleOCR-OpenVINO-CSharp
```

## 获取预测模型

项目中所使用的模型均来自于 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) ，模型目录可以参考

-  [PP-OCR 系列模型列表](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/models_list.md)

- [PP-Structure 系列模型列表](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/ppstructure/docs/models_list.md)

> <div><b>
>  <font color=red size="4">注意：</font>
> </b></div>
>
> &emsp;经过测试，OpenVINO目前已经支持**PP-OCR 系列模型列表、PP-Structure 系列模型列表**中的所有模型，并且支持Paddlepaddle格式的模型，用户在下载后可以直接使用，但是表格识别模型*ppstructure_mobile_v2.0_SLANet*需要进行转换才可以使用，需要固定模行输入形状为[1, 3, 488, 488]，转换方式keyi 参考该文章：[Paddle2ONNX](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/deploy/paddle2onnx/readme.md).

为了方便大家快速使用，此处提供了模型的快速下载方式：

```bash
cd model
./ocr_model_download.sh
./stru_model_download.sh
```

下载后模型文件目录结构为：

```
model
   ├──── paddle
            ├──── ch_ppocr_mobile_v2.0_cls_infer
            ├──── ch_PP-OCRv4_det_infer
            ├──── ch_PP-OCRv4_rec_infer
            ├──── ch_ppstructure_mobile_v2.0_SLANet_infer
            ├──── en_ppstructure_mobile_v2.0_SLANet_infer
            ├──── picodet_lcnet_x1_0_fgd_layout_cdla_infer
```

## OCR识别

可以直接通过**Visual Studio**直接运行该项目或者通过**dotnet run**指令运行该项目，**dotnet run**命令如下：

```bash
cd PaddleOCR-OpenVINO-CSharp/sample
dotnet run ./../../
```

程序运行后输出如下图所示：

| <span><img src="https://s2.loli.net/2023/12/23/pJBGrle9AFDjOEP.png" width=1000/></span> | <span><img src="https://s2.loli.net/2023/12/22/ESbjL24Ydxq1ePH.png" width=400 /></span> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

# 📱 Contact 

如果您准备使用OpenVINO部署PaddleOCR模型，欢迎参考本案例。在使用中有任何问题，可以通过以下方式与我联系。

<div align=center><span><img src="https://s2.loli.net/2023/10/18/d6QUWL7HG523BuR.png" height=300/></span></div>