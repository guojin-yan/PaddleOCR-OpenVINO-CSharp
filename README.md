![OpenVINO™ C# API](https://socialify.git.ci/guojin-yan/OpenVINO-CSharp-API/image?description=1&descriptionEditable=💞%20OpenVINO%20wrapper%20for%20.NET💞%20&forks=1&issues=1&logo=https%3A%2F%2Fs2.loli.net%2F2023%2F01%2F26%2FylE1K5JPogMqGSW.png&name=1&owner=1&pattern=Circuit%20Board&pulls=1&stargazers=1&theme=Light)

[简体中文](README_cn.md) | English

# Deploying PaddleOCR using OpenVINO<sup>TM</sup> C # API

This project is mainly based on the development of the [OpenVINO<sup>TM</sup>C # API](https://github.com/guojin-yan/OpenVINO-CSharp-API) project, using the C # programming language NET framework, use [OpenVINO<sup>TM</sup>]( https://github.com/openvinotoolkit/openvino) deployment tool to deploy [PaddleOCR](https://github.com/paddlepaddle/paddleocr) series models under Baidu PaddlePaddle, and realize image text recognition, layout analysis, table recognition and other functions.
The project provides simple cases and API interfaces for secondary development, which can be developed and used according to your own needs.

# 🛠 Project Environment

In this project, we mainly use our self-developed **OpenVINO<sup>TM</sup> C # API ** project and **OpenCvSharp4 **project. The **NuGet Package ** package and installation method used are as follows:

## <img title="NuGet" src="https://s2.loli.net/2023/08/08/jE6BHu59L4WXQFg.png" alt="" width="40"> NuGet Package

- **OpenVINO.CSharp.API >= 2023.2.0.2**

- **OpenVINO.runtime.win >= 2023.2.0.1**
- **OpenCvSharp4.Windows >= 4.8.0.20230708**
- **OpenCvSharp4.Extensions >= 4.8.0.20230708**

## 😇 Installation

NuGet Package can be installed through Visual Studio or through the **dotnet ** command. The installation method is as follows:

```shell
dotnet add package OpenVINO.CSharp.API
dotnet add package OpenVINO.runtime.win
dotnet add package OpenCvSharp4.Windows
dotnet add package OpenCvSharp4.Extensions
```

# 🎯Quick Start

## Clone Source Code

```bash
git clone https://github.com/guojin-yan/PaddleOCR-OpenVINO-CSharp.git
cd PaddleOCR-OpenVINO-CSharp
```

## Obtain Prediction Model

The models used in the project are all from [PaddleOCR](https://github.com/paddlepaddle/paddleocr ) The model directory can be referenced

-  [PP-OCR  Model List](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/models_list.md)

- [PP-Structure  Model List](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/ppstructure/docs/models_list.md)

> <div><b>
> <font color=red size="4">Attention：</font>
> </b></div>
>
>
> &emsp;After testing, OpenVINO currently supports all models in the **PP-OCR Model List and PP Structure Model List,** as well as models in PaddlePaddle format. Users can use them directly after downloading, but the table recognition model *ppstructure_ Mobile_ V2.0_ SLANet* needs to be converted before it can be used, and the input shape of the fixed mold line needs to be [1, 3, 488, 488]. The conversion method keyi refers to the article: [Paddle2ONNX](https://github.com/paddlepaddle/paddleocr/blob/release/2.7/deploy/paddle2onnx/readme.md )

For the convenience of quick use, here is a quick download method for the model:

```bash
cd model
./ocr_model_download.sh
./stru_model_download.sh
```

The directory structure of the downloaded model file is:

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

## OCR Rec

You can run the project directly through **Visual Studio ** or through the **dotnet run ** command. The **dotnet run ** command is as follows:

```bash
cd PaddleOCR-OpenVINO-CSharp/sample
dotnet run ./../../
```

The output after running the program is shown in the following figure:

| <span><img src="https://s2.loli.net/2023/12/23/pJBGrle9AFDjOEP.png" width=1000/></span> | <span><img src="https://s2.loli.net/2023/12/22/ESbjL24Ydxq1ePH.png" width=400 /></span> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

# 📱 Contact 

If you are planning to deploy the PaddleOCR model using OpenVINO, please refer to this case. If you have any questions during use, you can contact me through the following methods.

<div align=center><span><img src="https://s2.loli.net/2023/10/18/d6QUWL7HG523BuR.png" height=300/></span></div>