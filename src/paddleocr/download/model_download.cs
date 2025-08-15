using OpenVinoSharp.Extensions.utility;
using System;
using System.IO;
using System.Threading.Tasks;
using Path = System.IO.Path;

namespace OpenVinoSharp.Extensions.model.PaddleOCR
{
    public static class OCRDetModels
    {
        public enum OCRDetModelsType
        {
            /// <summary>
            /// 【最新】原始超轻量模型，支持中英文、多语种文本检测
            /// </summary>
            ch_PP_OCRv4_det,
            /// <summary>
            /// 【最新】原始高精度模型，支持中英文、多语种文本检测
            /// </summary>
            ch_PP_OCRv4_server_det,
            /// <summary>
            /// 原始超轻量模型，支持中英文、多语种文本检测
            /// </summary>
            ch_PP_OCRv3_det_slim,
            /// <summary>
            /// slim量化+蒸馏版超轻量模型，支持中英文、多语种文本检测
            /// </summary>
            ch_PP_OCRv3_det,
            /// <summary>
            /// slim量化+蒸馏版超轻量模型，支持中英文、多语种文本检测
            /// </summary>
            ch_PP_OCRv2_det_slim,
            /// <summary>
            /// 原始超轻量模型，支持中英文、多语种文本检测
            /// </summary>
            ch_PP_OCRv2_det,
            /// <summary>
            /// 通用模型，支持中英文、多语种文本检测，比超轻量模型更大，但效果更好
            /// </summary>
            ch_ppocr_mobile_slim_v2_det,
            /// <summary>
            /// 原始超轻量模型，支持中英文、多语种文本检测
            /// </summary>
            ch_ppocr_mobile_v2_det,
            /// <summary>
            /// slim裁剪版超轻量模型，支持中英文、多语种文本检测
            /// </summary>
            ch_ppocr_server_v2_det,
            /// <summary>
            /// 最新】slim量化版超轻量模型，支持英文、数字检测
            /// </summary>
            en_PP_OCRv3_det_slim,
            /// <summary>
            /// 【最新】原始超轻量模型，支持英文、数字检测
            /// </summary>
            en_PP_OCRv3_det,
            /// <summary>
            /// 【最新】slim量化版超轻量模型，支持多语言检测
            /// </summary>
            ml_PP_OCRv3_det_slim,
            /// <summary>
            /// 【最新】原始超轻量模型，支持多语言检测
            /// </summary>
            ml_PP_OCRv3_det,
            /// <summary>
            /// PP_OCRv5 文字识别模型
            /// </summary>
            PP_OCRv5_server_det,
            /// <summary>
            /// PP_OCRv5 文字识别轻量化模型
            /// </summary>
            PP_OCRv5_mobile_det

        }

        public static async Task<string> Get(OCRDetModelsType type, string path = "./")
        {
            string url = "";
            if (type == OCRDetModelsType.ch_PP_OCRv4_det)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar";
            }
            else if (type == OCRDetModelsType.ch_PP_OCRv4_server_det)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_server_infer.tar";
            }
            else if (type == OCRDetModelsType.ch_PP_OCRv3_det_slim)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_slim_infer.tar";
            }
            else if (type == OCRDetModelsType.ch_PP_OCRv3_det)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar";
            }
            else if (type == OCRDetModelsType.ch_PP_OCRv2_det_slim)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_slim_quant_infer.tar";
            }
            else if (type == OCRDetModelsType.ch_PP_OCRv2_det)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar";
            }
            else if (type == OCRDetModelsType.ch_ppocr_server_v2_det)
            {
                url = "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar";
            }
            else if (type == OCRDetModelsType.ch_ppocr_mobile_v2_det)
            {
                url = "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar";
            }
            else if (type == OCRDetModelsType.ch_ppocr_mobile_slim_v2_det)
            {
                url = "https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_det_prune_infer.tar";
            }
            else if (type == OCRDetModelsType.en_PP_OCRv3_det)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar";
            }
            else if (type == OCRDetModelsType.en_PP_OCRv3_det_slim)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_slim_infer.tar";
            }
            else if (type == OCRDetModelsType.ml_PP_OCRv3_det)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar";
            }
            else if (type == OCRDetModelsType.ml_PP_OCRv3_det_slim)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_slim_infer.tar";
            }
            else if (type == OCRDetModelsType.PP_OCRv5_server_det)
            {
                url = "https://github.com/guojin-yan/PaddleOCR-OpenVINO-CSharp/releases/download/Modelv5/PP-OCRv5_server_det_onnx.onnx";
            }
            else if (type == OCRDetModelsType.PP_OCRv5_mobile_det)
            {
                url = "https://github.com/guojin-yan/PaddleOCR-OpenVINO-CSharp/releases/download/Modelv5/PP-OCRv5_mobile_det_onnx.onnx";
            }


            else
            {
                throw new Exception("Model selection error!");
            }
            Uri uri = new Uri(url);
            string file_name = System.IO.Path.GetFileName(uri.LocalPath);
            string file_path = Path.Combine(path, file_name);
            if (!File.Exists(file_path))
                _ = Download.download_file_async(url, file_path).Result;
            if (type == OCRDetModelsType.PP_OCRv5_server_det)
            {
                return Path.Combine(path, "PP-OCRv5_server_det_onnx.onnx");
            }
            else if (type == OCRDetModelsType.PP_OCRv5_mobile_det)
            {
                return Path.Combine(path, "PP-OCRv5_mobile_det_onnx.onnx");
            }
            else
            {
                Download.unzip(file_path, path);
                return Path.Combine(path, System.IO.Path.GetFileNameWithoutExtension(uri.LocalPath), "inference.pdmodel");
            }
        }
    }

    public static class OCRRecModels
    {
        public enum OCRRecModelsType
        {
            /// <summary>
            /// 【最新】超轻量模型，支持中英文、数字识别 
            /// </summary>
            ch_PP_OCRv4_rec,
            /// <summary>
            /// 【最新】高精度模型，支持中英文、数字识别
            /// </summary>
            ch_PP_OCRv4_server_rec,
            /// <summary>
            ///  slim量化版超轻量模型，支持中英文、数字识别
            /// </summary>
            ch_PP_OCRv3_rec_slim,
            /// <summary>
            /// 原始超轻量模型，支持中英文、数字识别
            /// </summary>
            ch_PP_OCRv3_rec,
            /// <summary>
            /// slim量化版超轻量模型，支持中英文、数字识别
            /// </summary>
            ch_PP_OCRv2_rec_slim,
            /// <summary>
            /// 原始超轻量模型，支持中英文、数字识别
            /// </summary>
            ch_PP_OCRv2_rec,
            /// <summary>
            /// slim裁剪量化版超轻量模型，支持中英文、数字识别
            /// </summary>
            ch_ppocr_mobile_slim_v2_rec,
            /// <summary>
            /// 原始超轻量模型，支持中英文、数字识别
            /// </summary>
            ch_ppocr_mobile_v2_rec,
            /// <summary>
            /// 通用模型，支持中英文、数字识别
            /// </summary>
            ch_ppocr_server_v2_rec,
            /// <summary>
            /// 【最新】原始超轻量模型，支持英文、数字识别
            /// </summary>
            en_PP_OCRv4_rec,
            /// <summary>
            /// slim量化版超轻量模型，支持英文、数字识别
            /// </summary>
            en_PP_OCRv3_rec_slim,
            /// <summary>
            /// 原始超轻量模型，支持英文、数字识别
            /// </summary>
            en_PP_OCRv3_rec,
            /// <summary>
            /// slim裁剪量化版超轻量模型，支持英文、数字识别
            /// </summary>
            en_number_mobile_slim_v2_rec,
            /// <summary>
            /// 原始超轻量模型，支持英文、数字识别
            /// </summary>
            en_number_mobile_v2_rec,
            /// <summary>
            /// 韩文识别 ppocr/utils/dict/korean_dict.txt
            /// </summary>
            korean_PP_OCRv3_rec,
            /// <summary>
            /// 日文识别 ppocr/utils/dict/japan_dict.txt
            /// </summary>
            japan_PP_OCRv3_rec,
            /// <summary>
            /// 中文繁体识别 ppocr/utils/dict/chinese_cht_dict.txt
            /// </summary>
            chinese_cht_PP_OCRv3_rec,
            /// <summary>
            /// 泰卢固文识别 ppocr/utils/dict/te_dict.txt
            /// </summary>
            te_PP_OCRv3_rec,
            /// <summary>
            /// 卡纳达文识别 ppocr/utils/dict/ka_dict.txt
            /// </summary>
            ka_PP_OCRv3_rec,
            /// <summary>
            /// 泰米尔文识别 ppocr/utils/dict/ta_dict.txt
            /// </summary>
            ta_PP_OCRv3_rec,
            /// <summary>
            /// 拉丁文识别 ppocr/utils/dict/latin_dict.txt
            /// </summary>
            latin_PP_OCRv3_rec,
            /// <summary>
            /// 阿拉伯字母 ppocr/utils/dict/arabic_dict.txt
            /// </summary>
            arabic_PP_OCRv3_rec,
            /// <summary>
            /// 斯拉夫字母 ppocr/utils/dict/cyrillic_dict.txt
            /// </summary>
            cyrillic_PP_OCRv3_rec,
            /// <summary>
            /// 梵文字母 ppocr/utils/dict/devanagari_dict.txt
            /// </summary>
            devanagari_PP_OCRv3_rec,
            /// <summary>
            /// PP_OCRv5 文字识别模型
            /// </summary>
            PP_OCRv5_server_rec,
            /// <summary>
            /// PP_OCRv5 文字识别轻量化模型
            /// </summary>
            PP_OCRv5_mobile_rec
        }

        public static async Task<string> Get(OCRRecModelsType type, string path = "./")
        {
            string url = "";
            if (type == OCRRecModelsType.ch_PP_OCRv4_rec)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar";
            }
            else if (type == OCRRecModelsType.ch_PP_OCRv4_server_rec)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_server_infer.tar";
            }
            else if (type == OCRRecModelsType.ch_PP_OCRv3_rec_slim)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_slim_infer.tar";
            }
            else if (type == OCRRecModelsType.ch_PP_OCRv3_rec)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar";
            }
            else if (type == OCRRecModelsType.ch_PP_OCRv2_rec_slim)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_slim_quant_infer.tar";
            }
            else if (type == OCRRecModelsType.ch_PP_OCRv2_rec)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar";
            }
            else if (type == OCRRecModelsType.ch_ppocr_mobile_slim_v2_rec)
            {
                url = "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_slim_infer.tar";
            }
            else if (type == OCRRecModelsType.ch_ppocr_mobile_v2_rec)
            {
                url = "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar";
            }
            else if (type == OCRRecModelsType.ch_ppocr_server_v2_rec)
            {
                url = "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar";
            }
            else if (type == OCRRecModelsType.en_PP_OCRv4_rec)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar";
            }
            else if (type == OCRRecModelsType.en_PP_OCRv3_rec_slim)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_slim_infer.tar";
            }
            else if (type == OCRRecModelsType.en_PP_OCRv3_rec)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar";
            }
            else if (type == OCRRecModelsType.en_number_mobile_slim_v2_rec)
            {
                url = "https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/en_number_mobile_v2.0_rec_slim_infer.tar";
            }
            else if (type == OCRRecModelsType.en_number_mobile_v2_rec)
            {
                url = "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_infer.tar";
            }
            else if (type == OCRRecModelsType.korean_PP_OCRv3_rec)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_infer.tar";
            }
            else if (type == OCRRecModelsType.japan_PP_OCRv3_rec)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_infer.tar";
            }
            else if (type == OCRRecModelsType.chinese_cht_PP_OCRv3_rec)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/chinese_cht_PP-OCRv3_rec_infer.tar";
            }
            else if (type == OCRRecModelsType.te_PP_OCRv3_rec)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/te_PP-OCRv3_rec_infer.tar";
            }
            else if (type == OCRRecModelsType.ka_PP_OCRv3_rec)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ka_PP-OCRv3_rec_infer.tar";
            }
            else if (type == OCRRecModelsType.ta_PP_OCRv3_rec)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ta_PP-OCRv3_rec_infer.tar";
            }
            else if (type == OCRRecModelsType.latin_PP_OCRv3_rec)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_infer.tar";
            }
            else if (type == OCRRecModelsType.arabic_PP_OCRv3_rec)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/arabic_PP-OCRv3_rec_infer.tar";
            }
            else if (type == OCRRecModelsType.cyrillic_PP_OCRv3_rec)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_infer.tar";
            }
            else if (type == OCRRecModelsType.devanagari_PP_OCRv3_rec)
            {
                url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/devanagari_PP-OCRv3_rec_infer.tar";
            }
            else if (type == OCRRecModelsType.PP_OCRv5_server_rec)
            {
                url = "https://github.com/guojin-yan/PaddleOCR-OpenVINO-CSharp/releases/download/Modelv5/PP-OCRv5_server_rec_onnx.onnx";
            }
            else if (type == OCRRecModelsType.PP_OCRv5_mobile_rec)
            {
                url = "https://github.com/guojin-yan/PaddleOCR-OpenVINO-CSharp/releases/download/Modelv5/PP-OCRv5_mobile_rec_onnx.onnx";
            }


            else
            {
                throw new Exception("Model selection error!");
            }
            Uri uri = new Uri(url);
            string file_name = System.IO.Path.GetFileName(uri.LocalPath);
            string file_path = Path.Combine(path, file_name);
            if (!File.Exists(file_path))
                _ = Download.download_file_async(url, file_path).Result;
            if (type == OCRRecModelsType.PP_OCRv5_server_rec)
            {
                return Path.Combine(path, "PP-OCRv5_server_rec_onnx.onnx");
            }
            else if (type == OCRRecModelsType.PP_OCRv5_mobile_rec)
            {
                return Path.Combine(path, "PP-OCRv5_mobile_rec_onnx.onnx");
            }
            else
            {
                Download.unzip(file_path, path);
                return Path.Combine(path, System.IO.Path.GetFileNameWithoutExtension(uri.LocalPath), "inference.pdmodel");
            }

        }
    }
    public static class OCRClsModels
    {
        public enum OCRClsModelsType
        {
            /// <summary>
            /// slim量化版模型，对检测到的文本行文字角度分类
            /// </summary>
            ch_ppocr_mobile_slim_v2_cls,
            /// <summary>
            /// 原始分类器模型，对检测到的文本行文字角度分类
            /// </summary>
            ch_ppocr_mobile_v2_cls,
            /// <summary>
            /// PP_OCRv5 文字识别模型
            /// </summary>
            PP_OCRv5_server_cls,
            /// <summary>
            /// PP_OCRv5 文字识别轻量化模型
            /// </summary>
            PP_OCRv5_mobile_cls
        }

        public static async Task<string> Get(OCRClsModelsType type, string path = "./")
        {
            string url = "";
            if (type == OCRClsModelsType.ch_ppocr_mobile_slim_v2_cls)
            {
                url = "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_slim_infer.tar";
            }
            else if (type == OCRClsModelsType.ch_ppocr_mobile_v2_cls)
            {
                url = "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar";
            }
            else if (type == OCRClsModelsType.PP_OCRv5_server_cls)
            {
                url = "https://github.com/guojin-yan/PaddleOCR-OpenVINO-CSharp/releases/download/Modelv5/PP-OCRv5_server_cls_onnx.onnx";
            }
            else if (type == OCRClsModelsType.PP_OCRv5_mobile_cls)
            {
                url = "https://github.com/guojin-yan/PaddleOCR-OpenVINO-CSharp/releases/download/Modelv5/PP-OCRv5_mobile_cls_onnx.onnx";
            }

            else
            {
                throw new Exception("Model selection error!");
            }
            Uri uri = new Uri(url);
            string file_name = System.IO.Path.GetFileName(uri.LocalPath);
            string file_path = Path.Combine(path, file_name);
            if (!File.Exists(file_path))
                _ = Download.download_file_async(url, file_path).Result;
            if (type == OCRClsModelsType.PP_OCRv5_server_cls)
            {
                return Path.Combine(path, "PP-OCRv5_server_cls_onnx.onnx");
            }
            else if (type == OCRClsModelsType.PP_OCRv5_mobile_cls)
            {
                return Path.Combine(path, "PP-OCRv5_mobile_cls_onnx.onnx");
            }
            else
            {
                Download.unzip(file_path, path);
                return Path.Combine(path, System.IO.Path.GetFileNameWithoutExtension(uri.LocalPath), "inference.pdmodel");
            }

        }
    }
}
