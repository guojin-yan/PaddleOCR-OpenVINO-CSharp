using OpenVinoSharp.Extensions.utility;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static OpenVinoSharp.Extensions.model.PaddleOCR.OCRClsModels;

namespace OpenVinoSharp.Extensions.model.PaddleOCR
{
    public class OCRDicts
    {
        public enum OCRDictsType 
        {
            /// <summary>
            /// 韩文识别
            /// </summary>
            korean_dict,
            /// <summary>
            /// 日文识别
            /// </summary>
            japan_dict,
            /// <summary>
            /// 中文繁体识别
            /// </summary>
            chinese_cht_dict,
            /// <summary>
            /// 泰卢固文识别
            /// </summary>
            te_dict,
            /// <summary>
            /// 卡纳达文识别
            /// </summary>
            ka_dict,
            /// <summary>
            /// 泰米尔文识别
            /// </summary>
            ta_dict,
            /// <summary>
            /// 拉丁文识别
            /// </summary>
            latin_dict,
            /// <summary>
            /// 阿拉伯字母
            /// </summary>
            arabic_dict,
            /// <summary>
            /// 斯拉夫字母
            /// </summary>
            cyrillic_dict,
            /// <summary>
            /// 梵文字母
            /// </summary>
            devanagari_dict,
            /// <summary>
            /// 英文识别模型
            /// </summary>
            en_dict,
            /// <summary>
            /// 中文识别模型
            /// </summary>
            ppocr_keys_v1,
            /// <summary>
            /// 英文表格字典
            /// </summary>
            table_structure_dict,
            /// <summary>
            /// 中文表格字典
            /// </summary>
            table_structure_dict_ch,
            /// <summary>
            /// PP-OCRv5字典
            /// </summary>
            ppocrv5_dict

        }
        public static async Task<string> Get(OCRDictsType type, string path = "./")
        {
            string url = "";
            if (type == OCRDictsType.ppocr_keys_v1)
            {
                url = "https://github.com/guojin-yan/PaddleOCR-OpenVINO-CSharp/releases/download/dict/ppocr_keys_v1.txt";
            }
            else if (type == OCRDictsType.en_dict)
            {
                url = "https://github.com/guojin-yan/PaddleOCR-OpenVINO-CSharp/releases/download/dict/en_dict.txt";
            }
            else if (type == OCRDictsType.arabic_dict)
            {
                url = "https://github.com/guojin-yan/PaddleOCR-OpenVINO-CSharp/releases/download/dict/arabic_dict.txt";
            }
            else if (type == OCRDictsType.chinese_cht_dict)
            {
                url = "https://github.com/guojin-yan/PaddleOCR-OpenVINO-CSharp/releases/download/dict/chinese_cht_dict.txt";
            }
            else if (type == OCRDictsType.cyrillic_dict)
            {
                url = "https://github.com/guojin-yan/PaddleOCR-OpenVINO-CSharp/releases/download/dict/cyrillic_dict.txt";
            }
            else if (type == OCRDictsType.devanagari_dict)
            {
                url = "https://github.com/guojin-yan/PaddleOCR-OpenVINO-CSharp/releases/download/dict/devanagari_dict.txt";
            }
            else if (type == OCRDictsType.japan_dict)
            {
                url = "https://github.com/guojin-yan/PaddleOCR-OpenVINO-CSharp/releases/download/dict/japan_dict.txt";
            }
            else if (type == OCRDictsType.ka_dict)
            {
                url = "https://github.com/guojin-yan/PaddleOCR-OpenVINO-CSharp/releases/download/dict/ka_dict.txt";
            }
            else if (type == OCRDictsType.korean_dict)
            {
                url = "https://github.com/guojin-yan/PaddleOCR-OpenVINO-CSharp/releases/download/dict/korean_dict.txt";
            }
            else if (type == OCRDictsType.latin_dict)
            {
                url = "https://github.com/guojin-yan/PaddleOCR-OpenVINO-CSharp/releases/download/dict/latin_dict.txt";
            }
            else if (type == OCRDictsType.ta_dict)
            {
                url = "https://github.com/guojin-yan/PaddleOCR-OpenVINO-CSharp/releases/download/dict/ta_dict.txt";
            }
            else if (type == OCRDictsType.te_dict)
            {
                url = "https://github.com/guojin-yan/PaddleOCR-OpenVINO-CSharp/releases/download/dict/te_dict.txt";
            }
            else if (type == OCRDictsType.table_structure_dict)
            {
                url = "https://github.com/guojin-yan/PaddleOCR-OpenVINO-CSharp/releases/download/dict/table_structure_dict.txt";
            }
            else if (type == OCRDictsType.table_structure_dict_ch)
            {
                url = "https://github.com/guojin-yan/PaddleOCR-OpenVINO-CSharp/releases/download/dict/table_structure_dict_ch.txt";
            }
            else if (type == OCRDictsType.ppocrv5_dict)
            {
                url = "https://github.com/guojin-yan/PaddleOCR-OpenVINO-CSharp/releases/download/dict/ppocrv5_dict.txt";
            }
            else
            {
                throw new Exception("Dict selection error!");
            }
            Uri uri = new Uri(url);
            string file_name = System.IO.Path.GetFileName(uri.LocalPath);
            string file_path = Path.Combine(path, file_name);
            if (!File.Exists(file_path))
                _ = Download.download_file_async(url, file_path).Result;
            return Path.Combine(path, file_name);
        }
    }
}
