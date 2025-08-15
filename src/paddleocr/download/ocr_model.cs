using System;
using System.Threading.Tasks;
using static OpenVinoSharp.Extensions.model.PaddleOCR.OCRClsModels;
using static OpenVinoSharp.Extensions.model.PaddleOCR.OCRDetModels;
using static OpenVinoSharp.Extensions.model.PaddleOCR.OCRDicts;
using static OpenVinoSharp.Extensions.model.PaddleOCR.OCRRecModels;
namespace OpenVinoSharp.Extensions.model.PaddleOCR
{
    public class OcrModel
    {
        public string det_model_path = null;
        public string cls_model_path = null;
        public string rec_model_path = null;
        public string dict_path = null;

        public OcrModel() { }

        public OcrModel(string det_model_path, string cls_model_path, string rec_model_path, string dict_path)
        {
            this.det_model_path = det_model_path;
            this.cls_model_path = cls_model_path;
            this.rec_model_path = rec_model_path;
            this.dict_path = dict_path;
        }

        public static async Task<OcrModel> GetOnlineOcrModel(Language language = Language.ch_PP_OCRv4,
            bool det = true, bool cls = true, bool rec = true)
        {
            OcrModel model = new OcrModel();

            if (language == Language.ch_PP_OCRv4)
            {
                if (det) model.det_model_path = await OCRDetModels.Get(OCRDetModelsType.ch_PP_OCRv4_det);
                if (cls) model.cls_model_path = await OCRClsModels.Get(OCRClsModelsType.ch_ppocr_mobile_v2_cls);
                if (rec) model.rec_model_path = await OCRRecModels.Get(OCRRecModelsType.ch_PP_OCRv4_rec);
                model.dict_path = await OCRDicts.Get(OCRDictsType.ppocr_keys_v1);
            }
            else if (language == Language.ch_PP_OCRv4_server)
            {
                if (det) model.det_model_path = await OCRDetModels.Get(OCRDetModelsType.ch_PP_OCRv4_server_det);
                if (cls) model.cls_model_path = await OCRClsModels.Get(OCRClsModelsType.ch_ppocr_mobile_v2_cls);
                if (rec) model.rec_model_path = await OCRRecModels.Get(OCRRecModelsType.ch_PP_OCRv4_server_rec);
                model.dict_path = await OCRDicts.Get(OCRDictsType.ppocr_keys_v1);
            }
            else if (language == Language.ch_PP_OCRv3_slim)
            {
                if (det) model.det_model_path = await OCRDetModels.Get(OCRDetModelsType.ch_PP_OCRv3_det_slim);
                if (cls) model.cls_model_path = await OCRClsModels.Get(OCRClsModelsType.ch_ppocr_mobile_v2_cls);
                if (rec) model.rec_model_path = await OCRRecModels.Get(OCRRecModelsType.ch_PP_OCRv3_rec_slim);
                model.dict_path = await OCRDicts.Get(OCRDictsType.ppocr_keys_v1);
            }
            else if (language == Language.ch_PP_OCRv3)
            {
                if (det) model.det_model_path = await OCRDetModels.Get(OCRDetModelsType.ch_PP_OCRv3_det);
                if (cls) model.cls_model_path = await OCRClsModels.Get(OCRClsModelsType.ch_ppocr_mobile_v2_cls);
                if (rec) model.rec_model_path = await OCRRecModels.Get(OCRRecModelsType.ch_PP_OCRv3_rec);
                model.dict_path = await OCRDicts.Get(OCRDictsType.ppocr_keys_v1);
            }
            else if (language == Language.ch_PP_OCRv2_slim)
            {
                if (det) model.det_model_path = await OCRDetModels.Get(OCRDetModelsType.ch_PP_OCRv2_det_slim);
                if (cls) model.cls_model_path = await OCRClsModels.Get(OCRClsModelsType.ch_ppocr_mobile_v2_cls);
                if (rec) model.rec_model_path = await OCRRecModels.Get(OCRRecModelsType.ch_PP_OCRv2_rec_slim);
                model.dict_path = await OCRDicts.Get(OCRDictsType.ppocr_keys_v1);
            }
            else if (language == Language.ch_PP_OCRv2)
            {
                if (det) model.det_model_path = await OCRDetModels.Get(OCRDetModelsType.ch_PP_OCRv2_det);
                if (cls) model.cls_model_path = await OCRClsModels.Get(OCRClsModelsType.ch_ppocr_mobile_v2_cls);
                if (rec) model.rec_model_path = await OCRRecModels.Get(OCRRecModelsType.ch_PP_OCRv2_rec);
                model.dict_path = await OCRDicts.Get(OCRDictsType.ppocr_keys_v1);
            }
            else if (language == Language.ch_ppocr_mobile_slim_v2)
            {
                if (det) model.det_model_path = await OCRDetModels.Get(OCRDetModelsType.ch_ppocr_mobile_slim_v2_det);
                if (cls) model.cls_model_path = await OCRClsModels.Get(OCRClsModelsType.ch_ppocr_mobile_v2_cls);
                if (rec) model.rec_model_path = await OCRRecModels.Get(OCRRecModelsType.ch_ppocr_mobile_slim_v2_rec);
                model.dict_path = await OCRDicts.Get(OCRDictsType.ppocr_keys_v1);
            }
            else if (language == Language.ch_ppocr_mobile_v2)
            {
                if (det) model.det_model_path = await OCRDetModels.Get(OCRDetModelsType.ch_ppocr_mobile_v2_det);
                if (cls) model.cls_model_path = await OCRClsModels.Get(OCRClsModelsType.ch_ppocr_mobile_v2_cls);
                if (rec) model.rec_model_path = await OCRRecModels.Get(OCRRecModelsType.ch_ppocr_mobile_v2_rec);
                model.dict_path = await OCRDicts.Get(OCRDictsType.ppocr_keys_v1);
            }
            else if (language == Language.ch_ppocr_server_v2)
            {
                if (det) model.det_model_path = await OCRDetModels.Get(OCRDetModelsType.ch_PP_OCRv4_det);
                if (cls) model.cls_model_path = await OCRClsModels.Get(OCRClsModelsType.ch_ppocr_mobile_v2_cls);
                if (rec) model.rec_model_path = await OCRRecModels.Get(OCRRecModelsType.ch_ppocr_server_v2_rec);
                model.dict_path = await OCRDicts.Get(OCRDictsType.ppocr_keys_v1);
            }
            else if (language == Language.en_PP_OCRv4)
            {
                if (det) model.det_model_path = await OCRDetModels.Get(OCRDetModelsType.ch_PP_OCRv4_det);
                if (cls) model.cls_model_path = await OCRClsModels.Get(OCRClsModelsType.ch_ppocr_mobile_v2_cls);
                if (rec) model.rec_model_path = await OCRRecModels.Get(OCRRecModelsType.en_PP_OCRv4_rec);
                model.dict_path = await OCRDicts.Get(OCRDictsType.en_dict);
            }
            else if (language == Language.en_PP_OCRv3_slim)
            {
                if (det) model.det_model_path = await OCRDetModels.Get(OCRDetModelsType.ch_PP_OCRv4_det);
                if (cls) model.cls_model_path = await OCRClsModels.Get(OCRClsModelsType.ch_ppocr_mobile_v2_cls);
                if (rec) model.rec_model_path = await OCRRecModels.Get(OCRRecModelsType.en_PP_OCRv3_rec_slim);
                model.dict_path = await OCRDicts.Get(OCRDictsType.en_dict);
            }
            else if (language == Language.en_PP_OCRv3)
            {
                if (det) model.det_model_path = await OCRDetModels.Get(OCRDetModelsType.ch_PP_OCRv4_det);
                if (cls) model.cls_model_path = await OCRClsModels.Get(OCRClsModelsType.ch_ppocr_mobile_v2_cls);
                if (rec) model.rec_model_path = await OCRRecModels.Get(OCRRecModelsType.en_PP_OCRv3_rec);
                model.dict_path = await OCRDicts.Get(OCRDictsType.en_dict);
            }
            else if (language == Language.en_number_mobile_slim_v2)
            {
                if (det) model.det_model_path = await OCRDetModels.Get(OCRDetModelsType.ch_PP_OCRv4_det);
                if (cls) model.cls_model_path = await OCRClsModels.Get(OCRClsModelsType.ch_ppocr_mobile_v2_cls);
                if (rec) model.rec_model_path = await OCRRecModels.Get(OCRRecModelsType.en_number_mobile_slim_v2_rec);
                model.dict_path = await OCRDicts.Get(OCRDictsType.en_dict);
            }
            else if (language == Language.en_number_mobile_v2)
            {
                if (det) model.det_model_path = await OCRDetModels.Get(OCRDetModelsType.ch_PP_OCRv4_det);
                if (cls) model.cls_model_path = await OCRClsModels.Get(OCRClsModelsType.ch_ppocr_mobile_v2_cls);
                if (rec) model.rec_model_path = await OCRRecModels.Get(OCRRecModelsType.en_number_mobile_v2_rec);
                model.dict_path = await OCRDicts.Get(OCRDictsType.en_dict);
            }
            else if (language == Language.korean_PP_OCRv3)
            {
                if (det) model.det_model_path = await OCRDetModels.Get(OCRDetModelsType.ch_PP_OCRv4_det);
                if (cls) model.cls_model_path = await OCRClsModels.Get(OCRClsModelsType.ch_ppocr_mobile_v2_cls);
                if (rec) model.rec_model_path = await OCRRecModels.Get(OCRRecModelsType.korean_PP_OCRv3_rec);
                model.dict_path = await OCRDicts.Get(OCRDictsType.korean_dict);
            }
            else if (language == Language.japan_PP_OCRv3)
            {
                if (det) model.det_model_path = await OCRDetModels.Get(OCRDetModelsType.ch_PP_OCRv4_det);
                if (cls) model.cls_model_path = await OCRClsModels.Get(OCRClsModelsType.ch_ppocr_mobile_v2_cls);
                if (rec) model.rec_model_path = await OCRRecModels.Get(OCRRecModelsType.japan_PP_OCRv3_rec);
                model.dict_path = await OCRDicts.Get(OCRDictsType.japan_dict);
            }
            else if (language == Language.chinese_cht_PP_OCRv3)
            {
                if (det) model.det_model_path = await OCRDetModels.Get(OCRDetModelsType.ch_PP_OCRv4_det);
                if (cls) model.cls_model_path = await OCRClsModels.Get(OCRClsModelsType.ch_ppocr_mobile_v2_cls);
                if (rec) model.rec_model_path = await OCRRecModels.Get(OCRRecModelsType.chinese_cht_PP_OCRv3_rec);
                model.dict_path = await OCRDicts.Get(OCRDictsType.chinese_cht_dict);
            }
            else if (language == Language.te_PP_OCRv3)
            {
                if (det) model.det_model_path = await OCRDetModels.Get(OCRDetModelsType.ch_PP_OCRv4_det);
                if (cls) model.cls_model_path = await OCRClsModels.Get(OCRClsModelsType.ch_ppocr_mobile_v2_cls);
                if (rec) model.rec_model_path = await OCRRecModels.Get(OCRRecModelsType.te_PP_OCRv3_rec);
                model.dict_path = await OCRDicts.Get(OCRDictsType.te_dict);
            }
            else if (language == Language.ka_PP_OCRv3)
            {
                if (det) model.det_model_path = await OCRDetModels.Get(OCRDetModelsType.ch_PP_OCRv4_det);
                if (cls) model.cls_model_path = await OCRClsModels.Get(OCRClsModelsType.ch_ppocr_mobile_v2_cls);
                if (rec) model.rec_model_path = await OCRRecModels.Get(OCRRecModelsType.ka_PP_OCRv3_rec);
                model.dict_path = await OCRDicts.Get(OCRDictsType.ka_dict);
            }
            else if (language == Language.ta_PP_OCRv3)
            {
                if (det) model.det_model_path = await OCRDetModels.Get(OCRDetModelsType.ch_PP_OCRv4_det);
                if (cls) model.cls_model_path = await OCRClsModels.Get(OCRClsModelsType.ch_ppocr_mobile_v2_cls);
                if (rec) model.rec_model_path = await OCRRecModels.Get(OCRRecModelsType.ta_PP_OCRv3_rec);
                model.dict_path = await OCRDicts.Get(OCRDictsType.ta_dict);
            }
            else if (language == Language.latin_PP_OCRv3)
            {
                if (det) model.det_model_path = await OCRDetModels.Get(OCRDetModelsType.ch_PP_OCRv4_det);
                if (cls) model.cls_model_path = await OCRClsModels.Get(OCRClsModelsType.ch_ppocr_mobile_v2_cls);
                if (rec) model.rec_model_path = await OCRRecModels.Get(OCRRecModelsType.latin_PP_OCRv3_rec);
                model.dict_path = await OCRDicts.Get(OCRDictsType.latin_dict);
            }
            else if (language == Language.arabic_PP_OCRv3)
            {
                if (det) model.det_model_path = await OCRDetModels.Get(OCRDetModelsType.ch_PP_OCRv4_det);
                if (cls) model.cls_model_path = await OCRClsModels.Get(OCRClsModelsType.ch_ppocr_mobile_v2_cls);
                if (rec) model.rec_model_path = await OCRRecModels.Get(OCRRecModelsType.arabic_PP_OCRv3_rec);
                model.dict_path = await OCRDicts.Get(OCRDictsType.arabic_dict);
            }
            else if (language == Language.cyrillic_PP_OCRv3)
            {
                if (det) model.det_model_path = await OCRDetModels.Get(OCRDetModelsType.ch_PP_OCRv4_det);
                if (cls) model.cls_model_path = await OCRClsModels.Get(OCRClsModelsType.ch_ppocr_mobile_v2_cls);
                if (rec) model.rec_model_path = await OCRRecModels.Get(OCRRecModelsType.cyrillic_PP_OCRv3_rec);
                model.dict_path = await OCRDicts.Get(OCRDictsType.cyrillic_dict);
            }
            else if (language == Language.devanagari_PP_OCRv3)
            {
                ; if (det) model.det_model_path = await OCRDetModels.Get(OCRDetModelsType.ch_PP_OCRv4_det);
                if (cls) model.cls_model_path = await OCRClsModels.Get(OCRClsModelsType.ch_ppocr_mobile_v2_cls);
                if (rec) model.rec_model_path = await OCRRecModels.Get(OCRRecModelsType.devanagari_PP_OCRv3_rec);
                model.dict_path = await OCRDicts.Get(OCRDictsType.devanagari_dict);
            }

            else if (language == Language.PP_OCRv5_server)
            {
                if (det) model.det_model_path = await OCRDetModels.Get(OCRDetModelsType.PP_OCRv5_server_det);
                if (cls) model.cls_model_path = await OCRClsModels.Get(OCRClsModelsType.PP_OCRv5_server_cls);
                if (rec) model.rec_model_path = await OCRRecModels.Get(OCRRecModelsType.PP_OCRv5_server_rec);
                model.dict_path = await OCRDicts.Get(OCRDictsType.ppocrv5_dict);
            }
            else if (language == Language.PP_OCRv5_mobile)
            {
                ; if (det) model.det_model_path = await OCRDetModels.Get(OCRDetModelsType.PP_OCRv5_mobile_det);
                if (cls) model.cls_model_path = await OCRClsModels.Get(OCRClsModelsType.PP_OCRv5_mobile_cls);
                if (rec) model.rec_model_path = await OCRRecModels.Get(OCRRecModelsType.PP_OCRv5_mobile_rec);
                model.dict_path = await OCRDicts.Get(OCRDictsType.ppocrv5_dict);
            }

            else
            {
                throw new Exception("Model selection error!");
            }

            return model;
        }
    }
    public enum Language
    {
        /// <summary>
        /// 【最新】超轻量模型，支持中英文、数字识别 
        /// </summary>
        ch_PP_OCRv4,
        /// <summary>
        /// 【最新】高精度模型，支持中英文、数字识别
        /// </summary>
        ch_PP_OCRv4_server,
        /// <summary>
        ///  slim量化版超轻量模型，支持中英文、数字识别
        /// </summary>
        ch_PP_OCRv3_slim,
        /// <summary>
        /// 原始超轻量模型，支持中英文、数字识别
        /// </summary>
        ch_PP_OCRv3,
        /// <summary>
        /// slim量化版超轻量模型，支持中英文、数字识别
        /// </summary>
        ch_PP_OCRv2_slim,
        /// <summary>
        /// 原始超轻量模型，支持中英文、数字识别
        /// </summary>
        ch_PP_OCRv2,
        /// <summary>
        /// slim裁剪量化版超轻量模型，支持中英文、数字识别
        /// </summary>
        ch_ppocr_mobile_slim_v2,
        /// <summary>
        /// 原始超轻量模型，支持中英文、数字识别
        /// </summary>
        ch_ppocr_mobile_v2,
        /// <summary>
        /// 通用模型，支持中英文、数字识别
        /// </summary>
        ch_ppocr_server_v2,
        /// <summary>
        /// 【最新】原始超轻量模型，支持英文、数字识别
        /// </summary>
        en_PP_OCRv4,
        /// <summary>
        /// slim量化版超轻量模型，支持英文、数字识别
        /// </summary>
        en_PP_OCRv3_slim,
        /// <summary>
        /// 原始超轻量模型，支持英文、数字识别
        /// </summary>
        en_PP_OCRv3,
        /// <summary>
        /// slim裁剪量化版超轻量模型，支持英文、数字识别
        /// </summary>
        en_number_mobile_slim_v2,
        /// <summary>
        /// 原始超轻量模型，支持英文、数字识别
        /// </summary>
        en_number_mobile_v2,
        /// <summary>
        /// 韩文识别 ppocr/utils/dict/korean_dict.txt
        /// </summary>
        korean_PP_OCRv3,
        /// <summary>
        /// 日文识别 ppocr/utils/dict/japan_dict.txt
        /// </summary>
        japan_PP_OCRv3,
        /// <summary>
        /// 中文繁体识别 ppocr/utils/dict/chinese_cht_dict.txt
        /// </summary>
        chinese_cht_PP_OCRv3,
        /// <summary>
        /// 泰卢固文识别 ppocr/utils/dict/te_dict.txt
        /// </summary>
        te_PP_OCRv3,
        /// <summary>
        /// 卡纳达文识别 ppocr/utils/dict/ka_dict.txt
        /// </summary>
        ka_PP_OCRv3,
        /// <summary>
        /// 泰米尔文识别 ppocr/utils/dict/ta_dict.txt
        /// </summary>
        ta_PP_OCRv3,
        /// <summary>
        /// 拉丁文识别 ppocr/utils/dict/latin_dict.txt
        /// </summary>
        latin_PP_OCRv3,
        /// <summary>
        /// 阿拉伯字母 ppocr/utils/dict/arabic_dict.txt
        /// </summary>
        arabic_PP_OCRv3,
        /// <summary>
        /// 斯拉夫字母 ppocr/utils/dict/cyrillic_dict.txt
        /// </summary>
        cyrillic_PP_OCRv3,
        /// <summary>
        /// 梵文字母 ppocr/utils/dict/devanagari_dict.txt
        /// </summary>
        devanagari_PP_OCRv3,
        /// <summary>
        /// PP_OCRv5 文字识别模型
        /// </summary>
        PP_OCRv5_server,
        /// <summary>
        /// PP_OCRv5 文字识别轻量化模型
        /// </summary>
        PP_OCRv5_mobile,
    }

}
