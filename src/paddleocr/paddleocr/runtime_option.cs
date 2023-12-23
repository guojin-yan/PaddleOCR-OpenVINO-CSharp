using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PaddleOCR
{
    public class OcrConfig 
    {
        // ocr det
        public class DetOption
        {
            public string device = "CPU";
            public float[] mean = new float[] { 0.485f, 0.456f, 0.406f };
            public float[] scale = new float[] { 1 / 0.229f, 1 / 0.224f, 1 / 0.225f };
            public long[] input_size = new long[] { 1, 3, 960, 960 };
            public bool is_scale = true;
            public bool use_gpu = false;
            public float det_db_thresh = 0.3f;
            public float det_db_box_thresh = 0.5f;
            public int limit_side_len = 960;
            public string limit_type = "max";
            public string db_score_mode = "slow";
            public float db_unclip_ratio = 2.0f;
        }
        public DetOption det_option;
        // Ocr cls
        public class ClsOption
        {
            public string device = "CPU";
            public float[] mean = new float[] { 0.5f, 0.5f, 0.5f };
            public float[] scale = new float[] { 1 / 0.5f, 1 / 0.5f, 1 / 0.5f };
            public long[] input_size = new long[] { 1, 3, 48, 192 };
            public bool is_scale = true;
            public bool use_gpu = false;
            public float cls_thresh = 0.9f;
            public int batch_num = 1;
        }
        public ClsOption cls_option;
        // Ocr rec
        public class RecOption
        {
            public string device = "CPU";
            public string label_path = "dict/ppocr_keys_v1.txt";
            public float[] mean = new float[] { 0.5f, 0.5f, 0.5f };
            public float[] scale = new float[] { 1 / 0.5f, 1 / 0.5f, 1 / 0.5f };
            public long[] input_size = new long[] { 1, 3, 48, 320 };
            public bool is_scale = true;
            public bool use_gpu = false;
            public int batch_num = 1;
        }
        public RecOption rec_option;
        public class StruTabRecOption
        {
            public string device = "CPU";
            public string label_path = "dict/table_structure_dict_ch.txt";
            public float[] mean = new float[] { 0.485f, 0.456f, 0.406f };
            public float[] scale = new float[] { 1 / 0.229f, 1 / 0.224f, 1 / 0.225f };
            public long[] input_size = new long[] { 1, 3, 488, 488 };
            public bool is_scale = true;
            public bool use_gpu = false;
            public int batch_num = 1;
            public float thresh = 0.9f;
            public bool merge_no_span_structure = true;
        }
        public StruTabRecOption strutabrec_option;

        public class StruLayRecOption
        {
            public string device = "CPU";
            public string label_path = "dict/layout_cdla_dict.txt";
            public float[] mean = new float[] { 0.485f, 0.456f, 0.406f };
            public float[] scale = new float[] { 1 / 0.229f, 1 / 0.224f, 1 / 0.225f };
            public long[] input_size = new long[] { 1, 3, 800, 608 };
            public bool is_scale = true;
            public bool use_gpu = false;
            public int batch_num = 1;
            public double score_threshold = 0.4;
            public double nms_threshold = 0.5;
            public List<int> fpn_stride = new List<int>(new int[] { 8, 16, 32, 64 });
        }
        public StruLayRecOption strulayrec_option;

        public string det_model_path = null;
        public string cls_model_path = null;
        public string rec_model_path = null;
        public string table_rec_model_path = null;
        public string strulay_rec_model_path = null;

        public OcrConfig() 
        {
            det_option = new DetOption();
            cls_option = new ClsOption();
            rec_option = new RecOption();
            strutabrec_option = new StruTabRecOption();
            strulayrec_option = new StruLayRecOption();
        }

        public void set_det_option(DetOption op) => det_option = op;
        public void set_cls_option(ClsOption op) => cls_option = op;
        public void set_rec_option(RecOption op) => rec_option = op;
        public void set_table_option(StruTabRecOption op) => strutabrec_option = op;
        public void set_layout_option(StruLayRecOption op) => strulayrec_option = op;

        public void set_dict_path(string path) 
        {
            rec_option.label_path = Path.Combine(path, rec_option.label_path);
            strulayrec_option.label_path = Path.Combine(path, strulayrec_option.label_path);
            strutabrec_option.label_path = Path.Combine(path, strutabrec_option.label_path);
        }
    };

    public static class RuntimeOption
    {
        // ocr det
        public static class DetOption
        {
            public static string device = "CPU";
            public static float[] mean = new float[] { 0.485f, 0.456f, 0.406f };
            public static float[] scale = new float[] { 1 / 0.229f, 1 / 0.224f, 1 / 0.225f };
            public static long[] input_size = new long[] { 1, 3, 960, 960 };
            public static bool is_scale = true;
            public static bool use_gpu = false;
            public static float det_db_thresh = 0.3f;
            public static float det_db_box_thresh = 0.5f;
            public static int limit_side_len = 960;
            public static string limit_type = "max";
            public static string db_score_mode = "slow";
            public static float db_unclip_ratio = 2.0f;
        }

        // Ocr cls
        public static class ClsOption
        {
            public static string device = "CPU";
            public static float[] mean = new float[] { 0.5f, 0.5f, 0.5f };
            public static float[] scale = new float[] { 1 / 0.5f, 1 / 0.5f, 1 / 0.5f };
            public static long[] input_size = new long[] { 1, 3, 48, 192 };
            public static bool is_scale = true;
            public static bool use_gpu = false;
            public static float cls_thresh = 0.9f;
            public static int batch_num = 1;
        }

        // Ocr rec
        public static class RecOption
        {
            public static string device = "CPU";
            public static string label_path = "dict/ppocr_keys_v1.txt";
            public static float[] mean = new float[] { 0.5f, 0.5f, 0.5f };
            public static float[] scale = new float[] { 1 / 0.5f, 1 / 0.5f, 1 / 0.5f };
            public static long[] input_size = new long[] { 1, 3, 48, 320 };
            public static bool is_scale = true;
            public static bool use_gpu = false;
            public static int batch_num = 1;
        }
        public static class StruTabRecOption
        {
            public static string device = "CPU";
            public static string label_path = "dict/table_structure_dict_ch.txt";
            public static float[] mean = new float[] { 0.485f, 0.456f, 0.406f };
            public static float[] scale = new float[] { 1 / 0.229f, 1 / 0.224f, 1 / 0.225f };
            public static long[] input_size = new long[] { 1, 3, 488, 488 };
            public static bool is_scale = true;
            public static bool use_gpu = false;
            public static int batch_num = 1;
            public static float thresh = 0.9f;
        }

        public static class StruLayRecOption
        {
            public static string device = "CPU";
            public static string label_path = "dict/layout_cdla_dict.txt";
            public static float[] mean = new float[] { 0.485f, 0.456f, 0.406f };
            public static float[] scale = new float[] { 1 / 0.229f, 1 / 0.224f, 1 / 0.225f };
            public static long[] input_size = new long[] { 1, 3, 800, 608 };
            public static bool is_scale = true;
            public static bool use_gpu = false;
            public static int batch_num = 1;
            public static double score_threshold = 0.4; 
            public static double nms_threshold = 0.5;
            public static  List<int> fpn_stride = new List<int>(new int[] { 8, 16, 32, 64 });

        }

    }
}
