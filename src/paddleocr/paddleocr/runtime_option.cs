using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PaddleOCR
{
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
            public static string label_path = "./../../../../../dict/ppocr_keys_v1.txt";
            public static float[] mean = new float[] { 0.5f, 0.5f, 0.5f };
            public static float[] scale = new float[] { 1 / 0.5f, 1 / 0.5f, 1 / 0.5f };
            public static long[] input_size = new long[] { 1, 3, 48, 320 };
            public static bool is_scale = true;
            public static bool use_gpu = false;
            public static int batch_num = 1;
        }

    }
}
