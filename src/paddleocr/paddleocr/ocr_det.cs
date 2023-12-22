using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using OpenVinoSharp;

namespace PaddleOCR
{
    using det_opt = RuntimeOption.DetOption;
    public class OcrDet : Predictor
    {
        private float m_det_db_thresh;
        private float m_det_db_box_thresh;
        private string m_det_db_score_mode;
        private float m_det_db_unclip_ratio;
        string m_limit_type;
        int m_limit_side_len;


        public OcrDet(string det_model, string? device = null, bool? use_gpu = null, bool? is_scale = null, 
            float[]? mean = null, float[]? scale = null, float? db_thresh = null, float? db_box_thresh = null, 
            long[]? input_size = null, string? db_score_mode = null, float? db_unclip_ratio = null, 
            string? limit_type = null, int? limit_side_len = null)
            : base(det_model, device ?? det_opt.device, mean ?? det_opt.mean, 
                  scale ?? det_opt.scale, input_size??det_opt.input_size, true, use_gpu??det_opt.use_gpu)
        {
            m_det_db_thresh = db_thresh ?? det_opt.det_db_thresh;
            m_det_db_box_thresh = db_box_thresh ?? det_opt.det_db_box_thresh;
            m_det_db_score_mode = db_score_mode ?? det_opt.db_score_mode;
            m_det_db_unclip_ratio = db_unclip_ratio ?? det_opt.db_unclip_ratio;
            m_limit_type = limit_type ?? det_opt.limit_type;
            m_limit_side_len = limit_side_len ?? det_opt.limit_side_len;
        }


        public OcrDet(OcrConfig config)
            : base(config.det_model_path, config.det_option.device, config.det_option.mean,
                config.det_option.scale, config.det_option.input_size, true, config.det_option.use_gpu)
        {
            m_det_db_thresh = config.det_option.det_db_thresh;
            m_det_db_box_thresh = config.det_option.det_db_box_thresh;
            m_det_db_score_mode = config.det_option.db_score_mode;
            m_det_db_unclip_ratio = config.det_option.db_unclip_ratio;
            m_limit_type = config.det_option.limit_type;
            m_limit_side_len = config.det_option.limit_side_len;
        }

        // To detect redundant calls
        private bool m_disposed_value;
        protected override void Dispose(bool disposing)
        {
            if (!m_disposed_value)
            {
                if (disposing)
                {
                }

                m_disposed_value = true;
            }
            // Call base class implementation.
            base.Dispose(disposing);
        }
        public List<List<List<int>>> predict(Mat image) 
        {
            float ratio_h;
            float ratio_w;
            Mat input_img = PreProcess.resize_imgtype0(image, m_limit_type, m_limit_side_len, out ratio_h, out ratio_w);
            input_img = PreProcess.normalize(input_img, m_mean, m_scale, m_is_scale);
            Mat cbuf_map = new Mat();
            Mat pred_map = new Mat();
            if (m_use_gpu) 
            {
                Mat max_image = Mat.Zeros(new OpenCvSharp.Size(960, 960), MatType.CV_32FC3);
                Rect roi = new Rect(0, 0, image.Cols, image.Rows);
                input_img.CopyTo(new Mat(max_image, roi));
                float[] input_data = PreProcess.permute(max_image);
                float[] result_det = infer(input_data, new long[] { 1, 3, 960, 960 });
                // 将模型输出转为byte格式
                byte[] result_det_byte = new byte[result_det.Length];
                for (int i = 0; i < result_det.Length; i++)
                {
                    result_det_byte[i] = (byte)(result_det[i] * 255);
                }
                // 重构结果图像
                Mat cbuf_map_t = new Mat(960, 960, MatType.CV_8UC1, result_det_byte);
                Mat pred_map_t = new Mat(960, 960, MatType.CV_32F, result_det);
                cbuf_map = new Mat(cbuf_map_t, roi);
                pred_map = new Mat(pred_map_t, roi);
            }
            else
            {
                float[] input_data = PreProcess.permute(input_img);
                float[] result_det = infer(input_data, new long[] { 1, 3, input_img.Rows, input_img.Cols });
                // 将模型输出转为byte格式
                byte[] result_det_byte = new byte[result_det.Length];
                for (int i = 0; i < result_det.Length; i++)
                {
                    result_det_byte[i] = (byte)(result_det[i] * 255);
                }
                // 重构结果图像
                cbuf_map = new Mat(input_img.Rows, input_img.Cols, MatType.CV_8UC1, result_det_byte);
                pred_map = new Mat(input_img.Rows, input_img.Cols, MatType.CV_32F, result_det);
            }
     
            double threshold = m_det_db_thresh * 255;
            double maxvalue = 255;
            // 图像阈值处理
            Mat bit_map = new Mat();
            //Cv2.ImShow("pred_map", pred_map);
            //Cv2.WaitKey(0);
            Cv2.Threshold(cbuf_map, bit_map, threshold, maxvalue, ThresholdTypes.Binary);
            //Cv2.ImShow("bit_map", bit_map);
            //Cv2.WaitKey(0);

            List<List<List<int>>> boxes = PostProcessor.boxes_from_bitmap(pred_map, bit_map, m_det_db_box_thresh, m_det_db_unclip_ratio,
                m_det_db_score_mode);
            Console.WriteLine("-------------------:" + boxes.Count);
            boxes = PostProcessor.filter_tag_det_res(boxes, ratio_h, ratio_w, image);
            Console.WriteLine("-------------------:" + boxes.Count);
            return boxes;
        }
    }
}