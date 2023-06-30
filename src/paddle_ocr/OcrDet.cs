using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using OpenVinoSharp;

namespace paddleocr
{
    public class OcrDet : Predictor
    {
        private float m_det_db_thresh = 0.3f;
        private float m_det_db_box_thresh = 0.5f;
        private string m_det_db_score_mode = "slow";
        private float m_det_db_unclip_ratio = 2.0f;
        string m_limit_type = "max";
        int m_limit_side_len = 960;


        private PostProcessor m_post_processor = new PostProcessor();
        private PreProcess m_preprocess = new PreProcess();

        public OcrDet(string det_model, string device, string input_name, string output_name, 
            ulong[] input_size_det, EnumDataType type, double det_db_thresh = 0.3, double det_db_box_thresh = 0.5) 
        {
            m_core = new Core(det_model, device);
            m_type = type;
            m_input_name = input_name;
            m_output_name = output_name;
            m_det_db_thresh = (float)det_db_thresh;
            m_det_db_box_thresh = (float)det_db_box_thresh;
            // 设置模型节点形状
            m_core.set_input_sharp(m_input_name, input_size_det);
        }

        public List<List<List<int>>> predict(Mat image) 
        {
            float ratio_h;
            float ratio_w;
            Mat resize_img = m_preprocess.ResizeImgType0(image, m_limit_type, m_limit_side_len, out ratio_h, out ratio_w);

            ratio_h = (float)(640.0 / image.Cols);
            ratio_w = (float)(640.0 / image.Rows);

            int result_det_length = 640 * 640;
            float[] result_det = infer(resize_img, result_det_length);
            
            // 将模型输出转为byte格式
            byte[] result_det_byte = new byte[result_det_length];
            for (int i = 0; i < result_det_length; i++)
            {
                result_det_byte[i] = (byte)(result_det[i] * 255);
            }
            // 重构结果图像
            Mat cbuf_map = new Mat(640, 640, MatType.CV_8UC1, result_det_byte);
            Mat pred_map = new Mat(640, 640, MatType.CV_32F, result_det);

            double threshold = m_det_db_thresh * 255;
            double maxvalue = 255;
            // 图像阈值处理
            Mat bit_map = new Mat();
            //Cv2.ImShow("pred_map", pred_map);
            //Cv2.WaitKey(0);
            Cv2.Threshold(cbuf_map, bit_map, threshold, maxvalue, ThresholdTypes.Binary);
            //Cv2.ImShow("bit_map", bit_map);
            //Cv2.WaitKey(0);
  
            List<List<List<int>>> boxes = m_post_processor.BoxesFromBitmap(pred_map, bit_map, m_det_db_box_thresh, m_det_db_unclip_ratio,
                m_det_db_score_mode);
            Console.WriteLine("-------------------:" + boxes.Count);
            boxes = m_post_processor.FilterTagDetRes(boxes, ratio_h, ratio_w, image);
            Console.WriteLine("-------------------:" + boxes.Count);
            return boxes;
        }
    }
}