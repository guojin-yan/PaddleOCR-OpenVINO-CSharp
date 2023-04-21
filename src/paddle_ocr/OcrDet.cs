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
        private Core m_core;
        private EnumDataType m_type = 0;
        private string m_input_name;
        private string m_output_name;
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
            Mat re_image = m_preprocess.ResizeImgType0(image, m_limit_type, m_limit_side_len, out ratio_h, out ratio_w);

            byte[] image_data_det = new byte[2048 * 2048 * 3];
            ulong image_size_det = new ulong();
            image_data_det = re_image.ImEncode(".bmp");
            image_size_det = Convert.ToUInt64(image_data_det.Length);
            // 将图片数据加载到模型
            m_core.load_input_data(m_input_name, image_data_det, image_size_det, 0);

            //*******************4.模型推理****************//
            // 模型推理
            m_core.infer();
            //*******************5.读取模型输出数据****************//
            int result_det_length = 640 * 640;
            float[] result_det = m_core.read_infer_result<float>(m_output_name, result_det_length);

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
            Cv2.Threshold(cbuf_map, bit_map, threshold, maxvalue, ThresholdTypes.Binary);

            List<List<List<int>>> boxes = m_post_processor.BoxesFromBitmap(pred_map, bit_map, m_det_db_box_thresh, m_det_db_unclip_ratio,
                m_det_db_score_mode);
            Mat srcimg = new Mat();
            boxes = m_post_processor.FilterTagDetRes(boxes, ratio_h, ratio_w, srcimg);
            return boxes;
        }
    }
}
