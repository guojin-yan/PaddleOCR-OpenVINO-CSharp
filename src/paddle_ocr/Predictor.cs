using OpenCvSharp;
using OpenVinoSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace paddleocr
{
    public class Predictor
    {
        protected Core m_core;
        protected EnumDataType m_type = 0;
        protected string m_input_name;
        protected string m_output_name;

        protected float[] m_mean = new float[3];
        protected float[] m_scale = new float[3];

        protected float[] infer(Mat img, int result_length) {
            byte[] image_data_det = img.ImEncode(".bmp");
            ulong image_size_det = Convert.ToUInt64(image_data_det.Length);
            // 将图片数据加载到模型
            m_core.load_input_data(m_input_name, image_data_det, image_size_det, (int)m_type, m_mean, m_scale);

            //*******************4.模型推理****************//
            // 模型推理
            m_core.infer();
            float[] result = m_core.read_infer_result<float>(m_output_name, result_length);
            return result;
        }
    }
}