using OpenCvSharp;
using OpenVinoSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace paddleocr
{
    public class OcrCls : Predictor
    {
        private Core m_core;
        private EnumDataType m_type = 0;
        private string m_input_name;
        private string m_output_name;
        public float m_cls_thresh = 0.9f;

        private float[] m_mean = new float[3] { 0.5f * 255, 0.5f * 255, 0.5f * 255 };
        private float[] m_scale = new float[3] { 0.5f * 255, 0.5f * 255, 0.5f * 255 };

        private PreProcess m_preprocess = new PreProcess();


        public OcrCls(string cls_model, string device, string input_name, string output_name,
            ulong[] input_size_det, EnumDataType type, double cls_thresh=0.9)
        {
            m_core = new Core(cls_model, device);
            m_type = type;
            m_input_name = input_name;
            m_output_name = output_name;
            m_cls_thresh = (float)cls_thresh;
            // 设置模型节点形状
            m_core.set_input_sharp(m_input_name, input_size_det);
        }

        public void predict(List<Mat> img_list, List<int> lables, List<float> scores) 
        {

            

            int img_num = img_list.Count;
            List<int> cls_image_shape = new List<int> { 3, 48, 192 };
            for (int n = 0; n < img_num; n++) 
            {
                Mat resize_img = m_preprocess.ClsResizeImg(img_list[n], cls_image_shape);

                if (resize_img.Cols < cls_image_shape[2])
                {
                    Cv2.CopyMakeBorder(resize_img, resize_img, 0, 0, 0, cls_image_shape[2] - resize_img.Cols,
                        BorderTypes.Constant, new Scalar(0, 0, 0));
                }

                byte[] image_data_det = resize_img.ImEncode(".bmp");
                ulong image_size_det = Convert.ToUInt64(image_data_det.Length);
                // 将图片数据加载到模型
                m_core.load_input_data(m_input_name, image_data_det, image_size_det, (int)m_type, m_mean, m_scale);

                //*******************4.模型推理****************//
                // 模型推理
                m_core.infer();
                int result_cls_length = 2;
                float[] result_cls = m_core.read_infer_result<float>(m_output_name, result_cls_length);
                int lable = result_cls[0] > result_cls[1] ? 0 : 1;
                lables.Add(lable);
                scores.Add(result_cls[lable]);
                //Console.WriteLine("({0}, {1})", result_cls[0], result_cls[1]);

            }
        }
    }
}
