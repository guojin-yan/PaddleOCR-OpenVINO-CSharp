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
        public float m_cls_thresh = 0.9f;
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

            m_mean = new float[3] { 0.5f * 255, 0.5f * 255, 0.5f * 255 };
            m_scale = new float[3] { 0.5f * 255, 0.5f * 255, 0.5f * 255 };
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
                int result_cls_length = 2;
                float[] result_cls = infer(resize_img, result_cls_length);
                int lable = result_cls[0] > result_cls[1] ? 0 : 1;
                lables.Add(lable);
                scores.Add(result_cls[lable]);
                //Console.WriteLine("({0}, {1})", result_cls[0], result_cls[1]);

            }
        }
    }
}
