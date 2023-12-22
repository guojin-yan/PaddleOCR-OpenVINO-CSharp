using OpenCvSharp;
using OpenVinoSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PaddleOCR
{
    using cls_opt = RuntimeOption.ClsOption;
    public class OcrCls : Predictor
    {
        public float m_cls_thresh;
        private int m_cls_batch_num;
        private long[] m_input_size;
        public OcrCls(string cls_model, string? device = null, bool? use_gpu = null, bool? is_scale = null,
            float[]? mean = null, float[]? scale = null, long[]? input_size = null, float? cls_thresh = null,
            int? batch_num = null)
             : base(cls_model, device ?? cls_opt.device, mean ?? cls_opt.mean, scale ?? cls_opt.scale,
                   input_size ?? cls_opt.input_size, is_scale ?? cls_opt.is_scale, use_gpu ?? cls_opt.use_gpu)
        {
            m_cls_batch_num = batch_num ?? cls_opt.batch_num;
            m_cls_thresh = cls_thresh ?? cls_opt.cls_thresh;
            m_input_size = input_size ?? cls_opt.input_size;
        }

        public OcrCls(OcrConfig config)
            : base(config.cls_model_path, config.cls_option.device, config.cls_option.mean, config.cls_option.scale,
                   config.cls_option.input_size, config.cls_option.is_scale, config.cls_option.use_gpu)
        {
            m_cls_batch_num = config.cls_option.batch_num;
            m_cls_thresh = config.cls_option.cls_thresh;
            m_input_size = config.cls_option.input_size;
        }
        public void predict(List<Mat> img_list, List<int> lables, List<float> scores)
        {
            int img_num = img_list.Count;
            List<int> cls_image_shape = new List<int> { 3, 48, 192 };
            if (m_use_gpu)
            {
                m_cls_batch_num = 1;
            }
            for (int beg_img_no = 0; beg_img_no < img_num; beg_img_no += m_cls_batch_num)
            {
                int end_img_no = Math.Min(img_num, beg_img_no + m_cls_batch_num);

                int batch_num = end_img_no - beg_img_no;
                m_input_size[0] = batch_num;
                // preprocess
                List<Mat> norm_img_batch = new List<Mat>();
                for (int ino = beg_img_no; ino < end_img_no; ino++)
                {
                    Mat srcimg = new Mat();
                    img_list[ino].CopyTo(srcimg);
                    Mat resize_img = PreProcess.cls_resize_img(srcimg, cls_image_shape);

                    PreProcess.normalize(resize_img, m_mean, m_scale, m_is_scale);
                    if (resize_img.Cols < cls_image_shape[2])
                    {
                        Cv2.CopyMakeBorder(resize_img, resize_img, 0, 0, 0, cls_image_shape[2] - resize_img.Cols,
                            BorderTypes.Constant, new Scalar(0, 0, 0));
                    }
                    norm_img_batch.Add(resize_img);
                }

                float[] input_data = PreProcess.permute_batch(norm_img_batch);

                float[] result_cls = infer(input_data, m_input_size);

                for (int batch_idx = 0; batch_idx < batch_num; batch_idx++)
                {
                    int lable = result_cls[2 * batch_idx + 0] > result_cls[2 * batch_idx + 1] ? 0 : 1;
                    lables.Add(lable);
                    scores.Add(result_cls[2 * batch_idx + lable]);
                }
                //Console.WriteLine("({0}, {1})", result_cls[0], result_cls[1]);

            }
        }
    }
}

