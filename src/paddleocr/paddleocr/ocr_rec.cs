using OpenCvSharp;
using OpenVinoSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace PaddleOCR
{
    using rec_opt = RuntimeOption.RecOption;
    public class OcrRec : Predictor
    {
        private int[] m_rec_image_shape;
        private long[] m_input_size;

        List<string> m_label_list;
        private int m_rec_batch_num;

        public OcrRec(string rec_model, string? device = null, string? label_path = null, bool? use_gpu = null,
            bool? is_scale = null, float[]? mean = null, float[]? scale = null, long[]? input_size = null,
            int? batch_num = null)
            : base(rec_model, device ?? rec_opt.device, mean ?? rec_opt.mean, scale ?? rec_opt.scale,
                   input_size ?? rec_opt.input_size, is_scale ?? rec_opt.is_scale, use_gpu ?? rec_opt.use_gpu)
        {
            m_label_list = PaddleOcrUtility.read_dict(label_path ?? rec_opt.label_path);
            m_label_list.Insert(0, "#");
            m_label_list.Add(" ");
            m_rec_batch_num = batch_num ?? rec_opt.batch_num;
            m_input_size = input_size ?? rec_opt.input_size;
            m_rec_image_shape = new int[] { (int)m_input_size[1], (int)m_input_size[2], (int)m_input_size[3] };
        }

        public OcrRec(OcrConfig config)
            : base(config.rec_model_path, config.rec_option.device, config.rec_option.mean, config.rec_option.scale,
                   config.rec_option.input_size, config.rec_option.is_scale, config.rec_option.use_gpu)
        {
            m_label_list = PaddleOcrUtility.read_dict(config.rec_option.label_path);
            m_label_list.Insert(0, "#");
            m_label_list.Add(" ");
            m_rec_batch_num = config.rec_option.batch_num;
            m_input_size = config.rec_option.input_size;
            m_rec_image_shape = new int[] { (int)m_input_size[1], (int)m_input_size[2], (int)m_input_size[3] };
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
        public void predict(List<Mat> img_list, List<string> rec_texts, List<float> rec_text_scores)
        {
            int img_num = img_list.Count;
            List<float> width_list = new List<float>();
            for (int i = 0; i < img_num; i++)
            {
                width_list.Add((float)(img_list[i].Cols) / img_list[i].Rows);
            }
            List<int> indices = PaddleOcrUtility.argsort(width_list);

            for (int beg_img_no = 0; beg_img_no < img_num; beg_img_no += m_rec_batch_num)
            {
                int end_img_no = Math.Min(img_num, beg_img_no + m_rec_batch_num);
                int batch_num = end_img_no - beg_img_no;
                int imgH = m_rec_image_shape[1];
                int imgW = m_rec_image_shape[2];
                float max_wh_ratio = (imgW * 1.0f) / imgH;
                for (int ino = beg_img_no; ino < end_img_no; ino++)
                {
                    int h = img_list[indices[ino]].Rows;
                    int w = img_list[indices[ino]].Cols;
                    float wh_ratio = (w * 1.0f) / h;
                    max_wh_ratio = Math.Min(max_wh_ratio, wh_ratio);
                }

                int batch_width = 0;
                List<Mat> norm_img_batch = new List<Mat>();
                for (int ino = beg_img_no; ino < end_img_no; ino++)
                {
                    Mat srcimg = new Mat();
                    img_list[indices[ino]].CopyTo(srcimg);
                    Mat resize_img = PreProcess.crnn_resize_img(srcimg, max_wh_ratio, m_rec_image_shape);
                    PreProcess.normalize(resize_img, m_mean, m_scale, m_is_scale);
                    norm_img_batch.Add(resize_img);
                    batch_width = Math.Max(resize_img.Cols, batch_width);
                }
                float[] input_data = PreProcess.permute_batch(norm_img_batch);

                float[] predict_batch = infer(input_data, new long[] { batch_num, 3, 48, batch_width });

                for (int m = 0; m < m_rec_batch_num; m++)
                {
                    if (beg_img_no + m >= img_num)
                        return;
                    string str_res = "";
                    int argmax_idx;
                    int last_index = 0;
                    float score = 0.0f;
                    int count = 0;
                    float max_value = 0.0f;

                    for (int n = 0; n < (int)Math.Round((double)(predict_batch.Length / 6625)); n++)
                    {
                        float[] res = new float[6625];
                        Array.Copy(predict_batch, 6625 * n, res, 0, 6625);
                        // get idx and score
                        argmax_idx = (int)(PaddleOcrUtility.argmax(res, out score));
                        // get score
               

                        if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index)))
                        {
                            score += max_value;
                            count += 1;
                            str_res += m_label_list[argmax_idx];
                        }
                        last_index = argmax_idx;
                    }
                    //score /= count;
                    if (score == 0.0f)
                    {
                        continue;
                    }
                    rec_texts[indices[beg_img_no + m]] = str_res;
                    rec_text_scores[indices[beg_img_no + m]] = score;
                }

            }
        }
    }
}


