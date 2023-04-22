using OpenCvSharp;
using OpenVinoSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace paddleocr
{
    public class OcrRec : Predictor
    {
        private int[] m_rec_image_shape;

        List<string> m_label_list;

        private PostProcessor m_post_processor = new PostProcessor();
        private PreProcess m_preprocess = new PreProcess();

        public OcrRec(string det_model, string device, string input_name, string output_name,
            ulong[] input_size_rec, EnumDataType type, string label_path)
        {
            m_core = new Core(det_model, device);
            m_type = type;
            m_input_name = input_name;
            m_output_name = output_name;
            m_label_list = Utility.ReadDict(label_path);
            m_label_list.Insert(0, "#");
            m_label_list.Add(" ");
            m_rec_image_shape = new int[] { (int)input_size_rec[1], (int)input_size_rec[2], (int)input_size_rec[3]};
            // 设置模型节点形状
            m_core.set_input_sharp(m_input_name, input_size_rec);

            m_mean = new float[3] { 0.5f * 255, 0.5f * 255, 0.5f * 255 };
            m_scale = new float[3] { 0.5f * 255, 0.5f * 255, 0.5f * 255 };
        }

        public void predict(List<Mat> img_list, List<string> rec_texts, List<float> rec_text_scores)
        {
            int img_num = img_list.Count;
            List<float> width_list = new List<float>();
            for (int i = 0; i < img_num; i++)
            {
                width_list.Add((float)(img_list[i].Cols) / img_list[i].Rows);
            }
            List<int> indices = Utility.argsort(width_list);
            for (int n = 0; n < img_num; n++)
            {
                int imgH = m_rec_image_shape[1];
                int imgW = m_rec_image_shape[2];
                float max_wh_ratio = imgW * 1.0f / imgH;
                int h = img_list[n].Rows;
                int w = img_list[n].Cols;
                float wh_ratio = w * 1.0f / h;
                max_wh_ratio = Math.Max(max_wh_ratio, wh_ratio);
                Mat resize_img = m_preprocess.CrnnResizeImg(img_list[n].Clone(), max_wh_ratio, m_rec_image_shape);

                //Cv2.ImShow("resize_img", resize_img);
                //Cv2.WaitKey(0);

                int result_cls_length = 40 * 6625;
                float[] result_cls = infer(resize_img, result_cls_length);

                

                string str_res = "";
                int argmax_idx;
                int last_index = 0;
                float score = 0.0f;
                int count = 0;
                float max_value = 0.0f;


                for (int r = 0; r < 40; r++)
                {
                    float[] temp = new float[6625];
                    for (int j = 0; j < 6625; j++)
                    {
                        temp[j] = result_cls[r * 6625 + j];
                    }

                    argmax_idx = Utility.argmax(temp, out max_value);
                    //Console.WriteLine("{0}  {1}  {2}", argmax_idx, max_value, temp[0]);

                    if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index)))
                    {
                        score += max_value;
                        count += 1;
                        str_res += m_label_list[argmax_idx];
                    }
                    last_index = argmax_idx;
                }
                score /= count;
                rec_texts.Add(str_res);
                rec_text_scores.Add(score);
                //Console.WriteLine(str_res);
            }
        }
    }
}


