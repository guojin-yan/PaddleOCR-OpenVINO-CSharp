using OpenCvSharp;
using OpenVinoSharp;
using Org.BouncyCastle.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PaddleOCR
{
    using str_opt = RuntimeOption.StruTabRecOption;
    public class StruTabRec : Predictor
    {
        public float m_thresh;
        private int m_batch_num = 1;
        private long[] m_input_size;


        private TablePostProcessor m_table_post;
        public StruTabRec(string table_model, string? device = null, string? label_path = null, bool? use_gpu = null,
            bool? is_scale = null, float[]? mean = null, float[]? scale = null, long[]? input_size = null,
            int? batch_num = null, bool merge_no_span_structure = true)
            : base(table_model, device ?? str_opt.device, mean ?? str_opt.mean, scale ?? str_opt.scale,
           input_size ?? str_opt.input_size, is_scale ?? str_opt.is_scale, use_gpu ?? str_opt.use_gpu)
        {
            m_batch_num = batch_num ?? str_opt.batch_num;
            m_input_size = input_size ?? str_opt.input_size;
            string label_path_ = label_path ?? str_opt.label_path;
            m_table_post = new TablePostProcessor(label_path_, merge_no_span_structure);
        }
        public StruTabRec(OcrConfig config) 
            : base(config.table_rec_model_path, config.strutabrec_option.device, config.strutabrec_option.mean, config.strutabrec_option.scale,
                config.strutabrec_option.input_size, config.strutabrec_option.is_scale, config.strutabrec_option.use_gpu)
        {
            m_batch_num = config.strutabrec_option.batch_num;
            m_input_size = config.strutabrec_option.input_size;
            string label_path_ = config.strutabrec_option.label_path;
            m_table_post = new TablePostProcessor(label_path_, config.strutabrec_option.merge_no_span_structure);
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

        public void predict(List<Mat> img_list, List<List<string>> structure_html_tags, List<float> structure_scores, List<List<List<int>>> structure_boxes)
        {
            int img_num = img_list.Count;
            for (int beg_img_no = 0; beg_img_no < img_num;
                 beg_img_no += this.m_batch_num)
            {
                // preprocess
                int end_img_no = Math.Min(img_num, beg_img_no + this.m_batch_num);
                int batch_num = end_img_no - beg_img_no;
                List<Mat> norm_img_batch = new List<Mat>();
                List<int> width_list = new List<int>();
                List<int> height_list = new List<int>();
                for (int ino = beg_img_no; ino < end_img_no; ino++)
                {
                    Mat srcimg = new Mat();
                    img_list[ino].CopyTo(srcimg);
                    Mat resize_img = new Mat();
                    Mat pad_img = new Mat();
                    PreProcess.TableResizeImg(srcimg, resize_img, (int)this.m_input_size[2]);
                    resize_img = PreProcess.normalize(resize_img, this.m_mean, this.m_scale, this.m_is_scale);
                    pad_img = PreProcess.TablePadImg(resize_img, (int)this.m_input_size[2]);
                    norm_img_batch.Add(pad_img);
                    width_list.Add(srcimg.Cols);
                    height_list.Add(srcimg.Rows);
                }

                float[] input = PreProcess.permute_batch(norm_img_batch);

                // inference.

                Tensor input_tensor = m_infer_request.get_input_tensor();
                input_tensor.set_shape(new Shape(new long[] { batch_num, m_input_size[1], m_input_size[2], m_input_size[3] }));

                input_tensor.set_data(input);
                m_infer_request.infer();
                Tensor output_tensor1 = m_infer_request.get_output_tensor(0);
                Tensor output_tensor2 = m_infer_request.get_output_tensor(1);
                long[] shape0 = output_tensor1.get_shape().ToArray();
                long[] shape1 = output_tensor2.get_shape().ToArray();
                List<int> predict_shape0 = new List<int>(new int[] { (int)shape0[0], (int)shape0[1], (int)shape0[2] });
                List<int> predict_shape1 = new List<int>(new int[] { (int)shape1[0], (int)shape1[1], (int)shape1[2] });

                float[] loc_preds = output_tensor1.get_data<float>((int)output_tensor1.get_size());
                float[] structure_probs = output_tensor2.get_data<float>((int)output_tensor2.get_size());


                // postprocess

                List<List<string>> structure_html_tag_batch = new List<List<string>>();
                List<float> structure_score_batch = new List<float>();
                List<List<List<int>>> structure_boxes_batch = new List<List<List<int>>>();
                m_table_post.Run(new List<float>(loc_preds), new List<float>(structure_probs), structure_score_batch,predict_shape0,
                    predict_shape1, structure_html_tag_batch, structure_boxes_batch, width_list, height_list);
                for (int m = 0; m < predict_shape0[0]; m++)
                {

                    structure_html_tag_batch[m].Insert(0, "<table>");
                    structure_html_tag_batch[m].Insert(0, "<body>");
                    structure_html_tag_batch[m].Insert(0, "<html>");
                    structure_html_tag_batch[m].Add("</table>");
                    structure_html_tag_batch[m].Add("</body>");
                    structure_html_tag_batch[m].Add("</html>");
                    structure_html_tags.Add(structure_html_tag_batch[m]);
                    structure_scores.Add(structure_score_batch[m]);
                    structure_boxes.Add(structure_boxes_batch[m]);
                }

            }
        }
    }
}
