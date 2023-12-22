using OpenCvSharp;
using OpenVinoSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PaddleOCR
{
    using str_opt = RuntimeOption.StruLayRecOption;
    public class StruLayRec : Predictor
    {
        public float m_thresh;
        private int m_batch_num = 1;
        private long[] m_input_size;
        double m_score_threshold;
        double m_nms_threshold;
        List<int> m_fpn_stride;

        PicodetPostProcessor post_processor_;
        public StruLayRec(string layout_model, string? device = null, string? label_path = null, bool? use_gpu = null,
            bool? is_scale = null, float[]? mean = null, float[]? scale = null, long[]? input_size = null,
            int? batch_num = null, double? score_threshold = null, double? nms_threshold = null, List<int>? fpn_stride =null)
            : base(layout_model, device ?? str_opt.device, mean ?? str_opt.mean, scale ?? str_opt.scale,
           input_size ?? str_opt.input_size, is_scale ?? str_opt.is_scale, use_gpu ?? str_opt.use_gpu)
        {
            m_batch_num = batch_num ?? str_opt.batch_num;
            m_input_size = input_size ?? str_opt.input_size;
            string label_path_ = label_path ?? str_opt.label_path;
            m_score_threshold = score_threshold ?? str_opt.score_threshold;
            m_nms_threshold = nms_threshold ?? str_opt.nms_threshold;
            m_fpn_stride = fpn_stride ?? str_opt.fpn_stride;
             post_processor_ = new PicodetPostProcessor(label_path_, m_fpn_stride, m_score_threshold, m_nms_threshold);
        }

        public StruLayRec(OcrConfig config) 
            : base(config.strulay_rec_model_path, config.strulayrec_option.device, config.strulayrec_option.mean, config.strulayrec_option.scale,
                config.strulayrec_option.input_size, config.strulayrec_option.is_scale, config.strulayrec_option.use_gpu)
        {
            m_batch_num = config.strulayrec_option.batch_num;
            m_input_size = config.strulayrec_option.input_size;
            string label_path_ = config.strulayrec_option.label_path;
            m_score_threshold = config.strulayrec_option.score_threshold;
            m_nms_threshold = config.strulayrec_option.nms_threshold;
            m_fpn_stride = config.strulayrec_option.fpn_stride;
            post_processor_ = new PicodetPostProcessor(label_path_, m_fpn_stride, m_score_threshold, m_nms_threshold);
        }

        public List<StructurePredictResult> predict(Mat img, List<StructurePredictResult> result) 
        {
            Mat srcimg = new Mat();
            img.CopyTo(srcimg);
            Mat resize_img = PreProcess.Resize(srcimg, 800, 608);
            resize_img =  PreProcess.normalize(resize_img, this.m_mean, this.m_scale, this.m_is_scale);

            float[] input = PreProcess.permute(resize_img);


            // inference.
            Tensor input_tensor = m_infer_request.get_input_tensor();
            input_tensor.set_shape(new Shape(m_input_size));
            input_tensor.set_data(input);
            m_infer_request.infer();
            // Get output tensor
            List<List<float>> out_tensor_list = new List<List<float>>();
            List<List<long>> output_shape_list = new List<List<long>>();
            ulong output_size = m_model.get_outputs_size();
            for (ulong j = 0; j < output_size; j++)
            {
                var output_tensor = m_infer_request.get_output_tensor(j);
                Shape output_shape = output_tensor.get_shape();

                output_shape_list.Add(output_shape.ToList());
                float[] out_data = output_tensor.get_data<float>((int)output_tensor.get_size());
                out_tensor_list.Add(new List<float>(out_data));
            }

            // postprocess


            List<int> bbox_num = new List<int>();
            int reg_max = 0;
            for (int i = 0; i < out_tensor_list.Count; i++)
            {
                if (i == this.post_processor_.fpn_stride_.Count)
                {
                    reg_max = (int)output_shape_list[i][2] / 4;
                    break;
                }
            }
            List<int> ori_shape = new List<int>{ srcimg.Rows, srcimg.Cols };
            List<int> resize_shape = new List<int> { resize_img.Rows, resize_img.Cols };
            this.post_processor_.Run(result, out_tensor_list, ori_shape, resize_shape,
                                      reg_max);
            bbox_num.Add(result.Count);

            return result;
        }

    }
}
