using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using paddleocr;

namespace test_ocr
{
    internal class test_ocr
    {
        public void test()
        {
            string det_path = @"E:\Text_Model\PP-OCR\ppocr_model_v3\det_onnx\model.onnx";
            string cls_path = @"E:\Text_Model\PP-OCR\ppocr_model_v3\cls_onnx\model.onnx";
            string rec_path = @"E:\Text_Model\PP-OCR\ppocr_model_v3\rec_onnx\model.onnx";
            // 模型路径
            Dictionary<string, string> model_path = new Dictionary<string, string>();
            model_path.Add("det_model", det_path);
            model_path.Add("cls_model", cls_path);
            model_path.Add("rec_model", rec_path);

            PaddleOCR paddleOCR = new PaddleOCR(model_path);
        }

        
    }
}
