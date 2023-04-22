using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using paddleocr;

namespace test_ocr
{
    internal class test_ocr
    {
        public void test()
        {
            string det_path = @".\..\..\..\..\..\model\ppocr_model_v3\det_onnx\model.onnx";
            string cls_path = @".\..\..\..\..\..\model\\ppocr_model_v3\cls_onnx\model.onnx";
            //string cls_path = @"E:\Text_Model\PP-OCR\ppocr_model_v3\cls_paddle\inference.pdmodel";
            string rec_path = @".\..\..\..\..\..\model\\ppocr_model_v3\rec_onnx\model.onnx";
            // 模型路径
            Dictionary<string, string> model_path = new Dictionary<string, string>();
            model_path.Add("det_model", det_path);
            model_path.Add("cls_model", cls_path);
            model_path.Add("rec_model", rec_path);

            PaddleOCR paddleOCR = new PaddleOCR(model_path);

            string image_path = @".\..\..\..\..\..\image\demo_1.jpg";
            Mat image = Cv2.ImRead(image_path);
            List<OCRPredictResult> ocr_results = paddleOCR.predict(image);
            Utility.print_result(ocr_results);
            string save_path = Path.GetDirectoryName(image_path) +"\\" + Path.GetFileNameWithoutExtension(image_path) + "_reault.jpg";
            Utility.VisualizeBboxes(image.Clone(), ocr_results, save_path);
        }   
    }
}

