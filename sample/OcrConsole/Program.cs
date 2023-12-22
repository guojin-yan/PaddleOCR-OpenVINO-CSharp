using OpenCvSharp;
using PaddleOCR;

namespace OcrConsole
{
    internal class Program
    {
        static void Main(string[] args)
        {
            OcrConfig config = new OcrConfig();
            config.det_model_path = "./../../../../../model/paddle/ch_PP-OCRv4_det_infer/inference.pdmodel";
            config.cls_model_path = "./../../../../../model/paddle/ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel";
            config .rec_model_path = "./../../../../../model/paddle/ch_PP-OCRv3_rec_infer/inference.pdmodel";

            OCRPredictor ocr = new OCRPredictor(config);

            Mat image = Cv2.ImRead("./../../../../../image/demo_3.jpg");
            List<OCRPredictResult> ocr_result = ocr.ocr(image, true, true, true);
            PaddleOcrUtility.print_result(ocr_result);
            Mat new_image = PaddleOcrUtility.visualize_bboxes(image, ocr_result);
            Cv2.ImShow("result", new_image);
            Cv2.WaitKey(0);
        }
    }
}
