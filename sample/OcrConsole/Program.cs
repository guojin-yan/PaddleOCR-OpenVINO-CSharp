using OpenCvSharp;
using PaddleOCR;
using OpenVinoSharp.Extensions.Utility;
using System.Reflection.Metadata.Ecma335;
namespace OcrConsole
{
    internal class Program
    {
        static void Main(string[] args)
        {
            OcrConfig config = new OcrConfig();
           
            string image_path = "";
            if (args.Length == 1) 
            {
                string base_path = Path.GetFullPath(args[0]);
                if (!Utility.chech_path(base_path))
                {
                    return;
                }
                config.det_model_path = Path.Combine(base_path , "model/paddle/ch_PP-OCRv4_det_infer/inference.pdmodel");
                config.cls_model_path = Path.Combine(base_path , "model/paddle/ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel");
                config.rec_model_path = Path.Combine(base_path , "model/paddle/ch_PP-OCRv3_rec_infer/inference.pdmodel");
                image_path = Path.Combine(base_path + "/image/demo_1.jpg");
                config.set_dict_path(base_path);
            } 
            else if (args.Length == 5)
            {
                image_path = args[0];
                config.det_model_path = args[1];
                config.cls_model_path = args[2];
                config.rec_model_path = args[3];
                config.set_dict_path(args[4]);
            }
            else 
            {
                Console.WriteLine("Please add command function parameters, such as:");
                Console.WriteLine(" >dotnet run <dir_path>");
                Console.WriteLine(" >dotnet run <image_path> <det_model_path> <cls_model_path> <rec_model_path> <dict_path_path>");
                return;
            }

            if (Utility.chech_file(config.det_model_path) || Utility.chech_file(config.cls_model_path) || Utility.chech_file(config.rec_model_path)) { }
            else { return; }

            if (!Utility.chech_file(config.rec_option.label_path) )  { return; }

            if (!Utility.chech_file(image_path))
            {
                return;
            }


            OCRPredictor ocr = new OCRPredictor(config);

            Mat image = Cv2.ImRead(image_path);
            List<OCRPredictResult> ocr_result = ocr.ocr(image, true, true, true);
            PaddleOcrUtility.print_result(ocr_result);
            Mat new_image = PaddleOcrUtility.visualize_bboxes(image, ocr_result);
            Cv2.ImShow("result", new_image);
            Cv2.WaitKey(0);
        }
    }
}
