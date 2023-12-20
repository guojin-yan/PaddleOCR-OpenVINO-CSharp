using OpenCvSharp;
using PaddleOCR;
using PaddleOCR.paddleocr;
using Spire.Xls;
using System.Net.Http;

namespace test_struct
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");
            Mat image = Cv2.ImRead("./../../../../../image/demo_6.jpg");
            //Cv2.ImShow("aa", image);
            //Cv2.WaitKey(0);
            string str_model = "./../../../../../model\\ir\\ch_ppstructure_mobile_v2.0_SLANet_infer\\model.xml";
            string det_model = "./../../../../../model/paddle/ch_PP-OCRv4_det_infer/inference.pdmodel";
            string cls_model = "./../../../../../model/paddle/ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel";
            string rec_model = "./../../../../../model/paddle/ch_PP-OCRv4_rec_infer/inference.pdmodel";
            StructurePredictor predictor = new StructurePredictor(str_model, det_model, cls_model, rec_model);

            StructurePredictResult structure_result = new StructurePredictResult();
            structure_result = predictor.table(image, structure_result);
            PaddleOcrUtility.VisualizeBboxes(image, structure_result, "./examplejpg");
            Console.WriteLine(structure_result.html);
            // 指定要保存的文件路径及名称
            string filePath = "example.html";

            // 将HTML内容写入到文件中
            File.WriteAllText(filePath, structure_result.html);
            Workbook workbook = new Workbook();
            workbook.LoadFromHtml(filePath);

            //保存为Excel
            workbook.SaveToFile("example.xlsx", FileFormat.Version2013);
            Console.WriteLine("Hello, World!");
            
        }
    }
}
