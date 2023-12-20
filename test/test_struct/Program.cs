using OpenCvSharp;
using PaddleOCR;
using Spire.Xls;
using System.Net.Http;

namespace test_struct
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //test_lable();
            //test_layout();
            test_structure();
            Console.WriteLine("Hello, World!");
            
        }

        static void test_structure()
        {
            string image_path = "./../../../../../image/demo_7.jpg";
            //Cv2.ImShow("aa", image);
            //Cv2.WaitKey(0);
            string lay_model = "./../../../../../model/paddle/picodet_lcnet_x1_0_fgd_layout_cdla_infer/model.pdmodel";
            string tab_model = "./../../../../../model/ir/ch_ppstructure_mobile_v2.0_SLANet_infer/model.xml";
            string det_model = "./../../../../../model/paddle/ch_PP-OCRv4_det_infer/inference.pdmodel";
            string cls_model = "./../../../../../model/paddle/ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel";
            string rec_model = "./../../../../../model/paddle/ch_PP-OCRv4_rec_infer/inference.pdmodel";

            StructurePredictor engine = new StructurePredictor(lay_model, tab_model, det_model, rec_model, cls_model); 

            Console.WriteLine ("predict img: "+ image_path );
            Mat img = Cv2.ImRead(image_path);
            Cv2.ImShow("aa", img);
            Cv2.WaitKey(0);
            //if (img.Data != IntPtr.Zero)
            //{
            //    Console.WriteLine("[ERROR] image read failed! image path: " + image_path);
            //    return;
            //}

            List<StructurePredictResult> structure_results = engine.structure( img, true, true, true && true);

            string msg = "";

            for (int j = 0; j < structure_results.Count; j++)
            {
                msg+=( j + "\ttype: " + structure_results[j].type +", region: [");
                msg += (structure_results[j].box[0] + "," + structure_results[j].box[1] + ","  + structure_results[j].box[2] + ","
                          + structure_results[j].box[3] + "], score: "+ structure_results[j].confidence + ", res: ");

                if (structure_results[j].type == "table")
                {
                    msg+= structure_results[j].html;
                    Console.WriteLine(msg);
                }
                else
                {
                    string msg1 = "";
                    Console.WriteLine( "count of ocr result is : " + structure_results[j].text_res.Count);
                    if (structure_results[j].text_res.Count > 0)
                    {
                        Console.WriteLine("********** print ocr result " + "**********");
                        PaddleOcrUtility.print_result(structure_results[j].text_res);
                        Console.WriteLine("********** end print ocr result " + "**********");
                    }
                }
            }

        }


        static void test_layout() {
            Console.WriteLine("Hello, World!");
            Mat image = Cv2.ImRead("./../../../../../image/demo_7.jpg");
            //Cv2.ImShow("aa", image);
            //Cv2.WaitKey(0);
            string lay_model = "./../../../../../model/paddle/picodet_lcnet_x1_0_fgd_layout_cdla_infer/model.pdmodel";
            string tab_model = "./../../../../../model/ir/ch_ppstructure_mobile_v2.0_SLANet_infer/model.xml";
            string det_model = "./../../../../../model/paddle/ch_PP-OCRv4_det_infer/inference.pdmodel";
            string cls_model = "./../../../../../model/paddle/ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel";
            string rec_model = "./../../../../../model/paddle/ch_PP-OCRv4_rec_infer/inference.pdmodel";
            StructurePredictor predictor = new StructurePredictor(lay_model,tab_model, det_model, rec_model, cls_model);

            List< StructurePredictResult> structure_results = new List<StructurePredictResult>();
            structure_results = predictor.layout(image, structure_results);
            for (int i = 0; i < structure_results.Count; i++) 
            PaddleOcrUtility.visualize_bboxes(image, structure_results[i]);
        }
        

        static void test_lable()
        {
            Console.WriteLine("Hello, World!");
            Mat image = Cv2.ImRead("./../../../../../image/demo_6.jpg");
            //Cv2.ImShow("aa", image);
            //Cv2.WaitKey(0);
            string str_model = "./../../../../../model/ir/ch_ppstructure_mobile_v2.0_SLANet_infer/model.xml";
            string det_model = "./../../../../../model/paddle/ch_PP-OCRv4_det_infer/inference.pdmodel";
            string cls_model = "./../../../../../model/paddle/ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel";
            string rec_model = "./../../../../../model/paddle/ch_PP-OCRv4_rec_infer/inference.pdmodel";
            StructurePredictor predictor = new StructurePredictor(null, str_model, det_model, cls_model, rec_model);

            StructurePredictResult structure_result = new StructurePredictResult();
            structure_result = predictor.table(image, structure_result);
            //PaddleOcrUtility.VisualizeBboxes(image, structure_result, "./examplejpg");
            Console.WriteLine(structure_result.html);
            // 指定要保存的文件路径及名称
            string filePath = "example.html";

            // 将HTML内容写入到文件中
            File.WriteAllText(filePath, structure_result.html);
            Workbook workbook = new Workbook();
            workbook.LoadFromHtml(filePath);

            //保存为Excel
            workbook.SaveToFile("example.xlsx", FileFormat.Version2013);
        }
    }
}
