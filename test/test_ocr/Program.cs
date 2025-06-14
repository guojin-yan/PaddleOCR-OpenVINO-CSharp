using OpenCvSharp;
using OpenVinoSharp.Extensions.model.PaddleOCR;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection.Emit;

namespace test_ocr
{
    internal class Program
    {
        static void Main(string[] args)
        {
            test_ocr();
        }

        static void test_ocr()
        {
            string image_path = @"E:\Data\ocr\11.jpg";
            Mat image = Cv2.ImRead(image_path);

            //string det_model = @"E:\Model\ppocrv4\det\det.onnx";
            //string cls_model = @"E:\Model\ppocrv4\cls\cls.onnx";
            //string rec_model = @"E:\Model\ppocrv4\rec\rec.onnx";
            string det_model = @"E:\Model\ppocrv5\det\det.onnx";
            string cls_model = @"E:\Model\ppocrv5\cls\cls.onnx";
            string rec_model = @"E:\Model\ocr\PP-OCRv5_mobile_rec_onnx.onnx";


            RuntimeOption.RecOption.label_path = @"E:\Model\ppocrv5\ppocrv5_dict.txt";
            //RuntimeOption.RecOption.label_path = @"E:\Model\ppocrv4\ppocr_keys_v1.txt";
            //RuntimeOption.ClsOption.batch_num = 10;
            //RuntimeOption.RecOption.batch_num = 10;

            //RuntimeOption.RecOption.use_gpu = true;
            //RuntimeOption.RecOption.device = "GPU";
            //RuntimeOption.ClsOption.use_gpu = true;
            //RuntimeOption.ClsOption.device = "GPU";

            OCRPredictor ocr = new OCRPredictor(det_model, cls_model, rec_model);
            List<OCRPredictResult> ocr_result = ocr.ocr(image, true, true, true);
            Stopwatch sw = new Stopwatch();
            sw.Start();
            for (int i = 0; i < 10; ++i)
            {
                ocr_result = ocr.ocr(image, true, true, true);
            }

            sw.Stop();
            PaddleOcrUtility.print_result(ocr_result);
            Mat result =  PaddleOcrUtility.visualize_bboxes(image, ocr_result);
            Console.WriteLine("总推理时间： " + sw.ElapsedMilliseconds/10 + " ms");
            Cv2.ImShow("result", result);
            string result_path = Path.Combine(Path.GetDirectoryName(image_path), Path.GetFileNameWithoutExtension(image_path) + "_result.jpg");
            Cv2.ImWrite(result_path, result);
            Cv2.WaitKey(0);
        }

        static void test_det()
        {
            Console.WriteLine("Hello, World!");
            OcrDet ocrDet = new OcrDet("./../../../../../model/ir/ch_PP-OCRv4_det_infer/inference.xml");

            Mat image = Cv2.ImRead("./../../../../../image/demo_1.jpg");
            List<OCRPredictResult> ocr_result = new List<OCRPredictResult>();
            // 文字区域识别
            List<List<List<int>>> boxes = ocrDet.predict(image);

            for (int i = 0; i < boxes.Count; i++)
            {
                OCRPredictResult res = new OCRPredictResult();
                res.box = boxes[i];
                ocr_result.Add(res);
            }
            ocr_result = ocr_result.OrderBy(t => t.box[0][1]).ThenBy(t => t.box[0][0]).ToList();

            for (int n = 0; n < ocr_result.Count; n++)
            {
                Point[] rook_points = new Point[4];
                rook_points[0] = new Point((int)(ocr_result[n].box[0][0]), (int)(ocr_result[n].box[0][1]));
                rook_points[1] = new Point((int)(ocr_result[n].box[2][0]), (int)(ocr_result[n].box[2][1]));
                rook_points[2] = new Point((int)(ocr_result[n].box[3][0]), (int)(ocr_result[n].box[3][1]));
                rook_points[3] = new Point((int)(ocr_result[n].box[1][0]), (int)(ocr_result[n].box[1][1]));
                for (int m = 0; m < ocr_result[n].box.Count; m++)
                {

                }

                Point[][] ppt = { rook_points };
                Cv2.Polylines(image, ppt, true, new Scalar(0, 255, 0), 2, LineTypes.Link8, 0);

            }

            Cv2.ImShow("image", image);
            Cv2.WaitKey(0);
        }

        static void test_cls()
        {
            Console.WriteLine("Hello, World!");
            OcrCls ocrCls = new OcrCls("./../../../../../model/ir/ch_ppocr_mobile_v2.0_cls_infer/inference.xml");

            List<Mat> imgs = new List<Mat>();
            imgs.Add(Cv2.ImRead("./../../../../../image/demo_9.png"));
            imgs.Add(Cv2.ImRead("./../../../../../image/demo_10.jpg"));
            imgs.Add(Cv2.ImRead("./../../../../../image/demo_11.jpg"));

            List<int> lables = new List<int>();
            List<float> scores = new List<float>();
            ocrCls.predict(imgs, lables, scores);
            ocrCls.predict(imgs, lables, scores);
            ocrCls.predict(imgs, lables, scores);
            Console.WriteLine("Hello, World!");

        }

        static void test_rec()
        {
            Console.WriteLine("Hello, World!");
            OcrRec ocrRec = new OcrRec("./../../../../../model/ir/ch_PP-OCRv4_rec_infer/inference.xml");

            List<Mat> imgs = new List<Mat>();
            imgs.Add(Cv2.ImRead("./../../../../../image/demo_9.png"));
            imgs.Add(Cv2.ImRead("./../../../../../image/demo_12.jpg"));
            imgs.Add(Cv2.ImRead("./../../../../../image/demo_14.jpg"));

            List<string> rec_texts = new List<string>(new string[imgs.Count]);
            List<float> rec_text_scores = new List<float>(new float[imgs.Count]);
            ocrRec.predict(imgs, rec_texts, rec_text_scores);
            Console.WriteLine("Hello, World!");
        }
    }
}

