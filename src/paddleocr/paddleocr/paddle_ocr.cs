using OpenCvSharp;
using System.Collections.Generic;

namespace PaddleOCR
{
    public class OCRPredictor
    {
        protected OcrDet ocrDet;
        protected OcrCls ocrCls;
        protected OcrRec ocrRec;
        protected bool flag_det = false;
        protected bool flag_rec = false;
        protected bool flag_cls = false;

        public OCRPredictor(OcrConfig config) 
        {
            if (config.det_model_path != null)
            {
                flag_det = true;
                ocrDet = new OcrDet(config);
            }

            if (config.cls_model_path != null)
            {
                flag_cls = true;
                ocrCls = new OcrCls(config);
            }
            if (config.rec_model_path != null)
            {
                flag_rec = true;
                ocrRec = new OcrRec(config);
            }
        }

        public OCRPredictor(string det_model = null, string cls_model = null, string rec_model = null)
        {
            if (det_model != null)
            {
                flag_det = true;
                ocrDet = new OcrDet(det_model);
            }

            if (cls_model != null)
            {
                flag_cls = true;
                ocrCls = new OcrCls(cls_model);
            }
            if (rec_model != null)
            {
                flag_rec = true;
                ocrRec = new OcrRec(rec_model);
            }
        }

        public List<OCRPredictResult> det(Mat image)
        {
            // 文本区域识别
            List<OCRPredictResult> ocr_results = new List<OCRPredictResult>();
            det(image, ocr_results);
            return ocr_results;
        }

        public List<OCRPredictResult> det(Mat image, List<OCRPredictResult> ocr_results)
        {
            // 文字区域识别
            List<List<List<int>>> boxes = ocrDet.predict(image);

            for (int i = 0; i < boxes.Count; i++)
            {
                OCRPredictResult res = new OCRPredictResult();
                res.box = boxes[i];
                ocr_results.Add(res);
            }
            return PaddleOcrUtility.sorted_boxes(ocr_results);
        }


        public List<OCRPredictResult> cls(List<Mat> img_list, List<OCRPredictResult> ocr_results)
        {
            List<int> lables = new List<int>();
            List<float> scores = new List<float>();
            ocrCls.predict(img_list, lables, scores);
            // output cls results
            for (int i = 0; i < lables.Count; i++)
            {
                ocr_results[i].cls_label = lables[i];
                ocr_results[i].cls_score = scores[i];
            }
            return ocr_results;
        }


        public List<OCRPredictResult> rec(List<Mat> img_list, List<OCRPredictResult> ocr_results)
        {
            List<string> rec_texts = new List<string>(new string[img_list.Count]);
            List<float> rec_text_scores = new List<float>(new float[img_list.Count]);
            ocrRec.predict(img_list, rec_texts, rec_text_scores);
            for (int i = 0; i < rec_texts.Count; i++)
            {
                ocr_results[i].text = rec_texts[i];
                ocr_results[i].score = rec_text_scores[i];
            }
            return ocr_results;
        }

        public List<OCRPredictResult> ocr(Mat img, bool det, bool rec, bool cls)
        {
            List<OCRPredictResult> ocr_result = new List<OCRPredictResult>();
            // det
            if (!flag_det)
            {
                throw new Exception("The ocrDet is not init!");
            }
            ocr_result = this.det(img, ocr_result);


            // crop image
            List<Mat> img_list = new List<Mat>();
            for (int j = 0; j < ocr_result.Count; j++)
            {
                Mat crop_img = new Mat();
                crop_img = PaddleOcrUtility.get_rotate_crop_image(img, ocr_result[j].box);
                img_list.Add(crop_img);
            }
            // cls
            if (cls)
            {
                if (!flag_cls)
                {
                    throw new Exception("The ocrCls is not init!");
                }
                ocr_result = this.cls(img_list, ocr_result);
                for (int i = 0; i < img_list.Count; i++)
                {
                    if (ocr_result[i].cls_label % 2 == 1 && ocr_result[i].cls_score > ocrCls.m_cls_thresh)
                    {
                        Cv2.Rotate(img_list[i], img_list[i], RotateFlags.Rotate180);
                    }
                }
            }
            // rec
            if (rec)
            {
                if (!flag_rec)
                {
                    throw new Exception("The ocrRec is not init!");
                }
                ocr_result = this.rec(img_list, ocr_result);
            }
            return ocr_result;
        }

        public List<List<OCRPredictResult>> ocr(List<Mat> img_list, bool det, bool rec, bool cls)
        {
            List<List<OCRPredictResult>> results = new List<List<OCRPredictResult>>();
            foreach (Mat img in img_list) 
            {
                results.Add(ocr(img, det, rec, cls));
            }
            return results;
        }

    }
}
