using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace PaddleOCR
{
    public class StructurePredictor : OCRPredictor
    {
        StruTabRec table_model;
        StruLayRec layout_model;
        bool flag_table_model = false;
        bool flag_layout_model = false;
        public StructurePredictor(string layout_model_path = null, string table_model_path = null, string det_model = null, string rec_model = null, string cls_model = null)
            : base(det_model, cls_model, rec_model)
        {

            if (layout_model_path != null)
            {
                layout_model = new StruLayRec(layout_model_path);
                flag_layout_model = true;
            }
            if (table_model_path != null)
            {
                table_model = new StruTabRec(table_model_path);
                flag_table_model = true;
            }
        }
        public List<StructurePredictResult> structure(Mat srcimg, bool layout, bool table, bool ocr)
        {
            Mat img = new Mat();
            srcimg.CopyTo(img);

            List<StructurePredictResult> structure_results = new List<StructurePredictResult> ();

            if (layout)
            {
                if (!flag_layout_model) 
                {
                    throw new Exception("The StruLayRec is not init!");
                }
                structure_results = this.layout(img, structure_results);
            }
            else
            {
                StructurePredictResult res =new StructurePredictResult();
                res.type = "table";
                res.box = new List<float>() { 0.0f, 0.0f, 0.0f, 0.0f };
                res.box[2] = img.Cols;
                res.box[3] = img.Rows;
                structure_results.Add(res);
            }
            Mat roi_img =new Mat();
            for (int i = 0; i < structure_results.Count; i++)
            {
                // crop image
                roi_img = PaddleOcrUtility.crop_image(img, structure_results[i].box);
                Cv2.ImShow("aa", roi_img);
                Cv2.WaitKey(0);
                if (structure_results[i].type == "table" && table)
                {
                    if (!flag_table_model)
                    {
                        throw new Exception("The StruTabRec is not init!");
                    }
                    structure_results[i] = this.table(roi_img, structure_results[i]);
                }
                else if (ocr)
                {
                    structure_results[i].text_res = this.ocr(roi_img, true, true, false);
                }
            }

            return structure_results;
        }


        public List<StructurePredictResult> layout(Mat img, List<StructurePredictResult> structure_result)
        {
           return layout_model.predict(img, structure_result);
        }


        public StructurePredictResult table(Mat img, StructurePredictResult structure_result)
        {
            // predict structure
            List<List<string>> structure_html_tags = new List<List<string>>();
            List<float> structure_scores = new List<float>(new float[] { 0 });
            List<List<List<int>>> structure_boxes = new List<List<List<int>>>();
            List<Mat> img_list = new List<Mat>();
            img_list.Add(img);

            this.table_model.predict(img_list, structure_html_tags, structure_scores, structure_boxes);


            List<OCRPredictResult> ocr_result = new List<OCRPredictResult>();
            string html = "";
            int expand_pixel = 3;

            for (int i = 0; i < img_list.Count; i++)
            {
                // det
                ocr_result = this.det(img_list[i], ocr_result);
                //

                // crop image
                List<Mat> rec_img_list = new List<Mat>();
                List<int> ocr_box = new List<int>();
                for (int j = 0; j < ocr_result.Count; j++)
                {
                    ocr_box = PaddleOcrUtility.xyxyxyxy2xyxy(ocr_result[j].box);
                    ocr_box[0] = Math.Max(0, ocr_box[0] - expand_pixel);
                    ocr_box[1] = Math.Max(0, ocr_box[1] - expand_pixel);
                    ocr_box[2] = Math.Min(img_list[i].Cols, ocr_box[2] + expand_pixel);
                    ocr_box[3] = Math.Min(img_list[i].Rows, ocr_box[3] + expand_pixel);

                    Mat crop_img = PaddleOcrUtility.crop_image(img_list[i], ocr_box);
                    rec_img_list.Add(crop_img);
                    //Cv2.ImShow("aa", rec_img_list[j]);
                    //Cv2.WaitKey(0);
                }

                // rec
                this.rec(rec_img_list, ocr_result);
                // rebuild table
                html = this.rebuild_table(structure_html_tags[i], structure_boxes[i],
                                           ocr_result);
                structure_result.html = html;
                structure_result.cell_box = structure_boxes[i];
                structure_result.html_score = structure_scores[i];
            }
            return structure_result;
        }
        float dis(List<int> box1, List<int> box2)
        {
            int x1_1 = box1[0];
            int y1_1 = box1[1];
            int x2_1 = box1[2];
            int y2_1 = box1[3];

            int x1_2 = box2[0];
            int y1_2 = box2[1];
            int x2_2 = box2[2];
            int y2_2 = box2[3];

            float dis =
                Math.Abs(x1_2 - x1_1) + Math.Abs(y1_2 - y1_1) + Math.Abs(x2_2 - x2_1) + Math.Abs(y2_2 - y2_1);
            float dis_2 = Math.Abs(x1_2 - x1_1) + Math.Abs(y1_2 - y1_1);
            float dis_3 = Math.Abs(x2_2 - x2_1) + Math.Abs(y2_2 - y2_1);
            return dis + Math.Min(dis_2, dis_3);
        }

        string rebuild_table(List<string> structure_html_tags, List<List<int>> structure_boxes, List<OCRPredictResult> ocr_result)
        {
            // match text in same cell
            List<List<string>> matched = new List<List<string>>();
            for (int i = 0; i < structure_boxes.Count; ++i)
            {
                matched.Add(new List<string>());
            }

            List<int> ocr_box = new List<int>();
            List<int> structure_box = new List<int>();
            for (int i = 0; i < ocr_result.Count; i++)
            {
                ocr_box = PaddleOcrUtility.xyxyxyxy2xyxy(ocr_result[i].box);
                ocr_box[0] -= 1;
                ocr_box[1] -= 1;
                ocr_box[2] += 1;
                ocr_box[3] += 1;
                List<List<float>> dis_list = new List<List<float>>();
                //std::vector<float>(3, 100000.0));
                for (int c = 0; c < structure_boxes.Count; ++c)
                {
                    dis_list.Add(new List<float> { 100000.0f, 100000.0f, 100000.0f });
                }

                for (int j = 0; j < structure_boxes.Count; j++)
                {
                    if (structure_boxes[j].Count == 8)
                    {
                        structure_box = PaddleOcrUtility.xyxyxyxy2xyxy(structure_boxes[j]);
                    }
                    else
                    {
                        structure_box = structure_boxes[j];
                    }
                    dis_list[j][0] = this.dis(ocr_box, structure_box);
                    dis_list[j][1] = 1 - PaddleOcrUtility.iou(ocr_box, structure_box);
                    dis_list[j][2] = j;
                }
                // find min dis idx

                dis_list = dis_list.OrderBy(x => x[0]).ThenBy(x => x[1]).ToList();
                matched[(int)dis_list[0][2]].Add(ocr_result[i].text);
            }

            // get pred html
            string html_str = "";
            int td_tag_idx = 0;
            for (int i = 0; i < structure_html_tags.Count; i++)
            {
                if (structure_html_tags[i].Contains("</td>"))
                {
                    if (structure_html_tags[i].Contains("<td></td>"))
                    {
                        html_str += "<td>";
                    }
                    if (matched[td_tag_idx].Count > 0)
                    {
                        bool b_with = false;
                        if (matched[td_tag_idx][0].Contains("<b>") &&
                            matched[td_tag_idx].Count > 1)
                        {
                            b_with = true;
                            html_str += "<b>";
                        }
                        for (int j = 0; j < matched[td_tag_idx].Count; j++)
                        {
                            string content = matched[td_tag_idx][j];
                            if (matched[td_tag_idx].Count > 1)
                            {
                                // remove blank, <b> and </b>
                                if (content.Length > 0 && content[0] == ' ')
                                {
                                    content = content.Substring(0);
                                }
                                if (content.Length > 2 && content.Substring(0, 3) == "<b>")
                                {
                                    content = content.Substring(3);
                                }
                                if (content.Length > 4 &&
                                    content.Substring(content.Length - 4) == "</b>")
                                {
                                    content = content.Substring(0, content.Length - 4);
                                }
                                if (content.Length == 0)
                                {
                                    continue;
                                }
                                // add blank
                                if (j != matched[td_tag_idx].Count - 1 &&
                                    content[content.Length - 1] != ' ')
                                {
                                    content += ' ';
                                }
                            }
                            html_str += content;
                        }
                        if (b_with)
                        {
                            html_str += "</b>";
                        }
                    }
                    if (structure_html_tags[i].Contains("<td></td>"))
                    {
                        html_str += "</td>";
                    }
                    else
                    {
                        html_str += structure_html_tags[i];
                    }
                    td_tag_idx += 1;
                }
                else
                {
                    html_str += structure_html_tags[i];
                }
            }
            return html_str;
        }
    }
}
