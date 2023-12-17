using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using OpenVinoSharp.Extensions.Utility;
namespace PaddleOCR
{

    public class OCRPredictResult
    {
        public List<List<int>> box = new List<List<int>>();
        public string text = "";
        public float score = -1.0f;
        public float cls_score = -1.0f;
        public int cls_label = -1;
    }
    public class StructurePredictResult
    {
        List<float> box = new List<float>();
        List<Rect> cell_box = new List<Rect>();
        string type = "";
        List<OCRPredictResult> text_res = new List<OCRPredictResult>();
        string html = "";
        float html_score = -1.0f;
        float confidence = -1.0f;
    }

    public enum EnumDataType
    {
        Normal_Standard_Deviation = 0,
        Normal_Normalization = 1,
        Normal_Non = 2,
        Affine_Standard_Deviation = 3,
        Affine_Normalization = 4,
        Affine_Non = 5,
        Normal_Standard_custom_Deviation = 6,

    }

    public class PaddleOcrUtility 
    {
        public static List<string> read_dict(string path) {
            List<string> list = new List<string>();
            StreamReader str = new StreamReader(path);
            while (true)
            {
                string line = str.ReadLine();
                if (line == null)
                {
                    break;
                }
                list.Add(line);
            }
            return list;
        }


        public static void print_result(List<OCRPredictResult> ocr_result)
        {
            for (int i = 0; i < ocr_result.Count; i++)
            {
                string mes = "";
                mes += String.Format("{0} \t", i);
                // det
                List<List<int>> boxes = ocr_result[i].box;
                if (boxes.Count > 0)
                {
                    mes += "det boxes: [";
                    for (int n = 0; n < boxes.Count; n++)
                    {
                        mes += String.Format("[{0} , {1}]", boxes[n][0], boxes[n][1]);
                        if (n != boxes.Count - 1)
                        {
                            mes += ',';
                        }
                    }
                    mes += "] \t";
                }
                // rec
                if (ocr_result[i].score != -1.0)
                {
                    mes += String.Format("rec text:  {0}\t  rec score: {1} \t", ocr_result[i].text,ocr_result[i].score);
                }

                // cls
                if (ocr_result[i].cls_label != -1)
                {
                    mes += String.Format("cls label:  {0}\t  cls score: {1} \t", ocr_result[i].cls_label, ocr_result[i].cls_score);
                }
                Console.WriteLine(mes);
            }
        }

        public static void visualize_bboxes(Mat srcimg, List<OCRPredictResult> ocr_result, string save_path)
        {
            Mat img_vis = srcimg.Clone();;
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
                Cv2.Polylines(img_vis, ppt, true, new Scalar(0, 255, 0), 2, LineTypes.Link8, 0);
                
            }

            Cv2.ImWrite(save_path, img_vis);
            Console.WriteLine("The detection visualized image saved in {0}.", save_path);
            Cv2.ImShow("result", img_vis);
            Cv2.WaitKey(0);
        }

        public static Mat get_rotate_crop_image(Mat srcimage, List<List<int>> box)
        {
            Mat image = srcimage.Clone();
            List<List<int>> points = Utility.Clone<List<List<int>>>(box);

            List<int> x_collect = new List<int> { box[0][0], box[1][0], box[2][0], box[3][0] };
            List<int> y_collect = new List<int> { box[0][1], box[1][1], box[2][1], box[3][1] };
            int left = x_collect.Min();
            int right = x_collect.Max();
            int top = y_collect.Min();
            int bottom = y_collect.Max();

            Mat img_crop = new Mat(image, new Rect(left, top, right - left, bottom - top));


            // 倒影变换
            for (int i = 0; i < points.Count; i++)
            {
                points[i][0] -= left;
                points[i][1] -= top;
            }

            int img_crop_height = (int)Math.Sqrt(Math.Pow(points[0][0] - points[1][0], 2) +
                                          Math.Pow(points[0][1] - points[1][1], 2));
            int img_crop_width = (int)Math.Sqrt(Math.Pow(points[0][0] - points[2][0], 2) +
                                           Math.Pow(points[0][1] - points[2][1], 2));

            Point2f[] pts_std = new Point2f[4];
            // 变换后坐标点
            pts_std[0] = new Point2f(0.0f, 0.0f);
            pts_std[1] = new Point2f(img_crop_width, 0.0f);
            pts_std[2] = new Point2f(img_crop_width, img_crop_height);
            pts_std[3] = new Point2f(0.0f, img_crop_height);
            // 变换前坐标点
            Point2f[] pointsf = new Point2f[4];
            pointsf[0] = new Point2f(points[0][0], points[0][1]);
            pointsf[3] = new Point2f(points[1][0], points[1][1]);
            pointsf[1] = new Point2f(points[2][0], points[2][1]);
            pointsf[2] = new Point2f(points[3][0], points[3][1]);

            // 获取变化矩阵
            Mat M = Cv2.GetPerspectiveTransform(pointsf, pts_std);

            Mat dst_img = new Mat();
            Cv2.WarpPerspective(img_crop, dst_img, M, new Size(img_crop_width, img_crop_height),
                InterpolationFlags.Linear, BorderTypes.Replicate);

            //Cv2.ImShow("resize_img", dst_img);
            //Cv2.WaitKey(0);
            if ((float)dst_img.Rows >= (float)dst_img.Cols * 1.5)
            {
                Mat srcCopy = new Mat(dst_img.Rows, dst_img.Cols, dst_img.Depth());
                Cv2.Transpose(dst_img, srcCopy);
                Cv2.Flip(srcCopy, srcCopy, 0);
                return srcCopy;
            }
            else
            {
                return dst_img;
            }
        }

        public static List<int> argsort(List<float> array)
        {
            int array_len = array.Count;
            
            //生成值和索引的列表
            List<float[]> new_array = new List<float[]> { };
            for (int i = 0; i < array_len; i++)
            {
                new_array.Add(new float[] { array[i], i });
            }
            //对列表按照值小到大进行排序
            new_array.Sort((a, b) => a[0].CompareTo(b[0]));
            //获取排序后的原索引
            List<int> array_index = new List<int>();
            foreach (float[] item in new_array)
            {
                array_index.Add((int)item[1]);
            }
            return array_index;
        }
        public static int argmax(float[] data, out float max)
        {
             max = data[0];
            int index = 0;
            for (int i = 0; i < data.Length; i++)
            {
                if (max < data[i])
                {
                    
                    index = i;
                    max = data[i];
                }
            }
            return index;
        }

    }

  

}
