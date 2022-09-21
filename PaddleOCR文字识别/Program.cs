using System;
using OpenCvSharp;

namespace OpenVinoSharpPaddleOCR
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // 测试图片路径
            string image_path = @"E:\Git_space\基于Csharp和OpenVINO部署PaddleOCR模型\image\demo_1.jpg";
            Mat image = Cv2.ImRead(image_path);
            Cv2.ImShow("image", image);


            //------------------------------一、文字区域识别-----------------------//
            Console.WriteLine("//------------------------------一、文字区域识别-----------------------//");

            //*******************1.加载模型相关信息****************//
            // 模型相关参数
            // 模型路径
            string model_file_path_det = @"D:\model\det_server_onnx\model.onnx";
            // 设备名
            string device_name = "CPU";

            // 模型输入节点
            string input_node_name_det = "x";
            // 模型输出节点
            string output_node_name_det = "save_infer_model/scale_0.tmp_1";

            //*******************2.初始化推理核心****************//
            Core pridector_det = new Core(model_file_path_det, device_name);

            //*******************3.配置模型推理输入数据****************//

            // 设置模型节点形状
            ulong[] input_size_det = new ulong[] { 1, 3, 640, 640 };
            pridector_det.set_input_sharp(input_node_name_det, input_size_det);
            // 设置输入数据
            byte[] image_data_det = new byte[2048 * 2048 * 3];
            ulong image_size_det = new ulong();
            image_data_det = image.ImEncode(".bmp");
            image_size_det = Convert.ToUInt64(image_data_det.Length);
            // 将图片数据加载到模型
            pridector_det.load_input_data(input_node_name_det, image_data_det, image_size_det, 0);

            //*******************4.模型推理****************//
            // 模型推理
            pridector_det.infer();
            //*******************5.读取模型输出数据****************//
            int result_det_length = 640 * 640;
            float[] result_det = pridector_det.read_infer_result<float>(output_node_name_det, result_det_length);

            //*******************6.处理模型推理数据****************//
            // 将模型输出转为byte格式
            byte[] result_det_byte = new byte[result_det_length];
            for (int i = 0; i < result_det_length; i++) {
                result_det_byte[i] = (byte)(result_det[i] * 255);
            }
            // 重构结果图像
            Mat image_det = new Mat(640, 640, MatType.CV_8UC3, result_det_byte);
        }


        public static Rect[] find_rect(Mat source_image) {
            Mat image = source_image.Clone();
            //中值滤波或腐蚀去除噪点
            Cv2.MedianBlur(image, image, 3);
            Mat element = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(1, 1), new Point(-1, -1));
            Cv2.Erode(image, image, element, new Point(-1, -1), 1, BorderTypes.Default, new Scalar());
            //Cv2.ImShow("erode", diff);
            Mat element2 = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(2, 2), new Point(-1, -1));
            Cv2.Dilate(image, image, element2, new Point(-1, -1), 1, BorderTypes.Default, new Scalar());
            //Cv2.ImShow("dilate", diff);

            Point[][] contours;
            HierarchyIndex[] hierarchy; //轮廓拓扑结构变量
            Cv2.FindContours(image, out contours, out hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxNone);

            Rect[] rects = new Rect[contours.Length];
            for (int i = 0; i < contours.Length; i++) { 
                Rect rect = Cv2.BoundingRect(contours[i]);
            }
            return rects;
        }

        public static Rect enlarge_rect(Rect rect) {
            Rect rect_temp = new Rect();
            Point point = new Point(rect.X - rect.Width / 2, rect.Y - rect.Height / 2);
            int width = 0;
            int height = 0;
            // 判断矩形区域横纵向
            if (rect.Width > rect.Height)
            {
                if (rect.Width < 80)
                {
                    width = (int)((double)rect.Width * 1.5);
                }
                else
                {
                    width = (int)((double)rect.Width * 1.1);
                }
                height = (int)((double)rect.Height * 2.5);
            }
            else {
                if (rect.Height < 80)
                {
                    height = (int)((double)rect.Height * 1.5);
                }
                else
                {
                    height = (int)((double)rect.Height * 1.1);
                }
                width = (int)((double)rect.Width * 2.5);

            }
            // 判断矩形框是否超边界



            return rect_temp;
        }
    }
}