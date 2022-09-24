using System;
using OpenCvSharp;

namespace OpenVinoSharpPaddleOCR
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // 文字识别测试
            //PaddleOCR.paddleocr_openvino();

            // 时间测试
            PaddleOCR paddleOCR = new PaddleOCR();
            TimeSpan[] time = new TimeSpan[8];
            for (int i = 0; i < 100; i++)
            {
                TimeSpan[] temp = paddleOCR.paddleocr_openvino_time();
                for (int j = 0; j < 8; j++)
                {
                    time[j] = time[j] + (temp[j]);

                }

            }
            for (int i = 0; i < 8; i++)
            {
                Console.WriteLine((time[i].Ticks / 10000) / 100);
            }

        }



    }
}