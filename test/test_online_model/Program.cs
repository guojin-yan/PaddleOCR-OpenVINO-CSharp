using System.IO;
using System;
using PaddleOCR;
using iTextSharp.xmp.impl.xpath;
using System.IO.Compression;
using iTextSharp.xmp.impl;
using SharpCompress.Readers.Tar;
using OpenVinoSharp.Extensions.Utility;


namespace test_online_model
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");
            string url = "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.rar";
            string path = "./../../../../model/";
            string file_path = Path.Combine(path, Path.GetFileName(url));
            //_ = Utility.download_file_async(url, file_path).Result;
            //ZipFile.ExtractToDirectory(file_path, path);

            Utility.unzip(file_path, path);
            Console.WriteLine("Hello, World!");

            //ConsoleUtility.WriteProgressBar(0);
            //for (var i = 0; i <= 100; ++i)
            //{
            //    ConsoleUtility.WriteProgressBar(i, true);
            //    Thread.Sleep(50);
            //}
            //Console.WriteLine();
            //ConsoleUtility.WriteProgress(0);
            //for (var i = 0; i <= 100; ++i)
            //{
            //    ConsoleUtility.WriteProgress(i, true);
            //    Thread.Sleep(50);
            //}

        }
    }
}
