using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Wrapper;

namespace tp4Csharp
{
    class Program
    {
        static void Main(string[] args)
        {
            WrapperClass wc = new WrapperClass();
            double px, ic;
            int M = 50000;
            double T = 2;
            double S0 = 100;
            double K = 110;
            double L = 80;
            double sigma = 0.2;
            double r = 0.05;
            int J = 24;

            wc.getPriceOption(M, T, S0, K, L, sigma, r, J);
            px = wc.getPrice();
            ic = wc.getIC();

            System.Console.WriteLine("prix: %f\nic: %f\n", px, ic); 
            System.Console.ReadKey();
        }
    }
}
