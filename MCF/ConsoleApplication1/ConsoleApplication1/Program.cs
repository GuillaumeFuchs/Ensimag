using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ConsoleApplication1
{
    class Program
    {

        static void Main(string[] args)
        {

            int i = 5;

            B b = new B();

            b.Index = i;

            A a = new A();

            a.AddTen(b, i);

            Console.Write(b.Index + " - " + i);
            Console.ReadKey();

        }

        public class A
        {

            public void AddTen(B b, int i)
            {

                b.Index += 10;

                i += 10;

            }

        }

        public class B
        {

            public int Index { get; set; }

        }

    }
}
